//! G6 SERVE-PERF A/B harness: the frozen HEAD `neural-mesh serve` (baseline exe) vs the working-tree one.
//!
//! Everything is end to end against the real binaries (std only, no new crates, loopback 127.0.0.1 only):
//!   equiv  - the same deterministic request script (valid ops, error forms, blank/odd/invalid-UTF-8 lines,
//!            lines around the 1 MiB limit, pipelined bursts, random chunking, byte-at-a-time, EOF mid-line,
//!            half-close, shutdown mid-script, CLI writes interleaved) is sent to both; the full reply byte
//!            streams, exit codes and final state files must be identical. Controls run base-vs-base to show
//!            the harness itself is deterministic.
//!   bench  - decisions/s (decision = route + feedback) serial (1 conn lockstep, stdio) and N parallel
//!            connections, old vs new interleaved ABAB, medians + spread over reps, server CPU/decision, memory.
//!
//! Usage: g6_serve_ab equiv [--base EXE] [--new EXE] [--quick] [--out FILE]
//!        g6_serve_ab bench [--base EXE] [--new EXE] [--secs 1.5] [--reps 5] [--out FILE]
//! Baseline exe: `bash results/g6_build_baseline.sh` (builds HEAD's serve.rs from a git worktree).

#[path = "g6_common/mod.rs"]
mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Barrier, Mutex};
use std::time::{Duration, Instant};

use common::*;
use serde_json::{Value, json};

// =============================================================================================
// equivalence

#[derive(Default)]
struct Run {
    replies: Vec<u8>,
    exit: Option<i32>,
    state: Option<Vec<u8>>,
    note: String,
}

fn wait_exit(child: &mut Child, max: Duration) -> Option<i32> {
    let t0 = Instant::now();
    loop {
        if let Ok(Some(st)) = child.try_wait() {
            return st.code();
        }
        if t0.elapsed() > max {
            let _ = child.kill();
            let _ = child.wait();
            return Some(-9999);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn send_chunks(w: &mut dyn Write, data: &[u8], chunk: Chunking) {
    match chunk {
        Chunking::Whole => {
            let _ = w.write_all(data);
        }
        Chunking::Random(seed, max) => {
            let mut r = Rng(seed);
            let mut i = 0;
            while i < data.len() {
                let n = (1 + r.below(max)).min(data.len() - i);
                if w.write_all(&data[i..i + n]).is_err() {
                    return;
                }
                let _ = w.flush();
                i += n;
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Chunking {
    Whole,
    Random(u64, usize),
}

fn serve_args(state: &Path, mode: &[&str]) -> Vec<String> {
    let mut a = vec!["serve".to_string(), "--state".into(), state.to_str().unwrap().into()];
    a.extend(mode.iter().map(|s| s.to_string()));
    a.extend(["--flush-every", "7", "--flush-ms", "600000"].map(String::from));
    a
}

fn run_stdio(exe: &Path, tag: &str, data: Vec<u8>, chunk: Chunking) -> Run {
    let dir = tmp_dir(tag);
    let state = dir.join("state.json");
    let mut cmd = Command::new(exe);
    apply_env(&mut cmd, exe);
    let mut child = cmd.args(serve_args(&state, &["--stdio"])).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let mut sin = child.stdin.take().unwrap();
    let mut sout = child.stdout.take().unwrap();
    let wr = std::thread::spawn(move || {
        send_chunks(&mut sin, &data, chunk);
    });
    let mut replies = Vec::new();
    let _ = sout.read_to_end(&mut replies);
    let _ = wr.join();
    let exit = wait_exit(&mut child, Duration::from_secs(30));
    let st = std::fs::read(&state).ok();
    let _ = std::fs::remove_dir_all(&dir);
    Run { replies, exit, state: st, note: String::new() }
}

#[derive(Clone, Copy, PartialEq)]
enum TcpMode {
    Lockstep,
    Burst,
    Chunked(u64, usize),
    EofNoShutdown,
    HalfCloseMidLine,
}

fn run_tcp(exe: &Path, tag: &str, lines: &[Vec<u8>], mode: TcpMode) -> Run {
    let srv = spawn_tcp(exe, tag, 7, 600_000);
    let mut s = TcpStream::connect(&srv.addr).unwrap();
    s.set_nodelay(true).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let mut note = String::new();
    let mut replies: Vec<u8> = Vec::new();
    let shutdown_line = line("{\"cmd\":\"shutdown\",\"id\":\"bye\"}");
    let with_shutdown = !matches!(mode, TcpMode::EofNoShutdown | TcpMode::HalfCloseMidLine);
    match mode {
        TcpMode::Lockstep => {
            let mut br = BufReader::new(s.try_clone().unwrap());
            let all: Vec<&Vec<u8>> = lines.iter().chain(std::iter::once(&shutdown_line)).collect();
            for l in all {
                if s.write_all(l).is_err() {
                    note += "write failed; ";
                    break;
                }
                if expects_reply(l) {
                    let mut got = Vec::new();
                    match br.read_until(b'\n', &mut got) {
                        Ok(0) => {
                            note += "early EOF; ";
                            break;
                        }
                        Ok(_) => replies.extend_from_slice(&got),
                        Err(e) => {
                            note += &format!("read error {e}; ");
                            break;
                        }
                    }
                }
            }
            // drain until the server closes
            let _ = br.read_to_end(&mut replies);
        }
        _ => {
            let mut data = flatten(lines);
            match mode {
                TcpMode::HalfCloseMidLine => {
                    data.extend_from_slice(b"{\"cmd\":\"stats\",\"id\":\"no-newline-at-eof\"}");
                }
                _ if with_shutdown => data.extend_from_slice(&shutdown_line),
                _ => {}
            }
            let chunk = match mode {
                TcpMode::Chunked(seed, max) => Chunking::Random(seed, max),
                _ => Chunking::Whole,
            };
            let mut ws = s.try_clone().unwrap();
            let half = matches!(mode, TcpMode::EofNoShutdown | TcpMode::HalfCloseMidLine);
            let wr = std::thread::spawn(move || {
                send_chunks(&mut ws, &data, chunk);
                if half {
                    let _ = ws.shutdown(Shutdown::Write);
                }
            });
            let _ = s.read_to_end(&mut replies);
            let _ = wr.join();
        }
    }
    let mut srv = srv;
    let exit = if with_shutdown {
        wait_exit(&mut srv.child, Duration::from_secs(30))
    } else {
        None
    };
    let st = if with_shutdown { std::fs::read(&srv.state).ok() } else { None };
    srv.kill();
    Run { replies, exit, state: st, note }
}

/// Lockstep with the one-shot CLI changing the state file between requests (exercises the server's
/// reload+replay path). The CLI is always the baseline exe, for both servers.
fn run_tcp_ext(exe: &Path, cli: &Path, tag: &str) -> Run {
    let srv = spawn_tcp(exe, tag, 5, 600_000);
    let mut s = TcpStream::connect(&srv.addr).unwrap();
    s.set_nodelay(true).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let mut br = BufReader::new(s.try_clone().unwrap());
    let mut r = Rng(77);
    let mut replies = Vec::new();
    let mut ask = |s: &mut TcpStream, l: &str, replies: &mut Vec<u8>| {
        s.write_all(&line(l)).unwrap();
        let mut got = Vec::new();
        br.read_until(b'\n', &mut got).unwrap();
        replies.extend_from_slice(&got);
    };
    ask(&mut s, "{\"cmd\":\"route\",\"features\":[\"warm\"],\"candidates\":[\"a0\",\"a1\",\"a2\"]}", &mut replies);
    for i in 0..60 {
        let f = format!("kw{}", r.below(6));
        match i % 6 {
            0 | 3 => ask(&mut s, &format!("{{\"cmd\":\"route\",\"features\":[\"{f}\"],\"id\":{i}}}"), &mut replies),
            1 | 4 => ask(&mut s, &format!("{{\"cmd\":\"feedback\",\"features\":[\"{f}\"],\"agent\":\"a{}\",\"reward\":0.{}}}", r.below(3), 1 + r.below(8)), &mut replies),
            2 => {
                // an out-of-band CLI write
                let out = Command::new(cli)
                    .args(["feedback", "--state", srv.state.to_str().unwrap(), "--features", &f, "--agent", &format!("cli{}", i % 2), "--reward", "0.9"])
                    .output()
                    .unwrap();
                replies.extend_from_slice(&out.stdout);
            }
            _ => ask(&mut s, &format!("{{\"cmd\":\"stats\",\"id\":{i}}}"), &mut replies),
        }
    }
    ask(&mut s, "{\"cmd\":\"shutdown\"}", &mut replies);
    let mut srv = srv;
    let exit = wait_exit(&mut srv.child, Duration::from_secs(30));
    let st = std::fs::read(&srv.state).ok();
    srv.kill();
    Run { replies, exit, state: st, note: String::new() }
}

fn first_diff(a: &[u8], b: &[u8]) -> String {
    let n = a.len().min(b.len());
    let i = (0..n).find(|&i| a[i] != b[i]).unwrap_or(n);
    let ctx = |x: &[u8]| String::from_utf8_lossy(&x[i.saturating_sub(60)..(i + 100).min(x.len())]).replace('\n', "\\n");
    format!("first differing byte {i} (len {} vs {}): base=[{}] new=[{}]", a.len(), b.len(), ctx(a), ctx(b))
}

struct Cmp {
    name: String,
    reply_bytes: usize,
    reply_lines: usize,
    ok_true: usize,
    ok_false: usize,
    ok: bool,
    detail: String,
}

fn compare(name: &str, a: &Run, b: &Run) -> Cmp {
    let mut ok = true;
    let mut d = Vec::new();
    if a.replies != b.replies {
        ok = false;
        d.push(first_diff(&a.replies, &b.replies));
    }
    if a.exit != b.exit {
        ok = false;
        d.push(format!("exit {:?} vs {:?}", a.exit, b.exit));
    }
    if a.state != b.state {
        ok = false;
        d.push(format!("state file differs ({:?} vs {:?} bytes)", a.state.as_ref().map(|v| v.len()), b.state.as_ref().map(|v| v.len())));
    }
    if !a.note.is_empty() || !b.note.is_empty() {
        d.push(format!("notes: base='{}' new='{}'", a.note, b.note));
        if a.note != b.note {
            ok = false;
        }
    }
    let count = |pat: &[u8]| a.replies.windows(pat.len()).filter(|w| *w == pat).count();
    Cmp {
        name: name.into(),
        reply_bytes: a.replies.len(),
        reply_lines: a.replies.iter().filter(|&&c| c == b'\n').count(),
        ok_true: count(b"\"ok\":true"),
        ok_false: count(b"\"ok\":false"),
        ok,
        detail: d.join(" | "),
    }
}

fn equiv(base: &Path, newx: &Path, quick: bool, out: Option<String>) {
    let small = script(0x6606, if quick { 300 } else { 900 }, if quick { 500 } else { 1800 }, false);
    let big = script(0x6607, 200, 300, true);
    let tiny: Vec<Vec<u8>> = small.iter().take(160).cloned().collect();
    let small_bytes = flatten(&small);
    let big_bytes = flatten(&big);
    let n_requests = small.len();
    println!("script: small {} lines / {} bytes; big {} lines / {} bytes; tiny {} lines", small.len(), small_bytes.len(), big.len(), big_bytes.len(), tiny.len());

    let mut results: Vec<Cmp> = Vec::new();
    let mut controls: Vec<Cmp> = Vec::new();
    // each scenario: closure(exe, tag) -> Run
    type Scn<'a> = (&'a str, Box<dyn Fn(&Path, &str) -> Run + 'a>);
    let sd = |v: &[u8]| {
        let mut x = v.to_vec();
        x.extend_from_slice(&line("{\"cmd\":\"shutdown\",\"id\":\"bye\"}"));
        x.extend_from_slice(&line("{\"cmd\":\"stats\",\"id\":\"after-shutdown-must-be-ignored\"}"));
        x
    };
    let scenarios: Vec<Scn> = vec![
        ("stdio: whole script, EOF end", Box::new(|e, t| run_stdio(e, t, small_bytes.clone(), Chunking::Whole))),
        ("stdio: whole script + shutdown mid-stream (+ignored tail)", Box::new(|e, t| run_stdio(e, t, sd(&small_bytes), Chunking::Whole))),
        ("stdio: random 1..64-byte writes", Box::new(|e, t| run_stdio(e, t, small_bytes.clone(), Chunking::Random(5, 64)))),
        ("stdio: EOF in the middle of the last line (no newline)", Box::new(|e, t| {
            let mut d = small_bytes.clone();
            d.extend_from_slice(b"{\"cmd\":\"stats\",\"id\":\"partial-at-eof\"}");
            run_stdio(e, t, d, Chunking::Whole)
        })),
        ("stdio: big/boundary lines (1 MiB limit, 3 MiB, invalid UTF-8)", Box::new(|e, t| run_stdio(e, t, sd(&big_bytes), Chunking::Whole))),
        ("tcp: lockstep (one request, wait reply)", Box::new(|e, t| run_tcp(e, t, &small, TcpMode::Lockstep))),
        ("tcp: whole script in one pipelined burst + shutdown", Box::new(|e, t| run_tcp(e, t, &small, TcpMode::Burst))),
        ("tcp: random 1..300-byte writes (split lines)", Box::new(|e, t| run_tcp(e, t, &small, TcpMode::Chunked(9, 300)))),
        ("tcp: byte at a time (160 lines)", Box::new(|e, t| run_tcp(e, t, &tiny, TcpMode::Chunked(1, 1)))),
        ("tcp: big/boundary lines, pipelined burst", Box::new(|e, t| run_tcp(e, t, &big, TcpMode::Burst))),
        ("tcp: big/boundary lines, lockstep", Box::new(|e, t| run_tcp(e, t, &big, TcpMode::Lockstep))),
        ("tcp: client half-closes after burst (EOF, no shutdown)", Box::new(|e, t| run_tcp(e, t, &small, TcpMode::EofNoShutdown))),
        ("tcp: half-close in the middle of the last line", Box::new(|e, t| run_tcp(e, t, &small, TcpMode::HalfCloseMidLine))),
        ("tcp: CLI writes interleaved (reload + journal replay)", Box::new(|e, t| run_tcp_ext(e, base, t))),
    ];
    let mut all_ok = true;
    for (i, (name, f)) in scenarios.iter().enumerate() {
        let a = f(base, &format!("b{i}"));
        let b = f(newx, &format!("n{i}"));
        let c = f(base, &format!("c{i}"));
        let r = compare(name, &a, &b);
        let ctl = compare(name, &a, &c);
        println!("[{}] {:<64} replies {:>9} B / {:>5} lines ({:>4} ok, {:>4} err)  new==base: {}   control base==base: {}  {}", if r.ok { "OK " } else { "FAIL" }, name, r.reply_bytes, r.reply_lines, r.ok_true, r.ok_false, r.ok, ctl.ok, if !r.ok { r.detail.clone() } else { String::new() });
        if !ctl.ok {
            println!("   control mismatch (harness not deterministic here): {}", ctl.detail);
        }
        all_ok &= r.ok && ctl.ok;
        results.push(r);
        controls.push(ctl);
    }
    // the harness must be able to see a difference: change one byte of the request script
    let mut mutated = small_bytes.clone();
    let at = mutated.windows(3).position(|w| w == b"\"kw").unwrap() + 2;
    mutated[at] = b'x';
    let (ra, rb) = (run_stdio(base, "st-a", small_bytes.clone(), Chunking::Whole), run_stdio(base, "st-b", mutated, Chunking::Whole));
    let sees_diff = !compare("selftest", &ra, &rb).ok;
    println!("harness self-test (one request byte changed, same exe): difference detected = {sees_diff}");
    all_ok &= sees_diff;
    println!("\nEQUIVALENCE: {} ({} scenarios, base vs new; {} controls)", if all_ok { "ALL BYTE-IDENTICAL" } else { "MISMATCH" }, results.len(), controls.len());
    let _ = n_requests;
    if let Some(p) = out {
        let rows: Vec<Value> = results
            .iter()
            .zip(&controls)
            .map(|(r, c)| json!({"scenario": r.name, "reply_bytes": r.reply_bytes, "reply_lines": r.reply_lines, "ok_true_replies": r.ok_true, "ok_false_replies": r.ok_false, "new_equals_base": r.ok, "control_base_equals_base": c.ok, "detail": r.detail}))
            .collect();
        let v = json!({"all_identical": all_ok, "small_script_lines": small.len(), "small_script_bytes": small_bytes.len(), "big_script_bytes": big_bytes.len(), "scenarios": rows});
        std::fs::write(p, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    }
    if !all_ok {
        std::process::exit(1);
    }
}

// =============================================================================================
// benchmark

const TASKS: usize = 240;

fn task_feats(t: usize) -> (usize, usize) {
    (t % 24, (t * 7 + 3) % 24)
}

fn route_line(t: usize) -> String {
    let (a, b) = task_feats(t);
    format!("{{\"cmd\":\"route\",\"features\":[\"kw{a}\",\"kw{b}\"]}}\n")
}
fn feedback_line(t: usize, agent: usize) -> String {
    let (a, b) = task_feats(t);
    let r = if agent == a % 6 { 0.9 } else { 0.1 };
    format!("{{\"cmd\":\"feedback\",\"features\":[\"kw{a}\",\"kw{b}\"],\"agent\":\"a{agent}\",\"reward\":{r}}}\n")
}
fn warm_line() -> String {
    "{\"cmd\":\"route\",\"features\":[\"warm\"],\"candidates\":[\"a0\",\"a1\",\"a2\",\"a3\",\"a4\",\"a5\"]}\n".to_string()
}
fn agent_of(reply: &str) -> usize {
    reply.find("\"agent\":\"a").and_then(|i| reply[i + 10..].chars().next()).and_then(|c| c.to_digit(10)).unwrap_or(0) as usize % 6
}

#[derive(Clone, Debug, Default)]
struct B {
    dps: f64,
    p50_us: f64,
    p99_us: f64,
    cpu_us_per_dec: f64,
    peak_ws_mb: f64,
    peak_pf_mb: f64,
    bg_pct: f64,
    decisions: u64,
    bad_replies: u64,
}

fn bench_tcp(exe: &Path, tag: &str, clients: usize, secs: f64, flush_every: usize, pipelined: bool) -> B {
    let bg = cpu_busy_pct(250);
    let srv = spawn_tcp(exe, tag, flush_every, if flush_every >= 1_000_000 { 600_000 } else { 1000 });
    {
        let mut s = TcpStream::connect(&srv.addr).unwrap();
        s.write_all(warm_line().as_bytes()).unwrap();
        let mut br = BufReader::new(s.try_clone().unwrap());
        let mut l = String::new();
        br.read_line(&mut l).unwrap();
    }
    let stop = AtomicBool::new(false);
    let barrier = Barrier::new(clients + 1);
    let totals: Vec<Mutex<(u64, u64, Vec<f32>)>> = (0..clients).map(|_| Mutex::new((0, 0, vec![]))).collect();
    let addr = srv.addr.clone();
    let mut cpu0 = (0.0, 0.0);
    let mut cpu1 = (0.0, 0.0);
    let mut elapsed = 0.0;
    std::thread::scope(|sc| {
        for tid in 0..clients {
            let (stop, barrier, totals, addr) = (&stop, &barrier, &totals, addr.clone());
            sc.spawn(move || {
                let mut s = TcpStream::connect(&addr).unwrap();
                s.set_nodelay(true).ok();
                let mut br = BufReader::new(s.try_clone().unwrap());
                let mut t = (tid * 37) % TASKS;
                let (mut n, mut bad) = (0u64, 0u64);
                let mut lat: Vec<f32> = Vec::with_capacity(8192);
                let mut resp = String::new();
                let mut wbuf = String::new();
                barrier.wait();
                while !stop.load(Ordering::Relaxed) {
                    let t0 = Instant::now();
                    if pipelined {
                        const BATCH: usize = 32;
                        wbuf.clear();
                        for k in 0..BATCH {
                            wbuf.push_str(&route_line((t + k) % TASKS));
                        }
                        for k in 0..BATCH {
                            let tt = (t + k) % TASKS;
                            wbuf.push_str(&feedback_line(tt, (task_feats(tt).0) % 6));
                        }
                        if s.write_all(wbuf.as_bytes()).is_err() {
                            break;
                        }
                        let mut dead = false;
                        for _ in 0..2 * BATCH {
                            resp.clear();
                            if br.read_line(&mut resp).unwrap_or(0) == 0 {
                                dead = true;
                                break;
                            }
                            if !resp.starts_with("{\"") || !resp.contains("\"ok\":true") {
                                bad += 1;
                            }
                        }
                        if dead {
                            break;
                        }
                        n += BATCH as u64;
                        t = (t + BATCH) % TASKS;
                        lat.push(t0.elapsed().as_nanos() as f32 / 1000.0 / BATCH as f32);
                    } else {
                        if s.write_all(route_line(t).as_bytes()).is_err() {
                            break;
                        }
                        resp.clear();
                        if br.read_line(&mut resp).unwrap_or(0) == 0 {
                            break;
                        }
                        if !resp.contains("\"ok\":true") {
                            bad += 1;
                        }
                        let ag = agent_of(&resp);
                        if s.write_all(feedback_line(t, ag).as_bytes()).is_err() {
                            break;
                        }
                        resp.clear();
                        if br.read_line(&mut resp).unwrap_or(0) == 0 {
                            break;
                        }
                        if !resp.contains("\"ok\":true") {
                            bad += 1;
                        }
                        lat.push(t0.elapsed().as_nanos() as f32 / 1000.0);
                        n += 1;
                        t = (t + 1) % TASKS;
                    }
                }
                *totals[tid].lock().unwrap() = (n, bad, lat);
            });
        }
        barrier.wait();
        cpu0 = child_cpu_ms(&srv.child);
        let t0 = Instant::now();
        std::thread::sleep(Duration::from_secs_f64(secs));
        stop.store(true, Ordering::Relaxed);
        elapsed = t0.elapsed().as_secs_f64();
        cpu1 = child_cpu_ms(&srv.child);
    });
    let (pws, ppf, _) = child_mem(&srv.child);
    srv.kill();
    let (mut dec, mut bad, mut lat) = (0u64, 0u64, Vec::<f32>::new());
    for t in totals {
        let (n, b, l) = t.into_inner().unwrap();
        dec += n;
        bad += b;
        lat.extend(l);
    }
    lat.sort_by(|a, b| a.total_cmp(b));
    let pick = |q: f64| if lat.is_empty() { 0.0 } else { lat[((lat.len() - 1) as f64 * q) as usize] as f64 };
    let cpu_ms = (cpu1.0 + cpu1.1) - (cpu0.0 + cpu0.1);
    B {
        dps: dec as f64 / elapsed,
        p50_us: pick(0.5),
        p99_us: pick(0.99),
        cpu_us_per_dec: if dec > 0 { cpu_ms * 1000.0 / dec as f64 } else { 0.0 },
        peak_ws_mb: pws as f64 / 1048576.0,
        peak_pf_mb: ppf as f64 / 1048576.0,
        bg_pct: bg,
        decisions: dec,
        bad_replies: bad,
    }
}

/// `serve --stdio`: n_dec route+feedback pairs through pipes (no sockets): the request-path CPU ceiling.
fn bench_stdio(exe: &Path, tag: &str, n_dec: usize, flush_every: usize) -> B {
    let bg = cpu_busy_pct(250);
    let dir = tmp_dir(tag);
    let state = dir.join("state.json");
    let mut text = warm_line();
    for i in 0..n_dec {
        let t = i % TASKS;
        text.push_str(&route_line(t));
        text.push_str(&feedback_line(t, task_feats(t).0 % 6));
    }
    let fe = flush_every.to_string();
    let ms = if flush_every >= 1_000_000 { "600000" } else { "1000" };
    let mut cmd = Command::new(exe);
    apply_env(&mut cmd, exe);
    let mut child = cmd
        .args(["serve", "--state", state.to_str().unwrap(), "--stdio", "--flush-every", &fe, "--flush-ms", ms])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut sin = child.stdin.take().unwrap();
    let sout = child.stdout.take().unwrap();
    let t0 = Instant::now();
    let wr = std::thread::spawn(move || {
        let _ = sin.write_all(text.as_bytes());
    });
    let (mut lines, mut bad) = (0usize, 0u64);
    for l in BufReader::new(sout).lines() {
        let Ok(l) = l else { break };
        lines += 1;
        if !l.contains("\"ok\":true") {
            bad += 1;
        }
    }
    let el = t0.elapsed().as_secs_f64();
    let _ = wr.join();
    let _ = child.wait();
    let (u, k) = child_cpu_ms(&child);
    let (pws, ppf, _) = child_mem(&child);
    let _ = std::fs::remove_dir_all(&dir);
    let dec = (lines.saturating_sub(1)) / 2;
    B {
        dps: dec as f64 / el,
        p50_us: el * 1e6 / (2.0 * n_dec as f64),
        p99_us: 0.0,
        cpu_us_per_dec: (u + k) * 1000.0 / dec.max(1) as f64,
        peak_ws_mb: pws as f64 / 1048576.0,
        peak_pf_mb: ppf as f64 / 1048576.0,
        bg_pct: bg,
        decisions: dec as u64,
        bad_replies: bad,
    }
}

struct Cfg {
    name: String,
    clients: usize,
    flush_every: usize,
    pipelined: bool,
    stdio: bool,
}

fn bench(base: &Path, newx: &Path, secs: f64, reps: usize, out: Option<String>, filter: Option<String>) {
    let mut cfgs: Vec<Cfg> = Vec::new();
    for n in [1usize, 2, 4, 8] {
        cfgs.push(Cfg { name: format!("tcp lockstep N={n} (defaults flush 64/1s)"), clients: n, flush_every: 64, pipelined: false, stdio: false });
    }
    for n in [1usize, 8] {
        cfgs.push(Cfg { name: format!("tcp lockstep N={n} (flush off)"), clients: n, flush_every: 1_000_000, pipelined: false, stdio: false });
    }
    for n in [1usize, 4, 8] {
        cfgs.push(Cfg { name: format!("tcp pipelined 32+32 per write N={n} (defaults flush)"), clients: n, flush_every: 64, pipelined: true, stdio: false });
    }
    cfgs.push(Cfg { name: "stdio pipe, 20k decisions (defaults flush)".into(), clients: 1, flush_every: 64, pipelined: false, stdio: true });
    cfgs.push(Cfg { name: "stdio pipe, 20k decisions (flush off)".into(), clients: 1, flush_every: 1_000_000, pipelined: false, stdio: true });

    let exes = [("base", base), ("new", newx)];
    let mut rows: Vec<Value> = Vec::new();
    println!("secs/run {secs}, reps {reps}, ABAB interleaved (order alternates each rep). decision = route + feedback.\n");
    println!("{:<58} | {:>10} {:>10} | {:>6} | {:>9} {:>9} | {:>8} {:>8} | {:>7} {:>7}", "config", "base dec/s", "new dec/s", "x", "base p50", "new p50", "base cpu", "new cpu", "base WS", "new WS");
    for (ci, c) in cfgs.iter().enumerate() {
        if filter.as_ref().is_some_and(|f| !c.name.contains(f.as_str())) {
            continue;
        }
        let mut res: [Vec<B>; 2] = [vec![], vec![]];
        for rep in 0..reps {
            let order: [usize; 2] = if rep % 2 == 0 { [0, 1] } else { [1, 0] };
            for &e in &order {
                let tag = format!("{}{}_{}_{}", exes[e].0, ci, rep, e);
                let b = if c.stdio {
                    bench_stdio(exes[e].1, &tag, 20_000, c.flush_every)
                } else {
                    bench_tcp(exes[e].1, &tag, c.clients, secs, c.flush_every, c.pipelined)
                };
                res[e].push(b);
            }
        }
        let col = |e: usize, f: fn(&B) -> f64| res[e].iter().map(f).collect::<Vec<f64>>();
        let (bd, nd) = (col(0, |b| b.dps), col(1, |b| b.dps));
        // paired ratios (rep k base vs rep k new, run back to back)
        let ratios: Vec<f64> = bd.iter().zip(&nd).map(|(b, n)| n / b).collect();
        let bad: u64 = res.iter().flatten().map(|b| b.bad_replies).sum();
        println!(
            "{:<58} | {:>10.0} {:>10.0} | {:>5.2}x | {:>7.0}us {:>7.0}us | {:>6.0}us {:>6.0}us | {:>5.1}MB {:>5.1}MB{}",
            c.name,
            median(&bd),
            median(&nd),
            median(&nd) / median(&bd),
            median(&col(0, |b| b.p50_us)),
            median(&col(1, |b| b.p50_us)),
            median(&col(0, |b| b.cpu_us_per_dec)),
            median(&col(1, |b| b.cpu_us_per_dec)),
            median(&col(0, |b| b.peak_ws_mb)),
            median(&col(1, |b| b.peak_ws_mb)),
            if bad > 0 { format!("  !! {bad} bad replies") } else { String::new() }
        );
        let side = |e: usize| {
            let d = col(e, |b| b.dps);
            let (lo, hi) = minmax(&d);
            json!({
                "decisions_per_s_median": median(&d), "decisions_per_s_min": lo, "decisions_per_s_max": hi, "decisions_per_s_all": d,
                "p50_us_median": median(&col(e, |b| b.p50_us)), "p99_us_median": median(&col(e, |b| b.p99_us)),
                "server_cpu_us_per_decision_median": median(&col(e, |b| b.cpu_us_per_dec)),
                "peak_working_set_mb_median": median(&col(e, |b| b.peak_ws_mb)), "peak_pagefile_mb_median": median(&col(e, |b| b.peak_pf_mb)),
                "bg_cpu_busy_pct_before_runs": col(e, |b| b.bg_pct), "bad_replies": res[e].iter().map(|b| b.bad_replies).sum::<u64>(),
            })
        };
        let (rl, rh) = minmax(&ratios);
        rows.push(json!({"config": c.name, "clients": c.clients, "base": side(0), "new": side(1),
            "ratio_of_medians": median(&nd) / median(&bd), "paired_ratio_median": median(&ratios), "paired_ratio_min": rl, "paired_ratio_max": rh, "paired_ratios": ratios}));
    }
    if let Some(p) = out {
        let v = json!({"secs_per_run": secs, "reps": reps, "rows": rows});
        std::fs::write(p, serde_json::to_string_pretty(&v).unwrap()).unwrap();
    }
}

// =============================================================================================

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exe = std::env::current_exe().unwrap();
    let rel = exe.parent().unwrap();
    let default_new = rel.join(if cfg!(windows) { "neural-mesh.exe" } else { "neural-mesh" });
    let default_base: PathBuf = rel.parent().unwrap().join("g6").join(if cfg!(windows) { "neural-mesh-base.exe" } else { "neural-mesh-base" });
    let base = arg(&args, "--base").map(PathBuf::from).unwrap_or(default_base);
    let newx = arg(&args, "--new").map(PathBuf::from).unwrap_or(default_new);
    for p in [&base, &newx] {
        if !p.exists() {
            eprintln!("missing exe {} (build baseline with results/g6_build_baseline.sh; new with cargo build --release --bin neural-mesh)", p.display());
            std::process::exit(2);
        }
    }
    println!("base: {}\nnew:  {}", base.display(), newx.display());
    if let Some(kv) = arg(&args, "--new-env") {
        let (k, v) = kv.split_once('=').expect("--new-env K=V");
        EXTRA_ENV.lock().unwrap().push((newx.clone(), k.to_string(), v.to_string()));
        println!("new side runs with env {k}={v}");
    }
    let out = arg(&args, "--out");
    match args.first().map(String::as_str) {
        Some("equiv") => equiv(&base, &newx, args.iter().any(|a| a == "--quick"), out),
        Some("bench") => {
            let secs: f64 = arg(&args, "--secs").and_then(|s| s.parse().ok()).unwrap_or(1.5);
            let reps: usize = arg(&args, "--reps").and_then(|s| s.parse().ok()).unwrap_or(5);
            bench(&base, &newx, secs, reps, out, arg(&args, "--filter"))
        }
        _ => eprintln!("usage: g6_serve_ab (equiv|bench) [--base EXE] [--new EXE] [--quick] [--secs S] [--reps N] [--out FILE]"),
    }
}
