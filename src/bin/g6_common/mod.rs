//! G6 shared helpers (std only): RNG, stats, Windows process CPU/memory via kernel32 (one FFI block),
//! machine-load probe, server spawning, and the deterministic request scripts.
#![allow(dead_code)]

use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

pub const MAX_LINE: usize = 1 << 20;

// ---------------------------------------------------------------------------------------------
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    pub fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { 0.5 * (s[n / 2 - 1] + s[n / 2]) }
}
pub fn minmax(v: &[f64]) -> (f64, f64) {
    (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
}

// ---------------------------------------------------------------------------------------------
// Windows process accounting (child handle) and system load

#[cfg(windows)]
mod win {
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    pub struct Ft {
        pub lo: u32,
        pub hi: u32,
    }
    #[repr(C)]
    pub struct Pmc {
        pub cb: u32,
        pub page_faults: u32,
        pub peak_ws: usize,
        pub ws: usize,
        pub qpp: usize,
        pub qp: usize,
        pub qpnp: usize,
        pub qnp: usize,
        pub pagefile: usize,
        pub peak_pagefile: usize,
    }
    unsafe extern "system" {
        pub fn GetProcessTimes(h: isize, c: *mut Ft, e: *mut Ft, k: *mut Ft, u: *mut Ft) -> i32;
        pub fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
        pub fn GetSystemTimes(idle: *mut Ft, kernel: *mut Ft, user: *mut Ft) -> i32;
    }
    pub fn u(f: Ft) -> u64 {
        ((f.hi as u64) << 32) | f.lo as u64
    }
}

/// (user_ms, kernel_ms) CPU time consumed so far by a child process.
#[cfg(windows)]
pub fn child_cpu_ms(c: &Child) -> (f64, f64) {
    use std::os::windows::io::AsRawHandle;
    let h = c.as_raw_handle() as isize;
    let (mut a, mut b, mut k, mut u) = (win::Ft::default(), win::Ft::default(), win::Ft::default(), win::Ft::default());
    unsafe { win::GetProcessTimes(h, &mut a, &mut b, &mut k, &mut u) };
    (win::u(u) as f64 / 1e4, win::u(k) as f64 / 1e4)
}
#[cfg(not(windows))]
pub fn child_cpu_ms(_c: &Child) -> (f64, f64) {
    (0.0, 0.0)
}

/// (peak working set, peak pagefile/commit, current working set) bytes of a child process.
#[cfg(windows)]
pub fn child_mem(c: &Child) -> (usize, usize, usize) {
    use std::os::windows::io::AsRawHandle;
    let h = c.as_raw_handle() as isize;
    let mut p = win::Pmc { cb: std::mem::size_of::<win::Pmc>() as u32, page_faults: 0, peak_ws: 0, ws: 0, qpp: 0, qp: 0, qpnp: 0, qnp: 0, pagefile: 0, peak_pagefile: 0 };
    let ok = unsafe { win::K32GetProcessMemoryInfo(h, &mut p, p.cb) };
    if ok == 0 { (0, 0, 0) } else { (p.peak_ws, p.peak_pagefile, p.ws) }
}
#[cfg(not(windows))]
pub fn child_mem(_c: &Child) -> (usize, usize, usize) {
    (0, 0, 0)
}

#[cfg(windows)]
fn sys_times() -> (u64, u64, u64) {
    let (mut i, mut k, mut u) = (win::Ft::default(), win::Ft::default(), win::Ft::default());
    unsafe { win::GetSystemTimes(&mut i, &mut k, &mut u) };
    (win::u(i), win::u(k), win::u(u))
}
#[cfg(not(windows))]
fn sys_times() -> (u64, u64, u64) {
    (0, 1, 0)
}

/// System-wide CPU busy % over `ms` milliseconds.
pub fn cpu_busy_pct(ms: u64) -> f64 {
    let a = sys_times();
    std::thread::sleep(std::time::Duration::from_millis(ms));
    let b = sys_times();
    let (di, dk, du) = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
    100.0 * (1.0 - di as f64 / (dk + du).max(1) as f64)
}

/// Wait up to `max_s` for the box to drop below `thresh` % busy; returns the last reading.
pub fn wait_quiet(max_s: u64, thresh: f64) -> f64 {
    let t0 = std::time::Instant::now();
    loop {
        let l = cpu_busy_pct(300);
        if l < thresh || t0.elapsed().as_secs() >= max_s {
            return l;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// server processes

/// Extra environment for children started from a given exe (used to switch an opt-in knob on for one side).
pub static EXTRA_ENV: std::sync::Mutex<Vec<(PathBuf, String, String)>> = std::sync::Mutex::new(Vec::new());

pub fn apply_env(cmd: &mut Command, exe: &Path) {
    for (p, k, v) in EXTRA_ENV.lock().unwrap().iter() {
        if p == exe {
            cmd.env(k, v);
        }
    }
}

pub struct Srv {
    pub child: Child,
    pub addr: String,
    pub dir: PathBuf,
    pub state: PathBuf,
    _out: BufReader<std::process::ChildStdout>,
}

pub fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("g6_{}_{}", std::process::id(), tag));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `serve --port 0` on 127.0.0.1 only; returns once it reports its listening address.
pub fn spawn_tcp(exe: &Path, tag: &str, flush_every: usize, flush_ms: u64) -> Srv {
    let dir = tmp_dir(tag);
    let state = dir.join("state.json");
    let mut cmd = Command::new(exe);
    apply_env(&mut cmd, exe);
    let mut child = cmd
        .args(["serve", "--state", state.to_str().unwrap(), "--port", "0", "--flush-every", &flush_every.to_string(), "--flush-ms", &flush_ms.to_string()])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn server");
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    out.read_line(&mut line).unwrap();
    let v: serde_json::Value = serde_json::from_str(line.trim()).expect("listening line");
    let addr = v["listening"].as_str().unwrap().to_string();
    assert!(addr.starts_with("127.0.0.1:"), "server must bind loopback only, got {addr}");
    Srv { child, addr, dir, state, _out: out }
}

impl Srv {
    pub fn kill(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

// ---------------------------------------------------------------------------------------------
// request scripts (each element = one complete line, newline included)

pub fn line(s: &str) -> Vec<u8> {
    let mut v = s.as_bytes().to_vec();
    v.push(b'\n');
    v
}

/// Would the server answer this raw line (newline included)? Mirrors the documented protocol:
/// too long and bad UTF-8 are answered, blank lines are not.
pub fn expects_reply(raw: &[u8]) -> bool {
    if raw.len() > MAX_LINE {
        return true;
    }
    match std::str::from_utf8(raw) {
        Err(_) => true,
        Ok(t) => !t.trim().is_empty(),
    }
}

fn feats(r: &mut Rng) -> String {
    let k = 1 + r.below(3);
    let v: Vec<String> = (0..k).map(|_| format!("\"kw{}\"", r.below(24))).collect();
    format!("[{}]", v.join(","))
}

fn valid_request(r: &mut Rng, id: usize) -> String {
    let agents = ["a0", "a1", "a2", "a3", "a4", "a5", "a6"];
    let ag = |r: &mut Rng| agents[r.below(agents.len())];
    let idpart = match r.below(4) {
        0 => String::new(),
        1 => format!(",\"id\":{id}"),
        2 => format!(",\"id\":\"req-{id}\""),
        _ => format!(",\"id\":{{\"n\":{id},\"x\":[1,2]}}"),
    };
    match r.below(10) {
        0..=3 => {
            let mut extra = String::new();
            if r.below(4) == 0 {
                extra += &format!(",\"team\":{}", 1 + r.below(3));
            }
            if r.below(5) == 0 {
                extra += &format!(",\"seed\":{}", r.below(100000));
            }
            if r.below(6) == 0 {
                extra += &format!(",\"prior\":\"{}\"", ag(r));
            }
            if r.below(8) == 0 {
                extra += ",\"type\":\"sql\"";
            }
            if r.below(8) == 0 {
                extra += &format!(",\"candidates\":[\"{}\",\"{}\",\"{}\"]", ag(r), ag(r), ag(r));
            }
            if r.below(10) == 0 {
                extra += ",\"prior_k\":4";
            }
            format!("{{\"cmd\":\"route\",\"features\":{}{extra}{idpart}}}", feats(r))
        }
        4..=6 => {
            let team = if r.below(3) == 0 { format!(",\"team\":[\"{}\",\"{}\"]", ag(r), ag(r)) } else { String::new() };
            format!("{{\"cmd\":\"feedback\",\"features\":{},\"agent\":\"{}\",\"reward\":{:.3}{team}{idpart}}}", feats(r), ag(r), r.f())
        }
        7 => format!("{{\"cmd\":\"stats\"{idpart}}}"),
        8 => {
            let c = match r.below(4) {
                0 => "\"eta\":0.02".to_string(),
                1 => format!("\"temperature\":{:.2}", 0.1 + r.f()),
                2 => format!("\"team\":{}", 1 + r.below(3)),
                _ => "\"decay\":0.001,\"epsilon\":0.05".to_string(),
            };
            format!("{{\"cmd\":\"config\",{c}{idpart}}}")
        }
        _ => format!("{{\"cmd\":\"flush\"{idpart}}}"),
    }
}

/// Malformed / hostile / edge-case lines, hand-written.
pub fn error_lines() -> Vec<Vec<u8>> {
    let mut v: Vec<Vec<u8>> = Vec::new();
    let texts: Vec<String> = vec![
        r#""#.into(),
        r#"   "#.into(),
        "\t".into(),
        "\r".into(),
        r#"not json"#.into(),
        r#"{"#.into(),
        r#"{"cmd":"#.into(),
        r#"[]"#.into(),
        r#"[1,2,3]"#.into(),
        r#""cmd""#.into(),
        r#"null"#.into(),
        r#"123"#.into(),
        r#"true"#.into(),
        r#"{}"#.into(),
        r#"{"cmd":1}"#.into(),
        r#"{"cmd":null}"#.into(),
        r#"{"cmd":["route"]}"#.into(),
        r#"{"cmd":""}"#.into(),
        r#"{"cmd":"ROUTE"}"#.into(),
        r#"{"cmd":"nope","id":7}"#.into(),
        r#"{"cmd":"nope","id":null}"#.into(),
        r#"{"id":5}"#.into(),
        r#"{"cmd":"route"}"#.into(),
        r#"{"cmd":"route","id":9}"#.into(),
        r#"{"cmd":"route","features":[],"candidates":[]}"#.into(),
        r#"{"cmd":"route","features":["x"],"candidates":["",""]}"#.into(),
        r#"{"cmd":"route","features":["x"],"candidates":"a0,a1","team":0}"#.into(),
        r#"{"cmd":"route","features":["x"],"candidates":"a0,a1","team":"two"}"#.into(),
        r#"{"cmd":"route","features":["x"],"candidates":"a0,a1","team":-1}"#.into(),
        r#"{"cmd":"route","features":["x"],"candidates":"a0,a1","seed":-5}"#.into(),
        r#"{"cmd":"route","features":["x"],"candidates":"a0,a1","seed":1.5}"#.into(),
        r#"{"cmd":"route","features":["x"],"candidates":"a0,a1","seed":18446744073709551615}"#.into(),
        r#"{"cmd":"route","features":["x"],"candidates":"a0,a1","seed":99999999999999999999999}"#.into(),
        r#"{"cmd":"route","features":["x"],"prior":"a0,a1"}"#.into(),
        r#"{"cmd":"route","features":["x"],"prior":""}"#.into(),
        r#"{"cmd":"route","features":["x"],"bogus":1,"id":3}"#.into(),
        r#"{"cmd":"route","features":["x"],"Features":1}"#.into(),
        r#"{"cmd":"route","features":{"a":1}}"#.into(),
        r#"{"cmd":"route","features":true}"#.into(),
        r#"{"cmd":"route","features":["a",["b"]],"id":4}"#.into(),
        r#"{"cmd":"route","features":["a",null]}"#.into(),
        r#"{"cmd":"route","features":[1,2.5,"c"]}"#.into(),
        r#"{"cmd":"route","features":null}"#.into(),
        r#"{"cmd":"route","features":"a,b,a, c","candidates":["a0","a1"]}"#.into(),
        r#"{"cmd":"route","features":["é中😀"],"id":"é"}"#.into(),
        r#"{"cmd":"feedback"}"#.into(),
        r#"{"cmd":"feedback","features":["x"],"agent":"a0"}"#.into(),
        r#"{"cmd":"feedback","features":["x"],"agent":"a0","reward":2}"#.into(),
        r#"{"cmd":"feedback","features":["x"],"agent":"a0","reward":-0.1}"#.into(),
        r#"{"cmd":"feedback","features":["x"],"agent":"a0","reward":"NaN"}"#.into(),
        r#"{"cmd":"feedback","features":["x"],"agent":"a0","reward":1e999}"#.into(),
        r#"{"cmd":"feedback","features":["x"],"agent":"a0,a1","reward":1}"#.into(),
        r#"{"cmd":"feedback","features":["x"],"agent":"","reward":1}"#.into(),
        r#"{"cmd":"feedback","features":["x"],"agent":5,"reward":1,"id":12}"#.into(),
        r#"{"cmd":"feedback","features":["x"],"agent":"a0","reward":0.5,"team":""}"#.into(),
        r#"{"cmd":"feedback","features":["x"],"agent":"a0","reward":"0.5"}"#.into(),
        r#"{"cmd":"stats","extra":1}"#.into(),
        r#"{"cmd":"stats","eta":0.5}"#.into(),
        r#"{"cmd":"stats","eta":"abc"}"#.into(),
        r#"{"cmd":"stats","eta":-1}"#.into(),
        r#"{"cmd":"stats","eta":0.01}"#.into(),
        r#"{"cmd":"config"}"#.into(),
        r#"{"cmd":"config","eta":9999}"#.into(),
        r#"{"cmd":"config","prior_k":-2}"#.into(),
        r#"{"cmd":"config","prior-k":3,"id":1}"#.into(),
        r#"{"cmd":"config","team":0}"#.into(),
        r#"{"cmd":"config","state":"/etc/passwd"}"#.into(),
        r#"{"cmd":"route","state":"/tmp/x","features":["a"]}"#.into(),
        r#"{"cmd":"flush","extra":1,"id":2}"#.into(),
        r#"{"cmd":"flush","id":{"deep":[1,{"a":null}]}}"#.into(),
        r#"{"cmd":"stats","id":1,"id":2}"#.into(),
        r#"{"cmd":"route","cmd":"stats"}"#.into(),
        r#"  {"cmd":"stats","id":"ws"}  "#.into(),
        r#"{"cmd":"stats"} trailing"#.into(),
        r#"{"cmd":"stats"}{"cmd":"stats"}"#.into(),
        r#"{"cmd":"stats",}"#.into(),
        r#"{'cmd':'stats'}"#.into(),
        r#"{"cmd":"stats","id":1e400}"#.into(),
        r#"{"cmd":"stats","id":0.1e-5}"#.into(),
        r#"{"cmd":"stats","id":-0}"#.into(),
        r#"{"cmd":"stats","id":123456789012345678901234567890}"#.into(),
    ];
    for s in &texts {
        v.push(line(s));
    }
    // raw-byte oddities
    v.push(b"\xff\xfe{\"cmd\":\"stats\"}\n".to_vec());
    v.push(b"{\"cmd\":\"stats\",\"id\":\"\xc3\x28\"}\n".to_vec());
    v.push(b"\x00\n".to_vec());
    v.push(b"{\"cmd\":\"stats\"}\x00\n".to_vec());
    v.push(b"\xEF\xBB\xBF{\"cmd\":\"stats\"}\n".to_vec());
    v.push(b"{\"cmd\":\"stats\",\"id\":1}\r\n".to_vec());
    v.push(b"\r\n".to_vec());
    v.push(b"\n".to_vec());
    v.push(b"\n\n\n".to_vec());
    v.push("\u{a0}\u{2003}{\"cmd\":\"stats\",\"id\":\"unicode-ws\"}\u{2003}\n".as_bytes().to_vec());
    v.push("\u{a0}\n".as_bytes().to_vec());
    // deep nesting (serde_json recursion limit) and long-but-legal lines
    v.push(line(&format!("{}1{}", "[".repeat(200), "]".repeat(200))));
    v.push(line(&format!("{{\"cmd\":\"stats\",\"id\":{}1{}}}", "[".repeat(100), "]".repeat(100))));
    v.push(line(&format!("{{\"cmd\":\"stats\",\"id\":{}1{}}}", "[".repeat(140), "]".repeat(140))));
    v.push(line(&format!("{{\"cmd\":\"route\",\"features\":[\"{}\"],\"candidates\":[\"a0\",\"a1\"]}}", "f".repeat(5000))));
    let many: Vec<String> = (0..400).map(|i| format!("\"m{i}\"")).collect();
    v.push(line(&format!("{{\"cmd\":\"route\",\"features\":[{}],\"candidates\":[\"a0\",\"a1\"]}}", many.join(","))));
    v
}

/// Size-boundary lines around MAX_LINE (bytes counted with the trailing newline). Valid JSON, so a
/// line within the limit gets a real reply and one over it gets "request line too long".
pub fn boundary_lines() -> Vec<Vec<u8>> {
    let mut v = Vec::new();
    let base = "{\"cmd\":\"stats\",\"id\":\"\"}";
    for total in [MAX_LINE - 1, MAX_LINE, MAX_LINE + 1, MAX_LINE + 2] {
        let pad = total - 1 - base.len();
        let mut s = String::from("{\"cmd\":\"stats\",\"id\":\"");
        s.push_str(&"p".repeat(pad));
        s.push_str("\"}");
        assert_eq!(s.len() + 1, total);
        v.push(line(&s));
        v.push(line("{\"cmd\":\"stats\",\"id\":\"after-boundary\"}"));
    }
    // far over the limit: valid JSON, garbage, and invalid UTF-8; each followed by a good line
    v.push(line(&format!("{{\"cmd\":\"route\",\"features\":\"{}\"}}", "a".repeat(3 * MAX_LINE))));
    v.push(line("{\"cmd\":\"stats\",\"id\":\"after-3mb-valid\"}"));
    v.push(line(&"z".repeat(2 * MAX_LINE + 17)));
    v.push(line("{\"cmd\":\"stats\",\"id\":\"after-2mb-garbage\"}"));
    let mut bad = vec![0xffu8; MAX_LINE + 5];
    bad.push(b'\n');
    v.push(bad);
    v.push(line("{\"cmd\":\"stats\",\"id\":\"after-bad-utf8-big\"}"));
    v
}

/// Mutate a valid request: truncation, byte flips, duplicated/inserted junk.
fn fuzz_line(r: &mut Rng, id: usize) -> Vec<u8> {
    let mut b = valid_request(r, id).into_bytes();
    let n = 1 + r.below(3);
    for _ in 0..n {
        if b.is_empty() {
            break;
        }
        match r.below(6) {
            0 => {
                let k = r.below(b.len());
                b.truncate(k);
            }
            1 => {
                let k = r.below(b.len());
                b[k] = (r.next() & 0xff) as u8;
            }
            2 => {
                let k = r.below(b.len());
                b.remove(k);
            }
            3 => {
                let k = r.below(b.len() + 1);
                let junk = [b'"', b',', b'{', b'}', b'[', b']', b':', b' ', b'\\', 0xc3, 0x00][r.below(11)];
                b.insert(k, junk);
            }
            4 => {
                let k = r.below(b.len());
                let m = r.below(b.len() - k);
                let seg = b[k..k + m].to_vec();
                b.splice(k..k, seg);
            }
            _ => {
                let k = r.below(b.len());
                b[k] = b[k].to_ascii_uppercase();
            }
        }
    }
    b.retain(|&c| c != b'\n');
    b.push(b'\n');
    b
}

/// The deterministic request script. `big` adds the >1 MiB boundary lines (about 20 MiB total).
pub fn script(seed: u64, n_ops: usize, n_fuzz: usize, big: bool) -> Vec<Vec<u8>> {
    let mut r = Rng(seed);
    let mut out: Vec<Vec<u8>> = Vec::new();
    // seed the agent pool so route-without-candidates works
    out.push(line("{\"cmd\":\"route\",\"features\":[\"warm\"],\"candidates\":[\"a0\",\"a1\",\"a2\",\"a3\"],\"id\":\"warm\"}"));
    let errs = error_lines();
    let mut ei = 0;
    for i in 0..n_ops {
        out.push(line(&valid_request(&mut r, i)));
        if i % 7 == 3 && ei < errs.len() {
            out.push(errs[ei].clone());
            ei += 1;
        }
        if i % 97 == 50 {
            out.push(b"\n".to_vec());
        }
    }
    out.extend(errs[ei..].iter().cloned());
    for i in 0..n_fuzz {
        out.push(fuzz_line(&mut r, i));
    }
    out.push(line("{\"cmd\":\"stats\",\"id\":\"mid-stats\"}"));
    if big {
        out.extend(boundary_lines());
    }
    for i in 0..30 {
        out.push(line(&valid_request(&mut r, 90000 + i)));
    }
    out.push(line("{\"cmd\":\"stats\",\"id\":\"final-stats\"}"));
    out
}

pub fn flatten(lines: &[Vec<u8>]) -> Vec<u8> {
    let mut v = Vec::with_capacity(lines.iter().map(|l| l.len()).sum());
    for l in lines {
        v.extend_from_slice(l);
    }
    v
}
