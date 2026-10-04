//! F3 RESOURCE-REDUCTION part 3: real wall-clock routes/sec with N parallel agent-clients sharing ONE brain.
//!
//! Variants (all std threads, no new crates):
//!   crate-mutex      Mutex<Guarded<HebbianRouter>>  (what `neural-mesh serve` effectively is: one global lock)
//!   compact-mutex    Mutex<compact guarded u16/q16>
//!   compact-rwlock   RwLock<compact guarded u16/q16>: route = read lock (pure), feedback = write lock
//!   compact-sharded  lock-free row lookup (open addressing, hashed ids) + one Mutex per feature row
//!   independent      N private brains (no sharing; upper bound = pure compute scaling)
//!   serve-tcp        the real `neural-mesh serve --port` process, N TCP clients, JSON lines (end to end)
//! Each client loop = one route + one feedback ("a decision"), strings -> hashed ids included.
//!
//! Usage: cargo build --release --bin neural-mesh --bin f3_throughput
//!        f3_throughput [--threads 1,2,4,8] [--secs 1.5] [--reps 3] [--out results/f3_throughput.json] [--no-serve]

#[path = "f3_common/mod.rs"]
mod common;

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use common::brain::*;
use common::*;
use neural_mesh::defaults::recommended_router;
use neural_mesh::prior::Guarded;
use neural_mesh::router::{Rng, Router};
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::sim::Scenario;
use serde_json::{Value, json};

type U16 = Fx16;
type Q = Q16<1024>;
type Dense = Compact<DenseStore<U16, Q>>;

// =============================================================================================
// lock-sharded compact brain (one mutex per feature row, lock-free row lookup)

struct Shard {
    key: AtomicU64, // id + 1, 0 = empty
    row: Mutex<Option<DenseStore<U16, Q>>>,
}
struct Sharded {
    n: usize,
    p: Params,
    kw: Arc<[(u32, u16)]>,
    tab: Box<[Shard]>,
    mask: usize,
    rows: AtomicU32,
}
impl Sharded {
    fn new(n: usize, kw: Arc<[(u32, u16)]>, cap_pow2: usize) -> Self {
        let tab = (0..cap_pow2).map(|_| Shard { key: AtomicU64::new(0), row: Mutex::new(None) }).collect::<Vec<_>>().into_boxed_slice();
        Sharded { n, p: Params::recommended(), kw, tab, mask: cap_pow2 - 1, rows: AtomicU32::new(0) }
    }
    #[inline]
    fn start(&self, id: u32) -> usize {
        (id.wrapping_mul(0x9E37_79B1) as usize >> 7) & self.mask
    }
    fn find(&self, id: u32) -> Option<usize> {
        let mut i = self.start(id);
        loop {
            let k = self.tab[i].key.load(Ordering::Acquire);
            if k == id as u64 + 1 {
                return Some(i);
            }
            if k == 0 {
                return None;
            }
            i = (i + 1) & self.mask;
        }
    }
    fn find_or_create(&self, id: u32) -> usize {
        let mut i = self.start(id);
        loop {
            let k = self.tab[i].key.load(Ordering::Acquire);
            if k == id as u64 + 1 {
                return i;
            }
            if k == 0 {
                match self.tab[i].key.compare_exchange(0, id as u64 + 1, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_) => {
                        self.rows.fetch_add(1, Ordering::Relaxed);
                        return i;
                    }
                    Err(cur) if cur == id as u64 + 1 => return i,
                    Err(_) => continue, // someone else took this slot with another key: re-read it
                }
            }
            i = (i + 1) & self.mask;
        }
    }
    fn route(&self, raw: &[u32], cand: &[usize], rng: &mut Rng) -> usize {
        let n = self.n;
        let mut agg = Agg::new(n);
        agg.s[..n].fill(0.0);
        agg.f[..n].fill(0.0);
        agg.score[..n].fill(0.0);
        let mut seen = [0u32; MAXF];
        let mut ns = 0;
        for &id in raw {
            let first = !seen[..ns].contains(&id);
            if first {
                seen[ns] = id;
                ns += 1;
            }
            let slot = self.find(id);
            let mut have = false;
            if let Some(i) = slot {
                let g = self.tab[i].row.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(st) = g.as_ref() {
                    have = true;
                    st.add_counts(0, &mut agg.s[..n], &mut agg.f[..n]);
                    if first {
                        st.add_scores(0, &mut agg.score[..n]);
                    }
                }
            }
            if !have && first {
                for a in 0..n {
                    agg.score[a] += neural_mesh::hebbian::INIT_WEIGHT;
                }
            }
        }
        let kwc = keyword_choice(&self.kw, raw, n, cand);
        decide_select(&self.p, n, kwc, raw.len(), &agg, cand, rng).0
    }
    fn feedback(&self, raw: &[u32], agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        let mut seen = [0u32; MAXF];
        let mut ns = 0;
        for &id in raw {
            let first = !seen[..ns].contains(&id);
            if first {
                seen[ns] = id;
                ns += 1;
            }
            let i = self.find_or_create(id);
            let mut g = self.tab[i].row.lock().unwrap_or_else(|e| e.into_inner());
            let st = g.get_or_insert_with(|| {
                let mut s = <DenseStore<U16, Q> as Store>::new(self.n);
                s.push_row();
                s
            });
            st.cnt_update(0, &self.p, agent, r);
            if first {
                st.heb_update(0, &self.p, agent, r);
            }
        }
    }
}

// =============================================================================================
// harness

struct Work {
    env: Env,
    tb: Tables2,
    /// hashed ids per task
    ids: Vec<IdBuf>,
    kw: Arc<[(u32, u16)]>,
    n: usize,
}
struct Tables2 {
    n: usize,
    out: Vec<u8>,
}

fn make_work(spec: &WorldSpec, seed: u64) -> Work {
    let env = spec.make(seed);
    let (n, len) = (env.n_agents(), env.len());
    let mut out = vec![0u8; n * len];
    for t in 0..len {
        for a in 0..n {
            out[t * n + a] = env.outcome(t, a) as u8;
        }
    }
    let mut ids = Vec::with_capacity(len);
    for t in 0..len {
        ids.push(hash_ids(&env.task(t).features));
    }
    let mut v: Vec<(u32, u16)> = env.kw_map().iter().map(|(k, &a)| (hash32(k), a as u16)).collect();
    v.sort_by_key(|e| e.0);
    Work { env, tb: Tables2 { n, out }, ids, kw: v.into(), n }
}

#[inline]
fn hash_ids(features: &[String]) -> IdBuf {
    let mut b = IdBuf { a: [0; MAXF], len: 0 };
    for f in features.iter().take(MAXF) {
        b.a[b.len] = hash32(f);
        b.len += 1;
    }
    b
}

/// A shared service under test.
trait Service: Sync {
    /// one decision for task index `t` using the caller's rng; returns the agent
    fn decide(&self, w: &Work, t: usize, rng: &mut Rng) -> usize;
}

struct CrateMutex(Mutex<Guarded<HebbianRouter>>);
impl Service for CrateMutex {
    fn decide(&self, w: &Work, t: usize, rng: &mut Rng) -> usize {
        let task = w.env.task(t);
        let cands: [usize; 64] = std::array::from_fn(|i| i);
        let a = self.0.lock().unwrap().select(task, &cands[..w.n], rng);
        let r = w.tb.out[t * w.tb.n + a] as f64;
        self.0.lock().unwrap().update(task, a, r);
        a
    }
}
struct CompactMutex(Mutex<Dense>);
impl Service for CompactMutex {
    fn decide(&self, w: &Work, t: usize, rng: &mut Rng) -> usize {
        let ids = hash_ids(&w.env.task(t).features); // strings -> ids is part of the request path
        let cands: [usize; 64] = std::array::from_fn(|i| i);
        let a = self.0.lock().unwrap().route_ids_mut(ids.as_slice(), &cands[..w.n], rng);
        let r = w.tb.out[t * w.tb.n + a] as f64;
        self.0.lock().unwrap().feedback_ids(ids.as_slice(), a, r);
        a
    }
}
struct CompactRw(RwLock<Dense>);
impl Service for CompactRw {
    fn decide(&self, w: &Work, t: usize, rng: &mut Rng) -> usize {
        let ids = hash_ids(&w.env.task(t).features);
        let cands: [usize; 64] = std::array::from_fn(|i| i);
        let (a, _) = self.0.read().unwrap().route_ids(ids.as_slice(), &cands[..w.n], rng);
        let r = w.tb.out[t * w.tb.n + a] as f64;
        self.0.write().unwrap().feedback_ids(ids.as_slice(), a, r);
        a
    }
}
struct CompactSharded(Sharded);
impl Service for CompactSharded {
    fn decide(&self, w: &Work, t: usize, rng: &mut Rng) -> usize {
        let ids = hash_ids(&w.env.task(t).features);
        let cands: [usize; 64] = std::array::from_fn(|i| i);
        let a = self.0.route(ids.as_slice(), &cands[..w.n], rng);
        let r = w.tb.out[t * w.tb.n + a] as f64;
        self.0.feedback(ids.as_slice(), a, r);
        a
    }
}

struct RunOut {
    decisions: u64,
    secs: f64,
    p50_us: f64,
    p99_us: f64,
}

/// N threads hammer `svc` for `secs`; each decision is one route + one feedback.
fn hammer(svc: &dyn Service, w: &Work, threads: usize, secs: f64) -> RunOut {
    let stop = AtomicBool::new(false);
    let barrier = std::sync::Barrier::new(threads + 1);
    let totals: Vec<Mutex<(u64, Vec<f32>)>> = (0..threads).map(|_| Mutex::new((0, vec![]))).collect();
    let len = w.env.len();
    let mut elapsed = 0.0;
    std::thread::scope(|s| {
        for tid in 0..threads {
            let (stop, barrier, totals) = (&stop, &barrier, &totals);
            s.spawn(move || {
                let mut rng = Rng::new(1000 + tid as u64);
                let mut t = (tid * 997) % len;
                let mut n = 0u64;
                let mut lat: Vec<f32> = Vec::with_capacity(4096);
                barrier.wait();
                while !stop.load(Ordering::Relaxed) {
                    for k in 0..32 {
                        if k == 0 && lat.len() < 200_000 {
                            let t0 = Instant::now();
                            std::hint::black_box(svc.decide(w, t, &mut rng));
                            lat.push(t0.elapsed().as_nanos() as f32 / 1000.0);
                        } else {
                            std::hint::black_box(svc.decide(w, t, &mut rng));
                        }
                        t += 1;
                        if t == len {
                            t = 0;
                        }
                        n += 1;
                    }
                }
                *totals[tid].lock().unwrap() = (n, lat);
            });
        }
        barrier.wait();
        let t0 = Instant::now();
        std::thread::sleep(Duration::from_secs_f64(secs));
        stop.store(true, Ordering::Relaxed);
        elapsed = t0.elapsed().as_secs_f64();
    });
    let mut dec = 0u64;
    let mut lat: Vec<f32> = vec![];
    for t in totals {
        let (n, l) = t.into_inner().unwrap();
        dec += n;
        lat.extend(l);
    }
    lat.sort_by(|a, b| a.total_cmp(b));
    let pick = |q: f64| if lat.is_empty() { 0.0 } else { lat[((lat.len() - 1) as f64 * q) as usize] as f64 };
    RunOut { decisions: dec, secs: elapsed, p50_us: pick(0.5), p99_us: pick(0.99) }
}

fn pretrain_dense(w: &Work, k: usize) -> Dense {
    let mut b = Dense::new(w.n, Params::recommended(), w.kw.clone());
    let cands: Vec<usize> = (0..w.n).collect();
    let mut rng = Rng::new(5);
    for t in 0..k.min(w.env.len()) {
        let a = b.route_ids_mut(w.ids[t].as_slice(), &cands, &mut rng);
        b.feedback_ids(w.ids[t].as_slice(), a, w.tb.out[t * w.n + a] as f64);
    }
    b
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

// =============================================================================================
// the real server

fn serve_bench(w: &Work, threads: usize, secs: f64, flush_every: Option<usize>) -> Option<RunOut> {
    let exe = std::env::current_exe().ok()?;
    let srv_exe = exe.parent()?.join(if cfg!(windows) { "neural-mesh.exe" } else { "neural-mesh" });
    if !srv_exe.exists() {
        return None;
    }
    let dir = std::env::temp_dir().join(format!("f3_serve_{}_{}", std::process::id(), threads));
    let _ = std::fs::create_dir_all(&dir);
    let state = dir.join("state.json");
    let mut cmd = std::process::Command::new(&srv_exe);
    cmd.args(["serve", "--state", state.to_str()?, "--port", "0"]);
    if let Some(f) = flush_every {
        cmd.args(["--flush-every", &f.to_string(), "--flush-ms", "600000"]);
    }
    cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).stdin(std::process::Stdio::null());
    let mut child = cmd.spawn().ok()?;
    let mut rd = BufReader::new(child.stdout.take()?);
    let mut line = String::new();
    rd.read_line(&mut line).ok()?;
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    let addr = v["listening"].as_str()?.to_string();
    let names: Vec<String> = (0..w.n).map(|i| format!("a{i}")).collect();
    // register agents + brief warm-up
    {
        let mut s = TcpStream::connect(&addr).ok()?;
        s.set_nodelay(true).ok();
        let mut br = BufReader::new(s.try_clone().ok()?);
        let req = json!({"cmd":"route","features":["warm"],"candidates":names});
        writeln!(s, "{req}").ok()?;
        let mut l = String::new();
        br.read_line(&mut l).ok()?;
    }
    let stop = AtomicBool::new(false);
    let barrier = std::sync::Barrier::new(threads + 1);
    let totals: Vec<Mutex<(u64, Vec<f32>)>> = (0..threads).map(|_| Mutex::new((0, vec![]))).collect();
    let len = w.env.len();
    let mut elapsed = 0.0;
    std::thread::scope(|sc| {
        for tid in 0..threads {
            let (stop, barrier, totals, addr) = (&stop, &barrier, &totals, addr.clone());
            sc.spawn(move || {
                let mut s = TcpStream::connect(&addr).unwrap();
                s.set_nodelay(true).ok();
                let mut br = BufReader::new(s.try_clone().unwrap());
                let mut t = (tid * 997) % len;
                let mut n = 0u64;
                let mut lat: Vec<f32> = Vec::with_capacity(4096);
                let mut resp = String::new();
                barrier.wait();
                while !stop.load(Ordering::Relaxed) {
                    let t0 = Instant::now();
                    let feats = &w.env.task(t).features;
                    let req = json!({"cmd":"route","features":feats});
                    if s.write_all(format!("{req}{}", (10u8) as char).as_bytes()).is_err() {
                        break;
                    }
                    resp.clear();
                    if br.read_line(&mut resp).unwrap_or(0) == 0 {
                        break;
                    }
                    let v: Value = serde_json::from_str(resp.trim()).unwrap_or(Value::Null);
                    let agent = v["agent"].as_str().unwrap_or("a0").to_string();
                    let ai: usize = agent.trim_start_matches('a').parse().unwrap_or(0).min(w.n - 1);
                    let r = w.tb.out[t * w.tb.n + ai] as f64;
                    let req = json!({"cmd":"feedback","features":feats,"agent":agent,"reward":r});
                    if s.write_all(format!("{req}{}", (10u8) as char).as_bytes()).is_err() {
                        break;
                    }
                    resp.clear();
                    if br.read_line(&mut resp).unwrap_or(0) == 0 {
                        break;
                    }
                    lat.push(t0.elapsed().as_nanos() as f32 / 1000.0);
                    n += 1;
                    t += 1;
                    if t == len {
                        t = 0;
                    }
                }
                *totals[tid].lock().unwrap() = (n, lat);
            });
        }
        barrier.wait();
        let t0 = Instant::now();
        std::thread::sleep(Duration::from_secs_f64(secs));
        stop.store(true, Ordering::Relaxed);
        elapsed = t0.elapsed().as_secs_f64();
    });
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
    let mut dec = 0u64;
    let mut lat: Vec<f32> = vec![];
    for t in totals {
        let (n, l) = t.into_inner().unwrap();
        dec += n;
        lat.extend(l);
    }
    lat.sort_by(|a, b| a.total_cmp(b));
    let pick = |q: f64| if lat.is_empty() { 0.0 } else { lat[((lat.len() - 1) as f64 * q) as usize] as f64 };
    Some(RunOut { decisions: dec, secs: elapsed, p50_us: pick(0.5), p99_us: pick(0.99) })
}

fn spawn_server(w: &Work, tag: usize, flush_every: Option<usize>) -> Option<(std::process::Child, String, std::path::PathBuf)> {
    let exe = std::env::current_exe().ok()?;
    let srv_exe = exe.parent()?.join(if cfg!(windows) { "neural-mesh.exe" } else { "neural-mesh" });
    if !srv_exe.exists() {
        return None;
    }
    let dir = std::env::temp_dir().join(format!("f3_serve_{}_{}", std::process::id(), tag));
    let _ = std::fs::create_dir_all(&dir);
    let state = dir.join("state.json");
    let mut cmd = std::process::Command::new(&srv_exe);
    cmd.args(["serve", "--state", state.to_str()?, "--port", "0"]);
    if let Some(f) = flush_every {
        cmd.args(["--flush-every", &f.to_string(), "--flush-ms", "600000"]);
    }
    cmd.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).stdin(std::process::Stdio::null());
    let mut child = cmd.spawn().ok()?;
    let mut rd = BufReader::new(child.stdout.take()?);
    let mut line = String::new();
    rd.read_line(&mut line).ok()?;
    let v: Value = serde_json::from_str(line.trim()).ok()?;
    let addr = v["listening"].as_str()?.to_string();
    let names: Vec<String> = (0..w.n).map(|i| format!("a{i}")).collect();
    let mut s = TcpStream::connect(&addr).ok()?;
    s.set_nodelay(true).ok();
    let mut br = BufReader::new(s.try_clone().ok()?);
    writeln!(s, "{}", json!({"cmd":"route","features":["warm"],"candidates":names})).ok()?;
    let mut l = String::new();
    br.read_line(&mut l).ok()?;
    Some((child, addr, dir))
}

/// Pipelined load on the real server: each connection sends batches of 32 route + 32 feedback lines in one write
/// and then reads 64 replies (synthetic feedback agent: measures SERVER capacity, not client round trips).
fn serve_pipe(w: &Work, conns: usize, secs: f64, flush_every: Option<usize>) -> Option<RunOut> {
    let (mut child, addr, dir) = spawn_server(w, 1000 + conns, flush_every)?;
    let stop = AtomicBool::new(false);
    let barrier = std::sync::Barrier::new(conns + 1);
    let totals: Vec<AtomicU64> = (0..conns).map(|_| AtomicU64::new(0)).collect();
    let len = w.env.len();
    let mut elapsed = 0.0;
    std::thread::scope(|sc| {
        for tid in 0..conns {
            let (stop, barrier, totals, addr) = (&stop, &barrier, &totals, addr.clone());
            sc.spawn(move || {
                let mut s = TcpStream::connect(&addr).unwrap();
                s.set_nodelay(true).ok();
                let mut br = BufReader::new(s.try_clone().unwrap());
                let mut t = (tid * 997) % len;
                let mut n = 0u64;
                let mut resp = String::new();
                barrier.wait();
                while !stop.load(Ordering::Relaxed) {
                    let mut buf = String::new();
                    for k in 0..32 {
                        let feats = &w.env.task((t + k) % len).features;
                        buf.push_str(&json!({"cmd":"route","features":feats}).to_string());
                        buf.push((10u8) as char);
                        buf.push_str(&json!({"cmd":"feedback","features":feats,"agent":format!("a{}", (t + k) % w.n),"reward":1.0}).to_string());
                        buf.push((10u8) as char);
                    }
                    if s.write_all(buf.as_bytes()).is_err() {
                        break;
                    }
                    for _ in 0..64 {
                        resp.clear();
                        if br.read_line(&mut resp).unwrap_or(0) == 0 {
                            return;
                        }
                    }
                    t = (t + 32) % len;
                    n += 32;
                }
                totals[tid].store(n, Ordering::Relaxed);
            });
        }
        barrier.wait();
        let t0 = Instant::now();
        std::thread::sleep(Duration::from_secs_f64(secs));
        stop.store(true, Ordering::Relaxed);
        elapsed = t0.elapsed().as_secs_f64();
    });
    let _ = child.kill();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
    let dec: u64 = totals.iter().map(|c| c.load(Ordering::Relaxed)).sum();
    Some(RunOut { decisions: dec, secs: elapsed, p50_us: 0.0, p99_us: 0.0 })
}

// =============================================================================================

// =============================================================================================
// serve diagnosis: server CPU ceiling without a network (stdio) and the cost of its unbuffered TCP writes

struct CountW {
    calls: usize,
    bytes: usize,
}
impl Write for CountW {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.calls += 1;
        self.bytes += b.len();
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// `neural-mesh serve --stdio`: n_dec route+feedback pairs pushed through a pipe (no sockets): the CPU ceiling
/// of the server's request path (JSON parse + command + JSON build), one global mutex.
fn serve_stdio(w: &Work, n_dec: usize) -> Option<(f64, f64)> {
    let exe = std::env::current_exe().ok()?;
    let srv_exe = exe.parent()?.join(if cfg!(windows) { "neural-mesh.exe" } else { "neural-mesh" });
    if !srv_exe.exists() {
        return None;
    }
    let dir = std::env::temp_dir().join(format!("f3_stdio_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    let state = dir.join("state.json");
    let names: Vec<String> = (0..w.n).map(|i| format!("a{i}")).collect();
    let mut text = String::new();
    text.push_str(&json!({"cmd":"route","features":["warm"],"candidates":names}).to_string());
    text.push('\n');
    let len = w.env.len();
    for i in 0..n_dec {
        let feats = &w.env.task(i % len).features;
        text.push_str(&json!({"cmd":"route","features":feats}).to_string());
        text.push('\n');
        text.push_str(&json!({"cmd":"feedback","features":feats,"agent":format!("a{}", i % w.n),"reward":1.0}).to_string());
        text.push('\n');
    }
    let mut child = std::process::Command::new(&srv_exe)
        .args(["serve", "--state", state.to_str()?, "--stdio", "--flush-every", "1000000", "--flush-ms", "600000"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let mut sin = child.stdin.take()?;
    let sout = child.stdout.take()?;
    let t0 = Instant::now();
    let writer = std::thread::spawn(move || {
        let _ = sin.write_all(text.as_bytes());
        // dropping stdin = EOF -> server flushes and exits
    });
    let mut lines = 0usize;
    for l in BufReader::new(sout).lines() {
        if l.is_err() {
            break;
        }
        lines += 1;
    }
    let el = t0.elapsed().as_secs_f64();
    let _ = writer.join();
    let _ = child.wait();
    let _ = std::fs::remove_dir_all(&dir);
    if lines < 2 * n_dec {
        return None;
    }
    Some((n_dec as f64 / el, el * 1e6 / (2.0 * n_dec as f64)))
}

fn write_path_diag() -> Value {
    // a typical 6-agent route response of the real server
    let sample = r#"{"agent":"a3","features":["kw1","kw7"],"health":{"a0":1.0,"a1":1.0,"a2":1.0,"a3":1.0,"a4":1.0,"a5":1.0},"ok":true,"scores":{"a0":0.2,"a1":0.1,"a2":0.1,"a3":0.3,"a4":0.1,"a5":0.1},"team":["a3"]}"#;
    let v: Value = serde_json::from_str(sample).unwrap();
    let mut c = CountW { calls: 0, bytes: 0 };
    writeln!(c, "{v}").unwrap();
    let n = 3000usize;
    let mut out = json!({"response_bytes": c.bytes, "write_calls_per_response_with_writeln_on_unbuffered_stream": c.calls});
    for variant in ["unbuffered writeln!(stream, \"{value}\") + flush [what serve.rs does]", "format to String then one write_all", "BufWriter + writeln + flush"] {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let rd = std::thread::spawn(move || {
            let (s, _) = listener.accept().unwrap();
            s.set_nodelay(true).ok();
            let mut br = BufReader::new(s);
            let mut l = String::new();
            let mut got = 0;
            while got < n {
                l.clear();
                if br.read_line(&mut l).unwrap_or(0) == 0 {
                    break;
                }
                got += 1;
            }
            got
        });
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_nodelay(true).ok();
        let t0 = Instant::now();
        match variant.chars().next().unwrap() {
            'u' => {
                for _ in 0..n {
                    writeln!(s, "{v}").unwrap();
                    s.flush().unwrap();
                }
            }
            'f' => {
                for _ in 0..n {
                    let line = format!("{v}\n");
                    s.write_all(line.as_bytes()).unwrap();
                }
            }
            _ => {
                let mut bw = std::io::BufWriter::new(&s);
                for _ in 0..n {
                    writeln!(bw, "{v}").unwrap();
                    bw.flush().unwrap();
                }
            }
        }
        let got = rd.join().unwrap();
        let el = t0.elapsed().as_secs_f64();
        out[variant] = json!({"us_per_response": el * 1e6 / n as f64, "received": got});
    }
    out
}

fn selftest(w: &Work) {
    // the lock-sharded brain must make the same single-threaded decisions as the dense compact brain
    let mut a = Dense::new(w.n, Params::recommended(), w.kw.clone());
    let b = Sharded::new(w.n, w.kw.clone(), 8192);
    let cands: Vec<usize> = (0..w.n).collect();
    let (mut r1, mut r2) = (Rng::new(9), Rng::new(9));
    for t in 0..w.env.len() {
        let x = a.route_ids_mut(w.ids[t].as_slice(), &cands, &mut r1);
        let y = b.route(w.ids[t].as_slice(), &cands, &mut r2);
        assert_eq!(x, y, "sharded diverged at {t}");
        let r = w.tb.out[t * w.n + x] as f64;
        a.feedback_ids(w.ids[t].as_slice(), x, r);
        b.feedback(w.ids[t].as_slice(), y, r);
    }
    println!("[selftest] lock-sharded brain == dense compact brain on {} decisions", w.env.len());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let threads: Vec<usize> = arg("--threads").map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or(vec![1, 2, 4, 8]);
    let secs: f64 = arg("--secs").and_then(|s| s.parse().ok()).unwrap_or(1.5);
    let reps: usize = arg("--reps").and_then(|s| s.parse().ok()).unwrap_or(3);
    let out_path = arg("--out").unwrap_or_else(|| "results/f3_throughput.json".into());
    let no_serve = args.iter().any(|a| a == "--no-serve");
    let quiet: u64 = arg("--quiet-wait").and_then(|s| s.parse().ok()).unwrap_or(90);
    println!("# f3_throughput  logical cores={}  threads={:?}  secs/point={}  reps={}", n_threads(), threads, secs, reps);
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("powershell").args(["-NoProfile", "-Command", "(Get-CimInstance Win32_Processor | Select -First 1 | % { $_.Name + ' / ' + $_.NumberOfCores + ' cores / ' + $_.NumberOfLogicalProcessors + ' threads' }); '' + [math]::Round((Get-Counter '\\Processor(_Total)\\% Processor Time').CounterSamples[0].CookedValue,1) + ' % cpu busy before run'"]).status();
    }

    let mut results = vec![];
    for spec in [WorldSpec::Sim(Scenario::Fixed), WorldSpec::Big] {
        let w = make_work(&spec, 31);
        selftest(&w);
        println!("\n=== world {} ({} agents, {} tasks) ===", spec.label(), w.n, w.env.len());
        println!("{:<26} {:>3} {:>14} {:>9} {:>11} {:>11}", "service", "thr", "decisions/s", "scale", "p50 us", "p99 us");
        let mut per_service = vec![];
        let mut run_point = |name: &str, mkc: &dyn Fn() -> Box<dyn Service>, base: &mut f64, rows: &mut Vec<Value>| {
            for &th in &threads {
                let bg = quiet_bg();
                let mut rates = vec![];
                let (mut p50, mut p99) = (0.0, 0.0);
                for _ in 0..reps {
                    let svc = mkc();
                    let o = hammer(svc.as_ref(), &w, th, secs);
                    rates.push(o.decisions as f64 / o.secs);
                    p50 = o.p50_us;
                    p99 = o.p99_us;
                }
                let r = median(&mut rates);
                if th == threads[0] {
                    *base = r / th as f64;
                }
                let scale = r / (*base * th as f64);
                println!("{:<26} {:>3} {:>14.0} {:>8.0}% {:>11.2} {:>11.2}   bg cpu {:>3.0}%", name, th, r, scale * 100.0, p50, p99, bg);
                rows.push(json!({"bg_cpu_pct_before": bg, "service": name, "threads": th, "decisions_per_s": r, "scaling_efficiency_vs_ideal": scale, "p50_us": p50, "p99_us": p99, "rates_all_reps": rates}));
            }
        };
        // crate-mutex
        let mut base = 0.0;
        let mut rows = vec![];
        let wref = &w;
        run_point("crate-mutex(global lock)", &|| {
            let g = recommended_router(wref.env.kw_map(), wref.n);
            let mut g = g;
            let cands: Vec<usize> = (0..wref.n).collect();
            let mut rng = Rng::new(5);
            for t in 0..2000.min(wref.env.len()) {
                let task = wref.env.task(t);
                let a = g.select(task, &cands, &mut rng);
                g.update(task, a, wref.tb.out[t * wref.n + a] as f64);
            }
            Box::new(CrateMutex(Mutex::new(g)))
        }, &mut base, &mut rows);
        per_service.extend(rows);
        let mut base = 0.0;
        let mut rows = vec![];
        run_point("compact-mutex", &|| Box::new(CompactMutex(Mutex::new(pretrain_dense(wref, 2000)))), &mut base, &mut rows);
        per_service.extend(rows);
        let mut base = 0.0;
        let mut rows = vec![];
        run_point("compact-rwlock", &|| Box::new(CompactRw(RwLock::new(pretrain_dense(wref, 2000)))), &mut base, &mut rows);
        per_service.extend(rows);
        let mut base = 0.0;
        let mut rows = vec![];
        run_point("compact-sharded(row locks)", &|| {
            let b = Sharded::new(wref.n, wref.kw.clone(), 8192);
            let cands: Vec<usize> = (0..wref.n).collect();
            let mut rng = Rng::new(5);
            for t in 0..2000.min(wref.env.len()) {
                let a = b.route(wref.ids[t].as_slice(), &cands, &mut rng);
                b.feedback(wref.ids[t].as_slice(), a, wref.tb.out[t * wref.n + a] as f64);
            }
            Box::new(CompactSharded(b))
        }, &mut base, &mut rows);
        per_service.extend(rows);

        // independent brains: each thread owns a private pre-trained brain (no sharing at all)
        {
            let mut base = 0.0;
            for &th in &threads {
                let mut rates = vec![];
                for _ in 0..reps {
                    let stop = AtomicBool::new(false);
                    let barrier = std::sync::Barrier::new(th + 1);
                    let counts: Vec<AtomicU64> = (0..th).map(|_| AtomicU64::new(0)).collect();
                    let mut elapsed = 0.0;
                    std::thread::scope(|s| {
                        for tid in 0..th {
                            let (stop, barrier, counts, w) = (&stop, &barrier, &counts, &w);
                            s.spawn(move || {
                                let mut b = pretrain_dense(w, 2000);
                                let cands: Vec<usize> = (0..w.n).collect();
                                let mut rng = Rng::new(77 + tid as u64);
                                let mut t = (tid * 997) % w.env.len();
                                let mut n = 0u64;
                                barrier.wait();
                                while !stop.load(Ordering::Relaxed) {
                                    for _ in 0..32 {
                                        let ids = hash_ids(&w.env.task(t).features);
                                        let a = b.route_ids_mut(ids.as_slice(), &cands, &mut rng);
                                        b.feedback_ids(ids.as_slice(), a, w.tb.out[t * w.n + a] as f64);
                                        t += 1;
                                        if t == w.env.len() {
                                            t = 0;
                                        }
                                        n += 1;
                                    }
                                }
                                counts[tid].store(n, Ordering::Relaxed);
                            });
                        }
                        barrier.wait();
                        let t0 = Instant::now();
                        std::thread::sleep(Duration::from_secs_f64(secs));
                        stop.store(true, Ordering::Relaxed);
                        elapsed = t0.elapsed().as_secs_f64();
                    });
                    let tot: u64 = counts.iter().map(|c| c.load(Ordering::Relaxed)).sum();
                    rates.push(tot as f64 / elapsed);
                }
                let r = median(&mut rates);
                if th == threads[0] {
                    base = r / th as f64;
                }
                let scale = r / (base * th as f64);
                println!("{:<26} {:>3} {:>14.0} {:>8.0}% {:>11} {:>11}", "independent brains", th, r, scale * 100.0, "-", "-");
                per_service.push(json!({"service": "independent brains (no sharing)", "threads": th, "decisions_per_s": r, "scaling_efficiency_vs_ideal": scale, "rates_all_reps": rates}));
            }
        }

        // the real server (only the 6-agent sim world: the CLI state is the mesh+Hebbian router)
        if !no_serve && matches!(spec, WorldSpec::Sim(_)) {
            for (label, fe) in [("serve-tcp (defaults flush 64/1s)", None), ("serve-tcp (flush off)", Some(1_000_000usize))] {
                let mut base = 0.0;
                let mut any = false;
                for &th in &threads {
                    let mut rates = vec![];
                    let (mut p50, mut p99) = (0.0, 0.0);
                    for _ in 0..reps.min(2) {
                        if let Some(o) = serve_bench(&w, th, secs.max(2.0), fe) {
                            any = true;
                            rates.push(o.decisions as f64 / o.secs);
                            p50 = o.p50_us;
                            p99 = o.p99_us;
                        }
                    }
                    if rates.is_empty() {
                        continue;
                    }
                    let r = median(&mut rates);
                    if th == threads[0] {
                        base = r / th as f64;
                    }
                    let scale = r / (base * th as f64);
                    println!("{:<26} {:>3} {:>14.0} {:>8.0}% {:>11.1} {:>11.1}", label, th, r, scale * 100.0, p50, p99);
                    per_service.push(json!({"service": label, "threads": th, "decisions_per_s": r, "scaling_efficiency_vs_ideal": scale, "p50_us": p50, "p99_us": p99, "note": "route+feedback = 2 JSON-lines requests over loopback TCP; clients share the same CPU", "rates_all_reps": rates}));
                }
                if !any {
                    println!("(serve-tcp skipped: neural-mesh binary not found next to this exe; build with `cargo build --release --bin neural-mesh`)");
                    break;
                }
            }
        }
        if !no_serve && matches!(spec, WorldSpec::Sim(_)) {
            for (label, fe) in [("serve-tcp pipelined (flush 64/1s)", None), ("serve-tcp pipelined (flush off)", Some(1_000_000usize))] {
                let mut base = 0.0;
                for &th in &threads {
                    let bg = quiet_bg();
                    let mut rates = vec![];
                    for _ in 0..reps.min(2) {
                        if let Some(o) = serve_pipe(&w, th, secs.max(2.0), fe) {
                            rates.push(o.decisions as f64 / o.secs);
                        }
                    }
                    if rates.is_empty() {
                        continue;
                    }
                    let r = median(&mut rates);
                    if th == threads[0] {
                        base = r / th as f64;
                    }
                    let scale = r / (base * th as f64);
                    println!("{:<26} {:>3} {:>14.0} {:>8.0}% {:>11} {:>11}   bg cpu {:>3.0}%", label, th, r, scale * 100.0, "-", "-", bg);
                    per_service.push(json!({"bg_cpu_pct_before": bg, "service": label, "threads": th, "decisions_per_s": r, "scaling_efficiency_vs_ideal": scale, "note": "synthetic feedback agent, 32 route + 32 feedback lines per write: server-capacity probe", "rates_all_reps": rates}));
                }
            }
        }
        results.push(json!({"world": spec.label(), "agents": w.n, "tasks": w.env.len(), "points": per_service}));
    }

    // ---- what limits the shipped server?
    let mut diag = json!({});
    if !no_serve {
        let w = make_work(&WorldSpec::Sim(Scenario::Fixed), 31);
        println!("
=== serve diagnosis ===");
        diag = write_path_diag();
        println!("{}", serde_json::to_string_pretty(&diag).unwrap());
        let mut so = vec![];
        for _ in 0..3 {
            if let Some((dps, us)) = serve_stdio(&w, 20_000) {
                so.push((dps, us));
            }
        }
        if !so.is_empty() {
            so.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (dps, us) = so[so.len() / 2];
            println!("serve --stdio (pipe, no sockets, 1 stream, global mutex): {dps:.0} decisions/s  = {us:.1} us per request (median of {})", so.len());
            diag["stdio_decisions_per_s"] = json!(dps);
            diag["stdio_us_per_request"] = json!(us);
        }
    }
    let out = json!({
        "serve_diagnosis": diag,
        "tool": "f3_throughput", "logical_cores": n_threads(), "threads": threads, "secs_per_point": secs, "reps": reps,
        "note": "decision = route + feedback (2 calls); numbers are wall-clock on a SHARED machine (other agents running); medians of reps",
        "results": results,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nwrote {out_path}");
}

/// Background CPU just before a measurement; if the box is busy, wait up to 20 s for it to calm down.
fn quiet_bg() -> f64 {
    let b = cpu_busy_pct(150);
    if b > 40.0 { wait_quiet(20, 40.0) } else { b }
}
