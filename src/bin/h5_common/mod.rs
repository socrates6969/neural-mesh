//! h5 shared pieces (snapshot of g3_common worlds/stats; R4: Big worlds are parameterised by the agent count, plus an open-stream world): seeds, std-thread par_map, paired/TOST statistics, worlds (snapshot of the
//! f2 helpers, copied so concurrent f2 edits cannot break f3), counting allocator, process RSS.
//! Included from the h5_* bins with `#[path = "h5_common/mod.rs"] mod common;`.
#![allow(dead_code)]

use std::collections::HashMap;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;


use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{paired_diff, t_crit_95, t_two_sided_p, PairedDiff};

pub fn seeds(lo: u64, hi: u64) -> Vec<u64> {
    (lo..=hi).collect()
}
pub fn tune_seeds() -> Vec<u64> {
    seeds(1, 8)
}
pub fn held_seeds() -> Vec<u64> {
    seeds(31, 60)
}
pub fn confirm_seeds() -> Vec<u64> {
    seeds(61, 90)
}

/// Equivalence margin on expected success (absolute): "lossless" needs the CI inside +-MARGIN.
pub const MARGIN: f64 = 0.003;

/// Logical cores of the machine (reported in results).
pub fn n_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Worker threads for the (short, deterministic) accuracy sweeps: capped because the box is shared.
pub fn n_workers() -> usize {
    std::env::var("H5_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or_else(|| n_threads().min(6)).max(1)
}

/// Dynamic-schedule parallel map with std threads (order-preserving).
pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n = items.len();
    let nt = n_workers().min(n).max(1);
    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<Option<R>>> = (0..n).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..nt {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let r = f(&items[i]);
                    *out[i].lock().unwrap() = Some(r);
                }
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

// =============================================================================================
// Statistics helpers

/// One-sided-pair TOST p-value for |mean diff| < margin (paired differences `d`).
pub fn tost_p(d: &[f64], margin: f64) -> f64 {
    let n = d.len();
    if n < 2 {
        return 1.0;
    }
    let m = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    if se < 1e-15 {
        return if m.abs() < margin { 0.0 } else { 1.0 };
    }
    let df = n - 1;
    let one_sided_upper = |t: f64| {
        let p2 = t_two_sided_p(t.abs(), df);
        if t >= 0.0 { p2 / 2.0 } else { 1.0 - p2 / 2.0 }
    };
    let p1 = one_sided_upper((m + margin) / se); // H0: diff <= -margin
    let p2 = one_sided_upper(-(m - margin) / se); // H0: diff >= +margin
    p1.max(p2)
}

#[derive(Clone, Debug)]
pub struct Cmp {
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
    pub p_equiv: f64,
    pub p_diff: f64,
    pub wins: usize,
    pub losses: usize,
    pub n: usize,
}

pub fn compare(a: &[f64], b: &[f64]) -> Cmp {
    let pd: PairedDiff = paired_diff(a, b);
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    Cmp {
        mean: pd.diff.mean,
        lo: pd.diff.lo,
        hi: pd.diff.hi,
        p_equiv: tost_p(&d, MARGIN),
        p_diff: neural_mesh::stats::paired_t_pvalue(a, b),
        wins: pd.wins,
        losses: pd.losses,
        n: d.len(),
    }
}

pub fn cmp_json(c: &Cmp, holm_equiv: Option<f64>) -> serde_json::Value {
    serde_json::json!({
        "diff": c.mean, "ci_lo": c.lo, "ci_hi": c.hi, "n": c.n, "wins": c.wins, "losses": c.losses,
        "p_diff": c.p_diff, "p_equiv_tost": c.p_equiv, "holm_p_equiv": holm_equiv,
        "ci_inside_margin": c.lo > -MARGIN && c.hi < MARGIN,
        "lossless_claim": holm_equiv.map(|h| h < 0.05 && c.lo > -MARGIN && c.hi < MARGIN).unwrap_or(false),
    })
}

pub fn fmt_cmp(c: &Cmp) -> String {
    format!("{:+.4} [{:+.4},{:+.4}]", c.mean, c.lo, c.hi)
}

pub fn ci_half(xs: &[f64]) -> (f64, f64, f64) {
    let n = xs.len();
    let m = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n.max(2) - 1) as f64;
    let hw = t_crit_95(n.max(2) - 1) * (var / n as f64).sqrt();
    (m, m - hw, m + hw)
}

// =============================================================================================
// Worlds

pub const BIG_AGENTS: usize = 32;
pub const BIG_TYPES: usize = 48;
pub const BIG_TASKS: usize = 4000;
const BIG_NOISE: usize = 40;
pub const TAIL_VOCAB: usize = 3000;
pub const TAIL_PROB: f64 = 0.60;

/// My own larger world (32 agents, 48 task types x 4 keywords + 40 noise keywords, Zipf type
/// popularity, 35% trap keyword map, mid-run drift on 25% of types).
pub struct BigWorld {
    n: usize,
    tasks: Vec<Task>,
    ty: Vec<usize>,
    u: Vec<f64>,
    p_pre: Vec<Vec<f64>>,
    p_post: Vec<Vec<f64>>,
    kw: HashMap<String, usize>,
}

fn uni(r: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * r.f64()
}

impl BigWorld {
    pub fn new(seed: u64, agents: usize) -> Self {
        Self::new_with(seed, agents, BIG_NOISE, 0.30)
    }
    pub fn new_with(seed: u64, agents: usize, noise_vocab: usize, noise_prob: f64) -> Self {
        let big_agents = agents;
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xB16_B16);
        let mut p_pre = vec![vec![0.0; big_agents]; BIG_TYPES];
        let mut kwa = vec![0usize; BIG_TYPES];
        let mut best_of = vec![0usize; BIG_TYPES];
        for t in 0..BIG_TYPES {
            for a in 0..big_agents {
                p_pre[t][a] = uni(&mut r, 0.15, 0.45);
            }
            let best = r.below(big_agents);
            p_pre[t][best] = uni(&mut r, 0.80, 0.90);
            let mut second = r.below(big_agents - 1);
            if second >= best {
                second += 1;
            }
            p_pre[t][second] = uni(&mut r, 0.55, 0.68);
            best_of[t] = best;
            kwa[t] = if r.bernoulli(0.65) {
                best
            } else {
                let mut w = r.below(big_agents - 1);
                if w >= best {
                    w += 1;
                }
                w
            };
        }
        let mut p_post = p_pre.clone();
        for t in 0..BIG_TYPES {
            if r.bernoulli(0.25) {
                let old = best_of[t];
                let mut new = r.below(big_agents - 1);
                if new >= old {
                    new += 1;
                }
                p_post[t][old] = uni(&mut r, 0.30, 0.50);
                p_post[t][new] = uni(&mut r, 0.80, 0.90);
            }
        }
        // zipf popularity
        let mut cdf = Vec::with_capacity(BIG_TYPES);
        let mut acc = 0.0;
        for t in 0..BIG_TYPES {
            acc += 1.0 / (t as f64 + 1.0);
            cdf.push(acc);
        }
        let total = acc;
        let mut tasks = Vec::with_capacity(BIG_TASKS);
        let mut ty = Vec::with_capacity(BIG_TASKS);
        let mut u = Vec::with_capacity(BIG_TASKS);
        for id in 0..BIG_TASKS {
            let x = r.f64() * total;
            let t = cdf.iter().position(|&c| x < c).unwrap_or(BIG_TYPES - 1);
            let k = 2 + r.below(2);
            let mut idx = [0usize, 1, 2, 3];
            for i in (1..4).rev() {
                let j = r.below(i + 1);
                idx.swap(i, j);
            }
            let mut features: Vec<String> = idx[..k].iter().map(|&j| format!("t{t}k{j}")).collect();
            if r.bernoulli(0.25) {
                let mut o = r.below(BIG_TYPES - 1);
                if o >= t {
                    o += 1;
                }
                features.push(format!("t{o}k{}", r.below(4)));
            }
            if r.bernoulli(noise_prob) {
                features.push(format!("n{}", r.below(noise_vocab)));
            }
            for i in (1..features.len()).rev() {
                let j = r.below(i + 1);
                features.swap(i, j);
            }
            tasks.push(Task { id: id as u64, task_type: None, features });
            ty.push(t);
            u.push(r.f64());
        }
        let mut kw = HashMap::new();
        for t in 0..BIG_TYPES {
            for j in 0..4 {
                kw.insert(format!("t{t}k{j}"), kwa[t]);
            }
        }
        BigWorld { n: big_agents, tasks, ty, u, p_pre, p_post, kw }
    }
}

pub enum Env {
    Sim(World),
    Big(BigWorld),
}

impl Env {
    pub fn sim(seed: u64, sc: Scenario) -> Env {
        Env::Sim(World::new(seed, WorldConfig { n_tasks: 2000, scenario: sc, ..WorldConfig::default() }))
    }
    pub fn big_tail(seed: u64, agents: usize) -> Env {
        Env::Big(BigWorld::new_with(seed, agents, TAIL_VOCAB, TAIL_PROB))
    }
    /// open token stream: (almost) every noise token is a never-seen-before string
    pub fn big_open(seed: u64, agents: usize) -> Env {
        Env::Big(BigWorld::new_with(seed, agents, 1_000_000_000, TAIL_PROB))
    }
    pub fn big(seed: u64, agents: usize) -> Env {
        Env::Big(BigWorld::new(seed, agents))
    }
    pub fn n_agents(&self) -> usize {
        match self {
            Env::Sim(w) => w.n_agents(),
            Env::Big(b) => b.n,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Env::Sim(w) => w.len(),
            Env::Big(b) => b.tasks.len(),
        }
    }
    pub fn task(&self, t: usize) -> &Task {
        match self {
            Env::Sim(w) => &w.task(t).task,
            Env::Big(b) => &b.tasks[t],
        }
    }
    pub fn prob(&self, t: usize, a: usize) -> f64 {
        match self {
            Env::Sim(w) => w.success_prob(t, &[a]),
            Env::Big(b) => {
                if t < b.tasks.len() / 2 { b.p_pre[b.ty[t]][a] } else { b.p_post[b.ty[t]][a] }
            }
        }
    }
    pub fn outcome(&self, t: usize, a: usize) -> f64 {
        match self {
            Env::Sim(w) => w.outcome(t, &[a]),
            Env::Big(b) => {
                if b.u[t] < self.prob(t, a) { 1.0 } else { 0.0 }
            }
        }
    }
    pub fn kw_map(&self) -> HashMap<String, usize> {
        match self {
            Env::Sim(w) => w.keyword_map_for(),
            Env::Big(b) => b.kw.clone(),
        }
    }
    pub fn tasks_vec(&self) -> Vec<Task> {
        (0..self.len()).map(|t| self.task(t).clone()).collect()
    }
}

/// World specs used by every part.
#[derive(Clone, Debug)]
pub enum WorldSpec {
    Sim(Scenario),
    Big(usize),
    BigTail(usize),
    BigOpen(usize),
}
impl WorldSpec {
    pub fn label(&self) -> String {
        match self {
            WorldSpec::Sim(s) => format!("sim-{}", s.label()),
            WorldSpec::Big(n) => format!("big-{n}agents"),
            WorldSpec::BigTail(n) => format!("big-{n}agents-longtail"),
            WorldSpec::BigOpen(n) => format!("big-{n}agents-open"),
        }
    }
    pub fn make(&self, seed: u64) -> Env {
        match self {
            WorldSpec::Sim(s) => Env::sim(seed, *s),
            WorldSpec::Big(n) => Env::big(seed, *n),
            WorldSpec::BigTail(n) => Env::big_tail(seed, *n),
            WorldSpec::BigOpen(n) => Env::big_open(seed, *n),
        }
    }
}
/// the 9 h5 worlds: the 5 g3 worlds + open-stream (32) + 64 and 128 agents
pub fn all_worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec::Sim(Scenario::Fixed),
        WorldSpec::Sim(Scenario::Random),
        WorldSpec::Sim(Scenario::NoTraps),
        WorldSpec::Big(32),
        WorldSpec::BigTail(32),
        WorldSpec::BigOpen(32),
        WorldSpec::Big(64),
        WorldSpec::Big(128),
    ]
}

#[derive(Clone, Debug, Default)]
pub struct RunStats {
    pub h250: f64,
    pub h500: f64,
    pub full: f64,
}

/// Sequential online loop: select, observe the realised Bernoulli outcome, update.
pub fn run_policy<R: Router>(env: &Env, pol: &mut R, seed: u64) -> RunStats {
    let n = env.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut cum = 0.0;
    let mut out = RunStats::default();
    for t in 0..env.len() {
        let task = env.task(t);
        let a = pol.select(task, &cands, &mut rng);
        cum += env.prob(t, a);
        pol.update(task, a, env.outcome(t, a));
        if t + 1 == 250 {
            out.h250 = cum / 250.0;
        }
        if t + 1 == 500 {
            out.h500 = cum / 500.0;
        }
    }
    out.full = cum / env.len() as f64;
    out
}


// =============================================================================================
// Counting allocator (requested bytes + block counts). Counting is gated by `ON` so timing runs
// can switch it off; measurements are always taken single-threaded in `main`.

pub mod alloc_count {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering::Relaxed};

    pub struct Counting;
    pub static ON: AtomicBool = AtomicBool::new(false);
    pub static LIVE: AtomicI64 = AtomicI64::new(0);
    pub static PEAK: AtomicI64 = AtomicI64::new(0);
    pub static BLOCKS: AtomicI64 = AtomicI64::new(0);
    pub static ALLOCS: AtomicU64 = AtomicU64::new(0);
    pub static REALLOCS: AtomicU64 = AtomicU64::new(0);

    #[inline]
    fn bump(sz: i64) {
        let now = LIVE.fetch_add(sz, Relaxed) + sz;
        if sz > 0 && now > PEAK.load(Relaxed) {
            PEAK.fetch_max(now, Relaxed);
        }
    }

    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(l) };
            if !p.is_null() && ON.load(Relaxed) {
                bump(l.size() as i64);
                BLOCKS.fetch_add(1, Relaxed);
                ALLOCS.fetch_add(1, Relaxed);
            }
            p
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc_zeroed(l) };
            if !p.is_null() && ON.load(Relaxed) {
                bump(l.size() as i64);
                BLOCKS.fetch_add(1, Relaxed);
                ALLOCS.fetch_add(1, Relaxed);
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            if ON.load(Relaxed) {
                bump(-(l.size() as i64));
                BLOCKS.fetch_sub(1, Relaxed);
            }
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            let q = unsafe { System.realloc(p, l, new) };
            if !q.is_null() && ON.load(Relaxed) {
                bump(new as i64 - l.size() as i64);
                REALLOCS.fetch_add(1, Relaxed);
            }
            q
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Snap {
        pub live: i64,
        pub peak: i64,
        pub blocks: i64,
        pub allocs: u64,
        pub reallocs: u64,
    }

    pub fn start() {
        LIVE.store(0, Relaxed);
        PEAK.store(0, Relaxed);
        BLOCKS.store(0, Relaxed);
        ALLOCS.store(0, Relaxed);
        REALLOCS.store(0, Relaxed);
        ON.store(true, Relaxed);
    }
    pub fn snap() -> Snap {
        Snap {
            live: LIVE.load(Relaxed),
            peak: PEAK.load(Relaxed),
            blocks: BLOCKS.load(Relaxed),
            allocs: ALLOCS.load(Relaxed),
            reallocs: REALLOCS.load(Relaxed),
        }
    }
    pub fn stop() -> Snap {
        let s = snap();
        ON.store(false, Relaxed);
        s
    }
    /// Run `f` with counting on; returns (result, snapshot taken BEFORE `f`'s result is dropped).
    pub fn measure<T>(f: impl FnOnce() -> T) -> (T, Snap) {
        start();
        let r = f();
        let s = stop();
        (r, s)
    }
}

// =============================================================================================
// Process memory (Windows commit + working set via kernel32; elsewhere 0)

#[cfg(windows)]
pub fn proc_mem() -> (usize, usize) {
    #[repr(C)]
    struct Pmc {
        cb: u32,
        page_faults: u32,
        peak_ws: usize,
        ws: usize,
        qpp: usize,
        qp: usize,
        qpnp: usize,
        qnp: usize,
        pagefile: usize,
        peak_pagefile: usize,
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
    }
    let mut p = Pmc { cb: std::mem::size_of::<Pmc>() as u32, page_faults: 0, peak_ws: 0, ws: 0, qpp: 0, qp: 0, qpnp: 0, qnp: 0, pagefile: 0, peak_pagefile: 0 };
    let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb) };
    if ok == 0 { (0, 0) } else { (p.pagefile, p.ws) }
}
#[cfg(not(windows))]
pub fn proc_mem() -> (usize, usize) {
    (0, 0)
}

// =============================================================================================
// Machine load (other workers share this box): system-wide CPU busy % over a short window

#[cfg(windows)]
fn sys_times() -> (u64, u64, u64) {
    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct Ft {
        lo: u32,
        hi: u32,
    }
    unsafe extern "system" {
        fn GetSystemTimes(idle: *mut Ft, kernel: *mut Ft, user: *mut Ft) -> i32;
    }
    let (mut i, mut k, mut u) = (Ft::default(), Ft::default(), Ft::default());
    unsafe { GetSystemTimes(&mut i, &mut k, &mut u) };
    let c = |f: Ft| ((f.hi as u64) << 32) | f.lo as u64;
    (c(i), c(k), c(u))
}
#[cfg(not(windows))]
fn sys_times() -> (u64, u64, u64) {
    (0, 1, 0)
}

/// System-wide busy percentage over `ms` milliseconds (kernel time includes idle).
pub fn cpu_busy_pct(ms: u64) -> f64 {
    let a = sys_times();
    std::thread::sleep(std::time::Duration::from_millis(ms));
    let b = sys_times();
    let (di, dk, du) = (b.0 - a.0, b.1 - a.1, b.2 - a.2);
    let tot = (dk + du).max(1);
    100.0 * (1.0 - di as f64 / tot as f64)
}

/// Wait (up to `max_s`) until the machine is quieter than `thresh` percent; returns the last reading.
pub fn wait_quiet(max_s: u64, thresh: f64) -> f64 {
    let t0 = std::time::Instant::now();
    loop {
        let l = cpu_busy_pct(500);
        if l < thresh || t0.elapsed().as_secs() >= max_s {
            return l;
        }
    }
}

// =============================================================================================
// g3 equivalence statistics (paired by seed): TOST at an explicit margin, 90% and 95% CIs

/// Two-sided critical t value for `alpha` (bisection on the crate's t CDF).
pub fn t_crit(df: usize, alpha: f64) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 60.0f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if t_two_sided_p(mid, df.max(1)) > alpha { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

#[derive(Clone, Debug)]
pub struct EqRes {
    pub n: usize,
    pub mean: f64,
    pub sd: f64,
    pub lo95: f64,
    pub hi95: f64,
    pub lo90: f64,
    pub hi90: f64,
    pub margin: f64,
    pub p_tost: f64,
    pub p_diff: f64,
    /// one-sided non-inferiority p-value (H0: diff <= -margin)
    pub p_ni: f64,
    pub wins: usize,
    pub losses: usize,
}

/// `a - b` paired by index. TOST at +-margin (alpha .05 <=> the 90% CI inside +-margin).
pub fn eq_test(a: &[f64], b: &[f64], margin: f64) -> EqRes {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let n = d.len();
    let m = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n.max(2) - 1) as f64;
    let se = (var / n as f64).sqrt();
    let (h95, h90) = (t_crit(n - 1, 0.05) * se, t_crit(n - 1, 0.10) * se);
    EqRes {
        n,
        mean: m,
        sd: var.sqrt(),
        lo95: m - h95,
        hi95: m + h95,
        lo90: m - h90,
        hi90: m + h90,
        margin,
        p_tost: tost_p(&d, margin),
        p_ni: ni_p(&d, margin),
        p_diff: neural_mesh::stats::paired_t_pvalue(a, b),
        wins: d.iter().filter(|x| **x > 1e-12).count(),
        losses: d.iter().filter(|x| **x < -1e-12).count(),
    }
}

impl EqRes {
    pub fn ci90_inside(&self) -> bool {
        self.lo90 > -self.margin && self.hi90 < self.margin
    }
    pub fn ci95_inside(&self) -> bool {
        self.lo95 > -self.margin && self.hi95 < self.margin
    }
    pub fn json(&self, holm_p: f64) -> serde_json::Value {
        serde_json::json!({
            "n": self.n, "diff_mean": self.mean, "diff_sd": self.sd,
            "ci95": [self.lo95, self.hi95], "ci90": [self.lo90, self.hi90], "margin": self.margin,
            "p_tost": self.p_tost, "holm_p_tost": holm_p, "p_ni": self.p_ni, "p_diff": self.p_diff,
            "wins": self.wins, "losses": self.losses,
            "ci90_inside_margin": self.ci90_inside(), "ci95_inside_margin": self.ci95_inside(),
            "certified_equivalent": holm_p < 0.05 && self.ci90_inside(),
        })
    }
    pub fn fmt(&self) -> String {
        format!("{:+.4} 90%[{:+.4},{:+.4}] 95%[{:+.4},{:+.4}] p_tost={:.4}", self.mean, self.lo90, self.hi90, self.lo95, self.hi95, self.p_tost)
    }
}

/// One-sided non-inferiority p-value for diff > -margin (H0: diff <= -margin), paired differences `d`.
pub fn ni_p(d: &[f64], margin: f64) -> f64 {
    let n = d.len();
    if n < 2 {
        return 1.0;
    }
    let m = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    if se < 1e-15 {
        return if m > -margin { 0.0 } else { 1.0 };
    }
    let t = (m + margin) / se;
    let p2 = t_two_sided_p(t.abs(), n - 1);
    if t >= 0.0 { p2 / 2.0 } else { 1.0 - p2 / 2.0 }
}

impl Env {
    /// Replace the keyword map (Big worlds only; the Sim world keeps its own). Used by h5_shared: many
    /// brains over ONE prototype must share one keyword table (`fork_blank` shares it by design).
    pub fn with_kw(mut self, kw: HashMap<String, usize>) -> Env {
        if let Env::Big(b) = &mut self {
            b.kw = kw;
        }
        self
    }
}
