//! F2(c): compute-budget scheduler. Easy tasks take a cheap path (keyword rule or a memoised
//! best-known agent backed by cheap counts), hard/uncertain tasks take the full learned router
//! (Warm<HebbianRouter>, the repo's reference learner). Quality-vs-cost frontier.
//!
//! "Cost" is reported in two honest ways: (1) deterministic COUNTS of full-router selects and
//! full-router updates (fraction of tasks), (2) nanoseconds, from unit costs micro-benchmarked on
//! this machine and from timed end-to-end runs. In this crate both paths cost microseconds, so the
//! absolute saving is tiny; it only matters if the full path is expensive (that is an ASSUMPTION
//! about a deployment, shown as a modelled column, not a measurement).
//!
//! Protocol: operating points are chosen on tune seeds 1-8 only (rule below), then frozen;
//! held-out 31-60, confirm 61-90, plus a 91-250 power extension; paired 95% t-CI against the
//! always-full router, TOST equivalence margin 0.003, Holm; random-gate control at matched
//! full-router fraction; 4 worlds (3 sim scenarios + my 32-agent world).
//!
//! Usage: cargo run --release --bin f2_sched [-- --out results/f2_sched.json]

#[path = "f2_common/mod.rs"]
mod common;

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use common::*;
use neural_mesh::baselines::KeywordRouter;
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, KeywordPrior, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::holm_adjust;
use serde_json::{json, Value};

fn beta_bounds(a: f64, b: f64, z: f64) -> (f64, f64) {
    let n = a + b;
    let m = a / n;
    let sd = (m * (1.0 - m) / (n + 1.0)).sqrt();
    (m - z * sd, m + z * sd)
}

/// Cheap per-(feature, agent) discounted success/failure counts, pooled over the task's features.
#[derive(Clone, Default)]
struct Cheap {
    counts: HashMap<String, Vec<(f64, f64)>>,
    n_agents: usize,
    discount: f64,
}
impl Cheap {
    fn new(n_agents: usize, discount: f64) -> Self {
        Cheap { counts: HashMap::new(), n_agents, discount }
    }
    fn pooled(&self, task: &Task, a: usize) -> (f64, f64) {
        if task.features.is_empty() {
            return (0.0, 0.0);
        }
        let (mut s, mut f) = (0.0, 0.0);
        for k in &task.features {
            if let Some(&(cs, cf)) = self.counts.get(k).and_then(|r| r.get(a)) {
                s += cs;
                f += cf;
            }
        }
        let n = task.features.len() as f64;
        (s / n, f / n)
    }
    fn update(&mut self, task: &Task, a: usize, r: f64) {
        let r = r.clamp(0.0, 1.0);
        let n = self.n_agents;
        for k in &task.features {
            if !self.counts.contains_key(k) {
                self.counts.insert(k.clone(), vec![(0.0, 0.0); n]);
            }
            let row = self.counts.get_mut(k).unwrap();
            if self.discount < 1.0 {
                for c in row.iter_mut() {
                    c.0 *= self.discount;
                    c.1 *= self.discount;
                }
            }
            row[a].0 += r;
            row[a].1 += 1.0 - r;
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// cheap path = the keyword rule's agent, taken while its lower confidence bound >= theta
    Kw,
    /// cheap path = best pooled-mean agent of the cheap table (n >= min_n), taken while its LCB >= theta
    Cache,
}
#[derive(Clone, Copy, Debug, PartialEq)]
enum Learn {
    /// the full router is updated on every task
    Always,
    /// only on tasks that went through the full path
    Routed,
}
#[derive(Clone, Copy, Debug)]
struct SchedCfg {
    mode: Mode,
    theta: f64,
    learn: Learn,
    z: f64,
    kw_a: f64,
    kw_b: f64,
    min_n: f64,
    discount: f64,
}
impl SchedCfg {
    fn label(&self) -> String {
        format!("{}@{:.2}/{}", if self.mode == Mode::Kw { "kw-gate" } else { "cache-gate" }, self.theta, if self.learn == Learn::Always { "learn-all" } else { "learn-routed" })
    }
}

struct Sched {
    full: Warm<HebbianRouter>,
    prior: KeywordPrior,
    cheap: Cheap,
    cfg: SchedCfg,
    last_full: bool,
    n_sel_full: u64,
    n_sel_cheap: u64,
    n_upd_full: u64,
    n_tasks: u64,
}

fn new_full(n: usize, kw: &HashMap<String, usize>) -> Warm<HebbianRouter> {
    Warm::new(HebbianRouter::new(n, heb_robust_cfg()), kw.clone(), n, 1.0).with_evidence(Evidence::MeanFeature)
}

impl Sched {
    fn new(n: usize, kw: &HashMap<String, usize>, cfg: SchedCfg) -> Self {
        Sched {
            full: new_full(n, kw),
            prior: KeywordPrior::new(kw.clone(), n),
            cheap: Cheap::new(n, cfg.discount),
            cfg,
            last_full: true,
            n_sel_full: 0,
            n_sel_cheap: 0,
            n_upd_full: 0,
            n_tasks: 0,
        }
    }

    /// Some(agent) when the cheap path is confident enough.
    fn gate(&self, task: &Task, cands: &[usize]) -> Option<usize> {
        match self.cfg.mode {
            Mode::Kw => {
                let votes = self.prior.votes(task, cands);
                let k = votes.iter().fold(None, |b: Option<(usize, usize)>, &(a, v)| match b {
                    Some((_, bv)) if bv >= v => b,
                    _ => Some((a, v)),
                })?;
                let (s, f) = self.cheap.pooled(task, k.0);
                let (lcb, _) = beta_bounds(self.cfg.kw_a + s, self.cfg.kw_b + f, self.cfg.z);
                if lcb >= self.cfg.theta { Some(k.0) } else { None }
            }
            Mode::Cache => {
                let mut best: Option<(f64, f64, usize)> = None; // (mean, lcb, agent)
                for &a in cands {
                    let (s, f) = self.cheap.pooled(task, a);
                    if s + f < self.cfg.min_n {
                        continue;
                    }
                    let (lcb, _) = beta_bounds(1.0 + s, 1.0 + f, self.cfg.z);
                    let m = (1.0 + s) / (2.0 + s + f);
                    if best.is_none_or(|b| m > b.0) {
                        best = Some((m, lcb, a));
                    }
                }
                match best {
                    Some((_, lcb, a)) if lcb >= self.cfg.theta => Some(a),
                    _ => None,
                }
            }
        }
    }
}

impl Router for Sched {
    fn name(&self) -> String {
        format!("sched[{}]", self.cfg.label())
    }
    fn select(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> usize {
        self.n_tasks += 1;
        match self.gate(task, cands) {
            Some(a) => {
                self.last_full = false;
                self.n_sel_cheap += 1;
                a
            }
            None => {
                self.last_full = true;
                self.n_sel_full += 1;
                self.full.select(task, cands, rng)
            }
        }
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.cheap.update(task, agent, reward);
        if self.cfg.learn == Learn::Always || self.last_full {
            self.n_upd_full += 1;
            self.full.update(task, agent, reward);
        }
    }
}

/// Control: same cheap path (keyword) and learning rule, but the full/cheap choice is a coin flip.
struct RandGate {
    full: Warm<HebbianRouter>,
    kw: KeywordRouter,
    q: f64,
    learn: Learn,
    last_full: bool,
    coin: Rng,
    n_sel_full: u64,
    n_upd_full: u64,
    n_tasks: u64,
}
impl Router for RandGate {
    fn name(&self) -> String {
        format!("randgate(q={:.2})", self.q)
    }
    fn select(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> usize {
        self.n_tasks += 1;
        self.last_full = self.coin.f64() < self.q;
        if self.last_full {
            self.n_sel_full += 1;
            self.full.select(task, cands, rng)
        } else {
            self.kw.select(task, cands, rng)
        }
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        if self.learn == Learn::Always || self.last_full {
            self.n_upd_full += 1;
            self.full.update(task, agent, reward);
        }
    }
}

#[derive(Clone, Debug)]
enum Kind {
    Full,
    Keyword,
    Guarded,
    Sched(SchedCfg2),
    Rand(f64, bool),
}
// SchedCfg is Copy but not Debug-comparable in enum; wrap for Debug
#[derive(Clone, Copy, Debug)]
struct SchedCfg2(SchedCfg);

#[derive(Clone, Debug, Default)]
struct Out {
    full_q: f64,
    h500: f64,
    frac_sel_full: f64,
    frac_upd_full: f64,
}

fn run_kind(kind: &Kind, spec: &WorldSpec, seed: u64) -> Out {
    let env = spec.make(seed);
    let n = env.n_agents();
    let kw = env.kw_map();
    match kind {
        Kind::Full => {
            let mut p = new_full(n, &kw);
            let st = run_policy(&env, &mut p, seed);
            Out { full_q: st.full, h500: st.h500, frac_sel_full: 1.0, frac_upd_full: 1.0 }
        }
        Kind::Keyword => {
            let mut p = KeywordRouter::new(kw, n);
            let st = run_policy(&env, &mut p, seed);
            Out { full_q: st.full, h500: st.h500, frac_sel_full: 0.0, frac_upd_full: 0.0 }
        }
        Kind::Guarded => {
            let g = GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true };
            let mut p = Guarded::new(HebbianRouter::new(n, heb_robust_cfg()), kw, n, g);
            let st = run_policy(&env, &mut p, seed);
            // guarded: inner router is consulted on decisions[2], trained on every outcome
            let tot = (p.decisions[0] + p.decisions[1] + p.decisions[2]).max(1) as f64;
            Out { full_q: st.full, h500: st.h500, frac_sel_full: p.decisions[2] as f64 / tot, frac_upd_full: 1.0 }
        }
        Kind::Sched(c) => {
            let mut p = Sched::new(n, &kw, c.0);
            let st = run_policy(&env, &mut p, seed);
            let t = p.n_tasks.max(1) as f64;
            Out { full_q: st.full, h500: st.h500, frac_sel_full: p.n_sel_full as f64 / t, frac_upd_full: p.n_upd_full as f64 / t }
        }
        Kind::Rand(q, routed) => {
            let mut p = RandGate {
                full: new_full(n, &kw),
                kw: KeywordRouter::new(kw, n),
                q: *q,
                learn: if *routed { Learn::Routed } else { Learn::Always },
                last_full: true,
                coin: Rng::new(seed ^ 0xC01),
                n_sel_full: 0,
                n_upd_full: 0,
                n_tasks: 0,
            };
            let st = run_policy(&env, &mut p, seed);
            let t = p.n_tasks.max(1) as f64;
            Out { full_q: st.full, h500: st.h500, frac_sel_full: p.n_sel_full as f64 / t, frac_upd_full: p.n_upd_full as f64 / t }
        }
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn runs(kind: &Kind, spec: &WorldSpec, sd: &[u64]) -> Vec<Out> {
    par_map(sd, |&s| run_kind(kind, spec, s))
}

// =============================================================================================
// Unit costs (ns), micro-benchmarked single-threaded

#[derive(Clone, Debug)]
struct Units {
    full_sel: f64,
    full_upd: f64,
    gate_kw: f64,
    gate_cache: f64,
    kw_route: f64,
    cheap_upd: f64,
    coin: f64,
}

fn bench_units() -> Units {
    let env = Env::sim(31, neural_mesh::sim::Scenario::Random);
    let n = env.n_agents();
    let kw = env.kw_map();
    let cands: Vec<usize> = (0..n).collect();
    let tasks = env.tasks_vec();
    // train a state first
    let cfg_kw = SchedCfg { mode: Mode::Kw, theta: 0.5, learn: Learn::Always, z: 0.5, kw_a: 2.0, kw_b: 1.0, min_n: 4.0, discount: 0.97 };
    let mut s = Sched::new(n, &kw, cfg_kw);
    let mut rng = Rng::new(1);
    for (t, task) in tasks.iter().enumerate() {
        let a = s.select(task, &cands, &mut rng);
        s.update(task, a, env.outcome(t, a));
    }
    let reps = 9;
    let per = |f: &mut dyn FnMut()| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            f();
            best = best.min(t.elapsed().as_nanos() as f64 / tasks.len() as f64);
        }
        best
    };
    let mut acc = 0usize;
    let full_sel = per(&mut || {
        for task in &tasks {
            acc += s.full.select(task, &cands, &mut rng);
        }
    });
    let full_upd = per(&mut || {
        for (i, task) in tasks.iter().enumerate() {
            s.full.update(task, i % n, 1.0);
        }
    });
    let gate_kw = per(&mut || {
        for task in &tasks {
            acc += s.gate(task, &cands).unwrap_or(0);
        }
    });
    let mut sc = Sched::new(n, &kw, SchedCfg { mode: Mode::Cache, ..cfg_kw });
    sc.cheap = s.cheap.clone();
    let gate_cache = per(&mut || {
        for task in &tasks {
            acc += sc.gate(task, &cands).unwrap_or(0);
        }
    });
    let mut kr = KeywordRouter::new(kw.clone(), n);
    let kw_route = per(&mut || {
        for task in &tasks {
            acc += kr.select(task, &cands, &mut rng);
        }
    });
    let cheap_upd = per(&mut || {
        for (i, task) in tasks.iter().enumerate() {
            s.cheap.update(task, i % n, 1.0);
        }
    });
    let mut coin = Rng::new(9);
    let coin_ns = per(&mut || {
        for _ in &tasks {
            acc += (coin.f64() < 0.5) as usize;
        }
    });
    std::hint::black_box(acc);
    Units { full_sel, full_upd, gate_kw, gate_cache, kw_route, cheap_upd, coin: coin_ns }
}

/// Estimated ns/task of a policy from its path fractions and the unit costs.
fn est_ns(kind: &Kind, o: &Out, u: &Units) -> f64 {
    match kind {
        Kind::Full => u.full_sel + u.full_upd,
        Kind::Keyword => u.kw_route,
        Kind::Guarded => f64::NAN, // measured end to end instead
        Kind::Sched(c) => {
            let gate = if c.0.mode == Mode::Kw { u.gate_kw } else { u.gate_cache };
            gate + u.cheap_upd + o.frac_sel_full * u.full_sel + o.frac_upd_full * u.full_upd
        }
        Kind::Rand(..) => u.coin + (1.0 - o.frac_sel_full) * u.kw_route + o.frac_sel_full * u.full_sel + o.frac_upd_full * u.full_upd,
    }
}

/// Measured end-to-end ns per task (route + update), single thread, min over reps and averaged over seeds.
fn measured_ns(kind: &Kind, spec: &WorldSpec, sd: &[u64]) -> f64 {
    let mut per_seed = Vec::new();
    for &s in sd {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let _ = run_kind(kind, spec, s);
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        per_seed.push(best);
    }
    // includes world construction: subtract it
    let mut wc = Vec::new();
    for &s in sd {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let e = spec.make(s);
            std::hint::black_box(e.len());
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        wc.push(best);
    }
    let len = spec.make(sd[0]).len() as f64;
    mean(&per_seed.iter().zip(&wc).map(|(a, b)| (a - b) / len).collect::<Vec<_>>())
}

// =============================================================================================


fn percentile(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * p).round() as usize]
}

/// Per-task (select + update) latency in ns for a policy on one world/seed (min over reps per task).
fn latency_profile(kind: &Kind, spec: &WorldSpec, seed: u64) -> (f64, f64, f64, f64) {
    let reps = 5;
    let env0 = spec.make(seed);
    let len = env0.len();
    let mut best = vec![f64::MAX; len];
    for _ in 0..reps {
        let env = spec.make(seed);
        let n = env.n_agents();
        let kw = env.kw_map();
        let cands: Vec<usize> = (0..n).collect();
        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
        macro_rules! lp {
            ($p:expr) => {{
                let mut p = $p;
                for t in 0..len {
                    let task = env.task(t);
                    let t0 = Instant::now();
                    let a = p.select(task, &cands, &mut rng);
                    let r = env.outcome(t, a);
                    p.update(task, a, r);
                    let d = t0.elapsed().as_nanos() as f64;
                    if d < best[t] {
                        best[t] = d;
                    }
                }
            }};
        }
        match kind {
            Kind::Full => lp!(new_full(n, &kw)),
            Kind::Keyword => lp!(KeywordRouter::new(kw, n)),
            Kind::Guarded => {
                let g = GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true };
                lp!(Guarded::new(HebbianRouter::new(n, heb_robust_cfg()), kw, n, g))
            }
            Kind::Sched(c) => lp!(Sched::new(n, &kw, c.0)),
            Kind::Rand(q, routed) => lp!(RandGate {
                full: new_full(n, &kw),
                kw: KeywordRouter::new(kw, n),
                q: *q,
                learn: if *routed { Learn::Routed } else { Learn::Always },
                last_full: true,
                coin: Rng::new(seed ^ 0xC01),
                n_sel_full: 0,
                n_upd_full: 0,
                n_tasks: 0,
            }),
        }
    }
    (mean(&best), percentile(&best, 0.5), percentile(&best, 0.9), percentile(&best, 0.99))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned();
    let t0 = Instant::now();
    let worlds = all_worlds();

    println!("cores available: {}", n_threads());

    // ---- frontier configs -------------------------------------------------------------------
    let thetas = [0.35f64, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8];
    let mut cfgs: Vec<SchedCfg> = Vec::new();
    for &mode in &[Mode::Kw, Mode::Cache] {
        for &learn in &[Learn::Always, Learn::Routed] {
            for &theta in &thetas {
                cfgs.push(SchedCfg { mode, theta, learn, z: 0.5, kw_a: 2.0, kw_b: 1.0, min_n: 4.0, discount: 0.97 });
            }
        }
    }
    let sched_kinds: Vec<Kind> = cfgs.iter().map(|c| Kind::Sched(SchedCfg2(*c))).collect();

    // ---- tune on seeds 1-8: choose two operating points --------------------------------------
    let tune = tune_seeds();
    let full_tune: Vec<Vec<Out>> = worlds.iter().map(|w| runs(&Kind::Full, w, &tune)).collect();
    println!("\n--- tuning on seeds 1-8 ---");
    let mut tune_rows = Vec::new();
    let mut summary: Vec<(usize, f64, f64)> = Vec::new(); // (cfg idx, worst-world dQ, mean frac_sel_full)
    for (ci, k) in sched_kinds.iter().enumerate() {
        let mut dqs = Vec::new();
        let mut fr = Vec::new();
        for (wi, w) in worlds.iter().enumerate() {
            let o = runs(k, w, &tune);
            let a: Vec<f64> = o.iter().map(|x| x.full_q).collect();
            let b: Vec<f64> = full_tune[wi].iter().map(|x| x.full_q).collect();
            dqs.push(mean(&a) - mean(&b));
            fr.push(mean(&o.iter().map(|x| x.frac_sel_full).collect::<Vec<_>>()));
        }
        let worst = dqs.iter().cloned().fold(f64::MAX, f64::min);
        let mf = mean(&fr);
        summary.push((ci, worst, mf));
        tune_rows.push(json!({"config": cfgs[ci].label(), "worst_world_dq": worst, "mean_frac_full_select": mf, "dq_by_world": dqs}));
        println!("  {:<30} worst-world dQ {:+.4}  mean full-select fraction {:.3}", cfgs[ci].label(), worst, mf);
    }
    let pick = |tol: f64| -> Option<usize> {
        summary
            .iter()
            .filter(|(_, w, _)| *w >= -tol)
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .map(|x| x.0)
    };
    let op_a = pick(0.003);
    let op_c = pick(0.006);
    let op_b = pick(0.010);
    println!("  OP-A (strict, tune worst-world dQ >= -0.003): {:?}", op_a.map(|i| cfgs[i].label()));
    println!("  OP-C (mid,    tune worst-world dQ >= -0.006): {:?}", op_c.map(|i| cfgs[i].label()));
    println!("  OP-B (loose,  tune worst-world dQ >= -0.010): {:?}", op_b.map(|i| cfgs[i].label()));
    // matched random-gate control fractions (tune seeds)
    let frac_of = |i: usize| summary[i].2;

    // ---- frozen evaluation -------------------------------------------------------------------
    let mut named: Vec<(String, Kind)> = vec![("keyword".into(), Kind::Keyword), ("guarded(x1)".into(), Kind::Guarded)];
    let mut ops: Vec<(String, usize)> = Vec::new();
    if let Some(i) = op_a {
        ops.push(("OP-A".into(), i));
    }
    if let Some(i) = op_c {
        ops.push(("OP-C".into(), i));
    }
    if let Some(i) = op_b {
        ops.push(("OP-B".into(), i));
    }
    for (nm, i) in &ops {
        named.push((format!("{nm} {}", cfgs[*i].label()), sched_kinds[*i].clone()));
        let routed = cfgs[*i].learn == Learn::Routed;
        named.push((format!("{nm}-control randgate q={:.2}", frac_of(*i)), Kind::Rand(frac_of(*i), routed)));
    }

    let units = bench_units();
    println!("\nunit costs (ns, single thread, min of 9): {:?}", units);

    let held = held_seeds();
    let confirm = confirm_seeds();
    let ext = seeds(91, 250);
    let mut doc_splits = Vec::new();
    for (split, sd) in [("held-out 31-60", &held), ("confirm 61-90 (frozen)", &confirm), ("power extension 91-250 (frozen)", &ext)] {
        println!("\n=== F2(c) {split}: quality vs always-full router (Warm<Hebbian>), n={} ===", sd.len());
        let mut cells: Vec<(String, String, Cmp, Out, f64)> = Vec::new(); // (policy, world, cmp, mean out, est ns)
        let mut diff_ctrl: Vec<(String, String, Cmp)> = Vec::new();
        for (wi, w) in worlds.iter().enumerate() {
            let full = runs(&Kind::Full, w, sd);
            let fq: Vec<f64> = full.iter().map(|o| o.full_q).collect();
            let mut by_name: HashMap<String, Vec<Out>> = HashMap::new();
            for (nm, k) in &named {
                let o = runs(k, w, sd);
                by_name.insert(nm.clone(), o.clone());
                let q: Vec<f64> = o.iter().map(|x| x.full_q).collect();
                let mo = Out {
                    full_q: mean(&q),
                    h500: mean(&o.iter().map(|x| x.h500).collect::<Vec<_>>()),
                    frac_sel_full: mean(&o.iter().map(|x| x.frac_sel_full).collect::<Vec<_>>()),
                    frac_upd_full: mean(&o.iter().map(|x| x.frac_upd_full).collect::<Vec<_>>()),
                };
                let e = est_ns(k, &mo, &units);
                cells.push((nm.clone(), w.label(), compare(&q, &fq), mo, e));
            }
            // OP vs matched random-gate control (superiority; two-sided paired t)
            for (nm, i) in &ops {
                let a = format!("{nm} {}", cfgs[*i].label());
                let c = format!("{nm}-control randgate q={:.2}", frac_of(*i));
                let qa: Vec<f64> = by_name[&a].iter().map(|x| x.full_q).collect();
                let qc: Vec<f64> = by_name[&c].iter().map(|x| x.full_q).collect();
                diff_ctrl.push((format!("{nm} minus matched random gate"), w.label(), compare(&qa, &qc)));
                let qg: Vec<f64> = by_name["guarded(x1)"].iter().map(|x| x.full_q).collect();
                diff_ctrl.push((format!("{nm} minus guarded(x1)"), w.label(), compare(&qa, &qg)));
            }
            let _ = wi;
        }
        let ps: Vec<f64> = cells.iter().map(|c| c.2.p_equiv).collect();
        let holm = holm_adjust(&ps);
        let pc: Vec<f64> = diff_ctrl.iter().map(|c| c.2.p_diff).collect();
        let holm_c = holm_adjust(&pc);
        println!("{:<44} {:<14} {:>8} {:>8} {:>9} {:>23} {:>6} {:>5}", "policy", "world", "fullSel", "fullUpd", "estNs", "dQ vs full [95% CI]", "Holm", "LL");
        let mut js = Vec::new();
        for (c, h) in cells.iter().zip(&holm) {
            let ll = *h < 0.05 && c.2.lo > -MARGIN && c.2.hi < MARGIN;
            println!(
                "{:<44} {:<14} {:>8.3} {:>8.3} {:>9.0} {:>23} {:>6.3} {:>5}",
                c.0, c.1, c.3.frac_sel_full, c.3.frac_upd_full, c.4, fmt_cmp(&c.2), h, if ll { "YES" } else { "no" }
            );
            let mut j = cmp_json(&c.2, Some(*h));
            j["policy"] = json!(c.0);
            j["world"] = json!(c.1);
            j["frac_full_select"] = json!(c.3.frac_sel_full);
            j["frac_full_update"] = json!(c.3.frac_upd_full);
            j["est_ns_per_task"] = json!(c.4);
            j["mean_quality"] = json!(c.3.full_q);
            js.push(j);
        }
        println!("-- OP minus matched random gate (paired, two-sided) --");
        let mut jc = Vec::new();
        for (c, h) in diff_ctrl.iter().zip(&holm_c) {
            println!("  {:<34} {:<14} {}   Holm p {:.4}", c.0, c.1, fmt_cmp(&c.2), h);
            let mut j = cmp_json(&c.2, None);
            j["comparison"] = json!(c.0);
            j["world"] = json!(c.1);
            j["holm_p_diff"] = json!(h);
            jc.push(j);
        }
        // whole frontier (held-out only, for display; not used for selection)
        let mut frontier = Vec::new();
        if split.starts_with("held") {
            println!("-- full frontier on held-out (display only; operating points were fixed on seeds 1-8) --");
            for (ci, k) in sched_kinds.iter().enumerate() {
                let mut row = Vec::new();
                let mut line = format!("  {:<30}", cfgs[ci].label());
                for (_wi, w) in worlds.iter().enumerate() {
                    let full = runs(&Kind::Full, w, sd);
                    let o = runs(k, w, sd);
                    let c = compare(&o.iter().map(|x| x.full_q).collect::<Vec<_>>(), &full.iter().map(|x| x.full_q).collect::<Vec<_>>());
                    let fr = mean(&o.iter().map(|x| x.frac_sel_full).collect::<Vec<_>>());
                    line += &format!(" | {} fs={:.2}", fmt_cmp(&c), fr);
                    row.push(json!({"world": w.label(), "frac_full_select": fr,
                        "frac_full_update": mean(&o.iter().map(|x| x.frac_upd_full).collect::<Vec<_>>()),
                        "dq": c.mean, "ci_lo": c.lo, "ci_hi": c.hi}));
                }
                println!("{line}");
                frontier.push(json!({"config": cfgs[ci].label(), "worlds": row}));
            }
        }
        doc_splits.push(json!({"split": split, "n_seeds": sd.len(), "cells": js, "op_vs_random_gate": jc, "frontier": frontier}));
    }

    // ---- measured end-to-end ns and latency percentiles (sim-random world, held-out seeds) ---
    println!("\n--- measured end-to-end ns/task and per-task (select+update) latency, world sim-random ---");
    let wspec = WorldSpec::Sim(neural_mesh::sim::Scenario::Random);
    let lat_seeds: Vec<u64> = (31..=36).collect();
    let mut timing = Vec::new();
    let mut kinds: Vec<(String, Kind)> = vec![("always-full".into(), Kind::Full)];
    kinds.extend(named.iter().cloned());
    for (nm, k) in &kinds {
        let m = measured_ns(k, &wspec, &lat_seeds);
        let (mean_l, p50, p90, p99) = latency_profile(k, &wspec, 31);
        println!("  {:<46} end-to-end {:>8.0} ns/task   latency mean {:>7.0} p50 {:>7.0} p90 {:>7.0} p99 {:>7.0} ns", nm, m, mean_l, p50, p90, p99);
        timing.push(json!({"policy": nm, "measured_ns_per_task": m, "latency_ns": {"mean": mean_l, "p50": p50, "p90": p90, "p99": p99}}));
    }

    // modelled column: if the full path costs R x the cheap path (ASSUMPTION, not measured)
    let mut model = Vec::new();
    for (ops_nm, i) in &ops {
        let fs = frac_of(*i);
        for r in [1.0f64, 10.0, 100.0, 1000.0] {
            let cost = fs * r + (1.0 - fs);
            model.push(json!({"op": ops_nm, "assumed_full_to_cheap_cost_ratio": r, "relative_cost_vs_always_full": cost / r,
                "note": "tune-seed full-select fraction only; assumes cheap-path+gate cost = 1 unit and ignores update cost"}));
        }
    }

    let doc = json!({
        "part": "F2(c) compute-budget scheduler",
        "protocol": {
            "tune": "1-8: operating points chosen (OP-A: min mean full-select fraction s.t. worst-world dQ >= -0.003; OP-B: >= -0.010), then frozen",
            "held_out": "31-60", "confirm": "61-90 frozen", "power_extension": "91-250 frozen",
            "reference": "always-full router = Warm(k=1, MeanFeature) over HebbianRouter(hebbian-robust)",
            "equivalence": "TOST margin 0.003 + Holm over (policy x world) per split; lossless_claim = Holm p<0.05 AND CI inside margin",
            "control": "random gate with the SAME full-select fraction (tune seeds), same keyword cheap path and learning rule",
            "worlds": worlds.iter().map(|w| w.label()).collect::<Vec<_>>(),
        },
        "tuning": tune_rows,
        "operating_points": ops.iter().map(|(n, i)| json!({"name": n, "config": cfgs[*i].label(), "tune_full_select_fraction": frac_of(*i)})).collect::<Vec<_>>(),
        "unit_costs_ns": {"full_select": units.full_sel, "full_update": units.full_upd, "gate_kw": units.gate_kw,
            "gate_cache": units.gate_cache, "keyword_route": units.kw_route, "cheap_update": units.cheap_upd, "coin": units.coin},
        "splits": doc_splits,
        "timing": timing,
        "modelled_cost_if_full_path_is_R_times_more_expensive": model,
        "cores": n_threads(),
        "runtime_s": t0.elapsed().as_secs_f64(),
        "honesty": [
            "Both paths cost microseconds here; ns savings are real but negligible in absolute terms. The scheduler only pays off if the full path is expensive in a deployment (assumption).",
            "Quality claims are against the always-full router in synthetic worlds written by the same author; no real traffic.",
            "The cheap path is trained from the same outcome stream, so the scheduler is not 'free learning'."
        ],
    });
    if let Some(p) = out_path {
        std::fs::write(&p, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        println!("wrote {p}");
    }
    let _ = BTreeMap::<u8, u8>::new();
    let _ = Value::Null;
    println!("runtime {:.1}s", t0.elapsed().as_secs_f64());
}
