//! G20 / W21 (SONA-B): can SonaPlus-style routing (guarded keyword hand-over + Hebbian learner, fed the reuse signal)
//! be improved by EXPLORATION (pattern priors, Thompson over pattern posteriors, per-pattern annealed
//! uncertainty-directed exploration budget) and DECAY/FORGETTING (lazy time decay, EWC-like protection of proven
//! patterns, CUSUM drift-detection reset)?
//!
//! SYNTHETIC-until-validated-on-real-traffic. No claim beyond "paired difference of expected success in these generators".
//!
//! Worlds (fixed before any policy ran, g20_common): stationary traps(main) / correct-keywords / many-topics(16)
//! (mechanisms must not hurt), drift-abrupt / drift-gradual / drift-recurrent (decay should help; recurrent punishes
//! forgetting), sim-fixed / sim-random / sim-notraps (library sim, contain their own drift types).
//! Modes: NORMAL (realised outcome) and WEAK (hook bit now + 30% delayed graded reuse signal, x3/d7 convention).
//! Every SonaB arm (rt = all mechanisms off, ablations, full) is fed the SAME signal; rt is the ablation WITHOUT the mechanism.
//! Baselines: keyword, eps-greedy, pooled-ts, hebbian-robust, warm(hebr), guarded(x1) [+ guarded-rt]; weak worlds: raw-fed
//! baselines AND guarded/warm/rt fed the reuse signal (strict pool).
//!
//! Protocol: tune seeds 1-8 (regret-averse objective vs best of the 6 plain baselines, ONE config over ALL worlds), held-out
//! --held 31-60 (default), fresh confirm --held 61-90 with --tuned-in (frozen tuning), paired 95% t-CI, Holm per family.
//!
//! Usage: g20_sona_b [--out results/g19b_main.json] [--held 31-60] [--tuned-out FILE | --tuned-in FILE] [--quick] [--selftest]

#[path = "g20_common/mod.rs"]
mod common;

use std::collections::{HashMap, VecDeque};

use common::*;
use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const HORIZONS: [usize; 5] = [100, 250, 500, 1000, 2000];

// ------------------------------------------------------------------ weak observation model (x3 WeakCfg / d7)
const TOOL_BASE: f64 = 0.01;
const TOOL_SLOPE: f64 = 0.03;
const REUSE_FRAC: f64 = 0.3;
const REUSE_DELAY: usize = 25;
const REUSE_NOISE: f64 = 0.1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Normal,
    WeakRaw,
    WeakReuse,
}

trait Pol {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize;
    fn upd(&mut self, t: &Task, a: usize, r: f64);
    fn weak(&mut self, t: &Task, a: usize, raw_ok: bool) {
        if !raw_ok {
            self.upd(t, a, 0.0)
        }
    }
    fn delayed(&mut self, t: &Task, a: usize, v: f64) {
        self.upd(t, a, v)
    }
    fn diag(&self) -> Option<[u64; 5]> {
        None
    }
}

struct Plain<R: Router>(R);
impl<R: Router> Pol for Plain<R> {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(t, c, rng)
    }
    fn upd(&mut self, t: &Task, a: usize, r: f64) {
        self.0.update(t, a, r)
    }
}

struct PolB(SonaB);
impl Pol for PolB {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(t, c, rng)
    }
    fn upd(&mut self, t: &Task, a: usize, r: f64) {
        self.0.update(t, a, r)
    }
    fn weak(&mut self, t: &Task, a: usize, raw_ok: bool) {
        self.0.update_weak(t, a, raw_ok)
    }
    fn delayed(&mut self, t: &Task, a: usize, v: f64) {
        self.0.update_delayed(t, a, v)
    }
    fn diag(&self) -> Option<[u64; 5]> {
        Some(self.0.stats)
    }
}

fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

fn x1_guard() -> GuardConfig {
    GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps,
    Pts,
    HebR,
    WarmHebR,
    Guard(GuardConfig),
    B(BCfg),
}

fn build(k: &Kind, w: &World) -> Box<dyn Pol> {
    let n = w.agents;
    let km = w.kwmap.clone();
    match k {
        Kind::Keyword => Box::new(Plain(KeywordRouter::new(km, n))),
        Kind::Eps => Box::new(Plain(EpsGreedyRouter::new(n, 0.95, 0.02))),
        Kind::Pts => Box::new(Plain(PooledThompsonRouter::new(n, 1.0))),
        Kind::HebR => Box::new(Plain(HebbianRouter::new(n, heb_robust_cfg()))),
        Kind::WarmHebR => Box::new(Plain(Warm::new(HebbianRouter::new(n, heb_robust_cfg()), km, n, 1.0).with_evidence(Evidence::MeanFeature))),
        Kind::Guard(g) => Box::new(Plain(Guarded::new(HebbianRouter::new(n, heb_robust_cfg()), km, n, *g))),
        Kind::B(c) => Box::new(PolB(SonaB::new(n, &km, c.clone()))),
    }
}

fn run_full(kind: &Kind, w: &World, seed: u64, mode: Mode) -> ([f64; 5], Option<[u64; 5]>) {
    let cands: Vec<usize> = (0..w.agents).collect();
    let mut pol = build(kind, w);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut orng = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x3EAC));
    let draws: Vec<(f64, f64, f64)> = (0..w.tasks.len()).map(|_| (orng.f64(), orng.f64(), orng.f64())).collect();
    let mut pending: VecDeque<(usize, usize, usize, f64)> = VecDeque::new();
    let (mut cum, mut out, mut hi) = (0.0, [0.0; 5], 0usize);
    for t in 0..w.tasks.len() {
        while pending.front().is_some_and(|p| p.0 <= t) {
            let (_, ti, a, v) = pending.pop_front().unwrap();
            pol.delayed(&w.tasks[ti], a, v);
        }
        let task = &w.tasks[t];
        let a = pol.pick(task, &cands, &mut rng);
        let p = w.p[t][a];
        let s = if w.u[t] < p { 1.0 } else { 0.0 };
        cum += p;
        match mode {
            Mode::Normal => pol.upd(task, a, s),
            _ => {
                let (u_tool, u_reuse, u_noise) = draws[t];
                let raw_ok = !w.down[t][a] && u_tool >= TOOL_BASE + TOOL_SLOPE * (1.0 - s);
                if mode == Mode::WeakRaw {
                    pol.upd(task, a, if raw_ok { 1.0 } else { 0.0 });
                } else {
                    pol.weak(task, a, raw_ok);
                    if raw_ok && u_reuse < REUSE_FRAC {
                        let v = if u_noise < REUSE_NOISE { 1.0 - s } else { s };
                        pending.push_back((t + REUSE_DELAY, t, a, v));
                    }
                }
            }
        }
        if hi < 5 && t + 1 == HORIZONS[hi] {
            out[hi] = cum / (t + 1) as f64;
            hi += 1;
        }
    }
    (out, pol.diag())
}

fn run(kind: &Kind, w: &World, seed: u64, mode: Mode) -> [f64; 5] {
    run_full(kind, w, seed, mode).0
}

fn eval(kind: &Kind, mode: Mode, seeds: &[u64], s: &WorldSpec) -> Vec<[f64; 5]> {
    seeds.par_iter().map(|&sd| run(kind, &gen_world(sd, s), sd, mode)).collect()
}

fn col(r: &[[f64; 5]], h: usize) -> Vec<f64> {
    r.iter().map(|x| x[h]).collect()
}

fn means(kind: &Kind, mode: Mode, seeds: &[u64], ws: &[WorldSpec]) -> Vec<[f64; 5]> {
    ws.iter()
        .map(|s| {
            let r = eval(kind, mode, seeds, s);
            let mut m = [0.0; 5];
            for h in 0..5 {
                m[h] = r.iter().map(|x| x[h]).sum::<f64>() / r.len() as f64;
            }
            m
        })
        .collect()
}

/// Regret-averse objective on tune seeds vs best baseline per (world, horizon): mean(d) + mean(min(d,0)).
fn obj(kind: &Kind, mode: Mode, seeds: &[u64], ws: &[WorldSpec], best_b: &[[f64; 5]]) -> f64 {
    let m = means(kind, mode, seeds, ws);
    let mut d = vec![];
    for (w, row) in m.iter().enumerate() {
        for h in 0..5 {
            d.push(row[h] - best_b[w][h]);
        }
    }
    let n = d.len() as f64;
    d.iter().sum::<f64>() / n + d.iter().map(|x| x.min(0.0)).sum::<f64>() / n
}

fn tune(label: &str, cands: Vec<BCfg>, mode: Mode, seeds: &[u64], ws: &[WorldSpec], best_b: &[[f64; 5]]) -> (f64, BCfg) {
    let n = cands.len();
    let mut sc: Vec<(f64, BCfg)> = cands.into_iter().map(|c| (obj(&Kind::B(c.clone()), mode, seeds, ws, best_b), c)).collect();
    sc.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("[tune 1-8] {label}: {n} configs, best obj {:+.5}\n    {:?}", sc[0].0, sc[0].1);
    sc.swap_remove(0)
}

fn best_of(baselines: &[(String, Kind, Mode)], seeds: &[u64], ws: &[WorldSpec]) -> Vec<[f64; 5]> {
    let bm: Vec<Vec<[f64; 5]>> = baselines.iter().map(|(_, k, m)| means(k, *m, seeds, ws)).collect();
    (0..ws.len())
        .map(|w| {
            let mut m = [f64::NEG_INFINITY; 5];
            for b in &bm {
                for h in 0..5 {
                    m[h] = m[h].max(b[w][h]);
                }
            }
            m
        })
        .collect()
}

// ------------------------------------------------------------------ grids (all built before any result is looked at)
fn rt_grid(weak: bool, quick: bool) -> Vec<BCfg> {
    let mut v = vec![];
    for &tau in &[0.45, 0.55, 0.65] {
        for &z in &[0.5, 1.0] {
            for &(ka, kb) in &[(2.0, 1.0), (4.0, 1.0)] {
                for &disc in &[1.0, 0.97, 0.94] {
                    for &ch in &[true, false] {
                        let ms: &[usize] = if weak { &[1, 2, 3] } else { &[1] };
                        for &m in ms {
                            v.push(BCfg { tau, z, kw_a: ka, kw_b: kb, discount: disc, challenge: ch, delayed_mult: m, ..BCfg::default() });
                        }
                    }
                }
            }
        }
    }
    if quick {
        v.truncate(6);
    }
    v
}

fn grid_expl(b: &BCfg, quick: bool) -> Vec<BCfg> {
    let mut v = vec![];
    for &eps in &[0.03, 0.08, 0.15] {
        for &bud in &[2u32, 5, 10] {
            for &pl in &[true, false] {
                for &h in &[0.0, 3.0] {
                    v.push(BCfg { xp_eps: eps, xp_budget: bud, xp_plaus: pl, hier_lambda: h, ..b.clone() });
                }
            }
        }
    }
    if quick {
        v.truncate(4);
    }
    v
}

fn grid_ts(b: &BCfg, quick: bool) -> Vec<BCfg> {
    let mut v = vec![];
    for &tau in &[0.45, 0.55, 0.65] {
        for &sc in &[0.5, 1.0, 2.0] {
            for &h in &[0.0, 3.0] {
                for &opt in &[0.0, 1.0] {
                    v.push(BCfg { inner: InnerKind::Ts, tau, ts_scale: sc, hier_lambda: h, ts_opt: opt, ..b.clone() });
                }
            }
        }
    }
    if quick {
        v.truncate(4);
    }
    v
}

fn grid_tdecay(b: &BCfg, quick: bool) -> Vec<BCfg> {
    let mut v = vec![];
    for &tt in &[150.0, 400.0, 1000.0, 3000.0] {
        for &d in &[1.0, 0.97, 0.94] {
            v.push(BCfg { time_tau: tt, discount: d, ..b.clone() });
        }
    }
    if quick {
        v.truncate(3);
    }
    v
}

fn grid_protect(b: &BCfg, quick: bool) -> Vec<BCfg> {
    let mut v = vec![];
    for &p in &[0.5, 0.8, 0.95] {
        for &pv in &[0.6, 0.75] {
            for &d in &[0.9, 0.94, 0.97] {
                v.push(BCfg { protect: p, protect_v: pv, discount: d, ..b.clone() });
            }
        }
    }
    if quick {
        v.truncate(3);
    }
    v
}

fn grid_detect(b: &BCfg, quick: bool, r2: bool) -> Vec<BCfg> {
    let mut v = vec![];
    let bans: &[u32] = if r2 { &[0, 25, 80, 200] } else { &[0] };
    for &dl in &[0.1, 0.25] {
        for &lam in &[1.5, 3.0, 5.0] {
            for &keep in &[0.2, 0.5] {
                for &d in &[1.0, 0.97] {
                    for &ban in bans {
                        v.push(BCfg { det_delta: dl, det_lambda: lam, det_keep: keep, discount: d, det_ban: ban, ..b.clone() });
                    }
                }
            }
        }
    }
    if quick {
        v.truncate(4);
    }
    v
}

/// copy the exploration / inner / forgetting fields of `from` onto `base`.
fn merge(base: &BCfg, expl: Option<&BCfg>, ts: Option<&BCfg>, forget: &[&BCfg]) -> BCfg {
    let mut c = base.clone();
    if let Some(e) = expl {
        c.xp_eps = e.xp_eps;
        c.xp_budget = e.xp_budget;
        c.xp_plaus = e.xp_plaus;
        c.hier_lambda = e.hier_lambda;
    }
    if let Some(t) = ts {
        c.inner = t.inner;
        c.tau = t.tau;
        c.ts_scale = t.ts_scale;
        c.ts_opt = t.ts_opt;
        c.hier_lambda = c.hier_lambda.max(t.hier_lambda);
    }
    for f in forget {
        if f.time_tau > 0.0 {
            c.time_tau = f.time_tau;
            c.discount = f.discount;
        }
        if f.protect > 0.0 {
            c.protect = f.protect;
            c.protect_v = f.protect_v;
            c.discount = f.discount;
        }
        if f.det_lambda > 0.0 {
            c.det_delta = f.det_delta;
            c.det_lambda = f.det_lambda;
            c.det_keep = f.det_keep;
            c.det_ban = f.det_ban;
            c.discount = f.discount;
        }
    }
    c
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Tuned {
    rt: BCfg,
    expl: BCfg,
    ts: BCfg,
    tdecay: BCfg,
    protect: BCfg,
    detect: BCfg,
    full: BCfg,
    full_parts: String,
    obj: HashMap<String, f64>,
}

fn tune_mode(label: &str, weak: bool, quick: bool, r2: bool, seeds: &[u64], ws: &[WorldSpec], best_b: &[[f64; 5]], mode: Mode) -> Tuned {
    println!("\n######## tuning ({label}) ########");
    let mut objs = HashMap::new();
    let (o_rt, rt) = tune("rt (guarded-hebr retuned, no new mechanism)", rt_grid(weak, quick), mode, seeds, ws, best_b);
    objs.insert("rt".to_string(), o_rt);
    let (o, expl) = tune("exploration (pattern prior + annealed uncertainty-directed budget)", grid_expl(&rt, quick), mode, seeds, ws, best_b);
    objs.insert("expl".to_string(), o);
    let (o, ts) = tune("Thompson over pattern posteriors (inner)", grid_ts(&rt, quick), mode, seeds, ws, best_b);
    objs.insert("ts".to_string(), o);
    let (o, tdecay) = tune("forgetting: lazy time decay", grid_tdecay(&rt, quick), mode, seeds, ws, best_b);
    objs.insert("tdecay".to_string(), o);
    let (o, protect) = tune("forgetting: EWC-like protection", grid_protect(&rt, quick), mode, seeds, ws, best_b);
    objs.insert("protect".to_string(), o);
    let (o, detect) = tune("forgetting: CUSUM drift reset", grid_detect(&rt, quick, r2), mode, seeds, ws, best_b);
    objs.insert("detect".to_string(), o);
    // full = best subset of {expl, ts, forget parts} (rt itself included => full >= rt on the tuning objective)
    let mut combos: Vec<(String, BCfg)> = vec![];
    for use_e in [false, true] {
        for use_t in [false, true] {
            for fm in 0..8u8 {
                let mut fs: Vec<&BCfg> = vec![];
                let mut nm = String::new();
                if fm & 1 != 0 {
                    fs.push(&tdecay);
                    nm += "tdecay+";
                }
                if fm & 2 != 0 {
                    fs.push(&protect);
                    nm += "protect+";
                }
                if fm & 4 != 0 {
                    fs.push(&detect);
                    nm += "detect+";
                }
                let name = format!("{}{}{}", if use_e { "expl+" } else { "" }, if use_t { "ts+" } else { "" }, nm);
                combos.push((name, merge(&rt, use_e.then_some(&expl), use_t.then_some(&ts), &fs)));
            }
        }
    }
    if quick {
        combos.truncate(4);
    }
    let mut scored: Vec<(f64, String, BCfg)> = combos.into_iter().map(|(n, c)| (obj(&Kind::B(c.clone()), mode, seeds, ws, best_b), n, c)).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("[tune 1-8] full-combo ranking (top 4 of {}):", scored.len());
    for (o, n, _) in scored.iter().take(4) {
        println!("      {:+.5}  {}", o, if n.is_empty() { "(rt only)" } else { n });
    }
    let (o_full, full_parts, full) = scored.swap_remove(0);
    objs.insert("full".into(), o_full);
    println!("[tune 1-8] objectives: {objs:?}");
    Tuned { rt, expl, ts, tdecay, protect, detect, full, full_parts, obj: objs }
}

// ------------------------------------------------------------------ comparison family
struct Arm {
    name: String,
    kind: Kind,
    mode: Mode,
}

struct Cell {
    world: String,
    cand: String,
    h: usize,
    diff: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    p: f64,
    best: String,
    cand_mean: f64,
    base_mean: f64,
}

fn oracle_row(held: &[u64], s: &WorldSpec) -> [f64; 5] {
    let rs: Vec<[f64; 5]> = held
        .iter()
        .map(|&sd| {
            let w = gen_world(sd, s);
            let (mut cum, mut out, mut hi) = (0.0, [0.0; 5], 0usize);
            for t in 0..w.tasks.len() {
                cum += w.p[t].iter().cloned().fold(0.0, f64::max);
                if hi < 5 && t + 1 == HORIZONS[hi] {
                    out[hi] = cum / (t + 1) as f64;
                    hi += 1;
                }
            }
            out
        })
        .collect();
    let mut m = [0.0; 5];
    for h in 0..5 {
        m[h] = rs.iter().map(|x| x[h]).sum::<f64>() / rs.len() as f64;
    }
    m
}

fn family(ws: &[WorldSpec], bases: &[Arm], cands: &[Arm], pools: &[(&str, Vec<usize>)], held: &[u64], tables: &mut Vec<Value>, tag: &str) -> Vec<(String, Vec<Cell>)> {
    let mut out: Vec<(String, Vec<Cell>)> = pools.iter().map(|(n, _)| (n.to_string(), vec![])).collect();
    for s in ws {
        let br: Vec<Vec<[f64; 5]>> = bases.iter().map(|a| eval(&a.kind, a.mode, held, s)).collect();
        let cr: Vec<Vec<[f64; 5]>> = cands.iter().map(|a| eval(&a.kind, a.mode, held, s)).collect();
        println!("\n=== [{tag}] world {} ({}) | held mean expected success over first H ===", s.name, s.class);
        println!("{:<44}{:>8}{:>8}{:>8}{:>8}{:>8}", "arm", 100, 250, 500, 1000, 2000);
        let mut rows = vec![];
        for (a, r) in bases.iter().zip(&br).chain(cands.iter().zip(&cr)) {
            let m: Vec<f64> = (0..5).map(|h| summarize(&col(r, h)).mean).collect();
            println!("{:<44}{}", a.name, m.iter().map(|x| format!("{:>8.4}", x)).collect::<String>());
            rows.push(json!({"arm": a.name, "means": m}));
        }
        let o = oracle_row(held, s);
        println!("{:<44}{}", "(oracle: best agent per task)", o.iter().map(|x| format!("{:>8.4}", x)).collect::<String>());
        rows.push(json!({"arm": "oracle", "means": o}));
        tables.push(json!({"mode": tag, "world": s.name, "class": s.class, "rows": rows}));
        for (pi, (_, pool)) in pools.iter().enumerate() {
            for h in 0..5 {
                let bi = *pool.iter().max_by(|&&a, &&b| summarize(&col(&br[a], h)).mean.partial_cmp(&summarize(&col(&br[b], h)).mean).unwrap()).unwrap();
                let b = col(&br[bi], h);
                for (ci, ca) in cands.iter().enumerate() {
                    let c = col(&cr[ci], h);
                    let d = paired_diff(&c, &b);
                    out[pi].1.push(Cell {
                        world: s.name.to_string(),
                        cand: ca.name.clone(),
                        h: HORIZONS[h],
                        diff: d.diff.mean,
                        lo: d.diff.lo,
                        hi: d.diff.hi,
                        wins: d.wins,
                        p: paired_t_pvalue(&c, &b),
                        best: bases[bi].name.clone(),
                        cand_mean: summarize(&c).mean,
                        base_mean: summarize(&b).mean,
                    });
                }
            }
        }
    }
    out
}

/// Direct paired comparison of every candidate against ONE arm (the ablation without the mechanism).
fn direct_family(ws: &[WorldSpec], base: &Arm, cands: &[Arm], held: &[u64]) -> Vec<Cell> {
    let mut cells = vec![];
    for s in ws {
        let rb = eval(&base.kind, base.mode, held, s);
        for ca in cands {
            let rc = eval(&ca.kind, ca.mode, held, s);
            for h in 0..5 {
                let (c, b) = (col(&rc, h), col(&rb, h));
                let d = paired_diff(&c, &b);
                cells.push(Cell {
                    world: s.name.to_string(),
                    cand: ca.name.clone(),
                    h: HORIZONS[h],
                    diff: d.diff.mean,
                    lo: d.diff.lo,
                    hi: d.diff.hi,
                    wins: d.wins,
                    p: paired_t_pvalue(&c, &b),
                    best: base.name.clone(),
                    cand_mean: summarize(&c).mean,
                    base_mean: summarize(&b).mean,
                });
            }
        }
    }
    cells
}

struct Rep {
    title: String,
    json: Value,
    /// (world, cand, H, diff, holm_p)
    rows: Vec<(String, String, usize, f64, f64)>,
}

fn report(title: &str, cells: &[Cell], verbose: bool) -> Rep {
    let adj = holm_adjust(&cells.iter().map(|c| c.p).collect::<Vec<_>>());
    println!("\n### {title}  (Holm over {} tests)", cells.len());
    if verbose {
        println!("{:<18} {:<26} {:>5} {:>9} {:>20} {:>5} {:>8}  baseline", "world", "candidate", "H", "diff", "95% CI", "wins", "Holm p");
    }
    let (mut out, mut rows) = (vec![], vec![]);
    for (c, a) in cells.iter().zip(&adj) {
        let mark = if *a < 0.05 { if c.diff > 0.0 { " WIN" } else { " LOSS" } } else { "" };
        if verbose && (c.h == 500 || c.h == 2000 || !mark.is_empty()) {
            println!("{:<18} {:<26} {:>5} {:>+9.4} [{:>+8.4},{:>+8.4}] {:>5} {:>8.4}  {}{}", c.world, c.cand, c.h, c.diff, c.lo, c.hi, c.wins, a, c.best, mark);
        }
        out.push(json!({"world": c.world, "candidate": c.cand, "H": c.h, "diff": c.diff, "ci": [c.lo, c.hi], "wins": c.wins,
            "p": c.p, "holm_p": a, "baseline": c.best, "cand_mean": c.cand_mean, "base_mean": c.base_mean}));
        rows.push((c.world.clone(), c.cand.clone(), c.h, c.diff, *a));
    }
    let (w, l) = rows.iter().fold((0, 0), |(w, l), r| if r.4 < 0.05 { if r.3 > 0.0 { (w + 1, l) } else { (w, l + 1) } } else { (w, l) });
    println!("--> Holm wins {w}, losses {l} of {}", cells.len());
    Rep { title: title.to_string(), json: json!({"title": title, "holm_wins": w, "holm_losses": l, "cells": out}), rows }
}

/// Per-candidate world-level summary of a family: worlds with a Holm WIN / LOSS at any H, plus the H2000 diff.
fn summary(rep: &Rep, cands: &[&str], ws: &[WorldSpec]) -> Value {
    println!("\n  -- world-level summary: {}", rep.title);
    println!("  {:<22} {:<18} {:<6} {:>9} {:>9} {:>9} {:>9}   Holm WIN Hs / LOSS Hs", "candidate", "world", "class", "H500", "H1000", "H2000", "H2000p");
    let mut js = vec![];
    for c in cands {
        let (mut win_worlds, mut loss_worlds, mut win_stat, mut loss_stat, mut win_drift, mut loss_drift) = (vec![], vec![], 0, 0, 0, 0);
        for s in ws {
            let mine: Vec<&(String, String, usize, f64, f64)> = rep.rows.iter().filter(|r| r.0 == s.name && r.1 == *c).collect();
            let wh: Vec<usize> = mine.iter().filter(|r| r.4 < 0.05 && r.3 > 0.0).map(|r| r.2).collect();
            let lh: Vec<usize> = mine.iter().filter(|r| r.4 < 0.05 && r.3 < 0.0).map(|r| r.2).collect();
            let get = |h: usize| mine.iter().find(|r| r.2 == h).map(|r| r.3).unwrap_or(f64::NAN);
            let p2k = mine.iter().find(|r| r.2 == 2000).map(|r| r.4).unwrap_or(f64::NAN);
            println!("  {:<22} {:<18} {:<6} {:>+9.4} {:>+9.4} {:>+9.4} {:>9.4}   {:?} / {:?}", c, s.name, s.class, get(500), get(1000), get(2000), p2k, wh, lh);
            if !wh.is_empty() {
                win_worlds.push(s.name);
                if s.class == "stat" { win_stat += 1 } else if s.class == "drift" { win_drift += 1 }
            }
            if !lh.is_empty() {
                loss_worlds.push(s.name);
                if s.class == "stat" { loss_stat += 1 } else if s.class == "drift" { loss_drift += 1 }
            }
        }
        println!("  ==> {c}: Holm-WIN worlds {} {:?} | Holm-LOSS worlds {} {:?}", win_worlds.len(), win_worlds, loss_worlds.len(), loss_worlds);
        js.push(json!({"candidate": c, "win_worlds": win_worlds, "loss_worlds": loss_worlds, "win_stationary": win_stat, "loss_stationary": loss_stat, "win_drift": win_drift, "loss_drift": loss_drift}));
    }
    json!({"title": rep.title, "per_candidate": js})
}

fn selftest(ws: &[WorldSpec]) {
    for s in ws.iter().take(4).chain(ws.iter().skip(6).take(1)) {
        for sd in 1..=3u64 {
            let w = gen_world(sd, s);
            for mode in [Mode::Normal, Mode::WeakReuse] {
                let a = run(&Kind::Guard(x1_guard()), &w, sd, mode);
                let b = run(&Kind::B(BCfg::default()), &w, sd, mode);
                assert_eq!(a, b, "SonaB(default) != Guarded(hebr) on {} seed {sd} {mode:?}", s.name);
            }
        }
    }
    println!("selftest OK: SonaB(all mechanisms off) is bit-identical to prior::Guarded<HebbianRouter> (5 worlds x 3 seeds x 2 modes)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let argv = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let out_path = argv("--out").unwrap_or("results/g19b_main.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let r2 = args.iter().any(|a| a == "--round2");
    let (h0, h1) = argv("--held").and_then(|r| r.split_once('-').map(|(a, b)| (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap()))).unwrap_or((31, 60));
    let held: Vec<u64> = (h0..=h1).collect();
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let ws = all_worlds();
    selftest(&ws);
    if args.iter().any(|a| a == "--selftest") {
        return;
    }
    println!("g20_sona_b: {N_TASKS} tasks/run, {} worlds, tune seeds 1-8, held-out {h0}-{h1}{}", ws.len(), if quick { " (QUICK)" } else { "" });

    let base6 = |mode: Mode| -> Vec<(String, Kind, Mode)> {
        vec![
            ("keyword".into(), Kind::Keyword, mode),
            ("eps-greedy".into(), Kind::Eps, mode),
            ("pooled-ts".into(), Kind::Pts, mode),
            ("hebbian-robust".into(), Kind::HebR, mode),
            ("warm(hebr)".into(), Kind::WarmHebR, mode),
            ("guarded-safe(x1)".into(), Kind::Guard(x1_guard()), mode),
        ]
    };

    // ---------------- tuning (or load frozen tuning)
    let tuned_in = argv("--tuned-in");
    let (tn, tw): (Tuned, Tuned) = if let Some(f) = &tuned_in {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
        println!("loaded FROZEN tuning from {f}");
        (serde_json::from_value(v["normal"].clone()).unwrap(), serde_json::from_value(v["weak"].clone()).unwrap())
    } else {
        let bn = best_of(&base6(Mode::Normal), &tune_seeds, &ws);
        let tn = tune_mode("NORMAL feedback", false, quick, r2, &tune_seeds, &ws, &bn, Mode::Normal);
        let bw = best_of(&base6(Mode::WeakRaw), &tune_seeds, &ws);
        let tw = tune_mode("WEAK feedback + reuse signal", true, quick, r2, &tune_seeds, &ws, &bw, Mode::WeakReuse);
        let tuned_out = argv("--tuned-out").unwrap_or("results/g19b_tuned.json".into());
        std::fs::write(&tuned_out, serde_json::to_string_pretty(&json!({"normal": tn, "weak": tw})).unwrap()).unwrap();
        println!("wrote frozen tuning {tuned_out}");
        (tn, tw)
    };

    let arm = |n: &str, k: Kind, m: Mode| Arm { name: n.to_string(), kind: k, mode: m };
    let cand_arms = |t: &Tuned, m: Mode| -> Vec<Arm> {
        vec![
            arm("expl", Kind::B(t.expl.clone()), m),
            arm("ts", Kind::B(t.ts.clone()), m),
            arm("tdecay", Kind::B(t.tdecay.clone()), m),
            arm("protect", Kind::B(t.protect.clone()), m),
            arm("detect", Kind::B(t.detect.clone()), m),
            arm("full", Kind::B(t.full.clone()), m),
        ]
    };
    let cand_names = ["expl", "ts", "tdecay", "protect", "detect", "full"];
    let mut tables = vec![];
    let mut reports: Vec<Value> = vec![];
    let mut summaries: Vec<Value> = vec![];

    // ---------------- NORMAL feedback
    {
        let mut bases: Vec<Arm> = base6(Mode::Normal).into_iter().map(|(n, k, m)| Arm { name: n, kind: k, mode: m }).collect();
        bases.push(arm("rt(guarded-hebr retuned)", Kind::B(tn.rt.clone()), Mode::Normal));
        let cands = cand_arms(&tn, Mode::Normal);
        let fam = family(&ws, &bases, &cands, &[("best-of-7 fair baselines (incl. retuned guarded)", (0..7).collect()), ("best-of-6 spec set", (0..6).collect()), ("best of {guarded-safe, warm(hebr)}", vec![4, 5])], &held, &mut tables, "NORMAL");
        for (t, c) in &fam {
            let r = report(&format!("NORMAL: candidates vs {t}"), c, false);
            summaries.push(summary(&r, &cand_names, &ws));
            reports.push(r.json);
        }
        let dir = direct_family(&ws, &bases[6], &cands, &held);
        let r = report("NORMAL: candidates - rt (SAME signal, mechanism OFF = ablation)", &dir, true);
        summaries.push(summary(&r, &cand_names, &ws));
        reports.push(r.json);
    }

    // ---------------- WEAK feedback (+ reuse)
    {
        let mut bases: Vec<Arm> = base6(Mode::WeakRaw).into_iter().map(|(n, k, m)| Arm { name: format!("{n} [raw]"), kind: k, mode: m }).collect();
        bases.push(arm("guarded-safe(x1) [reuse]", Kind::Guard(x1_guard()), Mode::WeakReuse));
        bases.push(arm("warm(hebr) [reuse]", Kind::WarmHebR, Mode::WeakReuse));
        bases.push(arm("rt [reuse]", Kind::B(tw.rt.clone()), Mode::WeakReuse));
        let cands = cand_arms(&tw, Mode::WeakReuse);
        let fam = family(
            &ws,
            &bases,
            &cands,
            &[
                ("STRICT pool: raw baselines + guarded-safe/warm/rt fed the SAME reuse signal", (0..9).collect()),
                ("best of {guarded-safe, warm(hebr)} fed the reuse signal", vec![6, 7]),
                ("raw-fed baselines only (x3 convention)", (0..6).collect()),
            ],
            &held,
            &mut tables,
            "WEAK",
        );
        for (t, c) in &fam {
            let r = report(&format!("WEAK: candidates vs {t}"), c, false);
            summaries.push(summary(&r, &cand_names, &ws));
            reports.push(r.json);
        }
        let dir = direct_family(&ws, &bases[8], &cands, &held);
        let r = report("WEAK: candidates - rt[reuse] (SAME signal, mechanism OFF = ablation)", &dir, true);
        summaries.push(summary(&r, &cand_names, &ws));
        reports.push(r.json);
    }

    // ---------------- diagnostics: how often do mechanisms fire (seed = first held seed)
    println!("\n### diagnostics (seed {h0}, NORMAL mode): [kw followed, challenger, inner, explorations, drift alarms]");
    let mut diag = vec![];
    for s in &ws {
        let w = gen_world(h0, s);
        for (n, cfg) in [("rt", &tn.rt), ("expl", &tn.expl), ("ts", &tn.ts), ("detect", &tn.detect), ("full", &tn.full)] {
            let (_, d) = run_full(&Kind::B(cfg.clone()), &w, h0, Mode::Normal);
            println!("  {:<18} {:<8} {:?}", s.name, n, d);
            diag.push(json!({"world": s.name, "arm": n, "stats": d}));
        }
    }

    let res = json!({
        "label": "SYNTHETIC-until-validated-on-real-traffic",
        "protocol": {"tune_seeds": "1-8", "heldout": format!("{h0}-{h1}"), "frozen_tuning_from": tuned_in, "horizons": HORIZONS, "n_tasks": N_TASKS,
            "metric": "prefix mean expected success of routed agent", "weak_model": {"tool_base": TOOL_BASE, "tool_slope": TOOL_SLOPE, "reuse_frac": REUSE_FRAC, "reuse_delay": REUSE_DELAY, "reuse_noise": REUSE_NOISE}},
        "tuned_normal": tn, "tuned_weak": tw,
        "tables": tables, "reports": reports, "summaries": summaries, "diagnostics": diag,
    });
    let _ = std::fs::create_dir_all("results");
    std::fs::write(&out_path, serde_json::to_string_pretty(&res).unwrap()).unwrap();
    println!("\nwrote {out_path}");
}
