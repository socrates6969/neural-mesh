//! D6 knowledge lab: retrieval-before-route over our own store, and the learn-by-doing loop.
//!
//! Worlds: sim.rs base worlds (keywords, traps, drift, outage) with an added recurring
//! "task context" (an embedding cluster) that determines part of the true skill:
//!   p(a|t) = (1-w) * sim_skill(type,a,t) + w * ctx_skill(context(t), a)      (0 when a is down)
//! Treatment worlds: context recurs and matters. Controls: context recurs but is irrelevant
//! (w=0), context matters but never recurs (unique embedding per task), neither.
//!
//! Regimes: "O" = realised-outcome feedback (isolates retrieval); "W" = weak feedback
//! (raw hook bit; graded delayed downstream-reuse signal for the knowledge loop, as in x3).
//!
//! Protocol: tune on seeds 1-8, held-out 31-60 (and frozen confirmation 61-90 via --held),
//! prefix mean EXPECTED success at H100/250/500/2000, paired 95% t-CI vs the best (per
//! world x horizon, by held-out mean) fair baseline, Holm over each regime's family.
//!
//! Usage: cargo run --release --bin d6_knowledge [-- --held 61-90 --out results/d6_knowledge_confirm61-90.json]

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::time::Instant;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::knowledge::{AnchorRouter, FeatMode, Knowledge, KnowledgeConfig, KnowledgeRouter, VoteConfig};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

const HORIZONS: [usize; 4] = [100, 250, 500, 2000];
const DIM: usize = 32;
const NA: usize = 6;

// =============================================================================================
// Worlds

#[derive(Clone, Debug)]
struct Spec {
    name: String,
    scenario: Scenario,
    /// number of recurring contexts; 0 = unique context per task (never recurs)
    k: usize,
    /// weight of the context skill in the true success probability
    w: f64,
    /// embedding noise norm around the context centroid
    noise: f64,
    /// treatment (retrieval is expected to help) or control
    treatment: bool,
}

struct CtxWorld {
    w: World,
    emb: Vec<Vec<f32>>,
    pctx: Vec<[f64; NA]>,
    ctx: Vec<usize>,
    wgt: f64,
    /// per task (u_tool, u_reuse, u_noise)
    draws: Vec<(f64, f64, f64)>,
}

fn uni(r: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * r.f64()
}
fn normal(r: &mut Rng) -> f64 {
    let u1 = r.f64().max(1e-12);
    let u2 = r.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
fn unit_vec(r: &mut Rng) -> Vec<f32> {
    let v: Vec<f32> = (0..DIM).map(|_| normal(r) as f32).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.into_iter().map(|x| x / n).collect()
}

impl CtxWorld {
    fn new(seed: u64, sp: &Spec) -> Self {
        let w = World::new(seed, WorldConfig { n_tasks: 2000, scenario: sp.scenario, ..WorldConfig::default() });
        let n = w.len();
        let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xC7C7_1234));
        let kk = if sp.k == 0 { n } else { sp.k };
        let cents: Vec<Vec<f32>> = (0..kk).map(|_| unit_vec(&mut r)).collect();
        let mut pctx = Vec::with_capacity(kk);
        for _ in 0..kk {
            let best = r.below(NA);
            let mut row = [0.0; NA];
            for (a, p) in row.iter_mut().enumerate() {
                *p = if a == best { uni(&mut r, 0.80, 0.90) } else { uni(&mut r, 0.15, 0.50) };
            }
            pctx.push(row);
        }
        let mut ctx = Vec::with_capacity(n);
        let mut emb = Vec::with_capacity(n);
        for t in 0..n {
            let c = if sp.k == 0 { t } else { r.below(kk) };
            ctx.push(c);
            let mut e: Vec<f32> = cents[c].clone();
            let s = sp.noise / (DIM as f64).sqrt();
            for x in e.iter_mut() {
                *x += (normal(&mut r) * s) as f32;
            }
            emb.push(e);
        }
        let draws = (0..n).map(|_| (r.f64(), r.f64(), r.f64())).collect();
        CtxWorld { w, emb, pctx, ctx, wgt: sp.w, draws }
    }
    fn p(&self, t: usize, a: usize) -> f64 {
        if self.w.is_down(t, a) {
            return 0.0;
        }
        let ty = self.w.task(t).type_idx;
        (1.0 - self.wgt) * self.w.skill(t, ty, a) + self.wgt * self.pctx[self.ctx[t]][a]
    }
    fn outcome(&self, t: usize, a: usize) -> f64 {
        if self.w.task(t).u < self.p(t, a) { 1.0 } else { 0.0 }
    }
    fn oracle(&self) -> f64 {
        let n = self.w.len();
        (0..n).map(|t| (0..NA).map(|a| self.p(t, a)).fold(0.0, f64::max)).sum::<f64>() / n as f64
    }
}

#[derive(Clone, Debug)]
struct WeakCfg {
    tool_base: f64,
    tool_slope: f64,
    reuse_frac: f64,
    reuse_delay: usize,
    reuse_noise: f64,
}
const WEAK: WeakCfg = WeakCfg { tool_base: 0.01, tool_slope: 0.03, reuse_frac: 0.3, reuse_delay: 25, reuse_noise: 0.1 };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    Outcome,
    Raw,
    /// immediate 0 on raw failure, delayed graded reuse otherwise (x3 default)
    Reuse,
}
impl Signal {
    fn label(self) -> &'static str {
        match self {
            Signal::Outcome => "outcome",
            Signal::Raw => "raw",
            Signal::Reuse => "reuse",
        }
    }
}

// =============================================================================================
// Policies

trait Pol {
    fn pick(&mut self, t: &Task, emb: &[f32], c: &[usize], rng: &mut Rng) -> usize;
    fn fb(&mut self, t: &Task, a: usize, r: Option<f64>);
}

struct Plain<R: Router>(R);
impl<R: Router> Pol for Plain<R> {
    fn pick(&mut self, t: &Task, _e: &[f32], c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(t, c, rng)
    }
    fn fb(&mut self, t: &Task, a: usize, r: Option<f64>) {
        if let Some(r) = r {
            self.0.update(t, a, r)
        }
    }
}

struct KnPol<R: Router>(KnowledgeRouter<R>);
impl<R: Router> Pol for KnPol<R> {
    fn pick(&mut self, t: &Task, e: &[f32], c: &[usize], rng: &mut Rng) -> usize {
        self.0.set_context(e);
        self.0.select(t, c, rng)
    }
    fn fb(&mut self, t: &Task, a: usize, r: Option<f64>) {
        if let Some(r) = r {
            self.0.update(t, a, r)
        }
    }
}

/// knowledge configuration under test (frozen after tuning on seeds 1-8)
#[derive(Clone, Debug)]
struct KnCfg {
    label: String,
    feat: FeatMode,
    k: usize,
    sim_min: f32,
    vote: Option<(f64, f64, Option<usize>, Option<(f64, f64, usize)>)>, // (min_n, tau, explore_after, gate)
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm,
    Guard,
    Kn(KnCfg),
}

fn x1_guard() -> GuardConfig {
    GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
}

fn build(k: &Kind, w: &World) -> Box<dyn Pol> {
    let kw = w.keyword_map_for();
    match k {
        Kind::Keyword => Box::new(Plain(KeywordRouter::new(kw, NA))),
        Kind::Eps => Box::new(Plain(EpsGreedyRouter::new(NA, 0.99, 0.02))),
        Kind::Pts => Box::new(Plain(PooledThompsonRouter::new(NA, 1.0))),
        Kind::HebR => Box::new(Plain(HebbianRouter::new(NA, heb_robust_cfg()))),
        Kind::Warm => Box::new(Plain(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, 1.0).with_evidence(Evidence::MeanFeature))),
        Kind::Guard => Box::new(Plain(Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, x1_guard()))),
        Kind::Kn(c) => {
            let mut kc = KnowledgeConfig::new(DIM);
            kc.k = c.k;
            kc.sim_min = c.sim_min;
            kc.feat = c.feat;
            kc.vote = c.vote.map(|(min_n, tau, ex, gate)| VoteConfig { min_n, tau, prior: (1.0, 1.0), explore_after: ex, gate });
            let kn = Knowledge::new_shared(kc);
            let anchors = Rc::new(RefCell::new(HashMap::new()));
            if c.feat == FeatMode::Hidden {
                let inner = AnchorRouter::new(HebbianRouter::new(NA, heb_robust_cfg()), anchors.clone());
                Box::new(KnPol(KnowledgeRouter::new(Guarded::new(inner, kw, NA, x1_guard()), kn, anchors)))
            } else {
                let front = Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, x1_guard());
                Box::new(KnPol(KnowledgeRouter::new(front, kn, anchors)))
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
struct RunOut {
    h: [f64; 4],
    fail: f64,
    all: f64,
}

fn run(k: &Kind, sp: &Spec, sig: Signal, seed: u64) -> RunOut {
    let cw = CtxWorld::new(seed, sp);
    let w = &cw.w;
    let cands: Vec<usize> = (0..NA).collect();
    let mut pol = build(k, w);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut pending: VecDeque<(usize, usize, usize, f64)> = VecDeque::new();
    let (mut cum, mut fs, mut fnn) = (0.0f64, 0.0f64, 0.0f64);
    let mut out = RunOut::default();
    for t in 0..w.len() {
        while pending.front().is_some_and(|p| p.0 <= t) {
            let (_, ti, a, v) = pending.pop_front().unwrap();
            pol.fb(&w.task(ti).task, a, Some(v));
        }
        let st = w.task(t);
        let a = pol.pick(&st.task, &cw.emb[t], &cands, &mut rng);
        let pe = cw.p(t, a);
        let s = cw.outcome(t, a);
        match sig {
            Signal::Outcome => pol.fb(&st.task, a, Some(s)),
            _ => {
                let (u_tool, u_reuse, u_noise) = cw.draws[t];
                let raw_ok = !w.is_down(t, a) && u_tool >= WEAK.tool_base + WEAK.tool_slope * (1.0 - s);
                match sig {
                    Signal::Raw => pol.fb(&st.task, a, Some(if raw_ok { 1.0 } else { 0.0 })),
                    _ => {
                        pol.fb(&st.task, a, if raw_ok { None } else { Some(0.0) });
                        if raw_ok && u_reuse < WEAK.reuse_frac {
                            let v = if u_noise < WEAK.reuse_noise { 1.0 - s } else { s };
                            pending.push_back((t + WEAK.reuse_delay, t, a, v));
                        }
                    }
                }
            }
        }
        cum += pe;
        if w.in_failure(t) {
            fs += pe;
            fnn += 1.0;
        }
        for (i, &hz) in HORIZONS.iter().enumerate() {
            if t + 1 == hz {
                out.h[i] = cum / hz as f64;
            }
        }
    }
    out.fail = fs / fnn.max(1.0);
    out.all = cum / w.len() as f64;
    out
}

fn eval(k: &Kind, sp: &Spec, sig: Signal, seeds: &[u64]) -> Vec<RunOut> {
    seeds.par_iter().map(|&s| run(k, sp, sig, s)).collect()
}

fn col(r: &[RunOut], i: usize) -> Vec<f64> {
    r.iter().map(|x| x.h[i]).collect()
}

// =============================================================================================
// Comparison family

struct Cell {
    world: String,
    cand: String,
    h: usize,
    diff: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    p: f64,
    best_base: String,
    cand_mean: f64,
    base_mean: f64,
    treatment: bool,
}

type Arm = (String, Kind, Signal);

fn argmax_by<T>(xs: &[T], f: impl Fn(&T) -> f64) -> usize {
    let mut b = 0;
    for i in 1..xs.len() {
        if f(&xs[i]) > f(&xs[b]) {
            b = i;
        }
    }
    b
}

fn family(specs: &[Spec], bases: &[Arm], cands: &[Arm], held: &[u64], tables: &mut Vec<Value>, tag: &str) -> Vec<Cell> {
    let mut cells = vec![];
    for sp in specs {
        let br: Vec<Vec<RunOut>> = bases.iter().map(|(_, k, s)| eval(k, sp, *s, held)).collect();
        let cr: Vec<Vec<RunOut>> = cands.iter().map(|(_, k, s)| eval(k, sp, *s, held)).collect();
        let show = !tag.ends_with("abl");
        let orc: f64 = held.iter().map(|&s| CtxWorld::new(s, sp).oracle()).sum::<f64>() / held.len() as f64;
        if show { println!("\n=== [{tag}] world {} (K={} w={} noise={}) oracle {:.3} | mean expected success (prefix H) | fail-window ===", sp.name, sp.k, sp.w, sp.noise, orc); }
        if show { println!("{:<40} {:>8} {:>8} {:>8} {:>8} {:>8}", "policy", "H100", "H250", "H500", "H2000", "fail"); }
        let mut rows = vec![];
        for ((name, _, s), r) in bases.iter().zip(&br).chain(cands.iter().zip(&cr)) {
            let m: Vec<f64> = (0..4).map(|i| summarize(&col(r, i)).mean).collect();
            let f = r.iter().map(|x| x.fail).sum::<f64>() / r.len() as f64;
            if show { println!("{:<40} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}", format!("{name} [{}]", s.label()), m[0], m[1], m[2], m[3], f); }
            rows.push(json!({"policy": name, "signal": s.label(), "h": m, "fail": f}));
        }
        if show { tables.push(json!({"regime": tag, "world": sp.name, "oracle": orc, "rows": rows})); }
        for i in 0..4 {
            let bi = argmax_by(&br, |r| summarize(&col(r, i)).mean);
            let b = col(&br[bi], i);
            for (ci, (cname, _, _)) in cands.iter().enumerate() {
                let c = col(&cr[ci], i);
                let d = paired_diff(&c, &b);
                cells.push(Cell {
                    world: sp.name.clone(),
                    cand: cname.clone(),
                    h: HORIZONS[i],
                    diff: d.diff.mean,
                    lo: d.diff.lo,
                    hi: d.diff.hi,
                    wins: d.wins,
                    p: paired_t_pvalue(&c, &b),
                    best_base: bases[bi].0.clone(),
                    cand_mean: summarize(&c).mean,
                    base_mean: summarize(&b).mean,
                    treatment: sp.treatment,
                });
            }
        }
    }
    cells
}

fn report(title: &str, cells: &[Cell], n_seeds: usize) -> Value {
    let p: Vec<f64> = cells.iter().map(|c| c.p).collect();
    let adj = holm_adjust(&p);
    println!("\n### {title}  (Holm over {} tests, vs best fair baseline)", cells.len());
    println!("{:<18} {:<26} {:>5} {:>9} {:>20} {:>6} {:>8}  best baseline", "world", "candidate", "H", "diff", "95% CI", "wins", "Holm p");
    let mut out = vec![];
    let (mut nw, mut nl) = (0, 0);
    for (c, a) in cells.iter().zip(&adj) {
        let mark = if *a < 0.05 {
            if c.diff > 0.0 {
                nw += 1;
                " WIN"
            } else {
                nl += 1;
                " LOSS"
            }
        } else {
            ""
        };
        println!(
            "{:<18} {:<26} {:>5} {:>+9.4} [{:>+8.4},{:>+8.4}] {:>6} {:>8.4}  {}{}",
            c.world, c.cand, c.h, c.diff, c.lo, c.hi, format!("{}/{}", c.wins, n_seeds), a, c.best_base, mark
        );
        out.push(json!({"world": c.world, "treatment": c.treatment, "candidate": c.cand, "H": c.h, "diff": c.diff, "ci": [c.lo, c.hi],
            "wins": c.wins, "n_seeds": n_seeds, "p": c.p, "holm_p": a, "best_baseline": c.best_base, "cand_mean": c.cand_mean, "base_mean": c.base_mean}));
    }
    println!("-> Holm WINS {nw}, Holm LOSSES {nl}");
    json!({"title": title, "holm_wins": nw, "holm_losses": nl, "cells": out})
}

// =============================================================================================

fn cfg_grid() -> Vec<KnCfg> {
    let mut v = vec![];
    for &k in &[3usize, 5, 8] {
        for &sm in &[0.6f32, 0.7] {
            v.push(KnCfg { label: format!("feat-hid k{k} s{sm}"), feat: FeatMode::Hidden, k, sim_min: sm, vote: None });
            v.push(KnCfg { label: format!("feat-dir k{k} s{sm}"), feat: FeatMode::Direct, k, sim_min: sm, vote: None });
            for &(mn, tau) in &[(1.0, 0.65), (2.0, 0.7)] {
                for ex in [None, Some(3usize)] {
                    for gate in [None, Some((6.0, 0.02, 25usize))] {
                        v.push(KnCfg { label: format!("vote k{k} s{sm} n{mn} t{tau} x{ex:?} g{}", gate.is_some()), feat: FeatMode::Off, k, sim_min: sm, vote: Some((mn, tau, ex, gate)) });
                        v.push(KnCfg { label: format!("both-dir k{k} s{sm} n{mn} t{tau} x{ex:?} g{}", gate.is_some()), feat: FeatMode::Direct, k, sim_min: sm, vote: Some((mn, tau, ex, gate)) });
                    }
                }
            }
        }
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/d6_knowledge.json".into());
    let held: Vec<u64> = match args.iter().position(|a| a == "--held").and_then(|i| args.get(i + 1)) {
        Some(r) => {
            let (a, b) = r.split_once('-').expect("--held a-b");
            (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
        }
        None => (31..=60).collect(),
    };
    let tune: Vec<u64> = (1..=8).collect();
    let t0 = Instant::now();

    let sp = |name: &str, sc: Scenario, k: usize, w: f64, noise: f64, treatment: bool| Spec { name: name.into(), scenario: sc, k, w, noise, treatment };
    let worlds = vec![
        sp("ctx-strong-fixed", Scenario::Fixed, 12, 0.8, 0.5, true),
        sp("ctx-mild-random", Scenario::Random, 24, 0.5, 0.5, true),
        sp("ctx-noisy-emb", Scenario::Fixed, 12, 0.8, 1.2, true),
        sp("ctl-irrelevant-ctx", Scenario::Fixed, 12, 0.0, 0.5, false),
        sp("ctl-unique-ctx", Scenario::Fixed, 0, 0.8, 0.5, false),
        sp("ctl-none-random", Scenario::Random, 0, 0.0, 0.5, false),
    ];
    println!("d6_knowledge: held-out {}-{} ({} seeds), tune seeds 1-8, horizons {HORIZONS:?}", held[0], held[held.len() - 1], held.len());

    // ---- tune the knowledge config on seeds 1-8 in regime W (default reuse path) ----------
    // objective (regret-averse): mean margin vs best baseline + mean of negative margins,
    // over all worlds x horizons; baselines evaluated under the same regime.
    let base_w: Vec<Arm> = vec![
        ("keyword".into(), Kind::Keyword, Signal::Raw),
        ("eps-greedy".into(), Kind::Eps, Signal::Raw),
        ("pooled-ts".into(), Kind::Pts, Signal::Raw),
        ("hebbian-robust".into(), Kind::HebR, Signal::Raw),
        ("warm(hebr)".into(), Kind::Warm, Signal::Raw),
        ("guarded(hebr)".into(), Kind::Guard, Signal::Raw),
        ("guarded(hebr)+reuse".into(), Kind::Guard, Signal::Reuse),
        ("warm(hebr)+reuse".into(), Kind::Warm, Signal::Reuse),
    ];
    let base_o: Vec<Arm> = vec![
        ("keyword".into(), Kind::Keyword, Signal::Outcome),
        ("eps-greedy".into(), Kind::Eps, Signal::Outcome),
        ("pooled-ts".into(), Kind::Pts, Signal::Outcome),
        ("hebbian-robust".into(), Kind::HebR, Signal::Outcome),
        ("warm(hebr)".into(), Kind::Warm, Signal::Outcome),
        ("guarded(hebr)".into(), Kind::Guard, Signal::Outcome),
    ];
    let tune_bases: Vec<Vec<Vec<RunOut>>> = worlds.iter().map(|s| base_w.iter().map(|(_, k, sg)| eval(k, s, *sg, &tune)).collect()).collect();
    let mut scored: Vec<(f64, KnCfg)> = vec![];
    for c in cfg_grid() {
        let (mut margins, mut tot) = (vec![], 0.0);
        for (wi, s) in worlds.iter().enumerate() {
            let r = eval(&Kind::Kn(c.clone()), s, Signal::Reuse, &tune);
            for i in 0..4 {
                let best = tune_bases[wi].iter().map(|b| summarize(&col(b, i)).mean).fold(f64::NEG_INFINITY, f64::max);
                let m = summarize(&col(&r, i)).mean - best;
                margins.push(m);
                tot += m;
            }
        }
        let neg: f64 = margins.iter().map(|m| m.min(0.0)).sum::<f64>() / margins.len() as f64;
        let obj = tot / margins.len() as f64 + neg;
        println!("[tune] {:<36} obj {:+.4}", c.label, obj);
        scored.push((obj, c));
    }
    let pick = |pred: &dyn Fn(&KnCfg) -> bool| -> KnCfg {
        scored.iter().filter(|(_, c)| pred(c)).max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap().1.clone()
    };
    let mut c_feat = pick(&|c| c.feat == FeatMode::Hidden && c.vote.is_none());
    let mut c_dir = pick(&|c| c.feat == FeatMode::Direct);
    let mut c_vote = pick(&|c| c.feat == FeatMode::Off);
    let mut c_both = pick(&|c| c.feat == FeatMode::Direct && c.vote.is_some());
    c_feat.label = "kn-feat(hidden)".into();
    c_dir.label = "kn-feat(direct)".into();
    c_vote.label = "kn-vote".into();
    c_both.label = "kn-direct+vote".into();
    println!("[tune] selected: feat {:?} | dir {:?} | vote {:?} | both {:?}", c_feat, c_dir, c_vote, c_both);

    let mut tables = vec![];
    let cand = |c: &KnCfg, sg: Signal| -> Arm { (c.label.clone(), Kind::Kn(c.clone()), sg) };

    // Regime O: outcome feedback for everyone (isolates retrieval).
    let cands_o = vec![cand(&c_feat, Signal::Outcome), cand(&c_dir, Signal::Outcome), cand(&c_vote, Signal::Outcome), cand(&c_both, Signal::Outcome)];
    let cells_o = family(&worlds, &base_o, &cands_o, &held, &mut tables, "O");
    let rep_o = report("REGIME O (outcome feedback): knowledge vs best fair baseline", &cells_o, held.len());
    let g_o: Vec<Arm> = vec![("guarded(hebr)".into(), Kind::Guard, Signal::Outcome)];
    let rep_oa = report("ABLATION O: knowledge vs the SAME front without knowledge, guarded(hebr), outcome feedback (must-not-hurt)", &family(&worlds, &g_o, &cands_o, &held, &mut vec![], "O-abl"), held.len());

    // Regime W: weak feedback; knowledge loop uses the graded delayed reuse signal behind Guarded.
    let mut cands_w = vec![cand(&c_feat, Signal::Reuse), cand(&c_dir, Signal::Reuse), cand(&c_vote, Signal::Reuse), cand(&c_both, Signal::Reuse)];
    let mut raw_abl = cand(&c_both, Signal::Raw);
    raw_abl.0 = "kn-direct+vote (raw ablation)".into();
    cands_w.push(raw_abl);
    let cells_w = family(&worlds, &base_w, &cands_w, &held, &mut tables, "W");
    let rep_w = report("REGIME W (weak feedback; knowledge uses reuse signal): knowledge vs best fair baseline (incl. guarded+reuse)", &cells_w, held.len());
    let g_w: Vec<Arm> = vec![("guarded(hebr)+reuse".into(), Kind::Guard, Signal::Reuse)];
    let rep_wa = report("ABLATION W: knowledge vs the SAME front without knowledge, guarded(hebr)+reuse (must-not-hurt)", &family(&worlds, &g_w, &cands_w[..4], &held, &mut vec![], "W-abl"), held.len());

    // Feedback-path check: does reuse beat raw INSIDE the knowledge loop? (paired, no baselines)
    println!("\n### learn-by-doing check: kn-direct+vote reuse vs raw (same knowledge layer), diff of H2000 / H500 / H250");
    let mut fb_rows = vec![];
    let mut fcells = vec![];
    for s in &worlds {
        let a = eval(&Kind::Kn(c_both.clone()), s, Signal::Reuse, &held);
        let b = eval(&Kind::Kn(c_both.clone()), s, Signal::Raw, &held);
        for (i, hz) in HORIZONS.iter().enumerate().skip(1) {
            let (ca, cb) = (col(&a, i), col(&b, i));
            let d = paired_diff(&ca, &cb);
            fcells.push(Cell { world: s.name.clone(), cand: "reuse-vs-raw".into(), h: *hz, diff: d.diff.mean, lo: d.diff.lo, hi: d.diff.hi, wins: d.wins, p: paired_t_pvalue(&ca, &cb), best_base: "kn raw".into(), cand_mean: summarize(&ca).mean, base_mean: summarize(&cb).mean, treatment: s.treatment });
        }
    }
    let rep_f = report("FEEDBACK PATH: knowledge loop with reuse signal vs same loop with raw hook bit", &fcells, held.len());
    fb_rows.push(rep_f);

    let doc = json!({
        "lab": "d6_knowledge", "held": [held[0], held[held.len()-1]], "tune": [1, 8], "horizons": HORIZONS,
        "selected": {"feat": format!("{:?}", c_feat), "direct": format!("{:?}", c_dir), "vote": format!("{:?}", c_vote), "both": format!("{:?}", c_both)},
        "worlds": worlds.iter().map(|s| json!({"name": s.name, "scenario": s.scenario.label(), "k": s.k, "w": s.w, "noise": s.noise, "treatment": s.treatment})).collect::<Vec<_>>(),
        "regime_O": rep_o, "ablation_O": rep_oa, "regime_W": rep_w, "ablation_W": rep_wa, "feedback_path": fb_rows, "tables": tables,
        "elapsed_s": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("\nwrote {out_path} in {:.1}s", t0.elapsed().as_secs_f64());
}
