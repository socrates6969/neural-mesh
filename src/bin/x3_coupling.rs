//! X3 coupling lab: does a closed router <-> mesh <-> memory learning loop beat its parts?
//!
//! H1 router<->mesh signal exchange (health prior, outage shield, router-gated probes)
//! H2 shared experience replay (consolidation) vs an eta-matched no-replay control
//! H3 feedback quality: a weak-feedback world where the true quality is graded but the
//!    hook only reports "tool failed or not" (~always 1); which signal makes routing learn?
//! H4 memory budget: bytes of the combined state, ns/route, f32/u16/u8 + capped memory.
//!
//! Protocol: tune on seeds 1-8 only, held-out seeds 31-60, horizons 100/250/500/2000
//! (prefix mean of EXPECTED success of the routed agent), paired 95% t-CI, Holm, compare
//! against the best (per world x horizon, by held-out mean) of {keyword, eps-greedy,
//! pooled-thompson, hebbian-robust, warm(hebbian-robust) k=1 MeanFeature, warm tuned,
//! mesh-robust}.
//!
//! Usage: cargo run --release --bin x3_coupling [-- --out results/x3_coupling.json]

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::{
    heb_robust_cfg, quantise_router, CoupledBrain, CouplingConfig, Precision, RewardShaper, ShapingConfig, StateBytes,
    WarmEvidence,
};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::mesh::{MeshConfig, NeuralMesh};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig, TYPES};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

const HORIZONS: [usize; 4] = [100, 250, 500, 2000];

// =============================================================================================
// Weak-feedback observation model

#[derive(Clone, Debug)]
struct WeakCfg {
    /// latency multiplier of a failed attempt: lat *= 1 + kappa * (1 - s)
    kappa: f64,
    /// lognormal latency noise sigma
    sigma: f64,
    /// per-agent speed factor spread: speed in [1 - spread/2, 1 + spread/2] (confound)
    speed_spread: f64,
    /// P(tool reports failure) = base + slope * (1 - s) (when the agent is up)
    tool_base: f64,
    tool_slope: f64,
    /// fraction of tasks whose output is reused downstream (delayed graded signal)
    reuse_frac: f64,
    reuse_delay: usize,
    /// probability that the reuse signal is flipped
    reuse_noise: f64,
}

impl Default for WeakCfg {
    fn default() -> Self {
        WeakCfg {
            kappa: 0.6,
            sigma: 0.35,
            speed_spread: 0.3,
            tool_base: 0.01,
            tool_slope: 0.03,
            reuse_frac: 0.3,
            reuse_delay: 25,
            reuse_noise: 0.1,
        }
    }
}

/// Per-seed hidden latency structure + per-task pre-drawn uniforms (paired across policies).
struct WeakObs {
    base: [f64; 8],
    speed: Vec<f64>,
    /// per task: (u_tool, z_lat, u_reuse, u_noise)
    draws: Vec<(f64, f64, f64, f64)>,
}

fn normal(r: &mut Rng) -> f64 {
    let u1 = r.f64().max(1e-12);
    let u2 = r.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

impl WeakObs {
    fn new(seed: u64, n: usize, n_agents: usize, wc: &WeakCfg) -> Self {
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x3EAC));
        let mut base = [0.0; 8];
        for b in base.iter_mut() {
            *b = (r.f64() * 2.0 - 0.7).exp(); // ~0.5 .. 3.7
        }
        let speed = (0..n_agents).map(|_| 1.0 - wc.speed_spread / 2.0 + wc.speed_spread * r.f64()).collect();
        let draws = (0..n).map(|_| (r.f64(), normal(&mut r), r.f64(), r.f64())).collect();
        WeakObs { base, speed, draws }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signal {
    /// realised Bernoulli outcome (the classic sim feedback)
    Outcome,
    /// hook bit only: tool failed or not
    Raw,
    /// raw + latency shaping
    Lat,
    /// raw failures now + delayed downstream reuse
    Reuse,
    /// latency shaping + delayed reuse
    LatReuse,
}

impl Signal {
    fn label(self) -> &'static str {
        match self {
            Signal::Outcome => "outcome",
            Signal::Raw => "raw",
            Signal::Lat => "lat",
            Signal::Reuse => "reuse",
            Signal::LatReuse => "lat+reuse",
        }
    }
}

// =============================================================================================
// Policies

trait Pol {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize;
    /// health signal for the mesh part, router reward (None = no router update now)
    fn fb(&mut self, t: &Task, a: usize, health: f64, r: Option<f64>);
    fn fb_delayed(&mut self, t: &Task, a: usize, r: f64);
    fn bytes(&self) -> Option<StateBytes> {
        None
    }
}

struct Plain<R: Router>(R);
impl<R: Router> Pol for Plain<R> {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(t, c, rng)
    }
    fn fb(&mut self, t: &Task, a: usize, _h: f64, r: Option<f64>) {
        if let Some(r) = r {
            self.0.update(t, a, r)
        }
    }
    fn fb_delayed(&mut self, t: &Task, a: usize, r: f64) {
        self.0.update(t, a, r)
    }
}

struct MeshPol(NeuralMesh<HebbianRouter>);
impl Pol for MeshPol {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.route(t, c, rng)[0]
    }
    fn fb(&mut self, t: &Task, a: usize, _h: f64, r: Option<f64>) {
        if let Some(r) = r {
            self.0.feedback(t, &[a], r)
        }
    }
    fn fb_delayed(&mut self, t: &Task, a: usize, r: f64) {
        self.0.feedback(t, &[a], r)
    }
}

struct CoupledPol(CoupledBrain, Precision);
impl Pol for CoupledPol {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.route(t, c, rng)
    }
    fn fb(&mut self, t: &Task, a: usize, h: f64, r: Option<f64>) {
        self.0.feedback_split(t, a, h, r);
        if r.is_some() {
            quantise_router(self.0.router.inner_mut(), self.1);
        }
    }
    fn fb_delayed(&mut self, t: &Task, a: usize, r: f64) {
        self.0.feedback_delayed(t, a, r);
        quantise_router(self.0.router.inner_mut(), self.1);
    }
    fn bytes(&self) -> Option<StateBytes> {
        Some(self.0.state_bytes())
    }
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps,
    Pts,
    HebR,
    /// warm(hebbian with cfg) k, evidence
    Warm(HebbianConfig, f64, WarmEvidence),
    MeshR,
    /// router-thinker's Guarded<hebbian-robust> (reference / composition partner)
    Guard(GuardConfig),
    Coupled(CouplingConfig, Precision),
}

/// router-thinker's frozen x1_lab config (tau .55, z .5, keyword prior Beta(2,1), discount .97).
fn x1_guard() -> GuardConfig {
    GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
}

fn heb_eta(eta: f64) -> HebbianConfig {
    HebbianConfig { eta, ..heb_robust_cfg() }
}

fn build(k: &Kind, w: &World) -> Box<dyn Pol> {
    let n = w.n_agents();
    let kw = w.keyword_map_for();
    match k {
        Kind::Keyword => Box::new(Plain(KeywordRouter::new(kw, n))),
        Kind::Eps => Box::new(Plain(EpsGreedyRouter::new(n, 0.99, 0.02))),
        Kind::Pts => Box::new(Plain(PooledThompsonRouter::new(n, 1.0))),
        Kind::HebR => Box::new(Plain(HebbianRouter::new(n, heb_robust_cfg()))),
        Kind::Warm(h, k, e) => {
            let ev = match e {
                WarmEvidence::FeatureSet => Evidence::FeatureSet,
                WarmEvidence::MeanFeature => Evidence::MeanFeature,
                WarmEvidence::Global => Evidence::Global,
                WarmEvidence::SetFailures => Evidence::SetFailures,
            };
            Box::new(Plain(Warm::new(HebbianRouter::new(n, h.clone()), kw, n, *k).with_evidence(ev)))
        }
        Kind::MeshR => {
            let m = MeshConfig { edge_eta: 0.01, edge_decay: 0.0, team_size: 1, ..MeshConfig::default() };
            Box::new(MeshPol(NeuralMesh::new(n, HebbianRouter::new(n, heb_robust_cfg()), m)))
        }
        Kind::Guard(g) => Box::new(Plain(Guarded::new(HebbianRouter::new(n, heb_robust_cfg()), kw, n, g.clone()))),
        Kind::Coupled(c, p) => Box::new(CoupledPol(CoupledBrain::new(n, kw, c.clone()), *p)),
    }
}

// =============================================================================================
// Worlds and runs

#[derive(Clone, Debug)]
struct Spec {
    name: String,
    scenario: Scenario,
    weak: Option<WeakCfg>,
}

fn world_cfg(s: Scenario) -> WorldConfig {
    WorldConfig { n_tasks: 2000, scenario: s, ..WorldConfig::default() }
}

#[derive(Clone, Debug, Default)]
struct RunOut {
    /// prefix mean expected success at HORIZONS
    h: [f64; 4],
    /// mean expected success during the failure window
    fail: f64,
    bytes: Option<StateBytes>,
    ns_per_task: f64,
}

fn run(k: &Kind, spec: &Spec, signal: Signal, shaping: &ShapingConfig, seed: u64, time_it: bool) -> RunOut {
    let w = World::new(seed, world_cfg(spec.scenario));
    let n = w.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut pol = build(k, &w);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let wc = spec.weak.clone().unwrap_or_default();
    let obs = WeakObs::new(seed, w.len(), n, &wc);
    let mut shaper = RewardShaper::new(ShapingConfig {
        use_latency: matches!(signal, Signal::Lat | Signal::LatReuse),
        immediate_only_failures: signal == Signal::Reuse,
        ..shaping.clone()
    });
    let mut pending: VecDeque<(usize, usize, usize, f64)> = VecDeque::new(); // (due, t, agent, value)
    let mut cum = 0.0;
    let mut out = RunOut::default();
    let (mut fs, mut fnn) = (0.0f64, 0.0f64);
    let t0 = Instant::now();
    for t in 0..w.len() {
        while pending.front().is_some_and(|p| p.0 <= t) {
            let (_, ti, a, v) = pending.pop_front().unwrap();
            pol.fb_delayed(&w.task(ti).task, a, v);
        }
        let st = w.task(t);
        let a = pol.pick(&st.task, &cands, &mut rng);
        let pe = w.success_prob(t, &[a]);
        let s = w.outcome(t, &[a]);
        match signal {
            Signal::Outcome => pol.fb(&st.task, a, s, Some(s)),
            _ => {
                let (u_tool, z, u_reuse, u_noise) = obs.draws[t];
                let down = w.is_down(t, a);
                let raw_ok = !down && u_tool >= wc.tool_base + wc.tool_slope * (1.0 - s);
                let raw = if raw_ok { 1.0 } else { 0.0 };
                let lat = obs.base[st.type_idx] * obs.speed[a] * (wc.sigma * z).exp() * (1.0 + wc.kappa * (1.0 - s));
                let r = match signal {
                    Signal::Raw => Some(raw),
                    _ => shaper.immediate(&st.task, a, raw_ok, Some(lat)),
                };
                pol.fb(&st.task, a, raw, r);
                if matches!(signal, Signal::Reuse | Signal::LatReuse) && raw_ok && u_reuse < wc.reuse_frac {
                    let v = if u_noise < wc.reuse_noise { 1.0 - s } else { s };
                    pending.push_back((t + wc.reuse_delay, t, a, v));
                }
            }
        }
        cum += pe;
        if w.in_failure(t) {
            fs += pe;
            fnn += 1.0;
        }
        for (i, &h) in HORIZONS.iter().enumerate() {
            if t + 1 == h {
                out.h[i] = cum / h as f64;
            }
        }
    }
    if time_it {
        out.ns_per_task = t0.elapsed().as_nanos() as f64 / w.len() as f64;
    }
    out.fail = fs / fnn.max(1.0);
    out.bytes = pol.bytes();
    out
}

fn eval(k: &Kind, spec: &Spec, sig: Signal, sh: &ShapingConfig, seeds: &[u64]) -> Vec<RunOut> {
    seeds.par_iter().map(|&s| run(k, spec, sig, sh, s, false)).collect()
}

/// tuning objective: mean over horizons and seeds
fn obj(k: &Kind, specs: &[Spec], sig: Signal, sh: &ShapingConfig, seeds: &[u64]) -> f64 {
    let mut tot = 0.0;
    for sp in specs {
        let r = eval(k, sp, sig, sh, seeds);
        tot += r.iter().map(|x| x.h.iter().sum::<f64>() / 4.0).sum::<f64>() / r.len() as f64;
    }
    tot / specs.len() as f64
}

fn col(r: &[RunOut], i: usize) -> Vec<f64> {
    r.iter().map(|x| x.h[i]).collect()
}

fn argmax_by<T>(xs: &[T], f: impl Fn(&T) -> f64) -> usize {
    let mut b = 0;
    for i in 1..xs.len() {
        if f(&xs[i]) > f(&xs[b]) {
            b = i;
        }
    }
    b
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
}

/// Compare every candidate against the per-(world, horizon) best baseline (by held-out mean).
type Arm = (String, Kind, Signal, ShapingConfig);

fn family(specs: &[Spec], bases: &[Arm], cands: &[Arm], held: &[u64], tables: &mut Vec<Value>) -> Vec<Cell> {
    let mut cells = vec![];
    for sp in specs {
        let br: Vec<Vec<RunOut>> = bases.iter().map(|(_, k, s, sh)| eval(k, sp, *s, sh, held)).collect();
        let cr: Vec<Vec<RunOut>> = cands.iter().map(|(_, k, s, sh)| eval(k, sp, *s, sh, held)).collect();
        println!("\n=== world {} | held-out 31-60 | mean expected success (prefix H) | fail-window ===", sp.name);
        println!("{:<44} {:>8} {:>8} {:>8} {:>8} {:>8}", "policy", "H100", "H250", "H500", "H2000", "fail");
        let mut rows = vec![];
        for ((name, _, s, _), r) in bases.iter().zip(&br).chain(cands.iter().zip(&cr)) {
            let m: Vec<f64> = (0..4).map(|i| summarize(&col(r, i)).mean).collect();
            let f = r.iter().map(|x| x.fail).sum::<f64>() / r.len() as f64;
            let lab = format!("{name} [{}]", s.label());
            println!("{:<44} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}", lab, m[0], m[1], m[2], m[3], f);
            rows.push(json!({"policy": name, "signal": s.label(), "h": m, "fail": f}));
        }
        tables.push(json!({"world": sp.name, "rows": rows}));
        for i in 0..4 {
            let bi = argmax_by(&br, |r| summarize(&col(r, i)).mean);
            let b = col(&br[bi], i);
            for (ci, (cname, _, _, _)) in cands.iter().enumerate() {
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
                });
            }
        }
    }
    cells
}

fn report(title: &str, cells: &[Cell]) -> Value {
    let p: Vec<f64> = cells.iter().map(|c| c.p).collect();
    let adj = holm_adjust(&p);
    println!("\n### {title}  (Holm over {} tests)", cells.len());
    println!("{:<12} {:<34} {:>5} {:>9} {:>20} {:>5} {:>8}  best baseline", "world", "candidate", "H", "diff", "95% CI", "wins", "Holm p");
    let mut out = vec![];
    for (c, a) in cells.iter().zip(&adj) {
        let mark = if *a < 0.05 { if c.diff > 0.0 { " WIN" } else { " LOSS" } } else { "" };
        println!(
            "{:<12} {:<34} {:>5} {:>+9.4} [{:>+8.4},{:>+8.4}] {:>5} {:>8.4}  {}{}",
            c.world, c.cand, c.h, c.diff, c.lo, c.hi, c.wins, a, c.best_base, mark
        );
        out.push(json!({"world": c.world, "candidate": c.cand, "H": c.h, "diff": c.diff, "ci": [c.lo, c.hi], "wins_of_30": c.wins,
            "p": c.p, "holm_p": a, "best_baseline": c.best_base, "cand_mean": c.cand_mean, "base_mean": c.base_mean}));
    }
    json!({"title": title, "cells": out})
}

// =============================================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/x3_coupling.json".into());
    let t_start = Instant::now();
    let tune: Vec<u64> = (1..=8).collect();
    // --held a-b : evaluation seeds (default 31-60; confirmation runs use 61-90)
    let held: Vec<u64> = match args.iter().position(|a| a == "--held").and_then(|i| args.get(i + 1)) {
        Some(r) => {
            let (a, b) = r.split_once('-').expect("--held a-b");
            (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
        }
        None => (31..=60).collect(),
    };
    let none = ShapingConfig::default();

    let normal: Vec<Spec> = vec![
        Spec { name: "traps".into(), scenario: Scenario::Fixed, weak: None },
        Spec { name: "correct-kw".into(), scenario: Scenario::NoTraps, weak: None },
        Spec { name: "random".into(), scenario: Scenario::Random, weak: None },
    ];
    let weak: Vec<Spec> = vec![
        Spec { name: "weak-traps".into(), scenario: Scenario::Fixed, weak: Some(WeakCfg::default()) },
        Spec { name: "weak-random".into(), scenario: Scenario::Random, weak: Some(WeakCfg::default()) },
    ];
    println!("x3_coupling: 2000 tasks/run, tune seeds 1-8, held-out 31-60, horizons {HORIZONS:?}");

    // ---- tune the warm baseline (so coupling gains are not warm-tuning gains) ------------
    let mut best_warm = (f64::NEG_INFINITY, 1.0, WarmEvidence::MeanFeature);
    for &k in &[1.0, 3.0, 10.0, 30.0] {
        for &e in &[WarmEvidence::MeanFeature, WarmEvidence::Global, WarmEvidence::FeatureSet] {
            let o = obj(&Kind::Warm(heb_robust_cfg(), k, e), &normal, Signal::Outcome, &none, &tune);
            if o > best_warm.0 {
                best_warm = (o, k, e);
            }
        }
    }
    let (wk, we) = (best_warm.1, best_warm.2);
    println!("[tune] warm(hebbian-robust): best k={wk} evidence={we:?} obj {:.4}", best_warm.0);
    let base_cc = CouplingConfig { warm_k: wk, warm_evidence: we, ..CouplingConfig::default() };

    // ---- H1 tuning: health / shield / probe gate ------------------------------------------
    let mut h1 = vec![];
    for &al in &[0.2, 0.4] {
        for &th in &[0.3, 0.5] {
            for &re in &[10u64, 20, 40] {
                for &sh in &[false, true] {
                    for &pg in &[false, true] {
                        let c = CouplingConfig { health: true, health_alpha: al, health_threshold: th, retry_every: re, shield: sh, probe_gate: pg, ..base_cc.clone() };
                        let o = obj(&Kind::Coupled(c.clone(), Precision::F64), &normal, Signal::Outcome, &none, &tune);
                        h1.push((o, c));
                    }
                }
            }
        }
    }
    h1.sort_by(|a, b| b.0.total_cmp(&a.0));
    let cc_h1 = h1[0].1.clone();
    let best_plain_health = h1.iter().find(|(_, c)| !c.shield && !c.probe_gate).unwrap().clone();
    println!(
        "[tune] H1 best obj {:.4}: alpha {} thr {} retry {} shield {} probe_gate {}  | best health-only (no exchange) {:.4}  | warm {:.4}",
        h1[0].0, cc_h1.health_alpha, cc_h1.health_threshold, cc_h1.retry_every, cc_h1.shield, cc_h1.probe_gate, best_plain_health.0, best_warm.0
    );

    // ---- H2 tuning: replay + eta-matched control -----------------------------------------
    let mut h2 = vec![];
    for &n in &[1usize, 2, 4, 8] {
        for &ms in &[0.99, 0.8, 0.6] {
            for &cap in &[0usize, 256] {
                let c = CouplingConfig { replay_n: n, replay_min_sim: ms, replay_cap: cap, ..base_cc.clone() };
                let o = obj(&Kind::Coupled(c.clone(), Precision::F64), &normal, Signal::Outcome, &none, &tune);
                h2.push((o, c));
            }
        }
    }
    h2.sort_by(|a, b| b.0.total_cmp(&a.0));
    let cc_h2 = h2[0].1.clone();
    let mut eta_ctl = (f64::NEG_INFINITY, 0.01);
    for &eta in &[0.01, 0.02, 0.03, 0.05, 0.08, 0.12] {
        let o = obj(&Kind::Warm(heb_eta(eta), wk, we), &normal, Signal::Outcome, &none, &tune);
        if o > eta_ctl.0 {
            eta_ctl = (o, eta);
        }
    }
    println!(
        "[tune] H2 best obj {:.4}: replay_n {} min_sim {} cap {}  | eta-matched control best eta {} obj {:.4}",
        h2[0].0, cc_h2.replay_n, cc_h2.replay_min_sim, cc_h2.replay_cap, eta_ctl.1, eta_ctl.0
    );
    let cc_full = CouplingConfig { replay_n: cc_h2.replay_n, replay_min_sim: cc_h2.replay_min_sim, replay_cap: cc_h2.replay_cap, ..cc_h1.clone() };
    let full_obj = obj(&Kind::Coupled(cc_full.clone(), Precision::F64), &normal, Signal::Outcome, &none, &tune);
    println!("[tune] full (H1+H2) obj {full_obj:.4}");

    // ---- H3 tuning: shaping on the weak worlds --------------------------------------------
    // brain for the weak world: health on raw (outage info), router on shaped reward.
    // Tuned jointly on seeds 1-8: front (warm-tuned | guarded), signal, agent debias,
    // logistic slope, inner eta.
    let weak_brain = CouplingConfig { health: true, ..base_cc.clone() };
    let mut h3: Vec<(f64, &'static str, Signal, CouplingConfig, ShapingConfig)> = vec![];
    for (front, gcfg) in [("warm", None), ("guarded", Some(x1_guard()))] {
        for &eta in &[0.01, 0.03] {
            let brain = CouplingConfig { guard: gcfg.clone(), hebbian: heb_eta(eta), ..weak_brain.clone() };
            for &sig in &[Signal::Raw, Signal::Lat, Signal::Reuse, Signal::LatReuse] {
                let lat = matches!(sig, Signal::Lat | Signal::LatReuse);
                for &deb in &[false, true] {
                    for &slope in &[0.75, 1.5, 3.0] {
                        if !lat && (deb || slope != 1.5) {
                            continue;
                        }
                        let sh = ShapingConfig { lat_slope: slope, debias_agent: deb, ..ShapingConfig::default() };
                        let o = obj(&Kind::Coupled(brain.clone(), Precision::F64), &weak, sig, &sh, &tune);
                        h3.push((o, front, sig, brain.clone(), sh));
                    }
                }
            }
        }
    }
    h3.sort_by(|a, b| b.0.total_cmp(&a.0));
    let best_for = |sig: Signal, front: Option<&str>| -> (f64, CouplingConfig, ShapingConfig) {
        let x = h3.iter().find(|e| e.2 == sig && front.is_none_or(|f| f == e.1)).unwrap();
        (x.0, x.3.clone(), x.4.clone())
    };
    for sig in [Signal::Raw, Signal::Lat, Signal::Reuse, Signal::LatReuse] {
        for f in ["warm", "guarded"] {
            let (o, c, sh) = best_for(sig, Some(f));
            println!(
                "[tune] H3 {:<9} front {:<8} obj {o:.4}  eta {} debias {} slope {}",
                sig.label(), f, c.hebbian.eta, sh.debias_agent, sh.lat_slope
            );
        }
    }
    let raw_obj = best_for(Signal::Raw, Some("warm")).0;
    let sh_best = best_for(Signal::LatReuse, None).2;

    // ---- held-out families ----------------------------------------------------------------
    let o = Signal::Outcome;
    let nn = || none.clone();
    let bases: Vec<Arm> = vec![
        ("keyword".into(), Kind::Keyword, o, nn()),
        ("eps-greedy".into(), Kind::Eps, o, nn()),
        ("pooled-ts".into(), Kind::Pts, o, nn()),
        ("hebbian-robust".into(), Kind::HebR, o, nn()),
        ("warm(hebr,k1,MeanFeature)".into(), Kind::Warm(heb_robust_cfg(), 1.0, WarmEvidence::MeanFeature), o, nn()),
        (format!("warm(hebr,k{wk},{we:?})"), Kind::Warm(heb_robust_cfg(), wk, we), o, nn()),
        ("mesh-robust".into(), Kind::MeshR, o, nn()),
    ];
    let cc_gh1 = CouplingConfig { guard: Some(x1_guard()), ..cc_h1.clone() };
    let cands: Vec<Arm> = vec![
        ("coupled-H1(health+exchange)".into(), Kind::Coupled(cc_h1.clone(), Precision::F64), o, nn()),
        ("coupled-H1-noexchange(ctl)".into(), Kind::Coupled(best_plain_health.1.clone(), Precision::F64), o, nn()),
        ("coupled-H2(replay)".into(), Kind::Coupled(cc_h2.clone(), Precision::F64), o, nn()),
        (format!("warm eta={} (eta-matched ctl)", eta_ctl.1), Kind::Warm(heb_eta(eta_ctl.1), wk, we), o, nn()),
        ("coupled-full(H1+H2)".into(), Kind::Coupled(cc_full.clone(), Precision::F64), o, nn()),
        ("guarded(x1, reference)".into(), Kind::Guard(x1_guard()), o, nn()),
        ("guarded+H1(composition)".into(), Kind::Coupled(cc_gh1.clone(), Precision::F64), o, nn()),
    ];
    let mut tables = vec![];
    let fam_n = family(&normal, &bases, &cands, &held, &mut tables);
    let rep_n = report("FAMILY N: normal-feedback worlds, candidates vs best baseline", &fam_n);

    // direct ablations (paired, held-out): what does each coupling add to its own base?
    println!("\n=== direct ablations (held-out 31-60): diff [95% CI] per H ===");
    let mut abl = vec![];
    let pairs: Vec<(&str, Kind, Kind)> = vec![
        ("H1 exchange: coupled-H1 - coupled-H1-noexchange", Kind::Coupled(cc_h1.clone(), Precision::F64), Kind::Coupled(best_plain_health.1.clone(), Precision::F64)),
        ("H1 total: coupled-H1 - warm(tuned)", Kind::Coupled(cc_h1.clone(), Precision::F64), Kind::Warm(heb_robust_cfg(), wk, we)),
        ("H2 replay: coupled-H2 - warm(tuned)", Kind::Coupled(cc_h2.clone(), Precision::F64), Kind::Warm(heb_robust_cfg(), wk, we)),
        ("H2 replay vs eta-matched", Kind::Coupled(cc_h2.clone(), Precision::F64), Kind::Warm(heb_eta(eta_ctl.1), wk, we)),
        ("composition: guarded+H1 - guarded", Kind::Coupled(cc_gh1.clone(), Precision::F64), Kind::Guard(x1_guard())),
    ];
    let mut abl_p = vec![];
    for (lab, ka, kb) in &pairs {
        for sp in &normal {
            let ra = eval(ka, sp, Signal::Outcome, &none, &held);
            let rb = eval(kb, sp, Signal::Outcome, &none, &held);
            print!("{lab:<50} {:<11}", sp.name);
            let mut ds = vec![];
            for i in 0..4 {
                let d = paired_diff(&col(&ra, i), &col(&rb, i));
                let p = paired_t_pvalue(&col(&ra, i), &col(&rb, i));
                abl_p.push(p);
                print!(" H{}: {:+.4}[{:+.4},{:+.4}]", HORIZONS[i], d.diff.mean, d.diff.lo, d.diff.hi);
                ds.push(json!({"H": HORIZONS[i], "diff": d.diff.mean, "ci": [d.diff.lo, d.diff.hi], "p": p}));
            }
            let fa = ra.iter().map(|x| x.fail).collect::<Vec<_>>();
            let fb = rb.iter().map(|x| x.fail).collect::<Vec<_>>();
            let df = paired_diff(&fa, &fb);
            let pf = paired_t_pvalue(&fa, &fb);
            abl_p.push(pf);
            println!("  fail-window {:+.4}[{:+.4},{:+.4}]", df.diff.mean, df.diff.lo, df.diff.hi);
            abl.push(json!({"ablation": lab, "world": sp.name, "h": ds, "fail_window": [df.diff.mean, df.diff.lo, df.diff.hi], "fail_p": pf}));
        }
    }
    let abl_adj = holm_adjust(&abl_p);
    let mut k = 0;
    for e in abl.iter_mut() {
        for h in e["h"].as_array_mut().unwrap() {
            h["holm_p"] = json!(abl_adj[k]);
            k += 1;
        }
        e["fail_holm_p"] = json!(abl_adj[k]);
        k += 1;
    }
    for e in &abl {
        let hs: Vec<String> = e["h"].as_array().unwrap().iter().map(|h| format!("H{} {:+.4} p{:.3}", h["H"], h["diff"].as_f64().unwrap(), h["holm_p"].as_f64().unwrap())).collect();
        println!("[holm] {:<50} {:<11} {}  fail {:+.4} p{:.4}", e["ablation"].as_str().unwrap(), e["world"].as_str().unwrap(), hs.join("  "), e["fail_window"][0].as_f64().unwrap(), e["fail_holm_p"].as_f64().unwrap());
    }

    // weak worlds: baselines get the raw hook signal (the real ruflo situation)
    let mut bases_w: Vec<Arm> = bases.iter().map(|(n, k, _, sh)| (n.clone(), k.clone(), Signal::Raw, sh.clone())).collect();
    bases_w.push(("guarded(x1)".into(), Kind::Guard(x1_guard()), Signal::Raw, nn()));
    let arm = |name: &str, sig: Signal, front: Option<&str>| -> Arm {
        let (_, c, sh) = best_for(sig, front);
        let lab = format!("{name}[{}]", if c.guard.is_some() { "guarded" } else { "warm" });
        (lab, Kind::Coupled(c, Precision::F64), sig, sh)
    };
    let cands_w: Vec<Arm> = vec![
        arm("coupled+raw(ctl)", Signal::Raw, None),
        arm("coupled+shaped(lat)", Signal::Lat, None),
        arm("coupled+shaped(reuse)", Signal::Reuse, None),
        arm("coupled+shaped(lat+reuse)", Signal::LatReuse, None),
        arm("coupled+shaped(lat+reuse)", Signal::LatReuse, Some(if best_for(Signal::LatReuse, None).1.guard.is_some() { "warm" } else { "guarded" })),
    ];
    let fam_w = family(&weak, &bases_w, &cands_w, &held, &mut tables);
    let rep_w = report("FAMILY W: weak-feedback worlds, shaped coupled brain vs best baseline fed the raw hook", &fam_w);

    // ---- H3 diagnostics: every learner x every signal (held-out, weak-traps & weak-random) --
    println!("\n=== H3 signal x learner matrix (held-out mean expected success, H500 / H2000) ===");
    let learners: Vec<(String, Kind)> = vec![
        ("keyword".into(), Kind::Keyword),
        ("eps-greedy".into(), Kind::Eps),
        ("pooled-ts".into(), Kind::Pts),
        ("hebbian-robust".into(), Kind::HebR),
        (format!("warm(hebr,k{wk})"), Kind::Warm(heb_robust_cfg(), wk, we)),
        ("mesh-robust".into(), Kind::MeshR),
        ("guarded(x1)".into(), Kind::Guard(x1_guard())),
        ("coupled(health,warm)".into(), Kind::Coupled(weak_brain.clone(), Precision::F64)),
        ("coupled(health,guarded)".into(), Kind::Coupled(CouplingConfig { guard: Some(x1_guard()), ..weak_brain.clone() }, Precision::F64)),
    ];
    let sigs = [Signal::Outcome, Signal::Raw, Signal::Lat, Signal::Reuse, Signal::LatReuse];
    let mut matrix = vec![];
    for sp in &weak {
        println!("-- {} --", sp.name);
        print!("{:<22}", "learner");
        for s in &sigs {
            print!(" {:>17}", s.label());
        }
        println!();
        for (ln, k) in &learners {
            print!("{:<22}", ln);
            let mut row = vec![];
            for &s in &sigs {
                let r = eval(k, sp, s, &sh_best, &held);
                let m500 = summarize(&col(&r, 2)).mean;
                let m2k = summarize(&col(&r, 3)).mean;
                print!(" {:>8.4}/{:>8.4}", m500, m2k);
                row.push(json!({"signal": s.label(), "h": (0..4).map(|i| summarize(&col(&r, i)).mean).collect::<Vec<_>>()}));
            }
            println!();
            matrix.push(json!({"world": sp.name, "learner": ln, "cells": row}));
        }
    }
    // paired: each learner shaped(lat+reuse) vs same learner raw, and raw vs keyword (can raw learn at all?)
    println!("\n=== H3 paired: signal effect per learner (weak-random, H2000, held-out) ===");
    let mut sig_eff = vec![];
    let mut p_sig = vec![];
    for sp in &weak {
        let kwr = eval(&Kind::Keyword, sp, Signal::Raw, &sh_best, &held);
        for (ln, k) in learners.iter().skip(1) {
            let raw = eval(k, sp, Signal::Raw, &sh_best, &held);
            let shp = eval(k, sp, Signal::LatReuse, &sh_best, &held);
            let orc = eval(k, sp, Signal::Outcome, &sh_best, &held);
            let d_rk = paired_diff(&col(&raw, 3), &col(&kwr, 3));
            let d_sr = paired_diff(&col(&shp, 3), &col(&raw, 3));
            let gap_closed = {
                let (r, s, o) = (summarize(&col(&raw, 3)).mean, summarize(&col(&shp, 3)).mean, summarize(&col(&orc, 3)).mean);
                if (o - r).abs() > 1e-9 { (s - r) / (o - r) } else { f64::NAN }
            };
            p_sig.push(paired_t_pvalue(&col(&shp, 3), &col(&raw, 3)));
            println!(
                "{:<12} {:<22} raw-keyword {:+.4} [{:+.4},{:+.4}]   shaped-raw {:+.4} [{:+.4},{:+.4}]   gap-to-outcome closed {:.0}%",
                sp.name, ln, d_rk.diff.mean, d_rk.diff.lo, d_rk.diff.hi, d_sr.diff.mean, d_sr.diff.lo, d_sr.diff.hi, 100.0 * gap_closed
            );
            sig_eff.push(json!({"world": sp.name, "learner": ln, "raw_minus_keyword": [d_rk.diff.mean, d_rk.diff.lo, d_rk.diff.hi],
                "shaped_minus_raw": [d_sr.diff.mean, d_sr.diff.lo, d_sr.diff.hi], "gap_closed_frac": gap_closed}));
        }
    }
    let adj_sig = holm_adjust(&p_sig);
    for (e, a) in sig_eff.iter_mut().zip(&adj_sig) {
        e["holm_p_shaped_vs_raw"] = json!(a);
    }

    // ---- H3 sensitivity: how strong must the latency signal be? ---------------------------
    println!("\n=== H3 sensitivity (weak-random, coupled, H2000, held-out): shaped(lat) - raw ===");
    let mut sens = vec![];
    for &kappa in &[0.0, 0.2, 0.6, 1.0] {
        for &spread in &[0.0, 0.3, 0.6] {
            let sp = Spec { name: "weak-random".into(), scenario: Scenario::Random, weak: Some(WeakCfg { kappa, speed_spread: spread, ..WeakCfg::default() }) };
            let (_, cl, shl) = best_for(Signal::Lat, None);
            let k = Kind::Coupled(cl, Precision::F64);
            let raw = eval(&Kind::Coupled(best_for(Signal::Raw, None).1, Precision::F64), &sp, Signal::Raw, &sh_best, &held);
            let lat = eval(&k, &sp, Signal::Lat, &shl, &held);
            let d = paired_diff(&col(&lat, 3), &col(&raw, 3));
            println!("kappa {kappa:.1} speed-spread {spread:.1}: {:+.4} [{:+.4},{:+.4}] (raw {:.4}, lat {:.4})", d.diff.mean, d.diff.lo, d.diff.hi, summarize(&col(&raw, 3)).mean, summarize(&col(&lat, 3)).mean);
            sens.push(json!({"kappa": kappa, "speed_spread": spread, "diff": [d.diff.mean, d.diff.lo, d.diff.hi]}));
        }
    }
    for &rf in &[0.05, 0.1, 0.3, 0.6] {
        let sp = Spec { name: "weak-random".into(), scenario: Scenario::Random, weak: Some(WeakCfg { reuse_frac: rf, ..WeakCfg::default() }) };
        let (_, cr, shr) = best_for(Signal::Reuse, None);
        let raw = eval(&Kind::Coupled(best_for(Signal::Raw, None).1, Precision::F64), &sp, Signal::Raw, &sh_best, &held);
        let reu = eval(&Kind::Coupled(cr, Precision::F64), &sp, Signal::Reuse, &shr, &held);
        let d = paired_diff(&col(&reu, 3), &col(&raw, 3));
        println!("reuse fraction {rf:.2}: shaped(reuse) - raw {:+.4} [{:+.4},{:+.4}]", d.diff.mean, d.diff.lo, d.diff.hi);
        sens.push(json!({"reuse_frac": rf, "diff": [d.diff.mean, d.diff.lo, d.diff.hi]}));
    }

    // ---- H4 memory budget -------------------------------------------------------------------
    println!("\n=== H4 memory budget & latency (world random, held-out) ===");
    let rnd = &normal[2];
    let mut mem_rows = vec![];
    let timed: Vec<(String, Kind)> = vec![
        ("keyword".into(), Kind::Keyword),
        ("hebbian-robust".into(), Kind::HebR),
        (format!("warm(hebr,k{wk})"), Kind::Warm(heb_robust_cfg(), wk, we)),
        ("mesh-robust".into(), Kind::MeshR),
        ("coupled-H1".into(), Kind::Coupled(cc_h1.clone(), Precision::F64)),
        ("coupled-full".into(), Kind::Coupled(cc_full.clone(), Precision::F64)),
    ];
    for (n, k) in &timed {
        // single-threaded timing, 5 seeds
        let ns: Vec<f64> = held.iter().take(5).map(|&s| run(k, rnd, Signal::Outcome, &none, s, true).ns_per_task).collect();
        let r = run(k, rnd, Signal::Outcome, &none, 31, false);
        println!("{:<22} {:>9.0} ns/task(route+feedback)  bytes {:?}", n, summarize(&ns).mean, r.bytes);
        mem_rows.push(json!({"policy": n, "ns_per_task": summarize(&ns).mean, "bytes": r.bytes}));
    }
    // precision + memory cap ablation of the full brain
    let full_f64 = eval(&Kind::Coupled(cc_full.clone(), Precision::F64), rnd, Signal::Outcome, &none, &held);
    let mut prec = vec![];
    for (lab, k) in [
        ("f32", Kind::Coupled(cc_full.clone(), Precision::F32)),
        ("u16", Kind::Coupled(cc_full.clone(), Precision::U16)),
        ("u8", Kind::Coupled(cc_full.clone(), Precision::U8)),
        ("replay cap 64", Kind::Coupled(CouplingConfig { replay_cap: 64, ..cc_full.clone() }, Precision::F64)),
        ("replay cap 256", Kind::Coupled(CouplingConfig { replay_cap: 256, ..cc_full.clone() }, Precision::F64)),
        ("replay dim 16 cap 64 + u16", Kind::Coupled(CouplingConfig { replay_cap: 64, replay_dim: 16, ..cc_full.clone() }, Precision::U16)),
    ] {
        let r = eval(&k, rnd, Signal::Outcome, &none, &held);
        let mut ds = vec![];
        for i in [0, 3] {
            let d = paired_diff(&col(&r, i), &col(&full_f64, i));
            ds.push(json!({"H": HORIZONS[i], "diff": [d.diff.mean, d.diff.lo, d.diff.hi]}));
            print!("{lab:<28} H{:<5} vs f64 {:+.5} [{:+.5},{:+.5}]   ", HORIZONS[i], d.diff.mean, d.diff.lo, d.diff.hi);
        }
        let b = r[0].bytes.clone();
        println!("bytes {:?}", b.as_ref().map(|b| (b.router_f64, b.router_f32, b.router_sparse_f32, b.memory)));
        prec.push(json!({"variant": lab, "diffs": ds, "bytes": b}));
    }

    let res = json!({
        "elapsed_s": t_start.elapsed().as_secs_f64(),
        "protocol": {"tune_seeds": "1-8", "heldout": format!("{}-{}", held[0], held[held.len() - 1]), "horizons": HORIZONS, "metric": "prefix mean expected success of routed agent", "n_tasks": 2000},
        "weak_model": format!("{:?}", WeakCfg::default()),
        "types": TYPES,
        "tuning": {
            "warm": {"k": wk, "evidence": format!("{we:?}"), "obj": best_warm.0},
            "h1_best": {"obj": h1[0].0, "cfg": format!("{:?}", cc_h1)},
            "h1_top5": h1.iter().take(5).map(|(o, c)| json!({"obj": o, "alpha": c.health_alpha, "thr": c.health_threshold, "retry": c.retry_every, "shield": c.shield, "probe_gate": c.probe_gate})).collect::<Vec<_>>(),
            "h1_best_no_exchange": {"obj": best_plain_health.0},
            "h2_best": {"obj": h2[0].0, "replay_n": cc_h2.replay_n, "min_sim": cc_h2.replay_min_sim, "cap": cc_h2.replay_cap},
            "h2_top5": h2.iter().take(5).map(|(o, c)| json!({"obj": o, "n": c.replay_n, "min_sim": c.replay_min_sim, "cap": c.replay_cap})).collect::<Vec<_>>(),
            "eta_control": {"eta": eta_ctl.1, "obj": eta_ctl.0},
            "full_obj": full_obj,
            "h3": h3.iter().map(|(o, f, s, c, sh)| json!({"front": f, "signal": s.label(), "eta": c.hebbian.eta, "debias": sh.debias_agent, "slope": sh.lat_slope, "obj": o})).collect::<Vec<_>>(),
            "h3_raw_obj": raw_obj,
        },
        "tables": tables,
        "family_normal": rep_n,
        "ablations_normal": abl,
        "family_weak": rep_w,
        "h3_matrix": matrix,
        "h3_signal_effect": sig_eff,
        "h3_sensitivity": sens,
        "h4_memory": mem_rows,
        "h4_precision": prec,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&res).unwrap()).expect("write results");
    println!("\nwrote {out_path} in {:.1}s", t_start.elapsed().as_secs_f64());
}
