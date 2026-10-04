//! D4 (ROUTER-ADDON): graded-reward shaper from per-task proxies, confidence/abstain and
//! calibration, as wrappers around ANY router (src/addons.rs).
//!
//! Weak-feedback worlds (sim.rs Fixed/Random + my observation models): the hook reports only
//! the tool-ok bit (~97-99% ok); every task additionally exposes response proxies
//! (retry count, latency, length). Four observation models:
//!   lat-only    X3's world (only latency carries quality; per-topic base + per-agent speed confound)
//!   rich        retries + latency + length all carry quality (noisy)
//!   confounded  as rich with strong per-agent speed/style/retry-rate confounds
//!   useless     proxies are pure noise (no-harm world)
//! Protocol: tune on seeds 1-8 (shaper center/gain/debias/factor per learner), held-out 31-60,
//! confirm 61-90 (--held 61-90, frozen tuning), horizons 100/250/500/2000, paired 95% t-CI, Holm
//! over candidates x worlds x horizons, vs the best fair RAW-feedback baseline per (world, H):
//! {keyword, eps-greedy, pooled-ts, hebbian-robust, warm(hebr) k=1/k=10, guarded(hebr) x1}.
//! The shaper uses only signals present on EVERY task (no delayed reuse signal).
//!
//! Usage: cargo run --release --bin d4_addons [-- --held 31-60] [--out results/d4_addons.json]

use std::collections::HashMap;
use std::time::Instant;

use neural_mesh::addons::{
    Abstain, AbstainConfig, Channel, ConfCalibrator, Shaped, ShaperConfig, Transform,
};
use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, KeywordPrior, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

const HORIZONS: [usize; 4] = [100, 250, 500, 2000];

fn heb_robust() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}
fn x1_guard() -> GuardConfig {
    GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
}

// =============================================================================================
// Observation models

#[derive(Clone, Debug)]
struct ObsCfg {
    name: &'static str,
    kappa: f64,
    lat_sigma: f64,
    speed_spread: f64,
    tool_base: f64,
    tool_slope: f64,
    /// Some((r0, r1)): retries ~ Poisson((r0 + r1 (1-s)) * agent_factor)
    retry: Option<(f64, f64)>,
    retry_agent_spread: f64,
    /// Some((slope, sigma, style_spread)): ln len = base + style - slope (1-s) + sigma z
    len: Option<(f64, f64, f64)>,
}

fn obs_models() -> Vec<ObsCfg> {
    let b = ObsCfg {
        name: "lat-only", kappa: 0.6, lat_sigma: 0.35, speed_spread: 0.3, tool_base: 0.01, tool_slope: 0.03,
        retry: None, retry_agent_spread: 0.0, len: None,
    };
    vec![
        b.clone(),
        ObsCfg { name: "rich", retry: Some((0.15, 1.2)), len: Some((0.6, 0.5, 0.3)), ..b.clone() },
        ObsCfg {
            name: "confounded", speed_spread: 0.6, retry: Some((0.15, 1.2)), retry_agent_spread: 0.6, len: Some((0.6, 0.5, 0.8)), ..b.clone()
        },
        ObsCfg { name: "useless", kappa: 0.0, retry: Some((0.4, 0.0)), len: Some((0.0, 0.5, 0.3)), ..b.clone() },
    ]
}

fn normal(r: &mut Rng) -> f64 {
    let u1 = r.f64().max(1e-12);
    let u2 = r.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn poisson_inv(lambda: f64, u: f64) -> f64 {
    let (mut k, mut p) = (0.0, (-lambda).exp());
    let mut cdf = p;
    while u > cdf && k < 20.0 {
        k += 1.0;
        p *= lambda / k;
        cdf += p;
    }
    k
}

struct Obs {
    base_lat: [f64; 8],
    base_len: [f64; 8],
    speed: Vec<f64>,
    style: Vec<f64>,
    ragent: Vec<f64>,
    /// (u_tool, z_lat, z_len, u_retry)
    draws: Vec<(f64, f64, f64, f64)>,
}

impl Obs {
    fn new(seed: u64, n: usize, na: usize, c: &ObsCfg) -> Self {
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(0xD4D4));
        let mut base_lat = [0.0; 8];
        let mut base_len = [0.0; 8];
        for i in 0..8 {
            base_lat[i] = (r.f64() * 2.0 - 0.7).exp();
            base_len[i] = (r.f64() * 2.0 - 0.7).exp() * 100.0;
        }
        let speed = (0..na).map(|_| 1.0 - c.speed_spread / 2.0 + c.speed_spread * r.f64()).collect();
        let style = (0..na).map(|_| (c.len.map(|l| l.2).unwrap_or(0.0) * (r.f64() - 0.5)).exp()).collect();
        let ragent = (0..na).map(|_| 1.0 - c.retry_agent_spread / 2.0 + c.retry_agent_spread * r.f64()).collect();
        let draws = (0..n).map(|_| (r.f64(), normal(&mut r), normal(&mut r), r.f64())).collect();
        Obs { base_lat, base_len, speed, style, ragent, draws }
    }
}

// =============================================================================================
// Routers

struct DynR(Box<dyn Router>);
impl Router for DynR {
    fn name(&self) -> String {
        self.0.name()
    }
    fn select(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(t, c, rng)
    }
    fn update(&mut self, t: &Task, a: usize, r: f64) {
        self.0.update(t, a, r)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm1,
    Warm10,
    Guard,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Keyword => "keyword",
            Kind::Eps => "eps-greedy",
            Kind::Pts => "pooled-ts",
            Kind::HebR => "hebbian-robust",
            Kind::Warm1 => "warm(hebr) k1",
            Kind::Warm10 => "warm(hebr) k10",
            Kind::Guard => "guarded(hebr)",
        }
    }
}

fn build(k: Kind, w: &World) -> DynR {
    let n = w.n_agents();
    let kw = w.keyword_map_for();
    DynR(match k {
        Kind::Keyword => Box::new(KeywordRouter::new(kw, n)),
        Kind::Eps => Box::new(EpsGreedyRouter::new(n, 0.99, 0.02)),
        Kind::Pts => Box::new(PooledThompsonRouter::new(n, 1.0)),
        Kind::HebR => Box::new(HebbianRouter::new(n, heb_robust())),
        Kind::Warm1 => Box::new(Warm::new(HebbianRouter::new(n, heb_robust()), kw, n, 1.0).with_evidence(Evidence::MeanFeature)),
        Kind::Warm10 => Box::new(Warm::new(HebbianRouter::new(n, heb_robust()), kw, n, 10.0).with_evidence(Evidence::MeanFeature)),
        Kind::Guard => Box::new(Guarded::new(HebbianRouter::new(n, heb_robust()), kw, n, x1_guard())),
    })
}

/// shaper parameters (tuned per learner on seeds 1-8)
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sp {
    center: f64,
    gain: f64,
    debias_agent: f64,
    factor: bool,
}

fn shaper_cfg(sp: Option<Sp>) -> ShaperConfig {
    let Some(sp) = sp else { return ShaperConfig::off() };
    let ch = |name: &str, tr, dir, df, w| Channel {
        name: name.into(), transform: tr, dir, debias_feature: df, debias_agent: if df { sp.debias_agent } else { 0.0 }, weight: w,
    };
    ShaperConfig {
        enabled: true,
        channels: vec![
            ch("retry", Transform::Log1p, -1, false, 1.0),
            ch("lat", Transform::Ln, -1, true, 1.0),
            ch("len", Transform::Ln, 0, true, 1.0),
        ],
        center: sp.center,
        gain: sp.gain,
        fail_reward: 0.0,
        factor: sp.factor,
        warmup: 100,
        alpha_min: 0.005,
    }
}

// =============================================================================================
// Worlds / runs

#[derive(Clone, Debug)]
struct Spec {
    name: String,
    scenario: Scenario,
    obs: Option<ObsCfg>,
}

fn world_cfg(s: Scenario) -> WorldConfig {
    WorldConfig { n_tasks: 2000, scenario: s, ..WorldConfig::default() }
}

#[derive(Clone, Debug, Default)]
struct RunOut {
    h: [f64; 4],
    fail: f64,
}

/// Mode: None = raw feedback (weak worlds) / outcome feedback (normal worlds); Some = shaped.
fn run(k: Kind, sp: Option<Sp>, spec: &Spec, seed: u64) -> RunOut {
    let w = World::new(seed, world_cfg(spec.scenario));
    let n = w.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut pol = Shaped::new(build(k, &w), shaper_cfg(sp));
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let oc = spec.obs.clone();
    let obs = oc.as_ref().map(|c| Obs::new(seed, w.len(), n, c));
    let (mut cum, mut fs, mut fnn) = (0.0, 0.0, 0.0f64);
    let mut out = RunOut::default();
    for t in 0..w.len() {
        let st = w.task(t);
        let a = pol.select(&st.task, &cands, &mut rng);
        let pe = w.success_prob(t, &[a]);
        match (&oc, &obs) {
            (Some(c), Some(o)) => {
                let s = w.outcome(t, &[a]);
                let (u_tool, zl, zn, ur) = o.draws[t];
                let ok = !w.is_down(t, a) && u_tool >= c.tool_base + c.tool_slope * (1.0 - pe.min(1.0));
                let lat = o.base_lat[st.type_idx] * o.speed[a] * (c.lat_sigma * zl).exp() * (1.0 + c.kappa * (1.0 - pe));
                let retry = c.retry.map(|(r0, r1)| poisson_inv(((r0 + r1 * (1.0 - pe)) * o.ragent[a]).max(1e-6), ur));
                let len = c.len.map(|(sl, sg, _)| o.base_len[st.type_idx] * o.style[a] * (sg * zn - sl * (1.0 - pe)).exp());
                let _ = s;
                // channels: [retry, lat, len]; lat-only world exposes only latency
                let vals = [retry, Some(lat), len];
                pol.observe(&st.task, a, ok, &vals);
            }
            _ => {
                let s = w.outcome(t, &[a]);
                pol.update(&st.task, a, s);
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
    out.fail = fs / fnn.max(1.0);
    out
}

fn eval(k: Kind, sp: Option<Sp>, spec: &Spec, seeds: &[u64]) -> Vec<RunOut> {
    seeds.par_iter().map(|&s| run(k, sp, spec, s)).collect()
}
fn col(r: &[RunOut], i: usize) -> Vec<f64> {
    r.iter().map(|x| x.h[i]).collect()
}
fn obj(k: Kind, sp: Option<Sp>, specs: &[Spec], seeds: &[u64]) -> f64 {
    let mut tot = 0.0;
    for s in specs {
        let r = eval(k, sp, s, seeds);
        tot += r.iter().map(|x| x.h.iter().sum::<f64>() / 4.0).sum::<f64>() / r.len() as f64;
    }
    tot / specs.len() as f64
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
// Abstain / calibration evaluation

#[derive(Clone, Debug, Default)]
struct AbOut {
    /// mean expected success of what is finally executed
    final_p: f64,
    /// mean expected success of the inner proposal (counterfactual, = executed when no abstain)
    inner_p: f64,
    /// abstain rate
    rate: f64,
    /// selective risk (1 - mean inner p) at coverage 25/50/75/100 (top confidence first)
    risk: [f64; 4],
    /// expected calibration error of confidence vs true p of the inner proposal
    ece: f64,
    /// spearman-ish: mean p of top-half confidence minus bottom-half
    spread: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_abstain(k: Kind, sp: Option<Sp>, spec: &Spec, seed: u64, acfg: AbstainConfig, calib: bool) -> AbOut {
    let w = World::new(seed, world_cfg(spec.scenario));
    let n = w.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let ab = Abstain::new(build(k, &w), KeywordPrior::new(w.keyword_map_for(), n), acfg, ConfCalibrator::new(calib, 0.02));
    let mut pol = Shaped::new(ab, shaper_cfg(sp));
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let oc = spec.obs.clone();
    let obs = oc.as_ref().map(|c| Obs::new(seed, w.len(), n, c));
    let (mut fin, mut inn, mut nab) = (0.0, 0.0, 0.0);
    let mut recs: Vec<(f64, f64)> = vec![];
    for t in 0..w.len() {
        let st = w.task(t);
        let a = pol.select(&st.task, &cands, &mut rng);
        let info = pol.inner.last;
        let pe = w.success_prob(t, &[a]);
        let pi = w.success_prob(t, &[info.inner_pick]);
        fin += pe;
        inn += pi;
        if info.abstained {
            nab += 1.0;
        }
        if t >= 100 {
            recs.push((info.conf, pi));
        }
        match (&oc, &obs) {
            (Some(c), Some(o)) => {
                let (u_tool, zl, zn, ur) = o.draws[t];
                let ok = !w.is_down(t, a) && u_tool >= c.tool_base + c.tool_slope * (1.0 - pe);
                let lat = o.base_lat[st.type_idx] * o.speed[a] * (c.lat_sigma * zl).exp() * (1.0 + c.kappa * (1.0 - pe));
                let retry = c.retry.map(|(r0, r1)| poisson_inv(((r0 + r1 * (1.0 - pe)) * o.ragent[a]).max(1e-6), ur));
                let len = c.len.map(|(sl, sg, _)| o.base_len[st.type_idx] * o.style[a] * (sg * zn - sl * (1.0 - pe)).exp());
                pol.observe(&st.task, a, ok, &[retry, Some(lat), len]);
            }
            _ => {
                let s = w.outcome(t, &[a]);
                pol.update(&st.task, a, s);
            }
        }
    }
    let len = w.len() as f64;
    let mut out = AbOut { final_p: fin / len, inner_p: inn / len, rate: nab / len, ..Default::default() };
    recs.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap_or(std::cmp::Ordering::Equal));
    for (i, cov) in [0.25, 0.5, 0.75, 1.0].iter().enumerate() {
        let m = ((recs.len() as f64 * cov) as usize).max(1);
        out.risk[i] = 1.0 - recs[..m].iter().map(|r| r.1).sum::<f64>() / m as f64;
    }
    let h = recs.len() / 2;
    out.spread = recs[..h].iter().map(|r| r.1).sum::<f64>() / h as f64 - recs[h..].iter().map(|r| r.1).sum::<f64>() / (recs.len() - h) as f64;
    // ECE, 10 equal-width bins of conf
    let mut bins = [(0.0, 0.0, 0.0); 10];
    for r in &recs {
        let b = ((r.0 * 10.0) as usize).min(9);
        bins[b].0 += r.0;
        bins[b].1 += r.1;
        bins[b].2 += 1.0;
    }
    out.ece = bins.iter().filter(|b| b.2 > 0.0).map(|b| (b.0 - b.1).abs()).sum::<f64>() / recs.len() as f64;
    out
}

// =============================================================================================

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

type Arm = (String, Kind, Option<Sp>);

fn family(specs: &[Spec], bases: &[Arm], cands: &[Arm], held: &[u64], tables: &mut Vec<Value>) -> Vec<Cell> {
    let mut cells = vec![];
    for sp in specs {
        let br: Vec<Vec<RunOut>> = bases.iter().map(|(_, k, s)| eval(*k, *s, sp, held)).collect();
        let cr: Vec<Vec<RunOut>> = cands.iter().map(|(_, k, s)| eval(*k, *s, sp, held)).collect();
        println!("\n=== world {} | mean expected success (prefix H) | fail-window ===", sp.name);
        println!("{:<34} {:>8} {:>8} {:>8} {:>8} {:>8}", "policy", "H100", "H250", "H500", "H2000", "fail");
        let mut rows = vec![];
        for ((name, _, _), r) in bases.iter().zip(&br).chain(cands.iter().zip(&cr)) {
            let m: Vec<f64> = (0..4).map(|i| summarize(&col(r, i)).mean).collect();
            let f = r.iter().map(|x| x.fail).sum::<f64>() / r.len() as f64;
            println!("{:<34} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}", name, m[0], m[1], m[2], m[3], f);
            rows.push(json!({"policy": name, "h": m, "fail": f}));
        }
        tables.push(json!({"world": sp.name, "rows": rows}));
        for i in 0..4 {
            let bi = argmax_by(&br, |r| summarize(&col(r, i)).mean);
            let b = col(&br[bi], i);
            for (ci, (cname, _, _)) in cands.iter().enumerate() {
                let c = col(&cr[ci], i);
                let d = paired_diff(&c, &b);
                cells.push(Cell {
                    world: sp.name.clone(), cand: cname.clone(), h: HORIZONS[i], diff: d.diff.mean, lo: d.diff.lo, hi: d.diff.hi,
                    wins: d.wins, p: paired_t_pvalue(&c, &b), best_base: bases[bi].0.clone(), cand_mean: summarize(&c).mean,
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
    println!("{:<20} {:<24} {:>5} {:>9} {:>20} {:>5} {:>8}  best baseline", "world", "candidate", "H", "diff", "95% CI", "wins", "Holm p");
    let mut out = vec![];
    let (mut nwin, mut nloss) = (0, 0);
    for (c, a) in cells.iter().zip(&adj) {
        let mark = if *a < 0.05 {
            if c.diff > 0.0 { nwin += 1; " WIN" } else { nloss += 1; " LOSS" }
        } else { "" };
        println!("{:<20} {:<24} {:>5} {:>+9.4} [{:>+8.4},{:>+8.4}] {:>5} {:>8.4}  {}{}", c.world, c.cand, c.h, c.diff, c.lo, c.hi, c.wins, a, c.best_base, mark);
        out.push(json!({"world": c.world, "candidate": c.cand, "H": c.h, "diff": c.diff, "ci": [c.lo, c.hi], "wins_of_30": c.wins,
            "p": c.p, "holm_p": a, "best_baseline": c.best_base, "cand_mean": c.cand_mean, "base_mean": c.base_mean}));
    }
    println!("  -> Holm WINS {nwin}, Holm LOSSES {nloss}");
    json!({"title": title, "holm_wins": nwin, "holm_losses": nloss, "cells": out})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/d4_addons.json".into());
    let t0 = Instant::now();
    let tune: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = match args.iter().position(|a| a == "--held").and_then(|i| args.get(i + 1)) {
        Some(r) => {
            let (a, b) = r.split_once('-').expect("--held a-b");
            (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
        }
        None => (31..=60).collect(),
    };

    let mut weak: Vec<Spec> = vec![];
    for sc in [Scenario::Fixed, Scenario::Random] {
        for o in obs_models() {
            weak.push(Spec { name: format!("{}-{}", sc.label(), o.name), scenario: sc, obs: Some(o) });
        }
    }
    let normal: Vec<Spec> = vec![
        Spec { name: "n-traps".into(), scenario: Scenario::Fixed, obs: None },
        Spec { name: "n-correct-kw".into(), scenario: Scenario::NoTraps, obs: None },
        Spec { name: "n-random".into(), scenario: Scenario::Random, obs: None },
    ];
    // tuning set: informative worlds only (useless excluded: tuning must not need the answer)
    let tune_specs: Vec<Spec> = weak.iter().filter(|s| !s.name.ends_with("useless")).cloned().collect();
    println!("d4_addons: tune seeds 1-8, held {:?}..{:?}, {} weak worlds", held.first(), held.last(), weak.len());

    // ---- tune shaper per learner on seeds 1-8 ------------------------------------------------
    let learners = [Kind::HebR, Kind::Warm1, Kind::Guard];
    let mut chosen: Vec<(Kind, Sp, f64)> = vec![];
    let mut tune_log = vec![];
    for &k in &learners {
        let mut best = (f64::NEG_INFINITY, Sp { center: 0.75, gain: 0.15, debias_agent: 0.0, factor: true });
        for &center in &[0.6, 0.75, 0.9] {
            for &gain in &[0.1, 0.2, 0.3] {
                for &da in &[0.0, 1.0] {
                    for &factor in &[false, true] {
                        let sp = Sp { center, gain, debias_agent: da, factor };
                        let o = obj(k, Some(sp), &tune_specs, &tune);
                        if o > best.0 {
                            best = (o, sp);
                        }
                    }
                }
            }
        }
        println!("tuned shaped({}): {:?} obj {:.4}", k.label(), best.1, best.0);
        tune_log.push(json!({"learner": k.label(), "params": format!("{:?}", best.1), "obj": best.0}));
        chosen.push((k, best.1, best.0));
    }

    // ---- main family: shaped candidates vs best raw baseline in weak worlds --------------------
    let bases: Vec<Arm> = [Kind::Keyword, Kind::Eps, Kind::Pts, Kind::HebR, Kind::Warm1, Kind::Warm10, Kind::Guard]
        .iter()
        .map(|&k| (k.label().to_string(), k, None))
        .collect();
    let cands: Vec<Arm> = chosen.iter().map(|(k, sp, _)| (format!("shaped({})", k.label()), *k, Some(*sp))).collect();
    let mut tables = vec![];
    let cells = family(&weak, &bases, &cands, &held, &mut tables);
    let main_rep = report("MAIN: shaped(learner) on per-task proxies vs best raw-feedback baseline", &cells);

    // ---- ablation: factor weights vs equal weights vs lat-only channel (rich worlds) ------------
    let rich: Vec<Spec> = weak.iter().filter(|s| s.name.ends_with("rich") || s.name.ends_with("confounded") || s.name.ends_with("useless")).cloned().collect();
    let mut abl = vec![];
    println!("\n### ablation (held-out, mean over H, shaped(guarded) tuned params): factor on/off, agent debias on/off");
    let gsp = chosen.iter().find(|c| c.0 == Kind::Guard).unwrap().1;
    for sp in &rich {
        let mut line = format!("{:<22}", sp.name);
        let mut row = vec![];
        for (lab, v) in [
            ("tuned", gsp),
            ("factor-off", Sp { factor: false, ..gsp }),
            ("factor-on", Sp { factor: true, ..gsp }),
            ("agent-debias-0", Sp { debias_agent: 0.0, ..gsp }),
            ("agent-debias-1", Sp { debias_agent: 1.0, ..gsp }),
        ] {
            let r = eval(Kind::Guard, Some(v), sp, &held);
            let m = r.iter().map(|x| x.h.iter().sum::<f64>() / 4.0).sum::<f64>() / r.len() as f64;
            line += &format!(" {lab}={m:.4}");
            row.push(json!({"variant": lab, "mean_h": m}));
        }
        let rb = eval(Kind::Guard, None, sp, &held);
        let m = rb.iter().map(|x| x.h.iter().sum::<f64>() / 4.0).sum::<f64>() / rb.len() as f64;
        line += &format!(" raw={m:.4}");
        let rk = eval(Kind::Keyword, None, sp, &held);
        let m2 = rk.iter().map(|x| x.h.iter().sum::<f64>() / 4.0).sum::<f64>() / rk.len() as f64;
        line += &format!(" keyword={m2:.4}");
        println!("{line}");
        abl.push(json!({"world": sp.name, "rows": row, "raw_guarded": m, "keyword": m2}));
    }

    // ---- abstain / confidence -------------------------------------------------------------------
    let mut ab_all = vec![];
    for bk in [Kind::Warm1, Kind::Guard] {
    println!("
### confidence / abstain-to-keyword (base = shaped({}) in weak worlds, plain outcome-feedback in normal worlds)", bk.label());
    let wsp = chosen.iter().find(|c| c.0 == bk).unwrap().1;
    let ab_specs: Vec<(Spec, Option<Sp>)> = weak
        .iter()
        .filter(|s| s.name.ends_with("rich") || s.name.ends_with("lat-only") || s.name.ends_with("useless"))
        .map(|s| (s.clone(), Some(wsp)))
        .chain(normal.iter().map(|s| (s.clone(), None)))
        .collect();
    // tune tau on seeds 1-8 (per base learner Warm1/HebR), jointly over informative worlds
    let ab_tune: Vec<&(Spec, Option<Sp>)> = ab_specs.iter().filter(|(s, _)| !s.name.ends_with("useless")).collect();
    let mut best_tau = (f64::NEG_INFINITY, 0.0);
    for &tau in &[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9] {
        let a = AbstainConfig { enabled: true, tau, ..AbstainConfig::off() };
        let mut tot = 0.0;
        for (s, sp) in &ab_tune {
            let r: Vec<AbOut> = tune.par_iter().map(|&sd| run_abstain(bk, *sp, s, sd, a, false)).collect();
            let base: Vec<AbOut> = tune.par_iter().map(|&sd| run_abstain(bk, *sp, s, sd, AbstainConfig::off(), false)).collect();
            tot += r.iter().zip(&base).map(|(x, y)| x.final_p - y.final_p).sum::<f64>() / r.len() as f64;
        }
        println!("  tune tau={tau:.2}: mean gain over informative worlds {:+.4}", tot / ab_tune.len() as f64);
        if tot > best_tau.0 {
            best_tau = (tot, tau);
        }
    }
    let tau = best_tau.1;
    println!("  chosen tau {tau:.2}");
    let acfg = AbstainConfig { enabled: true, tau, ..AbstainConfig::off() };
    let mut ab_rows = vec![];
    println!("{:<22} {:>8} {:>8} {:>8} {:>7} | risk@25 @50 @75 @100 | spread  ece(raw) ece(calib)", "world", "base", "abstain", "d(95%CI)", "rate");
    for (s, sp) in &ab_specs {
        let on: Vec<AbOut> = held.par_iter().map(|&sd| run_abstain(bk, *sp, s, sd, acfg, false)).collect();
        let on_c: Vec<AbOut> = held.par_iter().map(|&sd| run_abstain(bk, *sp, s, sd, acfg, true)).collect();
        let off: Vec<AbOut> = held.par_iter().map(|&sd| run_abstain(bk, *sp, s, sd, AbstainConfig::off(), false)).collect();
        let kw: Vec<f64> = held.par_iter().map(|&sd| run(Kind::Keyword, None, s, sd).h.iter().sum::<f64>() / 4.0).collect();
        let a: Vec<f64> = on.iter().map(|x| x.final_p).collect();
        let b: Vec<f64> = off.iter().map(|x| x.final_p).collect();
        let d = paired_diff(&a, &b);
        let pv = paired_t_pvalue(&a, &b);
        let m = |f: &dyn Fn(&AbOut) -> f64, v: &[AbOut]| v.iter().map(f).sum::<f64>() / v.len() as f64;
        let risk: Vec<f64> = (0..4).map(|i| m(&|x| x.risk[i], &on)).collect();
        println!(
            "{:<22} {:>8.4} {:>8.4} {:>+7.4}[{:+.4},{:+.4}] {:>5.2} | {:.3} {:.3} {:.3} {:.3} | {:+.3} {:.3} {:.3}  p={:.3}",
            s.name, m(&|x| x.final_p, &off), m(&|x| x.final_p, &on), d.diff.mean, d.diff.lo, d.diff.hi, m(&|x| x.rate, &on),
            risk[0], risk[1], risk[2], risk[3], m(&|x| x.spread, &on), m(&|x| x.ece, &on), m(&|x| x.ece, &on_c), pv
        );
        ab_rows.push(json!({"world": s.name, "base": m(&|x| x.final_p, &off), "abstain": m(&|x| x.final_p, &on), "diff": d.diff.mean, "ci": [d.diff.lo, d.diff.hi],
            "p": pv, "abstain_rate": m(&|x| x.rate, &on), "risk_at_coverage_25_50_75_100": risk, "spread_topHalf_minus_bottomHalf": m(&|x| x.spread, &on),
            "ece_raw": m(&|x| x.ece, &on), "ece_calibrated": m(&|x| x.ece, &on_c),
            "final_p_calibrated": m(&|x| x.final_p, &on_c), "keyword_mean_h": kw.iter().sum::<f64>() / kw.len() as f64}));
    }

    ab_all.push(json!({"base": bk.label(), "tau": tau, "rows": ab_rows}));
    }

    // ---- no-harm control: shaper under outcome feedback is passthrough (diff must be exactly 0) ----
    let mut max_abs = 0.0f64;
    for sp in &normal {
        for &k in &learners {
            let a = eval(k, None, sp, &held);
            let b = eval(k, Some(chosen.iter().find(|c| c.0 == k).unwrap().1), sp, &held); // outcome feedback bypasses observe()
            for i in 0..4 {
                for (x, y) in col(&a, i).iter().zip(col(&b, i)) {
                    max_abs = max_abs.max((x - y).abs());
                }
            }
        }
    }
    println!("\nnormal-world passthrough check (shaper configured, outcome feedback): max |diff| = {max_abs:.2e}");

    let j = json!({
        "held": format!("{}-{}", held.first().unwrap(), held.last().unwrap()),
        "tuning": tune_log, "main": main_rep, "tables": tables, "ablation": abl, "abstain": ab_all,
        "normal_passthrough_max_abs_diff": max_abs, "runtime_s": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("\nwrote {out_path} in {:.0}s", t0.elapsed().as_secs_f64());
    let _ = HashMap::<u8, u8>::new();
}
