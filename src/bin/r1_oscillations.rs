//! R1: brain-oscillation wrappers (alpha/gamma/theta-gamma/beta) around the
//! warm-started robust Hebbian router, with non-oscillating and phase-randomised
//! controls. Protocol: tune on seeds 1-8 ONLY, evaluate paired on held-out seeds
//! 31-60 against the best fair baseline per (world, window), Holm over the whole
//! candidate family. Worlds: the four sim.rs Scenarios (fixed/random/no-synergy/
//! no-traps, all with drift + a node-failure window) plus a stationary shortlab
//! style world where nothing should help (control). Windows: all, pre-drift,
//! failure window, early-post-drift, late-post-drift.
//!
//! Usage: cargo run --release --bin r1_oscillations [-- --out results/r1_oscillations.json]

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::oscillations::*;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, Window, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashMap;

const NA: usize = 6;
const COLS: [&str; 5] = ["all", "pre-drift", "failure", "early-post", "late-post"];

struct Env {
    tasks: Vec<Task>,
    probs: Vec<[f64; NA]>,
    u: Vec<f64>,
    /// 0 pre-drift, 1 failure, 2 early-post, 3 late-post
    win: Vec<u8>,
    kmap: HashMap<String, usize>,
}

fn sim_env(seed: u64, sc: Scenario) -> Env {
    let w = World::new(seed, WorldConfig { scenario: sc, ..WorldConfig::default() });
    let mut e = Env { tasks: vec![], probs: vec![], u: vec![], win: vec![], kmap: w.keyword_map_for() };
    for t in 0..w.len() {
        let st = w.task(t);
        e.tasks.push(st.task.clone());
        let mut p = [0.0; NA];
        for (a, pa) in p.iter_mut().enumerate() {
            *pa = w.success_prob(t, &[a]);
        }
        e.probs.push(p);
        e.u.push(st.u);
        e.win.push(if w.in_failure(t) {
            1
        } else {
            match w.window(t) {
                Window::PreDrift => 0,
                Window::EarlyPost => 2,
                Window::LatePost => 3,
            }
        });
    }
    e
}

/// Stationary shortlab-style world: 8 topics, 6 agents, 2 trap topics, no drift,
/// no failures. Mechanisms for non-stationarity should NOT help here.
fn static_env(seed: u64) -> Env {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(11));
    let (tn, an, n) = (8usize, NA, 2000usize);
    let mut p = vec![[0.0; NA]; tn];
    let mut best = vec![0usize; tn];
    let mut second = vec![0usize; tn];
    for t in 0..tn {
        best[t] = rng.below(an);
        loop {
            second[t] = rng.below(an);
            if second[t] != best[t] {
                break;
            }
        }
        for a in 0..an {
            p[t][a] = if a == best[t] {
                0.85
            } else if a == second[t] {
                0.5
            } else {
                0.15 + 0.25 * rng.f64()
            };
        }
    }
    let mut order: Vec<usize> = (0..tn).collect();
    for i in (1..tn).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let trap: Vec<bool> = (0..tn).map(|t| order.iter().take(2).any(|&x| x == t)).collect();
    let mut kmap = HashMap::new();
    for t in 0..tn {
        for j in 0..3 {
            let a = if trap[t] {
                loop {
                    let a = rng.below(an);
                    if a != best[t] && a != second[t] {
                        break a;
                    }
                }
            } else if rng.f64() < 0.10 {
                rng.below(an)
            } else {
                best[t]
            };
            kmap.insert(format!("t{t}k{j}"), a);
        }
    }
    let mut e = Env { tasks: vec![], probs: vec![], u: vec![], win: vec![], kmap };
    for i in 0..n {
        let t = rng.below(tn);
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(0.25) {
            let o = (t + 1 + rng.below(tn - 1)) % tn;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        e.tasks.push(Task { id: i as u64, task_type: None, features: f });
        e.probs.push(p[t]);
        e.u.push(rng.f64());
        e.win.push(0);
    }
    e
}

#[derive(Clone, Copy, Debug)]
enum EvK {
    Set,
    Mean,
    Global,
    FailSet,
    FailMean,
}
fn ev<R>(e: EvK) -> Evidence<R> {
    match e {
        EvK::Set => Evidence::FeatureSet,
        EvK::Mean => Evidence::MeanFeature,
        EvK::Global => Evidence::Global,
        EvK::FailSet => Evidence::SetFailures,
        EvK::FailMean => Evidence::FeatureFailures,
    }
}

fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

#[derive(Clone, Debug)]
enum Spec {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    HebR,
    WarmHebR(f64, EvK),
    Osc { k: f64, evk: EvK, cfg: OscConfig, eta_mult: f64 },
}

fn build_osc(k: f64, evk: EvK, cfg: &OscConfig, eta_mult: f64, env: &Env, seed: u64) -> OscRouter {
    let mut hc = heb_robust_cfg();
    hc.eta *= eta_mult;
    let w = Warm::new(HebbianRouter::new(NA, hc), env.kmap.clone(), NA, k).with_evidence(ev(evk));
    OscRouter::new(w, cfg.clone(), seed.wrapping_mul(13).wrapping_add(7))
}

fn build(spec: &Spec, env: &Env, seed: u64) -> Box<dyn Router> {
    match spec {
        Spec::Keyword => Box::new(KeywordRouter::new(env.kmap.clone(), NA)),
        Spec::Eps(g, e) => Box::new(EpsGreedyRouter::new(NA, *g, *e)),
        Spec::Pts(g) => Box::new(PooledThompsonRouter::new(NA, *g)),
        Spec::HebR => Box::new(HebbianRouter::new(NA, heb_robust_cfg())),
        Spec::WarmHebR(k, evk) => {
            Box::new(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), env.kmap.clone(), NA, *k).with_evidence(ev(*evk)))
        }
        Spec::Osc { k, evk, cfg, eta_mult } => Box::new(build_osc(*k, *evk, cfg, *eta_mult, env, seed)),
    }
}

/// [all, pre, failure, early-post, late-post] mean expected success (NaN if window empty).
fn drive(r: &mut dyn Router, env: &Env, seed: u64) -> [f64; 5] {
    let cands: Vec<usize> = (0..NA).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let (mut s, mut c) = ([0.0; 5], [0usize; 5]);
    for (i, task) in env.tasks.iter().enumerate() {
        let a = r.select(task, &cands, &mut rng);
        let p = env.probs[i][a];
        r.update(task, a, if env.u[i] < p { 1.0 } else { 0.0 });
        s[0] += p;
        c[0] += 1;
        let w = 1 + env.win[i] as usize;
        s[w] += p;
        c[w] += 1;
    }
    let mut out = [f64::NAN; 5];
    for j in 0..5 {
        if c[j] > 0 {
            out[j] = s[j] / c[j] as f64;
        }
    }
    out
}

/// results[seed_idx] for one spec on one world's per-seed envs.
fn eval(spec: &Spec, envs: &[Env], seeds: &[u64]) -> Vec<[f64; 5]> {
    envs.par_iter().zip(seeds.par_iter()).map(|(e, &s)| drive(build(spec, e, s).as_mut(), e, s)).collect()
}

struct Bank {
    names: Vec<&'static str>,
    tune: Vec<Vec<Env>>,
    held: Vec<Vec<Env>>,
    tune_seeds: Vec<u64>,
    held_seeds: Vec<u64>,
}

fn make_envs(wi: usize, seeds: &[u64]) -> Vec<Env> {
    seeds
        .par_iter()
        .map(|&s| match wi {
            0 => sim_env(s, Scenario::Fixed),
            1 => sim_env(s, Scenario::Random),
            2 => sim_env(s, Scenario::NoSynergy),
            3 => sim_env(s, Scenario::NoTraps),
            _ => static_env(s),
        })
        .collect()
}

/// Tuning objective: mean over worlds of mean-over-seeds of the "all" column.
fn obj(spec: &Spec, bank: &Bank) -> f64 {
    let mut t = 0.0;
    for w in 0..bank.names.len() {
        let r = eval(spec, &bank.tune[w], &bank.tune_seeds);
        t += r.iter().map(|x| x[0]).sum::<f64>() / r.len() as f64;
    }
    t / bank.names.len() as f64
}

fn best_of(cands: Vec<Spec>, bank: &Bank, label: &str) -> (Spec, f64) {
    let n = cands.len();
    let mut best: Option<(f64, Spec)> = None;
    for k in cands {
        let o = obj(&k, bank);
        if best.as_ref().is_none_or(|b| o > b.0) {
            best = Some((o, k));
        }
    }
    let (o, k) = best.unwrap();
    println!("[tune 1-8] {label}: {n} configs, best {o:.4}");
    (k, o)
}

fn col(r: &[[f64; 5]], j: usize) -> Vec<f64> {
    r.iter().map(|x| x[j]).collect()
}

fn cfg_json(c: &OscConfig) -> serde_json::Value {
    serde_json::to_value(c).unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/r1_oscillations.json".into());
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let held_seeds: Vec<u64> = (31..=60).collect();
    let names = vec!["fixed", "random", "no-synergy", "no-traps", "static(control)"];
    let bank = Bank {
        tune: (0..5).map(|w| make_envs(w, &tune_seeds)).collect(),
        held: (0..5).map(|w| make_envs(w, &held_seeds)).collect(),
        names,
        tune_seeds,
        held_seeds,
    };
    println!("R1 oscillations: tune seeds 1-8, held-out 31-60, worlds {:?}", bank.names);

    // ---------- fair baselines (tuned on 1-8) ----------
    let mut c = vec![];
    for &g in &[0.95, 0.99, 1.0] {
        for &e in &[0.02, 0.05, 0.1] {
            c.push(Spec::Eps(g, e));
        }
    }
    let (eps_spec, _) = best_of(c, &bank, "eps-greedy");
    let (pts_spec, _) = best_of([0.9, 0.95, 0.99, 1.0].iter().map(|&g| Spec::Pts(g)).collect(), &bank, "pooled-ts");
    let evs = [EvK::Set, EvK::Mean, EvK::Global, EvK::FailSet, EvK::FailMean];
    let mut c = vec![];
    for &k in &[3.0, 10.0, 30.0, 100.0, 300.0] {
        for &e in &evs {
            c.push(Spec::WarmHebR(k, e));
        }
    }
    let (warm_spec, warm_obj) = best_of(c, &bank, "warm(hebbian-robust)");
    let Spec::WarmHebR(wk, wev) = warm_spec.clone() else { unreachable!() };
    println!("   -> warm(hebbian-robust) k={wk} evidence={wev:?} obj={warm_obj:.4}");

    let osc = |cfg: OscConfig| Spec::Osc { k: wk, evk: wev, cfg, eta_mult: 1.0 };
    let mut tuned: Vec<(String, f64)> = vec![]; // label -> tuning objective
    let mut tuned_cfg: HashMap<String, OscConfig> = HashMap::new();
    let mut pick = |label: &str, cands: Vec<OscConfig>, tuned: &mut Vec<(String, f64)>| -> OscConfig {
        let (s, o) = best_of(cands.into_iter().map(&osc).collect(), &bank, label);
        let Spec::Osc { cfg, .. } = s else { unreachable!() };
        tuned.push((label.to_string(), o));
        tuned_cfg.insert(label.to_string(), cfg.clone());
        cfg
    };

    // ---------- (a) alpha ----------
    let alpha_grid = |mode: AlphaMode| -> Vec<OscConfig> {
        let mut v = vec![];
        for &p in &[25usize, 50, 100, 200, 400] {
            for &f in &[0.5, 0.8] {
                for &(e, t) in &[(0.1, 0.0), (0.3, 0.0), (0.0, 0.02), (0.0, 0.08)] {
                    v.push(OscConfig {
                        alpha: Some(AlphaCfg { period: p, inhibit_frac: f, eps_open: e, temp_open: t, mode }),
                        ..OscConfig::default()
                    });
                }
            }
        }
        v
    };
    let a_sq = pick("alpha-square", alpha_grid(AlphaMode::Square), &mut tuned);
    let a_const = pick("alpha-const(best constant)", alpha_grid(AlphaMode::Const), &mut tuned);
    let a_rand = pick("alpha-random(best phase-randomised)", alpha_grid(AlphaMode::Random), &mut tuned);
    let a_mm = {
        let mut a = a_sq.alpha.clone().unwrap();
        a.mode = AlphaMode::Const;
        OscConfig { alpha: Some(a), ..OscConfig::default() }
    };
    let a_rr = {
        let mut a = a_sq.alpha.clone().unwrap();
        a.mode = AlphaMode::Random;
        OscConfig { alpha: Some(a), ..OscConfig::default() }
    };

    // ---------- (b) gamma ----------
    let mut gate_grid = |timing: Timing| {
        let mut v = vec![];
        for &thr in &[0.15, 0.25, 0.35] {
            for &ea in &[0.1, 0.3] {
                for &bp in &[50usize, 200] {
                    v.push(OscConfig {
                        gate: Some(GateCfg { ema_alpha: ea, thr, burst_period: bp, burst_len: 5, timing }),
                        ..OscConfig::default()
                    });
                }
            }
        }
        v
    };
    let g_gate = pick("gamma-gate", gate_grid(Timing::Periodic), &mut tuned);
    let g_gate_r = {
        let mut g = g_gate.gate.clone().unwrap();
        g.timing = Timing::Random;
        OscConfig { gate: Some(g), ..OscConfig::default() }
    };
    let batch_grid = |timing: Timing| -> Vec<OscConfig> {
        [2usize, 5, 10, 25, 50]
            .iter()
            .map(|&e| OscConfig { batch: Some(BatchCfg { every: e, timing }), ..OscConfig::default() })
            .collect()
    };
    let g_batch = pick("gamma-batch", batch_grid(Timing::Periodic), &mut tuned);
    let g_batch_r = {
        let mut b = g_batch.batch.clone().unwrap();
        b.timing = Timing::Random;
        OscConfig { batch: Some(b), ..OscConfig::default() }
    };

    // ---------- (c) theta-gamma replay ----------
    let mut theta_cfgs: Vec<(usize, OscConfig, OscConfig, f64)> = vec![]; // k, replay, random-timing control, eta-matched mult
    for &k in &[2usize, 4, 8] {
        let mut v = vec![];
        for &p in &[10usize, 25, 50, 100] {
            for &pk in &[ReplayPick::Surprise, ReplayPick::Recent] {
                v.push(OscConfig {
                    theta: Some(ThetaCfg { period: p, k, pick: pk, timing: Timing::Periodic }),
                    ..OscConfig::default()
                });
            }
        }
        let c = pick(&format!("theta-replay k={k}"), v, &mut tuned);
        let mut th = c.theta.clone().unwrap();
        let mult = 1.0 + k as f64 / th.period as f64;
        th.timing = Timing::Random;
        theta_cfgs.push((k, c, OscConfig { theta: Some(th), ..OscConfig::default() }, mult));
    }

    // ---------- (d) beta ----------
    let mut v = vec![];
    for &sp in &[0.5, 0.8, 0.95] {
        for &thr in &[0.5, 0.7] {
            for &ea in &[0.1, 0.3] {
                v.push(OscConfig {
                    beta: Some(BetaCfg { stick_p: sp, ema_alpha: ea, burst_thr: thr, release: Release::Surprise }),
                    ..OscConfig::default()
                });
            }
        }
    }
    let b_cfg = pick("beta-status-quo", v, &mut tuned);
    let b_never = {
        let mut b = b_cfg.beta.clone().unwrap();
        b.release = Release::Never;
        OscConfig { beta: Some(b), ..OscConfig::default() }
    };
    // matched random-release rate measured on the tuning runs
    let rate = {
        let mut tot = 0.0;
        let mut n = 0.0;
        for w in 0..bank.names.len() {
            for (e, &s) in bank.tune[w].iter().zip(&bank.tune_seeds) {
                let mut r = build_osc(wk, wev, &b_cfg, 1.0, e, s);
                drive(&mut r, e, s);
                tot += r.burst_rate();
                n += 1.0;
            }
        }
        tot / n
    };
    let b_rand = {
        let mut b = b_cfg.beta.clone().unwrap();
        b.release = Release::Random(rate);
        OscConfig { beta: Some(b), ..OscConfig::default() }
    };
    println!("   beta tuned burst rate on tuning seeds: {rate:.4}");

    // ---------- combined stack (only if >=2 singles show tuning-seed promise) ----------
    let promising: Vec<(String, f64)> = tuned
        .iter()
        .filter(|(l, o)| !l.contains("const") && !l.contains("random") && *o > warm_obj + 0.001)
        .cloned()
        .collect();
    println!("   singles with tuning gain > +0.001 over warm: {:?}", promising);
    let mut combined: Option<OscConfig> = None;
    if promising.len() >= 2 {
        let mut c = OscConfig::default();
        for (l, _) in &promising {
            let t = &tuned_cfg[l];
            if l.starts_with("alpha") {
                c.alpha = t.alpha.clone();
            }
            if l.starts_with("gamma-gate") {
                c.gate = t.gate.clone();
            }
            if l.starts_with("gamma-batch") {
                c.batch = t.batch.clone();
            }
            if l.starts_with("theta") && c.theta.is_none() {
                c.theta = t.theta.clone();
            }
            if l.starts_with("beta") {
                c.beta = t.beta.clone();
            }
        }
        combined = Some(c);
    }

    // ---------- held-out evaluation ----------
    struct Row {
        label: String,
        spec: Spec,
        kind: &'static str, // baseline | candidate | control
        control_of: Option<String>,
    }
    let mut rows: Vec<Row> = vec![];
    let mut add = |label: &str, spec: Spec, kind: &'static str, control_of: Option<&str>| {
        rows.push(Row { label: label.into(), spec, kind, control_of: control_of.map(|s| s.to_string()) });
    };
    add("keyword", Spec::Keyword, "baseline", None);
    add("eps-greedy(tuned)", eps_spec.clone(), "baseline", None);
    add("pooled-ts(tuned)", pts_spec.clone(), "baseline", None);
    add("hebbian-robust", Spec::HebR, "baseline", None);
    add("warm(hebbian-robust)", warm_spec.clone(), "baseline", None);
    let eta_ctl = |cfg: &OscConfig, m: f64| Spec::Osc { k: wk, evk: wev, cfg: cfg.clone(), eta_mult: m };
    add("A alpha square", osc(a_sq.clone()), "candidate", None);
    add("A ctl const (mean-matched)", osc(a_mm.clone()), "control", Some("A alpha square"));
    add("A ctl phase-random (same duty)", osc(a_rr.clone()), "control", Some("A alpha square"));
    add("A ctl best-tuned const", osc(a_const.clone()), "control", Some("A alpha square"));
    add("A ctl best-tuned phase-random", osc(a_rand.clone()), "control", Some("A alpha square"));
    add("B gamma gate", osc(g_gate.clone()), "candidate", None);
    add("B ctl gate random-bursts", osc(g_gate_r.clone()), "control", Some("B gamma gate"));
    add("B gamma batch", osc(g_batch.clone()), "candidate", None);
    add("B ctl batch random-times", osc(g_batch_r.clone()), "control", Some("B gamma batch"));
    for (k, c, cr, m) in &theta_cfgs {
        let l = format!("C theta replay k={k}");
        add(&l, osc(c.clone()), "candidate", None);
        add(&format!("C ctl replay random-times k={k}"), osc(cr.clone()), "control", Some(&l));
        add(&format!("C ctl eta-matched no-replay k={k}"), eta_ctl(&OscConfig::default(), *m), "control", Some(&l));
    }
    add("D beta status-quo", osc(b_cfg.clone()), "candidate", None);
    add("D ctl never-release", osc(b_never.clone()), "control", Some("D beta status-quo"));
    add("D ctl random-release (matched rate)", osc(b_rand.clone()), "control", Some("D beta status-quo"));
    if let Some(c) = &combined {
        add("E combined stack", osc(c.clone()), "candidate", None);
    }

    let nw = bank.names.len();
    // res[row][world] = Vec<[f64;5]> over held-out seeds
    let res: Vec<Vec<Vec<[f64; 5]>>> =
        rows.iter().map(|r| (0..nw).map(|w| eval(&r.spec, &bank.held[w], &bank.held_seeds)).collect()).collect();
    let base_idx: Vec<usize> = rows.iter().enumerate().filter(|(_, r)| r.kind == "baseline").map(|(i, _)| i).collect();
    let warm_i = rows.iter().position(|r| r.label == "warm(hebbian-robust)").unwrap();
    // best fair baseline per (world, col) by held-out mean (conservative)
    let mut best_b = vec![[0usize; 5]; nw];
    for w in 0..nw {
        for j in 0..5 {
            let mut bi = base_idx[0];
            let mut bm = f64::NEG_INFINITY;
            for &i in &base_idx {
                let m = summarize(&col(&res[i][w], j)).mean;
                if m.is_finite() && m > bm {
                    bm = m;
                    bi = i;
                }
            }
            best_b[w][j] = bi;
        }
    }

    // table: overall mean per router per world
    println!("\n=== held-out seeds 31-60: mean expected success, window=all ===");
    print!("{:<40}", "router");
    for n in &bank.names {
        print!("{:>16}", n);
    }
    println!();
    for (i, r) in rows.iter().enumerate() {
        print!("{:<40}", r.label);
        for w in 0..nw {
            let s = summarize(&col(&res[i][w], 0));
            print!("{:>16}", format!("{:.4}+-{:.4}", s.mean, s.half_width));
        }
        println!();
    }

    // Holm over the candidate family: candidates x worlds x valid windows, vs best fair baseline
    let mut cells: Vec<(usize, usize, usize)> = vec![]; // (row, world, col)
    for (i, r) in rows.iter().enumerate() {
        if r.kind != "candidate" {
            continue;
        }
        for w in 0..nw {
            for j in 0..5 {
                if col(&res[i][w], j).iter().all(|x| x.is_finite()) {
                    cells.push((i, w, j));
                }
            }
        }
    }
    let pv: Vec<f64> = cells
        .iter()
        .map(|&(i, w, j)| paired_t_pvalue(&col(&res[i][w], j), &col(&res[best_b[w][j]][w], j)))
        .collect();
    let adj = holm_adjust(&pv);
    println!("\n=== candidates vs BEST FAIR BASELINE (per world/window), diff [95% CI], Holm p over {} cells ===", cells.len());
    let mut jtests = vec![];
    let mut wins: HashMap<String, Vec<String>> = HashMap::new();
    for (ci, &(i, w, j)) in cells.iter().enumerate() {
        let b = best_b[w][j];
        let x = col(&res[i][w], j);
        let d = paired_diff(&x, &col(&res[b][w], j));
        let dw = paired_diff(&x, &col(&res[warm_i][w], j));
        let win = adj[ci] < 0.05 && d.diff.mean > 0.0;
        let lose = adj[ci] < 0.05 && d.diff.mean < 0.0;
        if win {
            wins.entry(rows[i].label.clone()).or_default().push(format!("{}/{}", bank.names[w], COLS[j]));
        }
        if j == 0 || win || lose {
            println!(
                "{:<26} {:<16} {:<11} vs {:<22} {:+.4} [{:+.4},{:+.4}] holm p={:.4} {}",
                rows[i].label, bank.names[w], COLS[j], rows[b].label, d.diff.mean, d.diff.lo, d.diff.hi, adj[ci],
                if win { "WIN" } else if lose { "LOSE" } else { "" }
            );
        }
        jtests.push(json!({
            "candidate": rows[i].label, "world": bank.names[w], "window": COLS[j],
            "mean": summarize(&x).mean, "best_baseline": rows[b].label,
            "vs_best_baseline": {"diff": d.diff, "wins": d.wins, "losses": d.losses, "holm_p": adj[ci], "win": win, "lose": lose},
            "vs_warm_hebr": {"diff": dw.diff, "wins": dw.wins, "losses": dw.losses},
        }));
    }
    // acceptance: win in >1 world
    let mut accepted = vec![];
    for (l, ws) in &wins {
        let mut worlds: Vec<&str> = ws.iter().map(|s| s.split('/').next().unwrap()).collect();
        worlds.sort();
        worlds.dedup();
        println!("WIN cells for {l}: {:?}", ws);
        if worlds.len() > 1 {
            accepted.push(l.clone());
        }
    }
    println!("accepted by bar (Holm win vs best fair baseline in >1 world): {:?}", accepted);

    // controls: candidate minus its controls (paired), window all + failure + early-post
    println!("\n=== oscillation vs its controls (candidate - control), diff [95% CI]; unadjusted ===");
    let mut jctl = vec![];
    for (ci, r) in rows.iter().enumerate() {
        let Some(parent) = &r.control_of else { continue };
        let pi = rows.iter().position(|x| &x.label == parent).unwrap();
        println!("-- {} (control of {})", r.label, parent);
        for w in 0..nw {
            let mut line = format!("   {:<16}", bank.names[w]);
            for j in [0usize, 2, 3] {
                let (a, b) = (col(&res[pi][w], j), col(&res[ci][w], j));
                if a.iter().any(|x| !x.is_finite()) {
                    continue;
                }
                let d = paired_diff(&a, &b);
                line.push_str(&format!(" {}:{:+.4}[{:+.4},{:+.4}]", COLS[j], d.diff.mean, d.diff.lo, d.diff.hi));
                jctl.push(json!({"candidate": parent, "control": r.label, "world": bank.names[w], "window": COLS[j], "cand_minus_control": d.diff}));
            }
            println!("{line}");
        }
    }

    // summary table json
    let table: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            json!({"router": r.label, "kind": r.kind, "worlds": (0..nw).map(|w| json!({
                "world": bank.names[w],
                "by_window": (0..5).map(|j| json!({"window": COLS[j], "summary": summarize(&col(&res[i][w], j))})).collect::<Vec<_>>()
            })).collect::<Vec<_>>()})
        })
        .collect();
    let j = json!({
        "protocol": "tune seeds 1-8 (objective: mean over 5 worlds of window=all expected success); held-out seeds 31-60; paired 95% t-CI; Holm over candidates x worlds x windows vs best fair baseline per cell",
        "worlds": bank.names,
        "metric": "mean expected success prob of chosen agent (sim success_prob), per window",
        "baseline_tuned": {"eps": format!("{:?}", eps_spec), "pooled_ts": format!("{:?}", pts_spec), "warm_hebr": format!("{:?}", warm_spec)},
        "tuned_objectives_seeds1_8": tuned.iter().map(|(l, o)| json!({"label": l, "obj": o, "delta_vs_warm": o - warm_obj})).collect::<Vec<_>>(),
        "warm_tuning_objective": warm_obj,
        "tuned_configs": tuned_cfg.iter().map(|(k, v)| (k.clone(), cfg_json(v))).collect::<HashMap<_, _>>(),
        "beta_burst_rate": rate,
        "combined_config": combined.as_ref().map(cfg_json),
        "table": table,
        "tests_vs_best_baseline": jtests,
        "controls": jctl,
        "wins": wins,
        "accepted": accepted,
    });
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("\nwrote {out}");
}
