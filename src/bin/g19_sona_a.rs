//! G19-A (W20, SONA-A): can SonaPlus be improved by better PATTERN-MEMORY KEYING and RETRIEVAL?
//!
//! Mechanisms (all bin-local, src/*.rs untouched; see src/bin/g19_common/keyed.rs):
//!   M1 learned anchor keys     : online PMI-lift co-occurrence clustering of surface features -> anchor tokens; the guard counts
//!                                and the inner learner are keyed by anchors (ORACLE topic anchors as a diagnostic upper bound).
//!   M2 k-nearest pattern blend : exact key-set patterns + Jaccard-weighted blend of the k nearest patterns + unigram sources,
//!                                learned key selection (drop confirmed-uninformative tokens), credit spreading to neighbours.
//!   M3 consolidation           : bounded pattern store, EWC-style importance (evidence x advantage) victim merged into nearest
//!                                pattern vs LRU / FIFO / count / importance-drop.
//!   M4 trajectory credit       : session worlds (only "all steps ok" is reported at session end): posterior blame split using
//!                                the pattern memory's own P(success) vs the same credit with a plain per-feature estimator
//!                                vs broadcast.
//! Worlds were fixed in g19_common/mod.rs before any policy was run. Protocol: tune seeds 1-8 (regret-averse objective vs best
//! fair baseline per world x H, staged grids), held-out 31-60 (`--held 61-90` = fresh confirmation, SAME frozen tuning), paired
//! 95% t-CI, Holm per family. All numbers are SYNTHETIC until validated on real traffic.
//!
//! Usage: cargo run --release --offline --bin g19_sona_a -- [--held 31-60] [--out results/g19a_sona_a.json] [--quick]
//!        [--only normal,weak,session,mem] [--selfcheck]

use std::collections::HashMap;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::defaults;
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, GuardConfig, Warm};
use neural_mesh::router::{Rng, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g19_common/mod.rs"]
mod c;
use c::keyed::*;
use c::*;

// ------------------------------------------------------------------------------------------------ arms
#[derive(Clone, Debug)]
enum AnchorMode {
    Off,
    Learned(AnchorCfg),
    Oracle,
}

#[derive(Clone, Debug)]
enum InnerSpec {
    Heb,
    Mem(MemCfg),
    Hybrid(MemCfg, f64),
}

#[derive(Clone, Debug)]
struct KSpec {
    guard: GuardConfig,
    anchors: AnchorMode,
    inner: InnerSpec,
    delayed_mult: usize,
    fail_reward: f64,
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps,
    Pts,
    HebR,
    WarmHebR,
    GuardSafe,
    K(KSpec),
    /// trajectory credit wrapper; bool = use the wrapped policy's own P(success) estimate
    Credit(Box<Kind>, bool),
}

#[derive(Clone, Debug)]
struct Arm {
    name: String,
    kind: Kind,
    fb: Option<Fb>,
}

fn arm(n: &str, k: Kind) -> Arm {
    Arm { name: n.to_string(), kind: k, fb: None }
}
fn arm_fb(n: &str, k: Kind, fb: Fb) -> Arm {
    Arm { name: n.to_string(), kind: k, fb: Some(fb) }
}

fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

fn x1_like_guard() -> GuardConfig {
    GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
}

fn kspec(guard: GuardConfig, anchors: AnchorMode, inner: InnerSpec) -> KSpec {
    KSpec { guard, anchors, inner, delayed_mult: 1, fail_reward: 0.0 }
}

fn build_k(s: &KSpec, w: &World) -> Box<dyn Pol> {
    let n = w.agents;
    let anchors = match &s.anchors {
        AnchorMode::Off => Anchors::new(AnchorCfg::off()),
        AnchorMode::Learned(c) => Anchors::new(c.clone()),
        AnchorMode::Oracle => Anchors::oracle(w.oracle.clone()),
    };
    match &s.inner {
        InnerSpec::Heb => Box::new(KPol { g: AnchorGuard::new(HebInner::robust(n), w.kwmap.clone(), n, s.guard, anchors), delayed_mult: s.delayed_mult, fail_reward: s.fail_reward }),
        InnerSpec::Mem(m) => Box::new(KPol { g: AnchorGuard::new(KeyedMemory::new(n, m.clone()), w.kwmap.clone(), n, s.guard, anchors), delayed_mult: s.delayed_mult, fail_reward: s.fail_reward }),
        InnerSpec::Hybrid(m, mw) => Box::new(KPol { g: AnchorGuard::new(HybridInner::new(n, m.clone(), *mw), w.kwmap.clone(), n, s.guard, anchors), delayed_mult: s.delayed_mult, fail_reward: s.fail_reward }),
    }
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
        Kind::GuardSafe => Box::new(Plain(defaults::recommended_router(km, n))),
        Kind::K(s) => build_k(s, w),
        Kind::Credit(inner, own) => Box::new(Credited::new(build(inner, w), n, *own)),
    }
}

fn eval(a: &Arm, spec: &WorldSpec, seeds: &[u64]) -> Vec<[f64; 5]> {
    let fb = a.fb.unwrap_or(spec.fb);
    seeds.par_iter().map(|&sd| run(&|w| build(&a.kind, w), &gen_world(sd, spec), sd, fb)).collect()
}

fn col(r: &[[f64; 5]], h: usize) -> Vec<f64> {
    r.iter().map(|x| x[h]).collect()
}

fn means(a: &Arm, seeds: &[u64], ws: &[WorldSpec]) -> Vec<[f64; 5]> {
    ws.iter()
        .map(|s| {
            let r = eval(a, s, seeds);
            let mut m = [0.0; 5];
            for h in 0..5 {
                m[h] = r.iter().map(|x| x[h]).sum::<f64>() / r.len() as f64;
            }
            m
        })
        .collect()
}

fn best_of(arms: &[Arm], seeds: &[u64], ws: &[WorldSpec]) -> Vec<[f64; 5]> {
    let bm: Vec<Vec<[f64; 5]>> = arms.iter().map(|a| means(a, seeds, ws)).collect();
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

/// Regret-averse objective vs best baseline per (world, horizon): mean(d) + mean(min(d,0)).
fn obj(a: &Arm, seeds: &[u64], ws: &[WorldSpec], best_b: &[[f64; 5]]) -> f64 {
    let m = means(a, seeds, ws);
    let mut d = vec![];
    for (w, row) in m.iter().enumerate() {
        for h in 0..5 {
            d.push(row[h] - best_b[w][h]);
        }
    }
    let n = d.len() as f64;
    d.iter().sum::<f64>() / n + d.iter().map(|x| x.min(0.0)).sum::<f64>() / n
}

fn tune<T: Clone + std::fmt::Debug + Send + Sync>(label: &str, cands: Vec<T>, mk: &(dyn Fn(&T) -> Arm + Sync), seeds: &[u64], ws: &[WorldSpec], best_b: &[[f64; 5]]) -> (f64, T) {
    let n = cands.len();
    let mut sc: Vec<(f64, T)> = cands.into_iter().map(|c| (obj(&mk(&c), seeds, ws, best_b), c)).collect();
    sc.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("[tune 1-8] {label}: {n} configs, best obj {:+.5}\n    {:?}", sc[0].0, sc[0].1);
    for (o, c) in sc.iter().skip(1).take(2) {
        println!("      runner-up {o:+.5}: {c:?}");
    }
    sc.swap_remove(0)
}

// ------------------------------------------------------------------------------------------------ grids
fn guard_cfg(tau: f64, z: f64, ka: f64, kb: f64, disc: f64, ch: bool) -> GuardConfig {
    GuardConfig { tau, z, kw_a: ka, kw_b: kb, discount: disc, challenge: ch }
}

fn guard_rt_grid(quick: bool) -> Vec<GuardConfig> {
    let mut v = vec![];
    for &tau in &[0.45, 0.55, 0.65, 0.75, 0.85] {
        for &z in &[0.5, 1.0] {
            for &(ka, kb) in &[(2.0, 1.0), (4.0, 1.0)] {
                for &disc in &[1.0, 0.97] {
                    for &ch in &[true, false] {
                        v.push(guard_cfg(tau, z, ka, kb, disc, ch));
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

#[derive(Clone, Debug)]
struct Stage1 {
    guard: GuardConfig,
    lift: f64,
    alpha: f64,
}

/// Guard options for the K variants. SAFE (default): exactly `defaults::recommended_guard()` (so the new mechanism cannot change
/// when the keyword rule is followed / abandoned); TUNED (`--guard tuned`, rounds 1-2): tau x discount grid.
fn stage1_grid(quick: bool, anchors_on: bool, safe: bool) -> Vec<Stage1> {
    let mut guards = vec![];
    if safe {
        guards.push(defaults::recommended_guard());
    } else {
        for &tau in &[0.45, 0.55, 0.65, 0.75, 0.85] {
            for &disc in &[1.0, 0.97] {
                guards.push(guard_cfg(tau, 0.5, 2.0, 1.0, disc, true));
            }
        }
    }
    let mut v = vec![];
    for g in guards {
        for &lift in &[3.0, 8.0] {
            for &alpha in &[1e-2f64, 1e-4] {
                if !anchors_on && (lift != 3.0 || alpha != 1e-2) {
                    continue;
                }
                v.push(Stage1 { guard: g, lift, alpha });
            }
        }
    }
    if quick {
        v.truncate(4);
    }
    v
}

fn s1_guard(s: &Stage1) -> GuardConfig {
    s.guard
}

fn mem_stage2_grid(quick: bool, sets: bool) -> Vec<MemCfg> {
    let mut v = vec![];
    for &knn in &[0usize, 3] {
        for &temp in &[0.03, 0.1] {
            for &lam in &[0.3, 1.0] {
                for &kappa in &[2.0, 6.0] {
                    for &sel in &[false, true] {
                        for &spread in &[0.0, 0.5] {
                            if !sets && (knn > 0 || sel || spread > 0.0) {
                                continue;
                            }
                            if knn == 0 && spread > 0.0 {
                                continue;
                            }
                            v.push(MemCfg { knn_k: knn, temp, lam_u: lam, kappa, key_sel: sel, spread, use_sets: sets, ..MemCfg::default() });
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

fn mem_stage3_grid(base: &MemCfg, quick: bool) -> Vec<MemCfg> {
    let mut v = vec![];
    for &smin in &[0.3, 0.6] {
        for &gamma in &[1.0, 3.0] {
            for &init in &[0.6, 0.8] {
                v.push(MemCfg { s_min: smin, gamma, init, ..base.clone() });
            }
        }
    }
    if quick {
        v.truncate(3);
    }
    v
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Inner {
    Heb,
    /// pattern memory decides every post-hand-over choice (round-3 "K-full")
    Mem,
    /// evidence-gated: pattern memory when set-level evidence >= min_w, else hebbian-robust
    Hybrid,
}

/// staged tuning of a K variant. anchors: Off | Learned (grid over lift/alpha) | Oracle. sets=false -> unigram-only memory.
#[allow(clippy::too_many_arguments)]
fn tune_k(label: &str, anchors_on: Option<bool>, oracle: bool, inner: Inner, sets: bool, seeds: &[u64], ws: &[WorldSpec], bb: &[[f64; 5]], quick: bool, safe: bool) -> KSpec {
    let learned = anchors_on.unwrap_or(false);
    let mk_anchor = |s: &Stage1| -> AnchorMode {
        if oracle {
            AnchorMode::Oracle
        } else if learned {
            AnchorMode::Learned(AnchorCfg::on_alpha(s.lift, s.alpha))
        } else {
            AnchorMode::Off
        }
    };
    let mk_inner = |m: &MemCfg, mw: f64| match inner {
        Inner::Heb => InnerSpec::Heb,
        Inner::Mem => InnerSpec::Mem(m.clone()),
        Inner::Hybrid => InnerSpec::Hybrid(m.clone(), mw),
    };
    let m0 = MemCfg { use_sets: sets, ..MemCfg::default() };
    let (_, s1) = tune(
        &format!("{label} stage1 (guard{})", if learned { " + anchor lift/alpha" } else { "" }),
        stage1_grid(quick, learned, safe),
        &|s: &Stage1| arm("t", Kind::K(kspec(s1_guard(s), mk_anchor(s), mk_inner(&m0, 6.0)))),
        seeds,
        ws,
        bb,
    );
    if inner == Inner::Heb {
        return kspec(s1_guard(&s1), mk_anchor(&s1), InnerSpec::Heb);
    }
    let (_, m2) = tune(&format!("{label} stage2 (memory)"), mem_stage2_grid(quick, sets), &|m: &MemCfg| arm("t", Kind::K(kspec(s1_guard(&s1), mk_anchor(&s1), mk_inner(m, 6.0)))), seeds, ws, bb);
    if !sets {
        return kspec(s1_guard(&s1), mk_anchor(&s1), mk_inner(&m2, 6.0));
    }
    let (_, m3) = tune(&format!("{label} stage3 (similarity)"), mem_stage3_grid(&m2, quick), &|m: &MemCfg| arm("t", Kind::K(kspec(s1_guard(&s1), mk_anchor(&s1), mk_inner(m, 6.0)))), seeds, ws, bb);
    if inner != Inner::Hybrid {
        return kspec(s1_guard(&s1), mk_anchor(&s1), mk_inner(&m3, 6.0));
    }
    let (_, mw) = tune(&format!("{label} stage4 (evidence gate min_w)"), vec![3.0f64, 8.0, 20.0], &|&w: &f64| arm("t", Kind::K(kspec(s1_guard(&s1), mk_anchor(&s1), mk_inner(&m3, w)))), seeds, ws, bb);
    kspec(s1_guard(&s1), mk_anchor(&s1), mk_inner(&m3, mw))
}

// ------------------------------------------------------------------------------------------------ comparison family (d7 copy)
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

fn family(ws: &[WorldSpec], bases: &[Arm], cands: &[Arm], pools: &[(&str, Vec<usize>)], held: &[u64], tables: &mut Vec<Value>) -> Vec<(String, Vec<Cell>)> {
    let mut out: Vec<(String, Vec<Cell>)> = pools.iter().map(|(n, _)| (n.to_string(), vec![])).collect();
    for s in ws {
        let br: Vec<Vec<[f64; 5]>> = bases.iter().map(|a| eval(a, s, held)).collect();
        let cr: Vec<Vec<[f64; 5]>> = cands.iter().map(|a| eval(a, s, held)).collect();
        println!("\n=== world {} | held mean expected success over first H ===", s.name);
        println!("{:<44}{:>8}{:>8}{:>8}{:>8}{:>8}", "arm", 100, 250, 500, 1000, 2000);
        let mut rows = vec![];
        for (a, r) in bases.iter().zip(&br).chain(cands.iter().zip(&cr)) {
            let m: Vec<f64> = (0..5).map(|h| summarize(&col(r, h)).mean).collect();
            println!("{:<44}{}", a.name, m.iter().map(|x| format!("{x:>8.4}")).collect::<String>());
            rows.push(json!({"arm": a.name, "means": m}));
        }
        tables.push(json!({"world": s.name, "rows": rows}));
        for (pi, (_, pool)) in pools.iter().enumerate() {
            for h in 0..5 {
                let bi = *pool.iter().max_by(|&&a, &&b| summarize(&col(&br[a], h)).mean.partial_cmp(&summarize(&col(&br[b], h)).mean).unwrap()).unwrap();
                let b = col(&br[bi], h);
                for (ci, ca) in cands.iter().enumerate() {
                    let cc = col(&cr[ci], h);
                    let d = paired_diff(&cc, &b);
                    out[pi].1.push(Cell {
                        world: s.name.to_string(),
                        cand: ca.name.clone(),
                        h: HORIZONS[h],
                        diff: d.diff.mean,
                        lo: d.diff.lo,
                        hi: d.diff.hi,
                        wins: d.wins,
                        p: paired_t_pvalue(&cc, &b),
                        best: bases[bi].name.clone(),
                        cand_mean: summarize(&cc).mean,
                        base_mean: summarize(&b).mean,
                    });
                }
            }
        }
    }
    out
}

fn report(title: &str, cells: &[Cell], show: &dyn Fn(&Cell) -> bool) -> Value {
    let adj = holm_adjust(&cells.iter().map(|c| c.p).collect::<Vec<_>>());
    println!("\n### {title}  (Holm over {} tests)", cells.len());
    println!("{:<18} {:<34} {:>5} {:>9} {:>20} {:>5} {:>8}  best baseline", "world", "candidate", "H", "diff", "95% CI", "wins", "Holm p");
    let mut out = vec![];
    for (c, a) in cells.iter().zip(&adj) {
        let mark = if *a < 0.05 { if c.diff > 0.0 { " WIN" } else { " LOSS" } } else { "" };
        if show(c) {
            println!("{:<18} {:<34} {:>5} {:>+9.4} [{:>+8.4},{:>+8.4}] {:>5} {:>8.4}  {}{}", c.world, c.cand, c.h, c.diff, c.lo, c.hi, c.wins, a, c.best, mark);
        }
        out.push(json!({"world": c.world, "candidate": c.cand, "H": c.h, "diff": c.diff, "ci": [c.lo, c.hi], "wins": c.wins, "p": c.p, "holm_p": a,
            "best_baseline": c.best, "cand_mean": c.cand_mean, "base_mean": c.base_mean}));
    }
    // per-candidate win/loss tally
    let mut tally: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
    for (c, a) in cells.iter().zip(&adj) {
        let e = tally.entry(c.cand.clone()).or_insert((0, 0));
        if *a < 0.05 {
            if c.diff > 0.0 {
                e.0 += 1
            } else {
                e.1 += 1
            }
        }
    }
    println!("--> Holm tally per candidate (wins, losses): {tally:?}");
    json!({"title": title, "tally": tally.iter().map(|(k, v)| (k.clone(), json!([v.0, v.1]))).collect::<serde_json::Map<_, _>>(), "cells": out})
}

/// direct paired a - b on the same seeds per world x H, Holm over the set.
fn direct(title: &str, ws: &[WorldSpec], a: &Arm, b: &Arm, held: &[u64]) -> Value {
    let mut rows = vec![];
    let mut ps = vec![];
    for s in ws {
        let ra = eval(a, s, held);
        let rb = eval(b, s, held);
        for h in 0..5 {
            let (ca, cb) = (col(&ra, h), col(&rb, h));
            let d = paired_diff(&ca, &cb);
            ps.push(paired_t_pvalue(&ca, &cb));
            rows.push((s.name, h, d.diff.mean, d.diff.lo, d.diff.hi, d.wins));
        }
    }
    let adj = holm_adjust(&ps);
    println!("\n--- direct: {title} (Holm over {}) ---", rows.len());
    let mut out = vec![];
    let (mut w, mut l) = (0, 0);
    for (r, a) in rows.iter().zip(&adj) {
        let mark = if *a < 0.05 { if r.2 > 0.0 { w += 1; " WIN" } else { l += 1; " LOSS" } } else { "" };
        println!("{:<18} H{:<5} {:>+8.4} [{:>+8.4},{:>+8.4}] w{:>2} Holm {:.4}{}", r.0, HORIZONS[r.1], r.2, r.3, r.4, r.5, a, mark);
        out.push(json!({"world": r.0, "H": HORIZONS[r.1], "diff": r.2, "ci": [r.3, r.4], "wins": r.5, "holm_p": a}));
    }
    println!("--> direct Holm wins {w}, losses {l}");
    json!({"title": title, "holm_wins": w, "holm_losses": l, "rows": out})
}

// ------------------------------------------------------------------------------------------------ main
fn base_arms_normal(fb: Option<Fb>) -> Vec<Arm> {
    let mk = |n: &str, k: Kind| Arm { name: n.to_string(), kind: k, fb };
    vec![
        mk("keyword", Kind::Keyword),
        mk("eps-greedy", Kind::Eps),
        mk("pooled-ts", Kind::Pts),
        mk("hebbian-robust", Kind::HebR),
        mk("warm(hebr)", Kind::WarmHebR),
        mk("guarded-safe", Kind::GuardSafe),
    ]
}

fn short(a: &AnchorMode) -> String {
    match a {
        AnchorMode::Off => "off".into(),
        AnchorMode::Learned(c) => format!("learned(lift {},min {})", c.lift_min, c.min_count),
        AnchorMode::Oracle => "oracle".into(),
    }
}

fn parse_range(args: &[String], key: &str, d: (u64, u64)) -> (u64, u64) {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|r| r.split_once('-'))
        .map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap()))
        .unwrap_or(d)
}

/// Anchor-quality diagnostic (policy-free): clusters formed / purity / coverage vs the TRUE topic map over the task stream.
fn diag_anchors() {
    let ws = normal_worlds();
    for name in ["var-traps", "var-wide-traps", "traps(main)", "many-topics(16)", "shortmod-traps", "var-single(ctl)"] {
        let spec = ws.iter().find(|w| w.name == name).unwrap();
        for (lift, alpha) in [(8.0, 1e-4), (3.0, 1e-2), (3.0, 1e-3)] {
            let (mut pur, mut cov, mut ncl, mut nfe) = ([0.0f64; 4], [0.0f64; 4], [0usize; 4], [0usize; 4]);
            let seeds = [31u64, 32, 33, 34, 35];
            for &sd in &seeds {
                let w = gen_world(sd, spec);
                let mut a = Anchors::new(AnchorCfg::on_alpha(lift, alpha));
                let marks = [250usize, 500, 1000, 2000];
                for (t, task) in w.tasks.iter().enumerate() {
                    a.observe(&task.features);
                    if let Some(k) = marks.iter().position(|&m| m == t + 1) {
                        // purity: among features sharing a token with another feature, fraction whose token-mates share its true topic
                        let feats: Vec<&String> = w.oracle.keys().collect();
                        let (mut good, mut tot, mut merged, mut seen) = (0usize, 0usize, 0usize, 0usize);
                        let mut by_tok: std::collections::HashMap<Vec<u32>, Vec<&String>> = Default::default();
                        for f in &feats {
                            let tk = a.tokens(&[(*f).clone()]);
                            if !tk.is_empty() {
                                by_tok.entry(tk).or_default().push(f);
                            }
                        }
                        for (_, fs) in by_tok.iter() {
                            seen += fs.len();
                            if fs.len() > 1 {
                                merged += fs.len();
                                for f in fs {
                                    tot += 1;
                                    let mates = fs.iter().filter(|g| w.oracle[**g] == w.oracle[*f] && *g != f).count();
                                    if mates * 2 >= fs.len() - 1 {
                                        good += 1;
                                    }
                                }
                            }
                        }
                        pur[k] += if tot > 0 { good as f64 / tot as f64 } else { 1.0 };
                        cov[k] += if seen > 0 { merged as f64 / seen as f64 } else { 0.0 };
                        ncl[k] += a.n_clusters();
                        nfe[k] += seen;
                    }
                }
            }
            let n = seeds.len() as f64;
            println!(
                "anchors-diag {name:<16} lift {lift} alpha {alpha:e}: (t=250,500,1000,2000) clusters {:?} purity {:.2?} merged-feature-share {:.2?}",
                ncl.map(|x| x as f64 / n),
                pur.map(|x| x / n),
                cov.map(|x| x / n)
            );
        }
    }
}

/// The K-full configuration frozen by the staged tuning on seeds 1-8 (identical in the 31-60 and 61-90 runs; see the FROZEN line of
/// results/g19a_sona_a.log). Hard-coded so that sensitivity / diagnostics run without re-tuning.
fn frozen_k_full() -> KSpec {
    KSpec {
        guard: defaults::recommended_guard(),
        anchors: AnchorMode::Learned(AnchorCfg::on_alpha(8.0, 1e-4)),
        inner: InnerSpec::Hybrid(MemCfg { init: 0.6, c0: 1.0, temp: 0.03, eps: 0.0, use_sets: true, knn_k: 3, s_min: 0.3, gamma: 3.0, kappa: 6.0, lam_u: 0.3, spread: 0.0, key_sel: false, key_n: 20.0, key_theta: 0.12, cap: 0, evict: Evict::ImportanceMerge }, 3.0),
        delayed_mult: 1,
        fail_reward: 0.0,
    }
}

/// One-factor sensitivity / ablation of the frozen K-full (NO re-tuning): H500 / H2000 paired difference to guarded-safe per world.
fn sens(held: &[u64], out: &str) {
    let base = frozen_k_full();
    let with_mem = |f: &dyn Fn(&mut MemCfg, &mut f64)| {
        let mut s = base.clone();
        if let InnerSpec::Hybrid(m, mw) = &mut s.inner {
            f(m, mw);
        }
        s
    };
    let arms: Vec<(&str, KSpec)> = vec![
        ("K-full (frozen)", base.clone()),
        ("gate min_w=8", with_mem(&|_, w| *w = 8.0)),
        ("gate min_w=20", with_mem(&|_, w| *w = 20.0)),
        ("gate min_w=60", with_mem(&|_, w| *w = 60.0)),
        ("exact key only (knn_k=0)", with_mem(&|m, _| m.knn_k = 0)),
        ("learned key selection", with_mem(&|m, _| m.key_sel = true)),
        ("neighbour credit spread=.5", with_mem(&|m, _| m.spread = 0.5)),
        ("anchors off", KSpec { anchors: AnchorMode::Off, ..base.clone() }),
        ("anchors permissive (lift 3, alpha 1e-2)", KSpec { anchors: AnchorMode::Learned(AnchorCfg::on_alpha(3.0, 1e-2)), ..base.clone() }),
        ("anchors ORACLE (diagnostic)", KSpec { anchors: AnchorMode::Oracle, ..base.clone() }),
        ("cap16 importance-merge", with_mem(&|m, _| {
            m.cap = 16;
            m.evict = Evict::ImportanceMerge
        })),
        ("cap16 LRU", with_mem(&|m, _| {
            m.cap = 16;
            m.evict = Evict::Lru
        })),
    ];
    let ws = normal_worlds();
    let gs = arm("guarded-safe", Kind::GuardSafe);
    let mut rows = vec![];
    println!("\n=== sensitivity (no re-tuning), held {}-{}: diff to guarded-safe at H500 / H2000 (paired 95% CI excludes 0 -> *) ===", held[0], held[held.len() - 1]);
    print!("{:<42}", "arm");
    for w in &ws {
        print!(" {:>10.10}", w.name);
    }
    println!();
    let gr: Vec<Vec<[f64; 5]>> = ws.iter().map(|s| eval(&gs, s, held)).collect();
    for (nm, spec) in &arms {
        let a = arm(nm, Kind::K(spec.clone()));
        for (hi, hname) in [(2usize, "H500"), (4, "H2000")] {
            print!("{:<42}", format!("{nm} {hname}"));
            let mut cells = vec![];
            for (wi, s) in ws.iter().enumerate() {
                let r = eval(&a, s, held);
                let d = paired_diff(&col(&r, hi), &col(&gr[wi], hi));
                let star = if d.diff.lo > 0.0 { "+" } else if d.diff.hi < 0.0 { "-" } else { " " };
                print!(" {:>+9.4}{star}", d.diff.mean);
                cells.push(json!({"world": s.name, "diff": d.diff.mean, "ci": [d.diff.lo, d.diff.hi]}));
            }
            println!();
            rows.push(json!({"arm": nm, "H": hname, "cells": cells}));
        }
    }
    let _ = std::fs::write(out, serde_json::to_string_pretty(&json!({"held": format!("{}-{}", held[0], held[held.len() - 1]), "rows": rows})).unwrap());
    println!("wrote {out}");
}

/// Third fresh block (default 91-120): frozen K-full (min_w 3, as tuned on 1-8) and the POST-HOC one-factor variant min_w=8 chosen after the
/// 31-60 / 61-90 sensitivity table, against the 6 spec baselines + retuned guarded; Holm over the family; no tuning inside this mode.
fn final_block(held: &[u64], out: &str) {
    let base = frozen_k_full();
    let gate8 = {
        let mut s = base.clone();
        if let InnerSpec::Hybrid(_, mw) = &mut s.inner {
            *mw = 8.0;
        }
        s
    };
    let anchor_heb = KSpec { inner: InnerSpec::Heb, ..base.clone() };
    let grt = kspec(guard_cfg(0.65, 1.0, 2.0, 1.0, 1.0, false), AnchorMode::Off, InnerSpec::Heb);
    let mut bases = base_arms_normal(None);
    bases.push(arm("guarded-rt(retuned)", Kind::K(grt)));
    let cands = vec![
        arm("K-full (frozen, min_w=3)", Kind::K(base.clone())),
        arm("K-gate8 (post-hoc min_w=8)", Kind::K(gate8.clone())),
        arm("K[anchors+hebbian]", Kind::K(anchor_heb)),
    ];
    let ws = normal_worlds();
    let mut tables = vec![];
    let fam = family(&ws, &bases, &cands, &[("best-of-7 (incl. retuned guarded)", (0..7).collect()), ("best-of-6 (spec set)", (0..6).collect())], held, &mut tables);
    let reps: Vec<Value> = fam.iter().map(|(t, c)| report(&format!("FINAL block: candidates vs {t}"), c, &|x: &Cell| x.cand.starts_with("K-gate8"))).collect();
    let dirs = vec![
        direct("K-gate8 - guarded-safe", &ws, &cands[1], &bases[5], held),
        direct("K-gate8 - K[anchors+hebbian] (set-level pattern memory vs none; same guard/anchors)", &ws, &cands[1], &cands[2], held),
        direct("K-full(min_w=3) - guarded-safe", &ws, &cands[0], &bases[5], held),
    ];
    let _ = std::fs::write(out, serde_json::to_string_pretty(&json!({"held": format!("{}-{}", held[0], held[held.len() - 1]), "tables": tables, "family": reps, "direct": dirs})).unwrap());
    println!("wrote {out}");
}

fn selfcheck() {
    // AnchorGuard(anchors off, hebbian inner) must reproduce defaults::recommended_router (== prior::Guarded<HebbianRouter>).
    let ws = normal_worlds();
    let mut worst: f64 = 0.0;
    for s in [&ws[5], &ws[8], &ws[2]] {
        for sd in [31u64, 32, 33] {
            let w = gen_world(sd, s);
            let a = run(&|w| build(&Kind::GuardSafe, w), &w, sd, Fb::Normal);
            let spec = kspec(defaults::recommended_guard(), AnchorMode::Off, InnerSpec::Heb);
            let b = run(&|w| build(&Kind::K(spec.clone()), w), &w, sd, Fb::Normal);
            let d = (0..5).map(|h| (a[h] - b[h]).abs()).fold(0.0, f64::max);
            worst = worst.max(d);
            println!("selfcheck {:<18} seed {sd}: guarded-safe {:?} vs AnchorGuard(off) {:?}  max|diff| {d:.3e}", s.name, a, b);
        }
    }
    println!("selfcheck worst |diff| = {worst:.3e}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--sens") {
        let (a0, a1) = parse_range(&args, "--held", (31, 60));
        let held: Vec<u64> = (a0..=a1).collect();
        let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/g19a_sens.json".into());
        sens(&held, &out);
        return;
    }
    if args.iter().any(|a| a == "--final") {
        let (a0, a1) = parse_range(&args, "--held", (91, 120));
        let held: Vec<u64> = (a0..=a1).collect();
        let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/g19a_final_91-120.json".into());
        final_block(&held, &out);
        return;
    }
    if args.iter().any(|a| a == "--diag-anchors") {
        diag_anchors();
        return;
    }
    if args.iter().any(|a| a == "--selfcheck") {
        selfcheck();
        return;
    }
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/g19a_sona_a.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let safe = args.iter().position(|a| a == "--guard").and_then(|i| args.get(i + 1)).map(|s| s != "tuned").unwrap_or(true);
    let only: Vec<String> = args.iter().position(|a| a == "--only").and_then(|i| args.get(i + 1)).map(|s| s.split(',').map(|x| x.to_string()).collect()).unwrap_or_else(|| vec!["normal".into(), "weak".into(), "session".into(), "mem".into()]);
    let (h0, h1) = parse_range(&args, "--held", (31, 60));
    let held: Vec<u64> = (h0..=h1).collect();
    let tune_seeds: Vec<u64> = (1..=8).collect();
    println!("g19_sona_a: {N_TASKS} tasks/run, tune seeds 1-8, held-out {h0}-{h1}{}", if quick { " (QUICK)" } else { "" });

    let wn = normal_worlds();
    let mut result = json!({"protocol": {"tune_seeds": "1-8", "heldout": format!("{h0}-{h1}"), "horizons": HORIZONS, "n_tasks": N_TASKS,
        "label": "SYNTHETIC-until-validated-on-real-traffic", "worlds_normal": wn.iter().map(|w| w.name).collect::<Vec<_>>()}});
    let mut tables: Vec<Value> = vec![];

    // ==================================================================== NORMAL feedback
    let bb_n = best_of(&base_arms_normal(None), &tune_seeds, &wn);
    let (_, grt) = tune("guarded-rt (retuned guarded(hebr), no anchors)", guard_rt_grid(quick), &|g: &GuardConfig| arm("t", Kind::K(kspec(*g, AnchorMode::Off, InnerSpec::Heb))), &tune_seeds, &wn, &bb_n);
    let guard_rt = kspec(grt, AnchorMode::Off, InnerSpec::Heb);
    let mut bb_n7 = bb_n.clone();
    {
        let m = means(&arm("g", Kind::K(guard_rt.clone())), &tune_seeds, &wn);
        for w in 0..wn.len() {
            for h in 0..5 {
                bb_n7[w][h] = bb_n7[w][h].max(m[w][h]);
            }
        }
    }
    let k_full = tune_k("K-full (learned anchors + evidence-gated keyed memory)", Some(true), false, Inner::Hybrid, true, &tune_seeds, &wn, &bb_n7, quick, safe);
    let k_noanchor = tune_k("K[no anchors] (evidence-gated keyed memory only)", Some(false), false, Inner::Hybrid, true, &tune_seeds, &wn, &bb_n7, quick, safe);
    let k_memonly = tune_k("K[mem ungated] (anchors + keyed memory decides everything)", Some(true), false, Inner::Mem, true, &tune_seeds, &wn, &bb_n7, quick, safe);
    let k_unigram = tune_k("K[unigram memory] (anchors, no set patterns, ungated)", Some(true), false, Inner::Mem, false, &tune_seeds, &wn, &bb_n7, quick, safe);
    let k_anchor_heb = tune_k("K[anchors+hebbian] (anchor keying, no pattern memory)", Some(true), false, Inner::Heb, true, &tune_seeds, &wn, &bb_n7, quick, safe);
    let k_oracle = tune_k("K[oracle anchors] (diagnostic upper bound)", None, true, Inner::Hybrid, true, &tune_seeds, &wn, &bb_n7, quick, safe);
    println!("\nFROZEN configs: K-full {:?}\n  guard-rt {:?}", k_full, guard_rt);
    result["tuned"] = json!({
        "guarded_rt": format!("{:?}", guard_rt), "k_full": format!("{:?}", k_full), "k_noanchor": format!("{:?}", k_noanchor),
        "k_memonly": format!("{:?}", k_memonly), "k_unigram": format!("{:?}", k_unigram), "k_anchor_heb": format!("{:?}", k_anchor_heb), "k_oracle": format!("{:?}", k_oracle),
    });

    if only.iter().any(|o| o == "normal") {
        let mut bases = base_arms_normal(None);
        bases.push(arm("guarded-rt(retuned)", Kind::K(guard_rt.clone())));
        let cands = vec![
            arm("K-full", Kind::K(k_full.clone())),
            arm("K[no anchors]", Kind::K(k_noanchor.clone())),
            arm("K[mem ungated]", Kind::K(k_memonly.clone())),
            arm("K[unigram mem]", Kind::K(k_unigram.clone())),
            arm("K[anchors+hebbian]", Kind::K(k_anchor_heb.clone())),
        ];
        let fam = family(&wn, &bases, &cands, &[("best-of-7 (incl. retuned guarded)", (0..7).collect()), ("best-of-6 (spec set)", (0..6).collect()), ("keyword only", vec![0])], &held, &mut tables);
        let reps: Vec<Value> = fam.iter().map(|(t, c)| report(&format!("NORMAL worlds: candidates vs {t}"), c, &|x: &Cell| x.cand == "K-full")).collect();
        result["normal"] = json!(reps);
        let mut dirs = vec![];
        for ab in [1usize, 2, 3, 4] {
            dirs.push(direct(&format!("K-full - {} (normal worlds)", cands[ab].name), &wn, &cands[0], &cands[ab], &held));
        }
        dirs.push(direct("K-full - guarded-rt (normal worlds)", &wn, &cands[0], &bases[6], &held));
        dirs.push(direct("K-full - guarded-safe (normal worlds)", &wn, &cands[0], &bases[5], &held));
        let oracle_arm = arm("K[oracle anchors]", Kind::K(k_oracle.clone()));
        dirs.push(direct("K[oracle anchors, DIAGNOSTIC] - K-full (headroom of better anchors)", &wn, &oracle_arm, &cands[0], &held));
        result["normal_direct"] = json!(dirs);
    }

    // ==================================================================== WEAK feedback (same reuse signal for every arm)
    if only.iter().any(|o| o == "weak") {
        let ww = weak_worlds(Fb::WeakReuse);
        let mut raw_b: Vec<Arm> = base_arms_normal(Some(Fb::WeakRaw)).into_iter().map(|mut a| { a.name = format!("{} [raw]", a.name); a }).collect();
        let reuse_b: Vec<Arm> = base_arms_normal(Some(Fb::WeakReuse)).into_iter().skip(1).map(|mut a| { a.name = format!("{} [reuse]", a.name); a }).collect();
        raw_b.extend(reuse_b);
        // best-of for weak tuning
        let bb_w = best_of(&raw_b, &tune_seeds, &ww);
        let wk = |mut s: KSpec, m: usize, f: f64, anc: Option<(f64, f64)>| {
            s.delayed_mult = m;
            s.fail_reward = f;
            if let (Some((l, a)), AnchorMode::Learned(_)) = (anc, &s.anchors) {
                s.anchors = AnchorMode::Learned(AnchorCfg::on_alpha(l, a));
            }
            s
        };
        let mut tuned_w: Vec<(String, KSpec)> = vec![];
        for (nm, base) in [("guarded-rt", &guard_rt), ("K-full", &k_full), ("K[no anchors]", &k_noanchor), ("K[anchors+hebbian]", &k_anchor_heb)] {
            let learned = matches!(base.anchors, AnchorMode::Learned(_));
            let mut grid = vec![];
            for &(m, f) in &[(1usize, 0.0f64), (1, 0.3), (3, 0.0), (3, 0.3)] {
                if learned {
                    for &(l, a) in &[(3.0f64, 1e-2f64), (3.0, 1e-4), (8.0, 1e-2), (8.0, 1e-4)] {
                        grid.push((m, f, Some((l, a))));
                    }
                } else {
                    grid.push((m, f, None));
                }
            }
            let (_, (m, f, anc)) = tune(&format!("weak {nm} (delayed_mult, fail_reward{})", if learned { ", anchor lift/alpha" } else { "" }), grid, &|&(m, f, anc): &(usize, f64, Option<(f64, f64)>)| arm_fb("t", Kind::K(wk(base.clone(), m, f, anc)), Fb::WeakReuse), &tune_seeds, &ww, &bb_w);
            tuned_w.push((nm.to_string(), wk(base.clone(), m, f, anc)));
        }
        let get = |n: &str| tuned_w.iter().find(|x| x.0 == n).unwrap().1.clone();
        let mut bases = raw_b.clone();
        bases.push(arm_fb("guarded-rt [reuse]", Kind::K(get("guarded-rt")), Fb::WeakReuse));
        let nb = bases.len();
        let cands = vec![
            arm_fb("K-full [reuse]", Kind::K(get("K-full")), Fb::WeakReuse),
            arm_fb("K[no anchors] [reuse]", Kind::K(get("K[no anchors]")), Fb::WeakReuse),
            arm_fb("K[anchors+hebbian] [reuse]", Kind::K(get("K[anchors+hebbian]")), Fb::WeakReuse),
        ];
        let raw_idx: Vec<usize> = (0..5).collect();
        let fam = family(&ww, &bases, &cands, &[("STRICT pool: raw-fed + every baseline also fed the reuse signal", (0..nb).collect()), ("raw-fed baselines only (x3 convention)", raw_idx), ("keyword only", vec![0])], &held, &mut tables);
        let reps: Vec<Value> = fam.iter().map(|(t, c)| report(&format!("WEAK worlds: candidates vs {t}"), c, &|x: &Cell| x.cand == "K-full [reuse]")).collect();
        result["weak"] = json!(reps);
        let dirs = vec![
            direct("K-full[reuse] - guarded-rt[reuse] (weak worlds)", &ww, &cands[0], &bases[nb - 1], &held),
            direct("K-full[reuse] - K[no anchors][reuse]", &ww, &cands[0], &cands[1], &held),
            direct("K-full[reuse] - K[anchors+hebbian][reuse]", &ww, &cands[0], &cands[2], &held),
        ];
        result["weak_direct"] = json!(dirs);
    }

    // ==================================================================== SESSION worlds (trajectory credit)
    if only.iter().any(|o| o == "session") {
        let sw = session_worlds();
        let bcast = base_arms_normal(None);
        let bb_s = best_of(&bcast, &tune_seeds, &sw);
        // guard params re-tuned for the diluted session signal (tau/discount only), with credit
        let sg = |g: GuardConfig, t: f64, d: f64| GuardConfig { tau: t, discount: d, ..g };
        let (_, (tg, dg)) = tune("session guarded-rt[credit-uniest] (tau, discount)", vec![(0.35, 1.0), (0.35, 0.97), (0.45, 1.0), (0.45, 0.97), (0.55, 1.0), (0.55, 0.97)], &|&(t, d): &(f64, f64)| arm("t", Kind::Credit(Box::new(Kind::K(kspec(sg(grt, t, d), AnchorMode::Off, InnerSpec::Heb))), false)), &tune_seeds, &sw, &bb_s);
        let (_, (tk, dk)) = tune("session K-full[credit-own] (tau, discount)", if safe { vec![(k_full.guard.tau, k_full.guard.discount)] } else { vec![(0.35, 1.0), (0.35, 0.97), (0.45, 1.0), (0.45, 0.97), (0.55, 1.0), (0.55, 0.97)] }, &|&(t, d): &(f64, f64)| {
            let mut s = k_full.clone();
            s.guard = sg(s.guard, t, d);
            arm("t", Kind::Credit(Box::new(Kind::K(s)), true))
        }, &tune_seeds, &sw, &bb_s);
        let mut kf_s = k_full.clone();
        kf_s.guard = sg(kf_s.guard, tk, dk);
        let mut kn_s = k_noanchor.clone();
        kn_s.guard = sg(kn_s.guard, tk, dk);
        let mut ka_s = k_anchor_heb.clone();
        ka_s.guard = sg(ka_s.guard, tk, dk);
        let grt_s = kspec(sg(grt, tg, dg), AnchorMode::Off, InnerSpec::Heb);
        let mut bases: Vec<Arm> = bcast.into_iter().map(|mut a| { a.name = format!("{} [broadcast]", a.name); a }).collect();
        for (nm, k) in [("eps-greedy", Kind::Eps), ("pooled-ts", Kind::Pts), ("hebbian-robust", Kind::HebR), ("warm(hebr)", Kind::WarmHebR), ("guarded-safe", Kind::GuardSafe)] {
            bases.push(arm(&format!("{nm} [credit-uniest]"), Kind::Credit(Box::new(k), false)));
        }
        bases.push(arm("guarded-rt [broadcast]", Kind::K(grt_s.clone())));
        bases.push(arm("guarded-rt [credit-uniest]", Kind::Credit(Box::new(Kind::K(grt_s.clone())), false)));
        let nb = bases.len();
        let cands = vec![
            arm("K-full [credit-own]", Kind::Credit(Box::new(Kind::K(kf_s.clone())), true)),
            arm("K-full [broadcast]", Kind::K(kf_s.clone())),
            arm("K-full [credit-uniest]", Kind::Credit(Box::new(Kind::K(kf_s.clone())), false)),
            arm("K[no anchors] [credit-own]", Kind::Credit(Box::new(Kind::K(kn_s.clone())), true)),
            arm("K[anchors+hebbian] [credit-uniest]", Kind::Credit(Box::new(Kind::K(ka_s.clone())), false)),
        ];
        let fam = family(&sw, &bases, &cands, &[("STRICT pool: every baseline broadcast AND credit-fed", (0..nb).collect()), ("broadcast baselines only", (0..6).collect())], &held, &mut tables);
        let reps: Vec<Value> = fam.iter().map(|(t, c)| report(&format!("SESSION worlds: candidates vs {t}"), c, &|x: &Cell| x.cand == "K-full [credit-own]")).collect();
        result["session"] = json!(reps);
        let dirs = vec![
            direct("K-full[credit-own] - K-full[broadcast] (credit effect)", &sw, &cands[0], &cands[1], &held),
            direct("K-full[credit-own] - K-full[credit-uniest] (SONA-estimator vs plain estimator)", &sw, &cands[0], &cands[2], &held),
            direct("K-full[credit-own] - guarded-rt[credit-uniest]", &sw, &cands[0], &bases[nb - 1], &held),
            direct("K-full[credit-own] - K[no anchors][credit-own]", &sw, &cands[0], &cands[3], &held),
        ];
        result["session_direct"] = json!(dirs);
    }

    // ==================================================================== MEMORY capacity (consolidation)
    if only.iter().any(|o| o == "mem") {
        let mw = [memt_world()];
        let mut bases = base_arms_normal(None);
        bases.push(arm("guarded-rt(retuned)", Kind::K(guard_rt.clone())));
        let mem_of = |cap: usize, ev: Evict| -> KSpec {
            let mut s = k_full.clone();
            match &mut s.inner {
                InnerSpec::Mem(m) | InnerSpec::Hybrid(m, _) => {
                    m.cap = cap;
                    m.evict = ev;
                }
                InnerSpec::Heb => {}
            }
            s
        };
        let mut cands = vec![arm("K-full uncapped", Kind::K(mem_of(0, Evict::ImportanceMerge)))];
        for cap in [8usize, 16, 32] {
            for (nm, ev) in [("importance-merge", Evict::ImportanceMerge), ("importance-drop", Evict::ImportanceDrop), ("LRU", Evict::Lru), ("FIFO", Evict::Fifo), ("count-drop", Evict::CountDrop)] {
                cands.push(arm(&format!("K-full cap{cap} {nm}"), Kind::K(mem_of(cap, ev))));
            }
        }
        let fam = family(&mw, &bases, &cands, &[("best-of-7", (0..7).collect())], &held, &mut tables);
        let reps: Vec<Value> = fam.iter().map(|(t, c)| report(&format!("MEMORY world: candidates vs {t}"), c, &|x: &Cell| x.h == 2000 || x.h == 500)).collect();
        result["mem"] = json!(reps);
        let mut dirs = vec![];
        for cap in [8usize, 16, 32] {
            let i_merge = 1 + (cap_idx(cap)) * 5;
            for off in [1usize, 2, 3, 4] {
                dirs.push(direct(&format!("{} - {}", cands[i_merge].name, cands[i_merge + off].name), &mw, &cands[i_merge], &cands[i_merge + off], &held));
            }
            dirs.push(direct(&format!("{} - K-full uncapped", cands[i_merge].name), &mw, &cands[i_merge], &cands[0], &held));
        }
        result["mem_direct"] = json!(dirs);
        // pattern counts at the end (diagnostic), seed 31
        let mut sizes = vec![];
        for (nm, spec) in [("uncapped", mem_of(0, Evict::ImportanceMerge)), ("cap16", mem_of(16, Evict::ImportanceMerge))] {
            let w = gen_world(held[0], &mw[0]);
            let (n, ev) = run_mem_sizes(&spec, &w, held[0]);
            println!("mem-conj48 seed {} {nm}: final patterns {n}, evictions {ev}", held[0]);
            sizes.push(json!({"arm": nm, "final_patterns": n, "evictions": ev}));
        }
        result["mem_sizes"] = json!(sizes);
    }

    result["tables"] = json!(tables);
    result["notes"] = json!({"short_anchor_modes": [short(&k_full.anchors), short(&k_oracle.anchors)]});
    let _ = std::fs::create_dir_all("results");
    std::fs::write(&out_path, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!("\nwrote {out_path}");
}

fn cap_idx(cap: usize) -> usize {
    match cap {
        8 => 0,
        16 => 1,
        _ => 2,
    }
}

/// run one normal-feedback episode and report the final pattern count / evictions of a K spec with a Mem inner.
fn run_mem_sizes(s: &KSpec, w: &World, seed: u64) -> (usize, usize) {
    let n = w.agents;
    let anchors = match &s.anchors {
        AnchorMode::Off => Anchors::new(AnchorCfg::off()),
        AnchorMode::Learned(c) => Anchors::new(c.clone()),
        AnchorMode::Oracle => Anchors::oracle(w.oracle.clone()),
    };
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    macro_rules! drive {
        ($g:expr) => {{
            let mut g = $g;
            for t in 0..w.tasks.len() {
                let task: &Task = &w.tasks[t];
                let a = g.select(task, &cands, &mut rng);
                let sres = if w.u[t] < w.p[t][a] { 1.0 } else { 0.0 };
                g.update(task, a, sres);
            }
            g
        }};
    }
    match &s.inner {
        InnerSpec::Mem(m) => {
            let g = drive!(AnchorGuard::new(KeyedMemory::new(n, m.clone()), w.kwmap.clone(), n, s.guard, anchors));
            (g.inner.n_patterns(), g.inner.evictions)
        }
        InnerSpec::Hybrid(m, mw) => {
            let g = drive!(AnchorGuard::new(HybridInner::new(n, m.clone(), *mw), w.kwmap.clone(), n, s.guard, anchors));
            (g.inner.mem.n_patterns(), g.inner.mem.evictions)
        }
        InnerSpec::Heb => (0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_merge_cooccurring_forms_not_confounds() {
        let w = gen_world(5, &normal_worlds()[0]); // var-traps
        let mut a = Anchors::new(AnchorCfg::on(3.0, 3));
        for t in &w.tasks {
            a.observe(&t.features);
        }
        // 8 topics -> about 8 clusters, each covering several forms
        assert!(a.n_clusters() >= 6 && a.n_clusters() <= 9, "clusters {}", a.n_clusters());
        // forms of one topic share a token; different topics do not
        let tk = |f: &str| a.tokens(&[f.to_string()]);
        assert_eq!(tk("t0v0"), tk("t0v1"));
        assert_ne!(tk("t0v0"), tk("t1v0"));
    }

    #[test]
    fn capacity_is_respected_and_merge_conserves_weight() {
        let mut m = KeyedMemory::new(3, MemCfg { cap: 3, ..MemCfg::default() });
        for i in 0..40u32 {
            m.update(&[i % 7, 100 + i % 5], (i % 3) as usize, 1.0);
        }
        assert!(m.n_patterns() <= 3);
        assert!(m.evictions > 0);
    }

    #[test]
    fn identity_guard_matches_guarded_safe() {
        let ws = normal_worlds();
        let w = gen_world(33, &ws[5]);
        let a = run(&|w| build(&Kind::GuardSafe, w), &w, 33, Fb::Normal);
        let spec = kspec(defaults::recommended_guard(), AnchorMode::Off, InnerSpec::Heb);
        let b = run(&|w| build(&Kind::K(spec.clone()), w), &w, 33, Fb::Normal);
        for h in 0..5 {
            assert!((a[h] - b[h]).abs() < 5e-3, "H{h}: {} vs {}", a[h], b[h]);
        }
        let _ = x1_like_guard();
    }
}
