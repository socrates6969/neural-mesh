//! F2(a): quantised / sparse / adaptive-precision / pruned router state: bytes vs routing quality.
//!
//! Builds on coupling.rs' x3 finding (u16 lossless, u8 broke learning). New here:
//!   * block-floating-point rows (shared f32 row scale, decay + row cap folded into the scale,
//!     only the Hebbian delta is rounded), stochastic vs nearest rounding,
//!   * 8/4-bit codes, per-row ADAPTIVE precision (rows start 8-bit, hot rows promoted to 16-bit),
//!   * pruning of cold feature rows under a row budget (evict by decayed update count),
//!   * a real packed serialisation (dense or sparse per row) so "bytes" are bytes of a state file.
//! Protocol: schemes tuned on seeds 1-8 (adaptive threshold only), held-out 31-60, confirm 61-90
//! frozen, paired 95% t-CI vs the f64 router, TOST equivalence (margin 0.003) with Holm, 4 worlds.
//!
//! Usage: cargo run --release --bin f2_quant [-- --out results/f2_quant.json]

#[path = "f2_common/mod.rs"]
mod common;

use common::*;
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router};
use neural_mesh::stats::holm_adjust;
use serde_json::{json, Value};

type WarmMini = Warm<MiniHeb>;

fn warm_mini(env: &Env, sch: &Scheme) -> WarmMini {
    let n = env.n_agents();
    Warm::new(MiniHeb::new(n, sch.clone()), env.kw_map(), n, 1.0).with_evidence(Evidence::Inner(mean_feature_n))
}

// =============================================================================================
// Self tests (the numbers below are only meaningful if these pass)

fn selftest() {
    // 1. f64 MiniHeb == crate HebbianRouter inside Warm(MeanFeature): identical decisions + weights
    for (seed, spec) in [(3u64, WorldSpec::Sim(neural_mesh::sim::Scenario::Fixed)), (4, WorldSpec::Big)] {
        let env = spec.make(seed);
        let n = env.n_agents();
        let mut a = warm_mini(&env, &Scheme::new("f64", WKind::F64, BKind::F64));
        let mut b = Warm::new(HebbianRouter::new(n, heb_robust_cfg()), env.kw_map(), n, 1.0)
            .with_evidence(Evidence::MeanFeature);
        let cands: Vec<usize> = (0..n).collect();
        let (mut r1, mut r2) = (Rng::new(7), Rng::new(7));
        for t in 0..env.len() {
            let task = env.task(t);
            let x = a.select(task, &cands, &mut r1);
            let y = b.select(task, &cands, &mut r2);
            assert_eq!(x, y, "MiniHeb(f64) diverged from HebbianRouter at {t} in {}", spec.label());
            let rw = env.outcome(t, x);
            a.update(task, x, rw);
            b.update(task, y, rw);
        }
        let mut worst = 0.0f64;
        for (f, row) in b.inner.rows() {
            for (ag, &w) in row.iter().enumerate() {
                worst = worst.max((w - a.inner.weight(f, ag)).abs());
            }
        }
        assert!(worst < 1e-12, "weights differ by {worst}");
    }
    // 2. packed state round-trips exactly (decoded cells, baselines, counts) for every layout
    let env = WorldSpec::Big.make(5);
    for sch in [
        Scheme::new("f64", WKind::F64, BKind::F64),
        Scheme::new("f32", WKind::F32, BKind::F32),
        Scheme::new("f16", WKind::F16, BKind::U16),
        Scheme::new("fx16", WKind::Fx { bits: 16, sr: false }, BKind::U16),
        Scheme::new("bf8", WKind::Bf { bits: 8, sr: true }, BKind::U8Sr),
        Scheme::new("bf4", WKind::Bf { bits: 4, sr: true }, BKind::U16),
        Scheme { promote_after: Some(30), ..Scheme::new("ad", WKind::Bf { bits: 8, sr: true }, BKind::U16) },
    ] {
        let mut w = warm_mini(&env, &sch);
        let cands: Vec<usize> = (0..env.n_agents()).collect();
        let mut rng = Rng::new(1);
        for t in 0..1500 {
            let task = env.task(t);
            let a = w.select(task, &cands, &mut rng);
            w.update(task, a, env.outcome(t, a));
        }
        let bytes = w.inner.to_bytes(true);
        let back = MiniHeb::from_bytes(&bytes, sch.clone());
        assert_eq!(back.n_rows(), w.inner.n_rows());
        assert_eq!(w.inner.max_cell_diff(&back), 0.0, "{} cells not round-trip exact", sch.name);
        for (name, _) in env.task(0).features.iter().map(|f| (f.clone(), 0)) {
            assert_eq!(w.inner.baseline(&name), back.baseline(&name), "{} baseline", sch.name);
        }
    }
    println!("selftest ok: MiniHeb(f64) decision-identical to HebbianRouter; packed state round-trips exactly");
}

// =============================================================================================
// One free-running evaluation

#[derive(Clone, Debug, Default)]
struct Out {
    full: f64,
    h500: f64,
    bytes_named: usize,
    bytes_hashed: usize,
    rows: usize,
    frac_hot: f64,
    evictions: u64,
}

fn run_scheme(spec: &WorldSpec, sch: &Scheme, seed: u64) -> Out {
    let env = spec.make(seed);
    let mut w = warm_mini(&env, sch);
    let st = run_policy(&env, &mut w, seed);
    Out {
        full: st.full,
        h500: st.h500,
        bytes_named: w.inner.to_bytes(true).len(),
        bytes_hashed: w.inner.to_bytes(false).len(),
        rows: w.inner.n_rows(),
        frac_hot: w.inner.frac_hot(),
        evictions: w.inner.evictions,
    }
}

/// Teacher-forced decision agreement: the reference drives; each twin sees the same history and
/// is asked, with the SAME random stream, what it would pick.
fn agreement(spec: &WorldSpec, schemes: &[Scheme], seed: u64) -> Vec<(f64, f64)> {
    let env = spec.make(seed);
    let n = env.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut driver = warm_mini(&env, &Scheme::new("f64", WKind::F64, BKind::F64));
    let mut twins: Vec<WarmMini> = schemes.iter().map(|s| warm_mini(&env, s)).collect();
    let mut hit = vec![0usize; schemes.len()];
    let mut dq = vec![0.0f64; schemes.len()];
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    for t in 0..env.len() {
        let task = env.task(t);
        let before = rng.clone();
        let a = driver.select(task, &cands, &mut rng);
        for (i, tw) in twins.iter_mut().enumerate() {
            let mut r = before.clone();
            let b = tw.select(task, &cands, &mut r);
            if b == a {
                hit[i] += 1;
            }
            dq[i] += env.prob(t, b) - env.prob(t, a);
        }
        let rw = env.outcome(t, a);
        driver.update(task, a, rw);
        for tw in twins.iter_mut() {
            tw.update(task, a, rw);
        }
    }
    hit.iter().zip(&dq).map(|(&h, &d)| (h as f64 / env.len() as f64, d / env.len() as f64)).collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

struct Cell {
    scheme: String,
    world: String,
    cmp: Cmp,
    cmp500: Cmp,
    bytes_named: f64,
    bytes_hashed: f64,
    ref_bytes_named: f64,
    rows: f64,
    frac_hot: f64,
    evictions: f64,
    agree: f64,
    tf: Cmp,
    hold_full: f64,
}

fn eval_split(worlds: &[WorldSpec], schemes: &[Scheme], sd: &[u64], with_agree: bool) -> Vec<Cell> {
    let refsch = Scheme::new("f64", WKind::F64, BKind::F64);
    let mut cells = Vec::new();
    for spec in worlds {
        let refs: Vec<Out> = par_map(sd, |&s| run_scheme(spec, &refsch, s));
        let ag: Vec<Vec<(f64, f64)>> = if with_agree { par_map(sd, |&s| agreement(spec, schemes, s)) } else { vec![] };
        for (si, sch) in schemes.iter().enumerate() {
            let outs: Vec<Out> = par_map(sd, |&s| run_scheme(spec, sch, s));
            let a: Vec<f64> = outs.iter().map(|o| o.full).collect();
            let b: Vec<f64> = refs.iter().map(|o| o.full).collect();
            let a5: Vec<f64> = outs.iter().map(|o| o.h500).collect();
            let b5: Vec<f64> = refs.iter().map(|o| o.h500).collect();
            cells.push(Cell {
                scheme: sch.name.clone(),
                world: spec.label(),
                cmp: compare(&a, &b),
                cmp500: compare(&a5, &b5),
                bytes_named: mean(&outs.iter().map(|o| o.bytes_named as f64).collect::<Vec<_>>()),
                bytes_hashed: mean(&outs.iter().map(|o| o.bytes_hashed as f64).collect::<Vec<_>>()),
                ref_bytes_named: mean(&refs.iter().map(|o| o.bytes_named as f64).collect::<Vec<_>>()),
                rows: mean(&outs.iter().map(|o| o.rows as f64).collect::<Vec<_>>()),
                frac_hot: mean(&outs.iter().map(|o| o.frac_hot).collect::<Vec<_>>()),
                evictions: mean(&outs.iter().map(|o| o.evictions as f64).collect::<Vec<_>>()),
                agree: if with_agree { mean(&ag.iter().map(|v| v[si].0).collect::<Vec<_>>()) } else { f64::NAN },
                tf: if with_agree {
                    let d: Vec<f64> = ag.iter().map(|v| v[si].1).collect();
                    compare(&d, &vec![0.0; d.len()])
                } else {
                    compare(&[0.0, 0.0], &[0.0, 0.0])
                },
                hold_full: mean(&a),
            });
        }
    }
    cells
}

/// Real-crate state size for context: JSON of the HebbianRouter's learned state (weights + baselines),
/// i.e. what the CLI persists today, at the end of a run.
fn json_state_bytes(spec: &WorldSpec, seed: u64) -> (usize, usize) {
    let env = spec.make(seed);
    let n = env.n_agents();
    let mut b = Warm::new(HebbianRouter::new(n, heb_robust_cfg()), env.kw_map(), n, 1.0).with_evidence(Evidence::MeanFeature);
    run_policy(&env, &mut b, seed);
    let v = serde_json::to_value(&b.inner).unwrap();
    let w = serde_json::to_string(&v["weights"]).unwrap().len() + serde_json::to_string(&v["baselines"]).unwrap().len();
    // resident estimate: dense f64 (x3 router_bytes formula)
    let rb = neural_mesh::coupling::router_bytes(&b.inner);
    (w, rb.dense_f64)
}

fn print_cells(title: &str, cells: &[Cell], holm: &[f64]) {
    println!("\n=== {title} ===");
    println!(
        "{:<30} {:<13} {:>6} {:>5} {:>4} {:>23} {:>6} {:>4} {:>6} {:>8} {:>4}",
        "scheme", "world", "bytes", "x_ref", "rows", "dQ(full) [95% CI]", "Holm", "LL", "agree", "dec-loss", "hot"
    );
    for (c, h) in cells.iter().zip(holm) {
        let ll = *h < 0.05 && c.cmp.lo > -MARGIN && c.cmp.hi < MARGIN;
        println!(
            "{:<30} {:<13} {:>6.0} {:>5.2} {:>4.0} {:>23} {:>6.3} {:>4} {:>6.4} {:>+8.5} {:>4.2}",
            c.scheme,
            c.world,
            c.bytes_named,
            c.bytes_named / c.ref_bytes_named,
            c.rows,
            fmt_cmp(&c.cmp),
            h,
            if ll { "YES" } else { "no" },
            c.agree,
            c.tf.mean,
            c.frac_hot
        );
    }
}

fn cells_json(cells: &[Cell], holm: &[f64]) -> Vec<Value> {
    cells
        .iter()
        .zip(holm)
        .map(|(c, h)| {
            json!({
                "scheme": c.scheme, "world": c.world,
                "bytes_named": c.bytes_named, "bytes_hashed_names": c.bytes_hashed,
                "bytes_ratio_vs_f64_packed": c.bytes_named / c.ref_bytes_named,
                "f64_packed_bytes": c.ref_bytes_named,
                "rows": c.rows, "frac_rows_16bit": c.frac_hot, "evictions": c.evictions,
                "teacher_forced_agreement": c.agree, "mean_full_quality": c.hold_full,
                "quality_full_vs_f64": cmp_json(&c.cmp, Some(*h)),
                "quality_h500_vs_f64": cmp_json(&c.cmp500, None),
                "decision_loss_same_history": cmp_json(&c.tf, None),
            })
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned();
    let t0 = std::time::Instant::now();
    selftest();
    let worlds = all_worlds();
    println!("cores available: {}", n_threads());

    // ---------------------------------------------------------------------------------------
    // tuning on seeds 1-8: adaptive-precision rule parameters
    let tune = tune_seeds();
    let bf8 = |name: &str, p: Option<u32>, g: Option<u16>| {
        let mut s = Scheme::new(name, WKind::Bf { bits: 8, sr: true }, BKind::U8Sr);
        s.promote_after = p;
        s.promote_gap = g;
        s
    };
    let mut adaptive: Vec<Scheme> = Vec::new();
    for p in [8u32, 32, 128] {
        adaptive.push(bf8(&format!("adaptive-hot@{p}"), Some(p), None));
    }
    for g in [2u16, 4, 8, 16, 32] {
        adaptive.push(bf8(&format!("adaptive-gap<{g}"), None, Some(g)));
    }
    let anchors = vec![
        Scheme::new("bf16-sr", WKind::Bf { bits: 16, sr: true }, BKind::U16),
        bf8("bf8-sr+u8base", None, None),
    ];
    println!("\n--- tuning on seeds 1-8 (worst-world dQ over 4 worlds, bytes relative to packed f64) ---");
    let mut tune_list = anchors.clone();
    tune_list.extend(adaptive.iter().cloned());
    let tc = eval_split(&worlds, &tune_list, &tune, false);
    let nw = worlds.len();
    let mut tune_rows = Vec::new();
    let mut chosen: Option<usize> = None;
    let mut best_bytes = f64::MAX;
    for (i, sch) in tune_list.iter().enumerate() {
        let cs: Vec<&Cell> = (0..nw).map(|w| &tc[w * tune_list.len() + i]).collect();
        let worst = cs.iter().map(|c| c.cmp.mean).fold(f64::MAX, f64::min);
        let rel = mean(&cs.iter().map(|c| c.bytes_named / c.ref_bytes_named).collect::<Vec<_>>());
        println!("  {:<22} worst-world dQ {:+.4}   bytes/f64 {:.3}", sch.name, worst, rel);
        tune_rows.push(json!({"scheme": sch.name, "worst_world_dq": worst, "bytes_ratio": rel}));
        // rule: smallest bytes among ADAPTIVE candidates with worst-world dQ >= -0.004
        if sch.adaptive() && worst >= -0.004 && rel < best_bytes {
            best_bytes = rel;
            chosen = Some(i);
        }
    }
    let chosen_scheme: Scheme = match chosen {
        Some(i) => tune_list[i].clone(),
        None => anchors[0].clone(),
    };
    println!(
        "  -> frozen adaptive/prune base scheme: {} (rule: min bytes s.t. tune worst-world dQ >= -0.004; fallback bf16-sr)",
        chosen_scheme.name
    );

    // reference row counts on tune seeds -> row budgets as fractions
    let refsch = Scheme::new("f64", WKind::F64, BKind::F64);
    let mut ref_rows = Vec::new();
    for spec in &worlds {
        let o: Vec<Out> = par_map(&tune, |&s| run_scheme(spec, &refsch, s));
        ref_rows.push(mean(&o.iter().map(|x| x.rows as f64).collect::<Vec<_>>()));
    }
    println!("  mean rows per world (f64 ref, tune seeds): {:?}", ref_rows);

    // ---------------------------------------------------------------------------------------
    // final scheme list (frozen before any held-out data is looked at)
    let mut schemes: Vec<Scheme> = vec![
        Scheme::new("f32", WKind::F32, BKind::F32),
        Scheme::new("f16+u16base", WKind::F16, BKind::U16),
        Scheme::new("fx16-nearest(x3)", WKind::Fx { bits: 16, sr: false }, BKind::U16),
        Scheme::new("fx8-nearest(x3)", WKind::Fx { bits: 8, sr: false }, BKind::U16),
        Scheme::new("fx8-sr", WKind::Fx { bits: 8, sr: true }, BKind::U16),
        Scheme::new("bf16-sr", WKind::Bf { bits: 16, sr: true }, BKind::U16),
        Scheme::new("bf12-sr", WKind::Bf { bits: 12, sr: true }, BKind::U16),
        Scheme::new("bf8-nearest", WKind::Bf { bits: 8, sr: false }, BKind::U16),
        Scheme::new("bf8-sr", WKind::Bf { bits: 8, sr: true }, BKind::U16),
        Scheme::new("bf8-sr+u8base", WKind::Bf { bits: 8, sr: true }, BKind::U8Sr),
        Scheme::new("bf4-sr+u8base", WKind::Bf { bits: 4, sr: true }, BKind::U8Sr),
    ];
    schemes.extend(adaptive.iter().cloned());
    let fracs = [0.75f64, 0.5, 0.35, 0.25, 0.15];

    let held = held_seeds();
    let confirm = confirm_seeds();
    let ext = seeds(91, 250);
    let mut doc_splits = Vec::new();
    for (split, sd) in [
        ("held-out 31-60", &held),
        ("confirm 61-90 (frozen)", &confirm),
        ("power extension 91-250 (160 fresh seeds, frozen)", &ext),
    ] {
        let mut cells = eval_split(&worlds, &schemes, sd, true);
        for (wi, spec) in worlds.iter().enumerate() {
            let ps: Vec<Scheme> = fracs
                .iter()
                .map(|&f| {
                    let mut s = chosen_scheme.clone();
                    s.max_rows = Some(((ref_rows[wi] * f).ceil() as usize).max(4));
                    s.name = format!("{}+prune{:.0}%rows", chosen_scheme.name, f * 100.0);
                    s
                })
                .collect();
            cells.extend(eval_split(std::slice::from_ref(spec), &ps, sd, true));
        }
        let ps: Vec<f64> = cells.iter().map(|c| c.cmp.p_equiv).collect();
        let holm = holm_adjust(&ps);
        print_cells(&format!("F2(a) {split}: scheme vs f64 router (Warm MeanFeature), paired, n={} seeds", sd.len()), &cells, &holm);
        doc_splits.push(json!({"split": split, "n_seeds": sd.len(), "cells": cells_json(&cells, &holm)}));
    }

    let (jb_sim, dense_sim) = json_state_bytes(&WorldSpec::Sim(neural_mesh::sim::Scenario::Random), 31);
    let (jb_big, dense_big) = json_state_bytes(&WorldSpec::Big, 31);
    println!("\ncontext: crate JSON state (weights+baselines) sim-random seed31 = {jb_sim} B, big = {jb_big} B; dense-f64 resident estimate {dense_sim} / {dense_big} B");

    let doc = json!({
        "part": "F2(a) quantised/sparse/adaptive/pruned router state",
        "protocol": {
            "tune": "1-8 (adaptive-precision rule only; row budgets are fractions of the f64 row count on tune seeds)",
            "held_out": "31-60", "confirm": "61-90 frozen",
            "power_extension": "91-250 frozen (added because 30 seeds cannot resolve a 0.003 margin in noisy worlds)",
            "margin": MARGIN,
            "equivalence": "TOST per cell on mean expected success over the run, Holm over all cells of a split; lossless_claim = Holm p<0.05 AND 95% CI inside +-margin",
            "reference": "same Warm(k=1, MeanFeature) with f64 state (MiniHeb f64 verified decision-identical to HebbianRouter)",
            "worlds": worlds.iter().map(|w| w.label()).collect::<Vec<_>>(),
            "big_world": "my construction: 32 agents, 48 types x 4 keywords + 40 noise keywords, Zipf type popularity, 35% trap keyword map, drift on ~25% of types, 4000 tasks",
            "decision_loss_same_history": "teacher-forced: reference drives, twin is asked with the same random stream; mean over tasks of p(twin choice)-p(ref choice); low-variance one-step-ahead bound (no compounding)",
        },
        "tuning": tune_rows, "frozen_prune_base": chosen_scheme.name,
        "mean_ref_rows_tune": ref_rows,
        "crate_json_state_bytes": {"sim_random_seed31": jb_sim, "big_seed31": jb_big},
        "crate_dense_f64_resident_estimate": {"sim_random_seed31": dense_sim, "big_seed31": dense_big},
        "splits": doc_splits,
        "cores": n_threads(),
        "runtime_s": t0.elapsed().as_secs_f64(),
        "notes": [
            "bytes = length of the packed state (to_bytes), names stored (u8 len + bytes); bytes_hashed_names uses 4-byte hashes (not reloadable; collision risk).",
            "in-memory arithmetic is on the quantised grid; the in-memory vectors are not compact, the serialised state is (round-trip tested).",
            "the f64 packed reference uses the same container format with 8-byte cells (dense or sparse per row)."
        ],
    });
    if let Some(p) = out_path {
        std::fs::write(&p, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        println!("wrote {p}");
    }
    println!("runtime {:.1}s", t0.elapsed().as_secs_f64());
}
