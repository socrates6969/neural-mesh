//! H14 (R6-B): parity check of the LIBRARY `neural_mesh::popguard::PopGuarded` / `fit_prior` against the H9 bin implementation
//! (`h9_common`), and reproduction of the h9 `pop-guard+conf[M16,m1=1,cov0.5,inf=8][w0=0.8,cg=0.5]` numbers on held-out
//! seeds 61-90 (results/h9_priors_confirm61-90.json). A faithful port: NO re-tuning, the frozen H9 pick is hard-coded.
//!
//!   h14_popguard [--seeds 61-90] [--worlds A1,B1,B6,...|all] [--k 6] [--out results/h14_popguard.json]
//!
//! Checks per (world, seed): (1) library prior == h9 prior cell by cell (bit-equal), (2) library PopGuarded success curve ==
//! h9 PopGuarded curve (bit-equal f32), (3) the 12 metrics vs the stored per-seed values in the h9 confirm json (tolerance 1e-12:
//! serde_json's default float parser is not bit-exact), (4) empty-prior library PopGuarded == defaults::recommended_router.
//! SYNTHETIC-until-validated-on-real-traffic.
#![allow(dead_code)]

#[path = "f3_common/mod.rs"]
mod f3;
#[path = "h9_common/mod.rs"]
mod h9;

use std::collections::HashMap;

use f3::par_map;
use h9::*;
use neural_mesh::popguard as lib;
use serde_json::{json, Value};

const PICK_LABEL: &str = "pop-guard+conf[M16,m1=1,cov0.5,inf=8][w0=0.8,cg=0.5]";

fn frozen_fit() -> FitCfg {
    FitCfg { m_max: 16.0, m1: 1.0, n_min: 5.0, shrink: true, cov_min: 0.5 }
}
fn frozen_trust() -> TrustCfg {
    TrustCfg { w0: 0.8, cf: 1.0, cg: 0.5, delta: 0.97 }
}

fn to_pool(d: &Dep) -> lib::PoolLog {
    lib::PoolLog { cells: d.cells.clone() }
}

fn prior_equal(a: &PopPrior, b: &lib::PopPrior) -> bool {
    if a.cells.len() != b.cells.len() {
        return false;
    }
    a.cells.iter().all(|(f, row)| {
        b.cells.get(f).is_some_and(|r2| {
            row.len() == r2.len() && row.iter().zip(r2).all(|(x, y)| x.s.to_bits() == y.s.to_bits() && x.n.to_bits() == y.n.to_bits() && x.mu.to_bits() == y.mu.to_bits() && x.m.to_bits() == y.m.to_bits())
        })
    })
}

struct One {
    world: usize,
    seed: u64,
    prior_eq: bool,
    curve_eq: bool,
    empty_eq: bool,
    metrics: [f64; NM],
    h9_metrics: [f64; NM],
    n_informative: usize,
}

fn run_one(spec: &Spec, wi: usize, seed: u64, k: usize) -> One {
    let w = make_world(&spec.eval, seed);
    let deps = build_pool(&spec.pool, k, seed);
    let pool_n = deps.first().and_then(|d| d.cells.values().next().map(|v| v.len())).unwrap_or(w.n_agents);
    let n = w.n_agents;
    // h9 reference
    let prior9 = fit_prior(&deps, pool_n, frozen_fit());
    let mut r9 = PopGuarded::new(w.kw.clone(), n, prior9.clone(), Some(frozen_trust())).with_informed(8.0);
    let c9 = run_curve(&w, &mut r9, seed);
    // library
    let logs: Vec<lib::PoolLog> = deps.iter().map(to_pool).collect();
    let cfg = lib::FitCfg { m_max: 16.0, m1: 1.0, n_min: 5.0, shrink: true, cov_min: 0.5 };
    assert_eq!(cfg, lib::FitCfg::default());
    let priorl = lib::fit_prior(&logs, pool_n, cfg);
    let mut rl = lib::recommended_warm_router(w.kw.clone(), n, &logs);
    let cl = run_curve(&w, &mut rl, seed);
    // empty prior == recommended_router (library type)
    let mut base = neural_mesh::defaults::recommended_router(w.kw.clone(), n);
    let cb = run_curve(&w, &mut base, seed);
    let mut le = lib::PopGuarded::new(w.kw.clone(), n, lib::PopPrior::empty(), Some(lib::TrustCfg::default())).with_informed(8.0);
    let ce = run_curve(&w, &mut le, seed);
    One {
        world: wi,
        seed,
        prior_eq: prior_equal(&prior9, &priorl),
        curve_eq: c9 == cl,
        empty_eq: cb == ce,
        metrics: metrics(&cl, &w.oracle),
        h9_metrics: metrics(&c9, &w.oracle),
        n_informative: priorl.n_informative(),
    }
}

fn arg(a: &[String], key: &str, d: &str) -> String {
    a.iter().position(|x| x == key).and_then(|i| a.get(i + 1)).cloned().unwrap_or_else(|| d.to_string())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (lo, hi) = {
        let s = arg(&a, "--seeds", "61-90");
        let (x, y) = s.split_once('-').unwrap();
        (x.parse::<u64>().unwrap(), y.parse::<u64>().unwrap())
    };
    let k: usize = arg(&a, "--k", "6").parse().unwrap();
    let out = arg(&a, "--out", "results/h14_popguard.json");
    let confirm = arg(&a, "--ref", "results/h9_priors_confirm61-90.json");
    let wsel = arg(&a, "--worlds", "all");
    let specs = specs();
    let widx: Vec<usize> = if wsel == "all" {
        (0..specs.len()).collect()
    } else {
        let keys: Vec<&str> = wsel.split(',').collect();
        (0..specs.len()).filter(|&i| keys.iter().any(|k| specs[i].name.starts_with(k))).collect()
    };
    let seeds: Vec<u64> = (lo..=hi).collect();
    let jobs: Vec<(usize, u64)> = widx.iter().flat_map(|&w| seeds.iter().map(move |&s| (w, s))).collect();
    eprintln!("[h14_popguard] {} worlds x {} seeds, K={k}, policy {PICK_LABEL}", widx.len(), seeds.len());
    let res: Vec<One> = par_map(&jobs, |&(wi, s)| run_one(&specs[wi], wi, s, k));

    // reference numbers from the h9 json (only when the seed range is the stored one)
    let refj: Option<Value> = std::fs::read_to_string(&confirm).ok().and_then(|t| serde_json::from_str(&t).ok());
    let ref_ok = refj.as_ref().is_some_and(|r| r["held"] == json!([lo, hi]) && r["k_pool"] == json!(k));
    let mut world_json = vec![];
    let (mut all_prior, mut all_curve, mut all_empty, mut max_dev, mut ref_cells) = (true, true, true, 0.0f64, 0usize);
    println!("h14 popguard parity: library PopGuarded vs h9_common PopGuarded vs results/h9_priors_confirm61-90.json ({PICK_LABEL}, K={k}, seeds {lo}-{hi})");
    println!("SYNTHETIC-until-validated-on-real-traffic. library-vs-h9 equality is BIT-exact (f32 curves, f64 priors); json comparison uses tol 1e-12.");
    println!("{:<22} {:>5} | {:>7} {:>7} {:>7} {:>7} {:>8} | prior==h9 curve==h9 empty==recommended | max|d| vs json", "world", "seeds", "s50", "s100", "s250", "s500", "s2000");
    for &wi in &widx {
        let rows: Vec<&One> = res.iter().filter(|r| r.world == wi).collect();
        let mean = |m: usize| rows.iter().map(|r| r.metrics[m]).sum::<f64>() / rows.len() as f64;
        let pe = rows.iter().all(|r| r.prior_eq);
        let ce = rows.iter().all(|r| r.curve_eq && r.metrics.iter().zip(&r.h9_metrics).all(|(a, b)| a.to_bits() == b.to_bits()));
        let ee = rows.iter().all(|r| r.empty_eq);
        all_prior &= pe;
        all_curve &= ce;
        all_empty &= ee;
        // vs stored json (per-seed arrays)
        let mut dev = f64::NAN;
        if ref_ok {
            let r = refj.as_ref().unwrap();
            if let Some(wj) = r["worlds"].as_array().and_then(|ws| ws.iter().find(|w| w["name"] == json!(specs[wi].name))) {
                if let Some(pj) = wj["policies"].as_array().and_then(|ps| ps.iter().find(|p| p["name"] == json!(PICK_LABEL))) {
                    dev = 0.0;
                    for (m, nm) in MNAMES.iter().enumerate() {
                        let stored: Vec<f64> = pj["per_seed"][nm].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
                        assert_eq!(stored.len(), rows.len());
                        for (sv, r) in stored.iter().zip(&rows) {
                            // rows are in seed order because par_map preserves job order
                            dev = dev.max((sv - r.metrics[m]).abs());
                            ref_cells += 1;
                        }
                    }
                    max_dev = max_dev.max(dev);
                }
            }
        }
        println!("{:<22} {:>5} | {:7.4} {:7.4} {:7.4} {:7.4} {:8.4} | {:<9} {:<9} {:<18} | {}", specs[wi].name, rows.len(), mean(0), mean(1), mean(2), mean(3), mean(4), pe, ce, ee, if dev.is_nan() { "n/a".into() } else { format!("{dev:.2e}") });
        world_json.push(json!({"world": specs[wi].name, "role": format!("{:?}", specs[wi].role), "n_seeds": rows.len(),
            "means": MNAMES.iter().enumerate().map(|(m, n)| (n.to_string(), json!(mean(m)))).collect::<HashMap<_, _>>(),
            "prior_bit_equal_h9": pe, "curve_and_metrics_bit_equal_h9": ce, "empty_prior_equals_recommended_router": ee,
            "max_abs_dev_vs_h9_confirm_json": if dev.is_nan() { Value::Null } else { json!(dev) },
            "mean_informative_prior_cells": rows.iter().map(|r| r.n_informative as f64).sum::<f64>() / rows.len() as f64,
            "per_seed": MNAMES.iter().enumerate().map(|(m, n)| (n.to_string(), json!(rows.iter().map(|r| r.metrics[m]).collect::<Vec<_>>()))).collect::<HashMap<_, _>>()}));
    }
    println!("\nSUMMARY: prior bit-equal in all cells: {all_prior}; library curve+metrics bit-equal to h9_common PopGuarded: {all_curve}; empty prior == recommended_router: {all_empty}");
    if ref_ok {
        println!("         vs results/h9_priors_confirm61-90.json: {ref_cells} (world,seed,metric) values compared, max |dev| = {max_dev:.3e}");
    } else {
        println!("         (h9 confirm json not comparable: seeds/K differ or file missing)");
    }
    let ok = all_prior && all_curve && all_empty && (!ref_ok || max_dev < 1e-12);
    println!("PARITY: {}", if ok { "PASS" } else { "FAIL" });
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"note": "SYNTHETIC-until-validated-on-real-traffic. Library port parity of pop-guard+conf (R6-B/H14).", "policy": PICK_LABEL, "seeds": [lo, hi], "k_pool": k,
        "all_prior_bit_equal": all_prior, "all_curve_bit_equal": all_curve, "all_empty_equal_recommended": all_empty, "ref_values_compared": ref_cells, "max_dev_vs_h9_json": max_dev, "pass": ok, "worlds": world_json})).unwrap()).unwrap();
    println!("wrote {out}");
    if !ok {
        std::process::exit(1);
    }
}
