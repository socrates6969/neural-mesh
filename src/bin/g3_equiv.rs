//! G3 / W3 (MEMORY-BYTES): equivalence of `compact::CompactRouter` (u16 weights, Q16 counts, sparse
//! rows, interned ids) to the f64 recommended router `defaults::recommended_router`
//! (= guarded(hebbian-robust)).
//!
//! PRE-DECLARED PROTOCOL (fixed before any run; nothing is tuned: the compact router uses
//! `CompactConfig::default()` == defaults::recommended_guard + recommended_inner, count scale 1/1024):
//!   metric      full-run mean EXPECTED success (sum of the true p of the chosen agent / tasks)
//!   pairing     by seed (same world, same rng seed for both routers)
//!   margin      +-0.005 absolute (primary), +-0.003 (strict, reported on the power split)
//!   test        TOST at alpha .05 (<=> the 90% CI of the paired difference lies inside +-margin);
//!               95% CI also printed; Holm over the 5 worlds within (subject, split)
//!   splits      smoke 1-8 (nothing was tuned on it), held 31-60, confirm 61-90,
//!               pooled 31-90 (60 seeds, PRIMARY: gives the 32-agent worlds a CI that can certify),
//!               power 91-250 (160 fresh seeds, never used for any choice)
//!   worlds      sim Fixed / Random / NoTraps (6 agents, 2000 tasks), big-32agents (232 features,
//!               4000 tasks), big-32agents-longtail (1838 features)
//!   subjects    compact-u16 (the library CompactRouter, PRIMARY), compact-u16-hash32, compact-f32,
//!               compact-f64 (must be bit-identical: logic check)
//!   verdict     "CERTIFIED" iff Holm p < .05 AND the 90% CI is inside +-margin; otherwise
//!               "inconclusive" (CI crosses the margin) or "NOT equivalent" (CI entirely outside)
//!
//! Usage: cargo run --release --bin g3_equiv -- [--out results/g3_equiv.json] [--quick]
//! env G3_THREADS = worker threads (default min(cores,6)).

#[path = "g3_common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::time::Instant;

use common::*;
use neural_mesh::compact::{CompactConfig, CompactRouter, CompactRouterF32, CompactRouterF64, IdMode};
use neural_mesh::defaults::recommended_router;
use neural_mesh::stats::holm_adjust;
use serde_json::{Value, json};

const SUBJECTS: [&str; 4] = ["compact-u16", "compact-u16-hash32", "compact-f32", "compact-f64"];

fn run_subject(kind: usize, env: &Env, seed: u64) -> RunStats {
    let n = env.n_agents();
    let kw = env.kw_map();
    match kind {
        0 => run_policy(env, &mut CompactRouter::new(n, &kw, CompactConfig::default()).unwrap(), seed),
        1 => run_policy(env, &mut CompactRouter::new(n, &kw, CompactConfig { id_mode: IdMode::Hash32, ..CompactConfig::default() }).unwrap(), seed),
        2 => run_policy(env, &mut CompactRouterF32::new(n, &kw, CompactConfig::default()).unwrap(), seed),
        _ => run_policy(env, &mut CompactRouterF64::new(n, &kw, CompactConfig::default()).unwrap(), seed),
    }
}

/// per seed: [reference, subject0..3] each as (full, h250, h500)
fn run_seed(spec: &WorldSpec, seed: u64) -> Vec<RunStats> {
    let env = spec.make(seed);
    let mut out = Vec::with_capacity(1 + SUBJECTS.len());
    let kw: HashMap<String, usize> = env.kw_map();
    out.push(run_policy(&env, &mut recommended_router(kw, env.n_agents()), seed));
    for k in 0..SUBJECTS.len() {
        out.push(run_subject(k, &env, seed));
    }
    out
}

struct Split {
    name: &'static str,
    seeds: Vec<u64>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/g3_equiv.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let t0 = Instant::now();
    let worlds = all_worlds();
    let (smoke, held, conf, power) = if quick {
        (seeds(1, 4), seeds(31, 38), seeds(61, 68), seeds(91, 100))
    } else {
        (seeds(1, 8), seeds(31, 60), seeds(61, 90), seeds(91, 250))
    };
    let mut pooled = held.clone();
    pooled.extend(conf.iter().copied());
    let splits = vec![
        Split { name: "smoke", seeds: smoke },
        Split { name: "held", seeds: held },
        Split { name: "confirm", seeds: conf },
        Split { name: "pooled", seeds: pooled },
        Split { name: "power", seeds: power },
    ];
    // distinct seeds to run (pooled is the union of held+confirm)
    let mut all_seeds: Vec<u64> = splits.iter().flat_map(|s| s.seeds.iter().copied()).collect();
    all_seeds.sort_unstable();
    all_seeds.dedup();
    println!("g3_equiv: {} worlds x {} seeds x (1 ref + {} subjects), {} workers", worlds.len(), all_seeds.len(), SUBJECTS.len(), n_workers());

    // results[world][seed] -> Vec<RunStats>
    let mut table: Vec<HashMap<u64, Vec<RunStats>>> = Vec::new();
    for w in &worlds {
        let rs = par_map(&all_seeds, |&s| run_seed(w, s));
        println!("  {} done ({:.1}s)", w.label(), t0.elapsed().as_secs_f64());
        table.push(all_seeds.iter().copied().zip(rs).collect());
    }

    let pick = |wi: usize, seeds: &[u64], who: usize| -> Vec<f64> { seeds.iter().map(|s| table[wi][s][who].full).collect() };
    let mut results = Vec::new();
    let mut summary_lines = Vec::new();
    for (si, sp) in splits.iter().enumerate() {
        let _ = si;
        for (k, name) in SUBJECTS.iter().enumerate() {
            for &margin in &[0.005f64, 0.003] {
                // Holm over the worlds of this (split, subject, margin)
                let eqs: Vec<EqRes> = (0..worlds.len()).map(|wi| eq_test(&pick(wi, &sp.seeds, 1 + k), &pick(wi, &sp.seeds, 0), margin)).collect();
                let holm = holm_adjust(&eqs.iter().map(|e| e.p_tost).collect::<Vec<_>>());
                for (wi, e) in eqs.iter().enumerate() {
                    let verdict = if holm[wi] < 0.05 && e.ci90_inside() {
                        "CERTIFIED"
                    } else if e.lo90 > e.margin || e.hi90 < -e.margin {
                        "NOT equivalent"
                    } else {
                        "inconclusive"
                    };
                    let refm = pick(wi, &sp.seeds, 0).iter().sum::<f64>() / sp.seeds.len() as f64;
                    let subm = pick(wi, &sp.seeds, 1 + k).iter().sum::<f64>() / sp.seeds.len() as f64;
                    let mut j = e.json(holm[wi]);
                    j["split"] = json!(sp.name);
                    j["subject"] = json!(name);
                    j["world"] = json!(worlds[wi].label());
                    j["verdict"] = json!(verdict);
                    j["mean_reference"] = json!(refm);
                    j["mean_subject"] = json!(subm);
                    results.push(j);
                    if (margin - 0.005).abs() < 1e-12 || sp.name == "power" {
                        summary_lines.push(format!(
                            "{:8} {:19} +-{:.3} {:24} n={:3} ref {:.4} subj {:.4}  {}  holm={:.4}  -> {}",
                            sp.name, name, margin, worlds[wi].label(), e.n, refm, subm, e.fmt(), holm[wi], verdict
                        ));
                    }
                }
            }
        }
    }
    for l in &summary_lines {
        println!("{l}");
    }

    // headline: primary subject, pooled 60 seeds, +-0.005
    println!("\n=== PRIMARY: compact-u16 vs f64 recommended_router, pooled seeds 31-90, margin +-0.005 ===");
    for r in results.iter().filter(|r| r["subject"] == "compact-u16" && r["split"] == "pooled" && r["margin"] == 0.005) {
        println!(
            "{:24} diff {:+.4}  90%CI [{:+.4},{:+.4}]  95%CI [{:+.4},{:+.4}]  holm p={:.4}  {}",
            r["world"].as_str().unwrap(),
            r["diff_mean"].as_f64().unwrap(),
            r["ci90"][0].as_f64().unwrap(),
            r["ci90"][1].as_f64().unwrap(),
            r["ci95"][0].as_f64().unwrap(),
            r["ci95"][1].as_f64().unwrap(),
            r["holm_p_tost"].as_f64().unwrap(),
            r["verdict"].as_str().unwrap()
        );
    }
    // logic check: f64 compact must be bit-identical
    let max_abs = results.iter().filter(|r| r["subject"] == "compact-f64").map(|r| r["diff_mean"].as_f64().unwrap().abs().max(r["diff_sd"].as_f64().unwrap())).fold(0.0, f64::max);
    println!("\ncompact-f64 vs reference: max |mean diff| / sd over every cell = {max_abs:e} (must be 0)");

    let j = json!({
        "bin": "g3_equiv",
        "protocol": {
            "metric": "full-run mean expected success", "pairing": "by seed", "margins": [0.005, 0.003],
            "test": "TOST alpha .05 (90% CI inside margin), Holm over the 5 worlds per (subject, split, margin)",
            "splits": splits.iter().map(|s| json!({"name": s.name, "n": s.seeds.len(), "first": s.seeds[0], "last": s.seeds[s.seeds.len()-1]})).collect::<Vec<_>>(),
            "primary": "compact-u16, split pooled (31-90), margin 0.005",
            "tuning": "none; CompactConfig::default() == defaults::recommended_*; seeds 1-8 are a smoke run",
            "quick": quick,
        },
        "worlds": worlds.iter().map(|w| w.label()).collect::<Vec<_>>(),
        "subjects": SUBJECTS,
        "results": results,
        "compact_f64_max_abs_diff": max_abs,
        "runtime_s": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out} ({:.1}s)", t0.elapsed().as_secs_f64());
    let _: Value = j;
}
