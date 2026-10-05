//! G3 / W3 (MEMORY-BYTES): bytes, agents-per-GiB and decision latency of `compact::CompactRouter`
//! against the f64 recommended router `defaults::recommended_router`.
//!
//! DECLARED CRITERIA (before running):
//!   latency   compact must not be slower than the reference by more than +10% (min over 31 interleaved
//!             repetitions of route+feedback ns/decision, fresh router per repetition AND warm pass);
//!             otherwise the regression is reported bluntly with the trade-off.
//!   bytes     exact: live bytes from a counting allocator after a full run (K brains per world),
//!             cross-checked against the library's own `bytes()` accounting and Windows commit charge.
//!   fairness  both routers get the same precomputed outcome table (no env cost in the loop), same
//!             rng seed, same candidate list; strings -> ids work is INCLUDED for the compact router.
//!
//! Usage: cargo run --release --bin g3_perf -- [--out results/g3_perf.json] [--quick] [--quiet-wait 60]

#[path = "g3_common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::time::Instant;

use common::alloc_count;
use common::*;
use neural_mesh::compact::{CompactConfig, CompactRouter, SharedVocab};
use neural_mesh::defaults::recommended_router;
use neural_mesh::router::{Rng, Router};
use serde_json::json;

#[global_allocator]
static GLOBAL: alloc_count::Counting = alloc_count::Counting;

const GIB: f64 = (1u64 << 30) as f64;

struct Table {
    n: usize,
    out: Vec<f64>, // [t * n + a]
}

fn outcome_table(env: &Env) -> Table {
    let n = env.n_agents();
    let mut out = Vec::with_capacity(env.len() * n);
    for t in 0..env.len() {
        for a in 0..n {
            out.push(env.outcome(t, a));
        }
    }
    Table { n, out }
}

/// one closed-loop pass; returns ns per decision (route + feedback)
fn pass<R: Router>(r: &mut R, tasks: &[neural_mesh::router::Task], tab: &Table, cands: &[usize], rng: &mut Rng) -> f64 {
    let t0 = Instant::now();
    let mut sink = 0usize;
    for (t, task) in tasks.iter().enumerate() {
        let a = r.select(task, cands, rng);
        sink = sink.wrapping_add(a);
        r.update(task, a, tab.out[t * tab.n + a]);
    }
    std::hint::black_box(sink);
    t0.elapsed().as_nanos() as f64 / tasks.len() as f64
}

fn select_only<R: Router>(r: &mut R, tasks: &[neural_mesh::router::Task], cands: &[usize], rng: &mut Rng) -> f64 {
    let t0 = Instant::now();
    let mut sink = 0usize;
    for task in tasks {
        sink = sink.wrapping_add(r.select(task, cands, rng));
    }
    std::hint::black_box(sink);
    t0.elapsed().as_nanos() as f64 / tasks.len() as f64
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[derive(Default, Clone)]
struct Lat {
    fresh_min: f64,
    fresh_med: f64,
    warm_min: f64,
    warm_med: f64,
    sel_min: f64,
    sel_med: f64,
}

fn train<R: Router>(env: &Env, r: &mut R, seed: u64) {
    run_policy(env, r, seed);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/g3_perf.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let quiet_wait: u64 = args.iter().position(|a| a == "--quiet-wait").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(60);
    let reps = if quick { 9 } else { 31 };
    let t0 = Instant::now();
    let worlds = all_worlds();
    let mut world_json = Vec::new();
    let mut headline = Vec::new();

    for spec in &worlds {
        let label = spec.label();
        let k_brains = match (&spec, quick) {
            (WorldSpec::Sim(_), false) => 400,
            (WorldSpec::Big, false) => 120,
            (WorldSpec::BigTail, false) => 40,
            (WorldSpec::Sim(_), true) => 40,
            (_, true) => 12,
        };
        println!("== {label}");
        let seed = 31u64;
        let env = spec.make(seed);
        let n = env.n_agents();
        let kw = env.kw_map();
        let tasks = env.tasks_vec();
        let tab = outcome_table(&env);
        let cands: Vec<usize> = (0..n).collect();

        // ---------------- latency (interleaved A/B, min and median over `reps`) ----------------
        let bg = if quiet_wait > 0 { wait_quiet(quiet_wait, 12.0) } else { cpu_busy_pct(300) };
        let (mut rf, mut cf, mut rw, mut cw, mut rs, mut cs) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        for rep in 0..reps + 2 {
            let mut rr = recommended_router(kw.clone(), n);
            let mut cr = CompactRouter::new(n, &kw, CompactConfig::default()).unwrap();
            let (mut g1, mut g2) = (Rng::new(77), Rng::new(77));
            // alternate which runs first
            let (a_f, b_f, a_w, b_w, a_s, b_s);
            if rep % 2 == 0 {
                a_f = pass(&mut rr, &tasks, &tab, &cands, &mut g1);
                b_f = pass(&mut cr, &tasks, &tab, &cands, &mut g2);
                a_w = pass(&mut rr, &tasks, &tab, &cands, &mut g1);
                b_w = pass(&mut cr, &tasks, &tab, &cands, &mut g2);
                a_s = select_only(&mut rr, &tasks, &cands, &mut g1);
                b_s = select_only(&mut cr, &tasks, &cands, &mut g2);
            } else {
                b_f = pass(&mut cr, &tasks, &tab, &cands, &mut g2);
                a_f = pass(&mut rr, &tasks, &tab, &cands, &mut g1);
                b_w = pass(&mut cr, &tasks, &tab, &cands, &mut g2);
                a_w = pass(&mut rr, &tasks, &tab, &cands, &mut g1);
                b_s = select_only(&mut cr, &tasks, &cands, &mut g2);
                a_s = select_only(&mut rr, &tasks, &cands, &mut g1);
            }
            if rep >= 2 {
                rf.push(a_f);
                cf.push(b_f);
                rw.push(a_w);
                cw.push(b_w);
                rs.push(a_s);
                cs.push(b_s);
            }
        }
        let mk = |f: &Vec<f64>, w: &Vec<f64>, s: &Vec<f64>| Lat {
            fresh_min: f.iter().cloned().fold(f64::MAX, f64::min),
            fresh_med: median(&mut f.clone()),
            warm_min: w.iter().cloned().fold(f64::MAX, f64::min),
            warm_med: median(&mut w.clone()),
            sel_min: s.iter().cloned().fold(f64::MAX, f64::min),
            sel_med: median(&mut s.clone()),
        };
        let (lr, lc) = (mk(&rf, &rw, &rs), mk(&cf, &cw, &cs));
        let ratio_fresh = lc.fresh_min / lr.fresh_min;
        let ratio_warm = lc.warm_min / lr.warm_min;
        println!(
            "  ns/decision (route+feedback) fresh min/med: ref {:.0}/{:.0}  compact {:.0}/{:.0}  (x{:.2})   warm: ref {:.0}/{:.0} compact {:.0}/{:.0} (x{:.2})   select-only warm: ref {:.0} compact {:.0}   bg {:.0}%",
            lr.fresh_min, lr.fresh_med, lc.fresh_min, lc.fresh_med, ratio_fresh, lr.warm_min, lr.warm_med, lc.warm_min, lc.warm_med, ratio_warm, lr.sel_min, lc.sel_min, bg
        );

        // ---------------- allocations per decision (warm pass) ----------------
        let mut rr = recommended_router(kw.clone(), n);
        let mut cr = CompactRouter::new(n, &kw, CompactConfig::default()).unwrap();
        let mut g = Rng::new(5);
        pass(&mut rr, &tasks, &tab, &cands, &mut g);
        pass(&mut cr, &tasks, &tab, &cands, &mut g);
        alloc_count::start();
        pass(&mut rr, &tasks, &tab, &cands, &mut g);
        let sr = alloc_count::stop();
        alloc_count::start();
        pass(&mut cr, &tasks, &tab, &cands, &mut g);
        let sc = alloc_count::stop();
        let (ar, ac) = (sr.allocs as f64 / tasks.len() as f64, sc.allocs as f64 / tasks.len() as f64);
        println!("  allocations/decision (warm): ref {ar:.1}  compact {ac:.2}");
        drop((rr, cr));

        // ---------------- bytes: one trained brain, exact via counting allocator ----------------
        alloc_count::start();
        let mut b_ref = recommended_router(kw.clone(), n);
        train(&env, &mut b_ref, seed);
        let m_ref = alloc_count::stop().live;
        alloc_count::start();
        let mut b_c = CompactRouter::new(n, &kw, CompactConfig::default()).unwrap();
        train(&env, &mut b_c, seed);
        let m_c = alloc_count::stop().live;
        let rep_c = b_c.bytes();
        b_c.shrink_to_fit();
        let rep_c2 = b_c.bytes();
        let json_len = b_c.to_json().map(|s| s.len()).unwrap_or(0);
        println!(
            "  one trained brain (live bytes): ref {m_ref}  compact {m_c} (lib: owned {} + vocab~{} + kw {} = {}; reserved {}; after shrink owned {} reserved {}; rows {} cells {})  JSON state {json_len} B",
            rep_c.owned(), rep_c.shared_vocab_bytes, rep_c.shared_kw_bytes, rep_c.private_total(), rep_c.owned_reserved(), rep_c2.owned(), rep_c2.owned_reserved(), rep_c.rows, rep_c.touched_cells
        );
        drop((b_ref, b_c));

        // ---------------- K brains: agents per GiB ----------------
        let seeds_k: Vec<u64> = (0..k_brains as u64).map(|i| 1000 + i).collect();
        let build_k = |variant: usize| -> (f64, usize, usize) {
            // returns (live bytes per brain, commit-charge bytes per brain, aux)
            let (pf0, _) = proc_mem();
            alloc_count::start();
            let mut keep_ref: Vec<neural_mesh::prior::Guarded<neural_mesh::hebbian::HebbianRouter>> = Vec::new();
            let mut keep_c: Vec<CompactRouter> = Vec::new();
            let sv = SharedVocab::new();
            let proto = if variant == 3 { Some(CompactRouter::with_shared_vocab(n, &kw, CompactConfig::default(), &sv).unwrap()) } else { None };
            for &s in &seeds_k {
                let e = spec.make(s);
                match variant {
                    0 => {
                        let mut r = recommended_router(kw.clone(), n);
                        train(&e, &mut r, s);
                        keep_ref.push(r);
                    }
                    1 | 2 => {
                        let mut r = CompactRouter::new(n, &kw, CompactConfig::default()).unwrap();
                        train(&e, &mut r, s);
                        if variant == 2 {
                            r.shrink_to_fit();
                        }
                        keep_c.push(r);
                    }
                    _ => {
                        let mut r = proto.as_ref().unwrap().fork_blank();
                        train(&e, &mut r, s);
                        r.shrink_to_fit();
                        keep_c.push(r);
                    }
                }
            }
            let live = alloc_count::stop().live;
            let (pf1, _) = proc_mem();
            let lib_owned: usize = if variant == 3 { keep_c.iter().map(|r| r.bytes().owned()).sum::<usize>() } else { keep_c.iter().map(|r| r.bytes().private_total()).sum::<usize>() };
            let shared_est = if variant == 3 { sv.bytes_estimate() } else { 0 };
            drop((keep_ref, keep_c, proto));
            (live as f64 / k_brains as f64, (pf1.saturating_sub(pf0)) / k_brains, lib_owned / k_brains + shared_est / k_brains)
        };
        let names = ["ref f64 guarded(hebr) [shipped]", "compact u16/q16, private vocab+kw", "compact u16/q16, private, shrink_to_fit", "compact u16/q16, SHARED vocab+kw, shrink_to_fit"];
        let mut per = Vec::new();
        for v in 0..4 {
            let (live, commit, lib) = build_k(v);
            let apg = GIB / live;
            println!("  K={k_brains} brains, {:46}: {:>9.0} B live/brain, commit {:>9} B/brain, lib-accounted {:>9} B (private_total, or owned+shared/K)  -> {:>9.0} agents/GiB", names[v], live, commit, lib, apg);
            per.push(json!({"variant": names[v], "k": k_brains, "live_bytes_per_brain": live, "commit_bytes_per_brain": commit, "lib_accounted_bytes_per_brain": lib, "agents_per_gib": apg}));
        }
        let gain_private = per[0]["live_bytes_per_brain"].as_f64().unwrap() / per[2]["live_bytes_per_brain"].as_f64().unwrap();
        let gain_shared = per[0]["live_bytes_per_brain"].as_f64().unwrap() / per[3]["live_bytes_per_brain"].as_f64().unwrap();
        println!("  memory reduction vs shipped: x{gain_private:.1} (private compact), x{gain_shared:.1} (shared vocab)");

        headline.push(format!(
            "{label:24} ns fresh ref {:.0} -> compact {:.0} (x{:.2}); warm {:.0} -> {:.0} (x{:.2}); allocs/dec {ar:.1} -> {ac:.2}; bytes/brain {:.0} -> {:.0} private ({gain_private:.1}x) | {:.0} shared ({gain_shared:.1}x); agents/GiB {:.0} -> {:.0} | {:.0}",
            lr.fresh_min, lc.fresh_min, ratio_fresh, lr.warm_min, lc.warm_min, ratio_warm,
            per[0]["live_bytes_per_brain"].as_f64().unwrap(), per[2]["live_bytes_per_brain"].as_f64().unwrap(), per[3]["live_bytes_per_brain"].as_f64().unwrap(),
            per[0]["agents_per_gib"].as_f64().unwrap(), per[2]["agents_per_gib"].as_f64().unwrap(), per[3]["agents_per_gib"].as_f64().unwrap()
        ));
        world_json.push(json!({
            "world": label, "agents": n, "tasks": tasks.len(), "bg_cpu_pct": bg,
            "latency_ns": {
                "reference": {"fresh_min": lr.fresh_min, "fresh_median": lr.fresh_med, "warm_min": lr.warm_min, "warm_median": lr.warm_med, "select_only_min": lr.sel_min, "select_only_median": lr.sel_med},
                "compact":   {"fresh_min": lc.fresh_min, "fresh_median": lc.fresh_med, "warm_min": lc.warm_min, "warm_median": lc.warm_med, "select_only_min": lc.sel_min, "select_only_median": lc.sel_med},
                "ratio_fresh_min": ratio_fresh, "ratio_warm_min": ratio_warm, "reps": reps,
                "within_declared_margin_10pct": ratio_fresh <= 1.10 && ratio_warm <= 1.10,
            },
            "allocs_per_decision_warm": {"reference": ar, "compact": ac},
            "one_brain_live_bytes": {"reference": m_ref, "compact_private_as_built": m_c, "lib_report": rep_c, "lib_report_after_shrink": rep_c2, "compact_json_state_bytes": json_len},
            "agents_per_gib": per,
            "memory_reduction_private": gain_private, "memory_reduction_shared": gain_shared,
        }));
    }

    println!("\n=== HEADLINE (seed 31 world per row; ns = min of {reps} interleaved reps; bytes = counting-allocator live bytes) ===");
    for h in &headline {
        println!("{h}");
    }
    let j = json!({
        "bin": "g3_perf",
        "criteria": {"latency_regression_margin": 1.10, "reps": reps, "quick": quick},
        "worlds": world_json, "runtime_s": t0.elapsed().as_secs_f64(),
        "notes": ["ns is min/median of interleaved repetitions on a shared box (+-30% noise)", "compact ns includes string->id work", "outcome table precomputed: env cost excluded for both"],
    });
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out} ({:.1}s)", t0.elapsed().as_secs_f64());
    let _: HashMap<u8, u8> = HashMap::new();
}
