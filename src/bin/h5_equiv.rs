//! H5 / R4-B2 (MEMORY-REDUCTION): equivalence (TOST) and bytes of the CompactRouter levers
//!   (a) bounded state: per-brain row cap with VALUE-based eviction, bounded vocabulary (value-based,
//!       keywords pinned), Hash32 (no vocabulary at all);
//!   (b) smaller cell encodings: u12 / u8 weights, 12-bit (1/64) guard counts (packed 12-bit storage);
//! against the f64 recommended router `defaults::recommended_router` (= guarded(hebbian-robust)).
//!
//! PRE-DECLARED PROTOCOL (fixed before any run; no lever was tuned on the evaluation seeds):
//!   metric      full-run mean EXPECTED success (sum of the true p of the chosen agent / tasks)
//!   pairing     by seed (same world, same rng seed for subject and reference)
//!   margin      +-0.005 absolute (primary), +-0.003 (strict, reported)
//!   test        TOST alpha .05 (<=> 90% CI of the paired diff inside +-margin); 95% CI also tested;
//!               Holm over the 8 worlds within (subject, split, margin)
//!   splits      POOLED seeds 31-90 (n=60, as W3) and FRESH seeds 91-250 (n=160); seeds 1-8 are used
//!               ONLY to measure each world's natural row/vocab size (cap = fraction of it), nothing else
//!   worlds      sim Fixed/Random/NoTraps (6 agents, 2000 tasks), big-32agents, big-32agents-longtail,
//!               big-32agents-open (open token stream: ~every noise token is new), big-64agents,
//!               big-128agents (4000 tasks each). All synthetic (mine), not real traffic.
//!   caps        row cap = ceil(f * natural rows); vocab cap = pinned + ceil(f * (natural vocab - pinned)); f in {.75,.5,.25,.125}; ablation:
//!               row cap with pseudo-random (hash) eviction at f = .5,.25
//!   encodings   weights u16/u12/u8 x counts q16 (1/1024) / q12 (1/64, 12 bit)
//!   PASS        STAGE 1: a subject PASSES iff in EVERY world, on BOTH splits, margin +-0.005: Holm p < .05 AND the
//!               95% CI is inside the margin ("CERTIFIED95"). "CERTIFIED90" = TOST-only (W3's rule).
//!   keep rule   the lever with the smallest bytes among PASSing subjects is kept; failures are reported.
//!   STAGE 2     (declared AFTER stage 1, because the u16/q16 baseline itself could not be certified in the 64-agent world with 160 seeds: the
//!               paired sd grows with the agent count) the same tests on FRESH seeds 91-450 (n=360, 200 more never-used seeds);
//!               plus a one-sided non-inferiority test (Holm, diff > -0.005) as a secondary view, because a two-sided TOST cannot
//!               certify a subject that is BETTER than the reference by more than the margin.
//!   combos      the default subject list ends with combinations (rows+vocab, hash32+rowcap) that were added after
//!               seeing stage-1 numbers of the single levers (disclosed); `--combo` adds more subjects from the command line.
//!
//!   STAGE 3     (declared after stage 2) cell pruning (`prune_mass` 0.3/0.6/1.0/1.5, `prune_dev` 0.005) on the same seeds:
//!               `--only-combo --combo "u16/q16,1,none;u16/q16,0.3,prune;..."` -> results/h5_equiv_stage3.{json,log}.
//!   STAGE 4     (declared after stage 3) vocabulary caps below f = .125 (.0625, .03125, 0) to find the break point:
//!               `--only-combo --combo "u16/q16,1,none;u16/q16,0.0625,vocab;..."` -> results/h5_equiv_stage4.{json,log}.
//!
//! Usage: cargo run --release --bin h5_equiv -- [--out results/h5_equiv.json] [--quick] [--combo enc,f,kind;...]
//! env H5_THREADS = worker threads (default min(cores,6)).

#[path = "h5_common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::time::Instant;

use common::*;
use neural_mesh::compact::{Cell, CompactConfig, CompactRouterG, EvictPolicy, Fx8, Fx12, Fx16, IdMode, Q12, Q16};
use neural_mesh::defaults::recommended_router;
use neural_mesh::router::Router;
use neural_mesh::stats::holm_adjust;
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Lever {
    /// uncapped
    None,
    /// row cap = ceil(f * natural rows)
    Rows(f64, EvictPolicy),
    /// vocab cap = pinned keywords + ceil(f * (natural vocab - pinned))
    Vocab(f64),
    Hash32,
    /// stage-2 combos (declared after stage 1): row cap f (value) + vocab cap f
    RowsVocab(f64),
    /// Hash32 ids + row cap f (value)
    HashRows(f64),
    /// stage 3 (declared after stage 2): cell pruning with `prune_mass` = f (prune_dev 0.005)
    Prune(f64),
}

#[derive(Clone, Debug)]
struct Subject {
    name: String,
    /// (weight bits, count bits); (0,0) = f64/f64 cells
    enc: (u8, u8),
    lever: Lever,
}

fn subj(name: &str, enc: (u8, u8), lever: Lever) -> Subject {
    Subject { name: name.into(), enc, lever }
}

fn stage1_subjects() -> Vec<Subject> {
    let mut v = vec![subj("compact-f64", (0, 0), Lever::None), subj("u16/q16", (16, 16), Lever::None)];
    for (w, c) in [(12u8, 16u8), (8, 16), (16, 12), (12, 12), (8, 12)] {
        v.push(subj(&format!("u{w}/q{c}"), (w, c), Lever::None));
    }
    for f in [0.75, 0.5, 0.25, 0.125] {
        v.push(subj(&format!("rowcap{f}"), (16, 16), Lever::Rows(f, EvictPolicy::Value)));
    }
    for f in [0.5, 0.25] {
        v.push(subj(&format!("rowcap{f}-hash"), (16, 16), Lever::Rows(f, EvictPolicy::Hash)));
    }
    for f in [0.75, 0.5, 0.25, 0.125] {
        v.push(subj(&format!("vocabcap{f}"), (16, 16), Lever::Vocab(f)));
    }
    v.push(subj("hash32", (16, 16), Lever::Hash32));
    // combos declared after stage 1 (stage-1 results in results/h5_equiv_stage1.log)
    for f in [0.5, 0.25] {
        v.push(subj(&format!("rows+vocab{f}"), (16, 16), Lever::RowsVocab(f)));
        v.push(subj(&format!("hash32+rowcap{f}"), (16, 16), Lever::HashRows(f)));
    }
    v
}

/// per world: natural sizes measured on seeds 1-8 (uncapped u16/q16)
#[derive(Clone, Copy, Debug, Default)]
struct Nat {
    rows: f64,
    vocab: f64,
    pinned: f64,
}

#[derive(Clone, Debug, Default)]
struct Out {
    stats: RunStats,
    owned: usize,
    vocab_est: usize,
    rows: usize,
    cells: usize,
    row_evictions: u64,
    vocab_len: usize,
}

fn run_cell<W: Cell, C: Cell>(env: &Env, cfg: CompactConfig, seed: u64) -> Out {
    let n = env.n_agents();
    let kw = env.kw_map();
    let mut r = CompactRouterG::<W, C>::new(n, &kw, cfg).unwrap();
    let stats = run_policy(env, &mut r, seed);
    r.shrink_to_fit();
    let b = r.bytes();
    Out { stats, owned: b.owned(), vocab_est: b.shared_vocab_bytes, rows: b.rows, cells: b.touched_cells, row_evictions: r.row_evictions() + r.cells_pruned(), vocab_len: r.private_vocab_len() }
}

fn run_subject(s: &Subject, env: &Env, nat: &Nat, seed: u64) -> Out {
    let mut cfg = CompactConfig::default();
    match s.lever {
        Lever::None => {}
        Lever::Rows(f, pol) => {
            cfg.max_rows = ((f * nat.rows).ceil() as usize).max(1);
            cfg.evict = pol;
        }
        Lever::Vocab(f) => cfg.vocab_cap = ((nat.pinned + f * (nat.vocab - nat.pinned).max(0.0)).ceil() as usize).max(1),
        Lever::Hash32 => cfg.id_mode = IdMode::Hash32,
        Lever::RowsVocab(f) => {
            cfg.max_rows = ((f * nat.rows).ceil() as usize).max(1);
            cfg.vocab_cap = ((nat.pinned + f * (nat.vocab - nat.pinned).max(0.0)).ceil() as usize).max(1);
        }
        Lever::Prune(f) => cfg.prune_mass = f,
        Lever::HashRows(f) => {
            cfg.id_mode = IdMode::Hash32;
            cfg.max_rows = ((f * nat.rows).ceil() as usize).max(1);
        }
    }
    match s.enc {
        (0, 0) => run_cell::<f64, f64>(env, cfg, seed),
        (16, 16) => run_cell::<Fx16, Q16>(env, cfg, seed),
        (12, 16) => run_cell::<Fx12, Q16>(env, cfg, seed),
        (8, 16) => run_cell::<Fx8, Q16>(env, cfg, seed),
        (16, 12) => run_cell::<Fx16, Q12>(env, cfg, seed),
        (12, 12) => run_cell::<Fx12, Q12>(env, cfg, seed),
        (8, 12) => run_cell::<Fx8, Q12>(env, cfg, seed),
        e => panic!("encoding {e:?} not wired"),
    }
}

struct Split {
    name: &'static str,
    seeds: Vec<u64>,
}

fn parse_combo(s: &str) -> Vec<Subject> {
    // "u12/q12,0.5,rows;u16/q16,0.25,vocab" -> subjects
    let mut v = Vec::new();
    for part in s.split(';').filter(|p| !p.is_empty()) {
        let f: Vec<&str> = part.split(',').collect();
        let enc = match f[0] {
            "f64" => (0, 0),
            "u16/q16" => (16, 16),
            "u12/q16" => (12, 16),
            "u8/q16" => (8, 16),
            "u16/q12" => (16, 12),
            "u12/q12" => (12, 12),
            "u8/q12" => (8, 12),
            e => panic!("bad enc {e}"),
        };
        let frac: f64 = f.get(1).and_then(|x| x.parse().ok()).unwrap_or(1.0);
        let lever = match f.get(2).copied().unwrap_or("none") {
            "rows" => Lever::Rows(frac, EvictPolicy::Value),
            "vocab" => Lever::Vocab(frac),
            "hash32" => Lever::Hash32,
            "prune" => Lever::Prune(frac),
            _ => Lever::None,
        };
        v.push(subj(&format!("combo[{part}]"), enc, lever));
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/h5_equiv.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let combo = args.iter().position(|a| a == "--combo").and_then(|i| args.get(i + 1)).cloned();
    let only_combo = args.iter().any(|a| a == "--only-combo");
    let t0 = Instant::now();
    let worlds = all_worlds();
    let (nat_seeds, pooled, power, topup) = if quick { (seeds(1, 3), seeds(31, 38), seeds(91, 100), seeds(101, 108)) } else { (seeds(1, 8), seeds(31, 90), seeds(91, 250), seeds(251, 450)) };
    let mut fresh: Vec<u64> = power.clone();
    fresh.extend(topup.iter().copied());
    let splits = vec![Split { name: "pooled", seeds: pooled.clone() }, Split { name: "power", seeds: power.clone() }, Split { name: "topup", seeds: topup.clone() }, Split { name: "fresh", seeds: fresh.clone() }];
    let mut subjects = if only_combo { Vec::new() } else { stage1_subjects() };
    if let Some(c) = &combo {
        subjects.extend(parse_combo(c));
    }
    let mut all_seeds: Vec<u64> = splits.iter().flat_map(|s| s.seeds.iter().copied()).collect();
    all_seeds.sort_unstable();
    all_seeds.dedup();
    println!("h5_equiv: {} worlds x {} seeds x (1 ref + {} subjects), {} workers", worlds.len(), all_seeds.len(), subjects.len(), n_workers());

    // natural sizes (seeds 1-8 only)
    let mut nats = Vec::new();
    for w in &worlds {
        let outs = par_map(&nat_seeds, |&s| {
            let env = w.make(s);
            run_cell::<Fx16, Q16>(&env, CompactConfig::default(), s)
        });
        let kwn = w.make(1).kw_map().len() as f64;
        let nat = Nat { rows: outs.iter().map(|o| o.rows as f64).sum::<f64>() / outs.len() as f64, vocab: outs.iter().map(|o| o.vocab_len as f64).sum::<f64>() / outs.len() as f64, pinned: kwn };
        println!("  natural size {:24} rows {:7.1}  vocab {:7.1}  (keyword ids pinned: {})", w.label(), nat.rows, nat.vocab, nat.pinned);
        nats.push(nat);
    }

    // table[world][seed] = (reference stats, per-subject Out)
    let mut table: Vec<HashMap<u64, (RunStats, Vec<Out>)>> = Vec::new();
    for (wi, w) in worlds.iter().enumerate() {
        let nat = nats[wi];
        let rs = par_map(&all_seeds, |&s| {
            let env = w.make(s);
            let r = run_policy(&env, &mut recommended_router(env.kw_map(), env.n_agents()), s);
            let o: Vec<Out> = subjects.iter().map(|sb| run_subject(sb, &env, &nat, s)).collect();
            (r, o)
        });
        println!("  {} done ({:.1}s)", w.label(), t0.elapsed().as_secs_f64());
        table.push(all_seeds.iter().copied().zip(rs).collect());
    }

    let mut results = Vec::new();
    let mut verdicts: HashMap<(usize, &'static str, u32), Vec<(bool, bool, bool)>> = HashMap::new(); // (subject, split, margin*1e4) -> per world (cert90, cert95)
    for sp in &splits {
        for (k, sb) in subjects.iter().enumerate() {
            for &margin in &[0.005f64, 0.003] {
                let eqs: Vec<EqRes> = (0..worlds.len())
                    .map(|wi| {
                        let a: Vec<f64> = sp.seeds.iter().map(|s| table[wi][s].1[k].stats.full).collect();
                        let b: Vec<f64> = sp.seeds.iter().map(|s| table[wi][s].0.full).collect();
                        eq_test(&a, &b, margin)
                    })
                    .collect();
                let holm = holm_adjust(&eqs.iter().map(|e| e.p_tost).collect::<Vec<_>>());
                let holm_ni = holm_adjust(&eqs.iter().map(|e| e.p_ni).collect::<Vec<_>>());
                for (wi, e) in eqs.iter().enumerate() {
                    let c90 = holm[wi] < 0.05 && e.ci90_inside();
                    let c95 = holm[wi] < 0.05 && e.ci95_inside();
                    let ni = holm_ni[wi] < 0.05;
                    let verdict = if c95 {
                        "CERTIFIED95"
                    } else if c90 {
                        "CERTIFIED90"
                    } else if e.lo90 > e.margin || e.hi90 < -e.margin {
                        "NOT equivalent"
                    } else {
                        "inconclusive"
                    };
                    let refm = sp.seeds.iter().map(|s| table[wi][s].0.full).sum::<f64>() / sp.seeds.len() as f64;
                    let subm = sp.seeds.iter().map(|s| table[wi][s].1[k].stats.full).sum::<f64>() / sp.seeds.len() as f64;
                    verdicts.entry((k, sp.name, (margin * 1e4).round() as u32)).or_default().push((c90, c95, ni));
                    let mut j = e.json(holm[wi]);
                    j["split"] = json!(sp.name);
                    j["subject"] = json!(sb.name);
                    j["world"] = json!(worlds[wi].label());
                    j["verdict"] = json!(verdict);
                    j["noninferior_holm"] = json!(ni);
                    j["holm_p_ni"] = json!(holm_ni[wi]);
                    j["mean_reference"] = json!(refm);
                    j["mean_subject"] = json!(subm);
                    results.push(j);
                }
            }
        }
    }

    // ---- bytes (mean over the pooled seeds) ----
    let mut bytes_json = Vec::new();
    println!("\n=== MEAN BYTES PER TRAINED BRAIN (pooled seeds 31-90; owned = ids+rows+struct after shrink_to_fit; vocab = private vocab estimate) ===");
    for (wi, w) in worlds.iter().enumerate() {
        println!("-- {}", w.label());
        let base: f64 = pooled.iter().map(|s| table[wi][s].1.iter().zip(subjects.iter()).find(|(_, sb)| sb.name == "u16/q16" || sb.name.starts_with("combo[u16/q16,1,none")).map(|(o, _)| o.owned as f64).unwrap_or(0.0)).sum::<f64>() / pooled.len() as f64;
        for (k, sb) in subjects.iter().enumerate() {
            let m = |f: &dyn Fn(&Out) -> f64| pooled.iter().map(|s| f(&table[wi][s].1[k])).sum::<f64>() / pooled.len() as f64;
            let (owned, vocab, rows, cells, ev) = (m(&|o| o.owned as f64), m(&|o| o.vocab_est as f64), m(&|o| o.rows as f64), m(&|o| o.cells as f64), m(&|o| o.row_evictions as f64));
            let vlen = m(&|o| o.vocab_len as f64);
            let ok95 = verdicts.get(&(k, "pooled", 50)).map(|v| v[wi].1).unwrap_or(false) && verdicts.get(&(k, "power", 50)).map(|v| v[wi].1).unwrap_or(false);
            println!("   {:22} owned {:8.0} B ({:5.1}% of u16/q16)  vocab~{:8.0} B  rows {:7.1}  cells {:8.1}  row-evictions {:7.1}  vocab ids {:7.1}   cert95 both splits: {}", sb.name, owned, 100.0 * owned / base.max(1.0), vocab, rows, cells, ev, vlen, ok95);
            bytes_json.push(json!({"world": w.label(), "subject": sb.name, "owned_bytes": owned, "owned_pct_of_u16q16": 100.0 * owned / base.max(1.0), "vocab_bytes_est": vocab, "rows": rows, "cells": cells, "row_evictions": ev, "private_vocab_ids": vlen, "cert95_both_splits_0005": ok95}));
        }
    }

    // ---- diff tables ----
    for spn in ["pooled", "power", "fresh"] {
        println!("\n=== PAIRED DIFF (subject - f64 reference) of mean expected success, split {spn}, margin +-0.005: mean [95% CI] verdict ===");
        for sb in &subjects {
            let mut line = format!("{:20}", sb.name);
            for w in &worlds {
                if let Some(r) = results.iter().find(|r| r["subject"] == sb.name.as_str() && r["split"] == spn && r["margin"] == 0.005 && r["world"] == w.label().as_str()) {
                    let v = r["verdict"].as_str().unwrap();
                    let tag = match v { "CERTIFIED95" => "C95", "CERTIFIED90" => "C90", "NOT equivalent" => "NOT", _ => "inc" };
                    line += &format!(" | {:+.4}[{:+.4},{:+.4}]{}", r["diff_mean"].as_f64().unwrap(), r["ci95"][0].as_f64().unwrap(), r["ci95"][1].as_f64().unwrap(), tag);
                }
            }
            println!("{line}");
        }
        println!("{:20} | worlds in order: {}", "", worlds.iter().map(|w| w.label()).collect::<Vec<_>>().join(" | "));
    }

    // ---- headline: PASS tables ----
    let mut pass_json = Vec::new();
    for (stage, spl) in [("STAGE 1 (declared): pooled 31-90 AND power 91-250", vec!["pooled", "power"]), ("STAGE 2 (declared after stage 1, more fresh power): FRESH seeds 91-450 (n=360)", vec!["fresh"])] {
        println!("
=== PASS TABLE {stage}: all {} worlds, +-0.005, Holm p<.05 and 95% CI inside (CERT95) | TOST-only (CERT90) | +-0.003 TOST | non-inferior (one-sided, Holm, +-0.005) ===", worlds.len());
        for (k, sb) in subjects.iter().enumerate() {
            let get = |sp: &'static str, m: u32| verdicts.get(&(k, sp, m)).cloned().unwrap_or_default();
            let sel5: Vec<Vec<(bool, bool, bool)>> = spl.iter().map(|s| get(s, 50)).collect();
            let sel3: Vec<Vec<(bool, bool, bool)>> = spl.iter().map(|s| get(s, 30)).collect();
            let all95 = sel5.iter().flatten().all(|v| v.1);
            let all90 = sel5.iter().flatten().all(|v| v.0);
            let all_ni = sel5.iter().flatten().all(|v| v.2);
            let all90_003 = sel3.iter().flatten().all(|v| v.0);
            let failing: Vec<String> = (0..worlds.len()).filter(|&wi| !sel5.iter().all(|s| s[wi].1)).map(|wi| worlds[wi].label()).collect();
            let ni_fail: Vec<String> = (0..worlds.len()).filter(|&wi| !sel5.iter().all(|s| s[wi].2)).map(|wi| worlds[wi].label()).collect();
            println!("{:22} PASS95: {:5}  CERT90: {:5}  +-.003: {:5}  NONINF: {:5}  failing(95%): {}   not-noninferior: {}", sb.name, all95, all90, all90_003, all_ni, if failing.is_empty() { "-".into() } else { failing.join(", ") }, if ni_fail.is_empty() { "-".into() } else { ni_fail.join(", ") });
            pass_json.push(json!({"stage": stage, "subject": sb.name, "pass_95": all95, "cert_90": all90, "cert_90_strict_0003": all90_003, "noninferior_all_worlds": all_ni, "failing_worlds_95": failing, "not_noninferior_worlds": ni_fail}));
        }
    }
    let max_abs = results.iter().filter(|r| r["subject"] == "compact-f64").map(|r| r["diff_mean"].as_f64().unwrap().abs().max(r["diff_sd"].as_f64().unwrap())).fold(0.0, f64::max);
    println!("\ncompact-f64 (new sorted-row/u128-mask layout) vs reference: max |mean diff| / sd over every cell = {max_abs:e} (must be 0 when present)");
    let j = json!({
        "bin": "h5_equiv",
        "protocol": {"metric": "full-run mean expected success", "margins": [0.005, 0.003], "splits": splits.iter().map(|s| json!({"name": s.name, "n": s.seeds.len()})).collect::<Vec<_>>(),
            "pass": "stage 1: all worlds, pooled and power splits; stage 2: fresh 91-450; +-0.005, Holm p<.05 and 95% CI inside margin; secondary: one-sided non-inferiority", "quick": quick, "natural_size_seeds": nat_seeds},
        "worlds": worlds.iter().map(|w| w.label()).collect::<Vec<_>>(),
        "natural_sizes": nats.iter().map(|n| json!({"rows": n.rows, "vocab": n.vocab, "pinned": n.pinned})).collect::<Vec<_>>(),
        "subjects": subjects.iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
        "pass": pass_json, "bytes": bytes_json, "results": results,
        "compact_f64_max_abs_diff": max_abs, "runtime_s": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out} ({:.1}s)", t0.elapsed().as_secs_f64());
    let _: Option<&dyn Router> = None;
}
