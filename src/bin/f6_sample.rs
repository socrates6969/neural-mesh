//! F6 part 3: SAMPLE EFFICIENCY of semantic retrieval. Tasks (and graded outcomes) needed until the trailing-window mean
//! EXPECTED success first reaches a target, with MiniLM retrieval vs hash retrieval vs none.
//!
//! "none" = the same Guarded(hebr) front without knowledge (in W: guarded+reuse, which receives the same delayed graded
//! reuse values). Baselines = keyword, eps-greedy, pooled-TS, hebbian-robust, warm, guarded, guarded+reuse, warm+reuse
//! (best fair baseline per world/target). Candidates = kn-feat (hidden behind Guarded) and kn-direct+vote, each with f5's
//! leader/vote anchors and with the f6 anchor fix (frozen from f6_anchor tune).
//! Regimes: W = weak raw hook bit + delayed graded reuse (default path), O = outcome feedback every task (isolates retrieval).
//! Metrics: T = tasks until the trailing-WIN-task mean first reaches the target (all arms see the same task stream);
//! G = graded outcomes delivered by then (W: delayed reuse values, only meaningful among arms that receive them; O: G == T).
//! Censoring: a run that never reaches the target counts the whole run (reached% reported).
//! Targets: per world and regime = best fair baseline's late level (tasks 1000-1999, TUNE seeds 1-8) + fixed deltas
//! (W: +.02/.04/.06/.08, O: +.05/.10/.15/.20); window 200. Held-out 31-60, frozen confirmation 61-90, paired 95% t-CI, Holm per family.
//!
//! cargo run --release --offline --features real-embed --bin f6_sample -- [--held 61-90] [--win 200]

#[path = "f6_common/mod.rs"]
mod common;

use std::time::Instant;

use common::*;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use serde_json::{json, Value};

/// fixed deltas above the best fair baseline's late level (seeds 1-8)
fn deltas(reg: &str) -> [f64; 4] {
    if reg == "W" { [0.02, 0.04, 0.06, 0.08] } else { [0.05, 0.10, 0.15, 0.20] }
}
static WIN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(200);
fn win() -> usize {
    WIN.load(std::sync::atomic::Ordering::Relaxed)
}

/// late level of the best fair baseline on TUNE seeds 1-8 -> targets for this world/regime
fn world_targets(sp: &Spec, reg: &str, bases: &[(String, A)]) -> (f64, Vec<f64>) {
    let seeds: Vec<u64> = (1..=8).collect();
    let mut best = f64::NEG_INFINITY;
    for (_, a) in bases {
        let rs = eval(a, sp, &seeds);
        let lv = summarize(&rs.iter().map(|r| r.pe[1000..2000].iter().map(|&x| x as f64).sum::<f64>() / 1000.0).collect::<Vec<_>>()).mean;
        best = best.max(lv);
    }
    (best, deltas(reg).iter().map(|d| ((best + d) * 1000.0).round() / 1000.0).collect())
}

fn cols(prefix: &str) -> Vec<String> {
    (1..=4).map(|i| format!("{prefix}{i}")).collect()
}

fn best_new(path: &str) -> Method {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|_| panic!("{path} missing: run f6_anchor --part tune"))).unwrap();
    method_from_json(&v["selected"]["best_new"])
}

fn world_names(src: &str) -> Vec<String> {
    ["q", "subj3", "q-mild", "q-mixed50", "q-k6", "para", "q2para", "ctl-irrel", "ctl-unique", "ctl-none"].iter().map(|s| format!("{src}-{s}")).collect()
}

fn arms_for(src: &str, reg_sig: Signal, new: &Method) -> Vec<(String, A)> {
    let vm = VoteCfg { min_n: 3.0, tau: 0.7, gate: Some((6.0, 0.02, 25)) };
    let mut a = vec![];
    if src == "mini" {
        a.push(("mini:feat-f5".to_string(), (Kind2::Kn(f5_mini_method()), reg_sig)));
        a.push(("mini:feat-new".to_string(), (Kind2::Kn(new.clone()), reg_sig)));
        a.push(("mini:vote-f5".to_string(), (Kind2::KnVote(f5_mini_vote().0, f5_mini_vote().1), reg_sig)));
        a.push(("mini:vote-new".to_string(), (Kind2::KnVote(new.clone(), vm.clone()), reg_sig)));
    } else {
        a.push(("hash:feat-f5".to_string(), (Kind2::Kn(f5_hash_method()), reg_sig)));
        a.push(("hash:feat-new".to_string(), (Kind2::Kn(new.clone()), reg_sig)));
        a.push(("hash:vote-f5".to_string(), (Kind2::KnVote(f5_hash_vote().0, f5_hash_vote().1), reg_sig)));
        a.push(("hash:vote-new".to_string(), (Kind2::KnVote(new.clone(), vm), reg_sig)));
    }
    a
}

/// mean expected success per 100-task block (20 values) over seeds
fn curve(rs: &[RunOut]) -> Vec<f64> {
    (0..20).map(|b| rs.iter().map(|r| r.pe[b * 100..(b + 1) * 100].iter().map(|&x| x as f64).sum::<f64>() / 100.0).sum::<f64>() / rs.len() as f64).collect()
}

fn main() {
    let t0 = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str, d: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned().unwrap_or(d.into());
    WIN.store(arg("--win", "200").parse().unwrap(), std::sync::atomic::Ordering::Relaxed);
    let held = parse_range(&args, "--held", (31, 60));
    let confirm = held[0] == 61;
    let new = best_new(&arg("--selected", "results/f6_anchor_selected.json"));
    println!("f6_sample: held {}-{}, window {}, new anchor method {} (frozen), real-embed {}", held[0], held[held.len() - 1], win(), new.label(), real_embed_feature());
    let out = arg("--out", if confirm { "results/f6_sample_confirm61-90.json" } else { "results/f6_sample.json" });
    let ns = held.len();
    let mut doc = serde_json::Map::new();
    let mut tables = vec![];
    let mut tgdoc = serde_json::Map::new();
    let ct = cols("T");
    let cg = cols("G");
    let ext_t = |r: &RunOut| r.tt.clone();
    let ext_g = |r: &RunOut| r.gg.clone();
    for (reg, sg) in [("W", Signal::Reuse), ("O", Signal::Outcome)] {
        let bases = if reg == "W" { bases_w() } else { bases_o() };
        let nb = bases.len();
        let front_name = if reg == "W" { "guarded(hebr)+reuse" } else { "guarded(hebr)" };
        let mut runs_by_src = vec![];
        for src in ["mini", "hash"] {
            let mut specs: Vec<Spec> = world_names(src).iter().map(|n| spec(n)).collect();
            for sp in specs.iter_mut() {
                let (lvl, tg) = world_targets(sp, reg, &bases);
                sp.targets = tg.clone();
                sp.win = win();
                tgdoc.insert(format!("{reg}/{}", sp.name), json!({"best_baseline_late_level_seeds1_8": lvl, "targets": tg}));
                println!("  targets {reg} {:<16} best-baseline late level {:.3} -> {:?}", sp.name, lvl, tg);
            }
            let mut arms = bases.clone();
            arms.extend(arms_for(src, sg, &new));
            let tag = format!("{src}-{reg}");
            println!("\n######## {tag}: {} worlds x {} arms x {} seeds | T1..T4 = tasks until the trailing-{} mean expected success >= (best fair baseline late level + delta) (censored at 2000); lower is better", specs.len(), arms.len(), ns, win());
            let runs = run_all(&specs, &arms, &held);
            print_table(&runs, &ct, &ext_t, &format!("{tag}-T"), &mut tables);
            if reg == "W" {
                print_table(&runs, &cg, &ext_g, &format!("{tag}-G"), &mut tables);
            }
            // reached fraction + curves
            let mut reach = vec![];
            for (si, sp) in runs.specs.iter().enumerate() {
                let mut rows = vec![];
                for (ai, n) in runs.names.iter().enumerate() {
                    let rr: Vec<f64> = (0..4).map(|ti| summarize(&runs.r[si][ai].iter().map(|r| r.rr[ti]).collect::<Vec<_>>()).mean).collect();
                    rows.push(json!({"policy": n, "reached": rr, "curve_per100": curve(&runs.r[si][ai])}));
                }
                reach.push(json!({"world": sp.name, "targets": sp.targets, "rows": rows}));
                if sp.name.ends_with("-q") || sp.name.ends_with("-q-k6") || sp.name.ends_with("-para") || sp.name.ends_with("-q-mixed50") {
                    println!("  reached% [{}] targets {:?}: {}", sp.name, sp.targets, runs.names.iter().enumerate().filter(|(ai, _)| *ai >= nb || runs.names[*ai] == front_name || runs.names[*ai] == "keyword").map(|(ai, n)| format!("{n}: {}", (0..4).map(|ti| format!("{:.0}", 100.0 * summarize(&runs.r[si][ai].iter().map(|r| r.rr[ti]).collect::<Vec<_>>()).mean)).collect::<Vec<_>>().join("/"))).collect::<Vec<_>>().join(" | "));
                }
            }
            doc.insert(format!("{tag}_reached_curves"), json!(reach));
            let base_idx: Vec<usize> = (0..nb).collect();
            let cand_idx: Vec<usize> = (nb..arms.len()).collect();
            // tasks-to-target vs the best fair baseline
            let cells = cells_vs_best(&runs, &base_idx, &cand_idx, &ct, &ext_t, false);
            let (rep, adj) = report(&format!("[{tag}] TASKS-to-target vs best fair baseline (negative = fewer tasks = better)"), &cells, ns, false);
            matrix(&format!("{tag} tasks vs best fair baseline"), &cells, &adj, "T2", false);
            doc.insert(format!("{tag}_T_vs_best"), rep);
            let front = runs.names.iter().position(|n| n == front_name).unwrap();
            let cells_f = cells_vs_fixed(&runs, front, &cand_idx, &ct, &ext_t);
            let (rep_f, adj_f) = report(&format!("[{tag}] TASKS-to-target vs none (= {front_name})"), &cells_f, ns, false);
            matrix(&format!("{tag} tasks vs none"), &cells_f, &adj_f, "T2", false);
            doc.insert(format!("{tag}_T_vs_none"), rep_f);
            if reg == "W" {
                // graded outcomes: only arms that receive the delayed graded reuse values are comparable
                let reuse_bases: Vec<usize> = ["guarded(hebr)+reuse", "warm(hebr)+reuse"].iter().map(|n| runs.names.iter().position(|x| x == n).unwrap()).collect();
                let cells_g = cells_vs_best(&runs, &reuse_bases, &cand_idx, &cg, &ext_g, false);
                let (rep_g, adj_g) = report(&format!("[{tag}] GRADED OUTCOMES-to-target vs best reuse-fed baseline (guarded/warm + reuse)"), &cells_g, ns, false);
                matrix(&format!("{tag} graded outcomes vs reuse baselines"), &cells_g, &adj_g, "G2", false);
                doc.insert(format!("{tag}_G_vs_reusebase"), rep_g);
            }
            if src == "mini" {
                // new anchors vs f5 anchors, same source and arm
                let pairs = [("mini:feat-new", "mini:feat-f5"), ("mini:vote-new", "mini:vote-f5")];
                let mut cells_n = vec![];
                for (c, r) in pairs {
                    let ri = runs.names.iter().position(|n| n == r).unwrap();
                    let ci = runs.names.iter().position(|n| n == c).unwrap();
                    cells_n.extend(cells_vs_fixed(&runs, ri, &[ci], &ct, &ext_t).into_iter().map(|mut x| {
                        x.cand = format!("{c} - {}", r.trim_start_matches("mini:"));
                        x
                    }));
                }
                let (rep_n, adj_n) = report(&format!("[{tag}] TASKS: f6 anchors minus f5 anchors (same retrieval arm)"), &cells_n, ns, false);
                matrix(&format!("{tag} new - f5 anchors"), &cells_n, &adj_n, "T2", false);
                doc.insert(format!("{tag}_T_new_vs_f5anchor"), rep_n);
            }
            runs_by_src.push(runs);
        }
        // mini vs hash, same task stream (paired by seed); the targets differ per world only through the (identical) baselines,
        // and the mini/hash worlds share the same baselines & stream, so the targets are identical
        let (rm, rh) = (&runs_by_src[0], &runs_by_src[1]);
        let mut cm = vec![];
        let mut ps = vec![];
        for (an, bn) in [("mini:feat-new", "hash:feat-f5"), ("mini:vote-new", "hash:vote-f5"), ("mini:feat-f5", "hash:feat-f5"), ("mini:vote-f5", "hash:vote-f5")] {
            let ai = rm.names.iter().position(|n| n == an).unwrap();
            let bi = rh.names.iter().position(|n| n == bn).unwrap();
            for (si, sp) in rm.specs.iter().enumerate() {
                for (ci, cn) in ct.iter().enumerate() {
                    let a: Vec<f64> = rm.r[si][ai].iter().map(|r| r.tt[ci]).collect();
                    let b: Vec<f64> = rh.r[si][bi].iter().map(|r| r.tt[ci]).collect();
                    let d = paired_diff(&a, &b);
                    ps.push(paired_t_pvalue(&a, &b));
                    cm.push((sp.name.clone(), format!("{an} - {bn}"), cn.clone(), d.diff.mean, d.diff.lo, d.diff.hi, summarize(&a).mean, summarize(&b).mean, sp.treatment));
                }
            }
        }
        let adj = holm_adjust(&ps);
        let (mut nw, mut nl) = (0, 0);
        let mut js = vec![];
        for (c, a) in cm.iter().zip(&adj) {
            if *a < 0.05 {
                if c.3 < 0.0 { nw += 1 } else { nl += 1 }
            }
            js.push(json!({"world": c.0, "pair": c.1, "col": c.2, "diff": c.3, "ci": [c.4, c.5], "mini_mean": c.6, "hash_mean": c.7, "holm_p": a, "treatment": c.8}));
        }
        println!("\n### [{reg}] MiniLM minus hash arms on the same stream, tasks-to-target (Holm over {} tests) -> mini needs fewer {nw}, more {nl}", cm.len());
        for (c, a) in cm.iter().zip(&adj) {
            if c.2 == "T2" && c.8 && (c.1.starts_with("mini:feat-new") || c.1.starts_with("mini:vote-new")) {
                println!("  {:<14} {:<30} T2 diff {:+9.1} [{:+8.1},{:+8.1}] (mini {:.0} hash {:.0}) Holm {:.4}", c.0, c.1, c.3, c.4, c.5, c.6, c.7, a);
            }
        }
        doc.insert(format!("{reg}_mini_vs_hash_T"), json!({"holm_mini_fewer": nw, "holm_mini_more": nl, "cells": js}));
    }
    doc.insert("lab".into(), json!("f6_sample"));
    doc.insert("held".into(), json!([held[0], held[held.len() - 1]]));
    doc.insert("window".into(), json!(win()));
    doc.insert("deltas".into(), json!({"W": deltas("W"), "O": deltas("O")}));
    doc.insert("world_targets".into(), Value::Object(tgdoc));
    doc.insert("new_anchor_method".into(), method_json(&new));
    doc.insert("note".into(), json!("T_i = tasks until the trailing-window mean expected success >= best-fair-baseline late level (seeds 1-8) + delta_i; G_i = graded outcomes delivered by then (W: delayed reuse values, comparable only among reuse-fed arms; O: G == T); censored at run end"));
    doc.insert("tables".into(), json!(tables));
    doc.insert("elapsed_s".into(), json!(t0.elapsed().as_secs_f64()));
    std::fs::write(&out, serde_json::to_string_pretty(&Value::Object(doc)).unwrap()).unwrap();
    println!("\nwrote {out} in {:.0}s", t0.elapsed().as_secs_f64());
}
