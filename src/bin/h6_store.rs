//! H6-store (R4-B3, memory-reduction track): compression of the KNOWLEDGE STORE and measured downstream quality.
//!   check   run6 (flat/f32) == g16::run bit-exactly                                    (sanity of the copied engine)
//!   loop    closed loop (g16 worlds, R0 and FROZEN FINAL policies) with quantised stores, exact-dup suppression, anchor-summary rows,
//!           k truncation / adaptive k, TOST +-0.01 on H2000 vs the f32 flat store (held-out seeds 31-60 corpus A; --confirm: corpus B, 61-90, same arms)
//!   docq    document-level retrieval with quantised stores on the real MiniLM/BGE fixtures: recall@k vs f32, F1@3, hit@1, MRR, TOST
//!   ctx     retrieved-context bytes frontier: fixed k / adaptive k / byte budget / exact-dup suppression, quality vs injected chars
//! SYNTHETIC-until-validated-on-real-traffic. Nothing changes defaults; no src/*.rs edits.
//! Usage: h6_store <mode> [--corpus A] [--held 31-60] [--confirm] [--out results]

use std::collections::HashMap;
use std::time::Instant;

use neural_mesh::router::Rng;
use neural_mesh::stats::summarize;
use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
#[path = "g16_common/mod.rs"]
mod g16;
#[path = "h6_common/mod.rs"]
mod h6;
use common::*;
use g16::{cmp, holm, PolCfg, Spec};
use h6::*;

fn load_cfgs() -> (PolCfg, PolCfg) {
    let v: Value = serde_json::from_str(&std::fs::read_to_string("results/g16_picks.json").expect("results/g16_picks.json")).unwrap();
    let find = |n: &str| cfg_from(v["arms"].as_array().unwrap().iter().find(|a| a["name"] == n).unwrap_or_else(|| panic!("arm {n}")));
    (find("R0 (g1 retrieval)"), find("FINAL[mini-hybrid]"))
}

struct Arm {
    name: String,
    var: usize,
    kind: StoreKind,
    cfg_mod: Option<(usize, Option<f32>)>, // (k override, adapt override)
    thr: Option<f32>,
    group: &'static str,
}

fn dense_bytes_of(name: &str, d: usize) -> usize {
    match name {
        "f32" => Q::F32.bytes(d),
        "f16" => Q::F16.bytes(d),
        "bfp16" => Q::BlockFp16.bytes(d),
        "int8" => Q::Int8.bytes(d),
        "int8q" => Q::Int8QQ.bytes(d),
        "tv4" => tv_bytes(4, d),
        "tv3" => tv_bytes(3, d),
        "tv2" => tv_bytes(2, d),
        _ => unreachable!(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_default();
    let tag = arg(&args, "--corpus", "A");
    let confirm = args.iter().any(|a| a == "--confirm");
    let held = parse_range(&args, "--held", if confirm { (61, 90) } else { (31, 60) });
    let out_dir = arg(&args, "--out", "results");
    let t0 = Instant::now();
    match mode.as_str() {
        "check" | "loop" => {
            let tag = if confirm && !args.iter().any(|a| a == "--corpus") { "B".to_string() } else { tag };
            let corpus = load_corpus(&tag);
            let pool = g16::build_pool(&corpus);
            let specs: Vec<(String, Option<Q>, Option<usize>)> = vec![
                ("f32".into(), Some(Q::F32), None),
                ("f16".into(), Some(Q::F16), None),
                ("bfp16".into(), Some(Q::BlockFp16), None),
                ("int8".into(), Some(Q::Int8), None),
                ("int8q".into(), Some(Q::Int8QQ), None),
                ("tv4".into(), None, Some(4)),
                ("tv3".into(), None, Some(3)),
                ("tv2".into(), None, Some(2)),
            ];
            let ss6 = build_sims6(&corpus, &pool, &specs);
            println!("corpus {tag}: nq {} dense variants {} built in {:.1}s", ss6.nq, specs.len(), t0.elapsed().as_secs_f64());
            let ws = g16::worlds();
            let (r0, fin) = load_cfgs();
            let d = ss6.d;
            let mk_var = |n: &str, rr: Option<(&str, usize)>| Var {
                name: match rr { Some((e, f)) => format!("{n}+rerank{f}x({e})"), None => n.to_string() },
                approx: ss6.key(n),
                rerank: rr.map(|(e, f)| (ss6.key(e), f)),
                bytes: dense_bytes_of(n, d),
                exact_bytes: rr.map(|(e, _)| dense_bytes_of(e, d)).unwrap_or(0),
            };
            if mode == "check" {
                let ss = g16::sims(&pool);
                let v = mk_var("f32", None);
                let mut worst = 0.0f64;
                for sp in ws.iter().filter(|s| ["sib16", "ctl-irrel", "sib16-drift"].contains(&s.name)) {
                    for seed in [31u64, 32, 33] {
                        for cfg in [&r0, &fin] {
                            let a = g16::run(cfg, sp, seed, &corpus, &pool, &ss);
                            let b = run6(cfg, sp, seed, &corpus, &ss6, &v, StoreKind::Flat, false);
                            let dm = (0..4).map(|i| (a.h[i] - b.h[i]).abs()).fold(0.0, f64::max).max((a.late - b.late).abs());
                            worst = worst.max(dm);
                        }
                    }
                }
                println!("CHECK run6(flat,f32) vs g16::run: max |d| over H100..H2000,late = {worst:.3e}");
                return;
            }
            // ---------------- arms
            let mut arms: Vec<Arm> = vec![];
            let flat = |n: &str, v: usize, g: &'static str| Arm { name: n.into(), var: v, kind: StoreKind::Flat, cfg_mod: None, thr: None, group: g };
            let mut vars: Vec<Var> = vec![];
            let mut addv = |v: Var| -> usize { vars.push(v); vars.len() - 1 };
            let v_f32 = addv(mk_var("f32", None));
            let v_f16 = addv(mk_var("f16", None));
            let v_bfp = addv(mk_var("bfp16", None));
            let v_i8 = addv(mk_var("int8", None));
            let v_i8q = addv(mk_var("int8q", None));
            let v_t4 = addv(mk_var("tv4", None));
            let v_t3 = addv(mk_var("tv3", None));
            let v_t2 = addv(mk_var("tv2", None));
            let v_t4r = addv(mk_var("tv4", Some(("f32", 4))));
            let v_t2r = addv(mk_var("tv2", Some(("f32", 4))));
            let v_t2r16 = addv(mk_var("tv2", Some(("f32", 16))));
            arms.push(flat("flat f32 (incumbent)", v_f32, "base"));
            for (n, v) in [("flat f16", v_f16), ("flat block-fp16", v_bfp), ("flat int8 (f32 query)", v_i8), ("flat int8 (int8 query)", v_i8q), ("flat turbovec-4bit", v_t4), ("flat turbovec-3bit", v_t3), ("flat turbovec-2bit", v_t2), ("flat tv4 + f32 rerank 4x", v_t4r), ("flat tv2 + f32 rerank 4x", v_t2r), ("flat tv2 + f32 rerank 16x", v_t2r16)] {
                arms.push(flat(n, v, "quant"));
            }
            for (n, v) in [("dedup f32", v_f32), ("dedup block-fp16", v_bfp), ("dedup int8", v_i8)] {
                arms.push(Arm { name: n.into(), var: v, kind: StoreKind::Dedup, cfg_mod: None, thr: None, group: "dedup" });
            }
            arms.push(Arm { name: "summary rows f32 thr(frozen)".into(), var: v_f32, kind: StoreKind::Summary, cfg_mod: None, thr: None, group: "summary" });
            for thr in [0.55f32, 0.65] {
                arms.push(Arm { name: format!("summary rows f32 thr{thr:.2} (post-hoc)"), var: v_f32, kind: StoreKind::Summary, cfg_mod: None, thr: Some(thr), group: "summary" });
            }
            arms.push(Arm { name: "summary rows + dedup n/a".into(), var: v_f32, kind: StoreKind::Flat, cfg_mod: None, thr: None, group: "skip" });
            arms.pop();
            for k in [1usize, 2, 3, 5, 12] {
                arms.push(Arm { name: format!("k={k}"), var: v_f32, kind: StoreKind::Flat, cfg_mod: Some((k, None)), thr: None, group: "k" });
            }
            for dd in [0.03f32, 0.06, 0.10] {
                arms.push(Arm { name: format!("adaptive k (drop {dd:.2}, kmax 8)"), var: v_f32, kind: StoreKind::Flat, cfg_mod: Some((8, Some(dd))), thr: None, group: "k" });
            }
            let n_seeds = held.len();
            let pols = [("R0", &r0), ("FINAL", &fin)];
            let mut report: Vec<Value> = vec![];
            let mut md = String::new();
            md.push_str(&format!("# H6 closed-loop store compression, corpus {tag}, seeds {}..{} (n={n_seeds}), primary metric H2000, TOST +-0.01 vs flat f32 of the SAME policy; Holm over the 8 worlds per arm\n\n", held[0], held[held.len() - 1]));
            for (pn, pc) in pols {
                // base results per world
                let wsr: Vec<&Spec> = ws.iter().collect();
                let mut base_h: Vec<Vec<f64>> = vec![];
                let mut base_late: Vec<Vec<f64>> = vec![];
                for (ai, arm) in arms.iter().enumerate() {
                    let mut cfg = pc.clone();
                    if let Some((k, ad)) = arm.cfg_mod {
                        cfg.k = k;
                        cfg.adapt = ad;
                    }
                    if let Some(t) = arm.thr {
                        cfg.thr = t;
                    }
                    let var = &vars[arm.var];
                    let diag = arm.kind == StoreKind::Flat && (arm.group == "quant");
                    let mut per_world: Vec<(Vec<f64>, Vec<f64>, f64, f64, f64, f64, f64)> = vec![];
                    for sp in &wsr {
                        let r: Vec<Out6> = held.par_iter().map(|&s| run6(&cfg, sp, s, &corpus, &ss6, var, arm.kind, diag)).collect();
                        let h2: Vec<f64> = r.iter().map(|x| x.h[3]).collect();
                        let lt: Vec<f64> = r.iter().map(|x| x.late).collect();
                        let rows = r.iter().map(|x| x.rows as f64).sum::<f64>() / r.len() as f64;
                        let nbr = { let v: Vec<f64> = r.iter().map(|x| x.nb_recall).filter(|x| x.is_finite()).collect(); if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 } };
                        let nbm = r.iter().map(|x| x.nb_mean).sum::<f64>() / r.len() as f64;
                        let rd = r.iter().map(|x| x.reads_per_task).sum::<f64>() / r.len() as f64;
                        per_world.push((h2, lt, rows, nbr, nbm, rd, 0.0));
                    }
                    if ai == 0 {
                        base_h = per_world.iter().map(|p| p.0.clone()).collect();
                        base_late = per_world.iter().map(|p| p.1.clone()).collect();
                    }
                    // comparisons vs base per world
                    let cmps: Vec<g16::Cmp> = per_world.iter().enumerate().map(|(wi, p)| cmp(&p.0, &base_h[wi], 0.01)).collect();
                    let tp: Vec<f64> = cmps.iter().map(|c| c.tost_p).collect();
                    let adj = holm(&tp);
                    let eq_all = adj.iter().all(|&p| p < 0.05);
                    let worst = cmps.iter().map(|c| c.diff).fold(f64::INFINITY, f64::min);
                    let worst_lo = cmps.iter().map(|c| c.lo).fold(f64::INFINITY, f64::min);
                    let worst_hi = cmps.iter().map(|c| c.hi).fold(f64::NEG_INFINITY, f64::max);
                    let rows_mean = per_world.iter().map(|p| p.2).sum::<f64>() / per_world.len() as f64;
                    let rowb = row_bytes(var.bytes);
                    let store_kb = rows_mean * rowb as f64 / 1024.0;
                    let base_rows = 2000.0 * row_bytes(vars[v_f32].bytes) as f64 / 1024.0;
                    md.push_str(""); // header per policy below
                    let jw: Vec<Value> = wsr.iter().enumerate().map(|(wi, sp)| {
                        json!({"world": sp.name, "H2000": summarize(&per_world[wi].0).mean, "late": summarize(&per_world[wi].1).mean, "diff_H2000": cmps[wi].diff, "ci": [cmps[wi].lo, cmps[wi].hi], "tost_p": cmps[wi].tost_p, "tost_p_holm": adj[wi], "equiv": adj[wi] < 0.05,
                            "diff_late": summarize(&per_world[wi].1.iter().zip(&base_late[wi]).map(|(a, b)| a - b).collect::<Vec<_>>()).mean, "rows": per_world[wi].2, "nb_recall": per_world[wi].3, "nb_mean": per_world[wi].4, "rerank_reads_per_task": per_world[wi].5})
                    }).collect();
                    report.push(json!({"policy": pn, "arm": arm.name, "group": arm.group, "store": format!("{:?}", arm.kind), "dense_bytes_per_vec": var.bytes, "exact_copy_bytes_per_vec": var.exact_bytes, "row_bytes": rowb,
                        "mean_rows": rows_mean, "store_kb_at_end": store_kb, "store_kb_baseline_f32_flat": base_rows, "equiv_all_worlds": eq_all, "worst_world_diff": worst, "worst_ci_lo": worst_lo, "best_ci_hi": worst_hi, "max_tost_p_raw": tp.iter().cloned().fold(0.0, f64::max), "worlds": jw}));
                    println!("[{pn}] {:<34} {:<8} rows {:>7.1} store {:>8.1} KB  worst-world diff {:+.4} [CI lo {:+.4}]  equiv-all(Holm/8,+-.01) {}  maxTOSTp {:.3e}  ({:.0}s)", arm.name, format!("{:?}", arm.kind), rows_mean, store_kb, worst, worst_lo, eq_all, tp.iter().cloned().fold(0.0, f64::max), t0.elapsed().as_secs_f64());
                }
            }
            let path = format!("{out_dir}/h6_loop_{tag}.json");
            std::fs::write(&path, serde_json::to_string_pretty(&json!({"corpus": tag, "seeds": [held[0], held[held.len() - 1]], "margin": 0.01, "metric": "H2000", "arms": report})).unwrap()).unwrap();
            let _ = md;
            println!("wrote {path} in {:.0}s", t0.elapsed().as_secs_f64());
        }
        "loopbase" => {
            // retrieval gain context: H2000 of the SAME front without retrieval (guarded(hebr)) vs f32 flat store R0/FINAL, per world
            let tag = if confirm && !args.iter().any(|a| a == "--corpus") { "B".to_string() } else { tag };
            let corpus = load_corpus(&tag);
            let pool = g16::build_pool(&corpus);
            let ss = g16::sims(&pool);
            let ss6 = build_sims6(&corpus, &pool, &[("f32".into(), Some(Q::F32), None)]);
            let (r0, fin) = load_cfgs();
            let v = Var { name: "f32".into(), approx: 0, rerank: None, bytes: 1536, exact_bytes: 0 };
            let g = PolCfg::base_kind("guarded(hebr)", g16::Base::Guard);
            let mut rows = vec![];
            for sp in g16::worlds() {
                let rg: Vec<f64> = held.par_iter().map(|&s| g16::run(&g, &sp, s, &corpus, &pool, &ss).h[3]).collect();
                let a: Vec<f64> = held.par_iter().map(|&s| run6(&r0, &sp, s, &corpus, &ss6, &v, StoreKind::Flat, false).h[3]).collect();
                let b: Vec<f64> = held.par_iter().map(|&s| run6(&fin, &sp, s, &corpus, &ss6, &v, StoreKind::Flat, false).h[3]).collect();
                let (mg, ma, mb) = (summarize(&rg).mean, summarize(&a).mean, summarize(&b).mean);
                println!("{tag} {:<15} no-retrieval {mg:.4}  R0 {ma:.4} ({:+.4})  FINAL {mb:.4} ({:+.4})", sp.name, ma - mg, mb - mg);
                rows.push(json!({"world": sp.name, "no_retrieval": mg, "R0": ma, "FINAL": mb}));
            }
            std::fs::write(format!("{out_dir}/h6_loopbase_{tag}.json"), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
        }
        _ => {
            eprintln!("usage: h6_store check|loop|docq|ctx [--corpus A] [--held a-b] [--confirm] [--out results]");
        }
    }
    let _ = (HashMap::<u8, u8>::new(), Rng::new(1));
}
