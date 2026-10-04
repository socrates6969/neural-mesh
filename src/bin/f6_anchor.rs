//! F6 part 2: fix MiniLM anchor fragmentation (and check paraphrase robustness) + downstream knowledge gain.
//!
//! Problem (f5): with the library's leader/vote clustering (`Knowledge::retrieve`: top-k neighbours >= sim_min vote
//! for an anchor, else a new anchor is founded) the anchors of MiniLM FRAGMENT as the threshold rises (anchor pair-F1
//! .87 at sim .5 -> .62 at .7) while the neighbour list stays near perfect; the best threshold for the anchor is
//! (almost) the neighbour threshold, and a retune is needed per embedder.
//!
//! Anchor methods implemented in the bin (src is untouched), all online/causal:
//!   leader      = the library's Knowledge anchor (f5 baseline)
//!   centroid    = a query joins the best anchor CENTROID (cos >= tau_a) else founds an anchor (centroids average out template noise)
//!   centroid+merge = + agglomerative merge of anchor pairs whose centroids have cos >= tau_m (repairs early fragments)
//!   bridge      = kNN leader + union of anchors that both hold >= `support` neighbours >= tau_m of one query
//! The neighbour list (k, thr) is a SEPARATE stage from the anchor threshold, so the f4 "threshold conflict" disappears
//! by construction: neighbour retrieval keeps its own (k3/k5, .5-.6) setting.
//!
//! Protocol: tune on seeds 1-8 (anchor pair-F1 over 5 worlds), held-out 31-60 / frozen confirmation 61-90, paired 95% t-CI,
//! Holm per family; downstream = kn-feat (hidden behind Guarded) in the weak-feedback regime W (and the outcome regime O).
//!
//! cargo run --release --offline --features real-embed --bin f6_anchor -- [--part tune|retrieval|downstream|all] [--held 61-90]

#[path = "f6_common/mod.rs"]
mod common;

use std::time::Instant;

use common::*;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue};
use rayon::prelude::*;
use serde_json::{json, Value};

fn world_list(names: &[&str]) -> Vec<Spec> {
    names.iter().map(|n| spec(n)).collect()
}

/// [method][world][seed]
fn eval_methods(methods: &[Method], worlds: &[Spec], seeds: &[u64]) -> Vec<Vec<Vec<Anc>>> {
    let jobs: Vec<(usize, u64)> = (0..worlds.len()).flat_map(|w| seeds.iter().map(move |&s| (w, s))).collect();
    let res: Vec<Vec<Anc>> = jobs
        .par_iter()
        .map(|&(wi, s)| {
            let cw = CtxWorld::new(s, &worlds[wi]);
            let post = worlds[wi].corp == Corp::QThenPara;
            methods.iter().map(|m| anchor_metrics(&cw, DIM, m, post)).collect()
        })
        .collect();
    let mut out = vec![vec![vec![]; worlds.len()]; methods.len()];
    for (j, &(wi, _)) in jobs.iter().enumerate() {
        for mi in 0..methods.len() {
            out[mi][wi].push(res[j][mi].clone());
        }
    }
    out
}

fn tune_grid() -> Vec<Method> {
    let mut v = vec![];
    for &c in &[false, true] {
        for &k in &[5usize, 8] {
            for &s in &[0.4f32, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7] {
                v.push(Method { anchor: AnchorCfg::Leader { k, sim: s }, center: c });
            }
        }
        for &a in &[0.35f32, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65] {
            v.push(Method { anchor: AnchorCfg::Centroid { tau_a: a, tau_m: None, merge_min: 1, knn: None, avg: false }, center: c });
        }
        for &a in &[0.4f32, 0.45, 0.5, 0.55] {
            for &m in &[0.45f32, 0.5, 0.55, 0.6, 0.65, 0.7] {
                for &mm in &[1u32, 3, 6] {
                    v.push(Method { anchor: AnchorCfg::Centroid { tau_a: a, tau_m: Some(m), merge_min: mm, knn: None, avg: false }, center: c });
                }
            }
        }
        // average-link (cluster-mean similarity instead of the normalised centroid)
        for &a in &[0.3f32, 0.35, 0.4, 0.45] {
            for &m in &[None, Some(0.3f32), Some(0.35), Some(0.4), Some(0.45)] {
                v.push(Method { anchor: AnchorCfg::Centroid { tau_a: a, tau_m: m, merge_min: 1, knn: None, avg: true }, center: c });
            }
        }
        // union of the kNN-vote view and the centroid view
        for &a in &[0.45f32, 0.5, 0.55] {
            for &tn in &[0.5f32, 0.55, 0.6, 0.65] {
                for &m in &[None, Some(0.55f32), Some(0.65)] {
                    v.push(Method { anchor: AnchorCfg::Centroid { tau_a: a, tau_m: m, merge_min: 3, knn: Some((8, tn)), avg: false }, center: c });
                }
            }
        }
    }
    for &k in &[5usize, 8] {
        for &n in &[0.45f32, 0.5, 0.55, 0.6] {
            for &m in &[0.55f32, 0.6, 0.65, 0.7] {
                for &sup in &[1usize, 2] {
                    v.push(Method { anchor: AnchorCfg::Bridge { k, tau_n: n, tau_m: m, support: sup }, center: false });
                }
            }
        }
    }
    v
}

const TUNE_WORLDS: [&str; 5] = ["mini-q", "mini-subj3", "mini-q-mixed50", "mini-para", "mini-q2para"];

fn tune(out_sel: &str) -> Value {
    let t0 = Instant::now();
    let seeds: Vec<u64> = (1..=8).collect();
    let worlds = world_list(&TUNE_WORLDS);
    let grid = tune_grid();
    println!("tune: {} configs x {} worlds x {} seeds (anchor pair-F1, mean over worlds)", grid.len(), worlds.len(), seeds.len());
    let res = eval_methods(&grid, &worlds, &seeds);
    let mut scored: Vec<(f64, usize)> = (0..grid.len())
        .map(|mi| (mean_of((0..worlds.len()).map(|wi| mean_of(res[mi][wi].iter().map(|a| a.f1)))), mi))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let pick = |pred: &dyn Fn(&Method) -> bool| -> (f64, Method) {
        let (s, mi) = scored.iter().find(|(_, mi)| pred(&grid[*mi])).unwrap();
        (*s, grid[*mi].clone())
    };
    let (sl, leader) = pick(&|m| matches!(m.anchor, AnchorCfg::Leader { .. }));
    let (sc, cen) = pick(&|m| matches!(m.anchor, AnchorCfg::Centroid { tau_m: None, knn: None, avg: false, .. }));
    let (sm, cm) = pick(&|m| matches!(m.anchor, AnchorCfg::Centroid { tau_m: Some(_), knn: None, avg: false, .. }));
    let (sa, av) = pick(&|m| matches!(m.anchor, AnchorCfg::Centroid { avg: true, .. }));
    let (su, un) = pick(&|m| matches!(m.anchor, AnchorCfg::Centroid { knn: Some(_), .. }));
    let (sb, br) = pick(&|m| matches!(m.anchor, AnchorCfg::Bridge { .. }));
    println!("tuned (mean anchor F1 over {} tune worlds, seeds 1-8):", worlds.len());
    for (n, s, m) in [("leader", sl, &leader), ("centroid", sc, &cen), ("centroid+merge", sm, &cm), ("cen+knn union", su, &un), ("avg-link", sa, &av), ("bridge", sb, &br)] {
        println!("  {:<15} {:.4}  {}", n, s, m.label());
    }
    println!("top 8 overall:");
    for (s, mi) in scored.iter().take(8) {
        println!("  {:.4}  {}", s, grid[*mi].label());
    }
    // leader f5 baselines on the same objective for reference
    let refs = [f5_d6_method(), f5_mini_method()];
    let rr = eval_methods(&refs, &worlds, &seeds);
    let rscore: Vec<f64> = (0..refs.len()).map(|mi| mean_of((0..worlds.len()).map(|wi| mean_of(rr[mi][wi].iter().map(|a| a.f1))))).collect();
    println!("  reference f5 d6cfg leader(k8 s.6) {:.4}, f5 mini-tuned leader(k8 s.5) {:.4}", rscore[0], rscore[1]);
    let best_new = [(sc, &cen), (sm, &cm), (su, &un), (sa, &av), (sb, &br)].into_iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
    let doc = json!({
        "lab": "f6_anchor/tune", "tune_seeds": [1, 8], "worlds": TUNE_WORLDS, "objective": "mean over worlds of mean-over-seeds anchor pair-F1 at arrival",
        "n_configs": grid.len(),
        "selected": {"leader": method_json(&leader), "centroid": method_json(&cen), "centroid_merge": method_json(&cm), "union": method_json(&un), "avglink": method_json(&av), "bridge": method_json(&br), "best_new": method_json(best_new.1)},
        "scores": {"leader": sl, "centroid": sc, "centroid_merge": sm, "union": su, "avglink": sa, "bridge": sb, "f5_d6cfg_leader": rscore[0], "f5_mini_tuned_leader": rscore[1]},
        "top": scored.iter().take(20).map(|(s, mi)| json!({"score": s, "method": grid[*mi].label()})).collect::<Vec<_>>(),
        "elapsed_s": t0.elapsed().as_secs_f64()
    });
    std::fs::write(out_sel, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out_sel} in {:.0}s", t0.elapsed().as_secs_f64());
    doc
}

struct Sel {
    leader: Method,
    cen: Method,
    cm: Method,
    un: Method,
    av: Method,
    br: Method,
}
fn load_sel(path: &str) -> Sel {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|_| panic!("{path} missing: run --part tune first"))).unwrap();
    let s = &v["selected"];
    Sel { leader: method_from_json(&s["leader"]), cen: method_from_json(&s["centroid"]), cm: method_from_json(&s["centroid_merge"]), un: method_from_json(&s["union"]), av: method_from_json(&s["avglink"]), br: method_from_json(&s["bridge"]) }
}

fn retrieval(sel: &Sel, held: &[u64], out: &str) -> Value {
    let t0 = Instant::now();
    let methods: Vec<(String, Method)> = vec![
        ("L-d6cfg".into(), f5_d6_method()),
        ("L-f5tuned".into(), f5_mini_method()),
        ("L-retuned".into(), sel.leader.clone()),
        ("centroid".into(), sel.cen.clone()),
        ("cen+merge".into(), sel.cm.clone()),
        ("cen+knn".into(), sel.un.clone()),
        ("avg-link".into(), sel.av.clone()),
        ("bridge".into(), sel.br.clone()),
    ];
    let ms: Vec<Method> = methods.iter().map(|m| m.1.clone()).collect();
    let wl = world_list(&["mini-q", "mini-subj3", "mini-q-mixed50", "mini-para", "mini-qpara", "mini-q2para", "mini-ctl-unique", "hash-q", "hash-subj3", "hash-para"]);
    let res = eval_methods(&ms, &wl, held);
    println!("\n=== ANCHOR layer vs ground-truth concept (held {}-{}) ===", held[0], held[held.len() - 1]);
    println!("P/R/F1 = pairwise anchor clustering over concept tasks (anchor as seen at arrival); frag = anchors/concept; early = pair-F1 over first 300 concept tasks;");
    println!("fin = pair-F1 after later merges; fh1 = first occurrences that joined an existing anchor; contam = one-off tasks sharing an anchor with a concept task; post* = q2para unseen-phrasing tasks (ok/wrong/new anchor)");
    println!("{:<16} {:<11} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>6} | {:>5} {:>5} {:>5}", "world", "method", "P", "R", "F1", "frag", "early", "fin", "fh1", "contam", "pOK", "pWR", "pNEW");
    let mut rows = vec![];
    for (wi, w) in wl.iter().enumerate() {
        for (mi, (mn, m)) in methods.iter().enumerate() {
            let a = &res[mi][wi];
            let g = |f: &dyn Fn(&Anc) -> f64| mean_of(a.iter().map(f));
            let v = [g(&|x| x.p), g(&|x| x.r), g(&|x| x.f1), g(&|x| x.frag), g(&|x| x.early_f1), g(&|x| x.final_f1), g(&|x| x.fh_first), g(&|x| x.contam), g(&|x| x.post_ok), g(&|x| x.post_wrong), g(&|x| x.post_new), g(&|x| x.n_anchors), g(&|x| x.post_f1)];
            println!("{:<16} {:<11} {:>5.2} {:>5.2} {:>5.3} {:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>6.2} | {:>5.2} {:>5.2} {:>5.2}", w.name, mn, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10]);
            rows.push(json!({"world": w.name, "method": mn, "label": m.label(), "p": v[0], "r": v[1], "f1": v[2], "frag": v[3], "early_f1": v[4], "final_f1": v[5], "fh_first": v[6], "contam": v[7],
                "post_ok": v[8], "post_wrong": v[9], "post_new": v[10], "n_anchors": v[11], "post_f1": v[12]}));
        }
    }
    // paired F1 comparisons vs the f5 baselines (treatment worlds)
    let mut pc = vec![];
    for (wi, w) in wl.iter().enumerate() {
        if w.name.contains("ctl") {
            continue;
        }
        for (rn, ri) in [("L-d6cfg", 0usize), ("L-f5tuned", 1usize)] {
            for ci in 2..methods.len() {
                let a: Vec<f64> = res[ci][wi].iter().map(|x| x.f1).collect();
                let b: Vec<f64> = res[ri][wi].iter().map(|x| x.f1).collect();
                let d = paired_diff(&a, &b);
                pc.push((w.name.clone(), methods[ci].0.clone(), rn, d.diff.mean, d.diff.lo, d.diff.hi, paired_t_pvalue(&a, &b)));
            }
        }
    }
    let adj = holm_adjust(&pc.iter().map(|x| x.6).collect::<Vec<_>>());
    println!("\n--- paired anchor-F1 difference (method - f5 baseline), 95% t-CI, Holm over {} tests ---", pc.len());
    let mut pcj = vec![];
    for (x, a) in pc.iter().zip(&adj) {
        let mk = if *a < 0.05 { if x.3 > 0.0 { " WIN" } else { " LOSS" } } else { "" };
        if x.2 == "L-f5tuned" {
            println!("{:<16} {:<10} - L-f5tuned: {:+.4} [{:+.4},{:+.4}] Holm {:.4}{}", x.0, x.1, x.3, x.4, x.5, a, mk);
        }
        pcj.push(json!({"world": x.0, "method": x.1, "vs": x.2, "diff": x.3, "ci": [x.4, x.5], "p": x.6, "holm_p": a}));
    }

    // ---- threshold sensitivity (held) ----
    println!("\n=== threshold SENSITIVITY of anchor pair-F1 (held) ===");
    let thr = [0.4f32, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7];
    let mut sens_methods: Vec<(String, Method)> = vec![];
    for &a in &thr {
        sens_methods.push((format!("leader k8 s{a}"), Method { anchor: AnchorCfg::Leader { k: 8, sim: a }, center: false }));
    }
    for &a in &thr {
        sens_methods.push((format!("centroid a{a}"), Method { anchor: AnchorCfg::Centroid { tau_a: a, tau_m: None, merge_min: 1, knn: None, avg: false }, center: false }));
    }
    for &a in &thr {
        sens_methods.push((format!("cen+merge a{a} m{:.2}", a + 0.1), Method { anchor: AnchorCfg::Centroid { tau_a: a, tau_m: Some(a + 0.1), merge_min: 3, knn: None, avg: false }, center: false }));
    }
    for &a in &thr {
        sens_methods.push((format!("avglink a{a} m{a}"), Method { anchor: AnchorCfg::Centroid { tau_a: a, tau_m: Some(a), merge_min: 1, knn: None, avg: true }, center: false }));
    }
    for &a in &thr {
        sens_methods.push((format!("bridge n{a} m{:.2} s{}", a + 0.1, 2), Method { anchor: AnchorCfg::Bridge { k: 8, tau_n: a, tau_m: a + 0.1, support: 2 }, center: false }));
    }
    let sm: Vec<Method> = sens_methods.iter().map(|x| x.1.clone()).collect();
    let sw = world_list(&["mini-q", "mini-para", "mini-q-mixed50", "mini-subj3"]);
    let sres = eval_methods(&sm, &sw, held);
    println!("{:<26} {:>9} {:>9} {:>9} {:>9}", "method (threshold)", "q", "para", "mixed50", "subj3");
    let mut sjs = vec![];
    for (mi, (n, _)) in sens_methods.iter().enumerate() {
        let f: Vec<f64> = (0..sw.len()).map(|wi| mean_of(sres[mi][wi].iter().map(|a| a.f1))).collect();
        println!("{:<26} {:>9.3} {:>9.3} {:>9.3} {:>9.3}", n, f[0], f[1], f[2], f[3]);
        sjs.push(json!({"method": n, "f1": {"q": f[0], "para": f[1], "mixed50": f[2], "subj3": f[3]}}));
    }
    for fam in ["leader", "centroid a", "cen+merge", "avglink", "bridge"] {
        let idx: Vec<usize> = sens_methods.iter().enumerate().filter(|(_, x)| x.0.starts_with(fam)).map(|(i, _)| i).collect();
        let spread: Vec<f64> = (0..sw.len())
            .map(|wi| {
                let v: Vec<f64> = idx.iter().map(|&mi| mean_of(sres[mi][wi].iter().map(|a| a.f1))).collect();
                v.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - v.iter().cloned().fold(f64::INFINITY, f64::min)
            })
            .collect();
        println!("  F1 spread over thresholds .40-.70 [{}]: q {:.3} para {:.3} mixed50 {:.3} subj3 {:.3}", fam, spread[0], spread[1], spread[2], spread[3]);
    }

    // ---- neighbour list (separate stage) ----
    println!("\n=== NEIGHBOUR list P/R/F1 (independent of the anchor threshold), held ===");
    println!("{:<14} {:>2} {:>4} | {:>6} {:>6} {:>6}", "world", "k", "thr", "P", "R", "F1");
    let mut nbj = vec![];
    for wn in ["mini-q", "mini-para", "mini-q2para", "hash-q", "hash-para"] {
        let sp = spec(wn);
        for &k in &[3usize, 5, 8] {
            for &thr in &[0.5f32, 0.6] {
                let from = if sp.corp == Corp::QThenPara { 1000 } else { 0 };
                let rs: Vec<Nbr> = held.par_iter().map(|&s| nbr_metrics(&CtxWorld::new(s, &sp), DIM, k, thr, false, from)).collect();
                let (p, r, f) = (mean_of(rs.iter().map(|x| x.p)), mean_of(rs.iter().map(|x| x.r)), mean_of(rs.iter().map(|x| x.f1)));
                println!("{:<14} {:>2} {:>4} | {:>6.3} {:>6.3} {:>6.3}{}", wn, k, thr, p, r, f, if from > 0 { "  (queries after the phrasing switch)" } else { "" });
                nbj.push(json!({"world": wn, "k": k, "thr": thr, "p": p, "r": r, "f1": f, "from_t": from}));
            }
        }
    }
    let doc = json!({"lab": "f6_anchor/retrieval", "held": [held[0], held[held.len() - 1]], "rows": rows, "paired_vs_f5": pcj, "sensitivity": sjs, "neighbour": nbj, "elapsed_s": t0.elapsed().as_secs_f64()});
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out} in {:.0}s", t0.elapsed().as_secs_f64());
    doc
}

fn downstream(sel: &Sel, held: &[u64], out: &str) {
    let t0 = Instant::now();
    let ns = held.len();
    let cols: Vec<String> = HORIZONS.iter().map(|h| format!("H{h}")).collect();
    let ex = |r: &RunOut| -> Vec<f64> { r.h.to_vec() };
    let mut doc = serde_json::Map::new();
    let mut tables = vec![];
    for (src, wnames) in [
        ("mini", vec!["mini-q", "mini-subj3", "mini-q-mild", "mini-q-mixed50", "mini-para", "mini-qpara", "mini-q2para", "mini-ctl-irrel", "mini-ctl-unique", "mini-ctl-none"]),
        ("hash", vec!["hash-q", "hash-para", "hash-ctl-irrel"]),
    ] {
        let specs = world_list(&wnames);
        for (reg, sg) in [("W", Signal::Reuse), ("O", Signal::Outcome)] {
            if src == "hash" && reg == "O" {
                continue;
            }
            let bases = if reg == "W" { bases_w() } else { bases_o() };
            let nb = bases.len();
            let mut arms = bases.clone();
            let vm = VoteCfg { min_n: 3.0, tau: 0.7, gate: Some((6.0, 0.02, 25)) };
            let cand_defs: Vec<(String, Kind2)> = if src == "mini" {
                vec![
                    ("kn:L-d6cfg".into(), Kind2::Kn(f5_d6_method())),
                    ("kn:L-f5tuned".into(), Kind2::Kn(f5_mini_method())),
                    ("kn:L-retuned".into(), Kind2::Kn(sel.leader.clone())),
                    ("kn:centroid".into(), Kind2::Kn(sel.cen.clone())),
                    ("kn:cen+merge".into(), Kind2::Kn(sel.cm.clone())),
                    ("kn:cen+knn".into(), Kind2::Kn(sel.un.clone())),
                    ("kn:avglink".into(), Kind2::Kn(sel.av.clone())),
                    ("kn:bridge".into(), Kind2::Kn(sel.br.clone())),
                    ("vote:L-d6cfg".into(), Kind2::KnVote(f5_d6_method(), d6_vote())),
                    ("vote:L-f5tuned".into(), Kind2::KnVote(f5_mini_vote().0, f5_mini_vote().1)),
                    ("vote:centroid".into(), Kind2::KnVote(sel.cen.clone(), vm.clone())),
                    ("vote:cen+merge".into(), Kind2::KnVote(sel.cm.clone(), vm.clone())),
                    ("vote:cen+knn".into(), Kind2::KnVote(sel.un.clone(), vm.clone())),
                    ("vote:avglink".into(), Kind2::KnVote(sel.av.clone(), vm.clone())),
                    ("kn:ORACLE".into(), Kind2::Kn(Method { anchor: AnchorCfg::Oracle, center: false })),
                    ("vote:ORACLE".into(), Kind2::KnVote(Method { anchor: AnchorCfg::Oracle, center: false }, vm.clone())),
                ]
            } else {
                vec![
                    ("kn:L-f5tuned".into(), Kind2::Kn(f5_hash_method())),
                    ("kn:cen+merge".into(), Kind2::Kn(sel.cm.clone())),
                    ("vote:L-f5tuned".into(), Kind2::KnVote(f5_hash_vote().0, f5_hash_vote().1)),
                    ("vote:cen+merge".into(), Kind2::KnVote(sel.cm.clone(), vm.clone())),
                ]
            };
            for (n, k) in &cand_defs {
                arms.push((n.clone(), (k.clone(), sg)));
            }
            let tag = format!("{src}-{reg}");
            println!("\n######## {tag}: {} worlds x {} arms x {} seeds", specs.len(), arms.len(), ns);
            let runs = run_all(&specs, &arms, held);
            print_table(&runs, &cols, &ex, &tag, &mut tables);
            let base_idx: Vec<usize> = (0..nb).collect();
            let cand_idx: Vec<usize> = (nb..arms.len()).collect();
            let cells = cells_vs_best(&runs, &base_idx, &cand_idx, &cols, &ex, true);
            let (rep, adj) = report(&format!("[{tag}] kn-feat vs best fair baseline"), &cells, ns, true);
            matrix(&format!("{tag} vs best fair baseline"), &cells, &adj, "H2000", true);
            matrix(&format!("{tag} vs best fair baseline"), &cells, &adj, "H500", true);
            doc.insert(format!("{tag}_vs_best"), rep);
            // must-not-hurt: vs the same guarded front
            let front = runs.names.iter().position(|n| n == if reg == "W" { "guarded(hebr)+reuse" } else { "guarded(hebr)" }).unwrap();
            let cells_f = cells_vs_fixed(&runs, front, &cand_idx, &cols, &ex);
            let (rep_f, adj_f) = report(&format!("[{tag}] kn-feat vs the same guarded(hebr) front (must-not-hurt)"), &cells_f, ns, true);
            matrix(&format!("{tag} vs same front"), &cells_f, &adj_f, "H2000", true);
            doc.insert(format!("{tag}_vs_front"), rep_f);
            // direct: new anchor methods vs f5's retuned leader (same source)
            if src == "mini" {
                let r5 = runs.names.iter().position(|n| n == "kn:L-f5tuned").unwrap();
                let new_idx: Vec<usize> = ["kn:L-retuned", "kn:centroid", "kn:cen+merge", "kn:cen+knn", "kn:avglink", "kn:bridge"].iter().filter_map(|n| runs.names.iter().position(|x| x == n)).collect();
                let vnew: Vec<usize> = ["vote:centroid", "vote:cen+merge", "vote:cen+knn", "vote:avglink"].iter().filter_map(|n| runs.names.iter().position(|x| x == n)).collect();
                let vref = runs.names.iter().position(|n| n == "vote:L-f5tuned").unwrap();
                let cells_v = cells_vs_fixed(&runs, vref, &vnew, &cols, &ex);
                let (rep_v, adj_v) = report(&format!("[{tag}] new anchors in the VOTE arm vs f5's best vote arm (leader k5 s.5 ctr, n3)"), &cells_v, ns, true);
                matrix(&format!("{tag} vote vs f5 vote"), &cells_v, &adj_v, "H2000", true);
                matrix(&format!("{tag} vote vs f5 vote"), &cells_v, &adj_v, "H500", true);
                doc.insert(format!("{tag}_vote_vs_f5vote"), rep_v);
                let cells_d = cells_vs_fixed(&runs, r5, &new_idx, &cols, &ex);
                let (rep_d, adj_d) = report(&format!("[{tag}] new anchor methods vs f5's best kn-feat (leader k8 s.5)"), &cells_d, ns, true);
                matrix(&format!("{tag} vs f5 leader(k8,.5)"), &cells_d, &adj_d, "H2000", true);
                matrix(&format!("{tag} vs f5 leader(k8,.5)"), &cells_d, &adj_d, "H500", true);
                doc.insert(format!("{tag}_vs_f5leader"), rep_d);
                let r6 = runs.names.iter().position(|n| n == "kn:L-d6cfg").unwrap();
                let cells_d6 = cells_vs_fixed(&runs, r6, &new_idx, &cols, &ex);
                let (rep_d6, adj_d6) = report(&format!("[{tag}] new anchor methods vs f5's d6-config kn-feat (leader k8 s.6)"), &cells_d6, ns, true);
                matrix(&format!("{tag} vs f5 d6cfg leader(k8,.6)"), &cells_d6, &adj_d6, "H2000", true);
                doc.insert(format!("{tag}_vs_d6leader"), rep_d6);
            }
        }
    }
    // reproduction check of the downstream pipeline vs published f5 numbers (mini-q, W, d6cfg, H500/H2000)
    if let Ok(s) = std::fs::read_to_string("results/f5_knowledge.json") {
        let f5: Value = serde_json::from_str(&s).unwrap();
        if held[0] == f5["held"][0].as_u64().unwrap() {
            let sp = spec("mini-q");
            let r = eval(&(Kind2::Kn(f5_d6_method()), Signal::Reuse), &sp, held);
            let m = [col(&r, 2), col(&r, 3)].map(|c| c.iter().sum::<f64>() / c.len() as f64);
            let cells = f5["mini_regime_W"]["cells"].as_array().unwrap();
            let get = |h: u64| cells.iter().find(|c| c["world"] == "mini-q" && c["candidate"] == "kn-feat(d6cfg)" && c["H"] == h).map(|c| c["cand_mean"].as_f64().unwrap());
            println!("\nREPRO check mini-q W kn-feat(d6cfg): mine H500 {:.9} H2000 {:.9} | f5 json H500 {:?} H2000 {:?}", m[0], m[1], get(500), get(2000));
            doc.insert("repro_vs_f5".into(), json!({"mine_h500": m[0], "mine_h2000": m[1], "f5_h500": get(500), "f5_h2000": get(2000)}));
            let r = eval(&(Kind2::KnVote(f5_d6_method(), d6_vote()), Signal::Outcome), &sp, held);
            let m = [col(&r, 2), col(&r, 3)].map(|c| c.iter().sum::<f64>() / c.len() as f64);
            let cells = f5["mini_regime_O"]["cells"].as_array().unwrap();
            let get = |h: u64| cells.iter().find(|c| c["world"] == "mini-q" && c["candidate"] == "kn-d+vote(d6cfg)" && c["H"] == h).map(|c| c["cand_mean"].as_f64().unwrap());
            println!("REPRO check mini-q O kn-d+vote(d6cfg): mine H500 {:.9} H2000 {:.9} | f5 json H500 {:?} H2000 {:?}", m[0], m[1], get(500), get(2000));
            doc.insert("repro_vote_vs_f5".into(), json!({"mine_h500": m[0], "mine_h2000": m[1], "f5_h500": get(500), "f5_h2000": get(2000)}));
        }
    }
    doc.insert("lab".into(), json!("f6_anchor/downstream"));
    doc.insert("held".into(), json!([held[0], held[held.len() - 1]]));
    doc.insert("selected".into(), json!({"leader": method_json(&sel.leader), "centroid": method_json(&sel.cen), "centroid_merge": method_json(&sel.cm), "union": method_json(&sel.un), "avglink": method_json(&sel.av), "bridge": method_json(&sel.br)}));
    doc.insert("tables".into(), json!(tables));
    doc.insert("elapsed_s".into(), json!(t0.elapsed().as_secs_f64()));
    std::fs::write(out, serde_json::to_string_pretty(&Value::Object(doc)).unwrap()).unwrap();
    println!("wrote {out} in {:.0}s", t0.elapsed().as_secs_f64());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str, d: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned().unwrap_or(d.into());
    let part = arg("--part", "all");
    let held = parse_range(&args, "--held", (31, 60));
    let confirm = held[0] == 61;
    let sel_path = arg("--selected", "results/f6_anchor_selected.json");
    let (dr, dd) = if confirm { ("results/f6_anchor_retrieval_confirm61-90.json", "results/f6_anchor_downstream_confirm61-90.json") } else { ("results/f6_anchor_retrieval.json", "results/f6_anchor_downstream.json") };
    println!("f6_anchor: part {part}, held {}-{}, real-embed feature {}, model sha {}", held[0], held[held.len() - 1], real_embed_feature(), &corpus().model_sha[..12.min(corpus().model_sha.len())]);
    if part == "tune" || (part == "all" && !confirm) {
        tune(&sel_path);
    }
    if part == "tune" {
        return;
    }
    let sel = load_sel(&sel_path);
    if part == "retrieval" || part == "all" {
        retrieval(&sel, &held, &arg("--out-r", dr));
    }
    if part == "downstream" || part == "all" {
        downstream(&sel, &held, &arg("--out-d", dd));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    /// two well separated clusters (4-d), interleaved: every anchor method must found exactly 2 anchors that respect the clusters
    #[test]
    fn anchorers_separate_two_clusters_and_merge_fragments() {
        let a = [unit(&[1.0, 0.1, 0.0, 0.0]), unit(&[1.0, 0.0, 0.1, 0.0]), unit(&[1.0, 0.05, 0.05, 0.0]), unit(&[0.9, 0.1, 0.1, 0.1])];
        let b = [unit(&[0.0, 0.0, 1.0, 0.1]), unit(&[0.0, 0.1, 1.0, 0.0]), unit(&[0.1, 0.0, 1.0, 0.05]), unit(&[0.0, 0.1, 0.9, 0.2])];
        let methods = [
            AnchorCfg::Leader { k: 5, sim: 0.6 },
            AnchorCfg::Centroid { tau_a: 0.6, tau_m: None, merge_min: 1, knn: None, avg: false },
            AnchorCfg::Centroid { tau_a: 0.6, tau_m: Some(0.8), merge_min: 1, knn: None, avg: true },
            AnchorCfg::Bridge { k: 5, tau_n: 0.6, tau_m: 0.8, support: 2 },
        ];
        for m in methods {
            let mut an = make_anchorer(&m, 4);
            let mut ids = vec![];
            for i in 0..4 {
                ids.push((0, an.assign(&a[i]).0));
                ids.push((1, an.assign(&b[i]).0));
            }
            let (a_ids, b_ids): (Vec<u64>, Vec<u64>) = (ids.iter().filter(|x| x.0 == 0).map(|x| an.root(x.1)).collect(), ids.iter().filter(|x| x.0 == 1).map(|x| an.root(x.1)).collect());
            assert!(a_ids.iter().all(|&x| x == a_ids[0]), "{m:?}: cluster a split {a_ids:?}");
            assert!(b_ids.iter().all(|&x| x == b_ids[0]), "{m:?}: cluster b split {b_ids:?}");
            assert_ne!(a_ids[0], b_ids[0], "{m:?}: clusters merged");
        }
        // a lone outlier founds a NEW anchor (is_new = true) and a repeat of it does not
        let mut an = make_anchorer(&AnchorCfg::Centroid { tau_a: 0.6, tau_m: Some(0.8), merge_min: 1, knn: None, avg: true }, 4);
        assert!(an.assign(&a[0]).1);
        assert!(!an.assign(&a[1]).1);
        assert!(an.assign(&b[0]).1);
    }

    /// centroid + merge repairs an early fragment: two seeds founded apart (cos < tau_a) are bridged by a later query and merged
    #[test]
    fn merge_repairs_fragmentation() {
        let s1 = unit(&[1.0, 0.0, 0.0]);
        let s2 = unit(&[0.5, 0.87, 0.0]); // cos(s1,s2) = 0.5 < 0.6: two anchors
        let bridge = unit(&[0.75, 0.43, 0.0]);
        let mut plain = make_anchorer(&AnchorCfg::Centroid { tau_a: 0.6, tau_m: None, merge_min: 1, knn: None, avg: false }, 3);
        let mut merged = make_anchorer(&AnchorCfg::Centroid { tau_a: 0.6, tau_m: Some(0.65), merge_min: 1, knn: None, avg: false }, 3);
        let (p1, p2, m1, m2) = (plain.assign(&s1).0, plain.assign(&s2).0, merged.assign(&s1).0, merged.assign(&s2).0);
        assert_ne!(p1, p2);
        assert_ne!(m1, m2);
        plain.assign(&bridge);
        merged.assign(&bridge);
        assert_ne!(plain.root(p1), plain.root(p2), "no merge without tau_m");
        assert_eq!(merged.root(m1), merged.root(m2), "merge joins the two fragments");
    }

    #[test]
    fn oracle_anchor_is_the_truth() {
        let mut an = make_anchorer_t(&AnchorCfg::Oracle, 3, &[2, 2, -1, 5, 2]);
        let r: Vec<(u64, bool)> = (0..5).map(|_| an.assign(&[0.0, 0.0, 0.0])).collect();
        assert_eq!(r[0], (2, true));
        assert_eq!(r[1], (2, false));
        assert!(r[2].1 && r[2].0 >= 1_000_000);
        assert_eq!(r[3], (5, true));
        assert_eq!(r[4], (2, false));
    }
}
