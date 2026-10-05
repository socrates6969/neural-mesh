//! H6-docq (R4-B3): document/chunk-level retrieval with quantised stores (recall@k vs f32, F1@3, hit@1, MRR, TOST +-0.01 per-concept paired),
//! and the retrieved-context bytes frontier (fixed k / adaptive k / byte budget / exact-dup suppression).
//!   h6_docq docq [--out results]
//!   h6_docq ctx  [--out results]
//! Real MiniLM/BGE fixture embeddings of the SYNTHETIC templated g1 corpora A and B (780 docs, 2120 queries; 240 concepts, 2 relevant docs each).
//! Synthetic-until-validated-on-real-traffic. No defaults changed.

use neural_mesh::router::Rng;
use neural_mesh::stats::summarize;
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
#[path = "g16_common/mod.rs"]
mod g16;
#[path = "h6_common/mod.rs"]
mod h6;
use common::*;
use g16::{cmp, holm};
use h6::*;

fn top_idx(row: &[f32], n: usize) -> Vec<usize> {
    let mut v: Vec<(f32, usize)> = row.iter().cloned().zip(0..).collect();
    if v.len() > n {
        v.select_nth_unstable_by(n, |a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        v.truncate(n);
    }
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    v.into_iter().map(|x| x.1).collect()
}
fn rerank_list(approx: &[f32], exact: &[f32], n: usize, f: usize) -> Vec<usize> {
    let cand = top_idx(approx, n * f);
    let mut v: Vec<(f32, usize)> = cand.into_iter().map(|j| (exact[j], j)).collect();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
    v.truncate(n);
    v.into_iter().map(|x| x.1).collect()
}
/// per-concept mean of a per-query vector (queries with concept >= 0)
fn agg_concept(vals: &[f64], concept: &[i32], nconc: usize) -> Vec<f64> {
    let mut s = vec![0.0; nconc];
    let mut n = vec![0.0; nconc];
    for (v, &c) in vals.iter().zip(concept) {
        if c >= 0 {
            s[c as usize] += v;
            n[c as usize] += 1.0;
        }
    }
    (0..nconc).filter(|&k| n[k] > 0.0).map(|k| s[k] / n[k]).collect()
}
const DM: [&str; 7] = ["recall@1", "recall@3", "recall@10", "P@3", "F1@3", "hit@1", "MRR@10"];

fn mode_docq(out_dir: &str) {
    let mut report: Vec<Value> = vec![];
    for tag in ["A", "B"] {
        let corpus = load_corpus(tag);
        for model in ["mini", "bge"] {
            let (m, qv) = if model == "mini" { (load_model(&corpus, "mini"), QV::Spell) } else { (load_model(&corpus, "bgei"), QV::Raw) };
            let dm = load_model(&corpus, model);
            let d = m.dim;
            let nq = corpus.queries.len();
            let qs: Vec<f32> = (0..nq).flat_map(|i| m.qvec(qv, i).to_vec()).collect();
            let concept: Vec<i32> = corpus.queries.iter().map(|q| q.concept).collect();
            let stores: Vec<(&str, &Vec<f32>, Vec<u32>)> = vec![("docs", &dm.doc, (0..corpus.docs.len() as u32).collect()), ("chunks", &dm.chunk, dm.chunk_doc.clone())];
            for (store_name, store, owner) in stores {
                let ns = store.len() / d;
                let doc_conc: Vec<i32> = owner.iter().map(|&o| corpus.docs[o as usize].concept).collect();
                let exact = quant_matrix(Q::F32, &qs, store, d);
                let mut mats: Vec<(String, Vec<f32>, usize)> = vec![("f32".into(), exact.clone(), Q::F32.bytes(d))];
                for q in [Q::F16, Q::BlockFp16, Q::Int8, Q::Int8QQ] {
                    mats.push((q.name().into(), quant_matrix(q, &qs, store, d), q.bytes(d)));
                }
                for b in [4usize, 3, 2] {
                    mats.push((format!("turbovec-{b}bit"), tv_matrix(b, &qs, store, d), tv_bytes(b, d)));
                }
                let t4 = mats.iter().position(|x| x.0 == "turbovec-4bit").unwrap();
                let t2 = mats.iter().position(|x| x.0 == "turbovec-2bit").unwrap();
                let mut names: Vec<String> = mats.iter().map(|x| x.0.clone()).collect();
                let mut lists: Vec<Vec<Vec<usize>>> = mats.iter().map(|(_, mt, _)| (0..nq).map(|i| top_idx(&mt[i * ns..(i + 1) * ns], 10)).collect()).collect();
                let mut bytes: Vec<usize> = mats.iter().map(|x| x.2).collect();
                let mut exact_b: Vec<usize> = vec![0; mats.len()];
                for (n, ai, f) in [("turbovec-4bit + f32 rerank 4x", t4, 4usize), ("turbovec-2bit + f32 rerank 4x", t2, 4), ("turbovec-2bit + f32 rerank 16x", t2, 16)] {
                    names.push(n.into());
                    lists.push((0..nq).map(|i| rerank_list(&mats[ai].1[i * ns..(i + 1) * ns], &exact[i * ns..(i + 1) * ns], 10, f)).collect());
                    bytes.push(mats[ai].2);
                    exact_b.push(4 * d);
                }
                let metrics = |li: &Vec<Vec<usize>>| -> Vec<[f64; 7]> {
                    (0..nq)
                        .map(|i| {
                            let ex = &lists[0][i];
                            let l = &li[i];
                            let rec = |k: usize| {
                                let kk = k.min(ex.len());
                                if kk == 0 { 1.0 } else { l.iter().take(k).filter(|x| ex[..kk].contains(x)).count() as f64 / kk as f64 }
                            };
                            let c = concept[i];
                            let rel = |j: usize| c >= 0 && doc_conc[j] == c;
                            let top3: Vec<usize> = l.iter().take(3).cloned().collect();
                            let p3 = top3.iter().filter(|&&j| rel(j)).count() as f64 / 3.0;
                            let f1 = if store_name == "docs" {
                                let dh: std::collections::HashSet<u32> = top3.iter().filter(|&&j| rel(j)).map(|&j| owner[j]).collect();
                                let r = dh.len() as f64 / 2.0;
                                if p3 + r > 0.0 { 2.0 * p3 * r / (p3 + r) } else { 0.0 }
                            } else {
                                p3
                            };
                            let hit1 = if rel(l[0]) { 1.0 } else { 0.0 };
                            let mrr = l.iter().position(|&j| rel(j)).map(|p| 1.0 / (p + 1) as f64).unwrap_or(0.0);
                            [rec(1), rec(3), rec(10), p3, f1, hit1, mrr]
                        })
                        .collect()
                };
                let base = metrics(&lists[0]);
                let base_c: Vec<Vec<f64>> = (0..7).map(|mi| agg_concept(&base.iter().map(|r| r[mi]).collect::<Vec<_>>(), &concept, corpus.nconc)).collect();
                let mut tps: Vec<f64> = vec![];
                let mut rows_json: Vec<Value> = vec![];
                for (vi, name) in names.iter().enumerate() {
                    let me = metrics(&lists[vi]);
                    let mut j = json!({"corpus": tag, "model": model, "store": store_name, "n_store": ns, "variant": name, "bytes_per_vec": bytes[vi], "exact_copy_bytes_per_vec": exact_b[vi], "dim": d});
                    for mi in 0..7 {
                        let vc = agg_concept(&me.iter().map(|r| r[mi]).collect::<Vec<_>>(), &concept, corpus.nconc);
                        let c = cmp(&vc, &base_c[mi], 0.01);
                        j[DM[mi]] = json!({"mean": summarize(&vc).mean, "diff": c.diff, "ci": [c.lo, c.hi], "tost_p": c.tost_p});
                        if vi > 0 && (mi == 4 || mi == 5 || mi == 2) {
                            tps.push(c.tost_p);
                        }
                    }
                    rows_json.push(j);
                }
                let adj = holm(&tps);
                // NOTE: tps was pushed in metric order mi = 2 (recall@10), 4 (F1), 5 (hit) -> fix the label order
                let mut ti = 0;
                for (vi, j) in rows_json.iter_mut().enumerate() {
                    if vi > 0 {
                        for n in ["recall@10", "F1@3", "hit@1"] {
                            j[n]["tost_p_holm"] = json!(adj[ti]);
                            ti += 1;
                        }
                    }
                }
                for j in &rows_json {
                    let g = |a: &str, b: &str| j[a][b].as_f64().unwrap_or(f64::NAN);
                    println!(
                        "{tag} {model:<4} {store_name:<6} {:<32} B/vec {:>4}(+{:>4}f32) rec@1 {:.3} rec@3 {:.3} rec@10 {:.3} | F1@3 {:.3} ({:+.4}) hit@1 {:.3} ({:+.4}) MRR {:.3} | TOSTholm F1 {:.1e} hit {:.1e} rec10 {:.1e}",
                        j["variant"].as_str().unwrap(), j["bytes_per_vec"], j["exact_copy_bytes_per_vec"], g("recall@1", "mean"), g("recall@3", "mean"), g("recall@10", "mean"), g("F1@3", "mean"), g("F1@3", "diff"), g("hit@1", "mean"), g("hit@1", "diff"), g("MRR@10", "mean"),
                        g("F1@3", "tost_p_holm"), g("hit@1", "tost_p_holm"), g("recall@10", "tost_p_holm")
                    );
                }
                report.extend(rows_json);
            }
        }
    }
    let path = format!("{out_dir}/h6_docq.json");
    std::fs::write(&path, serde_json::to_string_pretty(&json!({"margin": 0.01, "unit": "per-concept means (240 concepts), paired vs f32; Holm over (variant x {recall@10,F1@3,hit@1}) per corpus/model/store", "rows": report})).unwrap()).unwrap();
    println!("wrote {path}");
}

// ================================================================================================
// ctx: injected-context bytes frontier
#[derive(Clone, Debug)]
enum Pol {
    Fixed(usize),
    Drop(f32, usize),
    Budget(usize),
}
impl Pol {
    fn name(&self) -> String {
        match self {
            Pol::Fixed(k) => format!("fixed k={k}"),
            Pol::Drop(d, k) => format!("adaptive drop {d:.2} (kmax {k})"),
            Pol::Budget(b) => format!("byte budget {b} chars"),
        }
    }
}
/// returns indices into the ranked list
fn select(p: &Pol, rank: &[(f32, usize)], chars: &dyn Fn(usize) -> usize) -> Vec<usize> {
    match p {
        Pol::Fixed(k) => rank.iter().take(*k).map(|x| x.1).collect(),
        Pol::Drop(d, kmax) => {
            let top = rank[0].0;
            rank.iter().take(*kmax).filter(|x| x.0 >= top - d).map(|x| x.1).collect()
        }
        Pol::Budget(b) => {
            let mut out = vec![];
            let mut used = 0;
            for x in rank.iter().take(8) {
                let c = chars(x.1);
                if !out.is_empty() && used + c > *b {
                    break;
                }
                used += c;
                out.push(x.1);
            }
            out
        }
    }
}

fn mode_ctx(out_dir: &str) {
    let mut report: Vec<Value> = vec![];
    for tag in ["A", "B"] {
        let corpus = load_corpus(tag);
        let m = load_model(&corpus, "mini");
        let d = m.dim;
        let nq = corpus.queries.len();
        let qs: Vec<f32> = (0..nq).flat_map(|i| m.qvec(QV::Spell, i).to_vec()).collect();
        let concept: Vec<i32> = corpus.queries.iter().map(|q| q.concept).collect();
        let nd = corpus.docs.len();
        let dchars: Vec<usize> = corpus.docs.iter().map(|x| x.text.len()).collect();
        // ----- store with 30% exact re-ingested duplicates (labels I attach; no new text embedded)
        let mut rng = Rng::new(0x6C7_0001);
        let mut owner: Vec<usize> = (0..nd).collect();
        for i in 0..nd {
            if rng.f64() < 0.3 {
                owner.push(i);
            }
        }
        let store: Vec<f32> = owner.iter().flat_map(|&o| m.dvec(o).to_vec()).collect();
        let ns = owner.len();
        let sc = quant_matrix(Q::F32, &qs, &store, d);
        let is_dup: Vec<bool> = (0..ns).map(|j| j >= nd).collect();
        let pols: Vec<Pol> = {
            let mut v: Vec<Pol> = (1..=6).map(Pol::Fixed).collect();
            for dd in [0.02f32, 0.05, 0.1, 0.2] {
                v.push(Pol::Drop(dd, 6));
            }
            for b in [150usize, 250, 400, 600, 900, 1400] {
                v.push(Pol::Budget(b));
            }
            v
        };
        for dedup in [false, true] {
            let mut rows: Vec<(Pol, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = vec![];
            for p in &pols {
                let (mut f1v, mut anyv, mut chv, mut p_v) = (vec![], vec![], vec![], vec![]);
                for i in 0..nq {
                    if concept[i] < 0 {
                        continue;
                    }
                    let mut rank: Vec<(f32, usize)> = (0..ns).filter(|&j| !(dedup && is_dup[j])).map(|j| (sc[i * ns + j], j)).collect();
                    rank.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                    rank.truncate(12);
                    let chars = |j: usize| dchars[owner[j]];
                    let sel = select(p, &rank, &chars);
                    let rel = |j: usize| corpus.docs[owner[j]].concept == concept[i];
                    let hits = sel.iter().filter(|&&j| rel(j)).count() as f64;
                    let dh: std::collections::HashSet<usize> = sel.iter().filter(|&&j| rel(j)).map(|&j| owner[j]).collect();
                    let pr = hits / sel.len() as f64;
                    let r = dh.len() as f64 / 2.0;
                    f1v.push(if pr + r > 0.0 { 2.0 * pr * r / (pr + r) } else { 0.0 });
                    anyv.push(if hits > 0.0 { 1.0 } else { 0.0 });
                    p_v.push(pr);
                    chv.push(sel.iter().map(|&j| chars(j)).sum::<usize>() as f64);
                }
                rows.push((p.clone(), f1v, anyv, chv, p_v));
            }
            let qc: Vec<i32> = concept.iter().cloned().filter(|&c| c >= 0).collect();
            // best F1 policy = reference for the knee test
            let aggs: Vec<(Vec<f64>, Vec<f64>, f64)> = rows.iter().map(|r| (agg_concept(&r.1, &qc, corpus.nconc), agg_concept(&r.2, &qc, corpus.nconc), summarize(&r.3).mean)).collect();
            let best = (0..rows.len()).max_by(|&a, &b| summarize(&aggs[a].0).mean.partial_cmp(&summarize(&aggs[b].0).mean).unwrap()).unwrap();
            for (i, r) in rows.iter().enumerate() {
                let c = cmp(&aggs[i].0, &aggs[best].0, 0.01);
                let ch = aggs[i].2;
                let tail = if dedup { "dedup" } else { "no-dedup" };
                println!("{tag} {tail:<8} {:<30} chars {:>7.1}  F1 {:.4}  hit_any {:.4}  P {:.4}  | vs best-F1 ({}): {:+.4} [{:+.4},{:+.4}] TOSTp {:.2e}", r.0.name(), ch, summarize(&aggs[i].0).mean, summarize(&aggs[i].1).mean, summarize(&agg_concept(&r.4, &qc, corpus.nconc)).mean, rows[best].0.name(), c.diff, c.lo, c.hi, c.tost_p);
                report.push(json!({"corpus": tag, "dedup": dedup, "policy": r.0.name(), "chars": ch, "F1": summarize(&aggs[i].0).mean, "hit_any": summarize(&aggs[i].1).mean, "P": summarize(&agg_concept(&r.4, &qc, corpus.nconc)).mean, "best": rows[best].0.name(), "diff_vs_best_F1": c.diff, "ci": [c.lo, c.hi], "tost_p_vs_best": c.tost_p, "rows_in_store": if dedup { nd } else { ns }}));
            }
        }
    }
    let path = format!("{out_dir}/h6_ctx.json");
    std::fs::write(&path, serde_json::to_string_pretty(&json!({"note": "mini+spell dense retrieval, 30% exact duplicate re-ingestion; relevant = the 2 docs of the query's concept; units = per-concept means", "rows": report})).unwrap()).unwrap();
    println!("wrote {path}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = arg(&args, "--out", "results");
    match args.get(1).map(|s| s.as_str()) {
        Some("docq") => mode_docq(&out),
        Some("ctx") => mode_ctx(&out),
        _ => eprintln!("usage: h6_docq docq|ctx [--out results]"),
    }
}
