//! H14 (R6-B): library `neural_mesh::dedup::Deduper` reproduces the H6 "exact-duplicate suppression" effect on doc-level F1@k.
//!
//!   h14_dedup [--out results/h14_dedup.json]
//!
//! Same setup as `h6_docq ctx`: g1 corpora A and B (templated, SYNTHETIC), MiniLM + spell-corrected queries, store = all docs + a 30 %
//! sample re-ingested as exact duplicates (same Rng seed as h6, so the stream is identical). Differences from h6: h6 removed the
//! duplicates using the labels it attached (oracle); here the LIBRARY decides, from text hash / vector hash only, which rows to keep.
//! Also reports the number of rows the library keeps vs the oracle, text-only / vector-only variants, and compares against
//! results/h6_ctx.json (F1 at fixed k, dedup and no-dedup). Units = per-concept means (240 concepts), paired CI over concepts.

use neural_mesh::dedup::{DedupConfig, Deduper, Verdict};
use neural_mesh::router::Rng;
use neural_mesh::stats::summarize;
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
use common::*;

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

/// per-concept F1@k over the rows in `keep` (mirrors h6_docq::mode_ctx for Pol::Fixed(k))
fn f1_at_k(corpus: &Corpus, owner: &[usize], keep: &[bool], sc: &[f32], ns: usize, concept: &[i32], k: usize) -> (Vec<f64>, f64, Vec<f64>) {
    let (mut f1v, mut chars, mut recv) = (vec![], 0.0, vec![]);
    let mut cnt = 0.0;
    for (i, &ci) in concept.iter().enumerate() {
        if ci < 0 {
            continue;
        }
        let mut rank: Vec<(f32, usize)> = (0..ns).filter(|&j| keep[j]).map(|j| (sc[i * ns + j], j)).collect();
        rank.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
        rank.truncate(12);
        let sel: Vec<usize> = rank.iter().take(k).map(|x| x.1).collect();
        let rel = |j: usize| corpus.docs[owner[j]].concept == ci;
        let hits = sel.iter().filter(|&&j| rel(j)).count() as f64;
        let dh: std::collections::HashSet<usize> = sel.iter().filter(|&&j| rel(j)).map(|&j| owner[j]).collect();
        let pr = hits / sel.len() as f64;
        let r = dh.len() as f64 / 2.0;
        recv.push(r);
        f1v.push(if pr + r > 0.0 { 2.0 * pr * r / (pr + r) } else { 0.0 });
        chars += sel.iter().map(|&j| corpus.docs[owner[j]].text.len()).sum::<usize>() as f64;
        cnt += 1.0;
    }
    let qc: Vec<i32> = concept.iter().cloned().filter(|&c| c >= 0).collect();
    (agg_concept(&f1v, &qc, corpus.nconc), chars / cnt, agg_concept(&recv, &qc, corpus.nconc))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = arg(&args, "--out", "results/h14_dedup.json");
    let h6: Option<Value> = std::fs::read_to_string("results/h6_ctx.json").ok().and_then(|t| serde_json::from_str(&t).ok());
    let mut report: Vec<Value> = vec![];
    let mut max_dev = 0.0f64;
    let mut ncmp = 0;
    println!("h14 dedup: library Deduper on the h6 'ctx' store (30% exact re-ingested duplicates), MiniLM+spell, per-concept means. SYNTHETIC-until-validated.");
    for tag in ["A", "B"] {
        let corpus = load_corpus(tag);
        let m = load_model(&corpus, "mini");
        let nq = corpus.queries.len();
        let qs: Vec<f32> = (0..nq).flat_map(|i| m.qvec(QV::Spell, i).to_vec()).collect();
        let concept: Vec<i32> = corpus.queries.iter().map(|q| q.concept).collect();
        let nd = corpus.docs.len();
        // identical stream to h6_docq::mode_ctx
        let mut rng = Rng::new(0x6C7_0001);
        let mut owner: Vec<usize> = (0..nd).collect();
        for i in 0..nd {
            if rng.f64() < 0.3 {
                owner.push(i);
            }
        }
        let ns = owner.len();
        let d = m.dim;
        // scores: f32 dot (quant_matrix(F32) in h6 = sequential f32 dot, same as `dot` here)
        let mut sc = vec![0f32; nq * ns];
        for i in 0..nq {
            for j in 0..ns {
                sc[i * ns + j] = dot(&qs[i * d..(i + 1) * d], m.dvec(owner[j]));
            }
        }
        let oracle_keep: Vec<bool> = (0..ns).map(|j| j < nd).collect();
        let all_keep = vec![true; ns];
        let variants: Vec<(&str, DedupConfig)> = vec![
            ("text+vector (default)", DedupConfig::default()),
            ("text only", DedupConfig { by_vector: false, ..DedupConfig::default() }),
            ("vector only", DedupConfig { by_text: false, ..DedupConfig::default() }),
        ];
        let qc: Vec<i32> = concept.iter().cloned().filter(|&c| c >= 0).collect();
        let _ = qc;
        for (vn, cfg) in variants {
            let mut dd = Deduper::new(cfg);
            let keep: Vec<bool> = (0..ns).map(|j| dd.check(j as u64, Some(&corpus.docs[owner[j]].text), Some(m.dvec(owner[j]))) == Verdict::New).collect();
            let kept = keep.iter().filter(|&&k| k).count();
            let diff_vs_oracle = (0..ns).filter(|&j| keep[j] != oracle_keep[j]).count();
            let st = dd.stats();
            println!("\n[{tag}] {vn}: rows {ns} -> {kept} (oracle {nd}); rows differing from the h6 label-oracle: {diff_vs_oracle}; suppressed by text {} / by vector {}", st.suppressed_text, st.suppressed_vector);
            for k in 1..=4usize {
                let (f_no, ch_no, r_no) = f1_at_k(&corpus, &owner, &all_keep, &sc, ns, &concept, k);
                let (f_lib, ch_lib, r_lib) = f1_at_k(&corpus, &owner, &keep, &sc, ns, &concept, k);
                let (f_orc, _, _) = f1_at_k(&corpus, &owner, &oracle_keep, &sc, ns, &concept, k);
                let diff: Vec<f64> = f_lib.iter().zip(&f_no).map(|(a, b)| a - b).collect();
                let s = summarize(&diff);
                let (mn, ml, mo) = (summarize(&f_no).mean, summarize(&f_lib).mean, summarize(&f_orc).mean);
                println!("   k={k}: F1 no-dedup {mn:.4} -> library {ml:.4} ({:+.4}, 95% CI [{:+.4},{:+.4}]) | h6 label-oracle {mo:.4} | distinct-relevant-doc recall {:.4} -> {:.4} | context chars {ch_no:.0} -> {ch_lib:.0}", s.mean, s.lo, s.hi, summarize(&r_no).mean, summarize(&r_lib).mean);
                let mut refrow = Value::Null;
                if let Some(h) = &h6 {
                    let find = |dedup: bool| h["rows"].as_array().and_then(|r| r.iter().find(|x| x["corpus"] == json!(tag) && x["dedup"] == json!(dedup) && x["policy"] == json!(format!("fixed k={k}"))).map(|x| x["F1"].as_f64().unwrap()));
                    if let (Some(a), Some(b)) = (find(false), find(true)) {
                        let dv = (a - mn).abs().max((b - mo).abs());
                        max_dev = max_dev.max(dv);
                        ncmp += 2;
                        refrow = json!({"h6_no_dedup_F1": a, "h6_oracle_dedup_F1": b, "abs_dev_vs_ours": dv});
                    }
                }
                report.push(json!({"corpus": tag, "variant": vn, "k": k, "rows_before": ns, "rows_kept": kept, "rows_oracle": nd, "rows_differing_from_oracle": diff_vs_oracle,
                    "F1_no_dedup": mn, "F1_library": ml, "F1_h6_oracle": mo, "distinct_recall_no_dedup": summarize(&r_no).mean, "distinct_recall_library": summarize(&r_lib).mean, "diff": s.mean, "ci": [s.lo, s.hi], "chars_no_dedup": ch_no, "chars_library": ch_lib, "h6_reference": refrow}));
            }
        }
    }
    println!("\nreproduction vs results/h6_ctx.json: {ncmp} F1 values compared (no-dedup F1 and label-oracle dedup F1 recomputed here), max |dev| = {max_dev:.3e}");
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"note": "SYNTHETIC-until-validated-on-real-traffic. Library Deduper vs h6 ctx exact-duplicate suppression (R6-B/H14).", "max_dev_vs_h6_ctx_json": max_dev, "n_compared": ncmp, "rows": report})).unwrap()).unwrap();
    println!("wrote {out}");
}
