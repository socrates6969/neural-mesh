//! H2-retrieval (R4-A2): DOCUMENT-level retrieval (query -> its concept's 2 docs among 780) for the h2 variants: dense+spell, BM25, the g1 tuned-alpha z-hybrid,
//! Reciprocal Rank Fusion (k=60), Relative Score Fusion, and the cheap top-20 rerankers (token overlap / BM25 / small logistic on features).
//! Reports P@3/R@3/F1@3/hit@1/MRR (paired, Holm), bytes, measured latency incl. rerank-scorer cost per call. Retrieval gains are NOT the headline: h2_route.rs judges end-to-end.
//!
//! Usage: cargo run --release --offline --bin h2_retrieval -- --corpus A            (tune on tune concepts (id%5==0); held-out concepts, seeds 31-60 -> results/h2_retrieval.json, h2_retrieval_picks.json)
//!        cargo run --release --offline --bin h2_retrieval -- --corpus B|C2|D2 --confirm   (frozen picks, seeds 61-90 -> results/h2_retrieval_confirm_<tag>.json)

use std::time::Instant;

use neural_mesh::router::Rng;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue};
use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
#[path = "h2_common/mod.rs"]
mod h2;
use common::*;
use h2::{fit_logit, load_corpus_p, load_mini_p, sigmoid};

const DEPTH: usize = 100;
const M: usize = 20;
const NFD: usize = 6;

struct DM {
    c: Corpus,
    lex: Lex,
    nq: usize,
    nd: usize,
    d: Vec<f32>,
    b: Vec<f32>,
    o: Vec<f32>,
    qt: Vec<Vec<u32>>,
    mini: Model,
}
fn build_dm(tag: &str) -> DM {
    let c = load_corpus_p(tag);
    let lex = Lex::build(&c.docs);
    let mini = load_mini_p(&c);
    let (nq, nd) = (c.queries.len(), c.docs.len());
    let qt: Vec<Vec<u32>> = c.queries.iter().map(|q| lex.q_terms(&q.spell)).collect();
    let rows: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = (0..nq)
        .into_par_iter()
        .map(|i| {
            let q = mini.qvec(QV::Spell, i);
            let d: Vec<f32> = (0..nd).map(|j| dot(q, mini.dvec(j))).collect();
            let b = lex.bm25(&c.queries[i].spell);
            let o: Vec<f32> = (0..nd).map(|j| lex.overlap(&qt[i], j)).collect();
            (d, b, o)
        })
        .collect();
    let (mut d, mut b, mut o) = (vec![], vec![], vec![]);
    for (x, y, z) in rows {
        d.extend(x);
        b.extend(y);
        o.extend(z);
    }
    DM { c, lex, nq, nd, d, b, o, qt, mini }
}
impl DM {
    fn dq(&self, qi: usize) -> &[f32] { &self.d[qi * self.nd..(qi + 1) * self.nd] }
    fn bq(&self, qi: usize) -> &[f32] { &self.b[qi * self.nd..(qi + 1) * self.nd] }
    fn oq(&self, qi: usize) -> &[f32] { &self.o[qi * self.nd..(qi + 1) * self.nd] }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Sys {
    Dense,
    Bm25,
    /// g1-style z-score hybrid: (a/10) z(dense) + (1-a/10) z(bm25)
    Hyb(u8),
    Rrf,
    Rsf(u8),
    /// top-20 of Hyb(a) re-scored z(first) + (g/10) z(overlap)
    RerOv(u8, u8),
    RerBm(u8, u8),
    RerLogit(u8),
}
impl Sys {
    fn label(&self) -> String {
        match self {
            Sys::Dense => "dense+spell".into(),
            Sys::Bm25 => "BM25(spell)".into(),
            Sys::Hyb(a) => format!("hybrid-z(a={:.1})", *a as f32 / 10.0),
            Sys::Rrf => "RRF(k=60)".into(),
            Sys::Rsf(a) => format!("RSF(a={:.1})", *a as f32 / 10.0),
            Sys::RerOv(a, g) => format!("rerank20-overlap(a={:.1},g={:.1})", *a as f32 / 10.0, *g as f32 / 10.0),
            Sys::RerBm(a, g) => format!("rerank20-bm25(a={:.1},g={:.1})", *a as f32 / 10.0, *g as f32 / 10.0),
            Sys::RerLogit(a) => format!("rerank20-logit(a={:.1})", *a as f32 / 10.0),
        }
    }
}
fn rank_of(v: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
    idx
}
fn hyb(dm: &DM, qi: usize, a10: u8) -> Vec<f32> {
    let a = a10 as f32 / 10.0;
    let (zd, zb) = (zscore(dm.dq(qi)), zscore(dm.bq(qi)));
    zd.iter().zip(&zb).map(|(x, y)| a * x + (1.0 - a) * y).collect()
}
fn logit_feats(dm: &DM, qi: usize, first: &[f32], j: usize, rank: usize, bmax: f32) -> [f64; NFD] {
    [1.0, dm.dq(qi)[j] as f64, (dm.bq(qi)[j] / bmax) as f64, dm.oq(qi)[j] as f64, first[j] as f64, 1.0 / (rank + 1) as f64]
}
/// final doc score vector for one query under system `s` (candidates of the reranker sit above the rest) + number of scorer calls
fn score(dm: &DM, qi: usize, s: Sys, w: &[f64; NFD]) -> (Vec<f32>, usize) {
    let nd = dm.nd;
    match s {
        Sys::Dense => (dm.dq(qi).to_vec(), 0),
        Sys::Bm25 => (dm.bq(qi).to_vec(), 0),
        Sys::Hyb(a) => (hyb(dm, qi, a), 0),
        Sys::Rrf | Sys::Rsf(_) => {
            let mut out = vec![0.0f32; nd];
            let lists = [dm.dq(qi), dm.bq(qi)];
            for (li, l) in lists.iter().enumerate() {
                let idx = rank_of(l);
                let top = &idx[..DEPTH.min(nd)];
                match s {
                    Sys::Rrf => {
                        for (r, &j) in top.iter().enumerate() {
                            out[j] += 1.0 / (RRF + (r + 1) as f32);
                        }
                    }
                    Sys::Rsf(a10) => {
                        let wgt = if li == 0 { a10 as f32 / 10.0 } else { 1.0 - a10 as f32 / 10.0 };
                        let (mx, mn) = (l[top[0]], l[*top.last().unwrap()]);
                        for &j in top {
                            out[j] += wgt * (l[j] - mn) / (mx - mn).max(1e-6);
                        }
                    }
                    _ => unreachable!(),
                }
            }
            (out, 0)
        }
        Sys::RerOv(a, g) | Sys::RerBm(a, g) => {
            let f = hyb(dm, qi, a);
            let idx = rank_of(&f);
            let cand = &idx[..M];
            let extra: Vec<f32> = match s {
                Sys::RerOv(..) => cand.iter().map(|&j| dm.oq(qi)[j]).collect(),
                _ => cand.iter().map(|&j| dm.bq(qi)[j]).collect(),
            };
            let (zf, ze) = (zscore(&cand.iter().map(|&j| f[j]).collect::<Vec<_>>()), zscore(&extra));
            let mut out: Vec<f32> = f.iter().map(|x| x - 1000.0).collect();
            for (k, &j) in cand.iter().enumerate() {
                out[j] = 100.0 + zf[k] + g as f32 / 10.0 * ze[k];
            }
            (out, M)
        }
        Sys::RerLogit(a) => {
            let f = hyb(dm, qi, a);
            let idx = rank_of(&f);
            let bmax = dm.bq(qi).iter().cloned().fold(1e-6f32, f32::max);
            let mut out: Vec<f32> = f.iter().map(|x| x - 1000.0).collect();
            for (r, &j) in idx[..M].iter().enumerate() {
                let x = logit_feats(dm, qi, &f, j, r, bmax);
                out[j] = 100.0 + sigmoid(w.iter().zip(&x).map(|(a, b)| a * b).sum()) as f32;
            }
            (out, M)
        }
    }
}
const RRF: f32 = 60.0;

fn train_doc_logit(a10: u8) -> [f64; NFD] {
    let dm = build_dm("C");
    let (mut x, mut y) = (vec![], vec![]);
    for qi in 0..dm.nq {
        let q = &dm.c.queries[qi];
        if q.concept < 0 {
            continue;
        }
        let f = hyb(&dm, qi, a10);
        let idx = rank_of(&f);
        let bmax = dm.bq(qi).iter().cloned().fold(1e-6f32, f32::max);
        for (r, &j) in idx[..M].iter().enumerate() {
            x.push(logit_feats(&dm, qi, &f, j, r, bmax));
            y.push((dm.c.docs[j].concept == q.concept) as u8 as f64);
        }
    }
    let w = fit_logit::<NFD>(&x, &y, 1.0);
    println!("doc-level logistic trained on corpus C: {} candidate rows ({} pos); w = {:?}", x.len(), y.iter().filter(|&&v| v > 0.5).count(), w);
    w
}

fn qm_all(dm: &DM, qis: &[usize], s: Sys, w: &[f64; NFD]) -> Vec<QM> {
    qis.par_iter().map(|&qi| eval_scores(&dm.c, &dm.c.queries[qi], &score(dm, qi, s, w).0)).collect()
}
/// per-seed means (random 300-query subsets of `qis`) of metric m
fn seed_means(per_q: &[QM], seeds: &[u64], m: usize) -> Vec<f64> {
    seeds
        .iter()
        .map(|&s| {
            let mut r = Rng::new(s.wrapping_mul(0x9E37).wrapping_add(11));
            let mut idx: Vec<usize> = (0..per_q.len()).collect();
            for i in (1..idx.len()).rev() {
                let j = r.below(i + 1);
                idx.swap(i, j);
            }
            idx.truncate(300);
            idx.iter().map(|&i| per_q[i].arr()[m]).sum::<f64>() / idx.len() as f64
        })
        .collect()
}
const MET: [(&str, usize); 5] = [("hit1", 0), ("mrr", 1), ("p3", 2), ("r3", 3), ("f13", 4)];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tag = arg(&args, "--corpus", "A");
    let confirm = args.iter().any(|a| a == "--confirm");
    let seeds = parse_range(&args, "--held", if confirm { (61, 90) } else { (31, 60) });
    let dm = build_dm(&tag);
    let held: Vec<usize> = (0..dm.nq).filter(|&i| dm.c.queries[i].concept >= 0 && !is_tune_concept(dm.c.queries[i].concept)).collect();
    let tune: Vec<usize> = (0..dm.nq).filter(|&i| is_tune_concept(dm.c.queries[i].concept)).collect();
    println!("corpus {tag}: {} queries ({} held-out-concept, {} tune-concept), {} docs", dm.nq, held.len(), tune.len(), dm.nd);

    // ------------------------------------------------ pick (tune split) or load frozen picks
    let picks_path = "results/h2_retrieval_picks.json";
    let mut systems: Vec<Sys> = vec![];
    let mut w = [0.0f64; NFD];
    let mut tune_log = vec![];
    if confirm {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(picks_path).expect("run corpus A first")).unwrap();
        let g = |k: &str| v[k].as_u64().unwrap() as u8;
        let (ha, ra) = (g("hyb_a"), g("rsf_a"));
        systems = vec![Sys::Dense, Sys::Bm25, Sys::Hyb(ha), Sys::Rrf, Sys::Rsf(ra), Sys::RerOv(g("ov_a"), g("ov_g")), Sys::RerBm(g("bm_a"), g("bm_g")), Sys::RerLogit(g("lg_a"))];
        for (i, x) in v["logit_w"].as_array().unwrap().iter().enumerate() {
            w[i] = x.as_f64().unwrap();
        }
    } else {
        let mrr_tune = |s: Sys, w: &[f64; NFD]| -> f64 { let q = qm_all(&dm, &tune, s, w); q.iter().map(|x| x.mrr).sum::<f64>() / q.len() as f64 };
        let best = |cands: Vec<Sys>, w: &[f64; NFD]| -> (Sys, f64) { cands.into_iter().map(|s| (s, mrr_tune(s, w))).fold((Sys::Dense, -1.0), |b, x| if x.1 > b.1 { x } else { b }) };
        let (hyb_s, m1) = best([5u8, 6, 7, 8, 9].iter().map(|&a| Sys::Hyb(a)).collect(), &w);
        let (rsf_s, m2) = best([3u8, 5, 7, 9].iter().map(|&a| Sys::Rsf(a)).collect(), &w);
        let ha = if let Sys::Hyb(a) = hyb_s { a } else { 7 };
        let mut c = vec![];
        for g in [5u8, 10, 20] {
            c.push(Sys::RerOv(ha, g));
        }
        let (ov_s, m3) = best(c, &w);
        let mut c = vec![];
        for g in [5u8, 10, 20] {
            c.push(Sys::RerBm(ha, g));
        }
        let (bm_s, m4) = best(c, &w);
        w = train_doc_logit(ha);
        let m5 = mrr_tune(Sys::RerLogit(ha), &w);
        println!("tune MRR: hyb {:?} {m1:.4} | rsf {:?} {m2:.4} | ov {:?} {m3:.4} | bm {:?} {m4:.4} | logit {m5:.4}", hyb_s, rsf_s, ov_s, bm_s);
        tune_log.push(json!({"hyb": hyb_s.label(), "m": m1, "rsf": rsf_s.label(), "m2": m2, "ov": ov_s.label(), "m3": m3, "bm": bm_s.label(), "m4": m4, "logit_mrr": m5}));
        let (ra, (oa, og), (ba, bg)) = (if let Sys::Rsf(a) = rsf_s { a } else { 5 }, if let Sys::RerOv(a, g) = ov_s { (a, g) } else { (7, 10) }, if let Sys::RerBm(a, g) = bm_s { (a, g) } else { (7, 10) });
        systems = vec![Sys::Dense, Sys::Bm25, Sys::Hyb(ha), Sys::Rrf, Sys::Rsf(ra), Sys::RerOv(oa, og), Sys::RerBm(ba, bg), Sys::RerLogit(ha)];
        std::fs::write(picks_path, serde_json::to_string_pretty(&json!({"hyb_a": ha, "rsf_a": ra, "ov_a": oa, "ov_g": og, "bm_a": ba, "bm_g": bg, "lg_a": ha, "logit_w": w, "tune_log": tune_log, "note": "alpha/gamma tuned on tune-concept queries (id%5==0) of corpus A by MRR; logistic trained on g1 corpus C"})).unwrap()).unwrap();
    }

    // ------------------------------------------------ held-out evaluation
    let per: Vec<Vec<QM>> = systems.iter().map(|&s| qm_all(&dm, &held, s, &w)).collect();
    let sm: Vec<Vec<Vec<f64>>> = per.iter().map(|p| MET.iter().map(|&(_, m)| seed_means(p, &seeds, m)).collect()).collect();
    println!("\n{:<40} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7}", "system (held-out concepts)", "hit1", "mrr", "P@3", "R@3", "F1@3", "calls/q");
    let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
    let mut rows = vec![];
    for (i, s) in systems.iter().enumerate() {
        let m: Vec<f64> = (0..5).map(|k| mean(&sm[i][k])).collect();
        let calls = score(&dm, held[0], *s, &w).1;
        println!("{:<40} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>7}", s.label(), m[0], m[1], m[2], m[3], m[4], calls);
        rows.push(json!({"system": s.label(), "hit1": m[0], "mrr": m[1], "p3": m[2], "r3": m[3], "f13": m[4], "scorer_calls_per_query": calls}));
    }
    // paired comparisons (30 seeds; subsets of a finite pool, NOT independent corpora), Holm over the family
    let pairs: Vec<(usize, usize)> = {
        let ix = |s: &str| systems.iter().position(|x| x.label().starts_with(s)).unwrap();
        let (d, h, r, f) = (ix("dense"), ix("hybrid"), ix("RRF"), ix("RSF"));
        let mut v = vec![(h, d), (r, d), (f, d), (r, h), (f, h), (ix("rerank20-overlap"), h), (ix("rerank20-bm25"), h), (ix("rerank20-logit"), h), (ix("rerank20-logit"), d), (ix("rerank20-overlap"), d), (ix("rerank20-bm25"), d), (ix("BM25"), d)];
        v.dedup();
        v
    };
    let mut cmps = vec![];
    for &(a, b) in &pairs {
        for &(mn, mi) in &MET {
            let d = paired_diff(&sm[a][mi], &sm[b][mi]);
            cmps.push((a, b, mn, d.diff.mean, d.diff.lo, d.diff.hi, d.wins, paired_t_pvalue(&sm[a][mi], &sm[b][mi])));
        }
    }
    let adj = holm_adjust(&cmps.iter().map(|c| c.7).collect::<Vec<_>>());
    let mut jc = vec![];
    println!("\npaired (a - b), 95% CI, Holm verdict (* win, ! loss):");
    for (k, c) in cmps.iter().enumerate() {
        let v = if adj[k] < 0.05 { if c.3 > 0.0 { "*" } else { "!" } } else { "" };
        if c.2 == "mrr" || c.2 == "f13" {
            println!("  {:<38} - {:<30} {:<5} {:+.4} [{:+.4},{:+.4}] wins {:>2}/30 {}", systems[c.0].label(), systems[c.1].label(), c.2, c.3, c.4, c.5, c.6, v);
        }
        jc.push(json!({"a": systems[c.0].label(), "b": systems[c.1].label(), "metric": c.2, "diff": c.3, "lo": c.4, "hi": c.5, "wins": c.6, "p": c.7, "holm_p": adj[k], "verdict": v}));
    }

    // ------------------------------------------------ measured latency / bytes (corpus-A-like single thread; live computation, not the precomputed matrices)
    let mut lat = json!({});
    {
        let qs: Vec<usize> = held.iter().step_by(held.len() / 200).take(200).cloned().collect();
        let reps = 5;
        let time = |f: &dyn Fn(usize)| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..reps {
                let t = Instant::now();
                for &qi in &qs {
                    f(qi);
                }
                best = best.min(t.elapsed().as_secs_f64() * 1e6 / qs.len() as f64);
            }
            best
        };
        let nd = dm.nd;
        let live_dense = |qi: usize| -> Vec<f32> { let q = dm.mini.qvec(QV::Spell, qi); (0..nd).map(|j| dot(q, dm.mini.dvec(j))).collect() };
        let t_dense = time(&|qi| { std::hint::black_box(live_dense(qi)); });
        let t_bm = time(&|qi| { std::hint::black_box(dm.lex.bm25(&dm.c.queries[qi].spell)); });
        let (dd, bb): (Vec<Vec<f32>>, Vec<Vec<f32>>) = (qs.iter().map(|&qi| live_dense(qi)).collect(), qs.iter().map(|&qi| dm.lex.bm25(&dm.c.queries[qi].spell)).collect());
        let idxmap: std::collections::HashMap<usize, usize> = qs.iter().enumerate().map(|(k, &q)| (q, k)).collect();
        let t_zfuse = time(&|qi| { let k = idxmap[&qi]; let (zd, zb) = (zscore(&dd[k]), zscore(&bb[k])); std::hint::black_box(zd.iter().zip(&zb).map(|(x, y)| 0.7 * x + 0.3 * y).fold(f32::MIN, f32::max)); });
        let t_rrf = time(&|qi| {
            let k = idxmap[&qi];
            let mut out = vec![0.0f32; nd];
            for l in [&dd[k], &bb[k]] {
                let mut idx: Vec<usize> = (0..nd).collect();
                idx.select_nth_unstable_by(DEPTH, |&a, &b| l[b].partial_cmp(&l[a]).unwrap());
                idx.truncate(DEPTH);
                idx.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap());
                for (r, &j) in idx.iter().enumerate() {
                    out[j] += 1.0 / (RRF + (r + 1) as f32);
                }
            }
            std::hint::black_box(out.iter().cloned().fold(0.0, f32::max));
        });
        let t_rsf = time(&|qi| {
            let k = idxmap[&qi];
            let mut out = vec![0.0f32; nd];
            for (li, l) in [&dd[k], &bb[k]].iter().enumerate() {
                let mut idx: Vec<usize> = (0..nd).collect();
                idx.select_nth_unstable_by(DEPTH, |&a, &b| l[b].partial_cmp(&l[a]).unwrap());
                idx.truncate(DEPTH);
                let (mx, mn) = idx.iter().fold((f32::MIN, f32::MAX), |(a, b), &j| (a.max(l[j]), b.min(l[j])));
                for &j in &idx {
                    out[j] += (if li == 0 { 0.5 } else { 0.5 }) * (l[j] - mn) / (mx - mn).max(1e-6);
                }
            }
            std::hint::black_box(out.iter().cloned().fold(0.0, f32::max));
        });
        // rerank scorers on the top-20 of the z-hybrid: per-call cost = time(scorer part)/20
        let tops: Vec<Vec<usize>> = qs.iter().map(|&qi| { let f = hyb(&dm, qi, 7); rank_of(&f)[..M].to_vec() }).collect();
        let fz: Vec<Vec<f32>> = qs.iter().map(|&qi| hyb(&dm, qi, 7)).collect();
        let t_ov = time(&|qi| { let k = idxmap[&qi]; std::hint::black_box(tops[k].iter().map(|&j| dm.lex.overlap(&dm.qt[qi], j)).sum::<f32>()); });
        let t_bmr = time(&|qi| { let k = idxmap[&qi]; let s = dm.lex.bm25_subset(&dm.c.queries[qi].spell, &tops[k]); std::hint::black_box(s.iter().sum::<f32>()); });
        let wl = w;
        let t_lg = time(&|qi| {
            let k = idxmap[&qi];
            let bmax = bb[k].iter().cloned().fold(1e-6f32, f32::max);
            let mut tot = 0.0;
            for (r, &j) in tops[k].iter().enumerate() {
                let x = [1.0, dd[k][j] as f64, (bb[k][j] / bmax) as f64, dm.lex.overlap(&dm.qt[qi], j) as f64, fz[k][j] as f64, 1.0 / (r + 1) as f64];
                tot += sigmoid(wl.iter().zip(&x).map(|(a, b)| a * b).sum());
            }
            std::hint::black_box(tot);
        });
        println!("\nlatency (us/query, single thread, live computation, 780 docs): dense scan {t_dense:.1} | BM25 {t_bm:.1} | z-hybrid fuse {t_zfuse:.1} | RRF fuse (2 rankings) {t_rrf:.1} | RSF fuse {t_rsf:.1}");
        println!("rerank scorers on 20 candidates (us/query; per call = /20): overlap {t_ov:.1} ({:.2}/call) | bm25 subset {t_bmr:.1} ({:.2}/call) | logistic incl. features {t_lg:.1} ({:.2}/call)", t_ov / 20.0, t_bmr / 20.0, t_lg / 20.0);
        lat = json!({"dense_scan_us": t_dense, "bm25_us": t_bm, "zhybrid_fuse_us": t_zfuse, "rrf_fuse_us": t_rrf, "rsf_fuse_us": t_rsf, "rerank_overlap_us_per_query": t_ov, "rerank_bm25_us_per_query": t_bmr, "rerank_logit_us_per_query": t_lg,
                     "per_call_us": {"overlap": t_ov / 20.0, "bm25": t_bmr / 20.0, "logit": t_lg / 20.0}, "note": "single-thread microbench on a shared box; rerank scorers are the incremental cost on top of the first stage"});
    }
    let bytes = json!({"dense_doc_vectors": dm.nd * 384 * 4, "bm25_index": dm.lex.index_bytes(), "rrf_rsf_transient_per_query": 2 * DEPTH * 12, "logistic_weights": NFD * 8, "rerank_extra_index": 0});
    println!("bytes: dense doc vectors {} B, BM25 index {} B, RRF/RSF transient {} B/query, logistic weights {} B, rerank extra index 0", dm.nd * 384 * 4, dm.lex.index_bytes(), 2 * DEPTH * 12, NFD * 8);
    let out = json!({"corpus": tag, "confirm": confirm, "seeds": [seeds[0], seeds[seeds.len() - 1]], "systems": rows, "paired": jc, "latency": lat, "bytes": bytes});
    let path = if confirm { format!("results/h2_retrieval_confirm_{tag}.json") } else { "results/h2_retrieval.json".to_string() };
    std::fs::write(&path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote {path}");
}

/// BM25 of a query against a SUBSET of docs only (what a reranker pays for 20 candidates): postings lookup by binary search
fn bm25_subset_impl(lex: &Lex, q: &str, cands: &[usize]) -> Vec<f32> {
    let (k1, b) = (1.2f32, 0.75f32);
    let mut s = vec![0.0f32; cands.len()];
    for t in lex.q_terms(q) {
        let p = &lex.postings[t as usize];
        for (ci, &d) in cands.iter().enumerate() {
            if let Ok(pos) = p.binary_search_by_key(&(d as u32), |x| x.0) {
                let tf = p[pos].1;
                let dl = lex.doc_len[d];
                s[ci] += lex.idf[t as usize] * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / lex.avg_len));
            }
        }
    }
    s
}
trait Bm25Sub {
    fn bm25_subset(&self, q: &str, cands: &[usize]) -> Vec<f32>;
}
impl Bm25Sub for Lex {
    fn bm25_subset(&self, q: &str, cands: &[usize]) -> Vec<f32> {
        bm25_subset_impl(self, q, cands)
    }
}
