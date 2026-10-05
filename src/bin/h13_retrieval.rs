//! H13 part 1 (+ corpus statistics, g1 bridge, latency/bytes): END-TO-END-ready retrieval comparison on the REAL-text h10 fixtures.
//!
//! Systems (doc level, each in a raw-query and a spell-corrected-query flavour): dense (MiniLM), BM25, hybrid z-score fusion, hybrid RRF (k=60,
//! top-50 lists), hybrid RSF (min-max in the top-50 lists), cheap top-20 rerank (idf-weighted query-token coverage over a z-fusion first stage).
//! Metrics: hit@1, MRR, hit@5, P/R/F1 @1/@3/@5 vs the STRUCTURAL relevance of h10 (section membership for headings, source paragraph for held-out sentences).
//!
//! PRE-DECLARED PROTOCOL (fixed before the first run; mirrors h10):
//!   tune   = world A seeds 1-8 (tune pool): ONLY the fusion weights (z alpha, RSF alpha, rerank beta) per flavor, by mix25 MRR (grid .4..1.0 / {.5,1.0}).
//!            RRF k=60 and the rerank first stage alpha=.7 are fixed, not tuned.
//!   frozen = A held seeds 31-60 (disjoint pool), A confirm 61-90, B confirm 61-90 (independent text world).
//!   units  = per-seed means (paired t-CI over seeds, as in h10) AND query-cluster bootstrap (resample the DISTINCT queries, 4000 reps); the bootstrap
//!            p-value (normal approx. from the bootstrap SE) is the PRIMARY one; Holm over 14 contrasts x 3 slices x 3 metrics = 126 tests per split.
//!   a "robust win" = Holm-significant in the same direction under BOTH the seed t-test and the cluster bootstrap.
//! Extras: lexical-overlap stratification, corpus statistics (query-doc token overlap), g1 templated corpora bridge (same engine), latency/bytes,
//!         top-8 lists of the final configuration for the routing loop (results/h13_top_{A,B}.bin).
//! Usage: h13_retrieval [--out results/h13_retrieval.json]

#[path = "h13_common/mod.rs"]
mod common;

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::time::Instant;

use common::*;
use serde_json::{Value, json};

type Evals = HashMap<(u32, u32), TextEval>;

struct Ctx {
    w: World,
    st: Store,
}

fn load_ctx(tag: &str) -> Ctx {
    let w = load_world(tag);
    let demb = w.emb[..w.ndoc * DIM].to_vec();
    let st = Store::new(&w.texts[..w.ndoc], demb, DIM);
    Ctx { w, st }
}

fn nthreads() -> usize {
    std::env::var("H13_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(6usize)
}

fn build_evals(ctx: &Ctx, seeds_: &[u64]) -> Evals {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for s in seeds_ {
            for r in &ctx.w.inst[&s.to_string()].rows {
                for v in 0..r.t.len() {
                    for fl in 0..2 {
                        let p = (r.q as u32, r.t[v][fl] as u32);
                        if seen.insert(p) {
                            pairs.push(p);
                        }
                    }
                }
            }
        }
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<TextEval>>> = pairs.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..nthreads() {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= pairs.len() {
                    break;
                }
                let (q, tid) = pairs[i];
                let te = eval_text(&ctx.st, &ctx.w.texts[tid as usize], ctx.w.vec(tid as usize), &ctx.w.queries[q as usize].rel);
                *out[i].lock().unwrap() = Some(te);
            });
        }
    });
    pairs.into_iter().zip(out).map(|(p, m)| (p, m.into_inner().unwrap().unwrap())).collect()
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Clean,
    Typo,
    Mix(f64),
}

fn row_val(ctx: &Ctx, ev: &Evals, pk: &Picks, r: &Row, v: usize, sys: usize, m: M) -> f64 {
    let te = &ev[&(r.q as u32, r.t[v][sys % 2] as u32)];
    metric(&te.outs[pk.kind(sys)], ctx.w.queries[r.q].rel.len(), m)
}

fn seed_val(ctx: &Ctx, ev: &Evals, pk: &Picks, rows: &[Row], sys: usize, fam: Option<&str>, kind: Kind, m: M) -> f64 {
    let fams: Vec<&str> = match fam {
        Some(f) => vec![f],
        None => vec!["H", "S"],
    };
    let var_mean = |f: &str, v: usize| -> f64 {
        let xs: Vec<f64> = rows.iter().filter(|r| r.fam == f).map(|r| row_val(ctx, ev, pk, r, v, sys, m)).collect();
        mean(&xs)
    };
    let one = |f: &str| -> f64 {
        match kind {
            Kind::Clean => var_mean(f, 0),
            Kind::Typo => 0.5 * (var_mean(f, 1) + var_mean(f, 2)),
            Kind::Mix(p) => (1.0 - p) * var_mean(f, 0) + p * 0.5 * (var_mean(f, 1) + var_mean(f, 2)),
        }
    };
    mean(&fams.iter().map(|f| one(f)).collect::<Vec<_>>())
}

fn series(ctx: &Ctx, ev: &Evals, pk: &Picks, sds: &[u64], sys: usize, fam: Option<&str>, kind: Kind, m: M) -> Vec<f64> {
    sds.iter().map(|s| seed_val(ctx, ev, pk, &ctx.w.inst[&s.to_string()].rows, sys, fam, kind, m)).collect()
}

fn sidx(n: &str) -> usize {
    SYS_NAMES.iter().position(|x| *x == n).unwrap_or_else(|| panic!("sys {n}"))
}

const CONTRASTS: [(&str, &str); 14] = [
    ("bm25_s", "dense_s"),
    ("hybz_s", "dense_s"),
    ("hybrrf_s", "dense_s"),
    ("hybrsf_s", "dense_s"),
    ("rerank_s", "dense_s"),
    ("hybz_s", "bm25_s"),
    ("hybz_s", "hybrrf_s"),
    ("hybz_s", "hybrsf_s"),
    ("hybz_s", "rerank_s"),
    ("dense_s", "dense_raw"),
    ("hybz_s", "hybz_raw"),
    ("hybz_raw", "dense_raw"),
    ("dense_raw", "bm25_raw"),
    ("dense_s", "bm25_s"),
];
const FAM_M: [M; 3] = [M::Mrr, M::Hit1, M::F1at3];
const SLICES: [(&str, Kind); 3] = [("mix25", Kind::Mix(0.25)), ("clean", Kind::Clean), ("typo", Kind::Typo)];

/// per-distinct-query values: [sys][fam_metric][variant 0..3] summed over appearances + count
struct QAcc {
    sums: Vec<[[f64; 3]; 3]>, // [sys][metric][variant]
    n: f64,
}

fn query_table(ctx: &Ctx, ev: &Evals, pk: &Picks, sds: &[u64]) -> BTreeMap<usize, QAcc> {
    let mut acc: BTreeMap<usize, QAcc> = BTreeMap::new();
    for s in sds {
        for r in &ctx.w.inst[&s.to_string()].rows {
            let e = acc.entry(r.q).or_insert_with(|| QAcc { sums: vec![[[0.0; 3]; 3]; 12], n: 0.0 });
            e.n += 1.0;
            for sys in 0..12 {
                for (mi, m) in FAM_M.iter().enumerate() {
                    for v in 0..3 {
                        e.sums[sys][mi][v] += row_val(ctx, ev, pk, r, v, sys, *m);
                    }
                }
            }
        }
    }
    acc
}
fn qval(a: &QAcc, sys: usize, mi: usize, kind: Kind) -> f64 {
    let c = a.sums[sys][mi][0] / a.n;
    let t = 0.5 * (a.sums[sys][mi][1] + a.sums[sys][mi][2]) / a.n;
    match kind {
        Kind::Clean => c,
        Kind::Typo => t,
        Kind::Mix(p) => (1.0 - p) * c + p * t,
    }
}

fn report_split(label: &str, ctx: &Ctx, ev: &Evals, pk: &Picks, sds: &[u64], log: &mut String, with_tests: bool) -> Value {
    let mut out = json!({});
    // ---- levels
    let mut lv = json!({});
    log.push_str(&format!("\n=== {label}: mean over {} seeds ===\n  system        mix25 hit1/MRR/F1@3     clean hit1/MRR/F1@3    typo hit1/MRR/F1@3\n", sds.len()));
    for sys in 0..12 {
        let mut row = json!({});
        let mut line = format!("  {:11}", SYS_NAMES[sys]);
        for (sn, fam, kind) in [("mix25", None, Kind::Mix(0.25)), ("clean", None, Kind::Clean), ("typo", None, Kind::Typo), ("mix25-H", Some("H"), Kind::Mix(0.25)), ("mix25-S", Some("S"), Kind::Mix(0.25))] {
            let mut cell = json!({});
            for m in ALL_M {
                cell[format!("{:?}", m)] = json!(mean(&series(ctx, ev, pk, sds, sys, fam, kind, m)));
            }
            if matches!(sn, "mix25" | "clean" | "typo") {
                line.push_str(&format!(" | {:.3}/{:.3}/{:.3}", cell["Hit1"].as_f64().unwrap(), cell["Mrr"].as_f64().unwrap(), cell["F1at3"].as_f64().unwrap()));
            }
            row[sn] = cell;
        }
        log.push_str(&line);
        log.push('\n');
        lv[SYS_NAMES[sys]] = row;
    }
    out["levels"] = lv;
    if !with_tests {
        return out;
    }
    // ---- tests: seed t-CI + cluster bootstrap, Holm each
    let qt = query_table(ctx, ev, pk, sds);
    let nq = qt.len();
    let mut tests: Vec<(usize, usize, usize, usize)> = Vec::new(); // (contrast, slice, metric idx)
    for c in 0..CONTRASTS.len() {
        for s in 0..SLICES.len() {
            for mi in 0..FAM_M.len() {
                tests.push((c, s, mi, 0));
            }
        }
    }
    let mut seed_cmp: Vec<Cmp> = Vec::new();
    let mut diffs: Vec<Vec<f64>> = Vec::new();
    for &(c, s, mi, _) in &tests {
        let (a, b) = (sidx(CONTRASTS[c].0), sidx(CONTRASTS[c].1));
        let (_, kind) = SLICES[s];
        let sa = series(ctx, ev, pk, sds, a, None, kind, FAM_M[mi]);
        let sb = series(ctx, ev, pk, sds, b, None, kind, FAM_M[mi]);
        seed_cmp.push(compare(&sa, &sb));
        diffs.push(qt.values().map(|q| qval(q, a, mi, kind) - qval(q, b, mi, kind)).collect());
    }
    let boots = boot_cluster_multi(&diffs, 4000, 20240613);
    let pt: Vec<f64> = seed_cmp.iter().map(|c| c.p).collect();
    let pb: Vec<f64> = boots.iter().map(|b| norm_two_sided_p(b.0 / b.3.max(1e-12))).collect();
    let (ht, hb) = (holm(&pt), holm(&pb));
    let mut cj = Vec::new();
    log.push_str(&format!("--- paired contrasts a-b ({} distinct queries; seed t-CI | cluster-bootstrap CI; Holm over {} tests: * win ! loss; robust = both) ---\n", nq, tests.len()));
    for (i, &(c, s, mi, _)) in tests.iter().enumerate() {
        let sg = |h: f64, d: f64| if h < 0.05 { if d > 0.0 { 1 } else { -1 } } else { 0 };
        let (st_, sb_) = (sg(ht[i], seed_cmp[i].mean), sg(hb[i], boots[i].0));
        let robust = if st_ != 0 && st_ == sb_ { st_ } else { 0 };
        let mark = match robust { 1 => "*", -1 => "!", _ => "" };
        if SLICES[s].0 == "mix25" || SLICES[s].0 == "typo" && FAM_M[mi] == M::Mrr || SLICES[s].0 == "clean" && FAM_M[mi] == M::Mrr {
            log.push_str(&format!("  {:9}-{:9} {:5} {:6}: {:+.4} t[{:+.4},{:+.4}] boot[{:+.4},{:+.4}] holm_t {:.3} holm_b {:.3} {}\n", CONTRASTS[c].0, CONTRASTS[c].1, SLICES[s].0, format!("{:?}", FAM_M[mi]), seed_cmp[i].mean, seed_cmp[i].lo, seed_cmp[i].hi, boots[i].1, boots[i].2, ht[i], hb[i], mark));
        }
        cj.push(json!({"a": CONTRASTS[c].0, "b": CONTRASTS[c].1, "slice": SLICES[s].0, "metric": format!("{:?}", FAM_M[mi]), "diff_seed": seed_cmp[i].mean, "t_lo": seed_cmp[i].lo, "t_hi": seed_cmp[i].hi, "p_t": pt[i], "holm_t": ht[i],
            "diff_boot": boots[i].0, "boot_lo": boots[i].1, "boot_hi": boots[i].2, "boot_se": boots[i].3, "p_boot": pb[i], "holm_boot": hb[i], "robust": robust}));
    }
    let nrob_w = cj.iter().filter(|x| x["robust"] == 1).count();
    let nrob_l = cj.iter().filter(|x| x["robust"] == -1).count();
    log.push_str(&format!("  robust wins {nrob_w}, robust losses {nrob_l} of {}\n", cj.len()));
    out["compare"] = Value::Array(cj);
    out["n_queries"] = json!(nq);
    out
}

fn tune_picks(ctx: &Ctx, ev: &Evals, sds: &[u64], log: &mut String) -> (Picks, Value) {
    let mut pk = Picks::default();
    let mut tab = json!({});
    for fl in 0..2 {
        let probe = |base: usize, gi_of: &dyn Fn(usize) -> Picks| -> Vec<f64> {
            let n = if base == 4 { 2 } else { GRID.len() };
            (0..n).map(|g| mean(&series(ctx, ev, &gi_of(g), sds, base * 2 + fl, None, Kind::Mix(0.25), M::Mrr))).collect()
        };
        let argmax = |v: &[f64]| v.iter().enumerate().fold((0usize, f64::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0;
        let zs = probe(2, &|g| Picks { z: [g, g], ..Default::default() });
        let rs = probe(4, &|g| Picks { rsf: [g, g], ..Default::default() });
        let rr: Vec<f64> = (0..2).map(|g| mean(&series(ctx, ev, &Picks { rr: [g, g], ..Default::default() }, sds, 10 + fl, None, Kind::Mix(0.25), M::Mrr))).collect();
        pk.z[fl] = argmax(&zs);
        pk.rsf[fl] = argmax(&rs);
        pk.rr[fl] = argmax(&rr);
        let fln = ["raw", "spell"][fl];
        tab[fln] = json!({"z_mrr": zs, "rsf_mrr": rs, "rerank_mrr": rr});
        log.push_str(&format!("TUNE ({fln}, A seeds 1-8 tune pool, mix25 MRR): z alpha {} (MRR {:.4}), RSF alpha {} ({:.4}), rerank beta {} ({:.4}); z grid {:?}\n", GRID[pk.z[fl]], zs[pk.z[fl]], GRID[pk.rsf[fl]], rs[pk.rsf[fl]], RR_BETA[pk.rr[fl]], rr[pk.rr[fl]], zs.iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>()));
    }
    (pk, tab)
}

/// per-query lexical overlap of the clean query text with its best-covering relevant doc
fn overlap_stats(ctx: &Ctx, sds: &[u64]) -> BTreeMap<usize, (String, f64, f64)> {
    let mut m = BTreeMap::new();
    for s in sds {
        for r in &ctx.w.inst[&s.to_string()].rows {
            if m.contains_key(&r.q) {
                continue;
            }
            let qt = ctx.st.qtoks(&ctx.w.texts[r.t[0][0]]);
            let rel = &ctx.w.queries[r.q].rel;
            let idfc = rel.iter().map(|&d| ctx.st.coverage(&qt, d)).fold(0.0f32, f32::max) as f64;
            let plain = rel.iter().map(|&d| ctx.st.coverage_plain(&qt, d)).fold(0.0f32, f32::max) as f64;
            m.insert(r.q, (r.fam.clone(), idfc, plain));
        }
    }
    m
}

fn quantiles(v: &mut Vec<f64>) -> Value {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
    json!({"n": v.len(), "mean": mean(v), "p10": q(0.1), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p90": q(0.9)})
}

/// hybrid-minus-dense gain by lexical-overlap tertile (per-query mix25 MRR diffs; cluster bootstrap CI inside each tertile)
fn tertile_gain(ctx: &Ctx, qt: &BTreeMap<usize, QAcc>, ov: &BTreeMap<usize, (String, f64, f64)>, log: &mut String, label: &str) -> Value {
    let mut items: Vec<(f64, f64, f64)> = Vec::new(); // (overlap, hybz_s-dense_s, dense_s-bm25_s)
    for (q, a) in qt {
        let o = ov[q].1;
        items.push((o, qval(a, sidx("hybz_s"), 0, Kind::Mix(0.25)) - qval(a, sidx("dense_s"), 0, Kind::Mix(0.25)), qval(a, sidx("dense_s"), 0, Kind::Mix(0.25)) - qval(a, sidx("bm25_s"), 0, Kind::Mix(0.25))));
    }
    items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = items.len();
    let mut res = json!({});
    let _ = ctx;
    for (ti, (lo, hi)) in [(0, n / 3), (n / 3, 2 * n / 3), (2 * n / 3, n)].iter().enumerate() {
        let sl = &items[*lo..*hi];
        let d1: Vec<f64> = sl.iter().map(|x| x.1).collect();
        let d2: Vec<f64> = sl.iter().map(|x| x.2).collect();
        let b = boot_cluster_multi(&[d1.clone(), d2.clone()], 2000, 99 + ti as u64);
        let (om_lo, om_hi) = (sl[0].0, sl[sl.len() - 1].0);
        log.push_str(&format!("  {label} overlap tertile {} [{om_lo:.2},{om_hi:.2}] n={}: hybz_s-dense_s MRR {:+.4} [{:+.4},{:+.4}] | dense_s-bm25_s {:+.4} [{:+.4},{:+.4}]\n", ti + 1, sl.len(), b[0].0, b[0].1, b[0].2, b[1].0, b[1].1, b[1].2));
        res[format!("t{}", ti + 1)] = json!({"overlap_range": [om_lo, om_hi], "n": sl.len(), "hybz_minus_dense": {"d": b[0].0, "lo": b[0].1, "hi": b[0].2}, "dense_minus_bm25": {"d": b[1].0, "lo": b[1].1, "hi": b[1].2}});
    }
    // Pearson correlation overlap vs hybrid gain
    let xs: Vec<f64> = items.iter().map(|x| x.0).collect();
    let ys: Vec<f64> = items.iter().map(|x| x.1).collect();
    let (mx, my) = (mean(&xs), mean(&ys));
    let cov: f64 = xs.iter().zip(&ys).map(|(a, b)| (a - mx) * (b - my)).sum();
    let r = cov / (xs.iter().map(|a| (a - mx) * (a - mx)).sum::<f64>().sqrt() * ys.iter().map(|b| (b - my) * (b - my)).sum::<f64>().sqrt()).max(1e-12);
    log.push_str(&format!("  {label} corr(overlap, hybz_s-dense_s MRR per query) = {r:+.3}\n"));
    res["corr_overlap_vs_hybrid_gain"] = json!(r);
    res
}

// ---------------------------------------------------------------------------------------------
// g1 templated-corpus bridge: same engine / same fusion kernels on the g1 synthetic corpora

struct G1 {
    st: Store,
    qs: Vec<(String, String, String, i32, Vec<f32>, Vec<f32>)>, // (raw text, spell text, cls, concept, raw vec, spell vec)
    rel: Vec<Vec<usize>>,
}

fn load_g1(tag: &str) -> G1 {
    let c: Value = serde_json::from_slice(&std::fs::read(format!("results/g1_corpus_{tag}.json")).unwrap()).unwrap();
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read(format!("results/g1_texts_{tag}.json")).unwrap()).unwrap();
    let raw = std::fs::read(format!("results/g1_emb_{tag}_mini.f32")).unwrap();
    let emb: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    assert_eq!(emb.len(), texts.len() * DIM);
    let tix: HashMap<&str, usize> = texts.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();
    let docs: Vec<String> = c["docs"].as_array().unwrap().iter().map(|d| d["text"].as_str().unwrap().to_string()).collect();
    let dconc: Vec<i64> = c["docs"].as_array().unwrap().iter().map(|d| d["concept"].as_i64().unwrap()).collect();
    let mut demb = Vec::new();
    for d in &docs {
        let i = tix[d.as_str()];
        demb.extend_from_slice(&emb[i * DIM..(i + 1) * DIM]);
    }
    let st = Store::new(&docs, demb, DIM);
    let mut qs = Vec::new();
    let mut rel = Vec::new();
    for q in c["queries"].as_array().unwrap() {
        let conc = q["concept"].as_i64().unwrap();
        if conc < 0 {
            continue;
        }
        let (t, s) = (q["text"].as_str().unwrap(), q["spell"].as_str().unwrap());
        let v = |x: &str| emb[tix[x] * DIM..(tix[x] + 1) * DIM].to_vec();
        qs.push((t.to_string(), s.to_string(), q["cls"].as_str().unwrap().to_string(), conc as i32, v(t), v(s)));
        rel.push((0..docs.len()).filter(|&d| dconc[d] == conc).collect());
    }
    G1 { st, qs, rel }
}

fn g1_bridge(tag: &str, log: &mut String) -> Value {
    let g = load_g1(tag);
    let nq = g.qs.len();
    // eval raw and spell flavour of every query
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<(TextEval, TextEval)>>> = (0..nq).map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..nthreads() {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= nq {
                    break;
                }
                let q = &g.qs[i];
                let a = eval_text(&g.st, &q.0, &q.4, &g.rel[i]);
                let b = eval_text(&g.st, &q.1, &q.5, &g.rel[i]);
                *out[i].lock().unwrap() = Some((a, b));
            });
        }
    });
    let ev: Vec<(TextEval, TextEval)> = out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect();
    let val = |i: usize, pk: &Picks, sys: usize, m: M| -> f64 {
        let te = if sys % 2 == 0 { &ev[i].0 } else { &ev[i].1 };
        metric(&te.outs[pk.kind(sys)], g.rel[i].len(), m)
    };
    let tune_idx: Vec<usize> = (0..nq).filter(|&i| g.qs[i].3 % 5 == 0).collect();
    let eval_idx: Vec<usize> = (0..nq).filter(|&i| g.qs[i].3 % 5 != 0).collect();
    let mut pk = Picks::default();
    for fl in 0..2 {
        let best = |base: usize, n: usize, mk: &dyn Fn(usize) -> Picks| -> usize {
            (0..n).map(|k| (k, mean(&tune_idx.iter().map(|&i| val(i, &mk(k), base * 2 + fl, M::Mrr)).collect::<Vec<_>>()))).fold((0, f64::MIN), |b, x| if x.1 > b.1 { x } else { b }).0
        };
        pk.z[fl] = best(2, GRID.len(), &|k| Picks { z: [k, k], ..Default::default() });
        pk.rsf[fl] = best(4, GRID.len(), &|k| Picks { rsf: [k, k], ..Default::default() });
        pk.rr[fl] = best(5, 2, &|k| Picks { rr: [k, k], ..Default::default() });
    }
    log.push_str(&format!("g1 {tag}: {} queries (ood dropped), tune concepts id%5==0 -> picks z {:?} rsf {:?} rerank beta {:?}\n", nq, [GRID[pk.z[0]], GRID[pk.z[1]]], [GRID[pk.rsf[0]], GRID[pk.rsf[1]]], [RR_BETA[pk.rr[0]], RR_BETA[pk.rr[1]]]));
    let mut res = json!({"picks": pk.to_json(), "n_queries_eval": eval_idx.len()});
    // levels
    let slices: [(&str, Box<dyn Fn(&str) -> bool>); 5] = [("all", Box::new(|_| true)), ("clean", Box::new(|c| c == "clean")), ("typo", Box::new(|c| c == "typo")), ("terse", Box::new(|c| c == "terse")), ("chatty", Box::new(|c| c == "chatty"))];
    let mut lv = json!({});
    log.push_str("  system       all hit1/MRR/F1@3     clean MRR  typo MRR  terse MRR chatty MRR\n");
    for sys in 0..12 {
        let mut row = json!({});
        let mut line = format!("  {:11}", SYS_NAMES[sys]);
        for (sn, f) in slices.iter() {
            let idx: Vec<usize> = eval_idx.iter().copied().filter(|&i| f(&g.qs[i].2)).collect();
            let mut cell = json!({});
            for m in ALL_M {
                cell[format!("{:?}", m)] = json!(mean(&idx.iter().map(|&i| val(i, &pk, sys, m)).collect::<Vec<_>>()));
            }
            if *sn == "all" {
                line.push_str(&format!(" | {:.3}/{:.3}/{:.3}", cell["Hit1"].as_f64().unwrap(), cell["Mrr"].as_f64().unwrap(), cell["F1at3"].as_f64().unwrap()));
            } else {
                line.push_str(&format!(" | {:.3}", cell["Mrr"].as_f64().unwrap()));
            }
            row[*sn] = cell;
        }
        log.push_str(&line);
        log.push('\n');
        lv[SYS_NAMES[sys]] = row;
    }
    res["levels"] = lv;
    // contrasts, cluster = concept, family 'all' x 3 metrics Holm; other slices descriptive
    let mut conc_ids: Vec<i32> = eval_idx.iter().map(|&i| g.qs[i].3).collect();
    conc_ids.sort();
    conc_ids.dedup();
    let mut cj = Vec::new();
    let mut diffs = Vec::new();
    let mut meta = Vec::new();
    for (sn, f) in slices.iter() {
        for (c, (a, b)) in CONTRASTS.iter().enumerate() {
            for (mi, m) in FAM_M.iter().enumerate() {
                let (sa, sb) = (sidx(a), sidx(b));
                let per_c: Vec<f64> = conc_ids
                    .iter()
                    .filter_map(|&cc| {
                        let idx: Vec<usize> = eval_idx.iter().copied().filter(|&i| g.qs[i].3 == cc && f(&g.qs[i].2)).collect();
                        if idx.is_empty() { None } else { Some(mean(&idx.iter().map(|&i| val(i, &pk, sa, *m) - val(i, &pk, sb, *m)).collect::<Vec<_>>())) }
                    })
                    .collect();
                diffs.push(per_c);
                meta.push((*sn, c, mi));
            }
        }
    }
    // bootstrap per slice (different unit counts) -> group by slice
    let mut by_slice: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, (sn, _, _)) in meta.iter().enumerate() {
        by_slice.entry(*sn).or_default().push(i);
    }
    let mut boots = vec![(0.0, 0.0, 0.0, 0.0); diffs.len()];
    for (_, ids) in &by_slice {
        let ds: Vec<Vec<f64>> = ids.iter().map(|&i| diffs[i].clone()).collect();
        let b = boot_cluster_multi(&ds, 4000, 777);
        for (k, &i) in ids.iter().enumerate() {
            boots[i] = b[k];
        }
    }
    let all_ids = &by_slice["all"];
    let pb: Vec<f64> = all_ids.iter().map(|&i| norm_two_sided_p(boots[i].0 / boots[i].3.max(1e-12))).collect();
    let hb = holm(&pb);
    for (k, &i) in all_ids.iter().enumerate() {
        let (c, mi) = (meta[i].1, meta[i].2);
        let mark = if hb[k] < 0.05 { if boots[i].0 > 0.0 { "*" } else { "!" } } else { "" };
        if matches!(FAM_M[mi], M::Mrr) {
            log.push_str(&format!("  g1 {tag} {:9}-{:9} all MRR {:+.4} boot[{:+.4},{:+.4}] holm {:.3} {}\n", CONTRASTS[c].0, CONTRASTS[c].1, boots[i].0, boots[i].1, boots[i].2, hb[k], mark));
        }
        cj.push(json!({"a": CONTRASTS[c].0, "b": CONTRASTS[c].1, "metric": format!("{:?}", FAM_M[mi]), "diff": boots[i].0, "lo": boots[i].1, "hi": boots[i].2, "holm": hb[k], "win": hb[k] < 0.05 && boots[i].0 > 0.0, "loss": hb[k] < 0.05 && boots[i].0 < 0.0}));
    }
    res["compare_all"] = Value::Array(cj);
    let mut cs = Vec::new();
    for (i, (sn, c, mi)) in meta.iter().enumerate() {
        if *sn != "all" && matches!(FAM_M[*mi], M::Mrr) {
            cs.push(json!({"slice": sn, "a": CONTRASTS[*c].0, "b": CONTRASTS[*c].1, "diff": boots[i].0, "lo": boots[i].1, "hi": boots[i].2}));
        }
    }
    res["compare_slices_mrr"] = Value::Array(cs);
    // overlap statistics of the g1 queries
    let mut ov: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut ovp: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for i in 0..nq {
        let qt = g.st.qtoks(&g.qs[i].0);
        let idfc = g.rel[i].iter().map(|&d| g.st.coverage(&qt, d)).fold(0.0f32, f32::max) as f64;
        let pl = g.rel[i].iter().map(|&d| g.st.coverage_plain(&qt, d)).fold(0.0f32, f32::max) as f64;
        ov.entry(g.qs[i].2.clone()).or_default().push(idfc);
        ovp.entry(g.qs[i].2.clone()).or_default().push(pl);
        ov.entry("ALL".into()).or_default().push(idfc);
        ovp.entry("ALL".into()).or_default().push(pl);
    }
    let mut ost = json!({});
    for (k, mut v) in ov {
        ost[k.clone()] = json!({"idf_cov": quantiles(&mut v), "plain_cov": quantiles(&mut ovp.get_mut(&k).unwrap().clone())});
    }
    res["overlap"] = ost;
    res
}

// ---------------------------------------------------------------------------------------------
// latency (single thread, live computation) and bytes

fn bench(ctx: &Ctx, ev_rows: &[(usize, usize)], log: &mut String) -> Value {
    let st = &ctx.st;
    let n = ev_rows.len();
    let t = |f: &mut dyn FnMut(usize)| -> f64 {
        let t0 = Instant::now();
        for i in 0..n {
            f(i);
        }
        t0.elapsed().as_secs_f64() * 1e6 / n as f64
    };
    let mut dense = Vec::new();
    let mut bm = Vec::new();
    let mut buf = Vec::new();
    let mut sink = 0usize;
    let us_dense = t(&mut |i| {
        st.dense(ctx.w.vec(ev_rows[i].1), &mut dense);
        sink += dense.len();
    });
    let us_bm = t(&mut |i| {
        st.bm.score(&ctx.w.texts[ev_rows[i].1], &mut bm);
        sink += bm.len();
    });
    // pre-compute inputs so each fusion cost excludes the scans
    let ins: Vec<(Vec<f32>, Vec<f32>)> = ev_rows
        .iter()
        .map(|&(_, tid)| {
            let mut d = Vec::new();
            st.dense(ctx.w.vec(tid), &mut d);
            let mut b = Vec::new();
            st.bm.score(&ctx.w.texts[tid], &mut b);
            (d, b)
        })
        .collect();
    let us_top8_dense = t(&mut |i| {
        sink += top_n(&ins[i].0, TOPN).len();
    });
    let us_z = t(&mut |i| {
        let zd = zscore(&ins[i].0);
        let zb = zscore(&ins[i].1);
        fuse_z(&zd, &zb, 0.7, &mut buf);
        sink += top_n(&buf, TOPN).len();
    });
    let us_rrf = t(&mut |i| {
        let dt = top_n(&ins[i].0, LIST_N);
        let mut bt = top_n(&ins[i].1, LIST_N);
        bt.retain(|&d| ins[i].1[d] > 0.0);
        fuse_rrf(&dt, &bt, st.ndoc, &mut buf);
        sink += top_n(&buf, TOPN).len();
    });
    let us_rsf = t(&mut |i| {
        let dt = top_n(&ins[i].0, LIST_N);
        let mut bt = top_n(&ins[i].1, LIST_N);
        bt.retain(|&d| ins[i].1[d] > 0.0);
        fuse_rsf(&ins[i].0, &dt, &ins[i].1, &bt, 0.7, &mut buf);
        sink += top_n(&buf, TOPN).len();
    });
    let mut buf2 = Vec::new();
    let us_rr = t(&mut |i| {
        let zd = zscore(&ins[i].0);
        let zb = zscore(&ins[i].1);
        fuse_z(&zd, &zb, RR_ALPHA, &mut buf);
        let qt = st.qtoks(&ctx.w.texts[ev_rows[i].1]);
        rerank20(st, &qt, &buf, 0.5, &mut buf2);
        sink += top_n(&buf2, TOPN).len();
    });
    let us_rr_only = {
        // rerank stage alone (given the first-stage scores): coverage of 20 candidates
        let zs: Vec<Vec<f32>> = ins.iter().map(|(d, b)| { let mut o = Vec::new(); fuse_z(&zscore(d), &zscore(b), RR_ALPHA, &mut o); o }).collect();
        t(&mut |i| {
            let qt = st.qtoks(&ctx.w.texts[ev_rows[i].1]);
            rerank20(st, &qt, &zs[i], 0.5, &mut buf2);
            sink += buf2.len();
        })
    };
    let r = json!({
        "n_docs": st.ndoc, "n_queries_timed": n,
        "us_per_query": {"dense_scan": us_dense, "bm25": us_bm, "top8_of_dense": us_top8_dense, "z_fusion_incl_top8": us_z, "rrf_incl_top8": us_rrf, "rsf_incl_top8": us_rsf, "rerank20_incl_first_stage_z_top8": us_rr, "rerank20_stage_only": us_rr_only,
            "total_dense": us_dense + us_top8_dense, "total_bm25": us_bm, "total_hybz": us_dense + us_bm + us_z, "total_hybrrf": us_dense + us_bm + us_rrf, "total_hybrsf": us_dense + us_bm + us_rsf, "total_rerank20": us_dense + us_bm + us_rr},
        "bytes": {"dense_doc_vectors": st.bytes_dense(), "bm25_index_approx": st.bytes_bm25(), "doc_token_sets_for_rerank": st.bytes_tokens(),
                  "transient_per_query_dense_bm25_vectors": 2 * st.ndoc * 4, "transient_z_fusion": 4 * st.ndoc * 4, "transient_rrf_rsf_lists": 2 * LIST_N * 16 + st.ndoc * 4, "rerank_candidates": RERANK_N * 8},
        "note": "single thread, brute-force scan, live computation on a shared box (timings noisy +-50%); the MiniLM query encode (3.3-7.6 ms, results/h10_embmeta_A.json) and the python spell corrector (0.03-0.08 ms) are NOT in these numbers and dominate the retrieval cost",
        "sink": sink});
    log.push_str(&format!("\nLATENCY us/query ({} docs, single thread): dense scan {us_dense:.0}, BM25 {us_bm:.1}, top8(dense) {us_top8_dense:.0}, z-fusion+top8 {us_z:.0}, RRF+top8 {us_rrf:.0}, RSF+top8 {us_rsf:.0}, rerank20 (z first stage + 20 coverage + top8) {us_rr:.0} (stage only {us_rr_only:.1})\n", st.ndoc));
    log.push_str(&format!("BYTES: dense vectors {} B, BM25 index ~{} B, doc token sets {} B\n", st.bytes_dense(), st.bytes_bm25(), st.bytes_tokens()));
    r
}

fn write_tops(tag: &str, ctx: &Ctx, ev: &Evals, pk: &Picks, sds: &[u64]) -> usize {
    // keys (tid, flavor) used by the given seeds; 6 base kinds (dense, bm25, hybz, rrf, rsf, rerank) x TOPN x (u32 doc, f32 dense cosine)
    let mut keys: BTreeMap<(u32, u8), u32> = BTreeMap::new(); // -> q
    for s in sds {
        for r in &ctx.w.inst[&s.to_string()].rows {
            for v in 0..r.t.len() {
                for fl in 0..2u8 {
                    keys.entry((r.t[v][fl as usize] as u32, fl)).or_insert(r.q as u32);
                }
            }
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(format!("results/h13_top_{tag}.bin")).unwrap());
    f.write_all(&(keys.len() as u32).to_le_bytes()).unwrap();
    for (&(tid, fl), &q) in &keys {
        let te = &ev[&(q, tid)];
        f.write_all(&tid.to_le_bytes()).unwrap();
        f.write_all(&[fl]).unwrap();
        for base in 0..6usize {
            let kind = pk.kind(base * 2 + fl as usize);
            for &(d, c) in &te.tops[kind] {
                f.write_all(&d.to_le_bytes()).unwrap();
                f.write_all(&c.to_le_bytes()).unwrap();
            }
        }
    }
    keys.len()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let outp = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/h13_retrieval.json".into());
    let t0 = Instant::now();
    let mut log = String::new();
    let a = load_ctx("A");
    let b = load_ctx("B");
    println!("loaded A {} docs, B {} docs ({:.1}s)", a.w.ndoc, b.w.ndoc, t0.elapsed().as_secs_f64());
    let (s_tune, s_held, s_conf) = (seeds(1, 8), seeds(31, 60), seeds(61, 90));
    let mut sa: Vec<u64> = s_tune.clone();
    sa.extend(&s_held);
    sa.extend(&s_conf);
    let eva = build_evals(&a, &sa);
    println!("A evals {} ({:.1}s)", eva.len(), t0.elapsed().as_secs_f64());
    let evb = build_evals(&b, &s_conf);
    println!("B evals {} ({:.1}s)", evb.len(), t0.elapsed().as_secs_f64());

    let (pk, tune_tab) = tune_picks(&a, &eva, &s_tune, &mut log);
    let mut result = json!({"picks": pk.to_json(), "tune_table": tune_tab});
    result["tune_A_1_8"] = report_split("A tune seeds 1-8 (tune pool; descriptive)", &a, &eva, &pk, &s_tune, &mut log, false);
    result["held_A_31_60"] = report_split("A held-out seeds 31-60", &a, &eva, &pk, &s_held, &mut log, true);
    result["confirm_A_61_90"] = report_split("A confirm seeds 61-90 (frozen)", &a, &eva, &pk, &s_conf, &mut log, true);
    result["confirm_B_61_90"] = report_split("B confirm seeds 61-90 (frozen, independent text)", &b, &evb, &pk, &s_conf, &mut log, true);

    // ---- corpus statistics + overlap stratification
    log.push_str("\n=== lexical overlap of clean query with best relevant doc (idf-weighted / plain coverage of unique content tokens) ===\n");
    let mut ostats = json!({});
    let mut tert = json!({});
    for (label, ctx, ev, sds) in [("A", &a, &eva, &s_conf), ("A_held", &a, &eva, &s_held), ("B", &b, &evb, &s_conf)] {
        let ov = overlap_stats(ctx, sds);
        let mut cell = json!({});
        for fam in ["H", "S", "ALL"] {
            let mut i: Vec<f64> = ov.values().filter(|x| fam == "ALL" || x.0 == fam).map(|x| x.1).collect();
            let mut p: Vec<f64> = ov.values().filter(|x| fam == "ALL" || x.0 == fam).map(|x| x.2).collect();
            cell[fam] = json!({"idf_cov": quantiles(&mut i), "plain_cov": quantiles(&mut p)});
            log.push_str(&format!("  {label} {fam}: idf-cov mean {:.3} median {:.3}; plain-cov mean {:.3} median {:.3} (n={})\n", mean(&i), i[i.len() / 2], mean(&p), p[p.len() / 2], i.len()));
        }
        ostats[label] = cell;
        let qt = query_table(ctx, ev, &pk, sds);
        tert[label] = tertile_gain(ctx, &qt, &ov, &mut log, label);
    }
    result["overlap_real"] = ostats;
    result["overlap_tertiles"] = tert;

    // ---- g1 bridge
    let mut g1 = json!({});
    for tag in ["A", "B"] {
        g1[tag] = g1_bridge(tag, &mut log);
    }
    result["g1_bridge"] = g1;

    // ---- latency / bytes
    let rows: Vec<(usize, usize)> = a.w.inst["31"].rows.iter().take(200).map(|r| (r.q, r.t[0][0])).collect();
    result["latency_bytes"] = bench(&a, &rows, &mut log);

    // ---- top lists for the routing loop
    let na = write_tops("A", &a, &eva, &pk, &sa);
    let nb = write_tops("B", &b, &evb, &pk, &s_conf);
    log.push_str(&format!("\nwrote results/h13_top_A.bin ({na} (tid,flavor) entries), results/h13_top_B.bin ({nb})\n"));
    json_write("results/h13_picks.json", &pk.to_json());
    result["wall_s"] = json!(t0.elapsed().as_secs_f64());
    json_write(&outp, &result);
    println!("{log}");
    let _ = std::fs::write(outp.replace(".json", ".txt"), &log);
    println!("wrote {outp} in {:.1}s", t0.elapsed().as_secs_f64());
}
