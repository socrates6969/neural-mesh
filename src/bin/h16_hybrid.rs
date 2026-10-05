//! H16-A (R8): PARITY + BENCHMARK of the productionised library types `neural_mesh::hybrid::{HybridIndex, SpellCorrector}`.
//!
//! PARITY 1 (doc-level numbers): re-computes the h13 `hybz_s` system (z-score fusion, spell-corrected query text, alpha from results/h13_picks.json)
//!   with the LIBRARY on the real-text fixtures (results/h10_{texts,emb,inst}_{A,B}.*) and compares every level in results/h13_retrieval.json
//!   (A tune 1-8 / A held 31-60 / A confirm 61-90 / B confirm 61-90 x slices mix25/clean/typo/mix25-H/mix25-S x 9 metrics) within 1e-12.
//! PARITY 2 (top lists): the top-8 (doc id, dense cosine bits) of the `hybz` flavours in results/h13_top_{A,B}.bin vs the library, entry by entry.
//! PARITY 3 (spell): the Rust `SpellCorrector` vs the python Speller's corrected texts stored in the fixtures (sp1 = doc vocabulary known,
//!   sp2 = + query words with query-df >= 3), for every query text of both corpora.
//! BENCH: single-thread latency / bytes of the hybrid index on corpus A (6,831 real documentation paragraphs).
//! Usage: h16_hybrid [--out results/h16_hybrid.json]   (H16_THREADS, default 2, only for the parity evaluation; the benchmark is single-threaded)

#[path = "h13_common/mod.rs"]
mod common;

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use common::{ALL_M, DIM, Out, M, World, best_rank, load_world, mean, metric, seeds};
use neural_mesh::hybrid::{HybridConfig, HybridIndex, SpellCorrector};
use serde_json::{Value, json};

const TOPN: usize = 8;
const TOL: f64 = 1e-12;

fn nthreads() -> usize {
    std::env::var("H16_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(2usize)
}

struct Ctx {
    w: World,
    ix: HybridIndex,
    docs: Vec<String>,
}

fn load_ctx(tag: &str, alpha: f64) -> Ctx {
    let w = load_world(tag);
    let docs: Vec<String> = w.texts[..w.ndoc].to_vec();
    let dense = w.emb[..w.ndoc * DIM].to_vec();
    let ix = HybridIndex::new(&docs, dense, DIM, HybridConfig { alpha }).expect("index");
    Ctx { w, ix, docs }
}

#[derive(Clone)]
struct TextEval {
    out: Out,
    top: [(u32, f32); TOPN],
}

fn eval_text(ctx: &Ctx, tid: usize, rel: &[usize]) -> TextEval {
    let mut sc = Vec::new();
    ctx.ix.fused_scores(&ctx.w.texts[tid], ctx.w.vec(tid), &mut sc).unwrap();
    let mut dense = Vec::new();
    ctx.ix.dense_scores(ctx.w.vec(tid), &mut dense).unwrap();
    let top = neural_mesh::hybrid::top_n(&sc, TOPN);
    let mut t = [(0u32, 0f32); TOPN];
    for (i, &d) in top.iter().enumerate() {
        t[i] = (d as u32, dense[d]);
    }
    let hk = |k: usize| top.iter().take(k).filter(|d| rel.contains(d)).count() as u8;
    TextEval { out: Out { rank: best_rank(&sc, rel) as u32, h1: hk(1), h3: hk(3), h5: hk(5) }, top: t }
}

type Evals = HashMap<(u32, u32), TextEval>;

/// evaluate every distinct (query, text) pair used by `sds` for text flavours `fls`
fn build_evals(ctx: &Ctx, sds: &[u64], fls: &[usize]) -> Evals {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for s in sds {
        for r in &ctx.w.inst[&s.to_string()].rows {
            for v in 0..r.t.len() {
                for &fl in fls {
                    let p = (r.q as u32, r.t[v][fl] as u32);
                    if seen.insert(p) {
                        pairs.push(p);
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
                let te = eval_text(ctx, tid as usize, &ctx.w.queries[q as usize].rel);
                *out[i].lock().unwrap() = Some(te);
            });
        }
    });
    pairs.into_iter().zip(out).map(|(p, m)| (p, m.into_inner().unwrap().unwrap())).collect()
}

#[derive(Clone, Copy)]
enum Kind {
    Clean,
    Typo,
    Mix(f64),
}

/// copy of h13_retrieval::seed_val for one system flavour `fl`
fn seed_val(ctx: &Ctx, ev: &Evals, rows: &[common::Row], fl: usize, fam: Option<&str>, kind: Kind, m: M) -> f64 {
    let fams: Vec<&str> = match fam {
        Some(f) => vec![f],
        None => vec!["H", "S"],
    };
    let var_mean = |f: &str, v: usize| -> f64 {
        let xs: Vec<f64> = rows
            .iter()
            .filter(|r| r.fam == f)
            .map(|r| {
                let te = &ev[&(r.q as u32, r.t[v][fl] as u32)];
                metric(&te.out, ctx.w.queries[r.q].rel.len(), m)
            })
            .collect();
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

fn read_f(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}

struct Parity {
    n: usize,
    max_abs: f64,
    worst: String,
}

fn levels_parity(label: &str, key: &str, ctx: &Ctx, ev: &Evals, sds: &[u64], reference: &Value, flavor: usize, sysname: &str, par: &mut Parity, log: &mut String) {
    let lv = &reference[key]["levels"][sysname];
    let mut local_max = 0.0f64;
    let mut n = 0;
    for (sn, fam, kind) in [("mix25", None, Kind::Mix(0.25)), ("clean", None, Kind::Clean), ("typo", None, Kind::Typo), ("mix25-H", Some("H"), Kind::Mix(0.25)), ("mix25-S", Some("S"), Kind::Mix(0.25))] {
        for m in ALL_M {
            let v: Vec<f64> = sds.iter().map(|s| seed_val(ctx, ev, &ctx.w.inst[&s.to_string()].rows, flavor, fam, kind, m)).collect();
            let mine = mean(&v);
            let theirs = read_f(&lv[sn][format!("{:?}", m)]);
            let d = (mine - theirs).abs();
            n += 1;
            if d > local_max {
                local_max = d;
            }
            if d > par.max_abs {
                par.max_abs = d;
                par.worst = format!("{label} {sysname} {sn} {m:?}: lib {mine:.17} vs h13 {theirs:.17}");
            }
        }
    }
    par.n += n;
    log.push_str(&format!("  [{label}] {sysname}: {n} levels compared, max |lib - h13| = {local_max:.3e}\n"));
    // headline row
    let h = |sn: &str, m: &str| read_f(&lv[sn][m]);
    log.push_str(&format!("      h13 json: mix25 MRR {:.6} hit1 {:.6} F1@3 {:.6}\n", h("mix25", "Mrr"), h("mix25", "Hit1"), h("mix25", "F1at3")));
}

fn read_top_bin(path: &str) -> BTreeMap<(u32, u8), Vec<[(u32, u32); TOPN]>> {
    // (tid, flavour) -> 6 bases x top-8 (doc, cosine bits)
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
    let mut p = 4;
    let mut out = BTreeMap::new();
    for _ in 0..n {
        let tid = u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]);
        let fl = b[p + 4];
        p += 5;
        let mut bases = Vec::new();
        for _ in 0..6 {
            let mut t = [(0u32, 0u32); TOPN];
            for e in t.iter_mut() {
                let d = u32::from_le_bytes([b[p], b[p + 1], b[p + 2], b[p + 3]]);
                let c = u32::from_le_bytes([b[p + 4], b[p + 5], b[p + 6], b[p + 7]]);
                *e = (d, c);
                p += 8;
            }
            bases.push(t);
        }
        out.insert((tid, fl), bases);
    }
    assert_eq!(p, b.len());
    out
}

fn tops_parity(tag: &str, ctx: &Ctx, sds: &[u64], log: &mut String) -> Value {
    let reference = read_top_bin(&format!("results/h13_top_{tag}.bin"));
    // our own evaluation of every key present in the reference
    let mut q_of: BTreeMap<(u32, u8), u32> = BTreeMap::new();
    for s in sds {
        for r in &ctx.w.inst[&s.to_string()].rows {
            for v in 0..r.t.len() {
                for fl in 0..2u8 {
                    q_of.entry((r.t[v][fl as usize] as u32, fl)).or_insert(r.q as u32);
                }
            }
        }
    }
    let keys: Vec<(u32, u8)> = q_of.keys().copied().collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<[(u32, f32); TOPN]>>> = keys.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..nthreads() {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= keys.len() {
                    break;
                }
                let (tid, _) = keys[i];
                let q = q_of[&keys[i]];
                *out[i].lock().unwrap() = Some(eval_text(ctx, tid as usize, &ctx.w.queries[q as usize].rel).top);
            });
        }
    });
    let (mut n_ok, mut n_bad) = (0usize, 0usize);
    let mut missing = 0usize;
    for (k, o) in keys.iter().zip(out) {
        let mine = o.into_inner().unwrap().unwrap();
        match reference.get(k) {
            None => missing += 1,
            Some(b) => {
                let theirs = &b[2]; // base 2 = hybz
                if (0..TOPN).all(|i| mine[i].0 == theirs[i].0 && mine[i].1.to_bits() == theirs[i].1) {
                    n_ok += 1;
                } else {
                    n_bad += 1;
                }
            }
        }
    }
    log.push_str(&format!("  tops {tag}: {} keys, {} identical (doc ids + dense-cosine f32 bits), {} differ, {} missing from the reference file ({} reference entries)\n", keys.len(), n_ok, n_bad, missing, reference.len()));
    json!({"keys": keys.len(), "identical": n_ok, "differ": n_bad, "missing_in_reference": missing, "reference_entries": reference.len()})
}

fn spell_parity(tag: &str, ctx: &Ctx, log: &mut String) -> Value {
    let c: Value = serde_json::from_slice(&std::fs::read(format!("results/h10_corpus_{tag}.json")).unwrap()).unwrap();
    // python: known1 = doc vocabulary (all words); known2 = known1 + query words with query-df >= 3 (df over the unique words of each query)
    let mut qdf: HashMap<String, usize> = HashMap::new();
    for q in c["queries"].as_array().unwrap() {
        let t = q["text"].as_str().unwrap().to_lowercase();
        let mut seen = std::collections::HashSet::new();
        let mut cur = String::new();
        let mut flush = |cur: &mut String, seen: &mut std::collections::HashSet<String>| {
            if !cur.is_empty() {
                seen.insert(std::mem::take(cur));
            }
        };
        for ch in t.chars() {
            if ch.is_ascii_lowercase() {
                cur.push(ch);
            } else {
                flush(&mut cur, &mut seen);
            }
        }
        flush(&mut cur, &mut seen);
        for w in seen {
            *qdf.entry(w).or_insert(0) += 1;
        }
    }
    let sp1 = SpellCorrector::from_docs(&ctx.docs);
    let sp2 = SpellCorrector::from_docs(&ctx.docs).with_known(qdf.iter().filter(|(_, n)| **n >= 3).map(|(w, _)| w.clone()));
    let mut tot = [0usize; 2];
    let mut ok = [0usize; 2];
    let mut changed = [0usize; 2];
    let mut examples: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (ikey, inst) in &ctx.w.inst {
        for r in &inst.rows {
            for v in 0..r.t.len() {
                if ikey == "R" && v == 1 {
                    continue; // real-typo family: variant 1 is the human-fixed clean text copied to all three flavours (no speller involved)
                }
                let raw = r.t[v][0];
                for (fi, sp) in [(1usize, &sp1), (2usize, &sp2)] {
                    let tid = r.t[v][fi];
                    if !seen.insert((raw, tid, fi)) {
                        continue;
                    }
                    tot[fi - 1] += 1;
                    let mine = sp.correct(&ctx.w.texts[raw]);
                    if mine == ctx.w.texts[tid] {
                        ok[fi - 1] += 1;
                    } else if examples.len() < 5 {
                        examples.push(format!("{:?} -> lib {:?} vs py {:?}", ctx.w.texts[raw], mine, ctx.w.texts[tid]));
                    }
                    if ctx.w.texts[tid] != ctx.w.texts[raw].to_lowercase() {
                        changed[fi - 1] += 1;
                    }
                }
            }
        }
    }
    log.push_str(&format!(
        "  spell {tag}: sp1 {}/{} identical to the python Speller texts ({} actually corrected), sp2 {}/{} ({} corrected); first mismatches: {:?}\n",
        ok[0], tot[0], changed[0], ok[1], tot[1], changed[1], examples
    ));
    json!({"sp1": {"identical": ok[0], "total": tot[0], "texts_changed_by_python": changed[0]}, "sp2": {"identical": ok[1], "total": tot[1], "texts_changed_by_python": changed[1]}, "first_mismatches": examples})
}

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p) as usize]
}

fn bench(ctx: &Ctx, log: &mut String) -> Value {
    // 400 real clean query texts of corpus A (seed 31 rows, variant 0 = raw text, flavour 1 = spell-corrected text)
    let rows: Vec<usize> = ctx.w.inst["31"].rows.iter().take(400).map(|r| r.t[0][1]).collect();
    let n = rows.len();
    let mut sink = 0usize;
    let mut run = |f: &mut dyn FnMut(usize)| -> (f64, f64, f64) {
        // warm-up pass then 3 timed passes; report mean and p50/p95 of per-query latency (us)
        for i in 0..n {
            f(i);
        }
        let mut lat = Vec::new();
        for _ in 0..3 {
            for i in 0..n {
                let t0 = Instant::now();
                f(i);
                lat.push(t0.elapsed().as_secs_f64() * 1e6);
            }
        }
        let m = lat.iter().sum::<f64>() / lat.len() as f64;
        (m, pct(&mut lat.clone(), 0.5), pct(&mut lat, 0.95))
    };
    let mut dense = Vec::new();
    let dense_only = run(&mut |i| {
        ctx.ix.dense_scores(ctx.w.vec(rows[i]), &mut dense).unwrap();
        sink += neural_mesh::hybrid::top_n(&dense, TOPN).len();
    });
    let hybrid = run(&mut |i| {
        sink += ctx.ix.search(&ctx.w.texts[rows[i]], ctx.w.vec(rows[i]), TOPN).unwrap().len();
    });
    let mut bm = Vec::new();
    let bm25_only = run(&mut |i| {
        ctx.ix.bm25().score(&ctx.w.texts[rows[i]], &mut bm);
        sink += bm.len();
    });
    let sp = SpellCorrector::from_docs(&ctx.docs);
    let raw_q: Vec<&String> = ctx.w.inst["31"].rows.iter().take(400).map(|r| &ctx.w.texts[r.t[1][0]]).collect();
    let spell = run(&mut |i| {
        sink += sp.correct(raw_q[i % raw_q.len()]).len();
    });
    // build costs
    let t0 = Instant::now();
    let dense_copy = ctx.w.emb[..ctx.w.ndoc * DIM].to_vec();
    let built = HybridIndex::new(&ctx.docs, dense_copy, DIM, HybridConfig::default()).unwrap();
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let sp_built = SpellCorrector::from_docs(&ctx.docs);
    let spell_build_ms = t0.elapsed().as_secs_f64() * 1e3;
    let bm_json = serde_json::to_vec(built.bm25()).unwrap().len();
    let sp_json = serde_json::to_vec(&sp_built).unwrap().len();
    let overhead = (hybrid.0 - dense_only.0) / dense_only.0 * 100.0;
    log.push_str(&format!(
        "  BENCH corpus A ({} docs, dim {}), single thread, {} queries x3 passes:\n    dense scan+top8: mean {:.1} us (p50 {:.1}, p95 {:.1})\n    hybrid search (dense scan + BM25 + 2 z + fuse + top8): mean {:.1} us (p50 {:.1}, p95 {:.1}) = {:+.1}% vs dense\n    BM25 scoring alone: mean {:.1} us; spell correct(): mean {:.1} us/query\n    index build {:.0} ms (BM25 + copy of dense); spell build {:.0} ms\n    bytes: dense matrix {} ({:.2} MB), BM25 payload {} ({:.2} MB, +{:.1}% over dense), BM25 serde_json {} ({:.2} MB), spell serde_json {} ({:.2} MB), vocab {} terms\n",
        ctx.w.ndoc, DIM, n, dense_only.0, dense_only.1, dense_only.2, hybrid.0, hybrid.1, hybrid.2, overhead, bm25_only.0, spell.0, build_ms, spell_build_ms,
        built.bytes_dense(), built.bytes_dense() as f64 / 1e6, built.bytes_bm25(), built.bytes_bm25() as f64 / 1e6, 100.0 * built.bytes_bm25() as f64 / built.bytes_dense() as f64,
        bm_json, bm_json as f64 / 1e6, sp_json, sp_json as f64 / 1e6, built.bm25().vocab_size()
    ));
    let _ = sink;
    json!({"docs": ctx.w.ndoc, "dim": DIM, "n_queries": n, "passes": 3,
        "dense_scan_top8_us": {"mean": dense_only.0, "p50": dense_only.1, "p95": dense_only.2},
        "hybrid_search_us": {"mean": hybrid.0, "p50": hybrid.1, "p95": hybrid.2},
        "bm25_score_us": {"mean": bm25_only.0, "p50": bm25_only.1, "p95": bm25_only.2},
        "spell_correct_us": {"mean": spell.0, "p50": spell.1, "p95": spell.2},
        "hybrid_overhead_pct_vs_dense": overhead, "index_build_ms": build_ms, "spell_build_ms": spell_build_ms,
        "bytes": {"dense": built.bytes_dense(), "bm25_payload": built.bytes_bm25(), "bm25_pct_of_dense": 100.0 * built.bytes_bm25() as f64 / built.bytes_dense() as f64, "bm25_serde_json": bm_json, "spell_serde_json": sp_json, "bm25_vocab_terms": built.bm25().vocab_size()}})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let outp = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/h16_hybrid.json".into());
    let t0 = Instant::now();
    let mut log = String::new();
    let picks: Value = serde_json::from_slice(&std::fs::read("results/h13_picks.json").unwrap()).unwrap();
    let z_alpha = [read_f(&picks["z_alpha"][0]), read_f(&picks["z_alpha"][1])];
    assert!((z_alpha[0] - 0.7).abs() < 1e-12 && (z_alpha[1] - 0.7).abs() < 1e-12, "h13 picks changed: {z_alpha:?}");
    let reference: Value = serde_json::from_slice(&std::fs::read("results/h13_retrieval.json").unwrap()).unwrap();
    let a = load_ctx("A", z_alpha[1]);
    let b = load_ctx("B", z_alpha[1]);
    log.push_str(&format!("loaded A {} docs, B {} docs ({:.1}s); library alpha {} (h13 pick, hybz_s = z-fusion on the spell-corrected texts)\n", a.w.ndoc, b.w.ndoc, t0.elapsed().as_secs_f64(), z_alpha[1]));
    let mut par = Parity { n: 0, max_abs: 0.0, worst: String::new() };

    log.push_str("\n== PARITY 1: doc-level levels of hybz_s vs results/h13_retrieval.json ==\n");
    let (s_tune, s_held, s_conf) = (seeds(1, 8), seeds(31, 60), seeds(61, 90));
    let mut sa: Vec<u64> = s_tune.clone();
    sa.extend(&s_held);
    sa.extend(&s_conf);
    let eva = build_evals(&a, &sa, &[1]);
    log.push_str(&format!("  A evals {} ({:.1}s)\n", eva.len(), t0.elapsed().as_secs_f64()));
    levels_parity("A tune 1-8", "tune_A_1_8", &a, &eva, &s_tune, &reference, 1, "hybz_s", &mut par, &mut log);
    levels_parity("A held 31-60", "held_A_31_60", &a, &eva, &s_held, &reference, 1, "hybz_s", &mut par, &mut log);
    levels_parity("A confirm 61-90", "confirm_A_61_90", &a, &eva, &s_conf, &reference, 1, "hybz_s", &mut par, &mut log);
    drop(eva);
    let evb = build_evals(&b, &s_conf, &[1]);
    log.push_str(&format!("  B evals {} ({:.1}s)\n", evb.len(), t0.elapsed().as_secs_f64()));
    levels_parity("B confirm 61-90", "confirm_B_61_90", &b, &evb, &s_conf, &reference, 1, "hybz_s", &mut par, &mut log);
    drop(evb);
    // raw flavour too (hybz_raw: z alpha .7 on the raw text)
    let eva_raw = build_evals(&a, &s_conf, &[0]);
    levels_parity("A confirm 61-90 (raw flavour)", "confirm_A_61_90", &a, &eva_raw, &s_conf, &reference, 0, "hybz_raw", &mut par, &mut log);
    drop(eva_raw);
    let pass1 = par.max_abs <= TOL;
    log.push_str(&format!("  PARITY 1 {}: {} levels, max |diff| {:.3e} (tolerance {:.0e}); worst: {}\n", if pass1 { "PASS" } else { "FAIL" }, par.n, par.max_abs, TOL, par.worst));

    log.push_str("\n== PARITY 2: top-8 lists (results/h13_top_{A,B}.bin, hybz base) ==\n");
    let ta = tops_parity("A", &a, &sa, &mut log);
    let tb = tops_parity("B", &b, &s_conf, &mut log);
    let pass2 = ta["differ"] == 0 && tb["differ"] == 0 && ta["missing_in_reference"] == 0 && tb["missing_in_reference"] == 0;
    log.push_str(&format!("  PARITY 2 {}\n", if pass2 { "PASS" } else { "FAIL" }));

    log.push_str("\n== PARITY 3: Rust SpellCorrector vs python Speller texts in the fixtures ==\n");
    let spa = spell_parity("A", &a, &mut log);
    let spb = spell_parity("B", &b, &mut log);
    let pass3 = [&spa, &spb].iter().all(|v| ["sp1", "sp2"].iter().all(|k| v[*k]["identical"] == v[*k]["total"]));
    log.push_str(&format!("  PARITY 3 {}\n", if pass3 { "PASS" } else { "FAIL (see mismatches; divergence documented in the board entry)" }));

    log.push_str("\n== BENCH ==\n");
    let bn = bench(&a, &mut log);

    let doc = json!({"parity1_levels": {"pass": pass1, "levels_compared": par.n, "max_abs_diff": par.max_abs, "tolerance": TOL, "worst": par.worst},
        "parity2_tops": {"pass": pass2, "A": ta, "B": tb}, "parity3_spell": {"pass": pass3, "A": spa, "B": spb}, "bench": bn, "alpha": z_alpha[1], "wall_s": t0.elapsed().as_secs_f64()});
    std::fs::write(&outp, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    let _ = std::fs::write(outp.replace(".json", ".txt"), &log);
    println!("{log}");
    println!("wrote {outp} in {:.1}s", t0.elapsed().as_secs_f64());
    if !(pass1 && pass2) {
        std::process::exit(1);
    }
}
