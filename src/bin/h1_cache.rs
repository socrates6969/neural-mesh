//! H1 / part 3: QUERY-EMBEDDING + RESULT CACHE with invalidation on memory writes.
//! Arms (same query stream, same index): none | emb (MemoEmbedding) | emb+res (MemoEmbedding + CachedIndex).
//! Streams: repeat-heavy (12 concepts, ~93 % repeats), zipf, mixed50, NO-REPEAT (unique distractors) -- with and
//! without interleaved memory writes. Index: FlatIndex (exact; results must be identical across arms) and
//! HnswIndex (patched crate; cached == inner on every verified query).
//! Real-encoder cost is modelled at 2.4 ms/text (results/f6_py_latency.json); the measured part is real CPU.
//!   cargo run --release --offline --bin h1_cache
//! SYNTHETIC f5/f6 corpus (templated text, MiniLM table vectors); not real traffic.
#[path = "f6_common/mod.rs"]
mod f6c;

use std::sync::Arc;
use std::time::Instant;

use f6c::*;
use neural_mesh::embedding::{MemoEmbedding, TableEmbedding};
use neural_mesh::memory::{CachedIndex, FlatIndex, HnswIndex, HnswParams, MemoryIndex};
use neural_mesh::router::Rng;

const ENC_MS: f64 = 2.4;
const NQ: usize = 4000;
const K: usize = 5;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    None,
    Emb,
    EmbRes,
    EmbResSel,
}
fn an(a: Arm) -> &'static str {
    match a {
        Arm::None => "none",
        Arm::Emb => "emb-cache",
        Arm::EmbRes => "emb+result-cache",
        Arm::EmbResSel => "emb+result(selective)",
    }
}

struct Stream {
    name: String,
    texts: Vec<usize>, // corpus text index per query
}

fn streams(f: &Corpus, writes_pool: &std::collections::HashSet<usize>) -> Vec<Stream> {
    let mut r = Rng::new(20251004);
    let mut v = vec![];
    // repeat-heavy: 12 concepts, question paraphrases
    let mut t = vec![];
    for _ in 0..NQ {
        let c = r.below(f.nconc.min(12));
        t.push(f.q[c][r.below(f.q[c].len())]);
    }
    v.push(Stream { name: "repeat-heavy(12 concepts)".into(), texts: t });
    // zipf over the memory texts
    let pool: Vec<usize> = (0..f.texts.len()).filter(|i| !writes_pool.contains(i)).collect();
    let mut cdf = vec![];
    let mut s = 0.0;
    for i in 0..pool.len() {
        s += 1.0 / ((i + 1) as f64).powf(1.1);
        cdf.push(s);
    }
    let mut t = vec![];
    for _ in 0..NQ {
        let u = r.f64() * s;
        let i = cdf.partition_point(|&x| x < u).min(pool.len() - 1);
        t.push(pool[i]);
    }
    v.push(Stream { name: "zipf(s=1.1)".into(), texts: t });
    // mixed50: half repeat-heavy, half unique distractors
    let mut dp = f.dist.clone();
    for i in (1..dp.len()).rev() {
        let j = r.below(i + 1);
        dp.swap(i, j);
    }
    let mut t = vec![];
    let mut di = 0;
    for _ in 0..NQ {
        if r.f64() < 0.5 {
            let c = r.below(f.nconc.min(12));
            t.push(f.q[c][r.below(f.q[c].len())]);
        } else {
            t.push(dp[di % dp.len()]);
            di += 1;
        }
    }
    v.push(Stream { name: "mixed50".into(), texts: t });
    // no repeat at all: distinct distractor texts
    let t: Vec<usize> = dp.iter().copied().take(NQ.min(dp.len())).collect();
    v.push(Stream { name: "NO-REPEAT(unique)".into(), texts: t });
    v
}

struct Res {
    wall_s: f64,
    enc_calls: u64,
    searches: u64,
    results: Vec<Vec<(u64, u32)>>,
    mismatches: usize,
    cache_bytes: usize,
    emb_hits: u64,
    res_hits: u64,
    rec_cached: f64,
    rec_inner: f64,
}

fn run(f: &Corpus, table: &Arc<TableEmbedding>, mem_ids: &[usize], write_ids: &[usize], st: &Stream, arm: Arm, hnsw: bool, wr: usize, verify: bool) -> Res {
    let dim = DIM;
    let n0 = mem_ids.len();
    let ids: Vec<u64> = (0..n0 as u64).collect();
    let flat: Vec<f32> = mem_ids.iter().flat_map(|&i| f.mini[i].iter().copied()).collect();
    let mk_inner = || -> Box<dyn MemoryIndex> {
        if hnsw {
            let mut h = HnswIndex::new(dim, HnswParams { max_elements: n0 + write_ids.len() + 8, ..Default::default() });
            h.add_batch(&ids, &flat);
            Box::new(h)
        } else {
            let mut x = FlatIndex::new(dim);
            x.add_batch(&ids, &flat);
            Box::new(x)
        }
    };
    // arms 0/1: cap 0 pass-through CachedIndex (so the type is identical; zero caching); arm 2: cap 2048
    let cap = if arm == Arm::EmbRes || arm == Arm::EmbResSel { 2048 } else { 0 };
    let mut idx = CachedIndex::new(mk_inner(), cap).selective(arm == Arm::EmbResSel);
    let mut truth = FlatIndex::new(dim);
    truth.add_batch(&ids, &flat);
    let (mut rc, mut ri) = (0.0f64, 0.0f64);
    let ecap = if arm == Arm::None { 0 } else { 4096 };
    let memo = MemoEmbedding::new(table.clone(), ecap);
    let mut results = Vec::with_capacity(st.texts.len());
    let mut mism = 0usize;
    let mut next_w = 0usize;
    let mut verify_time = 0.0;
    let t0 = Instant::now();
    for (qi, &ti) in st.texts.iter().enumerate() {
        if wr > 0 && qi > 0 && qi % wr == 0 && next_w < write_ids.len() {
            let wi = write_ids[next_w];
            idx.add((n0 + next_w) as u64, &f.mini[wi]);
            if verify {
                truth.add((n0 + next_w) as u64, &f.mini[wi]);
            }
            next_w += 1;
        }
        let q = memo.embed_shared(&f.texts[ti]);
        let r = idx.search_shared(&q, K);
        if !verify {
            results.push(r.iter().map(|x| (x.0, x.1.to_bits())).collect());
        } else {
            let v0 = Instant::now();
            let direct = idx.inner().search(&q, K);
            let exact = truth.search(&q, K);
            rc += neural_mesh::memory::recall_at_k(&exact, &r, K);
            ri += neural_mesh::memory::recall_at_k(&exact, &direct, K);
            // conservative (clear-all) mode must return exactly what the inner index returns now; the selective mode may
            // return a still-valid earlier answer (exact backends: identical; HNSW: judged by recall vs exact instead)
            if arm == Arm::EmbRes && direct.iter().map(|x| (x.0, x.1.to_bits())).ne(r.iter().map(|x| (x.0, x.1.to_bits()))) {
                mism += 1;
            }
            if arm == Arm::EmbResSel && !hnsw && direct.iter().map(|x| (x.0, x.1.to_bits())).ne(r.iter().map(|x| (x.0, x.1.to_bits()))) {
                mism += 1;
            }
            verify_time += v0.elapsed().as_secs_f64();
            results.push(r.iter().map(|x| (x.0, x.1.to_bits())).collect());
        }
    }
    let wall_s = t0.elapsed().as_secs_f64() - verify_time;
    Res {
        wall_s,
        enc_calls: memo.encoder_calls(),
        searches: idx.misses(),
        results,
        mismatches: mism,
        cache_bytes: memo.bytes() + idx.bytes().saturating_sub(idx.inner().bytes()),
        emb_hits: memo.hits(),
        res_hits: idx.hits(),
        rec_cached: rc / st.texts.len() as f64,
        rec_inner: ri / st.texts.len() as f64,
    }
}

fn main() {
    let f = corpus();
    let table = Arc::new(TableEmbedding::from_pairs("minilm-fixture", Some(DIM), f.texts.iter().cloned().zip(f.mini.iter().cloned()), false, false).unwrap());
    // memory = all texts except a write pool of 300 held-out ones (taken from the tail of the distractor pool so streams never query them)
    let write_ids: Vec<usize> = f.dist.iter().rev().take(300).copied().collect();
    let wset: std::collections::HashSet<usize> = write_ids.iter().copied().collect();
    let mem_ids: Vec<usize> = (0..f.texts.len()).filter(|i| !wset.contains(i)).collect();
    let sts = streams(f, &wset);
    println!("h1_cache: memory N={} d={}, {} queries/stream, k={K}; encoder cost model {ENC_MS} ms/text; synthetic corpus", mem_ids.len(), DIM, NQ);
    let mut rows = vec![];
    let mut all_equal = true;
    let mut total_mismatch = 0usize;
    let mut min_recall_gap = 0.0f64;
    for hnsw in [false, true] {
        let ix = if hnsw { "HNSW(patched)" } else { "Flat" };
        for wr in [0usize, 50] {
            println!("--- index {ix}, writes: {}", if wr == 0 { "none".to_string() } else { format!("1 per {wr} queries (~{} writes)", NQ / wr) });
            println!("{:<26} {:<22} {:>7} {:>8} {:>8} {:>8} {:>9} {:>10} {:>10} {:>9}", "stream", "arm", "enc/q", "emb hit", "res hit", "srch/q", "meas us/q", "model ms/q", "saved%", "cacheKB");
            for st in &sts {
                let n = st.texts.len() as f64;
                let mut base_model = 0.0;
                let mut base_res: Option<Vec<Vec<(u64, u32)>>> = None;
                for arm in [Arm::None, Arm::Emb, Arm::EmbRes, Arm::EmbResSel] {
                    // median of 3 timing reps; verification pass separate (not timed)
                    let mut walls = vec![];
                    let mut last = None;
                    for _ in 0..3 {
                        let r = run(f, &table, &mem_ids, &write_ids, st, arm, hnsw, wr, false);
                        walls.push(r.wall_s);
                        last = Some(r);
                    }
                    walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let r = last.unwrap();
                    let wall = walls[1];
                    let model = wall / n * 1e3 + r.enc_calls as f64 / n * ENC_MS;
                    if arm == Arm::None {
                        base_model = model;
                    }
                    let saved = 100.0 * (1.0 - model / base_model);
                    // exactness: Flat is deterministic -> every arm must return the same (id, score bits) on every query
                    if !hnsw {
                        match &base_res {
                            None => base_res = Some(r.results.clone()),
                            Some(b) => {
                                if *b != r.results {
                                    all_equal = false;
                                }
                            }
                        }
                    }
                    let (mism, rcv, riv) = if arm == Arm::EmbRes || arm == Arm::EmbResSel {
                        let v = run(f, &table, &mem_ids, &write_ids, st, arm, hnsw, wr, true);
                        total_mismatch += v.mismatches;
                        (v.mismatches, v.rec_cached, v.rec_inner)
                    } else {
                        (0, f64::NAN, f64::NAN)
                    };
                    if hnsw && (arm == Arm::EmbRes || arm == Arm::EmbResSel) {
                        min_recall_gap = min_recall_gap.min(rcv - riv);
                    }
                    println!("{:<26} {:<22} {:>7.3} {:>8.3} {:>8.3} {:>8.3} {:>9.2} {:>10.4} {:>10.1} {:>9.0}{}", st.name, an(arm), r.enc_calls as f64 / n, r.emb_hits as f64 / n, r.res_hits as f64 / n, r.searches as f64 / n, wall / n * 1e6, model, saved, r.cache_bytes as f64 / 1024.0, if mism > 0 { format!("  STALE x{mism}") } else { String::new() });
                    rows.push(serde_json::json!({"index": ix, "writes_per_queries": wr, "stream": st.name, "arm": an(arm), "enc_per_query": r.enc_calls as f64 / n, "emb_hit_rate": r.emb_hits as f64 / n, "res_hit_rate": r.res_hits as f64 / n, "inner_searches_per_query": r.searches as f64 / n, "measured_us_per_query": wall / n * 1e6, "modelled_ms_per_query": model, "saved_pct_vs_none": saved, "cache_kb": r.cache_bytes as f64 / 1024.0, "stale_mismatches": mism, "recall_cached_vs_exact": rcv, "recall_inner_vs_exact": riv}));
                }
            }
        }
    }
    println!("Flat: all arms return identical (id, score-bits) lists on every query (incl. interleaved writes): {all_equal}");
    println!("HNSW: worst (cached-answer recall@{K} minus inner-answer recall@{K}) over streams/arms = {min_recall_gap:+.4} (vs exact flat truth at the same write state)");
    println!("verification pass (cached result vs inner index answer, every query, both indexes, with/without writes): {total_mismatch} stale/mismatching results");
    let out = serde_json::json!({"rows": rows, "flat_all_arms_identical": all_equal, "stale_mismatches_total": total_mismatch, "enc_ms_model": ENC_MS, "n_memory": mem_ids.len(), "nq": NQ, "k": K, "note": "synthetic f5/f6 corpus; modelled = measured CPU + encoder_calls*2.4ms; shared loaded box"});
    std::fs::write("results/h1_cache.json", serde_json::to_string_pretty(&out).unwrap()).unwrap();
    if !all_equal || total_mismatch > 0 {
        std::process::exit(1);
    }
}
