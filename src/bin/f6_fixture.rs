//! F6 part 1: prove the MiniLM vectors go through the embedding abstraction.
//!
//!  --write   : write results/f6_minilm_fixture.json (embedding.rs TableEmbedding JSON: {"name","dim","vectors":{text:[..]}})
//!              from results/f5_corpus.json + results/f5_embeddings.f32 (f5 concept texts exact f32; the 2100 one-off
//!              distractor texts rounded to 6 decimals to stay under ~10 MB).
//!  (default) : load the fixture THROUGH the abstraction (feature real-embed: EmbeddingConfig::Real{source}.build();
//!              else TableEmbedding::from_path), compare vectors with the raw f5 f32 file, re-run f5's retrieval-layer
//!              P/R/F1 harness (worlds mini-q, mini-subj3, mini-q-mixed50) on the fixture vectors AND on the raw f5
//!              vectors AND against the published results/f5_retrieval.json rows, and measure latency / memory.
//!
//! cargo run --release --offline --features real-embed --bin f6_fixture -- [--write] [--held 31-60] [--out results/f6_fixture.json]

#[path = "f6_common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use common::*;
use neural_mesh::embedding::{CachedEmbedding, Embedding, HashEmbedding, TableEmbedding};
use neural_mesh::knowledge::{FeatMode, Knowledge, KnowledgeConfig};
use rayon::prelude::*;
use serde_json::{json, Value};

fn write_fixture() {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string("results/f5_corpus.json").unwrap()).unwrap();
    let bytes = std::fs::read("results/f5_embeddings.f32").unwrap();
    let items = meta["items"].as_array().unwrap();
    assert_eq!(bytes.len(), items.len() * DIM * 4);
    let mut s = String::with_capacity(12_000_000);
    s.push_str("{\"name\":\"minilm-l6\",\"dim\":384,\"note\":\"all-MiniLM-L6-v2 (ONNX, local ruvector copy), mean-pooled, L2-normalised, 384-d, computed OFFLINE by py/f5_embed.py; keys = exact texts of results/f5_corpus.json (SYNTHETIC templated corpus). Concept texts are exact f32 (shortest round-trip), the 2100 one-off distractor texts are rounded to 6 decimals.\",\"model_sha256\":");
    s.push_str(&serde_json::to_string(&meta["model_sha256"]).unwrap());
    s.push_str(",\"tokenizer_sha256\":");
    s.push_str(&serde_json::to_string(&meta["tokenizer_sha256"]).unwrap());
    s.push_str(",\"vectors\":{");
    let mut seen = std::collections::HashSet::new();
    for (i, it) in items.iter().enumerate() {
        let text = it["text"].as_str().unwrap();
        assert!(seen.insert(text.to_string()), "duplicate text {text:?}");
        let exact = it["kind"].as_str().unwrap() != "dist";
        if i > 0 {
            s.push(',');
        }
        s.push_str(&serde_json::to_string(text).unwrap());
        s.push_str(":[");
        for j in 0..DIM {
            let v = f32::from_le_bytes(bytes[(i * DIM + j) * 4..(i * DIM + j) * 4 + 4].try_into().unwrap());
            if j > 0 {
                s.push(',');
            }
            if exact {
                s.push_str(&format!("{v}"));
            } else {
                s.push_str(&format!("{v:.6}"));
            }
        }
        s.push(']');
    }
    s.push_str("}}");
    std::fs::write(FIXTURE, &s).unwrap();
    println!("wrote {FIXTURE}: {} texts, {} bytes ({:.2} MB)", items.len(), s.len(), s.len() as f64 / 1e6);
}

/// raw f5 vectors (mini) as a Corpus (labels/hash from the fixture-path corpus, vectors from the f32 file)
fn raw_corpus() -> Corpus {
    let mut c = load_corpus();
    let bytes = std::fs::read("results/f5_embeddings.f32").unwrap();
    let n5 = bytes.len() / (DIM * 4);
    for i in 0..n5 {
        c.mini[i] = (0..DIM).map(|j| f32::from_le_bytes(bytes[(i * DIM + j) * 4..(i * DIM + j) * 4 + 4].try_into().unwrap())).collect();
    }
    c
}

/// f5-compatible retrieval row: nbr P/R/F1 + anchor P/R/F1 + frag + spur + contam (means over seeds)
fn rows_for(f: &Corpus, sp: &Spec, k: usize, sim: f32, ctr: bool, seeds: &[u64]) -> [f64; 9] {
    let rs: Vec<(Nbr, Anc)> = seeds
        .par_iter()
        .map(|&s| {
            let cw = CtxWorld::new_in(s, sp, f);
            let m = Method { anchor: AnchorCfg::Leader { k, sim }, center: ctr };
            (nbr_metrics(&cw, DIM, k, sim, ctr, 0), anchor_metrics(&cw, DIM, &m, false))
        })
        .collect();
    let mu = |g: &dyn Fn(&(Nbr, Anc)) -> f64| mean_of(rs.iter().map(g));
    [mu(&|x| x.0.p), mu(&|x| x.0.r), mu(&|x| x.0.f1), mu(&|x| x.1.p), mu(&|x| x.1.r), mu(&|x| x.1.f1), mu(&|x| x.1.frag), mu(&|x| x.0.spurious), mu(&|x| x.1.contam)]
}

const ROW_KEYS: [&str; 9] = ["nbr_p", "nbr_r", "nbr_f1", "anchor_p", "anchor_r", "anchor_f1", "frag", "spurious_dist", "contam"];

fn bench<F: FnMut(usize)>(label: &str, n: usize, mut f: F) -> Value {
    for i in 0..n.min(2000) {
        f(i);
    }
    let t0 = Instant::now();
    for i in 0..n {
        f(i);
    }
    let ns = t0.elapsed().as_nanos() as f64 / n as f64;
    println!("  {label:<46} {ns:>10.0} ns/embed");
    json!({"what": label, "ns_per_embed": ns})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--write") {
        write_fixture();
        return;
    }
    let held = parse_range(&args, "--held", (31, 60));
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/f6_fixture.json".into());
    let t0 = Instant::now();
    println!("f6_fixture: real-embed feature = {} (path = {})", real_embed_feature(), if real_embed_feature() { "EmbeddingConfig::Real{source}.build()" } else { "TableEmbedding::from_path (feature off)" });

    // ---- load timing + vector equivalence -----------------------------------------------------
    let tl = Instant::now();
    let emb = load_embedder(FIXTURE, false);
    let load_ms = tl.elapsed().as_secs_f64() * 1e3;
    println!("fixture load via abstraction: {load_ms:.0} ms, name {}, dim {}", emb.name(), emb.dim());
    let f = corpus();
    let raw = raw_corpus();
    let n5 = std::fs::read("results/f5_embeddings.f32").unwrap().len() / (DIM * 4);
    let (mut exact, mut tot_c, mut maxd_dist, mut maxd_c) = (0usize, 0usize, 0f32, 0f32);
    for i in 0..n5 {
        let d = f.mini[i].iter().zip(&raw.mini[i]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        if f.kind[i] == Kind::Dist {
            maxd_dist = maxd_dist.max(d);
        } else {
            tot_c += 1;
            maxd_c = maxd_c.max(d);
            if f.mini[i].iter().zip(&raw.mini[i]).all(|(a, b)| a.to_bits() == b.to_bits()) {
                exact += 1;
            }
        }
    }
    println!("concept texts bit-identical to f5 f32 file: {exact}/{tot_c} (max |diff| {maxd_c:.2e}); distractors max |diff| {maxd_dist:.2e}");
    // normalised-by-the-abstraction variant (EmbeddingConfig default normalize=true)
    let tn = TableEmbedding::from_path(FIXTURE, true, false).unwrap();
    let mut maxd_norm = 0f32;
    for i in 0..n5 {
        let v = tn.embed(&f.texts[i]);
        maxd_norm = maxd_norm.max(v.iter().zip(&raw.mini[i]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
    }
    println!("TableEmbedding(normalize=true) max |diff| vs f5 f32: {maxd_norm:.2e}");

    // ---- retrieval-layer reproduction -----------------------------------------------------------
    let f5: Value = serde_json::from_str(&std::fs::read_to_string("results/f5_retrieval.json").unwrap()).unwrap();
    let f5rows = f5["rows"].as_array().unwrap();
    let f5_held = format!("{}-{}", f5["held"][0], f5["held"][1]);
    println!("\nretrieval layer (held {}-{}); f5 published rows are for held {f5_held}", held[0], held[held.len() - 1]);
    println!("{:<14} {:>3} {:>4} {:>5} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} | {:>5} {:>5} {:>6} | max|fix-raw| max|fix-f5json|", "world", "k", "sim", "ctr", "nbrP", "nbrR", "nbrF1", "ancP", "ancR", "ancF1", "frag", "spur", "contam");
    let mut repro = vec![];
    let mut worst_raw = 0f64;
    let mut worst_json = 0f64;
    let same_held = held == (31..=60).collect::<Vec<u64>>() && f5_held == "31-60";
    for wname in ["mini-q", "mini-subj3", "mini-q-mixed50"] {
        let sp = spec(wname);
        for &sim in &[0.5f32, 0.6, 0.7] {
            for &ctr in &[false, true] {
                let k = 8usize;
                let a = rows_for(f, &sp, k, sim, ctr, &held);
                let b = rows_for(&raw, &sp, k, sim, ctr, &held);
                let dr = (0..9).filter(|&i| a[i].is_finite() && b[i].is_finite()).map(|i| (a[i] - b[i]).abs()).fold(0f64, f64::max);
                let mut dj = f64::NAN;
                if same_held {
                    if let Some(row) = f5rows.iter().find(|r| r["world"] == wname && r["k"] == k && (r["sim_min"].as_f64().unwrap() - sim as f64).abs() < 1e-6 && r["center"] == ctr) {
                        dj = (0..9).filter(|&i| a[i].is_finite()).map(|i| (a[i] - row[ROW_KEYS[i]].as_f64().unwrap_or(f64::NAN)).abs()).filter(|x| x.is_finite()).fold(0f64, f64::max);
                    }
                }
                worst_raw = worst_raw.max(if wname == "mini-q-mixed50" { 0.0 } else { dr });
                if dj.is_finite() && wname != "mini-q-mixed50" {
                    worst_json = worst_json.max(dj);
                }
                println!(
                    "{:<14} {:>3} {:>4} {:>5} | {:>6.4} {:>6.4} {:>6.4} | {:>6.4} {:>6.4} {:>6.4} | {:>5.2} {:>5.2} {:>6.2} | {:>9.2e} {:>11.2e}",
                    wname, k, sim, ctr, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], dr, dj
                );
                repro.push(json!({"world": wname, "k": k, "sim_min": sim, "center": ctr, "fixture": a, "raw_f5_vectors": b, "max_abs_diff_fixture_vs_raw": dr, "max_abs_diff_fixture_vs_f5_json": dj, "keys": ROW_KEYS}));
            }
        }
    }
    println!("worst |fixture - raw| over mini-q/mini-subj3 (exact vectors): {worst_raw:.2e}; worst |fixture - f5 json| : {worst_json:.2e}");

    // ---- latency ------------------------------------------------------------------------------
    println!("\nper-embed latency (single thread, release):");
    let texts: Vec<String> = (0..f.texts.len()).filter(|&i| f.kind[i] != Kind::Dist).map(|i| f.texts[i].clone()).collect();
    let mut lat = vec![];
    {
        let t = TableEmbedding::from_path(FIXTURE, false, false).unwrap();
        let mut sink = 0f32;
        lat.push(bench("TableEmbedding lookup (HashMap get + clone 384xf32)", 400_000, |i| sink += t.embed(&texts[i % texts.len()])[0]));
        let c = CachedEmbedding::new(TableEmbedding::from_path(FIXTURE, false, false).unwrap(), 4096);
        for tx in &texts {
            c.embed(tx);
        }
        lat.push(bench("CachedEmbedding(Table) HIT", 400_000, |i| sink += c.embed(&texts[i % texts.len()])[0]));
        let c0 = CachedEmbedding::new(TableEmbedding::from_path(FIXTURE, false, false).unwrap(), 0);
        lat.push(bench("CachedEmbedding(Table) cap0 (always miss)", 200_000, |i| sink += c0.embed(&texts[i % texts.len()])[0]));
        lat.push(bench("Arc<dyn Embedding> from EmbeddingConfig (Real/Table)", 400_000, |i| sink += emb.embed(&texts[i % texts.len()])[0]));
        let h = HashEmbedding::new(DIM);
        lat.push(bench("HashEmbedding(384).embed (whitespace words)", 200_000, |i| sink += h.embed(&texts[i % texts.len()])[0]));
        let wh = WordHash { dim: DIM };
        lat.push(bench("WordHash(384).embed (f5 hash source)", 200_000, |i| sink += wh.embed(&texts[i % texts.len()])[0]));
        std::hint::black_box(sink);
        println!("  (miss path of TableEmbedding with MissPolicy::Zero allocates a zero vector; not benchmarked)");
    }
    let py: Value = std::fs::read_to_string("results/f6_py_latency.json").ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null);
    println!("  offline python embed (onnxruntime 1.23.2 CPU, 2 intra-op threads, incl. tokenisation): single {:.2} ms/text, batch-32 {:.2} ms/text   [results/f6_py_latency.json]", py["single_ms_per_text_new"].as_f64().unwrap_or(f64::NAN), py["batch32_ms_per_text"].as_f64().unwrap_or(f64::NAN));

    // ---- memory ---------------------------------------------------------------------------------
    let fsize = std::fs::metadata(FIXTURE).map(|m| m.len()).unwrap_or(0);
    let psize = std::fs::metadata(PARA_FIXTURE).map(|m| m.len()).unwrap_or(0);
    let payload: usize = f.texts.iter().take(n5).map(|t| t.len()).sum::<usize>() + n5 * DIM * 4;
    let mut kc = KnowledgeConfig::new(DIM);
    kc.feat = FeatMode::Hidden;
    let mut kn = Knowledge::new_shared(kc);
    let cw = CtxWorld::new(31, &spec("mini-q"));
    for e in &cw.emb {
        kn.ingest(e);
    }
    let mut kc64 = KnowledgeConfig::new(64);
    kc64.feat = FeatMode::Hidden;
    let mut kn64 = Knowledge::new_shared(kc64);
    let h64 = HashEmbedding::new(64);
    for t in 0..2000 {
        kn64.ingest(&h64.embed(&f.texts[t % f.texts.len()]));
    }
    println!("\nmemory: bytes per vector = {} (384 x f32); fixture {} B = {:.2} MB for {} texts ({:.0} B/text on disk as JSON); para fixture {:.2} MB for {} texts", DIM * 4, fsize, fsize as f64 / 1e6, n5, fsize as f64 / n5 as f64, psize as f64 / 1e6, f.texts.len() - n5);
    println!("  in-memory table payload (text bytes + vectors, excl. HashMap overhead ~40 B/entry): {:.2} MB; Knowledge store for 2000 records: {} B at 384-d ({:.0} B/record) vs {} B at 64-d hash ({:.0} B/record)", payload as f64 / 1e6, kn.bytes(), kn.bytes() as f64 / 2000.0, kn64.bytes(), kn64.bytes() as f64 / 2000.0);

    let _: Option<Arc<dyn Embedding>> = None;
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for k in &f.kind {
        *counts.entry(match k { Kind::Q => "q", Kind::Subj => "subj", Kind::Dist => "dist", Kind::NewSubj => "newsubj", Kind::NewQ => "newq", Kind::Typo => "typo" }).or_insert(0) += 1;
    }
    let doc = json!({
        "lab": "f6_fixture", "real_embed_feature": real_embed_feature(), "held": [held[0], held[held.len() - 1]],
        "fixture": {"path": FIXTURE, "bytes": fsize, "n_texts": n5, "dim": DIM, "bytes_per_vector_mem": DIM * 4, "load_ms": load_ms, "para_path": PARA_FIXTURE, "para_bytes": psize},
        "vector_equivalence": {"concept_texts_bit_identical": exact, "concept_texts": tot_c, "concept_max_abs_diff": maxd_c, "distractor_max_abs_diff_6dec": maxd_dist, "normalize_true_max_abs_diff": maxd_norm},
        "retrieval_reproduction": repro, "worst_abs_diff_fixture_vs_raw_exact_worlds": worst_raw, "worst_abs_diff_fixture_vs_f5_json_exact_worlds": worst_json,
        "latency_ns": lat, "python_latency": py,
        "memory": {"table_payload_bytes": payload, "knowledge_2000rec_384d_bytes": kn.bytes(), "knowledge_2000rec_64d_bytes": kn64.bytes()},
        "kinds": counts, "elapsed_s": t0.elapsed().as_secs_f64()
    });
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out} in {:.1}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_reproduces_f5_retrieval_and_vectors() {
        let f = corpus();
        let raw = raw_corpus();
        let n5 = std::fs::read("results/f5_embeddings.f32").unwrap().len() / (DIM * 4);
        for i in 0..n5 {
            if f.kind[i] != Kind::Dist {
                assert!(f.mini[i].iter().zip(&raw.mini[i]).all(|(a, b)| a.to_bits() == b.to_bits()), "concept vector {i} not bit-identical");
            } else {
                assert!(f.mini[i].iter().zip(&raw.mini[i]).all(|(a, b)| (a - b).abs() < 1e-6));
            }
        }
        for w in ["mini-q", "mini-subj3"] {
            let sp = spec(w);
            for &(sim, ctr) in &[(0.5f32, false), (0.6, true)] {
                let a = rows_for(f, &sp, 8, sim, ctr, &[31, 32, 33]);
                let b = rows_for(&raw, &sp, 8, sim, ctr, &[31, 32, 33]);
                for i in 0..9 {
                    if a[i].is_finite() || b[i].is_finite() {
                        assert_eq!(a[i].to_bits(), b[i].to_bits(), "{w} sim {sim} ctr {ctr} metric {i}: {} vs {}", a[i], b[i]);
                    }
                }
            }
        }
        // the abstraction path really is the table: unknown text -> zero vector, known -> the stored vector
        let e = load_embedder(FIXTURE, false);
        assert_eq!(e.dim(), DIM);
        assert_eq!(e.embed(&f.texts[0]), f.mini[0]);
        assert!(e.embed("this text is not in the fixture").iter().all(|&x| x == 0.0));
    }
}
