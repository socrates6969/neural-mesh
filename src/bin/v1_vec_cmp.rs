//! v1 (vector-lab): neural-mesh store arms on a corpus/query split, for comparison with ruvector.
//! Usage: v1_vec_cmp <corpus.f32> <queries.f32> <out.json>   (384-d, row-major f32, L2-normalised)
//! Arms: flat-int8-rr4, pca192-int8-rr4 (PCA fit on corpus, rerank in 192-d), hnsw (M16 efc200 ef64).
//! Outputs per arm: top-10 ids per query, p50/p95 single-query latency (us), build seconds, bytes.
use neural_mesh::flatq::{FlatIndex, Metric, Tier};
use neural_mesh::memory::{HnswIndex, HnswParams, MemoryIndex};
use neural_mesh::project::Projector;
use std::time::Instant;
fn rd(p: &str) -> Vec<f32> { std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
fn pct(mut t: Vec<f64>) -> (f64, f64) { t.sort_by(|a, b| a.partial_cmp(b).unwrap()); (t[t.len() / 2], t[(t.len() as f64 * 0.95) as usize]) }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (c, q) = (rd(&a[1]), rd(&a[2]));
    let d = 384; let n = c.len() / d; let nq = q.len() / d;
    let mut out = serde_json::Map::new();
    let mut emit = |name: &str, build: f64, bytes: usize, res: Vec<Vec<u64>>, lat: Vec<f64>| {
        let (p50, p95) = pct(lat);
        out.insert(name.into(), serde_json::json!({"build_s": build, "bytes": bytes, "p50_us": p50, "p95_us": p95, "top": res}));
    };
    // flat int8 + rerank
    {
        let t0 = Instant::now();
        let mut ix = FlatIndex::new(d, Tier::Int8, Metric::Cosine).with_fast_kernels(true).with_rerank(4);
        for i in 0..n { ix.add(i as u64, &c[i * d..(i + 1) * d]).unwrap(); }
        let b = t0.elapsed().as_secs_f64();
        let (mut res, mut lat) = (vec![], vec![]);
        for i in 0..nq { let t = Instant::now(); let r = ix.search(&q[i * d..(i + 1) * d], 10).unwrap(); lat.push(t.elapsed().as_secs_f64() * 1e6); res.push(r.iter().map(|x| x.0).collect()); }
        emit("nm_flat_int8_rr4", b, ix.bytes(), res, lat);
    }
    // PCA192 + int8 (rerank factor 4 within the projected space)
    {
        let t0 = Instant::now();
        let pj = Projector::fit(&c, d, 192);
        let pc = pj.apply_all(&c);
        let mut ix = FlatIndex::new(192, Tier::Int8, Metric::Cosine).with_fast_kernels(true).with_rerank(4);
        for i in 0..n { ix.add(i as u64, &pc[i * 192..(i + 1) * 192]).unwrap(); }
        let b = t0.elapsed().as_secs_f64();
        let (mut res, mut lat) = (vec![], vec![]);
        for i in 0..nq { let t = Instant::now(); let qq = pj.apply(&q[i * d..(i + 1) * d]); let r = ix.search(&qq, 10).unwrap(); lat.push(t.elapsed().as_secs_f64() * 1e6); res.push(r.iter().map(|x| x.0).collect()); }
        emit("nm_pca192_int8_rr4", b, ix.bytes() + 192 * d * 4, res, lat);
    }
    // HNSW
    {
        let t0 = Instant::now();
        let mut ix = HnswIndex::new(d, HnswParams { m: 16, ef_construction: 200, ef_search: 64, max_elements: n + 10 });
        let ids: Vec<u64> = (0..n as u64).collect();
        ix.add_batch(&ids, &c);
        let b = t0.elapsed().as_secs_f64();
        let (mut res, mut lat) = (vec![], vec![]);
        for i in 0..nq { let t = Instant::now(); let r = ix.search(&q[i * d..(i + 1) * d], 10); lat.push(t.elapsed().as_secs_f64() * 1e6); res.push(r.iter().map(|x| x.0).collect()); }
        emit("nm_hnsw_m16_ef64", b, ix.bytes(), res, lat);
    }
    std::fs::write(&a[3], serde_json::Value::Object(out).to_string()).unwrap();
}
