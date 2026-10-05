//! R106: single-process end-to-end memory lookup (pure Rust): text -> Rust encoder -> HybridIndex int8 + search_fast(k=10), corpus A (6,831 docs), 300 real queries.
//! Usage: n18_e2e_lookup <weights.f32> <tokenizer.json>
use neural_mesh::encoder::Encoder;
use neural_mesh::hybrid::{HybridConfig, HybridIndex};
use std::time::Instant;
fn p50(v: &mut Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read("results/h10_texts_A.json").unwrap()).unwrap();
    let raw = std::fs::read("results/h10_emb_A.f32").unwrap();
    let emb: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let dim = 384; let n = emb.len() / dim;
    let ix = HybridIndex::new(&texts[..n], emb.clone(), dim, HybridConfig::default()).unwrap().with_int8_dense(true);
    let qs: Vec<String> = (0..300).map(|i| texts[(i * 23) % n].split_whitespace().take(12).collect::<Vec<_>>().join(" ")).collect();
    for (name, enc) in [("f32 1thr", Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap()), ("f32 4thr", Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap().with_threads(4))] {
        for bs in [1usize, 16] {
            let (mut enc_ms, mut srch_ms, mut agree) = (vec![], vec![], 0usize);
            for c in qs.chunks(bs) {
                let refs: Vec<&str> = c.iter().map(|s| s.as_str()).collect();
                let t0 = Instant::now(); let v = enc.encode_batch(&refs); enc_ms.push(t0.elapsed().as_secs_f64() * 1e3 / c.len() as f64);
                for (q, qv) in c.iter().zip(&v) {
                    let t0 = Instant::now(); let r = ix.search_fast(q, qv, 10).unwrap(); srch_ms.push(t0.elapsed().as_secs_f64() * 1e3);
                    if texts[r[0].0].starts_with(q.as_str()) { agree += 1; }
                }
            }
            println!("{name} bs={bs}: encode {:.2} ms/q + search_fast {:.3} ms = {:.2} ms/query end-to-end; top-1 doc starts with the query words in {}/{}", p50(&mut enc_ms), p50(&mut srch_ms), p50(&mut enc_ms) + p50(&mut srch_ms), agree, qs.len());
        }
    }
}
