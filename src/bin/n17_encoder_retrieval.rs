//! R101b: retrieval with the pure-Rust encoder: hybrid .7 (int8 dense shadow) over the real corpora A / B; query vector = stored ORT fp32 embedding (base) vs Rust f32 encoder vs Rust int8 encoder
//! (documents keep their ORT embeddings). Tie-pessimistic rank of the first relevant doc (full fused-score vector). Writes results/r101_ranks.json.
//! Usage: n17_encoder_retrieval <weights.f32> <tokenizer.json>
use neural_mesh::encoder::Encoder;
use neural_mesh::hybrid::{HybridConfig, HybridIndex};
use serde_json::{json, Value};
use std::collections::BTreeMap;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let e32 = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap();
    let e8 = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap().with_int8(true);
    let mut out = serde_json::Map::new();
    for tag in ["A", "B"] {
        let texts: Vec<String> = serde_json::from_slice(&std::fs::read(format!("results/h10_texts_{tag}.json")).unwrap()).unwrap();
        let raw = std::fs::read(format!("results/h10_emb_{tag}.f32")).unwrap();
        let emb: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let dim = 384;
        let inst: Value = serde_json::from_slice(&std::fs::read(format!("results/h10_inst_{tag}.json")).unwrap()).unwrap();
        let ndoc = inst["ndoc"].as_u64().unwrap() as usize;
        let mut qmap: BTreeMap<u64, usize> = BTreeMap::new();
        for (s, v) in inst["inst"].as_object().unwrap() { if s == "R" { continue; } for row in v["rows"].as_array().unwrap() { qmap.insert(row["q"].as_u64().unwrap(), row["t"][0][0].as_u64().unwrap() as usize); } }
        let rels: Vec<Vec<usize>> = inst["queries"].as_array().unwrap().iter().map(|q| q["rel"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()).collect();
        let docs: Vec<String> = texts[..ndoc].to_vec();
        let ix = HybridIndex::new(&docs, emb[..ndoc * dim].to_vec(), dim, HybridConfig::default()).unwrap().with_int8_dense(true);
        let (mut rb, mut r32, mut r8, mut cs32, mut cs8) = (vec![], vec![], vec![], 0f64, 0f64);
        for (&q, &tid) in &qmap {
            let qb = emb[tid * dim..(tid + 1) * dim].to_vec();
            let (q32, q8) = (e32.encode(&texts[tid]), e8.encode(&texts[tid]));
            cs32 += qb.iter().zip(&q32).map(|(x, y)| (x * y) as f64).sum::<f64>();
            cs8 += qb.iter().zip(&q8).map(|(x, y)| (x * y) as f64).sum::<f64>();
            let tp = |qv: &[f32]| -> i64 { let mut sc = Vec::new(); ix.fused_scores(&texts[tid], qv, &mut sc).unwrap(); rels[q as usize].iter().map(|&d| sc.iter().filter(|&&x| x >= sc[d]).count() as i64 - 1).min().unwrap_or(-1) };
            rb.push(tp(&qb)); r32.push(tp(&q32)); r8.push(tp(&q8));
        }
        let n = qmap.len() as f64;
        let mrr = |v: &Vec<i64>| v.iter().map(|&r| if r < 0 { 0.0 } else { 1.0 / (r as f64 + 1.0) }).sum::<f64>() / n;
        println!("corpus {tag}: {} queries; mean cosine to stored ORT query vec: rust f32 {:.5}, rust int8 {:.5}; MRR (tie-pessimistic) stored {:.4} | rust f32 {:.4} | rust int8 {:.4}", qmap.len(), cs32 / n, cs8 / n, mrr(&rb), mrr(&r32), mrr(&r8));
        out.insert(tag.to_string(), json!({"base": rb, "rust_f32": r32, "rust_i8": r8}));
    }
    std::fs::write("results/r101_ranks.json", serde_json::to_string(&Value::Object(out)).unwrap()).unwrap();
}
