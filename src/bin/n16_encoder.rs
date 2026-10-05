//! R101: pure-Rust MiniLM encoder vs the python ORT embeddings (E_fp32.npy / E_q8.npy of R85) on the 2,968 real queries; prints cosine stats and latency p50.
//! Usage: n16_encoder <weights.f32> <tokenizer.json> <qtexts.json> <E_fp32.npy> <E_q8.npy>
use neural_mesh::encoder::Encoder;
use std::time::Instant;
fn npy(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    b[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn stats(name: &str, a: &[Vec<f32>], r: &[f32]) {
    let mut cs: Vec<f32> = a.iter().enumerate().map(|(i, v)| v.iter().zip(&r[i * 384..(i + 1) * 384]).map(|(x, y)| x * y).sum::<f32>()).collect();
    cs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("{name}: cosine: min {:.6} p1 {:.6} p50 {:.6} mean {:.6}", cs[0], cs[cs.len() / 100], cs[cs.len() / 2], cs.iter().sum::<f32>() / cs.len() as f32);
}
fn run(enc: &Encoder, texts: &[String]) -> (Vec<Vec<f32>>, Vec<f64>) {
    for t in texts.iter().take(30) {
        enc.encode(t);
    }
    let (mut v, mut lat) = (vec![], vec![]);
    for t in texts {
        let t0 = Instant::now();
        v.push(enc.encode(t));
        lat.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v, lat)
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let q: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[3]).unwrap()).unwrap();
    let mut texts: Vec<String> = Vec::new();
    for t in ["A", "B"] {
        for s in q[t]["text"].as_array().unwrap() {
            texts.push(s.as_str().unwrap().to_string());
        }
    }
    let e32 = npy(&a[4]);
    let e8 = npy(&a[5]);
    let enc = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap();
    let (v, lat) = run(&enc, &texts);
    stats("rust f32 vs python fp32 ORT", &v, &e32);
    println!("  rust f32 latency p50 {:.3} ms p90 {:.3} ms (1 thread, {} queries)", lat[lat.len() / 2], lat[lat.len() * 9 / 10], texts.len());
    let enc8 = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap().with_int8(true);
    let (v8, lat8) = run(&enc8, &texts);
    stats("rust int8 vs python fp32 ORT", &v8, &e32);
    stats("rust int8 vs python int8 ORT", &v8, &e8);
    println!("  rust int8 latency p50 {:.3} ms p90 {:.3} ms", lat8[lat8.len() / 2], lat8[lat8.len() * 9 / 10]);
}
