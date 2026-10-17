//! v1 (vector-lab): encode sentences (JSON array of strings) with the pure-Rust MiniLM encoder; write n x 384 f32 (LE).
//! Usage: v1_encode <weights.f32> <tokenizer.json> <sentences.json> <out.f32>
use neural_mesh::encoder::Encoder;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let s: Vec<String> = serde_json::from_slice(&std::fs::read(&a[3]).unwrap()).unwrap();
    let enc = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap();
    let mut out = Vec::new();
    for t in &s { for x in enc.encode(t) { out.extend_from_slice(&x.to_le_bytes()); } }
    std::fs::write(&a[4], out).unwrap();
}
