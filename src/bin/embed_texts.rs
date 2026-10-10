//! embed_texts: batch-encode a JSON array of strings with the pure-Rust MiniLM encoder (src/encoder.rs, all-MiniLM-L6-v2, 384-d, L2-normalised)
//! into a row-major little-endian f32 file. Used by the lifelong ingestor (ingest/vectors.py). Opt-in tooling; uses only the public encoder API.
//! Usage: embed_texts <weights.f32> <tokenizer.json> <texts.json> <out.f32> [--threads N] [--batch B]
use neural_mesh::encoder::Encoder;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 4 {
        eprintln!("usage: embed_texts <weights.f32> <tokenizer.json> <texts.json> <out.f32> [--threads N] [--batch B]");
        std::process::exit(2);
    }
    let (mut threads, mut batch) = (4usize, 16usize);
    let mut i = 4;
    while i < a.len() {
        match a[i].as_str() {
            "--threads" => {
                i += 1;
                threads = a.get(i).and_then(|x| x.parse().ok()).unwrap_or(threads);
            }
            "--batch" => {
                i += 1;
                batch = a.get(i).and_then(|x| x.parse().ok()).unwrap_or(batch).max(1);
            }
            other => {
                eprintln!("unknown flag {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let enc = match Encoder::load(std::path::Path::new(&a[0]), std::path::Path::new(&a[1])) {
        Ok(e) => e.with_threads(threads),
        Err(e) => {
            eprintln!("encoder load failed: {}", e.0);
            std::process::exit(1);
        }
    };
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read(&a[2]).expect("read texts")).expect("texts.json must be a JSON array of strings");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&a[3]).expect("create out"));
    let t0 = std::time::Instant::now();
    for chunk in texts.chunks(batch) {
        let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        for v in enc.encode_batch(&refs) {
            for x in v {
                out.write_all(&x.to_le_bytes()).unwrap();
            }
        }
    }
    out.flush().unwrap();
    println!("{{\"ok\":true,\"n\":{},\"dim\":384,\"ms\":{:.0}}}", texts.len(), t0.elapsed().as_secs_f64() * 1e3);
}
