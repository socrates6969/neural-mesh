//! h17: per-text cost of the hash encoders (repo HashEmbedding dim 384 / lowercase-token variant) on REAL h10 texts, single thread, in-process (no tokenizer model, no file).
//! Usage: h17_hashbench  -> results/h17_hashbench.json
use std::time::Instant;

use neural_mesh::embedding::{Embedding, HashEmbedding};
use serde_json::{Value, json};

fn main() {
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read("results/h10_texts_A.json").unwrap()).unwrap();
    let docs: Vec<&String> = texts.iter().take(6831).collect();
    let qs: Vec<&String> = texts.iter().skip(6831).step_by(7).take(3000).collect();
    let h = HashEmbedding::new(384);
    let mut res = json!({});
    for (name, set) in [("docs", &docs), ("queries", &qs)] {
        let mut sink = 0.0f32;
        let t = Instant::now();
        for _ in 0..5 {
            for s in set.iter() {
                sink += h.embed(s)[0];
            }
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / (5 * set.len()) as f64;
        res[name] = json!({"n": set.len(), "us_per_text": us, "mean_chars": set.iter().map(|s| s.len()).sum::<usize>() as f64 / set.len() as f64, "sink": sink});
    }
    res["vector_bytes_f32_dim384"] = json!(384 * 4);
    res["vector_bytes_f32_dim64_repo_default"] = json!(64 * 4);
    res["model_bytes"] = Value::from(0);
    std::fs::write("results/h17_hashbench.json", serde_json::to_string_pretty(&res).unwrap()).unwrap();
    println!("{res}");
}
