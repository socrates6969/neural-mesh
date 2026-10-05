//! R99b: HybridIndex::search_fast vs search on the two REAL corpora (A 6,831 / B 4,181 paragraphs), every unique query: top-10 identity, rank-of-first-relevant equality, MRR, latency p50.
use neural_mesh::hybrid::{HybridConfig, HybridIndex};
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Instant;
fn main() {
    let kk: usize = std::env::var("K").ok().and_then(|s| s.parse().ok()).unwrap_or(1000);
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
        for (name, ix) in [("f32", HybridIndex::new(&docs, emb[..ndoc * dim].to_vec(), dim, HybridConfig::default()).unwrap()), ("int8", HybridIndex::new(&docs, emb[..ndoc * dim].to_vec(), dim, HybridConfig::default()).unwrap().with_int8_dense(true))] {
            let (mut same10, mut same_rank, mut n, mut m0, mut m1) = (0, 0, 0, 0.0f64, 0.0f64);
            let (mut ta, mut tb) = (vec![], vec![]);
            for (&q, &tid) in &qmap {
                let qv = &emb[tid * dim..(tid + 1) * dim];
                let t0 = Instant::now(); let a = ix.search(&texts[tid], qv, kk).unwrap(); ta.push(t0.elapsed().as_secs_f64() * 1e6);
                let t0 = Instant::now(); let b = ix.search_fast(&texts[tid], qv, kk).unwrap(); tb.push(t0.elapsed().as_secs_f64() * 1e6);
                let rank = |l: &Vec<(usize, f32)>| l.iter().position(|(d, _)| rels[q as usize].contains(d));
                let (ra, rb) = (rank(&a), rank(&b));
                if a.iter().take(10).map(|x| x.0).collect::<Vec<_>>() == b.iter().take(10).map(|x| x.0).collect::<Vec<_>>() { same10 += 1; }
                if ra == rb { same_rank += 1; }
                m0 += ra.map(|r| 1.0 / (r as f64 + 1.0)).unwrap_or(0.0); m1 += rb.map(|r| 1.0 / (r as f64 + 1.0)).unwrap_or(0.0);
                n += 1;
            }
            ta.sort_by(|a, b| a.partial_cmp(b).unwrap()); tb.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("corpus {tag} {name}: {n} queries; top-10 identical {:.4}; first-relevant rank identical {:.4}; MRR@1000 search {:.4} vs fast {:.4}; p50 (k={kk}) {:.0} vs {:.0} us (x{:.2})", same10 as f64 / n as f64, same_rank as f64 / n as f64, m0 / n as f64, m1 / n as f64, ta[ta.len() / 2], tb[tb.len() / 2], ta[ta.len() / 2] / tb[tb.len() / 2], kk = kk);
        }
    }
}
