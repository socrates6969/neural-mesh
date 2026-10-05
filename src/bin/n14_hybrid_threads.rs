//! R99: threaded int8 dense scan inside HybridIndex (`with_threads`). Real corpus A (6,831 paragraphs, d=384) tiled to N = 20k..300k with noise-perturbed
//! vectors (texts repeated), 100 real queries. Prints p50 us of dense_scores and of the full hybrid search(k=10) for threads = serial / 2 / 4 / 8.
use neural_mesh::hybrid::{HybridConfig, HybridIndex};
use std::time::Instant;
fn p50(v: &mut Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn main() {
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read("results/h10_texts_A.json").unwrap()).unwrap();
    let raw = std::fs::read("results/h10_emb_A.f32").unwrap();
    let emb: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let dim = 384;
    let base_n = emb.len() / dim;
    let mut st = 99u64;
    let mut rnd = move || { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((st >> 33) as f32 / 2147483648.0) - 0.5 };
    let qids: Vec<usize> = (0..100).map(|i| (i * 67) % base_n).collect();
    for &n in &[20_000usize, 100_000, 300_000] {
        let mut docs = Vec::with_capacity(n);
        let mut dense = Vec::with_capacity(n * dim);
        for i in 0..n {
            let b = i % base_n;
            docs.push(texts[b].clone());
            let mut v: Vec<f32> = emb[b * dim..(b + 1) * dim].iter().map(|x| x + if i >= base_n { 0.05 * rnd() } else { 0.0 }).collect();
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= nrm);
            dense.extend(v);
        }
        let ix = HybridIndex::new(&docs, dense, dim, HybridConfig::default()).unwrap().with_int8_dense(true);
        let mut line = format!("N={n}:");
        let mut sc = Vec::new();
        let mut serial_res = Vec::new();
        let mut diff = 0usize;
        for &t in &[0usize, 2, 4, 8] {
            let ixt = ix.clone().with_threads(t);
            let (mut dd, mut ss) = (vec![], vec![]);
            for rep in 0..3 {
                for &q in &qids {
                    let qv = &emb[q * dim..(q + 1) * dim];
                    let t0 = Instant::now(); ixt.dense_scores(qv, &mut sc).unwrap(); let d = t0.elapsed().as_secs_f64() * 1e6;
                    let t0 = Instant::now(); let r = ixt.search(&texts[q], qv, 10).unwrap(); let s = t0.elapsed().as_secs_f64() * 1e6;
                    if rep > 0 { dd.push(d); ss.push(s); }
                    if rep == 0 && t == 0 { serial_res.push(r.iter().map(|x| x.0).collect::<Vec<_>>()); } else if rep == 0 { assert_eq!(serial_res[qids.iter().position(|&x| x == q).unwrap()], r.iter().map(|x| x.0).collect::<Vec<_>>()); }
                }
            }
            let mut sf = vec![];
            for rep in 0..3 { for &q in &qids { let qv = &emb[q * dim..(q + 1) * dim]; let t0 = Instant::now(); let r = ixt.search_fast(&texts[q], qv, 10).unwrap(); let s = t0.elapsed().as_secs_f64() * 1e6; if rep > 0 { sf.push(s); } if rep == 0 { let want = &serial_res[qids.iter().position(|&x| x == q).unwrap()]; let got: Vec<usize> = r.iter().map(|x| x.0).collect(); if &got != want { diff += 1; } } } }
            line += &format!("  t={t}: dense {:.0}, search {:.0}, search_fast {:.0} us |", p50(&mut dd), p50(&mut ss), p50(&mut sf));
        }
        println!("{line} search_fast top-10 != search in {diff} of {} query-runs", 4 * qids.len());
    }
}

