//! R46: Rust PCA192 projector + int8 FAST flat scan on the real corpus vs (a) exact f32 top-10 and (b) the python SVD PCA192 top-10 (results/n3_py_pca_top10.json).
use neural_mesh::flatq::{FlatIndex, Metric, Tier};
use neural_mesh::project::Projector;
use serde_json::Value;
use std::time::Instant;
fn main() {
    let raw = std::fs::read("results/h10_emb_A.f32").unwrap();
    let emb: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let dim = 384;
    let inst: Value = serde_json::from_slice(&std::fs::read("results/h10_inst_A.json").unwrap()).unwrap();
    let ndoc = inst["ndoc"].as_u64().unwrap() as usize;
    let py: Value = serde_json::from_slice(&std::fs::read("results/n3_py_pca_top10.json").unwrap()).unwrap();
    let t0 = Instant::now();
    let pr = Projector::fit(&emb[..ndoc * dim], dim, 192);
    println!("fit PCA192 on {ndoc}x{dim}: {:.2} s", t0.elapsed().as_secs_f64());
    let dp = pr.apply_all(&emb[..ndoc * dim]);
    let mut exact = FlatIndex::new(dim, Tier::F32, Metric::Cosine);
    let mut pca_f = FlatIndex::new(192, Tier::F32, Metric::Cosine);
    let mut pca_i8 = FlatIndex::new(192, Tier::Int8, Metric::Cosine).with_fast_kernels(true);
    for i in 0..ndoc {
        exact.add(i as u64, &emb[i * dim..(i + 1) * dim]).unwrap();
        pca_f.add(i as u64, &dp[i * 192..(i + 1) * 192]).unwrap();
        pca_i8.add(i as u64, &dp[i * 192..(i + 1) * 192]).unwrap();
    }
    let (mut r_ex, mut r_py_f, mut r_py_i8, mut n) = (0.0, 0.0, 0.0, 0.0);
    let mut lat = Vec::new();
    for (k, v) in py.as_object().unwrap() {
        let t: usize = k.parse().unwrap();
        let q = &emb[t * dim..(t + 1) * dim];
        let qp = pr.apply(q);
        let ex: std::collections::HashSet<u64> = exact.search(q, 10).unwrap().into_iter().map(|x| x.0).collect();
        let t1 = Instant::now();
        let ri8: Vec<u64> = pca_i8.search(&qp, 10).unwrap().into_iter().map(|x| x.0).collect();
        lat.push(t1.elapsed().as_secs_f64() * 1e6);
        let rf: Vec<u64> = pca_f.search(&qp, 10).unwrap().into_iter().map(|x| x.0).collect();
        let pyset: std::collections::HashSet<u64> = v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
        r_ex += ri8.iter().filter(|i| ex.contains(i)).count() as f64 / 10.0;
        r_py_f += rf.iter().filter(|i| pyset.contains(i)).count() as f64 / 10.0;
        r_py_i8 += ri8.iter().filter(|i| pyset.contains(i)).count() as f64 / 10.0;
        n += 1.0;
    }
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("queries {n}: Rust PCA192 f32 vs python SVD PCA192 f32 top-10 overlap {:.4}", r_py_f / n);
    println!("Rust PCA192+int8(FAST) vs python PCA192 f32 overlap {:.4}; vs EXACT 384-d f32 top-10 overlap {:.4}", r_py_i8 / n, r_ex / n);
    println!("PCA192+int8 FAST scan p50 {:.0} us, bytes/vec {}", lat[lat.len() / 2], pca_i8.bytes_per_vec());
}
