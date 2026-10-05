//! R58: flat (FAST kernels) vs HNSW crossover. Real corpus N=6,831 (d=384) and synthetic clustered N=50k / 200k (d=384).
//! Reports build time, p50/p95 query latency (single thread) and recall@10 vs the exact flat scan.
use hnsw_rs::prelude::*;
use neural_mesh::flatq::{FlatIndex, Metric, Tier};
use serde_json::Value;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 }
}
fn unit(mut v: Vec<f32>) -> Vec<f32> { let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter_mut().for_each(|x| *x /= n); v }
fn pct(v: &mut Vec<f64>, p: f64) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[((v.len() - 1) as f64 * p) as usize] }

fn bench(name: &str, data: &[Vec<f32>], queries: &[Vec<f32>]) {
    let (n, dim) = (data.len(), data[0].len());
    let mut flat = FlatIndex::new(dim, Tier::F32, Metric::Cosine).with_fast_kernels(true);
    for (i, v) in data.iter().enumerate() { flat.add(i as u64, v).unwrap(); }
    let truth: Vec<Vec<u64>> = queries.iter().map(|q| flat.search(q, 10).unwrap().into_iter().map(|x| x.0).collect()).collect();
    let mut t = Vec::new();
    for q in queries { let t0 = Instant::now(); std::hint::black_box(flat.search(q, 10).unwrap()); t.push(t0.elapsed().as_secs_f64() * 1e6); }
    println!("{name} N={n}: flat f32 FAST p50 {:.0} us p95 {:.0} us", pct(&mut t.clone(), 0.5), pct(&mut t, 0.95));
    for (m, efc, efs) in [(16usize, 64usize, 64usize), (16, 64, 128), (24, 100, 100)] {
        let t0 = Instant::now();
        let h: Hnsw<f32, DistCosine> = Hnsw::new(m, n, 16, efc, DistCosine {});
        for (i, v) in data.iter().enumerate() { h.insert_slice((v.as_slice(), i)); }
        let build = t0.elapsed().as_secs_f64();
        let mut lat = Vec::new();
        let mut rec = 0.0;
        for (qi, q) in queries.iter().enumerate() {
            let t1 = Instant::now();
            let r = h.search(q, 10, efs);
            lat.push(t1.elapsed().as_secs_f64() * 1e6);
            let ids: std::collections::HashSet<u64> = r.iter().map(|x| x.d_id as u64).collect();
            rec += truth[qi].iter().filter(|i| ids.contains(i)).count() as f64 / 10.0;
        }
        println!("   hnsw M={m} efc={efc} efs={efs}: build {build:.1} s, p50 {:.0} us p95 {:.0} us, recall@10 {:.3}", pct(&mut lat.clone(), 0.5), pct(&mut lat, 0.95), rec / queries.len() as f64);
    }
}
fn main() {
    let dim = 384;
    // real corpus
    let raw = std::fs::read("results/h10_emb_A.f32").unwrap();
    let emb: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let inst: Value = serde_json::from_slice(&std::fs::read("results/h10_inst_A.json").unwrap()).unwrap();
    let ndoc = inst["ndoc"].as_u64().unwrap() as usize;
    let rows = inst["inst"]["31"]["rows"].as_array().unwrap();
    let data: Vec<Vec<f32>> = (0..ndoc).map(|i| emb[i * dim..(i + 1) * dim].to_vec()).collect();
    let queries: Vec<Vec<f32>> = rows.iter().take(200).map(|r| { let t = r["t"][0][0].as_u64().unwrap() as usize; emb[t * dim..(t + 1) * dim].to_vec() }).collect();
    bench("real", &data, &queries);
    for n in [50_000usize, 200_000] {
        let mut r = Rng(0xDEADBEEF12345);
        let centers: Vec<Vec<f32>> = (0..500).map(|_| (0..dim).map(|_| r.next()).collect()).collect();
        let mk = |r: &mut Rng, c: &Vec<f32>| unit((0..dim).map(|j| c[j] + 0.35 * r.next()).collect());
        let data: Vec<Vec<f32>> = (0..n).map(|i| { let c = centers[i % 500].clone(); mk(&mut r, &c) }).collect();
        let queries: Vec<Vec<f32>> = (0..100).map(|i| { let c = centers[(i * 7) % 500].clone(); mk(&mut r, &c) }).collect();
        bench("synthetic-clustered", &data, &queries);
    }
}
