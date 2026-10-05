//! R70: validate `flatq::suggest_tier`'s measured p50 table (H6, d=384) against the CURRENT FlatIndex, scalar vs fast kernels, single thread.
//! Prints p50 (us) per (tier, N, kernels) and compares with the CANDS table row.
use neural_mesh::flatq::{FlatIndex, Metric, Tier};
use std::time::Instant;
struct Rng(u64);
impl Rng { fn next(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 } }
fn unit(mut v: Vec<f32>) -> Vec<f32> { let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter_mut().for_each(|x| *x /= n); v }
fn main() {
    let dim: usize = std::env::var("DIM").ok().and_then(|s| s.parse().ok()).unwrap_or(384); let ns: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().unwrap()).collect();
    let mut r = Rng(0x9E3779B97F4A7C15);
    for &n in &ns {
        let qs: Vec<Vec<f32>> = (0..30).map(|_| unit((0..dim).map(|_| r.next()).collect())).collect();
        let data: Vec<f32> = { let mut d = Vec::with_capacity(n * dim); for _ in 0..n { d.extend(unit((0..dim).map(|_| r.next()).collect())); } d };
        for (name, tier, rerank) in [("f32", Tier::F32, 0usize), ("blockfp16", Tier::BlockFp16, 0), ("int8+rerank4", Tier::Int8, 4)] {
            for fast in [false, true] {
                let mut ix = FlatIndex::new(dim, tier, Metric::Cosine).with_fast_kernels(fast).with_rerank(rerank);
                for i in 0..n { ix.add(i as u64, &data[i * dim..(i + 1) * dim]).unwrap(); }
                let mut t = Vec::new();
                for q in qs.iter().chain(qs.iter().take(10)) { let t0 = Instant::now(); std::hint::black_box(ix.search(q, 10).unwrap()); t.push(t0.elapsed().as_secs_f64() * 1e6); }
                t.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!("N={n:8} {name:13} fast={fast:5} p50 {:9.0} us  p95 {:9.0} us", t[t.len() / 2], t[(t.len() as f64 * 0.95) as usize]);
            }
        }
    }
}
