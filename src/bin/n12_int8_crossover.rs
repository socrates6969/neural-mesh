//! R92: flat int8 (R91 integer dot) vs the HNSW numbers of R58 on the same synthetic clustered data (500 clusters, d=384, 100 queries). Flat-only (no HNSW build): reports p50 and recall@10 vs exact f32.
use neural_mesh::flatq::{FlatIndex, Metric, Tier};
use std::time::Instant;
struct Rng(u64);
impl Rng { fn next(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 } }
fn unit(mut v: Vec<f32>) -> Vec<f32> { let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter_mut().for_each(|x| *x /= n); v }
fn p50(v: &mut Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn main() {
    let dim = 384;
    for n in std::env::args().skip(1).map(|a| a.parse::<usize>().unwrap()) {
        let mut r = Rng(0xDEADBEEF12345);
        let centers: Vec<Vec<f32>> = (0..500).map(|_| (0..dim).map(|_| r.next()).collect()).collect();
        let mk = |r: &mut Rng, c: &Vec<f32>| unit((0..dim).map(|j| c[j] + 0.35 * r.next()).collect());
        let data: Vec<Vec<f32>> = (0..n).map(|i| { let c = centers[i % 500].clone(); mk(&mut r, &c) }).collect();
        let queries: Vec<Vec<f32>> = (0..100).map(|i| { let c = centers[(i * 7) % 500].clone(); mk(&mut r, &c) }).collect();
        let mut exact = FlatIndex::new(dim, Tier::F32, Metric::Cosine).with_fast_kernels(true);
        for (i, v) in data.iter().enumerate() { exact.add(i as u64, v).unwrap(); }
        let truth: Vec<Vec<u64>> = queries.iter().map(|q| exact.search(q, 10).unwrap().into_iter().map(|x| x.0).collect()).collect();
        let run = |name: &str, ix: &FlatIndex| {
            let mut t = Vec::new(); let mut rec = 0.0;
            for (qi, q) in queries.iter().enumerate() { let t0 = Instant::now(); let r = ix.search(q, 10).unwrap(); t.push(t0.elapsed().as_secs_f64() * 1e6); rec += truth[qi].iter().filter(|i| r.iter().any(|x| x.0 == **i)).count() as f64 / 10.0; }
            println!("N={n:7} {name:34} p50 {:9.0} us  recall@10 {:.4}", p50(&mut t), rec / queries.len() as f64);
        };
        run("f32 FAST (shipped R45)", &exact);
        for (name, th, rerank) in [("int8 dot, 1 thread", 1usize, 0usize), ("int8 dot, 4 threads", 4, 0), ("int8 dot + rerank4, 1 thread", 1, 4), ("int8 dot + rerank4, 4 threads", 4, 4)] {
            let mut ix = FlatIndex::new(dim, Tier::Int8, Metric::Cosine).with_int8_dot(true).with_threads(th).with_rerank(rerank);
            for (i, v) in data.iter().enumerate() { ix.add(i as u64, v).unwrap(); }
            run(name, &ix);
        }
        println!();
    }
}
