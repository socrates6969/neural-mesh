//! R57: reachability of HNSW items vs ef_construction (raw hnsw_rs, vendored fix). Counts items that are NOT their own top-1 (clustered (60 clusters, noise .15), n=3000, d=32, M=16).
use hnsw_rs::prelude::*;
fn main() {
    let (n, dim) = (3000usize, 32usize);
    for efc in [8usize, 16, 24, 32, 40, 48, 64, 200] {
        let mut tot = 0usize;
        let mut worst = 0usize;
        let seeds = 6u64;
        for seed in 1..=seeds {
            let mut s = seed * 0x9E3779B97F4A7C15;
            let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 };
            let centers: Vec<Vec<f32>> = (0..60).map(|_| (0..dim).map(|_| rnd()).collect()).collect();
            let data: Vec<Vec<f32>> = (0..n).map(|i| { let c = &centers[i % 60]; let mut v: Vec<f32> = (0..dim).map(|j| c[j] + 0.15 * rnd()).collect(); let nn = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter_mut().for_each(|x| *x /= nn); v }).collect();
            let h: Hnsw<f32, DistCosine> = Hnsw::new(16, n, 16, efc, DistCosine {});
            for (i, v) in data.iter().enumerate() { h.insert_slice((v.as_slice(), i)); }
            let miss = (0..n).filter(|&i| h.search(&data[i], 1, 64).first().map(|x| x.d_id) != Some(i)).count();
            tot += miss;
            worst = worst.max(miss);
        }
        println!("ef_construction {efc:3}: mean misses {:7.1} / {n}  worst {worst}", tot as f64 / seeds as f64);
    }
}
