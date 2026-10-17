//! km_bench: verify-before-keep benchmark for the "solved-problems" memory upgrades.
//! #1 exact-repeat cache (CachedIndex); #3 HNSW ef_search tuning (recall vs latency).
//! CPU only, ~0 VRAM. Additive — no shipped code touched, can't regress the runtime.
//! run: cargo run --release --bin km_bench
use std::time::{Duration, Instant};
use neural_mesh::memory::{CachedIndex, FlatIndex, HnswIndex, HnswParams, MemoryIndex, recall_at_k, synth::Clustered};

fn avg_us(t: Duration, n: usize) -> f64 { t.as_secs_f64() * 1e6 / n as f64 }

fn main() {
    let (dim, n, k, q, n_repeat) = (384usize, 5000usize, 5usize, 2000usize, 50usize);
    let corpus = Clustered::new(dim, 40, 1.0, 7).sample(n);
    let ids: Vec<u64> = (0..n as u64).collect();
    let fresh = Clustered::new(dim, 40, 1.0, 99).sample(q);
    let mut stream: Vec<Vec<f32>> = Vec::with_capacity(q);
    for i in 0..q {
        if (i as f64 / q as f64) < 0.5 { let r = i % n_repeat; stream.push(corpus[r*dim..(r+1)*dim].to_vec()); }
        else { stream.push(fresh[i*dim..(i+1)*dim].to_vec()); }
    }
    let mut flat = FlatIndex::new(dim); flat.add_batch(&ids, &corpus);
    println!("# solved-problems bench n={n} dim={dim} k={k} queries={q} (CPU, ~0 VRAM)");

    println!("\n#3 HNSW ef_search sweep (recall@{k} vs exact flat truth):");
    let mut hnsw = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
    hnsw.add_batch(&ids, &corpus);
    for &ef in &[16usize, 32, 64, 128, 256] {
        hnsw.set_ef_search(ef);
        let mut rec = 0.0; let t = Instant::now();
        for v in &stream { rec += recall_at_k(&flat.search(v, k), &hnsw.search(v, k), k); }
        println!("  ef={ef:<4} recall={:.3}  {:6.1} us/query", rec / q as f64, avg_us(t.elapsed(), q));
    }
    { let t = Instant::now(); for v in &stream { let _ = flat.search(v, k); }
      println!("  flat (exact)          {:6.1} us/query", avg_us(t.elapsed(), q)); }

    println!("\n#1 solved-problems cache (CachedIndex over HNSW ef=64):");
    let mut base = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
    base.set_ef_search(64); base.add_batch(&ids, &corpus);
    let cached = CachedIndex::new(base, 4096).selective(true);
    for v in &stream { let _ = cached.search(v, k); }               // warm
    let (h0, m0) = (cached.hits(), cached.misses());
    let t = Instant::now(); for v in &stream { let _ = cached.search(v, k); } let cu = avg_us(t.elapsed(), q);
    let (hits, misses) = (cached.hits()-h0, cached.misses()-m0);
    let mut plain = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
    plain.set_ef_search(64); plain.add_batch(&ids, &corpus);
    let t = Instant::now(); for v in &stream { let _ = plain.search(v, k); } let pu = avg_us(t.elapsed(), q);
    println!("  timed pass: {hits} hits / {misses} misses ({:.0}% hit-rate)", 100.0*hits as f64/q as f64);
    println!("  uncached HNSW         {pu:6.1} us/query");
    println!("  with cache            {cu:6.1} us/query  ({:.1}x on this stream)", pu/cu.max(1e-9));
}