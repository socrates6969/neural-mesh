//! R91: scan kernels (src/scan.rs) vs the shipped FlatIndex FAST kernel. Real corpus A (6,831 x 384, 200 real queries) for recall; synthetic random unit vectors for N = 1e5 / 1e6 speed.
//! usage: n10_kernels real | n10_kernels syn <N> [dim] [threads...]
use neural_mesh::flatq::{FlatIndex, Metric, Tier};
use neural_mesh::scan::*;
use std::time::Instant;
struct Rng(u64);
impl Rng { fn next(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 } }
fn unit(v: &mut [f32]) { let n = v.iter().map(|x| x * x).sum::<f32>().sqrt(); v.iter_mut().for_each(|x| *x /= n); }
fn p50<F: FnMut()>(mut f: F, reps: usize) -> (f64, f64) { let mut t = Vec::new(); for _ in 0..reps { let t0 = Instant::now(); f(); t.push(t0.elapsed().as_secs_f64() * 1e6); } t.sort_by(|a, b| a.partial_cmp(b).unwrap()); (t[t.len() / 2], t[(t.len() as f64 * 0.9) as usize]) }
fn read_f32(p: &str) -> Vec<f32> { let b = std::fs::read(p).unwrap(); b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (data, qs, dim): (Vec<f32>, Vec<Vec<f32>>, usize) = if a[1] == "real" {
        let d = read_f32("results/n10_docs_A.f32"); let q = read_f32("results/n10_queries_A.f32");
        (d, q.chunks(384).take(200).map(|c| c.to_vec()).collect(), 384)
    } else {
        let n: usize = a[2].parse().unwrap(); let dim: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(384); let mut r = Rng(0x9E3779B97F4A7C15);
        let mut d = vec![0f32; n * dim]; for row in d.chunks_mut(dim) { row.iter_mut().for_each(|x| *x = r.next()); unit(row); }
        (d, (0..30).map(|_| { let mut v: Vec<f32> = (0..dim).map(|_| r.next()).collect(); unit(&mut v); v }).collect(), dim)
    };
    let n = data.len() / dim; let k = 10; println!("N={n} dim={dim} queries={}", qs.len());
    let reps = qs.len().min(30);
    // exact reference ids
    let exact: Vec<Vec<usize>> = qs.iter().map(|q| scan_f32_baseline(&data, dim, q, k).into_iter().map(|x| x.1).collect()).collect();
    let mut i = 0usize; let (m, p) = p50(|| { std::hint::black_box(scan_f32_baseline(&data, dim, &qs[i % reps], k)); i += 1; }, reps * 2); println!("baseline (default-ISA dot8, bounded top-k)  p50 {m:10.0} us p90 {p:10.0}");
    let base = m;
    if n <= 200_000 {
        let mut ix = FlatIndex::new(dim, Tier::F32, Metric::Dot).with_fast_kernels(true); for r in 0..n { ix.add(r as u64, &data[r * dim..(r + 1) * dim]).unwrap(); }
        let mut i = 0usize; let (m, p) = p50(|| { std::hint::black_box(ix.search(&qs[i % reps], k).unwrap()); i += 1; }, reps * 2); println!("FlatIndex F32 FAST (shipped R45)            p50 {m:10.0} us p90 {p:10.0}");
    }
    let mut i = 0usize; let (m, p) = p50(|| { std::hint::black_box(scan_f32(&data, dim, &qs[i % reps], k)); i += 1; }, reps * 2); println!("scan_f32 (AVX2 dispatch)                    p50 {m:10.0} us p90 {p:10.0}  x{:.2} vs baseline", base / m);
    let mut ok = true; for (j, q) in qs.iter().enumerate() { let g: Vec<usize> = scan_f32(&data, dim, q, k).into_iter().map(|x| x.1).collect(); if g != exact[j] { ok = false; } } println!("   top-{k} id lists identical to baseline on all {} queries: {ok}", qs.len());
    let ths: Vec<usize> = if a.len() > 4 { a[4..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![2, 4, 8, 16] };
    for t in ths { let mut i = 0usize; let (m, p) = p50(|| { std::hint::black_box(scan_f32_par(&data, dim, &qs[i % reps], k, t)); i += 1; }, reps * 2); println!("scan_f32_par threads={t:<2}                      p50 {m:10.0} us p90 {p:10.0}  x{:.2} vs baseline", base / m); }
    // int8 rows
    let mut rows = vec![0i8; n * dim]; let mut sc = vec![0f32; n]; for r in 0..n { let (c, s) = quantize_i8(&data[r * dim..(r + 1) * dim]); rows[r * dim..(r + 1) * dim].copy_from_slice(&c); sc[r] = s; }
    let qq: Vec<(Vec<i8>, f32)> = qs.iter().map(|q| quantize_i8(q)).collect();
    let mut i = 0usize; let (m, p) = p50(|| { let (c, s) = &qq[i % reps]; std::hint::black_box(scan_i8(&rows, &sc, dim, c, *s, k)); i += 1; }, reps * 2);
    let mut ov = 0.0; let mut t1 = 0.0; for (j, (c, s)) in qq.iter().enumerate() { let g: Vec<usize> = scan_i8(&rows, &sc, dim, c, *s, k).into_iter().map(|x| x.1).collect(); ov += g.iter().filter(|x| exact[j].contains(x)).count() as f64 / k as f64; if g[0] == exact[j][0] { t1 += 1.0; } }
    println!("scan_i8 (int8 rows x int8 query)            p50 {m:10.0} us p90 {p:10.0}  x{:.2} vs baseline; recall@{k} vs exact {:.4}, top-1 agree {:.4}", base / m, ov / qs.len() as f64, t1 / qs.len() as f64);
    let ths2: Vec<usize> = if a.len() > 4 { a[4..].iter().map(|s| s.parse().unwrap()).collect() } else { vec![2, 4, 8, 16] };
    for t in ths2 { let mut i = 0usize; let (m, p) = p50(|| { let (c, s) = &qq[i % reps]; std::hint::black_box(scan_i8_par(&rows, &sc, dim, c, *s, k, t)); i += 1; }, reps * 2); println!("scan_i8_par threads={t:<2}                       p50 {m:10.0} us p90 {p:10.0}  x{:.2} vs baseline", base / m); }
    // int8 + exact f32 rerank of the top 4k candidates (what FlatIndex int8+rerank4 does), threads=1
    let mut ov2 = 0.0; for (j, (c, s)) in qq.iter().enumerate() { let cand = scan_i8(&rows, &sc, dim, c, *s, 4 * k); let mut sc2: Vec<(f32, usize)> = cand.iter().map(|&(_, r)| (neural_mesh::flatq::dot8(&data[r * dim..(r + 1) * dim], &qs[j]), r)).collect(); sc2.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1))); ov2 += sc2.iter().take(k).filter(|x| exact[j].contains(&x.1)).count() as f64 / k as f64; }
    let mut i = 0usize; let (m, _) = p50(|| { let (c, s) = &qq[i % reps]; let cand = scan_i8_par(&rows, &sc, dim, c, *s, 4 * k, 4); let mut v: Vec<(f32, usize)> = cand.iter().map(|&(_, r)| (neural_mesh::flatq::dot8(&data[r * dim..(r + 1) * dim], &qs[i % reps]), r)).collect(); v.sort_by(|x, y| y.0.total_cmp(&x.0)); std::hint::black_box(v); i += 1; }, reps * 2);
    println!("int8 scan (4 thr) + exact f32 rerank of top-{}  p50 {m:10.0} us  x{:.2} vs baseline; recall@{k} {:.4}", 4 * k, base / m, ov2 / qs.len() as f64);
}
