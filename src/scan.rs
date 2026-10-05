//! R91 (opt-in): faster dense scan kernels for `[n x d]` row-major matrices, std-only.
//! * `scan_f32`: the R45 8-accumulator dot, compiled once in an AVX2 `#[target_feature]` function and chosen by RUNTIME detection (scores are BIT-IDENTICAL to `flatq::dot8`: the same 8 independent
//!   accumulators, no FMA contraction), with a portable fallback;
//! * `scan_f32_par`: the same scan split over `threads` chunks (std::thread::scope), merged deterministically (score desc, row asc);
//! * `scan_i8`: int8 rows x int8-quantised query with i32 accumulation (query quantised once per search); approximate (RD-LOG R91 reports recall).
//! `TopK` is a bounded selection (no O(n) pair vector).

use std::cmp::Ordering;

#[inline]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0f32; 8];
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

fn scan_f32_portable(data: &[f32], dim: usize, q: &[f32], lo: usize, hi: usize, k: usize) -> Vec<(f32, usize)> {
    let mut t = TopK::new(k);
    for r in lo..hi {
        t.push(dot8(&data[r * dim..(r + 1) * dim], q), r);
    }
    t.finish()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_f32_avx2(data: &[f32], dim: usize, q: &[f32], lo: usize, hi: usize, k: usize) -> Vec<(f32, usize)> {
    scan_f32_portable(data, dim, q, lo, hi, k)
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}
#[cfg(not(target_arch = "x86_64"))]
fn has_avx2() -> bool {
    false
}

fn scan_range(data: &[f32], dim: usize, q: &[f32], lo: usize, hi: usize, k: usize) -> Vec<(f32, usize)> {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: avx2 availability was just checked at runtime
            return unsafe { scan_f32_avx2(data, dim, q, lo, hi, k) };
        }
    }
    scan_f32_portable(data, dim, q, lo, hi, k)
}

/// bounded top-k by (score desc, row asc)
pub struct TopK {
    k: usize,
    v: Vec<(f32, usize)>,
    thr: f32,
}
impl TopK {
    pub fn new(k: usize) -> Self {
        TopK { k: k.max(1), v: Vec::with_capacity(k.max(1) * 2 + 2), thr: f32::NEG_INFINITY }
    }
    #[inline]
    pub fn push(&mut self, s: f32, r: usize) {
        if s < self.thr || s.is_nan() {
            return;
        }
        self.v.push((s, r));
        if self.v.len() > self.k * 2 {
            self.compact();
        }
    }
    fn cmp(a: &(f32, usize), b: &(f32, usize)) -> Ordering {
        b.0.total_cmp(&a.0).then(a.1.cmp(&b.1))
    }
    fn compact(&mut self) {
        let k = self.k;
        if self.v.len() > k {
            self.v.select_nth_unstable_by(k - 1, Self::cmp);
            self.v.truncate(k);
            // the k-th best (worst kept) score; a later candidate strictly below it can never enter the top-k
            self.thr = self.v.iter().map(|x| x.0).fold(f32::INFINITY, f32::min);
        }
    }
    pub fn finish(mut self) -> Vec<(f32, usize)> {
        self.compact();
        self.v.sort_by(Self::cmp);
        self.v
    }
}

fn merge(parts: Vec<Vec<(f32, usize)>>, k: usize) -> Vec<(f32, usize)> {
    let mut t = TopK::new(k);
    for p in parts {
        for (s, r) in p {
            t.push(s, r);
        }
    }
    t.finish()
}

/// the same scan WITHOUT runtime ISA dispatch (baseline for benchmarks: what the default build target compiles)
pub fn scan_f32_baseline(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<(f32, usize)> {
    scan_f32_portable(data, dim, q, 0, data.len() / dim, k)
}

/// top-`k` rows of `data` (row-major, `dim` columns) by dot product with `q`; single thread
pub fn scan_f32(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<(f32, usize)> {
    scan_range(data, dim, q, 0, data.len() / dim, k)
}

/// below this many rows per thread the thread spawn costs more than it saves
pub const PAR_MIN_ROWS: usize = 20_000;

/// as [`scan_f32`] over up to `threads` chunks (one thread when the matrix is smaller than `PAR_MIN_ROWS` rows per thread)
pub fn scan_f32_par(data: &[f32], dim: usize, q: &[f32], k: usize, threads: usize) -> Vec<(f32, usize)> {
    let n = data.len() / dim;
    let t = threads.max(1).min(n / PAR_MIN_ROWS).max(1);
    if t == 1 {
        return scan_f32(data, dim, q, k);
    }
    let step = n.div_ceil(t);
    let parts: Vec<Vec<(f32, usize)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..t).map(|i| sc.spawn(move || scan_range(data, dim, q, (i * step).min(n), ((i + 1) * step).min(n), k))).collect();
        hs.into_iter().map(|h| h.join().expect("scan thread")).collect()
    });
    merge(parts, k)
}

/// per-vector symmetric int8 quantisation: (codes, scale) with x ~= code * scale
pub fn quantize_i8(v: &[f32]) -> (Vec<i8>, f32) {
    let m = v.iter().fold(0f32, |a, x| a.max(x.abs()));
    if m <= 0.0 {
        return (vec![0; v.len()], 0.0);
    }
    let s = m / 127.0;
    (v.iter().map(|x| (x / s).round().clamp(-127.0, 127.0) as i8).collect(), s)
}

#[inline]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0i32; 16];
    for (x, y) in ca.zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] as i32 * y[i] as i32;
        }
    }
    let mut s: i32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += *x as i32 * *y as i32;
    }
    s
}

/// explicit AVX2: sign-extend 16 i8 -> 16 i16 and multiply-add pairs into 8 i32 lanes (vpmaddwd), 2 independent accumulators
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
        let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
        let a1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i + 16) as *const __m128i));
        let b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i + 16) as *const __m128i));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0, b0));
        acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a1, b1));
        i += 32;
    }
    if i + 16 <= n {
        let a0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
        let b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
        acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0, b0));
        i += 16;
    }
    let acc = _mm256_add_epi32(acc0, acc1);
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut s: i32 = lanes.iter().sum();
    while i < n {
        s += *a.get_unchecked(i) as i32 * *b.get_unchecked(i) as i32;
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scan_i8_range_avx2(rows: &[i8], scales: &[f32], dim: usize, q: &[i8], qs: f32, lo: usize, hi: usize, k: usize) -> Vec<(f32, usize)> {
    let mut t = TopK::new(k);
    for r in lo..hi {
        t.push(dot_i8_avx2(&rows[r * dim..(r + 1) * dim], q) as f32 * scales[r] * qs, r);
    }
    t.finish()
}

fn scan_i8_range(rows: &[i8], scales: &[f32], dim: usize, q: &[i8], qs: f32, lo: usize, hi: usize, k: usize) -> Vec<(f32, usize)> {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: avx2 availability was just checked at runtime
            return unsafe { scan_i8_range_avx2(rows, scales, dim, q, qs, lo, hi, k) };
        }
    }
    let mut t = TopK::new(k);
    for r in lo..hi {
        t.push(dot_i8(&rows[r * dim..(r + 1) * dim], q) as f32 * scales[r] * qs, r);
    }
    t.finish()
}

/// full int8 score vector: `out[r] = dot_i8(rows[r], q) * scales[r] * qs` (ISA chosen once per call)
pub fn scores_i8(rows: &[i8], scales: &[f32], dim: usize, q: &[i8], qs: f32, out: &mut Vec<f32>) {
    let n = rows.len() / dim;
    out.clear();
    out.reserve(n);
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            // SAFETY: avx2 availability was just checked at runtime
            unsafe { scores_i8_avx2(rows, scales, dim, q, qs, out) };
            return;
        }
    }
    for r in 0..n {
        out.push(dot_i8(&rows[r * dim..(r + 1) * dim], q) as f32 * scales[r] * qs);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scores_i8_avx2(rows: &[i8], scales: &[f32], dim: usize, q: &[i8], qs: f32, out: &mut Vec<f32>) {
    for r in 0..rows.len() / dim {
        out.push(dot_i8_avx2(&rows[r * dim..(r + 1) * dim], q) as f32 * scales[r] * qs);
    }
}

/// R99: [`scores_i8`] with the rows split over up to `threads` chunks (one thread below `PAR_MIN_ROWS` rows per thread); the result equals the serial one exactly (integer dots, same f32 multiply order per row)
pub fn scores_i8_par(rows: &[i8], scales: &[f32], dim: usize, q: &[i8], qs: f32, out: &mut Vec<f32>, threads: usize) {
    let n = rows.len() / dim;
    let t = threads.max(1).min(n / PAR_MIN_ROWS).max(1);
    if t == 1 {
        return scores_i8(rows, scales, dim, q, qs, out);
    }
    out.clear();
    out.resize(n, 0.0);
    let step = n.div_ceil(t);
    std::thread::scope(|sc| {
        for (i, chunk) in out.chunks_mut(step).enumerate() {
            let lo = i * step;
            let r = &rows[lo * dim..(lo + chunk.len()) * dim];
            let s = &scales[lo..lo + chunk.len()];
            sc.spawn(move || {
                let mut tmp = Vec::new();
                scores_i8(r, s, dim, q, qs, &mut tmp);
                chunk.copy_from_slice(&tmp);
            });
        }
    });
}

/// as [`scan_i8`] over up to `threads` chunks (one thread below `PAR_MIN_ROWS` rows per thread); equals the serial result exactly (integer dots)
pub fn scan_i8_par(rows: &[i8], scales: &[f32], dim: usize, q: &[i8], qs: f32, k: usize, threads: usize) -> Vec<(f32, usize)> {
    let n = rows.len() / dim;
    let t = threads.max(1).min(n / PAR_MIN_ROWS).max(1);
    if t == 1 {
        return scan_i8_range(rows, scales, dim, q, qs, 0, n, k);
    }
    let step = n.div_ceil(t);
    let parts: Vec<Vec<(f32, usize)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..t).map(|i| sc.spawn(move || scan_i8_range(rows, scales, dim, q, qs, (i * step).min(n), ((i + 1) * step).min(n), k))).collect();
        hs.into_iter().map(|h| h.join().expect("scan thread")).collect()
    });
    merge(parts, k)
}

/// top-`k` over int8 rows (`scales[r]` = per-row scale) with a query already quantised by [`quantize_i8`]
pub fn scan_i8(rows: &[i8], scales: &[f32], dim: usize, q: &[i8], qs: f32, k: usize) -> Vec<(f32, usize)> {
    scan_i8_range(rows, scales, dim, q, qs, 0, rows.len() / dim, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatq::dot8 as dot8_ref;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn scan_matches_reference_dot8_bit_for_bit_and_parallel_equals_serial() {
        let (n, d) = (50_000, 40);
        let data = rnd(n * d, 7);
        let q = rnd(d, 9);
        let got = scan_f32(&data, d, &q, 10);
        let mut all: Vec<(f32, usize)> = (0..n).map(|r| (dot8_ref(&data[r * d..(r + 1) * d], &q), r)).collect();
        all.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        all.truncate(10);
        assert_eq!(got, all, "scores and order must equal the reference kernel exactly");
        for t in [2usize, 3, 4] {
            assert_eq!(scan_f32_par(&data, d, &q, 10, t), got, "parallel result must equal serial (threads={t})");
        }
        assert_eq!(scan_f32_par(&data[..d * 100], d, &q, 5, 8), scan_f32(&data[..d * 100], d, &q, 5));
    }

    #[test]
    fn avx2_int8_dot_equals_scalar_for_odd_dims_and_scores_i8_matches_scan() {
        for d in [1usize, 7, 15, 16, 17, 31, 32, 33, 63, 100, 384] {
            let a: Vec<i8> = rnd(d * 3, 5).iter().map(|x| (x * 127.0) as i8).collect();
            let q: Vec<i8> = rnd(d, 6).iter().map(|x| (x * 127.0) as i8).collect();
            let scales = vec![0.5f32, 1.0, 2.0];
            let mut out = Vec::new();
            scores_i8(&a, &scales, d, &q, 0.25, &mut out);
            for r in 0..3 {
                assert_eq!(out[r], dot_i8(&a[r * d..(r + 1) * d], &q) as f32 * scales[r] * 0.25, "dim {d} row {r}");
            }
            let top = scan_i8(&a, &scales, d, &q, 0.25, 3);
            assert_eq!(top.len(), 3);
            assert!(top.windows(2).all(|w| w[0].0 >= w[1].0));
        }
    }

    #[test]
    fn i8_scan_has_high_overlap_with_exact_on_random_unit_vectors() {
        let (n, d) = (3000, 64);
        let mut data = rnd(n * d, 11);
        for r in 0..n {
            let nrm = data[r * d..(r + 1) * d].iter().map(|x| x * x).sum::<f32>().sqrt();
            data[r * d..(r + 1) * d].iter_mut().for_each(|x| *x /= nrm);
        }
        let mut rows = vec![0i8; n * d];
        let mut sc = vec![0f32; n];
        for r in 0..n {
            let (c, s) = quantize_i8(&data[r * d..(r + 1) * d]);
            rows[r * d..(r + 1) * d].copy_from_slice(&c);
            sc[r] = s;
        }
        let mut hit = 0;
        for qi in 0..20 {
            let q = rnd(d, 100 + qi);
            let exact: Vec<usize> = scan_f32(&data, d, &q, 10).into_iter().map(|x| x.1).collect();
            let (qq, qs) = quantize_i8(&q);
            let approx: Vec<usize> = scan_i8(&rows, &sc, d, &qq, qs, 10).into_iter().map(|x| x.1).collect();
            hit += approx.iter().filter(|r| exact.contains(r)).count();
        }
        assert!(hit as f64 / 200.0 > 0.9, "overlap {hit}/200");
    }
}
