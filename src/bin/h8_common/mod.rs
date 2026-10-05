//! H8 (copy of g15_common + extensions) shared helpers: counting allocator, Windows memory probes (FFI, no crates), AVX2 kernels with
//! runtime dispatch (default build target, no -C target-cpu), quantised flat stores, PQ-lite, data loading,
//! ground truth, latency statistics.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::OnceLock;
use std::time::Instant;

use neural_mesh::memory::{self, synth};
use rayon::prelude::*;

// ------------------------------------------------------------------ counting allocator
pub struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size(), Relaxed);
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size(), Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Relaxed);
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, n) };
        if !q.is_null() {
            LIVE.fetch_add(n, Relaxed);
            LIVE.fetch_sub(l.size(), Relaxed);
        }
        q
    }
}
pub fn live_bytes() -> usize {
    LIVE.load(Relaxed)
}

// ------------------------------------------------------------------ Windows memory probes
#[cfg(windows)]
mod win {
    #[repr(C)]
    pub struct Pmc {
        pub cb: u32,
        pub page_faults: u32,
        pub peak_ws: usize,
        pub ws: usize,
        pub a: usize,
        pub b: usize,
        pub c: usize,
        pub d: usize,
        pub pagefile: usize,
        pub peak_pagefile: usize,
    }
    #[repr(C)]
    pub struct MemStatus {
        pub length: u32,
        pub load: u32,
        pub total_phys: u64,
        pub avail_phys: u64,
        pub total_page: u64,
        pub avail_page: u64,
        pub total_virt: u64,
        pub avail_virt: u64,
        pub avail_ext: u64,
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub fn GetCurrentProcess() -> isize;
        pub fn K32GetProcessMemoryInfo(h: isize, p: *mut Pmc, cb: u32) -> i32;
        pub fn GlobalMemoryStatusEx(p: *mut MemStatus) -> i32;
    }
}
/// (working set MB, peak working set MB) of this process.
pub fn rss_mb() -> (f64, f64) {
    #[cfg(windows)]
    unsafe {
        let mut p: win::Pmc = std::mem::zeroed();
        p.cb = std::mem::size_of::<win::Pmc>() as u32;
        if win::K32GetProcessMemoryInfo(win::GetCurrentProcess(), &mut p, p.cb) != 0 {
            return (p.ws as f64 / 1048576.0, p.peak_ws as f64 / 1048576.0);
        }
    }
    (0.0, 0.0)
}
/// Free physical memory in GB (machine-wide).
pub fn free_gb() -> f64 {
    #[cfg(windows)]
    unsafe {
        let mut m: win::MemStatus = std::mem::zeroed();
        m.length = std::mem::size_of::<win::MemStatus>() as u32;
        if win::GlobalMemoryStatusEx(&mut m) != 0 {
            return m.avail_phys as f64 / 1073741824.0;
        }
    }
    0.0
}

// ------------------------------------------------------------------ kernels
pub fn has_avx2() -> bool {
    static H: OnceLock<bool> = OnceLock::new();
    *H.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}
pub fn has_f16c() -> bool {
    static H: OnceLock<bool> = OnceLock::new();
    *H.get_or_init(|| has_avx2() && is_x86_feature_detected!("f16c"))
}

use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: __m256) -> f32 {
    unsafe {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_hadd_ps(s, s);
        let s = _mm_hadd_ps(s, s);
        _mm_cvtss_f32(s)
    }
}
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256_i(v: __m256i) -> i32 {
    unsafe {
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256(v, 1);
        let s = _mm_add_epi32(lo, hi);
        let s = _mm_hadd_epi32(s, s);
        let s = _mm_hadd_epi32(s, s);
        _mm_cvtsi128_si32(s)
    }
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_avx2(a: *const f32, b: *const f32, n: usize) -> f32 {
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i + 8)), _mm256_loadu_ps(b.add(i + 8)), s1);
            i += 16;
        }
        while i + 8 <= n {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.add(i)), _mm256_loadu_ps(b.add(i)), s0);
            i += 8;
        }
        let mut s = hsum256(_mm256_add_ps(s0, s1));
        while i < n {
            s += *a.add(i) * *b.add(i);
            i += 1;
        }
        s
    }
}
/// Best available f32 dot (AVX2+FMA if present).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    if has_avx2() {
        unsafe { dot_avx2(a.as_ptr(), b.as_ptr(), a.len().min(b.len())) }
    } else {
        memory::dot(a, b)
    }
}

/// Scores of `q` against every row of `base` (AVX2 kernel looping internally, so the per-row call inlines).
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_f32_avx2(base: &[f32], dim: usize, q: &[f32], out: &mut [f32]) {
    unsafe {
        for (i, row) in base.chunks_exact(dim).enumerate() {
            out[i] = dot_avx2(row.as_ptr(), q.as_ptr(), dim);
        }
    }
}

pub struct TopK {
    k: usize,
    pub items: Vec<(f32, u32)>, // descending
}
impl TopK {
    pub fn new(k: usize) -> Self {
        Self { k, items: Vec::with_capacity(k + 1) }
    }
    #[inline]
    pub fn thr(&self) -> f32 {
        if self.items.len() < self.k { f32::NEG_INFINITY } else { self.items[self.k - 1].0 }
    }
    #[inline]
    pub fn push(&mut self, s: f32, id: u32) {
        if s <= self.thr() {
            return;
        }
        let pos = self.items.partition_point(|&(x, _)| x >= s);
        self.items.insert(pos, (s, id));
        self.items.truncate(self.k);
    }
    pub fn ids(&self) -> Vec<u32> {
        self.items.iter().map(|x| x.1).collect()
    }
}

/// Exact f32 flat top-k. `lib=true` uses the library's autovectorised (SSE2) dot like src/memory.rs FlatIndex.
pub fn flat_topk(base: &[f32], dim: usize, q: &[f32], k: usize, lib: bool) -> Vec<u32> {
    let n = base.len() / dim;
    let mut top = TopK::new(k);
    if lib || !has_avx2() {
        for (i, row) in base.chunks_exact(dim).enumerate() {
            top.push(memory::dot(row, q), i as u32);
        }
    } else {
        let mut buf = [0f32; 2048];
        let mut i0 = 0;
        while i0 < n {
            let m = (n - i0).min(2048);
            unsafe { scan_f32_avx2(&base[i0 * dim..(i0 + m) * dim], dim, q, &mut buf[..m]) };
            for (j, &s) in buf[..m].iter().enumerate() {
                top.push(s, (i0 + j) as u32);
            }
            i0 += m;
        }
    }
    top.ids()
}

/// Exact rerank of candidate ids using f32 rows from `base`.
pub fn rerank_f32(base: &[f32], dim: usize, q: &[f32], cands: &[u32], k: usize) -> Vec<u32> {
    let mut top = TopK::new(k);
    for &c in cands {
        let row = &base[c as usize * dim..(c as usize + 1) * dim];
        top.push(dot(row, q), c);
    }
    top.ids()
}

// ------------------------------------------------------------------ reduced-precision flat stores
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prec {
    F16,
    Bf16,
    BlockFp16,
    Int8,
}
impl Prec {
    pub fn name(self) -> &'static str {
        match self {
            Prec::F16 => "f16",
            Prec::Bf16 => "bf16",
            Prec::BlockFp16 => "blockfp16(32,i16+f32 scale)",
            Prec::Int8 => "int8(per-vector scale)",
        }
    }
}

pub struct LowStore {
    pub prec: Prec,
    pub dim: usize,
    pub n: usize,
    u16s: Vec<u16>,
    i16s: Vec<i16>,
    i8s: Vec<i8>,
    scales: Vec<f32>, // blockfp16: n*dim/32 ; int8: n
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn f32_to_f16_row(src: *const f32, dst: *mut u16, n: usize) {
    unsafe {
        let mut i = 0;
        while i + 8 <= n {
            let h = _mm256_cvtps_ph::<0>(_mm256_loadu_ps(src.add(i)));
            _mm_storeu_si128(dst.add(i) as *mut __m128i, h);
            i += 8;
        }
        assert!(i == n, "dim must be multiple of 8");
    }
}
#[inline]
fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

impl LowStore {
    pub fn build(prec: Prec, base: &[f32], dim: usize) -> Self {
        let n = base.len() / dim;
        let mut s = LowStore { prec, dim, n, u16s: vec![], i16s: vec![], i8s: vec![], scales: vec![] };
        match prec {
            Prec::F16 => {
                assert!(has_f16c(), "F16C required");
                s.u16s = vec![0u16; n * dim];
                s.u16s.par_chunks_mut(dim).zip(base.par_chunks(dim)).for_each(|(d, r)| unsafe {
                    f32_to_f16_row(r.as_ptr(), d.as_mut_ptr(), dim)
                });
            }
            Prec::Bf16 => {
                s.u16s = base.iter().map(|&x| f32_to_bf16(x)).collect();
            }
            Prec::BlockFp16 => {
                assert!(dim % 32 == 0);
                s.i16s = vec![0i16; n * dim];
                s.scales = vec![0f32; n * dim / 32];
                for (r, (row, o)) in base.chunks_exact(dim).zip(s.i16s.chunks_exact_mut(dim)).enumerate() {
                    for (b, (blk, ob)) in row.chunks_exact(32).zip(o.chunks_exact_mut(32)).enumerate() {
                        let mx = blk.iter().fold(0f32, |m, &x| m.max(x.abs()));
                        let sc = if mx > 0.0 { mx / 32767.0 } else { 1.0 };
                        s.scales[r * (dim / 32) + b] = sc;
                        for (x, o) in blk.iter().zip(ob.iter_mut()) {
                            *o = (x / sc).round().clamp(-32767.0, 32767.0) as i16;
                        }
                    }
                }
            }
            Prec::Int8 => {
                s.i8s = vec![0i8; n * dim];
                s.scales = vec![0f32; n];
                for (r, (row, o)) in base.chunks_exact(dim).zip(s.i8s.chunks_exact_mut(dim)).enumerate() {
                    let mx = row.iter().fold(0f32, |m, &x| m.max(x.abs()));
                    let sc = if mx > 0.0 { mx / 127.0 } else { 1.0 };
                    s.scales[r] = sc;
                    for (x, o) in row.iter().zip(o.iter_mut()) {
                        *o = (x / sc).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            }
        }
        s
    }
    /// Nominal stored bytes (payload + scales).
    pub fn nominal_bytes(&self) -> usize {
        self.u16s.len() * 2 + self.i16s.len() * 2 + self.i8s.len() + self.scales.len() * 4
    }
    pub fn scores(&self, q: &[f32], out: &mut Vec<f32>) {
        out.clear();
        out.resize(self.n, 0.0);
        let dim = self.dim;
        match self.prec {
            Prec::F16 => unsafe { scan_f16(&self.u16s, dim, q, out) },
            Prec::Bf16 => unsafe { scan_bf16(&self.u16s, dim, q, out) },
            Prec::BlockFp16 => unsafe { scan_bfp(&self.i16s, &self.scales, dim, q, out) },
            Prec::Int8 => {
                let mx = q.iter().fold(0f32, |m, &x| m.max(x.abs()));
                let sq = if mx > 0.0 { mx / 127.0 } else { 1.0 };
                let qi: Vec<i8> = q.iter().map(|&x| (x / sq).round().clamp(-127.0, 127.0) as i8).collect();
                unsafe { scan_i8(&self.i8s, &self.scales, dim, &qi, sq, out) };
            }
        }
    }
    /// Top-`k*factor` ids by approximate score, then (optionally) exact f32 rerank down to k.
    pub fn topk(&self, q: &[f32], k: usize, rerank: Option<(&[f32], usize)>, tmp: &mut Vec<f32>) -> Vec<u32> {
        self.scores(q, tmp);
        let fetch = match rerank {
            Some((_, f)) => k * f,
            None => k,
        };
        let mut top = TopK::new(fetch);
        for (i, &s) in tmp.iter().enumerate() {
            top.push(s, i as u32);
        }
        match rerank {
            None => top.ids(),
            Some((base, _)) => rerank_f32(base, self.dim, q, &top.ids(), k),
        }
    }
}

#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn scan_f16(h: &[u16], dim: usize, q: &[f32], out: &mut [f32]) {
    unsafe {
        for (r, row) in h.chunks_exact(dim).enumerate() {
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 8 <= dim {
                let x = _mm256_cvtph_ps(_mm_loadu_si128(row.as_ptr().add(i) as *const __m128i));
                acc = _mm256_fmadd_ps(x, _mm256_loadu_ps(q.as_ptr().add(i)), acc);
                i += 8;
            }
            out[r] = hsum256(acc);
        }
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_bf16(h: &[u16], dim: usize, q: &[f32], out: &mut [f32]) {
    unsafe {
        for (r, row) in h.chunks_exact(dim).enumerate() {
            let mut acc = _mm256_setzero_ps();
            let mut i = 0;
            while i + 8 <= dim {
                let w = _mm256_cvtepu16_epi32(_mm_loadu_si128(row.as_ptr().add(i) as *const __m128i));
                let x = _mm256_castsi256_ps(_mm256_slli_epi32(w, 16));
                acc = _mm256_fmadd_ps(x, _mm256_loadu_ps(q.as_ptr().add(i)), acc);
                i += 8;
            }
            out[r] = hsum256(acc);
        }
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_bfp(m: &[i16], sc: &[f32], dim: usize, q: &[f32], out: &mut [f32]) {
    unsafe {
        let nb = dim / 32;
        for (r, row) in m.chunks_exact(dim).enumerate() {
            let mut tot = _mm256_setzero_ps();
            for b in 0..nb {
                let p = row.as_ptr().add(b * 32);
                let qp = q.as_ptr().add(b * 32);
                let mut a = _mm256_setzero_ps();
                for g in 0..4 {
                    let x = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm_loadu_si128(p.add(g * 8) as *const __m128i)));
                    a = _mm256_fmadd_ps(x, _mm256_loadu_ps(qp.add(g * 8)), a);
                }
                tot = _mm256_fmadd_ps(a, _mm256_set1_ps(sc[r * nb + b]), tot);
            }
            out[r] = hsum256(tot);
        }
    }
}
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_i8(d: &[i8], sc: &[f32], dim: usize, qi: &[i8], sq: f32, out: &mut [f32]) {
    unsafe {
        for (r, row) in d.chunks_exact(dim).enumerate() {
            let mut acc = _mm256_setzero_si256();
            let mut i = 0;
            while i + 16 <= dim {
                let x = _mm256_cvtepi8_epi16(_mm_loadu_si128(row.as_ptr().add(i) as *const __m128i));
                let y = _mm256_cvtepi8_epi16(_mm_loadu_si128(qi.as_ptr().add(i) as *const __m128i));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, y));
                i += 16;
            }
            out[r] = hsum256_i(acc) as f32 * sc[r] * sq;
        }
    }
}

// ------------------------------------------------------------------ PQ-lite (std only)
pub struct Pq {
    pub dim: usize,
    pub sub: usize,
    pub m: usize,
    cent: Vec<f32>, // m*256*sub
    codes: Vec<u8>, // n*m
    pub n: usize,
}
impl Pq {
    pub fn build(base: &[f32], dim: usize, sub: usize, seed: u64) -> Self {
        let n = base.len() / dim;
        let m = dim / sub;
        let ns = n.min(20000);
        let mut rng = synth::Rng::new(seed);
        let sample_idx: Vec<usize> = (0..ns).map(|_| rng.below(n)).collect();
        let cent: Vec<Vec<f32>> = (0..m)
            .into_par_iter()
            .map(|j| {
                let pts: Vec<f32> = sample_idx
                    .iter()
                    .flat_map(|&i| base[i * dim + j * sub..i * dim + (j + 1) * sub].iter().copied())
                    .collect();
                kmeans(&pts, sub, 256, 8, seed ^ (j as u64 * 7919))
            })
            .collect();
        let cent: Vec<f32> = cent.into_iter().flatten().collect();
        let mut codes = vec![0u8; n * m];
        codes.par_chunks_mut(m).zip(base.par_chunks(dim)).for_each(|(c, row)| {
            for j in 0..m {
                let v = &row[j * sub..(j + 1) * sub];
                let cj = &cent[j * 256 * sub..(j + 1) * 256 * sub];
                let mut best = (f32::INFINITY, 0usize);
                for (ci, cv) in cj.chunks_exact(sub).enumerate() {
                    let d: f32 = v.iter().zip(cv).map(|(a, b)| (a - b) * (a - b)).sum();
                    if d < best.0 {
                        best = (d, ci);
                    }
                }
                c[j] = best.1 as u8;
            }
        });
        Pq { dim, sub, m, cent, codes, n }
    }
    pub fn nominal_bytes(&self) -> usize {
        self.codes.len() + self.cent.len() * 4
    }
    pub fn topk(&self, q: &[f32], k: usize, rerank: Option<(&[f32], usize)>) -> Vec<u32> {
        let (m, sub) = (self.m, self.sub);
        let mut lut = vec![0f32; m * 256];
        for j in 0..m {
            let qj = &q[j * sub..(j + 1) * sub];
            for (ci, cv) in self.cent[j * 256 * sub..(j + 1) * 256 * sub].chunks_exact(sub).enumerate() {
                lut[j * 256 + ci] = qj.iter().zip(cv).map(|(a, b)| a * b).sum();
            }
        }
        let fetch = rerank.map_or(k, |(_, f)| k * f);
        let mut top = TopK::new(fetch);
        for (i, c) in self.codes.chunks_exact(m).enumerate() {
            let (mut s0, mut s1, mut s2, mut s3) = (0f32, 0f32, 0f32, 0f32);
            let mut j = 0;
            while j + 4 <= m {
                s0 += lut[j * 256 + c[j] as usize];
                s1 += lut[(j + 1) * 256 + c[j + 1] as usize];
                s2 += lut[(j + 2) * 256 + c[j + 2] as usize];
                s3 += lut[(j + 3) * 256 + c[j + 3] as usize];
                j += 4;
            }
            while j < m {
                s0 += lut[j * 256 + c[j] as usize];
                j += 1;
            }
            top.push(s0 + s1 + s2 + s3, i as u32);
        }
        match rerank {
            None => top.ids(),
            Some((base, _)) => rerank_f32(base, self.dim, q, &top.ids(), k),
        }
    }
}
fn kmeans(pts: &[f32], d: usize, kc: usize, iters: usize, seed: u64) -> Vec<f32> {
    let n = pts.len() / d;
    let mut rng = synth::Rng::new(seed);
    let mut c: Vec<f32> = Vec::with_capacity(kc * d);
    for _ in 0..kc {
        let i = rng.below(n);
        c.extend_from_slice(&pts[i * d..(i + 1) * d]);
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        for (i, p) in pts.chunks_exact(d).enumerate() {
            let mut best = (f32::INFINITY, 0usize);
            for (ci, cv) in c.chunks_exact(d).enumerate() {
                let mut dd = 0f32;
                for t in 0..d {
                    let x = p[t] - cv[t];
                    dd += x * x;
                }
                if dd < best.0 {
                    best = (dd, ci);
                }
            }
            assign[i] = best.1;
        }
        let mut sum = vec![0f32; kc * d];
        let mut cnt = vec![0usize; kc];
        for (i, p) in pts.chunks_exact(d).enumerate() {
            cnt[assign[i]] += 1;
            for t in 0..d {
                sum[assign[i] * d + t] += p[t];
            }
        }
        for ci in 0..kc {
            if cnt[ci] == 0 {
                let i = rng.below(n);
                c[ci * d..(ci + 1) * d].copy_from_slice(&pts[i * d..(i + 1) * d]);
            } else {
                for t in 0..d {
                    c[ci * d + t] = sum[ci * d + t] / cnt[ci] as f32;
                }
            }
        }
    }
    c
}

// ------------------------------------------------------------------ data
pub struct Ds {
    pub name: String,
    pub dim: usize,
    pub base: Vec<f32>,
    pub queries: Vec<f32>,
    pub note: String,
}
impl Ds {
    pub fn n(&self) -> usize {
        self.base.len() / self.dim
    }
    pub fn nq(&self) -> usize {
        self.queries.len() / self.dim
    }
    pub fn q(&self, i: usize) -> &[f32] {
        &self.queries[i * self.dim..(i + 1) * self.dim]
    }
}
pub const SEED: u64 = 0x6715_C0DE;
/// Synthetic hierarchical mixture (same generator/params as memory_bench): queries are the FIRST `nq` draws, base the
/// next `n` (so prefixes of the base are consistent across N).
pub fn synth_ds(dim: usize, n: usize, nq: usize) -> Ds {
    let mut g = synth::Clustered::hierarchical(dim, 64, 16, 0.8, 0.6, SEED);
    let queries = g.sample(nq);
    let mut base = Vec::with_capacity(n * dim);
    let mut left = n;
    while left > 0 {
        let c = left.min(100_000);
        base.extend(g.sample(c));
        left -= c;
    }
    Ds {
        name: format!("synth{}-n{}", dim, n),
        dim,
        base,
        queries,
        note: "hierarchical gaussian mixture 64 topics x16 subs (sub_noise .8, noise .6), L2-normalised; queries = fresh draws".into(),
    }
}
fn read_f32(p: &str) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn norm_rows(v: &mut [f32], dim: usize) {
    v.chunks_exact_mut(dim).for_each(memory::normalize_in_place);
}
/// Dedupe exact-duplicate rows (keep first). Returns (kept, dropped).
fn dedupe(v: Vec<f32>, dim: usize) -> (Vec<f32>, usize) {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(v.len());
    let mut dropped = 0;
    for r in v.chunks_exact(dim) {
        let key: Vec<u32> = r.iter().map(|x| x.to_bits()).collect();
        if seen.insert(key) {
            out.extend_from_slice(r);
        } else {
            dropped += 1;
        }
    }
    (out, dropped)
}
/// Real fixtures: corpora A-D of results/g1_emb_{A..D}_{mini,bge}.f32 concatenated (exact duplicates removed).
/// mini: queries = nq random held-out corpus rows (no stored MiniLM query embeddings). bge: queries = real
/// query-instruction embeddings g1_emb_*_bgei.f32.
pub fn real_ds(model: &str, nq: usize, results_dir: &str) -> Ds {
    let (dim, qsuffix) = if model == "mini" { (384usize, None) } else { (768usize, Some("bgei")) };
    let mut base = Vec::new();
    let mut qall = Vec::new();
    for t in ["A", "B", "C", "D"] {
        base.extend(read_f32(&format!("{results_dir}/g1_emb_{t}_{model}.f32")));
        if let Some(s) = qsuffix {
            qall.extend(read_f32(&format!("{results_dir}/g1_emb_{t}_{s}.f32")));
        }
    }
    norm_rows(&mut base, dim);
    let n0 = base.len() / dim;
    let (mut base, dropped) = dedupe(base, dim);
    let mut rng = synth::Rng::new(SEED ^ 0x77);
    let mut queries = Vec::with_capacity(nq * dim);
    if qsuffix.is_some() {
        norm_rows(&mut qall, dim);
        let nqa = qall.len() / dim;
        for _ in 0..nq {
            let i = rng.below(nqa);
            queries.extend_from_slice(&qall[i * dim..(i + 1) * dim]);
        }
    } else {
        // hold out nq rows: swap-remove
        let mut n = base.len() / dim;
        for _ in 0..nq {
            let i = rng.below(n);
            queries.extend_from_slice(&base[i * dim..(i + 1) * dim]);
            let last = n - 1;
            if i != last {
                let (a, b) = base.split_at_mut(last * dim);
                a[i * dim..(i + 1) * dim].copy_from_slice(&b[..dim]);
            }
            base.truncate(last * dim);
            n -= 1;
        }
    }
    Ds {
        name: format!("real-{model}"),
        dim,
        note: format!(
            "{} rows read from results/g1_emb_[A-D]_{model}.f32, {dropped} exact duplicates dropped -> {} base rows; queries: {}. SYNTHETIC templated corpora (near-duplicate-heavy), real model embeddings.",
            n0,
            base.len() / dim,
            if qsuffix.is_some() { "real bge query-instruction embeddings (g1_emb_*_bgei)" } else { "held-out base rows (no MiniLM query embeddings stored)" }
        ),
        base,
        queries,
    }
}

/// Exact top-k ground truth (f32 AVX2 flat scan), parallel over queries on the global rayon pool.
pub fn ground_truth(base: &[f32], dim: usize, queries: &[f32], k: usize) -> Vec<Vec<u32>> {
    queries.par_chunks(dim).map(|q| flat_topk(base, dim, q, k, false)).collect()
}
pub fn recall(gt: &[u32], got: &[u32], k: usize) -> f64 {
    let kk = k.min(gt.len());
    if kk == 0 {
        return 1.0;
    }
    let hit = got.iter().take(k).filter(|g| gt[..kk].contains(g)).count();
    hit as f64 / kk as f64
}

// ------------------------------------------------------------------ stats
pub fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(((v.len() - 1) as f64) * p).round() as usize]
}
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}
pub fn minmax(v: &[f64]) -> (f64, f64) {
    (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
}
/// {mean,min,max,reps} json of a rep series.
pub fn spread(v: &[f64]) -> serde_json::Value {
    let (lo, hi) = minmax(v);
    serde_json::json!({"mean": mean(v), "min": lo, "max": hi, "reps": v})
}

pub struct Bench {
    pub recall: f64,
    pub recall_min_q: f64,
    pub p50: Vec<f64>,
    pub p99: Vec<f64>,
    pub mean: Vec<f64>,
}
/// Run all queries serially `reps` times (after one untimed warm pass); recall from the warm pass.
pub fn bench_search<F: FnMut(&[f32]) -> Vec<u32>>(mut f: F, ds: &Ds, gt: &[Vec<u32>], k: usize, reps: usize) -> Bench {
    let nq = ds.nq();
    let mut rec_sum = 0.0;
    let mut rec_min: f64 = 1.0;
    for i in 0..nq {
        let got = f(ds.q(i));
        let r = recall(&gt[i], &got, k);
        rec_sum += r;
        rec_min = rec_min.min(r);
    }
    let (mut p50, mut p99, mut mn) = (vec![], vec![], vec![]);
    for _ in 0..reps {
        let mut lat = Vec::with_capacity(nq);
        for i in 0..nq {
            let t = Instant::now();
            let got = f(ds.q(i));
            std::hint::black_box(&got);
            lat.push(t.elapsed().as_secs_f64() * 1e6);
        }
        mn.push(mean(&lat));
        p50.push(pct(&mut lat, 0.5));
        p99.push(pct(&mut lat, 0.99));
    }
    Bench { recall: rec_sum / nq as f64, recall_min_q: rec_min, p50, p99, mean: mn }
}
impl Bench {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "recall_at_10": self.recall, "recall_min_query": self.recall_min_q,
            "p50_us": spread(&self.p50), "p99_us": spread(&self.p99), "mean_us": spread(&self.mean),
        })
    }
}

/// Non-dominated (recall up, p99 down) subset; points are (recall, p99, payload).
pub fn pareto<T: Clone>(pts: &[(f64, f64, T)]) -> Vec<(f64, f64, T)> {
    let mut v: Vec<_> = pts.to_vec();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(b.0.partial_cmp(&a.0).unwrap()));
    let mut out: Vec<(f64, f64, T)> = vec![];
    let mut best = -1.0;
    for p in v {
        if p.0 > best + 1e-12 {
            best = p.0;
            out.push(p);
        }
    }
    out
}

/// Fixed in-cache AVX2 flat-scan micro-benchmark (ms, median of 15): a machine-load drift probe.
pub fn calib_ms() -> f64 {
    let ds = synth_ds(384, 4000, 1);
    let mut ts = vec![];
    for _ in 0..15 {
        let t = Instant::now();
        for _ in 0..20 {
            std::hint::black_box(flat_topk(&ds.base, 384, ds.q(0), 10, false));
        }
        ts.push(t.elapsed().as_secs_f64() * 1e3);
    }
    pct(&mut ts, 0.5)
}

// =====================================================================================================
// ===================================== H8 EXTENSIONS =================================================
// =====================================================================================================
use hnsw_rs::prelude::{Distance, Hnsw};
use std::io::Write;

// ------------------------------------------------------------------ streaming synthetic data (never holds the whole base)
pub struct SynthStream {
    g: synth::Clustered,
    pub dim: usize,
    pub left: usize,
}
/// Same generator/params as `synth_ds`; queries first, then `n` base rows served in chunks.
pub fn synth_stream(dim: usize, n: usize, nq: usize) -> (Vec<f32>, SynthStream) {
    let mut g = synth::Clustered::hierarchical(dim, 64, 16, 0.8, 0.6, SEED);
    let q = g.sample(nq);
    (q, SynthStream { g, dim, left: n })
}
impl SynthStream {
    pub fn next(&mut self, max: usize) -> Option<Vec<f32>> {
        if self.left == 0 {
            return None;
        }
        let c = self.left.min(max);
        self.left -= c;
        Some(self.g.sample(c))
    }
}
pub fn flat_topk_scored(base: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<(f32, u32)> {
    let n = base.len() / dim;
    let mut top = TopK::new(k);
    let mut buf = [0f32; 2048];
    let mut i0 = 0;
    while i0 < n {
        let m = (n - i0).min(2048);
        unsafe { scan_f32_avx2(&base[i0 * dim..(i0 + m) * dim], dim, q, &mut buf[..m]) };
        for (j, &s) in buf[..m].iter().enumerate() {
            top.push(s, (i0 + j) as u32);
        }
        i0 += m;
    }
    top.items
}
/// Exact top-k ground truth streamed chunk by chunk (memory = one chunk).
pub fn ground_truth_stream(mut s: SynthStream, queries: &[f32], dim: usize, k: usize, chunk: usize) -> Vec<Vec<u32>> {
    let nq = queries.len() / dim;
    let mut tops: Vec<TopK> = (0..nq).map(|_| TopK::new(k)).collect();
    let mut off = 0usize;
    while let Some(c) = s.next(chunk) {
        let nc = c.len() / dim;
        let res: Vec<Vec<(f32, u32)>> = queries.par_chunks(dim).map(|q| flat_topk_scored(&c, dim, q, k)).collect();
        for (t, r) in tops.iter_mut().zip(res) {
            for (sc, id) in r {
                t.push(sc, id + off as u32);
            }
        }
        off += nc;
    }
    tops.iter().map(|t| t.ids()).collect()
}

// ------------------------------------------------------------------ quantised vectors as HNSW point data
/// block-fp16 layout in Vec<i16>: [dim mantissas][dim/32 f32 scales, each as 2 i16 (lo,hi)] = 816 B at dim 384.
pub fn encode_bfp(row: &[f32]) -> Vec<i16> {
    let dim = row.len();
    let nb = dim / 32;
    let mut o = vec![0i16; dim + 2 * nb];
    for (b, blk) in row.chunks_exact(32).enumerate() {
        let mx = blk.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let sc = if mx > 0.0 { mx / 32767.0 } else { 1.0 };
        let bits = sc.to_bits();
        o[dim + 2 * b] = (bits & 0xFFFF) as u16 as i16;
        o[dim + 2 * b + 1] = (bits >> 16) as u16 as i16;
        for (j, x) in blk.iter().enumerate() {
            o[b * 32 + j] = (x / sc).round().clamp(-32767.0, 32767.0) as i16;
        }
    }
    o
}
#[inline]
fn bfp_scale(a: &[i16], dim: usize, b: usize) -> f32 {
    f32::from_bits((a[dim + 2 * b] as u16 as u32) | ((a[dim + 2 * b + 1] as u16 as u32) << 16))
}
#[target_feature(enable = "avx2,fma")]
unsafe fn bfp_dot_avx2(a: &[i16], b: &[i16], dim: usize) -> f32 {
    unsafe {
        let nb = dim / 32;
        let mut tot = _mm256_setzero_ps();
        for blk in 0..nb {
            let pa = a.as_ptr().add(blk * 32);
            let pb = b.as_ptr().add(blk * 32);
            let m0 = _mm256_madd_epi16(_mm256_loadu_si256(pa as *const __m256i), _mm256_loadu_si256(pb as *const __m256i));
            let m1 = _mm256_madd_epi16(_mm256_loadu_si256(pa.add(16) as *const __m256i), _mm256_loadu_si256(pb.add(16) as *const __m256i));
            let f = _mm256_add_ps(_mm256_cvtepi32_ps(m0), _mm256_cvtepi32_ps(m1));
            tot = _mm256_fmadd_ps(f, _mm256_set1_ps(bfp_scale(a, dim, blk) * bfp_scale(b, dim, blk)), tot);
        }
        hsum256(tot)
    }
}
pub fn bfp_dot(a: &[i16], b: &[i16]) -> f32 {
    let dim = a.len() * 16 / 17;
    if has_avx2() {
        unsafe { bfp_dot_avx2(a, b, dim) }
    } else {
        let mut t = 0f32;
        for blk in 0..dim / 32 {
            let mut s = 0i64;
            for j in 0..32 {
                s += a[blk * 32 + j] as i64 * b[blk * 32 + j] as i64;
            }
            t += s as f32 * bfp_scale(a, dim, blk) * bfp_scale(b, dim, blk);
        }
        t
    }
}
#[derive(Default, Clone, Copy)]
pub struct BfpDist;
impl Distance<i16> for BfpDist {
    #[inline]
    fn eval(&self, a: &[i16], b: &[i16]) -> f32 {
        (1.0 - bfp_dot(a, b)).max(0.0)
    }
}
/// int8 layout in Vec<i8>: [dim values][4 bytes f32 scale] (388 B at dim 384).
pub fn encode_i8(row: &[f32]) -> Vec<i8> {
    let dim = row.len();
    let mx = row.iter().fold(0f32, |m, &x| m.max(x.abs()));
    let sc = if mx > 0.0 { mx / 127.0 } else { 1.0 };
    let mut o = vec![0i8; dim + 4];
    for (x, o) in row.iter().zip(o.iter_mut()) {
        *o = (x / sc).round().clamp(-127.0, 127.0) as i8;
    }
    for (j, b) in sc.to_le_bytes().iter().enumerate() {
        o[dim + j] = *b as i8;
    }
    o
}
#[target_feature(enable = "avx2,fma")]
unsafe fn i8_dot_avx2(a: &[i8], b: &[i8], dim: usize) -> i32 {
    unsafe {
        let mut acc = _mm256_setzero_si256();
        let mut i = 0;
        while i + 16 <= dim {
            let x = _mm256_cvtepi8_epi16(_mm_loadu_si128(a.as_ptr().add(i) as *const __m128i));
            let y = _mm256_cvtepi8_epi16(_mm_loadu_si128(b.as_ptr().add(i) as *const __m128i));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, y));
            i += 16;
        }
        hsum256_i(acc)
    }
}
pub fn i8_dot(a: &[i8], b: &[i8]) -> f32 {
    let dim = a.len() - 4;
    let sa = f32::from_le_bytes([a[dim] as u8, a[dim + 1] as u8, a[dim + 2] as u8, a[dim + 3] as u8]);
    let sb = f32::from_le_bytes([b[dim] as u8, b[dim + 1] as u8, b[dim + 2] as u8, b[dim + 3] as u8]);
    let s = if has_avx2() { unsafe { i8_dot_avx2(a, b, dim) } } else { (0..dim).map(|i| a[i] as i32 * b[i] as i32).sum() };
    s as f32 * sa * sb
}
#[derive(Default, Clone, Copy)]
pub struct I8Dist;
impl Distance<i8> for I8Dist {
    #[inline]
    fn eval(&self, a: &[i8], b: &[i8]) -> f32 {
        (1.0 - i8_dot(a, b)).max(0.0)
    }
}
#[derive(Default, Clone, Copy)]
pub struct AvxDist;
impl Distance<f32> for AvxDist {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        (1.0 - dot(a, b)).max(0.0)
    }
}

/// Type-erased ANN index so f32 / block-fp16 / int8 stored HNSW variants share one harness.
pub trait Ann: Send + Sync {
    fn insert_batch(&self, rows: &[f32], dim: usize, id0: usize);
    fn insert_one(&self, v: &[f32], id: usize);
    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<u32>;
}
pub struct HnG<T: Clone + Send + Sync + 'static, D: Distance<T> + Send + Sync + 'static> {
    pub h: Hnsw<'static, T, D>,
    pub enc: fn(&[f32]) -> Vec<T>,
}
impl<T: Clone + Send + Sync + 'static, D: Distance<T> + Send + Sync + 'static> Ann for HnG<T, D> {
    fn insert_batch(&self, rows: &[f32], dim: usize, id0: usize) {
        let enc: Vec<Vec<T>> = rows.chunks_exact(dim).map(|r| (self.enc)(r)).collect();
        let items: Vec<(&[T], usize)> = enc.iter().enumerate().map(|(i, v)| (v.as_slice(), id0 + i)).collect();
        self.h.parallel_insert_slice(&items);
    }
    fn insert_one(&self, v: &[f32], id: usize) {
        let e = (self.enc)(v);
        self.h.insert_slice((&e, id));
    }
    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<u32> {
        let e = (self.enc)(q);
        self.h.search(&e, k, ef.max(k)).into_iter().map(|n| n.d_id as u32).collect()
    }
}
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Store {
    F32Lib,
    F32Avx,
    Bfp16,
    Int8,
}
impl Store {
    pub fn name(self) -> &'static str {
        match self {
            Store::F32Lib => "f32 (lib kernel)",
            Store::F32Avx => "f32 (avx2 kernel)",
            Store::Bfp16 => "blockfp16 stored (i16 mantissa+scales, custom dist)",
            Store::Int8 => "int8 stored (custom dist)",
        }
    }
    pub fn short(self) -> &'static str {
        match self {
            Store::F32Lib => "f32lib",
            Store::F32Avx => "f32avx",
            Store::Bfp16 => "bfp16",
            Store::Int8 => "int8",
        }
    }
    pub fn point_bytes(self, dim: usize) -> usize {
        match self {
            Store::F32Lib | Store::F32Avx => dim * 4,
            Store::Bfp16 => (dim + 2 * dim / 32) * 2,
            Store::Int8 => dim + 4,
        }
    }
}
fn id_f32(r: &[f32]) -> Vec<f32> {
    r.to_vec()
}
pub fn new_ann(s: Store, m: usize, efc: usize, maxel: usize) -> Box<dyn Ann> {
    let mx = maxel.max(1);
    match s {
        Store::F32Lib => Box::new(HnG { h: Hnsw::new(m, mx, 16, efc, neural_mesh::memory::CosineDist), enc: id_f32 }),
        Store::F32Avx => Box::new(HnG { h: Hnsw::new(m, mx, 16, efc, AvxDist), enc: id_f32 }),
        Store::Bfp16 => Box::new(HnG { h: Hnsw::new(m, mx, 16, efc, BfpDist), enc: encode_bfp }),
        Store::Int8 => Box::new(HnG { h: Hnsw::new(m, mx, 16, efc, I8Dist), enc: encode_i8 }),
    }
}
pub struct BuiltAnn {
    pub h: Box<dyn Ann>,
    pub build_s: f64,
    pub heap_bytes: usize,
    pub n: usize,
}
/// Build from an in-memory base (2048-row parallel_insert batches), measure heap delta with the counting allocator.
pub fn build_ann(s: Store, base: &[f32], dim: usize, m: usize, efc: usize) -> BuiltAnn {
    let l0 = live_bytes();
    let t = Instant::now();
    let n = base.len() / dim;
    let h = new_ann(s, m, efc, n);
    let mut off = 0;
    while off < n {
        let c = (n - off).min(2048);
        h.insert_batch(&base[off * dim..(off + c) * dim], dim, off);
        off += c;
    }
    BuiltAnn { h, build_s: t.elapsed().as_secs_f64(), heap_bytes: live_bytes().saturating_sub(l0), n }
}
/// Build from a stream (the base is never resident): generates chunks, inserts each in 2048-row batches.
pub fn build_ann_stream(s: Store, mut st: SynthStream, dim: usize, m: usize, efc: usize, n: usize) -> BuiltAnn {
    let l0 = live_bytes();
    let t = Instant::now();
    let h = new_ann(s, m, efc, n);
    let mut off = 0;
    while let Some(c) = st.next(50_000) {
        let nc = c.len() / dim;
        let mut o = 0;
        while o < nc {
            let cc = (nc - o).min(2048);
            h.insert_batch(&c[o * dim..(o + cc) * dim], dim, off + o);
            o += cc;
        }
        off += nc;
    }
    BuiltAnn { h, build_s: t.elapsed().as_secs_f64(), heap_bytes: live_bytes().saturating_sub(l0), n }
}
/// Self-recall check: query every stored vector (or an evenly spaced sample of `max_q`) with its OWN vector; count items whose own id is
/// not the top-1 result, and items absent from the top-10.  Returns (queried, not_top1, not_in_top10).
pub fn self_recall(h: &dyn Ann, base: &[f32], dim: usize, ef: usize, max_q: usize) -> (usize, usize, usize) {
    let n = base.len() / dim;
    let step = (n / max_q.max(1)).max(1);
    let (mut q, mut m1, mut m10) = (0, 0, 0);
    let mut i = 0;
    while i < n {
        let r = h.search(&base[i * dim..(i + 1) * dim], 10, ef);
        q += 1;
        if r.first() != Some(&(i as u32)) {
            m1 += 1;
        }
        if !r.contains(&(i as u32)) {
            m10 += 1;
        }
        i += step;
    }
    (q, m1, m10)
}

// ------------------------------------------------------------------ file-backed (SSD) rerank store
#[cfg(windows)]
use std::os::windows::fs::{FileExt, OpenOptionsExt};
pub struct FileRows {
    f: std::fs::File,
    pub dim: usize,
    pub n: usize,
    pub direct: bool,
    pub path: String,
}
impl FileRows {
    pub fn write_stream(path: &str, st: &mut SynthStream, dim: usize) -> usize {
        let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(path).unwrap());
        let mut n = 0;
        while let Some(c) = st.next(50_000) {
            let b: Vec<u8> = c.iter().flat_map(|x| x.to_le_bytes()).collect();
            f.write_all(&b).unwrap();
            n += c.len() / dim;
        }
        f.flush().unwrap();
        n
    }
    pub fn write_slice(path: &str, base: &[f32], dim: usize) -> usize {
        let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(path).unwrap());
        for c in base.chunks(dim * 20000) {
            let b: Vec<u8> = c.iter().flat_map(|x| x.to_le_bytes()).collect();
            f.write_all(&b).unwrap();
        }
        f.flush().unwrap();
        base.len() / dim
    }
    /// direct=true: FILE_FLAG_NO_BUFFERING (bypasses the OS page cache: every row read hits the device), 4 KiB-aligned reads.
    pub fn open(path: &str, dim: usize, n: usize, direct: bool) -> Self {
        let mut o = std::fs::OpenOptions::new();
        o.read(true);
        #[cfg(windows)]
        if direct {
            o.custom_flags(0x2000_0000 | 0x1000_0000); // NO_BUFFERING | RANDOM_ACCESS
        }
        FileRows { f: o.open(path).unwrap(), dim, n, direct, path: path.to_string() }
    }
    pub fn row_into(&self, id: usize, out: &mut [f32]) {
        let rb = self.dim * 4;
        let off = (id * rb) as u64;
        #[cfg(windows)]
        {
            if !self.direct {
                let mut buf = vec![0u8; rb];
                self.f.seek_read(&mut buf, off).unwrap();
                for (o, c) in out.iter_mut().zip(buf.chunks_exact(4)) {
                    *o = f32::from_le_bytes(c.try_into().unwrap());
                }
            } else {
                let a0 = off / 4096 * 4096;
                let a1 = (off + rb as u64).div_ceil(4096) * 4096;
                let len = (a1 - a0) as usize;
                let mut v = vec![0u8; len + 4096];
                let p = v.as_ptr() as usize;
                let sh = (4096 - p % 4096) % 4096;
                let b = &mut v[sh..sh + len];
                let _ = self.f.seek_read(b, a0).unwrap();
                let s = (off - a0) as usize;
                for (o, c) in out.iter_mut().zip(b[s..s + rb].chunks_exact(4)) {
                    *o = f32::from_le_bytes(c.try_into().unwrap());
                }
            }
        }
    }
}
/// Where the exact f32 rows used for rerank live.
pub enum Rr<'a> {
    Ram(&'a [f32]),
    File(&'a FileRows),
}
impl Rr<'_> {
    pub fn name(&self) -> &'static str {
        match self {
            Rr::Ram(_) => "f32 in RAM",
            Rr::File(f) if f.direct => "f32 on SSD, NO_BUFFERING (cold)",
            Rr::File(_) => "f32 in file, buffered (OS page cache)",
        }
    }
    pub fn rerank(&self, dim: usize, q: &[f32], cands: &[u32], k: usize) -> Vec<u32> {
        match self {
            Rr::Ram(b) => {
                let ok: Vec<u32> = cands.iter().copied().filter(|&i| (i as usize + 1) * dim <= b.len()).collect();
                rerank_f32(b, dim, q, &ok, k)
            }
            Rr::File(f) => {
                let mut c: Vec<u32> = cands.iter().copied().filter(|&i| (i as usize) < f.n).collect();
                c.sort_unstable();
                let mut row = vec![0f32; dim];
                let mut top = TopK::new(k);
                for id in c {
                    f.row_into(id as usize, &mut row);
                    top.push(dot(&row, q), id);
                }
                top.ids()
            }
        }
    }
}

// ------------------------------------------------------------------ RaBitQ-style multi-bit codes (std only)
/// Random orthogonal rotation (Gram-Schmidt of a gaussian matrix), B-bit uniform grid per rotated coordinate with a per-vector
/// grid scale chosen to maximise cos(code, vector), rescale g = 1/<code-mid, o'>, scan over bit-planes
/// with an int8-quantised rotated query.  Code bytes = bits*dim/8 + 4 (one f32 g).
pub struct RaBitQ {
    pub dim: usize,
    pub bits: usize,
    pub n: usize,
    rot: Vec<f32>,
    planes: Vec<u8>,
    g: Vec<f32>,
}
pub fn random_rotation(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = synth::Rng::new(seed);
    let mut m: Vec<f64> = (0..dim * dim).map(|_| rng.normal() as f64).collect();
    for i in 0..dim {
        for j in 0..i {
            let mut d = 0.0;
            for t in 0..dim {
                d += m[i * dim + t] * m[j * dim + t];
            }
            for t in 0..dim {
                m[i * dim + t] -= d * m[j * dim + t];
            }
        }
        let nrm = (0..dim).map(|t| m[i * dim + t] * m[i * dim + t]).sum::<f64>().sqrt();
        for t in 0..dim {
            m[i * dim + t] /= nrm;
        }
    }
    m.iter().map(|&x| x as f32).collect()
}
impl RaBitQ {
    pub fn new(dim: usize, bits: usize, seed: u64) -> Self {
        assert!(dim % 32 == 0 && (1..=8).contains(&bits));
        RaBitQ { dim, bits, n: 0, rot: random_rotation(dim, seed), planes: vec![], g: vec![] }
    }
    pub fn bytes_per_vec(&self) -> usize {
        self.bits * self.dim / 8 + 4
    }
    pub fn rotate(&self, v: &[f32]) -> Vec<f32> {
        self.rot.chunks_exact(self.dim).map(|r| dot(r, v)).collect()
    }
    fn encode(&self, o: &[f32], out: &mut [u8]) -> f32 {
        let (dim, b) = (self.dim, self.bits);
        let mid = ((1u32 << b) - 1) as f32 / 2.0;
        let mx = o.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let mut best = (-1f32, vec![0u8; dim], 0f32);
        let mut c = vec![0u8; dim];
        let steps = if b == 1 { 1 } else { 14 };
        for s in 0..steps {
            let t = if b == 1 { 1.0 } else { 0.35 + 0.65 * s as f32 / (steps - 1) as f32 };
            let delta = (t * mx / mid).max(1e-12);
            let (mut dp, mut nn) = (0f32, 0f32);
            for i in 0..dim {
                let q = (o[i] / delta + mid).round().clamp(0.0, (1u32 << b) as f32 - 1.0);
                c[i] = q as u8;
                let z = q - mid;
                dp += z * o[i];
                nn += z * z;
            }
            let cs = dp / nn.sqrt().max(1e-12);
            if cs > best.0 {
                best = (cs, c.clone(), dp);
            }
        }
        out.fill(0);
        let pb = dim / 8;
        for (i, &q) in best.1.iter().enumerate() {
            for p in 0..b {
                if (q >> p) & 1 == 1 {
                    out[p * pb + i / 8] |= 1 << (i % 8);
                }
            }
        }
        1.0 / best.2.max(1e-12)
    }
    pub fn add(&mut self, rows: &[f32]) {
        let (dim, b) = (self.dim, self.bits);
        let rb = b * dim / 8;
        let nn = rows.len() / dim;
        let mut planes = vec![0u8; nn * rb];
        let me: &RaBitQ = self;
        let g: Vec<f32> = planes
            .par_chunks_mut(rb)
            .zip(rows.par_chunks(dim))
            .map(|(out, r)| {
                let o = me.rotate(r);
                me.encode(&o, out)
            })
            .collect();
        self.planes.extend_from_slice(&planes);
        self.g.extend_from_slice(&g);
        self.n += nn;
    }
    /// approximate scores of all rows (higher = more similar)
    pub fn scores(&self, q: &[f32], out: &mut Vec<f32>) {
        let qr = self.rotate(q);
        let mx = qr.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let sq = if mx > 0.0 { mx / 127.0 } else { 1.0 };
        let qi: Vec<i8> = qr.iter().map(|&x| (x / sq).round().clamp(-127.0, 127.0) as i8).collect();
        let qsum: f32 = qr.iter().sum();
        let mid = ((1u32 << self.bits) - 1) as f32 / 2.0;
        out.clear();
        out.resize(self.n, 0.0);
        let rb = self.bits * self.dim / 8;
        assert!(has_avx2(), "rabitq scan needs AVX2 in this harness");
        unsafe { scan_rabitq(&self.planes, rb, self.bits, self.dim, &qi, &mut out[..]) };
        for (s, &g) in out.iter_mut().zip(&self.g) {
            *s = g * (*s * sq - mid * qsum);
        }
    }
    pub fn topk(&self, q: &[f32], k: usize, rerank: Option<(&Rr, usize)>, tmp: &mut Vec<f32>) -> Vec<u32> {
        self.scores(q, tmp);
        let fetch = rerank.map_or(k, |(_, f)| k * f);
        let mut top = TopK::new(fetch);
        for (i, &s) in tmp.iter().enumerate() {
            top.push(s, i as u32);
        }
        match rerank {
            None => top.ids(),
            Some((rr, _)) => rr.rerank(self.dim, q, &top.ids(), k),
        }
    }
}
/// out[r] = sum_b 2^b * sum_{i: bit b of code i set} qi[i]   (as f32, exact integer)
#[target_feature(enable = "avx2,fma")]
unsafe fn scan_rabitq(planes: &[u8], rb: usize, bits: usize, dim: usize, qi: &[i8], out: &mut [f32]) {
    unsafe {
        let pb = dim / 8;
        let idx = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        let bitm = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        let ones8 = _mm256_set1_epi8(1);
        let ones16 = _mm256_set1_epi16(1);
        let nchunk = dim / 32;
        for (r, row) in planes.chunks_exact(rb).enumerate() {
            let mut tot = _mm256_setzero_si256();
            for p in 0..bits {
                let mut acc16 = _mm256_setzero_si256();
                for c in 0..nchunk {
                    let w = (row.as_ptr().add(p * pb + c * 4) as *const u32).read_unaligned();
                    let v = _mm256_shuffle_epi8(_mm256_set1_epi32(w as i32), idx);
                    let m = _mm256_cmpeq_epi8(_mm256_and_si256(v, bitm), bitm);
                    let qv = _mm256_loadu_si256(qi.as_ptr().add(c * 32) as *const __m256i);
                    let sel = _mm256_and_si256(m, qv);
                    acc16 = _mm256_add_epi16(acc16, _mm256_maddubs_epi16(ones8, sel));
                }
                let a32 = _mm256_madd_epi16(acc16, ones16);
                tot = _mm256_add_epi32(tot, _mm256_sllv_epi32(a32, _mm256_set1_epi32(p as i32)));
            }
            out[r] = hsum256_i(tot) as f32;
        }
    }
}

impl LowStore {
    /// Append another store of the same precision (used to build from chunks without holding the whole f32 base).
    pub fn merge(&mut self, o: LowStore) {
        assert!(self.prec == o.prec && self.dim == o.dim);
        self.n += o.n;
        self.u16s.extend(o.u16s);
        self.i16s.extend(o.i16s);
        self.i8s.extend(o.i8s);
        self.scales.extend(o.scales);
    }
    pub fn empty(prec: Prec, dim: usize) -> Self {
        LowStore { prec, dim, n: 0, u16s: vec![], i16s: vec![], i8s: vec![], scales: vec![] }
    }
}

impl RaBitQ {
    /// float reference of the estimator for one row (decodes the bit-planes), for debugging the SIMD scan
    pub fn debug_exact(&self, q: &[f32], row: usize) -> (f32, f32) {
        let qr = self.rotate(q);
        let (dim, b) = (self.dim, self.bits);
        let pb = dim / 8;
        let rb = b * dim / 8;
        let r = &self.planes[row * rb..(row + 1) * rb];
        let mid = ((1u32 << b) - 1) as f32 / 2.0;
        let mut s = 0f32;
        let mut sq = 0f32;
        for i in 0..dim {
            let mut c = 0u32;
            for p in 0..b {
                c |= (((r[p * pb + i / 8] >> (i % 8)) & 1) as u32) << p;
            }
            s += (c as f32 - mid) * qr[i];
            sq += c as f32;
        }
        (self.g[row] * s, sq)
    }
}

impl RaBitQ {
    pub fn debug_s(&self, q: &[f32], row: usize) -> (f32, f32, f32) {
        let qr = self.rotate(q);
        let mx = qr.iter().fold(0f32, |m, &x| m.max(x.abs()));
        let sq = mx / 127.0;
        let qi: Vec<i8> = qr.iter().map(|&x| (x / sq).round().clamp(-127.0, 127.0) as i8).collect();
        let (dim, b) = (self.dim, self.bits);
        let pb = dim / 8;
        let rb = b * dim / 8;
        let r = &self.planes[row * rb..(row + 1) * rb];
        let mut s = 0i64;
        for i in 0..dim {
            let mut c = 0i64;
            for p in 0..b {
                c |= (((r[p * pb + i / 8] >> (i % 8)) & 1) as i64) << p;
            }
            s += c * qi[i] as i64;
        }
        let mut out = vec![0f32; self.n];
        unsafe { scan_rabitq(&self.planes, rb, self.bits, self.dim, &qi, &mut out[..]) };
        (s as f32, out[row], sq)
    }
}

impl LowStore {
    pub fn shrink(&mut self) {
        self.u16s.shrink_to_fit();
        self.i16s.shrink_to_fit();
        self.i8s.shrink_to_fit();
        self.scales.shrink_to_fit();
    }
}
impl RaBitQ {
    pub fn shrink(&mut self) {
        self.planes.shrink_to_fit();
        self.g.shrink_to_fit();
    }
}

/// Load-robust latency: `reps` interleaved passes over all queries; per-query MINIMUM over the passes, then p50/p99 of those minima.
/// (Preemption/contention spikes rarely hit the same query in every pass; deterministic slow queries stay.)
pub fn bench_min_multi(fs: &mut [&mut dyn FnMut(&[f32]) -> Vec<u32>], ds: &Ds, gt: &[Vec<u32>], k: usize, reps: usize) -> Vec<(f64, f64, f64, f64, f64)> {
    let nq = ds.nq();
    let nf = fs.len();
    let mut rec = vec![0f64; nf];
    for (fi, f) in fs.iter_mut().enumerate() {
        for i in 0..nq {
            let got = f(ds.q(i));
            rec[fi] += recall(&gt[i], &got, k);
        }
    }
    let mut mins = vec![vec![f64::INFINITY; nq]; nf];
    let mut sums = vec![0f64; nf];
    let mut cnt = 0usize;
    let mut raw99: Vec<Vec<f64>> = vec![vec![]; nf];
    for _ in 0..reps {
        for (fi, f) in fs.iter_mut().enumerate() {
            let mut lat = Vec::with_capacity(nq);
            for i in 0..nq {
                let t = Instant::now();
                let got = f(ds.q(i));
                std::hint::black_box(&got);
                let us = t.elapsed().as_secs_f64() * 1e6;
                lat.push(us);
                if us < mins[fi][i] {
                    mins[fi][i] = us;
                }
            }
            sums[fi] += lat.iter().sum::<f64>();
            raw99[fi].push(pct(&mut lat, 0.99));
        }
        cnt += nq;
    }
    (0..nf)
        .map(|fi| {
            let mut m = mins[fi].clone();
            (rec[fi] / nq as f64, pct(&mut m.clone(), 0.5), pct(&mut m, 0.99), sums[fi] / cnt as f64, mean(&raw99[fi]))
        })
        .collect()
}
