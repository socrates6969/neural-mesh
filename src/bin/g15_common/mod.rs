//! G15 shared helpers: counting allocator, Windows memory probes (FFI, no crates), AVX2 kernels with
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
