//! Compact re-implementation of guarded(hebbian-robust) (the recommended default router,
//! `defaults::recommended_router`) over interned/hashed feature ids and selectable cell precision.
//!
//! * decisions are bit-identical to `prior::Guarded<HebbianRouter>` when W = C = f64 (selftest),
//! * cell precision: f64 / f32 / fixed-point u16 / u8 (weights+baselines) and f64 / f32 / Q16 / Q8 (guard counts),
//! * one row per feature = {Hebbian weights[n], baseline, guard counts (s,f)[n]},
//! * optional row budget with LRU / decayed-LFU eviction ("drop cold features"),
//! * zero heap allocations per decision (stack scratch), ids come from a shared `Vocab` or a 32-bit hash.
//!
//! std only. Included via `common::brain`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use neural_mesh::hebbian::{INIT_BASELINE, INIT_WEIGHT};
use neural_mesh::router::{Rng, Router, Task};

pub const MAXN: usize = 64;
pub const MAXF: usize = 32;

// ---------------------------------------------------------------------------------------------
// hashing / vocabulary

#[derive(Default, Clone, Copy)]
pub struct Fnv(u64);
impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        let mut h = if self.0 == 0 { 0xcbf2_9ce4_8422_2325 } else { self.0 };
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = h;
    }
}
pub type FnvMap<K, V> = HashMap<K, V, BuildHasherDefault<Fnv>>;

pub fn hash32(s: &str) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    ((h >> 32) ^ h) as u32
}

/// Feature string -> dense u32 id table (shared across brains when wrapped in an Arc<RwLock>).
#[derive(Default, Clone)]
pub struct Vocab {
    map: FnvMap<Box<str>, u32>,
}
impl Vocab {
    pub fn new() -> Self {
        Vocab::default()
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.map.get(s) {
            return i;
        }
        let i = self.map.len() as u32;
        self.map.insert(s.into(), i);
        i
    }
}

/// How feature strings become ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdMode {
    /// per-process interner (dense ids)
    Table,
    /// stateless 32-bit FNV hash (no table at all; collisions possible, ~n^2 / 2^33)
    Hash32,
}

// ---------------------------------------------------------------------------------------------
// cell precisions

pub trait Cell: Copy + Default + Send + Sync + 'static {
    const NAME: &'static str;
    const BYTES: usize;
    fn enc(x: f64) -> Self;
    fn dec(self) -> f64;
}
impl Cell for f64 {
    const NAME: &'static str = "f64";
    const BYTES: usize = 8;
    #[inline]
    fn enc(x: f64) -> Self {
        x
    }
    #[inline]
    fn dec(self) -> f64 {
        self
    }
}
impl Cell for f32 {
    const NAME: &'static str = "f32";
    const BYTES: usize = 4;
    #[inline]
    fn enc(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn dec(self) -> f64 {
        self as f64
    }
}
/// u16 fixed point on [0,1]
#[derive(Clone, Copy, Default, Debug)]
pub struct Fx16(pub u16);
impl Cell for Fx16 {
    const NAME: &'static str = "u16";
    const BYTES: usize = 2;
    #[inline]
    fn enc(x: f64) -> Self {
        Fx16((x.clamp(0.0, 1.0) * 65535.0).round() as u16)
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 / 65535.0
    }
}
/// u8 fixed point on [0,1]
#[derive(Clone, Copy, Default, Debug)]
pub struct Fx8(pub u8);
impl Cell for Fx8 {
    const NAME: &'static str = "u8";
    const BYTES: usize = 1;
    #[inline]
    fn enc(x: f64) -> Self {
        Fx8((x.clamp(0.0, 1.0) * 255.0).round() as u8)
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 / 255.0
    }
}
/// IEEE-754 binary16 (software conversion, round-to-nearest-even)
#[derive(Clone, Copy, Default, Debug)]
pub struct F16(pub u16);
pub fn f32_to_f16(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut h = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign);
        }
        let mut m = man;
        let mut e: i32 = 127 - 15 + 1;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        return f32::from_bits(sign | ((e as u32) << 23) | (m << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (man << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
}
impl Cell for F16 {
    const NAME: &'static str = "f16";
    const BYTES: usize = 2;
    #[inline]
    fn enc(x: f64) -> Self {
        F16(f32_to_f16(x as f32))
    }
    #[inline]
    fn dec(self) -> f64 {
        f16_to_f32(self.0) as f64
    }
}
/// bfloat16 (top 16 bits of an f32, round-to-nearest-even)
#[derive(Clone, Copy, Default, Debug)]
pub struct Bf16(pub u16);
impl Cell for Bf16 {
    const NAME: &'static str = "bf16";
    const BYTES: usize = 2;
    #[inline]
    fn enc(x: f64) -> Self {
        let b = (x as f32).to_bits();
        let r = ((b >> 16) & 1) + 0x7fff;
        Bf16(((b.wrapping_add(r)) >> 16) as u16)
    }
    #[inline]
    fn dec(self) -> f64 {
        f32::from_bits((self.0 as u32) << 16) as f64
    }
}
/// counts in 1/S units, u16 (max 65535/S)
#[derive(Clone, Copy, Default, Debug)]
pub struct Q16<const S: u32>(pub u16);
impl<const S: u32> Cell for Q16<S> {
    const NAME: &'static str = "q16";
    const BYTES: usize = 2;
    #[inline]
    fn enc(x: f64) -> Self {
        Q16((x * S as f64).round().clamp(0.0, 65535.0) as u16)
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 / S as f64
    }
}
/// counts in 1/S units, u8 (max 255/S)
#[derive(Clone, Copy, Default, Debug)]
pub struct Q8<const S: u32>(pub u8);
impl<const S: u32> Cell for Q8<S> {
    const NAME: &'static str = "q8";
    const BYTES: usize = 1;
    #[inline]
    fn enc(x: f64) -> Self {
        Q8((x * S as f64).round().clamp(0.0, 255.0) as u8)
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 / S as f64
    }
}

// ---------------------------------------------------------------------------------------------
// parameters (recommended defaults: defaults::recommended_guard + recommended_inner)

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub eta: f64,
    pub decay: f64,
    pub base_alpha: f64,
    pub row_cap: f64,
    pub temp: f64,
    pub eps: f64,
    pub tau: f64,
    pub z: f64,
    pub kw_a: f64,
    pub kw_b: f64,
    pub discount: f64,
    pub challenge: bool,
}
impl Params {
    /// `defaults::recommended_router` parameters.
    pub fn recommended() -> Params {
        Params {
            eta: 0.01,
            decay: 0.01,
            base_alpha: 0.1,
            row_cap: 1.0,
            temp: 0.002,
            eps: 0.0,
            tau: 0.7,
            z: 0.5,
            kw_a: 4.0,
            kw_b: 1.0,
            discount: 0.98,
            challenge: true,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// shared maths (used by Compact and by the lock-sharded brain of f3_throughput)

#[inline]
pub fn beta_bounds(a: f64, b: f64, z: f64) -> (f64, f64) {
    let n = a + b;
    let m = a / n;
    let sd = (m * (1.0 - m) / (n + 1.0)).sqrt();
    (m - z * sd, m + z * sd)
}

pub fn softmax_sample(logits: &[f64], rng: &mut Rng) -> usize {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = [0.0f64; MAXN];
    let mut total = 0.0;
    for (i, &l) in logits.iter().enumerate() {
        let e = if l.is_finite() { (l - max).exp() } else { 0.0 };
        exps[i] = e;
        total += e;
    }
    let u = rng.f64();
    if !(total.is_finite() && total > 0.0) {
        return 0;
    }
    let mut acc = 0.0;
    let target = u * total;
    for i in 0..logits.len() {
        acc += exps[i];
        if target < acc {
            return i;
        }
    }
    logits.len() - 1
}

/// Per-decision aggregates over the task's features.
pub struct Agg {
    /// pooled guard counts (sum over raw features) per agent
    pub s: [f64; MAXN],
    pub f: [f64; MAXN],
    /// sum of Hebbian weights over active (deduped) features per agent
    pub score: [f64; MAXN],
}
impl Agg {
    #[inline]
    pub fn new(n: usize) -> Agg {
        let mut a = Agg { s: [0.0; MAXN], f: [0.0; MAXN], score: [0.0; MAXN] };
        let _ = n;
        a.s[0] = 0.0;
        a
    }
}

/// Keyword choice: most votes among candidates, ties to the lowest agent index. `None` = no vote.
#[inline]
pub fn keyword_choice(kw: &[(u32, u16)], raw: &[u32], n: usize, cand: &[usize]) -> Option<usize> {
    let mut votes = [0u16; MAXN];
    let mut any = false;
    for &id in raw {
        if let Ok(p) = kw.binary_search_by_key(&id, |e| e.0) {
            let a = kw[p].1 as usize;
            if a < n && cand.contains(&a) {
                votes[a] += 1;
                any = true;
            }
        }
    }
    if !any {
        return None;
    }
    let mut best: Option<(usize, u16)> = None;
    for a in 0..n {
        let v = votes[a];
        if v == 0 {
            continue;
        }
        match best {
            Some((_, bv)) if bv >= v => {}
            _ => best = Some((a, v)),
        }
    }
    best.map(|b| b.0)
}

/// Guarded decision + inner Hebbian choice from the aggregates.
/// Returns (agent, kind) with kind 0 = keyword, 1 = challenger, 2 = inner learner.
pub fn decide_select(p: &Params, n: usize, kwc: Option<usize>, nraw: usize, agg: &Agg, cand: &[usize], rng: &mut Rng) -> (usize, u8) {
    let nf = nraw as f64;
    let pooled = |a: usize| -> (f64, f64) {
        if nraw == 0 { (0.0, 0.0) } else { (agg.s[a] / nf, agg.f[a] / nf) }
    };
    if let Some(k) = kwc {
        let (s, f) = pooled(k);
        let (_, ucb_k) = beta_bounds(p.kw_a + s, p.kw_b + f, p.z);
        if ucb_k >= p.tau {
            if p.challenge {
                let mut best = (ucb_k, usize::MAX);
                // agents with no evidence share one lower bound (exactly the same expression, computed once)
                let lcb0 = beta_bounds(1.0 + 0.0, 1.0 + 0.0, p.z).0;
                for &a in cand {
                    if a == k {
                        continue;
                    }
                    let lcb = if agg.s[a] == 0.0 && agg.f[a] == 0.0 {
                        lcb0
                    } else {
                        let (s, f) = pooled(a);
                        beta_bounds(1.0 + s, 1.0 + f, p.z).0
                    };
                    if lcb > best.0 {
                        best = (lcb, a);
                    }
                }
                if best.1 != usize::MAX {
                    return (best.1, 1);
                }
            }
            return (k, 0);
        }
    }
    // inner Hebbian: epsilon draw (always consumed), then softmax over candidate scores
    if rng.f64() < p.eps {
        return (cand[rng.below(cand.len())], 2);
    }
    if p.temp <= 0.0 {
        let _ = rng.f64();
        let mut b = 0;
        for i in 1..cand.len() {
            if agg.score[cand[i]] > agg.score[cand[b]] {
                b = i;
            }
        }
        return (cand[b], 2);
    }
    let mut logits = [0.0f64; MAXN];
    for (i, &c) in cand.iter().enumerate() {
        logits[i] = agg.score[c] / p.temp;
    }
    (cand[softmax_sample(&logits[..cand.len()], rng)], 2)
}

/// Hebbian update of one row (weights[n], baseline) for the chosen agent.
#[inline]
pub fn heb_update_row<W: Cell>(p: &Params, w: &mut [W], base: &mut W, agent: usize, r: f64) {
    let n = w.len();
    let mut row = [0.0f64; MAXN];
    for i in 0..n {
        row[i] = w[i].dec();
    }
    let b = base.dec();
    let e = r - b;
    let delta = p.eta * 1.0 * e;
    for x in row[..n].iter_mut() {
        *x *= 1.0 - p.decay;
    }
    row[agent] += delta;
    for x in row[..n].iter_mut() {
        *x = x.clamp(0.0, 1.0);
    }
    let sum: f64 = row[..n].iter().sum();
    if p.row_cap > 0.0 && sum > p.row_cap {
        let k = p.row_cap / sum;
        for x in row[..n].iter_mut() {
            *x *= k;
        }
    }
    for i in 0..n {
        w[i] = W::enc(row[i]);
    }
    *base = W::enc(b + p.base_alpha * (r - b));
}

/// Guard-count update of one row (interleaved (s,f) pairs, 2n cells).
#[inline]
pub fn cnt_update_row<C: Cell>(p: &Params, c: &mut [C], agent: usize, r: f64) {
    if p.discount < 1.0 {
        for x in c.iter_mut() {
            *x = C::enc(x.dec() * p.discount);
        }
    }
    let s = c[2 * agent].dec() + r;
    let f = c[2 * agent + 1].dec() + (1.0 - r);
    c[2 * agent] = C::enc(s);
    c[2 * agent + 1] = C::enc(f);
}

pub fn init_row<W: Cell, C: Cell>(n: usize, w: &mut Vec<W>, base: &mut Vec<W>, cnt: &mut Vec<C>) {
    for _ in 0..n {
        w.push(W::enc(INIT_WEIGHT));
    }
    base.push(W::enc(INIT_BASELINE));
    for _ in 0..2 * n {
        cnt.push(C::enc(0.0));
    }
}

/// Dedupe ids in order (Hebbian active features).
#[inline]
pub fn dedupe(raw: &[u32], out: &mut [u32; MAXF]) -> usize {
    let mut k = 0;
    for &id in raw {
        if !out[..k].contains(&id) {
            out[k] = id;
            k += 1;
        }
    }
    k
}


// ---------------------------------------------------------------------------------------------
// Row stores: dense (n cells per row) and sparse (only agents ever chosen on the row)

pub trait Store: Clone + Send + Sync + 'static {
    fn new(n: usize) -> Self;
    /// append a fresh row (init contents), returns its slot
    fn push_row(&mut self) -> usize;
    fn reset_row(&mut self, slot: usize);
    fn add_counts(&self, slot: usize, s: &mut [f64], f: &mut [f64]);
    fn add_scores(&self, slot: usize, score: &mut [f64]);
    fn cnt_update(&mut self, slot: usize, p: &Params, agent: usize, r: f64);
    fn heb_update(&mut self, slot: usize, p: &Params, agent: usize, r: f64);
    fn heap_bytes(&self) -> usize;
    fn used_bytes(&self) -> usize;
    fn shrink(&mut self);
    /// agents with non-zero guard counts in this row
    fn touched(&self, slot: usize) -> usize;
}

#[derive(Clone)]
pub struct DenseStore<W: Cell, C: Cell> {
    n: usize,
    w: Vec<W>,
    base: Vec<W>,
    cnt: Vec<C>,
}

impl<W: Cell, C: Cell> Store for DenseStore<W, C> {
    fn new(n: usize) -> Self {
        DenseStore { n, w: Vec::new(), base: Vec::new(), cnt: Vec::new() }
    }
    fn push_row(&mut self) -> usize {
        init_row(self.n, &mut self.w, &mut self.base, &mut self.cnt);
        self.base.len() - 1
    }
    fn reset_row(&mut self, s: usize) {
        let n = self.n;
        for i in 0..n {
            self.w[s * n + i] = W::enc(INIT_WEIGHT);
        }
        self.base[s] = W::enc(INIT_BASELINE);
        for i in 0..2 * n {
            self.cnt[s * 2 * n + i] = C::enc(0.0);
        }
    }
    #[inline]
    fn add_counts(&self, slot: usize, s: &mut [f64], f: &mut [f64]) {
        let b = slot * 2 * self.n;
        for a in 0..self.n {
            s[a] += self.cnt[b + 2 * a].dec();
            f[a] += self.cnt[b + 2 * a + 1].dec();
        }
    }
    #[inline]
    fn add_scores(&self, slot: usize, score: &mut [f64]) {
        let b = slot * self.n;
        for a in 0..self.n {
            score[a] += self.w[b + a].dec();
        }
    }
    fn cnt_update(&mut self, slot: usize, p: &Params, agent: usize, r: f64) {
        let n = self.n;
        cnt_update_row(p, &mut self.cnt[slot * 2 * n..(slot + 1) * 2 * n], agent, r);
    }
    fn heb_update(&mut self, slot: usize, p: &Params, agent: usize, r: f64) {
        let n = self.n;
        let (wl, bl) = (&mut self.w[slot * n..(slot + 1) * n], &mut self.base[slot]);
        heb_update_row(p, wl, bl, agent, r);
    }
    fn heap_bytes(&self) -> usize {
        self.w.capacity() * W::BYTES + self.base.capacity() * W::BYTES + self.cnt.capacity() * C::BYTES
    }
    fn used_bytes(&self) -> usize {
        self.w.len() * W::BYTES + self.base.len() * W::BYTES + self.cnt.len() * C::BYTES
    }
    fn shrink(&mut self) {
        self.w.shrink_to_fit();
        self.base.shrink_to_fit();
        self.cnt.shrink_to_fit();
    }
    fn touched(&self, slot: usize) -> usize {
        let b = slot * 2 * self.n;
        (0..self.n).filter(|&a| self.cnt[b + 2 * a].dec() + self.cnt[b + 2 * a + 1].dec() > 0.0).count()
    }
}

/// Sparse rows. Exactness argument: every cell of a row decays / is rescaled by the same factors, so
/// all agents never chosen on the row hold one common weight (`unt`) and zero guard counts; only the
/// chosen ("touched") agents need cells. Decoding a row (untouched <- unt) and re-encoding after an
/// update is bit-identical to the dense layout.
#[derive(Clone)]
pub struct SparseStore<W: Cell, C: Cell> {
    n: usize,
    mask: Vec<u64>,
    off: Vec<u32>,
    unt: Vec<W>,
    base: Vec<W>,
    /// touched cells ordered by agent: weights and (s,f) pairs
    wa: Vec<W>,
    ca: Vec<C>,
    garbage: usize,
}

impl<W: Cell, C: Cell> SparseStore<W, C> {
    #[inline]
    fn rank(mask: u64, a: usize) -> usize {
        (mask & ((1u64 << a) - 1)).count_ones() as usize
    }
    fn compact(&mut self) {
        let mut wa = Vec::with_capacity(self.wa.len() - self.garbage);
        let mut ca = Vec::with_capacity(self.ca.len() - 2 * self.garbage);
        for s in 0..self.mask.len() {
            let len = self.mask[s].count_ones() as usize;
            let o = self.off[s] as usize;
            self.off[s] = wa.len() as u32;
            wa.extend_from_slice(&self.wa[o..o + len]);
            ca.extend_from_slice(&self.ca[2 * o..2 * (o + len)]);
        }
        self.wa = wa;
        self.ca = ca;
        self.garbage = 0;
    }
    /// make sure `agent` has a cell in row `s` (relocating the row with one more cell if needed)
    fn touch_cell(&mut self, s: usize, agent: usize) {
        let m = self.mask[s];
        if m & (1u64 << agent) != 0 {
            return;
        }
        let r = Self::rank(m, agent);
        let len = m.count_ones() as usize;
        let o = self.off[s] as usize;
        let no = self.wa.len();
        for i in 0..r {
            let (w, c0, c1) = (self.wa[o + i], self.ca[2 * (o + i)], self.ca[2 * (o + i) + 1]);
            self.wa.push(w);
            self.ca.push(c0);
            self.ca.push(c1);
        }
        self.wa.push(self.unt[s]);
        self.ca.push(C::enc(0.0));
        self.ca.push(C::enc(0.0));
        for i in r..len {
            let (w, c0, c1) = (self.wa[o + i], self.ca[2 * (o + i)], self.ca[2 * (o + i) + 1]);
            self.wa.push(w);
            self.ca.push(c0);
            self.ca.push(c1);
        }
        self.garbage += len;
        self.mask[s] = m | (1u64 << agent);
        self.off[s] = no as u32;
        if self.garbage > 64 && self.garbage * 2 > self.wa.len() {
            self.compact();
        }
    }
}

impl<W: Cell, C: Cell> Store for SparseStore<W, C> {
    fn new(n: usize) -> Self {
        assert!(n <= 64);
        SparseStore { n, mask: Vec::new(), off: Vec::new(), unt: Vec::new(), base: Vec::new(), wa: Vec::new(), ca: Vec::new(), garbage: 0 }
    }
    fn push_row(&mut self) -> usize {
        self.mask.push(0);
        self.off.push(self.wa.len() as u32);
        self.unt.push(W::enc(INIT_WEIGHT));
        self.base.push(W::enc(INIT_BASELINE));
        self.mask.len() - 1
    }
    fn reset_row(&mut self, s: usize) {
        self.garbage += self.mask[s].count_ones() as usize;
        self.mask[s] = 0;
        self.off[s] = self.wa.len() as u32;
        self.unt[s] = W::enc(INIT_WEIGHT);
        self.base[s] = W::enc(INIT_BASELINE);
    }
    #[inline]
    fn add_counts(&self, s: usize, sv: &mut [f64], fv: &mut [f64]) {
        let mut m = self.mask[s];
        let mut k = self.off[s] as usize;
        while m != 0 {
            let a = m.trailing_zeros() as usize;
            m &= m - 1;
            sv[a] += self.ca[2 * k].dec();
            fv[a] += self.ca[2 * k + 1].dec();
            k += 1;
        }
    }
    #[inline]
    fn add_scores(&self, s: usize, score: &mut [f64]) {
        let m = self.mask[s];
        let u = self.unt[s].dec();
        let mut k = self.off[s] as usize;
        for a in 0..self.n {
            if m & (1u64 << a) != 0 {
                score[a] += self.wa[k].dec();
                k += 1;
            } else {
                score[a] += u;
            }
        }
    }
    fn cnt_update(&mut self, s: usize, p: &Params, agent: usize, r: f64) {
        self.touch_cell(s, agent);
        let m = self.mask[s];
        let o = self.off[s] as usize;
        let len = m.count_ones() as usize;
        // discount: untouched cells are exactly 0 and stay 0, so only touched cells need work
        if p.discount < 1.0 {
            for x in self.ca[2 * o..2 * (o + len)].iter_mut() {
                *x = C::enc(x.dec() * p.discount);
            }
        }
        let k = o + Self::rank(m, agent);
        let sv = self.ca[2 * k].dec() + r;
        let fv = self.ca[2 * k + 1].dec() + (1.0 - r);
        self.ca[2 * k] = C::enc(sv);
        self.ca[2 * k + 1] = C::enc(fv);
    }
    fn heb_update(&mut self, s: usize, p: &Params, agent: usize, r: f64) {
        let n = self.n;
        // the agent is touched already (cnt_update ran first for every row of the task)
        let m = self.mask[s];
        debug_assert!(m & (1u64 << agent) != 0);
        let o = self.off[s] as usize;
        let len = m.count_ones() as usize;
        // identical arithmetic to heb_update_row, untouched cells handled as one value
        let b = self.base[s].dec();
        let e = r - b;
        let delta = p.eta * 1.0 * e;
        let keep = 1.0 - p.decay;
        let mut u = (self.unt[s].dec() * keep).clamp(0.0, 1.0);
        let mut tv = [0.0f64; MAXN];
        for i in 0..len {
            tv[i] = self.wa[o + i].dec() * keep;
        }
        let ka = o + Self::rank(m, agent) - o;
        tv[ka] += delta;
        for x in tv[..len].iter_mut() {
            *x = x.clamp(0.0, 1.0);
        }
        // row sum in agent order (same order as the dense layout)
        let mut sum = 0.0f64;
        let mut i = 0;
        for a in 0..n {
            if m & (1u64 << a) != 0 {
                sum += tv[i];
                i += 1;
            } else {
                sum += u;
            }
        }
        if p.row_cap > 0.0 && sum > p.row_cap {
            let k = p.row_cap / sum;
            for x in tv[..len].iter_mut() {
                *x *= k;
            }
            u *= k;
        }
        for i in 0..len {
            self.wa[o + i] = W::enc(tv[i]);
        }
        if len < n {
            self.unt[s] = W::enc(u);
        }
        self.base[s] = W::enc(b + p.base_alpha * (r - b));
    }
    fn heap_bytes(&self) -> usize {
        self.mask.capacity() * 8
            + self.off.capacity() * 4
            + self.unt.capacity() * W::BYTES
            + self.base.capacity() * W::BYTES
            + self.wa.capacity() * W::BYTES
            + self.ca.capacity() * C::BYTES
    }
    fn used_bytes(&self) -> usize {
        self.mask.len() * 8
            + self.off.len() * 4
            + self.unt.len() * W::BYTES
            + self.base.len() * W::BYTES
            + (self.wa.len() - self.garbage) * W::BYTES
            + (self.ca.len() - 2 * self.garbage) * C::BYTES
    }
    fn shrink(&mut self) {
        if self.garbage > 0 {
            self.compact();
        }
        self.mask.shrink_to_fit();
        self.off.shrink_to_fit();
        self.unt.shrink_to_fit();
        self.base.shrink_to_fit();
        self.wa.shrink_to_fit();
        self.ca.shrink_to_fit();
    }
    fn touched(&self, s: usize) -> usize {
        self.mask[s].count_ones() as usize
    }
}

// ---------------------------------------------------------------------------------------------
// Compact brain over a row store

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evict {
    None,
    Lru,
    /// decayed-count LFU: count halves every `half` ticks
    Lfu { half: u32 },
}

#[derive(Clone)]
pub struct Compact<S: Store> {
    pub n: usize,
    pub p: Params,
    pub kw: Arc<[(u32, u16)]>,
    idx: Vec<(u32, u32)>,
    st: S,
    /// (last tick, decayed count) per slot when eviction is on
    meta: Vec<(u32, f32)>,
    free: Vec<u32>,
    pub max_rows: usize,
    pub evict: Evict,
    /// frequency-gated admission: a feature gets a row only after `admit_min` sightings (hashed u8 counters)
    pub admit_min: u8,
    filter: Vec<u8>,
    tick: u32,
    pub evictions: u64,
    pub dec: [u64; 3],
}

impl<S: Store> Compact<S> {
    pub fn new(n: usize, p: Params, kw: Arc<[(u32, u16)]>) -> Self {
        assert!(n <= MAXN);
        Compact {
            n,
            p,
            kw,
            idx: Vec::new(),
            st: S::new(n),
            meta: Vec::new(),
            free: Vec::new(),
            max_rows: 0,
            evict: Evict::None,
            admit_min: 0,
            filter: Vec::new(),
            tick: 0,
            evictions: 0,
            dec: [0; 3],
        }
    }
    pub fn with_budget(mut self, max_rows: usize, evict: Evict) -> Self {
        self.max_rows = max_rows;
        self.evict = evict;
        self
    }
    /// Row only after `min` sightings; `slots` u8 counters (power of two).
    pub fn with_admission(mut self, min: u8, slots: usize) -> Self {
        assert!(slots.is_power_of_two());
        self.admit_min = min;
        self.filter = vec![0u8; slots];
        self
    }
    pub fn rows(&self) -> usize {
        self.idx.len()
    }
    /// (rows, mean touched agents per row, max touched, fraction of rows with <= 4 touched)
    pub fn touched_stats(&self) -> (usize, f64, usize, f64) {
        let (mut sum, mut mx, mut le4) = (0usize, 0usize, 0usize);
        for &(_, slot) in &self.idx {
            let t = self.st.touched(slot as usize);
            sum += t;
            mx = mx.max(t);
            if t <= 4 {
                le4 += 1;
            }
        }
        let r = self.idx.len().max(1);
        (self.idx.len(), sum as f64 / r as f64, mx, le4 as f64 / r as f64)
    }
    /// Heap payload by capacity (what this struct owns; excludes the shared kw map / vocab).
    pub fn heap_bytes(&self) -> usize {
        self.idx.capacity() * 8 + self.st.heap_bytes() + self.meta.capacity() * 8 + self.free.capacity() * 4 + self.filter.capacity()
    }
    /// Same by length (what a perfectly trimmed copy would need).
    pub fn used_bytes(&self) -> usize {
        self.idx.len() * 8 + self.st.used_bytes() + self.meta.len() * 8 + self.filter.len()
    }
    pub fn shrink(&mut self) {
        self.idx.shrink_to_fit();
        self.st.shrink();
        self.meta.shrink_to_fit();
        self.free.shrink_to_fit();
    }

    #[inline]
    fn find(&self, id: u32) -> Option<usize> {
        self.idx.binary_search_by_key(&id, |e| e.0).ok().map(|i| self.idx[i].1 as usize)
    }

    fn touch(&mut self, slot: usize) {
        if self.evict == Evict::None {
            return;
        }
        let t = self.tick;
        let m = &mut self.meta[slot];
        match self.evict {
            Evict::Lru => m.0 = t,
            Evict::Lfu { half } => {
                let dt = t.saturating_sub(m.0) as f32;
                m.1 = m.1 * (0.5f32).powf(dt / half as f32) + 1.0;
                m.0 = t;
            }
            Evict::None => {}
        }
    }

    fn victim(&self, keep: &[u32]) -> Option<usize> {
        let t = self.tick;
        let mut best: Option<(f32, usize)> = None; // lowest score is evicted
        for (pos, &(fid, slot)) in self.idx.iter().enumerate() {
            if keep.contains(&fid) {
                continue;
            }
            let m = self.meta[slot as usize];
            let score = match self.evict {
                Evict::Lru => m.0 as f32,
                Evict::Lfu { half } => {
                    let dt = t.saturating_sub(m.0) as f32;
                    m.1 * (0.5f32).powf(dt / half as f32)
                }
                Evict::None => return None,
            };
            if best.is_none_or(|b| score < b.0) {
                best = Some((score, pos));
            }
        }
        best.map(|b| b.1)
    }

    fn slot_or_create(&mut self, id: u32, keep: &[u32]) -> Option<usize> {
        match self.idx.binary_search_by_key(&id, |e| e.0) {
            Ok(i) => Some(self.idx[i].1 as usize),
            Err(pos) => {
                if self.admit_min > 1 {
                    let h = (id.wrapping_mul(0x9E37_79B1) >> 5) as usize & (self.filter.len() - 1);
                    if self.filter[h] + 1 < self.admit_min {
                        self.filter[h] += 1;
                        return None;
                    }
                }
                if self.max_rows > 0 && self.idx.len() >= self.max_rows {
                    if let Some(vpos) = self.victim(keep) {
                        let (_, vs) = self.idx.remove(vpos);
                        self.free.push(vs);
                        self.evictions += 1;
                        return self.slot_or_create(id, keep); // position may have shifted
                    }
                }
                let slot = if let Some(s) = self.free.pop() {
                    self.st.reset_row(s as usize);
                    s as usize
                } else {
                    let s = self.st.push_row();
                    if self.evict != Evict::None {
                        self.meta.push((self.tick, 0.0));
                    }
                    s
                };
                if self.evict != Evict::None {
                    self.meta[slot] = (self.tick, 0.0);
                }
                self.idx.insert(pos, (id, slot as u32));
                Some(slot)
            }
        }
    }

    /// Route from feature ids (raw: task.features order, duplicates kept). Returns (agent, kind).
    pub fn route_ids(&self, raw: &[u32], cand: &[usize], rng: &mut Rng) -> (usize, u8) {
        let n = self.n;
        let mut agg = Agg::new(n);
        agg.s[..n].fill(0.0);
        agg.f[..n].fill(0.0);
        agg.score[..n].fill(0.0);
        for &id in raw {
            if let Some(slot) = self.find(id) {
                self.st.add_counts(slot, &mut agg.s[..n], &mut agg.f[..n]);
            }
        }
        let mut act = [0u32; MAXF];
        let na = dedupe(raw, &mut act);
        for &id in &act[..na] {
            match self.find(id) {
                Some(slot) => self.st.add_scores(slot, &mut agg.score[..n]),
                None => {
                    for a in 0..n {
                        agg.score[a] += INIT_WEIGHT;
                    }
                }
            }
        }
        let kwc = keyword_choice(&self.kw, raw, n, cand);
        decide_select(&self.p, n, kwc, raw.len(), &agg, cand, rng)
    }

    pub fn route_ids_mut(&mut self, raw: &[u32], cand: &[usize], rng: &mut Rng) -> usize {
        let (a, k) = self.route_ids(raw, cand, rng);
        self.dec[k as usize] += 1;
        a
    }

    pub fn feedback_ids(&mut self, raw: &[u32], agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        self.tick = self.tick.wrapping_add(1);
        let mut keep = [0u32; MAXF];
        let nk = raw.len().min(MAXF);
        keep[..nk].copy_from_slice(&raw[..nk]);
        for &id in raw {
            if let Some(slot) = self.slot_or_create(id, &keep[..nk]) {
                self.st.cnt_update(slot, &self.p, agent, r);
            }
        }
        let mut act = [0u32; MAXF];
        let na = dedupe(raw, &mut act);
        for &id in &act[..na] {
            if let Some(slot) = self.find(id) {
                self.st.heb_update(slot, &self.p, agent, r);
                self.touch(slot);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ids from strings

pub struct IdBuf {
    pub a: [u32; MAXF],
    pub len: usize,
}
impl IdBuf {
    pub fn as_slice(&self) -> &[u32] {
        &self.a[..self.len]
    }
}

// ---------------------------------------------------------------------------------------------
// object-safe view so schemes can be tables of function pointers

pub trait Brain: Send {
    fn route(&mut self, ids: &[u32], cand: &[usize], rng: &mut Rng) -> usize;
    fn feedback(&mut self, ids: &[u32], agent: usize, reward: f64);
    fn rows(&self) -> usize;
    fn heap_bytes(&self) -> usize;
    fn used_bytes(&self) -> usize;
    fn shrink(&mut self);
    fn clone_box(&self) -> Box<dyn Brain>;
    fn evictions(&self) -> u64;
    fn touched(&self) -> (usize, f64, usize, f64);
}
impl<S: Store> Brain for Compact<S> {
    fn route(&mut self, ids: &[u32], cand: &[usize], rng: &mut Rng) -> usize {
        self.route_ids_mut(ids, cand, rng)
    }
    fn feedback(&mut self, ids: &[u32], agent: usize, reward: f64) {
        self.feedback_ids(ids, agent, reward)
    }
    fn rows(&self) -> usize {
        Compact::rows(self)
    }
    fn heap_bytes(&self) -> usize {
        Compact::heap_bytes(self)
    }
    fn used_bytes(&self) -> usize {
        Compact::used_bytes(self)
    }
    fn shrink(&mut self) {
        Compact::shrink(self)
    }
    fn clone_box(&self) -> Box<dyn Brain> {
        Box::new(self.clone())
    }
    fn evictions(&self) -> u64 {
        self.evictions
    }
    fn touched(&self) -> (usize, f64, usize, f64) {
        self.touched_stats()
    }
}

/// (row budget (0 = unbounded), eviction policy, admission sightings (<= 1 = off))
pub type Budget = Option<(usize, Evict, u8)>;
pub type Maker = fn(usize, Arc<[(u32, u16)]>, Budget) -> Box<dyn Brain>;

fn build<S: Store>(n: usize, kw: Arc<[(u32, u16)]>, budget: Budget) -> Box<dyn Brain> {
    let mut c = Compact::<S>::new(n, Params::recommended(), kw);
    if let Some((m, e, admit)) = budget {
        if m > 0 {
            c = c.with_budget(m, e);
        }
        if admit > 1 {
            c = c.with_admission(admit, 2048);
        }
    }
    Box::new(c)
}
/// dense rows with cell types W (weights/baselines) and C (guard counts)
pub fn mk<W: Cell, C: Cell>(n: usize, kw: Arc<[(u32, u16)]>, budget: Budget) -> Box<dyn Brain> {
    build::<DenseStore<W, C>>(n, kw, budget)
}
/// sparse rows
pub fn mks<W: Cell, C: Cell>(n: usize, kw: Arc<[(u32, u16)]>, budget: Budget) -> Box<dyn Brain> {
    build::<SparseStore<W, C>>(n, kw, budget)
}

/// Router adapter over any `Brain` (string features -> ids via a private vocab or the hash).
pub struct DynRouter {
    pub b: Box<dyn Brain>,
    pub vocab: Vocab,
    pub mode: IdMode,
    label: String,
}
impl DynRouter {
    pub fn new(n: usize, kw: &HashMap<String, usize>, mode: IdMode, mk_fn: Maker, budget: Budget, label: &str) -> Self {
        let mut vocab = Vocab::new();
        let mut v: Vec<(u32, u16)> = kw
            .iter()
            .map(|(k, &a)| (if mode == IdMode::Table { vocab.intern(k) } else { hash32(k) }, a as u16))
            .collect();
        v.sort_by_key(|e| e.0);
        let arc: Arc<[(u32, u16)]> = v.into();
        DynRouter { b: mk_fn(n, arc, budget), vocab, mode, label: label.to_string() }
    }
    fn ids(&mut self, task: &Task) -> IdBuf {
        assert!(task.task_type.is_none());
        let mut buf = IdBuf { a: [0; MAXF], len: 0 };
        for f in task.features.iter().take(MAXF) {
            buf.a[buf.len] = match self.mode {
                IdMode::Table => self.vocab.intern(f),
                IdMode::Hash32 => hash32(f),
            };
            buf.len += 1;
        }
        buf
    }
}
impl Router for DynRouter {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let ids = self.ids(task);
        self.b.route(ids.as_slice(), candidates, rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let ids = self.ids(task);
        self.b.feedback(ids.as_slice(), agent, reward);
    }
}
