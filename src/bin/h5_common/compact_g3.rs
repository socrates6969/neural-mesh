//! VERBATIM COPY of the R3/W3 src/compact.rs (tests stripped, `crate::` -> `neural_mesh::`) used ONLY as the
//! A/B baseline in the h5 benches. Do not edit.
#![allow(dead_code, unused_imports)]
//! `CompactRouter`: the "smallest viable brain" as a library type (opt-in, new module).
//!
//! It is the SAME decision rule as [`crate::defaults::recommended_router`]
//! (`Guarded<HebbianRouter>`: keyword map in front, Hebbian learner behind, Beta-bound
//! hand-over) with a different STATE ENCODING:
//!
//! * feature strings are interned once into a (optionally shared) [`Vocab`] (or hashed to 32 bits,
//!   [`IdMode::Hash32`], with no table at all); rows are found by binary search over `(id, slot)`;
//! * ONE row per feature holds the Hebbian weights, the reward baseline and the guard counts;
//! * rows are SPARSE: an agent that was never chosen on a feature row has weight = the row's common
//!   "untouched" value and zero counts, so only chosen agents get cells (an exact transform: the
//!   sparse and dense layouts are bit-identical, see the tests);
//! * cells are quantised: weights/baselines as `u16` fixed point on [0,1], guard counts as `u16` in
//!   1/1024 units ([`Fx16`], [`Q16`]). The generic core also instantiates with `f32` and `f64` cells
//!   (`CompactRouterF32`, `CompactRouterF64`); the `f64` instance is decision- and rng-identical to
//!   `defaults::recommended_router` (unit-tested), so every difference of the `u16` instance is
//!   representation error, not logic.
//!
//! Limits (documented, enforced by the constructor): at most 64 agents (sparse mask is a `u64`);
//! only the plain Hebbian rule (the opt-in `meta`/`weighting`/`ucb`/`agg` options are rejected);
//! the agent count is fixed (updates for an agent index >= n are ignored, candidates must be < n).
//! Nothing here changes a default; the type is not wired into the CLI.
//!
//! The router is not internally synchronised (the `Router` trait takes `&mut self`); the vocabulary
//! may be shared between many brains through [`SharedVocab`] (an `Arc<RwLock<Vocab>>`).

use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::{Arc, RwLock};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use neural_mesh::defaults::{recommended_guard, recommended_inner};
use neural_mesh::hebbian::{AggKind, HebbianConfig, INIT_BASELINE, INIT_WEIGHT, MetaKind, WeightKind};
use neural_mesh::prior::GuardConfig;
use neural_mesh::router::{Rng, Router, Task};

/// Maximum number of agents (sparse touched-agent mask is one `u64`).
pub const MAX_AGENTS: usize = 64;
const MAXF: usize = 32;
const TEMP_BASE: u32 = u32::MAX;

// ---------------------------------------------------------------------------------------------
// errors

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactError(pub String);
impl fmt::Display for CompactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "compact router: {}", self.0)
    }
}
impl std::error::Error for CompactError {}
fn err<T>(s: impl Into<String>) -> Result<T, CompactError> {
    Err(CompactError(s.into()))
}

// ---------------------------------------------------------------------------------------------
// cells

/// Storage precision of one table cell.
pub trait Cell: Copy + Default + Send + Sync + 'static + Serialize + DeserializeOwned {
    const BYTES: usize;
    const NAME: &'static str;
    fn enc(x: f64) -> Self;
    fn dec(self) -> f64;
}
impl Cell for f64 {
    const BYTES: usize = 8;
    const NAME: &'static str = "f64";
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
    const BYTES: usize = 4;
    const NAME: &'static str = "f32";
    #[inline]
    fn enc(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn dec(self) -> f64 {
        self as f64
    }
}
/// Fixed point on [0,1], 65535 steps (weights and baselines).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fx16(pub u16);
impl Cell for Fx16 {
    const BYTES: usize = 2;
    const NAME: &'static str = "u16";
    #[inline]
    fn enc(x: f64) -> Self {
        Fx16((x.clamp(0.0, 1.0) * 65535.0).round() as u16)
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 / 65535.0
    }
}
/// Guard counts in 1/1024 units, saturating at 63.999 (counts of the recommended guard stay below
/// 1/(1-0.98) = 50 because of the 0.98 discount).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Q16(pub u16);
impl Cell for Q16 {
    const BYTES: usize = 2;
    const NAME: &'static str = "q16";
    #[inline]
    fn enc(x: f64) -> Self {
        Q16((x * 1024.0).round().clamp(0.0, 65535.0) as u16)
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 / 1024.0
    }
}

// ---------------------------------------------------------------------------------------------
// vocabulary

#[derive(Default, Clone, Copy)]
struct Fnv(u64);
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
type FnvMap<K, V> = HashMap<K, V, BuildHasherDefault<Fnv>>;

/// Stateless 32-bit FNV-1a fold (collision probability ~ n^2 / 2^33 for n features).
pub fn hash32(s: &str) -> u32 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    ((h >> 32) ^ h) as u32
}

/// Feature string <-> dense `u32` id table.
#[derive(Default, Clone, Debug)]
pub struct Vocab {
    map: FnvMap<Arc<str>, u32>,
    names: Vec<Arc<str>>,
}
impl Vocab {
    pub fn new() -> Self {
        Vocab::default()
    }
    pub fn len(&self) -> usize {
        self.names.len()
    }
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
    pub fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }
    pub fn name(&self, id: u32) -> Option<&str> {
        self.names.get(id as usize).map(|a| &**a)
    }
    pub fn intern(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.map.get(s) {
            return i;
        }
        let i = self.names.len() as u32;
        let a: Arc<str> = Arc::from(s);
        self.map.insert(a.clone(), i);
        self.names.push(a);
        i
    }
    /// Estimated heap bytes (hashbrown layout: buckets x (key 16 + value 4, padded to 24) + 1 control
    /// byte each; name vector 16 B per entry; each string once + 16 B Arc header). An ESTIMATE; the
    /// g3 benchmark cross-checks it against a counting allocator.
    pub fn bytes_estimate(&self) -> usize {
        let cap = self.map.capacity();
        let buckets = if cap == 0 { 0 } else { (cap * 8 / 7).next_power_of_two() };
        let strings: usize = self.names.iter().map(|s| s.len() + 16).sum();
        buckets * 25 + self.names.capacity() * 16 + strings
    }
}

/// A vocabulary shared between many brains (interning takes the write lock only for NEW strings).
#[derive(Clone, Default, Debug)]
pub struct SharedVocab(Arc<RwLock<Vocab>>);
impl SharedVocab {
    pub fn new() -> Self {
        SharedVocab::default()
    }
    fn read(&self) -> std::sync::RwLockReadGuard<'_, Vocab> {
        self.0.read().unwrap_or_else(|e| e.into_inner())
    }
    fn write(&self) -> std::sync::RwLockWriteGuard<'_, Vocab> {
        self.0.write().unwrap_or_else(|e| e.into_inner())
    }
    pub fn len(&self) -> usize {
        self.read().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn intern(&self, s: &str) -> u32 {
        if let Some(i) = self.read().get(s) {
            return i;
        }
        self.write().intern(s)
    }
    pub fn bytes_estimate(&self) -> usize {
        self.read().bytes_estimate()
    }
    pub fn same(&self, other: &SharedVocab) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

/// How feature strings become ids.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdMode {
    /// interned through a [`Vocab`] (private or shared); exact.
    #[default]
    Table,
    /// stateless 32-bit hash, no table at all (collisions possible, ~n^2/2^33).
    Hash32,
}

// ---------------------------------------------------------------------------------------------
// configuration

/// Parameters of the compact brain (flat; `Default` = `defaults::recommended_guard()` +
/// `defaults::recommended_inner()`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CompactConfig {
    // inner Hebbian rule
    pub eta: f64,
    pub epsilon: f64,
    pub temperature: f64,
    pub decay: f64,
    pub baseline_alpha: f64,
    pub row_cap: f64,
    // guard
    pub tau: f64,
    pub z: f64,
    pub kw_a: f64,
    pub kw_b: f64,
    pub discount: f64,
    pub challenge: bool,
    pub id_mode: IdMode,
}

impl Default for CompactConfig {
    fn default() -> Self {
        Self::from_configs(&recommended_guard(), &recommended_inner()).expect("recommended config is plain")
    }
}

impl CompactConfig {
    /// Build from the crate's configs; rejects Hebbian options this encoding does not implement.
    pub fn from_configs(g: &GuardConfig, h: &HebbianConfig) -> Result<Self, CompactError> {
        if h.meta != MetaKind::Off || h.weighting != WeightKind::Off || h.ucb_c > 0.0 || h.agg != AggKind::Sum {
            return err("only the plain Hebbian rule is supported (meta/weighting/ucb/agg must be off)");
        }
        Ok(CompactConfig {
            eta: h.eta,
            epsilon: h.epsilon,
            temperature: h.temperature,
            decay: h.decay,
            baseline_alpha: h.baseline_alpha,
            row_cap: h.row_cap,
            tau: g.tau,
            z: g.z,
            kw_a: g.kw_a,
            kw_b: g.kw_b,
            discount: g.discount,
            challenge: g.challenge,
            id_mode: IdMode::Table,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// sparse row store

/// One entry per feature row: common value of the untouched agents' weights, reward baseline, and the
/// cells (weight, success count, failure count) of the agents chosen on that row.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
struct Rows<W: Cell, C: Cell> {
    mask: Vec<u64>,
    off: Vec<u32>,
    unt: Vec<W>,
    base: Vec<W>,
    wa: Vec<W>,
    ca: Vec<C>,
    garbage: usize,
}

impl<W: Cell, C: Cell> Rows<W, C> {
    fn new() -> Self {
        Rows { mask: Vec::new(), off: Vec::new(), unt: Vec::new(), base: Vec::new(), wa: Vec::new(), ca: Vec::new(), garbage: 0 }
    }
    fn len(&self) -> usize {
        self.mask.len()
    }
    fn push_row(&mut self) -> usize {
        self.mask.push(0);
        self.off.push(self.wa.len() as u32);
        self.unt.push(W::enc(INIT_WEIGHT));
        self.base.push(W::enc(INIT_BASELINE));
        self.mask.len() - 1
    }
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
    /// give `agent` a cell in row `s` (the row is re-appended with one more cell; old copy = garbage)
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
    fn add_scores(&self, s: usize, n: usize, score: &mut [f64]) {
        let m = self.mask[s];
        let u = self.unt[s].dec();
        let mut k = self.off[s] as usize;
        for (a, sc) in score.iter_mut().enumerate().take(n) {
            if m & (1u64 << a) != 0 {
                *sc += self.wa[k].dec();
                k += 1;
            } else {
                *sc += u;
            }
        }
    }
    fn cnt_update(&mut self, s: usize, discount: f64, agent: usize, r: f64) {
        self.touch_cell(s, agent);
        let m = self.mask[s];
        let o = self.off[s] as usize;
        let len = m.count_ones() as usize;
        // untouched cells are exactly 0 and stay 0, so only touched cells are discounted
        if discount < 1.0 {
            for x in self.ca[2 * o..2 * (o + len)].iter_mut() {
                *x = C::enc(x.dec() * discount);
            }
        }
        let k = o + Self::rank(m, agent);
        let sv = self.ca[2 * k].dec() + r;
        let fv = self.ca[2 * k + 1].dec() + (1.0 - r);
        self.ca[2 * k] = C::enc(sv);
        self.ca[2 * k + 1] = C::enc(fv);
    }
    /// Hebbian update of row `s` (same arithmetic and summation order as the dense rule).
    fn heb_update(&mut self, s: usize, cfg: &CompactConfig, n: usize, agent: usize, r: f64) {
        self.touch_cell(s, agent);
        let m = self.mask[s];
        let o = self.off[s] as usize;
        let len = m.count_ones() as usize;
        let b = self.base[s].dec();
        let delta = cfg.eta * 1.0 * (r - b);
        let keep = 1.0 - cfg.decay;
        let mut u = (self.unt[s].dec() * keep).clamp(0.0, 1.0);
        let mut tv = [0.0f64; MAX_AGENTS];
        for i in 0..len {
            tv[i] = self.wa[o + i].dec() * keep;
        }
        tv[Self::rank(m, agent)] += delta;
        for x in tv[..len].iter_mut() {
            *x = x.clamp(0.0, 1.0);
        }
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
        if cfg.row_cap > 0.0 && sum > cfg.row_cap {
            let k = cfg.row_cap / sum;
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
        self.base[s] = W::enc(b + cfg.baseline_alpha * (r - b));
    }
    fn used_bytes(&self) -> usize {
        self.mask.len() * 8
            + self.off.len() * 4
            + self.unt.len() * W::BYTES
            + self.base.len() * W::BYTES
            + (self.wa.len() - self.garbage) * W::BYTES
            + (self.ca.len() - 2 * self.garbage) * C::BYTES
    }
    fn reserved_bytes(&self) -> usize {
        self.mask.capacity() * 8
            + self.off.capacity() * 4
            + self.unt.capacity() * W::BYTES
            + self.base.capacity() * W::BYTES
            + self.wa.capacity() * W::BYTES
            + self.ca.capacity() * C::BYTES
    }
    fn touched_cells(&self) -> usize {
        self.mask.iter().map(|m| m.count_ones() as usize).sum()
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
    fn validate(&self, n: usize) -> Result<(), CompactError> {
        let r = self.mask.len();
        if self.off.len() != r || self.unt.len() != r || self.base.len() != r || self.garbage != 0 {
            return err("corrupt snapshot: row vector lengths");
        }
        let mut expect = 0usize;
        for s in 0..r {
            if self.off[s] as usize != expect {
                return err("corrupt snapshot: row offsets");
            }
            if n < 64 && self.mask[s] >> n != 0 {
                return err("corrupt snapshot: mask bit beyond n_agents");
            }
            expect += self.mask[s].count_ones() as usize;
        }
        if self.wa.len() != expect || self.ca.len() != 2 * expect {
            return err("corrupt snapshot: cell counts");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// decision maths (identical to prior::Guarded + HebbianRouter with the plain rule)

#[inline]
fn beta_bounds(a: f64, b: f64, z: f64) -> (f64, f64) {
    let n = a + b;
    let m = a / n;
    let sd = (m * (1.0 - m) / (n + 1.0)).sqrt();
    (m - z * sd, m + z * sd)
}

fn softmax_sample(logits: &[f64], rng: &mut Rng) -> usize {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = [0.0f64; MAX_AGENTS];
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
    for (i, e) in exps[..logits.len()].iter().enumerate() {
        acc += e;
        if target < acc {
            return i;
        }
    }
    logits.len() - 1
}

/// small list of ids that stays on the stack for the usual task sizes
struct IdList {
    buf: [u32; MAXF],
    len: usize,
    heap: Vec<u32>,
    spilled: bool,
}
impl IdList {
    fn new() -> Self {
        IdList { buf: [0; MAXF], len: 0, heap: Vec::new(), spilled: false }
    }
    #[inline]
    fn push(&mut self, id: u32) {
        if self.spilled {
            self.heap.push(id);
        } else if self.len < MAXF {
            self.buf[self.len] = id;
            self.len += 1;
        } else {
            self.heap.extend_from_slice(&self.buf);
            self.heap.push(id);
            self.spilled = true;
        }
    }
    #[inline]
    fn as_slice(&self) -> &[u32] {
        if self.spilled { &self.heap } else { &self.buf[..self.len] }
    }
    fn as_mut_slice(&mut self) -> &mut [u32] {
        if self.spilled { &mut self.heap } else { &mut self.buf[..self.len] }
    }
    fn dedupe_into(&self, out: &mut IdList) {
        for &id in self.as_slice() {
            if !out.as_slice().contains(&id) {
                out.push(id);
            }
        }
    }
}

/// feature strings of one task resolved to ids
struct Resolved {
    raw: IdList,
    /// id of the raw `task_type` string (keyword votes only)
    kw_type: Option<u32>,
    /// id of "type:<task_type>" (Hebbian feature)
    type_row: Option<u32>,
    /// strings without an id yet (Table mode); temp id of unknown[i] = TEMP_BASE - i
    unknown: Vec<String>,
}

fn resolve_in(get: &dyn Fn(&str) -> Option<u32>, task: &Task) -> Resolved {
    let mut unknown: Vec<String> = Vec::new();
    let mut id_of = |s: &str| -> u32 {
        match get(s) {
            Some(id) => id,
            None => {
                if let Some(p) = unknown.iter().position(|u| u == s) {
                    TEMP_BASE - p as u32
                } else {
                    unknown.push(s.to_string());
                    TEMP_BASE - (unknown.len() as u32 - 1)
                }
            }
        }
    };
    let mut raw = IdList::new();
    for f in &task.features {
        raw.push(id_of(f));
    }
    let (kw_type, type_row) = match &task.task_type {
        Some(t) => {
            let k = id_of(t);
            let tr = id_of(&format!("type:{t}"));
            (Some(k), Some(tr))
        }
        None => (None, None),
    };
    Resolved { raw, kw_type, type_row, unknown }
}

// ---------------------------------------------------------------------------------------------
// the router

enum IdSource {
    Hash32,
    Private(Vocab),
    Shared(SharedVocab),
}

/// Byte accounting of one brain (see the module docs; all figures in bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ByteReport {
    /// `size_of` the router struct itself
    pub struct_bytes: usize,
    /// feature index `(id, slot)` entries, by length (8 B per feature row)
    pub index_bytes: usize,
    /// sparse rows, by length (mask 8 + offset 4 + untouched 2 + baseline 2 per row, + cells)
    pub row_bytes: usize,
    /// heap actually reserved (capacity based) for index + rows; >= index_bytes + row_bytes
    pub reserved_bytes: usize,
    /// feature rows
    pub rows: usize,
    /// touched (feature, agent) cells (each = 1 weight + 2 counts = 6 B in the u16 instance)
    pub touched_cells: usize,
    /// shared vocabulary (estimate); 0 in Hash32 mode
    pub shared_vocab_bytes: usize,
    /// keyword table (8 B per keyword; shared via Arc between forks)
    pub shared_kw_bytes: usize,
}
impl ByteReport {
    /// bytes owned by this brain alone (exact, length based)
    pub fn owned(&self) -> usize {
        self.struct_bytes + self.index_bytes + self.row_bytes
    }
    /// owned bytes if the allocator rounds up to capacity
    pub fn owned_reserved(&self) -> usize {
        self.struct_bytes + self.reserved_bytes
    }
    /// owned + shared pieces (what the brain costs if it had to carry vocab and keyword map privately)
    pub fn private_total(&self) -> usize {
        self.owned() + self.shared_vocab_bytes + self.shared_kw_bytes
    }
}

/// Serialisable state (`Rows` holds only plain integers/floats). Feature strings are stored once
/// (Table mode) as `names`; `idx` refers to positions in `names` (Hash32 mode: to the hash ids).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Snapshot<W: Cell, C: Cell> {
    pub version: u32,
    pub n_agents: usize,
    pub cfg: CompactConfig,
    pub names: Vec<String>,
    idx: Vec<(u32, u32)>,
    pub keywords: Vec<(String, u16)>,
    rows: Rows<W, C>,
    pub updates: u64,
    pub decisions: [u64; 3],
}

pub struct CompactRouterG<W: Cell, C: Cell> {
    n: usize,
    cfg: CompactConfig,
    ids: IdSource,
    /// keyword id -> agent, sorted by id (shared between forks)
    kw: Arc<[(u32, u16)]>,
    /// keyword strings (kept only in Hash32 mode, where ids cannot be inverted)
    kw_names: Arc<[(Box<str>, u16)]>,
    idx: Vec<(u32, u32)>,
    rows: Rows<W, C>,
    updates: u64,
    /// decisions taken: (keyword, challenger, inner learner)
    pub decisions: [u64; 3],
}

/// The recommended compact brain: sparse rows, u16 weights, Q16 guard counts.
pub type CompactRouter = CompactRouterG<Fx16, Q16>;
/// f32 cells (decision-identical to f64 in the g3/f3 measurements; half the bytes).
pub type CompactRouterF32 = CompactRouterG<f32, f32>;
/// f64 cells: decision- and rng-identical to `defaults::recommended_router` (sparse layout only).
pub type CompactRouterF64 = CompactRouterG<f64, f64>;

fn build_kw(map: &HashMap<String, usize>, n: usize, mut id_of: impl FnMut(&str) -> u32) -> Vec<(u32, u16)> {
    let mut v: Vec<(u32, u16)> = map.iter().filter(|(_, a)| **a < n.max(1) && **a <= u16::MAX as usize).map(|(k, &a)| (id_of(k), a as u16)).collect();
    v.sort_by_key(|e| e.0);
    v.dedup_by_key(|e| e.0);
    v
}

impl<W: Cell, C: Cell> CompactRouterG<W, C> {
    fn check(n: usize, cfg: &CompactConfig) -> Result<(), CompactError> {
        if n == 0 || n > MAX_AGENTS {
            return err(format!("n_agents must be in 1..={MAX_AGENTS}, got {n}"));
        }
        if !(cfg.tau.is_finite() && cfg.z.is_finite() && cfg.kw_a > 0.0 && cfg.kw_b > 0.0) {
            return err("invalid guard parameters");
        }
        Ok(())
    }

    /// New brain with a PRIVATE vocabulary (or no vocabulary in Hash32 mode).
    pub fn new(n_agents: usize, keywords: &HashMap<String, usize>, cfg: CompactConfig) -> Result<Self, CompactError> {
        Self::check(n_agents, &cfg)?;
        match cfg.id_mode {
            IdMode::Hash32 => {
                let kw = build_kw(keywords, n_agents, hash32);
                Ok(Self::assemble(n_agents, cfg, IdSource::Hash32, kw, Self::names_of(keywords, n_agents, true)))
            }
            IdMode::Table => {
                let mut v = Vocab::new();
                let kw = build_kw(keywords, n_agents, |s| v.intern(s));
                Ok(Self::assemble(n_agents, cfg, IdSource::Private(v), kw, Arc::from(Vec::new())))
            }
        }
    }

    /// New brain over a SHARED vocabulary (Table mode only; `cfg.id_mode` is forced to Table).
    pub fn with_shared_vocab(n_agents: usize, keywords: &HashMap<String, usize>, mut cfg: CompactConfig, vocab: &SharedVocab) -> Result<Self, CompactError> {
        cfg.id_mode = IdMode::Table;
        Self::check(n_agents, &cfg)?;
        let kw = build_kw(keywords, n_agents, |s| vocab.intern(s));
        Ok(Self::assemble(n_agents, cfg, IdSource::Shared(vocab.clone()), kw, Arc::from(Vec::new())))
    }

    fn names_of(map: &HashMap<String, usize>, n: usize, keep: bool) -> Arc<[(Box<str>, u16)]> {
        if !keep {
            return Arc::from(Vec::new());
        }
        let mut v: Vec<(Box<str>, u16)> = map.iter().filter(|(_, a)| **a < n && **a <= u16::MAX as usize).map(|(k, &a)| (k.as_str().into(), a as u16)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v.into()
    }

    fn assemble(n: usize, cfg: CompactConfig, ids: IdSource, kw: Vec<(u32, u16)>, kw_names: Arc<[(Box<str>, u16)]>) -> Self {
        CompactRouterG { n, cfg, ids, kw: kw.into(), kw_names, idx: Vec::new(), rows: Rows::new(), updates: 0, decisions: [0; 3] }
    }

    /// A blank (untrained) brain with the same agents, config, keyword table and (shared) vocabulary:
    /// the cheap way to create many independent brains. Private-vocab brains get a COPY of the vocab.
    pub fn fork_blank(&self) -> Self {
        let ids = match &self.ids {
            IdSource::Hash32 => IdSource::Hash32,
            IdSource::Private(v) => IdSource::Private(v.clone()),
            IdSource::Shared(s) => IdSource::Shared(s.clone()),
        };
        CompactRouterG { n: self.n, cfg: self.cfg.clone(), ids, kw: self.kw.clone(), kw_names: self.kw_names.clone(), idx: Vec::new(), rows: Rows::new(), updates: 0, decisions: [0; 3] }
    }

    pub fn n_agents(&self) -> usize {
        self.n
    }
    pub fn config(&self) -> &CompactConfig {
        &self.cfg
    }
    /// number of feature rows
    pub fn n_rows(&self) -> usize {
        self.idx.len()
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    /// cell precision label, e.g. "u16/q16"
    pub fn precision() -> String {
        format!("{}/{}", W::NAME, C::NAME)
    }

    // ---- byte accounting ----------------------------------------------------------------------

    pub fn bytes(&self) -> ByteReport {
        let shared_vocab_bytes = match &self.ids {
            IdSource::Hash32 => 0,
            IdSource::Private(v) => v.bytes_estimate(),
            IdSource::Shared(s) => s.bytes_estimate(),
        };
        ByteReport {
            struct_bytes: std::mem::size_of::<Self>(),
            index_bytes: self.idx.len() * 8,
            row_bytes: self.rows.used_bytes(),
            reserved_bytes: self.idx.capacity() * 8 + self.rows.reserved_bytes(),
            rows: self.idx.len(),
            touched_cells: self.rows.touched_cells(),
            shared_vocab_bytes,
            shared_kw_bytes: self.kw.len() * 8,
        }
    }

    /// Release slack capacity and defragment the rows (call after warm-up / before holding many brains).
    pub fn shrink_to_fit(&mut self) {
        self.idx.shrink_to_fit();
        self.rows.shrink();
    }

    // ---- id resolution ------------------------------------------------------------------------

    fn resolve(&self, task: &Task) -> Resolved {
        match &self.ids {
            IdSource::Hash32 => resolve_in(&|s| Some(hash32(s)), task),
            IdSource::Private(v) => resolve_in(&|s| v.get(s), task),
            IdSource::Shared(sv) => {
                let g = sv.read();
                resolve_in(&|s| g.get(s), task)
            }
        }
    }

    /// intern the unknown strings of `res` and patch their temp ids
    fn intern_unknown(&mut self, res: &mut Resolved) {
        if res.unknown.is_empty() {
            return;
        }
        let real: Vec<u32> = match &mut self.ids {
            IdSource::Hash32 => res.unknown.iter().map(|s| hash32(s)).collect(),
            IdSource::Private(v) => res.unknown.iter().map(|s| v.intern(s)).collect(),
            IdSource::Shared(sv) => res.unknown.iter().map(|s| sv.intern(s)).collect(),
        };
        let lo = TEMP_BASE - (real.len() as u32 - 1);
        let patch = |id: &mut u32| {
            if *id >= lo {
                *id = real[(TEMP_BASE - *id) as usize];
            }
        };
        res.raw.as_mut_slice().iter_mut().for_each(|x| patch(x));
        if let Some(x) = res.kw_type.as_mut() {
            patch(x);
        }
        if let Some(x) = res.type_row.as_mut() {
            patch(x);
        }
        res.unknown.clear();
    }

    #[inline]
    fn find(&self, id: u32) -> Option<usize> {
        self.idx.binary_search_by_key(&id, |e| e.0).ok().map(|i| self.idx[i].1 as usize)
    }

    fn slot_or_create(&mut self, id: u32) -> usize {
        match self.idx.binary_search_by_key(&id, |e| e.0) {
            Ok(i) => self.idx[i].1 as usize,
            Err(pos) => {
                let s = self.rows.push_row();
                self.idx.insert(pos, (id, s as u32));
                s
            }
        }
    }

    // ---- decision -----------------------------------------------------------------------------

    fn keyword_choice(&self, res: &Resolved, cand: &[usize]) -> Option<usize> {
        let mut votes = [0u32; MAX_AGENTS];
        let mut any = false;
        let mut vote = |id: u32| {
            if let Ok(p) = self.kw.binary_search_by_key(&id, |e| e.0) {
                let a = self.kw[p].1 as usize;
                if a < self.n && cand.contains(&a) {
                    votes[a] += 1;
                    any = true;
                }
            }
        };
        for &id in res.raw.as_slice() {
            vote(id);
        }
        if let Some(t) = res.kw_type {
            vote(t);
        }
        if !any {
            return None;
        }
        let mut best: Option<(usize, u32)> = None;
        for (a, &v) in votes.iter().enumerate().take(self.n) {
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

    /// Returns (agent, kind) with kind 0 = keyword, 1 = challenger, 2 = inner learner.
    fn decide(&self, task: &Task, cand: &[usize], rng: &mut Rng) -> (usize, u8) {
        let n = self.n;
        let c = &self.cfg;
        let res = self.resolve(task);
        let mut sv = [0.0f64; MAX_AGENTS];
        let mut fv = [0.0f64; MAX_AGENTS];
        let mut score = [0.0f64; MAX_AGENTS];
        for &id in res.raw.as_slice() {
            if let Some(slot) = self.find(id) {
                self.rows.add_counts(slot, &mut sv[..n], &mut fv[..n]);
            }
        }
        let mut act = IdList::new();
        res.raw.dedupe_into(&mut act);
        if let Some(t) = res.type_row
            && !act.as_slice().contains(&t)
        {
            act.push(t);
        }
        for &id in act.as_slice() {
            match self.find(id) {
                Some(slot) => self.rows.add_scores(slot, n, &mut score[..n]),
                None => {
                    for s in score[..n].iter_mut() {
                        *s += INIT_WEIGHT;
                    }
                }
            }
        }
        let nraw = task.features.len();
        let nf = nraw as f64;
        let pooled = |a: usize| -> (f64, f64) { if nraw == 0 { (0.0, 0.0) } else { (sv[a] / nf, fv[a] / nf) } };
        if let Some(k) = self.keyword_choice(&res, cand) {
            let (s, f) = pooled(k);
            let (_, ucb_k) = beta_bounds(c.kw_a + s, c.kw_b + f, c.z);
            if ucb_k >= c.tau {
                if c.challenge {
                    let mut best = (ucb_k, usize::MAX);
                    // agents without evidence share one lower bound
                    let lcb0 = beta_bounds(1.0, 1.0, c.z).0;
                    for &a in cand {
                        if a == k {
                            continue;
                        }
                        let lcb = if sv[a] == 0.0 && fv[a] == 0.0 {
                            lcb0
                        } else {
                            let (s, f) = pooled(a);
                            beta_bounds(1.0 + s, 1.0 + f, c.z).0
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
        // inner Hebbian: epsilon draw always consumed, then softmax / argmax over candidate scores
        let eps = c.epsilon.clamp(0.0, 1.0);
        if rng.f64() < eps {
            return (cand[rng.below(cand.len())], 2);
        }
        if c.temperature <= 0.0 {
            let _ = rng.f64();
            let mut b = 0;
            for i in 1..cand.len() {
                if score[cand[i]] > score[cand[b]] {
                    b = i;
                }
            }
            return (cand[b], 2);
        }
        let mut logits = [0.0f64; MAX_AGENTS];
        for (i, &a) in cand.iter().enumerate() {
            logits[i] = score[a] / c.temperature;
        }
        (cand[softmax_sample(&logits[..cand.len()], rng)], 2)
    }

    // ---- persistence --------------------------------------------------------------------------

    /// Self-contained serialisable copy (rows defragmented, feature strings stored once).
    pub fn snapshot(&self) -> Snapshot<W, C> {
        let mut rows = self.rows.clone();
        rows.shrink();
        let (names, idx, keywords) = match &self.ids {
            IdSource::Hash32 => {
                let kws = self.kw_names.iter().map(|(k, a)| (k.to_string(), *a)).collect();
                (Vec::new(), self.idx.clone(), kws)
            }
            IdSource::Private(_) | IdSource::Shared(_) => {
                let name_of = |id: u32| -> String {
                    match &self.ids {
                        IdSource::Private(v) => v.name(id).unwrap_or("").to_string(),
                        IdSource::Shared(s) => s.read().name(id).unwrap_or("").to_string(),
                        IdSource::Hash32 => unreachable!(),
                    }
                };
                let mut all: Vec<u32> = self.idx.iter().map(|e| e.0).chain(self.kw.iter().map(|e| e.0)).collect();
                all.sort_unstable();
                all.dedup();
                let names: Vec<String> = all.iter().map(|&id| name_of(id)).collect();
                let idx = self.idx.iter().map(|&(id, slot)| (all.binary_search(&id).unwrap() as u32, slot)).collect();
                let kws = self.kw.iter().map(|&(id, a)| (name_of(id), a)).collect();
                (names, idx, kws)
            }
        };
        Snapshot { version: 1, n_agents: self.n, cfg: self.cfg.clone(), names, idx, keywords, rows, updates: self.updates, decisions: self.decisions }
    }

    /// Rebuild with a private vocabulary (Table mode) or hashing (Hash32 mode).
    pub fn from_snapshot(s: Snapshot<W, C>) -> Result<Self, CompactError> {
        Self::restore(s, None)
    }

    /// Rebuild over a shared vocabulary: snapshot ids are remapped by string.
    pub fn from_snapshot_shared(s: Snapshot<W, C>, vocab: &SharedVocab) -> Result<Self, CompactError> {
        Self::restore(s, Some(vocab))
    }

    fn restore(s: Snapshot<W, C>, shared: Option<&SharedVocab>) -> Result<Self, CompactError> {
        if s.version != 1 {
            return err(format!("unsupported snapshot version {}", s.version));
        }
        Self::check(s.n_agents, &s.cfg)?;
        s.rows.validate(s.n_agents)?;
        if s.idx.iter().any(|e| e.1 as usize >= s.rows.len()) || s.idx.len() != s.rows.len() {
            return err("corrupt snapshot: index/rows mismatch");
        }
        let n = s.n_agents;
        if s.cfg.id_mode == IdMode::Hash32 {
            if shared.is_some() {
                return err("a Hash32 snapshot has no vocabulary to share");
            }
            if s.idx.windows(2).any(|w| w[0].0 >= w[1].0) {
                return err("corrupt snapshot: index not sorted");
            }
            let mut kw: Vec<(u32, u16)> = s.keywords.iter().map(|(k, a)| (hash32(k), *a)).collect();
            kw.sort_by_key(|e| e.0);
            let names: Arc<[(Box<str>, u16)]> = s.keywords.iter().map(|(k, a)| (k.as_str().into(), *a)).collect::<Vec<_>>().into();
            return Ok(CompactRouterG { n, cfg: s.cfg, ids: IdSource::Hash32, kw: kw.into(), kw_names: names, idx: s.idx, rows: s.rows, updates: s.updates, decisions: s.decisions });
        }
        if s.idx.iter().any(|e| e.0 as usize >= s.names.len()) {
            return err("corrupt snapshot: id beyond names");
        }
        let (mut ids, gid): (IdSource, Vec<u32>) = match shared {
            Some(v) => (IdSource::Shared(v.clone()), s.names.iter().map(|x| v.intern(x)).collect()),
            None => {
                let mut v = Vocab::new();
                let g = s.names.iter().map(|x| v.intern(x)).collect();
                (IdSource::Private(v), g)
            }
        };
        let mut idx: Vec<(u32, u32)> = s.idx.iter().map(|&(l, slot)| (gid[l as usize], slot)).collect();
        idx.sort_by_key(|e| e.0);
        if idx.windows(2).any(|w| w[0].0 == w[1].0) {
            return err("corrupt snapshot: duplicate feature");
        }
        let mut kw: Vec<(u32, u16)> = Vec::with_capacity(s.keywords.len());
        for (k, a) in &s.keywords {
            let id = match &mut ids {
                IdSource::Private(v) => v.intern(k),
                IdSource::Shared(sv) => sv.intern(k),
                IdSource::Hash32 => unreachable!(),
            };
            kw.push((id, *a));
        }
        kw.sort_by_key(|e| e.0);
        Ok(CompactRouterG { n, cfg: s.cfg, ids, kw: kw.into(), kw_names: Arc::from(Vec::new()), idx, rows: s.rows, updates: s.updates, decisions: s.decisions })
    }

    pub fn to_json(&self) -> Result<String, CompactError> {
        serde_json::to_string(&self.snapshot()).map_err(|e| CompactError(e.to_string()))
    }
    pub fn from_json(s: &str) -> Result<Self, CompactError> {
        let snap: Snapshot<W, C> = serde_json::from_str(s).map_err(|e| CompactError(e.to_string()))?;
        Self::from_snapshot(snap)
    }
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CompactError> {
        std::fs::write(path, self.to_json()?).map_err(|e| CompactError(e.to_string()))
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self, CompactError> {
        let s = std::fs::read_to_string(path).map_err(|e| CompactError(e.to_string()))?;
        Self::from_json(&s)
    }
}

impl<W: Cell, C: Cell> Router for CompactRouterG<W, C> {
    fn name(&self) -> String {
        format!("compact-guarded(hebbian,{})", Self::precision())
    }

    /// Panics if `candidates` is empty or contains an index >= `n_agents`.
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "CompactRouter::select: no candidates");
        assert!(candidates.iter().all(|&c| c < self.n), "CompactRouter::select: candidate >= n_agents");
        let (a, k) = self.decide(task, candidates, rng);
        self.decisions[k as usize] += 1;
        a
    }

    /// Ignores non-finite rewards and agents >= `n_agents`.
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        if !reward.is_finite() || agent >= self.n {
            return;
        }
        let r = reward.clamp(0.0, 1.0);
        let mut res = self.resolve(task);
        self.intern_unknown(&mut res);
        self.updates += 1;
        let (disc, n) = (self.cfg.discount, self.n);
        // guard counts: once per raw feature occurrence (duplicates count twice, as in `Guarded`)
        for i in 0..res.raw.as_slice().len() {
            let id = res.raw.as_slice()[i];
            let slot = self.slot_or_create(id);
            self.rows.cnt_update(slot, disc, agent, r);
        }
        // Hebbian rule: once per distinct active feature (+ "type:<task_type>")
        let mut act = IdList::new();
        res.raw.dedupe_into(&mut act);
        if let Some(t) = res.type_row
            && !act.as_slice().contains(&t)
        {
            act.push(t);
        }
        for i in 0..act.as_slice().len() {
            let id = act.as_slice()[i];
            let slot = self.slot_or_create(id);
            self.rows.heb_update(slot, &self.cfg, n, agent, r);
        }
    }
}

