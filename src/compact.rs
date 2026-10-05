//! `CompactRouter`: the "smallest viable brain" as a library type (opt-in, new module).
//!
//! It is the SAME decision rule as [`crate::defaults::recommended_router`]
//! (`Guarded<HebbianRouter>`: keyword map in front, Hebbian learner behind, Beta-bound
//! hand-over) with a different STATE ENCODING:
//!
//! * feature strings are interned once into a (optionally shared) [`Vocab`] (or hashed to 32 bits,
//!   [`IdMode::Hash32`], with no table at all); rows are kept SORTED by feature id and found by binary
//!   search (R4/h5: no `(id, slot)` index any more, 4 B/row less);
//! * ONE row per feature holds the Hebbian weights, the reward baseline and the guard counts;
//! * rows are SPARSE: an agent that was never chosen on a feature row has weight = the row's common
//!   "untouched" value and zero counts, so only chosen agents get cells (an exact transform: the
//!   sparse and dense layouts are bit-identical, see the tests); the touched-agent mask is stored in
//!   `ceil(n/8)` bytes per row (R4/h5; R3 used 8 B always), which also lifts the limit to 128 agents;
//! * cells are quantised: weights/baselines as `u16` fixed point on [0,1], guard counts as `u16` in
//!   1/1024 units ([`Fx16`], [`Q16`]). The generic core also instantiates with `f32` and `f64` cells
//!   (`CompactRouterF32`, `CompactRouterF64`); the `f64` instance is decision- and rng-identical to
//!   `defaults::recommended_router` (unit-tested), so every difference of the `u16` instance is
//!   representation error, not logic. 12-bit / 8-bit cells ([`Fx12`], [`Q12`], [`Fx8`]) exist for the
//!   h5 precision study; see `results/h5_board_entry.md` for where equivalence holds and where not.
//!
//! R4/h5 additions (all opt-in, defaults unchanged, `max_rows == vocab_cap == 0` is the R3 behaviour):
//! * `CompactConfig::max_rows`: per-brain row cap with VALUE-based eviction (lowest decayed guard
//!   evidence mass first, i.e. the rarest features) in batches of 12.5%;
//! * [`Vocab::bounded`]/[`SharedVocab::bounded`]: bounded vocabulary, lowest-hit non-keyword strings
//!   are evicted in batches (keyword strings are pinned); a recycled id carries a new 7-bit generation
//!   so rows of other brains that still refer to the evicted string can never match the new one
//!   (they are swept the next time that brain updates);
//! * compact binary state ([`Snapshot::to_bytes`], [`CompactRouterG::to_bytes`]/`from_bytes`: delta+varint
//!   ids, bit-packed cells, exact round trip) and a state-only form for cold brains ([`BrainPool`]).
//!
//! Limits (documented, enforced by the constructor): at most 128 agents; only the plain Hebbian rule
//! (the opt-in `meta`/`weighting`/`ucb`/`agg` options are rejected); the agent count is fixed (updates
//! for an agent index >= n are ignored, candidates must be < n). Nothing here changes a default; the
//! type is not wired into the CLI.
//!
//! The router is not internally synchronised (the `Router` trait takes `&mut self`); the vocabulary
//! may be shared between many brains through [`SharedVocab`] (an `Arc<RwLock<Vocab>>`).

use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicU16, Ordering::Relaxed};
use std::sync::{Arc, RwLock};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::defaults::{recommended_guard, recommended_inner};
use crate::hebbian::{AggKind, HebbianConfig, INIT_BASELINE, INIT_WEIGHT, MetaKind, WeightKind};
use crate::prior::GuardConfig;
use crate::router::{Rng, Router, Task};

/// Maximum number of agents (R3: 64 with a `u64` mask; R4/h5: 128, masks are stored in `ceil(n/8)` bytes).
pub const MAX_AGENTS: usize = 128;
const MAXF: usize = 32;
const TEMP_BASE: u32 = u32::MAX;
const SLOT_MASK: u32 = (1 << 24) - 1;
const GEN_MASK: u8 = 0x7f;
const PINNED: u16 = u16::MAX;

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
// columns and cells

/// Backing store of one cell array (`Vec<T>` for byte-sized cells).
pub trait Column<T: Copy>: Clone + Default + Send + Sync + fmt::Debug + Serialize + DeserializeOwned {
    fn len(&self) -> usize;
    fn get(&self, i: usize) -> T;
    fn set(&mut self, i: usize, v: T);
    fn push(&mut self, v: T);
    fn with_capacity(n: usize) -> Self;
    fn shrink(&mut self);
    /// reserved heap bytes
    fn cap_bytes(&self) -> usize;
    /// bytes needed for `len` elements
    fn bytes_for(len: usize) -> usize;
}

impl<T: Copy + Send + Sync + fmt::Debug + Serialize + DeserializeOwned> Column<T> for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }
    #[inline]
    fn get(&self, i: usize) -> T {
        self[i]
    }
    #[inline]
    fn set(&mut self, i: usize, v: T) {
        self[i] = v;
    }
    #[inline]
    fn push(&mut self, v: T) {
        Vec::push(self, v)
    }
    fn with_capacity(n: usize) -> Self {
        Vec::with_capacity(n)
    }
    fn shrink(&mut self) {
        self.shrink_to_fit()
    }
    fn cap_bytes(&self) -> usize {
        self.capacity() * std::mem::size_of::<T>()
    }
    fn bytes_for(len: usize) -> usize {
        len * std::mem::size_of::<T>()
    }
}

/// 12-bit values packed back to back (1.5 B per element).
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct P12<T> {
    n: usize,
    b: Vec<u8>,
    #[serde(skip)]
    _t: std::marker::PhantomData<T>,
}
impl<T: Cell12> P12<T> {
    #[inline]
    fn raw(&self, i: usize) -> u16 {
        let bit = i * 12;
        let (byte, sh) = (bit >> 3, bit & 7);
        let lo = self.b[byte] as u32;
        let hi = *self.b.as_slice().get(byte + 1).unwrap_or(&0) as u32;
        (((lo | (hi << 8)) >> sh) & 0xfff) as u16
    }
    #[inline]
    fn set_raw(&mut self, i: usize, v: u16) {
        let bit = i * 12;
        let (byte, sh) = (bit >> 3, bit & 7);
        let mask: u32 = 0xfff << sh;
        let cur = (self.b[byte] as u32) | ((*self.b.as_slice().get(byte + 1).unwrap_or(&0) as u32) << 8);
        let new = (cur & !mask) | (((v as u32) & 0xfff) << sh);
        self.b[byte] = new as u8;
        if byte + 1 < self.b.len() {
            self.b[byte + 1] = (new >> 8) as u8;
        }
    }
}
/// A cell whose code fits 12 bits (storable in [`P12`]).
pub trait Cell12: Copy {
    fn code(self) -> u16;
    fn from_code(c: u16) -> Self;
}
impl<T: Cell12 + Default + Send + Sync + fmt::Debug + Serialize + DeserializeOwned> Column<T> for P12<T> {
    fn len(&self) -> usize {
        self.n
    }
    #[inline]
    fn get(&self, i: usize) -> T {
        T::from_code(self.raw(i))
    }
    #[inline]
    fn set(&mut self, i: usize, v: T) {
        self.set_raw(i, v.code())
    }
    #[inline]
    fn push(&mut self, v: T) {
        let need = ((self.n + 1) * 12).div_ceil(8);
        if self.b.len() < need {
            self.b.resize(need, 0);
        }
        self.n += 1;
        let i = self.n - 1;
        self.set_raw(i, v.code());
    }
    fn with_capacity(n: usize) -> Self {
        P12 { n: 0, b: Vec::with_capacity((n * 12).div_ceil(8)), _t: std::marker::PhantomData }
    }
    fn shrink(&mut self) {
        self.b.shrink_to_fit()
    }
    fn cap_bytes(&self) -> usize {
        self.b.capacity()
    }
    fn bytes_for(len: usize) -> usize {
        (len * 12).div_ceil(8)
    }
}

/// Storage precision of one table cell.
pub trait Cell: Copy + Default + Send + Sync + fmt::Debug + 'static + Serialize + DeserializeOwned {
    /// bytes per value when stored in a plain `Vec` (untouched value / baseline of a row)
    const BYTES: usize;
    /// bits per value in the binary state format
    const BITS: u32;
    const NAME: &'static str;
    /// backing store of the per-cell arrays (weights, counts)
    type Col: Column<Self>;
    fn enc(x: f64) -> Self;
    fn dec(self) -> f64;
    fn to_bits(self) -> u64;
    fn from_bits(b: u64) -> Self;
}
impl Cell for f64 {
    const BYTES: usize = 8;
    const BITS: u32 = 64;
    const NAME: &'static str = "f64";
    type Col = Vec<f64>;
    #[inline]
    fn enc(x: f64) -> Self {
        x
    }
    #[inline]
    fn dec(self) -> f64 {
        self
    }
    fn to_bits(self) -> u64 {
        f64::to_bits(self)
    }
    fn from_bits(b: u64) -> Self {
        f64::from_bits(b)
    }
}
impl Cell for f32 {
    const BYTES: usize = 4;
    const BITS: u32 = 32;
    const NAME: &'static str = "f32";
    type Col = Vec<f32>;
    #[inline]
    fn enc(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn dec(self) -> f64 {
        self as f64
    }
    fn to_bits(self) -> u64 {
        f32::to_bits(self) as u64
    }
    fn from_bits(b: u64) -> Self {
        f32::from_bits(b as u32)
    }
}
/// Fixed point on [0,1], 65535 steps (weights and baselines).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fx16(pub u16);
impl Cell for Fx16 {
    const BYTES: usize = 2;
    const BITS: u32 = 16;
    const NAME: &'static str = "u16";
    type Col = Vec<Fx16>;
    #[inline]
    fn enc(x: f64) -> Self {
        Fx16((x.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16)
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 * (1.0 / 65535.0)
    }
    fn to_bits(self) -> u64 {
        self.0 as u64
    }
    fn from_bits(b: u64) -> Self {
        Fx16(b as u16)
    }
}
/// Guard counts in 1/1024 units, saturating at 63.999 (counts of the recommended guard stay below
/// 1/(1-0.98) = 50 because of the 0.98 discount).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Q16(pub u16);
impl Cell for Q16 {
    const BYTES: usize = 2;
    const BITS: u32 = 16;
    const NAME: &'static str = "q16";
    type Col = Vec<Q16>;
    #[inline]
    fn enc(x: f64) -> Self {
        Q16((x * 1024.0 + 0.5) as u16) // non-negative, saturating cast
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 / 1024.0
    }
    fn to_bits(self) -> u64 {
        self.0 as u64
    }
    fn from_bits(b: u64) -> Self {
        Q16(b as u16)
    }
}
/// Fixed point on [0,1], 4095 steps, PACKED 12 bits per cell (study type, see the h5 board entry).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fx12(pub u16);
impl Cell12 for Fx12 {
    fn code(self) -> u16 {
        self.0
    }
    fn from_code(c: u16) -> Self {
        Fx12(c)
    }
}
impl Cell for Fx12 {
    const BYTES: usize = 2;
    const BITS: u32 = 12;
    const NAME: &'static str = "u12";
    type Col = P12<Fx12>;
    #[inline]
    fn enc(x: f64) -> Self {
        Fx12((x.clamp(0.0, 1.0) * 4095.0 + 0.5) as u16)
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 * (1.0 / 4095.0)
    }
    fn to_bits(self) -> u64 {
        self.0 as u64
    }
    fn from_bits(b: u64) -> Self {
        Fx12(b as u16 & 0xfff)
    }
}
/// Guard counts in 1/64 units in 12 bits (max 63.98), PACKED (study type).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Q12(pub u16);
impl Cell12 for Q12 {
    fn code(self) -> u16 {
        self.0
    }
    fn from_code(c: u16) -> Self {
        Q12(c)
    }
}
impl Cell for Q12 {
    const BYTES: usize = 2;
    const BITS: u32 = 12;
    const NAME: &'static str = "q12";
    type Col = P12<Q12>;
    #[inline]
    fn enc(x: f64) -> Self {
        Q12(((x * 64.0 + 0.5) as u16).min(4095))
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 / 64.0
    }
    fn to_bits(self) -> u64 {
        self.0 as u64
    }
    fn from_bits(b: u64) -> Self {
        Q12(b as u16 & 0xfff)
    }
}
/// Fixed point on [0,1], 255 steps (study type; one byte per weight).
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fx8(pub u8);
impl Cell for Fx8 {
    const BYTES: usize = 1;
    const BITS: u32 = 8;
    const NAME: &'static str = "u8";
    type Col = Vec<Fx8>;
    #[inline]
    fn enc(x: f64) -> Self {
        Fx8((x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
    }
    #[inline]
    fn dec(self) -> f64 {
        self.0 as f64 * (1.0 / 255.0)
    }
    fn to_bits(self) -> u64 {
        self.0 as u64
    }
    fn from_bits(b: u64) -> Self {
        Fx8(b as u8)
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

/// Feature string <-> `u32` id table. UNBOUNDED by default (ids are dense, `id == slot`); with
/// [`Vocab::bounded`] at most `cap` strings are live: when full, the lowest-hit non-pinned strings are
/// evicted in a batch (>= cap/8), the surviving hit counters are halved (aging), and a recycled slot
/// gets a new 7-bit generation in the id's top bits so a stale id can never alias the new string.
/// Strings interned as `pinned` (the keyword table) are never evicted; if everything is pinned the
/// table grows past `cap` (soft cap).
#[derive(Default, Debug)]
pub struct Vocab {
    map: FnvMap<Arc<str>, u32>,
    names: Vec<Option<Arc<str>>>,
    gens: Vec<u8>,
    hits: Vec<AtomicU16>,
    free: Vec<u32>,
    cap: usize,
    live: usize,
    evictions: u64,
}
impl Clone for Vocab {
    fn clone(&self) -> Self {
        Vocab {
            map: self.map.clone(),
            names: self.names.clone(),
            gens: self.gens.clone(),
            hits: self.hits.iter().map(|h| AtomicU16::new(h.load(Relaxed))).collect(),
            free: self.free.clone(),
            cap: self.cap,
            live: self.live,
            evictions: self.evictions,
        }
    }
}
impl Vocab {
    pub fn new() -> Self {
        Vocab::default()
    }
    /// Bounded vocabulary (`cap` in 1..=2^24-1 live strings).
    pub fn bounded(cap: usize) -> Self {
        assert!(cap >= 1 && cap < (1 << 24), "vocab cap must be in 1..2^24");
        Vocab { cap, ..Vocab::default() }
    }
    /// 0 = unbounded
    pub fn cap(&self) -> usize {
        self.cap
    }
    /// number of eviction passes so far
    pub fn evictions(&self) -> u64 {
        self.evictions
    }
    pub fn len(&self) -> usize {
        self.live
    }
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
    pub fn get(&self, s: &str) -> Option<u32> {
        self.map.get(s).copied()
    }
    #[inline]
    fn slot(&self, id: u32) -> usize {
        if self.cap == 0 { id as usize } else { (id & SLOT_MASK) as usize }
    }
    pub fn name(&self, id: u32) -> Option<&str> {
        let s = self.slot(id);
        match self.names.get(s) {
            Some(Some(a)) if self.cap == 0 || self.gens[s] == (id >> 24) as u8 => Some(&**a),
            _ => None,
        }
    }
    pub fn is_live(&self, id: u32) -> bool {
        self.name(id).is_some()
    }
    /// count a use of `id` (bounded vocabularies only; saturating, relaxed)
    #[inline]
    pub fn touch(&self, id: u32) {
        if self.cap == 0 {
            return;
        }
        let s = self.slot(id);
        if let Some(h) = self.hits.get(s)
            && self.gens[s] == (id >> 24) as u8
        {
            let v = h.load(Relaxed);
            if v < PINNED - 1 {
                h.store(v + 1, Relaxed);
            }
        }
    }
    pub fn intern(&mut self, s: &str) -> u32 {
        self.intern_batch(&[s], false, &[])[0]
    }
    /// Intern several strings; at most one eviction pass (before any insertion), never evicting
    /// `protect`ed ids nor ids of strings of this batch that are already present. `pinned` strings
    /// can never be evicted (also upgrades already present ones).
    pub fn intern_batch(&mut self, strs: &[&str], pinned: bool, protect: &[u32]) -> Vec<u32> {
        if self.cap > 0 {
            let mut fresh: Vec<&str> = Vec::new();
            let mut prot: Vec<u32> = protect.to_vec();
            for s in strs {
                match self.map.get(*s) {
                    Some(&id) => prot.push(id),
                    None => {
                        if !fresh.contains(s) {
                            fresh.push(s);
                        }
                    }
                }
            }
            if self.live + fresh.len() > self.cap {
                self.evict(fresh.len(), &prot);
            }
        }
        strs.iter().map(|s| self.intern_one(s, pinned)).collect()
    }
    fn intern_one(&mut self, s: &str, pinned: bool) -> u32 {
        if let Some(&id) = self.map.get(s) {
            if pinned && self.cap > 0 {
                let sl = self.slot(id);
                self.hits[sl].store(PINNED, Relaxed);
            }
            return id;
        }
        let a: Arc<str> = Arc::from(s);
        let (slot, gn) = match self.free.pop() {
            Some(sl) => (sl as usize, self.gens[sl as usize]),
            None => {
                self.names.push(None);
                if self.cap > 0 {
                    self.gens.push(0);
                    self.hits.push(AtomicU16::new(0));
                    assert!(self.names.len() < (1 << 24), "vocab slot space exhausted");
                }
                (self.names.len() - 1, 0)
            }
        };
        let id = if self.cap == 0 { slot as u32 } else { ((gn as u32) << 24) | slot as u32 };
        self.names[slot] = Some(a.clone());
        self.map.insert(a, id);
        self.live += 1;
        if self.cap > 0 {
            self.hits[slot].store(if pinned { PINNED } else { 1 }, Relaxed);
        }
        id
    }
    fn evict(&mut self, need: usize, protect: &[u32]) {
        let target = self.cap.saturating_sub(need.max(self.cap / 8));
        let mut prot: Vec<usize> = protect.iter().map(|&i| self.slot(i)).collect();
        prot.sort_unstable();
        let mut cands: Vec<(u16, u32)> = (0..self.names.len())
            .filter(|&s| self.names[s].is_some() && self.hits[s].load(Relaxed) != PINNED && prot.binary_search(&s).is_err())
            .map(|s| (self.hits[s].load(Relaxed), s as u32))
            .collect();
        let k = self.live.saturating_sub(target).min(cands.len());
        if k == 0 {
            return;
        }
        cands.sort_unstable();
        for &(_, s) in &cands[..k] {
            let s = s as usize;
            if let Some(name) = self.names[s].take() {
                self.map.remove(&*name);
            }
            self.gens[s] = self.gens[s].wrapping_add(1) & GEN_MASK;
            self.free.push(s as u32);
            self.live -= 1;
        }
        for s in 0..self.names.len() {
            if self.names[s].is_some() {
                let h = self.hits[s].load(Relaxed);
                if h != PINNED {
                    self.hits[s].store(h >> 1, Relaxed);
                }
            }
        }
        self.evictions += 1;
    }
    /// Estimated heap bytes (hashbrown layout: buckets x (key 16 + value 4, padded to 24) + 1 control
    /// byte each; name vector 16 B per slot; each string once + 16 B Arc header; + 3 B per slot and
    /// free list when bounded). An ESTIMATE; the benchmarks cross-check it against a counting allocator.
    pub fn bytes_estimate(&self) -> usize {
        let cap = self.map.capacity();
        let buckets = if cap == 0 { 0 } else { (cap * 8 / 7).next_power_of_two() };
        let strings: usize = self.names.iter().flatten().map(|s| s.len() + 16).sum();
        buckets * 25 + self.names.capacity() * 16 + strings + self.gens.capacity() + self.hits.capacity() * 2 + self.free.capacity() * 4
    }
}

/// A vocabulary shared between many brains (interning takes the write lock only for NEW strings).
#[derive(Clone, Default, Debug)]
pub struct SharedVocab(Arc<RwLock<Vocab>>);
impl SharedVocab {
    pub fn new() -> Self {
        SharedVocab::default()
    }
    pub fn bounded(cap: usize) -> Self {
        SharedVocab(Arc::new(RwLock::new(Vocab::bounded(cap))))
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
    pub fn cap(&self) -> usize {
        self.read().cap()
    }
    pub fn evictions(&self) -> u64 {
        self.read().evictions()
    }
    pub fn is_live(&self, id: u32) -> bool {
        self.read().is_live(id)
    }
    pub fn name_of(&self, id: u32) -> Option<String> {
        self.read().name(id).map(|s| s.to_string())
    }
    pub fn intern(&self, s: &str) -> u32 {
        if let Some(i) = self.read().get(s) {
            return i;
        }
        self.write().intern(s)
    }
    /// see [`Vocab::intern_batch`]
    pub fn intern_batch(&self, strs: &[&str], pinned: bool, protect: &[u32]) -> Vec<u32> {
        {
            let g = self.read();
            if !pinned || g.cap == 0 {
                let ids: Vec<Option<u32>> = strs.iter().map(|s| g.get(s)).collect();
                if ids.iter().all(|i| i.is_some()) {
                    return ids.into_iter().map(|i| i.unwrap()).collect();
                }
            }
        }
        self.write().intern_batch(strs, pinned, protect)
    }
    pub fn touch_many(&self, ids: &[u32]) {
        let g = self.read();
        for &i in ids {
            g.touch(i);
        }
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

/// Which rows a capped brain evicts first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictPolicy {
    /// lowest decayed guard evidence mass first (the rarest features), ties by id
    #[default]
    Value,
    /// pseudo-random by id hash (ablation: what the cap costs without the value ordering)
    Hash,
}

// ---------------------------------------------------------------------------------------------
// configuration

/// Parameters of the compact brain (flat; `Default` = `defaults::recommended_guard()` +
/// `defaults::recommended_inner()`, no caps).
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
    /// R4/h5: max feature rows per brain (0 = unbounded, the R3 behaviour); eviction in batches of 12.5%
    pub max_rows: usize,
    /// R4/h5: cap of a PRIVATE vocabulary created by `new` (0 = unbounded). Shared vocabularies are
    /// bounded by constructing them with [`SharedVocab::bounded`].
    pub vocab_cap: usize,
    pub evict: EvictPolicy,
    /// R4/h5 cell pruning (0 = off): after a guard update of a row, other agents' cells whose decayed evidence
    /// mass (s+f) is below `prune_mass` AND whose weight is within `prune_dev` of the row's untouched value
    /// are dropped back to "untouched" (a rare (feature, agent) pair keeps neither weight nor counts)
    pub prune_mass: f64,
    pub prune_dev: f64,
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
            max_rows: 0,
            vocab_cap: 0,
            evict: EvictPolicy::Value,
            prune_mass: 0.0,
            prune_dev: 0.005,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// sparse row store (rows sorted by feature id)

/// One entry per feature row (sorted by id): common value of the untouched agents' weights, reward
/// baseline, a touched-agent bit mask (`mb` = ceil(n/8) bytes) and the cells (weight, success count,
/// failure count) of the agents chosen on that row.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
struct Rows<W: Cell, C: Cell> {
    mb: u8,
    ids: Vec<u32>,
    masks: Vec<u8>,
    off: Vec<u32>,
    unt: Vec<W>,
    base: Vec<W>,
    wa: W::Col,
    ca: C::Col,
    garbage: usize,
}

impl<W: Cell, C: Cell> Rows<W, C> {
    fn wcol(n: usize) -> W::Col {
        <W::Col as Column<W>>::with_capacity(n)
    }
    fn ccol(n: usize) -> C::Col {
        <C::Col as Column<C>>::with_capacity(n)
    }
    fn new(n: usize) -> Self {
        Rows { mb: n.div_ceil(8) as u8, ids: Vec::new(), masks: Vec::new(), off: Vec::new(), unt: Vec::new(), base: Vec::new(), wa: Default::default(), ca: Default::default(), garbage: 0 }
    }
    fn len(&self) -> usize {
        self.ids.len()
    }
    #[inline]
    fn find(&self, id: u32) -> Result<usize, usize> {
        self.ids.binary_search(&id)
    }
    #[inline]
    fn mask64(&self, s: usize) -> u64 {
        let mb = self.mb as usize;
        let b = &self.masks[s * mb..s * mb + mb];
        match mb {
            1 => b[0] as u64,
            2 => u16::from_le_bytes([b[0], b[1]]) as u64,
            4 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64,
            8 => u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
            _ => b.iter().enumerate().fold(0u64, |acc, (i, &x)| acc | ((x as u64) << (8 * i))),
        }
    }
    #[inline]
    fn mask(&self, s: usize) -> u128 {
        if self.mb <= 8 {
            return self.mask64(s) as u128;
        }
        let mb = self.mb as usize;
        let mut buf = [0u8; 16];
        buf[..mb].copy_from_slice(&self.masks[s * mb..s * mb + mb]);
        u128::from_le_bytes(buf)
    }
    #[inline]
    fn set_mask(&mut self, s: usize, m: u128) {
        let mb = self.mb as usize;
        self.masks[s * mb..s * mb + mb].copy_from_slice(&m.to_le_bytes()[..mb]);
    }
    fn insert_row(&mut self, pos: usize, id: u32) {
        self.ids.insert(pos, id);
        let mb = self.mb as usize;
        let at = pos * mb;
        self.masks.splice(at..at, std::iter::repeat_n(0u8, mb));
        self.off.insert(pos, self.wa.len() as u32);
        self.unt.insert(pos, W::enc(INIT_WEIGHT));
        self.base.insert(pos, W::enc(INIT_BASELINE));
    }
    #[inline]
    fn rank(mask: u128, a: usize) -> usize {
        (mask & ((1u128 << a) - 1)).count_ones() as usize
    }
    fn compact(&mut self) {
        let live = self.wa.len() - self.garbage;
        let mut wa = Self::wcol(live);
        let mut ca = Self::ccol(2 * live);
        for s in 0..self.ids.len() {
            let len = self.mask(s).count_ones() as usize;
            let o = self.off[s] as usize;
            self.off[s] = wa.len() as u32;
            for i in 0..len {
                wa.push(self.wa.get(o + i));
                ca.push(self.ca.get(2 * (o + i)));
                ca.push(self.ca.get(2 * (o + i) + 1));
            }
        }
        self.wa = wa;
        self.ca = ca;
        self.garbage = 0;
    }
    /// give `agent` a cell in row `s` (the row is re-appended with one more cell; old copy = garbage)
    fn touch_cell(&mut self, s: usize, agent: usize) {
        let m = self.mask(s);
        if m & (1u128 << agent) != 0 {
            return;
        }
        let r = Self::rank(m, agent);
        let len = m.count_ones() as usize;
        let o = self.off[s] as usize;
        let no = self.wa.len();
        for i in 0..r {
            let (w, c0, c1) = (self.wa.get(o + i), self.ca.get(2 * (o + i)), self.ca.get(2 * (o + i) + 1));
            self.wa.push(w);
            self.ca.push(c0);
            self.ca.push(c1);
        }
        self.wa.push(self.unt[s]);
        self.ca.push(C::enc(0.0));
        self.ca.push(C::enc(0.0));
        for i in r..len {
            let (w, c0, c1) = (self.wa.get(o + i), self.ca.get(2 * (o + i)), self.ca.get(2 * (o + i) + 1));
            self.wa.push(w);
            self.ca.push(c0);
            self.ca.push(c1);
        }
        self.garbage += len;
        self.set_mask(s, m | (1u128 << agent));
        self.off[s] = no as u32;
        if self.garbage > 64 && self.garbage * 2 > self.wa.len() {
            self.compact();
        }
    }
    /// Drop back to "untouched" the cells of row `s` (except `keep_agent`'s) whose decayed evidence mass is
    /// below `thr_mass` and whose weight is within `dev` of the row's untouched value. Returns the number dropped.
    fn prune_row(&mut self, s: usize, thr_mass: f64, dev: f64, keep_agent: usize) -> usize {
        let m = self.mask(s);
        let len = m.count_ones() as usize;
        if len <= 1 {
            return 0;
        }
        let o = self.off[s] as usize;
        let u = self.unt[s].dec();
        let mut drop_mask = 0u128;
        let mut k = 0usize;
        let mut mm = m;
        while mm != 0 {
            let a = mm.trailing_zeros() as usize;
            mm &= mm - 1;
            if a != keep_agent {
                let mass = self.ca.get(2 * (o + k)).dec() + self.ca.get(2 * (o + k) + 1).dec();
                if mass < thr_mass && (self.wa.get(o + k).dec() - u).abs() < dev {
                    drop_mask |= 1u128 << a;
                }
            }
            k += 1;
        }
        if drop_mask == 0 {
            return 0;
        }
        let no = self.wa.len();
        let mut kept = 0usize;
        let mut mm = m;
        let mut k = 0usize;
        while mm != 0 {
            let a = mm.trailing_zeros() as usize;
            mm &= mm - 1;
            if drop_mask & (1u128 << a) == 0 {
                let (w, c0, c1) = (self.wa.get(o + k), self.ca.get(2 * (o + k)), self.ca.get(2 * (o + k) + 1));
                self.wa.push(w);
                self.ca.push(c0);
                self.ca.push(c1);
                kept += 1;
            }
            k += 1;
        }
        self.garbage += len;
        self.set_mask(s, m & !drop_mask);
        self.off[s] = no as u32;
        if self.garbage > 64 && self.garbage * 2 > self.wa.len() {
            self.compact();
        }
        len - kept
    }
    #[inline]
    fn add_counts(&self, s: usize, sv: &mut [f64], fv: &mut [f64]) {
        let mut k = self.off[s] as usize;
        if self.mb <= 8 {
            let mut m = self.mask64(s);
            while m != 0 {
                let a = m.trailing_zeros() as usize;
                m &= m - 1;
                sv[a] += self.ca.get(2 * k).dec();
                fv[a] += self.ca.get(2 * k + 1).dec();
                k += 1;
            }
        } else {
            let mut m = self.mask(s);
            while m != 0 {
                let a = m.trailing_zeros() as usize;
                m &= m - 1;
                sv[a] += self.ca.get(2 * k).dec();
                fv[a] += self.ca.get(2 * k + 1).dec();
                k += 1;
            }
        }
    }
    #[inline]
    fn add_scores(&self, s: usize, n: usize, score: &mut [f64]) {
        let u = self.unt[s].dec();
        let mut k = self.off[s] as usize;
        if self.mb <= 8 {
            let m = self.mask64(s);
            for (a, sc) in score.iter_mut().enumerate().take(n) {
                if m & (1u64 << a) != 0 {
                    *sc += self.wa.get(k).dec();
                    k += 1;
                } else {
                    *sc += u;
                }
            }
        } else {
            let m = self.mask(s);
            for (a, sc) in score.iter_mut().enumerate().take(n) {
                if m & (1u128 << a) != 0 {
                    *sc += self.wa.get(k).dec();
                    k += 1;
                } else {
                    *sc += u;
                }
            }
        }
    }
    fn cnt_update(&mut self, s: usize, discount: f64, agent: usize, r: f64) {
        self.touch_cell(s, agent);
        let m = self.mask(s);
        let o = self.off[s] as usize;
        let len = m.count_ones() as usize;
        // untouched cells are exactly 0 and stay 0, so only touched cells are discounted
        if discount < 1.0 {
            for i in 2 * o..2 * (o + len) {
                let x = self.ca.get(i);
                self.ca.set(i, C::enc(x.dec() * discount));
            }
        }
        let k = o + Self::rank(m, agent);
        let sv = self.ca.get(2 * k).dec() + r;
        let fv = self.ca.get(2 * k + 1).dec() + (1.0 - r);
        self.ca.set(2 * k, C::enc(sv));
        self.ca.set(2 * k + 1, C::enc(fv));
    }
    /// Hebbian update of row `s` (same arithmetic and summation order as the dense rule).
    fn heb_update<const N: usize>(&mut self, s: usize, cfg: &CompactConfig, n: usize, agent: usize, r: f64) {
        self.touch_cell(s, agent);
        let m = self.mask(s);
        let o = self.off[s] as usize;
        let len = m.count_ones() as usize;
        let b = self.base[s].dec();
        let delta = cfg.eta * 1.0 * (r - b);
        let keep = 1.0 - cfg.decay;
        let mut u = (self.unt[s].dec() * keep).clamp(0.0, 1.0);
        let mut tv = [0.0f64; N];
        for (i, t) in tv.iter_mut().enumerate().take(len) {
            *t = self.wa.get(o + i).dec() * keep;
        }
        tv[Self::rank(m, agent)] += delta;
        for x in tv[..len].iter_mut() {
            *x = x.clamp(0.0, 1.0);
        }
        let mut sum = 0.0f64;
        let mut i = 0;
        for a in 0..n {
            if m & (1u128 << a) != 0 {
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
        for (i, t) in tv.iter().enumerate().take(len) {
            self.wa.set(o + i, W::enc(*t));
        }
        if len < n {
            self.unt[s] = W::enc(u);
        }
        self.base[s] = W::enc(b + cfg.baseline_alpha * (r - b));
    }
    /// decayed guard evidence mass of row `s` (+ a tiny weight-deviation term so rows that only have
    /// Hebbian state still rank above untouched ones)
    fn value(&self, s: usize) -> f64 {
        let m = self.mask(s);
        let len = m.count_ones() as usize;
        let o = self.off[s] as usize;
        let u = self.unt[s].dec();
        let mut mass = 0.0;
        let mut dev = 0.0;
        for i in 0..len {
            mass += self.ca.get(2 * (o + i)).dec() + self.ca.get(2 * (o + i) + 1).dec();
            dev += (self.wa.get(o + i).dec() - u).abs();
        }
        mass + 1e-3 * dev
    }
    fn used_bytes_split(&self) -> (usize, usize) {
        let ix = self.ids.len() * 4;
        let rest = self.masks.len()
            + self.off.len() * 4
            + self.unt.len() * W::BYTES
            + self.base.len() * W::BYTES
            + <W::Col as Column<W>>::bytes_for(self.wa.len() - self.garbage)
            + <C::Col as Column<C>>::bytes_for(self.ca.len() - 2 * self.garbage);
        (ix, rest)
    }
    fn reserved_bytes(&self) -> usize {
        self.ids.capacity() * 4
            + self.masks.capacity()
            + self.off.capacity() * 4
            + self.unt.capacity() * W::BYTES
            + self.base.capacity() * W::BYTES
            + self.wa.cap_bytes()
            + self.ca.cap_bytes()
    }
    fn touched_cells(&self) -> usize {
        (0..self.len()).map(|s| self.mask(s).count_ones() as usize).sum()
    }
    fn shrink(&mut self) {
        if self.garbage > 0 {
            self.compact();
        }
        self.ids.shrink_to_fit();
        self.masks.shrink_to_fit();
        self.off.shrink_to_fit();
        self.unt.shrink_to_fit();
        self.base.shrink_to_fit();
        self.wa.shrink();
        self.ca.shrink();
    }
    /// New tightly packed rows consisting of source rows `order` (in that order) with ids `ids`
    /// (the caller guarantees `ids` is strictly increasing).
    fn rebuild(&self, order: &[usize], ids: Vec<u32>) -> Self {
        let mb = self.mb as usize;
        let cells: usize = order.iter().map(|&s| self.mask(s).count_ones() as usize).sum();
        let mut out: Rows<W, C> = Rows { mb: self.mb, ids, masks: Vec::with_capacity(order.len() * mb), off: Vec::with_capacity(order.len()), unt: Vec::with_capacity(order.len()), base: Vec::with_capacity(order.len()), wa: Self::wcol(cells), ca: Self::ccol(2 * cells), garbage: 0 };
        for &s in order {
            let len = self.mask(s).count_ones() as usize;
            let o = self.off[s] as usize;
            out.masks.extend_from_slice(&self.masks[s * mb..s * mb + mb]);
            out.off.push(out.wa.len() as u32);
            out.unt.push(self.unt[s]);
            out.base.push(self.base[s]);
            for i in 0..len {
                out.wa.push(self.wa.get(o + i));
                out.ca.push(self.ca.get(2 * (o + i)));
                out.ca.push(self.ca.get(2 * (o + i) + 1));
            }
        }
        out
    }
    fn validate(&self, n: usize) -> Result<(), CompactError> {
        let r = self.ids.len();
        if self.mb as usize != n.div_ceil(8) || self.masks.len() != r * self.mb as usize {
            return err("corrupt snapshot: mask bytes");
        }
        if self.off.len() != r || self.unt.len() != r || self.base.len() != r || self.garbage != 0 {
            return err("corrupt snapshot: row vector lengths");
        }
        if self.ids.windows(2).any(|w| w[0] >= w[1]) {
            return err("corrupt snapshot: ids not strictly increasing");
        }
        let mut expect = 0usize;
        for s in 0..r {
            if self.off[s] as usize != expect {
                return err("corrupt snapshot: row offsets");
            }
            let m = self.mask(s);
            if n < 128 && m >> n != 0 {
                return err("corrupt snapshot: mask bit beyond n_agents");
            }
            expect += m.count_ones() as usize;
        }
        if self.wa.len() != expect || self.ca.len() != 2 * expect {
            return err("corrupt snapshot: cell counts");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// binary primitives (std only)

struct Wr(Vec<u8>);
impl Wr {
    fn varint(&mut self, mut x: u64) {
        while x >= 0x80 {
            self.0.push((x as u8 & 0x7f) | 0x80);
            x >>= 7;
        }
        self.0.push(x as u8);
    }
    fn f64(&mut self, x: f64) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.0.extend_from_slice(b);
    }
    fn str(&mut self, s: &str) {
        self.varint(s.len() as u64);
        self.bytes(s.as_bytes());
    }
}
struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}
impl<'a> Rd<'a> {
    fn u8(&mut self) -> Result<u8, CompactError> {
        let v = *self.b.get(self.p).ok_or_else(|| CompactError("truncated state".into()))?;
        self.p += 1;
        Ok(v)
    }
    fn varint(&mut self) -> Result<u64, CompactError> {
        let (mut x, mut sh) = (0u64, 0u32);
        loop {
            let b = self.u8()?;
            if sh >= 64 {
                return err("bad varint");
            }
            x |= ((b & 0x7f) as u64) << sh;
            if b & 0x80 == 0 {
                return Ok(x);
            }
            sh += 7;
        }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], CompactError> {
        if self.p + n > self.b.len() {
            return err("truncated state");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn f64(&mut self) -> Result<f64, CompactError> {
        let mut a = [0u8; 8];
        a.copy_from_slice(self.take(8)?);
        Ok(f64::from_le_bytes(a))
    }
    fn str(&mut self) -> Result<String, CompactError> {
        let n = self.varint()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| CompactError("bad utf8".into()))
    }
    fn count(&mut self, max: usize) -> Result<usize, CompactError> {
        let n = self.varint()? as usize;
        if n > max {
            return err("corrupt state: count too large");
        }
        Ok(n)
    }
}
#[derive(Default)]
struct BitW {
    buf: Vec<u8>,
    acc: u64,
    n: u32,
}
impl BitW {
    fn put(&mut self, v: u64, bits: u32) {
        if bits > 32 {
            self.put(v & 0xffff_ffff, 32);
            self.put(v >> 32, bits - 32);
            return;
        }
        self.acc |= (v & ((1u64 << bits) - 1)) << self.n;
        self.n += bits;
        while self.n >= 8 {
            self.buf.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.buf.push(self.acc as u8);
        }
        self.buf
    }
}
struct BitR<'a> {
    b: &'a [u8],
    p: usize,
    acc: u64,
    n: u32,
}
impl<'a> BitR<'a> {
    fn get(&mut self, bits: u32) -> Result<u64, CompactError> {
        if bits > 32 {
            let lo = self.get(32)?;
            let hi = self.get(bits - 32)?;
            return Ok(lo | (hi << 32));
        }
        while self.n < bits {
            let byte = *self.b.get(self.p).ok_or_else(|| CompactError("truncated cell stream".into()))?;
            self.p += 1;
            self.acc |= (byte as u64) << self.n;
            self.n += 8;
        }
        let v = self.acc & ((1u64 << bits) - 1);
        self.acc >>= bits;
        self.n -= bits;
        Ok(v)
    }
}

impl<W: Cell, C: Cell> Rows<W, C> {
    /// delta+varint ids, raw masks, bit-packed cells (rows in order, so no offsets are stored).
    fn encode(&self, w: &mut Wr) {
        w.varint(self.len() as u64);
        let mut prev = 0u32;
        for (i, &id) in self.ids.iter().enumerate() {
            w.varint(if i == 0 { id as u64 } else { (id - prev - 1) as u64 });
            prev = id;
        }
        w.bytes(&self.masks);
        let mut bw = BitW::default();
        for s in 0..self.len() {
            bw.put(self.unt[s].to_bits(), W::BITS);
            bw.put(self.base[s].to_bits(), W::BITS);
        }
        for s in 0..self.len() {
            let (len, o) = (self.mask(s).count_ones() as usize, self.off[s] as usize);
            for i in 0..len {
                bw.put(self.wa.get(o + i).to_bits(), W::BITS);
            }
        }
        for s in 0..self.len() {
            let (len, o) = (self.mask(s).count_ones() as usize, self.off[s] as usize);
            for i in 0..2 * len {
                bw.put(self.ca.get(2 * o + i).to_bits(), C::BITS);
            }
        }
        w.bytes(&bw.finish());
    }
    fn decode(r: &mut Rd, n: usize) -> Result<Self, CompactError> {
        let rows = r.count(1 << 26)?;
        let mb = n.div_ceil(8);
        let mut ids = Vec::with_capacity(rows);
        let mut prev = 0u64;
        for i in 0..rows {
            let v = r.varint()?;
            let id = if i == 0 { v } else { prev + 1 + v };
            if id > u32::MAX as u64 {
                return err("corrupt state: id overflow");
            }
            ids.push(id as u32);
            prev = id;
        }
        let masks = r.take(rows * mb)?.to_vec();
        let mut out: Rows<W, C> = Rows { mb: mb as u8, ids, masks, off: Vec::with_capacity(rows), unt: Vec::with_capacity(rows), base: Vec::with_capacity(rows), wa: Default::default(), ca: Default::default(), garbage: 0 };
        let mut cells = 0usize;
        for s in 0..rows {
            out.off.push(cells as u32);
            let m = out.mask(s);
            if n < 128 && m >> n != 0 {
                return err("corrupt state: mask bit beyond n_agents");
            }
            cells += m.count_ones() as usize;
        }
        let rest = &r.b[r.p..];
        let need_bits = rows * 2 * W::BITS as usize + cells * W::BITS as usize + 2 * cells * C::BITS as usize;
        if rest.len() < need_bits.div_ceil(8) {
            return err("truncated cell stream");
        }
        let mut br = BitR { b: rest, p: 0, acc: 0, n: 0 };
        for _ in 0..rows {
            out.unt.push(W::from_bits(br.get(W::BITS)?));
            out.base.push(W::from_bits(br.get(W::BITS)?));
        }
        out.wa = Self::wcol(cells);
        out.ca = Self::ccol(2 * cells);
        for _ in 0..cells {
            out.wa.push(W::from_bits(br.get(W::BITS)?));
        }
        for _ in 0..2 * cells {
            out.ca.push(C::from_bits(br.get(C::BITS)?));
        }
        r.p += need_bits.div_ceil(8);
        Ok(out)
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

fn softmax_sample<const N: usize>(logits: &[f64], rng: &mut Rng) -> usize {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = [0.0f64; N];
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
    /// sorted feature id array, by length (4 B per feature row)
    pub index_bytes: usize,
    /// sparse rows, by length (mask ceil(n/8) + offset 4 + untouched + baseline per row, + cells)
    pub row_bytes: usize,
    /// heap actually reserved (capacity based) for ids + rows; >= index_bytes + row_bytes
    pub reserved_bytes: usize,
    /// feature rows
    pub rows: usize,
    /// touched (feature, agent) cells (each = 1 weight + 2 counts)
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
/// (Table mode) as `names`; the row ids refer to positions in `names` (Hash32 mode: they are the hash ids).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Snapshot<W: Cell, C: Cell> {
    pub version: u32,
    pub n_agents: usize,
    pub cfg: CompactConfig,
    pub names: Vec<String>,
    pub keywords: Vec<(String, u16)>,
    rows: Rows<W, C>,
    pub updates: u64,
    pub decisions: [u64; 3],
}

const SNAP_VERSION: u32 = 2;
const MAGIC: &[u8; 4] = b"CRB2";

impl<W: Cell, C: Cell> Snapshot<W, C> {
    /// Compact binary form (delta+varint ids, bit-packed cells, default config stored as one flag).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = Wr(Vec::new());
        w.bytes(MAGIC);
        w.0.push(W::BITS as u8);
        w.0.push(C::BITS as u8);
        w.varint(self.n_agents as u64);
        let hash = self.cfg.id_mode == IdMode::Hash32;
        let dflt = CompactConfig { id_mode: self.cfg.id_mode, ..CompactConfig::default() };
        w.0.push(hash as u8 | if self.cfg == dflt { 2 } else { 0 });
        if self.cfg != dflt {
            let c = &self.cfg;
            for x in [c.eta, c.epsilon, c.temperature, c.decay, c.baseline_alpha, c.row_cap, c.tau, c.z, c.kw_a, c.kw_b, c.discount] {
                w.f64(x);
            }
            w.0.push(c.challenge as u8);
            w.varint(c.max_rows as u64);
            w.varint(c.vocab_cap as u64);
            w.0.push(c.evict as u8);
            w.f64(c.prune_mass);
            w.f64(c.prune_dev);
        }
        w.varint(self.updates);
        for d in self.decisions {
            w.varint(d);
        }
        w.varint(self.names.len() as u64);
        for s in &self.names {
            w.str(s);
        }
        w.varint(self.keywords.len() as u64);
        if hash {
            for (k, a) in &self.keywords {
                w.str(k);
                w.varint(*a as u64);
            }
        } else {
            // keyword names are in `names` (in id order, as are the keywords): store (name index delta, agent)
            let pos: HashMap<&str, usize> = self.names.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
            let mut prev = 0usize;
            for (k, a) in &self.keywords {
                let i = pos.get(k.as_str()).copied().unwrap_or(0);
                w.varint(i.wrapping_sub(prev) as u64);
                prev = i;
                w.varint(*a as u64);
            }
        }
        self.rows.encode(&mut w);
        w.0
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self, CompactError> {
        let mut r = Rd { b, p: 0 };
        if r.take(4)? != MAGIC {
            return err("bad magic");
        }
        if r.u8()? as u32 != W::BITS || r.u8()? as u32 != C::BITS {
            return err("cell precision mismatch");
        }
        let n = r.varint()? as usize;
        if n == 0 || n > MAX_AGENTS {
            return err("bad n_agents");
        }
        let flags = r.u8()?;
        let id_mode = if flags & 1 != 0 { IdMode::Hash32 } else { IdMode::Table };
        let cfg = if flags & 2 != 0 {
            CompactConfig { id_mode, ..CompactConfig::default() }
        } else {
            let mut f = [0.0f64; 11];
            for x in f.iter_mut() {
                *x = r.f64()?;
            }
            let challenge = r.u8()? != 0;
            let max_rows = r.varint()? as usize;
            let vocab_cap = r.varint()? as usize;
            let evict = if r.u8()? == 0 { EvictPolicy::Value } else { EvictPolicy::Hash };
            let (prune_mass, prune_dev) = (r.f64()?, r.f64()?);
            CompactConfig { eta: f[0], epsilon: f[1], temperature: f[2], decay: f[3], baseline_alpha: f[4], row_cap: f[5], tau: f[6], z: f[7], kw_a: f[8], kw_b: f[9], discount: f[10], challenge, id_mode, max_rows, vocab_cap, evict, prune_mass, prune_dev }
        };
        let updates = r.varint()?;
        let decisions = [r.varint()?, r.varint()?, r.varint()?];
        let nn = r.count(1 << 26)?;
        let mut names = Vec::with_capacity(nn);
        for _ in 0..nn {
            names.push(r.str()?);
        }
        let nk = r.count(1 << 26)?;
        let mut keywords = Vec::with_capacity(nk);
        if flags & 1 != 0 {
            for _ in 0..nk {
                let k = r.str()?;
                keywords.push((k, r.varint()? as u16));
            }
        } else {
            let mut prev = 0usize;
            for _ in 0..nk {
                let i = prev.wrapping_add(r.varint()? as usize);
                prev = i;
                let a = r.varint()? as u16;
                keywords.push((names.get(i).ok_or_else(|| CompactError("bad keyword index".into()))?.clone(), a));
            }
        }
        let rows = Rows::<W, C>::decode(&mut r, n)?;
        Ok(Snapshot { version: SNAP_VERSION, n_agents: n, cfg, names, keywords, rows, updates, decisions })
    }
}

pub struct CompactRouterG<W: Cell, C: Cell> {
    n: usize,
    cfg: CompactConfig,
    ids: IdSource,
    /// the vocabulary is bounded (touch counts / sweeps needed)
    vbound: bool,
    /// keyword id -> agent, sorted by id (shared between forks)
    kw: Arc<[(u32, u16)]>,
    /// keyword strings (kept only in Hash32 mode, where ids cannot be inverted)
    kw_names: Arc<[(Box<str>, u16)]>,
    rows: Rows<W, C>,
    updates: u64,
    /// vocabulary eviction passes already swept by this brain
    seen_evict: u64,
    row_evictions: u64,
    cells_pruned: u64,
    /// decisions taken: (keyword, challenger, inner learner)
    pub decisions: [u64; 3],
}

/// The recommended compact brain: sparse rows, u16 weights, Q16 guard counts.
pub type CompactRouter = CompactRouterG<Fx16, Q16>;
/// f32 cells (decision-identical to f64 in the g3/f3 measurements; half the bytes).
pub type CompactRouterF32 = CompactRouterG<f32, f32>;
/// f64 cells: decision- and rng-identical to `defaults::recommended_router` (sparse layout only).
pub type CompactRouterF64 = CompactRouterG<f64, f64>;

fn build_kw(map: &HashMap<String, usize>, n: usize, intern: impl FnOnce(&[&str]) -> Vec<u32>) -> Vec<(u32, u16)> {
    let ents: Vec<(&String, usize)> = map.iter().filter(|(_, a)| **a < n.max(1) && **a <= u16::MAX as usize).map(|(k, &a)| (k, a)).collect();
    let strs: Vec<&str> = ents.iter().map(|(k, _)| k.as_str()).collect();
    let ids = intern(&strs);
    let mut v: Vec<(u32, u16)> = ids.into_iter().zip(ents.iter()).map(|(id, (_, a))| (id, *a as u16)).collect();
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
        if cfg.vocab_cap >= (1 << 24) {
            return err("vocab_cap must be < 2^24");
        }
        Ok(())
    }

    /// New brain with a PRIVATE vocabulary (or no vocabulary in Hash32 mode).
    pub fn new(n_agents: usize, keywords: &HashMap<String, usize>, cfg: CompactConfig) -> Result<Self, CompactError> {
        Self::check(n_agents, &cfg)?;
        match cfg.id_mode {
            IdMode::Hash32 => {
                let kw = build_kw(keywords, n_agents, |ss| ss.iter().map(|s| hash32(s)).collect());
                Ok(Self::assemble(n_agents, cfg, IdSource::Hash32, false, kw, Self::names_of(keywords, n_agents, true)))
            }
            IdMode::Table => {
                let mut v = if cfg.vocab_cap > 0 { Vocab::bounded(cfg.vocab_cap) } else { Vocab::new() };
                let kw = build_kw(keywords, n_agents, |ss| v.intern_batch(ss, true, &[]));
                let vb = cfg.vocab_cap > 0;
                Ok(Self::assemble(n_agents, cfg, IdSource::Private(v), vb, kw, Arc::from(Vec::new())))
            }
        }
    }

    /// New brain over a SHARED vocabulary (Table mode only; `cfg.id_mode` is forced to Table).
    pub fn with_shared_vocab(n_agents: usize, keywords: &HashMap<String, usize>, mut cfg: CompactConfig, vocab: &SharedVocab) -> Result<Self, CompactError> {
        cfg.id_mode = IdMode::Table;
        Self::check(n_agents, &cfg)?;
        let kw = build_kw(keywords, n_agents, |ss| vocab.intern_batch(ss, true, &[]));
        let vb = vocab.cap() > 0;
        Ok(Self::assemble(n_agents, cfg, IdSource::Shared(vocab.clone()), vb, kw, Arc::from(Vec::new())))
    }

    fn names_of(map: &HashMap<String, usize>, n: usize, keep: bool) -> Arc<[(Box<str>, u16)]> {
        if !keep {
            return Arc::from(Vec::new());
        }
        let mut v: Vec<(Box<str>, u16)> = map.iter().filter(|(_, a)| **a < n && **a <= u16::MAX as usize).map(|(k, &a)| (k.as_str().into(), a as u16)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v.into()
    }

    fn assemble(n: usize, cfg: CompactConfig, ids: IdSource, vbound: bool, kw: Vec<(u32, u16)>, kw_names: Arc<[(Box<str>, u16)]>) -> Self {
        CompactRouterG { n, cfg, ids, vbound, kw: kw.into(), kw_names, rows: Rows::new(n), updates: 0, seen_evict: 0, row_evictions: 0, cells_pruned: 0, decisions: [0; 3] }
    }

    /// A blank (untrained) brain with the same agents, config, keyword table and (shared) vocabulary:
    /// the cheap way to create many independent brains. Private-vocab brains get a COPY of the vocab.
    pub fn fork_blank(&self) -> Self {
        let ids = match &self.ids {
            IdSource::Hash32 => IdSource::Hash32,
            IdSource::Private(v) => IdSource::Private(v.clone()),
            IdSource::Shared(s) => IdSource::Shared(s.clone()),
        };
        CompactRouterG { n: self.n, cfg: self.cfg.clone(), ids, vbound: self.vbound, kw: self.kw.clone(), kw_names: self.kw_names.clone(), rows: Rows::new(self.n), updates: 0, seen_evict: 0, row_evictions: 0, cells_pruned: 0, decisions: [0; 3] }
    }

    pub fn n_agents(&self) -> usize {
        self.n
    }
    pub fn config(&self) -> &CompactConfig {
        &self.cfg
    }
    /// number of feature rows
    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }
    pub fn updates(&self) -> u64 {
        self.updates
    }
    /// rows evicted by the `max_rows` cap so far
    pub fn row_evictions(&self) -> u64 {
        self.row_evictions
    }
    /// cells dropped by `prune_mass` so far
    pub fn cells_pruned(&self) -> u64 {
        self.cells_pruned
    }
    /// strings in this brain's PRIVATE vocabulary (0 for shared / hashed)
    pub fn private_vocab_len(&self) -> usize {
        match &self.ids {
            IdSource::Private(v) => v.len(),
            _ => 0,
        }
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
        let (index_bytes, row_bytes) = self.rows.used_bytes_split();
        ByteReport {
            struct_bytes: std::mem::size_of::<Self>(),
            index_bytes,
            row_bytes,
            reserved_bytes: self.rows.reserved_bytes(),
            rows: self.rows.len(),
            touched_cells: self.rows.touched_cells(),
            shared_vocab_bytes,
            shared_kw_bytes: self.kw.len() * 8,
        }
    }

    /// Release slack capacity and defragment the rows (call after warm-up / before holding many brains).
    pub fn shrink_to_fit(&mut self) {
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

    fn known_ids(res: &Resolved) -> Vec<u32> {
        let lo = if res.unknown.is_empty() { u32::MAX } else { TEMP_BASE - (res.unknown.len() as u32 - 1) };
        let mut v: Vec<u32> = res.raw.as_slice().iter().copied().filter(|&i| i < lo).collect();
        for x in [res.kw_type, res.type_row].into_iter().flatten() {
            if x < lo {
                v.push(x);
            }
        }
        v
    }

    /// intern the unknown strings of `res` and patch their temp ids
    fn intern_unknown(&mut self, res: &mut Resolved) {
        if res.unknown.is_empty() {
            return;
        }
        let protect = if self.vbound { Self::known_ids(res) } else { Vec::new() };
        let strs: Vec<&str> = res.unknown.iter().map(|s| s.as_str()).collect();
        let real: Vec<u32> = match &mut self.ids {
            IdSource::Hash32 => strs.iter().map(|s| hash32(s)).collect(),
            IdSource::Private(v) => v.intern_batch(&strs, false, &protect),
            IdSource::Shared(sv) => sv.intern_batch(&strs, false, &protect),
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

    /// count the use of the known ids of this task (bounded vocabularies only)
    fn touch(&self, res: &Resolved) {
        if !self.vbound {
            return;
        }
        let ids = Self::known_ids(res);
        match &self.ids {
            IdSource::Private(v) => ids.iter().for_each(|&i| v.touch(i)),
            IdSource::Shared(s) => s.touch_many(&ids),
            IdSource::Hash32 => {}
        }
    }

    /// drop rows whose string the (bounded) vocabulary has evicted since this brain last looked
    fn sync_vocab(&mut self) {
        if !self.vbound {
            return;
        }
        let ev = match &self.ids {
            IdSource::Private(v) => v.evictions(),
            IdSource::Shared(s) => s.evictions(),
            IdSource::Hash32 => return,
        };
        if ev != self.seen_evict {
            self.seen_evict = ev;
            self.sweep_stale();
        }
    }

    fn live_rows(&self) -> Vec<usize> {
        let live = |id: u32| match &self.ids {
            IdSource::Private(v) => v.is_live(id),
            IdSource::Shared(s) => s.is_live(id),
            IdSource::Hash32 => true,
        };
        match &self.ids {
            IdSource::Shared(s) => {
                let g = s.read();
                (0..self.rows.len()).filter(|&i| g.is_live(self.rows.ids[i])).collect()
            }
            _ => (0..self.rows.len()).filter(|&i| live(self.rows.ids[i])).collect(),
        }
    }

    fn sweep_stale(&mut self) {
        let keep = self.live_rows();
        if keep.len() < self.rows.len() {
            let ids: Vec<u32> = keep.iter().map(|&i| self.rows.ids[i]).collect();
            self.rows = self.rows.rebuild(&keep, ids);
        }
    }

    /// evict the lowest-value rows down to 87.5% of the cap
    fn evict_rows(&mut self) {
        let cap = self.cfg.max_rows;
        let r = self.rows.len();
        if cap == 0 || r <= cap {
            return;
        }
        let target = (cap - cap / 8).max(1);
        let mut order: Vec<(u64, u32, usize)> = (0..r)
            .map(|i| {
                let id = self.rows.ids[i];
                let key = match self.cfg.evict {
                    EvictPolicy::Value => self.rows.value(i).to_bits(),
                    EvictPolicy::Hash => (id as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11,
                };
                (key, id, i)
            })
            .collect();
        order.sort_unstable();
        let mut keep: Vec<usize> = order[r - target..].iter().map(|e| e.2).collect();
        keep.sort_unstable();
        let ids: Vec<u32> = keep.iter().map(|&i| self.rows.ids[i]).collect();
        self.row_evictions += (r - target) as u64;
        self.rows = self.rows.rebuild(&keep, ids);
    }

    #[inline]
    fn find(&self, id: u32) -> Option<usize> {
        self.rows.find(id).ok()
    }

    fn slot_or_create(&mut self, id: u32) -> usize {
        match self.rows.find(id) {
            Ok(i) => i,
            Err(pos) => {
                self.rows.insert_row(pos, id);
                pos
            }
        }
    }

    // ---- decision -----------------------------------------------------------------------------

    fn keyword_choice<const N: usize>(&self, res: &Resolved, cand: &[usize]) -> Option<usize> {
        let mut votes = [0u32; N];
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
        if self.n <= 64 { self.decide_n::<64>(task, cand, rng) } else { self.decide_n::<128>(task, cand, rng) }
    }

    fn decide_n<const N: usize>(&self, task: &Task, cand: &[usize], rng: &mut Rng) -> (usize, u8) {
        let n = self.n;
        let c = &self.cfg;
        let res = self.resolve(task);
        let mut sv = [0.0f64; N];
        let mut fv = [0.0f64; N];
        let mut score = [0.0f64; N];
        // one pass over the raw ids: guard counts for every occurrence, Hebbian scores for the first
        // occurrence of each distinct id (same accumulation order as the dense reference), one row lookup each
        {
            let raw = res.raw.as_slice();
            for (i, &id) in raw.iter().enumerate() {
                let slot = self.find(id);
                if let Some(s) = slot {
                    self.rows.add_counts(s, &mut sv[..n], &mut fv[..n]);
                }
                if !raw[..i].contains(&id) {
                    match slot {
                        Some(s) => self.rows.add_scores(s, n, &mut score[..n]),
                        None => {
                            for s in score[..n].iter_mut() {
                                *s += INIT_WEIGHT;
                            }
                        }
                    }
                }
            }
            if let Some(t) = res.type_row
                && !raw.contains(&t)
            {
                match self.find(t) {
                    Some(s) => self.rows.add_scores(s, n, &mut score[..n]),
                    None => {
                        for s in score[..n].iter_mut() {
                            *s += INIT_WEIGHT;
                        }
                    }
                }
            }
        }
        let nraw = task.features.len();
        let nf = nraw as f64;
        let pooled = |a: usize| -> (f64, f64) { if nraw == 0 { (0.0, 0.0) } else { (sv[a] / nf, fv[a] / nf) } };
        if let Some(k) = self.keyword_choice::<N>(&res, cand) {
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
        let mut logits = [0.0f64; N];
        for (i, &a) in cand.iter().enumerate() {
            logits[i] = score[a] / c.temperature;
        }
        (cand[softmax_sample::<N>(&logits[..cand.len()], rng)], 2)
    }

    fn update_n<const N: usize>(&mut self, task: &Task, agent: usize, r: f64) {
        let mut res = self.resolve(task);
        self.touch(&res);
        self.intern_unknown(&mut res);
        self.sync_vocab();
        self.updates += 1;
        let (disc, n) = (self.cfg.discount, self.n);
        // guard counts: once per raw feature occurrence (duplicates count twice, as in `Guarded`)
        for i in 0..res.raw.as_slice().len() {
            let id = res.raw.as_slice()[i];
            let slot = self.slot_or_create(id);
            self.rows.cnt_update(slot, disc, agent, r);
            if self.cfg.prune_mass > 0.0 {
                self.cells_pruned += self.rows.prune_row(slot, self.cfg.prune_mass, self.cfg.prune_dev, agent) as u64;
            }
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
            self.rows.heb_update::<N>(slot, &self.cfg, n, agent, r);
        }
        if self.cfg.max_rows > 0 && self.rows.len() > self.cfg.max_rows {
            self.evict_rows();
        }
    }

    // ---- persistence --------------------------------------------------------------------------

    /// Self-contained serialisable copy (rows defragmented, feature strings stored once).
    pub fn snapshot(&self) -> Snapshot<W, C> {
        let all_rows: Vec<usize>;
        let (names, rows, keywords) = match &self.ids {
            IdSource::Hash32 => {
                all_rows = (0..self.rows.len()).collect();
                let kws = self.kw_names.iter().map(|(k, a)| (k.to_string(), *a)).collect();
                (Vec::new(), self.rows.rebuild(&all_rows, self.rows.ids.clone()), kws)
            }
            IdSource::Private(_) | IdSource::Shared(_) => {
                let live = self.live_rows();
                let name_of = |id: u32| -> String {
                    match &self.ids {
                        IdSource::Private(v) => v.name(id).unwrap_or("").to_string(),
                        IdSource::Shared(s) => s.name_of(id).unwrap_or_default(),
                        IdSource::Hash32 => unreachable!(),
                    }
                };
                let mut all: Vec<u32> = live.iter().map(|&i| self.rows.ids[i]).chain(self.kw.iter().map(|e| e.0)).collect();
                all.sort_unstable();
                all.dedup();
                let names: Vec<String> = all.iter().map(|&id| name_of(id)).collect();
                let new_ids: Vec<u32> = live.iter().map(|&i| all.binary_search(&self.rows.ids[i]).unwrap() as u32).collect();
                let kws = self.kw.iter().map(|&(id, a)| (name_of(id), a)).collect();
                (names, self.rows.rebuild(&live, new_ids), kws)
            }
        };
        Snapshot { version: SNAP_VERSION, n_agents: self.n, cfg: self.cfg.clone(), names, keywords, rows, updates: self.updates, decisions: self.decisions }
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
        if s.version != SNAP_VERSION {
            return err(format!("unsupported snapshot version {}", s.version));
        }
        Self::check(s.n_agents, &s.cfg)?;
        s.rows.validate(s.n_agents)?;
        let n = s.n_agents;
        if s.cfg.id_mode == IdMode::Hash32 {
            if shared.is_some() {
                return err("a Hash32 snapshot has no vocabulary to share");
            }
            let mut kw: Vec<(u32, u16)> = s.keywords.iter().map(|(k, a)| (hash32(k), *a)).collect();
            kw.sort_by_key(|e| e.0);
            let names: Arc<[(Box<str>, u16)]> = s.keywords.iter().map(|(k, a)| (k.as_str().into(), *a)).collect::<Vec<_>>().into();
            return Ok(CompactRouterG { n, cfg: s.cfg, ids: IdSource::Hash32, vbound: false, kw: kw.into(), kw_names: names, rows: s.rows, updates: s.updates, seen_evict: 0, row_evictions: 0, cells_pruned: 0, decisions: s.decisions });
        }
        if s.rows.ids.iter().any(|&e| e as usize >= s.names.len()) {
            return err("corrupt snapshot: id beyond names");
        }
        let kw_strs: Vec<&str> = s.keywords.iter().map(|(k, _)| k.as_str()).collect();
        let name_strs: Vec<&str> = s.names.iter().map(|x| x.as_str()).collect();
        let (ids, vbound, kwids, gid): (IdSource, bool, Vec<u32>, Vec<u32>) = match shared {
            Some(v) => {
                let kwids = v.intern_batch(&kw_strs, true, &[]);
                let gid = v.intern_batch(&name_strs, false, &kwids);
                (IdSource::Shared(v.clone()), v.cap() > 0, kwids, gid)
            }
            None => {
                let mut v = if s.cfg.vocab_cap > 0 { Vocab::bounded(s.cfg.vocab_cap) } else { Vocab::new() };
                let kwids = v.intern_batch(&kw_strs, true, &[]);
                let gid = v.intern_batch(&name_strs, false, &kwids);
                let vb = v.cap() > 0;
                (IdSource::Private(v), vb, kwids, gid)
            }
        };
        let mut order: Vec<(u32, usize)> = s.rows.ids.iter().enumerate().map(|(i, &l)| (gid[l as usize], i)).collect();
        order.sort_unstable();
        if order.windows(2).any(|w| w[0].0 == w[1].0) {
            return err("corrupt snapshot: duplicate feature");
        }
        let src: Vec<usize> = order.iter().map(|e| e.1).collect();
        let nids: Vec<u32> = order.iter().map(|e| e.0).collect();
        let rows = s.rows.rebuild(&src, nids);
        let mut kw: Vec<(u32, u16)> = kwids.into_iter().zip(s.keywords.iter()).map(|(id, (_, a))| (id, *a)).collect();
        kw.sort_by_key(|e| e.0);
        let mut me = CompactRouterG { n, cfg: s.cfg, ids, vbound, kw: kw.into(), kw_names: Arc::from(Vec::new()), rows, updates: s.updates, seen_evict: 0, row_evictions: 0, cells_pruned: 0, decisions: s.decisions };
        me.sweep_stale();
        Ok(me)
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

    /// Self-contained COMPACT BINARY state (exact round trip; see [`Snapshot::to_bytes`]).
    pub fn to_bytes(&self) -> Vec<u8> {
        self.snapshot().to_bytes()
    }
    pub fn from_bytes(b: &[u8]) -> Result<Self, CompactError> {
        Self::from_snapshot(Snapshot::from_bytes(b)?)
    }
    pub fn from_bytes_shared(b: &[u8], vocab: &SharedVocab) -> Result<Self, CompactError> {
        Self::from_snapshot_shared(Snapshot::from_bytes(b)?, vocab)
    }
    pub fn save_bin(&self, path: impl AsRef<Path>) -> Result<(), CompactError> {
        std::fs::write(path, self.to_bytes()).map_err(|e| CompactError(e.to_string()))
    }
    pub fn load_bin(path: impl AsRef<Path>) -> Result<Self, CompactError> {
        let b = std::fs::read(path).map_err(|e| CompactError(e.to_string()))?;
        Self::from_bytes(&b)
    }

    /// STATE-ONLY binary form for cold brains (rows + counters, vocabulary ids as they are): valid
    /// only against a brain forked from the same prototype (same vocabulary, keywords, config).
    pub fn state_bytes(&self) -> Vec<u8> {
        let mut w = Wr(Vec::new());
        w.varint(self.updates);
        for d in self.decisions {
            w.varint(d);
        }
        w.varint(self.row_evictions);
        self.rows.encode(&mut w);
        w.0
    }
    /// Restore a [`CompactRouterG::state_bytes`] blob into a blank fork of `proto`; rows whose
    /// vocabulary string has been evicted in the meantime are dropped.
    pub fn from_state_bytes(proto: &Self, b: &[u8]) -> Result<Self, CompactError> {
        let mut me = proto.fork_blank();
        let mut r = Rd { b, p: 0 };
        me.updates = r.varint()?;
        me.decisions = [r.varint()?, r.varint()?, r.varint()?];
        me.row_evictions = r.varint()?;
        me.rows = Rows::decode(&mut r, me.n)?;
        me.rows.validate(me.n)?;
        me.seen_evict = match &me.ids {
            IdSource::Private(v) => v.evictions(),
            IdSource::Shared(s) => s.evictions(),
            IdSource::Hash32 => 0,
        };
        if me.vbound {
            me.sweep_stale();
        }
        Ok(me)
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
        if self.n <= 64 { self.update_n::<64>(task, agent, r) } else { self.update_n::<128>(task, agent, r) }
    }
}

// ---------------------------------------------------------------------------------------------
// cold-brain pool

/// Counters of a [`BrainPool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct PoolStats {
    pub spills: u64,
    pub reloads: u64,
    pub created: u64,
    pub hits: u64,
    pub spilled_bytes_total: u64,
}

/// Many independent brains (one per user / team / tenant key) over one prototype (shared vocabulary,
/// keyword table, config). At most `max_hot` brains stay live; the least recently used one is
/// serialised to its compact STATE-ONLY binary form and reloaded exactly (bit-identical decisions)
/// when its key is used again.
pub struct BrainPool<W: Cell, C: Cell> {
    proto: CompactRouterG<W, C>,
    max_hot: usize,
    hot: HashMap<u64, (CompactRouterG<W, C>, u64)>,
    cold: HashMap<u64, Box<[u8]>>,
    tick: u64,
    pub stats: PoolStats,
}

impl<W: Cell, C: Cell> BrainPool<W, C> {
    /// `proto` should be built with [`CompactRouterG::with_shared_vocab`] (a private vocabulary would be
    /// copied into every brain).
    pub fn new(proto: CompactRouterG<W, C>, max_hot: usize) -> Self {
        BrainPool { proto, max_hot: max_hot.max(1), hot: HashMap::new(), cold: HashMap::new(), tick: 0, stats: PoolStats::default() }
    }
    pub fn len(&self) -> usize {
        self.hot.len() + self.cold.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn hot_len(&self) -> usize {
        self.hot.len()
    }
    pub fn cold_len(&self) -> usize {
        self.cold.len()
    }
    /// serialized bytes of the cold brains
    pub fn cold_bytes(&self) -> usize {
        self.cold.values().map(|b| b.len()).sum()
    }
    /// owned bytes of the hot brains (exact, length based; excludes the shared vocabulary)
    pub fn hot_bytes(&self) -> usize {
        self.hot.values().map(|(b, _)| b.bytes().owned()).sum()
    }
    /// Brain of `key` (created blank on first use, reloaded if it was spilled).
    pub fn get(&mut self, key: u64) -> Result<&mut CompactRouterG<W, C>, CompactError> {
        self.tick += 1;
        if !self.hot.contains_key(&key) {
            while self.hot.len() >= self.max_hot {
                self.spill_lru();
            }
            let brain = match self.cold.remove(&key) {
                Some(b) => match CompactRouterG::from_state_bytes(&self.proto, &b) {
                    Ok(x) => {
                        self.stats.reloads += 1;
                        x
                    }
                    Err(e) => {
                        self.cold.insert(key, b); // keep the blob, report the error
                        return Err(e);
                    }
                },
                None => {
                    self.stats.created += 1;
                    self.proto.fork_blank()
                }
            };
            self.hot.insert(key, (brain, self.tick));
        } else {
            self.stats.hits += 1;
        }
        let e = self.hot.get_mut(&key).expect("just inserted");
        e.1 = self.tick;
        Ok(&mut e.0)
    }
    fn spill_lru(&mut self) {
        let Some((&k, _)) = self.hot.iter().min_by_key(|e| (e.1.1, *e.0)) else { return };
        self.spill(k);
    }
    fn spill(&mut self, key: u64) {
        if let Some((brain, _)) = self.hot.remove(&key) {
            let b = brain.state_bytes().into_boxed_slice();
            self.stats.spills += 1;
            self.stats.spilled_bytes_total += b.len() as u64;
            self.cold.insert(key, b);
        }
    }
    /// serialise every hot brain
    pub fn spill_all(&mut self) {
        let keys: Vec<u64> = self.hot.keys().copied().collect();
        for k in keys {
            self.spill(k);
        }
    }
    /// forget a key entirely
    pub fn remove(&mut self, key: u64) -> bool {
        self.hot.remove(&key).is_some() | self.cold.remove(&key).is_some()
    }
}

// ---------------------------------------------------------------------------------------------
// tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::defaults::recommended_router;

    fn kw_map() -> HashMap<String, usize> {
        let mut m = HashMap::new();
        for t in 0..6 {
            for j in 0..3 {
                m.insert(format!("t{t}k{j}"), (t * 5 + 1) % 8); // some traps
            }
        }
        m
    }

    fn stream(n_tasks: usize, seed: u64) -> Vec<Task> {
        let mut r = Rng::new(seed);
        (0..n_tasks)
            .map(|id| {
                let t = r.below(6);
                let mut f = vec![format!("t{t}k{}", r.below(3)), format!("t{t}k{}", r.below(3))]; // may repeat
                if r.bernoulli(0.3) {
                    f.push(format!("noise{}", r.below(40))); // unknown-to-keyword features
                }
                if r.bernoulli(0.1) {
                    f.push(format!("fresh{id}")); // never seen again
                }
                let task_type = if r.bernoulli(0.25) { Some(format!("ty{}", r.below(4))) } else { None };
                Task { id: id as u64, task_type, features: f }
            })
            .collect()
    }

    fn truth(task: &Task, a: usize) -> f64 {
        // deterministic hidden quality: agent (hash of first feature) is good
        let g = (task.features[0].len() * 7 + task.features[0].bytes().map(|b| b as usize).sum::<usize>()) % 8;
        if a == g { 0.9 } else { 0.25 }
    }

    /// run a router over a stream, returning (choices, final rng draw)
    fn run<R: Router>(r: &mut R, tasks: &[Task], seed: u64, n: usize) -> (Vec<usize>, u64) {
        let mut rng = Rng::new(seed);
        let mut out = Vec::new();
        let all: Vec<usize> = (0..n).collect();
        for (i, t) in tasks.iter().enumerate() {
            // vary the candidate set sometimes
            let cands: Vec<usize> = if i % 7 == 3 { all.iter().copied().filter(|a| a % 2 == 0).collect() } else { all.clone() };
            let a = r.select(t, &cands, &mut rng);
            assert!(cands.contains(&a));
            let rew = if rng.f64() < truth(t, a) { 1.0 } else { 0.0 };
            r.update(t, a, rew);
            out.push(a);
        }
        (out, rng.next_u64())
    }

    #[test]
    fn f64_instance_is_decision_and_rng_identical_to_recommended_router() {
        let tasks = stream(1500, 11);
        for n in [3usize, 8, 40] {
            let map: HashMap<String, usize> = kw_map().into_iter().map(|(k, a)| (k, a % n)).collect();
            let mut reference = recommended_router(map.clone(), n);
            let mut compact = CompactRouterF64::new(n, &map, CompactConfig::default()).unwrap();
            let a = run(&mut reference, &tasks, 5, n);
            let b = run(&mut compact, &tasks, 5, n);
            assert_eq!(a, b, "n={n}: decisions/rng diverged");
            assert_eq!(reference.decisions, compact.decisions, "decision-kind counts");
        }
    }

    #[test]
    fn f64_instance_is_identical_beyond_64_agents() {
        // the R4/h5 limit lift: 65, 100 and 128 agents (u128 masks, 128-wide scratch arrays)
        let tasks = stream(600, 12);
        for n in [65usize, 100, 128] {
            let map: HashMap<String, usize> = kw_map().into_iter().map(|(k, a)| (k, a % n)).collect();
            let mut reference = recommended_router(map.clone(), n);
            let mut compact = CompactRouterF64::new(n, &map, CompactConfig::default()).unwrap();
            let a = run(&mut reference, &tasks, 6, n);
            let b = run(&mut compact, &tasks, 6, n);
            assert_eq!(a, b, "n={n}: decisions/rng diverged");
        }
        assert!(CompactRouter::new(129, &HashMap::new(), CompactConfig::default()).is_err());
    }

    #[test]
    fn hash32_mode_matches_table_mode_without_collisions() {
        let tasks = stream(600, 3);
        let n = 8;
        let map = kw_map();
        let mut t = CompactRouter::new(n, &map, CompactConfig::default()).unwrap();
        let mut h = CompactRouter::new(n, &map, CompactConfig { id_mode: IdMode::Hash32, ..CompactConfig::default() }).unwrap();
        assert_eq!(run(&mut t, &tasks, 9, n), run(&mut h, &tasks, 9, n));
    }

    #[test]
    fn u16_instance_is_deterministic_and_close_to_f64() {
        let tasks = stream(1500, 21);
        let n = 8;
        let map = kw_map();
        let a = run(&mut CompactRouter::new(n, &map, CompactConfig::default()).unwrap(), &tasks, 4, n);
        let b = run(&mut CompactRouter::new(n, &map, CompactConfig::default()).unwrap(), &tasks, 4, n);
        assert_eq!(a, b, "same seed, same decisions");
        let c = run(&mut CompactRouter::new(n, &map, CompactConfig::default()).unwrap(), &tasks, 5, n);
        assert_ne!(a.0, c.0, "different rng seed should change some sampled choices");
        let f = run(&mut recommended_router(map, n), &tasks, 4, n);
        let agree = a.0.iter().zip(&f.0).filter(|(x, y)| x == y).count() as f64 / a.0.len() as f64;
        eprintln!("u16/q16 vs f64 decision agreement: {agree:.3}");
        let q = |c: &[usize]| c.iter().zip(&tasks).map(|(&a, t)| truth(t, a)).sum::<f64>() / c.len() as f64;
        assert!((q(&a.0) - q(&f.0)).abs() < 0.08, "quality gap too large: {} vs {}", q(&a.0), q(&f.0));
    }

    #[test]
    fn serde_roundtrip_is_exact_and_resumes_identically() {
        let tasks = stream(800, 31);
        let n = 8;
        let map = kw_map();
        let mut a = CompactRouter::new(n, &map, CompactConfig::default()).unwrap();
        let first = run(&mut a, &tasks[..400], 2, n);
        let json = a.to_json().unwrap();
        let mut b = CompactRouter::from_json(&json).unwrap();
        assert_eq!(json, b.to_json().unwrap(), "save -> load -> save is byte-identical");
        assert_eq!(a.n_rows(), b.n_rows());
        assert_eq!(a.bytes().owned(), b.bytes().owned());
        // continue both with the same rng: identical
        let mut ra = Rng::new(77);
        let mut rb = Rng::new(77);
        let all: Vec<usize> = (0..n).collect();
        for t in &tasks[400..] {
            let x = a.select(t, &all, &mut ra);
            let y = b.select(t, &all, &mut rb);
            assert_eq!(x, y);
            let rew = if (x + t.id as usize) % 3 == 0 { 1.0 } else { 0.0 };
            a.update(t, x, rew);
            b.update(t, y, rew);
        }
        let _ = first;
        // file round trip
        let p = std::env::temp_dir().join(format!("compact_rt_{}.json", std::process::id()));
        a.save(&p).unwrap();
        let c = CompactRouter::load(&p).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!(c.to_json().unwrap(), a.to_json().unwrap());
        // Hash32 + shared-vocab restores
        let mut h = CompactRouter::new(n, &map, CompactConfig { id_mode: IdMode::Hash32, ..CompactConfig::default() }).unwrap();
        run(&mut h, &tasks[..200], 1, n);
        let hj = h.to_json().unwrap();
        assert_eq!(CompactRouter::from_json(&hj).unwrap().to_json().unwrap(), hj);
        let sv = SharedVocab::new();
        sv.intern("zzz-already-there");
        let s = CompactRouter::from_snapshot_shared(a.snapshot(), &sv).unwrap();
        assert_eq!(s.n_rows(), a.n_rows());
        let mut rs = Rng::new(5);
        let mut s2 = s;
        let mut ra = Rng::new(5);
        for t in &tasks[..50] {
            assert_eq!(s2.select(t, &all, &mut rs), a.select(t, &all, &mut ra), "shared-vocab restore decides identically");
        }
    }

    #[test]
    fn corrupt_snapshot_is_rejected() {
        let n = 4;
        let mut a = CompactRouter::new(n, &kw_map().into_iter().map(|(k, v)| (k, v % n)).collect(), CompactConfig::default()).unwrap();
        run(&mut a, &stream(100, 1), 1, n);
        let mut v: serde_json::Value = serde_json::from_str(&a.to_json().unwrap()).unwrap();
        v["rows"]["off"][1] = serde_json::json!(9999);
        assert!(CompactRouter::from_json(&v.to_string()).is_err());
        assert!(CompactRouter::from_json("{not json").is_err());
    }

    #[test]
    fn trait_conformance() {
        let n = 5;
        let map: HashMap<String, usize> = [("sql".to_string(), 2usize)].into_iter().collect();
        let mut boxed: Box<dyn Router> = Box::new(CompactRouter::new(n, &map, CompactConfig::default()).unwrap());
        assert!(boxed.name().contains("compact"));
        let mut rng = Rng::new(1);
        let t = Task { id: 0, task_type: None, features: vec!["sql".into()] };
        // keyword first (prior trusted), like the recommended router
        assert_eq!(boxed.select(&t, &[0, 1, 2, 3, 4], &mut rng), 2);
        // keyword agent not eligible -> some other eligible agent
        let a = boxed.select(&t, &[0, 1, 3], &mut rng);
        assert!([0, 1, 3].contains(&a));
        // empty feature list, only a task type, is fine
        let e = Task { id: 1, task_type: Some("x".into()), features: vec![] };
        let a = boxed.select(&e, &[1, 4], &mut rng);
        assert!(a == 1 || a == 4);
        boxed.update(&e, a, 1.0);
        // ignored updates must not panic: non-finite reward, unknown agent
        boxed.update(&t, 2, f64::NAN);
        boxed.update(&t, 99, 1.0);
        boxed.update(&t, 2, 7.5); // clamped to 1
        // keyword is abandoned after enough proven failures, a proven-better challenger takes over
        for _ in 0..200 {
            boxed.update(&t, 2, 0.0);
            boxed.update(&t, 4, 1.0);
        }
        let mut hits = 0;
        for _ in 0..50 {
            if boxed.select(&t, &[0, 1, 2, 3, 4], &mut rng) == 4 {
                hits += 1;
            }
        }
        assert!(hits >= 40, "learned the better agent: {hits}/50");
    }

    #[test]
    #[should_panic(expected = "no candidates")]
    fn empty_candidates_panic() {
        let mut r = CompactRouter::new(3, &HashMap::new(), CompactConfig::default()).unwrap();
        r.select(&Task::default(), &[], &mut Rng::new(1));
    }

    #[test]
    fn constructor_limits() {
        let m = HashMap::new();
        assert!(CompactRouter::new(0, &m, CompactConfig::default()).is_err());
        assert!(CompactRouter::new(129, &m, CompactConfig::default()).is_err());
        assert!(CompactRouter::new(128, &m, CompactConfig::default()).is_ok());
        let fancy = HebbianConfig { ucb_c: 0.5, ..recommended_inner() };
        assert!(CompactConfig::from_configs(&recommended_guard(), &fancy).is_err());
        let fancy = HebbianConfig { meta: MetaKind::Dbd, ..recommended_inner() };
        assert!(CompactConfig::from_configs(&recommended_guard(), &fancy).is_err());
        // the default config IS the recommended pair
        let d = CompactConfig::default();
        assert_eq!((d.eta, d.temperature, d.tau, d.kw_a, d.discount), (0.01, 0.002, 0.7, 4.0, 0.98));
        assert_eq!((d.max_rows, d.vocab_cap), (0, 0), "caps are opt-in");
    }

    #[test]
    fn shared_vocab_forks_are_independent_and_share_one_table() {
        let n = 8;
        let map = kw_map();
        let sv = SharedVocab::new();
        let proto = CompactRouter::with_shared_vocab(n, &map, CompactConfig::default(), &sv).unwrap();
        let mut a = proto.fork_blank();
        let mut b = proto.fork_blank();
        let tasks = stream(300, 8);
        run(&mut a, &tasks, 1, n);
        let before = b.n_rows();
        assert_eq!(before, 0, "fork must not see the sibling's rows");
        run(&mut b, &tasks[..50], 1, n);
        assert!(a.n_rows() > b.n_rows());
        assert!(sv.len() >= a.n_rows(), "both brains interned into the one table");
        let (ra, rb) = (a.bytes(), b.bytes());
        assert_eq!(ra.shared_vocab_bytes, rb.shared_vocab_bytes);
        assert!(ra.owned() > rb.owned());
    }

    #[test]
    fn byte_accounting_is_exact_for_the_u16_layout() {
        let n = 8;
        let map = kw_map();
        let mut r = CompactRouter::new(n, &map, CompactConfig::default()).unwrap();
        assert_eq!(r.bytes().owned(), std::mem::size_of::<CompactRouter>());
        run(&mut r, &stream(500, 5), 1, n);
        r.shrink_to_fit();
        let b = r.bytes();
        // R4/h5 layout: per row: id 4 + mask ceil(n/8) + off 4 + untouched 2 + baseline 2 (R3 was 8+8+4+2+2);
        // per touched cell: weight 2 + 2 counts x 2
        assert_eq!(b.index_bytes + b.row_bytes, b.rows * (4 + 1 + 4 + 2 + 2) + b.touched_cells * 6);
        assert!(b.reserved_bytes >= b.index_bytes + b.row_bytes);
        assert!(b.reserved_bytes <= b.index_bytes + b.row_bytes + 64, "after shrink_to_fit the slack is gone (got {} vs {})", b.reserved_bytes, b.index_bytes + b.row_bytes);
        assert!(b.touched_cells <= b.rows * n);
        assert!(b.shared_vocab_bytes > 0 && b.private_total() > b.owned());
    }

    #[test]
    fn many_features_beyond_the_stack_buffer() {
        let n = 4;
        let mut r = CompactRouterF64::new(n, &HashMap::new(), CompactConfig::default()).unwrap();
        let mut f64r = recommended_router(HashMap::new(), n);
        let t = Task { id: 0, task_type: None, features: (0..100).map(|i| format!("f{}", i % 70)).collect() };
        let (mut ra, mut rb) = (Rng::new(3), Rng::new(3));
        let all = [0usize, 1, 2, 3];
        for _ in 0..30 {
            let x = r.select(&t, &all, &mut ra);
            let y = f64r.select(&t, &all, &mut rb);
            assert_eq!(x, y, "100 features (heap spill) decide like the reference");
            r.update(&t, x, 1.0);
            f64r.update(&t, y, 1.0);
        }
        assert_eq!(r.n_rows(), 70);
    }

    // ---------------- R4/h5 additions ----------------

    fn bin_roundtrip_exact<W: Cell, C: Cell>() {
        let tasks = stream(700, 41);
        let n = 8;
        let map = kw_map();
        for hash in [false, true] {
            let cfg = CompactConfig { id_mode: if hash { IdMode::Hash32 } else { IdMode::Table }, ..CompactConfig::default() };
            let mut a = CompactRouterG::<W, C>::new(n, &map, cfg).unwrap();
            run(&mut a, &tasks[..350], 2, n);
            let bin = a.to_bytes();
            let mut b = CompactRouterG::<W, C>::from_bytes(&bin).unwrap();
            assert_eq!(bin, b.to_bytes(), "bin -> load -> bin identical ({})", CompactRouterG::<W, C>::precision());
            assert_eq!(a.to_json().unwrap(), b.to_json().unwrap(), "JSON of the reloaded brain is identical");
            assert_eq!(a.decisions, b.decisions);
            let (mut ra, mut rb) = (Rng::new(9), Rng::new(9));
            let all: Vec<usize> = (0..n).collect();
            for t in &tasks[350..] {
                let x = a.select(t, &all, &mut ra);
                assert_eq!(x, b.select(t, &all, &mut rb));
                let rew = if (x + t.id as usize) % 3 == 0 { 1.0 } else { 0.0 };
                a.update(t, x, rew);
                b.update(t, x, rew);
            }
            assert_eq!(a.to_bytes(), b.to_bytes());
        }
    }

    #[test]
    fn binary_roundtrip_is_exact_for_every_cell_type() {
        bin_roundtrip_exact::<Fx16, Q16>();
        bin_roundtrip_exact::<f64, f64>();
        bin_roundtrip_exact::<f32, f32>();
        bin_roundtrip_exact::<Fx12, Q12>();
        bin_roundtrip_exact::<Fx8, Q16>();
    }

    #[test]
    fn binary_is_smaller_than_json_and_rejects_garbage() {
        let n = 8;
        let mut a = CompactRouter::new(n, &kw_map(), CompactConfig::default()).unwrap();
        run(&mut a, &stream(800, 5), 1, n);
        let (bin, json) = (a.to_bytes(), a.to_json().unwrap());
        assert!(bin.len() * 2 < json.len(), "binary {} vs json {}", bin.len(), json.len());
        assert!(CompactRouter::from_bytes(&bin[..bin.len() / 2]).is_err(), "truncated");
        assert!(CompactRouter::from_bytes(b"nope").is_err(), "magic");
        let mut bad = bin.clone();
        bad[0] ^= 1;
        assert!(CompactRouter::from_bytes(&bad).is_err());
        assert!(CompactRouterF64::from_bytes(&bin).is_err(), "precision mismatch is detected");
        // non-default config survives
        let cfg = CompactConfig { max_rows: 40, eta: 0.02, ..CompactConfig::default() };
        let mut c = CompactRouter::new(n, &kw_map(), cfg.clone()).unwrap();
        run(&mut c, &stream(300, 6), 1, n);
        assert_eq!(CompactRouter::from_bytes(&c.to_bytes()).unwrap().config(), &cfg);
    }

    #[test]
    fn packed_12_bit_column_matches_a_vec() {
        let mut p: P12<Q12> = Default::default();
        let mut v: Vec<u16> = Vec::new();
        let mut x = 12345u32;
        for i in 0..1001 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let c = ((x >> 8) & 0xfff) as u16;
            p.push(Q12(c));
            v.push(c);
            if i % 7 == 0 {
                let j = (x as usize >> 3) % p.len();
                let d = ((x >> 4) & 0xfff) as u16;
                p.set(j, Q12(d));
                v[j] = d;
            }
        }
        assert_eq!(p.len(), 1001);
        assert!(v.iter().enumerate().all(|(i, &c)| p.get(i).0 == c));
        assert_eq!(<P12<Q12> as Column<Q12>>::bytes_for(1001), (1001 * 12usize).div_ceil(8));
    }

    #[test]
    fn row_cap_evicts_the_rare_keeps_the_frequent_and_is_deterministic() {
        let n = 8;
        let map = kw_map();
        let cfg = CompactConfig { max_rows: 40, ..CompactConfig::default() };
        let mut a = CompactRouter::new(n, &map, cfg.clone()).unwrap();
        let mut b = CompactRouter::new(n, &map, cfg).unwrap();
        let tasks = stream(1500, 7); // 18 keyword features + 40 noise + 1500*0.1 one-shot fresh features
        let ra = run(&mut a, &tasks, 3, n);
        let rb = run(&mut b, &tasks, 3, n);
        assert_eq!(ra, rb);
        assert!(a.n_rows() <= 40, "cap respected: {}", a.n_rows());
        assert!(a.row_evictions() > 100, "one-shot features were evicted: {}", a.row_evictions());
        // the 18 frequent keyword-ish features should all still have rows
        let sv = match &a.ids {
            IdSource::Private(v) => v.clone(),
            _ => unreachable!(),
        };
        let kept = (0..6).flat_map(|t| (0..3).map(move |j| format!("t{t}k{j}"))).filter(|s| sv.get(s).map(|id| a.find(id).is_some()).unwrap_or(false)).count();
        assert!(kept >= 16, "frequent features survive value-based eviction: {kept}/18");
        // snapshot of a capped brain round-trips
        let a2 = CompactRouter::from_bytes(&a.to_bytes()).unwrap();
        assert_eq!(a2.n_rows(), a.n_rows());
    }

    #[test]
    fn bounded_vocab_stays_bounded_pins_keywords_and_never_aliases() {
        let n = 8;
        let map = kw_map();
        let sv = SharedVocab::bounded(60);
        let proto = CompactRouter::with_shared_vocab(n, &map, CompactConfig::default(), &sv).unwrap();
        let mut a = proto.fork_blank();
        let mut b = proto.fork_blank();
        let tasks = stream(1500, 9);
        run(&mut a, &tasks, 3, n);
        run(&mut b, &tasks[500..], 4, n);
        assert!(sv.len() <= 60, "vocab bounded: {}", sv.len());
        assert!(sv.evictions() > 0);
        for t in 0..6 {
            for j in 0..3 {
                assert!(sv.is_live(sv.intern(&format!("t{t}k{j}"))), "keyword pinned");
            }
        }
        // a recycled slot must carry a new generation: ids of evicted strings are dead
        let id1 = sv.intern("brand-new-string-A");
        for i in 0..400 {
            sv.intern(&format!("flood{i}"));
        }
        assert!(!sv.is_live(id1) || sv.name_of(id1).as_deref() == Some("brand-new-string-A"));
        let id2 = sv.intern("brand-new-string-B");
        assert_eq!(sv.name_of(id2).as_deref(), Some("brand-new-string-B"));
        if (id1 & SLOT_MASK) == (id2 & SLOT_MASK) {
            assert_ne!(id1, id2, "same slot, different generation");
        }
        // after the flood the brain's old rows are stale; the next update sweeps them
        let t = &tasks[0];
        a.update(t, 1, 1.0);
        for i in 0..a.n_rows() {
            assert!(sv.is_live(a.rows.ids[i]), "no stale row survives a sweep");
        }
        // snapshot/restore of a brain on a bounded shared vocab is exact on the live rows
        let a2 = CompactRouter::from_bytes_shared(&a.to_bytes(), &sv).unwrap();
        assert_eq!(a2.n_rows(), a.n_rows());
        // private bounded vocab through the config
        let mut p = CompactRouter::new(n, &map, CompactConfig { vocab_cap: 50, ..CompactConfig::default() }).unwrap();
        run(&mut p, &tasks, 3, n);
        assert!(p.private_vocab_len() <= 50);
        assert!(p.n_rows() <= 50);
    }

    #[test]
    fn unbounded_default_is_unchanged_by_the_cap_machinery() {
        // cap 0 must behave exactly like the plain brain: same rows, same bytes, same decisions
        let n = 8;
        let map = kw_map();
        let tasks = stream(900, 13);
        let mut a = CompactRouter::new(n, &map, CompactConfig::default()).unwrap();
        let mut b = CompactRouter::new(n, &map, CompactConfig { max_rows: 100_000, vocab_cap: 100_000, ..CompactConfig::default() }).unwrap();
        assert_eq!(run(&mut a, &tasks, 3, n), run(&mut b, &tasks, 3, n), "caps that never bind do not change decisions");
        assert_eq!(a.n_rows(), b.n_rows());
    }

    #[test]
    fn cell_pruning_drops_rare_pairs_is_off_by_default_and_roundtrips() {
        let n = 8;
        let map = kw_map();
        let tasks = stream(1500, 19);
        let mut base = CompactRouter::new(n, &map, CompactConfig::default()).unwrap();
        let mut pr = CompactRouter::new(n, &map, CompactConfig { prune_mass: 0.6, ..CompactConfig::default() }).unwrap();
        let mut pr2 = CompactRouter::new(n, &map, CompactConfig { prune_mass: 0.6, ..CompactConfig::default() }).unwrap();
        run(&mut base, &tasks, 3, n);
        let a = run(&mut pr, &tasks, 3, n);
        assert_eq!(a, run(&mut pr2, &tasks, 3, n), "deterministic");
        assert_eq!(base.cells_pruned(), 0, "off by default");
        assert!(pr.cells_pruned() > 0, "something was pruned");
        base.shrink_to_fit();
        pr.shrink_to_fit();
        assert!(pr.bytes().touched_cells <= base.bytes().touched_cells);
        assert_eq!(CompactRouter::from_bytes(&pr.to_bytes()).unwrap().to_bytes(), pr.to_bytes());
        assert_eq!(CompactRouter::from_json(&pr.to_json().unwrap()).unwrap().config().prune_mass, 0.6);
    }

    #[test]
    fn pool_spill_and_reload_is_bit_exact() {
        let n = 8;
        let map = kw_map();
        let sv = SharedVocab::new();
        let proto = CompactRouter::with_shared_vocab(n, &map, CompactConfig::default(), &sv).unwrap();
        let mut pool = BrainPool::new(proto.fork_blank(), 2);
        let mut hotall: Vec<CompactRouter> = (0..7).map(|_| proto.fork_blank()).collect();
        let tasks = stream(700, 17);
        let all: Vec<usize> = (0..n).collect();
        let mut rngs: Vec<Rng> = (0..7).map(|k| Rng::new(100 + k)).collect();
        let mut rngs2 = rngs.clone();
        let mut order = Rng::new(5);
        for t in &tasks {
            let k = order.below(7);
            let x = pool.get(k as u64).unwrap().select(t, &all, &mut rngs[k]);
            let y = hotall[k].select(t, &all, &mut rngs2[k]);
            assert_eq!(x, y, "spilled/reloaded brain {k} decides like the always-hot one");
            let rew = if (x + t.id as usize) % 3 == 0 { 1.0 } else { 0.0 };
            pool.get(k as u64).unwrap().update(t, x, rew);
            hotall[k].update(t, y, rew);
        }
        assert!(pool.stats.spills > 20 && pool.stats.reloads > 20, "{:?}", pool.stats);
        assert_eq!(pool.len(), 7);
        pool.spill_all();
        assert_eq!(pool.hot_len(), 0);
        for k in 0..7u64 {
            let b = pool.get(k).unwrap();
            assert_eq!(b.to_bytes(), hotall[k as usize].to_bytes(), "state of brain {k} identical after the whole churn");
        }
    }

}
