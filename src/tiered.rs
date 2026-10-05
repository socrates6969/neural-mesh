//! Tiered agent memory: working buffer -> long-term index -> compressed tier.
//!
//! ```text
//!   add ──► short-term ring (exact scan, bounded)
//!              │ evicted (oldest first)
//!              ▼
//!          long-term "hot" store   (a MemoryIndex: exact flat, or HNSW)
//!              │ consolidate(): lowest-value items beyond `hot_cap`
//!              ▼
//!          cold tier (turbovec low-bit codes; no f32 copy, approximate scores)
//!              │ consolidate(): over the byte budget -> forget lowest value
//!              ▼
//!            gone
//! ```
//!
//! * `search` merges the three tiers by score. Short-term and hot are exact
//!   (for the flat hot store); only cold-tier scores are quantised estimates.
//!   That is the "exact rerank only for the hot set" rule: a cold item can be
//!   found but never exactly re-scored, because its f32 vector is gone.
//! * Every item carries usefulness stats: retrieval count, rewards from
//!   [`TieredMemory::record_outcome`], and a time-decayed `value`
//!   (`value = Σ reward · 2^(-Δtick / halflife)`; new items start at
//!   `init_value`). Ticks are logical: one per `add`.
//! * Forgetting removes the lowest-value items (ties: oldest first) until the
//!   byte estimate is under `byte_budget`. With no outcomes recorded every
//!   value is just `init_value` decayed, which is exactly FIFO, so "forgetting
//!   without feedback" is the recency-only baseline.
//!
//! `MemoryIndex` has no delete, so the removable stores below are local:
//! [`FlatRm`] (swap-remove), [`HnswRm`] (tombstones + rebuild from a retained
//! f32 shadow copy, which is counted in `bytes()`), and [`ColdTv`]
//! (`turbovec::IdMapIndex::remove`).
//!
//! Byte accounting is an estimate in the style of `memory.rs`: payload +
//! main side tables, tight allocation (len-based, not capacity-based), plus
//! `ENTRY_OVERHEAD` bytes per tracked item for the stats table.

use std::collections::{HashMap, HashSet, VecDeque};

use turbovec::IdMapIndex;

use crate::memory::{dot, normalize_in_place, normalized, HnswIndex, HnswParams, MemoryIndex};

/// Estimated bytes per tracked item in the stats table (key + entry + map slack).
pub const ENTRY_OVERHEAD: usize = 88;

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

#[inline]
fn push_topk(top: &mut Vec<(u64, f32)>, k: usize, id: u64, s: f32) {
    if k == 0 {
        return;
    }
    if top.len() == k && s <= top[k - 1].1 {
        return;
    }
    let pos = top.partition_point(|&(_, x)| x >= s);
    top.insert(pos, (id, s));
    top.truncate(k);
}

// ---------------------------------------------------------------------------
// Removable stores
// ---------------------------------------------------------------------------

/// A [`MemoryIndex`] that can also delete and hand back stored vectors.
pub trait RemovableIndex: MemoryIndex + Send + Sync {
    /// Delete `id`; `true` if it was present.
    fn remove(&mut self, id: u64) -> bool;
    /// Normalised stored vector of `id`, if the store keeps f32 copies.
    fn vector(&self, id: u64) -> Option<Vec<f32>>;
    /// Reclaim space after a batch of removals.
    fn compact(&mut self) {}
}

/// Exact flat store with O(1) swap-remove.
pub struct FlatRm {
    dim: usize,
    data: Vec<f32>,
    ids: Vec<u64>,
    pos: HashMap<u64, usize>,
}

impl FlatRm {
    pub fn new(dim: usize) -> Self {
        Self { dim, data: Vec::new(), ids: Vec::new(), pos: HashMap::new() }
    }
}

impl MemoryIndex for FlatRm {
    fn add(&mut self, id: u64, v: &[f32]) {
        assert_eq!(v.len(), self.dim, "FlatRm: dim mismatch");
        if self.pos.contains_key(&id) {
            self.remove(id);
        }
        let start = self.data.len();
        self.data.extend_from_slice(v);
        normalize_in_place(&mut self.data[start..]);
        self.pos.insert(id, self.ids.len());
        self.ids.push(id);
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(q.len(), self.dim, "FlatRm: query dim mismatch");
        let qn = normalized(q);
        let mut top = Vec::with_capacity(k + 1);
        for (row, id) in self.data.chunks_exact(self.dim).zip(&self.ids) {
            push_topk(&mut top, k, *id, dot(&qn, row));
        }
        top
    }
    fn len(&self) -> usize {
        self.ids.len()
    }
    fn bytes(&self) -> usize {
        self.ids.len() * (self.dim * 4 + 8 + 24)
    }
    fn name(&self) -> String {
        "flat-rm".into()
    }
}

impl RemovableIndex for FlatRm {
    fn remove(&mut self, id: u64) -> bool {
        let Some(idx) = self.pos.remove(&id) else { return false };
        let last = self.ids.len() - 1;
        if idx != last {
            let (head, tail) = self.data.split_at_mut(last * self.dim);
            head[idx * self.dim..(idx + 1) * self.dim].copy_from_slice(&tail[..self.dim]);
            let moved = self.ids[last];
            self.ids[idx] = moved;
            self.pos.insert(moved, idx);
        }
        self.ids.pop();
        self.data.truncate(last * self.dim);
        true
    }
    fn vector(&self, id: u64) -> Option<Vec<f32>> {
        let i = *self.pos.get(&id)?;
        Some(self.data[i * self.dim..(i + 1) * self.dim].to_vec())
    }
    fn compact(&mut self) {
        self.data.shrink_to_fit();
        self.ids.shrink_to_fit();
        self.pos.shrink_to_fit();
    }
}

/// HNSW long-term store. HNSW has no delete: removal tombstones the id and
/// `compact` rebuilds the graph from a retained f32 shadow copy. The shadow
/// copy is real memory and is included in `bytes()`.
pub struct HnswRm {
    dim: usize,
    params: HnswParams,
    inner: HnswIndex,
    shadow: HashMap<u64, Vec<f32>>,
    dead: HashSet<u64>,
}

impl HnswRm {
    pub fn new(dim: usize, params: HnswParams) -> Self {
        Self { dim, params, inner: HnswIndex::new(dim, params), shadow: HashMap::new(), dead: HashSet::new() }
    }
}

impl MemoryIndex for HnswRm {
    fn add(&mut self, id: u64, v: &[f32]) {
        assert_eq!(v.len(), self.dim, "HnswRm: dim mismatch");
        if self.shadow.contains_key(&id) || self.dead.contains(&id) {
            self.remove(id);
            self.compact();
        }
        let nv = normalized(v);
        self.inner.add(id, &nv);
        self.shadow.insert(id, nv);
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.shadow.is_empty() || k == 0 {
            return Vec::new();
        }
        let fetch = k + self.dead.len().min(8 * k);
        let mut r: Vec<(u64, f32)> =
            self.inner.search(q, fetch).into_iter().filter(|(id, _)| !self.dead.contains(id)).collect();
        r.truncate(k);
        r
    }
    fn len(&self) -> usize {
        self.shadow.len()
    }
    fn bytes(&self) -> usize {
        self.inner.bytes() + self.shadow.len() * (self.dim * 4 + 56) + self.dead.len() * 16
    }
    fn name(&self) -> String {
        format!("hnsw-rm-M{}", self.params.m)
    }
}

impl RemovableIndex for HnswRm {
    fn remove(&mut self, id: u64) -> bool {
        if self.shadow.remove(&id).is_some() {
            self.dead.insert(id);
            true
        } else {
            false
        }
    }
    fn vector(&self, id: u64) -> Option<Vec<f32>> {
        self.shadow.get(&id).cloned()
    }
    fn compact(&mut self) {
        if self.dead.is_empty() {
            return;
        }
        let mut idx = HnswIndex::new(self.dim, self.params);
        let mut ids = Vec::with_capacity(self.shadow.len());
        let mut flat = Vec::with_capacity(self.shadow.len() * self.dim);
        for (id, v) in &self.shadow {
            ids.push(*id);
            flat.extend_from_slice(v);
        }
        idx.add_batch(&ids, &flat);
        self.inner = idx;
        self.dead.clear();
        self.shadow.shrink_to_fit();
    }
}

/// Compressed tier: turbovec codes only, no f32 copy. Scores are estimates.
pub struct ColdTv {
    inner: IdMapIndex,
    dim: usize,
    bit_width: usize,
}

impl ColdTv {
    pub fn new(dim: usize, bit_width: usize) -> Self {
        Self { inner: IdMapIndex::new(dim, bit_width).expect("turbovec: invalid dim/bit_width"), dim, bit_width }
    }
    pub fn add_batch(&mut self, ids: &[u64], vs: &[f32]) {
        let mut norm = vs.to_vec();
        norm.chunks_exact_mut(self.dim).for_each(normalize_in_place);
        self.inner.add_with_ids(&norm, ids).expect("turbovec add_with_ids");
    }
    pub fn remove(&mut self, id: u64) -> bool {
        self.inner.remove(id)
    }
    pub fn prepare(&self) {
        self.inner.prepare();
    }
    pub fn len(&self) -> usize {
        self.inner.len()
    }
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.inner.is_empty() || k == 0 {
            return Vec::new();
        }
        let qn = normalized(q);
        let (scores, ids) = self.inner.search(&qn, k.min(self.inner.len()));
        ids.into_iter().zip(scores).collect()
    }
    pub fn bytes(&self) -> usize {
        let n = self.inner.len();
        // same estimate as memory::TurbovecIndex: codes + norm + ids + map + rotation
        n * (self.dim * self.bit_width).div_ceil(8) + n * (4 + 8 + 16) + self.dim * self.dim * 4
    }
}

/// Scalar-quantised cold storage kind (R4 / h4): a simple alternative to turbovec codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColdKind {
    /// turbovec low-bit codes (`Compression::bit_width`).
    Turbovec,
    /// IEEE half precision, 2 bytes per coordinate.
    F16,
    /// symmetric per-vector int8, 1 byte per coordinate + one f32 scale.
    Int8,
}

pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
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
    let mut h = (((e as u32) << 10) | (man >> 13)) as u16;
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign);
        }
        let v = man as f32 * (1.0 / 16_777_216.0); // 2^-24
        return if sign != 0 { -v } else { v };
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (man << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
}

#[inline]
fn dot_i8(row: &[i8], q: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (rc, qc) = (row.chunks_exact(8), q.chunks_exact(8));
    let (rr, qr) = (rc.remainder(), qc.remainder());
    for (r, x) in rc.zip(qc) {
        for i in 0..8 {
            acc[i] += r[i] as f32 * x[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (r, x) in rr.iter().zip(qr) {
        s += *r as f32 * x;
    }
    s
}

#[inline]
fn dot_f16(row: &[u16], q: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (rc, qc) = (row.chunks_exact(8), q.chunks_exact(8));
    let (rr, qr) = (rc.remainder(), qc.remainder());
    for (r, x) in rc.zip(qc) {
        for i in 0..8 {
            acc[i] += f16_to_f32(r[i]) * x[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (r, x) in rr.iter().zip(qr) {
        s += f16_to_f32(*r) * x;
    }
    s
}

/// Brute-force scalar-quantised cold store (f16 or int8). No f32 copy.
pub struct ColdQ {
    dim: usize,
    kind: ColdKind,
    ids: Vec<u64>,
    pos: HashMap<u64, usize>,
    h: Vec<u16>,
    q: Vec<i8>,
    scale: Vec<f32>,
}

impl ColdQ {
    pub fn new(dim: usize, kind: ColdKind) -> Self {
        assert!(kind != ColdKind::Turbovec);
        Self { dim, kind, ids: Vec::new(), pos: HashMap::new(), h: Vec::new(), q: Vec::new(), scale: Vec::new() }
    }
    pub fn add_batch(&mut self, ids: &[u64], vs: &[f32]) {
        for (id, v) in ids.iter().zip(vs.chunks_exact(self.dim)) {
            if self.pos.contains_key(id) {
                self.remove(*id);
            }
            let n = normalized(v);
            self.pos.insert(*id, self.ids.len());
            self.ids.push(*id);
            match self.kind {
                ColdKind::F16 => self.h.extend(n.iter().map(|&x| f32_to_f16(x))),
                ColdKind::Int8 => {
                    let mx = n.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
                    let sc = if mx > 0.0 { mx / 127.0 } else { 1.0 };
                    self.scale.push(sc);
                    self.q.extend(n.iter().map(|&x| (x / sc).round().clamp(-127.0, 127.0) as i8));
                }
                ColdKind::Turbovec => unreachable!(),
            }
        }
    }
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(i) = self.pos.remove(&id) else { return false };
        let last = self.ids.len() - 1;
        let d = self.dim;
        if i != last {
            self.ids[i] = self.ids[last];
            self.pos.insert(self.ids[i], i);
            match self.kind {
                ColdKind::F16 => {
                    let (a, b) = self.h.split_at_mut(last * d);
                    a[i * d..(i + 1) * d].copy_from_slice(&b[..d]);
                }
                ColdKind::Int8 => {
                    let (a, b) = self.q.split_at_mut(last * d);
                    a[i * d..(i + 1) * d].copy_from_slice(&b[..d]);
                    self.scale[i] = self.scale[last];
                }
                ColdKind::Turbovec => unreachable!(),
            }
        }
        self.ids.pop();
        match self.kind {
            ColdKind::F16 => self.h.truncate(last * d),
            ColdKind::Int8 => {
                self.q.truncate(last * d);
                self.scale.pop();
            }
            ColdKind::Turbovec => unreachable!(),
        }
        true
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.ids.is_empty() || k == 0 {
            return Vec::new();
        }
        let qn = normalized(q);
        let d = self.dim;
        let mut top = Vec::with_capacity(k + 1);
        match self.kind {
            ColdKind::F16 => {
                for (row, id) in self.h.chunks_exact(d).zip(&self.ids) {
                    let s = dot_f16(row, &qn);
                    push_topk(&mut top, k, *id, s);
                }
            }
            ColdKind::Int8 => {
                for ((row, id), sc) in self.q.chunks_exact(d).zip(&self.ids).zip(&self.scale) {
                    let s = dot_i8(row, &qn);
                    push_topk(&mut top, k, *id, s * sc);
                }
            }
            ColdKind::Turbovec => unreachable!(),
        }
        top
    }
    pub fn bytes(&self) -> usize {
        let payload = match self.kind {
            ColdKind::F16 => self.dim * 2,
            ColdKind::Int8 => self.dim + 4,
            ColdKind::Turbovec => 0,
        };
        self.ids.len() * (payload + 8 + 24)
    }
}

/// The compressed tier: turbovec codes or scalar-quantised (f16/int8) rows.
pub enum ColdTier {
    Tv(ColdTv),
    Q(ColdQ),
}

impl ColdTier {
    fn add_batch(&mut self, ids: &[u64], vs: &[f32]) {
        match self {
            ColdTier::Tv(c) => c.add_batch(ids, vs),
            ColdTier::Q(c) => c.add_batch(ids, vs),
        }
    }
    fn remove(&mut self, id: u64) -> bool {
        match self {
            ColdTier::Tv(c) => c.remove(id),
            ColdTier::Q(c) => c.remove(id),
        }
    }
    fn prepare(&self) {
        if let ColdTier::Tv(c) = self {
            c.prepare();
        }
    }
    fn len(&self) -> usize {
        match self {
            ColdTier::Tv(c) => c.len(),
            ColdTier::Q(c) => c.len(),
        }
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        match self {
            ColdTier::Tv(c) => c.search(q, k),
            ColdTier::Q(c) => c.search(q, k),
        }
    }
    fn bytes(&self) -> usize {
        match self {
            ColdTier::Tv(c) => c.bytes(),
            ColdTier::Q(c) => c.bytes(),
        }
    }
    /// bytes that do not scale with the number of items (turbovec rotation matrix)
    fn fixed_bytes(&self, dim: usize) -> usize {
        match self {
            ColdTier::Tv(_) => dim * dim * 4,
            ColdTier::Q(_) => 0,
        }
    }
}

// ---------------------------------------------------------------------------
// TieredMemory
// ---------------------------------------------------------------------------

/// Long-term (hot) store backend.
#[derive(Clone, Copy, Debug)]
pub enum LongKind {
    /// Exact flat scan (reclaims bytes immediately on removal).
    Flat,
    /// HNSW graph (+ f32 shadow copy so it can be rebuilt after forgetting).
    Hnsw(HnswParams),
}

/// Compression tier settings.
#[derive(Clone, Copy, Debug)]
pub struct Compression {
    /// turbovec bits per coordinate (2, 3 or 4).
    pub bit_width: usize,
    /// Max items kept as exact f32 in the hot store; lowest-value ones above
    /// this move to the cold tier at consolidation.
    pub hot_cap: usize,
}

#[derive(Clone, Debug)]
pub struct TieredConfig {
    pub dim: usize,
    /// Working-buffer capacity (items).
    pub short_cap: usize,
    pub long: LongKind,
    /// Drop lowest-value items to stay under `byte_budget` (budget is only
    /// enforced when this is on).
    pub forgetting: bool,
    pub byte_budget: usize,
    pub compression: Option<Compression>,
    /// Half-life of the usefulness value, in ticks (adds).
    pub value_halflife: f64,
    /// Starting value of a new item.
    pub init_value: f32,
    /// Run `consolidate` automatically every this many adds (0 = manual only).
    pub consolidate_every: usize,
    /// Which items are forgotten first when over the byte budget (default `Value`).
    pub policy: ForgetPolicy,
    /// Compressed-tier representation (default turbovec; only used when `compression` is set).
    pub cold_kind: ColdKind,
    /// Importance-merge: when over budget, absorb near-duplicate hot items (same `meta`,
    /// cosine >= this) into a surviving neighbour (weights and values add) before forgetting.
    pub merge_sim: Option<f32>,
    /// Max lowest-ranked hot items scanned per merge pass.
    pub merge_scan: usize,
    /// Enforce the byte budget on every `add` that exceeds it (not only at `consolidate`).
    pub hard_budget: bool,
}

/// Eviction order under the byte budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForgetPolicy {
    /// lowest time-decayed usefulness value first (ties: oldest). Default.
    Value,
    /// oldest added first (ignores feedback).
    Fifo,
    /// least recently retrieved (or added) first.
    Lru,
}

impl TieredConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            short_cap: 256,
            long: LongKind::Flat,
            forgetting: false,
            byte_budget: usize::MAX,
            compression: None,
            value_halflife: 10_000.0,
            init_value: 0.2,
            consolidate_every: 128,
            policy: ForgetPolicy::Value,
            cold_kind: ColdKind::Turbovec,
            merge_sim: None,
            merge_scan: 128,
            hard_budget: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Short,
    Hot,
    Cold,
}

#[derive(Clone, Debug)]
struct Entry {
    meta: String,
    tier: Tier,
    added: u64,
    retrievals: u32,
    rewards: u32,
    reward_sum: f32,
    value: f32,
    value_tick: u64,
    weight: u32,
    last_access: u64,
}

/// Usefulness snapshot of one item.
#[derive(Clone, Debug)]
pub struct ItemStats {
    pub tier: Tier,
    pub added: u64,
    pub retrievals: u32,
    pub rewards: u32,
    pub reward_sum: f32,
    /// Current time-decayed value.
    pub value: f32,
    /// Number of original items this entry stands for (1 + absorbed by importance-merge).
    pub weight: u32,
}

#[derive(Clone, Debug, Default)]
pub struct TieredStats {
    pub short: usize,
    pub hot: usize,
    pub cold: usize,
    pub forgotten: u64,
    pub demoted: u64,
    pub merged: u64,
    pub bytes: usize,
}

pub struct TieredMemory {
    cfg: TieredConfig,
    short: VecDeque<(u64, Vec<f32>)>,
    hot: Box<dyn RemovableIndex>,
    cold: Option<ColdTier>,
    entries: HashMap<u64, Entry>,
    tick: u64,
    since_consolidate: usize,
    meta_bytes: usize,
    forgotten: u64,
    demoted: u64,
    merged: u64,
}

impl TieredMemory {
    pub fn new(cfg: TieredConfig) -> Self {
        let hot: Box<dyn RemovableIndex> = match cfg.long {
            LongKind::Flat => Box::new(FlatRm::new(cfg.dim)),
            LongKind::Hnsw(p) => Box::new(HnswRm::new(cfg.dim, p)),
        };
        let cold = cfg.compression.map(|c| match cfg.cold_kind {
            ColdKind::Turbovec => ColdTier::Tv(ColdTv::new(cfg.dim, c.bit_width)),
            k => ColdTier::Q(ColdQ::new(cfg.dim, k)),
        });
        Self {
            cfg,
            short: VecDeque::new(),
            hot,
            cold,
            entries: HashMap::new(),
            tick: 0,
            since_consolidate: 0,
            meta_bytes: 0,
            forgotten: 0,
            demoted: 0,
            merged: 0,
        }
    }

    pub fn config(&self) -> &TieredConfig {
        &self.cfg
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn contains(&self, id: u64) -> bool {
        self.entries.contains_key(&id)
    }
    pub fn meta(&self, id: u64) -> Option<&str> {
        self.entries.get(&id).map(|e| e.meta.as_str())
    }
    pub fn tier_of(&self, id: u64) -> Option<Tier> {
        self.entries.get(&id).map(|e| e.tier)
    }
    pub fn item_stats(&self, id: u64) -> Option<ItemStats> {
        self.entries.get(&id).map(|e| ItemStats {
            tier: e.tier,
            added: e.added,
            retrievals: e.retrievals,
            rewards: e.rewards,
            reward_sum: e.reward_sum,
            value: self.value_now(e),
            weight: e.weight,
        })
    }

    /// Eviction rank (lower = forgotten first); ties are broken by `added` (oldest first).
    fn rank(&self, e: &Entry) -> f64 {
        match self.cfg.policy {
            ForgetPolicy::Value => self.value_now(e) as f64,
            ForgetPolicy::Fifo => 0.0,
            ForgetPolicy::Lru => e.last_access as f64,
        }
    }

    fn value_now(&self, e: &Entry) -> f32 {
        let dt = self.tick.saturating_sub(e.value_tick) as f64;
        (e.value as f64 * (-(dt / self.cfg.value_halflife) * std::f64::consts::LN_2).exp()) as f32
    }

    /// Insert an item. Returns `false` (and does nothing) if `id` is already stored.
    pub fn add(&mut self, id: u64, vec: &[f32], meta: &str) -> bool {
        assert_eq!(vec.len(), self.cfg.dim, "TieredMemory: dim mismatch");
        if self.entries.contains_key(&id) {
            return false;
        }
        self.tick += 1;
        self.entries.insert(
            id,
            Entry {
                meta: meta.to_string(),
                tier: Tier::Short,
                added: self.tick,
                retrievals: 0,
                rewards: 0,
                reward_sum: 0.0,
                value: self.cfg.init_value,
                value_tick: self.tick,
                weight: 1,
                last_access: self.tick,
            },
        );
        self.meta_bytes += meta.len();
        self.short.push_back((id, normalized(vec)));
        while self.short.len() > self.cfg.short_cap {
            let (eid, v) = self.short.pop_front().unwrap();
            self.hot.add(eid, &v);
            if let Some(e) = self.entries.get_mut(&eid) {
                e.tier = Tier::Hot;
            }
        }
        self.since_consolidate += 1;
        if self.cfg.hard_budget && self.cfg.forgetting && self.bytes() > self.cfg.byte_budget {
            self.consolidate();
        } else if self.cfg.consolidate_every > 0 && self.since_consolidate >= self.cfg.consolidate_every {
            self.consolidate();
        }
        true
    }

    /// Read-only merged top-`k`: `(id, score, tier)` best first.
    pub fn search_tiered(&self, q: &[f32], k: usize) -> Vec<(u64, f32, Tier)> {
        assert_eq!(q.len(), self.cfg.dim, "TieredMemory: query dim mismatch");
        if k == 0 {
            return Vec::new();
        }
        let qn = normalized(q);
        let mut all: Vec<(u64, f32, Tier)> = Vec::with_capacity(3 * k);
        let mut top = Vec::with_capacity(k + 1);
        for (id, v) in &self.short {
            push_topk(&mut top, k, *id, dot(&qn, v));
        }
        all.extend(top.into_iter().map(|(i, s)| (i, s, Tier::Short)));
        all.extend(self.hot.search(&qn, k).into_iter().map(|(i, s)| (i, s, Tier::Hot)));
        if let Some(c) = &self.cold {
            all.extend(c.search(&qn, k).into_iter().map(|(i, s)| (i, s, Tier::Cold)));
        }
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(k);
        all
    }

    /// Read-only merged top-`k` `(id, score)`.
    pub fn search_ro(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.search_tiered(q, k).into_iter().map(|(i, s, _)| (i, s)).collect()
    }

    /// Merged top-`k`; also counts a retrieval for each returned item.
    pub fn search(&mut self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        let r = self.search_ro(q, k);
        let tick = self.tick;
        for (id, _) in &r {
            if let Some(e) = self.entries.get_mut(id) {
                e.retrievals = e.retrievals.saturating_add(1);
                e.last_access = tick;
            }
        }
        r
    }

    /// Report how useful a retrieved item was (`reward` clamped to [0,1]).
    /// Returns `false` if the id is not stored (e.g. already forgotten).
    pub fn record_outcome(&mut self, id: u64, reward: f64) -> bool {
        let r = if reward.is_finite() { reward.clamp(0.0, 1.0) as f32 } else { 0.0 };
        let now = self.value_now_for(id);
        let tick = self.tick;
        match (self.entries.get_mut(&id), now) {
            (Some(e), Some(v)) => {
                e.value = v + r;
                e.value_tick = tick;
                e.rewards += 1;
                e.reward_sum += r;
                true
            }
            _ => false,
        }
    }

    fn value_now_for(&self, id: u64) -> Option<f32> {
        self.entries.get(&id).map(|e| self.value_now(e))
    }

    /// Estimated resident bytes of everything the memory holds.
    pub fn bytes(&self) -> usize {
        let short = self.short.len() * (self.cfg.dim * 4 + 8 + 24);
        short
            + self.hot.bytes()
            + self.cold.as_ref().map_or(0, |c| c.bytes())
            + self.entries.len() * ENTRY_OVERHEAD
            + self.meta_bytes
    }

    pub fn stats(&self) -> TieredStats {
        TieredStats {
            short: self.short.len(),
            hot: self.hot.len(),
            cold: self.cold.as_ref().map_or(0, |c| c.len()),
            forgotten: self.forgotten,
            demoted: self.demoted,
            merged: self.merged,
            bytes: self.bytes(),
        }
    }

    /// Demote lowest-value hot items to the cold tier (if compression is on)
    /// and forget lowest-value items until under the byte budget (if
    /// forgetting is on).
    pub fn consolidate(&mut self) {
        self.since_consolidate = 0;
        if self.cfg.compression.is_some() {
            self.demote();
        }
        if self.cfg.forgetting && self.cfg.byte_budget < usize::MAX {
            if self.cfg.merge_sim.is_some() && self.bytes() > self.cfg.byte_budget {
                self.merge_pass();
            }
            self.enforce_budget();
        }
    }

    /// Importance-merge: for the lowest-ranked hot items, absorb an item into its nearest hot
    /// neighbour when both carry the same `meta` and cosine >= `merge_sim`. The survivor's vector
    /// is unchanged; weights and values add, so no usefulness evidence is lost.
    fn merge_pass(&mut self) {
        let Some(thr) = self.cfg.merge_sim else { return };
        let mut cand: Vec<(f64, u64, u64)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.tier == Tier::Hot)
            .map(|(id, e)| (self.rank(e), e.added, *id))
            .collect();
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        cand.truncate(self.cfg.merge_scan);
        let mut any = false;
        for (_, _, id) in cand {
            if !self.entries.get(&id).is_some_and(|e| e.tier == Tier::Hot) {
                continue;
            }
            let Some(v) = self.hot.vector(id) else { continue };
            let meta = self.entries[&id].meta.clone();
            let best = self.hot.search(&v, 6).into_iter().find(|&(oid, s)| {
                oid != id && s >= thr && self.entries.get(&oid).is_some_and(|e| e.tier == Tier::Hot && e.meta == meta)
            });
            let Some((oid, _)) = best else { continue };
            let (w, val) = {
                let e = &self.entries[&id];
                (e.weight, self.value_now(e))
            };
            let sv = self.value_now(&self.entries[&oid]);
            let tick = self.tick;
            if let Some(o) = self.entries.get_mut(&oid) {
                o.weight = o.weight.saturating_add(w);
                o.value = sv + val;
                o.value_tick = tick;
            }
            self.hot.remove(id);
            if let Some(e) = self.entries.remove(&id) {
                self.meta_bytes -= e.meta.len();
            }
            self.merged += 1;
            any = true;
        }
        if any {
            self.hot.compact();
        }
    }

    fn demote(&mut self) {
        let Some(c) = self.cfg.compression else { return };
        let hyst = (c.hot_cap / 10).max(8);
        if self.hot.len() < c.hot_cap + hyst {
            return;
        }
        let need = self.hot.len() - c.hot_cap;
        let mut cand: Vec<(f64, u64, u64)> = self
            .entries
            .iter()
            .filter(|(_, e)| e.tier == Tier::Hot)
            .map(|(id, e)| (self.rank(e), e.added, *id))
            .collect();
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        cand.truncate(need);
        let mut ids = Vec::with_capacity(cand.len());
        let mut flat = Vec::with_capacity(cand.len() * self.cfg.dim);
        for (_, _, id) in &cand {
            if let Some(v) = self.hot.vector(*id) {
                self.hot.remove(*id);
                ids.push(*id);
                flat.extend_from_slice(&v);
                if let Some(e) = self.entries.get_mut(id) {
                    e.tier = Tier::Cold;
                }
            }
        }
        if let Some(cold) = self.cold.as_mut() {
            cold.add_batch(&ids, &flat);
            cold.prepare();
        }
        self.demoted += ids.len() as u64;
        self.hot.compact();
    }

    fn enforce_budget(&mut self) {
        let budget = self.cfg.byte_budget;
        for _ in 0..6 {
            let b = self.bytes();
            if b <= budget {
                return;
            }
            let target = (budget as f64 * 0.97) as usize;
            let excess = b.saturating_sub(target) as f64;
            let hot_avg = if self.hot.len() > 0 { self.hot.bytes() as f64 / self.hot.len() as f64 } else { 0.0 };
            let cold_avg = match &self.cold {
                Some(c) if !c.is_empty() => (c.bytes() - c.fixed_bytes(self.cfg.dim)) as f64 / c.len() as f64,
                _ => 0.0,
            };
            let mut cand: Vec<(f64, u64, u64, Tier)> = self
                .entries
                .iter()
                .filter(|(_, e)| e.tier != Tier::Short)
                .map(|(id, e)| (self.rank(e), e.added, *id, e.tier))
                .collect();
            if cand.is_empty() {
                return;
            }
            cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            let mut freed = 0.0;
            let mut touched_cold = false;
            for (_, _, id, tier) in cand {
                if freed >= excess {
                    break;
                }
                let per = ENTRY_OVERHEAD as f64
                    + self.entries.get(&id).map_or(0, |e| e.meta.len()) as f64
                    + if tier == Tier::Hot { hot_avg } else { cold_avg };
                match tier {
                    Tier::Hot => {
                        self.hot.remove(id);
                    }
                    Tier::Cold => {
                        if let Some(c) = self.cold.as_mut() {
                            c.remove(id);
                            touched_cold = true;
                        }
                    }
                    Tier::Short => unreachable!(),
                }
                if let Some(e) = self.entries.remove(&id) {
                    self.meta_bytes -= e.meta.len();
                }
                self.forgotten += 1;
                freed += per;
            }
            self.hot.compact();
            if touched_cold {
                if let Some(c) = &self.cold {
                    c.prepare();
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::synth::Rng;

    const DIM: usize = 32;

    fn rand_vec(rng: &mut Rng) -> Vec<f32> {
        (0..DIM).map(|_| rng.normal()).collect()
    }

    fn cfg(short: usize) -> TieredConfig {
        let mut c = TieredConfig::new(DIM);
        c.short_cap = short;
        c.consolidate_every = 0;
        c
    }

    #[test]
    fn eviction_moves_oldest_to_long_term() {
        let mut m = TieredMemory::new(cfg(4));
        let mut rng = Rng::new(1);
        for i in 0..10u64 {
            assert!(m.add(i, &rand_vec(&mut rng), "x"));
        }
        let s = m.stats();
        assert_eq!(s.short, 4);
        assert_eq!(s.hot, 6);
        assert_eq!(m.tier_of(0), Some(Tier::Hot));
        assert_eq!(m.tier_of(9), Some(Tier::Short));
        assert!(!m.add(3, &rand_vec(&mut rng), "dup"), "duplicate id rejected");
        assert_eq!(m.len(), 10);
    }

    #[test]
    fn search_merges_tiers_and_is_exact_for_flat() {
        let mut m = TieredMemory::new(cfg(5));
        let mut rng = Rng::new(2);
        let vs: Vec<Vec<f32>> = (0..40).map(|_| rand_vec(&mut rng)).collect();
        for (i, v) in vs.iter().enumerate() {
            m.add(i as u64, v, "");
        }
        // exact reference
        for qi in [0usize, 7, 33, 39] {
            let q = &vs[qi];
            let r = m.search_ro(q, 5);
            let mut truth: Vec<(u64, f32)> =
                vs.iter().enumerate().map(|(i, v)| (i as u64, dot(&normalized(q), &normalized(v)))).collect();
            truth.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let t: Vec<u64> = truth.iter().take(5).map(|x| x.0).collect();
            let g: Vec<u64> = r.iter().map(|x| x.0).collect();
            assert_eq!(g, t);
            assert_eq!(r[0].0, qi as u64);
        }
    }

    #[test]
    fn retrieval_counts_and_outcomes_recorded() {
        let mut m = TieredMemory::new(cfg(2));
        let mut rng = Rng::new(3);
        let v = rand_vec(&mut rng);
        m.add(1, &v, "a");
        m.add(2, &rand_vec(&mut rng), "b");
        m.add(3, &rand_vec(&mut rng), "c");
        let r = m.search(&v, 1);
        assert_eq!(r[0].0, 1);
        assert!(m.record_outcome(1, 1.0));
        assert!(m.record_outcome(1, 5.0)); // clamped to 1
        let st = m.item_stats(1).unwrap();
        assert_eq!(st.retrievals, 1);
        assert_eq!(st.rewards, 2);
        assert!((st.reward_sum - 2.0).abs() < 1e-6);
        assert!(st.value > 2.0);
        assert!(!m.record_outcome(999, 1.0));
        assert_eq!(m.meta(2), Some("b"));
    }

    #[test]
    fn value_decays_with_ticks() {
        let mut c = cfg(1);
        c.value_halflife = 10.0;
        let mut m = TieredMemory::new(c);
        let mut rng = Rng::new(4);
        m.add(0, &rand_vec(&mut rng), "");
        m.record_outcome(0, 1.0);
        let v0 = m.item_stats(0).unwrap().value;
        for i in 1..=10u64 {
            m.add(i, &rand_vec(&mut rng), "");
        }
        let v1 = m.item_stats(0).unwrap().value;
        assert!((v1 / v0 - 0.5).abs() < 0.02, "halves after one half-life: {v0} -> {v1}");
    }

    fn budget_cfg(budget: usize, forgetting: bool) -> TieredConfig {
        let mut c = cfg(10);
        c.forgetting = forgetting;
        c.byte_budget = budget;
        c.consolidate_every = 20;
        c.value_halflife = 1e9;
        c
    }

    #[test]
    fn forgetting_respects_budget_and_keeps_rewarded() {
        let per = DIM * 4 + 8 + 24 + ENTRY_OVERHEAD;
        let budget = 100 * per;
        let mut m = TieredMemory::new(budget_cfg(budget, true));
        let mut rng = Rng::new(5);
        let mut keep = Vec::new();
        for i in 0..1000u64 {
            let v = rand_vec(&mut rng);
            m.add(i, &v, "");
            if i % 100 == 5 {
                // mark as useful while still retrievable
                let r = m.search(&v, 1);
                assert_eq!(r[0].0, i);
                m.record_outcome(i, 1.0);
                keep.push(i);
            }
        }
        m.consolidate();
        assert!(m.bytes() <= budget, "bytes {} > budget {}", m.bytes(), budget);
        assert!(m.stats().forgotten > 800);
        for id in keep {
            assert!(m.contains(id), "rewarded item {id} should survive");
        }
        // oldest unrewarded item is gone (FIFO among equals)
        assert!(!m.contains(0));
        // forgotten ids are not returned
        assert!(m.search_ro(&rand_vec(&mut rng), 20).iter().all(|(id, _)| m.contains(*id)));
    }

    #[test]
    fn without_forgetting_nothing_is_dropped() {
        let mut m = TieredMemory::new(budget_cfg(1000, false));
        let mut rng = Rng::new(6);
        for i in 0..200u64 {
            m.add(i, &rand_vec(&mut rng), "");
        }
        assert_eq!(m.len(), 200);
        assert!(m.bytes() > 1000);
    }

    #[test]
    fn compression_demotes_low_value_and_keeps_them_findable() {
        let mut c = cfg(10);
        c.compression = Some(Compression { bit_width: 4, hot_cap: 50 });
        c.consolidate_every = 25;
        let mut m = TieredMemory::new(c);
        let mut rng = Rng::new(7);
        let vs: Vec<Vec<f32>> = (0..400).map(|_| rand_vec(&mut rng)).collect();
        for (i, v) in vs.iter().enumerate() {
            m.add(i as u64, v, "");
            if i == 3 {
                m.record_outcome(3, 1.0); // value protects it from demotion
            }
        }
        let s = m.stats();
        assert!(s.cold > 250, "cold = {}", s.cold);
        assert!(s.hot <= 50 + 10 + 5, "hot = {}", s.hot);
        assert_eq!(s.short + s.hot + s.cold, 400);
        assert_eq!(m.tier_of(3), Some(Tier::Hot), "rewarded item stays exact");
        // cold items are still found (self-query recall)
        let mut found = 0;
        let mut cold_total = 0;
        for (i, v) in vs.iter().enumerate() {
            if m.tier_of(i as u64) == Some(Tier::Cold) {
                cold_total += 1;
                if m.search_ro(v, 3).iter().any(|(id, _)| *id == i as u64) {
                    found += 1;
                }
            }
        }
        assert!(found as f64 / cold_total as f64 > 0.9, "{found}/{cold_total}");
        // compressed memory is smaller than keeping everything as f32
        let flat_bytes = 400 * (DIM * 4 + 32 + ENTRY_OVERHEAD);
        assert!(m.bytes() < flat_bytes, "{} vs {}", m.bytes(), flat_bytes);
    }

    #[test]
    fn forgetting_also_drops_cold_items() {
        let mut c = cfg(10);
        c.compression = Some(Compression { bit_width: 4, hot_cap: 20 });
        c.forgetting = true;
        c.byte_budget = DIM * DIM * 4 + 30 * (DIM * 4 + 32 + ENTRY_OVERHEAD) + 80 * (DIM / 2 + 28 + ENTRY_OVERHEAD);
        c.consolidate_every = 25;
        c.value_halflife = 1e9;
        let mut m = TieredMemory::new(c.clone());
        let mut rng = Rng::new(8);
        for i in 0..600u64 {
            m.add(i, &rand_vec(&mut rng), "");
        }
        m.consolidate();
        assert!(m.bytes() <= c.byte_budget, "{} > {}", m.bytes(), c.byte_budget);
        let s = m.stats();
        assert!(s.forgotten > 0 && s.cold > 0);
        assert_eq!(s.short + s.hot + s.cold, m.len());
    }

    #[test]
    fn hnsw_long_term_forgets_and_rebuilds() {
        let mut c = cfg(5);
        c.long = LongKind::Hnsw(HnswParams { m: 8, ef_construction: 50, ef_search: 32, max_elements: 1000 });
        let mut m = TieredMemory::new(c);
        let mut rng = Rng::new(9);
        let vs: Vec<Vec<f32>> = (0..120).map(|_| rand_vec(&mut rng)).collect();
        for (i, v) in vs.iter().enumerate() {
            m.add(i as u64, v, "");
        }
        let r = m.search_ro(&vs[10], 3);
        assert_eq!(r[0].0, 10);
        // forget through the removable interface
        let before = m.hot.bytes();
        for id in 0..60u64 {
            m.hot.remove(id);
        }
        m.hot.compact();
        assert!(m.hot.bytes() < before);
        assert_eq!(m.hot.len(), 120 - 5 - 60);
        let r = m.hot.search(&vs[100], 3);
        assert_eq!(r[0].0, 100);
        assert!(m.hot.search(&vs[100], 50).iter().all(|(id, _)| *id >= 60));
    }

    #[test]
    fn flat_rm_swap_remove_keeps_search_correct() {
        let mut f = FlatRm::new(DIM);
        let mut rng = Rng::new(10);
        let vs: Vec<Vec<f32>> = (0..50).map(|_| rand_vec(&mut rng)).collect();
        for (i, v) in vs.iter().enumerate() {
            f.add(i as u64, v);
        }
        assert!(f.remove(0));
        assert!(!f.remove(0));
        assert!(f.remove(25));
        assert_eq!(f.len(), 48);
        for i in [1usize, 49, 30] {
            assert_eq!(f.search(&vs[i], 1)[0].0, i as u64);
        }
        assert!(f.vector(25).is_none());
        assert!(f.vector(49).is_some());
    }

    #[test]
    fn deterministic() {
        let run = || {
            let mut m = TieredMemory::new(budget_cfg(60 * (DIM * 4 + 120), true));
            let mut rng = Rng::new(11);
            for i in 0..300u64 {
                m.add(i, &rand_vec(&mut rng), "");
            }
            let q = rand_vec(&mut rng);
            (m.search_ro(&q, 5), m.bytes(), m.stats().forgotten)
        };
        assert_eq!(run(), run());
    }

    // ---- R4 / h4 additions ----

    #[test]
    fn f16_roundtrip_error_is_small() {
        let mut rng = Rng::new(21);
        for _ in 0..2000 {
            let x = rng.normal() * 0.3;
            let y = f16_to_f32(f32_to_f16(x));
            assert!((x - y).abs() <= x.abs() * 1.0e-3 + 1.0e-6, "{x} -> {y}");
        }
        assert_eq!(f16_to_f32(f32_to_f16(0.0)), 0.0);
        assert_eq!(f16_to_f32(f32_to_f16(1.0)), 1.0);
        assert_eq!(f16_to_f32(f32_to_f16(-2.5)), -2.5);
    }

    #[test]
    fn scalar_cold_tiers_find_exact_queries() {
        for kind in [ColdKind::F16, ColdKind::Int8] {
            let mut c = ColdQ::new(DIM, kind);
            let mut rng = Rng::new(22);
            let vs: Vec<Vec<f32>> = (0..60).map(|_| rand_vec(&mut rng)).collect();
            let ids: Vec<u64> = (0..60).collect();
            let flat: Vec<f32> = vs.iter().flatten().copied().collect();
            c.add_batch(&ids, &flat);
            for i in [0usize, 17, 59] {
                let r = c.search(&vs[i], 1);
                assert_eq!(r[0].0, i as u64);
                assert!(r[0].1 > 0.98, "{kind:?} score {}", r[0].1);
            }
            assert!(c.remove(17));
            assert!(!c.remove(17));
            assert_eq!(c.len(), 59);
            assert_eq!(c.search(&vs[59], 1)[0].0, 59);
            assert_eq!(c.search(&vs[3], 1)[0].0, 3);
            let per = c.bytes() / c.len();
            assert!(per < DIM * 2 + 40);
        }
    }

    #[test]
    fn scalar_cold_tier_in_tiered_memory_saves_bytes() {
        let run = |kind: ColdKind| {
            let mut c = cfg(8);
            c.compression = Some(Compression { bit_width: 4, hot_cap: 20 });
            c.cold_kind = kind;
            c.consolidate_every = 16;
            let mut m = TieredMemory::new(c);
            let mut rng = Rng::new(23);
            for i in 0..200u64 {
                m.add(i, &rand_vec(&mut rng), "");
            }
            m.consolidate();
            (m.stats().cold, m.bytes())
        };
        let (cold_i8, b_i8) = run(ColdKind::Int8);
        let (cold_f16, b_f16) = run(ColdKind::F16);
        assert!(cold_i8 > 100 && cold_f16 > 100);
        assert!(b_i8 < b_f16, "int8 smaller than f16");
    }

    #[test]
    fn fifo_and_lru_policies_order_eviction() {
        let mk = |p: ForgetPolicy| {
            let mut c = cfg(1);
            c.forgetting = true;
            c.policy = p;
            c.hard_budget = true;
            c.byte_budget = 6 * (DIM * 4 + 8 + 24 + ENTRY_OVERHEAD) + 6;
            c.value_halflife = 1.0e9;
            TieredMemory::new(c)
        };
        let mut rng = Rng::new(24);
        let vs: Vec<Vec<f32>> = (0..12).map(|_| rand_vec(&mut rng)).collect();
        // FIFO: oldest hot items go first, regardless of retrievals
        let mut f = mk(ForgetPolicy::Fifo);
        for (i, v) in vs.iter().enumerate() {
            f.add(i as u64, v, "");
            f.search(&vs[0], 1);
            f.record_outcome(0, 1.0);
        }
        assert!(!f.contains(0), "FIFO forgets the oldest even if useful");
        assert!(f.contains(11));
        // LRU: keeps the recently retrieved item 0
        let mut l = mk(ForgetPolicy::Lru);
        for (i, v) in vs.iter().enumerate() {
            l.add(i as u64, v, "");
            l.search(&vs[0], 1);
        }
        assert!(l.contains(0), "LRU keeps the frequently retrieved item");
        assert!(l.bytes() <= l.config().byte_budget);
        // Value: keeps the rewarded item
        let mut v = mk(ForgetPolicy::Value);
        for (i, x) in vs.iter().enumerate() {
            v.add(i as u64, x, "");
            v.record_outcome(0, 1.0);
        }
        assert!(v.contains(0), "value policy keeps the rewarded item");
        assert!(v.bytes() <= v.config().byte_budget, "hard budget holds after every add");
    }

    #[test]
    fn importance_merge_absorbs_same_meta_near_duplicates() {
        let mut c = cfg(1);
        c.forgetting = true;
        c.hard_budget = true;
        c.merge_sim = Some(0.9);
        c.byte_budget = 10 * (DIM * 4 + 8 + 24 + ENTRY_OVERHEAD + 2);
        let mut m = TieredMemory::new(c);
        let mut rng = Rng::new(25);
        let base = rand_vec(&mut rng);
        let other: Vec<f32> = rand_vec(&mut rng);
        for i in 0..40u64 {
            let mut v = base.clone();
            for x in v.iter_mut() {
                *x += 0.02 * rng.normal();
            }
            m.add(i, &v, "a|1");
        }
        m.add(100, &other, "b|1");
        let st = m.stats();
        assert!(st.merged > 0, "near-duplicates merged: {st:?}");
        // total represented weight is preserved for survivors of the base cluster
        let w: u32 = (0..40u64).filter_map(|i| m.item_stats(i)).map(|s| s.weight).sum();
        assert!(w as usize + st.forgotten as usize >= 30, "weights {w} forgotten {}", st.forgotten);
        assert!(m.bytes() <= m.config().byte_budget);
    }
}
