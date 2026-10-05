//! Shared vector memory for the mesh.
//!
//! All agents query one [`SharedMemory`]; behind it sits any [`MemoryInde//!   SIMD scan), optionally with exact`]
//! backend:
//!
//! * [`FlatIndex`]     – exact brute-force f32 scan. Ground truth.
//! * [`TurbovecIndex`] – crates.io `turbovec` (TurboQuant, 2–4 bit codes,
//!   SIMD scan), optionally with exact f32 re-ranking of an oversampled
//!   candidate list.
//! * [`HnswIndex`]     – crates.io `hnsw_rs` graph index.
//!
//! Similarity is cosine: every backend L2-normalises vectors on `add` and
//! queries on `search`, then ranks by inner product. Scores are cosine
//! similarities (higher = closer); approximate backends return estimated
//! scores. Ids are caller-chosen and must be unique per index (re-adding an
//! id is a caller error; behaviour is backend-specific).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard};

use hnsw_rs::prelude::{Distance, Hnsw};
use turbovec::IdMapIndex;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A k-NN index over cosine similarity.
pub trait MemoryIndex {
    /// Insert one vector under `id`. The vector is normalised internally.
    fn add(&mut self, id: u64, v: &[f32]);
    /// Top-`k` `(id, cosine)` pairs, best first.
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)>;
    /// Number of stored vectors.
    fn len(&self) -> usize;
    /// Approximate resident bytes of the index (payload + main side tables).
    fn bytes(&self) -> usize;

    /// `true` if no vectors are stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Bulk insert: `vs` is row-major `ids.len() × dim`. Backends override
    /// this when a batched build is much faster than repeated `add`.
    fn add_batch(&mut self, ids: &[u64], vs: &[f32]) {
        if ids.is_empty() {
            return;
        }
        let dim = vs.len() / ids.len();
        for (id, v) in ids.iter().zip(vs.chunks_exact(dim)) {
            self.add(*id, v);
        }
    }
    /// Short backend label for reports.
    fn name(&self) -> String;
}

// ---------------------------------------------------------------------------
// Kernels & helpers
// ---------------------------------------------------------------------------

/// Inner product, 8-lane unrolled so it auto-vectorises.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
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

/// Returns an L2-normalised copy (zero vectors stay zero).
pub fn normalized(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    normalize_in_place(&mut out);
    out
}

/// L2-normalise in place (zero vectors stay zero).
pub fn normalize_in_place(v: &mut [f32]) {
    let n = dot(v, v).sqrt();
    if n > 0.0 {
        let inv = 1.0 / n;
        v.iter_mut().for_each(|x| *x *= inv);
    }
}

/// Bounded best-k collector (k is small, so a sorted Vec beats a heap).
struct TopK {
    k: usize,
    items: Vec<(u64, f32)>, // sorted descending by score
}

impl TopK {
    fn new(k: usize) -> Self {
        Self { k, items: Vec::with_capacity(k + 1) }
    }
    #[inline]
    fn threshold(&self) -> f32 {
        if self.items.len() < self.k { f32::NEG_INFINITY } else { self.items[self.k - 1].1 }
    }
    #[inline]
    fn push(&mut self, id: u64, s: f32) {
        if self.k == 0 || s <= self.threshold() {
            return;
        }
        let pos = self.items.partition_point(|&(_, x)| x >= s);
        self.items.insert(pos, (id, s));
        self.items.truncate(self.k);
    }
    fn into_vec(self) -> Vec<(u64, f32)> {
        self.items
    }
}

/// recall@k = |truth_k ∩ got_k| / k  (ids only).
pub fn recall_at_k(truth: &[(u64, f32)], got: &[(u64, f32)], k: usize) -> f64 {
    let k_eff = k.min(truth.len());
    if k_eff == 0 {
        return 1.0;
    }
    let t: std::collections::HashSet<u64> = truth.iter().take(k_eff).map(|x| x.0).collect();
    let hit = got.iter().take(k).filter(|x| t.contains(&x.0)).count();
    hit as f64 / k_eff as f64
}

// ---------------------------------------------------------------------------
// FlatIndex
// ---------------------------------------------------------------------------

/// Exact brute-force index over contiguous normalised f32 rows.
pub struct FlatIndex {
    dim: usize,
    data: Vec<f32>,
    ids: Vec<u64>,
}

impl FlatIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, data: Vec::new(), ids: Vec::new() }
    }
    pub fn with_capacity(dim: usize, n: usize) -> Self {
        Self { dim, data: Vec::with_capacity(n * dim), ids: Vec::with_capacity(n) }
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Exact scan with an already-normalised query.
    fn scan(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut top = TopK::new(k);
        for (row, id) in self.data.chunks_exact(self.dim).zip(&self.ids) {
            top.push(*id, dot(q, row));
        }
        top.into_vec()
    }
}

impl MemoryIndex for FlatIndex {
    fn add(&mut self, id: u64, v: &[f32]) {
        assert_eq!(v.len(), self.dim, "FlatIndex: dim mismatch");
        let start = self.data.len();
        self.data.extend_from_slice(v);
        normalize_in_place(&mut self.data[start..]);
        self.ids.push(id);
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(q.len(), self.dim, "FlatIndex: query dim mismatch");
        self.scan(&normalized(q), k)
    }
    fn len(&self) -> usize {
        self.ids.len()
    }
    fn bytes(&self) -> usize {
        self.data.capacity() * 4 + self.ids.capacity() * 8
    }
    fn add_batch(&mut self, ids: &[u64], vs: &[f32]) {
        assert_eq!(vs.len(), ids.len() * self.dim, "FlatIndex: batch shape");
        self.data.reserve(vs.len());
        self.ids.reserve(ids.len());
        for (id, v) in ids.iter().zip(vs.chunks_exact(self.dim)) {
            self.add(*id, v);
        }
    }
    fn name(&self) -> String {
        "flat-f32".into()
    }
}

// ---------------------------------------------------------------------------
// TurbovecIndex
// ---------------------------------------------------------------------------

/// Exact f32 copies used to re-rank turbovec's oversampled candidates.
struct Rerank {
    factor: usize,
    store: FlatIndex,
    slot: HashMap<u64, usize>,
}

/// Wrapper over `turbovec::IdMapIndex` (TurboQuant codes, SIMD scan).
///
/// `dim` must be a positive multiple of 8 and `bit_width` ∈ {2,3,4}.
/// Note: for large indexes (≳32k vectors) turbovec parallelises a single
/// query across the global rayon pool.
pub struct TurbovecIndex {
    inner: IdMapIndex,
    dim: usize,
    bit_width: usize,
    rerank: Option<Rerank>,
}

impl TurbovecIndex {
    /// Compressed-only index.
    pub fn new(dim: usize, bit_width: usize) -> Self {
        let inner = IdMapIndex::new(dim, bit_width).expect("turbovec: invalid dim/bit_width");
        Self { inner, dim, bit_width, rerank: None }
    }
    /// Index that fetches `factor * k` candidates from the codes, then
    /// re-scores them exactly from a retained f32 copy (costs flat memory).
    pub fn with_rerank(dim: usize, bit_width: usize, factor: usize) -> Self {
        let mut s = Self::new(dim, bit_width);
        s.rerank = Some(Rerank { factor: factor.max(1), store: FlatIndex::new(dim), slot: HashMap::new() });
        s
    }
    /// Build lazily-materialised search structures now (else the first
    /// search after an add pays for it).
    pub fn prepare(&self) {
        self.inner.prepare();
    }
    fn keep_exact(&mut self, ids: &[u64], vs: &[f32]) {
        if let Some(r) = self.rerank.as_mut() {
            for (id, v) in ids.iter().zip(vs.chunks_exact(self.dim)) {
                r.slot.insert(*id, r.store.len());
                r.store.add(*id, v);
            }
        }
    }
}

impl MemoryIndex for TurbovecIndex {
    fn add(&mut self, id: u64, v: &[f32]) {
        self.add_batch(&[id], v);
    }
    fn add_batch(&mut self, ids: &[u64], vs: &[f32]) {
        assert_eq!(vs.len(), ids.len() * self.dim, "TurbovecIndex: batch shape");
        let mut norm = vs.to_vec();
        norm.chunks_exact_mut(self.dim).for_each(normalize_in_place);
        self.inner.add_with_ids(&norm, ids).expect("turbovec add_with_ids");
        self.keep_exact(ids, &norm);
        if ids.len() > 1 {
            self.inner.prepare();
        }
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(q.len(), self.dim, "TurbovecIndex: query dim mismatch");
        if self.inner.is_empty() || k == 0 {
            return Vec::new();
        }
        let qn = normalized(q);
        let fetch = match &self.rerank {
            Some(r) => (k * r.factor).min(self.inner.len()),
            None => k.min(self.inner.len()),
        };
        let (scores, ids) = self.inner.search(&qn, fetch);
        match &self.rerank {
            None => ids.into_iter().zip(scores).take(k).collect(),
            Some(r) => {
                let mut top = TopK::new(k);
                for id in ids {
                    if let Some(&s) = r.slot.get(&id) {
                        let row = &r.store.data[s * self.dim..(s + 1) * self.dim];
                        top.push(id, dot(&qn, row));
                    }
                }
                top.into_vec()
            }
        }
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn bytes(&self) -> usize {
        let n = self.inner.len();
        // codes + per-vector norm + id (slot_to_id) + id->slot map entry,
        // plus the dim×dim rotation matrix.
        let codes = n * (self.dim * self.bit_width).div_ceil(8);
        let base = codes + n * (4 + 8 + 16) + self.dim * self.dim * 4;
        let rr = self.rerank.as_ref().map_or(0, |r| r.store.bytes() + r.slot.capacity() * 24);
        base + rr
    }
    fn name(&self) -> String {
        match &self.rerank {
            None => format!("turbovec-{}bit", self.bit_width),
            Some(r) => format!("turbovec-{}bit+rerank{}x", self.bit_width, r.factor),
        }
    }
}

// ---------------------------------------------------------------------------
// HnswIndex
// ---------------------------------------------------------------------------

/// `1 - dot` on normalised vectors, clamped at 0 (anndists' `DistDot`
/// asserts `dot <= 1`, which float error can violate for duplicates).
#[derive(Default, Clone, Copy)]
pub struct CosineDist;

impl Distance<f32> for CosineDist {
    #[inline]
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        (1.0 - dot(va, vb)).max(0.0)
    }
}

/// HNSW parameters.
#[derive(Clone, Copy, Debug)]
pub struct HnswParams {
    /// Max neighbours per node per layer (M).
    pub m: usize,
    pub ef_construction: usize,
    /// Beam width at query time (raised to k if smaller).
    pub ef_search: usize,
    /// Capacity hint.
    pub max_elements: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self { m: 16, ef_construction: 200, ef_search: 64, max_elements: 100_000 }
    }
}

/// Wrapper over `hnsw_rs::Hnsw` with a cosine distance.
pub struct HnswIndex {
    hnsw: Hnsw<'static, f32, CosineDist>,
    dim: usize,
    params: HnswParams,
    /// Query beam width; shared so it can be retuned while the index sits
    /// inside a `SharedMemory` (see [`HnswIndex::ef_handle`]).
    ef: Arc<AtomicUsize>,
    n: usize,
}

impl HnswIndex {
    pub fn new(dim: usize, params: HnswParams) -> Self {
        let hnsw = Hnsw::new(params.m, params.max_elements.max(1), 16, params.ef_construction, CosineDist);
        let ef = Arc::new(AtomicUsize::new(params.ef_search));
        Self { hnsw, dim, params, ef, n: 0 }
    }
    pub fn set_ef_search(&self, ef: usize) {
        self.ef.store(ef, Ordering::Relaxed);
    }
    /// Handle for retuning `ef_search` at runtime (recall/latency knob).
    pub fn ef_handle(&self) -> Arc<AtomicUsize> {
        self.ef.clone()
    }

    /// Every stored point. NOTE: hnsw_rs keeps each point only in the layer list of its TOP level, so `get_layer_iterator(0)`
    /// alone yields just the level-0 points (about 94 %); walk all layers.
    fn all_points(&self) -> Vec<Arc<hnsw_rs::prelude::Point<'static, f32>>> {
        let pi = self.hnsw.get_point_indexation();
        (0..=self.hnsw.get_max_level_observed() as usize).flat_map(|l| pi.get_layer_iterator(l).collect::<Vec<_>>()).collect()
    }

    /// Sampled self-recall audit (H1): queries `sample` evenly spaced stored points with their own vector and
    /// returns `(checked, lost)` where `lost` counts points that are not their own top-1 hit (distance > 1e-6).
    /// Cost = `sample` searches (about 0.1-1 ms each). Healthy index: lost = 0 (with the vendored hnsw_rs fix). The
    /// fixed crate still has a RARE build mode (about 0.5 % of builds at N=2000, d=32) in which 5-30 % of the points are
    /// unreachable; a sample of 200 flags it with probability ~1 (see results/h1_hnsw_probe*.log).
    pub fn audit(&self, sample: usize) -> (usize, usize) {
        if self.n == 0 || sample == 0 {
            return (0, 0);
        }
        let mut pts = self.all_points();
        pts.sort_by_key(|p| p.get_origin_id());
        let want = sample.min(pts.len());
        let ef = self.ef.load(Ordering::Relaxed).max(1);
        let (mut checked, mut lost) = (0usize, 0usize);
        for j in 0..want {
            let p = &pts[j * pts.len() / want];
            checked += 1;
            let ok = self.hnsw.search(p.get_v(), 1, ef).first().is_some_and(|nb| nb.d_id == p.get_origin_id() || nb.distance <= 1e-6);
            if !ok {
                lost += 1;
            }
        }
        (checked, lost)
    }

    /// Rebuild the graph from the vectors it already holds (fresh random levels, sequential insert). The caller's ids are kept.
    pub fn rebuild(&mut self) {
        let mut items: Vec<(usize, Vec<f32>)> = self.all_points().iter().map(|p| (p.get_origin_id(), p.get_v().to_vec())).collect();
        items.sort_by_key(|x| x.0);
        let hnsw = Hnsw::new(self.params.m, self.params.max_elements.max(items.len()).max(1), 16, self.params.ef_construction, CosineDist);
        for (id, v) in &items {
            hnsw.insert_slice((v.as_slice(), *id));
        }
        self.hnsw = hnsw;
    }

    /// `audit(sample)`; if more than `max_lost` of the sample are lost, `rebuild()` and re-audit, at most `max_rebuilds`
    /// times. Returns `(lost_in_first_audit, rebuilds_done)`. Opt-in: nothing calls this by default.
    pub fn audit_and_repair(&mut self, sample: usize, max_lost: usize, max_rebuilds: usize) -> (usize, usize) {
        let first = self.audit(sample).1;
        let mut done = 0;
        let mut lost = first;
        while lost > max_lost && done < max_rebuilds {
            self.rebuild();
            done += 1;
            lost = self.audit(sample).1;
        }
        (first, done)
    }
}

impl MemoryIndex for HnswIndex {
    fn add(&mut self, id: u64, v: &[f32]) {
        assert_eq!(v.len(), self.dim, "HnswIndex: dim mismatch");
        self.hnsw.insert_slice((&normalized(v), id as usize));
        self.n += 1;
    }
    fn add_batch(&mut self, ids: &[u64], vs: &[f32]) {
        assert_eq!(vs.len(), ids.len() * self.dim, "HnswIndex: batch shape");
        let mut norm = vs.to_vec();
        norm.chunks_exact_mut(self.dim).for_each(normalize_in_place);
        let items: Vec<(&[f32], usize)> =
            norm.chunks_exact(self.dim).zip(ids).map(|(v, id)| (v, *id as usize)).collect();
        self.hnsw.parallel_insert_slice(&items);
        self.n += ids.len();
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        assert_eq!(q.len(), self.dim, "HnswIndex: query dim mismatch");
        if self.n == 0 || k == 0 {
            return Vec::new();
        }
        let ef = self.ef.load(Ordering::Relaxed).max(k);
        self.hnsw
            .search(&normalized(q), k, ef)
            .into_iter()
            .map(|nb| (nb.d_id as u64, 1.0 - nb.distance))
            .collect()
    }
    fn len(&self) -> usize {
        self.n
    }
    fn bytes(&self) -> usize {
        // vectors + layer-0 neighbour lists (2M entries). hnsw_rs keeps
        // Arc'd points and per-layer Vec<Vec<..>> with spare capacity; the
        // ~150 B/edge factor is calibrated against a counting allocator
        // (memory_bench: ~6.5 KB/vector at dim 384, M 16).
        self.n * (self.dim * 4 + 2 * self.params.m * 150 + 160)
    }
    fn name(&self) -> String {
        format!("hnsw-M{}-ef{}", self.params.m, self.ef.load(Ordering::Relaxed))
    }
}

/// Integrity probe for the `hnsw_rs` reverse-edge bug (see `results/g17_board_entry.md`): builds one HNSW
/// index of `n` seeded random unit vectors with the wrapper's default parameters and returns how many items
/// are NOT their own top-1 hit (0 with the vendored fix; ~0.7 % of `n` with the unpatched registry crate,
/// so `n = 2000` detects it with probability ~1). `[patch.crates-io]` only applies at the workspace root:
/// a crate that depends on `neural-mesh` should call this once (about 0.2 s) or carry the same `[patch]`.
pub fn hnsw_integrity_probe(n: usize, dim: usize, seed: u64) -> usize {
    let mut rng = synth::Rng::new(seed);
    let data: Vec<Vec<f32>> = (0..n).map(|_| normalized(&(0..dim).map(|_| rng.normal()).collect::<Vec<f32>>())).collect();
    let mut h = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
    for (i, v) in data.iter().enumerate() {
        h.add(i as u64, v);
    }
    (0..n).filter(|&i| h.search(&data[i], 1).first().map(|x| x.0) != Some(i as u64)).count()
}

/// A boxed index is an index (lets wrappers such as [`CachedIndex`] sit on `Box<dyn MemoryIndex>`).
impl<T: MemoryIndex + ?Sized> MemoryIndex for Box<T> {
    fn add(&mut self, id: u64, v: &[f32]) {
        (**self).add(id, v)
    }
    fn add_batch(&mut self, ids: &[u64], vs: &[f32]) {
        (**self).add_batch(ids, vs)
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        (**self).search(q, k)
    }
    fn len(&self) -> usize {
        (**self).len()
    }
    fn bytes(&self) -> usize {
        (**self).bytes()
    }
    fn name(&self) -> String {
        (**self).name()
    }
}

// ---------------------------------------------------------------------------
// CachedIndex (opt-in query-result cache, H1)
// ---------------------------------------------------------------------------

/// Opt-in result cache in front of any [`MemoryIndex`]: repeated `search(q, k)` calls with the
/// bit-identical query vector return the stored result list instead of re-scanning / re-walking
/// the graph. Writes need `&mut self`, so a search can never race a write. The cache key is the query's
/// f32 bit pattern plus `k` (hash first, then an exact slice comparison; a collision is a miss).
/// Bounded FIFO (`cap` entries; `cap == 0` = pass-through). Results are what the inner index returned
/// (bit-identical), so recall is unchanged by construction.
///
/// Invalidation (the part that must never be wrong):
/// * DEFAULT (conservative): ANY write (`add`, `add_batch`, and for [`crate::tiered::RemovableIndex`] `remove` /
///   `compact`) clears the whole cache before returning. Safe for every backend.
/// * [`CachedIndex::selective`] (opt-in, for backends whose returned scores are EXACT cosines: `FlatIndex`, `HnswIndex`,
///   `FlatRm`, `HnswRm`; NOT `TurbovecIndex`): an `add` only drops the entries whose top-k the new vector can enter
///   (entry has fewer than `k` results, or `cos(q, v) >= k-th score - 1e-6`); a `remove` only drops entries that
///   contain the removed id; `add_batch` of more vectors than cache entries and `compact` clear everything. A kept entry is
///   exactly the exact-top-k the index would now return (for the approximate HNSW: an answer at least as good as a
///   fresh search, though not necessarily the same list a fresh graph walk would give).
///
/// Caveat: if the inner index's *search behaviour* is retuned without a write (e.g. `HnswIndex::set_ef_search`
/// through its shared handle) call [`CachedIndex::invalidate`] afterwards.
pub struct CachedIndex<I: MemoryIndex> {
    inner: I,
    cap: usize,
    selective: bool,
    state: std::sync::Mutex<ResultCacheState>,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
    wgen: u64,
    kept: u64,
    dropped: u64,
}

struct CacheEntry {
    q: Box<[f32]>,
    /// L2-normalised query (only filled in selective mode)
    qn: Box<[f32]>,
    k: usize,
    res: Arc<[(u64, f32)]>,
}

#[derive(Default)]
struct ResultCacheState {
    map: HashMap<u64, CacheEntry>,
    order: std::collections::VecDeque<u64>,
}

#[inline]
fn query_key(q: &[f32], k: usize) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ k as u64;
    for x in q {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

impl<I: MemoryIndex> CachedIndex<I> {
    pub fn new(inner: I, cap: usize) -> Self {
        Self { inner, cap, selective: false, state: Default::default(), hits: Default::default(), misses: Default::default(), wgen: 0, kept: 0, dropped: 0 }
    }
    /// Switch to selective invalidation (see the type docs; only for exact-score backends).
    pub fn selective(mut self, on: bool) -> Self {
        self.selective = on;
        self
    }
    pub fn inner(&self) -> &I {
        &self.inner
    }
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed) as u64
    }
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed) as u64
    }
    /// Number of writes seen so far.
    pub fn generation(&self) -> u64 {
        self.wgen
    }
    /// (entries kept, entries dropped) across all writes (selective mode bookkeeping).
    pub fn kept_dropped(&self) -> (u64, u64) {
        (self.kept, self.dropped)
    }
    pub fn cached_len(&self) -> usize {
        self.state.lock().map(|s| s.map.len()).unwrap_or(0)
    }
    /// Drop every cached result (call after retuning the inner index without a write).
    pub fn invalidate(&self) {
        if let Ok(mut s) = self.state.lock() {
            s.map.clear();
            s.order.clear();
        }
    }
    fn clear_all(&mut self) {
        // exclusive access: nothing can race this clear
        if let Ok(s) = self.state.get_mut() {
            self.dropped += s.map.len() as u64;
            s.map.clear();
            s.order.clear();
        }
    }
    /// Drop the entries for which `drop_it` is true (selective bookkeeping).
    fn retain_entries(&mut self, drop_it: impl Fn(&CacheEntry) -> bool) {
        if let Ok(s) = self.state.get_mut() {
            let before = s.map.len();
            s.map.retain(|_, e| !drop_it(e));
            let after = s.map.len();
            let ResultCacheState { map, order } = &mut *s;
            order.retain(|h| map.contains_key(h));
            self.dropped += (before - after) as u64;
            self.kept += after as u64;
        }
    }
    fn on_add(&mut self, vs: &[f32], dim: usize) {
        self.wgen += 1;
        if !self.selective || dim == 0 || vs.len() / dim > self.cap.max(1) {
            return self.clear_all();
        }
        let vn: Vec<Vec<f32>> = vs.chunks_exact(dim).map(normalized).collect();
        self.retain_entries(|e| {
            e.res.len() < e.k || e.qn.len() != dim || {
                let kth = e.res[e.res.len() - 1].1;
                vn.iter().any(|v| dot(&e.qn, v) >= kth - 1e-6)
            }
        });
    }
    /// Cached top-`k` as a shared slice (no copy on a hit).
    pub fn search_shared(&self, q: &[f32], k: usize) -> Arc<[(u64, f32)]> {
        if self.cap == 0 {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return Arc::from(self.inner.search(q, k));
        }
        let h = query_key(q, k);
        if let Ok(st) = self.state.lock() {
            if let Some(e) = st.map.get(&h) {
                if e.k == k && e.q.len() == q.len() && e.q.iter().zip(q).all(|(a, b)| a.to_bits() == b.to_bits()) {
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    return e.res.clone();
                }
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let r: Arc<[(u64, f32)]> = Arc::from(self.inner.search(q, k));
        let qn: Box<[f32]> = if self.selective { normalized(q).into() } else { Box::default() };
        if let Ok(mut st) = self.state.lock() {
            if st.map.insert(h, CacheEntry { q: q.into(), qn, k, res: r.clone() }).is_none() {
                st.order.push_back(h);
            }
            while st.map.len() > self.cap {
                match st.order.pop_front() {
                    Some(old) => {
                        st.map.remove(&old);
                    }
                    None => break,
                }
            }
        }
        r
    }
}

impl<I: MemoryIndex> MemoryIndex for CachedIndex<I> {
    fn add(&mut self, id: u64, v: &[f32]) {
        self.on_add(v, v.len());
        self.inner.add(id, v);
    }
    fn add_batch(&mut self, ids: &[u64], vs: &[f32]) {
        let dim = if ids.is_empty() { 0 } else { vs.len() / ids.len() };
        self.on_add(vs, dim);
        self.inner.add_batch(ids, vs);
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.search_shared(q, k).to_vec()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn bytes(&self) -> usize {
        let c: usize = self.state.lock().map(|s| s.map.values().map(|e| (e.q.len() + e.qn.len()) * 4 + e.res.len() * 16 + 96).sum()).unwrap_or(0);
        self.inner.bytes() + c
    }
    fn name(&self) -> String {
        format!("cached{}[{}]", if self.selective { "-sel" } else { "" }, self.inner.name())
    }
}

impl<I: crate::tiered::RemovableIndex> crate::tiered::RemovableIndex for CachedIndex<I> {
    fn remove(&mut self, id: u64) -> bool {
        self.wgen += 1;
        if self.selective {
            self.retain_entries(|e| e.res.iter().any(|r| r.0 == id));
        } else {
            self.clear_all();
        }
        self.inner.remove(id)
    }
    fn vector(&self, id: u64) -> Option<Vec<f32>> {
        self.inner.vector(id)
    }
    fn compact(&mut self) {
        self.wgen += 1;
        self.clear_all();
        self.inner.compact();
    }
}

// ---------------------------------------------------------------------------
// SharedMemory
// ---------------------------------------------------------------------------

/// Boxed, thread-shareable index.
pub type DynIndex = Box<dyn MemoryIndex + Send + Sync>;

/// Cheaply clonable handle: many readers search concurrently (shared read
/// lock); writes take the exclusive lock.
#[derive(Clone)]
pub struct SharedMemory {
    inner: Arc<RwLock<DynIndex>>,
}

impl SharedMemory {
    pub fn new(index: DynIndex) -> Self {
        Self { inner: Arc::new(RwLock::new(index)) }
    }
    pub fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.read().search(q, k)
    }
    pub fn add(&self, id: u64, v: &[f32]) {
        self.inner.write().unwrap_or_else(|e| e.into_inner()).add(id, v);
    }
    pub fn add_batch(&self, ids: &[u64], vs: &[f32]) {
        self.inner.write().unwrap_or_else(|e| e.into_inner()).add_batch(ids, vs);
    }
    pub fn len(&self) -> usize {
        self.read().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn bytes(&self) -> usize {
        self.read().bytes()
    }
    pub fn name(&self) -> String {
        self.read().name()
    }
    /// Hold the read lock for a batch of queries.
    pub fn read(&self) -> RwLockReadGuard<'_, DynIndex> {
        self.inner.read().unwrap_or_else(|e| e.into_inner())
    }
}

// ---------------------------------------------------------------------------
// Synthetic data
// ---------------------------------------------------------------------------

/// Seeded synthetic embeddings (no external RNG dependency).
pub mod synth {
    /// SplitMix64.
    pub struct Rng(u64);

    impl Rng {
        pub fn new(seed: u64) -> Self {
            Self(seed)
        }
        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// Uniform in (0, 1].
        pub fn uniform(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
        }
        /// Standard normal (Box–Muller).
        pub fn normal(&mut self) -> f32 {
            let (u1, u2) = (self.uniform(), self.uniform());
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        }
        pub fn below(&mut self, n: usize) -> usize {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Gaussian-mixture embeddings: `n_clusters` random centroids (unit
    /// norm) plus isotropic noise of total norm ≈ `noise`; rows normalised.
    pub struct Clustered {
        pub dim: usize,
        centroids: Vec<f32>,
        noise: f32,
        rng: Rng,
    }

    impl Clustered {
        pub fn new(dim: usize, n_clusters: usize, noise: f32, seed: u64) -> Self {
            let mut rng = Rng::new(seed);
            let mut centroids: Vec<f32> = (0..dim * n_clusters).map(|_| rng.normal()).collect();
            centroids.chunks_exact_mut(dim).for_each(super::normalize_in_place);
            Self { dim, centroids, noise, rng }
        }
        /// Two-level mixture (closer to real embedding spaces than one
        /// isotropic blob per cluster): `n_topics` unit centroids, each with
        /// `subs` sub-centroids at offset norm `sub_noise`; samples add
        /// point noise of norm `noise` around a random sub-centroid.
        pub fn hierarchical(
            dim: usize,
            n_topics: usize,
            subs: usize,
            sub_noise: f32,
            noise: f32,
            seed: u64,
        ) -> Self {
            let mut top = Self::new(dim, n_topics, sub_noise, seed);
            let mut centroids = Vec::with_capacity(dim * n_topics * subs);
            let scale = sub_noise / (dim as f32).sqrt();
            for t in 0..n_topics {
                for _ in 0..subs {
                    let start = centroids.len();
                    for j in 0..dim {
                        let x = top.centroids[t * dim + j] + scale * top.rng.normal();
                        centroids.push(x);
                    }
                    super::normalize_in_place(&mut centroids[start..]);
                }
            }
            Self { dim, centroids, noise, rng: Rng::new(seed ^ 0xA5A5_A5A5) }
        }

        /// `n` fresh rows, row-major, each L2-normalised.
        pub fn sample(&mut self, n: usize) -> Vec<f32> {
            let nc = self.centroids.len() / self.dim;
            let scale = self.noise / (self.dim as f32).sqrt();
            let mut out = Vec::with_capacity(n * self.dim);
            for _ in 0..n {
                let c = self.rng.below(nc);
                let start = out.len();
                for j in 0..self.dim {
                    let x = self.centroids[c * self.dim + j] + scale * self.rng.normal();
                    out.push(x);
                }
                super::normalize_in_place(&mut out[start..]);
            }
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::synth::{Clustered, Rng};
    use super::*;
    use std::thread;

    fn naive_topk(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        let qn = normalized(q);
        let mut all: Vec<(u64, f32)> = data
            .chunks_exact(dim)
            .enumerate()
            .map(|(i, r)| {
                let rn = normalized(r);
                let s: f64 = qn.iter().zip(&rn).map(|(a, b)| *a as f64 * *b as f64).sum();
                (i as u64, s as f32)
            })
            .collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all.truncate(k);
        all
    }

    fn dataset(n: usize, dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut g = Clustered::new(dim, 20, 1.0, seed);
        (g.sample(n), g.sample(50))
    }

    fn mean_recall(idx: &dyn MemoryIndex, truth: &FlatIndex, qs: &[f32], dim: usize, k: usize) -> f64 {
        let mut r = 0.0;
        let nq = qs.len() / dim;
        for q in qs.chunks_exact(dim) {
            r += recall_at_k(&truth.search(q, k), &idx.search(q, k), k);
        }
        r / nq as f64
    }

    #[test]
    fn flat_exact_matches_naive_sort() {
        let dim = 40; // deliberately not a multiple of 8 (exercises the remainder path)
        let mut rng = Rng::new(7);
        let data: Vec<f32> = (0..500 * dim).map(|_| rng.normal()).collect();
        let mut f = FlatIndex::new(dim);
        for (i, v) in data.chunks_exact(dim).enumerate() {
            f.add(i as u64, v);
        }
        assert_eq!(f.len(), 500);
        for _ in 0..20 {
            let q: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
            let got = f.search(&q, 10);
            let want = naive_topk(&data, dim, &q, 10);
            assert_eq!(got.len(), 10);
            for (g, w) in got.iter().zip(&want) {
                assert!((g.1 - w.1).abs() < 1e-4, "score {} vs {}", g.1, w.1);
            }
            // ids equal except possibly ties
            assert!(recall_at_k(&want, &got, 10) >= 0.9);
            assert!(got.windows(2).all(|w| w[0].1 >= w[1].1), "sorted descending");
        }
    }

    #[test]
    fn flat_self_query_is_top1() {
        let dim = 64;
        let (data, _) = dataset(300, dim, 3);
        let mut f = FlatIndex::new(dim);
        f.add_batch(&(0..300).collect::<Vec<u64>>(), &data);
        for i in [0usize, 17, 299] {
            let r = f.search(&data[i * dim..(i + 1) * dim], 1);
            assert_eq!(r[0].0, i as u64);
            assert!((r[0].1 - 1.0).abs() < 1e-4);
        }
        assert!(f.search(&data[..dim], 0).is_empty());
        assert_eq!(f.search(&data[..dim], 1000).len(), 300);
    }

    #[test]
    fn approximate_backends_recall() {
        let (n, dim, k) = (3000, 128, 10);
        let (data, qs) = dataset(n, dim, 11);
        let ids: Vec<u64> = (0..n as u64).collect();
        let mut flat = FlatIndex::new(dim);
        flat.add_batch(&ids, &data);

        let mut hnsw = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
        hnsw.add_batch(&ids, &data);
        let mut tv4 = TurbovecIndex::new(dim, 4);
        tv4.add_batch(&ids, &data);
        let mut tv2 = TurbovecIndex::new(dim, 2);
        tv2.add_batch(&ids, &data);
        let mut tvr = TurbovecIndex::with_rerank(dim, 4, 4);
        tvr.add_batch(&ids, &data);

        let r_h = mean_recall(&hnsw, &flat, &qs, dim, k);
        let r_4 = mean_recall(&tv4, &flat, &qs, dim, k);
        let r_2 = mean_recall(&tv2, &flat, &qs, dim, k);
        let r_r = mean_recall(&tvr, &flat, &qs, dim, k);
        println!("recall@10 n={n} dim={dim}: hnsw={r_h:.3} tv4={r_4:.3} tv2={r_2:.3} tv4+rerank4x={r_r:.3}");
        assert!(r_h >= 0.9, "hnsw recall {r_h}");
        assert!(r_r >= 0.9, "turbovec+rerank recall {r_r}");
        // Compressed-only codes: report, require clearly-better-than-random.
        assert!(r_4 >= 0.5, "turbovec 4-bit recall {r_4}");
        assert!(r_2 >= 0.3, "turbovec 2-bit recall {r_2}");
        assert!(tv4.bytes() < flat.bytes() / 3);
    }

    #[test]
    fn incremental_add_matches_batch() {
        let (n, dim) = (400, 64);
        let (data, qs) = dataset(n, dim, 5);
        let mut a = TurbovecIndex::new(dim, 4);
        let mut h = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
        for (i, v) in data.chunks_exact(dim).enumerate() {
            a.add(i as u64, v);
            h.add(i as u64, v);
        }
        assert_eq!(a.len(), n);
        assert_eq!(h.len(), n);
        assert_eq!(a.search(&qs[..dim], 10).len(), 10);
        assert_eq!(h.search(&qs[..dim], 10).len(), 10);
    }

    #[test]
    fn concurrent_reads_are_consistent() {
        let (n, dim, k) = (2000, 64, 10);
        let (data, qs) = dataset(n, dim, 9);
        let ids: Vec<u64> = (0..n as u64).collect();
        let backends: Vec<DynIndex> = vec![
            Box::new(FlatIndex::new(dim)),
            Box::new(TurbovecIndex::new(dim, 4)),
            Box::new(HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() })),
        ];
        for b in backends {
            let mem = SharedMemory::new(b);
            mem.add_batch(&ids, &data);
            let expected: Vec<Vec<(u64, f32)>> = qs.chunks_exact(dim).map(|q| mem.search(q, k)).collect();
            let expected = Arc::new(expected);
            let qs = Arc::new(qs.clone());
            let handles: Vec<_> = (0..8)
                .map(|t| {
                    let (mem, exp, qs) = (mem.clone(), expected.clone(), qs.clone());
                    thread::spawn(move || {
                        for rep in 0..5 {
                            for (i, q) in qs.chunks_exact(dim).enumerate() {
                                if (i + t + rep) % 2 == 0 {
                                    let got = mem.search(q, k);
                                    let ids_got: Vec<u64> = got.iter().map(|x| x.0).collect();
                                    let ids_exp: Vec<u64> = exp[i].iter().map(|x| x.0).collect();
                                    assert_eq!(ids_got, ids_exp, "{} thread {t} q{i}", mem.name());
                                }
                            }
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().expect("reader panicked");
            }
            // Phase 2: readers keep querying while a writer inserts.
            let readers: Vec<_> = (0..4)
                .map(|_| {
                    let (mem, qs) = (mem.clone(), qs.clone());
                    thread::spawn(move || {
                        for q in qs.chunks_exact(dim) {
                            assert_eq!(mem.search(q, k).len(), k);
                        }
                    })
                })
                .collect();
            let w = {
                let mem = mem.clone();
                thread::spawn(move || {
                    let mut g = Clustered::new(dim, 3, 1.0, 99);
                    let extra = g.sample(5);
                    for (j, v) in extra.chunks_exact(dim).enumerate() {
                        mem.add(10_000_000 + j as u64, v);
                    }
                })
            };
            for h in readers {
                h.join().expect("reader panicked");
            }
            w.join().unwrap();
            assert_eq!(mem.len(), n + 5);
        }
    }

    /// R4/H1 regression for the hnsw_rs reverse-edge bug (vendor_g17/hnsw_rs_patched via
    /// `[patch.crates-io]`). With the UNPATCHED registry crate about 0.7-1.2 % of stored items (all
    /// level >= 1 points) are not even their own top-1 hit in EVERY build (0 clean builds of 2100, h1_hnsw_probe);
    /// with the fixed crate 0 are lost in ~99.5 % of builds. The fixed crate still has a RARE second mode (about 0.5 %
    /// of builds lose 5-30 % of the items, present with and without the fix, see results/h1_board_entry.md), so the test
    /// allows at most 2 lossy builds out of 16 (P(>=3 | p=0.6 %) < 1e-4) and requires an exact 0 in all the others.
    /// HNSW levels are OS-random (each build independent); checked for `add` and parallel `add_batch`, iid and clustered.
    #[test]
    fn hnsw_loses_no_items_n2000_d32_several_seeds() {
        let (n, dim) = (2000usize, 32usize);
        let ids: Vec<u64> = (0..n as u64).collect();
        let miss = |h: &HnswIndex, data: &[f32]| -> usize {
            h.set_ef_search(64);
            (0..n).filter(|&i| h.search(&data[i * dim..(i + 1) * dim], 1).first().map(|x| x.0) != Some(i as u64)).count()
        };
        let (mut builds, mut lossy) = (0usize, vec![]);
        for seed in 1..=4u64 {
            let mut rng = Rng::new(seed * 1_000_003);
            let iid: Vec<f32> = (0..n * dim).map(|_| rng.normal()).collect();
            let mut cl = Clustered::new(dim, n / 50, 0.8, seed);
            let clust = cl.sample(n);
            for (name, data) in [("iid", &iid), ("clust", &clust)] {
                let mut a = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
                for (i, v) in data.chunks_exact(dim).enumerate() {
                    a.add(i as u64, v);
                }
                let mut b = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
                b.add_batch(&ids, data);
                for (path, h) in [("add", &a), ("add_batch", &b)] {
                    builds += 1;
                    let m = miss(h, data);
                    if m > 0 {
                        lossy.push(format!("{path}/{name}/seed{seed}: {m} lost"));
                    }
                }
            }
        }
        let probe = hnsw_integrity_probe(2000, 32, 99);
        if probe > 0 {
            lossy.push(format!("probe: {probe} lost"));
        }
        assert!(lossy.len() <= 2, "hnsw_rs reverse-edge fix not effective? {} of {} builds lost items: {:?}", lossy.len(), builds + 1, lossy);
        // a bad build (rare second mode) must at least be detected by the sampled audit used for repair
        let mut rng = Rng::new(5);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.normal()).collect();
        let mut h = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
        for (i, v) in data.chunks_exact(dim).enumerate() {
            h.add(i as u64, v);
        }
        let full = miss(&h, &data);
        let (_c, aud) = h.audit(200);
        assert!(aud <= 200 * full / n + 12, "audit {aud} inconsistent with full lost {full}");
    }

    /// Selective invalidation on an EXACT backend: after every add / remove the cached answer is bit-identical to a
    /// fresh uncached search (so entries that were kept are provably still the exact top-k), and it keeps entries
    /// (hit rate under writes is higher than with clear-all).
    #[test]
    fn cached_index_selective_never_stale_and_keeps_entries() {
        use crate::tiered::{FlatRm, RemovableIndex};
        let dim = 16;
        let mut rng = Rng::new(21);
        let vecs: Vec<Vec<f32>> = (0..600).map(|_| (0..dim).map(|_| rng.normal()).collect()).collect();
        let queries: Vec<Vec<f32>> = (0..10).map(|_| (0..dim).map(|_| rng.normal()).collect()).collect();
        let mut sel = CachedIndex::new(FlatRm::new(dim), 64).selective(true);
        let mut all = CachedIndex::new(FlatRm::new(dim), 64);
        let mut plain = FlatRm::new(dim);
        for i in 0..40usize {
            sel.add(i as u64, &vecs[i]);
            all.add(i as u64, &vecs[i]);
            plain.add(i as u64, &vecs[i]);
        }
        let mut next = 40usize;
        for step in 0..400usize {
            match step % 6 {
                0..=2 => {
                    let v = if step % 12 == 0 { queries[step % 10].clone() } else { vecs[next].clone() }; // some adds land inside a cached top-k
                    sel.add(next as u64, &v);
                    all.add(next as u64, &v);
                    plain.add(next as u64, &v);
                    next += 1;
                }
                3 => {
                    if let Some(&(id, _)) = plain.search(&queries[step % 10], 1).first() {
                        assert!(sel.remove(id) && all.remove(id) && plain.remove(id));
                    }
                }
                4 => {
                    // remove an id that is probably not in anyone's top-k
                    let id = (step as u64 * 7) % next as u64;
                    let a = sel.remove(id);
                    assert_eq!(a, all.remove(id));
                    assert_eq!(a, plain.remove(id));
                }
                _ => {
                    sel.compact();
                    all.compact();
                    plain.compact();
                }
            }
            for q in &queries {
                for k in [1usize, 5] {
                    let want = plain.search(q, k);
                    assert_eq!(sel.search(q, k), want, "selective cache stale (step {step})");
                    assert_eq!(all.search(q, k), want);
                    assert_eq!(sel.search(q, k), want);
                }
            }
        }
        let (kept, dropped) = sel.kept_dropped();
        assert!(kept > 0 && dropped > 0, "selective mode should both keep and drop (kept {kept}, dropped {dropped})");
        assert!(sel.hits() > all.hits(), "selective hits {} should exceed clear-all hits {}", sel.hits(), all.hits());
    }

    /// audit()/rebuild() see EVERY point (hnsw_rs lists a point only in its top layer) and keep ids.
    #[test]
    fn hnsw_audit_and_rebuild_cover_all_points() {
        let (n, dim) = (900usize, 16usize);
        let mut rng = Rng::new(77);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.normal()).collect();
        let mut h = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
        for (i, v) in data.chunks_exact(dim).enumerate() {
            h.add(i as u64, v);
        }
        let mut ids: Vec<usize> = h.all_points().iter().map(|p| p.get_origin_id()).collect();
        ids.sort();
        assert_eq!(ids, (0..n).collect::<Vec<_>>(), "all_points must include the upper-layer points");
        let (checked, lost) = h.audit(n);
        assert_eq!(checked, n);
        assert!(lost <= n / 20, "audit reports {lost} lost of {n}");
        h.rebuild();
        assert_eq!(h.len(), n);
        let mut ids2: Vec<usize> = h.all_points().iter().map(|p| p.get_origin_id()).collect();
        ids2.sort();
        assert_eq!(ids2, ids, "rebuild dropped or duplicated points");
        let (first, rb) = h.audit_and_repair(100, 3, 2);
        assert!(first <= 3 && rb == 0, "healthy rebuilt graph flagged: first {first}, rebuilds {rb}");
    }

    /// CachedIndex: repeated queries hit, any write invalidates; the cached answer ALWAYS equals the
    /// inner index's own answer (no stale results after add / remove / compact).
    #[test]
    fn cached_index_never_stale_after_writes() {
        use crate::tiered::{FlatRm, RemovableIndex};
        let dim = 16;
        let mut rng = Rng::new(5);
        let vecs: Vec<Vec<f32>> = (0..300).map(|_| (0..dim).map(|_| rng.normal()).collect()).collect();
        let queries: Vec<Vec<f32>> = (0..12).map(|_| (0..dim).map(|_| rng.normal()).collect()).collect();
        let mut c = CachedIndex::new(FlatRm::new(dim), 64);
        let mut plain = FlatRm::new(dim);
        let mut live: Vec<u64> = vec![];
        let check = |c: &CachedIndex<FlatRm>, plain: &FlatRm| {
            for q in &queries {
                for k in [1usize, 5] {
                    let got = c.search(q, k);
                    assert_eq!(got, plain.search(q, k), "stale/incorrect result");
                    assert_eq!(c.search(q, k), got); // second call = hit, same answer
                }
            }
        };
        for step in 0..300usize {
            match step % 5 {
                0 | 1 | 2 => {
                    c.add(step as u64, &vecs[step]);
                    plain.add(step as u64, &vecs[step]);
                    live.push(step as u64);
                }
                3 => {
                    // remove the item that is currently top-1 for query 0 (the most damaging staleness case)
                    if let Some(&(id, _)) = c.search(&queries[0], 1).first() {
                        assert!(c.remove(id));
                        assert!(plain.remove(id));
                        live.retain(|&x| x != id);
                        assert!(c.search(&queries[0], 3).iter().all(|r| r.0 != id), "removed id returned from cache");
                    }
                }
                _ => {
                    c.compact();
                    plain.compact();
                }
            }
            check(&c, &plain);
        }
        assert!(c.hits() > c.misses() / 2, "expected many hits on repeated queries ({} hits / {} misses)", c.hits(), c.misses());
        assert!(c.generation() >= 300);
        // cap 0 = pass-through; bounded cache never exceeds cap
        let mut p = CachedIndex::new(FlatIndex::new(dim), 0);
        p.add(1, &vecs[0]);
        assert_eq!(p.search(&vecs[0], 1)[0].0, 1);
        assert_eq!((p.hits(), p.cached_len()), (0, 0));
        let mut b = CachedIndex::new(FlatIndex::new(dim), 3);
        b.add_batch(&(0..50u64).collect::<Vec<_>>(), &vecs[..50].concat());
        for q in &vecs[..20] {
            b.search(q, 2);
        }
        assert!(b.cached_len() <= 3);
        // HNSW inner: cached == inner after every write
        let mut h = CachedIndex::new(HnswIndex::new(dim, HnswParams { max_elements: 400, ..Default::default() }), 32);
        for i in 0..200usize {
            h.add(i as u64, &vecs[i]);
            if i % 20 == 0 {
                for q in &queries {
                    let a = h.search(q, 5);
                    assert_eq!(a, h.inner().search(q, 5));
                    assert_eq!(h.search(q, 5), a);
                }
            }
        }
    }
}
