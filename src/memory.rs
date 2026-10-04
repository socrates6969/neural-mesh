//! Shared vector memory for the mesh.
//!
//! All agents query one [`SharedMemory`]; behind it sits any [`MemoryIndex`]
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
}
