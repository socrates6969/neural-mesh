//! Opt-in flat vector index with storage tiers (R4-B3 / H6, productionised in R6-B / H14).
//!
//! * `Tier::F32`       exact, 4 B/value (the incumbent).
//! * `Tier::BlockFp16` shared f32 scale per block of 32 values, i16 mantissa (`2*d + 4*ceil(d/32)` B/vec; 816 B at d=384).
//!                     H6: identical routing decisions in the closed loop, 0 of 3,000 top-10 slots differ (g15).
//! * `Tier::Int8`      per-vector f32 scale, i8 payload (`d + 4` B/vec; 388 B at d=384), f32 query. H6: TOST-equivalent in the
//!                     closed loop, but id-set recall@10 falls to ~0.96 at N=1e6 -> add the exact rerank above ~1e5 vectors.
//! * optional exact rerank: an f32 copy is kept in RAM and the top `factor*k` approximate candidates are re-scored exactly.
//!
//! Cosine metric = L2-normalise on insert/query (then dot). Std-only; scans are plain iterator loops (no SIMD intrinsics:
//! the H6 AVX2 kernels in `src/bin/g15_common` are 2-3x faster, this module trades that for portability).
//! [`suggest_tier`] ports the H6 size-gating rule (results/h6_gate_rule.md). All quality numbers are
//! SYNTHETIC-until-validated-on-real-data; latency numbers come from a loaded, noisy box (trust ratios, not absolutes).
//! Nothing here changes a default.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tier {
    F32,
    BlockFp16,
    Int8,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::F32 => "f32",
            Tier::BlockFp16 => "block-fp16",
            Tier::Int8 => "int8",
        }
    }
    /// Stored bytes of one vector of dimension `dim` in this tier (payload + scales, no exact copy, no ids).
    pub fn row_bytes(self, dim: usize) -> usize {
        match self {
            Tier::F32 => 4 * dim,
            Tier::BlockFp16 => 2 * dim + 4 * dim.div_ceil(BLOCK),
            Tier::Int8 => dim + 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Metric {
    /// plain inner product of the stored vectors
    Dot,
    /// vectors are L2-normalised on insert and on query
    Cosine,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FlatError {
    DimMismatch { expected: usize, got: usize },
    NonFinite,
}
impl std::fmt::Display for FlatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlatError::DimMismatch { expected, got } => write!(f, "dimension mismatch: expected {expected}, got {got}"),
            FlatError::NonFinite => write!(f, "vector contains a non-finite value"),
        }
    }
}
impl std::error::Error for FlatError {}

const BLOCK: usize = 32;

fn l2_normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// Flat (brute-force) index. Ids are caller-chosen `u64`; `add` of an existing id replaces it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlatIndex {
    dim: usize,
    tier: Tier,
    metric: Metric,
    /// 0 = no rerank; otherwise an exact f32 copy is kept and the top `rerank_factor * k` candidates are re-scored
    rerank_factor: usize,
    ids: Vec<u64>,
    alive: Vec<bool>,
    /// F32 tier payload (n*dim)
    f32s: Vec<f32>,
    /// BlockFp16 mantissas (n*dim) and per-block scales (n*ceil(dim/32))
    i16s: Vec<i16>,
    /// BlockFp16 block scales or Int8 per-vector scales
    scales: Vec<f32>,
    i8s: Vec<i8>,
    /// exact f32 copy for rerank (n*dim) when tier != F32 and rerank_factor > 0
    exact: Vec<f32>,
    #[serde(skip)]
    map: HashMap<u64, usize>,
    live: usize,
    /// R45 (opt-in, default false): 8-accumulator scan kernels (different float summation order, ~1e-7 relative score differences)
    #[serde(default)]
    fast: bool,
}

impl FlatIndex {
    pub fn new(dim: usize, tier: Tier, metric: Metric) -> Self {
        assert!(dim > 0, "dim must be positive");
        FlatIndex { dim, tier, metric, rerank_factor: 0, ids: vec![], alive: vec![], f32s: vec![], i16s: vec![], scales: vec![], i8s: vec![], exact: vec![], map: HashMap::new(), live: 0, fast: false }
    }
    /// R45 (opt-in): use the multi-accumulator scan kernels (see RD-LOG R45; scores differ from the scalar kernels by ~1e-7 relative).
    pub fn with_fast_kernels(mut self, on: bool) -> Self {
        self.fast = on;
        self
    }
    /// Enable the exact rerank (`factor` >= 1; 0 disables). Must be set before adding vectors (it decides whether the f32 copy is kept).
    pub fn with_rerank(mut self, factor: usize) -> Self {
        assert!(self.ids.is_empty(), "set the rerank factor before adding vectors");
        self.rerank_factor = if self.tier == Tier::F32 { 0 } else { factor };
        self
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn tier(&self) -> Tier {
        self.tier
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }
    pub fn rerank_factor(&self) -> usize {
        self.rerank_factor
    }
    /// Live vectors.
    pub fn len(&self) -> usize {
        self.live
    }
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
    /// Allocated slots (live + tombstoned; tombstones are reclaimed by `compact`).
    pub fn slots(&self) -> usize {
        self.ids.len()
    }
    pub fn contains(&self, id: u64) -> bool {
        self.map.contains_key(&id)
    }
    /// Exact bytes of vector storage currently held: tier payload + scales + the f32 rerank copy (if any), over ALL slots
    /// (tombstones included until `compact`). Ids / liveness flags are counted in [`meta_bytes`](Self::meta_bytes).
    pub fn bytes(&self) -> usize {
        self.f32s.len() * 4 + self.i16s.len() * 2 + self.i8s.len() + self.scales.len() * 4 + self.exact.len() * 4
    }
    /// 8 B id + 1 B liveness per slot.
    pub fn meta_bytes(&self) -> usize {
        self.ids.len() * 9
    }
    /// Nominal bytes per stored vector for this configuration (payload + scales + rerank copy).
    pub fn bytes_per_vec(&self) -> usize {
        self.tier.row_bytes(self.dim) + if self.rerank_factor > 0 { 4 * self.dim } else { 0 }
    }

    /// Insert (or replace) a vector.
    pub fn add(&mut self, id: u64, v: &[f32]) -> Result<(), FlatError> {
        if v.len() != self.dim {
            return Err(FlatError::DimMismatch { expected: self.dim, got: v.len() });
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(FlatError::NonFinite);
        }
        if self.map.contains_key(&id) {
            self.remove(id);
        }
        let mut x = v.to_vec();
        if self.metric == Metric::Cosine {
            l2_normalize(&mut x);
        }
        match self.tier {
            Tier::F32 => self.f32s.extend_from_slice(&x),
            Tier::BlockFp16 => {
                for blk in x.chunks(BLOCK) {
                    let mx = blk.iter().fold(0f32, |m, &a| m.max(a.abs()));
                    let sc = if mx > 0.0 { mx / 32767.0 } else { 1.0 };
                    self.scales.push(sc);
                    self.i16s.extend(blk.iter().map(|&a| (a / sc).round().clamp(-32767.0, 32767.0) as i16));
                }
            }
            Tier::Int8 => {
                let mx = x.iter().fold(0f32, |m, &a| m.max(a.abs()));
                let sc = if mx > 0.0 { mx / 127.0 } else { 1.0 };
                self.scales.push(sc);
                self.i8s.extend(x.iter().map(|&a| (a / sc).round().clamp(-127.0, 127.0) as i8));
            }
        }
        if self.rerank_factor > 0 {
            self.exact.extend_from_slice(&x);
        }
        self.map.insert(id, self.ids.len());
        self.ids.push(id);
        self.alive.push(true);
        self.live += 1;
        Ok(())
    }

    /// Tombstone `id`; its bytes are reclaimed by `compact`. Returns whether it existed.
    pub fn remove(&mut self, id: u64) -> bool {
        match self.map.remove(&id) {
            Some(slot) => {
                self.alive[slot] = false;
                self.live -= 1;
                true
            }
            None => false,
        }
    }

    /// Drop tombstoned slots and reclaim their bytes. Search results are unchanged by compaction.
    pub fn compact(&mut self) {
        if self.live == self.ids.len() {
            return;
        }
        let (d, nb) = (self.dim, self.dim.div_ceil(BLOCK));
        let mut out = FlatIndex { ids: vec![], alive: vec![], f32s: vec![], i16s: vec![], scales: vec![], i8s: vec![], exact: vec![], map: HashMap::new(), live: 0, ..self.clone_config() };
        for s in 0..self.ids.len() {
            if !self.alive[s] {
                continue;
            }
            out.map.insert(self.ids[s], out.ids.len());
            out.ids.push(self.ids[s]);
            out.alive.push(true);
            match self.tier {
                Tier::F32 => out.f32s.extend_from_slice(&self.f32s[s * d..(s + 1) * d]),
                Tier::BlockFp16 => {
                    out.i16s.extend_from_slice(&self.i16s[s * d..(s + 1) * d]);
                    out.scales.extend_from_slice(&self.scales[s * nb..(s + 1) * nb]);
                }
                Tier::Int8 => {
                    out.i8s.extend_from_slice(&self.i8s[s * d..(s + 1) * d]);
                    out.scales.push(self.scales[s]);
                }
            }
            if self.rerank_factor > 0 {
                out.exact.extend_from_slice(&self.exact[s * d..(s + 1) * d]);
            }
            out.live += 1;
        }
        *self = out;
    }

    fn clone_config(&self) -> FlatIndex {
        FlatIndex { dim: self.dim, tier: self.tier, metric: self.metric, rerank_factor: self.rerank_factor, ids: vec![], alive: vec![], f32s: vec![], i16s: vec![], scales: vec![], i8s: vec![], exact: vec![], map: HashMap::new(), live: 0, fast: self.fast }
    }

    /// Approximate (tier) score of slot `s` against the prepared query.
    #[inline]
    fn score_slot(&self, s: usize, q: &[f32]) -> f32 {
        let d = self.dim;
        match self.tier {
            Tier::F32 => {
                if self.fast {
                    dot8(&self.f32s[s * d..(s + 1) * d], q)
                } else {
                    dot(&self.f32s[s * d..(s + 1) * d], q)
                }
            }
            Tier::BlockFp16 => {
                let nb = d.div_ceil(BLOCK);
                let mut acc = 0f32;
                for (b, (m, qb)) in self.i16s[s * d..(s + 1) * d].chunks(BLOCK).zip(q.chunks(BLOCK)).enumerate() {
                    let p = if self.fast {
                        dot_i16_8(m, qb)
                    } else {
                        let mut p = 0f32;
                        for (&mi, &qi) in m.iter().zip(qb) {
                            p += mi as f32 * qi;
                        }
                        p
                    };
                    acc += p * self.scales[s * nb + b];
                }
                acc
            }
            Tier::Int8 => {
                let row = &self.i8s[s * d..(s + 1) * d];
                let p = if self.fast {
                    dot_i8_8(row, q)
                } else {
                    let mut p = 0f32;
                    for (&mi, &qi) in row.iter().zip(q) {
                        p += mi as f32 * qi;
                    }
                    p
                };
                p * self.scales[s]
            }
        }
    }

    fn prep_query(&self, query: &[f32]) -> Result<Vec<f32>, FlatError> {
        if query.len() != self.dim {
            return Err(FlatError::DimMismatch { expected: self.dim, got: query.len() });
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(FlatError::NonFinite);
        }
        let mut q = query.to_vec();
        if self.metric == Metric::Cosine {
            l2_normalize(&mut q);
        }
        Ok(q)
    }

    /// Top-`k` by score (descending; ties -> earlier-inserted slot first). With rerank, the returned scores of the
    /// top-k are the EXACT f32 scores; without, they are the tier's dequantised scores.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u64, f32)>, FlatError> {
        let q = self.prep_query(query)?;
        if k == 0 || self.live == 0 {
            return Ok(vec![]);
        }
        let mut cand: Vec<(f32, usize)> = (0..self.ids.len()).filter(|&s| self.alive[s]).map(|s| (self.score_slot(s, &q), s)).collect();
        if self.rerank_factor > 0 {
            let m = (k * self.rerank_factor).max(k);
            cand = top(cand, m);
            let d = self.dim;
            for c in cand.iter_mut() {
                c.0 = dot(&self.exact[c.1 * d..(c.1 + 1) * d], &q);
            }
        }
        Ok(top(cand, k).into_iter().map(|(sc, s)| (self.ids[s], sc)).collect())
    }

    /// Exact brute-force over the f32 data when it is held (F32 tier or rerank copy), else `None`. For audits: measure the recall of a tier against exact.
    pub fn search_exact(&self, query: &[f32], k: usize) -> Option<Vec<(u64, f32)>> {
        let src: &[f32] = if self.tier == Tier::F32 { &self.f32s } else if self.rerank_factor > 0 { &self.exact } else { return None };
        let q = self.prep_query(query).ok()?;
        let d = self.dim;
        let cand: Vec<(f32, usize)> = (0..self.ids.len()).filter(|&s| self.alive[s]).map(|s| (dot(&src[s * d..(s + 1) * d], &q), s)).collect();
        Some(top(cand, k).into_iter().map(|(sc, s)| (self.ids[s], sc)).collect())
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("FlatIndex serialises")
    }
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        let mut ix: FlatIndex = serde_json::from_str(s)?;
        ix.map = ix.ids.iter().enumerate().filter(|(i, _)| ix.alive[*i]).map(|(i, &id)| (id, i)).collect();
        Ok(ix)
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// 8 independent accumulators (lets the compiler vectorise; the scalar `dot` is a single dependent add chain)
#[inline]
pub fn dot8(a: &[f32], b: &[f32]) -> f32 {
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
#[inline]
fn dot_i8_8(a: &[i8], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0f32; 8];
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] as f32 * y[i];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for (x, y) in ra.iter().zip(rb) {
        s += *x as f32 * y;
    }
    s
}
#[inline]
fn dot_i16_8(a: &[i16], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    let mut acc = [0f32; 8];
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] as f32 * y[i];
        }
    }
    let mut s = (acc[0] + acc[4]) + (acc[1] + acc[5]) + (acc[2] + acc[6]) + (acc[3] + acc[7]);
    for (x, y) in ra.iter().zip(rb) {
        s += *x as f32 * y;
    }
    s
}

/// k largest by (score desc, slot asc); deterministic.
fn top(mut v: Vec<(f32, usize)>, k: usize) -> Vec<(f32, usize)> {
    let cmp = |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    if v.len() > k {
        v.select_nth_unstable_by(k, cmp);
        v.truncate(k);
    }
    v.sort_by(cmp);
    v
}

// ------------------------------------------------------------------------------------------------ size gating
/// Backends the H6 rule chooses among (preference order = least compression / least machinery first).
/// Only the first four are implemented by [`FlatIndex`]; the rest are advice (turbovec / RaBitQ-style codes need
/// an external index; the f32 copy "on disk" variants need disk reads, +~4.6 ms/query at 4x rerank, unmeasured at QD>1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    F32,
    BlockFp16,
    /// int8 + exact f32 rerank 4x, f32 copy in RAM: `FlatIndex::new(d, Int8, m).with_rerank(4)`
    Int8RerankRam,
    /// as above but the f32 copy lives on disk (not implemented here)
    Int8RerankDisk,
    Tv4RerankRam,
    Tv4RerankDisk,
    Tv2Rerank16Ram,
    Tv2Rerank16Disk,
    Rabit1Rerank32Ram,
    Rabit1Rerank32Disk,
    /// no flat configuration meets the budgets: needs an ANN graph / more RAM / more latency
    NeedsAnnGraph,
}

impl Advice {
    pub fn label(self) -> &'static str {
        match self {
            Advice::F32 => "flat f32",
            Advice::BlockFp16 => "flat block-fp16",
            Advice::Int8RerankRam => "int8 + f32 rerank 4x (f32 RAM)",
            Advice::Int8RerankDisk => "int8 + f32 rerank 4x (f32 disk)",
            Advice::Tv4RerankRam => "turbovec-4bit + f32 rerank 4x (f32 RAM)",
            Advice::Tv4RerankDisk => "turbovec-4bit + f32 rerank 4x (f32 disk)",
            Advice::Tv2Rerank16Ram => "turbovec-2bit + f32 rerank 16x (f32 RAM)",
            Advice::Tv2Rerank16Disk => "turbovec-2bit + f32 rerank 16x (f32 disk)",
            Advice::Rabit1Rerank32Ram => "RaBitQ-1bit + f32 rerank 32x (f32 RAM)",
            Advice::Rabit1Rerank32Disk => "RaBitQ-1bit + f32 rerank 32x (f32 disk)",
            Advice::NeedsAnnGraph => "NONE (needs ANN graph / more RAM / more latency)",
        }
    }
    /// (tier, rerank factor) if this advice can be built with [`FlatIndex`].
    pub fn flat_config(self) -> Option<(Tier, usize)> {
        match self {
            Advice::F32 => Some((Tier::F32, 0)),
            Advice::BlockFp16 => Some((Tier::BlockFp16, 0)),
            Advice::Int8RerankRam => Some((Tier::Int8, 4)),
            _ => None,
        }
    }
}

struct Cand {
    adv: Advice,
    /// measured p50 (us, single thread, d=384) at N = 1e4, 1e5, 1e6 (results/h6_gate_rule.md; rounded to 1 us)
    p50: [f64; 3],
    recall: [f64; 3],
}

const GRID_N: [f64; 3] = [1e4, 1e5, 1e6];
const CANDS: [Cand; 10] = [
    Cand { adv: Advice::F32, p50: [466.0, 5136.0, 65836.0], recall: [1.0, 1.0, 1.0] },
    Cand { adv: Advice::BlockFp16, p50: [298.0, 3699.0, 43614.0], recall: [1.0, 1.0, 1.0] },
    Cand { adv: Advice::Int8RerankRam, p50: [150.0, 1588.0, 18114.0], recall: [1.0, 1.0, 1.0] },
    Cand { adv: Advice::Int8RerankDisk, p50: [4425.0, 6145.0, 24334.0], recall: [1.0, 1.0, 1.0] },
    Cand { adv: Advice::Tv4RerankRam, p50: [241.0, 1107.0, 14038.0], recall: [1.0, 1.0, 1.0] },
    Cand { adv: Advice::Tv4RerankDisk, p50: [4635.0, 5266.0, 25127.0], recall: [1.0, 1.0, 1.0] },
    Cand { adv: Advice::Tv2Rerank16Ram, p50: [272.0, 1283.0, 10167.0], recall: [1.0, 1.0, 0.972] },
    Cand { adv: Advice::Tv2Rerank16Disk, p50: [17192.0, 20810.0, 35256.0], recall: [1.0, 1.0, 0.972] },
    Cand { adv: Advice::Rabit1Rerank32Ram, p50: [613.0, 4564.0, 45531.0], recall: [1.0, 1.0, 0.927] },
    Cand { adv: Advice::Rabit1Rerank32Disk, p50: [34618.0, 39612.0, 108214.0], recall: [1.0, 1.0, 0.927] },
];

/// RAM bytes per vector of a candidate at dimension `dim` (the f32 copy counts only for the "RAM" variants).
fn adv_ram_per_vec(a: Advice, dim: usize) -> usize {
    let f32c = 4 * dim;
    let tv = |bits: usize| (dim * bits).div_ceil(8) + 4;
    let rabit = dim.next_power_of_two() / 8 + 4;
    match a {
        Advice::F32 => f32c,
        Advice::BlockFp16 => Tier::BlockFp16.row_bytes(dim),
        Advice::Int8RerankRam => Tier::Int8.row_bytes(dim) + f32c,
        Advice::Int8RerankDisk => Tier::Int8.row_bytes(dim),
        Advice::Tv4RerankRam => tv(4) + f32c,
        Advice::Tv4RerankDisk => tv(4),
        Advice::Tv2Rerank16Ram => tv(2) + f32c,
        Advice::Tv2Rerank16Disk => tv(2),
        Advice::Rabit1Rerank32Ram => rabit + f32c,
        Advice::Rabit1Rerank32Disk => rabit,
        Advice::NeedsAnnGraph => usize::MAX,
    }
}

/// Least-squares line p50 = a + b*n through the three measured points (what h6_gate.rs `rule::lat` does).
fn modelled_p50_us(c: &Cand, n: f64, dim: usize) -> f64 {
    let k = 3.0;
    let sx: f64 = GRID_N.iter().sum();
    let sy: f64 = c.p50.iter().sum();
    let sxx: f64 = GRID_N.iter().map(|x| x * x).sum();
    let sxy: f64 = GRID_N.iter().zip(&c.p50).map(|(x, y)| x * y).sum();
    let b = (k * sxy - sx * sy) / (k * sxx - sx * sx);
    let a = (sy - b * sx) / k;
    (a + b * n).max(0.0) * dim as f64 / 384.0
}

/// Recall of the measured grid point nearest to `n` in log distance (never extrapolated upward).
fn recall_at(c: &Cand, n: f64) -> f64 {
    let mut best = 0;
    for i in 1..3 {
        if (GRID_N[i].ln() - n.ln()).abs() < (GRID_N[best].ln() - n.ln()).abs() {
            best = i;
        }
    }
    c.recall[best]
}

/// H6 size-gating rule: the LEAST-compressed backend whose RAM (n * bytes/vec) <= `ram_budget_bytes`, modelled single-thread
/// p50 <= `latency_budget_ms`, and recall >= 0.99; `Advice::NeedsAnnGraph` if none. The latency model is the H6 linear fit on
/// 3 measured sizes (d=384 synthetic mixture, loaded box ~1.4x at 1e6; scaled linearly in dim/384), so:
/// * rows at N = 1e4/1e5/1e6 reproduce the measured grid (27/27 validation cells, results/h6_gate_rule.md),
/// * everything else (e.g. 3e5, > 1e6, dims other than 384, multi-thread) is MODEL OUTPUT, not measurement.
pub fn suggest_tier(n: usize, dim: usize, latency_budget_ms: f64, ram_budget_bytes: usize) -> Advice {
    let nf = n as f64;
    let lat_us = latency_budget_ms * 1000.0;
    for c in CANDS.iter() {
        let ram = (adv_ram_per_vec(c.adv, dim) as f64) * nf;
        if ram <= ram_budget_bytes as f64 && modelled_p50_us(c, nf, dim) <= lat_us && recall_at(c, nf) >= 0.99 {
            return c.adv;
        }
    }
    Advice::NeedsAnnGraph
}

/// R70: p50 (us, single thread, d=384, random unit vectors, N = 1e4 / 1e5 / 1e6) measured on THIS crate's [`FlatIndex`] with
/// `with_fast_kernels(true)` (`src/bin/n9_tier_validate.rs`). The H6 table above was produced by `h6_gate.rs`'s own kernels: for f32 it agrees
/// with FlatIndex-FAST (466/5136/65836 vs 634/4445/50686 us), but for block-fp16 and int8+rerank FlatIndex is 2-4.5x SLOWER than the H6 rows
/// (298/3699/43614 and 150/1588/18114 us), so [`suggest_tier`] is optimistic for advice that `build_suggested` then builds.
const FLAT_FAST_P50: [(Advice, [f64; 3]); 3] = [
    (Advice::F32, [634.0, 4445.0, 50686.0]),
    (Advice::BlockFp16, [910.0, 7963.0, 82453.0]),
    (Advice::Int8RerankRam, [781.0, 6837.0, 69653.0]),
];

/// Flat-scan advice calibrated to what [`FlatIndex`] (FAST kernels) actually delivers (opt-in; [`suggest_tier`] is unchanged).
/// With the fast kernels exact f32 is both the fastest and the most accurate flat tier; compressing only buys RAM
/// (block-fp16 = 816 B/vec at d=384, ~1.6x slower). The int8+rerank variant keeps an f32 copy in RAM, so it never wins here.
/// Returns `None` when neither f32 nor block-fp16 meets the budgets (use an ANN graph, see [`flat_is_better_than_hnsw`]).
/// Latency is the linear fit through the 3 measured sizes scaled by dim/384: only N in [1e4, 1e6] and d near 384 are measurements.
pub fn suggest_flat_fast(n: usize, dim: usize, latency_budget_ms: f64, ram_budget_bytes: usize) -> Option<Advice> {
    let nf = n as f64;
    for (adv, p50) in FLAT_FAST_P50.iter().take(2) {
        let c = Cand { adv: *adv, p50: *p50, recall: [1.0; 3] };
        if (adv_ram_per_vec(*adv, dim) as f64) * nf <= ram_budget_bytes as f64 && modelled_p50_us(&c, nf, dim) <= latency_budget_ms * 1000.0 {
            return Some(*adv);
        }
    }
    None
}

/// R58: below this many vectors an exact FAST flat scan beats an HNSW graph (real corpus N=6,831 d=384: flat 364 us vs HNSW M16 413 us @ recall .981).
/// Synthetic clustered data: N=50k flat 2.35 ms vs HNSW 0.35-0.70 ms (recall .97-1.0); N=200k flat 13.5 ms vs HNSW M24/efc100/efs100 0.53 ms @ .999.
pub const FLAT_HNSW_CROSSOVER_N: usize = 15_000;
pub fn flat_is_better_than_hnsw(n: usize) -> bool {
    n < FLAT_HNSW_CROSSOVER_N
}

/// Build the index suggested by [`suggest_tier`] if it is implementable here (`None` for advice that needs external machinery).
pub fn build_suggested(n: usize, dim: usize, latency_budget_ms: f64, ram_budget_bytes: usize, metric: Metric) -> Option<FlatIndex> {
    let (t, r) = suggest_tier(n, dim, latency_budget_ms, ram_budget_bytes).flat_config()?;
    Some(FlatIndex::new(dim, t, metric).with_rerank(r))
}

#[cfg(test)]
mod tests {
    #[test]
    fn fast_kernels_match_scalar_within_rounding() {
        let dim = 77; // not a multiple of 8 on purpose
        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        for tier in [Tier::F32, Tier::BlockFp16, Tier::Int8] {
            let mut a = FlatIndex::new(dim, tier, Metric::Cosine);
            let mut b = FlatIndex::new(dim, tier, Metric::Cosine).with_fast_kernels(true);
            for id in 0..200u64 {
                let v: Vec<f32> = (0..dim).map(|_| rnd()).collect();
                a.add(id, &v).unwrap();
                b.add(id, &v).unwrap();
            }
            let q: Vec<f32> = (0..dim).map(|_| rnd()).collect();
            let (ra, rb) = (a.search(&q, 10).unwrap(), b.search(&q, 10).unwrap());
            for (x, y) in ra.iter().zip(&rb) {
                assert!((x.1 - y.1).abs() < 1e-5, "{tier:?} {x:?} {y:?}");
            }
            assert_eq!(ra[0].0, rb[0].0);
        }
        let (u, v): (Vec<f32>, Vec<f32>) = ((0..50).map(|i| i as f32 * 0.1).collect(), (0..50).map(|i| 1.0 - i as f32 * 0.01).collect());
        assert!((dot(&u, &v) - dot8(&u, &v)).abs() < 1e-3);
        assert_eq!(dot8(&[], &[]), 0.0);
    }

    use super::*;
    use crate::router::Rng;

    fn gauss(r: &mut Rng) -> f32 {
        let (u1, u2) = (r.f64().max(1e-12), r.f64());
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
    fn data(seed: u64, n: usize, d: usize) -> Vec<Vec<f32>> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| (0..d).map(|_| gauss(&mut r)).collect()).collect()
    }
    fn ids(v: &[(u64, f32)]) -> Vec<u64> {
        v.iter().map(|x| x.0).collect()
    }
    fn recall(a: &[(u64, f32)], truth: &[(u64, f32)]) -> f64 {
        let t = ids(truth);
        ids(a).iter().filter(|i| t.contains(i)).count() as f64 / t.len() as f64
    }
    fn build(tier: Tier, rerank: usize, d: usize, base: &[Vec<f32>]) -> FlatIndex {
        let mut ix = FlatIndex::new(d, tier, Metric::Cosine).with_rerank(rerank);
        for (i, v) in base.iter().enumerate() {
            ix.add(i as u64, v).unwrap();
        }
        ix
    }
    /// brute force over the raw f32 data (independent of the index code)
    fn brute(base: &[Vec<f32>], alive: &dyn Fn(u64) -> bool, q: &[f32], k: usize, cosine: bool) -> Vec<u64> {
        let nrm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let qn = if cosine { nrm(q) } else { 1.0 };
        let mut s: Vec<(f32, u64)> = base
            .iter()
            .enumerate()
            .filter(|(i, _)| alive(*i as u64))
            .map(|(i, v)| {
                let vn = if cosine { nrm(v) } else { 1.0 };
                (v.iter().zip(q).map(|(a, b)| a * b).sum::<f32>() / (vn * qn), i as u64)
            })
            .collect();
        s.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        s.into_iter().take(k).map(|x| x.1).collect()
    }

    #[test]
    fn f32_tier_matches_independent_brute_force() {
        let (n, d) = (500, 48);
        let base = data(1, n, d);
        let ix = build(Tier::F32, 0, d, &base);
        for q in data(2, 20, d) {
            let got = ids(&ix.search(&q, 10).unwrap());
            assert_eq!(got, brute(&base, &|_| true, &q, 10, true));
        }
    }

    #[test]
    fn blockfp16_recall_at_10_equals_f32_on_random_data() {
        let (n, d) = (3000, 96); // 3 blocks, exercises the shared exponent per 32 values
        let base = data(3, n, d);
        let f = build(Tier::F32, 0, d, &base);
        let b = build(Tier::BlockFp16, 0, d, &base);
        let qs = data(4, 200, d);
        let mut rec = 0.0;
        let mut same_order = 0;
        for q in &qs {
            let (rf, rb) = (f.search(q, 10).unwrap(), b.search(q, 10).unwrap());
            rec += recall(&rb, &rf);
            same_order += (ids(&rb) == ids(&rf)) as usize;
        }
        let rec = rec / qs.len() as f64;
        println!("block-fp16 recall@10 vs f32 = {rec:.5}, identical ordered top-10 in {same_order}/{}", qs.len());
        assert!(rec == 1.0, "recall@10 {rec}");
    }

    #[test]
    fn blockfp16_handles_dim_not_multiple_of_32_and_zero_vectors() {
        let d = 50;
        let mut base = data(5, 100, d);
        base.push(vec![0.0; d]);
        let f = build(Tier::F32, 0, d, &base);
        let b = build(Tier::BlockFp16, 0, d, &base);
        for q in data(6, 20, d) {
            assert_eq!(recall(&b.search(&q, 5).unwrap(), &f.search(&q, 5).unwrap()), 1.0);
        }
        assert_eq!(b.bytes(), 101 * Tier::BlockFp16.row_bytes(d));
        assert_eq!(Tier::BlockFp16.row_bytes(50), 100 + 8);
        assert_eq!(Tier::BlockFp16.row_bytes(384), 816);
        assert_eq!(Tier::Int8.row_bytes(384), 388);
    }

    #[test]
    fn int8_recall_with_rerank_at_least_099_and_exact_scores() {
        let (n, d) = (3000, 96);
        let base = data(7, n, d);
        let f = build(Tier::F32, 0, d, &base);
        let raw = build(Tier::Int8, 0, d, &base);
        let rr = build(Tier::Int8, 4, d, &base);
        let qs = data(8, 200, d);
        let (mut r_raw, mut r_rr) = (0.0, 0.0);
        for q in &qs {
            let t = f.search(q, 10).unwrap();
            r_raw += recall(&raw.search(q, 10).unwrap(), &t);
            let got = rr.search(q, 10).unwrap();
            r_rr += recall(&got, &t);
            // reranked scores are the exact f32 scores
            for (id, sc) in &got {
                let ex = t.iter().find(|x| x.0 == *id).map(|x| x.1);
                if let Some(e) = ex {
                    assert!((e - sc).abs() < 1e-5);
                }
            }
        }
        let (r_raw, r_rr) = (r_raw / qs.len() as f64, r_rr / qs.len() as f64);
        println!("int8 recall@10 raw {r_raw:.4}, rerank4x {r_rr:.4}");
        assert!(r_rr >= 0.99, "rerank recall {r_rr}");
        assert!(r_raw >= 0.90, "raw int8 recall {r_raw} (sanity)");
    }

    #[test]
    fn byte_accounting_is_exact() {
        let (n, d) = (257, 70);
        let base = data(9, n, d);
        for (tier, rr, per) in [
            (Tier::F32, 0usize, 4 * 70),
            (Tier::BlockFp16, 0, 2 * 70 + 4 * 3),
            (Tier::Int8, 0, 70 + 4),
            (Tier::Int8, 4, 70 + 4 + 4 * 70),
            (Tier::BlockFp16, 2, 2 * 70 + 4 * 3 + 4 * 70),
            (Tier::F32, 4, 4 * 70), // rerank on the exact tier is a no-op, no extra copy
        ] {
            let mut ix = build(tier, rr, d, &base);
            assert_eq!(ix.bytes(), n * per, "{tier:?} rerank {rr}");
            assert_eq!(ix.bytes_per_vec(), per);
            assert_eq!(ix.meta_bytes(), n * 9);
            // tombstones keep their bytes until compact, then exactly the live rows remain
            for i in 0..100u64 {
                assert!(ix.remove(i));
            }
            assert_eq!(ix.bytes(), n * per);
            ix.compact();
            assert_eq!(ix.bytes(), (n - 100) * per);
            assert_eq!(ix.len(), n - 100);
            assert_eq!(ix.slots(), n - 100);
        }
        // paper numbers at d=384
        assert_eq!(Tier::F32.row_bytes(384), 1536);
    }

    #[test]
    fn remove_and_compact_match_brute_force() {
        let (n, d) = (800, 40);
        let base = data(10, n, d);
        let qs = data(11, 30, d);
        for (tier, rr) in [(Tier::F32, 0usize), (Tier::BlockFp16, 0), (Tier::Int8, 4)] {
            let mut ix = build(tier, rr, d, &base);
            let dead = |i: u64| i % 3 == 0 || (500..560).contains(&i);
            for i in 0..n as u64 {
                if dead(i) {
                    assert!(ix.remove(i));
                }
            }
            assert!(!ix.remove(0), "double remove");
            assert!(!ix.contains(0) && ix.contains(1));
            let before: Vec<Vec<(u64, f32)>> = qs.iter().map(|q| ix.search(q, 10).unwrap()).collect();
            for (q, b) in qs.iter().zip(&before) {
                assert!(ids(b).iter().all(|&i| !dead(i)), "removed id returned");
                if tier == Tier::F32 {
                    assert_eq!(ids(b), brute(&base, &|i| !dead(i), q, 10, true));
                }
            }
            ix.compact();
            let after: Vec<Vec<(u64, f32)>> = qs.iter().map(|q| ix.search(q, 10).unwrap()).collect();
            assert_eq!(before, after, "compact must not change results ({tier:?})");
            // tiered results vs exact brute force over alive data
            let mut rec = 0.0;
            for (q, a) in qs.iter().zip(&after) {
                let t = brute(&base, &|i| !dead(i), q, 10, true);
                rec += ids(a).iter().filter(|i| t.contains(i)).count() as f64 / 10.0;
            }
            assert!(rec / qs.len() as f64 >= 0.99, "{tier:?} recall vs brute after compact {}", rec / qs.len() as f64);
            // re-add a removed id, replace an existing id
            ix.add(0, &base[0]).unwrap();
            assert!(ix.contains(0));
            ix.add(0, &base[1]).unwrap();
            let n_dead = (0..n as u64).filter(|&i| dead(i)).count();
            assert_eq!(ix.len(), n - n_dead + 1);
            let top1 = ix.search(&base[1], 3).unwrap();
            assert!(top1.iter().any(|x| x.0 == 0) && top1.iter().any(|x| x.0 == 1));
        }
    }

    #[test]
    fn dot_metric_and_errors_and_json_roundtrip() {
        let d = 16;
        let mut ix = FlatIndex::new(d, Tier::BlockFp16, Metric::Dot).with_rerank(2);
        assert_eq!(ix.add(1, &[1.0; 3]), Err(FlatError::DimMismatch { expected: 16, got: 3 }));
        let mut bad = vec![0.0; d];
        bad[2] = f32::NAN;
        assert_eq!(ix.add(1, &bad), Err(FlatError::NonFinite));
        let base = data(12, 50, d);
        for (i, v) in base.iter().enumerate() {
            ix.add(i as u64, v).unwrap();
        }
        ix.remove(3);
        let q = &base[7];
        let r = ix.search(q, 5).unwrap();
        assert_eq!(r[0].0, 7, "unnormalised dot: a vector is its own best match only if it has the largest norm; check order is by exact dot");
        let json = ix.to_json();
        let ix2 = FlatIndex::from_json(&json).unwrap();
        assert_eq!(ix2.search(q, 5).unwrap(), r);
        assert!(ix2.contains(4) && !ix2.contains(3));
        assert_eq!(ix2.bytes(), ix.bytes());
        assert!(ix.search(&[0.0; 3], 3).is_err());
        assert!(FlatIndex::new(4, Tier::F32, Metric::Dot).search(&[0.0; 4], 3).unwrap().is_empty());
    }

    #[test]
    fn suggest_tier_matches_h6_rule_table() {
        const MB: usize = 1_000_000;
        // rows of results/h6_gate_rule.md (d=384; budgets 128 MB / 512 MB / 2 GB)
        let rows: [(usize, f64, [Advice; 3]); 10] = [
            (10_000, 1.0, [Advice::F32; 3]),
            (30_000, 1.0, [Advice::BlockFp16; 3]),
            (30_000, 5.0, [Advice::F32; 3]),
            (100_000, 1.0, [Advice::NeedsAnnGraph; 3]),
            (100_000, 5.0, [Advice::BlockFp16; 3]),
            (100_000, 20.0, [Advice::BlockFp16, Advice::F32, Advice::F32]),
            (300_000, 5.0, [Advice::NeedsAnnGraph, Advice::Tv2Rerank16Ram, Advice::Tv4RerankRam]),
            (300_000, 20.0, [Advice::Int8RerankDisk, Advice::F32, Advice::F32]),
            (1_000_000, 5.0, [Advice::NeedsAnnGraph; 3]),
            (1_000_000, 20.0, [Advice::NeedsAnnGraph, Advice::NeedsAnnGraph, Advice::Int8RerankRam]),
        ];
        for (n, lat, exp) in rows {
            for (i, ram) in [128 * MB, 512 * MB, 2000 * MB].into_iter().enumerate() {
                assert_eq!(suggest_tier(n, 384, lat, ram), exp[i], "n={n} lat={lat}ms ram={ram}");
            }
        }
        // small n: always the exact f32 flat index when it fits
        assert_eq!(suggest_tier(1000, 384, 1.0, 128 * MB), Advice::F32);
        // implementable advice builds an index of the right shape
        let ix = build_suggested(1_000_000, 384, 20.0, 2000 * MB, Metric::Cosine).unwrap();
        assert_eq!((ix.tier(), ix.rerank_factor()), (Tier::Int8, 4));
        assert!(build_suggested(300_000, 384, 5.0, 2000 * MB, Metric::Cosine).is_none(), "tv4 is advice only");
    }

    #[test]
    fn suggest_flat_fast_follows_measured_flatindex_grid() {
        const MB: usize = 1_000_000;
        // measured grid: f32 634 / 4445 / 50686 us
        assert_eq!(suggest_flat_fast(10_000, 384, 1.0, 128 * MB), Some(Advice::F32));
        assert_eq!(suggest_flat_fast(100_000, 384, 5.0, 2000 * MB), Some(Advice::F32));
        // 100k vectors f32 = 154 MB: a 128 MB budget forces block-fp16 (82 MB) which needs ~8 ms
        assert_eq!(suggest_flat_fast(100_000, 384, 20.0, 128 * MB), Some(Advice::BlockFp16));
        assert_eq!(suggest_flat_fast(100_000, 384, 5.0, 128 * MB), None);
        assert_eq!(suggest_flat_fast(1_000_000, 384, 20.0, 2000 * MB), None, "1e6 flat needs ~50 ms");
        assert!(flat_is_better_than_hnsw(6_831) && !flat_is_better_than_hnsw(50_000));
    }
}
