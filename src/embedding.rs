//! Embedding abstraction: ONE trait behind which the knowledge/research layers get their vectors.
//!
//! * [`Embedding`] (`Send + Sync`): `dim()`, `embed(text)`, `name()`, plus `embed_task(task)`.
//! * [`HashEmbedding`]: the DEFAULT. Wraps [`crate::knowledge::hash_embed`] exactly, so
//!   `HashEmbedding { dim }.embed_task(t) == hash_embed(t, dim)` bit for bit, and
//!   `.embed(text)` equals `hash_embed` of a task whose features are `text.split_whitespace()`.
//! * [`CachedEmbedding`]: memoising wrapper (key = FNV-1a 64 of the text, bounded FIFO cache,
//!   the stored text is compared so a hash collision is a miss, never a wrong vector).
//! * [`MemoEmbedding`] / [`Embedded`]: opt-in compute-once memo (alloc-free exact key, `Arc<[f32]>` values, encoder-call counter).
//! * [`TableEmbedding`]: in-memory `text -> vector` lookup loaded from a documented JSON fixture
//!   (std + serde_json only). This is what a real backend uses OFFLINE: embeddings are
//!   precomputed elsewhere and shipped as a file.
//! * [`VectorSource`] / [`HookEmbedding`]: interface only for a local hook (command-free, no
//!   process spawn, no network in this crate).
//! * [`EmbeddingConfig`] (serde): `Hash { dim }` (default) or `Real { source, dim, cache, .. }`.
//!   Real-backend code (`RealEmbedding`, loading `source`) is behind the cargo feature
//!   `real-embed` (off by default; no extra dependencies). Without the feature,
//!   [`EmbeddingConfig::build`] returns [`EmbeddingError::FeatureDisabled`] and
//!   [`EmbeddingConfig::build_or_hash`] falls back to the hash embedder and returns a warning
//!   string (callers should surface it).
//!
//! ## Fixture format (JSON, UTF-8)
//! ```json
//! { "name": "minilm-l6", "dim": 3,
//!   "vectors": { "refund policy": [0.1, -0.2, 0.3], "reset password": [0.0, 0.5, 0.5] } }
//! ```
//! Alternatively `"items": [ {"text": "refund policy", "vec": [0.1, -0.2, 0.3]}, ... ]`.
//! `name` is optional; `dim` is optional (inferred from the first vector) but every vector must
//! have exactly that length and be finite. Lookup is exact on the text (callers decide the
//! canonicalisation; for tasks see [`task_text`]).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::knowledge::hash_embed;
use crate::memory::normalize_in_place;
use crate::router::Task;

// ---------------------------------------------------------------------------
// Trait

/// A text/task embedder. Implementations must be deterministic for a fixed configuration.
pub trait Embedding: Send + Sync {
    fn dim(&self) -> usize;
    fn embed(&self, text: &str) -> Vec<f32>;
    fn name(&self) -> &str;
    /// Embed a routed task. Default: embed [`task_text`] (features joined by spaces, then the
    /// `type:<t>` token if any). [`HashEmbedding`] overrides this to hash features exactly.
    fn embed_task(&self, task: &Task) -> Vec<f32> {
        self.embed(&task_text(task))
    }
}

impl<E: Embedding + ?Sized> Embedding for Box<E> {
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        (**self).embed(text)
    }
    fn name(&self) -> &str {
        (**self).name()
    }
    fn embed_task(&self, task: &Task) -> Vec<f32> {
        (**self).embed_task(task)
    }
}

impl<E: Embedding + ?Sized> Embedding for Arc<E> {
    fn dim(&self) -> usize {
        (**self).dim()
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        (**self).embed(text)
    }
    fn name(&self) -> &str {
        (**self).name()
    }
    fn embed_task(&self, task: &Task) -> Vec<f32> {
        (**self).embed_task(task)
    }
}

/// Canonical text of a task for text-keyed backends: features joined by a single space,
/// followed by `type:<t>` when the task has a type.
pub fn task_text(task: &Task) -> String {
    let mut s = task.features.join(" ");
    if let Some(t) = &task.task_type {
        if !s.is_empty() {
            s.push(' ');
        }
        s.push_str("type:");
        s.push_str(t);
    }
    s
}

/// FNV-1a 64 of a string (same constants as `hash_embed`).
pub fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------------
// Hash embedding (default)

/// Signed feature-hashing embedding; the legacy behaviour, unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HashEmbedding {
    pub dim: usize,
}

impl HashEmbedding {
    pub fn new(dim: usize) -> Self {
        HashEmbedding { dim }
    }
}

impl Default for HashEmbedding {
    fn default() -> Self {
        HashEmbedding { dim: 64 }
    }
}

impl Embedding for HashEmbedding {
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        let t = Task { id: 0, task_type: None, features: text.split_whitespace().map(String::from).collect() };
        hash_embed(&t, self.dim)
    }
    fn name(&self) -> &str {
        "hash"
    }
    fn embed_task(&self, task: &Task) -> Vec<f32> {
        hash_embed(task, self.dim)
    }
}

// ---------------------------------------------------------------------------
// Cache

/// Memoising wrapper. Bounded: oldest entry evicted first (FIFO) once `cap` is reached.
pub struct CachedEmbedding<E: Embedding> {
    inner: E,
    cap: usize,
    name: String,
    state: Mutex<CacheState>,
    hits: AtomicU64,
    misses: AtomicU64,
}

#[derive(Default)]
struct CacheState {
    map: HashMap<u64, (String, Vec<f32>)>,
    order: VecDeque<u64>,
}

impl<E: Embedding> CachedEmbedding<E> {
    /// `cap` = maximum number of cached vectors (0 disables caching: every call is a miss).
    pub fn new(inner: E, cap: usize) -> Self {
        let name = format!("cached({})", inner.name());
        CachedEmbedding { inner, cap, name, state: Mutex::new(CacheState::default()), hits: AtomicU64::new(0), misses: AtomicU64::new(0) }
    }
    pub fn inner(&self) -> &E {
        &self.inner
    }
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
    pub fn len(&self) -> usize {
        self.state.lock().map(|s| s.map.len()).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn cached(&self, key_text: &str, compute: impl FnOnce() -> Vec<f32>) -> Vec<f32> {
        let k = fnv1a(key_text);
        if self.cap > 0 {
            if let Ok(st) = self.state.lock() {
                if let Some((t, v)) = st.map.get(&k) {
                    if t == key_text {
                        self.hits.fetch_add(1, Ordering::Relaxed);
                        return v.clone();
                    }
                }
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let v = compute();
        if self.cap > 0 {
            if let Ok(mut st) = self.state.lock() {
                if st.map.insert(k, (key_text.to_string(), v.clone())).is_none() {
                    st.order.push_back(k);
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
        }
        v
    }
}

impl<E: Embedding> Embedding for CachedEmbedding<E> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        // "T" / "K" prefixes keep text and task keys apart
        self.cached(&format!("T{text}"), || self.inner.embed(text))
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn embed_task(&self, task: &Task) -> Vec<f32> {
        // feature boundaries matter for the hash embedder, so key on a separator-joined form
        let mut key = String::from("K");
        for f in &task.features {
            key.push_str(f);
            key.push('\u{1f}');
        }
        if let Some(t) = &task.task_type {
            key.push_str("\u{1e}type:");
            key.push_str(t);
        }
        self.cached(&key, || self.inner.embed_task(task))
    }
}

// ---------------------------------------------------------------------------
// Compute-once memo (opt-in; H1)

/// An embedding computed ONCE and shared (`Arc<[f32]>`): clone it for free, hand it to every
/// consumer of the same decision (knowledge retrieval, case router, lessons, result cache).
pub type Embedded = Arc<[f32]>;

#[derive(Clone, Debug, PartialEq, Eq)]
enum MemoKey {
    Text(Box<str>),
    Task(Box<[Box<str>]>, Option<Box<str>>),
}

/// FNV-1a over a text with a domain tag (no allocation).
#[inline]
fn fnv_text(text: &str) -> u64 {
    let mut h = fnv1a(text);
    h ^= 0x54;
    h.wrapping_mul(0x0000_0100_0000_01b3)
}

/// FNV-1a over a task's features + type, feature boundaries and type marked by separators (no allocation).
#[inline]
fn fnv_task(task: &Task) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ 0x4b;
    for f in &task.features {
        for b in f.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h ^= 0x1f;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if let Some(t) = &task.task_type {
        h ^= 0x1e;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        for b in t.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn task_key_eq(k: &MemoKey, task: &Task) -> bool {
    match k {
        MemoKey::Task(fs, ty) => {
            fs.len() == task.features.len()
                && fs.iter().zip(&task.features).all(|(a, b)| &**a == b.as_str())
                && match (ty, &task.task_type) {
                    (Some(a), Some(b)) => &**a == b.as_str(),
                    (None, None) => true,
                    _ => false,
                }
        }
        _ => false,
    }
}

#[derive(Default)]
struct MemoState {
    map: HashMap<u64, (MemoKey, Embedded)>,
    order: VecDeque<u64>,
}

/// Compute-once embedding memo: wraps ANY [`Embedding`], is itself an [`Embedding`] (so it can be
/// handed to `Knowledge::with_embedder` or a `CaseRouter` closure, see
/// [`MemoEmbedding::task_embedder`]), and exposes the shared [`Embedded`] value so repeated
/// consumers of the same text pay neither a second encoder call nor a vector copy.
///
/// Differences from [`CachedEmbedding`]: (1) the key is hashed from the borrowed text/features with
/// no allocation and verified by an EXACT comparison (a hash collision is a miss, never a wrong
/// vector); (2) values are `Arc<[f32]>` (the `Embedding` trait still returns `Vec<f32>`, so only the
/// `*_shared` methods are copy-free); (3) it counts real encoder calls (`encoder_calls`).
/// Returned vectors are bit-identical to the wrapped embedder's (the first call stores exactly what
/// `inner` returned). Bounded FIFO; `cap == 0` disables memoisation (every call hits the encoder).
/// A pure function of the text: it never goes stale and needs no invalidation (unlike a result cache).
pub struct MemoEmbedding<E: Embedding> {
    inner: E,
    cap: usize,
    name: String,
    state: Mutex<MemoState>,
    hits: AtomicU64,
    calls: AtomicU64,
}

impl<E: Embedding> MemoEmbedding<E> {
    pub fn new(inner: E, cap: usize) -> Self {
        let name = format!("memo({})", inner.name());
        MemoEmbedding { inner, cap, name, state: Mutex::new(MemoState::default()), hits: AtomicU64::new(0), calls: AtomicU64::new(0) }
    }
    pub fn inner(&self) -> &E {
        &self.inner
    }
    /// Memo hits (encoder NOT called).
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }
    /// Number of times the wrapped encoder was actually called (= memo misses).
    pub fn encoder_calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
    pub fn len(&self) -> usize {
        self.state.lock().map(|s| s.map.len()).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Approximate resident bytes of the memo (vectors + keys + map overhead).
    pub fn bytes(&self) -> usize {
        let key_bytes = |k: &MemoKey| match k {
            MemoKey::Text(t) => t.len() + 16,
            MemoKey::Task(fs, ty) => fs.iter().map(|f| f.len() + 16).sum::<usize>() + ty.as_ref().map_or(0, |t| t.len() + 16) + 32,
        };
        self.state.lock().map(|s| s.map.values().map(|(k, v)| v.len() * 4 + 16 + 48 + key_bytes(k)).sum()).unwrap_or(0)
    }
    pub fn clear(&self) {
        if let Ok(mut s) = self.state.lock() {
            s.map.clear();
            s.order.clear();
        }
    }

    fn lookup(&self, h: u64, eq: impl Fn(&MemoKey) -> bool) -> Option<Embedded> {
        if self.cap == 0 {
            return None;
        }
        let st = self.state.lock().ok()?;
        match st.map.get(&h) {
            Some((k, v)) if eq(k) => Some(v.clone()),
            _ => None,
        }
    }
    fn store(&self, h: u64, key: MemoKey, v: &Embedded) {
        if self.cap == 0 {
            return;
        }
        if let Ok(mut st) = self.state.lock() {
            if st.map.insert(h, (key, v.clone())).is_none() {
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
    }

    /// Embedding of free text, computed at most once while resident.
    pub fn embed_shared(&self, text: &str) -> Embedded {
        let h = fnv_text(text);
        if let Some(v) = self.lookup(h, |k| matches!(k, MemoKey::Text(t) if &**t == text)) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return v;
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        let v: Embedded = Arc::from(self.inner.embed(text));
        self.store(h, MemoKey::Text(text.into()), &v);
        v
    }

    /// Embedding of a routed task, computed at most once while resident.
    pub fn embed_task_shared(&self, task: &Task) -> Embedded {
        let h = fnv_task(task);
        if let Some(v) = self.lookup(h, |k| task_key_eq(k, task)) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return v;
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        let v: Embedded = Arc::from(self.inner.embed_task(task));
        let key = MemoKey::Task(task.features.iter().map(|f| f.as_str().into()).collect(), task.task_type.as_deref().map(Into::into));
        self.store(h, key, &v);
        v
    }
}

impl<E: Embedding + 'static> MemoEmbedding<E> {
    /// A `CaseRouter` embedder closure (`Task -> Option<Vec<f32>>`) backed by this memo, so the
    /// router's `local_mass` / `posteriors` / `update` calls for one decision hit the encoder once.
    pub fn task_embedder(self: &Arc<Self>) -> Arc<dyn Fn(&Task) -> Option<Vec<f32>> + Send + Sync> {
        let me = self.clone();
        Arc::new(move |t: &Task| Some(me.embed_task_shared(t).to_vec()))
    }
}

impl<E: Embedding> Embedding for MemoEmbedding<E> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        self.embed_shared(text).to_vec()
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn embed_task(&self, task: &Task) -> Vec<f32> {
        self.embed_task_shared(task).to_vec()
    }
}

// ---------------------------------------------------------------------------
// Table embedding (offline "real" vectors)

/// What a [`TableEmbedding`] returns for a text that is not in the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MissPolicy {
    /// The zero vector (cosine 0 with everything; [`TableEmbedding::misses`] counts them).
    #[default]
    Zero,
    /// Hash embedding of the text at the table's dim (keeps retrieval functional, but the
    /// vector lives in a different space; use only for smoke tests / graceful degradation).
    Hash,
}

#[derive(Debug)]
pub enum EmbeddingError {
    /// `EmbeddingConfig::Real` was requested but the crate was built without `real-embed`.
    FeatureDisabled,
    /// I/O error reading the source file.
    Io(String),
    /// Malformed fixture.
    Parse(String),
    /// Unsupported source (extension/scheme) or inconsistent config.
    Unsupported(String),
}

impl std::fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbeddingError::FeatureDisabled => write!(f, "real embeddings require building with `--features real-embed`"),
            EmbeddingError::Io(e) => write!(f, "embedding source I/O error: {e}"),
            EmbeddingError::Parse(e) => write!(f, "embedding fixture parse error: {e}"),
            EmbeddingError::Unsupported(e) => write!(f, "unsupported embedding source: {e}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

/// In-memory `text -> vector` lookup.
pub struct TableEmbedding {
    name: String,
    dim: usize,
    map: HashMap<String, Vec<f32>>,
    miss: MissPolicy,
    misses: AtomicU64,
}

impl TableEmbedding {
    /// Build from pairs. Errors on a dimension mismatch or non-finite value. `normalize` L2
    /// normalises every vector; `center` first subtracts the table mean (then normalises if asked).
    pub fn from_pairs<I: IntoIterator<Item = (String, Vec<f32>)>>(
        name: &str,
        dim: Option<usize>,
        pairs: I,
        normalize: bool,
        center: bool,
    ) -> Result<Self, EmbeddingError> {
        let mut map: HashMap<String, Vec<f32>> = HashMap::new();
        let mut dim = dim;
        for (t, v) in pairs {
            let d = *dim.get_or_insert(v.len());
            if v.len() != d {
                return Err(EmbeddingError::Parse(format!("vector for {t:?} has length {} but dim is {d}", v.len())));
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err(EmbeddingError::Parse(format!("vector for {t:?} has a non-finite value")));
            }
            map.insert(t, v);
        }
        let dim = dim.ok_or_else(|| EmbeddingError::Parse("empty table and no dim given".into()))?;
        if center && !map.is_empty() {
            let mut mean = vec![0.0f64; dim];
            for v in map.values() {
                for (m, x) in mean.iter_mut().zip(v) {
                    *m += *x as f64;
                }
            }
            let n = map.len() as f64;
            for v in map.values_mut() {
                for (x, m) in v.iter_mut().zip(&mean) {
                    *x -= (*m / n) as f32;
                }
            }
        }
        if normalize {
            for v in map.values_mut() {
                normalize_in_place(v);
            }
        }
        Ok(TableEmbedding { name: name.to_string(), dim, map, miss: MissPolicy::Zero, misses: AtomicU64::new(0) })
    }

    /// Parse the JSON fixture (see module docs).
    pub fn from_json_str(s: &str, normalize: bool, center: bool) -> Result<Self, EmbeddingError> {
        let v: serde_json::Value = serde_json::from_str(s).map_err(|e| EmbeddingError::Parse(e.to_string()))?;
        let obj = v.as_object().ok_or_else(|| EmbeddingError::Parse("top level must be an object".into()))?;
        let name = obj.get("name").and_then(|x| x.as_str()).unwrap_or("table").to_string();
        let dim = obj.get("dim").and_then(|x| x.as_u64()).map(|d| d as usize);
        let vec_of = |a: &serde_json::Value| -> Result<Vec<f32>, EmbeddingError> {
            let arr = a.as_array().ok_or_else(|| EmbeddingError::Parse("vector must be an array".into()))?;
            arr.iter().map(|x| x.as_f64().map(|f| f as f32).ok_or_else(|| EmbeddingError::Parse("vector entries must be numbers".into()))).collect()
        };
        let mut pairs: Vec<(String, Vec<f32>)> = Vec::new();
        if let Some(m) = obj.get("vectors") {
            let m = m.as_object().ok_or_else(|| EmbeddingError::Parse("`vectors` must be an object".into()))?;
            for (t, a) in m {
                pairs.push((t.clone(), vec_of(a)?));
            }
        } else if let Some(items) = obj.get("items") {
            let items = items.as_array().ok_or_else(|| EmbeddingError::Parse("`items` must be an array".into()))?;
            for it in items {
                let t = it.get("text").and_then(|x| x.as_str()).ok_or_else(|| EmbeddingError::Parse("item without `text`".into()))?;
                let a = it.get("vec").ok_or_else(|| EmbeddingError::Parse("item without `vec`".into()))?;
                pairs.push((t.to_string(), vec_of(a)?));
            }
        } else {
            return Err(EmbeddingError::Parse("need `vectors` or `items`".into()));
        }
        // deterministic order (serde_json maps may be unordered): irrelevant for the result but
        // makes error messages stable
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        Self::from_pairs(&name, dim, pairs, normalize, center)
    }

    /// Read the JSON fixture from a local file.
    pub fn from_path(path: &str, normalize: bool, center: bool) -> Result<Self, EmbeddingError> {
        let s = std::fs::read_to_string(path).map_err(|e| EmbeddingError::Io(format!("{path}: {e}")))?;
        Self::from_json_str(&s, normalize, center)
    }

    pub fn with_miss(mut self, p: MissPolicy) -> Self {
        self.miss = p;
        self
    }
    pub fn lookup(&self, text: &str) -> Option<&[f32]> {
        self.map.get(text).map(|v| v.as_slice())
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// Number of `embed` calls that were not in the table.
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }
}

impl Embedding for TableEmbedding {
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        match self.map.get(text) {
            Some(v) => v.clone(),
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                match self.miss {
                    MissPolicy::Zero => vec![0.0; self.dim],
                    MissPolicy::Hash => HashEmbedding::new(self.dim).embed(text),
                }
            }
        }
    }
    fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// Local hook interface (definition only)

/// A local, in-process vector provider (e.g. a precomputed index or a model the caller already
/// holds). This crate never spawns a process or opens a socket for it.
pub trait VectorSource: Send + Sync {
    fn dim(&self) -> usize;
    /// `None` = unknown text.
    fn vector(&self, text: &str) -> Option<Vec<f32>>;
    fn name(&self) -> &str {
        "hook"
    }
}

/// [`Embedding`] over a [`VectorSource`]; unknown texts give the zero vector.
pub struct HookEmbedding<S: VectorSource> {
    pub src: S,
}

impl<S: VectorSource> Embedding for HookEmbedding<S> {
    fn dim(&self) -> usize {
        self.src.dim()
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        self.src.vector(text).unwrap_or_else(|| vec![0.0; self.src.dim()])
    }
    fn name(&self) -> &str {
        self.src.name()
    }
}

// ---------------------------------------------------------------------------
// Real backend (feature `real-embed`)

/// Offline "real" embeddings: a [`TableEmbedding`] loaded from a local fixture file, optionally
/// memoised. Only compiled with `--features real-embed`.
#[cfg(feature = "real-embed")]
pub struct RealEmbedding {
    inner: Box<dyn Embedding>,
    name: String,
}

#[cfg(feature = "real-embed")]
impl RealEmbedding {
    /// `source`: path to a `.json` fixture (see module docs). `dim` (0 = whatever the fixture
    /// says) must match the fixture. NO network and NO process spawn.
    pub fn open(source: &str, dim: usize, cache: bool, normalize: bool, center: bool) -> Result<Self, EmbeddingError> {
        if !source.to_ascii_lowercase().ends_with(".json") {
            return Err(EmbeddingError::Unsupported(format!("{source}: only .json fixtures are supported")));
        }
        let t = TableEmbedding::from_path(source, normalize, center)?;
        if dim != 0 && t.dim() != dim {
            return Err(EmbeddingError::Unsupported(format!("fixture dim {} != configured dim {dim}", t.dim())));
        }
        let name = format!("real({})", t.name());
        let inner: Box<dyn Embedding> = if cache { Box::new(CachedEmbedding::new(t, 4096)) } else { Box::new(t) };
        Ok(RealEmbedding { inner, name })
    }
}

#[cfg(feature = "real-embed")]
impl Embedding for RealEmbedding {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        self.inner.embed(text)
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn embed_task(&self, task: &Task) -> Vec<f32> {
        self.inner.embed_task(task)
    }
}

// ---------------------------------------------------------------------------
// Config

fn yes() -> bool {
    true
}

/// Which embedder to use. `Default` = the legacy hash embedder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EmbeddingConfig {
    /// Signed feature hashing. `dim` 0 = inherit the owning config's dimension (for
    /// `ResearchConfig` that is `ResearchConfig::dim`, 64 by default) so defaults are unchanged.
    Hash {
        #[serde(default)]
        dim: usize,
    },
    /// Precomputed real vectors from a local fixture (needs feature `real-embed`).
    Real {
        /// Path to a local `.json` fixture (see module docs). No network, no process spawn.
        source: String,
        /// Expected dimension (0 = take it from the fixture).
        #[serde(default)]
        dim: usize,
        /// Memoise lookups.
        #[serde(default)]
        cache: bool,
        /// L2-normalise the table vectors (default true).
        #[serde(default = "yes")]
        normalize: bool,
        /// Subtract the table mean first (default false).
        #[serde(default)]
        center: bool,
    },
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        EmbeddingConfig::Hash { dim: 0 }
    }
}

impl EmbeddingConfig {
    pub fn is_default(&self) -> bool {
        *self == EmbeddingConfig::default()
    }

    /// Build the embedder. `default_dim` is used by `Hash { dim: 0 }` (and `Real` with dim 0 and
    /// no fixture info does not need it). Without the `real-embed` feature, `Real` returns
    /// [`EmbeddingError::FeatureDisabled`].
    pub fn build(&self, default_dim: usize) -> Result<Arc<dyn Embedding>, EmbeddingError> {
        match self {
            EmbeddingConfig::Hash { dim } => Ok(Arc::new(HashEmbedding::new(if *dim == 0 { default_dim } else { *dim }))),
            #[cfg(feature = "real-embed")]
            EmbeddingConfig::Real { source, dim, cache, normalize, center } => Ok(Arc::new(RealEmbedding::open(source, *dim, *cache, *normalize, *center)?)),
            #[cfg(not(feature = "real-embed"))]
            EmbeddingConfig::Real { .. } => Err(EmbeddingError::FeatureDisabled),
        }
    }

    /// Like [`build`](Self::build) but never fails: on error falls back to the hash embedder of
    /// `default_dim` and returns a human-readable warning (documented fallback; log it).
    pub fn build_or_hash(&self, default_dim: usize) -> (Arc<dyn Embedding>, Option<String>) {
        match self.build(default_dim) {
            Ok(e) => (e, None),
            Err(err) => (Arc::new(HashEmbedding::new(default_dim)), Some(format!("embedding config {self:?} unavailable ({err}); falling back to hash embedding dim {default_dim}"))),
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn task(fs: &[&str], ty: Option<&str>) -> Task {
        Task { id: 0, task_type: ty.map(String::from), features: fs.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn default_config_equals_legacy_hash_embed_bitwise() {
        let (e, w) = EmbeddingConfig::default().build_or_hash(64);
        assert!(w.is_none());
        assert_eq!(e.dim(), 64);
        for t in [task(&["refund", "policy"], None), task(&["a", "b", "c"], Some("billing")), task(&[], None)] {
            let a = e.embed_task(&t);
            let b = hash_embed(&t, 64);
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
        // text path == whitespace-split features
        let a = e.embed("refund   policy now");
        let b = hash_embed(&task(&["refund", "policy", "now"], None), 64);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
        // explicit dim wins over default_dim
        assert_eq!(EmbeddingConfig::Hash { dim: 16 }.build(64).unwrap().dim(), 16);
    }

    #[test]
    fn config_serde_defaults() {
        let c: EmbeddingConfig = serde_json::from_str(r#"{"Hash":{}}"#).unwrap();
        assert_eq!(c, EmbeddingConfig::default());
        let c: EmbeddingConfig = serde_json::from_str(r#"{"Real":{"source":"x.json"}}"#).unwrap();
        assert_eq!(c, EmbeddingConfig::Real { source: "x.json".into(), dim: 0, cache: false, normalize: true, center: false });
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(serde_json::from_str::<EmbeddingConfig>(&s).unwrap(), c);
    }

    #[test]
    fn cache_hit_equals_miss_and_is_bounded() {
        let c = CachedEmbedding::new(HashEmbedding::new(32), 2);
        let a = c.embed("alpha beta");
        let b = c.embed("alpha beta");
        assert_eq!(a, b);
        assert_eq!((c.hits(), c.misses()), (1, 1));
        assert_eq!(a, HashEmbedding::new(32).embed("alpha beta"));
        c.embed("x");
        c.embed("y"); // evicts "alpha beta" (FIFO, cap 2)
        assert_eq!(c.len(), 2);
        let m0 = c.misses();
        assert_eq!(c.embed("alpha beta"), a);
        assert_eq!(c.misses(), m0 + 1);
        // task path caches too and equals the uncached value
        let t = task(&["p", "q"], Some("ty"));
        let u = c.embed_task(&t);
        assert_eq!(u, c.embed_task(&t));
        assert_eq!(u, HashEmbedding::new(32).embed_task(&t));
        // cap 0 = no caching, still correct
        let z = CachedEmbedding::new(HashEmbedding::new(8), 0);
        assert_eq!(z.embed("a"), z.embed("a"));
        assert_eq!((z.hits(), z.len()), (0, 0));
    }

    const FIX: &str = r#"{"name":"toy","dim":3,"vectors":{"cat":[1,0,0],"dog":[0.8,0.6,0],"car":[0,0,2]}}"#;

    #[test]
    fn table_lookup_json_and_items_forms() {
        let t = TableEmbedding::from_json_str(FIX, true, false).unwrap();
        assert_eq!((t.dim(), t.len(), t.name()), (3, 3, "toy"));
        assert_eq!(t.embed("cat"), vec![1.0, 0.0, 0.0]);
        assert_eq!(t.embed("car"), vec![0.0, 0.0, 1.0]); // normalised
        assert_eq!(t.embed("unknown"), vec![0.0; 3]);
        assert_eq!(t.misses(), 1);
        let h = TableEmbedding::from_json_str(FIX, true, false).unwrap().with_miss(MissPolicy::Hash);
        assert_eq!(h.embed("zzz"), HashEmbedding::new(3).embed("zzz"));
        let items = r#"{"items":[{"text":"a","vec":[3,4]},{"text":"b","vec":[0,1]}]}"#;
        let t2 = TableEmbedding::from_json_str(items, true, false).unwrap();
        assert_eq!(t2.dim(), 2);
        let a = t2.embed("a");
        assert!((a[0] - 0.6).abs() < 1e-6 && (a[1] - 0.8).abs() < 1e-6);
        // default embed_task goes through task_text
        assert_eq!(task_text(&task(&["x", "y"], Some("t"))), "x y type:t");
        let one = TableEmbedding::from_pairs("o", None, vec![("cat".to_string(), vec![1.0, 0.0])], false, false).unwrap();
        assert_eq!(one.embed_task(&task(&["cat"], None)), vec![1.0, 0.0]);
    }

    #[test]
    fn table_rejects_bad_fixtures_and_centers() {
        assert!(TableEmbedding::from_json_str(r#"{"dim":2,"vectors":{"a":[1,2,3]}}"#, false, false).is_err());
        assert!(TableEmbedding::from_json_str(r#"{"vectors":{}}"#, false, false).is_err());
        assert!(TableEmbedding::from_json_str("[1]", false, false).is_err());
        assert!(TableEmbedding::from_json_str(r#"{"vectors":{"a":["x"]}}"#, false, false).is_err());
        let t = TableEmbedding::from_json_str(r#"{"vectors":{"a":[1,0],"b":[3,0]}}"#, false, true).unwrap();
        assert_eq!(t.embed("a"), vec![-1.0, 0.0]);
        assert_eq!(t.embed("b"), vec![1.0, 0.0]);
    }

    struct Fixed;
    impl VectorSource for Fixed {
        fn dim(&self) -> usize {
            2
        }
        fn vector(&self, t: &str) -> Option<Vec<f32>> {
            (t == "k").then(|| vec![1.0, 2.0])
        }
    }

    #[test]
    fn hook_embedding_and_dyn_object() {
        let h: Box<dyn Embedding> = Box::new(HookEmbedding { src: Fixed });
        assert_eq!(h.embed("k"), vec![1.0, 2.0]);
        assert_eq!(h.embed("nope"), vec![0.0, 0.0]);
        let a: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(4));
        assert_eq!(a.dim(), 4); // Arc<dyn> usable as Embedding
    }

    /// counts encoder calls so the memo's saving is observable
    struct Counting(HashEmbedding, AtomicU64);
    impl Embedding for Counting {
        fn dim(&self) -> usize {
            self.0.dim()
        }
        fn embed(&self, t: &str) -> Vec<f32> {
            self.1.fetch_add(1, Ordering::Relaxed);
            self.0.embed(t)
        }
        fn name(&self) -> &str {
            "counting"
        }
        fn embed_task(&self, t: &Task) -> Vec<f32> {
            self.1.fetch_add(1, Ordering::Relaxed);
            self.0.embed_task(t)
        }
    }

    #[test]
    fn memo_is_bit_identical_computes_once_and_keys_are_exact() {
        let m = MemoEmbedding::new(Counting(HashEmbedding::new(32), AtomicU64::new(0)), 8);
        let plain = HashEmbedding::new(32);
        // text path
        let a = m.embed("alpha beta");
        assert!(a.iter().zip(&plain.embed("alpha beta")).all(|(x, y)| x.to_bits() == y.to_bits()));
        for _ in 0..5 {
            assert_eq!(m.embed("alpha beta"), a);
        }
        assert_eq!(m.inner().1.load(Ordering::Relaxed), 1);
        assert_eq!((m.encoder_calls(), m.hits()), (1, 5));
        // shared value is the same allocation on a hit
        let s1 = m.embed_shared("alpha beta");
        let s2 = m.embed_shared("alpha beta");
        assert!(Arc::ptr_eq(&s1, &s2));
        // task path: feature boundaries and the type are part of the key
        let t1 = task(&["ab", "c"], None);
        let t2 = task(&["a", "bc"], None);
        let t3 = task(&["ab", "c"], Some("x"));
        for t in [&t1, &t2, &t3, &t1, &t2, &t3] {
            let got = m.embed_task(t);
            let want = plain.embed_task(t);
            assert!(got.iter().zip(&want).all(|(x, y)| x.to_bits() == y.to_bits()));
        }
        assert_eq!(m.encoder_calls(), 1 + 3);
        // text and task keys never alias (embed("a b") != embed_task of a differently-featured task is irrelevant; same text, two paths = two entries)
        let _ = m.embed_task(&task(&["alpha", "beta"], None));
        assert_eq!(m.encoder_calls(), 5);
        // bounded (FIFO) and cap 0 = pass-through
        let small = MemoEmbedding::new(HashEmbedding::new(8), 2);
        small.embed("x");
        small.embed("y");
        small.embed("z");
        assert_eq!(small.len(), 2);
        let off = MemoEmbedding::new(Counting(HashEmbedding::new(8), AtomicU64::new(0)), 0);
        off.embed("q");
        off.embed("q");
        assert_eq!((off.encoder_calls(), off.hits(), off.len()), (2, 0, 0));
        // Arc<MemoEmbedding> is an Embedding (Knowledge::with_embedder takes Arc<dyn Embedding>) and the CaseRouter closure shares it
        let shared = Arc::new(MemoEmbedding::new(Counting(HashEmbedding::new(16), AtomicU64::new(0)), 4));
        let as_dyn: Arc<dyn Embedding> = shared.clone();
        let f = shared.task_embedder();
        let tk = task(&["k1", "k2"], None);
        let v1 = as_dyn.embed_task(&tk);
        assert_eq!(f(&tk).unwrap(), v1);
        assert_eq!(f(&tk).unwrap(), v1);
        assert_eq!(shared.encoder_calls(), 1);
    }

    #[cfg(not(feature = "real-embed"))]
    #[test]
    fn real_without_feature_errors_and_falls_back_to_hash() {
        let c = EmbeddingConfig::Real { source: "whatever.json".into(), dim: 8, cache: true, normalize: true, center: false };
        assert!(matches!(c.build(64), Err(EmbeddingError::FeatureDisabled)));
        let (e, w) = c.build_or_hash(64);
        assert!(w.unwrap().contains("real-embed"));
        assert_eq!(e.name(), "hash");
        assert_eq!(e.dim(), 64);
        let t = task(&["a", "b"], None);
        assert_eq!(e.embed_task(&t), hash_embed(&t, 64));
    }

    #[cfg(feature = "real-embed")]
    #[test]
    fn real_with_feature_loads_fixture() {
        let dir = std::env::temp_dir().join("nm_embed_fixture_test.json");
        std::fs::write(&dir, FIX).unwrap();
        let c = EmbeddingConfig::Real { source: dir.to_string_lossy().into(), dim: 3, cache: true, normalize: true, center: false };
        let e = c.build(64).unwrap();
        assert_eq!((e.dim(), e.name()), (3, "real(toy)"));
        assert_eq!(e.embed("cat"), vec![1.0, 0.0, 0.0]);
        // wrong dim -> clear error, fallback keeps working
        let bad = EmbeddingConfig::Real { source: dir.to_string_lossy().into(), dim: 5, cache: false, normalize: true, center: false };
        assert!(bad.build(64).is_err());
        let (fb, w) = bad.build_or_hash(64);
        assert!(w.is_some() && fb.dim() == 64);
        let _ = std::fs::remove_file(&dir);
    }
}
