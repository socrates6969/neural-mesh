//! Knowledge layer: retrieval-augmented routing over OUR OWN store (no model training).
//!
//! A `Knowledge` keeps one record per routed task in a vector store (the existing
//! [`SharedMemory`] / [`FlatIndex`], or the [`TieredMemory`]). Before routing, the task's
//! embedding is used to pull the top-k most similar past records; that evidence is injected
//! as context in one or both of two ways:
//!
//! * **feature**: the neighbours vote for a context *anchor* (leader clustering that
//!   emerges from retrieval itself); the router sees an extra feature `kn:<anchor>` and
//!   learns per-anchor agent weights ([`FeatMode::Direct`]: the front sees the feature too;
//!   [`FeatMode::Hidden`]: only the inner learner sees it, a confidence-gated front such as
//!   `Guarded` keeps gating on the original features).
//! * **vote**: neighbours that carry graded rewards vote for an agent; a confident vote
//!   short-cuts the inner router ([`KnowledgeConfig::vote`]).
//!
//! Learn-by-doing: [`KnowledgeRouter::update`] credits the retrieved record with the same
//! reward the inner router receives. Callers should feed the graded, delayed downstream
//! reuse signal behind a confidence-gated front (the x3 finding) as the default feedback path;
//! `Knowledge` itself is agnostic.
//!
//! Optional approved-source fetch: [`SourceFetcher`] with a stub ([`NoFetch`], never touches
//! the network) and a test double ([`FixtureFetcher`]); [`ApprovedSources`] is an allow-list
//! guard. There is NO live fetch implementation in this crate.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use crate::embedding::Embedding;
use crate::memory::{normalized, FlatIndex, SharedMemory};
use crate::router::{Rng, Router, Task};
use crate::tiered::{TieredConfig, TieredMemory};

// ---------------------------------------------------------------------------
// Embedding of text-like tasks (feature hashing; deterministic, dependency free)

/// Signed feature-hashing embedding of the task's features (+ type), L2-normalised.
pub fn hash_embed(task: &Task, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    let mut put = |s: &str| {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        let idx = (h % dim as u64) as usize;
        let sign = if (h >> 40) & 1 == 0 { 1.0 } else { -1.0 };
        v[idx] += sign;
    };
    for f in &task.features {
        put(f);
    }
    if let Some(t) = &task.task_type {
        put(&format!("type:{t}"));
    }
    normalized(&v)
}

// ---------------------------------------------------------------------------
// Store backend

/// Vector-store backend of [`Knowledge`].
pub enum Backend {
    Shared(SharedMemory),
    Tiered(TieredMemory),
}

impl Backend {
    fn add(&mut self, id: u64, v: &[f32], meta: &str) {
        match self {
            Backend::Shared(m) => m.add(id, v),
            Backend::Tiered(t) => {
                t.add(id, v, meta);
            }
        }
    }
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
        match self {
            Backend::Shared(m) => m.search(q, k),
            Backend::Tiered(t) => t.search_ro(q, k),
        }
    }
    fn credit(&mut self, id: u64, r: f64) {
        if let Backend::Tiered(t) = self {
            t.record_outcome(id, r);
        }
    }
    pub fn bytes(&self) -> usize {
        match self {
            Backend::Shared(m) => m.bytes(),
            Backend::Tiered(t) => t.bytes(),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Backend::Shared(m) => m.len(),
            Backend::Tiered(t) => t.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Config / records

/// How the retrieved anchor is injected as a feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatMode {
    /// No feature injection.
    Off,
    /// The routed task carries `kn:<anchor>` everywhere (front and inner see it).
    Direct,
    /// Only the inner learner sees it (needs [`AnchorRouter`] inside the front).
    Hidden,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VoteConfig {
    /// Minimum weighted number of rewarded neighbours backing an agent.
    pub min_n: f64,
    /// Minimum (prior-smoothed) mean reward of the agent among neighbours.
    pub tau: f64,
    /// Beta prior (a, b) of the neighbour mean.
    pub prior: (f64, f64),
    /// Satisficing exploration: when no agent reaches `tau` and the anchor has at least this many tries
    /// , try a candidate not yet used in the anchor. Stops as
    /// soon as some agent qualifies, so the cost per recurring context is bounded.
    pub explore_after: Option<usize>,
    /// Confidence gate on the vote itself (min_n decisions of each kind, margin, probe_every):
    /// once >= min_n vote decisions and >= min_n front decisions carry rewards, the vote is
    /// suppressed while its mean reward is below the front's mean by more than margin
    /// (every probe_every-th eligible vote is still taken so the gate can recover).
    pub gate: Option<(f64, f64, usize)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct KnowledgeConfig {
    pub dim: usize,
    /// Neighbours retrieved.
    pub k: usize,
    /// Minimum cosine for a neighbour to count (anchor vote and agent vote).
    pub sim_min: f32,
    pub feat: FeatMode,
    pub vote: Option<VoteConfig>,
}

impl KnowledgeConfig {
    pub fn new(dim: usize) -> Self {
        KnowledgeConfig { dim, k: 5, sim_min: 0.6, feat: FeatMode::Hidden, vote: None }
    }
}

#[derive(Clone, Debug)]
struct Rec {
    anchor: u64,
    agent: Option<usize>,
    /// reward sum / count credited to this record
    s: f64,
    n: f64,
    by_vote: bool,
}

/// Result of one retrieval (read only).
#[derive(Clone, Debug)]
pub struct Retrieval {
    /// (record id, cosine), best first, all neighbours (not thresholded).
    pub hits: Vec<(u64, f32)>,
    /// Anchor chosen by the neighbours (None: nothing similar enough).
    pub anchor: Option<u64>,
}

pub struct Knowledge {
    pub cfg: KnowledgeConfig,
    mem: Backend,
    recs: Vec<Rec>,
    /// anchor -> per-agent (tries, reward sum, rewards)
    astats: HashMap<u64, Vec<(f64, f64, f64)>>,
    /// gate statistics: (vote reward sum, vote n, front reward sum, front n)
    gate: [f64; 4],
    probe: usize,
    /// Optional configured embedder (None = legacy [`hash_embed`] at `cfg.dim`).
    embedder: Option<Arc<dyn Embedding>>,
}

impl Knowledge {
    pub fn new_shared(cfg: KnowledgeConfig) -> Self {
        let m = SharedMemory::new(Box::new(FlatIndex::new(cfg.dim)));
        Knowledge { cfg, mem: Backend::Shared(m), recs: Vec::new(), astats: HashMap::new(), gate: [0.0; 4], probe: 0, embedder: None }
    }
    pub fn new_tiered(cfg: KnowledgeConfig, tc: TieredConfig) -> Self {
        assert_eq!(cfg.dim, tc.dim, "Knowledge: tiered dim mismatch");
        Knowledge { cfg, mem: Backend::Tiered(TieredMemory::new(tc)), recs: Vec::new(), astats: HashMap::new(), gate: [0.0; 4], probe: 0, embedder: None }
    }
    /// Use `e` to embed tasks/texts (its `dim()` must equal `cfg.dim`). Without this call the
    /// legacy [`hash_embed`] is used, bit-identically to before.
    pub fn with_embedder(mut self, e: Arc<dyn Embedding>) -> Self {
        assert_eq!(e.dim(), self.cfg.dim, "Knowledge: embedder dim mismatch");
        self.embedder = Some(e);
        self
    }
    /// Embedding of a routed task (configured embedder, else `hash_embed` at `cfg.dim`).
    pub fn embed_task(&self, task: &Task) -> Vec<f32> {
        match &self.embedder {
            Some(e) => e.embed_task(task),
            None => hash_embed(task, self.cfg.dim),
        }
    }
    /// Embedding of free text (configured embedder, else hash of whitespace-split features).
    pub fn embed_text(&self, text: &str) -> Vec<f32> {
        match &self.embedder {
            Some(e) => e.embed(text),
            None => hash_embed(&Task { id: 0, task_type: None, features: text.split_whitespace().map(String::from).collect() }, self.cfg.dim),
        }
    }
    pub fn len(&self) -> usize {
        self.recs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.recs.is_empty()
    }
    pub fn bytes(&self) -> usize {
        self.mem.bytes() + self.recs.len() * std::mem::size_of::<Rec>()
    }
    pub fn backend(&self) -> &Backend {
        &self.mem
    }

    /// Top-k retrieval and anchor vote (similarity-weighted over neighbours >= `sim_min`).
    pub fn retrieve(&self, emb: &[f32]) -> Retrieval {
        if self.recs.is_empty() {
            return Retrieval { hits: vec![], anchor: None };
        }
        let hits = self.mem.search(emb, self.cfg.k);
        let mut votes: Vec<(u64, f32)> = vec![];
        for &(id, c) in &hits {
            if c < self.cfg.sim_min {
                continue;
            }
            let Some(r) = self.recs.get(id as usize) else { continue };
            match votes.iter_mut().find(|v| v.0 == r.anchor) {
                Some(v) => v.1 += c,
                None => votes.push((r.anchor, c)),
            }
        }
        let anchor = votes
            .iter()
            .fold(None, |b: Option<(u64, f32)>, &v| match b {
                Some(x) if x.1 >= v.1 => b,
                _ => Some(v),
            })
            .map(|v| v.0);
        Retrieval { hits, anchor }
    }

    /// The feature injected for this retrieval (None when nothing similar was seen).
    pub fn feature(&self, r: &Retrieval) -> Option<String> {
        r.anchor.map(|a| format!("kn:{a}"))
    }

    /// `task` plus the retrieved feature.
    pub fn enrich(&self, task: &Task, r: &Retrieval) -> Task {
        let mut t = task.clone();
        if let Some(f) = self.feature(r) {
            t.features.push(f);
        }
        t
    }

    /// Store this task as a new record (anchor inherited from the retrieval, else its own id).
    /// Returns the record id.
    pub fn remember(&mut self, emb: &[f32], r: &Retrieval) -> u64 {
        let id = self.recs.len() as u64;
        let anchor = r.anchor.unwrap_or(id);
        self.recs.push(Rec { anchor, agent: None, s: 0.0, n: 0.0, by_vote: false });
        self.mem.add(id, emb, &format!("a{anchor}"));
        id
    }

    /// Store a record that came from an external document (no agent / reward).
    pub fn ingest(&mut self, emb: &[f32]) -> u64 {
        let r = self.retrieve(emb);
        self.remember(emb, &r)
    }

    pub fn commit(&mut self, rec: u64, agent: usize) {
        let Some(r) = self.recs.get_mut(rec as usize) else { return };
        r.agent = Some(agent);
        let a = r.anchor;
        let row = self.astats.entry(a).or_default();
        if row.len() <= agent {
            row.resize(agent + 1, (0.0, 0.0, 0.0));
        }
        row[agent].0 += 1.0;
    }

    /// Learn by doing: credit a reward (clamped to [0,1]) to a record (and its anchor's
    /// per-agent statistics).
    pub fn credit(&mut self, rec: u64, reward: f64) {
        let r = if reward.is_finite() { reward.clamp(0.0, 1.0) } else { 0.0 };
        if let Some(x) = self.recs.get_mut(rec as usize) {
            x.s += r;
            x.n += 1.0;
            if x.by_vote {
                self.gate[0] += r;
                self.gate[1] += 1.0;
            } else {
                self.gate[2] += r;
                self.gate[3] += 1.0;
            }
            if let (Some(a), Some(row)) = (x.agent, self.astats.get_mut(&x.anchor)) {
                if let Some(c) = row.get_mut(a) {
                    c.1 += r;
                    c.2 += 1.0;
                }
            }
            self.mem.credit(rec, r);
        }
    }

    /// Mark a record as decided by the vote (for the gate statistics).
    pub fn mark_vote(&mut self, rec: u64) {
        if let Some(r) = self.recs.get_mut(rec as usize) {
            r.by_vote = true;
        }
    }

    /// Whether the vote may be used now (see [`VoteConfig::gate`]). Takes a probe turn when closed.
    pub fn vote_gate_open(&mut self) -> bool {
        let Some((min_n, margin, every)) = self.cfg.vote.as_ref().and_then(|v| v.gate) else { return true };
        let g = self.gate;
        if g[1] < min_n || g[3] < min_n {
            return true;
        }
        if g[0] / g[1] >= g[2] / g[3] - margin {
            return true;
        }
        self.probe += 1;
        every > 0 && self.probe % every == 0
    }

    /// Agent voted from the retrieved anchor's per-agent reward statistics, if confident:
    /// the best agent with >= `min_n` credited rewards and smoothed mean >= `tau`. If none
    /// qualifies and `explore_after` is set, try a candidate never used in this anchor
    /// (satisficing exploration, <= one try per agent per anchor).
    pub fn vote(&self, r: &Retrieval, candidates: &[usize]) -> Option<(usize, f64)> {
        let vc = self.cfg.vote.as_ref()?;
        let row = self.astats.get(&r.anchor?)?;
        let mut best: Option<(usize, f64)> = None;
        for &a in candidates {
            let Some(&(_, s, n)) = row.get(a) else { continue };
            if n < vc.min_n {
                continue;
            }
            let m = (vc.prior.0 + s) / (vc.prior.0 + vc.prior.1 + n);
            if m < vc.tau {
                continue;
            }
            best = match best {
                Some((_, bm)) if bm >= m => best,
                _ => Some((a, m)),
            };
        }
        if best.is_none() {
            if let Some(m) = vc.explore_after {
                let total: f64 = row.iter().map(|c| c.0).sum();
                if total >= m as f64 {
                    let untried: Vec<usize> = candidates.iter().copied().filter(|&a| row.get(a).map_or(true, |c| c.0 < 1.0)).collect();
                    if !untried.is_empty() {
                        return Some((untried[self.recs.len() % untried.len()], 0.0));
                    }
                }
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// Routers

/// Map task id -> anchor feature shared between [`KnowledgeRouter`] and [`AnchorRouter`].
pub type AnchorMap = Rc<RefCell<HashMap<u64, String>>>;

/// Inner-learner wrapper that adds the retrieved anchor feature (looked up by task id) before
/// delegating. Used inside a confidence-gated front so the gate keeps seeing the original
/// features ([`FeatMode::Hidden`]).
pub struct AnchorRouter<R: Router> {
    pub inner: R,
    pub anchors: AnchorMap,
}

impl<R: Router> AnchorRouter<R> {
    pub fn new(inner: R, anchors: AnchorMap) -> Self {
        AnchorRouter { inner, anchors }
    }
    fn aug(&self, task: &Task) -> Task {
        match self.anchors.borrow().get(&task.id) {
            Some(f) => {
                let mut t = task.clone();
                t.features.push(f.clone());
                t
            }
            None => task.clone(),
        }
    }
}

impl<R: Router> Router for AnchorRouter<R> {
    fn name(&self) -> String {
        format!("anchor({})", self.inner.name())
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let t = self.aug(task);
        self.inner.select(&t, candidates, rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let t = self.aug(task);
        self.inner.update(&t, agent, reward)
    }
}

/// Retrieval-before-route wrapper. Call [`KnowledgeRouter::set_context`] with the task's
/// embedding before `select`; without a context (or with everything off) it is exactly `front`.
pub struct KnowledgeRouter<R: Router> {
    pub front: R,
    pub kn: Knowledge,
    pub anchors: AnchorMap,
    ctx: Option<Vec<f32>>,
    /// task id -> (enriched task for Direct mode, record id)
    live: HashMap<u64, (Option<Task>, u64)>,
    order: VecDeque<u64>,
    cap: usize,
    /// decisions: [vote, front]
    pub decisions: [u64; 2],
}

impl<R: Router> KnowledgeRouter<R> {
    pub fn new(front: R, kn: Knowledge, anchors: AnchorMap) -> Self {
        KnowledgeRouter { front, kn, anchors, ctx: None, live: HashMap::new(), order: VecDeque::new(), cap: 512, decisions: [0; 2] }
    }
    pub fn set_context(&mut self, emb: &[f32]) {
        self.ctx = Some(emb.to_vec());
    }
    /// Convenience: embed `task` with the store's embedder and set it as the context.
    pub fn set_context_task(&mut self, task: &Task) {
        let e = self.kn.embed_task(task);
        self.ctx = Some(e);
    }
    fn forget_old(&mut self) {
        while self.order.len() > self.cap {
            if let Some(id) = self.order.pop_front() {
                self.live.remove(&id);
                self.anchors.borrow_mut().remove(&id);
            }
        }
    }
}

impl<R: Router> Router for KnowledgeRouter<R> {
    fn name(&self) -> String {
        format!("knowledge({})", self.front.name())
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let Some(emb) = self.ctx.take() else {
            return self.front.select(task, candidates, rng);
        };
        let r = self.kn.retrieve(&emb);
        let rec = self.kn.remember(&emb, &r);
        let mut routed: Option<Task> = None;
        match self.kn.cfg.feat {
            FeatMode::Off => {}
            FeatMode::Direct => routed = Some(self.kn.enrich(task, &r)),
            FeatMode::Hidden => {
                if let Some(f) = self.kn.feature(&r) {
                    self.anchors.borrow_mut().insert(task.id, f);
                }
            }
        }
        self.live.insert(task.id, (routed.clone(), rec));
        self.order.push_back(task.id);
        self.forget_old();
        let voted = match self.kn.vote(&r, candidates) {
            Some((a, _)) if self.kn.vote_gate_open() => Some(a),
            _ => None,
        };
        let agent = match voted {
            Some(a) => {
                self.decisions[0] += 1;
                self.kn.mark_vote(rec);
                a
            }
            None => {
                self.decisions[1] += 1;
                self.front.select(routed.as_ref().unwrap_or(task), candidates, rng)
            }
        };
        self.kn.commit(rec, agent);
        agent
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let (routed, rec) = match self.live.get(&task.id) {
            Some((t, r)) => (t.clone(), Some(*r)),
            None => (None, None),
        };
        if let Some(r) = rec {
            self.kn.credit(r, reward);
        }
        self.front.update(routed.as_ref().unwrap_or(task), agent, reward);
    }
}

// ---------------------------------------------------------------------------
// Approved-source fetch interface (no network code in this crate)

#[derive(Clone, Debug, PartialEq)]
pub struct FetchedDoc {
    pub source: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FetchError {
    /// Fetching is not enabled (the stub).
    Disabled,
    /// The source is not on the approved list.
    NotApproved(String),
    Other(String),
}

/// Fetch documents for a query from one approved source.
pub trait SourceFetcher {
    fn name(&self) -> String;
    fn fetch(&self, source: &str, query: &str) -> Result<Vec<FetchedDoc>, FetchError>;
}

/// Stub: always [`FetchError::Disabled`]. The default; performs no I/O.
pub struct NoFetch;

impl SourceFetcher for NoFetch {
    fn name(&self) -> String {
        "no-fetch".into()
    }
    fn fetch(&self, _source: &str, _query: &str) -> Result<Vec<FetchedDoc>, FetchError> {
        Err(FetchError::Disabled)
    }
}

/// Test double: serves canned documents per (source, query).
#[derive(Default)]
pub struct FixtureFetcher {
    docs: HashMap<(String, String), Vec<FetchedDoc>>,
}

impl FixtureFetcher {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with(mut self, source: &str, query: &str, docs: Vec<FetchedDoc>) -> Self {
        self.docs.insert((source.to_string(), query.to_string()), docs);
        self
    }
}

impl SourceFetcher for FixtureFetcher {
    fn name(&self) -> String {
        "fixture".into()
    }
    fn fetch(&self, source: &str, query: &str) -> Result<Vec<FetchedDoc>, FetchError> {
        Ok(self.docs.get(&(source.to_string(), query.to_string())).cloned().unwrap_or_default())
    }
}

/// Allow-list guard: only sources explicitly approved are forwarded to the inner fetcher.
pub struct ApprovedSources<F: SourceFetcher> {
    pub inner: F,
    allow: Vec<String>,
}

impl<F: SourceFetcher> ApprovedSources<F> {
    pub fn new(inner: F, allow: &[&str]) -> Self {
        ApprovedSources { inner, allow: allow.iter().map(|s| s.to_string()).collect() }
    }
}

impl<F: SourceFetcher> SourceFetcher for ApprovedSources<F> {
    fn name(&self) -> String {
        format!("approved({})", self.inner.name())
    }
    fn fetch(&self, source: &str, query: &str) -> Result<Vec<FetchedDoc>, FetchError> {
        if !self.allow.iter().any(|a| a == source) {
            return Err(FetchError::NotApproved(source.to_string()));
        }
        self.inner.fetch(source, query)
    }
}

/// Fetch from an approved source and ingest the documents into the store (embedding via
/// `embed`). Returns the number of ingested records.
pub fn ingest_from<F: SourceFetcher>(
    kn: &mut Knowledge,
    f: &F,
    source: &str,
    query: &str,
    embed: impl Fn(&str) -> Vec<f32>,
) -> Result<usize, FetchError> {
    let docs = f.fetch(source, query)?;
    for d in &docs {
        kn.ingest(&embed(&d.text));
    }
    Ok(docs.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::KeywordRouter;

    fn unit(dim: usize, i: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[i % dim] = 1.0;
        v
    }

    fn task(id: u64) -> Task {
        Task { id, task_type: None, features: vec!["x".into()] }
    }

    #[test]
    fn retrieval_inherits_anchor_of_recurring_context() {
        let cfg = KnowledgeConfig::new(8);
        let mut kn = Knowledge::new_shared(cfg);
        let r0 = kn.retrieve(&unit(8, 0));
        assert!(r0.anchor.is_none());
        let a = kn.remember(&unit(8, 0), &r0);
        let _b = kn.remember(&unit(8, 3), &kn.retrieve(&unit(8, 3)));
        let r = kn.retrieve(&unit(8, 0));
        assert_eq!(r.anchor, Some(a));
        assert_eq!(kn.feature(&r), Some(format!("kn:{a}")));
        let r2 = kn.retrieve(&unit(8, 3));
        assert_ne!(r2.anchor, Some(a));
    }

    #[test]
    fn off_without_context_equals_front() {
        let mut map = HashMap::new();
        map.insert("x".to_string(), 2usize);
        let kn = Knowledge::new_shared(KnowledgeConfig::new(8));
        let mut kr = KnowledgeRouter::new(KeywordRouter::new(map.clone(), 4), kn, Rc::new(RefCell::new(HashMap::new())));
        let mut plain = KeywordRouter::new(map, 4);
        let mut rng = Rng::new(1);
        let mut rng2 = Rng::new(1);
        for i in 0..20 {
            let t = task(i);
            assert_eq!(kr.select(&t, &[0, 1, 2, 3], &mut rng), plain.select(&t, &[0, 1, 2, 3], &mut rng2));
        }
        assert!(kr.kn.is_empty());
    }

    #[test]
    fn vote_prefers_rewarded_agent_and_credit_updates() {
        let mut cfg = KnowledgeConfig::new(8);
        cfg.vote = Some(VoteConfig { min_n: 1.0, tau: 0.6, prior: (1.0, 1.0), explore_after: None, gate: None });
        cfg.feat = FeatMode::Off;
        let mut kn = Knowledge::new_shared(cfg);
        let e = unit(8, 1);
        for (agent, rew) in [(2usize, 1.0), (2, 1.0), (1, 0.0)] {
            let r = kn.retrieve(&e);
            let id = kn.remember(&e, &r);
            kn.commit(id, agent);
            kn.credit(id, rew);
        }
        let r = kn.retrieve(&e);
        assert_eq!(kn.vote(&r, &[0, 1, 2]).map(|v| v.0), Some(2));
        assert_eq!(kn.vote(&r, &[0, 1]), None); // agent 2 not eligible, agent 1 is bad
    }

    #[test]
    fn tiered_backend_credits_items() {
        let tc = TieredConfig::new(8);
        let mut kn = Knowledge::new_tiered(KnowledgeConfig::new(8), tc);
        let e = unit(8, 5);
        let r = kn.retrieve(&e);
        let id = kn.remember(&e, &r);
        kn.credit(id, 0.8);
        assert_eq!(kn.len(), 1);
        if let Backend::Tiered(t) = kn.backend() {
            assert_eq!(t.item_stats(id).unwrap().rewards, 1);
        } else {
            panic!("expected tiered");
        }
    }

    #[test]
    fn fetch_stub_is_disabled_and_guard_blocks_unapproved() {
        assert_eq!(NoFetch.fetch("docs.example", "q"), Err(FetchError::Disabled));
        let fx = FixtureFetcher::new().with("docs.example", "q", vec![FetchedDoc { source: "docs.example".into(), text: "alpha beta".into() }]);
        let g = ApprovedSources::new(fx, &["docs.example"]);
        assert_eq!(g.fetch("evil.example", "q"), Err(FetchError::NotApproved("evil.example".into())));
        let mut kn = Knowledge::new_shared(KnowledgeConfig::new(16));
        let n = ingest_from(&mut kn, &g, "docs.example", "q", |s| {
            hash_embed(&Task { id: 0, task_type: None, features: s.split_whitespace().map(String::from).collect() }, 16)
        })
        .unwrap();
        assert_eq!(n, 1);
        assert_eq!(kn.len(), 1);
        assert!(ingest_from(&mut kn, &NoFetch, "docs.example", "q", |_| vec![0.0; 16]).is_err());
    }

    #[test]
    fn hash_embed_is_deterministic_unit() {
        let t = Task { id: 1, task_type: Some("a".into()), features: vec!["f1".into(), "f2".into()] };
        let a = hash_embed(&t, 32);
        assert_eq!(a, hash_embed(&t, 32));
        let n: f32 = a.iter().map(|x| x * x).sum();
        assert!((n - 1.0).abs() < 1e-4);
    }

    #[test]
    fn embedder_routing_default_is_hash_embed_and_custom_is_used() {
        use crate::embedding::{HashEmbedding, TableEmbedding};
        let t = Task { id: 1, task_type: Some("ty".into()), features: vec!["a".into(), "b".into()] };
        let plain = Knowledge::new_shared(KnowledgeConfig::new(16));
        assert_eq!(plain.embed_task(&t), hash_embed(&t, 16));
        let wrapped = Knowledge::new_shared(KnowledgeConfig::new(16)).with_embedder(Arc::new(HashEmbedding::new(16)));
        assert_eq!(wrapped.embed_task(&t), plain.embed_task(&t));
        assert_eq!(wrapped.embed_text("a b"), plain.embed_text("a b"));
        let tab = TableEmbedding::from_pairs("t", None, vec![("a b type:ty".to_string(), vec![0.0, 1.0])], false, false).unwrap();
        let custom = Knowledge::new_shared(KnowledgeConfig::new(2)).with_embedder(Arc::new(tab));
        assert_eq!(custom.embed_task(&t), vec![0.0, 1.0]);
    }
}
