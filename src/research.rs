//! Research-session scaffolding (opt-in; no network code).
//!
//! A [`ResearchSession`] wires four existing, separately verified pieces into one
//! retrieve -> route -> draft -> verify -> learn loop:
//!
//! 1. **Retrieve** (`retrieval`): before the task is routed, [`Knowledge`] (hashed task
//!    embedding over the shared vector memory) is searched; the retrieved anchor's best
//!    verified note (if any) is handed to the generator as a [`Context`]. Notes are only
//!    stored for outcomes graded >= `store_min` ("learn by doing"), and fetched documents
//!    (see [`ResearchSession::ingest`]) enter the same store.
//! 2. **Route**: the current default router - `warm(hebbian-robust)` with per-agent health
//!    benching ([`CoupledBrain`]); `guarded(hebr)` is an opt-in choice ([`RouterChoice`]).
//! 3. **Generator -> critic** (`critic`): after the draft, the VERIFIED second-opinion gate
//!    ([`SecondOpinion`] / [`HierSecondOpinion`]) decides from calibrated per-(feature, agent)
//!    estimates whether a second agent should check the draft; a flag triggers a retry by
//!    the best other agent. Every check is an extra agent call (reported in [`SessionStats`]).
//! 4. **Record** (`reuse`): graded outcomes arrive later (`delay` tasks) and are fed back
//!    through the coupling reuse path ([`CoupledBrain::feedback_delayed`]), the critic's
//!    calibration, and the knowledge store. Without `reuse` the router only gets the raw
//!    immediate hook bit (the x3 "weak feedback" convention).
//!
//! Everything is off by default: `ResearchConfig::default()` routes exactly like the bare
//! `CoupledBrain` it wraps (unit-tested).
//!
//! Fetching: [`ApprovedSources`] + [`SourceFetcher`] from `knowledge`. [`NoFetch`] is the
//! default (always `FetchError::Disabled`); [`FixtureFetcher`] serves tests; a real source
//! implements `SourceFetcher` elsewhere. This crate ships NO network code.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::coupling::{heb_robust_cfg, CoupledBrain, CouplingConfig, WarmEvidence};
use crate::embedding::{Embedding, EmbeddingConfig};
use crate::knowledge::{
    ApprovedSources, FetchError, Knowledge, KnowledgeConfig, NoFetch, SourceFetcher,
};
use crate::prior::GuardConfig;
use crate::router::{Rng, Task};
use crate::variants::{CheckPlan, HierSecondOpinion, HierSoConfig, SecondOpinion, SecondOpinionConfig};

// =============================================================================================
// Config

/// Serializable mirror of [`GuardConfig`]; default = the x1-frozen guarded(hebr) setting.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GuardParams {
    pub tau: f64,
    pub z: f64,
    pub kw_a: f64,
    pub kw_b: f64,
    pub discount: f64,
    pub challenge: bool,
}

impl Default for GuardParams {
    fn default() -> Self {
        GuardParams { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
    }
}

impl GuardParams {
    pub fn to_guard(&self) -> GuardConfig {
        GuardConfig { tau: self.tau, z: self.z, kw_a: self.kw_a, kw_b: self.kw_b, discount: self.discount, challenge: self.challenge }
    }
}

/// Which router front the session uses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum RouterChoice {
    /// `warm(hebbian-robust)` k, MeanFeature evidence - the current default.
    #[default]
    WarmHebr,
    /// `guarded(hebr)` - opt-in.
    GuardedHebr(GuardParams),
}

/// Which verified second-opinion gate drives the critic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CriticKind {
    Flat(SecondOpinionConfig),
    Hier(HierSoConfig),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResearchConfig {
    pub router: RouterChoice,
    /// Warm-start scale of the keyword prior (warm router).
    pub warm_k: f64,
    /// Per-agent health benching (mesh rule) in front of the router.
    pub health: bool,
    /// Retrieve context (and store verified notes) before every task.
    pub retrieval: bool,
    /// Generator -> critic pass through the second-opinion gate (None = off).
    pub critic: Option<CriticKind>,
    /// Feed graded outcomes to the router through the delayed reuse path
    /// (otherwise the router only sees the raw immediate hook bit).
    pub reuse: bool,
    /// Tasks between a delivery and the arrival of its graded outcome.
    pub delay: usize,
    /// A graded outcome >= this counts as a success for the critic's calibration.
    pub succ_thr: f64,
    /// Minimum grade for a result to be stored as a verified note.
    pub store_min: f64,
    /// Note strength given to ingested (fetched) documents.
    pub doc_note: f64,
    /// Embedding dimension / neighbours / minimum cosine of the knowledge store.
    pub dim: usize,
    pub k: usize,
    pub sim_min: f32,
    /// Approved fetch sources (allow-list); empty = nothing may be fetched.
    pub allow: Vec<String>,
    /// Embedder of the knowledge store. Default `Hash { dim: 0 }` = legacy hash embedding at
    /// `dim` (bit-identical to before). A non-default embedder defines the store dimension.
    pub embedding: EmbeddingConfig,
}

impl Default for ResearchConfig {
    fn default() -> Self {
        ResearchConfig {
            router: RouterChoice::WarmHebr,
            warm_k: 1.0,
            health: true,
            retrieval: false,
            critic: None,
            reuse: false,
            delay: 20,
            succ_thr: 0.5,
            store_min: 0.6,
            doc_note: 0.5,
            dim: 64,
            k: 5,
            sim_min: 0.6,
            allow: vec![],
            embedding: EmbeddingConfig::default(),
        }
    }
}

impl ResearchConfig {
    pub fn coupling(&self) -> CouplingConfig {
        let mut c = CouplingConfig { hebbian: heb_robust_cfg(), warm_k: self.warm_k, warm_evidence: WarmEvidence::MeanFeature, health: self.health, ..CouplingConfig::default() };
        if let RouterChoice::GuardedHebr(g) = &self.router {
            c.guard = Some(g.to_guard());
        }
        c
    }
    /// Allow-list fetcher around the stub: the default, performs no I/O.
    pub fn default_fetcher(&self) -> ApprovedSources<NoFetch> {
        self.guarded_fetcher(NoFetch)
    }
    /// Wrap any [`SourceFetcher`] in this config's allow-list.
    pub fn guarded_fetcher<F: SourceFetcher>(&self, inner: F) -> ApprovedSources<F> {
        let allow: Vec<&str> = self.allow.iter().map(|s| s.as_str()).collect();
        ApprovedSources::new(inner, &allow)
    }
}

// =============================================================================================
// Executor (the world / the real agents)

/// What retrieval hands the generator.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Context {
    pub anchor: Option<u64>,
    /// strength in [0,1] of the best verified note for the anchor, if any
    pub note: Option<f64>,
    pub hits: usize,
}

/// The side that actually does the work. In a real deployment these are agent calls;
/// in `e3_research` it is a seeded synthetic world.
pub trait Executor {
    type Out: Clone;
    /// Generator call: `agent` drafts an answer for `task` with `ctx`.
    fn draft(&mut self, task: &Task, agent: usize, ctx: &Context) -> Self::Out;
    /// Immediate hook bit (tool/call succeeded) of a draft.
    fn tool_ok(&mut self, out: &Self::Out) -> bool;
    /// Critic call: does `checker` flag this draft?
    fn check(&mut self, task: &Task, checker: usize, out: &Self::Out) -> bool;
    /// Graded outcome of a draft in [0,1], if one is (eventually) available.
    fn grade(&mut self, task: &Task, out: &Self::Out) -> Option<f64>;
    /// Which output to deliver after a flag. Default: always the retry.
    fn prefer_retry(&mut self, _first: &Self::Out, _retry: &Self::Out) -> bool {
        true
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionStats {
    pub tasks: u64,
    /// total agent calls (drafts + critic calls + retries)
    pub calls: u64,
    pub checks: u64,
    pub flags: u64,
    pub ctx_hits: u64,
}

#[derive(Clone, Debug)]
pub struct TaskReport<O> {
    pub agent: usize,
    pub delivered: O,
    pub delivered_agent: usize,
    pub calls: u32,
    pub checked: bool,
    pub flagged: bool,
    pub ctx_hit: bool,
}

enum Gate {
    Flat(SecondOpinion),
    Hier(HierSecondOpinion),
}

impl Gate {
    fn decide(&mut self, task: &Task, leader: usize, c: &[usize]) -> Option<CheckPlan> {
        match self {
            Gate::Flat(g) => g.decide(task, leader, c),
            Gate::Hier(g) => g.decide(task, leader, c),
        }
    }
    fn observe_attempt(&mut self, task: &Task, a: usize, ok: bool) {
        match self {
            Gate::Flat(g) => g.observe_attempt(task, a, ok),
            Gate::Hier(g) => g.observe_attempt(task, a, ok),
        }
    }
    fn observe_check(&mut self, task: &Task, c: usize, leader_failed: bool, flagged: bool) {
        match self {
            Gate::Flat(g) => g.observe_check(task, c, leader_failed, flagged),
            Gate::Hier(g) => g.observe_check(task, c, leader_failed, flagged),
        }
    }
}

struct Event {
    due: u64,
    task: Task,
    leader: usize,
    g_first: Option<f64>,
    g_final: Option<f64>,
    retry: Option<usize>,
    g_retry: Option<f64>,
    check: Option<(usize, bool)>,
    rec: Option<u64>,
    anchor: Option<u64>,
}

pub struct ResearchSession {
    pub cfg: ResearchConfig,
    pub brain: CoupledBrain,
    pub kn: Option<Knowledge>,
    notes: HashMap<u64, f64>,
    critic: Option<Gate>,
    pending: VecDeque<Event>,
    tick: u64,
    pub stats: SessionStats,
    emb: Arc<dyn Embedding>,
    /// Set when `cfg.embedding` could not be built and the hash embedder was used instead.
    pub embed_warning: Option<String>,
}

impl ResearchSession {
    pub fn new(n_agents: usize, keyword_map: HashMap<String, usize>, cfg: ResearchConfig) -> Self {
        let brain = CoupledBrain::new(n_agents, keyword_map, cfg.coupling());
        let (emb, embed_warning) = cfg.embedding.build_or_hash(cfg.dim);
        let kn = if cfg.retrieval {
            let mut kc = KnowledgeConfig::new(emb.dim());
            kc.k = cfg.k;
            kc.sim_min = cfg.sim_min;
            Some(Knowledge::new_shared(kc).with_embedder(emb.clone()))
        } else {
            None
        };
        let critic = cfg.critic.as_ref().map(|k| match k {
            CriticKind::Flat(c) => Gate::Flat(SecondOpinion::new(n_agents, c.clone())),
            CriticKind::Hier(c) => Gate::Hier(HierSecondOpinion::new(n_agents, c.clone())),
        });
        ResearchSession { cfg, brain, kn, notes: HashMap::new(), critic, pending: VecDeque::new(), tick: 0, stats: SessionStats::default(), emb, embed_warning }
    }

    pub fn note_count(&self) -> usize {
        self.notes.len()
    }

    /// Fetch from an approved source and add the documents to the knowledge store as
    /// notes of strength `doc_note`. Errors: `Disabled` (NoFetch), `NotApproved` (source not
    /// on the fetcher's allow-list), or `Other` when retrieval is off.
    pub fn ingest<F: SourceFetcher>(&mut self, f: &F, source: &str, query: &str) -> Result<usize, FetchError> {
        if self.kn.is_none() {
            return Err(FetchError::Other("retrieval disabled".into()));
        }
        let docs = f.fetch(source, query)?;
        let strength = self.cfg.doc_note;
        let emb_src = self.emb.clone();
        let kn = self.kn.as_mut().expect("checked above");
        for d in &docs {
            let emb = emb_src.embed(&d.text);
            let r = kn.retrieve(&emb);
            let rec = kn.remember(&emb, &r);
            let anchor = r.anchor.unwrap_or(rec);
            let e = self.notes.entry(anchor).or_insert(0.0);
            *e = e.max(strength);
        }
        Ok(docs.len())
    }

    /// Deliver all graded outcomes whose delay has elapsed.
    fn flush_due(&mut self) {
        while self.pending.front().is_some_and(|e| e.due <= self.tick) {
            let ev = self.pending.pop_front().expect("front checked");
            self.apply(ev);
        }
    }

    /// Deliver every outstanding graded outcome now (end of a run).
    pub fn flush_all(&mut self) {
        while let Some(ev) = self.pending.pop_front() {
            self.apply(ev);
        }
    }

    fn apply(&mut self, ev: Event) {
        let thr = self.cfg.succ_thr;
        let checked = ev.check.is_some();
        let leader_g = if checked { ev.g_first } else { ev.g_final };
        if self.cfg.reuse {
            if let Some(g) = leader_g {
                self.brain.feedback_delayed(&ev.task, ev.leader, g);
            }
        }
        if let Some(c) = self.critic.as_mut() {
            if let Some(g) = leader_g {
                c.observe_attempt(&ev.task, ev.leader, g >= thr);
                if let Some((ck, fl)) = ev.check {
                    c.observe_check(&ev.task, ck, g < thr, fl);
                }
            }
            if let (Some(ra), Some(g)) = (ev.retry, ev.g_retry) {
                c.observe_attempt(&ev.task, ra, g >= thr);
            }
        }
        if let (Some(kn), Some(rec)) = (self.kn.as_mut(), ev.rec) {
            if let Some(g) = ev.g_final {
                kn.credit(rec, g);
                if g >= self.cfg.store_min {
                    if let Some(a) = ev.anchor {
                        let n = self.notes.entry(a).or_insert(g);
                        *n = 0.7 * *n + 0.3 * g;
                    }
                }
            }
        }
    }

    /// Run one research task: retrieve -> route -> draft -> (critic) -> record.
    pub fn run_task<E: Executor>(&mut self, ex: &mut E, task: &Task, candidates: &[usize], rng: &mut Rng) -> TaskReport<E::Out> {
        self.tick += 1;
        self.flush_due();
        self.stats.tasks += 1;

        // 1. retrieve
        let mut ctx = Context::default();
        let mut rec: Option<u64> = None;
        let mut anchor: Option<u64> = None;
        if let Some(kn) = self.kn.as_mut() {
            let emb = self.emb.embed_task(task);
            let r = kn.retrieve(&emb);
            ctx.hits = r.hits.iter().filter(|h| h.1 >= self.cfg.sim_min).count();
            ctx.anchor = r.anchor;
            ctx.note = r.anchor.and_then(|a| self.notes.get(&a).copied());
            let id = kn.remember(&emb, &r);
            rec = Some(id);
            anchor = Some(r.anchor.unwrap_or(id));
        }
        let ctx_hit = ctx.note.is_some();
        if ctx_hit {
            self.stats.ctx_hits += 1;
        }

        // 2. route
        let agent = self.brain.route(task, candidates, rng);
        if let (Some(kn), Some(id)) = (self.kn.as_mut(), rec) {
            kn.commit(id, agent);
        }

        // 3. generator
        let first = ex.draft(task, agent, &ctx);
        let mut calls = 1u32;
        let ok = ex.tool_ok(&first);
        let ok_f = if ok { 1.0 } else { 0.0 };
        self.brain.feedback_split(task, agent, ok_f, if self.cfg.reuse { None } else { Some(ok_f) });

        // 4. critic
        let mut delivered = first.clone();
        let mut delivered_agent = agent;
        let (mut checked, mut flagged) = (false, false);
        let mut check: Option<(usize, bool)> = None;
        let mut retry: Option<usize> = None;
        let mut retry_out: Option<E::Out> = None;
        if candidates.len() >= 2 {
            if let Some(plan) = self.critic.as_mut().and_then(|g| g.decide(task, agent, candidates)) {
                checked = true;
                calls += 1;
                self.stats.checks += 1;
                flagged = ex.check(task, plan.checker, &first);
                check = Some((plan.checker, flagged));
                if flagged {
                    self.stats.flags += 1;
                    calls += 1;
                    let second = ex.draft(task, plan.retry, &ctx);
                    retry = Some(plan.retry);
                    if ex.prefer_retry(&first, &second) {
                        delivered = second.clone();
                        delivered_agent = plan.retry;
                    }
                    retry_out = Some(second);
                }
            }
        }
        self.stats.calls += calls as u64;

        // 5. record: graded outcomes arrive `delay` tasks later
        let g_first = ex.grade(task, &first);
        let g_final = if retry.is_some() && delivered_agent != agent { ex.grade(task, &delivered) } else { g_first };
        let g_retry = match (&retry, &retry_out) {
            (Some(_), Some(o)) => ex.grade(task, o),
            _ => None,
        };
        if self.cfg.reuse || self.critic.is_some() || self.kn.is_some() {
            self.pending.push_back(Event { due: self.tick + self.cfg.delay as u64, task: task.clone(), leader: agent, g_first, g_final, retry, g_retry, check, rec, anchor });
        }
        TaskReport { agent, delivered, delivered_agent, calls, checked, flagged, ctx_hit }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::FetchedDoc;
    use crate::knowledge::FixtureFetcher;
    use crate::router::Router;

    /// Deterministic stub world: agent a succeeds iff (task.id + a) % 3 != 0.
    struct Stub;
    impl Executor for Stub {
        type Out = (usize, f64);
        fn draft(&mut self, task: &Task, agent: usize, _c: &Context) -> (usize, f64) {
            (agent, if (task.id as usize + agent) % 3 != 0 { 1.0 } else { 0.0 })
        }
        fn tool_ok(&mut self, _o: &(usize, f64)) -> bool {
            true
        }
        fn check(&mut self, _t: &Task, _c: usize, o: &(usize, f64)) -> bool {
            o.1 < 0.5
        }
        fn grade(&mut self, _t: &Task, o: &(usize, f64)) -> Option<f64> {
            Some(o.1)
        }
    }

    fn task(i: u64) -> Task {
        Task { id: i, task_type: None, features: vec![format!("t{}", i % 4), "common".into()] }
    }

    fn kmap() -> HashMap<String, usize> {
        let mut m = HashMap::new();
        for t in 0..4 {
            m.insert(format!("t{t}"), t % 3);
        }
        m
    }

    #[test]
    fn off_equals_plain_routing() {
        let n = 4;
        let mut s = ResearchSession::new(n, kmap(), ResearchConfig::default());
        let mut bare = CoupledBrain::new(n, kmap(), ResearchConfig::default().coupling());
        let (mut r1, mut r2) = (Rng::new(5), Rng::new(5));
        let mut ex = Stub;
        for i in 0..300u64 {
            let t = task(i);
            let rep = s.run_task(&mut ex, &t, &[0, 1, 2, 3], &mut r1);
            let a = bare.select(&t, &[0, 1, 2, 3], &mut r2);
            assert_eq!(rep.agent, a, "decision {i} differs");
            bare.update(&t, a, 1.0); // raw hook bit (Stub tool_ok == true)
            assert!(!rep.checked && !rep.ctx_hit && rep.calls == 1);
            assert_eq!(rep.delivered_agent, a);
        }
        assert_eq!(s.stats.calls, 300);
        assert!(s.kn.is_none());
    }

    #[test]
    fn critic_costs_calls_and_retries_on_flag() {
        let cfg = ResearchConfig {
            critic: Some(CriticKind::Flat(SecondOpinionConfig { cost: 0.01, kappa: 0.0, ..SecondOpinionConfig::default() })),
            reuse: true,
            delay: 0,
            ..ResearchConfig::default()
        };
        let mut s = ResearchSession::new(4, kmap(), cfg);
        let mut rng = Rng::new(1);
        let mut ex = Stub;
        let mut extra = 0;
        for i in 0..200u64 {
            let rep = s.run_task(&mut ex, &task(i), &[0, 1, 2, 3], &mut rng);
            extra += rep.calls - 1;
            if rep.flagged {
                assert!(rep.checked && rep.calls == 3 && rep.delivered_agent != rep.agent);
            }
        }
        assert!(extra > 0 && s.stats.checks > 0 && s.stats.calls == 200 + extra as u64);
        // a single candidate can never be checked
        let rep = s.run_task(&mut ex, &task(999), &[2], &mut rng);
        assert!(!rep.checked && rep.calls == 1);
    }

    #[test]
    fn nofetch_errors_and_allow_list_blocks_unapproved() {
        let cfg = ResearchConfig { retrieval: true, allow: vec!["docs.example".into()], ..ResearchConfig::default() };
        let mut s = ResearchSession::new(4, kmap(), cfg.clone());
        // default fetcher: stub -> Disabled even for an approved source
        assert_eq!(s.ingest(&cfg.default_fetcher(), "docs.example", "q"), Err(FetchError::Disabled));
        let fx = FixtureFetcher::new().with("docs.example", "q", vec![FetchedDoc { source: "docs.example".into(), text: "t1 common alpha".into() }]).with("evil.example", "q", vec![FetchedDoc { source: "evil.example".into(), text: "x".into() }]);
        let g = cfg.guarded_fetcher(fx);
        assert_eq!(s.ingest(&g, "evil.example", "q"), Err(FetchError::NotApproved("evil.example".into())));
        assert_eq!(s.kn.as_ref().unwrap().len(), 0, "blocked source must add nothing");
        assert_eq!(s.ingest(&g, "docs.example", "q"), Ok(1));
        assert_eq!(s.kn.as_ref().unwrap().len(), 1);
        // retrieval off: nothing can be ingested at all
        let mut off = ResearchSession::new(4, kmap(), ResearchConfig::default());
        assert!(matches!(off.ingest(&g, "docs.example", "q"), Err(FetchError::Other(_))));
    }

    #[test]
    fn empty_allow_list_approves_nothing_and_ingested_doc_becomes_context() {
        let cfg = ResearchConfig { retrieval: true, ..ResearchConfig::default() };
        let fx = FixtureFetcher::new().with("s", "q", vec![FetchedDoc { source: "s".into(), text: "t1 common".into() }]);
        let mut s = ResearchSession::new(4, kmap(), cfg.clone());
        assert_eq!(s.ingest(&cfg.guarded_fetcher(FixtureFetcher::new()), "s", "q"), Err(FetchError::NotApproved("s".into())));
        let cfg2 = ResearchConfig { allow: vec!["s".into()], ..cfg };
        assert_eq!(s.ingest(&cfg2.guarded_fetcher(fx), "s", "q"), Ok(1));
        let mut rng = Rng::new(2);
        let rep = s.run_task(&mut Stub, &task(1), &[0, 1, 2, 3], &mut rng); // features t1 + common
        assert!(rep.ctx_hit, "ingested document must be retrievable as context");
    }

    #[test]
    fn config_default_is_serde_default_and_off() {
        let c: ResearchConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c, ResearchConfig::default());
        assert!(!c.retrieval && c.critic.is_none() && !c.reuse);
        let j = serde_json::to_string(&ResearchConfig { critic: Some(CriticKind::Hier(HierSoConfig::default())), router: RouterChoice::GuardedHebr(GuardParams::default()), ..ResearchConfig::default() }).unwrap();
        let back: ResearchConfig = serde_json::from_str(&j).unwrap();
        assert!(matches!(back.router, RouterChoice::GuardedHebr(_)) && matches!(back.critic, Some(CriticKind::Hier(_))));
    }

    #[test]
    fn verified_outcomes_become_notes_only_above_store_min() {
        let cfg = ResearchConfig { retrieval: true, delay: 0, store_min: 0.6, ..ResearchConfig::default() };
        let mut s = ResearchSession::new(4, kmap(), cfg);
        let mut rng = Rng::new(3);
        let mut ex = Stub;
        for i in 0..40u64 {
            s.run_task(&mut ex, &task(i), &[0, 1, 2, 3], &mut rng);
        }
        s.flush_all();
        assert!(s.note_count() > 0, "good outcomes stored");
        assert!(s.stats.ctx_hits > 0, "recurring topics are retrieved");
    }

    /// Default embedding config is the legacy hash path: a retrieval session with an explicit
    /// `Hash { dim: 64 }` (== the default dim) behaves identically to the default config.
    #[test]
    fn default_embedding_is_legacy_hash_and_explicit_hash_same_dim_is_identical() {
        assert!(ResearchConfig::default().embedding.is_default());
        let run = |cfg: ResearchConfig| {
            let mut s = ResearchSession::new(4, kmap(), cfg);
            assert!(s.embed_warning.is_none());
            let (mut rng, mut ex) = (Rng::new(11), Stub);
            let mut out = Vec::new();
            for i in 0..120u64 {
                let r = s.run_task(&mut ex, &task(i), &[0, 1, 2, 3], &mut rng);
                out.push((r.agent, r.ctx_hit));
            }
            s.flush_all();
            (out, s.note_count(), s.stats.ctx_hits)
        };
        let base = ResearchConfig { retrieval: true, delay: 0, ..ResearchConfig::default() };
        let a = run(base.clone());
        let b = run(ResearchConfig { embedding: EmbeddingConfig::Hash { dim: 64 }, ..base });
        assert_eq!(a, b);
        assert!(a.2 > 0);
    }

    /// A table embedder drives retrieval (store dim comes from the embedder, not `cfg.dim`).
    #[test]
    fn table_embedder_drives_the_store() {
        use crate::embedding::TableEmbedding;
        let table = TableEmbedding::from_pairs("t", None, vec![("t0 common type:x".to_string(), vec![1.0, 0.0, 0.0])], true, false).unwrap();
        let cfg = ResearchConfig { retrieval: true, delay: 0, ..ResearchConfig::default() };
        let mut s = ResearchSession::new(4, kmap(), cfg);
        // swap in a 3-d embedder the same way ResearchSession::new would build one
        s.emb = Arc::new(table);
        s.kn = Some(Knowledge::new_shared(KnowledgeConfig::new(3)).with_embedder(s.emb.clone()));
        let mut rng = Rng::new(1);
        let t = Task { id: 0, task_type: Some("x".into()), features: vec!["t0".into(), "common".into()] };
        let r1 = s.run_task(&mut Stub, &t, &[0, 1, 2, 3], &mut rng);
        let r2 = s.run_task(&mut Stub, &t, &[0, 1, 2, 3], &mut rng);
        assert!(!r1.ctx_hit);
        assert_eq!(s.kn.as_ref().unwrap().cfg.dim, 3);
        assert_eq!(s.kn.as_ref().unwrap().len(), 2);
        let _ = r2;
    }

    /// `Real` without the `real-embed` feature (or with a missing fixture) falls back to hash
    /// and records a warning; the session still works.
    #[test]
    fn real_embedding_unavailable_falls_back_with_warning() {
        let cfg = ResearchConfig {
            retrieval: true,
            embedding: EmbeddingConfig::Real { source: "no/such/fixture.json".into(), dim: 0, cache: false, normalize: true, center: false },
            ..ResearchConfig::default()
        };
        let mut s = ResearchSession::new(4, kmap(), cfg);
        assert!(s.embed_warning.is_some());
        assert_eq!(s.kn.as_ref().unwrap().cfg.dim, 64);
        let mut rng = Rng::new(1);
        s.run_task(&mut Stub, &task(1), &[0, 1, 2, 3], &mut rng);
    }
}
