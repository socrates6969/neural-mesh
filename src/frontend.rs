//! LLM-DESIGN feature 1 (opt-in, DEFAULT OFF): the mesh as a front-end that either answers a request locally or forwards it to
//! Claude with retrieved, trimmed context. This module holds only the deterministic policy; the signals come from elsewhere:
//! * `cache_hit`: a VERIFIED-cache hit as defined by `llm/frontend.py::verified_cache_hit` (all required): exact normalised match against
//!   strictly earlier rows of the same project, outcome success, judge_score >= 0.9 with a judge_model or rubric recorded, exactly one
//!   distinct answer for that key, age <= TTL (24 h), and git HEAD unchanged when the instruction touches code. This module trusts the flag;
//! * `tool`: [`tool_signal`] = [`crate::knnroute::KnnRouter::route_guarded`] (R80/R88 guard: tau .6, smin .4) + an allow-list; `None` = no tool route;
//! * `local_conf`: mean token log-prob of the local LLM's greedy answer (python `llm/frontend.py`, via Ollama `logprobs`);
//!   `local_tokens`: that answer's length.
//!
//! * `max_sim`: max MiniLM-384 cosine of the request to the local model's training set (OOD abstain, like R88's smin).
//!
//! Order: disabled -> Forward; cache -> Cache; tool -> Tool; in-distribution (max_sim >= min_sim) AND local_conf >= min_conf AND the answer
//! is short enough -> Local;
//! otherwise Forward (abstain). Forwarded requests carry [`trim_context`]: highest-scoring retrieved chunks, de-duplicated,
//! that fit a token budget. Measured scope (see llm/README.md): on replayed real traffic the threshold must be tuned on
//! earlier sessions and checked on later ones; R88 found tool routes save ~0 tokens on delegated agent work.

use std::collections::BTreeSet;

use crate::knnroute::KnnRouter;

#[derive(Clone, Debug, PartialEq)]
pub struct FrontendCfg {
    /// master switch; `false` (the default) forwards everything, i.e. today's behaviour
    pub enabled: bool,
    /// abstain threshold on the local answer's mean token log-prob (higher = more confident; tuned on earlier sessions)
    pub min_conf: f32,
    /// never answer locally with more than this many tokens (long answers are where small models drift)
    pub max_local_tokens: usize,
    /// token budget for retrieved context attached to a forwarded request
    pub ctx_budget: usize,
    /// at most this many context chunks
    pub max_chunks: usize,
    /// drop retrieved chunks scoring below this
    pub min_chunk_score: f32,
    /// OOD abstain: answer locally only if the request's max cosine (MiniLM-384) to what the local model was trained on is >= this;
    /// a missing similarity fails closed (forward) whenever this is > 0
    pub min_sim: f32,
}

impl Default for FrontendCfg {
    fn default() -> Self {
        FrontendCfg { enabled: false, min_conf: -0.15, max_local_tokens: 400, ctx_budget: 800, max_chunks: 5, min_chunk_score: 0.0, min_sim: 0.5 }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Signals {
    pub cache_hit: bool,
    pub tool: Option<String>,
    pub local_conf: Option<f32>,
    pub local_tokens: usize,
    /// max cosine of the request to the local model's training set (None = unknown)
    pub max_sim: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Cache,
    Tool(String),
    Local,
    Forward,
}

impl Decision {
    pub fn name(&self) -> &str {
        match self {
            Decision::Cache => "cache",
            Decision::Tool(_) => "tool",
            Decision::Local => "local",
            Decision::Forward => "forward",
        }
    }
    pub fn is_local(&self) -> bool {
        !matches!(self, Decision::Forward)
    }
}

/// Why [`decide_reason`] chose what it chose. The four `Forward` reasons are checked in this order: `Ood` (max_sim missing / non-finite /
/// below min_sim while min_sim > 0), `NoConf` (no finite local confidence, or an empty local answer), `LowConf` (conf < min_conf, or a
/// non-finite min_conf), `TooLong` (answer longer than max_local_tokens).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Disabled,
    Cache,
    Tool,
    Local,
    Ood,
    LowConf,
    NoConf,
    TooLong,
}

impl Reason {
    pub fn name(&self) -> &'static str {
        match self {
            Reason::Disabled => "disabled",
            Reason::Cache => "cache",
            Reason::Tool => "tool",
            Reason::Local => "local",
            Reason::Ood => "ood",
            Reason::LowConf => "low_conf",
            Reason::NoConf => "no_conf",
            Reason::TooLong => "too_long",
        }
    }
}

/// The policy plus its reason. Fails closed: any missing / NaN / inf signal forwards.
pub fn decide_reason(cfg: &FrontendCfg, s: &Signals) -> (Decision, Reason) {
    if !cfg.enabled {
        return (Decision::Forward, Reason::Disabled);
    }
    if s.cache_hit {
        return (Decision::Cache, Reason::Cache);
    }
    if let Some(t) = s.tool.as_ref().map(|t| t.trim()).filter(|t| !t.is_empty()) {
        return (Decision::Tool(t.to_string()), Reason::Tool);
    }
    let in_dist = cfg.min_sim <= 0.0 || s.max_sim.is_some_and(|m| m.is_finite() && m >= cfg.min_sim);
    if !in_dist {
        return (Decision::Forward, Reason::Ood);
    }
    let c = match s.local_conf {
        Some(c) if c.is_finite() && s.local_tokens > 0 => c,
        _ => return (Decision::Forward, Reason::NoConf),
    };
    if !cfg.min_conf.is_finite() || c < cfg.min_conf {
        return (Decision::Forward, Reason::LowConf);
    }
    if s.local_tokens > cfg.max_local_tokens {
        return (Decision::Forward, Reason::TooLong);
    }
    (Decision::Local, Reason::Local)
}

pub fn decide(cfg: &FrontendCfg, s: &Signals) -> Decision {
    decide_reason(cfg, s).0
}

/// The `tool` signal: [`KnnRouter::route_guarded`] (vote share >= `tau` AND nearest-exemplar cosine >= `smin`) restricted to `allow`
/// (an empty allow-list allows nothing). Abstains (None) on a wrong-sized, non-finite or all-zero query, an empty router, a low share, a far
/// query or a label outside the allow-list. O(exemplars x dim).
pub fn tool_signal(router: &KnnRouter, q: &[f32], tau: f32, smin: f32, allow: &[String]) -> Option<String> {
    if q.is_empty() || q.iter().any(|x| !x.is_finite()) || q.iter().all(|x| *x == 0.0) || !tau.is_finite() || !smin.is_finite() {
        return None;
    }
    let (label, _share) = router.route_guarded(q, tau, smin)?;
    allow.iter().any(|a| a == &label).then_some(label)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub id: String,
    pub score: f32,
    pub tokens: usize,
}

/// Greedy by score (ties: fewer tokens, then id), skipping non-finite / below-floor / duplicate ids and any chunk that does not
/// fit the remaining budget. Returns the chosen ids in score order and the tokens used. Deterministic.
pub fn trim_context(chunks: &[Chunk], budget: usize, max_chunks: usize, min_score: f32) -> (Vec<String>, usize) {
    let mut c: Vec<&Chunk> = chunks.iter().filter(|c| c.score.is_finite() && c.score >= min_score && c.tokens > 0).collect();
    c.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.tokens.cmp(&b.tokens)).then(a.id.cmp(&b.id)));
    let (mut seen, mut out, mut used) = (BTreeSet::new(), Vec::new(), 0usize);
    for ch in c {
        if out.len() >= max_chunks {
            break;
        }
        if seen.contains(&ch.id) || used + ch.tokens > budget {
            continue;
        }
        seen.insert(ch.id.clone());
        used += ch.tokens;
        out.push(ch.id.clone());
    }
    (out, used)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> FrontendCfg {
        FrontendCfg { enabled: true, ..FrontendCfg::default() }
    }

    #[test]
    fn default_is_off_and_forwards_everything() {
        let s = Signals { cache_hit: true, tool: Some("calc".into()), local_conf: Some(0.0), local_tokens: 5, max_sim: Some(1.0) };
        assert!(!FrontendCfg::default().enabled);
        assert_eq!(decide(&FrontendCfg::default(), &s), Decision::Forward);
    }

    #[test]
    fn precedence_cache_then_tool_then_local() {
        let mut s = Signals { cache_hit: true, tool: Some("calc".into()), local_conf: Some(0.0), local_tokens: 5, max_sim: Some(0.9) };
        assert_eq!(decide(&on(), &s), Decision::Cache);
        s.cache_hit = false;
        assert_eq!(decide(&on(), &s), Decision::Tool("calc".into()));
        s.tool = Some("  ".into());
        assert_eq!(decide(&on(), &s), Decision::Local);
        assert!(decide(&on(), &s).is_local() && decide(&on(), &s).name() == "local");
    }

    #[test]
    fn abstains_below_threshold_on_long_or_missing_answers() {
        let cfg = FrontendCfg { min_conf: -0.2, max_local_tokens: 100, min_sim: 0.0, ..on() };
        let s = |c: Option<f32>, n: usize| Signals { local_conf: c, local_tokens: n, ..Signals::default() };
        assert_eq!(decide(&cfg, &s(Some(-0.2), 50)), Decision::Local); // threshold inclusive
        assert_eq!(decide(&cfg, &s(Some(-0.21), 50)), Decision::Forward);
        assert_eq!(decide(&cfg, &s(Some(-0.1), 101)), Decision::Forward);
        assert_eq!(decide(&cfg, &s(Some(-0.1), 0)), Decision::Forward);
        assert_eq!(decide(&cfg, &s(None, 10)), Decision::Forward);
        assert_eq!(decide(&cfg, &s(Some(f32::NAN), 10)), Decision::Forward);
        assert_eq!(decide(&cfg, &s(Some(f32::INFINITY), 10)), Decision::Forward);
    }

    #[test]
    fn ood_requests_are_forwarded_and_unknown_similarity_fails_closed() {
        let cfg = FrontendCfg { min_conf: -1.0, min_sim: 0.5, ..on() };
        let s = |m: Option<f32>| Signals { local_conf: Some(-0.1), local_tokens: 10, max_sim: m, ..Signals::default() };
        assert_eq!(decide(&cfg, &s(Some(0.5))), Decision::Local);
        assert_eq!(decide(&cfg, &s(Some(0.49))), Decision::Forward);
        assert_eq!(decide(&cfg, &s(None)), Decision::Forward);
        assert_eq!(decide(&cfg, &s(Some(f32::NAN))), Decision::Forward);
        // the guard only gates the local LLM: a cache hit is still served
        assert_eq!(decide(&cfg, &Signals { cache_hit: true, ..s(Some(0.1)) }), Decision::Cache);
    }

    #[test]
    fn decide_reason_covers_every_branch() {
        let cfg = FrontendCfg { min_conf: -0.2, max_local_tokens: 100, min_sim: 0.5, ..on() };
        let base = Signals { local_conf: Some(-0.1), local_tokens: 50, max_sim: Some(0.9), ..Signals::default() };
        let r = |c: &FrontendCfg, s: &Signals| decide_reason(c, s);
        assert_eq!(r(&FrontendCfg::default(), &Signals { cache_hit: true, ..base.clone() }), (Decision::Forward, Reason::Disabled));
        assert_eq!(r(&cfg, &Signals { cache_hit: true, max_sim: None, ..base.clone() }), (Decision::Cache, Reason::Cache));
        assert_eq!(r(&cfg, &Signals { tool: Some("calc".into()), local_conf: None, ..base.clone() }), (Decision::Tool("calc".into()), Reason::Tool));
        assert_eq!(r(&cfg, &base), (Decision::Local, Reason::Local));
        assert_eq!(r(&cfg, &Signals { max_sim: Some(0.4), ..base.clone() }), (Decision::Forward, Reason::Ood));
        assert_eq!(r(&cfg, &Signals { max_sim: None, ..base.clone() }), (Decision::Forward, Reason::Ood));
        assert_eq!(r(&cfg, &Signals { max_sim: Some(f32::NAN), ..base.clone() }), (Decision::Forward, Reason::Ood));
        assert_eq!(r(&cfg, &Signals { local_conf: None, ..base.clone() }), (Decision::Forward, Reason::NoConf));
        assert_eq!(r(&cfg, &Signals { local_conf: Some(f32::NEG_INFINITY), ..base.clone() }), (Decision::Forward, Reason::NoConf));
        assert_eq!(r(&cfg, &Signals { local_tokens: 0, ..base.clone() }), (Decision::Forward, Reason::NoConf));
        assert_eq!(r(&cfg, &Signals { local_conf: Some(-0.3), ..base.clone() }), (Decision::Forward, Reason::LowConf));
        assert_eq!(r(&FrontendCfg { min_conf: f32::NAN, ..cfg.clone() }, &base), (Decision::Forward, Reason::LowConf));
        assert_eq!(r(&cfg, &Signals { local_tokens: 101, ..base.clone() }), (Decision::Forward, Reason::TooLong));
        // min_sim <= 0 switches the OOD guard off, also for a negative or a missing similarity
        let open = FrontendCfg { min_sim: 0.0, ..cfg.clone() };
        assert_eq!(r(&open, &Signals { max_sim: Some(-0.7), ..base.clone() }), (Decision::Local, Reason::Local));
        assert_eq!(r(&open, &Signals { max_sim: None, ..base.clone() }), (Decision::Local, Reason::Local));
        for s in [base.clone(), Signals { local_conf: None, ..base.clone() }, Signals { cache_hit: true, ..base.clone() }] {
            assert_eq!(decide(&cfg, &s), decide_reason(&cfg, &s).0);
        }
        let all = [Reason::Disabled, Reason::Cache, Reason::Tool, Reason::Local, Reason::Ood, Reason::LowConf, Reason::NoConf, Reason::TooLong];
        assert_eq!(all.iter().map(|x| x.name()).collect::<BTreeSet<_>>().len(), 8);
    }

    fn router() -> KnnRouter {
        let mut r = KnnRouter::new(3, 3);
        for (l, v) in [("calc", [1.0, 0.1, 0.0]), ("calc", [0.9, 0.0, 0.1]), ("units", [0.0, 1.0, 0.0]), ("units", [0.1, 0.9, 0.0]), ("sandbox", [0.0, 0.0, 1.0])] {
            assert!(r.add(l, &v));
        }
        r
    }

    #[test]
    fn tool_signal_routes_only_confident_near_allowed_labels() {
        let r = router();
        let allow: Vec<String> = ["calc", "units", "csp"].iter().map(|s| s.to_string()).collect();
        assert_eq!(tool_signal(&r, &[1.0, 0.05, 0.0], 0.6, 0.4, &allow).as_deref(), Some("calc"));
        // low vote share (a calc/units tie) abstains
        assert_eq!(tool_signal(&r, &[1.0, 1.0, 0.0], 0.8, 0.0, &allow), None);
        // a far query (best cosine < smin) abstains although some label wins the vote
        assert_eq!(tool_signal(&r, &[-1.0, -1.0, -1.0], 0.0, 0.4, &allow), None);
        // wrong dimension, NaN, all-zero query abstain
        assert_eq!(tool_signal(&r, &[1.0, 0.0], 0.6, 0.4, &allow), None);
        assert_eq!(tool_signal(&r, &[1.0, f32::NAN, 0.0], 0.6, 0.4, &allow), None);
        assert_eq!(tool_signal(&r, &[0.0, 0.0, 0.0], 0.0, 0.0, &allow), None);
        // a confident, near route to a label outside the allow-list abstains
        assert_eq!(tool_signal(&r, &[0.0, 0.05, 1.0], 0.6, 0.4, &allow), None);
        assert_eq!(tool_signal(&r, &[0.0, 0.05, 1.0], 0.6, 0.4, &["sandbox".to_string()]).as_deref(), Some("sandbox"));
        assert_eq!(tool_signal(&r, &[1.0, 0.05, 0.0], 0.6, 0.4, &[]), None);
        assert_eq!(tool_signal(&KnnRouter::new(3, 3), &[1.0, 0.0, 0.0], 0.0, 0.0, &allow), None);
    }

    #[test]
    fn trim_context_respects_budget_order_floor_and_duplicates() {
        let ch = |id: &str, score: f32, tokens: usize| Chunk { id: id.into(), score, tokens };
        let cs = vec![ch("a", 0.9, 500), ch("b", 0.8, 400), ch("c", 0.7, 200), ch("a", 0.95, 10), ch("d", f32::NAN, 1), ch("e", -1.0, 1), ch("f", 0.7, 100)];
        // a(.95,10) first; a(.9) is a duplicate id; b 400 fits (410); f(.7,100) before c(.7,200) on the token tie-break -> 510; c -> 710
        let (ids, used) = trim_context(&cs, 800, 10, 0.0);
        assert_eq!(ids, vec!["a", "b", "f", "c"]);
        assert_eq!(used, 710);
        let (ids, used) = trim_context(&cs, 800, 2, 0.0);
        assert_eq!((ids, used), (vec!["a".to_string(), "b".to_string()], 410));
        let (ids, _) = trim_context(&cs, 50, 10, 0.0);
        assert_eq!(ids, vec!["a"]);
        assert!(trim_context(&[], 100, 5, 0.0).0.is_empty());
        assert!(trim_context(&cs, 0, 5, 0.0).0.is_empty());
    }
}
