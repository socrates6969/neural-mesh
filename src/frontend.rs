//! LLM-DESIGN feature 1 (opt-in, DEFAULT OFF): the mesh as a front-end that either answers a request locally or forwards it to
//! Claude with retrieved, trimmed context. This module holds only the deterministic policy; the signals come from elsewhere:
//! * `cache_hit`: an exact (normalised) earlier request with a verified answer;
//! * `tool`: [`crate::knnroute::KnnRouter::route_guarded`] (R80/R88 guard: tau .6, smin .4); `None` = no tool route;
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

pub fn decide(cfg: &FrontendCfg, s: &Signals) -> Decision {
    if !cfg.enabled {
        return Decision::Forward;
    }
    if s.cache_hit {
        return Decision::Cache;
    }
    if let Some(t) = s.tool.as_ref().map(|t| t.trim()).filter(|t| !t.is_empty()) {
        return Decision::Tool(t.to_string());
    }
    let in_dist = cfg.min_sim <= 0.0 || s.max_sim.is_some_and(|m| m.is_finite() && m >= cfg.min_sim);
    match s.local_conf {
        Some(c) if in_dist && c.is_finite() && c >= cfg.min_conf && s.local_tokens > 0 && s.local_tokens <= cfg.max_local_tokens => Decision::Local,
        _ => Decision::Forward,
    }
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
