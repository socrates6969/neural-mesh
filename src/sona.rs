//! Port of ruflo's SONA optimizer routing logic (`@claude-flow/cli` v3,
//! `dist/src/memory/sona-optimizer.js`, class `SONAOptimizer`) as a [`Router`].
//!
//! What is ported, with the JS line numbers (of the installed build) it comes from:
//!
//! * pattern memory keyed by `agent:sortedKeywords` (`createPatternKey`, L527-L530),
//!   created on first outcome with confidence 0.5 (`processTrajectoryOutcome`, L196-L236);
//! * boolean-outcome confidence update: success `c += 0.1*(1-c)` capped 0.99, failure
//!   `c -= 0.15*c` floored 0.1 (L224-L233, constants L21-L28);
//! * pruning above 1000 patterns down to 800 by `confidence*recency` (L613-L630);
//! * routing priority (L275-L355): (1) native @ruvector/sona engine, (2) best learned
//!   pattern if `confidence*matchRatio >= 0.6`, (3) Q-learning router if its softmax
//!   confidence >= 0.5, (4) static keyword-category heuristic, (5) default agent;
//! * `findBestPatternMatch` (L534-L555): `score = |matched| / max(|pattern kw|, |task kw|) * confidence`,
//!   strictly-greater wins, ties go to the earliest-inserted pattern;
//! * `matchKeywordsToAgent` (L559-L589): per agent `|matched| / |category|`, strictly-greater wins;
//! * the Q-learning router (`ruvector/q-learning-router.js`): lr 0.1, terminal updates,
//!   reward +1 / -0.5 (sona L243), 8 actions, argmax with softmax confidence,
//!   prioritized replay (batch 32, buffer 1000), per-state route cache with entry
//!   invalidation on update and full invalidation every 50 updates.
//!
//! Deviations (unavoidable, all documented in the lab report):
//! * Text keyword extraction is replaced by `Task::features` (already keywords), de-duplicated,
//!   first 10 kept (L510). `KEYWORD_CATEGORIES` becomes the simulation's keyword->agent map
//!   (each agent's "category" = the keywords mapped to it).
//! * Priority 1 (native engine) is dead code in the installed build (the constructor call
//!   `new SonaEngine({mode:'real-time'})` throws a napi u32 conversion error, caught,
//!   `sonaEngine = null`), so it is not ported.
//! * The Q-learner's state key is the sorted feature set (the JS hashes substring presence of
//!   32 fixed English keywords + length buckets; for opaque sim keywords that would collapse to
//!   one state, so this is *more* discriminating than the original and generous to SONA).
//! * Wall-clock quantities (`Date.now()` recency, 300 s cache TTL, 1 day temporal decay) use a
//!   logical clock / are ignored: the simulation runs in << 1 ms of wall time per outcome.
//! * Q route names outside `0..n_agents` (the JS has 8 actions) or outside `candidates` are
//!   treated as "no usable suggestion" and fall through to the keyword heuristic.
//! * Reward in [0,1] is mapped to the JS boolean `success` by `reward >= 0.5`.
//!
//! Opt-in improvements (all OFF in `SonaConfig::default()`): see the `explore_*`,
//! `per_feature` fields. They are not part of ruflo's SONA.

use std::collections::{HashMap, HashSet};

use crate::router::{Rng, Router, Task};

const Q_ACTIONS: usize = 8;

#[derive(Clone, Debug)]
pub struct SonaConfig {
    pub conf_init: f64,
    pub conf_min: f64,
    pub conf_max: f64,
    pub conf_inc: f64,
    pub conf_dec: f64,
    /// Minimum `confidence*matchRatio` for the learned-pattern stage (JS: 0.6).
    pub threshold: f64,
    pub max_patterns: usize,
    pub max_keywords: usize,
    /// Enable the Q-learning stage (JS: enabled whenever the module loads).
    pub use_q: bool,
    /// Minimum Q-softmax confidence (JS: 0.5).
    pub q_threshold: f64,
    // ---------- opt-in improvements (not in ruflo) ----------
    /// With this probability pick a uniformly random candidate.
    pub explore_eps: f64,
    /// When no confident pattern exists, with this probability try a uniformly random
    /// candidate that is still unproven (no pattern, or confidence below `threshold`).
    pub explore_untried: f64,
    /// Pattern granularity: one pattern per (keyword, agent) instead of per exact keyword set;
    /// the agent score is `sum_kw conf(kw,agent) / |task keywords|`.
    pub per_feature: bool,
}

impl Default for SonaConfig {
    fn default() -> Self {
        SonaConfig {
            conf_init: 0.5,
            conf_min: 0.1,
            conf_max: 0.99,
            conf_inc: 0.1,
            conf_dec: 0.15,
            threshold: 0.6,
            max_patterns: 1000,
            max_keywords: 10,
            use_q: true,
            q_threshold: 0.5,
            explore_eps: 0.0,
            explore_untried: 0.0,
            per_feature: false,
        }
    }
}

#[derive(Clone, Debug)]
struct Pattern {
    keywords: Vec<String>,
    agent: usize,
    conf: f64,
    last_used: u64,
}

#[derive(Clone, Debug)]
struct Experience {
    state: String,
    action: usize,
    reward: f64,
    priority: f64,
}

/// Port of `QLearningRouter` restricted to what SONA uses (`route(task,false)` / `update(task,agent,reward)`).
#[derive(Clone, Debug)]
struct QPort {
    table: HashMap<String, [f64; Q_ACTIONS]>,
    lr: f64,
    buf: Vec<Experience>,
    buf_idx: usize,
    buf_cap: usize,
    batch: usize,
    cache: HashMap<String, (usize, f64)>,
    cache_order: Vec<String>,
    cache_cap: usize,
    updates: u64,
    rng: Rng,
}

impl QPort {
    fn new() -> Self {
        QPort {
            table: HashMap::new(),
            lr: 0.1,
            buf: vec![],
            buf_idx: 0,
            buf_cap: 1000,
            batch: 32,
            cache: HashMap::new(),
            cache_order: vec![],
            cache_cap: 256,
            updates: 0,
            rng: Rng::new(0x50_4A),
        }
    }

    fn qvals(&self, s: &str) -> [f64; Q_ACTIONS] {
        self.table.get(s).copied().unwrap_or([0.0; Q_ACTIONS])
    }

    /// (action, softmax confidence); exploit-only with LRU route cache.
    fn route(&mut self, s: &str) -> (usize, f64) {
        if let Some(&d) = self.cache.get(s) {
            self.cache_order.retain(|k| k != s);
            self.cache_order.push(s.to_string());
            return d;
        }
        let q = self.qvals(s);
        let mut a = 0;
        let mut m = q[0];
        for (i, &v) in q.iter().enumerate().skip(1) {
            if v > m {
                m = v;
                a = i;
            }
        }
        let max = q.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = q.iter().map(|v| (v - max).exp()).sum();
        let conf = (q[a] - max).exp() / sum;
        while self.cache.len() >= self.cache_cap && !self.cache_order.is_empty() {
            let o = self.cache_order.remove(0);
            self.cache.remove(&o);
        }
        self.cache.insert(s.to_string(), (a, conf));
        self.cache_order.push(s.to_string());
        (a, conf)
    }

    fn q_update(&mut self, s: &str, a: usize, r: f64) -> f64 {
        let e = self.table.entry(s.to_string()).or_insert([0.0; Q_ACTIONS]);
        let td = r - e[a];
        e[a] += self.lr * td;
        td
    }

    fn update(&mut self, s: &str, a: usize, r: f64) {
        let exp = Experience { state: s.to_string(), action: a, reward: r, priority: r.abs() + 0.1 };
        if self.buf.len() < self.buf_cap {
            self.buf.push(exp);
        } else {
            self.buf[self.buf_idx] = exp;
        }
        self.buf_idx = (self.buf_idx + 1) % self.buf_cap;
        self.q_update(s, a, r);
        if self.cache.remove(s).is_some() {
            self.cache_order.retain(|k| k != s);
        }
        if self.buf.len() >= self.batch {
            self.replay();
        }
        self.updates += 1;
        if self.updates % 50 == 0 {
            self.cache.clear();
            self.cache_order.clear();
        }
    }

    fn replay(&mut self) {
        let total: f64 = self.buf.iter().map(|e| e.priority).sum();
        let mut picked: Vec<usize> = vec![];
        let mut sel: HashSet<usize> = HashSet::new();
        while picked.len() < self.batch && sel.len() < self.buf.len() {
            let thr = self.rng.f64() * total;
            let mut cum = 0.0;
            for i in 0..self.buf.len() {
                if sel.contains(&i) {
                    continue;
                }
                cum += self.buf[i].priority;
                if cum >= thr {
                    picked.push(i);
                    sel.insert(i);
                    break;
                }
            }
        }
        for i in picked {
            let (s, a, r) = (self.buf[i].state.clone(), self.buf[i].action, self.buf[i].reward);
            let td = self.q_update(&s, a, r);
            self.buf[i].priority = td.abs() + 0.01;
        }
    }
}

/// SONA as a [`Router`]. `kwmap` is the static keyword->agent knowledge (SONA's `KEYWORD_CATEGORIES`).
#[derive(Clone, Debug)]
pub struct SonaRouter {
    pub cfg: SonaConfig,
    n_agents: usize,
    /// per-agent keyword category (sorted for determinism not needed: membership only)
    cats: Vec<HashSet<String>>,
    patterns: Vec<Pattern>,
    index: HashMap<String, usize>,
    q: QPort,
    clock: u64,
    /// Selection sources: [pattern, q-learning, keyword, default, explore-eps, explore-untried]
    pub src: [u64; 6],
}

impl SonaRouter {
    pub fn new(n_agents: usize, kwmap: &HashMap<String, usize>, cfg: SonaConfig) -> Self {
        let mut cats = vec![HashSet::new(); n_agents];
        for (k, &a) in kwmap {
            if a < n_agents {
                cats[a].insert(k.clone());
            }
        }
        SonaRouter { cfg, n_agents, cats, patterns: vec![], index: HashMap::new(), q: QPort::new(), clock: 0, src: [0; 6] }
    }

    pub fn n_patterns(&self) -> usize {
        self.patterns.len()
    }

    /// JS `extractKeywords` analogue: de-duplicated features, first `max_keywords` kept.
    fn keywords(&self, task: &Task) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut v = vec![];
        for f in task.features.iter().chain(task.task_type.iter()) {
            if seen.insert(f.clone()) {
                v.push(f.clone());
            }
        }
        v.truncate(self.cfg.max_keywords);
        v
    }

    fn pattern_key(agent: usize, kws: &[String]) -> String {
        let mut s: Vec<&str> = kws.iter().map(|k| k.as_str()).collect();
        s.sort_unstable();
        format!("{}:{}", agent, s.join("+"))
    }

    fn q_state(kws: &[String]) -> String {
        let mut s: Vec<&str> = kws.iter().map(|k| k.as_str()).collect();
        s.sort_unstable();
        s.join("+")
    }

    /// Best learned pattern for `kws` among `candidates`: (agent, confidence*matchRatio).
    fn best_pattern(&self, kws: &[String], candidates: &[usize]) -> Option<(usize, f64)> {
        if kws.is_empty() || self.patterns.is_empty() {
            return None;
        }
        if self.cfg.per_feature {
            let mut best: Option<(usize, f64)> = None;
            for &a in candidates {
                let mut sum = 0.0;
                let mut any = false;
                for k in kws {
                    if let Some(&i) = self.index.get(&Self::pattern_key(a, std::slice::from_ref(k))) {
                        sum += self.patterns[i].conf;
                        any = true;
                    }
                }
                if !any {
                    continue;
                }
                let s = sum / kws.len() as f64;
                if best.map_or(true, |(_, b)| s > b) {
                    best = Some((a, s));
                }
            }
            return best;
        }
        let mut best_score = 0.0;
        let mut best: Option<(usize, f64)> = None;
        for p in &self.patterns {
            if !candidates.contains(&p.agent) {
                continue;
            }
            let matched = p.keywords.iter().filter(|k| kws.contains(k)).count();
            let ratio = matched as f64 / p.keywords.len().max(kws.len()) as f64;
            let score = ratio * p.conf;
            if score > best_score && matched >= 1 {
                best_score = score;
                best = Some((p.agent, p.conf * ratio));
            }
        }
        best
    }

    /// JS `matchKeywordsToAgent`.
    fn keyword_agent(&self, kws: &[String], candidates: &[usize]) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_score = 0.0;
        for a in 0..self.n_agents {
            if !candidates.contains(&a) || self.cats[a].is_empty() {
                continue;
            }
            let m = kws.iter().filter(|k| self.cats[a].contains(*k)).count();
            if m > 0 {
                let s = m as f64 / self.cats[a].len() as f64;
                if s > best_score {
                    best_score = s;
                    best = Some(a);
                }
            }
        }
        best
    }

    /// Agent has no pattern for this task, or its confidence is below the threshold.
    fn is_unproven(&self, a: usize, kws: &[String]) -> bool {
        let conf = if self.cfg.per_feature {
            let v: Vec<f64> = kws
                .iter()
                .filter_map(|k| self.index.get(&Self::pattern_key(a, std::slice::from_ref(k))).map(|&i| self.patterns[i].conf))
                .collect();
            if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / kws.len() as f64) }
        } else {
            self.index.get(&Self::pattern_key(a, kws)).map(|&i| self.patterns[i].conf)
        };
        conf.map_or(true, |c| c < self.cfg.threshold)
    }

    fn touch(&mut self, key: String, kws: Vec<String>, agent: usize, success: bool) {
        self.clock += 1;
        let c = &self.cfg;
        let idx = match self.index.get(&key) {
            Some(&i) => i,
            None => {
                self.patterns.push(Pattern { keywords: kws, agent, conf: c.conf_init, last_used: self.clock });
                let i = self.patterns.len() - 1;
                self.index.insert(key, i);
                i
            }
        };
        let p = &mut self.patterns[idx];
        if success {
            p.conf = c.conf_max.min(p.conf + c.conf_inc * (1.0 - p.conf));
        } else {
            p.conf = c.conf_min.max(p.conf - c.conf_dec * p.conf);
        }
        p.last_used = self.clock;
    }

    /// JS `prunePatterns`: when over the cap keep the best 80% by confidence (recency ties by logical age).
    fn prune(&mut self) {
        if self.patterns.len() <= self.cfg.max_patterns {
            return;
        }
        let mut order: Vec<usize> = (0..self.patterns.len()).collect();
        order.sort_by(|&a, &b| {
            let (pa, pb) = (&self.patterns[a], &self.patterns[b]);
            pa.conf.partial_cmp(&pb.conf).unwrap().then(pa.last_used.cmp(&pb.last_used))
        });
        let keep_n = (self.cfg.max_patterns as f64 * 0.8).floor() as usize;
        let remove: HashSet<usize> = order.iter().take(self.patterns.len() - keep_n).cloned().collect();
        let old = std::mem::take(&mut self.patterns);
        self.index.clear();
        for (i, p) in old.into_iter().enumerate() {
            if !remove.contains(&i) {
                let key = Self::pattern_key(p.agent, &p.keywords);
                self.index.insert(key, self.patterns.len());
                self.patterns.push(p);
            }
        }
    }
}

impl Router for SonaRouter {
    fn name(&self) -> String {
        format!("sona(thr={},q={},eps={},untried={},feat={})", self.cfg.threshold, self.cfg.use_q, self.cfg.explore_eps, self.cfg.explore_untried, self.cfg.per_feature)
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "SonaRouter::select: no candidates");
        if self.cfg.explore_eps > 0.0 && rng.f64() < self.cfg.explore_eps {
            self.src[4] += 1;
            return candidates[rng.below(candidates.len())];
        }
        let kws = self.keywords(task);
        // Priority 2: learned pattern
        if let Some((a, conf)) = self.best_pattern(&kws, candidates)
            && conf >= self.cfg.threshold
        {
            self.src[0] += 1;
            return a;
        }
        // opt-in: try an agent we have no pattern for
        if self.cfg.explore_untried > 0.0 && rng.f64() < self.cfg.explore_untried {
            let untried: Vec<usize> = candidates.iter().cloned().filter(|&a| self.is_unproven(a, &kws)).collect();
            if !untried.is_empty() {
                self.src[5] += 1;
                return untried[rng.below(untried.len())];
            }
        }
        // Priority 3: Q-learning
        if self.cfg.use_q {
            let (a, conf) = self.q.route(&Self::q_state(&kws));
            if conf >= self.cfg.q_threshold && candidates.contains(&a) {
                self.src[1] += 1;
                return a;
            }
        }
        // Priority 4: keyword heuristic
        if let Some(a) = self.keyword_agent(&kws, candidates) {
            self.src[2] += 1;
            return a;
        }
        // Priority 5: default ('coder' = first agent)
        self.src[3] += 1;
        if candidates.contains(&0) { 0 } else { candidates[0] }
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let kws = self.keywords(task);
        if kws.is_empty() {
            return; // JS: learned=false, nothing stored
        }
        let success = reward >= 0.5;
        if self.cfg.per_feature {
            for k in &kws {
                let one = vec![k.clone()];
                self.touch(Self::pattern_key(agent, &one), one, agent, success);
            }
        } else {
            self.touch(Self::pattern_key(agent, &kws), kws.clone(), agent, success);
        }
        self.prune();
        if self.cfg.use_q && agent < self.n_agents {
            self.q.update(&Self::q_state(&kws), agent, if success { 1.0 } else { -0.5 });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(f: &[&str]) -> Task {
        Task { id: 0, task_type: None, features: f.iter().map(|s| s.to_string()).collect() }
    }
    fn map() -> HashMap<String, usize> {
        [("sql", 1), ("db", 1), ("css", 2), ("api", 0)].iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn cold_start_follows_keyword_heuristic() {
        let mut r = SonaRouter::new(3, &map(), SonaConfig::default());
        let mut rng = Rng::new(1);
        assert_eq!(r.select(&t(&["sql", "db"]), &[0, 1, 2], &mut rng), 1);
        assert_eq!(r.select(&t(&["css"]), &[0, 1, 2], &mut rng), 2);
        // nothing known -> default agent 0
        assert_eq!(r.select(&t(&["zzz"]), &[0, 1, 2], &mut rng), 0);
    }

    #[test]
    fn confidence_dynamics_match_js_constants() {
        let mut r = SonaRouter::new(3, &map(), SonaConfig { use_q: false, ..SonaConfig::default() });
        let task = t(&["zzz"]);
        r.update(&task, 2, 1.0); // 0.5 -> 0.55
        assert!((r.patterns[0].conf - 0.55).abs() < 1e-12);
        r.update(&task, 2, 0.0); // 0.55 -> 0.4675
        assert!((r.patterns[0].conf - 0.4675).abs() < 1e-12);
        for _ in 0..200 {
            r.update(&task, 2, 1.0);
        }
        assert!(r.patterns[0].conf <= 0.99 + 1e-12 && r.patterns[0].conf > 0.98);
    }

    #[test]
    fn locks_in_after_enough_successes() {
        // confidence must reach 0.6: 0.5 -> .55 -> .595 -> .6355 (3 successes)
        let mut r = SonaRouter::new(3, &map(), SonaConfig { use_q: false, ..SonaConfig::default() });
        let task = t(&["zzz"]);
        let mut rng = Rng::new(1);
        for i in 0..3 {
            assert_eq!(r.select(&task, &[0, 1, 2], &mut rng), 0, "default until learned (i={i})");
            r.update(&task, 2, 1.0);
            if i < 2 {
                assert!(r.best_pattern(&["zzz".to_string()], &[0, 1, 2]).unwrap().1 < 0.6);
            }
        }
        assert_eq!(r.select(&task, &[0, 1, 2], &mut rng), 2);
    }

    #[test]
    fn never_explores_by_default() {
        // keyword says agent 1 (bad); SONA never tries agent 2 (good).
        let mut r = SonaRouter::new(3, &map(), SonaConfig::default());
        let task = t(&["sql"]);
        let mut rng = Rng::new(5);
        let p = [0.2, 0.3, 0.95];
        let mut picked_good = 0;
        for _ in 0..500 {
            let a = r.select(&task, &[0, 1, 2], &mut rng);
            if a == 2 {
                picked_good += 1;
            }
            r.update(&task, a, if rng.bernoulli(p[a]) { 1.0 } else { 0.0 });
        }
        assert_eq!(picked_good, 0);
    }

    #[test]
    fn explore_untried_escapes_the_trap() {
        let mut r = SonaRouter::new(3, &map(), SonaConfig { explore_untried: 0.3, ..SonaConfig::default() });
        let task = t(&["sql"]);
        let mut rng = Rng::new(5);
        let p = [0.2, 0.3, 0.95];
        let mut late_good = 0;
        for i in 0..1500 {
            let a = r.select(&task, &[0, 1, 2], &mut rng);
            if i >= 1000 && a == 2 {
                late_good += 1;
            }
            r.update(&task, a, if rng.bernoulli(p[a]) { 1.0 } else { 0.0 });
        }
        assert!(late_good > 350, "late good picks {late_good}/500");
    }

    #[test]
    fn prune_caps_pattern_count() {
        let mut r = SonaRouter::new(3, &map(), SonaConfig { use_q: false, max_patterns: 50, ..SonaConfig::default() });
        for i in 0..200 {
            let k = format!("kw{i}");
            r.update(&t(&[k.as_str()]), (i % 3) as usize, 1.0);
        }
        assert!(r.n_patterns() <= 50);
        assert_eq!(r.index.len(), r.n_patterns());
    }
}

// =====================================================================================
// SonaPlus ("SONA++"): opt-in rebuild of the SONA idea on verified building blocks.
// `SonaRouter` above is unchanged. SonaPlus combines
//   * keyword prior + confidence-gated hand-over (`prior::Guarded`),
//   * exploration: (a) while the keyword rule is followed, with small decaying probability
//     try a non-keyword agent (uniform or UCB-chosen) so a challenger can earn evidence,
//     (b) exploration inside the learner (epsilon / softmax),
//   * a graded feedback path (`update_weak` / `update_delayed`) for worlds where the hook
//     only reports "tool ok" and graded quality arrives rarely and late,
//   * SONA's pattern memory as an optional learner (`PlusLearner::Patterns`): graded confidence
//     per (keyword, agent) and per exact keyword set, instead of boolean +0.1/-0.15 steps.
// SONA's Q-learner and boolean threshold stage are deliberately NOT carried over.

use crate::hebbian::{HebbianConfig, HebbianRouter};
use crate::prior::{GuardConfig, Guarded};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PlusLearner {
    /// hebbian-robust (eta .01, temperature .002, decay .01), as in `guarded(hebr)`.
    #[default]
    Hebbian,
    /// SONA-style pattern memory with graded confidence updates.
    Patterns,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PlusExplore {
    #[default]
    Uniform,
    /// non-keyword candidate with the highest Beta(1+s,1+f) upper bound on the pooled counts.
    Ucb,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SonaPlusConfig {
    // ---- guard (prior::GuardConfig, flattened) ----
    pub tau: f64,
    pub z: f64,
    pub kw_a: f64,
    pub kw_b: f64,
    pub discount: f64,
    pub challenge: bool,
    // ---- learner used after hand-over ----
    pub learner: PlusLearner,
    pub heb_eta: f64,
    pub heb_epsilon: f64,
    pub heb_temperature: f64,
    pub pat_init: f64,
    pub pat_lr: f64,
    pub pat_temp: f64,
    pub pat_eps: f64,
    /// weight of the exact-keyword-set pattern vs. the per-keyword patterns (0 = per-keyword only)
    pub pat_set_weight: f64,
    // ---- exploration while the keyword rule is followed ----
    pub explore_kw: f64,
    /// eps = explore_kw * k/(k+n) with n = keyword agent's pooled evidence (0 = constant eps)
    pub explore_decay_k: f64,
    pub explore_mode: PlusExplore,
    // ---- weak-feedback path ----
    /// how many times a delayed graded signal is applied (sparse signal amplification)
    pub delayed_mult: usize,
    /// reward sent on a raw tool failure by `update_weak`
    pub fail_reward: f64,
}

impl Default for SonaPlusConfig {
    /// Defaults == frozen x1 `guarded(hebbian-robust)`; exploration and weak-path tweaks all off.
    fn default() -> Self {
        SonaPlusConfig {
            tau: 0.55,
            z: 0.5,
            kw_a: 2.0,
            kw_b: 1.0,
            discount: 0.97,
            challenge: true,
            learner: PlusLearner::Hebbian,
            heb_eta: 0.01,
            heb_epsilon: 0.0,
            heb_temperature: 0.002,
            pat_init: 0.7,
            pat_lr: 0.15,
            pat_temp: 0.05,
            pat_eps: 0.0,
            pat_set_weight: 0.0,
            explore_kw: 0.0,
            explore_decay_k: 0.0,
            explore_mode: PlusExplore::Uniform,
            delayed_mult: 1,
            fail_reward: 0.0,
        }
    }
}

impl SonaPlusConfig {
    pub fn guard(&self) -> GuardConfig {
        GuardConfig { tau: self.tau, z: self.z, kw_a: self.kw_a, kw_b: self.kw_b, discount: self.discount, challenge: self.challenge }
    }
}

/// SONA pattern memory with graded confidence (the learner used after hand-over).
#[derive(Clone, Debug)]
pub struct PatternLearner {
    n_agents: usize,
    init: f64,
    lr: f64,
    temp: f64,
    eps: f64,
    set_weight: f64,
    feat: HashMap<String, Vec<(f64, f64)>>,
    sets: HashMap<String, Vec<(f64, f64)>>,
}

impl PatternLearner {
    pub fn new(n_agents: usize, c: &SonaPlusConfig) -> Self {
        PatternLearner {
            n_agents,
            init: c.pat_init,
            lr: c.pat_lr,
            temp: c.pat_temp,
            eps: c.pat_eps,
            set_weight: c.pat_set_weight,
            feat: HashMap::new(),
            sets: HashMap::new(),
        }
    }

    fn set_key(task: &Task) -> String {
        let mut f: Vec<&str> = task.features.iter().map(|s| s.as_str()).collect();
        f.sort_unstable();
        f.dedup();
        f.join("+")
    }

    /// Confidence per agent for `task`.
    pub fn scores(&self, task: &Task) -> Vec<f64> {
        let n = self.n_agents;
        let mut out = vec![self.init; n];
        if !task.features.is_empty() {
            let mut sum = vec![0.0; n];
            for k in &task.features {
                for (a, s) in sum.iter_mut().enumerate() {
                    *s += self.feat.get(k).and_then(|r| r.get(a)).map_or(self.init, |c| c.0);
                }
            }
            for a in 0..n {
                out[a] = sum[a] / task.features.len() as f64;
            }
        }
        if self.set_weight > 0.0 {
            if let Some(row) = self.sets.get(&Self::set_key(task)) {
                for a in 0..n.min(row.len()) {
                    let w = self.set_weight * row[a].1 / (row[a].1 + 3.0);
                    out[a] = (1.0 - w) * out[a] + w * row[a].0;
                }
            }
        }
        out
    }
}

impl Router for PatternLearner {
    fn name(&self) -> String {
        format!("patterns(init={},lr={},T={},eps={})", self.init, self.lr, self.temp, self.eps)
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "PatternLearner::select: no candidates");
        if self.eps > 0.0 && rng.f64() < self.eps {
            return candidates[rng.below(candidates.len())];
        }
        let mut s = self.scores(task);
        if s.len() <= *candidates.iter().max().unwrap() {
            s.resize(*candidates.iter().max().unwrap() + 1, self.init);
        }
        if self.temp <= 0.0 {
            let mut b = candidates[0];
            for &c in candidates {
                if s[c] > s[b] {
                    b = c;
                }
            }
            return b;
        }
        let m = candidates.iter().map(|&c| s[c]).fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = candidates.iter().map(|&c| ((s[c] - m) / self.temp).exp()).collect();
        let tot: f64 = w.iter().sum();
        let mut u = rng.f64() * tot;
        for (i, &x) in w.iter().enumerate() {
            u -= x;
            if u <= 0.0 {
                return candidates[i];
            }
        }
        *candidates.last().unwrap()
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        let n = self.n_agents.max(agent + 1);
        self.n_agents = n;
        let (init, lr) = (self.init, self.lr);
        let bump = |row: &mut Vec<(f64, f64)>| {
            if row.len() < n {
                row.resize(n, (init, 0.0));
            }
            let c = &mut row[agent];
            c.0 += lr * (r - c.0);
            c.1 += 1.0;
        };
        for k in &task.features {
            bump(self.feat.entry(k.clone()).or_insert_with(|| vec![(init, 0.0); n]));
        }
        if self.set_weight > 0.0 {
            bump(self.sets.entry(Self::set_key(task)).or_insert_with(|| vec![(init, 0.0); n]));
        }
    }
}

/// Learner after hand-over: hebbian-robust or SONA pattern memory.
#[derive(Clone, Debug)]
pub enum PlusInner {
    Heb(HebbianRouter),
    Pat(PatternLearner),
}

impl Router for PlusInner {
    fn name(&self) -> String {
        match self {
            PlusInner::Heb(h) => h.name(),
            PlusInner::Pat(p) => p.name(),
        }
    }
    fn select(&mut self, task: &Task, c: &[usize], rng: &mut Rng) -> usize {
        match self {
            PlusInner::Heb(h) => h.select(task, c, rng),
            PlusInner::Pat(p) => p.select(task, c, rng),
        }
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        match self {
            PlusInner::Heb(h) => h.update(task, agent, reward),
            PlusInner::Pat(p) => p.update(task, agent, reward),
        }
    }
}

/// SONA++ router (opt-in). `Router::update` takes an ordinary graded reward; in
/// weak-feedback worlds use `update_weak` (raw tool bit) and `update_delayed` (graded
/// downstream signal that arrives later).
pub struct SonaPlus {
    pub cfg: SonaPlusConfig,
    g: Guarded<PlusInner>,
    /// phases taken: [keyword followed, challenger, learner, exploration]
    pub src: [u64; 4],
}

impl SonaPlus {
    pub fn new(n_agents: usize, kwmap: &HashMap<String, usize>, cfg: SonaPlusConfig) -> Self {
        let inner = match cfg.learner {
            PlusLearner::Hebbian => PlusInner::Heb(HebbianRouter::new(
                n_agents,
                HebbianConfig { eta: cfg.heb_eta, temperature: cfg.heb_temperature, decay: 0.01, epsilon: cfg.heb_epsilon, ..HebbianConfig::default() },
            )),
            PlusLearner::Patterns => PlusInner::Pat(PatternLearner::new(n_agents, &cfg)),
        };
        let g = Guarded::new(inner, kwmap.clone(), n_agents, cfg.guard());
        SonaPlus { cfg, g, src: [0; 4] }
    }

    /// Hook reported only tool ok / failed: a failure is evidence (reward `fail_reward`),
    /// a success carries no information about quality and is NOT sent to the learner
    /// (feeding a constant 1 makes every learner copy its own prior).
    pub fn update_weak(&mut self, task: &Task, agent: usize, raw_ok: bool) {
        if !raw_ok {
            let r = self.cfg.fail_reward;
            self.update(task, agent, r);
        }
    }

    /// Graded signal for an earlier routing (e.g. downstream reuse).
    pub fn update_delayed(&mut self, task: &Task, agent: usize, value: f64) {
        for _ in 0..self.cfg.delayed_mult.max(1) {
            self.update(task, agent, value);
        }
    }

    /// Guard decisions so far: (keyword followed, challenger chosen, learner).
    pub fn decisions(&self) -> [u64; 3] {
        self.g.decisions
    }
}

impl Router for SonaPlus {
    fn name(&self) -> String {
        format!("sona+(tau={},{:?},xkw={},{:?})", self.cfg.tau, self.cfg.learner, self.cfg.explore_kw, self.cfg.explore_mode)
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "SonaPlus::select: no candidates");
        let before = self.g.decisions;
        let a = self.g.select(task, candidates, rng);
        let phase = if self.g.decisions[0] > before[0] {
            0
        } else if self.g.decisions[1] > before[1] {
            1
        } else {
            2
        };
        self.src[phase] += 1;
        if phase == 0 && self.cfg.explore_kw > 0.0 && candidates.len() > 1 {
            let (s, f) = self.g.pooled(task, a);
            let k = self.cfg.explore_decay_k;
            let eps = if k > 0.0 { self.cfg.explore_kw * k / (k + s + f) } else { self.cfg.explore_kw };
            if rng.f64() < eps {
                let others: Vec<usize> = candidates.iter().cloned().filter(|&c| c != a).collect();
                let pick = match self.cfg.explore_mode {
                    PlusExplore::Uniform => others[rng.below(others.len())],
                    PlusExplore::Ucb => {
                        let mut best = others[0];
                        let mut bu = f64::NEG_INFINITY;
                        for &c in &others {
                            let (cs, cf) = self.g.pooled(task, c);
                            let (sa, sb) = (1.0 + cs, 1.0 + cf);
                            let n = sa + sb;
                            let m = sa / n;
                            let u = m + self.cfg.z.max(1.0) * (m * (1.0 - m) / (n + 1.0)).sqrt();
                            if u > bu {
                                bu = u;
                                best = c;
                            }
                        }
                        best
                    }
                };
                self.src[3] += 1;
                return pick;
            }
        }
        a
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.g.update(task, agent, reward);
    }
}

#[cfg(test)]
mod sona_plus_tests {
    use super::*;

    fn t(f: &[&str]) -> Task {
        Task { id: 0, task_type: None, features: f.iter().map(|s| s.to_string()).collect() }
    }
    fn map() -> HashMap<String, usize> {
        [("sql", 1), ("db", 1), ("css", 2), ("api", 0)].iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn defaults_equal_guarded_hebr() {
        let cfg = SonaPlusConfig::default();
        let mut sp = SonaPlus::new(3, &map(), cfg.clone());
        let hcfg = HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() };
        let mut g = Guarded::new(HebbianRouter::new(3, hcfg), map(), 3, cfg.guard());
        let (mut r1, mut r2, mut rw) = (Rng::new(3), Rng::new(3), Rng::new(9));
        let tasks = [t(&["sql"]), t(&["css", "api"]), t(&["db", "sql"]), t(&["zzz"])];
        let p = [0.3, 0.2, 0.9];
        for i in 0..400 {
            let task = &tasks[i % 4];
            let a = sp.select(task, &[0, 1, 2], &mut r1);
            let b = g.select(task, &[0, 1, 2], &mut r2);
            assert_eq!(a, b, "diverged at {i}");
            let rew = if rw.bernoulli(p[a]) { 1.0 } else { 0.0 };
            sp.update(task, a, rew);
            g.update(task, b, rew);
        }
    }

    #[test]
    fn config_serde_default_and_roundtrip() {
        let c: SonaPlusConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(c, SonaPlusConfig::default());
        let c2: SonaPlusConfig = serde_json::from_str("{\"explore_kw\":0.05,\"learner\":\"Patterns\"}").unwrap();
        assert_eq!(c2.learner, PlusLearner::Patterns);
        assert!((c2.explore_kw - 0.05).abs() < 1e-12 && c2.tau == c.tau);
        let s = serde_json::to_string(&c2).unwrap();
        assert_eq!(serde_json::from_str::<SonaPlusConfig>(&s).unwrap(), c2);
    }

    #[test]
    fn keyword_phase_exploration_finds_better_agent() {
        // keyword says agent 1 (p .6, above tau so it is never abandoned); agent 2 is .95.
        let run = |x: f64| {
            let cfg = SonaPlusConfig { explore_kw: x, explore_mode: PlusExplore::Ucb, tau: 0.45, ..SonaPlusConfig::default() };
            let mut r = SonaPlus::new(3, &map(), cfg);
            let task = t(&["sql"]);
            let mut rng = Rng::new(5);
            let p = [0.2, 0.6, 0.95];
            let mut late_good = 0;
            for i in 0..1500 {
                let a = r.select(&task, &[0, 1, 2], &mut rng);
                if i >= 1000 && a == 2 {
                    late_good += 1;
                }
                r.update(&task, a, if rng.bernoulli(p[a]) { 1.0 } else { 0.0 });
            }
            late_good
        };
        let (off, on) = (run(0.0), run(0.1));
        assert!(on > off + 100, "exploration did not help: off {off} on {on}");
    }

    #[test]
    fn weak_path_ignores_raw_success_and_learns_from_delayed() {
        let cfg = SonaPlusConfig { delayed_mult: 2, ..SonaPlusConfig::default() };
        let mut r = SonaPlus::new(3, &map(), cfg);
        let task = t(&["sql"]);
        for _ in 0..50 {
            r.update_weak(&task, 1, true); // no information
        }
        assert_eq!(r.g.pooled(&task, 1), (0.0, 0.0));
        r.update_weak(&task, 1, false);
        assert!(r.g.pooled(&task, 1).1 > 0.9);
        r.update_delayed(&task, 2, 1.0);
        assert!(r.g.pooled(&task, 2).0 > 1.5); // applied twice (with discount on the row)
    }

    #[test]
    fn pattern_learner_is_graded_and_prefers_better_agent() {
        let c = SonaPlusConfig { learner: PlusLearner::Patterns, pat_temp: 0.0, ..SonaPlusConfig::default() };
        let mut p = PatternLearner::new(3, &c);
        let task = t(&["x"]);
        for _ in 0..30 {
            p.update(&task, 0, 0.2);
            p.update(&task, 1, 0.9);
        }
        let s = p.scores(&task);
        assert!(s[1] > s[0] && s[0] < c.pat_init);
        let mut rng = Rng::new(1);
        assert_eq!(p.select(&task, &[0, 1], &mut rng), 1);
    }
}
