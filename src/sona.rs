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
