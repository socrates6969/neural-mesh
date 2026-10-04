//! Lessons from mistakes.
//!
//! When a task fails, [`LessonStore`] keeps a lesson (task feature vector,
//! agent, failure tag, short text) in a [`TieredMemory`]. On a new task it
//! retrieves the most similar past lessons and turns them into a per-agent
//! *warning score*; [`LessonStore::adjust_scores`] subtracts
//! `weight * warning` from a router's per-agent scores so agents with relevant
//! past failures are down-weighted.
//!
//! Warning for agent `a` on task `x`:
//!
//! ```text
//!   s_i   = cosine(x, lesson_i)                       (top-k lessons)
//!   w_i   = ((s_i - min_sim) / (1 - min_sim))^power · conf_i      for s_i >= min_sim
//!   warn_a = 1 - exp(-Σ_{i: agent_i = a} w_i)         in [0, 1)
//! ```
//!
//! `conf_i = (hits_i + 1) / (hits_i + misses_i + 2)`: a lesson is *confirmed*
//! (hit) when its agent was chosen again and failed, and *refuted* (miss) when
//! its agent was chosen again and succeeded. This is what stops a lesson from a
//! region where the agent fails only sometimes from vetoing that agent forever.
//! The only feedback a lesson can get is when its agent is actually chosen, so
//! a lesson that fully suppresses its agent is never refuted except through the
//! router's own exploration.
//!
//! Failure-only memory cannot tell "this agent fails here 95% of the time" from
//! "this agent fails here 15% of the time": both leave lessons of similar
//! similarity. With `track_successes` the store also keeps a (cheap) record of
//! each success and the warning becomes a local, similarity-weighted failure
//! *rate* excess over a prior instead of a failure *count*:
//!
//! ```text
//!   rate_a = (F_a + n0*p0) / (F_a + S_a + n0),   warn_a = clamp((rate_a - p0)/(1 - p0), 0, 1)
//! ```
//!
//! where `F_a`/`S_a` are the ramp-weighted neighbour failures/successes of
//! agent `a`. This departs from "lessons from mistakes" (it is a kNN outcome
//! model) and is reported as a separate arm.
//!
//! Lessons are stored with the tiered memory's usefulness stats: a confirmed
//! lesson gets reward 1, a refuted one reward 0, so under a byte budget useless
//! lessons are forgotten first.

use std::collections::HashMap;

use crate::tiered::{TieredConfig, TieredMemory};

const SEP: char = '\u{1f}';

/// A stored lesson (as returned by [`LessonStore::lessons_for`]).
#[derive(Clone, Debug, PartialEq)]
pub struct Lesson {
    pub id: u64,
    pub agent: usize,
    pub tag: String,
    pub text: String,
    /// Cosine similarity of the lesson's task vector to the query.
    pub similarity: f32,
    /// Times this lesson's agent was chosen again and failed / succeeded.
    pub hits: u32,
    pub misses: u32,
}

#[derive(Clone, Debug)]
pub struct LessonConfig {
    /// Lessons retrieved per task.
    pub k: usize,
    /// Lessons less similar than this are ignored.
    pub min_sim: f32,
    /// Penalty scale: `score[a] -= weight * warn[a]`.
    pub weight: f64,
    /// Sharpness of the similarity ramp.
    pub power: f64,
    /// If a retrieved lesson of the same agent is at least this similar to a
    /// new failure, reinforce it instead of storing a duplicate.
    pub dedup_sim: Option<f32>,
    /// Also record successes and use failure *rates* (see module docs).
    pub track_successes: bool,
    /// Prior failure rate `p0` and its strength `n0` (only with `track_successes`).
    pub prior_rate: f64,
    pub prior_strength: f64,
    pub memory: TieredConfig,
}

impl LessonConfig {
    pub fn new(dim: usize) -> Self {
        let mut memory = TieredConfig::new(dim);
        memory.short_cap = 128;
        Self {
            k: 8,
            min_sim: 0.7,
            weight: 1.0,
            power: 1.0,
            dedup_sim: None,
            track_successes: false,
            prior_rate: 0.25,
            prior_strength: 2.0,
            memory,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Pending {
    id: u64,
    agent: usize,
    sim: f32,
}

pub struct LessonStore {
    cfg: LessonConfig,
    mem: TieredMemory,
    counts: HashMap<u64, (u32, u32)>, // (hits, misses)
    pending: Vec<Pending>,
    next_id: u64,
}

fn encode(agent: usize, tag: &str, text: &str) -> String {
    format!("{agent}{SEP}{tag}{SEP}{text}")
}

/// Success notes are stored with this tag.
const OK_TAG: &str = "__ok__";

fn decode(meta: &str) -> Option<(usize, &str, &str)> {
    let mut it = meta.splitn(3, SEP);
    let agent = it.next()?.parse().ok()?;
    Some((agent, it.next()?, it.next()?))
}

impl LessonStore {
    pub fn new(cfg: LessonConfig) -> Self {
        let mem = TieredMemory::new(cfg.memory.clone());
        Self { cfg, mem, counts: HashMap::new(), pending: Vec::new(), next_id: 0 }
    }

    pub fn config(&self) -> &LessonConfig {
        &self.cfg
    }
    pub fn set_weight(&mut self, w: f64) {
        self.cfg.weight = w;
    }
    pub fn len(&self) -> usize {
        self.mem.len()
    }
    pub fn is_empty(&self) -> bool {
        self.mem.is_empty()
    }
    pub fn bytes(&self) -> usize {
        self.mem.bytes() + self.counts.len() * 32
    }
    pub fn memory(&self) -> &TieredMemory {
        &self.mem
    }

    /// Store a lesson unconditionally. Returns its id.
    pub fn record_failure(&mut self, task_vec: &[f32], agent: usize, tag: &str, text: &str) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.mem.add(id, task_vec, &encode(agent, tag, text));
        self.counts.insert(id, (1, 0));
        self.prune_counts();
        id
    }

    fn prune_counts(&mut self) {
        if self.counts.len() > self.mem.len() + 32 {
            let mem = &self.mem;
            self.counts.retain(|id, _| mem.contains(*id));
        }
    }

    fn conf(&self, id: u64) -> f64 {
        let (h, m) = self.counts.get(&id).copied().unwrap_or((0, 0));
        (h as f64 + 1.0) / (h as f64 + m as f64 + 2.0)
    }

    fn warn_from(&self, hits: &[(u64, f32)], n_agents: usize) -> (Vec<f64>, Vec<Pending>) {
        let mut fail = vec![0.0f64; n_agents];
        let mut succ = vec![0.0f64; n_agents];
        let mut pend = Vec::new();
        for &(id, sim) in hits {
            if sim < self.cfg.min_sim {
                continue;
            }
            let Some((agent, tag, _)) = self.mem.meta(id).and_then(decode) else { continue };
            if agent >= n_agents {
                continue;
            }
            let ramp = ((sim - self.cfg.min_sim) / (1.0 - self.cfg.min_sim).max(1e-6)).clamp(0.0, 1.0) as f64;
            let r = ramp.powf(self.cfg.power);
            if tag == OK_TAG {
                succ[agent] += r;
            } else {
                fail[agent] += r * self.conf(id);
                pend.push(Pending { id, agent, sim });
            }
        }
        let warn = if self.cfg.track_successes {
            let (p0, n0) = (self.cfg.prior_rate, self.cfg.prior_strength);
            (0..n_agents)
                .map(|a| {
                    let rate = (fail[a] + n0 * p0) / (fail[a] + succ[a] + n0);
                    ((rate - p0) / (1.0 - p0)).clamp(0.0, 1.0)
                })
                .collect()
        } else {
            fail.into_iter().map(|x| 1.0 - (-x).exp()).collect()
        };
        (warn, pend)
    }

    /// Per-agent warning in [0, 1) for a task (read-only; no stats recorded).
    pub fn warning_scores(&self, task_vec: &[f32], n_agents: usize) -> Vec<f64> {
        let hits = self.mem.search_ro(task_vec, self.cfg.k);
        self.warn_from(&hits, n_agents).0
    }

    /// Down-weight agents with relevant past failures:
    /// `scores[a] -= weight * warning[a]`. Remembers which lessons fired so a
    /// following [`LessonStore::observe`] can confirm or refute them.
    pub fn adjust_scores(&mut self, task_vec: &[f32], scores: &mut [f64]) {
        let hits = self.mem.search(task_vec, self.cfg.k);
        let (warn, pend) = self.warn_from(&hits, scores.len());
        self.pending = pend;
        for (s, w) in scores.iter_mut().zip(&warn) {
            *s -= self.cfg.weight * w;
        }
    }

    /// Feed back the outcome of the agent that was chosen after the last
    /// [`adjust_scores`](Self::adjust_scores): confirm its lessons if it failed,
    /// refute them if it succeeded. Returns the id of the strongest lesson that
    /// fired for this agent, if any.
    pub fn observe(&mut self, agent: usize, success: bool) -> Option<u64> {
        let pend = std::mem::take(&mut self.pending);
        let mut best: Option<Pending> = None;
        for p in pend.into_iter().filter(|p| p.agent == agent) {
            let c = self.counts.entry(p.id).or_insert((1, 0));
            if success {
                c.1 += 1;
            } else {
                c.0 += 1;
            }
            self.mem.record_outcome(p.id, if success { 0.0 } else { 1.0 });
            if best.map_or(true, |b| p.sim > b.sim) {
                best = Some(p);
            }
        }
        self.prune_counts();
        best.map(|b| b.id)
    }

    /// One-call task epilogue: [`observe`](Self::observe), and on failure store
    /// a lesson unless a sufficiently similar lesson for this agent already
    /// fired (see `dedup_sim`). Returns the new lesson id, if one was stored.
    pub fn after_task(
        &mut self,
        task_vec: &[f32],
        agent: usize,
        success: bool,
        tag: &str,
        text: &str,
    ) -> Option<u64> {
        let dup_sim = self
            .pending
            .iter()
            .filter(|p| p.agent == agent)
            .map(|p| p.sim)
            .fold(f32::NEG_INFINITY, f32::max);
        self.observe(agent, success);
        if success {
            if self.cfg.track_successes {
                let id = self.next_id;
                self.next_id += 1;
                self.mem.add(id, task_vec, &encode(agent, OK_TAG, ""));
            }
            return None;
        }
        if let Some(t) = self.cfg.dedup_sim {
            if dup_sim >= t {
                return None;
            }
        }
        Some(self.record_failure(task_vec, agent, tag, text))
    }

    /// Top lessons for inspection / explanation (read-only).
    pub fn lessons_for(&self, task_vec: &[f32]) -> Vec<Lesson> {
        self.mem
            .search_ro(task_vec, self.cfg.k)
            .into_iter()
            .filter_map(|(id, sim)| {
                let (agent, tag, text) = decode(self.mem.meta(id)?)?;
                if tag == OK_TAG {
                    return None;
                }
                let (hits, misses) = self.counts.get(&id).copied().unwrap_or((0, 0));
                Some(Lesson { id, agent, tag: tag.to_string(), text: text.to_string(), similarity: sim, hits, misses })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{normalized, synth::Rng};

    const DIM: usize = 32;

    fn unit(rng: &mut Rng) -> Vec<f32> {
        normalized(&(0..DIM).map(|_| rng.normal()).collect::<Vec<_>>())
    }

    fn near(c: &[f32], rng: &mut Rng, noise: f32) -> Vec<f32> {
        let n: Vec<f32> = (0..DIM).map(|_| rng.normal()).collect();
        normalized(&c.iter().zip(&n).map(|(a, b)| a + noise * b / (DIM as f32).sqrt()).collect::<Vec<_>>())
    }

    fn store() -> LessonStore {
        let mut c = LessonConfig::new(DIM);
        c.min_sim = 0.6;
        c.weight = 2.0;
        LessonStore::new(c)
    }

    #[test]
    fn similar_task_penalises_failed_agent_only() {
        let mut rng = Rng::new(1);
        let c = unit(&mut rng);
        let mut s = store();
        s.record_failure(&near(&c, &mut rng, 0.2), 2, "timeout", "agent 2 times out on big inputs");
        let mut scores = vec![1.0, 1.0, 1.0, 1.0];
        s.adjust_scores(&near(&c, &mut rng, 0.2), &mut scores);
        assert!(scores[2] < 0.95, "{scores:?}");
        assert_eq!(scores[0], 1.0);
        assert_eq!(scores[1], 1.0);
        assert_eq!(scores[3], 1.0);
    }

    #[test]
    fn dissimilar_task_is_untouched() {
        let mut rng = Rng::new(2);
        let a = unit(&mut rng);
        let mut b = unit(&mut rng);
        while crate::memory::dot(&a, &b) > 0.2 {
            b = unit(&mut rng);
        }
        let mut s = store();
        s.record_failure(&a, 1, "t", "x");
        let mut scores = vec![0.5, 0.5, 0.5];
        s.adjust_scores(&b, &mut scores);
        assert_eq!(scores, vec![0.5, 0.5, 0.5]);
        assert!(s.warning_scores(&b, 3).iter().all(|w| *w == 0.0));
    }

    #[test]
    fn more_lessons_and_higher_similarity_warn_more() {
        let mut rng = Rng::new(3);
        let c = unit(&mut rng);
        let mut s = store();
        s.record_failure(&near(&c, &mut rng, 0.3), 0, "t", "");
        let w1 = s.warning_scores(&c, 2)[0];
        for _ in 0..4 {
            s.record_failure(&near(&c, &mut rng, 0.3), 0, "t", "");
        }
        let w5 = s.warning_scores(&c, 2)[0];
        assert!(w5 > w1 && w5 < 1.0, "{w1} {w5}");
    }

    #[test]
    fn refuted_lesson_loses_force_confirmed_gains() {
        let mut rng = Rng::new(4);
        let c = unit(&mut rng);
        let mut s = store();
        s.record_failure(&c, 1, "t", "");
        let w0 = s.warning_scores(&c, 2)[1];
        for _ in 0..6 {
            let mut sc = vec![0.0, 0.0];
            s.adjust_scores(&c, &mut sc);
            s.observe(1, true); // chosen anyway, succeeded -> refuted
        }
        let w_refuted = s.warning_scores(&c, 2)[1];
        assert!(w_refuted < w0 * 0.6, "{w0} -> {w_refuted}");
        for _ in 0..12 {
            let mut sc = vec![0.0, 0.0];
            s.adjust_scores(&c, &mut sc);
            s.observe(1, false); // failed again -> confirmed
        }
        assert!(s.warning_scores(&c, 2)[1] > w_refuted);
    }

    #[test]
    fn observe_only_touches_chosen_agent_and_after_task_stores_lessons() {
        let mut rng = Rng::new(5);
        let c = unit(&mut rng);
        let mut s = store();
        s.record_failure(&c, 0, "a", "");
        s.record_failure(&c, 1, "b", "");
        let mut sc = vec![0.0, 0.0, 0.0];
        s.adjust_scores(&c, &mut sc);
        // agent 2 chosen: neither lesson is touched
        let n0 = s.len();
        assert_eq!(s.after_task(&c, 2, true, "", ""), None);
        assert_eq!(s.len(), n0);
        for l in s.lessons_for(&c) {
            assert_eq!((l.hits, l.misses), (1, 0));
        }
        // failure of agent 2 stores a lesson
        assert!(s.after_task(&c, 2, false, "x", "new").is_some());
        assert_eq!(s.len(), n0 + 1);
        assert!(s.lessons_for(&c).iter().any(|l| l.agent == 2 && l.tag == "x" && l.text == "new"));
    }

    #[test]
    fn dedup_reinforces_instead_of_storing() {
        let mut rng = Rng::new(6);
        let c = unit(&mut rng);
        let mut cfg = LessonConfig::new(DIM);
        cfg.min_sim = 0.6;
        cfg.dedup_sim = Some(0.95);
        let mut s = LessonStore::new(cfg);
        s.record_failure(&c, 3, "t", "");
        let mut sc = vec![0.0; 4];
        s.adjust_scores(&c, &mut sc);
        assert_eq!(s.after_task(&c, 3, false, "t", ""), None);
        assert_eq!(s.len(), 1);
        let l = &s.lessons_for(&c)[0];
        assert_eq!(l.hits, 2);
    }

    #[test]
    fn budgeted_store_forgets_unconfirmed_lessons_first() {
        let mut rng = Rng::new(7);
        let mut cfg = LessonConfig::new(DIM);
        cfg.memory.short_cap = 8;
        cfg.memory.forgetting = true;
        cfg.memory.consolidate_every = 16;
        cfg.memory.value_halflife = 1e9;
        let per = DIM * 4 + 32 + crate::tiered::ENTRY_OVERHEAD + 24;
        cfg.memory.byte_budget = 60 * per;
        let mut s = LessonStore::new(cfg);
        let precious = unit(&mut rng);
        let pid = s.record_failure(&precious, 1, "keep", "");
        let mut sc = vec![0.0; 2];
        s.adjust_scores(&precious, &mut sc);
        s.observe(1, false); // confirmed -> reward 1
        for _ in 0..500 {
            let v = unit(&mut rng);
            s.record_failure(&v, 0, "noise", "");
        }
        assert!(s.bytes() < 80 * per, "{} vs {}", s.bytes(), 80 * per);
        assert!(s.memory().contains(pid), "confirmed lesson survives");
        assert!(s.len() < 100);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let m = encode(7, "tag", "text with spaces");
        assert_eq!(decode(&m), Some((7, "tag", "text with spaces")));
    }
}
