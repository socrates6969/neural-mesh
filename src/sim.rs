//! Deterministic simulated world for benchmarking routers and the NeuralMesh.
//! See DESIGN.md "Simulation".
//!
//! A `World` is fully determined by `(seed, WorldConfig)`. It pre-draws the whole
//! task stream (type, noisy keyword features, team flag) and one uniform per task,
//! so every router evaluated on the same seed sees identical tasks and the outcome
//! of "agent/team X on task t" is the same for all routers (paired comparison).
//!
//! Hidden structure:
//! * 6 agents, 8 task types, success-probability table `P[type][agent]`.
//! * Two TRAP types (`bugfix`, `performance`) whose keywords point the static
//!   keyword map at a mediocre agent while another agent is clearly better.
//! * Mid-run drift at `T/2`: for `feature` and `code-review` the old best agent
//!   declines (stays > 50%) and another agent becomes clearly better.
//! * Team tasks (`team_frac` of tasks): success = clamp(P[type][leader] +
//!   sum of hidden pair synergies over all pairs in the team). A few pairs have
//!   strong positive synergy; every other pair carries a small coordination cost.
//! * A node-failure window: one agent always fails for `fail_frac * T` tasks.
//!
//! Scenarios (`WorldConfig::scenario`): `Fixed` is the original hand-designed world
//! (kept bit-for-bit so old numbers reproduce). `Random`, `NoSynergy` and `NoTraps`
//! draw the hidden structure per seed from a SEPARATE rng stream (so the task
//! stream for a seed is identical across scenarios): which types are traps, which
//! agent is best per type, which 2-4 pairs are synergistic (size 0.1-0.3), which
//! types drift and to whom, and the failure-window length. `NoSynergy` has no
//! synergistic pair (every pair costs `PAIR_COST`, so teams only hurt); `NoTraps`
//! has a keyword map that is correct before drift (only drift can fool it).

use crate::router::{Rng, Task};
use std::collections::HashMap;

pub const AGENTS: [&str; 6] = ["coder", "tester", "reviewer", "security-auditor", "architect", "docs-writer"];
pub const CODER: usize = 0;
pub const TESTER: usize = 1;
pub const REVIEWER: usize = 2;
pub const SECURITY: usize = 3;
pub const ARCHITECT: usize = 4;
pub const DOCS: usize = 5;

pub const TYPES: [&str; 8] = [
    "feature", "testing", "code-review", "security", "architecture", "documentation", "bugfix", "performance",
];
pub const TRAP_TYPES: [usize; 2] = [6, 7];
pub const DRIFT_TYPES: [usize; 2] = [0, 2];

/// Type keywords (each task draws 2-3 of its type's keywords).
const KEYWORDS: [[&str; 4]; 8] = [
    ["implement", "endpoint", "feature", "function"],
    ["unit-test", "coverage", "flaky", "assertion"],
    ["pull-request", "review", "lint", "style"],
    ["vulnerability", "auth", "cve", "injection"],
    ["design", "schema", "scalability", "interface"],
    ["readme", "docs", "tutorial", "changelog"],
    ["bug", "crash", "regression", "stacktrace"],
    ["latency", "profiling", "memory", "throughput"],
];

/// Where the static keyword map sends each type's keywords (the "obvious" agent).
/// Correct for types 0-5; WRONG for the traps (bugfix -> coder, performance -> architect).
const KEYWORD_AGENT: [usize; 8] = [CODER, TESTER, REVIEWER, SECURITY, ARCHITECT, DOCS, CODER, ARCHITECT];

/// Hidden success probabilities before drift. Rows: types, cols: agents
/// (coder, tester, reviewer, security-auditor, architect, docs-writer).
const P_PRE: [[f64; 6]; 8] = [
    [0.85, 0.45, 0.50, 0.35, 0.55, 0.20], // feature
    [0.45, 0.85, 0.55, 0.40, 0.30, 0.25], // testing
    [0.50, 0.55, 0.82, 0.60, 0.50, 0.35], // code-review
    [0.30, 0.40, 0.55, 0.85, 0.45, 0.15], // security
    [0.50, 0.30, 0.45, 0.40, 0.83, 0.30], // architecture
    [0.35, 0.30, 0.45, 0.20, 0.40, 0.88], // documentation
    [0.45, 0.80, 0.50, 0.35, 0.30, 0.15], // bugfix (TRAP: map says coder, tester is best)
    [0.80, 0.45, 0.40, 0.30, 0.42, 0.15], // performance (TRAP: map says architect, coder is best)
];

/// Hidden pair synergy for team tasks (symmetric). Unlisted pairs: `PAIR_COST`.
const SYNERGY: [(usize, usize, f64); 3] = [(CODER, TESTER, 0.22), (ARCHITECT, SECURITY, 0.22), (REVIEWER, DOCS, 0.18)];
pub const PAIR_COST: f64 = -0.06;
pub const P_MIN: f64 = 0.02;
pub const P_MAX: f64 = 0.97;

/// Which hidden-structure family a world is drawn from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    /// The original hand-designed world (fixed traps/synergies/drift).
    Fixed,
    /// Per-seed random skills, 1-3 traps, 1-3 drift types, 2-4 synergistic pairs.
    Random,
    /// As `Random` but no synergistic pair at all.
    NoSynergy,
    /// As `Random` but no trap types (keyword map correct pre-drift).
    NoTraps,
}

impl Scenario {
    pub fn is_random(self) -> bool {
        self != Scenario::Fixed
    }
    pub fn label(self) -> &'static str {
        match self {
            Scenario::Fixed => "fixed",
            Scenario::Random => "random",
            Scenario::NoSynergy => "no-synergy",
            Scenario::NoTraps => "no-traps",
        }
    }
}

#[derive(Clone, Debug)]
pub struct WorldConfig {
    pub n_tasks: usize,
    /// fraction of tasks that are team tasks
    pub team_frac: f64,
    /// probability that a confound keyword from another type is appended
    pub confound_prob: f64,
    /// length of the node-failure window as a fraction of the run
    pub fail_frac: f64,
    /// max team size the oracle may use on team tasks (for regret)
    pub oracle_team_max: usize,
    /// expose `task_type` to routers (default false: routers only see noisy keywords)
    pub reveal_type: bool,
    /// hidden-structure family (default `Fixed`)
    pub scenario: Scenario,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            n_tasks: 2000,
            team_frac: 0.3,
            confound_prob: 0.25,
            fail_frac: 0.10,
            oracle_team_max: 3,
            reveal_type: false,
            scenario: Scenario::Fixed,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimTask {
    pub task: Task,
    pub type_idx: usize,
    /// team task: hidden pair synergy applies to the assigned team. On solo tasks
    /// only the first (leader) agent of a team matters.
    pub team_task: bool,
    /// pre-drawn uniform: outcome = (u < success_prob)
    pub u: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    PreDrift,
    EarlyPost,
    LatePost,
}

pub struct World {
    pub seed: u64,
    pub cfg: WorldConfig,
    tasks: Vec<SimTask>,
    p_pre: [[f64; 6]; 8],
    p_post: [[f64; 6]; 8],
    synergy: [[f64; 6]; 6],
    keyword_agent: [usize; 8],
    /// trap types of THIS world (keyword map points at a clearly worse agent)
    pub trap_types: Vec<usize>,
    /// types whose best agent changes at `T/2` in THIS world
    pub drift_types: Vec<usize>,
    /// synergistic pairs (a, b, size) of THIS world
    pub synergy_pairs: Vec<(usize, usize, f64)>,
    pub fail_agent: usize,
    /// failure window [fail_start, fail_end)
    pub fail_start: usize,
    pub fail_end: usize,
}

/// Hidden structure of one world (skills, keyword map, traps, drift, synergy).
struct Hidden {
    p_pre: [[f64; 6]; 8],
    p_post: [[f64; 6]; 8],
    keyword_agent: [usize; 8],
    trap_types: Vec<usize>,
    drift_types: Vec<usize>,
    synergy_pairs: Vec<(usize, usize, f64)>,
}

fn uni(rng: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * rng.f64()
}

fn shuffle<T>(rng: &mut Rng, v: &mut [T]) {
    for i in (1..v.len()).rev() {
        let j = rng.below(i + 1);
        v.swap(i, j);
    }
}

impl Hidden {
    fn fixed() -> Self {
        let mut p_post = P_PRE;
        // feature: coder declines .85 -> .62, architect rises .55 -> .86
        p_post[0][CODER] = 0.62;
        p_post[0][ARCHITECT] = 0.86;
        // code-review: reviewer declines .82 -> .60, security-auditor rises .60 -> .84
        p_post[2][REVIEWER] = 0.60;
        p_post[2][SECURITY] = 0.84;
        Hidden {
            p_pre: P_PRE,
            p_post,
            keyword_agent: KEYWORD_AGENT,
            trap_types: TRAP_TYPES.to_vec(),
            drift_types: DRIFT_TYPES.to_vec(),
            synergy_pairs: SYNERGY.to_vec(),
        }
    }

    fn wrng(seed: u64) -> Rng {
        Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0x5EED_BA5E_0F17_C0DE)
    }

    /// Failure-window length for randomised worlds: 5%-20% of the run.
    fn random_fail_frac(seed: u64) -> f64 {
        let mut r = Self::wrng(seed ^ 0xFA11);
        uni(&mut r, 0.05, 0.20)
    }

    fn random(seed: u64, sc: Scenario) -> Self {
        let mut r = Self::wrng(seed);
        let nt = TYPES.len();
        let na = AGENTS.len();
        // trap types (1-3 unless NoTraps)
        let n_traps = if sc == Scenario::NoTraps { 0 } else { 1 + r.below(3) };
        let mut order: Vec<usize> = (0..nt).collect();
        shuffle(&mut r, &mut order);
        let trap_types: Vec<usize> = order[..n_traps].to_vec();
        // drift types: 1-3 non-trap types
        let n_drift = 1 + r.below(3);
        let drift_types: Vec<usize> = order[n_traps..n_traps + n_drift].to_vec();

        let mut p_pre = [[0.0; 6]; 8];
        let mut keyword_agent = [0usize; 8];
        for ty in 0..nt {
            let trap = trap_types.contains(&ty);
            let best = r.below(na);
            let best_p = if trap { uni(&mut r, 0.80, 0.90) } else { uni(&mut r, 0.78, 0.90) };
            for a in 0..na {
                p_pre[ty][a] = if a == best {
                    best_p
                } else if trap {
                    uni(&mut r, 0.15, 0.50)
                } else {
                    uni(&mut r, 0.15, 0.55)
                };
            }
            keyword_agent[ty] = best;
            if trap {
                // the map points at a clearly worse agent (gap >= 0.3 by construction)
                let mut w = r.below(na - 1);
                if w >= best {
                    w += 1;
                }
                p_pre[ty][w] = uni(&mut r, 0.30, 0.50);
                keyword_agent[ty] = w;
            }
        }
        let mut p_post = p_pre;
        for &ty in &drift_types {
            let old = (0..na).max_by(|&a, &b| p_pre[ty][a].partial_cmp(&p_pre[ty][b]).unwrap()).unwrap();
            let mut new = r.below(na - 1);
            if new >= old {
                new += 1;
            }
            let old_post = uni(&mut r, 0.52, 0.65);
            p_post[ty][old] = old_post;
            p_post[ty][new] = (old_post + uni(&mut r, 0.20, 0.30)).min(0.95);
        }
        // synergy: 2-4 distinct pairs of size 0.1-0.3 (none for NoSynergy)
        let mut synergy_pairs = vec![];
        if sc != Scenario::NoSynergy {
            let mut pairs: Vec<(usize, usize)> = vec![];
            for a in 0..na {
                for b in (a + 1)..na {
                    pairs.push((a, b));
                }
            }
            shuffle(&mut r, &mut pairs);
            let n = 2 + r.below(3);
            for &(a, b) in pairs.iter().take(n) {
                synergy_pairs.push((a, b, uni(&mut r, 0.10, 0.30)));
            }
        }
        Hidden { p_pre, p_post, keyword_agent, trap_types, drift_types, synergy_pairs }
    }
}

impl World {
    pub fn new(seed: u64, cfg: WorldConfig) -> Self {
        let mut rng = Rng::new(seed.wrapping_mul(0x5851_F42D_4C95_7F2D) ^ 0xA5A5_0123);
        let t = cfg.n_tasks;
        let mut tasks = Vec::with_capacity(t);
        for id in 0..t {
            let ty = rng.below(TYPES.len());
            // 2 or 3 distinct keywords of the true type
            let k = 2 + rng.below(2);
            let mut idx = [0usize, 1, 2, 3];
            for i in (1..4).rev() {
                let j = rng.below(i + 1);
                idx.swap(i, j);
            }
            let mut features: Vec<String> = idx[..k].iter().map(|&i| KEYWORDS[ty][i].to_string()).collect();
            if rng.bernoulli(cfg.confound_prob) {
                let mut other = rng.below(TYPES.len() - 1);
                if other >= ty {
                    other += 1;
                }
                features.push(KEYWORDS[other][rng.below(4)].to_string());
            }
            // arbitrary order: a confound keyword may come first
            for i in (1..features.len()).rev() {
                let j = rng.below(i + 1);
                features.swap(i, j);
            }
            let team_task = rng.bernoulli(cfg.team_frac);
            let u = rng.f64();
            tasks.push(SimTask {
                task: Task {
                    id: id as u64,
                    task_type: if cfg.reveal_type { Some(TYPES[ty].to_string()) } else { None },
                    features,
                },
                type_idx: ty,
                team_task,
                u,
            });
        }

        let hidden = if cfg.scenario.is_random() { Hidden::random(seed, cfg.scenario) } else { Hidden::fixed() };
        let Hidden { p_pre, p_post, keyword_agent, trap_types, drift_types, synergy_pairs } = hidden;
        let mut synergy = [[PAIR_COST; 6]; 6];
        for &(a, b, s) in synergy_pairs.iter() {
            synergy[a][b] = s;
            synergy[b][a] = s;
        }

        let fail_agent = rng.below(AGENTS.len());
        let fail_frac = if cfg.scenario.is_random() { Hidden::random_fail_frac(seed) } else { cfg.fail_frac };
        let len = ((fail_frac * t as f64).round() as usize).max(1).min(t);
        let lo = (0.15 * t as f64) as usize;
        let hi = ((0.85 * t as f64) as usize).saturating_sub(len).max(lo);
        let fail_start = (lo + rng.below(hi - lo + 1)).min(t - len);
        World {
            seed,
            cfg,
            tasks,
            p_pre,
            p_post,
            synergy,
            keyword_agent,
            trap_types,
            drift_types,
            synergy_pairs,
            fail_agent,
            fail_start,
            fail_end: fail_start + len,
        }
    }

    pub fn n_agents(&self) -> usize {
        AGENTS.len()
    }
    pub fn agent_names(&self) -> Vec<String> {
        AGENTS.iter().map(|s| s.to_string()).collect()
    }
    pub fn len(&self) -> usize {
        self.tasks.len()
    }
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }
    pub fn task(&self, t: usize) -> &SimTask {
        &self.tasks[t]
    }
    pub fn tasks(&self) -> &[SimTask] {
        &self.tasks
    }
    pub fn drift_at(&self) -> usize {
        self.len() / 2
    }

    /// Static keyword -> agent map used by `KeywordRouter`.
    pub fn keyword_map() -> HashMap<String, usize> {
        let mut m = HashMap::new();
        for (ty, kws) in KEYWORDS.iter().enumerate() {
            for kw in kws {
                m.insert(kw.to_string(), KEYWORD_AGENT[ty]);
            }
        }
        m
    }

    /// Keyword -> agent map for THIS world (equals `keyword_map()` for `Fixed`).
    pub fn keyword_map_for(&self) -> HashMap<String, usize> {
        let mut m = HashMap::new();
        for (ty, kws) in KEYWORDS.iter().enumerate() {
            for kw in kws {
                m.insert(kw.to_string(), self.keyword_agent[ty]);
            }
        }
        m
    }

    /// Raw pre-drift / post-drift skill tables (diagnostics, tests).
    pub fn skill_tables(&self) -> (&[[f64; 6]; 8], &[[f64; 6]; 8]) {
        (&self.p_pre, &self.p_post)
    }

    pub fn keyword_agent(&self, ty: usize) -> usize {
        self.keyword_agent[ty]
    }

    pub fn window(&self, t: usize) -> Window {
        let d = self.drift_at();
        if t < d {
            Window::PreDrift
        } else if t < d + (self.len() - d) / 2 {
            Window::EarlyPost
        } else {
            Window::LatePost
        }
    }

    pub fn in_failure(&self, t: usize) -> bool {
        t >= self.fail_start && t < self.fail_end
    }

    pub fn is_down(&self, t: usize, agent: usize) -> bool {
        agent == self.fail_agent && self.in_failure(t)
    }

    /// Hidden single-agent skill at time `t` (ignores failures).
    pub fn skill(&self, t: usize, ty: usize, agent: usize) -> f64 {
        if t < self.drift_at() { self.p_pre[ty][agent] } else { self.p_post[ty][agent] }
    }

    pub fn synergy(&self, a: usize, b: usize) -> f64 {
        self.synergy[a][b]
    }

    /// True success probability of `team` (leader first) on task `t`.
    /// Solo task: only the leader matters. Team task: leader skill + pair synergies,
    /// where a pair involving a down agent only contributes the coordination cost.
    /// A down leader always fails.
    pub fn success_prob(&self, t: usize, team: &[usize]) -> f64 {
        assert!(!team.is_empty(), "team must contain a leader");
        let st = &self.tasks[t];
        let leader = team[0];
        if self.is_down(t, leader) {
            return 0.0;
        }
        let base = self.skill(t, st.type_idx, leader);
        if !st.team_task || team.len() == 1 {
            return base;
        }
        let mut s = 0.0;
        for i in 0..team.len() {
            for j in (i + 1)..team.len() {
                let (a, b) = (team[i], team[j]);
                if a == b {
                    continue;
                }
                s += if self.is_down(t, a) || self.is_down(t, b) { PAIR_COST.min(0.0) } else { self.synergy[a][b] };
            }
        }
        (base + s).clamp(P_MIN, P_MAX)
    }

    /// Paired, deterministic outcome: 1.0 on success, 0.0 on failure.
    pub fn outcome(&self, t: usize, team: &[usize]) -> f64 {
        if self.tasks[t].u < self.success_prob(t, team) { 1.0 } else { 0.0 }
    }

    /// Best achievable success probability on task `t` (best single agent on solo
    /// tasks; best team of size <= oracle_team_max on team tasks).
    pub fn oracle_prob(&self, t: usize) -> f64 {
        let n = self.n_agents();
        let mut best = 0.0f64;
        for a in 0..n {
            best = best.max(self.success_prob(t, &[a]));
        }
        if !self.tasks[t].team_task {
            return best;
        }
        let m = self.cfg.oracle_team_max;
        for a in 0..n {
            for b in 0..n {
                if b == a {
                    continue;
                }
                if m >= 2 {
                    best = best.max(self.success_prob(t, &[a, b]));
                }
                if m >= 3 {
                    for c in (b + 1)..n {
                        if c == a {
                            continue;
                        }
                        best = best.max(self.success_prob(t, &[a, b, c]));
                    }
                }
            }
        }
        best
    }

    /// Best single agent for a task at time `t` (ground truth; for tests/diagnostics).
    pub fn best_agent(&self, t: usize) -> usize {
        let ty = self.tasks[t].type_idx;
        (0..self.n_agents())
            .max_by(|&a, &b| self.skill(t, ty, a).partial_cmp(&self.skill(t, ty, b)).unwrap())
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(seed: u64) -> World {
        World::new(seed, WorldConfig::default())
    }

    #[test]
    fn deterministic_per_seed() {
        let a = world(7);
        let b = world(7);
        let c = world(8);
        assert_eq!(a.len(), 2000);
        for t in 0..a.len() {
            assert_eq!(a.task(t).task, b.task(t).task);
            assert_eq!(a.task(t).u, b.task(t).u);
            assert_eq!(a.task(t).team_task, b.task(t).team_task);
        }
        assert_eq!((a.fail_agent, a.fail_start), (b.fail_agent, b.fail_start));
        assert!((0..a.len()).any(|t| a.task(t).task != c.task(t).task));
    }

    #[test]
    fn type_hidden_by_default_revealable() {
        let w = world(4);
        assert!(w.tasks().iter().all(|st| st.task.task_type.is_none()));
        let r = World::new(4, WorldConfig { reveal_type: true, ..WorldConfig::default() });
        for t in 0..r.len() {
            assert_eq!(r.task(t).task.task_type.as_deref(), Some(TYPES[r.task(t).type_idx]));
            // same stream otherwise
            assert_eq!(r.task(t).task.features, w.task(t).task.features);
            assert_eq!(r.task(t).u, w.task(t).u);
        }
        // a confound keyword is sometimes listed first
        let first_confound = w
            .tasks()
            .iter()
            .filter(|st| !KEYWORDS[st.type_idx].contains(&st.task.features[0].as_str()))
            .count();
        assert!(first_confound > 20);
    }

    #[test]
    fn at_least_six_types_and_agents() {
        assert!(TYPES.len() >= 6);
        assert_eq!(AGENTS.len(), 6);
        assert_eq!(AGENTS, ["coder", "tester", "reviewer", "security-auditor", "architect", "docs-writer"]);
        let w = world(1);
        let mut seen = [0usize; 8];
        for st in w.tasks() {
            seen[st.type_idx] += 1;
        }
        assert!(seen.iter().all(|&c| c > 150), "all types occur: {seen:?}");
    }

    #[test]
    fn features_are_noisy_keywords() {
        let w = world(3);
        let mut confounds = 0;
        for st in w.tasks() {
            let own = &KEYWORDS[st.type_idx];
            let n_own = st.task.features.iter().filter(|f| own.contains(&f.as_str())).count();
            assert!((2..=3).contains(&n_own), "2-3 type keywords");
            let extra = st.task.features.len() - n_own;
            assert!(extra <= 1);
            confounds += extra;
            // features distinct
            let mut f = st.task.features.clone();
            f.sort();
            f.dedup();
            assert_eq!(f.len(), st.task.features.len());
        }
        let rate = confounds as f64 / w.len() as f64;
        assert!((0.21..0.29).contains(&rate), "confound rate {rate}");
    }

    #[test]
    fn traps_mislead_keyword_map() {
        let m = World::keyword_map();
        let w = world(1);
        for &ty in TRAP_TYPES.iter() {
            let mapped = m[KEYWORDS[ty][0]];
            let best = (0..6).max_by(|&a, &b| P_PRE[ty][a].partial_cmp(&P_PRE[ty][b]).unwrap()).unwrap();
            assert_ne!(mapped, best, "trap type {} must map to a wrong agent", TYPES[ty]);
            assert!(P_PRE[ty][best] - P_PRE[ty][mapped] >= 0.3);
            assert_eq!(w.skill(0, ty, best), P_PRE[ty][best]);
        }
        // non-trap types map to the true best agent (pre-drift)
        for ty in 0..6 {
            let best = (0..6).max_by(|&a, &b| P_PRE[ty][a].partial_cmp(&P_PRE[ty][b]).unwrap()).unwrap();
            assert_eq!(m[KEYWORDS[ty][0]], best);
        }
    }

    #[test]
    fn drift_is_mild_but_changes_best() {
        let w = world(1);
        let pre = 0;
        let post = w.drift_at();
        for &ty in DRIFT_TYPES.iter() {
            let old_best = (0..6).max_by(|&a, &b| w.skill(pre, ty, a).partial_cmp(&w.skill(pre, ty, b)).unwrap()).unwrap();
            let new_best = (0..6).max_by(|&a, &b| w.skill(post, ty, a).partial_cmp(&w.skill(post, ty, b)).unwrap()).unwrap();
            assert_ne!(old_best, new_best);
            assert!(w.skill(post, ty, old_best) > 0.5, "old best stays > 50%");
            assert!(w.skill(post, ty, new_best) - w.skill(post, ty, old_best) >= 0.2, "clearly better");
        }
        assert_eq!(w.window(0), Window::PreDrift);
        assert_eq!(w.window(1000), Window::EarlyPost);
        assert_eq!(w.window(1499), Window::EarlyPost);
        assert_eq!(w.window(1500), Window::LatePost);
    }

    #[test]
    fn team_synergy_and_solo_tasks() {
        let w = world(2);
        let t_team = (0..w.len()).find(|&t| w.task(t).team_task && !w.in_failure(t) && t < w.drift_at()).unwrap();
        let t_solo = (0..w.len()).find(|&t| !w.task(t).team_task && !w.in_failure(t) && t < w.drift_at()).unwrap();
        let ty = w.task(t_team).type_idx;
        let base = w.skill(t_team, ty, CODER);
        let good = w.success_prob(t_team, &[CODER, TESTER]);
        let bad = w.success_prob(t_team, &[CODER, DOCS]);
        assert!((good - (base + 0.22).clamp(P_MIN, P_MAX)).abs() < 1e-12);
        assert!((bad - (base - 0.06).clamp(P_MIN, P_MAX)).abs() < 1e-12);
        // solo task: helpers do not matter
        let ty2 = w.task(t_solo).type_idx;
        assert_eq!(w.success_prob(t_solo, &[CODER, TESTER, DOCS]), w.skill(t_solo, ty2, CODER));
        // oracle >= any team
        assert!(w.oracle_prob(t_team) >= good - 1e-12);
        assert!(w.oracle_prob(t_team) >= w.success_prob(t_team, &[w.best_agent(t_team)]));
    }

    #[test]
    fn failure_window_always_fails() {
        let w = world(5);
        assert_eq!(w.fail_end - w.fail_start, 200);
        assert!(w.fail_start >= 300 && w.fail_end <= 1700);
        for t in w.fail_start..w.fail_end {
            assert_eq!(w.success_prob(t, &[w.fail_agent]), 0.0);
            assert_eq!(w.outcome(t, &[w.fail_agent, (w.fail_agent + 1) % 6]), 0.0);
        }
        let t = w.fail_end; // just after: back to normal
        if t < w.len() {
            assert!(w.success_prob(t, &[w.fail_agent]) > 0.0);
        }
        // failure agent varies across seeds
        let agents: std::collections::HashSet<usize> = (1..40).map(|s| world(s).fail_agent).collect();
        assert!(agents.len() >= 4);
    }

    #[test]
    fn outcomes_paired_and_calibrated() {
        let w = world(11);
        // same team -> same outcome (pre-drawn uniform)
        for t in 0..50 {
            assert_eq!(w.outcome(t, &[CODER]), w.outcome(t, &[CODER]));
        }
        // empirical success of oracle-best single agent near its probability
        let (mut s, mut p) = (0.0, 0.0);
        for t in 0..w.len() {
            let b = w.best_agent(t);
            s += w.outcome(t, &[b]);
            p += w.success_prob(t, &[b]);
        }
        let n = w.len() as f64;
        assert!(((s - p) / n).abs() < 0.04);
        // monotone: higher prob never turns a success into a failure
        for t in 0..w.len() {
            if w.outcome(t, &[ARCHITECT]) == 1.0 && w.success_prob(t, &[CODER]) >= w.success_prob(t, &[ARCHITECT]) {
                assert_eq!(w.outcome(t, &[CODER]), 1.0);
            }
        }
    }

    // ---- randomised scenarios ----

    fn rworld(seed: u64, sc: Scenario) -> World {
        World::new(seed, WorldConfig { scenario: sc, ..WorldConfig::default() })
    }

    #[test]
    fn fixed_scenario_is_the_original_world() {
        let w = world(3);
        let (pre, post) = w.skill_tables();
        assert_eq!(*pre, P_PRE);
        assert_eq!(post[0][CODER], 0.62);
        assert_eq!(w.trap_types, TRAP_TYPES.to_vec());
        assert_eq!(w.drift_types, DRIFT_TYPES.to_vec());
        assert_eq!(w.synergy_pairs.len(), 3);
        assert_eq!(w.keyword_map_for(), World::keyword_map());
    }

    #[test]
    fn random_worlds_deterministic_per_seed() {
        for sc in [Scenario::Random, Scenario::NoSynergy, Scenario::NoTraps] {
            let (a, b, c) = (rworld(9, sc), rworld(9, sc), rworld(10, sc));
            assert_eq!(a.skill_tables(), b.skill_tables());
            assert_eq!(a.trap_types, b.trap_types);
            assert_eq!(a.drift_types, b.drift_types);
            assert_eq!(a.synergy_pairs, b.synergy_pairs);
            assert_eq!((a.fail_agent, a.fail_start, a.fail_end), (b.fail_agent, b.fail_start, b.fail_end));
            for t in 0..a.len() {
                assert_eq!(a.task(t).task, b.task(t).task);
                assert_eq!(a.task(t).u, b.task(t).u);
            }
            assert_ne!(a.skill_tables(), c.skill_tables(), "different seeds give different worlds");
        }
    }

    #[test]
    fn random_scenarios_share_the_fixed_task_stream() {
        let f = world(12);
        let r = rworld(12, Scenario::Random);
        for t in 0..f.len() {
            assert_eq!(f.task(t).task, r.task(t).task);
            assert_eq!(f.task(t).u, r.task(t).u);
            assert_eq!(f.task(t).team_task, r.task(t).team_task);
        }
        assert_ne!(f.skill_tables(), r.skill_tables());
    }

    #[test]
    fn random_worlds_have_valid_probabilities() {
        for sc in [Scenario::Random, Scenario::NoSynergy, Scenario::NoTraps] {
            for seed in 1..=40u64 {
                let w = rworld(seed, sc);
                let (pre, post) = w.skill_tables();
                for ty in 0..8 {
                    for a in 0..6 {
                        for p in [pre[ty][a], post[ty][a]] {
                            assert!((P_MIN..=P_MAX).contains(&p), "{sc:?} seed {seed} p={p}");
                        }
                    }
                }
                for &(a, b, s) in &w.synergy_pairs {
                    assert!(a < b && b < 6 && (0.1..=0.3).contains(&s));
                }
                for t in (0..w.len()).step_by(37) {
                    for team in [vec![0usize], vec![1, 2], vec![3, 4, 5], vec![w.fail_agent, 0]] {
                        let p = w.success_prob(t, &team);
                        assert!((0.0..=1.0).contains(&p));
                    }
                    let o = w.oracle_prob(t);
                    assert!((0.0..=1.0).contains(&o));
                    assert!(o >= w.success_prob(t, &[w.best_agent(t)]) - 1e-12);
                }
                assert!(w.fail_end <= w.len() && w.fail_end > w.fail_start);
                let frac = (w.fail_end - w.fail_start) as f64 / w.len() as f64;
                assert!((0.049..=0.201).contains(&frac), "fail frac {frac}");
            }
        }
    }

    #[test]
    fn random_structure_matches_scenario_contract() {
        let mut distinct_traps = std::collections::HashSet::new();
        let mut distinct_best = std::collections::HashSet::new();
        for seed in 1..=60u64 {
            let r = rworld(seed, Scenario::Random);
            assert!((1..=3).contains(&r.trap_types.len()));
            assert!((2..=4).contains(&r.synergy_pairs.len()));
            let mut ps: Vec<(usize, usize)> = r.synergy_pairs.iter().map(|p| (p.0, p.1)).collect();
            ps.sort();
            ps.dedup();
            assert_eq!(ps.len(), r.synergy_pairs.len(), "pairs distinct");
            distinct_traps.insert(r.trap_types.clone());
            let (pre, post) = r.skill_tables();
            for ty in 0..8 {
                let best = (0..6).max_by(|&a, &b| pre[ty][a].partial_cmp(&pre[ty][b]).unwrap()).unwrap();
                distinct_best.insert((ty, best));
                if r.trap_types.contains(&ty) {
                    let m = r.keyword_agent(ty);
                    assert_ne!(m, best);
                    assert!(pre[ty][best] - pre[ty][m] >= 0.3 - 1e-12, "trap gap");
                } else {
                    assert_eq!(r.keyword_agent(ty), best, "non-trap keyword map is correct pre-drift");
                }
            }
            assert!(r.drift_types.iter().all(|d| !r.trap_types.contains(d)));
            assert!((1..=3).contains(&r.drift_types.len()));
            for &ty in &r.drift_types {
                let old = (0..6).max_by(|&a, &b| pre[ty][a].partial_cmp(&pre[ty][b]).unwrap()).unwrap();
                let new = (0..6).max_by(|&a, &b| post[ty][a].partial_cmp(&post[ty][b]).unwrap()).unwrap();
                assert_ne!(old, new);
                assert!(post[ty][old] > 0.5);
                assert!(post[ty][new] - post[ty][old] >= 0.2 - 1e-12);
            }
            let ns = rworld(seed, Scenario::NoSynergy);
            assert!(ns.synergy_pairs.is_empty());
            assert!(!ns.trap_types.is_empty());
            let nt = rworld(seed, Scenario::NoTraps);
            assert!(nt.trap_types.is_empty());
            assert!(!nt.synergy_pairs.is_empty());
            assert_eq!(nt.keyword_map_for().len(), 32);
        }
        assert!(distinct_traps.len() > 10, "traps vary across seeds");
        assert!(distinct_best.len() > 30, "best agents vary across seeds");
    }

    #[test]
    fn no_synergy_world_teams_only_hurt() {
        let w = rworld(5, Scenario::NoSynergy);
        let t = (0..w.len()).find(|&t| w.task(t).team_task && !w.in_failure(t)).unwrap();
        let solo = w.success_prob(t, &[CODER]);
        assert!(w.success_prob(t, &[CODER, TESTER]) <= solo);
        assert!(w.success_prob(t, &[CODER, TESTER, DOCS]) <= solo);
        assert!((w.oracle_prob(t) - (0..6).map(|a| w.success_prob(t, &[a])).fold(0.0, f64::max)).abs() < 1e-12);
    }
}
