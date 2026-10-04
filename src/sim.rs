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
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig { n_tasks: 2000, team_frac: 0.3, confound_prob: 0.25, fail_frac: 0.10, oracle_team_max: 3, reveal_type: false }
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
    p_post: [[f64; 6]; 8],
    synergy: [[f64; 6]; 6],
    pub fail_agent: usize,
    /// failure window [fail_start, fail_end)
    pub fail_start: usize,
    pub fail_end: usize,
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

        let mut p_post = P_PRE;
        // feature: coder declines .85 -> .62, architect rises .55 -> .86
        p_post[0][CODER] = 0.62;
        p_post[0][ARCHITECT] = 0.86;
        // code-review: reviewer declines .82 -> .60, security-auditor rises .60 -> .84
        p_post[2][REVIEWER] = 0.60;
        p_post[2][SECURITY] = 0.84;

        let mut synergy = [[PAIR_COST; 6]; 6];
        for &(a, b, s) in SYNERGY.iter() {
            synergy[a][b] = s;
            synergy[b][a] = s;
        }

        let fail_agent = rng.below(AGENTS.len());
        let len = ((cfg.fail_frac * t as f64).round() as usize).max(1).min(t);
        let lo = (0.15 * t as f64) as usize;
        let hi = ((0.85 * t as f64) as usize).saturating_sub(len).max(lo);
        let fail_start = (lo + rng.below(hi - lo + 1)).min(t - len);
        World { seed, cfg, tasks, p_post, synergy, fail_agent, fail_start, fail_end: fail_start + len }
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
        if t < self.drift_at() { P_PRE[ty][agent] } else { self.p_post[ty][agent] }
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
}
