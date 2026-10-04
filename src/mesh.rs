//! NeuralMesh: a leader router plus Hebbian co-activation edges between agents
//! and per-node health for fault tolerance. See DESIGN.md.
//!
//! * **Leader**: chosen by the wrapped router from the *eligible* candidates.
//! * **Helpers**: the remaining eligible candidates ranked by edge strength to
//!   the leader; the top `team_size - 1` fire (sparse activation). Each helper
//!   slot is replaced by a uniformly random remaining candidate with
//!   probability `helper_epsilon` (small exploration). Exact edge ties are
//!   broken randomly.
//! * **Edges** (symmetric, clamped to `[-1, 1]`): on feedback for a team of
//!   size >= 2, every edge decays by `edge_decay`, then every pair in the team
//!   gets `edge_eta * (reward - baseline)`, where `baseline` is an EMA of the
//!   team rewards of teams led by the same leader (pre-update value used).
//! * **Health**: per-node EMA of the rewards of teams the node was part of
//!   (starts at 1.0). At routing time a candidate with
//!   `health < health_threshold * max_health(candidates)` is benched (skipped).
//!   The threshold is *relative* to the healthiest candidate so that a hard
//!   workload (all reward EMAs low) does not bench every node. After
//!   `retry_every` routes on the bench a node is put on probation: its health
//!   is raised to exactly the current threshold, which makes it eligible
//!   again; one more bad outcome benches it again.
//! * If no candidate is eligible, the healthiest candidates are used, so the
//!   team is never empty when at least one candidate exists.
//!
//! * **Gated recruitment** (optional, `with_gate`): `k` becomes an upper bound.
//!   Helper candidates are tried strongest-edge first; helper `h` joins leader
//!   `l` only if its learned marginal value `with[l][h] - baseline` exceeds
//!   `GateConfig::min_gain`, where `with[l][h]` is an EMA of the team reward of
//!   teams led by `l` that included `h`, and `baseline` is the leader's EMA
//!   solo reward (or the best `with` of helpers already recruited). Candidates
//!   are ordered proven-helpful, unmeasured, known-harmful (edge strength within
//!   each group), so a stale or decayed edge cannot hide a proven helper. Recruitment
//!   stops at the first helper that fails the gate. Unmeasured pairs are tried
//!   optimistically after `min_obs` solo observations exist, and every failed
//!   gate is overridden with probability `explore` so values keep updating.
//!
//! RNG: with `team_size == 1` the mesh consumes randomness *only* through the
//! leader router, and while no node is benched the leader sees exactly the
//! caller's candidate list. Hence a mesh with team size 1 makes the same
//! choices as the bare router (whenever no node is benched; set
//! `health_threshold = 0` to disable benching entirely).

use serde::{Deserialize, Serialize};

use crate::router::{Rng, Router, Task};

/// Opt-in faster bench/recover policy (see `NeuralMesh::with_fault`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FaultConfig {
    /// Bench a node right away after this many consecutive failed outcomes
    /// (reward < `fail_below`). 0 = off (health EMA only).
    pub streak: u32,
    pub fail_below: f64,
    /// When a probation attempt fails the node's retry interval is multiplied by this
    /// (>= 1; 1 = off), capped at `max_mult`; a success resets it to 1.
    pub backoff: f64,
    pub max_mult: f64,
}

impl Default for FaultConfig {
    fn default() -> Self {
        FaultConfig { streak: 0, fail_below: 0.5, backoff: 1.0, max_mult: 16.0 }
    }
}

/// Opt-in context-specific benching (see `NeuralMesh::with_context_bench`).
///
/// Health is additionally tracked per (task feature, agent): an EMA of the rewards
/// the agent earned on tasks carrying that feature (starts at 1.0). For a task, an
/// agent's *context health* is the mean over the task's features. A candidate whose
/// context health is below `threshold * best context health among candidates` is
/// skipped for THIS task only (it stays eligible for tasks in other contexts). Every
/// `retry_every`-th time an agent is context-benched it is let through anyway
/// (probation), so a recovered agent is re-measured. Deterministic (no rng).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextBenchConfig {
    pub alpha: f64,
    pub threshold: f64,
    pub retry_every: u64,
}

impl Default for ContextBenchConfig {
    fn default() -> Self {
        ContextBenchConfig { alpha: 0.3, threshold: 0.5, retry_every: 20 }
    }
}

/// Per-(feature, agent) health memory for context benching.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct ContextHealth {
    h: std::collections::HashMap<String, Vec<f64>>,
    skips: Vec<u64>,
}

impl ContextHealth {
    fn score(&self, task: &Task, a: usize) -> f64 {
        if task.features.is_empty() {
            return 1.0;
        }
        let s: f64 = task.features.iter().map(|f| self.h.get(f).and_then(|r| r.get(a)).copied().unwrap_or(1.0)).sum();
        s / task.features.len() as f64
    }

    /// Filter `eligible` by context health; never returns an empty list.
    fn filter(&mut self, cfg: &ContextBenchConfig, task: &Task, eligible: Vec<usize>) -> Vec<usize> {
        let n = eligible.iter().copied().max().map_or(0, |m| m + 1);
        if self.skips.len() < n {
            self.skips.resize(n, 0);
        }
        let scores: Vec<f64> = eligible.iter().map(|&a| self.score(task, a)).collect();
        let best = scores.iter().cloned().fold(0.0_f64, f64::max);
        let thr = cfg.threshold * best;
        let mut out = Vec::with_capacity(eligible.len());
        for (&a, &s) in eligible.iter().zip(&scores) {
            if s >= thr {
                out.push(a);
                continue;
            }
            self.skips[a] += 1;
            if cfg.retry_every > 0 && self.skips[a] >= cfg.retry_every {
                self.skips[a] = 0;
                out.push(a);
            }
        }
        if out.is_empty() { eligible } else { out }
    }

    fn observe(&mut self, task: &Task, members: &[usize], reward: f64, alpha: f64, n_agents: usize) {
        for f in &task.features {
            let row = self.h.entry(f.clone()).or_insert_with(|| vec![1.0; n_agents]);
            if row.len() < n_agents {
                row.resize(n_agents, 1.0);
            }
            for &a in members {
                row[a] += alpha * (reward - row[a]);
            }
        }
    }
}

/// Default per-slot helper exploration probability.
pub const DEFAULT_HELPER_EPSILON: f64 = 0.05;
/// EMA rate of the per-leader team-reward baseline used by edge learning.
pub const EDGE_BASELINE_ALPHA: f64 = 0.1;
/// Initial per-leader team-reward baseline.
pub const INIT_EDGE_BASELINE: f64 = 0.5;

/// Learned-value gate for helper recruitment (sparse activation).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateConfig {
    /// A helper must raise expected reward over the baseline by more than this.
    pub min_gain: f64,
    /// Probability of recruiting a helper even though the gate says no.
    pub explore: f64,
    /// EMA rate for the value estimates (running mean until `1/alpha` samples).
    pub value_alpha: f64,
    /// Observations needed (solo per leader, team per pair) before trusting them.
    pub min_obs: u32,
}

impl Default for GateConfig {
    fn default() -> Self {
        GateConfig { min_gain: 0.02, explore: 0.05, value_alpha: 0.1, min_obs: 3 }
    }
}

/// Per-leader solo reward and per-(leader, helper) team reward estimates.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct GateState {
    solo: Vec<f64>,
    n_solo: Vec<u32>,
    with: Vec<Vec<f64>>,
    n_with: Vec<Vec<u32>>,
}

fn ema(v: &mut f64, n: &mut u32, x: f64, alpha: f64) {
    *n = n.saturating_add(1);
    let a = alpha.max(1.0 / *n as f64);
    *v += a * (x - *v);
}

impl GateState {
    fn ensure(&mut self, n: usize) {
        if self.solo.len() >= n && self.with.len() >= n {
            return;
        }
        self.solo.resize(n, 0.0);
        self.n_solo.resize(n, 0);
        self.with.resize(n, Vec::new());
        self.n_with.resize(n, Vec::new());
        for r in self.with.iter_mut() {
            r.resize(n, 0.0);
        }
        for r in self.n_with.iter_mut() {
            r.resize(n, 0);
        }
    }

    fn observe(&mut self, members: &[usize], reward: f64, alpha: f64) {
        let l = members[0];
        if members.len() == 1 {
            ema(&mut self.solo[l], &mut self.n_solo[l], reward, alpha);
            return;
        }
        for &h in &members[1..] {
            ema(&mut self.with[l][h], &mut self.n_with[l][h], reward, alpha);
        }
    }

    /// Recruitment priority of `h` for leader `l`: 0 = measured and clears the
    /// solo baseline, 1 = not yet measured, 2 = measured and does not help.
    fn rank(&self, cfg: &GateConfig, l: usize, h: usize) -> u8 {
        if self.n_solo[l] < cfg.min_obs || self.n_with[l][h] < cfg.min_obs {
            1
        } else if self.with[l][h] - self.solo[l] > cfg.min_gain {
            0
        } else {
            2
        }
    }

    /// Does helper `h` clear the gate for leader `l`, given helpers already in `team[1..]`?
    fn admits(&self, cfg: &GateConfig, team: &[usize], h: usize) -> bool {
        let l = team[0];
        if self.n_solo[l] < cfg.min_obs {
            return false; // measure the solo arm first
        }
        if self.n_with[l][h] < cfg.min_obs {
            return true; // optimistic: unmeasured pair
        }
        let mut base = self.solo[l];
        for &i in &team[1..] {
            if self.n_with[l][i] >= cfg.min_obs {
                base = base.max(self.with[l][i]);
            }
        }
        self.with[l][h] - base > cfg.min_gain
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshConfig {
    pub edge_eta: f64,
    pub edge_decay: f64,
    pub health_alpha: f64,
    pub health_threshold: f64,
    pub retry_every: u64,
    pub team_size: usize,
}

impl Default for MeshConfig {
    fn default() -> Self {
        MeshConfig {
            edge_eta: 0.2,
            edge_decay: 0.01,
            health_alpha: 0.2,
            health_threshold: 0.3,
            retry_every: 20,
            team_size: 1,
        }
    }
}

fn default_helper_eps() -> f64 {
    DEFAULT_HELPER_EPSILON
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuralMesh<R: Router> {
    pub cfg: MeshConfig,
    /// Per-helper-slot exploration probability.
    #[serde(default = "default_helper_eps")]
    pub helper_epsilon: f64,
    leader: R,
    n_agents: usize,
    /// Symmetric edge matrix, row-major `n_agents * n_agents`.
    edges: Vec<Vec<f64>>,
    health: Vec<f64>,
    /// Route counter value at which a node was benched (None = not benched).
    benched_since: Vec<Option<u64>>,
    /// Per-leader EMA of team rewards.
    edge_baseline: Vec<f64>,
    routes: u64,
    /// Optional alternative edge rule (hatcher-style asymmetric trust). When set,
    /// it replaces the co-activation edges for both team assembly and learning.
    #[serde(default)]
    trust: Option<crate::variants::TrustEdges>,
    /// Optional gated recruitment (see module docs). `None` = fill up to `k`.
    #[serde(default)]
    pub gate: Option<GateConfig>,
    #[serde(default)]
    gate_state: GateState,
    #[serde(default)]
    last_gated: bool,
    #[serde(default)]
    team_size_sum: u64,
    #[serde(default)]
    teams_routed: u64,
    /// Opt-in fault policy. `None` = original behaviour.
    #[serde(default)]
    pub fault: Option<FaultConfig>,
    #[serde(default)]
    fail_streak: Vec<u32>,
    #[serde(default)]
    retry_mult: Vec<f64>,
    #[serde(default)]
    probation: Vec<bool>,
    /// Opt-in context-specific benching. `None` = original behaviour.
    #[serde(default)]
    pub context_bench: Option<ContextBenchConfig>,
    #[serde(default)]
    ctx_health: ContextHealth,
}

impl<R: Router> NeuralMesh<R> {
    pub fn new(n_agents: usize, leader: R, cfg: MeshConfig) -> Self {
        let mut m = NeuralMesh {
            cfg,
            helper_epsilon: DEFAULT_HELPER_EPSILON,
            leader,
            n_agents: 0,
            edges: Vec::new(),
            health: Vec::new(),
            benched_since: Vec::new(),
            edge_baseline: Vec::new(),
            routes: 0,
            trust: None,
            gate: None,
            gate_state: GateState::default(),
            last_gated: false,
            fault: None,
            fail_streak: Vec::new(),
            retry_mult: Vec::new(),
            probation: Vec::new(),
            context_bench: None,
            ctx_health: ContextHealth::default(),
            team_size_sum: 0,
            teams_routed: 0,
        };
        m.ensure_agents(n_agents);
        m
    }

    pub fn n_agents(&self) -> usize {
        self.n_agents
    }

    /// Grow internal tables to at least `n` agents.
    pub fn ensure_agents(&mut self, n: usize) {
        // tables of the opt-in fault policy may be missing in old serialized states
        let m = n.max(self.n_agents);
        self.fail_streak.resize(m, 0);
        self.retry_mult.resize(m, 1.0);
        self.probation.resize(m, false);
        if n <= self.n_agents {
            return;
        }
        for row in self.edges.iter_mut() {
            row.resize(n, 0.0);
        }
        self.edges.resize(n, vec![0.0; n]);
        self.health.resize(n, 1.0);
        self.benched_since.resize(n, None);
        self.fail_streak.resize(n, 0);
        self.retry_mult.resize(n, 1.0);
        self.probation.resize(n, false);
        self.edge_baseline.resize(n, INIT_EDGE_BASELINE);
        if let Some(t) = self.trust.as_mut() {
            t.ensure_agents(n);
        }
        self.n_agents = n;
    }

    /// Switch the edge rule to asymmetric trust edges (see `variants::TrustEdges`).
    pub fn with_trust_edges(mut self, cfg: crate::variants::TrustEdgesConfig) -> Self {
        self.trust = Some(crate::variants::TrustEdges::new(self.n_agents, cfg));
        self
    }

    /// Enable the faster bench/recover policy (opt-in; default behaviour unchanged).
    pub fn with_fault(mut self, f: FaultConfig) -> Self {
        self.fault = Some(f);
        self
    }

    /// Enable context-specific benching (opt-in; default behaviour unchanged).
    pub fn with_context_bench(mut self, c: ContextBenchConfig) -> Self {
        self.context_bench = Some(c);
        self
    }

    /// Context health of agent `a` for `task` (mean per-feature health; 1.0 if unseen).
    pub fn context_health(&self, task: &Task, a: usize) -> f64 {
        self.ctx_health.score(task, a)
    }

    /// Enable gated recruitment: `k` becomes an upper bound on team size.
    pub fn with_gate(mut self, gate: GateConfig) -> Self {
        self.gate = Some(gate);
        self
    }

    /// True if the last route stopped recruiting because a helper failed the gate.
    pub fn last_team_was_gated(&self) -> bool {
        self.last_gated
    }

    /// Mean size of all teams routed so far (1.0 before any route).
    pub fn mean_team_size(&self) -> f64 {
        if self.teams_routed == 0 { 1.0 } else { self.team_size_sum as f64 / self.teams_routed as f64 }
    }

    fn count_team(&mut self, n: usize) {
        self.team_size_sum += n as u64;
        self.teams_routed += 1;
    }

    /// Gate decision for helper `h` (always true when no gate is configured).
    fn admit(&mut self, team: &[usize], h: usize, rng: &mut Rng) -> bool {
        let Some(cfg) = self.gate.clone() else { return true };
        self.gate_state.ensure(self.n_agents);
        rng.f64() < cfg.explore || self.gate_state.admits(&cfg, team, h)
    }

    /// Which edge rule is active: "co-activation" or "trust".
    pub fn edge_rule(&self) -> &'static str {
        if self.trust.is_some() { "trust" } else { "co-activation" }
    }

    /// Single points of failure in the current edge graph: (bridges, articulation points).
    /// Only available with trust edges; returns None for co-activation.
    pub fn fragility(&self, threshold: f64) -> Option<(usize, usize)> {
        self.trust.as_ref().map(|t| t.fragility(threshold))
    }

    pub fn leader(&self) -> &R {
        &self.leader
    }

    pub fn leader_mut(&mut self) -> &mut R {
        &mut self.leader
    }

    pub fn routes(&self) -> u64 {
        self.routes
    }

    /// Co-activation edge strength between `a` and `b` (0 if unknown, 0 for a == b).
    pub fn edge(&self, a: usize, b: usize) -> f64 {
        if let Some(t) = self.trust.as_ref() {
            return if a == b { 0.0 } else { t.edge(a, b) };
        }
        if a == b {
            return 0.0;
        }
        self.edges.get(a).and_then(|r| r.get(b)).copied().unwrap_or(0.0)
    }

    /// Health EMA of node `a` (1.0 if unknown).
    pub fn health(&self, a: usize) -> f64 {
        self.health.get(a).copied().unwrap_or(1.0)
    }

    /// Whether node `a` is currently benched.
    pub fn is_benched(&self, a: usize) -> bool {
        self.benched_since.get(a).is_some_and(|b| b.is_some())
    }

    /// Route with the configured `team_size`.
    pub fn route(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> Vec<usize> {
        let k = self.cfg.team_size;
        self.route_k(task, candidates, rng, k)
    }

    /// Route a team of at most `k` (>= 1) agents, leader first.
    /// Returns an empty team only if `candidates` is empty.
    pub fn route_k(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng, k: usize) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let k = k.max(1);
        if let Some(&mx) = candidates.iter().max() {
            self.ensure_agents(mx + 1);
        }
        // Deduplicate, keep order.
        let mut cands: Vec<usize> = Vec::with_capacity(candidates.len());
        for &c in candidates {
            if !cands.contains(&c) {
                cands.push(c);
            }
        }
        let now = self.routes;
        self.routes += 1;
        // Benching is relative to the healthiest candidate, so a uniformly hard
        // workload (every node's reward EMA is low) does not bench everyone,
        // while a node that fails always falls far below its peers.
        let best_h = cands.iter().map(|&c| self.health[c]).fold(0.0_f64, f64::max);
        let thr = self.cfg.health_threshold * best_h;
        let mut eligible: Vec<usize> = Vec::with_capacity(cands.len());
        for &c in &cands {
            if self.health[c] >= thr {
                self.benched_since[c] = None;
                eligible.push(c);
                continue;
            }
            match self.benched_since[c] {
                None => self.benched_since[c] = Some(now),
                Some(since) if now.saturating_sub(since) >= (self.cfg.retry_every as f64 * self.retry_mult[c]).ceil() as u64 => {
                    // Probation: give it exactly the threshold and let it compete.
                    self.probation[c] = true;
                    self.health[c] = thr;
                    self.benched_since[c] = None;
                    eligible.push(c);
                }
                Some(_) => {}
            }
        }
        if eligible.is_empty() {
            let best = cands.iter().map(|&c| self.health[c]).fold(f64::NEG_INFINITY, f64::max);
            eligible = cands.iter().copied().filter(|&c| self.health[c] >= best).collect();
        }
        if let Some(cb) = self.context_bench.clone() {
            eligible = self.ctx_health.filter(&cb, task, eligible);
        }

        let leader = self.leader.select(task, &eligible, rng);
        let mut team = vec![leader];
        self.last_gated = false;
        if k == 1 {
            self.count_team(1);
            return team;
        }
        let mut rest: Vec<usize> = eligible.iter().copied().filter(|&c| c != leader).collect();
        // Random tie-break: shuffle, then stable sort by edge strength (desc).
        for i in (1..rest.len()).rev() {
            let j = rng.below(i + 1);
            rest.swap(i, j);
        }
        rest.sort_by(|&a, &b| self.edge(leader, b).total_cmp(&self.edge(leader, a)));
        if let Some(cfg) = self.gate.clone() {
            // Stable: proven helpers first, then unmeasured, then known-harmful; edge order within.
            self.gate_state.ensure(self.n_agents);
            rest.sort_by_key(|&h| self.gate_state.rank(&cfg, leader, h));
        }
        while team.len() < k && !rest.is_empty() {
            let idx = if rng.f64() < self.helper_epsilon { rng.below(rest.len()) } else { 0 };
            if !self.admit(&team, rest[idx], rng) {
                self.last_gated = true;
                break;
            }
            team.push(rest.remove(idx));
        }
        self.count_team(team.len());
        team
    }

    /// Feedback for a routed team (leader first). Updates the leader router
    /// (for `team[0]`), co-activation edges (teams of size >= 2) and health of
    /// every member.
    pub fn feedback(&mut self, task: &Task, team: &[usize], reward: f64) {
        if team.is_empty() || !reward.is_finite() {
            return;
        }
        let reward = reward.clamp(0.0, 1.0);
        if let Some(&mx) = team.iter().max() {
            self.ensure_agents(mx + 1);
        }
        let leader = team[0];
        self.leader.update(task, leader, reward);

        // Unique members.
        let mut members: Vec<usize> = Vec::with_capacity(team.len());
        for &a in team {
            if !members.contains(&a) {
                members.push(a);
            }
        }

        if members.len() >= 2 && self.trust.is_some() {
            if let Some(t) = self.trust.as_mut() {
                t.record_team(&members, reward);
            }
        } else if members.len() >= 2 {
            let keep = 1.0 - self.cfg.edge_decay;
            for row in self.edges.iter_mut() {
                for e in row.iter_mut() {
                    *e *= keep;
                }
            }
            let b = self.edge_baseline[leader];
            let delta = self.cfg.edge_eta * (reward - b);
            for (i, &a) in members.iter().enumerate() {
                for &c in &members[i + 1..] {
                    let v = (self.edges[a][c] + delta).clamp(-1.0, 1.0);
                    self.edges[a][c] = v;
                    self.edges[c][a] = v;
                }
            }
            self.edge_baseline[leader] = b + EDGE_BASELINE_ALPHA * (reward - b);
        }

        if let Some(alpha) = self.gate.as_ref().map(|g| g.value_alpha) {
            self.gate_state.ensure(self.n_agents);
            self.gate_state.observe(&members, reward, alpha);
        }

        if let Some(alpha) = self.context_bench.as_ref().map(|c| c.alpha) {
            let n = self.n_agents;
            self.ctx_health.observe(task, &members, reward, alpha, n);
        }

        let al = self.cfg.health_alpha;
        for &a in &members {
            self.health[a] += al * (reward - self.health[a]);
        }
        if let Some(f) = self.fault.clone() {
            for &a in &members {
                let failed = reward < f.fail_below;
                if self.probation[a] {
                    self.probation[a] = false;
                    self.retry_mult[a] =
                        if failed { (self.retry_mult[a] * f.backoff.max(1.0)).min(f.max_mult.max(1.0)) } else { 1.0 };
                }
                if failed {
                    self.fail_streak[a] += 1;
                    if f.streak > 0 && self.fail_streak[a] >= f.streak {
                        self.health[a] = 0.0;
                        self.fail_streak[a] = 0;
                    }
                } else {
                    self.fail_streak[a] = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::ThompsonRouter;
    use crate::hebbian::{HebbianConfig, HebbianRouter};

    fn task(i: u64) -> Task {
        let ty = ["a", "b", "c"][(i % 3) as usize];
        Task { id: i, task_type: Some(ty.into()), features: vec![format!("kw-{ty}")] }
    }

    fn p_success(t: &Task, a: usize) -> f64 {
        let ty = t.task_type.as_deref().unwrap();
        let best = match ty { "a" => 0, "b" => 1, _ => 2 };
        if a == best { 0.97 } else { 0.9 }
    }

    fn check_equivalence<R: Router + Clone>(make: impl Fn() -> R, cfg: MeshConfig) {
        let mut bare = make();
        let mut mesh = NeuralMesh::new(4, make(), cfg);
        let (mut r1, mut r2) = (Rng::new(42), Rng::new(42));
        let (mut env1, mut env2) = (Rng::new(7), Rng::new(7));
        let cands = [0, 1, 2, 3];
        for i in 0..3000 {
            let t = task(i);
            let a = bare.select(&t, &cands, &mut r1);
            let team = mesh.route(&t, &cands, &mut r2);
            assert_eq!(team, vec![a], "step {i}");
            let ra = if env1.bernoulli(p_success(&t, a)) { 1.0 } else { 0.0 };
            let rb = if env2.bernoulli(p_success(&t, team[0])) { 1.0 } else { 0.0 };
            bare.update(&t, a, ra);
            mesh.feedback(&t, &team, rb);
        }
        // identical rng streams afterwards too
        assert_eq!(r1.next_u64(), r2.next_u64());
    }

    #[test]
    fn team_size_one_reproduces_leader_router_exactly() {
        // Benching disabled: exact for any reward stream.
        let cfg = MeshConfig { team_size: 1, health_threshold: 0.0, ..MeshConfig::default() };
        check_equivalence(|| HebbianRouter::new(4, HebbianConfig::default()), cfg.clone());
        check_equivalence(|| ThompsonRouter::new(4), cfg);
        // Default config in a healthy world (no node ever benched): still exact.
        check_equivalence(|| HebbianRouter::new(4, HebbianConfig::default()), MeshConfig::default());
    }

    #[test]
    fn fault_policy_streak_benches_fast_and_default_is_unchanged() {
        let t = task(0);
        let mk = |fault: Option<FaultConfig>| {
            let cfg = MeshConfig { team_size: 1, retry_every: 10, ..MeshConfig::default() };
            let mut m = NeuralMesh::new(3, HebbianRouter::new(3, HebbianConfig::default()), cfg);
            if let Some(f) = fault {
                m = m.with_fault(f);
            }
            m
        };
        // Two failures: default health EMA does not bench, a streak-2 policy does.
        let mut plain = mk(None);
        let mut fast = mk(Some(FaultConfig { streak: 2, ..FaultConfig::default() }));
        let mut rng = Rng::new(3);
        for m in [&mut plain, &mut fast] {
            m.feedback(&t, &[2], 0.0);
            m.feedback(&t, &[2], 0.0);
        }
        let _ = plain.route(&t, &[0, 1, 2], &mut rng);
        let _ = fast.route(&t, &[0, 1, 2], &mut rng);
        assert!(!plain.is_benched(2));
        assert!(fast.is_benched(2));
        // An all-default FaultConfig behaves exactly like no policy.
        let mut a = mk(None);
        let mut b = mk(Some(FaultConfig::default()));
        let (mut r1, mut r2) = (Rng::new(9), Rng::new(9));
        for i in 0..200u64 {
            let ti = task(i);
            let ta = a.route(&ti, &[0, 1, 2], &mut r1);
            let tb = b.route(&ti, &[0, 1, 2], &mut r2);
            assert_eq!(ta, tb);
            let r = if ta[0] == 2 { 0.0 } else { 0.8 };
            a.feedback(&ti, &ta, r);
            b.feedback(&ti, &tb, r);
        }
    }

    #[test]
    fn fault_backoff_lengthens_retry_after_failed_probation() {
        let t = task(0);
        let cfg = MeshConfig { team_size: 1, retry_every: 5, ..MeshConfig::default() };
        let mut m = NeuralMesh::new(3, HebbianRouter::new(3, HebbianConfig::default()), cfg)
            .with_fault(FaultConfig { streak: 2, backoff: 2.0, ..FaultConfig::default() });
        let mut rng = Rng::new(1);
        m.feedback(&t, &[2], 0.0);
        m.feedback(&t, &[2], 0.0);
        for _ in 0..5 {
            let _ = m.route(&t, &[0, 1, 2], &mut rng);
        }
        let _ = m.route(&t, &[2, 0], &mut rng); // probation
        m.feedback(&t, &[2], 0.0); // probation failed -> interval doubles to 10
        let _ = m.route(&t, &[0, 1, 2], &mut rng); // benched again
        assert!(m.is_benched(2));
        for _ in 0..9 {
            let _ = m.route(&t, &[0, 1, 2], &mut rng);
            assert!(m.is_benched(2));
        }
    }

    #[test]
    fn skips_repeatedly_failing_node_then_retries_it() {
        let cfg = MeshConfig { team_size: 1, retry_every: 10, ..MeshConfig::default() };
        let mut mesh = NeuralMesh::new(3, HebbianRouter::new(3, HebbianConfig::default()), cfg);
        let mut rng = Rng::new(1);
        let t = task(0);
        // Node 2 fails repeatedly.
        for _ in 0..10 {
            mesh.feedback(&t, &[2], 0.0);
        }
        assert!(mesh.health(2) < 0.3);
        // While benched (routes 0..9) it must never be chosen.
        for i in 0..10 {
            let team = mesh.route(&t, &[0, 1, 2], &mut rng);
            assert!(!team.contains(&2), "benched node routed at {i}");
            assert!(mesh.is_benched(2));
        }
        // After retry_every routes on the bench it is back on probation.
        let _ = mesh.route(&t, &[2, 0], &mut rng);
        assert!(!mesh.is_benched(2), "node should be on probation");
        assert!((mesh.health(2) - 0.3).abs() < 1e-12);
        // Probation success restores it; failure re-benches it on the next route.
        mesh.feedback(&t, &[2], 0.0);
        let _ = mesh.route(&t, &[0, 1, 2], &mut rng);
        assert!(mesh.is_benched(2));
        // Recovery path: wait out the bench, then succeed.
        for _ in 0..10 {
            let _ = mesh.route(&t, &[0, 1], &mut rng);
        }
        let _ = mesh.route(&t, &[2], &mut rng);
        for _ in 0..5 {
            mesh.feedback(&t, &[2], 1.0);
        }
        assert!(mesh.health(2) > 0.7);
        let _ = mesh.route(&t, &[0, 1, 2], &mut rng);
        assert!(!mesh.is_benched(2));
        // Never an empty team: a dead node that is the only candidate is still used.
        for _ in 0..20 {
            mesh.feedback(&t, &[2], 0.0);
        }
        assert_eq!(mesh.route(&t, &[2], &mut rng), vec![2]);
    }

    #[test]
    fn traffic_flows_away_from_failed_leader() {
        // Agent 0 is best, then fails completely for a window.
        let cfg = MeshConfig { team_size: 1, ..MeshConfig::default() };
        let mut mesh = NeuralMesh::new(3, HebbianRouter::new(3, HebbianConfig::default()), cfg);
        let mut rng = Rng::new(3);
        let mut env = Rng::new(4);
        let t = Task { id: 0, task_type: Some("x".into()), features: vec![] };
        let p = [0.95, 0.7, 0.3];
        for _ in 0..1000 {
            let team = mesh.route(&t, &[0, 1, 2], &mut rng);
            let r = if env.bernoulli(p[team[0]]) { 1.0 } else { 0.0 };
            mesh.feedback(&t, &team, r);
        }
        let mut ok = 0;
        for _ in 0..300 {
            let team = mesh.route(&t, &[0, 1, 2], &mut rng);
            let r = if team[0] == 0 { 0.0 } else if env.bernoulli(p[team[0]]) { 1.0 } else { 0.0 };
            mesh.feedback(&t, &team, r);
            ok += r as i32;
        }
        assert!(ok > 150, "success during failure window too low: {ok}/300");
    }

    #[test]
    fn edge_strengthens_for_synergistic_pair() {
        let cfg = MeshConfig { team_size: 2, ..MeshConfig::default() };
        let mut mesh = NeuralMesh::new(4, HebbianRouter::new(4, HebbianConfig::default()), cfg);
        let mut rng = Rng::new(9);
        let mut env = Rng::new(10);
        let t = Task { id: 0, task_type: Some("team".into()), features: vec![] };
        // Hidden synergy: {0,1} succeed 90%, any other pair 30%.
        let mut synergy_teams = 0;
        for i in 0..3000 {
            let team = mesh.route(&t, &[0, 1, 2, 3], &mut rng);
            assert!(!team.is_empty() && team.len() <= 2);
            if team.len() == 2 {
                assert_ne!(team[0], team[1]);
            }
            let syn = team.contains(&0) && team.contains(&1);
            let r = if env.bernoulli(if syn { 0.9 } else { 0.3 }) { 1.0 } else { 0.0 };
            mesh.feedback(&t, &team, r);
            if i >= 2500 && syn {
                synergy_teams += 1;
            }
        }
        assert!(mesh.edge(0, 1) > 0.0, "edge(0,1) = {}", mesh.edge(0, 1));
        assert_eq!(mesh.edge(0, 1), mesh.edge(1, 0));
        for other in [2, 3] {
            assert!(mesh.edge(0, 1) > mesh.edge(0, other));
            assert!(mesh.edge(0, 1) > mesh.edge(1, other));
        }
        assert!(synergy_teams > 350, "synergistic team used {synergy_teams}/500");
    }

    #[test]
    fn edge_update_direction_and_clamp() {
        let cfg = MeshConfig { team_size: 2, edge_eta: 1.0, edge_decay: 0.0, ..MeshConfig::default() };
        let mut mesh = NeuralMesh::new(3, HebbianRouter::new(3, HebbianConfig::default()), cfg);
        let t = Task::default();
        mesh.feedback(&t, &[0, 1], 1.0);
        assert!(mesh.edge(0, 1) > 0.0);
        mesh.feedback(&t, &[0, 2], 0.0);
        assert!(mesh.edge(0, 2) < 0.0);
        for _ in 0..50 {
            mesh.feedback(&t, &[1, 2], 1.0);
        }
        assert!(mesh.edge(1, 2) <= 1.0);
        assert_eq!(mesh.edge(1, 1), 0.0);
    }

    #[test]
    fn never_empty_and_grows() {
        let mut mesh = NeuralMesh::new(1, HebbianRouter::new(1, HebbianConfig::default()), MeshConfig::default());
        let mut rng = Rng::new(0);
        assert!(mesh.route(&Task::default(), &[], &mut rng).is_empty());
        let team = mesh.route_k(&Task::default(), &[5], &mut rng, 3);
        assert_eq!(team, vec![5]);
        assert_eq!(mesh.n_agents(), 6);
        for a in 0..6 {
            mesh.feedback(&Task::default(), &[a], 0.0);
            mesh.feedback(&Task::default(), &[a], 0.0);
            mesh.feedback(&Task::default(), &[a], 0.0);
            mesh.feedback(&Task::default(), &[a], 0.0);
            mesh.feedback(&Task::default(), &[a], 0.0);
            mesh.feedback(&Task::default(), &[a], 0.0);
        }
        let team = mesh.route_k(&Task::default(), &[0, 1, 2], &mut rng, 2);
        assert!(!team.is_empty());
    }

    #[test]
    fn serde_roundtrip() {
        let mut mesh = NeuralMesh::new(3, HebbianRouter::new(3, HebbianConfig::default()), MeshConfig::default());
        mesh.feedback(&task(1), &[0, 1], 1.0);
        let js = serde_json::to_string(&mesh).unwrap();
        let back: NeuralMesh<HebbianRouter> = serde_json::from_str(&js).unwrap();
        assert_eq!(back, mesh);
    }

    #[test]
    fn trust_edge_rule_learns_synergy_and_reports_fragility() {
        use crate::hebbian::{HebbianConfig, HebbianRouter};
        use crate::variants::TrustEdgesConfig;
        let mut m = NeuralMesh::new(4, HebbianRouter::new(4, HebbianConfig::default()), MeshConfig { team_size: 2, ..MeshConfig::default() })
            .with_trust_edges(TrustEdgesConfig::default());
        assert_eq!(m.edge_rule(), "trust");
        let t = Task { id: 0, task_type: Some("x".into()), features: vec!["x".into()] };
        for _ in 0..200 {
            m.feedback(&t, &[0, 1], 1.0); // reliable pair
            m.feedback(&t, &[0, 2], 0.0); // failing pair
        }
        assert!(m.edge(0, 1) > 0.5, "reliable pair edge {}", m.edge(0, 1));
        assert!(m.edge(0, 2) < 0.05, "failing pair edge {}", m.edge(0, 2));
        assert_eq!(m.edge(1, 1), 0.0);
        assert!(m.fragility(0.5).is_some());
        let js = serde_json::to_string(&m).unwrap();
        let back: NeuralMesh<HebbianRouter> = serde_json::from_str(&js).unwrap();
        assert_eq!(back.edge_rule(), "trust");
    }

    #[test]
    fn old_state_without_trust_field_still_loads() {
        use crate::hebbian::{HebbianConfig, HebbianRouter};
        let m = NeuralMesh::new(2, HebbianRouter::new(2, HebbianConfig::default()), MeshConfig::default());
        let mut v: serde_json::Value = serde_json::to_value(&m).unwrap();
        v.as_object_mut().unwrap().remove("trust");
        let back: NeuralMesh<HebbianRouter> = serde_json::from_value(v).unwrap();
        assert_eq!(back.edge_rule(), "co-activation");
    }

    // ---- gated recruitment ----

    /// Always picks the first candidate, so tests control the leader.
    #[derive(Clone, Debug)]
    struct FirstRouter;
    impl Router for FirstRouter {
        fn name(&self) -> String { "first".into() }
        fn select(&mut self, _t: &Task, c: &[usize], _r: &mut Rng) -> usize { c[0] }
        fn update(&mut self, _t: &Task, _a: usize, _r: f64) {}
    }

    fn gated_mesh(k: usize, trust: bool) -> NeuralMesh<FirstRouter> {
        let cfg = MeshConfig { team_size: k, health_threshold: 0.0, ..MeshConfig::default() };
        let m = NeuralMesh::new(4, FirstRouter, cfg).with_gate(GateConfig::default());
        if trust { m.with_trust_edges(crate::variants::TrustEdgesConfig::default()) } else { m }
    }

    /// Pair {0,1} exactly: 0.9; any other team: 0.3; solo: 0.55.
    fn toy_reward(team: &[usize], env: &mut Rng) -> f64 {
        let p = match team.len() {
            1 => 0.55,
            2 if team.contains(&0) && team.contains(&1) => 0.9,
            _ => 0.3,
        };
        if env.bernoulli(p) { 1.0 } else { 0.0 }
    }

    fn run_toy(trust: bool) {
        let mut mesh = gated_mesh(3, trust);
        let (mut rng, mut env) = (Rng::new(5), Rng::new(6));
        let t = Task::default();
        let (mut l0, mut l0_syn, mut l2, mut l2_size) = (0, 0, 0, 0);
        for i in 0..6000 {
            let cands: [usize; 4] = if i % 2 == 0 { [0, 1, 2, 3] } else { [2, 0, 1, 3] };
            let team = mesh.route(&t, &cands, &mut rng);
            if i >= 4000 {
                if team[0] == 0 {
                    l0 += 1;
                    l0_syn += (team.len() == 2 && team[1] == 1) as i32;
                } else {
                    l2 += 1;
                    l2_size += team.len();
                }
            }
            let r = toy_reward(&team, &mut env);
            mesh.feedback(&t, &team, r);
        }
        assert!(l0_syn as f64 / l0 as f64 > 0.85, "leader 0 recruited 1 only {l0_syn}/{l0}");
        let mean2 = l2_size as f64 / l2 as f64;
        assert!(mean2 < 1.25, "leader 2 mean team size {mean2}");
        assert!(mesh.mean_team_size() < 2.0);
    }

    #[test]
    fn gate_recruits_synergistic_helper_and_goes_solo_otherwise() {
        run_toy(false);
    }

    #[test]
    fn gate_works_with_trust_edges() {
        run_toy(true);
    }

    #[test]
    fn gate_solo_only_workload_converges_to_one_plus_explore() {
        let mut mesh = gated_mesh(3, false);
        let (mut rng, mut env) = (Rng::new(1), Rng::new(2));
        let t = Task::default();
        let (mut sum, mut n) = (0usize, 0usize);
        for i in 0..5000 {
            let team = mesh.route(&t, &[0, 1, 2, 3], &mut rng);
            if i >= 3000 {
                sum += team.len();
                n += 1;
            }
            // Solo 0.8, any team 0.6: helpers never help.
            let r = if env.bernoulli(if team.len() == 1 { 0.8 } else { 0.6 }) { 1.0 } else { 0.0 };
            mesh.feedback(&t, &team, r);
        }
        let mean = sum as f64 / n as f64;
        assert!(mean < 1.15, "mean team size {mean}");
        assert!(mean >= 1.0);
        assert!(mesh.mean_team_size() >= 1.0);
    }

    #[test]
    fn gate_serde_roundtrip_and_old_state_loads() {
        let m = NeuralMesh::new(3, HebbianRouter::new(3, HebbianConfig::default()), MeshConfig { team_size: 2, ..MeshConfig::default() })
            .with_gate(GateConfig { min_gain: 0.1, ..GateConfig::default() });
        let js = serde_json::to_string(&m).unwrap();
        let back: NeuralMesh<HebbianRouter> = serde_json::from_str(&js).unwrap();
        assert_eq!(back, m);
        // Old state without any gate field loads with the gate off and works.
        let mut v: serde_json::Value = serde_json::to_value(&m).unwrap();
        for key in ["gate", "gate_state", "last_gated", "team_size_sum", "teams_routed"] {
            v.as_object_mut().unwrap().remove(key);
        }
        let mut old: NeuralMesh<HebbianRouter> = serde_json::from_value(v).unwrap();
        assert!(old.gate.is_none());
        let mut rng = Rng::new(4);
        let team = old.route(&task(0), &[0, 1, 2], &mut rng);
        assert_eq!(team.len(), 2);
        old.feedback(&task(0), &team, 1.0);
        assert!(!old.last_team_was_gated());
    }

    #[test]
    fn gate_off_by_default_fills_to_k() {
        let cfg = MeshConfig { team_size: 3, ..MeshConfig::default() };
        let mut mesh = NeuralMesh::new(4, FirstRouter, cfg);
        let mut rng = Rng::new(0);
        assert_eq!(mesh.route(&task(0), &[0, 1, 2, 3], &mut rng).len(), 3);
        assert_eq!(mesh.mean_team_size(), 3.0);
    }

    fn ftask(f: &str) -> Task {
        Task { id: 0, task_type: None, features: vec![f.to_string()] }
    }

    #[test]
    fn context_bench_is_context_specific() {
        // agent 0 always fails on "sec" tasks but succeeds on "docs" tasks
        let cfg = MeshConfig { team_size: 1, health_threshold: 0.0, ..MeshConfig::default() };
        let mut m = NeuralMesh::new(3, FirstRouter, cfg).with_context_bench(ContextBenchConfig { retry_every: 1000, ..Default::default() });
        let mut rng = Rng::new(1);
        for _ in 0..10 {
            let (s, d) = (ftask("sec"), ftask("docs"));
            let a = m.route(&s, &[0, 1, 2], &mut rng)[0];
            m.feedback(&s, &[a], if a == 0 { 0.0 } else { 1.0 });
            let b = m.route(&d, &[0, 1, 2], &mut rng)[0];
            m.feedback(&d, &[b], 1.0);
        }
        assert_ne!(m.route(&ftask("sec"), &[0, 1, 2], &mut rng)[0], 0, "benched for sec");
        assert_eq!(m.route(&ftask("docs"), &[0, 1, 2], &mut rng)[0], 0, "still used for docs");
        assert!(m.context_health(&ftask("sec"), 0) < 0.5 && m.context_health(&ftask("docs"), 0) > 0.9);
    }

    #[test]
    fn context_bench_probation_and_never_empty() {
        let cfg = MeshConfig { team_size: 1, health_threshold: 0.0, ..MeshConfig::default() };
        let mut m = NeuralMesh::new(2, FirstRouter, cfg).with_context_bench(ContextBenchConfig { retry_every: 3, ..Default::default() });
        let mut rng = Rng::new(1);
        let s = ftask("sec");
        for _ in 0..5 {
            m.feedback(&s, &[0], 0.0);
        }
        let picks: Vec<usize> = (0..6).map(|_| m.route(&s, &[0, 1], &mut rng)[0]).collect();
        assert_eq!(picks.iter().filter(|&&a| a == 0).count(), 2, "probation every 3rd skip: {picks:?}");
        assert_eq!(m.route(&s, &[0], &mut rng), vec![0], "sole candidate is never dropped");
    }

    #[test]
    fn context_bench_off_by_default_and_serde_compatible() {
        let m = NeuralMesh::new(2, FirstRouter, MeshConfig::default());
        assert!(m.context_bench.is_none());
        let mut v: serde_json::Value = serde_json::to_value(NeuralMesh::new(2, ThompsonRouter::new(2), MeshConfig::default())).unwrap();
        v.as_object_mut().unwrap().remove("context_bench");
        v.as_object_mut().unwrap().remove("ctx_health");
        let back: NeuralMesh<ThompsonRouter> = serde_json::from_value(v).unwrap();
        assert!(back.context_bench.is_none());
    }
}
