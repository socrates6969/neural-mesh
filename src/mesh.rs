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
//! RNG: with `team_size == 1` the mesh consumes randomness *only* through the
//! leader router, and while no node is benched the leader sees exactly the
//! caller's candidate list. Hence a mesh with team size 1 makes the same
//! choices as the bare router (whenever no node is benched; set
//! `health_threshold = 0` to disable benching entirely).

use serde::{Deserialize, Serialize};

use crate::router::{Rng, Router, Task};

/// Default per-slot helper exploration probability.
pub const DEFAULT_HELPER_EPSILON: f64 = 0.05;
/// EMA rate of the per-leader team-reward baseline used by edge learning.
pub const EDGE_BASELINE_ALPHA: f64 = 0.1;
/// Initial per-leader team-reward baseline.
pub const INIT_EDGE_BASELINE: f64 = 0.5;

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
        };
        m.ensure_agents(n_agents);
        m
    }

    pub fn n_agents(&self) -> usize {
        self.n_agents
    }

    /// Grow internal tables to at least `n` agents.
    pub fn ensure_agents(&mut self, n: usize) {
        if n <= self.n_agents {
            return;
        }
        for row in self.edges.iter_mut() {
            row.resize(n, 0.0);
        }
        self.edges.resize(n, vec![0.0; n]);
        self.health.resize(n, 1.0);
        self.benched_since.resize(n, None);
        self.edge_baseline.resize(n, INIT_EDGE_BASELINE);
        self.n_agents = n;
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
                Some(since) if now.saturating_sub(since) >= self.cfg.retry_every => {
                    // Probation: give it exactly the threshold and let it compete.
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

        let leader = self.leader.select(task, &eligible, rng);
        let mut team = vec![leader];
        if k == 1 {
            return team;
        }
        let mut rest: Vec<usize> = eligible.iter().copied().filter(|&c| c != leader).collect();
        // Random tie-break: shuffle, then stable sort by edge strength (desc).
        for i in (1..rest.len()).rev() {
            let j = rng.below(i + 1);
            rest.swap(i, j);
        }
        rest.sort_by(|&a, &b| self.edge(leader, b).total_cmp(&self.edge(leader, a)));
        while team.len() < k && !rest.is_empty() {
            let idx = if rng.f64() < self.helper_epsilon { rng.below(rest.len()) } else { 0 };
            team.push(rest.remove(idx));
        }
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

        if members.len() >= 2 {
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

        let al = self.cfg.health_alpha;
        for &a in &members {
            self.health[a] += al * (reward - self.health[a]);
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
}
