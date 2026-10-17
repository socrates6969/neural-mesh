# neural-mesh: Hebbian-routed agent mesh — design contract

Goal: route tasks to agents (and assemble agent teams) by learning from outcomes,
and prove — with paired statistics on a fixed simulation — that it beats the
current routers before anything is wired into ruflo or howler-agents.

## Ground rules
- The Hebbian rule is implemented INDEPENDENTLY from the published description
  in qScaleg/hebbian-router's README
- No change ships unless the benchmark shows a statistically significant win
  (95% paired t-CI over >= 30 seeds excludes 0) AND all tests pass.
- Deterministic: every stochastic component takes a seed.

## Router (src/router.rs) — shared trait, do not change its signature
```rust
pub struct Task { pub id: u64, pub task_type: Option<String>, pub features: Vec<String> }
pub trait Router {
    fn name(&self) -> String;
    /// Pick one agent from `candidates` (indices into the agent list, all eligible).
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize;
    /// Outcome feedback for the agent that was chosen. reward in [0,1].
    fn update(&mut self, task: &Task, agent: usize, reward: f64);
}
```
Baselines: KeywordRouter (static feature->agent map), ThompsonRouter (Beta per feature x agent).

## HebbianRouter — three-factor rule (from the public description)
weights w[feature][agent] in [0,1]; score(agent) = sum over task features of w.
select: softmax(score / temperature) with epsilon-greedy exploration.
update (chosen agent only): baseline = EMA(reward);
  delta = eta * pre * (reward - baseline); w = clamp(w*(1-decay) + delta, 0, 1);
  rescale a feature row if its sum exceeds `row_cap`.
Defaults to start from: eta 0.5, epsilon 0.02, temperature 0.05, decay 0.01 —
re-tune on seeds 1-8 only; evaluate on seeds 1-30 (report both).

## NeuralMesh (src/mesh.rs) — the "merge"
Nodes = agents. Two learned structures:
1. Router weights (feature -> agent): who leads a task.
2. Co-activation edges (agent <-> agent), Hebbian: when a team of agents works a
   task, every pair (i,j) gets e_ij += eta_e * (reward - baseline), with decay.
   Team assembly: leader from the router, then add helpers by strongest edges
   from the leader (sparse activation: only top-k fire).
3. Fault tolerance: per-node health EMA; nodes below a threshold are
   temporarily skipped (re-tried on an exploration schedule) and traffic flows
   to the next-strongest path.
Persistence: state serialises to JSON (serde) so a CLI can load/save it between calls.

## Simulation (src/sim.rs) + experiment binary (src/bin/experiment.rs)
World: >= 6 task types, >= 6 agents, hidden success probabilities with two
"trap" types where the keyword map is wrong, keyword noise + confound
keywords, mid-run skill drift, plus (for the mesh) team tasks where some agent
PAIRS have hidden synergy, and node-failure windows (an agent fails always for
a stretch). Metrics: success rate, regret, pre/early-post/late-post drift
windows, team success, success during failure windows.
Output: table + paired per-seed differences (mean ± 95% t-CI, seed wins) and
results/results.json.

## CLI (src/main.rs) — integration surface for ruflo (JS) and howler (Python)
neural-mesh route   --state <file> --features a,b,c [--candidates x,y] [--team k] -> JSON {agent, team, scores}
neural-mesh feedback --state <file> --features a,b,c --agent X [--team X,Y] --reward 0..1
neural-mesh stats   --state <file>
Agents are named strings in the CLI; state file maps names <-> indices.

## Exact module API (so parallel work compiles together)
- `src/hebbian.rs`: `pub struct HebbianConfig { pub eta, epsilon, temperature, decay, baseline_alpha, row_cap: f64 }` (impl Default with the defaults above),
  `pub struct HebbianRouter` with `pub fn new(n_agents: usize, cfg: HebbianConfig) -> Self`, implements `Router`.
  Also `pub fn scores(&self, task: &Task) -> Vec<f64>` (one per agent). Derive Serialize/Deserialize.
- `src/baselines.rs`: `KeywordRouter::new(map: HashMap<String, usize>, n_agents)`, `ThompsonRouter::new(n_agents)`; both implement `Router`.
- `src/mesh.rs`: `pub struct MeshConfig { pub edge_eta, edge_decay, health_alpha, health_threshold: f64, pub retry_every: u64, pub team_size: usize }` (Default),
  `pub struct NeuralMesh<R: Router>` wrapping a leader router, with
  `pub fn new(n_agents: usize, leader: R, cfg: MeshConfig) -> Self`,
  `pub fn route(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> Vec<usize>` (team, leader first, len <= team_size, skips unhealthy nodes),
  `pub fn feedback(&mut self, task: &Task, team: &[usize], reward: f64)` (updates leader router, co-activation edges, health),
  `pub fn edge(&self, a: usize, b: usize) -> f64`, `pub fn health(&self, a: usize) -> f64`.
  team_size = 1 must behave exactly like the leader router alone (test this).
- `src/sim.rs`, `src/stats.rs`, `src/bin/experiment.rs`: simulation, paired stats, experiment runner.
- `src/main.rs`: the `neural-mesh` CLI (bin name `neural-mesh`).
