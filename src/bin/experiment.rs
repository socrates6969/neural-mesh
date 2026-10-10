//! Benchmark runner v2: evaluates routers and meshes across SIX scenarios with paired
//! statistics, Holm-Bonferroni correction and a cost (agent calls per task) column.
//!
//! Scenarios (all 2000 tasks/run, see src/sim.rs):
//!   fixed       the original hand-designed world
//!   random      per-seed random skills/traps/synergy pairs/drift/failure window
//!   no-synergy  random, but no synergistic pair (teams only cost)
//!   no-traps    random, but the keyword map is right until drift
//!   revealed    random + task_type revealed to the routers
//!   delay25     random + feedback arrives 25 tasks late
//!
//! Protocol:
//! 1. TUNING on seeds 1-8 ONLY. Two selection criteria over the same grid:
//!      tuned  = best mean success over the three randomised scenarios
//!               (random, no-synergy, no-traps);
//!      robust = best mean RANK over all six scenarios (also reports worst-case gain).
//!    Grids are extended outward (x2 / x3 steps) until the optimum is interior or a
//!    hard parameter bound is hit; the final grid and interior flags are printed.
//! 2. EVALUATION on held-out seeds 31-60 per scenario.
//! 3. Holm-Bonferroni over the family { X beats keyword, mesh beats its own leader,
//!    gated beats ungated } (two-sided paired t-test, alpha 0.05) for the tuned set and,
//!    separately, the robust set.
//!
//! Usage: cargo run --release --bin experiment [-- --tasks 2000 --out results/results.json]

use neural_mesh::baselines::{sample_beta, EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::defaults::{recommended_guard, recommended_inner};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::Guarded;
use neural_mesh::mesh::{GateConfig, MeshConfig, NeuralMesh};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, Window, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use neural_mesh::variants::{PgActorConfig, PgActorRouter, RStdpRouter, RStdpRouterConfig, TrustEdges, TrustEdgesConfig};
use serde::Serialize;
use serde_json::json;
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Policies

trait Policy {
    fn assign(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> Vec<usize>;
    fn feedback(&mut self, task: &Task, team: &[usize], reward: f64);
    /// Feedback for a task that failed for an ENVIRONMENT reason (skill-independent). Default = unshielded: an ordinary reward-0 outcome.
    fn feedback_env(&mut self, task: &Task, team: &[usize]) {
        self.feedback(task, team, 0.0);
    }
}

struct Single<R: Router>(R);
impl<R: Router> Policy for Single<R> {
    fn assign(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> Vec<usize> {
        vec![self.0.select(task, cands, rng)]
    }
    fn feedback(&mut self, task: &Task, team: &[usize], reward: f64) {
        self.0.update(task, team[0], reward);
    }
}

struct Mesh<R: Router>(NeuralMesh<R>);
impl<R: Router> Policy for Mesh<R> {
    fn assign(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> Vec<usize> {
        self.0.route(task, cands, rng)
    }
    fn feedback(&mut self, task: &Task, team: &[usize], reward: f64) {
        self.0.feedback(task, team, reward);
    }
}

/// R117 (opt-in failure class): environment failures go to the mesh's dispatch-risk score only (`NeuralMesh::feedback_env`); skill weights, health and
/// benching are untouched. Environment-clean outcomes also decay the leader's risk.
struct ShieldMesh<R: Router>(NeuralMesh<R>);
impl<R: Router> Policy for ShieldMesh<R> {
    fn assign(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> Vec<usize> {
        self.0.route(task, cands, rng)
    }
    fn feedback(&mut self, task: &Task, team: &[usize], reward: f64) {
        self.0.feedback(task, team, reward);
        self.0.feedback_env(team[0], false);
    }
    fn feedback_env(&mut self, _task: &Task, team: &[usize]) {
        self.0.feedback_env(team[0], true);
    }
}

/// Mesh variant whose helper selection uses `TrustEdges` directly (no health/benching, no gate).
const TRUST_HELPER_EPS: f64 = 0.05;
struct TrustMesh<R: Router> {
    leader: R,
    edges: TrustEdges,
    team_size: usize,
}
impl<R: Router> Policy for TrustMesh<R> {
    fn assign(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> Vec<usize> {
        let l = self.leader.select(task, cands, rng);
        let mut team = vec![l];
        if self.team_size > 1 {
            let mut rest: Vec<usize> = self.edges.top_partners(l, cands.len(), cands);
            for _ in 1..self.team_size {
                if rest.is_empty() {
                    break;
                }
                let i = if rng.f64() < TRUST_HELPER_EPS { rng.below(rest.len()) } else { 0 };
                team.push(rest.remove(i));
            }
        }
        team
    }
    fn feedback(&mut self, task: &Task, team: &[usize], reward: f64) {
        self.leader.update(task, team[0], reward);
        if team.len() >= 2 {
            self.edges.record_team(team, reward);
        }
    }
}

/// Team baseline: tuned Hebbian leader + a per-(leader, helper) Beta-Thompson helper bandit
/// (no co-activation edges, no health, no gate). Team size is fixed to `k`.
struct HelperBandit<R: Router> {
    leader: R,
    ab: Vec<Vec<(f64, f64)>>,
    k: usize,
}
impl<R: Router> Policy for HelperBandit<R> {
    fn assign(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> Vec<usize> {
        let l = self.leader.select(task, cands, rng);
        let mut team = vec![l];
        if self.k > 1 {
            let mut scored: Vec<(f64, usize)> =
                cands.iter().filter(|&&h| h != l).map(|&h| (sample_beta(self.ab[l][h].0, self.ab[l][h].1, rng), h)).collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            team.extend(scored.iter().take(self.k - 1).map(|x| x.1));
        }
        team
    }
    fn feedback(&mut self, task: &Task, team: &[usize], reward: f64) {
        self.leader.update(task, team[0], reward);
        for &h in &team[1..] {
            let e = &mut self.ab[team[0]][h];
            e.0 += reward;
            e.1 += 1.0 - reward;
        }
    }
}

/// Team baseline: keyword leader + a STATIC designer-supplied partner per leader. The partner
/// map is the synergy pairing of the FIXED world (coder-tester, architect-security,
/// reviewer-docs), i.e. oracle knowledge of the fixed world and stale in randomised worlds.
struct KeywordStatic {
    leader: KeywordRouter,
    k: usize,
}
const STATIC_PARTNER: [usize; 6] = [1, 0, 5, 4, 3, 2];
impl Policy for KeywordStatic {
    fn assign(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> Vec<usize> {
        let l = self.leader.select(task, cands, rng);
        let mut team = vec![l];
        if self.k > 1 && cands.contains(&STATIC_PARTNER[l]) {
            team.push(STATIC_PARTNER[l]);
        }
        team
    }
    fn feedback(&mut self, _: &Task, _: &[usize], _: f64) {}
}

#[derive(Clone, Debug, Serialize)]
enum Spec {
    PooledTs(f64),
    EpsGreedy,
    HelperBandit(HebbianConfig, usize),
    KeywordStatic(usize),
    Keyword,
    PgActor(PgActorConfig),
    RStdp(RStdpRouterConfig),
    Hebbian(HebbianConfig),
    /// NeuralMesh co-activation edges (ungated)
    Mesh(HebbianConfig, MeshConfig),
    /// R117: `Mesh` + environment-failure shielding (`feedback_env`)
    MeshShield(HebbianConfig, MeshConfig),
    /// R117: the recommended Guarded(hebbian-robust) router (keyword gate in front)
    Guarded,
    /// NeuralMesh + TrustEdges (ungated)
    MeshTrust(HebbianConfig, MeshConfig, TrustEdgesConfig),
    /// standalone TrustMesh policy
    TrustMesh(HebbianConfig, TrustEdgesConfig, usize),
    /// NeuralMesh (+ optional TrustEdges) with learned-value gate; team_size = upper bound
    Gated(HebbianConfig, MeshConfig, GateConfig, Option<TrustEdgesConfig>),
}

fn build(spec: &Spec, n: usize, kw: HashMap<String, usize>) -> Box<dyn Policy> {
    match spec {
        Spec::Keyword => Box::new(Single(KeywordRouter::new(kw, n))),
        Spec::PooledTs(g) => Box::new(Single(PooledThompsonRouter::new(n, *g))),
        Spec::EpsGreedy => Box::new(Single(EpsGreedyRouter::new(n, 0.99, 0.02))),
        Spec::HelperBandit(c, k) => Box::new(HelperBandit {
            leader: HebbianRouter::new(n, c.clone()),
            ab: vec![vec![(1.0, 1.0); n]; n],
            k: *k,
        }),
        Spec::KeywordStatic(k) => Box::new(KeywordStatic { leader: KeywordRouter::new(kw, n), k: *k }),
        Spec::PgActor(c) => Box::new(Single(PgActorRouter::new(n, c.clone()))),
        Spec::RStdp(c) => Box::new(Single(RStdpRouter::new(n, c.clone()))),
        Spec::Hebbian(c) => Box::new(Single(HebbianRouter::new(n, c.clone()))),
        Spec::Mesh(c, m) => Box::new(Mesh(NeuralMesh::new(n, HebbianRouter::new(n, c.clone()), m.clone()))),
        Spec::MeshShield(c, m) => Box::new(ShieldMesh(NeuralMesh::new(n, HebbianRouter::new(n, c.clone()), m.clone()))),
        Spec::Guarded => {
            let g: Guarded<HebbianRouter> = Guarded::new(HebbianRouter::new(n, recommended_inner()), kw, n, recommended_guard());
            Box::new(Single(g))
        }
        Spec::MeshTrust(c, m, t) => Box::new(Mesh(
            NeuralMesh::new(n, HebbianRouter::new(n, c.clone()), m.clone()).with_trust_edges(t.clone()),
        )),
        Spec::TrustMesh(c, t, k) => Box::new(TrustMesh {
            leader: HebbianRouter::new(n, c.clone()),
            edges: TrustEdges::new(n, t.clone()),
            team_size: *k,
        }),
        Spec::Gated(c, m, g, t) => {
            let mut mesh = NeuralMesh::new(n, HebbianRouter::new(n, c.clone()), m.clone());
            if let Some(t) = t {
                mesh = mesh.with_trust_edges(t.clone());
            }
            Box::new(Mesh(mesh.with_gate(g.clone())))
        }
    }
}

// ---------------------------------------------------------------------------
// One run = one policy on one seed.

#[derive(Clone, Debug, Default, Serialize)]
struct RunMetrics {
    success: f64,
    regret: f64,
    pre: f64,
    early_post: f64,
    late_post: f64,
    fail_window: f64,
    team_tasks: f64,
    solo_tasks: f64,
    /// agents ASSIGNED per task (what a system that cannot tell team tasks apart pays)
    calls_assigned: f64,
    /// agents actually engaged per task in the sim (solo tasks engage only the leader)
    calls_used: f64,
}

fn rate(s: f64, n: f64) -> f64 {
    if n > 0.0 { s / n } else { f64::NAN }
}

/// Feedback regime.
/// `informative = true` (the LEAKY mode): the policy is told which agents actually worked, i.e. on
/// solo tasks only the leader, so it implicitly learns the hidden team/solo flag.
/// `informative = false` (HEADLINE): the policy always receives the full team it routed.
/// `cost`: per-helper cost; the learning reward is `clamp(r - cost * (team_len - 1))`.
#[derive(Clone, Copy, Debug)]
struct Mode {
    informative: bool,
    cost: f64,
}
const HEADLINE: Mode = Mode { informative: false, cost: 0.0 };

/// `delay`: feedback for task t is delivered after task t+delay was routed.
/// R117 environment-failure layer (skill-independent, per-agent heterogeneous). `env` = mean failure rate (0 = off, bit-identical to before).
/// Agent `a` fails for environmental reasons with probability `env * M[a]` where `M` is a seeded permutation of a fixed multiplier set with mean 1.
/// Drawn from its own rng stream, so the task stream, hidden structure and outcomes of the other scenarios are untouched.
fn env_rates(seed: u64, env: f64, n: usize) -> Vec<f64> {
    const MULT: [f64; 6] = [0.2, 0.5, 0.8, 1.2, 1.6, 1.7];
    let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xE4F0_0D15);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..idx.len()).rev() {
        idx.swap(i, r.below(i + 1));
    }
    idx.iter().map(|&i| env * MULT[i % MULT.len()]).collect()
}

fn run(spec: &Spec, seed: u64, wcfg: &WorldConfig, delay: usize, env: f64, mode: Mode) -> RunMetrics {
    let world = World::new(seed, wcfg.clone());
    let n = world.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut pol = build(spec, n, world.keyword_map_for());
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let rates = env_rates(seed, env, n);
    let mut env_rng = Rng::new(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ 0x0E17_F00D);
    let mut acc = [[0.0f64; 2]; 6];
    let (mut succ, mut regret, mut assigned, mut used_n) = (0.0, 0.0, 0.0, 0.0);
    let mut pending: std::collections::VecDeque<(usize, Vec<usize>, f64, bool)> = std::collections::VecDeque::new();
    for t in 0..world.len() {
        let st = world.task(t);
        let team = pol.assign(&st.task, &cands, &mut rng);
        assert!(!team.is_empty(), "policy returned an empty team");
        let used: &[usize] = if st.team_task { &team } else { &team[..1] };
        let mut r = world.outcome(t, used);
        let env_failed = env > 0.0 && env_rng.f64() < rates[team[0]];
        if env_failed {
            r = 0.0;
        }
        regret += world.oracle_prob(t) - world.success_prob(t, used);
        succ += r;
        assigned += team.len() as f64;
        used_n += used.len() as f64;
        let fb_team: &[usize] = if mode.informative { used } else { &team };
        let fb_r = (r - mode.cost * (fb_team.len() - 1) as f64).clamp(0.0, 1.0);
        pending.push_back((t, fb_team.to_vec(), fb_r, env_failed));
        while let Some((t0, _, _, _)) = pending.front() {
            if t - *t0 < delay {
                break;
            }
            let (t0, tm, rr, ef) = pending.pop_front().unwrap();
            if ef {
                pol.feedback_env(&world.task(t0).task, &tm);
            } else {
                pol.feedback(&world.task(t0).task, &tm, rr);
            }
        }
        let w = match world.window(t) {
            Window::PreDrift => 0,
            Window::EarlyPost => 1,
            Window::LatePost => 2,
        };
        acc[w][0] += r;
        acc[w][1] += 1.0;
        if world.in_failure(t) {
            acc[3][0] += r;
            acc[3][1] += 1.0;
        }
        let k = if st.team_task { 4 } else { 5 };
        acc[k][0] += r;
        acc[k][1] += 1.0;
    }
    let nt = world.len() as f64;
    RunMetrics {
        success: succ / nt,
        regret: regret / nt,
        pre: rate(acc[0][0], acc[0][1]),
        early_post: rate(acc[1][0], acc[1][1]),
        late_post: rate(acc[2][0], acc[2][1]),
        fail_window: rate(acc[3][0], acc[3][1]),
        team_tasks: rate(acc[4][0], acc[4][1]),
        solo_tasks: rate(acc[5][0], acc[5][1]),
        calls_assigned: assigned / nt,
        calls_used: used_n / nt,
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(1)
}

/// Order-preserving parallel map.
fn par_map<T: Sync, R: Send>(items: &[T], f: &(dyn Fn(&T) -> R + Sync)) -> Vec<R> {
    let chunk = items.len().div_ceil(threads()).max(1);
    let mut out: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    std::thread::scope(|s| {
        for (slot, its) in out.chunks_mut(chunk).zip(items.chunks(chunk)) {
            s.spawn(move || {
                for (o, it) in slot.iter_mut().zip(its) {
                    *o = Some(f(it));
                }
            });
        }
    });
    out.into_iter().map(|o| o.unwrap()).collect()
}

fn field(ms: &[RunMetrics], f: fn(&RunMetrics) -> f64) -> Vec<f64> {
    ms.iter().map(f).collect()
}

// ---------------------------------------------------------------------------
// Scenarios

#[derive(Clone)]
struct Scen {
    name: &'static str,
    wcfg: WorldConfig,
    delay: usize,
    /// mean environment-failure rate (R117 layer); 0 for the six original scenarios
    env: f64,
}

fn scenarios(n_tasks: usize) -> Vec<Scen> {
    let base = WorldConfig { n_tasks, ..WorldConfig::default() };
    let with = |sc: Scenario| WorldConfig { scenario: sc, ..base.clone() };
    vec![
        Scen { name: "fixed", wcfg: with(Scenario::Fixed), delay: 0, env: 0.0 },
        Scen { name: "random", wcfg: with(Scenario::Random), delay: 0, env: 0.0 },
        Scen { name: "no-synergy", wcfg: with(Scenario::NoSynergy), delay: 0, env: 0.0 },
        Scen { name: "no-traps", wcfg: with(Scenario::NoTraps), delay: 0, env: 0.0 },
        Scen { name: "revealed", wcfg: WorldConfig { reveal_type: true, ..with(Scenario::Random) }, delay: 0, env: 0.0 },
        Scen { name: "delay25", wcfg: with(Scenario::Random), delay: 25, env: 0.0 },
    ]
}
/// indices (into `scenarios()`) of the randomised family used by the `tuned` criterion
const RAND_FAMILY: [usize; 3] = [1, 2, 3];

fn run_seeds(spec: &Spec, seeds: &[u64], sc: &Scen, mode: Mode) -> Vec<RunMetrics> {
    seeds.iter().map(|&s| run(spec, s, &sc.wcfg, sc.delay, sc.env, mode)).collect()
}

fn mean_success_seq(spec: &Spec, seeds: &[u64], sc: &Scen) -> f64 {
    seeds.iter().map(|&s| run(spec, s, &sc.wcfg, sc.delay, sc.env, HEADLINE).success).sum::<f64>() / seeds.len() as f64
}

// ---------------------------------------------------------------------------
// Grid tuner with automatic outward extension.

#[derive(Clone)]
struct Axis {
    name: &'static str,
    vals: Vec<f64>,
    lo: f64,
    hi: f64,
    ratio: f64,
    int: bool,
}
fn ax(name: &'static str, vals: &[f64], lo: f64, hi: f64, ratio: f64) -> Axis {
    Axis { name, vals: vals.to_vec(), lo, hi, ratio, int: false }
}
fn ax_int(name: &'static str, vals: &[f64], lo: f64, hi: f64) -> Axis {
    Axis { name, vals: vals.to_vec(), lo, hi, ratio: 1.0, int: true }
}

#[derive(Clone, Debug, Serialize)]
struct Tuned {
    what: String,
    axes: Vec<String>,
    final_grid: Vec<Vec<f64>>,
    n_evaluated: usize,
    extensions: usize,
    /// tuned criterion
    best_tuned: Vec<f64>,
    score_tuned: f64,
    /// robust criterion
    best_robust: Vec<f64>,
    mean_rank_robust: f64,
    worst_gain_robust: f64,
    /// per axis: "interior" | "edge(hard bound)" | "edge(not extended)"
    pos_tuned: Vec<String>,
    pos_robust: Vec<String>,
}

fn position(axes: &[Axis], best: &[f64]) -> Vec<String> {
    axes.iter()
        .zip(best)
        .map(|(a, &b)| {
            let (first, last) = (a.vals[0], *a.vals.last().unwrap());
            if (b == first && first <= a.lo + 1e-12) || (b == last && last >= a.hi - 1e-12) {
                "edge(hard bound)".into()
            } else if b == first || b == last {
                "edge(not extended)".into()
            } else {
                "interior".into()
            }
        })
        .collect()
}

fn combos(axes: &[Axis]) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = vec![vec![]];
    for a in axes {
        let mut next = vec![];
        for p in &out {
            for &v in &a.vals {
                let mut q = p.clone();
                q.push(v);
                next.push(q);
            }
        }
        out = next;
    }
    out
}

fn key(p: &[f64]) -> Vec<u64> {
    p.iter().map(|v| v.to_bits()).collect()
}

fn tune(
    what: &str,
    mut axes: Vec<Axis>,
    build_spec: &(dyn Fn(&[f64]) -> Spec + Sync),
    scens: &[Scen],
    seeds: &[u64],
    kw_scores: &[f64],
) -> Tuned {
    let mut cache: HashMap<Vec<u64>, Vec<f64>> = HashMap::new();
    let mut extensions = 0;
    let (mut best_a, mut best_b);
    let (mut sc_a, mut rank_b, mut worst_b);
    loop {
        let todo: Vec<Vec<f64>> = combos(&axes).into_iter().filter(|c| !cache.contains_key(&key(c))).collect();
        let res = par_map(&todo, &|c: &Vec<f64>| {
            let spec = build_spec(c);
            scens.iter().map(|s| mean_success_seq(&spec, seeds, s)).collect::<Vec<f64>>()
        });
        for (c, r) in todo.into_iter().zip(res) {
            cache.insert(key(&c), r);
        }
        let all = combos(&axes);
        let scores: Vec<&Vec<f64>> = all.iter().map(|c| &cache[&key(c)]).collect();
        // criterion A: mean over randomised family
        let crit_a: Vec<f64> =
            scores.iter().map(|s| RAND_FAMILY.iter().map(|&i| s[i]).sum::<f64>() / RAND_FAMILY.len() as f64).collect();
        // criterion B: mean rank over all scenarios (rank 1 = best)
        let mut rank_sum = vec![0.0; all.len()];
        for si in 0..scens.len() {
            let mut idx: Vec<usize> = (0..all.len()).collect();
            idx.sort_by(|&i, &j| scores[j][si].partial_cmp(&scores[i][si]).unwrap());
            for (r, &i) in idx.iter().enumerate() {
                rank_sum[i] += (r + 1) as f64;
            }
        }
        let ia = (0..all.len()).max_by(|&i, &j| crit_a[i].partial_cmp(&crit_a[j]).unwrap()).unwrap();
        let ib = (0..all.len()).min_by(|&i, &j| rank_sum[i].partial_cmp(&rank_sum[j]).unwrap()).unwrap();
        best_a = all[ia].clone();
        best_b = all[ib].clone();
        sc_a = crit_a[ia];
        rank_b = rank_sum[ib] / scens.len() as f64;
        worst_b = scores[ib].iter().zip(kw_scores).map(|(s, k)| s - k).fold(f64::INFINITY, f64::min);

        // extend outward where either optimum sits on a soft edge
        let mut changed = false;
        for (k, a) in axes.iter_mut().enumerate() {
            for b in [&best_a, &best_b] {
                let (first, last) = (a.vals[0], *a.vals.last().unwrap());
                if b[k] == first && first > a.lo + 1e-12 {
                    let v = if a.int { first - 1.0 } else { (first / a.ratio).max(a.lo) };
                    if v < first && !a.vals.contains(&v) {
                        a.vals.insert(0, v);
                        changed = true;
                    }
                }
                let last = if changed { *a.vals.last().unwrap() } else { last };
                if b[k] == last && last < a.hi - 1e-12 {
                    let v = if a.int { last + 1.0 } else { (last * a.ratio).min(a.hi) };
                    if v > last && !a.vals.contains(&v) {
                        a.vals.push(v);
                        changed = true;
                    }
                }
            }
        }
        if !changed || extensions >= 5 {
            break;
        }
        extensions += 1;
    }
    Tuned {
        what: what.to_string(),
        axes: axes.iter().map(|a| a.name.to_string()).collect(),
        final_grid: axes.iter().map(|a| a.vals.clone()).collect(),
        n_evaluated: cache.len(),
        extensions,
        pos_tuned: position(&axes, &best_a),
        pos_robust: position(&axes, &best_b),
        best_tuned: best_a,
        score_tuned: sc_a,
        best_robust: best_b,
        mean_rank_robust: rank_b,
        worst_gain_robust: worst_b,
    }
}

fn print_tuned(t: &Tuned) {
    println!("\n[tune seeds 1-8] {}: {} grid points ({} outward extensions)", t.what, t.n_evaluated, t.extensions);
    for (i, a) in t.axes.iter().enumerate() {
        println!(
            "  {:<12} grid {:?}  tuned={} [{}]  robust={} [{}]",
            a, t.final_grid[i], t.best_tuned[i], t.pos_tuned[i], t.best_robust[i], t.pos_robust[i]
        );
    }
    println!(
        "  tuned score (mean success, random family) {:.4};  robust: mean rank {:.1}, worst-case gain vs keyword {:+.4}",
        t.score_tuned, t.mean_rank_robust, t.worst_gain_robust
    );
}

// ---------------------------------------------------------------------------
// Reporting helpers

#[derive(Clone)]
struct Evaluated {
    name: String,
    runs: Vec<RunMetrics>,
}

fn mean_of(e: &Evaluated, f: fn(&RunMetrics) -> f64) -> f64 {
    summarize(&field(&e.runs, f)).mean
}

fn print_table(title: &str, evals: &[Evaluated]) {
    println!("\n=== {title} ===");
    println!(
        "{:<22} {:>16} {:>7} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>7}",
        "config", "success ±95%CI", "regret", "pre", "early", "late", "failW", "team", "solo", "calls/t", "used/t"
    );
    for e in evals {
        let s = summarize(&field(&e.runs, |m| m.success));
        println!(
            "{:<22} {:>8.4} ±{:>6.4} {:>7.4} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>7.2} {:>7.2}",
            e.name,
            s.mean,
            s.half_width,
            mean_of(e, |x| x.regret),
            mean_of(e, |x| x.pre),
            mean_of(e, |x| x.early_post),
            mean_of(e, |x| x.late_post),
            mean_of(e, |x| x.fail_window),
            mean_of(e, |x| x.team_tasks),
            mean_of(e, |x| x.solo_tasks),
            mean_of(e, |x| x.calls_assigned),
            mean_of(e, |x| x.calls_used),
        );
    }
}

#[derive(Clone, Debug, Serialize)]
struct Test {
    scenario: String,
    kind: String, // "beats-keyword" | "beats-leader" | "gated-vs-ungated"
    a: String,
    b: String,
    diff: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    losses: usize,
    p: f64,
    p_holm: f64,
    /// "BEATS" / "WORSE" / "n.s." after Holm at 0.05
    verdict: String,
}

fn make_test(scen: &str, kind: &str, a: &Evaluated, b: &Evaluated) -> Test {
    let (x, y) = (field(&a.runs, |m| m.success), field(&b.runs, |m| m.success));
    let d = paired_diff(&x, &y);
    Test {
        scenario: scen.into(),
        kind: kind.into(),
        a: a.name.clone(),
        b: b.name.clone(),
        diff: d.diff.mean,
        lo: d.diff.lo,
        hi: d.diff.hi,
        wins: d.wins,
        losses: d.losses,
        p: paired_t_pvalue(&x, &y),
        p_holm: 1.0,
        verdict: String::new(),
    }
}

fn apply_holm(tests: &mut [Test]) {
    let adj = holm_adjust(&tests.iter().map(|t| t.p).collect::<Vec<_>>());
    for (t, a) in tests.iter_mut().zip(adj) {
        t.p_holm = a;
        t.verdict = if a <= 0.05 { if t.diff > 0.0 { "BEATS" } else { "WORSE" } } else { "n.s." }.into();
    }
}

fn find<'a>(ev: &'a [Evaluated], name: &str) -> &'a Evaluated {
    ev.iter().find(|e| e.name == name).unwrap_or_else(|| panic!("missing config {name}"))
}

fn fmt_params(spec: &Spec) -> String {
    format!("{spec:?}")
}

// ---------------------------------------------------------------------------
// Evaluation, families and printers

const BASELINES: [&str; 4] = ["keyword", "pooled-ts-1.0", "pooled-ts-0.99", "eps-greedy"];
const CANDS: [&str; 6] = ["hebbian", "mesh", "trust-mesh", "mesh-trust", "gated-mesh", "gated-trust"];
const MESHY: [&str; 5] = ["mesh", "trust-mesh", "mesh-trust", "gated-mesh", "gated-trust"];
type Evals = Vec<(String, Vec<Evaluated>)>;

fn eval_all(configs: &[(String, Spec)], scens: &[Scen], seeds: &[u64], mode: Mode) -> Evals {
    scens
        .iter()
        .map(|sc| {
            let runs = par_map(configs, &|(_, s): &(String, Spec)| run_seeds(s, seeds, sc, mode));
            let ev = configs.iter().zip(runs).map(|((n, _), r)| Evaluated { name: n.clone(), runs: r }).collect();
            (sc.name.to_string(), ev)
        })
        .collect()
}

/// Best of the fair baselines by mean success IN THIS EVALUATION (favours the baseline, i.e.
/// conservative for the claims). Returns (renamed copy, original name).
fn best_baseline(ev: &[Evaluated]) -> (Evaluated, String) {
    let b = BASELINES
        .iter()
        .map(|n| find(ev, n))
        .max_by(|a, b| mean_of(a, |m| m.success).partial_cmp(&mean_of(b, |m| m.success)).unwrap())
        .unwrap();
    (Evaluated { name: "best-baseline".into(), runs: b.runs.clone() }, b.name.clone())
}

/// Holm family for one parameter set (`sfx` = "tuned" | "robust") over all scenarios:
///  * each candidate beats the best fair baseline            (beats-baseline)
///  * each mesh-like candidate beats its own leader hebbian   (beats-leader)
///  * gated vs ungated counterpart                            (gated-vs-ungated)
///  * each mesh-like candidate beats BOTH team baselines      (beats-team-baseline)
fn family(all: &Evals, sfx: &str) -> Vec<Test> {
    let mut tests = vec![];
    for (sn, ev) in all {
        let (bb, _) = best_baseline(ev);
        let leader = format!("hebbian-{sfx}");
        for c in CANDS {
            tests.push(make_test(sn, "beats-baseline", find(ev, &format!("{c}-{sfx}")), &bb));
        }
        for c in MESHY {
            tests.push(make_test(sn, "beats-leader", find(ev, &format!("{c}-{sfx}")), find(ev, &leader)));
        }
        tests.push(make_test(sn, "gated-vs-ungated", find(ev, &format!("gated-mesh-{sfx}")), find(ev, &format!("mesh-{sfx}"))));
        tests.push(make_test(sn, "gated-vs-ungated", find(ev, &format!("gated-trust-{sfx}")), find(ev, &format!("mesh-trust-{sfx}"))));
        for c in MESHY {
            for tb in [format!("hbandit-{sfx}"), "keyword+static-partner".to_string()] {
                tests.push(make_test(sn, "beats-team-baseline", find(ev, &format!("{c}-{sfx}")), find(ev, &tb)));
            }
        }
    }
    apply_holm(&mut tests);
    tests
}

fn vchar(tests: &[Test], sn: &str, a: &str, kind: &str, b: Option<&str>) -> &'static str {
    let ts: Vec<&Test> = tests.iter().filter(|t| t.scenario == sn && t.a == a && t.kind == kind && b.map_or(true, |b| t.b == b)).collect();
    if ts.is_empty() {
        return "-";
    }
    if ts.iter().any(|t| t.verdict == "WORSE") {
        "W"
    } else if ts.iter().all(|t| t.verdict == "BEATS") {
        "B"
    } else {
        "."
    }
}

fn print_summary(title: &str, all: &Evals, tests: &[Test], sfx: &str) {
    println!("\n=== {title} ===");
    println!("bb = best of {{keyword, pooled-ts-1.0, pooled-ts-0.99, eps-greedy}} in that scenario; ld = vs own leader hebbian-{sfx};");
    println!("gu = gated vs ungated counterpart; tb = vs BOTH team baselines (hbandit-{sfx}, keyword+static-partner).");
    println!("B = beats, W = worse, . = not significant after Holm (alpha 0.05); - = n/a");
    println!(
        "{:<11} {:<20} {:>6} {:>5} | {:<18} {:>6} {:>7} {:>5} {:>3} {:>3} {:>3} {:>3}",
        "scenario", "best config", "succ", "calls", "candidate", "succ", "Δbb", "calls", "bb", "ld", "gu", "tb"
    );
    for (sn, ev) in all {
        let best = ev.iter().max_by(|a, b| mean_of(a, |m| m.success).partial_cmp(&mean_of(b, |m| m.success)).unwrap()).unwrap();
        let (bb, bbn) = best_baseline(ev);
        let bbm = mean_of(&bb, |m| m.success);
        println!("{:<11} (best baseline: {} {:.4})", sn, bbn, bbm);
        let mut first = true;
        for c in CANDS {
            let name = format!("{c}-{sfx}");
            let e = find(ev, &name);
            let gu = match c {
                "gated-mesh" => vchar(tests, sn, &name, "gated-vs-ungated", Some(&format!("mesh-{sfx}"))),
                "gated-trust" => vchar(tests, sn, &name, "gated-vs-ungated", Some(&format!("mesh-trust-{sfx}"))),
                _ => "-",
            };
            println!(
                "{:<11} {:<20} {:>6} {:>5} | {:<18} {:>6.4} {:>+7.4} {:>5.2} {:>3} {:>3} {:>3} {:>3}",
                "",
                if first { best.name.as_str() } else { "" },
                if first { format!("{:.4}", mean_of(best, |m| m.success)) } else { String::new() },
                if first { format!("{:.2}", mean_of(best, |m| m.calls_assigned)) } else { String::new() },
                name,
                mean_of(e, |m| m.success),
                mean_of(e, |m| m.success) - bbm,
                mean_of(e, |m| m.calls_assigned),
                vchar(tests, sn, &name, "beats-baseline", None),
                if c == "hebbian" { "-" } else { vchar(tests, sn, &name, "beats-leader", None) },
                gu,
                if c == "hebbian" { "-" } else { vchar(tests, sn, &name, "beats-team-baseline", None) },
            );
            first = false;
        }
    }
}

/// Compact grid: rows = scenarios, one cell per candidate: success[verdict vs best baseline]/calls[/net].
fn print_compact(title: &str, all: &Evals, tests: &[Test], sfx: &str, cost: f64) {
    println!("\n=== {title} ===");
    println!("cell = success[B/W/. vs best baseline, Holm]/calls per task{}", if cost > 0.0 { format!("/net (= success - {cost} * (calls - 1))") } else { String::new() });
    print!("{:<11} {:<22}", "scenario", "best baseline");
    for c in CANDS {
        print!(" {:<22}", format!("{c}-{sfx}"));
    }
    println!();
    for (sn, ev) in all {
        let (bb, bbn) = best_baseline(ev);
        print!("{:<11} {:<22}", sn, format!("{} {:.4}", bbn, mean_of(&bb, |m| m.success)));
        for c in CANDS {
            let name = format!("{c}-{sfx}");
            let e = find(ev, &name);
            let (su, ca) = (mean_of(e, |m| m.success), mean_of(e, |m| m.calls_assigned));
            let cell = if cost > 0.0 {
                format!("{:.4}{}/{:.2}/{:.4}", su, vchar(tests, sn, &name, "beats-baseline", None), ca, su - cost * (ca - 1.0))
            } else {
                format!("{:.4}{}/{:.2}", su, vchar(tests, sn, &name, "beats-baseline", None), ca)
            };
            print!(" {:<22}", cell);
        }
        if cost > 0.0 {
            print!("   (bb net {:.4})", mean_of(&bb, |m| m.success) - cost * (mean_of(&bb, |m| m.calls_assigned) - 1.0));
        }
        println!();
    }
}

fn print_family_stats(name: &str, tests: &[Test]) {
    let surv = tests.iter().filter(|t| t.verdict != "n.s.").count();
    let raw = tests.iter().filter(|t| t.p <= 0.05).count();
    let beats = tests.iter().filter(|t| t.verdict == "BEATS").count();
    println!("  [{name}] {} tests; raw p<=0.05: {raw}; survive Holm: {surv} (BEATS {beats}, WORSE {})", tests.len(), surv - beats);
}

// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let n_tasks: usize = get("--tasks").and_then(|s| s.parse().ok()).unwrap_or(2000);
    let out_path = get("--out").unwrap_or_else(|| "results/results.json".to_string());
    let scens = scenarios(n_tasks);
    let tune_seeds: Vec<u64> = (1..=8).collect();
    // `--held A-B` picks the evaluation seeds (default 31-60 = held-out; 61-90 = the confirm split). Tuning is always seeds 1-8, so a 61-90 run
    // uses exactly the configs of the 31-60 run (frozen) and reproduces the committed 31-60 numbers when `--held` is absent.
    let held_seeds: Vec<u64> = match get("--held").and_then(|h| h.split_once('-').map(|(a, b)| (a.parse::<u64>(), b.parse::<u64>()))) {
        Some((Ok(a), Ok(b))) if a <= b => (a..=b).collect(),
        _ => (31..=60).collect(),
    };
    let held_label = format!("{}-{}", held_seeds[0], held_seeds[held_seeds.len() - 1]);
    let t_start = Instant::now();
    println!("neural-mesh experiment v2: {} tasks/run, {} scenarios, {} threads", n_tasks, scens.len(), threads());

    // best-baseline reference on the tuning seeds (for the worst-case-gain statistic)
    let base_specs = [Spec::Keyword, Spec::PooledTs(1.0), Spec::PooledTs(0.99), Spec::EpsGreedy];
    let kw_scores: Vec<f64> = scens
        .iter()
        .map(|s| base_specs.iter().map(|b| mean_success_seq(b, &tune_seeds, s)).fold(f64::NEG_INFINITY, f64::max))
        .collect();
    println!("[tune seeds 1-8] BEST-OF-BASELINES {{keyword, pooled-ts 1.0/0.99, eps-greedy}} per scenario: {}", scens.iter().zip(&kw_scores).map(|(s, k)| format!("{} {:.4}", s.name, k)).collect::<Vec<_>>().join(", "));
    println!("[tune seeds 1-8] tuning uses the HEADLINE feedback mode (uninformative: mesh receives its full routed team), cost 0");

    // ---- 1a. hebbian -------------------------------------------------------
    let base_h = HebbianConfig::default();
    let th = tune(
        "hebbian",
        vec![
            ax("eta", &[0.02, 0.05, 0.1, 0.25, 0.5, 0.8], 0.005, 2.0, 2.0),
            ax("temperature", &[0.01, 0.02, 0.05, 0.1, 0.2], 0.002, 1.0, 2.0),
            ax("decay", &[0.0, 0.001, 0.005, 0.01, 0.03], 0.0, 0.3, 3.0),
            ax("epsilon", &[0.0, 0.02, 0.05, 0.1], 0.0, 0.4, 2.0),
        ],
        &|p| Spec::Hebbian(HebbianConfig { eta: p[0], temperature: p[1], decay: p[2], epsilon: p[3], ..base_h.clone() }),
        &scens,
        &tune_seeds,
        &kw_scores,
    );
    print_tuned(&th);
    let mk_h = |p: &[f64]| HebbianConfig { eta: p[0], temperature: p[1], decay: p[2], epsilon: p[3], ..base_h.clone() };
    let h_t = mk_h(&th.best_tuned);
    let h_r = mk_h(&th.best_robust);

    // ---- 1b. mesh (co-activation edges), per leader ------------------------
    let base_m = MeshConfig::default();
    let mesh_axes = || {
        vec![
            ax("edge_eta", &[0.05, 0.1, 0.2, 0.4], 0.01, 1.5, 2.0),
            ax("edge_decay", &[0.0, 0.003, 0.01, 0.03], 0.0, 0.3, 3.0),
            ax_int("team_size", &[1.0, 2.0, 3.0, 4.0], 1.0, 5.0),
        ]
    };
    let mk_m = |p: &[f64]| MeshConfig { edge_eta: p[0], edge_decay: p[1], team_size: p[2] as usize, ..base_m.clone() };
    let tm_t = tune("mesh [leader = hebbian-tuned]", mesh_axes(), &|p| Spec::Mesh(h_t.clone(), mk_m(p)), &scens, &tune_seeds, &kw_scores);
    print_tuned(&tm_t);
    let tm_r = tune("mesh [leader = hebbian-robust]", mesh_axes(), &|p| Spec::Mesh(h_r.clone(), mk_m(p)), &scens, &tune_seeds, &kw_scores);
    print_tuned(&tm_r);
    let m_t = mk_m(&tm_t.best_tuned);
    let m_r = mk_m(&tm_r.best_robust);

    // ---- 1c. trust edges (standalone TrustMesh and NeuralMesh+trust) -------
    let base_te = TrustEdgesConfig::default();
    let mk_te = |p: &[f64]| TrustEdgesConfig { mu: p[0], psi: p[1], ..base_te.clone() };
    let te_axes = || {
        vec![
            ax("mu", &[0.05, 0.1, 0.25, 0.5], 0.01, 2.0, 2.0),
            ax("psi", &[0.0, 0.02, 0.05, 0.1], 0.0, 0.5, 2.5),
            ax_int("team_size", &[1.0, 2.0, 3.0, 4.0], 1.0, 5.0),
        ]
    };
    let tt_t = tune("trust-mesh [leader = hebbian-tuned]", te_axes(), &|p| Spec::TrustMesh(h_t.clone(), mk_te(p), p[2] as usize), &scens, &tune_seeds, &kw_scores);
    print_tuned(&tt_t);
    let tt_r = tune("trust-mesh [leader = hebbian-robust]", te_axes(), &|p| Spec::TrustMesh(h_r.clone(), mk_te(p), p[2] as usize), &scens, &tune_seeds, &kw_scores);
    print_tuned(&tt_r);
    let (te_t, k_te_t) = (mk_te(&tt_t.best_tuned), tt_t.best_tuned[2] as usize);
    let (te_r, k_te_r) = (mk_te(&tt_r.best_robust), tt_r.best_robust[2] as usize);

    let mt_t = tune(
        "mesh+trust-edges [leader = hebbian-tuned]",
        te_axes(),
        &|p| Spec::MeshTrust(h_t.clone(), MeshConfig { team_size: p[2] as usize, ..base_m.clone() }, mk_te(p)),
        &scens,
        &tune_seeds,
        &kw_scores,
    );
    print_tuned(&mt_t);
    let mt_r = tune(
        "mesh+trust-edges [leader = hebbian-robust]",
        te_axes(),
        &|p| Spec::MeshTrust(h_r.clone(), MeshConfig { team_size: p[2] as usize, ..base_m.clone() }, mk_te(p)),
        &scens,
        &tune_seeds,
        &kw_scores,
    );
    print_tuned(&mt_r);
    let (mte_t, mk_t) = (mk_te(&mt_t.best_tuned), mt_t.best_tuned[2] as usize);
    let (mte_r, mk_r) = (mk_te(&mt_r.best_robust), mt_r.best_robust[2] as usize);

    // ---- 1d. gated meshes (edge params inherited from the ungated counterpart) -
    let gate_axes = || {
        vec![
            ax("min_gain", &[0.0, 0.02, 0.05], 0.0, 0.3, 2.5),
            ax("explore", &[0.02, 0.05, 0.1], 0.0, 0.5, 2.0),
            ax_int("team_size", &[2.0, 3.0, 4.0], 1.0, 6.0),
        ]
    };
    let mk_g = |p: &[f64]| GateConfig { min_gain: p[0], explore: p[1], ..GateConfig::default() };
    let gm_t = tune(
        "gated mesh [co-activation; leader+edges from tuned]",
        gate_axes(),
        &|p| Spec::Gated(h_t.clone(), MeshConfig { team_size: p[2] as usize, ..m_t.clone() }, mk_g(p), None),
        &scens,
        &tune_seeds,
        &kw_scores,
    );
    print_tuned(&gm_t);
    let gm_r = tune(
        "gated mesh [co-activation; leader+edges from robust]",
        gate_axes(),
        &|p| Spec::Gated(h_r.clone(), MeshConfig { team_size: p[2] as usize, ..m_r.clone() }, mk_g(p), None),
        &scens,
        &tune_seeds,
        &kw_scores,
    );
    print_tuned(&gm_r);
    let gt_t = tune(
        "gated mesh+trust [leader+trust from tuned]",
        gate_axes(),
        &|p| Spec::Gated(h_t.clone(), MeshConfig { team_size: p[2] as usize, ..base_m.clone() }, mk_g(p), Some(mte_t.clone())),
        &scens,
        &tune_seeds,
        &kw_scores,
    );
    print_tuned(&gt_t);
    let gt_r = tune(
        "gated mesh+trust [leader+trust from robust]",
        gate_axes(),
        &|p| Spec::Gated(h_r.clone(), MeshConfig { team_size: p[2] as usize, ..base_m.clone() }, mk_g(p), Some(mte_r.clone())),
        &scens,
        &tune_seeds,
        &kw_scores,
    );
    print_tuned(&gt_r);

    // ---- 1e. other single-agent learners (tuned the same way) ---------------
    let base_pg = PgActorConfig::default();
    let tpg = tune(
        "pg-actor",
        vec![ax("eta", &[0.1, 0.25, 0.5, 1.0, 2.0], 0.01, 8.0, 2.0), ax("theta_clip", &[2.0, 4.0, 8.0], 1.0, 32.0, 2.0)],
        &|p| Spec::PgActor(PgActorConfig { eta: p[0], theta_clip: p[1], ..base_pg.clone() }),
        &scens,
        &tune_seeds,
        &kw_scores,
    );
    print_tuned(&tpg);
    let base_rs = RStdpRouterConfig::default();
    let trs = tune(
        "r-stdp",
        vec![
            ax("eta", &[0.25, 0.5, 1.0], 0.05, 4.0, 2.0),
            ax("temperature", &[0.05, 0.1, 0.2], 0.01, 1.0, 2.0),
            ax("tau_e", &[0.5, 1.0, 2.0, 4.0], 0.1, 16.0, 2.0),
        ],
        &|p| Spec::RStdp(RStdpRouterConfig { eta: p[0] as f32, temperature: p[1], tau_e: p[2] as f32, ..base_rs.clone() }),
        &scens,
        &tune_seeds,
        &kw_scores,
    );
    print_tuned(&trs);

    // ---- 2. configs --------------------------------------------------------
    let gated_mesh_t = Spec::Gated(h_t.clone(), MeshConfig { team_size: gm_t.best_tuned[2] as usize, ..m_t.clone() }, mk_g(&gm_t.best_tuned), None);
    let gated_mesh_r = Spec::Gated(h_r.clone(), MeshConfig { team_size: gm_r.best_robust[2] as usize, ..m_r.clone() }, mk_g(&gm_r.best_robust), None);
    let gated_trust_t = Spec::Gated(
        h_t.clone(),
        MeshConfig { team_size: gt_t.best_tuned[2] as usize, ..base_m.clone() },
        mk_g(&gt_t.best_tuned),
        Some(mte_t.clone()),
    );
    let gated_trust_r = Spec::Gated(
        h_r.clone(),
        MeshConfig { team_size: gt_r.best_robust[2] as usize, ..base_m.clone() },
        mk_g(&gt_r.best_robust),
        Some(mte_r.clone()),
    );
    let configs: Vec<(String, Spec)> = vec![
        ("keyword".into(), Spec::Keyword),
        ("pooled-ts-1.0".into(), Spec::PooledTs(1.0)),
        ("pooled-ts-0.99".into(), Spec::PooledTs(0.99)),
        ("eps-greedy".into(), Spec::EpsGreedy),
        ("hebbian-default".into(), Spec::Hebbian(base_h.clone())),
        ("hebbian-tuned".into(), Spec::Hebbian(h_t.clone())),
        ("hebbian-robust".into(), Spec::Hebbian(h_r.clone())),
        ("mesh-default-k2".into(), Spec::Mesh(base_h.clone(), MeshConfig { team_size: 2, ..base_m.clone() })),
        ("mesh-tuned".into(), Spec::Mesh(h_t.clone(), m_t.clone())),
        ("mesh-robust".into(), Spec::Mesh(h_r.clone(), m_r.clone())),
        ("trust-mesh-tuned".into(), Spec::TrustMesh(h_t.clone(), te_t.clone(), k_te_t)),
        ("trust-mesh-robust".into(), Spec::TrustMesh(h_r.clone(), te_r.clone(), k_te_r)),
        ("mesh-trust-tuned".into(), Spec::MeshTrust(h_t.clone(), MeshConfig { team_size: mk_t, ..base_m.clone() }, mte_t.clone())),
        ("mesh-trust-robust".into(), Spec::MeshTrust(h_r.clone(), MeshConfig { team_size: mk_r, ..base_m.clone() }, mte_r.clone())),
        ("gated-mesh-tuned".into(), gated_mesh_t),
        ("gated-mesh-robust".into(), gated_mesh_r),
        ("gated-trust-tuned".into(), gated_trust_t),
        ("gated-trust-robust".into(), gated_trust_r),
        ("hbandit-tuned".into(), Spec::HelperBandit(h_t.clone(), 2)),
        ("hbandit-robust".into(), Spec::HelperBandit(h_r.clone(), 2)),
        ("keyword+static-partner".into(), Spec::KeywordStatic(2)),
        ("pg-actor-tuned".into(), Spec::PgActor(PgActorConfig { eta: tpg.best_tuned[0], theta_clip: tpg.best_tuned[1], ..base_pg.clone() })),
        (
            "r-stdp-tuned".into(),
            Spec::RStdp(RStdpRouterConfig {
                eta: trs.best_tuned[0] as f32,
                temperature: trs.best_tuned[1],
                tau_e: trs.best_tuned[2] as f32,
                ..base_rs.clone()
            }),
        ),
    ];
    println!("\n=== final configurations ===");
    for (n, s) in &configs {
        println!("  {:<22} {}", n, fmt_params(s));
    }

    // ---- 3. HEADLINE: held-out seeds 31-60, uninformative feedback, cost 0, 2000 tasks -------
    let headline = eval_all(&configs, &scens, &held_seeds, HEADLINE);
    for (sn, ev) in &headline {
        print_table(&format!("HEADLINE held-out seeds 31-60 | scenario: {sn} | uninformative feedback"), ev);
    }
    let mut fam_json = vec![];
    println!("\n=== Holm-Bonferroni families (headline mode) ===");
    let fam_t = family(&headline, "tuned");
    let fam_r = family(&headline, "robust");
    print_family_stats("tuned", &fam_t);
    print_family_stats("robust", &fam_r);
    print_summary("SUMMARY (headline, TUNED set)", &headline, &fam_t, "tuned");
    print_summary("SUMMARY (headline, ROBUST set)", &headline, &fam_r, "robust");
    for (nm, t) in [("headline-tuned", &fam_t), ("headline-robust", &fam_r)] {
        fam_json.push(json!({"family": nm, "n_tests": t.len(), "tests": t}));
    }
    println!("\n--- every test in the headline TUNED family (candidate - comparator, held-out success) ---");
    for t in &fam_t {
        println!(
            "  {:<11} {:<19} {:<19} vs {:<22} {:+.4} [{:+.4},{:+.4}] {:>2}/{:<2} p_holm {:.4} {}",
            t.scenario, t.kind, t.a, t.b, t.diff, t.lo, t.hi, t.wins, t.losses, t.p_holm, t.verdict
        );
    }

    // gating: solo-task penalty and cost
    println!("\n=== GATING (headline): solo-task success and cost, gated vs ungated (held-out means) ===");
    println!("{:<11} {:<20} {:>8} {:>8} {:>8} {:>8} {:>8}", "scenario", "config", "success", "solo", "team", "calls/t", "used/t");
    for (sn, ev) in &headline {
        for n in [
            "hebbian-tuned", "mesh-tuned", "gated-mesh-tuned", "mesh-trust-tuned", "gated-trust-tuned", "mesh-robust", "gated-mesh-robust",
            "mesh-trust-robust", "gated-trust-robust",
        ] {
            let e = find(ev, n);
            println!(
                "{:<11} {:<20} {:>8.4} {:>8.4} {:>8.4} {:>8.2} {:>8.2}",
                sn,
                n,
                mean_of(e, |m| m.success),
                mean_of(e, |m| m.solo_tasks),
                mean_of(e, |m| m.team_tasks),
                mean_of(e, |m| m.calls_assigned),
                mean_of(e, |m| m.calls_used)
            );
        }
    }

    // ---- 3b. informative (LEAKY) feedback ----------------------------------------------------
    let leaky = eval_all(&configs, &scens, &held_seeds, Mode { informative: true, cost: 0.0 });
    let leaky_t = family(&leaky, "tuned");
    let leaky_r = family(&leaky, "robust");
    println!("\n=== LEAKY mode (mesh told which agents actually worked => learns the hidden team/solo flag) ===");
    print_family_stats("leaky tuned", &leaky_t);
    print_family_stats("leaky robust", &leaky_r);
    print_summary("SUMMARY (LEAKY feedback, TUNED set; configs tuned under uninformative feedback)", &leaky, &leaky_t, "tuned");
    println!("\n  mesh minus own leader (held-out success), headline vs leaky:");
    println!("  {:<11} {:<14} {:>10} {:>10}", "scenario", "config", "headline", "leaky");
    for ((sn, h), (_, l)) in headline.iter().zip(&leaky) {
        for c in MESHY {
            let n = format!("{c}-tuned");
            let d = |ev: &[Evaluated]| mean_of(find(ev, &n), |m| m.success) - mean_of(find(ev, "hebbian-tuned"), |m| m.success);
            println!("  {:<11} {:<14} {:>+10.4} {:>+10.4}", sn, c, d(h), d(l));
        }
    }
    println!("
  teaming configs (these actually use helpers) minus their leader, headline vs leaky:");
    println!("  {:<11} {:<34} {:>10} {:>10}", "scenario", "config - leader", "headline", "leaky");
    for ((sn, h), (_, l)) in headline.iter().zip(&leaky) {
        for (a, b) in [
            ("mesh-default-k2", "hebbian-default"),
            ("hbandit-tuned", "hebbian-tuned"),
            ("gated-mesh-robust", "hebbian-robust"),
            ("gated-trust-tuned", "hebbian-tuned"),
        ] {
            let d = |ev: &[Evaluated]| mean_of(find(ev, a), |m| m.success) - mean_of(find(ev, b), |m| m.success);
            println!("  {:<11} {:<34} {:>+10.4} {:>+10.4}", sn, format!("{a} - {b}"), d(h), d(l));
        }
    }
    fam_json.push(json!({"family": "leaky-tuned", "n_tests": leaky_t.len(), "tests": leaky_t}));
    fam_json.push(json!({"family": "leaky-robust", "n_tests": leaky_r.len(), "tests": leaky_r}));

    // ---- 3c. helper cost ---------------------------------------------------------------------
    let mut cost_json = vec![];
    for c in [0.05, 0.1] {
        let ev = eval_all(&configs, &scens, &held_seeds, Mode { informative: false, cost: c });
        let t = family(&ev, "tuned");
        print_compact(&format!("HELPER COST c = {c} (learning reward -= c per helper; configs NOT retuned), tuned set"), &ev, &t, "tuned", c);
        print_family_stats(&format!("cost {c} tuned"), &t);
        let t_r = family(&ev, "robust");
        print_compact(&format!("HELPER COST c = {c}, robust set"), &ev, &t_r, "robust", c);
        print_family_stats(&format!("cost {c} robust"), &t_r);
        cost_json.push(json!({"cost": c, "tuned_tests": t, "robust_tests": t_r}));
    }

    // ---- 3d. horizons ------------------------------------------------------------------------
    let mut hz_json = vec![];
    for h in [500usize, 1000, 6000] {
        let sc_h = scenarios(h);
        let ev = eval_all(&configs, &sc_h, &held_seeds, HEADLINE);
        let t = family(&ev, "tuned");
        print_compact(&format!("HORIZON {h} tasks (configs tuned at 2000; headline mode), tuned set"), &ev, &t, "tuned", 0.0);
        print_family_stats(&format!("horizon {h} tuned"), &t);
        let t_r = family(&ev, "robust");
        print_compact(&format!("HORIZON {h} tasks, robust set"), &ev, &t_r, "robust", 0.0);
        print_family_stats(&format!("horizon {h} robust"), &t_r);
        hz_json.push(json!({"horizon": h, "tuned_tests": t, "robust_tests": t_r}));
    }
    print_compact("HORIZON 2000 tasks (headline), tuned set", &headline, &fam_t, "tuned", 0.0);

    // ---- 4. cross-scenario robustness (headline) ---------------------------------------------
    println!("\n=== CROSS-SCENARIO (headline, held-out): mean success, worst-case gain vs best baseline, mean/worst rank, calls ===");
    println!("{:<24} {:>8} {:>10} {:>9} {:>9} {:>7}", "config", "mean", "worst Δbb", "mean rank", "worst rk", "calls");
    let nconf = configs.len();
    let mut ranks = vec![vec![0usize; headline.len()]; nconf];
    for (si, (_, ev)) in headline.iter().enumerate() {
        let mut idx: Vec<usize> = (0..nconf).collect();
        idx.sort_by(|&i, &j| mean_of(&ev[j], |m| m.success).partial_cmp(&mean_of(&ev[i], |m| m.success)).unwrap());
        for (r, &i) in idx.iter().enumerate() {
            ranks[i][si] = r + 1;
        }
    }
    let mut cross = vec![];
    for (i, (name, _)) in configs.iter().enumerate() {
        let succ: Vec<f64> = headline.iter().map(|(_, ev)| mean_of(&ev[i], |m| m.success)).collect();
        let gain: Vec<f64> =
            headline.iter().map(|(_, ev)| mean_of(&ev[i], |m| m.success) - mean_of(&best_baseline(ev).0, |m| m.success)).collect();
        let calls: f64 = headline.iter().map(|(_, ev)| mean_of(&ev[i], |m| m.calls_assigned)).sum::<f64>() / headline.len() as f64;
        let mr = ranks[i].iter().sum::<usize>() as f64 / headline.len() as f64;
        let wr = *ranks[i].iter().max().unwrap();
        let ms = succ.iter().sum::<f64>() / succ.len() as f64;
        let wg = gain.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("{:<24} {:>8.4} {:>+10.4} {:>9.2} {:>9} {:>7.2}", name, ms, wg, mr, wr, calls);
        cross.push(json!({"config": name, "mean_success": ms, "worst_gain_vs_best_baseline": wg, "mean_rank": mr, "worst_rank": wr, "mean_calls": calls}));
    }


    // ---- 5. R117 EXTRA ROWS (additive: nothing above changes) ---------------------------------------------------------------------------------
    // (a) `guarded-robust` = the recommended Guarded(hebbian-robust) router (Guarded row, item 4);
    // (b) scenario `env18`: random world + ~18% skill-independent ENVIRONMENT failures (per-agent heterogeneous, mean .18) and the failure-class
    //     shield `mesh-robust-shield` (env failures only update the dispatch-risk score) vs `mesh-robust` (env failure = ordinary reward 0).
    let mut x_scens = scens.clone();
    x_scens.push(Scen { name: "env18", wcfg: scens[1].wcfg.clone(), delay: 0, env: 0.18 });
    let x_configs: Vec<(String, Spec)> = vec![
        ("keyword".into(), Spec::Keyword),
        ("pooled-ts-1.0".into(), Spec::PooledTs(1.0)),
        ("pooled-ts-0.99".into(), Spec::PooledTs(0.99)),
        ("eps-greedy".into(), Spec::EpsGreedy),
        ("hebbian-robust".into(), Spec::Hebbian(h_r.clone())),
        ("mesh-robust".into(), Spec::Mesh(h_r.clone(), m_r.clone())),
        ("mesh-robust-shield".into(), Spec::MeshShield(h_r.clone(), m_r.clone())),
        ("guarded-robust".into(), Spec::Guarded),
    ];
    println!("\n=== R117 extra rows ===");
    for (n, sp) in &x_configs[6..] {
        println!("  {:<20} {}", n, fmt_params(sp));
    }
    let extra = eval_all(&x_configs, &x_scens, &held_seeds, HEADLINE);
    for (sn, ev) in &extra {
        print_table(&format!("R117 EXTRA held-out seeds {held_label} | scenario: {sn} | uninformative feedback"), ev);
    }
    let mut xtests: Vec<Test> = vec![];
    for (sn, ev) in &extra {
        let (bb, _) = best_baseline(ev);
        for (a, b, kind) in [
            ("guarded-robust", "keyword", "guarded-vs-keyword"),
            ("guarded-robust", "mesh-robust", "guarded-vs-mesh"),
        ] {
            xtests.push(make_test(sn, kind, find(ev, a), find(ev, b)));
        }
        if *sn == "env18" {
            xtests.push(make_test(sn, "shield-vs-unshielded", find(ev, "mesh-robust-shield"), find(ev, "mesh-robust")));
            xtests.push(make_test(sn, "shield-vs-baseline", find(ev, "mesh-robust-shield"), &bb));
        }
    }
    apply_holm(&mut xtests);
    // the shield is a no-op where no environment failure exists: it must equal mesh-robust EXACTLY in the six original scenarios
    let mut shield_noop = true;
    for (sn, ev) in extra.iter().filter(|(n, _)| n != "env18") {
        let (a, b) = (find(ev, "mesh-robust-shield"), find(ev, "mesh-robust"));
        let same = a.runs.iter().zip(&b.runs).all(|(x, y)| x.success == y.success);
        shield_noop &= same;
        if !same {
            println!("  WARNING shield differs from mesh-robust in scenario {sn}");
        }
    }
    println!("  shield == mesh-robust bit-identically in the six original scenarios (no regression possible): {shield_noop}");
    println!("\n--- R117 tests (candidate - comparator, held-out {held_label}, success; Holm over {} tests; EQ = 95% CI inside +-0.01) ---", xtests.len());
    for t in &xtests {
        let eq = if t.lo > -0.01 && t.hi < 0.01 { "EQ" } else { "" };
        println!(
            "  {:<11} {:<20} {:<18} vs {:<14} {:+.4} [{:+.4},{:+.4}] {:>2}/{:<2} p_holm {:.4} {} {}",
            t.scenario, t.kind, t.a, t.b, t.diff, t.lo, t.hi, t.wins, t.losses, t.p_holm, t.verdict, eq
        );
    }
    println!("  (mean success per scenario, held {held_label}:)");
    for (sn, ev) in &extra {
        println!(
            "  {:<11} {}",
            sn,
            x_configs.iter().map(|(n, _)| format!("{} {:.4}", n, mean_of(find(ev, n), |m| m.success))).collect::<Vec<_>>().join(" | ")
        );
    }
    let extra_json = json!({"held": held_label, "tests": xtests, "shield_noop_in_original_scenarios": shield_noop,
        "per_seed_success": extra.iter().map(|(sn, ev)| json!({"scenario": sn, "configs": ev.iter().map(|e| json!({"name": e.name, "success": field(&e.runs, |x| x.success)})).collect::<Vec<_>>()})).collect::<Vec<_>>()});

    // ---- JSON ---------------------------------------------------------------
    let tuned_json = |t: &Tuned| serde_json::to_value(t).unwrap();
    let scen_json: Vec<serde_json::Value> = headline
        .iter()
        .map(|(sn, ev)| {
            let cfgs: serde_json::Map<String, serde_json::Value> = ev
                .iter()
                .map(|e| {
                    let mut m = serde_json::Map::new();
                    for (mn, f) in [
                        ("success", (|x: &RunMetrics| x.success) as fn(&RunMetrics) -> f64),
                        ("regret", |x| x.regret),
                        ("fail_window", |x| x.fail_window),
                        ("team_tasks", |x| x.team_tasks),
                        ("solo_tasks", |x| x.solo_tasks),
                        ("calls_assigned", |x| x.calls_assigned),
                        ("calls_used", |x| x.calls_used),
                    ] {
                        m.insert(mn.into(), serde_json::to_value(summarize(&field(&e.runs, f))).unwrap());
                    }
                    (e.name.clone(), json!({"summary": m, "per_seed_success": field(&e.runs, |x| x.success)}))
                })
                .collect();
            json!({"scenario": sn, "seeds": held_seeds, "configs": cfgs})
        })
        .collect();
    let out = json!({
        "n_tasks": n_tasks,
        "scenarios": scens.iter().map(|s| json!({"name": s.name, "delay": s.delay, "world": format!("{:?}", s.wcfg)})).collect::<Vec<_>>(),
        "tuning": {
            "seeds": tune_seeds,
            "mode": "uninformative feedback, cost 0, 2000 tasks",
            "tuned_criterion": "mean success over scenarios random,no-synergy,no-traps",
            "robust_criterion": "mean rank over all six scenarios",
            "hebbian": tuned_json(&th), "mesh_tuned_leader": tuned_json(&tm_t), "mesh_robust_leader": tuned_json(&tm_r),
            "trust_mesh_tuned_leader": tuned_json(&tt_t), "trust_mesh_robust_leader": tuned_json(&tt_r),
            "mesh_trust_tuned_leader": tuned_json(&mt_t), "mesh_trust_robust_leader": tuned_json(&mt_r),
            "gated_mesh_tuned": tuned_json(&gm_t), "gated_mesh_robust": tuned_json(&gm_r),
            "gated_trust_tuned": tuned_json(&gt_t), "gated_trust_robust": tuned_json(&gt_r),
            "pg_actor": tuned_json(&tpg), "r_stdp": tuned_json(&trs),
        },
        "configs": configs.iter().map(|(n, s)| json!({"name": n, "spec": s})).collect::<Vec<_>>(),
        "heldout_31_60_headline": scen_json,
        "holm_families": fam_json,
        "cost_runs": cost_json,
        "horizon_runs": hz_json,
        "cross_scenario": cross,
        "r117_extra": extra_json,
        "elapsed_s": t_start.elapsed().as_secs_f64(),
    });
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).expect("write results");
    println!("\nwrote {out_path} ({:.1}s)", t_start.elapsed().as_secs_f64());
}
