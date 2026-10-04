//! Benchmark runner: evaluates routers and the NeuralMesh on the simulated world
//! (src/sim.rs) with paired per-seed statistics (src/stats.rs).
//!
//! Protocol (see DESIGN.md):
//! 1. Tune small grids for Hebbian (eta, temperature, decay) and the mesh
//!    (edge_eta, edge_decay, team_size) on seeds 1-8 ONLY.
//! 2. Evaluate all configs (defaults + tuned) on seeds 1-30 (overlaps tuning)
//!    and on held-out seeds 31-60 (the honest number).
//! 3. Paired per-seed differences vs keyword and vs thompson (mean ± 95% t-CI, wins).
//! 4. Robustness: held-out seeds with the task type revealed to the routers.
//! 5. Routing latency (ns per route call), single-threaded.
//!
//! Usage: cargo run --release --bin experiment [-- --tasks 2000 --out results/results.json]

use neural_mesh::baselines::{KeywordRouter, ThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::mesh::{MeshConfig, NeuralMesh};
use neural_mesh::router::{Rng, Router};
use neural_mesh::sim::{Window, World, WorldConfig};
use neural_mesh::variants::{PgActorConfig, PgActorRouter, RStdpRouter, RStdpRouterConfig, TrustEdges, TrustEdgesConfig};
use neural_mesh::stats::{paired_diff, summarize, PairedDiff, Summary};
use serde::Serialize;
use serde_json::json;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Policies: a uniform "assign a team / receive feedback" interface.

trait Policy {
    /// Team for the task (leader first, len >= 1).
    fn assign(&mut self, task: &neural_mesh::router::Task, cands: &[usize], rng: &mut Rng) -> Vec<usize>;
    /// Feedback for the team that actually worked the task.
    fn feedback(&mut self, task: &neural_mesh::router::Task, team: &[usize], reward: f64);
}

struct Single<R: Router>(R);
impl<R: Router> Policy for Single<R> {
    fn assign(&mut self, task: &neural_mesh::router::Task, cands: &[usize], rng: &mut Rng) -> Vec<usize> {
        vec![self.0.select(task, cands, rng)]
    }
    fn feedback(&mut self, task: &neural_mesh::router::Task, team: &[usize], reward: f64) {
        self.0.update(task, team[0], reward);
    }
}

struct Mesh<R: Router>(NeuralMesh<R>);
impl<R: Router> Policy for Mesh<R> {
    fn assign(&mut self, task: &neural_mesh::router::Task, cands: &[usize], rng: &mut Rng) -> Vec<usize> {
        self.0.route(task, cands, rng)
    }
    fn feedback(&mut self, task: &neural_mesh::router::Task, team: &[usize], reward: f64) {
        self.0.feedback(task, team, reward);
    }
}

/// Mesh variant whose helper selection uses hatcher-style `TrustEdges`
/// (src/variants.rs) instead of NeuralMesh co-activation edges. No health/benching.
/// Helpers: top partners of the leader by edge weight; each slot is replaced by a
/// random remaining candidate with probability `TRUST_HELPER_EPS` (same exploration
/// rate as NeuralMesh's default helper epsilon). Edges are only updated on team
/// feedback (len >= 2), since solo tasks carry no pair traffic.
const TRUST_HELPER_EPS: f64 = 0.05;
struct TrustMesh<R: Router> {
    leader: R,
    edges: TrustEdges,
    team_size: usize,
}
impl<R: Router> Policy for TrustMesh<R> {
    fn assign(&mut self, task: &neural_mesh::router::Task, cands: &[usize], rng: &mut Rng) -> Vec<usize> {
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
    fn feedback(&mut self, task: &neural_mesh::router::Task, team: &[usize], reward: f64) {
        self.leader.update(task, team[0], reward);
        if team.len() >= 2 {
            self.edges.record_team(team, reward);
        }
    }
}

#[derive(Clone, Debug, Serialize)]
enum Spec {
    PgActor(PgActorConfig),
    RStdp(RStdpRouterConfig),
    TrustMeshHebbian(HebbianConfig, TrustEdgesConfig, usize),
    MeshTrustHebbian(HebbianConfig, MeshConfig, TrustEdgesConfig),
    Keyword,
    Thompson,
    Hebbian(HebbianConfig),
    MeshHebbian(HebbianConfig, MeshConfig),
    MeshThompson(MeshConfig),
}

fn build(spec: &Spec, n: usize) -> Box<dyn Policy> {
    match spec {
        Spec::Keyword => Box::new(Single(KeywordRouter::new(World::keyword_map(), n))),
        Spec::PgActor(c) => Box::new(Single(PgActorRouter::new(n, c.clone()))),
        Spec::RStdp(c) => Box::new(Single(RStdpRouter::new(n, c.clone()))),
        Spec::TrustMeshHebbian(c, t, k) => Box::new(TrustMesh {
            leader: HebbianRouter::new(n, c.clone()),
            edges: TrustEdges::new(n, t.clone()),
            team_size: *k,
        }),
        Spec::Thompson => Box::new(Single(ThompsonRouter::new(n))),
        Spec::Hebbian(c) => Box::new(Single(HebbianRouter::new(n, c.clone()))),
        Spec::MeshHebbian(c, m) => Box::new(Mesh(NeuralMesh::new(n, HebbianRouter::new(n, c.clone()), m.clone()))),
        Spec::MeshTrustHebbian(c, m, t) => Box::new(Mesh(
            NeuralMesh::new(n, HebbianRouter::new(n, c.clone()), m.clone()).with_trust_edges(t.clone()),
        )),
        Spec::MeshThompson(m) => Box::new(Mesh(NeuralMesh::new(n, ThompsonRouter::new(n), m.clone()))),
    }
}

// ---------------------------------------------------------------------------
// One run = one policy on one seed.

#[derive(Clone, Debug, Default, Serialize)]
struct RunMetrics {
    success: f64,
    /// mean expected regret: oracle success prob - success prob of the team used
    regret: f64,
    pre: f64,
    early_post: f64,
    late_post: f64,
    fail_window: f64,
    team_tasks: f64,
    solo_tasks: f64,
    /// mean team size actually used on team tasks
    team_size_used: f64,
    route_ns: f64,
}

fn rate(s: f64, n: f64) -> f64 {
    if n > 0.0 { s / n } else { f64::NAN }
}

/// `delay`: feedback for task t is delivered only after task t+delay has been routed
/// (0 = immediate). Outcomes/metrics are unaffected; only learning is delayed.
fn run(spec: &Spec, seed: u64, wcfg: &WorldConfig, time_it: bool, delay: usize) -> RunMetrics {
    let world = World::new(seed, wcfg.clone());
    let n = world.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut pol = build(spec, n);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut acc = [[0.0f64; 2]; 6]; // [succ, count] for pre, early, late, fail, team, solo
    let (mut succ, mut regret, mut team_sz) = (0.0, 0.0, 0.0);
    let mut route_ns = 0u128;
    let mut pending: std::collections::VecDeque<(usize, Vec<usize>, f64)> = std::collections::VecDeque::new();
    for t in 0..world.len() {
        let st = world.task(t);
        let team = if time_it {
            let t0 = Instant::now();
            let team = pol.assign(&st.task, &cands, &mut rng);
            route_ns += t0.elapsed().as_nanos();
            team
        } else {
            pol.assign(&st.task, &cands, &mut rng)
        };
        assert!(!team.is_empty(), "policy returned an empty team");
        // On solo tasks only the leader works; helpers are not engaged.
        let used: &[usize] = if st.team_task { &team } else { &team[..1] };
        let r = world.outcome(t, used);
        regret += world.oracle_prob(t) - world.success_prob(t, used);
        succ += r;
        pending.push_back((t, used.to_vec(), r));
        while let Some((t0, _, _)) = pending.front() {
            if t - *t0 < delay {
                break;
            }
            let (t0, tm, rr) = pending.pop_front().unwrap();
            pol.feedback(&world.task(t0).task, &tm, rr);
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
        if st.team_task {
            team_sz += used.len() as f64;
        }
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
        team_size_used: rate(team_sz, acc[4][1]),
        route_ns: if time_it { route_ns as f64 / nt } else { f64::NAN },
    }
}

/// Run a spec over seeds in parallel; results in seed order.
fn run_seeds(spec: &Spec, seeds: &[u64], wcfg: &WorldConfig, delay: usize) -> Vec<RunMetrics> {
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).max(1);
    let chunk = seeds.len().div_ceil(threads).max(1);
    let mut out = vec![RunMetrics::default(); seeds.len()];
    std::thread::scope(|s| {
        for (slot, ss) in out.chunks_mut(chunk).zip(seeds.chunks(chunk)) {
            s.spawn(move || {
                for (o, &seed) in slot.iter_mut().zip(ss) {
                    *o = run(spec, seed, wcfg, false, delay);
                }
            });
        }
    });
    out
}

fn field(ms: &[RunMetrics], f: fn(&RunMetrics) -> f64) -> Vec<f64> {
    ms.iter().map(f).collect()
}

// ---------------------------------------------------------------------------
// Reporting.

const METRICS: [(&str, fn(&RunMetrics) -> f64); 9] = [
    ("success", |m| m.success),
    ("regret", |m| m.regret),
    ("pre_drift", |m| m.pre),
    ("early_post", |m| m.early_post),
    ("late_post", |m| m.late_post),
    ("fail_window", |m| m.fail_window),
    ("team_tasks", |m| m.team_tasks),
    ("solo_tasks", |m| m.solo_tasks),
    ("team_size_used", |m| m.team_size_used),
];

struct Evaluated {
    name: String,
    runs: Vec<RunMetrics>,
}

fn print_table(title: &str, evals: &[Evaluated], ns: &[(String, f64)]) {
    println!("\n=== {title} ===");
    println!(
        "{:<30} {:>16} {:>7} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>5} {:>8}",
        "config", "success ±95%CI", "regret", "pre", "early", "late", "failW", "team", "solo", "tsz", "ns/route"
    );
    for e in evals {
        let s = summarize(&field(&e.runs, |m| m.success));
        let m = |f: fn(&RunMetrics) -> f64| summarize(&field(&e.runs, f)).mean;
        let nsr = ns.iter().find(|(n, _)| *n == e.name).map(|x| x.1).unwrap_or(f64::NAN);
        println!(
            "{:<30} {:>8.4} ±{:>6.4} {:>7.4} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>5.2} {:>8.0}",
            e.name,
            s.mean,
            s.half_width,
            m(|x| x.regret),
            m(|x| x.pre),
            m(|x| x.early_post),
            m(|x| x.late_post),
            m(|x| x.fail_window),
            m(|x| x.team_tasks),
            m(|x| x.solo_tasks),
            m(|x| x.team_size_used),
            nsr
        );
    }
}

fn print_diffs(evals: &[Evaluated], base: &str, metric: (&str, fn(&RunMetrics) -> f64)) -> Vec<(String, PairedDiff)> {
    let b = evals.iter().find(|e| e.name == base).expect("baseline present");
    let bv = field(&b.runs, metric.1);
    println!("\n  paired {} diff vs {base} (config - {base}): mean [95% CI]  wins/losses/ties", metric.0);
    let mut out = vec![];
    for e in evals.iter().filter(|e| e.name != base) {
        let d = paired_diff(&field(&e.runs, metric.1), &bv);
        println!(
            "  {:<30} {:+.4} [{:+.4}, {:+.4}]  {:>2}/{:>2}/{:>2}  {}",
            e.name,
            d.diff.mean,
            d.diff.lo,
            d.diff.hi,
            d.wins,
            d.losses,
            d.ties,
            if d.significant { if d.diff.mean > 0.0 { "SIG +" } else { "SIG -" } } else { "n.s." }
        );
        out.push((e.name.clone(), d));
    }
    out
}

fn metric_summaries(runs: &[RunMetrics]) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    for (name, f) in METRICS.iter() {
        let s: Summary = summarize(&field(runs, *f));
        m.insert(name.to_string(), serde_json::to_value(s).unwrap());
    }
    serde_json::Value::Object(m)
}

fn eval_block(
    title: &str,
    configs: &[(String, Spec)],
    seeds: &[u64],
    wcfg: &WorldConfig,
    ns: &[(String, f64)],
    verbose: bool,
    delay: usize,
) -> (Vec<Evaluated>, serde_json::Value) {
    let evals: Vec<Evaluated> =
        configs.iter().map(|(n, s)| Evaluated { name: n.clone(), runs: run_seeds(s, seeds, wcfg, delay) }).collect();
    print_table(title, &evals, ns);
    let mut diffs = serde_json::Map::new();
    for base in ["keyword", "thompson"] {
        let mut per_metric = serde_json::Map::new();
        let metrics: Vec<(&str, fn(&RunMetrics) -> f64)> =
            if verbose { vec![METRICS[0], METRICS[1], METRICS[5], METRICS[6]] } else { vec![METRICS[0]] };
        for metric in metrics {
            let d = print_diffs(&evals, base, metric);
            per_metric.insert(
                metric.0.to_string(),
                serde_json::Value::Object(d.into_iter().map(|(n, d)| (n, serde_json::to_value(d).unwrap())).collect()),
            );
        }
        diffs.insert(format!("vs_{base}"), serde_json::Value::Object(per_metric));
    }
    let configs_json: serde_json::Map<String, serde_json::Value> = evals
        .iter()
        .map(|e| {
            (
                e.name.clone(),
                json!({ "summary": metric_summaries(&e.runs), "per_seed_success": field(&e.runs, |m| m.success) }),
            )
        })
        .collect();
    // Ablations: each mesh against its own bare leader (does the mesh add value?).
    let pairs: [(&str, &str); 6] = [
        ("mesh-k1-hebbian", "hebbian"),
        ("mesh-k2-hebbian", "hebbian"),
        ("mesh-k3-hebbian", "hebbian"),
        ("mesh-tuned", "hebbian-tuned"),
        ("mesh-k2-thompson", "thompson"),
        ("mesh-k3-thompson", "thompson"),
    ];
    let mut abl = serde_json::Map::new();
    println!("
  ABLATION: mesh vs its bare leader (mesh - leader): mean [95% CI] wins/losses/ties");
    let mut pairs: Vec<(String, String)> = pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    pairs.push(("trust-mesh-tuned".to_string(), "hebbian-tuned".to_string()));
    pairs.push(("mesh-tuned".to_string(), "trust-mesh-tuned".to_string()));
    pairs.push(("mesh-trust-tuned".to_string(), "hebbian-tuned".to_string()));
    pairs.push(("mesh-tuned".to_string(), "mesh-trust-tuned".to_string()));
    pairs.push(("mesh-k2-trust".to_string(), "hebbian".to_string()));
    pairs.push(("pg-actor-tuned".to_string(), "hebbian-tuned".to_string()));
    pairs.push(("r-stdp-tuned".to_string(), "hebbian-tuned".to_string()));
    if let Some(e) = evals.iter().find(|e| e.name.starts_with("mesh-tuned-")) {
        pairs.push((e.name.clone(), "hebbian-tuned".to_string()));
        pairs.push(("mesh-tuned".to_string(), e.name.clone()));
    }
    for (a, b) in pairs {
        let (Some(ea), Some(eb)) = (evals.iter().find(|e| e.name == a), evals.iter().find(|e| e.name == b)) else {
            continue;
        };
        let mut row = serde_json::Map::new();
        let mut line = format!("  {:<16} vs {:<14}", a, b);
        for (mn, f) in [METRICS[0], METRICS[5], METRICS[6], METRICS[7]] {
            let d = paired_diff(&field(&ea.runs, f), &field(&eb.runs, f));
            line += &format!(
                " | {} {:+.4} [{:+.4},{:+.4}] {}/{}{}",
                mn, d.diff.mean, d.diff.lo, d.diff.hi, d.wins, d.losses, if d.significant { "*" } else { "" }
            );
            row.insert(mn.to_string(), serde_json::to_value(d).unwrap());
        }
        println!("{line}");
        abl.insert(format!("{a}__vs__{b}"), serde_json::Value::Object(row));
    }
    let v = json!({ "title": title, "seeds": seeds, "configs": configs_json, "paired": diffs, "ablation_vs_leader": abl });
    (evals, v)
}

// ---------------------------------------------------------------------------

fn mean_success(spec: &Spec, seeds: &[u64], wcfg: &WorldConfig) -> f64 {
    summarize(&field(&run_seeds(spec, seeds, wcfg, 0), |m| m.success)).mean
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let n_tasks: usize = get("--tasks").and_then(|s| s.parse().ok()).unwrap_or(2000);
    let out_path = get("--out").unwrap_or_else(|| "results/results.json".to_string());
    let wcfg = WorldConfig { n_tasks, ..WorldConfig::default() };
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let eval_seeds: Vec<u64> = (1..=30).collect();
    let held_seeds: Vec<u64> = (31..=60).collect();
    let t_start = Instant::now();

    println!("neural-mesh experiment: {} tasks/run, world = {:?}", n_tasks, wcfg);

    // ---- 1. tuning on seeds 1-8 only -------------------------------------
    let base_h = HebbianConfig::default();
    let mut h_grid = vec![];
    for &eta in &[0.05, 0.1, 0.25, 0.5, 0.8] {
        for &temperature in &[0.02, 0.05, 0.1, 0.2] {
            for &decay in &[0.0, 0.001, 0.005, 0.01, 0.03] {
                let c = HebbianConfig { eta, temperature, decay, ..base_h.clone() };
                let sc = mean_success(&Spec::Hebbian(c.clone()), &tune_seeds, &wcfg);
                h_grid.push((c, sc));
            }
        }
    }
    h_grid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let h_best = h_grid[0].0.clone();
    println!("\n[tuning seeds 1-8] hebbian grid: {} configs; top 5:", h_grid.len());
    for (c, s) in h_grid.iter().take(5) {
        println!("  eta {:<5} temp {:<5} decay {:<6} -> success {:.4}", c.eta, c.temperature, c.decay, s);
    }
    let h_default_tune = h_grid.iter().find(|(c, _)| *c == base_h).map(|x| x.1).unwrap_or(f64::NAN);
    println!("  (default hebbian on tuning seeds: {:.4})", h_default_tune);

    let base_m = MeshConfig::default();
    let mut m_grid = vec![];
    for &edge_eta in &[0.05, 0.1, 0.2, 0.4] {
        for &edge_decay in &[0.0, base_m.edge_decay, 0.01] {
            for &team_size in &[1usize, 2, 3] {
                let m = MeshConfig { edge_eta, edge_decay, team_size, ..base_m.clone() };
                let sc = mean_success(&Spec::MeshHebbian(h_best.clone(), m.clone()), &tune_seeds, &wcfg);
                m_grid.push((m, sc));
            }
        }
    }
    m_grid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    m_grid.dedup_by(|a, b| a.0 == b.0);
    let m_best = m_grid[0].0.clone();
    println!("\n[tuning seeds 1-8] mesh grid (leader = tuned hebbian): {} configs; top 5:", m_grid.len());
    for (m, s) in m_grid.iter().take(5) {
        println!(
            "  edge_eta {:<5} edge_decay {:<6} team_size {} -> success {:.4}",
            m.edge_eta, m.edge_decay, m.team_size, s
        );
    }
    // Extra ablation config: if the tuned mesh uses teams, also evaluate it with
    // team_size 1 (health/benching only); otherwise evaluate the best team config.
    let (abl_name, m_abl) = if m_best.team_size >= 2 {
        ("mesh-tuned-k1", MeshConfig { team_size: 1, ..m_best.clone() })
    } else {
        ("mesh-tuned-team", m_grid.iter().find(|(m, _)| m.team_size >= 2).map(|x| x.0.clone()).unwrap())
    };

    // variants (src/variants.rs): PgActor, R-STDP, TrustEdges mesh - same protocol
    let base_pg = PgActorConfig::default();
    let mut pg_grid = vec![];
    for &eta in &[0.1, 0.25, 0.5, 1.0, 2.0] {
        for &theta_clip in &[2.0, 4.0, 8.0] {
            let c = PgActorConfig { eta, theta_clip, ..base_pg.clone() };
            let sc = mean_success(&Spec::PgActor(c.clone()), &tune_seeds, &wcfg);
            pg_grid.push((c, sc));
        }
    }
    pg_grid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let pg_best = pg_grid[0].0.clone();
    println!("
[tuning seeds 1-8] pg-actor grid: {} configs; best eta {} clip {} -> {:.4}", pg_grid.len(), pg_best.eta, pg_best.theta_clip, pg_grid[0].1);

    let base_rs = RStdpRouterConfig::default();
    let mut rs_grid = vec![];
    for &eta in &[0.25f32, 0.5, 1.0] {
        for &temperature in &[0.05, 0.1, 0.2] {
            for &tau_e in &[0.5f32, 1.0, 2.0, 4.0] {
                let c = RStdpRouterConfig { eta, temperature, tau_e, ..base_rs.clone() };
                let sc = mean_success(&Spec::RStdp(c.clone()), &tune_seeds, &wcfg);
                rs_grid.push((c, sc));
            }
        }
    }
    rs_grid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let rs_best = rs_grid[0].0.clone();
    println!("[tuning seeds 1-8] r-stdp grid: {} configs; best eta {} temp {} tau_e {} -> {:.4}", rs_grid.len(), rs_best.eta, rs_best.temperature, rs_best.tau_e, rs_grid[0].1);

    let base_te = TrustEdgesConfig::default();
    let mut te_grid = vec![];
    for &mu in &[0.1, 0.25] {
        for &psi in &[0.0, 0.05] {
            for &k in &[2usize, 3] {
                let t = TrustEdgesConfig { mu, psi, ..base_te.clone() };
                let sc = mean_success(&Spec::TrustMeshHebbian(h_best.clone(), t.clone(), k), &tune_seeds, &wcfg);
                te_grid.push(((t, k), sc));
            }
        }
    }
    te_grid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let (te_best, te_k) = te_grid[0].0.clone();
    println!("[tuning seeds 1-8] trust-mesh grid (leader = tuned hebbian): {} configs; best mu {} psi {} k {} -> {:.4}", te_grid.len(), te_best.mu, te_best.psi, te_k, te_grid[0].1);

    let mut mt_grid = vec![];
    for &mu in &[0.1, 0.25] {
        for &psi in &[0.0, 0.05] {
            for &k in &[2usize, 3] {
                let t = TrustEdgesConfig { mu, psi, ..TrustEdgesConfig::default() };
                let m = MeshConfig { team_size: k, ..base_m.clone() };
                let sc = mean_success(&Spec::MeshTrustHebbian(h_best.clone(), m.clone(), t.clone()), &tune_seeds, &wcfg);
                mt_grid.push(((m, t), sc));
            }
        }
    }
    mt_grid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let (mt_m, mt_t) = mt_grid[0].0.clone();
    println!("[tuning seeds 1-8] NeuralMesh+with_trust_edges grid: {} configs; best mu {} psi {} k {} -> {:.4}", mt_grid.len(), mt_t.mu, mt_t.psi, mt_m.team_size, mt_grid[0].1);

    // ---- 2. configs -------------------------------------------------------
    let h_nodecay = HebbianConfig { decay: 0.0, ..base_h.clone() };
    let ms = |k: usize| MeshConfig { team_size: k, ..base_m.clone() };
    let configs: Vec<(String, Spec)> = vec![
        ("keyword".into(), Spec::Keyword),
        ("thompson".into(), Spec::Thompson),
        ("hebbian".into(), Spec::Hebbian(base_h.clone())),
        ("hebbian-no-decay".into(), Spec::Hebbian(h_nodecay)),
        ("hebbian-tuned".into(), Spec::Hebbian(h_best.clone())),
        ("mesh-k1-hebbian".into(), Spec::MeshHebbian(base_h.clone(), ms(1))),
        ("mesh-k2-hebbian".into(), Spec::MeshHebbian(base_h.clone(), ms(2))),
        ("mesh-k3-hebbian".into(), Spec::MeshHebbian(base_h.clone(), ms(3))),
        ("mesh-k2-thompson".into(), Spec::MeshThompson(ms(2))),
        ("mesh-k3-thompson".into(), Spec::MeshThompson(ms(3))),
        ("mesh-tuned".into(), Spec::MeshHebbian(h_best.clone(), m_best.clone())),
        (abl_name.into(), Spec::MeshHebbian(h_best.clone(), m_abl.clone())),
        ("pg-actor".into(), Spec::PgActor(base_pg.clone())),
        ("pg-actor-tuned".into(), Spec::PgActor(pg_best.clone())),
        ("r-stdp".into(), Spec::RStdp(base_rs.clone())),
        ("r-stdp-tuned".into(), Spec::RStdp(rs_best.clone())),
        ("mesh-k2-trust".into(), Spec::MeshTrustHebbian(base_h.clone(), ms(2), TrustEdgesConfig::default())),
        ("mesh-k3-trust".into(), Spec::MeshTrustHebbian(base_h.clone(), ms(3), TrustEdgesConfig::default())),
        ("mesh-trust-tuned".into(), Spec::MeshTrustHebbian(h_best.clone(), mt_m.clone(), mt_t.clone())),
        ("trust-mesh-tuned".into(), Spec::TrustMeshHebbian(h_best.clone(), te_best.clone(), te_k)),
    ];

    // ---- 5. latency (single-threaded, seed 1) -----------------------------
    // Instant::now() pair overhead is measured and subtracted.
    let overhead = {
        let n = 200_000;
        let t0 = Instant::now();
        let mut acc = 0u128;
        for _ in 0..n {
            let a = Instant::now();
            acc += a.elapsed().as_nanos();
        }
        let _ = t0;
        acc as f64 / n as f64
    };
    let ns: Vec<(String, f64)> = configs
        .iter()
        .map(|(n, s)| {
            let reps = 3;
            let v: f64 = (0..reps).map(|_| run(s, 1, &wcfg, true, 0).route_ns).sum::<f64>() / reps as f64;
            (n.clone(), (v - overhead).max(0.0))
        })
        .collect();
    println!("\n[latency] Instant overhead {:.0} ns subtracted; ns per route call (seed 1, mean of 3 runs)", overhead);

    // ---- 3. evaluation ----------------------------------------------------
    let (_, j_eval) = eval_block("EVAL seeds 1-30 (seeds 1-8 used for tuning)", &configs, &eval_seeds, &wcfg, &ns, true, 0);
    let (held, j_held) = eval_block("HELD-OUT seeds 31-60 (honest)", &configs, &held_seeds, &wcfg, &ns, true, 0);

    // ---- 4. robustness: task type revealed --------------------------------
    let wcfg_rev = WorldConfig { reveal_type: true, ..wcfg.clone() };
    let (_, j_rev) =
        eval_block("ROBUSTNESS held-out seeds 31-60, task_type revealed", &configs, &held_seeds, &wcfg_rev, &ns, false, 0);

    // ---- 4b. robustness: delayed reward (feedback arrives DELAY tasks later) --
    const DELAY: usize = 25;
    let (_, j_delay) = eval_block(
        &format!("ROBUSTNESS held-out seeds 31-60, reward delayed by {DELAY} tasks"),
        &configs,
        &held_seeds,
        &wcfg,
        &ns,
        false,
        DELAY,
    );

    // ---- verdict helper: significant held-out success wins/losses ---------
    println!("\n=== held-out significance summary (success; CI excludes 0) ===");
    let mut sig = vec![];
    for a in &held {
        for b in &held {
            if a.name == b.name {
                continue;
            }
            let d = paired_diff(&field(&a.runs, |m| m.success), &field(&b.runs, |m| m.success));
            if d.significant && d.diff.mean > 0.0 {
                sig.push(json!({"better": a.name, "worse": b.name, "diff": d.diff.mean, "lo": d.diff.lo, "hi": d.diff.hi, "wins": d.wins}));
                println!("  {:<20} > {:<20} {:+.4} [{:+.4}, {:+.4}] wins {}/{}", a.name, b.name, d.diff.mean, d.diff.lo, d.diff.hi, d.wins, d.diff.n);
            }
        }
    }

    let out = json!({
        "world": {
            "n_tasks": wcfg.n_tasks, "team_frac": wcfg.team_frac, "confound_prob": wcfg.confound_prob,
            "fail_frac": wcfg.fail_frac, "oracle_team_max": wcfg.oracle_team_max, "reveal_type_main": wcfg.reveal_type,
            "agents": neural_mesh::sim::AGENTS, "types": neural_mesh::sim::TYPES,
            "trap_types": neural_mesh::sim::TRAP_TYPES, "drift_types": neural_mesh::sim::DRIFT_TYPES,
        },
        "tuning": {
            "seeds": tune_seeds,
            "hebbian_grid": h_grid.iter().map(|(c, s)| json!({"cfg": c, "success": s})).collect::<Vec<_>>(),
            "hebbian_best": h_best,
            "mesh_grid": m_grid.iter().map(|(m, s)| json!({"cfg": m, "success": s})).collect::<Vec<_>>(),
            "mesh_best": m_best,
            "mesh_ablation": { "name": abl_name, "cfg": m_abl },
            "pg_actor_grid": pg_grid.iter().map(|(c, s)| json!({"cfg": c, "success": s})).collect::<Vec<_>>(),
            "pg_actor_best": pg_best,
            "r_stdp_grid": rs_grid.iter().map(|(c, s)| json!({"cfg": c, "success": s})).collect::<Vec<_>>(),
            "r_stdp_best": rs_best,
            "trust_mesh_grid": te_grid.iter().map(|((t, k), s)| json!({"cfg": t, "team_size": k, "success": s})).collect::<Vec<_>>(),
            "mesh_with_trust_edges_grid": mt_grid.iter().map(|((m, t), s)| json!({"mesh": m, "trust": t, "success": s})).collect::<Vec<_>>(),
            "mesh_with_trust_edges_best": { "mesh": mt_m, "trust": mt_t },
            "trust_mesh_best": { "cfg": te_best, "team_size": te_k },
        },
        "configs": configs.iter().map(|(n, s)| json!({"name": n, "spec": s})).collect::<Vec<_>>(),
        "latency_ns_per_route": ns.iter().map(|(n, v)| (n.clone(), json!(v))).collect::<serde_json::Map<_, _>>(),
        "eval_seeds_1_30": j_eval,
        "heldout_seeds_31_60": j_held,
        "robustness_type_revealed_31_60": j_rev,
        "robustness_delayed_reward_31_60": j_delay,
        "heldout_significant_success_wins": sig,
        "elapsed_s": t_start.elapsed().as_secs_f64(),
    });
    if let Some(dir) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).expect("write results");
    println!("\nwrote {out_path} ({:.1}s)", t_start.elapsed().as_secs_f64());
}
