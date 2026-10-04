//! D3 lab (ROUTER-ALGO): hierarchical pooling of evidence across a task's features
//! (empirical-Bayes style sharing between co-occurring features and shrinkage of a
//! challenger's prior toward its pooled per-agent rate) inside the guarded keyword
//! hand-over (prior::PooledGuarded). Opt-in; defaults of everything else unchanged.
//!
//! Protocol = x1_lab: tune on seeds 1-8 ONLY (deterministic, so the chosen config is
//! the same for any evaluation range = frozen), evaluate on --held a-b (default
//! 31-60; 61-90 = fresh confirmation), paired 95% t-CI vs the best of {keyword,
//! eps-greedy, pooled-thompson, hebbian-robust, warm(hebbian-robust), guarded(hebr)}
//! per (world, H), Holm over candidates x worlds x horizons.
//! Usage: cargo run --release --bin d3_pool [-- --out results/d3_pool.json] [--held 61-90]

use std::collections::HashMap;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, PoolConfig, PooledGuarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World as SimWorld, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::json;

const HORIZONS: [usize; 5] = [100, 250, 500, 1000, 2000];
const N_TASKS: usize = 2000;

// ------------------------------------------------------------------ worlds
#[derive(Clone, Debug)]
enum WorldKind {
    /// shortlab/w1_lab generator (copied; read-only reuse)
    Short { topics: usize, traps: usize, map_err: f64, confound: f64 },
    /// sim.rs world (drift at midpoint, node-failure window), solo routing
    Sim(Scenario),
}

#[derive(Clone, Debug)]
struct WorldSpec {
    name: &'static str,
    kind: WorldKind,
}

/// Unified routing world: tasks, expected success per (task, agent), pre-drawn u.
struct World {
    agents: usize,
    tasks: Vec<Task>,
    p: Vec<Vec<f64>>,
    u: Vec<f64>,
    kwmap: HashMap<String, usize>,
}

fn gen_short(seed: u64, topics: usize, traps: usize, map_err: f64, confound: f64) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(11));
    let (t_n, a_n) = (topics, 6usize);
    let mut p = vec![vec![0.0; a_n]; t_n];
    let mut best = vec![0usize; t_n];
    let mut second = vec![0usize; t_n];
    for t in 0..t_n {
        best[t] = rng.below(a_n);
        loop {
            second[t] = rng.below(a_n);
            if second[t] != best[t] {
                break;
            }
        }
        for a in 0..a_n {
            p[t][a] = if a == best[t] {
                0.85
            } else if a == second[t] {
                0.5
            } else {
                0.15 + 0.25 * rng.f64()
            };
        }
    }
    let mut order: Vec<usize> = (0..t_n).collect();
    for i in (1..t_n).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let trap: Vec<bool> = (0..t_n).map(|t| order.iter().take(traps).any(|&x| x == t)).collect();
    let mut kwmap = HashMap::new();
    for t in 0..t_n {
        for j in 0..3 {
            let a = if trap[t] {
                loop {
                    let a = rng.below(a_n);
                    if a != best[t] && a != second[t] {
                        break a;
                    }
                }
            } else if rng.f64() < map_err {
                rng.below(a_n)
            } else {
                best[t]
            };
            kwmap.insert(format!("t{t}k{j}"), a);
        }
    }
    let (mut tasks, mut pp, mut u) = (vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let t = rng.below(t_n);
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(confound) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        pp.push(p[t].clone());
        u.push(rng.f64());
    }
    World { agents: a_n, tasks, p: pp, u, kwmap }
}

fn gen_sim(seed: u64, sc: Scenario) -> World {
    let w = SimWorld::new(seed, WorldConfig { n_tasks: N_TASKS, scenario: sc, ..WorldConfig::default() });
    let n = w.n_agents();
    let tasks: Vec<Task> = w.tasks().iter().map(|s| s.task.clone()).collect();
    let p: Vec<Vec<f64>> = (0..w.len()).map(|t| (0..n).map(|a| w.success_prob(t, &[a])).collect()).collect();
    let u: Vec<f64> = w.tasks().iter().map(|s| s.u).collect();
    World { agents: n, tasks, p, u, kwmap: w.keyword_map_for() }
}

fn gen_world(seed: u64, s: &WorldSpec) -> World {
    match &s.kind {
        WorldKind::Short { topics, traps, map_err, confound } => gen_short(seed, *topics, *traps, *map_err, *confound),
        WorldKind::Sim(sc) => gen_sim(seed, *sc),
    }
}

fn worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec { name: "traps(main)", kind: WorldKind::Short { topics: 8, traps: 2, map_err: 0.10, confound: 0.25 } },
        WorldSpec { name: "correct-keywords", kind: WorldKind::Short { topics: 8, traps: 0, map_err: 0.0, confound: 0.25 } },
        WorldSpec { name: "many-topics(16)", kind: WorldKind::Short { topics: 16, traps: 2, map_err: 0.10, confound: 0.25 } },
        WorldSpec { name: "sim-random", kind: WorldKind::Sim(Scenario::Random) },
        WorldSpec { name: "sim-notraps", kind: WorldKind::Sim(Scenario::NoTraps) },
    ]
}

// ------------------------------------------------------------------ routers
fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

/// The x1 (router-thinker) verified guard configuration.
fn x1_guard() -> GuardConfig {
    GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    HebR,
    WarmHebR,
    GuardHebR(GuardConfig),
    Pool(PoolConfig),
}

fn drive<R: Router>(mut r: R, w: &World, seed: u64) -> ([f64; 5], R) {
    let cands: Vec<usize> = (0..w.agents).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut cum = 0.0;
    let mut out = [0.0; 5];
    let mut hi = 0;
    for (i, task) in w.tasks.iter().enumerate() {
        let a = r.select(task, &cands, &mut rng);
        let p = w.p[i][a];
        cum += p;
        r.update(task, a, if w.u[i] < p { 1.0 } else { 0.0 });
        if hi < 5 && i + 1 == HORIZONS[hi] {
            out[hi] = cum / (i + 1) as f64;
            hi += 1;
        }
    }
    (out, r)
}


/// Approximate learned-state bytes: rows x agents x bytes-per-cell (+ 32 B/row key overhead).
fn heb_bytes(h: &HebbianRouter) -> usize {
    h.rows().len() * (h.n_agents() * 8 + 8 + 32)
}

fn run(kind: &Kind, w: &World, seed: u64) -> ([f64; 5], usize) {
    let a = w.agents;
    let km = || w.kwmap.clone();
    let nf = {
        let mut s: Vec<&String> = w.tasks.iter().flat_map(|t| t.features.iter()).collect();
        s.sort();
        s.dedup();
        s.len()
    };
    match kind {
        Kind::Keyword => (drive(KeywordRouter::new(km(), a), w, seed).0, w.kwmap.len() * 40),
        Kind::Eps(g, e) => (drive(EpsGreedyRouter::new(a, *g, *e), w, seed).0, nf * (a * 16 + 32)),
        Kind::Pts(g) => (drive(PooledThompsonRouter::new(a, *g), w, seed).0, nf * (a * 16 + 32)),
        Kind::HebR => {
            let (o, r) = drive(HebbianRouter::new(a, heb_robust_cfg()), w, seed);
            (o, heb_bytes(&r))
        }
        Kind::WarmHebR => {
            let r = Warm::new(HebbianRouter::new(a, heb_robust_cfg()), km(), a, 1.0).with_evidence(Evidence::MeanFeature);
            let (o, r) = drive(r, w, seed);
            (o, heb_bytes(&r.inner) + nf * 40 + w.kwmap.len() * 40)
        }
        Kind::GuardHebR(g) => {
            let (o, r) = drive(Guarded::new(HebbianRouter::new(a, heb_robust_cfg()), km(), a, *g), w, seed);
            (o, heb_bytes(&r.inner) + r.n_rows() * (a * 16 + 32) + w.kwmap.len() * 40)
        }
        Kind::Pool(c) => {
            let (o, r) = drive(PooledGuarded::new(HebbianRouter::new(a, heb_robust_cfg()), km(), a, *c), w, seed);
            // counts rows + co-occurrence pairs (~3 features/task => ~nf*3 pairs x 24 B)
            (o, heb_bytes(&r.inner) + nf * (a * 16 + 32) + nf * 3 * 24 + w.kwmap.len() * 40)
        }
    }
}

fn eval(kind: &Kind, seeds: &[u64], s: &WorldSpec) -> Vec<[f64; 5]> {
    seeds.par_iter().map(|&sd| run(kind, &gen_world(sd, s), sd).0).collect()
}

fn col(r: &[[f64; 5]], h: usize) -> Vec<f64> {
    r.iter().map(|x| x[h]).collect()
}

/// Mean prefix-success per (world, horizon) on `seeds`.
fn means(kind: &Kind, seeds: &[u64], ws: &[WorldSpec]) -> Vec<[f64; 5]> {
    ws.iter()
        .map(|s| {
            let r = eval(kind, seeds, s);
            let mut m = [0.0; 5];
            for h in 0..5 {
                m[h] = r.iter().map(|x| x[h]).sum::<f64>() / r.len() as f64;
            }
            m
        })
        .collect()
}

/// Regret-averse tuning objective on seeds 1-8: with d = cand - best baseline per
/// (world, horizon), obj = mean(d) + mean(min(d, 0)) (losses count double).
fn obj(kind: &Kind, seeds: &[u64], ws: &[WorldSpec], best_b: &[[f64; 5]]) -> f64 {
    let m = means(kind, seeds, ws);
    let mut d = vec![];
    for (w, row) in m.iter().enumerate() {
        for h in 0..5 {
            d.push(row[h] - best_b[w][h]);
        }
    }
    let n = d.len() as f64;
    d.iter().sum::<f64>() / n + d.iter().map(|x| x.min(0.0)).sum::<f64>() / n
}

fn tune(label: &str, cands: Vec<Kind>, seeds: &[u64], ws: &[WorldSpec], best_b: &[[f64; 5]]) -> Kind {
    let n = cands.len();
    let scored: Vec<(f64, Kind)> = cands.into_iter().map(|k| (obj(&k, seeds, ws, best_b), k)).collect();
    let best = scored.into_iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
    println!("[tune 1-8] {label}: {n} configs, best obj {:+.5} -> {:?}", best.0, best.1);
    best.1
}

fn bench(kind: &Kind, s: &WorldSpec) -> (f64, usize) {
    let w = gen_world(1, s);
    let reps = 10u64;
    let t0 = std::time::Instant::now();
    let mut bytes = 0;
    for r in 0..reps {
        bytes = run(kind, &w, r).1;
    }
    (t0.elapsed().as_nanos() as f64 / (reps as f64 * N_TASKS as f64), bytes)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/d3_pool.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let ws = worlds();
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let (h0, h1) = args
        .iter()
        .position(|a| a == "--held")
        .and_then(|i| args.get(i + 1))
        .and_then(|r| r.split_once('-'))
        .map(|(a, b)| (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap()))
        .unwrap_or((31, 60));
    let held: Vec<u64> = (h0..=h1).collect();
    println!("d3_pool: {N_TASKS} tasks/run, tune seeds 1-8 (all 5 worlds), held-out {h0}-{h1}");

    let baselines: Vec<(&str, Kind)> = vec![
        ("keyword", Kind::Keyword),
        ("eps-greedy(tuned)", Kind::Eps(0.95, 0.02)),
        ("pooled-ts(tuned)", Kind::Pts(1.0)),
        ("hebbian-robust", Kind::HebR),
        ("warm(hebbian-robust)", Kind::WarmHebR),
        ("guarded(hebr)", Kind::GuardHebR(x1_guard())),
    ];
    // ---------------- tuning (seeds 1-8 only) ----------------
    let bm: Vec<Vec<[f64; 5]>> = baselines.iter().map(|(_, k)| means(k, &tune_seeds, &ws)).collect();
    let best_b: Vec<[f64; 5]> = (0..ws.len())
        .map(|w| {
            let mut m = [f64::NEG_INFINITY; 5];
            for b in &bm {
                for h in 0..5 {
                    m[h] = m[h].max(b[w][h]);
                }
            }
            m
        })
        .collect();
    let mut grid = vec![];
    for &lam in &[0.0, 0.5, 1.0, 2.0, 4.0] {
        for &jmin in &[0.2, 0.35, 0.5] {
            for &agent_m in &[0.0, 5.0, 20.0] {
                if lam == 0.0 && jmin != 0.2 {
                    continue; // jmin irrelevant without pooling
                }
                for &tau in &[0.55, 0.65] {
                    for &disc in &[0.97, 0.99] {
                        for &norm in &[false, true] { for &kw_own in &[false, true] { if lam == 0.0 && (norm || kw_own) { continue; } grid.push(PoolConfig { guard: GuardConfig { tau, discount: disc, ..x1_guard() }, lam, jmin, agent_m, norm, kw_own }); } }
                    }
                }
            }
        }
    }
    if quick {
        grid.truncate(8);
    }
    let sub = |f: &dyn Fn(&PoolConfig) -> bool| -> Vec<Kind> { grid.iter().filter(|c| f(c)).map(|c| Kind::Pool(*c)).collect() };
    let off = tune("pguarded OFF (lam=0,m=0; guard retune only)", sub(&|c| c.lam == 0.0 && c.agent_m == 0.0), &tune_seeds, &ws, &best_b);
    let lam = tune("pguarded feature-group pooling (lam>0, m=0)", sub(&|c| c.lam > 0.0 && c.agent_m == 0.0), &tune_seeds, &ws, &best_b);
    let agent = tune("pguarded agent-prior shrink (lam=0, m>0)", sub(&|c| c.lam == 0.0 && c.agent_m > 0.0), &tune_seeds, &ws, &best_b);
    let both = tune("pguarded both (lam>0, m>0)", sub(&|c| c.lam > 0.0 && c.agent_m > 0.0), &tune_seeds, &ws, &best_b);
    let cands: Vec<(&str, Kind)> = vec![
        ("pg group-pool", lam),
        ("pg agent-shrink", agent),
        ("pg both", both),
        ("pg off (guard retune only)", off),
    ];

    // ---------------- held-out evaluation ----------------
    struct Test {
        world: &'static str,
        cand: usize,
        h: usize,
        best_b: usize,
        d: neural_mesh::stats::PairedDiff,
        dg: neural_mesh::stats::PairedDiff,
        p: f64,
    }
    let gi = baselines.iter().position(|b| b.0 == "guarded(hebr)").unwrap();
    let mut tests: Vec<Test> = vec![];
    let mut means_json = vec![];
    for s in &ws {
        let base: Vec<Vec<[f64; 5]>> = baselines.iter().map(|(_, k)| eval(k, &held, s)).collect();
        let cr: Vec<Vec<[f64; 5]>> = cands.iter().map(|(_, k)| eval(k, &held, s)).collect();
        println!("\n=== world {} | held-out {h0}-{h1} | mean expected success over first H ===", s.name);
        println!("{:<28}{:>8}{:>8}{:>8}{:>8}{:>8}", "router", 100, 250, 500, 1000, 2000);
        let mut mj = vec![];
        for (i, (n, _)) in baselines.iter().enumerate() {
            let m: Vec<f64> = (0..5).map(|h| summarize(&col(&base[i], h)).mean).collect();
            println!("{:<28}{}", n, m.iter().map(|x| format!("{:>8.4}", x)).collect::<String>());
            mj.push(json!({"router": n, "means": m}));
        }
        for (i, (n, _)) in cands.iter().enumerate() {
            let m: Vec<f64> = (0..5).map(|h| summarize(&col(&cr[i], h)).mean).collect();
            println!("{:<28}{}   <- candidate", n, m.iter().map(|x| format!("{:>8.4}", x)).collect::<String>());
            mj.push(json!({"router": n, "means": m}));
        }
        means_json.push(json!({"world": s.name, "rows": mj}));
        for h in 0..5 {
            let bb = (0..base.len())
                .max_by(|&a, &b| summarize(&col(&base[a], h)).mean.partial_cmp(&summarize(&col(&base[b], h)).mean).unwrap())
                .unwrap();
            for ci in 0..cands.len() {
                let (c, b) = (col(&cr[ci], h), col(&base[bb], h));
                tests.push(Test {
                    world: s.name,
                    cand: ci,
                    h,
                    best_b: bb,
                    d: paired_diff(&c, &b),
                    dg: paired_diff(&c, &col(&base[gi], h)),
                    p: paired_t_pvalue(&c, &b),
                });
            }
        }
    }
    let adj = holm_adjust(&tests.iter().map(|t| t.p).collect::<Vec<_>>());
    println!("\n=== paired diffs vs BEST baseline per (world,H) (incl. guarded(hebr)); Holm over {} tests ===", tests.len());
    let mut tj = vec![];
    for (i, t) in tests.iter().enumerate() {
        let sig = adj[i] < 0.05;
        let verdict = if sig && t.d.diff.mean > 0.0 { "WIN" } else if sig { "LOSS" } else { "ns" };
        println!(
            "{:<18} {:<26} H={:<5} vs {:<22} {:+.4} [{:+.4},{:+.4}] holm p={:.4} {:<4} | vs guarded {:+.4} [{:+.4},{:+.4}]",
            t.world, cands[t.cand].0, HORIZONS[t.h], baselines[t.best_b].0,
            t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, adj[i], verdict, t.dg.diff.mean, t.dg.diff.lo, t.dg.diff.hi
        );
        tj.push(json!({"world": t.world, "cand": cands[t.cand].0, "horizon": HORIZONS[t.h], "best_baseline": baselines[t.best_b].0,
            "diff": t.d.diff, "wins": t.d.wins, "losses": t.d.losses, "p_raw": t.p, "holm_p": adj[i], "verdict": verdict,
            "diff_vs_guarded": t.dg.diff}));
    }
    println!("\n=== cost: ns per (select+update) and approx learned-state bytes after 2000 tasks (traps(main), seed 1) ===");
    let mut bj = vec![];
    for (n, k) in baselines.iter().chain(cands.iter()) {
        let (ns, bytes) = bench(k, &ws[0]);
        println!("{:<28}{:>10.0} ns/route {:>10} B", n, ns, bytes);
        bj.push(json!({"router": n, "ns_per_route": ns, "bytes": bytes}));
    }
    let j = json!({
        "held_out": format!("{h0}-{h1}"),
        "protocol": "tune seeds 1-8 (all worlds, regret-averse mean over horizons vs best of 6 baselines incl guarded(hebr)); held-out seeds as given; paired t vs best baseline per (world,H); Holm over candidate x world x horizon",
        "tuned": cands.iter().map(|(n, k)| json!({"cand": n, "config": format!("{:?}", k)})).collect::<Vec<_>>(),
        "means": means_json, "tests": tj, "cost": bj
    });
    let _ = std::fs::create_dir_all("results");
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out}");
}
