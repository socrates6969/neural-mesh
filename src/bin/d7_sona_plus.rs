//! D7 SONA++ lab: does `sona::SonaPlus` (guarded keyword hand-over + exploration + graded
//! reuse-signal path + optional SONA pattern memory) beat keyword and the best fair baseline?
//!
//! Worlds
//!  * normal-feedback (x1_lab generators, copied read-only): traps(main), correct-keywords,
//!    many-topics(16), sim-random, sim-notraps. Feedback = realised Bernoulli outcome.
//!  * weak-feedback (x3_coupling WeakCfg style, copied read-only): the hook reports only "tool ok"
//!    (P(fail) = .01 + .03(1-s), or agent down); 30% of ok tasks reveal the realised outcome 25 tasks
//!    later (10% flipped). Worlds: weak-traps (sim Fixed), weak-random (sim Random), weak-short-traps
//!    (x1 traps(main) generator + the same observation model).
//!
//! Protocol: tune on seeds 1-8 only (regret-averse objective vs best fair baseline per world x H),
//! held-out 31-60 (confirmation 61-90 with --held 61-90, frozen tuning), horizons 100..2000
//! (prefix mean of EXPECTED success), paired 95% t-CI, Holm over each family.
//! Baselines: keyword, eps-greedy, pooled-ts, hebbian-robust, warm(hebr), guarded(x1 frozen) and
//! guarded(hebr RETUNED on the same grid/objective). Weak worlds: baselines get the raw hook bit;
//! the strict pool additionally contains guarded fed the same reuse signal as SonaPlus.
//!
//! Usage: cargo run --release --offline --bin d7_sona_plus [-- --out results/d7_sona_plus.json] [--held 61-90] [--quick]

use std::collections::{HashMap, VecDeque};

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World as SimWorld, WorldConfig};
use neural_mesh::sona::{PlusExplore, PlusLearner, SonaConfig, SonaPlus, SonaPlusConfig, SonaRouter};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

const HORIZONS: [usize; 5] = [100, 250, 500, 1000, 2000];
const N_TASKS: usize = 2000;

// ------------------------------------------------------------------ worlds (copied from x1_lab / x3)
#[derive(Clone, Debug)]
enum WorldKind {
    Short { topics: usize, traps: usize, map_err: f64, confound: f64 },
    Sim(Scenario),
}

#[derive(Clone, Debug)]
struct WorldSpec {
    name: &'static str,
    kind: WorldKind,
}

struct World {
    agents: usize,
    tasks: Vec<Task>,
    p: Vec<Vec<f64>>,
    u: Vec<f64>,
    kwmap: HashMap<String, usize>,
    topic: Vec<usize>,
    down: Vec<Vec<bool>>,
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
    let (mut tasks, mut pp, mut u, mut topic) = (vec![], vec![], vec![], vec![]);
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
        topic.push(t);
    }
    World { agents: a_n, tasks, p: pp, u, kwmap, topic, down: vec![vec![false; a_n]; N_TASKS] }
}

fn gen_sim(seed: u64, sc: Scenario) -> World {
    let w = SimWorld::new(seed, WorldConfig { n_tasks: N_TASKS, scenario: sc, ..WorldConfig::default() });
    let n = w.n_agents();
    let tasks: Vec<Task> = w.tasks().iter().map(|s| s.task.clone()).collect();
    let p: Vec<Vec<f64>> = (0..w.len()).map(|t| (0..n).map(|a| w.success_prob(t, &[a])).collect()).collect();
    let u: Vec<f64> = w.tasks().iter().map(|s| s.u).collect();
    let topic: Vec<usize> = w.tasks().iter().map(|s| s.type_idx).collect();
    let down: Vec<Vec<bool>> = (0..w.len()).map(|t| (0..n).map(|a| w.is_down(t, a)).collect()).collect();
    World { agents: n, tasks, p, u, kwmap: w.keyword_map_for(), topic, down }
}

fn gen_world(seed: u64, s: &WorldSpec) -> World {
    match &s.kind {
        WorldKind::Short { topics, traps, map_err, confound } => gen_short(seed, *topics, *traps, *map_err, *confound),
        WorldKind::Sim(sc) => gen_sim(seed, *sc),
    }
}

fn normal_worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec { name: "traps(main)", kind: WorldKind::Short { topics: 8, traps: 2, map_err: 0.10, confound: 0.25 } },
        WorldSpec { name: "correct-keywords", kind: WorldKind::Short { topics: 8, traps: 0, map_err: 0.0, confound: 0.25 } },
        WorldSpec { name: "many-topics(16)", kind: WorldKind::Short { topics: 16, traps: 2, map_err: 0.10, confound: 0.25 } },
        WorldSpec { name: "sim-random", kind: WorldKind::Sim(Scenario::Random) },
        WorldSpec { name: "sim-notraps", kind: WorldKind::Sim(Scenario::NoTraps) },
    ]
}

fn weak_worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec { name: "weak-traps(sim)", kind: WorldKind::Sim(Scenario::Fixed) },
        WorldSpec { name: "weak-random(sim)", kind: WorldKind::Sim(Scenario::Random) },
        WorldSpec { name: "weak-short-traps", kind: WorldKind::Short { topics: 8, traps: 2, map_err: 0.10, confound: 0.25 } },
    ]
}

// ------------------------------------------------------------------ weak observation model (x3 WeakCfg)
const TOOL_BASE: f64 = 0.01;
const TOOL_SLOPE: f64 = 0.03;
const REUSE_FRAC: f64 = 0.3;
const REUSE_DELAY: usize = 25;
const REUSE_NOISE: f64 = 0.1;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    /// ordinary feedback: realised outcome
    Normal,
    /// the hook bit only (what ruflo-style baselines see)
    WeakRaw,
    /// raw failures now + delayed graded reuse value; no update on raw successes
    WeakReuse,
}

// ------------------------------------------------------------------ policies
trait Pol {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize;
    fn upd(&mut self, t: &Task, a: usize, r: f64);
    fn weak(&mut self, t: &Task, a: usize, raw_ok: bool) {
        if !raw_ok {
            self.upd(t, a, 0.0)
        }
    }
    fn delayed(&mut self, t: &Task, a: usize, v: f64) {
        self.upd(t, a, v)
    }
}

struct Plain<R: Router>(R);
impl<R: Router> Pol for Plain<R> {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(t, c, rng)
    }
    fn upd(&mut self, t: &Task, a: usize, r: f64) {
        self.0.update(t, a, r)
    }
}

struct PlusPol(SonaPlus);
impl Pol for PlusPol {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(t, c, rng)
    }
    fn upd(&mut self, t: &Task, a: usize, r: f64) {
        self.0.update(t, a, r)
    }
    fn weak(&mut self, t: &Task, a: usize, raw_ok: bool) {
        self.0.update_weak(t, a, raw_ok)
    }
    fn delayed(&mut self, t: &Task, a: usize, v: f64) {
        self.0.update_delayed(t, a, v)
    }
}

fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

fn x1_guard() -> GuardConfig {
    GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps,
    Pts,
    HebR,
    WarmHebR,
    Guard(GuardConfig),
    Plus(SonaPlusConfig),
    Legacy,
}

fn build(k: &Kind, w: &World) -> Box<dyn Pol> {
    let n = w.agents;
    let km = w.kwmap.clone();
    match k {
        Kind::Keyword => Box::new(Plain(KeywordRouter::new(km, n))),
        Kind::Eps => Box::new(Plain(EpsGreedyRouter::new(n, 0.95, 0.02))),
        Kind::Pts => Box::new(Plain(PooledThompsonRouter::new(n, 1.0))),
        Kind::HebR => Box::new(Plain(HebbianRouter::new(n, heb_robust_cfg()))),
        Kind::WarmHebR => Box::new(Plain(Warm::new(HebbianRouter::new(n, heb_robust_cfg()), km, n, 1.0).with_evidence(Evidence::MeanFeature))),
        Kind::Guard(g) => Box::new(Plain(Guarded::new(HebbianRouter::new(n, heb_robust_cfg()), km, n, *g))),
        Kind::Plus(c) => Box::new(PlusPol(SonaPlus::new(n, &km, c.clone()))),
        Kind::Legacy => Box::new(Plain(SonaRouter::new(n, &km, SonaConfig::default()))),
    }
}

fn run(kind: &Kind, w: &World, seed: u64, mode: Mode) -> [f64; 5] {
    let cands: Vec<usize> = (0..w.agents).collect();
    let mut pol = build(kind, w);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut orng = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x3EAC));
    let draws: Vec<(f64, f64, f64)> = (0..w.tasks.len()).map(|_| (orng.f64(), orng.f64(), orng.f64())).collect();
    let mut pending: VecDeque<(usize, usize, usize, f64)> = VecDeque::new();
    let (mut cum, mut out, mut hi) = (0.0, [0.0; 5], 0usize);
    for t in 0..w.tasks.len() {
        while pending.front().is_some_and(|p| p.0 <= t) {
            let (_, ti, a, v) = pending.pop_front().unwrap();
            pol.delayed(&w.tasks[ti], a, v);
        }
        let task = &w.tasks[t];
        let a = pol.pick(task, &cands, &mut rng);
        let p = w.p[t][a];
        let s = if w.u[t] < p { 1.0 } else { 0.0 };
        cum += p;
        match mode {
            Mode::Normal => pol.upd(task, a, s),
            _ => {
                let (u_tool, u_reuse, u_noise) = draws[t];
                let raw_ok = !w.down[t][a] && u_tool >= TOOL_BASE + TOOL_SLOPE * (1.0 - s);
                if mode == Mode::WeakRaw {
                    pol.upd(task, a, if raw_ok { 1.0 } else { 0.0 });
                } else {
                    pol.weak(task, a, raw_ok);
                    if raw_ok && u_reuse < REUSE_FRAC {
                        let v = if u_noise < REUSE_NOISE { 1.0 - s } else { s };
                        pending.push_back((t + REUSE_DELAY, t, a, v));
                    }
                }
            }
        }
        if hi < 5 && t + 1 == HORIZONS[hi] {
            out[hi] = cum / (t + 1) as f64;
            hi += 1;
        }
    }
    out
}

fn eval(kind: &Kind, mode: Mode, seeds: &[u64], s: &WorldSpec) -> Vec<[f64; 5]> {
    seeds.par_iter().map(|&sd| run(kind, &gen_world(sd, s), sd, mode)).collect()
}

fn col(r: &[[f64; 5]], h: usize) -> Vec<f64> {
    r.iter().map(|x| x[h]).collect()
}

fn means(kind: &Kind, mode: Mode, seeds: &[u64], ws: &[WorldSpec]) -> Vec<[f64; 5]> {
    ws.iter()
        .map(|s| {
            let r = eval(kind, mode, seeds, s);
            let mut m = [0.0; 5];
            for h in 0..5 {
                m[h] = r.iter().map(|x| x[h]).sum::<f64>() / r.len() as f64;
            }
            m
        })
        .collect()
}

/// Regret-averse objective on tune seeds vs best baseline per (world, horizon): mean(d) + mean(min(d,0)).
fn obj(kind: &Kind, mode: Mode, seeds: &[u64], ws: &[WorldSpec], best_b: &[[f64; 5]]) -> f64 {
    let m = means(kind, mode, seeds, ws);
    let mut d = vec![];
    for (w, row) in m.iter().enumerate() {
        for h in 0..5 {
            d.push(row[h] - best_b[w][h]);
        }
    }
    let n = d.len() as f64;
    d.iter().sum::<f64>() / n + d.iter().map(|x| x.min(0.0)).sum::<f64>() / n
}

fn tune(label: &str, cands: Vec<SonaPlusConfig>, mode: Mode, seeds: &[u64], ws: &[WorldSpec], best_b: &[[f64; 5]]) -> (f64, SonaPlusConfig) {
    let n = cands.len();
    let mut sc: Vec<(f64, SonaPlusConfig)> = cands.into_iter().map(|c| (obj(&Kind::Plus(c.clone()), mode, seeds, ws, best_b), c)).collect();
    sc.sort_by(|a, b| b.0.total_cmp(&a.0));
    println!("[tune 1-8] {label}: {n} configs, best obj {:+.5}
    {:?}", sc[0].0, sc[0].1);
    for (o, c) in sc.iter().skip(1).take(3) {
        println!("      runner-up {:+.5}: {:?} tau {} disc {} xkw {} {:?} pat(init {},lr {},T {},eps {},sw {})", o, c.learner, c.tau, c.discount, c.explore_kw, c.explore_mode, c.pat_init, c.pat_lr, c.pat_temp, c.pat_eps, c.pat_set_weight);
    }
    sc.swap_remove(0)
}

fn best_of(baselines: &[(String, Kind, Mode)], seeds: &[u64], ws: &[WorldSpec]) -> Vec<[f64; 5]> {
    let bm: Vec<Vec<[f64; 5]>> = baselines.iter().map(|(_, k, m)| means(k, *m, seeds, ws)).collect();
    (0..ws.len())
        .map(|w| {
            let mut m = [f64::NEG_INFINITY; 5];
            for b in &bm {
                for h in 0..5 {
                    m[h] = m[h].max(b[w][h]);
                }
            }
            m
        })
        .collect()
}

// ------------------------------------------------------------------ grids
fn guard_grid(quick: bool) -> Vec<SonaPlusConfig> {
    let mut v = vec![];
    for &tau in &[0.45, 0.55, 0.65] {
        for &z in &[0.5, 1.0] {
            for &(ka, kb) in &[(2.0, 1.0), (4.0, 1.0)] {
                for &disc in &[1.0, 0.97] {
                    for &ch in &[true, false] {
                        for &(he, ht) in &[(0.0, 0.002), (0.02, 0.02)] {
                            v.push(SonaPlusConfig { tau, z, kw_a: ka, kw_b: kb, discount: disc, challenge: ch, heb_epsilon: he, heb_temperature: ht, ..SonaPlusConfig::default() });
                        }
                    }
                }
            }
        }
    }
    if quick {
        v.truncate(6);
    }
    v
}

/// stage B: exploration + learner choice. Hebbian: around the retuned guard. Patterns: joint
/// grid over tau/discount (the pattern learner has its own optimum) x learner params x exploration.
fn plus_grid(base: &SonaPlusConfig, quick: bool) -> Vec<SonaPlusConfig> {
    let mut v = vec![];
    for &x in &[0.0, 0.02, 0.05, 0.1] {
        for &mode in &[PlusExplore::Uniform, PlusExplore::Ucb] {
            for &dk in &[0.0, 50.0] {
                if x == 0.0 && (mode == PlusExplore::Ucb || dk > 0.0) {
                    continue;
                }
                v.push(SonaPlusConfig { explore_kw: x, explore_mode: mode, explore_decay_k: dk, learner: PlusLearner::Hebbian, ..base.clone() });
            }
        }
    }
    for &tau in &[0.45, 0.55, 0.65] {
        for &disc in &[1.0, 0.97] {
            for &init in &[0.6, 0.8] {
                for &lr in &[0.1, 0.25] {
                    for &temp in &[0.03, 0.1] {
                        for &eps in &[0.0, 0.02] {
                            for &(x, dk) in &[(0.0, 0.0), (0.03, 50.0)] {
                                for &sw in &[0.0, 0.5] {
                                    v.push(SonaPlusConfig {
                                        tau,
                                        discount: disc,
                                        explore_kw: x,
                                        explore_mode: PlusExplore::Ucb,
                                        explore_decay_k: dk,
                                        learner: PlusLearner::Patterns,
                                        pat_init: init,
                                        pat_lr: lr,
                                        pat_temp: temp,
                                        pat_eps: eps,
                                        pat_set_weight: sw,
                                        ..base.clone()
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if quick {
        v.truncate(10);
    }
    v
}

fn weak_guard_grid(quick: bool) -> Vec<SonaPlusConfig> {
    let mut v = vec![];
    for &tau in &[0.45, 0.55, 0.65] {
        for &z in &[0.5, 1.0] {
            for &(ka, kb) in &[(2.0, 1.0), (4.0, 1.0)] {
                for &disc in &[1.0, 0.97] {
                    for &m in &[1usize, 2, 3] {
                        v.push(SonaPlusConfig { tau, z, kw_a: ka, kw_b: kb, discount: disc, delayed_mult: m, ..SonaPlusConfig::default() });
                    }
                }
            }
        }
    }
    if quick {
        v.truncate(6);
    }
    v
}

fn weak_plus_grid(base: &SonaPlusConfig, quick: bool) -> Vec<SonaPlusConfig> {
    let mut v = vec![];
    for g in plus_grid(base, quick) {
        for &fr in &[0.0, 0.3] {
            v.push(SonaPlusConfig { fail_reward: fr, ..g.clone() });
        }
    }
    v
}

// ------------------------------------------------------------------ comparison family
struct Arm {
    name: String,
    kind: Kind,
    mode: Mode,
}

struct Cell {
    world: String,
    cand: String,
    h: usize,
    diff: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    p: f64,
    best: String,
    cand_mean: f64,
    base_mean: f64,
}

/// Evaluate every arm once per world; compare each candidate with the best (by held-out mean) baseline in `pool`.
fn family(
    ws: &[WorldSpec],
    bases: &[Arm],
    cands: &[Arm],
    pools: &[(&str, Vec<usize>)],
    held: &[u64],
    tables: &mut Vec<Value>,
) -> Vec<(String, Vec<Cell>)> {
    let mut out: Vec<(String, Vec<Cell>)> = pools.iter().map(|(n, _)| (n.to_string(), vec![])).collect();
    for s in ws {
        let br: Vec<Vec<[f64; 5]>> = bases.iter().map(|a| eval(&a.kind, a.mode, held, s)).collect();
        let cr: Vec<Vec<[f64; 5]>> = cands.iter().map(|a| eval(&a.kind, a.mode, held, s)).collect();
        println!("\n=== world {} | held mean expected success over first H ===", s.name);
        println!("{:<44}{:>8}{:>8}{:>8}{:>8}{:>8}", "arm", 100, 250, 500, 1000, 2000);
        let mut rows = vec![];
        for (a, r) in bases.iter().zip(&br).map(|(a, r)| (a, r)).chain(cands.iter().zip(&cr)) {
            let m: Vec<f64> = (0..5).map(|h| summarize(&col(r, h)).mean).collect();
            println!("{:<44}{}", a.name, m.iter().map(|x| format!("{:>8.4}", x)).collect::<String>());
            rows.push(json!({"arm": a.name, "means": m}));
        }
        tables.push(json!({"world": s.name, "rows": rows}));
        for (pi, (_, pool)) in pools.iter().enumerate() {
            for h in 0..5 {
                let bi = *pool
                    .iter()
                    .max_by(|&&a, &&b| summarize(&col(&br[a], h)).mean.partial_cmp(&summarize(&col(&br[b], h)).mean).unwrap())
                    .unwrap();
                let b = col(&br[bi], h);
                for (ci, ca) in cands.iter().enumerate() {
                    let c = col(&cr[ci], h);
                    let d = paired_diff(&c, &b);
                    out[pi].1.push(Cell {
                        world: s.name.to_string(),
                        cand: ca.name.clone(),
                        h: HORIZONS[h],
                        diff: d.diff.mean,
                        lo: d.diff.lo,
                        hi: d.diff.hi,
                        wins: d.wins,
                        p: paired_t_pvalue(&c, &b),
                        best: bases[bi].name.clone(),
                        cand_mean: summarize(&c).mean,
                        base_mean: summarize(&b).mean,
                    });
                }
            }
        }
    }
    out
}

fn report(title: &str, cells: &[Cell], only_cand: Option<&str>) -> Value {
    let adj = holm_adjust(&cells.iter().map(|c| c.p).collect::<Vec<_>>());
    println!("\n### {title}  (Holm over {} tests)", cells.len());
    println!("{:<18} {:<30} {:>5} {:>9} {:>20} {:>5} {:>8}  best baseline", "world", "candidate", "H", "diff", "95% CI", "wins", "Holm p");
    let mut out = vec![];
    for (c, a) in cells.iter().zip(&adj) {
        let mark = if *a < 0.05 { if c.diff > 0.0 { " WIN" } else { " LOSS" } } else { "" };
        if only_cand.is_none_or(|n| c.cand.starts_with(n)) {
            println!(
                "{:<18} {:<30} {:>5} {:>+9.4} [{:>+8.4},{:>+8.4}] {:>5} {:>8.4}  {}{}",
                c.world, c.cand, c.h, c.diff, c.lo, c.hi, c.wins, a, c.best, mark
            );
        }
        out.push(json!({"world": c.world, "candidate": c.cand, "H": c.h, "diff": c.diff, "ci": [c.lo, c.hi], "wins": c.wins,
            "p": c.p, "holm_p": a, "best_baseline": c.best, "cand_mean": c.cand_mean, "base_mean": c.base_mean}));
    }
    let (w, l) = cells.iter().zip(&adj).fold((0, 0), |(w, l), (c, a)| if *a < 0.05 { if c.diff > 0.0 { (w + 1, l) } else { (w, l + 1) } } else { (w, l) });
    println!("--> Holm wins {w}, losses {l} of {}", cells.len());
    json!({"title": title, "holm_wins": w, "holm_losses": l, "cells": out})
}

/// Direct paired comparison a - b on the same seeds per world x H, Holm over the set.
fn direct(title: &str, ws: &[WorldSpec], a: &Arm, b: &Arm, held: &[u64]) -> Value {
    let mut rows = vec![];
    let mut ps = vec![];
    for s in ws {
        let ra = eval(&a.kind, a.mode, held, s);
        let rb = eval(&b.kind, b.mode, held, s);
        for h in 0..5 {
            let (ca, cb) = (col(&ra, h), col(&rb, h));
            let d = paired_diff(&ca, &cb);
            ps.push(paired_t_pvalue(&ca, &cb));
            rows.push((s.name, h, d.diff.mean, d.diff.lo, d.diff.hi, d.wins));
        }
    }
    let adj = holm_adjust(&ps);
    println!("\n--- direct: {title} (Holm over {}) ---", rows.len());
    let mut out = vec![];
    for (r, a) in rows.iter().zip(&adj) {
        let mark = if *a < 0.05 { if r.2 > 0.0 { " WIN" } else { " LOSS" } } else { "" };
        println!("{:<18} H{:<5} {:>+8.4} [{:>+8.4},{:>+8.4}] w{:>2} Holm {:.4}{}", r.0, HORIZONS[r.1], r.2, r.3, r.4, r.5, a, mark);
        out.push(json!({"world": r.0, "H": HORIZONS[r.1], "diff": r.2, "ci": [r.3, r.4], "wins": r.5, "holm_p": a}));
    }
    json!({"title": title, "rows": out})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/d7_sona_plus.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let (h0, h1) = args
        .iter()
        .position(|a| a == "--held")
        .and_then(|i| args.get(i + 1))
        .and_then(|r| r.split_once('-'))
        .map(|(a, b)| (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap()))
        .unwrap_or((31, 60));
    let held: Vec<u64> = (h0..=h1).collect();
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let wn = normal_worlds();
    let ww = weak_worlds();
    println!("d7_sona_plus: {N_TASKS} tasks/run, tune seeds 1-8, held-out {h0}-{h1}{}", if quick { " (QUICK)" } else { "" });

    let base6 = |mode: Mode| -> Vec<(String, Kind, Mode)> {
        vec![
            ("keyword".into(), Kind::Keyword, mode),
            ("eps-greedy".into(), Kind::Eps, mode),
            ("pooled-ts".into(), Kind::Pts, mode),
            ("hebbian-robust".into(), Kind::HebR, mode),
            ("warm(hebr)".into(), Kind::WarmHebR, mode),
            ("guarded(x1)".into(), Kind::Guard(x1_guard()), mode),
        ]
    };

    // ------------------------------------------------ tuning: normal-feedback worlds
    let best_b_n = best_of(&base6(Mode::Normal), &tune_seeds, &wn);
    let (obj_rt, guard_rt) = tune("guarded-rt (retuned guarded(hebr), no exploration)", guard_grid(quick), Mode::Normal, &tune_seeds, &wn, &best_b_n);
    let (obj_full, full_n) = tune("sona+ (explore + learner)", plus_grid(&guard_rt, quick), Mode::Normal, &tune_seeds, &wn, &best_b_n);
    let heb_only: Vec<SonaPlusConfig> = plus_grid(&guard_rt, quick).into_iter().filter(|c| c.learner == PlusLearner::Hebbian).collect();
    let pat_only: Vec<SonaPlusConfig> = plus_grid(&guard_rt, quick).into_iter().filter(|c| c.learner == PlusLearner::Patterns).collect();
    let (obj_heb, heb_n) = tune("sona+ ablation: hebbian learner only (no pattern memory)", heb_only, Mode::Normal, &tune_seeds, &wn, &best_b_n);
    let (obj_pat, pat_n) = tune("sona+ ablation: pattern learner only", pat_only, Mode::Normal, &tune_seeds, &wn, &best_b_n);
    let noexp_n = SonaPlusConfig { explore_kw: 0.0, ..full_n.clone() };
    let obj_noexp = obj(&Kind::Plus(noexp_n.clone()), Mode::Normal, &tune_seeds, &wn, &best_b_n);
    println!("[tune 1-8] obj: guarded-rt {obj_rt:+.5} | sona+ {obj_full:+.5} | heb-only {obj_heb:+.5} | pat-only {obj_pat:+.5} | sona+ w/o kw-exploration {obj_noexp:+.5}");

    // ------------------------------------------------ tuning: weak-feedback worlds (Reuse signal)
    let best_b_w = best_of(&base6(Mode::WeakRaw), &tune_seeds, &ww);
    let (wobj_rt, wguard_rt) = tune("weak: guarded-rt + reuse", weak_guard_grid(quick), Mode::WeakReuse, &tune_seeds, &ww, &best_b_w);
    let (wobj_full, full_w) = tune("weak: sona+ + reuse", weak_plus_grid(&wguard_rt, quick), Mode::WeakReuse, &tune_seeds, &ww, &best_b_w);
    let wheb: Vec<SonaPlusConfig> = weak_plus_grid(&wguard_rt, quick).into_iter().filter(|c| c.learner == PlusLearner::Hebbian).collect();
    let wpat: Vec<SonaPlusConfig> = weak_plus_grid(&wguard_rt, quick).into_iter().filter(|c| c.learner == PlusLearner::Patterns).collect();
    let (_, heb_w) = tune("weak: sona+ ablation hebbian-only", wheb, Mode::WeakReuse, &tune_seeds, &ww, &best_b_w);
    let (_, pat_w) = tune("weak: sona+ ablation pattern-only", wpat, Mode::WeakReuse, &tune_seeds, &ww, &best_b_w);
    let noexp_w = SonaPlusConfig { explore_kw: 0.0, ..full_w.clone() };
    println!("[tune 1-8] weak obj: guarded-rt+reuse {wobj_rt:+.5} | sona+ {wobj_full:+.5}");

    // ------------------------------------------------ held-out: normal worlds
    let arm = |n: &str, k: Kind, m: Mode| Arm { name: n.to_string(), kind: k, mode: m };
    let mut bases_n: Vec<Arm> = base6(Mode::Normal).into_iter().map(|(n, k, m)| Arm { name: n, kind: k, mode: m }).collect();
    bases_n.push(arm("guarded-rt(hebr retuned)", Kind::Plus(guard_rt.clone()), Mode::Normal));
    let cands_n = vec![
        arm("sona+", Kind::Plus(full_n.clone()), Mode::Normal),
        arm("sona+ [hebbian only]", Kind::Plus(heb_n.clone()), Mode::Normal),
        arm("sona+ [pattern only]", Kind::Plus(pat_n.clone()), Mode::Normal),
        arm("sona+ [no kw-explore]", Kind::Plus(noexp_n.clone()), Mode::Normal),
        arm("sona(legacy, ruflo port)", Kind::Legacy, Mode::Normal),
    ];
    let mut tables = vec![];
    let pool_all: Vec<usize> = (0..bases_n.len()).collect();
    let pool_six: Vec<usize> = (0..6).collect();
    let fam_n = family(&wn, &bases_n, &cands_n, &[("best-of-7 (incl. retuned guarded)", pool_all.clone()), ("best-of-6 (spec set)", pool_six.clone()), ("keyword only", vec![0])], &held, &mut tables);
    let rep_n: Vec<Value> = fam_n
        .iter()
        .map(|(t, c)| report(&format!("NORMAL worlds: candidates vs {t}"), c, Some("sona+")))
        .collect();
    let dir_n = vec![
        direct("sona+ - guarded-rt (normal worlds)", &wn, &cands_n[0], &bases_n[6], &held),
        direct("sona+ - guarded(x1) (normal worlds)", &wn, &cands_n[0], &bases_n[5], &held),
    ];

    // ------------------------------------------------ held-out: weak worlds
    let mut bases_w: Vec<Arm> = base6(Mode::WeakRaw).into_iter().map(|(n, k, m)| Arm { name: format!("{n} [raw]"), kind: k, mode: m }).collect();
    bases_w.push(arm("guarded(x1) [reuse]", Kind::Guard(x1_guard()), Mode::WeakReuse));
    bases_w.push(arm("guarded-rt [reuse]", Kind::Plus(wguard_rt.clone()), Mode::WeakReuse));
    let cands_w = vec![
        arm("sona+ [reuse]", Kind::Plus(full_w.clone()), Mode::WeakReuse),
        arm("sona+ [hebbian only,reuse]", Kind::Plus(heb_w.clone()), Mode::WeakReuse),
        arm("sona+ [pattern only,reuse]", Kind::Plus(pat_w.clone()), Mode::WeakReuse),
        arm("sona+ [no kw-explore,reuse]", Kind::Plus(noexp_w.clone()), Mode::WeakReuse),
        arm("sona+ [normal-tuned cfg,reuse]", Kind::Plus(full_n.clone()), Mode::WeakReuse),
        arm("sona+ [weak cfg on raw bit]", Kind::Plus(full_w.clone()), Mode::WeakRaw),
        arm("sona(legacy) [raw]", Kind::Legacy, Mode::WeakRaw),
    ];
    let fam_w = family(
        &ww,
        &bases_w,
        &cands_w,
        &[("strict pool: raw baselines + guarded fed the reuse signal", (0..8).collect()), ("raw-fed baselines only", (0..6).collect()), ("keyword only", vec![0])],
        &held,
        &mut tables,
    );
    let rep_w: Vec<Value> = fam_w
        .iter()
        .map(|(t, c)| report(&format!("WEAK worlds: candidates vs {t}"), c, Some("sona+ [reuse]")))
        .collect();
    let dir_w = vec![
        direct("sona+ [reuse] - guarded-rt [reuse] (weak worlds)", &ww, &cands_w[0], &bases_w[7], &held),
        direct("sona+ [reuse] - guarded(x1) [reuse] (weak worlds)", &ww, &cands_w[0], &bases_w[6], &held),
    ];

    let res = json!({
        "protocol": {"tune_seeds": "1-8", "heldout": format!("{h0}-{h1}"), "horizons": HORIZONS, "n_tasks": N_TASKS,
            "metric": "prefix mean expected success of routed agent", "weak_model": {"tool_base": TOOL_BASE, "tool_slope": TOOL_SLOPE, "reuse_frac": REUSE_FRAC, "reuse_delay": REUSE_DELAY, "reuse_noise": REUSE_NOISE}},
        "tuned": {
            "guarded_rt": format!("{guard_rt:?}"), "sona_plus_normal": format!("{full_n:?}"), "heb_only": format!("{heb_n:?}"), "pat_only": format!("{pat_n:?}"),
            "weak_guarded_rt": format!("{wguard_rt:?}"), "sona_plus_weak": format!("{full_w:?}"), "weak_heb_only": format!("{heb_w:?}"), "weak_pat_only": format!("{pat_w:?}"),
            "obj": {"guarded_rt": obj_rt, "sona_plus": obj_full, "heb_only": obj_heb, "pat_only": obj_pat, "no_kw_explore": obj_noexp, "weak_guarded_rt": wobj_rt, "weak_sona_plus": wobj_full}
        },
        "tables": tables, "normal": rep_n, "normal_direct": dir_n, "weak": rep_w, "weak_direct": dir_w,
    });
    let _ = std::fs::create_dir_all("results");
    std::fs::write(&out_path, serde_json::to_string_pretty(&res).unwrap()).unwrap();
    println!("\nwrote {out_path}");
}
