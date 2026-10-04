//! D1 (MESH-1): continuous self-improvement loop = mesh health/benching <-> router uncertainty.
//!
//! Mesh -> router: NeuralMesh health benching (+ attribution: do not teach the router from
//! outcomes of an agent the mesh believes is down). Router -> mesh: the router's own uncertainty
//! (its Beta-bound width on the chosen agent) or the leader's last failure triggers a cost-accounted
//! liveness PROBE of the chosen leader before use (reroute if dead); benched agents may be probed
//! for recovery instead of risking a real task on probation.
//! Worlds: x1_lab generators (copied) + random outage windows overlaid (p=0 while down).
//! Metric: NET = mean expected success over first H tasks - c * probes / H  (c = probe cost).
//! Protocol: tune seeds 1-8 (regret-averse vs best fair baseline), held-out 31-60 (or --held a-b),
//! paired 95% t-CI vs best fair baseline per (world,H), Holm over all candidate x world x H tests.
//! Usage: cargo run --release --bin d1_probe [-- --held 61-90 --out results/d1_probe.json --quick]

use std::collections::HashMap;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::mesh::{MeshConfig, NeuralMesh};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World as SimWorld, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::json;

const HORIZONS: [usize; 2] = [500, 2000];
const N_TASKS: usize = 2000;
const PROBE_COST: f64 = 0.05;
const PROBE_ACC: f64 = 0.95;

#[derive(Clone, Debug)]
enum WKind {
    Short { topics: usize, traps: usize, map_err: f64, confound: f64 },
    Sim(Scenario),
}
#[derive(Clone, Debug)]
struct WSpec {
    name: &'static str,
    kind: WKind,
    n_out: usize,
    dur: (usize, usize),
}

struct World {
    agents: usize,
    tasks: Vec<Task>,
    p: Vec<Vec<f64>>,
    u: Vec<f64>,
    kwmap: HashMap<String, usize>,
    outs: Vec<(usize, usize, usize)>, // agent, start, end
}
impl World {
    fn down(&self, t: usize, a: usize) -> bool {
        self.outs.iter().any(|&(ag, s, e)| ag == a && t >= s && t < e)
    }
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
            p[t][a] = if a == best[t] { 0.85 } else if a == second[t] { 0.5 } else { 0.15 + 0.25 * rng.f64() };
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
    World { agents: a_n, tasks, p: pp, u, kwmap, outs: vec![] }
}

fn gen_sim(seed: u64, sc: Scenario) -> World {
    let w = SimWorld::new(seed, WorldConfig { n_tasks: N_TASKS, scenario: sc, ..WorldConfig::default() });
    let n = w.n_agents();
    let tasks: Vec<Task> = w.tasks().iter().map(|s| s.task.clone()).collect();
    // base skill WITHOUT sim's own failure window; outages are added below
    let p: Vec<Vec<f64>> = (0..w.len())
        .map(|t| (0..n).map(|a| if w.is_down(t, a) { w.skill(t, w.task(t).type_idx, a) } else { w.success_prob(t, &[a]) }).collect())
        .collect();
    let u: Vec<f64> = w.tasks().iter().map(|s| s.u).collect();
    World { agents: n, tasks, p, u, kwmap: w.keyword_map_for(), outs: vec![] }
}

fn gen_world(seed: u64, s: &WSpec) -> World {
    let mut w = match &s.kind {
        WKind::Short { topics, traps, map_err, confound } => gen_short(seed, *topics, *traps, *map_err, *confound),
        WKind::Sim(sc) => gen_sim(seed, *sc),
    };
    let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(777));
    for _ in 0..s.n_out {
        let dur = s.dur.0 + rng.below(s.dur.1 - s.dur.0 + 1);
        let start = rng.below(N_TASKS - dur);
        let agent = if rng.bernoulli(0.7) {
            // hit the best agent of a random task: outages hurt where it matters
            let t = rng.below(N_TASKS);
            (0..w.agents).max_by(|&a, &b| w.p[t][a].partial_cmp(&w.p[t][b]).unwrap()).unwrap()
        } else {
            rng.below(w.agents)
        };
        w.outs.push((agent, start, start + dur));
    }
    w
}

fn worlds() -> Vec<WSpec> {
    vec![
        WSpec { name: "traps+outages", kind: WKind::Short { topics: 8, traps: 2, map_err: 0.10, confound: 0.25 }, n_out: 4, dur: (40, 160) },
        WSpec { name: "correct-kw+outages", kind: WKind::Short { topics: 8, traps: 0, map_err: 0.0, confound: 0.25 }, n_out: 4, dur: (40, 160) },
        WSpec { name: "sim-random+outages", kind: WKind::Sim(Scenario::Random), n_out: 4, dur: (40, 160) },
        WSpec { name: "many-topics16+short-outages", kind: WKind::Short { topics: 16, traps: 2, map_err: 0.10, confound: 0.25 }, n_out: 8, dur: (20, 60) },
        WSpec { name: "traps+HEAVY-outages", kind: WKind::Short { topics: 8, traps: 2, map_err: 0.10, confound: 0.25 }, n_out: 10, dur: (60, 200) },
        WSpec { name: "traps-NO-outages(control)", kind: WKind::Short { topics: 8, traps: 2, map_err: 0.10, confound: 0.25 }, n_out: 0, dur: (40, 160) },
    ]
}

// ------------------------------------------------------------------ policies
#[derive(Clone, Debug, PartialEq)]
enum Trig {
    None,
    LastFail,
    Streak(u32),
    Width(f64),
    Random(f64),
    Always,
}

#[derive(Clone, Debug)]
struct Pol {
    mesh: bool,
    thr: f64,
    alpha: f64,
    trig: Trig,
    /// probe benched agents every m tasks (and disable task-risking probation)
    rec: Option<u64>,
    /// skip router update when leader health < this
    attr: Option<f64>,
}
impl Pol {
    fn plain() -> Self {
        Pol { mesh: false, thr: 0.0, alpha: 0.2, trig: Trig::None, rec: None, attr: None }
    }
    fn bench() -> Self {
        Pol { mesh: true, thr: 0.3, alpha: 0.2, trig: Trig::None, rec: None, attr: None }
    }
}

#[derive(Clone, Debug)]
enum Base {
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm,
    Guard(GuardConfig),
}

fn heb_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}
fn x1_guard() -> GuardConfig {
    GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
}

/// [mean p at H500, mean p at H2000, probes at H500, probes at H2000]
fn drive<R: Router>(r: R, w: &World, seed: u64, pol: &Pol, unc: &dyn Fn(&R, &Task, usize) -> f64) -> [f64; 4] {
    let n = w.agents;
    let cfg = MeshConfig {
        health_alpha: pol.alpha,
        health_threshold: if pol.mesh { pol.thr } else { 0.0 },
        retry_every: if pol.rec.is_some() { 1_000_000_000 } else { 20 },
        team_size: 1,
        ..MeshConfig::default()
    };
    let mut mesh = NeuralMesh::new(n, r, cfg);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut prng = Rng::new(seed.wrapping_mul(31).wrapping_add(5)); // probe rng (never shared with routing)
    let all: Vec<usize> = (0..n).collect();
    let (mut cum, mut probes) = (0.0, 0u64);
    let mut last_fail = vec![false; n];
    let mut streak = vec![0u32; n];
    let mut out = [0.0; 4];
    for (i, task) in w.tasks.iter().enumerate() {
        let do_probe = |mesh: &mut NeuralMesh<R>, prng: &mut Rng, a: usize, probes: &mut u64| -> bool {
            *probes += 1;
            let truth = !w.down(i, a);
            let alive = if prng.bernoulli(PROBE_ACC) { truth } else { !truth };
            mesh.observe_probe(a, alive);
            alive
        };
        if let Some(m) = pol.rec {
            if i as u64 % m == 0 {
                for a in 0..n {
                    if mesh.is_benched(a) {
                        do_probe(&mut mesh, &mut prng, a, &mut probes);
                    }
                }
            }
        }
        let mut cands = all.clone();
        let mut a = mesh.route(task, &cands, &mut rng)[0];
        for _ in 0..3 {
            let fire = match &pol.trig {
                Trig::None => false,
                Trig::LastFail => last_fail[a],
                Trig::Streak(k) => streak[a] >= *k,
                Trig::Width(wd) => unc(mesh.leader(), task, a) > *wd,
                Trig::Random(q) => do_rand(&mut prng, *q),
                Trig::Always => true,
            };
            if !fire {
                break;
            }
            let alive = do_probe(&mut mesh, &mut prng, a, &mut probes);
            if alive {
                streak[a] = 0;
            }
            if alive || cands.len() <= 1 {
                break;
            }
            cands.retain(|&c| c != a);
            a = mesh.route(task, &cands, &mut rng)[0];
        }
        let p = if w.down(i, a) { 0.0 } else { w.p[i][a] };
        cum += p;
        let reward = if w.u[i] < p { 1.0 } else { 0.0 };
        last_fail[a] = reward < 0.5;
        streak[a] = if last_fail[a] { streak[a] + 1 } else { 0 };
        let upd = !pol.attr.is_some_and(|th| mesh.health(a) < th);
        mesh.feedback_opt(task, &[a], reward, upd);
        for (hi, &h) in HORIZONS.iter().enumerate() {
            if i + 1 == h {
                out[hi] = cum / h as f64;
                out[2 + hi] = probes as f64;
            }
        }
    }
    out
}

fn do_rand(r: &mut Rng, q: f64) -> bool {
    r.bernoulli(q)
}

fn run(base: &Base, pol: &Pol, w: &World, seed: u64) -> [f64; 4] {
    let a = w.agents;
    let km = || w.kwmap.clone();
    match base {
        Base::Keyword => drive(KeywordRouter::new(km(), a), w, seed, pol, &|_, _, _| 0.0),
        Base::Eps => drive(EpsGreedyRouter::new(a, 0.95, 0.02), w, seed, pol, &|_, _, _| 0.0),
        Base::Pts => drive(PooledThompsonRouter::new(a, 1.0), w, seed, pol, &|_, _, _| 0.0),
        Base::HebR => drive(HebbianRouter::new(a, heb_cfg()), w, seed, pol, &|_, _, _| 0.0),
        Base::Warm => {
            let r = Warm::new(HebbianRouter::new(a, heb_cfg()), km(), a, 1.0).with_evidence(Evidence::MeanFeature);
            drive(r, w, seed, pol, &|_, _, _| 0.0)
        }
        Base::Guard(g) => {
            let r = Guarded::new(HebbianRouter::new(a, heb_cfg()), km(), a, *g);
            // router uncertainty = width of the pooled Beta bounds of the chosen agent
            drive(r, w, seed, pol, &|r: &Guarded<HebbianRouter>, t: &Task, ag: usize| {
                let (lo, hi) = r.keyword_bounds(t, ag);
                hi - lo
            })
        }
    }
}

fn net(o: &[f64; 4], hi: usize, c: f64) -> f64 {
    o[hi] - c * o[2 + hi] / HORIZONS[hi] as f64
}

fn eval(base: &Base, pol: &Pol, seeds: &[u64], s: &WSpec) -> Vec<[f64; 4]> {
    seeds.par_iter().map(|&sd| run(base, pol, &gen_world(sd, s), sd)).collect()
}

fn mean_net(r: &[[f64; 4]], hi: usize, c: f64) -> f64 {
    r.iter().map(|o| net(o, hi, c)).sum::<f64>() / r.len() as f64
}

#[derive(Clone, Debug)]
struct Cand {
    name: String,
    base: Base,
    pol: Pol,
}

fn obj(c: &Cand, seeds: &[u64], ws: &[WSpec], best_b: &[[f64; 2]]) -> f64 {
    let mut d = vec![];
    for (wi, s) in ws.iter().enumerate() {
        let r = eval(&c.base, &c.pol, seeds, s);
        for h in 0..2 {
            d.push(mean_net(&r, h, PROBE_COST) - best_b[wi][h]);
        }
    }
    let n = d.len() as f64;
    d.iter().sum::<f64>() / n + d.iter().map(|x| x.min(0.0)).sum::<f64>() / n
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/d1_probe.json".into());
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
    let ws = worlds();
    let g = x1_guard();
    println!("d1_probe: {N_TASKS} tasks, probe cost {PROBE_COST}, acc {PROBE_ACC}; tune seeds 1-8, held-out {h0}-{h1}");

    let baselines: Vec<Cand> = vec![
        Cand { name: "keyword".into(), base: Base::Keyword, pol: Pol::plain() },
        Cand { name: "eps-greedy".into(), base: Base::Eps, pol: Pol::plain() },
        Cand { name: "pooled-ts".into(), base: Base::Pts, pol: Pol::plain() },
        Cand { name: "hebbian-robust".into(), base: Base::HebR, pol: Pol::plain() },
        Cand { name: "warm(hebr)".into(), base: Base::Warm, pol: Pol::plain() },
        Cand { name: "guarded(hebr)".into(), base: Base::Guard(g), pol: Pol::plain() },
        Cand { name: "mesh-bench(keyword)".into(), base: Base::Keyword, pol: Pol::bench() },
        Cand { name: "mesh-bench(warm)".into(), base: Base::Warm, pol: Pol::bench() },
        Cand { name: "mesh-bench(guarded)".into(), base: Base::Guard(g), pol: Pol::bench() },
    ];
    let bm: Vec<Vec<[f64; 2]>> = baselines
        .iter()
        .map(|c| {
            ws.iter()
                .map(|s| {
                    let r = eval(&c.base, &c.pol, &tune_seeds, s);
                    [mean_net(&r, 0, PROBE_COST), mean_net(&r, 1, PROBE_COST)]
                })
                .collect()
        })
        .collect();
    let best_b: Vec<[f64; 2]> =
        (0..ws.len()).map(|w| [0, 1].map(|h| bm.iter().map(|b| b[w][h]).fold(f64::NEG_INFINITY, f64::max))).collect();

    // ---- tuning families (seeds 1-8 only)
    let mk = |trig: Trig, rec: Option<u64>, attr: Option<f64>, thr: f64, alpha: f64| Pol { mesh: true, thr, alpha, trig, rec, attr };
    let mut fams: Vec<(&str, Vec<Trig>)> = vec![
        ("coupled-lastfail", vec![Trig::LastFail]),
        ("coupled-streak", vec![Trig::Streak(2), Trig::Streak(3)]),
        ("coupled-width", vec![Trig::Width(0.2), Trig::Width(0.3), Trig::Width(0.45)]),
        ("probe-rec-only", vec![Trig::None]),
        ("coupled-always", vec![Trig::Always]),
    ];
    if quick {
        fams.truncate(2);
    }
    let mut tuned: Vec<Cand> = vec![];
    for (fname, trigs) in &fams {
        let mut grid = vec![];
        for t in trigs {
            for rec in [None, Some(5u64), Some(10), Some(25)] {
                if *fname == "probe-rec-only" && rec.is_none() {
                    continue;
                }
                for attr in [None, Some(0.3), Some(0.5)] {
                    for thr in [0.3, 0.5] {
                        for alpha in [0.2, 0.4] {
                            grid.push(Cand { name: fname.to_string(), base: Base::Guard(g), pol: mk(t.clone(), rec, attr, thr, alpha) });
                        }
                    }
                }
            }
        }
        if quick {
            grid.truncate(6);
        }
        let n = grid.len();
        let scored: Vec<(f64, Cand)> = grid.into_iter().map(|c| (obj(&c, &tune_seeds, &ws, &best_b), c)).collect();
        let best = scored.into_iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
        println!("[tune 1-8] {fname}: {n} configs best obj {:+.5} -> {:?}", best.0, best.1.pol);
        tuned.push(best.1);
    }
    // control: random-time probing matched to the mean probe rate of coupled-streak,
    // same rec/attr/thr/alpha
    let ref_c = tuned[1.min(tuned.len() - 1)].clone();
    let rate = {
        let mut tot = 0.0;
        for s in &ws {
            let r = eval(&ref_c.base, &ref_c.pol, &tune_seeds, s);
            tot += r.iter().map(|o| o[3]).sum::<f64>() / r.len() as f64 / N_TASKS as f64;
        }
        tot / ws.len() as f64
    };
    println!("[control] matched random probe rate {:.4}/task", rate);
    let mut rc = ref_c.clone();
    rc.name = "control-random-matched".into();
    rc.pol.trig = Trig::Random(rate);
    tuned.push(rc);

    // ---- held-out
    struct T {
        world: &'static str,
        cand: usize,
        h: usize,
        bb: usize,
        d: neural_mesh::stats::PairedDiff,
        dg: neural_mesh::stats::PairedDiff,
        dm: neural_mesh::stats::PairedDiff,
        p: f64,
    }
    let mut tests: Vec<T> = vec![];
    let mut means_json = vec![];
    let mut sens = vec![];
    for s in &ws {
        let base: Vec<Vec<[f64; 4]>> = baselines.iter().map(|c| eval(&c.base, &c.pol, &held, s)).collect();
        let cr: Vec<Vec<[f64; 4]>> = tuned.iter().map(|c| eval(&c.base, &c.pol, &held, s)).collect();
        println!("\n=== {} | held-out {h0}-{h1} | NET (c={PROBE_COST}) at H=500 / 2000 ; probes/2000 ===", s.name);
        let mut mj = vec![];
        for (i, c) in baselines.iter().enumerate() {
            let m = [mean_net(&base[i], 0, PROBE_COST), mean_net(&base[i], 1, PROBE_COST)];
            let pr = base[i].iter().map(|o| o[3]).sum::<f64>() / base[i].len() as f64;
            println!("{:<28}{:>9.4}{:>9.4}{:>9.1}", c.name, m[0], m[1], pr);
            mj.push(json!({"router": c.name, "net": m, "probes2000": pr}));
        }
        for (i, c) in tuned.iter().enumerate() {
            let m = [mean_net(&cr[i], 0, PROBE_COST), mean_net(&cr[i], 1, PROBE_COST)];
            let pr = cr[i].iter().map(|o| o[3]).sum::<f64>() / cr[i].len() as f64;
            println!("{:<28}{:>9.4}{:>9.4}{:>9.1}  <- candidate", c.name, m[0], m[1], pr);
            mj.push(json!({"router": c.name, "net": m, "probes2000": pr}));
            for cc in [0.0, 0.02, 0.1, 0.25] {
                let bestb = (0..base.len()).map(|b| mean_net(&base[b], 1, cc)).fold(f64::NEG_INFINITY, f64::max);
                sens.push(json!({"world": s.name, "cand": c.name, "c": cc, "net2000": mean_net(&cr[i], 1, cc), "best_baseline_net2000": bestb}));
            }
        }
        means_json.push(json!({"world": s.name, "rows": mj}));
        for h in 0..2 {
            let colb = |b: usize| -> Vec<f64> { base[b].iter().map(|o| net(o, h, PROBE_COST)).collect() };
            let bb = (0..base.len()).max_by(|&a, &b| summarize(&colb(a)).mean.partial_cmp(&summarize(&colb(b)).mean).unwrap()).unwrap();
            for ci in 0..tuned.len() {
                let c: Vec<f64> = cr[ci].iter().map(|o| net(o, h, PROBE_COST)).collect();
                tests.push(T {
                    world: s.name,
                    cand: ci,
                    h,
                    bb,
                    d: paired_diff(&c, &colb(bb)),
                    dg: paired_diff(&c, &colb(5)),
                    dm: paired_diff(&c, &colb(8)),
                    p: paired_t_pvalue(&c, &colb(bb)),
                });
            }
        }
    }
    let adj = holm_adjust(&tests.iter().map(|t| t.p).collect::<Vec<_>>());
    println!("\n=== paired NET diffs vs BEST fair baseline per (world,H); Holm over {} tests ===", tests.len());
    let mut tj = vec![];
    for (i, t) in tests.iter().enumerate() {
        let sig = adj[i] < 0.05;
        let v = if sig && t.d.diff.mean > 0.0 { "WIN" } else if sig { "LOSS" } else { "ns" };
        println!(
            "{:<28} {:<24} H={:<5} vs {:<20} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.4} {:<4} | vs guarded {:+.4} [{:+.4},{:+.4}] | vs mesh-bench(guarded) {:+.4} [{:+.4},{:+.4}]",
            t.world, tuned[t.cand].name, HORIZONS[t.h], baselines[t.bb].name, t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, t.d.wins, t.d.losses, adj[i], v,
            t.dg.diff.mean, t.dg.diff.lo, t.dg.diff.hi, t.dm.diff.mean, t.dm.diff.lo, t.dm.diff.hi
        );
        tj.push(json!({"world": t.world, "cand": tuned[t.cand].name, "horizon": HORIZONS[t.h], "best_baseline": baselines[t.bb].name,
            "diff": t.d.diff, "wins": t.d.wins, "losses": t.d.losses, "p_raw": t.p, "holm_p": adj[i], "verdict": v,
            "vs_guarded": t.dg.diff, "vs_mesh_bench_guarded": t.dm.diff}));
    }
    let j = json!({
        "held_out": format!("{h0}-{h1}"), "probe_cost": PROBE_COST, "probe_acc": PROBE_ACC,
        "protocol": "tune seeds 1-8 (regret-averse vs best fair baseline NET); held-out given; paired t on NET vs best baseline per (world,H); Holm over all",
        "tuned": tuned.iter().map(|c| json!({"cand": c.name, "pol": format!("{:?}", c.pol)})).collect::<Vec<_>>(),
        "means": means_json, "tests": tj, "cost_sensitivity": sens
    });
    let _ = std::fs::create_dir_all("results");
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out}");
}
