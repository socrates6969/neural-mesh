//! E1 (TUNE-DEFAULT): choose ONE recommended DEFAULT router config.
//!
//! Open issue: d5_quality showed guarded(hebr) with its x1-frozen config is weaker
//! than warm(hebr) after drift and with misleading keywords. This lab
//!  (1) re-tunes guarded(hebr) (config of prior::Guarded + the inner learner) on a
//!      drift-heavy + misleading-keyword world family, tune seeds 1-8 ONLY;
//!  (2) bakes off warm(hebr), guarded(x1 frozen) and the re-tuned guarded configs
//!      against fair baselines {keyword, eps-greedy, pooled-thompson, hebbian-robust}
//!      across 9 worlds x horizons {100,250,500,2000}, held-out seeds (default
//!      31-60; 61-90 = confirmation with the SAME frozen tuning), paired 95% t-CI,
//!      Holm correction (families: vs keyword; vs best fair baseline; head-to-head);
//!  (3) reports whether any single config is never significantly worse than
//!      keyword in any world, and which one is best on average.
//!
//! Worlds (metric: mean EXPECTED score of the chosen agent over the first H tasks):
//!  required 7: traps(main), correct-keywords, many-topics(16), sim-random,
//!    sim-notraps (x1_lab generators), drift-heavy (3 drift events, stale keyword
//!    map), misleading (50% of topics advertise a bad agent + decoy keywords +
//!    mislabelled tasks);
//!  supplementary graded-quality worlds (d5_quality generators, specialist pool):
//!    q-misleading, q-drift.
//!  tune-only extra family members: drift-heavy-b, misleading-b.
//! Usage: cargo run --release --offline --bin e1_default -- [--held 31-60] [--out results/e1_default.json] [--quick]

use std::collections::HashMap;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World as SimWorld, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize, PairedDiff};
use rayon::prelude::*;
use serde_json::json;

const HORIZONS: [usize; 4] = [100, 250, 500, 2000];
const N_TASKS: usize = 2000;
const NH: usize = 4;

// ------------------------------------------------------------------ worlds
struct World {
    agents: usize,
    tasks: Vec<Task>,
    exp: Vec<Vec<f64>>,
    real: Vec<Vec<f64>>,
    kwmap: HashMap<String, usize>,
}

#[derive(Clone, Debug)]
struct ShortSpec {
    topics: usize,
    traps: usize,
    map_err: f64,
    confound: f64,
    mislabel: f64,
    drift_at: Vec<usize>,
    drift_frac: f64,
}

#[derive(Clone, Debug)]
enum WorldKind {
    Short(ShortSpec),
    Sim(Scenario),
    Quality { decoy_p: f64, mislabel_p: f64, trap_frac: f64, drift: bool },
}

#[derive(Clone, Debug)]
struct WorldSpec {
    name: &'static str,
    kind: WorldKind,
}

fn make_row(rng: &mut Rng, a_n: usize) -> Vec<f64> {
    let best = rng.below(a_n);
    let mut second;
    loop {
        second = rng.below(a_n);
        if second != best {
            break;
        }
    }
    (0..a_n)
        .map(|a| if a == best { 0.85 } else if a == second { 0.5 } else { 0.15 + 0.25 * rng.f64() })
        .collect()
}

fn argmax(xs: &[f64]) -> usize {
    let mut b = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[b] {
            b = i;
        }
    }
    b
}

fn gen_short(seed: u64, sp: &ShortSpec) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(11));
    let (t_n, a_n) = (sp.topics, 6usize);
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
    let trap: Vec<bool> = (0..t_n).map(|t| order.iter().take(sp.traps).any(|&x| x == t)).collect();
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
            } else if rng.f64() < sp.map_err {
                rng.below(a_n)
            } else {
                best[t]
            };
            kwmap.insert(format!("t{t}k{j}"), a);
        }
    }
    // drift schedule (separate RNG so non-drift worlds keep the x1 stream)
    let mut drng = Rng::new(seed.wrapping_mul(104_729).wrapping_add(5));
    let mut cur = p.clone();
    let (mut tasks, mut exp, mut real) = (vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        if sp.drift_at.contains(&i) {
            let mut ts: Vec<usize> = (0..t_n).collect();
            for k in (1..t_n).rev() {
                ts.swap(k, drng.below(k + 1));
            }
            let nd = ((sp.drift_frac * t_n as f64).round() as usize).min(t_n);
            for &t in ts.iter().take(nd) {
                cur[t] = make_row(&mut drng, a_n);
            }
        }
        let t = rng.below(t_n);
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(sp.confound) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        if sp.mislabel > 0.0 && rng.bernoulli(sp.mislabel) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{o}k{j}")).collect();
            if f.is_empty() {
                f.push(format!("t{o}k{}", rng.below(3)));
            }
        }
        let u = rng.f64();
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        real.push(cur[t].iter().map(|&q| if u < q { 1.0 } else { 0.0 }).collect());
        exp.push(cur[t].clone());
    }
    World { agents: a_n, tasks, exp, real, kwmap }
}

fn gen_sim(seed: u64, sc: Scenario) -> World {
    let w = SimWorld::new(seed, WorldConfig { n_tasks: N_TASKS, scenario: sc, ..WorldConfig::default() });
    let n = w.n_agents();
    let tasks: Vec<Task> = w.tasks().iter().map(|s| s.task.clone()).collect();
    let exp: Vec<Vec<f64>> = (0..w.len()).map(|t| (0..n).map(|a| w.success_prob(t, &[a])).collect()).collect();
    let real: Vec<Vec<f64>> = w
        .tasks()
        .iter()
        .enumerate()
        .map(|(i, s)| exp[i].iter().map(|&p| if s.u < p { 1.0 } else { 0.0 }).collect())
        .collect();
    World { agents: n, tasks, exp, real, kwmap: w.keyword_map_for() }
}

// ---- graded-quality generator (copied from d5_quality, specialist pool only) ----
const NS: usize = 16;
const NA: usize = 8;
const SUB: [(&str, [&str; 3]); NS] = [
    ("rust", ["rust", "cargo", "borrowck"]),
    ("python", ["python", "pip", "pandas"]),
    ("javascript", ["javascript", "npm", "typescript"]),
    ("sql", ["sql", "postgres", "join"]),
    ("algebra", ["algebra", "polynomial", "equation"]),
    ("proof", ["proof", "lemma", "induction"]),
    ("calculus", ["calculus", "integral", "derivative"]),
    ("statistics", ["statistics", "regression", "variance"]),
    ("physics", ["physics", "momentum", "thermo"]),
    ("chemistry", ["chemistry", "reaction", "molar"]),
    ("biology", ["biology", "protein", "genome"]),
    ("circuits", ["circuits", "voltage", "resistor"]),
    ("stack-trace", ["stack-trace", "panic", "traceback"]),
    ("segfault", ["segfault", "core-dump", "null-ptr"]),
    ("race-condition", ["race", "deadlock", "mutex"]),
    ("memory-leak", ["leak", "heap", "valgrind"]),
];
const DOMKW: [&str; 4] = ["code", "math", "science", "debug"];
const GENERIC: [&str; 6] = ["fix", "help", "urgent", "please", "explain", "review"];

fn gauss(rng: &mut Rng) -> f64 {
    let u1 = rng.f64().max(1e-12);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn shuffle<T>(v: &mut [T], rng: &mut Rng) {
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i + 1));
    }
}
fn cl(x: f64) -> f64 {
    x.clamp(0.02, 0.99)
}

fn gen_quality(seed: u64, wi: u64, decoy_p: f64, mislabel_p: f64, trap_frac: f64, drift: bool) -> World {
    let map_err = 0.05;
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(wi * 101 + 3));
    // agents: 6 specialists + 2 weak generalists
    let mut s = vec![[0.0f64; NS]; NA];
    let sd = [0.08f64; NA];
    let mut perm: Vec<usize> = (0..6).collect();
    shuffle(&mut perm, &mut rng);
    for a in 0..6 {
        let owned: Vec<usize> = (0..NS).filter(|&x| perm[x % 6] == a).collect();
        for x in 0..NS {
            s[a][x] = if owned.contains(&x) {
                0.88 + 0.04 * gauss(&mut rng)
            } else if owned.iter().any(|&o| o / 4 == x / 4) {
                0.40 + 0.08 * rng.f64()
            } else {
                0.28 + 0.08 * rng.f64()
            };
        }
    }
    for a in 6..8 {
        for x in 0..NS {
            s[a][x] = 0.58 + 0.05 * gauss(&mut rng);
        }
    }
    for a in 0..NA {
        for x in 0..NS {
            s[a][x] = cl(s[a][x]);
        }
    }
    let mut s1 = s.clone();
    if drift {
        let mut sp: Vec<usize> = (0..6).collect();
        shuffle(&mut sp, &mut rng);
        for &a in sp.iter().take(3) {
            for x in 0..NS {
                if s[a][x] > 0.8 {
                    s1[a][x] = 0.43;
                    let mut b = rng.below(NA);
                    while b == a {
                        b = rng.below(NA);
                    }
                    s1[b][x] = 0.9;
                }
            }
        }
    }
    let mut ad = [0usize; NS];
    for x in 0..NS {
        let col: Vec<f64> = (0..NA).map(|a| s[a][x]).collect();
        ad[x] = argmax(&col);
        if rng.f64() < map_err {
            ad[x] = rng.below(NA);
        }
    }
    if trap_frac > 0.0 {
        let mut subs: Vec<usize> = (0..NS).collect();
        shuffle(&mut subs, &mut rng);
        let nt = (trap_frac * NS as f64).round() as usize;
        for &x in subs.iter().take(nt) {
            let mut order: Vec<usize> = (0..NA).collect();
            order.sort_by(|&p, &q| s[p][x].partial_cmp(&s[q][x]).unwrap());
            ad[x] = order[rng.below(NA / 2)];
        }
    }
    let mut kwmap = HashMap::new();
    for x in 0..NS {
        for k in SUB[x].1 {
            kwmap.insert(k.to_string(), ad[x]);
        }
    }
    for d in 0..4 {
        let m: Vec<f64> = (0..NA).map(|a| (0..4).map(|j| s[a][d * 4 + j]).sum::<f64>()).collect();
        kwmap.insert(DOMKW[d].to_string(), argmax(&m));
    }
    let tdrift = N_TASKS / 2;
    let (mut tasks, mut exp, mut real) = (vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let sub = rng.below(NS);
        let diff = rng.f64();
        let mut f: Vec<String> = vec![];
        let shown1 = if rng.bernoulli(mislabel_p) { rng.below(NS) } else { sub };
        let before = f.len();
        for k in SUB[shown1].1 {
            if rng.bernoulli(0.55) {
                f.push(k.to_string());
            }
        }
        if f.len() == before && rng.bernoulli(0.85) {
            f.push(SUB[shown1].1[rng.below(3)].to_string());
        }
        if rng.bernoulli(0.4) {
            f.push(DOMKW[sub / 4].to_string());
        }
        if sub / 4 == 3 && rng.bernoulli(0.35) {
            f.push(SUB[rng.below(4)].1[rng.below(3)].to_string());
        }
        if rng.bernoulli(decoy_p) {
            let o = rng.below(NS);
            f.push(SUB[o].1[rng.below(3)].to_string());
        }
        for _ in 0..2 {
            if rng.bernoulli(0.3) {
                f.push(GENERIC[rng.below(6)].to_string());
            }
        }
        f.dedup();
        let sk = if i >= tdrift { &s1 } else { &s };
        let mut e = vec![0.0; NA];
        let mut r = vec![0.0; NA];
        for a in 0..NA {
            let q = cl(sk[a][sub] * (1.0 - 0.3 * diff));
            e[a] = q;
            let _cu = rng.f64(); // keep the d5 stream layout (no crash agents here)
            let z = gauss(&mut rng);
            r[a] = (q + sd[a] * z).clamp(0.0, 1.0);
        }
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        exp.push(e);
        real.push(r);
    }
    World { agents: NA, tasks, exp, real, kwmap }
}

fn gen_world(seed: u64, s: &WorldSpec, wi: u64) -> World {
    match &s.kind {
        WorldKind::Short(sp) => gen_short(seed, sp),
        WorldKind::Sim(sc) => gen_sim(seed, *sc),
        WorldKind::Quality { decoy_p, mislabel_p, trap_frac, drift } => {
            gen_quality(seed, wi, *decoy_p, *mislabel_p, *trap_frac, *drift)
        }
    }
}

fn base_spec(topics: usize, traps: usize, map_err: f64, confound: f64) -> ShortSpec {
    ShortSpec { topics, traps, map_err, confound, mislabel: 0.0, drift_at: vec![], drift_frac: 0.0 }
}

/// Indices 0..9 = evaluation worlds (0..7 required, 7..9 supplementary quality);
/// 9..11 = tune-only family members.
fn worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec { name: "traps(main)", kind: WorldKind::Short(base_spec(8, 2, 0.10, 0.25)) },
        WorldSpec { name: "correct-keywords", kind: WorldKind::Short(base_spec(8, 0, 0.0, 0.25)) },
        WorldSpec { name: "many-topics(16)", kind: WorldKind::Short(base_spec(16, 2, 0.10, 0.25)) },
        WorldSpec { name: "sim-random", kind: WorldKind::Sim(Scenario::Random) },
        WorldSpec { name: "sim-notraps", kind: WorldKind::Sim(Scenario::NoTraps) },
        WorldSpec {
            name: "drift-heavy",
            kind: WorldKind::Short(ShortSpec { drift_at: vec![400, 900, 1400], drift_frac: 0.6, ..base_spec(8, 1, 0.05, 0.25) }),
        },
        WorldSpec {
            name: "misleading",
            kind: WorldKind::Short(ShortSpec { mislabel: 0.15, ..base_spec(8, 4, 0.15, 0.5) }),
        },
        WorldSpec { name: "q-misleading", kind: WorldKind::Quality { decoy_p: 0.3, mislabel_p: 0.15, trap_frac: 0.3, drift: false } },
        WorldSpec { name: "q-drift", kind: WorldKind::Quality { decoy_p: 0.0, mislabel_p: 0.0, trap_frac: 0.0, drift: true } },
        // tune-only
        WorldSpec {
            name: "drift-heavy-b",
            kind: WorldKind::Short(ShortSpec { drift_at: vec![1000], drift_frac: 1.0, ..base_spec(12, 0, 0.0, 0.25) }),
        },
        WorldSpec {
            name: "misleading-b",
            kind: WorldKind::Short(ShortSpec { mislabel: 0.10, ..base_spec(16, 6, 0.10, 0.4) }),
        },
    ]
}
const N_EVAL: usize = 9;
const N_REQ: usize = 7;
/// tune family = drift-heavy + misleading (+ their graded-quality twins and variants)
const FAMILY: [usize; 6] = [5, 6, 7, 8, 9, 10];

// ------------------------------------------------------------------ routers
fn heb_cfg(decay: f64) -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay, epsilon: 0.0, ..HebbianConfig::default() }
}

#[derive(Clone, Copy, Debug)]
enum Inner {
    Heb(f64),
    Eps(f64, f64),
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    HebR,
    WarmHebR,
    Guard(GuardConfig, Inner),
}

fn drive<R: Router>(mut r: R, w: &World, seed: u64) -> [f64; NH] {
    let cands: Vec<usize> = (0..w.agents).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut cum = 0.0;
    let mut out = [0.0; NH];
    let mut hi = 0;
    for (i, task) in w.tasks.iter().enumerate() {
        let a = r.select(task, &cands, &mut rng);
        cum += w.exp[i][a];
        r.update(task, a, w.real[i][a]);
        if hi < NH && i + 1 == HORIZONS[hi] {
            out[hi] = cum / (i + 1) as f64;
            hi += 1;
        }
    }
    out
}

fn run(kind: &Kind, w: &World, seed: u64) -> [f64; NH] {
    let a = w.agents;
    let km = || w.kwmap.clone();
    match kind {
        Kind::Keyword => drive(KeywordRouter::new(km(), a), w, seed),
        Kind::Eps(g, e) => drive(EpsGreedyRouter::new(a, *g, *e), w, seed),
        Kind::Pts(g) => drive(PooledThompsonRouter::new(a, *g), w, seed),
        Kind::HebR => drive(HebbianRouter::new(a, heb_cfg(0.01)), w, seed),
        Kind::WarmHebR => {
            let r = Warm::new(HebbianRouter::new(a, heb_cfg(0.01)), km(), a, 1.0).with_evidence(Evidence::MeanFeature);
            drive(r, w, seed)
        }
        Kind::Guard(g, Inner::Heb(d)) => drive(Guarded::new(HebbianRouter::new(a, heb_cfg(*d)), km(), a, *g), w, seed),
        Kind::Guard(g, Inner::Eps(gm, e)) => drive(Guarded::new(EpsGreedyRouter::new(a, *gm, *e), km(), a, *g), w, seed),
    }
}

fn label(k: &Kind) -> String {
    match k {
        Kind::Keyword => "keyword".into(),
        Kind::Eps(g, e) => format!("eps-greedy(gamma={g},eps={e})"),
        Kind::Pts(g) => format!("pooled-ts(gamma={g})"),
        Kind::HebR => "hebbian-robust".into(),
        Kind::WarmHebR => "warm(hebbian-robust,k=1,MeanFeature)".into(),
        Kind::Guard(g, i) => format!(
            "guarded(tau={},z={},kw=Beta({},{}),discount={},challenge={}; inner={:?})",
            g.tau, g.z, g.kw_a, g.kw_b, g.discount, g.challenge, i
        ),
    }
}

// ------------------------------------------------------------------ tuning
struct Cache {
    /// [world][seed index]
    w: Vec<Vec<World>>,
}

/// Mean over seeds of prefix score per (world, horizon).
fn tmeans(kind: &Kind, c: &Cache) -> Vec<[f64; NH]> {
    c.w.iter()
        .map(|ws| {
            let mut m = [0.0; NH];
            for (si, w) in ws.iter().enumerate() {
                let o = run(kind, w, (si + 1) as u64);
                for h in 0..NH {
                    m[h] += o[h];
                }
            }
            for h in 0..NH {
                m[h] /= ws.len() as f64;
            }
            m
        })
        .collect()
}

fn mean_obj(m: &[[f64; NH]], sel: &[usize]) -> f64 {
    let mut s = 0.0;
    let mut n = 0.0;
    for &w in sel {
        for h in 0..NH {
            s += m[w][h];
            n += 1.0;
        }
    }
    s / n
}

/// Regret-averse objectives over `sel` worlds. dB = cand - best fair incl. warm,
/// dK = cand - keyword.
///  "regret": mean(dB) + mean(min(dB,0)) + mean(min(dK,0))
///  "safe":   mean(dB) + 4*mean(min(dK,0))
fn objs(m: &[[f64; NH]], bestb: &[[f64; NH]], kw: &[[f64; NH]], sel: &[usize]) -> (f64, f64) {
    let (mut sb, mut nb, mut nk, mut n) = (0.0, 0.0, 0.0, 0.0);
    for &w in sel {
        for h in 0..NH {
            let db = m[w][h] - bestb[w][h];
            let dk = m[w][h] - kw[w][h];
            sb += db;
            nb += db.min(0.0);
            nk += dk.min(0.0);
            n += 1.0;
        }
    }
    (sb / n + nb / n + nk / n, sb / n + 4.0 * nk / n)
}

fn best_of(rows: &[Vec<[f64; NH]>]) -> Vec<[f64; NH]> {
    (0..rows[0].len())
        .map(|w| {
            let mut m = [f64::NEG_INFINITY; NH];
            for r in rows {
                for h in 0..NH {
                    m[h] = m[h].max(r[w][h]);
                }
            }
            m
        })
        .collect()
}

fn tune_simple(label_: &str, cands: Vec<Kind>, c: &Cache, all: &[usize]) -> Kind {
    let n = cands.len();
    let scored: Vec<(f64, Kind)> = cands.into_par_iter().map(|k| (mean_obj(&tmeans(&k, c), all), k)).collect();
    let best = scored
        .into_iter()
        .fold(None, |b: Option<(f64, Kind)>, x| match b {
            Some(bb) if bb.0 >= x.0 => Some(bb),
            _ => Some(x),
        })
        .unwrap();
    println!("[tune 1-8] {label_}: {n} configs, best mean score {:.5} -> {}", best.0, label(&best.1));
    best.1
}

// ------------------------------------------------------------------ main
fn parse_range(args: &[String], key: &str, def: (u64, u64)) -> (u64, u64) {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|r| r.split_once('-'))
        .map(|(a, b)| (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap()))
        .unwrap_or(def)
}

fn col(r: &[[f64; NH]], h: usize) -> Vec<f64> {
    r.iter().map(|x| x[h]).collect()
}

struct Test {
    world: usize,
    cand: usize,
    h: usize,
    other: String,
    d: PairedDiff,
    p: f64,
}

fn verdict(adj: f64, mean: f64) -> &'static str {
    if adj < 0.05 && mean > 0.0 {
        "WIN"
    } else if adj < 0.05 {
        "LOSS"
    } else {
        "ns"
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/e1_default.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let (h0, h1) = parse_range(&args, "--held", (31, 60));
    let held: Vec<u64> = (h0..=h1).collect();
    let ws = worlds();
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let t_start = std::time::Instant::now();
    println!("e1_default: {N_TASKS} tasks/run, tune seeds 1-8 on {} worlds (family = drift-heavy/misleading incl. q-twins + variants), held-out {h0}-{h1}", ws.len());

    // ---------------- tuning set (seeds 1-8 only) ----------------
    let cache = Cache {
        w: ws
            .par_iter()
            .enumerate()
            .map(|(wi, s)| tune_seeds.iter().map(|&sd| gen_world(sd, s, wi as u64)).collect())
            .collect(),
    };
    let all_idx: Vec<usize> = (0..ws.len()).collect();

    // fair baselines tuned for the SAME objective (mean score over all tune worlds x horizons)
    let mut eg = vec![];
    for &g in &[1.0, 0.99, 0.95, 0.9] {
        for &e in &[0.0, 0.02, 0.05, 0.1] {
            eg.push(Kind::Eps(g, e));
        }
    }
    let pg: Vec<Kind> = [1.0, 0.99, 0.95, 0.9].iter().map(|&g| Kind::Pts(g)).collect();
    let eps = tune_simple("eps-greedy", eg, &cache, &all_idx);
    let pts = tune_simple("pooled-ts", pg, &cache, &all_idx);
    let baselines: Vec<(String, Kind)> = vec![
        ("keyword".into(), Kind::Keyword),
        ("eps-greedy(tuned)".into(), eps),
        ("pooled-ts(tuned)".into(), pts),
        ("hebbian-robust".into(), Kind::HebR),
    ];
    let warm = Kind::WarmHebR;
    let mut ref_rows: Vec<Vec<[f64; NH]>> = baselines.iter().map(|(_, k)| tmeans(k, &cache)).collect();
    ref_rows.push(tmeans(&warm, &cache));
    let kw_m = ref_rows[0].clone();
    let bestb = best_of(&ref_rows);

    // guarded grid: gate config x inner learner
    let x1 = GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true };
    let mut grid: Vec<Kind> = vec![Kind::Guard(x1, Inner::Heb(0.01))];
    let inners = [Inner::Heb(0.01), Inner::Heb(0.05), Inner::Eps(0.9, 0.0), Inner::Eps(0.95, 0.02)];
    for &inner in &inners {
        for &tau in &[0.4, 0.5, 0.6, 0.7, 0.8] {
            for &z in &[0.5, 1.0, 2.0] {
                for &(ka, kb) in &[(2.0, 1.0), (4.0, 1.0), (8.0, 2.0)] {
                    for &disc in &[1.0, 0.98, 0.95, 0.9] {
                        for &ch in &[false, true] {
                            grid.push(Kind::Guard(GuardConfig { tau, z, kw_a: ka, kw_b: kb, discount: disc, challenge: ch }, inner));
                        }
                    }
                }
            }
        }
    }
    if quick {
        grid.truncate(24);
    }
    let ng = grid.len();
    let scored: Vec<(Kind, Vec<[f64; NH]>)> = grid
        .into_par_iter()
        .map(|k| {
            let m = tmeans(&k, &cache);
            (k, m)
        })
        .collect();
    let pick = |sel: &[usize], which: usize| -> Kind {
        let mut best: Option<(f64, &Kind)> = None;
        for (k, m) in &scored {
            let (a, b) = objs(m, &bestb, &kw_m, sel);
            let v = if which == 0 { a } else { b };
            if best.map_or(true, |bb| v > bb.0) {
                best = Some((v, k));
            }
        }
        let (v, k) = best.unwrap();
        println!(
            "[tune 1-8] guarded {} objective on {} worlds: {} configs, best {:+.5} -> {}",
            if which == 0 { "regret" } else { "safe" },
            sel.len(),
            ng,
            v,
            label(k)
        );
        k.clone()
    };
    let g_fam = pick(&FAMILY, 0);
    let g_all = pick(&all_idx, 0);
    let g_safe = pick(&all_idx, 1);
    let g_x1 = Kind::Guard(x1, Inner::Heb(0.01));
    let cands: Vec<(String, Kind)> = vec![
        ("warm(hebr)".into(), warm.clone()),
        ("guarded-x1(frozen)".into(), g_x1),
        ("guarded-fam(retuned)".into(), g_fam),
        ("guarded-all(retuned)".into(), g_all),
        ("guarded-safe(retuned)".into(), g_safe),
    ];
    println!("tuning done in {:.1}s", t_start.elapsed().as_secs_f64());
    // consistency check: the shipped constant (src/defaults.rs) must equal the tuned "safe" config
    let defaults_match = matches!(&cands[4].1, Kind::Guard(g, Inner::Heb(d)) if *g == neural_mesh::defaults::recommended_guard()
        && (*d - neural_mesh::defaults::recommended_inner().decay).abs() < 1e-12);
    println!("src/defaults.rs recommended_guard() == tuned guarded-safe config: {defaults_match}");

    // ---------------- held-out evaluation ----------------
    let nb = baselines.len();
    let nc = cands.len();
    let mut means_json = vec![];
    let mut res: Vec<Vec<Vec<[f64; NH]>>> = vec![]; // [world][router][seed]
    for (wi, s) in ws.iter().enumerate().take(N_EVAL) {
        let worlds_h: Vec<World> = held.par_iter().map(|&sd| gen_world(sd, s, wi as u64)).collect();
        let all_k: Vec<&Kind> = baselines.iter().map(|(_, k)| k).chain(cands.iter().map(|(_, k)| k)).collect();
        let rr: Vec<Vec<[f64; NH]>> = all_k
            .par_iter()
            .map(|k| held.iter().enumerate().map(|(i, &sd)| run(k, &worlds_h[i], sd)).collect())
            .collect();
        println!("\n=== world {} | held-out {h0}-{h1} | mean expected score over first H ===", s.name);
        println!("{:<26}{:>8}{:>8}{:>8}{:>8}", "router", 100, 250, 500, 2000);
        let mut mj = vec![];
        let names: Vec<&str> = baselines.iter().map(|b| b.0.as_str()).chain(cands.iter().map(|c| c.0.as_str())).collect();
        for (i, n) in names.iter().enumerate() {
            let m: Vec<f64> = (0..NH).map(|h| summarize(&col(&rr[i], h)).mean).collect();
            println!("{:<26}{}{}", n, m.iter().map(|x| format!("{:>8.4}", x)).collect::<String>(), if i >= nb { "   <- candidate" } else { "" });
            mj.push(json!({"router": n, "means": m}));
        }
        means_json.push(json!({"world": s.name, "rows": mj}));
        res.push(rr);
    }

    // tests A: candidate vs keyword; B: vs best fair baseline; C: head-to-head vs warm / x1
    let mut ta: Vec<Test> = vec![];
    let mut tb: Vec<Test> = vec![];
    let mut tc: Vec<Test> = vec![];
    let mut tbase: Vec<Test> = vec![];
    for wi in 0..N_EVAL {
        for h in 0..NH {
            let bb = (0..nb)
                .max_by(|&a, &b| summarize(&col(&res[wi][a], h)).mean.partial_cmp(&summarize(&col(&res[wi][b], h)).mean).unwrap())
                .unwrap();
            for ci in 0..nc {
                let c = col(&res[wi][nb + ci], h);
                let k = col(&res[wi][0], h);
                ta.push(Test { world: wi, cand: ci, h, other: "keyword".into(), d: paired_diff(&c, &k), p: paired_t_pvalue(&c, &k) });
                let b = col(&res[wi][bb], h);
                tb.push(Test { world: wi, cand: ci, h, other: baselines[bb].0.clone(), d: paired_diff(&c, &b), p: paired_t_pvalue(&c, &b) });
                if ci >= 2 {
                    for (oj, on) in [(0usize, "warm(hebr)"), (1usize, "guarded-x1(frozen)")] {
                        let o = col(&res[wi][nb + oj], h);
                        tc.push(Test { world: wi, cand: ci, h, other: on.into(), d: paired_diff(&c, &o), p: paired_t_pvalue(&c, &o) });
                    }
                }
            }
            for bi in 1..nb {
                let c = col(&res[wi][bi], h);
                let k = col(&res[wi][0], h);
                tbase.push(Test { world: wi, cand: bi, h, other: "keyword".into(), d: paired_diff(&c, &k), p: paired_t_pvalue(&c, &k) });
            }
        }
    }
    let adj = |t: &[Test]| holm_adjust(&t.iter().map(|x| x.p).collect::<Vec<_>>());
    let (aa, ab, ac, abase) = (adj(&ta), adj(&tb), adj(&tc), adj(&tbase));
    let cname = |i: usize| cands[i].0.as_str();

    let print_tests = |title: &str, ts: &[Test], ad: &[f64], names: &dyn Fn(usize) -> String| {
        println!("\n=== {title}: Holm over {} tests (WIN/LOSS only listed; ns omitted) ===", ts.len());
        for (i, t) in ts.iter().enumerate() {
            let v = verdict(ad[i], t.d.diff.mean);
            if v != "ns" {
                println!(
                    "{:<18} {:<22} H={:<5} vs {:<22} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.4} {}",
                    ws[t.world].name, names(t.cand), HORIZONS[t.h], t.other, t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, t.d.wins, t.d.losses, ad[i], v
                );
            }
        }
    };
    print_tests("A) candidates vs KEYWORD", &ta, &aa, &|i| cname(i).to_string());
    print_tests("B) candidates vs BEST FAIR baseline", &tb, &ab, &|i| cname(i).to_string());
    print_tests("C) retuned guarded head-to-head", &tc, &ac, &|i| cname(i).to_string());
    print_tests("D) baselines vs keyword", &tbase, &abase, &|i| baselines[i].0.clone());

    // ---------------- per-candidate summary ----------------
    println!("\n=== candidate summary (held-out {h0}-{h1}; 9 worlds x 4 horizons = 36 cells; required = first 7 worlds) ===");
    println!(
        "{:<24}{:>9}{:>10}{:>10}{:>6}{:>6}{:>9}{:>6}{:>6}  worst cell vs keyword",
        "candidate", "mean", "dKW", "dBest", "A-W", "A-L", "A-L(raw)", "B-W", "B-L"
    );
    let mut sj = vec![];
    let cell = |wi: usize, ri: usize, h: usize| summarize(&col(&res[wi][ri], h)).mean;
    let mut rec: Option<(usize, f64)> = None;
    let mut least_bad: Option<(usize, usize, f64)> = None;
    for ci in 0..nc {
        let mut mean = 0.0;
        let mut mean_req = 0.0;
        for wi in 0..N_EVAL {
            for h in 0..NH {
                let v = cell(wi, nb + ci, h);
                mean += v / (N_EVAL * NH) as f64;
                if wi < N_REQ {
                    mean_req += v / (N_REQ * NH) as f64;
                }
            }
        }
        let (mut dkw, mut dbest, mut aw, mut al, mut alraw, mut bw, mut bl) = (0.0, 0.0, 0, 0, 0, 0, 0);
        let (mut al_req, mut worst) = (0, (f64::INFINITY, 0usize, 0usize));
        for (i, t) in ta.iter().enumerate() {
            if t.cand != ci {
                continue;
            }
            dkw += t.d.diff.mean / (N_EVAL * NH) as f64;
            let v = verdict(aa[i], t.d.diff.mean);
            if v == "WIN" {
                aw += 1;
            }
            if v == "LOSS" {
                al += 1;
                if t.world < N_REQ {
                    al_req += 1;
                }
            }
            if t.d.diff.hi < 0.0 {
                alraw += 1;
            }
            if t.d.diff.mean < worst.0 {
                worst = (t.d.diff.mean, t.world, t.h);
            }
        }
        for (i, t) in tb.iter().enumerate() {
            if t.cand != ci {
                continue;
            }
            dbest += t.d.diff.mean / (N_EVAL * NH) as f64;
            match verdict(ab[i], t.d.diff.mean) {
                "WIN" => bw += 1,
                "LOSS" => bl += 1,
                _ => {}
            }
        }
        println!(
            "{:<24}{:>9.4}{:>+10.4}{:>+10.4}{:>6}{:>6}{:>9}{:>6}{:>6}  {:+.4} ({} H{})",
            cname(ci), mean, dkw, dbest, aw, al, alraw, bw, bl, worst.0, ws[worst.1].name, HORIZONS[worst.2]
        );
        sj.push(json!({"candidate": cname(ci), "config": label(&cands[ci].1), "mean_score_all": mean, "mean_score_required7": mean_req,
            "mean_diff_vs_keyword": dkw, "mean_diff_vs_best_fair": dbest, "holm_wins_vs_keyword": aw, "holm_losses_vs_keyword": al,
            "holm_losses_vs_keyword_required7": al_req, "raw_ci_losses_vs_keyword": alraw, "holm_wins_vs_best": bw, "holm_losses_vs_best": bl,
            "worst_cell_vs_keyword": {"diff": worst.0, "world": ws[worst.1].name, "horizon": HORIZONS[worst.2]}}));
        // recommendation: zero Holm losses vs keyword (all 9 worlds), best mean score
        if al == 0 && rec.map_or(true, |r| mean > r.1) {
            rec = Some((ci, mean));
        }
        if least_bad.map_or(true, |lb| (al, -worst.0) < (lb.1, -lb.2)) {
            least_bad = Some((ci, al, worst.0));
        }
    }
    let (rec_json, rec_txt) = match rec {
        Some((ci, m)) => (
            json!({"status": "satisfies never-significantly-worse-than-keyword (Holm, 9 worlds)", "candidate": cname(ci), "config": label(&cands[ci].1), "mean_score": m}),
            format!("{} (mean score {:.4})", cname(ci), m),
        ),
        None => {
            let (ci, al, w) = least_bad.unwrap();
            (
                json!({"status": "NO candidate satisfies the criterion; least-bad", "candidate": cname(ci), "config": label(&cands[ci].1), "holm_losses_vs_keyword": al, "worst_cell": w}),
                format!("NONE satisfies; least-bad = {} ({} Holm losses vs keyword)", cname(ci), al),
            )
        }
    };
    println!("\nRECOMMENDED DEFAULT (this run): {rec_txt}");

    // ---------------- cost ----------------
    println!("\n=== cost: ns per (select+update), traps(main) seed 1 ===");
    let w0 = gen_world(1, &ws[0], 0);
    let mut cj = vec![];
    for (n, k) in baselines.iter().chain(cands.iter()) {
        let t0 = std::time::Instant::now();
        let reps = 10u64;
        for r in 0..reps {
            run(k, &w0, r);
        }
        let ns = t0.elapsed().as_nanos() as f64 / (reps as f64 * N_TASKS as f64);
        println!("{:<26}{:>10.0} ns/route", n, ns);
        cj.push(json!({"router": n, "ns_per_route": ns}));
    }

    let tj = |ts: &[Test], ad: &[f64], names: &dyn Fn(usize) -> String| -> Vec<serde_json::Value> {
        ts.iter()
            .enumerate()
            .map(|(i, t)| {
                json!({"world": ws[t.world].name, "cand": names(t.cand), "horizon": HORIZONS[t.h], "vs": t.other, "diff": t.d.diff,
                "wins": t.d.wins, "losses": t.d.losses, "p_raw": t.p, "holm_p": ad[i], "verdict": verdict(ad[i], t.d.diff.mean)})
            })
            .collect()
    };
    let j = json!({
        "held_out": format!("{h0}-{h1}"),
        "protocol": "tune seeds 1-8 only (11 tune worlds; fair eps/pts tuned on mean score; guarded grid: gate config x inner learner, 3 objectives: family-regret, all-regret, all-safe); held-out seeds as given; paired 95% t-CI; Holm per family (A vs keyword, B vs best fair baseline, C head-to-head, D baselines vs keyword)",
        "horizons": HORIZONS,
        "tuned": {"baselines": baselines.iter().map(|(n,k)| json!({"router": n, "config": label(k)})).collect::<Vec<_>>(),
                   "candidates": cands.iter().map(|(n,k)| json!({"router": n, "config": label(k), "debug": format!("{:?}", k)})).collect::<Vec<_>>(),
                   "guarded_grid_size": ng},
        "worlds": ws.iter().take(N_EVAL).map(|w| w.name).collect::<Vec<_>>(),
        "means": means_json,
        "tests_A_vs_keyword": tj(&ta, &aa, &|i| cname(i).to_string()),
        "tests_B_vs_best_fair": tj(&tb, &ab, &|i| cname(i).to_string()),
        "tests_C_head_to_head": tj(&tc, &ac, &|i| cname(i).to_string()),
        "tests_D_baselines_vs_keyword": tj(&tbase, &abase, &|i| baselines[i].0.clone()),
        "summary": sj, "recommended": rec_json, "cost": cj,
        "runtime_s": t_start.elapsed().as_secs_f64(),
    });
    let _ = std::fs::create_dir_all("results");
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out}  (total {:.1}s)", t_start.elapsed().as_secs_f64());
}
