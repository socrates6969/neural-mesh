//! F7-B: STRESS the e1 recommended DEFAULT router (`defaults::recommended_router`, guarded-safe)
//! on extra worlds that were NOT used for any tuning. NOTHING is tuned here: the router is the
//! frozen constant from src/defaults.rs; the fair baselines use e1's tuned values (eps-greedy
//! gamma .9 eps 0, pooled-ts gamma 1, hebbian-robust, warm(hebr) k=1 MeanFeature, guarded-x1).
//!
//! Extra worlds (all mine, synthetic; metric = mean EXPECTED score of the chosen agent over the
//! first H tasks, H in {100,250,500,2000}):
//!   abrupt-drift      : 8 topics, 5 full re-draws (t=300,700,1100,1500,1800), stale keyword map
//!   gradual-drift     : every topic's skill row morphs linearly p0 -> p1 over ~800 tasks (staggered)
//!   agents-join       : 6 agents; at t=600 two NEW agents join (one 0.95 specialist on 4 topics);
//!                       keyword map knows only the original agents
//!   agents-leave      : 6 agents; at t=1000 the two agents advertised most often LEAVE
//!   churn-both        : joins at t=500 and leaves at t=1200 together
//!   noisy+failing     : d5_quality's world (crashing specialists, noisy generalists, 8 agents)
//!   many-generalists  : d5_quality's world (generalists, 8 agents)
//!   bursty-mix        : 12 topics, topic sequence is a Markov chain (stay prob .95), 3 traps
//!   bursty-drift      : bursty + 3 drift events
//!   pool32-traps      : 32 agents, 24 topics, 4 traps, map err .1
//!   pool32-correct    : 32 agents, 24 topics, honest keywords (no-harm control)
//!   ref-traps(main), ref-correct-keywords : e1's own generators re-run as a REFERENCE
//!                       (check that this copy of the harness reproduces e1's numbers)
//!
//! Usage: cargo run --release --offline --bin f7_stress -- [--held 31-60] [--out results/f7_stress.json]

use std::collections::HashMap;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::defaults::{recommended_guard, recommended_inner, recommended_router};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize, PairedDiff};
use rayon::prelude::*;
use serde_json::json;

const HORIZONS: [usize; 4] = [100, 250, 500, 2000];
const N_TASKS: usize = 2000;
const NH: usize = 4;

struct World {
    agents: usize,
    tasks: Vec<Task>,
    exp: Vec<Vec<f64>>,
    real: Vec<Vec<f64>>,
    kwmap: HashMap<String, usize>,
    /// per agent: (available from task, available until task (exclusive))
    avail: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
struct Spec {
    topics: usize,
    agents: usize,
    traps: usize,
    map_err: f64,
    confound: f64,
    drift_at: Vec<usize>,
    drift_frac: f64,
    burst: f64,
    gradual: bool,
    /// (initial agents, join time, leave time, n leavers)
    churn: Option<(usize, usize, usize, usize)>,
}

fn base(topics: usize, traps: usize, map_err: f64, confound: f64) -> Spec {
    Spec { topics, agents: 6, traps, map_err, confound, drift_at: vec![], drift_frac: 0.0, burst: 0.0, gradual: false, churn: None }
}

#[derive(Clone, Debug)]
enum Kind {
    Short(Spec),
    #[allow(dead_code)]
    Sim(neural_mesh::sim::Scenario),
    D5(D5Kind),
    /// e1's generators, for reference
    E1Short { topics: usize, traps: usize, map_err: f64 },
}

#[derive(Clone, Copy, Debug)]
enum D5Kind {
    NoisyFailing,
    Generalists,
}

struct WSpec {
    name: &'static str,
    kind: Kind,
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
    (0..a_n).map(|a| if a == best { 0.85 } else if a == second { 0.5 } else { 0.15 + 0.25 * rng.f64() }).collect()
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

/// A short-generator world (see module doc). Agents `0..n0` exist from the start; the churn
/// variants add/remove agents later. The keyword map is built from the INITIAL agents only.
fn gen_short(seed: u64, sp: &Spec, salt: u64) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(11 + salt * 131));
    let t_n = sp.topics;
    let a_n = sp.agents;
    let n0 = sp.churn.map_or(a_n, |c| c.0);
    let mut p = vec![vec![0.0; a_n]; t_n];
    let mut best = vec![0usize; t_n];
    let mut second = vec![0usize; t_n];
    for t in 0..t_n {
        best[t] = rng.below(n0);
        loop {
            second[t] = rng.below(n0);
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
        // new agents (churn): agent n0 is a strong specialist on the first 4 topics, agent n0+1 mediocre everywhere
        if sp.churn.is_some() && a_n >= n0 + 2 {
            p[t][n0] = if t < 4 { 0.95 } else { 0.15 + 0.25 * rng.f64() };
            p[t][n0 + 1] = 0.6;
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
                    let a = rng.below(n0);
                    if n0 <= 2 || (a != best[t] && a != second[t]) {
                        break a;
                    }
                }
            } else if rng.f64() < sp.map_err {
                rng.below(n0)
            } else {
                best[t]
            };
            kwmap.insert(format!("t{t}k{j}"), a);
        }
    }
    // availability
    let mut avail = vec![(0usize, usize::MAX); a_n];
    if let Some((_, tj, tl, nl)) = sp.churn {
        for a in n0..a_n {
            avail[a].0 = tj;
        }
        // leavers: the agents advertised by the most keywords (most useful to a keyword router)
        let mut cnt = vec![0usize; a_n];
        for v in kwmap.values() {
            cnt[*v] += 1;
        }
        let mut idx: Vec<usize> = (0..n0).collect();
        idx.sort_by(|&x, &y| cnt[y].cmp(&cnt[x]).then(x.cmp(&y)));
        for &a in idx.iter().take(nl) {
            avail[a].1 = tl;
        }
    }
    // gradual drift: second skill table + per-topic morph window
    let mut p1 = p.clone();
    let mut morph = vec![(0usize, 1usize); t_n];
    if sp.gradual {
        let mut grng = Rng::new(seed.wrapping_mul(15_485_863).wrapping_add(3 + salt));
        for t in 0..t_n {
            p1[t] = make_row(&mut grng, a_n);
            let s = grng.below(1100);
            morph[t] = (s, s + 800);
        }
    }
    let mut drng = Rng::new(seed.wrapping_mul(104_729).wrapping_add(5 + salt));
    let mut cur = p.clone();
    let (mut tasks, mut exp, mut real) = (vec![], vec![], vec![]);
    let mut prev_t = 0usize;
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
        let t = if sp.burst > 0.0 && i > 0 && rng.f64() < sp.burst { prev_t } else { rng.below(t_n) };
        prev_t = t;
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(sp.confound) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        let u = rng.f64();
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        let row: Vec<f64> = if sp.gradual {
            let (s, e) = morph[t];
            let lam = ((i as f64 - s as f64) / (e - s) as f64).clamp(0.0, 1.0);
            (0..a_n).map(|a| (1.0 - lam) * p[t][a] + lam * p1[t][a]).collect()
        } else {
            cur[t].clone()
        };
        real.push(row.iter().map(|&q| if u < q { 1.0 } else { 0.0 }).collect());
        exp.push(row);
    }
    World { agents: a_n, tasks, exp, real, kwmap, avail }
}

/// e1's `gen_short` (verbatim logic, no drift/mislabel) as a reference.
fn gen_e1_short(seed: u64, topics: usize, traps: usize, map_err: f64) -> World {
    let sp = Spec { topics, agents: 6, traps, map_err, confound: 0.25, ..base(topics, traps, map_err, 0.25) };
    // identical RNG stream to e1 for non-drift worlds
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
    let (mut tasks, mut exp, mut real) = (vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let t = rng.below(t_n);
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(sp.confound) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        let u = rng.f64();
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        real.push(p[t].iter().map(|&q| if u < q { 1.0 } else { 0.0 }).collect());
        exp.push(p[t].clone());
    }
    World { agents: a_n, tasks, exp, real, kwmap, avail: vec![(0, usize::MAX); a_n] }
}

// ---------------------------------------------------------------- 32-agent pool
fn gen_pool32(seed: u64, topics: usize, traps: usize, map_err: f64, salt: u64) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(11 + salt * 131));
    let a_n = 32usize;
    let t_n = topics;
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
    let (mut tasks, mut exp, mut real) = (vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let t = rng.below(t_n);
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(0.25) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        let u = rng.f64();
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        real.push(p[t].iter().map(|&q| if u < q { 1.0 } else { 0.0 }).collect());
        exp.push(p[t].clone());
    }
    World { agents: a_n, tasks, exp, real, kwmap, avail: vec![(0, usize::MAX); a_n] }
}

// ---------------------------------------------------------------- d5_quality generators
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

/// d5_quality's `build_agents` + `gen_world` for the NoisyFailing / Generalists kinds
/// (same RNG layout as d5_quality; world index `wi` is the d5 index of that world).
fn gen_d5(seed: u64, kind: D5Kind, wi: u64, map_err: f64) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(wi * 101 + 3));
    let mut s = vec![[0.0f64; NS]; NA];
    let mut sd = [0.08f64; NA];
    let mut crash = [0.0f64; NA];
    let specialist = |s: &mut Vec<[f64; NS]>, rng: &mut Rng, nown: usize, owner: &dyn Fn(usize) -> usize| {
        for a in 0..nown {
            let owned: Vec<usize> = (0..NS).filter(|&x| owner(x) == a).collect();
            for x in 0..NS {
                s[a][x] = if owned.contains(&x) {
                    0.88 + 0.04 * gauss(rng)
                } else if owned.iter().any(|&o| o / 4 == x / 4) {
                    0.40 + 0.08 * rng.f64()
                } else {
                    0.28 + 0.08 * rng.f64()
                };
            }
        }
    };
    let mut nominal: Option<Vec<[f64; NS]>> = None;
    match kind {
        D5Kind::Generalists => {
            for a in 0..6 {
                let b = 0.62 + 0.16 * rng.f64();
                for x in 0..NS {
                    s[a][x] = b + 0.07 * gauss(&mut rng);
                }
            }
            let mut subs: Vec<usize> = (0..NS).collect();
            shuffle(&mut subs, &mut rng);
            for a in 6..8 {
                for x in 0..NS {
                    s[a][x] = 0.6 + 0.04 * gauss(&mut rng);
                }
                for j in 0..2 {
                    s[a][subs[(a - 6) * 2 + j]] = 0.88;
                }
            }
            sd = [0.1; NA];
        }
        D5Kind::NoisyFailing => {
            let mut p4: Vec<usize> = (0..4).collect();
            shuffle(&mut p4, &mut rng);
            specialist(&mut s, &mut rng, 4, &|x| p4[x % 4]);
            for a in 0..4 {
                for x in 0..NS {
                    if s[a][x] > 0.8 {
                        s[a][x] = 0.82 + 0.03 * gauss(&mut rng);
                    }
                }
            }
            let mut subs: Vec<usize> = (0..NS).collect();
            shuffle(&mut subs, &mut rng);
            let mut nom = s.clone();
            for a in 4..6 {
                for x in 0..NS {
                    s[a][x] = 0.30 + 0.05 * rng.f64();
                    nom[a][x] = s[a][x];
                }
                for j in 0..3 {
                    s[a][subs[(a - 4) * 3 + j]] = 0.93;
                    nom[a][subs[(a - 4) * 3 + j]] = 0.93;
                }
                crash[a] = 0.5;
            }
            for a in 6..8 {
                for x in 0..NS {
                    s[a][x] = 0.68 + 0.05 * gauss(&mut rng);
                    nom[a][x] = s[a][x];
                }
                sd[a] = 0.3;
            }
            nominal = Some(nom);
        }
    }
    for a in 0..NA {
        for x in 0..NS {
            s[a][x] = cl(s[a][x]);
        }
    }
    let nominal = nominal.unwrap_or_else(|| s.clone());
    let mut ad = [0usize; NS];
    for x in 0..NS {
        let col: Vec<f64> = (0..NA).map(|a| nominal[a][x]).collect();
        ad[x] = argmax(&col);
        if rng.f64() < map_err {
            ad[x] = rng.below(NA);
        }
    }
    let mut kwmap = HashMap::new();
    for x in 0..NS {
        for k in SUB[x].1 {
            kwmap.insert(k.to_string(), ad[x]);
        }
    }
    for d in 0..4 {
        let m: Vec<f64> = (0..NA).map(|a| (0..4).map(|j| nominal[a][d * 4 + j]).sum::<f64>()).collect();
        kwmap.insert(DOMKW[d].to_string(), argmax(&m));
    }
    let (mut tasks, mut exp, mut real) = (vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let s1 = rng.below(NS);
        let _mixed = rng.bernoulli(0.0);
        let diff = rng.f64();
        let mut f: Vec<String> = vec![];
        let _shown = rng.bernoulli(0.0);
        let before = f.len();
        for k in SUB[s1].1 {
            if rng.bernoulli(0.55) {
                f.push(k.to_string());
            }
        }
        if f.len() == before && rng.bernoulli(0.85) {
            f.push(SUB[s1].1[rng.below(3)].to_string());
        }
        if rng.bernoulli(0.4) {
            f.push(DOMKW[s1 / 4].to_string());
        }
        if s1 / 4 == 3 && rng.bernoulli(0.35) {
            f.push(SUB[rng.below(4)].1[rng.below(3)].to_string());
        }
        let _decoy = rng.bernoulli(0.0);
        for _ in 0..2 {
            if rng.bernoulli(0.3) {
                f.push(GENERIC[rng.below(6)].to_string());
            }
        }
        f.dedup();
        let mut e = vec![0.0; NA];
        let mut r = vec![0.0; NA];
        for a in 0..NA {
            let q = cl(s[a][s1] * (1.0 - 0.3 * diff));
            e[a] = (1.0 - crash[a]) * q + crash[a] * 0.05;
            let cu = rng.f64();
            let z = gauss(&mut rng);
            r[a] = if cu < crash[a] { 0.05 } else { (q + sd[a] * z).clamp(0.0, 1.0) };
        }
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        exp.push(e);
        real.push(r);
    }
    World { agents: NA, tasks, exp, real, kwmap, avail: vec![(0, usize::MAX); NA] }
}

fn worlds() -> Vec<WSpec> {
    vec![
        WSpec { name: "abrupt-drift", kind: Kind::Short(Spec { drift_at: vec![300, 700, 1100, 1500, 1800], drift_frac: 1.0, ..base(8, 1, 0.05, 0.25) }) },
        WSpec { name: "gradual-drift", kind: Kind::Short(Spec { gradual: true, ..base(8, 1, 0.05, 0.25) }) },
        WSpec { name: "agents-join", kind: Kind::Short(Spec { agents: 8, churn: Some((6, 600, usize::MAX, 0)), ..base(8, 2, 0.10, 0.25) }) },
        WSpec { name: "agents-leave", kind: Kind::Short(Spec { agents: 6, churn: Some((6, 0, 1000, 2)), ..base(8, 2, 0.10, 0.25) }) },
        WSpec { name: "churn-both", kind: Kind::Short(Spec { agents: 8, churn: Some((6, 500, 1200, 2)), ..base(8, 2, 0.10, 0.25) }) },
        WSpec { name: "noisy+failing", kind: Kind::D5(D5Kind::NoisyFailing) },
        WSpec { name: "many-generalists", kind: Kind::D5(D5Kind::Generalists) },
        WSpec { name: "bursty-mix", kind: Kind::Short(Spec { burst: 0.95, ..base(12, 3, 0.10, 0.25) }) },
        WSpec { name: "bursty-drift", kind: Kind::Short(Spec { burst: 0.95, drift_at: vec![500, 1000, 1500], drift_frac: 0.6, ..base(12, 2, 0.05, 0.25) }) },
        WSpec { name: "pool32-traps", kind: Kind::Short(Spec { agents: 32, ..base(24, 4, 0.10, 0.25) }) },
        WSpec { name: "pool32-correct", kind: Kind::Short(Spec { agents: 32, ..base(24, 0, 0.0, 0.25) }) },
        WSpec { name: "ref-traps(main)", kind: Kind::E1Short { topics: 8, traps: 2, map_err: 0.10 } },
        WSpec { name: "ref-correct-keywords", kind: Kind::E1Short { topics: 8, traps: 0, map_err: 0.0 } },
    ]
}

fn gen_world(seed: u64, s: &WSpec, wi: u64) -> World {
    match &s.kind {
        Kind::Short(sp) => {
            if sp.agents == 32 {
                // pool32 uses its own generator
                gen_pool32(seed, sp.topics, sp.traps, sp.map_err, wi)
            } else {
                gen_short(seed, sp, wi)
            }
        }
        Kind::Sim(sc) => {
            let w = neural_mesh::sim::World::new(seed, neural_mesh::sim::WorldConfig { n_tasks: N_TASKS, scenario: *sc, ..Default::default() });
            let n = w.n_agents();
            let tasks: Vec<Task> = w.tasks().iter().map(|s| s.task.clone()).collect();
            let exp: Vec<Vec<f64>> = (0..w.len()).map(|t| (0..n).map(|a| w.success_prob(t, &[a])).collect()).collect();
            let real: Vec<Vec<f64>> = w.tasks().iter().enumerate().map(|(i, s)| exp[i].iter().map(|&p| if s.u < p { 1.0 } else { 0.0 }).collect()).collect();
            World { agents: n, tasks, exp, real, kwmap: w.keyword_map_for(), avail: vec![(0, usize::MAX); n] }
        }
        Kind::D5(k) => match k {
            D5Kind::NoisyFailing => gen_d5(seed, *k, 4, 0.05),
            D5Kind::Generalists => gen_d5(seed, *k, 1, 0.10),
        },
        Kind::E1Short { topics, traps, map_err } => gen_e1_short(seed, *topics, *traps, *map_err),
    }
}

// ---------------------------------------------------------------- routers (all FROZEN)
fn heb_cfg(decay: f64) -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay, epsilon: 0.0, ..HebbianConfig::default() }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum R {
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm,
    X1,
    Safe,
}
const ALL: [(R, &str); 7] = [
    (R::Keyword, "keyword"),
    (R::Eps, "eps-greedy(.9,0)"),
    (R::Pts, "pooled-ts(1)"),
    (R::HebR, "hebbian-robust"),
    (R::Warm, "warm(hebr)"),
    (R::X1, "guarded-x1"),
    (R::Safe, "DEFAULT=guarded-safe"),
];

fn drive<Rt: Router>(mut r: Rt, w: &World, seed: u64, viol: &mut usize) -> [f64; NH] {
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut cum = 0.0;
    let mut out = [0.0; NH];
    let mut hi = 0;
    let mut cands: Vec<usize> = Vec::with_capacity(w.agents);
    for (i, task) in w.tasks.iter().enumerate() {
        cands.clear();
        for a in 0..w.agents {
            if i >= w.avail[a].0 && i < w.avail[a].1 {
                cands.push(a);
            }
        }
        let mut a = r.select(task, &cands, &mut rng);
        if !cands.contains(&a) {
            *viol += 1;
            a = cands[0];
        }
        cum += w.exp[i][a];
        r.update(task, a, w.real[i][a]);
        if hi < NH && i + 1 == HORIZONS[hi] {
            out[hi] = cum / (i + 1) as f64;
            hi += 1;
        }
    }
    out
}

fn run(k: R, w: &World, seed: u64, viol: &mut usize) -> [f64; NH] {
    let a = w.agents;
    let km = || w.kwmap.clone();
    match k {
        R::Keyword => drive(KeywordRouter::new(km(), a), w, seed, viol),
        R::Eps => drive(EpsGreedyRouter::new(a, 0.9, 0.0), w, seed, viol),
        R::Pts => drive(PooledThompsonRouter::new(a, 1.0), w, seed, viol),
        R::HebR => drive(HebbianRouter::new(a, heb_cfg(0.01)), w, seed, viol),
        R::Warm => drive(Warm::new(HebbianRouter::new(a, heb_cfg(0.01)), km(), a, 1.0).with_evidence(Evidence::MeanFeature), w, seed, viol),
        R::X1 => drive(Guarded::new(HebbianRouter::new(a, heb_cfg(0.01)), km(), a, GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }), w, seed, viol),
        R::Safe => drive(recommended_router(km(), a), w, seed, viol),
    }
}

// ---------------------------------------------------------------- main
fn col(r: &[[f64; NH]], h: usize) -> Vec<f64> {
    r.iter().map(|x| x[h]).collect()
}
struct Test {
    world: usize,
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
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/f7_stress.json".into());
    let (h0, h1) = args
        .iter()
        .position(|a| a == "--held")
        .and_then(|i| args.get(i + 1))
        .and_then(|r| r.split_once('-'))
        .map(|(a, b)| (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap()))
        .unwrap_or((31, 60));
    let held: Vec<u64> = (h0..=h1).collect();
    let ws = worlds();
    let t_start = std::time::Instant::now();
    // sanity: frozen constants are what e1 reported
    assert_eq!(recommended_guard(), GuardConfig { tau: 0.7, z: 0.5, kw_a: 4.0, kw_b: 1.0, discount: 0.98, challenge: true });
    assert_eq!(recommended_inner().decay, 0.01);
    println!("f7_stress: NO tuning; DEFAULT = defaults::recommended_router; {} worlds x {} routers x seeds {h0}-{h1}", ws.len(), ALL.len());

    let mut res: Vec<Vec<Vec<[f64; NH]>>> = vec![]; // [world][router][seed]
    let mut violations: Vec<(String, String, usize)> = vec![];
    for (wi, s) in ws.iter().enumerate() {
        let worlds_h: Vec<World> = held.par_iter().map(|&sd| gen_world(sd, s, wi as u64)).collect();
        let rr: Vec<(Vec<[f64; NH]>, usize)> = ALL
            .par_iter()
            .map(|(k, _)| {
                let mut v = 0usize;
                let rows: Vec<[f64; NH]> = held.iter().enumerate().map(|(i, &sd)| run(*k, &worlds_h[i], sd, &mut v)).collect();
                (rows, v)
            })
            .collect();
        println!("\n=== {} | seeds {h0}-{h1} | mean expected score over first H ({} agents) ===", s.name, worlds_h[0].agents);
        println!("{:<22}{:>8}{:>8}{:>8}{:>8}", "router", 100, 250, 500, 2000);
        for (i, (_, n)) in ALL.iter().enumerate() {
            let m: Vec<f64> = (0..NH).map(|h| summarize(&col(&rr[i].0, h)).mean).collect();
            println!("{:<22}{}", n, m.iter().map(|x| format!("{:>8.4}", x)).collect::<String>());
            if rr[i].1 > 0 {
                violations.push((s.name.to_string(), n.to_string(), rr[i].1));
            }
        }
        res.push(rr.into_iter().map(|x| x.0).collect());
    }
    let ki = 0usize; // keyword
    let wi_warm = 4usize;
    let x1i = 5usize;
    let si = 6usize;
    let fair = [0usize, 1, 2, 3]; // keyword, eps, pts, hebr (e1's "fair baselines")
    let mut ta: Vec<Test> = vec![]; // SAFE vs keyword
    let mut tb: Vec<Test> = vec![]; // SAFE vs warm
    let mut tc: Vec<Test> = vec![]; // SAFE vs best fair baseline
    let mut td: Vec<Test> = vec![]; // SAFE vs x1
    let mut te: Vec<Test> = vec![]; // warm vs keyword (reference)
    for (wi, _) in ws.iter().enumerate() {
        for h in 0..NH {
            let sc = col(&res[wi][si], h);
            let mk = |o: usize, name: &str| {
                let ov = col(&res[wi][o], h);
                Test { world: wi, h, other: name.to_string(), d: paired_diff(&sc, &ov), p: paired_t_pvalue(&sc, &ov) }
            };
            ta.push(mk(ki, "keyword"));
            tb.push(mk(wi_warm, "warm(hebr)"));
            td.push(mk(x1i, "guarded-x1"));
            let bb = *fair.iter().max_by(|&&a, &&b| summarize(&col(&res[wi][a], h)).mean.partial_cmp(&summarize(&col(&res[wi][b], h)).mean).unwrap()).unwrap();
            tc.push(mk(bb, ALL[bb].1));
            let wv = col(&res[wi][wi_warm], h);
            let kv = col(&res[wi][ki], h);
            te.push(Test { world: wi, h, other: "keyword".into(), d: paired_diff(&wv, &kv), p: paired_t_pvalue(&wv, &kv) });
        }
    }
    let adj = |t: &[Test]| holm_adjust(&t.iter().map(|x| x.p).collect::<Vec<_>>());
    let (aa, ab, ac, ad, ae) = (adj(&ta), adj(&tb), adj(&tc), adj(&td), adj(&te));
    let show = |title: &str, ts: &[Test], a: &[f64]| {
        println!("\n=== {title}: Holm over {} tests (WIN/LOSS only) ===", ts.len());
        let (mut nw, mut nl, mut raw_l) = (0, 0, 0);
        for (i, t) in ts.iter().enumerate() {
            let v = verdict(a[i], t.d.diff.mean);
            if v == "WIN" {
                nw += 1;
            }
            if v == "LOSS" {
                nl += 1;
            }
            if t.d.diff.hi < 0.0 {
                raw_l += 1;
            }
            if v != "ns" {
                println!("{:<22} H={:<5} vs {:<18} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.4} {}", ws[t.world].name, HORIZONS[t.h], t.other, t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, t.d.wins, t.d.losses, a[i], v);
            }
        }
        println!("-> {nw} WIN / {nl} LOSS (Holm); raw-CI-below-0 cells: {raw_l}");
    };
    show("A) DEFAULT vs KEYWORD", &ta, &aa);
    show("B) DEFAULT vs warm(hebr)", &tb, &ab);
    show("C) DEFAULT vs BEST FAIR baseline {keyword,eps,pts,hebr}", &tc, &ac);
    show("D) DEFAULT vs guarded-x1", &td, &ad);
    show("E) (reference) warm(hebr) vs keyword", &te, &ae);
    // conservative pooled family: Holm over A+B+C+D together (e1-like multiplicity)
    let pooled_p: Vec<f64> = ta.iter().chain(tb.iter()).chain(tc.iter()).chain(td.iter()).map(|t| t.p).collect();
    let pooled_adj = holm_adjust(&pooled_p);
    let sets: [(&str, &Vec<Test>, usize); 4] = [("A vs keyword", &ta, 0), ("B vs warm", &tb, ta.len()), ("C vs best fair", &tc, ta.len() + tb.len()), ("D vs x1", &td, ta.len() + tb.len() + tc.len())];
    let mut pooled_json = vec![];
    println!("\n=== POOLED Holm over A+B+C+D = {} tests (more conservative) ===", pooled_p.len());
    for (nm, ts, off) in sets.iter() {
        let (mut w, mut l) = (0, 0);
        for (i, t) in ts.iter().enumerate() {
            match verdict(pooled_adj[off + i], t.d.diff.mean) {
                "WIN" => w += 1,
                "LOSS" => {
                    l += 1;
                    println!("  LOSS {nm:<16} {:<22} H={:<5} {:+.4} [{:+.4},{:+.4}] holm {:.3}", ws[t.world].name, HORIZONS[t.h], t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, pooled_adj[off + i]);
                }
                _ => {}
            }
        }
        println!("  {nm}: {w} WIN / {l} LOSS");
        pooled_json.push(json!({"family": nm, "holm_wins": w, "holm_losses": l}));
    }
    if !violations.is_empty() {
        println!("\nWARNING candidate-contract violations (router returned a non-candidate, replaced by cands[0]): {violations:?}");
    }

    // raw (uncorrected) listing of every cell where DEFAULT's 95% CI is below keyword / warm
    println!("\n=== raw-CI losses of DEFAULT (95% CI entirely below 0), no multiplicity correction ===");
    for (nm, ts, a) in [("keyword", &ta, &aa), ("warm(hebr)", &tb, &ab)] {
        for (i, t) in ts.iter().enumerate() {
            if t.d.diff.hi < 0.0 {
                println!("  vs {nm:<10} {:<22} H={:<5} {:+.4} [{:+.4},{:+.4}] holm {:.3}", ws[t.world].name, HORIZONS[t.h], t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, a[i]);
            }
        }
    }

    let tj = |ts: &[Test], a: &[f64]| -> Vec<serde_json::Value> {
        ts.iter()
            .enumerate()
            .map(|(i, t)| json!({"world": ws[t.world].name, "horizon": HORIZONS[t.h], "vs": t.other, "diff": t.d.diff, "wins": t.d.wins, "losses": t.d.losses, "p_raw": t.p, "holm_p": a[i], "verdict": verdict(a[i], t.d.diff.mean)}))
            .collect()
    };
    let means: Vec<serde_json::Value> = ws
        .iter()
        .enumerate()
        .map(|(wi, s)| json!({"world": s.name, "rows": ALL.iter().enumerate().map(|(ri, (_, n))| json!({"router": n, "means": (0..NH).map(|h| summarize(&col(&res[wi][ri], h)).mean).collect::<Vec<_>>()})).collect::<Vec<_>>()}))
        .collect();
    let j = json!({
        "experiment": "f7_stress", "held_out": format!("{h0}-{h1}"),
        "protocol": "NO tuning. DEFAULT = src/defaults.rs recommended_router (frozen). baselines: keyword, eps-greedy(.9,0), pooled-ts(1), hebbian-robust, warm(hebr,k=1,MeanFeature), guarded-x1. paired 95% t-CI, Holm per family (A vs keyword, B vs warm, C vs best fair {kw,eps,pts,hebr} per cell, D vs x1, E warm vs keyword).",
        "horizons": HORIZONS, "worlds": ws.iter().map(|w| w.name).collect::<Vec<_>>(), "means": means,
        "tests_A_default_vs_keyword": tj(&ta, &aa), "tests_B_default_vs_warm": tj(&tb, &ab), "tests_C_default_vs_best_fair": tj(&tc, &ac),
        "tests_D_default_vs_x1": tj(&td, &ad), "tests_E_warm_vs_keyword": tj(&te, &ae),
        "pooled_holm_A_to_D": pooled_json, "contract_violations": violations.iter().map(|v| json!({"world": v.0, "router": v.1, "count": v.2})).collect::<Vec<_>>(),
        "runtime_s": t_start.elapsed().as_secs_f64(),
    });
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out} (total {:.1}s)", t_start.elapsed().as_secs_f64());
}
