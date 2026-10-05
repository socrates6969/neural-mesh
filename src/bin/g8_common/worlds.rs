//! G8 world definitions. FIXED BEFORE any policy was run (see board entry).
//!
//! All worlds: 6 agents, expected-success metric (sim-style: mean prob of the chosen agent).
//! A "cat-delayed-id"  categorical topics (8 topics x 3 keywords, 2 trap topics, 10% keyword noise),
//!                      drift at n/2 (3 topics change best agent, keyword map stale), n=2000,
//!                      reward arrives after U{1..20} steps WITH the decision id.
//! B "cat-anon-delay"   categorical topics (10 topics x 4 keywords, 2 traps, 10% noise), no drift,
//!                      n=2500, reward arrives after Geometric(1/6) steps (mean 6) WITHOUT id:
//!                      the router only sees (reward bit, arrival time).
//! C "cont-smooth"      x in [0,1]^2 uniform; agent a succeeds w.p. 0.10 + 0.80*exp(-|x-c_a|^2/(2*0.20^2)).
//!                      Baselines get tokens: 4x4 coarse cell (keyword map keys), 8x8 fine cell, 8-bin x0, 8-bin x1.
//! D "cont-ridge"      x as C; agent a succeeds w.p. 0.12 + 0.76*sigmoid(10*((x-.5).dir_a - b_a)), random
//!                      directions/offsets (oblique half-planes). Same token set for baselines.
//! E "static-immediate" (control) categorical, no delay, no drift, no traps beyond A's: nothing should help.
//! F "cont-noinfo"     (control) C's success probabilities, but the learners/baselines are shown an
//!                      independent random x / tokens (no information): nobody can beat the best constant agent.
//! Counted for the keep-bar: A, B, C, D. E and F are controls.

use neural_mesh::router::{Rng, Task};
use std::collections::HashMap;

pub const NA: usize = 6;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Feedback {
    Immediate,
    DelayedId,
    Anon,
}

pub struct Env {
    pub tasks: Vec<Task>,
    pub x: Vec<[f64; 2]>,
    pub probs: Vec<[f64; NA]>,
    pub u: Vec<f64>,
    pub delay: Vec<usize>,
    pub kmap: HashMap<String, usize>,
    pub fb: Feedback,
}
impl Env {
    pub fn n(&self) -> usize {
        self.tasks.len()
    }
}

pub const WORLD_NAMES: [&str; 6] = ["A cat-delayed-id", "B cat-anon-delay", "C cont-smooth", "D cont-ridge", "E static-immediate(ctl)", "F cont-noinfo(ctl)"];
pub const N_COUNTED: usize = 4;

pub fn make_env(w: usize, seed: u64) -> Env {
    match w {
        0 => cat_env(seed, 101, 8, 3, 2000, 2, true, Feedback::DelayedId),
        1 => cat_env(seed, 202, 10, 4, 2500, 2, false, Feedback::Anon),
        2 => cont_env(seed, 303, false, false),
        3 => cont_env(seed, 404, true, false),
        4 => cat_env(seed, 505, 8, 3, 2000, 2, false, Feedback::Immediate),
        _ => cont_env(seed, 303, false, true),
    }
}

#[allow(clippy::too_many_arguments)]
fn cat_env(seed: u64, tag: u64, tn: usize, kn: usize, n: usize, ntrap: usize, drift: bool, fb: Feedback) -> Env {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(tag));
    let an = NA;
    let mut p = vec![[0.0; NA]; tn];
    let mut best = vec![0usize; tn];
    let mut second = vec![0usize; tn];
    let draw = |rng: &mut Rng, t: usize, p: &mut Vec<[f64; NA]>, best: &mut Vec<usize>, second: &mut Vec<usize>| {
        let old = best[t];
        loop {
            best[t] = rng.below(an);
            if best[t] != old || p[t][0] == 0.0 {
                break;
            }
        }
        loop {
            second[t] = rng.below(an);
            if second[t] != best[t] {
                break;
            }
        }
        for a in 0..an {
            p[t][a] = if a == best[t] {
                0.85
            } else if a == second[t] {
                0.5
            } else {
                0.15 + 0.25 * rng.f64()
            };
        }
    };
    for t in 0..tn {
        draw(&mut rng, t, &mut p, &mut best, &mut second);
    }
    let mut order: Vec<usize> = (0..tn).collect();
    for i in (1..tn).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let trap: Vec<bool> = (0..tn).map(|t| order.iter().take(ntrap).any(|&x| x == t)).collect();
    let mut kmap = HashMap::new();
    for t in 0..tn {
        for j in 0..kn {
            let a = if trap[t] {
                loop {
                    let a = rng.below(an);
                    if a != best[t] && a != second[t] {
                        break a;
                    }
                }
            } else if rng.f64() < 0.10 {
                rng.below(an)
            } else {
                best[t]
            };
            kmap.insert(format!("t{t}k{j}"), a);
        }
    }
    let drift_topics: Vec<usize> = (0..3).map(|i| order[(ntrap + 1 + i) % tn]).collect();
    let mut e = Env { tasks: vec![], x: vec![], probs: vec![], u: vec![], delay: vec![], kmap, fb };
    let mut drng = Rng::new(seed.wrapping_mul(104729).wrapping_add(tag + 7));
    for i in 0..n {
        if drift && i == n / 2 {
            for &t in &drift_topics {
                draw(&mut rng, t, &mut p, &mut best, &mut second);
            }
        }
        let t = rng.below(tn);
        let mut f: Vec<String> = (0..kn).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(kn)));
        }
        if rng.bernoulli(0.25) {
            let o = (t + 1 + rng.below(tn - 1)) % tn;
            f.push(format!("t{o}k{}", rng.below(kn)));
        }
        e.tasks.push(Task { id: i as u64, task_type: None, features: f });
        e.x.push([0.0, 0.0]);
        e.probs.push(p[t]);
        e.u.push(rng.f64());
        e.delay.push(match fb {
            Feedback::Immediate => 0,
            Feedback::DelayedId => 1 + drng.below(20),
            Feedback::Anon => {
                let mut d = 1;
                while !drng.bernoulli(1.0 / 6.0) && d < 80 {
                    d += 1;
                }
                d
            }
        });
    }
    e
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn cont_env(seed: u64, tag: u64, ridge: bool, noinfo: bool) -> Env {
    let n = 2000usize;
    let mut rng = Rng::new(seed.wrapping_mul(6007).wrapping_add(tag));
    // agent parameters
    let mut cen = [[0.0; 2]; NA];
    let mut dir = [[0.0; 2]; NA];
    let mut off = [0.0; NA];
    for a in 0..NA {
        cen[a] = [0.1 + 0.8 * rng.f64(), 0.1 + 0.8 * rng.f64()];
        let th = 2.0 * std::f64::consts::PI * rng.f64();
        dir[a] = [th.cos(), th.sin()];
        off[a] = -0.25 + 0.5 * rng.f64();
    }
    let prob = |x: [f64; 2]| -> [f64; NA] {
        let mut p = [0.0; NA];
        for a in 0..NA {
            p[a] = if ridge {
                let s = (x[0] - 0.5) * dir[a][0] + (x[1] - 0.5) * dir[a][1];
                0.12 + 0.76 * sigmoid(10.0 * (s - off[a]))
            } else {
                let d2 = (x[0] - cen[a][0]).powi(2) + (x[1] - cen[a][1]).powi(2);
                0.10 + 0.80 * (-d2 / (2.0 * 0.20 * 0.20)).exp()
            };
        }
        p
    };
    // keyword map over the 4x4 coarse cells: best agent at the cell centre; 3 trap cells wrong, 10% noise
    let mut kmap = HashMap::new();
    let mut cells: Vec<usize> = (0..16).collect();
    for i in (1..16).rev() {
        cells.swap(i, rng.below(i + 1));
    }
    for c in 0..16 {
        let (ci, cj) = (c / 4, c % 4);
        let p = prob([(ci as f64 + 0.5) / 4.0, (cj as f64 + 0.5) / 4.0]);
        let mut b = 0;
        for a in 1..NA {
            if p[a] > p[b] {
                b = a;
            }
        }
        let a = if cells.iter().take(3).any(|&x| x == c) {
            loop {
                let a = rng.below(NA);
                if a != b {
                    break a;
                }
            }
        } else if rng.f64() < 0.10 {
            rng.below(NA)
        } else {
            b
        };
        kmap.insert(format!("c{ci}{cj}"), a);
    }
    let mut e = Env { tasks: vec![], x: vec![], probs: vec![], u: vec![], delay: vec![], kmap, fb: Feedback::Immediate };
    for i in 0..n {
        let xt = [rng.f64(), rng.f64()];
        let xs = if noinfo { [rng.f64(), rng.f64()] } else { xt };
        let bin = |v: f64, k: usize| ((v * k as f64) as usize).min(k - 1);
        let f = vec![
            format!("c{}{}", bin(xs[0], 4), bin(xs[1], 4)),
            format!("f{}_{}", bin(xs[0], 8), bin(xs[1], 8)),
            format!("a{}", bin(xs[0], 8)),
            format!("b{}", bin(xs[1], 8)),
        ];
        e.tasks.push(Task { id: i as u64, task_type: None, features: f });
        e.x.push(xs);
        e.probs.push(prob(xt));
        e.u.push(rng.f64());
        e.delay.push(0);
    }
    e
}
