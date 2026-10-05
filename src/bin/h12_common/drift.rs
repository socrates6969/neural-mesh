//! H12 new synthetic worlds (written and FIXED before any policy was run): multi-event drift worlds and two controls.
//! SIMULATION ONLY. They measure routing (which fixed agent gets a task), not model reasoning.
//!
//! kw-drift4    keyword world, 8 agents x 12 task types, 4 epochs (3 drift events at T/4, T/2, 3T/4): at each event ~40% of
//!              the types re-draw their best agent. The keyword map is exact for epoch 0 and goes stale after. (family `kw-drift`)
//! xor-drift3   xor-cont stream (6 continuous dims, agents 0/1 win on sign(x0 x1)), role swap at every epoch boundary
//!              (3 events). Tokens are the 8-bin cut of each dim; the keyword map is the informed marginal prior of epoch 0. (family `xor-cont`)
//! ctl-noinfo   NEGATIVE CONTROL (no-context / uninformative tokens): tokens are random, agent skills are static and
//!              task-independent, the keyword map is garbage. Only the global best agent matters.
//! ctl-static   CONTROL (static keyword world): 8 agents x 12 types, NO drift, exact keyword map. The keyword rule is optimal.

use std::collections::HashMap;
use std::sync::Arc;

use neural_mesh::router::{Rng, Task};

use super::{uni, BINS};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrKind {
    KwDrift4,
    XorDrift3,
    CtlNoInfo,
    CtlStatic,
}

impl DrKind {
    pub fn label(self) -> &'static str {
        match self {
            DrKind::KwDrift4 => "kw-drift4",
            DrKind::XorDrift3 => "xor-drift3",
            DrKind::CtlNoInfo => "ctl-noinfo",
            DrKind::CtlStatic => "ctl-static",
        }
    }
    pub fn family(self) -> &'static str {
        match self {
            DrKind::KwDrift4 => "kw-drift",
            DrKind::XorDrift3 => "xor-cont",
            DrKind::CtlNoInfo => "ctl-noinfo",
            DrKind::CtlStatic => "ctl-static",
        }
    }
    pub fn n_agents(self) -> usize {
        match self {
            DrKind::XorDrift3 | DrKind::CtlNoInfo => 6,
            _ => 8,
        }
    }
}

pub struct DriftW {
    pub kind: DrKind,
    pub n: usize,
    pub tasks: Vec<Task>,
    pub dense: Option<Arc<Vec<Vec<f32>>>>,
    pub u: Vec<f64>,
    pub prob: Vec<f32>,
    pub kw: HashMap<String, usize>,
    pub drifts: Vec<usize>,
}

const TYPES: usize = 12;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl DriftW {
    pub fn new(seed: u64, kind: DrKind, t_tasks: usize) -> Self {
        let salt: u64 = match kind {
            DrKind::KwDrift4 => 0x51,
            DrKind::XorDrift3 => 0x52,
            DrKind::CtlNoInfo => 0x53,
            DrKind::CtlStatic => 0x54,
        };
        let mut r = Rng::new(seed.wrapping_mul(0xA24B_AED4_963E_E407) ^ (0x1212_0000 + salt));
        let n = kind.n_agents();
        match kind {
            DrKind::KwDrift4 | DrKind::CtlStatic => Self::kw_world(&mut r, kind, n, t_tasks),
            DrKind::XorDrift3 => Self::xor_world(&mut r, n, t_tasks),
            DrKind::CtlNoInfo => Self::noinfo_world(&mut r, n, t_tasks),
        }
    }

    fn kw_world(r: &mut Rng, kind: DrKind, n: usize, t_tasks: usize) -> Self {
        let epochs = if kind == DrKind::KwDrift4 { 4 } else { 1 };
        let elen = t_tasks.div_ceil(epochs);
        // per-epoch skill tables
        let mut p: Vec<Vec<Vec<f64>>> = Vec::with_capacity(epochs);
        let mut p0 = vec![vec![0.0; n]; TYPES];
        let mut best0 = vec![0usize; TYPES];
        for t in 0..TYPES {
            for a in 0..n {
                p0[t][a] = uni(r, 0.15, 0.40);
            }
            let best = r.below(n);
            p0[t][best] = uni(r, 0.80, 0.90);
            let mut second = r.below(n - 1);
            if second >= best {
                second += 1;
            }
            p0[t][second] = uni(r, 0.55, 0.65);
            best0[t] = best;
        }
        p.push(p0.clone());
        let mut cur = p0;
        let mut cur_best = best0.clone();
        for _e in 1..epochs {
            let mut nx = cur.clone();
            for t in 0..TYPES {
                if r.bernoulli(0.40) {
                    let old = cur_best[t];
                    let mut new = r.below(n - 1);
                    if new >= old {
                        new += 1;
                    }
                    nx[t][old] = uni(r, 0.30, 0.45);
                    nx[t][new] = uni(r, 0.80, 0.90);
                    cur_best[t] = new;
                }
            }
            p.push(nx.clone());
            cur = nx;
        }
        // zipf type frequency
        let mut cdf = Vec::with_capacity(TYPES);
        let mut acc = 0.0;
        for t in 0..TYPES {
            acc += 1.0 / (t as f64 + 1.0);
            cdf.push(acc);
        }
        let mut tasks = Vec::with_capacity(t_tasks);
        let mut u = Vec::with_capacity(t_tasks);
        let mut prob = vec![0.0f32; t_tasks * n];
        for id in 0..t_tasks {
            let x = r.f64() * acc;
            let t = cdf.iter().position(|&c| x < c).unwrap_or(TYPES - 1);
            let k = 2 + r.below(2);
            let mut idx = [0usize, 1, 2, 3];
            for i in (1..4).rev() {
                let j = r.below(i + 1);
                idx.swap(i, j);
            }
            let mut features: Vec<String> = idx[..k].iter().map(|&j| format!("t{t}k{j}")).collect();
            if r.bernoulli(0.25) {
                let mut o = r.below(TYPES - 1);
                if o >= t {
                    o += 1;
                }
                features.push(format!("t{o}k{}", r.below(4)));
            }
            if r.bernoulli(0.30) {
                features.push(format!("n{}", r.below(30)));
            }
            for i in (1..features.len()).rev() {
                let j = r.below(i + 1);
                features.swap(i, j);
            }
            let e = (id / elen).min(epochs - 1);
            for a in 0..n {
                prob[id * n + a] = p[e][t][a] as f32;
            }
            tasks.push(Task { id: id as u64, task_type: None, features });
            u.push(r.f64());
        }
        let mut kw = HashMap::new();
        for t in 0..TYPES {
            for j in 0..4 {
                kw.insert(format!("t{t}k{j}"), best0[t]);
            }
        }
        let drifts: Vec<usize> = (1..epochs).map(|e| e * elen).collect();
        DriftW { kind, n, tasks, dense: None, u, prob, kw, drifts }
    }

    fn xor_world(r: &mut Rng, n: usize, t_tasks: usize) -> Self {
        let d = 6usize;
        let mut flat = vec![0.0f64; n];
        for a in 0..n {
            flat[a] = uni(r, 0.30, 0.50);
        }
        let epochs = 4usize;
        let elen = t_tasks.div_ceil(epochs);
        let mut tasks = Vec::with_capacity(t_tasks);
        let mut dense: Vec<Vec<f32>> = Vec::with_capacity(t_tasks);
        let mut u = Vec::with_capacity(t_tasks);
        let mut prob = vec![0.0f32; t_tasks * n];
        for t in 0..t_tasks {
            let mut x = vec![0.0f32; d];
            let mut feats = Vec::with_capacity(d);
            for j in 0..d {
                x[j] = uni(r, -1.0, 1.0) as f32;
                let b = (((x[j] as f64 + 1.0) / 2.0 * BINS as f64) as usize).min(BINS - 1);
                feats.push(format!("x{j}:b{b}"));
            }
            let e = (t / elen).min(epochs - 1);
            let swap = e % 2 == 1;
            let xx = x[0] as f64 * x[1] as f64;
            let (s0, s1) = if swap { (-1.0, 1.0) } else { (1.0, -1.0) };
            for a in 0..n {
                let p = match a {
                    0 => 0.15 + 0.75 * sigmoid(12.0 * s0 * xx),
                    1 => 0.15 + 0.75 * sigmoid(12.0 * s1 * xx),
                    _ => flat[a] + 0.05 * x[2] as f64 * if a % 2 == 0 { 1.0 } else { -1.0 },
                };
                prob[t * n + a] = p.clamp(0.02, 0.97) as f32;
            }
            u.push(r.f64());
            tasks.push(Task { id: t as u64, task_type: None, features: feats });
            dense.push(x);
        }
        // informed marginal keyword map from epoch 0 (as in g12 xor-cont-drift)
        let upto = elen.min(t_tasks);
        let mut acc: HashMap<String, (Vec<f64>, f64)> = HashMap::new();
        for t in 0..upto {
            for f in &tasks[t].features {
                let e = acc.entry(f.clone()).or_insert_with(|| (vec![0.0; n], 0.0));
                for a in 0..n {
                    e.0[a] += prob[t * n + a] as f64;
                }
                e.1 += 1.0;
            }
        }
        let mut kw = HashMap::new();
        for (k, (s, c)) in acc {
            let mut best = 0;
            for a in 1..n {
                if s[a] / c > s[best] / c + 1e-9 {
                    best = a;
                }
            }
            kw.insert(k, best);
        }
        let drifts: Vec<usize> = (1..epochs).map(|e| e * elen).collect();
        DriftW { kind: DrKind::XorDrift3, n, tasks, dense: Some(Arc::new(dense)), u, prob, kw, drifts }
    }

    fn noinfo_world(r: &mut Rng, n: usize, t_tasks: usize) -> Self {
        let mut p = vec![0.0f64; n];
        for a in 0..n {
            p[a] = uni(r, 0.30, 0.55);
        }
        let best = r.below(n);
        p[best] = 0.75;
        let mut tasks = Vec::with_capacity(t_tasks);
        let mut u = Vec::with_capacity(t_tasks);
        let mut prob = vec![0.0f32; t_tasks * n];
        for id in 0..t_tasks {
            let mut f: Vec<String> = (0..3).map(|_| format!("r{}", r.below(20))).collect();
            f.sort();
            f.dedup();
            for a in 0..n {
                prob[id * n + a] = p[a] as f32;
            }
            tasks.push(Task { id: id as u64, task_type: None, features: f });
            u.push(r.f64());
        }
        let mut kw = HashMap::new();
        for k in 0..20 {
            kw.insert(format!("r{k}"), r.below(n));
        }
        DriftW { kind: DrKind::CtlNoInfo, n, tasks, dense: None, u, prob, kw, drifts: vec![] }
    }
}
