//! G4 FROZEN WORLD DEFINITION (value function + agent structures).
//! This file was written and hashed BEFORE any diversity policy was run
//! (see results/g4_world_frozen.md). Do not edit after freezing.
//!
//! Value of one executed task in domain d by agent a at time t:
//!   q   = clamp(Q[a][d] + N(0, Q_NOISE))                    (hidden per-domain quality)
//!   o   = style_a(t) + N(0, OUT_NOISE * I_4)                (output embedding, 4-D, hidden style)
//!   nov = min(1, min_{p in last WINDOW outputs of domain d} ||o - p||_{val dims} / NOV_SCALE)  (1 if no history)
//!   creative domain : value = q * (FLOOR + (1-FLOOR) * nov)
//!   correctness dom.: value = q                              (repeats are fine)
//! The reward a router receives is `value` (a scalar). Nothing else is revealed to fair baselines.
//! Policies that claim to exploit novelty additionally observe the output embedding restricted to
//! `obs_dims` (default all 4 dims; world R1 shows only dims 0..2 while value uses dims 2..4).

use neural_mesh::router::{Rng, Task};
use std::collections::{HashMap, VecDeque};

pub const SD: usize = 4;
pub const FLOOR: f64 = 0.25;
pub const NOV_SCALE: f64 = 1.0;
pub const WINDOW: usize = 8;
pub const OUT_NOISE: f64 = 0.08;
pub const Q_NOISE: f64 = 0.08;
pub const N_CONF: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Structure {
    /// every agent q ~ U(.55,.85) per domain, styles ~ random unit vectors
    MultiStyle,
    /// per domain one star agent (q~U(.88,.94)), others q~U(.5,.7), styles random
    StarPlus,
    /// 3 style clusters (jitter .15), quality correlated with style[0], styles drift at drift_at
    ClusteredDrift,
}

#[derive(Clone, Debug)]
pub struct WorldCfg {
    pub name: String,
    pub agents: usize,
    pub domains: usize,
    pub n_tasks: usize,
    pub creative: Vec<bool>,
    pub structure: Structure,
    /// dims of the output embedding that policies may observe
    pub obs_dims: (usize, usize),
    /// dims used by the novelty computation of the world
    pub val_dims: (usize, usize),
    pub drift_at: Option<usize>,
}

impl WorldCfg {
    pub fn new(name: &str, structure: Structure, creative: Vec<bool>) -> Self {
        WorldCfg {
            name: name.into(),
            agents: 8,
            domains: creative.len(),
            n_tasks: 1200,
            creative,
            structure,
            obs_dims: (0, SD),
            val_dims: (0, SD),
            drift_at: None,
        }
    }
    pub fn n_creative(&self) -> usize {
        self.creative.iter().filter(|c| **c).count()
    }
}

pub struct SimTask {
    pub domain: usize,
    pub conf: usize,
    pub out_noise: [f64; SD],
    pub q_noise: f64,
}

pub struct World {
    pub cfg: WorldCfg,
    pub q: Vec<Vec<f64>>,       // [agent][domain]
    pub style0: Vec<[f64; SD]>, // initial style
    pub style1: Vec<[f64; SD]>, // post-drift style
    pub tasks: Vec<SimTask>,
    pub kw: HashMap<String, usize>,
}

fn normal(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - rng.f64();
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn unit(v: [f64; SD]) -> [f64; SD] {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
    let mut o = [0.0; SD];
    for i in 0..SD {
        o[i] = v[i] / n;
    }
    o
}

fn rand_unit(rng: &mut Rng) -> [f64; SD] {
    let mut v = [0.0; SD];
    for x in v.iter_mut() {
        *x = normal(rng);
    }
    unit(v)
}

impl World {
    pub fn new(seed: u64, cfg: WorldCfg) -> World {
        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
        let (na, nd) = (cfg.agents, cfg.domains);
        let mut style0 = vec![[0.0; SD]; na];
        let mut style1 = vec![[0.0; SD]; na];
        match cfg.structure {
            Structure::ClusteredDrift => {
                let centers: Vec<[f64; SD]> = (0..3).map(|_| rand_unit(&mut rng)).collect();
                for a in 0..na {
                    let c = centers[a % 3];
                    let mut v = c;
                    for x in v.iter_mut() {
                        *x += 0.15 * normal(&mut rng);
                    }
                    style0[a] = unit(v);
                    let mut w = style0[a];
                    let r = rand_unit(&mut rng);
                    for i in 0..SD {
                        w[i] += 1.0 * r[i];
                    }
                    style1[a] = unit(w);
                }
            }
            _ => {
                for a in 0..na {
                    style0[a] = rand_unit(&mut rng);
                    style1[a] = style0[a];
                }
            }
        }
        // R1-style worlds: dims outside obs are independent of dims inside (re-draw so that
        // dims 2..4 carry no information about dims 0..2)
        if cfg.val_dims != cfg.obs_dims {
            for a in 0..na {
                let mut v = [0.0; SD];
                for x in v.iter_mut() {
                    *x = normal(&mut rng);
                }
                let h = SD / 2;
                let n1 = (v[..h].iter().map(|x| x * x).sum::<f64>()).sqrt().max(1e-9);
                let n2 = (v[h..].iter().map(|x| x * x).sum::<f64>()).sqrt().max(1e-9);
                for i in 0..SD {
                    v[i] /= if i < h { n1 } else { n2 };
                    v[i] *= std::f64::consts::FRAC_1_SQRT_2;
                }
                style0[a] = v;
                style1[a] = v;
            }
        }
        let mut q = vec![vec![0.0; nd]; na];
        match cfg.structure {
            Structure::MultiStyle => {
                for a in 0..na {
                    for d in 0..nd {
                        q[a][d] = 0.55 + 0.30 * rng.f64();
                    }
                }
            }
            Structure::StarPlus => {
                for d in 0..nd {
                    let star = (d * 3 + 1) % na;
                    for a in 0..na {
                        q[a][d] = if a == star { 0.88 + 0.06 * rng.f64() } else { 0.5 + 0.2 * rng.f64() };
                    }
                }
            }
            Structure::ClusteredDrift => {
                for a in 0..na {
                    for d in 0..nd {
                        q[a][d] = (0.55 + 0.30 * style0[a][0] + 0.10 * (rng.f64() - 0.5)).clamp(0.3, 0.95);
                    }
                }
            }
        }
        let mut tasks = Vec::with_capacity(cfg.n_tasks);
        for _ in 0..cfg.n_tasks {
            let domain = rng.below(nd);
            let conf = rng.below(N_CONF);
            let mut out_noise = [0.0; SD];
            for x in out_noise.iter_mut() {
                *x = OUT_NOISE * normal(&mut rng);
            }
            let q_noise = Q_NOISE * normal(&mut rng);
            tasks.push(SimTask { domain, conf, out_noise, q_noise });
        }
        let mut kw = HashMap::new();
        for d in 0..nd {
            let best = (0..na).max_by(|&x, &y| q[x][d].partial_cmp(&q[y][d]).unwrap()).unwrap();
            let a = if rng.f64() < 0.7 { best } else { rng.below(na) };
            kw.insert(format!("dom{d}"), a);
        }
        World { cfg, q, style0, style1, tasks, kw }
    }

    pub fn style(&self, a: usize, t: usize) -> [f64; SD] {
        match self.cfg.drift_at {
            Some(td) if t >= td => self.style1[a],
            _ => self.style0[a],
        }
    }

    pub fn task(&self, t: usize) -> Task {
        let st = &self.tasks[t];
        Task { id: t as u64, task_type: None, features: vec![format!("dom{}", st.domain), format!("conf{}", st.conf)] }
    }
}

pub fn dist_dims(a: &[f64], b: &[f64], lo: usize, hi: usize) -> f64 {
    let mut s = 0.0;
    for i in lo..hi {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

/// World novelty of output `o` vs the domain history.
pub fn world_novelty(o: &[f64], hist: &VecDeque<[f64; SD]>, val_dims: (usize, usize)) -> f64 {
    if hist.is_empty() {
        return 1.0;
    }
    let mut m = f64::INFINITY;
    for p in hist {
        m = m.min(dist_dims(o, p, val_dims.0, val_dims.1));
    }
    (m / NOV_SCALE).min(1.0)
}

/// Read-only view handed to policies. ONLY the oracle-ish arm may read anything but `t`;
/// every other policy ignores it (verifiable by grep: `v.` appears only in the Oracle impl).
pub struct View<'a> {
    pub w: &'a World,
    pub t: usize,
    pub hist: &'a [VecDeque<[f64; SD]>],
    pub last_nov: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RunOut {
    pub overall: f64,
    pub creative: f64,
    pub correct: f64,
    pub nov_creative: f64,
    pub early: f64,
    pub n_creative_tasks: usize,
    pub n_correct_tasks: usize,
    pub diag: Vec<f64>,
}

pub trait Pol {
    fn name(&self) -> String;
    fn select(&mut self, task: &Task, cand: &[usize], rng: &mut Rng, v: &View) -> usize;
    /// `out` = output embedding restricted to the world's obs_dims.
    fn observe(&mut self, task: &Task, agent: usize, reward: f64, out: &[f64], v: &View);
    fn diag(&self) -> Vec<f64> {
        vec![]
    }
}

pub fn run(w: &World, pol: &mut dyn Pol, policy_seed: u64) -> RunOut {
    let cfg = &w.cfg;
    let cand: Vec<usize> = (0..cfg.agents).collect();
    let mut hist: Vec<VecDeque<[f64; SD]>> = (0..cfg.domains).map(|_| VecDeque::new()).collect();
    let mut prng = Rng::new(policy_seed.wrapping_mul(7919).wrapping_add(13));
    let (mut sum, mut sc, mut sk, mut nc, mut nk, mut sn, mut se) = (0.0, 0.0, 0.0, 0usize, 0usize, 0.0, 0.0);
    for t in 0..cfg.n_tasks {
        let st = &w.tasks[t];
        let d = st.domain;
        let task = w.task(t);
        let v = View { w, t, hist: &hist, last_nov: 1.0 };
        let a = pol.select(&task, &cand, &mut prng, &v);
        let s = w.style(a, t);
        let mut o = [0.0; SD];
        for i in 0..SD {
            o[i] = s[i] + st.out_noise[i];
        }
        let q = (w.q[a][d] + st.q_noise).clamp(0.0, 1.0);
        let nov = world_novelty(&o, &hist[d], cfg.val_dims);
        let value = if cfg.creative[d] { q * (FLOOR + (1.0 - FLOOR) * nov) } else { q };
        let v2 = View { w, t, hist: &hist, last_nov: nov };
        pol.observe(&task, a, value, &o[cfg.obs_dims.0..cfg.obs_dims.1], &v2);
        hist[d].push_back(o);
        if hist[d].len() > WINDOW {
            hist[d].pop_front();
        }
        sum += value;
        if t < 300 {
            se += value;
        }
        if cfg.creative[d] {
            sc += value;
            nc += 1;
            sn += nov;
        } else {
            sk += value;
            nk += 1;
        }
    }
    let n = cfg.n_tasks as f64;
    RunOut {
        overall: sum / n,
        creative: if nc > 0 { sc / nc as f64 } else { f64::NAN },
        correct: if nk > 0 { sk / nk as f64 } else { f64::NAN },
        nov_creative: if nc > 0 { sn / nc as f64 } else { f64::NAN },
        early: se / 300.0,
        n_creative_tasks: nc,
        n_correct_tasks: nk,
        diag: pol.diag(),
    }
}

/// The 9 evaluation worlds (definition fixed before policy results).
pub fn worlds() -> Vec<WorldCfg> {
    let all_c = |n: usize| vec![true; n];
    let all_k = |n: usize| vec![false; n];
    let mut c3 = WorldCfg::new("C3-clustered-drift", Structure::ClusteredDrift, all_c(4));
    c3.drift_at = Some(600);
    let mut k3 = WorldCfg::new("K3-clustered-drift-correct", Structure::ClusteredDrift, all_k(4));
    k3.drift_at = Some(600);
    let mut r1 = WorldCfg::new("R1-rigged-hidden-novelty-dims", Structure::MultiStyle, all_c(4));
    r1.obs_dims = (0, 2);
    r1.val_dims = (2, 4);
    vec![
        WorldCfg::new("C1-multistyle", Structure::MultiStyle, all_c(4)),
        WorldCfg::new("C2-star-plus-diverse", Structure::StarPlus, all_c(4)),
        c3,
        WorldCfg::new("K1-multistyle-correct", Structure::MultiStyle, all_k(4)),
        WorldCfg::new("K2-star-correct", Structure::StarPlus, all_k(4)),
        k3,
        WorldCfg::new("M1-mixed-3of6-creative", Structure::MultiStyle, vec![true, true, true, false, false, false]),
        WorldCfg::new("M2-mixed-1of6-creative-star", Structure::StarPlus, vec![true, false, false, false, false, false]),
        r1,
    ]
}
