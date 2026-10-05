//! H7 WORLD DEFINITIONS (frozen before any H7 policy was run; see results/h7_worlds_frozen.md for sha256 + timestamp).
//! Contains: (a) an adapter that re-expresses the frozen W4 generator (g4 world.rs) with a pluggable value function,
//! (b) four NEW generators whose value functions differ from W4's:
//!   N1 grid/QD-archive : outputs = 2-D behaviour coords; value = q*(floor+(1-floor)*g), g=1 iff the 8x8 archive cell of the domain was NEVER filled
//!                        (permanent archive = fixed coverage objective), else regain=.15; per-cell quality landscape (amp .2).
//!   N2 corpus          : 5-D unit styles; novelty = mean of 3 nearest distances to a GLOBAL corpus (60 seed outputs + every delivered output, all domains), scale .8
//!   N3 modal-concave   : every agent emits from TWO modes (3-D); novelty vs last 6 outputs of the domain (min dist, scale .8) passed through a concave
//!                        utility (1-exp(-4 nov))/(1-exp(-4))  (diminishing returns)
//!   N4 grid-recent24   : like N1 but the archive only remembers the last 24 delivered cells of the domain (novelty practically unattainable: 8 agents x ~5 cells)
//!   NK*                : the same generators with value = q only (correctness controls); NM1/NM2: 3 of 6 domains creative.
use super::world::{World, WorldCfg, Structure, SD, OUT_NOISE, Q_NOISE};
use neural_mesh::router::{Rng, Task};
use std::collections::{HashMap, VecDeque};

pub fn normal(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - rng.f64();
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
pub fn hash3(a: u64, b: u64, c: u64) -> u64 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ a.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 29)).wrapping_add(b.wrapping_mul(0x94D0_49BB_1331_11EB));
    h = (h ^ (h >> 31)).wrapping_add(c.wrapping_mul(0x2545_F491_4F6C_DD1D));
    h ^ (h >> 33)
}
pub fn dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

#[derive(Clone, Debug)]
pub enum Val {
    /// novelty vs last `window` outputs of the domain; value q*(floor+(1-floor)*u(nov)); conc>0 => concave u
    Win { floor: f64, window: usize, scale: f64, mean_mode: bool, conc: f64 },
    /// archive of m x m cells over the first two coords in [0,1]; window 0 = permanent
    Arch { m: usize, regain: f64, floor: f64, window: usize },
    /// global corpus kNN novelty
    Corp { k: usize, scale: f64, floor: f64 },
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gen {
    W4,
    Modal,
    Grid,
    Corp5,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TagMode {
    Truth,
    AllTrue,
    FalsePos(f64),
}

#[derive(Clone, Debug)]
pub struct Cand {
    pub agent: usize,
    pub q: f64,
    pub out: Vec<f64>,
}

pub struct HEnv {
    pub name: String,
    pub gk: Gen,
    pub val: Val,
    pub seed: u64,
    pub na: usize,
    pub nd: usize,
    pub nt: usize,
    pub dom: Vec<usize>,
    pub conf: Vec<usize>,
    pub vcr: Vec<bool>,      // value rewards novelty in this domain
    pub tag_dom: Vec<bool>,  // declared-creative per domain
    pub tag_task: Vec<bool>, // tag shown with each task
    pub q: Vec<Vec<f64>>,
    pub land: Vec<Vec<f64>>,
    pub modes: Vec<Vec<Vec<f64>>>, // [agent][mode][dim]
    pub dim: usize,
    pub obs: (usize, usize),
    pub noise_sd: f64,
    pub space_scale: f64,
    pub seed_corpus: Vec<Vec<f64>>,
    pub shell: World, // W4 world object (real world for Gen::W4; tiny shell otherwise): used only to build g4 policies + View
}

#[derive(Clone, Default)]
pub struct St {
    pub win: Vec<VecDeque<Vec<f64>>>,
    pub cnt: Vec<HashMap<u64, u32>>,
    pub cellwin: Vec<VecDeque<u64>>,
    pub corpus: Vec<Vec<f64>>,
}

fn rand_unit_n(rng: &mut Rng, n: usize) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|_| normal(rng)).collect();
    let nn = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
    for x in v.iter_mut() {
        *x /= nn;
    }
    v
}

fn tags(vcr: &[bool], dom: &[usize], seed: u64, tm: TagMode) -> Vec<bool> {
    let mut r = Rng::new(hash3(seed, 99, 7));
    dom.iter()
        .map(|&d| {
            let u = r.f64();
            if vcr[d] {
                true
            } else {
                match tm {
                    TagMode::Truth => false,
                    TagMode::AllTrue => true,
                    TagMode::FalsePos(p) => u < p,
                }
            }
        })
        .collect()
}

fn shell_world(seed: u64, na: usize, nd: usize, creative: &[bool]) -> World {
    let mut cfg = WorldCfg::new("shell", Structure::MultiStyle, creative.to_vec());
    cfg.agents = na;
    cfg.domains = nd;
    cfg.n_tasks = 1;
    World::new(seed, cfg)
}

/// Adapter: the frozen W4 generator (world.rs World::new) with value function `val` (frozen = Win{.25,8,1.0,false,0}).
pub fn w4_env(cfg: &WorldCfg, seed: u64, val: Val, tm: TagMode) -> HEnv {
    let w = World::new(seed, cfg.clone());
    let (na, nd, nt) = (cfg.agents, cfg.domains, cfg.n_tasks);
    let dom: Vec<usize> = w.tasks.iter().map(|t| t.domain).collect();
    let conf: Vec<usize> = w.tasks.iter().map(|t| t.conf).collect();
    let vcr = cfg.creative.clone();
    let tag_task = tags(&vcr, &dom, seed, tm);
    let modes: Vec<Vec<Vec<f64>>> = (0..na).map(|a| vec![w.style0[a].to_vec(), w.style1[a].to_vec()]).collect();
    HEnv {
        name: cfg.name.clone(),
        gk: Gen::W4,
        val,
        seed,
        na,
        nd,
        nt,
        dom,
        conf,
        tag_dom: vcr.clone(),
        vcr,
        tag_task,
        q: w.q.clone(),
        land: vec![],
        modes,
        dim: SD,
        obs: cfg.obs_dims,
        noise_sd: OUT_NOISE,
        space_scale: 1.0,
        seed_corpus: vec![],
        shell: w,
    }
}

#[derive(Clone, Debug)]
pub struct NCfg {
    pub name: String,
    pub gk: Gen,
    pub val: Val,
    pub creative: Vec<bool>,
}
pub fn n_worlds() -> Vec<NCfg> {
    let c4 = vec![true; 4];
    let k4 = vec![false; 4];
    let m6 = vec![true, true, true, false, false, false];
    let arch_perm = Val::Arch { m: 8, regain: 0.15, floor: 0.25, window: 0 };
    let arch_w24 = Val::Arch { m: 8, regain: 0.15, floor: 0.25, window: 24 };
    let corp = Val::Corp { k: 3, scale: 0.8, floor: 0.25 };
    let modal = Val::Win { floor: 0.25, window: 6, scale: 0.8, mean_mode: false, conc: 4.0 };
    let n = |s: &str, g: Gen, v: &Val, c: &Vec<bool>| NCfg { name: s.into(), gk: g, val: v.clone(), creative: c.clone() };
    vec![
        n("N1-grid-archive", Gen::Grid, &arch_perm, &c4),
        n("N2-global-corpus", Gen::Corp5, &corp, &c4),
        n("N3-modal-concave", Gen::Modal, &modal, &c4),
        n("N4-grid-recent24-unattainable", Gen::Grid, &arch_w24, &c4),
        n("NK1-grid-correct", Gen::Grid, &arch_perm, &k4),
        n("NK2-corpus-correct", Gen::Corp5, &corp, &k4),
        n("NK3-modal-correct", Gen::Modal, &modal, &k4),
        n("NM1-grid-mixed-3of6", Gen::Grid, &arch_perm, &m6),
        n("NM2-corpus-mixed-3of6", Gen::Corp5, &corp, &m6),
    ]
}

pub fn n_env(cfg: &NCfg, seed: u64, tm: TagMode) -> HEnv {
    let (na, nd, nt) = (8usize, cfg.creative.len(), 1200usize);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(4242));
    let (dim, noise_sd, space_scale) = match cfg.gk {
        Gen::Grid => (2, 0.15, 0.5),
        Gen::Corp5 => (5, OUT_NOISE, 1.0),
        Gen::Modal => (3, OUT_NOISE, 1.0),
        Gen::W4 => unreachable!(),
    };
    let mut modes: Vec<Vec<Vec<f64>>> = vec![];
    for _ in 0..na {
        match cfg.gk {
            Gen::Grid => modes.push(vec![vec![0.1 + 0.8 * rng.f64(), 0.1 + 0.8 * rng.f64()]]),
            Gen::Corp5 => modes.push(vec![rand_unit_n(&mut rng, 5)]),
            Gen::Modal => modes.push(vec![rand_unit_n(&mut rng, 3), rand_unit_n(&mut rng, 3)]),
            Gen::W4 => unreachable!(),
        }
    }
    let mut q = vec![vec![0.0; nd]; na];
    for a in 0..na {
        for d in 0..nd {
            q[a][d] = 0.55 + 0.30 * rng.f64();
        }
    }
    let mut land = vec![];
    if cfg.gk == Gen::Grid {
        for _ in 0..nd {
            land.push((0..64).map(|_| rng.f64()).collect::<Vec<f64>>());
        }
    }
    let mut seed_corpus = vec![];
    if let Val::Corp { .. } = cfg.val {
        let prior: Vec<Vec<f64>> = (0..20).map(|_| rand_unit_n(&mut rng, 5)).collect();
        for p in &prior {
            for _ in 0..3 {
                seed_corpus.push(p.iter().map(|x| x + OUT_NOISE * normal(&mut rng)).collect());
            }
        }
    }
    let mut dom = vec![];
    let mut conf = vec![];
    for _ in 0..nt {
        dom.push(rng.below(nd));
        conf.push(rng.below(4));
    }
    let vcr = cfg.creative.clone();
    let tag_task = tags(&vcr, &dom, seed, tm);
    let mut shell = shell_world(seed, na, nd, &vcr);
    // keyword map as in W4: best agent by q w.p. .7 else random
    let mut kw = HashMap::new();
    for d in 0..nd {
        let best = (0..na).max_by(|&x, &y| q[x][d].partial_cmp(&q[y][d]).unwrap()).unwrap();
        let a = if rng.f64() < 0.7 { best } else { rng.below(na) };
        kw.insert(format!("dom{d}"), a);
    }
    shell.kw = kw;
    HEnv {
        name: cfg.name.clone(),
        gk: cfg.gk,
        val: cfg.val.clone(),
        seed,
        na,
        nd,
        nt,
        dom,
        conf,
        tag_dom: vcr.clone(),
        vcr,
        tag_task,
        q,
        land,
        modes,
        dim,
        obs: (0, dim),
        noise_sd,
        space_scale,
        seed_corpus,
        shell,
    }
}

pub struct VsOff {
    pub sigma: f64,
    pub kappa: f64,
}

fn cell_key(out: &[f64], m: usize) -> u64 {
    let f = |x: f64| ((x.clamp(0.0, 0.999_999) * m as f64) as usize).min(m - 1) as u64;
    f(out[0]) * m as u64 + f(out[1])
}

impl HEnv {
    pub fn task(&self, t: usize) -> Task {
        Task { id: t as u64, task_type: None, features: vec![format!("dom{}", self.dom[t]), format!("conf{}", self.conf[t])] }
    }
    pub fn new_state(&self) -> St {
        St {
            win: (0..self.nd).map(|_| VecDeque::new()).collect(),
            cnt: (0..self.nd).map(|_| HashMap::new()).collect(),
            cellwin: (0..self.nd).map(|_| VecDeque::new()).collect(),
            corpus: self.seed_corpus.clone(),
        }
    }
    /// candidate for task t from agent a, noise slot `slot` (slot 0 = the native single call; common random numbers across policies).
    /// `vs`: verbalized/temperature offset (extra output spread, quality penalty kappa*||offset||/space_scale)
    pub fn emit(&self, t: usize, a: usize, slot: usize, vs: Option<&VsOff>) -> Cand {
        let d = self.dom[t];
        let mut r = Rng::new(hash3(self.seed, t as u64, slot as u64 + 1000));
        let mut nz: Vec<f64> = (0..self.dim).map(|_| normal(&mut r)).collect();
        let mut qn = normal(&mut r) * Q_NOISE;
        let coin = r.f64();
        if self.gk == Gen::W4 && slot == 0 {
            let st = &self.shell.tasks[t];
            nz = st.out_noise.iter().map(|x| x / OUT_NOISE).collect();
            qn = st.q_noise;
        }
        let base: Vec<f64> = match self.gk {
            Gen::W4 => self.shell.style(a, t).to_vec(),
            Gen::Modal => self.modes[a][if coin < 0.5 { 0 } else { 1 }].clone(),
            _ => self.modes[a][0].clone(),
        };
        let mut out: Vec<f64> = base.iter().zip(&nz).map(|(b, e)| b + self.noise_sd * e).collect();
        let mut q = self.q[a][d] + qn;
        if self.gk == Gen::Grid {
            for x in out.iter_mut() {
                *x = x.clamp(0.0, 1.0);
            }
            q += 0.2 * (self.land[d][cell_key(&out, 8) as usize] - 0.5);
        }
        if let Some(v) = vs {
            let mut r2 = Rng::new(hash3(self.seed, t as u64, slot as u64 + 5000));
            let off: Vec<f64> = (0..self.dim).map(|_| v.sigma * self.space_scale * normal(&mut r2)).collect();
            let n = off.iter().map(|x| x * x).sum::<f64>().sqrt();
            for i in 0..self.dim {
                out[i] += off[i];
            }
            if self.gk == Gen::Grid {
                for x in out.iter_mut() {
                    *x = x.clamp(0.0, 1.0);
                }
            }
            q -= v.kappa * n / self.space_scale;
        }
        Cand { agent: a, q: q.clamp(0.0, 1.0), out }
    }
    /// (value, novelty-or-gain) of delivering `out` with quality q in domain d given state st.
    pub fn value(&self, st: &St, d: usize, q: f64, out: &[f64]) -> (f64, f64) {
        if !self.vcr[d] {
            return (q, 1.0);
        }
        match &self.val {
            Val::Win { floor, window: _, scale, mean_mode, conc } => {
                let w = &st.win[d];
                let nov = if w.is_empty() {
                    1.0
                } else if *mean_mode {
                    (w.iter().map(|p| dist(out, p)).sum::<f64>() / w.len() as f64 / scale).min(1.0)
                } else {
                    (w.iter().map(|p| dist(out, p)).fold(f64::INFINITY, f64::min) / scale).min(1.0)
                };
                let u = if *conc > 0.0 { (1.0 - (-conc * nov).exp()) / (1.0 - (-conc).exp()) } else { nov };
                (q * (floor + (1.0 - floor) * u), nov)
            }
            Val::Arch { m, regain, floor, window } => {
                let k = cell_key(out, *m);
                let fresh = if *window == 0 { st.cnt[d].get(&k).copied().unwrap_or(0) == 0 } else { !st.cellwin[d].contains(&k) };
                let g = if fresh { 1.0 } else { *regain };
                (q * (floor + (1.0 - floor) * g), g)
            }
            Val::Corp { k, scale, floor } => {
                let mut ds: Vec<f64> = st.corpus.iter().map(|p| dist(out, p)).collect();
                ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let kk = ds.len().min(*k).max(1);
                let nov = (ds[..kk].iter().sum::<f64>() / kk as f64 / scale).min(1.0);
                (q * (floor + (1.0 - floor) * nov), nov)
            }
        }
    }
    pub fn push(&self, st: &mut St, d: usize, out: &[f64]) {
        match &self.val {
            Val::Win { window, .. } => {
                st.win[d].push_back(out.to_vec());
                if st.win[d].len() > *window {
                    st.win[d].pop_front();
                }
            }
            Val::Arch { m, window, .. } => {
                let k = cell_key(out, *m);
                *st.cnt[d].entry(k).or_insert(0) += 1;
                if *window > 0 {
                    st.cellwin[d].push_back(k);
                    if st.cellwin[d].len() > *window {
                        st.cellwin[d].pop_front();
                    }
                }
            }
            Val::Corp { .. } => st.corpus.push(out.to_vec()),
        }
    }
}
