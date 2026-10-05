//! G12 common: worlds (existing sim / big worlds with a task-count parameter + new synthetic worlds
//! with continuous features and nonlinear XOR / RBF skill structure), the tiny feed-forward learned
//! router (`AnnRouter`), arm factory, run driver and a counting allocator.
//!
//! EVERYTHING is a seeded simulation written by me. It measures ROUTING only: picking which of a fixed
//! set of agents gets a task. An outer net cannot raise a frozen model's reasoning.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::defaults::recommended_router;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};

pub fn seeds(lo: u64, hi: u64) -> Vec<u64> {
    (lo..=hi).collect()
}

// =============================================================================================
// Worlds

pub const BIG_AGENTS: usize = 32;
pub const BIG_TYPES: usize = 48;
pub const BIG_TASKS_NATIVE: usize = 4000;
const BIG_NOISE: usize = 40;
pub const TAIL_VOCAB: usize = 3000;
pub const TAIL_PROB: f64 = 0.60;

fn uni(r: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * r.f64()
}

/// Copy of `f3_common::BigWorld` with the task count as a parameter (same RNG draw order, so
/// `t_tasks == 4000` reproduces f3's world bit-for-bit; checked by a unit test against f3_common).
pub struct BigW {
    tasks: Vec<Task>,
    ty: Vec<usize>,
    u: Vec<f64>,
    p_pre: Vec<Vec<f64>>,
    p_post: Vec<Vec<f64>>,
    kw: HashMap<String, usize>,
}

impl BigW {
    pub fn new_with(seed: u64, noise_vocab: usize, noise_prob: f64, t_tasks: usize) -> Self {
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xB16_B16);
        let mut p_pre = vec![vec![0.0; BIG_AGENTS]; BIG_TYPES];
        let mut kwa = vec![0usize; BIG_TYPES];
        let mut best_of = vec![0usize; BIG_TYPES];
        for t in 0..BIG_TYPES {
            for a in 0..BIG_AGENTS {
                p_pre[t][a] = uni(&mut r, 0.15, 0.45);
            }
            let best = r.below(BIG_AGENTS);
            p_pre[t][best] = uni(&mut r, 0.80, 0.90);
            let mut second = r.below(BIG_AGENTS - 1);
            if second >= best {
                second += 1;
            }
            p_pre[t][second] = uni(&mut r, 0.55, 0.68);
            best_of[t] = best;
            kwa[t] = if r.bernoulli(0.65) {
                best
            } else {
                let mut w = r.below(BIG_AGENTS - 1);
                if w >= best {
                    w += 1;
                }
                w
            };
        }
        let mut p_post = p_pre.clone();
        for t in 0..BIG_TYPES {
            if r.bernoulli(0.25) {
                let old = best_of[t];
                let mut new = r.below(BIG_AGENTS - 1);
                if new >= old {
                    new += 1;
                }
                p_post[t][old] = uni(&mut r, 0.30, 0.50);
                p_post[t][new] = uni(&mut r, 0.80, 0.90);
            }
        }
        let mut cdf = Vec::with_capacity(BIG_TYPES);
        let mut acc = 0.0;
        for t in 0..BIG_TYPES {
            acc += 1.0 / (t as f64 + 1.0);
            cdf.push(acc);
        }
        let total = acc;
        let mut tasks = Vec::with_capacity(t_tasks);
        let mut ty = Vec::with_capacity(t_tasks);
        let mut u = Vec::with_capacity(t_tasks);
        for id in 0..t_tasks {
            let x = r.f64() * total;
            let t = cdf.iter().position(|&c| x < c).unwrap_or(BIG_TYPES - 1);
            let k = 2 + r.below(2);
            let mut idx = [0usize, 1, 2, 3];
            for i in (1..4).rev() {
                let j = r.below(i + 1);
                idx.swap(i, j);
            }
            let mut features: Vec<String> = idx[..k].iter().map(|&j| format!("t{t}k{j}")).collect();
            if r.bernoulli(0.25) {
                let mut o = r.below(BIG_TYPES - 1);
                if o >= t {
                    o += 1;
                }
                features.push(format!("t{o}k{}", r.below(4)));
            }
            if r.bernoulli(noise_prob) {
                features.push(format!("n{}", r.below(noise_vocab)));
            }
            for i in (1..features.len()).rev() {
                let j = r.below(i + 1);
                features.swap(i, j);
            }
            tasks.push(Task { id: id as u64, task_type: None, features });
            ty.push(t);
            u.push(r.f64());
        }
        let mut kw = HashMap::new();
        for t in 0..BIG_TYPES {
            for j in 0..4 {
                kw.insert(format!("t{t}k{j}"), kwa[t]);
            }
        }
        BigW { tasks, ty, u, p_pre, p_post, kw }
    }
    fn prob(&self, t: usize, a: usize) -> f64 {
        if t < self.tasks.len() / 2 { self.p_pre[self.ty[t]][a] } else { self.p_post[self.ty[t]][a] }
    }
}

// ---------------------------------------------------------------------------------------------
// New synthetic worlds

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynKind {
    /// 6 continuous dims, skill_a = .5+.42 tanh(w_a.x + b_a): LINEAR (additive) structure.
    Lin,
    /// 6 continuous dims, agents 0/1 win on sign(x0*x1) > 0 / < 0 (XOR), 4 flat agents, 4 noise dims.
    XorCont,
    /// XorCont with roles of agents 0 and 1 swapped at T/2 (non-stationary).
    XorDrift,
    /// token-only: bits p,q as tokens `p:0|1`, `q:0|1` + up to 3 noise tokens; agent 0 wins iff p==q, agent 1 iff p!=q.
    XorTok,
    /// 10 continuous dims (3 informative), 8 agents, skill_a = .12+.78 exp(-|x_inf - c_a|^2/(2 .9^2)): smooth nonlinear.
    Rich,
}

impl SynKind {
    pub fn label(self) -> &'static str {
        match self {
            SynKind::Lin => "lin-cont",
            SynKind::XorCont => "xor-cont",
            SynKind::XorDrift => "xor-cont-drift",
            SynKind::XorTok => "xor-tok",
            SynKind::Rich => "rich-cont",
        }
    }
    pub fn n_agents(self) -> usize {
        match self {
            SynKind::Rich => 8,
            _ => 6,
        }
    }
    pub fn dim(self) -> usize {
        match self {
            SynKind::Lin | SynKind::XorCont | SynKind::XorDrift => 6,
            SynKind::XorTok => 0,
            SynKind::Rich => 10,
        }
    }
}

pub const BINS: usize = 8;

pub struct SynW {
    pub kind: SynKind,
    n: usize,
    tasks: Vec<Task>,
    dense: Arc<Vec<Vec<f32>>>,
    u: Vec<f64>,
    prob: Vec<f32>,
    kw: HashMap<String, usize>,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl SynW {
    pub fn new(seed: u64, kind: SynKind, t_tasks: usize) -> Self {
        let salt: u64 = match kind {
            SynKind::Lin => 0x11,
            SynKind::XorCont | SynKind::XorDrift => 0x22, // same task stream + hidden draws
            SynKind::XorTok => 0x33,
            SynKind::Rich => 0x44,
        };
        let mut r = Rng::new(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) ^ (0x6612_0000 + salt));
        let n = kind.n_agents();
        let d = kind.dim();
        // ---- hidden structure (drawn first)
        let mut lw = vec![vec![0.0f64; d]; n];
        let mut lb = vec![0.0f64; n];
        let mut flat = vec![0.0f64; n];
        let mut centers = vec![vec![0.0f64; 4]; n];
        for a in 0..n {
            for j in 0..d {
                lw[a][j] = uni(&mut r, -1.2, 1.2);
            }
            lb[a] = uni(&mut r, -0.4, 0.4);
            flat[a] = uni(&mut r, 0.30, 0.50);
            for c in centers[a].iter_mut() {
                *c = uni(&mut r, -0.8, 0.8);
            }
        }
        let drift_at = t_tasks / 2;
        // ---- tasks
        let mut tasks = Vec::with_capacity(t_tasks);
        let mut dense: Vec<Vec<f32>> = Vec::with_capacity(t_tasks);
        let mut u = Vec::with_capacity(t_tasks);
        let mut prob = vec![0.0f32; t_tasks * n];
        for t in 0..t_tasks {
            let mut feats: Vec<String> = vec![];
            let mut x = vec![0.0f32; d];
            let mut bits = (0usize, 0usize);
            if kind == SynKind::XorTok {
                bits = (r.below(2), r.below(2));
                feats.push(format!("p:{}", bits.0));
                feats.push(format!("q:{}", bits.1));
                for _ in 0..3 {
                    feats.push(format!("n{}", r.below(24)));
                }
                feats.sort();
                feats.dedup();
                for i in (1..feats.len()).rev() {
                    let j = r.below(i + 1);
                    feats.swap(i, j);
                }
            } else {
                for j in 0..d {
                    x[j] = uni(&mut r, -1.0, 1.0) as f32;
                    let b = (((x[j] as f64 + 1.0) / 2.0 * BINS as f64) as usize).min(BINS - 1);
                    feats.push(format!("x{j}:b{b}"));
                }
            }
            for a in 0..n {
                let p = match kind {
                    SynKind::Lin => {
                        let z: f64 = (0..d).map(|j| lw[a][j] * x[j] as f64).sum::<f64>() + lb[a];
                        0.5 + 0.42 * z.tanh()
                    }
                    SynKind::XorCont | SynKind::XorDrift => {
                        let xx = x[0] as f64 * x[1] as f64;
                        let swap = kind == SynKind::XorDrift && t >= drift_at;
                        let (s0, s1) = if swap { (-1.0, 1.0) } else { (1.0, -1.0) };
                        match a {
                            0 => 0.15 + 0.75 * sigmoid(12.0 * s0 * xx),
                            1 => 0.15 + 0.75 * sigmoid(12.0 * s1 * xx),
                            _ => flat[a] + 0.05 * x[2] as f64 * if a % 2 == 0 { 1.0 } else { -1.0 },
                        }
                    }
                    SynKind::XorTok => match a {
                        0 => {
                            if bits.0 == bits.1 { 0.88 } else { 0.18 }
                        }
                        1 => {
                            if bits.0 != bits.1 { 0.88 } else { 0.18 }
                        }
                        _ => flat[a],
                    },
                    SynKind::Rich => {
                        let d2: f64 = (0..3).map(|j| (x[j] as f64 - centers[a][j]).powi(2)).sum();
                        0.12 + 0.78 * (-d2 / (2.0 * 0.9 * 0.9)).exp()
                    }
                };
                prob[t * n + a] = p.clamp(0.02, 0.97) as f32;
            }
            u.push(r.f64());
            tasks.push(Task { id: t as u64, task_type: None, features: feats });
            dense.push(x);
        }
        // ---- keyword map: per token, the agent with the best AVERAGE true success over the tasks
        // carrying it (pre-drift half if the world drifts). Informed marginal prior: right whenever the
        // skill is additive in the tokens, blind to interactions. Ties break to the lowest index.
        let upto = if kind == SynKind::XorDrift { drift_at } else { t_tasks };
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
        SynW { kind, n, tasks, dense: Arc::new(dense), u, prob, kw }
    }
}

pub enum Env {
    Sim(World),
    Big(BigW),
    Syn(SynW),
}

impl Env {
    pub fn n_agents(&self) -> usize {
        match self {
            Env::Sim(w) => w.n_agents(),
            Env::Big(_) => BIG_AGENTS,
            Env::Syn(s) => s.n,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Env::Sim(w) => w.len(),
            Env::Big(b) => b.tasks.len(),
            Env::Syn(s) => s.tasks.len(),
        }
    }
    pub fn task(&self, t: usize) -> &Task {
        match self {
            Env::Sim(w) => &w.task(t).task,
            Env::Big(b) => &b.tasks[t],
            Env::Syn(s) => &s.tasks[t],
        }
    }
    pub fn prob(&self, t: usize, a: usize) -> f64 {
        match self {
            Env::Sim(w) => w.success_prob(t, &[a]),
            Env::Big(b) => b.prob(t, a),
            Env::Syn(s) => s.prob[t * s.n + a] as f64,
        }
    }
    pub fn outcome(&self, t: usize, a: usize) -> f64 {
        let u = match self {
            Env::Sim(w) => return w.outcome(t, &[a]),
            Env::Big(b) => b.u[t],
            Env::Syn(s) => s.u[t],
        };
        if u < self.prob(t, a) { 1.0 } else { 0.0 }
    }
    pub fn oracle(&self, t: usize) -> f64 {
        (0..self.n_agents()).map(|a| self.prob(t, a)).fold(0.0, f64::max)
    }
    pub fn kw_map(&self) -> HashMap<String, usize> {
        match self {
            Env::Sim(w) => w.keyword_map_for(),
            Env::Big(b) => b.kw.clone(),
            Env::Syn(s) => s.kw.clone(),
        }
    }
    /// Side table of continuous features (indexed by `Task::id`), only for the continuous synthetic worlds.
    pub fn dense(&self) -> Option<Arc<Vec<Vec<f32>>>> {
        match self {
            Env::Syn(s) if s.kind.dim() > 0 => Some(s.dense.clone()),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    Low,
    Mid,
    High,
}
impl Regime {
    pub fn label(self) -> &'static str {
        match self {
            Regime::Low => "low",
            Regime::Mid => "mid6000",
            Regime::High => "high20000",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WorldSpec {
    Sim(Scenario),
    Big,
    BigTail,
    Syn(SynKind),
}
impl WorldSpec {
    pub fn label(&self) -> String {
        match self {
            WorldSpec::Sim(s) => format!("sim-{}", s.label()),
            WorldSpec::Big => "big-32agents".into(),
            WorldSpec::BigTail => "big-32agents-longtail".into(),
            WorldSpec::Syn(k) => k.label().into(),
        }
    }
    /// independence family: wins must come from >= 2 DIFFERENT families to count
    pub fn family(&self) -> &'static str {
        match self {
            WorldSpec::Sim(_) => "sim",
            WorldSpec::Big | WorldSpec::BigTail => "big",
            WorldSpec::Syn(SynKind::Lin) => "lin-cont",
            WorldSpec::Syn(SynKind::XorCont) | WorldSpec::Syn(SynKind::XorDrift) => "xor-cont",
            WorldSpec::Syn(SynKind::XorTok) => "xor-tok",
            WorldSpec::Syn(SynKind::Rich) => "rich-cont",
        }
    }
    /// structure class: additive/linear (a net should NOT help) vs interaction/nonlinear
    pub fn nonlinear(&self) -> bool {
        matches!(self, WorldSpec::Syn(SynKind::XorCont | SynKind::XorDrift | SynKind::XorTok | SynKind::Rich))
    }
    pub fn t_for(&self, reg: Regime) -> usize {
        match reg {
            Regime::Low => match self {
                WorldSpec::Big | WorldSpec::BigTail => BIG_TASKS_NATIVE,
                _ => 2000,
            },
            Regime::Mid => 6000,
            Regime::High => 20000,
        }
    }
    pub fn make(&self, seed: u64, t_tasks: usize) -> Env {
        match self {
            WorldSpec::Sim(s) => Env::Sim(World::new(seed, WorldConfig { n_tasks: t_tasks, scenario: *s, ..WorldConfig::default() })),
            WorldSpec::Big => Env::Big(BigW::new_with(seed, BIG_NOISE, 0.30, t_tasks)),
            WorldSpec::BigTail => Env::Big(BigW::new_with(seed, TAIL_VOCAB, TAIL_PROB, t_tasks)),
            WorldSpec::Syn(k) => Env::Syn(SynW::new(seed, *k, t_tasks)),
        }
    }
}

pub fn all_worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec::Sim(Scenario::Fixed),
        WorldSpec::Sim(Scenario::Random),
        WorldSpec::Sim(Scenario::NoTraps),
        WorldSpec::Big,
        WorldSpec::BigTail,
        WorldSpec::Syn(SynKind::Lin),
        WorldSpec::Syn(SynKind::XorCont),
        WorldSpec::Syn(SynKind::XorDrift),
        WorldSpec::Syn(SynKind::XorTok),
        WorldSpec::Syn(SynKind::Rich),
    ]
}

// =============================================================================================
// The learned router: tiny feed-forward net, features -> per-agent logits.

pub const DH: usize = 256; // hashed token slots
pub const MAXI: usize = 12; // max active tokens / dense dims kept per sample
const MAXH: usize = 64;
const MAXN: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    /// per-agent success-probability regression (BCE on the chosen agent's output), softmax(p/tau) exploration
    Reg,
    /// REINFORCE with a running-mean baseline, softmax(logits) policy
    Pg,
}

#[derive(Clone, Copy, Debug)]
pub struct AnnCfg {
    pub mode: Mode,
    pub h: usize,
    pub lr: f32,
    pub tau0: f32,
    pub tau_min: f32,
    /// the exploration gap tau0 - tau_min halves every `half` updates
    pub half: f32,
    pub replay_k: usize,
    pub replay_cap: usize,
    /// give the net the continuous features (baselines only ever see their 8-bin tokens)
    pub use_dense: bool,
    /// feed the keyword map in as a prior: a learned skip connection from keyword votes to the logits
    pub use_kw: bool,
}

impl AnnCfg {
    pub fn show(&self) -> String {
        format!(
            "{}{}{} H{} lr{} tau{}->{} half{} rk{}",
            match self.mode {
                Mode::Reg => "reg",
                Mode::Pg => "pg",
            },
            if self.use_dense { "" } else { "-tok" },
            if self.use_kw { "+kw" } else { "" },
            self.h,
            self.lr,
            self.tau0,
            self.tau_min,
            self.half,
            self.replay_k
        )
    }
}

#[derive(Clone, Copy)]
struct Sample {
    kwa: [u8; MAXI],
    nk: u8,
    idx: [u16; MAXI],
    ni: u8,
    dense: [f32; MAXI],
    nd: u8,
    a: u8,
    r: f32,
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

struct Net {
    h: usize,
    n: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    skip: f32,
}

impl Net {
    fn new(h: usize, n: usize, dd: usize, rng: &mut Rng) -> Self {
        let rows = DH + dd;
        let s1 = 0.5f64;
        let w1 = (0..rows * h).map(|_| uni(rng, -s1, s1) as f32).collect();
        let w2 = (0..n * h).map(|_| uni(rng, -0.1, 0.1) as f32).collect();
        Net { h, n, w1, b1: vec![0.0; h], w2, b2: vec![0.0; n], skip: 1.0 }
    }
    fn n_params(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len() + 1
    }
    fn forward(&self, s: &Sample, hid: &mut [f32; MAXH], z: &mut [f32; MAXN]) {
        let h = self.h;
        hid[..h].copy_from_slice(&self.b1);
        for &i in &s.idx[..s.ni as usize] {
            let row = &self.w1[i as usize * h..(i as usize + 1) * h];
            for k in 0..h {
                hid[k] += row[k];
            }
        }
        for j in 0..s.nd as usize {
            let x = s.dense[j];
            let row = &self.w1[(DH + j) * h..(DH + j + 1) * h];
            for k in 0..h {
                hid[k] += row[k] * x;
            }
        }
        for k in 0..h {
            hid[k] = hid[k].tanh();
        }
        for a in 0..self.n {
            let row = &self.w2[a * h..(a + 1) * h];
            let mut acc = self.b2[a];
            for k in 0..h {
                acc += row[k] * hid[k];
            }
            z[a] = acc;
        }
        if s.nk > 0 {
            let w = self.skip / s.nk as f32;
            for j in 0..s.nk as usize {
                z[s.kwa[j] as usize] += w;
            }
        }
    }
    /// gradient DESCENT step given dz = dLoss/dlogits (zero entries are skipped)
    fn backward(&mut self, s: &Sample, hid: &[f32; MAXH], dz: &[f32; MAXN], lr: f32) {
        let h = self.h;
        let mut dpre = [0.0f32; MAXH];
        for a in 0..self.n {
            if dz[a] != 0.0 {
                let row = &self.w2[a * h..(a + 1) * h];
                for k in 0..h {
                    dpre[k] += dz[a] * row[k];
                }
            }
        }
        for k in 0..h {
            dpre[k] *= 1.0 - hid[k] * hid[k];
        }
        for a in 0..self.n {
            if dz[a] != 0.0 {
                let g = lr * dz[a];
                let row = &mut self.w2[a * h..(a + 1) * h];
                for k in 0..h {
                    row[k] -= g * hid[k];
                }
                self.b2[a] -= g;
            }
        }
        if s.nk > 0 {
            let mut g = 0.0f32;
            for j in 0..s.nk as usize {
                g += dz[s.kwa[j] as usize] / s.nk as f32;
            }
            self.skip -= lr * g;
        }
        for k in 0..h {
            self.b1[k] -= lr * dpre[k];
        }
        for &i in &s.idx[..s.ni as usize] {
            let row = &mut self.w1[i as usize * h..(i as usize + 1) * h];
            for k in 0..h {
                row[k] -= lr * dpre[k];
            }
        }
        for j in 0..s.nd as usize {
            let x = s.dense[j];
            let row = &mut self.w1[(DH + j) * h..(DH + j + 1) * h];
            for k in 0..h {
                row[k] -= lr * dpre[k] * x;
            }
        }
    }
}

pub struct AnnRouter {
    pub cfg: AnnCfg,
    n: usize,
    net: Net,
    dense: Option<Arc<Vec<Vec<f32>>>>,
    kw: Option<HashMap<String, usize>>,
    t: usize,
    rng: Rng,
    buf: Vec<Sample>,
    pos: usize,
    base: f32,
}

impl AnnRouter {
    pub fn new(n_agents: usize, cfg: AnnCfg, dense: Option<Arc<Vec<Vec<f32>>>>, kw: Option<HashMap<String, usize>>, seed: u64) -> Self {
        assert!(cfg.h <= MAXH && n_agents <= MAXN);
        let dense = if cfg.use_dense { dense } else { None };
        let dd = dense.as_ref().map(|d| d[0].len().min(MAXI)).unwrap_or(0);
        let mut rng = Rng::new(seed ^ 0xA22_A22);
        let net = Net::new(cfg.h, n_agents, dd, &mut rng);
        let kw = if cfg.use_kw { kw } else { None };
        AnnRouter { cfg, n: n_agents, net, dense, kw, t: 0, rng, buf: vec![], pos: 0, base: 0.5 }
    }
    /// bytes of learned state: f32 params + replay buffer at its current fill
    pub fn state_bytes(&self) -> usize {
        self.net.n_params() * 4 + self.buf.len() * std::mem::size_of::<Sample>()
    }
    pub fn param_bytes(&self) -> usize {
        self.net.n_params() * 4
    }
    pub fn n_params(&self) -> usize {
        self.net.n_params()
    }
    fn sample_of(&self, task: &Task, a: usize, r: f64) -> Sample {
        let mut s = Sample { kwa: [0; MAXI], nk: 0, idx: [0; MAXI], ni: 0, dense: [0.0; MAXI], nd: 0, a: a as u8, r: r as f32 };
        for f in &task.features {
            if let Some(m) = &self.kw {
                if let Some(&ka) = m.get(f) {
                    if (s.nk as usize) < MAXI {
                        s.kwa[s.nk as usize] = ka as u8;
                        s.nk += 1;
                    }
                }
            }
            let i = (fnv(f) % DH as u64) as u16;
            if (s.ni as usize) < MAXI && !s.idx[..s.ni as usize].contains(&i) {
                s.idx[s.ni as usize] = i;
                s.ni += 1;
            }
        }
        if let Some(d) = &self.dense {
            let v = &d[task.id as usize];
            let k = v.len().min(MAXI);
            s.dense[..k].copy_from_slice(&v[..k]);
            s.nd = k as u8;
        }
        s
    }
    fn tau(&self) -> f32 {
        self.cfg.tau_min + (self.cfg.tau0 - self.cfg.tau_min) * 0.5f32.powf(self.t as f32 / self.cfg.half)
    }
    fn sgd_reg(&mut self, s: &Sample) {
        let mut hid = [0.0f32; MAXH];
        let mut z = [0.0f32; MAXN];
        self.net.forward(s, &mut hid, &mut z);
        let p = 1.0 / (1.0 + (-z[s.a as usize]).exp());
        let mut dz = [0.0f32; MAXN];
        dz[s.a as usize] = p - s.r;
        self.net.backward(s, &hid, &dz, self.cfg.lr);
    }
    fn sgd_pg(&mut self, s: &Sample, adv: f32) {
        let mut hid = [0.0f32; MAXH];
        let mut z = [0.0f32; MAXN];
        self.net.forward(s, &mut hid, &mut z);
        let n = self.n;
        let m = z[..n].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut pi = [0.0f32; MAXN];
        let mut sum = 0.0;
        for a in 0..n {
            pi[a] = (z[a] - m).exp();
            sum += pi[a];
        }
        let mut dz = [0.0f32; MAXN];
        for a in 0..n {
            let p = pi[a] / sum;
            // loss = -adv * log pi(a_chosen): dLoss/dz_b = -adv * (1[b==a] - pi_b)
            dz[a] = -adv * ((if a == s.a as usize { 1.0 } else { 0.0 }) - p);
        }
        self.net.backward(s, &hid, &dz, self.cfg.lr);
    }
}

impl Router for AnnRouter {
    fn name(&self) -> String {
        format!("ann[{}]", self.cfg.show())
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let s = self.sample_of(task, 0, 0.0);
        let mut hid = [0.0f32; MAXH];
        let mut z = [0.0f32; MAXN];
        self.net.forward(&s, &mut hid, &mut z);
        let mut w = [0.0f64; MAXN];
        let mut tot = 0.0;
        match self.cfg.mode {
            Mode::Reg => {
                let tau = self.tau().max(1e-4) as f64;
                let pm = candidates.iter().map(|&c| z[c]).fold(f32::NEG_INFINITY, f32::max);
                let pm = 1.0 / (1.0 + (-pm).exp());
                for (i, &c) in candidates.iter().enumerate() {
                    let p = 1.0 / (1.0 + (-z[c]).exp());
                    w[i] = ((p - pm) as f64 / tau).exp();
                    tot += w[i];
                }
            }
            Mode::Pg => {
                let m = candidates.iter().map(|&c| z[c]).fold(f32::NEG_INFINITY, f32::max);
                for (i, &c) in candidates.iter().enumerate() {
                    w[i] = ((z[c] - m) as f64).exp();
                    tot += w[i];
                }
            }
        }
        let mut x = rng.f64() * tot;
        for (i, &c) in candidates.iter().enumerate() {
            x -= w[i];
            if x < 0.0 {
                return c;
            }
        }
        *candidates.last().unwrap()
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let s = self.sample_of(task, agent, reward.clamp(0.0, 1.0));
        match self.cfg.mode {
            Mode::Reg => {
                self.sgd_reg(&s);
                if self.cfg.replay_k > 0 {
                    if self.buf.len() < self.cfg.replay_cap {
                        self.buf.push(s);
                    } else {
                        self.buf[self.pos] = s;
                        self.pos = (self.pos + 1) % self.cfg.replay_cap;
                    }
                    for _ in 0..self.cfg.replay_k {
                        let j = self.rng.below(self.buf.len());
                        let sj = self.buf[j];
                        self.sgd_reg(&sj);
                    }
                }
            }
            Mode::Pg => {
                let adv = s.r - self.base;
                self.base += 0.02 * (s.r - self.base);
                self.sgd_pg(&s, adv);
            }
        }
        self.t += 1;
    }
}

// =============================================================================================
// Arms

/// Pooled Thompson over single features AND all pairwise conjunctions of features (a cheap
/// non-neural way to see 2-way interactions). Reference arm, not one of the requested baselines.
pub struct PairsThompson {
    inner: PooledThompsonRouter,
}
impl PairsThompson {
    pub fn new(n: usize, gamma: f64) -> Self {
        PairsThompson { inner: PooledThompsonRouter::new(n, gamma) }
    }
    fn expand(task: &Task) -> Task {
        let mut f: Vec<String> = task.features.clone();
        f.sort();
        let mut out = f.clone();
        for i in 0..f.len() {
            for j in (i + 1)..f.len() {
                out.push(format!("{}&{}", f[i], f[j]));
            }
        }
        Task { id: task.id, task_type: task.task_type.clone(), features: out }
    }
}
impl Router for PairsThompson {
    fn name(&self) -> String {
        "pts-pairs".into()
    }
    fn select(&mut self, task: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.inner.select(&Self::expand(task), c, rng)
    }
    fn update(&mut self, task: &Task, a: usize, r: f64) {
        self.inner.update(&Self::expand(task), a, r)
    }
}

#[derive(Clone, Debug)]
pub enum Kind {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    PtsPairs(f64),
    HebR,
    Warm(f64),
    Safe,
    Ann(AnnCfgKey),
}

/// AnnCfg is Copy but has f32 fields; wrap for Debug/Clone in Kind
pub type AnnCfgKey = AnnCfg;

pub fn mk_router(kind: &Kind, env: &Env, seed: u64) -> Box<dyn Router> {
    let n = env.n_agents();
    let km = || env.kw_map();
    match kind {
        Kind::Keyword => Box::new(KeywordRouter::new(km(), n)),
        Kind::Eps(g, e) => Box::new(EpsGreedyRouter::new(n, *g, *e)),
        Kind::Pts(g) => Box::new(PooledThompsonRouter::new(n, *g)),
        Kind::PtsPairs(g) => Box::new(PairsThompson::new(n, *g)),
        Kind::HebR => Box::new(HebbianRouter::new(n, heb_robust_cfg())),
        Kind::Warm(k) => Box::new(Warm::new(HebbianRouter::new(n, heb_robust_cfg()), km(), n, *k).with_evidence(Evidence::MeanFeature)),
        Kind::Safe => Box::new(recommended_router(km(), n)),
        Kind::Ann(c) => Box::new(AnnRouter::new(n, *c, env.dense(), if c.use_kw { Some(env.kw_map()) } else { None }, seed)),
    }
}

pub const BIN: usize = 250;

#[derive(Clone, Debug, Default)]
pub struct RunRes {
    /// mean expected success of the chosen agent per window of BIN tasks
    pub bins: Vec<f32>,
    pub full: f64,
    /// mean over the last quarter of the stream
    pub late: f64,
}

/// Sequential online loop (identical protocol to f3_common::run_policy): select, observe the realised
/// Bernoulli outcome, update. Records the EXPECTED success probability of the chosen agent.
pub fn run_policy(env: &Env, pol: &mut dyn Router, seed: u64) -> RunRes {
    let n = env.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let len = env.len();
    let mut per = vec![0.0f32; len];
    for t in 0..len {
        let task = env.task(t);
        let a = pol.select(task, &cands, &mut rng);
        per[t] = env.prob(t, a) as f32;
        pol.update(task, a, env.outcome(t, a));
    }
    summarize_run(&per)
}

pub fn summarize_run(per: &[f32]) -> RunRes {
    let len = per.len();
    let bins: Vec<f32> = per.chunks(BIN).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect();
    let full = per.iter().map(|&x| x as f64).sum::<f64>() / len as f64;
    let q = len - len / 4;
    let late = per[q..].iter().map(|&x| x as f64).sum::<f64>() / (len - q) as f64;
    RunRes { bins, full, late }
}

pub fn oracle_run(env: &Env) -> RunRes {
    let per: Vec<f32> = (0..env.len()).map(|t| env.oracle(t) as f32).collect();
    summarize_run(&per)
}

/// prefix mean over the first `h` tasks from the bins (h a multiple of BIN)
pub fn prefix_mean(r: &RunRes, h: usize) -> f64 {
    let k = (h / BIN).min(r.bins.len());
    r.bins[..k].iter().map(|&x| x as f64).sum::<f64>() / k as f64
}

// =============================================================================================
// Counting allocator (requested bytes), gated by ON so normal runs only pay one relaxed load.

pub mod alloc_count {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, AtomicI64, Ordering::Relaxed};

    pub struct Counting;
    pub static ON: AtomicBool = AtomicBool::new(false);
    pub static LIVE: AtomicI64 = AtomicI64::new(0);
    pub static PEAK: AtomicI64 = AtomicI64::new(0);

    #[inline]
    fn bump(sz: i64) {
        let now = LIVE.fetch_add(sz, Relaxed) + sz;
        if sz > 0 && now > PEAK.load(Relaxed) {
            PEAK.fetch_max(now, Relaxed);
        }
    }
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc(l) };
            if !p.is_null() && ON.load(Relaxed) {
                bump(l.size() as i64);
            }
            p
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            let p = unsafe { System.alloc_zeroed(l) };
            if !p.is_null() && ON.load(Relaxed) {
                bump(l.size() as i64);
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            if ON.load(Relaxed) {
                bump(-(l.size() as i64));
            }
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            let q = unsafe { System.realloc(p, l, new) };
            if !q.is_null() && ON.load(Relaxed) {
                bump(new as i64 - l.size() as i64);
            }
            q
        }
    }
    pub fn start() {
        LIVE.store(0, Relaxed);
        PEAK.store(0, Relaxed);
        ON.store(true, Relaxed);
    }
    pub fn live() -> i64 {
        LIVE.load(Relaxed)
    }
    pub fn peak() -> i64 {
        PEAK.load(Relaxed)
    }
    pub fn stop() {
        ON.store(false, Relaxed);
    }
}
