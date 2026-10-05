//! H12 opt-in ONLINE ADAPTATION layers (std-only, f32/f64, no new crates). SIMULATION framing: these only change
//! which of a fixed set of agents gets a task; they do not raise a frozen model's reasoning, and none of this is
//! weight learning of the model.
//!
//! 1. `ResidualEns`: wraps any incumbent `Router`. The incumbent decides first (consuming the shared rng exactly as
//!    it would alone). A tiny tanh net (output layer initialised to ZERO, on top of a per-agent global base rate)
//!    learns a residual correction to per-agent success estimates from graded outcomes. It may override the
//!    incumbent only if (a) >= `n_min` observations seen, (b) for the candidate agent the net's PREQUENTIAL log-loss
//!    beats a token-additive discounted predictor (the incumbent-like baseline) by `delta` on a rolling EMA
//!    (causal held-in validation, per agent), and (c) its predicted success exceeds the incumbent pick's by `margin`.
//!    With `e0 = 0` and gates closed the layer is bit-for-bit the incumbent (unit-tested).
//! 2. `Portfolio`: a discounted-Thompson bandit that picks, per task, WHICH member router acts. Every member learns from
//!    every graded outcome (shared data). With members = warm(hebr) under different hyper-parameters it is the
//!    "online hyper-parameter adaptation by bandit-over-configs"; with members = {keyword, warm(hebr), guarded-safe,
//!    residual} it is the meta-selector. `agree=true` credits every member whose own pick equals the executed action.

use std::collections::HashMap;
use std::sync::Arc;

use neural_mesh::baselines::KeywordRouter;
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};

use super::{fnv, mk_router, Env, Kind, MAXI};

pub const RH: usize = 512; // hashed token slots of the layer (net + additive predictor)
const MAXH: usize = 32;
const MAXN: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct ResCfg {
    pub h: usize,
    pub lr: f32,
    pub n_min: usize,
    pub margin: f32,
    /// layer-level exploration: eps_t = e0 * 0.5^(t / e_half), uniform over candidates (counted as real cost)
    pub e0: f64,
    pub e_half: f64,
    /// gate: EMA(logloss_additive - logloss_net) must exceed delta
    pub delta: f32,
    pub beta: f32,
    pub replay_k: usize,
    pub gamma_add: f32,
    /// ablation: never close the gate once n_min is reached
    pub nogate: bool,
}

impl ResCfg {
    pub fn default_cfg() -> Self {
        ResCfg { h: 24, lr: 0.05, n_min: 100, margin: 0.03, e0: 0.05, e_half: 1000.0, delta: 0.005, beta: 0.02, replay_k: 3, gamma_add: 0.98, nogate: false }
    }
    pub fn show(&self) -> String {
        format!("res H{} lr{} nmin{} m{} e0={} {}", self.h, self.lr, self.n_min, self.margin, self.e0, if self.nogate { "NOGATE" } else { "gated" })
    }
}

#[derive(Clone, Copy)]
struct Samp {
    idx: [u16; MAXI],
    ni: u8,
    dense: [f32; MAXI],
    nd: u8,
    a: u8,
    r: f32,
}

fn sig(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-4, 1.0 - 1e-4);
    (p / (1.0 - p)).ln()
}
fn bce(p: f32, r: f32) -> f32 {
    let p = p.clamp(1e-4, 1.0 - 1e-4);
    -(r * p.ln() + (1.0 - r) * (1.0 - p).ln())
}

pub struct ResidualEns {
    inner: Box<dyn Router>,
    pub cfg: ResCfg,
    n: usize,
    dd: usize,
    dense: Option<Arc<Vec<Vec<f32>>>>,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    bs: Vec<f64>,
    bn: Vec<f64>,
    add: Vec<f32>, // [RH][n][2] discounted (successes, failures)
    gate: Vec<f32>,
    nobs: Vec<u32>,
    t: usize,
    rng: Rng,
    buf: Vec<Samp>,
    pos: usize,
    pub overrides: u64,
    pub explores: u64,
}

impl ResidualEns {
    pub fn new(inner: Box<dyn Router>, n_agents: usize, cfg: ResCfg, dense: Option<Arc<Vec<Vec<f32>>>>, seed: u64) -> Self {
        assert!(cfg.h <= MAXH && n_agents <= MAXN);
        let dd = dense.as_ref().map(|d| d[0].len().min(MAXI)).unwrap_or(0);
        let mut rng = Rng::new(seed ^ 0x12_12_AB);
        let rows = RH + dd;
        let w1 = (0..rows * cfg.h).map(|_| (rng.f64() * 1.0 - 0.5) as f32).collect();
        ResidualEns {
            inner,
            cfg,
            n: n_agents,
            dd,
            dense,
            w1,
            b1: vec![0.0; cfg.h],
            w2: vec![0.0; n_agents * cfg.h],
            b2: vec![0.0; n_agents],
            bs: vec![0.0; n_agents],
            bn: vec![0.0; n_agents],
            add: vec![0.0; RH * n_agents * 2],
            gate: vec![0.0; n_agents],
            nobs: vec![0; n_agents],
            t: 0,
            rng,
            buf: vec![],
            pos: 0,
            overrides: 0,
            explores: 0,
        }
    }
    /// bytes of the layer's own state (params f32 + additive table + replay ring at current fill + small vecs)
    pub fn layer_bytes(&self) -> usize {
        (self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len() + self.add.len() + self.gate.len()) * 4
            + (self.bs.len() + self.bn.len()) * 8
            + self.nobs.len() * 4
            + self.buf.len() * std::mem::size_of::<Samp>()
    }
    fn sample(&self, task: &Task, a: usize, r: f64) -> Samp {
        let mut s = Samp { idx: [0; MAXI], ni: 0, dense: [0.0; MAXI], nd: 0, a: a as u8, r: r as f32 };
        for f in &task.features {
            let i = (fnv(f) % RH as u64) as u16;
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
    fn forward(&self, s: &Samp, hid: &mut [f32; MAXH]) {
        let h = self.cfg.h;
        hid[..h].copy_from_slice(&self.b1);
        for &i in &s.idx[..s.ni as usize] {
            let row = &self.w1[i as usize * h..(i as usize + 1) * h];
            for k in 0..h {
                hid[k] += row[k];
            }
        }
        for j in 0..s.nd as usize {
            let x = s.dense[j];
            let row = &self.w1[(RH + j) * h..(RH + j + 1) * h];
            for k in 0..h {
                hid[k] += row[k] * x;
            }
        }
        for k in 0..h {
            hid[k] = hid[k].tanh();
        }
    }
    fn base(&self, a: usize) -> f32 {
        ((self.bs[a] + 1.0) / (self.bn[a] + 2.0)) as f32
    }
    fn p_net(&self, hid: &[f32; MAXH], a: usize) -> f32 {
        let h = self.cfg.h;
        let row = &self.w2[a * h..(a + 1) * h];
        let mut acc = self.b2[a];
        for k in 0..h {
            acc += row[k] * hid[k];
        }
        sig(logit(self.base(a)) + acc)
    }
    fn p_add(&self, s: &Samp, a: usize) -> f32 {
        if s.ni == 0 {
            return self.base(a);
        }
        let mut m = 0.0f32;
        for &i in &s.idx[..s.ni as usize] {
            let o = (i as usize * self.n + a) * 2;
            m += (self.add[o] + 1.0) / (self.add[o] + self.add[o + 1] + 2.0);
        }
        m / s.ni as f32
    }
    fn open(&self, a: usize) -> bool {
        self.t >= self.cfg.n_min && self.nobs[a] >= 30 && (self.cfg.nogate || self.gate[a] > self.cfg.delta)
    }
    fn sgd(&mut self, s: &Samp) {
        let h = self.cfg.h;
        let lr = self.cfg.lr;
        let mut hid = [0.0f32; MAXH];
        self.forward(s, &mut hid);
        let a = s.a as usize;
        let p = self.p_net(&hid, a);
        let dz = p - s.r;
        let mut dpre = [0.0f32; MAXH];
        {
            let row = &self.w2[a * h..(a + 1) * h];
            for k in 0..h {
                dpre[k] = dz * row[k] * (1.0 - hid[k] * hid[k]);
            }
        }
        {
            let row = &mut self.w2[a * h..(a + 1) * h];
            for k in 0..h {
                row[k] -= lr * dz * hid[k];
            }
        }
        self.b2[a] -= lr * dz;
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
            let row = &mut self.w1[(RH + j) * h..(RH + j + 1) * h];
            for k in 0..h {
                row[k] -= lr * dpre[k] * x;
            }
        }
    }
}

impl Router for ResidualEns {
    fn name(&self) -> String {
        format!("resid[{}]", self.cfg.show())
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let a_inc = self.inner.select(task, candidates, rng);
        let eps = self.cfg.e0 * 0.5f64.powf(self.t as f64 / self.cfg.e_half);
        if eps > 1e-6 && self.rng.f64() < eps {
            self.explores += 1;
            return candidates[self.rng.below(candidates.len())];
        }
        if self.t < self.cfg.n_min {
            return a_inc;
        }
        if !candidates.iter().any(|&c| self.open(c)) {
            return a_inc;
        }
        let s = self.sample(task, 0, 0.0);
        let mut hid = [0.0f32; MAXH];
        self.forward(&s, &mut hid);
        let p_inc = if self.open(a_inc) { self.p_net(&hid, a_inc) } else { self.p_add(&s, a_inc) };
        let mut best = (f32::MIN, a_inc);
        for &c in candidates {
            if c != a_inc && self.open(c) {
                let p = self.p_net(&hid, c);
                if p > best.0 {
                    best = (p, c);
                }
            }
        }
        if best.1 != a_inc && best.0 > p_inc + self.cfg.margin {
            self.overrides += 1;
            return best.1;
        }
        a_inc
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.inner.update(task, agent, reward);
        let r = reward.clamp(0.0, 1.0);
        let s = self.sample(task, agent, r);
        // prequential gate (predict before learning)
        let mut hid = [0.0f32; MAXH];
        self.forward(&s, &mut hid);
        let ln = bce(self.p_net(&hid, agent), s.r);
        let la = bce(self.p_add(&s, agent), s.r);
        self.gate[agent] = (1.0 - self.cfg.beta) * self.gate[agent] + self.cfg.beta * (la - ln);
        self.nobs[agent] += 1;
        // learn
        self.sgd(&s);
        if self.cfg.replay_k > 0 {
            if self.buf.len() < 1000 {
                self.buf.push(s);
            } else {
                self.buf[self.pos] = s;
                self.pos = (self.pos + 1) % 1000;
            }
            for _ in 0..self.cfg.replay_k {
                let j = self.rng.below(self.buf.len());
                let sj = self.buf[j];
                self.sgd(&sj);
            }
        }
        for &i in &s.idx[..s.ni as usize] {
            let o = i as usize * self.n * 2;
            if self.cfg.gamma_add < 1.0 {
                for v in &mut self.add[o..o + self.n * 2] {
                    *v *= self.cfg.gamma_add;
                }
            }
            self.add[o + agent * 2] += s.r;
            self.add[o + agent * 2 + 1] += 1.0 - s.r;
        }
        self.bs[agent] += r;
        self.bn[agent] += 1.0;
        self.t += 1;
    }
}

// ---------------------------------------------------------------------------------------------

fn randn(rng: &mut Rng) -> f64 {
    let u1 = rng.f64().max(1e-12);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

pub struct Portfolio {
    label: String,
    members: Vec<Box<dyn Router>>,
    mrng: Vec<Rng>,
    s: Vec<f64>,
    f: Vec<f64>,
    gamma: f64,
    agree: bool,
    brng: Rng,
    last: usize,
    picks: Vec<usize>,
    pub pulls: Vec<u64>,
}

impl Portfolio {
    pub fn new(label: &str, members: Vec<Box<dyn Router>>, gamma: f64, agree: bool, prior_idx: usize, prior_n: f64, seed: u64) -> Self {
        let k = members.len();
        let mut s = vec![0.0; k];
        let mut f = vec![0.0; k];
        if prior_n > 0.0 {
            s[prior_idx] = 0.6 * prior_n;
            f[prior_idx] = 0.4 * prior_n;
        }
        Portfolio {
            label: label.into(),
            mrng: (0..k).map(|i| Rng::new(seed ^ (0x77 + i as u64).wrapping_mul(0x9E37_79B9))).collect(),
            members,
            s,
            f,
            gamma,
            agree,
            brng: Rng::new(seed ^ 0xBA1D17),
            last: 0,
            picks: vec![0; k],
            pulls: vec![0; k],
        }
    }
}

impl Router for Portfolio {
    fn name(&self) -> String {
        format!("portfolio[{} g={} agree={}]", self.label, self.gamma, self.agree)
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let k = self.members.len();
        let mut best = (f64::MIN, 0usize);
        for i in 0..k {
            let m = (1.0 + self.s[i]) / (2.0 + self.s[i] + self.f[i]);
            let sd = (m * (1.0 - m) / (3.0 + self.s[i] + self.f[i])).sqrt();
            let th = m + sd * randn(&mut self.brng);
            if th > best.0 {
                best = (th, i);
            }
        }
        self.last = best.1;
        self.pulls[best.1] += 1;
        if self.agree {
            for i in 0..k {
                self.picks[i] = self.members[i].select(task, candidates, &mut self.mrng[i]);
            }
            self.picks[best.1]
        } else {
            self.members[best.1].select(task, candidates, rng)
        }
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        for i in 0..self.members.len() {
            self.s[i] *= self.gamma;
            self.f[i] *= self.gamma;
        }
        if self.agree {
            for i in 0..self.members.len() {
                if self.picks[i] == agent {
                    self.s[i] += r;
                    self.f[i] += 1.0 - r;
                }
            }
        } else {
            self.s[self.last] += r;
            self.f[self.last] += 1.0 - r;
        }
        for m in self.members.iter_mut() {
            m.update(task, agent, reward);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Arms

#[derive(Clone, Debug)]
pub enum Arm {
    Base(Kind),
    ResidWarm { k: f64, cfg: ResCfg },
    ResidSafe { cfg: ResCfg },
    /// bandit over warm(hebr) hyper-parameter configs (all learn from all outcomes)
    CfgBandit { k: f64, gamma: f64, agree: bool },
    /// bandit over {keyword, warm(hebr), guarded-safe, residual-on-warm}
    Meta { k: f64, rcfg: ResCfg, gamma: f64, agree: bool, prior_n: f64 },
}

/// The fixed hyper-parameter portfolio of the config bandit (fixed BEFORE any policy was run).
pub fn heb_portfolio() -> Vec<(&'static str, HebbianConfig)> {
    let b = heb_robust_cfg();
    vec![
        ("base", b.clone()),
        ("eta.05", HebbianConfig { eta: 0.05, ..b.clone() }),
        ("decay.05", HebbianConfig { decay: 0.05, ..b.clone() }),
        ("eps.05", HebbianConfig { epsilon: 0.05, ..b.clone() }),
        ("temp.02", HebbianConfig { temperature: 0.02, ..b.clone() }),
        ("slow", HebbianConfig { eta: 0.003, decay: 0.003, ..b.clone() }),
    ]
}

fn warm_with(env: &Env, n: usize, cfg: HebbianConfig, k: f64) -> Warm<HebbianRouter> {
    Warm::new(HebbianRouter::new(n, cfg), env.kw_map(), n, k).with_evidence(Evidence::MeanFeature)
}

pub fn mk_arm(arm: &Arm, env: &Env, seed: u64) -> Box<dyn Router> {
    let n = env.n_agents();
    match arm {
        Arm::Base(k) => mk_router(k, env, seed),
        Arm::ResidWarm { k, cfg } => Box::new(ResidualEns::new(mk_router(&Kind::Warm(*k), env, seed), n, *cfg, env.dense(), seed)),
        Arm::ResidSafe { cfg } => Box::new(ResidualEns::new(mk_router(&Kind::Safe, env, seed), n, *cfg, env.dense(), seed)),
        Arm::CfgBandit { k, gamma, agree } => {
            let members: Vec<Box<dyn Router>> = heb_portfolio().into_iter().map(|(_, c)| Box::new(warm_with(env, n, c, *k)) as Box<dyn Router>).collect();
            Box::new(Portfolio::new("cfg", members, *gamma, *agree, 0, 0.0, seed))
        }
        Arm::Meta { k, rcfg, gamma, agree, prior_n } => {
            let members: Vec<Box<dyn Router>> = vec![
                Box::new(KeywordRouter::new(env.kw_map(), n)),
                mk_router(&Kind::Warm(*k), env, seed),
                mk_router(&Kind::Safe, env, seed),
                Box::new(ResidualEns::new(mk_router(&Kind::Warm(*k), env, seed), n, *rcfg, env.dense(), seed)),
            ];
            Box::new(Portfolio::new("meta", members, *gamma, *agree, 2, *prior_n, seed))
        }
    }
}

#[allow(dead_code)]
pub fn unused(_: HashMap<String, usize>) {}
