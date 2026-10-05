//! Native spiking pieces (no external SNN crate): LIF / Izhikevich neurons run as a
//! winner-take-all selector with lateral inhibition, population coding of
//! continuous features, and a three-factor learner with eligibility traces.
//!
//! Conventions follow the public Izhikevich (2003) model (two dt/2 Euler sub-steps
//! for v, then u); the oxicuda-snn / neuromod sources were only READ for naming and
//! parameter conventions, nothing is built from or copied out of them.

use super::worlds::NA;
use neural_mesh::router::{Rng, Task};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NeuronKind {
    /// v <- leak*v + I (+noise - inhibition); spike at v >= 1, reset 0.
    Lif { leak: f64 },
    /// Izhikevich (a,b,c,d), dt=1ms, spike at v>=30.
    Izh { a: f64, b: f64, c: f64, d: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WtaCfg {
    pub kind: NeuronKind,
    pub steps: usize,
    /// expected total number of input spikes per step (sum_j p_j)
    pub lambda_in: f64,
    /// drive scale: mean input current = gain * yhat
    pub gain: f64,
    /// lateral inhibition: each spike subtracts `inh` from every other candidate neuron
    pub inh: f64,
    /// half-range scale of uniform membrane/current noise
    pub noise: f64,
    /// stop at the first step in which any neuron spikes (latency code)
    pub first_spike: bool,
}

/// Winner-take-all selector over candidate-agent neurons.
pub struct Wta {
    v: [f64; NA],
    u: [f64; NA],
    cnt: [u32; NA],
    ft: [u32; NA],
    act: Vec<usize>,
    /// decisions, decisions with >=1 spike, total spikes (diagnostics for the honesty report)
    pub n_dec: u64,
    pub n_dec_spk: u64,
    pub n_spk: u64,
}

impl Default for Wta {
    fn default() -> Self {
        Wta { v: [0.0; NA], u: [0.0; NA], cnt: [0; NA], ft: [0; NA], act: Vec::with_capacity(32), n_dec: 0, n_dec_spk: 0, n_spk: 0 }
    }
}

impl Wta {
    /// `w` is the NA x ni weight matrix (row = agent), `inputs` the normalised rates.
    pub fn run(&mut self, c: &WtaCfg, w: &[f64], ni: usize, inputs: &[(usize, f64)], cand: &[usize], rng: &mut Rng) -> usize {
        let (v0, u0) = match c.kind {
            NeuronKind::Lif { .. } => (0.0, 0.0),
            NeuronKind::Izh { b, c: cc, .. } => (cc, b * cc),
        };
        for &a in cand {
            self.v[a] = v0;
            self.u[a] = u0;
            self.cnt[a] = 0;
            self.ft[a] = u32::MAX;
        }
        let scale = c.gain / c.lambda_in.max(1e-9);
        let mut best_pre = [f64::NEG_INFINITY; NA];
        for t in 0..c.steps {
            // Poisson-like input spikes this step
            self.act.clear();
            for &(j, r) in inputs {
                let p = (c.lambda_in * r).min(1.0);
                if rng.f64() < p {
                    self.act.push(j);
                }
            }
            let mut spiked = [false; NA];
            let mut any = false;
            for &a in cand {
                let mut drive = 0.0;
                let row = &w[a * ni..(a + 1) * ni];
                for &j in &self.act {
                    drive += row[j];
                }
                let noise = c.noise * (rng.f64() - 0.5) * 3.4641016;
                let i_in = scale * drive + noise;
                match c.kind {
                    NeuronKind::Lif { leak } => {
                        let v = leak * self.v[a] + i_in;
                        if v >= 1.0 {
                            best_pre[a] = v;
                            spiked[a] = true;
                            self.v[a] = 0.0;
                        } else {
                            self.v[a] = v.max(-1.0);
                        }
                    }
                    NeuronKind::Izh { a: pa, b, c: cc, d } => {
                        let mut v = self.v[a];
                        let mut u = self.u[a];
                        v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + i_in);
                        v += 0.5 * (0.04 * v * v + 5.0 * v + 140.0 - u + i_in);
                        u += pa * (b * v - u);
                        if v >= 30.0 {
                            best_pre[a] = v;
                            spiked[a] = true;
                            v = cc;
                            u += d;
                        }
                        self.v[a] = v.max(-90.0);
                        self.u[a] = u;
                    }
                }
                if spiked[a] {
                    self.cnt[a] += 1;
                    if self.ft[a] == u32::MAX {
                        self.ft[a] = t as u32;
                    }
                    any = true;
                }
            }
            if any && c.inh > 0.0 {
                // lateral inhibition: every spike hits all the OTHER candidate neurons
                let ns = cand.iter().filter(|&&a| spiked[a]).count();
                for &a in cand {
                    let from_others = ns - usize::from(spiked[a]);
                    if from_others > 0 {
                        let floor = match c.kind {
                            NeuronKind::Lif { .. } => -1.0,
                            NeuronKind::Izh { .. } => -90.0,
                        };
                        self.v[a] = (self.v[a] - c.inh * from_others as f64).max(floor);
                    }
                }
            }
            if any && c.first_spike {
                break;
            }
        }
        let tot: u32 = cand.iter().map(|&a| self.cnt[a]).sum();
        self.n_dec += 1;
        self.n_spk += tot as u64;
        if tot > 0 {
            self.n_dec_spk += 1;
        }
        // winner: most spikes, then earliest first spike, then highest potential / pre-spike v, then random
        let mut best = cand[0];
        let mut bk = f64::NEG_INFINITY;
        for &a in cand {
            let ftv = if self.ft[a] == u32::MAX { 1e3 } else { self.ft[a] as f64 };
            let vv = if self.cnt[a] > 0 { best_pre[a] } else { self.v[a] };
            let key = self.cnt[a] as f64 * 1e6 - ftv * 1e3 + vv.clamp(-500.0, 500.0) + 1e-6 * rng.f64();
            if key > bk {
                bk = key;
                best = a;
            }
        }
        best
    }
}

// ---------------------------------------------------------------- population code

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pop {
    None,
    /// m Gaussian tuning curves per dimension (separate 1-D codes, additive)
    Sep { m: usize, sigma: f64 },
    /// g x g grid of 2-D Gaussian tuning curves
    Grid { g: usize, sigma: f64 },
}
impl Pop {
    pub fn n(&self) -> usize {
        match *self {
            Pop::None => 0,
            Pop::Sep { m, .. } => 2 * m,
            Pop::Grid { g, .. } => g * g,
        }
    }
    pub fn encode(&self, x: [f64; 2], out: &mut Vec<(usize, f64)>) {
        match *self {
            Pop::None => {}
            Pop::Sep { m, sigma } => {
                for d in 0..2 {
                    for k in 0..m {
                        let mu = k as f64 / (m as f64 - 1.0);
                        let r = (-(x[d] - mu).powi(2) / (2.0 * sigma * sigma)).exp();
                        if r > 0.02 {
                            out.push((d * m + k, r));
                        }
                    }
                }
            }
            Pop::Grid { g, sigma } => {
                for kx in 0..g {
                    for ky in 0..g {
                        let mx = kx as f64 / (g as f64 - 1.0);
                        let my = ky as f64 / (g as f64 - 1.0);
                        let r = (-((x[0] - mx).powi(2) + (x[1] - my).powi(2)) / (2.0 * sigma * sigma)).exp();
                        if r > 0.02 {
                            out.push((kx * g + ky, r));
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------- learner

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sel {
    /// non-spiking control: softmax over the same yhat, temperature tau
    Rate { tau: f64 },
    Wta(WtaCfg),
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Tok {
    None,
    /// only tokens that are keys of the keyword map
    Kw,
    All,
}
/// credit assignment for ANONYMOUS delayed reward
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Credit {
    /// eligibility trace with per-decision decay lambda
    Trace(f64),
    /// attribute to the decision exactly L steps ago
    Lag(usize),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LCfg {
    pub sel: Sel,
    pub pop: Pop,
    pub tok: Tok,
    pub eta: f64,
    pub w0: f64,
    pub kwb: f64,
    pub eps: f64,
    pub credit: Credit,
}

pub const NT: usize = 96; // token neuron slots
const BIAS_RATE: f64 = 0.1;

struct Snap {
    agent: usize,
    inp: Vec<(usize, f64)>,
}

pub struct Learner {
    pub cfg: LCfg,
    np: usize,
    pub ni: usize,
    w: Vec<f64>,
    e: Vec<f64>,
    tok: HashMap<String, usize>,
    kmap: HashMap<String, usize>,
    snaps: Vec<Snap>,
    wta: Wta,
    scratch: Vec<(usize, f64)>,
}

impl Learner {
    pub fn new(cfg: LCfg, kmap: HashMap<String, usize>) -> Self {
        let np = cfg.pop.n();
        let ni = np + NT + 1;
        let w = vec![cfg.w0; NA * ni];
        Learner {
            cfg,
            np,
            ni,
            w,
            e: if matches!(cfg.credit, Credit::Trace(_)) { vec![0.0; NA * ni] } else { Vec::new() },
            tok: HashMap::new(),
            kmap,
            snaps: Vec::new(),
            wta: Wta::default(),
            scratch: Vec::new(),
        }
    }

    /// (agent neurons, input neurons incl. the token slots actually used and the bias unit)
    pub fn n_neurons(&self) -> (usize, usize) {
        (NA, self.np + self.tok.len() + 1)
    }
    /// (decisions, decisions with >=1 spike, total spikes) of the WTA selector
    pub fn spike_stats(&self) -> (u64, u64, u64) {
        (self.wta.n_dec, self.wta.n_dec_spk, self.wta.n_spk)
    }
    /// bytes of learned/online state actually held (weights + trace + token table)
    pub fn state_bytes(&self) -> usize {
        8 * (self.w.len() + self.e.len()) + self.tok.keys().map(|k| k.len() + 16).sum::<usize>()
    }

    fn slot(&mut self, f: &str) -> Option<usize> {
        if let Some(&s) = self.tok.get(f) {
            return Some(s);
        }
        let kw = self.kmap.get(f).copied();
        match self.cfg.tok {
            Tok::None => return None,
            Tok::Kw if kw.is_none() => return None,
            _ => {}
        }
        if self.tok.len() >= NT {
            return None;
        }
        let s = self.tok.len();
        self.tok.insert(f.to_string(), s);
        if let Some(a) = kw {
            if self.cfg.kwb > 0.0 {
                let j = self.np + s;
                let ni = self.ni;
                self.w[a * ni + j] = (self.w[a * ni + j] + self.cfg.kwb).min(1.0);
            }
        }
        Some(s)
    }

    fn encode(&mut self, task: &Task, x: [f64; 2]) -> Vec<(usize, f64)> {
        let mut out = std::mem::take(&mut self.scratch);
        out.clear();
        self.cfg.pop.encode(x, &mut out);
        if self.cfg.tok != Tok::None {
            for f in &task.features {
                if let Some(s) = self.slot(f) {
                    let j = self.np + s;
                    if !out.iter().any(|&(k, _)| k == j) {
                        out.push((j, 1.0));
                    }
                }
            }
        }
        out.push((self.np + NT, BIAS_RATE));
        let s: f64 = out.iter().map(|p| p.1).sum();
        for p in out.iter_mut() {
            p.1 /= s;
        }
        out
    }

    fn yhat(&self, a: usize, inp: &[(usize, f64)]) -> f64 {
        let row = &self.w[a * self.ni..(a + 1) * self.ni];
        inp.iter().map(|&(j, r)| row[j] * r).sum()
    }

    pub fn decide(&mut self, task: &Task, x: [f64; 2], cand: &[usize], rng: &mut Rng) -> usize {
        let inp = self.encode(task, x);
        let a = if self.cfg.eps > 0.0 && rng.f64() < self.cfg.eps {
            cand[rng.below(cand.len())]
        } else {
            match self.cfg.sel {
                Sel::Rate { tau } => {
                    let mut m = f64::NEG_INFINITY;
                    let ys: Vec<f64> = cand.iter().map(|&a| self.yhat(a, &inp)).collect();
                    for &y in &ys {
                        m = m.max(y);
                    }
                    let ws: Vec<f64> = ys.iter().map(|&y| ((y - m) / tau).exp()).collect();
                    let tot: f64 = ws.iter().sum();
                    let mut u = rng.f64() * tot;
                    let mut pick = cand[cand.len() - 1];
                    for (k, &wk) in ws.iter().enumerate() {
                        if u < wk {
                            pick = cand[k];
                            break;
                        }
                        u -= wk;
                    }
                    pick
                }
                Sel::Wta(c) => self.wta.run(&c, &self.w, self.ni, &inp, cand, rng),
            }
        };
        if let Credit::Trace(lam) = self.cfg.credit {
            for v in self.e.iter_mut() {
                *v *= lam;
            }
            let ni = self.ni;
            for &(j, r) in &inp {
                self.e[a * ni + j] += r;
            }
        }
        self.snaps.push(Snap { agent: a, inp: inp.clone() });
        self.scratch = inp;
        a
    }

    fn apply_snap(&mut self, idx: usize, r: f64) {
        let (a, inp) = {
            let s = &self.snaps[idx];
            (s.agent, s.inp.clone())
        };
        let y = self.yhat(a, &inp);
        let d = r - y;
        let n2: f64 = inp.iter().map(|p| p.1 * p.1).sum::<f64>().max(1e-9);
        let ni = self.ni;
        for &(j, rj) in &inp {
            let k = a * ni + j;
            self.w[k] = (self.w[k] + self.cfg.eta * d * rj / n2).clamp(0.0, 1.0);
        }
    }

    /// reward with a known decision id (immediate or delayed)
    pub fn reward_id(&mut self, idx: usize, r: f64) {
        self.apply_snap(idx, r);
    }

    /// anonymous reward arriving at step `now` (decisions 0..now-1 exist)
    pub fn reward_anon(&mut self, now: usize, r: f64) {
        match self.cfg.credit {
            Credit::Lag(l) => {
                if now >= l && now - l < self.snaps.len() {
                    self.apply_snap(now - l, r);
                }
            }
            Credit::Trace(_) => {
                let ni = self.ni;
                for a in 0..NA {
                    let m: f64 = self.e[a * ni..(a + 1) * ni].iter().sum();
                    if m < 1e-6 {
                        continue;
                    }
                    let mut y = 0.0;
                    let mut n2 = 0.0;
                    for j in 0..ni {
                        let eb = self.e[a * ni + j] / m;
                        y += self.w[a * ni + j] * eb;
                        n2 += eb * eb;
                    }
                    let d = r - y;
                    let c = m.min(1.0);
                    for j in 0..ni {
                        let eb = self.e[a * ni + j] / m;
                        if eb > 0.0 {
                            let k = a * ni + j;
                            self.w[k] = (self.w[k] + self.cfg.eta * c * d * eb / n2.max(1e-9)).clamp(0.0, 1.0);
                        }
                    }
                }
            }
        }
    }
}
