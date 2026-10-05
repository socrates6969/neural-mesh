//! H7 policies: the W4 nbucb / nbreg learners re-implemented (same arithmetic as g4 pols.rs NbUcb/NbReg) with ADDITIONAL gate modes
//! designed to answer W4's V2 failure ("novelty is not purchasable, the observational gate cannot tell"):
//!   RX{eps}  : counterfactual gate. With prob eps a task is an EXPERIMENT: uniformly random agent, bonus off. On experiment tasks only we regress the
//!              reward on x = nov_est(agent) - mean_cands nov_est (pre-selection prediction => randomized => unconfounded). The bonus is ON iff the
//!              slope is significantly positive (total causal effect of buying predicted novelty on reward, quality cost included).
//!   AB{eps}  : periodic gate-off exploration. Both arms (bonus on / off) are kept as two-armed bandit per domain: with prob eps a random arm is forced,
//!              otherwise the arm with the higher (discounted) mean value is used; bonus on iff mean_on > mean_off (both need >= 8 samples).
//!   Tag      : declared creative domains.   Learned: W4's observational gate (copied through g4 Gate).   Never: layer disabled.
//! Constants are fixed a priori (eps .10, t_on 2.0, t_off .5, b_on .10, discount .995); NOT tuned.
use super::pols::{dom_of, Div, Gate, GateKind};
use super::world::{Pol, View};
use neural_mesh::router::{Rng, Task};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum XG {
    Learned,
    Tag,
    RX(f64),
    AB(f64),
    /// disagreement experiments: only on tasks where the bonus-on pick differs from the bonus-off pick; with prob q (always until 6 samples/arm)
    /// the pick is randomised 50/50 between the two and the realised reward is credited to that arm; otherwise follow the arm with the higher
    /// (discounted .99) mean reward. Cost of the experiment ~ q * half the harm; power concentrated on the tasks where the layer matters.
    DX(f64),
    /// DX with an evidence requirement: arm 'on' is used only if mean_on - mean_off > 0 with Welch t >= 2 (default OFF)
    DXT(f64),
    Never,
}
#[derive(Clone, Debug)]
pub struct XSpec {
    /// false: nbucb (additive bonus beta*nov_est); true: nbreg (regression-calibrated)
    pub reg: bool,
    pub beta: f64,
    pub c: f64,
    pub gate: XG,
}
impl XSpec {
    pub fn label(&self) -> String {
        let g = match self.gate {
            XG::Learned => "learned".to_string(),
            XG::Tag => "declared".to_string(),
            XG::RX(e) => format!("RX{e}"),
            XG::AB(e) => format!("AB{e}"),
            XG::DX(e) => format!("DX{e}"),
            XG::DXT(e) => format!("DXT{e}"),
            XG::Never => "off".to_string(),
        };
        if self.reg { format!("nbreg(c={})|{}", self.c, g) } else { format!("nbucb(b={},c={})|{}", self.beta, self.c, g) }
    }
}

#[derive(Clone, Default)]
struct Reg {
    n: f64,
    sx: f64,
    sy: f64,
    sxx: f64,
    sxy: f64,
    syy: f64,
    on: bool,
}
impl Reg {
    fn add(&mut self, x: f64, y: f64) {
        let g = 0.995;
        self.n *= g;
        self.sx *= g;
        self.sy *= g;
        self.sxx *= g;
        self.sxy *= g;
        self.syy *= g;
        self.n += 1.0;
        self.sx += x;
        self.sy += y;
        self.sxx += x * x;
        self.sxy += x * y;
        self.syy += y * y;
        if self.n >= 12.0 {
            let (mx, my) = (self.sx / self.n, self.sy / self.n);
            let vx = self.sxx / self.n - mx * mx;
            if vx > 1e-9 {
                let cov = self.sxy / self.n - mx * my;
                let b = cov / vx;
                let vy = (self.syy / self.n - my * my).max(0.0);
                let resid = ((vy - b * cov) * self.n / (self.n - 2.0)).max(1e-12);
                let t = b / (resid / (vx * self.n)).sqrt();
                if !self.on && t > 2.0 && b > 0.10 {
                    self.on = true;
                } else if self.on && t < 0.5 {
                    self.on = false;
                }
            }
        }
    }
}
#[derive(Clone, Default)]
struct Ab {
    n: [f64; 2], // [off, on]
    s: [f64; 2],
    ss: [f64; 2],
}

#[derive(Clone, Default)]
struct Pend {
    rx: bool,
    x: f64,
    arm_on: bool,
    dx: Option<bool>,
}

pub struct NbX {
    spec: XSpec,
    gamma: f64,
    s: Vec<Vec<f64>>,
    n: Vec<Vec<f64>>,
    nd_tot: Vec<f64>,
    div: Div,
    lg: Gate,
    rx: Vec<Reg>,
    ab: Vec<Ab>,
    dx: Vec<Ab>,
    pend: Pend,
    truth: Vec<bool>,
    declared: Vec<bool>,
    dg: [f64; 4],
    name: String,
}
impl NbX {
    pub fn new(spec: &XSpec, nd: usize, na: usize, declared: &[bool], truth: &[bool]) -> Self {
        NbX {
            spec: spec.clone(),
            gamma: 0.97,
            s: vec![vec![0.0; na]; nd],
            n: vec![vec![0.0; na]; nd],
            nd_tot: vec![0.0; nd],
            div: Div::new(nd, na),
            lg: Gate::new(GateKind::Learned, declared, na),
            rx: vec![Reg::default(); nd],
            ab: vec![Ab::default(); nd],
            dx: vec![Ab::default(); nd],
            pend: Pend::default(),
            truth: truth.to_vec(),
            declared: declared.to_vec(),
            dg: [0.0; 4],
            name: spec.label(),
        }
    }
    fn dx_state(&self, d: usize) -> bool {
        let a = &self.dx[d];
        if !(a.n[0] >= 6.0 && a.n[1] >= 6.0) {
            return false;
        }
        let (m0, m1) = (a.s[0] / a.n[0], a.s[1] / a.n[1]);
        if let XG::DXT(_) = self.spec.gate {
            let v0 = (a.ss[0] / a.n[0] - m0 * m0).max(1e-6) / a.n[0];
            let v1 = (a.ss[1] / a.n[1] - m1 * m1).max(1e-6) / a.n[1];
            (m1 - m0) / (v0 + v1).sqrt() >= 2.0
        } else {
            m1 > m0
        }
    }
    fn score_pick(&self, d: usize, cand: &[usize], act: bool, noise: &[f64]) -> usize {
        let slope = if act && self.spec.reg { self.lg.slope(d).clamp(0.0, 1.5) } else { 0.0 };
        let lnn = (self.nd_tot[d] + 2.0).ln();
        let mut best = cand[0];
        let mut bv = f64::NEG_INFINITY;
        for (i, &a) in cand.iter().enumerate() {
            let v = if self.spec.reg {
                let (mr, mn, cnt) = self.lg.agent_means(d, a);
                let mean = if cnt > 0.5 { mr + slope * (self.div.nov_est(d, a) - mn) } else { 0.9 + slope };
                mean + self.spec.c * (lnn / (self.n[d][a] + 1.0)).sqrt() + 1e-9 * noise[i]
            } else {
                let n = self.n[d][a];
                let mean = if n > 1e-9 { self.s[d][a] / n } else { 0.9 };
                let bonus = if act { self.spec.beta * self.div.nov_est(d, a) } else { 0.0 };
                mean + self.spec.c * (lnn / (n + 1.0)).sqrt() + bonus + 1e-9 * noise[i]
            };
            if v > bv {
                bv = v;
                best = a;
            }
        }
        best
    }
    fn nov_adv(&self, d: usize, a: usize, cand: &[usize]) -> f64 {
        let m = cand.iter().map(|&c| self.div.nov_est(d, c)).sum::<f64>() / cand.len() as f64;
        self.div.nov_est(d, a) - m
    }
}
impl Pol for NbX {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn select(&mut self, task: &Task, cand: &[usize], rng: &mut Rng, _v: &View) -> usize {
        let d = dom_of(task);
        let (mut act, mut experiment) = (false, false);
        match self.spec.gate {
            XG::Learned => act = self.lg.learned_on(d),
            XG::Tag => act = self.declared[d],
            XG::Never | XG::DX(_) | XG::DXT(_) => {}
            XG::RX(eps) => {
                if rng.f64() < eps {
                    experiment = true;
                } else {
                    act = self.rx[d].on;
                }
            }
            XG::AB(eps) => {
                let ab = &self.ab[d];
                let greedy = if ab.n[0] >= 8.0 && ab.n[1] >= 8.0 { ab.s[1] / ab.n[1] > ab.s[0] / ab.n[0] } else { rng.f64() < 0.5 };
                act = if rng.f64() < eps { rng.f64() < 0.5 } else { greedy };
            }
        }
        let mut dx_arm: Option<bool> = None;
        let mut dx_chosen: Option<usize> = None;
        let dxq = match self.spec.gate { XG::DX(q) | XG::DXT(q) => Some(q), _ => None };
        if let Some(qe) = dxq {
            let noise: Vec<f64> = cand.iter().map(|_| rng.f64()).collect();
            let off = self.score_pick(d, cand, false, &noise);
            let on = self.score_pick(d, cand, true, &noise);
            if on == off {
                dx_chosen = Some(on);
                act = self.dx_state(d);
            } else {
                let have = self.dx[d].n[0] >= 6.0 && self.dx[d].n[1] >= 6.0;
                let explore = !have || rng.f64() < qe;
                let arm_on = if explore { rng.f64() < 0.5 } else { self.dx_state(d) };
                if explore {
                    dx_arm = Some(arm_on);
                }
                act = arm_on;
                dx_chosen = Some(if arm_on { on } else { off });
            }
        }
        let i = if self.truth[d] { 0 } else { 2 };
        self.dg[i + 1] += 1.0;
        if act {
            self.dg[i] += 1.0;
        }
        let chosen = if let Some(c) = dx_chosen {
            c
        } else if experiment {
            cand[rng.below(cand.len())]
        } else {
            let noise: Vec<f64> = cand.iter().map(|_| rng.f64()).collect();
            self.score_pick(d, cand, act, &noise)
        };
        self.pend = Pend { rx: experiment, x: self.nov_adv(d, chosen, cand), arm_on: act, dx: dx_arm };
        chosen
    }
    fn observe(&mut self, task: &Task, agent: usize, reward: f64, out: &[f64], _v: &View) {
        let d = dom_of(task);
        let nov_obs = self.div.nov_point(d, out);
        self.lg.update(d, agent, reward, nov_obs);
        if let XG::RX(_) = self.spec.gate {
            if self.pend.rx {
                let x = self.pend.x;
                self.rx[d].add(x, reward);
            }
        }
        if let (Some(_), Some(k)) = (match self.spec.gate { XG::DX(q) | XG::DXT(q) => Some(q), _ => None }, self.pend.dx) {
            let k = k as usize;
            for j in 0..2 {
                self.dx[d].n[j] *= 0.99;
                self.dx[d].s[j] *= 0.99;
                self.dx[d].ss[j] *= 0.99;
            }
            self.dx[d].n[k] += 1.0;
            self.dx[d].s[k] += reward;
            self.dx[d].ss[k] += reward * reward;
        }
        if let XG::AB(_) = self.spec.gate {
            let k = self.pend.arm_on as usize;
            let g = 0.995;
            for j in 0..2 {
                self.ab[d].n[j] *= g;
                self.ab[d].s[j] *= g;
            }
            self.ab[d].n[k] += 1.0;
            self.ab[d].s[k] += reward;
        }
        self.div.observe(d, agent, out);
        for a in 0..self.n[d].len() {
            self.s[d][a] *= self.gamma;
            self.n[d][a] *= self.gamma;
        }
        self.s[d][agent] += reward;
        self.n[d][agent] += 1.0;
        self.nd_tot[d] += 1.0;
    }
    fn diag(&self) -> Vec<f64> {
        let f = |a: f64, b: f64| if b > 0.0 { a / b } else { f64::NAN };
        vec![f(self.dg[0], self.dg[1]), f(self.dg[2], self.dg[3])]
    }
}
