//! G11 learned team model, re-implemented independently of W7's code (same idea, own implementation).
//!   value(team | task) = b0 + sum_{a in team} (g_a + mean_{f in task} H[f][a]) + [pair] sum_{a<b in team} J_ab
//! trained ONLINE by normalised LMS on the realized team reward only (it never sees the world), keyword map used
//! only as an initial prior on H (the same information the keyword/warm/guarded baselines get). The team is chosen
//! by plain GREEDY (forward add + swap polish) on the model, plus an eps swap. Features are interned strings, so
//! the model is world-agnostic (no hard-coded feature index).
//!
//! Policies:
//!   L-add   : additive only (J absent).
//!   L-pair  : additive + pairwise.
//!   L-gate  : both models are trained on every observation; the pairwise model is USED only while a prequential
//!             test says it predicts the next reward significantly better than the additive one
//!             (EW-mean of d = e_add^2 - e_pair^2, z = mean / sqrt(var / n_eff) > zthr). Otherwise additive.
//!   L-full  : L-gate + a causal controller gate: with prob q (and for the first `warm` decisions) the controller
//!             (learned team model vs sequential-select-with-team-credit) is picked UNIFORMLY AT RANDOM, and only
//!             those randomized decisions feed the per-controller reward means; otherwise the controller with the
//!             higher (decayed) randomized-trial mean is used (LRN must beat SEQ by `margin` to be chosen). Both
//!             controllers learn from every played team (shared experience).
use super::base::{SeqTeam, TeamPol};
use neural_mesh::router::{Rng, Task};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LCfg {
    /// NLMS step (1 = fit the sample exactly)
    pub lr: f64,
    pub l2: f64,
    /// eps swap exploration
    pub eps: f64,
    /// optimism on agent count / pair count
    pub c1: f64,
    pub c2: f64,
    /// keyword prior mass on H[f][lead]
    pub rho: f64,
    /// count-based step boost: step *= 1 + boost/(1 + n_cell/30) (0 = plain NLMS)
    pub boost: f64,
}

impl Default for LCfg {
    fn default() -> Self {
        LCfg { lr: 0.3, l2: 0.0005, eps: 0.05, c1: 0.0, c2: 0.0, rho: 0.1, boost: 0.0 }
    }
}

#[derive(Clone, Debug)]
pub struct GCfg {
    pub zthr: f64,
    /// forgetting of the prequential loss-difference statistics
    pub lam: f64,
}
impl Default for GCfg {
    fn default() -> Self {
        GCfg { zthr: 2.0, lam: 0.995 }
    }
}

#[derive(Clone, Debug)]
pub struct CCfg {
    pub q: f64,
    pub margin: f64,
    pub warm: usize,
    pub decay: f64,
}
impl Default for CCfg {
    fn default() -> Self {
        CCfg { q: 0.12, margin: 0.0, warm: 60, decay: 0.998 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode {
    Add,
    Pair,
    Gate,
}

struct Model {
    n: usize,
    pair: bool,
    b0: f64,
    g: Vec<f64>,
    hf: Vec<f64>, // rows * n
    hcnt: Vec<f64>,
    j: Vec<f64>,  // n*n (empty if !pair)
    nvis: Vec<f64>,
    npair: Vec<f64>,
}

impl Model {
    fn new(n: usize, pair: bool) -> Model {
        Model { n, pair, b0: 0.3, g: vec![0.0; n], hf: vec![], hcnt: vec![], j: if pair { vec![0.0; n * n] } else { vec![] }, nvis: vec![0.0; n], npair: if pair { vec![0.0; n * n] } else { vec![] } }
    }
    fn add_row(&mut self, lead: Option<usize>, rho: f64) {
        let base = self.hf.len();
        self.hf.extend(std::iter::repeat(0.0).take(self.n));
        self.hcnt.extend(std::iter::repeat(0.0).take(self.n));
        if let Some(l) = lead {
            self.hf[base + l] = rho;
        }
    }
    fn hvec(&self, fi: &[usize], out: &mut Vec<f64>) {
        out.clear();
        let inv = 1.0 / fi.len().max(1) as f64;
        for i in 0..self.n {
            let mut m = 0.0;
            for &f in fi {
                m += self.hf[f * self.n + i];
            }
            out.push(self.g[i] + m * inv);
        }
    }
    fn predict(&self, fi: &[usize], team: &[usize]) -> f64 {
        let inv = 1.0 / fi.len().max(1) as f64;
        let mut v = self.b0;
        for (x, &a) in team.iter().enumerate() {
            let mut m = 0.0;
            for &f in fi {
                m += self.hf[f * self.n + a];
            }
            v += self.g[a] + m * inv;
            if self.pair {
                for &b in &team[x + 1..] {
                    v += self.j[a * self.n + b];
                }
            }
        }
        v
    }
    fn update(&mut self, fi: &[usize], team: &[usize], reward: f64, c: &LCfg) {
        let pred = self.predict(fi, team);
        let err = reward - pred;
        let kf = team.len() as f64;
        let pairs = if self.pair { kf * (kf - 1.0) / 2.0 } else { 0.0 };
        let step = c.lr * err / (1.0 + 2.0 * kf + pairs);
        let shrink = 1.0 - c.l2;
        self.b0 += step;
        let bo = c.boost;
        for &a in team {
            let mg = 1.0 + bo / (1.0 + self.nvis[a] / 30.0);
            self.g[a] = self.g[a] * shrink + step * mg;
            for &f in fi {
                let ix = f * self.n + a;
                let mh = 1.0 + bo / (1.0 + self.hcnt[ix] / 30.0);
                self.hf[ix] = self.hf[ix] * shrink + step * mh;
                self.hcnt[ix] += 1.0;
            }
            self.nvis[a] += 1.0;
        }
        if self.pair {
            let n = self.n;
            for (x, &a) in team.iter().enumerate() {
                for &b in &team[x + 1..] {
                    let v = self.j[a * n + b] * shrink + step;
                    self.j[a * n + b] = v;
                    self.j[b * n + a] = v;
                    self.npair[a * n + b] += 1.0;
                    self.npair[b * n + a] += 1.0;
                }
            }
        }
    }
}

/// forward greedy + swap polish on h (+ J if given). J is row-major n*n symmetric.
pub fn greedy_team(n: usize, k: usize, h: &[f64], j: Option<&[f64]>) -> Vec<usize> {
    let mut team: Vec<usize> = Vec::with_capacity(k);
    let jj = |a: usize, b: usize| j.map(|m| m[a * n + b]).unwrap_or(0.0);
    for _ in 0..k {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for b in 0..n {
            if team.contains(&b) {
                continue;
            }
            let g = h[b] + team.iter().map(|&a| jj(a, b)).sum::<f64>();
            if g > best.0 {
                best = (g, b);
            }
        }
        team.push(best.1);
    }
    if j.is_some() {
        // swap polish (additive-only case is already optimal)
        for _ in 0..(k * n) {
            let mut bm = (1e-12, usize::MAX, 0usize);
            for pos in 0..k {
                let old = team[pos];
                for b in 0..n {
                    if team.contains(&b) {
                        continue;
                    }
                    let mut d = h[b] - h[old];
                    for (x, &c) in team.iter().enumerate() {
                        if x != pos {
                            d += jj(c, b) - jj(c, old);
                        }
                    }
                    if d > bm.0 {
                        bm = (d, pos, b);
                    }
                }
            }
            if bm.1 == usize::MAX {
                break;
            }
            team[bm.1] = bm.2;
        }
    }
    team
}

pub struct Learner {
    cfg: LCfg,
    gate: GCfg,
    mode: Mode,
    n: usize,
    k: usize,
    intern: HashMap<String, usize>,
    lead: Vec<Option<usize>>,
    kw: HashMap<String, usize>,
    m_add: Option<Model>,
    m_pair: Option<Model>,
    gm: f64,
    gv: f64,
    gate_on: bool,
    pub gate_on_steps: u64,
    pub steps: u64,
    label: String,
    hbuf: Vec<f64>,
}

impl Learner {
    pub fn new(cfg: LCfg, gate: GCfg, mode: Mode, kw: &HashMap<String, usize>, n: usize, k: usize) -> Learner {
        let label = match mode {
            Mode::Add => "L-add",
            Mode::Pair => "L-pair",
            Mode::Gate => "L-gate",
        };
        let mut l = Learner {
            cfg,
            gate,
            mode,
            n,
            k,
            intern: HashMap::new(),
            lead: vec![],
            kw: kw.clone(),
            m_add: if mode == Mode::Pair { None } else { Some(Model::new(n, false)) },
            m_pair: if mode == Mode::Add { None } else { Some(Model::new(n, true)) },
            gm: 0.0,
            gv: 1e-4,
            gate_on: false,
            gate_on_steps: 0,
            steps: 0,
            label: label.into(),
            hbuf: Vec::with_capacity(n),
        };
        let mut names: Vec<String> = kw.keys().cloned().collect();
        names.sort();
        for nm in names {
            l.id_of(&nm);
        }
        l
    }

    fn id_of(&mut self, name: &str) -> usize {
        if let Some(&i) = self.intern.get(name) {
            return i;
        }
        let i = self.lead.len();
        let lead = self.kw.get(name).cloned();
        self.lead.push(lead);
        self.intern.insert(name.to_string(), i);
        let rho = self.cfg.rho;
        if let Some(m) = self.m_add.as_mut() {
            m.add_row(lead, rho);
        }
        if let Some(m) = self.m_pair.as_mut() {
            m.add_row(lead, rho);
        }
        i
    }

    fn ids(&mut self, task: &Task) -> Vec<usize> {
        task.features.iter().map(|f| self.id_of(f)).collect()
    }

    pub fn gate_on_fraction(&self) -> f64 {
        if self.steps > 0 { self.gate_on_steps as f64 / self.steps as f64 } else { 0.0 }
    }

    fn use_pair(&self) -> bool {
        match self.mode {
            Mode::Add => false,
            Mode::Pair => true,
            Mode::Gate => self.gate_on,
        }
    }
}

impl TeamPol for Learner {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn diag(&self) -> (f64, f64) {
        (self.gate_on_fraction(), 1.0)
    }
    fn select(&mut self, task: &Task, rng: &mut Rng) -> Vec<usize> {
        let fi = self.ids(task);
        let up = self.use_pair();
        let n = self.n;
        let m = if up { self.m_pair.as_ref().unwrap() } else { self.m_add.as_ref().unwrap() };
        let mut h = std::mem::take(&mut self.hbuf);
        m.hvec(&fi, &mut h);
        if self.cfg.c1 > 0.0 {
            for i in 0..n {
                h[i] += self.cfg.c1 / (1.0 + m.nvis[i]).sqrt();
            }
        }
        let mut team = if up {
            if self.cfg.c2 > 0.0 {
                let mut j = m.j.clone();
                for a in 0..n {
                    for b in 0..n {
                        if a != b {
                            j[a * n + b] += self.cfg.c2 / (1.0 + m.npair[a * n + b]).sqrt();
                        }
                    }
                }
                greedy_team(n, self.k, &h, Some(&j))
            } else {
                greedy_team(n, self.k, &h, Some(&m.j))
            }
        } else {
            greedy_team(n, self.k, &h, None)
        };
        self.hbuf = h;
        if rng.f64() < self.cfg.eps {
            let pos = rng.below(self.k);
            loop {
                let b = rng.below(n);
                if !team.contains(&b) {
                    team[pos] = b;
                    break;
                }
            }
        }
        team
    }
    fn observe(&mut self, task: &Task, team: &[usize], reward: f64) {
        let fi = self.ids(task);
        self.steps += 1;
        if self.gate_on {
            self.gate_on_steps += 1;
        }
        if self.mode == Mode::Gate {
            let pa = self.m_add.as_ref().unwrap().predict(&fi, team);
            let pp = self.m_pair.as_ref().unwrap().predict(&fi, team);
            let d = (reward - pa).powi(2) - (reward - pp).powi(2);
            let lam = self.gate.lam;
            self.gm = lam * self.gm + (1.0 - lam) * d;
            self.gv = lam * self.gv + (1.0 - lam) * (d - self.gm).powi(2);
            let neff = 1.0 / (1.0 - lam);
            let z = self.gm / (self.gv.max(1e-12) / neff).sqrt();
            // only trust after enough evidence has accumulated
            self.gate_on = self.steps >= 100 && self.gm > 0.0 && z > self.gate.zthr;
        }
        let c = self.cfg.clone();
        if let Some(m) = self.m_add.as_mut() {
            m.update(&fi, team, reward, &c);
        }
        if let Some(m) = self.m_pair.as_mut() {
            m.update(&fi, team, reward, &c);
        }
    }
}

/// causal controller gate between the learned team model and sequential-select-with-team-credit
pub struct Gated {
    l: Learner,
    s: SeqTeam,
    cc: CCfg,
    sum: [f64; 2],
    cnt: [f64; 2],
    t: usize,
    last: usize,
    last_explore: bool,
    pub lrn_picks: u64,
    pub picks: u64,
    label: String,
}

impl Gated {
    pub fn new(l: Learner, s: SeqTeam, cc: CCfg) -> Gated {
        Gated { l, s, cc, sum: [0.0; 2], cnt: [0.0; 2], t: 0, last: 0, last_explore: false, lrn_picks: 0, picks: 0, label: "L-full".into() }
    }
    pub fn lrn_fraction(&self) -> f64 {
        if self.picks > 0 { self.lrn_picks as f64 / self.picks as f64 } else { 0.0 }
    }
    pub fn gate_on_fraction(&self) -> f64 {
        self.l.gate_on_fraction()
    }
}

impl TeamPol for Gated {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn diag(&self) -> (f64, f64) {
        (self.gate_on_fraction(), self.lrn_fraction())
    }
    fn select(&mut self, task: &Task, rng: &mut Rng) -> Vec<usize> {
        self.t += 1;
        let explore = self.t <= self.cc.warm || rng.f64() < self.cc.q;
        let arm = if explore {
            rng.below(2)
        } else {
            let m = |i: usize| if self.cnt[i] > 0.0 { self.sum[i] / self.cnt[i] } else { 0.0 };
            if m(0) > m(1) + self.cc.margin { 0 } else { 1 }
        };
        self.last = arm;
        self.last_explore = explore;
        self.picks += 1;
        if arm == 0 {
            self.lrn_picks += 1;
            self.l.select(task, rng)
        } else {
            self.s.select(task, rng)
        }
    }
    fn observe(&mut self, task: &Task, team: &[usize], reward: f64) {
        if self.last_explore {
            let a = self.last;
            self.sum[a] = self.sum[a] * self.cc.decay + reward;
            self.cnt[a] = self.cnt[a] * self.cc.decay + 1.0;
        }
        self.l.observe(task, team, reward);
        self.s.observe(task, team, reward);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_fits_repeated_sample() {
        let mut m = Model::new(5, true);
        m.add_row(None, 0.0);
        let c = LCfg { lr: 0.5, ..LCfg::default() };
        for _ in 0..200 {
            m.update(&[0], &[0, 2, 4], 0.9, &c);
        }
        assert!((m.predict(&[0], &[0, 2, 4]) - 0.9).abs() < 0.02);
    }

    #[test]
    fn greedy_matches_exhaustive_on_additive_and_mostly_on_pairs() {
        let mut rng = Rng::new(9);
        let (n, k) = (9, 3);
        let mut ok = 0;
        for _ in 0..60 {
            let h: Vec<f64> = (0..n).map(|_| rng.f64()).collect();
            let mut j = vec![0.0; n * n];
            for a in 0..n {
                for b in a + 1..n {
                    let v = 0.6 * (rng.f64() - 0.5);
                    j[a * n + b] = v;
                    j[b * n + a] = v;
                }
            }
            let g = greedy_team(n, k, &h, Some(&j));
            let val = |t: &[usize]| -> f64 {
                let mut v: f64 = t.iter().map(|&a| h[a]).sum();
                for (x, &a) in t.iter().enumerate() {
                    for &b in &t[x + 1..] {
                        v += j[a * n + b];
                    }
                }
                v
            };
            let mut best = f64::NEG_INFINITY;
            super::super::world::combos(n, k, |t| best = best.max(val(t)));
            if val(&g) >= best - 1e-9 {
                ok += 1;
            }
            // additive: exact
            let ga = greedy_team(n, k, &h, None);
            let mut b2 = f64::NEG_INFINITY;
            super::super::world::combos(n, k, |t| b2 = b2.max(t.iter().map(|&a| h[a]).sum::<f64>()));
            assert!((ga.iter().map(|&a| h[a]).sum::<f64>() - b2).abs() < 1e-9);
        }
        assert!(ok >= 45, "greedy+polish optimal on only {ok}/60 random pair instances");
    }
}
