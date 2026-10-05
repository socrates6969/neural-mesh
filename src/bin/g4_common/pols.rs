//! G4 policies. Fair baselines wrap the public `Router` implementations unchanged.
//! Diversity arms are wrappers (candidate shortlisting) around a fair base `Router`, or an own
//! additive novelty-bonus UCB learner. They see: task features, scalar reward, and the output embedding
//! restricted to the world's obs dims. They never see Q, true styles or the value formula
//! (ONLY `Oracle` reads the `View`).

use super::world::*;
use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::defaults;
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use std::collections::{HashMap, VecDeque};

pub fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvK {
    Set,
    Mean,
    FailSet,
}

#[derive(Clone, Debug)]
pub enum Kind {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    HebR,
    Warm(f64, EvK),
    GuardedSafe,
}

impl Kind {
    pub fn label(&self) -> String {
        match self {
            Kind::Keyword => "keyword".into(),
            Kind::Eps(g, e) => format!("eps-greedy(g={g},e={e})"),
            Kind::Pts(g) => format!("pooled-ts(g={g})"),
            Kind::HebR => "hebbian-robust".into(),
            Kind::Warm(k, e) => format!("warm(hebR,k={k},{e:?})"),
            Kind::GuardedSafe => "guarded-safe".into(),
        }
    }
    pub fn family(&self) -> &'static str {
        match self {
            Kind::Keyword => "keyword",
            Kind::Eps(..) => "eps-greedy",
            Kind::Pts(..) => "pooled-thompson",
            Kind::HebR => "hebbian-robust",
            Kind::Warm(..) => "warm(hebr)",
            Kind::GuardedSafe => "guarded-safe",
        }
    }
}

pub fn make_router(kind: &Kind, w: &World) -> Box<dyn Router> {
    let a = w.cfg.agents;
    match kind {
        Kind::Keyword => Box::new(KeywordRouter::new(w.kw.clone(), a)),
        Kind::Eps(g, e) => Box::new(EpsGreedyRouter::new(a, *g, *e)),
        Kind::Pts(g) => Box::new(PooledThompsonRouter::new(a, *g)),
        Kind::HebR => Box::new(HebbianRouter::new(a, heb_robust_cfg())),
        Kind::Warm(k, e) => {
            let ev: Evidence<HebbianRouter> = match e {
                EvK::Set => Evidence::FeatureSet,
                EvK::Mean => Evidence::MeanFeature,
                EvK::FailSet => Evidence::SetFailures,
            };
            Box::new(Warm::new(HebbianRouter::new(a, heb_robust_cfg()), w.kw.clone(), a, *k).with_evidence(ev))
        }
        Kind::GuardedSafe => Box::new(defaults::recommended_router(w.kw.clone(), a)),
    }
}

pub fn dom_of(task: &Task) -> usize {
    task.features[0][3..].parse().unwrap_or(0)
}

// ------------------------------------------------------------------ fair policy (reward only)
pub struct FairPol {
    pub r: Box<dyn Router>,
    pub label: String,
}
impl FairPol {
    pub fn new(kind: &Kind, w: &World) -> Self {
        FairPol { r: make_router(kind, w), label: kind.label() }
    }
}
impl Pol for FairPol {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn select(&mut self, task: &Task, cand: &[usize], rng: &mut Rng, _v: &View) -> usize {
        self.r.select(task, cand, rng)
    }
    fn observe(&mut self, task: &Task, agent: usize, reward: f64, _out: &[f64], _v: &View) {
        self.r.update(task, agent, reward);
    }
}

// ------------------------------------------------------------------ shared diversity state
pub const WP: usize = 10; // policy's own window (world uses 8)
const CENT_ALPHA: f64 = 0.15;

pub struct Div {
    win: Vec<VecDeque<Vec<f64>>>,
    cent: Vec<Option<Vec<f64>>>,
    cent_n: Vec<f64>,
    recent: Vec<VecDeque<usize>>,
    last_fill: Vec<HashMap<u32, usize>>,
    cnt_d: Vec<usize>,
}

impl Div {
    pub fn new(nd: usize, na: usize) -> Self {
        Div {
            win: (0..nd).map(|_| VecDeque::new()).collect(),
            cent: vec![None; na],
            cent_n: vec![0.0; na],
            recent: (0..nd).map(|_| VecDeque::new()).collect(),
            last_fill: (0..nd).map(|_| HashMap::new()).collect(),
            cnt_d: vec![0; nd],
        }
    }
    /// policy's own novelty of a point vs its domain window: mean of the (up to) 3 nearest distances, scale 1.
    pub fn nov_point(&self, d: usize, p: &[f64]) -> f64 {
        let w = &self.win[d];
        if w.is_empty() {
            return 1.0;
        }
        let mut ds: Vec<f64> = w.iter().map(|q| dist_dims(p, q, 0, p.len())).collect();
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let k = ds.len().min(3);
        (ds[..k].iter().sum::<f64>() / k as f64).min(1.0)
    }
    pub fn nov_est(&self, d: usize, a: usize) -> f64 {
        match &self.cent[a] {
            None => 1.0,
            Some(c) => self.nov_point(d, c),
        }
    }
    pub fn cell_of(p: &[f64]) -> u32 {
        let mut c = 0u32;
        for (i, x) in p.iter().enumerate() {
            if *x > 0.0 {
                c |= 1 << i;
            }
        }
        c
    }
    pub fn staleness(&self, d: usize, a: usize, t_scale: f64) -> f64 {
        match &self.cent[a] {
            None => 1.0,
            Some(c) => match self.last_fill[d].get(&Self::cell_of(c)) {
                None => 1.0,
                Some(&f) => (((self.cnt_d[d] - f) as f64) / t_scale).min(1.0),
            },
        }
    }
    pub fn recent_agents(&self, d: usize, m: usize) -> Vec<usize> {
        self.recent[d].iter().rev().take(m).cloned().collect()
    }
    pub fn observe(&mut self, d: usize, a: usize, o: &[f64]) {
        match &mut self.cent[a] {
            None => {
                self.cent[a] = Some(o.to_vec());
                self.cent_n[a] = 1.0;
            }
            Some(c) => {
                self.cent_n[a] += 1.0;
                let al = CENT_ALPHA.max(1.0 / self.cent_n[a]);
                for i in 0..c.len() {
                    c[i] += al * (o[i] - c[i]);
                }
            }
        }
        self.win[d].push_back(o.to_vec());
        if self.win[d].len() > WP {
            self.win[d].pop_front();
        }
        self.recent[d].push_back(a);
        if self.recent[d].len() > 8 {
            self.recent[d].pop_front();
        }
        self.last_fill[d].insert(Self::cell_of(o), self.cnt_d[d]);
        self.cnt_d[d] += 1;
    }
}

// ------------------------------------------------------------------ gate
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateKind {
    /// novelty layer on in every domain
    None,
    /// ORACLE-ISH: told which domains are creative
    Declared,
    /// infers per domain from observed (reward, own novelty) whether novelty matters
    Learned,
    /// never on (reference arm: same machinery, novelty layer disabled)
    Off,
}
impl GateKind {
    pub fn tag(&self) -> &'static str {
        match self {
            GateKind::None => "nogate",
            GateKind::Declared => "declared",
            GateKind::Learned => "learned",
            GateKind::Off => "off",
        }
    }
}

#[derive(Clone, Default)]
struct Lg {
    sxy: f64,
    sxx: f64,
    syy: f64,
    n: f64,
    on: bool,
    mr: Vec<f64>,
    mn: Vec<f64>,
    cnt: Vec<f64>,
}

pub const LG_DISC: f64 = 0.995;
pub const LG_MIN_N: f64 = 30.0;
pub const LG_T_ON: f64 = 2.5;
pub const LG_T_OFF: f64 = 1.0;
pub const LG_B_ON: f64 = 0.15;

pub struct Gate {
    kind: GateKind,
    declared: Vec<bool>,
    lg: Vec<Lg>,
    // diagnostics: [on_creative, total_creative, on_correct, total_correct]
    pub dg: [f64; 4],
    truth: Vec<bool>, // used ONLY for the diagnostic counters
}

impl Gate {
    pub fn new(kind: GateKind, declared: &[bool], na: usize) -> Self {
        let nd = declared.len();
        let lg = (0..nd).map(|_| Lg { mr: vec![0.0; na], mn: vec![0.0; na], cnt: vec![0.0; na], ..Default::default() }).collect();
        Gate { kind, declared: declared.to_vec(), lg, dg: [0.0; 4], truth: declared.to_vec() }
    }
    pub fn on(&mut self, d: usize) -> bool {
        let on = match self.kind {
            GateKind::None => true,
            GateKind::Declared => self.declared[d],
            GateKind::Learned => self.lg[d].on,
            GateKind::Off => false,
        };
        let i = if self.truth[d] { 0 } else { 2 };
        self.dg[i + 1] += 1.0;
        if on {
            self.dg[i] += 1.0;
        }
        on
    }
    /// regression slope of reward on observed novelty (within-agent), 0 if not enough data
    pub fn slope(&self, d: usize) -> f64 {
        let g = &self.lg[d];
        if g.n >= LG_MIN_N && g.sxx > 1e-9 { g.sxy / g.sxx } else { 0.0 }
    }
    pub fn learned_on(&self, d: usize) -> bool {
        self.lg[d].on
    }
    pub fn declared_on(&self, d: usize) -> bool {
        self.declared[d]
    }
    /// (EMA reward, EMA observed novelty, count) of agent a in domain d
    pub fn agent_means(&self, d: usize, a: usize) -> (f64, f64, f64) {
        let g = &self.lg[d];
        (g.mr[a], g.mn[a], g.cnt[a])
    }
    pub fn update(&mut self, d: usize, a: usize, r: f64, nov_obs: f64) {
        if self.kind != GateKind::Learned {
            return;
        }
        let g = &mut self.lg[d];
        if g.cnt[a] >= 3.0 {
            let x = nov_obs - g.mn[a];
            let y = r - g.mr[a];
            g.sxy *= LG_DISC;
            g.sxx *= LG_DISC;
            g.syy *= LG_DISC;
            g.n *= LG_DISC;
            g.sxy += x * y;
            g.sxx += x * x;
            g.syy += y * y;
            g.n += 1.0;
            if g.n >= LG_MIN_N && g.sxx > 1e-9 {
                let b = g.sxy / g.sxx;
                let resid = ((g.syy - b * g.sxy) / (g.n - 2.0)).max(1e-12);
                let t = b / (resid / g.sxx).sqrt();
                if !g.on && t > LG_T_ON && b > LG_B_ON {
                    g.on = true;
                } else if g.on && (t < LG_T_OFF || b < LG_B_ON * 0.33) {
                    g.on = false;
                }
            }
        }
        let al = 0.05f64.max(1.0 / (g.cnt[a] + 1.0));
        g.mr[a] += al * (r - g.mr[a]);
        g.mn[a] += al * (nov_obs - g.mn[a]);
        g.cnt[a] += 1.0;
    }
}

// ------------------------------------------------------------------ wrapper arms
#[derive(Clone, Debug)]
pub enum Base {
    Warm(f64),
    Pts(f64),
}
impl Base {
    pub fn kind(&self) -> Kind {
        match self {
            Base::Warm(k) => Kind::Warm(*k, EvK::Mean),
            Base::Pts(g) => Kind::Pts(*g),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Mech {
    /// anti-repeat: exclude the last m distinct agents used in this domain (id-only, no output observation)
    Mask(usize),
    /// shortlist top-k agents by own novelty estimate (centroid vs window)
    Nov(usize),
    /// MAP-Elites-flavoured archive over sign-pattern cells of style space: shortlist top-k by cell staleness (scale T)
    Arch(usize, f64),
    /// own UCB learner with additive novelty bonus beta*nov_est, UCB constant c
    Nb(f64, f64),
    /// regression-calibrated novelty (NO beta): prediction = EMA reward of the agent + slope_d * (nov_est(a) - EMA nov_obs(a)),
    /// slope_d = within-agent regression of reward on the policy's own observed novelty (clipped to [0,1.5]); UCB constant c.
    /// Added AFTER seeing the misspecification-world results (see board entry): post-hoc design response, evaluated with no tuning.
    Reg(f64),
}

#[derive(Clone, Debug)]
pub struct ArmSpec {
    pub mech: Mech,
    pub base: Base,
    pub gate: GateKind,
}
impl ArmSpec {
    pub fn label(&self) -> String {
        let m = match &self.mech {
            Mech::Mask(m) => format!("antirepeat(m={m})"),
            Mech::Nov(k) => format!("novfilter(k={k})"),
            Mech::Arch(k, t) => format!("archive(k={k},T={t})"),
            Mech::Nb(b, c) => format!("nbucb(b={b},c={c})"),
            Mech::Reg(c) => format!("nbreg(c={c})"),
        };
        match &self.mech {
            Mech::Nb(..) | Mech::Reg(..) => format!("{m}|{}", self.gate.tag()),
            _ => format!("{m}/{}|{}", self.base.kind().label(), self.gate.tag()),
        }
    }
    pub fn family(&self) -> String {
        let m = match &self.mech {
            Mech::Mask(_) => "antirepeat",
            Mech::Nov(_) => "novfilter",
            Mech::Arch(..) => "archive",
            Mech::Nb(..) => "nbucb",
            Mech::Reg(..) => "nbreg",
        };
        format!("{m}|{}", self.gate.tag())
    }
}

pub struct Wrap {
    spec: ArmSpec,
    base: Box<dyn Router>,
    div: Div,
    gate: Gate,
    name: String,
}

impl Wrap {
    pub fn new(spec: &ArmSpec, w: &World) -> Self {
        let (nd, na) = (w.cfg.domains, w.cfg.agents);
        Wrap {
            spec: spec.clone(),
            base: make_router(&spec.base.kind(), w),
            div: Div::new(nd, na),
            gate: Gate::new(spec.gate, &w.cfg.creative, na),
            name: spec.label(),
        }
    }
}

fn top_k(cand: &[usize], key: impl Fn(usize) -> f64, k: usize, rng: &mut Rng) -> Vec<usize> {
    let mut v: Vec<(f64, usize)> = cand.iter().map(|&a| (key(a) + 1e-6 * rng.f64(), a)).collect();
    v.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    v.into_iter().take(k.max(1)).map(|x| x.1).collect()
}

impl Pol for Wrap {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn select(&mut self, task: &Task, cand: &[usize], rng: &mut Rng, _v: &View) -> usize {
        let d = dom_of(task);
        if !self.gate.on(d) {
            return self.base.select(task, cand, rng);
        }
        let sl: Vec<usize> = match &self.spec.mech {
            Mech::Mask(m) => {
                let ex = self.div.recent_agents(d, *m);
                let v: Vec<usize> = cand.iter().cloned().filter(|a| !ex.contains(a)).collect();
                if v.is_empty() { cand.to_vec() } else { v }
            }
            Mech::Nov(k) => top_k(cand, |a| self.div.nov_est(d, a), *k, rng),
            Mech::Arch(k, t) => top_k(cand, |a| self.div.staleness(d, a, *t), *k, rng),
            Mech::Nb(..) | Mech::Reg(..) => unreachable!(),
        };
        self.base.select(task, &sl, rng)
    }
    fn observe(&mut self, task: &Task, agent: usize, reward: f64, out: &[f64], _v: &View) {
        let d = dom_of(task);
        let nov_obs = self.div.nov_point(d, out);
        self.gate.update(d, agent, reward, nov_obs);
        self.div.observe(d, agent, out);
        self.base.update(task, agent, reward);
    }
    fn diag(&self) -> Vec<f64> {
        gate_diag(&self.gate)
    }
}

pub fn gate_diag(g: &Gate) -> Vec<f64> {
    let f = |a: f64, b: f64| if b > 0.0 { a / b } else { f64::NAN };
    vec![f(g.dg[0], g.dg[1]), f(g.dg[2], g.dg[3])]
}

// ------------------------------------------------------------------ additive novelty-bonus UCB (own learner)
pub struct NbUcb {
    beta: f64,
    c: f64,
    gamma: f64,
    s: Vec<Vec<f64>>,
    n: Vec<Vec<f64>>,
    nd_tot: Vec<f64>,
    div: Div,
    gate: Gate,
    name: String,
}
impl NbUcb {
    pub fn new(spec: &ArmSpec, w: &World) -> Self {
        let (nd, na) = (w.cfg.domains, w.cfg.agents);
        let (beta, c) = match spec.mech {
            Mech::Nb(b, c) => (b, c),
            _ => panic!("not Nb"),
        };
        NbUcb {
            beta,
            c,
            gamma: 0.97,
            s: vec![vec![0.0; na]; nd],
            n: vec![vec![0.0; na]; nd],
            nd_tot: vec![0.0; nd],
            div: Div::new(nd, na),
            gate: Gate::new(spec.gate, &w.cfg.creative, na),
            name: spec.label(),
        }
    }
}
impl Pol for NbUcb {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn select(&mut self, task: &Task, cand: &[usize], rng: &mut Rng, _v: &View) -> usize {
        let d = dom_of(task);
        let on = self.gate.on(d);
        let lnn = (self.nd_tot[d] + 2.0).ln();
        let mut best = cand[0];
        let mut bv = f64::NEG_INFINITY;
        for &a in cand {
            let n = self.n[d][a];
            let mean = if n > 1e-9 { self.s[d][a] / n } else { 0.9 };
            let ucb = self.c * (lnn / (n + 1.0)).sqrt();
            let bonus = if on { self.beta * self.div.nov_est(d, a) } else { 0.0 };
            let v = mean + ucb + bonus + 1e-9 * rng.f64();
            if v > bv {
                bv = v;
                best = a;
            }
        }
        best
    }
    fn observe(&mut self, task: &Task, agent: usize, reward: f64, out: &[f64], _v: &View) {
        let d = dom_of(task);
        let nov_obs = self.div.nov_point(d, out);
        self.gate.update(d, agent, reward, nov_obs);
        self.div.observe(d, agent, out);
        for a in 0..self.s[d].len() {
            self.s[d][a] *= self.gamma;
            self.n[d][a] *= self.gamma;
        }
        self.s[d][agent] += reward;
        self.n[d][agent] += 1.0;
        self.nd_tot[d] += 1.0;
    }
    fn diag(&self) -> Vec<f64> {
        gate_diag(&self.gate)
    }
}

pub fn make_arm(spec: &ArmSpec, w: &World) -> Box<dyn Pol> {
    match spec.mech {
        Mech::Nb(..) => Box::new(NbUcb::new(spec, w)),
        Mech::Reg(..) => Box::new(NbReg::new(spec, w)),
        _ => Box::new(Wrap::new(spec, w)),
    }
}

// ------------------------------------------------------------------ ORACLE-ISH ceiling
/// Knows true current styles, the exact novelty window and the exact value form and which domains are
/// creative; learns only the hidden quality (deconfounded). NOT a candidate: it bounds what routing on
/// diversity could gain in each world.
pub struct Oracle {
    s: Vec<Vec<f64>>,
    n: Vec<Vec<f64>>,
    tot: Vec<f64>,
}
impl Oracle {
    pub fn new(w: &World) -> Self {
        Oracle { s: vec![vec![0.0; w.cfg.agents]; w.cfg.domains], n: vec![vec![0.0; w.cfg.agents]; w.cfg.domains], tot: vec![0.0; w.cfg.domains] }
    }
}
impl Pol for Oracle {
    fn name(&self) -> String {
        "ORACLE-ISH(exact value form, true styles)".into()
    }
    fn select(&mut self, task: &Task, cand: &[usize], rng: &mut Rng, v: &View) -> usize {
        let d = dom_of(task);
        let cr = v.w.cfg.creative[d];
        let lnn = (self.tot[d] + 2.0).ln();
        let mut best = cand[0];
        let mut bv = f64::NEG_INFINITY;
        for &a in cand {
            let n = self.n[d][a];
            let qh = if n > 1e-9 { self.s[d][a] / n + 0.05 * (lnn / n).sqrt() } else { 1.0 };
            let f = if cr { FLOOR + (1.0 - FLOOR) * world_novelty(&v.w.style(a, v.t), &v.hist[d], v.w.cfg.val_dims) } else { 1.0 };
            let sc = qh * f + 1e-9 * rng.f64();
            if sc > bv {
                bv = sc;
                best = a;
            }
        }
        best
    }
    fn observe(&mut self, task: &Task, agent: usize, reward: f64, _out: &[f64], v: &View) {
        let d = dom_of(task);
        let f = if v.w.cfg.creative[d] { FLOOR + (1.0 - FLOOR) * v.last_nov } else { 1.0 };
        self.s[d][agent] += reward / f;
        self.n[d][agent] += 1.0;
        self.tot[d] += 1.0;
    }
}

// ------------------------------------------------------------------ regression-calibrated novelty bandit
pub struct NbReg {
    c: f64,
    kind: GateKind,
    gamma: f64,
    n: Vec<Vec<f64>>,
    nd_tot: Vec<f64>,
    div: Div,
    stats: Gate, // always a Learned gate (regression statistics); `kind` decides when the slope is used
    diag_g: Gate, // diagnostic counters only
    name: String,
}
impl NbReg {
    pub fn new(spec: &ArmSpec, w: &World) -> Self {
        let (nd, na) = (w.cfg.domains, w.cfg.agents);
        let c = match spec.mech {
            Mech::Reg(c) => c,
            _ => panic!("not Reg"),
        };
        NbReg {
            c,
            kind: spec.gate,
            gamma: 0.97,
            n: vec![vec![0.0; na]; nd],
            nd_tot: vec![0.0; nd],
            div: Div::new(nd, na),
            stats: Gate::new(GateKind::Learned, &w.cfg.creative, na),
            diag_g: Gate::new(GateKind::None, &w.cfg.creative, na),
            name: spec.label(),
        }
    }
    fn active(&self, d: usize) -> bool {
        match self.kind {
            GateKind::None => true,
            GateKind::Declared => self.stats.declared_on(d),
            GateKind::Learned => self.stats.learned_on(d),
            GateKind::Off => false,
        }
    }
}
impl Pol for NbReg {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn select(&mut self, task: &Task, cand: &[usize], rng: &mut Rng, _v: &View) -> usize {
        let d = dom_of(task);
        let act = self.active(d);
        // diagnostic: record activity (None-kind Gate::on always returns true, so count manually)
        let i = if self.stats.declared_on(d) { 0 } else { 2 };
        self.diag_g.dg[i + 1] += 1.0;
        if act {
            self.diag_g.dg[i] += 1.0;
        }
        let slope = if act { self.stats.slope(d).clamp(0.0, 1.5) } else { 0.0 };
        let lnn = (self.nd_tot[d] + 2.0).ln();
        let mut best = cand[0];
        let mut bv = f64::NEG_INFINITY;
        for &a in cand {
            let (mr, mn, cnt) = self.stats.agent_means(d, a);
            let mean = if cnt > 0.5 { mr + slope * (self.div.nov_est(d, a) - mn) } else { 0.9 + slope };
            let ucb = self.c * (lnn / (self.n[d][a] + 1.0)).sqrt();
            let v = mean + ucb + 1e-9 * rng.f64();
            if v > bv {
                bv = v;
                best = a;
            }
        }
        best
    }
    fn observe(&mut self, task: &Task, agent: usize, reward: f64, out: &[f64], _v: &View) {
        let d = dom_of(task);
        let nov_obs = self.div.nov_point(d, out);
        self.stats.update(d, agent, reward, nov_obs);
        self.div.observe(d, agent, out);
        for a in 0..self.n[d].len() {
            self.n[d][a] *= self.gamma;
        }
        self.n[d][agent] += 1.0;
        self.nd_tot[d] += 1.0;
    }
    fn diag(&self) -> Vec<f64> {
        gate_diag(&self.diag_g)
    }
}
