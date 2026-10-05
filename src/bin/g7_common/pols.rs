//! G7 policies. Fair baselines wrap the public `Router`s unchanged. Quantum-inspired arms learn a binary
//! cost Hamiltonian ONLINE from reward only (they never see the world's hidden tables) and solve it with
//! simulated annealing / parallel tempering / a QAOA-style layered schedule (see solve.rs).
use super::solve::*;
use super::world::*;
use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::defaults;
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::mesh::{MeshConfig, NeuralMesh};
use neural_mesh::prior::{Evidence, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use std::collections::HashMap;

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

pub fn fair_grid() -> Vec<Kind> {
    let mut fg = vec![];
    for &g in &[0.9, 0.95, 0.99, 1.0] {
        for &e in &[0.02, 0.05, 0.1] {
            fg.push(Kind::Eps(g, e));
        }
    }
    for &g in &[0.9, 0.95, 0.99, 1.0] {
        fg.push(Kind::Pts(g));
    }
    for &k in &[3.0, 10.0, 30.0, 100.0] {
        for &e in &[EvK::Set, EvK::Mean, EvK::FailSet] {
            fg.push(Kind::Warm(k, e));
        }
    }
    fg.push(Kind::HebR);
    fg.push(Kind::Keyword);
    fg.push(Kind::GuardedSafe);
    fg
}

pub fn make_router(kind: &Kind, kw: &HashMap<String, usize>, a: usize) -> Box<dyn Router> {
    match kind {
        Kind::Keyword => Box::new(KeywordRouter::new(kw.clone(), a)),
        Kind::Eps(g, e) => Box::new(EpsGreedyRouter::new(a, *g, *e)),
        Kind::Pts(g) => Box::new(PooledThompsonRouter::new(a, *g)),
        Kind::HebR => Box::new(HebbianRouter::new(a, heb_robust_cfg())),
        Kind::Warm(k, e) => {
            let ev: Evidence<HebbianRouter> = match e {
                EvK::Set => Evidence::FeatureSet,
                EvK::Mean => Evidence::MeanFeature,
                EvK::FailSet => Evidence::SetFailures,
            };
            Box::new(Warm::new(HebbianRouter::new(a, heb_robust_cfg()), kw.clone(), a, *k).with_evidence(ev))
        }
        Kind::GuardedSafe => Box::new(defaults::recommended_router(kw.clone(), a)),
    }
}

// ============================================================ Part A policies
pub trait TeamPol {
    fn name(&self) -> String;
    fn select(&mut self, task: &Task, rng: &mut Rng) -> Vec<usize>;
    fn observe(&mut self, task: &Task, team: &[usize], reward: f64);
    /// optimizer move evaluations per decision so far (0 for non-optimizers)
    fn evals(&self) -> f64 {
        0.0
    }
}

fn pad_team(mut team: Vec<usize>, n: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    team.dedup();
    while team.len() < k {
        let b = rng.below(n);
        if !team.contains(&b) {
            team.push(b);
        }
    }
    team.truncate(k);
    team
}

/// Sequential-router team: K successive `select`s over the shrinking candidate set; every member is credited with the team reward.
pub struct SeqTeam {
    r: Box<dyn Router>,
    n: usize,
    k: usize,
    label: String,
}
impl SeqTeam {
    pub fn new(kind: &Kind, kw: &HashMap<String, usize>, n: usize, k: usize) -> Self {
        SeqTeam { r: make_router(kind, kw, n), n, k, label: format!("seq:{}", kind.label()) }
    }
}
impl TeamPol for SeqTeam {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn select(&mut self, task: &Task, rng: &mut Rng) -> Vec<usize> {
        let mut rem: Vec<usize> = (0..self.n).collect();
        let mut team = vec![];
        for _ in 0..self.k {
            let a = self.r.select(task, &rem, rng);
            let a = if rem.contains(&a) { a } else { rem[0] };
            team.push(a);
            rem.retain(|&x| x != a);
        }
        team
    }
    fn observe(&mut self, task: &Task, team: &[usize], reward: f64) {
        for &a in team {
            self.r.update(task, a, reward);
        }
    }
}

/// NeuralMesh (leader router + learned co-activation edges) with team_size = K.
pub enum MeshTeam {
    Guarded(NeuralMesh<Guarded<HebbianRouter>>, usize, usize),
    Heb(NeuralMesh<HebbianRouter>, usize, usize),
}
impl MeshTeam {
    pub fn guarded(kw: &HashMap<String, usize>, n: usize, k: usize) -> Self {
        let cfg = MeshConfig { team_size: k, ..MeshConfig::default() };
        MeshTeam::Guarded(NeuralMesh::new(n, defaults::recommended_router(kw.clone(), n), cfg), n, k)
    }
    pub fn heb(n: usize, k: usize) -> Self {
        let cfg = MeshConfig { team_size: k, ..MeshConfig::default() };
        MeshTeam::Heb(NeuralMesh::new(n, HebbianRouter::new(n, heb_robust_cfg()), cfg), n, k)
    }
}
impl TeamPol for MeshTeam {
    fn name(&self) -> String {
        match self {
            MeshTeam::Guarded(..) => "mesh(guarded-safe)".into(),
            MeshTeam::Heb(..) => "mesh(hebbian-robust)".into(),
        }
    }
    fn select(&mut self, task: &Task, rng: &mut Rng) -> Vec<usize> {
        match self {
            MeshTeam::Guarded(m, n, k) => {
                let c: Vec<usize> = (0..*n).collect();
                let t = m.route(task, &c, rng);
                pad_team(t, *n, *k, rng)
            }
            MeshTeam::Heb(m, n, k) => {
                let c: Vec<usize> = (0..*n).collect();
                let t = m.route(task, &c, rng);
                pad_team(t, *n, *k, rng)
            }
        }
    }
    fn observe(&mut self, task: &Task, team: &[usize], reward: f64) {
        match self {
            MeshTeam::Guarded(m, ..) => m.feedback(task, team, reward),
            MeshTeam::Heb(m, ..) => m.feedback(task, team, reward),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Solver {
    Sa(usize),
    Pt(usize),
    Qaoa(usize),
    Exact,
    Greedy,
}

impl Solver {
    pub fn label(&self) -> String {
        match self {
            Solver::Sa(i) => format!("sa{i}"),
            Solver::Pt(i) => format!("pt{i}"),
            Solver::Qaoa(i) => format!("qaoa{i}"),
            Solver::Exact => "exact".into(),
            Solver::Greedy => "greedy".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct QCfg {
    /// NLMS step (1 = fit the sample exactly)
    pub lr: f64,
    pub l2: f64,
    pub eps: f64,
    /// optimism bonus on agent count / pair count
    pub c1: f64,
    pub c2: f64,
    /// keyword prior: initial feature->lead single-agent value
    pub rho: f64,
    /// learn the pairwise (J) terms
    pub pair: bool,
}

impl Default for QCfg {
    fn default() -> Self {
        QCfg { lr: 0.3, l2: 0.0005, eps: 0.05, c1: 0.0, c2: 0.0, rho: 0.1, pair: true }
    }
}

/// Online QUBO surrogate + annealer. value(team | task) = b0 + sum_i (g_i + mean_f H[f][i]) x_i + sum_{i<j} J_ij x_i x_j.
pub struct QPol {
    cfg: QCfg,
    solver: Solver,
    n: usize,
    k: usize,
    b0: f64,
    g: Vec<f64>,
    hf: Vec<Vec<f64>>,
    j: Vec<f64>,
    nvis: Vec<f64>,
    npair: Vec<f64>,
    decisions: f64,
    evals: f64,
    label: String,
    t0: f64,
    t1: f64,
}

impl QPol {
    pub fn new(cfg: QCfg, solver: Solver, kw: &HashMap<String, usize>, n: usize, k: usize) -> Self {
        let mut hf = vec![vec![0.0; n]; NFEAT];
        for (name, &lead) in kw {
            hf[feat_index(name)][lead] = cfg.rho;
        }
        let label = format!("q{}-{}", if cfg.pair { "" } else { "add" }, solver.label());
        QPol { cfg, solver, n, k, b0: 0.3, g: vec![0.0; n], hf, j: vec![0.0; n * n], nvis: vec![0.0; n], npair: vec![0.0; n * n], decisions: 0.0, evals: 0.0, label, t0: 0.08, t1: 0.004 }
    }

    fn feats(task: &Task) -> Vec<usize> {
        task.features.iter().map(|f| feat_index(f)).collect()
    }

    fn model(&self, fi: &[usize]) -> (Vec<f64>, &Vec<f64>) {
        let mut h = vec![0.0; self.n];
        for i in 0..self.n {
            let m: f64 = fi.iter().map(|&f| self.hf[f][i]).sum::<f64>() / fi.len() as f64;
            h[i] = self.g[i] + m;
        }
        (h, &self.j)
    }

    pub fn predict(&self, fi: &[usize], team: &[usize]) -> f64 {
        let (h, j) = self.model(fi);
        let mut v = self.b0;
        for (x, &a) in team.iter().enumerate() {
            v += h[a];
            if self.cfg.pair {
                for &b in &team[x + 1..] {
                    v += j[a * self.n + b];
                }
            }
        }
        v
    }
}

impl TeamPol for QPol {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn select(&mut self, task: &Task, rng: &mut Rng) -> Vec<usize> {
        let fi = Self::feats(task);
        let (mut h, j0) = {
            let (h, j) = self.model(&fi);
            (h, j.clone())
        };
        let n = self.n;
        for i in 0..n {
            h[i] += self.cfg.c1 / (1.0 + self.nvis[i]).sqrt();
        }
        let mut j = j0;
        if self.cfg.pair {
            for a in 0..n {
                for b in 0..n {
                    if a != b {
                        j[a * n + b] += self.cfg.c2 / (1.0 + self.npair[a * n + b]).sqrt();
                    }
                }
            }
        } else {
            j.iter_mut().for_each(|x| *x = 0.0);
        }
        let q = QuboK { n, k: self.k, h, j };
        let (team, ev) = match self.solver {
            Solver::Sa(it) => {
                let (s, e) = sa(&q, rng, it, self.t0, self.t1, None);
                (s.members, e)
            }
            Solver::Pt(it) => {
                let (s, e) = pt(&q, rng, it, 6, self.t1, self.t0, None);
                (s.members, e)
            }
            Solver::Qaoa(it) => {
                let (s, e) = qaoa(&q, rng, it, 6, 6, 1.0 / self.t0, 1.0 / self.t1, 0.4, None);
                (s.members, e)
            }
            Solver::Exact => q.exact(),
            Solver::Greedy => q.greedy(),
        };
        self.decisions += 1.0;
        self.evals += ev as f64;
        let mut team = team;
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
        let fi = Self::feats(task);
        let pred = self.predict(&fi, team);
        let err = reward - pred;
        let kf = team.len() as f64;
        let pairs = if self.cfg.pair { kf * (kf - 1.0) / 2.0 } else { 0.0 };
        let norm = 1.0 + 2.0 * kf + pairs;
        let step = self.cfg.lr * err / norm;
        let shrink = 1.0 - self.cfg.l2;
        self.b0 += step;
        for &a in team {
            self.g[a] = self.g[a] * shrink + step;
            for &f in &fi {
                self.hf[f][a] = self.hf[f][a] * shrink + step;
            }
            self.nvis[a] += 1.0;
        }
        if self.cfg.pair {
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
    fn evals(&self) -> f64 {
        if self.decisions > 0.0 { self.evals / self.decisions } else { 0.0 }
    }
}

// ============================================================ Part B policies
pub trait AsgPol {
    fn name(&self) -> String;
    fn select(&mut self, task: &Task, rng: &mut Rng) -> usize;
    fn observe(&mut self, task: &Task, agent: usize, reward: f64);
    fn evals(&self) -> f64 {
        0.0
    }
}

pub struct FairAsg {
    r: Box<dyn Router>,
    n: usize,
    label: String,
}
impl FairAsg {
    pub fn new(kind: &Kind, kw: &HashMap<String, usize>, n: usize) -> Self {
        FairAsg { r: make_router(kind, kw, n), n, label: kind.label() }
    }
}
impl AsgPol for FairAsg {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn select(&mut self, task: &Task, rng: &mut Rng) -> usize {
        let c: Vec<usize> = (0..self.n).collect();
        self.r.select(task, &c, rng)
    }
    fn observe(&mut self, task: &Task, agent: usize, reward: f64) {
        self.r.update(task, agent, reward);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ZSolver {
    Sa(usize),
    Pt(usize),
    Qaoa(usize),
    /// decoupled per-feature argmax (the exact solution of the problem WITHOUT the load term)
    Argmax,
    /// argmax + steepest-ascent polish of the load-coupled objective
    Greedy,
}

impl ZSolver {
    pub fn label(&self) -> String {
        match self {
            ZSolver::Sa(i) => format!("sa{i}"),
            ZSolver::Pt(i) => format!("pt{i}"),
            ZSolver::Qaoa(i) => format!("qaoa{i}"),
            ZSolver::Argmax => "argmax".into(),
            ZSolver::Greedy => "greedy".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ZCfg {
    pub period: usize,
    pub lam: f64,
    pub cap: f64,
    pub c: f64,
    pub rho: f64,
    pub eps: f64,
    pub gamma: f64,
    /// prior mean of a cell that is not the keyword lead (1.0 = optimistic initialisation)
    pub pm_other: f64,
}

impl Default for ZCfg {
    fn default() -> Self {
        ZCfg { period: 20, lam: 3.0, cap: 0.3, c: 0.1, rho: 4.0, eps: 0.04, gamma: 1.0, pm_other: 0.5 }
    }
}

/// Feature -> agent binary assignment Z (one-hot per feature), re-solved every `period` observations from the
/// reward-only statistics; objective = traffic-weighted value  -  lam * sum_a (load_a - cap)_+^2.
pub struct ZPol {
    cfg: ZCfg,
    solver: ZSolver,
    na: usize,
    cnt: Vec<f64>,
    sum: Vec<f64>,
    freq: Vec<f64>,
    lead: Vec<Option<usize>>,
    assign: Vec<usize>,
    seen: usize,
    decisions: f64,
    evals: f64,
    label: String,
}

impl ZPol {
    pub fn new(cfg: ZCfg, solver: ZSolver, kw: &HashMap<String, usize>, na: usize) -> Self {
        let mut lead = vec![None; NFEAT];
        for (name, &a) in kw {
            lead[feat_index(name)] = Some(a);
        }
        let label = format!("z-{}", solver.label());
        let mut z = ZPol { cfg, solver, na, cnt: vec![0.0; NFEAT * na], sum: vec![0.0; NFEAT * na], freq: vec![1.0; NFEAT], lead, assign: vec![0; NFEAT], seen: 0, decisions: 0.0, evals: 0.0, label };
        z.resolve(&mut Rng::new(1));
        z
    }

    fn mean(&self, f: usize, a: usize) -> f64 {
        let pm = if self.lead[f] == Some(a) { 0.8 } else { self.cfg.pm_other };
        let i = f * self.na + a;
        (self.sum[i] + self.cfg.rho * pm) / (self.cnt[i] + self.cfg.rho)
    }

    fn resolve(&mut self, rng: &mut Rng) {
        let tot: f64 = self.freq.iter().sum();
        let w: Vec<f64> = self.freq.iter().map(|x| (x / tot).max(1e-4)).collect();
        let mut val = vec![0.0; NFEAT * self.na];
        for f in 0..NFEAT {
            for a in 0..self.na {
                let i = f * self.na + a;
                val[i] = w[f] * (self.mean(f, a) + self.cfg.c / (1.0 + self.cnt[i]).sqrt());
            }
        }
        let lam = if self.solver == ZSolver::Argmax { 0.0 } else { self.cfg.lam };
        let p = Potts { nf: NFEAT, na: self.na, val, w, cap: self.cfg.cap, lam };
        let am = p.argmax();
        let warm = if self.seen == 0 { am.clone() } else { p.state_from(&self.assign) };
        let (t_hi, t_lo) = (0.03, 0.0005);
        let (s, ev) = match self.solver {
            ZSolver::Argmax => (am, 0),
            ZSolver::Greedy => p.polish(am),
            ZSolver::Sa(it) => {
                let (s, e) = sa(&p, rng, it, t_hi, t_lo, Some(warm));
                (s, e)
            }
            ZSolver::Pt(it) => {
                let (s, e) = pt(&p, rng, it, 6, t_lo, t_hi, Some(warm));
                (s, e)
            }
            ZSolver::Qaoa(it) => {
                let (s, e) = qaoa(&p, rng, it, 6, 6, 1.0 / t_hi, 1.0 / t_lo, 0.4, Some(warm));
                (s, e)
            }
        };
        // never return something worse than the warm start / argmax under the coupled objective
        self.assign = s.assign;
        self.evals += ev as f64;
    }
}

impl AsgPol for ZPol {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn select(&mut self, task: &Task, rng: &mut Rng) -> usize {
        self.decisions += 1.0;
        let fi: Vec<usize> = task.features.iter().map(|f| feat_index(f)).collect();
        for &f in &fi {
            self.freq[f] = self.freq[f] * 0.995 + 1.0;
        }
        if rng.f64() < self.cfg.eps {
            return rng.below(self.na);
        }
        let mut sc = vec![0.0; self.na];
        for &f in &fi {
            let a = self.assign[f];
            sc[a] += 1.0 + 1e-3 * self.mean(f, a);
        }
        (0..self.na).max_by(|&a, &b| sc[a].total_cmp(&sc[b])).unwrap()
    }
    fn observe(&mut self, task: &Task, agent: usize, reward: f64) {
        for f in task.features.iter().map(|f| feat_index(f)) {
            for a in 0..self.na {
                self.cnt[f * self.na + a] *= self.cfg.gamma;
                self.sum[f * self.na + a] *= self.cfg.gamma;
            }
            self.cnt[f * self.na + agent] += 1.0;
            self.sum[f * self.na + agent] += reward;
        }
        self.seen += 1;
        if self.seen % self.cfg.period == 0 {
            let mut r = Rng::new(0x2222 + self.seen as u64);
            self.resolve(&mut r);
        }
    }
    fn evals(&self) -> f64 {
        if self.decisions > 0.0 { self.evals / self.decisions } else { 0.0 }
    }
}
