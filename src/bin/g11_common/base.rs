//! G11 fair baselines: the repo's public routers, unchanged, wrapped as team policies.
//! * `SeqTeam`  : K successive `select`s over the shrinking candidate set, every member credited with the team reward.
//! * `MeshTeam` : NeuralMesh (leader router + co-activation edges) with team_size = K.
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

/// the repo's standard baseline grid (12 eps, 4 pts, 12 warm, hebR, keyword, guarded-safe)
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

pub trait TeamPol {
    fn name(&self) -> String;
    fn select(&mut self, task: &Task, rng: &mut Rng) -> Vec<usize>;
    fn observe(&mut self, task: &Task, team: &[usize], reward: f64);
    /// (fraction of steps the pairwise model was active, fraction of decisions taken by the learned controller); (0,0) for baselines
    fn diag(&self) -> (f64, f64) {
        (0.0, 0.0)
    }
}

pub fn pad_team(mut team: Vec<usize>, n: usize, k: usize, rng: &mut Rng) -> Vec<usize> {
    let mut seen = vec![];
    team.retain(|x| {
        if seen.contains(x) {
            false
        } else {
            seen.push(*x);
            true
        }
    });
    while team.len() < k {
        let b = rng.below(n);
        if !team.contains(&b) {
            team.push(b);
        }
    }
    team.truncate(k);
    team
}

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
