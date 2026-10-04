//! R2: cognitive-skill routing world + opt-in mechanisms (quality-diversity archive,
//! novelty bonus, generator->critic sequencing, domain-hierarchical routing).
//!
//! Self-contained: nothing in sim.rs is touched. Everything is seeded/deterministic.
//!
//! WORLD (hidden structure the routers never see)
//! * 5 domains: coding, science, design(creative), business, marketing(creative).
//! * A agents, K approaches ("styles") per domain, 3 topics per domain.
//! * Each task is a 3-step chain: PLAN (agent) -> BUILD (agent + approach) -> CRITIQUE (agent).
//!   - plan score qp in [0,1] scales the build (p_eff = skill * (0.7 + 0.3 qp)).
//!   - CORRECTNESS domains (coding, science, business): build is right w.p. p_eff * aff[agent][approach].
//!     A critic flags a wrong answer w.p. sens, a right one w.p. fa (false alarm); a flagged answer is
//!     regenerated once (fresh attempt, right w.p. p). Net gain of a critic = p(1-p)(sens-fa) - cost,
//!     so a critic with sens < fa HURTS. Value = final correct - critic cost.
//!   - CREATIVE domains (design, marketing): quality q = skill * aff[agent][approach] * plan factor + noise,
//!     value = q * (1 - lam + lam * novelty) where novelty is the mean clamped distance of the new solution's
//!     embedding to the last W solutions IN THAT DOMAIN (shared across all agents). Critique refines
//!     v += 0.3 * critic_skill * (1 - v). lam = 0 in the "novelty-worthless" and is a world parameter.
//! * Agents have a signature approach (used w.p. 0.85, else a random other approach) = their habit.
//!   Flat agent routing therefore cannot steer approach; niche routing (agent, approach) can.
//!
//! Routers see: per-stage Task{task_type=stage, features=["dom:d", noisy topic keywords]}, a reward in
//! [0,1], and (mechanisms only, via the hooks in `BuildPolicy`/`CriticPolicy`) the observed approach label,
//! a quality score separate from novelty, and (critic "verdict" mode) whether the critic verdict matched
//! the later ground truth. All of that is documented as an information assumption.

use std::collections::{HashMap, VecDeque};

use crate::baselines::{sample_beta, EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use crate::hebbian::{HebbianConfig, HebbianRouter};
use crate::prior::{Evidence, Warm};
use crate::router::{Rng, Router, Task};

pub const N_DOMAINS: usize = 5;
pub const DOMAIN_NAMES: [&str; N_DOMAINS] = ["coding", "science", "design", "business", "marketing"];
pub const CREATIVE: [bool; N_DOMAINS] = [false, false, true, false, true];
const HABIT: f64 = 0.85;
const DIST_SCALE: f64 = 1.6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Structure {
    /// each agent has one strong domain, one medium, rest weak
    Specialist,
    /// skill ~ a global agent ability, nearly independent of domain
    Generalist,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Peak {
    /// creative domains: several agents with distinct signature approaches reach similar quality
    Multi,
    /// creative domains: one approach is just better for everybody
    Single,
}

#[derive(Clone, Debug)]
pub struct WorldCfg {
    pub name: String,
    pub agents: usize,
    pub approaches: usize,
    pub topics_per_domain: usize,
    pub n_tasks: usize,
    pub structure: Structure,
    pub peak: Peak,
    /// weight of novelty in creative value
    pub lam: f64,
    /// quality noise sd
    pub noise: f64,
    /// at this task index the visible-agent -> hidden-skill map rotates by one
    pub drift_at: Option<usize>,
    /// every critic has the same sens/fa (critic choice irrelevant)
    pub critic_equal: bool,
    /// solution embedding is tied to the approach label (false: embedding independent of label)
    pub labels_informative: bool,
    /// false: the approach does not change quality (aff = 1 everywhere) - nothing for niche routing to exploit
    pub approach_matters: bool,
    pub window: usize,
    pub critic_cost: f64,
    pub map_err: f64,
    pub traps: usize,
    pub confound: f64,
}

impl WorldCfg {
    pub fn base(name: &str) -> Self {
        WorldCfg {
            name: name.into(),
            agents: 8,
            approaches: 4,
            topics_per_domain: 3,
            n_tasks: 1500,
            structure: Structure::Specialist,
            peak: Peak::Multi,
            lam: 0.6,
            noise: 0.08,
            drift_at: None,
            critic_equal: false,
            labels_informative: true,
            approach_matters: true,
            window: 10,
            critic_cost: 0.02,
            map_err: 0.10,
            traps: 3,
            confound: 0.25,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TaskSpec {
    pub domain: usize,
    pub topic: usize, // global topic index
    pub features: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct Draws {
    n_plan: f64,
    u_c1: f64,
    u_flag: f64,
    u_retry: f64,
    n_q: f64,
    u_habit: f64,
    u_appr: f64,
    u_center: f64,
    emb: [f64; 3],
}

pub struct World {
    pub cfg: WorldCfg,
    /// [agent][global topic]
    pub build: Vec<Vec<f64>>,
    /// [agent][domain]
    pub plan: Vec<Vec<f64>>,
    pub sens: Vec<Vec<f64>>,
    pub fa: Vec<Vec<f64>>,
    pub crit_skill: Vec<Vec<f64>>,
    /// [agent][domain][approach] multiplier (1.0 at the signature)
    pub aff: Vec<Vec<Vec<f64>>>,
    pub sig: Vec<Vec<usize>>,
    pub centers: Vec<[f64; 3]>,
    pub tasks: Vec<TaskSpec>,
    draws: Vec<Draws>,
    /// per stage (0 plan, 1 build, 2 critic) keyword -> agent
    pub kw: [HashMap<String, usize>; 3],
}

fn normal(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - rng.f64();
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn skill_matrix(st: Structure, a_n: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; N_DOMAINS]; a_n];
    match st {
        Structure::Specialist => {
            let mut perm: Vec<usize> = (0..N_DOMAINS).collect();
            for i in (1..N_DOMAINS).rev() {
                perm.swap(i, rng.below(i + 1));
            }
            for a in 0..a_n {
                let prim = perm[a % N_DOMAINS];
                let sec = perm[(a + 1 + a / N_DOMAINS) % N_DOMAINS];
                for d in 0..N_DOMAINS {
                    m[a][d] = if d == prim {
                        0.85 + 0.05 * (rng.f64() - 0.5)
                    } else if d == sec {
                        0.6 + 0.1 * (rng.f64() - 0.5)
                    } else {
                        0.3 + 0.2 * rng.f64()
                    };
                }
            }
        }
        Structure::Generalist => {
            for a in 0..a_n {
                let g = 0.35 + 0.5 * rng.f64();
                for d in 0..N_DOMAINS {
                    m[a][d] = clamp01(g + 0.06 * (2.0 * rng.f64() - 1.0));
                }
            }
        }
    }
    m
}

pub fn gen_world(seed: u64, c: &WorldCfg) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(101));
    let (a_n, k_n, tpd) = (c.agents, c.approaches, c.topics_per_domain);
    let g_n = N_DOMAINS * tpd;
    let base_b = skill_matrix(c.structure, a_n, &mut rng);
    let plan = skill_matrix(c.structure, a_n, &mut rng);
    let qc = skill_matrix(c.structure, a_n, &mut rng);
    let mut build = vec![vec![0.0; g_n]; a_n];
    for a in 0..a_n {
        for g in 0..g_n {
            build[a][g] = (base_b[a][g / tpd] + 0.08 * (2.0 * rng.f64() - 1.0)).clamp(0.05, 0.98);
        }
    }
    let (mut sens, mut fa, mut crit_skill) = (vec![vec![0.0; N_DOMAINS]; a_n], vec![vec![0.0; N_DOMAINS]; a_n], vec![vec![0.0; N_DOMAINS]; a_n]);
    for c_i in 0..a_n {
        for d in 0..N_DOMAINS {
            if c.critic_equal {
                sens[c_i][d] = 0.65;
                fa[c_i][d] = 0.15;
                crit_skill[c_i][d] = 0.6;
            } else {
                sens[c_i][d] = 0.2 + 0.7 * qc[c_i][d];
                fa[c_i][d] = (0.45 - 0.4 * qc[c_i][d]).max(0.03);
                crit_skill[c_i][d] = qc[c_i][d];
            }
        }
    }
    // signatures + affinities
    let mut sig = vec![vec![0usize; N_DOMAINS]; a_n];
    let mut aff = vec![vec![vec![1.0; k_n]; N_DOMAINS]; a_n];
    for d in 0..N_DOMAINS {
        let kstar = rng.below(k_n);
        let doff = rng.below(k_n);
        let mut order: Vec<usize> = (0..a_n).collect();
        order.sort_by(|&x, &y| base_b[y][d].partial_cmp(&base_b[x][d]).unwrap());
        for (rank, &a) in order.iter().enumerate() {
            sig[a][d] = if CREATIVE[d] {
                match c.peak {
                    Peak::Multi => {
                        if rank < k_n { (rank + doff) % k_n } else { rng.below(k_n) }
                    }
                    Peak::Single => kstar,
                }
            } else {
                rng.below(k_n)
            };
        }
        for a in 0..a_n {
            for k in 0..k_n {
                aff[a][d][k] = if k == sig[a][d] || !c.approach_matters {
                    1.0
                } else if CREATIVE[d] {
                    match c.peak {
                        Peak::Multi => 0.55 + 0.15 * rng.f64(),
                        Peak::Single => 0.5,
                    }
                } else {
                    0.75 + 0.15 * rng.f64()
                };
            }
        }
    }
    // approach centres: scaled tetrahedron-like corners (pairwise dist ~1.6 for k<=4)
    let corners = [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]];
    let centers: Vec<[f64; 3]> = (0..k_n)
        .map(|k| {
            let c0 = corners[k % 4];
            let s = 0.566 * if k >= 4 { 1.3 } else { 1.0 };
            [c0[0] * s, c0[1] * s, c0[2] * s]
        })
        .collect();

    // tasks + draws
    let mut tasks = vec![];
    let mut draws = vec![];
    for _ in 0..c.n_tasks {
        let d = rng.below(N_DOMAINS);
        let g = d * tpd + rng.below(tpd);
        let mut f = vec![format!("dom:{d}")];
        let mut any = false;
        for j in 0..3 {
            if rng.bernoulli(0.5) {
                f.push(format!("kw{g}.{j}"));
                any = true;
            }
        }
        if !any && rng.bernoulli(0.9) {
            f.push(format!("kw{g}.{}", rng.below(3)));
        }
        if rng.bernoulli(c.confound) {
            let o = (g + 1 + rng.below(g_n - 1)) % g_n;
            f.push(format!("kw{o}.{}", rng.below(3)));
        }
        tasks.push(TaskSpec { domain: d, topic: g, features: f });
        draws.push(Draws {
            n_plan: normal(&mut rng),
            u_c1: rng.f64(),
            u_flag: rng.f64(),
            u_retry: rng.f64(),
            n_q: normal(&mut rng),
            u_habit: rng.f64(),
            u_appr: rng.f64(),
            u_center: rng.f64(),
            emb: [normal(&mut rng), normal(&mut rng), normal(&mut rng)],
        });
    }
    // keyword maps from the TRUE (pre-drift) per-stage optimum, with traps/errors
    let mut order: Vec<usize> = (0..g_n).collect();
    for i in (1..g_n).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let trap: Vec<bool> = (0..g_n).map(|g| order.iter().take(c.traps).any(|&x| x == g)).collect();
    let mut kw: [HashMap<String, usize>; 3] = [HashMap::new(), HashMap::new(), HashMap::new()];
    for stage in 0..3 {
        for g in 0..g_n {
            let d = g / tpd;
            let score: Vec<f64> = (0..a_n)
                .map(|a| match stage {
                    0 => plan[a][d],
                    1 => build[a][g],
                    _ => sens[a][d] - fa[a][d],
                })
                .collect();
            let mut idx: Vec<usize> = (0..a_n).collect();
            idx.sort_by(|&x, &y| score[y].partial_cmp(&score[x]).unwrap());
            let (best, second) = (idx[0], idx[1]);
            for j in 0..3 {
                let a = if trap[g] {
                    loop {
                        let a = rng.below(a_n);
                        if a != best && a != second {
                            break a;
                        }
                    }
                } else if rng.f64() < c.map_err {
                    rng.below(a_n)
                } else {
                    best
                };
                kw[stage].insert(format!("kw{g}.{j}"), a);
            }
        }
    }
    World { cfg: c.clone(), build, plan, sens, fa, crit_skill, aff, sig, centers, tasks, draws, kw }
}

impl World {
    fn hid(&self, a: usize, i: usize) -> usize {
        match self.cfg.drift_at {
            Some(t0) if i >= t0 => (a + 1) % self.cfg.agents,
            _ => a,
        }
    }
}

/// Novelty of solution `x` against the window: mean of clamp(dist / 1.6, 0, 1); 0.7 for an empty window.
pub fn novelty(window: &VecDeque<[f64; 3]>, x: &[f64; 3]) -> f64 {
    if window.is_empty() {
        return 0.7;
    }
    let s: f64 = window
        .iter()
        .map(|y| {
            let d = ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt();
            (d / DIST_SCALE).clamp(0.0, 1.0)
        })
        .sum();
    s / window.len() as f64
}

pub fn dom_of(task: &Task) -> usize {
    for f in &task.features {
        if let Some(s) = f.strip_prefix("dom:") {
            return s.parse().unwrap_or(0).min(N_DOMAINS - 1);
        }
    }
    0
}

// ---------------------------------------------------------------------------------------------
// Policy traits

pub trait BuildPolicy {
    /// returns (visible agent, optional forced approach)
    fn act(&mut self, task: &Task, rng: &mut Rng) -> (usize, Option<usize>);
    /// reward = what a plain router sees; quality = quality score separate from novelty
    fn feedback(&mut self, task: &Task, agent: usize, approach: usize, reward: f64, quality: f64);
}

pub trait CriticPolicy {
    /// None = skip the critique stage
    fn pick(&mut self, task: &Task, rng: &mut Rng) -> Option<usize>;
    /// `critic` None = stage was skipped; `reward` = net value of the task after the stage (or v0 if skipped);
    /// `verdict_ok` = verdict matched ground truth (correctness domains only).
    fn feedback(&mut self, task: &Task, critic: Option<usize>, reward: f64, verdict_ok: Option<bool>);
}

pub struct BoxR(pub Box<dyn Router>);
impl Router for BoxR {
    fn name(&self) -> String {
        self.0.name()
    }
    fn select(&mut self, t: &Task, c: &[usize], r: &mut Rng) -> usize {
        self.0.select(t, c, r)
    }
    fn update(&mut self, t: &Task, a: usize, rw: f64) {
        self.0.update(t, a, rw)
    }
}

/// Domain-hierarchical wrapper: one independent sub-router per domain.
pub struct HierRouter {
    subs: Vec<Box<dyn Router>>,
}
impl HierRouter {
    pub fn new(mut f: impl FnMut(usize) -> Box<dyn Router>) -> Self {
        HierRouter { subs: (0..N_DOMAINS).map(|d| f(d)).collect() }
    }
}
impl Router for HierRouter {
    fn name(&self) -> String {
        format!("hier({})", self.subs[0].name())
    }
    fn select(&mut self, t: &Task, c: &[usize], r: &mut Rng) -> usize {
        self.subs[dom_of(t)].select(t, c, r)
    }
    fn update(&mut self, t: &Task, a: usize, rw: f64) {
        self.subs[dom_of(t)].update(t, a, rw)
    }
}

pub struct PlainBuild(pub Box<dyn Router>, pub usize);
impl BuildPolicy for PlainBuild {
    fn act(&mut self, task: &Task, rng: &mut Rng) -> (usize, Option<usize>) {
        let c: Vec<usize> = (0..self.1).collect();
        (self.0.select(task, &c, rng), None)
    }
    fn feedback(&mut self, task: &Task, agent: usize, _approach: usize, reward: f64, _q: f64) {
        self.0.update(task, agent, reward)
    }
}

pub struct HierBuild {
    subs: Vec<Box<dyn BuildPolicy>>,
}
impl HierBuild {
    pub fn new(mut f: impl FnMut(usize) -> Box<dyn BuildPolicy>) -> Self {
        HierBuild { subs: (0..N_DOMAINS).map(|d| f(d)).collect() }
    }
}
impl BuildPolicy for HierBuild {
    fn act(&mut self, t: &Task, r: &mut Rng) -> (usize, Option<usize>) {
        self.subs[dom_of(t)].act(t, r)
    }
    fn feedback(&mut self, t: &Task, a: usize, ap: usize, rw: f64, q: f64) {
        self.subs[dom_of(t)].feedback(t, a, ap, rw, q)
    }
}

/// creative domains -> `creative`, others -> `other`
pub struct CreativeSplit {
    pub creative: Box<dyn BuildPolicy>,
    pub other: Box<dyn BuildPolicy>,
}
impl BuildPolicy for CreativeSplit {
    fn act(&mut self, t: &Task, r: &mut Rng) -> (usize, Option<usize>) {
        if CREATIVE[dom_of(t)] { self.creative.act(t, r) } else { self.other.act(t, r) }
    }
    fn feedback(&mut self, t: &Task, a: usize, ap: usize, rw: f64, q: f64) {
        if CREATIVE[dom_of(t)] { self.creative.feedback(t, a, ap, rw, q) } else { self.other.feedback(t, a, ap, rw, q) }
    }
}

// ---------------------------------------------------------------------------------------------
// Quality-diversity bandit (MAP-Elites-style niche archive) + novelty bonus

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoveltyScope {
    Off,
    /// bonus only in creative domains
    Creative,
    /// bonus in every domain (control)
    All,
}

#[derive(Clone, Debug)]
pub struct QdConfig {
    /// arms are (agent, approach) niches; false = arms are agents (approach is the agent's habit)
    pub niche: bool,
    pub scope: NoveltyScope,
    /// assumed novelty weight lam_hat
    pub lam: f64,
    /// assumed novelty drop of a repeated approach: nov_hat(k) = 1 - rho * freq_k
    pub rho: f64,
    /// window of recent approach labels per domain used for nov_hat
    pub window: usize,
    /// EMA rate of the elite quality estimate (floor of 1/n)
    pub alpha: f64,
    /// optimism c / sqrt(1 + n)
    pub ucb: f64,
    pub eps: f64,
    pub q0: f64,
}

impl QdConfig {
    pub fn default_niche() -> Self {
        QdConfig { niche: true, scope: NoveltyScope::Creative, lam: 0.6, rho: 0.65, window: 10, alpha: 0.1, ucb: 0.1, eps: 0.05, q0: 0.5 }
    }
}

pub struct QdBuild {
    pub cfg: QdConfig,
    a: usize,
    k: usize,
    /// [domain][arm] elite quality estimate and visit count
    q: Vec<Vec<f64>>,
    n: Vec<Vec<f64>>,
    /// [domain][agent][approach] habit counts
    cnt: Vec<Vec<Vec<f64>>>,
    win: Vec<VecDeque<usize>>,
}

impl QdBuild {
    pub fn new(a: usize, k: usize, cfg: QdConfig) -> Self {
        let arms = if cfg.niche { a * k } else { a };
        QdBuild {
            q: vec![vec![cfg.q0; arms]; N_DOMAINS],
            n: vec![vec![0.0; arms]; N_DOMAINS],
            cnt: vec![vec![vec![0.0; k]; a]; N_DOMAINS],
            win: vec![VecDeque::new(); N_DOMAINS],
            cfg,
            a,
            k,
        }
    }
    fn lam_d(&self, d: usize) -> f64 {
        match self.cfg.scope {
            NoveltyScope::Off => 0.0,
            NoveltyScope::Creative => if CREATIVE[d] { self.cfg.lam } else { 0.0 },
            NoveltyScope::All => self.cfg.lam,
        }
    }
    fn freq(&self, d: usize, k: usize) -> f64 {
        if self.win[d].is_empty() {
            return 0.0;
        }
        self.win[d].iter().filter(|&&x| x == k).count() as f64 / self.win[d].len() as f64
    }
    fn nov_hat(&self, d: usize, arm: usize) -> f64 {
        if self.cfg.niche {
            1.0 - self.cfg.rho * self.freq(d, arm % self.k)
        } else {
            let c = &self.cnt[d][arm];
            let tot: f64 = c.iter().sum::<f64>() + 0.5 * self.k as f64;
            (0..self.k).map(|k| (c[k] + 0.5) / tot * (1.0 - self.cfg.rho * self.freq(d, k))).sum()
        }
    }
    pub fn score(&self, d: usize, arm: usize) -> f64 {
        let lam = self.lam_d(d);
        self.q[d][arm] * (1.0 - lam + lam * self.nov_hat(d, arm)) + self.cfg.ucb / (1.0 + self.n[d][arm]).sqrt()
    }
    pub fn n_arms(&self) -> usize {
        self.q[0].len()
    }
    /// number of distinct niches (agent, approach) with at least one visit in domain d
    pub fn covered(&self, d: usize) -> usize {
        let mut s = 0;
        for a in 0..self.a {
            for k in 0..self.k {
                if self.cnt[d][a][k] > 0.0 {
                    s += 1;
                }
            }
        }
        s
    }
}

impl BuildPolicy for QdBuild {
    fn act(&mut self, task: &Task, rng: &mut Rng) -> (usize, Option<usize>) {
        let d = dom_of(task);
        let arms = self.n_arms();
        let arm = if rng.bernoulli(self.cfg.eps) {
            rng.below(arms)
        } else {
            let mut best = (f64::NEG_INFINITY, 0);
            for arm in 0..arms {
                let s = self.score(d, arm);
                if s > best.0 + 1e-12 {
                    best = (s, arm);
                }
            }
            best.1
        };
        if self.cfg.niche { (arm / self.k, Some(arm % self.k)) } else { (arm, None) }
    }
    fn feedback(&mut self, task: &Task, agent: usize, approach: usize, _reward: f64, quality: f64) {
        let d = dom_of(task);
        let arm = if self.cfg.niche { agent * self.k + approach } else { agent };
        self.n[d][arm] += 1.0;
        let rate = self.cfg.alpha.max(1.0 / self.n[d][arm]);
        self.q[d][arm] += rate * (quality - self.q[d][arm]);
        self.cnt[d][agent][approach] += 1.0;
        self.win[d].push_back(approach);
        while self.win[d].len() > self.cfg.window {
            self.win[d].pop_front();
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Critic policies

pub struct NoCritic;
impl CriticPolicy for NoCritic {
    fn pick(&mut self, _t: &Task, _r: &mut Rng) -> Option<usize> {
        None
    }
    fn feedback(&mut self, _t: &Task, _c: Option<usize>, _r: f64, _v: Option<bool>) {}
}

pub struct RandomCritic(pub usize);
impl CriticPolicy for RandomCritic {
    fn pick(&mut self, _t: &Task, r: &mut Rng) -> Option<usize> {
        Some(r.below(self.0))
    }
    fn feedback(&mut self, _t: &Task, _c: Option<usize>, _r: f64, _v: Option<bool>) {}
}

pub struct FixedCritic(pub usize);
impl CriticPolicy for FixedCritic {
    fn pick(&mut self, _t: &Task, _r: &mut Rng) -> Option<usize> {
        Some(self.0)
    }
    fn feedback(&mut self, _t: &Task, _c: Option<usize>, _r: f64, _v: Option<bool>) {}
}

pub struct RouterCritic(pub Box<dyn Router>, pub usize);
impl CriticPolicy for RouterCritic {
    fn pick(&mut self, t: &Task, r: &mut Rng) -> Option<usize> {
        let c: Vec<usize> = (0..self.1).collect();
        Some(self.0.select(t, &c, r))
    }
    fn feedback(&mut self, t: &Task, c: Option<usize>, reward: f64, _v: Option<bool>) {
        if let Some(c) = c {
            self.0.update(t, c, clamp01(reward));
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CriticMode {
    /// learn from the net task value after the stage; a "skip" arm is available
    Final,
    /// learn each critic's verdict accuracy (needs ground truth revealed later; correctness domains)
    Verdict,
}

#[derive(Clone, Debug)]
pub struct LcConfig {
    pub mode: CriticMode,
    /// per-observation discount of all pseudo-counts of the domain (1.0 = none)
    pub gamma: f64,
    /// Verdict mode: skip critics when the best posterior accuracy mean is below this (after 10 obs)
    pub gate: f64,
    /// pseudo-count mass of a keyword-map warm prior over critics per domain (0 = none)
    pub prior_k: f64,
}

/// Critic chosen by learned per-domain reliability (Thompson sampling on Beta posteriors).
pub struct LearnedCritic {
    pub cfg: LcConfig,
    a: usize,
    s: Vec<Vec<f64>>,
    f: Vec<Vec<f64>>,
    obs: Vec<f64>,
}
impl LearnedCritic {
    pub fn new(a: usize, cfg: LcConfig) -> Self {
        LearnedCritic { cfg, a, s: vec![vec![0.0; a + 1]; N_DOMAINS], f: vec![vec![0.0; a + 1]; N_DOMAINS], obs: vec![0.0; N_DOMAINS] }
    }
    fn final_mode(&self, d: usize) -> bool {
        self.cfg.mode == CriticMode::Final || CREATIVE[d]
    }
    pub fn mean(&self, d: usize, arm: usize) -> f64 {
        (1.0 + self.s[d][arm]) / (2.0 + self.s[d][arm] + self.f[d][arm])
    }
    /// Warm prior: the keyword map's votes for the topics of each domain become optimistic pseudo-counts
    /// (mean 0.8) on the voted critics. Uses only the same keyword map the keyword/warm baselines get.
    pub fn with_keyword_prior(mut self, map: &HashMap<String, usize>, topics_per_domain: usize) -> Self {
        if self.cfg.prior_k <= 0.0 {
            return self;
        }
        for d in 0..N_DOMAINS {
            let mut votes = vec![0.0; self.a];
            let mut tot = 0.0;
            for g in d * topics_per_domain..(d + 1) * topics_per_domain {
                for j in 0..3 {
                    if let Some(&a) = map.get(&format!("kw{g}.{j}")) {
                        votes[a.min(self.a - 1)] += 1.0;
                        tot += 1.0;
                    }
                }
            }
            if tot > 0.0 {
                for a in 0..self.a {
                    let m = self.cfg.prior_k * votes[a] / tot;
                    self.s[d][a] += 0.8 * m;
                    self.f[d][a] += 0.2 * m;
                }
            }
        }
        self
    }
}
impl CriticPolicy for LearnedCritic {
    fn pick(&mut self, t: &Task, rng: &mut Rng) -> Option<usize> {
        let d = dom_of(t);
        let arms = if self.final_mode(d) { self.a + 1 } else { self.a };
        let mut best = (f64::NEG_INFINITY, 0);
        for arm in 0..arms {
            let x = sample_beta(1.0 + self.s[d][arm], 1.0 + self.f[d][arm], rng);
            if x > best.0 {
                best = (x, arm);
            }
        }
        if !self.final_mode(d) && self.cfg.gate > 0.0 && self.obs[d] >= 10.0 {
            let bm = (0..self.a).map(|a| self.mean(d, a)).fold(0.0, f64::max);
            if bm < self.cfg.gate {
                return None;
            }
        }
        if best.1 == self.a { None } else { Some(best.1) }
    }
    fn feedback(&mut self, t: &Task, critic: Option<usize>, reward: f64, verdict_ok: Option<bool>) {
        let d = dom_of(t);
        let arm = critic.unwrap_or(self.a);
        let (succ, usable) = if self.final_mode(d) {
            (clamp01(reward), true)
        } else {
            match (critic, verdict_ok) {
                (Some(_), Some(ok)) => (if ok { 1.0 } else { 0.0 }, true),
                _ => (0.0, false),
            }
        };
        if !usable {
            return;
        }
        if self.cfg.gamma < 1.0 {
            for x in self.s[d].iter_mut().chain(self.f[d].iter_mut()) {
                *x *= self.cfg.gamma;
            }
        }
        self.s[d][arm] += succ;
        self.f[d][arm] += 1.0 - succ;
        self.obs[d] += 1.0;
    }
}

pub struct HierCritic {
    subs: Vec<Box<dyn CriticPolicy>>,
}
impl HierCritic {
    pub fn new(mut f: impl FnMut(usize) -> Box<dyn CriticPolicy>) -> Self {
        HierCritic { subs: (0..N_DOMAINS).map(|d| f(d)).collect() }
    }
}
impl CriticPolicy for HierCritic {
    fn pick(&mut self, t: &Task, r: &mut Rng) -> Option<usize> {
        self.subs[dom_of(t)].pick(t, r)
    }
    fn feedback(&mut self, t: &Task, c: Option<usize>, rw: f64, v: Option<bool>) {
        self.subs[dom_of(t)].feedback(t, c, rw, v)
    }
}

// ---------------------------------------------------------------------------------------------
// Fair baseline kinds

#[derive(Clone, Copy, Debug)]
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
    WarmHebR(f64, EvK),
}

impl Kind {
    pub fn label(&self) -> String {
        match self {
            Kind::Keyword => "keyword".into(),
            Kind::Eps(g, e) => format!("eps-greedy(g={g},e={e})"),
            Kind::Pts(g) => format!("pooled-ts(g={g})"),
            Kind::HebR => "hebbian-robust".into(),
            Kind::WarmHebR(k, e) => format!("warm(hebbian-robust,k={k},{e:?})"),
        }
    }
}

pub fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

pub fn make_router(kind: &Kind, w: &World, stage: usize) -> Box<dyn Router> {
    let a = w.cfg.agents;
    match kind {
        Kind::Keyword => Box::new(KeywordRouter::new(w.kw[stage].clone(), a)),
        Kind::Eps(g, e) => Box::new(EpsGreedyRouter::new(a, *g, *e)),
        Kind::Pts(g) => Box::new(PooledThompsonRouter::new(a, *g)),
        Kind::HebR => Box::new(HebbianRouter::new(a, heb_robust_cfg())),
        Kind::WarmHebR(k, e) => {
            let ev: Evidence<BoxR> = match e {
                EvK::Set => Evidence::FeatureSet,
                EvK::Mean => Evidence::MeanFeature,
                EvK::FailSet => Evidence::SetFailures,
            };
            Box::new(Warm::new(BoxR(Box::new(HebbianRouter::new(a, heb_robust_cfg()))), w.kw[stage].clone(), a, *k).with_evidence(ev))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Policy specs

#[derive(Clone, Debug)]
pub enum BuildSpec {
    Fair { kind: Kind, hier: bool },
    /// creative domains use a QdBuild; other domains use the fair `other` kind (flat or per-domain)
    Qd { cfg: QdConfig, other: Kind, hier: bool },
}

#[derive(Clone, Debug)]
pub enum CriticSpec {
    None,
    Random,
    Fixed(usize),
    Fair { kind: Kind, hier: bool },
    Learned(LcConfig),
}

#[derive(Clone, Debug)]
pub struct PolicySpec {
    pub name: String,
    pub plan: Kind,
    pub plan_hier: bool,
    pub build: BuildSpec,
    pub critic: CriticSpec,
}

impl PolicySpec {
    /// a fair baseline: same learner in every stage, flat, critic always on.
    pub fn fair(kind: &Kind) -> Self {
        PolicySpec {
            name: kind.label(),
            plan: kind.clone(),
            plan_hier: false,
            build: BuildSpec::Fair { kind: kind.clone(), hier: false },
            critic: CriticSpec::Fair { kind: kind.clone(), hier: false },
        }
    }
}

pub struct Policy {
    pub name: String,
    pub plan: Box<dyn Router>,
    pub build: Box<dyn BuildPolicy>,
    pub critic: Box<dyn CriticPolicy>,
}

fn build_policy_for(b: &BuildSpec, w: &World) -> Box<dyn BuildPolicy> {
    let a = w.cfg.agents;
    let fair = |kind: &Kind, hier: bool| -> Box<dyn BuildPolicy> {
        if hier {
            Box::new(HierBuild::new(|_| Box::new(PlainBuild(make_router(kind, w, 1), a))))
        } else {
            Box::new(PlainBuild(make_router(kind, w, 1), a))
        }
    };
    match b {
        BuildSpec::Fair { kind, hier } => fair(kind, *hier),
        BuildSpec::Qd { cfg, other, hier } => {
            if cfg.scope == NoveltyScope::All {
                // control: the mechanism is applied to EVERY domain
                Box::new(QdBuild::new(a, w.cfg.approaches, cfg.clone()))
            } else {
                Box::new(CreativeSplit { creative: Box::new(QdBuild::new(a, w.cfg.approaches, cfg.clone())), other: fair(other, *hier) })
            }
        }
    }
}

pub fn build_policy(spec: &PolicySpec, w: &World) -> Policy {
    let a = w.cfg.agents;
    let plan: Box<dyn Router> =
        if spec.plan_hier { Box::new(HierRouter::new(|_| make_router(&spec.plan, w, 0))) } else { make_router(&spec.plan, w, 0) };
    let critic: Box<dyn CriticPolicy> = match &spec.critic {
        CriticSpec::None => Box::new(NoCritic),
        CriticSpec::Random => Box::new(RandomCritic(a)),
        CriticSpec::Fixed(i) => Box::new(FixedCritic(*i)),
        CriticSpec::Fair { kind, hier } => {
            if *hier {
                Box::new(HierCritic::new(|_| Box::new(RouterCritic(make_router(kind, w, 2), a))))
            } else {
                Box::new(RouterCritic(make_router(kind, w, 2), a))
            }
        }
        CriticSpec::Learned(cfg) => Box::new(LearnedCritic::new(a, cfg.clone()).with_keyword_prior(&w.kw[2], w.cfg.topics_per_domain)),
    };
    Policy { name: spec.name.clone(), plan, build: build_policy_for(&spec.build, w), critic }
}

// ---------------------------------------------------------------------------------------------
// Simulation

#[derive(Clone, Debug)]
pub struct RunOut {
    pub values: Vec<f64>,
    pub doms: Vec<u8>,
    /// mean realised novelty of creative solutions
    pub creative_novelty: f64,
    /// Shannon entropy (nats) of the approach histogram of creative solutions
    pub creative_entropy: f64,
}

impl RunOut {
    pub fn mean_where(&self, f: impl Fn(usize, usize) -> bool) -> f64 {
        let (mut s, mut n) = (0.0, 0.0);
        for (i, (&v, &d)) in self.values.iter().zip(&self.doms).enumerate() {
            if f(i, d as usize) {
                s += v;
                n += 1.0;
            }
        }
        if n > 0.0 { s / n } else { f64::NAN }
    }
    pub fn overall(&self) -> f64 {
        self.mean_where(|_, _| true)
    }
    pub fn creative(&self) -> f64 {
        self.mean_where(|_, d| CREATIVE[d])
    }
    pub fn correctness(&self) -> f64 {
        self.mean_where(|_, d| !CREATIVE[d])
    }
    pub fn early(&self, h: usize) -> f64 {
        self.mean_where(|i, _| i < h)
    }
}

pub fn run(w: &World, pol: &mut Policy, seed: u64) -> RunOut {
    let c = &w.cfg;
    let cands: Vec<usize> = (0..c.agents).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut wins: Vec<VecDeque<[f64; 3]>> = vec![VecDeque::new(); N_DOMAINS];
    let (mut values, mut doms) = (Vec::with_capacity(c.n_tasks), Vec::with_capacity(c.n_tasks));
    let (mut nov_sum, mut nov_n) = (0.0, 0.0);
    let mut hist = vec![0.0f64; c.approaches];
    for (i, t) in w.tasks.iter().enumerate() {
        let (d, g) = (t.domain, t.topic);
        let dr = &w.draws[i];
        let mk = |stage: &str| Task { id: i as u64, task_type: Some(stage.into()), features: t.features.clone() };
        // PLAN
        let pt = mk("plan");
        let pa = pol.plan.select(&pt, &cands, &mut rng);
        let hp = w.hid(pa, i);
        let qp = clamp01(w.plan[hp][d] + c.noise * dr.n_plan);
        pol.plan.update(&pt, pa, qp);
        let pf = 0.7 + 0.3 * qp;
        // BUILD
        let bt = mk("build");
        let (ba, forced) = pol.build.act(&bt, &mut rng);
        let hb = w.hid(ba, i);
        let k = match forced {
            Some(k) => k,
            None => {
                if dr.u_habit < HABIT {
                    w.sig[hb][d]
                } else {
                    let o = (dr.u_appr * (c.approaches - 1) as f64) as usize;
                    if o >= w.sig[hb][d] { o + 1 } else { o }
                }
            }
        };
        let k = k.min(c.approaches - 1);
        let (v0, reward_b, quality, p_eff, correct1);
        if CREATIVE[d] {
            let q = clamp01(w.build[hb][g] * w.aff[hb][d][k] * pf + c.noise * dr.n_q);
            let ctr = if c.labels_informative { w.centers[k] } else { w.centers[((dr.u_center * c.approaches as f64) as usize).min(c.approaches - 1)] };
            let x = [ctr[0] + 0.25 * dr.emb[0], ctr[1] + 0.25 * dr.emb[1], ctr[2] + 0.25 * dr.emb[2]];
            let nov = novelty(&wins[d], &x);
            wins[d].push_back(x);
            while wins[d].len() > c.window {
                wins[d].pop_front();
            }
            nov_sum += nov;
            nov_n += 1.0;
            hist[k] += 1.0;
            let v = q * (1.0 - c.lam + c.lam * nov);
            v0 = v;
            reward_b = v;
            quality = q;
            p_eff = 0.0;
            correct1 = false;
        } else {
            let p = clamp01(w.build[hb][g] * w.aff[hb][d][k] * pf);
            let ok = dr.u_c1 < p;
            v0 = if ok { 1.0 } else { 0.0 };
            reward_b = v0;
            quality = v0;
            p_eff = p;
            correct1 = ok;
        }
        pol.build.feedback(&bt, ba, k, reward_b, quality);
        // CRITIQUE
        let ct = mk("critic");
        let cr = pol.critic.pick(&ct, &mut rng);
        let (val, verdict_ok);
        match cr {
            None => {
                val = v0;
                verdict_ok = None;
                pol.critic.feedback(&ct, None, clamp01(v0), None);
            }
            Some(cv) => {
                let hc = w.hid(cv, i);
                if CREATIVE[d] {
                    let v = clamp01(v0 + 0.3 * w.crit_skill[hc][d] * (1.0 - v0)) - c.critic_cost;
                    val = v;
                    verdict_ok = None;
                } else {
                    let flagged = if !correct1 { dr.u_flag < w.sens[hc][d] } else { dr.u_flag < w.fa[hc][d] };
                    let fin = if flagged { dr.u_retry < p_eff } else { correct1 };
                    val = (if fin { 1.0 } else { 0.0 }) - c.critic_cost;
                    verdict_ok = Some(flagged == !correct1);
                }
                pol.critic.feedback(&ct, Some(cv), clamp01(val), verdict_ok);
            }
        }
        values.push(val);
        doms.push(d as u8);
    }
    let tot: f64 = hist.iter().sum();
    let ent = if tot > 0.0 { -hist.iter().filter(|&&x| x > 0.0).map(|&x| (x / tot) * (x / tot).ln()).sum::<f64>() } else { 0.0 };
    RunOut { values, doms, creative_novelty: if nov_n > 0.0 { nov_sum / nov_n } else { 0.0 }, creative_entropy: ent }
}

/// build the world + policy and run (deterministic in (spec, cfg, seed)).
pub fn run_spec(spec: &PolicySpec, cfg: &WorldCfg, seed: u64) -> RunOut {
    let w = gen_world(seed, cfg);
    let mut p = build_policy(spec, &w);
    run(&w, &mut p, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(d: usize) -> Task {
        Task { id: 0, task_type: Some("build".into()), features: vec![format!("dom:{d}")] }
    }

    #[test]
    fn world_is_deterministic() {
        let c = WorldCfg::base("t");
        let a = gen_world(5, &c);
        let b = gen_world(5, &c);
        assert_eq!(a.tasks[10].features, b.tasks[10].features);
        assert_eq!(a.build, b.build);
        let o1 = run_spec(&PolicySpec::fair(&Kind::Pts(0.99)), &c, 5);
        let o2 = run_spec(&PolicySpec::fair(&Kind::Pts(0.99)), &c, 5);
        assert_eq!(o1.values, o2.values);
        let o3 = run_spec(&PolicySpec::fair(&Kind::Pts(0.99)), &c, 6);
        assert_ne!(o1.values, o3.values);
    }

    #[test]
    fn novelty_basics() {
        let mut w = VecDeque::new();
        assert!((novelty(&w, &[0.0; 3]) - 0.7).abs() < 1e-12);
        w.push_back([0.0; 3]);
        assert!(novelty(&w, &[0.0; 3]) < 1e-12);
        assert!((novelty(&w, &[5.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn values_in_range_and_drift_rotates() {
        let mut c = WorldCfg::base("t");
        c.drift_at = Some(100);
        c.n_tasks = 300;
        let w = gen_world(1, &c);
        assert_eq!(w.hid(3, 50), 3);
        assert_eq!(w.hid(3, 150), 4);
        assert_eq!(w.hid(7, 150), 0);
        let o = run_spec(&PolicySpec::fair(&Kind::HebR), &c, 1);
        assert!(o.values.iter().all(|v| *v >= -0.03 && *v <= 1.0));
        assert_eq!(o.values.len(), 300);
    }

    #[test]
    fn qd_covers_more_niches_with_novelty_than_without() {
        // identical quality everywhere in domain 2 (creative): novelty bonus must spread over approaches,
        // no bonus collapses onto one niche (ties -> lowest arm, tiny eps).
        let run_cfg = |scope| {
            let cfg = QdConfig { scope, eps: 0.0, ucb: 0.0, ..QdConfig::default_niche() };
            let mut q = QdBuild::new(4, 4, cfg);
            let mut rng = Rng::new(1);
            let t = task(2);
            let mut seen = std::collections::HashSet::new();
            for _ in 0..200 {
                let (a, k) = q.act(&t, &mut rng);
                let k = k.unwrap();
                seen.insert(k);
                q.feedback(&t, a, k, 0.5, 0.5);
            }
            seen.len()
        };
        assert!(run_cfg(NoveltyScope::Creative) >= 3, "novelty should spread approaches");
        assert_eq!(run_cfg(NoveltyScope::Off), 1, "no bonus -> one niche (greedy)");
    }

    #[test]
    fn qd_scope_creative_ignores_correctness_domains() {
        // domain 0 (coding) is non-creative: Creative scope must behave exactly like Off there.
        let mk = |scope| QdBuild::new(3, 3, QdConfig { scope, eps: 0.0, ..QdConfig::default_niche() });
        let (mut a, mut b) = (mk(NoveltyScope::Creative), mk(NoveltyScope::Off));
        let t = task(0);
        let (mut r1, mut r2) = (Rng::new(2), Rng::new(2));
        for i in 0..60 {
            let x = a.act(&t, &mut r1);
            let y = b.act(&t, &mut r2);
            assert_eq!(x, y);
            let q = 0.3 + 0.01 * (i % 7) as f64;
            a.feedback(&t, x.0, x.1.unwrap(), q, q);
            b.feedback(&t, y.0, y.1.unwrap(), q, q);
        }
        // but scope All differs from Off on a domain with approach history
        let mut c = mk(NoveltyScope::All);
        let mut rr = Rng::new(2);
        let mut ks = std::collections::HashSet::new();
        for _ in 0..80 {
            let (ag, k) = c.act(&t, &mut rr);
            ks.insert(k.unwrap());
            c.feedback(&t, ag, k.unwrap(), 0.5, 0.5);
        }
        assert!(ks.len() >= 2);
    }

    #[test]
    fn learned_critic_prefers_accurate_critic() {
        let mut lc = LearnedCritic::new(4, LcConfig { mode: CriticMode::Verdict, gamma: 1.0, gate: 0.0, prior_k: 0.0 });
        let t = task(0);
        let mut rng = Rng::new(3);
        // critic 2 is 90% accurate, others 50%
        for i in 0..400 {
            let c = lc.pick(&t, &mut rng).unwrap();
            let acc = if c == 2 { 0.9 } else { 0.5 };
            let ok = rng.f64() < acc;
            lc.feedback(&t, Some(c), 0.0, Some(ok));
            let _ = i;
        }
        let best = (0..4).max_by(|&a, &b| lc.mean(0, a).partial_cmp(&lc.mean(0, b)).unwrap()).unwrap();
        assert_eq!(best, 2);
        // gate: all-bad critics -> skip
        let mut g = LearnedCritic::new(2, LcConfig { mode: CriticMode::Verdict, gamma: 1.0, gate: 0.6, prior_k: 0.0 });
        for _ in 0..40 {
            g.feedback(&t, Some(0), 0.0, Some(false));
            g.feedback(&t, Some(1), 0.0, Some(false));
        }
        assert!(g.pick(&t, &mut rng).is_none());
    }

    #[test]
    fn hier_isolates_domains() {
        let mut h = HierRouter::new(|_| Box::new(EpsGreedyRouter::new(3, 1.0, 0.0)));
        let (t0, t1) = (task(0), task(1));
        for _ in 0..20 {
            for a in 0..3 {
                h.update(&t0, a, if a == 2 { 1.0 } else { 0.0 });
                h.update(&t1, a, if a == 0 { 1.0 } else { 0.0 });
            }
        }
        let mut rng = Rng::new(1);
        let c = [0usize, 1, 2];
        // each domain learned its own favourite; a flat router would have mixed them
        assert_eq!(h.select(&t0, &c, &mut rng), 2);
        assert_eq!(h.select(&t1, &c, &mut rng), 0);
    }

    #[test]
    fn good_critic_helps_bad_critic_hurts_correctness() {
        let mut c = WorldCfg::base("t");
        c.n_tasks = 600;
        let mk = |critic: CriticSpec| PolicySpec {
            name: "x".into(),
            plan: Kind::Pts(0.99),
            plan_hier: false,
            build: BuildSpec::Fair { kind: Kind::Pts(0.99), hier: false },
            critic,
        };
        let (mut none, mut best, mut worst) = (0.0, 0.0, 0.0);
        for s in 1..=6u64 {
            let w = gen_world(s, &c);
            // pick best / worst critic by sens-fa on domain 0 (coding)
            let mut idx: Vec<usize> = (0..c.agents).collect();
            idx.sort_by(|&a, &b| (w.sens[b][0] - w.fa[b][0]).partial_cmp(&(w.sens[a][0] - w.fa[a][0])).unwrap());
            let (b, wr) = (idx[0], idx[c.agents - 1]);
            let m = |spec: &PolicySpec| run_spec(spec, &c, s).mean_where(|_, d| d == 0);
            none += m(&mk(CriticSpec::None));
            best += m(&mk(CriticSpec::Fixed(b)));
            worst += m(&mk(CriticSpec::Fixed(wr)));
        }
        assert!(best > none, "best critic should beat none: {best} vs {none}");
        assert!(worst < best);
    }

    #[test]
    fn rigged_world_has_equal_critics() {
        let mut c = WorldCfg::base("rig");
        c.critic_equal = true;
        let w = gen_world(3, &c);
        for a in 1..c.agents {
            assert_eq!(w.sens[a][1], w.sens[0][1]);
            assert_eq!(w.fa[a][1], w.fa[0][1]);
        }
    }
}
