//! H9 common: worlds (fixed BEFORE any policy ran), previous-deployment pools, warm-start priors
//! (hierarchical shrinkage + learned trust), candidate/baseline policy factory, run loop, metrics.
//! EVERYTHING is a seeded SIMULATION written by me (synthetic-until-validated-on-real-traffic).
//! Generator A = sim::World / f3 BigWorld (reused read-only). Generator B = my own g5-style
//! multi-domain world (5 domains x 4 sub-skills, three different agent-pool structures).
#![allow(dead_code)]

use std::collections::HashMap;

use neural_mesh::baselines::{sample_beta, EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::defaults::{recommended_inner, recommended_router};
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::Scenario;

use crate::f3::Env;

pub const N_TASKS: usize = 2000;
pub const N_POOL_TASKS: usize = 1000;
pub const POOL_BASE: u64 = 1_000_000;

// ===================================================================================== worlds
pub const NDOM: usize = 5;
pub const NSUBPD: usize = 4;
pub const NSUB: usize = NDOM * NSUBPD;
const DOMS: [&str; NDOM] = ["design", "business", "marketing", "coding", "math"];
const SUBS: [[&str; NSUBPD]; NDOM] = [
    ["ux-flow", "visual-ui", "design-system", "accessibility"],
    ["financial-model", "go-to-market", "ops-plan", "pricing"],
    ["copywriting", "seo", "campaign", "social"],
    ["rust", "sql", "debug", "api"],
    ["algebra", "proof", "stats", "calculus"],
];
const FLUFF: [&str; 6] = ["please", "urgent", "review", "draft", "client", "iterate"];

fn hrand(a: u64, b: u64, c: u64) -> f64 {
    let mut z = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ c.wrapping_mul(0x94D0_49BB_1331_11EB) ^ 0x1234_5678_9ABC;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}
fn rnorm(rng: &mut Rng) -> f64 {
    let (u1, u2) = (rng.f64().max(1e-12), rng.f64());
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoolKind {
    /// 10 agents: 2 generalists, 5 domain specialists, 1 design+marketing creative, 2 sub-skill experts
    Agency,
    /// 20 agents: 3 specialists per domain + 5 generalists
    Big,
    /// 10 agents whose skills come from 3 latent factors (no domain-specialist structure)
    Latent,
    /// 8 identical agents (p = .7)
    Equal,
    /// 10 agents, skills drawn independently per deployment (NO shared population structure)
    RandSkills,
}
impl PoolKind {
    pub fn n_agents(self) -> usize {
        match self {
            PoolKind::Agency | PoolKind::Latent | PoolKind::RandSkills => 10,
            PoolKind::Big => 20,
            PoolKind::Equal => 8,
        }
    }
}

/// Population skill table [NSUB][n_agents]: what ALL deployments of a family share before per-deployment noise.
pub fn pop_table(kind: PoolKind) -> Vec<Vec<f64>> {
    let n = kind.n_agents();
    let mut p = vec![vec![0.5; n]; NSUB];
    for s in 0..NSUB {
        let (d, sub) = (s / NSUBPD, s % NSUBPD);
        for a in 0..n {
            let h = hrand(a as u64, s as u64, 1);
            p[s][a] = match kind {
                PoolKind::Agency => match a {
                    0 => 0.64,
                    1 => 0.68,
                    2..=6 => {
                        if a - 2 == d {
                            0.86 + 0.02 * ((a % 3) as f64)
                        } else {
                            0.38 + 0.08 * h
                        }
                    }
                    7 => {
                        if d == 0 || d == 2 { 0.82 } else { 0.40 + 0.06 * h }
                    }
                    8 => {
                        if d == 3 && sub < 2 { 0.86 } else { 0.48 + 0.04 * h }
                    }
                    _ => {
                        if d == 4 && (sub == 1 || sub == 2) { 0.86 } else { 0.48 + 0.04 * h }
                    }
                },
                PoolKind::Big => {
                    if a < 15 {
                        if a / 3 == d { 0.80 + 0.03 * ((a % 3) as f64) } else { 0.36 + 0.08 * h }
                    } else {
                        0.60 + 0.06 * h
                    }
                }
                PoolKind::Latent => {
                    let mut z = 0.0;
                    for f in 0..3u64 {
                        let zs = 2.0 * hrand(s as u64, f, 11) - 1.0;
                        let ld = 2.0 * hrand(a as u64, f, 12) - 1.0;
                        z += zs * ld;
                    }
                    (0.55 + 0.30 * (1.8 * z).tanh()).clamp(0.2, 0.9)
                }
                PoolKind::Equal => 0.7,
                PoolKind::RandSkills => 0.5,
            };
        }
    }
    p
}

#[derive(Clone, Debug)]
pub struct MdCfg {
    pub pool: PoolKind,
    /// P(a domain's specialist structure is moved to a random other agent) relative to the population
    pub rho: f64,
    /// force every domain to be moved (the prior is WRONG about every domain)
    pub derange: bool,
    pub sigma: f64,
    /// P(a keyword advertises a random wrong agent)
    pub decoy: f64,
    pub drift: bool,
    pub mix: [f64; NDOM],
    /// extra agents unknown to the pool (best on domains 1,3)
    pub extra: usize,
    /// fraction of sub-skill keywords renamed (unseen by the pool)
    pub vocab_ren: f64,
    /// uninformative feedback: the reward the router OBSERVES is an agent-independent coin (p=.5); quality metrics stay true
    pub blind: bool,
}
impl MdCfg {
    pub fn base(pool: PoolKind) -> Self {
        MdCfg { pool, rho: 0.1, derange: false, sigma: 0.05, decoy: 0.3, drift: false, mix: [0.2; NDOM], extra: 0, vocab_ren: 0.0, blind: false }
    }
}

pub struct Md {
    p_pre: Vec<Vec<f64>>,
    p_post: Vec<Vec<f64>>,
    sub_of: Vec<usize>,
    u: Vec<f64>,
    drift_at: usize,
    blind: bool,
}

pub enum Back {
    E(Env),
    M(Md),
}

pub struct W {
    pub back: Back,
    pub tasks: Vec<Task>,
    pub n_agents: usize,
    pub oracle: Vec<f64>,
    pub kw: HashMap<String, usize>,
    pub len: usize,
}

impl W {
    pub fn prob(&self, t: usize, a: usize) -> f64 {
        match &self.back {
            Back::E(e) => e.prob(t, a),
            Back::M(m) => {
                if m.drift_at <= t { m.p_post[m.sub_of[t]][a] } else { m.p_pre[m.sub_of[t]][a] }
            }
        }
    }
    pub fn outcome(&self, t: usize, a: usize) -> f64 {
        match &self.back {
            Back::E(e) => e.outcome(t, a),
            Back::M(m) => {
                let p = if m.blind { 0.5 } else { self.prob(t, a) };
                if m.u[t] < p { 1.0 } else { 0.0 }
            }
        }
    }
    fn finish(back: Back, tasks: Vec<Task>, n_agents: usize, kw: HashMap<String, usize>, len: usize) -> W {
        let mut w = W { back, tasks, n_agents, oracle: vec![], kw, len };
        w.oracle = (0..len).map(|t| (0..n_agents).map(|a| w.prob(t, a)).fold(0.0, f64::max)).collect();
        w
    }
}

#[derive(Clone, Debug)]
pub enum Gen {
    Sim(Scenario),
    Big,
    Md(MdCfg),
}

pub fn make_world(g: &Gen, seed: u64) -> W {
    match g {
        Gen::Sim(sc) => {
            let e = Env::sim(seed, *sc);
            let (tasks, na, kw, len) = (e.tasks_vec(), e.n_agents(), e.kw_map(), e.len().min(N_TASKS));
            W::finish(Back::E(e), tasks, na, kw, len)
        }
        Gen::Big => {
            let e = Env::big(seed);
            let (tasks, na, kw, len) = (e.tasks_vec(), e.n_agents(), e.kw_map(), e.len().min(N_TASKS));
            W::finish(Back::E(e), tasks, na, kw, len)
        }
        Gen::Md(c) => gen_md(c, seed),
    }
}

fn gen_md(c: &MdCfg, seed: u64) -> W {
    let mut r = Rng::new(seed.wrapping_mul(0xA24B_AED4_963E_E407) ^ 0x4D44_5F57_4F52_4C44);
    let base_n = c.pool.n_agents();
    let n = base_n + c.extra;
    let pop = pop_table(c.pool);
    let mut p = vec![vec![0.0; n]; NSUB];
    if c.pool == PoolKind::RandSkills {
        for s in 0..NSUB {
            for a in 0..n {
                p[s][a] = 0.20 + 0.30 * r.f64();
            }
            let best = r.below(n);
            p[s][best] = 0.84 + 0.04 * r.f64();
            let mut sec = r.below(n - 1);
            if sec >= best {
                sec += 1;
            }
            p[s][sec] = 0.55 + 0.10 * r.f64();
        }
    } else {
        for s in 0..NSUB {
            for a in 0..base_n {
                p[s][a] = pop[s][a];
            }
        }
        if c.pool != PoolKind::Equal {
            for d in 0..NDOM {
                if c.derange || r.bernoulli(c.rho) {
                    let mut astar = 0;
                    let mut bestv = -1.0;
                    for a in 0..base_n {
                        let v: f64 = (0..NSUBPD).map(|i| pop[d * NSUBPD + i][a]).sum();
                        if v > bestv {
                            bestv = v;
                            astar = a;
                        }
                    }
                    let mut b = r.below(base_n - 1);
                    if b >= astar {
                        b += 1;
                    }
                    for i in 0..NSUBPD {
                        p[d * NSUBPD + i].swap(astar, b);
                    }
                }
            }
            for s in 0..NSUB {
                for a in 0..base_n {
                    p[s][a] = (p[s][a] + c.sigma * rnorm(&mut r)).clamp(0.03, 0.97);
                }
            }
        }
        for e in 0..c.extra {
            let a = base_n + e;
            let dom = (1 + 2 * e) % NDOM;
            for s in 0..NSUB {
                p[s][a] = if s / NSUBPD == dom { 0.95 } else { 0.40 };
            }
        }
    }
    let p_pre = p.clone();
    let mut p_post = p;
    if c.drift {
        for d in 0..NDOM {
            if r.bernoulli(0.35) {
                let mut astar = 0;
                let mut bv = -1.0;
                for a in 0..n {
                    let v: f64 = (0..NSUBPD).map(|i| p_pre[d * NSUBPD + i][a]).sum();
                    if v > bv {
                        bv = v;
                        astar = a;
                    }
                }
                let mut b = r.below(n - 1);
                if b >= astar {
                    b += 1;
                }
                for i in 0..NSUBPD {
                    let s = d * NSUBPD + i;
                    p_post[s][astar] = (p_pre[s][astar] - 0.32).max(0.05);
                    p_post[s][b] = 0.86;
                }
            }
        }
    }
    // keyword map (+ renamed vocabulary)
    let renamed: Vec<bool> = (0..NSUB).map(|_| r.bernoulli(c.vocab_ren)).collect();
    let sub_name = |s: usize| -> String {
        let nm = SUBS[s / NSUBPD][s % NSUBPD];
        if renamed[s] { format!("{nm}~x") } else { nm.to_string() }
    };
    let argmax = |v: &dyn Fn(usize) -> f64| -> usize {
        let mut b = 0;
        for a in 1..n {
            if v(a) > v(b) {
                b = a;
            }
        }
        b
    };
    let mut kw = HashMap::new();
    for s in 0..NSUB {
        let best = argmax(&|a| p_pre[s][a]);
        let ag = if r.bernoulli(c.decoy) {
            let mut w = r.below(n - 1);
            if w >= best {
                w += 1;
            }
            w
        } else {
            best
        };
        kw.insert(sub_name(s), ag);
    }
    for d in 0..NDOM {
        let best = argmax(&|a| (0..NSUBPD).map(|i| p_pre[d * NSUBPD + i][a]).sum::<f64>());
        let ag = if r.bernoulli(c.decoy) {
            let mut w = r.below(n - 1);
            if w >= best {
                w += 1;
            }
            w
        } else {
            best
        };
        kw.insert(DOMS[d].to_string(), ag);
    }
    // tasks
    let tot: f64 = c.mix.iter().sum();
    let mut tasks = Vec::with_capacity(N_TASKS);
    let mut sub_of = Vec::with_capacity(N_TASKS);
    let mut u = Vec::with_capacity(N_TASKS);
    for id in 0..N_TASKS {
        let mut x = r.f64() * tot;
        let mut d = NDOM - 1;
        for (i, m) in c.mix.iter().enumerate() {
            if x < *m {
                d = i;
                break;
            }
            x -= *m;
        }
        let s = d * NSUBPD + r.below(NSUBPD);
        let mut f = vec![sub_name(s)];
        if r.bernoulli(0.7) {
            f.push(DOMS[d].to_string());
        }
        if r.bernoulli(0.5) {
            f.push(FLUFF[r.below(FLUFF.len())].to_string());
        }
        if r.bernoulli(0.2) {
            let mut o = r.below(NSUB - 1);
            if o >= s {
                o += 1;
            }
            f.push(sub_name(o));
        }
        for i in (1..f.len()).rev() {
            let j = r.below(i + 1);
            f.swap(i, j);
        }
        tasks.push(Task { id: id as u64, task_type: None, features: f });
        sub_of.push(s);
        u.push(r.f64());
    }
    let md = Md { p_pre, p_post, sub_of, u, drift_at: N_TASKS / 2, blind: c.blind };
    W::finish(Back::M(md), tasks, n, kw, N_TASKS)
}

// ===================================================================================== world specs
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// related pool: an informative prior should help
    Treat,
    /// prior is WRONG / unrelated / shifted: must recover, cost is bounded
    Stress,
    /// nothing to learn or keyword already ~oracle: prior must be harmless
    Control,
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub name: &'static str,
    pub eval: Gen,
    pub pool: Gen,
    pub role: Role,
    /// used for tuning (seeds 1-8). Worlds with tune=false are NEVER seen by the tuner.
    pub tune: bool,
    pub note: &'static str,
}

pub fn specs() -> Vec<Spec> {
    let b1 = MdCfg::base(PoolKind::Agency);
    let mut v = vec![];
    let mut push = |name, eval, pool, role, tune, note| v.push(Spec { name, eval, pool, role, tune, note });
    push("A1-sim-fixed", Gen::Sim(Scenario::Fixed), Gen::Sim(Scenario::Fixed), Role::Treat, true,
        "sim Fixed (hand-designed traps+drift); pool = other seeds of the SAME hidden skills (related)");
    push("A2-sim-random", Gen::Sim(Scenario::Random), Gen::Sim(Scenario::Random), Role::Control, true,
        "sim Random: hidden skills redrawn per seed; pool shares only vocabulary/structure, no agent identity (randomized world)");
    push("A3-sim-notraps", Gen::Sim(Scenario::NoTraps), Gen::Sim(Scenario::NoTraps), Role::Control, false,
        "sim NoTraps: keyword map correct pre-drift; skills random per seed");
    push("A4-big32", Gen::Big, Gen::Big, Role::Control, true,
        "f3 BigWorld 32 agents x 48 types, skills random per seed (first 2000 tasks, pre-drift)");
    push("A5-sim-mismatch", Gen::Sim(Scenario::Random), Gen::Sim(Scenario::Fixed), Role::Stress, false,
        "WRONG prior: pool = sim Fixed, eval = sim Random (same vocabulary, unrelated skills)");
    push("B1-md-related", Gen::Md(b1.clone()), Gen::Md(b1.clone()), Role::Treat, true,
        "multi-domain agency pool, 10% of domains re-drawn, sigma .05, decoy .3; pool = same family");
    push("B2-md-partial", Gen::Md(MdCfg { rho: 0.4, ..b1.clone() }), Gen::Md(MdCfg { rho: 0.4, ..b1.clone() }), Role::Treat, true,
        "as B1 but 40% of domains re-drawn per deployment (partially related)");
    push("B3-md-big20", Gen::Md(MdCfg { rho: 0.2, ..MdCfg::base(PoolKind::Big) }), Gen::Md(MdCfg { rho: 0.2, ..MdCfg::base(PoolKind::Big) }), Role::Treat, false,
        "20-agent pool (3 specialists/domain + generalists), rho .2");
    push("B4-md-shift", Gen::Md(MdCfg { sigma: 0.08, decoy: 0.5, drift: true, extra: 2, mix: [0.1, 0.35, 0.1, 0.35, 0.1], ..b1.clone() }), Gen::Md(b1.clone()), Role::Treat, false,
        "POOL DISTRIBUTION SHIFT: eval has drift, 2 new agents unknown to the pool, decoy .5, different domain mix");
    push("B5-md-heldout-family", Gen::Md(MdCfg { vocab_ren: 0.5, ..MdCfg::base(PoolKind::Latent) }), Gen::Md(b1.clone()), Role::Stress, false,
        "HELD-OUT FAMILY: pool = agency structure; eval = latent-factor pool, half the vocabulary renamed");
    push("B6-md-wrong", Gen::Md(MdCfg { derange: true, ..b1.clone() }), Gen::Md(b1.clone()), Role::Stress, true,
        "WRONG prior: every domain's specialist structure moved to a different agent");
    push("B7-md-equal", Gen::Md(MdCfg::base(PoolKind::Equal)), Gen::Md(MdCfg::base(PoolKind::Equal)), Role::Control, true,
        "8 identical agents: nothing to win (sanity control)");
    push("B9-md-uninformative-fb", Gen::Md(MdCfg { blind: true, ..b1.clone() }), Gen::Md(b1.clone()), Role::Control, false,
        "uninformative feedback: related pool, but the reward the router sees in this deployment is an agent-independent coin");
    push("B8-md-noinfo", Gen::Md(MdCfg::base(PoolKind::RandSkills)), Gen::Md(MdCfg::base(PoolKind::RandSkills)), Role::Control, false,
        "no-prior-info world: skills random per deployment, pool deployments are mutually inconsistent");
    v
}

// ===================================================================================== pool + prior
#[derive(Clone, Debug, Default)]
pub struct Dep {
    /// feature -> per-agent (success mass, trials)
    pub cells: HashMap<String, Vec<(f64, f64)>>,
    pub kw_s: f64,
    pub kw_n: f64,
}

/// What the keyword map would vote for (None if no feature votes).
pub fn kw_choice(map: &HashMap<String, usize>, task: &Task, na: usize) -> Option<usize> {
    let mut votes = vec![0usize; na];
    let mut any = false;
    for f in &task.features {
        if let Some(&a) = map.get(f) {
            if a < na {
                votes[a] += 1;
                any = true;
            }
        }
    }
    if !any {
        return None;
    }
    let mut b = 0;
    for a in 1..na {
        if votes[a] > votes[b] {
            b = a;
        }
    }
    Some(b)
}

/// One previous deployment: the DEFAULT router (guarded-safe) runs `n_tasks` tasks; its logged
/// (feature, agent, reward) outcomes are the only thing that survives.
pub fn log_dep(w: &W, n_tasks: usize, seed: u64) -> Dep {
    let n = w.n_agents;
    let mut router = recommended_router(w.kw.clone(), n);
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(99));
    let mut d = Dep::default();
    for t in 0..n_tasks.min(w.len) {
        let task = &w.tasks[t];
        let a = router.select(task, &cands, &mut rng);
        let r = w.outcome(t, a);
        router.update(task, a, r);
        for f in &task.features {
            let row = d.cells.entry(f.clone()).or_insert_with(|| vec![(0.0, 0.0); n]);
            row[a].0 += r;
            row[a].1 += 1.0;
        }
        if kw_choice(&w.kw, task, n) == Some(a) {
            d.kw_s += r;
            d.kw_n += 1.0;
        }
    }
    d
}

pub fn build_pool(g: &Gen, k: usize, eval_seed: u64) -> Vec<Dep> {
    (0..k)
        .map(|j| {
            let ps = POOL_BASE + eval_seed * 100 + j as u64;
            let w = make_world(g, ps);
            log_dep(&w, N_POOL_TASKS, ps)
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Cell {
    pub s: f64,
    pub n: f64,
    pub mu: f64,
    /// prior strength in pseudo-trials (0 = no information)
    pub m: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct FitCfg {
    /// cap on the pseudo-count of one (feature, agent) cell
    pub m_max: f64,
    /// pseudo-count when a cell was seen in a single deployment only
    pub m1: f64,
    /// minimum pooled trials for a cell to carry any prior
    pub n_min: f64,
    /// true = hierarchical shrinkage (strength from between-deployment consistency); false = naive pooled counts
    pub shrink: bool,
    /// a cell must be observed (>= 3 trials) in at least this fraction of the pool's deployments, and its
    /// strength is scaled by that fraction (guards against selection bias of logged data: a cell that only
    /// survives in a few deployments is not evidence about a population)
    pub cov_min: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PopPrior {
    pub cells: HashMap<String, Vec<Cell>>,
    /// success rate of the keyword agent across the pool (selection-biased: see report)
    pub kw_acc: f64,
    pub n_deps: usize,
    /// per-agent global success rate over the pool (population prior for cells without their own evidence)
    pub agent_mu: Vec<f64>,
}

impl PopPrior {
    pub fn cell(&self, f: &str, a: usize) -> Option<&Cell> {
        self.cells.get(f).and_then(|v| v.get(a)).filter(|c| c.m > 0.0)
    }
    /// Empty prior (== no information).
    pub fn empty() -> Self {
        PopPrior { cells: HashMap::new(), kw_acc: 0.8, n_deps: 0, agent_mu: vec![] }
    }
    /// Keyword map expressed as pseudo-counts (the "(a) keyword prior" in pooled-TS form).
    pub fn from_keywords(map: &HashMap<String, usize>, n: usize, mu: f64, m: f64) -> Self {
        let mut cells = HashMap::new();
        for (f, &a) in map {
            let mut v = vec![Cell::default(); n];
            if a < n {
                v[a] = Cell { s: 0.0, n: 0.0, mu, m };
            }
            cells.insert(f.clone(), v);
        }
        PopPrior { cells, kw_acc: mu, n_deps: 0, agent_mu: vec![] }
    }
}

pub fn fit_prior(deps: &[Dep], n_agents: usize, cfg: FitCfg) -> PopPrior {
    let mut feats: Vec<&String> = deps.iter().flat_map(|d| d.cells.keys()).collect();
    feats.sort();
    feats.dedup();
    let mut cells = HashMap::new();
    for f in feats {
        let mut row = vec![Cell::default(); n_agents];
        for a in 0..n_agents {
            let obs: Vec<(f64, f64)> = deps
                .iter()
                .filter_map(|d| d.cells.get(f).and_then(|v| v.get(a)).copied())
                .filter(|&(_, n)| n >= 3.0)
                .collect();
            let ntot: f64 = obs.iter().map(|x| x.1).sum();
            let cov = obs.len() as f64 / deps.len().max(1) as f64;
            if ntot < cfg.n_min || obs.is_empty() || cov < cfg.cov_min {
                continue;
            }
            let stot: f64 = obs.iter().map(|x| x.0).sum();
            let mu = (stot + 0.5) / (ntot + 1.0);
            let k = obs.len() as f64;
            let m = if !cfg.shrink {
                ntot.min(cfg.m_max)
            } else if obs.len() < 2 {
                cfg.m1.min(ntot)
            } else {
                let ss: f64 = obs.iter().map(|&(s, n)| n * (s / n - mu) * (s / n - mu)).sum();
                let n2: f64 = obs.iter().map(|x| x.1 * x.1).sum();
                let denom = ntot - n2 / ntot;
                let tau2 = ((ss - (k - 1.0) * mu * (1.0 - mu)) / denom.max(1e-9)).max(0.0);
                let m0 = if tau2 < 1e-9 { cfg.m_max } else { (mu * (1.0 - mu) / tau2 - 1.0).clamp(0.0, cfg.m_max) };
                m0.min(ntot)
            };
            let m = m * cov;
            row[a] = Cell { s: stot, n: ntot, mu, m };
        }
        cells.insert(f.clone(), row);
    }
    let (ks, kn): (f64, f64) = deps.iter().fold((0.0, 0.0), |a, d| (a.0 + d.kw_s, a.1 + d.kw_n));
    let mut agent_mu = vec![0.5; n_agents];
    for a in 0..n_agents {
        let (mut s, mut n) = (0.0, 0.0);
        for d in deps {
            for row in d.cells.values() {
                if let Some(&(cs, cn)) = row.get(a) {
                    s += cs;
                    n += cn;
                }
            }
        }
        agent_mu[a] = (s + 1.0) / (n + 2.0);
    }
    PopPrior { cells, kw_acc: ((ks + 1.0) / (kn + 2.0)).clamp(0.3, 0.95), n_deps: deps.len(), agent_mu }
}

/// Pool-corrected keyword map: replace a feature's agent where the pool shows a clearly better one.
/// Returns (map, accuracy of the final map's agents in the pool logs).
pub fn corrected_map(base: &HashMap<String, usize>, prior: &PopPrior, n_cell: f64, delta: f64, add_thr: f64) -> (HashMap<String, usize>, f64) {
    let mut map = base.clone();
    let mut feats: Vec<&String> = prior.cells.keys().collect();
    feats.sort();
    for f in feats {
        let row = &prior.cells[f];
        let mut best: Option<(usize, f64)> = None;
        for (a, c) in row.iter().enumerate() {
            if c.n >= n_cell && c.m > 0.0 && best.is_none_or(|(_, v)| c.mu > v) {
                best = Some((a, c.mu));
            }
        }
        let Some((b, mb)) = best else { continue };
        match base.get(f) {
            Some(&k) => {
                let mk = row.get(k).filter(|c| c.n > 0.0).map(|c| c.mu).unwrap_or(0.5);
                if b != k && mb - mk >= delta {
                    map.insert(f.clone(), b);
                }
            }
            None => {
                if mb >= add_thr {
                    map.insert(f.clone(), b);
                }
            }
        }
    }
    let (mut s, mut n) = (0.0, 0.0);
    for (f, &a) in &map {
        if let Some(c) = prior.cells.get(f).and_then(|r| r.get(a)) {
            s += c.s;
            n += c.n;
        }
    }
    (map, ((s + 1.0) / (n + 2.0)).clamp(0.3, 0.95))
}

/// Static-policy diagnostics of a prior on a world: expected success of "follow the prior's argmax"
/// and of "follow the keyword map" over the first `h` tasks, the oracle, and prior coverage.
pub fn prior_diag(w: &W, prior: &PopPrior, h: usize) -> (f64, f64, f64, f64) {
    let n = w.n_agents;
    let h = h.min(w.len);
    let (mut sp, mut sk, mut so, mut cov) = (0.0, 0.0, 0.0, 0.0);
    for t in 0..h {
        let task = &w.tasks[t];
        let mut best = (0usize, f64::NEG_INFINITY);
        let mut any = false;
        for a in 0..n {
            let (mut sm, mut k) = (0.0f64, 0.0f64);
            for f in &task.features {
                if let Some(c) = prior.cell(f, a) {
                    sm += c.mu;
                    any = true;
                } else {
                    sm += 0.5;
                }
                k += 1.0;
            }
            let v = sm / k.max(1.0);
            if v > best.1 {
                best = (a, v);
            }
        }
        if any {
            cov += 1.0;
        }
        sp += w.prob(t, best.0);
        let kc = kw_choice(&w.kw, task, n).unwrap_or(task.id as usize % n);
        sk += w.prob(t, kc);
        so += w.oracle[t];
    }
    let hf = h as f64;
    (sp / hf, sk / hf, so / hf, cov / hf)
}

// ===================================================================================== PopTs
#[derive(Clone, Copy, Debug)]
pub struct TrustCfg {
    pub w0: f64,
    pub cf: f64,
    pub cg: f64,
    pub delta: f64,
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-4, 1.0 - 1e-4);
    (p / (1.0 - p)).ln()
}
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
}

/// Pooled Thompson sampling whose per-(feature, agent) Beta cells are initialised from a prior
/// (pseudo-counts m*mu, m*(1-mu)), each scaled by a trust weight learned online (prequential
/// prior-vs-data log score, discounted). With an empty prior it is EXACTLY `PooledThompsonRouter(gamma=1)`.
pub struct PopTs {
    pub label: String,
    n: usize,
    prior: PopPrior,
    data: HashMap<String, Vec<(f64, f64)>>,
    trust: Option<TrustCfg>,
    lp: HashMap<String, (f64, f64)>,
    g: (f64, f64),
    /// pseudo-count of the per-agent population mean used for (feature, agent) cells without own prior (0 = off)
    m_agent: f64,
}

impl PopTs {
    pub fn new(label: &str, n: usize, prior: PopPrior, trust: Option<TrustCfg>) -> Self {
        PopTs { label: label.to_string(), n, prior, data: HashMap::new(), trust, lp: HashMap::new(), g: (0.0, 0.0), m_agent: 0.0 }
    }
    pub fn with_agent_prior(mut self, m: f64) -> Self {
        self.m_agent = m;
        self
    }
    pub fn feat_trust(&self, f: &str) -> f64 {
        match &self.trust {
            None => 1.0,
            Some(t) => {
                let (lp, ld) = self.lp.get(f).copied().unwrap_or((0.0, 0.0));
                sigmoid(logit(t.w0) + t.cf * (lp - ld) + t.cg * (self.g.0 - self.g.1))
            }
        }
    }
}

impl Router for PopTs {
    fn name(&self) -> String {
        self.label.clone()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let mut best = candidates[0];
        let mut best_v = f64::NEG_INFINITY;
        for &c in candidates {
            let (mut al, mut be) = (1.0, 1.0);
            for f in &task.features {
                if let Some(&(s, fl)) = self.data.get(f).and_then(|r| r.get(c)) {
                    al += s;
                    be += fl;
                }
                if let Some(cell) = self.prior.cell(f, c) {
                    let w = self.feat_trust(f);
                    al += w * cell.m * cell.mu;
                    be += w * cell.m * (1.0 - cell.mu);
                } else if self.m_agent > 0.0 {
                    if let Some(&mu) = self.prior.agent_mu.get(c) {
                        al += self.m_agent * mu;
                        be += self.m_agent * (1.0 - mu);
                    }
                }
            }
            let v = sample_beta(al, be, rng);
            if v > best_v {
                best_v = v;
                best = c;
            }
        }
        best
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        self.n = self.n.max(agent + 1);
        let n = self.n;
        if let Some(t) = self.trust {
            let (mut gl, mut gd, mut cnt) = (0.0, 0.0, 0.0);
            for f in &task.features {
                if let Some(cell) = self.prior.cell(f, agent).copied() {
                    let (ds, df) = self.data.get(f).and_then(|row| row.get(agent)).copied().unwrap_or((0.0, 0.0));
                    let pd = ((1.0 + ds) / (2.0 + ds + df)).clamp(0.02, 0.98);
                    let pp = cell.mu.clamp(0.02, 0.98);
                    let sp = r * pp.ln() + (1.0 - r) * (1.0 - pp).ln();
                    let sd = r * pd.ln() + (1.0 - r) * (1.0 - pd).ln();
                    let e = self.lp.entry(f.clone()).or_insert((0.0, 0.0));
                    e.0 = e.0 * t.delta + sp;
                    e.1 = e.1 * t.delta + sd;
                    gl += sp;
                    gd += sd;
                    cnt += 1.0;
                }
            }
            if cnt > 0.0 {
                self.g.0 = self.g.0 * t.delta + gl / cnt;
                self.g.1 = self.g.1 * t.delta + gd / cnt;
            }
        }
        for f in &task.features {
            let row = self.data.entry(f.clone()).or_insert_with(|| vec![(0.0, 0.0); n]);
            if row.len() < n {
                row.resize(n, (0.0, 0.0));
            }
            row[agent].0 += r;
            row[agent].1 += 1.0 - r;
        }
    }
}


// ===================================================================================== PopGuarded
/// `prior::Guarded<HebbianRouter>` (the recommended default) with its per-(feature, agent) success/failure
/// counts INITIALISED from the pool prior (pseudo-counts m*mu, m*(1-mu), the prior part scaled by an online trust
/// weight when `conf` is set). Decision rule, discounting and inner learner are exactly Guarded's; with an empty
/// prior it is bit-identical to `defaults::recommended_router` (selftest).
pub struct PopGuarded {
    pub inner: HebbianRouter,
    prior_kw: neural_mesh::prior::KeywordPrior,
    pop: PopPrior,
    cfg: GuardConfig,
    n: usize,
    /// feature -> (data counts, prior counts) per agent
    rows: HashMap<String, (Vec<(f64, f64)>, Vec<(f64, f64)>)>,
    trust: Option<TrustCfg>,
    lp: HashMap<String, (f64, f64)>,
    g: (f64, f64),
    /// when the keyword rule is abandoned / absent, follow the best candidate whose pooled evidence contains at least
    /// `inf_min` pseudo-trials of POOL prior (and whose LCB beats the keyword UCB) instead of asking the cold inner
    /// learner. 0 = off (decisions are then exactly Guarded's).
    inf_min: f64,
    pub decisions: [u64; 3],
}

fn bounds(a: f64, b: f64, z: f64) -> (f64, f64) {
    let n = a + b;
    let m = a / n;
    let sd = (m * (1.0 - m) / (n + 1.0)).sqrt();
    (m - z * sd, m + z * sd)
}

impl PopGuarded {
    pub fn new(map: HashMap<String, usize>, n: usize, pop: PopPrior, trust: Option<TrustCfg>) -> Self {
        PopGuarded {
            inner: HebbianRouter::new(n, recommended_inner()),
            prior_kw: neural_mesh::prior::KeywordPrior::new(map, n),
            pop,
            cfg: neural_mesh::defaults::recommended_guard(),
            n,
            rows: HashMap::new(),
            trust,
            lp: HashMap::new(),
            g: (0.0, 0.0),
            inf_min: 0.0,
            decisions: [0; 3],
        }
    }
    pub fn with_informed(mut self, inf_min: f64) -> Self {
        self.inf_min = inf_min;
        self
    }
    /// Best candidate by LCB among those carrying >= inf_min (trust-weighted) pool pseudo-trials, if its LCB > floor.
    fn informed(&self, task: &Task, candidates: &[usize], floor: f64) -> Option<usize> {
        if self.inf_min <= 0.0 || task.features.is_empty() {
            return None;
        }
        let nf = task.features.len() as f64;
        let mut best: Option<(usize, f64)> = None;
        for &a in candidates {
            let mut mass = 0.0;
            for k in &task.features {
                let w = self.weight(k);
                let p = match self.rows.get(k) {
                    Some((_, pr)) => pr.get(a).copied().unwrap_or((0.0, 0.0)),
                    None => self.seed(k, a),
                };
                mass += w * (p.0 + p.1);
            }
            if mass / nf < self.inf_min {
                continue;
            }
            let (s, f) = self.pooled(task, a);
            let (lcb, _) = bounds(1.0 + s, 1.0 + f, self.cfg.z);
            if best.is_none_or(|(_, v)| lcb > v) {
                best = Some((a, lcb));
            }
        }
        best.filter(|&(_, l)| l > floor).map(|(a, _)| a)
    }
    fn seed(&self, f: &str, a: usize) -> (f64, f64) {
        match self.pop.cell(f, a) {
            Some(c) => (c.m * c.mu, c.m * (1.0 - c.mu)),
            None => (0.0, 0.0),
        }
    }
    pub fn weight(&self, f: &str) -> f64 {
        match &self.trust {
            None => 1.0,
            Some(t) => {
                let (lp, ld) = self.lp.get(f).copied().unwrap_or((0.0, 0.0));
                sigmoid(logit(t.w0) + t.cf * (lp - ld) + t.cg * (self.g.0 - self.g.1))
            }
        }
    }
    /// Pooled (successes, failures) of `agent` over the task's features (mean), prior part weighted by trust.
    fn pooled(&self, task: &Task, agent: usize) -> (f64, f64) {
        if task.features.is_empty() {
            return (0.0, 0.0);
        }
        let (mut s, mut f) = (0.0, 0.0);
        for k in &task.features {
            let w = self.weight(k);
            let (d, p) = match self.rows.get(k) {
                Some((dr, pr)) => (dr.get(agent).copied().unwrap_or((0.0, 0.0)), pr.get(agent).copied().unwrap_or((0.0, 0.0))),
                None => ((0.0, 0.0), self.seed(k, agent)),
            };
            s += d.0 + w * p.0;
            f += d.1 + w * p.1;
        }
        let n = task.features.len() as f64;
        (s / n, f / n)
    }
    fn decide(&self, task: &Task, candidates: &[usize]) -> (u8, usize) {
        let votes = self.prior_kw.votes(task, candidates);
        let Some(k) = votes
            .iter()
            .fold(None, |b: Option<(usize, usize)>, &(a, v)| match b {
                Some((_, bv)) if bv >= v => b,
                _ => Some((a, v)),
            })
            .map(|(a, _)| a)
        else {
            return match self.informed(task, candidates, 0.6) {
                Some(a) => (1, a),
                None => (2, 0),
            };
        };
        let (s, f) = self.pooled(task, k);
        let (_, ucb_k) = bounds(self.cfg.kw_a + s, self.cfg.kw_b + f, self.cfg.z);
        if ucb_k < self.cfg.tau {
            return match self.informed(task, candidates, ucb_k) {
                Some(a) => (1, a),
                None => (2, 0),
            };
        }
        if self.cfg.challenge {
            let mut best = (ucb_k, usize::MAX);
            for &a in candidates {
                if a == k {
                    continue;
                }
                let (s, f) = self.pooled(task, a);
                let (lcb, _) = bounds(1.0 + s, 1.0 + f, self.cfg.z);
                if lcb > best.0 {
                    best = (lcb, a);
                }
            }
            if best.1 != usize::MAX {
                return (1, best.1);
            }
        }
        (0, k)
    }
}

impl Router for PopGuarded {
    fn name(&self) -> String {
        "pop-guard".into()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let (d, a) = self.decide(task, candidates);
        self.decisions[d as usize] += 1;
        if d == 2 { self.inner.select(task, candidates, rng) } else { a }
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        let n = self.n.max(agent + 1);
        if let Some(t) = self.trust {
            let (mut gl, mut gd, mut cnt) = (0.0, 0.0, 0.0);
            for f in &task.features {
                if let Some(cell) = self.pop.cell(f, agent).copied() {
                    let (ds, df) = self.rows.get(f).and_then(|(dr, _)| dr.get(agent)).copied().unwrap_or((0.0, 0.0));
                    let pd = ((1.0 + ds) / (2.0 + ds + df)).clamp(0.02, 0.98);
                    let pp = cell.mu.clamp(0.02, 0.98);
                    let sp = r * pp.ln() + (1.0 - r) * (1.0 - pp).ln();
                    let sd = r * pd.ln() + (1.0 - r) * (1.0 - pd).ln();
                    let e = self.lp.entry(f.clone()).or_insert((0.0, 0.0));
                    e.0 = e.0 * t.delta + sp;
                    e.1 = e.1 * t.delta + sd;
                    gl += sp;
                    gd += sd;
                    cnt += 1.0;
                }
            }
            if cnt > 0.0 {
                self.g.0 = self.g.0 * t.delta + gl / cnt;
                self.g.1 = self.g.1 * t.delta + gd / cnt;
            }
        }
        for k in &task.features {
            if !self.rows.contains_key(k) {
                let pri: Vec<(f64, f64)> = (0..n).map(|a| self.seed(k, a)).collect();
                self.rows.insert(k.clone(), (vec![(0.0, 0.0); n], pri));
            }
            let (dr, pr) = self.rows.get_mut(k).unwrap();
            if dr.len() < n {
                dr.resize(n, (0.0, 0.0));
                pr.resize(n, (0.0, 0.0));
            }
            if self.cfg.discount < 1.0 {
                for c in dr.iter_mut().chain(pr.iter_mut()) {
                    c.0 *= self.cfg.discount;
                    c.1 *= self.cfg.discount;
                }
            }
            dr[agent].0 += r;
            dr[agent].1 += 1.0 - r;
        }
        self.inner.update(task, agent, reward);
    }
}

// ===================================================================================== policies
#[derive(Clone, Debug)]
pub enum Pol {
    Keyword,
    Eps { g: f64, e: f64 },
    Pts { g: f64 },
    Hebr,
    Warm { k: f64 },
    Guarded,
    KwTs { m: f64 },
    PopTs { fit: FitCfg, conf: Option<TrustCfg>, m_agent: f64 },
    PopGuard { fit: FitCfg, conf: Option<TrustCfg>, inf: f64 },
    PopMap { fit: FitCfg, n_cell: f64, delta: f64, m: f64 },
    KwCal { m: f64, tau: f64 },
}

fn fit_label(f: &FitCfg) -> String {
    format!("M{},m1={},cov{}{}", f.m_max, f.m1, f.cov_min, if f.shrink { "" } else { ",noshrink" })
}

impl Pol {
    pub fn label(&self) -> String {
        match self {
            Pol::Keyword => "keyword".into(),
            Pol::Eps { g, e } => format!("eps-greedy(g{g},e{e})"),
            Pol::Pts { g } => format!("pooled-ts(g{g})"),
            Pol::Hebr => "hebbian-robust".into(),
            Pol::Warm { k } => format!("warm(hebr,k{k})"),
            Pol::Guarded => "guarded-safe".into(),
            Pol::KwTs { m } => format!("kw-ts(m{m})"),
            Pol::PopTs { fit, conf, m_agent } => {
                let mut s = format!("pop-ts{}[{},ma={}]", if conf.is_some() { "+conf" } else { "" }, fit_label(fit), m_agent);
                if let Some(c) = conf {
                    s += &format!("[w0={},cg={}]", c.w0, c.cg);
                }
                s
            }
            Pol::PopGuard { fit, conf, inf } => {
                let mut s = format!("pop-guard{}[{},inf={}]", if conf.is_some() { "+conf" } else { "" }, fit_label(fit), inf);
                if let Some(c) = conf {
                    s += &format!("[w0={},cg={}]", c.w0, c.cg);
                }
                s
            }
            Pol::PopMap { fit, n_cell, delta, m } => format!("pop-map[{},nc{},d{},m{}]", fit_label(fit), n_cell, delta, m),
            Pol::KwCal { m, tau } => format!("kw-cal[m{m},tau{tau}]"),
        }
    }

    pub fn build(&self, w: &W, deps: &[Dep]) -> Box<dyn Router> {
        let n = w.n_agents;
        let pool_n = deps.first().and_then(|d| d.cells.values().next().map(|v| v.len())).unwrap_or(n);
        match self {
            Pol::Keyword => Box::new(KeywordRouter::new(w.kw.clone(), n)),
            Pol::Eps { g, e } => Box::new(EpsGreedyRouter::new(n, *g, *e)),
            Pol::Pts { g } => Box::new(PooledThompsonRouter::new(n, *g)),
            Pol::Hebr => Box::new(HebbianRouter::new(n, heb_robust_cfg())),
            Pol::Warm { k } => Box::new(Warm::new(HebbianRouter::new(n, heb_robust_cfg()), w.kw.clone(), n, *k).with_evidence(Evidence::MeanFeature)),
            Pol::Guarded => Box::new(recommended_router(w.kw.clone(), n)),
            Pol::KwTs { m } => Box::new(PopTs::new("kw-ts", n, PopPrior::from_keywords(&w.kw, n, 0.8, *m), None)),
            Pol::PopTs { fit, conf, m_agent } => {
                let prior = fit_prior(deps, pool_n, *fit);
                Box::new(PopTs::new(&self.label(), n, prior, *conf).with_agent_prior(*m_agent))
            }
            Pol::PopGuard { fit, conf, inf } => {
                let prior = fit_prior(deps, pool_n, *fit);
                Box::new(PopGuarded::new(w.kw.clone(), n, prior, *conf).with_informed(*inf))
            }
            Pol::PopMap { fit, n_cell, delta, m } => {
                let prior = fit_prior(deps, pool_n, *fit);
                let (map, acc) = corrected_map(&w.kw, &prior, *n_cell, *delta, 0.6);
                let cfg = GuardConfig { tau: 0.7, z: 0.5, kw_a: m * acc, kw_b: m * (1.0 - acc), discount: 0.98, challenge: true };
                Box::new(Guarded::new(HebbianRouter::new(n, recommended_inner()), map, n, cfg))
            }
            Pol::KwCal { m, tau } => {
                let (ks, kn): (f64, f64) = deps.iter().fold((0.0, 0.0), |a, d| (a.0 + d.kw_s, a.1 + d.kw_n));
                let acc = ((ks + 1.0) / (kn + 2.0)).clamp(0.3, 0.95);
                let cfg = GuardConfig { tau: *tau, z: 0.5, kw_a: m * acc, kw_b: m * (1.0 - acc), discount: 0.98, challenge: true };
                Box::new(Guarded::new(HebbianRouter::new(n, recommended_inner()), w.kw.clone(), n, cfg))
            }
        }
    }
}

// ===================================================================================== run + metrics
pub fn run_curve(w: &W, r: &mut dyn Router, seed: u64) -> Vec<f32> {
    let n = w.n_agents;
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut out = Vec::with_capacity(w.len);
    for t in 0..w.len {
        let task = &w.tasks[t];
        let a = r.select(task, &cands, &mut rng);
        out.push(w.prob(t, a) as f32);
        r.update(task, a, w.outcome(t, a));
    }
    out
}

pub const NM: usize = 12;
pub const MNAMES: [&str; NM] = ["s50", "s100", "s250", "s500", "s2000", "w100-250", "w250-500", "w500-1000", "w1000-2000", "reg2000", "ttt95", "ttt90"];
/// metric indices that are "higher is better" (success metrics); reg/ttt are lower-is-better
pub fn higher_better(m: usize) -> bool {
    m < 9
}

fn mean_range(c: &[f32], lo: usize, hi: usize) -> f64 {
    let hi = hi.min(c.len());
    let lo = lo.min(hi);
    if hi == lo {
        return 0.0;
    }
    c[lo..hi].iter().map(|x| *x as f64).sum::<f64>() / (hi - lo) as f64
}

pub fn metrics(c: &[f32], oracle: &[f64]) -> [f64; NM] {
    let n = c.len();
    let reg = |h: usize| -> f64 { (0..h.min(n)).map(|t| oracle[t] - c[t] as f64).sum::<f64>() };
    // tasks-to-target: first t (>= 50) with trailing-50 mean success >= q * trailing-50 oracle mean (censored at n)
    let ttt = |q: f64| -> f64 {
        let (mut sc, mut so) = (0.0, 0.0);
        for t in 0..n {
            sc += c[t] as f64;
            so += oracle[t];
            if t >= 50 {
                sc -= c[t - 50] as f64;
                so -= oracle[t - 50];
            }
            if t >= 49 && sc >= q * so {
                return (t + 1) as f64;
            }
        }
        n as f64
    };
    [
        mean_range(c, 0, 50),
        mean_range(c, 0, 100),
        mean_range(c, 0, 250),
        mean_range(c, 0, 500),
        mean_range(c, 0, 2000),
        mean_range(c, 100, 250),
        mean_range(c, 250, 500),
        mean_range(c, 500, 1000),
        mean_range(c, 1000, 2000),
        reg(2000),
        ttt(0.95),
        ttt(0.90),
    ]
}

/// One (world, seed): build eval world and pool once, run every policy.
pub fn run_all(spec: &Spec, seed: u64, k_pool: usize, pols: &[Pol]) -> Vec<[f64; NM]> {
    let w = make_world(&spec.eval, seed);
    let deps = build_pool(&spec.pool, k_pool, seed);
    pols.iter()
        .map(|p| {
            let mut r = p.build(&w, &deps);
            let c = run_curve(&w, r.as_mut(), seed);
            metrics(&c, &w.oracle)
        })
        .collect()
}
