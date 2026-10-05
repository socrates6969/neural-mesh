//! G20/W21 shared code: worlds (fixed before any policy) + `SonaB`, a re-implementation of
//! `prior::Guarded<HebbianRouter>` (counts are private in the library) with OPT-IN exploration and
//! forgetting mechanisms. With every mechanism off it is bit-identical to `Guarded<HebbianRouter>`
//! (checked at start-up by `selftest`).
//!
//! SYNTHETIC worlds only; nothing here is validated on real traffic.

#![allow(dead_code)]

use std::collections::HashMap;

use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{GuardConfig, KeywordPrior};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World as SimWorld, WorldConfig};
use serde::{Deserialize, Serialize};

pub const N_TASKS: usize = 2000;

// ================================================================== worlds
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriftKind {
    /// half of the topics get a new best agent at a seed-dependent change point in [600,1100]
    Abrupt,
    /// the same change, but a linear ramp over tasks 500..1500
    Gradual,
    /// regime 2 during [500,1000) and [1500,2000); regime 1 otherwise (old knowledge returns)
    Recurrent,
}

#[derive(Clone, Debug)]
pub enum WorldKind {
    Short { topics: usize, traps: usize, map_err: f64, confound: f64 },
    Drift(DriftKind),
    Sim(Scenario),
}

#[derive(Clone, Debug)]
pub struct WorldSpec {
    pub name: &'static str,
    pub kind: WorldKind,
    /// "stat" = stationary (new mechanisms must not hurt), "drift" = drifting (decay should help), "sim" = library sim (contains its own drift types)
    pub class: &'static str,
}

pub struct World {
    pub agents: usize,
    pub tasks: Vec<Task>,
    pub p: Vec<Vec<f64>>,
    pub u: Vec<f64>,
    pub kwmap: HashMap<String, usize>,
    pub topic: Vec<usize>,
    pub down: Vec<Vec<bool>>,
}

fn gen_short_p(seed: u64, topics: usize, traps: usize, map_err: f64, confound: f64) -> (World, Vec<Vec<f64>>) {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(11));
    let (t_n, a_n) = (topics, 6usize);
    let mut p = vec![vec![0.0; a_n]; t_n];
    let mut best = vec![0usize; t_n];
    let mut second = vec![0usize; t_n];
    for t in 0..t_n {
        best[t] = rng.below(a_n);
        loop {
            second[t] = rng.below(a_n);
            if second[t] != best[t] {
                break;
            }
        }
        for a in 0..a_n {
            p[t][a] = if a == best[t] {
                0.85
            } else if a == second[t] {
                0.5
            } else {
                0.15 + 0.25 * rng.f64()
            };
        }
    }
    let mut order: Vec<usize> = (0..t_n).collect();
    for i in (1..t_n).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let trap: Vec<bool> = (0..t_n).map(|t| order.iter().take(traps).any(|&x| x == t)).collect();
    let mut kwmap = HashMap::new();
    for t in 0..t_n {
        for j in 0..3 {
            let a = if trap[t] {
                loop {
                    let a = rng.below(a_n);
                    if a != best[t] && a != second[t] {
                        break a;
                    }
                }
            } else if rng.f64() < map_err {
                rng.below(a_n)
            } else {
                best[t]
            };
            kwmap.insert(format!("t{t}k{j}"), a);
        }
    }
    let (mut tasks, mut pp, mut u, mut topic) = (vec![], vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let t = rng.below(t_n);
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(confound) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        pp.push(p[t].clone());
        u.push(rng.f64());
        topic.push(t);
    }
    (World { agents: a_n, tasks, p: pp, u, kwmap, topic, down: vec![vec![false; a_n]; N_TASKS] }, p)
}

fn gen_drift(seed: u64, kind: DriftKind) -> World {
    // identical stationary base as traps(main) (common random numbers with the stationary world)
    let (mut w, p0) = gen_short_p(seed, 8, 2, 0.10, 0.25);
    let t_n = p0.len();
    let a_n = w.agents;
    let mut r = Rng::new(seed.wrapping_mul(104_729).wrapping_add(977));
    // regime-2 profile: half of the topics get a different best agent (best/second/others re-drawn)
    let mut p1 = p0.clone();
    let mut changed = vec![false; t_n];
    for t in 0..t_n {
        if r.bernoulli(0.5) {
            changed[t] = true;
            // new best = a random agent that is not the old best or second best; old best demoted to .3
            let mut order: Vec<usize> = (0..a_n).collect();
            order.sort_by(|&x, &y| p0[t][y].total_cmp(&p0[t][x]));
            let (ob, os) = (order[0], order[1]);
            let nb = loop {
                let a = r.below(a_n);
                if a != ob && a != os {
                    break a;
                }
            };
            let ns = loop {
                let a = r.below(a_n);
                if a != nb {
                    break a;
                }
            };
            for a in 0..a_n {
                p1[t][a] = if a == nb {
                    0.85
                } else if a == ns {
                    0.5
                } else if a == ob {
                    0.25
                } else {
                    p0[t][a].min(0.4)
                };
            }
        }
    }
    let change = 600 + r.below(500);
    for i in 0..N_TASKS {
        let t = w.topic[i];
        let wt: f64 = match kind {
            DriftKind::Abrupt => (i >= change) as u8 as f64,
            DriftKind::Gradual => ((i as f64 - 500.0) / 1000.0).clamp(0.0, 1.0),
            DriftKind::Recurrent => ((500..1000).contains(&i) || i >= 1500) as u8 as f64,
        };
        if changed[t] {
            for a in 0..a_n {
                w.p[i][a] = (1.0 - wt) * p0[t][a] + wt * p1[t][a];
            }
        }
    }
    w
}

fn gen_sim(seed: u64, sc: Scenario) -> World {
    let w = SimWorld::new(seed, WorldConfig { n_tasks: N_TASKS, scenario: sc, ..WorldConfig::default() });
    let n = w.n_agents();
    let tasks: Vec<Task> = w.tasks().iter().map(|s| s.task.clone()).collect();
    let p: Vec<Vec<f64>> = (0..w.len()).map(|t| (0..n).map(|a| w.success_prob(t, &[a])).collect()).collect();
    let u: Vec<f64> = w.tasks().iter().map(|s| s.u).collect();
    let topic: Vec<usize> = w.tasks().iter().map(|s| s.type_idx).collect();
    let down: Vec<Vec<bool>> = (0..w.len()).map(|t| (0..n).map(|a| w.is_down(t, a)).collect()).collect();
    World { agents: n, tasks, p, u, kwmap: w.keyword_map_for(), topic, down }
}

pub fn gen_world(seed: u64, s: &WorldSpec) -> World {
    match &s.kind {
        WorldKind::Short { topics, traps, map_err, confound } => gen_short_p(seed, *topics, *traps, *map_err, *confound).0,
        WorldKind::Drift(k) => gen_drift(seed, *k),
        WorldKind::Sim(sc) => gen_sim(seed, *sc),
    }
}

/// FIXED before any policy was run (see results/g19b_board_entry.md).
pub fn all_worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec { name: "traps(main)", kind: WorldKind::Short { topics: 8, traps: 2, map_err: 0.10, confound: 0.25 }, class: "stat" },
        WorldSpec { name: "correct-keywords", kind: WorldKind::Short { topics: 8, traps: 0, map_err: 0.0, confound: 0.25 }, class: "stat" },
        WorldSpec { name: "many-topics(16)", kind: WorldKind::Short { topics: 16, traps: 2, map_err: 0.10, confound: 0.25 }, class: "stat" },
        WorldSpec { name: "drift-abrupt", kind: WorldKind::Drift(DriftKind::Abrupt), class: "drift" },
        WorldSpec { name: "drift-gradual", kind: WorldKind::Drift(DriftKind::Gradual), class: "drift" },
        WorldSpec { name: "drift-recurrent", kind: WorldKind::Drift(DriftKind::Recurrent), class: "drift" },
        WorldSpec { name: "sim-fixed", kind: WorldKind::Sim(Scenario::Fixed), class: "sim" },
        WorldSpec { name: "sim-random", kind: WorldKind::Sim(Scenario::Random), class: "sim" },
        WorldSpec { name: "sim-notraps", kind: WorldKind::Sim(Scenario::NoTraps), class: "sim" },
    ]
}

// ================================================================== SonaB
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InnerKind {
    /// hebbian-robust (as guarded(hebr))
    Heb,
    /// Thompson sampling over the pattern-conditioned (pooled-over-task-features) posteriors of the guard counts
    Ts,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BCfg {
    // ---- guard (== prior::GuardConfig) ----
    pub tau: f64,
    pub z: f64,
    pub kw_a: f64,
    pub kw_b: f64,
    pub discount: f64,
    pub challenge: bool,
    // ---- inner learner ----
    pub inner: InnerKind,
    pub heb_eta: f64,
    pub heb_epsilon: f64,
    pub heb_temperature: f64,
    // ---- weak path (same as SonaPlus) ----
    pub delayed_mult: usize,
    pub fail_reward: f64,
    // ---- EXPLORATION (all off at 0) ----
    /// pattern prior: agent a's prior on a new pattern = Beta(1 + lam*m_a, 1 + lam*(1-m_a)), m_a = its global smoothed mean success
    pub hier_lambda: f64,
    /// while following the keyword rule: with this prob. try the most promising challenger ...
    pub xp_eps: f64,
    /// ... at most this many times per pattern (budget; refilled on a drift alarm)
    pub xp_budget: u32,
    /// only challengers whose UCB reaches the keyword agent's LCB (uncertainty-directed)
    pub xp_plaus: bool,
    /// Thompson inner: posterior count scale (<1 widens, >1 sharpens)
    pub ts_scale: f64,
    /// Thompson inner: optimistic pseudo-successes for agents with < 3 pooled outcomes on the pattern
    pub ts_opt: f64,
    // ---- FORGETTING (all off at 0) ----
    /// lazy exponential decay of a pattern row by elapsed tasks since its last update (0 = off)
    pub time_tau: f64,
    /// EWC-like: weaken the per-update discount of rows whose best agent is proven good (0 = off)
    pub protect: f64,
    pub protect_v: f64,
    /// CUSUM drift detector on the chosen agent's pattern cell (det_lambda 0 = off)
    pub det_delta: f64,
    pub det_lambda: f64,
    /// on alarm the pattern row's evidence is scaled by this factor
    pub det_keep: f64,
    /// round-2 (post-hoc, see board entry): on an alarm the pattern bypasses the keyword rule/challenger (inner learner decides) for this many tasks (0 = off)
    pub det_ban: u32,
}

impl Default for BCfg {
    /// == frozen x1 guarded(hebbian-robust); every new mechanism off.
    fn default() -> Self {
        BCfg {
            tau: 0.55,
            z: 0.5,
            kw_a: 2.0,
            kw_b: 1.0,
            discount: 0.97,
            challenge: true,
            inner: InnerKind::Heb,
            heb_eta: 0.01,
            heb_epsilon: 0.0,
            heb_temperature: 0.002,
            delayed_mult: 1,
            fail_reward: 0.0,
            hier_lambda: 0.0,
            xp_eps: 0.0,
            xp_budget: 0,
            xp_plaus: true,
            ts_scale: 1.0,
            ts_opt: 0.0,
            time_tau: 0.0,
            protect: 0.0,
            protect_v: 0.7,
            det_delta: 0.15,
            det_lambda: 0.0,
            det_keep: 0.3,
            det_ban: 0,
        }
    }
}

impl BCfg {
    pub fn guard(&self) -> GuardConfig {
        GuardConfig { tau: self.tau, z: self.z, kw_a: self.kw_a, kw_b: self.kw_b, discount: self.discount, challenge: self.challenge }
    }
    pub fn explore_on(&self) -> bool {
        self.xp_eps > 0.0 && self.xp_budget > 0
    }
    pub fn forget_on(&self) -> bool {
        self.time_tau > 0.0 || self.protect > 0.0 || self.det_lambda > 0.0
    }
}

#[derive(Clone, Debug)]
struct Row {
    cells: Vec<(f64, f64)>,
    g: Vec<f64>,
    last_t: u64,
    explored: u32,
    ban_until: u64,
}

pub struct SonaB {
    pub cfg: BCfg,
    prior: KeywordPrior,
    heb: HebbianRouter,
    rows: HashMap<String, Row>,
    glob: Vec<(f64, f64)>,
    n_agents: usize,
    t: u64,
    /// [keyword followed, challenger, inner, exploration, drift alarms]
    pub stats: [u64; 5],
}

fn beta_bounds(a: f64, b: f64, z: f64) -> (f64, f64) {
    let n = a + b;
    let m = a / n;
    let sd = (m * (1.0 - m) / (n + 1.0)).sqrt();
    (m - z * sd, m + z * sd)
}

fn normal(rng: &mut Rng) -> f64 {
    let u1 = rng.f64().max(1e-300);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gamma(shape>=1,1) (Marsaglia-Tsang); shape<1 boosted.
fn gamma(rng: &mut Rng, shape: f64) -> f64 {
    if shape < 1.0 {
        let g = gamma(rng, shape + 1.0);
        return g * rng.f64().max(1e-300).powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = rng.f64().max(1e-300);
        if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

fn beta_sample(rng: &mut Rng, a: f64, b: f64) -> f64 {
    let x = gamma(rng, a);
    let y = gamma(rng, b);
    x / (x + y).max(1e-300)
}

impl SonaB {
    pub fn new(n_agents: usize, kwmap: &HashMap<String, usize>, cfg: BCfg) -> Self {
        let heb = HebbianRouter::new(
            n_agents,
            HebbianConfig { eta: cfg.heb_eta, temperature: cfg.heb_temperature, decay: 0.01, epsilon: cfg.heb_epsilon, ..HebbianConfig::default() },
        );
        SonaB { cfg, prior: KeywordPrior::new(kwmap.clone(), n_agents), heb, rows: HashMap::new(), glob: vec![(0.0, 0.0); n_agents], n_agents, t: 0, stats: [0; 5] }
    }

    #[inline]
    fn tfac(&self, row: &Row) -> f64 {
        if self.cfg.time_tau > 0.0 { (-((self.t.saturating_sub(row.last_t)) as f64) / self.cfg.time_tau).exp() } else { 1.0 }
    }

    /// pooled (successes, failures) of `agent` over the task's features (mean), with lazy time decay.
    fn pooled(&self, task: &Task, agent: usize) -> (f64, f64) {
        if task.features.is_empty() {
            return (0.0, 0.0);
        }
        let (mut s, mut f) = (0.0, 0.0);
        for k in &task.features {
            if let Some(row) = self.rows.get(k)
                && let Some(&(cs, cf)) = row.cells.get(agent)
            {
                let fac = self.tfac(row);
                s += cs * fac;
                f += cf * fac;
            }
        }
        let n = task.features.len() as f64;
        (s / n, f / n)
    }

    fn keyword_bounds(&self, task: &Task, agent: usize) -> (f64, f64) {
        let (s, f) = self.pooled(task, agent);
        beta_bounds(self.cfg.kw_a + s, self.cfg.kw_b + f, self.cfg.z)
    }

    /// prior (a,b) of a NON-keyword agent on a pattern.
    fn agent_prior(&self, agent: usize) -> (f64, f64) {
        let l = self.cfg.hier_lambda;
        if l <= 0.0 {
            return (1.0, 1.0);
        }
        let (gs, gf) = self.glob.get(agent).copied().unwrap_or((0.0, 0.0));
        let m = (gs + 1.0) / (gs + gf + 2.0);
        (1.0 + l * m, 1.0 + l * (1.0 - m))
    }

    fn decide(&self, task: &Task, candidates: &[usize]) -> (u8, usize) {
        if self.cfg.det_ban > 0 && !task.features.is_empty() {
            let nb = task.features.iter().filter(|k| self.rows.get(*k).is_some_and(|r| r.ban_until > self.t)).count();
            if nb * 2 >= task.features.len() {
                return (2, 0);
            }
        }
        let votes = self.prior.votes(task, candidates);
        let Some(k) = votes
            .iter()
            .fold(None, |b: Option<(usize, usize)>, &(a, v)| match b {
                Some((_, bv)) if bv >= v => b,
                _ => Some((a, v)),
            })
            .map(|(a, _)| a)
        else {
            return (2, 0);
        };
        let (_, ucb_k) = self.keyword_bounds(task, k);
        if ucb_k < self.cfg.tau {
            return (2, 0);
        }
        if self.cfg.challenge {
            let mut best = (ucb_k, usize::MAX);
            for &a in candidates {
                if a == k {
                    continue;
                }
                let (s, f) = self.pooled(task, a);
                let (pa, pb) = self.agent_prior(a);
                let (lcb, _) = beta_bounds(pa + s, pb + f, self.cfg.z);
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

    fn select_ts(&self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let mut best = (f64::NEG_INFINITY, candidates[0]);
        for &a in candidates {
            let (s, f) = self.pooled(task, a);
            let (pa, pb) = self.agent_prior(a);
            let opt = if self.cfg.ts_opt > 0.0 && s + f < 3.0 { self.cfg.ts_opt } else { 0.0 };
            let x = beta_sample(rng, pa + self.cfg.ts_scale * s + opt, pb + self.cfg.ts_scale * f);
            if x > best.0 {
                best = (x, a);
            }
        }
        best.1
    }

    fn budget_left(&self, task: &Task) -> bool {
        if task.features.is_empty() {
            return false;
        }
        task.features.iter().all(|k| self.rows.get(k).map_or(0, |r| r.explored) < self.cfg.xp_budget)
    }
}

impl Router for SonaB {
    fn name(&self) -> String {
        format!("sonab(tau={},{:?})", self.cfg.tau, self.cfg.inner)
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "SonaB::select: no candidates");
        self.t += 1;
        let (d, a) = self.decide(task, candidates);
        if d == 2 {
            self.stats[2] += 1;
            return match self.cfg.inner {
                InnerKind::Heb => self.heb.select(task, candidates, rng),
                InnerKind::Ts => self.select_ts(task, candidates, rng),
            };
        }
        if d == 0 && self.cfg.explore_on() && candidates.len() > 1 && self.budget_left(task) && rng.f64() < self.cfg.xp_eps {
            let lcb_k = self.keyword_bounds(task, a).0;
            let mut best: Option<(f64, usize)> = None;
            for &c in candidates {
                if c == a {
                    continue;
                }
                let (s, f) = self.pooled(task, c);
                let (pa, pb) = self.agent_prior(c);
                let (_, ucb) = beta_bounds(pa + s, pb + f, self.cfg.z.max(1.0));
                if self.cfg.xp_plaus && ucb < lcb_k {
                    continue;
                }
                if best.is_none_or(|(bu, _)| ucb > bu) {
                    best = Some((ucb, c));
                }
            }
            if let Some((_, c)) = best {
                let (na, tn) = (self.n_agents, self.t);
                for k in &task.features {
                    // an empty row is equivalent to a missing one everywhere else
                    self.rows.entry(k.clone()).or_insert_with(|| Row { cells: vec![(0.0, 0.0); na], g: vec![0.0; na], last_t: tn, explored: 0, ban_until: 0 }).explored += 1;
                }
                self.stats[3] += 1;
                return c;
            }
        }
        self.stats[d as usize] += 1;
        a
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        let n = self.n_agents.max(agent + 1);
        let c = self.cfg.clone();
        for k in &task.features {
            let t_now = self.t;
            let row = self.rows.entry(k.clone()).or_insert_with(|| Row { cells: vec![(0.0, 0.0); n], g: vec![0.0; n], last_t: t_now, explored: 0, ban_until: 0 });
            if row.cells.len() < n {
                row.cells.resize(n, (0.0, 0.0));
                row.g.resize(n, 0.0);
            }
            // lazy time decay: materialise
            if c.time_tau > 0.0 {
                let fac = (-((t_now.saturating_sub(row.last_t)) as f64) / c.time_tau).exp();
                if fac < 1.0 {
                    for cell in row.cells.iter_mut() {
                        cell.0 *= fac;
                        cell.1 *= fac;
                    }
                }
            }
            row.last_t = t_now;
            // drift detector (downward shift of the chosen agent's success on this pattern)
            if c.det_lambda > 0.0 {
                let (s, f) = row.cells[agent];
                if s + f >= 4.0 {
                    let m = s / (s + f);
                    let g = (row.g[agent] + (m - r) - c.det_delta).max(0.0);
                    row.g[agent] = g;
                    if g > c.det_lambda {
                        for cell in row.cells.iter_mut() {
                            cell.0 *= c.det_keep;
                            cell.1 *= c.det_keep;
                        }
                        for x in row.g.iter_mut() {
                            *x = 0.0;
                        }
                        row.explored = 0;
                        if c.det_ban > 0 {
                            row.ban_until = t_now + c.det_ban as u64;
                        }
                        self.stats[4] += 1;
                    }
                }
            }
            // per-update discount (EWC-like protection of proven rows)
            let mut disc = c.discount;
            if c.protect > 0.0 && disc < 1.0 {
                let mut stab = 0.0;
                for &(s, f) in row.cells.iter() {
                    let nn = s + f;
                    if nn >= 3.0 && s / nn >= c.protect_v {
                        stab = f64::max(stab, nn / (nn + 5.0));
                    }
                }
                disc = 1.0 - (1.0 - disc) * (1.0 - c.protect * stab);
            }
            if disc < 1.0 {
                for cell in row.cells.iter_mut() {
                    cell.0 *= disc;
                    cell.1 *= disc;
                }
            }
            row.cells[agent].0 += r;
            row.cells[agent].1 += 1.0 - r;
        }
        if self.glob.len() < n {
            self.glob.resize(n, (0.0, 0.0));
        }
        self.glob[agent].0 += r;
        self.glob[agent].1 += 1.0 - r;
        if c.inner == InnerKind::Heb {
            self.heb.update(task, agent, reward);
        }
    }
}

impl SonaB {
    pub fn update_weak(&mut self, task: &Task, agent: usize, raw_ok: bool) {
        if !raw_ok {
            let r = self.cfg.fail_reward;
            self.update(task, agent, r);
        }
    }
    pub fn update_delayed(&mut self, task: &Task, agent: usize, value: f64) {
        for _ in 0..self.cfg.delayed_mult.max(1) {
            self.update(task, agent, value);
        }
    }
}
