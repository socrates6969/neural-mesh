//! G18 common (W19, ORCHESTRATION track O): a simulated multi-step research-task world, single-shot baselines and
//! five orchestrator levels (decompose+route, +retrieval, +verify/retry/reroute, +learned stopping, +parallel fan-out).
//!
//! EVERYTHING here is a seeded simulation written by me. It measures whether decomposing a 3-6 step task, routing each
//! step by type, giving each step recurring context, verifying steps (imperfect recall) and re-routing failures improves a
//! graded quality that I defined (and what it costs). The effect sizes are set by world parameters I chose (specialist gap,
//! propagation strength, context dependence, verifier noise); it says nothing about real LLM behaviour or discovery.
//!
//! WORLDS ARE FIXED BEFORE POLICIES: the definitions in `worlds()` were written and `--show`n before any router/orchestrator was run.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::defaults::recommended_router;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};

pub const N_TASKS: usize = 1500;
pub const REVEAL: f64 = 0.6;
pub const DELAY: u64 = 15;
pub const NTYPE: usize = 4;
pub const TYPES: [&str; NTYPE] = ["lookup", "compute", "synthesize", "check"];
/// a step is "correct" iff its (post-propagation) quality >= THETA
pub const THETA: f64 = 0.6;
/// the verifier flags a step iff its noisy signal < THETA_V
pub const THETA_V: f64 = 0.55;
pub const CR: f64 = 0.05; // cost of one retrieval
pub const GRADE_SD: f64 = 0.05;
pub const STEP_DIFF_SD: f64 = 0.07;
pub const MAX_RETRY: usize = 2;
pub const STORE_MIN: f64 = 0.6;
pub const MU_A: f64 = 0.1;
pub const MU_B: f64 = 0.3;
pub const MU_C: f64 = 0.03;

// ---------------------------------------------------------------- stateless randomness (common random numbers)
pub fn hash(seed: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ a.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ b.wrapping_mul(0x94D0_49BB_1331_11EB) ^ c.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
pub fn uni(seed: u64, a: u64, b: u64, c: u64) -> f64 {
    (hash(seed, a, b, c) >> 11) as f64 / (1u64 << 53) as f64
}
pub fn norm(seed: u64, a: u64, b: u64, c: u64) -> f64 {
    let (u1, u2) = (uni(seed, a, b, c).max(1e-12), uni(seed, a, b, c ^ 0xABCD));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn rnorm(rng: &mut Rng) -> f64 {
    let (u1, u2) = (rng.f64().max(1e-12), rng.f64());
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

// ---------------------------------------------------------------- world config
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gen {
    /// generator A: template pipelines, specialist agents, mean-propagation
    Pipe,
    /// generator B: random DAGs, agents from 3 latent factors, min-propagation (weakest input dominates)
    Latent,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pool {
    /// 8 agents: 2 generalists (~.64), 4 type specialists (~.64+gap in own type, ~.46 off), 1 lookup+check dual, 1 weak
    Spec,
    /// 8 agents from 3 latent factors (cross-type correlation)
    Latent,
    /// agent 0 best at every type (.88), others ~.6: decomposition has no specialisation to exploit
    Dominant,
    /// all agents ~.92 on every type, low noise: errors are rare
    HighSkill,
    /// 8 identical agents (.7)
    Equal,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Chain,
    /// each step depends on its predecessor (p .6) or a random earlier step, sometimes a second one; last step joins all leaves
    Dag,
    /// independent roots, a merge step depending on all roots, then a short chain
    Wide,
    /// random edges with probability p
    RandDag(f64),
    /// no dependencies at all
    Indep,
    /// one step per task
    Single,
}

#[derive(Clone, Debug)]
pub struct WorldCfg {
    pub name: &'static str,
    pub family: Gen,
    pub pool: Pool,
    pub shape: Shape,
    pub nmin: usize,
    pub nmax: usize,
    pub type_mix: [f64; NTYPE],
    /// propagation strength: q = raw * fit * (1 - prop * (1 - q_in))
    pub prop: f64,
    /// agent noise sd (per task x step x agent, fixed across arms)
    pub noise: f64,
    /// context dependence per step type (a draft with no brief keeps 1-d of its quality)
    pub d: [f64; NTYPE],
    pub dscale: f64,
    pub projects: usize,
    /// P(retrieval returns the right project's note when it exists)
    pub p_id: f64,
    /// extra penalty for a WRONG note
    pub mis: f64,
    /// verifier noise sd
    pub sig_v: f64,
    /// cost of one verification call (an agent step costs 1)
    pub cv: f64,
    /// P(a type keyword advertises a random wrong agent)
    pub decoy: f64,
    /// extra loss of quality when ONE agent does the whole task in one call (0 in all main worlds)
    pub mono_pen: f64,
    pub spec_gap: f64,
}

const D_PIPE: [f64; NTYPE] = [0.25, 0.05, 0.18, 0.08];
const D_LAT: [f64; NTYPE] = [0.20, 0.08, 0.20, 0.10];
const MIX_PIPE: [f64; NTYPE] = [0.30, 0.25, 0.25, 0.20];
const MIX_LAT: [f64; NTYPE] = [0.25, 0.25, 0.30, 0.20];

pub fn worlds() -> Vec<WorldCfg> {
    let pipe = WorldCfg {
        name: "pipe-main", family: Gen::Pipe, pool: Pool::Spec, shape: Shape::Dag, nmin: 3, nmax: 6, type_mix: MIX_PIPE, prop: 0.5, noise: 0.10, d: D_PIPE, dscale: 1.0,
        projects: 10, p_id: 0.9, mis: 0.5, sig_v: 0.10, cv: 0.3, decoy: 0.3, mono_pen: 0.0, spec_gap: 0.26,
    };
    let lat = WorldCfg {
        name: "lat-main", family: Gen::Latent, pool: Pool::Latent, shape: Shape::RandDag(0.35), nmin: 3, nmax: 6, type_mix: MIX_LAT, prop: 0.6, noise: 0.10, d: D_LAT, dscale: 1.0,
        projects: 12, p_id: 0.85, mis: 0.5, sig_v: 0.10, cv: 0.3, decoy: 0.3, mono_pen: 0.0, spec_gap: 0.0,
    };
    vec![
        // ---------------- treatment worlds, generator A (pipelines, specialists)
        pipe.clone(),
        WorldCfg { name: "pipe-chain", shape: Shape::Chain, ..pipe.clone() },
        WorldCfg { name: "pipe-wide", shape: Shape::Wide, ..pipe.clone() },
        WorldCfg { name: "pipe-weakctx", dscale: 0.4, ..pipe.clone() },
        WorldCfg { name: "pipe-lowrec", projects: 300, ..pipe.clone() },
        WorldCfg { name: "pipe-badver", sig_v: 0.22, ..pipe.clone() },
        // ---------------- treatment worlds, generator B (random DAGs, latent-factor agents)
        lat.clone(),
        WorldCfg { name: "lat-highprop", prop: 0.9, ..lat.clone() },
        WorldCfg { name: "lat-noisyag", noise: 0.16, ..lat.clone() },
        WorldCfg { name: "lat-pricever", cv: 0.8, ..lat.clone() },
        // ---------------- controls
        WorldCfg { name: "ctl-single-step", shape: Shape::Single, nmin: 1, nmax: 1, ..pipe.clone() },
        WorldCfg { name: "ctl-indep-dominant", shape: Shape::Indep, pool: Pool::Dominant, ..pipe.clone() },
        WorldCfg { name: "ctl-indep-spec", shape: Shape::Indep, ..pipe.clone() },
        WorldCfg { name: "ctl-rare-pricever", pool: Pool::HighSkill, noise: 0.06, cv: 1.0, d: [0.05, 0.02, 0.04, 0.02], ..pipe.clone() },
        WorldCfg { name: "ctl-equal-agents", pool: Pool::Equal, ..pipe.clone() },
    ]
}
pub fn is_control(name: &str) -> bool {
    name.starts_with("ctl-")
}

// ---------------------------------------------------------------- instance
#[derive(Clone, Debug)]
pub struct Step {
    pub ty: usize,
    pub deps: Vec<usize>,
    /// task x step difficulty offset (shared by all agents and arms)
    pub diff: f64,
}
#[derive(Clone, Debug)]
pub struct TaskI {
    pub project: usize,
    pub steps: Vec<Step>,
    /// view for a single-shot router (whole task)
    pub whole: Task,
}
pub struct Instance {
    pub n_agents: usize,
    /// true competence per agent x step type (hidden from routers)
    pub comp: Vec<[f64; NTYPE]>,
    pub kw: HashMap<String, usize>,
    pub tasks: Vec<TaskI>,
}

fn build_comp(w: &WorldCfg, r: &mut Rng) -> Vec<[f64; NTYPE]> {
    let jit = |r: &mut Rng, base: f64, sp: f64| (base + sp * (r.f64() * 2.0 - 1.0)).clamp(0.05, 0.99);
    let mut v: Vec<[f64; NTYPE]> = Vec::new();
    match w.pool {
        Pool::Spec => {
            for _ in 0..2 {
                v.push([jit(r, 0.64, 0.03), jit(r, 0.64, 0.03), jit(r, 0.64, 0.03), jit(r, 0.64, 0.03)]);
            }
            for t in 0..NTYPE {
                let mut c = [0.0; NTYPE];
                for (u, x) in c.iter_mut().enumerate() {
                    *x = if u == t { jit(r, 0.64 + w.spec_gap, 0.03) } else { jit(r, 0.46, 0.05) };
                }
                v.push(c);
            }
            v.push([jit(r, 0.80, 0.03), jit(r, 0.5, 0.05), jit(r, 0.5, 0.05), jit(r, 0.80, 0.03)]);
            v.push([jit(r, 0.42, 0.04), jit(r, 0.42, 0.04), jit(r, 0.42, 0.04), jit(r, 0.42, 0.04)]);
        }
        Pool::Latent => {
            const LOAD: [[f64; 3]; NTYPE] = [[0.8, 0.1, 0.3], [0.1, 0.9, 0.2], [0.3, 0.2, 0.9], [0.4, 0.6, 0.3]];
            for _ in 0..8 {
                let z = [rnorm(r), rnorm(r), rnorm(r)];
                let mut c = [0.0; NTYPE];
                for t in 0..NTYPE {
                    let s: f64 = (0..3).map(|k| LOAD[t][k] * z[k]).sum();
                    c[t] = (0.66 + 0.11 * s + 0.03 * rnorm(r)).clamp(0.05, 0.97);
                }
                v.push(c);
            }
        }
        Pool::Dominant => {
            let a = jit(r, 0.88, 0.02);
            v.push([a; NTYPE]);
            for _ in 0..7 {
                v.push([jit(r, 0.6, 0.05), jit(r, 0.6, 0.05), jit(r, 0.6, 0.05), jit(r, 0.6, 0.05)]);
            }
        }
        Pool::HighSkill => {
            for _ in 0..8 {
                v.push([jit(r, 0.92, 0.02), jit(r, 0.92, 0.02), jit(r, 0.92, 0.02), jit(r, 0.92, 0.02)]);
            }
        }
        Pool::Equal => {
            for _ in 0..8 {
                v.push([0.7; NTYPE]);
            }
        }
    }
    for i in (1..v.len()).rev() {
        let j = r.below(i + 1);
        v.swap(i, j);
    }
    v
}

fn draw_type(mix: &[f64; NTYPE], r: &mut Rng) -> usize {
    let mut u = r.f64() * mix.iter().sum::<f64>();
    for (t, m) in mix.iter().enumerate() {
        if u < *m {
            return t;
        }
        u -= m;
    }
    NTYPE - 1
}

pub fn build(w: &WorldCfg, seed: u64, widx: u64) -> Instance {
    let mut r = Rng::new(seed.wrapping_mul(1_000_003) ^ (widx * 7919 + 17));
    let comp = build_comp(w, &mut r);
    let na = comp.len();
    let mut kw = HashMap::new();
    for (t, name) in TYPES.iter().enumerate() {
        let best = (0..na).fold(0, |b, a| if comp[a][t] > comp[b][t] { a } else { b });
        let a = if r.f64() < w.decoy { r.below(na) } else { best };
        kw.insert(name.to_string(), a);
    }
    let mut tasks = Vec::with_capacity(N_TASKS);
    for _ in 0..N_TASKS {
        let ti = tasks.len();
        let n = w.nmin + r.below(w.nmax - w.nmin + 1);
        let mut types: Vec<usize> = (0..n).map(|_| draw_type(&w.type_mix, &mut r)).collect();
        if n >= 3 && matches!(w.family, Gen::Pipe) && r.f64() < 0.7 {
            types[n - 1] = 2; // pipelines usually end in a synthesis
        }
        let mut deps: Vec<Vec<usize>> = vec![vec![]; n];
        match w.shape {
            Shape::Chain => {
                for k in 1..n {
                    deps[k] = vec![k - 1];
                }
            }
            Shape::Dag => {
                for k in 1..n {
                    let p = if r.f64() < 0.6 { k - 1 } else { r.below(k) };
                    deps[k].push(p);
                    if k >= 2 && r.f64() < 0.25 {
                        let q = r.below(k);
                        if !deps[k].contains(&q) {
                            deps[k].push(q);
                        }
                    }
                }
                if n >= 3 {
                    let mut has_child = vec![false; n];
                    for k in 0..n {
                        for &d in &deps[k] {
                            has_child[d] = true;
                        }
                    }
                    for j in 0..n - 1 {
                        if !has_child[j] && !deps[n - 1].contains(&j) {
                            deps[n - 1].push(j);
                        }
                    }
                }
            }
            Shape::Wide => {
                let m = if n <= 4 { 2 } else { n - 3 };
                let m = m.max(2).min(n - 1);
                for k in m..n {
                    deps[k] = if k == m { (0..m).collect() } else { vec![k - 1] };
                }
            }
            Shape::RandDag(p) => {
                for k in 1..n {
                    for j in 0..k {
                        if r.f64() < p {
                            deps[k].push(j);
                        }
                    }
                    if deps[k].is_empty() && r.f64() < 0.3 {
                        deps[k].push(k - 1);
                    }
                }
            }
            Shape::Indep | Shape::Single => {}
        }
        let steps: Vec<Step> = (0..n).map(|k| Step { ty: types[k], deps: deps[k].clone(), diff: STEP_DIFF_SD * rnorm(&mut r) }).collect();
        let project = r.below(w.projects);
        let mut feats: Vec<String> = types.iter().map(|&t| TYPES[t].to_string()).collect();
        feats.sort();
        feats.dedup();
        let mut sig: Vec<usize> = types.clone();
        sig.sort();
        let tt = sig.iter().map(|&t| TYPES[t]).collect::<Vec<_>>().join("+");
        feats.push(format!("proj:{project}"));
        let whole = Task { id: ti as u64, task_type: Some(tt), features: feats };
        tasks.push(TaskI { project, steps, whole });
    }
    Instance { n_agents: na, comp, kw, tasks }
}

// ---------------------------------------------------------------- execution model
fn q_in(w: &WorldCfg, tk: &TaskI, si: usize, qs: &[f64]) -> f64 {
    let deps = &tk.steps[si].deps;
    if deps.is_empty() {
        return 1.0;
    }
    match w.family {
        Gen::Pipe => deps.iter().map(|&d| qs[d]).sum::<f64>() / deps.len() as f64,
        Gen::Latent => deps.iter().map(|&d| qs[d]).fold(1.0, f64::min),
    }
}

/// quality of agent `agent` on step `si` of task `ti`. Deterministic given the world: the same agent always produces the same raw
/// answer to the same step; context (`neff`) and upstream quality (`qin`) modulate it.
pub fn step_q(w: &WorldCfg, inst: &Instance, seed: u64, ti: usize, si: usize, agent: usize, qin: f64, neff: f64, mono: bool) -> f64 {
    let st = &inst.tasks[ti].steps[si];
    let c = inst.comp[agent][st.ty];
    let raw = (c + st.diff + w.noise * norm(seed, 0x51, (ti * 8 + si) as u64, agent as u64)).clamp(0.0, 1.0);
    let u = 0.5 + 0.5 * c;
    let d = w.d[st.ty] * w.dscale;
    let fit = (1.0 - d * (1.0 - neff * u)).clamp(0.0, 1.0);
    let prop = 1.0 - w.prop * (1.0 - qin);
    let mut q = raw * fit * prop;
    if mono {
        q *= 1.0 - w.mono_pen;
    }
    q.clamp(0.0, 1.0)
}

/// recurring-project notes: key (project, step type). Retrieval identification is stochastic (CRN-hashed).
#[derive(Default)]
pub struct Notes {
    s: HashMap<(usize, usize), f64>,
    per_type: [usize; NTYPE],
}
impl Notes {
    pub fn retrieve(&self, w: &WorldCfg, seed: u64, ti: usize, si: usize, proj: usize, ty: usize) -> f64 {
        let has = self.s.get(&(proj, ty)).copied();
        let other = self.per_type[ty] > usize::from(has.is_some());
        let u = uni(seed, 0x4E, (ti * 8 + si) as u64, 0);
        match has {
            Some(s) => {
                if u < w.p_id {
                    s
                } else if other && u < w.p_id + (1.0 - w.p_id) * 0.5 {
                    -w.mis
                } else {
                    0.0
                }
            }
            None => {
                if other && u < (1.0 - w.p_id) * 0.5 {
                    -w.mis
                } else {
                    0.0
                }
            }
        }
    }
    pub fn store(&mut self, key: (usize, usize), g: f64, store_min: f64) {
        if g < store_min {
            return;
        }
        match self.s.get_mut(&key) {
            Some(s) => *s = (0.7 * *s + 0.3 * g).min(1.0),
            None => {
                self.s.insert(key, g.min(1.0));
                self.per_type[key.1] += 1;
            }
        }
    }
}

// ---------------------------------------------------------------- routers
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SsKind {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    HebR,
    Warm(f64),
    Safe,
    OracleSingle,
}
pub fn mk_ss(kind: &SsKind, inst: &Instance) -> Option<Box<dyn Router>> {
    let n = inst.n_agents;
    let km = || inst.kw.clone();
    Some(match kind {
        SsKind::Keyword => Box::new(KeywordRouter::new(km(), n)),
        SsKind::Eps(g, e) => Box::new(EpsGreedyRouter::new(n, *g, *e)),
        SsKind::Pts(g) => Box::new(PooledThompsonRouter::new(n, *g)),
        SsKind::HebR => Box::new(HebbianRouter::new(n, heb_robust_cfg())),
        SsKind::Warm(k) => Box::new(Warm::new(HebbianRouter::new(n, heb_robust_cfg()), km(), n, *k).with_evidence(Evidence::MeanFeature)),
        SsKind::Safe => Box::new(recommended_router(km(), n)),
        SsKind::OracleSingle => return None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepRouter {
    Warm(f64),
    Safe,
    /// best agent per step type by TRUE competence (reference only); bool = also perfect context
    Oracle(bool),
}
#[derive(Clone, Copy, Debug)]
pub struct OrchCfg {
    pub router: StepRouter,
    pub retr: bool,
    pub verify: bool,
    /// verification signal is also used as immediate router feedback and to store notes immediately
    pub ver_learn: bool,
    pub stopping: bool,
    pub parallel: bool,
    pub max_retry: usize,
    /// learned stopping: verify iff ucb(flag rate of (agent,type)) * r_val >= cv, or w.p. eps_ex
    pub r_val: f64,
    pub eps_ex: f64,
    pub store_min: f64,
}
impl OrchCfg {
    pub fn level(l: u8, k: f64, r_val: f64, eps_ex: f64) -> OrchCfg {
        OrchCfg {
            router: StepRouter::Warm(k),
            retr: l >= 2,
            verify: l >= 3,
            ver_learn: l >= 3,
            stopping: l >= 4,
            parallel: l >= 5,
            max_retry: MAX_RETRY,
            r_val,
            eps_ex,
            store_min: STORE_MIN,
        }
    }
}

// ---------------------------------------------------------------- run output
pub struct RunOut {
    pub q: Vec<f32>,
    pub ok: Vec<u8>,
    pub cost: Vec<f32>,
    pub calls: Vec<u8>,
    pub vcalls: Vec<u8>,
    pub lat: Vec<f32>,
}
impl RunOut {
    fn new() -> Self {
        RunOut { q: Vec::with_capacity(N_TASKS), ok: vec![], cost: vec![], calls: vec![], vcalls: vec![], lat: vec![] }
    }
}

struct Ev {
    due: u64,
    upd: Vec<(Task, usize, f64)>,
    notes: Vec<((usize, usize), f64)>,
}

fn drain(pend: &mut VecDeque<Ev>, tick: u64, mut router: Option<&mut Box<dyn Router>>, notes: &mut Notes, store_min: f64, retr: bool) {
    while pend.front().is_some_and(|e| e.due <= tick) {
        let ev = pend.pop_front().unwrap();
        if let Some(r) = router.as_deref_mut() {
            for (t, a, g) in &ev.upd {
                r.update(t, *a, *g);
            }
        }
        if retr {
            for (k, g) in &ev.notes {
                notes.store(*k, *g, store_min);
            }
        }
    }
}

fn mean_f(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

/// SINGLE-SHOT: one router call per task; the chosen agent does every step. `retr`: reference arm that gives that agent the same notes.
pub fn run_single(w: &WorldCfg, inst: &Instance, kind: &SsKind, retr: bool, seed: u64) -> RunOut {
    let na = inst.n_agents;
    let cands: Vec<usize> = (0..na).collect();
    let mut router = mk_ss(kind, inst);
    let mut rng = Rng::new(seed * 7 + 1);
    let mut notes = Notes::default();
    let mut pend: VecDeque<Ev> = VecDeque::new();
    let mut out = RunOut::new();
    for (ti, tk) in inst.tasks.iter().enumerate() {
        let tick = ti as u64 + 1;
        drain(&mut pend, tick, router.as_mut(), &mut notes, STORE_MIN, retr);
        let n = tk.steps.len();
        let agent = match router.as_mut() {
            Some(r) => r.select(&tk.whole, &cands, &mut rng),
            None => {
                let score = |a: usize| tk.steps.iter().map(|s| inst.comp[a][s.ty]).sum::<f64>();
                (0..na).fold(0, |b, a| if score(a) > score(b) { a } else { b })
            }
        };
        let mut qs = vec![0.0; n];
        let mut cost = n as f64;
        for si in 0..n {
            let qin = q_in(w, tk, si, &qs);
            let neff = if retr {
                cost += CR;
                notes.retrieve(w, seed, ti, si, tk.project, tk.steps[si].ty)
            } else {
                0.0
            };
            qs[si] = step_q(w, inst, seed, ti, si, agent, qin, neff, true);
        }
        let tq = mean_f(&qs);
        out.q.push(tq as f32);
        out.ok.push(u8::from(qs.iter().all(|&q| q >= THETA)));
        out.cost.push(cost as f32);
        out.calls.push(1);
        out.vcalls.push(0);
        out.lat.push(n as f32);
        if router.is_some() && uni(seed, 0x2E, ti as u64, 0) < REVEAL {
            let g = (tq + GRADE_SD * norm(seed, 0x6D, ti as u64, 99)).clamp(0.0, 1.0);
            pend.push_back(Ev { due: tick + DELAY, upd: vec![(tk.whole.clone(), agent, g)], notes: tk.steps.iter().map(|s| ((tk.project, s.ty), g)).collect() });
        }
    }
    out
}

/// immediate (same-task) information from a verification call
struct Imm {
    task: Task,
    agent: usize,
    v: f64,
    key: (usize, usize),
    flagged: bool,
    note: (usize, usize),
}

fn flush(deferred: &mut Vec<Imm>, router: &mut Option<Box<dyn Router>>, vstats: &mut HashMap<(usize, usize), (f64, f64)>, notes: &mut Notes, oc: &OrchCfg) {
    for im in deferred.drain(..) {
        let e = vstats.entry(im.key).or_insert((0.0, 0.0));
        e.0 += 1.0;
        if im.flagged {
            e.1 += 1.0;
        }
        if oc.ver_learn {
            if let Some(r) = router.as_mut() {
                r.update(&im.task, im.agent, im.v);
            }
            if oc.retr && !im.flagged {
                notes.store(im.note, im.v, oc.store_min);
            }
        }
    }
}

/// ORCHESTRATOR (levels 1-5 by flags).
pub fn run_orch(w: &WorldCfg, inst: &Instance, oc: &OrchCfg, seed: u64) -> RunOut {
    let na = inst.n_agents;
    let all: Vec<usize> = (0..na).collect();
    let mut rng = Rng::new(seed * 7 + 1);
    let mut router: Option<Box<dyn Router>> = match oc.router {
        StepRouter::Warm(k) => Some(Box::new(Warm::new(HebbianRouter::new(na, heb_robust_cfg()), inst.kw.clone(), na, k).with_evidence(Evidence::MeanFeature))),
        StepRouter::Safe => Some(Box::new(recommended_router(inst.kw.clone(), na))),
        StepRouter::Oracle(_) => None,
    };
    let perfect_ctx = matches!(oc.router, StepRouter::Oracle(true));
    let mut notes = Notes::default();
    let mut pend: VecDeque<Ev> = VecDeque::new();
    let mut vstats: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    let mut out = RunOut::new();
    for (ti, tk) in inst.tasks.iter().enumerate() {
        let tick = ti as u64 + 1;
        drain(&mut pend, tick, router.as_mut(), &mut notes, oc.store_min, oc.retr);
        let n = tk.steps.len();
        let mut qs = vec![0.0; n];
        let mut fin = vec![0.0f64; n];
        let (mut cost, mut calls, mut vcalls, mut lat_seq) = (0.0f64, 0u32, 0u32, 0.0f64);
        let mut fb: Vec<(Task, usize, f64)> = vec![];
        let mut note_fb: Vec<((usize, usize), f64)> = vec![];
        let mut deferred: Vec<Imm> = vec![];
        for si in 0..n {
            let st = &tk.steps[si];
            let qin = q_in(w, tk, si, &qs);
            let stask = Task { id: (ti * 8 + si) as u64, task_type: Some(TYPES[st.ty].to_string()), features: vec![TYPES[st.ty].to_string()] };
            let neff = if perfect_ctx {
                1.0
            } else if oc.retr {
                cost += CR;
                notes.retrieve(w, seed, ti, si, tk.project, st.ty)
            } else {
                0.0
            };
            let mut tried: Vec<usize> = vec![];
            let mut best: (f64, f64, usize) = (-1.0, 0.0, 0);
            let mut log: Vec<(usize, f64)> = vec![];
            let mut step_lat = 0.0;
            let chosen: (usize, f64);
            loop {
                let cands: Vec<usize> = all.iter().copied().filter(|a| !tried.contains(a)).collect();
                let a = match router.as_mut() {
                    Some(r) => r.select(&stask, &cands, &mut rng),
                    None => cands.iter().copied().fold(cands[0], |b, a| if inst.comp[a][st.ty] > inst.comp[b][st.ty] { a } else { b }),
                };
                tried.push(a);
                let q = step_q(w, inst, seed, ti, si, a, qin, neff, false);
                cost += 1.0;
                calls += 1;
                step_lat += 1.0;
                log.push((a, q));
                let do_ver = oc.verify
                    && (!oc.stopping || {
                        let (nv, nf) = vstats.get(&(a, st.ty)).copied().unwrap_or((0.0, 0.0));
                        let p = (nf + 0.5) / (nv + 2.5);
                        let ucb = p + (p * (1.0 - p) / (nv + 3.0)).sqrt();
                        ucb * oc.r_val >= w.cv || uni(seed, 0x57, (ti * 8 + si) as u64, tried.len() as u64) < oc.eps_ex
                    });
                if !do_ver {
                    chosen = (a, q);
                    break;
                }
                vcalls += 1;
                cost += w.cv;
                step_lat += w.cv;
                let v = (q + w.sig_v * norm(seed, 0x7E, (ti * 8 + si) as u64, a as u64)).clamp(0.0, 1.0);
                let flagged = v < THETA_V;
                deferred.push(Imm { task: stask.clone(), agent: a, v, key: (a, st.ty), flagged, note: (tk.project, st.ty) });
                if !flagged {
                    chosen = (a, q);
                    break;
                }
                if v > best.0 {
                    best = (v, q, a);
                }
                if tried.len() > oc.max_retry || tried.len() >= na {
                    chosen = (best.2, best.1);
                    break;
                }
            }
            qs[si] = chosen.1;
            let ready = st.deps.iter().map(|&d| fin[d]).fold(0.0, f64::max);
            fin[si] = ready + step_lat;
            lat_seq += step_lat;
            for &(a, q) in &log {
                let g = (q + GRADE_SD * norm(seed, 0x6D, (ti * 8 + si) as u64, a as u64)).clamp(0.0, 1.0);
                fb.push((stask.clone(), a, g));
            }
            let gq = (chosen.1 + GRADE_SD * norm(seed, 0x6D, (ti * 8 + si) as u64, chosen.0 as u64)).clamp(0.0, 1.0);
            note_fb.push(((tk.project, st.ty), gq));
            if !oc.parallel {
                flush(&mut deferred, &mut router, &mut vstats, &mut notes, oc);
            }
        }
        if oc.parallel {
            flush(&mut deferred, &mut router, &mut vstats, &mut notes, oc);
        }
        let tq = mean_f(&qs);
        out.q.push(tq as f32);
        out.ok.push(u8::from(qs.iter().all(|&q| q >= THETA)));
        out.cost.push(cost as f32);
        out.calls.push(calls as u8);
        out.vcalls.push(vcalls as u8);
        let lat = if oc.parallel { fin.iter().copied().fold(0.0, f64::max) } else { lat_seq };
        out.lat.push(lat as f32);
        if uni(seed, 0x2E, ti as u64, 0) < REVEAL {
            pend.push_back(Ev { due: tick + DELAY, upd: fb, notes: note_fb });
        }
    }
    out
}

// ---------------------------------------------------------------- oracles (reference rows; not policies)
pub fn run_oracle_decomp(w: &WorldCfg, inst: &Instance, perfect_ctx: bool, seed: u64) -> RunOut {
    let oc = OrchCfg { router: StepRouter::Oracle(perfect_ctx), retr: false, verify: false, ver_learn: false, stopping: false, parallel: false, max_retry: 0, r_val: 0.0, eps_ex: 0.0, store_min: STORE_MIN };
    run_orch(w, inst, &oc, seed)
}

// ---------------------------------------------------------------- metrics
pub const NM: usize = 16;
pub const METRICS: [&str; NM] = [
    "q_h250", "q_h500", "q_h1500", "ok_h250", "ok_h1500", "cost", "qc_h1500", "util_a", "util_b", "lat", "calls", "vcalls", "q_late300", "q_first100", "ok_h500", "util_c",
];
pub fn mi(name: &str) -> usize {
    METRICS.iter().position(|m| *m == name).unwrap_or_else(|| panic!("metric {name}"))
}
pub fn mean_f32(xs: &[f32]) -> f64 {
    xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len().max(1) as f64
}
fn head(xs: &[f32], n: usize) -> f64 {
    mean_f32(&xs[..n.min(xs.len())])
}
pub fn summarize_run(r: &RunOut, inst: &Instance) -> [f64; NM] {
    let okf: Vec<f32> = r.ok.iter().map(|&x| x as f32).collect();
    let cost = mean_f32(&r.cost);
    let q = mean_f32(&r.q);
    let util = |mu: f64| r.q.iter().zip(&r.cost).zip(&inst.tasks).map(|((&q, &c), t)| q as f64 - mu * (c as f64 / t.steps.len() as f64 - 1.0)).sum::<f64>() / r.q.len() as f64;
    let calls: Vec<f32> = r.calls.iter().map(|&x| x as f32).collect();
    let vcalls: Vec<f32> = r.vcalls.iter().map(|&x| x as f32).collect();
    [
        head(&r.q, 250),
        head(&r.q, 500),
        q,
        head(&okf, 250),
        mean_f32(&okf),
        cost,
        q / cost,
        util(MU_A),
        util(MU_B),
        mean_f32(&r.lat),
        mean_f32(&calls),
        mean_f32(&vcalls),
        mean_f32(&r.q[r.q.len() - 300..]),
        head(&r.q, 100),
        head(&okf, 500),
        util(MU_C),
    ]
}
/// tasks until the trailing-100 mean of q reaches `target` (1501 = never)
pub fn tasks_to(q: &[f32], target: f64) -> f64 {
    let mut s: f64 = q[..100].iter().map(|&x| x as f64).sum();
    if s / 100.0 >= target {
        return 100.0;
    }
    for i in 100..q.len() {
        s += q[i] as f64 - q[i - 100] as f64;
        if s / 100.0 >= target {
            return (i + 1) as f64;
        }
    }
    (q.len() + 1) as f64
}
pub fn tail_mean(q: &[f32], n: usize) -> f64 {
    mean_f32(&q[q.len() - n..])
}

pub fn describe(w: &WorldCfg, widx: u64, seed: u64) -> String {
    use std::fmt::Write;
    let inst = build(w, seed, widx);
    let mut s = String::new();
    let _ = writeln!(
        s,
        "== world {} (family {:?}, pool {:?}, shape {:?}, steps {}-{}, prop {}, noise {}, d {:?} x{}, projects {}, p_id {}, sig_v {}, cv {}, decoy {}, mono_pen {})",
        w.name, w.family, w.pool, w.shape, w.nmin, w.nmax, w.prop, w.noise, w.d, w.dscale, w.projects, w.p_id, w.sig_v, w.cv, w.decoy, w.mono_pen
    );
    let _ = writeln!(s, "agents (true competence per type {:?}) ; keyword map {:?}", TYPES, TYPES.iter().map(|t| inst.kw[*t]).collect::<Vec<_>>());
    for (a, c) in inst.comp.iter().enumerate() {
        let _ = writeln!(s, "  agent {a}: {:.2} {:.2} {:.2} {:.2}", c[0], c[1], c[2], c[3]);
    }
    for tk in inst.tasks.iter().take(4) {
        let st: Vec<String> = tk.steps.iter().enumerate().map(|(i, x)| format!("{i}:{}<-{:?}", TYPES[x.ty], x.deps)).collect();
        let _ = writeln!(s, "  task proj {} [{}]  single-shot view: type={:?} feats={:?}", tk.project, st.join(" "), tk.whole.task_type, tk.whole.features);
    }
    let nst: f64 = inst.tasks.iter().map(|t| t.steps.len() as f64).sum();
    let ed: f64 = inst.tasks.iter().map(|t| t.steps.iter().map(|x| x.deps.len() as f64).sum::<f64>()).sum::<f64>() / nst;
    let _ = writeln!(s, "  mean steps/task {:.2}, mean deps/step {ed:.2}", nst / inst.tasks.len() as f64);
    s
}
