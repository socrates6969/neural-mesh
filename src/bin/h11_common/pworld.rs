//! h11 world P: a COPY of the g18 multi-step orchestration world (generator A "pipelines", generator B "latent factors", 5 controls; worlds/instances/step
//! quality model are bit-identical to g18_common) extended with (a) AGENT PRICES, (b) a token-level resource model, (c) cost levers:
//!   verification gate (calibrated posterior), early-stop vs best-of-n, retrieval score gate / oracle type-skip, note compression, cost-aware routing,
//!   price-ordered cascade, and hindsight oracles (quality, cheapest-that-works, utility).
//! Everything is a SEEDED SIMULATION; token counts and prices are SYNTHETIC parameters (see h11_common/mod.rs). Not a real bill.
//!
//! Token model per generator step call: prompt = B_STEP + DEP_TOK*|deps| (+ injected note tokens), completion = O_STEP; all x agent price pi_a.
//! A verifier call costs the world's `cv` x one reference step call (cv=.3 default): v_out = V_OUT completion tokens and v_in = cv*S_REF - 4*V_OUT prompt
//! tokens (so at r_out = 4 the verifier costs exactly cv*S_REF). Retrieval costs ms (encode 2.6 + scan 0.02) and injects C_NOTE*gamma note tokens per attempt.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};

use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};

use super::{Prices, Res, cost_te};

pub const N_TASKS: usize = 1500;
pub const REVEAL: f64 = 0.6;
pub const DELAY: u64 = 15;
pub const NTYPE: usize = 4;
pub const TYPES: [&str; NTYPE] = ["lookup", "compute", "synthesize", "check"];
pub const THETA: f64 = 0.6;
pub const THETA_V: f64 = 0.55;
pub const GRADE_SD: f64 = 0.05;
pub const STEP_DIFF_SD: f64 = 0.07;
pub const MAX_RETRY: usize = 2;
pub const STORE_MIN: f64 = 0.6;

pub const B_STEP: f64 = 500.0;
pub const DEP_TOK: f64 = 60.0;
pub const O_STEP: f64 = 350.0;
pub const C_NOTE: f64 = 120.0;
pub const V_OUT: f64 = 40.0;
/// reference step call = 500 + 4*350 TE at pi = 1 (no context)
pub const S_REF: f64 = 1900.0;
pub const EMB_MS: f64 = 2.6;
pub const SEARCH_MS: f64 = 0.02;

// ---------------------------------------------------------------- stateless randomness (identical to g18_common)
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

// ---------------------------------------------------------------- world config (copy of g18 + price mode)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gen {
    Pipe,
    Latent,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pool {
    Spec,
    Latent,
    Dominant,
    HighSkill,
    Equal,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Chain,
    Dag,
    Wide,
    RandDag(f64),
    Indep,
    Single,
}
/// how agent prices relate to competence (declared before any policy ran)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PriceMode {
    /// price = clamp(0.25 + 2.5*(mean competence - 0.4), 0.25, 3): bigger/better models cost more; a type specialist is cheaper than its own-type quality suggests
    Cap,
    /// price independent of competence: exp(0.5 N(0,1)) clipped [.3, 3] (marketplace noise: cheap-and-good agents exist)
    Rand,
    /// every agent costs 1
    Flat,
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
    pub prop: f64,
    pub noise: f64,
    pub d: [f64; NTYPE],
    pub dscale: f64,
    pub projects: usize,
    pub p_id: f64,
    pub mis: f64,
    pub sig_v: f64,
    pub cv: f64,
    pub decoy: f64,
    pub mono_pen: f64,
    pub spec_gap: f64,
    pub pm: PriceMode,
}

const D_PIPE: [f64; NTYPE] = [0.25, 0.05, 0.18, 0.08];
const D_LAT: [f64; NTYPE] = [0.20, 0.08, 0.20, 0.10];
const MIX_PIPE: [f64; NTYPE] = [0.30, 0.25, 0.25, 0.20];
const MIX_LAT: [f64; NTYPE] = [0.25, 0.25, 0.30, 0.20];

pub fn worlds() -> Vec<WorldCfg> {
    let pipe = WorldCfg {
        name: "pipe-main", family: Gen::Pipe, pool: Pool::Spec, shape: Shape::Dag, nmin: 3, nmax: 6, type_mix: MIX_PIPE, prop: 0.5, noise: 0.10, d: D_PIPE, dscale: 1.0,
        projects: 10, p_id: 0.9, mis: 0.5, sig_v: 0.10, cv: 0.3, decoy: 0.3, mono_pen: 0.0, spec_gap: 0.26, pm: PriceMode::Cap,
    };
    let lat = WorldCfg {
        name: "lat-main", family: Gen::Latent, pool: Pool::Latent, shape: Shape::RandDag(0.35), nmin: 3, nmax: 6, type_mix: MIX_LAT, prop: 0.6, noise: 0.10, d: D_LAT, dscale: 1.0,
        projects: 12, p_id: 0.85, mis: 0.5, sig_v: 0.10, cv: 0.3, decoy: 0.3, mono_pen: 0.0, spec_gap: 0.0, pm: PriceMode::Cap,
    };
    vec![
        pipe.clone(),
        WorldCfg { name: "pipe-chain", shape: Shape::Chain, ..pipe.clone() },
        WorldCfg { name: "pipe-wide", shape: Shape::Wide, ..pipe.clone() },
        WorldCfg { name: "pipe-weakctx", dscale: 0.4, ..pipe.clone() },
        WorldCfg { name: "pipe-lowrec", projects: 300, ..pipe.clone() },
        WorldCfg { name: "pipe-badver", sig_v: 0.22, ..pipe.clone() },
        lat.clone(),
        WorldCfg { name: "lat-highprop", prop: 0.9, ..lat.clone() },
        WorldCfg { name: "lat-noisyag", noise: 0.16, ..lat.clone() },
        WorldCfg { name: "lat-pricever", cv: 0.8, ..lat.clone() },
        WorldCfg { name: "ctl-single-step", shape: Shape::Single, nmin: 1, nmax: 1, ..pipe.clone() },
        WorldCfg { name: "ctl-indep-dominant", shape: Shape::Indep, pool: Pool::Dominant, ..pipe.clone() },
        WorldCfg { name: "ctl-indep-spec", shape: Shape::Indep, ..pipe.clone() },
        WorldCfg { name: "ctl-rare-pricever", pool: Pool::HighSkill, noise: 0.06, cv: 1.0, d: [0.05, 0.02, 0.04, 0.02], ..pipe.clone() },
        WorldCfg { name: "ctl-equal-agents", pool: Pool::Equal, ..pipe.clone() },
    ]
}
/// price-structure sensitivity worlds (NOT in the main panel): same as pipe-main / lat-main with independent random prices, and with flat prices
pub fn worlds_price_sens() -> Vec<WorldCfg> {
    let w = worlds();
    let (p, l) = (w[0].clone(), w[6].clone());
    vec![
        WorldCfg { name: "pipe-main-pricerand", pm: PriceMode::Rand, ..p.clone() },
        WorldCfg { name: "lat-main-pricerand", pm: PriceMode::Rand, ..l.clone() },
        WorldCfg { name: "pipe-main-priceflat", pm: PriceMode::Flat, ..p },
        WorldCfg { name: "lat-main-priceflat", pm: PriceMode::Flat, ..l },
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
    pub diff: f64,
}
#[derive(Clone, Debug)]
pub struct TaskI {
    pub project: usize,
    pub steps: Vec<Step>,
    pub whole: Task,
}
pub struct Instance {
    pub n_agents: usize,
    pub comp: Vec<[f64; NTYPE]>,
    pub price: Vec<f64>,
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
            types[n - 1] = 2;
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
    // prices: computed AFTER the instance (separate rng) so the instance stream stays identical to g18
    let mut pr = Rng::new(seed.wrapping_mul(0x51ED_270B).wrapping_add(widx * 131 + 9));
    let price: Vec<f64> = comp
        .iter()
        .map(|c| match w.pm {
            PriceMode::Cap => {
                let mc = c.iter().sum::<f64>() / NTYPE as f64;
                (0.25 + 2.5 * (mc - 0.4)).clamp(0.25, 3.0)
            }
            PriceMode::Rand => (0.5 * rnorm(&mut pr)).exp().clamp(0.3, 3.0),
            PriceMode::Flat => 1.0,
        })
        .collect();
    Instance { n_agents: na, comp, price, kw, tasks }
}

// ---------------------------------------------------------------- execution model (identical to g18)
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

/// retrieval result: kind 0 = nothing, 1 = the right project note, 2 = a wrong note. `score` = retrieval confidence (synthetic: right ~ N(.75,.12), wrong ~ N(.55,.12))
#[derive(Clone, Copy, Debug)]
pub struct Ret {
    pub neff: f64,
    pub score: f64,
    pub kind: u8,
}
#[derive(Default)]
pub struct Notes {
    s: HashMap<(usize, usize), f64>,
    per_type: [usize; NTYPE],
}
impl Notes {
    pub fn retrieve(&self, w: &WorldCfg, seed: u64, ti: usize, si: usize, proj: usize, ty: usize) -> Ret {
        let has = self.s.get(&(proj, ty)).copied();
        let other = self.per_type[ty] > usize::from(has.is_some());
        let u = uni(seed, 0x4E, (ti * 8 + si) as u64, 0);
        let (neff, kind) = match has {
            Some(s) => {
                if u < w.p_id {
                    (s, 1u8)
                } else if other && u < w.p_id + (1.0 - w.p_id) * 0.5 {
                    (-w.mis, 2)
                } else {
                    (0.0, 0)
                }
            }
            None => {
                if other && u < (1.0 - w.p_id) * 0.5 {
                    (-w.mis, 2)
                } else {
                    (0.0, 0)
                }
            }
        };
        let z = norm(seed, 0x53, (ti * 8 + si) as u64, kind as u64);
        let score = match kind {
            1 => 0.75 + 0.12 * z,
            2 => 0.55 + 0.12 * z,
            _ => 0.0,
        };
        Ret { neff, score, kind }
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

// ---------------------------------------------------------------- configuration of the cost levers
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VMode {
    Never,
    Always,
    /// g18 L4 learned stopping
    Online { r_val: f64, eps: f64 },
    /// verify iff the router posterior (mean graded quality m of (agent, step type)) says (1-m) * rv >= cv (break-even in the world's verifier price cv,
    /// i.e. a quality-dependent threshold tau = 1 - cv/rv) or fewer than nmin graded samples. (v1 of this study used a constant tau: see results/h11_pipe_v1_constTau.*)
    Gate { rv: f64, nmin: f64 },
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RMode {
    Off,
    Always,
    /// inject the retrieved note only if its retrieval score >= theta (the encode+scan cost is paid either way)
    Gate { theta: f64 },
    /// ORACLE BOUND: skip retrieval entirely (no cost, no note) for step types whose true context dependence d*dscale < dmin
    OracleSkip { dmin: f64 },
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rt {
    /// g18's per-step router: Warm(Hebbian-robust) with the type keyword map, k
    Warm(f64),
    /// cost-adjusted score router: q_hat(agent,type) + explore bonus - mu * price (own graded table; keyword prior as Warm); mu = 0 -> same router, price-blind (control)
    CA { mu: f64 },
    /// try agents in ascending price order (FrugalGPT-style); needs verification to escalate
    Cascade,
    /// hindsight: best realised quality / cheapest agent with q >= THETA / argmax q - mu*price   (reference rows, not policies)
    OracleQ,
    OracleCheap,
    OracleUtil { mu: f64 },
}
#[derive(Clone, Copy, Debug)]
pub struct PCfg {
    pub rt: Rt,
    pub rmode: RMode,
    pub vmode: VMode,
    /// false = no early stop: run `n_bo` attempts, verify each, keep the highest signal
    pub early_stop: bool,
    pub n_bo: usize,
    pub max_retry: usize,
    /// note compression: tokens x gamma, note strength x gamma^kappa (kappa small = diminishing returns; kappa 1 = linear, no free lunch)
    pub gamma: f64,
    pub kappa: f64,
    pub store_min: f64,
}
impl PCfg {
    /// g18's levels L2 (retrieval, no verify), L3 (+ always verify/retry), L4 (+ learned stopping k=1,R,eps)
    pub fn l2() -> PCfg {
        PCfg { rt: Rt::Warm(1.0), rmode: RMode::Always, vmode: VMode::Never, early_stop: true, n_bo: 3, max_retry: MAX_RETRY, gamma: 1.0, kappa: 0.3, store_min: STORE_MIN }
    }
    pub fn l3() -> PCfg {
        PCfg { vmode: VMode::Always, ..PCfg::l2() }
    }
    pub fn l4(r_val: f64, eps: f64) -> PCfg {
        PCfg { vmode: VMode::Online { r_val, eps }, ..PCfg::l2() }
    }
}

// ---------------------------------------------------------------- summaries
#[derive(Clone, Debug, Default)]
pub struct PSum {
    pub q: f64,
    pub ok: f64,
    pub q250: f64,
    pub qlate: f64,
    /// mean per-TASK resources
    pub res: Res,
    /// mean per-task (resources / #steps): the normalisation used by the utility (cost per step relative to one reference call)
    pub resn: Res,
}
/// utility = quality - mu * (cost per step / S_REF - 1)   (g18's util(mu), costs now in TE)
pub fn util(s: &PSum, p: &Prices, mu: f64) -> f64 {
    s.q - mu * (cost_te(&s.resn, p) / S_REF - 1.0)
}

struct Ev {
    due: u64,
    upd: Vec<(Task, usize, f64)>,
    notes: Vec<((usize, usize), f64)>,
    tab: Vec<(usize, usize, f64)>,
}
struct Tab {
    s: Vec<[f64; NTYPE]>,
    n: Vec<[f64; NTYPE]>,
}
fn drain(pend: &mut VecDeque<Ev>, tick: u64, mut router: Option<&mut Box<dyn Router>>, notes: &mut Notes, tab: &mut Tab, store_min: f64, retr: bool) {
    while pend.front().is_some_and(|e| e.due <= tick) {
        let ev = pend.pop_front().unwrap();
        if let Some(r) = router.as_deref_mut() {
            for (t, a, g) in &ev.upd {
                r.update(t, *a, *g);
            }
        }
        for &(a, ty, g) in &ev.tab {
            tab.s[a][ty] += g;
            tab.n[a][ty] += 1.0;
        }
        if retr {
            for (k, g) in &ev.notes {
                notes.store(*k, *g, store_min);
            }
        }
    }
}
struct Imm {
    task: Task,
    agent: usize,
    v: f64,
    key: (usize, usize),
    flagged: bool,
    note: (usize, usize),
}
fn flush(deferred: &mut Vec<Imm>, router: &mut Option<Box<dyn Router>>, vstats: &mut HashMap<(usize, usize), (f64, f64)>, notes: &mut Notes, ver_learn: bool, retr: bool, store_min: f64) {
    for im in deferred.drain(..) {
        let e = vstats.entry(im.key).or_insert((0.0, 0.0));
        e.0 += 1.0;
        if im.flagged {
            e.1 += 1.0;
        }
        if ver_learn {
            if let Some(r) = router.as_mut() {
                r.update(&im.task, im.agent, im.v);
            }
            if retr && !im.flagged {
                notes.store(im.note, im.v, store_min);
            }
        }
    }
}
fn mean_f(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

const CA_W0: f64 = 2.0;
const CA_BONUS: f64 = 0.12;

/// the orchestrator with every cost lever. With rt=Warm, rmode=Always, vmode in {Never, Always, Online}, gamma=1 it reproduces g18 L2/L3/L4 quality exactly (unit-tested).
pub fn run_p(w: &WorldCfg, inst: &Instance, c: &PCfg, seed: u64) -> PSum {
    let na = inst.n_agents;
    let all: Vec<usize> = (0..na).collect();
    let mut rng = Rng::new(seed * 7 + 1);
    let mut router: Option<Box<dyn Router>> = match c.rt {
        Rt::Warm(k) => Some(Box::new(Warm::new(HebbianRouter::new(na, heb_robust_cfg()), inst.kw.clone(), na, k).with_evidence(Evidence::MeanFeature))),
        _ => None,
    };
    let retr = !matches!(c.rmode, RMode::Off);
    let ver_learn = !matches!(c.vmode, VMode::Never);
    let mut notes = Notes::default();
    let mut tab = Tab { s: vec![[0.0; NTYPE]; na], n: vec![[0.0; NTYPE]; na] };
    let mut pend: VecDeque<Ev> = VecDeque::new();
    let mut vstats: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    let mut seen_key: HashSet<(usize, usize)> = HashSet::new();
    let (mut qv, mut okv) = (Vec::with_capacity(N_TASKS), 0.0f64);
    let (mut res_sum, mut resn_sum) = (Res::default(), Res::default());
    let n_att_bo = if c.early_stop { c.max_retry + 1 } else { c.n_bo };
    for (ti, tk) in inst.tasks.iter().enumerate() {
        let tick = ti as u64 + 1;
        drain(&mut pend, tick, router.as_mut(), &mut notes, &mut tab, c.store_min, retr);
        let n = tk.steps.len();
        let mut qs = vec![0.0; n];
        let mut res = Res::default();
        let mut fb: Vec<(Task, usize, f64)> = vec![];
        let mut note_fb: Vec<((usize, usize), f64)> = vec![];
        let mut tab_fb: Vec<(usize, usize, f64)> = vec![];
        let mut deferred: Vec<Imm> = vec![];
        for si in 0..n {
            let st = &tk.steps[si];
            let qin = q_in(w, tk, si, &qs);
            let stask = Task { id: (ti * 8 + si) as u64, task_type: Some(TYPES[st.ty].to_string()), features: vec![TYPES[st.ty].to_string()] };
            // ---- retrieval
            let (mut neff, mut ctx_tok) = (0.0, 0.0);
            let do_ret = match c.rmode {
                RMode::Off => false,
                RMode::OracleSkip { dmin } => w.d[st.ty] * w.dscale >= dmin,
                _ => true,
            };
            if do_ret {
                res.n_ret += 1.0;
                res.ms += EMB_MS + SEARCH_MS;
                res.ms_cached += SEARCH_MS + if seen_key.insert((tk.project, st.ty)) { EMB_MS } else { 0.0 };
                let r = notes.retrieve(w, seed, ti, si, tk.project, st.ty);
                let pass = match c.rmode {
                    RMode::Gate { theta } => r.kind != 0 && r.score >= theta,
                    _ => r.kind != 0,
                };
                if pass {
                    neff = r.neff * c.gamma.powf(c.kappa);
                    ctx_tok = C_NOTE * c.gamma;
                    res.n_inj += 1.0;
                }
            }
            // ---- attempts
            let mut tried: Vec<usize> = vec![];
            let mut best: (f64, f64, usize) = (-1.0, 0.0, 0);
            let mut log: Vec<(usize, f64)> = vec![];
            let chosen: (usize, f64);
            loop {
                let cands: Vec<usize> = all.iter().copied().filter(|a| !tried.contains(a)).collect();
                let a = match c.rt {
                    Rt::Warm(_) => router.as_mut().unwrap().select(&stask, &cands, &mut rng),
                    Rt::CA { mu } => {
                        let mut bs = (f64::MIN, cands[0]);
                        for &a in &cands {
                            let (s, nn) = (tab.s[a][st.ty], tab.n[a][st.ty]);
                            let m0 = if inst.kw[TYPES[st.ty]] == a { 0.75 } else { 0.55 };
                            let m = (s + CA_W0 * m0) / (nn + CA_W0);
                            let sc = m + CA_BONUS / (nn + 1.0).sqrt() - mu * inst.price[a];
                            if sc > bs.0 {
                                bs = (sc, a);
                            }
                        }
                        bs.1
                    }
                    Rt::Cascade => cands.iter().copied().fold(cands[0], |b, a| if inst.price[a] < inst.price[b] { a } else { b }),
                    Rt::OracleQ | Rt::OracleCheap | Rt::OracleUtil { .. } => {
                        let qa: Vec<(usize, f64)> = cands.iter().map(|&a| (a, step_q(w, inst, seed, ti, si, a, qin, neff, false))).collect();
                        match c.rt {
                            Rt::OracleQ => qa.iter().fold(qa[0], |b, x| if x.1 > b.1 { *x } else { b }).0,
                            Rt::OracleCheap => {
                                let ok: Vec<&(usize, f64)> = qa.iter().filter(|x| x.1 >= THETA).collect();
                                if ok.is_empty() {
                                    qa.iter().fold(qa[0], |b, x| if x.1 > b.1 { *x } else { b }).0
                                } else {
                                    ok.iter().fold(*ok[0], |b, x| if inst.price[x.0] < inst.price[b.0] { **x } else { b }).0
                                }
                            }
                            Rt::OracleUtil { mu } => qa.iter().fold(qa[0], |b, x| if x.1 - mu * inst.price[x.0] > b.1 - mu * inst.price[b.0] { *x } else { b }).0,
                            _ => unreachable!(),
                        }
                    }
                };
                tried.push(a);
                let q = step_q(w, inst, seed, ti, si, a, qin, neff, false);
                let pi = inst.price[a];
                res.g_base += pi * (B_STEP + DEP_TOK * st.deps.len() as f64);
                res.g_ctx += pi * ctx_tok;
                res.g_out += pi * O_STEP;
                res.n_gen += 1.0;
                log.push((a, q));
                let do_ver = match c.vmode {
                    VMode::Never => false,
                    VMode::Always => true,
                    VMode::Online { r_val, eps } => {
                        let (nv, nf) = vstats.get(&(a, st.ty)).copied().unwrap_or((0.0, 0.0));
                        let p = (nf + 0.5) / (nv + 2.5);
                        let ucb = p + (p * (1.0 - p) / (nv + 3.0)).sqrt();
                        ucb * r_val >= w.cv || uni(seed, 0x57, (ti * 8 + si) as u64, tried.len() as u64) < eps
                    }
                    VMode::Gate { rv, nmin } => {
                        let (s, nn) = (tab.s[a][st.ty], tab.n[a][st.ty]);
                        nn < nmin || (1.0 - s / nn) * rv >= w.cv
                    }
                };
                if !do_ver {
                    chosen = (a, q);
                    break;
                }
                res.v_in += w.cv * S_REF - 4.0 * V_OUT;
                res.v_out += V_OUT;
                res.n_ver += 1.0;
                let v = (q + w.sig_v * norm(seed, 0x7E, (ti * 8 + si) as u64, a as u64)).clamp(0.0, 1.0);
                let flagged = v < THETA_V;
                deferred.push(Imm { task: stask.clone(), agent: a, v, key: (a, st.ty), flagged, note: (tk.project, st.ty) });
                if c.early_stop {
                    if !flagged {
                        chosen = (a, q);
                        break;
                    }
                    if v > best.0 {
                        best = (v, q, a);
                    }
                    if tried.len() > c.max_retry || tried.len() >= na {
                        chosen = (best.2, best.1);
                        break;
                    }
                } else {
                    if v > best.0 {
                        best = (v, q, a);
                    }
                    if tried.len() >= n_att_bo || tried.len() >= na {
                        chosen = (best.2, best.1);
                        break;
                    }
                }
            }
            qs[si] = chosen.1;
            for &(a, q) in &log {
                let g = (q + GRADE_SD * norm(seed, 0x6D, (ti * 8 + si) as u64, a as u64)).clamp(0.0, 1.0);
                fb.push((stask.clone(), a, g));
                tab_fb.push((a, st.ty, g));
            }
            let gq = (chosen.1 + GRADE_SD * norm(seed, 0x6D, (ti * 8 + si) as u64, chosen.0 as u64)).clamp(0.0, 1.0);
            note_fb.push(((tk.project, st.ty), gq));
            flush(&mut deferred, &mut router, &mut vstats, &mut notes, ver_learn, retr, c.store_min);
        }
        let tq = mean_f(&qs);
        qv.push(tq);
        okv += f64::from(u8::from(qs.iter().all(|&q| q >= THETA)));
        res_sum.add(&res);
        resn_sum.add(&res.scaled(1.0 / n as f64));
        if uni(seed, 0x2E, ti as u64, 0) < REVEAL {
            pend.push_back(Ev { due: tick + DELAY, upd: fb, notes: note_fb, tab: tab_fb });
        }
    }
    let nt = inst.tasks.len() as f64;
    PSum { q: mean_f(&qv), ok: okv / nt, q250: mean_f(&qv[..250]), qlate: mean_f(&qv[qv.len() - 300..]), res: res_sum.scaled(1.0 / nt), resn: resn_sum.scaled(1.0 / nt) }
}
