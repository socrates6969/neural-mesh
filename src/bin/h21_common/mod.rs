//! H21 common (R9-E): a simulated multi-step RESEARCH-SESSION world: search (query reformulation / facet drill-down), database lookups
//! (query can be wrong, row transcription can be wrong), document reading (imperfect extractor), compute (mental arithmetic vs a REAL exact
//! calculator = `neural_mesh::tools::Calc`), verification (source-span check, cross-source agreement, independent re-query, tool re-computation,
//! consistency) and synthesis. Values are REAL integers, so errors propagate through the dependency graph by arithmetic, not by a fudge factor.
//!
//! EVERYTHING here is a seeded simulation written by me. It measures whether decomposing a 3-8 step research session, routing steps by type,
//! retrieving per step, offloading exact sub-steps to tools, verifying steps (imperfect verifier that can only see LOCAL errors: it cannot see
//! a wrong INPUT), gating verification by a posterior, and re-planning failed searches/queries raises the all-steps-correct rate and a graded
//! quality that I defined, and what it costs under the h11 token-equivalent (TE) cost model. It does NOT measure LLM behaviour, reasoning,
//! or knowledge. The size of every gain is set by world parameters I chose; the sensitivity runs show where each lever vanishes or turns negative.
//!
//! WORLDS ARE FIXED BEFORE POLICIES: `worlds()` was written and `--show`n before any router/policy was run.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::defaults::recommended_router;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::tools::Calc;
use neural_mesh::verifygate::PosteriorGate;

pub const N_TASKS: usize = 1000;
pub const REVEAL: f64 = 0.6;
pub const DELAY: u64 = 15;
pub const GRADE_SD: f64 = 0.05;
pub const STEP_DIFF_SD: f64 = 0.07;
pub const MAX_RETRY: usize = 2;
pub const MAX_REFORM: usize = 2;
pub const MU_A: f64 = 0.1;
pub const MU_C: f64 = 0.03;
pub static MU_BITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// the exchange rate used for the `util_c` metric / tuning objective (default MU_C = .03; `--mu` overrides)
pub fn mu_c() -> f64 {
    let b = MU_BITS.load(std::sync::atomic::Ordering::Relaxed);
    if b == 0 { MU_C } else { f64::from_bits(b) }
}
pub const NT: usize = 5;
pub const SEARCH: usize = 0;
pub const DB: usize = 1;
pub const READ: usize = 2;
pub const COMP: usize = 3;
pub const SYN: usize = 4;
pub const TYPES: [&str; NT] = ["search", "db", "read", "compute", "synth"];
pub const NONE: i64 = -1;

// ---- h11 token-equivalent (TE) cost model: S_REF = 500 prompt + 350 completion x r_out(4) = 1900 TE for a mid-price (pi = 1) generator step
pub const S_REF: f64 = 1900.0;
pub const PROMPT_TE: f64 = 500.0;
pub const PER_DEP_TE: f64 = 60.0;
pub const COMPLETION_TE: f64 = 1400.0;
pub const DOC_TE: f64 = 300.0; // an injected document / schema chunk (pi-priced prompt tokens)
pub const CR: f64 = 0.05; // one retrieval (g18 convention), in reference calls
pub const C_TOOL: f64 = 0.10; // one tool run (verifygate::C_TOOL / g9), in reference calls
pub const TOOL_MULT: f64 = 0.20; // extra prompt/completion to formulate + read back a tool call (fraction of the base step)
/// verifier price per step type in reference calls (search title check, db re-query, span check, tool recompute, consistency)
pub const VER_COST: [f64; NT] = [0.12, 0.35, 0.20, 0.10, 0.25];
pub const CROSS_COST: f64 = 0.55;
const VER_REC: [f64; NT] = [0.85, 0.70, 0.75, 0.97, 0.85];
const VER_FP: [f64; NT] = [0.06, 0.08, 0.06, 0.01, 0.04];
const VER_NODOC: f64 = 0.95; // span check on a READ that has no source
const VER_NONE_SEARCH: f64 = 0.98;
const CROSS_REC: f64 = 0.90;

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

// ---------------------------------------------------------------- ground-truth functions of the hidden corpus / database
/// value of attribute `attr` in document `d`
pub fn fact(d: i64, attr: i64) -> i64 {
    100 + (hash(0xFA, d as u64, attr as u64, 1) % 9900) as i64
}
/// row value for `key` in table/column variant `tab`
pub fn dbval(key: i64, tab: i64) -> i64 {
    100 + (hash(0xDB, key as u64, tab as u64, 2) % 9900) as i64
}
fn slip(seed: u64, key: u64, ag: u64, tag: u64) -> i64 {
    const S: [i64; 8] = [10, -10, 100, -100, 1000, 1, -1, -1000];
    S[(hash(seed, key, ag, tag) % 8) as usize]
}
fn halluc(seed: u64, key: u64, ag: u64, truth: i64) -> i64 {
    let v = 100 + (hash(seed, key, ag, 0xA11) % 9900) as i64;
    if v == truth { v + 1 } else { v }
}
fn wrong_doc(seed: u64, key: u64, ag: u64, truth: i64) -> i64 {
    let v = 5000 + (hash(seed, key, ag, 0xD0C) % 5000) as i64;
    if v == truth { v + 1 } else { v }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Add,
    Sub,
    AddSub,
    Scale,
}
pub fn op_native(op: Op, v: &[i64]) -> i64 {
    match op {
        Op::Add => v.iter().sum(),
        Op::Sub => v[0] - v[1],
        Op::AddSub => v[0] + v[1] - v[2],
        Op::Scale => v[0] * 3 + v[1],
    }
}
/// the expression string handed to the real calculator
pub fn op_spec(op: Op, v: &[i64]) -> String {
    let p = |x: i64| if x < 0 { format!("({x})") } else { x.to_string() };
    match op {
        Op::Add => v.iter().map(|&x| p(x)).collect::<Vec<_>>().join("+"),
        Op::Sub => format!("{}-{}", p(v[0]), p(v[1])),
        Op::AddSub => format!("{}+{}-{}", p(v[0]), p(v[1]), p(v[2])),
        Op::Scale => format!("{}*3+{}", p(v[0]), p(v[1])),
    }
}
/// a tool-call FORMULATION error: the model writes the wrong operator
pub fn op_spec_mut(op: Op, v: &[i64]) -> String {
    let s = op_spec(op, v);
    if let Some(i) = s.find('+') {
        let mut b = s.into_bytes();
        b[i] = b'-';
        String::from_utf8(b).unwrap()
    } else if let Some(i) = s[1..].find('-') {
        let mut b = s.into_bytes();
        b[i + 1] = b'+';
        String::from_utf8(b).unwrap()
    } else {
        s
    }
}
/// evaluate via the REAL exact calculator (falls back to native on tool error, which a unit test asserts never happens)
pub fn calc_eval(spec: &str, fallback: i64) -> i64 {
    match Calc::eval_str(spec) {
        Ok(r) if r.is_int() => r.numer() as i64,
        _ => fallback,
    }
}

// ---------------------------------------------------------------- world config
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gen {
    /// generator A: template research sessions (facts -> compute -> synth), specialist agent pool
    A,
    /// generator B: random interleaved DAG of fact blocks / compute / keyed joins, latent-factor agent pool, different failure mix
    B,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pool {
    /// 8 agents: 2 generalists (.66), 5 type specialists (.90 own / .48 off), 1 weak (.42)
    Spec,
    /// 8 agents from 3 latent factors
    Latent,
    /// agent 0 .88 everywhere, others ~.6
    Dominant,
    /// all agents ~.985
    HighSkill,
    /// 8 identical agents (.72)
    Equal,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Mixed,
    /// each later fact is a DB join keyed on the previous fact's value
    Chain,
    /// many independent facts, one aggregation
    Wide,
    /// independent facts, no aggregation (no error propagation)
    Indep,
    /// one DB step per task
    Single,
    /// generator B random DAG
    Dag,
}

#[derive(Clone, Debug)]
pub struct WorldCfg {
    pub name: &'static str,
    pub family: Gen,
    pub pool: Pool,
    pub shape: Shape,
    /// P(doc fact = search+read) vs a DB lookup
    pub p_doc: f64,
    /// P(a DB lookup is keyed on an earlier value, i.e. a join whose key can be poisoned)
    pub key_p: f64,
    /// agent noise sd (per task x step x agent, fixed across arms)
    pub noise: f64,
    /// P(a fact is known from parametric memory when NO retrieval is given)
    pub kmem: f64,
    /// retrieval quality: P(hit) for an auto-query is hq x (.6 + .4 c)
    pub hq: f64,
    /// P(a search miss is VISIBLE (empty result) rather than a silent wrong document)
    pub p_vis: f64,
    /// P(the schema/context chunk for a DB step is retrieved)
    pub h_schema: f64,
    pub ar_scale: f64,
    pub ex_scale: f64,
    pub syn_scale: f64,
    /// verifier quality: 1 = nominal recall/false-flag, 0 = uninformative (flag rate = false-flag rate whatever the truth)
    pub vq: f64,
    /// verifier price multiplier
    pub vscale: f64,
    /// extra quality loss of ONE agent doing the whole session in one call (0 in all main worlds; sensitivity only)
    pub mono_pen: f64,
    pub decoy: f64,
    /// session budget = beta x steps x S_REF (extras = verification, retries, tools beyond the base call are blocked when the budget cannot cover them
    /// after reserving the remaining steps' base calls)
    pub beta: f64,
    /// P(a second independent source exists for a fact) and P(it conflicts with a correct reading)
    pub q_sec: f64,
    pub q_conf: f64,
    /// P(the corpus / database actually CONTAINS the needed answer). 1.0 in every world except ctl-noinfo. (Added post hoc, see board entry disclosure 2.)
    pub cover: f64,
}

pub fn worlds() -> Vec<WorldCfg> {
    let a = WorldCfg {
        name: "res-main", family: Gen::A, pool: Pool::Spec, shape: Shape::Mixed, p_doc: 0.65, key_p: 0.35, noise: 0.10, kmem: 0.30, hq: 0.70, p_vis: 0.5, h_schema: 0.85,
        ar_scale: 1.0, ex_scale: 1.0, syn_scale: 1.0, vq: 1.0, vscale: 1.0, mono_pen: 0.0, decoy: 0.3, beta: 2.5, q_sec: 0.6, q_conf: 0.12, cover: 1.0,
    };
    let b = WorldCfg { name: "ana-main", family: Gen::B, pool: Pool::Latent, shape: Shape::Dag, p_doc: 0.50, key_p: 0.50, noise: 0.10, kmem: 0.25, hq: 0.65, ex_scale: 1.3, ar_scale: 1.2, ..a.clone() };
    vec![
        // ---------------- treatment worlds, generator A
        a.clone(),
        WorldCfg { name: "res-chain", shape: Shape::Chain, ..a.clone() },
        WorldCfg { name: "res-wide", shape: Shape::Wide, ..a.clone() },
        WorldCfg { name: "res-lowret", hq: 0.45, h_schema: 0.60, ..a.clone() },
        WorldCfg { name: "res-badver", vq: 0.45, ..a.clone() },
        WorldCfg { name: "res-hardex", ex_scale: 2.0, ar_scale: 1.5, ..a.clone() },
        // ---------------- treatment worlds, generator B
        b.clone(),
        WorldCfg { name: "ana-dbheavy", p_doc: 0.20, key_p: 0.70, ..b.clone() },
        WorldCfg { name: "ana-noisyag", noise: 0.16, ..b.clone() },
        WorldCfg { name: "ana-pricever", vscale: 2.5, ..b.clone() },
        WorldCfg { name: "ana-lowbudget", beta: 1.4, ..b.clone() },
        // ---------------- controls
        WorldCfg { name: "ctl-single-step", shape: Shape::Single, ..a.clone() },
        WorldCfg { name: "ctl-indep", shape: Shape::Indep, ..a.clone() },
        WorldCfg { name: "ctl-noinfo", cover: 0.03, ..a.clone() },
        WorldCfg { name: "ctl-rare-pricever", pool: Pool::HighSkill, noise: 0.03, hq: 0.99, h_schema: 0.99, ex_scale: 0.3, ar_scale: 0.3, syn_scale: 0.3, vscale: 3.0, ..a.clone() },
        WorldCfg { name: "ctl-equal-agents", pool: Pool::Equal, ..a.clone() },
        WorldCfg { name: "ctl-dominant", pool: Pool::Dominant, ..a.clone() },
    ]
}
/// POST-HOC extra controls (added after the held-out look showed ctl-rare-pricever is not rare enough for skipping to pay at mu = .03); run separately (`--extra`)
pub fn worlds_extra() -> Vec<WorldCfg> {
    let a = worlds().into_iter().find(|w| w.name == "res-main").unwrap();
    vec![
        WorldCfg { name: "xtr-rare-vexp", pool: Pool::HighSkill, noise: 0.03, hq: 0.99, h_schema: 0.99, ex_scale: 0.3, ar_scale: 0.3, syn_scale: 0.3, vscale: 10.0, ..a.clone() },
        WorldCfg { name: "xtr-rarer", pool: Pool::HighSkill, noise: 0.01, hq: 0.9995, h_schema: 0.9995, ex_scale: 0.1, ar_scale: 0.1, syn_scale: 0.1, vscale: 3.0, ..a.clone() },
    ]
}
pub fn is_control(name: &str) -> bool {
    name.starts_with("xtr-") || name.starts_with("ctl-")
}

// ---------------------------------------------------------------- instance
#[derive(Clone, Debug)]
pub struct Step {
    pub ty: usize,
    pub deps: Vec<usize>,
    /// task x step difficulty offset (shared by all agents and arms)
    pub diff: f64,
    pub op: Op,
    pub attr: i64,
    pub tab: i64,
    pub entity: i64,
}
#[derive(Clone, Debug)]
pub struct TaskI {
    pub steps: Vec<Step>,
    /// true value per step (doc id for a search step)
    pub truth: Vec<i64>,
    pub whole: Task,
}
pub struct Instance {
    pub n_agents: usize,
    pub comp: Vec<[f64; NT]>,
    pub price: Vec<f64>,
    pub kw: HashMap<String, usize>,
    pub tasks: Vec<TaskI>,
}

fn build_comp(w: &WorldCfg, r: &mut Rng) -> Vec<[f64; NT]> {
    let jit = |r: &mut Rng, base: f64, sp: f64| (base + sp * (r.f64() * 2.0 - 1.0)).clamp(0.05, 0.99);
    let mut v: Vec<[f64; NT]> = Vec::new();
    match w.pool {
        Pool::Spec => {
            for _ in 0..2 {
                let mut c = [0.0; NT];
                for x in c.iter_mut() {
                    *x = jit(r, 0.66, 0.03);
                }
                v.push(c);
            }
            for t in 0..NT {
                let mut c = [0.0; NT];
                for (u, x) in c.iter_mut().enumerate() {
                    *x = if u == t { jit(r, 0.90, 0.03) } else { jit(r, 0.48, 0.05) };
                }
                v.push(c);
            }
            let mut c = [0.0; NT];
            for x in c.iter_mut() {
                *x = jit(r, 0.42, 0.04);
            }
            v.push(c);
        }
        Pool::Latent => {
            const LOAD: [[f64; 3]; NT] = [[0.8, 0.1, 0.3], [0.2, 0.9, 0.1], [0.9, 0.2, 0.2], [0.1, 0.8, 0.4], [0.3, 0.3, 0.9]];
            for _ in 0..8 {
                let z = [rnorm(r), rnorm(r), rnorm(r)];
                let mut c = [0.0; NT];
                for t in 0..NT {
                    let s: f64 = (0..3).map(|k| LOAD[t][k] * z[k]).sum();
                    c[t] = (0.66 + 0.11 * s + 0.03 * rnorm(r)).clamp(0.05, 0.97);
                }
                v.push(c);
            }
        }
        Pool::Dominant => {
            let a = jit(r, 0.88, 0.02);
            v.push([a; NT]);
            for _ in 0..7 {
                let mut c = [0.0; NT];
                for x in c.iter_mut() {
                    *x = jit(r, 0.60, 0.05);
                }
                v.push(c);
            }
        }
        Pool::HighSkill => {
            for _ in 0..8 {
                let mut c = [0.0; NT];
                for x in c.iter_mut() {
                    *x = jit(r, 0.985, 0.005);
                }
                v.push(c);
            }
        }
        Pool::Equal => {
            for _ in 0..8 {
                v.push([0.72; NT]);
            }
        }
    }
    for i in (1..v.len()).rev() {
        let j = r.below(i + 1);
        v.swap(i, j);
    }
    v
}

fn new_step(ty: usize, deps: Vec<usize>, r: &mut Rng) -> Step {
    let op = match deps.len() {
        3 => {
            if r.f64() < 0.5 { Op::Add } else { Op::AddSub }
        }
        2 => match r.below(3) {
            0 => Op::Add,
            1 => Op::Sub,
            _ => Op::Scale,
        },
        _ => Op::Add,
    };
    Step { ty, deps, diff: STEP_DIFF_SD * rnorm(r), op, attr: r.below(40) as i64, tab: r.below(60) as i64 * 8, entity: r.below(900_000) as i64 }
}

fn push_step(st: &mut Vec<Step>, ty: usize, deps: Vec<usize>, r: &mut Rng) -> usize {
    st.push(new_step(ty, deps, r));
    st.len() - 1
}

fn pick_deps(values: &[usize], k: usize, r: &mut Rng) -> Vec<usize> {
    let mut pool = values.to_vec();
    let mut deps = vec![];
    for _ in 0..k.min(pool.len()) {
        let i = r.below(pool.len());
        deps.push(pool.remove(i));
    }
    deps.sort();
    deps
}

fn leaves_of(st: &[Step]) -> Vec<usize> {
    let mut consumed = vec![false; st.len()];
    for s in st {
        for &d in &s.deps {
            consumed[d] = true;
        }
    }
    let l: Vec<usize> = (0..st.len()).filter(|&i| !consumed[i] && st[i].ty != SEARCH).collect();
    if l.is_empty() { vec![st.len() - 1] } else { l }
}

/// the step list (types / deps) of one session
fn build_steps(w: &WorldCfg, r: &mut Rng) -> Vec<Step> {
    let mut st: Vec<Step> = vec![];
    let mut values: Vec<usize> = vec![]; // steps producing a number (read / db / comp)
    match w.shape {
        Shape::Single => {
            push_step(&mut st, DB, vec![], r);
            return st;
        }
        Shape::Dag => {
            let target = 3 + r.below(6); // 3..8 steps
            while st.len() + 1 < target {
                let room = target - 1 - st.len();
                let u = r.f64();
                if room >= 2 && u < w.p_doc {
                    let s = push_step(&mut st, SEARCH, vec![], r);
                    let rd = push_step(&mut st, READ, vec![s], r);
                    values.push(rd);
                } else if values.len() >= 2 && r.f64() < 0.30 {
                    let k = if values.len() >= 3 && r.f64() < 0.4 { 3 } else { 2 };
                    let deps = pick_deps(&values, k, r);
                    let c = push_step(&mut st, COMP, deps, r);
                    values.push(c);
                } else {
                    let deps = if !values.is_empty() && r.f64() < w.key_p { vec![values[r.below(values.len())]] } else { vec![] };
                    let d = push_step(&mut st, DB, deps, r);
                    values.push(d);
                }
            }
            let l = leaves_of(&st);
            push_step(&mut st, SYN, l, r);
            return st;
        }
        _ => {}
    }
    // generator A templates
    let nf = match w.shape {
        Shape::Wide => 3 + r.below(2),
        Shape::Indep => 2 + r.below(3),
        _ => {
            let u = r.f64();
            if u < 0.15 {
                1
            } else if u < 0.65 {
                2
            } else {
                3
            }
        }
    };
    let cap = if w.shape == Shape::Indep { 8 } else { 6 };
    for f in 0..nf {
        let doc = if nf == 1 {
            true
        } else if w.shape == Shape::Chain && f > 0 {
            false
        } else {
            r.f64() < w.p_doc
        };
        let need = if doc { 2 } else { 1 };
        if st.len() + need > cap {
            break;
        }
        if doc {
            let s = push_step(&mut st, SEARCH, vec![], r);
            let rd = push_step(&mut st, READ, vec![s], r);
            values.push(rd);
        } else {
            let keyed = !values.is_empty() && w.shape != Shape::Wide && w.shape != Shape::Indep && (w.shape == Shape::Chain || r.f64() < w.key_p);
            let deps = if keyed {
                vec![if w.shape == Shape::Chain { *values.last().unwrap() } else { values[r.below(values.len())] }]
            } else {
                vec![]
            };
            let d = push_step(&mut st, DB, deps, r);
            values.push(d);
        }
    }
    if w.shape == Shape::Indep {
        return st;
    }
    if values.len() >= 2 {
        let ncomp = if values.len() >= 3 && r.f64() < 0.3 && st.len() < 7 { 2 } else { 1 };
        for _ in 0..ncomp {
            let k = if values.len() >= 3 && r.f64() < 0.4 { 3 } else { 2 };
            let deps = pick_deps(&values, k, r);
            if st.len() < 7 {
                let c = push_step(&mut st, COMP, deps, r);
                values.push(c);
            }
        }
    }
    let l = leaves_of(&st);
    push_step(&mut st, SYN, l, r);
    st
}

fn truths(st: &[Step], r: &mut Rng) -> Vec<i64> {
    let mut t = vec![0i64; st.len()];
    for i in 0..st.len() {
        let s = &st[i];
        t[i] = match s.ty {
            SEARCH => r.below(5000) as i64,
            READ => fact(t[s.deps[0]], s.attr),
            DB => {
                let key = if s.deps.is_empty() { s.entity } else { t[s.deps[0]] };
                dbval(key, s.tab)
            }
            COMP => {
                let v: Vec<i64> = s.deps.iter().map(|&d| t[d]).collect();
                op_native(s.op, &v)
            }
            _ => s.deps.iter().map(|&d| t[d]).sum(),
        };
    }
    t
}

pub fn build(w: &WorldCfg, seed: u64, widx: u64) -> Instance {
    let mut r = Rng::new(seed.wrapping_mul(1_000_003) ^ (widx * 7919 + 17));
    let comp = build_comp(w, &mut r);
    let na = comp.len();
    let price: Vec<f64> = comp.iter().map(|c| (0.25 + 2.5 * (c.iter().sum::<f64>() / NT as f64 - 0.4)).clamp(0.35, 2.0)).collect();
    let mut kw = HashMap::new();
    for (t, name) in TYPES.iter().enumerate() {
        let best = (0..na).fold(0, |b, a| if comp[a][t] > comp[b][t] { a } else { b });
        let a = if r.f64() < w.decoy { r.below(na) } else { best };
        kw.insert(name.to_string(), a);
    }
    let mut tasks = Vec::with_capacity(N_TASKS);
    for ti in 0..N_TASKS {
        let steps = build_steps(w, &mut r);
        let truth = truths(&steps, &mut r);
        let mut sig: Vec<usize> = steps.iter().map(|s| s.ty).collect();
        sig.sort();
        let tt = sig.iter().map(|&t| TYPES[t]).collect::<Vec<_>>().join("+");
        let mut feats: Vec<String> = sig.iter().map(|&t| TYPES[t].to_string()).collect();
        feats.dedup();
        let whole = Task { id: ti as u64, task_type: Some(tt), features: feats };
        tasks.push(TaskI { steps, truth, whole });
    }
    Instance { n_agents: na, comp, price, kw, tasks }
}

// ---------------------------------------------------------------- one step execution (deterministic given world/seed/task/step/agent/flags)
#[derive(Clone, Copy, Debug, Default)]
pub struct Fl {
    pub retr: bool,
    pub tools: bool,
    pub mono: bool,
    /// query reformulation level (0 = none): halves the miss probability of a search / db query per level
    pub reform: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct Exec {
    pub val: i64,
    /// the step performed ITS OWN operation faithfully on the inputs it was given (a verifier can only see this, never a wrong input)
    pub local_ok: bool,
    pub visible_fail: bool,
    /// TE
    pub cost: f64,
    pub tool: bool,
}

pub fn base_te(inst: &Instance, agent: usize, nd: usize) -> f64 {
    inst.price[agent] * (PROMPT_TE + PER_DEP_TE * nd as f64 + COMPLETION_TE)
}

pub fn exec_step(w: &WorldCfg, inst: &Instance, seed: u64, ti: usize, si: usize, agent: usize, obs: &[i64], fl: &Fl) -> Exec {
    let tk = &inst.tasks[ti];
    let st = &tk.steps[si];
    let key = (ti * 16 + si) as u64;
    let ag = agent as u64;
    let u = |tag: u64| uni(seed, key, ag, tag);
    let mut c = inst.comp[agent][st.ty] + st.diff + w.noise * norm(seed, key, ag, 0x51);
    if fl.mono {
        c -= w.mono_pen;
    }
    let c = c.clamp(0.02, 0.99);
    let pi = inst.price[agent];
    let nd = st.deps.len();
    let base = base_te(inst, agent, nd);
    let mut cost = base;
    let pm = w.kmem * (0.4 + 0.6 * c);
    let truth = tk.truth[si];
    let rf = 0.5f64.powi(fl.reform as i32);
    let mut tool = false;
    let (val, local_ok, visible) = match st.ty {
        SEARCH => {
            if !fl.retr {
                let hit = u(1) < pm;
                (if hit { truth } else { wrong_doc(seed, key, ag, truth) }, hit, false)
            } else {
                cost += CR * S_REF + pi * DOC_TE;
                let h_auto = w.hq * (0.6 + 0.4 * c);
                let h = 1.0 - (1.0 - h_auto) * rf;
                let covered = u(5) < w.cover;
                if covered && u(1) < h {
                    (truth, true, false)
                } else if u(2) < w.p_vis {
                    (NONE, false, true)
                } else {
                    (wrong_doc(seed, key, ag, truth), false, false)
                }
            }
        }
        READ => {
            let d_obs = obs[st.deps[0]];
            if !fl.retr {
                let hit = u(1) < pm;
                (if hit { truth } else { halluc(seed, key, ag, truth) }, hit, false)
            } else {
                cost += pi * DOC_TE;
                if d_obs == NONE {
                    let hit = u(1) < pm;
                    (if hit { truth } else { halluc(seed, key, ag, truth) }, false, false)
                } else {
                    let p_ex = (1.0 - 0.9 * w.ex_scale * (1.0 - c)).clamp(0.0, 1.0);
                    if u(1) < p_ex {
                        (fact(d_obs, st.attr), true, false)
                    } else if u(2) < 0.6 {
                        (fact(d_obs, st.attr + 7919), false, false)
                    } else {
                        (fact(d_obs, st.attr) + slip(seed, key, ag, 0x51F), false, false)
                    }
                }
            }
        }
        DB => {
            let key_obs = if st.deps.is_empty() { st.entity } else { obs[st.deps[0]] };
            if !fl.retr {
                let hit = u(1) < pm;
                (if hit { truth } else { halluc(seed, key, ag, truth) }, hit, false)
            } else {
                cost += CR * S_REF + pi * DOC_TE;
                let schema_ok = u(3) < w.h_schema;
                let p_q = c * if schema_ok { 1.0 } else { 0.6 };
                let p_q = 1.0 - (1.0 - p_q) * rf;
                let query_ok = u(5) < w.cover && u(1) < p_q;
                let tab = if query_ok { st.tab } else { st.tab + 1 + (hash(seed, key, ag, 0x7AB) % 5) as i64 };
                let raw = dbval(key_obs, tab);
                let e_tr = (0.30 * (1.0 - c) * w.ar_scale).clamp(0.0, 0.9);
                let tr_ok = fl.tools || u(2) >= e_tr;
                if fl.tools {
                    tool = true;
                }
                (if tr_ok { raw } else { raw + slip(seed, key, ag, 0x7A1) }, query_ok && tr_ok, false)
            }
        }
        COMP => {
            let vals: Vec<i64> = st.deps.iter().map(|&d| obs[d]).collect();
            let exact = op_native(st.op, &vals);
            if fl.tools {
                tool = true;
                let f_err = ((0.03 * (1.0 - c) + 0.01) * w.ar_scale).clamp(0.0, 0.9);
                let form_ok = u(1) >= f_err;
                let spec = if form_ok { op_spec(st.op, &vals) } else { op_spec_mut(st.op, &vals) };
                (calc_eval(&spec, exact), form_ok, false)
            } else {
                let e_ar = ((0.55 * (1.0 - c) + 0.04) * (1.0 + 0.25 * (nd as f64 - 2.0)) * w.ar_scale).clamp(0.0, 0.9);
                let ok = u(1) >= e_ar;
                (if ok { exact } else { exact + slip(seed, key, ag, 0xA51) }, ok, false)
            }
        }
        _ => {
            let vals: Vec<i64> = st.deps.iter().map(|&d| obs[d]).collect();
            let sum: i64 = vals.iter().sum();
            let e = (0.6 * (1.0 - c) * (1.0 + 0.15 * (nd as f64 - 2.0)) * w.syn_scale).clamp(0.0, 0.9);
            let ok = u(1) >= e;
            let v = if ok {
                sum
            } else if nd >= 2 {
                sum - vals[(hash(seed, key, ag, 0x5E) % nd as u64) as usize]
            } else {
                sum + slip(seed, key, ag, 0x5F)
            };
            (v, ok, false)
        }
    };
    if tool {
        cost += base * TOOL_MULT + C_TOOL * S_REF;
    }
    Exec { val, local_ok, visible_fail: visible, cost, tool }
}

/// quality of one attempt: 1 exact; else the better of (0.3 if it was faithful on wrong inputs, 0.5 x near-miss credit)
pub fn attempt_q(tk: &TaskI, si: usize, val: i64, local_ok: bool) -> f64 {
    let t = tk.truth[si];
    if val == t {
        return 1.0;
    }
    if tk.steps[si].ty == SEARCH {
        return 0.0;
    }
    let near = if val == NONE { 0.0 } else { 0.5 * (1.0 - 5.0 * (val - t).abs() as f64 / t.abs().max(1) as f64).max(0.0) };
    near.max(if local_ok { 0.3 } else { 0.0 })
}

/// verification of one executed step; returns (flag, TE cost)
pub fn verify(w: &WorldCfg, inst: &Instance, seed: u64, ti: usize, si: usize, agent: usize, ex: &Exec, obs: &[i64], cross: bool) -> (bool, f64) {
    let tk = &inst.tasks[ti];
    let st = &tk.steps[si];
    let key = (ti * 16 + si) as u64;
    let t = st.ty;
    let u = uni(seed, key, agent as u64, 0x7000 + t as u64 + if cross { 16 } else { 0 });
    let blend = |p: f64, fp: f64| fp + w.vq * (p - fp);
    if t == READ && cross {
        let second = uni(seed, key, 0, 0x5EC) < w.q_sec;
        let flag = if !second {
            false
        } else if ex.val != tk.truth[si] {
            u < blend(CROSS_REC, w.q_conf)
        } else {
            u < w.q_conf
        };
        return (flag, CROSS_COST * w.vscale * S_REF);
    }
    let cost = VER_COST[t] * w.vscale * S_REF;
    let flag = match t {
        SEARCH => {
            if ex.val == NONE {
                u < blend(VER_NONE_SEARCH, VER_FP[t])
            } else if !ex.local_ok {
                u < blend(VER_REC[t], VER_FP[t])
            } else {
                u < VER_FP[t]
            }
        }
        READ => {
            if obs[st.deps[0]] == NONE {
                u < blend(VER_NODOC, VER_FP[t])
            } else if !ex.local_ok {
                u < blend(VER_REC[t], VER_FP[t])
            } else {
                u < VER_FP[t]
            }
        }
        _ => {
            if !ex.local_ok {
                u < blend(VER_REC[t], VER_FP[t])
            } else {
                u < VER_FP[t]
            }
        }
    };
    (flag, cost)
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
pub enum Route {
    Warm(f64),
    Random,
    /// best agent per step type by TRUE competence (reference only)
    Oracle,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ver {
    None,
    Always,
    /// g18 L4: ucb of the flag rate of (agent, type) x r >= cv, explore with prob eps
    Online { r: f64, eps: f64 },
    /// verifygate::PosteriorGate: (1 - m) x rv >= cv, m = own delayed-graded mean quality of (agent, type)
    Post { rv: f64 },
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Vote {
    Off,
    /// every step: 3 distinct agents, plurality value
    Always,
    /// vote instead of verify when the posterior quality of the picked (agent, type) is below theta (n >= 3)
    Low(f64),
}
#[derive(Clone, Copy, Debug)]
pub struct Cfg {
    pub route: Route,
    pub retr: bool,
    pub tools: bool,
    pub ver: Ver,
    /// reformulate failed searches / db queries (visible failures and flagged ones) and, for a cross-source flag on a READ, re-search its document
    pub replan: bool,
    /// READ verification = cross-source agreement (global: sees a wrong document) instead of the local span check
    pub cross: bool,
    pub vote: Vote,
}
impl Cfg {
    pub fn l1() -> Cfg {
        Cfg { route: Route::Warm(1.0), retr: false, tools: false, ver: Ver::None, replan: false, cross: false, vote: Vote::Off }
    }
    pub fn level(name: &str, k: f64, rv: f64, r_on: f64, eps_on: f64, theta: f64) -> Cfg {
        let l1 = Cfg { route: Route::Warm(k), ..Cfg::l1() };
        let l2 = Cfg { retr: true, ..l1 };
        let l2t = Cfg { tools: true, ..l2 };
        let l3 = Cfg { ver: Ver::Always, ..l2t };
        let l4 = Cfg { ver: Ver::Post { rv }, ..l3 };
        let l5 = Cfg { replan: true, ..l4 };
        match name {
            "L1" => l1,
            "L2" => l2,
            "L2t" => l2t,
            "L3x" => l3,
            "L4" => l4,
            "L4o" => Cfg { ver: Ver::Online { r: r_on, eps: eps_on }, ..l3 },
            "L5" => l5,
            "BoK" => Cfg { vote: Vote::Always, ..l2t },
            "Stack" => Cfg { cross: true, vote: if theta > 0.0 { Vote::Low(theta) } else { Vote::Off }, ..l5 },
            _ => panic!("level {name}"),
        }
    }
}

// ---------------------------------------------------------------- run output
pub struct RunOut {
    pub q: Vec<f32>,
    pub ok: Vec<u8>,
    pub fin: Vec<u8>,
    /// session cost in reference calls (TE / S_REF)
    pub cost: Vec<f32>,
    pub calls: Vec<u8>,
    pub vcalls: Vec<u8>,
    pub blocked: Vec<u8>,
    pub nsteps: Vec<u8>,
}
impl RunOut {
    fn new() -> Self {
        RunOut { q: vec![], ok: vec![], fin: vec![], cost: vec![], calls: vec![], vcalls: vec![], blocked: vec![], nsteps: vec![] }
    }
    fn push(&mut self, tk: &TaskI, obs: &[i64], lok: &[bool], cost_te: f64, calls: u32, vcalls: u32, blocked: bool) {
        let n = tk.steps.len();
        let qs: Vec<f64> = (0..n).map(|i| attempt_q(tk, i, obs[i], lok[i])).collect();
        self.q.push((qs.iter().sum::<f64>() / n as f64) as f32);
        self.ok.push(u8::from((0..n).all(|i| obs[i] == tk.truth[i])));
        self.fin.push(u8::from(obs[n - 1] == tk.truth[n - 1]));
        self.cost.push((cost_te / S_REF) as f32);
        self.calls.push(calls.min(255) as u8);
        self.vcalls.push(vcalls.min(255) as u8);
        self.blocked.push(u8::from(blocked));
        self.nsteps.push(n as u8);
    }
}

struct Ev {
    due: u64,
    upd: Vec<(Task, usize, usize, f64)>,
}
type Post = HashMap<(usize, usize), (f64, f64)>;
fn drain(pend: &mut VecDeque<Ev>, tick: u64, router: &mut Option<Box<dyn Router>>, post: &mut Post) {
    while pend.front().is_some_and(|e| e.due <= tick) {
        let ev = pend.pop_front().unwrap();
        for (t, a, ty, g) in &ev.upd {
            if let Some(r) = router.as_mut() {
                r.update(t, *a, *g);
            }
            let e = post.entry((*a, *ty)).or_insert((0.0, 0.0));
            e.0 += *g;
            e.1 += 1.0;
        }
    }
}

fn stask(ti: usize, si: usize, ty: usize) -> Task {
    Task { id: (ti * 16 + si) as u64, task_type: Some(TYPES[ty].to_string()), features: vec![TYPES[ty].to_string()] }
}

fn pick(router: &mut Option<Box<dyn Router>>, route: Route, st: &Task, cands: &[usize], inst: &Instance, ty: usize, rng: &mut Rng) -> usize {
    match (route, router.as_mut()) {
        (Route::Warm(_), Some(r)) => r.select(st, cands, rng),
        (Route::Random, _) => cands[rng.below(cands.len())],
        _ => cands.iter().copied().fold(cands[0], |b, a| if inst.comp[a][ty] > inst.comp[b][ty] { a } else { b }),
    }
}

/// index of the plurality value (first among ties; NONE ignored)
fn plurality(vals: &[i64]) -> usize {
    let mut best = 0usize;
    let mut bc = 0usize;
    for i in 0..vals.len() {
        if vals[i] == NONE {
            continue;
        }
        let c = vals.iter().filter(|&&v| v == vals[i]).count();
        if c > bc {
            bc = c;
            best = i;
        }
    }
    best
}

fn cv_of(w: &WorldCfg, ty: usize, cross: bool) -> f64 {
    (if ty == READ && cross { CROSS_COST } else { VER_COST[ty] }) * w.vscale
}

/// ORCHESTRATOR: decompose -> route per step -> (retrieve) -> (tools) -> (verify, gated) -> (retry / replan) -> synthesize.
pub fn run_orch(w: &WorldCfg, inst: &Instance, cfg: &Cfg, seed: u64) -> RunOut {
    let na = inst.n_agents;
    let all: Vec<usize> = (0..na).collect();
    let mut rng = Rng::new(seed * 7 + 1);
    let mut router: Option<Box<dyn Router>> = match cfg.route {
        Route::Warm(k) => Some(Box::new(Warm::new(HebbianRouter::new(na, heb_robust_cfg()), inst.kw.clone(), na, k).with_evidence(Evidence::MeanFeature))),
        _ => None,
    };
    let mut post: Post = HashMap::new();
    let mut pend: VecDeque<Ev> = VecDeque::new();
    let mut vstats: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    let mut out = RunOut::new();
    let fl0 = Fl { retr: cfg.retr, tools: cfg.tools, mono: false, reform: 0 };
    let base_est = S_REF * (1.0 + if cfg.retr { 0.25 } else { 0.0 } + if cfg.tools { 0.2 } else { 0.0 });
    for (ti, tk) in inst.tasks.iter().enumerate() {
        let tick = ti as u64 + 1;
        drain(&mut pend, tick, &mut router, &mut post);
        let n = tk.steps.len();
        let budget = w.beta * n as f64 * S_REF;
        let mut spent = 0.0f64;
        let (mut calls, mut vcalls) = (0u32, 0u32);
        let mut blocked = false;
        let mut obs = vec![NONE; n];
        let mut lok = vec![false; n];
        let mut fb: Vec<(Task, usize, usize, f64)> = vec![];
        for si in 0..n {
            let st = &tk.steps[si];
            let ty = st.ty;
            let stk = stask(ti, si, ty);
            let reserve = (n - si - 1) as f64 * base_est;
            let mut tried: Vec<usize> = vec![];
            let mut attempts: Vec<(usize, Exec)> = vec![];
            let mut reform = 0usize;
            let mut redone = false;
            let mut n_retry = 0usize;
            let mut last_flagged;
            let mut cur_agent = pick(&mut router, cfg.route, &stk, &all, inst, ty, &mut rng);
            // ---- vote path
            let m_post = post.get(&(cur_agent, ty)).copied().unwrap_or((0.0, 0.0));
            let do_vote = match cfg.vote {
                Vote::Off => false,
                Vote::Always => true,
                Vote::Low(th) => m_post.1 >= 3.0 && m_post.0 / m_post.1 < th,
            };
            if do_vote && spent + 3.0 * base_est + reserve <= budget {
                let mut vals = vec![];
                let mut execs = vec![];
                let mut ags: Vec<usize> = vec![];
                for k in 0..3 {
                    let a = if k == 0 {
                        cur_agent
                    } else {
                        let cands: Vec<usize> = all.iter().copied().filter(|x| !ags.contains(x)).collect();
                        pick(&mut router, cfg.route, &stk, &cands, inst, ty, &mut rng)
                    };
                    ags.push(a);
                    let ex = exec_step(w, inst, seed, ti, si, a, &obs, &fl0);
                    spent += ex.cost;
                    calls += 1;
                    vals.push(ex.val);
                    execs.push(ex);
                    let g = (attempt_q(tk, si, ex.val, ex.local_ok) + GRADE_SD * norm(seed, 0x6D, (ti * 16 + si) as u64, a as u64)).clamp(0.0, 1.0);
                    fb.push((stk.clone(), a, ty, g));
                }
                let b = plurality(&vals);
                obs[si] = execs[b].val;
                lok[si] = execs[b].local_ok;
                continue;
            }
            // ---- sequential path
            loop {
                tried.push(cur_agent);
                let ex = exec_step(w, inst, seed, ti, si, cur_agent, &obs, &Fl { reform, ..fl0 });
                spent += ex.cost;
                calls += 1;
                attempts.retain(|(a, _)| *a != cur_agent);
                attempts.push((cur_agent, ex));
                last_flagged = false;
                // visible search failure: reformulate (same agent), no verifier needed
                if ex.visible_fail && cfg.replan && reform < MAX_REFORM {
                    if spent + base_est + reserve <= budget {
                        reform += 1;
                        tried.pop();
                        continue;
                    }
                    blocked = true;
                }
                let cvt = cv_of(w, ty, cfg.cross);
                let want_ver = match cfg.ver {
                    Ver::None => false,
                    Ver::Always => true,
                    Ver::Post { rv } => {
                        let (s, nn) = post.get(&(cur_agent, ty)).copied().unwrap_or((0.0, 0.0));
                        PosteriorGate { rv, cv: cvt, nmin: 3.0 }.should_verify_sums(s, nn)
                    }
                    Ver::Online { r, eps } => {
                        let (nv, nf) = vstats.get(&(cur_agent, ty)).copied().unwrap_or((0.0, 0.0));
                        let p = (nf + 0.5) / (nv + 2.5);
                        let ucb = p + (p * (1.0 - p) / (nv + 3.0)).sqrt();
                        ucb * r >= cvt || uni(seed, 0x57, (ti * 16 + si) as u64, tried.len() as u64) < eps
                    }
                };
                let vcost = cvt * S_REF;
                let can_ver = spent + vcost + reserve <= budget;
                if want_ver && !can_ver {
                    blocked = true;
                }
                if !(want_ver && can_ver) {
                    break;
                }
                let (flag, vc) = verify(w, inst, seed, ti, si, cur_agent, &ex, &obs, cfg.cross && ty == READ);
                spent += vc;
                vcalls += 1;
                let e = vstats.entry((cur_agent, ty)).or_insert((0.0, 0.0));
                e.0 += 1.0;
                if flag {
                    e.1 += 1.0;
                }
                if !flag {
                    break;
                }
                last_flagged = true;
                // flagged: retry?
                if n_retry >= MAX_RETRY || tried.len() >= na {
                    break;
                }
                if spent + base_est + reserve > budget {
                    blocked = true;
                    break;
                }
                n_retry += 1;
                if cfg.replan && ty == READ && cfg.cross && !redone {
                    // global flag on a READ: the document may be the problem -> re-search its document with a reformulated query, then re-read
                    let s = st.deps[0];
                    let sa = pick(&mut router, cfg.route, &stask(ti, s, SEARCH), &all, inst, SEARCH, &mut rng);
                    let sx = exec_step(w, inst, seed, ti, s, sa, &obs, &Fl { reform: 1, ..fl0 });
                    spent += sx.cost;
                    calls += 1;
                    obs[s] = sx.val;
                    lok[s] = sx.local_ok;
                    redone = true;
                    tried.pop();
                    continue; // same reader agent, new document
                }
                let cands: Vec<usize> = all.iter().copied().filter(|a| !tried.contains(a)).collect();
                cur_agent = pick(&mut router, cfg.route, &stk, &cands, inst, ty, &mut rng);
                if cfg.replan && (ty == SEARCH || ty == DB) {
                    reform = (reform + 1).min(MAX_REFORM);
                }
            }
            // delivered attempt: the last one if it passed (or was not checked); if the last was flagged and >= 2 attempts agree, the plurality; else the first
            let pick_i = if !last_flagged || attempts.len() == 1 {
                attempts.len() - 1
            } else {
                let vals: Vec<i64> = attempts.iter().map(|(_, e)| e.val).collect();
                let p = plurality(&vals);
                if vals.iter().filter(|&&v| v == vals[p]).count() >= 2 { p } else { 0 }
            };
            obs[si] = attempts[pick_i].1.val;
            lok[si] = attempts[pick_i].1.local_ok;
            for (a, e) in &attempts {
                let g = (attempt_q(tk, si, e.val, e.local_ok) + GRADE_SD * norm(seed, 0x6D, (ti * 16 + si) as u64, *a as u64)).clamp(0.0, 1.0);
                fb.push((stk.clone(), *a, ty, g));
            }
        }
        out.push(tk, &obs, &lok, spent, calls, vcalls, blocked);
        if uni(seed, 0x2E, ti as u64, 0) < REVEAL {
            pend.push_back(Ev { due: tick + DELAY, upd: fb });
        }
    }
    out
}

/// SINGLE-SHOT: one router call per task; the chosen agent does every step in one pass (charged step-units like g18). `react` gives that agent
/// retrieval + tools inside its loop (one query / one lookup per step, no verification, no reformulation, no per-step routing). `k_sessions` > 1 =
/// session-level best-of-k: k distinct agents each do the whole session, plurality of the final answers.
pub fn run_single(w: &WorldCfg, inst: &Instance, kind: &SsKind, react: bool, k_sessions: usize, seed: u64) -> RunOut {
    let na = inst.n_agents;
    let cands: Vec<usize> = (0..na).collect();
    let mut router = mk_ss(kind, inst);
    let mut rng = Rng::new(seed * 7 + 1);
    let mut pend: VecDeque<Ev> = VecDeque::new();
    let mut out = RunOut::new();
    let fl = Fl { retr: react, tools: react, mono: true, reform: 0 };
    for (ti, tk) in inst.tasks.iter().enumerate() {
        let tick = ti as u64 + 1;
        while pend.front().is_some_and(|e: &Ev| e.due <= tick) {
            let ev = pend.pop_front().unwrap();
            if let Some(r) = router.as_mut() {
                for (t, a, _, g) in &ev.upd {
                    r.update(t, *a, *g);
                }
            }
        }
        let n = tk.steps.len();
        let mut agents: Vec<usize> = vec![];
        for _ in 0..k_sessions {
            let c: Vec<usize> = cands.iter().copied().filter(|a| !agents.contains(a)).collect();
            let a = match router.as_mut() {
                Some(r) => r.select(&tk.whole, &c, &mut rng),
                None => {
                    let score = |a: usize| tk.steps.iter().map(|s| inst.comp[a][s.ty]).sum::<f64>();
                    c.iter().copied().fold(c[0], |b, a| if score(a) > score(b) { a } else { b })
                }
            };
            agents.push(a);
        }
        let mut sessions: Vec<(Vec<i64>, Vec<bool>)> = vec![];
        let mut cost = 0.0;
        for &a in &agents {
            let mut obs = vec![NONE; n];
            let mut lok = vec![false; n];
            for si in 0..n {
                let ex = exec_step(w, inst, seed, ti, si, a, &obs, &fl);
                cost += ex.cost;
                obs[si] = ex.val;
                lok[si] = ex.local_ok;
            }
            sessions.push((obs, lok));
        }
        let finals: Vec<i64> = sessions.iter().map(|(o, _)| o[n - 1]).collect();
        let ch = plurality(&finals);
        out.push(tk, &sessions[ch].0, &sessions[ch].1, cost, agents.len() as u32, 0, false);
        if router.is_some() && uni(seed, 0x2E, ti as u64, 0) < REVEAL {
            let tq = *out.q.last().unwrap() as f64;
            let g = (tq + GRADE_SD * norm(seed, 0x6D, ti as u64, 99)).clamp(0.0, 1.0);
            pend.push_back(Ev { due: tick + DELAY, upd: vec![(tk.whole.clone(), agents[ch], 0, g)] });
        }
    }
    out
}

// ---------------------------------------------------------------- metrics
pub const NM: usize = 16;
pub const METRICS: [&str; NM] = [
    "q_h250", "q_h1000", "ok_h250", "ok_h1000", "fin_h1000", "cost", "qpc", "okpc", "util_c", "util_a", "calls", "vcalls", "blocked", "q_late250", "q_first100", "ok_late250",
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
fn tailm(xs: &[f32], n: usize) -> f64 {
    mean_f32(&xs[xs.len().saturating_sub(n)..])
}
pub fn summarize_run(r: &RunOut) -> [f64; NM] {
    let f = |v: &[u8]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    let (ok, fin, calls, vcalls, blk) = (f(&r.ok), f(&r.fin), f(&r.calls), f(&r.vcalls), f(&r.blocked));
    let cost = mean_f32(&r.cost);
    let util = |mu: f64| (0..r.q.len()).map(|i| r.q[i] as f64 - mu * (r.cost[i] as f64 / r.nsteps[i] as f64 - 1.0)).sum::<f64>() / r.q.len() as f64;
    [
        head(&r.q, 250),
        mean_f32(&r.q),
        head(&ok, 250),
        mean_f32(&ok),
        mean_f32(&fin),
        cost,
        mean_f32(&r.q) / cost,
        mean_f32(&ok) / cost,
        util(mu_c()),
        util(MU_A),
        mean_f32(&calls),
        mean_f32(&vcalls),
        mean_f32(&blk),
        tailm(&r.q, 250),
        head(&r.q, 100),
        tailm(&ok, 250),
    ]
}
/// tasks until the trailing-100 mean of q reaches `target` (N+1 = never)
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
    tailm(q, n)
}

pub fn describe(w: &WorldCfg, widx: u64, seed: u64) -> String {
    use std::fmt::Write;
    let inst = build(w, seed, widx);
    let mut s = String::new();
    let _ = writeln!(
        s,
        "== world {} (family {:?}, pool {:?}, shape {:?}, p_doc {}, key_p {}, noise {}, kmem {}, hq {}, p_vis {}, h_schema {}, ar {} ex {} syn {}, vq {}, vscale {}, mono_pen {}, decoy {}, beta {}, q_sec {}, q_conf {}, cover {})",
        w.name, w.family, w.pool, w.shape, w.p_doc, w.key_p, w.noise, w.kmem, w.hq, w.p_vis, w.h_schema, w.ar_scale, w.ex_scale, w.syn_scale, w.vq, w.vscale, w.mono_pen, w.decoy, w.beta, w.q_sec, w.q_conf, w.cover
    );
    let _ = writeln!(s, "agents (true competence per type {:?}; price) ; keyword map {:?}", TYPES, TYPES.iter().map(|t| inst.kw[*t]).collect::<Vec<_>>());
    for (a, c) in inst.comp.iter().enumerate() {
        let _ = writeln!(s, "  agent {a}: {:.2} {:.2} {:.2} {:.2} {:.2}  price {:.2}", c[0], c[1], c[2], c[3], c[4], inst.price[a]);
    }
    for tk in inst.tasks.iter().take(4) {
        let st: Vec<String> = tk.steps.iter().enumerate().map(|(i, x)| format!("{i}:{}<-{:?}", TYPES[x.ty], x.deps)).collect();
        let _ = writeln!(s, "  session [{}]", st.join(" "));
    }
    let nst: f64 = inst.tasks.iter().map(|t| t.steps.len() as f64).sum();
    let mut cnt = [0usize; NT];
    for t in &inst.tasks {
        for x in &t.steps {
            cnt[x.ty] += 1;
        }
    }
    let ed: f64 = inst.tasks.iter().map(|t| t.steps.iter().map(|x| x.deps.len() as f64).sum::<f64>()).sum::<f64>() / nst;
    let mn = inst.tasks.iter().map(|t| t.steps.len()).min().unwrap();
    let mx = inst.tasks.iter().map(|t| t.steps.len()).max().unwrap();
    let _ = writeln!(s, "  steps/session mean {:.2} (min {mn}, max {mx}), deps/step {ed:.2}, step-type mix {:?}", nst / inst.tasks.len() as f64, cnt.iter().map(|&c| (c as f64 / nst * 100.0).round() / 100.0).collect::<Vec<_>>());
    s
}
