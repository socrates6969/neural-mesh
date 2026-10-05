//! H13 part 2: END-TO-END closed-loop routing with a retrieval layer on the REAL-text replay (h10 fixtures).
//!
//! REAL: task texts (section titles and held-out sentences of real documentation paragraphs, 25% with synthetic QWERTY typos), the document store
//!       (6831 / 4181 real paragraphs), the retrieval results (computed by results/h13_retrieval: dense MiniLM, BM25, z/RRF/RSF hybrids, rerank-20),
//!       the keyword features of the task text, the topic structure (k-means over the real paragraph embeddings).
//! SIMULATED (labelled): which agent is competent on which topic cluster (tables fixed by the seed BEFORE any policy runs), the keyword prior,
//!       the context-injection effect and every Bernoulli outcome.  Context helps ONLY IF the injected top-1 paragraph is STRUCTURALLY relevant
//!       (belongs to the task's source section / is its source paragraph, h10 `rel`): p' = p + gain*(1-p); an irrelevant injected paragraph
//!       distracts: p' = p*(1-0.05).  No simulated relevance labels are used for utility.
//! Policy = g16 knowledge layer: retrieve the top-1 paragraph (dense-cosine admission floor `thr`), anchor = the paragraph's topic cluster, vote the agent
//!       with the best smoothed success in that anchor (min_n, m>=.7), SAFE gate = g16 "G-post+g1" (Post{delta 0,gamma .8,floor 0} AND window gate
//!       (6,.02,25), kn:<anchor> + context injected only when the vote is open); 'g1' gate = g16 R0 (window gate, kn injected always).
//!       Front router of all arms = guarded(hebr) (g16/h2). Baselines: keyword, eps-greedy, pooled-thompson, hebbian-robust, warm(hebr), guarded.
//! PROTOCOL (pre-declared): tune (thr in {.3..70}, min_n in {2,3}) per retrieval system x gate on A seeds 1-8 on the panel {T12, T12-drift, Tctx, ctl-irrel}
//!       with the g16 constrained objective; held-out A 31-60; FROZEN confirm A 61-90 and B 61-90 (independent text). T48 and ctl-oneoff never tuned on.
//!       paired 95% t-CI over seeds; Holm per family; TOST +-0.01 for controls; wins must hold in both text worlds.
//! Usage: h13_route [--out results/h13_route.json] [--quick]

#[path = "h13_common/mod.rs"]
mod common;

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use common::{Row, World, load_world, seeds};
use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::{holm_adjust, inc_beta, paired_diff, paired_t_pvalue, summarize, t_two_sided_p};
use rayon::prelude::*;
use serde_json::{Value, json};

const NA: usize = 6;
const NT: usize = 2000;
const TOPN: usize = 8;
const ORACLE: usize = 12; // retrieval 'system' id of the oracle (a structurally relevant paragraph, cosine 1)
const SYS_NAMES: [&str; 13] = ["dense_raw", "dense_s", "bm25_raw", "bm25_s", "hybz_raw", "hybz_s", "hybrrf_raw", "hybrrf_s", "hybrsf_raw", "hybrsf_s", "rerank_raw", "rerank_s", "oracle"];
const PEN: f64 = 0.05;

// ------------------------------------------------------------------------------------------------ data

struct WData {
    w: World,
    cl: HashMap<usize, Vec<usize>>,                   // K -> cluster per doc
    kwtok: HashMap<usize, Vec<(String, usize)>>,      // K -> sorted (token, modal cluster)
    feat: HashMap<u32, Vec<String>>,                  // raw text id -> keyword features
    tops: HashMap<(u32, u8), Vec<(u32, f32)>>,        // (tid, flavor) -> 6 base kinds x TOPN
}

fn load_wdata(tag: &str) -> WData {
    let w = load_world(tag);
    let j: Value = serde_json::from_slice(&std::fs::read(format!("results/h13_world_{tag}.json")).expect("run py/h13_prep.py")).unwrap();
    let mut cl = HashMap::new();
    let mut kwtok = HashMap::new();
    for k in [12usize, 48] {
        cl.insert(k, j["cl"][k.to_string()].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect());
        let mut v: Vec<(String, usize)> = j["kwtok"][k.to_string()].as_object().unwrap().iter().map(|(a, b)| (a.clone(), b.as_u64().unwrap() as usize)).collect();
        v.sort();
        kwtok.insert(k, v);
    }
    let feat = j["feat"].as_object().unwrap().iter().map(|(k, v)| (k.parse().unwrap(), v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect())).collect();
    let raw = std::fs::read(format!("results/h13_top_{tag}.bin")).expect("run h13_retrieval first");
    let n = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let mut pos = 4;
    let mut tops = HashMap::with_capacity(n);
    for _ in 0..n {
        let tid = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
        let fl = raw[pos + 4];
        pos += 5;
        let mut v = Vec::with_capacity(6 * TOPN);
        for _ in 0..6 * TOPN {
            let d = u32::from_le_bytes(raw[pos..pos + 4].try_into().unwrap());
            let c = f32::from_le_bytes(raw[pos + 4..pos + 8].try_into().unwrap());
            v.push((d, c));
            pos += 8;
        }
        tops.insert((tid, fl), v);
    }
    WData { w, cl, kwtok, feat, tops }
}

// ------------------------------------------------------------------------------------------------ worlds (fixed before any policy ran)

#[derive(Clone, Debug)]
struct WSpec {
    name: &'static str,
    k: usize,
    topic_dep: bool,
    drift: bool,
    gain: f64,
    /// context value is real only if the injected doc is structurally relevant (false: every injected doc is irrelevant -> distraction only)
    ctx_real: bool,
    /// the task's source is absent from the store (relevant docs removed from the candidate lists)
    oneoff: bool,
    treat: bool,
    tune: bool,
    /// context-essential world: low base competence, large context gain (success is mostly decided by the injected paragraph)
    low_base: bool,
}
fn worlds() -> Vec<WSpec> {
    let s = |name, k, topic_dep, drift, gain, ctx_real, oneoff, treat, tune| WSpec { name, k, topic_dep, drift, gain, ctx_real, oneoff, treat, tune, low_base: false };
    vec![
        s("T12", 12, true, false, 0.15, true, false, true, true),
        s("T12-drift", 12, true, true, 0.15, true, false, true, true),
        s("Tctx", 12, false, false, 0.20, true, false, true, true),
        s("T48", 48, true, false, 0.15, true, false, true, false),
        s("T12-noctx", 12, true, false, 0.0, true, false, true, false),
        s("ctl-irrel", 12, false, false, 0.0, false, false, false, true),
        s("ctl-oneoff", 12, true, false, 0.15, true, true, false, false),
        WSpec { low_base: true, ..s("Tqa", 12, false, false, 0.85, true, false, true, false) },
    ]
}

struct Env {
    rows: Vec<usize>,
    var: Vec<u8>,
    u: Vec<f64>,
    cl_true: Vec<usize>,
    feats: Vec<Vec<String>>,
    p_pre: Vec<[f64; NA]>,
    p_post: Vec<[f64; NA]>,
    kw: HashMap<String, usize>,
}

fn uni(r: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * r.f64()
}

fn make_env(wd: &WData, ws: &WSpec, seed: u64) -> Env {
    let inst = &wd.w.inst[&seed.to_string()];
    let nrows = inst.rows.len();
    let mut name_hash = 0u64;
    for b in ws.name.bytes() {
        name_hash = name_hash.wrapping_mul(131).wrapping_add(b as u64);
    }
    let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x13_C0DE ^ (name_hash << 20));
    let mut p_pre = vec![[0.0; NA]; ws.k];
    let mut best_of = vec![0usize; ws.k];
    if ws.topic_dep {
        for c in 0..ws.k {
            for a in 0..NA {
                p_pre[c][a] = uni(&mut r, 0.15, 0.45);
            }
            let best = r.below(NA);
            p_pre[c][best] = uni(&mut r, 0.80, 0.90);
            let mut second = r.below(NA - 1);
            if second >= best {
                second += 1;
            }
            p_pre[c][second] = uni(&mut r, 0.55, 0.68);
            best_of[c] = best;
        }
    } else {
        // topic-independent competence: ONE global table (a global best agent, the topic carries no information about who is competent)
        let mut row = [0.0; NA];
        for a in 0..NA {
            row[a] = if ws.low_base { uni(&mut r, 0.10, 0.25) } else { uni(&mut r, 0.25, 0.50) };
        }
        let best = r.below(NA);
        row[best] = if ws.low_base { uni(&mut r, 0.25, 0.35) } else { uni(&mut r, 0.62, 0.72) };
        for c in 0..ws.k {
            p_pre[c] = row;
            best_of[c] = best;
        }
    }
    let mut p_post = p_pre.clone();
    if ws.drift {
        for c in 0..ws.k {
            if r.bernoulli(0.25) {
                let old = best_of[c];
                let mut new = r.below(NA - 1);
                if new >= old {
                    new += 1;
                }
                p_post[c][old] = uni(&mut r, 0.30, 0.50);
                p_post[c][new] = uni(&mut r, 0.80, 0.90);
            }
        }
    }
    let mut kw = HashMap::new();
    for (t, c) in &wd.kwtok[&ws.k] {
        let a = if !ws.topic_dep {
            r.below(NA)
        } else if r.bernoulli(0.65) {
            best_of[*c]
        } else {
            let mut x = r.below(NA - 1);
            if x >= best_of[*c] {
                x += 1;
            }
            x
        };
        kw.insert(t.clone(), a);
    }
    let cl = &wd.cl[&ws.k];
    let (mut rows, mut var, mut u, mut cl_true, mut feats) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..NT {
        let ri = r.below(nrows);
        let v = if r.f64() < 0.75 { 0u8 } else { 1 + r.below(2) as u8 };
        let row = &inst.rows[ri];
        rows.push(ri);
        var.push(v);
        u.push(r.f64());
        cl_true.push(cl[wd.w.queries[row.q].rel[0]]);
        feats.push(wd.feat.get(&(row.t[v as usize][0] as u32)).cloned().unwrap_or_default());
    }
    Env { rows, var, u, cl_true, feats, p_pre, p_post, kw }
}

// ------------------------------------------------------------------------------------------------ policies

#[derive(Clone, Copy, PartialEq, Debug)]
enum Base {
    GlobalTs,
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm,
    Guard,
}
#[derive(Clone, Copy, PartialEq, Debug)]
enum GateK {
    Safe,
    G1,
    /// context-only injection: the front picks the agent; the retrieved top-1 paragraph is injected iff the g1-style window gate is open (no anchor vote)
    Ctx,
}
#[derive(Clone, Debug)]
struct Arm {
    name: String,
    base: Option<Base>,
    sys: usize,
    gate: GateK,
    thr: f32,
    min_n: usize,
    k: usize,
}
impl Arm {
    fn baseline(name: &str, b: Base) -> Arm {
        Arm { name: name.into(), base: Some(b), sys: 0, gate: GateK::Safe, thr: 0.0, min_n: 2, k: 1 }
    }
    fn vote(sys: usize, gate: GateK, thr: f32, min_n: usize, k: usize) -> Arm {
        Arm { name: arm_name(gate, sys), base: None, sys, gate, thr, min_n, k }
    }
}
fn arm_name(g: GateK, sys: usize) -> String {
    format!("{}|{}", match g { GateK::Safe => "safe", GateK::G1 => "g1", GateK::Ctx => "ctx" }, SYS_NAMES[sys])
}

/// Thompson sampling over agents that ignores all task features (a plain global bandit): POST-HOC addition to the fair baselines (see header)
struct GlobalTs(PooledThompsonRouter);
impl Router for GlobalTs {
    fn name(&self) -> String {
        "global-ts".into()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        self.0.select(&Task { features: vec![], ..task.clone() }, candidates, rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.0.update(&Task { features: vec![], ..task.clone() }, agent, reward)
    }
}

fn make_front(b: Base, kw: &HashMap<String, usize>) -> Box<dyn Router> {
    let g = GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true };
    match b {
        Base::GlobalTs => Box::new(GlobalTs(PooledThompsonRouter::new(NA, 1.0))),
        Base::Keyword => Box::new(KeywordRouter::new(kw.clone(), NA)),
        Base::Eps => Box::new(EpsGreedyRouter::new(NA, 0.99, 0.02)),
        Base::Pts => Box::new(PooledThompsonRouter::new(NA, 1.0)),
        Base::HebR => Box::new(HebbianRouter::new(NA, heb_robust_cfg())),
        Base::Warm => Box::new(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), kw.clone(), NA, 1.0).with_evidence(Evidence::MeanFeature)),
        Base::Guard => Box::new(Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw.clone(), NA, g)),
    }
}

#[derive(Clone, Default)]
struct RunOut {
    h500: f64,
    h2000: f64,
    late: f64,
    vote_frac: f64,
    found_frac: f64,
    /// among vote-played tasks: share whose injected paragraph is structurally relevant / share whose anchor cluster is the task's true cluster
    inj_prec: f64,
    anchor_acc: f64,
    pe_vote: f64,
    pe_front: f64,
    us_per_task: f64,
}

#[derive(Clone)]
struct AStat {
    s: [f64; NA],
    n: [f64; NA],
}

fn run(arm: &Arm, wd: &WData, ws: &WSpec, env: &Env, seed: u64) -> RunOut {
    let t_start = Instant::now();
    let inst = &wd.w.inst[&seed.to_string()];
    let cl = &wd.cl[&ws.k];
    let cands: Vec<usize> = (0..NA).collect();
    let mut fr = make_front(arm.base.unwrap_or(Base::Guard), &env.kw);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let is_vote = arm.base.is_none();
    let mut astats: HashMap<u64, AStat> = HashMap::new();
    let mut gate = [0.0f64; 4];
    let (mut probe, pint) = (0usize, 25usize);
    let (mut fref, mut fref_n) = (0.5f64, 0.0f64);
    let (mut n_vote, mut n_found, mut n_rel, mut n_acc) = (0usize, 0usize, 0usize, 0usize);
    let (mut dv, mut df) = (0.0f64, 0.0f64);
    let (mut cum, mut h500) = (0.0f64, 0.0f64);
    let mut late = 0.0f64;
    let (fl, base_idx) = (arm.sys % 2, arm.sys / 2);
    for t in 0..NT {
        let row: &Row = &inst.rows[env.rows[t]];
        let v = env.var[t] as usize;
        let mut task = Task { id: t as u64, task_type: None, features: env.feats[t].clone() };
        let rel = &wd.w.queries[row.q].rel;
        let mut by_vote = false;
        let mut doc_rel = false;
        let mut anchor_ok = false;
        let mut found_anchor: Option<u64> = None;
        let a = if is_vote { 'blk: {
            // retrieval step: top-1 paragraph above the dense-cosine admission floor
            let admitted: Vec<(usize, f32)> = if arm.sys == ORACLE {
                vec![(rel[0], 1.0)]
            } else {
                let tid = row.t[v][fl] as u32;
                let list = &wd.tops[&(tid, fl as u8)][base_idx * TOPN..(base_idx + 1) * TOPN];
                list.iter().filter(|(d, c)| (!ws.oneoff || !rel.contains(&(*d as usize))) && *c >= arm.thr).take(arm.k).map(|&(d, c)| (d as usize, c)).collect()
            };
            if let Some(&(d0, _)) = admitted.first() {
                // anchor = topic cluster with the largest cosine-weighted vote among the admitted top-k paragraphs; context = the top-1 paragraph
                let mut votes: Vec<(usize, f32)> = vec![];
                for &(d, c) in &admitted {
                    match votes.iter_mut().find(|x| x.0 == cl[d]) {
                        Some(x) => x.1 += c,
                        None => votes.push((cl[d], c)),
                    }
                }
                let win = votes.iter().fold((usize::MAX, f32::MIN), |b, &x| if x.1 > b.1 || (x.1 == b.1 && x.0 < b.0) { x } else { b }).0;
                found_anchor = Some(win as u64);
                doc_rel = ws.ctx_real && rel.contains(&d0);
                anchor_ok = win == env.cl_true[t];
                n_found += 1;
            }
            if arm.gate == GateK::Ctx {
                let rate_ok = gate[1] < 6.0 || gate[3] < 6.0 || gate[0] / gate[1] >= gate[2] / gate[3] - 0.02;
                let open = !admitted.is_empty()
                    && (rate_ok || {
                        probe += 1;
                        if probe >= pint {
                            probe = 0;
                            true
                        } else {
                            false
                        }
                    });
                found_anchor = None;
                if open {
                    by_vote = true;
                    n_vote += 1;
                    if doc_rel {
                        n_rel += 1;
                    }
                    if anchor_ok {
                        n_acc += 1;
                    }
                }
                break 'blk fr.select(&task, &cands, &mut rng);
            }
            let mut best: Option<(usize, f64, f64, f64)> = None;
            if let Some(an) = found_anchor {
                if let Some(rw) = astats.get(&an) {
                    for a in 0..NA {
                        let (s, n) = (rw.s[a], rw.n[a]);
                        if n < arm.min_n as f64 - 1e-9 {
                            continue;
                        }
                        let m = (1.0 + s) / (2.0 + n);
                        if m >= 0.7 && best.is_none_or(|b| m > b.1) {
                            best = Some((a, m, s, n));
                        }
                    }
                }
            }
            let g1_open = best.is_none() || {
                let rate_ok = gate[1] < 6.0 || gate[3] < 6.0 || gate[0] / gate[1] >= gate[2] / gate[3] - 0.02;
                if rate_ok {
                    true
                } else {
                    probe += 1;
                    if probe >= pint {
                        probe = 0;
                        true
                    } else {
                        false
                    }
                }
            };
            let mut open = false;
            if let Some((_, _, s, n)) = best {
                open = match arm.gate {
                    GateK::G1 | GateK::Ctx => g1_open,
                    GateK::Safe => {
                        let conf = 1.0 - inc_beta(1.0 + s, 1.0 + (n - s), fref.clamp(1e-6, 1.0 - 1e-6));
                        conf >= 0.8 && g1_open
                    }
                };
            }
            let inj = open || (arm.gate == GateK::G1 && found_anchor.is_some());
            if let (true, Some(an)) = (inj, found_anchor) {
                task.features.push(format!("kn:{an}"));
            }
            match best {
                Some((a, ..)) if open => {
                    by_vote = true;
                    n_vote += 1;
                    if doc_rel {
                        n_rel += 1;
                    }
                    if anchor_ok {
                        n_acc += 1;
                    }
                    a
                }
                _ => fr.select(&task, &cands, &mut rng),
            }
        } } else {
            fr.select(&task, &cands, &mut rng)
        };
        // outcome model
        let tab = if ws.drift && t >= NT / 2 { &env.p_post } else { &env.p_pre };
        let mut pe = tab[env.cl_true[t]][a];
        if by_vote {
            pe = if doc_rel { pe + ws.gain * (1.0 - pe) } else { pe * (1.0 - PEN) };
        }
        let s = if env.u[t] < pe { 1.0 } else { 0.0 };
        fr.update(&task, a, s);
        if is_vote {
            if let Some(an) = found_anchor {
                let rw = astats.entry(an).or_insert(AStat { s: [0.0; NA], n: [0.0; NA] });
                rw.s[a] += s;
                rw.n[a] += 1.0;
            }
            if by_vote {
                gate[0] += s;
                gate[1] += 1.0;
            } else {
                gate[2] += s;
                gate[3] += 1.0;
                fref_n += 1.0;
                let al = (1.0 / (fref_n + 1.0)).max(0.03);
                fref += al * (s - fref);
            }
        }
        if by_vote { dv += pe } else { df += pe }
        cum += pe;
        if t + 1 == 500 {
            h500 = cum / 500.0;
        }
        if t >= 1000 {
            late += pe;
        }
    }
    RunOut {
        h500,
        h2000: cum / NT as f64,
        late: late / (NT - 1000) as f64,
        vote_frac: n_vote as f64 / NT as f64,
        found_frac: n_found as f64 / NT as f64,
        inj_prec: if n_vote > 0 { n_rel as f64 / n_vote as f64 } else { f64::NAN },
        anchor_acc: if n_vote > 0 { n_acc as f64 / n_vote as f64 } else { f64::NAN },
        pe_vote: if n_vote > 0 { dv / n_vote as f64 } else { f64::NAN },
        pe_front: df / (NT - n_vote).max(1) as f64,
        us_per_task: t_start.elapsed().as_secs_f64() * 1e6 / NT as f64,
    }
}

// ------------------------------------------------------------------------------------------------ stats helpers

fn one_sided_p(t: f64, df: usize) -> f64 {
    let p2 = t_two_sided_p(t, df);
    if t >= 0.0 { p2 / 2.0 } else { 1.0 - p2 / 2.0 }
}
#[derive(Clone, Debug)]
struct Cmp {
    diff: f64,
    lo: f64,
    hi: f64,
    p: f64,
    tost_p: f64,
}
fn cmp(a: &[f64], b: &[f64], eq: f64) -> Cmp {
    let d = paired_diff(a, b);
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let n = diffs.len();
    let sm = summarize(&diffs);
    let tost = if eq > 0.0 && n >= 2 && sm.sd > 1e-15 {
        let se = sm.sd / (n as f64).sqrt();
        one_sided_p((sm.mean + eq) / se, n - 1).max(one_sided_p((eq - sm.mean) / se, n - 1))
    } else if eq > 0.0 {
        if sm.mean.abs() < eq { 0.0 } else { 1.0 }
    } else {
        f64::NAN
    };
    Cmp { diff: d.diff.mean, lo: d.diff.lo, hi: d.diff.hi, p: paired_t_pvalue(a, b), tost_p: tost }
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

type Table = HashMap<(String, String), Vec<RunOut>>; // (arm, world) -> per seed

fn eval_all(arms: &[Arm], wl: &[WSpec], wd: &WData, sds: &[u64]) -> Table {
    // streams once per (world, seed)
    let envs: Vec<Vec<Env>> = wl.iter().map(|ws| sds.par_iter().map(|&s| make_env(wd, ws, s)).collect()).collect();
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new(); // (arm, world, seed idx)
    for ai in 0..arms.len() {
        for wi in 0..wl.len() {
            for si in 0..sds.len() {
                tasks.push((ai, wi, si));
            }
        }
    }
    let outs: Vec<RunOut> = tasks.par_iter().map(|&(ai, wi, si)| run(&arms[ai], wd, &wl[wi], &envs[wi][si], sds[si])).collect();
    let mut t: Table = HashMap::new();
    for (k, &(ai, wi, _)) in tasks.iter().enumerate() {
        t.entry((arms[ai].name.clone(), wl[wi].name.to_string())).or_default().push(outs[k].clone());
    }
    t
}

fn col(t: &Table, arm: &str, world: &str, m: usize) -> Vec<f64> {
    t[&(arm.to_string(), world.to_string())].iter().map(|r| if m == 0 { r.h500 } else { r.h2000 }).collect()
}
const METRICS: [&str; 2] = ["H500", "H2000"];
const BASELINES: [&str; 7] = ["keyword", "eps-greedy", "pooled-ts", "hebbian-robust", "warm(hebr)", "guarded", "global-ts"];
const BASELINES6: [&str; 6] = ["keyword", "eps-greedy", "pooled-ts", "hebbian-robust", "warm(hebr)", "guarded"];

fn best_fair(t: &Table, world: &str, m: usize) -> (String, Vec<f64>) {
    best_fair_of(t, world, m, &BASELINES)
}
fn best_fair_of(t: &Table, world: &str, m: usize, set: &[&str]) -> (String, Vec<f64>) {
    let mut b = (String::new(), vec![], f64::MIN);
    for n in set {
        let c = col(t, n, world, m);
        let mu = mean(&c);
        if mu > b.2 {
            b = (n.to_string(), c, mu);
        }
    }
    (b.0, b.1)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let outp = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/h13_route.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let t0 = Instant::now();
    let mut log = String::new();
    let wa = load_wdata("A");
    let wb = load_wdata("B");
    println!("loaded: A tops {}, B tops {} ({:.1}s)", wa.tops.len(), wb.tops.len(), t0.elapsed().as_secs_f64());
    let wl = worlds();
    let baselines: Vec<Arm> = vec![
        Arm::baseline("keyword", Base::Keyword),
        Arm::baseline("eps-greedy", Base::Eps),
        Arm::baseline("pooled-ts", Base::Pts),
        Arm::baseline("hebbian-robust", Base::HebR),
        Arm::baseline("warm(hebr)", Base::Warm),
        Arm::baseline("guarded", Base::Guard),
        Arm::baseline("global-ts", Base::GlobalTs),
    ];
    // ------------------------------------------------------------ TUNE (A seeds 1-8)
    let tune_seeds = if quick { seeds(1, 3) } else { seeds(1, 8) };
    let panel: Vec<WSpec> = wl.iter().filter(|w| w.tune).cloned().collect();
    let tb = eval_all(&baselines, &panel, &wa, &tune_seeds);
    let bf: HashMap<String, f64> = panel.iter().filter(|w| !w.treat).map(|w| (w.name.to_string(), (mean(&best_fair_of(&tb, w.name, 0, &BASELINES6).1) + mean(&best_fair_of(&tb, w.name, 1, &BASELINES6).1)) / 2.0)).collect();
    let thrs = [0.30f32, 0.40, 0.50, 0.60, 0.70];
    let mut picks: BTreeMap<String, (f32, usize, usize, f64)> = BTreeMap::new();
    let mut tune_log = Vec::new();
    for gate in [GateK::Safe, GateK::G1, GateK::Ctx] {
        for sys in 0..13 {
            let mut cfgs: Vec<Arm> = Vec::new();
            for &thr in if sys == ORACLE { &thrs[..1] } else { &thrs[..] } {
                for mn in if gate == GateK::Ctx { &[2usize][..] } else { &[2usize, 3][..] } {
                    for k in if sys == ORACLE || gate == GateK::Ctx { &[1usize][..] } else { &[1usize, 3, 5][..] } {
                        let mut a = Arm::vote(sys, gate, thr, *mn, *k);
                        a.name = format!("{}|thr{:.2}|n{}|k{}", arm_name(gate, sys), thr, mn, k);
                        cfgs.push(a);
                    }
                }
            }
            let tt = eval_all(&cfgs, &panel, &wa, &tune_seeds);
            let mut best = (f64::MIN, 0usize);
            for (ci, c) in cfgs.iter().enumerate() {
                let mut win = vec![];
                let mut pen = vec![];
                for w in &panel {
                    let v = (mean(&col(&tt, &c.name, w.name, 0)) + mean(&col(&tt, &c.name, w.name, 1))) / 2.0;
                    let ref_v = (mean(&col(&tb, "guarded", w.name, 0)) + mean(&col(&tb, "guarded", w.name, 1))) / 2.0;
                    if w.treat {
                        win.push(v - ref_v * 0.0);
                    } else {
                        pen.push((bf[w.name] - v - 0.003).max(0.0));
                    }
                }
                let obj = mean(&win) - 2.0 * mean(&pen);
                if obj > best.0 {
                    best = (obj, ci);
                }
            }
            let c = &cfgs[best.1];
            picks.insert(arm_name(gate, sys), (c.thr, c.min_n, c.k, best.0));
            tune_log.push(json!({"arm": arm_name(gate, sys), "thr": c.thr, "min_n": c.min_n, "k": c.k, "objective": best.0}));
        }
    }
    log.push_str("TUNE (A seeds 1-8 panel {T12,T12-drift,Tctx,ctl-irrel}, objective = mean treat (H500+H2000)/2 - 2 x control penalty vs best fair):\n");
    for (k, v) in &picks {
        log.push_str(&format!("  {k:18} thr {:.2} min_n {} k {} obj {:.4}
", v.0, v.1, v.2, v.3));
    }
    json_write("results/h13_route_picks.json", &json!({"picks": tune_log}));
    let mut arms = baselines.clone();
    for gate in [GateK::Safe, GateK::G1, GateK::Ctx] {
        for sys in 0..13 {
            let p = picks[&arm_name(gate, sys)];
            arms.push(Arm::vote(sys, gate, p.0, p.1, p.2));
        }
    }
    println!("tuned ({:.0}s)", t0.elapsed().as_secs_f64());

    // ------------------------------------------------------------ frozen evaluations
    let (s_held, s_conf) = if quick { (seeds(31, 36), seeds(61, 66)) } else { (seeds(31, 60), seeds(61, 90)) };
    let mut result = json!({"picks": tune_log, "worlds": wl.iter().map(|w| json!({"name": w.name, "k": w.k, "topic_dep": w.topic_dep, "drift": w.drift, "ctx_gain": w.gain, "ctx_real": w.ctx_real, "oneoff": w.oneoff, "treat": w.treat, "tuned_on": w.tune})).collect::<Vec<_>>()});
    let mut verdicts = json!({});
    for (label, wd, sds) in [("A_held_31_60", &wa, &s_held), ("A_confirm_61_90", &wa, &s_conf), ("B_confirm_61_90", &wb, &s_conf)] {
        let tt = eval_all(&arms, &wl, wd, sds);
        let (r, v) = report_split(label, &arms, &wl, &tt, sds.len(), &mut log);
        result[label] = r;
        verdicts[label] = v;
        println!("{label} done ({:.0}s)", t0.elapsed().as_secs_f64());
    }
    result["verdicts"] = verdicts;
    // ---- POST-HOC sensitivity (declared as such): context-only injection admission-floor sweep in the context-essential world Tqa, the T12 and the ctl-irrel control
    {
        let sw: Vec<WSpec> = wl.iter().filter(|w| ["Tqa", "T12", "ctl-irrel"].contains(&w.name)).cloned().collect();
        let mut sweep = json!({});
        log.push_str("
=== POST-HOC sweep (not tuned, not part of any verdict): ctx-only injection, H2000 by admission floor thr, seeds as the split ===
");
        for (label, wd, sds) in [("A_held_31_60", &wa, &s_held), ("B_confirm_61_90", &wb, &s_conf)] {
            let mut arms2: Vec<Arm> = vec![Arm::baseline("guarded", Base::Guard)];
            for sys in 0..13 {
                for &thr in &[0.30f32, 0.40, 0.50, 0.60, 0.70] {
                    let mut a = Arm::vote(sys, GateK::Ctx, if sys == ORACLE { 0.0 } else { thr }, 2, 1);
                    a.name = format!("ctx|{}|thr{:.2}", SYS_NAMES[sys], thr);
                    arms2.push(a);
                }
            }
            let tt = eval_all(&arms2, &sw, wd, sds);
            let mut cell = json!({});
            for w in &sw {
                log.push_str(&format!("{label} world {} (guarded {:.4}):  thr .30 .40 .50 .60 .70
", w.name, mean(&col(&tt, "guarded", w.name, 1))));
                for sys in 0..13 {
                    let vals: Vec<f64> = [0.30f32, 0.40, 0.50, 0.60, 0.70].iter().map(|thr| mean(&col(&tt, &format!("ctx|{}|thr{:.2}", SYS_NAMES[sys], thr), w.name, 1))).collect();
                    log.push_str(&format!("   {:11} {}
", SYS_NAMES[sys], vals.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>().join(" ")));
                    cell[format!("{}|{}", w.name, SYS_NAMES[sys])] = json!(vals);
                }
            }
            // paired contrasts in Tqa at thr .30 and .40 (Holm over the 2 x 8 pairs)
            let nm = |sys: &str, thr: &str| format!("ctx|{sys}|thr{thr}");
            let pairs = [("bm25_s", "dense_s"), ("hybz_s", "dense_s"), ("hybrrf_s", "dense_s"), ("hybrsf_s", "dense_s"), ("rerank_s", "dense_s"), ("hybz_s", "hybrrf_s"), ("dense_s", "dense_raw"), ("hybz_s", "hybz_raw")];
            let mut rows = vec![];
            for thr in ["0.30", "0.40"] {
                for (a, b) in pairs {
                    rows.push((thr, a, b, cmp(&col(&tt, &nm(a, thr), "Tqa", 1), &col(&tt, &nm(b, thr), "Tqa", 1), 0.0)));
                }
            }
            let adj = holm_adjust(&rows.iter().map(|r| r.3.p).collect::<Vec<_>>());
            let mut pj = vec![];
            for (i, (thr, a, b, c)) in rows.iter().enumerate() {
                log.push_str(&format!("   {label} Tqa thr {thr}: {a:9} - {b:9} {:+.4} [{:+.4},{:+.4}] holm {:.3} {}
", c.diff, c.lo, c.hi, adj[i], mark(adj[i], c.diff)));
                pj.push(json!({"thr": thr, "a": a, "b": b, "diff": c.diff, "lo": c.lo, "hi": c.hi, "holm": adj[i]}));
            }
            cell["Tqa_pairs"] = Value::Array(pj);
            sweep[label] = cell;
        }
        result["posthoc_ctx_sweep"] = sweep;
    }
    result["wall_s"] = json!(t0.elapsed().as_secs_f64());
    json_write(&outp, &result);
    let _ = std::fs::write(outp.replace(".json", ".txt"), &log);
    println!("{log}");
    println!("wrote {outp} in {:.0}s", t0.elapsed().as_secs_f64());
}

fn json_write(p: &str, v: &Value) {
    std::fs::write(p, serde_json::to_string_pretty(v).unwrap()).unwrap();
}

fn mark(adj: f64, d: f64) -> &'static str {
    if adj < 0.05 { if d > 0.0 { "*" } else { "!" } } else { "" }
}

fn report_split(label: &str, arms: &[Arm], wl: &[WSpec], t: &Table, nseeds: usize, log: &mut String) -> (Value, Value) {
    let mut out = json!({});
    log.push_str(&format!("\n=================== {label} ({nseeds} seeds) ===================\n"));
    // levels
    let mut lv = json!({});
    log.push_str(&format!("H2000 (mean expected success; vote% / precision of injected paragraph / anchor acc in brackets for safe arms)\n{:20}", "arm"));
    for w in wl {
        log.push_str(&format!(" {:>10}", w.name));
    }
    log.push('\n');
    for a in arms {
        let mut line = format!("{:20}", a.name);
        let mut row = json!({});
        for w in wl {
            let rs = &t[&(a.name.clone(), w.name.to_string())];
            let h2: Vec<f64> = rs.iter().map(|r| r.h2000).collect();
            let h5: Vec<f64> = rs.iter().map(|r| r.h500).collect();
            let lt: Vec<f64> = rs.iter().map(|r| r.late).collect();
            let nm = |f: &dyn Fn(&RunOut) -> f64| -> f64 {
                let v: Vec<f64> = rs.iter().map(f).filter(|x| x.is_finite()).collect();
                if v.is_empty() { f64::NAN } else { mean(&v) }
            };
            line.push_str(&format!(" {:>10.4}", mean(&h2)));
            row[w.name] = json!({"H500": mean(&h5), "H2000": mean(&h2), "late": mean(&lt), "vote_frac": nm(&|r| r.vote_frac), "found_frac": nm(&|r| r.found_frac), "inj_prec": nm(&|r| r.inj_prec), "anchor_acc": nm(&|r| r.anchor_acc), "pe_vote": nm(&|r| r.pe_vote), "pe_front": nm(&|r| r.pe_front), "us_per_task": nm(&|r| r.us_per_task)});
        }
        log.push_str(&line);
        log.push('\n');
        lv[a.name.clone()] = row;
    }
    out["levels"] = lv;
    let vote_arms: Vec<&Arm> = arms.iter().filter(|a| a.base.is_none()).collect();
    // F1: vs same front (guarded), F2: vs best fair; Holm per family. Include TOST for controls.
    let mut fam = |reff: &dyn Fn(&str, usize) -> (String, Vec<f64>), name: &str, log: &mut String| -> Value {
        let mut rows: Vec<(String, String, usize, Cmp)> = vec![];
        for a in &vote_arms {
            for w in wl {
                for m in 0..2 {
                    let x = col(t, &a.name, w.name, m);
                    let (_, r) = reff(w.name, m);
                    rows.push((a.name.clone(), w.name.to_string(), m, cmp(&x, &r, if w.treat { 0.0 } else { 0.01 })));
                }
            }
        }
        let adj = holm_adjust(&rows.iter().map(|r| r.3.p).collect::<Vec<_>>());
        let ctl_idx: Vec<usize> = (0..rows.len()).filter(|&i| !wl.iter().find(|w| w.name == rows[i].1).unwrap().treat).collect();
        let tadj = holm_adjust(&ctl_idx.iter().map(|&i| rows[i].3.tost_p).collect::<Vec<_>>());
        let mut tmap: HashMap<usize, f64> = HashMap::new();
        for (k, &i) in ctl_idx.iter().enumerate() {
            tmap.insert(i, tadj[k]);
        }
        let mut js = vec![];
        for (i, (a, w, m, c)) in rows.iter().enumerate() {
            js.push(json!({"arm": a, "world": w, "metric": METRICS[*m], "diff": c.diff, "lo": c.lo, "hi": c.hi, "p": c.p, "holm": adj[i], "win": adj[i] < 0.05 && c.diff > 0.0, "loss": adj[i] < 0.05 && c.diff < 0.0, "tost_holm_pm01": tmap.get(&i).copied()}));
        }
        log.push_str(&format!("--- {name}: H2000 diff [95% CI] (* Holm win / ! loss; in controls 'eq' = TOST +-.01 Holm-equivalent) ---\n{:20}", "arm"));
        for w in wl {
            log.push_str(&format!(" {:>16}", w.name));
        }
        log.push('\n');
        for a in &vote_arms {
            let mut line = format!("{:20}", a.name);
            for w in wl {
                let i = rows.iter().position(|r| r.0 == a.name && r.1 == w.name && r.2 == 1).unwrap();
                let c = &rows[i].3;
                let eq = tmap.get(&i).map(|&p| if p < 0.05 { " eq" } else { "" }).unwrap_or("");
                line.push_str(&format!(" {:>+7.3}{}{:3}", c.diff, mark(adj[i], c.diff), eq));
                line.push_str("    ");
            }
            log.push_str(&line);
            log.push('\n');
        }
        Value::Array(js)
    };
    let front = |w: &str, m: usize| ("guarded".to_string(), col(t, "guarded", w, m));
    let bfair = |w: &str, m: usize| best_fair(t, w, m);
    let bfair6 = |w: &str, m: usize| best_fair_of(t, w, m, &BASELINES6);
    out["vs_same_front"] = fam(&front, "vs the SAME front guarded(hebr) (no retrieval)", log);
    out["vs_best_fair"] = fam(&bfair, "vs BEST FAIR baseline per world (7 baselines incl. post-hoc global-ts)", log);
    out["vs_best_fair6"] = fam(&bfair6, "vs BEST FAIR of the 6 pre-declared baselines", log);
    // F3: pairs among safe arms vs safe|dense_s
    let mut f3 = vec![];
    let refs = "same-gate dense_s";
    let ref_of = |a: &Arm| arm_name(a.gate, 1);
    let pair_arms: Vec<&Arm> = vote_arms.iter().filter(|a| a.sys != 1).cloned().collect();
    let mut rows3: Vec<(String, String, usize, Cmp)> = vec![];
    for a in &pair_arms {
        for w in wl {
            for m in 0..2 {
                let (x, r) = (col(t, &a.name, w.name, m), col(t, &ref_of(a), w.name, m));
                rows3.push((a.name.clone(), w.name.to_string(), m, cmp(&x, &r, 0.0)));
            }
        }
    }
    let adj3 = holm_adjust(&rows3.iter().map(|r| r.3.p).collect::<Vec<_>>());
    for (i, (a, w, m, c)) in rows3.iter().enumerate() {
        f3.push(json!({"arm": a, "world": w, "metric": METRICS[*m], "diff": c.diff, "lo": c.lo, "hi": c.hi, "p": c.p, "holm": adj3[i], "win": adj3[i] < 0.05 && c.diff > 0.0, "loss": adj3[i] < 0.05 && c.diff < 0.0}));
    }
    log.push_str(&format!("--- vs {refs} (same gate, spell-corrected dense), H2000 diff (* Holm win / ! loss) ---\n{:20}", "arm"));
    for w in wl {
        log.push_str(&format!(" {:>10}", w.name));
    }
    log.push('\n');
    for a in &pair_arms {
        let mut line = format!("{:20}", a.name);
        for w in wl {
            let i = rows3.iter().position(|r| r.0 == a.name && r.1 == w.name && r.2 == 1).unwrap();
            line.push_str(&format!(" {:>+8.3}{:1} ", rows3[i].3.diff, mark(adj3[i], rows3[i].3.diff)));
        }
        log.push_str(&line);
        log.push('\n');
    }
    out["vs_safe_dense_s"] = Value::Array(f3);
    // verdicts
    let mut verd = json!({});
    let sg = |fam: &Value, arm: &str, world: &str, key: &str| -> bool { fam.as_array().unwrap().iter().any(|r| r["arm"] == arm && r["world"] == world && r["metric"] == "H2000" && r[key] == true) };
    for a in &vote_arms {
        let tw: Vec<&str> = wl.iter().filter(|w| w.treat).map(|w| w.name).collect();
        let cw: Vec<&str> = wl.iter().filter(|w| !w.treat).map(|w| w.name).collect();
        let nwin_front = tw.iter().filter(|w| sg(&out["vs_same_front"], &a.name, w, "win")).count();
        let ctl_loss_front = cw.iter().filter(|w| sg(&out["vs_same_front"], &a.name, w, "loss")).count();
        let nwin_bf = tw.iter().filter(|w| sg(&out["vs_best_fair"], &a.name, w, "win")).count();
        let ctl_loss_bf = cw.iter().filter(|w| sg(&out["vs_best_fair"], &a.name, w, "loss")).count();
        let nwin_bf6 = tw.iter().filter(|w| sg(&out["vs_best_fair6"], &a.name, w, "win")).count();
        let ctl_loss_bf6 = cw.iter().filter(|w| sg(&out["vs_best_fair6"], &a.name, w, "loss")).count();
        let (nwin_d, nloss_d) = if a.sys == 1 { (0, 0) } else { (tw.iter().filter(|w| sg(&out["vs_safe_dense_s"], &a.name, w, "win")).count(), wl.iter().filter(|w| sg(&out["vs_safe_dense_s"], &a.name, w.name, "loss")).count()) };
        verd[a.name.clone()] = json!({"treat_wins_vs_front": nwin_front, "control_losses_vs_front": ctl_loss_front, "keep_vs_front": nwin_front >= 2 && ctl_loss_front == 0,
            "treat_wins_vs_bestfair6": nwin_bf6, "control_losses_vs_bestfair6": ctl_loss_bf6, "keep_vs_bestfair6": nwin_bf6 >= 2 && ctl_loss_bf6 == 0,
            "treat_wins_vs_bestfair": nwin_bf, "control_losses_vs_bestfair": ctl_loss_bf, "keep_vs_bestfair": nwin_bf >= 2 && ctl_loss_bf == 0,
            "treat_wins_vs_safe_dense_s": nwin_d, "losses_vs_safe_dense_s_any_world": nloss_d, "beats_safe_dense_s": nwin_d >= 2 && nloss_d == 0});
    }
    log.push_str("--- verdicts (treat worlds = T12,T12-drift,Tctx,T48,T12-noctx,Tqa; controls = ctl-irrel, ctl-oneoff) ---\n");
    for a in &vote_arms {
        let v = &verd[a.name.clone()];
        log.push_str(&format!("  {:18} vs front: wins {}/6 ctl-losses {} keep={} | vs best fair: wins {}/6 ctl-losses {} keep={} | vs safe|dense_s: wins {}/6 losses {} beats={}\n", a.name, v["treat_wins_vs_front"], v["control_losses_vs_front"], v["keep_vs_front"], v["treat_wins_vs_bestfair"], v["control_losses_vs_bestfair"], v["keep_vs_bestfair"], v["treat_wins_vs_safe_dense_s"], v["losses_vs_safe_dense_s_any_world"], v["beats_safe_dense_s"]));
    }
    (out, verd)
}
