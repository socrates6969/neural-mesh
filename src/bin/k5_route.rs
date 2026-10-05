//! K5: CONFIDENCE-GATED RETRIEVAL, front router per world. Closed-loop routing on the REAL-text replay (h10/h13 fixtures, copied engine of h13_route).
//!
//! HONEST FRAMING: this studies the knowledge ACCESS layer of a routing system (retrieve a real paragraph, vote an agent from its topic anchor, inject context);
//! it does NOT add knowledge to the frozen model. REAL: task texts, the paragraph store, retrieval top lists (h13, frozen config hybz_s = z-hybrid alpha .7 + spell), keyword features, topic clusters.
//! SIMULATED (labelled): agent competence tables, keyword prior, context effect (relevant injected paragraph p' = p + gain (1-p); irrelevant p' = .95 p), every Bernoulli outcome.
//! Context utility is STRUCTURAL (injected top-1 paragraph is in the task's source section). No real traffic: still synthetic-until-validated.
//!
//! PRE-DECLARED (before any held-out/confirm run):
//!  worlds  : h13 worlds T12, T12-drift, Tctx, T48, T12-noctx, Tqa(post-hoc, no verdict), ctl-irrel, ctl-oneoff (+ NEW control ctl-noinfo: the TEXT (features + retrieval) of each task is that of a random
//!            OTHER query while the hidden best agent depends on the true cluster = text carries no information). Success worlds: T12, T12-drift, T48. Controls: ctl-irrel, ctl-oneoff, ctl-noinfo.
//!  fair set: BF7 = keyword, eps-greedy, pooled-ts, hebbian-robust, warm(hebr), guarded, global-ts (global-ts now PRE-DECLARED; in topic-independent worlds it is the reference).
//!            BF8 = BF7 + the adaptive meta-front WITHOUT retrieval (isolates the retrieval value; reported, stricter).
//!  fronts  : guarded(hebr) | warm(hebr) | global-ts | META = Thompson sampling over fronts on the reward of front-played tasks (all fronts are trained on every outcome).
//!  gates   : g1 (h13: window gate, inject kn always) | safe (h13 G-post+g1) | shrink (g16 FINAL: anchor evidence shrunk to the agent's global rate, Beta posterior vs front ref, window gate, inject only when open)
//!            | local (Bayes test p_vote > p_front-in-this-anchor, normal approx of the Beta difference, shrunk to the global front rate, asymmetric-loss threshold gamma) | + online A/B holdout wrapper
//!            (randomised share eps of gate-eligible tasks withholds the layer; switch the layer off when P(effect > 0) < gamma_ab; measure continues while off).
//!  tuning  : A seeds 1-8, panel {T12, T12-drift, Tctx, ctl-irrel, ctl-noinfo}; objective = mean_treat (H500+H2000)/2 - 2 x mean_ctl max(0, BF7 - v - .003); never tuned on: T48, T12-noctx, Tqa, ctl-oneoff.
//!            Retrieval parameters thr/min_n/k of g1/safe are the FROZEN h13 picks (hybz_s); META params tuned first on the front-only arm, then frozen for all META arms; each new gate tuned with META frozen.
//!  splits  : held A 31-60, frozen confirm A 61-90 and B 61-90 (independent text world). Paired 95% t-CI, Holm per family, TOST +-.01 in controls.
//! Usage: k5_route tune | eval held|confA|confB | check

#[path = "k5_common/mod.rs"]
mod common;

use std::collections::HashMap;
use std::time::Instant;

use common::{Row, World, load_world, seeds};
use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter, sample_beta};
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
const PEN: f64 = 0.05;
const HYBZ_S: usize = 5; // retrieval system id of z-hybrid + spell (h13 SYS_NAMES index)

// ------------------------------------------------------------------------------------------------ data

struct WData {
    w: World,
    cl: HashMap<usize, Vec<usize>>,
    kwtok: HashMap<usize, Vec<(String, usize)>>,
    feat: HashMap<u32, Vec<String>>,
    tops: HashMap<(u32, u8), Vec<(u32, f32)>>,
}

fn load_wdata(tag: &str) -> WData {
    let w = load_world(tag);
    let j: Value = serde_json::from_slice(&std::fs::read(format!("results/h13_world_{tag}.json")).expect("h13 fixtures")).unwrap();
    let mut cl = HashMap::new();
    let mut kwtok = HashMap::new();
    for k in [12usize, 48] {
        cl.insert(k, j["cl"][k.to_string()].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect());
        let mut v: Vec<(String, usize)> = j["kwtok"][k.to_string()].as_object().unwrap().iter().map(|(a, b)| (a.clone(), b.as_u64().unwrap() as usize)).collect();
        v.sort();
        kwtok.insert(k, v);
    }
    let feat = j["feat"].as_object().unwrap().iter().map(|(k, v)| (k.parse().unwrap(), v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect())).collect();
    let raw = std::fs::read(format!("results/h13_top_{tag}.bin")).expect("h13 top lists");
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
    ctx_real: bool,
    oneoff: bool,
    /// NEW (K5): task text (features + retrieval) comes from a random OTHER query; competence still depends on the true cluster
    noinfo: bool,
    treat: bool,
    tune: bool,
    low_base: bool,
    /// part of the pre-declared success criterion
    crit: bool,
}
fn worlds() -> Vec<WSpec> {
    let s = |name, k, topic_dep, drift, gain, ctx_real, oneoff, treat, tune, crit| WSpec { name, k, topic_dep, drift, gain, ctx_real, oneoff, noinfo: false, treat, tune, low_base: false, crit };
    vec![
        s("T12", 12, true, false, 0.15, true, false, true, true, true),
        s("T12-drift", 12, true, true, 0.15, true, false, true, true, true),
        s("Tctx", 12, false, false, 0.20, true, false, true, true, false),
        s("T48", 48, true, false, 0.15, true, false, true, false, true),
        s("T12-noctx", 12, true, false, 0.0, true, false, true, false, false),
        s("ctl-irrel", 12, false, false, 0.0, false, false, false, true, true),
        s("ctl-oneoff", 12, true, false, 0.15, true, true, false, false, true),
        WSpec { noinfo: true, ..s("ctl-noinfo", 12, true, false, 0.15, true, false, false, true, true) },
        WSpec { low_base: true, ..s("Tqa", 12, false, false, 0.85, true, false, true, false, false) },
    ]
}

struct Env {
    rows: Vec<usize>,  // row of the TRUE task (competence cluster, relevance)
    trows: Vec<usize>, // row whose TEXT the router/retrieval sees (== rows except in noinfo worlds)
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
    let (mut rows, mut trows, mut var, mut u, mut cl_true, mut feats) = (vec![], vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..NT {
        let ri = r.below(nrows);
        let v = if r.f64() < 0.75 { 0u8 } else { 1 + r.below(2) as u8 };
        let uu = r.f64();
        let ti = if ws.noinfo {
            // text of a random OTHER query (different query id)
            let mut x = r.below(nrows);
            let mut guard = 0;
            while inst.rows[x].q == inst.rows[ri].q && guard < 50 {
                x = r.below(nrows);
                guard += 1;
            }
            x
        } else {
            ri
        };
        let row = &inst.rows[ri];
        let trow = &inst.rows[ti];
        rows.push(ri);
        trows.push(ti);
        var.push(v);
        u.push(uu);
        cl_true.push(cl[wd.w.queries[row.q].rel[0]]);
        feats.push(wd.feat.get(&(trow.t[v as usize][0] as u32)).cloned().unwrap_or_default());
    }
    Env { rows, trows, var, u, cl_true, feats, p_pre, p_post, kw }
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
enum Gate {
    None,
    G1,
    Safe,
    Shrink,
    Local,
}
#[derive(Clone, Copy, Debug)]
struct Ab {
    eps: f64,
    gamma: f64,
    burn: usize,
    disc: f64,
}
#[derive(Clone, Debug)]
struct Cfg {
    name: String,
    fronts: Vec<Base>,
    /// per front: does the pipeline with this front run the retrieval layer? (empty = all true)
    retr_on: Vec<bool>,
    /// meta credit includes vote-played tasks (pipeline-level credit)
    credit_all: bool,
    /// prior mean of the retrieval pipelines in the meta Thompson (others .5): incumbent bias
    meta_p0: f64,
    /// context injection only if the dense cosine of the top-1 paragraph >= inj_thr (retrieval-level confidence gate on the CONTEXT; the vote itself is not affected)
    inj_thr: f32,
    meta_disc: f64,
    meta_n0: f64,
    gate: Gate,
    thr: f32,
    min_n: usize,
    k: usize,
    kappa: f64,
    gamma: f64,
    floor: f64,
    delta: f64,
    kappa_f: f64,
    ab: Option<Ab>,
}
impl Cfg {
    fn front_only(name: &str, fronts: Vec<Base>) -> Cfg {
        Cfg { name: name.into(), fronts, retr_on: vec![], credit_all: false, meta_p0: 0.5, inj_thr: 0.0, meta_disc: 0.995, meta_n0: 2.0, gate: Gate::None, thr: 0.3, min_n: 2, k: 5, kappa: 3.0, gamma: 0.5, floor: 0.6, delta: 0.0, kappa_f: 10.0, ab: None }
    }
    fn json(&self) -> Value {
        json!({"name": self.name, "retr_on": self.retr_on, "credit_all": self.credit_all, "meta_p0": self.meta_p0, "inj_thr": self.inj_thr, "fronts": self.fronts.iter().map(|f| format!("{f:?}")).collect::<Vec<_>>(), "meta_disc": self.meta_disc, "meta_n0": self.meta_n0, "gate": format!("{:?}", self.gate),
            "thr": self.thr, "min_n": self.min_n, "k": self.k, "kappa": self.kappa, "gamma": self.gamma, "floor": self.floor, "delta": self.delta, "kappa_f": self.kappa_f,
            "ab": self.ab.map(|a| json!({"eps": a.eps, "gamma": a.gamma, "burn": a.burn, "disc": a.disc}))})
    }
}

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

fn phi(z: f64) -> f64 {
    let p = common::norm_two_sided_p(z);
    if z >= 0.0 { 1.0 - p / 2.0 } else { p / 2.0 }
}
fn beta_mv(a: f64, b: f64) -> (f64, f64) {
    let n = a + b;
    (a / n, a * b / (n * n * (n + 1.0)))
}

#[derive(Clone, Default)]
struct RunOut {
    h250: f64,
    h500: f64,
    h2000: f64,
    late: f64,
    vote_frac: f64,
    inj_prec: f64,
    hold_frac: f64,
    on_frac: f64,
    elig_frac: f64,
    fshare: [f64; 3],
    us_per_task: f64,
}

#[derive(Clone)]
struct AStat {
    s: [f64; NA],
    n: [f64; NA],
}

fn run(cfg: &Cfg, wd: &WData, ws: &WSpec, env: &Env, seed: u64) -> RunOut {
    let t_start = Instant::now();
    let inst = &wd.w.inst[&seed.to_string()];
    let cl = &wd.cl[&ws.k];
    let cands: Vec<usize> = (0..NA).collect();
    let mut frs: Vec<Box<dyn Router>> = cfg.fronts.iter().map(|b| make_front(*b, &env.kw)).collect();
    let nf = frs.len();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let has_ret = cfg.gate != Gate::None;
    let mut astats: HashMap<u64, AStat> = HashMap::new();
    let mut fa: HashMap<u64, (f64, f64)> = HashMap::new();
    let (mut gs, mut gn) = ([0.0f64; NA], [0.0f64; NA]);
    let mut ms = [0.0f64; 3];
    let mut mn = [0.0f64; 3];
    let mut fused = [0.0f64; 3];
    let mut gate = [0.0f64; 4];
    let (mut probe, pint) = (0usize, 25usize);
    let (mut fref, mut fref_n) = (0.5f64, 0.0f64);
    let (mut n_vote, mut n_rel, mut n_inj) = (0usize, 0usize, 0usize);
    let (mut cum, mut h500, mut h250, mut late) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    // A/B state
    let (mut ts, mut tn, mut hs, mut hn, mut hraw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
    let mut ab_on = true;
    let (mut n_hold, mut n_elig, mut n_on) = (0usize, 0usize, 0usize);
    let shrinky = matches!(cfg.gate, Gate::Shrink | Gate::Local);
    let (fl, base_idx) = (HYBZ_S % 2, HYBZ_S / 2);
    for t in 0..NT {
        let row: &Row = &inst.rows[env.trows[t]];
        let true_row: &Row = &inst.rows[env.rows[t]];
        let v = env.var[t] as usize;
        let mut task = Task { id: t as u64, task_type: None, features: env.feats[t].clone() };
        let rel = &wd.w.queries[true_row.q].rel;
        let mut by_vote = false;
        let mut doc_rel = false;
        let mut ctx_ok = true;
        let mut found_anchor: Option<u64> = None;
        let fi = if nf == 1 {
            0
        } else {
            let mut bi = 0;
            let mut bv = f64::NEG_INFINITY;
            for f in 0..nf {
                let p0 = if cfg.retr_on.get(f).copied().unwrap_or(true) { cfg.meta_p0 } else { 0.5 };
                let x = sample_beta(p0 * cfg.meta_n0 + ms[f], (1.0 - p0) * cfg.meta_n0 + (mn[f] - ms[f]).max(0.0), &mut rng);
                if x > bv {
                    bv = x;
                    bi = f;
                }
            }
            bi
        };
        let pipe_ret = cfg.retr_on.get(fi).copied().unwrap_or(true);
        let mut vote_agent: Option<usize> = None;
        let mut hold = false;
        let mut treated = false;
        let mut elig = false;
        if has_ret {
            let tid = row.t[v][fl] as u32;
            let list = &wd.tops[&(tid, fl as u8)][base_idx * TOPN..(base_idx + 1) * TOPN];
            let admitted: Vec<(usize, f32)> = list.iter().filter(|(d, c)| (!ws.oneoff || !rel.contains(&(*d as usize))) && *c >= cfg.thr).take(cfg.k).map(|&(d, c)| (d as usize, c)).collect();
            if let Some(&(d0, _)) = admitted.first() {
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
                ctx_ok = admitted[0].1 >= cfg.inj_thr;
            }
            let mut best: Option<(usize, f64, f64, f64)> = None; // agent, m, s_eff, n_eff
            if let Some(an) = found_anchor {
                if let Some(rw) = astats.get(&an) {
                    for a in 0..NA {
                        let (s, n) = (rw.s[a], rw.n[a]);
                        if n < cfg.min_n as f64 - 1e-9 {
                            continue;
                        }
                        let (se, ne) = if shrinky {
                            let g = (1.0 + gs[a]) / (2.0 + gn[a]);
                            (s + cfg.kappa * g, n + cfg.kappa)
                        } else {
                            (s, n)
                        };
                        let m = (1.0 + se) / (2.0 + ne);
                        let floor = if shrinky { cfg.floor } else { 0.7 };
                        if m >= floor && best.is_none_or(|b| m > b.1) {
                            best = Some((a, m, se, ne));
                        }
                    }
                }
            }
            let g1_open = !pipe_ret || best.is_none() || {
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
            if let Some((_, _, se, ne)) = best {
                open = match cfg.gate {
                    Gate::G1 => g1_open,
                    Gate::Safe => {
                        // h13 safe: raw counts, gamma .8
                        let (s, n) = (se, ne);
                        let conf = 1.0 - inc_beta(1.0 + s, 1.0 + (n - s), fref.clamp(1e-6, 1.0 - 1e-6));
                        conf >= 0.8 && g1_open
                    }
                    Gate::Shrink => {
                        let conf = 1.0 - inc_beta(1.0 + se, 1.0 + (ne - se), (fref + cfg.delta).clamp(1e-6, 1.0 - 1e-6));
                        conf >= cfg.gamma && g1_open
                    }
                    Gate::Local => {
                        let an = found_anchor.unwrap();
                        let (fs, fnn) = fa.get(&an).copied().unwrap_or((0.0, 0.0));
                        let (af, bf) = (cfg.kappa_f * fref + fs, cfg.kappa_f * (1.0 - fref) + (fnn - fs));
                        let (mf, vf) = beta_mv(af, bf);
                        let (mv, vv) = beta_mv(1.0 + se, 1.0 + (ne - se));
                        let z = (mv - mf - cfg.delta) / (vv + vf).sqrt();
                        phi(z) >= cfg.gamma && g1_open
                    }
                    Gate::None => false,
                };
            }
            let mut inj = pipe_ret && (open || (cfg.gate == Gate::G1 && found_anchor.is_some()));
            if !pipe_ret {
                open = false;
            }
            elig = pipe_ret && (if cfg.gate == Gate::G1 { found_anchor.is_some() } else { open });
            if let (Some(ab), true) = (cfg.ab, elig) {
                n_elig += 1;
                let q = if hraw < ab.burn { 0.5 } else if ab_on { 1.0 - ab.eps } else { ab.eps };
                if ab_on || hraw < ab.burn {
                    n_on += 1;
                }
                if rng.f64() < q {
                    treated = true;
                } else {
                    hold = true;
                    n_hold += 1;
                    open = false;
                    inj = false;
                }
            } else if elig {
                n_elig += 1;
            }
            if let (true, Some(an)) = (inj, found_anchor) {
                task.features.push(format!("kn:{an}"));
            }
            if open {
                vote_agent = best.map(|b| b.0);
            }
        }
        let a = match vote_agent {
            Some(a) => {
                by_vote = true;
                n_vote += 1;
                if ctx_ok {
                    n_inj += 1;
                    if doc_rel {
                        n_rel += 1;
                    }
                }
                a
            }
            None => frs[fi].select(&task, &cands, &mut rng),
        };
        let tab = if ws.drift && t >= NT / 2 { &env.p_post } else { &env.p_pre };
        let mut pe = tab[env.cl_true[t]][a];
        if by_vote && ctx_ok {
            pe = if doc_rel { pe + ws.gain * (1.0 - pe) } else { pe * (1.0 - PEN) };
        }
        let s = if env.u[t] < pe { 1.0 } else { 0.0 };
        for f in frs.iter_mut() {
            f.update(&task, a, s);
        }
        gs[a] += s;
        gn[a] += 1.0;
        if has_ret {
            if let Some(an) = found_anchor {
                let rw = astats.entry(an).or_insert(AStat { s: [0.0; NA], n: [0.0; NA] });
                rw.s[a] += s;
                rw.n[a] += 1.0;
                if !by_vote {
                    let e = fa.entry(an).or_insert((0.0, 0.0));
                    e.0 += s;
                    e.1 += 1.0;
                }
            }
            if by_vote {
                gate[0] += s;
                gate[1] += 1.0;
            } else if pipe_ret {
                gate[2] += s;
                gate[3] += 1.0;
                fref_n += 1.0;
                let al = (1.0 / (fref_n + 1.0)).max(0.03);
                fref += al * (s - fref);
            }
            if let Some(ab) = cfg.ab {
                if treated || hold {
                    ts *= ab.disc;
                    tn *= ab.disc;
                    hs *= ab.disc;
                    hn *= ab.disc;
                    if treated {
                        ts += s;
                        tn += 1.0;
                    } else {
                        hs += s;
                        hn += 1.0;
                        hraw += 1;
                    }
                    if hraw >= ab.burn && tn >= 3.0 && hn >= 3.0 {
                        let (pt, ph) = ((ts + 0.5) / (tn + 1.0), (hs + 0.5) / (hn + 1.0));
                        let var = pt * (1.0 - pt) / (tn + 1.0) + ph * (1.0 - ph) / (hn + 1.0);
                        ab_on = phi((pt - ph) / var.sqrt()) >= ab.gamma;
                    }
                }
            }
        }
        if nf > 1 && (!by_vote || cfg.credit_all) {
            for f in 0..nf {
                ms[f] *= cfg.meta_disc;
                mn[f] *= cfg.meta_disc;
            }
            ms[fi] += s;
            mn[fi] += 1.0;
        }
        if !by_vote {
            let slot = match cfg.fronts[fi] {
                Base::Warm => 1,
                Base::GlobalTs => 2,
                _ => 0,
            };
            fused[slot] += 1.0;
        }
        cum += pe;
        if t + 1 == 250 {
            h250 = cum / 250.0;
        }
        if t + 1 == 500 {
            h500 = cum / 500.0;
        }
        if t >= 1000 {
            late += pe;
        }
    }
    let fsum = fused.iter().sum::<f64>().max(1.0);
    RunOut {
        h250,
        h500,
        h2000: cum / NT as f64,
        late: late / (NT - 1000) as f64,
        vote_frac: n_vote as f64 / NT as f64,
        inj_prec: if n_inj > 0 { n_rel as f64 / n_inj as f64 } else { f64::NAN },
        hold_frac: n_hold as f64 / NT as f64,
        on_frac: if n_elig > 0 { n_on as f64 / n_elig as f64 } else { f64::NAN },
        elig_frac: n_elig as f64 / NT as f64,
        fshare: [fused[0] / fsum, fused[1] / fsum, fused[2] / fsum],
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

type Table = HashMap<(String, String), Vec<RunOut>>;

fn eval_all(arms: &[Cfg], wl: &[WSpec], wd: &WData, sds: &[u64]) -> Table {
    let envs: Vec<Vec<Env>> = wl.iter().map(|ws| sds.par_iter().map(|&s| make_env(wd, ws, s)).collect()).collect();
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
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
const BF7: [&str; 7] = ["keyword", "eps-greedy", "pooled-ts", "hebbian-robust", "warm(hebr)", "guarded", "global-ts"];

fn best_of(t: &Table, world: &str, m: usize, set: &[&str]) -> (String, Vec<f64>) {
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

fn json_write(p: &str, v: &Value) {
    std::fs::write(p, serde_json::to_string_pretty(v).unwrap()).unwrap();
}
fn mark(adj: f64, d: f64) -> &'static str {
    if adj < 0.05 { if d > 0.0 { "*" } else { "!" } } else { "" }
}

// ------------------------------------------------------------------------------------------------ arm construction

fn baselines() -> Vec<Cfg> {
    vec![
        Cfg::front_only("keyword", vec![Base::Keyword]),
        Cfg::front_only("eps-greedy", vec![Base::Eps]),
        Cfg::front_only("pooled-ts", vec![Base::Pts]),
        Cfg::front_only("hebbian-robust", vec![Base::HebR]),
        Cfg::front_only("warm(hebr)", vec![Base::Warm]),
        Cfg::front_only("guarded", vec![Base::Guard]),
        Cfg::front_only("global-ts", vec![Base::GlobalTs]),
    ]
}

const PRIMARY: [&str; 10] = ["g1|guard", "g1|warm", "g1|meta3", "P2w", "P2w+", "P3", "P2g", "shrink|meta3", "local|meta3", "ab(g1)|meta3"];
const META3: [Base; 3] = [Base::Guard, Base::Warm, Base::GlobalTs];
const METAG: [Base; 2] = [Base::Guard, Base::GlobalTs];

fn retr(name: &str, fronts: &[Base], gate: Gate, meta: (f64, f64)) -> Cfg {
    // frozen h13 picks for hybz_s: g1 thr .30 min_n 2 k5; safe thr .30 min_n 2 k5
    let mut c = Cfg::front_only(name, fronts.to_vec());
    c.gate = gate;
    c.meta_disc = meta.0;
    c.meta_n0 = meta.1;
    c
}

/// pipeline-level meta arm: gate G1 (frozen h13 picks), credit on all plays
fn pipe_cfg(name: &str, fronts: &[Base], retr_on: &[bool]) -> Cfg {
    let mut c = Cfg::front_only(name, fronts.to_vec());
    c.gate = Gate::G1;
    c.retr_on = retr_on.to_vec();
    c.credit_all = true;
    c
}

fn load_picks() -> Value {
    serde_json::from_slice(&std::fs::read("results/k5_picks.json").expect("run `k5_route tune` first")).unwrap()
}
fn cfg_from(v: &Value) -> Cfg {
    let fr = |s: &str| match s {
        "Guard" => Base::Guard,
        "Warm" => Base::Warm,
        "GlobalTs" => Base::GlobalTs,
        "Keyword" => Base::Keyword,
        "Eps" => Base::Eps,
        "Pts" => Base::Pts,
        _ => Base::HebR,
    };
    let g = match v["gate"].as_str().unwrap() {
        "G1" => Gate::G1,
        "Safe" => Gate::Safe,
        "Shrink" => Gate::Shrink,
        "Local" => Gate::Local,
        _ => Gate::None,
    };
    Cfg {
        name: v["name"].as_str().unwrap().to_string(),
        retr_on: v["retr_on"].as_array().map(|a| a.iter().map(|x| x.as_bool().unwrap()).collect()).unwrap_or_default(),
        credit_all: v["credit_all"].as_bool().unwrap_or(false),
        meta_p0: v["meta_p0"].as_f64().unwrap_or(0.5),
        inj_thr: v["inj_thr"].as_f64().unwrap_or(0.0) as f32,
        fronts: v["fronts"].as_array().unwrap().iter().map(|x| fr(x.as_str().unwrap())).collect(),
        meta_disc: v["meta_disc"].as_f64().unwrap(),
        meta_n0: v["meta_n0"].as_f64().unwrap(),
        gate: g,
        thr: v["thr"].as_f64().unwrap() as f32,
        min_n: v["min_n"].as_u64().unwrap() as usize,
        k: v["k"].as_u64().unwrap() as usize,
        kappa: v["kappa"].as_f64().unwrap(),
        gamma: v["gamma"].as_f64().unwrap(),
        floor: v["floor"].as_f64().unwrap(),
        delta: v["delta"].as_f64().unwrap(),
        kappa_f: v["kappa_f"].as_f64().unwrap(),
        ab: if v["ab"].is_null() { None } else { Some(Ab { eps: v["ab"]["eps"].as_f64().unwrap(), gamma: v["ab"]["gamma"].as_f64().unwrap(), burn: v["ab"]["burn"].as_u64().unwrap() as usize, disc: v["ab"]["disc"].as_f64().unwrap() }) },
    }
}

/// objective of one config on the tuning table: mean treat (H500+H2000)/2 - 2 x mean control max(0, BF7 - v - .003)
fn objective(tt: &Table, name: &str, panel: &[WSpec], bf: &HashMap<String, f64>) -> f64 {
    let (mut win, mut pen) = (vec![], vec![]);
    for w in panel {
        let v = (mean(&col(tt, name, w.name, 0)) + mean(&col(tt, name, w.name, 1))) / 2.0;
        if w.treat {
            win.push(v);
        } else {
            pen.push((bf[w.name] - v - 0.003).max(0.0));
        }
    }
    mean(&win) - 2.0 * mean(&pen)
}

fn tune() {
    let t0 = Instant::now();
    let wa = load_wdata("A");
    let wl = worlds();
    let panel: Vec<WSpec> = wl.iter().filter(|w| w.tune).cloned().collect();
    let sds = seeds(1, 8);
    let base = baselines();
    let tb = eval_all(&base, &panel, &wa, &sds);
    let bf: HashMap<String, f64> = panel.iter().filter(|w| !w.treat).map(|w| (w.name.to_string(), (mean(&best_of(&tb, w.name, 0, &BF7).1) + mean(&best_of(&tb, w.name, 1, &BF7).1)) / 2.0)).collect();
    let mut log = String::new();
    log.push_str(&format!("tune panel {:?}; BF7 control refs {:?}\n", panel.iter().map(|w| w.name).collect::<Vec<_>>(), bf));
    let mut picks: Vec<Value> = vec![];
    // ---- stage 1: META params on the front-only arms
    let mut best1 = (f64::MIN, (0.995, 2.0));
    let mut cf: Vec<Cfg> = vec![];
    for &d in &[1.0, 0.995, 0.98] {
        for &n0 in &[2.0, 10.0] {
            let mut c = Cfg::front_only(&format!("meta3|d{d}|n{n0}"), META3.to_vec());
            c.meta_disc = d;
            c.meta_n0 = n0;
            cf.push(c);
        }
    }
    let tt = eval_all(&cf, &panel, &wa, &sds);
    for c in &cf {
        let o = objective(&tt, &c.name, &panel, &bf);
        log.push_str(&format!("  stage1 {:22} obj {:.4}\n", c.name, o));
        if o > best1.0 {
            best1 = (o, (c.meta_disc, c.meta_n0));
        }
    }
    let meta = best1.1;
    // ---- stage 1b: PIPELINE-level arms (credit on all plays): grid over META (n0) and the retrieval params thr / k / min_n (h13 picks were tuned for the guarded front)
    let mut best1b = (f64::MIN, (1.0, 2.0));
    {
        let fams: Vec<(&str, Vec<Base>, Vec<bool>)> = vec![
            ("g1|warm*", vec![Base::Warm], vec![true]),
            ("P2w", vec![Base::Warm, Base::GlobalTs], vec![true, false]),
            ("P2w+", vec![Base::Warm, Base::GlobalTs], vec![true, true]),
            ("P3", vec![Base::Guard, Base::Warm, Base::GlobalTs], vec![true, true, false]),
            ("P2g", vec![Base::Guard, Base::GlobalTs], vec![true, false]),
        ];
        for (fname, fr, ro) in fams {
            let mut cf: Vec<Cfg> = vec![];
            for &n0 in if fr.len() > 1 { &[10.0, 30.0][..] } else { &[2.0][..] } {
                for p0 in [0.5f64] {
                    for &thr in &[0.3f32, 0.5] {
                        for &k in &[3usize, 5] {
                            for &it in &[0.0f32, 0.6, 0.7, 0.8, 2.0] {
                                let mut c = pipe_cfg(&format!("{fname}|n{n0}|p{p0}|t{thr}|k{k}|i{it}"), &fr, &ro);
                                c.meta_disc = 1.0;
                                c.meta_n0 = n0;
                                c.meta_p0 = p0;
                                c.thr = thr;
                                c.k = k;
                                c.min_n = 2;
                                c.inj_thr = it;
                                cf.push(c);
                            }
                        }
                    }
                }
            }
            let tt = eval_all(&cf, &panel, &wa, &sds);
            let mut best = (f64::MIN, 0usize);
            for (i, c) in cf.iter().enumerate() {
                let o = objective(&tt, &c.name, &panel, &bf);
                if o > best.0 {
                    best = (o, i);
                }
            }
            let mut c = cf[best.1].clone();
            log.push_str(&format!("STAGE1b {fname:10} best {} obj {:.4}
", c.name, best.0));
            c.name = fname.trim_end_matches('*').to_string();
            let mut j = c.json();
            j["objective"] = json!(best.0);
            picks.push(j);
            if fname == "P2w" {
                best1b = (best.0, (c.meta_disc, c.meta_n0));
            }
        }
    }
    let mk = |name: &str, fronts: &[Base], gate: Gate| retr(name, fronts, gate, meta);
    // ---- stage 2: the new gates (META frozen). Each family: grid -> best objective.
    let mut fam: Vec<(String, Vec<Cfg>)> = vec![];
    // shrink
    {
        let mut v = vec![];
        for &kappa in &[3.0, 6.0] {
            for &gamma in &[0.5, 0.7] {
                for &mn in &[1usize, 2] {
                    for &floor in &[0.5, 0.6] {
                        let mut c = mk("shrink|meta3", &META3, Gate::Shrink);
                        c.kappa = kappa;
                        c.gamma = gamma;
                        c.min_n = mn;
                        c.floor = floor;
                        c.name = format!("shrink|meta3|k{kappa}|g{gamma}|n{mn}|f{floor}");
                        v.push(c);
                    }
                }
            }
        }
        fam.push(("shrink|meta3".into(), v));
    }
    // local
    {
        let mut v = vec![];
        for &gamma in &[0.5, 0.7, 0.9] {
            for &kf in &[5.0, 20.0] {
                for &kappa in &[3.0, 6.0] {
                    for &mn in &[1usize, 2] {
                        let mut c = mk("local|meta3", &META3, Gate::Local);
                        c.kappa = kappa;
                        c.gamma = gamma;
                        c.kappa_f = kf;
                        c.min_n = mn;
                        c.name = format!("local|meta3|g{gamma}|kf{kf}|k{kappa}|n{mn}");
                        v.push(c);
                    }
                }
            }
        }
        fam.push(("local|meta3".into(), v));
    }
    // A/B wrappers
    let ab_grid: Vec<Ab> = {
        let mut v = vec![];
        for &eps in &[0.1, 0.2] {
            for &gamma in &[0.6, 0.8] {
                for &burn in &[10usize, 25] {
                    for &disc in &[1.0, 0.995] {
                        v.push(Ab { eps, gamma, burn, disc });
                    }
                }
            }
        }
        v
    };
    for (fname, fronts, gate) in [("ab(g1)|meta3", &META3[..], Gate::G1), ("ab(shrink)|meta3", &META3[..], Gate::Shrink), ("ab(g1)|guard", &[Base::Guard][..], Gate::G1)] {
        let mut v = vec![];
        for ab in &ab_grid {
            let mut c = mk(fname, fronts, gate);
            if gate == Gate::G1 {
                c.min_n = 2;
                c.k = 5;
            } else {
                c.kappa = 3.0;
                c.gamma = 0.5;
                c.min_n = 1;
                c.floor = 0.6;
            }
            c.ab = Some(*ab);
            c.name = format!("{fname}|e{}|g{}|b{}|d{}", ab.eps, ab.gamma, ab.burn, ab.disc);
            v.push(c);
        }
        fam.push((fname.into(), v));
    }
    // front-switch only arms with the frozen h13 gates are NOT tuned (frozen picks)
    for (fname, cfgs) in &fam {
        let tt = eval_all(cfgs, &panel, &wa, &sds);
        let mut best = (f64::MIN, 0usize);
        for (i, c) in cfgs.iter().enumerate() {
            let o = objective(&tt, &c.name, &panel, &bf);
            if o > best.0 {
                best = (o, i);
            }
        }
        let mut c = cfgs[best.1].clone();
        log.push_str(&format!("STAGE2 {fname:20} best {} obj {:.4}\n", c.name, best.0));
        // show the top 3
        let mut ov: Vec<(f64, String)> = cfgs.iter().map(|c| (objective(&tt, &c.name, &panel, &bf), c.name.clone())).collect();
        ov.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for (o, n) in ov.iter().take(3) {
            log.push_str(&format!("      {n:50} {o:.4}\n"));
        }
        c.name = fname.clone();
        let mut j = c.json();
        j["objective"] = json!(best.0);
        picks.push(j);
    }
    json_write("results/k5_picks.json", &json!({"meta": {"disc": meta.0, "n0": meta.1, "obj": best1.0}, "meta_pipe": {"disc": best1b.1.0, "n0": best1b.1.1, "obj": best1b.0}, "picks": picks, "bf7_control_ref": bf}));
    let _ = std::fs::write("results/k5_tune.txt", &log);
    println!("{log}\ntune done in {:.0}s", t0.elapsed().as_secs_f64());
}

fn final_arms() -> Vec<Cfg> {
    let pj = load_picks();
    let meta = (pj["meta"]["disc"].as_f64().unwrap(), pj["meta"]["n0"].as_f64().unwrap());
    let mut arms = baselines();
    // front-only references
    let mut m3 = Cfg::front_only("meta3-front", META3.to_vec());
    m3.meta_disc = meta.0;
    m3.meta_n0 = meta.1;
    let mut mg = Cfg::front_only("metaG-front", METAG.to_vec());
    mg.meta_disc = meta.0;
    mg.meta_n0 = meta.1;
    arms.push(m3);
    arms.push(mg);
    // frozen h13 gates
    for (n, fr, g) in [
        ("g1|guard", vec![Base::Guard], Gate::G1),
        ("safe|guard", vec![Base::Guard], Gate::Safe),
        ("g1|meta3", META3.to_vec(), Gate::G1),
        ("safe|meta3", META3.to_vec(), Gate::Safe),
        ("g1|metaG", METAG.to_vec(), Gate::G1),
        ("safe|warm", vec![Base::Warm], Gate::Safe),
        ("g1|meta2", vec![Base::Guard, Base::Warm], Gate::G1),
    ] {
        arms.push(retr(n, &fr, g, meta));
    }
    for p in pj["picks"].as_array().unwrap() {
        arms.push(cfg_from(p));
    }
        // POST-HOC ablations (appended after the frozen runs; no verdict, outside the PRIMARY family): isolate the warm front and the context-injection switch
    let picked = |n: &str| arms.iter().find(|x| x.name == n).cloned().unwrap();
    let mut ab_arms: Vec<Cfg> = vec![];
    for (nm, base, inj) in [("g1|warm(ctx)", "g1|warm", 0.0f32), ("g1|guard(noctx)", "g1|guard", 2.0), ("g1|meta3(noctx)", "g1|meta3", 2.0), ("P2w(ctx)", "P2w", 0.0), ("P2w+(noctx)", "P2w+", 2.0), ("P2w+(ctx)", "P2w+", 0.0)] {
        let mut c = picked(base);
        c.name = nm.into();
        c.inj_thr = inj;
        ab_arms.push(c);
    }
    // feature-only ablation: the retrieval anchor is injected as the feature kn:<anchor> into the front, the vote never opens (min_n huge)
    for (nm, base) in [("feat|warm", "g1|warm"), ("feat|P2w", "P2w"), ("feat|P2w+", "P2w+"), ("feat|meta3", "g1|meta3")] {
        let mut c = picked(base);
        c.name = nm.into();
        c.min_n = 1_000_000_000;
        ab_arms.push(c);
    }
    arms.extend(ab_arms);
    arms
}

fn eval(split: &str) {
    let t0 = Instant::now();
    let (tag, sds) = match split {
        "held" => ("A", seeds(31, 60)),
        "confA" => ("A", seeds(61, 90)),
        "confB" => ("B", seeds(61, 90)),
        _ => panic!("split"),
    };
    let wd = load_wdata(tag);
    let wl = worlds();
    let arms = final_arms();
    let tt = eval_all(&arms, &wl, &wd, &sds);
    let mut log = String::new();
    let rep = report(split, &arms, &wl, &tt, sds.len(), &mut log);
    json_write(&format!("results/k5_{split}.json"), &rep);
    let _ = std::fs::write(format!("results/k5_{split}.txt"), &log);
    println!("{log}\n{split} done in {:.0}s", t0.elapsed().as_secs_f64());
}

/// retrieval diagnostics: P(top-1 paragraph structurally relevant | dense cosine of the top-1) and P(anchor == true cluster | cosine) on the tuning seeds, both text worlds
fn diag() {
    for tag in ["A", "B"] {
        let wd = load_wdata(tag);
        let ws = worlds().into_iter().find(|w| w.name == "T12").unwrap();
        let mut bins = vec![(0usize, 0usize, 0usize); 10];
        for seed in seeds(1, 8) {
            let env = make_env(&wd, &ws, seed);
            let inst = &wd.w.inst[&seed.to_string()];
            let cl = &wd.cl[&12];
            for t in 0..NT {
                let row = &inst.rows[env.trows[t]];
                let rel = &wd.w.queries[row.q].rel;
                let v = env.var[t] as usize;
                let tid = row.t[v][1] as u32;
                let list = &wd.tops[&(tid, 1u8)][2 * TOPN..3 * TOPN];
                let (d0, c0) = list[0];
                let b = ((c0.max(0.0) * 10.0) as usize).min(9);
                bins[b].0 += 1;
                if rel.contains(&(d0 as usize)) {
                    bins[b].1 += 1;
                }
                if cl[d0 as usize] == env.cl_true[t] {
                    bins[b].2 += 1;
                }
            }
        }
        println!("world {tag} (hybz_s top-1, T12 tasks, seeds 1-8): cos-bin  n  P(rel)  P(cluster ok)");
        for (i, b) in bins.iter().enumerate() {
            if b.0 > 0 {
                println!("  {:.1}-{:.1}  {:6}  {:.3}  {:.3}", i as f64 / 10.0, (i + 1) as f64 / 10.0, b.0, b.1 as f64 / b.0 as f64, b.2 as f64 / b.0 as f64);
            }
        }
    }
}

fn check() {
    // fidelity: the engine must reproduce h13's g1|hybz_s and safe|hybz_s (guarded front) levels (A held 31-60)
    let wd = load_wdata("A");
    let wl: Vec<WSpec> = worlds().into_iter().filter(|w| ["T12", "T12-drift", "ctl-irrel", "ctl-oneoff", "T48"].contains(&w.name)).collect();
    let arms = vec![Cfg::front_only("guarded", vec![Base::Guard]), retr("g1|guard", &[Base::Guard], Gate::G1, (0.995, 2.0)), retr("safe|guard", &[Base::Guard], Gate::Safe, (0.995, 2.0))];
    let tt = eval_all(&arms, &wl, &wd, &seeds(31, 60));
    let h13: Value = serde_json::from_slice(&std::fs::read("results/h13_route.json").unwrap()).unwrap();
    for (mine, theirs) in [("guarded", "guarded"), ("g1|guard", "g1|hybz_s"), ("safe|guard", "safe|hybz_s")] {
        for w in &wl {
            let a = mean(&col(&tt, mine, w.name, 1));
            let b = h13["A_held_31_60"]["levels"][theirs][w.name]["H2000"].as_f64().unwrap();
            println!("{mine:11} {:11} k5 {:.6} h13 {:.6} diff {:+.2e}", w.name, a, b, a - b);
        }
    }
}

// ------------------------------------------------------------------------------------------------ report

fn report(label: &str, arms: &[Cfg], wl: &[WSpec], t: &Table, nseeds: usize, log: &mut String) -> Value {
    let mut out = json!({});
    log.push_str(&format!("\n=================== {label} ({nseeds} seeds) ===================\nH2000 mean expected success\n{:24}", "arm"));
    for w in wl {
        log.push_str(&format!(" {:>10}", w.name));
    }
    log.push('\n');
    let mut lv = json!({});
    for a in arms {
        let mut line = format!("{:24}", a.name);
        let mut row = json!({});
        for w in wl {
            let rs = &t[&(a.name.clone(), w.name.to_string())];
            let nm = |f: &dyn Fn(&RunOut) -> f64| -> f64 {
                let v: Vec<f64> = rs.iter().map(f).filter(|x| x.is_finite()).collect();
                if v.is_empty() { f64::NAN } else { mean(&v) }
            };
            line.push_str(&format!(" {:>10.4}", nm(&|r| r.h2000)));
            row[w.name] = json!({"H250": nm(&|r| r.h250), "H500": nm(&|r| r.h500), "H2000": nm(&|r| r.h2000), "late": nm(&|r| r.late), "vote_frac": nm(&|r| r.vote_frac), "inj_prec": nm(&|r| r.inj_prec),
                "hold_frac": nm(&|r| r.hold_frac), "on_frac": nm(&|r| r.on_frac), "elig_frac": nm(&|r| r.elig_frac), "fshare_guard": nm(&|r| r.fshare[0]), "fshare_warm": nm(&|r| r.fshare[1]), "fshare_gts": nm(&|r| r.fshare[2]), "us_per_task": nm(&|r| r.us_per_task)});
        }
        log.push_str(&line);
        log.push('\n');
        lv[a.name.clone()] = row;
    }
    out["levels"] = lv;
    log.push_str("H500\n");
    for a in arms {
        let mut line = format!("{:24}", a.name);
        for w in wl {
            line.push_str(&format!(" {:>10.4}", mean(&col(t, &a.name, w.name, 0))));
        }
        log.push_str(&line);
        log.push('\n');
    }
    log.push_str("H250 (cold start)\n");
    for a in arms {
        let mut line = format!("{:24}", a.name);
        for w in wl {
            let v: Vec<f64> = t[&(a.name.clone(), w.name.to_string())].iter().map(|r| r.h250).collect();
            line.push_str(&format!(" {:>10.4}", mean(&v)));
        }
        log.push_str(&line);
        log.push('\n');
    }
    let ret_arms: Vec<&Cfg> = arms.iter().filter(|a| a.gate != Gate::None).collect();
    let _ = ret_arms;
    let test_arms: Vec<&Cfg> = arms.iter().filter(|a| a.gate != Gate::None || a.name.ends_with("-front")).collect();
    let bf8: Vec<&str> = BF7.iter().copied().chain(["meta3-front"]).collect();
    let fam = |reff: &dyn Fn(&str, usize) -> Vec<f64>, name: &str, log: &mut String| -> Value {
        let mut rows: Vec<(String, String, usize, Cmp)> = vec![];
        for a in &test_arms {
            for w in wl {
                for m in 0..2 {
                    rows.push((a.name.clone(), w.name.to_string(), m, cmp(&col(t, &a.name, w.name, m), &reff(w.name, m), if w.treat { 0.0 } else { 0.01 })));
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
            js.push(json!({"arm": a, "world": w, "metric": METRICS[*m], "diff": c.diff, "lo": c.lo, "hi": c.hi, "p": c.p, "holm": adj[i], "win": adj[i] < 0.05 && c.diff > 0.0, "loss": adj[i] < 0.05 && c.diff < 0.0, "tost_holm": tmap.get(&i).copied()}));
        }
        log.push_str(&format!("--- {name}: H2000 diff (* Holm win / ! Holm loss; 'e' = TOST +-.01 Holm-equivalent in controls) ---\n{:24}", "arm"));
        for w in wl {
            log.push_str(&format!(" {:>11}", w.name));
        }
        log.push('\n');
        for a in &test_arms {
            let mut line = format!("{:24}", a.name);
            for w in wl {
                let i = rows.iter().position(|r| r.0 == a.name && r.1 == w.name && r.2 == 1).unwrap();
                let eq = tmap.get(&i).map(|&p| if p < 0.05 { "e" } else { " " }).unwrap_or(" ");
                line.push_str(&format!(" {:>+7.3}{}{}  ", rows[i].3.diff, mark(adj[i], rows[i].3.diff), eq));
            }
            log.push_str(&line);
            log.push('\n');
        }
        Value::Array(js)
    };
    let r_bf7 = |w: &str, m: usize| best_of(t, w, m, &BF7).1;
    let r_bf8 = |w: &str, m: usize| best_of(t, w, m, &bf8).1;
    let r_m3 = |w: &str, m: usize| col(t, "meta3-front", w, m);
    let r_bf6 = |w: &str, m: usize| best_of(t, w, m, &BF7[..6]).1;
    let r_g = |w: &str, m: usize| col(t, "guarded", w, m);
    // best-fair identity lines
    log.push_str("best fair (BF7) per world on this split (H2000): ");
    for w in wl {
        log.push_str(&format!("{}={}({:.4}) ", w.name, best_of(t, w.name, 1, &BF7).0, mean(&best_of(t, w.name, 1, &BF7).1)));
    }
    log.push('\n');
    out["vs_bf7"] = fam(&r_bf7, "vs BEST FAIR BF7 (7 pre-declared baselines incl. global-ts) per world", log);
    out["vs_bf8"] = fam(&r_bf8, "vs BEST of BF7 + meta3-front (no retrieval)", log);
    out["vs_meta3"] = fam(&r_m3, "vs meta3-front (adaptive front WITHOUT retrieval) = retrieval value", log);
    out["vs_guarded"] = fam(&r_g, "vs the guarded(hebr) front (h13 'same front')", log);
    // verdicts
    let sg = |f: &Value, arm: &str, world: &str, key: &str| f.as_array().unwrap().iter().any(|r| r["arm"] == arm && r["world"] == world && r["metric"] == "H2000" && r[key] == true);
    let eqf = |f: &Value, arm: &str, world: &str| f.as_array().unwrap().iter().any(|r| r["arm"] == arm && r["world"] == world && r["metric"] == "H2000" && r["tost_holm"].as_f64().is_some_and(|p| p < 0.05));
    let crit_t: Vec<&str> = wl.iter().filter(|w| w.crit && w.treat).map(|w| w.name).collect();
    let crit_c: Vec<&str> = wl.iter().filter(|w| w.crit && !w.treat).map(|w| w.name).collect();
    let mut verd = json!({});
    log.push_str("--- verdicts (criterion worlds: wins in T12,T12-drift,T48; controls ctl-irrel,ctl-oneoff,ctl-noinfo) ---\n");
    for a in &test_arms {
        let mut v = json!({});
        for (k, f) in [("bf7", &out["vs_bf7"]), ("bf8", &out["vs_bf8"]), ("meta3", &out["vs_meta3"]), ("guarded", &out["vs_guarded"])] {
            let wins = crit_t.iter().filter(|w| sg(f, &a.name, w, "win")).count();
            let closs = crit_c.iter().filter(|w| sg(f, &a.name, w, "loss")).count();
            let ceq = crit_c.iter().filter(|w| eqf(f, &a.name, w)).count();
            v[k] = json!({"wins_of_3": wins, "ctl_holm_losses": closs, "ctl_tost_equiv": ceq, "pass_no_loss": wins == 3 && closs == 0, "pass_equiv": wins == 3 && ceq == 3});
        }
        log.push_str(&format!("  {:22} BF7: wins {}/3 ctl-loss {} eq {}/3 | BF8: wins {}/3 ctl-loss {} | vs meta3: wins {}/3 ctl-loss {} | vs guarded: wins {}/3 ctl-loss {}\n", a.name,
            v["bf7"]["wins_of_3"], v["bf7"]["ctl_holm_losses"], v["bf7"]["ctl_tost_equiv"], v["bf8"]["wins_of_3"], v["bf8"]["ctl_holm_losses"], v["meta3"]["wins_of_3"], v["meta3"]["ctl_holm_losses"], v["guarded"]["wins_of_3"], v["guarded"]["ctl_holm_losses"]));
        verd[a.name.clone()] = v;
    }
    out["verdicts"] = verd;
    // ---- PRIMARY verdict family (pre-declared before the confirm runs): candidate arms x criterion worlds x H2000 only, Holm + TOST inside the family
    let primary: Vec<&str> = PRIMARY.iter().copied().filter(|n| arms.iter().any(|a| a.name == *n)).collect();
    let mut prim = json!({});
    for (fname, reff) in [("bf7", &r_bf7 as &dyn Fn(&str, usize) -> Vec<f64>), ("bf6", &r_bf6), ("bf8", &r_bf8), ("meta3", &r_m3)] {
        let mut rows: Vec<(String, String, Cmp)> = vec![];
        for a in &primary {
            for w in wl.iter().filter(|w| w.crit) {
                rows.push((a.to_string(), w.name.to_string(), cmp(&col(t, a, w.name, 1), &reff(w.name, 1), if w.treat { 0.0 } else { 0.01 })));
            }
        }
        let adj = holm_adjust(&rows.iter().map(|r| r.2.p).collect::<Vec<_>>());
        let ci: Vec<usize> = (0..rows.len()).filter(|&i| !wl.iter().find(|w| w.name == rows[i].1).unwrap().treat).collect();
        let tadj = holm_adjust(&ci.iter().map(|&i| rows[i].2.tost_p).collect::<Vec<_>>());
        let mut tm: HashMap<usize, f64> = HashMap::new();
        for (k, &i) in ci.iter().enumerate() {
            tm.insert(i, tadj[k]);
        }
        log.push_str(&format!("=== PRIMARY family vs {fname} (H2000, {} arms x {} criterion worlds; diff [95% CI] Holm-mark / TOST) ===
", primary.len(), wl.iter().filter(|w| w.crit).count()));
        let mut pj = vec![];
        let mut verdict = json!({});
        for a in &primary {
            let mut line = format!("{:16}", a);
            let (mut wins, mut closs, mut ceq, mut cni) = (0, 0, 0, 0);
            for w in wl.iter().filter(|w| w.crit) {
                let i = rows.iter().position(|r| r.0 == *a && r.1 == w.name).unwrap();
                let c = &rows[i].2;
                let mk = mark(adj[i], c.diff);
                let eqm = tm.get(&i).map(|&p| p < 0.05).unwrap_or(false);
                if w.treat && mk == "*" {
                    wins += 1;
                }
                if !w.treat {
                    if mk == "!" {
                        closs += 1;
                    }
                    if eqm {
                        ceq += 1;
                    }
                    if mk != "!" && (eqm || c.lo >= 0.0) {
                        cni += 1;
                    }
                }
                line.push_str(&format!(" {}{:>+6.3}[{:+.3},{:+.3}]{}{}", &w.name[..w.name.len().min(6)], c.diff, c.lo, c.hi, mk, if eqm { "e" } else { "" }));
                pj.push(json!({"arm": a, "world": w.name, "diff": c.diff, "lo": c.lo, "hi": c.hi, "p": c.p, "holm": adj[i], "tost_holm": tm.get(&i)}));
            }
            line.push_str(&format!("  | wins {wins}/3 ctl-loss {closs} ctl-equiv {ceq}/3 ctl-ok(equiv|CI>=0, no loss) {cni}/3"));
            log.push_str(&line);
            log.push('\n');
            verdict[*a] = json!({"wins_of_3": wins, "ctl_holm_losses": closs, "ctl_equiv": ceq, "ctl_ok": cni, "pass_loose": wins == 3 && closs == 0, "pass_strict": wins == 3 && cni == 3});
        }
        prim[fname] = json!({"rows": pj, "verdict": verdict});
    }
    out["primary"] = prim;
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("tune") => tune(),
        Some("eval") => eval(args.get(2).expect("split")),
        Some("check") => check(),
        Some("diag") => diag(),
        _ => eprintln!("usage: k5_route tune | eval held|confA|confB | check"),
    }
}
