//! g16 shared pieces (W17, retrieval track R, round 3): downstream knowledge-routing loop with a PLUGGABLE retrieval policy
//! (confidence gate, identification gate, learned gate, adaptive k, freshness) on the g1 fixtures (corpus A/B, MiniLM / BGE / tf-idf).
//! Included from the g16_* bins with `#[path = "g16_common/mod.rs"] mod g16;` AFTER `#[path = "g1_common/mod.rs"] mod common;`.
//! Everything here is a seeded SIMULATION on top of the g1 fixtures. It measures routing success of the knowledge layer, not LLM quality.
#![allow(dead_code)]

use std::collections::HashMap;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, inc_beta, paired_diff, paired_t_pvalue, t_two_sided_p, summarize};
use rayon::prelude::*;

use crate::common::*;

pub const NA: usize = 6;
pub const NT: usize = 2000;
pub const HORIZONS: [usize; 4] = [100, 250, 500, 2000];

// ------------------------------------------------------------------------------------------------
// similarity pool (precomputed query x query matrices)

pub struct Pool {
    pub nq: usize,
    pub mats: Vec<Vec<f32>>,
    pub names: Vec<String>,
    pub concept: Vec<i32>,
    pub topic: Vec<i32>,
}
impl Pool {
    pub fn mid(&self, n: &str) -> usize {
        self.names.iter().position(|x| x == n).unwrap_or_else(|| panic!("matrix {n}"))
    }
}
#[derive(Clone, Debug)]
pub struct Sim {
    pub cand: Vec<(usize, f32)>,
    pub oracle: bool,
    pub label: String,
}
impl Sim {
    #[inline]
    pub fn s(&self, p: &Pool, i: usize, j: usize) -> f32 {
        let mut a = 0.0;
        for &(m, w) in &self.cand {
            a += w * p.mats[m][i * p.nq + j];
        }
        a
    }
}

fn dense_pair(m: &Model, qv: QV, nq: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nq * nq];
    out.par_chunks_mut(nq).enumerate().for_each(|(i, row)| {
        let a = m.qvec(qv, i);
        for (j, r) in row.iter_mut().enumerate() {
            *r = dot(a, m.qvec(qv, j));
        }
    });
    out
}
fn tfidf_pair(lex: &Lex, c: &Corpus, qv: QV) -> Vec<f32> {
    let nq = c.queries.len();
    let vs: Vec<Vec<(u32, f32)>> = c.queries.iter().map(|q| lex.tfidf(match qv { QV::Raw => &q.text, _ => &q.spell })).collect();
    let mut out = vec![0.0f32; nq * nq];
    out.par_chunks_mut(nq).enumerate().for_each(|(i, row)| {
        for (j, r) in row.iter_mut().enumerate() {
            *r = sparse_dot(&vs[i], &vs[j]);
        }
    });
    out
}
pub fn build_pool(corpus: &Corpus) -> Pool {
    let nq = corpus.queries.len();
    let lex = Lex::build(&corpus.docs);
    let mini = load_model(corpus, "mini");
    let bge = load_model(corpus, "bge");
    let mut names = vec![];
    let mut mats = vec![];
    for (n, v) in [
        ("mini", dense_pair(&mini, QV::Raw, nq)),
        ("bge", dense_pair(&bge, QV::Raw, nq)),
        ("mini_s", dense_pair(&mini, QV::Spell, nq)),
        ("bge_s", dense_pair(&bge, QV::Spell, nq)),
        ("tfidf", tfidf_pair(&lex, corpus, QV::Raw)),
        ("tfidf_s", tfidf_pair(&lex, corpus, QV::Spell)),
    ] {
        names.push(n.to_string());
        mats.push(v);
    }
    Pool { nq, mats, names, concept: corpus.queries.iter().map(|q| q.concept).collect(), topic: corpus.queries.iter().map(|q| q.topic).collect() }
}
/// similarity systems used downstream (index = sim id)
pub fn sims(p: &Pool) -> Vec<Sim> {
    let t = |n: &str| p.mid(n);
    let mk = |l: &str, c: Vec<(usize, f32)>| Sim { cand: c, oracle: false, label: l.into() };
    vec![
        mk("mini", vec![(t("mini"), 1.0)]),
        mk("mini+spell", vec![(t("mini_s"), 1.0)]),
        mk("mini-hybrid", vec![(t("mini_s"), 0.6), (t("tfidf_s"), 0.4)]),
        mk("bge", vec![(t("bge"), 1.0)]),
        mk("bge-hybrid", vec![(t("bge_s"), 0.6), (t("tfidf_s"), 0.4)]),
        mk("tfidf", vec![(t("tfidf"), 1.0)]),
        Sim { cand: vec![], oracle: true, label: "oracle(concept)".into() },
    ]
}
pub const SIM_MINI: usize = 0;
pub const SIM_MINI_S: usize = 1;
pub const SIM_MINI_H: usize = 2;
pub const SIM_BGE: usize = 3;
pub const SIM_BGE_H: usize = 4;
pub const SIM_TFIDF: usize = 5;
pub const SIM_ORACLE: usize = 6;

// ------------------------------------------------------------------------------------------------
// worlds. FIXED BEFORE any policy was run (see results/g16_board_entry.md).

#[derive(Clone, Debug)]
pub struct Spec {
    pub name: &'static str,
    pub sib: bool,
    /// number of topics (x8 sibling details); for !sib: 8*nt random concepts
    pub nt: usize,
    /// context dependence of the best agent (0 = control)
    pub w: f64,
    /// share of tasks that are unrelated one-off texts (ctx -1, w 0); 1.0 = ctl-oneoff
    pub mixed: f64,
    /// weak identification: with this prob the TEXT of a task is drawn from a random OTHER active concept than the one that decides the agent
    pub corrupt: f64,
    /// concept -> best-agent table is re-drawn at this task index (0 = never)
    pub drift_at: usize,
    pub treat: bool,
    /// part of the tuning panel (seeds 1-8)
    pub tune: bool,
}
pub fn worlds() -> Vec<Spec> {
    let s = |name, sib, nt, w, mixed, corrupt, drift_at, treat, tune| Spec { name, sib, nt, w, mixed, corrupt, drift_at, treat, tune };
    vec![
        s("sib16", true, 2, 0.8, 0.0, 0.0, 0, true, true),
        s("rand16", false, 2, 0.8, 0.0, 0.0, 0, true, true),
        s("sib16-mixed50", true, 2, 0.8, 0.5, 0.0, 0, true, true),
        s("sib16-weakid50", true, 2, 0.8, 0.0, 0.5, 0, true, true),
        s("sib16-drift", true, 2, 0.8, 0.0, 0.0, 1000, true, true),
        s("ctl-irrel", true, 2, 0.0, 0.0, 0.0, 0, false, true),
        s("sib64", true, 8, 0.8, 0.0, 0.0, 0, true, false),
        s("ctl-oneoff", true, 2, 0.0, 1.0, 0.0, 0, false, false),
    ]
}

pub struct Stream {
    pub w: World,
    pub q: Vec<usize>,
    pub ctx: Vec<i32>,
    pub wt: Vec<f64>,
    pub pctx: Vec<[f64; NA]>,
    pub pctx2: Option<Vec<[f64; NA]>>,
    pub drift_at: usize,
    pub row: Vec<usize>,
}
pub fn stream(c: &Corpus, seed: u64, sp: &Spec) -> Stream {
    let w = World::new(seed, WorldConfig { n_tasks: NT, scenario: Scenario::Fixed, ..WorldConfig::default() });
    let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xC0DE_6001)); // identical draw order to g1_route::stream
    let mut r2 = Rng::new(seed.wrapping_mul(104729).wrapping_add(0x16C0DE));
    let n = w.len();
    let ntopics = (*c.topic_of_concept.iter().max().unwrap() + 1) as usize;
    let mut active: Vec<usize> = vec![];
    if sp.sib {
        let mut topics: Vec<usize> = (0..ntopics).collect();
        for i in (1..topics.len()).rev() {
            let j = r.below(i + 1);
            topics.swap(i, j);
        }
        for &t in topics.iter().take(sp.nt) {
            active.extend((0..c.nconc).filter(|&k| c.topic_of_concept[k] as usize == t));
        }
    } else {
        let mut all: Vec<usize> = (0..c.nconc).collect();
        for i in (1..all.len()).rev() {
            let j = r.below(i + 1);
            all.swap(i, j);
        }
        active = all.into_iter().take(sp.nt * 8).collect();
    }
    let mut row = vec![usize::MAX; c.nconc];
    let mut pctx = vec![];
    for (k, &cid) in active.iter().enumerate() {
        row[cid] = k;
        let best = r.below(NA);
        let mut pr = [0.0; NA];
        for (a, p) in pr.iter_mut().enumerate() {
            *p = if a == best { 0.80 + 0.10 * r.f64() } else { 0.15 + 0.35 * r.f64() };
        }
        pctx.push(pr);
    }
    let pctx2 = if sp.drift_at > 0 {
        let mut r3 = Rng::new(seed.wrapping_mul(31337).wrapping_add(0xD21F7));
        let mut v = vec![];
        for _ in 0..active.len() {
            let best = r3.below(NA);
            let mut pr = [0.0; NA];
            for (a, p) in pr.iter_mut().enumerate() {
                *p = if a == best { 0.80 + 0.10 * r3.f64() } else { 0.15 + 0.35 * r3.f64() };
            }
            v.push(pr);
        }
        Some(v)
    } else {
        None
    };
    let by_c: Vec<Vec<usize>> = (0..c.nconc).map(|k| (0..c.queries.len()).filter(|&i| c.queries[i].concept == k as i32).collect()).collect();
    let ood: Vec<usize> = (0..c.queries.len()).filter(|&i| c.queries[i].concept < 0).collect();
    let (mut q, mut ctx, mut wt) = (vec![], vec![], vec![]);
    for _ in 0..n {
        if sp.mixed > 0.0 && r.f64() < sp.mixed {
            q.push(ood[r.below(ood.len())]);
            ctx.push(-1);
            wt.push(0.0);
        } else {
            let cid = active[r.below(active.len())];
            let mut qi = by_c[cid][r.below(by_c[cid].len())];
            if sp.corrupt > 0.0 && r2.f64() < sp.corrupt {
                // misleading text: drawn from a random other active concept
                let mut c2 = active[r2.below(active.len())];
                while c2 == cid {
                    c2 = active[r2.below(active.len())];
                }
                qi = by_c[c2][r2.below(by_c[c2].len())];
            }
            q.push(qi);
            ctx.push(cid as i32);
            wt.push(sp.w);
        }
    }
    Stream { w, q, ctx, wt, pctx, pctx2, drift_at: sp.drift_at, row }
}
impl Stream {
    pub fn p(&self, t: usize, a: usize) -> f64 {
        if self.w.is_down(t, a) {
            return 0.0;
        }
        let ty = self.w.task(t).type_idx;
        let base = self.w.skill(t, ty, a);
        let wg = self.wt[t];
        if wg == 0.0 {
            return base;
        }
        let tab = match &self.pctx2 {
            Some(t2) if t >= self.drift_at => t2,
            _ => &self.pctx,
        };
        (1.0 - wg) * base + wg * tab[self.row[self.ctx[t] as usize]][a]
    }
}

// ------------------------------------------------------------------------------------------------
// policies

#[derive(Clone, Debug, PartialEq)]
pub enum Base {
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm,
    Guard,
    Vote,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Gate {
    /// vote whenever the anchor has a candidate with smoothed mean >= .7 (no gate at all)
    Off,
    /// g1: candidate m>=.7 AND global vote-vs-front win-rate window gate (6,.02,25)
    G1,
    /// confidence gate: P(Beta posterior of anchor-best > front_ref + delta) >= gamma and m >= floor
    Post { delta: f64, gamma: f64, floor: f64 },
    /// online logistic regression predicting the success of the voted action from identification / evidence features;
    /// open iff p_hat >= front_ref + delta; closed gate still probes with prob eps (labels)
    Learned { lr: f64, delta: f64, eps: f64 },
}
#[derive(Clone, Debug)]
pub struct PolCfg {
    pub name: String,
    pub base: Base,
    /// front router of the Vote policies (Guard = g1 default; Warm; HebR)
    pub front: Base,
    pub sim: usize,
    pub k: usize,
    pub thr: f32,
    pub min_n: usize,
    pub gate: Gate,
    /// additionally require the g1 global window gate
    pub and_g1: bool,
    /// identification gate: (min agreement of neighbour weight with the winning anchor, min margin winner - runner-up)
    pub id: Option<(f32, f32)>,
    /// inject kn:<anchor> into the front even when the gate is closed (g1 behaviour)
    pub inj_always: bool,
    /// hierarchical shrinkage of the anchor evidence toward the agent's GLOBAL success rate with prior strength kappa (0 = off, raw m>=.7 rule)
    pub kappa: f64,
    /// g1 window gate parameters: min samples per side, margin, base probe interval, exponential probe backoff while the gate stays closed
    pub g1_min: f64,
    pub g1_margin: f64,
    pub probe0: usize,
    pub backoff: bool,
    /// lenient injection test (delta, gamma): inject kn:<anchor> iff the vote is open OR anchor-best passes P(p > front_ref+delta) >= gamma
    pub inj_conf: Option<(f64, f64)>,
    /// adaptive k: keep neighbours with score >= top - drop (k is then kmax)
    pub adapt: Option<f32>,
    /// per-task discount of anchor evidence (1.0 = none)
    pub decay: f64,
    /// reset an agent's anchor evidence after this many consecutive failures of that agent in the anchor (0 = off)
    pub reset_fail: usize,
}
impl PolCfg {
    pub fn base_kind(name: &str, b: Base) -> PolCfg {
        PolCfg { name: name.into(), base: b, front: Base::Guard, sim: 0, k: 5, thr: 0.5, min_n: 2, gate: Gate::G1, and_g1: false, id: None, inj_always: true, kappa: 0.0, g1_min: 6.0, g1_margin: 0.02, probe0: 25, backoff: false, inj_conf: None, adapt: None, decay: 1.0, reset_fail: 0 }
    }
    pub fn vote(name: &str, sim: usize, k: usize, thr: f32, min_n: usize, gate: Gate) -> PolCfg {
        PolCfg { name: name.into(), base: Base::Vote, front: Base::Guard, sim, k, thr, min_n, gate, and_g1: false, id: None, inj_always: true, kappa: 0.0, g1_min: 6.0, g1_margin: 0.02, probe0: 25, backoff: false, inj_conf: None, adapt: None, decay: 1.0, reset_fail: 0 }
    }
}

pub fn make_front(b: &Base, w: &World) -> Box<dyn Router> {
    let kw = w.keyword_map_for();
    let g = GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true };
    match b {
        Base::Keyword => Box::new(KeywordRouter::new(kw, NA)),
        Base::Eps => Box::new(EpsGreedyRouter::new(NA, 0.99, 0.02)),
        Base::Pts => Box::new(PooledThompsonRouter::new(NA, 1.0)),
        Base::HebR => Box::new(HebbianRouter::new(NA, heb_robust_cfg())),
        Base::Warm => Box::new(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, 1.0).with_evidence(Evidence::MeanFeature)),
        _ => Box::new(Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, g)),
    }
}

/// neighbours (score, record index into `past`), best first. Optional adaptive-k cut.
pub fn neighbors(sim: &Sim, p: &Pool, qi: usize, past: &[usize], k: usize, thr: f32, adapt: Option<f32>) -> Vec<(f32, usize)> {
    if sim.oracle {
        let c = p.concept[qi];
        if c < 0 {
            return vec![];
        }
        return past.iter().enumerate().rev().filter(|&(_, &m)| p.concept[m] == c).map(|(i, _)| (1.0, i)).take(k).collect();
    }
    let mut v: Vec<(f32, usize)> = past.iter().enumerate().map(|(i, &m)| (sim.s(p, qi, m), i)).filter(|x| x.0 >= thr).collect();
    if v.len() > k {
        v.select_nth_unstable_by(k, |a, b| b.0.partial_cmp(&a.0).unwrap());
        v.truncate(k);
    }
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    if let (Some(d), Some(top)) = (adapt, v.first().map(|x| x.0)) {
        v.retain(|x| x.0 >= top - d);
    }
    v
}

#[derive(Clone)]
struct AStat {
    s: [f64; NA],
    n: [f64; NA],
    cf: [u32; NA],
    t: usize,
}
impl AStat {
    fn new(t: usize) -> Self {
        AStat { s: [0.0; NA], n: [0.0; NA], cf: [0; NA], t }
    }
    fn decay_to(&mut self, t: usize, lam: f64) {
        if lam < 1.0 && t > self.t {
            let f = lam.powi((t - self.t) as i32);
            for a in 0..NA {
                self.s[a] *= f;
                self.n[a] *= f;
            }
        }
        self.t = t;
    }
}

#[derive(Clone, Default)]
pub struct RunOut {
    pub h: [f64; 4],
    /// mean expected success over tasks 1000..2000
    pub late: f64,
    pub pe: Vec<f32>,
    /// share of tasks where the vote action was played
    pub vote_frac: f64,
    /// share of tasks with a candidate (anchor-best passing the base criterion)
    pub cand_frac: f64,
    pub elapsed_us_per_task: f64,
    /// diagnostics: mean expected success of vote-played tasks / front-played tasks (NaN if none)
    pub pe_vote: f64,
    pub pe_front: f64,
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn run(cfg: &PolCfg, sp: &Spec, seed: u64, c: &Corpus, p: &Pool, ss: &[Sim]) -> RunOut {
    let t_start = std::time::Instant::now();
    let st = stream(c, seed, sp);
    let w = &st.w;
    let cands: Vec<usize> = (0..NA).collect();
    let mut fr = make_front(if cfg.base == Base::Vote { &cfg.front } else { &cfg.base }, w);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut rng2 = Rng::new(seed.wrapping_mul(2_654_435_761).wrapping_add(99));
    let mut out = RunOut { pe: Vec::with_capacity(w.len()), ..RunOut::default() };
    let mut cum = 0.0;
    let mut past: Vec<usize> = vec![];
    let mut anchor_of: Vec<u64> = vec![];
    let mut astats: HashMap<u64, AStat> = HashMap::new();
    let mut gate = [0.0f64; 4];
    let mut probe = 0usize;
    let mut pint = cfg.probe0;
    let is_vote = cfg.base == Base::Vote;
    let sim = &ss[cfg.sim];
    // front reference (EMA of outcomes of tasks decided by the front)
    let (mut fref, mut fref_n) = (0.5f64, 0.0f64);
    // learned gate weights: [bias, top1, margin, agree, ln(1+n)/3, post_mean]
    let mut lw = [0.0f64; 6];
    let (mut n_vote, mut n_cand) = (0usize, 0usize);
    let (mut dv, mut df) = (0.0f64, 0.0f64);
    let (mut gs, mut gn) = ([0.0f64; NA], [0.0f64; NA]);
    for t in 0..w.len() {
        let task0 = &w.task(t).task;
        let mut task = task0.clone();
        let mut by_vote = false;
        let mut probe_play = false;
        let mut anchor = t as u64;
        let mut feats = [0.0f64; 6];
        let a = if is_vote {
            let nb = neighbors(sim, p, st.q[t], &past, cfg.k, cfg.thr, cfg.adapt);
            let mut votes: Vec<(u64, f32)> = vec![];
            let mut tot = 0.0f32;
            for &(sc, i) in &nb {
                tot += sc;
                match votes.iter_mut().find(|v| v.0 == anchor_of[i]) {
                    Some(v) => v.1 += sc,
                    None => votes.push((anchor_of[i], sc)),
                }
            }
            let found = votes.iter().fold(None, |b: Option<(u64, f32)>, &v| match b { Some(x) if x.1 >= v.1 => b, _ => Some(v) });
            let (mut agree, mut margin) = (0.0f32, 0.0f32);
            if let Some((_, w1)) = found {
                let wa = found.unwrap().0;
                let w2 = votes.iter().filter(|v| v.0 != wa).map(|v| v.1).fold(0.0f32, f32::max);
                agree = if tot > 0.0 { w1 / tot } else { 0.0 };
                margin = if tot > 0.0 { (w1 - w2) / tot } else { 0.0 };
            }
            let top1 = nb.first().map(|x| x.0).unwrap_or(0.0);
            if let Some((an, _)) = found {
                anchor = an;
            }
            // candidate agent from the (decayed) anchor evidence
            let mut best: Option<(usize, f64, f64, f64)> = None; // (agent, post mean, s, n)
            if let Some((an, _)) = found {
                if let Some(row) = astats.get_mut(&an) {
                    row.decay_to(t, cfg.decay);
                    for a in 0..NA {
                        let (s, n) = (row.s[a], row.n[a]);
                        if n < cfg.min_n as f64 - 1e-9 {
                            continue;
                        }
                        if cfg.kappa > 0.0 {
                            let g = (gs[a] + 1.0) / (gn[a] + 2.0);
                            let (al, be) = (cfg.kappa * g + s, cfg.kappa * (1.0 - g) + (n - s));
                            let m = al / (al + be);
                            if best.is_none_or(|b| m > b.1) {
                                best = Some((a, m, al - 1.0, al + be - 2.0));
                            }
                        } else {
                            let m = (1.0 + s) / (2.0 + n);
                            if m >= 0.7 && best.is_none_or(|b| m > b.1) {
                                best = Some((a, m, s, n));
                            }
                        }
                    }
                }
            }
            if best.is_some() {
                n_cand += 1;
            }
            // identification gate
            let id_ok = match cfg.id {
                Some((ta, tm)) => agree >= ta && margin >= tm,
                None => true,
            };
            let g1_open = best.is_none() || {
                let rate_ok = gate[1] < cfg.g1_min || gate[3] < cfg.g1_min || gate[0] / gate[1] >= gate[2] / gate[3] - cfg.g1_margin;
                if rate_ok {
                    pint = cfg.probe0;
                    true
                } else {
                    probe += 1;
                    if probe >= pint {
                        probe = 0;
                        if cfg.backoff {
                            pint = (pint * 2).min(800);
                        }
                        true
                    } else {
                        false
                    }
                }
            };
            let mut open = false;
            if let Some((_, m, s, n)) = best {
                open = id_ok
                    && match &cfg.gate {
                        Gate::Off => true,
                        Gate::G1 => g1_open,
                        Gate::Post { delta, gamma, floor } => {
                            let conf = 1.0 - inc_beta(1.0 + s, 1.0 + (n - s), (fref + delta).clamp(1e-6, 1.0 - 1e-6));
                            conf >= *gamma && m >= *floor
                        }
                        Gate::Learned { delta, eps, .. } => {
                            feats = [1.0, top1 as f64, margin as f64, agree as f64, (1.0 + n).ln() / 3.0, m];
                            let ph = sigmoid(lw.iter().zip(&feats).map(|(a, b)| a * b).sum());
                            if ph >= fref + delta {
                                true
                            } else if rng2.f64() < *eps {
                                probe_play = true;
                                true
                            } else {
                                false
                            }
                        }
                    };
                if open && cfg.and_g1 && !matches!(cfg.gate, Gate::G1) {
                    open = g1_open;
                }
            }
            let _ = probe_play;
            // feature injection
            let inj_ok = open
                || cfg.inj_always
                || match (cfg.inj_conf, best) {
                    (Some((di, gi)), Some((_, _, s, n))) => 1.0 - inc_beta(1.0 + s, 1.0 + (n - s), (fref + di).clamp(1e-6, 1.0 - 1e-6)) >= gi,
                    _ => false,
                };
            if found.is_some() && inj_ok {
                task.features.push(format!("kn:{anchor}"));
            }
            match best {
                Some((a, ..)) if open => {
                    by_vote = true;
                    n_vote += 1;
                    a
                }
                _ => fr.select(&task, &cands, &mut rng),
            }
        } else {
            fr.select(&task, &cands, &mut rng)
        };
        let pe = st.p(t, a);
        let s = if w.task(t).u < pe { 1.0 } else { 0.0 };
        fr.update(&task, a, s);
        gs[a] += s;
        gn[a] += 1.0;
        if is_vote {
            past.push(st.q[t]);
            anchor_of.push(anchor);
            let row = astats.entry(anchor).or_insert_with(|| AStat::new(t));
            row.decay_to(t, cfg.decay);
            row.s[a] += s;
            row.n[a] += 1.0;
            if s > 0.5 {
                row.cf[a] = 0;
            } else {
                row.cf[a] += 1;
                if cfg.reset_fail > 0 && row.cf[a] as usize >= cfg.reset_fail {
                    row.s[a] = 0.0;
                    row.n[a] = 0.0;
                    row.cf[a] = 0;
                }
            }
            if by_vote {
                gate[0] += s;
                gate[1] += 1.0;
                if let Gate::Learned { lr, .. } = &cfg.gate {
                    // train on the observed outcome of the voted action
                    let ph = sigmoid(lw.iter().zip(&feats).map(|(a, b)| a * b).sum());
                    for i in 0..6 {
                        lw[i] += lr * (s - ph) * feats[i];
                    }
                }
            } else {
                gate[2] += s;
                gate[3] += 1.0;
                fref_n += 1.0;
                let al = (1.0 / (fref_n + 1.0)).max(0.03);
                fref += al * (s - fref);
            }
        }
        if by_vote { dv += pe; } else { df += pe; }
        out.pe.push(pe as f32);
        cum += pe;
        for (i, &hz) in HORIZONS.iter().enumerate() {
            if t + 1 == hz {
                out.h[i] = cum / hz as f64;
            }
        }
    }
    out.late = out.pe[1000..].iter().map(|&x| x as f64).sum::<f64>() / (NT - 1000) as f64;
    out.pe_vote = if n_vote > 0 { dv / n_vote as f64 } else { f64::NAN };
    out.pe_front = df / (NT - n_vote) as f64;
    out.vote_frac = n_vote as f64 / NT as f64;
    out.cand_frac = n_cand as f64 / NT as f64;
    out.elapsed_us_per_task = t_start.elapsed().as_secs_f64() * 1e6 / NT as f64;
    out
}

pub fn eval(cfg: &PolCfg, sp: &Spec, seeds: &[u64], c: &Corpus, p: &Pool, ss: &[Sim]) -> Vec<RunOut> {
    seeds.par_iter().map(|&s| run(cfg, sp, s, c, p, ss)).collect()
}

// ------------------------------------------------------------------------------------------------
// metrics / stats

/// metric ids: 0..3 = H100,H250,H500,H2000 prefix means; 4 = late (1000-2000); 5 = tasks-to-target (needs target)
pub fn metric(r: &RunOut, m: usize) -> f64 {
    if m < 4 { r.h[m] } else { r.late }
}
pub const METRIC_NAMES: [&str; 5] = ["H100", "H250", "H500", "H2000", "late"];
pub fn col(r: &[RunOut], m: usize) -> Vec<f64> {
    r.iter().map(|x| metric(x, m)).collect()
}
pub fn mean_of(r: &[RunOut], m: usize) -> f64 {
    summarize(&col(r, m)).mean
}
/// first t (>=200) at which the trailing-200 mean of expected success reaches `target`; NT+1 = never
pub fn tasks_to_target(r: &RunOut, target: f64) -> f64 {
    let win = 200usize;
    let mut s: f64 = r.pe[..win].iter().map(|&x| x as f64).sum();
    if s / win as f64 >= target {
        return win as f64;
    }
    for t in win..r.pe.len() {
        s += r.pe[t] as f64 - r.pe[t - win] as f64;
        if s / win as f64 >= target {
            return (t + 1) as f64;
        }
    }
    (NT + 1) as f64
}

#[derive(Clone, Debug)]
pub struct Cmp {
    pub diff: f64,
    pub lo: f64,
    pub hi: f64,
    pub wins: usize,
    pub losses: usize,
    pub p: f64,
    /// TOST p for equivalence within +-eq (two one-sided t tests); NaN if eq == 0
    pub tost_p: f64,
}
pub fn cmp(a: &[f64], b: &[f64], eq: f64) -> Cmp {
    let d = paired_diff(a, b);
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let n = diffs.len();
    let (m, s) = (summarize(&diffs).mean, summarize(&diffs).sd);
    let tost = if eq > 0.0 && n >= 2 && s > 1e-15 {
        let se = s / (n as f64).sqrt();
        // H0a: mean <= -eq  -> t = (m + eq)/se one-sided ; H0b: mean >= eq -> t = (eq - m)/se
        let pa = one_sided_p((m + eq) / se, n - 1);
        let pb = one_sided_p((eq - m) / se, n - 1);
        pa.max(pb)
    } else if eq > 0.0 && s <= 1e-15 {
        if m.abs() < eq { 0.0 } else { 1.0 }
    } else {
        f64::NAN
    };
    Cmp { diff: d.diff.mean, lo: d.diff.lo, hi: d.diff.hi, wins: d.wins, losses: d.losses, p: paired_t_pvalue(a, b), tost_p: tost }
}
/// P(T > t) for Student t with df
pub fn one_sided_p(t: f64, df: usize) -> f64 {
    let p2 = t_two_sided_p(t, df);
    if t >= 0.0 { p2 / 2.0 } else { 1.0 - p2 / 2.0 }
}
pub fn holm(ps: &[f64]) -> Vec<f64> {
    holm_adjust(ps)
}
pub fn mark(adj: f64, d: f64) -> &'static str {
    if adj < 0.05 { if d > 0.0 { "WIN" } else { "LOSS" } } else { "ns" }
}
