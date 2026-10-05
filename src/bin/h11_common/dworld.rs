//! h11 world D: a document-QA research session built on the g16/g1 fixtures (real MiniLM+tf-idf query similarity matrices, real hybrid doc rankings,
//! the g16 recurring-concept task stream: sib16/rand16/mixed50/weakid50/drift/ctl-irrel/sib64/ctl-oneoff + a duplicates-in-the-store world) with
//! agent PRICES, a token-level resource model and the cost levers: context size k (real rank-relevance profile), score gate, learned per-anchor context
//! bandit, exact-duplicate suppression, verification gate, result cache (exact / semantic, store-all / store-verified, reverify-on-hit, evict-on-bad-grade),
//! cost-aware anchor routing, and hindsight oracles.
//!
//! SYNTHETIC parts (all declared before any policy ran): agent caps/prices, concept difficulty, the context-utility model
//!   fit = (1 - d (1 - U(R))) (1 - eta I),   U(R) = 1 - (1-rho)^R   (diminishing returns: rho = .7 -> 1 relevant doc recovers 70% of the context gap),
//!   R = distinct relevant docs injected, I = irrelevant docs injected (each dilutes by eta = .03), p(success) = base(agent, concept) * fit,
//! the verifier flag model (false flag .08, recall .75) and token counts. REAL parts: which docs the retrieval returns, their scores and doc lengths,
//! which query texts recur, and which paraphrases/siblings the similarity matrix confuses. Not an LLM measurement.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use neural_mesh::router::Rng;
use rayon::prelude::*;

use crate::common::*;
use crate::h11::pworld::{norm, uni};
use crate::h11::*;

pub const NT: usize = 2000;
pub const NA: usize = 6;
pub const AG_PRICE: [f64; NA] = [0.25, 0.25, 1.0, 1.0, 3.0, 3.0];
pub const AG_CAP: [f64; NA] = [0.50, 0.50, 0.68, 0.68, 0.86, 0.86];
pub const B_D: f64 = 300.0;
pub const O_D: f64 = 250.0;
pub const VB_D: f64 = 250.0;
pub const V_OUT_D: f64 = 40.0;
/// reference call = one mid-price (pi=1) generator call, no context: 300 + 4*250
pub const S_REF_D: f64 = 1300.0;
pub const FALSE_FLAG: f64 = 0.08;
pub const RECALL: f64 = 0.75;
pub const RHO: f64 = 0.7;
pub const ETA: f64 = 0.03;
pub const REVEAL: f64 = 0.6;
pub const DELAY: usize = 15;
pub const EMB_MS: f64 = 2.6;
pub const SEARCH_MS: f64 = 0.05;
pub const ANCHOR_K: usize = 8;
pub const ANCHOR_THR: f32 = 0.75;
pub const KAPPA: f64 = 3.0;
pub const TOPK: usize = 8;

// ------------------------------------------------------------------------------------------------ data
#[derive(Clone, Copy, Debug)]
pub struct RDoc {
    /// doc id; >= nd = a re-ingested duplicate of dup_src[id - nd]
    pub id: usize,
    pub z: f32,
}
pub struct DData {
    pub tag: String,
    pub nq: usize,
    pub nd: usize,
    /// query x query similarity: .6 MiniLM(spell) + .4 tf-idf(spell)  (the g16 FINAL similarity)
    pub sim: Vec<f32>,
    pub qconcept: Vec<i32>,
    pub nconc: usize,
    pub topic_of_concept: Vec<i32>,
    pub by_c: Vec<Vec<usize>>,
    pub ood: Vec<usize>,
    pub doc_concept: Vec<i32>,
    /// raw tokens ~ chars/4
    pub doc_tok: Vec<f64>,
    pub ranks: Vec<Vec<RDoc>>,
    pub ranks_dup: Vec<Vec<RDoc>>,
    pub dup_src: Vec<usize>,
}
fn hash01(x: usize) -> u64 {
    (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 33
}
impl DData {
    pub fn build(ctx: &Ctx) -> DData {
        let c = &ctx.c;
        let (nq, nd) = (c.queries.len(), c.docs.len());
        let mini = &ctx.models["mini"];
        let tf: Vec<Vec<(u32, f32)>> = c.queries.iter().map(|q| ctx.lex.tfidf(&q.spell)).collect();
        let mut sim = vec![0.0f32; nq * nq];
        sim.par_chunks_mut(nq).enumerate().for_each(|(i, row)| {
            let a = mini.qvec(QV::Spell, i);
            for (j, r) in row.iter_mut().enumerate() {
                *r = 0.6 * dot(a, mini.qvec(QV::Spell, j)) + 0.4 * sparse_dot(&tf[i], &tf[j]);
            }
        });
        let mut cfg = Cfg::base("mini-hybrid", "mini");
        cfg.qv = QV::Spell;
        cfg.lexical = true;
        cfg.alpha = 0.7;
        let dup_src: Vec<usize> = (0..nd).filter(|&d| hash01(d + 5) % 10 < 3).collect();
        let ranks: Vec<Vec<RDoc>> = (0..nq)
            .into_par_iter()
            .map(|qi| {
                let s = ctx.score(&cfg, qi);
                let mut idx: Vec<usize> = (0..nd).collect();
                idx.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap().then(a.cmp(&b)));
                idx.truncate(TOPK);
                idx.into_iter().map(|j| RDoc { id: j, z: s[j] }).collect()
            })
            .collect();
        let ranks_dup: Vec<Vec<RDoc>> = ranks
            .iter()
            .map(|r| {
                let mut out = vec![];
                for d in r {
                    out.push(*d);
                    for (j, &src) in dup_src.iter().enumerate() {
                        if src == d.id {
                            out.push(RDoc { id: nd + j, z: d.z });
                        }
                    }
                }
                out
            })
            .collect();
        let by_c: Vec<Vec<usize>> = (0..c.nconc).map(|k| (0..nq).filter(|&i| c.queries[i].concept == k as i32).collect()).collect();
        let ood: Vec<usize> = (0..nq).filter(|&i| c.queries[i].concept < 0).collect();
        DData {
            tag: c.tag.clone(),
            nq,
            nd,
            sim,
            qconcept: c.queries.iter().map(|q| q.concept).collect(),
            nconc: c.nconc,
            topic_of_concept: c.topic_of_concept.clone(),
            by_c,
            ood,
            doc_concept: c.docs.iter().map(|d| d.concept).collect(),
            doc_tok: c.docs.iter().map(|d| d.text.len() as f64 / 4.0).collect(),
            ranks,
            ranks_dup,
            dup_src,
        }
    }
    pub fn base_of(&self, id: usize) -> usize {
        if id >= self.nd { self.dup_src[id - self.nd] } else { id }
    }
}

// ------------------------------------------------------------------------------------------------ worlds (fixed before any policy ran)
#[derive(Clone, Debug)]
pub struct DSpec {
    pub name: &'static str,
    pub sib: bool,
    pub nt: usize,
    /// context dependence of the answer (0 = control: context useless)
    pub d: f64,
    pub mixed: f64,
    pub corrupt: f64,
    pub drift_at: usize,
    pub dups: bool,
    pub tune: bool,
    pub treat: bool,
    /// every query TEXT used at most once (drawn without replacement from all concept queries): concepts recur, exact texts do not (control for the exact-text cache)
    pub uniq: bool,
}
pub fn dworlds() -> Vec<DSpec> {
    let s = |name, sib, nt, d, mixed, corrupt, drift_at, dups, tune, treat| DSpec { name, sib, nt, d, mixed, corrupt, drift_at, dups, tune, treat, uniq: false };
    vec![
        s("sib16", true, 2, 0.35, 0.0, 0.0, 0, false, true, true),
        s("rand16", false, 2, 0.35, 0.0, 0.0, 0, false, true, true),
        s("sib16-mixed50", true, 2, 0.35, 0.5, 0.0, 0, false, true, true),
        s("sib16-weakid50", true, 2, 0.35, 0.0, 0.5, 0, false, true, true),
        s("sib16-drift", true, 2, 0.35, 0.0, 0.0, 1000, false, true, true),
        s("ctl-none", true, 2, 0.0, 0.0, 0.0, 0, false, true, false),
        s("sib64", true, 8, 0.35, 0.0, 0.0, 0, false, false, true),
        s("ctl-oneoff", true, 2, 0.35, 1.0, 0.0, 0, false, false, false),
        s("sib16-dups30", true, 2, 0.35, 0.0, 0.0, 0, true, false, true),
        DSpec { name: "ctl-uniq", uniq: true, ..s("ctl-uniq", true, 2, 0.35, 0.0, 0.0, 0, false, false, false) },
    ]
}

pub struct DStream {
    pub q: Vec<usize>,
    /// the concept that decides the answer (-1 = off-topic one-off)
    pub cid: Vec<i32>,
}
pub fn dstream(dd: &DData, seed: u64, sp: &DSpec) -> DStream {
    let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xD0C5_0011));
    let mut r2 = Rng::new(seed.wrapping_mul(104729).wrapping_add(0xD0C5_0022));
    let ntopics = (*dd.topic_of_concept.iter().max().unwrap() + 1) as usize;
    let mut active: Vec<usize> = vec![];
    if sp.sib {
        let mut topics: Vec<usize> = (0..ntopics).collect();
        for i in (1..topics.len()).rev() {
            let j = r.below(i + 1);
            topics.swap(i, j);
        }
        for &t in topics.iter().take(sp.nt) {
            active.extend((0..dd.nconc).filter(|&k| dd.topic_of_concept[k] as usize == t));
        }
    } else {
        let mut all: Vec<usize> = (0..dd.nconc).collect();
        for i in (1..all.len()).rev() {
            let j = r.below(i + 1);
            all.swap(i, j);
        }
        active = all.into_iter().take(sp.nt * 8).collect();
    }
    let (mut q, mut cid) = (vec![], vec![]);
    if sp.uniq {
        let mut all: Vec<usize> = (0..dd.nq).filter(|&i| dd.qconcept[i] >= 0).collect();
        for i in (1..all.len()).rev() {
            let j = r.below(i + 1);
            all.swap(i, j);
        }
        for t in 0..NT {
            let qi = all[t % all.len()];
            q.push(qi);
            cid.push(dd.qconcept[qi]);
        }
        return DStream { q, cid };
    }
    for _ in 0..NT {
        if sp.mixed > 0.0 && r.f64() < sp.mixed {
            q.push(dd.ood[r.below(dd.ood.len())]);
            cid.push(-1);
        } else {
            let c = active[r.below(active.len())];
            let mut qi = dd.by_c[c][r.below(dd.by_c[c].len())];
            if sp.corrupt > 0.0 && r2.f64() < sp.corrupt {
                let mut c2 = active[r2.below(active.len())];
                while c2 == c {
                    c2 = active[r2.below(active.len())];
                }
                qi = dd.by_c[c2][r2.below(dd.by_c[c2].len())];
            }
            q.push(qi);
            cid.push(c as i32);
        }
    }
    DStream { q, cid }
}

/// base success probability of agent `a` on the question decided by `cid` (concept difficulty shared across agents + agent x concept idiosyncrasy);
/// off-topic one-offs: per-text difficulty, slightly harder. Tables are re-drawn at the drift point (epoch 1).
fn base_p(seed: u64, a: usize, cid: i32, qi: usize, epoch: usize) -> f64 {
    let v = if cid >= 0 {
        AG_CAP[a] + 0.12 * norm(seed, 0xB1, cid as u64, epoch as u64) + 0.08 * norm(seed, 0xB2, (cid as usize * 8 + a) as u64, epoch as u64)
    } else {
        AG_CAP[a] - 0.05 + 0.12 * norm(seed, 0xB3, qi as u64, 0) + 0.08 * norm(seed, 0xB4, (qi * 8 + a) as u64, 0)
    };
    v.clamp(0.05, 0.97)
}

// ------------------------------------------------------------------------------------------------ policies
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Route {
    /// price-blind, text-blind: global success rate with eps-greedy (the "one best model for everything" baseline)
    Global,
    /// anchor-conditioned agent evidence (g16 vote anchors + shrinkage to the agent's global rate); score = m + explore - mu * price * call-size
    Anchor { mu: f64 },
    OracleQ,
    OracleUtil { mu: f64 },
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CtxPol {
    /// always inject the top-k docs (k = 0: no context)
    K(usize),
    /// inject the top-k docs iff the top-1 margin (z1 - z2 of distinct docs) >= zmin (abstain otherwise)
    Gate { k: usize, zmin: f32 },
    /// per-anchor learned value of injecting (vs not), optimistic start, exploration eps; inject iff est. gain > mu * context cost
    Bandit { k: usize, eps: f64 },
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DV {
    Never,
    Always,
    /// verify iff the anchor posterior m of the chosen agent says (1-m) * rv >= cv_eff (break-even in this call's verifier price relative to a reference call)
    /// or the anchor has fewer than nmin graded samples for that agent
    Gate { rv: f64, nmin: f64 },
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemoKind {
    Off,
    Exact,
    Sem { tau: f32 },
}
#[derive(Clone, Copy, Debug)]
pub struct DCfg {
    pub route: Route,
    pub ctx: CtxPol,
    pub dedup: bool,
    pub vmode: DV,
    pub memo: MemoKind,
    /// store only answers that a verifier passed
    pub store_verified: bool,
    /// verify a cached answer before serving it (flag -> regenerate)
    pub reverify: bool,
    /// drop a cached answer when its delayed grade says it was wrong
    pub evict: bool,
    pub max_retry: usize,
}
impl DCfg {
    /// the default research-session pipeline: anchor router (price-blind), top-3 context, always verify with one retry, no cache
    pub fn d0() -> DCfg {
        DCfg { route: Route::Anchor { mu: 0.0 }, ctx: CtxPol::K(3), dedup: false, vmode: DV::Always, memo: MemoKind::Off, store_verified: true, reverify: false, evict: true, max_retry: 1 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DSum {
    pub succ: f64,
    pub succ250: f64,
    pub res: Res,
    /// share of tasks answered from the cache; share of tasks answered with a WRONG cached answer
    pub hit: f64,
    pub wrong_hit: f64,
    /// mean docs injected per generated task; share of generated tasks with any context injected
    pub k_inj: f64,
    pub inj_rate: f64,
    pub gen_rate: f64,
}
pub fn util_d(s: &DSum, p: &Prices, mu: f64) -> f64 {
    s.succ - mu * (cost_te(&s.res, p) / S_REF_D - 1.0)
}

#[derive(Clone, Copy, Default)]
struct AS {
    s: [f64; NA],
    n: [f64; NA],
}
#[derive(Clone, Copy, Default)]
struct CB {
    si: f64,
    ni: f64,
    sn: f64,
    nn: f64,
}
#[derive(Clone, Copy)]
struct Entry {
    correct: bool,
    concept: i32,
    qi: usize,
    epoch: usize,
}
struct Ev {
    due: usize,
    att: Vec<(Option<usize>, usize, bool)>,
    cb: Option<(Option<usize>, bool, bool)>,
    evict_key: Option<usize>,
    delivered: bool,
}
pub struct CtxInfo {
    tok: f64,
    r: usize,
    irr: usize,
    n_docs: usize,
}
pub fn build_ctx(dd: &DData, sp: &DSpec, qi: usize, cid: i32, k: usize, dedup: bool, zmin: Option<f32>) -> CtxInfo {
    let list = if sp.dups { &dd.ranks_dup[qi] } else { &dd.ranks[qi] };
    // abstain gate on the top-1 margin: fused z of the top doc minus the first doc with a DIFFERENT base id (g16 E4's margin gate; top-1 z alone does not separate
    // concept queries from off-topic ones in this fixture, see h11_doc --diag)
    let margin = |l: &Vec<RDoc>| -> f32 {
        let b0 = dd.base_of(l[0].id);
        l.iter().skip(1).find(|d| dd.base_of(d.id) != b0).map(|d| l[0].z - d.z).unwrap_or(0.0)
    };
    if k == 0 || list.is_empty() || zmin.is_some_and(|z| margin(list) < z) {
        return CtxInfo { tok: 0.0, r: 0, irr: 0, n_docs: 0 };
    }
    let (mut tok, mut irr, mut n) = (0.0, 0usize, 0usize);
    let mut seen_base: HashSet<usize> = HashSet::new();
    let mut rel: HashSet<usize> = HashSet::new();
    for d in list {
        if n >= k {
            break;
        }
        let b = dd.base_of(d.id);
        if dedup && !seen_base.insert(b) {
            continue;
        }
        n += 1;
        tok += dd.doc_tok[b];
        if dd.doc_concept[b] == cid && cid >= 0 {
            rel.insert(b);
        } else if cid < 0 || dd.doc_concept[b] != cid {
            irr += 1;
        }
    }
    CtxInfo { tok, r: rel.len(), irr, n_docs: n }
}
pub fn p_succ(base: f64, d: f64, cid: i32, r: usize, irr: usize) -> f64 {
    let de = if cid >= 0 { d } else { 0.0 };
    let u = 1.0 - (1.0 - RHO).powi(r as i32);
    (base * (1.0 - de * (1.0 - u)) * (1.0 - ETA * irr as f64)).clamp(0.0, 1.0)
}

/// shrunk mean success of agent a at the anchor (prior = the agent's global rate, strength KAPPA) and its anchor-level count
fn m_at(an: Option<usize>, astats: &HashMap<usize, AS>, gs: &[f64; NA], gn: &[f64; NA], a: usize) -> (f64, f64) {
    let g = (gs[a] + 1.0) / (gn[a] + 2.0);
    let (s, n) = an.and_then(|x| astats.get(&x)).map(|r| (r.s[a], r.n[a])).unwrap_or((0.0, 0.0));
    ((KAPPA * g + s) / (KAPPA + n), n)
}

/// the document-QA session: memo -> retrieve -> route -> generate -> verify/retry -> deliver -> memo store; delayed graded feedback
pub fn run_d(c: &DCfg, sp: &DSpec, seed: u64, dd: &DData) -> DSum {
    let st = dstream(dd, seed, sp);
    let nq = dd.nq;
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut res_sum = Res::default();
    let (mut succ, mut succ250, mut hits, mut wrong_hits) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut k_inj_sum, mut n_inj_task, mut n_gen_task) = (0.0f64, 0.0f64, 0.0f64);
    let mut past: Vec<usize> = vec![];
    let mut anchor_of: Vec<usize> = vec![];
    let mut astats: HashMap<usize, AS> = HashMap::new();
    let (mut gs, mut gn) = ([0.0f64; NA], [0.0f64; NA]);
    let mut cb_a: HashMap<usize, CB> = HashMap::new();
    let mut cb_g = CB::default();
    let mut entries: HashMap<usize, Entry> = HashMap::new();
    let mut stored: Vec<usize> = vec![];
    let mut seen_text: HashSet<usize> = HashSet::new();
    let mut evs: std::collections::VecDeque<Ev> = std::collections::VecDeque::new();
    let ctxs = Prices::base_d().ctx_scale;
    for t in 0..NT {
        // ---- delayed feedback
        while evs.front().is_some_and(|e| e.due <= t) {
            let e = evs.pop_front().unwrap();
            for (an, a, s) in e.att {
                gs[a] += f64::from(u8::from(s));
                gn[a] += 1.0;
                if let Some(x) = an {
                    let r = astats.entry(x).or_default();
                    r.s[a] += f64::from(u8::from(s));
                    r.n[a] += 1.0;
                }
            }
            if let Some((an, inj, ok)) = e.cb {
                let o = f64::from(u8::from(ok));
                let upd = |b: &mut CB| {
                    if inj {
                        b.si += o;
                        b.ni += 1.0;
                    } else {
                        b.sn += o;
                        b.nn += 1.0;
                    }
                };
                upd(&mut cb_g);
                if let Some(x) = an {
                    upd(cb_a.entry(x).or_default());
                }
            }
            if c.evict && !e.delivered {
                if let Some(k) = e.evict_key {
                    entries.remove(&k);
                }
            }
        }
        let qi = st.q[t];
        let cid = st.cid[t];
        let epoch = usize::from(sp.drift_at > 0 && t >= sp.drift_at);
        let mut res = Res::default();
        let mut embedded = false;
        let mut delivered: Option<bool> = None;
        let mut evict_key: Option<usize> = None;
        // ---- result cache
        if c.memo != MemoKind::Off {
            let mut hit: Option<usize> = entries.contains_key(&qi).then_some(qi);
            if hit.is_none() {
                if let MemoKind::Sem { tau } = c.memo {
                    res.ms += EMB_MS + SEARCH_MS;
                    res.ms_cached += EMB_MS * f64::from(u8::from(seen_text.insert(qi))) + SEARCH_MS;
                    embedded = true;
                    let mut best = (tau, usize::MAX);
                    for &kq in &stored {
                        if entries.contains_key(&kq) {
                            let s = dd.sim[qi * nq + kq];
                            if s >= best.0 {
                                best = (s, kq);
                            }
                        }
                    }
                    if best.1 != usize::MAX {
                        hit = Some(best.1);
                    }
                }
            } else {
                res.ms += 0.002;
                res.ms_cached += 0.002;
            }
            if let Some(kq) = hit {
                let e = entries[&kq];
                let same_q = if cid >= 0 { e.concept == cid } else { e.qi == qi };
                let correct_now = e.correct && same_q && e.epoch == epoch;
                let mut serve = true;
                if c.reverify {
                    // evidence for the check: the policy's context (retrieval + tokens), then one verifier call
                    let ci = build_ctx(dd, sp, qi, cid, match c.ctx { CtxPol::K(k) | CtxPol::Gate { k, .. } | CtxPol::Bandit { k, .. } => k }, c.dedup, None);
                    res.ms += SEARCH_MS;
                    res.ms_cached += SEARCH_MS;
                    res.g_ctx += ci.tok;
                    res.v_in += VB_D;
                    res.v_out += V_OUT_D;
                    res.n_ver += 1.0;
                    let u = uni(seed, 0xD2, t as u64, 9);
                    let flagged = if correct_now { u < FALSE_FLAG } else { u < RECALL };
                    if flagged {
                        serve = false;
                        entries.remove(&kq);
                    }
                }
                if serve {
                    delivered = Some(correct_now);
                    evict_key = Some(kq);
                    res.n_hit += 1.0;
                    hits += 1.0;
                    if !correct_now {
                        wrong_hits += 1.0;
                    }
                }
            }
        }
        // ---- generation path
        let mut an_for_ev: Option<usize> = None;
        let mut att_log: Vec<(Option<usize>, usize, bool)> = vec![];
        let mut cb_ev: Option<(Option<usize>, bool, bool)> = None;
        if delivered.is_none() {
            n_gen_task += 1.0;
            if !embedded {
                res.ms += EMB_MS + SEARCH_MS;
                res.ms_cached += EMB_MS * f64::from(u8::from(seen_text.insert(qi))) + SEARCH_MS;
            }
            // anchor (g16 vote anchor over past generated queries)
            let mut votes: Vec<(usize, f32)> = vec![];
            {
                let mut v: Vec<(f32, usize)> = past.iter().enumerate().map(|(i, &m)| (dd.sim[qi * nq + m], i)).filter(|x| x.0 >= ANCHOR_THR).collect();
                if v.len() > ANCHOR_K {
                    v.select_nth_unstable_by(ANCHOR_K, |a, b| b.0.partial_cmp(&a.0).unwrap());
                    v.truncate(ANCHOR_K);
                }
                for &(sc, i) in &v {
                    match votes.iter_mut().find(|x| x.0 == anchor_of[i]) {
                        Some(x) => x.1 += sc,
                        None => votes.push((anchor_of[i], sc)),
                    }
                }
            }
            let found = votes.iter().fold(None, |b: Option<(usize, f32)>, &v| match b { Some(x) if x.1 >= v.1 => b, _ => Some(v) });
            let an = found.map(|x| x.0);
            an_for_ev = an;
            // context decision
            let ctx_k = match c.ctx { CtxPol::K(k) | CtxPol::Gate { k, .. } | CtxPol::Bandit { k, .. } => k };
            let zmin = if let CtxPol::Gate { zmin, .. } = c.ctx { Some(zmin) } else { None };
            let full = build_ctx(dd, sp, qi, cid, ctx_k, c.dedup, zmin);
            res.ms += SEARCH_MS * f64::from(u8::from(ctx_k > 0));
            res.ms_cached += SEARCH_MS * f64::from(u8::from(ctx_k > 0));
            let inject = match c.ctx {
                CtxPol::Bandit { eps, .. } => {
                    let w0 = 2.0;
                    let gi = (cb_g.si + 0.7 * w0) / (cb_g.ni + w0);
                    let gnn = (cb_g.sn + 0.5 * w0) / (cb_g.nn + w0);
                    let b = an.and_then(|x| cb_a.get(&x)).copied().unwrap_or_default();
                    let mi = (b.si + 3.0 * gi) / (b.ni + 3.0);
                    let mn = (b.sn + 3.0 * gnn) / (b.nn + 3.0);
                    let cost_ratio = full.tok * ctxs / S_REF_D;
                    if uni(seed, 0xD6, t as u64, 0) < eps {
                        uni(seed, 0xD7, t as u64, 0) < 0.5
                    } else {
                        mi - mn > 0.03 * cost_ratio
                    }
                }
                _ => true,
            };
            let ci = if inject { full } else { CtxInfo { tok: 0.0, r: 0, irr: 0, n_docs: 0 } };
            if ci.n_docs > 0 {
                n_inj_task += 1.0;
                k_inj_sum += ci.n_docs as f64;
                res.n_inj += 1.0;
            }
            res.n_ret += f64::from(u8::from(ctx_k > 0));
            cb_ev = Some((an, ci.n_docs > 0, false));
            // attempts
            let n_att = 1 + c.max_retry;
            let mut tried: Vec<usize> = vec![];
            let mut best: (f64, bool, usize) = (-1.0, false, 0);
            let mut chosen: (bool, usize, bool) = (false, 0, false); // (correct, agent, verified-pass)
            for att in 0..n_att {
                let cands: Vec<usize> = (0..NA).filter(|a| !tried.contains(a)).collect();
                let call_ratio = (B_D + ci.tok * ctxs + 4.0 * O_D) / S_REF_D;
                let pv: Vec<f64> = (0..NA).map(|a| p_succ(base_p(seed, a, cid, qi, epoch), sp.d, cid, ci.r, ci.irr)).collect();
                let a = match c.route {
                    Route::Global => {
                        if rng.f64() < 0.05 {
                            cands[rng.below(cands.len())]
                        } else {
                            cands.iter().copied().fold((f64::MIN, cands[0]), |b, a| {
                                let m = (gs[a] + 1.0) / (gn[a] + 2.0) + 1e-6 * uni(seed, 0xD5, t as u64, a as u64);
                                if m > b.0 { (m, a) } else { b }
                            }).1
                        }
                    }
                    Route::Anchor { mu } => cands
                        .iter()
                        .copied()
                        .fold((f64::MIN, cands[0]), |b, a| {
                            let (m, n) = m_at(an, &astats, &gs, &gn, a);
                            let sc = m + 0.20 / (n + 1.0).sqrt() - mu * AG_PRICE[a] * call_ratio + 1e-6 * uni(seed, 0xD5, t as u64, a as u64);
                            if sc > b.0 { (sc, a) } else { b }
                        })
                        .1,
                    Route::OracleQ => cands.iter().copied().fold(cands[0], |b, a| if pv[a] > pv[b] { a } else { b }),
                    Route::OracleUtil { mu } => cands.iter().copied().fold(cands[0], |b, a| if pv[a] - mu * AG_PRICE[a] * call_ratio > pv[b] - mu * AG_PRICE[b] * call_ratio { a } else { b }),
                };
                tried.push(a);
                let correct = uni(seed, 0xD1, t as u64, att as u64 + 1) < pv[a];
                let pi = AG_PRICE[a];
                res.g_base += pi * B_D;
                res.g_ctx += pi * ci.tok;
                res.g_out += pi * O_D;
                res.n_gen += 1.0;
                att_log.push((an, a, correct));
                let do_ver = match c.vmode {
                    DV::Never => false,
                    DV::Always => true,
                    DV::Gate { rv, nmin } => {
                        let (m, n) = m_at(an, &astats, &gs, &gn, a);
                        let cv_eff = (VB_D + 4.0 * V_OUT_D + ci.tok * ctxs) / S_REF_D;
                        n < nmin || (1.0 - m) * rv >= cv_eff
                    }
                };
                if !do_ver {
                    chosen = (correct, a, false);
                    break;
                }
                res.v_in += VB_D;
                res.v_out += V_OUT_D;
                res.g_ctx += ci.tok;
                res.n_ver += 1.0;
                let u = uni(seed, 0xD2, t as u64, att as u64 + 1);
                let flagged = if correct { u < FALSE_FLAG } else { u < RECALL };
                // signal for best-of selection: unflagged > flagged
                let v = if flagged { 0.0 } else { 1.0 } + 0.01 * uni(seed, 0xD3, t as u64, att as u64);
                if !flagged {
                    chosen = (correct, a, true);
                    break;
                }
                if v > best.0 {
                    best = (v, correct, a);
                }
                if att + 1 == n_att {
                    chosen = (best.1, best.2, false);
                }
            }
            delivered = Some(chosen.0);
            if let Some(x) = cb_ev.as_mut() {
                x.2 = chosen.0;
            }
            // store in the cache
            if c.memo != MemoKind::Off && (!c.store_verified || chosen.2) {
                if !entries.contains_key(&qi) {
                    stored.push(qi);
                }
                entries.insert(qi, Entry { correct: chosen.0, concept: cid, qi, epoch });
                evict_key = Some(qi);
            }
            past.push(qi);
            anchor_of.push(an.unwrap_or(t));
            // an anchor created by this task is referenced by later neighbours
            an_for_ev = Some(an.unwrap_or(t));
        }
        let ok = delivered.unwrap();
        succ += f64::from(u8::from(ok));
        if t < 250 {
            succ250 += f64::from(u8::from(ok));
        }
        res_sum.add(&res);
        if (uni(seed, 0x2E, t as u64, 0)) < REVEAL {
            evs.push_back(Ev { due: t + DELAY, att: att_log.into_iter().map(|(_, a, s)| (an_for_ev, a, s)).collect(), cb: cb_ev.map(|(_, i, o)| (an_for_ev, i, o)), evict_key, delivered: ok });
        }
    }
    let nt = NT as f64;
    DSum { succ: succ / nt, succ250: succ250 / 250.0, res: res_sum.scaled(1.0 / nt), hit: hits / nt, wrong_hit: wrong_hits / nt, k_inj: if n_inj_task > 0.0 { k_inj_sum / n_inj_task } else { 0.0 }, inj_rate: n_inj_task / n_gen_task.max(1.0), gen_rate: n_gen_task / nt }
}
