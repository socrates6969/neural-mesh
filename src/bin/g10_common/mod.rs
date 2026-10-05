//! g10 shared pieces (W11, R&D round 3). Forked from f6_common / g1_common / f8; src/*.rs is NOT touched.
//!
//! * pools: (family x embedder) context-vector pools. Family F5 = results/f5_corpus.json (12 concepts, 24 question
//!   paraphrases + 3 bare subjects each, 2100 distractors) with hash / MiniLM (results/f5_embeddings.f32) / BGE-base
//!   (results/f8_bge_embeddings.f32). Family G1A / G1B = the HARD sibling-concept g1 corpora (240 concepts, 8 sibling
//!   details per topic; results/g1_*_{A,B}) with hash / MiniLM / BGE (results/g1_emb_{A,B}_{mini,bge}.f32).
//! * CtxWorld = d6/f5/f6 context world (+ an optional context-DRIFT: at stream fraction `at`, a fraction `frac` of the
//!   contexts get a NEW best agent; the text stream and every other draw are identical to the no-drift world).
//! * anchors: `Leader` (= knowledge::Knowledge leader/vote, hard sim_min), `Auto` (E1: the same Knowledge object but
//!   `cfg.sim_min` is set before every retrieval from a running Welford mean - kstd*std of the observed top-1 cosine),
//!   `Oracle`.
//! * VotePol = kn-direct+vote (KnowledgeRouter semantics) with an optional recency discount of the per-anchor/agent
//!   reward statistics (E2).
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::knowledge::{hash_embed, FeatMode, Knowledge, KnowledgeConfig};
use neural_mesh::memory::dot;
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize, t_two_sided_p};
use rayon::prelude::*;
use serde_json::{json, Value};

pub const NA: usize = 6;
pub const HORIZONS: [usize; 4] = [100, 250, 500, 2000];

pub fn parse_range(args: &[String], key: &str, default: (u64, u64)) -> Vec<u64> {
    let (a, b) = match args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)) {
        Some(r) => {
            let (a, b) = r.split_once('-').expect("range a-b");
            (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap())
        }
        None => default,
    };
    (a..=b).collect()
}
pub fn arg(args: &[String], key: &str, default: &str) -> String {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}
pub fn words(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_string()).collect()
}
fn hash_vec(text: &str) -> Vec<f32> {
    hash_embed(&Task { id: 0, task_type: None, features: words(text) }, 384)
}
pub fn mean_of(xs: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = xs.filter(|x| x.is_finite()).collect();
    if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
}

// ---------------------------------------------------------------------------------------------
// pools

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Src {
    Hash,
    Mini,
    Bge,
}
impl Src {
    pub fn label(self) -> &'static str {
        match self {
            Src::Hash => "hash",
            Src::Mini => "mini",
            Src::Bge => "bge",
        }
    }
    pub fn dim(self) -> usize {
        if self == Src::Bge { 768 } else { 384 }
    }
}
pub const SRCS: [Src; 3] = [Src::Hash, Src::Mini, Src::Bge];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Fam {
    F5,
    G1A,
    G1B,
}
impl Fam {
    pub fn label(self) -> &'static str {
        match self {
            Fam::F5 => "f5",
            Fam::G1A => "g1A",
            Fam::G1B => "g1B",
        }
    }
}

pub struct Pool {
    pub dim: usize,
    pub vecs: Vec<Vec<f32>>,
    /// per concept: item indices (F5: question paraphrases; G1: the concept's queries)
    pub conc: Vec<Vec<usize>>,
    /// F5 only: bare subject phrases per concept
    pub conc_alt: Vec<Vec<usize>>,
    pub topic_of_concept: Vec<i32>,
    pub dist: Vec<usize>,
    pub texts: Vec<String>,
}

static POOLS: OnceLock<Mutex<HashMap<(Fam, Src), Arc<Pool>>>> = OnceLock::new();
pub fn pool(f: Fam, s: Src) -> Arc<Pool> {
    let m = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(p) = m.lock().unwrap().get(&(f, s)) {
        return p.clone();
    }
    let p = Arc::new(match f {
        Fam::F5 => load_f5(s),
        Fam::G1A => load_g1("A", s),
        Fam::G1B => load_g1("B", s),
    });
    m.lock().unwrap().insert((f, s), p.clone());
    p
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|_| panic!("{path} missing"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn rows(all: &[f32], dim: usize) -> Vec<Vec<f32>> {
    all.chunks_exact(dim).map(|r| r.to_vec()).collect()
}

fn load_f5(s: Src) -> Pool {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string("results/f5_corpus.json").expect("results/f5_corpus.json")).unwrap();
    let items = meta["items"].as_array().unwrap();
    let nconc = meta["concepts"].as_array().unwrap().len();
    let texts: Vec<String> = items.iter().map(|i| i["text"].as_str().unwrap().to_string()).collect();
    let vecs: Vec<Vec<f32>> = match s {
        Src::Hash => texts.iter().map(|t| hash_vec(t)).collect(),
        Src::Mini => rows(&read_f32("results/f5_embeddings.f32"), 384),
        Src::Bge => rows(&read_f32("results/f8_bge_embeddings.f32"), 768),
    };
    assert_eq!(vecs.len(), items.len(), "f5 fixture size");
    let (mut conc, mut alt, mut dist) = (vec![vec![]; nconc], vec![vec![]; nconc], vec![]);
    for (i, it) in items.iter().enumerate() {
        let c = it["concept"].as_i64().unwrap();
        match it["kind"].as_str().unwrap() {
            "q" => conc[c as usize].push(i),
            "subj" => alt[c as usize].push(i),
            _ => dist.push(i),
        }
    }
    Pool { dim: s.dim(), vecs, conc, conc_alt: alt, topic_of_concept: vec![0; nconc], dist, texts }
}

fn load_g1(tag: &str, s: Src) -> Pool {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(format!("results/g1_corpus_{tag}.json")).expect("g1 corpus")).unwrap();
    let nconc = v["n_concepts"].as_u64().unwrap() as usize;
    let qs = v["queries"].as_array().unwrap();
    let texts: Vec<String> = qs.iter().map(|q| q["text"].as_str().unwrap().to_string()).collect();
    let vecs: Vec<Vec<f32>> = match s {
        Src::Hash => texts.iter().map(|t| hash_vec(t)).collect(),
        _ => {
            let all_texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(format!("results/g1_texts_{tag}.json")).unwrap()).unwrap();
            let ix: HashMap<&str, usize> = all_texts.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();
            let dim = s.dim();
            let all = read_f32(&format!("results/g1_emb_{tag}_{}.f32", if s == Src::Mini { "mini" } else { "bge" }));
            assert_eq!(all.len(), all_texts.len() * dim, "g1 fixture size");
            texts.iter().map(|t| all[ix[t.as_str()] * dim..(ix[t.as_str()] + 1) * dim].to_vec()).collect()
        }
    };
    let mut conc = vec![vec![]; nconc];
    let mut dist = vec![];
    for (i, q) in qs.iter().enumerate() {
        let c = q["concept"].as_i64().unwrap();
        if c >= 0 { conc[c as usize].push(i) } else { dist.push(i) }
    }
    let mut topic_of_concept = vec![0; nconc];
    for c in v["concepts"].as_array().unwrap() {
        topic_of_concept[c["id"].as_u64().unwrap() as usize] = c["topic"].as_i64().unwrap() as i32;
    }
    Pool { dim: s.dim(), vecs, conc, conc_alt: vec![vec![]; nconc], topic_of_concept, dist, texts }
}

// ---------------------------------------------------------------------------------------------
// worlds

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Drift {
    /// share of the active contexts whose best agent is redrawn (0 = no drift)
    pub frac: f64,
    /// stream position (share of the horizon) of the change
    pub at: f64,
}
pub const NO_DRIFT: Drift = Drift { frac: 0.0, at: 0.5 };

#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    pub fam: Fam,
    pub src: Src,
    /// F5: use the bare subject phrases instead of the question paraphrases
    pub subj: bool,
    /// G1: pick whole topics (8 sibling details each) instead of random concepts
    pub sib: bool,
    /// active concepts; 0 = unique context per task
    pub k: usize,
    pub w: f64,
    /// share of tasks that are unrelated one-off distractor texts
    pub mixed: f64,
    pub treatment: bool,
    pub drift: Drift,
}

pub struct CtxWorld {
    pub w: World,
    pub dim: usize,
    pub emb: Vec<Vec<f32>>,
    pctx: Vec<[f64; NA]>,
    pctx2: Vec<[f64; NA]>,
    drift_t: usize,
    ctx: Vec<usize>,
    wt: Vec<f64>,
    /// ground-truth concept per task (-1 = distractor / unique)
    pub truth: Vec<i32>,
    pub draws: Vec<(f64, f64, f64)>,
    /// first stream position that uses the post-drift mapping (usize::MAX = none)
    pub switch: usize,
    /// number of distinct distractor texts available (a unique-context stream repeats texts beyond this)
    pub first_pass: usize,
    spec: Spec,
    seed: u64,
}

fn draw_row(r: &mut Rng, best: usize) -> [f64; NA] {
    let mut row = [0.0; NA];
    for (ai, p) in row.iter_mut().enumerate() {
        *p = if ai == best { 0.80 + 0.10 * r.f64() } else { 0.15 + 0.35 * r.f64() };
    }
    row
}
fn draw_pctx(r: &mut Rng, kk: usize) -> Vec<[f64; NA]> {
    (0..kk).map(|_| { let b = r.below(NA); draw_row(r, b) }).collect()
}
fn best_of(row: &[f64; NA]) -> usize {
    (0..NA).max_by(|&a, &b| row[a].partial_cmp(&row[b]).unwrap()).unwrap()
}

impl CtxWorld {
    pub fn new(seed: u64, sp: &Spec) -> Self {
        let pl = pool(sp.fam, sp.src);
        let w = World::new(seed, WorldConfig { n_tasks: 2000, scenario: Scenario::Fixed, ..WorldConfig::default() });
        let n = w.len();
        let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xF5F5_1234));
        // active concepts
        let nconc = pl.conc.len();
        let mut active: Vec<usize> = vec![];
        if sp.k > 0 {
            if sp.fam == Fam::F5 {
                active = (0..sp.k.min(nconc)).collect();
            } else if sp.sib {
                let nt = (*pl.topic_of_concept.iter().max().unwrap() + 1) as usize;
                let mut topics: Vec<usize> = (0..nt).collect();
                for i in (1..topics.len()).rev() {
                    let j = r.below(i + 1);
                    topics.swap(i, j);
                }
                for &t in topics.iter().take(sp.k / 8) {
                    active.extend((0..nconc).filter(|&c| pl.topic_of_concept[c] as usize == t));
                }
            } else {
                let mut all: Vec<usize> = (0..nconc).collect();
                for i in (1..all.len()).rev() {
                    let j = r.below(i + 1);
                    all.swap(i, j);
                }
                active = all.into_iter().take(sp.k).collect();
            }
        }
        let kk = if sp.k == 0 { n } else { active.len() };
        let pctx = draw_pctx(&mut r, kk);
        let mut dpool = pl.dist.clone();
        for i in (1..dpool.len()).rev() {
            let j = r.below(i + 1);
            dpool.swap(i, j);
        }
        let mut di = 0usize;
        let (mut ctx, mut wt, mut truth, mut emb) = (vec![], vec![], vec![], vec![]);
        for t in 0..n {
            let isd = sp.k == 0 || (sp.mixed > 0.0 && r.f64() < sp.mixed);
            if isd {
                let idx = dpool[di % dpool.len()];
                di += 1;
                emb.push(pl.vecs[idx].clone());
                ctx.push(if sp.k == 0 { t } else { usize::MAX });
                wt.push(if sp.k == 0 { sp.w } else { 0.0 });
                truth.push(-1);
            } else {
                let ci = r.below(kk);
                let c = active[ci];
                let texts = if sp.subj { &pl.conc_alt[c] } else { &pl.conc[c] };
                let idx = texts[r.below(texts.len())];
                emb.push(pl.vecs[idx].clone());
                ctx.push(ci);
                wt.push(sp.w);
                truth.push(c as i32);
            }
        }
        let draws = (0..n).map(|_| (r.f64(), r.f64(), r.f64())).collect();
        // drift draws come AFTER everything else: texts / draws are identical to the no-drift world
        let mut pctx2 = pctx.clone();
        let mut switch = usize::MAX;
        if sp.drift.frac > 0.0 && sp.k > 0 {
            switch = ((n as f64) * sp.drift.at) as usize;
            let mut order: Vec<usize> = (0..kk).collect();
            for i in (1..order.len()).rev() {
                let j = r.below(i + 1);
                order.swap(i, j);
            }
            let nch = ((kk as f64) * sp.drift.frac).ceil() as usize;
            for &ci in order.iter().take(nch) {
                let old = best_of(&pctx[ci]);
                let mut b = r.below(NA - 1);
                if b >= old {
                    b += 1;
                }
                pctx2[ci] = draw_row(&mut r, b);
            }
        }
        let fp = dpool.len().min(n);
        CtxWorld { w, dim: pl.dim, emb, pctx, pctx2, drift_t: switch, ctx, wt, truth, draws, switch, first_pass: fp, spec: sp.clone(), seed }
    }
    pub fn p(&self, t: usize, a: usize) -> f64 {
        if self.w.is_down(t, a) {
            return 0.0;
        }
        let ty = self.w.task(t).type_idx;
        let base = self.w.skill(t, ty, a);
        let wg = self.wt[t];
        if wg == 0.0 {
            base
        } else {
            let pc = if t >= self.drift_t { &self.pctx2 } else { &self.pctx };
            (1.0 - wg) * base + wg * pc[self.ctx[t]][a]
        }
    }
    /// cached top-k neighbour lists of this world's text stream (see `compute_nbrs`)
    pub fn nbrs(&self) -> Arc<Nbrs> {
        nb_for(&self.spec, self.seed, &self.emb)
    }
    pub fn outcome(&self, t: usize, a: usize) -> f64 {
        if self.w.task(t).u < self.p(t, a) { 1.0 } else { 0.0 }
    }
}

fn mk(name: &str, fam: Fam, src: Src, subj: bool, sib: bool, k: usize, w: f64, mixed: f64, treatment: bool, drift: Drift) -> Spec {
    Spec { name: format!("{}-{}-{}", fam.label(), src.label(), name), fam, src, subj, sib, k, w, mixed, treatment, drift }
}

/// base worlds of a (family, embedder): treatments + the two controls
pub fn base_worlds(fam: Fam, src: Src) -> Vec<Spec> {
    if fam == Fam::F5 {
        vec![
            mk("q", fam, src, false, false, 12, 0.8, 0.0, true, NO_DRIFT),
            mk("subj3", fam, src, true, false, 12, 0.8, 0.0, true, NO_DRIFT),
            mk("q-mixed50", fam, src, false, false, 12, 0.8, 0.5, true, NO_DRIFT),
            mk("ctl-irrel", fam, src, false, false, 12, 0.0, 0.0, false, NO_DRIFT),
            mk("ctl-unique", fam, src, false, false, 0, 0.8, 0.0, false, NO_DRIFT),
        ]
    } else {
        vec![
            mk("sib16", fam, src, false, true, 16, 0.8, 0.0, true, NO_DRIFT),
            mk("rand16", fam, src, false, false, 16, 0.8, 0.0, true, NO_DRIFT),
            mk("sib16-mixed50", fam, src, false, true, 16, 0.8, 0.5, true, NO_DRIFT),
            mk("ctl-irrel", fam, src, false, true, 16, 0.0, 0.0, false, NO_DRIFT),
            mk("ctl-unique", fam, src, false, true, 0, 0.8, 0.0, false, NO_DRIFT),
        ]
    }
}
pub fn world(fam: Fam, src: Src, name: &str) -> Spec {
    base_worlds(fam, src).into_iter().find(|s| s.name.ends_with(&format!("-{name}"))).unwrap_or_else(|| panic!("no world {name}"))
}
// ---------------------------------------------------------------------------------------------
// neighbour cache: Knowledge stores EVERY query (anchor labels aside), so the top-k hits of query t over the t earlier
// queries do not depend on any threshold / anchor choice. We compute them once per (world texts, seed) with exactly the
// FlatIndex arithmetic (normalise, `memory::dot`, TopK) and replay the Knowledge::retrieve vote on top of them.
// `selfcheck_cache` (bin g10_calib --part selfcheck + unit test) proves the replay == the library object.

pub const KMAX: usize = 8;
pub struct Nbrs {
    /// per query t: up to KMAX (record id, cosine), best first
    pub hits: Vec<Vec<(u32, f32)>>,
    /// per query t >= 1: 4 cosines to pseudo-random earlier queries (chance level of this embedder)
    pub bg: Vec<[f32; 4]>,
}
pub fn compute_nbrs(emb: &[Vec<f32>]) -> Nbrs {
    use neural_mesh::memory::normalized;
    let nrm: Vec<Vec<f32>> = emb.iter().map(|v| normalized(v)).collect();
    let n = emb.len();
    let mut hits = Vec::with_capacity(n);
    let mut bg = Vec::with_capacity(n);
    for t in 0..n {
        let mut items: Vec<(u32, f32)> = Vec::with_capacity(KMAX + 1);
        for j in 0..t {
            let s = dot(&nrm[t], &nrm[j]);
            if items.len() >= KMAX && s <= items[KMAX - 1].1 {
                continue;
            }
            let pos = items.partition_point(|&(_, x)| x >= s);
            items.insert(pos, (j as u32, s));
            items.truncate(KMAX);
        }
        hits.push(items);
        let mut b = [0.0f32; 4];
        if t > 0 {
            for (j, x) in b.iter_mut().enumerate() {
                let i = ((t as u64).wrapping_mul(2654435761).wrapping_add((j as u64).wrapping_mul(40503)) >> 3) as usize % t;
                *x = dot(&nrm[t], &nrm[i]);
            }
        }
        bg.push(b);
    }
    Nbrs { hits, bg }
}

type NbKey = (Fam, Src, bool, bool, usize, u64, u64);
static NBCACHE: OnceLock<Mutex<HashMap<NbKey, Arc<OnceLock<Arc<Nbrs>>>>>> = OnceLock::new();
fn nb_for(sp: &Spec, seed: u64, emb: &[Vec<f32>]) -> Arc<Nbrs> {
    let key: NbKey = (sp.fam, sp.src, sp.subj, sp.sib, sp.k, sp.mixed.to_bits(), seed);
    let cell = {
        let mut m = NBCACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
        m.entry(key).or_insert_with(|| Arc::new(OnceLock::new())).clone()
    };
    cell.get_or_init(|| Arc::new(compute_nbrs(emb))).clone()
}
/// drop all cached neighbour lists (memory)
pub fn clear_nb_cache() {
    if let Some(m) = NBCACHE.get() {
        m.lock().unwrap().clear();
    }
}

#[derive(Clone, Debug)]
pub enum Thr {
    Fixed(f32),
    Auto { kstd: Option<f64>, n_boot: f64, boot: f64, bg: Option<f64> },
}
/// Knowledge::retrieve replay on cached neighbours (Leader and Auto)
pub struct CachedAnchor {
    nb: Arc<Nbrs>,
    k: usize,
    thr: Thr,
    t: usize,
    anchor_of: Vec<u64>,
    pub top1: Welford,
    pub bgw: Welford,
    pub last_thr: f64,
}
impl CachedAnchor {
    pub fn new(nb: Arc<Nbrs>, k: usize, thr: Thr) -> Self {
        CachedAnchor { nb, k: k.min(KMAX), thr, t: 0, anchor_of: vec![], top1: Welford::default(), bgw: Welford::default(), last_thr: 0.0 }
    }
    pub fn cur_thr(&self) -> f64 {
        match &self.thr {
            Thr::Fixed(s) => *s as f64,
            Thr::Auto { kstd, n_boot, boot, bg } => {
                let mut t = f64::NEG_INFINITY;
                if let Some(ks) = kstd {
                    if self.top1.n >= *n_boot && self.top1.n >= 2.0 {
                        t = t.max(self.top1.mean - ks * self.top1.std());
                    }
                }
                if let Some(z) = bg {
                    if self.bgw.n >= 4.0 * n_boot && self.bgw.n >= 8.0 {
                        t = t.max(self.bgw.mean + z * self.bgw.std());
                    }
                }
                if t == f64::NEG_INFINITY { *boot } else { t }
            }
        }
    }
}
impl Anchorer for CachedAnchor {
    fn assign(&mut self, _q: &[f32]) -> (u64, bool) {
        let thr = self.cur_thr();
        self.last_thr = thr;
        let thr32 = thr as f32;
        let hits = &self.nb.hits[self.t];
        let mut votes: Vec<(u64, f32)> = vec![];
        for &(id, c) in hits.iter().take(self.k) {
            if c < thr32 {
                continue;
            }
            let a = self.anchor_of[id as usize];
            match votes.iter_mut().find(|v| v.0 == a) {
                Some(v) => v.1 += c,
                None => votes.push((a, c)),
            }
        }
        let anchor = votes
            .iter()
            .fold(None, |b: Option<(u64, f32)>, &v| match b {
                Some(x) if x.1 >= v.1 => b,
                _ => Some(v),
            })
            .map(|v| v.0);
        if let Thr::Auto { bg, .. } = &self.thr {
            if let Some(h) = hits.first() {
                self.top1.push(h.1 as f64);
            }
            if bg.is_some() && self.t > 0 {
                for &x in &self.nb.bg[self.t] {
                    self.bgw.push(x as f64);
                }
            }
        }
        let a = anchor.unwrap_or(self.t as u64);
        self.anchor_of.push(a);
        self.t += 1;
        (a, anchor.is_none())
    }
}


// ---------------------------------------------------------------------------------------------
// anchors

pub trait Anchorer {
    /// Anchor id of this query (stored) and whether it founded a NEW anchor (nothing similar enough)
    fn assign(&mut self, q: &[f32]) -> (u64, bool);
    fn n_anchors(&self) -> usize {
        0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum AnchorCfg {
    /// the library's leader/vote clustering with a HARD sim_min
    Leader { k: usize, sim: f32 },
    /// E1: same object, sim_min = max(boot-or-(mean - kstd*std of top-1 cosine), optional background floor)
    Auto { k: usize, kstd: f32, n_boot: usize, boot: f32, bg: Option<f32> },
    Oracle,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Method {
    pub anchor: AnchorCfg,
}
impl Method {
    pub fn leader(k: usize, sim: f32) -> Self {
        Method { anchor: AnchorCfg::Leader { k, sim } }
    }
    pub fn auto(k: usize, kstd: f32) -> Self {
        Method { anchor: AnchorCfg::Auto { k, kstd, n_boot: 30, boot: 0.6, bg: None } }
    }
    pub fn label(&self) -> String {
        match &self.anchor {
            AnchorCfg::Leader { k, sim } => format!("leader(k{k} s{sim})"),
            AnchorCfg::Auto { k, kstd, n_boot, boot, bg } => {
                format!("auto(k{k} {} boot{n_boot}@{boot}{})", if *kstd >= 90.0 { "top1-off".to_string() } else { format!("kstd{kstd}") }, bg.map(|z| format!(" bg{z}")).unwrap_or_default())
            }
            AnchorCfg::Oracle => "oracle".into(),
        }
    }
}

struct LeaderKn {
    kn: Knowledge,
}
impl Anchorer for LeaderKn {
    fn assign(&mut self, q: &[f32]) -> (u64, bool) {
        let r = self.kn.retrieve(q);
        let a = r.anchor.unwrap_or(self.kn.len() as u64);
        self.kn.remember(q, &r);
        (a, r.anchor.is_none())
    }
}

/// Welford running mean / variance
#[derive(Clone, Debug, Default)]
pub struct Welford {
    pub n: f64,
    pub mean: f64,
    m2: f64,
}
impl Welford {
    pub fn push(&mut self, x: f64) {
        self.n += 1.0;
        let d = x - self.mean;
        self.mean += d / self.n;
        self.m2 += d * (x - self.mean);
    }
    pub fn std(&self) -> f64 {
        if self.n < 2.0 { 0.0 } else { (self.m2 / (self.n - 1.0)).sqrt() }
    }
}

/// E1 anchorer. Uses the library `Knowledge` unchanged except that `cfg.sim_min` is written before every retrieval
/// (what a lib patch would do inside `retrieve`).
pub struct AutoLeader {
    pub kn: Knowledge,
    pub top1: Welford,
    pub bgw: Welford,
    kstd: f64,
    n_boot: f64,
    boot: f64,
    bg: Option<f64>,
    items: Vec<Vec<f32>>,
    pub last_thr: f64,
}
impl AutoLeader {
    pub fn new(dim: usize, k: usize, kstd: f32, n_boot: usize, boot: f32, bg: Option<f32>) -> Self {
        let mut kc = KnowledgeConfig::new(dim);
        kc.k = k;
        kc.sim_min = boot;
        kc.feat = FeatMode::Off;
        AutoLeader { kn: Knowledge::new_shared(kc), top1: Welford::default(), bgw: Welford::default(), kstd: kstd as f64, n_boot: n_boot as f64, boot: boot as f64, bg: bg.map(|z| z as f64), items: vec![], last_thr: boot as f64 }
    }
    pub fn thr(&self) -> f64 {
        if self.top1.n < self.n_boot || self.top1.n < 2.0 {
            return self.boot;
        }
        let mut t = self.top1.mean - self.kstd * self.top1.std();
        if let Some(z) = self.bg {
            if self.bgw.n >= 8.0 {
                t = t.max(self.bgw.mean + z * self.bgw.std());
            }
        }
        t
    }
}
impl Anchorer for AutoLeader {
    fn assign(&mut self, q: &[f32]) -> (u64, bool) {
        let thr = self.thr();
        self.last_thr = thr;
        self.kn.cfg.sim_min = thr as f32;
        let r = self.kn.retrieve(q);
        if let Some(h) = r.hits.first() {
            self.top1.push(h.1 as f64);
        }
        if self.bg.is_some() && !self.items.is_empty() {
            // 4 deterministic pseudo-random stored items = the chance-level cosine of this embedder
            let n = self.items.len();
            for j in 0..4u64 {
                let i = ((self.kn.len() as u64).wrapping_mul(2654435761).wrapping_add(j.wrapping_mul(40503)) >> 3) as usize % n;
                self.bgw.push(dot(q, &self.items[i]) as f64);
            }
        }
        let a = r.anchor.unwrap_or(self.kn.len() as u64);
        self.kn.remember(q, &r);
        if self.bg.is_some() {
            self.items.push(q.to_vec());
        }
        (a, r.anchor.is_none())
    }
}

struct OracleAnchor {
    truth: Vec<i32>,
    t: usize,
    seen: std::collections::HashSet<i32>,
}
impl Anchorer for OracleAnchor {
    fn assign(&mut self, _q: &[f32]) -> (u64, bool) {
        let tr = self.truth[self.t];
        let t = self.t;
        self.t += 1;
        if tr < 0 {
            return (1_000_000 + t as u64, true);
        }
        let new = self.seen.insert(tr);
        (tr as u64, new)
    }
}

pub fn make_anchorer(a: &AnchorCfg, dim: usize, truth: &[i32], nb: Option<Arc<Nbrs>>) -> Box<dyn Anchorer> {
    if let Some(nb) = nb {
        match a {
            AnchorCfg::Leader { k, sim } => return Box::new(CachedAnchor::new(nb, *k, Thr::Fixed(*sim))),
            AnchorCfg::Auto { k, kstd, n_boot, boot, bg } => {
                return Box::new(CachedAnchor::new(nb, *k, Thr::Auto { kstd: if *kstd >= 90.0 { None } else { Some(*kstd as f64) }, n_boot: *n_boot as f64, boot: *boot as f64, bg: bg.map(|z| z as f64) }))
            }
            AnchorCfg::Oracle => {}
        }
    }
    match a {
        AnchorCfg::Oracle => Box::new(OracleAnchor { truth: truth.to_vec(), t: 0, seen: Default::default() }),
        AnchorCfg::Leader { k, sim } => {
            let mut kc = KnowledgeConfig::new(dim);
            kc.k = *k;
            kc.sim_min = *sim;
            kc.feat = FeatMode::Off;
            Box::new(LeaderKn { kn: Knowledge::new_shared(kc) })
        }
        AnchorCfg::Auto { k, kstd, n_boot, boot, bg } => Box::new(AutoLeader::new(dim, *k, *kstd, *n_boot, *boot, *bg)),
    }
}

// ---------------------------------------------------------------------------------------------
// anchor-layer metrics

#[derive(Clone, Debug, Default)]
pub struct Anc {
    pub p: f64,
    pub r: f64,
    pub f1: f64,
    /// distinct anchors among concept tasks / number of concepts (1 = perfect)
    pub frag: f64,
    /// pair-F1 over the first 300 concept tasks
    pub early_f1: f64,
    /// share of distractor/unique tasks (t >= 50, within the distinct-text window) that JOINED an existing anchor (spurious)
    pub spur: f64,
    /// share of distractor tasks whose anchor also holds a concept task (mixed worlds)
    pub contam: f64,
    /// mean share of tasks that founded a new anchor
    pub new_rate: f64,
}

fn f1(p: f64, r: f64) -> f64 {
    if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
}
fn pair_prf(anchor: &[u64], truth: &[i32], sel: impl Fn(usize) -> bool) -> (f64, f64, f64) {
    let mut by_a: HashMap<u64, usize> = HashMap::new();
    let mut by_c: HashMap<i32, usize> = HashMap::new();
    let mut by_ac: HashMap<(u64, i32), usize> = HashMap::new();
    for t in 0..anchor.len() {
        if truth[t] >= 0 && sel(t) {
            *by_a.entry(anchor[t]).or_insert(0) += 1;
            *by_c.entry(truth[t]).or_insert(0) += 1;
            *by_ac.entry((anchor[t], truth[t])).or_insert(0) += 1;
        }
    }
    let c2 = |x: usize| (x * x.saturating_sub(1) / 2) as f64;
    let (sa, sc, sac) = (by_a.values().map(|&x| c2(x)).sum::<f64>(), by_c.values().map(|&x| c2(x)).sum::<f64>(), by_ac.values().map(|&x| c2(x)).sum::<f64>());
    let (p, r) = (if sa > 0.0 { sac / sa } else { 0.0 }, if sc > 0.0 { sac / sc } else { 0.0 });
    (p, r, f1(p, r))
}

pub fn anchor_metrics(cw: &CtxWorld, m: &Method) -> Anc {
    let n = cw.emb.len();
    let mut an = make_anchorer(&m.anchor, cw.dim, &cw.truth, Some(cw.nbrs()));
    let mut anchor = vec![0u64; n];
    let mut is_new = vec![false; n];
    for t in 0..n {
        let (a, nw) = an.assign(&cw.emb[t]);
        anchor[t] = a;
        is_new[t] = nw;
    }
    let (p, r, f) = pair_prf(&anchor, &cw.truth, |_| true);
    let mut cnt = 0usize;
    let mut early = vec![false; n];
    for t in 0..n {
        if cw.truth[t] >= 0 && cnt < 300 {
            early[t] = true;
            cnt += 1;
        }
    }
    let (_, _, ef) = pair_prf(&anchor, &cw.truth, |t| early[t]);
    let mut anchors_c: HashMap<u64, usize> = HashMap::new();
    let mut concepts: HashMap<i32, usize> = HashMap::new();
    for t in 0..n {
        if cw.truth[t] >= 0 {
            *anchors_c.entry(anchor[t]).or_insert(0) += 1;
            *concepts.entry(cw.truth[t]).or_insert(0) += 1;
        }
    }
    let (mut dn, mut dc, mut sn, mut sj) = (0.0, 0.0, 0.0, 0.0);
    for t in 0..n {
        if cw.truth[t] < 0 {
            dn += 1.0;
            if anchors_c.contains_key(&anchor[t]) {
                dc += 1.0;
            }
            if t >= 50 && t < cw.first_pass {
                sn += 1.0;
                if !is_new[t] {
                    sj += 1.0;
                }
            }
        }
    }
    Anc {
        p,
        r,
        f1: f,
        frag: anchors_c.len() as f64 / concepts.len().max(1) as f64,
        early_f1: ef,
        spur: if sn > 0.0 { sj / sn } else { f64::NAN },
        contam: if dn > 0.0 { dc / dn } else { f64::NAN },
        new_rate: is_new.iter().filter(|&&x| x).count() as f64 / n as f64,
    }
}

// ---------------------------------------------------------------------------------------------
// policies / runs

#[derive(Clone, Debug, Default)]
pub struct RunOut {
    pub h: [f64; 4],
    pub fail: f64,
    /// expected success of the chosen agent per task
    pub pe: Vec<f32>,
    pub g: Vec<u32>,
    /// share of decisions taken by the vote (not the front)
    pub vote_share: f64,
    /// first post-drift position (usize::MAX = no drift)
    pub switch: usize,
}
pub fn col(r: &[RunOut], i: usize) -> Vec<f64> {
    r.iter().map(|x| x.h[i]).collect()
}
/// mean expected success over tasks [a, b)
pub fn win_mean(r: &RunOut, a: usize, b: usize) -> f64 {
    let b = b.min(r.pe.len());
    r.pe[a..b].iter().map(|&x| x as f64).sum::<f64>() / (b - a) as f64
}

pub trait Pol {
    fn pick(&mut self, t: &Task, emb: &[f32], c: &[usize], rng: &mut Rng) -> usize;
    fn fb(&mut self, t: &Task, a: usize, r: Option<f64>);
    fn vote_share(&self) -> f64 {
        0.0
    }
}
struct Plain<R: Router>(R);
impl<R: Router> Pol for Plain<R> {
    fn pick(&mut self, t: &Task, _e: &[f32], c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(t, c, rng)
    }
    fn fb(&mut self, t: &Task, a: usize, r: Option<f64>) {
        if let Some(r) = r {
            self.0.update(t, a, r)
        }
    }
}

/// E2: recency discount of the per-anchor/agent reward statistics
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Disc {
    None,
    /// every reward event of an anchor multiplies that anchor's (reward sum, count) of ALL agents by gamma first
    Event(f64),
    /// exponential decay with half-life h (in tasks) of the anchor's statistics (lazily, at write and at read)
    Time(f64),
}
impl Disc {
    pub fn label(&self) -> String {
        match self {
            Disc::None => "undisc".into(),
            Disc::Event(g) => format!("ev{g}"),
            Disc::Time(h) => format!("hl{h}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VoteCfg {
    pub min_n: f64,
    pub tau: f64,
    /// (min_n, margin, probe_every)
    pub gate: Option<(f64, f64, usize)>,
    pub disc: Disc,
    /// per-update decay of the vote-gate sums (1.0 = cumulative, as the library)
    pub gate_disc: f64,
}
/// the f5/f6 vote configuration (min_n 3, tau .7, gate 6/.02/25)
pub fn std_vote() -> VoteCfg {
    VoteCfg { min_n: 3.0, tau: 0.7, gate: Some((6.0, 0.02, 25)), disc: Disc::None, gate_disc: 1.0 }
}

struct Row {
    /// per agent (tries, reward sum, rewards)
    a: Vec<(f64, f64, f64)>,
    last: usize,
}
struct VotePol<R: Router> {
    front: R,
    anch: Box<dyn Anchorer>,
    vc: VoteCfg,
    astats: HashMap<u64, Row>,
    gate: [f64; 4],
    probe: usize,
    live: HashMap<u64, (u64, bool, usize, bool)>,
    order: VecDeque<u64>,
    now: usize,
    n_vote: f64,
    n_dec: f64,
}
impl<R: Router> VotePol<R> {
    fn decay_to_now(&self, row: &mut Row) {
        if let Disc::Time(h) = self.vc.disc {
            let f = 0.5f64.powf((self.now.saturating_sub(row.last)) as f64 / h);
            for c in row.a.iter_mut() {
                c.1 *= f;
                c.2 *= f;
            }
        }
        row.last = self.now;
    }
    fn vote(&self, a: u64, cands: &[usize]) -> Option<usize> {
        let row = self.astats.get(&a)?;
        let f = match self.vc.disc {
            Disc::Time(h) => 0.5f64.powf((self.now.saturating_sub(row.last)) as f64 / h),
            _ => 1.0,
        };
        let mut best: Option<(usize, f64)> = None;
        for &c in cands {
            let Some(&(_, s, n)) = row.a.get(c) else { continue };
            let (s, n) = (s * f, n * f);
            if n < self.vc.min_n {
                continue;
            }
            let m = (1.0 + s) / (2.0 + n);
            if m < self.vc.tau {
                continue;
            }
            best = match best {
                Some((_, bm)) if bm >= m => best,
                _ => Some((c, m)),
            };
        }
        best.map(|b| b.0)
    }
    fn gate_open(&mut self) -> bool {
        let Some((min_n, margin, every)) = self.vc.gate else { return true };
        let g = self.gate;
        if g[1] < min_n || g[3] < min_n {
            return true;
        }
        if g[0] / g[1] >= g[2] / g[3] - margin {
            return true;
        }
        self.probe += 1;
        every > 0 && self.probe % every == 0
    }
    fn enrich(t: &Task, a: u64, is_new: bool) -> Task {
        let mut x = t.clone();
        if !is_new {
            x.features.push(format!("kn:{a}"));
        }
        x
    }
}
impl<R: Router> Pol for VotePol<R> {
    fn pick(&mut self, t: &Task, e: &[f32], c: &[usize], rng: &mut Rng) -> usize {
        self.now += 1;
        let (a, is_new) = self.anch.assign(e);
        let voted = if is_new {
            None
        } else {
            match self.vote(a, c) {
                Some(x) if self.gate_open() => Some(x),
                _ => None,
            }
        };
        let (agent, by_vote) = match voted {
            Some(x) => (x, true),
            None => (self.front.select(&Self::enrich(t, a, is_new), c, rng), false),
        };
        self.n_dec += 1.0;
        if by_vote {
            self.n_vote += 1.0;
        }
        let row = self.astats.entry(a).or_insert_with(|| Row { a: vec![], last: self.now });
        if row.a.len() <= agent {
            row.a.resize(agent + 1, (0.0, 0.0, 0.0));
        }
        row.a[agent].0 += 1.0;
        self.live.insert(t.id, (a, is_new, agent, by_vote));
        self.order.push_back(t.id);
        while self.order.len() > 512 {
            if let Some(id) = self.order.pop_front() {
                self.live.remove(&id);
            }
        }
        agent
    }
    fn fb(&mut self, t: &Task, agent: usize, r: Option<f64>) {
        let Some(r) = r else { return };
        let (routed, rec) = match self.live.get(&t.id) {
            Some(&(a, n, ag, bv)) => (Self::enrich(t, a, n), Some((a, ag, bv))),
            None => (t.clone(), None),
        };
        if let Some((a, ag, bv)) = rec {
            let rr = if r.is_finite() { r.clamp(0.0, 1.0) } else { 0.0 };
            if self.vc.gate_disc < 1.0 {
                let gd = self.vc.gate_disc;
                self.gate.iter_mut().for_each(|g| *g *= gd);
            }
            if bv {
                self.gate[0] += rr;
                self.gate[1] += 1.0;
            } else {
                self.gate[2] += rr;
                self.gate[3] += 1.0;
            }
            if let Some(mut row) = self.astats.remove(&a) {
                self.decay_to_now(&mut row);
                if let Disc::Event(g) = self.vc.disc {
                    for c in row.a.iter_mut() {
                        c.1 *= g;
                        c.2 *= g;
                    }
                }
                if let Some(cc) = row.a.get_mut(ag) {
                    cc.1 += rr;
                    cc.2 += 1.0;
                }
                self.astats.insert(a, row);
            }
        }
        self.front.update(&routed, agent, r);
    }
    fn vote_share(&self) -> f64 {
        if self.n_dec > 0.0 { self.n_vote / self.n_dec } else { 0.0 }
    }
}

#[derive(Clone, Debug)]
pub enum Kind2 {
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm,
    Guard,
    KnVote(Method, VoteCfg),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signal {
    Outcome,
    Raw,
    Reuse,
}
pub type A = (Kind2, Signal);

fn x1_guard() -> GuardConfig {
    GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
}

fn build(k: &Kind2, w: &World, dim: usize, truth: &[i32], nb: Option<Arc<Nbrs>>) -> Box<dyn Pol> {
    let kw = w.keyword_map_for();
    match k {
        Kind2::Keyword => Box::new(Plain(KeywordRouter::new(kw, NA))),
        Kind2::Eps => Box::new(Plain(EpsGreedyRouter::new(NA, 0.99, 0.02))),
        Kind2::Pts => Box::new(Plain(PooledThompsonRouter::new(NA, 1.0))),
        Kind2::HebR => Box::new(Plain(HebbianRouter::new(NA, heb_robust_cfg()))),
        Kind2::Warm => Box::new(Plain(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, 1.0).with_evidence(Evidence::MeanFeature))),
        Kind2::Guard => Box::new(Plain(Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, x1_guard()))),
        Kind2::KnVote(m, vc) => Box::new(VotePol {
            front: Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, x1_guard()),
            anch: make_anchorer(&m.anchor, dim, truth, nb),
            vc: vc.clone(),
            astats: HashMap::new(),
            gate: [0.0; 4],
            probe: 0,
            live: HashMap::new(),
            order: VecDeque::new(),
            now: 0,
            n_vote: 0.0,
            n_dec: 0.0,
        }),
    }
}

struct Weak {
    tool_base: f64,
    tool_slope: f64,
    reuse_frac: f64,
    reuse_delay: usize,
    reuse_noise: f64,
}
const WEAK: Weak = Weak { tool_base: 0.01, tool_slope: 0.03, reuse_frac: 0.3, reuse_delay: 25, reuse_noise: 0.1 };

pub fn run(k: &Kind2, sp: &Spec, sig: Signal, seed: u64) -> RunOut {
    let cw = CtxWorld::new(seed, sp);
    let w = &cw.w;
    let cands: Vec<usize> = (0..NA).collect();
    let mut pol = build(k, w, cw.dim, &cw.truth, if matches!(k, Kind2::KnVote(..)) { Some(cw.nbrs()) } else { None });
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut pending: VecDeque<(usize, usize, usize, f64)> = VecDeque::new();
    let (mut cum, mut fs, mut fnn) = (0.0f64, 0.0f64, 0.0f64);
    let mut out = RunOut::default();
    out.pe.reserve(w.len());
    out.g.reserve(w.len());
    let mut graded = 0u32;
    for t in 0..w.len() {
        while pending.front().is_some_and(|p| p.0 <= t) {
            let (_, ti, a, v) = pending.pop_front().unwrap();
            pol.fb(&w.task(ti).task, a, Some(v));
            graded += 1;
        }
        let st = w.task(t);
        let a = pol.pick(&st.task, &cw.emb[t], &cands, &mut rng);
        let pe = cw.p(t, a);
        let s = cw.outcome(t, a);
        match sig {
            Signal::Outcome => {
                pol.fb(&st.task, a, Some(s));
                graded += 1;
            }
            _ => {
                let (u_tool, u_reuse, u_noise) = cw.draws[t];
                let raw_ok = !w.is_down(t, a) && u_tool >= WEAK.tool_base + WEAK.tool_slope * (1.0 - s);
                match sig {
                    Signal::Raw => pol.fb(&st.task, a, Some(if raw_ok { 1.0 } else { 0.0 })),
                    _ => {
                        pol.fb(&st.task, a, if raw_ok { None } else { Some(0.0) });
                        if raw_ok && u_reuse < WEAK.reuse_frac {
                            let v = if u_noise < WEAK.reuse_noise { 1.0 - s } else { s };
                            pending.push_back((t + WEAK.reuse_delay, t, a, v));
                        }
                    }
                }
            }
        }
        cum += pe;
        out.pe.push(pe as f32);
        out.g.push(graded);
        if w.in_failure(t) {
            fs += pe;
            fnn += 1.0;
        }
        for (i, &hz) in HORIZONS.iter().enumerate() {
            if t + 1 == hz {
                out.h[i] = cum / hz as f64;
            }
        }
    }
    out.fail = fs / fnn.max(1.0);
    out.vote_share = pol.vote_share();
    out.switch = cw.switch;
    out
}

pub fn eval(a: &A, sp: &Spec, seeds: &[u64]) -> Vec<RunOut> {
    seeds.par_iter().map(|&s| run(&a.0, sp, a.1, s)).collect()
}

/// [method][spec][seed] anchor metrics
pub fn eval_anchor(methods: &[Method], specs: &[Spec], seeds: &[u64]) -> Vec<Vec<Vec<Anc>>> {
    let jobs: Vec<(usize, u64)> = (0..specs.len()).flat_map(|w| seeds.iter().map(move |&s| (w, s))).collect();
    let res: Vec<Vec<Anc>> = jobs
        .par_iter()
        .map(|&(wi, s)| {
            let cw = CtxWorld::new(s, &specs[wi]);
            methods.iter().map(|m| anchor_metrics(&cw, m)).collect()
        })
        .collect();
    let mut out = vec![vec![vec![]; specs.len()]; methods.len()];
    for (j, &(wi, _)) in jobs.iter().enumerate() {
        for mi in 0..methods.len() {
            out[mi][wi].push(res[j][mi].clone());
        }
    }
    out
}

/// [arm][spec][seed] full routing runs
pub fn eval_runs(arms: &[A], specs: &[Spec], seeds: &[u64]) -> Vec<Vec<Vec<RunOut>>> {
    let jobs: Vec<(usize, usize, u64)> = (0..specs.len()).flat_map(|w| (0..arms.len()).flat_map(move |a| seeds.iter().map(move |&s| (w, a, s)))).collect();
    let res: Vec<RunOut> = jobs.par_iter().map(|&(wi, ai, s)| run(&arms[ai].0, &specs[wi], arms[ai].1, s)).collect();
    let mut out: Vec<Vec<Vec<RunOut>>> = vec![vec![vec![]; specs.len()]; arms.len()];
    for (j, &(wi, ai, _)) in jobs.iter().enumerate() {
        out[ai][wi].push(res[j].clone());
    }
    out
}

pub fn bases_o() -> Vec<(String, A)> {
    vec![
        ("keyword".into(), (Kind2::Keyword, Signal::Outcome)),
        ("eps-greedy".into(), (Kind2::Eps, Signal::Outcome)),
        ("pooled-ts".into(), (Kind2::Pts, Signal::Outcome)),
        ("hebbian-robust".into(), (Kind2::HebR, Signal::Outcome)),
        ("warm(hebr)".into(), (Kind2::Warm, Signal::Outcome)),
        ("guarded(hebr)".into(), (Kind2::Guard, Signal::Outcome)),
    ]
}

// ---------------------------------------------------------------------------------------------
// statistics helpers (TOST as in g2_common; paired t-CI + Holm come from neural_mesh::stats)

/// two one-sided paired t-tests for |mean diff| < margin; returns (p_tost, mean, se, df)
pub fn tost(d: &[f64], margin: f64) -> (f64, f64, f64, usize) {
    let n = d.len();
    if n < 2 {
        return (1.0, f64::NAN, f64::NAN, 0);
    }
    let m = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    if se < 1e-15 {
        return (if m.abs() < margin { 0.0 } else { 1.0 }, m, se, n - 1);
    }
    let df = n - 1;
    let one = |t: f64| {
        let p2 = t_two_sided_p(t.abs(), df);
        if t >= 0.0 { p2 / 2.0 } else { 1.0 - p2 / 2.0 }
    };
    let p1 = one((m + margin) / se); // H0: diff <= -margin
    let p2 = one(-(m - margin) / se); // H0: diff >= +margin
    (p1.max(p2), m, se, df)
}
/// t critical value for a two-sided level (1 - alpha) by bisection on t_two_sided_p
pub fn t_crit(alpha: f64, df: usize) -> f64 {
    let (mut lo, mut hi) = (0.0f64, 50.0f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if t_two_sided_p(mid, df) > alpha { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}
/// 90% CI of the mean of d (the TOST-equivalent interval)
pub fn ci90(d: &[f64]) -> (f64, f64, f64) {
    let (_, m, se, df) = tost(d, 0.0);
    let t = t_crit(0.10, df.max(1));
    (m, m - t * se, m + t * se)
}

pub fn method_json(m: &Method) -> Value {
    let a = match &m.anchor {
        AnchorCfg::Leader { k, sim } => json!({"type": "leader", "k": k, "sim": sim}),
        AnchorCfg::Auto { k, kstd, n_boot, boot, bg } => json!({"type": "auto", "k": k, "kstd": kstd, "n_boot": n_boot, "boot": boot, "bg": bg}),
        AnchorCfg::Oracle => json!({"type": "oracle"}),
    };
    json!({"anchor": a, "label": m.label()})
}
pub fn method_from_json(v: &Value) -> Method {
    let a = &v["anchor"];
    let f = |k: &str| a[k].as_f64().unwrap() as f32;
    let anchor = match a["type"].as_str().unwrap() {
        "leader" => AnchorCfg::Leader { k: a["k"].as_u64().unwrap() as usize, sim: f("sim") },
        "auto" => AnchorCfg::Auto { k: a["k"].as_u64().unwrap() as usize, kstd: f("kstd"), n_boot: a["n_boot"].as_u64().unwrap() as usize, boot: f("boot"), bg: a["bg"].as_f64().map(|x| x as f32) },
        _ => AnchorCfg::Oracle,
    };
    Method { anchor }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_matches_batch() {
        let xs = [0.3, 0.9, 0.5, 0.7, 0.8, 0.1];
        let mut w = Welford::default();
        xs.iter().for_each(|&x| w.push(x));
        let m = xs.iter().sum::<f64>() / 6.0;
        let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / 5.0;
        assert!((w.mean - m).abs() < 1e-12 && (w.std() - v.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn auto_bootstrap_then_adapts() {
        let mut a = AutoLeader::new(2, 3, 2.0, 5, 0.6, None);
        assert!((a.thr() - 0.6).abs() < 1e-6);
        for i in 0..40 {
            let ang = (i as f32) * 0.01;
            a.assign(&[ang.cos(), ang.sin()]);
        }
        // all queries near-identical: top-1 cosines ~1 with tiny std => threshold close to 1, not the bootstrap 0.6
        assert!(a.thr() > 0.9, "thr {}", a.thr());
    }

    #[test]
    fn cached_replay_equals_library_knowledge() {
        let mut r = Rng::new(5);
        let cent: Vec<Vec<f32>> = (0..5).map(|_| (0..16).map(|_| r.f64() as f32 - 0.5).collect()).collect();
        let emb: Vec<Vec<f32>> = (0..300).map(|i| cent[i * 7 % 5].iter().map(|x| x + 0.25 * (r.f64() as f32 - 0.5)).collect()).collect();
        let nb = Arc::new(compute_nbrs(&emb));
        for cfg in [AnchorCfg::Leader { k: 5, sim: 0.7 }, AnchorCfg::Leader { k: 8, sim: 0.5 }, AnchorCfg::Auto { k: 5, kstd: 1.5, n_boot: 20, boot: 0.6, bg: None }] {
            let mut a = make_anchorer(&cfg, 16, &[], Some(nb.clone()));
            let mut b = make_anchorer(&cfg, 16, &[], None);
            for e in &emb {
                assert_eq!(a.assign(e), b.assign(e), "{cfg:?}");
            }
        }
    }

    #[test]
    fn tost_detects_equivalence_and_difference() {
        let same: Vec<f64> = (0..30).map(|i| if i % 2 == 0 { 0.001 } else { -0.001 }).collect();
        assert!(tost(&same, 0.01).0 < 0.05);
        let diff: Vec<f64> = (0..30).map(|i| 0.02 + if i % 2 == 0 { 0.001 } else { -0.001 }).collect();
        assert!(tost(&diff, 0.01).0 > 0.5);
        let (m, lo, hi) = ci90(&same);
        assert!(lo < m && m < hi);
    }

    #[test]
    fn event_discount_forgets() {
        // an anchor whose best agent flips: undiscounted stats keep trusting the old agent longer than gamma=.8 stats
        let mk = |disc: Disc| {
            let vc = VoteCfg { min_n: 3.0, tau: 0.7, gate: None, disc, gate_disc: 1.0 };
            let mut p = VotePol {
                front: HebbianRouter::new(NA, heb_robust_cfg()),
                anch: Box::new(OracleAnchor { truth: vec![0; 400], t: 0, seen: Default::default() }),
                vc,
                astats: HashMap::new(),
                gate: [0.0; 4],
                probe: 0,
                live: HashMap::new(),
                order: VecDeque::new(),
                now: 0,
                n_vote: 0.0,
                n_dec: 0.0,
            };
            let t = Task { id: 1, task_type: None, features: vec![] };
            let mut rng = Rng::new(1);
            let c: Vec<usize> = (0..NA).collect();
            for i in 0..60u64 {
                let t = Task { id: i + 1, ..t.clone() };
                p.pick(&t, &[0.0], &c, &mut rng);
                p.live.insert(t.id, (0, false, 2, false));
                p.fb(&t, 2, Some(1.0));
            }
            // flip: agent 2 now pays 0
            for i in 60..80u64 {
                let t = Task { id: i + 1, ..t.clone() };
                p.pick(&t, &[0.0], &c, &mut rng);
                p.live.insert(t.id, (0, false, 2, false));
                p.fb(&t, 2, Some(0.0));
            }
            p.vote(0, &c)
        };
        assert_eq!(mk(Disc::None), Some(2), "undiscounted still trusts agent 2 after 20 failures out of 80");
        assert_eq!(mk(Disc::Event(0.8)), None, "discounted stats no longer qualify agent 2");
    }
}
