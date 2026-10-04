//! Shared pieces of the f6 lab (MiniLM through the embedding abstraction + anchor-fragmentation fix).
//! Included from the f6_* bins with `#[path = "f6_common/mod.rs"] mod common;`.
//!
//! * the MiniLM vectors reach the experiments ONLY through `embedding::Embedding` (feature `real-embed`:
//!   `EmbeddingConfig::Real{source: results/f6_minilm_fixture.json}`; without the feature the same fixture is
//!   loaded with `TableEmbedding::from_path`); the old `results/f5_embeddings.f32` file is used only by the
//!   fixture bin to prove equivalence.
//! * worlds = copy of f5's `CtxWorld` (mini / hash sources) + three paraphrase-robustness corpora.
//! * anchor methods (`Anchorer`): the library's leader/vote (`Knowledge`, = f5), centroid anchors, centroid
//!   anchors with agglomerative merge, kNN leader with bridge-merge.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::embedding::Embedding;
#[cfg(not(feature = "real-embed"))]
use neural_mesh::embedding::TableEmbedding;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::knowledge::{hash_embed, AnchorMap, AnchorRouter, FeatMode, Knowledge, KnowledgeConfig};
use neural_mesh::memory::dot;
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

pub const DIM: usize = 384;
pub const NA: usize = 6;
pub const HORIZONS: [usize; 4] = [100, 250, 500, 2000];
pub const FIXTURE: &str = "results/f6_minilm_fixture.json";
pub const PARA_FIXTURE: &str = "results/f6_para_fixture.json";

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

// ---------------------------------------------------------------------------------------------
// embedders

pub fn words(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_string()).collect()
}

/// f5's "hash" source as an `Embedding`: knowledge::hash_embed over the lower-cased alphanumeric words of the text.
pub struct WordHash {
    pub dim: usize,
}
impl Embedding for WordHash {
    fn dim(&self) -> usize {
        self.dim
    }
    fn embed(&self, text: &str) -> Vec<f32> {
        hash_embed(&Task { id: 0, task_type: None, features: words(text) }, self.dim)
    }
    fn name(&self) -> &str {
        "hash-words"
    }
}

/// Load the MiniLM fixture through the abstraction. With feature `real-embed` this is
/// `EmbeddingConfig::Real{..}.build()`; without it the same fixture is read by `TableEmbedding`.
pub fn load_embedder(path: &str, normalize: bool) -> Arc<dyn Embedding> {
    #[cfg(feature = "real-embed")]
    {
        use neural_mesh::embedding::EmbeddingConfig;
        EmbeddingConfig::Real { source: path.to_string(), dim: DIM, cache: false, normalize, center: false }.build(DIM).unwrap_or_else(|e| panic!("{path}: {e}"))
    }
    #[cfg(not(feature = "real-embed"))]
    {
        Arc::new(TableEmbedding::from_path(path, normalize, false).unwrap_or_else(|e| panic!("{path}: {e}")))
    }
}
pub fn real_embed_feature() -> bool {
    cfg!(feature = "real-embed")
}

// ---------------------------------------------------------------------------------------------
// corpus (labels from results/f5_corpus.json + results/f6_para_fixture.json meta; vectors from the fixtures)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Q,
    Subj,
    Dist,
    NewSubj,
    NewQ,
    Typo,
}

pub struct Corpus {
    pub texts: Vec<String>,
    pub kind: Vec<Kind>,
    pub concept: Vec<i32>,
    pub topic: Vec<i32>,
    pub mini: Vec<Vec<f32>>,
    pub hash: Vec<Vec<f32>>,
    pub q: Vec<Vec<usize>>,
    pub subj: Vec<Vec<usize>>,
    pub para: Vec<Vec<usize>>,
    pub qpara: Vec<Vec<usize>>,
    pub dist: Vec<usize>,
    pub nconc: usize,
    pub model_sha: String,
}

static CORPUS: OnceLock<Corpus> = OnceLock::new();
pub fn corpus() -> &'static Corpus {
    CORPUS.get_or_init(load_corpus)
}

pub fn load_corpus() -> Corpus {
    let meta: Value = serde_json::from_str(&std::fs::read_to_string("results/f5_corpus.json").expect("results/f5_corpus.json")).unwrap();
    let emb = load_embedder(FIXTURE, false);
    let hash = WordHash { dim: DIM };
    let nconc = meta["concepts"].as_array().unwrap().len();
    let (mut texts, mut kind, mut concept, mut topic) = (vec![], vec![], vec![], vec![]);
    for it in meta["items"].as_array().unwrap() {
        texts.push(it["text"].as_str().unwrap().to_string());
        kind.push(match it["kind"].as_str().unwrap() {
            "q" => Kind::Q,
            "subj" => Kind::Subj,
            _ => Kind::Dist,
        });
        concept.push(it["concept"].as_i64().unwrap() as i32);
        topic.push(it["topic"].as_i64().unwrap() as i32);
    }
    let pmeta: Value = serde_json::from_str(&std::fs::read_to_string(PARA_FIXTURE).expect("results/f6_para_fixture.json (run py/f6_embed.py)")).unwrap();
    for it in pmeta["meta"]["items"].as_array().unwrap() {
        texts.push(it["text"].as_str().unwrap().to_string());
        kind.push(match it["kind"].as_str().unwrap() {
            "newsubj" => Kind::NewSubj,
            "newq" => Kind::NewQ,
            _ => Kind::Typo,
        });
        concept.push(it["concept"].as_i64().unwrap() as i32);
        topic.push(it["topic"].as_i64().unwrap() as i32);
    }
    drop(pmeta);
    let pemb = load_embedder(PARA_FIXTURE, false);
    let n5 = meta["items"].as_array().unwrap().len();
    let mut mini = Vec::with_capacity(texts.len());
    let mut hv = Vec::with_capacity(texts.len());
    for (i, t) in texts.iter().enumerate() {
        let v = if i < n5 { emb.embed(t) } else { pemb.embed(t) };
        assert!(v.iter().map(|x| x * x).sum::<f32>() > 0.5, "text missing from fixture: {t:?}");
        mini.push(v);
        hv.push(hash.embed(t));
    }
    let (mut q, mut subj, mut para, mut dist) = (vec![vec![]; nconc], vec![vec![]; nconc], vec![vec![]; nconc], vec![]);
    for i in 0..texts.len() {
        let c = concept[i];
        match kind[i] {
            Kind::Q => q[c as usize].push(i),
            Kind::Subj => subj[c as usize].push(i),
            Kind::Dist => dist.push(i),
            _ => para[c as usize].push(i),
        }
    }
    let qpara: Vec<Vec<usize>> = (0..nconc).map(|c| q[c].iter().chain(&para[c]).copied().collect()).collect();
    Corpus { texts, kind, concept, topic, mini, hash: hv, q, subj, para, qpara, dist, nconc, model_sha: meta["model_sha256"].as_str().unwrap_or("").to_string() }
}

// ---------------------------------------------------------------------------------------------
// worlds (copy of f5's CtxWorld, mini / hash sources)

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Src {
    Mini,
    Hash,
}
impl Src {
    pub fn label(self) -> &'static str {
        match self {
            Src::Mini => "mini",
            Src::Hash => "hash",
        }
    }
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Corp {
    /// f5 question paraphrases (3 subject phrasings x 8 templates)
    Q,
    /// f5 bare subject phrases (3 per concept)
    Subj,
    /// held-out paraphrases only (5 new subject phrasings x 5 new templates + bare + typos)
    Para,
    /// f5 questions UNION held-out paraphrases
    QPara,
    /// first half of the stream uses f5 questions, second half the held-out paraphrases (unseen phrasings)
    QThenPara,
}

#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    pub src: Src,
    pub corp: Corp,
    pub scenario: Scenario,
    /// recurring concepts; 0 = unique context per task
    pub k: usize,
    pub w: f64,
    /// fraction of tasks that are unrelated one-off distractor texts (no context effect)
    pub mixed: f64,
    pub treatment: bool,
    /// optional sample-efficiency targets (trailing-`win` mean expected success); when non-empty `run` fills RunOut.tt/gg/rr
    pub targets: Vec<f64>,
    pub win: usize,
}

pub struct CtxWorld {
    pub w: World,
    pub emb: Vec<Vec<f32>>,
    pctx: Vec<[f64; NA]>,
    ctx: Vec<usize>,
    wt: Vec<f64>,
    /// ground-truth concept per task (-1 = distractor / unique)
    pub truth: Vec<i32>,
    pub topic: Vec<i32>,
    pub draws: Vec<(f64, f64, f64)>,
    pub switch: usize,
}

fn draw_pctx(r: &mut Rng, kk: usize) -> Vec<[f64; NA]> {
    let mut pctx = Vec::with_capacity(kk);
    for _ in 0..kk {
        let best = r.below(NA);
        let mut row = [0.0; NA];
        for (ai, p) in row.iter_mut().enumerate() {
            *p = if ai == best { 0.80 + 0.10 * r.f64() } else { 0.15 + 0.35 * r.f64() };
        }
        pctx.push(row);
    }
    pctx
}

impl CtxWorld {
    pub fn new(seed: u64, sp: &Spec) -> Self {
        Self::new_in(seed, sp, corpus())
    }
    pub fn new_in(seed: u64, sp: &Spec, f: &Corpus) -> Self {
        let w = World::new(seed, WorldConfig { n_tasks: 2000, scenario: sp.scenario, ..WorldConfig::default() });
        let n = w.len();
        let (mut ctx, mut wt, mut truth, mut topic, mut emb) = (vec![], vec![], vec![], vec![], vec![]);
        let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xF5F5_1234));
        let kk = if sp.k == 0 { n } else { sp.k.min(f.nconc) };
        let pctx = draw_pctx(&mut r, kk);
        let mut pool = f.dist.clone();
        for i in (1..pool.len()).rev() {
            let j = r.below(i + 1);
            pool.swap(i, j);
        }
        let mut di = 0usize;
        let switch = n / 2;
        let vec_of = |i: usize| -> Vec<f32> {
            if sp.src == Src::Mini { f.mini[i].clone() } else { f.hash[i].clone() }
        };
        for t in 0..n {
            let isd = sp.k == 0 || (sp.mixed > 0.0 && r.f64() < sp.mixed);
            if isd {
                let idx = pool[di % pool.len()];
                di += 1;
                emb.push(vec_of(idx));
                ctx.push(if sp.k == 0 { t } else { usize::MAX });
                wt.push(if sp.k == 0 { sp.w } else { 0.0 });
                truth.push(-1);
                topic.push(-1);
            } else {
                let c = r.below(kk);
                let texts = match sp.corp {
                    Corp::Q => &f.q[c],
                    Corp::Subj => &f.subj[c],
                    Corp::Para => &f.para[c],
                    Corp::QPara => &f.qpara[c],
                    Corp::QThenPara => {
                        if t < switch { &f.q[c] } else { &f.para[c] }
                    }
                };
                let idx = texts[r.below(texts.len())];
                emb.push(vec_of(idx));
                ctx.push(c);
                wt.push(sp.w);
                truth.push(c as i32);
                topic.push(f.topic[idx]);
            }
        }
        let draws = (0..n).map(|_| (r.f64(), r.f64(), r.f64())).collect();
        CtxWorld { w, emb, pctx, ctx, wt, truth, topic, draws, switch }
    }
    pub fn p(&self, t: usize, a: usize) -> f64 {
        if self.w.is_down(t, a) {
            return 0.0;
        }
        let ty = self.w.task(t).type_idx;
        let base = self.w.skill(t, ty, a);
        let wg = self.wt[t];
        if wg == 0.0 { base } else { (1.0 - wg) * base + wg * self.pctx[self.ctx[t]][a] }
    }
    pub fn outcome(&self, t: usize, a: usize) -> f64 {
        if self.w.task(t).u < self.p(t, a) { 1.0 } else { 0.0 }
    }
}

/// the world list: f5's mini/hash worlds + paraphrase-robustness worlds
pub fn worlds() -> Vec<Spec> {
    use Scenario::{Fixed, Random};
    let mut v = vec![];
    let mut mk = |name: String, src: Src, corp: Corp, sc: Scenario, k: usize, w: f64, mixed: f64, tr: bool| {
        v.push(Spec { name, src, corp, scenario: sc, k, w, mixed, treatment: tr, targets: vec![], win: 200 });
    };
    for src in [Src::Mini, Src::Hash] {
        let p = src.label();
        mk(format!("{p}-q"), src, Corp::Q, Fixed, 12, 0.8, 0.0, true);
        mk(format!("{p}-subj3"), src, Corp::Subj, Fixed, 12, 0.8, 0.0, true);
        mk(format!("{p}-q-mild"), src, Corp::Q, Random, 12, 0.5, 0.0, true);
        mk(format!("{p}-q-mixed50"), src, Corp::Q, Fixed, 12, 0.8, 0.5, true);
        mk(format!("{p}-q-k6"), src, Corp::Q, Fixed, 6, 0.8, 0.0, true);
        mk(format!("{p}-para"), src, Corp::Para, Fixed, 12, 0.8, 0.0, true);
        mk(format!("{p}-qpara"), src, Corp::QPara, Fixed, 12, 0.8, 0.0, true);
        mk(format!("{p}-q2para"), src, Corp::QThenPara, Fixed, 12, 0.8, 0.0, true);
        mk(format!("{p}-ctl-irrel"), src, Corp::Q, Fixed, 12, 0.0, 0.0, false);
        mk(format!("{p}-ctl-unique"), src, Corp::Q, Fixed, 0, 0.8, 0.0, false);
        mk(format!("{p}-ctl-none"), src, Corp::Q, Random, 0, 0.0, 0.0, false);
    }
    v
}
pub fn spec(name: &str) -> Spec {
    worlds().into_iter().find(|s| s.name == name).unwrap_or_else(|| panic!("no world {name}"))
}

// ---------------------------------------------------------------------------------------------
// causal online mean-centering (identical to f5 / e2)

pub struct Centerer {
    mu: Vec<f32>,
    n: f64,
}
impl Centerer {
    pub fn new(dim: usize) -> Self {
        Centerer { mu: vec![0.0; dim], n: 0.0 }
    }
    pub fn apply(&mut self, e: &[f32]) -> Vec<f32> {
        let v: Vec<f32> = if self.n > 0.0 { e.iter().zip(&self.mu).map(|(x, m)| x - m).collect() } else { e.to_vec() };
        let nn = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let alpha = (1.0 / (self.n + 1.0)).max(0.01) as f32;
        for (m, x) in self.mu.iter_mut().zip(e) {
            *m += alpha * (x - *m);
        }
        self.n += 1.0;
        v.into_iter().map(|x| x / nn).collect()
    }
}

// ---------------------------------------------------------------------------------------------
// anchor methods

pub trait Anchorer {
    /// Anchor id of this query (stored) and whether it founded a NEW anchor (nothing similar enough; the f5 pipeline
    /// injects NO `kn:` feature for such a task). Ids are stable feature keys; merged anchors report the surviving root.
    fn assign(&mut self, q: &[f32]) -> (u64, bool);
    /// current surviving root of an id handed out earlier (identity when nothing merges)
    fn root(&self, a: u64) -> u64 {
        a
    }
    fn n_anchors(&self) -> usize;
}

#[derive(Clone, Debug, PartialEq)]
pub enum AnchorCfg {
    /// the library's leader/vote clustering (= f5): top-k neighbours >= sim vote for an anchor, else a new one
    Leader { k: usize, sim: f32 },
    /// anchors carry a centroid; a query joins the best centroid with cos >= tau_a, else founds one;
    /// optional agglomerative merge of anchor pairs whose centroids have cos >= tau_m
    /// `merge_min`: a merge needs both anchors to hold >= merge_min members (1 = no restriction);
    /// `knn`: optional second criterion (k, tau_nn): the query also votes among its top-k stored neighbours >= tau_nn;
    /// when the kNN vote and the best centroid point to DIFFERENT anchors the two are unioned (single evidence from two views)
    Centroid { tau_a: f32, tau_m: Option<f32>, merge_min: u32, knn: Option<(usize, f32)>, avg: bool },
    /// kNN leader (k neighbours >= tau_n vote) plus bridge-merge: two anchors that each hold >= support of the
    /// neighbours >= tau_m of the same query are unioned
    Bridge { k: usize, tau_n: f32, tau_m: f32, support: usize },
    /// ORACLE (headroom bound, not a method): the ground-truth concept is the anchor; one-off tasks get unique anchors
    Oracle,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Method {
    pub anchor: AnchorCfg,
    pub center: bool,
}
impl Method {
    pub fn label(&self) -> String {
        let c = if self.center { "+ctr" } else { "" };
        match &self.anchor {
            AnchorCfg::Leader { k, sim } => format!("leader(k{k} s{sim}){c}"),
            AnchorCfg::Centroid { tau_a, tau_m, merge_min, knn, avg } => {
                let base = if *avg { "avglink" } else if tau_m.is_none() && knn.is_none() { "centroid" } else if knn.is_some() { "cen+knn" } else { "centroid+merge" };
                let m = tau_m.map(|m| format!(" m{m}")).unwrap_or_default();
                let mm = if *merge_min > 1 { format!(" mm{merge_min}") } else { String::new() };
                let kn = knn.map(|(k, t)| format!(" nn(k{k} {t})")).unwrap_or_default();
                format!("{base}(a{tau_a}{m}{mm}{kn}){c}")
            }
            AnchorCfg::Bridge { k, tau_n, tau_m, support } => format!("bridge(k{k} n{tau_n} m{tau_m} s{support}){c}"),
            AnchorCfg::Oracle => format!("oracle{c}"),
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
    fn n_anchors(&self) -> usize {
        0
    }
}

struct Centroid {
    tau_a: f32,
    tau_m: Option<f32>,
    merge_min: u32,
    knn: Option<(usize, f32)>,
    avg: bool,
    items: Vec<Vec<f32>>,
    item_anchor: Vec<usize>,
    sum: Vec<Vec<f32>>,
    unit: Vec<Vec<f32>>,
    n: Vec<u32>,
    parent: Vec<usize>,
    alive: Vec<bool>,
}
impl Centroid {
    fn find(&self, mut i: usize) -> usize {
        while self.parent[i] != i {
            i = self.parent[i];
        }
        i
    }
    fn renorm(&mut self, i: usize) {
        let nn = dot(&self.sum[i], &self.sum[i]).sqrt().max(1e-9);
        self.unit[i] = self.sum[i].iter().map(|x| x / nn).collect();
    }
}
impl Centroid {
    fn merge(&mut self, a: usize, j: usize) -> usize {
        let (keep, gone) = if self.n[a] >= self.n[j] { (a, j) } else { (j, a) };
        let g = std::mem::take(&mut self.sum[gone]);
        for (x, y) in self.sum[keep].iter_mut().zip(&g) {
            *x += y;
        }
        self.n[keep] += self.n[gone];
        self.alive[gone] = false;
        self.parent[gone] = keep;
        self.renorm(keep);
        keep
    }
}
impl Anchorer for Centroid {
    fn assign(&mut self, q: &[f32]) -> (u64, bool) {
        // view 1: kNN vote over stored items
        let mut lroot: Option<usize> = None;
        if let Some((k, tn)) = self.knn {
            let mut hits: Vec<(f32, usize)> = vec![];
            for (i, v) in self.items.iter().enumerate() {
                let sc = dot(q, v);
                if sc >= tn {
                    hits.push((sc, i));
                }
            }
            hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            hits.truncate(k);
            let mut votes: Vec<(usize, f32)> = vec![];
            for &(sc, i) in &hits {
                let r = self.find(self.item_anchor[i]);
                match votes.iter_mut().find(|v| v.0 == r) {
                    Some(v) => v.1 += sc,
                    None => votes.push((r, sc)),
                }
            }
            lroot = votes
                .iter()
                .fold(None, |b: Option<(usize, f32)>, &v| match b {
                    Some(x) if x.1 >= v.1 => b,
                    _ => Some(v),
                })
                .map(|v| v.0);
        }
        // view 2: best centroid
        let mut best = (f32::NEG_INFINITY, usize::MAX);
        for i in 0..self.alive.len() {
            if self.alive[i] {
                let sc = if self.avg { dot(q, &self.sum[i]) / self.n[i] as f32 } else { dot(q, &self.unit[i]) };
                if sc > best.0 {
                    best = (sc, i);
                }
            }
        }
        let croot = if best.1 != usize::MAX && best.0 >= self.tau_a { Some(best.1) } else { None };
        let mut created = false;
        let mut a = match (lroot, croot) {
            (Some(l), Some(c)) if l != c => self.merge(l, c),
            (Some(l), _) => l,
            (None, Some(c)) => c,
            (None, None) => {
                created = true;
                let i = self.alive.len();
                self.sum.push(vec![0.0; q.len()]);
                self.unit.push(q.to_vec());
                self.n.push(0);
                self.parent.push(i);
                self.alive.push(true);
                i
            }
        };
        for (x, y) in self.sum[a].iter_mut().zip(q) {
            *x += y;
        }
        self.n[a] += 1;
        self.renorm(a);
        if self.knn.is_some() {
            self.items.push(q.to_vec());
            self.item_anchor.push(a);
        }
        if let Some(tm) = self.tau_m {
            loop {
                let mut bj = (f32::NEG_INFINITY, usize::MAX);
                for j in 0..self.alive.len() {
                    if self.alive[j] && j != a && self.n[j] >= self.merge_min {
                        let sc = if self.avg { dot(&self.sum[a], &self.sum[j]) / (self.n[a] as f32 * self.n[j] as f32) } else { dot(&self.unit[a], &self.unit[j]) };
                        if sc > bj.0 {
                            bj = (sc, j);
                        }
                    }
                }
                if self.n[a] < self.merge_min || bj.1 == usize::MAX || bj.0 < tm {
                    break;
                }
                a = self.merge(a, bj.1);
                created = false;
            }
        }
        (self.find(a) as u64, created)
    }
    fn root(&self, a: u64) -> u64 {
        self.find(a as usize) as u64
    }
    fn n_anchors(&self) -> usize {
        self.alive.iter().filter(|&&x| x).count()
    }
}

struct Bridge {
    k: usize,
    tau_n: f32,
    tau_m: f32,
    support: usize,
    items: Vec<Vec<f32>>,
    item_node: Vec<usize>,
    parent: Vec<usize>,
}
impl Bridge {
    fn find(&self, mut i: usize) -> usize {
        while self.parent[i] != i {
            i = self.parent[i];
        }
        i
    }
}
impl Anchorer for Bridge {
    fn assign(&mut self, q: &[f32]) -> (u64, bool) {
        let lo = self.tau_n.min(self.tau_m);
        let mut hits: Vec<(f32, usize)> = vec![];
        for (i, v) in self.items.iter().enumerate() {
            let s = dot(q, v);
            if s >= lo {
                hits.push((s, i));
            }
        }
        hits.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        hits.truncate(self.k);
        // bridge-merge
        let mut cnt: Vec<(usize, usize)> = vec![];
        for &(s, i) in &hits {
            if s >= self.tau_m {
                let r = self.find(self.item_node[i]);
                match cnt.iter_mut().find(|c| c.0 == r) {
                    Some(c) => c.1 += 1,
                    None => cnt.push((r, 1)),
                }
            }
        }
        let strong: Vec<usize> = cnt.iter().filter(|c| c.1 >= self.support).map(|c| c.0).collect();
        if strong.len() > 1 {
            let keep = *strong.iter().min().unwrap();
            for &r in &strong {
                if r != keep {
                    self.parent[r] = keep;
                }
            }
        }
        // leader vote over neighbours >= tau_n
        let mut votes: Vec<(usize, f32)> = vec![];
        for &(s, i) in &hits {
            if s >= self.tau_n {
                let r = self.find(self.item_node[i]);
                match votes.iter_mut().find(|v| v.0 == r) {
                    Some(v) => v.1 += s,
                    None => votes.push((r, s)),
                }
            }
        }
        let (node, new) = match votes.iter().fold(None, |b: Option<(usize, f32)>, &v| match b {
            Some(x) if x.1 >= v.1 => b,
            _ => Some(v),
        }) {
            Some((r, _)) => (r, false),
            None => {
                let i = self.parent.len();
                self.parent.push(i);
                (i, true)
            }
        };
        self.items.push(q.to_vec());
        self.item_node.push(node);
        (node as u64, new)
    }
    fn root(&self, a: u64) -> u64 {
        self.find(a as usize) as u64
    }
    fn n_anchors(&self) -> usize {
        (0..self.parent.len()).filter(|&i| self.parent[i] == i).count()
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
    fn n_anchors(&self) -> usize {
        self.seen.len()
    }
}

pub fn make_anchorer(a: &AnchorCfg, dim: usize) -> Box<dyn Anchorer> {
    make_anchorer_t(a, dim, &[])
}

pub fn make_anchorer_t(a: &AnchorCfg, dim: usize, truth: &[i32]) -> Box<dyn Anchorer> {
    match a {
        AnchorCfg::Oracle => Box::new(OracleAnchor { truth: truth.to_vec(), t: 0, seen: Default::default() }),
        AnchorCfg::Leader { k, sim } => {
            let mut kc = KnowledgeConfig::new(dim);
            kc.k = *k;
            kc.sim_min = *sim;
            kc.feat = FeatMode::Off;
            Box::new(LeaderKn { kn: Knowledge::new_shared(kc) })
        }
        AnchorCfg::Centroid { tau_a, tau_m, merge_min, knn, avg } => Box::new(Centroid {
            tau_a: *tau_a,
            tau_m: *tau_m,
            merge_min: *merge_min,
            knn: *knn,
            avg: *avg,
            items: vec![],
            item_anchor: vec![],
            sum: vec![],
            unit: vec![],
            n: vec![],
            parent: vec![],
            alive: vec![],
        }),
        AnchorCfg::Bridge { k, tau_n, tau_m, support } => {
            Box::new(Bridge { k: *k, tau_n: *tau_n, tau_m: *tau_m, support: *support, items: vec![], item_node: vec![], parent: vec![] })
        }
    }
}

// ---------------------------------------------------------------------------------------------
// retrieval-layer metrics

#[derive(Clone, Debug, Default)]
pub struct Nbr {
    pub p: f64,
    pub r: f64,
    pub f1: f64,
    pub spurious: f64,
}

fn f1(p: f64, r: f64) -> f64 {
    if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
}

/// neighbour-list P/R/F1 vs ground-truth concept over queries t >= from_t (store = every earlier item)
pub fn nbr_metrics(cw: &CtxWorld, dim: usize, k: usize, sim_min: f32, center: bool, from_t: usize) -> Nbr {
    let n = cw.emb.len();
    let mut kc = KnowledgeConfig::new(dim);
    kc.k = k;
    kc.sim_min = sim_min;
    kc.feat = FeatMode::Off;
    let mut kn = Knowledge::new_shared(kc);
    let mut cen = Centerer::new(dim);
    let (mut tp, mut ret, mut poss) = (0.0, 0.0, 0.0);
    let (mut dq, mut dsp) = (0.0, 0.0);
    let mut same_prev: HashMap<i32, usize> = HashMap::new();
    for t in 0..n {
        let q = if center { cen.apply(&cw.emb[t]) } else { cw.emb[t].clone() };
        let r = kn.retrieve(&q);
        if t > 0 && t >= from_t {
            let kept: Vec<usize> = r.hits.iter().filter(|h| h.1 >= sim_min).map(|h| h.0 as usize).collect();
            if cw.truth[t] >= 0 {
                ret += kept.len() as f64;
                tp += kept.iter().filter(|&&i| cw.truth[i] == cw.truth[t]).count() as f64;
                poss += (*same_prev.get(&cw.truth[t]).unwrap_or(&0)).min(k) as f64;
            } else {
                dq += 1.0;
                if !kept.is_empty() {
                    dsp += 1.0;
                }
            }
        }
        kn.remember(&q, &r);
        if cw.truth[t] >= 0 {
            *same_prev.entry(cw.truth[t]).or_insert(0) += 1;
        }
    }
    let p = if ret > 0.0 { tp / ret } else { 0.0 };
    let rc = if poss > 0.0 { tp / poss } else { 0.0 };
    Nbr { p, r: rc, f1: f1(p, rc), spurious: if dq > 0.0 { dsp / dq } else { f64::NAN } }
}

#[derive(Clone, Debug, Default)]
pub struct Anc {
    pub p: f64,
    pub r: f64,
    pub f1: f64,
    /// distinct anchors among concept tasks / number of concepts (1 = perfect)
    pub frag: f64,
    /// fraction of distractor/unique tasks whose anchor also holds a concept task
    pub contam: f64,
    /// fraction of concept FIRST occurrences that joined an existing anchor (cross-concept absorption)
    pub fh_first: f64,
    /// pair-F1 over the first 300 concept tasks (cold start)
    pub early_f1: f64,
    /// pair-F1 after resolving later merges (final state)
    pub final_f1: f64,
    /// pair-F1 restricted to pairs of tasks at t >= switch (q2para world: unseen phrasings); NaN otherwise
    pub post_f1: f64,
    /// post-switch concept tasks that joined a pre-switch anchor whose pre-switch majority concept is correct / wrong / new anchor
    pub post_ok: f64,
    pub post_wrong: f64,
    pub post_new: f64,
    pub n_anchors: f64,
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

pub fn anchor_metrics(cw: &CtxWorld, dim: usize, m: &Method, post_switch: bool) -> Anc {
    let n = cw.emb.len();
    let mut an = make_anchorer(&m.anchor, dim);
    let mut cen = Centerer::new(dim);
    let mut anchor = vec![0u64; n];
    for t in 0..n {
        let q = if m.center { cen.apply(&cw.emb[t]) } else { cw.emb[t].clone() };
        anchor[t] = an.assign(&q).0;
    }
    let fin: Vec<u64> = anchor.iter().map(|&a| an.root(a)).collect();
    let (p, r, f) = pair_prf(&anchor, &cw.truth, |_| true);
    let (_, _, ff) = pair_prf(&fin, &cw.truth, |_| true);
    // early: first 300 concept tasks
    let mut cnt = 0usize;
    let mut early_set = vec![false; n];
    for t in 0..n {
        if cw.truth[t] >= 0 && cnt < 300 {
            early_set[t] = true;
            cnt += 1;
        }
    }
    let (_, _, ef) = pair_prf(&anchor, &cw.truth, |t| early_set[t]);
    let mut anchors_c: HashMap<u64, usize> = HashMap::new();
    let mut concepts: HashMap<i32, usize> = HashMap::new();
    for t in 0..n {
        if cw.truth[t] >= 0 {
            *anchors_c.entry(anchor[t]).or_insert(0) += 1;
            *concepts.entry(cw.truth[t]).or_insert(0) += 1;
        }
    }
    let (mut dn, mut dc) = (0.0, 0.0);
    for t in 0..n {
        if cw.truth[t] < 0 {
            dn += 1.0;
            if anchors_c.contains_key(&anchor[t]) {
                dc += 1.0;
            }
        }
    }
    // first occurrences joining an existing anchor
    let mut seen_anchor: HashMap<u64, usize> = HashMap::new();
    let mut seen_concept: HashMap<i32, bool> = HashMap::new();
    let (mut fo, mut fj) = (0.0, 0.0);
    for t in 0..n {
        let joined = seen_anchor.contains_key(&anchor[t]);
        if cw.truth[t] >= 0 && !seen_concept.contains_key(&cw.truth[t]) {
            seen_concept.insert(cw.truth[t], true);
            fo += 1.0;
            if joined {
                fj += 1.0;
            }
        }
        *seen_anchor.entry(anchor[t]).or_insert(0) += 1;
    }
    let (mut post_f1, mut post_ok, mut post_wrong, mut post_new) = (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    if post_switch {
        let sw = cw.switch;
        let (_, _, pf) = pair_prf(&anchor, &cw.truth, |t| t >= sw);
        post_f1 = pf;
        // pre-switch majority concept per resolved root (use the roots at switch time approximated by final roots among pre-switch tasks)
        let mut maj: HashMap<u64, HashMap<i32, usize>> = HashMap::new();
        for t in 0..sw {
            if cw.truth[t] >= 0 {
                *maj.entry(fin[t]).or_default().entry(cw.truth[t]).or_insert(0) += 1;
            }
        }
        let (mut tot, mut ok, mut wr, mut nw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for t in sw..n {
            if cw.truth[t] < 0 {
                continue;
            }
            tot += 1.0;
            match maj.get(&fin[t]) {
                Some(m) => {
                    let (bc, _) = m.iter().max_by_key(|e| (*e.1, std::cmp::Reverse(*e.0))).unwrap();
                    if *bc == cw.truth[t] { ok += 1.0 } else { wr += 1.0 }
                }
                None => nw += 1.0,
            }
        }
        post_ok = ok / tot.max(1.0);
        post_wrong = wr / tot.max(1.0);
        post_new = nw / tot.max(1.0);
    }
    Anc {
        p,
        r,
        f1: f,
        frag: anchors_c.len() as f64 / concepts.len().max(1) as f64,
        contam: if dn > 0.0 { dc / dn } else { f64::NAN },
        fh_first: if fo > 0.0 { fj / fo } else { f64::NAN },
        early_f1: ef,
        final_f1: ff,
        post_f1,
        post_ok,
        post_wrong,
        post_new,
        n_anchors: an.n_anchors() as f64,
    }
}

pub fn mean_of(xs: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = xs.filter(|x| x.is_finite()).collect();
    if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
}

// ---------------------------------------------------------------------------------------------
// policies / runs

#[derive(Clone, Debug, Default)]
pub struct RunOut {
    pub h: [f64; 4],
    pub fail: f64,
    /// expected success of the chosen agent per task
    pub pe: Vec<f32>,
    /// cumulative graded outcomes arrived after each task (outcome regime: 1/task; reuse regime: delayed reuse deliveries)
    pub g: Vec<u32>,
    /// per target (Spec.targets): tasks / graded outcomes to the target (censored at the run end) and whether it was reached
    pub tt: Vec<f64>,
    pub gg: Vec<f64>,
    pub rr: Vec<f64>,
}
pub fn col(r: &[RunOut], i: usize) -> Vec<f64> {
    r.iter().map(|x| x.h[i]).collect()
}

pub trait Pol {
    fn pick(&mut self, t: &Task, emb: &[f32], c: &[usize], rng: &mut Rng) -> usize;
    fn fb(&mut self, t: &Task, a: usize, r: Option<f64>);
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
/// kn-feat (hidden behind Guarded) with a pluggable anchor method: only the inner learner sees `kn:<anchor>`
struct KnPol<R: Router> {
    front: R,
    map: AnchorMap,
    anch: Box<dyn Anchorer>,
    cen: Option<Centerer>,
    order: VecDeque<u64>,
}
impl<R: Router> Pol for KnPol<R> {
    fn pick(&mut self, t: &Task, e: &[f32], c: &[usize], rng: &mut Rng) -> usize {
        let (a, is_new) = match &mut self.cen {
            Some(cn) => {
                let q = cn.apply(e);
                self.anch.assign(&q)
            }
            None => self.anch.assign(e),
        };
        if !is_new {
            self.map.borrow_mut().insert(t.id, format!("kn:{a}"));
        }
        self.order.push_back(t.id);
        while self.order.len() > 512 {
            if let Some(id) = self.order.pop_front() {
                self.map.borrow_mut().remove(&id);
            }
        }
        self.front.select(t, c, rng)
    }
    fn fb(&mut self, t: &Task, a: usize, r: Option<f64>) {
        if let Some(r) = r {
            self.front.update(t, a, r)
        }
    }
}

/// vote parameters (as knowledge::VoteConfig: prior (1,1), no satisficing exploration)
#[derive(Clone, Debug, PartialEq)]
pub struct VoteCfg {
    pub min_n: f64,
    pub tau: f64,
    /// (min_n, margin, probe_every)
    pub gate: Option<(f64, f64, usize)>,
}
pub fn d6_vote() -> VoteCfg {
    VoteCfg { min_n: 2.0, tau: 0.7, gate: Some((6.0, 0.02, 25)) }
}
pub fn f5_mini_vote() -> (Method, VoteCfg) {
    (Method { anchor: AnchorCfg::Leader { k: 5, sim: 0.5 }, center: true }, VoteCfg { min_n: 3.0, tau: 0.7, gate: Some((6.0, 0.02, 25)) })
}
pub fn f5_hash_vote() -> (Method, VoteCfg) {
    (Method { anchor: AnchorCfg::Leader { k: 8, sim: 0.8 }, center: false }, VoteCfg { min_n: 3.0, tau: 0.7, gate: Some((6.0, 0.02, 25)) })
}

/// kn-direct + vote (= KnowledgeRouter in FeatMode::Direct with VoteConfig) with a pluggable anchor method
struct VotePol<R: Router> {
    front: R,
    anch: Box<dyn Anchorer>,
    cen: Option<Centerer>,
    vc: VoteCfg,
    /// anchor -> per-agent (tries, reward sum, rewards)
    astats: HashMap<u64, Vec<(f64, f64, f64)>>,
    gate: [f64; 4],
    probe: usize,
    /// task id -> (anchor, is_new, agent, by_vote)
    live: HashMap<u64, (u64, bool, usize, bool)>,
    order: VecDeque<u64>,
}
impl<R: Router> VotePol<R> {
    fn vote(&self, a: u64, cands: &[usize]) -> Option<usize> {
        let row = self.astats.get(&a)?;
        let mut best: Option<(usize, f64)> = None;
        for &c in cands {
            let Some(&(_, s, n)) = row.get(c) else { continue };
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
        let (a, is_new) = match &mut self.cen {
            Some(cn) => {
                let q = cn.apply(e);
                self.anch.assign(&q)
            }
            None => self.anch.assign(e),
        };
        // a brand-new anchor has no statistics (the f5 pipeline's retrieval.anchor is None then)
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
        let row = self.astats.entry(a).or_default();
        if row.len() <= agent {
            row.resize(agent + 1, (0.0, 0.0, 0.0));
        }
        row[agent].0 += 1.0;
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
            if bv {
                self.gate[0] += rr;
                self.gate[1] += 1.0;
            } else {
                self.gate[2] += rr;
                self.gate[3] += 1.0;
            }
            if let Some(row) = self.astats.get_mut(&a) {
                if let Some(cc) = row.get_mut(ag) {
                    cc.1 += rr;
                    cc.2 += 1.0;
                }
            }
        }
        self.front.update(&routed, agent, r);
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
    Kn(Method),
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

fn build(k: &Kind2, w: &World, dim: usize, truth: &[i32]) -> Box<dyn Pol> {
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
            anch: make_anchorer_t(&m.anchor, dim, truth),
            cen: if m.center { Some(Centerer::new(dim)) } else { None },
            vc: vc.clone(),
            astats: HashMap::new(),
            gate: [0.0; 4],
            probe: 0,
            live: HashMap::new(),
            order: VecDeque::new(),
        }),
        Kind2::Kn(m) => {
            let map: AnchorMap = Rc::new(RefCell::new(HashMap::new()));
            let inner = AnchorRouter::new(HebbianRouter::new(NA, heb_robust_cfg()), map.clone());
            Box::new(KnPol {
                front: Guarded::new(inner, kw, NA, x1_guard()),
                map,
                anch: make_anchorer_t(&m.anchor, dim, truth),
                cen: if m.center { Some(Centerer::new(dim)) } else { None },
                order: VecDeque::new(),
            })
        }
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
    let dim = DIM;
    let mut pol = build(k, w, dim, &cw.truth);
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
    for &tg in &sp.targets {
        let (t, g, ok) = to_target(&out, tg, sp.win);
        out.tt.push(t);
        out.gg.push(g);
        out.rr.push(if ok { 1.0 } else { 0.0 });
    }
    out
}

pub fn eval(a: &A, sp: &Spec, seeds: &[u64]) -> Vec<RunOut> {
    seeds.par_iter().map(|&s| run(&a.0, sp, a.1, s)).collect()
}

// ---------------------------------------------------------------------------------------------
// cells / Holm families (generic over the metric columns)

pub struct Cell {
    pub world: String,
    pub cand: String,
    pub col: String,
    pub diff: f64,
    pub lo: f64,
    pub hi: f64,
    pub wins: usize,
    pub p: f64,
    pub best_base: String,
    pub cand_mean: f64,
    pub base_mean: f64,
    pub treat: bool,
}

/// `cols` = column names, `ex` extracts one value per column from a run. `higher_better` decides the
/// best baseline per column (and which sign is a "win").
#[allow(clippy::too_many_arguments)]
pub fn family(
    specs: &[Spec],
    bases: &[(String, A)],
    cands: &[(String, A)],
    seeds: &[u64],
    cols: &[String],
    ex: &(dyn Fn(&RunOut) -> Vec<f64> + Sync),
    higher_better: bool,
    show: bool,
    tag: &str,
    tables: &mut Vec<Value>,
) -> Vec<Cell> {
    let mut cells = vec![];
    let colvals = |r: &[RunOut], ci: usize| -> Vec<f64> { r.iter().map(|x| ex(x)[ci]).collect() };
    for sp in specs {
        let br: Vec<Vec<RunOut>> = bases.iter().map(|(_, a)| eval(a, sp, seeds)).collect();
        let cr: Vec<Vec<RunOut>> = cands.iter().map(|(_, a)| eval(a, sp, seeds)).collect();
        if show {
            println!("\n=== [{tag}] world {} | {} ===", sp.name, cols.join(" | "));
        }
        let mut rows = vec![];
        for ((name, _), r) in bases.iter().zip(&br).chain(cands.iter().zip(&cr)) {
            let m: Vec<f64> = (0..cols.len()).map(|i| summarize(&colvals(r, i)).mean).collect();
            if show {
                print!("{:<36}", name);
                for v in &m {
                    print!(" {:>9.4}", v);
                }
                println!();
            }
            rows.push(json!({"policy": name, "cols": m}));
        }
        tables.push(json!({"regime": tag, "world": sp.name, "cols": cols, "rows": rows}));
        for (ci, cn) in cols.iter().enumerate() {
            let means: Vec<f64> = br.iter().map(|r| summarize(&colvals(r, ci)).mean).collect();
            let mut bi = 0;
            for i in 1..means.len() {
                if (higher_better && means[i] > means[bi]) || (!higher_better && means[i] < means[bi]) {
                    bi = i;
                }
            }
            let b = colvals(&br[bi], ci);
            for (cidx, (cname, _)) in cands.iter().enumerate() {
                let c = colvals(&cr[cidx], ci);
                let d = paired_diff(&c, &b);
                cells.push(Cell {
                    world: sp.name.clone(),
                    cand: cname.clone(),
                    col: cn.clone(),
                    diff: d.diff.mean,
                    lo: d.diff.lo,
                    hi: d.diff.hi,
                    wins: d.wins,
                    p: paired_t_pvalue(&c, &b),
                    best_base: bases[bi].0.clone(),
                    cand_mean: summarize(&c).mean,
                    base_mean: summarize(&b).mean,
                    treat: sp.treatment,
                });
            }
        }
    }
    cells
}

pub fn report(title: &str, cells: &[Cell], n_seeds: usize, higher_better: bool) -> (Value, Vec<f64>) {
    let p: Vec<f64> = cells.iter().map(|c| c.p).collect();
    let adj = holm_adjust(&p);
    let mut out = vec![];
    let (mut nw, mut nl) = (0, 0);
    for (c, a) in cells.iter().zip(&adj) {
        if *a < 0.05 {
            if (c.diff > 0.0) == higher_better { nw += 1 } else { nl += 1 }
        }
        out.push(json!({"world": c.world, "treatment": c.treat, "candidate": c.cand, "col": c.col, "diff": c.diff, "ci": [c.lo, c.hi],
            "wins": c.wins, "n_seeds": n_seeds, "p": c.p, "holm_p": a, "best_baseline": c.best_base, "cand_mean": c.cand_mean, "base_mean": c.base_mean}));
    }
    println!("### {title}  (Holm over {} tests) -> Holm WINS {nw}, Holm LOSSES {nl}", cells.len());
    (json!({"title": title, "holm_wins": nw, "holm_losses": nl, "cells": out}), adj)
}

/// compact matrix for one column: rows worlds, columns candidates; "*" Holm win, "!" Holm loss
pub fn matrix(title: &str, cells: &[Cell], adj: &[f64], col: &str, higher_better: bool) {
    let mut worlds: Vec<String> = vec![];
    let mut cands: Vec<String> = vec![];
    for c in cells {
        if !worlds.contains(&c.world) {
            worlds.push(c.world.clone());
        }
        if !cands.contains(&c.cand) {
            cands.push(c.cand.clone());
        }
    }
    println!("\n--- {title} | {col} diff (* Holm WIN, ! Holm LOSS) ---");
    print!("{:<20}", "world");
    for c in &cands {
        print!(" {:>14}", &c[..c.len().min(14)]);
    }
    println!();
    for w in &worlds {
        print!("{:<20}", w);
        for cn in &cands {
            match cells.iter().position(|c| &c.world == w && &c.cand == cn && c.col == col) {
                Some(i) => {
                    let m = if adj[i] < 0.05 { if (cells[i].diff > 0.0) == higher_better { '*' } else { '!' } } else { ' ' };
                    print!(" {:>+13.4}{}", cells[i].diff, m);
                }
                None => print!(" {:>14}", "-"),
            }
        }
        println!();
    }
}

/// first task at which the trailing-`win` mean of the expected success reaches `target`; returns (tasks, graded events, reached)
/// censored at the end of the run (tasks = run length, graded = total delivered)
pub fn to_target(r: &RunOut, target: f64, win: usize) -> (f64, f64, bool) {
    let n = r.pe.len();
    let mut s: f64 = r.pe.iter().take(win).map(|&x| x as f64).sum();
    if n >= win && s / win as f64 >= target {
        return (win as f64, r.g[win - 1] as f64, true);
    }
    for t in win..n {
        s += r.pe[t] as f64 - r.pe[t - win] as f64;
        if s / win as f64 >= target {
            return ((t + 1) as f64, r.g[t] as f64, true);
        }
    }
    (n as f64, *r.g.last().unwrap_or(&0) as f64, false)
}

// ---------------------------------------------------------------------------------------------
// run all (spec x arm) once, then derive several comparisons from the same runs

pub struct Runs {
    pub specs: Vec<Spec>,
    pub names: Vec<String>,
    /// [spec][arm][seed]
    pub r: Vec<Vec<Vec<RunOut>>>,
}

pub fn run_all(specs: &[Spec], arms: &[(String, A)], seeds: &[u64]) -> Runs {
    let mut r = vec![];
    for sp in specs {
        let per: Vec<Vec<RunOut>> = arms.iter().map(|(_, a)| eval(a, sp, seeds)).collect();
        r.push(per);
    }
    Runs { specs: specs.to_vec(), names: arms.iter().map(|a| a.0.clone()).collect(), r }
}

pub type Extract<'a> = &'a (dyn Fn(&RunOut) -> Vec<f64> + Sync);

fn colv(rs: &[RunOut], ex: Extract, ci: usize) -> Vec<f64> {
    rs.iter().map(|x| ex(x)[ci]).collect()
}

/// candidates vs the best of `base_idx` (per spec and column); `hb` = higher is better
pub fn cells_vs_best(runs: &Runs, base_idx: &[usize], cand_idx: &[usize], cols: &[String], ex: Extract, hb: bool) -> Vec<Cell> {
    let mut cells = vec![];
    for (si, sp) in runs.specs.iter().enumerate() {
        for (ci, cn) in cols.iter().enumerate() {
            let means: Vec<f64> = base_idx.iter().map(|&b| summarize(&colv(&runs.r[si][b], ex, ci)).mean).collect();
            let mut bi = 0;
            for i in 1..means.len() {
                if (hb && means[i] > means[bi]) || (!hb && means[i] < means[bi]) {
                    bi = i;
                }
            }
            let bidx = base_idx[bi];
            let b = colv(&runs.r[si][bidx], ex, ci);
            for &c in cand_idx {
                cells.push(mk_cell(sp, &runs.names[c], cn, &colv(&runs.r[si][c], ex, ci), &b, &runs.names[bidx]));
            }
        }
    }
    cells
}

/// candidates vs one fixed reference arm
pub fn cells_vs_fixed(runs: &Runs, ref_idx: usize, cand_idx: &[usize], cols: &[String], ex: Extract) -> Vec<Cell> {
    let mut cells = vec![];
    for (si, sp) in runs.specs.iter().enumerate() {
        for (ci, cn) in cols.iter().enumerate() {
            let b = colv(&runs.r[si][ref_idx], ex, ci);
            for &c in cand_idx {
                cells.push(mk_cell(sp, &runs.names[c], cn, &colv(&runs.r[si][c], ex, ci), &b, &runs.names[ref_idx]));
            }
        }
    }
    cells
}

fn mk_cell(sp: &Spec, cand: &str, col: &str, c: &[f64], b: &[f64], base: &str) -> Cell {
    let d = paired_diff(c, b);
    Cell {
        world: sp.name.clone(),
        cand: cand.to_string(),
        col: col.to_string(),
        diff: d.diff.mean,
        lo: d.diff.lo,
        hi: d.diff.hi,
        wins: d.wins,
        p: paired_t_pvalue(c, b),
        best_base: base.to_string(),
        cand_mean: summarize(c).mean,
        base_mean: summarize(b).mean,
        treat: sp.treatment,
    }
}

pub fn print_table(runs: &Runs, cols: &[String], ex: Extract, tag: &str, tables: &mut Vec<Value>) {
    for (si, sp) in runs.specs.iter().enumerate() {
        println!("\n=== [{tag}] world {} | {} ===", sp.name, cols.join(" | "));
        let mut rows = vec![];
        for (ai, name) in runs.names.iter().enumerate() {
            let m: Vec<f64> = (0..cols.len()).map(|ci| summarize(&colv(&runs.r[si][ai], ex, ci)).mean).collect();
            print!("{:<36}", name);
            for v in &m {
                print!(" {:>9.4}", v);
            }
            println!();
            rows.push(json!({"policy": name, "cols": m}));
        }
        tables.push(json!({"regime": tag, "world": sp.name, "cols": cols, "rows": rows}));
    }
}

// ---------------------------------------------------------------------------------------------
// Method <-> json (the tuned selection is handed from f6_anchor to f6_sample)

pub fn method_json(m: &Method) -> Value {
    let a = match &m.anchor {
        AnchorCfg::Leader { k, sim } => json!({"type": "leader", "k": k, "sim": sim}),
        AnchorCfg::Centroid { tau_a, tau_m, merge_min, knn, avg } => {
            json!({"type": "centroid", "tau_a": tau_a, "tau_m": tau_m, "merge_min": merge_min, "knn": knn.map(|(k, t)| json!([k, t])), "avg": avg})
        }
        AnchorCfg::Bridge { k, tau_n, tau_m, support } => json!({"type": "bridge", "k": k, "tau_n": tau_n, "tau_m": tau_m, "support": support}),
        AnchorCfg::Oracle => json!({"type": "oracle"}),
    };
    json!({"anchor": a, "center": m.center, "label": m.label()})
}
pub fn method_from_json(v: &Value) -> Method {
    let a = &v["anchor"];
    let f = |k: &str| a[k].as_f64().unwrap() as f32;
    let anchor = match a["type"].as_str().unwrap() {
        "leader" => AnchorCfg::Leader { k: a["k"].as_u64().unwrap() as usize, sim: f("sim") },
        "centroid" => AnchorCfg::Centroid {
            tau_a: f("tau_a"),
            tau_m: a["tau_m"].as_f64().map(|x| x as f32),
            merge_min: a["merge_min"].as_u64().unwrap_or(1) as u32,
            knn: a["knn"].as_array().map(|x| (x[0].as_u64().unwrap() as usize, x[1].as_f64().unwrap() as f32)),
            avg: a["avg"].as_bool().unwrap_or(false),
        },
        "oracle" => AnchorCfg::Oracle,
        _ => AnchorCfg::Bridge { k: a["k"].as_u64().unwrap() as usize, tau_n: f("tau_n"), tau_m: f("tau_m"), support: a["support"].as_u64().unwrap() as usize },
    };
    Method { anchor, center: v["center"].as_bool().unwrap() }
}

/// the baselines of f5 (leader/vote anchors): d6 config and the configs f5 retuned (mini / hash)
pub fn f5_d6_method() -> Method {
    Method { anchor: AnchorCfg::Leader { k: 8, sim: 0.6 }, center: false }
}
pub fn f5_mini_method() -> Method {
    Method { anchor: AnchorCfg::Leader { k: 8, sim: 0.5 }, center: false }
}
pub fn f5_hash_method() -> Method {
    Method { anchor: AnchorCfg::Leader { k: 5, sim: 0.7 }, center: false }
}

pub fn bases_w() -> Vec<(String, A)> {
    vec![
        ("keyword".into(), (Kind2::Keyword, Signal::Raw)),
        ("eps-greedy".into(), (Kind2::Eps, Signal::Raw)),
        ("pooled-ts".into(), (Kind2::Pts, Signal::Raw)),
        ("hebbian-robust".into(), (Kind2::HebR, Signal::Raw)),
        ("warm(hebr)".into(), (Kind2::Warm, Signal::Raw)),
        ("guarded(hebr)".into(), (Kind2::Guard, Signal::Raw)),
        ("guarded(hebr)+reuse".into(), (Kind2::Guard, Signal::Reuse)),
        ("warm(hebr)+reuse".into(), (Kind2::Warm, Signal::Reuse)),
    ]
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
