//! H22 common: MLP-as-router on existing + REAL-TEXT-SHAPED worlds.
//!
//! Reuses g12_common (worlds, g12 AnnRouter, alloc counter) and g3_common (TOST) read-only via #[path]. New here:
//!   * `Mlp2`   : the g12 net generalised to a real dense input (PCA-whitened 16/64 or the raw 384-d MiniLM vector).
//!   * `Casc2`  : a small CASCADE-CORRELATION-style growing network (the FANN-flavoured variant): output layer with direct
//!                input->output connections ("stage 0" = a per-agent logistic/linear head), hidden units added one at a time, each
//!                trained as a candidate to maximise covariance with the current residual error on a replay buffer, then its input
//!                weights are FROZEN and the output layer keeps learning online (Fahlman & Lebiere 1990; FANN's cascade2 idea).
//!   * `RealW`  : real-text-shaped worlds. Tasks = real sentences (h10 fixture), embedding = REAL MiniLM vector of the sentence,
//!                competence = SIMULATED function of the embedding (cluster-based, or smooth function of random projections).
//!   * `Arena`  : common trait over g12 worlds and RealW; baselines can be given the information through anchor tokens
//!                (`kn:<id>` = nearest of M k-means anchors; the idealised offline version of the repo's knowledge-anchor feature).
//!
//! EVERYTHING outcome-related is a seeded simulation I wrote. It measures ROUTING only; no frozen model gets smarter.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::defaults::recommended_router;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue};
use serde_json::{json, Value};

#[path = "../g12_common/mod.rs"]
pub mod g12;
#[path = "../g3_common/mod.rs"]
pub mod g3;

pub use g12::{summarize_run, RunRes, BIN};

pub fn seeds(lo: u64, hi: u64) -> Vec<u64> {
    (lo..=hi).collect()
}

fn uni(r: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * r.f64()
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

// =============================================================================================
// Dense input source (real embeddings live in a side table indexed by Task::id -> pool row)

pub const MAXD: usize = 384;
pub const NPCA: usize = 64;
const MAXI: usize = 12;
const MAXH2: usize = 64;
const MAXN: usize = 64;
const HM: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Inp {
    None,
    Pca(usize),
    Raw,
}
impl Inp {
    pub fn dim(self) -> usize {
        match self {
            Inp::None => 0,
            Inp::Pca(d) => d,
            Inp::Raw => MAXD,
        }
    }
    pub fn label(self) -> String {
        match self {
            Inp::None => "tok".into(),
            Inp::Pca(d) => format!("pca{d}"),
            Inp::Raw => "raw384".into(),
        }
    }
}

#[derive(Clone)]
pub struct DenseSrc {
    /// n_pool x 64, PCA-whitened (unit variance per coordinate)
    pub pca: Arc<Vec<f32>>,
    /// n_pool x 384, the REAL MiniLM vector x sqrt(384) (unit RMS per coordinate on average)
    pub raw: Arc<Vec<f32>>,
    /// task id -> pool row
    pub row_of: Arc<Vec<u32>>,
}
impl DenseSrc {
    pub fn get(&self, id: u64, inp: Inp) -> &[f32] {
        let r = self.row_of[id as usize] as usize;
        match inp {
            Inp::None => &[],
            Inp::Pca(d) => &self.pca[r * NPCA..r * NPCA + d],
            Inp::Raw => &self.raw[r * MAXD..(r + 1) * MAXD],
        }
    }
}

// =============================================================================================
// Pool (real sentences + embeddings), loaded once per text source

pub struct Pool {
    pub tag: String,
    pub n: usize,
    pub tasks: Vec<Vec<String>>,
    pub emb: Arc<Vec<f32>>,  // unit vectors n x 384
    pub raw: Arc<Vec<f32>>,  // x sqrt(384)
    pub pca: Arc<Vec<f32>>,  // n x 64
    pub clusters: HashMap<usize, Vec<usize>>,
    pub kwtok: HashMap<usize, HashMap<String, usize>>,
    pub anchors: HashMap<usize, Vec<usize>>,
}

pub fn load_pool(tag: &str) -> Arc<Pool> {
    let rd = |n: String| std::fs::read(&n).unwrap_or_else(|e| panic!("{n}: {e} (run py/h22_prep.py)"));
    let j: Value = serde_json::from_slice(&rd(format!("results/h22_pool_{tag}.json"))).unwrap();
    let n = j["n"].as_u64().unwrap() as usize;
    let tasks: Vec<Vec<String>> = j["tasks"].as_array().unwrap().iter().map(|t| t.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()).collect();
    assert_eq!(tasks.len(), n);
    let f32s = |b: Vec<u8>| -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() };
    let emb = f32s(rd(format!("results/h22_emb_{tag}.f32")));
    assert_eq!(emb.len(), n * MAXD);
    let pca = f32s(rd(format!("results/h22_pca_{tag}.f32")));
    assert_eq!(pca.len(), n * NPCA);
    let s = (MAXD as f32).sqrt();
    let raw: Vec<f32> = emb.iter().map(|x| x * s).collect();
    let mut clusters = HashMap::new();
    for (k, v) in j["clusters"].as_object().unwrap() {
        clusters.insert(k.parse().unwrap(), v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect());
    }
    let mut kwtok = HashMap::new();
    for (k, v) in j["kwtok"].as_object().unwrap() {
        kwtok.insert(k.parse().unwrap(), v.as_object().unwrap().iter().map(|(a, b)| (a.clone(), b.as_u64().unwrap() as usize)).collect());
    }
    let mut anchors = HashMap::new();
    for (k, v) in j["anchors"].as_object().unwrap() {
        anchors.insert(k.parse().unwrap(), v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect());
    }
    Arc::new(Pool { tag: tag.to_string(), n, tasks, emb: Arc::new(emb), raw: Arc::new(raw), pca: Arc::new(pca), clusters, kwtok, anchors })
}

// =============================================================================================
// Arena trait: g12 worlds + real-text-shaped worlds

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Var {
    /// the world's own keyword tokens
    Tok,
    /// + `kn:<nearest of 32 anchors>`
    Anc32,
    /// + `kn:<nearest of 128 anchors>`
    Anc128,
    /// + `cl:<true competence cluster>` (cluster worlds only; an UPPER-BOUND control, not observable in a real system)
    Oracle,
}
impl Var {
    pub fn label(self) -> &'static str {
        match self {
            Var::Tok => "tok",
            Var::Anc32 => "+anc32",
            Var::Anc128 => "+anc128",
            Var::Oracle => "+oracle-id",
        }
    }
}

pub trait Arena: Sync {
    fn n_agents(&self) -> usize;
    fn len(&self) -> usize;
    fn task(&self, t: usize, v: Var) -> &Task;
    fn prob(&self, t: usize, a: usize) -> f64;
    fn outcome(&self, t: usize, a: usize) -> f64;
    fn kw_map(&self) -> HashMap<String, usize>;
    fn dense(&self) -> Option<DenseSrc>;
    fn g12env(&self) -> Option<&g12::Env> {
        None
    }
    fn oracle(&self, t: usize) -> f64 {
        (0..self.n_agents()).map(|a| self.prob(t, a)).fold(0.0, f64::max)
    }
}

pub struct Old(pub g12::Env);
impl Arena for Old {
    fn n_agents(&self) -> usize {
        self.0.n_agents()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn task(&self, t: usize, _v: Var) -> &Task {
        self.0.task(t)
    }
    fn prob(&self, t: usize, a: usize) -> f64 {
        self.0.prob(t, a)
    }
    fn outcome(&self, t: usize, a: usize) -> f64 {
        self.0.outcome(t, a)
    }
    fn kw_map(&self) -> HashMap<String, usize> {
        self.0.kw_map()
    }
    fn dense(&self) -> Option<DenseSrc> {
        None
    }
    fn g12env(&self) -> Option<&g12::Env> {
        Some(&self.0)
    }
}

// =============================================================================================
// Real-text-shaped worlds

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RKind {
    /// competence = table over k-means clusters of the sentence embedding (h10 generator: best .80-.90, second .55-.68, others .15-.45; 25% of clusters switch best agent at T/2)
    Cluster { k: usize, agents: usize },
    /// negative control: the competence cluster is drawn independently of the text (embedding and tokens carry no information)
    Shuffled { k: usize, agents: usize },
    /// smooth: skill_a = .5 + .42 tanh(w_a . z + b_a), z = 8 standardised random projections of the REAL embedding
    SmoothLin { agents: usize },
    /// smooth nonlinear: skill_a = .12 + .78 exp(-|z_{0..3} - c_a|^2 / (2*1.2^2)), centres at random pool points
    SmoothRbf { agents: usize },
    /// SmoothRbf with the agents' centres cyclically shifted at T/2
    SmoothRbfDrift { agents: usize },
}
impl RKind {
    pub fn agents(self) -> usize {
        match self {
            RKind::Cluster { agents, .. } | RKind::Shuffled { agents, .. } | RKind::SmoothLin { agents } | RKind::SmoothRbf { agents } | RKind::SmoothRbfDrift { agents } => agents,
        }
    }
    pub fn name(self) -> String {
        match self {
            RKind::Cluster { k, agents } => format!("clu{k}-{agents}ag"),
            RKind::Shuffled { k, agents } => format!("ctl-shuf{k}-{agents}ag"),
            RKind::SmoothLin { agents } => format!("smlin-{agents}ag"),
            RKind::SmoothRbf { agents } => format!("smrbf-{agents}ag"),
            RKind::SmoothRbfDrift { agents } => format!("smrbf-drift-{agents}ag"),
        }
    }
    pub fn family(self) -> &'static str {
        match self {
            RKind::Cluster { .. } => "cluster",
            RKind::Shuffled { .. } => "control",
            RKind::SmoothLin { .. } => "smooth-lin",
            RKind::SmoothRbf { .. } | RKind::SmoothRbfDrift { .. } => "smooth-rbf",
        }
    }
    pub fn is_control(self) -> bool {
        matches!(self, RKind::Shuffled { .. })
    }
    pub fn has_oracle_id(self) -> bool {
        matches!(self, RKind::Cluster { .. })
    }
}

#[derive(Clone, Debug)]
pub struct RealSpec {
    pub tag: &'static str,
    pub kind: RKind,
}
impl RealSpec {
    pub fn label(&self) -> String {
        format!("{}-{}", self.tag, self.kind.name())
    }
}

pub fn real_specs() -> Vec<RealSpec> {
    let mut v = vec![];
    for tag in ["A", "B"] {
        for kind in [
            RKind::Cluster { k: 12, agents: 6 },
            RKind::Cluster { k: 48, agents: 32 },
            RKind::SmoothLin { agents: 6 },
            RKind::SmoothRbf { agents: 8 },
            RKind::SmoothRbfDrift { agents: 8 },
            RKind::Shuffled { k: 12, agents: 6 },
        ] {
            v.push(RealSpec { tag, kind });
        }
    }
    v
}

pub struct RealW {
    pub pool: Arc<Pool>,
    pub kind: RKind,
    n_agents: usize,
    t_tasks: usize,
    tok: Vec<Task>,
    anc32: Vec<Task>,
    anc128: Vec<Task>,
    orc: Vec<Task>,
    row_of: Arc<Vec<u32>>,
    u: Vec<f64>,
    /// competence tables: cluster worlds [cluster][agent]; smooth worlds [pool row][agent]
    p_pre: Vec<f32>,
    p_post: Vec<f32>,
    /// per task: cluster id (cluster worlds) or pool row (smooth worlds)
    key: Vec<u32>,
    kw: HashMap<String, usize>,
}

impl RealW {
    pub fn new(pool: &Arc<Pool>, kind: RKind, seed: u64, t_tasks: usize) -> RealW {
        let n = kind.agents();
        let salt: u64 = match kind {
            RKind::Cluster { k, .. } => 0x100 + k as u64,
            RKind::Shuffled { k, .. } => 0x200 + k as u64,
            RKind::SmoothLin { .. } => 0x300,
            RKind::SmoothRbf { .. } => 0x400,
            RKind::SmoothRbfDrift { .. } => 0x400, // same hidden draws as SmoothRbf
        };
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x22_22_22 ^ (salt << 24) ^ (n as u64) << 40 ^ if pool.tag == "A" { 1 } else { 2 });
        let np = pool.n;
        let mut kw: HashMap<String, usize> = HashMap::new();
        let (p_pre, p_post);
        match kind {
            RKind::Cluster { k, .. } | RKind::Shuffled { k, .. } => {
                let kwtok = &pool.kwtok[&k];
                let mut pp = vec![0.0f32; k * n];
                let mut best_of = vec![0usize; k];
                for c in 0..k {
                    for a in 0..n {
                        pp[c * n + a] = uni(&mut r, 0.15, 0.45) as f32;
                    }
                    let best = r.below(n);
                    pp[c * n + best] = uni(&mut r, 0.80, 0.90) as f32;
                    let mut second = r.below(n - 1);
                    if second >= best {
                        second += 1;
                    }
                    pp[c * n + second] = uni(&mut r, 0.55, 0.68) as f32;
                    best_of[c] = best;
                }
                let mut pq = pp.clone();
                for c in 0..k {
                    if r.bernoulli(0.25) {
                        let old = best_of[c];
                        let mut nw = r.below(n - 1);
                        if nw >= old {
                            nw += 1;
                        }
                        pq[c * n + old] = uni(&mut r, 0.30, 0.50) as f32;
                        pq[c * n + nw] = uni(&mut r, 0.80, 0.90) as f32;
                    }
                }
                let mut toks: Vec<(&String, &usize)> = kwtok.iter().collect();
                toks.sort();
                for (t, &c) in toks {
                    let a = if r.bernoulli(0.65) {
                        best_of[c]
                    } else {
                        let mut w = r.below(n - 1);
                        if w >= best_of[c] {
                            w += 1;
                        }
                        w
                    };
                    kw.insert(t.clone(), a);
                }
                p_pre = pp;
                p_post = pq;
            }
            _ => {
                // 8 standardised random projections of the REAL embedding
                const NZ: usize = 8;
                let mut dirs = vec![0.0f32; NZ * MAXD];
                for d in dirs.iter_mut() {
                    // Box-Muller
                    let (u1, u2) = (r.f64().max(1e-12), r.f64());
                    *d = ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
                }
                let mut z = vec![0.0f32; np * NZ];
                for i in 0..np {
                    let e = &pool.emb[i * MAXD..(i + 1) * MAXD];
                    for j in 0..NZ {
                        let dj = &dirs[j * MAXD..(j + 1) * MAXD];
                        let mut s = 0.0f32;
                        for k in 0..MAXD {
                            s += dj[k] * e[k];
                        }
                        z[i * NZ + j] = s;
                    }
                }
                for j in 0..NZ {
                    let m = (0..np).map(|i| z[i * NZ + j] as f64).sum::<f64>() / np as f64;
                    let sd = ((0..np).map(|i| (z[i * NZ + j] as f64 - m).powi(2)).sum::<f64>() / np as f64).sqrt().max(1e-9);
                    for i in 0..np {
                        z[i * NZ + j] = ((z[i * NZ + j] as f64 - m) / sd) as f32;
                    }
                }
                let mut pp = vec![0.0f32; np * n];
                let mut pq = vec![0.0f32; np * n];
                match kind {
                    RKind::SmoothLin { .. } => {
                        let mut w = vec![vec![0.0f64; NZ]; n];
                        let mut b = vec![0.0f64; n];
                        for a in 0..n {
                            for j in 0..NZ {
                                w[a][j] = uni(&mut r, -0.8, 0.8);
                            }
                            b[a] = uni(&mut r, -0.4, 0.4);
                        }
                        for i in 0..np {
                            for a in 0..n {
                                let s: f64 = (0..NZ).map(|j| w[a][j] * z[i * NZ + j] as f64).sum::<f64>() + b[a];
                                let p = (0.5 + 0.42 * s.tanh()).clamp(0.02, 0.97) as f32;
                                pp[i * n + a] = p;
                                pq[i * n + a] = p;
                            }
                        }
                    }
                    _ => {
                        let mut cen = vec![[0.0f64; 3]; n];
                        for a in 0..n {
                            let row = r.below(np);
                            for j in 0..3 {
                                cen[a][j] = z[row * NZ + j] as f64;
                            }
                        }
                        for i in 0..np {
                            for a in 0..n {
                                let sk = |c: &[f64; 3]| -> f32 {
                                    let d2: f64 = (0..3).map(|j| (z[i * NZ + j] as f64 - c[j]).powi(2)).sum();
                                    (0.12 + 0.78 * (-d2 / (2.0 * 1.2 * 1.2)).exp()).clamp(0.02, 0.97) as f32
                                };
                                pp[i * n + a] = sk(&cen[a]);
                                pq[i * n + a] = if matches!(kind, RKind::SmoothRbfDrift { .. }) { sk(&cen[(a + 1) % n]) } else { pp[i * n + a] };
                            }
                        }
                    }
                }
                // informed marginal keyword prior: per token, the agent with best AVERAGE (pre-drift) competence over pool sentences carrying it
                let mut acc: HashMap<&str, (Vec<f64>, f64)> = HashMap::new();
                for i in 0..np {
                    for f in &pool.tasks[i] {
                        let e = acc.entry(f.as_str()).or_insert_with(|| (vec![0.0; n], 0.0));
                        for a in 0..n {
                            e.0[a] += pp[i * n + a] as f64;
                        }
                        e.1 += 1.0;
                    }
                }
                for (f, (s, c)) in acc {
                    if c < 3.0 {
                        continue;
                    }
                    let mut best = 0;
                    for a in 1..n {
                        if s[a] > s[best] + 1e-9 {
                            best = a;
                        }
                    }
                    kw.insert(f.to_string(), best);
                }
                p_pre = pp;
                p_post = pq;
            }
        }
        let kclu: Option<&Vec<usize>> = match kind {
            RKind::Cluster { k, .. } | RKind::Shuffled { k, .. } => Some(&pool.clusters[&k]),
            _ => None,
        };
        let a32 = &pool.anchors[&32];
        let a128 = &pool.anchors[&128];
        let mut tok = Vec::with_capacity(t_tasks);
        let mut anc32 = Vec::with_capacity(t_tasks);
        let mut anc128 = Vec::with_capacity(t_tasks);
        let mut orc = Vec::with_capacity(t_tasks);
        let mut row_of = Vec::with_capacity(t_tasks);
        let mut u = Vec::with_capacity(t_tasks);
        let mut key = Vec::with_capacity(t_tasks);
        for id in 0..t_tasks {
            let ti = r.below(np);
            let f = pool.tasks[ti].clone();
            let mk = |extra: Option<String>| {
                let mut g = f.clone();
                if let Some(e) = extra {
                    g.push(e);
                }
                Task { id: id as u64, task_type: None, features: g }
            };
            tok.push(mk(None));
            anc32.push(mk(Some(format!("kn:{}", a32[ti]))));
            anc128.push(mk(Some(format!("kn:{}", a128[ti]))));
            let kk = match kind {
                RKind::Cluster { .. } => kclu.unwrap()[ti] as u32,
                RKind::Shuffled { k, .. } => r.below(k) as u32,
                _ => ti as u32,
            };
            key.push(kk);
            orc.push(mk(if kind.has_oracle_id() { Some(format!("cl:{kk}")) } else { None }));
            row_of.push(ti as u32);
            u.push(r.f64());
        }
        RealW { pool: pool.clone(), kind, n_agents: n, t_tasks, tok, anc32, anc128, orc, row_of: Arc::new(row_of), u, p_pre, p_post, key, kw }
    }
}

impl Arena for RealW {
    fn n_agents(&self) -> usize {
        self.n_agents
    }
    fn len(&self) -> usize {
        self.t_tasks
    }
    fn task(&self, t: usize, v: Var) -> &Task {
        match v {
            Var::Tok => &self.tok[t],
            Var::Anc32 => &self.anc32[t],
            Var::Anc128 => &self.anc128[t],
            Var::Oracle => &self.orc[t],
        }
    }
    fn prob(&self, t: usize, a: usize) -> f64 {
        let tab = if t < self.t_tasks / 2 { &self.p_pre } else { &self.p_post };
        tab[self.key[t] as usize * self.n_agents + a] as f64
    }
    fn outcome(&self, t: usize, a: usize) -> f64 {
        if self.u[t] < self.prob(t, a) { 1.0 } else { 0.0 }
    }
    fn kw_map(&self) -> HashMap<String, usize> {
        self.kw.clone()
    }
    fn dense(&self) -> Option<DenseSrc> {
        Some(DenseSrc { pca: self.pool.pca.clone(), raw: self.pool.raw.clone(), row_of: self.row_of.clone() })
    }
}

// =============================================================================================
// Net 1: MLP with dense input (g12 AnnRouter generalised)

#[derive(Clone, Copy, Debug)]
pub struct MlpCfg {
    pub inp: Inp,
    pub dh: usize,
    pub h: usize,
    pub lr: f32,
    pub tau0: f32,
    pub tau_min: f32,
    pub half: f32,
    pub replay_k: usize,
    pub replay_cap: usize,
    pub use_kw: bool,
}
impl MlpCfg {
    pub fn show(&self) -> String {
        format!("mlp[{}{} H{} lr{} tau{}->{} half{} rk{} dh{}]", self.inp.label(), if self.use_kw { "+kw" } else { "" }, self.h, self.lr, self.tau0, self.tau_min, self.half, self.replay_k, self.dh)
    }
}

#[derive(Clone, Copy)]
struct Sample2 {
    id: u64,
    kwa: [u8; MAXI],
    nk: u8,
    idx: [u16; MAXI],
    ni: u8,
    a: u8,
    r: f32,
}

fn sample_of(task: &Task, dh: usize, kw: &Option<HashMap<String, usize>>, a: usize, r: f64) -> Sample2 {
    let mut s = Sample2 { id: task.id, kwa: [0; MAXI], nk: 0, idx: [0; MAXI], ni: 0, a: a as u8, r: r as f32 };
    for f in &task.features {
        if let Some(m) = kw {
            if let Some(&ka) = m.get(f) {
                if (s.nk as usize) < MAXI {
                    s.kwa[s.nk as usize] = ka as u8;
                    s.nk += 1;
                }
            }
        }
        let i = (fnv(f) % dh as u64) as u16;
        if (s.ni as usize) < MAXI && !s.idx[..s.ni as usize].contains(&i) {
            s.idx[s.ni as usize] = i;
            s.ni += 1;
        }
    }
    s
}

fn load_dense(src: &Option<DenseSrc>, inp: Inp, id: u64, buf: &mut [f32; MAXD]) -> usize {
    match src {
        Some(d) if inp != Inp::None => {
            let x = d.get(id, inp);
            buf[..x.len()].copy_from_slice(x);
            x.len()
        }
        _ => 0,
    }
}

struct Net2 {
    dh: usize,
    dd: usize,
    h: usize,
    n: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    skip: f32,
    lrd: f32, // dense lr multiplier
}

impl Net2 {
    fn new(dh: usize, dd: usize, h: usize, n: usize, rng: &mut Rng) -> Net2 {
        let rows = dh + dd;
        let ad = if dd > 0 { 3.0f64.sqrt() * 0.7 / (dd as f64).sqrt() } else { 0.0 };
        let mut w1 = vec![0.0f32; rows * h];
        for r in 0..rows {
            let a = if r < dh { 0.5 } else { ad };
            for k in 0..h {
                w1[r * h + k] = uni(rng, -a, a) as f32;
            }
        }
        let w2 = (0..n * h).map(|_| uni(rng, -0.1, 0.1) as f32).collect();
        Net2 { dh, dd, h, n, w1, b1: vec![0.0; h], w2, b2: vec![0.0; n], skip: 1.0, lrd: 6.0 / (dd.max(6) as f32) }
    }
    fn n_params(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len() + 1
    }
    fn fwd(&self, s: &Sample2, x: &[f32], hid: &mut [f32; MAXH2], z: &mut [f32; MAXN]) {
        let h = self.h;
        hid[..h].copy_from_slice(&self.b1);
        for &i in &s.idx[..s.ni as usize] {
            let row = &self.w1[i as usize * h..(i as usize + 1) * h];
            for k in 0..h {
                hid[k] += row[k];
            }
        }
        for j in 0..self.dd {
            let xj = x[j];
            let row = &self.w1[(self.dh + j) * h..(self.dh + j + 1) * h];
            for k in 0..h {
                hid[k] += row[k] * xj;
            }
        }
        for k in 0..h {
            hid[k] = hid[k].tanh();
        }
        for a in 0..self.n {
            let row = &self.w2[a * h..(a + 1) * h];
            let mut acc = self.b2[a];
            for k in 0..h {
                acc += row[k] * hid[k];
            }
            z[a] = acc;
        }
        if s.nk > 0 {
            let w = self.skip / s.nk as f32;
            for j in 0..s.nk as usize {
                z[s.kwa[j] as usize] += w;
            }
        }
    }
    fn bwd(&mut self, s: &Sample2, x: &[f32], hid: &[f32; MAXH2], dz: &[f32; MAXN], lr: f32) {
        let h = self.h;
        let mut dpre = [0.0f32; MAXH2];
        for a in 0..self.n {
            if dz[a] != 0.0 {
                let row = &self.w2[a * h..(a + 1) * h];
                for k in 0..h {
                    dpre[k] += dz[a] * row[k];
                }
            }
        }
        for k in 0..h {
            dpre[k] *= 1.0 - hid[k] * hid[k];
        }
        for a in 0..self.n {
            if dz[a] != 0.0 {
                let g = lr * dz[a];
                let row = &mut self.w2[a * h..(a + 1) * h];
                for k in 0..h {
                    row[k] -= g * hid[k];
                }
                self.b2[a] -= g;
            }
        }
        if s.nk > 0 {
            let mut g = 0.0f32;
            for j in 0..s.nk as usize {
                g += dz[s.kwa[j] as usize] / s.nk as f32;
            }
            self.skip -= lr * g;
        }
        for k in 0..h {
            self.b1[k] -= lr * dpre[k];
        }
        for &i in &s.idx[..s.ni as usize] {
            let row = &mut self.w1[i as usize * h..(i as usize + 1) * h];
            for k in 0..h {
                row[k] -= lr * dpre[k];
            }
        }
        let lrd = lr * self.lrd;
        for j in 0..self.dd {
            let g = lrd * x[j];
            let row = &mut self.w1[(self.dh + j) * h..(self.dh + j + 1) * h];
            for k in 0..h {
                row[k] -= g * dpre[k];
            }
        }
    }
}

pub struct Mlp2 {
    pub cfg: MlpCfg,
    n: usize,
    net: Net2,
    src: Option<DenseSrc>,
    kw: Option<HashMap<String, usize>>,
    t: usize,
    rng: Rng,
    buf: Vec<Sample2>,
    pos: usize,
}

fn tau_of(t: usize, tau0: f32, tau_min: f32, half: f32) -> f32 {
    tau_min + (tau0 - tau_min) * 0.5f32.powf(t as f32 / half)
}

fn pick_softmax(zp: &[f32], cands: &[usize], tau: f32, rng: &mut Rng) -> usize {
    // zp: logits per agent
    let pm = cands.iter().map(|&c| zp[c]).fold(f32::NEG_INFINITY, f32::max);
    let pm = 1.0 / (1.0 + (-pm).exp());
    let mut w = [0.0f64; MAXN];
    let mut tot = 0.0;
    let tau = tau.max(1e-4) as f64;
    for (i, &c) in cands.iter().enumerate() {
        let p = 1.0 / (1.0 + (-zp[c]).exp());
        w[i] = ((p - pm) as f64 / tau).exp();
        tot += w[i];
    }
    let mut x = rng.f64() * tot;
    for (i, &c) in cands.iter().enumerate() {
        x -= w[i];
        if x < 0.0 {
            return c;
        }
    }
    *cands.last().unwrap()
}

impl Mlp2 {
    pub fn new(n: usize, cfg: MlpCfg, src: Option<DenseSrc>, kw: Option<HashMap<String, usize>>, seed: u64) -> Mlp2 {
        assert!(cfg.h <= MAXH2 && n <= MAXN);
        let inp = if src.is_some() { cfg.inp } else { Inp::None };
        let mut cfg = cfg;
        cfg.inp = inp;
        let mut rng = Rng::new(seed ^ 0xB22_B22);
        let net = Net2::new(cfg.dh, inp.dim(), cfg.h, n, &mut rng);
        let kw = if cfg.use_kw { kw } else { None };
        Mlp2 { cfg, n, net, src, kw, t: 0, rng, buf: vec![], pos: 0 }
    }
    pub fn param_bytes(&self) -> usize {
        self.net.n_params() * 4
    }
    pub fn state_bytes(&self) -> usize {
        self.param_bytes() + self.buf.len() * std::mem::size_of::<Sample2>()
    }
    fn sgd(&mut self, s: &Sample2) {
        let mut xb = [0.0f32; MAXD];
        load_dense(&self.src, self.cfg.inp, s.id, &mut xb);
        let mut hid = [0.0f32; MAXH2];
        let mut z = [0.0f32; MAXN];
        self.net.fwd(s, &xb, &mut hid, &mut z);
        let p = 1.0 / (1.0 + (-z[s.a as usize]).exp());
        let mut dz = [0.0f32; MAXN];
        dz[s.a as usize] = p - s.r;
        self.net.bwd(s, &xb, &hid, &dz, self.cfg.lr);
    }
}

impl Router for Mlp2 {
    fn name(&self) -> String {
        self.cfg.show()
    }
    fn select(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> usize {
        let s = sample_of(task, self.cfg.dh, &self.kw, 0, 0.0);
        let mut xb = [0.0f32; MAXD];
        load_dense(&self.src, self.cfg.inp, task.id, &mut xb);
        let mut hid = [0.0f32; MAXH2];
        let mut z = [0.0f32; MAXN];
        self.net.fwd(&s, &xb, &mut hid, &mut z);
        pick_softmax(&z, cands, tau_of(self.t, self.cfg.tau0, self.cfg.tau_min, self.cfg.half), rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let s = sample_of(task, self.cfg.dh, &self.kw, agent, reward.clamp(0.0, 1.0));
        self.sgd(&s);
        if self.cfg.replay_k > 0 {
            if self.buf.len() < self.cfg.replay_cap {
                self.buf.push(s);
            } else {
                self.buf[self.pos] = s;
                self.pos = (self.pos + 1) % self.cfg.replay_cap;
            }
            for _ in 0..self.cfg.replay_k {
                let j = self.rng.below(self.buf.len());
                let sj = self.buf[j];
                self.sgd(&sj);
            }
        }
        self.t += 1;
    }
}

// =============================================================================================
// Net 2: cascade-correlation-style growing network (FANN-flavoured)

#[derive(Clone, Copy, Debug)]
pub struct CascCfg {
    pub inp: Inp,
    pub dh: usize,
    pub lr: f32,
    pub tau0: f32,
    pub tau_min: f32,
    pub half: f32,
    /// extra output-layer SGD steps per update drawn from the recent-sample buffer
    pub replay_k: usize,
    /// a growth decision every `grow_every` updates; grow if the windowed BCE improved by less than 3% (relative)
    pub grow_every: usize,
    /// max hidden units (0 = never grow = linear head only)
    pub hmax: usize,
    pub use_kw: bool,
    pub buf_cap: usize,
    pub cands: usize,
    pub epochs: usize,
}
impl CascCfg {
    pub fn show(&self) -> String {
        format!("casc[{}{} lr{} half{} rk{} G{} hmax{} dh{}]", self.inp.label(), if self.use_kw { "+kw" } else { "" }, self.lr, self.half, self.replay_k, self.grow_every, self.hmax, self.dh)
    }
}

struct HUnit {
    wt: Vec<f32>,
    wdn: Vec<f32>,
    b: f32,
    wp: Vec<f32>,
}

struct CascNet {
    dh: usize,
    dd: usize,
    n: usize,
    wt: Vec<f32>,  // n x dh   direct token -> output
    wdn: Vec<f32>, // n x dd   direct dense -> output
    b: Vec<f32>,
    hu: Vec<HUnit>,
    u: Vec<f32>, // n x HM
    skip: f32,
    lrd: f32,
}

impl CascNet {
    fn new(dh: usize, dd: usize, n: usize) -> CascNet {
        CascNet { dh, dd, n, wt: vec![0.0; n * dh], wdn: vec![0.0; n * dd], b: vec![0.0; n], hu: vec![], u: vec![0.0; n * HM], skip: 1.0, lrd: 6.0 / (dd.max(6) as f32) }
    }
    fn n_params(&self) -> usize {
        self.wt.len() + self.wdn.len() + self.b.len() + self.u.len().min(self.n * self.hu.len()) + self.hu.iter().map(|h| h.wt.len() + h.wdn.len() + 1 + h.wp.len()).sum::<usize>() + 1
    }
    fn hidden(&self, s: &Sample2, x: &[f32], v: &mut [f32; HM]) {
        for (hi, h) in self.hu.iter().enumerate() {
            let mut a = h.b;
            for &i in &s.idx[..s.ni as usize] {
                a += h.wt[i as usize];
            }
            for j in 0..self.dd {
                a += h.wdn[j] * x[j];
            }
            for g in 0..hi {
                a += h.wp[g] * v[g];
            }
            v[hi] = a.tanh();
        }
    }
    fn logits(&self, s: &Sample2, x: &[f32], v: &mut [f32; HM], z: &mut [f32; MAXN]) {
        self.hidden(s, x, v);
        let nh = self.hu.len();
        for o in 0..self.n {
            let mut a = self.b[o];
            let wt = &self.wt[o * self.dh..(o + 1) * self.dh];
            for &i in &s.idx[..s.ni as usize] {
                a += wt[i as usize];
            }
            let wd = &self.wdn[o * self.dd..(o + 1) * self.dd];
            for j in 0..self.dd {
                a += wd[j] * x[j];
            }
            for h in 0..nh {
                a += self.u[o * HM + h] * v[h];
            }
            z[o] = a;
        }
        if s.nk > 0 {
            let w = self.skip / s.nk as f32;
            for j in 0..s.nk as usize {
                z[s.kwa[j] as usize] += w;
            }
        }
    }
    /// one SGD step on the chosen agent's output (hidden input weights frozen)
    fn step(&mut self, s: &Sample2, x: &[f32], lr: f32) {
        let mut v = [0.0f32; HM];
        let mut z = [0.0f32; MAXN];
        self.logits(s, x, &mut v, &mut z);
        let o = s.a as usize;
        let p = 1.0 / (1.0 + (-z[o]).exp());
        let g = p - s.r;
        let wt = &mut self.wt[o * self.dh..(o + 1) * self.dh];
        for &i in &s.idx[..s.ni as usize] {
            wt[i as usize] -= lr * g;
        }
        let gd = lr * self.lrd * g;
        let wd = &mut self.wdn[o * self.dd..(o + 1) * self.dd];
        for j in 0..self.dd {
            wd[j] -= gd * x[j];
        }
        self.b[o] -= lr * g;
        for h in 0..self.hu.len() {
            self.u[o * HM + h] -= lr * g * v[h];
        }
        if s.nk > 0 {
            let gk = g; // d z[o] / d skip summed over votes for o / nk
            let mut cnt = 0.0f32;
            for j in 0..s.nk as usize {
                if s.kwa[j] as usize == o {
                    cnt += 1.0;
                }
            }
            self.skip -= lr * gk * cnt / s.nk as f32;
        }
    }
}

pub struct Casc2 {
    pub cfg: CascCfg,
    n: usize,
    net: CascNet,
    src: Option<DenseSrc>,
    kw: Option<HashMap<String, usize>>,
    t: usize,
    rng: Rng,
    buf: Vec<Sample2>,
    pos: usize,
    win_loss: f64,
    win_n: usize,
    prev_loss: Option<f64>,
    pub grown: usize,
}

impl Casc2 {
    pub fn new(n: usize, cfg: CascCfg, src: Option<DenseSrc>, kw: Option<HashMap<String, usize>>, seed: u64) -> Casc2 {
        assert!(n <= MAXN && cfg.hmax <= HM);
        let inp = if src.is_some() { cfg.inp } else { Inp::None };
        let mut cfg = cfg;
        cfg.inp = inp;
        let net = CascNet::new(cfg.dh, inp.dim(), n);
        let kw = if cfg.use_kw { kw } else { None };
        Casc2 { cfg, n, net, src, kw, t: 0, rng: Rng::new(seed ^ 0xC33_C33), buf: vec![], pos: 0, win_loss: 0.0, win_n: 0, prev_loss: None, grown: 0 }
    }
    pub fn param_bytes(&self) -> usize {
        self.net.n_params() * 4
    }
    pub fn state_bytes(&self) -> usize {
        self.param_bytes() + self.buf.len() * std::mem::size_of::<Sample2>()
    }
    pub fn hidden_units(&self) -> usize {
        self.net.hu.len()
    }

    /// Candidate phase: train `cands` candidate units by gradient ascent (Adam) on S = sum_agents |cov(v, residual)| over the buffer,
    /// install the best one with ZERO output weights; its input weights are frozen from then on.
    fn grow(&mut self) {
        let m = self.buf.len();
        let (dh, dd, n) = (self.cfg.dh, self.net.dd, self.n);
        let nh = self.net.hu.len();
        // inputs and residuals
        let mut xs = vec![0.0f32; m * dd];
        let mut vprev = vec![0.0f32; m * HM];
        let mut e = vec![0.0f32; m];
        for p in 0..m {
            let s = self.buf[p];
            let mut xb = [0.0f32; MAXD];
            load_dense(&self.src, self.cfg.inp, s.id, &mut xb);
            xs[p * dd..(p + 1) * dd].copy_from_slice(&xb[..dd]);
            let mut v = [0.0f32; HM];
            let mut z = [0.0f32; MAXN];
            self.net.logits(&s, &xb, &mut v, &mut z);
            vprev[p * HM..(p + 1) * HM].copy_from_slice(&v);
            e[p] = s.r - 1.0 / (1.0 + (-z[s.a as usize]).exp());
        }
        // per-agent residual means
        let mut cnt = vec![0.0f32; n];
        let mut esum = vec![0.0f32; n];
        for p in 0..m {
            cnt[self.buf[p].a as usize] += 1.0;
            esum[self.buf[p].a as usize] += e[p];
        }
        let ebar: Vec<f32> = (0..n).map(|o| if cnt[o] > 0.0 { esum[o] / cnt[o] } else { 0.0 }).collect();
        let np_ = dh + dd + 1 + nh; // params: tokens, dense, bias, prev
        let mut best: Option<(f32, Vec<f32>)> = None;
        for _c in 0..self.cfg.cands {
            let ad = if dd > 0 { 3.0f64.sqrt() * 0.7 / (dd as f64).sqrt() } else { 0.0 };
            let mut w = vec![0.0f32; np_];
            for i in 0..dh {
                w[i] = uni(&mut self.rng, -0.5, 0.5) as f32;
            }
            for j in 0..dd {
                w[dh + j] = uni(&mut self.rng, -ad, ad) as f32;
            }
            for g in 0..nh {
                w[dh + dd + 1 + g] = uni(&mut self.rng, -0.3, 0.3) as f32;
            }
            let mut mom = vec![0.0f32; np_];
            let mut vel = vec![0.0f32; np_];
            let mut vv = vec![0.0f32; m];
            let mut score = 0.0f32;
            for ep in 0..=self.cfg.epochs {
                // forward
                for p in 0..m {
                    let s = &self.buf[p];
                    let mut a = w[dh + dd];
                    for &i in &s.idx[..s.ni as usize] {
                        a += w[i as usize];
                    }
                    let x = &xs[p * dd..(p + 1) * dd];
                    for j in 0..dd {
                        a += w[dh + j] * x[j];
                    }
                    for g in 0..nh {
                        a += w[dh + dd + 1 + g] * vprev[p * HM + g];
                    }
                    vv[p] = a.tanh();
                }
                // per-agent covariance
                let mut vs = vec![0.0f32; n];
                for p in 0..m {
                    vs[self.buf[p].a as usize] += vv[p];
                }
                let vbar: Vec<f32> = (0..n).map(|o| if cnt[o] > 0.0 { vs[o] / cnt[o] } else { 0.0 }).collect();
                let mut cov = vec![0.0f32; n];
                for p in 0..m {
                    let o = self.buf[p].a as usize;
                    cov[o] += (vv[p] - vbar[o]) * (e[p] - ebar[o]);
                }
                score = cov.iter().map(|c| c.abs()).sum::<f32>() / m as f32;
                if ep == self.cfg.epochs {
                    break;
                }
                // gradient of S wrt params (ascent), mean over samples
                let mut gr = vec![0.0f32; np_];
                for p in 0..m {
                    let s = &self.buf[p];
                    let o = s.a as usize;
                    let sg = if cov[o] >= 0.0 { 1.0 } else { -1.0 };
                    let dpre = sg * (e[p] - ebar[o]) * (1.0 - vv[p] * vv[p]) / m as f32;
                    for &i in &s.idx[..s.ni as usize] {
                        gr[i as usize] += dpre;
                    }
                    let x = &xs[p * dd..(p + 1) * dd];
                    for j in 0..dd {
                        gr[dh + j] += dpre * x[j];
                    }
                    gr[dh + dd] += dpre;
                    for g in 0..nh {
                        gr[dh + dd + 1 + g] += dpre * vprev[p * HM + g];
                    }
                }
                // Adam ascent
                let (b1, b2, lr) = (0.9f32, 0.999f32, 0.05f32);
                let t = (ep + 1) as i32;
                for k in 0..np_ {
                    mom[k] = b1 * mom[k] + (1.0 - b1) * gr[k];
                    vel[k] = b2 * vel[k] + (1.0 - b2) * gr[k] * gr[k];
                    let mh = mom[k] / (1.0 - b1.powi(t));
                    let vh = vel[k] / (1.0 - b2.powi(t));
                    w[k] += lr * mh / (vh.sqrt() + 1e-8);
                }
            }
            if best.as_ref().map(|b| score > b.0).unwrap_or(true) {
                best = Some((score, w));
            }
        }
        if let Some((_, w)) = best {
            self.net.hu.push(HUnit { wt: w[..dh].to_vec(), wdn: w[dh..dh + dd].to_vec(), b: w[dh + dd], wp: w[dh + dd + 1..].to_vec() });
            self.grown += 1;
        }
    }
}

impl Router for Casc2 {
    fn name(&self) -> String {
        self.cfg.show()
    }
    fn select(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> usize {
        let s = sample_of(task, self.cfg.dh, &self.kw, 0, 0.0);
        let mut xb = [0.0f32; MAXD];
        load_dense(&self.src, self.cfg.inp, task.id, &mut xb);
        let mut v = [0.0f32; HM];
        let mut z = [0.0f32; MAXN];
        self.net.logits(&s, &xb, &mut v, &mut z);
        pick_softmax(&z, cands, tau_of(self.t, self.cfg.tau0, self.cfg.tau_min, self.cfg.half), rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let s = sample_of(task, self.cfg.dh, &self.kw, agent, reward.clamp(0.0, 1.0));
        let mut xb = [0.0f32; MAXD];
        let dd = load_dense(&self.src, self.cfg.inp, s.id, &mut xb);
        debug_assert_eq!(dd, self.net.dd);
        // windowed loss (pre-update prediction)
        {
            let mut v = [0.0f32; HM];
            let mut z = [0.0f32; MAXN];
            self.net.logits(&s, &xb, &mut v, &mut z);
            let p = (1.0 / (1.0 + (-z[agent]).exp())).clamp(1e-4, 1.0 - 1e-4) as f64;
            self.win_loss += -(s.r as f64 * p.ln() + (1.0 - s.r as f64) * (1.0 - p).ln());
            self.win_n += 1;
        }
        self.net.step(&s, &xb, self.cfg.lr);
        if self.buf.len() < self.cfg.buf_cap {
            self.buf.push(s);
        } else {
            self.buf[self.pos] = s;
            self.pos = (self.pos + 1) % self.cfg.buf_cap;
        }
        for _ in 0..self.cfg.replay_k {
            let j = self.rng.below(self.buf.len());
            let sj = self.buf[j];
            let mut xj = [0.0f32; MAXD];
            load_dense(&self.src, self.cfg.inp, sj.id, &mut xj);
            self.net.step(&sj, &xj, self.cfg.lr);
        }
        self.t += 1;
        if self.cfg.hmax > 0 && self.win_n >= self.cfg.grow_every {
            let l = self.win_loss / self.win_n as f64;
            self.win_loss = 0.0;
            self.win_n = 0;
            if let Some(pl) = self.prev_loss {
                if self.net.hu.len() < self.cfg.hmax && self.buf.len() >= 100 && pl - l < 0.03 * pl {
                    self.grow();
                }
            }
            self.prev_loss = Some(l);
        }
    }
}

// =============================================================================================
// Arms + driver

#[derive(Clone, Debug)]
pub enum K {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    HebR,
    Warm(f64),
    Safe,
    G12Ann(g12::AnnCfg),
    Mlp(MlpCfg),
    Casc(CascCfg),
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub name: String,
    pub k: K,
    pub var: Var,
}
pub fn arm(name: &str, k: K, var: Var) -> Arm {
    Arm { name: name.to_string(), k, var }
}

pub fn mk_arm(a: &Arm, env: &dyn Arena, seed: u64) -> Box<dyn Router> {
    let n = env.n_agents();
    let km = || env.kw_map();
    match &a.k {
        K::Keyword => Box::new(KeywordRouter::new(km(), n)),
        K::Eps(g, e) => Box::new(EpsGreedyRouter::new(n, *g, *e)),
        K::Pts(g) => Box::new(PooledThompsonRouter::new(n, *g)),
        K::HebR => Box::new(HebbianRouter::new(n, heb_robust_cfg())),
        K::Warm(k) => Box::new(Warm::new(HebbianRouter::new(n, heb_robust_cfg()), km(), n, *k).with_evidence(Evidence::MeanFeature)),
        K::Safe => Box::new(recommended_router(km(), n)),
        K::G12Ann(c) => g12::mk_router(&g12::Kind::Ann(*c), env.g12env().expect("g12 ann needs a g12 world"), seed),
        K::Mlp(c) => Box::new(Mlp2::new(n, *c, env.dense(), if c.use_kw { Some(km()) } else { None }, seed)),
        K::Casc(c) => Box::new(Casc2::new(n, *c, env.dense(), if c.use_kw { Some(km()) } else { None }, seed)),
    }
}

/// Same protocol as g12/f3: select, observe the realised Bernoulli outcome, update. Records the EXPECTED success of the chosen agent.
pub fn run_policy(env: &dyn Arena, pol: &mut dyn Router, var: Var, seed: u64) -> RunRes {
    let n = env.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let len = env.len();
    let mut per = vec![0.0f32; len];
    for t in 0..len {
        let task = env.task(t, var);
        let a = pol.select(task, &cands, &mut rng);
        per[t] = env.prob(t, a) as f32;
        pol.update(task, a, env.outcome(t, a));
    }
    summarize_run(&per)
}

pub fn oracle_run(env: &dyn Arena) -> RunRes {
    let per: Vec<f32> = (0..env.len()).map(|t| env.oracle(t) as f32).collect();
    summarize_run(&per)
}

// =============================================================================================
// statistics helpers (same conventions as g12)

pub fn fulls(v: &[RunRes]) -> Vec<f64> {
    v.iter().map(|r| r.full).collect()
}
pub fn lates(v: &[RunRes]) -> Vec<f64> {
    v.iter().map(|r| r.late).collect()
}
pub fn mean_of(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

pub fn verdict(adj: f64, mean: f64) -> &'static str {
    if adj < 0.05 && mean > 0.0 {
        "WIN"
    } else if adj < 0.05 {
        "LOSS"
    } else {
        "ns"
    }
}

#[derive(Clone, Debug)]
pub struct Test {
    pub regime: String,
    pub world: String,
    pub family: String,
    pub arm: String,
    pub vs: String,
    pub metric: &'static str,
    pub d: f64,
    pub lo: f64,
    pub hi: f64,
    pub wins: usize,
    pub losses: usize,
    pub p: f64,
    pub adj: f64,
    /// TOST (+-0.01) on the same paired differences: 90% CI inside the margin
    pub lo90: f64,
    pub hi90: f64,
}

pub fn mk_test(regime: &str, world: &str, family: &str, arm: &str, vs: &str, metric: &'static str, a: &[f64], b: &[f64]) -> Test {
    let pd = paired_diff(a, b);
    let eq = g3::eq_test(a, b, 0.01);
    Test { regime: regime.into(), world: world.into(), family: family.into(), arm: arm.into(), vs: vs.into(), metric, d: pd.diff.mean, lo: pd.diff.lo, hi: pd.diff.hi, wins: pd.wins, losses: pd.losses, p: paired_t_pvalue(a, b), adj: 1.0, lo90: eq.lo90, hi90: eq.hi90 }
}

pub fn holm_family(ts: &mut [Test]) {
    let p: Vec<f64> = ts.iter().map(|t| t.p).collect();
    let adj = holm_adjust(&p);
    for (t, a) in ts.iter_mut().zip(adj) {
        t.adj = a;
    }
}

pub fn tost_ok(t: &Test) -> bool {
    t.lo90 > -0.01 && t.hi90 < 0.01
}

pub fn test_json(t: &Test) -> Value {
    json!({"regime": t.regime, "world": t.world, "family": t.family, "arm": t.arm, "vs": t.vs, "metric": t.metric,
           "diff": t.d, "ci_lo": t.lo, "ci_hi": t.hi, "wins": t.wins, "losses": t.losses, "p": t.p, "holm_p": t.adj,
           "verdict": verdict(t.adj, t.d), "tost_pm0.01_ok": tost_ok(t), "ci90": [t.lo90, t.hi90]})
}

/// tasks-to-parity curve from per-window means (window = 500 tasks). Returns (parity, nonneg, sig_overtake) as tasks counts.
pub fn parity_curve(a: &[RunRes], r: &[RunRes], win_tasks: usize) -> (Option<usize>, Option<usize>, Option<usize>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let win = win_tasks / BIN;
    let nb = a[0].bins.len();
    let nw = nb / win;
    let mut diffs = vec![];
    let mut los = vec![];
    let mut am = vec![];
    let mut rm = vec![];
    for k in 0..nw {
        let wa: Vec<f64> = a.iter().map(|x| x.bins[k * win..(k + 1) * win].iter().map(|&y| y as f64).sum::<f64>() / win as f64).collect();
        let wb: Vec<f64> = r.iter().map(|x| x.bins[k * win..(k + 1) * win].iter().map(|&y| y as f64).sum::<f64>() / win as f64).collect();
        let pd = paired_diff(&wa, &wb);
        diffs.push(pd.diff.mean);
        los.push(pd.diff.lo);
        am.push(mean_of(&wa));
        rm.push(mean_of(&wb));
    }
    let sustained = |ok: &dyn Fn(usize) -> bool| -> Option<usize> {
        let mut first = None;
        for k in (0..nw).rev() {
            if ok(k) {
                first = Some(k);
            } else {
                break;
            }
        }
        first
    };
    let parity = sustained(&|k| diffs[k] >= -0.005).map(|k| (k + 1) * win_tasks);
    let nonneg = sustained(&|k| diffs[k] >= 0.0).map(|k| (k + 1) * win_tasks);
    let over = sustained(&|k| los[k] > 0.0).map(|k| (k + 1) * win_tasks);
    (parity, nonneg, over, diffs, los, am, rm)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_src(n_tasks: usize, dd: usize) -> DenseSrc {
        let mut r = Rng::new(5);
        let pca: Vec<f32> = (0..n_tasks * NPCA).map(|_| (r.f64() * 2.0 - 1.0) as f32).collect();
        let raw: Vec<f32> = (0..n_tasks * MAXD).map(|_| (r.f64() * 2.0 - 1.0) as f32).collect();
        let _ = dd;
        DenseSrc { pca: Arc::new(pca), raw: Arc::new(raw), row_of: Arc::new((0..n_tasks as u32).collect()) }
    }

    /// 2 agents; agent 0 wins iff x0 > 0 (a linear function of the dense input). Both nets must learn it.
    fn run_dense<R: Router>(mut r: R, src: &DenseSrc, inp: Inp, t: usize) -> f64 {
        let mut rng = Rng::new(9);
        let mut late = 0.0;
        let mut nl = 0.0;
        for i in 0..t {
            let x0 = src.get(i as u64, inp)[0];
            let task = Task { id: i as u64, task_type: None, features: vec!["w".into()] };
            let a = r.select(&task, &[0, 1], &mut rng);
            let p = if (x0 > 0.0) == (a == 0) { 0.9 } else { 0.1 };
            let rew = if rng.f64() < p { 1.0 } else { 0.0 };
            r.update(&task, a, rew);
            if i >= t * 3 / 4 {
                late += p;
                nl += 1.0;
            }
        }
        late / nl
    }

    #[test]
    fn mlp_and_casc_learn_a_dense_linear_rule() {
        let t = 4000;
        let src = tiny_src(t, 16);
        let mc = MlpCfg { inp: Inp::Pca(16), dh: 64, h: 16, lr: 0.15, tau0: 0.3, tau_min: 0.02, half: 100.0, replay_k: 3, replay_cap: 1000, use_kw: false };
        let m = run_dense(Mlp2::new(2, mc, Some(src.clone()), None, 1), &src, Inp::Pca(16), t);
        assert!(m > 0.75, "mlp late {m}");
        let cc = CascCfg { inp: Inp::Pca(16), dh: 64, lr: 0.15, tau0: 0.3, tau_min: 0.02, half: 100.0, replay_k: 2, grow_every: 300, hmax: 4, use_kw: false, buf_cap: 500, cands: 3, epochs: 20 };
        let c = run_dense(Casc2::new(2, cc, Some(src.clone()), None, 1), &src, Inp::Pca(16), t);
        assert!(c > 0.75, "casc late {c}");
    }

    /// XOR of the signs of x0 and x1: a linear head cannot, a grown hidden unit (or the MLP) can.
    #[test]
    fn cascade_grows_and_helps_on_xor() {
        let t = 8000;
        let src = tiny_src(t, 16);
        let run = |hmax: usize| -> (f64, usize) {
            let cc = CascCfg { inp: Inp::Pca(16), dh: 64, lr: 0.15, tau0: 0.3, tau_min: 0.02, half: 100.0, replay_k: 2, grow_every: 300, hmax, use_kw: false, buf_cap: 600, cands: 4, epochs: 30 };
            let mut r = Casc2::new(2, cc, Some(src.clone()), None, 3);
            let mut rng = Rng::new(9);
            let (mut late, mut nl) = (0.0, 0.0);
            for i in 0..t {
                let x = src.get(i as u64, Inp::Pca(16));
                let s = (x[0] > 0.0) == (x[1] > 0.0);
                let task = Task { id: i as u64, task_type: None, features: vec!["w".into()] };
                let a = r.select(&task, &[0, 1], &mut rng);
                let p = if s == (a == 0) { 0.9 } else { 0.1 };
                r.update(&task, a, if rng.f64() < p { 1.0 } else { 0.0 });
                if i >= t * 3 / 4 {
                    late += p;
                    nl += 1.0;
                }
            }
            (late / nl, r.hidden_units())
        };
        let (lin, nh0) = run(0);
        let (cas, nh) = run(6);
        assert_eq!(nh0, 0);
        assert!(nh >= 1, "no growth happened");
        assert!(lin < 0.62, "linear head should not solve XOR: {lin}");
        assert!(cas > lin + 0.03, "cascade {cas} vs linear {lin} (hidden {nh})");
    }

    #[test]
    fn real_world_is_deterministic_and_oracle_beats_random() {
        // synthetic tiny pool so the test needs no fixtures
        let n = 300;
        let mut r = Rng::new(1);
        let emb: Vec<f32> = (0..n * MAXD).map(|_| (r.f64() * 2.0 - 1.0) as f32 / 20.0).collect();
        let tasks: Vec<Vec<String>> = (0..n).map(|i| vec![format!("w{}", i % 17), format!("v{}", i % 5)]).collect();
        let mut clusters = HashMap::new();
        clusters.insert(12usize, (0..n).map(|i| i % 12).collect::<Vec<_>>());
        let mut kwtok = HashMap::new();
        kwtok.insert(12usize, (0..17).map(|i| (format!("w{i}"), i % 12)).collect::<HashMap<_, _>>());
        let mut anchors = HashMap::new();
        anchors.insert(32usize, (0..n).map(|i| i % 32).collect::<Vec<_>>());
        anchors.insert(128usize, (0..n).map(|i| i % 128).collect::<Vec<_>>());
        let pool = Arc::new(Pool { tag: "A".into(), n, tasks, emb: Arc::new(emb.clone()), raw: Arc::new(emb), pca: Arc::new(vec![0.0; n * NPCA]), clusters, kwtok, anchors });
        for kind in [RKind::Cluster { k: 12, agents: 6 }, RKind::SmoothLin { agents: 6 }, RKind::SmoothRbf { agents: 8 }, RKind::SmoothRbfDrift { agents: 8 }, RKind::Shuffled { k: 12, agents: 6 }] {
            let a = RealW::new(&pool, kind, 3, 500);
            let b = RealW::new(&pool, kind, 3, 500);
            for t in (0..500).step_by(37) {
                assert_eq!(a.task(t, Var::Tok), b.task(t, Var::Tok));
                for ag in 0..kind.agents() {
                    assert_eq!(a.prob(t, ag), b.prob(t, ag));
                    assert!(a.prob(t, ag) > 0.0 && a.prob(t, ag) < 1.0);
                }
            }
            let orc = oracle_run(&a).full;
            let mut kwr = mk_arm(&arm("k", K::Keyword, Var::Tok), &a, 1);
            let kr = run_policy(&a, kwr.as_mut(), Var::Tok, 1).full;
            assert!(orc >= kr - 1e-9, "{:?}: oracle {orc} keyword {kr}", kind);
        }
    }
}
