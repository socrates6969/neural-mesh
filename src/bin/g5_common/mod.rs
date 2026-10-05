//! G5 common: a simulated multi-domain "agency" task world (design / business / marketing + coding / math
//! controls), routers (incl. a domain-hierarchical shrinkage router), and the run driver.
//!
//! EVERYTHING here is synthetic and written by me. It measures whether routing a task to the right
//! specialist and handing that specialist verified context (a recurring brief / style guide) raises
//! graded task quality in a simulation whose structure I chose. It does NOT measure model creativity.
//! The size of the context effect (`WorldCfg::d`) is an assumption, not a measurement.

#![allow(dead_code)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::defaults::{recommended_guard, recommended_router};
use neural_mesh::embedding::{Embedding, EmbeddingConfig};
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::knowledge::{FetchedDoc, FixtureFetcher, Knowledge, KnowledgeConfig};
use neural_mesh::prior::{Evidence, KeywordPrior, Warm};
use neural_mesh::research::{Context, Executor, GuardParams, ResearchConfig, ResearchSession, RouterChoice};
use neural_mesh::router::{Rng, Router, Task};

pub const N_TASKS: usize = 1500;
pub const REVEAL: f64 = 0.6;
pub const DELAY: usize = 15;
pub const NDOM: usize = 5;
pub const NSUBPD: usize = 4;
pub const NSUB: usize = NDOM * NSUBPD;
pub const DOMS: [&str; NDOM] = ["design", "business", "marketing", "coding", "math"];
pub const SUBS: [[&str; NSUBPD]; NDOM] = [
    ["ux-flow", "visual-ui", "design-system", "accessibility"],
    ["financial-model", "go-to-market", "ops-plan", "pricing"],
    ["copywriting", "seo", "campaign", "social"],
    ["rust", "sql", "debug", "api"],
    ["algebra", "proof", "stats", "calculus"],
];
const FLUFF: [&str; 6] = ["please", "urgent", "review", "draft", "client", "iterate"];
const SHARED_PROJ: [&str; 6] = ["brand", "style", "tone", "guide", "deliverable", "audience"];

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
pub enum Pool {
    /// 10 agents: 2 generalists, 5 domain specialists, 1 design+marketing "creative", 2 sub-skill experts
    Agency,
    /// 8 agents, mostly generalists with noisy mild differences (+2 mild domain leaners)
    Generalists,
    /// 20 agents: 3 specialists per domain + 5 generalists
    Big,
    /// 10 agents whose skills come from 3 latent factors (cross-domain correlation: design ~ marketing)
    Latent,
    /// 8 identical agents (c = .7): nothing to win
    Equal,
}
impl Pool {
    pub fn n_agents(self) -> usize {
        match self {
            Pool::Agency | Pool::Latent => 10,
            Pool::Generalists | Pool::Equal => 8,
            Pool::Big => 20,
        }
    }
}

#[derive(Clone, Debug)]
pub struct WorldCfg {
    pub name: &'static str,
    pub pool: Pool,
    /// domain mix
    pub mix: [f64; NDOM],
    /// context dependence per domain: without the right brief a draft keeps (1-d) of its quality
    pub d: [f64; NDOM],
    /// recurring projects (each has a brief / style guide); 0 = tasks carry only unique tokens
    pub projects: usize,
    /// projects distinguished by ONE token among many shared ones (retrieval returns wrong briefs)
    pub confusable: bool,
    /// briefs are ingested into the knowledge store before task 0 (an approved-source fetch)
    pub briefs: bool,
    /// probability a sub-skill keyword advertises the wrong agent
    pub decoy: f64,
    /// penalty factor when a WRONG brief is applied (0 = a wrong brief is just no brief)
    pub mis: f64,
    /// how many of a project's 3 identifying tokens a task mentions (3 = strong id, 1 = weak id)
    pub id_tok: usize,
    /// multiplier on the context dependence `d`
    pub dscale: f64,
    /// how well an agent uses a brief: 0 = .5+.5c (default), 1 = uniform 1.0 (all agents use context equally), 2 = c^2 (only strong agents benefit)
    pub ukind: u8,
}

const D_STD: [f64; NDOM] = [0.30, 0.22, 0.28, 0.0, 0.0];
const MIX_CREATIVE: [f64; NDOM] = [0.30, 0.20, 0.25, 0.15, 0.10];

pub fn worlds() -> Vec<WorldCfg> {
    let base = WorldCfg { name: "agency", pool: Pool::Agency, mix: MIX_CREATIVE, d: D_STD, projects: 12, confusable: false, briefs: false, decoy: 0.30, mis: 0.5, id_tok: 3, dscale: 1.0, ukind: 0 };
    vec![
        // ---------- treatment worlds (context matters; independent structure choices) ----------
        base.clone(),
        WorldCfg { name: "agency-briefs", briefs: true, ..base.clone() },
        WorldCfg { name: "generalist-pool", pool: Pool::Generalists, ..base.clone() },
        WorldCfg { name: "big-pool", pool: Pool::Big, mix: [0.15, 0.40, 0.20, 0.15, 0.10], projects: 24, ..base.clone() },
        WorldCfg { name: "latent-factors", pool: Pool::Latent, ..base.clone() },
        WorldCfg { name: "weak-context", dscale: 0.4, ..base.clone() },
        WorldCfg { name: "low-recurrence", projects: 200, ..base.clone() },
        // ---------- stress / limit: retrieval returns the WRONG brief often ----------
        WorldCfg { name: "confusable-briefs", confusable: true, ..base.clone() },
        // ---------- controls ----------
        WorldCfg { name: "ctl-nocontext-design", mix: [0.70, 0.10, 0.20, 0.0, 0.0], d: [0.0; NDOM], projects: 0, ..base.clone() },
        WorldCfg { name: "ctl-coding-math", mix: [0.0, 0.0, 0.0, 0.55, 0.45], d: [0.0; NDOM], projects: 8, ..base.clone() },
        WorldCfg { name: "ctl-equal-agents", pool: Pool::Equal, d: [0.0; NDOM], ..base.clone() },
    ]
}
pub const N_TREAT: usize = 7; // worlds()[0..7] treatment, [7] stress, [8..] controls
pub const IDX_STRESS: usize = 7;

// ---------------------------------------------------------------- instance
#[derive(Clone, Debug)]
pub struct TaskRec {
    pub task: Task,
    pub dom: usize,
    pub sub_g: usize,
    pub proj: Option<usize>,
    pub dif: f64,
}

pub struct Instance {
    pub n_agents: usize,
    /// skill[a][global sub]
    pub skill: Vec<Vec<f64>>,
    pub tasks: Vec<TaskRec>,
    pub kw: HashMap<String, usize>,
    /// per project: (domain, brief text)
    pub projects: Vec<(usize, String)>,
    /// oracle (no-context) expected quality of the best agent, per task
    pub oracle: Vec<f64>,
}

fn jit(r: &mut Rng, base: f64, spread: f64) -> f64 {
    (base + spread * r.f64()).clamp(0.05, 0.97)
}

fn build_skill(pool: Pool, r: &mut Rng) -> Vec<Vec<f64>> {
    let na = pool.n_agents();
    let mut s = vec![vec![0.0; NSUB]; na];
    match pool {
        Pool::Agency => {
            for g in 0..NSUB {
                let dom = g / NSUBPD;
                let sub = g % NSUBPD;
                for a in 0..2 {
                    s[a][g] = jit(r, 0.50, 0.14);
                }
                for k in 0..NDOM {
                    s[2 + k][g] = if dom == k { jit(r, 0.78, 0.12) } else { jit(r, 0.28, 0.17) };
                }
                s[7][g] = if dom == 0 || dom == 2 { jit(r, 0.76, 0.12) } else { jit(r, 0.30, 0.15) };
                s[8][g] = if dom == 0 && sub == 3 { jit(r, 0.93, 0.04) } else if dom == 0 { jit(r, 0.50, 0.10) } else { jit(r, 0.30, 0.15) };
                s[9][g] = if dom == 1 && sub == 0 { jit(r, 0.93, 0.04) } else if dom == 1 { jit(r, 0.50, 0.10) } else { jit(r, 0.30, 0.15) };
            }
        }
        Pool::Generalists => {
            for a in 0..na {
                let base = 0.45 + 0.20 * r.f64();
                let lean = if a < 2 { Some(a * 2) } else { None };
                for g in 0..NSUB {
                    let mut v = base + 0.12 * (r.f64() - 0.5) * 2.0;
                    if lean == Some(g / NSUBPD) {
                        v += 0.22;
                    }
                    s[a][g] = v.clamp(0.1, 0.95);
                }
            }
        }
        Pool::Big => {
            for g in 0..NSUB {
                let dom = g / NSUBPD;
                for k in 0..NDOM {
                    for j in 0..3 {
                        let a = k * 3 + j;
                        s[a][g] = if dom == k { jit(r, 0.62 + 0.04 * j as f64, 0.26) } else { jit(r, 0.28, 0.17) };
                    }
                }
                for a in 15..20 {
                    s[a][g] = jit(r, 0.45, 0.2);
                }
            }
        }
        Pool::Latent => {
            // loadings of domains on (creative, analytic, technical)
            let l = [[1.0, 0.2, 0.2], [0.4, 1.0, 0.1], [0.9, 0.5, 0.1], [0.1, 0.2, 1.0], [0.0, 0.7, 0.6]];
            for a in 0..na {
                let z = [rnorm(r), rnorm(r), rnorm(r)];
                for g in 0..NSUB {
                    let dom = g / NSUBPD;
                    let lz: f64 = (0..3).map(|f| l[dom][f] * z[f]).sum();
                    s[a][g] = (0.55 + 0.22 * lz + 0.08 * rnorm(r)).clamp(0.1, 0.97);
                }
            }
        }
        Pool::Equal => {
            for a in 0..na {
                for g in 0..NSUB {
                    s[a][g] = 0.7;
                }
            }
        }
    }
    // shuffle agent order so index carries no information
    for i in (1..na).rev() {
        s.swap(i, r.below(i + 1));
    }
    s
}

pub fn build(w: &WorldCfg, seed: u64, widx: u64) -> Instance {
    let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(31 + widx * 977));
    let skill = build_skill(w.pool, &mut r);
    let na = w.pool.n_agents();
    // keyword map: sub-skill words advertise the best agent (decoy: a random one); domain word -> best domain-average agent
    let mut kw = HashMap::new();
    for dom in 0..NDOM {
        for sub in 0..NSUBPD {
            let g = dom * NSUBPD + sub;
            let best = (0..na).fold(0, |b, a| if skill[a][g] > skill[b][g] { a } else { b });
            let adv = if r.f64() < w.decoy { r.below(na) } else { best };
            for j in 0..3 {
                kw.insert(format!("{}-w{j}", SUBS[dom][sub]), adv);
            }
        }
        let avg = |a: usize| (0..NSUBPD).map(|s| skill[a][dom * NSUBPD + s]).sum::<f64>() / NSUBPD as f64;
        let best = (0..na).fold(0, |b, a| if avg(a) > avg(b) { a } else { b });
        kw.insert(DOMS[dom].to_string(), if r.f64() < w.decoy { r.below(na) } else { best });
    }
    // projects
    let mut projects: Vec<(usize, String)> = vec![];
    let mut proj_of_dom: Vec<Vec<usize>> = vec![vec![]; NDOM];
    if w.projects > 0 {
        for dom in 0..NDOM {
            if w.mix[dom] <= 0.0 {
                continue;
            }
            let n = ((w.projects as f64 * w.mix[dom]).round() as usize).max(1);
            for _ in 0..n {
                let id = projects.len();
                let text = if w.confusable {
                    format!("pr{id}a {} dom:{} {}", SHARED_PROJ.join(" "), DOMS[dom], DOMS[dom])
                } else {
                    format!("pr{id}a pr{id}b pr{id}c dom:{} {}", DOMS[dom], DOMS[dom])
                };
                projects.push((dom, text));
                proj_of_dom[dom].push(id);
            }
        }
    }
    // within-domain sub-skill weights (permuted per seed)
    let mut subw = [[0.0; NSUBPD]; NDOM];
    for dom in 0..NDOM {
        let mut base = [0.4, 0.3, 0.2, 0.1];
        for i in (1..NSUBPD).rev() {
            base.swap(i, r.below(i + 1));
        }
        subw[dom] = base;
    }
    let mixtot: f64 = w.mix.iter().sum();
    let (mut tasks, mut oracle) = (Vec::with_capacity(N_TASKS), Vec::with_capacity(N_TASKS));
    for i in 0..N_TASKS {
        let mut x = r.f64() * mixtot;
        let mut dom = NDOM - 1;
        for k in 0..NDOM {
            if x < w.mix[k] {
                dom = k;
                break;
            }
            x -= w.mix[k];
        }
        let mut y = r.f64();
        let mut sub = NSUBPD - 1;
        for k in 0..NSUBPD {
            if y < subw[dom][k] {
                sub = k;
                break;
            }
            y -= subw[dom][k];
        }
        let g = dom * NSUBPD + sub;
        let mut f: Vec<String> = vec![DOMS[dom].to_string(), format!("dom:{}", DOMS[dom])];
        // two of the three sub-skill words
        let skip = r.below(3);
        for j in 0..3 {
            if j != skip {
                f.push(format!("{}-w{j}", SUBS[dom][sub]));
            }
        }
        if r.f64() < 0.2 {
            // cross-talk word from another sub-skill of the same domain
            let o = (sub + 1 + r.below(NSUBPD - 1)) % NSUBPD;
            f.push(format!("{}-w{}", SUBS[dom][o], r.below(3)));
        }
        let mut proj = None;
        if !proj_of_dom[dom].is_empty() {
            let p = proj_of_dom[dom][r.below(proj_of_dom[dom].len())];
            proj = Some(p);
            if w.confusable {
                f.push(format!("pr{p}a"));
                let mut pool: Vec<&str> = SHARED_PROJ.to_vec();
                for i2 in (1..pool.len()).rev() {
                    pool.swap(i2, r.below(i2 + 1));
                }
                for t in pool.iter().take(3) {
                    f.push(t.to_string());
                }
            } else {
                let mut idx = [0usize, 1, 2];
                for i2 in (1..3).rev() {
                    idx.swap(i2, r.below(i2 + 1));
                }
                for &j in idx.iter().take(w.id_tok.clamp(1, 3)) {
                    f.push(format!("pr{p}{}", ["a", "b", "c"][j]));
                }
            }
        } else {
            for j in 0..2 {
                f.push(format!("u{i}x{j}"));
            }
        }
        for _ in 0..2 {
            f.push(FLUFF[r.below(FLUFF.len())].to_string());
        }
        let dif = 0.07 * rnorm(&mut r);
        oracle.push((0..na).map(|a| skill[a][g] * (1.0 - w.d[dom] * w.dscale)).fold(0.0, f64::max));
        tasks.push(TaskRec { task: Task { id: i as u64, task_type: Some(format!("{}/{}", DOMS[dom], SUBS[dom][sub])), features: f }, dom, sub_g: g, proj, dif });
    }
    Instance { n_agents: na, skill, tasks, kw, projects, oracle }
}

// ---------------------------------------------------------------- executor (the "agents")
#[derive(Clone, Debug)]
pub struct Draft {
    pub agent: usize,
    pub q: f64,
    pub tid: u64,
}

pub struct WorldExec<'a> {
    pub w: &'a WorldCfg,
    pub inst: &'a Instance,
    pub seed: u64,
    /// number of documents ingested before task 0 (record ids of tasks start after them)
    pub n_docs: u64,
    pub track: bool,
    /// anchor -> project -> count (executor-side ground truth about what each retrieved anchor really contains)
    pub anchor_proj: HashMap<u64, HashMap<usize, u32>>,
    pub ctx_hits: u32,
    pub purity_sum: f64,
    pub force_neff: Option<f64>,
}

impl<'a> WorldExec<'a> {
    pub fn new(w: &'a WorldCfg, inst: &'a Instance, seed: u64, track: bool) -> Self {
        WorldExec { w, inst, seed, n_docs: 0, track, anchor_proj: HashMap::new(), ctx_hits: 0, purity_sum: 0.0, force_neff: None }
    }
    /// Ground-truth registration of the ingested briefs (same deterministic ops as the session's store).
    pub fn register_doc(&mut self, anchor: u64, proj: usize) {
        *self.anchor_proj.entry(anchor).or_default().entry(proj).or_insert(0) += 2;
    }
}

impl Executor for WorldExec<'_> {
    type Out = Draft;
    fn draft(&mut self, task: &Task, agent: usize, ctx: &Context) -> Draft {
        let tr = &self.inst.tasks[task.id as usize];
        let tid = task.id;
        let c = self.inst.skill[agent][tr.sub_g];
        let d = self.w.d[tr.dom] * self.w.dscale;
        let mut neff = 0.0;
        if self.track {
            let anchor = ctx.anchor.unwrap_or(self.n_docs + tid);
            let key = tr.proj.unwrap_or(usize::MAX - tid as usize);
            if let (Some(n), Some(_)) = (ctx.note, ctx.anchor) {
                let cnt = self.anchor_proj.entry(anchor).or_default();
                let tot: u32 = cnt.values().sum();
                let same = *cnt.get(&key).unwrap_or(&0);
                let pur = if tot > 0 { same as f64 / tot as f64 } else { 1.0 };
                neff = n * (2.0 * pur - 1.0);
                self.ctx_hits += 1;
                self.purity_sum += pur;
            }
            *self.anchor_proj.entry(anchor).or_default().entry(key).or_insert(0) += 1;
        }
        let neff = self.force_neff.unwrap_or(neff);
        let u = match self.w.ukind { 1 => 1.0, 2 => c * c, _ => 0.5 + 0.5 * c };
        let fit = if neff >= 0.0 { 1.0 - d * (1.0 - neff * u) } else { 1.0 - d * (1.0 + self.w.mis * (-neff) * u) };
        let q0 = (c + tr.dif + 0.07 * norm(self.seed, tid, agent as u64, 13)).clamp(0.0, 1.0);
        Draft { agent, q: (q0 * fit).clamp(0.0, 1.0), tid }
    }
    fn tool_ok(&mut self, o: &Draft) -> bool {
        uni(self.seed, o.tid, o.agent as u64, 14) < 0.99
    }
    fn check(&mut self, _t: &Task, _c: usize, _o: &Draft) -> bool {
        false
    }
    fn grade(&mut self, t: &Task, o: &Draft) -> Option<f64> {
        if uni(self.seed, t.id, 0, 16) >= REVEAL {
            return None;
        }
        Some((o.q + 0.05 * norm(self.seed, t.id, o.agent as u64, 17)).clamp(0.0, 1.0))
    }
}

// ---------------------------------------------------------------- hierarchical routing
/// Domain-hierarchical shrinkage router. Three levels of per-agent mean-reward estimates:
/// global (agent) -> domain -> sub-skill. Each level is shrunk toward the level above with pseudo-count
/// `m_*`; the keyword-advertised agent gets `kw_w` pseudo-observations of reward `.8` at the sub-skill
/// level (a prior, overridden by data). Selection: Thompson draw N(mean, (xi*sigma)^2 / (n_eff+1)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HierParams {
    pub m_g: f64,
    pub m_d: f64,
    pub m_s: f64,
    pub xi: f64,
    pub kw_w: f64,
    /// false = ablation: no domain level (sub-skill estimates shrink straight to the agent's global mean)
    pub use_dom: bool,
}
pub const HIER_SIGMA: f64 = 0.2;

pub struct HierRouter {
    pub p: HierParams,
    n_agents: usize,
    glob: Vec<(f64, f64)>,
    dom: HashMap<String, Vec<(f64, f64)>>,
    sub: HashMap<String, Vec<(f64, f64)>>,
    kw: KeywordPrior,
    dom_names: Vec<String>,
}

impl HierRouter {
    pub fn new(n_agents: usize, kwmap: HashMap<String, usize>, p: HierParams) -> Self {
        HierRouter { p, n_agents, glob: vec![(0.0, 0.0); n_agents], dom: HashMap::new(), sub: HashMap::new(), kw: KeywordPrior::new(kwmap, n_agents), dom_names: DOMS.iter().map(|s| s.to_string()).collect() }
    }
    fn keys(task: &Task) -> (String, String) {
        let t = task.task_type.clone().unwrap_or_default();
        let d = t.split('/').next().unwrap_or("").to_string();
        (d, t)
    }
    fn kw_agent(&self, task: &Task, cands: &[usize]) -> Option<usize> {
        let v = self.kw.votes(task, cands);
        v.iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map(|x| x.0)
    }
    /// (mean, effective n) of the sub-skill level for `a`.
    fn level(&self, a: usize, dk: &str, sk: &str, kwa: Option<usize>) -> (f64, f64) {
        let p = &self.p;
        let (gs, gn) = self.glob[a];
        let g = (gs + p.m_g * 0.5) / (gn + p.m_g);
        let (ds, dn) = self.dom.get(dk).map_or((0.0, 0.0), |v| v[a]);
        let d = if p.use_dom { (ds + p.m_d * g) / (dn + p.m_d) } else { g };
        let (ss, sn) = self.sub.get(sk).map_or((0.0, 0.0), |v| v[a]);
        let kwn = if kwa == Some(a) { p.kw_w } else { 0.0 };
        let s = (ss + p.m_s * d + kwn * 0.8) / (sn + p.m_s + kwn);
        (s, sn + p.m_s + kwn)
    }
    pub fn means(&self, task: &Task, cands: &[usize]) -> Vec<f64> {
        let (dk, sk) = Self::keys(task);
        let kwa = self.kw_agent(task, cands);
        cands.iter().map(|&a| self.level(a, &dk, &sk, kwa).0).collect()
    }
}

impl Router for HierRouter {
    fn name(&self) -> String {
        "hier".into()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let (dk, sk) = Self::keys(task);
        let kwa = self.kw_agent(task, candidates);
        let mut best = candidates[0];
        let mut bv = f64::NEG_INFINITY;
        for &a in candidates {
            let (m, n) = self.level(a, &dk, &sk, kwa);
            let v = m + self.p.xi * HIER_SIGMA / (n + 1.0).sqrt() * rnorm(rng);
            if v > bv {
                bv = v;
                best = a;
            }
        }
        best
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        let (dk, sk) = Self::keys(task);
        let n = self.n_agents;
        let _ = &self.dom_names;
        self.glob[agent].0 += r;
        self.glob[agent].1 += 1.0;
        let c = self.dom.entry(dk).or_insert_with(|| vec![(0.0, 0.0); n]);
        c[agent].0 += r;
        c[agent].1 += 1.0;
        let c = self.sub.entry(sk).or_insert_with(|| vec![(0.0, 0.0); n]);
        c[agent].0 += r;
        c[agent].1 += 1.0;
    }
}

// ---------------------------------------------------------------- arms
#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    HebR,
    /// warm(hebr), k
    Warm(f64),
    /// defaults::recommended_router (guarded-safe) = "plain routing"
    Safe,
    Hier(HierParams),
    /// best expected skill, ignoring context (reference ceiling for routing)
    Oracle,
    /// best expected skill AND a perfect (pure, strength 1) brief: reference ceiling for routing + context
    OracleCtx,
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub name: String,
    pub kind: Kind,
    pub retr: bool,
    /// run through the real `research::ResearchSession` (otherwise the in-bin replica driver)
    pub session: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RetrParams {
    pub store_min: f64,
    pub sim_min: f64,
}

pub struct RunOut {
    pub q: Vec<f32>,
    pub ctx_hits: u32,
    pub purity: f64,
}

fn mk_router(kind: &Kind, inst: &Instance) -> Box<dyn Router> {
    let n = inst.n_agents;
    let km = || inst.kw.clone();
    match kind {
        Kind::Keyword => Box::new(KeywordRouter::new(km(), n)),
        Kind::Eps(g, e) => Box::new(EpsGreedyRouter::new(n, *g, *e)),
        Kind::Pts(g) => Box::new(PooledThompsonRouter::new(n, *g)),
        Kind::HebR => Box::new(HebbianRouter::new(n, heb_robust_cfg())),
        Kind::Warm(k) => Box::new(Warm::new(HebbianRouter::new(n, heb_robust_cfg()), km(), n, *k).with_evidence(Evidence::MeanFeature)),
        Kind::Safe => Box::new(recommended_router(km(), n)),
        Kind::Hier(p) => Box::new(HierRouter::new(n, km(), *p)),
        Kind::Oracle | Kind::OracleCtx => unreachable!("oracle handled separately"),
    }
}

struct Ev {
    due: u64,
    task: Task,
    agent: usize,
    g: Option<f64>,
    rec: Option<u64>,
    anchor: Option<u64>,
}

/// Replica of `ResearchSession::run_task` for critic = None / reuse = true / health off, around ANY router.
/// `retr = None`: no retrieval (plain routing with the same delayed graded feedback).
/// Validated against the real session in `g5_study --check` (identical per-task qualities).
pub fn run_replica(w: &WorldCfg, inst: &Instance, kind: &Kind, retr: Option<RetrParams>, seed: u64) -> RunOut {
    let na = inst.n_agents;
    let cands: Vec<usize> = (0..na).collect();
    let mut ex = WorldExec::new(w, inst, seed, retr.is_some());
    let mut rng = Rng::new(seed * 7 + 1);
    let mut q = Vec::with_capacity(N_TASKS);
    if matches!(kind, Kind::Oracle | Kind::OracleCtx) {
        if matches!(kind, Kind::OracleCtx) {
            ex.force_neff = Some(1.0);
        }
        for (i, tr) in inst.tasks.iter().enumerate() {
            let a = (0..na).fold(0, |b, a| if inst.skill[a][tr.sub_g] > inst.skill[b][tr.sub_g] { a } else { b });
            let d = ex.draft(&tr.task, a, &Context::default());
            q.push(d.q as f32);
            let _ = i;
        }
        return RunOut { q, ctx_hits: 0, purity: 0.0 };
    }
    let mut router = mk_router(kind, inst);
    let mut kn: Option<Knowledge> = None;
    let mut emb: Option<Arc<dyn Embedding>> = None;
    let mut notes: HashMap<u64, f64> = HashMap::new();
    if let Some(rp) = retr {
        let (e, _) = EmbeddingConfig::default().build_or_hash(64);
        let mut kc = KnowledgeConfig::new(e.dim());
        kc.k = 5;
        kc.sim_min = rp.sim_min as f32;
        let mut k = Knowledge::new_shared(kc).with_embedder(e.clone());
        if w.briefs {
            ex.n_docs = inst.projects.len() as u64;
            for (pi, (_dom, text)) in inst.projects.iter().enumerate() {
                let em = e.embed(text);
                let r = k.retrieve(&em);
                let rec = k.remember(&em, &r);
                let anchor = r.anchor.unwrap_or(rec);
                let ent = notes.entry(anchor).or_insert(0.0);
                *ent = ent.max(0.5);
                ex.register_doc(anchor, pi);
            }
        }
        kn = Some(k);
        emb = Some(e);
    }
    let mut pending: VecDeque<Ev> = VecDeque::new();
    let mut tick = 0u64;
    for tr in &inst.tasks {
        tick += 1;
        while pending.front().is_some_and(|e| e.due <= tick) {
            let ev = pending.pop_front().unwrap();
            if let Some(g) = ev.g {
                router.update(&ev.task, ev.agent, g);
            }
            if let (Some(k), Some(rec)) = (kn.as_mut(), ev.rec) {
                if let Some(g) = ev.g {
                    k.credit(rec, g);
                    if g >= retr.unwrap().store_min {
                        if let Some(a) = ev.anchor {
                            let n = notes.entry(a).or_insert(g);
                            *n = 0.7 * *n + 0.3 * g;
                        }
                    }
                }
            }
        }
        let task = &tr.task;
        let mut ctx = Context::default();
        let (mut rec, mut anchor) = (None, None);
        if let (Some(k), Some(e)) = (kn.as_mut(), emb.as_ref()) {
            let em = e.embed_task(task);
            let r = k.retrieve(&em);
            ctx.hits = r.hits.iter().filter(|h| h.1 >= retr.unwrap().sim_min as f32).count();
            ctx.anchor = r.anchor;
            ctx.note = r.anchor.and_then(|a| notes.get(&a).copied());
            let id = k.remember(&em, &r);
            rec = Some(id);
            anchor = Some(r.anchor.unwrap_or(id));
        }
        let a = router.select(task, &cands, &mut rng);
        if let (Some(k), Some(id)) = (kn.as_mut(), rec) {
            k.commit(id, a);
        }
        let d = ex.draft(task, a, &ctx);
        q.push(d.q as f32);
        let g = ex.grade(task, &d);
        pending.push_back(Ev { due: tick + DELAY as u64, task: task.clone(), agent: a, g, rec, anchor });
    }
    RunOut { q, ctx_hits: ex.ctx_hits, purity: if ex.ctx_hits > 0 { ex.purity_sum / ex.ctx_hits as f64 } else { 0.0 } }
}

/// The real `ResearchSession` (the verified pipeline). Router = `guarded(hebr)` with the
/// `defaults::recommended_guard` params, retrieval + reuse on, critic off.
pub fn run_session(w: &WorldCfg, inst: &Instance, kind: &Kind, retr: Option<RetrParams>, health: bool, seed: u64) -> RunOut {
    let na = inst.n_agents;
    let cands: Vec<usize> = (0..na).collect();
    let g = recommended_guard();
    let gp = GuardParams { tau: g.tau, z: g.z, kw_a: g.kw_a, kw_b: g.kw_b, discount: g.discount, challenge: g.challenge };
    let rp = retr.unwrap_or(RetrParams { store_min: 0.65, sim_min: 0.55 });
    let allow = vec!["briefs.local".to_string()];
    let cfg = ResearchConfig {
        router: match kind { Kind::Warm(_) => RouterChoice::WarmHebr, _ => RouterChoice::GuardedHebr(gp) },
        warm_k: if let Kind::Warm(k) = kind { *k } else { 1.0 },
        health,
        retrieval: retr.is_some(),
        reuse: true,
        delay: DELAY,
        store_min: rp.store_min,
        sim_min: rp.sim_min as f32,
        allow: allow.clone(),
        ..ResearchConfig::default()
    };
    let mut s = ResearchSession::new(na, inst.kw.clone(), cfg.clone());
    let mut ex = WorldExec::new(w, inst, seed, retr.is_some());
    if retr.is_some() && w.briefs {
        // approved-source fetch of the briefs (fixture), one document per project
        let mut fx = FixtureFetcher::new();
        for (pi, (_d, text)) in inst.projects.iter().enumerate() {
            fx = fx.with("briefs.local", &format!("p{pi}"), vec![FetchedDoc { source: "briefs.local".into(), text: text.clone() }]);
        }
        let f = cfg.guarded_fetcher(fx);
        ex.n_docs = inst.projects.len() as u64;
        // a mirror store reproduces the anchors the session's (identical, deterministic) store assigns
        let (e, _) = cfg.embedding.build_or_hash(cfg.dim);
        let mut kc = KnowledgeConfig::new(e.dim());
        kc.k = cfg.k;
        kc.sim_min = cfg.sim_min;
        let mut mirror = Knowledge::new_shared(kc).with_embedder(e.clone());
        for (pi, (_d, text)) in inst.projects.iter().enumerate() {
            let n = s.ingest(&f, "briefs.local", &format!("p{pi}")).expect("approved fixture fetch");
            assert_eq!(n, 1);
            let em = e.embed(text);
            let r = mirror.retrieve(&em);
            let rec = mirror.remember(&em, &r);
            ex.register_doc(r.anchor.unwrap_or(rec), pi);
        }
    }
    let mut rng = Rng::new(seed * 7 + 1);
    let mut q = Vec::with_capacity(N_TASKS);
    for tr in &inst.tasks {
        let rep = s.run_task(&mut ex, &tr.task, &cands, &mut rng);
        q.push(rep.delivered.q as f32);
    }
    RunOut { q, ctx_hits: ex.ctx_hits, purity: if ex.ctx_hits > 0 { ex.purity_sum / ex.ctx_hits as f64 } else { 0.0 } }
}

pub fn run_arm(w: &WorldCfg, inst: &Instance, arm: &Arm, retr: RetrParams, seed: u64) -> RunOut {
    if arm.session {
        run_session(w, inst, &arm.kind, if arm.retr { Some(retr) } else { None }, true, seed)
    } else {
        run_replica(w, inst, &arm.kind, if arm.retr { Some(retr) } else { None }, seed)
    }
}

// ---------------------------------------------------------------- metrics
pub const HORIZONS: [usize; 4] = [100, 250, 500, 1500];

pub fn mean_f32(xs: &[f32]) -> f64 {
    xs.iter().map(|&x| x as f64).sum::<f64>() / xs.len().max(1) as f64
}

pub fn horizon_means(q: &[f32]) -> [f64; 4] {
    let mut o = [0.0; 4];
    for (i, &h) in HORIZONS.iter().enumerate() {
        o[i] = mean_f32(&q[..h.min(q.len())]);
    }
    o
}

pub fn domain_means(q: &[f32], inst: &Instance) -> [f64; NDOM] {
    let (mut s, mut n) = ([0.0; NDOM], [0.0; NDOM]);
    for (i, t) in inst.tasks.iter().enumerate() {
        s[t.dom] += q[i] as f64;
        n[t.dom] += 1.0;
    }
    let mut o = [f64::NAN; NDOM];
    for d in 0..NDOM {
        if n[d] > 0.0 {
            o[d] = s[d] / n[d];
        }
    }
    o
}

/// First task index (1-based) at which the trailing-100-task mean quality reaches `frac` x the world's
/// oracle (best-agent, context-free) mean; N_TASKS+1 (censored) when never reached.
pub fn tasks_to_target(q: &[f32], oracle_mean: f64, frac: f64) -> f64 {
    let thr = frac * oracle_mean;
    let w = 100usize;
    let mut sum = 0.0;
    for i in 0..q.len() {
        sum += q[i] as f64;
        if i >= w {
            sum -= q[i - w] as f64;
        }
        if i + 1 >= w && sum / w as f64 >= thr {
            return (i + 1) as f64;
        }
    }
    (q.len() + 1) as f64
}

pub fn oracle_mean(inst: &Instance) -> f64 {
    inst.oracle.iter().sum::<f64>() / inst.oracle.len() as f64
}

pub fn used_unique(xs: &[usize]) -> usize {
    xs.iter().collect::<HashSet<_>>().len()
}

/// First task index (1-based) at which the trailing-100-task mean quality reaches the absolute `thr`
/// (N_TASKS+1 = censored). Used with `thr` = 95% of the best fair baseline's own late-run quality.
pub fn tasks_to_abs(q: &[f32], thr: f64) -> f64 {
    let w = 100usize;
    let mut sum = 0.0;
    for i in 0..q.len() {
        sum += q[i] as f64;
        if i >= w {
            sum -= q[i - w] as f64;
        }
        if i + 1 >= w && sum / w as f64 >= thr {
            return (i + 1) as f64;
        }
    }
    (q.len() + 1) as f64
}

/// Mean quality over the last `n` tasks.
pub fn tail_mean(q: &[f32], n: usize) -> f64 {
    let n = n.min(q.len());
    mean_f32(&q[q.len() - n..])
}

// ---------------------------------------------------------------- world description (for --show)
pub fn describe(w: &WorldCfg, widx: u64, seed: u64) -> String {
    use std::fmt::Write;
    let inst = build(w, seed, widx);
    let mut s = String::new();
    let na = inst.n_agents;
    writeln!(s, "world {} | pool {:?} | {} agents | mix {:?} | d {:?} x{} | projects {} | id tokens {}/3 | decoy {} | briefs-given {} | confusable {}", w.name, w.pool, na, w.mix, w.d, w.dscale, w.projects, w.id_tok, w.decoy, w.briefs, w.confusable).unwrap();
    writeln!(s, "agent mean competence by domain (hidden):").unwrap();
    let dm = |a: usize, d: usize| (0..NSUBPD).map(|k| inst.skill[a][d * NSUBPD + k]).sum::<f64>() / NSUBPD as f64;
    writeln!(s, "  agent   {}", DOMS.iter().map(|d| format!("{:>9}", d)).collect::<String>()).unwrap();
    for a in 0..na {
        writeln!(s, "  a{:<5}  {}", a, (0..NDOM).map(|d| format!("{:>9.2}", dm(a, d))).collect::<String>()).unwrap();
    }
    // cross-domain correlation of agents' domain-mean competence
    writeln!(s, "correlation across agents of domain-mean competence:").unwrap();
    for d1 in 0..NDOM {
        let mut row = format!("  {:<9}", DOMS[d1]);
        for d2 in 0..NDOM {
            let (x, y): (Vec<f64>, Vec<f64>) = (0..na).map(|a| (dm(a, d1), dm(a, d2))).unzip();
            let (mx, my) = (x.iter().sum::<f64>() / na as f64, y.iter().sum::<f64>() / na as f64);
            let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
            for i in 0..na {
                sxy += (x[i] - mx) * (y[i] - my);
                sxx += (x[i] - mx).powi(2);
                syy += (y[i] - my).powi(2);
            }
            let r = if sxx > 1e-12 && syy > 1e-12 { sxy / (sxx * syy).sqrt() } else { f64::NAN };
            row.push_str(&format!("{:>9.2}", r));
        }
        writeln!(s, "{row}").unwrap();
    }
    writeln!(s, "sample tasks (tags | text tokens | project | best agent (competence) | keyword-advertised agent):").unwrap();
    let mut kwr = KeywordRouter::new(inst.kw.clone(), na);
    let cands: Vec<usize> = (0..na).collect();
    let mut rng = Rng::new(1);
    for i in [0usize, 1, 2, 3, 4, 5, 6, 7] {
        let t = &inst.tasks[i];
        let best = (0..na).fold(0, |b, a| if inst.skill[a][t.sub_g] > inst.skill[b][t.sub_g] { a } else { b });
        let kwa = kwr.select(&t.task, &cands, &mut rng);
        writeln!(s, "  #{i} [{}] {:?} | proj {:?} | best a{best} ({:.2}) | keyword a{kwa} ({:.2})", t.task.task_type.clone().unwrap_or_default(), t.task.features, t.proj, inst.skill[best][t.sub_g], inst.skill[kwa][t.sub_g]).unwrap();
    }
    let nkwwrong = (0..NSUB).filter(|&g| {
        let (dom, sub) = (g / NSUBPD, g % NSUBPD);
        let a = inst.kw[&format!("{}-w0", SUBS[dom][sub])];
        let best = (0..na).fold(0, |b, a| if inst.skill[a][g] > inst.skill[b][g] { a } else { b });
        inst.skill[a][g] < inst.skill[best][g] - 0.05
    }).count();
    writeln!(s, "keyword map: {}/{} sub-skills advertise a clearly non-best agent; oracle (no-ctx) mean quality {:.3}", nkwwrong, NSUB, oracle_mean(&inst)).unwrap();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_is_deterministic_and_tasks_are_tagged() {
        let ws = worlds();
        let a = build(&ws[0], 5, 0);
        let b = build(&ws[0], 5, 0);
        assert_eq!(a.tasks.len(), N_TASKS);
        assert!(a.tasks.iter().zip(&b.tasks).all(|(x, y)| x.task == y.task && x.dif == y.dif));
        assert!(a.tasks.iter().all(|t| t.task.task_type.as_deref().unwrap().contains('/')));
        assert_eq!(a.skill, b.skill);
    }

    #[test]
    fn hier_router_learns_the_best_agent_per_subskill() {
        let mut km = HashMap::new();
        km.insert("w".to_string(), 0usize);
        let p = HierParams { m_g: 2.0, m_d: 4.0, m_s: 6.0, xi: 0.5, kw_w: 3.0, use_dom: true };
        let mut r = HierRouter::new(3, km, p);
        let t = Task { id: 0, task_type: Some("design/ux-flow".into()), features: vec!["w".into()] };
        let mut rng = Rng::new(3);
        let q = [0.3, 0.9, 0.5];
        let mut late = 0;
        for i in 0..600 {
            let a = r.select(&t, &[0, 1, 2], &mut rng);
            r.update(&t, a, q[a]);
            if i >= 400 && a == 1 {
                late += 1;
            }
        }
        assert!(late > 190, "late picks of the best agent: {late}");
    }

    #[test]
    fn perfect_brief_beats_no_brief_and_wrong_brief_hurts() {
        let ws = worlds();
        let w = &ws[0];
        let inst = build(w, 3, 0);
        let i = inst.tasks.iter().position(|t| t.dom == 0).unwrap();
        let t = &inst.tasks[i].task;
        let mut ex = WorldExec::new(w, &inst, 3, true);
        ex.force_neff = Some(1.0);
        let good = ex.draft(t, 0, &Context::default()).q;
        ex.force_neff = Some(0.0);
        let none = ex.draft(t, 0, &Context::default()).q;
        ex.force_neff = Some(-1.0);
        let bad = ex.draft(t, 0, &Context::default()).q;
        assert!(good >= none && none >= bad, "{good} {none} {bad}");
    }

    #[test]
    fn retrieval_is_neutral_when_context_is_irrelevant() {
        let ws = worlds();
        let wi = ws.iter().position(|w| w.name == "ctl-coding-math").unwrap();
        let inst = build(&ws[wi], 2, wi as u64);
        let rp = RetrParams { store_min: 0.55, sim_min: 0.75 };
        let a = run_replica(&ws[wi], &inst, &Kind::Warm(1.0), None, 2);
        let b = run_replica(&ws[wi], &inst, &Kind::Warm(1.0), Some(rp), 2);
        assert!(a.q == b.q);
    }
}
