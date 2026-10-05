//! G19 (W20 SONA-A): pattern-memory KEYING and RETRIEVAL mechanisms for a SonaPlus-style router.
//!
//! Everything here is bin-local (src/*.rs is read-only for this round):
//!  * [`Anchors`]   - learned feature keys: online co-occurrence (PMI-lift) clustering of surface features into anchor tokens
//!                    (or an ORACLE topic map, diagnostic only). Merges migrate stored statistics.
//!  * [`AnchorGuard`] - `prior::Guarded` generalised so that the outcome counts are keyed by anchor tokens instead of raw
//!                    features. With anchors disabled it is decision-identical to `prior::Guarded` (self-checked).
//!  * [`KeyedMemory`] - SONA-style pattern memory: unigram token tables + exact key-set patterns + k-nearest-pattern blending
//!                    (Jaccard similarity weighting), learned key selection (drop confirmed-uninformative tokens), bounded
//!                    capacity with EWC-style importance consolidation (merge victim into nearest neighbour) vs LRU/FIFO/count drop.

use std::collections::HashMap;

use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{GuardConfig, KeywordPrior};
use neural_mesh::router::{Rng, Router, Task};

// ------------------------------------------------------------------------------------------------ anchors
#[derive(Clone, Debug, PartialEq)]
pub struct AnchorCfg {
    pub enabled: bool,
    /// minimum lift P(a,b)/(P(a)P(b)) for a merge
    pub lift_min: f64,
    /// minimum co-occurrence count for a merge
    pub min_count: u32,
    /// maximum Poisson upper-tail probability P(X >= count | lambda = c_a c_b / n) for a merge (significance guard)
    pub alpha: f64,
    /// recompute components every `period` observed tasks
    pub period: usize,
}

impl AnchorCfg {
    pub fn off() -> Self {
        AnchorCfg { enabled: false, lift_min: 4.0, min_count: 3, alpha: 1.0, period: 25 }
    }
    pub fn on(lift_min: f64, min_count: u32) -> Self {
        AnchorCfg { enabled: true, lift_min, min_count, alpha: 1e-3, period: 25 }
    }
    pub fn on_alpha(lift_min: f64, alpha: f64) -> Self {
        AnchorCfg { enabled: true, lift_min, min_count: 3, alpha, period: 25 }
    }
}

fn erfc(x: f64) -> f64 {
    // Numerical Recipes erfcc, relative error < 1.2e-7
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))).exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

/// P(X >= k) for X ~ Poisson(lam).
pub fn poisson_tail(k: u32, lam: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if lam <= 0.0 {
        return 0.0;
    }
    if lam < 30.0 {
        let mut term = (-lam).exp();
        let mut cdf = 0.0;
        for i in 0..k {
            cdf += term;
            term *= lam / (i as f64 + 1.0);
        }
        (1.0 - cdf).max(0.0)
    } else {
        let z = (k as f64 - 0.5 - lam) / lam.sqrt();
        0.5 * erfc(z / std::f64::consts::SQRT_2)
    }
}

pub struct Anchors {
    pub cfg: AnchorCfg,
    ids: HashMap<String, u32>,
    cnt: Vec<u32>,
    /// feature-id history of every observed task (cluster-level statistics are recomputed from it)
    hist: Vec<Vec<u32>>,
    rep: Vec<u32>,
    since: usize,
    oracle: Option<HashMap<String, usize>>,
    oracle_root: HashMap<usize, u32>,
    pub rebuilds: usize,
}

fn uf_find(uf: &mut [u32], mut x: u32) -> u32 {
    while uf[x as usize] != x {
        let p = uf[uf[x as usize] as usize];
        uf[x as usize] = p;
        x = p;
    }
    x
}

impl Anchors {
    pub fn new(cfg: AnchorCfg) -> Self {
        Anchors { cfg, ids: HashMap::new(), cnt: vec![], hist: vec![], rep: vec![], since: 0, oracle: None, oracle_root: HashMap::new(), rebuilds: 0 }
    }
    /// Diagnostic upper bound: tokens are the TRUE topic of each feature.
    pub fn oracle(map: HashMap<String, usize>) -> Self {
        let mut a = Anchors::new(AnchorCfg { enabled: true, ..AnchorCfg::off() });
        a.oracle = Some(map);
        a
    }

    fn intern(&mut self, f: &str) -> u32 {
        if let Some(&i) = self.ids.get(f) {
            return i;
        }
        let i = self.cnt.len() as u32;
        self.ids.insert(f.to_string(), i);
        self.cnt.push(0);
        let tok = match self.oracle.as_ref().and_then(|m| m.get(f)).copied() {
            Some(topic) => *self.oracle_root.entry(topic).or_insert(i),
            None => i,
        };
        self.rep.push(tok);
        i
    }

    /// Token set (deduped, sorted) of `feats` without observing.
    pub fn tokens(&self, feats: &[String]) -> Vec<u32> {
        let mut v: Vec<u32> = feats.iter().filter_map(|f| self.ids.get(f).map(|&i| self.rep[i as usize])).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// Observe a task and return its tokens. Second value: tokens re-mapped by a rebuild (old token -> new token),
    /// empty when nothing changed.
    pub fn observe(&mut self, feats: &[String]) -> (Vec<u32>, Vec<(u32, u32)>) {
        let mut fids: Vec<u32> = feats.iter().map(|f| self.intern(f)).collect();
        fids.sort_unstable();
        fids.dedup();
        let mut moved = vec![];
        if self.cfg.enabled && self.oracle.is_none() {
            for &i in &fids {
                self.cnt[i as usize] += 1;
            }
            self.hist.push(fids.clone());
            self.since += 1;
            if self.since >= self.cfg.period.max(1) {
                self.since = 0;
                moved = self.rebuild();
            }
        }
        let mut t: Vec<u32> = fids.iter().map(|&i| self.rep[i as usize]).collect();
        t.sort_unstable();
        t.dedup();
        (t, moved)
    }

    /// Agglomerative merging from the CURRENT clustering (merge-only, so statistics can be migrated): token-level
    /// co-occurrence counts are recomputed from the task history; token pairs with count >= min_count and
    /// lift P(a,b)/(P(a)P(b)) >= lift_min are unioned; repeated until a fixpoint (<= 8 rounds).
    fn rebuild(&mut self) -> Vec<(u32, u32)> {
        let nf = self.cnt.len();
        let mut uf: Vec<u32> = self.rep.clone();
        let n = self.hist.len() as f64;
        for _ in 0..8 {
            let roots: Vec<u32> = (0..nf as u32).map(|i| uf_find(&mut uf, i)).collect();
            let mut c: HashMap<u32, u32> = HashMap::new();
            let mut p: HashMap<(u32, u32), u32> = HashMap::new();
            for task in &self.hist {
                let mut t: Vec<u32> = task.iter().map(|&f| roots[f as usize]).collect();
                t.sort_unstable();
                t.dedup();
                for &a in &t {
                    *c.entry(a).or_insert(0) += 1;
                }
                for x in 0..t.len() {
                    for y in x + 1..t.len() {
                        *p.entry((t[x], t[y])).or_insert(0) += 1;
                    }
                }
            }
            let mut merged = false;
            for (&(a, b), &k) in &p {
                if k < self.cfg.min_count {
                    continue;
                }
                let lift = k as f64 * n / (c[&a] as f64 * c[&b] as f64);
                if lift >= self.cfg.lift_min && poisson_tail(k, c[&a] as f64 * c[&b] as f64 / n) <= self.cfg.alpha {
                    let (ra, rb) = (uf_find(&mut uf, a), uf_find(&mut uf, b));
                    if ra != rb {
                        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
                        uf[hi as usize] = lo;
                        merged = true;
                    }
                }
            }
            if !merged {
                break;
            }
        }
        let new_rep: Vec<u32> = (0..nf as u32).map(|i| uf_find(&mut uf, i)).collect();
        let mut moved: Vec<(u32, u32)> = vec![];
        let mut seen: HashMap<u32, u32> = HashMap::new();
        for i in 0..nf {
            let (old, new) = (self.rep[i], new_rep[i]);
            if old != new && seen.insert(old, new).is_none() {
                moved.push((old, new));
            }
        }
        moved.sort_unstable();
        self.rep = new_rep;
        if !moved.is_empty() {
            self.rebuilds += 1;
        }
        moved
    }

    /// number of distinct tokens that cover > 1 feature (diagnostic)
    pub fn n_clusters(&self) -> usize {
        let mut sz: HashMap<u32, usize> = HashMap::new();
        for &r in &self.rep {
            *sz.entry(r).or_insert(0) += 1;
        }
        sz.values().filter(|&&c| c > 1).count()
    }
}

// ------------------------------------------------------------------------------------------------ inner learners over tokens
pub trait KInner {
    fn select(&mut self, toks: &[u32], cands: &[usize], rng: &mut Rng) -> usize;
    fn update(&mut self, toks: &[u32], agent: usize, r: f64);
    fn remap(&mut self, _m: &HashMap<u32, u32>) {}
    /// predicted success of `agent` for the task (None = no estimate)
    fn p_hat(&self, _toks: &[u32], _agent: usize) -> Option<f64> {
        None
    }
    fn name(&self) -> String;
}

fn tok_task(toks: &[u32]) -> Task {
    Task { id: 0, task_type: None, features: toks.iter().map(|t| format!("#{t}")).collect() }
}

/// hebbian-robust over token strings ("#id"); rows are NOT migrated on anchor merges (documented limitation).
pub struct HebInner(pub HebbianRouter);
impl HebInner {
    pub fn robust(n: usize) -> Self {
        HebInner(HebbianRouter::new(n, HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }))
    }
}
impl KInner for HebInner {
    fn select(&mut self, toks: &[u32], cands: &[usize], rng: &mut Rng) -> usize {
        self.0.select(&tok_task(toks), cands, rng)
    }
    fn update(&mut self, toks: &[u32], agent: usize, r: f64) {
        self.0.update(&tok_task(toks), agent, r)
    }
    fn name(&self) -> String {
        "hebbian-robust".into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evict {
    /// victim = least important (evidence x advantage); merged into its nearest pattern (consolidation)
    ImportanceMerge,
    /// same victim, simply dropped
    ImportanceDrop,
    Lru,
    Fifo,
    CountDrop,
}

#[derive(Clone, Debug)]
pub struct MemCfg {
    /// optimistic prior mean for an agent with no evidence in a source
    pub init: f64,
    /// prior strength (pseudo-count)
    pub c0: f64,
    /// softmax temperature (<= 0: argmax)
    pub temp: f64,
    pub eps: f64,
    /// exact key-set patterns on/off (off = unigram token tables only == per-feature SONA memory)
    pub use_sets: bool,
    /// number of nearest (non-identical) patterns blended in (0 = exact pattern only)
    pub knn_k: usize,
    /// minimum Jaccard similarity for a neighbour
    pub s_min: f64,
    /// similarity exponent in the blend weight
    pub gamma: f64,
    /// confidence saturation: source weight x W/(W+kappa)
    pub kappa: f64,
    /// weight of each unigram token source
    pub lam_u: f64,
    /// credit spread: neighbours with sim >= s_min get update weight spread*sim (0 = exact pattern only)
    pub spread: f64,
    /// learned key selection: drop tokens with >= key_n observations whose shrunk residual info < key_theta
    pub key_sel: bool,
    pub key_n: f64,
    pub key_theta: f64,
    /// capacity of the pattern set (0 = unlimited)
    pub cap: usize,
    pub evict: Evict,
}

impl Default for MemCfg {
    fn default() -> Self {
        MemCfg {
            init: 0.7,
            c0: 1.0,
            temp: 0.05,
            eps: 0.0,
            use_sets: true,
            knn_k: 3,
            s_min: 0.3,
            gamma: 1.0,
            kappa: 2.0,
            lam_u: 0.5,
            spread: 0.0,
            key_sel: false,
            key_n: 20.0,
            key_theta: 0.12,
            cap: 0,
            evict: Evict::ImportanceMerge,
        }
    }
}

#[derive(Clone, Debug)]
struct Pat {
    toks: Vec<u32>,
    s: Vec<f64>,
    w: Vec<f64>,
    last: u64,
    born: u64,
}

impl Pat {
    fn wtot(&self) -> f64 {
        self.w.iter().sum()
    }
}

pub struct KeyedMemory {
    pub cfg: MemCfg,
    n: usize,
    uni: HashMap<u32, (Vec<f64>, Vec<f64>)>,
    uni_obs: HashMap<u32, f64>,
    pats: Vec<Pat>,
    index: HashMap<Vec<u32>, usize>,
    gs: Vec<f64>,
    gw: Vec<f64>,
    clock: u64,
    pub evictions: usize,
    rr: u64,
}

fn jaccard(a: &[u32], b: &[u32]) -> f64 {
    let (mut i, mut j, mut inter) = (0, 0, 0usize);
    while i < a.len() && j < b.len() {
        if a[i] == b[j] {
            inter += 1;
            i += 1;
            j += 1;
        } else if a[i] < b[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    let uni = a.len() + b.len() - inter;
    if uni == 0 { 0.0 } else { inter as f64 / uni as f64 }
}

impl KeyedMemory {
    pub fn new(n_agents: usize, cfg: MemCfg) -> Self {
        KeyedMemory {
            cfg,
            n: n_agents,
            uni: HashMap::new(),
            uni_obs: HashMap::new(),
            pats: vec![],
            index: HashMap::new(),
            gs: vec![0.0; n_agents],
            gw: vec![0.0; n_agents],
            clock: 0,
            evictions: 0,
            rr: 0x9E37_79B9,
        }
    }

    pub fn n_patterns(&self) -> usize {
        self.pats.len()
    }

    fn grow(&mut self, agent: usize) {
        if agent >= self.n {
            self.n = agent + 1;
            self.gs.resize(self.n, 0.0);
            self.gw.resize(self.n, 0.0);
        }
    }

    fn global(&self, a: usize) -> f64 {
        if a < self.n && self.gw[a] > 0.0 { self.gs[a] / self.gw[a] } else { self.cfg.init }
    }

    /// learned key: tokens minus those confirmed uninformative.
    fn key_of(&self, toks: &[u32]) -> Vec<u32> {
        if !self.cfg.key_sel {
            return toks.to_vec();
        }
        let c = 4.0;
        let kept: Vec<u32> = toks
            .iter()
            .cloned()
            .filter(|t| {
                let obs = self.uni_obs.get(t).copied().unwrap_or(0.0);
                if obs < self.cfg.key_n {
                    return true;
                }
                let Some((s, w)) = self.uni.get(t) else { return true };
                let mut info: f64 = 0.0;
                for a in 0..self.n.min(s.len()) {
                    if w[a] >= 2.0 {
                        let g = self.global(a);
                        let d = (s[a] + g * c) / (w[a] + c) - g;
                        info = info.max(d.abs());
                    }
                }
                info >= self.cfg.key_theta
            })
            .collect();
        if kept.is_empty() { toks.to_vec() } else { kept }
    }

    /// blended per-agent score for the task; unseen sources fall back to `init_of(a)`.
    fn scores(&self, toks: &[u32], init_of: &dyn Fn(usize) -> f64) -> Vec<f64> {
        let n = self.n;
        let c0 = self.cfg.c0;
        let mut num = vec![0.0; n];
        let mut den = 0.0;
        let mut add = |wt: f64, s: &[f64], w: &[f64]| {
            if wt <= 0.0 {
                return;
            }
            den += wt;
            for a in 0..n {
                let (sa, wa) = (s.get(a).copied().unwrap_or(0.0), w.get(a).copied().unwrap_or(0.0));
                num[a] += wt * (sa + init_of(a) * c0) / (wa + c0);
            }
        };
        if self.cfg.use_sets {
            let key = self.key_of(toks);
            if let Some(&pi) = self.index.get(&key) {
                let p = &self.pats[pi];
                let wt = p.wtot() / (p.wtot() + self.cfg.kappa);
                add(wt, &p.s, &p.w);
            }
            if self.cfg.knn_k > 0 {
                let mut near: Vec<(f64, usize)> = self
                    .pats
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.toks != key)
                    .map(|(i, p)| (jaccard(&key, &p.toks), i))
                    .filter(|x| x.0 >= self.cfg.s_min)
                    .collect();
                near.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                for &(sim, i) in near.iter().take(self.cfg.knn_k) {
                    let p = &self.pats[i];
                    let wt = sim.powf(self.cfg.gamma) * p.wtot() / (p.wtot() + self.cfg.kappa);
                    add(wt, &p.s, &p.w);
                }
            }
        }
        if self.cfg.lam_u > 0.0 {
            for t in toks {
                if let Some((s, w)) = self.uni.get(t) {
                    let tot: f64 = w.iter().sum();
                    add(self.cfg.lam_u * tot / (tot + self.cfg.kappa), s, w);
                }
            }
        }
        if den <= 0.0 {
            return (0..n).map(init_of).collect();
        }
        num.iter().map(|x| x / den).collect()
    }

    /// pattern-level evidence available for `toks`: weight of the exact key-set pattern plus the similarity-weighted
    /// weight of the k nearest patterns (0 when set patterns are off). Used to gate the pattern memory against a fallback.
    pub fn set_support(&self, toks: &[u32]) -> f64 {
        if !self.cfg.use_sets {
            return 0.0;
        }
        let key = self.key_of(toks);
        let mut tot = self.index.get(&key).map_or(0.0, |&i| self.pats[i].wtot());
        if self.cfg.knn_k > 0 {
            let mut near: Vec<(f64, usize)> = self
                .pats
                .iter()
                .enumerate()
                .filter(|(_, p)| p.toks != key)
                .map(|(i, p)| (jaccard(&key, &p.toks), i))
                .filter(|x| x.0 >= self.cfg.s_min)
                .collect();
            near.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for &(sim, i) in near.iter().take(self.cfg.knn_k) {
                tot += sim * self.pats[i].wtot();
            }
        }
        tot
    }

    fn importance(&self, p: &Pat) -> f64 {
        // evidence x advantage of the pattern's best agent over its average agent (decision value, EWC-flavoured)
        let c0 = self.cfg.c0;
        let ms: Vec<f64> = (0..self.n).map(|a| (p.s.get(a).copied().unwrap_or(0.0) + self.cfg.init * c0) / (p.w.get(a).copied().unwrap_or(0.0) + c0)).collect();
        let best = ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        p.wtot() * (best - mean).max(0.0) + 1e-9 * p.wtot()
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, p) in self.pats.iter().enumerate() {
            self.index.insert(p.toks.clone(), i);
        }
    }

    fn evict_one(&mut self, protect: usize) {
        if self.pats.len() <= 1 {
            return;
        }
        let n = self.pats.len();
        let mut victim = usize::MAX;
        let mut vs = f64::INFINITY;
        for i in 0..n {
            if i == protect {
                continue;
            }
            let p = &self.pats[i];
            let sc = match self.cfg.evict {
                Evict::ImportanceMerge | Evict::ImportanceDrop => self.importance(p),
                Evict::Lru => p.last as f64,
                Evict::Fifo => p.born as f64,
                Evict::CountDrop => p.wtot(),
            };
            if sc < vs {
                vs = sc;
                victim = i;
            }
        }
        if victim == usize::MAX {
            return;
        }
        let v = self.pats.swap_remove(victim);
        // swap_remove moved the last element into `victim`; keep `protect` valid
        self.rebuild_index();
        self.evictions += 1;
        if self.cfg.evict == Evict::ImportanceMerge {
            let mut bi = usize::MAX;
            let mut bs = 0.0;
            for (i, p) in self.pats.iter().enumerate() {
                let sim = jaccard(&v.toks, &p.toks);
                if sim > bs {
                    bs = sim;
                    bi = i;
                }
            }
            if bi != usize::MAX {
                let tgt = &mut self.pats[bi];
                if tgt.s.len() < v.s.len() {
                    tgt.s.resize(v.s.len(), 0.0);
                    tgt.w.resize(v.w.len(), 0.0);
                }
                for a in 0..v.s.len() {
                    tgt.s[a] += bs * v.s[a];
                    tgt.w[a] += bs * v.w[a];
                }
            }
        }
    }
}

impl KInner for KeyedMemory {
    fn name(&self) -> String {
        format!("keyedmem(sets={},k={},sel={},cap={})", self.cfg.use_sets, self.cfg.knn_k, self.cfg.key_sel, self.cfg.cap)
    }

    fn select(&mut self, toks: &[u32], cands: &[usize], rng: &mut Rng) -> usize {
        assert!(!cands.is_empty());
        if self.cfg.eps > 0.0 && rng.f64() < self.cfg.eps {
            return cands[rng.below(cands.len())];
        }
        let init = self.cfg.init;
        let mut sc = self.scores(toks, &|_| init);
        let mx = *cands.iter().max().unwrap();
        if sc.len() <= mx {
            sc.resize(mx + 1, init);
        }
        if self.cfg.temp <= 0.0 {
            let mut b = cands[0];
            for &c in cands {
                if sc[c] > sc[b] {
                    b = c;
                }
            }
            return b;
        }
        let m = cands.iter().map(|&c| sc[c]).fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = cands.iter().map(|&c| ((sc[c] - m) / self.cfg.temp).exp()).collect();
        let tot: f64 = w.iter().sum();
        let mut u = rng.f64() * tot;
        for (i, &x) in w.iter().enumerate() {
            u -= x;
            if u <= 0.0 {
                return cands[i];
            }
        }
        *cands.last().unwrap()
    }

    fn update(&mut self, toks: &[u32], agent: usize, r: f64) {
        let r = r.clamp(0.0, 1.0);
        self.grow(agent);
        self.clock += 1;
        let n = self.n;
        self.gs[agent] += r;
        self.gw[agent] += 1.0;
        for &t in toks {
            let e = self.uni.entry(t).or_insert_with(|| (vec![0.0; n], vec![0.0; n]));
            if e.0.len() < n {
                e.0.resize(n, 0.0);
                e.1.resize(n, 0.0);
            }
            e.0[agent] += r;
            e.1[agent] += 1.0;
            *self.uni_obs.entry(t).or_insert(0.0) += 1.0;
        }
        if !self.cfg.use_sets {
            return;
        }
        let key = self.key_of(toks);
        let clock = self.clock;
        let pi = match self.index.get(&key) {
            Some(&i) => i,
            None => {
                self.pats.push(Pat { toks: key.clone(), s: vec![0.0; n], w: vec![0.0; n], last: clock, born: clock });
                let i = self.pats.len() - 1;
                self.index.insert(key.clone(), i);
                i
            }
        };
        {
            let p = &mut self.pats[pi];
            if p.s.len() < n {
                p.s.resize(n, 0.0);
                p.w.resize(n, 0.0);
            }
            p.s[agent] += r;
            p.w[agent] += 1.0;
            p.last = clock;
        }
        if self.cfg.spread > 0.0 {
            let sp = self.cfg.spread;
            let smin = self.cfg.s_min;
            for i in 0..self.pats.len() {
                if i == pi {
                    continue;
                }
                let sim = jaccard(&key, &self.pats[i].toks);
                if sim >= smin {
                    let p = &mut self.pats[i];
                    if p.s.len() < n {
                        p.s.resize(n, 0.0);
                        p.w.resize(n, 0.0);
                    }
                    p.s[agent] += sp * sim * r;
                    p.w[agent] += sp * sim;
                }
            }
        }
        if self.cfg.cap > 0 {
            while self.pats.len() > self.cfg.cap {
                let protect = self.index.get(&key).copied().unwrap_or(usize::MAX);
                self.evict_one(protect);
            }
        }
        self.rr = self.rr.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    }

    fn remap(&mut self, m: &HashMap<u32, u32>) {
        let f = |t: u32| m.get(&t).copied().unwrap_or(t);
        let mut old: Vec<(u32, (Vec<f64>, Vec<f64>))> = std::mem::take(&mut self.uni).into_iter().collect();
        old.sort_by_key(|x| x.0);
        for (t, (s, w)) in old {
            let nt = f(t);
            let e = self.uni.entry(nt).or_insert_with(|| (vec![0.0; s.len()], vec![0.0; s.len()]));
            if e.0.len() < s.len() {
                e.0.resize(s.len(), 0.0);
                e.1.resize(s.len(), 0.0);
            }
            for a in 0..s.len() {
                e.0[a] += s[a];
                e.1[a] += w[a];
            }
        }
        let mut oldo: Vec<(u32, f64)> = std::mem::take(&mut self.uni_obs).into_iter().collect();
        oldo.sort_by_key(|x| x.0);
        for (t, c) in oldo {
            *self.uni_obs.entry(f(t)).or_insert(0.0) += c;
        }
        let oldp = std::mem::take(&mut self.pats);
        let mut idx: HashMap<Vec<u32>, usize> = HashMap::new();
        for mut p in oldp {
            let mut k: Vec<u32> = p.toks.iter().map(|&t| f(t)).collect();
            k.sort_unstable();
            k.dedup();
            p.toks = k.clone();
            match idx.get(&k) {
                Some(&i) => {
                    let q: &mut Pat = &mut self.pats[i];
                    if q.s.len() < p.s.len() {
                        q.s.resize(p.s.len(), 0.0);
                        q.w.resize(p.w.len(), 0.0);
                    }
                    for a in 0..p.s.len() {
                        q.s[a] += p.s[a];
                        q.w[a] += p.w[a];
                    }
                    q.last = q.last.max(p.last);
                    q.born = q.born.min(p.born);
                }
                None => {
                    idx.insert(k, self.pats.len());
                    self.pats.push(p);
                }
            }
        }
        self.index = idx;
    }

    fn p_hat(&self, toks: &[u32], agent: usize) -> Option<f64> {
        let sc = self.scores(toks, &|a| self.global(a));
        sc.get(agent).copied()
    }
}

// ------------------------------------------------------------------------------------------------ guard over tokens
fn beta_bounds(a: f64, b: f64, z: f64) -> (f64, f64) {
    let n = a + b;
    let m = a / n;
    let sd = (m * (1.0 - m) / (n + 1.0)).sqrt();
    (m - z * sd, m + z * sd)
}

/// `prior::Guarded` with outcome counts keyed by anchor tokens. Keyword votes still use the raw features.
pub struct AnchorGuard<I: KInner> {
    pub inner: I,
    prior: KeywordPrior,
    pub cfg: GuardConfig,
    pub anchors: Anchors,
    counts: HashMap<u32, Vec<(f64, f64)>>,
    /// anchors also re-key the inner learner (false: inner sees raw-feature tokens; the guard still pools by anchor)
    pub decisions: [u64; 3],
}

impl<I: KInner> AnchorGuard<I> {
    pub fn new(inner: I, map: HashMap<String, usize>, n_agents: usize, cfg: GuardConfig, anchors: Anchors) -> Self {
        AnchorGuard { inner, prior: KeywordPrior::new(map, n_agents), cfg, anchors, counts: HashMap::new(), decisions: [0; 3] }
    }

    fn pooled(&self, toks: &[u32], agent: usize) -> (f64, f64) {
        if toks.is_empty() {
            return (0.0, 0.0);
        }
        let (mut s, mut f) = (0.0, 0.0);
        for t in toks {
            if let Some(&(cs, cf)) = self.counts.get(t).and_then(|r| r.get(agent)) {
                s += cs;
                f += cf;
            }
        }
        let n = toks.len() as f64;
        (s / n, f / n)
    }

    fn apply_moved(&mut self, moved: &[(u32, u32)]) {
        if moved.is_empty() {
            return;
        }
        let m: HashMap<u32, u32> = moved.iter().cloned().collect();
        let mut old: Vec<(u32, Vec<(f64, f64)>)> = std::mem::take(&mut self.counts).into_iter().collect();
        old.sort_by_key(|x| x.0);
        for (t, row) in old {
            let nt = m.get(&t).copied().unwrap_or(t);
            let e = self.counts.entry(nt).or_insert_with(|| vec![(0.0, 0.0); row.len()]);
            if e.len() < row.len() {
                e.resize(row.len(), (0.0, 0.0));
            }
            for (a, c) in row.iter().enumerate() {
                e[a].0 += c.0;
                e[a].1 += c.1;
            }
        }
        self.inner.remap(&m);
    }

    fn decide(&self, task: &Task, toks: &[u32], candidates: &[usize]) -> (u8, usize) {
        let votes = self.prior.votes(task, candidates);
        let Some(k) = votes
            .iter()
            .fold(None, |b: Option<(usize, usize)>, &(a, v)| match b {
                Some((_, bv)) if bv >= v => b,
                _ => Some((a, v)),
            })
            .map(|(a, _)| a)
        else {
            return (2, 0);
        };
        let (s, f) = self.pooled(toks, k);
        let (_, ucb_k) = beta_bounds(self.cfg.kw_a + s, self.cfg.kw_b + f, self.cfg.z);
        if ucb_k < self.cfg.tau {
            return (2, 0);
        }
        if self.cfg.challenge {
            let mut best = (ucb_k, usize::MAX);
            for &a in candidates {
                if a == k {
                    continue;
                }
                let (s, f) = self.pooled(toks, a);
                let (lcb, _) = beta_bounds(1.0 + s, 1.0 + f, self.cfg.z);
                if lcb > best.0 {
                    best = (lcb, a);
                }
            }
            if best.1 != usize::MAX {
                return (1, best.1);
            }
        }
        (0, k)
    }

    pub fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty());
        let (toks, moved) = self.anchors.observe(&task.features);
        self.apply_moved(&moved);
        let toks = if moved.is_empty() { toks } else { self.anchors.tokens(&task.features) };
        let (d, a) = self.decide(task, &toks, candidates);
        self.decisions[d as usize] += 1;
        if d == 2 { self.inner.select(&toks, candidates, rng) } else { a }
    }

    pub fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        let toks = self.anchors.tokens(&task.features);
        let n = self.prior.n_agents().max(agent + 1);
        for t in &toks {
            let row = self.counts.entry(*t).or_insert_with(|| vec![(0.0, 0.0); n]);
            if row.len() < n {
                row.resize(n, (0.0, 0.0));
            }
            if self.cfg.discount < 1.0 {
                for c in row.iter_mut() {
                    c.0 *= self.cfg.discount;
                    c.1 *= self.cfg.discount;
                }
            }
            row[agent].0 += r;
            row[agent].1 += 1.0 - r;
        }
        self.inner.update(&toks, agent, reward);
    }

    pub fn p_hat(&self, task: &Task, agent: usize) -> Option<f64> {
        let toks = self.anchors.tokens(&task.features);
        self.inner.p_hat(&toks, agent)
    }
}

// ------------------------------------------------------------------------------------------------ evidence-gated hybrid inner
/// Both learners are trained on every outcome; the decision comes from the pattern memory only when its pattern-level
/// evidence for this task's key reaches `min_w`, otherwise from hebbian-robust (the additive default).
pub struct HybridInner {
    pub heb: HebInner,
    pub mem: KeyedMemory,
    pub min_w: f64,
    pub used_mem: u64,
    pub used_heb: u64,
}
impl HybridInner {
    pub fn new(n: usize, cfg: MemCfg, min_w: f64) -> Self {
        HybridInner { heb: HebInner::robust(n), mem: KeyedMemory::new(n, cfg), min_w, used_mem: 0, used_heb: 0 }
    }
}
impl KInner for HybridInner {
    fn select(&mut self, toks: &[u32], cands: &[usize], rng: &mut Rng) -> usize {
        if self.mem.set_support(toks) >= self.min_w {
            self.used_mem += 1;
            self.mem.select(toks, cands, rng)
        } else {
            self.used_heb += 1;
            self.heb.select(toks, cands, rng)
        }
    }
    fn update(&mut self, toks: &[u32], agent: usize, r: f64) {
        self.heb.update(toks, agent, r);
        self.mem.update(toks, agent, r);
    }
    fn remap(&mut self, m: &HashMap<u32, u32>) {
        self.mem.remap(m);
    }
    fn p_hat(&self, toks: &[u32], agent: usize) -> Option<f64> {
        self.mem.p_hat(toks, agent)
    }
    fn name(&self) -> String {
        format!("hybrid(min_w={})", self.min_w)
    }
}
