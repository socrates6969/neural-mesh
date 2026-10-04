//! Shared pieces of the f2 efficiency lab (CAP-2): compact Hebbian state, worlds,
//! std-thread helpers, equivalence statistics. std only. Included from the f2_* bins
//! with `#[path = "f2_common/mod.rs"] mod common;`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::{HebbianConfig, INIT_BASELINE, INIT_WEIGHT};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{paired_diff, t_crit_95, t_two_sided_p, PairedDiff};

pub fn seeds(lo: u64, hi: u64) -> Vec<u64> {
    (lo..=hi).collect()
}
pub fn tune_seeds() -> Vec<u64> {
    seeds(1, 8)
}
pub fn held_seeds() -> Vec<u64> {
    seeds(31, 60)
}
pub fn confirm_seeds() -> Vec<u64> {
    seeds(61, 90)
}

/// Equivalence margin on expected success (absolute): "lossless" needs the CI inside +-MARGIN.
pub const MARGIN: f64 = 0.003;

pub fn n_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Dynamic-schedule parallel map with std threads (order-preserving).
pub fn par_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let n = items.len();
    let nt = n_threads().min(n).max(1);
    let next = AtomicUsize::new(0);
    let out: Vec<Mutex<Option<R>>> = (0..n).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..nt {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let r = f(&items[i]);
                    *out[i].lock().unwrap() = Some(r);
                }
            });
        }
    });
    out.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect()
}

// =============================================================================================
// Statistics helpers

/// One-sided-pair TOST p-value for |mean diff| < margin (paired differences `d`).
pub fn tost_p(d: &[f64], margin: f64) -> f64 {
    let n = d.len();
    if n < 2 {
        return 1.0;
    }
    let m = d.iter().sum::<f64>() / n as f64;
    let var = d.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1) as f64;
    let se = (var / n as f64).sqrt();
    if se < 1e-15 {
        return if m.abs() < margin { 0.0 } else { 1.0 };
    }
    let df = n - 1;
    let one_sided_upper = |t: f64| {
        let p2 = t_two_sided_p(t.abs(), df);
        if t >= 0.0 { p2 / 2.0 } else { 1.0 - p2 / 2.0 }
    };
    let p1 = one_sided_upper((m + margin) / se); // H0: diff <= -margin
    let p2 = one_sided_upper(-(m - margin) / se); // H0: diff >= +margin
    p1.max(p2)
}

#[derive(Clone, Debug)]
pub struct Cmp {
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
    pub p_equiv: f64,
    pub p_diff: f64,
    pub wins: usize,
    pub losses: usize,
    pub n: usize,
}

pub fn compare(a: &[f64], b: &[f64]) -> Cmp {
    let pd: PairedDiff = paired_diff(a, b);
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    Cmp {
        mean: pd.diff.mean,
        lo: pd.diff.lo,
        hi: pd.diff.hi,
        p_equiv: tost_p(&d, MARGIN),
        p_diff: neural_mesh::stats::paired_t_pvalue(a, b),
        wins: pd.wins,
        losses: pd.losses,
        n: d.len(),
    }
}

pub fn cmp_json(c: &Cmp, holm_equiv: Option<f64>) -> serde_json::Value {
    serde_json::json!({
        "diff": c.mean, "ci_lo": c.lo, "ci_hi": c.hi, "n": c.n, "wins": c.wins, "losses": c.losses,
        "p_diff": c.p_diff, "p_equiv_tost": c.p_equiv, "holm_p_equiv": holm_equiv,
        "ci_inside_margin": c.lo > -MARGIN && c.hi < MARGIN,
        "lossless_claim": holm_equiv.map(|h| h < 0.05 && c.lo > -MARGIN && c.hi < MARGIN).unwrap_or(false),
    })
}

pub fn fmt_cmp(c: &Cmp) -> String {
    format!("{:+.4} [{:+.4},{:+.4}]", c.mean, c.lo, c.hi)
}

pub fn ci_half(xs: &[f64]) -> (f64, f64, f64) {
    let n = xs.len();
    let m = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n.max(2) - 1) as f64;
    let hw = t_crit_95(n.max(2) - 1) * (var / n as f64).sqrt();
    (m, m - hw, m + hw)
}

// =============================================================================================
// Worlds

pub const BIG_AGENTS: usize = 32;
pub const BIG_TYPES: usize = 48;
pub const BIG_TASKS: usize = 4000;
const BIG_NOISE: usize = 40;

/// My own larger world (32 agents, 48 task types x 4 keywords + 40 noise keywords, Zipf type
/// popularity, 35% trap keyword map, mid-run drift on 25% of types).
pub struct BigWorld {
    tasks: Vec<Task>,
    ty: Vec<usize>,
    u: Vec<f64>,
    p_pre: Vec<Vec<f64>>,
    p_post: Vec<Vec<f64>>,
    kw: HashMap<String, usize>,
    n_agents: usize,
}

fn uni(r: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * r.f64()
}

impl BigWorld {
    pub fn new(seed: u64) -> Self {
        Self::new_sized(seed, BIG_AGENTS, BIG_TYPES, BIG_TASKS)
    }

    pub fn new_sized(seed: u64, na: usize, nt: usize, ntasks: usize) -> Self {
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xB16_B16);
        let mut p_pre = vec![vec![0.0; na]; nt];
        let mut kwa = vec![0usize; nt];
        let mut best_of = vec![0usize; nt];
        for t in 0..nt {
            for a in 0..na {
                p_pre[t][a] = uni(&mut r, 0.15, 0.45);
            }
            let best = r.below(na);
            p_pre[t][best] = uni(&mut r, 0.80, 0.90);
            let mut second = r.below(na - 1);
            if second >= best {
                second += 1;
            }
            p_pre[t][second] = uni(&mut r, 0.55, 0.68);
            best_of[t] = best;
            kwa[t] = if r.bernoulli(0.65) {
                best
            } else {
                let mut w = r.below(na - 1);
                if w >= best {
                    w += 1;
                }
                w
            };
        }
        let mut p_post = p_pre.clone();
        for t in 0..nt {
            if r.bernoulli(0.25) {
                let old = best_of[t];
                let mut new = r.below(na - 1);
                if new >= old {
                    new += 1;
                }
                p_post[t][old] = uni(&mut r, 0.30, 0.50);
                p_post[t][new] = uni(&mut r, 0.80, 0.90);
            }
        }
        // zipf popularity
        let mut cdf = Vec::with_capacity(nt);
        let mut acc = 0.0;
        for t in 0..nt {
            acc += 1.0 / (t as f64 + 1.0);
            cdf.push(acc);
        }
        let total = acc;
        let mut tasks = Vec::with_capacity(ntasks);
        let mut ty = Vec::with_capacity(ntasks);
        let mut u = Vec::with_capacity(ntasks);
        for id in 0..ntasks {
            let x = r.f64() * total;
            let t = cdf.iter().position(|&c| x < c).unwrap_or(nt - 1);
            let k = 2 + r.below(2);
            let mut idx = [0usize, 1, 2, 3];
            for i in (1..4).rev() {
                let j = r.below(i + 1);
                idx.swap(i, j);
            }
            let mut features: Vec<String> = idx[..k].iter().map(|&j| format!("t{t}k{j}")).collect();
            if r.bernoulli(0.25) {
                let mut o = r.below(nt - 1);
                if o >= t {
                    o += 1;
                }
                features.push(format!("t{o}k{}", r.below(4)));
            }
            if r.bernoulli(0.30) {
                features.push(format!("n{}", r.below(BIG_NOISE)));
            }
            for i in (1..features.len()).rev() {
                let j = r.below(i + 1);
                features.swap(i, j);
            }
            tasks.push(Task { id: id as u64, task_type: None, features });
            ty.push(t);
            u.push(r.f64());
        }
        let mut kw = HashMap::new();
        for t in 0..nt {
            for j in 0..4 {
                kw.insert(format!("t{t}k{j}"), kwa[t]);
            }
        }
        BigWorld { tasks, ty, u, p_pre, p_post, kw, n_agents: na }
    }
}

pub enum Env {
    Sim(World),
    Big(BigWorld),
}

impl Env {
    pub fn sim(seed: u64, sc: Scenario) -> Env {
        Env::Sim(World::new(seed, WorldConfig { n_tasks: 2000, scenario: sc, ..WorldConfig::default() }))
    }
    pub fn big(seed: u64) -> Env {
        Env::Big(BigWorld::new(seed))
    }
    pub fn big_sized(seed: u64, na: usize, nt: usize, ntasks: usize) -> Env {
        Env::Big(BigWorld::new_sized(seed, na, nt, ntasks))
    }
    pub fn n_agents(&self) -> usize {
        match self {
            Env::Sim(w) => w.n_agents(),
            Env::Big(b) => b.n_agents,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Env::Sim(w) => w.len(),
            Env::Big(b) => b.tasks.len(),
        }
    }
    pub fn task(&self, t: usize) -> &Task {
        match self {
            Env::Sim(w) => &w.task(t).task,
            Env::Big(b) => &b.tasks[t],
        }
    }
    pub fn prob(&self, t: usize, a: usize) -> f64 {
        match self {
            Env::Sim(w) => w.success_prob(t, &[a]),
            Env::Big(b) => {
                if t < b.tasks.len() / 2 { b.p_pre[b.ty[t]][a] } else { b.p_post[b.ty[t]][a] }
            }
        }
    }
    pub fn outcome(&self, t: usize, a: usize) -> f64 {
        match self {
            Env::Sim(w) => w.outcome(t, &[a]),
            Env::Big(b) => {
                if b.u[t] < self.prob(t, a) { 1.0 } else { 0.0 }
            }
        }
    }
    pub fn kw_map(&self) -> HashMap<String, usize> {
        match self {
            Env::Sim(w) => w.keyword_map_for(),
            Env::Big(b) => b.kw.clone(),
        }
    }
    pub fn tasks_vec(&self) -> Vec<Task> {
        (0..self.len()).map(|t| self.task(t).clone()).collect()
    }
}

/// World specs used by every part.
#[derive(Clone, Debug)]
pub enum WorldSpec {
    Sim(Scenario),
    Big,
}
impl WorldSpec {
    pub fn label(&self) -> String {
        match self {
            WorldSpec::Sim(s) => format!("sim-{}", s.label()),
            WorldSpec::Big => "big-32agents".into(),
        }
    }
    pub fn make(&self, seed: u64) -> Env {
        match self {
            WorldSpec::Sim(s) => Env::sim(seed, *s),
            WorldSpec::Big => Env::big(seed),
        }
    }
}
pub fn all_worlds() -> Vec<WorldSpec> {
    vec![WorldSpec::Sim(Scenario::Fixed), WorldSpec::Sim(Scenario::Random), WorldSpec::Sim(Scenario::NoTraps), WorldSpec::Big]
}

#[derive(Clone, Debug, Default)]
pub struct RunStats {
    pub h250: f64,
    pub h500: f64,
    pub full: f64,
}

/// Sequential online loop: select, observe the realised Bernoulli outcome, update.
pub fn run_policy<R: Router>(env: &Env, pol: &mut R, seed: u64) -> RunStats {
    let n = env.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut cum = 0.0;
    let mut out = RunStats::default();
    for t in 0..env.len() {
        let task = env.task(t);
        let a = pol.select(task, &cands, &mut rng);
        cum += env.prob(t, a);
        pol.update(task, a, env.outcome(t, a));
        if t + 1 == 250 {
            out.h250 = cum / 250.0;
        }
        if t + 1 == 500 {
            out.h500 = cum / 500.0;
        }
    }
    out.full = cum / env.len() as f64;
    out
}

// =============================================================================================
// MiniHeb: the Hebbian-robust router re-implemented over selectable state storage.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WKind {
    F64,
    F32,
    /// IEEE half precision per cell (every cell re-rounded after each update)
    F16,
    /// absolute fixed point on [0,1] (x3 style: the whole row is re-rounded after every update)
    Fx { bits: u8, sr: bool },
    /// block floating point: shared f32 row scale (decay and row-cap folded into the scale
    /// exactly, power-of-two re-expansion) + `bits`-bit integer codes; only the delta is rounded
    Bf { bits: u8, sr: bool },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BKind {
    F64,
    F32,
    U16,
    U8Sr,
}

#[derive(Clone, Debug)]
pub struct Scheme {
    pub name: String,
    pub w: WKind,
    pub b: BKind,
    /// Bf8 rows are promoted to 16 bits after this many updates (per-row adaptive precision).
    pub promote_after: Option<u32>,
    /// Bf8 rows whose top-2 cells are closer than this many 8-bit codes (after >= 8 updates) are promoted.
    pub promote_gap: Option<u16>,
    /// Row budget: when exceeded, the coldest row (by decayed update count) is evicted.
    pub max_rows: Option<usize>,
}

impl Scheme {
    pub fn new(name: &str, w: WKind, b: BKind) -> Scheme {
        Scheme { name: name.into(), w, b, promote_after: None, promote_gap: None, max_rows: None }
    }
}

#[derive(Clone, Debug)]
struct Row {
    s: f32,
    codes: Vec<u16>,
    wf: Vec<f64>,
    bits: u8,
    base: f64,
    n: u32,
    last: u64,
}

#[derive(Clone)]
pub struct MiniHeb {
    pub cfg: HebbianConfig,
    pub sch: Scheme,
    n_agents: usize,
    ids: HashMap<String, u32>,
    names: Vec<String>,
    rows: Vec<Option<Row>>,
    free: Vec<u32>,
    tick: u64,
    rngs: u64,
    pub evictions: u64,
}

fn qmax(bits: u8) -> f64 {
    ((1u32 << bits) - 1) as f64
}

impl MiniHeb {
    pub fn new(n_agents: usize, sch: Scheme) -> Self {
        MiniHeb {
            cfg: heb_robust_cfg(),
            sch,
            n_agents,
            ids: HashMap::new(),
            names: Vec::new(),
            rows: Vec::new(),
            free: Vec::new(),
            tick: 0,
            rngs: 0x1234_5678_9ABC_DEF1,
            evictions: 0,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.ids.len()
    }

    fn u01(&mut self) -> f64 {
        // xorshift64*
        let mut x = self.rngs;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rngs = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn rnd(&mut self, x: f64, sr: bool) -> f64 {
        if sr {
            let fl = x.floor();
            if self.u01() < x - fl { fl + 1.0 } else { fl }
        } else {
            (x + 0.5).floor()
        }
    }

    fn qbase(&mut self, x: f64) -> f64 {
        match self.sch.b {
            BKind::F64 => x,
            BKind::F32 => x as f32 as f64,
            BKind::U16 => (x.clamp(0.0, 1.0) * 65535.0).round() / 65535.0,
            BKind::U8Sr => self.rnd(x.clamp(0.0, 1.0) * 255.0, true).clamp(0.0, 255.0) / 255.0,
        }
    }

    fn init_bits(&self) -> u8 {
        match self.sch.w {
            WKind::Fx { bits, .. } | WKind::Bf { bits, .. } => bits,
            _ => 0,
        }
    }

    fn new_row(&self) -> Row {
        let n = self.n_agents;
        let bits = self.init_bits();
        let base = match self.sch.b {
            BKind::F64 | BKind::F32 | BKind::U16 => INIT_BASELINE,
            BKind::U8Sr => (INIT_BASELINE * 255.0).round() / 255.0,
        };
        match self.sch.w {
            WKind::F64 => Row { s: 1.0, codes: vec![], wf: vec![INIT_WEIGHT; n], bits, base, n: 0, last: self.tick },
            WKind::F32 => Row { s: 1.0, codes: vec![], wf: vec![INIT_WEIGHT as f32 as f64; n], bits, base, n: 0, last: self.tick },
            WKind::F16 => Row { s: 1.0, codes: vec![], wf: vec![round_f16(INIT_WEIGHT); n], bits, base, n: 0, last: self.tick },
            WKind::Fx { bits, .. } => {
                let c = (INIT_WEIGHT * qmax(bits)).round() as u16;
                Row { s: 1.0, codes: vec![c; n], wf: vec![], bits, base, n: 0, last: self.tick }
            }
            WKind::Bf { bits, .. } => {
                Row { s: INIT_WEIGHT as f32, codes: vec![qmax(bits) as u16; n], wf: vec![], bits, base, n: 0, last: self.tick }
            }
        }
    }

    fn cell(&self, r: &Row, a: usize) -> f64 {
        match self.sch.w {
            WKind::F64 | WKind::F32 | WKind::F16 => r.wf[a],
            WKind::Fx { .. } => r.codes[a] as f64 / qmax(r.bits),
            WKind::Bf { .. } => r.s as f64 * r.codes[a] as f64 / qmax(r.bits),
        }
    }

    pub fn weight_by_id(&self, id: u32, a: usize) -> f64 {
        match self.rows.get(id as usize).and_then(|x| x.as_ref()) {
            Some(r) => self.cell(r, a),
            None => INIT_WEIGHT,
        }
    }

    pub fn weight(&self, feature: &str, a: usize) -> f64 {
        match self.ids.get(feature) {
            Some(&id) => self.weight_by_id(id, a),
            None => INIT_WEIGHT,
        }
    }

    pub fn baseline(&self, feature: &str) -> f64 {
        match self.ids.get(feature).and_then(|&i| self.rows[i as usize].as_ref()) {
            Some(r) => r.base,
            None => INIT_BASELINE,
        }
    }

    /// Active features (dedup, order kept) -> row ids (u32::MAX for unseen).
    pub fn lookup_ids(&self, task: &Task) -> Vec<u32> {
        let mut seen: Vec<&str> = Vec::with_capacity(task.features.len() + 1);
        let mut out = Vec::with_capacity(task.features.len() + 1);
        for f in &task.features {
            if !seen.contains(&f.as_str()) {
                seen.push(f.as_str());
                out.push(self.ids.get(f.as_str()).copied().unwrap_or(u32::MAX));
            }
        }
        if let Some(t) = &task.task_type {
            let k = format!("type:{t}");
            out.push(self.ids.get(k.as_str()).copied().unwrap_or(u32::MAX));
        }
        out
    }

    pub fn score_ids(&self, ids: &[u32], a: usize) -> f64 {
        let mut s = 0.0;
        for &id in ids {
            s += if id == u32::MAX { INIT_WEIGHT } else { self.weight_by_id(id, a) };
        }
        s
    }

    /// Read-only routing decision (same draws as `HebbianRouter::select`).
    pub fn select_ids(&self, ids: &[u32], candidates: &[usize], rng: &mut Rng) -> usize {
        let eps = self.cfg.epsilon;
        if rng.f64() < eps {
            return candidates[rng.below(candidates.len())];
        }
        let scores: Vec<f64> = candidates.iter().map(|&c| self.score_ids(ids, c)).collect();
        if self.cfg.temperature <= 0.0 {
            let _ = rng.f64();
            let mut best = 0;
            for i in 1..scores.len() {
                if scores[i] > scores[best] {
                    best = i;
                }
            }
            return candidates[best];
        }
        let logits: Vec<f64> = scores.iter().map(|s| s / self.cfg.temperature).collect();
        candidates[softmax_sample(&logits, rng)]
    }

    pub fn select_ref(&self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let ids = self.lookup_ids(task);
        self.select_ids(&ids, candidates, rng)
    }

    fn get_or_make(&mut self, f: &str, protect: &[u32]) -> u32 {
        if let Some(&id) = self.ids.get(f) {
            return id;
        }
        if let Some(cap) = self.sch.max_rows
            && self.ids.len() >= cap
        {
            // evict the coldest row (decayed update count), never one used by this task
            let mut worst: Option<(f64, u32)> = None;
            for (i, r) in self.rows.iter().enumerate() {
                if let Some(r) = r {
                    if protect.contains(&(i as u32)) {
                        continue;
                    }
                    let age = (self.tick - r.last) as f64;
                    let u = r.n as f64 * 0.9995f64.powf(age);
                    if worst.is_none_or(|(wu, _)| u < wu) {
                        worst = Some((u, i as u32));
                    }
                }
            }
            if let Some((_, i)) = worst {
                let name = std::mem::take(&mut self.names[i as usize]);
                self.ids.remove(&name);
                self.rows[i as usize] = None;
                self.free.push(i);
                self.evictions += 1;
            }
        }
        let row = self.new_row();
        let id = if let Some(i) = self.free.pop() {
            self.rows[i as usize] = Some(row);
            self.names[i as usize] = f.to_string();
            i
        } else {
            self.rows.push(Some(row));
            self.names.push(f.to_string());
            (self.rows.len() - 1) as u32
        };
        self.ids.insert(f.to_string(), id);
        id
    }

    pub fn update_task(&mut self, task: &Task, agent: usize, reward: f64) {
        if !reward.is_finite() {
            return;
        }
        let reward = reward.clamp(0.0, 1.0);
        self.tick += 1;
        // active features, deduplicated
        let mut feats: Vec<String> = Vec::with_capacity(task.features.len() + 1);
        for f in &task.features {
            if !feats.contains(f) {
                feats.push(f.clone());
            }
        }
        if let Some(t) = &task.task_type {
            let k = format!("type:{t}");
            if !feats.contains(&k) {
                feats.push(k);
            }
        }
        let mut protect: Vec<u32> = Vec::new();
        for f in feats {
            let id = self.get_or_make(&f, &protect);
            protect.push(id);
            let mut row = self.rows[id as usize].take().unwrap();
            self.update_row(&mut row, agent, reward);
            row.n = row.n.saturating_add(1);
            row.last = self.tick;
            self.rows[id as usize] = Some(row);
        }
    }

    fn update_row(&mut self, row: &mut Row, agent: usize, reward: f64) {
        let (eta, decay, cap, alpha) = (self.cfg.eta, self.cfg.decay, self.cfg.row_cap, self.cfg.baseline_alpha);
        let b = row.base;
        let e = reward - b;
        let delta = eta * 1.0 * e;
        match self.sch.w {
            WKind::F64 | WKind::F32 | WKind::F16 => {
                for w in row.wf.iter_mut() {
                    *w *= 1.0 - decay;
                }
                row.wf[agent] += delta;
                for w in row.wf.iter_mut() {
                    *w = w.clamp(0.0, 1.0);
                }
                let sum: f64 = row.wf.iter().sum();
                if cap > 0.0 && sum > cap {
                    let k = cap / sum;
                    for w in row.wf.iter_mut() {
                        *w *= k;
                    }
                }
                if self.sch.w == WKind::F32 {
                    for w in row.wf.iter_mut() {
                        *w = *w as f32 as f64;
                    }
                }
                if self.sch.w == WKind::F16 {
                    for w in row.wf.iter_mut() {
                        *w = round_f16(*w);
                    }
                }
            }
            WKind::Fx { bits, sr } => {
                let qm = qmax(bits);
                let mut w: Vec<f64> = row.codes.iter().map(|&c| c as f64 / qm).collect();
                for x in w.iter_mut() {
                    *x *= 1.0 - decay;
                }
                w[agent] += delta;
                for x in w.iter_mut() {
                    *x = x.clamp(0.0, 1.0);
                }
                let sum: f64 = w.iter().sum();
                if cap > 0.0 && sum > cap {
                    let k = cap / sum;
                    for x in w.iter_mut() {
                        *x *= k;
                    }
                }
                for (i, x) in w.iter().enumerate() {
                    let c = self.rnd(x * qm, sr).clamp(0.0, qm);
                    row.codes[i] = c as u16;
                }
            }
            WKind::Bf { sr, .. } => {
                self.update_bf(row, agent, delta, sr, decay, cap);
            }
        }
        let nb = b + alpha * (reward - b);
        row.base = self.qbase(nb);
        // per-row adaptive precision: promote hot rows 8 -> 16 bits (exact: code * 257)
        let mut promote = false;
        if let (WKind::Bf { bits: 8, .. }, 8) = (self.sch.w, row.bits) {
            if let Some(p) = self.sch.promote_after
                && row.n + 1 >= p
            {
                promote = true;
            }
            if let Some(g) = self.sch.promote_gap
                && row.n + 1 >= 8
            {
                let (mut t1, mut t2) = (0u16, 0u16);
                for &c in &row.codes {
                    if c > t1 {
                        t2 = t1;
                        t1 = c;
                    } else if c > t2 {
                        t2 = c;
                    }
                }
                if t1 - t2 < g {
                    promote = true;
                }
            }
        }
        if promote {
            for c in row.codes.iter_mut() {
                *c *= 257;
            }
            row.bits = 16;
        }
    }

    fn update_bf(&mut self, row: &mut Row, agent: usize, delta: f64, sr: bool, decay: f64, cap: f64) {
        let qm = qmax(row.bits);
        let mut s = (row.s as f64 * (1.0 - decay)).max(1e-30);
        let dq = delta / s * qm;
        let mut nq = self.rnd(row.codes[agent] as f64 + dq, sr).max(0.0);
        while nq > qm {
            // deterministic halving (equal cells stay equal, so untouched cells keep a shared code
            // and the sparse layout stays exact and small)
            for c in row.codes.iter_mut() {
                *c = ((*c as u32 + 1) / 2) as u16;
            }
            nq = ((nq as u32 + 1) / 2) as f64;
            s *= 2.0;
        }
        // keep the weight <= 1
        if s * nq / qm > 1.0 {
            nq = (qm / s).floor();
        }
        row.codes[agent] = nq as u16;
        let sum_codes: f64 = row.codes.iter().map(|&c| c as f64).sum();
        let sum = s * sum_codes / qm;
        if cap > 0.0 && sum > cap {
            s *= cap / sum;
        }
        // exact power-of-two re-expansion
        let mut maxq = *row.codes.iter().max().unwrap() as u32;
        let qmu = qm as u32;
        while maxq > 0 && maxq * 2 <= qmu {
            for c in row.codes.iter_mut() {
                *c *= 2;
            }
            maxq *= 2;
            s /= 2.0;
        }
        row.s = s as f32;
    }

    // ---------------------------------------------------------------------------------------
    // Serialised (packed) state. This is the byte count that matters for persistence.

    fn raw_cell(&self, r: &Row, a: usize) -> (u64, u32) {
        match self.sch.w {
            WKind::F64 => (r.wf[a].to_bits(), 64),
            WKind::F32 => ((r.wf[a] as f32).to_bits() as u64, 32),
            WKind::F16 => (f16_bits(r.wf[a]) as u64, 16),
            _ => (r.codes[a] as u64, r.bits as u32),
        }
    }

    /// Packed state. `names`: store feature names (u8 length + bytes) or only 4-byte hashes.
    /// Per row: name, n (u16 saturating; bit 15 = sparse layout), baseline, [scale f32],
    /// [bits flag byte for adaptive], cells dense (n_agents * width) or sparse
    /// (default cell, count u8, then agent u8 + cell per touched cell).
    pub fn to_bytes(&self, names: bool) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.n_agents as u16).to_le_bytes());
        out.extend_from_slice(&(self.ids.len() as u32).to_le_bytes());
        let adaptive = self.sch.adaptive();
        let bf = matches!(self.sch.w, WKind::Bf { .. });
        let mut order: Vec<(&String, &u32)> = self.ids.iter().collect();
        order.sort();
        for (name, &id) in order {
            let r = self.rows[id as usize].as_ref().unwrap();
            if names {
                out.push(name.len() as u8);
                out.extend_from_slice(name.as_bytes());
            } else {
                out.extend_from_slice(&(fnv(name) as u32).to_le_bytes());
            }
            // sparse or dense?
            let (_, width) = self.raw_cell(r, 0);
            let cells: Vec<u64> = (0..self.n_agents).map(|a| self.raw_cell(r, a).0).collect();
            let mut cnt: HashMap<u64, usize> = HashMap::new();
            for &c in &cells {
                *cnt.entry(c).or_insert(0) += 1;
            }
            let (&dflt, _) = cnt.iter().max_by_key(|(c, k)| (**k, std::cmp::Reverse(**c))).unwrap();
            let touched: Vec<usize> = (0..self.n_agents).filter(|&a| cells[a] != dflt).collect();
            let dense_bits = self.n_agents * width as usize;
            let sparse_bits = width as usize + 8 + touched.len() * (8 + width as usize);
            let sparse = self.n_agents <= 256 && sparse_bits < dense_bits;
            let nflag = (r.n.min(0x7FFF) as u16) | if sparse { 0x8000 } else { 0 };
            out.extend_from_slice(&nflag.to_le_bytes());
            match self.sch.b {
                BKind::F64 => out.extend_from_slice(&r.base.to_le_bytes()),
                BKind::F32 => out.extend_from_slice(&(r.base as f32).to_le_bytes()),
                BKind::U16 => out.extend_from_slice(&((r.base * 65535.0).round() as u16).to_le_bytes()),
                BKind::U8Sr => out.push((r.base * 255.0).round() as u8),
            }
            if bf {
                out.extend_from_slice(&r.s.to_le_bytes());
            }
            if adaptive {
                out.push(r.bits);
            }
            let mut bw = BitWriter::default();
            if sparse {
                bw.put(dflt, width);
                bw.put(touched.len() as u64, 8);
                for &a in &touched {
                    bw.put(a as u64, 8);
                    bw.put(cells[a], width);
                }
            } else {
                for &c in &cells {
                    bw.put(c, width);
                }
            }
            out.extend_from_slice(&bw.finish());
        }
        out
    }

    /// Inverse of [`MiniHeb::to_bytes`] with names (for the round-trip test). Row counts
    /// beyond 32767 are saturated by the format.
    pub fn from_bytes(bytes: &[u8], sch: Scheme) -> MiniHeb {
        let n_agents = u16::from_le_bytes([bytes[0], bytes[1]]) as usize;
        let n_rows = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]) as usize;
        let mut m = MiniHeb::new(n_agents, sch);
        let adaptive = m.sch.adaptive();
        let bf = matches!(m.sch.w, WKind::Bf { .. });
        let mut p = 6usize;
        for _ in 0..n_rows {
            let l = bytes[p] as usize;
            p += 1;
            let name = String::from_utf8(bytes[p..p + l].to_vec()).unwrap();
            p += l;
            let nflag = u16::from_le_bytes([bytes[p], bytes[p + 1]]);
            p += 2;
            let sparse = nflag & 0x8000 != 0;
            let n = (nflag & 0x7FFF) as u32;
            let base = match m.sch.b {
                BKind::F64 => {
                    let v = f64::from_le_bytes(bytes[p..p + 8].try_into().unwrap());
                    p += 8;
                    v
                }
                BKind::F32 => {
                    let v = f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap()) as f64;
                    p += 4;
                    v
                }
                BKind::U16 => {
                    let v = u16::from_le_bytes([bytes[p], bytes[p + 1]]) as f64 / 65535.0;
                    p += 2;
                    v
                }
                BKind::U8Sr => {
                    let v = bytes[p] as f64 / 255.0;
                    p += 1;
                    v
                }
            };
            let mut s = 1.0f32;
            if bf {
                s = f32::from_le_bytes(bytes[p..p + 4].try_into().unwrap());
                p += 4;
            }
            let mut bits = m.init_bits();
            if adaptive {
                bits = bytes[p];
                p += 1;
            }
            let width: u32 = match m.sch.w {
                WKind::F64 => 64,
                WKind::F32 => 32,
                WKind::F16 => 16,
                _ => bits as u32,
            };
            let mut br = BitReader { b: &bytes[p..], pos: 0 };
            let mut cells = vec![0u64; n_agents];
            if sparse {
                let d = br.get(width);
                let k = br.get(8) as usize;
                for c in cells.iter_mut() {
                    *c = d;
                }
                for _ in 0..k {
                    let a = br.get(8) as usize;
                    cells[a] = br.get(width);
                }
            } else {
                for c in cells.iter_mut() {
                    *c = br.get(width);
                }
            }
            p += br.pos.div_ceil(8);
            let mut row = Row { s, codes: vec![], wf: vec![], bits, base, n, last: 0 };
            match m.sch.w {
                WKind::F64 => row.wf = cells.iter().map(|&c| f64::from_bits(c)).collect(),
                WKind::F32 => row.wf = cells.iter().map(|&c| f32::from_bits(c as u32) as f64).collect(),
                WKind::F16 => row.wf = cells.iter().map(|&c| f16_from_bits(c as u16)).collect(),
                _ => row.codes = cells.iter().map(|&c| c as u16).collect(),
            }
            let id = m.rows.len() as u32;
            m.rows.push(Some(row));
            m.names.push(name.clone());
            m.ids.insert(name, id);
        }
        m
    }

    /// Max |w - w'| over all (feature, agent) cells present in both routers (for tests).
    pub fn max_cell_diff(&self, other: &MiniHeb) -> f64 {
        let mut d = 0.0f64;
        for (name, &id) in &self.ids {
            for a in 0..self.n_agents {
                d = d.max((self.weight_by_id(id, a) - other.weight(name, a)).abs());
            }
        }
        d
    }
}

impl Router for MiniHeb {
    fn name(&self) -> String {
        format!("miniheb[{}]", self.sch.name)
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        self.select_ref(task, candidates, rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.update_task(task, agent, reward);
    }
}

fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[derive(Default)]
struct BitWriter {
    buf: Vec<u8>,
    bitpos: usize,
}
impl BitWriter {
    fn put(&mut self, v: u64, n: u32) {
        for i in 0..n {
            if self.bitpos % 8 == 0 {
                self.buf.push(0);
            }
            if (v >> i) & 1 == 1 {
                let last = self.buf.len() - 1;
                self.buf[last] |= 1 << (self.bitpos % 8);
            }
            self.bitpos += 1;
        }
    }
    fn finish(self) -> Vec<u8> {
        self.buf
    }
}
struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
}
impl BitReader<'_> {
    fn get(&mut self, n: u32) -> u64 {
        let mut v = 0u64;
        for i in 0..n {
            let byte = self.b[self.pos / 8];
            if (byte >> (self.pos % 8)) & 1 == 1 {
                v |= 1 << i;
            }
            self.pos += 1;
        }
        v
    }
}

/// Same sampler as the crate's (pub(crate)) `hebbian::softmax_sample`: one `rng.f64()`.
pub fn softmax_sample(logits: &[f64], rng: &mut Rng) -> usize {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| if l.is_finite() { (l - max).exp() } else { 0.0 }).collect();
    let total: f64 = exps.iter().sum();
    let u = rng.f64();
    if !(total.is_finite() && total > 0.0) {
        return 0;
    }
    let mut acc = 0.0;
    let target = u * total;
    for (i, e) in exps.iter().enumerate() {
        acc += e;
        if target < acc {
            return i;
        }
    }
    exps.len() - 1
}

impl MiniHeb {
    pub fn row_n(&self, f: &str) -> u32 {
        self.ids.get(f).and_then(|&i| self.rows[i as usize].as_ref()).map(|r| r.n).unwrap_or(0)
    }
    pub fn n_agents(&self) -> usize {
        self.n_agents
    }
    /// Average row width in bits (diagnostic for the adaptive scheme).
    pub fn frac_hot(&self) -> f64 {
        let (mut h, mut n) = (0usize, 0usize);
        for r in self.rows.iter().flatten() {
            n += 1;
            if r.bits == 16 {
                h += 1;
            }
        }
        if n == 0 { 0.0 } else { h as f64 / n as f64 }
    }
}

/// `Evidence::Inner` hook: MeanFeature evidence read from the compact router's own row counts
/// (so a pruned feature falls back to the keyword prior, as a real compact system would).
pub fn mean_feature_n(m: &MiniHeb, t: &Task) -> f64 {
    if t.features.is_empty() {
        return 0.0;
    }
    t.features.iter().map(|f| m.row_n(f) as f64).sum::<f64>() / t.features.len() as f64
}

impl Scheme {
    pub fn adaptive(&self) -> bool {
        self.promote_after.is_some() || self.promote_gap.is_some()
    }
}

/// Round to the nearest IEEE binary16 value (ties to even), including subnormals.
pub fn round_f16(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let a = x.abs();
    let sgn = if x < 0.0 { -1.0 } else { 1.0 };
    if a >= 65520.0 {
        return sgn * 65504.0;
    }
    let ex = a.log2().floor() as i32;
    let step = if ex < -14 { 2f64.powi(-24) } else { 2f64.powi(ex - 10) };
    sgn * (a / step).round_ties_even() * step
}

pub fn f16_bits(x: f64) -> u16 {
    let v = round_f16(x);
    if v == 0.0 {
        return 0;
    }
    let a = v.abs();
    let sign: u16 = if v < 0.0 { 0x8000 } else { 0 };
    let ex = a.log2().floor() as i32;
    if ex < -14 {
        return sign | ((a / 2f64.powi(-24)).round() as u16);
    }
    let mant = ((a / 2f64.powi(ex) - 1.0) * 1024.0).round() as u16;
    sign | (((ex + 15) as u16) << 10) | mant
}

pub fn f16_from_bits(b: u16) -> f64 {
    let sign = if b & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 10) & 0x1F) as i32;
    let m = (b & 0x3FF) as f64;
    if e == 0 {
        sign * m * 2f64.powi(-24)
    } else {
        sign * (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    }
}
