//! h4 shared pieces (R4, memory-reduction track): two independent SYNTHETIC world generators, a kernel-regression case-based
//! router on top of `tiered::TieredMemory`, and the forgetting policies under a hard byte budget.
//! Everything is a seeded simulation (synthetic-until-validated-on-real-traffic). The router quality it measures is
//! "did retrieval give the right advice / did the routed task succeed", not LLM quality.
#![allow(dead_code)]

use std::time::Instant;

use neural_mesh::memory::synth::Rng;
use neural_mesh::tiered::{ColdKind, Compression, ForgetPolicy, LongKind, TieredConfig, TieredMemory, ENTRY_OVERHEAD};

pub const D: usize = 64;
pub const SHORT_CAP: usize = 64;
pub const KNN: usize = 8;
pub const S_MIN: f32 = 0.5;
pub const PRIOR: f64 = 0.4;
pub const PRIOR_M: f64 = 0.5;
pub const EPS: f64 = 0.08;

// ------------------------------------------------------------------------------------------------
// worlds

#[derive(Clone)]
pub struct Task {
    pub fam: i32, // -1 = one-off noise task
    pub emb: Vec<f32>,
    pub u: f64,   // outcome uniform (common random numbers across policies)
    pub ue: f64,  // explore uniform
    pub ua: usize, // explore / random agent
}

pub struct World {
    pub name: String,
    pub k: usize,
    pub steps: usize,
    pub eval_from: usize,
    pub tasks: Vec<Task>,
    pub junk: Vec<Option<Vec<f32>>>,
    p0: Vec<Vec<f32>>,
    p1: Vec<Vec<f32>>,
    drift_at: usize,
    pub pnoise: f64,
    pub nfam: usize,
}

impl World {
    pub fn p_of(&self, fam: i32, a: usize, t: usize) -> f64 {
        if fam < 0 {
            return self.pnoise;
        }
        let tab = if t >= self.drift_at { &self.p1 } else { &self.p0 };
        tab[fam as usize][a] as f64
    }
    pub fn best_of(&self, fam: i32, t: usize) -> usize {
        let tab = if t >= self.drift_at { &self.p1 } else { &self.p0 };
        let r = &tab[fam as usize];
        (0..r.len()).max_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap()).unwrap()
    }
    /// success probability of the best agent per task (oracle ceiling on the eval window)
    pub fn oracle(&self) -> f64 {
        let mut s = 0.0;
        let mut n = 0.0;
        for t in self.eval_from..self.steps {
            let f = self.tasks[t].fam;
            s += if f < 0 { self.pnoise } else { self.p_of(f, self.best_of(f, t), t) };
            n += 1.0;
        }
        s / n
    }
}

fn unit_rand(rng: &mut Rng) -> Vec<f32> {
    let v: Vec<f32> = (0..D).map(|_| rng.normal()).collect();
    norm(&v)
}
fn norm(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / n).collect()
}
fn noisy(c: &[f32], noise: f32, rng: &mut Rng) -> Vec<f32> {
    let v: Vec<f32> = c.iter().map(|x| x + noise * rng.normal() / (D as f32).sqrt()).collect();
    norm(&v)
}
fn pick_cdf(cdf: &[f64], u: f64) -> usize {
    cdf.iter().position(|&c| u <= c).unwrap_or(cdf.len() - 1)
}

#[derive(Clone, Debug)]
pub struct AParams {
    pub name: &'static str,
    pub nfam: usize,
    pub zipf: f64,
    pub noise_frac: f64,
    pub drift_frac: f64,
    pub steps: usize,
    pub k: usize,
    pub fam_noise: f32,
    pub eval_from_frac: f64,
}

pub const A_ZIPF: AParams = AParams {
    name: "A-zipf", nfam: 60, zipf: 1.1, noise_frac: 0.30, drift_frac: 0.25, steps: 4000, k: 6, fam_noise: 0.45, eval_from_frac: 0.6,
};
/// control: every family equally valuable, no junk, no drift -> forgetting must cost something
pub const A_ALLVALUED: AParams = AParams {
    name: "A-allvalued", nfam: 500, zipf: 0.0, noise_frac: 0.0, drift_frac: 0.0, steps: 4000, k: 6, fam_noise: 0.45, eval_from_frac: 0.5,
};
/// control: no recurrence at all (every task is a fresh vector) -> memory cannot help, forgetting must be harmless
pub const A_NOREUSE: AParams = AParams {
    name: "A-noreuse", nfam: 60, zipf: 1.1, noise_frac: 1.0, drift_frac: 0.0, steps: 4000, k: 6, fam_noise: 0.45, eval_from_frac: 0.6,
};

/// World generator A: recurring task families with Zipf popularity (heavy-tailed value), one-off noise tasks that never
/// recur (useless cases accumulate), and a mid-stream drift that makes the old best agent of some families stale.
pub fn gen_a(seed: u64, p: &AParams) -> World {
    let mut rng = Rng::new(0xA11CE ^ seed.wrapping_mul(0x9E37_79B9));
    let k = p.k;
    let cents: Vec<Vec<f32>> = (0..p.nfam).map(|_| unit_rand(&mut rng)).collect();
    let mut p0 = vec![vec![0f32; k]; p.nfam];
    for f in 0..p.nfam {
        let best = rng.below(k);
        for a in 0..k {
            p0[f][a] = if a == best { 0.85 } else { 0.20 + 0.25 * rng.uniform() as f32 };
        }
    }
    let mut p1 = p0.clone();
    let ndrift = (p.drift_frac * p.nfam as f64).round() as usize;
    let mut order: Vec<usize> = (0..p.nfam).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
    }
    for &f in order.iter().take(ndrift) {
        let old = (0..k).max_by(|&a, &b| p0[f][a].partial_cmp(&p0[f][b]).unwrap()).unwrap();
        let mut nb = rng.below(k);
        while nb == old {
            nb = rng.below(k);
        }
        p1[f][old] = 0.20 + 0.25 * rng.uniform() as f32;
        p1[f][nb] = 0.85;
    }
    let mut cdf = Vec::with_capacity(p.nfam);
    let mut acc = 0.0;
    let ws: Vec<f64> = (0..p.nfam).map(|i| 1.0 / ((i + 1) as f64).powf(p.zipf)).collect();
    let z: f64 = ws.iter().sum();
    for w in &ws {
        acc += w / z;
        cdf.push(acc);
    }
    let mut tasks = Vec::with_capacity(p.steps);
    for _ in 0..p.steps {
        let (fam, emb) = if rng.uniform() < p.noise_frac {
            (-1, unit_rand(&mut rng))
        } else {
            let f = pick_cdf(&cdf, rng.uniform());
            (f as i32, noisy(&cents[f], p.fam_noise, &mut rng))
        };
        tasks.push(Task { fam, emb, u: rng.uniform(), ue: rng.uniform(), ua: rng.below(k) });
    }
    World {
        name: p.name.to_string(),
        k,
        steps: p.steps,
        eval_from: (p.eval_from_frac * p.steps as f64) as usize,
        tasks,
        junk: vec![None; p.steps],
        p0,
        p1,
        drift_at: if ndrift > 0 { p.steps / 2 } else { usize::MAX },
        pnoise: 0.40,
        nfam: p.nfam,
    }
}

/// World generator B (independent of A): bursty topical "lessons". Families live in time windows (stale after they end),
/// centroids share a global direction (confusable geometry), per-family benefit is log-normal (heavy-tailed value, many
/// near-worthless), K=8 agents, and a stream of junk "trace" entries (random vectors, never helpful) is stored alongside.
pub fn gen_b(seed: u64) -> World {
    let mut rng = Rng::new(0xB0B0 ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let (nfam, steps, k) = (300usize, 5000usize, 8usize);
    let g = unit_rand(&mut rng);
    let cents: Vec<Vec<f32>> = (0..nfam)
        .map(|_| {
            let r = unit_rand(&mut rng);
            let v: Vec<f32> = g.iter().zip(&r).map(|(a, b)| 0.5 * a + 0.866 * b).collect();
            norm(&v)
        })
        .collect();
    let mut p0 = vec![vec![0f32; k]; nfam];
    for f in 0..nfam {
        // log-normal uplift of the best agent over a ~0.3 base (median ~0.2, tail to 0.6)
        let z = rng.normal() as f64;
        let up = (-1.6 + 0.9 * z).exp().clamp(0.02, 0.60) as f32;
        let best = rng.below(k);
        for a in 0..k {
            p0[f][a] = if a == best { 0.30 + up } else { 0.25 + 0.10 * rng.uniform() as f32 };
        }
    }
    let start: Vec<usize> = (0..nfam).map(|_| (rng.uniform() * 0.9 * steps as f64) as usize).collect();
    let len: Vec<usize> = (0..nfam)
        .map(|_| (((0.25 * steps as f64).ln() + 0.5 * rng.normal() as f64).exp() as usize).clamp(200, steps))
        .collect();
    let pop: Vec<f64> = (0..nfam).map(|_| (rng.normal() as f64).exp()).collect();
    let mut tasks = Vec::with_capacity(steps);
    let mut junk = Vec::with_capacity(steps);
    for t in 0..steps {
        let mut cdf = Vec::new();
        let mut ids = Vec::new();
        let mut acc = 0.0;
        for f in 0..nfam {
            if t >= start[f] && t < start[f] + len[f] {
                acc += pop[f];
                cdf.push(acc);
                ids.push(f);
            }
        }
        let uu = rng.uniform();
        let (fam, emb) = if ids.is_empty() || uu < 0.20 {
            (-1, unit_rand(&mut rng))
        } else {
            let tot = acc;
            let c: Vec<f64> = cdf.iter().map(|x| x / tot).collect();
            let f = ids[pick_cdf(&c, rng.uniform())];
            (f as i32, noisy(&cents[f], 0.55, &mut rng))
        };
        tasks.push(Task { fam, emb, u: rng.uniform(), ue: rng.uniform(), ua: rng.below(k) });
        junk.push(if rng.uniform() < 0.6 { Some(unit_rand(&mut rng)) } else { None });
    }
    World {
        name: "B-burst".to_string(),
        k,
        steps,
        eval_from: steps / 2,
        tasks,
        junk,
        p1: p0.clone(),
        p0,
        drift_at: usize::MAX,
        pnoise: 0.35,
        nfam,
    }
}

// ------------------------------------------------------------------------------------------------
// policies

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Full,
    FullRd,
    NoMem,
    CtxFree,
    Fifo,
    Lru,
    Value,
    Merge,
    Tiered,
    TieredMerge,
}

#[derive(Clone, Debug)]
pub struct PolicyCfg {
    pub name: String,
    pub kind: Kind,
    pub halflife: f64,
    pub init: f32,
    pub merge: f32,
    pub hot_frac: f64,
    pub cold: ColdKind,
    pub hnsw: bool,
    pub rd_h: f64,
}

impl PolicyCfg {
    pub fn new(name: &str, kind: Kind) -> Self {
        let hot_frac = if kind == Kind::TieredMerge { FROZEN_HOT_TM } else { FROZEN_HOT };
        Self { name: name.to_string(), kind, halflife: FROZEN_HL, init: FROZEN_INIT, merge: FROZEN_MERGE, hot_frac, cold: ColdKind::Int8, hnsw: false, rd_h: FROZEN_RD }
    }
}

// Hyperparameters frozen after the tuning stage (seeds 1-8 only); see results/h4_tune.md. Defaults here are the pre-tuning guesses.
pub const FROZEN_HL: f64 = 500.0;
pub const FROZEN_INIT: f32 = 0.6;
pub const FROZEN_MERGE: f32 = 0.8;
pub const FROZEN_HOT: f64 = 0.15;
pub const FROZEN_RD: f64 = 500.0; // recency half-life (steps) of the full+recency-weighted-vote reference (sensitivity baseline)
pub const FROZEN_HOT_TM: f64 = 0.6; // tiered+merge tuned separately (seeds 1-8)

pub fn hot_item_bytes() -> usize {
    D * 4 + 8 + 24 + ENTRY_OVERHEAD + 3
}

fn build(pc: &PolicyCfg, budget: Option<usize>) -> TieredMemory {
    let mut c = TieredConfig::new(D);
    c.short_cap = SHORT_CAP;
    c.value_halflife = pc.halflife;
    c.init_value = pc.init;
    if pc.hnsw {
        c.long = LongKind::Hnsw(neural_mesh::memory::HnswParams::default());
    }
    if let Some(b) = budget {
        c.forgetting = true;
        c.byte_budget = b;
        c.hard_budget = true;
        c.consolidate_every = 64;
        match pc.kind {
            Kind::Fifo => c.policy = ForgetPolicy::Fifo,
            Kind::Lru => c.policy = ForgetPolicy::Lru,
            Kind::Merge => c.merge_sim = Some(pc.merge),
            Kind::Tiered | Kind::TieredMerge => {
                if pc.kind == Kind::TieredMerge {
                    c.merge_sim = Some(pc.merge);
                }
                let hot_cap = ((pc.hot_frac * b as f64) / hot_item_bytes() as f64).max(16.0) as usize;
                c.compression = Some(Compression { bit_width: 4, hot_cap });
                c.cold_kind = pc.cold;
            }
            _ => {}
        }
    } else {
        c.consolidate_every = 0;
    }
    TieredMemory::new(c)
}

// ------------------------------------------------------------------------------------------------
// one run

#[derive(Clone, Debug, Default)]
pub struct RunOut {
    pub success: f64,
    pub useful: f64,
    pub bytes_final: usize,
    pub bytes_peak: usize,
    pub budget: usize,
    pub items: usize,
    pub cold: usize,
    pub forgotten: u64,
    pub merged: u64,
    pub ns_per_search: f64,
    pub overshoot: bool,
}

fn parse_meta(m: &str) -> Option<(usize, u8)> {
    let (a, r) = m.split_once('|')?;
    Some((a.parse().ok()?, r.parse().ok()?))
}

pub fn run(w: &World, pc: &PolicyCfg, budget: Option<usize>) -> RunOut {
    let k = w.k;
    let mut out = RunOut::default();
    let mut tm = match pc.kind {
        Kind::NoMem | Kind::CtxFree => None,
        _ => Some(build(pc, if matches!(pc.kind, Kind::Full | Kind::FullRd) { None } else { budget })),
    };
    out.budget = budget.unwrap_or(0);
    let mut fam_of = vec![-2i32; 2 * w.steps];
    let (mut succ, mut nev, mut useful, mut nrec) = (0.0, 0.0, 0.0, 0.0);
    let (mut sa, mut na) = (vec![0.0f64; k], vec![0.0f64; k]);
    let mut search_ns = 0u128;
    let mut nsearch = 0u64;
    for t in 0..w.steps {
        let task = &w.tasks[t];
        let mut neigh: Vec<(u64, usize, u8, f64)> = Vec::new(); // id, agent, r, kappa*weight
        let agent;
        match (&mut tm, pc.kind) {
            (None, Kind::NoMem) => agent = task.ua,
            (None, _) => {
                // context-free eps-greedy bandit
                let est: Vec<f64> = (0..k).map(|a| (sa[a] + PRIOR * PRIOR_M) / (na[a] + PRIOR_M)).collect();
                agent = if task.ue < EPS { task.ua } else { argmax(&est) };
            }
            (Some(m), _) => {
                let t0 = Instant::now();
                let res = m.search(&task.emb, KNN);
                search_ns += t0.elapsed().as_nanos();
                nsearch += 1;
                let mut num = vec![0f64; k];
                let mut den = vec![0f64; k];
                for (id, sim) in res {
                    if sim < S_MIN {
                        continue;
                    }
                    let Some((a, r)) = m.meta(id).and_then(parse_meta) else { continue };
                    let kap = ((sim - S_MIN) / (1.0 - S_MIN)) as f64;
                    let mut wt = m.item_stats(id).map_or(1.0, |s| s.weight as f64);
                    if pc.kind == Kind::FullRd {
                        wt *= 0.5f64.powf((t as f64 - id as f64) / pc.rd_h);
                    }
                    num[a] += kap * wt * r as f64;
                    den[a] += kap * wt;
                    neigh.push((id, a, r, kap));
                }
                let est: Vec<f64> = (0..k).map(|a| (num[a] + PRIOR * PRIOR_M) / (den[a] + PRIOR_M)).collect();
                agent = if task.ue < EPS { task.ua } else { argmax(&est) };
            }
        }
        let ok = task.u < w.p_of(task.fam, agent, t);
        sa[agent] += ok as u8 as f64;
        na[agent] += 1.0;
        if t >= w.eval_from {
            nev += 1.0;
            succ += ok as u8 as f64;
            if task.fam >= 0 {
                nrec += 1.0;
                let best = w.best_of(task.fam, t);
                if neigh.iter().any(|&(id, a, r, _)| fam_of[id as usize] == task.fam && a == best && r == 1) {
                    useful += 1.0;
                }
            }
        }
        if let Some(m) = tm.as_mut() {
            for &(id, a, r, _) in &neigh {
                if a == agent && r == 1 {
                    m.record_outcome(id, if ok { 1.0 } else { 0.0 });
                }
            }
            let meta = format!("{}|{}", agent, ok as u8);
            fam_of[t] = task.fam;
            m.add(t as u64, &task.emb, &meta);
            if let Some(j) = &w.junk[t] {
                fam_of[w.steps + t] = -3;
                m.add((w.steps + t) as u64, j, "t");
            }
            out.bytes_peak = out.bytes_peak.max(m.bytes());
        }
    }
    out.success = succ / nev;
    out.useful = if nrec > 0.0 { useful / nrec } else { f64::NAN };
    if let Some(m) = &tm {
        let st = m.stats();
        out.bytes_final = st.bytes;
        out.items = m.len();
        out.cold = st.cold;
        out.forgotten = st.forgotten;
        out.merged = st.merged;
        out.ns_per_search = if nsearch > 0 { search_ns as f64 / nsearch as f64 } else { 0.0 };
        out.overshoot = budget.is_some_and(|b| !matches!(pc.kind, Kind::Full | Kind::FullRd) && out.bytes_peak > b);
    }
    out
}

fn argmax(v: &[f64]) -> usize {
    let mut b = 0;
    for i in 1..v.len() {
        if v[i] > v[b] {
            b = i;
        }
    }
    b
}

// ------------------------------------------------------------------------------------------------
// statistics (paired, TOST, Holm)

pub use neural_mesh::stats::{holm_adjust, mean, sd, t_crit_95, t_two_sided_p};

/// paired-difference summary: (mean, lo95, hi95, p_tost_at_margin, p_noninferiority_at_margin)
pub fn paired(d: &[f64], margin: f64) -> (f64, f64, f64, f64, f64) {
    let n = d.len();
    let m = mean(d);
    if n < 2 {
        return (m, f64::NAN, f64::NAN, 1.0, 1.0);
    }
    let s = sd(d);
    let se = s / (n as f64).sqrt();
    let hw = t_crit_95(n - 1) * se;
    let one = |t: f64| {
        let p2 = t_two_sided_p(t.abs(), n - 1);
        if t >= 0.0 { p2 / 2.0 } else { 1.0 - p2 / 2.0 }
    };
    let (p, p_ni) = if se < 1e-15 {
        (if m.abs() < margin { 0.0 } else { 1.0 }, if m > -margin { 0.0 } else { 1.0 })
    } else {
        let ni = one((m + margin) / se); // H0: diff <= -margin (non-inferiority)
        (ni.max(one(-(m - margin) / se)), ni)
    };
    (m, m - hw, m + hw, p, p_ni)
}
