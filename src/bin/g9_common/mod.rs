//! g9 shared pieces (W10 RESEARCH-ABILITY): stateless hashing, paired-test helpers, and the CRITIC WORLD
//! (generator with seeded real-error types + evidence-based / self-consistency / existing-gate critics).
//! Included from the g9_* bins with `#[path = "g9_common/mod.rs"] mod common;`.
//! Everything here is a seeded SIMULATION written by me; no real LLM behaviour is measured.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

use neural_mesh::research::*;
use neural_mesh::router::{Rng, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, PairedDiff};
use neural_mesh::variants::{HierSoConfig, SecondOpinionConfig};

// =============================================================================================
// stateless randomness (common random numbers across arms)

pub fn hash(seed: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ a.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ b.wrapping_mul(0x94D0_49BB_1331_11EB) ^ c.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
pub fn uni(seed: u64, a: u64, b: u64, c: u64) -> f64 {
    (hash(seed, a, b, c) >> 11) as f64 / (1u64 << 53) as f64
}
pub fn pick(seed: u64, a: u64, b: u64, c: u64, n: usize) -> usize {
    (hash(seed, a, b, c) % n.max(1) as u64) as usize
}

pub fn arg(args: &[String], key: &str, default: &str) -> String {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}
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

// =============================================================================================
// paired tests

pub fn verdict(adj: f64, mean: f64) -> &'static str {
    if adj < 0.05 && mean > 0.0 {
        "WIN"
    } else if adj < 0.05 {
        "LOSS"
    } else {
        "ns"
    }
}

#[derive(Clone)]
pub struct Test {
    pub family: String,
    pub world: String,
    pub metric: String,
    pub treat: String,
    pub vs: String,
    pub d: PairedDiff,
    pub p: f64,
    pub adj: f64,
}
pub fn mk_test(family: &str, world: &str, metric: &str, treat: &str, vs: &str, t: &[f64], b: &[f64]) -> Test {
    Test { family: family.into(), world: world.into(), metric: metric.into(), treat: treat.into(), vs: vs.into(), d: paired_diff(t, b), p: paired_t_pvalue(t, b), adj: 1.0 }
}
/// Holm within each family (in place)
pub fn holm_by_family(tests: &mut [Test]) {
    let mut fams: Vec<String> = tests.iter().map(|t| t.family.clone()).collect();
    fams.sort();
    fams.dedup();
    for f in fams {
        let idx: Vec<usize> = (0..tests.len()).filter(|&i| tests[i].family == f).collect();
        let adj = holm_adjust(&idx.iter().map(|&i| tests[i].p).collect::<Vec<_>>());
        for (k, &i) in idx.iter().enumerate() {
            tests[i].adj = adj[k];
        }
    }
}
pub fn test_json(t: &Test) -> serde_json::Value {
    serde_json::json!({"family": t.family, "world": t.world, "metric": t.metric, "treatment": t.treat, "vs": t.vs, "diff": t.d.diff, "wins": t.d.wins, "losses": t.d.losses, "p_raw": t.p, "holm_p": t.adj, "verdict": verdict(t.adj, t.d.diff.mean)})
}

// =============================================================================================
// CRITIC WORLD
//
// KB: 40 entities (8 families x 5 siblings) x 6 attributes; every fact has a current integer value and (with prob p_stale_fact) a
// stale previous value. A task asks for m claims (m = 1: fact QA, m = 4: a small report); a claim is a LOOKUP (value of e.a) or
// DERIVED (e.a1 +/- e.a2). The retrieval layer hands the critic a fixed, per-task pool of evidence snippets (independent of the
// arm). A generator agent errs with an agent x family specific rate; an error has one of four real types.

pub const NE: usize = 40;
pub const NFAM: usize = 8;
pub const NATT: usize = 6;
pub const NAG: usize = 5;
pub const ERR_NAMES: [&str; 4] = ["stale", "wrong-entity", "unsupported", "arithmetic"];

#[derive(Clone, Debug)]
pub struct CWorld {
    pub name: &'static str,
    pub family: &'static str,
    pub n_tasks: usize,
    pub m: usize,
    /// base error rate of a draft (per claim), scaled by agent x family multipliers
    pub eps: f64,
    /// error type mix (stale, wrong-entity, unsupported, arithmetic); arithmetic only arises on derived claims
    pub mix: [f64; 4],
    pub derived: f64,
    /// persistence of an error of each type across re-samples of the SAME agent (self-consistency can only see non-persistent ones)
    pub rho: [f64; 4],
    /// persistence of a correct latent answer
    pub rho_ok: f64,
    // retrieval / evidence
    pub p_ret: f64,
    pub p_b: f64,
    pub p_stale_fact: f64,
    pub p_stale_ret: f64,
    pub p_sib: f64,
    pub p_src_err: f64,
    /// prob that the evidence for an (e,a) is POISONED: only a confident, newest-dated WRONG snippet from a third source (+ the real doc w.p. p_ret)
    pub p_poison: f64,
    // second-agent checker model used by the EXISTING gate critic (research.rs); independent of any evidence
    pub chk_catch: [f64; 4],
    pub chk_fa: f64,
}

impl CWorld {
    pub fn base() -> CWorld {
        CWorld {
            name: "A1-mixed",
            family: "A",
            n_tasks: 1500,
            m: 1,
            eps: 0.20,
            mix: [0.30, 0.25, 0.25, 0.20],
            derived: 0.30,
            rho: [0.80, 0.70, 0.10, 0.15],
            rho_ok: 0.5,
            p_ret: 0.90,
            p_b: 0.50,
            p_stale_fact: 0.50,
            p_stale_ret: 0.60,
            p_sib: 0.40,
            p_src_err: 0.03,
            p_poison: 0.0,
            chk_catch: [0.25, 0.35, 0.60, 0.45],
            chk_fa: 0.05,
        }
    }
}

/// The FIXED world list (decided before any policy was tuned or run).
pub fn critic_worlds() -> Vec<CWorld> {
    let b = CWorld::base();
    vec![
        b.clone(),
        CWorld { name: "A2-stale-heavy", eps: 0.25, mix: [0.60, 0.15, 0.15, 0.10], derived: 0.2, ..b.clone() },
        CWorld { name: "A3-halluc-heavy", eps: 0.25, mix: [0.10, 0.10, 0.70, 0.10], ..b.clone() },
        CWorld { name: "A4-arith-heavy", eps: 0.25, mix: [0.10, 0.10, 0.10, 0.70], derived: 0.8, ..b.clone() },
        CWorld { name: "A5-persistent", rho: [0.95; 4], rho_ok: 0.95, ..b.clone() },
        CWorld { name: "A6-random-slips", rho: [0.0; 4], rho_ok: 0.0, ..b.clone() },
        CWorld { name: "A7-weak-retrieval", p_ret: 0.45, p_b: 0.20, p_stale_ret: 0.8, ..b.clone() },
        CWorld { name: "A8-poisoned-evidence", eps: 0.15, p_poison: 0.40, ..b.clone() },
        CWorld { name: "A9-rare-errors", eps: 0.03, ..b.clone() },
        CWorld { name: "A10-ctl-zero-errors", eps: 0.0, ..b.clone() },
        CWorld { name: "B1-report-mixed", family: "B", m: 4, eps: 0.12, ..b.clone() },
        CWorld { name: "B2-report-rare", family: "B", m: 4, eps: 0.03, ..b.clone() },
        CWorld { name: "B3-ctl-report-zero", family: "B", m: 4, eps: 0.0, ..b.clone() },
        // POST-HOC stress world (added after the first run showed that per-source voting survives A8): single source, half of the facts poisoned
        CWorld { name: "A11-single-source-poisoned", family: "A", eps: 0.15, p_ret: 0.60, p_b: 0.0, p_poison: 0.50, ..b.clone() },
    ]
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Look,
    Sum,
    Diff,
}
#[derive(Clone, Debug)]
pub struct ClaimSpec {
    pub e: usize,
    pub a: usize,
    pub a2: usize,
    pub op: Op,
}
#[derive(Clone, Debug)]
pub struct Snip {
    pub e: usize,
    pub a: usize,
    pub val: i64,
    pub date: i32,
    pub src: u8,
}
#[derive(Clone, Debug)]
pub struct CTask {
    pub task: Task,
    pub fam: usize,
    pub claims: Vec<ClaimSpec>,
    pub snips: Vec<Snip>,
    pub revealed: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Draft {
    pub agent: usize,
    pub vals: Vec<i64>,
    /// ground truth (hidden from critics): error type per claim
    pub err: Vec<Option<u8>>,
}
impl Draft {
    pub fn n_wrong(&self) -> usize {
        self.err.iter().filter(|e| e.is_some()).count()
    }
    pub fn any_wrong(&self) -> bool {
        self.err.iter().any(|e| e.is_some())
    }
    pub fn acc(&self) -> f64 {
        1.0 - self.n_wrong() as f64 / self.err.len() as f64
    }
}

pub struct CInst {
    pub w: CWorld,
    pub seed: u64,
    /// error-rate multiplier [agent][family]
    pub mult: Vec<Vec<f64>>,
    pub tasks: Vec<CTask>,
    pub kw: HashMap<String, usize>,
}

fn fam_of(e: usize) -> usize {
    e / (NE / NFAM)
}

impl CInst {
    fn ws(&self) -> u64 {
        self.seed.wrapping_mul(7919)
    }
    pub fn cur(&self, e: usize, a: usize) -> i64 {
        100 + (hash(self.ws(), 1, e as u64, a as u64) % 900) as i64
    }
    pub fn stale(&self, e: usize, a: usize) -> Option<i64> {
        if uni(self.ws(), 2, e as u64, a as u64) < self.w.p_stale_fact {
            let mut v = 100 + (hash(self.ws(), 3, e as u64, a as u64) % 900) as i64;
            if v == self.cur(e, a) {
                v += 7;
            }
            Some(v)
        } else {
            None
        }
    }
    fn sibling(&self, e: usize, k: u64) -> usize {
        let per = NE / NFAM;
        let f = e / per;
        f * per + (e % per + 1 + (k as usize % (per - 1))) % per
    }
    fn combine(op: Op, x: i64, y: i64) -> i64 {
        match op {
            Op::Look => x,
            Op::Sum => x + y,
            Op::Diff => (x - y).abs(),
        }
    }
    pub fn gold(&self, c: &ClaimSpec) -> i64 {
        Self::combine(c.op, self.cur(c.e, c.a), if c.op == Op::Look { 0 } else { self.cur(c.e, c.a2) })
    }

    pub fn eps_of(&self, agent: usize, fam: usize) -> f64 {
        (self.w.eps * self.mult[agent][fam]).clamp(0.0, 0.9)
    }

    /// one independent draw (error?, type, value) for claim `ci` of task `t` by `agent` with draw index `s`
    fn fresh(&self, t: &CTask, ci: usize, agent: usize, s: u64) -> (i64, Option<u8>) {
        let c = &t.claims[ci];
        let gold = self.gold(c);
        let key = t.task.id * 8 + ci as u64;
        let ak = agent as u64 * 64 + s;
        if uni(self.ws(), 10, key, ak) >= self.eps_of(agent, t.fam) {
            return (gold, None);
        }
        // type by mix (arithmetic only on derived claims; stale needs a stale version -> else unsupported)
        let mut w = self.w.mix;
        if c.op == Op::Look {
            w[3] = 0.0;
        }
        let tot: f64 = w.iter().sum();
        let mut x = uni(self.ws(), 11, key, ak) * tot;
        let mut ty = 0usize;
        for (i, wi) in w.iter().enumerate() {
            if x < *wi {
                ty = i;
                break;
            }
            x -= wi;
        }
        if ty == 0 && self.stale(c.e, c.a).is_none() {
            ty = 2;
        }
        let other = if c.op == Op::Look { 0 } else { self.cur(c.e, c.a2) };
        let v = match ty {
            0 => {
                let old = self.stale(c.e, c.a).unwrap();
                Self::combine(c.op, old, other)
            }
            1 => {
                let e2 = self.sibling(c.e, hash(self.ws(), 12, key, ak));
                let o2 = if c.op == Op::Look { 0 } else { self.cur(e2, c.a2) };
                Self::combine(c.op, self.cur(e2, c.a), o2)
            }
            2 => {
                let mut v = 100 + (hash(self.ws(), 13, key, ak) % if c.op == Op::Look { 900 } else { 1700 }) as i64;
                if v == gold {
                    v += 3;
                }
                v
            }
            _ => {
                const D: [i64; 8] = [1, -1, 2, -2, 10, -10, 100, -100];
                gold + D[pick(self.ws(), 14, key, ak, 8)]
            }
        };
        if v == gold {
            return (gold, None); // a "wrong-entity" value that coincides with gold is simply correct
        }
        (v, Some(ty as u8))
    }

    /// sample `s` of (task, agent): sample 0 is the primary (latent) draft; later samples copy the latent with a type-dependent persistence
    pub fn sample(&self, t: &CTask, agent: usize, s: u64) -> Draft {
        let mut vals = vec![];
        let mut err = vec![];
        for ci in 0..t.claims.len() {
            let lat = self.fresh(t, ci, agent, 0);
            let out = if s == 0 {
                lat
            } else {
                let pc = match lat.1 {
                    Some(ty) => self.w.rho[ty as usize],
                    None => self.w.rho_ok,
                };
                if uni(self.ws(), 15, t.task.id * 8 + ci as u64, agent as u64 * 64 + s) < pc {
                    lat
                } else {
                    self.fresh(t, ci, agent, s)
                }
            };
            vals.push(out.0);
            err.push(out.1);
        }
        Draft { agent, vals, err }
    }
}

pub fn build_critic(w: &CWorld, seed: u64, widx: u64) -> CInst {
    let s = seed.wrapping_mul(1000) + widx;
    let mut rng = Rng::new(s ^ 0xC17);
    // agent x family error multipliers: agent base in [.5,1.5] x family noise [.7,1.3]
    let base = [0.5, 0.8, 1.0, 1.2, 1.5];
    let mut mult = vec![vec![1.0; NFAM]; NAG];
    // shuffle which agent is which base (hidden)
    let mut order: Vec<usize> = (0..NAG).collect();
    for i in (1..NAG).rev() {
        order.swap(i, rng.below(i + 1));
    }
    for a in 0..NAG {
        for f in 0..NFAM {
            mult[a][f] = base[order[a]] * (0.7 + 0.6 * rng.f64());
        }
    }
    let mut inst = CInst { w: w.clone(), seed: s, mult, tasks: vec![], kw: HashMap::new() };
    // keyword map (best agent per family by multiplier, 30% decoys), as in the other research worlds
    for f in 0..NFAM {
        let best = (0..NAG).fold(0, |b, a| if inst.mult[a][f] < inst.mult[b][f] { a } else { b });
        let adv = if rng.f64() < 0.3 { rng.below(NAG) } else { best };
        inst.kw.insert(format!("fam{f}"), adv);
    }
    let per = NE / NFAM;
    let mut tasks = Vec::with_capacity(w.n_tasks);
    for i in 0..w.n_tasks {
        let fam = rng.below(NFAM);
        let mut claims = vec![];
        let mut e_used: Vec<usize> = vec![];
        for _ in 0..w.m {
            let e = fam * per + rng.below(per);
            let a = rng.below(NATT);
            let derived = rng.f64() < w.derived;
            let (a2, op) = if derived {
                let mut a2 = rng.below(NATT);
                if a2 == a {
                    a2 = (a + 1) % NATT;
                }
                (a2, if rng.bernoulli(0.5) { Op::Sum } else { Op::Diff })
            } else {
                (0, Op::Look)
            };
            e_used.push(e);
            claims.push(ClaimSpec { e, a, a2, op });
        }
        let features = vec![format!("fam{fam}"), format!("attr{}", claims[0].a), format!("n{}", rng.below(20))];
        let revealed = rng.f64() < 0.6;
        // ---- evidence pool
        let mut snips: Vec<Snip> = vec![];
        let mut need: Vec<(usize, usize)> = vec![];
        for c in &claims {
            need.push((c.e, c.a));
            if c.op != Op::Look {
                need.push((c.e, c.a2));
            }
        }
        need.sort();
        need.dedup();
        let tid = i as u64;
        for (k, &(e, a)) in need.iter().enumerate() {
            let kk = tid * 32 + k as u64;
            let ws = inst.ws();
            let cur = inst.cur(e, a);
            let srcv = |src: u64| -> i64 {
                if uni(ws, 20, kk, src) < w.p_src_err {
                    cur + 1 + (hash(ws, 21, kk, src) % 40) as i64
                } else {
                    cur
                }
            };
            let poisoned = uni(ws, 22, kk, 0) < w.p_poison;
            if poisoned {
                let bad = cur + 11 + (hash(ws, 23, kk, 0) % 80) as i64;
                snips.push(Snip { e, a, val: bad, date: 12, src: 2 });
            }
            if uni(ws, 24, kk, 0) < w.p_ret {
                snips.push(Snip { e, a, val: srcv(0), date: 10, src: 0 });
            }
            if uni(ws, 25, kk, 0) < w.p_b {
                snips.push(Snip { e, a, val: srcv(1), date: 10, src: 1 });
            }
            if let Some(old) = inst.stale(e, a) {
                if uni(ws, 26, kk, 0) < w.p_stale_ret {
                    snips.push(Snip { e, a, val: old, date: 5, src: 0 });
                }
            }
            if uni(ws, 27, kk, 0) < w.p_sib {
                let e2 = inst.sibling(e, hash(ws, 28, kk, 0));
                snips.push(Snip { e: e2, a, val: inst.cur(e2, a), date: 10, src: 0 });
            }
        }
        // shuffle snippet order (retrieval order carries no information)
        for j in (1..snips.len()).rev() {
            let r = pick(inst.ws(), 29, tid, j as u64, j + 1);
            snips.swap(j, r);
        }
        tasks.push(CTask { task: Task { id: tid, task_type: None, features }, fam, claims, snips, revealed });
    }
    inst.tasks = tasks;
    inst
}

// ---------------------------------------------------------------------------------------------
// evidence-based verification

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvKind {
    /// naive literal grounding: claimed value appears in ANY retrieved snippet (no entity check; derived values never appear)
    SupportAny,
    /// entity+attribute match, recency-blind (any matching snippet may support); derived = recompute from the first matching snippets
    SupportEntity,
    /// entity+attribute match, newest snippets win; derived = recompute
    Latest,
    /// per-source newest value, claim must be backed by >= theta of the sources that have evidence; derived = recompute from per-source votes
    Agree,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum V {
    Ok,
    Bad,
    Unv,
}

fn matching<'a>(t: &'a CTask, e: usize, a: usize) -> Vec<&'a Snip> {
    t.snips.iter().filter(|s| s.e == e && s.a == a).collect()
}
/// per-source newest value of (e,a): up to 3 sources
fn per_source(t: &CTask, e: usize, a: usize) -> Vec<(u8, i64, i32)> {
    let mut best: HashMap<u8, (i64, i32)> = HashMap::new();
    for s in matching(t, e, a) {
        let ent = best.entry(s.src).or_insert((s.val, s.date));
        if s.date > ent.1 {
            *ent = (s.val, s.date);
        }
    }
    let mut v: Vec<(u8, i64, i32)> = best.into_iter().map(|(k, (val, d))| (k, val, d)).collect();
    v.sort();
    v
}
/// the value a verifier takes for (e,a) under `kind` (None = no evidence)
fn resolve(kind: EvKind, t: &CTask, e: usize, a: usize) -> Option<i64> {
    let m = matching(t, e, a);
    if m.is_empty() {
        return None;
    }
    match kind {
        EvKind::SupportAny | EvKind::SupportEntity => Some(m[0].val),
        EvKind::Latest => {
            let d = m.iter().map(|s| s.date).max().unwrap();
            Some(m.iter().find(|s| s.date == d).unwrap().val)
        }
        EvKind::Agree => {
            let ps = per_source(t, e, a);
            // plurality of per-source newest values; tie -> the one with the newest date
            let mut cnt: Vec<(i64, usize, i32)> = vec![];
            for (_, v, d) in &ps {
                if let Some(c) = cnt.iter_mut().find(|c| c.0 == *v) {
                    c.1 += 1;
                    c.2 = c.2.max(*d);
                } else {
                    cnt.push((*v, 1, *d));
                }
            }
            cnt.sort_by(|x, y| y.1.cmp(&x.1).then(y.2.cmp(&x.2)));
            Some(cnt[0].0)
        }
    }
}

pub fn claim_verdict(kind: EvKind, theta: f64, t: &CTask, ci: usize, claimed: i64) -> V {
    let c = &t.claims[ci];
    match kind {
        EvKind::SupportAny => {
            if t.snips.iter().any(|s| s.val == claimed) {
                V::Ok
            } else {
                V::Bad
            }
        }
        _ => {
            if c.op == Op::Look {
                let m = matching(t, c.e, c.a);
                if m.is_empty() {
                    return V::Unv;
                }
                match kind {
                    EvKind::SupportEntity => {
                        if m.iter().any(|s| s.val == claimed) {
                            V::Ok
                        } else {
                            V::Bad
                        }
                    }
                    EvKind::Latest => {
                        let d = m.iter().map(|s| s.date).max().unwrap();
                        if m.iter().any(|s| s.date == d && s.val == claimed) {
                            V::Ok
                        } else {
                            V::Bad
                        }
                    }
                    _ => {
                        let ps = per_source(t, c.e, c.a);
                        let agree = ps.iter().filter(|p| p.1 == claimed).count() as f64 / ps.len() as f64;
                        if agree >= theta - 1e-9 {
                            V::Ok
                        } else {
                            V::Bad
                        }
                    }
                }
            } else {
                match (resolve(kind, t, c.e, c.a), resolve(kind, t, c.e, c.a2)) {
                    (Some(x), Some(y)) => {
                        if CInst::combine(c.op, x, y) == claimed {
                            V::Ok
                        } else {
                            V::Bad
                        }
                    }
                    _ => V::Unv,
                }
            }
        }
    }
}

/// task-level: (any Bad, any Unv)
pub fn task_verdict(kind: EvKind, theta: f64, t: &CTask, d: &Draft) -> (usize, usize) {
    let mut bad = 0;
    let mut unv = 0;
    for ci in 0..t.claims.len() {
        match claim_verdict(kind, theta, t, ci, d.vals[ci]) {
            V::Bad => bad += 1,
            V::Unv => unv += 1,
            V::Ok => {}
        }
    }
    (bad, unv)
}

// ---------------------------------------------------------------------------------------------
// arms

#[derive(Clone, Debug)]
pub enum ArmKind {
    None,
    /// research.rs critic through ResearchSession; weak = checker model of the world, strong = x1.6 catch / lower false alarm
    Exist { hier: bool, strong: bool, cost: f64 },
    /// research.rs gate, but the second "agent" runs the evidence verifier (Agree) as its check
    GateEv { cost: f64, theta: f64 },
    Ev { kind: EvKind, theta: f64, unv_flag: bool },
    /// self-consistency over k samples of the same agent; repair = per-claim plurality
    Sc { k: usize },
    /// Agree verdict; Unv -> self-consistency fallback k=3
    Cascade { theta: f64 },
    /// Agree + online precision gate on acting on flags (uses delayed graded outcomes)
    Auto { theta: f64, unv_flag: bool, cost: f64 },
    /// Agree, flag -> deliver the evidence-derived value (reader replaced by lookup); sensitivity, not a pure critic
    EvSub { theta: f64 },
    Oracle,
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub name: String,
    pub kind: ArmKind,
}
pub fn arm(name: &str, kind: ArmKind) -> Arm {
    Arm { name: name.to_string(), kind }
}

#[derive(Clone, Debug, Default)]
pub struct CRun {
    pub n: f64,
    pub acc_first: f64,
    pub acc_final: f64,
    /// agent calls beyond the first draft (extra samples, checker calls, retries)
    pub extra_calls: f64,
    /// verifier tool executions (evidence checks), each priced at C_TOOL agent calls
    pub tool_units: f64,
    pub tp: f64,
    pub fp: f64,
    pub fn_: f64,
    pub tn: f64,
    /// delivered answer is wrong AND carries no flag (silent error)
    pub silent: f64,
    pub delivered_wrong: f64,
    pub delivered_flagged: f64,
    /// retry replaced a correct first draft by a wrong one
    pub harmed: f64,
    pub rescued: f64,
    /// per error type: (errors, flagged)
    pub by_type: [(f64, f64); 4],
    pub realized_err: f64,
}

pub const C_TOOL: f64 = 0.1;
pub const CALL_COSTS: [f64; 3] = [0.02, 0.05, 0.10];

impl CRun {
    pub fn extra_total(&self, tool_cost: f64) -> f64 {
        (self.extra_calls + tool_cost * self.tool_units) / self.n
    }
    pub fn net(&self, c: f64, tool_cost: f64) -> f64 {
        self.acc_final / self.n - c * self.extra_total(tool_cost)
    }
    pub fn recall(&self) -> f64 {
        self.tp / (self.tp + self.fn_).max(1e-9)
    }
    pub fn precision(&self) -> f64 {
        self.tp / (self.tp + self.fp).max(1e-9)
    }
    pub fn false_flag(&self) -> f64 {
        self.fp / (self.fp + self.tn).max(1e-9)
    }
}

/// Executor for ResearchSession: draft = primary sample of (task, agent); check models the EXISTING second-agent critic
pub struct CExec<'a> {
    pub inst: &'a CInst,
    pub strong: bool,
    pub ev_check: Option<(f64, EvKind)>,
}
impl<'a> Executor for CExec<'a> {
    type Out = Draft;
    fn draft(&mut self, task: &Task, agent: usize, _ctx: &Context) -> Draft {
        self.inst.sample(&self.inst.tasks[task.id as usize], agent, 0)
    }
    fn tool_ok(&mut self, _out: &Draft) -> bool {
        true
    }
    fn check(&mut self, task: &Task, checker: usize, out: &Draft) -> bool {
        let t = &self.inst.tasks[task.id as usize];
        if let Some((theta, kind)) = self.ev_check {
            let (bad, _) = task_verdict(kind, theta, t, out);
            return bad > 0;
        }
        let w = &self.inst.w;
        let (cm, fa) = if self.strong { (1.6, 0.02) } else { (1.0, w.chk_fa) };
        let u = uni(self.inst.ws(), 30, task.id * 8 + checker as u64, out.agent as u64);
        if out.any_wrong() {
            // catch prob of the (first) error type, mildly checker-quality dependent
            let ty = out.err.iter().flatten().next().copied().unwrap() as usize;
            let q = 1.2 - 0.4 * self.inst.mult[checker][t.fam].min(1.5);
            let catch = (w.chk_catch[ty] * cm * q.clamp(0.6, 1.2)).min(0.95);
            u < catch
        } else {
            u < fa * t.claims.len() as f64
        }
    }
    fn grade(&mut self, task: &Task, out: &Draft) -> Option<f64> {
        let t = &self.inst.tasks[task.id as usize];
        if t.revealed {
            Some(out.acc())
        } else {
            None
        }
    }
}

struct Tracker {
    /// running mean of first-draft accuracy per (family, agent) from revealed grades (delay applied)
    sum: Vec<Vec<f64>>,
    cnt: Vec<Vec<f64>>,
    q: VecDeque<(u64, usize, usize, f64, bool, bool)>,
    // online calibration of flags (Auto): [flagged][wrong] counts
    fl_w: f64,
    fl_n: f64,
    wr: f64,
    nn: f64,
    /// graded accuracy of delivered retry drafts (delayed): (due, acc)
    rq: VecDeque<(u64, f64)>,
    ret_w: f64,
    ret_n: f64,
}
impl Tracker {
    fn new() -> Tracker {
        Tracker { sum: vec![vec![0.0; NAG]; NFAM], cnt: vec![vec![0.0; NAG]; NFAM], q: VecDeque::new(), fl_w: 0.0, fl_n: 0.0, wr: 0.0, nn: 0.0, rq: VecDeque::new(), ret_w: 0.0, ret_n: 0.0 }
    }
    fn flush(&mut self, tick: u64) {
        while self.rq.front().is_some_and(|e| e.0 <= tick) {
            let (_, acc) = self.rq.pop_front().unwrap();
            self.ret_n += 1.0;
            self.ret_w += 1.0 - acc;
        }
        while self.q.front().is_some_and(|e| e.0 <= tick) {
            let (_, fam, ag, acc, flagged, wrong) = self.q.pop_front().unwrap();
            self.sum[fam][ag] += acc;
            self.cnt[fam][ag] += 1.0;
            self.nn += 1.0;
            if wrong {
                self.wr += 1.0;
            }
            if flagged {
                self.fl_n += 1.0;
                if wrong {
                    self.fl_w += 1.0;
                }
            }
        }
    }
    fn best_other(&self, fam: usize, not: usize, salt: u64) -> usize {
        let mut best = usize::MAX;
        let mut bv = -1.0;
        for a in 0..NAG {
            if a == not {
                continue;
            }
            let m = (self.sum[fam][a] + 0.5 * 2.0) / (self.cnt[fam][a] + 2.0) + 1e-6 * ((hash(1, salt, a as u64, 0) % 1000) as f64);
            if m > bv {
                bv = m;
                best = a;
            }
        }
        best
    }
}

pub struct Delay {
    pub delay: usize,
}

fn plurality(vals: &[i64], primary: i64) -> (i64, bool) {
    // returns (value, strict majority exists)
    let mut best = (primary, 0usize);
    let mut cnt: Vec<(i64, usize)> = vec![];
    for &v in vals {
        if let Some(c) = cnt.iter_mut().find(|c| c.0 == v) {
            c.1 += 1;
        } else {
            cnt.push((v, 1));
        }
    }
    for (v, c) in &cnt {
        if *c > best.1 || (*c == best.1 && *v == primary) {
            best = (*v, *c);
        }
    }
    (best.0, best.1 * 2 > vals.len())
}

/// Run one (world instance, arm). Routing / retrieval-free ResearchSession with reuse (reuse feeds the delayed graded outcome of the FIRST draft to the router);
/// all arms see identical first drafts (same routes) so paired differences are tight.
pub fn run_critic_arm(inst: &CInst, arm: &Arm) -> CRun {
    let delay = 15usize;
    let mut cfg = ResearchConfig { reuse: true, delay, succ_thr: 0.99, ..ResearchConfig::default() };
    let mut ev_check = None;
    let mut strong = false;
    match &arm.kind {
        ArmKind::Exist { hier, strong: s, cost } => {
            strong = *s;
            let flat = SecondOpinionConfig { cost: *cost, kappa: 0.0, ..SecondOpinionConfig::default() };
            cfg.critic = Some(if *hier { CriticKind::Hier(HierSoConfig { base: flat, ..HierSoConfig::default() }) } else { CriticKind::Flat(flat) });
        }
        ArmKind::GateEv { cost, theta } => {
            ev_check = Some((*theta, EvKind::Agree));
            cfg.critic = Some(CriticKind::Flat(SecondOpinionConfig { cost: *cost, kappa: 0.0, ..SecondOpinionConfig::default() }));
        }
        _ => {}
    }
    let mut sess = ResearchSession::new(NAG, inst.kw.clone(), cfg);
    let mut ex = CExec { inst, strong, ev_check };
    let mut rng = Rng::new(inst.seed ^ 0x5EED);
    let cands: Vec<usize> = (0..NAG).collect();
    let mut tr = Tracker::new();
    let mut out = CRun::default();
    for (i, t) in inst.tasks.iter().enumerate() {
        let tick = (i + 1) as u64;
        tr.flush(tick);
        let rep = sess.run_task(&mut ex, &t.task, &cands, &mut rng);
        let agent = rep.agent;
        // first draft is always the primary sample of the routed agent (identical across arms)
        let first = inst.sample(t, agent, 0);
        let mut agent_extra = 0.0;
        let mut tool = 0.0;
        let mut flagged = false;
        let mut deliver_flag = false;
        let mut fin = first.clone();
        match &arm.kind {
            ArmKind::None => {}
            ArmKind::Exist { .. } | ArmKind::GateEv { .. } => {
                agent_extra = (rep.calls - 1) as f64;
                flagged = rep.flagged;
                if rep.flagged {
                    fin = inst.sample(t, rep.delivered_agent, 0);
                }
                // unchecked tasks: flagged=false even when wrong (a miss)
            }
            ArmKind::Ev { kind, theta, unv_flag } => {
                tool += 1.0;
                let (bad, unv) = task_verdict(*kind, *theta, t, &first);
                flagged = bad > 0 || (*unv_flag && unv > 0);
                if flagged {
                    let r = tr.best_other(t.fam, agent, tick);
                    agent_extra += 1.0;
                    tool += 1.0;
                    let second = inst.sample(t, r, 0);
                    let (b2, u2) = task_verdict(*kind, *theta, t, &second);
                    let fl2 = b2 > 0 || (*unv_flag && u2 > 0);
                    if !fl2 {
                        fin = second;
                    } else {
                        deliver_flag = true;
                        let (b1, _) = (bad, unv);
                        fin = if b2 <= b1 { second } else { first.clone() };
                    }
                }
            }
            ArmKind::Sc { k } => {
                let samples: Vec<Draft> = (1..*k as u64).map(|s| inst.sample(t, agent, s)).collect();
                agent_extra += (*k - 1) as f64;
                let mut vals = first.vals.clone();
                let mut errs = first.err.clone();
                let mut all_agree = true;
                let mut no_major = false;
                for ci in 0..t.claims.len() {
                    let mut vs = vec![first.vals[ci]];
                    vs.extend(samples.iter().map(|d| d.vals[ci]));
                    if vs.iter().any(|v| *v != vs[0]) {
                        all_agree = false;
                    }
                    let (pv, strict) = plurality(&vs, first.vals[ci]);
                    if !strict {
                        no_major = true;
                    }
                    vals[ci] = pv;
                    // error status of the plurality value: find a draft that produced it
                    let mut e = None;
                    let mut found = false;
                    for d in std::iter::once(&first).chain(samples.iter()) {
                        if d.vals[ci] == pv {
                            e = d.err[ci];
                            found = true;
                            break;
                        }
                    }
                    debug_assert!(found);
                    errs[ci] = e;
                }
                flagged = !all_agree;
                deliver_flag = no_major;
                fin = Draft { agent, vals, err: errs };
            }
            ArmKind::Cascade { theta } => {
                tool += 1.0;
                let (bad, unv) = task_verdict(EvKind::Agree, *theta, t, &first);
                if bad > 0 {
                    flagged = true;
                    let r = tr.best_other(t.fam, agent, tick);
                    agent_extra += 1.0;
                    tool += 1.0;
                    let second = inst.sample(t, r, 0);
                    let (b2, _) = task_verdict(EvKind::Agree, *theta, t, &second);
                    if b2 == 0 {
                        fin = second;
                    } else {
                        deliver_flag = true;
                        fin = if b2 <= bad { second } else { first.clone() };
                    }
                } else if unv > 0 {
                    // evidence inconclusive -> self-consistency k=3 on the same agent
                    let s1 = inst.sample(t, agent, 1);
                    let s2 = inst.sample(t, agent, 2);
                    agent_extra += 2.0;
                    let mut vals = first.vals.clone();
                    let mut errs = first.err.clone();
                    let mut all_agree = true;
                    let mut no_major = false;
                    for ci in 0..t.claims.len() {
                        let vs = [first.vals[ci], s1.vals[ci], s2.vals[ci]];
                        if vs.iter().any(|v| *v != vs[0]) {
                            all_agree = false;
                        }
                        let (pv, strict) = plurality(&vs, first.vals[ci]);
                        if !strict {
                            no_major = true;
                        }
                        vals[ci] = pv;
                        errs[ci] = [&first, &s1, &s2].iter().find(|d| d.vals[ci] == pv).map(|d| d.err[ci]).unwrap();
                    }
                    flagged = !all_agree;
                    deliver_flag = no_major;
                    fin = Draft { agent, vals, err: errs };
                }
            }
            ArmKind::Auto { theta, unv_flag, cost } => {
                tool += 1.0;
                let (bad, unv) = task_verdict(EvKind::Agree, *theta, t, &first);
                let fl = bad > 0 || (*unv_flag && unv > 0);
                flagged = fl;
                if fl {
                    // act iff the online estimate of P(first draft wrong | flagged) beats the break-even
                    let p_wf = (tr.fl_w + 1.0) / (tr.fl_n + 2.0);
                    let err_hat = (tr.wr + 1.0) / (tr.nn + 4.0);
                    // a reroute goes to the best OTHER agent, which is usually worse than the routed one: estimate its error online (prior: first-draft error + 0.05)
                    let ret_hat = (tr.ret_w + 8.0 * (err_hat + 0.05)) / (tr.ret_n + 8.0);
                    let be = ret_hat + cost * (1.0 + 2.0 * C_TOOL);
                    if p_wf > be {
                        let r = tr.best_other(t.fam, agent, tick);
                        agent_extra += 1.0;
                        tool += 1.0;
                        let second = inst.sample(t, r, 0);
                        if t.revealed {
                            tr.rq.push_back((tick + delay as u64, second.acc()));
                        }
                        let (b2, u2) = task_verdict(EvKind::Agree, *theta, t, &second);
                        let fl2 = b2 > 0 || (*unv_flag && u2 > 0);
                        if !fl2 {
                            fin = second;
                        } else {
                            deliver_flag = true;
                            fin = if b2 <= bad { second } else { first.clone() };
                        }
                    } else {
                        deliver_flag = true; // user is told "unverified"
                    }
                }
            }
            ArmKind::EvSub { theta } => {
                tool += 1.0;
                let (bad, _) = task_verdict(EvKind::Agree, *theta, t, &first);
                if bad > 0 {
                    flagged = true;
                    // deliver the evidence-derived value for each Bad claim (no further agent call)
                    let mut vals = first.vals.clone();
                    let mut errs = first.err.clone();
                    for ci in 0..t.claims.len() {
                        if claim_verdict(EvKind::Agree, *theta, t, ci, first.vals[ci]) == V::Bad {
                            let c = &t.claims[ci];
                            let v = if c.op == Op::Look {
                                resolve(EvKind::Agree, t, c.e, c.a)
                            } else {
                                match (resolve(EvKind::Agree, t, c.e, c.a), resolve(EvKind::Agree, t, c.e, c.a2)) {
                                    (Some(x), Some(y)) => Some(CInst::combine(c.op, x, y)),
                                    _ => None,
                                }
                            };
                            if let Some(v) = v {
                                vals[ci] = v;
                                errs[ci] = if v == inst.gold(c) { None } else { Some(2) };
                            }
                        }
                    }
                    fin = Draft { agent, vals, err: errs };
                }
            }
            ArmKind::Oracle => {
                tool += 1.0;
                if first.any_wrong() {
                    flagged = true;
                    let r = tr.best_other(t.fam, agent, tick);
                    agent_extra += 1.0;
                    tool += 1.0;
                    let second = inst.sample(t, r, 0);
                    deliver_flag = second.any_wrong();
                    fin = second;
                }
            }
        }
        // ---- bookkeeping
        let wrong = first.any_wrong();
        out.n += 1.0;
        out.acc_first += first.acc();
        out.acc_final += fin.acc();
        out.extra_calls += agent_extra;
        out.tool_units += tool;
        match (wrong, flagged) {
            (true, true) => out.tp += 1.0,
            (true, false) => out.fn_ += 1.0,
            (false, true) => out.fp += 1.0,
            (false, false) => out.tn += 1.0,
        }
        if wrong {
            out.realized_err += 1.0;
            let ty = first.err.iter().flatten().next().copied().unwrap() as usize;
            out.by_type[ty].0 += 1.0;
            if flagged {
                out.by_type[ty].1 += 1.0;
            }
        }
        let fin_wrong = fin.any_wrong();
        if fin_wrong {
            out.delivered_wrong += 1.0;
            if !(flagged && deliver_flag) {
                out.silent += 1.0;
            }
        }
        if flagged && deliver_flag {
            out.delivered_flagged += 1.0;
        }
        if !wrong && fin_wrong {
            out.harmed += 1.0;
        }
        if wrong && !fin_wrong {
            out.rescued += 1.0;
        }
        // delayed feedback for the retry-agent tracker and the Auto gate (first-draft grade, as the session sees it)
        if t.revealed {
            tr.q.push_back((tick + delay as u64, t.fam, agent, first.acc(), flagged, wrong));
        }
    }
    out
}
