//! G2 TASK B: do the EXISTING compact state representations keep routing accuracy within noise,
//! and what do they cost/save in bytes and per-decision latency?
//!
//! Systems under test (nothing new is invented here; all code is a snapshot of f2/f3):
//!   family A (F2 quant, `MiniHeb`, inside Warm/MeanFeature)   vs the crate's f64 `Warm<HebbianRouter>`
//!       block-float16 (bf16-sr: shared f32 row scale + 16-bit codes), IEEE binary16 (f16+u16 baseline),
//!       and two more aggressive points (bf12-sr, bf8-sr+u8 baseline) reported for context;
//!   family B (F3 compact guarded brain, `DynRouter`)          vs the crate's f64 `recommended_router`
//!       dense f32, dense u16/q16, SPARSE u16/q16, SPARSE f16/q16, SPARSE f16/f16.
//!
//! PRE-DECLARED protocol (frozen before any number was looked at; also written to the json):
//!   metric        mean expected success over the run (`full`), paired by seed, higher is better
//!   margin        +-0.005 absolute success (the TOST equivalence margin)
//!   test          TOST (two one-sided t-tests, paired) per cell, Holm-adjusted over all cells of a split;
//!                 a cell is EQUIVALENT in a split iff Holm p < 0.05 AND the 95% t-CI of the paired
//!                 difference lies inside (-0.005, +0.005)   [95% CI is stricter than the usual 90% TOST CI]
//!   splits        tune 1-8 (reported only: n=8 cannot resolve +-0.005; nothing is tuned), held-out 31-60,
//!                 confirm 61-90.  CLAIM per (system, world) = equivalent in BOTH held-out AND confirm.
//!                 Pooled 31-90 (n=60) and, with --ext, fresh seeds 91-250 are extra evidence, not the rule.
//!   worlds        sim-fixed, sim-random, sim-no-traps, big-32agents, big-32agents-longtail (5, >= 4 required)
//!   extension     (declared AFTER seeing that n=30 per split cannot resolve +-0.005 in the 32-agent worlds; fresh seeds, no
//!                 selection involved): with --ext, seeds 91-250 (n=160) get the SAME cell rule (Holm TOST + 95% CI inside margin).
//!                 Reported as a separate column; it does not rewrite the primary verdict.
//!   latency       single thread, scalar CPU, ns/decision of select+update over a whole run (outcomes
//!                 pre-tabulated so the world generator is not timed), 15 interleaved repetitions, medians;
//!                 "NOT HURT" iff median decisions/sec >= 0.90 x the f64 reference in EVERY world; a ratio
//!                 below 0.90 in any world is reported as a regression.
//!   bytes         family A: packed state file (to_bytes, dense-or-sparse per row) vs crate JSON state and
//!                 vs packed f64; family B: resident heap of the whole router (counting allocator).
//!
//! Usage: cargo run --release --offline --bin g2_memory_tost [-- --out results/g2_memory_tost.json] [--ext] [--quick]

#[path = "g2_common/mod.rs"]
mod common;

use std::time::Instant;

use common::alloc_count;
use common::brain::{Brain, DynRouter, F16, Fx16, IdBuf, IdMode, Maker, Q16, MAXF, mk, mks};
use common::miniheb::{BKind, MiniHeb, Scheme, WKind, mean_feature_n};
use common::*;
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::defaults::recommended_router;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router};
use neural_mesh::stats::holm_adjust;
use serde_json::{json, Value};

#[global_allocator]
static GLOBAL: alloc_count::Counting = alloc_count::Counting;

// ---- pre-declared constants -------------------------------------------------------------------
const MARGIN5: f64 = 0.005;
const ALPHA: f64 = 0.05;
const LAT_NOT_HURT: f64 = 0.90;
const LAT_REPS: usize = 15;

// =============================================================================================
// systems

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fam {
    A,
    B,
}

type MakeFn = Box<dyn Fn(&Env) -> Box<dyn Router> + Send + Sync>;

struct Sys {
    name: String,
    fam: Fam,
    is_ref: bool,
    claim: bool, // part of the headline claim (the others are context points)
    make: MakeFn,
}

fn warm_mini(env: &Env, sch: &Scheme) -> Warm<MiniHeb> {
    let n = env.n_agents();
    Warm::new(MiniHeb::new(n, sch.clone()), env.kw_map(), n, 1.0).with_evidence(Evidence::Inner(mean_feature_n))
}
fn warm_crate(env: &Env) -> Warm<HebbianRouter> {
    let n = env.n_agents();
    Warm::new(HebbianRouter::new(n, heb_robust_cfg()), env.kw_map(), n, 1.0).with_evidence(Evidence::MeanFeature)
}
fn dyn_router(env: &Env, maker: Maker, name: &str) -> DynRouter {
    DynRouter::new(env.n_agents(), &env.kw_map(), IdMode::Table, maker, None, name)
}

fn schemes_a() -> Vec<(Scheme, bool)> {
    vec![
        (Scheme::new("bf16-sr (block-float16)", WKind::Bf { bits: 16, sr: true }, BKind::U16), true),
        (Scheme::new("f16+u16base (IEEE binary16)", WKind::F16, BKind::U16), true),
        (Scheme::new("bf12-sr", WKind::Bf { bits: 12, sr: true }, BKind::U16), false),
        (Scheme::new("bf8-sr+u8base", WKind::Bf { bits: 8, sr: true }, BKind::U8Sr), false),
    ]
}

fn makers_b() -> Vec<(&'static str, Maker, bool)> {
    vec![
        ("B dense f32/f32", mk::<f32, f32> as Maker, false),
        ("B dense u16/q16", mk::<Fx16, Q16<1024>> as Maker, false),
        ("B SPARSE u16/q16", mks::<Fx16, Q16<1024>> as Maker, true),
        ("B SPARSE f16/q16", mks::<F16, Q16<1024>> as Maker, true),
        ("B SPARSE f16/f16", mks::<F16, F16> as Maker, true),
    ]
}

fn build_systems() -> Vec<Sys> {
    let mut v: Vec<Sys> = Vec::new();
    v.push(Sys { name: "A ref: crate Warm<Hebbian> f64".into(), fam: Fam::A, is_ref: true, claim: false, make: Box::new(|e| Box::new(warm_crate(e))) });
    for (sch, claim) in schemes_a() {
        let nm = format!("A {}", sch.name);
        v.push(Sys { name: nm, fam: Fam::A, is_ref: false, claim, make: Box::new(move |e| Box::new(warm_mini(e, &sch))) });
    }
    v.push(Sys { name: "B ref: crate guarded(hebr) f64".into(), fam: Fam::B, is_ref: true, claim: false, make: Box::new(|e| Box::new(recommended_router(e.kw_map(), e.n_agents()))) });
    for (nm, mkr, claim) in makers_b() {
        let label = nm.to_string();
        let l2 = label.clone();
        v.push(Sys { name: label, fam: Fam::B, is_ref: false, claim, make: Box::new(move |e| Box::new(dyn_router(e, mkr, &l2))) });
    }
    v
}

// =============================================================================================
// accuracy

fn run_full(env: &Env, pol: &mut dyn Router, seed: u64) -> (f64, f64) {
    let n = env.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut cum = 0.0;
    let mut h500 = 0.0;
    for t in 0..env.len() {
        let task = env.task(t);
        let a = pol.select(task, &cands, &mut rng);
        cum += env.prob(t, a);
        pol.update(task, a, env.outcome(t, a));
        if t + 1 == 500 {
            h500 = cum / 500.0;
        }
    }
    (cum / env.len() as f64, h500)
}

fn cmp5(a: &[f64], b: &[f64]) -> Cmp {
    let mut c = compare(a, b);
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    c.p_equiv = tost_p(&d, MARGIN5);
    c
}
fn ci_in(c: &Cmp) -> bool {
    c.lo > -MARGIN5 && c.hi < MARGIN5
}

struct SplitDef {
    label: &'static str,
    lo: u64,
    hi: u64,
    in_rule: bool,
}

struct CellRes {
    sys: usize,
    world: usize,
    split: usize,
    cmp: Cmp,
    holm: f64,
    equiv: bool,
}

// =============================================================================================
// selftests

fn selftest() {
    for (seed, spec) in [(3u64, WorldSpec::Sim(neural_mesh::sim::Scenario::Fixed)), (4, WorldSpec::Big)] {
        let env = spec.make(seed);
        let n = env.n_agents();
        let cands: Vec<usize> = (0..n).collect();
        // A: MiniHeb(f64) == crate Hebbian inside Warm
        let mut a = warm_mini(&env, &Scheme::new("f64", WKind::F64, BKind::F64));
        let mut b = warm_crate(&env);
        let (mut r1, mut r2) = (Rng::new(7), Rng::new(7));
        for t in 0..env.len() {
            let task = env.task(t);
            let x = a.select(task, &cands, &mut r1);
            let y = b.select(task, &cands, &mut r2);
            assert_eq!(x, y, "MiniHeb(f64) != crate Warm<Hebbian> at {t}");
            let rw = env.outcome(t, x);
            a.update(task, x, rw);
            b.update(task, y, rw);
        }
        // B: compact f64/f64 == crate guarded
        let mut c = dyn_router(&env, mk::<f64, f64>, "c64");
        let mut d = recommended_router(env.kw_map(), n);
        let (mut r1, mut r2) = (Rng::new(9), Rng::new(9));
        for t in 0..env.len() {
            let task = env.task(t);
            let x = c.select(task, &cands, &mut r1);
            let y = d.select(task, &cands, &mut r2);
            assert_eq!(x, y, "compact f64 != crate guarded at {t}");
            let rw = env.outcome(t, x);
            c.update(task, x, rw);
            d.update(task, y, rw);
        }
        assert_eq!(r1.next_u64(), r2.next_u64());
    }
    // packed block-float16 state round-trips exactly
    let env = WorldSpec::Big.make(5);
    let sch = Scheme::new("bf16", WKind::Bf { bits: 16, sr: true }, BKind::U16);
    let mut w = warm_mini(&env, &sch);
    let cands: Vec<usize> = (0..env.n_agents()).collect();
    let mut rng = Rng::new(1);
    for t in 0..1500 {
        let task = env.task(t);
        let a = w.select(task, &cands, &mut rng);
        w.update(task, a, env.outcome(t, a));
    }
    let bytes = w.inner.to_bytes(true);
    let back = MiniHeb::from_bytes(&bytes, sch);
    assert_eq!(w.inner.max_cell_diff(&back), 0.0);
    println!("selftest ok: MiniHeb(f64) == crate Warm<Hebbian>; compact f64 == crate guarded; bf16 packed state round-trips exactly");
}

// =============================================================================================
// latency

trait Runner {
    fn step(&mut self, env: &Env, t: usize, cands: &[usize], rng: &mut Rng, tab: &[f64]) -> usize;
}
struct RRun(Box<dyn Router>);
impl Runner for RRun {
    #[inline]
    fn step(&mut self, env: &Env, t: usize, cands: &[usize], rng: &mut Rng, tab: &[f64]) -> usize {
        let task = env.task(t);
        let a = self.0.select(task, cands, rng);
        self.0.update(task, a, tab[t * cands.len() + a]);
        a
    }
}
struct IdsRun {
    b: Box<dyn Brain>,
    ids: Vec<IdBuf>,
}
impl Runner for IdsRun {
    #[inline]
    fn step(&mut self, _env: &Env, t: usize, cands: &[usize], rng: &mut Rng, tab: &[f64]) -> usize {
        let ids = self.ids[t].as_slice();
        let a = self.b.route(ids, cands, rng);
        self.b.feedback(ids, a, tab[t * cands.len() + a]);
        a
    }
}

type MkRunner = Box<dyn Fn(&Env) -> Box<dyn Runner>>;

fn ids_runner(env: &Env, maker: Maker) -> Box<dyn Runner> {
    let mut r = dyn_router(env, maker, "ids");
    let mut ids = Vec::with_capacity(env.len());
    for t in 0..env.len() {
        let mut b = IdBuf { a: [0; MAXF], len: 0 };
        for f in env.task(t).features.iter().take(MAXF) {
            b.a[b.len] = r.vocab.intern(f);
            b.len += 1;
        }
        ids.push(b);
    }
    let brain = r.b;
    Box::new(IdsRun { b: brain, ids })
}

struct LatSys {
    name: String,
    fam: Fam,
    is_ref: bool,
    mk: MkRunner,
}

fn lat_systems() -> Vec<LatSys> {
    let mut v: Vec<LatSys> = Vec::new();
    v.push(LatSys { name: "A ref: crate Warm<Hebbian> f64".into(), fam: Fam::A, is_ref: true, mk: Box::new(|e| Box::new(RRun(Box::new(warm_crate(e))))) });
    v.push(LatSys { name: "A MiniHeb f64 (layout only)".into(), fam: Fam::A, is_ref: false, mk: Box::new(|e| Box::new(RRun(Box::new(warm_mini(e, &Scheme::new("f64", WKind::F64, BKind::F64)))))) });
    for (sch, _) in schemes_a() {
        let nm = format!("A {}", sch.name);
        v.push(LatSys { name: nm, fam: Fam::A, is_ref: false, mk: Box::new(move |e| Box::new(RRun(Box::new(warm_mini(e, &sch))))) });
    }
    v.push(LatSys { name: "B ref: crate guarded(hebr) f64".into(), fam: Fam::B, is_ref: true, mk: Box::new(|e| Box::new(RRun(Box::new(recommended_router(e.kw_map(), e.n_agents()))))) });
    v.push(LatSys { name: "B compact f64/f64 string-API (layout only)".into(), fam: Fam::B, is_ref: false, mk: Box::new(|e| Box::new(RRun(Box::new(dyn_router(e, mk::<f64, f64>, "c64"))))) });
    v.push(LatSys { name: "B compact f64/f64 ids-API (layout only)".into(), fam: Fam::B, is_ref: false, mk: Box::new(|e| ids_runner(e, mk::<f64, f64>)) });
    for (nm, mkr, _) in makers_b() {
        let l = format!("{nm} string-API");
        v.push(LatSys { name: l, fam: Fam::B, is_ref: false, mk: Box::new(move |e| Box::new(RRun(Box::new(dyn_router(e, mkr, "x"))))) });
        let l = format!("{nm} ids-API");
        v.push(LatSys { name: l, fam: Fam::B, is_ref: false, mk: Box::new(move |e| ids_runner(e, mkr)) });
    }
    v
}

/// Index of the same-implementation, same-API system with f64 cells (decode-cost baseline), if any.
fn layout_base(lat: &[LatSys], i: usize) -> Option<usize> {
    let n = &lat[i].name;
    if lat[i].is_ref || n.contains("(layout only)") {
        return None;
    }
    let want = if lat[i].fam == Fam::A {
        "A MiniHeb f64 (layout only)"
    } else if n.contains("ids-API") {
        "B compact f64/f64 ids-API (layout only)"
    } else {
        "B compact f64/f64 string-API (layout only)"
    };
    lat.iter().position(|s| s.name == want)
}

fn outcome_table(env: &Env) -> Vec<f64> {
    let n = env.n_agents();
    let mut tab = vec![0.0; env.len() * n];
    for t in 0..env.len() {
        for a in 0..n {
            tab[t * n + a] = env.outcome(t, a);
        }
    }
    tab
}

/// ns/decision samples per system (interleaved repetitions, rotating order, 1 warm-up round).
fn time_world(systems: &[LatSys], env: &Env, tab: &[f64], seed: u64, reps: usize) -> Vec<Vec<f64>> {
    let n = env.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let len = env.len();
    let s = systems.len();
    let mut out = vec![Vec::new(); s];
    let mut sink = 0usize;
    for rep in 0..reps + 1 {
        for k in 0..s {
            let i = (k + rep) % s;
            let mut r = (systems[i].mk)(env);
            let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
            let t0 = Instant::now();
            for t in 0..len {
                sink = sink.wrapping_add(r.step(env, t, &cands, &mut rng, tab));
            }
            let dt = t0.elapsed().as_nanos() as f64 / len as f64;
            if rep > 0 {
                out[i].push(dt);
            }
        }
    }
    std::hint::black_box(sink);
    out
}

fn med(v: &[f64]) -> f64 {
    let mut w = v.to_vec();
    w.sort_by(|a, b| a.total_cmp(b));
    w[w.len() / 2]
}

// =============================================================================================
// bytes

fn live_bytes<T>(f: impl FnOnce() -> T) -> (T, i64) {
    let (r, s) = alloc_count::measure(f);
    (r, s.live)
}

fn drive(env: &Env, r: &mut dyn Router, seed: u64) {
    let n = env.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    for t in 0..env.len() {
        let task = env.task(t);
        let a = r.select(task, &cands, &mut rng);
        r.update(task, a, env.outcome(t, a));
    }
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned();
    let ext = args.iter().any(|a| a == "--ext");
    let quick = args.iter().any(|a| a == "--quick");
    let t0 = Instant::now();
    selftest();
    let worlds = all_worlds();
    let systems = build_systems();
    let nw = worlds.len();
    let ns = systems.len();
    let ref_a = systems.iter().position(|s| s.fam == Fam::A && s.is_ref).unwrap();
    let ref_b = systems.iter().position(|s| s.fam == Fam::B && s.is_ref).unwrap();
    println!("workers {} (G2_THREADS to change), systems {}, worlds {}", n_workers(), ns, nw);

    let mut splits = vec![
        SplitDef { label: "tune 1-8 (info only, n=8)", lo: 1, hi: 8, in_rule: false },
        SplitDef { label: "held-out 31-60", lo: 31, hi: 60, in_rule: true },
        SplitDef { label: "confirm 61-90", lo: 61, hi: 90, in_rule: true },
        SplitDef { label: "pooled 31-90 (n=60)", lo: 31, hi: 90, in_rule: false },
    ];
    if ext {
        splits.push(SplitDef { label: "power extension 91-250 (n=160)", lo: 91, hi: 250, in_rule: false });
    }
    let max_seed = if ext { 250 } else { 90 };
    let all_seeds: Vec<u64> = (1..=max_seed).filter(|s| *s <= 8 || *s >= 31).collect();

    // ---- accuracy: res[w][seed_idx][sys] = (full, h500)
    println!("\n--- accuracy runs ({} seeds x {} worlds x {} systems) ---", all_seeds.len(), nw, ns);
    let mut res: Vec<Vec<Vec<(f64, f64)>>> = Vec::new();
    for spec in &worlds {
        let r = par_map(&all_seeds, |&sd| {
            let env = spec.make(sd);
            systems.iter().map(|s| { let mut p = (s.make)(&env); run_full(&env, p.as_mut(), sd) }).collect::<Vec<_>>()
        });
        println!("  {} done ({:.0}s)", spec.label(), t0.elapsed().as_secs_f64());
        res.push(r);
    }
    let seed_idx = |sd: u64| all_seeds.iter().position(|&x| x == sd).unwrap();

    // ---- statistics
    let mut cells: Vec<Vec<CellRes>> = Vec::new(); // per split
    for (si, sp) in splits.iter().enumerate() {
        let sidx: Vec<usize> = all_seeds.iter().copied().filter(|&s| s >= sp.lo && s <= sp.hi).map(seed_idx).collect();
        let mut cs: Vec<CellRes> = Vec::new();
        for w in 0..nw {
            for s in 0..ns {
                if systems[s].is_ref {
                    continue;
                }
                let rf = if systems[s].fam == Fam::A { ref_a } else { ref_b };
                let a: Vec<f64> = sidx.iter().map(|&i| res[w][i][s].0).collect();
                let b: Vec<f64> = sidx.iter().map(|&i| res[w][i][rf].0).collect();
                cs.push(CellRes { sys: s, world: w, split: si, cmp: cmp5(&a, &b), holm: 1.0, equiv: false });
            }
        }
        let ps: Vec<f64> = cs.iter().map(|c| c.cmp.p_equiv).collect();
        let h = holm_adjust(&ps);
        for (c, hp) in cs.iter_mut().zip(h) {
            c.holm = hp;
            c.equiv = hp < ALPHA && ci_in(&c.cmp);
        }
        cells.push(cs);
    }
    for (si, sp) in splits.iter().enumerate() {
        println!("\n=== TOST vs f64 router, margin +-{MARGIN5}, split {} ===", sp.label);
        println!("{:<34} {:<24} {:>24} {:>7} {:>5}", "system", "world", "dQ(full) [95% CI]", "Holm", "EQUIV");
        for c in &cells[si] {
            println!(
                "{:<34} {:<24} {:>24} {:>7.3} {:>5}",
                systems[c.sys].name,
                worlds[c.world].label(),
                fmt_cmp(&c.cmp),
                c.holm,
                if c.equiv { "YES" } else { "no" }
            );
        }
    }

    // ---- verdicts
    let held = splits.iter().position(|s| s.label.starts_with("held")).unwrap();
    let conf = splits.iter().position(|s| s.label.starts_with("confirm")).unwrap();
    let pooled = splits.iter().position(|s| s.label.starts_with("pooled")).unwrap();
    let extsp: Option<&Vec<CellRes>> = splits.iter().position(|s| s.label.starts_with("power")).map(|i| &cells[i]);
    let find = |si: usize, s: usize, w: usize| cells[si].iter().find(|c| c.sys == s && c.world == w).unwrap();
    let mut verdicts: Vec<Value> = Vec::new();
    println!("\n=== VERDICT (rule: equivalent in held-out AND confirm, per world) ===");
    for s in 0..ns {
        if systems[s].is_ref {
            continue;
        }
        let mut per_world = Vec::new();
        let mut npass = 0;
        let mut worst_lo = f64::MAX;
        let mut worst_hi = f64::MIN;
        let mut pooled_pass = 0;
        let mut ext_pass = 0;
        let mut ext_worst = (f64::MAX, f64::MIN);
        for w in 0..nw {
            let h = find(held, s, w);
            let c = find(conf, s, w);
            let p = find(pooled, s, w);
            let pass = h.equiv && c.equiv;
            if pass {
                npass += 1;
            }
            if p.equiv {
                pooled_pass += 1;
            }
            let (ext_eq, ext_txt) = match extsp {
                Some(e) => {
                    let x = e.iter().find(|c| c.sys == s && c.world == w).unwrap();
                    if x.equiv {
                        ext_pass += 1;
                    }
                    ext_worst = (ext_worst.0.min(x.cmp.lo), ext_worst.1.max(x.cmp.hi));
                    (Some(x.equiv), fmt_cmp(&x.cmp))
                }
                None => (None, String::new()),
            };
            worst_lo = worst_lo.min(h.cmp.lo).min(c.cmp.lo);
            worst_hi = worst_hi.max(h.cmp.hi).max(c.cmp.hi);
            per_world.push(json!({"world": worlds[w].label(), "held_equiv": h.equiv, "confirm_equiv": c.equiv, "pass": pass, "pooled_equiv": p.equiv, "ext_equiv": ext_eq,
                "held": fmt_cmp(&h.cmp), "confirm": fmt_cmp(&c.cmp), "pooled": fmt_cmp(&p.cmp), "ext": ext_txt}));
            if !pass {
                println!("   (!) {} NOT certified in {}: held {} (holm {:.3}) | confirm {} (holm {:.3}) | pooled {} | ext {} -> {}", systems[s].name, worlds[w].label(), fmt_cmp(&h.cmp), h.holm, fmt_cmp(&c.cmp), c.holm, fmt_cmp(&p.cmp), ext_txt, match ext_eq { Some(true) => "ext EQUIV", Some(false) => "ext not equiv", None => "" });
            }
        }
        println!(
            "  {:<34} primary rule {}/{} worlds; pooled-60 {}/{}; {}worst CI over held+confirm [{:+.4},{:+.4}]{}",
            systems[s].name,
            npass,
            nw,
            pooled_pass,
            nw,
            if extsp.is_some() { format!("ext(160 fresh seeds) {}/{} (worst CI [{:+.4},{:+.4}]); ", ext_pass, nw, ext_worst.0, ext_worst.1) } else { String::new() },
            worst_lo,
            worst_hi,
            if systems[s].claim { "  <- headline candidate" } else { "" }
        );
        verdicts.push(json!({"system": systems[s].name, "headline_candidate": systems[s].claim, "worlds_certified": npass, "n_worlds": nw,
            "all_worlds_certified": npass == nw, "pooled_certified": pooled_pass, "ext_certified": if extsp.is_some() { json!(ext_pass) } else { Value::Null },
            "worst_ci_lo": worst_lo, "worst_ci_hi": worst_hi, "per_world": per_world}));
    }

    // ---- bytes (sequential, counting allocator)
    println!("\n=== BYTES (seed 31, end of run) ===");
    let mut bytes_rows: Vec<Value> = Vec::new();
    for spec in &worlds {
        let env = spec.make(31);
        // family A
        let (rw, live_ref) = live_bytes(|| { let mut w = warm_crate(&env); drive(&env, &mut w, 31); w });
        let v = serde_json::to_value(&rw.inner).unwrap();
        let json_state = serde_json::to_string(&v["weights"]).unwrap().len() + serde_json::to_string(&v["baselines"]).unwrap().len();
        let json_full = serde_json::to_string(&rw.inner).unwrap().len();
        let dense_est = neural_mesh::coupling::router_bytes(&rw.inner).dense_f64;
        drop(rw);
        let mut mf64 = warm_mini(&env, &Scheme::new("f64", WKind::F64, BKind::F64));
        drive(&env, &mut mf64, 31);
        let packed_f64 = mf64.inner.to_bytes(true).len();
        println!("{}: crate Warm<Hebbian> f64: json weights+baselines {} B (full serde {} B), live heap {} B, dense-f64 estimate {} B; packed f64 {} B", spec.label(), json_state, json_full, live_ref, dense_est, packed_f64);
        let mut row = json!({"world": spec.label(), "A_ref_json_weights_baselines": json_state, "A_ref_json_full_serde": json_full, "A_ref_live_heap": live_ref, "A_ref_dense_f64_estimate": dense_est, "A_packed_f64": packed_f64, "A": [], "B": []});
        for (sch, _) in schemes_a() {
            let (w, live) = live_bytes(|| { let mut w = warm_mini(&env, &sch); drive(&env, &mut w, 31); w });
            let packed = w.inner.to_bytes(true).len();
            println!("   A {:<30} packed {:>7} B ({:.2}x of crate JSON, {:.2}x of packed f64); in-memory live heap {:>7} B (not compact in memory: Vec<f64>/Vec<u16> per row)", sch.name, packed, packed as f64 / json_state as f64, packed as f64 / packed_f64 as f64, live);
            row["A"].as_array_mut().unwrap().push(json!({"scheme": sch.name, "packed_bytes": packed, "ratio_vs_crate_json": packed as f64 / json_state as f64, "ratio_vs_packed_f64": packed as f64 / packed_f64 as f64, "in_memory_live_heap": live}));
        }
        let (rg, live_g) = live_bytes(|| { let mut r = recommended_router(env.kw_map(), env.n_agents()); drive(&env, &mut r, 31); r });
        drop(rg);
        println!("   B ref crate guarded(hebr) f64: live heap {live_g} B");
        row["B_ref_live_heap"] = json!(live_g);
        for (nm, mkr, _) in [("B dense f64/f64 (layout only)", mk::<f64, f64> as Maker, false), ("B SPARSE f64/f64 (layout only)", mks::<f64, f64> as Maker, false)].into_iter().chain(makers_b()) {
            let (r, live) = live_bytes(|| { let mut r = dyn_router(&env, mkr, nm); drive(&env, &mut r, 31); r });
            let brain = r.b.heap_bytes();
            let used = r.b.used_bytes();
            let rows = r.b.rows();
            println!("   {:<22} live heap {:>7} B ({:.2}x of crate), brain-only heap {:>7} B (used {} B), rows {}", nm, live, live as f64 / live_g as f64, brain, used, rows);
            row["B"].as_array_mut().unwrap().push(json!({"scheme": nm, "live_heap": live, "ratio_vs_crate_live": live as f64 / live_g as f64, "brain_only_heap": brain, "brain_only_used": used, "rows": rows}));
        }
        bytes_rows.push(row);
    }

    // ---- state load / save cost (big world)
    println!("\n=== STATE SAVE/LOAD (big-32agents, seed 31) ===");
    let envb = WorldSpec::Big.make(31);
    let mut wc = warm_crate(&envb);
    drive(&envb, &mut wc, 31);
    let js = serde_json::to_string(&wc.inner).unwrap();
    let reps = 200;
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(serde_json::to_string(&wc.inner).unwrap());
    }
    let json_save_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    let t = Instant::now();
    for _ in 0..reps {
        std::hint::black_box(serde_json::from_str::<HebbianRouter>(&js).unwrap());
    }
    let json_load_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    println!("crate JSON ({} B): save {:.1} us, load {:.1} us", js.len(), json_save_us, json_load_us);
    let mut loadrows = vec![json!({"format": "crate serde JSON (HebbianRouter)", "bytes": js.len(), "save_us": json_save_us, "load_us": json_load_us})];
    for (sch, _) in std::iter::once((Scheme::new("f64 (packed)", WKind::F64, BKind::F64), true)).chain(schemes_a()) {
        let mut w = warm_mini(&envb, &sch);
        drive(&envb, &mut w, 31);
        let bytes = w.inner.to_bytes(true);
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(w.inner.to_bytes(true));
        }
        let save = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        let t = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(MiniHeb::from_bytes(&bytes, sch.clone()));
        }
        let load = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
        println!("packed {:<30} {:>7} B: save {:>7.1} us, load (decode) {:>7.1} us", sch.name, bytes.len(), save, load);
        loadrows.push(json!({"format": format!("packed {}", sch.name), "bytes": bytes.len(), "save_us": save, "load_us": load}));
    }

    // ---- latency
    println!("\n=== LATENCY: ns per decision (select+update), single thread, median of {} interleaved reps over 2 seeds ===", if quick { 5 } else { LAT_REPS });
    let lat = lat_systems();
    let reps = if quick { 5 } else { LAT_REPS };
    let mut lat_json: Vec<Value> = Vec::new();
    // ratio[sys] over worlds for the verdict
    let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); lat.len()];
    let mut dec_ratios: Vec<Vec<f64>> = vec![Vec::new(); lat.len()];
    for spec in &worlds {
        let mut samples: Vec<Vec<f64>> = vec![Vec::new(); lat.len()];
        for sd in [31u64, 61] {
            let env = spec.make(sd);
            let tab = outcome_table(&env);
            let o = time_world(&lat, &env, &tab, sd, reps);
            for (i, v) in o.into_iter().enumerate() {
                samples[i].extend(v);
            }
        }
        let meds: Vec<f64> = samples.iter().map(|v| med(v)).collect();
        let mins: Vec<f64> = samples.iter().map(|v| v.iter().cloned().fold(f64::INFINITY, f64::min)).collect();
        println!("\n{} ({} tasks, {} agents)", spec.label(), spec_len(spec), spec_agents(spec));
        println!("{:<48} {:>9} {:>9} {:>12} {:>9} {:>11}", "system", "ns med", "ns min", "decisions/s", "vs ref", "vs f64-cells");
        let ra = lat.iter().position(|s| s.fam == Fam::A && s.is_ref).unwrap();
        let rb = lat.iter().position(|s| s.fam == Fam::B && s.is_ref).unwrap();
        let mut rows = Vec::new();
        for (i, s) in lat.iter().enumerate() {
            let rf = if s.fam == Fam::A { ra } else { rb };
            let ratio = meds[rf] / meds[i]; // >1 faster than ref
            ratios[i].push(ratio);
            // decode-cost diagnostic: same implementation, same API, but f64 cells
            let base = layout_base(&lat, i);
            let dec = base.map(|b| meds[b] / meds[i]);
            if let Some(d) = dec {
                dec_ratios[i].push(d);
            }
            println!("{:<48} {:>9.0} {:>9.0} {:>12.0} {:>8.2}x {:>11}", s.name, meds[i], mins[i], 1e9 / meds[i], ratio, dec.map_or("-".to_string(), |d| format!("{d:.2}x")));
            rows.push(json!({"system": s.name, "ns_median": meds[i], "ns_min": mins[i], "decisions_per_s_median": 1e9 / meds[i], "speed_vs_ref": ratio, "speed_vs_same_layout_f64_cells": dec,
                "ns_spread_p10_p90": [pct(&samples[i], 0.1), pct(&samples[i], 0.9)]}));
        }
        lat_json.push(json!({"world": spec.label(), "rows": rows}));
    }
    println!("\n=== LATENCY VERDICT (NOT HURT iff speed-vs-ref >= {LAT_NOT_HURT} in every world) ===");
    let mut lat_verdict = Vec::new();
    for (i, s) in lat.iter().enumerate() {
        if s.is_ref {
            continue;
        }
        let worst = ratios[i].iter().cloned().fold(f64::INFINITY, f64::min);
        let best = ratios[i].iter().cloned().fold(0.0, f64::max);
        let ok = worst >= LAT_NOT_HURT;
        let (dworst, dbest) = if dec_ratios[i].is_empty() {
            (f64::NAN, f64::NAN)
        } else {
            (dec_ratios[i].iter().cloned().fold(f64::INFINITY, f64::min), dec_ratios[i].iter().cloned().fold(0.0, f64::max))
        };
        println!(
            "  {:<48} vs crate f64: worst {:.2}x best {:.2}x [{}]   vs same-layout f64 cells (decode cost): worst {:.2}x best {:.2}x{}",
            s.name,
            worst,
            best,
            if ok { "not hurt" } else { "REGRESSION in >=1 world" },
            dworst,
            dbest,
            if dworst.is_finite() && dworst < LAT_NOT_HURT { "  <- decode cost >10% slower in >=1 world" } else { "" }
        );
        lat_verdict.push(json!({"system": s.name, "worst_speed_vs_ref": worst, "best_speed_vs_ref": best, "not_hurt": ok, "per_world_speed_vs_ref": ratios[i],
            "decode_cost_worst_speed_vs_same_layout_f64": dworst, "decode_cost_best": dbest, "per_world_speed_vs_same_layout_f64": dec_ratios[i]}));
    }

    let doc = json!({
        "task": "G2-B: TOST equivalence of existing compact state representations vs the f64 routers; bytes; latency",
        "pre_declared": {
            "metric": "mean expected success over the run (full), paired by seed",
            "margin_abs_success": MARGIN5, "alpha": ALPHA, "ci": "95% paired t-interval",
            "equivalent_cell_rule": "Holm-adjusted TOST p < 0.05 (Holm over all cells of the split) AND 95% CI inside (-margin, +margin)",
            "claim_rule": "equivalent in BOTH held-out 31-60 AND confirm 61-90, per world; headline candidates: bf16-sr block-float16 and f16+u16base (family A); SPARSE u16/q16, SPARSE f16/q16, SPARSE f16/f16 (family B)",
            "extension_rule": "declared after seeing that n=30/split cannot resolve +-0.005 in the 32-agent worlds: with --ext, fresh seeds 91-250 (n=160) get the same cell rule; separate column, does not rewrite the primary verdict",
            "latency_extra_diagnostic": "decode cost = speed vs the SAME implementation/API with f64 cells (compact f64/f64, MiniHeb f64); added because the crate reference is a slower implementation",
            "tune_split": "1-8 reported for information only; nothing is tuned in this lab (schemes frozen from f2/f3)",
            "latency_not_hurt": format!("median decisions/s >= {LAT_NOT_HURT} x f64 reference in every world"),
            "references": {"A": "crate Warm<HebbianRouter> (heb_robust_cfg, MeanFeature), verified decision-identical to MiniHeb f64", "B": "crate defaults::recommended_router = Guarded<HebbianRouter>, verified decision-identical to compact f64/f64"},
            "worlds": worlds.iter().map(|w| w.label()).collect::<Vec<_>>(),
        },
        "splits": splits.iter().enumerate().map(|(si, sp)| json!({
            "split": sp.label, "seeds": format!("{}-{}", sp.lo, sp.hi), "in_claim_rule": sp.in_rule,
            "cells": cells[si].iter().map(|c| json!({
                "system": systems[c.sys].name, "family": format!("{:?}", systems[c.sys].fam), "world": worlds[c.world].label(),
                "n": c.cmp.n, "diff": c.cmp.mean, "ci_lo": c.cmp.lo, "ci_hi": c.cmp.hi, "wins": c.cmp.wins, "losses": c.cmp.losses,
                "p_tost": c.cmp.p_equiv, "holm_p_tost": c.holm, "ci_inside_margin": ci_in(&c.cmp), "equivalent": c.equiv,
            })).collect::<Vec<_>>()
        })).collect::<Vec<_>>(),
        "verdicts": verdicts,
        "bytes": bytes_rows,
        "state_save_load_big_world": loadrows,
        "latency": lat_json,
        "latency_verdict": lat_verdict,
        "mean_quality_f64_refs_held": worlds.iter().enumerate().map(|(w, spec)| {
            let sidx: Vec<usize> = all_seeds.iter().copied().filter(|&s| (31..=60).contains(&s)).map(seed_idx).collect();
            json!({"world": spec.label(), "A_ref": mean(&sidx.iter().map(|&i| res[w][i][ref_a].0).collect::<Vec<_>>()), "B_ref": mean(&sidx.iter().map(|&i| res[w][i][ref_b].0).collect::<Vec<_>>())})
        }).collect::<Vec<_>>(),
        "cores": n_threads(), "workers": n_workers(), "runtime_s": t0.elapsed().as_secs_f64(),
        "notes": [
            "family A in-memory state is NOT compact (Vec<f64>/Vec<u16> per row); the bytes saved are the packed state file; per-decision latency is for the in-memory quantised arithmetic",
            "family B bytes are resident heap incl. the router's private vocab + keyword map (as in f3); ids-API rows skip string hashing",
            "timings come from a shared 16 GB box with other workers running: use ratios from interleaved reps, not absolute ns",
        ],
    });
    if let Some(p) = out_path {
        std::fs::write(&p, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        println!("wrote {p}");
    }
    println!("runtime {:.1}s", t0.elapsed().as_secs_f64());
}

fn pct(v: &[f64], q: f64) -> f64 {
    let mut w = v.to_vec();
    w.sort_by(|a, b| a.total_cmp(b));
    w[((w.len() - 1) as f64 * q).round() as usize]
}
fn spec_len(s: &WorldSpec) -> usize {
    s.make(31).len()
}
fn spec_agents(s: &WorldSpec) -> usize {
    s.make(31).n_agents()
}
