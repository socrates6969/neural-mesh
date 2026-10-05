//! G11 (W12): is a LEARNED team model (additive / additive+pairwise, online, plain greedy search) a real, mergeable,
//! opt-in capability, or an artifact of W7's by-construction worlds? New independent worlds (g11_common/world.rs,
//! frozen before any policy ran), fair baselines {keyword, eps-greedy, pooled-thompson, hebbian-robust, warm(hebR),
//! guarded-safe} as sequential-select-with-team-credit, NeuralMesh team_size=K (guarded-safe and hebbian-robust leaders).
//!
//! Protocol: ALL tuning on seeds 1-8 (baselines: per world, per family, generous; learned arms: ONE pooled config over
//! all worlds including the controls, conservative). Frozen configs are evaluated on held-out 31-60 (`--set held`) and
//! fresh 61-90 (`--set confirm`); `--set dev` (101-130) was used for development. Primary metric: noise-free expected
//! team value v of the chosen teams (policies never see it). Paired 95% t-CI vs the best fair baseline per world (best
//! chosen on the EVALUATION seeds = conservative), Holm over all arm x world cells; controls: TOST +-0.01 (Holm).
//!
//! Usage: g11_team [--set held|confirm|dev] [--out results/g11_team_held.json] [--worlds RC,ROLE,...] [--notime]

#[path = "g11_common/mod.rs"]
mod common;

use common::base::*;
use common::learn::*;
use common::world::*;
use neural_mesh::router::Rng;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, t_two_sided_p};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::time::Instant;

// ------------------------------------------------------------------ allocation counter (state-size measurement)
struct Counting;
static LIVE: AtomicIsize = AtomicIsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        LIVE.fetch_add(l.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        LIVE.fetch_add(new as isize - l.size() as isize, Ordering::Relaxed);
        unsafe { System.realloc(p, l, new) }
    }
}
#[global_allocator]
static A: Counting = Counting;

#[derive(Clone, Debug, Default)]
struct Out {
    v: f64,
    late: f64,
    real: f64,
    gate_on: f64,
    lrn: f64,
}

fn run(w: &World, pol: &mut dyn TeamPol, seed: u64) -> Out {
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let t = w.cfg.t;
    let (mut sv, mut sl, mut sr) = (0.0, 0.0, 0.0);
    for i in 0..t {
        let task = &w.tasks[i];
        let team = pol.select(task, &mut rng);
        debug_assert_eq!(team.len(), w.cfg.k);
        let v = w.value(i, &team);
        let r = (v + noise_sd() * w.z[i]).clamp(0.0, 1.0);
        pol.observe(task, &team, r);
        sv += v;
        sr += r;
        if i >= t * 3 / 4 {
            sl += v;
        }
    }
    let (g, l) = pol.diag();
    Out { v: sv / t as f64, late: sl / (t - t * 3 / 4) as f64, real: sr / t as f64, gate_on: g, lrn: l }
}

// ------------------------------------------------------------------ specs
#[derive(Clone, Debug)]
enum Spec {
    Fair(Kind),
    Mesh(bool),
    LAdd(LCfg),
    LPair(LCfg),
    LGate(LCfg, GCfg),
    LFull(LCfg, GCfg, CCfg, Kind),
}
impl Spec {
    fn label(&self) -> String {
        match self {
            Spec::Fair(k) => format!("seq:{}", k.label()),
            Spec::Mesh(true) => "mesh(guarded-safe)".into(),
            Spec::Mesh(false) => "mesh(hebbian-robust)".into(),
            Spec::LAdd(..) => "L-add".into(),
            Spec::LPair(..) => "L-pair".into(),
            Spec::LGate(..) => "L-gate".into(),
            Spec::LFull(..) => "L-full".into(),
        }
    }
    fn is_learned(&self) -> bool {
        matches!(self, Spec::LAdd(..) | Spec::LPair(..) | Spec::LGate(..) | Spec::LFull(..))
    }
    fn build(&self, w: &World) -> Box<dyn TeamPol> {
        let (n, k) = (w.cfg.n, w.cfg.k);
        match self {
            Spec::Fair(kind) => Box::new(SeqTeam::new(kind, &w.kw, n, k)),
            Spec::Mesh(true) => Box::new(MeshTeam::guarded(&w.kw, n, k)),
            Spec::Mesh(false) => Box::new(MeshTeam::heb(n, k)),
            Spec::LAdd(c) => Box::new(Learner::new(c.clone(), GCfg::default(), Mode::Add, &w.kw, n, k)),
            Spec::LPair(c) => Box::new(Learner::new(c.clone(), GCfg::default(), Mode::Pair, &w.kw, n, k)),
            Spec::LGate(c, g) => Box::new(Learner::new(c.clone(), g.clone(), Mode::Gate, &w.kw, n, k)),
            Spec::LFull(c, g, cc, fb) => Box::new(Gated::new(Learner::new(c.clone(), g.clone(), Mode::Gate, &w.kw, n, k), SeqTeam::new(fb, &w.kw, n, k), cc.clone())),
        }
    }
}

fn eval(spec: &Spec, cfg: &Cfg, seeds: &[u64]) -> Vec<Out> {
    seeds
        .par_iter()
        .map(|&s| {
            let w = World::new(s, cfg.clone());
            let mut p = spec.build(&w);
            run(&w, p.as_mut(), s)
        })
        .collect()
}

fn seq_score(spec: &Spec, cfg: &Cfg, seeds: &[u64]) -> f64 {
    let mut s = 0.0;
    for &sd in seeds {
        let w = World::new(sd, cfg.clone());
        let mut p = spec.build(&w);
        s += run(&w, p.as_mut(), sd).v;
    }
    s / seeds.len() as f64
}

fn mean_of(v: &[Out], f: fn(&Out) -> f64) -> f64 {
    v.iter().map(f).sum::<f64>() / v.len() as f64
}
fn fv(o: &Out) -> f64 {
    o.v
}
fn fl(o: &Out) -> f64 {
    o.late
}
fn fr(o: &Out) -> f64 {
    o.real
}
fn col(v: &[Out], f: fn(&Out) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}

// ------------------------------------------------------------------ tuning (seeds 1-8 only)
fn tune_fair(cfg: &Cfg, tune: &[u64]) -> Vec<Spec> {
    let grid = fair_grid();
    let sc: Vec<f64> = grid.par_iter().map(|k| seq_score(&Spec::Fair(k.clone()), cfg, tune)).collect();
    let mut best: Vec<(String, usize)> = vec![];
    for (i, k) in grid.iter().enumerate() {
        let fam = k.family().to_string();
        match best.iter_mut().find(|(f, _)| *f == fam) {
            Some(e) => {
                if sc[i] > sc[e.1] {
                    e.1 = i;
                }
            }
            None => best.push((fam, i)),
        }
    }
    let mut v: Vec<Spec> = best.iter().map(|(_, i)| Spec::Fair(grid[*i].clone())).collect();
    v.push(Spec::Mesh(true));
    v.push(Spec::Mesh(false));
    v
}

fn lgrid(boosts: &[f64]) -> Vec<LCfg> {
    let mut v = vec![];
    for &lr in &[0.1, 0.3] {
        for &eps in &[0.02, 0.08] {
            for &c1 in &[0.0, 0.1] {
                for &c2 in &[0.0, 0.05] {
                    for &rho in &[0.0, 0.15, 0.3] {
                        for &boost in boosts {
                            v.push(LCfg { lr, l2: 0.0005, eps, c1, c2, rho, boost });
                        }
                    }
                }
            }
        }
    }
    v
}

fn pooled_score(spec: &Spec, cfgs: &[Cfg], tune: &[u64]) -> f64 {
    cfgs.iter().map(|c| seq_score(spec, c, tune)).sum::<f64>() / cfgs.len() as f64
}

fn pick<T: Clone + Sync>(items: &[T], f: impl Fn(&T) -> f64 + Sync) -> (T, f64) {
    let sc: Vec<f64> = items.par_iter().map(|c| f(c)).collect();
    let mut bi = 0;
    for i in 1..sc.len() {
        if sc[i] > sc[bi] {
            bi = i;
        }
    }
    (items[bi].clone(), sc[bi])
}

struct Tuned {
    add: LCfg,
    pair: LCfg,
    gcfg: LCfg,
    gate: GCfg,
    cc: CCfg,
    fb: Kind,
}

fn tune_learned(cfgs: &[Cfg], tune: &[u64], boosts: &[f64]) -> Tuned {
    let grid = lgrid(boosts);
    let addg: Vec<LCfg> = grid.iter().filter(|c| c.c2 == 0.0).cloned().collect();
    let (add, s1) = pick(&addg, |c| pooled_score(&Spec::LAdd(c.clone()), cfgs, tune));
    println!("[tune] L-add  pooled mean v {s1:.4}: {add:?}");
    let (pair, s2) = pick(&grid, |c| pooled_score(&Spec::LPair(c.clone()), cfgs, tune));
    println!("[tune] L-pair pooled mean v {s2:.4}: {pair:?}");
    let g0 = GCfg { zthr: 2.5, lam: 0.995 };
    let (gl, _) = pick(&grid, |c| pooled_score(&Spec::LGate(c.clone(), g0.clone()), cfgs, tune));
    let gg: Vec<GCfg> = [1.5, 2.5, 4.0].iter().map(|&z| GCfg { zthr: z, lam: 0.995 }).collect();
    let (gate, s3) = pick(&gg, |g| pooled_score(&Spec::LGate(gl.clone(), g.clone()), cfgs, tune));
    println!("[tune] L-gate pooled mean v {s3:.4}: {gl:?} {gate:?}");
    let mut ccs = vec![];
    for &q in &[0.08, 0.2] {
        for &margin in &[0.0, 0.01] {
            ccs.push(CCfg { q, margin, warm: 60, decay: 0.998 });
        }
    }
    let (cc, s4) = pick(&ccs, |c| pooled_score(&Spec::LFull(gl.clone(), gate.clone(), c.clone(), Kind::GuardedSafe), cfgs, tune));
    println!("[tune] L-full pooled mean v {s4:.4}: {cc:?} (fallback seq:guarded-safe)");
    Tuned { add, pair, gcfg: gl, gate, cc, fb: Kind::GuardedSafe }
}

// ------------------------------------------------------------------ stats helpers
fn upper_p(t: f64, df: usize) -> f64 {
    let p2 = t_two_sided_p(t.abs(), df);
    if t >= 0.0 { p2 / 2.0 } else { 1.0 - p2 / 2.0 }
}
/// TOST p-value for H0: |mean diff| >= delta, paired differences d
fn tost_p(d: &[f64], delta: f64) -> f64 {
    let n = d.len();
    let m = neural_mesh::stats::mean(d);
    let s = neural_mesh::stats::sd(d);
    let se = s / (n as f64).sqrt();
    if se < 1e-15 {
        return if m.abs() < delta { 0.0 } else { 1.0 };
    }
    let p_lo = upper_p((m + delta) / se, n - 1);
    let p_hi = upper_p(-(m - delta) / se, n - 1);
    p_lo.max(p_hi)
}

struct Cell {
    set: u8,
    world: String,
    generator: String,
    own_class: bool,
    control: bool,
    arm: String,
    diff: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    losses: usize,
    p: f64,
    late_diff: f64,
    p_tost: f64,
    p_ni: f64,
    p_ni_gs: f64,
    ci90_in: bool,
    vs_mesh: f64,
    vs_mesh_p: f64,
    vs_gs: f64,
}

// ------------------------------------------------------------------ timing / bytes (single thread)
fn timed(spec: &Spec, w: &World, seed: u64) -> (f64, f64) {
    // returns (ns/decision, live heap bytes of the policy after the run)
    let before = LIVE.load(Ordering::Relaxed);
    let mut p = spec.build(w);
    let mut best = f64::INFINITY;
    for rep in 0..3 {
        if rep > 0 {
            p = spec.build(w);
        }
        let t0 = Instant::now();
        let _ = run(w, p.as_mut(), seed);
        best = best.min(t0.elapsed().as_nanos() as f64 / w.cfg.t as f64);
    }
    let after = LIVE.load(Ordering::Relaxed);
    drop(p);
    (best, (after - before) as f64)
}

struct NullPol {
    k: usize,
}
impl TeamPol for NullPol {
    fn name(&self) -> String {
        "null".into()
    }
    fn select(&mut self, _t: &neural_mesh::router::Task, _r: &mut Rng) -> Vec<usize> {
        (0..self.k).collect()
    }
    fn observe(&mut self, _t: &neural_mesh::router::Task, _team: &[usize], _r: f64) {}
}
fn null_ns(w: &World, seed: u64) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let mut p = NullPol { k: w.cfg.k };
        let t0 = Instant::now();
        let _ = run(w, &mut p, seed);
        best = best.min(t0.elapsed().as_nanos() as f64 / w.cfg.t as f64);
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let set = arg("--set").unwrap_or("held".into());
    let out = arg("--out");
    let only: Option<Vec<String>> = arg("--worlds").map(|s| s.split(',').map(|x| x.to_string()).collect());
    let boosts: Vec<f64> = arg("--boost").unwrap_or("0,4".into()).split(',').map(|x| x.parse().unwrap()).collect();
    if let Some(x) = arg("--noise") {
        set_noise(x.parse().unwrap());
    }
    let notime = args.iter().any(|a| a == "--notime");
    let tune: Vec<u64> = (1..=8).collect();
    let seeds: Vec<u64> = match set.as_str() {
        "held" => (31..=60).collect(),
        "confirm" => (61..=90).collect(),
        "dev" => (101..=130).collect(),
        _ => panic!("--set held|confirm|dev"),
    };
    let all_worlds = worlds();
    let sel: Vec<Cfg> = all_worlds.iter().filter(|c| only.as_ref().map(|o| o.iter().any(|x| x == c.name)).unwrap_or(true)).cloned().collect();
    println!("G11 learned team model vs fair baselines. eval set {set} ({} seeds); tuning only on seeds 1-8; learned configs pooled over ALL {} worlds (incl. controls)", seeds.len(), all_worlds.len());
    let per_world = arg("--tune").map(|x| x == "perworld").unwrap_or(false);
    let tu_pooled = if per_world { None } else { Some(tune_learned(&all_worlds, &tune, &boosts)) };
    let tu0 = tu_pooled.as_ref();
    let mut result = serde_json::Map::new();
    result.insert("set".into(), json!(set));
    result.insert("tune_mode".into(), json!(if per_world { "perworld" } else { "pooled" }));
    if let Some(tu) = tu0 {
        result.insert("tuned".into(), json!({"add": format!("{:?}", tu.add), "pair": format!("{:?}", tu.pair), "gate_learner": format!("{:?}", tu.gcfg), "gate": format!("{:?}", tu.gate), "cc": format!("{:?}", tu.cc)}));
    }
    let mut cells: Vec<Cell> = vec![];
    let mut wj = serde_json::Map::new();
    let mut armcmp: Vec<(String, u8, String, f64, f64, f64, f64)> = vec![];
    for cfg in &sel {
        let fair = tune_fair(cfg, &tune);
        let tu_w;
        let tu: &Tuned = match tu0 {
            Some(t) => t,
            None => {
                tu_w = tune_learned(std::slice::from_ref(cfg), &tune, &boosts);
                &tu_w
            }
        };
        let learned = vec![
            Spec::LAdd(tu.add.clone()),
            Spec::LPair(tu.pair.clone()),
            Spec::LGate(tu.gcfg.clone(), tu.gate.clone()),
            Spec::LFull(tu.gcfg.clone(), tu.gate.clone(), tu.cc.clone(), tu.fb.clone()),
        ];
        let (orc, rnd) = World::new(seeds[0], cfg.clone()).references(seeds[0]);
        println!("\n=== {} [gen {}{}{}] (N={}, K={}, T={}) refs[seed {}]: oracle v {:.3}, random-team v {:.3}", cfg.name, cfg.family, if cfg.own_class { ", OWN-CLASS by construction" } else { "" }, if cfg.control { ", CONTROL" } else { "" }, cfg.n, cfg.k, cfg.t, seeds[0], orc, rnd);
        let fres: Vec<(Spec, Vec<Out>)> = fair.iter().map(|s| (s.clone(), eval(s, cfg, &seeds))).collect();
        let lres: Vec<(Spec, Vec<Out>)> = learned.iter().map(|s| (s.clone(), eval(s, cfg, &seeds))).collect();
        let bf = (0..fres.len()).max_by(|&a, &b| mean_of(&fres[a].1, fv).total_cmp(&mean_of(&fres[b].1, fv))).unwrap();
        let mesh_i = fres.iter().position(|(s, _)| s.label() == "mesh(guarded-safe)").unwrap();
        let gs_i = fres.iter().position(|(s, _)| s.label() == "seq:guarded-safe").unwrap();
        let w0 = World::new(seeds[0], cfg.clone());
        let nn = if notime { 0.0 } else { null_ns(&w0, seeds[0]) };
        println!("{:<26} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9} {:>9}", "policy", "v", "late v", "real r", "gate-on", "lrn-frac", "ns/dec", "bytes");
        let mut rows = vec![];
        for (s, r) in fres.iter().chain(lres.iter()) {
            let (ns, by) = if notime { (0.0, 0.0) } else { timed(s, &w0, seeds[0]) };
            let ns_pol = (ns - nn).max(0.0);
            println!(
                "{:<26} {:>7.4} {:>7.4} {:>7.4} {:>7.2} {:>7.2} {:>9.0} {:>9.0}{}",
                s.label(),
                mean_of(r, fv),
                mean_of(r, fl),
                mean_of(r, fr),
                mean_of(r, |o| o.gate_on),
                mean_of(r, |o| o.lrn),
                ns_pol,
                by,
                if s.label() == fres[bf].0.label() { "   <- best fair" } else { "" }
            );
            rows.push(json!({"policy": s.label(), "spec": format!("{:?}", s), "v": mean_of(r, fv), "late_v": mean_of(r, fl), "real": mean_of(r, fr), "gate_on": mean_of(r, |o| o.gate_on), "lrn_frac": mean_of(r, |o| o.lrn), "ns_per_decision_policy_only": ns_pol, "ns_per_decision_total": ns, "heap_bytes": by}));
        }
        for (s, r) in &lres {
            let b = &fres[bf].1;
            let pd = paired_diff(&col(r, fv), &col(b, fv));
            let p = paired_t_pvalue(&col(r, fv), &col(b, fv));
            let bl = (0..fres.len()).max_by(|&a, &b| mean_of(&fres[a].1, fl).total_cmp(&mean_of(&fres[b].1, fl))).unwrap();
            let ld = paired_diff(&col(r, fl), &col(&fres[bl].1, fl));
            let d: Vec<f64> = col(r, fv).iter().zip(col(b, fv)).map(|(x, y)| x - y).collect();
            let pt = tost_p(&d, 0.01);
            let pni = ni_p(&d, 0.01);
            let nn_ = d.len() as f64;
            let sd = neural_mesh::stats::sd(&d);
            let se = sd / nn_.sqrt();
            let t90 = t90(d.len() - 1);
            let m = neural_mesh::stats::mean(&d);
            let ci90 = m - t90 * se > -0.01 && m + t90 * se < 0.01;
            let pm = paired_diff(&col(r, fv), &col(&fres[mesh_i].1, fv));
            let pmp = paired_t_pvalue(&col(r, fv), &col(&fres[mesh_i].1, fv));
            let pg = paired_diff(&col(r, fv), &col(&fres[gs_i].1, fv));
            let dg: Vec<f64> = col(r, fv).iter().zip(col(&fres[gs_i].1, fv)).map(|(x, y)| x - y).collect();
            let pni_gs = ni_p(&dg, 0.01);
            cells.push(Cell { set: cfg.set, world: cfg.name.into(), generator: cfg.family.into(), own_class: cfg.own_class, control: cfg.control, arm: s.label(), diff: pd.diff.mean, lo: pd.diff.lo, hi: pd.diff.hi, wins: pd.wins, losses: pd.losses, p, late_diff: ld.diff.mean, p_tost: pt, p_ni: pni, p_ni_gs: pni_gs, ci90_in: ci90, vs_mesh: pm.diff.mean, vs_mesh_p: pmp, vs_gs: pg.diff.mean });
        }
        for (s, r) in lres.iter().skip(1) {
            let pd = paired_diff(&col(r, fv), &col(&lres[0].1, fv));
            let p = paired_t_pvalue(&col(r, fv), &col(&lres[0].1, fv));
            armcmp.push((cfg.name.to_string(), cfg.set, s.label(), pd.diff.mean, pd.diff.lo, pd.diff.hi, p));
        }
        wj.insert(cfg.name.into(), json!({"generator": cfg.family, "own_class": cfg.own_class, "control": cfg.control, "oracle_v": orc, "random_v": rnd, "best_fair": fres[bf].0.label(), "rows": rows}));
    }
    result.insert("worlds".into(), Value::Object(wj));

    // ---------------------------------------------------------------- Holm + verdicts
    let ps: Vec<f64> = cells.iter().map(|c| c.p).collect();
    let adj = holm_by_set(&cells, &ps);
    let mps: Vec<f64> = cells.iter().map(|c| c.vs_mesh_p).collect();
    let madj = holm_by_set(&cells, &mps);
    let ctrl_idx: Vec<usize> = (0..cells.len()).filter(|&i| cells[i].control).collect();
    let tp: Vec<f64> = ctrl_idx.iter().map(|&i| cells[i].p_tost).collect();
    let tadj = holm_adjust(&tp);
    let nip: Vec<f64> = ctrl_idx.iter().map(|&i| cells[i].p_ni).collect();
    let niadj = holm_adjust(&nip);
    let nipg: Vec<f64> = ctrl_idx.iter().map(|&i| cells[i].p_ni_gs).collect();
    let niadj_gs = holm_adjust(&nipg);
    println!("\n=== learned arms vs BEST FAIR baseline per world (metric v; paired 95% t-CI; Holm within each world set; {} cells total)", cells.len());
    println!("{:<10} {:<8} {:>9} {:>20} {:>5}/{:<5} {:>7} {:>6} | {:>8} | vs mesh(gs) {:>8} (Holm) | vs seq:gs {:>8}", "world", "arm", "diff", "95% CI", "win", "loss", "p", "Holm", "late", "diff", "diff");
    let mut cj = vec![];
    for (i, c) in cells.iter().enumerate() {
        let verdict = if adj[i] < 0.05 && c.lo > 0.0 { "WIN" } else if adj[i] < 0.05 && c.hi < 0.0 { "LOSS" } else { "n.s." };
        let mverdict = if madj[i] < 0.05 && c.vs_mesh > 0.0 { "WIN" } else if madj[i] < 0.05 && c.vs_mesh < 0.0 { "LOSS" } else { "n.s." };
        let tpos = ctrl_idx.iter().position(|&x| x == i);
        let equiv = tpos.map(|k| tadj[k] < 0.05);
        println!(
            "{:<10} {:<8} {:>+9.4} [{:>+8.4},{:>+8.4}] {:>5}/{:<5} {:>7.4} {:>6.4} {:<4} | {:>+8.4} | {:>+8.4} ({:.4}) {:<4} | {:>+8.4}{}",
            c.world, c.arm, c.diff, c.lo, c.hi, c.wins, c.losses, c.p, adj[i], verdict, c.late_diff, c.vs_mesh, madj[i], mverdict, c.vs_gs,
            match equiv { Some(true) => format!("   TOST+-0.01: EQUIVALENT; NI-vs-bestfair Holm {:.3}; NI-vs-seq:gs Holm {:.3}", niadj[tpos.unwrap()], niadj_gs[tpos.unwrap()]), Some(false) => format!("   TOST+-0.01: not shown equivalent; NI-vs-bestfair Holm {:.3}; NI-vs-seq:gs Holm {:.3}", niadj[tpos.unwrap()], niadj_gs[tpos.unwrap()]), None => String::new() }
        );
        cj.push(json!({"world": c.world, "generator": c.generator, "own_class": c.own_class, "control": c.control, "arm": c.arm, "diff": c.diff, "lo": c.lo, "hi": c.hi, "wins": c.wins, "losses": c.losses, "p": c.p, "holm": adj[i], "verdict": verdict, "late_diff": c.late_diff, "vs_mesh_guarded_diff": c.vs_mesh, "vs_mesh_holm": madj[i], "vs_mesh_verdict": mverdict, "vs_seq_guarded_diff": c.vs_gs, "tost_p": c.p_tost, "tost_holm": tpos.map(|k| tadj[k]), "ci90_inside_pm0.01": c.ci90_in, "equivalent": equiv, "ni_holm_vs_bestfair": tpos.map(|k| niadj[k]), "ni_holm_vs_seq_guarded": tpos.map(|k| niadj_gs[k])}));
    }
    println!("\nGenerator-level summary (an arm 'wins' a generator if it has a Holm WIN vs best fair in >=1 of that generator's non-control worlds; OWN-CLASS generators excluded from the independent count):");
    let mut gsum = serde_json::Map::new();
    for (setn, arm) in [1u8, 2u8, 3u8].iter().flat_map(|s| ["L-add", "L-pair", "L-gate", "L-full"].iter().map(move |a| (*s, *a))) {
        let mut gens: Vec<String> = vec![];
        let mut gens_all: Vec<String> = vec![];
        for (i, c) in cells.iter().enumerate() {
            if c.set == setn && c.arm == arm && !c.control && adj[i] < 0.05 && c.lo > 0.0 {
                if !gens_all.contains(&c.generator) {
                    gens_all.push(c.generator.clone());
                }
                if !c.own_class && !gens.contains(&c.generator) {
                    gens.push(c.generator.clone());
                }
            }
        }
        let loss_ctrl: Vec<String> = cells.iter().enumerate().filter(|(i, c)| c.set == setn && c.arm == arm && c.control && adj[*i] < 0.05 && c.hi < 0.0).map(|(_, c)| c.world.clone()).collect();
        let nonequiv: Vec<String> = ctrl_idx.iter().enumerate().filter(|(k, i)| cells[**i].arm == arm && tadj[*k] >= 0.05).map(|(_, i)| cells[*i].world.clone()).collect();
        println!("  set{setn} {arm:<7} independent generators won: {:?} (count {}); all incl. own-class: {:?}; controls with Holm LOSS: {:?}; controls NOT shown equivalent (+-0.01): {:?}", gens, gens.len(), gens_all, loss_ctrl, nonequiv);
        gsum.insert(format!("set{setn}:{arm}"), json!({"independent_generators_won": gens, "all_won": gens_all, "control_losses": loss_ctrl, "controls_not_equivalent": nonequiv}));
    }
    {
        let ps: Vec<f64> = armcmp.iter().map(|a| a.6).collect();
        let ad = holm_adjust(&ps);
        println!("
=== ablation: arm minus L-add (same world, paired over seeds; metric v; Holm over {} cells) -- does the pairwise term / the gates add anything over the additive model?", armcmp.len());
        let mut aj = vec![];
        for (i, a) in armcmp.iter().enumerate() {
            println!("set{} {:<11} {:<7} {:>+8.4} [{:>+8.4},{:>+8.4}] Holm {:.4} {}", a.1, a.0, a.2, a.3, a.4, a.5, ad[i], if ad[i] < 0.05 && a.4 > 0.0 { "BETTER than L-add" } else if ad[i] < 0.05 && a.5 < 0.0 { "WORSE than L-add" } else { "n.s." });
            aj.push(json!({"set": a.1, "world": a.0, "arm": a.2, "diff_vs_ladd": a.3, "lo": a.4, "hi": a.5, "p": a.6, "holm": ad[i]}));
        }
        result.insert("ablation_vs_ladd".into(), json!(aj));
    }
    result.insert("cells".into(), json!(cj));
    result.insert("generator_summary".into(), Value::Object(gsum));
    if let Some(path) = out {
        std::fs::write(&path, serde_json::to_string_pretty(&Value::Object(result)).unwrap()).unwrap();
        println!("wrote {path}");
    }
}

fn t90(df: usize) -> f64 {
    // two-sided 90% t critical value (one-sided 5%), small table + normal limit
    const T: [(usize, f64); 12] = [(1, 6.314), (2, 2.920), (3, 2.353), (4, 2.132), (5, 2.015), (6, 1.943), (8, 1.860), (10, 1.812), (15, 1.753), (20, 1.725), (30, 1.697), (60, 1.671)];
    for &(d, v) in T.iter() {
        if df <= d {
            return v;
        }
    }
    1.645
}

/// Holm adjustment applied separately within each world set (set 1 = frozen-first worlds, set 2 = added after the dev run)
fn holm_by_set(cells: &[Cell], p: &[f64]) -> Vec<f64> {
    let mut out = vec![1.0; p.len()];
    for s in [1u8, 2u8, 3u8] {
        let idx: Vec<usize> = (0..cells.len()).filter(|&i| cells[i].set == s).collect();
        let sub: Vec<f64> = idx.iter().map(|&i| p[i]).collect();
        let adj = holm_adjust(&sub);
        for (j, &i) in idx.iter().enumerate() {
            out[i] = adj[j];
        }
    }
    out
}

/// one-sided non-inferiority p-value, H0: mean paired diff <= -delta (small p => learned arm is not worse than baseline by delta)
fn ni_p(d: &[f64], delta: f64) -> f64 {
    let n = d.len();
    let m = neural_mesh::stats::mean(d);
    let se = neural_mesh::stats::sd(d) / (n as f64).sqrt();
    if se < 1e-15 {
        return if m > -delta { 0.0 } else { 1.0 };
    }
    upper_p((m + delta) / se, n - 1)
}
