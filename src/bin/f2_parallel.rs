//! F2(b): batched / multi-core routing throughput, honestly.
//!
//! What is measured: wall-clock of routing M tasks against a FROZEN trained router, serially and
//! on T std threads (per-call scoped threads, and a persistent pool for small batches), for the
//! crate's own HebbianRouter, a pre-interned fast path, and quantised state. Decision equivalence
//! (parallel == serial, bit for bit) is checked on every run. Also measured: what batching does to
//! learning (route B tasks on frozen state, then feed back B outcomes) in sample-efficiency terms.
//!
//! What this is NOT: a compute multiplier. The same routing work is done; wall-clock per batch
//! drops, per-task latency does not, learning updates stay serial, and speedup is capped by cores,
//! memory bandwidth and Amdahl. No claim of "more work on the same compute" is made.
//!
//! Usage: cargo run --release --bin f2_parallel [-- --out results/f2_parallel.json]

#[path = "f2_common/mod.rs"]
mod common;

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

use common::*;
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::holm_adjust;
use serde_json::{json, Value};

const SEED: u64 = 0xF2_0B;

fn task_rng(id: u64) -> Rng {
    Rng::new(SEED ^ id.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

// =============================================================================================
// Routers under test (frozen)

struct Trained {
    label: String,
    n_agents: usize,
    n_rows: usize,
    tasks: Vec<Task>,
    crate_router: HebbianRouter,
    mini: MiniHeb,
    ids: Vec<Vec<u32>>,
    cands: Vec<usize>,
}

fn train(label: &str, env: &Env, sch: &Scheme) -> (HebbianRouter, MiniHeb) {
    let n = env.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut hr = HebbianRouter::new(n, heb_robust_cfg());
    let mut mh = MiniHeb::new(n, sch.clone());
    let mut rng = Rng::new(5);
    for t in 0..env.len() {
        let task = env.task(t);
        let a = mh.select(task, &cands, &mut rng);
        let r = env.outcome(t, a);
        mh.update(task, a, r);
        hr.update(task, a, r);
    }
    let _ = label;
    (hr, mh)
}

fn build(label: &str, env: &Env) -> Trained {
    let (hr, mh) = train(label, env, &Scheme::new("f64", WKind::F64, BKind::F64));
    let tasks = env.tasks_vec();
    let ids = tasks.iter().map(|t| mh.lookup_ids(t)).collect();
    Trained {
        label: label.into(),
        n_agents: env.n_agents(),
        n_rows: mh.n_rows(),
        tasks,
        crate_router: hr,
        mini: mh,
        ids,
        cands: (0..env.n_agents()).collect(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    CrateStrings,
    MiniStrings,
    MiniIds,
}
impl Variant {
    fn label(self) -> &'static str {
        match self {
            Variant::CrateStrings => "crate HebbianRouter::select (String features, clone per thread)",
            Variant::MiniStrings => "compact router, String lookup per task",
            Variant::MiniIds => "compact router, pre-interned feature ids",
        }
    }
}

/// Route tasks `start..start+out.len()` (indices cycle over the trained task list; ids are unique).
fn route_range(tr: &Trained, v: Variant, local: Option<&mut HebbianRouter>, start: usize, out: &mut [u32]) {
    let nt = tr.tasks.len();
    match v {
        Variant::CrateStrings => {
            let r = local.unwrap();
            for (k, o) in out.iter_mut().enumerate() {
                let i = start + k;
                let t = &tr.tasks[i % nt]; // select ignores the task id; the per-task rng is keyed by the batch index
                let mut rng = task_rng(i as u64);
                *o = r.select(t, &tr.cands, &mut rng) as u32;
            }
        }
        Variant::MiniStrings => {
            for (k, o) in out.iter_mut().enumerate() {
                let i = start + k;
                let t = &tr.tasks[i % nt];
                let mut rng = task_rng(i as u64);
                *o = tr.mini.select_ref(t, &tr.cands, &mut rng) as u32;
            }
        }
        Variant::MiniIds => {
            for (k, o) in out.iter_mut().enumerate() {
                let i = start + k;
                let mut rng = task_rng(i as u64);
                *o = tr.mini.select_ids(&tr.ids[i % nt], &tr.cands, &mut rng) as u32;
            }
        }
    }
}

fn run_serial(tr: &Trained, v: Variant, m: usize, out: &mut [u32]) {
    let mut local = tr.crate_router.clone();
    route_range(tr, v, Some(&mut local), 0, &mut out[..m]);
}

fn run_scoped(tr: &Trained, v: Variant, m: usize, threads: usize, locals: &mut [HebbianRouter], out: &mut [u32]) {
    let chunk = m.div_ceil(threads);
    std::thread::scope(|s| {
        for ((ci, o), l) in out[..m].chunks_mut(chunk).enumerate().zip(locals.iter_mut()) {
            s.spawn(move || {
                route_range(tr, v, Some(l), ci * chunk, o);
            });
        }
    });
}

fn median(xs: &mut Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// (best, median) seconds over `reps` runs of `f`.
fn time_reps(reps: usize, mut f: impl FnMut()) -> (f64, f64) {
    let mut v = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        f();
        v.push(t.elapsed().as_secs_f64());
    }
    let best = v.iter().cloned().fold(f64::MAX, f64::min);
    (best, median(&mut v))
}

fn calibrate() -> f64 {
    // fixed single-thread integer loop; ns per iteration (contention indicator, not a benchmark)
    let t = Instant::now();
    let mut x = 1u64;
    for i in 0..60_000_000u64 {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(i | 1);
    }
    std::hint::black_box(x);
    t.elapsed().as_nanos() as f64 / 60_000_000.0
}

// =============================================================================================
// Persistent pool (std only)

struct PoolShared {
    tr: Trained,
    out: Vec<AtomicU32>,
}

struct Pool {
    txs: Vec<mpsc::Sender<Option<(usize, usize)>>>,
    done: mpsc::Receiver<()>,
    handles: Vec<std::thread::JoinHandle<()>>,
    shared: Arc<PoolShared>,
}

impl Pool {
    fn new(n: usize, shared: Arc<PoolShared>) -> Pool {
        let (dtx, drx) = mpsc::channel();
        let mut txs = Vec::new();
        let mut handles = Vec::new();
        for _ in 0..n {
            let (tx, rx) = mpsc::channel::<Option<(usize, usize)>>();
            let sh = shared.clone();
            let dtx = dtx.clone();
            handles.push(std::thread::spawn(move || {
                let nt = sh.tr.tasks.len();
                while let Ok(Some((a, b))) = rx.recv() {
                    for i in a..b {
                        let mut rng = task_rng(i as u64);
                        let d = sh.tr.mini.select_ids(&sh.tr.ids[i % nt], &sh.tr.cands, &mut rng);
                        sh.out[i].store(d as u32, Ordering::Relaxed);
                    }
                    dtx.send(()).ok();
                }
            }));
            txs.push(tx);
        }
        Pool { txs, done: drx, handles, shared }
    }
    fn run(&self, m: usize) {
        let n = self.txs.len();
        let chunk = m.div_ceil(n);
        let mut sent = 0;
        for (k, tx) in self.txs.iter().enumerate() {
            let a = k * chunk;
            let b = ((k + 1) * chunk).min(m);
            if a < b {
                tx.send(Some((a, b))).unwrap();
                sent += 1;
            }
        }
        for _ in 0..sent {
            self.done.recv().unwrap();
        }
    }
    fn decisions(&self, m: usize) -> Vec<u32> {
        (0..m).map(|i| self.shared.out[i].load(Ordering::Relaxed)).collect()
    }
    fn stop(self) {
        for tx in &self.txs {
            tx.send(None).ok();
        }
        for h in self.handles {
            h.join().ok();
        }
    }
}

// =============================================================================================
// Batched feedback = frozen state within a batch (what parallel routing implies for learning)

fn run_batched(env: &Env, sch: &Scheme, seed: u64, b: usize) -> f64 {
    let n = env.n_agents();
    let mut pol: Warm<MiniHeb> =
        Warm::new(MiniHeb::new(n, sch.clone()), env.kw_map(), n, 1.0).with_evidence(Evidence::Inner(mean_feature_n));
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut cum = 0.0;
    let mut t = 0;
    while t < env.len() {
        let end = (t + b).min(env.len());
        let mut chosen = Vec::with_capacity(end - t);
        for u in t..end {
            let a = pol.select(env.task(u), &cands, &mut rng);
            cum += env.prob(u, a);
            chosen.push(a);
        }
        for (k, u) in (t..end).enumerate() {
            pol.update(env.task(u), chosen[k], env.outcome(u, chosen[k]));
        }
        t = end;
    }
    cum / env.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned();
    let t0 = Instant::now();
    let cores = n_threads();
    let cal0 = calibrate();
    println!("logical cores (available_parallelism): {cores}; calibration loop {cal0:.3} ns/iter");

    // ---- trained states: moderate (32 agents) and large (128 agents, ~1.9k rows) ------------
    let env_mid = Env::big(31);
    let env_large = Env::big_sized(31, 128, 400, 20000);
    let states = vec![build("mid: 32 agents", &env_mid), build("large: 128 agents", &env_large)];
    for s in &states {
        println!("state {}: {} rows x {} agents, {} tasks", s.label, s.n_rows, s.n_agents, s.tasks.len());
    }

    let m_pool = 1 << 18; // pool output buffer capacity
    let thread_grid: Vec<usize> = [1usize, 2, 4, 8, 12, 16, 24, 32].iter().cloned().filter(|&t| t <= cores.max(1)).collect();
    let mut throughput = Vec::new();
    let mut equivalence_ok = true;
    let mut cal_samples = vec![cal0];

    for st in &states {
        for v in [Variant::CrateStrings, Variant::MiniStrings, Variant::MiniIds] {
            // choose M so that a serial pass takes ~0.2 s
            let probe = 4096usize;
            let mut tmp = vec![0u32; probe];
            let (pb, _) = time_reps(3, || run_serial(st, v, probe, &mut tmp));
            let per_task = pb / probe as f64;
            let m = (((0.2 / per_task) as usize).clamp(4096, 1 << 18) / 1024).max(4) * 1024;
            let reps = 5;
            let mut serial_out = vec![0u32; m];
            // paired reps: serial immediately followed by each thread count, ratios per rep
            let mut ser_times = Vec::new();
            let mut par_times: Vec<Vec<f64>> = vec![Vec::new(); thread_grid.len()];
            let mut ratios: Vec<Vec<f64>> = vec![Vec::new(); thread_grid.len()];
            let mut locals_all: Vec<Vec<HebbianRouter>> =
                thread_grid.iter().map(|&th| (0..th).map(|_| st.crate_router.clone()).collect()).collect();
            let mut same_all = vec![true; thread_grid.len()];
            let mut outs: Vec<Vec<u32>> = thread_grid.iter().map(|_| vec![0u32; m]).collect();
            for _ in 0..reps {
                let t = Instant::now();
                run_serial(st, v, m, &mut serial_out);
                let ts = t.elapsed().as_secs_f64();
                ser_times.push(ts);
                for (k, &th) in thread_grid.iter().enumerate() {
                    let t = Instant::now();
                    run_scoped(st, v, m, th, &mut locals_all[k], &mut outs[k]);
                    let tp = t.elapsed().as_secs_f64();
                    par_times[k].push(tp);
                    ratios[k].push(ts / tp);
                    same_all[k] &= outs[k] == serial_out;
                }
            }
            equivalence_ok &= same_all.iter().all(|&b| b);
            let s_best = ser_times.iter().cloned().fold(f64::MAX, f64::min);
            let s_med = median(&mut ser_times.clone());
            let mut rows = Vec::new();
            for (k, &th) in thread_grid.iter().enumerate() {
                let best = par_times[k].iter().cloned().fold(f64::MAX, f64::min);
                let med = median(&mut par_times[k].clone());
                let mut rr = ratios[k].clone();
                let rmed = median(&mut rr);
                rows.push(json!({
                    "threads": th, "best_ms": best * 1e3, "median_ms": med * 1e3,
                    "tasks_per_s_best": m as f64 / best,
                    "speedup_ratio_of_best": s_best / best, "speedup_median_of_paired_ratios": rmed,
                    "parallel_efficiency_median": rmed / th as f64,
                    "decisions_identical_to_serial": same_all[k],
                }));
                println!(
                    "{:<17} {:<62} M={:<6} T={:<2} best {:>8.1} ms  speedup best/best {:>5.2}x  paired-median {:>5.2}x  eff {:>4.0}%  identical={}",
                    st.label, v.label(), m, th, best * 1e3, s_best / best, rmed, 100.0 * rmed / th as f64, same_all[k]
                );
            }
            cal_samples.push(calibrate());
            throughput.push(json!({
                "state": st.label, "variant": v.label(), "m": m,
                "serial_best_ms": s_best * 1e3, "serial_median_ms": s_med * 1e3,
                "serial_ns_per_task": s_best * 1e9 / m as f64,
                "rows": rows,
            }));
        }
    }
    let m_big = 20000usize;
    // cross-implementation equivalence (crate vs compact, serial): identical decisions
    let st = &states[0];
    let mut a = vec![0u32; 20000];
    let mut b = vec![0u32; 20000];
    let mut c = vec![0u32; 20000];
    run_serial(st, Variant::CrateStrings, 20000, &mut a);
    run_serial(st, Variant::MiniStrings, 20000, &mut b);
    run_serial(st, Variant::MiniIds, 20000, &mut c);
    let cross_ok = a == b && b == c;
    println!("crate vs compact(String) vs compact(ids) decisions identical on 20000 tasks: {cross_ok}");

    // ---- small batches: per-call scoped threads vs persistent pool ------------------------
    println!("\n--- small batches (state: mid, pre-interned compact router) ---");
    let mid = Arc::new(PoolShared { tr: build("mid: 32 agents", &env_mid), out: (0..m_pool).map(|_| AtomicU32::new(0)).collect() });
    let pool = Pool::new(cores, mid.clone());
    let mut small = Vec::new();
    for &m in &[16usize, 64, 256, 1024, 4096, 16384, 65536, 262144] {
        let iters = (400_000 / m).clamp(3, 1500);
        let mut ser_out = vec![0u32; m];
        let mut ser_t = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..iters {
                route_range(&mid.tr, Variant::MiniIds, None, 0, &mut ser_out);
            }
            ser_t = ser_t.min(t.elapsed().as_secs_f64() / iters as f64);
        }
        let mut locals: Vec<HebbianRouter> = Vec::new(); // unused for MiniIds
        let mut sc_out = vec![0u32; m];
        let mut sc_t = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..iters {
                // scoped threads, `cores` threads, no clones needed for the compact router
                let th = cores.min(m);
                let chunk = m.div_ceil(th);
                let trr = &mid.tr;
                std::thread::scope(|s| {
                    for (ci, o) in sc_out.chunks_mut(chunk).enumerate() {
                        s.spawn(move || route_range(trr, Variant::MiniIds, None, ci * chunk, o));
                    }
                });
            }
            sc_t = sc_t.min(t.elapsed().as_secs_f64() / iters as f64);
        }
        let _ = &mut locals;
        let mut pool_t = f64::MAX;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..iters {
                pool.run(m);
            }
            pool_t = pool_t.min(t.elapsed().as_secs_f64() / iters as f64);
        }
        let pd = pool.decisions(m);
        let ok = pd == ser_out && sc_out == ser_out;
        equivalence_ok &= ok;
        println!(
            "M={:<7} serial {:>10.1} us   scoped x{} {:>10.1} us ({:>5.2}x)   pool x{} {:>10.1} us ({:>5.2}x)  identical={}",
            m, ser_t * 1e6, cores, sc_t * 1e6, ser_t / sc_t, cores, pool_t * 1e6, ser_t / pool_t, ok
        );
        small.push(json!({"m": m, "serial_us": ser_t * 1e6, "scoped_us": sc_t * 1e6, "pool_us": pool_t * 1e6,
            "scoped_speedup": ser_t / sc_t, "pool_speedup": ser_t / pool_t, "identical": ok}));
    }
    drop(pool);

    // ---- route vs update cost (Amdahl): learning updates are NOT parallelised here -----------
    let env_w = Env::big(32);
    let sch = Scheme::new("f64", WKind::F64, BKind::F64);
    let mut mh = MiniHeb::new(32, sch.clone());
    let mut hr = HebbianRouter::new(32, heb_robust_cfg());
    let cands: Vec<usize> = (0..32).collect();
    let mut rng = Rng::new(3);
    let tasks = env_w.tasks_vec();
    let (mut route_ns, mut upd_ns, mut crate_upd_ns) = (f64::MAX, f64::MAX, f64::MAX);
    for _ in 0..5 {
        let t = Instant::now();
        let mut acc = 0usize;
        for task in tasks.iter() {
            acc += hr.select(task, &cands, &mut rng);
        }
        std::hint::black_box(acc);
        route_ns = route_ns.min(t.elapsed().as_nanos() as f64 / tasks.len() as f64);
        let t = Instant::now();
        for (i, task) in tasks.iter().enumerate() {
            mh.update(task, i % 32, (i % 3 == 0) as u8 as f64);
        }
        upd_ns = upd_ns.min(t.elapsed().as_nanos() as f64 / tasks.len() as f64);
        let t = Instant::now();
        for (i, task) in tasks.iter().enumerate() {
            hr.update(task, i % 32, (i % 3 == 0) as u8 as f64);
        }
        crate_upd_ns = crate_upd_ns.min(t.elapsed().as_nanos() as f64 / tasks.len() as f64);
    }
    let f_upd = crate_upd_ns / (crate_upd_ns + route_ns);
    let amdahl: Vec<Value> = thread_grid
        .iter()
        .map(|&t| json!({"threads": t, "max_speedup_route_plus_update": 1.0 / (f_upd + (1.0 - f_upd) / t as f64)}))
        .collect();
    println!("\ncrate router: route {route_ns:.0} ns/task, update {crate_upd_ns:.0} ns/task (update share {:.0}%); compact f64 update {upd_ns:.0} ns", f_upd * 100.0);

    // ---- frozen-state decision agreement of quantised states vs f64 (same per-task rng) ------
    // Teacher-forced: one f64 router drives; each twin is fed the identical (task, agent, reward)
    // history; afterwards every frozen state routes the same tasks with the same random stream.
    println!("\n--- frozen quantised router: decision agreement with the f64 router (same history) on 20000 tasks ---");
    let mut agree_rows = Vec::new();
    let tw_schemes = [
        ("f32", Scheme::new("f32", WKind::F32, BKind::F32)),
        ("f16", Scheme::new("f16", WKind::F16, BKind::U16)),
        ("bf16-sr", Scheme::new("bf16", WKind::Bf { bits: 16, sr: true }, BKind::U16)),
        ("bf8-sr", Scheme::new("bf8", WKind::Bf { bits: 8, sr: true }, BKind::U8Sr)),
        ("bf4-sr", Scheme::new("bf4", WKind::Bf { bits: 4, sr: true }, BKind::U8Sr)),
    ];
    let mut driver = MiniHeb::new(32, Scheme::new("f64", WKind::F64, BKind::F64));
    let mut twins: Vec<MiniHeb> = tw_schemes.iter().map(|(_, s)| MiniHeb::new(32, s.clone())).collect();
    {
        let cands: Vec<usize> = (0..32).collect();
        let mut rng = Rng::new(5);
        for t in 0..env_w.len() {
            let task = env_w.task(t);
            let a = driver.select(task, &cands, &mut rng);
            let r = env_w.outcome(t, a);
            driver.update(task, a, r);
            for tw in twins.iter_mut() {
                tw.update(task, a, r);
            }
        }
    }
    for ((label, _), q) in tw_schemes.iter().zip(&twins) {
        let mut hit = 0usize;
        for i in 0..20000usize {
            let t = &tasks[i % tasks.len()];
            let mut r1 = task_rng(i as u64);
            let mut r2 = task_rng(i as u64);
            if driver.select_ref(t, &cands, &mut r1) == q.select_ref(t, &cands, &mut r2) {
                hit += 1;
            }
        }
        println!("  {label:<8} agreement {:.4}", hit as f64 / 20000.0);
        agree_rows.push(json!({"scheme": label, "agreement": hit as f64 / 20000.0}));
    }

    // ---- batching & learning (sample efficiency) ------------------------------------------
    println!("\n--- batched feedback (route B on frozen state, then update B): quality vs B=1 ---");
    let worlds = all_worlds();
    let sch = Scheme::new("f64", WKind::F64, BKind::F64);
    let bs = [4usize, 16, 64, 256];
    let mut batch_doc = Vec::new();
    for (split, sd) in [("held-out 31-60", held_seeds()), ("confirm 61-90", confirm_seeds()), ("power ext 91-250", seeds(91, 250))] {
        let mut cells = Vec::new();
        for spec in &worlds {
            let base: Vec<f64> = par_map(&sd, |&s| run_batched(&spec.make(s), &sch, s, 1));
            for &b in &bs {
                let v: Vec<f64> = par_map(&sd, |&s| run_batched(&spec.make(s), &sch, s, b));
                cells.push((spec.label(), b, compare(&v, &base)));
            }
        }
        let ps: Vec<f64> = cells.iter().map(|c| c.2.p_equiv).collect();
        let holm = holm_adjust(&ps);
        println!("{split} (n={}):", sd.len());
        let mut js = Vec::new();
        for ((w, b, c), h) in cells.iter().zip(&holm) {
            println!("  {:<14} B={:<4} dQ {}  Holm-equiv {:.3}  {}", w, b, fmt_cmp(c), h, if *h < 0.05 && c.lo > -MARGIN && c.hi < MARGIN { "EQUIVALENT" } else { "-" });
            let mut j = cmp_json(c, Some(*h));
            j["world"] = json!(w);
            j["batch"] = json!(b);
            js.push(j);
        }
        batch_doc.push(json!({"split": split, "n_seeds": sd.len(), "cells": js}));
    }

    let cal1 = calibrate();
    println!("\ncalibration loop at end: {cal1:.3} ns/iter (start {cal0:.3})");
    println!("ALL parallel decisions identical to serial: {equivalence_ok}");

    let doc = json!({
        "part": "F2(b) batched / parallel routing",
        "machine": {"logical_cores": cores, "calibration_ns_start": cal0, "calibration_ns_end": cal1, "calibration_ns_samples": cal_samples,
            "note": "other jobs may share this machine (calibration drift is a contention indicator); numbers are best-of-5 wall clock"},
        "frozen_routing_throughput": throughput,
        "small_batches": small,
        "crate_route_ns": route_ns, "crate_update_ns": crate_upd_ns, "compact_f64_update_ns": upd_ns,
        "update_share_of_route_plus_update": f_upd,
        "amdahl_bound_if_only_routing_parallel": amdahl,
        "all_parallel_decisions_identical_to_serial": equivalence_ok,
        "crate_vs_compact_decisions_identical": cross_ok,
        "frozen_quantised_decision_agreement": agree_rows,
        "batched_feedback_quality": batch_doc,
        "honesty": [
            "Same total routing work; only wall-clock for a batch shrinks. Single-task latency does not improve (it gets worse with thread hand-off).",
            "Learning updates are serial in this design; the Amdahl bound above applies if updates stay serial.",
            "Speedup is limited by logical cores (hyperthreads are not full cores), shared cache/memory bandwidth and other jobs on the machine.",
            "Batching changes learning: routing B tasks on frozen state delays feedback; see batched_feedback_quality for the measured sample-efficiency cost.",
            "No claim of 'more work on the same compute'."
        ],
        "runtime_s": t0.elapsed().as_secs_f64(),
    });
    if let Some(p) = out_path {
        std::fs::write(&p, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        println!("wrote {p}");
    }
    println!("runtime {:.1}s", t0.elapsed().as_secs_f64());
}
