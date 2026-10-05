//! H5 / R4-B2 (MEMORY-REDUCTION): bytes, state size, cold-brain spill and decision latency of the
//! R4 CompactRouter against (a) the f64 recommended router and (b) the R3/W3 compact router (a verbatim copy
//! of the old src/compact.rs, `h5_common/compact_g3.rs`, compiled into this bin as the A/B baseline).
//!
//! DECLARED CRITERIA (before running):
//!   latency   new u16/q16 must not be slower than the R3 compact by more than +10% (min over interleaved
//!             repetitions of route+feedback ns/decision, fresh and warm pass); precision / cap variants
//!             are reported against both the R3 compact and the f64 reference (no pass/fail, a cost table).
//!   bytes     exact: live bytes from a counting allocator after K independently trained brains per world
//!             (K per world below, each trained on its own world seed 1000+k, world data freed), incl. the
//!             one shared vocabulary / keyword table when shared.
//!   state     serialised size of one trained brain: R3 JSON, R4 JSON, R4 binary (self-contained),
//!             R4 binary state-only (cold form); every binary form is round-trip checked EXACTLY here.
//!   pool      cold-brain spill: K fully TRAINED brains (own world each), 5% hot; resident bytes per brain (hot brains + cold blobs +
//!             shared vocab, counting allocator) and spill / reload ns; the pooled decisions are checked
//!             equal to an all-hot run (bit-exact) on a subset.
//!   fairness  every router gets the same precomputed outcome table, same rng seed, same candidate list;
//!             string->id work is INCLUDED in the compact ns. Shared box: only the interleaved ratios count.
//!
//! Usage: cargo run --release --bin h5_mem -- [--out results/h5_mem.json] [--quick] [--quiet-wait 60] [--latency-only]

#[path = "h5_common/mod.rs"]
mod common;
#[path = "h5_common/compact_g3.rs"]
mod compact_g3;

use std::time::Instant;

use common::alloc_count;
use common::*;
use neural_mesh::compact::{BrainPool, CompactConfig, CompactRouterG, EvictPolicy, Fx12, Fx16, IdMode, Q12, Q16, SharedVocab};
use neural_mesh::defaults::recommended_router;
use neural_mesh::router::{Rng, Router, Task};
use serde_json::json;

#[global_allocator]
static GLOBAL: alloc_count::Counting = alloc_count::Counting;

const GIB: f64 = (1u64 << 30) as f64;

struct Table {
    n: usize,
    out: Vec<f64>,
}
fn outcome_table(env: &Env) -> Table {
    let n = env.n_agents();
    let mut out = Vec::with_capacity(env.len() * n);
    for t in 0..env.len() {
        for a in 0..n {
            out.push(env.outcome(t, a));
        }
    }
    Table { n, out }
}
fn pass<R: Router + ?Sized>(r: &mut R, tasks: &[Task], tab: &Table, cands: &[usize], rng: &mut Rng) -> f64 {
    let t0 = Instant::now();
    let mut sink = 0usize;
    for (t, task) in tasks.iter().enumerate() {
        let a = r.select(task, cands, rng);
        sink = sink.wrapping_add(a);
        r.update(task, a, tab.out[t * tab.n + a]);
    }
    std::hint::black_box(sink);
    t0.elapsed().as_nanos() as f64 / tasks.len() as f64
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}
fn minf(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::MAX, f64::min)
}

type U16 = CompactRouterG<Fx16, Q16>;
type U12 = CompactRouterG<Fx12, Q12>;
type G3 = compact_g3::CompactRouter;

fn make_g3(n: usize, kw: &std::collections::HashMap<String, usize>) -> G3 {
    G3::new(n, kw, compact_g3::CompactConfig::default()).unwrap()
}

/// a latency subject: constructor of a fresh boxed router
struct Lat {
    name: &'static str,
    make: Box<dyn Fn() -> Box<dyn Router>>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/h5_mem.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let quiet_wait: u64 = args.iter().position(|a| a == "--quiet-wait").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(60);
    let lat_only = args.iter().any(|a| a == "--latency-only");
    let reps = if quick { 7 } else { 21 };
    let t0 = Instant::now();
    let worlds = all_worlds();
    let mut world_json = Vec::new();
    let mut headline = Vec::new();

    let only: Option<String> = args.iter().position(|a| a == "--world").and_then(|i| args.get(i + 1)).cloned();
    for spec in &worlds {
        let label = spec.label();
        if only.as_ref().map(|o| !label.contains(o.as_str())).unwrap_or(false) {
            continue;
        }
        let big = !matches!(spec, WorldSpec::Sim(_));
        let k_brains: usize = match (spec, quick) {
            (WorldSpec::Sim(_), false) => 400,
            (WorldSpec::Big(n), false) if *n >= 64 => 80,
            (WorldSpec::Big(_), false) => 120,
            (_, false) => 40,
            (WorldSpec::Sim(_), true) => 40,
            (_, true) => 12,
        };
        println!("== {label}");
        let seed = 31u64;
        let env = spec.make(seed);
        let n = env.n_agents();
        let kw = env.kw_map();
        let tasks = env.tasks_vec();
        let tab = outcome_table(&env);
        let cands: Vec<usize> = (0..n).collect();
        let g3_ok = n <= 64;

        // natural sizes for the cap variants (this seed's uncapped run)
        let nat_rows = {
            let mut r = U16::new(n, &kw, CompactConfig::default()).unwrap();
            run_policy(&env, &mut r, seed);
            r.n_rows()
        };
        let cap_half = (nat_rows / 2).max(1);

        // ---------------- latency: interleaved A/B ----------------
        let mut subs: Vec<Lat> = Vec::new();
        {
            let kw2 = kw.clone();
            subs.push(Lat { name: "ref f64 guarded(hebr)", make: Box::new(move || Box::new(recommended_router(kw2.clone(), n))) });
        }
        if g3_ok {
            let kw2 = kw.clone();
            subs.push(Lat { name: "R3 compact u16/q16", make: Box::new(move || Box::new(make_g3(n, &kw2))) });
        }
        {
            let kw2 = kw.clone();
            subs.push(Lat { name: "R4 compact u16/q16", make: Box::new(move || Box::new(U16::new(n, &kw2, CompactConfig::default()).unwrap())) });
        }
        {
            let kw2 = kw.clone();
            subs.push(Lat { name: "R4 u12/q12 packed", make: Box::new(move || Box::new(U12::new(n, &kw2, CompactConfig::default()).unwrap())) });
        }
        {
            let kw2 = kw.clone();
            subs.push(Lat { name: "R4 u16/q16 rowcap=50%", make: Box::new(move || Box::new(U16::new(n, &kw2, CompactConfig { max_rows: cap_half, ..CompactConfig::default() }).unwrap())) });
        }
        {
            let kw2 = kw.clone();
            subs.push(Lat { name: "R4 u16/q16 hash32", make: Box::new(move || Box::new(U16::new(n, &kw2, CompactConfig { id_mode: IdMode::Hash32, ..CompactConfig::default() }).unwrap())) });
        }
        let bg = if quiet_wait > 0 { wait_quiet(quiet_wait, 12.0) } else { cpu_busy_pct(300) };
        let ns = subs.len();
        let mut fresh: Vec<Vec<f64>> = vec![vec![]; ns];
        let mut warm: Vec<Vec<f64>> = vec![vec![]; ns];
        for rep in 0..reps + 2 {
            let mut routers: Vec<Box<dyn Router>> = subs.iter().map(|s| (s.make)()).collect();
            let mut rngs: Vec<Rng> = (0..ns).map(|_| Rng::new(77)).collect();
            let mut order: Vec<usize> = (0..ns).collect();
            order.rotate_left(rep % ns);
            if rep % 2 == 1 {
                order.reverse();
            }
            let mut f = vec![0.0; ns];
            let mut w = vec![0.0; ns];
            for &i in &order {
                f[i] = pass(routers[i].as_mut(), &tasks, &tab, &cands, &mut rngs[i]);
            }
            for &i in order.iter().rev() {
                w[i] = pass(routers[i].as_mut(), &tasks, &tab, &cands, &mut rngs[i]);
            }
            if rep >= 2 {
                for i in 0..ns {
                    fresh[i].push(f[i]);
                    warm[i].push(w[i]);
                }
            }
        }
        let find = |name: &str| subs.iter().position(|s| s.name == name);
        let (ir, ig, in_) = (find("ref f64 guarded(hebr)").unwrap(), find("R3 compact u16/q16"), find("R4 compact u16/q16").unwrap());
        println!("  ns/decision (route+feedback), min over {reps} interleaved reps [median]; ratio vs ref f64 | vs R3 compact (fresh / warm); bg CPU {bg:.0}%");
        let mut lat_json = Vec::new();
        for i in 0..ns {
            let (fm, wm) = (minf(&fresh[i]), minf(&warm[i]));
            let vs_ref = (fm / minf(&fresh[ir]), wm / minf(&warm[ir]));
            let vs_g3 = ig.map(|g| (fm / minf(&fresh[g]), wm / minf(&warm[g])));
            println!(
                "   {:26} fresh {:7.0} [{:7.0}]  warm {:7.0} [{:7.0}]   x{:.2}/{:.2} vs ref   {}",
                subs[i].name, fm, median(&fresh[i]), wm, median(&warm[i]), vs_ref.0, vs_ref.1,
                vs_g3.map(|v| format!("x{:.2}/{:.2} vs R3 compact", v.0, v.1)).unwrap_or_else(|| "(R3 compact limited to 64 agents)".into())
            );
            lat_json.push(json!({"subject": subs[i].name, "fresh_min": fm, "fresh_median": median(&fresh[i]), "warm_min": wm, "warm_median": median(&warm[i]),
                "ratio_vs_ref_fresh": vs_ref.0, "ratio_vs_ref_warm": vs_ref.1, "ratio_vs_r3_fresh": vs_g3.map(|v| v.0), "ratio_vs_r3_warm": vs_g3.map(|v| v.1)}));
        }
        let r4_vs_r3 = ig.map(|g| (minf(&fresh[in_]) / minf(&fresh[g]), minf(&warm[in_]) / minf(&warm[g])));
        let lat_ok = r4_vs_r3.map(|v| v.0 <= 1.10 && v.1 <= 1.10);
        println!("   declared criterion (R4 u16/q16 within +10% of R3 compact): {:?}", lat_ok);
        if lat_only {
            continue;
        }

        // ---------------- one trained brain: bytes and state size ----------------
        alloc_count::start();
        let mut b_ref = recommended_router(kw.clone(), n);
        run_policy(&env, &mut b_ref, seed);
        let m_ref = alloc_count::stop().live;
        drop(b_ref);
        let m_g3 = if g3_ok {
            alloc_count::start();
            let mut b = make_g3(n, &kw);
            run_policy(&env, &mut b, seed);
            b.shrink_to_fit();
            let m = alloc_count::stop().live;
            drop(b);
            Some(m)
        } else {
            None
        };
        alloc_count::start();
        let mut b_new = U16::new(n, &kw, CompactConfig::default()).unwrap();
        run_policy(&env, &mut b_new, seed);
        b_new.shrink_to_fit();
        let m_new = alloc_count::stop().live;
        let g3_json = if g3_ok {
            let mut b = make_g3(n, &kw);
            run_policy(&env, &mut b, seed);
            b.to_json().map(|s| s.len()).ok()
        } else {
            None
        };
        // state sizes + exact round trip + latency of (de)serialisation
        let json_new = b_new.to_json().unwrap();
        let bin = b_new.to_bytes();
        let state = b_new.state_bytes();
        let rt = U16::from_bytes(&bin).unwrap();
        let exact_bin = rt.to_bytes() == bin && rt.to_json().unwrap() == json_new;
        let exact_state = {
            let sv = SharedVocab::new();
            let proto = U16::with_shared_vocab(n, &kw, CompactConfig::default(), &sv).unwrap();
            let mut trained = proto.fork_blank();
            run_policy(&env, &mut trained, seed);
            let st = trained.state_bytes();
            let back = U16::from_state_bytes(&proto, &st).unwrap();
            back.state_bytes() == st && back.to_bytes() == trained.to_bytes()
        };
        let tm = |f: &dyn Fn()| -> f64 {
            let it = 30;
            let t = Instant::now();
            for _ in 0..it {
                f();
            }
            t.elapsed().as_nanos() as f64 / it as f64
        };
        let ns_to_json = tm(&|| {
            std::hint::black_box(b_new.to_json().unwrap());
        });
        let ns_from_json = tm(&|| {
            std::hint::black_box(U16::from_json(&json_new).unwrap());
        });
        let ns_to_bin = tm(&|| {
            std::hint::black_box(b_new.to_bytes());
        });
        let ns_from_bin = tm(&|| {
            std::hint::black_box(U16::from_bytes(&bin).unwrap());
        });
        let ns_state = tm(&|| {
            std::hint::black_box(b_new.state_bytes());
        });
        let sv0 = SharedVocab::new();
        let proto0 = U16::with_shared_vocab(n, &kw, CompactConfig::default(), &sv0).unwrap();
        let mut tr0 = proto0.fork_blank();
        run_policy(&env, &mut tr0, seed);
        let st0 = tr0.state_bytes();
        let ns_unstate = tm(&|| {
            std::hint::black_box(U16::from_state_bytes(&proto0, &st0).unwrap());
        });
        let (bf_bin, bf_state) = {
            let mut b12 = U12::new(n, &kw, CompactConfig::default()).unwrap();
            run_policy(&env, &mut b12, seed);
            (b12.to_bytes().len(), b12.state_bytes().len())
        };
        println!(
            "  one trained brain: live bytes ref {m_ref}  R3 compact {:?}  R4 compact {m_new} (owned {} + vocab~{} + kw {}); JSON R3 {:?} -> R4 {}  binary self-contained {}  binary state-only {}  (u12/q12: {} / {})",
            m_g3, b_new.bytes().owned(), b_new.bytes().shared_vocab_bytes, b_new.bytes().shared_kw_bytes, g3_json, json_new.len(), bin.len(), state.len(), bf_bin, bf_state
        );
        println!("  serde exact round trip: binary {exact_bin}, state-only {exact_state}; ns: to_json {ns_to_json:.0} from_json {ns_from_json:.0} | to_bytes {ns_to_bin:.0} from_bytes {ns_from_bin:.0} | state {ns_state:.0} restore {ns_unstate:.0}");

        // ---------------- K brains: agents per GiB ----------------
        let seeds_k: Vec<u64> = (0..k_brains as u64).map(|i| 1000 + i).collect();
        let train = |e: &Env, r: &mut dyn Router, s: u64| {
            let n = e.n_agents();
            let cands: Vec<usize> = (0..n).collect();
            let mut rng = Rng::new(s.wrapping_mul(1_000_003).wrapping_add(17));
            for t in 0..e.len() {
                let task = e.task(t);
                let a = r.select(task, &cands, &mut rng);
                r.update(task, a, e.outcome(t, a));
            }
        };
        // variants: (name, builder returning live bytes per brain)
        let names = [
            "ref f64 guarded(hebr) [shipped]",
            "R3 compact u16/q16 private",
            "R4 compact u16/q16 private",
            "R4 u16/q16 SHARED vocab",
            "R4 u12/q12 SHARED vocab",
            "R4 u16/q16 SHARED rowcap=50%",
            "R4 u12/q12 SHARED rowcap=50%",
            "R4 u16/q16 SHARED, cold state-only blobs",
        ];
        let mut per = Vec::new();
        for v in 0..names.len() {
            if v == 1 && !g3_ok {
                per.push(json!({"variant": names[v], "skipped": "R3 compact limited to 64 agents"}));
                continue;
            }
            let (pf0, _) = proc_mem();
            alloc_count::start();
            let mut keep_ref: Vec<neural_mesh::prior::Guarded<neural_mesh::hebbian::HebbianRouter>> = Vec::new();
            let mut keep_g3: Vec<G3> = Vec::new();
            let mut keep_a: Vec<U16> = Vec::new();
            let mut keep_b: Vec<U12> = Vec::new();
            let mut keep_cold: Vec<Box<[u8]>> = Vec::new();
            let sv16 = SharedVocab::new();
            let sv12 = SharedVocab::new();
            let cfg_cap = CompactConfig { max_rows: cap_half, ..CompactConfig::default() };
            let proto16 = if matches!(v, 3 | 5 | 7) { Some(U16::with_shared_vocab(n, &kw, if v == 5 { cfg_cap.clone() } else { CompactConfig::default() }, &sv16).unwrap()) } else { None };
            let proto12 = if matches!(v, 4 | 6) { Some(U12::with_shared_vocab(n, &kw, if v == 6 { cfg_cap.clone() } else { CompactConfig::default() }, &sv12).unwrap()) } else { None };
            for &s in &seeds_k {
                let e = spec.make(s);
                match v {
                    0 => {
                        let mut r = recommended_router(kw.clone(), n);
                        train(&e, &mut r, s);
                        keep_ref.push(r);
                    }
                    1 => {
                        let mut r = make_g3(n, &kw);
                        train(&e, &mut r, s);
                        r.shrink_to_fit();
                        keep_g3.push(r);
                    }
                    2 => {
                        let mut r = U16::new(n, &kw, CompactConfig::default()).unwrap();
                        train(&e, &mut r, s);
                        r.shrink_to_fit();
                        keep_a.push(r);
                    }
                    3 | 5 => {
                        let mut r = proto16.as_ref().unwrap().fork_blank();
                        train(&e, &mut r, s);
                        r.shrink_to_fit();
                        keep_a.push(r);
                    }
                    4 | 6 => {
                        let mut r = proto12.as_ref().unwrap().fork_blank();
                        train(&e, &mut r, s);
                        r.shrink_to_fit();
                        keep_b.push(r);
                    }
                    _ => {
                        let mut r = proto16.as_ref().unwrap().fork_blank();
                        train(&e, &mut r, s);
                        keep_cold.push(r.state_bytes().into_boxed_slice());
                    }
                }
            }
            let live = alloc_count::stop().live;
            let (pf1, _) = proc_mem();
            let per_brain = live as f64 / k_brains as f64;
            let apg = GIB / per_brain;
            println!("  K={k_brains} brains, {:42}: {:>9.0} B live/brain (incl. shared vocab/kw amortised)  commit {:>8} B/brain  -> {:>10.0} agents/GiB", names[v], per_brain, pf1.saturating_sub(pf0) / k_brains, apg);
            per.push(json!({"variant": names[v], "k": k_brains, "live_bytes_per_brain": per_brain, "agents_per_gib": apg, "commit_bytes_per_brain_unreliable": pf1.saturating_sub(pf0) / k_brains}));
            drop((keep_ref, keep_g3, keep_a, keep_b, keep_cold, proto16, proto12));
        }

        // ---------------- end-to-end state on disk per brain (K brains serialised) ----------------
        let sample = k_brains.min(if big { 12 } else { 40 });
        let (mut j_old, mut j_new, mut b_new_sum, mut b_state) = (0usize, 0usize, 0usize, 0usize);
        for &s in seeds_k.iter().take(sample) {
            let e = spec.make(s);
            let mut r = U16::new(n, &kw, CompactConfig::default()).unwrap();
            train(&e, &mut r, s);
            j_new += r.to_json().unwrap().len();
            b_new_sum += r.to_bytes().len();
            b_state += r.state_bytes().len();
            if g3_ok {
                let mut g = make_g3(n, &kw);
                train(&e, &mut g, s);
                j_old += g.to_json().unwrap().len();
            }
        }
        let sm = sample as f64;
        println!("  serialised state per brain (mean of {sample}): R3 JSON {:.0} B | R4 JSON {:.0} B | R4 binary {:.0} B | R4 binary state-only {:.0} B", j_old as f64 / sm, j_new as f64 / sm, b_new_sum as f64 / sm, b_state as f64 / sm);

        // ---------------- cold pool: K fully trained brains, 5% hot ----------------
        // every key is trained on its OWN world (seed 1000+k, the same data as the K-brain variants above) through
        // the pool (so the pool spills the LRU brain while it trains the next), then the pool is left as it is:
        // `max_hot` live brains + the rest as state-only blobs + the shared vocabulary (counting allocator).
        let pool_keys = k_brains.max(40);
        let max_hot = (pool_keys / 20).max(2);
        alloc_count::start();
        let sv = SharedVocab::new();
        let proto = U16::with_shared_vocab(n, &kw, CompactConfig::default(), &sv).unwrap();
        let mut pool = BrainPool::new(proto.fork_blank(), max_hot);
        for k in 0..pool_keys {
            let e = spec.make(1000 + k as u64);
            let b = pool.get(k as u64).unwrap();
            train(&e, b, 1000 + k as u64);
        }
        let live_pool = alloc_count::snap().live;
        let (hot_n, cold_n, cold_b, st) = (pool.hot_len(), pool.cold_len(), pool.cold_bytes(), pool.stats);
        alloc_count::stop();
        let per_brain_pool = live_pool as f64 / pool_keys as f64;
        // all-hot twins trained identically (outside the counting window)
        let mut twins: Vec<U16> = (0..pool_keys)
            .map(|k| {
                let e = spec.make(1000 + k as u64);
                let mut b = proto.fork_blank();
                train(&e, &mut b, 1000 + k as u64);
                b
            })
            .collect();
        // reload latency: every key once (state-only blob -> brain), measured one at a time
        let mut lat_reload = Vec::new();
        let mut lat_spill = Vec::new();
        for k in 0..pool_keys.min(60) {
            let key = (k * 7 % pool_keys) as u64;
            let t = Instant::now();
            let _ = pool.get(key).unwrap();
            lat_reload.push(t.elapsed().as_nanos() as f64);
            pool.spill_all();
            let t = Instant::now();
            let _ = pool.get(key).unwrap();
            lat_spill.push(t.elapsed().as_nanos() as f64);
        }
        // churn: uniform random keys, one decision each; pooled vs all-hot twins, same keys / rng seeds; decisions must match
        let churn = tasks.len().min(2000);
        let (mut rng_a, mut rng_b) = (Rng::new(9), Rng::new(9));
        let mut rngs_a: Vec<Rng> = (0..pool_keys).map(|k| Rng::new(500 + k as u64)).collect();
        let mut rngs_b = rngs_a.clone();
        let mut dec_a = Vec::with_capacity(churn);
        let t_pool = Instant::now();
        for (t, task) in tasks.iter().take(churn).enumerate() {
            let k = rng_a.below(pool_keys);
            let b = pool.get(k as u64).unwrap();
            let a = b.select(task, &cands, &mut rngs_a[k]);
            b.update(task, a, tab.out[t * tab.n + a]);
            dec_a.push(a);
        }
        let pool_ns = t_pool.elapsed().as_nanos() as f64 / churn as f64;
        let mut dec_b = Vec::with_capacity(churn);
        let t_hot = Instant::now();
        for (t, task) in tasks.iter().take(churn).enumerate() {
            let k = rng_b.below(pool_keys);
            let a = twins[k].select(task, &cands, &mut rngs_b[k]);
            twins[k].update(task, a, tab.out[t * tab.n + a]);
            dec_b.push(a);
        }
        let hot_ns = t_hot.elapsed().as_nanos() as f64 / churn as f64;
        let identical = dec_a == dec_b && {
            pool.spill_all();
            (0..pool_keys).all(|k| pool.get(k as u64).unwrap().to_bytes() == twins[k].to_bytes())
        };
        println!(
            "  cold pool: {pool_keys} TRAINED keys, max_hot {max_hot} ({}%): after training {hot_n} hot + {cold_n} cold blobs ({cold_b} B = {:.0} B/blob), live incl. vocab/kw {:.0} B/brain -> {:.0} agents/GiB; churn {pool_ns:.0} ns/decision vs all-hot {hot_ns:.0} (x{:.2}); spills {} reloads {}; reload ns median {:.0} (min {:.0}), spill+reload ns median {:.0}; decisions and final state identical to the all-hot twins: {identical}",
            100 * max_hot / pool_keys, cold_b as f64 / cold_n.max(1) as f64, per_brain_pool, GIB / per_brain_pool, pool_ns / hot_ns, st.spills, st.reloads, median(&lat_reload), minf(&lat_reload), median(&lat_spill)
        );

        let gain = |i: usize| per[0]["live_bytes_per_brain"].as_f64().unwrap() / per[i].get("live_bytes_per_brain").and_then(|x| x.as_f64()).unwrap_or(f64::NAN);
        headline.push(format!(
            "{label:24} agents/GiB: ref {:.0} | R3 private {} | R4 private {:.0} | R4 shared {:.0} | R4 u12/q12 shared {:.0} | R4 rowcap50 shared {:.0} | cold blobs {:.0} | pool(5% hot) {:.0};  R4/R3 ns x{}; state B (R3 json/R4 json/bin/state) {:.0}/{:.0}/{:.0}/{:.0}",
            per[0]["agents_per_gib"].as_f64().unwrap(),
            per[1].get("agents_per_gib").and_then(|x| x.as_f64()).map(|x| format!("{x:.0}")).unwrap_or("n/a".into()),
            per[2]["agents_per_gib"].as_f64().unwrap(), per[3]["agents_per_gib"].as_f64().unwrap(), per[4]["agents_per_gib"].as_f64().unwrap(), per[5]["agents_per_gib"].as_f64().unwrap(), per[7]["agents_per_gib"].as_f64().unwrap(), GIB / per_brain_pool,
            r4_vs_r3.map(|v| format!("{:.2}/{:.2}", v.0, v.1)).unwrap_or("n/a".into()),
            j_old as f64 / sm, j_new as f64 / sm, b_new_sum as f64 / sm, b_state as f64 / sm
        ));
        let _ = gain(2);
        world_json.push(json!({
            "world": label, "agents": n, "tasks": tasks.len(), "bg_cpu_pct": bg, "natural_rows": nat_rows,
            "latency_ns": lat_json, "r4_vs_r3_ratio_fresh_warm": r4_vs_r3.map(|v| vec![v.0, v.1]), "latency_within_10pct_of_r3": lat_ok,
            "one_brain": {"live_ref": m_ref, "live_r3_compact": m_g3, "live_r4_compact": m_new, "json_r3": g3_json, "json_r4": json_new.len(), "binary": bin.len(), "binary_state_only": state.len(), "binary_u12q12": bf_bin, "state_only_u12q12": bf_state,
                "exact_roundtrip_binary": exact_bin, "exact_roundtrip_state_only": exact_state,
                "ns": {"to_json": ns_to_json, "from_json": ns_from_json, "to_bytes": ns_to_bin, "from_bytes": ns_from_bin, "state_bytes": ns_state, "from_state_bytes": ns_unstate}},
            "agents_per_gib": per,
            "serialised_per_brain_mean": {"sample": sample, "r3_json": j_old as f64 / sm, "r4_json": j_new as f64 / sm, "r4_binary": b_new_sum as f64 / sm, "r4_state_only": b_state as f64 / sm},
            "pool": {"keys": pool_keys, "max_hot": max_hot, "cold_blobs": cold_n, "cold_bytes": cold_b, "live_bytes_per_brain_incl_shared": per_brain_pool, "agents_per_gib": GIB / per_brain_pool,
                "churn_ns_per_decision": pool_ns, "all_hot_ns_per_decision": hot_ns, "spills": st.spills, "reloads": st.reloads, "reload_ns_median": median(&lat_reload), "reload_ns_min": minf(&lat_reload), "spill_ns_median": median(&lat_spill), "pooled_state_equals_all_hot": identical},
        }));
        let _ = EvictPolicy::Value;
    }

    println!("\n=== HEADLINE (seed-31 world per row for latency/one-brain; K brains for agents/GiB; counting-allocator live bytes) ===");
    for h in &headline {
        println!("{h}");
    }
    let j = json!({"bin": "h5_mem", "criteria": {"latency_vs_r3_margin": 1.10, "reps": reps, "quick": quick}, "worlds": world_json, "runtime_s": t0.elapsed().as_secs_f64(),
        "notes": ["ns is min/median of interleaved repetitions on a shared box (+-30% noise); only ratios matter", "R3 compact = verbatim copy of the old src/compact.rs (<=64 agents)", "commit-charge column is unreliable for small K"]});
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out} ({:.1}s)", t0.elapsed().as_secs_f64());
}
