//! G15 / W16 (MAX-VECTORS): pure retrieval-infrastructure limits of the vector index. No model-IQ claims.
//!
//! Modes (each writes results/g15_<mode>*.json; a console markdown summary is printed too):
//!   sweep    hnsw_rs M x ef_construction x ef_search grid, recall@10 vs latency Pareto (synthetic + real MiniLM/BGE)
//!   confirm  rebuild the Pareto-picked HNSW configs 3x (parallel insert is nondeterministic) with lib / AVX2 kernel
//!   quant    turbovec 2/3/4-bit (+exact rerank), int8, PQ-lite, f16/bf16/block-fp16 flat stores: memory vs recall
//!   scale    N = 1e4,1e5,1e6 (RAM-guarded) build time, bytes/vector, recall, p50/p99
//!   conc     1/2/4 reader threads on ONE shared index; HNSW under a concurrent insert thread
//!   flat     flat-scan vs HNSW crossover N
//!
//! Usage: g15_vectors <mode> [--out results] [--reps 3] [--quick] [--data synth384:30000,mini,bge] ...
//! Run with RAYON_NUM_THREADS=2. Single-query latency is measured serially on one thread (turbovec inside a 1-thread
//! rayon pool, like memory_bench), recall is vs exact f32 brute force on the same normalised vectors.

#[path = "g15_common/mod.rs"]
mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use common::*;
use hnsw_rs::prelude::{Distance, Hnsw};
use neural_mesh::memory::CosineDist;
use serde_json::{json, Value};
use turbovec::TurboQuantIndex;

#[global_allocator]
static GLOBAL: Counting = Counting;

const K: usize = 10;

// ------------------------------------------------------------------ args
struct Args(Vec<String>);
impl Args {
    fn get(&self, k: &str) -> Option<String> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1)).cloned()
    }
    fn has(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
    fn usize(&self, k: &str, d: usize) -> usize {
        self.get(k).map(|v| v.parse().unwrap()).unwrap_or(d)
    }
    fn list(&self, k: &str, d: &str) -> Vec<String> {
        self.get(k).unwrap_or(d.to_string()).split(',').map(|s| s.to_string()).collect()
    }
    fn nums(&self, k: &str, d: &str) -> Vec<usize> {
        self.list(k, d).iter().map(|s| s.parse().unwrap()).collect()
    }
}

fn write_json(out: &str, name: &str, v: &Value) {
    std::fs::create_dir_all(out).ok();
    let p = format!("{out}/{name}");
    std::fs::write(&p, serde_json::to_string_pretty(v).unwrap()).unwrap();
    eprintln!("wrote {p}");
}
fn machine(tag: &str) -> Value {
    let (ws, peak) = rss_mb();
    json!({"tag": tag, "free_ram_gb": free_gb(), "calib_ms_fixed_scan": calib_ms(), "proc_ws_mb": ws, "proc_peak_ws_mb": peak,
           "avx2": has_avx2(), "f16c": has_f16c(), "rayon_threads": rayon::current_num_threads()})
}
fn load(spec: &str, nq: usize, _out: &str) -> Ds {
    if let Some(n) = spec.strip_prefix("synth384:") {
        synth_ds(384, n.parse().unwrap(), nq)
    } else if let Some(n) = spec.strip_prefix("synth768:") {
        synth_ds(768, n.parse().unwrap(), nq)
    } else if spec == "mini" || spec == "bge" {
        real_ds(spec, nq, &std::env::var("G15_FIXTURES").unwrap_or("results".into()))
    } else {
        panic!("unknown dataset {spec}")
    }
}

// ------------------------------------------------------------------ HNSW wrapper (library kernel vs AVX2 kernel)
#[derive(Default, Clone, Copy)]
struct Avx2Dist;
impl Distance<f32> for Avx2Dist {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        (1.0 - dot(a, b)).max(0.0)
    }
}
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kern {
    Lib,
    Avx,
}
impl Kern {
    fn name(self) -> &'static str {
        match self {
            Kern::Lib => "lib(sse2 autovec, as src/memory.rs)",
            Kern::Avx => "avx2+fma runtime dispatch",
        }
    }
}
enum Hn {
    Lib(Hnsw<'static, f32, CosineDist>),
    Avx(Hnsw<'static, f32, Avx2Dist>),
}
impl Hn {
    fn new(kern: Kern, m: usize, efc: usize, maxel: usize) -> Self {
        match kern {
            Kern::Lib => Hn::Lib(Hnsw::new(m, maxel.max(1), 16, efc, CosineDist)),
            Kern::Avx => Hn::Avx(Hnsw::new(m, maxel.max(1), 16, efc, Avx2Dist)),
        }
    }
    fn insert_batch(&self, rows: &[f32], dim: usize, id0: usize) {
        let items: Vec<(&[f32], usize)> = rows.chunks_exact(dim).enumerate().map(|(i, v)| (v, id0 + i)).collect();
        match self {
            Hn::Lib(h) => h.parallel_insert_slice(&items),
            Hn::Avx(h) => h.parallel_insert_slice(&items),
        }
    }
    fn insert_one(&self, v: &[f32], id: usize) {
        match self {
            Hn::Lib(h) => h.insert_slice((v, id)),
            Hn::Avx(h) => h.insert_slice((v, id)),
        }
    }
    fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<u32> {
        let r = match self {
            Hn::Lib(h) => h.search(q, k, ef.max(k)),
            Hn::Avx(h) => h.search(q, k, ef.max(k)),
        };
        r.into_iter().map(|n| n.d_id as u32).collect()
    }
}
struct BuiltH {
    h: Hn,
    build_s: f64,
    heap_bytes: usize,
    rss_delta_mb: f64,
}
fn build_hnsw(base: &[f32], dim: usize, m: usize, efc: usize, kern: Kern, maxel: usize) -> BuiltH {
    let l0 = live_bytes();
    let (r0, _) = rss_mb();
    let t = Instant::now();
    let h = Hn::new(kern, m, efc, maxel);
    let n = base.len() / dim;
    let mut off = 0;
    while off < n {
        let c = (n - off).min(2048);
        h.insert_batch(&base[off * dim..(off + c) * dim], dim, off);
        off += c;
    }
    let build_s = t.elapsed().as_secs_f64();
    BuiltH {
        h,
        build_s,
        heap_bytes: live_bytes().saturating_sub(l0),
        rss_delta_mb: rss_mb().0 - r0,
    }
}

// ------------------------------------------------------------------ turbovec helpers
fn tv_build(base: &[f32], dim: usize, bits: usize) -> TurboQuantIndex {
    let mut ix = TurboQuantIndex::new(dim, bits).expect("turbovec dim/bits");
    for ch in base.chunks(dim * 2048) {
        ix.add(ch);
    }
    ix.prepare();
    ix
}
fn tv_search(ix: &TurboQuantIndex, base: &[f32], dim: usize, q: &[f32], k: usize, rr: Option<usize>) -> Vec<u32> {
    let fetch = k * rr.unwrap_or(1);
    let r = ix.search(q, fetch);
    let ids: Vec<u32> = r.indices_for_query(0).iter().map(|&i| i as u32).collect();
    match rr {
        None => ids.into_iter().take(k).collect(),
        Some(_) => {
            let ok: Vec<u32> = ids.into_iter().filter(|&i| (i as usize + 1) * dim <= base.len()).collect();
            rerank_f32(base, dim, q, &ok, k)
        }
    }
}
fn pool1() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap()
}

fn md_row(cols: &[String]) -> String {
    format!("| {} |", cols.join(" | "))
}
fn f1(x: f64) -> String {
    format!("{x:.1}")
}
fn f3(x: f64) -> String {
    format!("{x:.3}")
}

// ================================================================== sweep
fn mode_sweep(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let quick = a.has("--quick");
    let reps = a.usize("--reps", 3);
    let nq = a.usize("--nq", if quick { 100 } else { 300 });
    let ms = a.nums("--m", if quick { "8,16" } else { "8,16,32" });
    let efcs = a.nums("--efc", if quick { "100,200" } else { "100,200,400" });
    let efs = a.nums("--ef", "10,16,24,32,48,64,96,128,192,256,384,512");
    let kern = if a.get("--kernel").as_deref() == Some("avx2") { Kern::Avx } else { Kern::Lib };
    for spec in a.list("--data", if quick { "synth384:5000" } else { "synth384:30000,mini,bge" }) {
        let ds = load(&spec, nq, &out);
        let gt = ground_truth(&ds.base, ds.dim, &ds.queries, K);
        let n = ds.n();
        eprintln!("[sweep] {} n={n} dim={} kernel={:?} free={:.1}GB", ds.name, ds.dim, kern, free_gb());
        let mut cfgs = vec![];
        let mut pts: Vec<(f64, f64, Value)> = vec![];
        let mach0 = machine("start");
        for &m in &ms {
            for &efc in &efcs {
                let b = build_hnsw(&ds.base, ds.dim, m, efc, kern, n);
                eprintln!("  M={m} efc={efc} build {:.1}s heap {:.0} B/vec", b.build_s, b.heap_bytes as f64 / n as f64);
                let mut rows = vec![];
                for &ef in &efs {
                    let r = bench_search(|q| b.h.search(q, K, ef), &ds, &gt, K, reps);
                    pts.push((r.recall, mean(&r.p99), json!({"m": m, "efc": efc, "ef": ef})));
                    rows.push(json!({"ef": ef, "bench": r.json()}));
                }
                cfgs.push(json!({"m": m, "efc": efc, "build_s": b.build_s, "heap_bytes_per_vec": b.heap_bytes as f64 / n as f64,
                    "rss_delta_mb": b.rss_delta_mb, "rows": rows}));
            }
        }
        let front = pareto(&pts);
        let pf: Vec<Value> = front.iter().map(|(r, p, c)| json!({"recall": r, "p99_us": p, "cfg": c})).collect();
        let v = json!({"mode": "sweep", "dataset": ds.name, "note": ds.note, "n": n, "dim": ds.dim, "nq": ds.nq(), "k": K, "reps": reps,
            "kernel": kern.name(), "machine_start": mach0, "machine_end": machine("end"), "configs": cfgs, "pareto": pf});
        let tag = format!("{}{}", if kern == Kern::Avx { "_avx2" } else { "" }, a.get("--tag").map(|t| format!("_{t}")).unwrap_or_default());
        write_json(&out, &format!("g15_hnsw_sweep_{}{tag}.json", ds.name), &v);
        println!("\n#### HNSW Pareto {} (n={n}, dim={}, kernel={})", ds.name, ds.dim, kern.name());
        println!("| recall@10 | p99 us | M | efc | ef |\n|---|---|---|---|---|");
        for (r, p, c) in &front {
            println!("{}", md_row(&[f3(*r), f1(*p), c["m"].to_string(), c["efc"].to_string(), c["ef"].to_string()]));
        }
    }
}

// ================================================================== confirm
fn mode_confirm(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let reps = a.usize("--reps", 3);
    let nq = a.usize("--nq", 300);
    let builds = a.usize("--builds", 3);
    let kerns: Vec<Kern> = a
        .list("--kernels", "lib,avx2")
        .iter()
        .map(|s| if s == "avx2" { Kern::Avx } else { Kern::Lib })
        .collect();
    let targets = [0.90, 0.95, 0.99];
    for spec in a.list("--data", "synth384:30000,mini,bge") {
        let ds = load(&spec, nq, &out);
        let mut sweep: Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{out}/g15_hnsw_sweep_{}.json", ds.name)).expect("run sweep first")).unwrap();
        for t in a.get("--tags").map(|x| x.split(',').map(|s| s.to_string()).collect::<Vec<_>>()).unwrap_or_default() {
            let extra: Value = serde_json::from_str(&std::fs::read_to_string(format!("{out}/g15_hnsw_sweep_{}_{t}.json", ds.name)).expect("extra sweep")).unwrap();
            let add = extra["configs"].as_array().unwrap().clone();
            sweep["configs"].as_array_mut().unwrap().extend(add);
        }
        let gt = ground_truth(&ds.base, ds.dim, &ds.queries, K);
        let n = ds.n();
        // lowest-mean-p99 config (any M/efc/ef) reaching each recall target, from the lib-kernel sweep
        let mut picks: Vec<(f64, Value)> = vec![];
        for &t in &targets {
            let mut best: Option<(f64, Value)> = None;
            for c in sweep["configs"].as_array().unwrap() {
                for r in c["rows"].as_array().unwrap() {
                    let rec = r["bench"]["recall_at_10"].as_f64().unwrap();
                    let p99 = r["bench"]["p99_us"]["mean"].as_f64().unwrap();
                    if rec >= t && best.as_ref().map_or(true, |b| p99 < b.0) {
                        best = Some((p99, json!({"m": c["m"], "efc": c["efc"], "ef": r["ef"]})));
                    }
                }
            }
            if let Some((_, cfg)) = best {
                picks.push((t, cfg));
            }
        }
        let mut res = vec![];
        for (t, cfg) in &picks {
            let (m, efc, ef) = (cfg["m"].as_u64().unwrap() as usize, cfg["efc"].as_u64().unwrap() as usize, cfg["ef"].as_u64().unwrap() as usize);
            for &kern in &kerns {
                let (mut rec, mut p50, mut p99, mut bt, mut bpv) = (vec![], vec![], vec![], vec![], vec![]);
                for _ in 0..builds {
                    let b = build_hnsw(&ds.base, ds.dim, m, efc, kern, n);
                    let r = bench_search(|q| b.h.search(q, K, ef), &ds, &gt, K, reps);
                    rec.push(r.recall);
                    p50.push(mean(&r.p50));
                    p99.push(mean(&r.p99));
                    bt.push(b.build_s);
                    bpv.push(b.heap_bytes as f64 / n as f64);
                }
                eprintln!("[confirm] {} target {t} M{m} efc{efc} ef{ef} {:?}: recall {:.3} p99 {:.0}us", ds.name, kern, mean(&rec), mean(&p99));
                res.push(json!({"target_recall": t, "m": m, "efc": efc, "ef": ef, "kernel": kern.name(), "builds": builds,
                    "recall_per_build": spread(&rec), "p50_us_per_build": spread(&p50), "p99_us_per_build": spread(&p99),
                    "build_s": spread(&bt), "heap_bytes_per_vec": spread(&bpv)}));
            }
        }
        let ctag = a.get("--tags").map(|t| format!("_{}", t.replace(',', "_"))).unwrap_or_default();
        write_json(&out, &format!("g15_hnsw_confirm_{}{ctag}.json", ds.name),
            &json!({"mode":"confirm","dataset":ds.name,"note":ds.note,"n":n,"dim":ds.dim,"nq":ds.nq(),"reps_per_build":reps,"results":res,"machine":machine("end")}));
    }
}

// ================================================================== quant
struct Row {
    name: String,
    n: usize,
    nominal: usize,
    heap: usize,
    needs_f32: bool,
    build_s: Vec<f64>,
    bench: Bench,
}
impl Row {
    fn json(&self) -> Value {
        let n = self.n as f64;
        json!({"name": self.name, "n": self.n, "nominal_bytes_per_vec": self.nominal as f64 / n, "measured_heap_bytes_per_vec": self.heap as f64 / n,
            "needs_f32_copy_in_ram_for_rerank": self.needs_f32, "build_s": spread(&self.build_s), "bench": self.bench.json()})
    }
    fn md(&self, dim: usize) -> String {
        let n = self.n as f64;
        let f32b = if self.needs_f32 { dim as f64 * 4.0 } else { 0.0 };
        let tot = self.heap as f64 / n + f32b;
        md_row(&[self.name.clone(), f1(self.nominal as f64 / n), f1(self.heap as f64 / n), f1(tot),
            f1(1e9 / tot / 1e6), f3(self.bench.recall), f1(mean(&self.bench.p50)), f1(mean(&self.bench.p99)), f1(mean(&self.build_s))])
    }
}
const MD_HDR: &str = "| config | nominal B/vec | heap B/vec (index) | RAM B/vec incl. f32 for rerank | M vectors/GB (RAM) | recall@10 | p50 us | p99 us | build s |\n|---|---|---|---|---|---|---|---|---|";

/// Build `reps` times (keeping the last), returning (value, times, heap delta of last build).
fn timed_build<T>(reps: usize, mut f: impl FnMut() -> T) -> (T, Vec<f64>, usize) {
    let mut times = vec![];
    let mut last: Option<T> = None;
    let mut heap = 0;
    for _ in 0..reps {
        drop(last.take());
        let l0 = live_bytes();
        let t = Instant::now();
        let v = f();
        times.push(t.elapsed().as_secs_f64());
        heap = live_bytes().saturating_sub(l0);
        last = Some(v);
    }
    (last.unwrap(), times, heap)
}

/// All flat-store / turbovec / PQ configs on one dataset. `rrs`: rerank factors.
fn quant_configs(ds: &Ds, gt: &[Vec<u32>], reps: usize, build_reps: usize, include_pq: bool, include_f16: bool) -> Vec<Row> {
    let (dim, n) = (ds.dim, ds.n());
    let mut rows = vec![];
    let p1 = pool1();
    // exact f32 flat, AVX2 and library kernels
    for (lib, nm) in [(false, "flat-f32 exact (avx2)"), (true, "flat-f32 exact (lib sse2 kernel)")] {
        let b = bench_search(|q| flat_topk(&ds.base, dim, q, K, lib), ds, gt, K, reps);
        rows.push(Row { name: nm.into(), n, nominal: n * dim * 4, heap: 0, needs_f32: false, build_s: vec![0.0], bench: b });
        if let Some(r) = rows.last_mut() {
            r.heap = n * dim * 4; // the base itself is the index
        }
    }
    // low-precision flat stores
    let mut precs = vec![Prec::Int8];
    if include_f16 {
        precs.extend([Prec::F16, Prec::Bf16, Prec::BlockFp16]);
    }
    for prec in precs {
        let (st, bt, heap) = timed_build(build_reps, || LowStore::build(prec, &ds.base, dim));
        let mut tmp = vec![];
        let b = bench_search(|q| st.topk(q, K, None, &mut tmp), ds, gt, K, reps);
        rows.push(Row { name: format!("flat-{} scan", prec.name()), n, nominal: st.nominal_bytes(), heap, needs_f32: false, build_s: bt, bench: b });
        if prec == Prec::Int8 || prec == Prec::BlockFp16 {
            for f in [4usize, 10] {
                let mut tmp = vec![];
                let b = bench_search(|q| st.topk(q, K, Some((&ds.base, f)), &mut tmp), ds, gt, K, reps);
                rows.push(Row { name: format!("flat-{} scan + f32 rerank {f}x", prec.name()), n, nominal: st.nominal_bytes(), heap, needs_f32: true, build_s: vec![0.0], bench: b });
            }
        }
    }
    // PQ-lite
    if include_pq {
        for sub in [8usize, 4] {
            let (pq, bt, heap) = timed_build(build_reps.min(1), || Pq::build(&ds.base, dim, sub, SEED));
            let b = bench_search(|q| pq.topk(q, K, None), ds, gt, K, reps);
            rows.push(Row { name: format!("pq-lite m={} (sub {sub}d, 256 ctr, ADC)", dim / sub), n, nominal: pq.nominal_bytes(), heap, needs_f32: false, build_s: bt.clone(), bench: b });
            for f in [10usize, 30] {
                let b = bench_search(|q| pq.topk(q, K, Some((&ds.base, f))), ds, gt, K, reps);
                rows.push(Row { name: format!("pq-lite m={} + f32 rerank {f}x", dim / sub), n, nominal: pq.nominal_bytes(), heap, needs_f32: true, build_s: vec![0.0], bench: b });
            }
        }
    }
    // turbovec
    for bits in [2usize, 3, 4] {
        let (ix, bt, heap) = timed_build(build_reps, || tv_build(&ds.base, dim, bits));
        let nominal = n * (dim * bits).div_ceil(8) + n * 4;
        let b = p1.install(|| bench_search(|q| tv_search(&ix, &ds.base, dim, q, K, None), ds, gt, K, reps));
        rows.push(Row { name: format!("turbovec-{bits}bit (1 thread)"), n, nominal, heap, needs_f32: false, build_s: bt, bench: b });
        for f in [2usize, 4, 8, 16] {
            let b = p1.install(|| bench_search(|q| tv_search(&ix, &ds.base, dim, q, K, Some(f)), ds, gt, K, reps));
            rows.push(Row { name: format!("turbovec-{bits}bit + f32 rerank {f}x (1 thread)"), n, nominal, heap, needs_f32: true, build_s: vec![0.0], bench: b });
        }
        if bits == 4 && n >= 32_000 {
            let b = bench_search(|q| tv_search(&ix, &ds.base, dim, q, K, Some(4)), ds, gt, K, reps);
            rows.push(Row { name: "turbovec-4bit + f32 rerank 4x (global pool, 2 threads)".into(), n, nominal, heap, needs_f32: true, build_s: vec![0.0], bench: b });
        }
    }
    rows
}

fn mode_quant(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let quick = a.has("--quick");
    let reps = a.usize("--reps", 3);
    let nq = a.usize("--nq", if quick { 100 } else { 300 });
    for spec in a.list("--data", if quick { "synth384:5000" } else { "synth384:100000,mini,bge" }) {
        let ds = load(&spec, nq, &out);
        let gt = ground_truth(&ds.base, ds.dim, &ds.queries, K);
        eprintln!("[quant] {} n={} dim={}", ds.name, ds.n(), ds.dim);
        let rows = quant_configs(&ds, &gt, reps, 3, true, true);
        println!("\n#### quantisation {} (n={}, dim={})\n{MD_HDR}", ds.name, ds.n(), ds.dim);
        for r in &rows {
            println!("{}", r.md(ds.dim));
        }
        write_json(&out, &format!("g15_quant_{}.json", ds.name),
            &json!({"mode":"quant","dataset":ds.name,"note":ds.note,"n":ds.n(),"dim":ds.dim,"nq":ds.nq(),"reps":reps,
              "rows": rows.iter().map(|r| r.json()).collect::<Vec<_>>(), "machine": machine("end")}));
    }
}

// ================================================================== scale
fn mode_scale(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let reps = a.usize("--reps", 3);
    let nq = a.usize("--nq", 200);
    let sizes = a.nums("--sizes", "10000,100000,1000000");
    let hnsw_max = a.usize("--hnsw-max", 100_000);
    let hn_builds = a.usize("--hnsw-build-reps", 1);
    let (m, efc) = (a.usize("--m", 16), a.usize("--efc", 200));
    let mut all = vec![];
    let mut bytes_per_vec_hnsw: Option<f64> = None;
    for &n in &sizes {
        let dim = 384usize;
        let need_gb = n as f64 * dim as f64 * 4.0 / 1e9 * 1.5 + 0.5;
        let free = free_gb();
        if n >= 500_000 && free < 4.0 {
            eprintln!("[scale] SKIP N={n}: free RAM {free:.2} GB < 4 GB (rule)");
            all.push(json!({"n": n, "skipped": format!("free RAM {free:.2} GB < 4 GB at start")}));
            continue;
        }
        if free < 2.0 {
            eprintln!("[scale] STOP: free RAM {free:.2} GB < 2 GB");
            all.push(json!({"n": n, "skipped": format!("free RAM {free:.2} GB < 2 GB")}));
            break;
        }
        eprintln!("[scale] N={n}: generating (need ~{need_gb:.1} GB), free {free:.1} GB");
        let t = Instant::now();
        let ds = synth_ds(dim, n, nq);
        eprintln!("  gen {:.1}s; computing ground truth", t.elapsed().as_secs_f64());
        let tg = Instant::now();
        let gt = ground_truth(&ds.base, dim, &ds.queries, K);
        eprintln!("  gt {:.1}s; free {:.1} GB", tg.elapsed().as_secs_f64(), free_gb());
        let big = n >= 500_000;
        let mach = machine("start");
        let rss_before = rss_mb().0;
        let mut rows: Vec<Row> = quant_configs_scale(&ds, &gt, reps, if big { 1 } else { 3 }, n <= 100_000);
        let mut hrows = vec![];
        // HNSW
        let proj_gb = bytes_per_vec_hnsw.map(|b| b * n as f64 / 1e9 + n as f64 * dim as f64 * 4.0 / 1e9);
        let hn_ok = n <= hnsw_max && proj_gb.map_or(true, |g| g + 1.5 < free_gb());
        if hn_ok {
            let (mut bts, mut last) = (vec![], None);
            for _ in 0..hn_builds {
                drop(last.take());
                let b = build_hnsw(&ds.base, dim, m, efc, Kern::Lib, n);
                eprintln!("  hnsw M{m} efc{efc} build {:.1}s heap {:.0} B/vec", b.build_s, b.heap_bytes as f64 / n as f64);
                bts.push(b.build_s);
                last = Some(b);
            }
            let b = last.unwrap();
            bytes_per_vec_hnsw = Some(b.heap_bytes as f64 / n as f64);
            for ef in [32usize, 64, 128, 256] {
                let r = bench_search(|q| b.h.search(q, K, ef), &ds, &gt, K, reps);
                hrows.push(json!({"name": format!("hnsw M{m} efc{efc} ef{ef} (lib kernel)"), "n": n, "build_s": spread(&bts),
                    "heap_bytes_per_vec": b.heap_bytes as f64 / n as f64, "rss_delta_mb": b.rss_delta_mb, "bench": r.json()}));
            }
            // AVX2 kernel variant at one build
            drop(b);
            let b2 = build_hnsw(&ds.base, dim, m, efc, Kern::Avx, n);
            for ef in [64usize, 128] {
                let r = bench_search(|q| b2.h.search(q, K, ef), &ds, &gt, K, reps);
                hrows.push(json!({"name": format!("hnsw M{m} efc{efc} ef{ef} (avx2 kernel)"), "n": n, "build_s": spread(&[b2.build_s]),
                    "heap_bytes_per_vec": b2.heap_bytes as f64 / n as f64, "rss_delta_mb": b2.rss_delta_mb, "bench": r.json()}));
            }
        } else {
            let why = if n > hnsw_max {
                format!("N={n} > --hnsw-max {hnsw_max}")
            } else {
                format!("projected {:.2} GB + 1.5 GB > free {:.2} GB", proj_gb.unwrap_or(0.0), free_gb())
            };
            eprintln!("  HNSW skipped: {why}");
            hrows.push(json!({"name": "hnsw", "n": n, "skipped": why, "projected_total_gb": proj_gb}));
        }
        let (ws, peak) = rss_mb();
        println!("\n#### scale N={n} dim={dim} (peak WS {peak:.0} MB, WS before {rss_before:.0} MB)\n{MD_HDR}");
        for r in &rows {
            println!("{}", r.md(dim));
        }
        for h in &hrows {
            if h.get("skipped").is_some() {
                println!("hnsw skipped: {}", h["skipped"]);
            } else {
                let bpv = h["heap_bytes_per_vec"].as_f64().unwrap() + 0.0;
                println!("| {} | - | {:.1} | {:.1} | {:.2} | {:.3} | {:.1} | {:.1} | {:.1} |", h["name"].as_str().unwrap(), bpv, bpv,
                    1e3 / bpv, h["bench"]["recall_at_10"].as_f64().unwrap(), h["bench"]["p50_us"]["mean"].as_f64().unwrap(),
                    h["bench"]["p99_us"]["mean"].as_f64().unwrap(), h["build_s"]["mean"].as_f64().unwrap());
            }
        }
        let rj: Vec<Value> = rows.drain(..).map(|r| r.json()).collect();
        all.push(json!({"n": n, "dim": dim, "nq": nq, "peak_ws_mb": peak, "ws_mb": ws, "free_gb_after": free_gb(), "machine_start": mach,
            "rows": rj, "hnsw": hrows}));
        drop(ds);
    }
    let stag = a.get("--tag").map(|t| format!("_{t}")).unwrap_or_default();
    write_json(&out, &format!("g15_scale{stag}.json"), &json!({"mode":"scale","k":K,"reps":reps,"hnsw_params":{"m":m,"efc":efc},"sizes":all,"machine":machine("end")}));
}

/// Subset of configs for the scaling study (keeps total time bounded).
fn quant_configs_scale(ds: &Ds, gt: &[Vec<u32>], reps: usize, build_reps: usize, pq: bool) -> Vec<Row> {
    let (dim, n) = (ds.dim, ds.n());
    let mut rows = vec![];
    let p1 = pool1();
    for (lib, nm) in [(false, "flat-f32 exact (avx2)"), (true, "flat-f32 exact (lib sse2 kernel)")] {
        if lib && n > 200_000 {
            continue;
        }
        let b = bench_search(|q| flat_topk(&ds.base, dim, q, K, lib), ds, gt, K, reps);
        rows.push(Row { name: nm.into(), n, nominal: n * dim * 4, heap: n * dim * 4, needs_f32: false, build_s: vec![0.0], bench: b });
    }
    {
        let (st, bt, heap) = timed_build(build_reps, || LowStore::build(Prec::Int8, &ds.base, dim));
        let mut tmp = vec![];
        let b = bench_search(|q| st.topk(q, K, None, &mut tmp), ds, gt, K, reps);
        rows.push(Row { name: "flat-int8 scan".into(), n, nominal: st.nominal_bytes(), heap, needs_f32: false, build_s: bt, bench: b });
        let b = bench_search(|q| st.topk(q, K, Some((&ds.base, 10)), &mut tmp), ds, gt, K, reps);
        rows.push(Row { name: "flat-int8 scan + f32 rerank 10x".into(), n, nominal: st.nominal_bytes(), heap, needs_f32: true, build_s: vec![0.0], bench: b });
    }
    for bits in [2usize, 4] {
        let (ix, bt, heap) = timed_build(build_reps, || tv_build(&ds.base, dim, bits));
        let nominal = n * (dim * bits).div_ceil(8) + n * 4;
        let b = p1.install(|| bench_search(|q| tv_search(&ix, &ds.base, dim, q, K, None), ds, gt, K, reps));
        rows.push(Row { name: format!("turbovec-{bits}bit (1 thread)"), n, nominal, heap, needs_f32: false, build_s: bt, bench: b });
        for f in [4usize, 16] {
            let b = p1.install(|| bench_search(|q| tv_search(&ix, &ds.base, dim, q, K, Some(f)), ds, gt, K, reps));
            rows.push(Row { name: format!("turbovec-{bits}bit + f32 rerank {f}x (1 thread)"), n, nominal, heap, needs_f32: true, build_s: vec![0.0], bench: b });
        }
        if n >= 32_000 && bits == 4 {
            let b = bench_search(|q| tv_search(&ix, &ds.base, dim, q, K, Some(4)), ds, gt, K, reps);
            rows.push(Row { name: "turbovec-4bit + f32 rerank 4x (global pool, 2 threads)".into(), n, nominal, heap, needs_f32: true, build_s: vec![0.0], bench: b });
        }
    }
    if pq {
        let (pq, bt, heap) = timed_build(1, || Pq::build(&ds.base, dim, 8, SEED));
        let b = bench_search(|q| pq.topk(q, K, Some((&ds.base, 30))), ds, gt, K, reps);
        rows.push(Row { name: format!("pq-lite m={} + f32 rerank 30x", dim / 8), n, nominal: pq.nominal_bytes(), heap, needs_f32: true, build_s: bt, bench: b });
    }
    rows
}

// ================================================================== conc
fn run_readers<F: Fn(usize, &[f32]) -> usize + Sync>(threads: usize, dur: Duration, ds: &Ds, f: &F, writer: Option<&(dyn Fn(&AtomicBool) -> usize + Sync)>) -> Value {
    let stop = AtomicBool::new(false);
    let nq = ds.nq();
    let t0 = Instant::now();
    let (lats, wcount): (Vec<Vec<f64>>, usize) = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let stop = &stop;
                s.spawn(move || {
                    let mut lat = Vec::with_capacity(1 << 16);
                    let mut i = (t * 37) % nq;
                    while !stop.load(Ordering::Relaxed) {
                        let t1 = Instant::now();
                        std::hint::black_box(f(t, ds.q(i)));
                        lat.push(t1.elapsed().as_secs_f64() * 1e6);
                        i = (i + 1) % nq;
                    }
                    lat
                })
            })
            .collect();
        let wh = writer.map(|w| {
            let stop = &stop;
            s.spawn(move || w(stop))
        });
        std::thread::sleep(dur);
        stop.store(true, Ordering::Relaxed);
        let l: Vec<Vec<f64>> = hs.into_iter().map(|h| h.join().unwrap()).collect();
        let wc = wh.map_or(0, |h| h.join().unwrap());
        (l, wc)
    });
    let el = t0.elapsed().as_secs_f64();
    let total: usize = lats.iter().map(|l| l.len()).sum();
    let mut all: Vec<f64> = lats.into_iter().flatten().collect();
    json!({"threads": threads, "qps": total as f64 / el, "p50_us": pct(&mut all, 0.5), "p99_us": pct(&mut all, 0.99), "queries": total,
        "writer_ops": wcount, "writer_ops_per_s": wcount as f64 / el})
}

fn mode_conc(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let n = a.usize("--n", 100_000);
    let reps = a.usize("--reps", 3);
    let secs = a.usize("--secs", 4) as u64;
    let nq = 300;
    let (m, efc, ef) = (a.usize("--m", 16), a.usize("--efc", 200), a.usize("--ef", 128));
    let extra = 20_000;
    let full = synth_ds(384, n + extra, nq);
    let dim = full.dim;
    let ds = Ds { name: full.name.clone(), dim, base: full.base[..n * dim].to_vec(), queries: full.queries.clone(), note: full.note.clone() };
    let fresh: Arc<Vec<f32>> = Arc::new(full.base[n * dim..].to_vec());
    drop(full);
    let dur = Duration::from_secs(secs);
    let mut res = vec![];
    let mach = machine("start");

    // ---- HNSW, read-only + insert thread
    eprintln!("[conc] building HNSW n={n}");
    let hb = build_hnsw(&ds.base, dim, m, efc, Kern::Lib, n + extra);
    eprintln!("  build {:.1}s", hb.build_s);
    let h = Arc::new(hb.h);
    let gt = ground_truth(&ds.base, dim, &ds.queries, K);
    let r1 = bench_search(|q| h.search(q, K, ef), &ds, &gt, K, 1);
    eprintln!("  recall@10 {:.3} (ef {ef})", r1.recall);
    // Interleaved per rep (read-only 1,2,4 then with-writer 1,2) so machine-load drift hits both arms alike.
    let mut ro: Vec<Vec<Value>> = vec![vec![]; 3];
    let mut rw: Vec<Vec<Value>> = vec![vec![]; 2];
    for rep in 0..reps {
        for (ti, &thr) in [1usize, 2, 4].iter().enumerate() {
            let hh = h.clone();
            ro[ti].push(run_readers(thr, dur, &ds, &move |_t, q| hh.search(q, K, ef).len(), None));
        }
        for (ti, &thr) in [1usize, 2].iter().enumerate() {
            let hh = h.clone();
            let hw = h.clone();
            let fr = fresh.clone();
            let cursor = Arc::new(AtomicUsize::new(rep * 5000));
            let w = move |stop: &AtomicBool| -> usize {
                let mut c = 0;
                while !stop.load(Ordering::Relaxed) {
                    let i = cursor.fetch_add(1, Ordering::Relaxed) % (fr.len() / 384);
                    hw.insert_one(&fr[i * 384..(i + 1) * 384], n + 1_000_000 + c + rep * 100_000);
                    c += 1;
                }
                c
            };
            rw[ti].push(run_readers(thr, dur, &ds, &move |_t, q| hh.search(q, K, ef).len(), Some(&w)));
        }
    }
    for (ti, &thr) in [1usize, 2, 4].iter().enumerate() {
        res.push(json!({"index": format!("hnsw M{m} efc{efc} ef{ef}"), "scenario": "read-only, shared Arc, no lock", "threads": thr, "reps": ro[ti]}));
    }
    for (ti, &thr) in [1usize, 2].iter().enumerate() {
        res.push(json!({"index": format!("hnsw M{m} efc{efc} ef{ef}"), "scenario": "readers + 1 concurrent insert thread (insert_slice on &self, no external lock)", "threads": thr, "reps": rw[ti]}));
    }
    drop(h);

    // ---- turbovec 4-bit + rerank 4x behind RwLock
    eprintln!("[conc] turbovec");
    let ix = tv_build(&ds.base, dim, 4);
    let base_arc = Arc::new(ds.base.clone());
    let lock = Arc::new(RwLock::new(ix));
    let r2 = bench_search(|q| tv_search(&lock.read().unwrap(), &base_arc, dim, q, K, Some(4)), &ds, &gt, K, 1);
    eprintln!("  tv4+rr4 recall {:.3}", r2.recall);
    for &thr in &[1usize, 2, 4] {
        let mut reps_v = vec![];
        for _ in 0..reps {
            let (l, b) = (lock.clone(), base_arc.clone());
            // each reader pins its own 1-thread rayon pool so turbovec does not fan out into the shared global pool
            let pools: Vec<rayon::ThreadPool> = (0..thr).map(|_| pool1()).collect();
            reps_v.push(run_readers(thr, dur, &ds, &move |t, q| {
                pools[t].install(|| tv_search(&l.read().unwrap(), &b, dim, q, K, Some(4)).len())
            }, None));
        }
        res.push(json!({"index": "turbovec-4bit + f32 rerank 4x, RwLock, per-reader private 1-thread rayon pool", "scenario": "read-only", "threads": thr, "reps": reps_v}));
    }
    // 2 readers + writer adding 64 vectors every 50 ms under the write lock
    {
        let mut reps_v = vec![];
        for rep in 0..reps {
            let (l, b) = (lock.clone(), base_arc.clone());
            let lw = lock.clone();
            let fr = fresh.clone();
            let w = move |stop: &AtomicBool| -> usize {
                let mut c = 0;
                let mut cur = rep * 3000;
                while !stop.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(50));
                    let s = cur % (fr.len() / 384 - 64);
                    lw.write().unwrap().add(&fr[s * 384..(s + 64) * 384]);
                    cur += 64;
                    c += 64;
                }
                c
            };
            let pools: Vec<rayon::ThreadPool> = (0..2).map(|_| pool1()).collect();
            reps_v.push(run_readers(2, dur, &ds, &move |t, q| {
                pools[t].install(|| tv_search(&l.read().unwrap(), &b, dim, q, K, Some(4)).len())
            }, Some(&w)));
        }
        res.push(json!({"index": "turbovec-4bit + rerank 4x, RwLock", "scenario": "2 readers + writer adding 64 vecs/50ms (added vectors are NOT in the f32 rerank store: rerank skips ids >= n)", "threads": 2, "reps": reps_v}));
    }
    // ---- flat f32 AVX2 baseline (shared immutable slice)
    {
        let b = Arc::new(ds.base.clone());
        for &thr in &[1usize, 2, 4] {
            let mut reps_v = vec![];
            for _ in 0..reps {
                let b = b.clone();
                reps_v.push(run_readers(thr, dur, &ds, &move |_t, q| flat_topk(&b, dim, q, K, false).len(), None));
            }
            res.push(json!({"index": "flat-f32 avx2", "scenario": "read-only", "threads": thr, "reps": reps_v}));
        }
    }
    println!("\n#### concurrency (n={n}, dim={dim}, {secs}s windows x{reps})\n| index | scenario | threads | qps (mean of reps) [min-max] | p50 us | p99 us | writer ops/s |\n|---|---|---|---|---|---|---|");
    for r in &res {
        let reps_a = r["reps"].as_array().unwrap();
        let q: Vec<f64> = reps_a.iter().map(|x| x["qps"].as_f64().unwrap()).collect();
        let p50: Vec<f64> = reps_a.iter().map(|x| x["p50_us"].as_f64().unwrap()).collect();
        let p99: Vec<f64> = reps_a.iter().map(|x| x["p99_us"].as_f64().unwrap()).collect();
        let w: Vec<f64> = reps_a.iter().map(|x| x["writer_ops_per_s"].as_f64().unwrap()).collect();
        let (lo, hi) = minmax(&q);
        println!("| {} | {} | {} | {:.0} [{:.0}-{:.0}] | {:.0} | {:.0} | {:.0} |", r["index"].as_str().unwrap(), r["scenario"].as_str().unwrap(),
            r["threads"], mean(&q), lo, hi, mean(&p50), mean(&p99), mean(&w));
    }
    write_json(&out, "g15_conc.json", &json!({"mode":"conc","n":n,"dim":dim,"window_s":secs,"reps":reps,"hnsw_recall":r1.recall,"tv_rerank4_recall":r2.recall,
        "results":res,"machine_start":mach,"machine_end":machine("end")}));
}

// ================================================================== flat crossover
fn mode_flat(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let reps = a.usize("--reps", 3);
    let nq = a.usize("--nq", 300);
    let sizes = a.nums("--sizes", "1000,2000,5000,10000,20000,30000,50000,100000");
    let (m, efc) = (a.usize("--m", 16), a.usize("--efc", 200));
    let target = 0.95;
    let dim = 384;
    let nmax = *sizes.iter().max().unwrap();
    let full = synth_ds(dim, nmax, nq);
    let efs = [16usize, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512];
    let h = Hn::new(Kern::Lib, m, efc, nmax);
    let mut built = 0usize;
    let p1 = pool1();
    let mut rows = vec![];
    for &n in &sizes {
        // grow the incremental hnsw to n
        let t = Instant::now();
        while built < n {
            let c = (n - built).min(2048);
            h.insert_batch(&full.base[built * dim..(built + c) * dim], dim, built);
            built += c;
        }
        let inc_s = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let ds = Ds { name: format!("synth384-n{n}"), dim, base: full.base[..n * dim].to_vec(), queries: full.queries.clone(), note: String::new() };
        let gt = ground_truth(&ds.base, dim, &ds.queries, K);
        let flat = bench_search(|q| flat_topk(&ds.base, dim, q, K, false), &ds, &gt, K, reps);
        let flat_lib = bench_search(|q| flat_topk(&ds.base, dim, q, K, true), &ds, &gt, K, reps);
        let st = LowStore::build(Prec::Int8, &ds.base, dim);
        let mut tmp = vec![];
        let i8rr = bench_search(|q| st.topk(q, K, Some((&ds.base, 10)), &mut tmp), &ds, &gt, K, reps);
        let ix = tv_build(&ds.base, dim, 4);
        let tv = p1.install(|| bench_search(|q| tv_search(&ix, &ds.base, dim, q, K, Some(4)), &ds, &gt, K, reps));
        // smallest ef reaching the recall target
        let mut hn = None;
        for &ef in &efs {
            let r = bench_search(|q| h.search(q, K, ef), &ds, &gt, K, reps);
            if r.recall >= target {
                hn = Some((ef, r));
                break;
            }
        }
        // note: queries are searched against the incremental graph holding exactly the first n rows
        let hrow = match &hn {
            Some((ef, r)) => json!({"ef": ef, "bench": r.json()}),
            None => json!({"ef": null, "note": "recall target not reached with ef<=512"}),
        };
        eprintln!("[flat] n={n}: flat p50 {:.0}us hnsw {:?}", mean(&flat.p50), hn.as_ref().map(|(e, r)| (e, mean(&r.p50))));
        rows.push(json!({"n": n, "hnsw_incremental_build_s_this_step": inc_s, "flat_f32_avx2": flat.json(), "flat_f32_lib_kernel": flat_lib.json(),
            "int8_scan_rerank10x": i8rr.json(), "turbovec4_rerank4x_1thread": tv.json(),
            "hnsw_lib_min_ef_for_recall_0.95": hrow, "step_wall_s": t.elapsed().as_secs_f64()}));
    }
    // crossover: first N where hnsw p50 < flat-avx2 p50 at recall >= 0.95
    let mut cross = Value::Null;
    for r in &rows {
        if let (Some(hp), fp) = (r["hnsw_lib_min_ef_for_recall_0.95"]["bench"]["p50_us"]["mean"].as_f64(), r["flat_f32_avx2"]["p50_us"]["mean"].as_f64().unwrap()) {
            if hp < fp {
                cross = r["n"].clone();
                break;
            }
        }
    }
    println!("\n#### flat vs HNSW crossover (dim 384, synthetic, single thread, p50 us mean of reps)\n| N | flat f32 avx2 | flat f32 lib | int8+rr10 | tv4+rr4 | hnsw (min ef for r>=0.95) | hnsw ef | hnsw recall |\n|---|---|---|---|---|---|---|---|");
    for r in &rows {
        let g = |k: &str| r[k]["p50_us"]["mean"].as_f64().unwrap_or(f64::NAN);
        let hb = &r["hnsw_lib_min_ef_for_recall_0.95"];
        println!("| {} | {:.0} | {:.0} | {:.0} | {:.0} | {} | {} | {} |", r["n"], g("flat_f32_avx2"), g("flat_f32_lib_kernel"), g("int8_scan_rerank10x"),
            g("turbovec4_rerank4x_1thread"), hb["bench"]["p50_us"]["mean"].as_f64().map_or("-".into(), |x| format!("{x:.0}")), hb["ef"], hb["bench"]["recall_at_10"]);
    }
    println!("crossover (hnsw faster than flat-avx2 at recall>=0.95): {cross}");
    write_json(&out, "g15_flat_crossover.json", &json!({"mode":"flat","dim":dim,"hnsw":{"m":m,"efc":efc,"build":"incremental parallel_insert in 2048 batches, one graph grown through the N list"},
        "target_recall":target,"rows":rows,"crossover_n_hnsw_faster_than_flat_avx2":cross,"machine":machine("end")}));
}

// ================================================================== turbovec memory anatomy
fn mode_tvmem(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let n = a.usize("--n", 100_000);
    let ds = synth_ds(384, n, 50);
    let dim = 384;
    let mut res = vec![];
    for bits in [2usize, 4] {
        for chunk in [n, 50_000, 10_000, 1_000] {
            let l0 = live_bytes();
            let (r0, _) = rss_mb();
            let mut ix = TurboQuantIndex::new(dim, bits).unwrap();
            for ch in ds.base.chunks(dim * chunk) {
                ix.add(ch);
            }
            let after_add = live_bytes() - l0;
            ix.prepare();
            let after_prepare = live_bytes() - l0;
            let _ = ix.search(ds.q(0), 10);
            let after_search = live_bytes() - l0;
            let r1 = rss_mb().0 - r0;
            eprintln!("[tvmem] {bits}bit chunk {chunk}: add {:.0} prepare {:.0} search {:.0} B/vec (rss +{r1:.0} MB)", after_add as f64 / n as f64, after_prepare as f64 / n as f64, after_search as f64 / n as f64);
            res.push(json!({"bits": bits, "add_chunk": chunk, "nominal_bytes_per_vec": (dim * bits).div_ceil(8) + 4,
                "heap_after_add": after_add as f64 / n as f64, "heap_after_prepare": after_prepare as f64 / n as f64,
                "heap_after_first_search": after_search as f64 / n as f64, "rss_delta_mb": r1}));
        }
    }
    write_json(&out, "g15_tvmem.json", &json!({"mode":"tvmem","n":n,"dim":dim,"rows":res}));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a = Args(args.clone());
    let threads = std::env::var("RAYON_NUM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
    let mode = args.get(1).cloned().unwrap_or_default();
    eprintln!("g15_vectors mode={mode} rayon={} free={:.2}GB", rayon::current_num_threads(), free_gb());
    match mode.as_str() {
        "sweep" => mode_sweep(&a),
        "confirm" => mode_confirm(&a),
        "quant" => mode_quant(&a),
        "scale" => mode_scale(&a),
        "conc" => mode_conc(&a),
        "flat" => mode_flat(&a),
        "tvmem" => mode_tvmem(&a),
        _ => eprintln!("modes: sweep confirm quant scale conc flat"),
    }
}
