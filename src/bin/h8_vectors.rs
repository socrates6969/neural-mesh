//! H8 / R4-V (MAX-VECTORS round 2): re-run of the g15 vector-index study on the FIXED hnsw_rs (Cargo [patch]), plus
//! measured bytes/vector (counting allocator), file-backed (SSD) rerank, quantised-vector HNSW, RaBitQ-style multi-bit
//! codes, streaming scale runs and concurrency. Pure retrieval infrastructure; no model-IQ claim.
//!
//! Modes (JSON to results/h8_*.json, markdown tables on stdout):
//!   selftest  sanity checks (stream == synth_ds, custom dots, rabitq estimator, file store)
//!   sweep     HNSW M x efc x ef grid on the fixed crate (recall@10 vs brute force, p50/p99, heap B/vec, self-recall)
//!   confirm   rebuild picked configs 3x (lib / avx2 kernel), self-recall each build
//!   cross     flat vs HNSW crossover (several dims, two HNSW recipes, recall targets .95/.99)
//!   memory    measured bytes/vec vs recall vs latency for flat/int8/bfp16/turbovec/RaBitQ/HNSW(+quantised store) incl. file rerank
//!   scale     streaming N = 1e5 / 3e5 / 1e6 (RAM guarded)
//!   conc      1/2/4 readers on one shared index, with a writer
//! Run with RAYON_NUM_THREADS=2. Single-query latency is serial on one thread (turbovec/rabitq inside a 1-thread pool).

#[path = "h8_common/mod.rs"]
mod common;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use common::*;
use serde_json::{json, Value};
use turbovec::TurboQuantIndex;

#[global_allocator]
static GLOBAL: Counting = Counting;

const K: usize = 10;

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
/// Pause (do not swap) while machine-wide free RAM is below `min_gb`.
fn wait_ram(min_gb: f64, why: &str) {
    let mut waited = 0;
    while free_gb() < min_gb {
        if waited % 6 == 0 {
            eprintln!("[ram] free {:.2} GB < {min_gb:.1} GB, pausing before {why}", free_gb());
        }
        std::thread::sleep(Duration::from_secs(20));
        waited += 1;
        if waited > 270 {
            eprintln!("[ram] gave up waiting (90 min)");
            std::process::exit(3);
        }
    }
}
fn load(spec: &str, nq: usize) -> Ds {
    if let Some(n) = spec.strip_prefix("synth384:") {
        synth_ds(384, n.parse().unwrap(), nq)
    } else if let Some(n) = spec.strip_prefix("synth128:") {
        synth_ds(128, n.parse().unwrap(), nq)
    } else if spec == "mini" || spec == "bge" {
        real_ds(spec, nq, &std::env::var("G15_FIXTURES").unwrap_or("results".into()))
    } else {
        panic!("unknown dataset {spec}")
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
fn kern_store(s: &str) -> Store {
    match s {
        "avx2" => Store::F32Avx,
        "bfp16" => Store::Bfp16,
        "int8" => Store::Int8,
        _ => Store::F32Lib,
    }
}
fn empty_ds(name: &str, dim: usize, queries: Vec<f32>) -> Ds {
    Ds { name: name.into(), dim, base: vec![], queries, note: String::new() }
}

// ============================================================ selftest
fn mode_selftest() {
    // 1. stream == synth_ds
    let d = synth_ds(384, 30_000, 20);
    let (q, mut st) = synth_stream(384, 30_000, 20);
    let mut base = vec![];
    while let Some(c) = st.next(7_000) {
        base.extend(c);
    }
    println!("stream==synth_ds: queries {} base {} (chunks 7000)", q == d.queries, base == d.base);
    // gen speed
    let t = Instant::now();
    let (_q, mut st) = synth_stream(384, 200_000, 20);
    let mut n = 0;
    while let Some(c) = st.next(50_000) {
        n += c.len() / 384;
    }
    println!("gen speed: {n} rows in {:.2}s", t.elapsed().as_secs_f64());
    // 2. custom dots
    let a = &d.base[..384];
    let b = &d.base[384..768];
    let (ea, eb) = (encode_bfp(a), encode_bfp(b));
    let (ia, ib) = (encode_i8(a), encode_i8(b));
    println!("dot f32 {:.6} bfp {:.6} int8 {:.6}", dot(a, b), bfp_dot(&ea, &eb), i8_dot(&ia, &ib));
    println!("self-dot f32 {:.6} bfp {:.6} int8 {:.6}", dot(a, a), bfp_dot(&ea, &ea), i8_dot(&ia, &ia));
    // 3. rabitq estimator quality
    for bits in [1usize, 2, 3, 4, 6] {
        let mut rq = RaBitQ::new(384, bits, SEED ^ 0x5DEE_CE66_D1CE_4E5B);
        rq.add(&d.base[..2000 * 384]);
        let mut tmp = vec![];
        rq.scores(d.q(0), &mut tmp);
        let (mut se, mut sd) = (0f64, 0f64);
        let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for i in 0..2000 {
            let tr = dot(&d.base[i * 384..(i + 1) * 384], d.q(0)) as f64;
            let es = tmp[i] as f64;
            se += (es - tr).abs();
            sd += tr.abs();
            sx += tr;
            sy += es;
            sxx += tr * tr;
            syy += es * es;
            sxy += tr * es;
        }
        let nn = 2000f64;
        let corr = (sxy - sx * sy / nn) / ((sxx - sx * sx / nn) * (syy - sy * sy / nn)).sqrt();
        { let ex: Vec<f32> = (0..5).map(|i| rq.debug_exact(d.q(0), i).0).collect(); println!("  rows0-4 simd {:?} exact-float {:?} true {:?}", &tmp[..5], ex, (0..5).map(|i| dot(&d.base[i*384..(i+1)*384], d.q(0))).collect::<Vec<_>>()); }
        { println!("  debug_s (scalar S, simd S, sq): {:?}", rq.debug_s(d.q(0), 0)); }
        let slope = (sxy - sx * sy / nn) / (sxx - sx * sx / nn);
        println!("rabitq {bits}-bit: {} B/vec, mean |err| {:.4} (mean |true| {:.4}), corr {:.4} slope est-on-true {:.4} mean true {:.4} mean est {:.4}", rq.bytes_per_vec(), se / nn, sd / nn, corr, slope, sx / nn, sy / nn);
    }
    {
        let rq = RaBitQ::new(384, 4, SEED ^ 0x5DEE_CE66_D1CE_4E5B);
        let (a, b) = (&d.base[..384], d.q(0));
        let (ra, rb) = (rq.rotate(a), rq.rotate(b));
        { let rr = rq.rotate(d.q(0)); println!("q norm {:.4} max|q| {:.3} max|Rq| {:.3} ; base0 max {:.3} max|R b0| {:.3}", dot(d.q(0), d.q(0)), d.q(0).iter().fold(0f32, |m, &x| m.max(x.abs())), rr.iter().fold(0f32, |m, &x| m.max(x.abs())), a.iter().fold(0f32, |m, &x| m.max(x.abs())), ra.iter().fold(0f32, |m, &x| m.max(x.abs()))); }
        println!("rotation check: dot {:.6} rotated {:.6} norm {:.6}", dot(a, b), dot(&ra, &rb), dot(&ra, &ra));
    }
    // 4. file store
    let p = "C:/Users/mariu/nm-target-r8/h8_tmp_selftest.bin";
    std::fs::create_dir_all("C:/Users/mariu/nm-target-r8").ok();
    FileRows::write_slice(p, &d.base[..5000 * 384], 384);
    for direct in [false, true] {
        let f = FileRows::open(p, 384, 5000, direct);
        let mut r = vec![0f32; 384];
        let mut ok = true;
        for id in [0usize, 1, 17, 4999, 2500] {
            f.row_into(id, &mut r);
            ok &= r == d.base[id * 384..(id + 1) * 384];
        }
        let t = Instant::now();
        for i in 0..2000 {
            f.row_into((i * 7919) % 5000, &mut r);
        }
        println!("file store direct={direct}: rows match {ok}, {:.1} us/row", t.elapsed().as_secs_f64() * 1e6 / 2000.0);
    }
    std::fs::remove_file(p).ok();
    // 5. self recall sanity on small HNSW
    let b = build_ann(Store::F32Lib, &d.base[..5000 * 384], 384, 16, 200);
    let (q, m1, m10) = self_recall(&*b.h, &d.base[..5000 * 384], 384, 64, 5000);
    println!("HNSW 5k M16 efc200 self-recall: queried {q}, not-top1 {m1}, not-in-top10 {m10}; heap {} B/vec", b.heap_bytes / 5000);
    for s in [Store::Bfp16, Store::Int8] {
        let b = build_ann(s, &d.base[..5000 * 384], 384, 16, 200);
        let (q, m1, m10) = self_recall(&*b.h, &d.base[..5000 * 384], 384, 64, 5000);
        println!("HNSW 5k M16 efc200 {} self-recall: queried {q}, not-top1 {m1}, not-in-top10 {m10}; heap {} B/vec", s.short(), b.heap_bytes / 5000);
    }
}

// ============================================================ sweep
fn mode_sweep(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let reps = a.usize("--reps", 3);
    let nq = a.usize("--nq", 300);
    let ms = a.nums("--m", "16,24,32,48,64");
    let efcs = a.nums("--efc", "100,200,400");
    let efs = a.nums("--ef", "10,16,24,32,48,64,96,128,256");
    let tag = a.get("--tag").map(|t| format!("_{t}")).unwrap_or_default();
    for spec in a.list("--data", "synth384:30000,mini,bge") {
        let ds = load(&spec, nq);
        let gt = ground_truth(&ds.base, ds.dim, &ds.queries, K);
        let n = ds.n();
        eprintln!("[sweep] {} n={n} dim={}", ds.name, ds.dim);
        let mut cfgs = vec![];
        let mut pts: Vec<(f64, f64, Value)> = vec![];
        let mach0 = machine("start");
        for &m in &ms {
            for &efc in &efcs {
                wait_ram(3.0, "hnsw build");
                let b = build_ann(Store::F32Lib, &ds.base, ds.dim, m, efc);
                let (sq, s1, s10) = self_recall(&*b.h, &ds.base, ds.dim, 64, 12_000);
                eprintln!("  M={m} efc={efc} build {:.1}s heap {:.0} B/vec; self-recall not-top1 {s1}/{sq} absent-top10 {s10}", b.build_s, b.heap_bytes as f64 / n as f64);
                let mut rows = vec![];
                for &ef in &efs {
                    let r = bench_search(|q| b.h.search(q, K, ef), &ds, &gt, K, reps);
                    pts.push((r.recall, mean(&r.p99), json!({"m": m, "efc": efc, "ef": ef})));
                    rows.push(json!({"ef": ef, "bench": r.json()}));
                }
                cfgs.push(json!({"m": m, "efc": efc, "build_s": b.build_s, "heap_bytes_per_vec": b.heap_bytes as f64 / n as f64,
                    "self_recall": {"queried": sq, "not_top1": s1, "absent_top10": s10}, "rows": rows}));
            }
        }
        let front = pareto(&pts);
        let pf: Vec<Value> = front.iter().map(|(r, p, c)| json!({"recall": r, "p99_us": p, "cfg": c})).collect();
        write_json(&out, &format!("h8_hnsw_sweep_{}{tag}.json", ds.name), &json!({"mode": "sweep", "dataset": ds.name, "note": ds.note, "n": n,
            "dim": ds.dim, "nq": ds.nq(), "k": K, "reps": reps, "crate": "hnsw_rs 0.3.4 + reverse-edge patch (vendor_g17)",
            "machine_start": mach0, "machine_end": machine("end"), "configs": cfgs, "pareto": pf}));
        println!("\n#### HNSW (fixed crate) Pareto {} (n={n}, dim={})\n| recall@10 | p99 us | M | efc | ef |\n|---|---|---|---|---|", ds.name, ds.dim);
        for (r, p, c) in &front {
            println!("{}", md_row(&[f3(*r), f1(*p), c["m"].to_string(), c["efc"].to_string(), c["ef"].to_string()]));
        }
    }
}

// ============================================================ confirm
fn mode_confirm(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let reps = a.usize("--reps", 3);
    let nq = a.usize("--nq", 300);
    let builds = a.usize("--builds", 3);
    let kerns: Vec<Store> = a.list("--kernels", "lib,avx2").iter().map(|s| kern_store(s)).collect();
    let targets = [0.90, 0.95, 0.99, 0.995];
    let tag = a.get("--tag").map(|t| format!("_{t}")).unwrap_or_default();
    for spec in a.list("--data", "synth384:30000,mini,bge") {
        let ds = load(&spec, nq);
        let sweep: Value = serde_json::from_str(&std::fs::read_to_string(format!("{out}/h8_hnsw_sweep_{}{tag}.json", ds.name)).expect("run sweep first")).unwrap();
        let gt = ground_truth(&ds.base, ds.dim, &ds.queries, K);
        let n = ds.n();
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
                if !picks.iter().any(|(_, c)| *c == cfg) {
                    picks.push((t, cfg));
                }
            }
        }
        let mut res = vec![];
        for (t, cfg) in &picks {
            let (m, efc, ef) = (cfg["m"].as_u64().unwrap() as usize, cfg["efc"].as_u64().unwrap() as usize, cfg["ef"].as_u64().unwrap() as usize);
            for &kern in &kerns {
                let (mut rec, mut p50, mut p99, mut bt, mut bpv, mut sm1, mut sm10) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
                for _ in 0..builds {
                    wait_ram(3.0, "hnsw rebuild");
                    let b = build_ann(kern, &ds.base, ds.dim, m, efc);
                    let r = bench_search(|q| b.h.search(q, K, ef), &ds, &gt, K, reps);
                    let (_, s1, s10) = self_recall(&*b.h, &ds.base, ds.dim, 64, 30_000);
                    rec.push(r.recall);
                    p50.push(mean(&r.p50));
                    p99.push(mean(&r.p99));
                    bt.push(b.build_s);
                    bpv.push(b.heap_bytes as f64 / n as f64);
                    sm1.push(s1 as f64);
                    sm10.push(s10 as f64);
                }
                eprintln!("[confirm] {} target {t} M{m} efc{efc} ef{ef} {}: recall {:.3} p99 {:.0}us self-miss(top1) {:?}", ds.name, kern.short(), mean(&rec), mean(&p99), sm1);
                res.push(json!({"target_recall": t, "m": m, "efc": efc, "ef": ef, "kernel": kern.name(), "builds": builds,
                    "recall_per_build": spread(&rec), "p50_us_per_build": spread(&p50), "p99_us_per_build": spread(&p99),
                    "build_s": spread(&bt), "heap_bytes_per_vec": spread(&bpv), "self_not_top1_per_build": sm1, "self_absent_top10_per_build": sm10}));
            }
        }
        write_json(&out, &format!("h8_hnsw_confirm_{}{tag}.json", ds.name),
            &json!({"mode":"confirm","dataset":ds.name,"note":ds.note,"n":n,"dim":ds.dim,"nq":ds.nq(),"reps_per_build":reps,"results":res,"machine":machine("end")}));
    }
}

// ============================================================ crossover
fn mode_cross(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let reps = a.usize("--reps", 3);
    let nq = a.usize("--nq", 300);
    let dim = a.usize("--dim", 384);
    let sizes = a.nums("--sizes", "1000,2000,5000,10000,20000,30000,50000,100000");
    let recipes: Vec<(usize, usize)> = a.list("--recipes", "16:200,32:100").iter().map(|s| { let p: Vec<usize> = s.split(':').map(|x| x.parse().unwrap()).collect(); (p[0], p[1]) }).collect();
    let nmax = *sizes.iter().max().unwrap();
    let full = synth_ds(dim, nmax, nq);
    let efs = [10usize, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512];
    let hs: Vec<Box<dyn Ann>> = recipes.iter().map(|&(m, efc)| new_ann(Store::F32Lib, m, efc, nmax)).collect();
    let mut built = 0usize;
    let p1 = pool1();
    let mut rows = vec![];
    for &n in &sizes {
        wait_ram(3.0, "crossover step");
        while built < n {
            let c = (n - built).min(2048);
            for h in &hs {
                h.insert_batch(&full.base[built * dim..(built + c) * dim], dim, built);
            }
            built += c;
        }
        let ds = Ds { name: format!("synth{dim}-n{n}"), dim, base: full.base[..n * dim].to_vec(), queries: full.queries.clone(), note: String::new() };
        let gt = ground_truth(&ds.base, dim, &ds.queries, K);
        let flat = bench_search(|q| flat_topk(&ds.base, dim, q, K, false), &ds, &gt, K, reps);
        let flat_lib = bench_search(|q| flat_topk(&ds.base, dim, q, K, true), &ds, &gt, K, reps);
        let st = LowStore::build(Prec::Int8, &ds.base, dim);
        let mut tmp = vec![];
        let i8rr = bench_search(|q| st.topk(q, K, Some((&ds.base, 10)), &mut tmp), &ds, &gt, K, reps);
        let bf = LowStore::build(Prec::BlockFp16, &ds.base, dim);
        let bfr = bench_search(|q| bf.topk(q, K, None, &mut tmp), &ds, &gt, K, reps);
        let ix = {
            let mut ix = TurboQuantIndex::new(dim, 4).unwrap();
            ix.add(&ds.base);
            ix.prepare();
            ix
        };
        let tv = p1.install(|| {
            bench_search(|q| {
                let r = ix.search(q, K * 4);
                let ids: Vec<u32> = r.indices_for_query(0).iter().map(|&i| i as u32).collect();
                rerank_f32(&ds.base, dim, q, &ids, K)
            }, &ds, &gt, K, reps)
        });
        let mut hrows = vec![];
        for (ri, h) in hs.iter().enumerate() {
            let mut per_t = vec![];
            for target in [0.95f64, 0.99] {
                let mut hit = None;
                for &ef in &efs {
                    let r = bench_search(|q| h.search(q, K, ef), &ds, &gt, K, reps);
                    if r.recall >= target {
                        hit = Some(json!({"target": target, "ef": ef, "bench": r.json()}));
                        break;
                    }
                }
                per_t.push(hit.unwrap_or(json!({"target": target, "ef": null, "note": "not reached with ef<=512"})));
            }
            let mut fixed = vec![];
            if let Some(fl) = a.get("--efs-fixed") {
                for ef in fl.split(',').map(|x| x.parse::<usize>().unwrap()) {
                    let r = bench_search(|q| h.search(q, K, ef), &ds, &gt, K, reps);
                    println!("  [fixed ef] dim{dim} n={n} M{}/efc{} ef{ef}: recall {:.3} p50 {:.0}us p99 {:.0}us | flat avx2 p50 {:.0}us (x{:.2})", recipes[ri].0, recipes[ri].1, r.recall, mean(&r.p50), mean(&r.p99), mean(&flat.p50), mean(&flat.p50) / mean(&r.p50));
                    fixed.push(json!({"ef": ef, "bench": r.json()}));
                }
            }
            hrows.push(json!({"m": recipes[ri].0, "efc": recipes[ri].1, "targets": per_t, "fixed_ef": fixed}));
        }
        eprintln!("[cross d{dim}] n={n}: flat avx2 p50 {:.0}us; hnsw {:?}", mean(&flat.p50), hrows.iter().map(|h| h["targets"][0]["bench"]["p50_us"]["mean"].as_f64()).collect::<Vec<_>>());
        rows.push(json!({"n": n, "flat_f32_avx2": flat.json(), "flat_f32_lib_kernel": flat_lib.json(), "flat_blockfp16": bfr.json(),
            "int8_scan_rerank10x": i8rr.json(), "turbovec4_rerank4x_1thread": tv.json(), "hnsw": hrows}));
    }
    println!("\n#### flat vs HNSW crossover (dim {dim}, synthetic, 1 thread, p50 us; HNSW = smallest ef reaching recall target)");
    println!("| N | flat avx2 | flat lib | flat bfp16 | int8+rr10 | tv4+rr4 | {} |\n|---|---|---|---|---|---|{}|", recipes.iter().flat_map(|(m, e)| [format!("M{m}/efc{e} r>=.95 (ef)"), format!("M{m}/efc{e} r>=.99 (ef)")]).collect::<Vec<_>>().join(" | "), "---|".repeat(recipes.len() * 2));
    for r in &rows {
        let g = |k: &str| r[k]["p50_us"]["mean"].as_f64().unwrap_or(f64::NAN);
        let hh: Vec<String> = r["hnsw"].as_array().unwrap().iter().flat_map(|h| h["targets"].as_array().unwrap().iter().map(|t| {
            t["bench"]["p50_us"]["mean"].as_f64().map_or("-".into(), |x| format!("{x:.0} ({})", t["ef"]))
        }).collect::<Vec<_>>()).collect();
        println!("| {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {} |", r["n"], g("flat_f32_avx2"), g("flat_f32_lib_kernel"), g("flat_blockfp16"), g("int8_scan_rerank10x"), g("turbovec4_rerank4x_1thread"), hh.join(" | "));
    }
    let ctag = a.get("--tag").map(|t| format!("_{t}")).unwrap_or_default();
    write_json(&out, &format!("h8_cross_d{dim}{ctag}.json"), &json!({"mode":"cross","dim":dim,"recipes":recipes,"rows":rows,"machine":machine("end")}));
}

// ============================================================ shared candidate machinery
struct Cand {
    name: String,
    n: usize,
    heap: usize,            // measured heap bytes of the index
    ram_extra: usize,       // exact-f32 rerank copy kept in RAM
    disk_extra: usize,      // exact-f32 rerank copy on disk
    build_s: f64,
    bench: Bench,
    extra: Value,
}
impl Cand {
    fn ram_per_vec(&self) -> f64 {
        (self.heap + self.ram_extra) as f64 / self.n as f64
    }
    fn json(&self) -> Value {
        let n = self.n as f64;
        json!({"name": self.name, "n": self.n, "index_heap_bytes_per_vec": self.heap as f64 / n, "ram_f32_rerank_bytes_per_vec": self.ram_extra as f64 / n,
            "disk_f32_rerank_bytes_per_vec": self.disk_extra as f64 / n, "ram_bytes_per_vec_total": self.ram_per_vec(),
            "m_vectors_per_decimal_GB_ram": 1e9 / self.ram_per_vec() / 1e6, "build_s": self.build_s, "bench": self.bench.json(), "extra": self.extra})
    }
    fn md(&self) -> String {
        md_row(&[self.name.clone(), f1(self.heap as f64 / self.n as f64), f1(self.ram_per_vec()), f1(self.disk_extra as f64 / self.n as f64),
            format!("{:.2}", 1e3 / self.ram_per_vec()), f3(self.bench.recall), f1(mean(&self.bench.p50)), f1(mean(&self.bench.p99)), f1(self.build_s)])
    }
}
const CAND_HDR: &str = "| config | index heap B/vec (measured) | RAM B/vec total | SSD B/vec | M vectors/GB RAM | recall@10 | p50 us | p99 us | build s |\n|---|---|---|---|---|---|---|---|---|";

/// Source of base rows in chunks.
#[derive(Clone, Copy)]
enum Src<'a> {
    Mem(&'a [f32]),
    Syn { dim: usize, n: usize, nq: usize },
}
impl Src<'_> {
    fn each(&self, dim: usize, chunk: usize, f: &mut dyn FnMut(&[f32])) {
        match *self {
            Src::Mem(b) => {
                for c in b.chunks(dim * chunk) {
                    f(c)
                }
            }
            Src::Syn { dim, n, nq } => {
                let (_q, mut st) = synth_stream(dim, n, nq);
                while let Some(c) = st.next(chunk) {
                    f(&c)
                }
            }
        }
    }
    fn n(&self, dim: usize) -> usize {
        match *self {
            Src::Mem(b) => b.len() / dim,
            Src::Syn { n, .. } => n,
        }
    }
}
fn tv_from(src: Src, dim: usize, bits: usize) -> (TurboQuantIndex, f64, usize) {
    let l0 = live_bytes();
    let t = Instant::now();
    let mut ix = TurboQuantIndex::new(dim, bits).expect("turbovec");
    match src {
        // resident base: one add (no retained scratch, see g15 tvmem); streams: 2048-row adds
        Src::Mem(b) => ix.add(b),
        _ => src.each(dim, 50_000, &mut |c| {
            for s in c.chunks(dim * 2048) {
                ix.add(s);
            }
        }),
    }
    ix.prepare();
    (ix, t.elapsed().as_secs_f64(), live_bytes().saturating_sub(l0))
}
fn low_from(src: Src, dim: usize, prec: Prec) -> (LowStore, f64, usize) {
    let l0 = live_bytes();
    let t = Instant::now();
    let mut st = LowStore::empty(prec, dim);
    src.each(dim, 50_000, &mut |c| st.merge(LowStore::build(prec, c, dim)));
    st.shrink();
    (st, t.elapsed().as_secs_f64(), live_bytes().saturating_sub(l0))
}
fn rq_from(src: Src, dim: usize, bits: usize) -> (RaBitQ, f64, usize) {
    let t = Instant::now();
    let mut rq = RaBitQ::new(dim, bits, SEED ^ 0x5DEE_CE66_D1CE_4E5B);
    let l0 = live_bytes(); // rotation matrix is part of the index (dim*dim*4 B, amortised)
    src.each(dim, 20_000, &mut |c| rq.add(c));
    rq.shrink();
    let heap = live_bytes().saturating_sub(l0) + dim * dim * 4;
    (rq, t.elapsed().as_secs_f64(), heap)
}
fn tv_search_rr(ix: &TurboQuantIndex, dim: usize, q: &[f32], rr: Option<(&Rr, usize)>) -> Vec<u32> {
    let fetch = K * rr.map_or(1, |x| x.1);
    let r = ix.search(q, fetch);
    let ids: Vec<u32> = r.indices_for_query(0).iter().map(|&i| i as u32).collect();
    match rr {
        None => ids.into_iter().take(K).collect(),
        Some((r, _)) => r.rerank(dim, q, &ids, K),
    }
}

// ============================================================ memory (max vectors per GB)
fn mode_memory(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let reps = a.usize("--reps", 3);
    let nq = a.usize("--nq", 300);
    let quick = a.has("--quick");
    let tmpdir = "C:/Users/mariu/nm-target-r8/h8_tmp";
    std::fs::create_dir_all(tmpdir).ok();
    for spec in a.list("--data", if quick { "synth384:5000" } else { "mini,bge,synth384:100000" }) {
        wait_ram(3.0, "memory mode dataset");
        let ds = load(&spec, nq);
        let (dim, n) = (ds.dim, ds.n());
        let gt = ground_truth(&ds.base, dim, &ds.queries, K);
        eprintln!("[memory] {} n={n} dim={dim}", ds.name);
        let fpath = format!("{tmpdir}/{}.f32", ds.name);
        FileRows::write_slice(&fpath, &ds.base, dim);
        let fbuf = FileRows::open(&fpath, dim, n, false);
        let fdir = FileRows::open(&fpath, dim, n, true);
        // warm the buffered file into the OS cache with one scan (honest: it is then served from page cache = RAM)
        {
            let mut r = vec![0f32; dim];
            for i in 0..n {
                fbuf.row_into(i, &mut r);
            }
        }
        let rams = Rr::Ram(&ds.base);
        let rfb = Rr::File(&fbuf);
        let rfd = Rr::File(&fdir);
        let f32b = n * dim * 4;
        let p1 = pool1();
        let mut cands: Vec<Cand> = vec![];
        let none = Value::Null;
        // exact flat f32
        let b = bench_search(|q| flat_topk(&ds.base, dim, q, K, false), &ds, &gt, K, reps);
        cands.push(Cand { name: "flat f32 exact (avx2)".into(), n, heap: f32b, ram_extra: 0, disk_extra: 0, build_s: 0.0, bench: b, extra: none.clone() });
        // flat low precision stores
        for prec in [Prec::F16, Prec::BlockFp16, Prec::Int8] {
            let (st, bt, heap) = low_from(Src::Mem(&ds.base), dim, prec);
            let mut tmp = vec![];
            let b = bench_search(|q| st.topk(q, K, None, &mut tmp), &ds, &gt, K, reps);
            cands.push(Cand { name: format!("flat {}", prec.name()), n, heap, ram_extra: 0, disk_extra: 0, build_s: bt, bench: b, extra: none.clone() });
            if prec == Prec::Int8 {
                for (rrn, rr, ram, disk) in [("RAM", &rams, f32b, 0), ("SSD buffered", &rfb, 0, f32b), ("SSD direct", &rfd, 0, f32b)] {
                    let b = bench_search(|q| {
                        st.scores(q, &mut tmp);
                        let mut top = TopK::new(K * 4);
                        for (i, &s) in tmp.iter().enumerate() {
                            top.push(s, i as u32);
                        }
                        rr.rerank(dim, q, &top.ids(), K)
                    }, &ds, &gt, K, reps);
                    cands.push(Cand { name: format!("flat int8 + rerank4x [{rrn}]"), n, heap, ram_extra: ram, disk_extra: disk, build_s: bt, bench: b, extra: none.clone() });
                }
            }
        }
        // turbovec
        for bits in [2usize, 3, 4] {
            let (ix, bt, heap) = tv_from(Src::Mem(&ds.base), dim, bits);
            let b = p1.install(|| bench_search(|q| tv_search_rr(&ix, dim, q, None), &ds, &gt, K, reps));
            cands.push(Cand { name: format!("turbovec {bits}-bit"), n, heap, ram_extra: 0, disk_extra: 0, build_s: bt, bench: b, extra: none.clone() });
            let facs: &[usize] = if bits == 2 { &[4, 8, 16] } else { &[2, 4, 8] };
            for &f in facs {
                for (rrn, rr, ram, disk) in [("RAM", &rams, f32b, 0), ("SSD buffered", &rfb, 0, f32b), ("SSD direct", &rfd, 0, f32b)] {
                    if rrn != "RAM" && f == facs[0] && bits != 4 && bits != 2 {
                        continue;
                    }
                    let b = p1.install(|| bench_search(|q| tv_search_rr(&ix, dim, q, Some((rr, f))), &ds, &gt, K, reps));
                    cands.push(Cand { name: format!("turbovec {bits}-bit + rerank{f}x [{rrn}]"), n, heap, ram_extra: ram, disk_extra: disk, build_s: bt, bench: b, extra: none.clone() });
                }
            }
        }
        // RaBitQ-style
        for bits in [1usize, 2, 3, 4, 5, 6] {
            let (rq, bt, heap) = rq_from(Src::Mem(&ds.base), dim, bits);
            let mut tmp = vec![];
            let b = p1.install(|| bench_search(|q| rq.topk(q, K, None, &mut tmp), &ds, &gt, K, reps));
            cands.push(Cand { name: format!("rabitq-style {bits}-bit"), n, heap, ram_extra: 0, disk_extra: 0, build_s: bt, bench: b, extra: none.clone() });
            if bits <= 4 {
                let facs: &[usize] = if bits == 1 { &[8, 16] } else { &[2, 4, 8] };
                for &f in facs {
                    for (rrn, rr, ram, disk) in [("RAM", &rams, f32b, 0), ("SSD buffered", &rfb, 0, f32b), ("SSD direct", &rfd, 0, f32b)] {
                        if rrn != "RAM" && f != 4 && !(bits == 1 && f == 8) {
                            continue;
                        }
                        let b = p1.install(|| bench_search(|q| rq.topk(q, K, Some((rr, f)), &mut tmp), &ds, &gt, K, reps));
                        cands.push(Cand { name: format!("rabitq-style {bits}-bit + rerank{f}x [{rrn}]"), n, heap, ram_extra: ram, disk_extra: disk, build_s: bt, bench: b, extra: none.clone() });
                    }
                }
            }
        }
        // HNSW with different stored vector formats
        for (m, efc, ef) in [(32usize, 100usize, 24usize), (16, 100, 24), (48, 200, 16)] {
            for st in [Store::F32Lib, Store::Bfp16, Store::Int8] {
                wait_ram(3.0, "hnsw build");
                let bh = build_ann(st, &ds.base, dim, m, efc);
                let (_, s1, s10) = self_recall(&*bh.h, &ds.base, dim, 64, 20_000);
                let b = bench_search(|q| bh.h.search(q, K, ef), &ds, &gt, K, reps);
                cands.push(Cand { name: format!("HNSW M{m} efc{efc} ef{ef} [{} stored]", st.short()), n, heap: bh.heap_bytes, ram_extra: 0, disk_extra: 0, build_s: bh.build_s, bench: b,
                    extra: json!({"self_not_top1": s1, "self_absent_top10": s10, "point_payload_bytes": st.point_bytes(dim), "kernel": st.name()}) });
            }
        }
        println!("\n#### max-vectors-per-GB, measured (heap via counting allocator) {} n={n} dim={dim}\n{CAND_HDR}", ds.name);
        for c in &cands {
            println!("{}", c.md());
        }
        write_json(&out, &format!("h8_memory_{}.json", ds.name), &json!({"mode":"memory","dataset":ds.name,"note":ds.note,"n":n,"dim":dim,"nq":ds.nq(),"reps":reps,
            "rows": cands.iter().map(|c| c.json()).collect::<Vec<_>>(), "machine": machine("end")}));
        drop(fbuf);
        drop(fdir);
        std::fs::remove_file(&fpath).ok();
    }
}

// ============================================================ scale (streaming)
fn mode_scale(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let reps = a.usize("--reps", 3);
    let nq = a.usize("--nq", 200);
    let dim = 384usize;
    let sizes = a.nums("--sizes", "100000,300000,1000000");
    let cap_gb = a.get("--cap-gb").map(|v| v.parse::<f64>().unwrap()).unwrap_or(2.5);
    let tag = a.get("--tag").map(|t| format!("_{t}")).unwrap_or_default();
    let do_hnsw = !a.has("--no-hnsw");
    let do_flat32 = !a.has("--no-flat32") && !a.has("--only-hnsw");
    let only_hnsw = a.has("--only-hnsw");
    let hn_builds = a.usize("--hnsw-builds", 1);
    let hn_cfgs: Vec<(usize, usize, usize, &str)> = a
        .list("--hnsw", "16:200:128:f32lib,32:100:32:f32lib,48:100:24:f32lib,32:100:32:bfp16,32:100:32:int8")
        .iter()
        .map(|s| { let p: Vec<&str> = s.split(':').collect(); (p[0].parse().unwrap(), p[1].parse().unwrap(), p[2].parse().unwrap(), Box::leak(p[3].to_string().into_boxed_str()) as &str) })
        .collect();
    let tmpdir = "C:/Users/mariu/nm-target-r8/h8_tmp";
    std::fs::create_dir_all(tmpdir).ok();
    let mut all = vec![];
    let mut hnsw_bpv: std::collections::HashMap<String, f64> = Default::default();
    for &n in &sizes {
        wait_ram(3.0, "scale size");
        eprintln!("[scale] N={n}: streaming queries + ground truth");
        let t = Instant::now();
        let (queries, st) = synth_stream(dim, n, nq);
        let gt = ground_truth_stream(st, &queries, dim, K, 50_000);
        eprintln!("  gt {:.1}s free {:.1}GB", t.elapsed().as_secs_f64(), free_gb());
        let ds = empty_ds(&format!("synth{dim}-n{n}"), dim, queries);
        let src = Src::Syn { dim, n, nq };
        let mach = machine("start");
        let p1 = pool1();
        let mut cands: Vec<Cand> = vec![];
        let none = Value::Null;
        // rerank file (SSD): written once per size, deleted at the end
        let fpath = format!("{tmpdir}/scale_{n}.f32");
        let (_q, mut st2) = synth_stream(dim, n, nq);
        let tw = Instant::now();
        FileRows::write_stream(&fpath, &mut st2, dim);
        eprintln!("  wrote rerank file {} ({:.1}s)", fpath, tw.elapsed().as_secs_f64());
        let fdir = FileRows::open(&fpath, dim, n, true);
        let fbuf = FileRows::open(&fpath, dim, n, false);
        let rfd = Rr::File(&fdir);
        let rfb = Rr::File(&fbuf);
        let f32b = n * dim * 4;
        // quantised flat stores
        for prec in if only_hnsw { vec![] } else { vec![Prec::BlockFp16, Prec::Int8] } {
            wait_ram(3.0, "flat store");
            let (stor, bt, heap) = low_from(src, dim, prec);
            let mut tmp = vec![];
            let b = bench_search(|q| stor.topk(q, K, None, &mut tmp), &ds, &gt, K, reps);
            cands.push(Cand { name: format!("flat {}", prec.name()), n, heap, ram_extra: 0, disk_extra: 0, build_s: bt, bench: b, extra: none.clone() });
            if prec == Prec::Int8 {
                for (rrn, rr) in [("SSD direct", &rfd), ("SSD buffered", &rfb)] {
                    let b = bench_search(|q| {
                        stor.scores(q, &mut tmp);
                        let mut top = TopK::new(K * 4);
                        for (i, &s) in tmp.iter().enumerate() {
                            top.push(s, i as u32);
                        }
                        rr.rerank(dim, q, &top.ids(), K)
                    }, &ds, &gt, K, reps);
                    cands.push(Cand { name: format!("flat int8 + rerank4x [{rrn}]"), n, heap, ram_extra: 0, disk_extra: f32b, build_s: bt, bench: b, extra: none.clone() });
                }
            }
        }
        // turbovec
        for bits in if only_hnsw { vec![] } else { vec![2usize, 3, 4] } {
            wait_ram(3.0, "turbovec");
            let (ix, bt, heap) = tv_from(src, dim, bits);
            let b = p1.install(|| bench_search(|q| tv_search_rr(&ix, dim, q, None), &ds, &gt, K, reps));
            cands.push(Cand { name: format!("turbovec {bits}-bit"), n, heap, ram_extra: 0, disk_extra: 0, build_s: bt, bench: b, extra: none.clone() });
            let facs: &[usize] = if bits == 2 { &[4, 16] } else if bits == 3 { &[8] } else { &[4] };
            for &f in facs {
                for (rrn, rr) in [("SSD direct", &rfd), ("SSD buffered", &rfb)] {
                    let b = p1.install(|| bench_search(|q| tv_search_rr(&ix, dim, q, Some((rr, f))), &ds, &gt, K, reps));
                    cands.push(Cand { name: format!("turbovec {bits}-bit + rerank{f}x [{rrn}]"), n, heap, ram_extra: 0, disk_extra: f32b, build_s: bt, bench: b, extra: none.clone() });
                }
            }
        }
        // RaBitQ-style
        for bits in if only_hnsw { vec![] } else { vec![1usize, 2, 3, 4] } {
            wait_ram(3.0, "rabitq");
            let (rq, bt, heap) = rq_from(src, dim, bits);
            let mut tmp = vec![];
            let b = p1.install(|| bench_search(|q| rq.topk(q, K, None, &mut tmp), &ds, &gt, K, reps));
            cands.push(Cand { name: format!("rabitq-style {bits}-bit"), n, heap, ram_extra: 0, disk_extra: 0, build_s: bt, bench: b, extra: none.clone() });
            let facs: &[usize] = if bits == 1 { &[16] } else if bits == 2 { &[8] } else { &[4] };
            for &f in facs {
                for (rrn, rr) in [("SSD direct", &rfd), ("SSD buffered", &rfb)] {
                    let b = p1.install(|| bench_search(|q| rq.topk(q, K, Some((rr, f)), &mut tmp), &ds, &gt, K, reps));
                    cands.push(Cand { name: format!("rabitq-style {bits}-bit + rerank{f}x [{rrn}]"), n, heap, ram_extra: 0, disk_extra: f32b, build_s: bt, bench: b, extra: none.clone() });
                }
            }
        }
        // HNSW
        if do_hnsw {
            for &(m, efc, ef, stn) in &hn_cfgs {
                let stv = kern_store(stn);
                let key = format!("M{m}-efc{efc}-{stn}");
                let proj = hnsw_bpv.get(&key).map(|b| b * n as f64 / 1e9);
                // first-time projection from the closest measured size, or a conservative estimate (graph ~ 150 B/edge * 2M + payload)
                let est = proj.unwrap_or((4600.0 + 55.0 * m as f64 + (stv.point_bytes(dim) as f64 - 1536.0).min(0.0)) * n as f64 / 1e9);
                if est > cap_gb - 0.15 || free_gb() < est + 1.5 {
                    eprintln!("  HNSW {key} SKIP: projected {est:.2} GB (cap {cap_gb} GB, free {:.1})", free_gb());
                    cands.push(Cand { name: format!("HNSW M{m} efc{efc} ef{ef} [{stn} stored] SKIPPED (projected {est:.2} GB > cap {cap_gb} GB or free RAM)"), n, heap: 0, ram_extra: 0, disk_extra: 0, build_s: 0.0,
                        bench: Bench { recall: f64::NAN, recall_min_q: f64::NAN, p50: vec![f64::NAN], p99: vec![f64::NAN], mean: vec![f64::NAN] }, extra: json!({"skipped": true, "projected_gb": est}) });
                    continue;
                }
                for bi in 0..hn_builds {
                    wait_ram(est + 1.5, "hnsw build");
                    let (_q, st3) = synth_stream(dim, n, nq);
                    let bh = build_ann_stream(stv, st3, dim, m, efc, n);
                    eprintln!("  HNSW M{m} efc{efc} {stn} build{bi}: {:.0}s heap {:.0} B/vec, ws {:.0} MB", bh.build_s, bh.heap_bytes as f64 / n as f64, rss_mb().0);
                    hnsw_bpv.insert(key.clone(), bh.heap_bytes as f64 / n as f64);
                    let mut efl = vec![ef, 32, 64, 128, 256];
                    efl.sort_unstable();
                    efl.dedup();
                    for &e in &efl {
                        let b = bench_search(|q| bh.h.search(q, K, e), &ds, &gt, K, reps);
                        cands.push(Cand { name: format!("HNSW M{m} efc{efc} ef{e} [{stn} stored]{}", if hn_builds > 1 { format!(" build{bi}") } else { String::new() }), n, heap: bh.heap_bytes, ram_extra: 0, disk_extra: 0, build_s: bh.build_s, bench: b,
                            extra: json!({"kernel": stv.name(), "ws_mb": rss_mb().0, "build_index": bi}) });
                    }
                }
            }
        }
        // exact flat f32 last (needs the f32 base resident): reference
        if do_flat32 && f32b as f64 / 1e9 + 0.6 < cap_gb && free_gb() > f32b as f64 / 1e9 + 2.0 {
            let mut base = Vec::with_capacity(n * dim);
            src.each(dim, 50_000, &mut |c| base.extend_from_slice(c));
            let b = bench_search(|q| flat_topk(&base, dim, q, K, false), &ds, &gt, K, reps);
            cands.push(Cand { name: "flat f32 exact (avx2)".into(), n, heap: f32b, ram_extra: 0, disk_extra: 0, build_s: 0.0, bench: b, extra: none.clone() });
            // also the in-RAM rerank reference for the strongest cheap codes
            let (ix, bt, heap) = tv_from(Src::Mem(&base), dim, 4);
            let rr = Rr::Ram(&base);
            let b = p1.install(|| bench_search(|q| tv_search_rr(&ix, dim, q, Some((&rr, 4))), &ds, &gt, K, reps));
            cands.push(Cand { name: "turbovec 4-bit + rerank4x [RAM f32]".into(), n, heap, ram_extra: f32b, disk_extra: 0, build_s: bt, bench: b, extra: none.clone() });
        }
        let (ws, peak) = rss_mb();
        println!("\n#### scale N={n} dim={dim} (peak WS {peak:.0} MB)\n{CAND_HDR}");
        for c in cands.iter().filter(|c| c.extra.get("skipped").is_none()) {
            println!("{}", c.md());
        }
        for c in cands.iter().filter(|c| c.extra.get("skipped").is_some()) {
            println!("{}", c.name);
        }
        all.push(json!({"n": n, "dim": dim, "nq": nq, "peak_ws_mb": peak, "ws_mb": ws, "free_gb_after": free_gb(), "machine_start": mach,
            "rows": cands.iter().map(|c| c.json()).collect::<Vec<_>>()}));
        drop(fdir);
        drop(fbuf);
        std::fs::remove_file(&fpath).ok();
        write_json(&out, &format!("h8_scale{tag}.json"), &json!({"mode":"scale","k":K,"reps":reps,"cap_gb":cap_gb,"sizes":all,"machine":machine("end")}));
    }
}

// ============================================================ concurrency
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
    let (m, efc, ef) = (a.usize("--m", 32), a.usize("--efc", 100), a.usize("--ef", 32));
    let extra = 20_000;
    wait_ram(3.0, "conc");
    let full = synth_ds(384, n + extra, nq);
    let dim = full.dim;
    let ds = Ds { name: full.name.clone(), dim, base: full.base[..n * dim].to_vec(), queries: full.queries.clone(), note: full.note.clone() };
    let fresh: Arc<Vec<f32>> = Arc::new(full.base[n * dim..].to_vec());
    drop(full);
    let dur = Duration::from_secs(secs);
    let mut res = vec![];
    let mach = machine("start");
    eprintln!("[conc] building HNSW M{m} efc{efc} n={n}");
    let hb = build_ann(Store::F32Lib, &ds.base, dim, m, efc);
    eprintln!("  build {:.1}s", hb.build_s);
    let h: Arc<Box<dyn Ann>> = Arc::new(hb.h);
    let gt = ground_truth(&ds.base, dim, &ds.queries, K);
    let r1 = bench_search(|q| h.search(q, K, ef), &ds, &gt, K, 1);
    eprintln!("  recall@10 {:.3} (ef {ef})", r1.recall);
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
    let nm = format!("hnsw M{m} efc{efc} ef{ef} (fixed crate)");
    for (ti, &thr) in [1usize, 2, 4].iter().enumerate() {
        res.push(json!({"index": nm, "scenario": "read-only, shared Arc, no lock", "threads": thr, "reps": ro[ti]}));
    }
    for (ti, &thr) in [1usize, 2].iter().enumerate() {
        res.push(json!({"index": nm, "scenario": "readers + 1 concurrent insert thread (insert_slice on &self, no external lock)", "threads": thr, "reps": rw[ti]}));
    }
    // recall of items inserted concurrently: self-recall of the 3 x ~tens of thousand writer inserts is not tracked (documented).
    drop(h);
    // int8 flat and turbovec4 + rr4 (RAM) read-only for context
    let (st8, _, _) = low_from(Src::Mem(&ds.base), dim, Prec::Int8);
    let base_arc = Arc::new(ds.base.clone());
    let st8 = Arc::new(st8);
    for &thr in &[1usize, 2, 4] {
        let mut reps_v = vec![];
        for _ in 0..reps {
            let (s, b) = (st8.clone(), base_arc.clone());
            let tmps: Vec<std::sync::Mutex<Vec<f32>>> = (0..thr).map(|_| std::sync::Mutex::new(vec![])).collect();
            reps_v.push(run_readers(thr, dur, &ds, &move |t, q| {
                let mut g = tmps[t].lock().unwrap();
                s.topk(q, K, Some((&b, 4)), &mut g).len()
            }, None));
        }
        res.push(json!({"index": "flat int8 + rerank4x (RAM f32)", "scenario": "read-only", "threads": thr, "reps": reps_v}));
    }
    drop(st8);
    let (ix, _, _) = tv_from(Src::Mem(&ds.base), dim, 4);
    let lock = Arc::new(RwLock::new(ix));
    for &thr in &[1usize, 2, 4] {
        let mut reps_v = vec![];
        for _ in 0..reps {
            let (l, b) = (lock.clone(), base_arc.clone());
            let pools: Vec<rayon::ThreadPool> = (0..thr).map(|_| pool1()).collect();
            reps_v.push(run_readers(thr, dur, &ds, &move |t, q| {
                pools[t].install(|| { let g = l.read().unwrap(); tv_search_rr(&g, dim, q, Some((&Rr::Ram(&b), 4))).len() })
            }, None));
        }
        res.push(json!({"index": "turbovec-4bit + rerank4x (RAM f32), RwLock, per-reader 1-thread rayon pool", "scenario": "read-only", "threads": thr, "reps": reps_v}));
    }
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
                pools[t].install(|| { let g = l.read().unwrap(); tv_search_rr(&g, dim, q, Some((&Rr::Ram(&b), 4))).len() })
            }, Some(&w)));
        }
        res.push(json!({"index": "turbovec-4bit + rerank4x, RwLock", "scenario": "2 readers + writer adding 64 vecs/50ms (new vectors not in rerank store: skipped by rerank filter only for the file variant; RAM variant would index OOB so ids >= n are filtered by rerank_f32 callers: see note)", "threads": 2, "reps": reps_v}));
    }
    println!("\n#### concurrency (n={n}, dim={dim}, {secs}s windows x{reps})\n| index | scenario | threads | qps mean [min-max] | p50 us | p99 us | writer ops/s |\n|---|---|---|---|---|---|---|");
    for r in &res {
        let reps_a = r["reps"].as_array().unwrap();
        let q: Vec<f64> = reps_a.iter().map(|x| x["qps"].as_f64().unwrap()).collect();
        let p50: Vec<f64> = reps_a.iter().map(|x| x["p50_us"].as_f64().unwrap()).collect();
        let p99: Vec<f64> = reps_a.iter().map(|x| x["p99_us"].as_f64().unwrap()).collect();
        let w: Vec<f64> = reps_a.iter().map(|x| x["writer_ops_per_s"].as_f64().unwrap()).collect();
        let (lo, hi) = minmax(&q);
        println!("| {} | {} | {} | {:.0} [{:.0}-{:.0}] | {:.0} | {:.0} | {:.0} |", r["index"].as_str().unwrap(), r["scenario"].as_str().unwrap(), r["threads"], mean(&q), lo, hi, mean(&p50), mean(&p99), mean(&w));
    }
    write_json(&out, "h8_conc.json", &json!({"mode":"conc","n":n,"dim":dim,"window_s":secs,"reps":reps,"hnsw_recall":r1.recall,"results":res,"machine_start":mach,"machine_end":machine("end")}));
}

// ============================================================ turbovec memory anatomy (bits 2/3/4, single add)
fn mode_tvmem(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let n = a.usize("--n", 100_000);
    let (q, mut st) = synth_stream(384, n, 20);
    let mut base = vec![];
    while let Some(c) = st.next(50_000) {
        base.extend(c);
    }
    let dim = 384;
    let mut res = vec![];
    for bits in [2usize, 3, 4] {
        let l0 = live_bytes();
        let mut ix = TurboQuantIndex::new(dim, bits).unwrap();
        ix.add(&base);
        let after_add = live_bytes() - l0;
        ix.prepare();
        let after_prepare = live_bytes() - l0;
        let _ = ix.search(&q[..dim], 10);
        let after_search = live_bytes() - l0;
        let nominal = (dim * bits).div_ceil(8) + 4;
        eprintln!("[tvmem] {bits}bit: nominal {nominal} add {:.1} prepare {:.1} first-search {:.1} B/vec", after_add as f64 / n as f64, after_prepare as f64 / n as f64, after_search as f64 / n as f64);
        res.push(json!({"bits": bits, "nominal_bytes_per_vec": nominal, "heap_after_add": after_add as f64 / n as f64, "heap_after_prepare": after_prepare as f64 / n as f64, "heap_after_first_search": after_search as f64 / n as f64}));
    }
    write_json(&out, "h8_tvmem.json", &json!({"mode":"tvmem","n":n,"dim":dim,"rows":res}));
}

// ============================================================ self-recall check (bug detector): own vector as query, ef 64 and 512
fn mode_selfcheck(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let builds = a.usize("--builds", 3);
    let tag = a.get("--tag").map(|t| format!("_{t}")).unwrap_or_default();
    let cfgs: Vec<(usize, usize)> = a.list("--recipes", "16:100,16:200,32:100").iter().map(|s| { let p: Vec<usize> = s.split(':').map(|x| x.parse().unwrap()).collect(); (p[0], p[1]) }).collect();
    let mut res = vec![];
    for spec in a.list("--data", "synth384:30000,mini,bge") {
        let ds = load(&spec, 10);
        let (dim, n) = (ds.dim, ds.n());
        for &(m, efc) in &cfgs {
            for b in 0..builds {
                wait_ram(3.0, "selfcheck build");
                let bh = build_ann(Store::F32Lib, &ds.base, dim, m, efc);
                let (q64, a1, a10) = self_recall(&*bh.h, &ds.base, dim, 64, n);
                let (q512, b1, b10) = self_recall(&*bh.h, &ds.base, dim, 512, n);
                println!("selfcheck {} n={n} M{m} efc{efc} build{b}: ef64 not-top1 {a1}/{q64} absent-top10 {a10}; ef512 not-top1 {b1}/{q512} absent-top10 {b10}", ds.name);
                res.push(json!({"dataset": ds.name, "n": n, "m": m, "efc": efc, "build": b, "ef64": {"queried": q64, "not_top1": a1, "absent_top10": a10},
                    "ef512": {"queried": q512, "not_top1": b1, "absent_top10": b10}}));
            }
        }
    }
    write_json(&out, &format!("h8_selfcheck{tag}.json"), &json!({"mode": "selfcheck", "rows": res, "machine": machine("end")}));
}

// ============================================================ lat: load-robust, interleaved latency of picked configs (lib vs avx2 kernel)
fn mode_lat(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let nq = a.usize("--nq", 300);
    let builds = a.usize("--builds", 3);
    let reps = a.usize("--reps", 7);
    let tag = a.get("--tag").map(|t| format!("_{t}")).unwrap_or_default();
    // picks: lowest-p99 sweep config reaching each recall target (same rule as confirm) unless --cfgs M:efc:ef,...
    let targets = [0.95, 0.99, 0.995];
    for spec in a.list("--data", "synth384:30000,mini,bge") {
        let ds = load(&spec, nq);
        let gt = ground_truth(&ds.base, ds.dim, &ds.queries, K);
        let n = ds.n();
        let mut picks: Vec<(usize, usize, usize)> = vec![];
        if let Some(c) = a.get("--cfgs") {
            for s in c.split(',') {
                let p: Vec<usize> = s.split(':').map(|x| x.parse().unwrap()).collect();
                picks.push((p[0], p[1], p[2]));
            }
        } else {
            let sweep: Value = serde_json::from_str(&std::fs::read_to_string(format!("{out}/h8_hnsw_sweep_{}.json", ds.name)).expect("run sweep first")).unwrap();
            for &t in &targets {
                let mut best: Option<(f64, (usize, usize, usize))> = None;
                for c in sweep["configs"].as_array().unwrap() {
                    for r in c["rows"].as_array().unwrap() {
                        let rec = r["bench"]["recall_at_10"].as_f64().unwrap();
                        let p99 = r["bench"]["p99_us"]["mean"].as_f64().unwrap();
                        let key = (c["m"].as_u64().unwrap() as usize, c["efc"].as_u64().unwrap() as usize, r["ef"].as_u64().unwrap() as usize);
                        // exclude efc < 2*M: hnsw_rs level-0 lists hold 2M entries and efc<2M graphs have unreachable items (see selfcheck)
                        if rec >= t && key.1 >= 2 * key.0 && best.as_ref().map_or(true, |b| p99 < b.0) {
                            best = Some((p99, key));
                        }
                    }
                }
                if let Some((_, k)) = best {
                    if !picks.contains(&k) {
                        picks.push(k);
                    }
                }
            }
        }
        let mut res = vec![];
        for &(m, efc, ef) in &picks {
            let mut per = vec![vec![]; 2];
            for b in 0..builds {
                wait_ram(3.0, "lat builds");
                let bl = build_ann(Store::F32Lib, &ds.base, ds.dim, m, efc);
                let bx = build_ann(Store::F32Avx, &ds.base, ds.dim, m, efc);
                let (hl, hx) = (&bl.h, &bx.h);
                let mut fl = |q: &[f32]| hl.search(q, K, ef);
                let mut fx = |q: &[f32]| hx.search(q, K, ef);
                let r = bench_min_multi(&mut [&mut fl, &mut fx], &ds, &gt, K, reps);
                eprintln!("[lat] {} M{m} efc{efc} ef{ef} build{b}: lib rec {:.3} p50min {:.0} p99min {:.0} | avx2 rec {:.3} p50min {:.0} p99min {:.0}", ds.name, r[0].0, r[0].1, r[0].2, r[1].0, r[1].1, r[1].2);
                for k in 0..2 {
                    per[k].push(json!({"recall": r[k].0, "p50_min_us": r[k].1, "p99_min_us": r[k].2, "mean_us": r[k].3, "p99_raw_mean_us": r[k].4,
                        "build_s": if k == 0 { bl.build_s } else { bx.build_s }, "heap_bytes_per_vec": (if k == 0 { bl.heap_bytes } else { bx.heap_bytes }) as f64 / n as f64}));
                }
            }
            res.push(json!({"m": m, "efc": efc, "ef": ef, "lib": per[0], "avx2": per[1]}));
        }
        write_json(&out, &format!("h8_lat_{}{tag}.json", ds.name), &json!({"mode": "lat", "dataset": ds.name, "n": n, "dim": ds.dim, "nq": ds.nq(), "reps": reps, "builds": builds,
            "stat": "p50/p99 of per-query minimum over interleaved passes (load-robust); p99_raw = mean of raw per-pass p99", "results": res, "machine": machine("end")}));
    }
}

// ============================================================ ssd: random-read cost of the f32 rerank file (direct I/O), parallel I/O effect
fn mode_ssd(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let n = a.usize("--n", 300_000);
    let nq = 200;
    let dim = 384usize;
    let tmpdir = "C:/Users/mariu/nm-target-r8/h8_tmp";
    std::fs::create_dir_all(tmpdir).ok();
    let fpath = format!("{tmpdir}/ssd_{n}.f32");
    let (queries, mut st) = synth_stream(dim, n, nq);
    FileRows::write_stream(&fpath, &mut st, dim);
    let fdir = FileRows::open(&fpath, dim, n, true);
    let fbuf = FileRows::open(&fpath, dim, n, false);
    let mut res = vec![];
    // raw random row reads
    let mut rng = synth_rng(7);
    let ids: Vec<usize> = (0..4000).map(|_| rng.below(n)).collect();
    for (nm, direct) in [("direct, ONE shared handle", true), ("direct, one handle per thread", true)] {
        let per_thread = nm.contains("per thread");
        for thr in [1usize, 2, 4, 8] {
            let handles: Vec<FileRows> = (0..if per_thread { thr } else { 1 }).map(|_| FileRows::open(&fpath, dim, n, direct)).collect();
            let t = Instant::now();
            std::thread::scope(|s| {
                for tt in 0..thr {
                    let ids = &ids;
                    let f = &handles[if per_thread { tt } else { 0 }];
                    s.spawn(move || {
                        let mut r = vec![0f32; dim];
                        for i in (tt..ids.len()).step_by(thr) {
                            f.row_into(ids[i], &mut r);
                        }
                    });
                }
            });
            let el = t.elapsed().as_secs_f64();
            println!("ssd raw {nm} threads {thr}: {:.1} us/row (wall/rows), {:.0} rows/s", el * 1e6 / ids.len() as f64, ids.len() as f64 / el);
            res.push(json!({"kind": "raw_random_row_read", "mode": nm, "threads": thr, "us_per_row_wall": el * 1e6 / ids.len() as f64, "rows_per_s": ids.len() as f64 / el}));
        }
    }
    // turbovec 4-bit + rerank4x with SSD rows fetched by 1/2/4 threads
    let (_q2, st2) = synth_stream(dim, n, nq);
    let src = Src::Syn { dim, n, nq };
    drop(st2);
    let (ix, _bt, heap) = tv_from(src, dim, 4);
    let (gtq, gts) = synth_stream(dim, n, nq);
    let gt = ground_truth_stream(gts, &gtq, dim, K, 50_000);
    let ds = empty_ds("ssd", dim, queries);
    let p1 = pool1();
    for thr in [1usize, 2, 4] {
        let handles: Vec<FileRows> = (0..thr).map(|_| FileRows::open(&fpath, dim, n, true)).collect();
        let b = p1.install(|| bench_search(|q| {
            let r = ix.search(q, K * 4);
            let c: Vec<u32> = r.indices_for_query(0).iter().map(|&i| i as u32).collect();
            let mut c = c;
            c.sort_unstable();
            let rows: Vec<Vec<(f32, u32)>> = std::thread::scope(|s| {
                let hs: Vec<_> = (0..thr).map(|tt| {
                    let (c, f) = (&c, &handles[tt]);
                    s.spawn(move || {
                        let mut row = vec![0f32; dim];
                        let mut o = vec![];
                        for i in (tt..c.len()).step_by(thr) {
                            f.row_into(c[i] as usize, &mut row);
                            o.push((dot(&row, q), c[i]));
                        }
                        o
                    })
                }).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut top = TopK::new(K);
            for (sc, id) in rows.into_iter().flatten() {
                top.push(sc, id);
            }
            top.ids()
        }, &ds, &gt, K, 3));
        println!("tv4 + rerank4x, f32 rows on SSD (direct I/O) fetched by {thr} thread(s): recall {:.3} p50 {:.0} us p99 {:.0} us (index heap {:.0} B/vec)", b.recall, mean(&b.p50), mean(&b.p99), heap as f64 / n as f64);
        res.push(json!({"kind": "tv4_rr4_ssd_direct", "io_threads": thr, "bench": b.json(), "index_heap_bytes_per_vec": heap as f64 / n as f64}));
    }
    drop(fdir);
    drop(fbuf);
    std::fs::remove_file(&fpath).ok();
    write_json(&out, "h8_ssd.json", &json!({"mode": "ssd", "n": n, "dim": dim, "rows": res, "machine": machine("end")}));
}
fn synth_rng(seed: u64) -> neural_mesh::memory::synth::Rng {
    neural_mesh::memory::synth::Rng::new(seed)
}

// ============================================================ concins: correctness of CONCURRENT inserts (2 writer threads + 2 readers)
fn mode_concins(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let n = a.usize("--n", 30_000);
    let extra = a.usize("--extra", 6_000);
    let (m, efc, ef) = (a.usize("--m", 32), a.usize("--efc", 100), a.usize("--ef", 32));
    let dim = 384;
    let full = synth_ds(dim, n + extra, 300);
    let base = &full.base[..n * dim];
    let fresh = &full.base[n * dim..];
    let h = build_ann(Store::F32Lib, base, dim, m, efc);
    let hh: &dyn Ann = &*h.h;
    let stop = AtomicBool::new(false);
    let reads = AtomicUsize::new(0);
    let t = Instant::now();
    std::thread::scope(|s| {
        let mut ws = vec![];
        for w in 0..2usize {
            ws.push(s.spawn(move || {
                for i in (w..extra).step_by(2) {
                    hh.insert_one(&fresh[i * dim..(i + 1) * dim], n + i);
                }
            }));
        }
        for r in 0..2usize {
            let (stop, reads, qs) = (&stop, &reads, &full.queries);
            s.spawn(move || {
                let nq = qs.len() / dim;
                let mut i = r * 50;
                while !stop.load(Ordering::Relaxed) {
                    std::hint::black_box(hh.search(&qs[(i % nq) * dim..(i % nq + 1) * dim], K, ef));
                    reads.fetch_add(1, Ordering::Relaxed);
                    i += 1;
                }
            });
        }
        for w in ws {
            w.join().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
    });
    let el = t.elapsed().as_secs_f64();
    eprintln!("concurrent phase {el:.1}s: {} inserts by 2 threads, {} concurrent reads", extra, reads.load(Ordering::Relaxed));
    // check: inserted items retrievable by their own vector; recall vs exact over base+inserted
    let (mut miss1, mut miss10) = (0, 0);
    for i in 0..extra {
        let r = hh.search(&fresh[i * dim..(i + 1) * dim], 10, 64);
        if r.first() != Some(&((n + i) as u32)) {
            miss1 += 1;
        }
        if !r.contains(&((n + i) as u32)) {
            miss10 += 1;
        }
    }
    let (mut b1, mut b10) = (0, 0);
    let nb = 6000.min(n);
    for i in (0..n).step_by((n / nb).max(1)) {
        let r = hh.search(&base[i * dim..(i + 1) * dim], 10, 64);
        if r.first() != Some(&(i as u32)) {
            b1 += 1;
        }
        if !r.contains(&(i as u32)) {
            b10 += 1;
        }
    }
    let ds = Ds { name: "concins".into(), dim, base: full.base.clone(), queries: full.queries.clone(), note: String::new() };
    let gt = ground_truth(&ds.base, dim, &ds.queries, K);
    let r = bench_search(|q| hh.search(q, K, ef), &ds, &gt, K, 1);
    println!("concurrent insert check n={n}+{extra} M{m} efc{efc}: inserted items own-vector not top-1 {miss1}/{extra}, absent from top-10 {miss10}; base items sampled not top-1 {b1}/{nb}; recall@10 (ef {ef}) vs exact over all {} vectors: {:.3}", n + extra, r.recall);
    write_json(&out, "h8_concins.json", &json!({"n": n, "extra": extra, "m": m, "efc": efc, "ef": ef, "inserted_not_top1": miss1, "inserted_absent_top10": miss10,
        "base_sampled": nb, "base_not_top1": b1, "base_absent_top10": b10, "recall_at_10_all": r.recall, "concurrent_phase_s": el, "reads_during": reads.load(Ordering::Relaxed), "machine": machine("end")}));
}

// ============================================================ lat2: fixed grid (M,efc) x ef list, 3 builds, interleaved lib/avx2, per-query-min latency
fn mode_lat2(a: &Args) {
    let out = a.get("--out").unwrap_or("results".into());
    let nq = a.usize("--nq", 300);
    let builds = a.usize("--builds", 3);
    let reps = a.usize("--reps", 7);
    let ms = a.nums("--m", "16,24,32,48");
    let efcs = a.nums("--efc", "100,200");
    let efs = a.nums("--ef", "10,16,24,32");
    let skip_avx = a.has("--no-avx");
    for spec in a.list("--data", "synth384:30000,mini,bge") {
        let ds = load(&spec, nq);
        let gt = ground_truth(&ds.base, ds.dim, &ds.queries, K);
        let n = ds.n();
        let mut res = vec![];
        for &m in &ms {
            for &efc in &efcs {
                if efc < 2 * m {
                    continue;
                }
                let mut per: Vec<Vec<Vec<Value>>> = vec![vec![vec![]; efs.len()]; 2];
                for b in 0..builds {
                    wait_ram(3.0, "lat2 builds");
                    let bl = build_ann(Store::F32Lib, &ds.base, ds.dim, m, efc);
                    let bx = if skip_avx { None } else { Some(build_ann(Store::F32Avx, &ds.base, ds.dim, m, efc)) };
                    for (ei, &ef) in efs.iter().enumerate() {
                        let hl = &bl.h;
                        let mut fl = |q: &[f32]| hl.search(q, K, ef);
                        let r = if let Some(bx) = &bx {
                            let hx = &bx.h;
                            let mut fx = |q: &[f32]| hx.search(q, K, ef);
                            bench_min_multi(&mut [&mut fl, &mut fx], &ds, &gt, K, reps)
                        } else {
                            bench_min_multi(&mut [&mut fl], &ds, &gt, K, reps)
                        };
                        for k in 0..r.len() {
                            per[k][ei].push(json!({"recall": r[k].0, "p50_min_us": r[k].1, "p99_min_us": r[k].2, "mean_us": r[k].3, "p99_raw_mean_us": r[k].4,
                                "build_s": if k == 0 { bl.build_s } else { bx.as_ref().unwrap().build_s }, "heap_bytes_per_vec": (if k == 0 { bl.heap_bytes } else { bx.as_ref().unwrap().heap_bytes }) as f64 / n as f64}));
                        }
                    }
                    eprintln!("[lat2] {} M{m} efc{efc} build{b} done", ds.name);
                }
                for (ei, &ef) in efs.iter().enumerate() {
                    res.push(json!({"m": m, "efc": efc, "ef": ef, "lib": per[0][ei], "avx2": if skip_avx { Value::Null } else { json!(per[1][ei]) }}));
                }
            }
        }
        write_json(&out, &format!("h8_lat2_{}.json", ds.name), &json!({"mode": "lat2", "dataset": ds.name, "n": n, "dim": ds.dim, "nq": ds.nq(), "reps": reps, "builds": builds,
            "stat": "p50/p99 of per-query minimum over interleaved passes (load-robust)", "results": res, "machine": machine("end")}));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a = Args(args.clone());
    let threads = std::env::var("RAYON_NUM_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().ok();
    let mode = args.get(1).cloned().unwrap_or_default();
    eprintln!("h8_vectors mode={mode} rayon={} free={:.2}GB", rayon::current_num_threads(), free_gb());
    match mode.as_str() {
        "selftest" => mode_selftest(),
        "sweep" => mode_sweep(&a),
        "confirm" => mode_confirm(&a),
        "cross" => mode_cross(&a),
        "memory" => mode_memory(&a),
        "scale" => mode_scale(&a),
        "conc" => mode_conc(&a),
        "tvmem" => mode_tvmem(&a),
        "selfcheck" => mode_selfcheck(&a),
        "lat2" => mode_lat2(&a),
        "concins" => mode_concins(&a),
        "ssd" => mode_ssd(&a),
        "lat" => mode_lat(&a),
        _ => eprintln!("modes: selftest sweep confirm cross memory scale conc"),
    }
}
