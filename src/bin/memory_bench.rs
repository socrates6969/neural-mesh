//! Shared-memory benchmark: build time, query latency (mean/p99), recall@10
//! vs exact FlatIndex, measured heap bytes, and 8-reader throughput.
//!
//!   cargo run --release --bin memory_bench [-- --quick]
//!
//! Writes results/memory.json and prints a table.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use neural_mesh::memory::{
    synth::Clustered, recall_at_k, DynIndex, FlatIndex, HnswIndex, HnswParams, MemoryIndex,
    SharedMemory, TurbovecIndex,
};
use serde::Serialize;

// ---- heap accounting -------------------------------------------------------

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size(), Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) };
        LIVE.fetch_sub(l.size(), Ordering::Relaxed);
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            LIVE.fetch_add(l.size(), Ordering::Relaxed);
        }
        p
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
            LIVE.fetch_add(new, Ordering::Relaxed);
        }
        q
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

// ---- config & results ------------------------------------------------------

const DIM: usize = 384;
const NQ: usize = 1000;
const K: usize = 10;
const READERS: usize = 8;
const SEED: u64 = 42;

#[derive(Serialize, Clone)]
struct Row {
    n: usize,
    backend: String,
    build_s: f64,
    mean_us: f64,
    p50_us: f64,
    p99_us: f64,
    /// Same, but letting turbovec use the global rayon pool for one query.
    mean_us_pool: f64,
    p99_us_pool: f64,
    recall_at_10: f64,
    heap_bytes_measured: usize,
    bytes_estimate: usize,
    bytes_per_vector: f64,
    qps_1_thread: f64,
    qps_8_readers: f64,
}

#[derive(Serialize)]
struct Report {
    dim: usize,
    queries: usize,
    k: usize,
    reader_threads: usize,
    seed: u64,
    data: String,
    cpu_threads: usize,
    notes: Vec<String>,
    rows: Vec<Row>,
}

// ---- measurement helpers ---------------------------------------------------

fn pct(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i]
}

/// Per-query latencies (µs) and results.
fn run_queries(idx: &dyn MemoryIndex, qs: &[f32]) -> (Vec<f64>, Vec<Vec<(u64, f32)>>) {
    // warm-up
    for q in qs.chunks_exact(DIM).take(20) {
        std::hint::black_box(idx.search(q, K));
    }
    let mut lat = Vec::with_capacity(NQ);
    let mut res = Vec::with_capacity(NQ);
    for q in qs.chunks_exact(DIM) {
        let t = Instant::now();
        let r = idx.search(q, K);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
        res.push(r);
    }
    (lat, res)
}

fn stats(mut lat: Vec<f64>) -> (f64, f64, f64) {
    let mean = lat.iter().sum::<f64>() / lat.len() as f64;
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (mean, pct(&lat, 0.5), pct(&lat, 0.99))
}

fn throughput(mem: &SharedMemory, qs: &[f32]) -> f64 {
    let t = Instant::now();
    std::thread::scope(|s| {
        for r in 0..READERS {
            let mem = mem.clone();
            s.spawn(move || {
                let nq = qs.len() / DIM;
                for i in 0..nq {
                    let j = (i + r * nq / READERS) % nq;
                    std::hint::black_box(mem.search(&qs[j * DIM..(j + 1) * DIM], K));
                }
            });
        }
    });
    (READERS * NQ) as f64 / t.elapsed().as_secs_f64()
}

struct Built {
    idx: DynIndex,
    build_s: f64,
    heap: usize,
}

fn build(make: impl FnOnce() -> DynIndex, ids: &[u64], data: &[f32]) -> Built {
    let before = live();
    let t = Instant::now();
    let mut idx = make();
    idx.add_batch(ids, data);
    let build_s = t.elapsed().as_secs_f64();
    let heap = live().saturating_sub(before);
    Built { idx, build_s, heap }
}

fn evaluate(
    n: usize,
    b: Built,
    qs: &[f32],
    truth: &[Vec<(u64, f32)>],
    serial: &rayon::ThreadPool,
) -> Row {
    let mem = SharedMemory::new(b.idx);
    evaluate_shared(n, &mem, b.build_s, b.heap, qs, truth, serial)
}

fn evaluate_shared(
    n: usize,
    mem: &SharedMemory,
    build_s: f64,
    heap: usize,
    qs: &[f32],
    truth: &[Vec<(u64, f32)>],
    serial: &rayon::ThreadPool,
) -> Row {
    let name = mem.name();
    eprintln!("  [{n}] {name}: built in {build_s:.2}s, measuring...");
    let (lat, res) = {
        let g = mem.read();
        serial.install(|| run_queries(g.as_ref(), qs))
    };
    let (mean, p50, p99) = stats(lat);
    let (lat_pool, _) = run_queries(mem.read().as_ref(), qs);
    let (mean_pool, _, p99_pool) = stats(lat_pool);
    let recall = res.iter().zip(truth).map(|(g, t)| recall_at_k(t, g, K)).sum::<f64>() / NQ as f64;
    let qps8 = throughput(mem, qs);
    Row {
        n,
        backend: name,
        build_s,
        mean_us: mean,
        p50_us: p50,
        p99_us: p99,
        mean_us_pool: mean_pool,
        p99_us_pool: p99_pool,
        recall_at_10: recall,
        heap_bytes_measured: heap,
        bytes_estimate: mem.bytes(),
        bytes_per_vector: heap as f64 / n as f64,
        qps_1_thread: 1e6 / mean,
        qps_8_readers: qps8,
    }
}

fn main() {
    let quick = std::env::args().any(|a| a == "--quick");
    let sizes: &[usize] = if quick { &[10_000] } else { &[10_000, 100_000] };
    let serial = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let cpu_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut rows = Vec::new();

    for &n in sizes {
        eprintln!("== N = {n}, dim = {DIM}");
        // 64 topics x 16 sub-topics; data and queries from the same mixture.
        let mut g = Clustered::hierarchical(DIM, 64, 16, 0.8, 0.6, SEED);
        let data = g.sample(n);
        let qs = g.sample(NQ);
        let ids: Vec<u64> = (0..n as u64).collect();

        // Ground truth = FlatIndex (also benchmarked as a backend).
        let flat = build(|| Box::new(FlatIndex::with_capacity(DIM, n)), &ids, &data);
        let truth: Vec<Vec<(u64, f32)>> =
            qs.chunks_exact(DIM).map(|q| flat.idx.search(q, K)).collect();
        rows.push(evaluate(n, flat, &qs, &truth, &serial));

        let makers: Vec<Box<dyn Fn() -> DynIndex>> = vec![
            Box::new(|| Box::new(TurbovecIndex::new(DIM, 4))),
            Box::new(|| Box::new(TurbovecIndex::new(DIM, 2))),
            Box::new(|| Box::new(TurbovecIndex::with_rerank(DIM, 4, 4))),
            Box::new(|| Box::new(TurbovecIndex::with_rerank(DIM, 2, 4))),
        ];
        for mk in makers {
            let b = build(mk, &ids, &data);
            rows.push(evaluate(n, b, &qs, &truth, &serial));
        }

        // HNSW: build once, sweep ef_search through the runtime handle.
        let hp = HnswParams { max_elements: n, ..Default::default() };
        let before = live();
        let t = Instant::now();
        let mut h = HnswIndex::new(DIM, hp);
        h.add_batch(&ids, &data);
        let ef = h.ef_handle();
        let built = Built { build_s: t.elapsed().as_secs_f64(), heap: live().saturating_sub(before), idx: Box::new(h) };
        let mem = SharedMemory::new(built.idx);
        for e in [32usize, 64, 128] {
            ef.store(e, std::sync::atomic::Ordering::Relaxed);
            rows.push(evaluate_shared(n, &mem, built.build_s, built.heap, &qs, &truth, &serial));
        }
    }

    // ---- table ----
    println!(
        "\n{:>7} {:<24} {:>8} {:>9} {:>9} {:>10} {:>8} {:>10} {:>8} {:>10}",
        "N", "backend", "build_s", "mean_us", "p99_us", "mean_pool", "recall", "heap_MB", "B/vec", "qps_8rd"
    );
    for r in &rows {
        println!(
            "{:>7} {:<24} {:>8.2} {:>9.1} {:>9.1} {:>10.1} {:>8.3} {:>10.1} {:>8.0} {:>10.0}",
            r.n,
            r.backend,
            r.build_s,
            r.mean_us,
            r.p99_us,
            r.mean_us_pool,
            r.recall_at_10,
            r.heap_bytes_measured as f64 / 1e6,
            r.bytes_per_vector,
            r.qps_8_readers
        );
    }

    let report = Report {
        dim: DIM,
        queries: NQ,
        k: K,
        reader_threads: READERS,
        seed: SEED,
        data: "hierarchical gaussian mixture: 64 topics x 16 sub-topics (sub_noise 0.8), point noise 0.6, L2-normalised; queries are fresh draws from the same mixture".into(),
        cpu_threads,
        notes: vec![
            "mean/p50/p99_us: single query, measured inside a 1-thread rayon pool (turbovec otherwise splits one query across cores at N>=~32k)".into(),
            "mean_us_pool/p99_us_pool: same queries with the global rayon pool (only turbovec uses it)".into(),
            "heap_bytes_measured: live heap delta across index construction + add_batch (counting global allocator); input data excluded".into(),
            "bytes_estimate: the index's own MemoryIndex::bytes() estimate".into(),
            "qps_8_readers: 8 OS threads sharing one SharedMemory (RwLock read side), 1000 queries each".into(),
            "recall_at_10: mean |exact top10 ∩ returned top10| / 10 vs FlatIndex".into(),
            "release build, default target (no target-cpu=native): FlatIndex dot kernel is SSE2; turbovec does runtime AVX2 dispatch".into(),
        ],
        rows: rows.clone(),
    };
    std::fs::create_dir_all("results").ok();
    std::fs::write("results/memory.json", serde_json::to_string_pretty(&report).unwrap()).unwrap();
    println!("\nwrote results/memory.json");
}
