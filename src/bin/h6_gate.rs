//! H6-gate (R4-B3): size gating of the vector store. Measures flat f32 / block-fp16 / int8 / turbovec 2-4 bit / own RaBitQ-style 1-bit codes,
//! each with and without exact f32 rerank, with the f32 copy either in RAM or on DISK (unbuffered QD1 reads: honest cold SSD cost), on synthetic
//! hierarchical-mixture vectors (g15 generator), 1 thread, plus the gating rule and its validation against the measured grid.
//!   h6_gate grid --n 100000 [--nq 200] [--reps 2] [--out results] [--disk-cold]
//!   h6_gate rule [--out results]        (reads results/h6_gate_grid_*.json, fits the cost model, prints the decision table, validates the rule)
//! RAM rule: aborts N >= 5e5 unless free RAM > 4.2 GB at start (3 GB working-set cap of the task; peak here = f32 base + one store).
//! Synthetic-until-validated-on-real-traffic. No defaults changed.

use std::alloc::{alloc, dealloc, Layout};
use std::fs::File;
use std::io::Write as _;
use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};
use turbovec::TurboQuantIndex;

#[path = "g15_common/mod.rs"]
mod g15;
use g15::*;

#[global_allocator]
static A: Counting = Counting;

// ---------------------------------------------------------------- args
struct Args(Vec<String>);
impl Args {
    fn get(&self, k: &str) -> Option<String> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1).cloned())
    }
    fn has(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
    fn usize(&self, k: &str, d: usize) -> usize {
        self.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
    }
}

// ---------------------------------------------------------------- RaBitQ-style 1-bit codes (own implementation, randomised-Hadamard rotation)
const DP: usize = 512; // padded dim (384 -> 512)
struct Rabit {
    signs1: Vec<f32>,
    signs2: Vec<f32>,
    dim: usize,
    codes: Vec<u8>, // n * 64
    inv: Vec<f32>,  // n : 1 / sum |o'_i|
}
fn fwht(v: &mut [f32]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}
impl Rabit {
    fn rot(&self, x: &[f32], out: &mut [f32]) {
        out.iter_mut().for_each(|o| *o = 0.0);
        for i in 0..self.dim {
            out[i] = x[i] * self.signs1[i];
        }
        fwht(out);
        for i in 0..DP {
            out[i] *= self.signs2[i];
        }
        fwht(out);
        let s = 1.0 / DP as f32; // two unnormalised transforms: scale by 1/DP keeps the norm
        out.iter_mut().for_each(|o| *o *= s);
    }
    fn build(base: &[f32], dim: usize) -> Rabit {
        let mut rng = memory_rng(0xAB17);
        let sg = |r: &mut u64, n: usize| -> Vec<f32> { (0..n).map(|_| { *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); if (*r >> 33) & 1 == 1 { 1.0 } else { -1.0 } }).collect() };
        let signs1 = sg(&mut rng, DP);
        let signs2 = sg(&mut rng, DP);
        let mut rb = Rabit { signs1, signs2, dim, codes: vec![], inv: vec![] };
        let n = base.len() / dim;
        let mut codes = vec![0u8; n * (DP / 8)];
        let mut inv = vec![0f32; n];
        codes.par_chunks_mut(DP / 8).zip(inv.par_iter_mut()).zip(base.par_chunks(dim)).for_each_init(|| vec![0f32; DP], |tmp, ((c, iv), row)| {
            rb.rot(row, tmp);
            let mut l1 = 0.0f32;
            for (b, byte) in c.iter_mut().enumerate() {
                let mut v = 0u8;
                for j in 0..8 {
                    let x = tmp[b * 8 + j];
                    l1 += x.abs();
                    if x >= 0.0 {
                        v |= 1 << j;
                    }
                }
                *byte = v;
            }
            *iv = if l1 > 0.0 { 1.0 / l1 } else { 0.0 };
        });
        rb.codes = codes;
        rb.inv = inv;
        rb
    }
    fn nominal_bytes(&self) -> usize {
        self.codes.len() + self.inv.len() * 4
    }
    /// top-`m` ids by the RaBitQ estimator sum_i s_i q'_i / sum |o'_i|
    fn topm(&self, q: &[f32], m: usize, lut: &mut Vec<f32>, qr: &mut Vec<f32>) -> Vec<u32> {
        qr.resize(DP, 0.0);
        self.rot(q, qr);
        let nb = DP / 8;
        lut.resize(nb * 256, 0.0);
        for b in 0..nb {
            let t = &mut lut[b * 256..(b + 1) * 256];
            let qs = &qr[b * 8..b * 8 + 8];
            t[0] = -qs.iter().sum::<f32>();
            for v in 1..256usize {
                let low = v.trailing_zeros() as usize;
                t[v] = t[v & (v - 1)] + 2.0 * qs[low];
            }
        }
        let mut top = TopK::new(m);
        for (i, c) in self.codes.chunks_exact(nb).enumerate() {
            let mut s = 0.0f32;
            for (b, &byte) in c.iter().enumerate() {
                s += lut[b * 256 + byte as usize];
            }
            top.push(s * self.inv[i], i as u32);
        }
        top.ids()
    }
}
fn memory_rng(seed: u64) -> u64 {
    seed.wrapping_mul(0x9E3779B97F4A7C15) | 1
}

// ---------------------------------------------------------------- disk-resident f32 rows (unbuffered, sector aligned)
struct DiskRows {
    f: File,
    dim: usize,
    unbuffered: bool,
    buf: std::cell::RefCell<(*mut u8, Layout)>,
    fetch_ns: std::cell::Cell<u128>,
    fetches: std::cell::Cell<u64>,
}
impl DiskRows {
    fn open(path: &str, dim: usize, cold: bool) -> DiskRows {
        use std::os::windows::fs::OpenOptionsExt;
        let rb = dim * 4;
        let ok_align = rb % 512 == 0;
        let mut oo = std::fs::OpenOptions::new();
        oo.read(true);
        let unbuffered = cold && ok_align;
        if unbuffered {
            oo.custom_flags(0x2000_0000); // FILE_FLAG_NO_BUFFERING
        }
        let f = oo.open(path).expect("open f32 file");
        let layout = Layout::from_size_align(rb.div_ceil(512) * 512, 4096).unwrap();
        let p = unsafe { alloc(layout) };
        DiskRows { f, dim, unbuffered, buf: std::cell::RefCell::new((p, layout)), fetch_ns: 0.into(), fetches: 0.into() }
    }
    fn read(&self, i: usize, out: &mut [f32]) {
        use std::os::windows::fs::FileExt;
        let t = Instant::now();
        let rb = self.dim * 4;
        let (p, l) = *self.buf.borrow();
        let sl = unsafe { std::slice::from_raw_parts_mut(p, l.size()) };
        let n = self.f.seek_read(&mut sl[..l.size().min(rb.div_ceil(512) * 512)], (i * rb) as u64).expect("seek_read");
        assert!(n >= rb, "short read");
        for (o, c) in out.iter_mut().zip(sl[..rb].chunks_exact(4)) {
            *o = f32::from_le_bytes(c.try_into().unwrap());
        }
        self.fetch_ns.set(self.fetch_ns.get() + t.elapsed().as_nanos());
        self.fetches.set(self.fetches.get() + 1);
    }
    fn rerank(&self, q: &[f32], cands: &[u32], k: usize) -> Vec<u32> {
        let mut row = vec![0f32; self.dim];
        let mut top = TopK::new(k);
        for &c in cands {
            self.read(c as usize, &mut row);
            top.push(dot(&row, q), c);
        }
        top.ids()
    }
}
impl Drop for DiskRows {
    fn drop(&mut self) {
        let (p, l) = *self.buf.borrow();
        unsafe { dealloc(p, l) };
    }
}

fn write_f32(path: &str, base: &[f32]) {
    let mut f = std::io::BufWriter::with_capacity(1 << 22, File::create(path).unwrap());
    for ch in base.chunks(1 << 20) {
        let b: &[u8] = unsafe { std::slice::from_raw_parts(ch.as_ptr() as *const u8, ch.len() * 4) };
        f.write_all(b).unwrap();
    }
    f.flush().unwrap();
}

fn pool1() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap()
}

struct Row {
    name: String,
    group: &'static str,
    ram_bytes_per_vec: f64,
    exact_ram_bytes_per_vec: f64,
    exact_disk_bytes_per_vec: f64,
    build_s: f64,
    bench: Bench,
    disk_fetch_us_per_query: f64,
    note: String,
}
impl Row {
    fn json(&self, n: usize) -> Value {
        json!({"config": self.name, "group": self.group, "n": n, "ram_bytes_per_vec": self.ram_bytes_per_vec, "exact_copy_in_ram_bytes_per_vec": self.exact_ram_bytes_per_vec, "exact_copy_on_disk_bytes_per_vec": self.exact_disk_bytes_per_vec,
            "ram_total_MB": (self.ram_bytes_per_vec + self.exact_ram_bytes_per_vec) * n as f64 / 1e6, "build_s": self.build_s, "result": self.bench.json(), "disk_fetch_us_per_query": self.disk_fetch_us_per_query, "note": self.note})
    }
}

fn mode_grid(a: &Args) {
    let n = a.usize("--n", 100_000);
    let nq = a.usize("--nq", 200);
    let reps = a.usize("--reps", 2);
    let out = a.get("--out").unwrap_or("results".into());
    let cold = a.has("--disk-cold");
    let dim = 384usize;
    let fg = free_gb();
    println!("N={n} dim={dim} free RAM at start {fg:.2} GB, cold-disk={cold}");
    let min_free: f64 = a.get("--min-free-gb").and_then(|v| v.parse().ok()).unwrap_or(4.2);
    if n >= 500_000 && fg < min_free {
        eprintln!("ABORT: free RAM {fg:.2} GB < {min_free} GB for N>=5e5 (3 GB working-set cap + headroom)");
        std::process::exit(2);
    }
    let scratch = a.get("--scratch").unwrap_or_else(|| std::env::var("TEMP").unwrap_or(".".into()));
    let path = format!("{scratch}\\h6_f32_{n}.bin");
    let t0 = Instant::now();
    let ds = synth_ds(dim, n, nq);
    println!("data {:.1}s, base {:.0} MB", t0.elapsed().as_secs_f64(), ds.base.len() as f64 * 4.0 / 1e6);
    write_f32(&path, &ds.base);
    let calib0 = calib_ms();
    let t0 = Instant::now();
    let gt = ground_truth(&ds.base, dim, &ds.queries, 10);
    println!("ground truth {:.1}s", t0.elapsed().as_secs_f64());
    let k = 10usize;
    let mut rows: Vec<Row> = vec![];
    let p1 = pool1();
    let disk = DiskRows::open(&path, dim, cold);
    let tmp = std::cell::RefCell::new(Vec::<f32>::new());
    let mk = |name: &str, group: &'static str, ram: f64, ex_ram: f64, ex_disk: f64, build_s: f64, bench: Bench, fetch: f64, note: &str| Row { name: name.into(), group, ram_bytes_per_vec: ram, exact_ram_bytes_per_vec: ex_ram, exact_disk_bytes_per_vec: ex_disk, build_s, bench, disk_fetch_us_per_query: fetch, note: note.into() };
    let n_q_total = |reps: usize| ((reps + 1) * nq) as f64;
    // ---- flat f32
    {
        let b = bench_search(|q| flat_topk(&ds.base, dim, q, k, false), &ds, &gt, k, reps);
        rows.push(mk("flat f32 (AVX2)", "flat", 1536.0, 0.0, 0.0, 0.0, b, 0.0, "exact; resident f32"));
    }
    // ---- low-precision flat stores (build one at a time)
    for (prec, nm) in [(Prec::BlockFp16, "flat block-fp16"), (Prec::Int8, "flat int8")] {
        let t = Instant::now();
        let st = LowStore::build(prec, &ds.base, dim);
        let bs = t.elapsed().as_secs_f64();
        let by = st.nominal_bytes() as f64 / n as f64;
        let b = bench_search(|q| st.topk(q, k, None, &mut tmp.borrow_mut()), &ds, &gt, k, reps);
        rows.push(mk(nm, "flat", by, 0.0, 0.0, bs, b, 0.0, "no f32 needed"));
        if prec == Prec::Int8 {
            for f in [4usize, 10] {
                let b = bench_search(|q| st.topk(q, k, Some((&ds.base, f)), &mut tmp.borrow_mut()), &ds, &gt, k, reps);
                rows.push(mk(&format!("int8 + f32 rerank {f}x (f32 in RAM)"), "rerank-ram", by, 1536.0, 0.0, bs, b, 0.0, "f32 resident"));
            }
            for f in [4usize] {
                disk.fetch_ns.set(0);
                disk.fetches.set(0);
                let b = bench_search(|q| { let c = st.topk(q, k * f, None, &mut tmp.borrow_mut()); disk.rerank(q, &c, k) }, &ds, &gt, k, reps);
                let fu = disk.fetch_ns.get() as f64 / 1e3 / n_q_total(reps);
                rows.push(mk(&format!("int8 + f32 rerank {f}x (f32 on disk{})", if disk.unbuffered { ", unbuffered" } else { ", OS-cached" }), "rerank-disk", by, 0.0, 1536.0, bs, b, fu, "QD1 random reads"));
            }
        }
        drop(st);
    }
    // ---- turbovec
    for (bits, fs) in [(4usize, vec![4usize]), (2, vec![4, 16])] {
        let t = Instant::now();
        let mut ix = TurboQuantIndex::new(dim, bits).unwrap();
        p1.install(|| {
            for ch in ds.base.chunks(dim * 2048) {
                ix.add(ch);
            }
            ix.prepare();
        });
        let bs = t.elapsed().as_secs_f64();
        let by = ((dim * bits).div_ceil(8) + 4) as f64;
        let search = |q: &[f32], fetch: usize| -> Vec<u32> { p1.install(|| ix.search(q, fetch).indices_for_query(0).iter().map(|&i| i as u32).collect()) };
        let b = bench_search(|q| search(q, k), &ds, &gt, k, reps);
        rows.push(mk(&format!("turbovec-{bits}bit"), "codes", by, 0.0, 0.0, bs, b, 0.0, "no rerank"));
        for &f in &fs {
            let b = bench_search(|q| rerank_f32(&ds.base, dim, q, &search(q, k * f), k), &ds, &gt, k, reps);
            rows.push(mk(&format!("turbovec-{bits}bit + f32 rerank {f}x (f32 in RAM)"), "rerank-ram", by, 1536.0, 0.0, bs, b, 0.0, "f32 resident"));
            disk.fetch_ns.set(0);
            disk.fetches.set(0);
            let b = bench_search(|q| disk.rerank(q, &search(q, k * f), k), &ds, &gt, k, reps);
            let fu = disk.fetch_ns.get() as f64 / 1e3 / n_q_total(reps);
            rows.push(mk(&format!("turbovec-{bits}bit + f32 rerank {f}x (f32 on disk{})", if disk.unbuffered { ", unbuffered" } else { ", OS-cached" }), "rerank-disk", by, 0.0, 1536.0, bs, b, fu, "QD1 random reads"));
        }
        drop(ix);
    }
    // ---- RaBitQ-style 1-bit (own)
    {
        let t = Instant::now();
        let rb = Rabit::build(&ds.base, dim);
        let bs = t.elapsed().as_secs_f64();
        let by = rb.nominal_bytes() as f64 / n as f64;
        let (lut, qr) = (std::cell::RefCell::new(vec![]), std::cell::RefCell::new(vec![]));
        let top = |q: &[f32], m: usize| rb.topm(q, m, &mut lut.borrow_mut(), &mut qr.borrow_mut());
        let b = bench_search(|q| top(q, k), &ds, &gt, k, reps);
        rows.push(mk("RaBitQ-style 1-bit (own)", "codes", by, 0.0, 0.0, bs, b, 0.0, "randomised-Hadamard rotation, LUT scan (not SIMD fast-scan)"));
        for f in [16usize, 32] {
            let b = bench_search(|q| rerank_f32(&ds.base, dim, q, &top(q, k * f), k), &ds, &gt, k, reps);
            rows.push(mk(&format!("RaBitQ-style 1-bit + f32 rerank {f}x (f32 in RAM)"), "rerank-ram", by, 1536.0, 0.0, bs, b, 0.0, "f32 resident"));
        }
        disk.fetch_ns.set(0);
        disk.fetches.set(0);
        let f = 32usize;
        let b = bench_search(|q| disk.rerank(q, &top(q, k * f), k), &ds, &gt, k, reps);
        let fu = disk.fetch_ns.get() as f64 / 1e3 / n_q_total(reps);
        rows.push(mk(&format!("RaBitQ-style 1-bit + f32 rerank {f}x (f32 on disk{})", if disk.unbuffered { ", unbuffered" } else { ", OS-cached" }), "rerank-disk", by, 0.0, 1536.0, bs, b, fu, "QD1 random reads"));
    }
    let calib1 = calib_ms();
    let (ws, peak) = rss_mb();
    println!("{:<62} {:>8} {:>8} {:>9} {:>9} {:>9} {:>8}", "config", "B/vec", "+f32", "recall@10", "p50 us", "p99 us", "fetch us");
    for r in &rows {
        let mut p50 = r.bench.p50.clone();
        let mut p99 = r.bench.p99.clone();
        println!("{:<62} {:>8.0} {:>8.0} {:>9.3} {:>9.0} {:>9.0} {:>8.0}", r.name, r.ram_bytes_per_vec, r.exact_ram_bytes_per_vec + r.exact_disk_bytes_per_vec, r.bench.recall, pct(&mut p50, 0.5), pct(&mut p99, 0.5), r.disk_fetch_us_per_query);
    }
    println!("peak working set {peak:.0} MB (ws {ws:.0} MB); calib {calib0:.2}/{calib1:.2} ms");
    let js = json!({"n": n, "dim": dim, "nq": nq, "reps": reps, "threads": 1, "free_gb_at_start": fg, "peak_ws_MB": peak, "calib_ms": [calib0, calib1], "disk_unbuffered": disk.unbuffered, "disk_file": path, "data": ds.note, "rows": rows.iter().map(|r| r.json(n)).collect::<Vec<_>>()});
    std::fs::write(format!("{out}/h6_gate_grid_{n}.json"), serde_json::to_string_pretty(&js).unwrap()).unwrap();
    drop(disk);
    let _ = std::fs::remove_file(&path);
}

fn main() {
    let a = Args(std::env::args().skip(1).collect());
    match a.0.first().map(|s| s.as_str()) {
        Some("grid") => mode_grid(&a),
        Some("rule") => rule::mode_rule(&a),
        _ => eprintln!("usage: h6_gate grid --n N [--nq 200] [--reps 2] [--disk-cold] | rule"),
    }
}

mod rule {
    use super::*;

    /// candidate backends in the order of preference (least compression / least machinery first)
    pub const ORDER: [(&str, &str); 10] = [
        ("flat f32", "flat f32 (AVX2)"),
        ("flat block-fp16", "flat block-fp16"),
        ("int8 + f32 rerank 4x (f32 RAM)", "int8 + f32 rerank 4x (f32 in RAM)"),
        ("int8 + f32 rerank 4x (f32 disk)", "int8 + f32 rerank 4x (f32 on disk"),
        ("turbovec-4bit + f32 rerank 4x (f32 RAM)", "turbovec-4bit + f32 rerank 4x (f32 in RAM)"),
        ("turbovec-4bit + f32 rerank 4x (f32 disk)", "turbovec-4bit + f32 rerank 4x (f32 on disk"),
        ("turbovec-2bit + f32 rerank 16x (f32 RAM)", "turbovec-2bit + f32 rerank 16x (f32 in RAM)"),
        ("turbovec-2bit + f32 rerank 16x (f32 disk)", "turbovec-2bit + f32 rerank 16x (f32 on disk"),
        ("RaBitQ-1bit + f32 rerank 32x (f32 RAM)", "RaBitQ-style 1-bit + f32 rerank 32x (f32 in RAM)"),
        ("RaBitQ-1bit + f32 rerank 32x (f32 disk)", "RaBitQ-style 1-bit + f32 rerank 32x (f32 on disk"),
    ];
    #[derive(Clone, Debug)]
    pub struct Pt {
        pub n: f64,
        pub p50: f64,
        pub recall: f64,
        pub ram_b: f64,
        pub ex_ram: f64,
    }
    pub struct Cand {
        pub label: &'static str,
        pub pts: Vec<Pt>,
    }
    fn mean_of(v: &Value) -> f64 {
        v["mean"].as_f64().unwrap()
    }
    pub fn load(ns: &[usize], out: &str) -> Vec<Cand> {
        let mut c: Vec<Cand> = ORDER.iter().map(|(l, _)| Cand { label: l, pts: vec![] }).collect();
        for &n in ns {
            let p = format!("{out}/h6_gate_grid_{n}.json");
            let Ok(t) = std::fs::read_to_string(&p) else { continue };
            let v: Value = serde_json::from_str(&t).unwrap();
            for r in v["rows"].as_array().unwrap() {
                let name = r["config"].as_str().unwrap();
                for (ci, (_, prefix)) in ORDER.iter().enumerate() {
                    if name.starts_with(prefix) {
                        c[ci].pts.push(Pt { n: n as f64, p50: mean_of(&r["result"]["p50_us"]), recall: r["result"]["recall_at_10"].as_f64().unwrap(), ram_b: r["ram_bytes_per_vec"].as_f64().unwrap(), ex_ram: r["exact_copy_in_ram_bytes_per_vec"].as_f64().unwrap() });
                    }
                }
            }
        }
        c
    }
    /// least-squares line p50 = a + b n over the measured points (1 point -> proportional)
    pub fn lat(c: &Cand, n: f64) -> f64 {
        let k = c.pts.len() as f64;
        if c.pts.len() == 1 {
            return c.pts[0].p50 * n / c.pts[0].n;
        }
        let (sx, sy) = (c.pts.iter().map(|p| p.n).sum::<f64>(), c.pts.iter().map(|p| p.p50).sum::<f64>());
        let sxx = c.pts.iter().map(|p| p.n * p.n).sum::<f64>();
        let sxy = c.pts.iter().map(|p| p.n * p.p50).sum::<f64>();
        let b = (k * sxy - sx * sy) / (k * sxx - sx * sx);
        let a = (sy - b * sx) / k;
        (a + b * n).max(0.0)
    }
    pub fn recall(c: &Cand, n: f64) -> f64 {
        // recall of the nearest measured N (not extrapolated upward)
        c.pts.iter().min_by(|a, b| (a.n.ln() - n.ln()).abs().partial_cmp(&(b.n.ln() - n.ln()).abs()).unwrap()).map(|p| p.recall).unwrap_or(0.0)
    }
    pub fn ram(c: &Cand, n: f64) -> f64 {
        c.pts.last().map(|p| p.ram_b + p.ex_ram).unwrap_or(0.0) * n
    }
    /// THE RULE: the least-compressed backend whose RAM <= ram_budget (bytes), modelled p50 <= lat_budget (us), recall >= target.
    pub fn choose(cs: &[Cand], n: f64, ram_budget: f64, lat_us: f64, target: f64) -> Option<usize> {
        (0..cs.len()).find(|&i| !cs[i].pts.is_empty() && ram(&cs[i], n) <= ram_budget && lat(&cs[i], n) <= lat_us && recall(&cs[i], n) >= target)
    }
    pub fn mode_rule(a: &Args) {
        let out = a.get("--out").unwrap_or("results".into());
        let cs = load(&[10_000, 100_000, 1_000_000], &out);
        let target = 0.99;
        let mut md = String::new();
        md.push_str("## Fitted cost model (p50 us = a + b N, single thread, rows from results/h6_gate_grid_*.json)\n");
        for c in &cs {
            if c.pts.is_empty() {
                continue;
            }
            md.push_str(&format!("- {:<44} RAM B/vec {:>6.0}  p50 @1e4/1e5/1e6 = {}  recall = {}  model p50(1e7) = {:.0} us (EXTRAPOLATED)\n", c.label, c.pts.last().unwrap().ram_b + c.pts.last().unwrap().ex_ram,
                c.pts.iter().map(|p| format!("{:.0}", p.p50)).collect::<Vec<_>>().join("/"), c.pts.iter().map(|p| format!("{:.3}", p.recall)).collect::<Vec<_>>().join("/"), lat(c, 1e7)));
        }
        let ram_b = [("128 MB", 128e6), ("512 MB", 512e6), ("2 GB", 2e9)];
        let lat_b = [("1 ms", 1000.0), ("5 ms", 5000.0), ("20 ms", 20000.0)];
        md.push_str("\n## Rule output: least-compressed backend with RAM <= budget, modelled p50 <= budget, recall >= 0.99 (f32 copy of rerank formats is ON DISK; its fetch time is inside p50)\n\n| N | latency budget | RAM 128 MB | RAM 512 MB | RAM 2 GB |\n|---|---|---|---|---|\n");
        let mut table: Vec<Value> = vec![];
        for n in [1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7] {
            for (ln, lv) in lat_b {
                let mut row = format!("| {n:.0e}{} | {ln} |", if n > 1e6 { " (extrap.)" } else { "" });
                for (rn, rv) in ram_b {
                    let ch = choose(&cs, n, rv, lv, target);
                    let nm = ch.map(|i| cs[i].label).unwrap_or("NONE (needs ANN graph / more RAM / more latency)");
                    row.push_str(&format!(" {nm} |"));
                    table.push(json!({"n": n, "lat_budget_us": lv, "ram_budget_B": rv, "pick": nm, "extrapolated": n > 1e6}));
                }
                md.push_str(&row);
                md.push('\n');
            }
        }
        // validation against MEASURED rows at the three grid sizes
        md.push_str("\n## Validation: pick vs. measured grid (N = 1e4, 1e5, 1e6; measured p50/recall/RAM of the picked row; agreement with 'least-compressed measured-feasible')\n\n| N | lat budget | RAM budget | rule pick | measured p50 us | measured recall | feasible by measurement | measured-least-compressed | agree |\n|---|---|---|---|---|---|---|---|---|\n");
        let (mut ok, mut tot) = (0, 0);
        for n in [1e4, 1e5, 1e6] {
            for (ln, lv) in lat_b {
                for (rn, rv) in ram_b {
                    let pick = choose(&cs, n, rv, lv, target);
                    let meas = |c: &Cand| c.pts.iter().find(|p| p.n == n).cloned();
                    let truth = (0..cs.len()).find(|&i| meas(&cs[i]).is_some_and(|p| (p.ram_b + p.ex_ram) * n <= rv && p.p50 <= lv && p.recall >= target));
                    let (mp, mr, feas) = match pick.and_then(|i| meas(&cs[i])) {
                        Some(p) => (format!("{:.0}", p.p50), format!("{:.3}", p.recall), (p.ram_b + p.ex_ram) * n <= rv && p.p50 <= lv && p.recall >= target),
                        None => ("-".into(), "-".into(), false),
                    };
                    let agree = pick == truth;
                    tot += 1;
                    if agree {
                        ok += 1;
                    }
                    md.push_str(&format!("| {n:.0e} | {ln} | {rn} | {} | {mp} | {mr} | {} | {} | {} |\n", pick.map(|i| cs[i].label).unwrap_or("NONE"), if pick.is_none() { "-".to_string() } else { feas.to_string() }, truth.map(|i| cs[i].label).unwrap_or("NONE"), agree));
                }
            }
        }
        md.push_str(&format!("\nRule vs measured-least-compressed agreement: {ok}/{tot} cells\n"));
        println!("{md}");
        std::fs::write(format!("{out}/h6_gate_rule.md"), &md).unwrap();
        std::fs::write(format!("{out}/h6_gate_rule.json"), serde_json::to_string_pretty(&json!({"target_recall": target, "agreement": [ok, tot], "table": table})).unwrap()).unwrap();
    }
}
