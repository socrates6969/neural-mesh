//! R3 part 2: persistent on-disk shared memory store (append-only log + snapshot
//! + flat/HNSW index; std only, no mmap). Measures latency (p50/p99), recall@k vs
//! exact, bytes on disk, warm-open vs cold replay, crash safety (random truncation
//! and bit flips) and multi-reader sanity (1 writer, 4 concurrent readers).
//!
//! Usage: cargo run --release --bin r3_store [-- --out results/r3_memory_spatial.json] [--n 50000]
//! All timings are with a warm OS file cache (the files were just written), so
//! "open" numbers are CPU+memcpy bound, not disk bound; say so when quoting them.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use neural_mesh::memory::{recall_at_k, synth, HnswParams};
use neural_mesh::router::Rng;
use neural_mesh::workmem::store::{IndexKind, Store};
use serde_json::{json, Value};

fn merge_json(path: &str, key: &str, v: Value) {
    let mut root: Value = std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_else(|| json!({}));
    root[key] = v;
    if let Some(d) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(d);
    }
    std::fs::write(path, serde_json::to_string_pretty(&root).unwrap()).unwrap();
}

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn median_ms<F: FnMut() -> f64>(mut f: F, reps: usize) -> f64 {
    let mut v: Vec<f64> = (0..reps).map(|_| f()).collect();
    pct(&mut v, 0.5)
}

fn rvec(seed: u64, dim: usize) -> Vec<f32> {
    let mut r = Rng::new(seed.wrapping_mul(2_654_435_761).wrapping_add(99));
    (0..dim).map(|_| r.f64() as f32 - 0.5).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/r3_memory_spatial.json".into());
    let n: usize = args.iter().position(|a| a == "--n").and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(30_000);
    let dim = 128usize;
    let nq = 1000usize;
    let root: PathBuf = std::env::temp_dir().join(format!("r3_store_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    println!("r3_store: N={n} dim={dim}, {nq} queries, dir {}", root.display());

    // data: clustered unit-ish vectors (same generator the memory lab uses)
    let mut cl = synth::Clustered::new(dim, 64, 0.6, 7);
    let data = cl.sample(n + nq);
    let (base, queries) = data.split_at(n * dim);

    // ------------------------------------------------------------ append + disk
    let d_flat = root.join("flat");
    let t0 = Instant::now();
    let mut w = Store::open(&d_flat, dim, IndexKind::Flat, true).unwrap();
    for i in 0..n {
        w.append(i as u64, format!("doc-{i}").as_bytes(), &base[i * dim..(i + 1) * dim]).unwrap();
    }
    let t_append = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    w.sync().unwrap();
    let t_fsync = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    w.checkpoint().unwrap();
    let t_ckpt = t0.elapsed().as_secs_f64();
    let log_bytes = w.log_bytes();
    let snap_bytes = w.snapshot_bytes();
    let raw_bytes = (n * dim * 4) as u64;
    drop(w);
    println!(
        "append {:.0} rec/s ({:.2}s, fsync {:.3}s, checkpoint {:.3}s); log {} B ({:.1} B/vec, raw f32 {} B, overhead {:.2}%), snapshot {} B",
        n as f64 / t_append, t_append, t_fsync, t_ckpt, log_bytes, log_bytes as f64 / n as f64, raw_bytes,
        100.0 * (log_bytes as f64 / raw_bytes as f64 - 1.0), snap_bytes
    );

    // ------------------------------------------------------------ open times
    let open_ms = |kind: IndexKind, snap: bool, dir: &PathBuf| -> f64 {
        let t = Instant::now();
        let s = Store::open_with(dir, dim, kind, false, snap, HnswParams::default()).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(s.len(), n);
        ms
    };
    let cold_flat = median_ms(|| open_ms(IndexKind::Flat, false, &d_flat), 5);
    let warm_flat = median_ms(|| open_ms(IndexKind::Flat, true, &d_flat), 5);
    println!("open flat: cold replay {:.1} ms, warm snapshot {:.1} ms ({:.1}x)", cold_flat, warm_flat, cold_flat / warm_flat);

    // HNSW store: build once by appending (sequential insert), then time the opens
    let d_h = root.join("hnsw");
    let t0 = Instant::now();
    let mut wh = Store::open(&d_h, dim, IndexKind::Hnsw, true).unwrap();
    for i in 0..n {
        wh.append(i as u64, b"", &base[i * dim..(i + 1) * dim]).unwrap();
    }
    wh.checkpoint().unwrap();
    let t_hbuild = t0.elapsed().as_secs_f64();
    drop(wh);
    let cold_h = median_ms(|| open_ms(IndexKind::Hnsw, false, &d_h), 2);
    let warm_h = median_ms(|| open_ms(IndexKind::Hnsw, true, &d_h), 2);
    println!("open hnsw: build-by-append {:.1}s, cold replay (sequential rebuild) {:.0} ms, warm snapshot (parallel batch rebuild) {:.0} ms", t_hbuild, cold_h, warm_h);

    // ------------------------------------------------------------ latency + recall
    let flat = Store::open(&d_flat, dim, IndexKind::Flat, false).unwrap();
    let k = 10usize;
    let truth: Vec<Vec<(u64, f32)>> = (0..nq).map(|i| flat.search_exact(&queries[i * dim..(i + 1) * dim], k)).collect();
    let mut flat_lat: Vec<f64> = (0..nq)
        .map(|i| {
            let t = Instant::now();
            let r = flat.search(&queries[i * dim..(i + 1) * dim], k);
            std::hint::black_box(r);
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    let (fp50, fp99) = (pct(&mut flat_lat, 0.5), pct(&mut flat_lat, 0.99));
    println!("flat exact: p50 {:.0} us p99 {:.0} us, recall@10 = 1.000 (by definition), resident {} B", fp50, fp99, flat.resident_bytes());

    let mut jh = vec![];
    for &ef in &[16usize, 32, 64, 128, 256] {
        let p = HnswParams { ef_search: ef, ..HnswParams::default() };
        let hs = Store::open_with(&d_h, dim, IndexKind::Hnsw, false, true, p).unwrap();
        let mut lat = Vec::with_capacity(nq);
        let mut rec = 0.0;
        for i in 0..nq {
            let q = &queries[i * dim..(i + 1) * dim];
            let t = Instant::now();
            let r = hs.search(q, k);
            lat.push(t.elapsed().as_secs_f64() * 1e6);
            rec += recall_at_k(&truth[i], &r, k);
        }
        let (p50, p99) = (pct(&mut lat, 0.5), pct(&mut lat, 0.99));
        println!("hnsw ef={ef:<4} p50 {:.0} us p99 {:.0} us recall@10 {:.3} (speedup vs flat p50 {:.1}x) resident ~{} B", p50, p99, rec / nq as f64, fp50 / p50, hs.resident_bytes());
        jh.push(json!({"ef_search": ef, "p50_us": p50, "p99_us": p99, "recall_at_10": rec / nq as f64, "speedup_p50_vs_flat": fp50 / p50, "resident_bytes": hs.resident_bytes()}));
    }

    // ------------------------------------------------------------ crash safety
    let trials = 300usize;
    let (mut ok_trunc, mut ok_flip) = (0usize, 0usize);
    let cdim = 16usize;
    for t in 0..trials {
        let dir = root.join(format!("crash{t}"));
        let mut w = Store::open(&dir, cdim, IndexKind::Flat, true).unwrap();
        let mut ends = vec![0u64];
        let nrec = 60;
        for i in 0..nrec {
            w.append(i as u64, &vec![7u8; (i * 3) % 40], &rvec(i as u64, cdim)).unwrap();
            ends.push(w.log_bytes());
        }
        drop(w);
        let p = dir.join("log.bin");
        let total = *ends.last().unwrap();
        let mut rng = Rng::new(1000 + t as u64);
        if t % 2 == 0 {
            // truncation at a random byte
            let cut = rng.below(total as usize + 1) as u64;
            std::fs::OpenOptions::new().write(true).open(&p).unwrap().set_len(cut).unwrap();
            let expect = ends.iter().filter(|&&e| e <= cut).count() - 1;
            let r = Store::open(&dir, cdim, IndexKind::Flat, false).unwrap();
            let mut good = r.len() == expect;
            if expect > 0 {
                good &= r.search(&rvec((expect - 1) as u64, cdim), 1)[0].0 == (expect - 1) as u64;
            }
            drop(r);
            // writer repairs and can append
            let mut wr = Store::open(&dir, cdim, IndexKind::Flat, true).unwrap();
            good &= wr.len() == expect;
            wr.append(9999, b"", &rvec(9999, cdim)).unwrap();
            drop(wr);
            good &= Store::open(&dir, cdim, IndexKind::Flat, false).unwrap().len() == expect + 1;
            if good {
                ok_trunc += 1;
            }
        } else {
            // flip one random bit somewhere in the file
            let mut b = std::fs::read(&p).unwrap();
            let pos = rng.below(b.len());
            b[pos] ^= 1 << rng.below(8);
            std::fs::write(&p, &b).unwrap();
            let rec_idx = ends.iter().rposition(|&e| e <= pos as u64).unwrap(); // record containing pos
            let r = Store::open(&dir, cdim, IndexKind::Flat, false).unwrap();
            // valid prefix must stop exactly at the damaged record (CRC32 detects every single-bit flip)
            if r.len() == rec_idx {
                ok_flip += 1;
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    println!("crash safety: truncation {}/{} recovered exactly, single-bit-flip {}/{} detected at the damaged record", ok_trunc, trials / 2, ok_flip, trials / 2);

    // ------------------------------------------------------------ multi-reader
    let d_mr = root.join("mr");
    let total = 20_000usize;
    let mdim = 32usize;
    let writer_done = Arc::new(AtomicBool::new(false));
    let written = Arc::new(AtomicUsize::new(0));
    let w = Store::open(&d_mr, mdim, IndexKind::Flat, true).unwrap();
    let mut wopt = Some(w);
    let (wd, wc) = (writer_done.clone(), written.clone());
    let wt = {
        let mut w = wopt.take().unwrap();
        std::thread::spawn(move || {
            for i in 0..total {
                w.append(i as u64, b"", &rvec(i as u64, mdim)).unwrap();
                wc.store(i + 1, Ordering::Release);
            }
            w.sync().unwrap();
            wd.store(true, Ordering::Release);
            w
        })
    };
    let readers: Vec<_> = (0..4)
        .map(|rid| {
            let (wd, wc, dir) = (writer_done.clone(), written.clone(), d_mr.clone());
            std::thread::spawn(move || {
                let mut r = Store::open(&dir, mdim, IndexKind::Flat, false).unwrap();
                let mut rng = Rng::new(50 + rid as u64);
                let (mut prev, mut violations, mut checks, mut refreshes) = (r.len(), 0usize, 0usize, 0usize);
                loop {
                    let done = wd.load(Ordering::Acquire);
                    r.refresh().unwrap();
                    refreshes += 1;
                    let len = r.len();
                    if len < prev || len > wc.load(Ordering::Acquire) + 1 {
                        violations += 1; // length must be monotone and never exceed what the writer finished
                    }
                    prev = len;
                    for _ in 0..3 {
                        if len > 0 {
                            let j = rng.below(len);
                            checks += 1;
                            if r.search(&rvec(j as u64, mdim), 1)[0].0 != j as u64 {
                                violations += 1;
                            }
                        }
                    }
                    if done && len == total {
                        break;
                    }
                }
                (violations, checks, refreshes, r.len())
            })
        })
        .collect();
    let mut mr = vec![];
    let mut all_clean = true;
    for r in readers {
        let (v, c, rf, l) = r.join().unwrap();
        all_clean &= v == 0 && l == total;
        mr.push(json!({"violations": v, "search_checks": c, "refreshes": rf, "final_len": l}));
    }
    let _ = wt.join().unwrap();
    println!("multi-reader: 1 writer ({total} appends) + 4 concurrent readers: clean={all_clean} {:?}", mr);

    let _ = std::fs::remove_dir_all(&root);
    let j = json!({
        "n": n, "dim": dim, "queries": nq,
        "mmap": "not used (std has no mmap; no new crates). 'warm open' = checkpoint snapshot + log-tail replay",
        "note_timings": "warm OS file cache; open times are CPU/memcpy bound",
        "disk": {"log_bytes": log_bytes, "bytes_per_vector": log_bytes as f64 / n as f64, "raw_f32_bytes": raw_bytes,
                 "log_overhead_pct": 100.0 * (log_bytes as f64 / raw_bytes as f64 - 1.0), "snapshot_bytes": snap_bytes},
        "append": {"records_per_s": n as f64 / t_append, "fsync_s": t_fsync, "checkpoint_s": t_ckpt},
        "open_ms": {"flat_cold_replay": cold_flat, "flat_warm_snapshot": warm_flat, "hnsw_cold_replay": cold_h, "hnsw_warm_snapshot": warm_h,
                    "hnsw_build_by_append_s": t_hbuild,
                    "caveat": "HNSW graph itself is not persisted; warm open still rebuilds the graph (parallel batch insert) from the snapshot vectors"},
        "flat_exact": {"p50_us": fp50, "p99_us": fp99, "resident_bytes": flat.resident_bytes()},
        "hnsw": jh,
        "crash_safety": {"trials_each": trials / 2, "truncation_recovered_exactly": ok_trunc, "bitflip_detected_at_damaged_record": ok_flip},
        "multi_reader": {"writer_appends": total, "readers": mr, "all_clean": all_clean},
    });
    merge_json(&out, "store", j);
    println!("wrote {out} (key store)");
}
