//! H1 / part 1: HNSW wrapper-level re-bench (self-recall, recall@10 vs brute force, build time, latency,
//! downstream lessons/cases retrievability). Run the SAME source against the patched build (repo as is)
//! and an unpatched copy (Cargo.toml without [patch]); `--label` tags the rows.
//!   h1_hnsw --label patched --n 2000 --dim 32 --mode iid --seeds 10 > results/h1_hnsw_patched_n2000_d32_iid.csv
//!   h1_hnsw --summarize a.csv b.csv ...      (mean per label/path/cell)
//! SYNTHETIC data only (iid Gaussian / unit-cluster mixtures).
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use neural_mesh::casebased::{CaseConfig, CaseRouter};
use neural_mesh::lessons::{LessonConfig, LessonStore};
use neural_mesh::memory::synth::Rng as SRng;
use neural_mesh::memory::{dot, normalized, FlatIndex, HnswIndex, HnswParams, MemoryIndex};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::tiered::{HnswRm, LongKind, RemovableIndex, TieredMemory};
use neural_mesh::workmem::store::{IndexKind, Store};

fn arg<T: std::str::FromStr>(k: &str, d: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn arg_s(k: &str, d: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned().unwrap_or(d.to_string())
}

fn gen_data(mode: &str, n: usize, dim: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut r = SRng::new(seed.wrapping_mul(7919).wrapping_add(13));
    let nc = if mode == "clust" { (n / 50).max(2) } else { 1 };
    let cents: Vec<Vec<f32>> = (0..nc).map(|_| normalized(&(0..dim).map(|_| r.normal()).collect::<Vec<_>>())).collect();
    let mut data = vec![];
    let mut lab = vec![];
    for i in 0..n {
        let v: Vec<f32> = if mode == "clust" {
            let c = r.below(nc);
            lab.push(c);
            (0..dim).map(|j| cents[c][j] + 0.8 * r.normal() / (dim as f32).sqrt()).collect()
        } else {
            lab.push(i % 3);
            (0..dim).map(|_| r.normal()).collect()
        };
        data.push(normalized(&v));
    }
    (data, lab)
}

struct Row {
    path: String,
    live: usize,
    self_miss: usize,
    rec10: f64,
    lat_us: f64,
    build_s: f64,
    extra: String,
}

/// metrics over a search closure on the `live` ids (index into `data`).
fn measure(
    search: &dyn Fn(&[f32], usize) -> Vec<(u64, f32)>,
    data: &[Vec<f32>],
    live: &[usize],
    queries: &[Vec<f32>],
) -> (usize, f64, f64) {
    let self_miss = live.iter().filter(|&&i| search(&data[i], 1).first().map(|x| x.0) != Some(i as u64)).count();
    let mut rec = 0.0;
    for q in queries {
        let mut sc: Vec<(f32, usize)> = live.iter().map(|&i| (dot(q, &data[i]), i)).collect();
        sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let truth: std::collections::HashSet<u64> = sc.iter().take(10).map(|x| x.1 as u64).collect();
        let got = search(q, 10);
        rec += got.iter().filter(|g| truth.contains(&g.0)).count() as f64 / 10.0;
    }
    let t0 = Instant::now();
    let mut sink = 0usize;
    for q in queries {
        sink += search(q, 10).len();
    }
    let lat = t0.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
    std::hint::black_box(sink);
    (self_miss, rec / queries.len() as f64, lat)
}

fn main() {
    if std::env::args().any(|a| a == "--summarize") {
        return summarize();
    }
    let label = arg_s("--label", "x");
    let n: usize = arg("--n", 2000);
    let dim: usize = arg("--dim", 32);
    let seeds: u64 = arg("--seeds", 10);
    let seed0: u64 = arg("--seed0", 1);
    let nq: usize = arg("--nq", 100);
    let mode = arg_s("--mode", "iid");
    println!("label,mode,n,dim,seed,path,live,self_miss,recall10,lat_us,build_s,extra");
    let p = HnswParams { max_elements: n, ..Default::default() };
    for seed in seed0..seed0 + seeds {
        let (all, lab_all) = gen_data(&mode, n + nq, dim, seed);
        let (data, queries) = all.split_at(n);
        let (data, queries) = (data.to_vec(), queries.to_vec());
        let lab = &lab_all[..n];
        let ids: Vec<u64> = (0..n as u64).collect();
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        let all_live: Vec<usize> = (0..n).collect();
        let mut rows: Vec<Row> = vec![];

        // A HnswIndex.add
        let t0 = Instant::now();
        let mut a = HnswIndex::new(dim, p);
        for (i, v) in data.iter().enumerate() {
            a.add(i as u64, v);
        }
        let bs = t0.elapsed().as_secs_f64();
        let (m, r, l) = measure(&|q, k| a.search(q, k), &data, &all_live, &queries);
        rows.push(Row { path: "HnswIndex.add".into(), live: n, self_miss: m, rec10: r, lat_us: l, build_s: bs, extra: String::new() });

        // B HnswIndex.add_batch (parallel)
        let t0 = Instant::now();
        let mut b = HnswIndex::new(dim, p);
        b.add_batch(&ids, &flat);
        let bs = t0.elapsed().as_secs_f64();
        let (m, r, l) = measure(&|q, k| b.search(q, k), &data, &all_live, &queries);
        rows.push(Row { path: "HnswIndex.add_batch".into(), live: n, self_miss: m, rec10: r, lat_us: l, build_s: bs, extra: String::new() });

        // C HnswRm.add ; D remove 10% + compact
        let t0 = Instant::now();
        let mut c = HnswRm::new(dim, p);
        for (i, v) in data.iter().enumerate() {
            c.add(i as u64, v);
        }
        let bs = t0.elapsed().as_secs_f64();
        let (m, r, l) = measure(&|q, k| c.search(q, k), &data, &all_live, &queries);
        rows.push(Row { path: "HnswRm.add".into(), live: n, self_miss: m, rec10: r, lat_us: l, build_s: bs, extra: String::new() });
        for i in (0..n).step_by(10) {
            c.remove(i as u64);
        }
        let t0 = Instant::now();
        c.compact();
        let cs = t0.elapsed().as_secs_f64();
        let live2: Vec<usize> = (0..n).filter(|i| i % 10 != 0).collect();
        let (m, r, l) = measure(&|q, k| c.search(q, k), &data, &live2, &queries);
        rows.push(Row { path: "HnswRm.remove10+compact".into(), live: live2.len(), self_miss: m, rec10: r, lat_us: l, build_s: cs, extra: "build_s=compact only".into() });

        // E workmem Store IndexKind::Hnsw
        let dir = std::env::temp_dir().join(format!("h1_store_{}_{}_{}", std::process::id(), seed, n));
        let _ = std::fs::remove_dir_all(&dir);
        let t0 = Instant::now();
        let mut s = Store::open_with(&dir, dim, IndexKind::Hnsw, true, true, p).unwrap();
        for (i, v) in data.iter().enumerate() {
            s.append(i as u64, b"x", v).unwrap();
        }
        let bs = t0.elapsed().as_secs_f64();
        let (m, r, l) = measure(&|q, k| s.search(q, k), &data, &all_live, &queries);
        rows.push(Row { path: "workmem.Store(Hnsw)".into(), live: n, self_miss: m, rec10: r, lat_us: l, build_s: bs, extra: String::new() });
        drop(s);
        let _ = std::fs::remove_dir_all(&dir);

        // F TieredMemory long=Hnsw (what LessonStore / Knowledge::new_tiered sit on)
        let mut tc = neural_mesh::tiered::TieredConfig::new(dim);
        tc.long = LongKind::Hnsw(p);
        let t0 = Instant::now();
        let mut tm = TieredMemory::new(tc.clone());
        for (i, v) in data.iter().enumerate() {
            tm.add(i as u64, v, "m");
        }
        let bs = t0.elapsed().as_secs_f64();
        let (m, r, l) = measure(&|q, k| tm.search_ro(q, k), &data, &all_live, &queries);
        rows.push(Row { path: "TieredMemory(long=Hnsw)".into(), live: n, self_miss: m, rec10: r, lat_us: l, build_s: bs, extra: String::new() });

        // G LessonStore (long=Hnsw): lessons that would fire (own id within top-k=8 of its own task vector)
        let mut lc = LessonConfig::new(dim);
        lc.memory.long = LongKind::Hnsw(p);
        lc.memory.consolidate_every = 128;
        let t0 = Instant::now();
        let mut ls = LessonStore::new(lc);
        for (i, v) in data.iter().enumerate() {
            ls.record_failure(v, i % 8, "t", "x");
        }
        let bs = t0.elapsed().as_secs_f64();
        let mut unfired = 0usize;
        for (i, v) in data.iter().enumerate() {
            let w = ls.warning_scores(v, 8);
            if !(w[i % 8] > 0.0) {
                unfired += 1;
            }
        }
        let (m, r, l) = measure(&|q, k| ls.memory().search_ro(q, k), &data, &all_live, &queries);
        rows.push(Row { path: "LessonStore(long=Hnsw)".into(), live: n, self_miss: m, rec10: r, lat_us: l, build_s: bs, extra: format!("lessons_never_firing_for_own_agent={unfired}") });

        // H CaseRouter::with_index(HnswIndex): own-case retrievability (k=1, agent = case id) + decision agreement vs flat
        let dat = Arc::new(all.clone());
        let emb: neural_mesh::casebased::Embedder = {
            let d = dat.clone();
            Arc::new(move |t: &Task| d.get(t.id as usize).cloned())
        };
        let mut c1 = CaseConfig::default();
        c1.k = 1;
        let t0 = Instant::now();
        let mut cr = CaseRouter::with_index(n, Box::new(HnswIndex::new(dim, p)), c1, emb.clone());
        for i in 0..n {
            cr.update(&Task { id: i as u64, task_type: None, features: vec![] }, i, 1.0);
        }
        let bs = t0.elapsed().as_secs_f64();
        let miss_case = (0..n).filter(|&i| !(cr.neighbour_counts(&data[i]).0[i].0 > 0.0)).count();
        // decision agreement: 3 agents, reward 1 iff agent == label%3 ; k=30 default config
        let mut rg = SRng::new(seed ^ 0xABCD);
        let outcomes: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let a = rg.below(3);
                (a, if a == lab[i] % 3 { 1.0 } else { 0.0 })
            })
            .collect();
        let mut chn = CaseRouter::with_index(3, Box::new(HnswIndex::new(dim, p)), CaseConfig::default(), emb.clone());
        let mut cfl = CaseRouter::new(3, dim, CaseConfig::default(), emb.clone());
        for i in 0..n {
            let t = Task { id: i as u64, task_type: None, features: vec![] };
            chn.update(&t, outcomes[i].0, outcomes[i].1);
            cfl.update(&t, outcomes[i].0, outcomes[i].1);
        }
        let (mut agree, mut dsum, mut acc_h, mut acc_f) = (0usize, 0.0, 0usize, 0usize);
        for qi in 0..nq {
            let t = Task { id: (n + qi) as u64, task_type: None, features: vec![] };
            let ph = chn.posteriors(&t, &[0, 1, 2]);
            let pf = cfl.posteriors(&t, &[0, 1, 2]);
            let mean = |p: &(f64, f64)| p.0 / (p.0 + p.1);
            let am = |p: &Vec<(f64, f64)>| (0..3).max_by(|&x, &y| mean(&p[x]).partial_cmp(&mean(&p[y])).unwrap()).unwrap();
            let (ah, af) = (am(&ph), am(&pf));
            if ah == af {
                agree += 1;
            }
            let truth = lab_all[n + qi] % 3;
            acc_h += (ah == truth) as usize;
            acc_f += (af == truth) as usize;
            dsum += (0..3).map(|a| (mean(&ph[a]) - mean(&pf[a])).abs()).sum::<f64>() / 3.0;
        }
        let _ = Rng::new(0);
        rows.push(Row {
            path: "CaseRouter.with_index(Hnsw)".into(),
            live: n,
            self_miss: miss_case,
            rec10: f64::NAN,
            lat_us: f64::NAN,
            build_s: bs,
            extra: format!("argmax_agree_vs_flat={}/{nq};mean_abs_dpost={:.5};acc_hnsw={}/{nq};acc_flat={}/{nq}", agree, dsum / nq as f64, acc_h, acc_f),
        });
        let _ = FlatIndex::new(1);

        for r in rows {
            println!("{},{},{},{},{},{},{},{},{:.4},{:.1},{:.3},{}", label, mode, n, dim, seed, r.path, r.live, r.self_miss, r.rec10, r.lat_us, r.build_s, r.extra);
        }
    }
}

fn summarize() {
    // key: (n,dim,mode,path) -> label -> columns
    let files: Vec<String> = std::env::args().skip_while(|a| a != "--summarize").skip(1).collect();
    let mut acc: BTreeMap<(String, String, String, String, String), Vec<(usize, usize, f64, f64, f64)>> = BTreeMap::new();
    for f in files {
        let txt = std::fs::read_to_string(&f).unwrap();
        for line in txt.lines().skip(1) {
            let c: Vec<&str> = line.splitn(12, ',').collect();
            if c.len() < 11 || c[0] == "label" {
                continue;
            }
            let live: usize = c[6].parse().unwrap_or(0);
            let sm: usize = c[7].parse().unwrap_or(0);
            let (r, l, b) = (c[8].parse().unwrap_or(f64::NAN), c[9].parse().unwrap_or(f64::NAN), c[10].parse().unwrap_or(f64::NAN));
            acc.entry((format!("n{}_d{}_{}", c[2], c[3], c[1]), c[5].to_string(), c[0].to_string(), String::new(), String::new())).or_default().push((live, sm, r, l, b));
        }
    }
    println!("cell | path | label | builds | mean self_miss (% of live) | builds with miss>0 | recall@10 | lat us | build s");
    for (k, v) in &acc {
        let nb = v.len() as f64;
        let sm = v.iter().map(|x| x.1 as f64).sum::<f64>() / nb;
        let lv = v.iter().map(|x| x.0 as f64).sum::<f64>() / nb;
        let bm = v.iter().filter(|x| x.1 > 0).count();
        let mean = |f: &dyn Fn(&(usize, usize, f64, f64, f64)) -> f64| v.iter().map(|x| f(x)).sum::<f64>() / nb;
        println!("{} | {} | {} | {} | {:.2} ({:.3}%) | {}/{} | {:.4} | {:.0} | {:.3}", k.0, k.1, k.2, v.len(), sm, 100.0 * sm / lv, bm, v.len(), mean(&|x| x.2), mean(&|x| x.3), mean(&|x| x.4));
    }
}
