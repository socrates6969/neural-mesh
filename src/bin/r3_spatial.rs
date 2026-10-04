//! R3 part 3: spatial memory. (a) correctness-checked benchmark of the k-d tree and
//! uniform grid against brute force (nearest, k-NN, box range, radius) at several N
//! in 2D and 3D; A* vs BFS on grid maps; AABB layout overlap (sweep-and-prune vs
//! brute). (b) Optional "visual-spatial tasks as router features": does appending
//! layout-derived spatial features to task features change routing success?
//!
//! Protocol for (b): tune baselines on seeds 1-8, evaluate paired on held-out seeds
//! 31-60 (95% t-CI, Holm over routers x worlds); informative world (best agent depends
//! on the layout class) vs uninformative control world (layout irrelevant).
//!
//! Usage: cargo run --release --bin r3_spatial [-- --out results/r3_memory_spatial.json]

use std::collections::HashMap;
use std::time::Instant;

use neural_mesh::router::{Rng, Task};
use neural_mesh::spatial::*;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use neural_mesh::workmem::ctxworld::Stream;
use neural_mesh::workmem::harness::*;
use rayon::prelude::*;
use serde_json::{json, Value};

fn merge_json(path: &str, key: &str, v: Value) {
    let mut root: Value = std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_else(|| json!({}));
    root[key] = v;
    if let Some(d) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(d);
    }
    std::fs::write(path, serde_json::to_string_pretty(&root).unwrap()).unwrap();
}

fn us<F: FnMut()>(mut f: F, reps: usize) -> f64 {
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() * 1e6 / reps as f64
}

fn bench_dim<const D: usize>(n: usize, clustered: bool, nq: usize, seed: u64) -> Value {
    let mut rng = Rng::new(seed);
    let side = 1000.0;
    let pts: Vec<Pt<D>> = (0..n)
        .map(|_| {
            let mut p = [0.0; D];
            if clustered {
                // 50 gaussian-ish blobs
                let c = (rng.f64() * 50.0).floor();
                let mut cr = Rng::new(c as u64 + 1);
                for i in 0..D {
                    p[i] = cr.f64() * side + (rng.f64() + rng.f64() + rng.f64() - 1.5) * 20.0;
                }
            } else {
                for i in 0..D {
                    p[i] = rng.f64() * side;
                }
            }
            p
        })
        .collect();
    let qs: Vec<Pt<D>> = (0..nq)
        .map(|_| {
            let mut p = [0.0; D];
            for i in 0..D {
                p[i] = rng.f64() * side;
            }
            p
        })
        .collect();
    // range boxes ~ target 20 expected hits (uniform): edge = side * (20/n)^(1/D)
    let edge = side * (20.0 / n as f64).powf(1.0 / D as f64);
    let t = Instant::now();
    let kd = KdTree::build(&pts);
    let t_kd = t.elapsed().as_secs_f64() * 1e3;
    let cell = Grid::<D>::auto_cell(&pts, 2.0);
    let t = Instant::now();
    let gr = Grid::build(&pts, cell);
    let t_gr = t.elapsed().as_secs_f64() * 1e3;

    // correctness on every query (distances for k-NN, exact sets for range/radius)
    let mut ok = true;
    for q in &qs {
        let b = brute_knn(&pts, q, 10);
        for r in [kd.knn(q, 10), gr.knn(q, 10)] {
            ok &= r.len() == b.len() && r.iter().zip(&b).all(|(x, y)| (x.1 - y.1).abs() <= 1e-9 * (1.0 + y.1));
        }
        let mut lo = *q;
        let mut hi = *q;
        for i in 0..D {
            hi[i] += edge;
            lo[i] -= 0.0;
        }
        let br = brute_range(&pts, &lo, &hi);
        ok &= kd.range(&lo, &hi) == br && gr.range(&lo, &hi) == br;
        let rad = edge * 0.6;
        let brr = brute_radius(&pts, q, rad);
        ok &= kd.radius(q, rad) == brr && gr.radius(q, rad) == brr;
    }

    let reps = nq;
    let mut qi = 0usize;
    let mut next = || {
        qi = (qi + 1) % qs.len();
        qs[qi]
    };
    let mut res = serde_json::Map::new();
    macro_rules! timeit {
        ($name:expr, $brute:expr, $kd:expr, $gr:expr) => {{
            let tb = us(|| { let q = next(); std::hint::black_box(($brute)(&q)); }, reps);
            let tk = us(|| { let q = next(); std::hint::black_box(($kd)(&q)); }, reps);
            let tg = us(|| { let q = next(); std::hint::black_box(($gr)(&q)); }, reps);
            res.insert($name.to_string(), json!({"brute_us": tb, "kdtree_us": tk, "grid_us": tg, "speedup_kdtree": tb / tk, "speedup_grid": tb / tg}));
            println!("  {:<10} brute {:>10.1} us | kd {:>9.2} us ({:>7.1}x) | grid {:>9.2} us ({:>7.1}x)", $name, tb, tk, tb / tk, tg, tb / tg);
        }};
    }
    println!("D={D} N={n} {} (kd build {:.1} ms, grid build {:.1} ms, correct={ok})", if clustered { "clustered" } else { "uniform" }, t_kd, t_gr);
    timeit!("nearest", |q: &Pt<D>| brute_nearest(&pts, q), |q: &Pt<D>| kd.nearest(q), |q: &Pt<D>| gr.nearest(q));
    timeit!("knn10", |q: &Pt<D>| brute_knn(&pts, q, 10), |q: &Pt<D>| kd.knn(q, 10), |q: &Pt<D>| gr.knn(q, 10));
    let box_of = |q: &Pt<D>| {
        let mut hi = *q;
        for i in 0..D {
            hi[i] += edge;
        }
        hi
    };
    timeit!("range", |q: &Pt<D>| brute_range(&pts, q, &box_of(q)), |q: &Pt<D>| kd.range(q, &box_of(q)), |q: &Pt<D>| gr.range(q, &box_of(q)));
    let rad = edge * 0.6;
    timeit!("radius", |q: &Pt<D>| brute_radius(&pts, q, rad), |q: &Pt<D>| kd.radius(q, rad), |q: &Pt<D>| gr.radius(q, rad));
    json!({"d": D, "n": n, "clustered": clustered, "correct": ok, "kd_build_ms": t_kd, "grid_build_ms": t_gr,
           "kd_bytes": kd.bytes(), "grid_bytes": gr.bytes(), "raw_bytes": n * D * 8, "queries": res})
}

// ---------------------------------------------------------------- spatial features for routing

struct SpWorld {
    stream_plain: Stream,
    stream_sp: Stream,
}

fn gen_sp_world(seed: u64, informative: bool) -> SpWorld {
    let (topics, agents, n_tasks) = (8usize, 6usize, 1500usize);
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(13));
    let draw = |rng: &mut Rng| -> Vec<f64> {
        let best = rng.below(agents);
        let mut second = rng.below(agents);
        while second == best {
            second = rng.below(agents);
        }
        (0..agents).map(|a| if a == best { 0.85 } else if a == second { 0.5 } else { 0.15 + 0.25 * rng.f64() }).collect()
    };
    let base: Vec<Vec<f64>> = (0..topics).map(|_| draw(&mut rng)).collect();
    // p[topic][class]: informative world re-draws ~70% of (topic,class) cells
    let mut p: Vec<Vec<Vec<f64>>> = vec![vec![vec![]; 3]; topics];
    for t in 0..topics {
        for c in 0..3 {
            p[t][c] = if informative && rng.f64() < 0.7 { draw(&mut rng) } else { base[t].clone() };
        }
    }
    let argmax = |v: &Vec<f64>| (0..v.len()).max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap()).unwrap();
    let mut kwmap = HashMap::new();
    for t in 0..topics {
        for j in 0..3 {
            let best = argmax(&base[t]);
            let a = if rng.f64() < 0.1 { rng.below(agents) } else { best };
            kwmap.insert(format!("t{t}k{j}"), a);
        }
    }
    let mk = |ns: usize| Stream { tasks: vec![], p: vec![], u: vec![], revisit: vec![], kwmap: kwmap.clone(), n_agents: ns };
    let (mut plain, mut sp) = (mk(agents), mk(agents));
    for i in 0..n_tasks {
        let t = rng.below(topics);
        let class = rng.below(3);
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(0.25) {
            let o = (t + 1 + rng.below(topics - 1)) % topics;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        // layout whose box count / overlap density depends on `class` (a real geometric computation)
        let (nb, maxsz) = match class {
            0 => (10, 5.0),
            1 => (30, 11.0),
            _ => (55, 18.0),
        };
        let boxes: Vec<Aabb<2>> = (0..nb)
            .map(|_| {
                let lo = [rng.f64() * 100.0, rng.f64() * 100.0];
                Aabb::new(lo, [lo[0] + 1.0 + rng.f64() * maxsz, lo[1] + 1.0 + rng.f64() * maxsz])
            })
            .collect();
        let mut fs = f.clone();
        fs.extend(layout_features(&boxes));
        let u = rng.f64();
        for (s, feats) in [(&mut plain, f), (&mut sp, fs)] {
            s.tasks.push(Task { id: i as u64, task_type: None, features: feats });
            s.p.push(p[t][class].clone());
            s.u.push(u);
            s.revisit.push(false);
        }
    }
    SpWorld { stream_plain: plain, stream_sp: sp }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/r3_memory_spatial.json".into());

    // ------------------------------------------------------------ (a) index benchmarks
    println!("== (a) spatial index benchmarks (single thread, correctness-checked vs brute force) ==");
    let mut jb = vec![];
    for &n in &[1_000usize, 10_000, 100_000, 1_000_000] {
        let nq = if n >= 1_000_000 { 100 } else { 300 };
        jb.push(bench_dim::<2>(n, false, nq, 11));
        jb.push(bench_dim::<3>(n, false, nq, 12));
    }
    jb.push(bench_dim::<2>(100_000, true, 300, 13));
    jb.push(bench_dim::<3>(100_000, true, 300, 14));

    // A*
    println!("\n== path length (4-connected grid, Manhattan A* vs BFS) ==");
    let mut ja = vec![];
    for &(w, density) in &[(256usize, 0.20), (512, 0.20), (512, 0.30)] {
        let mut rng = Rng::new(5);
        let m = GridMap::random(w, w, density, &mut rng, &[(0, 0), (w - 1, w - 1)]);
        let nq = 50;
        let pairs: Vec<((usize, usize), (usize, usize))> = (0..nq)
            .map(|_| {
                let mut p = (rng.below(w), rng.below(w));
                let mut g = (rng.below(w), rng.below(w));
                while m.blocked[p.1 * w + p.0] {
                    p = (rng.below(w), rng.below(w));
                }
                while m.blocked[g.1 * w + g.0] {
                    g = (rng.below(w), rng.below(w));
                }
                (p, g)
            })
            .collect();
        let mut agree = true;
        let mut exp_sum = 0usize;
        let mut reach = 0;
        for &(s, g) in &pairs {
            let b = m.bfs_len(s, g);
            let a = m.astar_len(s, g);
            agree &= b == a.map(|x| x.0);
            if let Some((_, e)) = a {
                exp_sum += e;
                reach += 1;
            }
        }
        let tb = us(|| { for &(s, g) in &pairs { std::hint::black_box(m.bfs_len(s, g)); } }, 3) / nq as f64;
        let ta = us(|| { for &(s, g) in &pairs { std::hint::black_box(m.astar_len(s, g)); } }, 3) / nq as f64;
        let free = m.blocked.iter().filter(|b| !**b).count();
        println!("  {w}x{w} dens {density}: BFS {tb:.0} us/query, A* {ta:.0} us/query ({:.1}x), agree={agree}, reachable {reach}/{nq}, A* expands {:.0} of {free} free cells (mean)", tb / ta, exp_sum as f64 / reach.max(1) as f64);
        ja.push(json!({"size": w, "density": density, "bfs_us": tb, "astar_us": ta, "speedup": tb / ta, "agree": agree, "reachable": reach, "mean_expanded": exp_sum as f64 / reach.max(1) as f64, "free_cells": free}));
    }

    // AABB overlap
    println!("\n== AABB layout overlap (sweep-and-prune vs brute) ==");
    let mut jo = vec![];
    for &n in &[500usize, 2000, 8000] {
        let mut rng = Rng::new(9);
        let b: Vec<Aabb<2>> = (0..n)
            .map(|_| {
                let lo = [rng.f64() * 1000.0, rng.f64() * 1000.0];
                Aabb::new(lo, [lo[0] + rng.f64() * 6.0, lo[1] + rng.f64() * 6.0])
            })
            .collect();
        let br = overlap_pairs_brute(&b);
        let sw = overlap_pairs_sweep(&b);
        let tb = us(|| { std::hint::black_box(overlap_pairs_brute(&b)); }, 3);
        let ts = us(|| { std::hint::black_box(overlap_pairs_sweep(&b)); }, 3);
        println!("  n={n}: pairs {} agree={} brute {:.0} us, sweep {:.0} us ({:.1}x)", br.len(), br == sw, tb, ts, tb / ts);
        jo.push(json!({"n": n, "pairs": br.len(), "agree": br == sw, "brute_us": tb, "sweep_us": ts, "speedup": tb / ts}));
    }

    // ------------------------------------------------------------ (b) router features
    println!("\n== (b) layout-derived spatial features as router features ==");
    let tune: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = (31..=60).collect();
    let tune_obj = |k: &Kind, informative: bool| -> f64 {
        let r: Vec<f64> = tune.par_iter().map(|&s| run(k, &gen_sp_world(s, informative).stream_plain, s).mean).collect();
        r.iter().sum::<f64>() / r.len() as f64
    };
    let pick = |c: Vec<Kind>| -> Kind {
        let mut best = (f64::NEG_INFINITY, c[0].clone());
        for k in c {
            let o = tune_obj(&k, true);
            if o > best.0 {
                best = (o, k);
            }
        }
        best.1
    };
    let mut c = vec![];
    for &g in &[0.95, 0.99, 1.0] {
        for &e in &[0.02, 0.05, 0.1] {
            c.push(Kind::Eps(g, e));
        }
    }
    let eps = pick(c);
    let pts = pick([0.9, 0.95, 0.99, 1.0].iter().map(|&g| Kind::Pts(g)).collect());
    let mut c = vec![];
    for &k in &[3.0, 10.0, 30.0, 100.0] {
        for &e in &[Ev::Set, Ev::Mean, Ev::Global, Ev::FailSet] {
            c.push(Kind::WarmHebR(k, e));
        }
    }
    let warm = pick(c);
    println!("tuned on seeds 1-8 (plain features, informative world): {:?} {:?} {:?}", eps, pts, warm);
    let routers: Vec<(&str, Kind)> = vec![("keyword", Kind::Keyword), ("eps-greedy", eps), ("pooled-ts", pts), ("hebbian-robust", Kind::HebR), ("warm(hebbian-robust)", warm)];

    let mut all_p = vec![];
    struct Cell {
        world: &'static str,
        router: &'static str,
        plain: Vec<f64>,
        sp: Vec<f64>,
    }
    let mut cells: Vec<Cell> = vec![];
    for (wname, informative) in [("informative (best agent depends on layout class)", true), ("CONTROL uninformative (layout irrelevant)", false)] {
        let worlds: Vec<SpWorld> = held.iter().map(|&s| gen_sp_world(s, informative)).collect();
        for (rname, k) in &routers {
            let plain: Vec<f64> = held.par_iter().zip(&worlds).map(|(&s, w)| run(k, &w.stream_plain, s).mean).collect();
            let sp: Vec<f64> = held.par_iter().zip(&worlds).map(|(&s, w)| run(k, &w.stream_sp, s).mean).collect();
            all_p.push(paired_t_pvalue(&sp, &plain));
            cells.push(Cell { world: wname, router: rname, plain, sp });
        }
    }
    let adj = holm_adjust(&all_p);
    let mut jr = vec![];
    for (i, c) in cells.iter().enumerate() {
        let d = paired_diff(&c.sp, &c.plain);
        let (a, b) = (summarize(&c.plain), summarize(&c.sp));
        println!(
            "  {:<48} {:<22} plain {:.3}  +spatial {:.3}  diff {:+.4} [{:+.4},{:+.4}] holm p={:.4}{}",
            c.world, c.router, a.mean, b.mean, d.diff.mean, d.diff.lo, d.diff.hi, adj[i],
            if adj[i] < 0.05 { if d.diff.mean > 0.0 { " BETTER" } else { " WORSE" } } else { "" }
        );
        jr.push(json!({"world": c.world, "router": c.router, "plain": a, "with_spatial": b, "diff": d.diff, "wins": d.wins, "losses": d.losses, "holm_p": adj[i]}));
    }
    println!("note: the informative-world gain is information added to the task description (the layout class), not a routing mechanism; the control shows what is lost when the feature is noise.");

    let j = json!({
        "benchmarks": jb, "astar": ja, "aabb_overlap": jo,
        "spatial_features_for_routing": {"protocol": "tune seeds 1-8; held-out 31-60; paired 95% t-CI; Holm over routers x worlds", "results": jr},
    });
    merge_json(&out, "spatial", j);
    println!("wrote {out} (key spatial)");
}
