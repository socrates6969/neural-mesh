//! G2 TASK A: dimension-generic spatial reasoning (2D / 3D / 4D) — correctness against brute force
//! (mismatch counts over many seeds incl. degenerate cases) and speed (queries/s vs brute force).
//!
//! Operations covered in D = 2, 3, 4:
//!   points  : nearest, k-NN, box range, radius                (KdTree, Grid)        vs brute force
//!   objects : nearest box, k-NN boxes, boxes intersecting a query box, boxes containing a point
//!             (BoxTree, a static BVH)                                               vs brute force
//!   overlap : all intersecting box pairs (3-D boxes, and 4-D hyper-boxes = x,y,z,t) (sweep-and-prune, BoxTree) vs brute
//!   paths   : A* on a D-dim unit-cost grid (D = 2,3,4) vs BFS; space-time A* ((x,y,t) and (x,y,z,t), time-monotone
//!             moves + waiting + moving obstacles, weighted wait/move costs) vs an exact layered DP
//!
//! Exact-equality checks (index AND squared distance, ties by lowest index; sorted index lists; path costs and
//! path validity), not tolerance checks. Degenerate generators: empty, single point, all-identical, lattice
//! (massive exact ties), heavy duplicates, flat (one axis constant), clustered, k = 0 / k > n, queries far
//! outside, inverted ranges; boxes: lattice-touching faces, zero-extent boxes, one huge slab among tiny boxes.
//!
//! Usage: cargo run --release --offline --bin g2_spatial [-- --out results/g2_spatial.json] [--quick]

use std::time::Instant;

use neural_mesh::router::Rng;
use neural_mesh::spatial::*;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------------------------
// generators

#[derive(Clone, Copy, Debug)]
enum Gen {
    Uniform,
    Lattice,
    Duplicates,
    Clustered,
    Identical,
    Flat,
}
const GENS: [(Gen, &str); 6] =
    [(Gen::Uniform, "uniform"), (Gen::Lattice, "lattice"), (Gen::Duplicates, "duplicates"), (Gen::Clustered, "clustered"), (Gen::Identical, "identical"), (Gen::Flat, "flat")];

fn gen_pts<const D: usize>(rng: &mut Rng, n: usize, g: Gen) -> Vec<Pt<D>> {
    let base: Vec<f64> = (0..D).map(|_| (rng.f64() * 5.0).floor()).collect();
    (0..n)
        .map(|_| {
            let mut p = [0.0; D];
            for i in 0..D {
                p[i] = match g {
                    Gen::Uniform => rng.f64() * 100.0,
                    Gen::Lattice => (rng.f64() * 6.0).floor(),
                    Gen::Duplicates => (rng.f64() * 3.0).floor() * 7.5,
                    Gen::Clustered => (rng.f64() * 8.0).floor() + rng.f64() * 0.01,
                    Gen::Identical => base[i],
                    Gen::Flat => {
                        if i == 0 {
                            3.0
                        } else {
                            rng.f64() * 10.0
                        }
                    }
                };
            }
            p
        })
        .collect()
}

fn gen_boxes<const D: usize>(rng: &mut Rng, n: usize, kind: u8) -> Vec<Aabb<D>> {
    (0..n)
        .map(|_| {
            let mut lo = [0.0; D];
            let mut hi = [0.0; D];
            for i in 0..D {
                match kind {
                    0 => {
                        lo[i] = rng.f64() * 100.0;
                        hi[i] = lo[i] + rng.f64() * 14.0;
                    }
                    1 => {
                        lo[i] = (rng.f64() * 8.0).floor();
                        hi[i] = lo[i] + (rng.f64() * 3.0).floor();
                    }
                    2 => {
                        lo[i] = (rng.f64() * 5.0).floor();
                        hi[i] = lo[i];
                    }
                    _ => {
                        if rng.below(10) == 0 {
                            lo[i] = 0.0;
                            hi[i] = 100.0;
                        } else {
                            lo[i] = rng.f64() * 100.0;
                            hi[i] = lo[i] + 0.5;
                        }
                    }
                }
            }
            Aabb::new(lo, hi)
        })
        .collect()
}
const BOX_KINDS: [&str; 4] = ["continuous", "lattice-touching", "zero-extent", "huge-slab+tiny"];

// ---------------------------------------------------------------------------------------------
// correctness

#[derive(Default, Clone)]
struct Tally {
    cases: u64,
    mismatches: u64,
}
impl Tally {
    fn add(&mut self, ok: bool) {
        self.cases += 1;
        if !ok {
            self.mismatches += 1;
        }
    }
    fn json(&self) -> Value {
        json!({"cases": self.cases, "mismatches": self.mismatches})
    }
}

#[derive(Default)]
struct PointTallies {
    kd_knn: Tally,
    kd_nearest: Tally,
    kd_range: Tally,
    kd_radius: Tally,
    grid_knn: Tally,
    grid_nearest: Tally,
    grid_range: Tally,
    grid_radius: Tally,
}
#[derive(Default)]
struct BoxTallies {
    knn: Tally,
    nearest: Tally,
    intersecting: Tally,
    containing: Tally,
    overlap_tree: Tally,
    overlap_sweep: Tally,
}

fn check_points<const D: usize>(seed: u64, g: Gen, n: usize, t: &mut PointTallies) {
    let mut rng = Rng::new(seed);
    let pts: Vec<Pt<D>> = gen_pts(&mut rng, n, g);
    let kd = KdTree::build(&pts);
    let grid = Grid::build(&pts, Grid::<D>::auto_cell(&pts, 3.0));
    for _ in 0..40 {
        let mut q = [0.0; D];
        for i in 0..D {
            q[i] = match g {
                Gen::Uniform | Gen::Clustered => rng.f64() * 110.0 - 5.0,
                _ => (rng.f64() * 8.0).floor() - 1.0,
            };
        }
        // some queries far outside
        if rng.below(10) == 0 {
            q[0] += 1.0e4;
        }
        let k = rng.below(n + 4);
        let b = brute_knn(&pts, &q, k);
        t.kd_knn.add(kd.knn(&q, k) == b);
        t.grid_knn.add(grid.knn(&q, k) == b);
        let bn = brute_nearest(&pts, &q);
        t.kd_nearest.add(kd.nearest(&q) == bn);
        t.grid_nearest.add(grid.nearest(&q) == bn);
        let mut lo = q;
        let mut hi = q;
        for i in 0..D {
            let w = (rng.f64() * 4.0).floor();
            lo[i] -= w;
            hi[i] += w + (rng.below(2) as f64) * 0.5;
        }
        if rng.below(15) == 0 {
            std::mem::swap(&mut lo, &mut hi); // inverted -> must be empty everywhere
        }
        let br = brute_range(&pts, &lo, &hi);
        t.kd_range.add(kd.range(&lo, &hi) == br);
        t.grid_range.add(grid.range(&lo, &hi) == br);
        let r = (rng.f64() * 4.0).floor() + (rng.below(2) as f64) * 0.5;
        let rb = brute_radius(&pts, &q, r);
        t.kd_radius.add(kd.radius(&q, r) == rb);
        t.grid_radius.add(grid.radius(&q, r) == rb);
    }
}

fn check_boxes<const D: usize>(seed: u64, kind: u8, n: usize, t: &mut BoxTallies) {
    let mut rng = Rng::new(seed);
    let boxes: Vec<Aabb<D>> = gen_boxes(&mut rng, n, kind);
    let tree = BoxTree::build(&boxes);
    for _ in 0..30 {
        let mut q = [0.0; D];
        for i in 0..D {
            q[i] = if kind == 0 || kind == 3 { rng.f64() * 120.0 - 10.0 } else { (rng.f64() * 10.0).floor() - 1.0 };
        }
        let k = rng.below(n + 3);
        t.knn.add(tree.knn(&q, k) == brute_box_knn(&boxes, &q, k));
        t.nearest.add(tree.nearest(&q) == brute_box_nearest(&boxes, &q));
        let qb = gen_boxes::<D>(&mut rng, 1, kind)[0];
        t.intersecting.add(tree.intersecting(&qb) == brute_box_query(&boxes, &qb));
        t.containing.add(tree.containing(&q) == brute_box_query(&boxes, &Aabb::new(q, q)));
    }
    let bf = overlap_pairs_brute(&boxes);
    t.overlap_tree.add(overlap_pairs_tree(&boxes) == bf);
    t.overlap_sweep.add(overlap_pairs_sweep(&boxes) == bf);
}

fn check_gridn<const D: usize>(seed: u64, dim: usize, t: &mut Tally, valid: &mut Tally) {
    let mut rng = Rng::new(seed);
    let mut dims = [dim; D];
    for i in 0..D {
        dims[i] = (dim + rng.below(4)).max(2);
    }
    let mut s = [0usize; D];
    let mut g = [0usize; D];
    for i in 0..D {
        s[i] = rng.below(dims[i]);
        g[i] = rng.below(dims[i]);
    }
    let m = GridN::random(dims, 0.05 + 0.45 * rng.f64(), &mut rng, &[s, g]);
    let bf = m.bfs_len(s, g);
    let a = m.astar(s, g);
    t.add(bf == a.as_ref().map(|p| p.len));
    if let Some(p) = a {
        let man: usize = (0..D).map(|i| s[i].abs_diff(g[i])).sum();
        valid.add(m.path_is_valid(&p.path, s, g) && p.path.len() == p.len + 1 && p.len >= man);
    }
}

fn check_spacetime<const S: usize>(seed: u64, side: usize, horizon: usize, t: &mut Tally, valid: &mut Tally, reach: &mut (u64, u64)) {
    let mut rng = Rng::new(seed);
    let mut dims = [side; S];
    for i in 0..S {
        dims[i] = (side + rng.below(3)).max(2);
    }
    let mut s = [0usize; S];
    let mut g = [0usize; S];
    for i in 0..S {
        s[i] = rng.below(dims[i]);
        g[i] = rng.below(dims[i]);
    }
    let mut w = SpaceTime::random(dims, horizon, 0.12 * rng.f64(), 0.05 + 0.3 * rng.f64(), &mut rng, &[(s, 0)]);
    w.move_cost = 1 + rng.below(3) as u32;
    w.wait_cost = rng.below(3) as u32; // includes free waiting
    let a = w.astar(s, g);
    let d = w.dp_min_cost(s, g);
    reach.1 += 1;
    match (&a, d) {
        (None, None) => t.add(true),
        (Some(p), Some((c, _))) => {
            reach.0 += 1;
            t.add(p.cost == c);
            let first = !p.path[..p.path.len() - 1].iter().any(|(cell, _)| *cell == g);
            valid.add(w.path_is_valid(&p.path, s, g, p.cost) && p.path.last().unwrap().1 == p.arrival && first);
        }
        _ => t.add(false),
    }
}

fn correctness(quick: bool) -> Value {
    let seeds: u64 = if quick { 40 } else { 300 };
    let sizes: [usize; 7] = [0, 1, 2, 7, 40, 150, 400];
    let t0 = Instant::now();
    let mut out = json!({});
    for (dname, d) in [("2D", 2usize), ("3D", 3), ("4D", 4)] {
        let mut pt = PointTallies::default();
        let mut bt = BoxTallies::default();
        let mut by_gen = json!({});
        for (g, gname) in GENS {
            let before = pt.kd_knn.cases;
            let mut local = PointTallies::default();
            for s in 1..=seeds {
                for &n in &sizes {
                    let sd = s * 1009 + n as u64 * 7 + d as u64;
                    match d {
                        2 => check_points::<2>(sd, g, n, &mut local),
                        3 => check_points::<3>(sd, g, n, &mut local),
                        _ => check_points::<4>(sd, g, n, &mut local),
                    }
                }
            }
            let mism = local.kd_knn.mismatches + local.kd_nearest.mismatches + local.kd_range.mismatches + local.kd_radius.mismatches + local.grid_knn.mismatches + local.grid_nearest.mismatches + local.grid_range.mismatches + local.grid_radius.mismatches;
            let _ = before;
            by_gen[gname] = json!({"cases_per_op": local.kd_knn.cases, "total_mismatches_all_ops": mism});
            macro_rules! merge { ($($f:ident),*) => { $( pt.$f.cases += local.$f.cases; pt.$f.mismatches += local.$f.mismatches; )* } }
            merge!(kd_knn, kd_nearest, kd_range, kd_radius, grid_knn, grid_nearest, grid_range, grid_radius);
        }
        let mut by_kind = json!({});
        for (kind, kname) in BOX_KINDS.iter().enumerate() {
            let mut local = BoxTallies::default();
            for s in 1..=seeds {
                for &n in &sizes {
                    let sd = s * 2003 + n as u64 * 11 + d as u64;
                    match d {
                        2 => check_boxes::<2>(sd, kind as u8, n, &mut local),
                        3 => check_boxes::<3>(sd, kind as u8, n, &mut local),
                        _ => check_boxes::<4>(sd, kind as u8, n, &mut local),
                    }
                }
            }
            let mism = local.knn.mismatches + local.nearest.mismatches + local.intersecting.mismatches + local.containing.mismatches + local.overlap_tree.mismatches + local.overlap_sweep.mismatches;
            by_kind[*kname] = json!({"cases_per_query_op": local.knn.cases, "overlap_cases": local.overlap_tree.cases, "total_mismatches_all_ops": mism});
            macro_rules! merge { ($($f:ident),*) => { $( bt.$f.cases += local.$f.cases; bt.$f.mismatches += local.$f.mismatches; )* } }
            merge!(knn, nearest, intersecting, containing, overlap_tree, overlap_sweep);
        }
        let mut path_astar = Tally::default();
        let mut path_valid = Tally::default();
        let mut st_astar = Tally::default();
        let mut st_valid = Tally::default();
        let mut st_reach = (0u64, 0u64);
        let np = if quick { 100 } else { 800 };
        for s in 0..np {
            match d {
                2 => {
                    check_gridn::<2>(s, 16, &mut path_astar, &mut path_valid);
                    check_spacetime::<1>(s, 9, 18, &mut st_astar, &mut st_valid, &mut st_reach); // (x,t)
                }
                3 => {
                    check_gridn::<3>(s, 8, &mut path_astar, &mut path_valid);
                    check_spacetime::<2>(s, 6, 14, &mut st_astar, &mut st_valid, &mut st_reach); // (x,y,t)
                }
                _ => {
                    check_gridn::<4>(s, 4, &mut path_astar, &mut path_valid);
                    check_spacetime::<3>(s + 77, 4, 12, &mut st_astar, &mut st_valid, &mut st_reach); // (x,y,z,t)
                }
            }
        }
        out[dname] = json!({
            "points": {"kd_knn": pt.kd_knn.json(), "kd_nearest": pt.kd_nearest.json(), "kd_range": pt.kd_range.json(), "kd_radius": pt.kd_radius.json(),
                       "grid_knn": pt.grid_knn.json(), "grid_nearest": pt.grid_nearest.json(), "grid_range": pt.grid_range.json(), "grid_radius": pt.grid_radius.json(),
                       "by_generator": by_gen},
            "boxes": {"boxtree_knn": bt.knn.json(), "boxtree_nearest": bt.nearest.json(), "boxtree_intersecting": bt.intersecting.json(), "boxtree_containing": bt.containing.json(),
                      "overlap_pairs_tree": bt.overlap_tree.json(), "overlap_pairs_sweep": bt.overlap_sweep.json(), "by_kind": by_kind},
            "paths": {"gridn_astar_vs_bfs_len": path_astar.json(), "gridn_path_valid": path_valid.json(),
                      "spacetime_astar_vs_dp_cost": st_astar.json(), "spacetime_path_valid_monotone_first_arrival": st_valid.json(),
                      "spacetime_reachable_cases": st_reach.0, "spacetime_cases": st_reach.1,
                      "spacetime_dims": match d { 2 => "(x,t)", 3 => "(x,y,t)", _ => "(x,y,z,t)" }},
        });
    }
    // totals
    let mut total_cases = 0u64;
    let mut total_mism = 0u64;
    fn walk(v: &Value, c: &mut u64, m: &mut u64) {
        match v {
            Value::Object(o) => {
                if let (Some(cs), Some(ms)) = (o.get("cases"), o.get("mismatches")) {
                    *c += cs.as_u64().unwrap_or(0);
                    *m += ms.as_u64().unwrap_or(0);
                } else {
                    for (_, x) in o {
                        walk(x, c, m);
                    }
                }
            }
            _ => {}
        }
    }
    walk(&out, &mut total_cases, &mut total_mism);
    out["total_cases"] = json!(total_cases);
    out["total_mismatches"] = json!(total_mism);
    out["seeds_per_cell"] = json!(seeds);
    out["set_sizes"] = json!(sizes);
    out["runtime_s"] = json!(t0.elapsed().as_secs_f64());
    println!("CORRECTNESS: {total_cases} exact comparisons, {total_mism} mismatches ({:.1}s)", t0.elapsed().as_secs_f64());
    for dn in ["2D", "3D", "4D"] {
        let o = &out[dn];
        println!(
            "  {dn}: kd knn {}/{} grid knn {}/{} | boxtree knn {}/{} overlap tree {}/{} sweep {}/{} | gridN A* {}/{} spacetime A* {}/{} (reachable {}/{})",
            o["points"]["kd_knn"]["mismatches"], o["points"]["kd_knn"]["cases"], o["points"]["grid_knn"]["mismatches"], o["points"]["grid_knn"]["cases"],
            o["boxes"]["boxtree_knn"]["mismatches"], o["boxes"]["boxtree_knn"]["cases"], o["boxes"]["overlap_pairs_tree"]["mismatches"], o["boxes"]["overlap_pairs_tree"]["cases"],
            o["boxes"]["overlap_pairs_sweep"]["mismatches"], o["boxes"]["overlap_pairs_sweep"]["cases"],
            o["paths"]["gridn_astar_vs_bfs_len"]["mismatches"], o["paths"]["gridn_astar_vs_bfs_len"]["cases"],
            o["paths"]["spacetime_astar_vs_dp_cost"]["mismatches"], o["paths"]["spacetime_astar_vs_dp_cost"]["cases"],
            o["paths"]["spacetime_reachable_cases"], o["paths"]["spacetime_cases"]
        );
    }
    out
}

// ---------------------------------------------------------------------------------------------
// speed

/// Median of 3 timed batches; returns queries per second.
fn qps<F: FnMut(usize) -> usize>(nq: usize, mut f: F) -> f64 {
    let mut rates = vec![];
    let mut sink = 0usize;
    for _ in 0..3 {
        let t = Instant::now();
        for i in 0..nq {
            sink = sink.wrapping_add(f(i));
        }
        rates.push(nq as f64 / t.elapsed().as_secs_f64().max(1e-9));
    }
    std::hint::black_box(sink);
    rates.sort_by(|a, b| a.total_cmp(b));
    rates[1]
}

fn pts_for_bench<const D: usize>(rng: &mut Rng, n: usize, clustered: bool) -> Vec<Pt<D>> {
    (0..n)
        .map(|_| {
            let mut p = [0.0; D];
            if clustered {
                let c = (rng.f64() * 50.0).floor();
                let mut cr = Rng::new(c as u64 + 1);
                for i in 0..D {
                    p[i] = cr.f64() * 1000.0 + (rng.f64() + rng.f64() + rng.f64() - 1.5) * 20.0;
                }
            } else {
                for i in 0..D {
                    p[i] = rng.f64() * 1000.0;
                }
            }
            p
        })
        .collect()
}

fn bench_points<const D: usize>(n: usize, clustered: bool, seed: u64) -> Value {
    let mut rng = Rng::new(seed);
    let pts = pts_for_bench::<D>(&mut rng, n, clustered);
    let t = Instant::now();
    let kd = KdTree::build(&pts);
    let kd_build = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let grid = Grid::build(&pts, Grid::<D>::auto_cell(&pts, 3.0));
    let grid_build = t.elapsed().as_secs_f64();
    let nq = (4_000_000 / n.max(1000)).clamp(200, 4000).min(if n >= 100_000 { 2000 } else { 4000 });
    let nqb = (30_000_000 / (n * D)).clamp(10, 2000).min(nq);
    // queries: uniform in the bounding cube (so clustered data gets mostly "far" queries)
    let qs: Vec<Pt<D>> = (0..nq).map(|_| { let mut q = [0.0; D]; for i in 0..D { q[i] = rng.f64() * 1000.0; } q }).collect();
    // range box with expected ~50 results (uniform): side = 1000 * (50/n)^(1/D)
    let side = 1000.0 * (50.0 / n as f64).powf(1.0 / D as f64);
    let boxes: Vec<(Pt<D>, Pt<D>)> = qs.iter().map(|q| { let mut hi = *q; for i in 0..D { hi[i] += side; } (*q, hi) }).collect();
    let radius = side * 0.62;
    // mismatch check of the first nqb queries vs brute
    let mut mism = 0u64;
    for i in 0..nqb {
        if kd.knn(&qs[i], 10) != brute_knn(&pts, &qs[i], 10) || grid.knn(&qs[i], 10) != brute_knn(&pts, &qs[i], 10) {
            mism += 1;
        }
        if kd.nearest(&qs[i]) != brute_nearest(&pts, &qs[i]) || grid.nearest(&qs[i]) != brute_nearest(&pts, &qs[i]) {
            mism += 1;
        }
        if kd.range(&boxes[i].0, &boxes[i].1) != brute_range(&pts, &boxes[i].0, &boxes[i].1) || grid.range(&boxes[i].0, &boxes[i].1) != brute_range(&pts, &boxes[i].0, &boxes[i].1) {
            mism += 1;
        }
        if kd.radius(&qs[i], radius) != brute_radius(&pts, &qs[i], radius) || grid.radius(&qs[i], radius) != brute_radius(&pts, &qs[i], radius) {
            mism += 1;
        }
    }
    let mut ops = json!({});
    macro_rules! op {
        ($name:expr, $kd:expr, $gr:expr, $br:expr) => {{
            let a = qps(nq, $kd);
            let b = qps(nq, $gr);
            let c = qps(nqb, $br);
            ops[$name] = json!({"kd_qps": a, "grid_qps": b, "brute_qps": c, "kd_speedup": a / c, "grid_speedup": b / c});
        }};
    }
    op!("nearest", |i| kd.nearest(&qs[i]).map_or(0, |x| x.0), |i| grid.nearest(&qs[i]).map_or(0, |x| x.0), |i| brute_nearest(&pts, &qs[i % qs.len()]).map_or(0, |x| x.0));
    op!("knn10", |i| kd.knn(&qs[i], 10).len(), |i| grid.knn(&qs[i], 10).len(), |i| brute_knn(&pts, &qs[i], 10).len());
    op!("range_~50", |i| kd.range(&boxes[i].0, &boxes[i].1).len(), |i| grid.range(&boxes[i].0, &boxes[i].1).len(), |i| brute_range(&pts, &boxes[i].0, &boxes[i].1).len());
    op!("radius_~50", |i| kd.radius(&qs[i], radius).len(), |i| grid.radius(&qs[i], radius).len(), |i| brute_radius(&pts, &qs[i], radius).len());
    {
        // fair brute force for k-NN: bounded heap instead of sorting all n
        let c = qps(nqb, |i| brute_knn_heap(&pts, &qs[i], 10).len());
        let (a, b) = (ops["knn10"]["kd_qps"].as_f64().unwrap(), ops["knn10"]["grid_qps"].as_f64().unwrap());
        ops["knn10"]["brute_heap_qps"] = json!(c);
        ops["knn10"]["kd_speedup_vs_heap_brute"] = json!(a / c);
        ops["knn10"]["grid_speedup_vs_heap_brute"] = json!(b / c);
        if (0..nqb).any(|i| brute_knn_heap(&pts, &qs[i], 10) != brute_knn(&pts, &qs[i], 10)) {
            mism += 1;
        }
    }
    json!({"dim": D, "n": n, "clustered": clustered, "nq": nq, "nq_brute": nqb, "kd_build_s": kd_build, "grid_build_s": grid_build,
           "kd_bytes": kd.bytes(), "grid_bytes": grid.bytes(), "mismatches_in_timed_sample": mism, "ops": ops})
}

fn bench_boxes<const D: usize>(n: usize, seed: u64) -> Value {
    let mut rng = Rng::new(seed);
    // boxes with edge ~ 2*(vol/n)^(1/D): modest overlap density
    let edge = 1000.0 * (4.0 / n as f64).powf(1.0 / D as f64);
    let boxes: Vec<Aabb<D>> = (0..n)
        .map(|_| {
            let mut lo = [0.0; D];
            let mut hi = [0.0; D];
            for i in 0..D {
                lo[i] = rng.f64() * 1000.0;
                hi[i] = lo[i] + rng.f64() * edge;
            }
            Aabb::new(lo, hi)
        })
        .collect();
    let t = Instant::now();
    let tree = BoxTree::build(&boxes);
    let build = t.elapsed().as_secs_f64();
    let nq = (4_000_000 / n.max(1000)).clamp(200, 4000).min(if n >= 100_000 { 2000 } else { 4000 });
    let nqb = (30_000_000 / (n * D)).clamp(10, 2000).min(nq);
    let qs: Vec<Pt<D>> = (0..nq).map(|_| { let mut q = [0.0; D]; for i in 0..D { q[i] = rng.f64() * 1000.0; } q }).collect();
    let qbs: Vec<Aabb<D>> = qs.iter().map(|q| { let mut hi = *q; for i in 0..D { hi[i] += edge * 3.0; } Aabb::new(*q, hi) }).collect();
    let mut mism = 0u64;
    for i in 0..nqb {
        if tree.knn(&qs[i], 10) != brute_box_knn(&boxes, &qs[i], 10) { mism += 1; }
        if tree.nearest(&qs[i]) != brute_box_nearest(&boxes, &qs[i]) { mism += 1; }
        if tree.intersecting(&qbs[i]) != brute_box_query(&boxes, &qbs[i]) { mism += 1; }
    }
    let mut ops = json!({});
    macro_rules! op {
        ($name:expr, $t:expr, $br:expr) => {{
            let a = qps(nq, $t);
            let c = qps(nqb, $br);
            ops[$name] = json!({"tree_qps": a, "brute_qps": c, "speedup": a / c});
        }};
    }
    op!("nearest_box", |i| tree.nearest(&qs[i]).map_or(0, |x| x.0), |i| brute_box_nearest(&boxes, &qs[i]).map_or(0, |x| x.0));
    op!("knn10_boxes", |i| tree.knn(&qs[i], 10).len(), |i| brute_box_knn(&boxes, &qs[i], 10).len());
    op!("boxes_intersecting_query", |i| tree.intersecting(&qbs[i]).len(), |i| brute_box_query(&boxes, &qbs[i]).len());
    {
        let c = qps(nqb, |i| brute_box_knn_heap(&boxes, &qs[i], 10).len());
        let a = ops["knn10_boxes"]["tree_qps"].as_f64().unwrap();
        ops["knn10_boxes"]["brute_heap_qps"] = json!(c);
        ops["knn10_boxes"]["speedup_vs_heap_brute"] = json!(a / c);
        if (0..nqb).any(|i| brute_box_knn_heap(&boxes, &qs[i], 10) != brute_box_knn(&boxes, &qs[i], 10)) {
            mism += 1;
        }
    }
    json!({"dim": D, "n": n, "nq": nq, "nq_brute": nqb, "build_s": build, "bytes": tree.bytes(), "mismatches_in_timed_sample": mism, "ops": ops})
}

fn bench_overlap<const D: usize>(n: usize, seed: u64) -> Value {
    let mut rng = Rng::new(seed);
    let edge = 1000.0 * (4.0 / n as f64).powf(1.0 / D as f64);
    let boxes: Vec<Aabb<D>> = (0..n)
        .map(|_| {
            let mut lo = [0.0; D];
            let mut hi = [0.0; D];
            for i in 0..D {
                lo[i] = rng.f64() * 1000.0;
                hi[i] = lo[i] + rng.f64() * edge;
            }
            Aabb::new(lo, hi)
        })
        .collect();
    let ms = |f: &mut dyn FnMut() -> usize| -> (f64, usize) {
        let mut v = vec![];
        let mut r = 0;
        for _ in 0..3 {
            let t = Instant::now();
            r = f();
            v.push(t.elapsed().as_secs_f64() * 1e3);
        }
        v.sort_by(|a, b| a.total_cmp(b));
        (v[1], r)
    };
    let (b_ms, b_n) = ms(&mut || overlap_pairs_brute(&boxes).len());
    let (s_ms, s_n) = ms(&mut || overlap_pairs_sweep(&boxes).len());
    let (t_ms, t_n) = ms(&mut || overlap_pairs_tree(&boxes).len());
    let same = overlap_pairs_brute(&boxes) == overlap_pairs_sweep(&boxes) && overlap_pairs_brute(&boxes) == overlap_pairs_tree(&boxes);
    json!({"dim": D, "n": n, "pairs": b_n, "brute_ms": b_ms, "sweep_ms": s_ms, "tree_ms": t_ms, "sweep_speedup": b_ms / s_ms, "tree_speedup": b_ms / t_ms,
           "identical_results": same && b_n == s_n && b_n == t_n})
}

fn bench_paths() -> Value {
    let mut rows = vec![];
    // GridN: A* vs BFS, expanded nodes
    macro_rules! gridn {
        ($d:expr, $dim:expr, $dens:expr, $label:expr) => {{
            let dims = [$dim; $d];
            let (mut ta, mut tb, mut exp_a, mut cells_vis) = (0.0, 0.0, 0usize, 0usize);
            let mut ok = 0usize;
            let mut mism = 0usize;
            let mut rng = Rng::new(5 + $d as u64);
            let m = GridN::random(dims, $dens, &mut rng, &[]);
            let trials = 12;
            let mut done = 0;
            while done < trials {
                let mut s = [0usize; $d];
                let mut g = [0usize; $d];
                for i in 0..$d {
                    s[i] = rng.below($dim);
                    g[i] = rng.below($dim);
                }
                if !m.is_open(&s) || !m.is_open(&g) {
                    continue;
                }
                done += 1;
                let t = Instant::now();
                let a = m.astar(s, g);
                ta += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let b = m.bfs_len(s, g);
                tb += t.elapsed().as_secs_f64();
                if a.as_ref().map(|p| p.len) != b {
                    mism += 1;
                }
                if let Some(p) = a {
                    ok += 1;
                    exp_a += p.expanded;
                }
                cells_vis += m.cells();
            }
            rows.push(json!({"kind": "GridN A* vs BFS", "label": $label, "dim": $d, "cells": m.cells(), "obstacle_density": $dens, "trials": trials, "reachable": ok, "mismatches": mism,
                "astar_ms": ta * 1e3 / trials as f64, "bfs_ms": tb * 1e3 / trials as f64, "speedup": tb / ta, "astar_expanded_mean": exp_a as f64 / ok.max(1) as f64, "grid_cells": cells_vis / trials}));
        }};
    }
    gridn!(2, 400, 0.2, "400x400");
    gridn!(3, 64, 0.2, "64^3");
    gridn!(3, 64, 0.0, "64^3 open");
    gridn!(4, 20, 0.2, "20^4");
    gridn!(4, 20, 0.0, "20^4 open");
    // SpaceTime: A* vs layered DP
    macro_rules! st {
        ($s:expr, $side:expr, $h:expr, $sd:expr, $dd:expr, $label:expr) => {{
            let dims = [$side; $s];
            let mut rng = Rng::new(11 + $s as u64);
            let (mut ta, mut tb, mut ex) = (0.0, 0.0, 0usize);
            let (mut ta_r, mut tb_r, mut ta_u, mut tb_u, mut n_u) = (0.0, 0.0, 0.0, 0.0, 0usize);
            let (mut ok, mut mism) = (0usize, 0usize);
            let trials = 10;
            let mut done = 0;
            let mut states = 0usize;
            while done < trials {
                let mut s = [0usize; $s];
                let mut g = [0usize; $s];
                for i in 0..$s {
                    s[i] = rng.below($side);
                    g[i] = rng.below($side);
                }
                let w = SpaceTime::random(dims, $h, $sd, $dd, &mut rng, &[(s, 0)]);
                done += 1;
                states = w.cells() * ($h + 1);
                let t = Instant::now();
                let a = w.astar(s, g);
                let da = t.elapsed().as_secs_f64();
                ta += da;
                let t = Instant::now();
                let b = w.dp_min_cost(s, g);
                let db = t.elapsed().as_secs_f64();
                tb += db;
                if a.is_some() {
                    ta_r += da;
                    tb_r += db;
                } else {
                    ta_u += da;
                    tb_u += db;
                    n_u += 1;
                }
                match (a, b) {
                    (None, None) => {}
                    (Some(p), Some((c, _))) => {
                        ok += 1;
                        ex += p.expanded;
                        if p.cost != c || !w.path_is_valid(&p.path, s, g, p.cost) {
                            mism += 1;
                        }
                    }
                    _ => mism += 1,
                }
            }
            rows.push(json!({"kind": "SpaceTime A* vs layered DP", "label": $label, "spatial_dims": $s, "total_dims": $s + 1, "states": states, "static_density": $sd, "dyn_density": $dd, "trials": trials,
                "reachable": ok, "unreachable": n_u, "mismatches": mism, "astar_ms": ta * 1e3 / trials as f64, "dp_ms": tb * 1e3 / trials as f64, "speedup": tb / ta,
                "astar_ms_reachable_only": ta_r * 1e3 / ok.max(1) as f64, "dp_ms_reachable_only": tb_r * 1e3 / ok.max(1) as f64, "speedup_reachable_only": tb_r / ta_r.max(1e-12),
                "astar_ms_unreachable_only": ta_u * 1e3 / n_u.max(1) as f64, "dp_ms_unreachable_only": tb_u * 1e3 / n_u.max(1) as f64, "astar_expanded_mean": ex as f64 / ok.max(1) as f64}));
        }};
    }
    st!(2, 60, 120, 0.1, 0.05, "(x,y,t) 60x60, T=120");
    st!(3, 24, 60, 0.1, 0.05, "(x,y,z,t) 24^3, T=60");
    st!(3, 24, 60, 0.1, 0.0, "(x,y,z,t) 24^3, T=60 static-only");
    st!(3, 32, 90, 0.15, 0.1, "(x,y,z,t) 32^3, T=90");
    json!(rows)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned();
    let quick = args.iter().any(|a| a == "--quick");
    let t0 = Instant::now();
    println!("G2 spatial: 2D/3D/4D correctness + speed (quick={quick})");
    let corr = correctness(quick);

    let ns: Vec<usize> = if quick { vec![1_000, 10_000] } else { vec![1_000, 10_000, 100_000] };
    let mut pts_rows = vec![];
    let mut box_rows = vec![];
    for &n in &ns {
        for (d, f) in [(2usize, 0u8), (3, 0), (4, 0)] {
            let _ = f;
            let r = match d {
                2 => bench_points::<2>(n, false, 100 + n as u64),
                3 => bench_points::<3>(n, false, 100 + n as u64),
                _ => bench_points::<4>(n, false, 100 + n as u64),
            };
            print_points(&r);
            pts_rows.push(r);
            let r = match d {
                2 => bench_boxes::<2>(n, 200 + n as u64),
                3 => bench_boxes::<3>(n, 200 + n as u64),
                _ => bench_boxes::<4>(n, 200 + n as u64),
            };
            print_boxes(&r);
            box_rows.push(r);
        }
    }
    // clustered data with uniform queries (known weak case)
    let nc = if quick { 10_000 } else { 100_000 };
    for d in [3usize, 4] {
        let r = if d == 3 { bench_points::<3>(nc, true, 77) } else { bench_points::<4>(nc, true, 78) };
        print_points(&r);
        pts_rows.push(r);
    }
    let mut ov_rows = vec![];
    let ov_ns: Vec<usize> = if quick { vec![500, 2000] } else { vec![500, 2000, 8000, 20000] };
    for &n in &ov_ns {
        for d in [2usize, 3, 4] {
            let r = match d {
                2 => bench_overlap::<2>(n, 300 + n as u64),
                3 => bench_overlap::<3>(n, 300 + n as u64),
                _ => bench_overlap::<4>(n, 300 + n as u64),
            };
            println!(
                "overlap D={} n={:>6}: pairs {:>7}  brute {:>9.2} ms  sweep {:>8.2} ms ({:>6.1}x)  tree {:>8.2} ms ({:>6.1}x)  identical={}",
                d, n, r["pairs"], r["brute_ms"].as_f64().unwrap(), r["sweep_ms"].as_f64().unwrap(), r["sweep_speedup"].as_f64().unwrap(), r["tree_ms"].as_f64().unwrap(), r["tree_speedup"].as_f64().unwrap(), r["identical_results"]
            );
            ov_rows.push(r);
        }
    }
    let paths = bench_paths();
    for r in paths.as_array().unwrap() {
        println!("path {} {}: {}", r["kind"].as_str().unwrap(), r["label"].as_str().unwrap(), r);
    }
    let doc = json!({
        "task": "G2-A: 2D/3D/4D spatial reasoning vs brute force",
        "correctness": corr,
        "points": pts_rows, "boxes": box_rows, "overlap": ov_rows, "paths": paths,
        "notes": [
            "all comparisons are exact (index + squared distance, ties by lowest index; sorted index lists; path costs); no tolerances",
            "qps = median of 3 timed batches on a shared 16 GB box (other workers running): use ratios, not absolutes",
            "brute force is timed on fewer queries (nq_brute) at large n; speedup = qps ratio",
            "query points for clustered data are uniform over the bounding cube, i.e. mostly in empty space (known weak case for pruning)",
        ],
        "runtime_s": t0.elapsed().as_secs_f64(),
    });
    if let Some(p) = out_path {
        std::fs::write(&p, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        println!("wrote {p}");
    }
    println!("runtime {:.1}s", t0.elapsed().as_secs_f64());
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}
fn print_points(r: &Value) {
    println!("points D={} n={:>6} {} (build kd {:.3}s grid {:.3}s; mismatches in sample {})", r["dim"], r["n"], if r["clustered"].as_bool().unwrap() { "CLUSTERED" } else { "uniform" }, f(&r["kd_build_s"]), f(&r["grid_build_s"]), r["mismatches_in_timed_sample"]);
    for op in ["nearest", "knn10", "range_~50", "radius_~50"] {
        let o = &r["ops"][op];
        println!(
            "    {:<11} kd {:>11.0} q/s ({:>8.1}x)  grid {:>11.0} q/s ({:>8.1}x)  brute {:>10.0} q/s",
            op, f(&o["kd_qps"]), f(&o["kd_speedup"]), f(&o["grid_qps"]), f(&o["grid_speedup"]), f(&o["brute_qps"])
        );
        if op == "knn10" {
            println!("    {:<11}   -> vs a fair heap-based brute force ({:.0} q/s): kd {:.1}x, grid {:.1}x", "", f(&o["brute_heap_qps"]), f(&o["kd_speedup_vs_heap_brute"]), f(&o["grid_speedup_vs_heap_brute"]));
        }
    }
}
fn print_boxes(r: &Value) {
    println!("boxes  D={} n={:>6} (build {:.3}s, mismatches in sample {})", r["dim"], r["n"], f(&r["build_s"]), r["mismatches_in_timed_sample"]);
    for op in ["nearest_box", "knn10_boxes", "boxes_intersecting_query"] {
        let o = &r["ops"][op];
        println!("    {:<26} tree {:>11.0} q/s ({:>8.1}x)  brute {:>10.0} q/s", op, f(&o["tree_qps"]), f(&o["speedup"]), f(&o["brute_qps"]));
        if op == "knn10_boxes" {
            println!("    {:<26}   -> vs a fair heap-based brute force ({:.0} q/s): tree {:.1}x", "", f(&o["brute_heap_qps"]), f(&o["speedup_vs_heap_brute"]));
        }
    }
}

/// A fair O(n log k) brute-force k-NN (bounded max-heap) — the reference `brute_knn` sorts all n.
fn brute_knn_heap<const D: usize>(pts: &[Pt<D>], q: &Pt<D>, k: usize) -> Vec<(usize, f64)> {
    let mut h: std::collections::BinaryHeap<(u64, usize)> = std::collections::BinaryHeap::with_capacity(k + 1);
    for (i, p) in pts.iter().enumerate() {
        let key = dist2(p, q).to_bits();
        if h.len() < k {
            h.push((key, i));
        } else if let Some(&(wk, wi)) = h.peek() {
            if key < wk || (key == wk && i < wi) {
                h.pop();
                h.push((key, i));
            }
        }
    }
    let mut v: Vec<(usize, f64)> = h.into_iter().map(|(k, i)| (i, f64::from_bits(k))).collect();
    v.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    v
}
fn brute_box_knn_heap<const D: usize>(boxes: &[Aabb<D>], q: &Pt<D>, k: usize) -> Vec<(usize, f64)> {
    let mut h: std::collections::BinaryHeap<(u64, usize)> = std::collections::BinaryHeap::with_capacity(k + 1);
    for (i, b) in boxes.iter().enumerate() {
        let key = b.dist2_point(q).to_bits();
        if h.len() < k {
            h.push((key, i));
        } else if let Some(&(wk, wi)) = h.peek() {
            if key < wk || (key == wk && i < wi) {
                h.pop();
                h.push((key, i));
            }
        }
    }
    let mut v: Vec<(usize, f64)> = h.into_iter().map(|(k, i)| (i, f64::from_bits(k))).collect();
    v.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    v
}
