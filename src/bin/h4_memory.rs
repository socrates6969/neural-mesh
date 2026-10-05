//! h4_memory (R4-B1): value-based forgetting + tiered memory under a hard byte budget, quality measured as routing success
//! and useful-retrieval rate in two independent synthetic world generators (+ controls).
//!
//! Stages (seed sets are disjoint): `tune` (seeds 1-8), `held` (31-60), `confirm` (61-90), `hnsw` (31-40).
//!   cargo run --release --offline --bin h4_memory -- tune|held|confirm|hnsw
//! Equivalence margin (pre-declared in results/h4_predeclare.md): |Δ success vs full memory| < 0.01 (strict tier 0.005),
//! paired TOST over seeds, Holm across the family of main-world cells, claimed only if it holds in BOTH held and confirm.
#![allow(clippy::type_complexity)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use neural_mesh::tiered::ColdKind;
use rayon::prelude::*;

#[path = "h4_common/mod.rs"]
mod h4;
use h4::*;

const MARGIN: f64 = 0.01;
const MARGIN_STRICT: f64 = 0.005;

#[derive(Clone)]
struct Cell {
    pc: PolicyCfg,
    frac: Option<f64>,
}

fn label(c: &Cell) -> String {
    match c.frac {
        Some(f) => format!("{}@{:.0}%", c.pc.name, f * 100.0),
        None => c.pc.name.clone(),
    }
}

fn final_grid() -> Vec<Cell> {
    let mut v = vec![
        Cell { pc: PolicyCfg::new("full", Kind::Full), frac: None },
        Cell { pc: PolicyCfg::new("nomem", Kind::NoMem), frac: None },
        Cell { pc: PolicyCfg::new("ctxfree", Kind::CtxFree), frac: None },
        Cell { pc: PolicyCfg::new("full-rd", Kind::FullRd), frac: None },
    ];
    for f in [0.5, 0.25, 0.10] {
        v.push(Cell { pc: PolicyCfg::new("fifo", Kind::Fifo), frac: Some(f) });
        v.push(Cell { pc: PolicyCfg::new("lru", Kind::Lru), frac: Some(f) });
        v.push(Cell { pc: PolicyCfg::new("value", Kind::Value), frac: Some(f) });
        v.push(Cell { pc: PolicyCfg::new("merge", Kind::Merge), frac: Some(f) });
    }
    for f in [1.0, 0.5, 0.25, 0.10] {
        let mut a = PolicyCfg::new("tiered-int8", Kind::Tiered);
        a.cold = ColdKind::Int8;
        v.push(Cell { pc: a, frac: Some(f) });
        let mut b = PolicyCfg::new("tiered-f16", Kind::Tiered);
        b.cold = ColdKind::F16;
        v.push(Cell { pc: b, frac: Some(f) });
        let mut c = PolicyCfg::new("tiered-int8+merge", Kind::TieredMerge);
        c.cold = ColdKind::Int8;
        v.push(Cell { pc: c, frac: Some(f) });
    }
    v
}

fn tune_grid() -> Vec<Cell> {
    let mut v = vec![
        Cell { pc: PolicyCfg::new("full", Kind::Full), frac: None },
        Cell { pc: PolicyCfg::new("nomem", Kind::NoMem), frac: None },
        Cell { pc: PolicyCfg::new("ctxfree", Kind::CtxFree), frac: None },
        Cell { pc: { let mut p = PolicyCfg::new("full-rd[H=250]", Kind::FullRd); p.rd_h = 250.0; p }, frac: None },
        Cell { pc: { let mut p = PolicyCfg::new("full-rd[H=500]", Kind::FullRd); p.rd_h = 500.0; p }, frac: None },
        Cell { pc: { let mut p = PolicyCfg::new("full-rd[H=1000]", Kind::FullRd); p.rd_h = 1000.0; p }, frac: None },
        Cell { pc: { let mut p = PolicyCfg::new("full-rd[H=2000]", Kind::FullRd); p.rd_h = 2000.0; p }, frac: None },
        Cell { pc: { let mut p = PolicyCfg::new("full-rd[H=4000]", Kind::FullRd); p.rd_h = 4000.0; p }, frac: None },
    ];
    for f in [0.25, 0.10] {
        for hf in [0.15, 0.3, 0.6] {
            let mut p = PolicyCfg::new(&format!("tiered-int8+merge[hot={hf}]"), Kind::TieredMerge);
            p.hot_frac = hf;
            v.push(Cell { pc: p, frac: Some(f) });
        }
        for hl in [500.0, 2000.0, 8000.0] {
            for init in [0.2f32, 0.6] {
                let mut p = PolicyCfg::new(&format!("value[hl={hl},init={init}]"), Kind::Value);
                p.halflife = hl;
                p.init = init;
                v.push(Cell { pc: p, frac: Some(f) });
            }
        }
        for m in [0.7f32, 0.8, 0.9] {
            let mut p = PolicyCfg::new(&format!("merge[sim={m}]"), Kind::Merge);
            p.merge = m;
            v.push(Cell { pc: p, frac: Some(f) });
        }
        for hf in [0.15, 0.3, 0.6] {
            let mut p = PolicyCfg::new(&format!("tiered-int8[hot={hf}]"), Kind::Tiered);
            p.hot_frac = hf;
            v.push(Cell { pc: p, frac: Some(f) });
        }
    }
    v
}

type Res = BTreeMap<String, Vec<RunOut>>; // cell label -> per-seed runs (seed order)

fn run_world(wname: &str, seeds: &[u64], grid: &[Cell]) -> Res {
    let per_seed: Vec<Vec<(String, RunOut)>> = seeds
        .par_iter()
        .map(|&s| {
            let w = match wname {
                "A-zipf" => gen_a(s, &A_ZIPF),
                "A-allvalued" => gen_a(s, &A_ALLVALUED),
                "A-noreuse" => gen_a(s, &A_NOREUSE),
                "B-burst" => gen_b(s),
                _ => panic!("world {wname}"),
            };
            let full = run(&w, &PolicyCfg::new("full", Kind::Full), None);
            let mut out = Vec::new();
            let mut fullh: Option<RunOut> = None;
            for c in grid {
                let r = if c.pc.kind == Kind::Full && !c.pc.hnsw {
                    full.clone()
                } else {
                    let basis = if c.pc.hnsw { fullh.get_or_insert_with(|| { let mut p = PolicyCfg::new("full-hnsw", Kind::Full); p.hnsw = true; run(&w, &p, None) }).bytes_final } else { full.bytes_final };
                    let budget = c.frac.map(|f| (f * basis as f64) as usize);
                    run(&w, &c.pc, budget)
                };
                out.push((label(c), r));
            }
            out
        })
        .collect();
    let mut res: Res = BTreeMap::new();
    for row in per_seed {
        for (k, r) in row {
            res.entry(k).or_default().push(r);
        }
    }
    res
}

fn col(v: &[RunOut], f: impl Fn(&RunOut) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}

fn d384_bytes(r: &RunOut, cold: ColdKind) -> f64 {
    let hot = (384 * 4 + 8 + 24 + 88 + 3) as f64;
    let cold_b = match cold {
        ColdKind::Int8 => (384 + 4 + 8 + 24 + 88 + 3) as f64,
        ColdKind::F16 => (384 * 2 + 8 + 24 + 88 + 3) as f64,
        ColdKind::Turbovec => (384 * 4 / 8 + 28 + 88 + 3) as f64,
    };
    (r.items - r.cold) as f64 * hot + r.cold as f64 * cold_b
}

struct Row {
    world: String,
    cell: String,
    policy: String,
    frac: Option<f64>,
    succ: f64,
    d: f64,
    lo: f64,
    hi: f64,
    p01: f64,
    pni: f64,
    adjni: f64,
    p005: f64,
    du: f64,
    drd: f64,
    drd_lo: f64,
    drd_hi: f64,
    bytes_frac: f64,
    items_frac: f64,
    lat_ratio: f64,
    proj384_frac: f64,
    overshoot: usize,
    in_family: bool,
    adj01: f64,
    adj005: f64,
    d_per_seed: Vec<f64>,
}

const FAMILY_POLICIES: [&str; 7] = ["fifo", "lru", "value", "merge", "tiered-int8", "tiered-f16", "tiered-int8+merge"];

fn analyse(world: &str, res: &Res, grid: &[Cell]) -> Vec<Row> {
    let full = &res["full"];
    let fs = col(full, |r| r.success);
    let fu = col(full, |r| r.useful);
    let mut rows = Vec::new();
    for c in grid {
        let l = label(c);
        let rf = if c.pc.hnsw && res.contains_key("full-hnsw") { &res["full-hnsw"] } else { full };
        let fb = mean(&col(rf, |r| r.bytes_final as f64));
        let fi = mean(&col(rf, |r| r.items as f64));
        let fl = mean(&col(rf, |r| r.ns_per_search));
        let v = &res[&l];
        let s = col(v, |r| r.success);
        let d: Vec<f64> = s.iter().zip(&fs).map(|(a, b)| a - b).collect();
        let (m, lo, hi, p01, pni) = paired(&d, MARGIN);
        let (_, _, _, p005, _) = paired(&d, MARGIN_STRICT);
        let du: Vec<f64> = col(v, |r| r.useful).iter().zip(&fu).map(|(a, b)| a - b).collect();
        let (drd, drd_lo, drd_hi) = match res.get("full-rd") {
            Some(rd) => {
                let dd: Vec<f64> = s.iter().zip(rd.iter()).map(|(a, b)| a - b.success).collect();
                let (m2, l2, h2, _, _) = paired(&dd, MARGIN);
                (m2, l2, h2)
            }
            None => (f64::NAN, f64::NAN, f64::NAN),
        };
        let bytes = mean(&col(v, |r| r.bytes_final as f64));
        let kcold = c.pc.cold;
        let proj = mean(&v.iter().map(|r| d384_bytes(r, kcold)).collect::<Vec<_>>());
        let fproj = mean(&full.iter().map(|r| d384_bytes(r, kcold)).collect::<Vec<_>>());
        let base = c.pc.name.split('[').next().unwrap().to_string();
        rows.push(Row {
            world: world.to_string(),
            cell: l,
            policy: base.clone(),
            frac: c.frac,
            succ: mean(&s),
            d: m,
            lo,
            hi,
            p01,
            pni,
            adjni: f64::NAN,
            p005,
            du: mean(&du),
            drd,
            drd_lo,
            drd_hi,
            bytes_frac: if matches!(c.pc.kind, Kind::NoMem | Kind::CtxFree) { 0.0 } else { bytes / fb },
            items_frac: mean(&col(v, |r| r.items as f64)) / fi,
            lat_ratio: if fl > 0.0 { mean(&col(v, |r| r.ns_per_search)) / fl } else { f64::NAN },
            proj384_frac: if matches!(c.pc.kind, Kind::NoMem | Kind::CtxFree) { 0.0 } else { proj / fproj },
            overshoot: v.iter().filter(|r| r.overshoot).count(),
            in_family: FAMILY_POLICIES.contains(&base.as_str()) && c.frac.is_some(),
            adj01: f64::NAN,
            adj005: f64::NAN,
            d_per_seed: d,
        });
    }
    rows
}

fn holm_family(rows: &mut [Row], main_worlds: &[&str]) {
    let idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].in_family && main_worlds.contains(&rows[i].world.as_str())).collect();
    let p01: Vec<f64> = idx.iter().map(|&i| rows[i].p01).collect();
    let p005: Vec<f64> = idx.iter().map(|&i| rows[i].p005).collect();
    let a01 = holm_adjust(&p01);
    let a005 = holm_adjust(&p005);
    let pni: Vec<f64> = idx.iter().map(|&i| rows[i].pni).collect();
    let ani = holm_adjust(&pni);
    for (j, &i) in idx.iter().enumerate() {
        rows[i].adj01 = a01[j];
        rows[i].adj005 = a005[j];
        rows[i].adjni = ani[j];
    }
}

fn eq_mark(adj: f64, raw: f64, in_family: bool) -> &'static str {
    let p = if in_family { adj } else { raw };
    if p.is_nan() { "-" } else if p < 0.05 { "EQ" } else { "no" }
}

fn md(rows: &[Row], oracle: &BTreeMap<String, f64>, nseeds: usize, stage: &str) -> String {
    let mut s = String::new();
    writeln!(s, "# h4 memory-reduction stage `{stage}` (synthetic-until-validated; n_seeds={nseeds})\n").unwrap();
    writeln!(s, "Margin pre-declared: +-{MARGIN} success (strict tier +-{MARGIN_STRICT}); reference = full (unbounded) memory, paired by seed. EQ = TOST p<0.05 (Holm-adjusted over the main-world family; controls use raw p). bytes = final resident estimate / full at D=64; proj384 = same retained set priced at D=384. lat = mean search time vs full (parallel runs, noisy). dSucc vs full-rd = sensitivity comparison against UNBOUNDED memory whose router down-weights old cases (recency half-life, tuned on seeds 1-8 only; added after the first held-out run, disclosed).\n").unwrap();
    let mut worlds: Vec<String> = rows.iter().map(|r| r.world.clone()).collect();
    worlds.dedup();
    for w in worlds {
        writeln!(s, "## {w}  (oracle success ceiling on eval window {:.3})\n", oracle[&w]).unwrap();
        writeln!(s, "| cell | success | dSucc vs full [95% CI] | dUseful | bytes | proj384 | items | lat | dSucc vs full-rd [95% CI] | p(.01) | EQ.01 | EQ.005 | NI.01 | ovr |").unwrap();
        writeln!(s, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|").unwrap();
        for r in rows.iter().filter(|r| r.world == w) {
            writeln!(
                s,
                "| {} | {:.4} | {:+.4} [{:+.4},{:+.4}] | {:+.4} | {:.3} | {:.3} | {:.3} | {:.2} | {:+.4} [{:+.4},{:+.4}] | {:.4} | {} | {} | {} | {} |",
                r.cell, r.succ, r.d, r.lo, r.hi, r.du, r.bytes_frac, r.proj384_frac, r.items_frac, r.lat_ratio, r.drd, r.drd_lo, r.drd_hi, r.p01,
                eq_mark(r.adj01, r.p01, r.in_family), eq_mark(r.adj005, r.p005, r.in_family), eq_mark(r.adjni, r.pni, r.in_family).replace("EQ", "NI"), r.overshoot
            )
            .unwrap();
        }
        writeln!(s).unwrap();
    }
    s
}

fn json(rows: &[Row]) -> String {
    let v: Vec<serde_json::Value> = rows
        .iter()
        .map(|r| {
            serde_json::json!({
                "world": r.world, "cell": r.cell, "policy": r.policy, "frac": r.frac, "success": r.succ,
                "d": r.d, "lo": r.lo, "hi": r.hi, "p_tost_01": r.p01, "p_ni_01": r.pni, "adj_ni_01": r.adjni, "p_tost_005": r.p005, "adj_01": r.adj01, "adj_005": r.adj005,
                "d_useful": r.du, "d_vs_fullrd": r.drd, "d_vs_fullrd_lo": r.drd_lo, "d_vs_fullrd_hi": r.drd_hi, "bytes_frac": r.bytes_frac, "proj384_frac": r.proj384_frac, "items_frac": r.items_frac,
                "lat_ratio": r.lat_ratio, "overshoot_runs": r.overshoot, "d_per_seed": r.d_per_seed,
            })
        })
        .collect();
    serde_json::to_string_pretty(&v).unwrap()
}

fn main() {
    let stage = std::env::args().nth(1).unwrap_or_else(|| "tune".into());
    let (seeds, worlds, grid): (Vec<u64>, Vec<&str>, Vec<Cell>) = match stage.as_str() {
        "tune" => ((1..=8).collect(), vec!["A-zipf", "B-burst"], tune_grid()),
        "held" => ((31..=60).collect(), vec!["A-zipf", "B-burst", "A-allvalued", "A-noreuse"], final_grid()),
        "confirm" => ((61..=90).collect(), vec!["A-zipf", "B-burst", "A-allvalued", "A-noreuse"], final_grid()),
        "hnsw" => {
            let mut g = vec![Cell { pc: PolicyCfg::new("full", Kind::Full), frac: None }];
            g.push(Cell { pc: PolicyCfg::new("value", Kind::Value), frac: Some(0.25) });
            let mut h = PolicyCfg::new("value-hnsw", Kind::Value);
            h.hnsw = true;
            g.push(Cell { pc: h, frac: Some(0.25) });
            let mut fh = PolicyCfg::new("full-hnsw", Kind::Full);
            fh.hnsw = true;
            g.push(Cell { pc: fh, frac: None });
            ((31..=40).collect(), vec!["A-zipf"], g)
        }
        _ => panic!("stage"),
    };
    let t0 = std::time::Instant::now();
    let mut all_rows: Vec<Row> = Vec::new();
    let mut oracle = BTreeMap::new();
    for w in &worlds {
        let res = run_world(w, &seeds, &grid);
        let wo = match *w {
            "A-zipf" => gen_a(seeds[0], &A_ZIPF),
            "A-allvalued" => gen_a(seeds[0], &A_ALLVALUED),
            "A-noreuse" => gen_a(seeds[0], &A_NOREUSE),
            _ => gen_b(seeds[0]),
        };
        oracle.insert(w.to_string(), wo.oracle());
        let rows = analyse(w, &res, &grid);
        eprintln!("[{stage}] world {w} done at {:.0}s", t0.elapsed().as_secs_f64());
        all_rows.extend(rows);
    }
    holm_family(&mut all_rows, &["A-zipf", "B-burst"]);
    let text = md(&all_rows, &oracle, seeds.len(), &stage);
    println!("{text}");
    let dir = std::path::Path::new("results");
    let _ = std::fs::write(dir.join(format!("h4_{stage}.md")), &text);
    let _ = std::fs::write(dir.join(format!("h4_{stage}.json")), json(&all_rows));
    eprintln!("[{stage}] total {:.0}s", t0.elapsed().as_secs_f64());
}
