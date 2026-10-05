//! H10 (REAL-TEXT replay, task 1): spell-correction + retrieval on REAL documentation text.
//!
//! Text is real (docs of vendored crates = world A; this project's prose + ruvnet docs = world B).  Typos injected into the real query
//! text are SYNTHETIC (QWERTY-adjacency single-edit model, py/h10_embed.py).  The only human-made misspellings are the n=10 'R' queries.
//!
//! PRE-DECLARED PROTOCOL (fixed before the first run):
//!   worlds   A: 6824 paragraph docs from ~700 crates' READMEs; queries H (section titles) and S (held-out real sentence), 120+120 per seed.
//!            B: 4176 docs (ruvnet docs + our docs + doc comments); H/S 100+100 per seed.  A and B share no text and no authors.
//!   splits   tune = A seeds 1-8 on the tune query pool (sections/paragraphs with hash%5==0); held-out = A seeds 31-60 on the disjoint pool;
//!            confirm-A = A seeds 61-90 (same disjoint pool, fresh typo draws / subsets); confirm-B = world B seeds 61-90 with FROZEN picks.
//!   tuned    only the hybrid weight alpha (z-score fusion, grid .4..1.0) for the raw and the spell-corrected arm, on tune seeds, mix25 MRR.
//!   systems  dense raw / dense+spell S1 (strict, doc vocab) / dense+spell S2 (g1-style query-log rule) / BM25 raw / BM25+S1 /
//!            hybrid raw / hybrid+S1 / hybrid+S2.
//!   metrics  hit@1, MRR, hit@5 (best relevant doc rank among all docs).  mix25 = 75% clean + 25% typo queries (typo = mean of 1-word and 2-word typos).
//!   stats    paired 95% t-CI over the 30 per-seed means; Holm over 10 comparisons x 5 slices x 2 metrics = 100 tests per split;
//!            headline contrasts also get a query-cluster bootstrap CI (resampling DISTINCT queries, 2000 reps).
//!   claim    g1: spell-correction lifts MRR on typo-heavy input (+.016..+.026 at 25% typos) and the whole 'system' gain is that one lever.
//!
//! Usage: h10_retrieval [--out results/h10_retrieval.json]

#[path = "h10_common/mod.rs"]
mod common;

use std::collections::BTreeMap;
use std::time::Instant;

use common::*;
use serde_json::{Value, json};

const BASE: usize = 5; // dense_raw, dense_s1, dense_s2, bm25_raw, bm25_s1
const GRID: [f64; 7] = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

#[derive(Clone)]
struct RowRes {
    q: usize,
    fam: String,
    /// ranks[var][sys]
    ranks: Vec<Vec<usize>>,
}

fn nsys(ng: usize) -> usize {
    BASE + 3 * ng
}

fn eval_row(w: &World, row: &Row, bm: &Bm25, grid: &[f64]) -> RowRes {
    let rel = &w.queries[row.q].rel;
    let mut ranks = Vec::new();
    let mut dbuf = Vec::new();
    let mut bbuf = Vec::new();
    for v in 0..row.t.len() {
        let ids = row.t[v];
        // dense + bm25 per slot (cache when ids coincide)
        let mut dense: Vec<Vec<f32>> = Vec::new();
        let mut bm25: Vec<Vec<f32>> = Vec::new();
        for s in 0..3 {
            if let Some(p) = (0..s).find(|&p| ids[p] == ids[s]) {
                let d = dense[p].clone();
                let b = bm25[p].clone();
                dense.push(d);
                bm25.push(b);
            } else {
                w.dense(ids[s], &mut dbuf);
                bm.score(&w.texts[ids[s]], &mut bbuf);
                dense.push(dbuf.clone());
                bm25.push(bbuf.clone());
            }
        }
        let mut r = vec![0usize; nsys(grid.len())];
        r[0] = best_rank(&dense[0], rel);
        r[1] = best_rank(&dense[1], rel);
        r[2] = best_rank(&dense[2], rel);
        r[3] = best_rank(&bm25[0], rel);
        r[4] = best_rank(&bm25[1], rel);
        let zd: Vec<Vec<f32>> = dense.iter().map(|d| zscore(d)).collect();
        let zb: Vec<Vec<f32>> = bm25.iter().map(|d| zscore(d)).collect();
        for (gi, &a) in grid.iter().enumerate() {
            for s in 0..3 {
                let f: Vec<f32> = zd[s].iter().zip(&zb[s]).map(|(d, b)| a as f32 * d + (1.0 - a as f32) * b).collect();
                r[BASE + 3 * gi + s] = best_rank(&f, rel);
            }
        }
        ranks.push(r);
    }
    RowRes { q: row.q, fam: row.fam.clone(), ranks }
}

fn par_seeds(w: &World, bm: &Bm25, seeds: &[u64], grid: &[f64]) -> Vec<(u64, Vec<RowRes>)> {
    let nt = std::env::var("H10_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(2usize);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out: Vec<std::sync::Mutex<Option<Vec<RowRes>>>> = seeds.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..nt {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= seeds.len() {
                    break;
                }
                let inst = &w.inst[&seeds[i].to_string()];
                let rows: Vec<RowRes> = inst.rows.iter().map(|r| eval_row(w, r, bm, grid)).collect();
                *out[i].lock().unwrap() = Some(rows);
            });
        }
    });
    seeds.iter().zip(out).map(|(&s, m)| (s, m.into_inner().unwrap().unwrap())).collect()
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Met {
    Hit1,
    Mrr,
    Hit5,
}
fn val(rank: usize, m: Met) -> f64 {
    match m {
        Met::Hit1 => (rank == 1) as u8 as f64,
        Met::Mrr => 1.0 / rank as f64,
        Met::Hit5 => (rank <= 5) as u8 as f64,
    }
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Clean,
    Var(usize),
    Typo,
    Mix(f64),
}

/// mean over the seed's queries (families weighted equally when `fam` is None)
fn seed_metric(rows: &[RowRes], sys: usize, fam: Option<&str>, kind: Kind, m: Met) -> f64 {
    let fams: Vec<&str> = match fam {
        Some(f) => vec![f],
        None => vec!["H", "S"],
    };
    let var_mean = |f: &str, v: usize| -> f64 {
        let xs: Vec<f64> = rows.iter().filter(|r| r.fam == f).map(|r| val(r.ranks[v][sys], m)).collect();
        mean(&xs)
    };
    let one = |f: &str| -> f64 {
        match kind {
            Kind::Clean => var_mean(f, 0),
            Kind::Var(v) => var_mean(f, v),
            Kind::Typo => 0.5 * (var_mean(f, 1) + var_mean(f, 2)),
            Kind::Mix(p) => (1.0 - p) * var_mean(f, 0) + p * 0.5 * (var_mean(f, 1) + var_mean(f, 2)),
        }
    };
    mean(&fams.iter().map(|f| one(f)).collect::<Vec<_>>())
}

fn series(res: &[(u64, Vec<RowRes>)], sys: usize, fam: Option<&str>, kind: Kind, m: Met) -> Vec<f64> {
    res.iter().map(|(_, rows)| seed_metric(rows, sys, fam, kind, m)).collect()
}

struct Sys {
    name: &'static str,
    idx: usize,
}

fn systems(final_grid: bool) -> Vec<Sys> {
    // final layout: grid = [alpha_raw, alpha_s1]; hybrid raw uses gi=0 slot0; hybrid s1/s2 use gi=1 slot 1/2
    let _ = final_grid;
    vec![
        Sys { name: "dense_raw", idx: 0 },
        Sys { name: "dense_s1", idx: 1 },
        Sys { name: "dense_s2", idx: 2 },
        Sys { name: "bm25_raw", idx: 3 },
        Sys { name: "bm25_s1", idx: 4 },
        Sys { name: "hyb_raw", idx: BASE },
        Sys { name: "hyb_s1", idx: BASE + 3 + 1 },
        Sys { name: "hyb_s2", idx: BASE + 3 + 2 },
    ]
}

fn sidx(name: &str) -> usize {
    systems(true).iter().find(|s| s.name == name).unwrap().idx
}

const COMPARES: [(&str, &str); 10] = [
    ("dense_s1", "dense_raw"),
    ("dense_s2", "dense_raw"),
    ("dense_s1", "dense_s2"),
    ("dense_s1", "bm25_raw"),
    ("dense_s1", "bm25_s1"),
    ("hyb_s1", "dense_s1"),
    ("hyb_s1", "bm25_s1"),
    ("hyb_raw", "dense_raw"),
    ("dense_raw", "bm25_raw"),
    ("hyb_s1", "hyb_raw"),
];

fn slices() -> Vec<(&'static str, Option<&'static str>, Kind)> {
    vec![
        ("mix25", None, Kind::Mix(0.25)),
        ("clean", None, Kind::Clean),
        ("typo", None, Kind::Typo),
        ("mix25-H", Some("H"), Kind::Mix(0.25)),
        ("mix25-S", Some("S"), Kind::Mix(0.25)),
    ]
}

fn report_split(label: &str, res: &[(u64, Vec<RowRes>)], log: &mut String) -> Value {
    let mut out = json!({});
    // ---- level table
    let mut lv = json!({});
    log.push_str(&format!("\n=== {label}: mean over {} seeds (hit@1 / MRR / hit@5) ===\n", res.len()));
    let sl_all: Vec<(&str, Option<&str>, Kind)> = {
        let mut v = slices();
        v.push(("typo1", None, Kind::Var(1)));
        v.push(("typo2", None, Kind::Var(2)));
        v.push(("mix50", None, Kind::Mix(0.5)));
        v.push(("clean-H", Some("H"), Kind::Clean));
        v.push(("clean-S", Some("S"), Kind::Clean));
        v.push(("typo-H", Some("H"), Kind::Typo));
        v.push(("typo-S", Some("S"), Kind::Typo));
        v
    };
    for s in systems(true) {
        let mut row = json!({});
        let mut line = format!("{:10}", s.name);
        for (sn, fam, kind) in &sl_all {
            let h1 = mean(&series(res, s.idx, *fam, *kind, Met::Hit1));
            let mr = mean(&series(res, s.idx, *fam, *kind, Met::Mrr));
            let h5 = mean(&series(res, s.idx, *fam, *kind, Met::Hit5));
            row[*sn] = json!({"hit1": h1, "mrr": mr, "hit5": h5});
            if matches!(*sn, "mix25" | "clean" | "typo" | "mix25-H" | "mix25-S") {
                line.push_str(&format!(" | {sn} {:.3}/{:.3}", h1, mr));
            }
        }
        log.push_str(&line);
        log.push('\n');
        lv[s.name] = row;
    }
    out["levels"] = lv;
    // ---- paired comparisons with Holm
    let mut tests: Vec<(String, String, &str, Met, Cmp)> = Vec::new();
    for (a, b) in COMPARES {
        for (sn, fam, kind) in slices() {
            for m in [Met::Mrr, Met::Hit1] {
                let sa = series(res, sidx(a), fam, kind, m);
                let sb = series(res, sidx(b), fam, kind, m);
                tests.push((a.to_string(), b.to_string(), sn, m, compare(&sa, &sb)));
            }
        }
    }
    let ps: Vec<f64> = tests.iter().map(|t| t.4.p).collect();
    let adj = holm(&ps);
    let mut cj = Vec::new();
    log.push_str(&format!("--- paired diffs (a-b), 95% t-CI over seeds, Holm over {} tests; * Holm win, ! Holm loss ---\n", tests.len()));
    for (i, (a, b, sn, m, c)) in tests.iter().enumerate() {
        let star = if adj[i] < 0.05 { if c.mean > 0.0 { "*" } else { "!" } } else { "" };
        if *sn == "mix25" || *sn == "clean" || *sn == "typo" {
            log.push_str(&format!("{a:9} - {b:9} {sn:8} {:?}: {:+.4} [{:+.4},{:+.4}] holm {:.4}{star} w{}/l{}\n", m, c.mean, c.lo, c.hi, adj[i], c.wins, c.losses));
        }
        cj.push(json!({"a": a, "b": b, "slice": sn, "metric": format!("{:?}", m), "diff": c.mean, "lo": c.lo, "hi": c.hi, "p": c.p, "holm": adj[i],
                       "wins": c.wins, "losses": c.losses, "n": c.n, "holm_win": adj[i] < 0.05 && c.mean > 0.0, "holm_loss": adj[i] < 0.05 && c.mean < 0.0}));
    }
    out["compare"] = Value::Array(cj);
    out
}

/// query-cluster bootstrap for headline contrasts: per DISTINCT query, mean over its appearances of the mix25 difference
fn cluster_boot(res: &[(u64, Vec<RowRes>)], a: &str, b: &str, m: Met, kind_p: f64, seed: u64) -> (f64, f64, f64, usize) {
    let mut acc: BTreeMap<usize, (f64, usize)> = BTreeMap::new();
    for (_, rows) in res {
        for r in rows {
            let f = |sys: usize| -> f64 {
                let c = val(r.ranks[0][sys], m);
                let t = 0.5 * (val(r.ranks[1][sys], m) + val(r.ranks[2][sys], m));
                (1.0 - kind_p) * c + kind_p * t
            };
            let d = f(sidx(a)) - f(sidx(b));
            let e = acc.entry(r.q).or_insert((0.0, 0));
            e.0 += d;
            e.1 += 1;
        }
    }
    let v: Vec<f64> = acc.values().map(|(s, n)| s / *n as f64).collect();
    let (mn, lo, hi) = boot_ci(&v, 2000, seed);
    (mn, lo, hi, v.len())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let outp = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/h10_retrieval.json".into());
    let t0 = Instant::now();
    let mut log = String::new();
    let wa = load_world("A");
    let wb = load_world("B");
    let bma = Bm25::new(&wa.texts[..wa.ndoc]);
    let bmb = Bm25::new(&wb.texts[..wb.ndoc]);
    println!("loaded A: {} docs, {} texts; B: {} docs, {} texts ({:.1}s)", wa.ndoc, wa.texts.len(), wb.ndoc, wb.texts.len(), t0.elapsed().as_secs_f64());

    // ---------------- TUNE (A seeds 1-8, tune pool): hybrid alpha only
    let tune = par_seeds(&wa, &bma, &seeds(1, 8), &GRID);
    let mut best_raw = (f64::MIN, 0usize);
    let mut best_s1 = (f64::MIN, 0usize);
    let mut tune_tab = Vec::new();
    for gi in 0..GRID.len() {
        let mr = mean(&series(&tune, BASE + 3 * gi, None, Kind::Mix(0.25), Met::Mrr));
        let ms = mean(&series(&tune, BASE + 3 * gi + 1, None, Kind::Mix(0.25), Met::Mrr));
        tune_tab.push(json!({"alpha": GRID[gi], "hyb_raw_mrr": mr, "hyb_s1_mrr": ms}));
        if mr > best_raw.0 {
            best_raw = (mr, gi);
        }
        if ms > best_s1.0 {
            best_s1 = (ms, gi);
        }
    }
    let (a_raw, a_s1) = (GRID[best_raw.1], GRID[best_s1.1]);
    log.push_str(&format!("TUNE seeds 1-8 (tune pool): alpha_raw={a_raw} (MRR {:.4}), alpha_s1={a_s1} (MRR {:.4}); grid {}\n", best_raw.0, best_s1.0, serde_json::to_string(&tune_tab).unwrap()));
    json_write("results/h10_picks.json", &json!({"alpha_raw": a_raw, "alpha_s1": a_s1, "tuned_on": "A seeds 1-8 tune pool, mix25 MRR", "table": tune_tab}));
    let grid = [a_raw, a_s1];

    // ---------------- FROZEN evaluations
    let held = par_seeds(&wa, &bma, &seeds(31, 60), &grid);
    let confa = par_seeds(&wa, &bma, &seeds(61, 90), &grid);
    let confb = par_seeds(&wb, &bmb, &seeds(61, 90), &grid);
    // tune split re-evaluated with the final layout for the level table
    let tune2 = par_seeds(&wa, &bma, &seeds(1, 8), &grid);
    let mut result = json!({"picks": {"alpha_raw": a_raw, "alpha_s1": a_s1}});
    result["tune_A_1_8"] = report_split("A tune seeds 1-8 (tune pool)", &tune2, &mut log);
    result["held_A_31_60"] = report_split("A held-out seeds 31-60", &held, &mut log);
    result["confirm_A_61_90"] = report_split("A confirm seeds 61-90 (frozen)", &confa, &mut log);
    result["confirm_B_61_90"] = report_split("B confirm seeds 61-90 (frozen, independent text)", &confb, &mut log);

    // ---------------- query-cluster bootstrap for headline contrasts
    let mut cb = Vec::new();
    log.push_str("\n=== query-cluster bootstrap (resample distinct queries, 2000 reps), mix25, diff [95% CI] ===\n");
    for (label, res) in [("A held 31-60", &held), ("A confirm 61-90", &confa), ("B confirm 61-90", &confb)] {
        for (a, b) in [("dense_s1", "dense_raw"), ("dense_s2", "dense_raw"), ("hyb_s1", "dense_s1"), ("dense_s1", "bm25_s1"), ("hyb_s1", "hyb_raw")] {
            for m in [Met::Mrr, Met::Hit1] {
                let (mn, lo, hi, n) = cluster_boot(res, a, b, m, 0.25, 12345);
                log.push_str(&format!("{label:16} {a:9} - {b:9} {:?}: {mn:+.4} [{lo:+.4},{hi:+.4}] over {n} distinct queries\n", m));
                cb.push(json!({"split": label, "a": a, "b": b, "metric": format!("{:?}", m), "diff": mn, "lo": lo, "hi": hi, "n_queries": n}));
            }
        }
    }
    result["cluster_bootstrap"] = Value::Array(cb);

    // ---------------- how often does the spell-corrector change a CLEAN query (harm), by family
    let mut alt = json!({});
    for (label, w, seeds_) in [("A", &wa, seeds(31, 60)), ("B", &wb, seeds(61, 90))] {
        let mut tot = BTreeMap::new();
        for s in seeds_ {
            for r in &w.inst[&s.to_string()].rows {
                let e = tot.entry(r.fam.clone()).or_insert((0usize, 0usize, 0usize, 0usize, 0usize));
                e.0 += 1;
                let lo = |i: usize| w.texts[i].to_lowercase(); // the spell texts are lower-cased; compare case-insensitively
                let clean = lo(r.t[0][0]);
                e.1 += (clean != lo(r.t[0][1])) as usize;
                e.2 += (clean != lo(r.t[0][2])) as usize;
                e.3 += (lo(r.t[1][1]) == clean) as usize;
                e.4 += (lo(r.t[2][1]) == clean) as usize;
            }
        }
        for (f, (n, a1, a2, r1, r2)) in tot {
            log.push_str(&format!("{label} {f}: clean queries altered by S1 {:.3} / S2 {:.3}; typo1 exactly restored by S1 {:.3}; typo2 exactly restored by S1 {:.3}\n", a1 as f64 / n as f64, a2 as f64 / n as f64, r1 as f64 / n as f64, r2 as f64 / n as f64));
            alt[format!("{label}-{f}")] = json!({"n": n, "clean_altered_s1": a1 as f64 / n as f64, "clean_altered_s2": a2 as f64 / n as f64, "typo1_restored_s1": r1 as f64 / n as f64, "typo2_restored_s1": r2 as f64 / n as f64});
        }
    }
    result["spell_side_effects"] = alt;

    // ---------------- REAL human misspellings (n=10, world A): descriptive only
    {
        let inst = &wa.inst["R"];
        let rr: Vec<RowRes> = inst.rows.iter().map(|r| eval_row(&wa, r, &bma, &grid)).collect();
        let mut tab = json!({});
        log.push_str(&format!("\n=== REAL human misspellings: n={} sentences (A); real = as written, fixed = typo corrected by hand ===\n", rr.len()));
        for s in systems(true) {
            for (vn, v) in [("real", 0usize), ("fixed", 1)] {
                let h1 = mean(&rr.iter().map(|r| val(r.ranks[v][s.idx], Met::Hit1)).collect::<Vec<_>>());
                let mr = mean(&rr.iter().map(|r| val(r.ranks[v][s.idx], Met::Mrr)).collect::<Vec<_>>());
                tab[format!("{}-{}", s.name, vn)] = json!({"hit1": h1, "mrr": mr});
            }
            log.push_str(&format!("{:10} real hit1 {:.2} mrr {:.3} | fixed hit1 {:.2} mrr {:.3}\n", s.name,
                tab[format!("{}-real", s.name)]["hit1"].as_f64().unwrap(), tab[format!("{}-real", s.name)]["mrr"].as_f64().unwrap(),
                tab[format!("{}-fixed", s.name)]["hit1"].as_f64().unwrap(), tab[format!("{}-fixed", s.name)]["mrr"].as_f64().unwrap()));
        }
        result["real_typos_n10"] = tab;
        result["real_typos_n"] = json!(rr.len());
    }

    // ---------------- cost (single thread, this box): per-query scoring
    {
        let rows: Vec<&Row> = wa.inst["31"].rows.iter().take(100).collect();
        let mut buf = Vec::new();
        let t = Instant::now();
        for r in &rows {
            wa.dense(r.t[0][0], &mut buf);
        }
        let dense_us = t.elapsed().as_secs_f64() * 1e6 / rows.len() as f64;
        let t = Instant::now();
        for r in &rows {
            bma.score(&wa.texts[r.t[0][0]], &mut buf);
        }
        let bm_us = t.elapsed().as_secs_f64() * 1e6 / rows.len() as f64;
        log.push_str(&format!("\nCOST (single thread, {} docs): dense scan {:.0} us/query, BM25 {:.0} us/query\n", wa.ndoc, dense_us, bm_us));
        result["cost_us_per_query"] = json!({"dense_scan": dense_us, "bm25": bm_us, "n_docs": wa.ndoc});
    }
    result["wall_s"] = json!(t0.elapsed().as_secs_f64());
    json_write(&outp, &result);
    println!("{log}");
    let _ = std::fs::write(outp.replace(".json", ".txt"), &log);
    println!("wrote {outp} in {:.1}s", t0.elapsed().as_secs_f64());
}

fn json_write(p: &str, v: &Value) {
    std::fs::write(p, serde_json::to_string_pretty(v).unwrap()).unwrap();
}
