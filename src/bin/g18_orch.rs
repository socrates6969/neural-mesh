//! G18 (W19 ORCHESTRATION): multi-step research tasks. Single-shot routing (6 fair baselines) vs five orchestrator levels:
//!   L1 decompose + route each step by type; L2 + per-step retrieval of recurring-project notes; L3 + per-step verification with
//!   retry/reroute on failure; L4 + learned stopping (skip verification when confidence is high); L5 + parallel fan-out of independent steps.
//! Everything is a seeded simulation (src/bin/g18_common/mod.rs). Protocol: tune seeds 1-8, held 31-60 (`--held 31-60`), fresh confirm 61-90
//! (`--held 61-90`, SAME frozen tuning because tuning always reruns on seeds 1-8), paired 95% t-CI, Holm per family.
//!
//! Usage: cargo run --release --offline --bin g18_orch -- [--show] [--held 31-60] [--out results/g18_orch.json] [--sens]

use std::collections::BTreeMap;
use std::time::Instant;

use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, PairedDiff};
use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g18_common/mod.rs"]
mod c;
use c::*;

// ------------------------------------------------------------------------------------------------ arms
#[derive(Clone, Debug)]
enum Kind {
    Ss(SsKind, bool),
    Orch(OrchCfg),
    OracleDecomp(bool),
}
#[derive(Clone, Debug)]
struct Arm {
    name: String,
    kind: Kind,
}

#[derive(Clone, Debug)]
struct Tuned {
    eps: (f64, f64),
    pts: f64,
    warm_k: f64,
    orch_k: f64,
    r_val: f64,
    eps_ex: f64,
}

const BASELINES: [&str; 6] = ["ss-keyword", "ss-eps", "ss-pts", "ss-hebr", "ss-warm", "ss-safe"];

fn arms_main(t: &Tuned) -> Vec<Arm> {
    let mut v = vec![
        Arm { name: "ss-keyword".into(), kind: Kind::Ss(SsKind::Keyword, false) },
        Arm { name: "ss-eps".into(), kind: Kind::Ss(SsKind::Eps(t.eps.0, t.eps.1), false) },
        Arm { name: "ss-pts".into(), kind: Kind::Ss(SsKind::Pts(t.pts), false) },
        Arm { name: "ss-hebr".into(), kind: Kind::Ss(SsKind::HebR, false) },
        Arm { name: "ss-warm".into(), kind: Kind::Ss(SsKind::Warm(t.warm_k), false) },
        Arm { name: "ss-safe".into(), kind: Kind::Ss(SsKind::Safe, false) },
        Arm { name: "ss-warm+retr".into(), kind: Kind::Ss(SsKind::Warm(t.warm_k), true) },
        Arm { name: "oracle-single".into(), kind: Kind::Ss(SsKind::OracleSingle, false) },
        Arm { name: "oracle-decomp".into(), kind: Kind::OracleDecomp(false) },
        Arm { name: "oracle-decomp+ctx".into(), kind: Kind::OracleDecomp(true) },
    ];
    for l in 1..=5u8 {
        v.push(Arm { name: format!("L{l}"), kind: Kind::Orch(OrchCfg::level(l, t.orch_k, t.r_val, t.eps_ex)) });
    }
    // ablations of L3: verification signal NOT used as same-task feedback / note store
    let mut nolearn = OrchCfg::level(3, t.orch_k, t.r_val, t.eps_ex);
    nolearn.ver_learn = false;
    v.push(Arm { name: "L3-nolearn".into(), kind: Kind::Orch(nolearn) });
    // L4 with a fixed always-verify-after-warmup rule is just L3; and L4 with stopping but no exploration:
    let mut noexp = OrchCfg::level(4, t.orch_k, t.r_val, 0.0);
    noexp.eps_ex = 0.0;
    v.push(Arm { name: "L4-noexplore".into(), kind: Kind::Orch(noexp) });
    v
}

fn run_arm(w: &WorldCfg, inst: &Instance, arm: &Arm, seed: u64) -> RunOut {
    match &arm.kind {
        Kind::Ss(k, retr) => run_single(w, inst, k, *retr, seed),
        Kind::Orch(oc) => run_orch(w, inst, oc, seed),
        Kind::OracleDecomp(ctx) => run_oracle_decomp(w, inst, *ctx, seed),
    }
}

struct Cell {
    m: [f64; NM],
    q: Vec<f32>,
}
/// result[arm][seed]
fn run_world(w: &WorldCfg, widx: u64, arms: &[Arm], seeds: &[u64]) -> Vec<Vec<Cell>> {
    let per_seed: Vec<Vec<Cell>> = seeds
        .par_iter()
        .map(|&s| {
            let inst = build(w, s, widx);
            arms.iter()
                .map(|a| {
                    let r = run_arm(w, &inst, a, s);
                    Cell { m: summarize_run(&r, &inst), q: r.q }
                })
                .collect()
        })
        .collect();
    // transpose
    let mut out: Vec<Vec<Cell>> = (0..arms.len()).map(|_| Vec::new()).collect();
    for row in per_seed {
        for (ai, c) in row.into_iter().enumerate() {
            out[ai].push(c);
        }
    }
    out
}

fn col(cells: &[Cell], m: usize) -> Vec<f64> {
    cells.iter().map(|c| c.m[m]).collect()
}
fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

// ------------------------------------------------------------------------------------------------ tuning (seeds 1-8 only)
fn tune(ws: &[WorldCfg], seeds: &[u64]) -> (Tuned, Value) {
    let eval = |arms: &[Arm], metric: &dyn Fn(&[f64; NM]) -> f64| -> Vec<f64> {
        let mut tot = vec![0.0; arms.len()];
        for (wi, w) in ws.iter().enumerate() {
            let res = run_world(w, wi as u64, arms, seeds);
            for (ai, cells) in res.iter().enumerate() {
                tot[ai] += mean(&cells.iter().map(|c| metric(&c.m)).collect::<Vec<_>>()) / ws.len() as f64;
            }
        }
        tot
    };
    let graded = |m: &[f64; NM]| (m[mi("q_h250")] + m[mi("q_h500")] + m[mi("q_h1500")]) / 3.0;
    let mut log = serde_json::Map::new();
    // A: baselines
    let mut cand: Vec<Arm> = vec![];
    for (g, e) in [(1.0, 0.0), (0.9, 0.0), (1.0, 0.05), (0.9, 0.05)] {
        cand.push(Arm { name: format!("eps({g},{e})"), kind: Kind::Ss(SsKind::Eps(g, e), false) });
    }
    for g in [1.0, 0.9] {
        cand.push(Arm { name: format!("pts({g})"), kind: Kind::Ss(SsKind::Pts(g), false) });
    }
    for k in [1.0, 3.0, 8.0] {
        cand.push(Arm { name: format!("warm(k={k})"), kind: Kind::Ss(SsKind::Warm(k), false) });
    }
    let sc = eval(&cand, &graded);
    let pick = |lo: usize, hi: usize| (lo..hi).fold(lo, |b, i| if sc[i] > sc[b] { i } else { b });
    let (ie, ip, iw) = (pick(0, 4), pick(4, 6), pick(6, 9));
    for (a, s) in cand.iter().zip(&sc) {
        log.insert(a.name.clone(), json!(s));
    }
    let eps = match &cand[ie].kind {
        Kind::Ss(SsKind::Eps(g, e), _) => (*g, *e),
        _ => unreachable!(),
    };
    let pts = match &cand[ip].kind {
        Kind::Ss(SsKind::Pts(g), _) => *g,
        _ => unreachable!(),
    };
    let warm_k = match &cand[iw].kind {
        Kind::Ss(SsKind::Warm(k), _) => *k,
        _ => unreachable!(),
    };
    // B: orchestrator step-router k (objective: L2 graded)
    let mut cand: Vec<Arm> = vec![];
    for k in [1.0, 3.0, 8.0] {
        cand.push(Arm { name: format!("L2(k={k})"), kind: Kind::Orch(OrchCfg::level(2, k, 2.0, 0.05)) });
    }
    let sc = eval(&cand, &graded);
    let ib = (0..3).fold(0, |b, i| if sc[i] > sc[b] { i } else { b });
    for (a, s) in cand.iter().zip(&sc) {
        log.insert(a.name.clone(), json!(s));
    }
    let orch_k = [1.0, 3.0, 8.0][ib];
    // C: L4 stopping parameters (objective: utility at mu=.03, H1500; see board entry: the first run used mu=.1 and hit the never-verify grid edge)
    let mut cand: Vec<Arm> = vec![];
    let mut grid = vec![];
    for r in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
        for e in [0.02, 0.10] {
            cand.push(Arm { name: format!("L4(R={r},eps={e})"), kind: Kind::Orch(OrchCfg::level(4, orch_k, r, e)) });
            grid.push((r, e));
        }
    }
    let util = |m: &[f64; NM]| m[mi("util_c")];
    let sc = eval(&cand, &util);
    let ic = (0..cand.len()).fold(0, |b, i| if sc[i] > sc[b] { i } else { b });
    for (a, s) in cand.iter().zip(&sc) {
        log.insert(a.name.clone(), json!(s));
    }
    let t = Tuned { eps, pts, warm_k, orch_k, r_val: grid[ic].0, eps_ex: grid[ic].1 };
    (t.clone(), json!({"tuned": format!("{:?}", t), "objective_scores": Value::Object(log)}))
}

// ------------------------------------------------------------------------------------------------ tests
#[derive(Clone)]
struct Test {
    family: String,
    world: String,
    metric: String,
    treat: String,
    vs: String,
    d: PairedDiff,
    p: f64,
    adj: f64,
}
fn mk(family: &str, world: &str, metric: &str, treat: &str, vs: &str, t: &[f64], b: &[f64]) -> Test {
    Test { family: family.into(), world: world.into(), metric: metric.into(), treat: treat.into(), vs: vs.into(), d: paired_diff(t, b), p: paired_t_pvalue(t, b), adj: 1.0 }
}
fn holm_families(tests: &mut [Test]) {
    let mut fams: Vec<String> = tests.iter().map(|t| t.family.clone()).collect();
    fams.sort();
    fams.dedup();
    for f in fams {
        let idx: Vec<usize> = (0..tests.len()).filter(|&i| tests[i].family == f).collect();
        let adj = holm_adjust(&idx.iter().map(|&i| tests[i].p).collect::<Vec<_>>());
        for (k, &i) in idx.iter().enumerate() {
            tests[i].adj = adj[k];
        }
    }
}
fn verdict(t: &Test) -> &'static str {
    // lower is better for cost / lat
    let lower_better = t.metric == "cost" || t.metric == "lat" || t.metric == "tte" || t.metric == "tte_own";
    let good = if lower_better { t.d.diff.mean < 0.0 } else { t.d.diff.mean > 0.0 };
    if t.adj < 0.05 {
        if good {
            "WIN"
        } else {
            "LOSS"
        }
    } else {
        "ns"
    }
}
fn test_json(t: &Test) -> Value {
    json!({"family": t.family, "world": t.world, "metric": t.metric, "treatment": t.treat, "vs": t.vs, "mean": t.d.diff.mean, "lo": t.d.diff.lo, "hi": t.d.diff.hi,
           "wins": t.d.wins, "losses": t.d.losses, "p_raw": t.p, "holm_p": t.adj, "verdict": verdict(t)})
}
fn fmt_t(t: &Test) -> String {
    format!("{:+.4} [{:+.4},{:+.4}] {}", t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, verdict(t))
}

// ------------------------------------------------------------------------------------------------ main study
struct WorldRes {
    name: String,
    arms: Vec<String>,
    cells: Vec<Vec<Cell>>,
    bf: usize,
}

fn study(ws: &[WorldCfg], t: &Tuned, seeds: &[u64], log: &mut String) -> (Vec<WorldRes>, Vec<Test>) {
    use std::fmt::Write;
    let arms = arms_main(t);
    let names: Vec<String> = arms.iter().map(|a| a.name.clone()).collect();
    let ai = |n: &str| names.iter().position(|x| x == n).unwrap_or_else(|| panic!("arm {n}"));
    let mut results = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let t0 = Instant::now();
        let cells = run_world(w, wi as u64, &arms, seeds);
        let q15 = mi("q_h1500");
        let bf = BASELINES.iter().map(|b| ai(b)).fold(ai(BASELINES[0]), |b, i| if mean(&col(&cells[i], q15)) > mean(&col(&cells[b], q15)) { i } else { b });
        let _ = writeln!(log, "[{}] world {} done in {:.1}s, best fair = {}", wi, w.name, t0.elapsed().as_secs_f64(), names[bf]);
        results.push(WorldRes { name: w.name.to_string(), arms: names.clone(), cells, bf });
    }
    let mut tests: Vec<Test> = vec![];
    let lv = ["L1", "L2", "L3", "L4", "L5"];
    for wr in &results {
        let cells = &wr.cells;
        let bf = wr.bf;
        let bfname = names[bf].clone();
        let get = |a: usize, m: &str| col(&cells[a], mi(m));
        // F1: each level vs best fair single-shot
        for l in lv {
            for m in ["q_h1500", "q_h250", "ok_h1500", "qc_h1500", "util_c", "util_a"] {
                tests.push(mk("F1-vs-bestfair", &wr.name, m, l, &bfname, &get(ai(l), m), &get(bf, m)));
            }
        }
        // F2: level vs previous level
        let pairs = [("L2", "L1"), ("L3", "L2"), ("L4", "L3"), ("L5", "L4"), ("L3", "L3-nolearn"), ("L4", "L2")];
        for (a, b) in pairs {
            for m in ["q_h1500", "ok_h1500", "qc_h1500", "util_c", "util_a", "cost", "lat"] {
                tests.push(mk("F2-vs-prev", &wr.name, m, a, b, &get(ai(a), m), &get(ai(b), m)));
            }
        }
        // F4: vs single-shot with the same notes
        let sr = ai("ss-warm+retr");
        for l in ["L2", "L3", "L4", "L5"] {
            for m in ["q_h1500", "ok_h1500", "util_c"] {
                tests.push(mk("F4-vs-ss+retr", &wr.name, m, l, "ss-warm+retr", &get(ai(l), m), &get(sr, m)));
            }
        }
        // F3: sample efficiency (tasks to reach 95% of best-fair late quality), per seed target
        let tgt: Vec<f64> = cells[bf].iter().map(|c| 0.95 * tail_mean(&c.q, 300)).collect();
        let tte = |a: usize| -> Vec<f64> { cells[a].iter().zip(&tgt).map(|(c, &g)| tasks_to(&c.q, g)).collect() };
        let bt = tte(bf);
        for l in lv {
            tests.push(mk("F3-tte", &wr.name, "tte", l, &bfname, &tte(ai(l)), &bt));
        }
        // sample efficiency vs OWN asymptote: tasks until trailing-100 reaches 95% of the arm's own last-300 quality
        let tte_own = |a: usize| -> Vec<f64> { cells[a].iter().map(|c| tasks_to(&c.q, 0.95 * tail_mean(&c.q, 300))).collect() };
        let bto = tte_own(bf);
        for l in lv {
            tests.push(mk("F3-tte-own", &wr.name, "tte_own", l, &bfname, &tte_own(ai(l)), &bto));
        }
    }
    holm_families(&mut tests);
    (results, tests)
}

fn find<'a>(tests: &'a [Test], fam: &str, world: &str, metric: &str, treat: &str, vs: Option<&str>) -> Option<&'a Test> {
    tests.iter().find(|t| t.family == fam && t.world == world && t.metric == metric && t.treat == treat && vs.is_none_or(|v| t.vs == v))
}

fn report(results: &[WorldRes], tests: &[Test], log: &mut String) {
    use std::fmt::Write;
    let m = |n: &str| mi(n);
    for wr in results {
        let _ = writeln!(log, "\n=== {} (best fair single-shot: {})", wr.name, wr.arms[wr.bf]);
        let _ = writeln!(log, "{:<20} {:>7} {:>7} {:>7} {:>7} {:>6} {:>7} {:>7} {:>7} {:>6} {:>6}", "arm", "q250", "q1500", "ok1500", "q/cost", "cost", "u.03", "u.1", "u.3", "lat", "vcall");
        for (ai, a) in wr.arms.iter().enumerate() {
            let c = &wr.cells[ai];
            let g = |n: &str| mean(&col(c, m(n)));
            let _ = writeln!(log, "{:<20} {:>7.3} {:>7.3} {:>7.3} {:>7.4} {:>6.2} {:>7.3} {:>7.3} {:>7.3} {:>6.2} {:>6.2}", a, g("q_h250"), g("q_h1500"), g("ok_h1500"), g("qc_h1500"), g("cost"), g("util_c"), g("util_a"), g("util_b"), g("lat"), g("vcalls"));
        }
        let _ = writeln!(log, "-- level vs best fair single-shot [95% CI, Holm]: graded H1500 | all-correct H1500 | Q/C | util(.03) | util(.1) | tasks-to-bf-target | tasks-to-own-95%");
        for l in ["L1", "L2", "L3", "L4", "L5"] {
            let f = |mt: &str| find(tests, "F1-vs-bestfair", &wr.name, mt, l, None).map(fmt_t).unwrap_or_default();
            let tt = find(tests, "F3-tte", &wr.name, "tte", l, None).map(|t| format!("{:+.0} {}", t.d.diff.mean, verdict(t))).unwrap_or_default();
            let to = find(tests, "F3-tte-own", &wr.name, "tte_own", l, None).map(|t| format!("{:+.0} {}", t.d.diff.mean, verdict(t))).unwrap_or_default();
            let _ = writeln!(log, "  {l}: q {} | ok {} | qc {} | u.03 {} | u.1 {} | tte-bf {tt} | tte-own {to}", f("q_h1500"), f("ok_h1500"), f("qc_h1500"), f("util_c"), f("util_a"));
        }
        let _ = writeln!(log, "-- level vs previous level (F2):");
        for (a, b) in [("L2", "L1"), ("L3", "L2"), ("L4", "L3"), ("L5", "L4"), ("L3", "L3-nolearn")] {
            let f = |mt: &str| find(tests, "F2-vs-prev", &wr.name, mt, a, Some(b)).map(fmt_t).unwrap_or_default();
            let _ = writeln!(log, "  {a} - {b}: q {} | ok {} | cost {} | lat {} | u.03 {} | u.1 {}", f("q_h1500"), f("ok_h1500"), f("cost"), f("lat"), f("util_c"), f("util_a"));
        }
        let _ = writeln!(log, "-- vs single-shot WITH the same notes (ss-warm+retr) (F4): L2 q {} | L3 q {} | L4 q {} | L4 u.03 {}", find(tests, "F4-vs-ss+retr", &wr.name, "q_h1500", "L2", None).map(fmt_t).unwrap_or_default(), find(tests, "F4-vs-ss+retr", &wr.name, "q_h1500", "L3", None).map(fmt_t).unwrap_or_default(), find(tests, "F4-vs-ss+retr", &wr.name, "q_h1500", "L4", None).map(fmt_t).unwrap_or_default(), find(tests, "F4-vs-ss+retr", &wr.name, "util_c", "L4", None).map(fmt_t).unwrap_or_default());
    }
}

fn summary_counts(results: &[WorldRes], tests: &[Test], log: &mut String) -> Value {
    use std::fmt::Write;
    let ws: Vec<&str> = results.iter().map(|r| r.name.as_str()).collect();
    let gens: [(&str, Vec<&str>); 3] = [
        ("genA", ws.iter().copied().filter(|w| w.starts_with("pipe-")).collect()),
        ("genB", ws.iter().copied().filter(|w| w.starts_with("lat-")).collect()),
        ("controls", ws.iter().copied().filter(|w| w.starts_with("ctl-")).collect()),
    ];
    let mut out = serde_json::Map::new();
    let _ = writeln!(log, "\n=== Holm WIN/LOSS counts vs best fair single-shot (worlds with WIN / LOSS / ns, of n):");
    for (metric, label) in [("q_h1500", "graded H1500"), ("ok_h1500", "all-correct H1500"), ("qc_h1500", "quality/cost"), ("util_c", "utility mu=.03"), ("util_a", "utility mu=.1"), ("q_h250", "graded H250")] {
        for l in ["L1", "L2", "L3", "L4", "L5"] {
            let mut parts = vec![];
            for (gn, wl) in &gens {
                let (mut w, mut lo, mut ns) = (0, 0, 0);
                for wn in wl {
                    match find(tests, "F1-vs-bestfair", wn, metric, l, None).map(verdict) {
                        Some("WIN") => w += 1,
                        Some("LOSS") => lo += 1,
                        _ => ns += 1,
                    }
                }
                parts.push(format!("{gn} {w}W/{lo}L/{ns}ns of {}", wl.len()));
                out.insert(format!("{metric}|{l}|{gn}"), json!([w, lo, ns]));
            }
            let _ = writeln!(log, "  {label:<18} {l}: {}", parts.join(" ; "));
        }
    }
    let _ = writeln!(log, "=== Holm WIN/LOSS vs previous level (graded H1500 | util | cost | lat):");
    for (a, b) in [("L2", "L1"), ("L3", "L2"), ("L4", "L3"), ("L5", "L4")] {
        for metric in ["q_h1500", "ok_h1500", "util_c", "util_a", "cost", "lat"] {
            let mut parts = vec![];
            for (gn, wl) in &gens {
                let (mut w, mut lo, mut ns) = (0, 0, 0);
                for wn in wl {
                    match find(tests, "F2-vs-prev", wn, metric, a, Some(b)).map(verdict) {
                        Some("WIN") => w += 1,
                        Some("LOSS") => lo += 1,
                        _ => ns += 1,
                    }
                }
                parts.push(format!("{gn} {w}W/{lo}L/{ns}ns"));
                out.insert(format!("prev|{metric}|{a}-{b}|{gn}"), json!([w, lo, ns]));
            }
            let _ = writeln!(log, "  {a}-{b} {metric:<8}: {}", parts.join(" ; "));
        }
    }
    let _ = writeln!(log, "=== Holm WIN/LOSS vs single-shot WITH THE SAME NOTES (ss-warm+retr), F4:");
    for metric in ["q_h1500", "ok_h1500", "util_c"] {
        for l in ["L2", "L3", "L4", "L5"] {
            let mut parts = vec![];
            for (gn, wl) in &gens {
                let (mut w, mut lo, mut ns) = (0, 0, 0);
                for wn in wl {
                    match find(tests, "F4-vs-ss+retr", wn, metric, l, None).map(verdict) {
                        Some("WIN") => w += 1,
                        Some("LOSS") => lo += 1,
                        _ => ns += 1,
                    }
                }
                parts.push(format!("{gn} {w}W/{lo}L/{ns}ns of {}", wl.len()));
                out.insert(format!("ssretr|{metric}|{l}|{gn}"), json!([w, lo, ns]));
            }
            let _ = writeln!(log, "  {metric:<8} {l} vs ss-warm+retr: {}", parts.join(" ; "));
        }
    }
    // mean deltas by group (graded H1500)
    let _ = writeln!(log, "=== mean paired deltas, graded H1500, by group (mean of per-world means):");
    for (a, b, fam) in [("L1", "bf", "F1-vs-bestfair"), ("L2", "L1", "F2-vs-prev"), ("L3", "L2", "F2-vs-prev"), ("L4", "L3", "F2-vs-prev"), ("L5", "L4", "F2-vs-prev"), ("L3", "L3-nolearn", "F2-vs-prev")] {
        let mut parts = vec![];
        for (gn, wl) in &gens {
            let ds: Vec<f64> = wl.iter().filter_map(|wn| find(tests, fam, wn, "q_h1500", a, if fam == "F2-vs-prev" { Some(b) } else { None })).map(|t| t.d.diff.mean).collect();
            let ok: Vec<f64> = wl.iter().filter_map(|wn| find(tests, fam, wn, "ok_h1500", a, if fam == "F2-vs-prev" { Some(b) } else { None })).map(|t| t.d.diff.mean).collect();
            parts.push(format!("{gn} q {:+.4} ok {:+.4}", mean(&ds), mean(&ok)));
            out.insert(format!("mean|{a}-{b}|{gn}"), json!([mean(&ds), mean(&ok)]));
        }
        let _ = writeln!(log, "  {a} - {b}: {}", parts.join(" ; "));
    }
    Value::Object(out)
}

// ------------------------------------------------------------------------------------------------ sensitivity
fn sens(t: &Tuned, seeds: &[u64], log: &mut String) -> Value {
    use std::fmt::Write;
    let base = worlds();
    let pm = base.iter().find(|w| w.name == "pipe-main").unwrap().clone();
    let lm = base.iter().find(|w| w.name == "lat-main").unwrap().clone();
    let mut variants: Vec<(String, WorldCfg, u64)> = vec![];
    let mut push = |label: String, w: WorldCfg, idx: u64| variants.push((label, w, idx));
    for g in [0.0, 0.1, 0.26] {
        push(format!("pipe spec_gap={g}"), WorldCfg { spec_gap: g, ..pm.clone() }, 0);
    }
    for p in [0.0, 0.5, 0.9] {
        push(format!("pipe prop={p}"), WorldCfg { prop: p, ..pm.clone() }, 0);
        push(format!("lat prop={p}"), WorldCfg { prop: p, ..lm.clone() }, 6);
    }
    for s in [0.05, 0.10, 0.22, 0.40] {
        push(format!("pipe sig_v={s}"), WorldCfg { sig_v: s, ..pm.clone() }, 0);
    }
    for cv in [0.1, 0.3, 0.8, 1.5] {
        push(format!("pipe cv={cv}"), WorldCfg { cv, ..pm.clone() }, 0);
    }
    for m in [0.0, 0.05, 0.10] {
        push(format!("pipe mono_pen={m}"), WorldCfg { mono_pen: m, ..pm.clone() }, 0);
    }
    for n in [0.06, 0.10, 0.16] {
        push(format!("pipe noise={n}"), WorldCfg { noise: n, ..pm.clone() }, 0);
    }
    for d in [0.0, 0.5, 1.5] {
        push(format!("pipe dscale={d}"), WorldCfg { dscale: d, ..pm.clone() }, 0);
    }
    let rare = base.iter().find(|w| w.name == "ctl-rare-pricever").unwrap().clone();
    for cv in [0.1, 0.3, 1.0] {
        push(format!("rare(HighSkill) cv={cv}"), WorldCfg { cv, ..rare.clone() }, 13);
    }
    let arms = arms_main(t);
    let names: Vec<String> = arms.iter().map(|a| a.name.clone()).collect();
    let ai = |n: &str| names.iter().position(|x| x == n).unwrap();
    let _ = writeln!(log, "\n=== SENSITIVITY (one-at-a-time from pipe-main / lat-main, held seeds, frozen tuning). deltas are paired means [95% CI]");
    let _ = writeln!(log, "{:<22} {:>6} | {:>22} | {:>22} | {:>22} | {:>10} | {:>22} | {:>9} | {:>22}", "variant", "bf q", "L1-bf q", "L2-L1 q", "L3-L2 q", "L3-L2 cost", "L4-L3 q", "L4-L3 cost", "L4-L2 util.03");
    let mut js = vec![];
    for (label, w, idx) in &variants {
        let cells = run_world(w, *idx, &arms, seeds);
        let q15 = mi("q_h1500");
        let bf = BASELINES.iter().map(|b| ai(b)).fold(ai(BASELINES[0]), |b, i| if mean(&col(&cells[i], q15)) > mean(&col(&cells[b], q15)) { i } else { b });
        let d = |a: &str, b: usize, m: &str| paired_diff(&col(&cells[ai(a)], mi(m)), &col(&cells[b], mi(m)));
        let f = |p: &PairedDiff| format!("{:+.4} [{:+.3},{:+.3}]", p.diff.mean, p.diff.lo, p.diff.hi);
        let (a, a2, b, bc, c4, c4c, u) = (d("L1", bf, "q_h1500"), d("L2", ai("L1"), "q_h1500"), d("L3", ai("L2"), "q_h1500"), d("L3", ai("L2"), "cost"), d("L4", ai("L3"), "q_h1500"), d("L4", ai("L3"), "cost"), d("L4", ai("L2"), "util_c"));
        let _ = writeln!(log, "{:<22} {:>6.3} | {:>22} | {:>22} | {:>22} | {:>10.2} | {:>22} | {:>9.2} | {:>22}", label, mean(&col(&cells[bf], q15)), f(&a), f(&a2), f(&b), bc.diff.mean, f(&c4), c4c.diff.mean, f(&u));
        let tri = |p: &PairedDiff| json!([p.diff.mean, p.diff.lo, p.diff.hi]);
        js.push(json!({"variant": label, "bf": names[bf], "bf_q": mean(&col(&cells[bf], q15)), "L1_minus_bf_q": tri(&a), "L2_minus_L1_q": tri(&a2), "L3_minus_L2_q": tri(&b), "L3_minus_L2_cost": tri(&bc),
            "L4_minus_L3_q": tri(&c4), "L4_minus_L3_cost": tri(&c4c), "L4_minus_L2_util_c": tri(&u)}));
    }
    Value::Array(js)
}

// ------------------------------------------------------------------------------------------------ main
fn arg(args: &[String], key: &str, default: &str) -> String {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}
fn range(s: &str) -> Vec<u64> {
    let (a, b) = s.split_once('-').expect("range a-b");
    (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ws = worlds();
    if args.iter().any(|a| a == "--show") {
        // world examples + non-policy oracle reference rows (seed 1)
        let mut s = String::new();
        let dummy = Tuned { eps: (1.0, 0.0), pts: 1.0, warm_k: 1.0, orch_k: 1.0, r_val: 2.0, eps_ex: 0.05 };
        for (wi, w) in ws.iter().enumerate() {
            s.push_str(&describe(w, wi as u64, 1));
            let inst = build(w, 1, wi as u64);
            let o1 = summarize_run(&run_single(w, &inst, &SsKind::OracleSingle, false, 1), &inst);
            let o2 = summarize_run(&run_oracle_decomp(w, &inst, false, 1), &inst);
            let o3 = summarize_run(&run_oracle_decomp(w, &inst, true, 1), &inst);
            s.push_str(&format!(
                "  oracle-single q {:.3} ok {:.3} | oracle-decomp q {:.3} ok {:.3} | oracle-decomp+ctx q {:.3} ok {:.3}\n",
                o1[mi("q_h1500")], o1[mi("ok_h1500")], o2[mi("q_h1500")], o2[mi("ok_h1500")], o3[mi("q_h1500")], o3[mi("ok_h1500")]
            ));
        }
        let _ = dummy;
        print!("{s}");
        return;
    }
    let held = range(&arg(&args, "--held", "31-60"));
    let tune_seeds = range(&arg(&args, "--tune", "1-8"));
    let out_path = arg(&args, "--out", "results/g18_orch.json");
    let t0 = Instant::now();
    let mut log = String::new();
    let (tuned, tune_json) = tune(&ws, &tune_seeds);
    log.push_str(&format!("tuned on seeds {:?}-{:?}: {:?}  ({:.0}s)\n", tune_seeds[0], tune_seeds.last().unwrap(), tuned, t0.elapsed().as_secs_f64()));
    if args.iter().any(|a| a == "--sens") {
        let sj = sens(&tuned, &held, &mut log);
        std::fs::write(&out_path, serde_json::to_string_pretty(&json!({"tuned": tune_json, "held": format!("{:?}", held), "sens": sj})).unwrap()).unwrap();
        print!("{log}");
        return;
    }
    let (results, tests) = study(&ws, &tuned, &held, &mut log);
    report(&results, &tests, &mut log);
    let counts = summary_counts(&results, &tests, &mut log);
    let mut worlds_json = BTreeMap::new();
    for wr in &results {
        let mut arms = BTreeMap::new();
        for (ai, a) in wr.arms.iter().enumerate() {
            let mut m = serde_json::Map::new();
            for (k, name) in METRICS.iter().enumerate() {
                m.insert(name.to_string(), json!(mean(&col(&wr.cells[ai], k))));
            }
            arms.insert(a.clone(), Value::Object(m));
        }
        worlds_json.insert(wr.name.clone(), json!({"best_fair": wr.arms[wr.bf], "arms": arms}));
    }
    let doc = json!({
        "protocol": {"tune_seeds": format!("{:?}", tune_seeds), "held_seeds": format!("{:?}", held), "n_tasks": N_TASKS, "reveal": REVEAL, "delay": DELAY, "theta": THETA, "theta_v": THETA_V, "max_retry": MAX_RETRY},
        "tuning": tune_json, "worlds": worlds_json, "counts": counts, "tests": tests.iter().map(test_json).collect::<Vec<_>>(),
        "runtime_s": t0.elapsed().as_secs_f64()
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    log.push_str(&format!("\nwrote {out_path} in {:.0}s\n", t0.elapsed().as_secs_f64()));
    print!("{log}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny(name: &str) -> (WorldCfg, Instance) {
        let w = worlds().into_iter().find(|w| w.name == name).unwrap();
        let inst = build(&w, 3, 0);
        (w, inst)
    }

    #[test]
    fn deterministic_and_cost_accounting() {
        let (w, inst) = tiny("pipe-main");
        let a = run_orch(&w, &inst, &OrchCfg::level(1, 1.0, 2.0, 0.05), 3);
        let b = run_orch(&w, &inst, &OrchCfg::level(1, 1.0, 2.0, 0.05), 3);
        assert_eq!(a.q, b.q);
        for (c, t) in a.cost.iter().zip(&inst.tasks) {
            assert!((*c as f64 - t.steps.len() as f64).abs() < 1e-4, "L1 cost = number of steps");
        }
        let l2 = run_orch(&w, &inst, &OrchCfg::level(2, 1.0, 2.0, 0.05), 3);
        for (c, t) in l2.cost.iter().zip(&inst.tasks) {
            assert!((*c as f64 - t.steps.len() as f64 * (1.0 + CR)).abs() < 1e-4, "L2 cost = steps * (1 + retrieval)");
        }
    }

    #[test]
    fn oracle_decomp_beats_oracle_single_with_specialists_and_fanout_cuts_latency() {
        let (w, inst) = tiny("pipe-main");
        let os = summarize_run(&run_single(&w, &inst, &SsKind::OracleSingle, false, 3), &inst);
        let od = summarize_run(&run_oracle_decomp(&w, &inst, false, 3), &inst);
        assert!(od[mi("q_h1500")] > os[mi("q_h1500")] + 0.05);
        let l4 = run_orch(&w, &inst, &OrchCfg::level(4, 1.0, 2.0, 0.05), 3);
        let l5 = run_orch(&w, &inst, &OrchCfg::level(5, 1.0, 2.0, 0.05), 3);
        assert!(mean_f32(&l5.lat) < mean_f32(&l4.lat));
    }

    #[test]
    fn dominant_agent_control_has_no_decomposition_headroom() {
        let (w, inst) = tiny("ctl-indep-dominant");
        let os = summarize_run(&run_single(&w, &inst, &SsKind::OracleSingle, false, 3), &inst);
        let od = summarize_run(&run_oracle_decomp(&w, &inst, false, 3), &inst);
        // oracle decomposition can only exploit keyword-decoy-free per-type bests, which here coincide with the dominant agent
        assert!((od[mi("q_h1500")] - os[mi("q_h1500")]).abs() < 0.03);
    }

    #[test]
    fn single_step_world_has_one_step() {
        let (_, inst) = tiny("ctl-single-step");
        assert!(inst.tasks.iter().all(|t| t.steps.len() == 1));
    }
}
