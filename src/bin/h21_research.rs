//! H21 (R9-E): multi-step RESEARCH sessions end to end. Single-shot routing (6 fair baselines + react / best-of-3 controls) vs orchestrator levels
//!   L1 decompose+route; L2 + per-step retrieval; L2t + tools (exact calculator, exact db lookup); L3x + per-step extraction verification + retry;
//!   L4 + posterior-gated verification (verifygate::PosteriorGate); L4o (g18 online stopping, comparison); L5 + replan-on-failure with query reformulation;
//!   BoK (L2t + per-step best-of-3); Stack (L5 + cross-source READ verification + vote-when-posterior-low); leave-one-out ablations of Stack.
//! Everything is a seeded simulation (src/bin/h21_common/mod.rs). Protocol: tune seeds 1-8, held 31-60 (`--held 31-60`), fresh confirm 61-90
//! (`--held 61-90`, SAME frozen tuning because tuning always reruns on seeds 1-8), paired 95% t-CI, Holm per family.
//!
//! Usage: cargo run --release --offline --bin h21_research -- [--show] [--held 31-60] [--out results/h21_research.json] [--sens]

use std::collections::BTreeMap;
use std::time::Instant;

use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, PairedDiff};
use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "h21_common/mod.rs"]
mod c;
use c::*;

// ------------------------------------------------------------------------------------------------ arms
#[derive(Clone, Debug)]
enum Kind {
    Ss { kind: SsKind, react: bool, k: usize },
    Orch(Cfg),
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
    rv: f64,
    r_on: f64,
    eps_on: f64,
    theta: f64,
    /// per-world gate scale tuned on seeds 1-8 for that world only (arms L4w / Stack-w)
    rv_world: Vec<f64>,
}

const RVS: [f64; 8] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];

/// per-world gate scale (tune seeds only) for an arbitrary world list
fn rv_world_for(ws: &[WorldCfg], idx0: u64, seeds: &[u64], orch_k: f64) -> Vec<f64> {
    let cand: Vec<Arm> = RVS.iter().map(|&rv| Arm { name: format!("L4(rv={rv})"), kind: Kind::Orch(Cfg::level("L4", orch_k, rv, 1.0, 0.05, 0.0)) }).collect();
    ws.iter().enumerate().map(|(wi, w)| {
        let res = run_world(w, idx0 + wi as u64, &cand, seeds);
        let per: Vec<f64> = res.iter().map(|cells| mean(&cells.iter().map(|c| c.m[mi("util_c")]).collect::<Vec<_>>())).collect();
        RVS[(0..RVS.len()).fold(0, |b, i| if per[i] > per[b] { i } else { b })]
    }).collect()
}

const BASELINES: [&str; 6] = ["ss-keyword", "ss-eps", "ss-pts", "ss-hebr", "ss-warm", "ss-safe"];
const STRONG: [&str; 8] = ["ss-keyword", "ss-eps", "ss-pts", "ss-hebr", "ss-warm", "ss-safe", "ss-react", "ss-bok3"];
const LEVELS: [&str; 11] = ["L1", "L2", "L2t", "L3x", "L4", "L4o", "L5", "BoK", "Stack", "L4w", "Stack-w"];
const ABL: [&str; 8] = ["S-noretr", "S-notools", "S-noverify", "S-nogate", "S-noreplan", "S-norouting", "S-nocross", "S-novote"];

fn lv(t: &Tuned, name: &str) -> Cfg {
    Cfg::level(name, t.orch_k, t.rv, t.r_on, t.eps_on, t.theta)
}

fn arms_main(t: &Tuned, wi: usize) -> Vec<Arm> {
    let rvw = t.rv_world.get(wi).copied().unwrap_or(t.rv);
    let ss = |n: &str, kind: SsKind, react: bool, k: usize| Arm { name: n.into(), kind: Kind::Ss { kind, react, k } };
    let mut v = vec![
        ss("ss-keyword", SsKind::Keyword, false, 1),
        ss("ss-eps", SsKind::Eps(t.eps.0, t.eps.1), false, 1),
        ss("ss-pts", SsKind::Pts(t.pts), false, 1),
        ss("ss-hebr", SsKind::HebR, false, 1),
        ss("ss-warm", SsKind::Warm(t.warm_k), false, 1),
        ss("ss-safe", SsKind::Safe, false, 1),
        ss("ss-react", SsKind::Warm(t.warm_k), true, 1),
        ss("ss-bok3", SsKind::Warm(t.warm_k), false, 3),
        ss("oracle-single", SsKind::OracleSingle, false, 1),
    ];
    v.push(Arm { name: "oracle-decomp".into(), kind: Kind::Orch(Cfg { route: Route::Oracle, ..Cfg::l1() }) });
    v.push(Arm { name: "oracle-stack".into(), kind: Kind::Orch(Cfg { route: Route::Oracle, ..lv(t, "Stack") }) });
    for l in ["L1", "L2", "L2t", "L3x", "L4", "L4o", "L5", "BoK", "Stack"] {
        v.push(Arm { name: l.into(), kind: Kind::Orch(lv(t, l)) });
    }
    v.push(Arm { name: "L4w".into(), kind: Kind::Orch(Cfg::level("L4", t.orch_k, rvw, t.r_on, t.eps_on, t.theta)) });
    v.push(Arm { name: "Stack-w".into(), kind: Kind::Orch(Cfg::level("Stack", t.orch_k, rvw, t.r_on, t.eps_on, t.theta)) });
    let s = lv(t, "Stack");
    let abl: Vec<(&str, Cfg)> = vec![
        ("S-noretr", Cfg { retr: false, ..s }),
        ("S-notools", Cfg { tools: false, ..s }),
        ("S-noverify", Cfg { ver: Ver::None, cross: false, ..s }),
        ("S-nogate", Cfg { ver: Ver::Always, ..s }),
        ("S-noreplan", Cfg { replan: false, ..s }),
        ("S-norouting", Cfg { route: Route::Random, ..s }),
        ("S-nocross", Cfg { cross: false, ..s }),
        ("S-novote", Cfg { vote: Vote::Off, ..s }),
    ];
    for (n, cfg) in abl {
        v.push(Arm { name: n.into(), kind: Kind::Orch(cfg) });
    }
    v
}

fn run_arm(w: &WorldCfg, inst: &Instance, arm: &Arm, seed: u64) -> RunOut {
    match &arm.kind {
        Kind::Ss { kind, react, k } => run_single(w, inst, kind, *react, *k, seed),
        Kind::Orch(cfg) => run_orch(w, inst, cfg, seed),
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
                    Cell { m: summarize_run(&r), q: r.q }
                })
                .collect()
        })
        .collect();
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
fn tune(ws: &[WorldCfg], seeds: &[u64], dummy: &Tuned) -> (Tuned, Value) {
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
    let graded = |m: &[f64; NM]| (m[mi("q_h250")] + m[mi("q_h1000")]) / 2.0;
    let util = |m: &[f64; NM]| m[mi("util_c")];
    let mut log = serde_json::Map::new();
    let ss = |n: String, kind: SsKind| Arm { name: n, kind: Kind::Ss { kind, react: false, k: 1 } };
    // A: baselines
    let mut cand: Vec<Arm> = vec![];
    for (g, e) in [(1.0, 0.0), (0.9, 0.0), (1.0, 0.05), (0.9, 0.05)] {
        cand.push(ss(format!("eps({g},{e})"), SsKind::Eps(g, e)));
    }
    for g in [1.0, 0.9] {
        cand.push(ss(format!("pts({g})"), SsKind::Pts(g)));
    }
    for k in [1.0, 3.0, 8.0] {
        cand.push(ss(format!("warm(k={k})"), SsKind::Warm(k)));
    }
    let sc = eval(&cand, &graded);
    let pick = |sc: &[f64], lo: usize, hi: usize| (lo..hi).fold(lo, |b, i| if sc[i] > sc[b] { i } else { b });
    let (ie, ip, iw) = (pick(&sc, 0, 4), pick(&sc, 4, 6), pick(&sc, 6, 9));
    for (a, s) in cand.iter().zip(&sc) {
        log.insert(a.name.clone(), json!(s));
    }
    let eps = match &cand[ie].kind {
        Kind::Ss { kind: SsKind::Eps(g, e), .. } => (*g, *e),
        _ => unreachable!(),
    };
    let pts = match &cand[ip].kind {
        Kind::Ss { kind: SsKind::Pts(g), .. } => *g,
        _ => unreachable!(),
    };
    let warm_k = match &cand[iw].kind {
        Kind::Ss { kind: SsKind::Warm(k), .. } => *k,
        _ => unreachable!(),
    };
    // B: orchestrator step-router k (objective: L2t graded)
    let mut cand: Vec<Arm> = vec![];
    for k in [1.0, 3.0, 8.0, 20.0] {
        cand.push(Arm { name: format!("L2t(k={k})"), kind: Kind::Orch(Cfg::level("L2t", k, 1.0, 1.0, 0.05, 0.0)) });
    }
    let sc = eval(&cand, &graded);
    for (a, s) in cand.iter().zip(&sc) {
        log.insert(a.name.clone(), json!(s));
    }
    let orch_k = [1.0, 3.0, 8.0, 20.0][pick(&sc, 0, 4)];
    // C: L4 posterior gate rv (objective: utility at mu=.03). Global pick over all worlds (arm L4/Stack) AND per-world picks (arms L4w/Stack-w)
    let rvs = RVS;
    let cand: Vec<Arm> = rvs.iter().map(|&rv| Arm { name: format!("L4(rv={rv})"), kind: Kind::Orch(Cfg::level("L4", orch_k, rv, 1.0, 0.05, 0.0)) }).collect();
    let mut sc = vec![0.0; cand.len()];
    let mut rv_world = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let res = run_world(w, wi as u64, &cand, seeds);
        let per: Vec<f64> = res.iter().map(|cells| mean(&cells.iter().map(|c| util(&c.m)).collect::<Vec<_>>())).collect();
        for (i, x) in per.iter().enumerate() {
            sc[i] += x / ws.len() as f64;
        }
        rv_world.push(rvs[pick(&per, 0, rvs.len())]);
    }
    for (a, s) in cand.iter().zip(&sc) {
        log.insert(a.name.clone(), json!(s));
    }
    log.insert("rv_world".into(), json!(rv_world.clone()));
    let rv = rvs[pick(&sc, 0, rvs.len())];
    // D: L4o online stopping (comparison arm): R x eps
    let mut cand: Vec<Arm> = vec![];
    let mut grid = vec![];
    for r in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0] {
        for e in [0.02, 0.10] {
            cand.push(Arm { name: format!("L4o(R={r},eps={e})"), kind: Kind::Orch(Cfg::level("L4o", orch_k, rv, r, e, 0.0)) });
            grid.push((r, e));
        }
    }
    let sc = eval(&cand, &util);
    for (a, s) in cand.iter().zip(&sc) {
        log.insert(a.name.clone(), json!(s));
    }
    let ig = pick(&sc, 0, cand.len());
    // E: Stack vote threshold theta (0 = vote off)
    let ths = [0.0, 0.3, 0.45, 0.6, 0.75];
    let cand: Vec<Arm> = ths.iter().map(|&th| Arm { name: format!("Stack(theta={th})"), kind: Kind::Orch(Cfg::level("Stack", orch_k, rv, 1.0, 0.05, th)) }).collect();
    let sc = eval(&cand, &util);
    for (a, s) in cand.iter().zip(&sc) {
        log.insert(a.name.clone(), json!(s));
    }
    let theta = ths[pick(&sc, 0, ths.len())];
    let _ = dummy;
    let t = Tuned { eps, pts, warm_k, orch_k, rv, r_on: grid[ig].0, eps_on: grid[ig].1, theta, rv_world };
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
    let lower_better = t.metric == "cost" || t.metric == "calls" || t.metric == "vcalls" || t.metric == "tte" || t.metric == "tte_own";
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

struct WorldRes {
    name: String,
    arms: Vec<String>,
    cells: Vec<Vec<Cell>>,
    bf: usize,
    bs: usize,
}

fn best_of(names: &[String], cells: &[Vec<Cell>], set: &[&str], m: usize) -> usize {
    let ai = |n: &str| names.iter().position(|x| x == n).unwrap_or_else(|| panic!("arm {n}"));
    set.iter().map(|b| ai(b)).fold(ai(set[0]), |b, i| if mean(&col(&cells[i], m)) > mean(&col(&cells[b], m)) { i } else { b })
}

fn study(ws: &[WorldCfg], t: &Tuned, seeds: &[u64], log: &mut String) -> (Vec<WorldRes>, Vec<Test>) {
    use std::fmt::Write;
    let names: Vec<String> = arms_main(t, 0).iter().map(|a| a.name.clone()).collect();
    let ai = |n: &str| names.iter().position(|x| x == n).unwrap_or_else(|| panic!("arm {n}"));
    let mut results = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let t0 = Instant::now();
        let arms = arms_main(t, wi);
        let widx = if w.name.starts_with("xtr-") { 100 + wi as u64 } else { wi as u64 };
        let cells = run_world(w, widx, &arms, seeds);
        let q = mi("q_h1000");
        let bf = best_of(&names, &cells, &BASELINES, q);
        let bs = best_of(&names, &cells, &STRONG, q);
        let _ = writeln!(log, "[{}] world {} done in {:.1}s, best fair plain = {}, best strong = {}", wi, w.name, t0.elapsed().as_secs_f64(), names[bf], names[bs]);
        results.push(WorldRes { name: w.name.to_string(), arms: names.clone(), cells, bf, bs });
    }
    let mut tests: Vec<Test> = vec![];
    for wr in &results {
        let cells = &wr.cells;
        let (bf, bs) = (wr.bf, wr.bs);
        let (bfname, bsname) = (names[bf].clone(), names[bs].clone());
        let get = |a: usize, m: &str| col(&cells[a], mi(m));
        for l in LEVELS {
            for m in ["q_h1000", "q_h250", "ok_h1000", "fin_h1000", "qpc", "okpc", "util_c", "util_a"] {
                tests.push(mk("F1-vs-bestfair", &wr.name, m, l, &bfname, &get(ai(l), m), &get(bf, m)));
            }
        }
        for l in ["L2t", "L5", "Stack"] {
            for m in ["q_h1000", "ok_h1000", "okpc", "qpc", "util_c"] {
                tests.push(mk("F1s-vs-beststrong", &wr.name, m, l, &bsname, &get(ai(l), m), &get(bs, m)));
            }
        }
        let pairs = [("L2", "L1"), ("L2t", "L2"), ("L3x", "L2t"), ("L4", "L3x"), ("L5", "L4"), ("Stack", "L5"), ("L4", "L4o"), ("Stack", "BoK"), ("BoK", "L2t"), ("L4w", "L4"), ("L4w", "L3x"), ("Stack-w", "Stack")];
        for (a, b) in pairs {
            for m in ["q_h1000", "ok_h1000", "fin_h1000", "cost", "okpc", "util_c", "util_a", "vcalls"] {
                tests.push(mk("F2-vs-prev", &wr.name, m, a, b, &get(ai(a), m), &get(ai(b), m)));
            }
        }
        for l in ["L2t", "L3x", "L4", "L5", "Stack"] {
            for m in ["q_h1000", "ok_h1000", "okpc", "util_c", "cost"] {
                tests.push(mk("F4-vs-react", &wr.name, m, l, "ss-react", &get(ai(l), m), &get(ai("ss-react"), m)));
            }
        }
        for a in ABL {
            for m in ["q_h1000", "ok_h1000", "cost", "okpc", "util_c"] {
                tests.push(mk("F5-ablation", &wr.name, m, a, "Stack", &get(ai(a), m), &get(ai("Stack"), m)));
            }
        }
        // F3: sample efficiency
        let tgt: Vec<f64> = cells[bf].iter().map(|c| 0.95 * tail_mean(&c.q, 250)).collect();
        let tte = |a: usize| -> Vec<f64> { cells[a].iter().zip(&tgt).map(|(c, &g)| tasks_to(&c.q, g)).collect() };
        let bt = tte(bf);
        for l in LEVELS {
            tests.push(mk("F3-tte", &wr.name, "tte", l, &bfname, &tte(ai(l)), &bt));
        }
        let tte_own = |a: usize| -> Vec<f64> { cells[a].iter().map(|c| tasks_to(&c.q, 0.95 * tail_mean(&c.q, 250))).collect() };
        let bto = tte_own(bf);
        for l in LEVELS {
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
    for wr in results {
        let _ = writeln!(log, "\n=== {} (best fair plain single-shot: {}; best strong: {})", wr.name, wr.arms[wr.bf], wr.arms[wr.bs]);
        let _ = writeln!(log, "{:<14} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>7} {:>7} {:>6} {:>6} {:>6} {:>5}", "arm", "q250", "q1000", "ok250", "ok1000", "fin", "cost", "q/cost", "ok/cost", "u.03", "calls", "vcall", "blk");
        for (ai, a) in wr.arms.iter().enumerate() {
            let c = &wr.cells[ai];
            let g = |n: &str| mean(&col(c, mi(n)));
            let _ = writeln!(
                log, "{:<14} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.2} {:>7.4} {:>7.4} {:>6.3} {:>6.2} {:>6.2} {:>5.2}",
                a, g("q_h250"), g("q_h1000"), g("ok_h250"), g("ok_h1000"), g("fin_h1000"), g("cost"), g("qpc"), g("okpc"), g("util_c"), g("calls"), g("vcalls"), g("blocked")
            );
        }
        let _ = writeln!(log, "-- level vs best fair plain [95% CI, Holm]: q | ok | okpc | u.03");
        for l in LEVELS {
            let f = |mt: &str| find(tests, "F1-vs-bestfair", &wr.name, mt, l, None).map(fmt_t).unwrap_or_default();
            let _ = writeln!(log, "  {l}: q {} | ok {} | okpc {} | u.03 {}", f("q_h1000"), f("ok_h1000"), f("okpc"), f("util_c"));
        }
        let _ = writeln!(log, "-- vs best STRONG single-shot ({}):", wr.arms[wr.bs]);
        for l in ["L2t", "L5", "Stack"] {
            let f = |mt: &str| find(tests, "F1s-vs-beststrong", &wr.name, mt, l, None).map(fmt_t).unwrap_or_default();
            let _ = writeln!(log, "  {l}: q {} | ok {} | okpc {} | u.03 {}", f("q_h1000"), f("ok_h1000"), f("okpc"), f("util_c"));
        }
        let _ = writeln!(log, "-- vs previous level (F2):");
        for (a, b) in [("L2", "L1"), ("L2t", "L2"), ("L3x", "L2t"), ("L4", "L3x"), ("L5", "L4"), ("Stack", "L5"), ("L4", "L4o"), ("BoK", "L2t")] {
            let f = |mt: &str| find(tests, "F2-vs-prev", &wr.name, mt, a, Some(b)).map(fmt_t).unwrap_or_default();
            let _ = writeln!(log, "  {a} - {b}: q {} | ok {} | cost {} | vcalls {} | okpc {} | u.03 {}", f("q_h1000"), f("ok_h1000"), f("cost"), f("vcalls"), f("okpc"), f("util_c"));
        }
        let _ = writeln!(log, "-- vs ss-react (F4): L2t ok {} | L5 ok {} | Stack ok {} | Stack okpc {} | Stack u.03 {}", find(tests, "F4-vs-react", &wr.name, "ok_h1000", "L2t", None).map(fmt_t).unwrap_or_default(), find(tests, "F4-vs-react", &wr.name, "ok_h1000", "L5", None).map(fmt_t).unwrap_or_default(), find(tests, "F4-vs-react", &wr.name, "ok_h1000", "Stack", None).map(fmt_t).unwrap_or_default(), find(tests, "F4-vs-react", &wr.name, "okpc", "Stack", None).map(fmt_t).unwrap_or_default(), find(tests, "F4-vs-react", &wr.name, "util_c", "Stack", None).map(fmt_t).unwrap_or_default());
        let _ = writeln!(log, "-- ablation: Stack minus lever (negative = the lever helps)  [ok | q | cost | okpc]:");
        for a in ABL {
            let f = |mt: &str| find(tests, "F5-ablation", &wr.name, mt, a, Some("Stack")).map(|t| format!("{:+.4}", t.d.diff.mean)).unwrap_or_default();
            let _ = writeln!(log, "  {a:<13} ok {} q {} cost {} okpc {}", f("ok_h1000"), f("q_h1000"), f("cost"), f("okpc"));
        }
    }
}

fn group_of(w: &str) -> &'static str {
    if w.starts_with("res-") {
        "genA"
    } else if w.starts_with("ana-") {
        "genB"
    } else {
        "controls"
    }
}

fn summary_counts(results: &[WorldRes], tests: &[Test], log: &mut String) -> Value {
    use std::fmt::Write;
    let gnames = ["genA", "genB", "controls"];
    let wl = |g: &str| -> Vec<&str> { results.iter().map(|r| r.name.as_str()).filter(|w| group_of(w) == g).collect() };
    let mut out = serde_json::Map::new();
    let cnt = |fam: &str, metric: &str, a: &str, b: Option<&str>, ws: &[&str]| -> (usize, usize, usize) {
        let (mut w, mut lo, mut ns) = (0, 0, 0);
        for wn in ws {
            match find(tests, fam, wn, metric, a, b).map(verdict) {
                Some("WIN") => w += 1,
                Some("LOSS") => lo += 1,
                _ => ns += 1,
            }
        }
        (w, lo, ns)
    };
    let _ = writeln!(log, "\n=== Holm WIN/LOSS counts vs best fair PLAIN single-shot (worlds WIN/LOSS/ns):");
    for (metric, label) in [("q_h1000", "graded H1000"), ("ok_h1000", "all-correct H1000"), ("okpc", "all-correct per cost"), ("qpc", "quality per cost"), ("util_c", "utility mu=.03"), ("util_a", "utility mu=.1"), ("q_h250", "graded H250")] {
        for l in LEVELS {
            let mut parts = vec![];
            for g in gnames {
                let (w, lo, ns) = cnt("F1-vs-bestfair", metric, l, None, &wl(g));
                parts.push(format!("{g} {w}W/{lo}L/{ns}ns of {}", wl(g).len()));
                out.insert(format!("{metric}|{l}|{g}"), json!([w, lo, ns]));
            }
            let _ = writeln!(log, "  {label:<20} {l:<6}: {}", parts.join(" ; "));
        }
    }
    let _ = writeln!(log, "=== vs best STRONG single-shot (incl. ss-react, ss-bok3):");
    for metric in ["q_h1000", "ok_h1000", "okpc", "util_c"] {
        for l in ["L2t", "L5", "Stack"] {
            let mut parts = vec![];
            for g in gnames {
                let (w, lo, ns) = cnt("F1s-vs-beststrong", metric, l, None, &wl(g));
                parts.push(format!("{g} {w}W/{lo}L/{ns}ns"));
                out.insert(format!("strong|{metric}|{l}|{g}"), json!([w, lo, ns]));
            }
            let _ = writeln!(log, "  {metric:<8} {l:<6}: {}", parts.join(" ; "));
        }
    }
    let _ = writeln!(log, "=== vs previous level (F2):");
    for (a, b) in [("L2", "L1"), ("L2t", "L2"), ("L3x", "L2t"), ("L4", "L3x"), ("L5", "L4"), ("Stack", "L5"), ("L4", "L4o"), ("BoK", "L2t")] {
        for metric in ["q_h1000", "ok_h1000", "cost", "okpc", "util_c", "util_a"] {
            let mut parts = vec![];
            for g in gnames {
                let (w, lo, ns) = cnt("F2-vs-prev", metric, a, Some(b), &wl(g));
                parts.push(format!("{g} {w}W/{lo}L/{ns}ns"));
                out.insert(format!("prev|{metric}|{a}-{b}|{g}"), json!([w, lo, ns]));
            }
            let _ = writeln!(log, "  {a}-{b} {metric:<8}: {}", parts.join(" ; "));
        }
    }
    let _ = writeln!(log, "=== vs ss-react (F4):");
    for metric in ["q_h1000", "ok_h1000", "okpc", "util_c"] {
        for l in ["L2t", "L3x", "L4", "L5", "Stack"] {
            let mut parts = vec![];
            for g in gnames {
                let (w, lo, ns) = cnt("F4-vs-react", metric, l, None, &wl(g));
                parts.push(format!("{g} {w}W/{lo}L/{ns}ns"));
                out.insert(format!("react|{metric}|{l}|{g}"), json!([w, lo, ns]));
            }
            let _ = writeln!(log, "  {metric:<8} {l:<6}: {}", parts.join(" ; "));
        }
    }
    let _ = writeln!(log, "=== ablations: Stack-minus-lever mean paired delta of (lever removed - Stack), mean of per-world means  [ok | q | cost | okpc | util.03], group genA | genB | controls:");
    for a in ABL {
        let mut parts = vec![];
        for g in gnames {
            let f = |mt: &str| mean(&wl(g).iter().filter_map(|wn| find(tests, "F5-ablation", wn, mt, a, Some("Stack"))).map(|t| t.d.diff.mean).collect::<Vec<_>>());
            parts.push(format!("{g}: ok {:+.3} q {:+.3} cost {:+.2} okpc {:+.4} u {:+.4}", f("ok_h1000"), f("q_h1000"), f("cost"), f("okpc"), f("util_c")));
            out.insert(format!("abl|{a}|{g}"), json!([f("ok_h1000"), f("q_h1000"), f("cost"), f("okpc"), f("util_c")]));
        }
        let _ = writeln!(log, "  {a:<12} {}", parts.join(" || "));
    }
    let _ = writeln!(log, "=== ABSOLUTE group means (mean of per-world means) [q | ok | cost | util(tune mu) | util(.1)]:");
    for arm in ["ss-react", "L2t", "L3x", "L4", "L4w", "L5", "Stack", "BoK"] {
        let mut parts = vec![];
        for g in gnames {
            let f = |mt: &str| mean(&results.iter().filter(|r| group_of(&r.name) == g).map(|r| { let ai = r.arms.iter().position(|x| x == arm).unwrap(); mean(&col(&r.cells[ai], mi(mt))) }).collect::<Vec<_>>());
            parts.push(format!("{g}: q {:.3} ok {:.3} cost {:.2} u {:.3} u.1 {:.3}", f("q_h1000"), f("ok_h1000"), f("cost"), f("util_c"), f("util_a")));
        }
        let _ = writeln!(log, "  {arm:<9} {}", parts.join(" || "));
    }
    let _ = writeln!(log, "=== mean paired deltas by group (mean of per-world means) [q | ok | cost | okpc]:");
    for (a, b, fam) in [("L1", "bf", "F1-vs-bestfair"), ("L2", "L1", "F2-vs-prev"), ("L2t", "L2", "F2-vs-prev"), ("L3x", "L2t", "F2-vs-prev"), ("L4", "L3x", "F2-vs-prev"), ("L5", "L4", "F2-vs-prev"), ("Stack", "L5", "F2-vs-prev"), ("L2t", "bf", "F1-vs-bestfair"), ("L3x", "bf", "F1-vs-bestfair"), ("L5", "bf", "F1-vs-bestfair"), ("Stack", "bf", "F1-vs-bestfair"), ("Stack", "bs", "F1s-vs-beststrong")] {
        let mut parts = vec![];
        for g in gnames {
            let vs = if fam == "F2-vs-prev" { Some(b) } else { None };
            let f = |mt: &str| mean(&wl(g).iter().filter_map(|wn| find(tests, fam, wn, mt, a, vs)).map(|t| t.d.diff.mean).collect::<Vec<_>>());
            let cost = if fam == "F2-vs-prev" { f("cost") } else { f64::NAN };
            parts.push(format!("{g} q {:+.4} ok {:+.4} cost {:+.2} okpc {:+.4}", f("q_h1000"), f("ok_h1000"), cost, f("okpc")));
            out.insert(format!("mean|{a}-{b}|{g}"), json!([f("q_h1000"), f("ok_h1000"), cost, f("okpc")]));
        }
        let _ = writeln!(log, "  {a} - {b}: {}", parts.join(" ; "));
    }
    Value::Object(out)
}

// ------------------------------------------------------------------------------------------------ controls: pre-declared equivalence / harm checks
fn controls_report(results: &[WorldRes], tests: &[Test], log: &mut String) -> Value {
    use std::fmt::Write;
    let _ = writeln!(log, "\n=== CONTROLS (pre-declared margins: +-0.01 graded q, +-0.02 all-correct; EQ = whole raw 95% CI inside the margin). Stack/L5/L4/L3x are compared with the lever's natural comparator:");
    let mut js = vec![];
    for wr in results.iter().filter(|r| is_control(&r.name)) {
        let g = |arm: &str, m: &str| -> f64 { let ai = wr.arms.iter().position(|x| x == arm).unwrap(); mean(&col(&wr.cells[ai], mi(m))) };
        let _ = writeln!(log, "  [{}] L2t q {:.3} ok {:.3} cost {:.2} | L3x q {:.3} ok {:.3} cost {:.2} vcalls {:.2} | L4 q {:.3} ok {:.3} cost {:.2} vcalls {:.2} | L5 q {:.3} ok {:.3} cost {:.2} | Stack q {:.3} ok {:.3} cost {:.2} | ss-react ok {:.3} cost {:.2} | bf({}) ok {:.3} cost {:.2}",
            wr.name, g("L2t", "q_h1000"), g("L2t", "ok_h1000"), g("L2t", "cost"), g("L3x", "q_h1000"), g("L3x", "ok_h1000"), g("L3x", "cost"), g("L3x", "vcalls"), g("L4", "q_h1000"), g("L4", "ok_h1000"), g("L4", "cost"), g("L4", "vcalls"),
            g("L5", "q_h1000"), g("L5", "ok_h1000"), g("L5", "cost"), g("Stack", "q_h1000"), g("Stack", "ok_h1000"), g("Stack", "cost"), g("ss-react", "ok_h1000"), g("ss-react", "cost"), wr.arms[wr.bf], g(&wr.arms[wr.bf], "ok_h1000"), g(&wr.arms[wr.bf], "cost"));
        // EQ tests: L4 vs L3x (gate must not lose quality), Stack vs ss-react (decomposition stack harmless where nothing to gain), L1 vs best plain
        let eq = |a: &str, b: &str, m: &str, margin: f64| -> (f64, f64, f64, bool) {
            let ai = wr.arms.iter().position(|x| x == a).unwrap();
            let bi = wr.arms.iter().position(|x| x == b).unwrap();
            let d = paired_diff(&col(&wr.cells[ai], mi(m)), &col(&wr.cells[bi], mi(m)));
            (d.diff.mean, d.diff.lo, d.diff.hi, d.diff.lo > -margin && d.diff.hi < margin)
        };
        let bfn = wr.arms[wr.bf].clone();
        let checks = [("L4", "L3x", "q_h1000", 0.01), ("L4", "L3x", "ok_h1000", 0.02), ("L1", bfn.as_str(), "q_h1000", 0.01), ("L2t", "ss-react", "q_h1000", 0.01), ("Stack", "ss-react", "q_h1000", 0.01), ("Stack", "L2t", "q_h1000", 0.01)];
        let mut row = vec![];
        for (a, b, m, mg) in checks {
            let (mn, lo, hi, e) = eq(a, b, m, mg);
            let _ = writeln!(log, "      {a} - {b} {m}: {mn:+.4} [{lo:+.4},{hi:+.4}] margin {mg} -> {}", if e { "EQ" } else if lo > 0.0 { "better" } else if hi < 0.0 { "WORSE" } else { "not-equivalent/ns" });
            row.push(json!({"a": a, "b": b, "metric": m, "mean": mn, "lo": lo, "hi": hi, "margin": mg, "equivalent": e}));
        }
        js.push(json!({"world": wr.name, "checks": row}));
    }
    let _ = tests;
    Value::Array(js)
}

// ------------------------------------------------------------------------------------------------ sensitivity
fn sens(t: &Tuned, seeds: &[u64], log: &mut String) -> Value {
    use std::fmt::Write;
    let base = worlds();
    let rm = base.iter().find(|w| w.name == "res-main").unwrap().clone();
    let am = base.iter().find(|w| w.name == "ana-main").unwrap().clone();
    let mut variants: Vec<(String, WorldCfg, u64)> = vec![];
    let mut push = |label: String, w: WorldCfg, idx: u64| variants.push((label, w, idx));
    for k in [0.0, 0.3, 0.7] {
        push(format!("res kmem={k}"), WorldCfg { kmem: k, ..rm.clone() }, 0);
    }
    for h in [0.45, 0.70, 0.90] {
        push(format!("res hq={h}"), WorldCfg { hq: h, ..rm.clone() }, 0);
        push(format!("ana hq={h}"), WorldCfg { hq: h, ..am.clone() }, 6);
    }
    for v in [0.0, 0.45, 1.0] {
        push(format!("res vq={v}"), WorldCfg { vq: v, ..rm.clone() }, 0);
        push(format!("ana vq={v}"), WorldCfg { vq: v, ..am.clone() }, 6);
    }
    for v in [0.25, 1.0, 2.5] {
        push(format!("res vscale={v}"), WorldCfg { vscale: v, ..rm.clone() }, 0);
        push(format!("ana vscale={v}"), WorldCfg { vscale: v, ..am.clone() }, 6);
    }
    for v in [0.5, 1.0, 2.0] {
        push(format!("res ex_scale={v}"), WorldCfg { ex_scale: v, ..rm.clone() }, 0);
        push(format!("res ar_scale={v}"), WorldCfg { ar_scale: v, ..rm.clone() }, 0);
    }
    for v in [1.3, 2.5, 5.0] {
        push(format!("res beta={v}"), WorldCfg { beta: v, ..rm.clone() }, 0);
        push(format!("ana beta={v}"), WorldCfg { beta: v, ..am.clone() }, 6);
    }
    for v in [0.0, 0.05] {
        push(format!("res mono_pen={v}"), WorldCfg { mono_pen: v, ..rm.clone() }, 0);
    }
    for v in [0.2, 0.5, 0.9] {
        push(format!("res p_vis={v}"), WorldCfg { p_vis: v, ..rm.clone() }, 0);
    }
    for v in [0.0, 0.3, 0.6] {
        push(format!("res q_sec={v}"), WorldCfg { q_sec: v, ..rm.clone() }, 0);
    }
    for v in [0.05, 0.10, 0.16] {
        push(format!("res noise={v}"), WorldCfg { noise: v, ..rm.clone() }, 0);
    }
    let names: Vec<String> = arms_main(t, 0).iter().map(|a| a.name.clone()).collect();
    let ai = |n: &str| names.iter().position(|x| x == n).unwrap();
    let _ = writeln!(log, "\n=== SENSITIVITY (one-at-a-time from res-main / ana-main, held seeds, frozen tuning). paired mean deltas [95% CI]; ok = all-steps-correct");
    let _ = writeln!(log, "{:<20} {:>6} {:>6} | {:>20} | {:>20} | {:>20} | {:>20} | {:>20} | {:>20} | {:>7} {:>7} {:>7}", "variant", "bf ok", "bf q", "L2t-L2 ok", "L3x-L2t ok", "L5-L4 ok", "Stack-bf ok", "Stack-react ok", "Stack-L5 ok", "c(L2t)", "c(L3x)", "c(Stack)");
    let mut js = vec![];
    for (label, w, idx) in &variants {
        let arms = arms_main(t, *idx as usize);
        let cells = run_world(w, *idx, &arms, seeds);
        let bf = best_of(&names, &cells, &BASELINES, mi("q_h1000"));
        let d = |a: &str, b: usize, m: &str| paired_diff(&col(&cells[ai(a)], mi(m)), &col(&cells[b], mi(m)));
        let f = |p: &PairedDiff| format!("{:+.3} [{:+.3},{:+.3}]", p.diff.mean, p.diff.lo, p.diff.hi);
        let cs = |a: &str| mean(&col(&cells[ai(a)], mi("cost")));
        let (x1, x2, x3, x4, x5, x6) = (d("L2t", ai("L2"), "ok_h1000"), d("L3x", ai("L2t"), "ok_h1000"), d("L5", ai("L4"), "ok_h1000"), d("Stack", bf, "ok_h1000"), d("Stack", ai("ss-react"), "ok_h1000"), d("Stack", ai("L5"), "ok_h1000"));
        let _ = writeln!(log, "{:<20} {:>6.3} {:>6.3} | {:>20} | {:>20} | {:>20} | {:>20} | {:>20} | {:>20} | {:>7.2} {:>7.2} {:>7.2}", label, mean(&col(&cells[bf], mi("ok_h1000"))), mean(&col(&cells[bf], mi("q_h1000"))), f(&x1), f(&x2), f(&x3), f(&x4), f(&x5), f(&x6), cs("L2t"), cs("L3x"), cs("Stack"));
        let tri = |p: &PairedDiff| json!([p.diff.mean, p.diff.lo, p.diff.hi]);
        let qd = |a: &str, b: usize| tri(&d(a, b, "q_h1000"));
        js.push(json!({"variant": label, "bf": names[bf], "bf_ok": mean(&col(&cells[bf], mi("ok_h1000"))), "L2t_minus_L2_ok": tri(&x1), "L3x_minus_L2t_ok": tri(&x2), "L5_minus_L4_ok": tri(&x3),
            "Stack_minus_bf_ok": tri(&x4), "Stack_minus_react_ok": tri(&x5), "Stack_minus_L5_ok": tri(&x6), "Stack_minus_bf_q": qd("Stack", bf),
            "cost": {"L2t": cs("L2t"), "L3x": cs("L3x"), "L4": cs("L4"), "Stack": cs("Stack")}, "util_c": {"L2t": mean(&col(&cells[ai("L2t")], mi("util_c"))), "L3x": mean(&col(&cells[ai("L3x")], mi("util_c"))), "L4": mean(&col(&cells[ai("L4")], mi("util_c"))), "Stack": mean(&col(&cells[ai("Stack")], mi("util_c"))), "bf": mean(&col(&cells[bf], mi("util_c")))}}));
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
    let dummy = Tuned { eps: (1.0, 0.0), pts: 1.0, warm_k: 1.0, orch_k: 1.0, rv: 1.0, r_on: 1.0, eps_on: 0.05, theta: 0.6, rv_world: vec![] };
    if args.iter().any(|a| a == "--show") {
        // world examples + non-policy reference rows (seed 1)
        let mut s = String::new();
        for (wi, w) in ws.iter().enumerate() {
            s.push_str(&describe(w, wi as u64, 1));
            let inst = build(w, 1, wi as u64);
            let o1 = summarize_run(&run_single(w, &inst, &SsKind::OracleSingle, false, 1, 1));
            let o2 = summarize_run(&run_orch(w, &inst, &Cfg { route: Route::Oracle, ..Cfg::l1() }, 1));
            let o3 = summarize_run(&run_orch(w, &inst, &Cfg { route: Route::Oracle, ..Cfg::level("L5", 1.0, 1.0, 1.0, 0.05, 0.0) }, 1));
            let o4 = summarize_run(&run_orch(w, &inst, &Cfg { route: Route::Oracle, ..Cfg::level("Stack", 1.0, 1.0, 1.0, 0.05, 0.0) }, 1));
            s.push_str(&format!(
                "  reference rows (not policies): oracle-single q {:.3} ok {:.3} cost {:.2} | oracle-decomp (no retr) q {:.3} ok {:.3} | oracle-L5 q {:.3} ok {:.3} cost {:.2} | oracle-Stack q {:.3} ok {:.3} cost {:.2}\n",
                o1[mi("q_h1000")], o1[mi("ok_h1000")], o1[mi("cost")], o2[mi("q_h1000")], o2[mi("ok_h1000")], o3[mi("q_h1000")], o3[mi("ok_h1000")], o3[mi("cost")], o4[mi("q_h1000")], o4[mi("ok_h1000")], o4[mi("cost")]
            ));
        }
        print!("{s}");
        return;
    }
    if let Some(i) = args.iter().position(|a| a == "--mu") {
        let m: f64 = args[i + 1].parse().unwrap();
        MU_BITS.store(m.to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
    let held = range(&arg(&args, "--held", "31-60"));
    let tune_seeds = range(&arg(&args, "--tune", "1-8"));
    let out_path = arg(&args, "--out", "results/h21_research.json");
    let t0 = Instant::now();
    let mut log = String::new();
    let (tuned, tune_json) = tune(&ws, &tune_seeds, &dummy);
    log.push_str(&format!("tuned on seeds {:?}-{:?}: {:?}  ({:.0}s)\n", tune_seeds[0], tune_seeds.last().unwrap(), tuned, t0.elapsed().as_secs_f64()));
    if args.iter().any(|a| a == "--sens") {
        let sj = sens(&tuned, &held, &mut log);
        std::fs::write(&out_path, serde_json::to_string_pretty(&json!({"tuned": tune_json, "held": format!("{:?}", held), "sens": sj})).unwrap()).unwrap();
        print!("{log}");
        return;
    }
    let (ws, tuned) = if args.iter().any(|a| a == "--extra") {
        // POST-HOC extra controls: global picks from the main tuning, per-world gate scale tuned on the same tune seeds for the extra worlds only
        let xs = worlds_extra();
        let mut t2 = tuned.clone();
        t2.rv_world = rv_world_for(&xs, 100, &tune_seeds, tuned.orch_k);
        log.push_str(&format!("EXTRA worlds {:?}: per-world rv {:?}
", xs.iter().map(|w| w.name).collect::<Vec<_>>(), t2.rv_world));
        (xs, t2)
    } else {
        (ws, tuned)
    };
    let (results, tests) = study(&ws, &tuned, &held, &mut log);
    report(&results, &tests, &mut log);
    let counts = summary_counts(&results, &tests, &mut log);
    let ctl = controls_report(&results, &tests, &mut log);
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
        worlds_json.insert(wr.name.clone(), json!({"best_fair": wr.arms[wr.bf], "best_strong": wr.arms[wr.bs], "arms": arms}));
    }
    let doc = json!({
        "protocol": {"tune_seeds": format!("{:?}", tune_seeds), "held_seeds": format!("{:?}", held), "n_tasks": N_TASKS, "reveal": REVEAL, "delay": DELAY, "max_retry": MAX_RETRY, "s_ref_te": S_REF},
        "tuning": tune_json, "worlds": worlds_json, "counts": counts, "controls": ctl, "tests": tests.iter().map(test_json).collect::<Vec<_>>(),
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
        let widx = worlds().iter().position(|x| x.name == name).unwrap() as u64;
        let inst = build(&w, 3, widx);
        (w, inst)
    }

    #[test]
    fn real_calculator_equals_native_arithmetic_and_formulation_errors_differ() {
        let mut r = neural_mesh::router::Rng::new(5);
        for _ in 0..3000 {
            let n = 2 + r.below(2);
            let v: Vec<i64> = (0..n).map(|_| 100 + r.below(9900) as i64 - if r.f64() < 0.2 { 6000 } else { 0 }).collect();
            for op in [Op::Add, Op::Sub, Op::AddSub, Op::Scale] {
                let need = match op { Op::Add => 2, Op::Sub => 2, Op::AddSub => 3, Op::Scale => 2 };
                if v.len() < need {
                    continue;
                }
                let exact = op_native(op, &v);
                assert_eq!(calc_eval(&op_spec(op, &v), i64::MIN), exact, "calc vs native for {}", op_spec(op, &v));
                let bad = calc_eval(&op_spec_mut(op, &v), i64::MIN);
                assert!(bad != i64::MIN, "mutated spec must still evaluate");
                assert!(bad != exact || v.iter().any(|&x| x == 0), "formulation error must change the value: {}", op_spec_mut(op, &v));
            }
        }
    }

    #[test]
    fn worlds_are_well_formed() {
        for w in worlds() {
            let widx = worlds().iter().position(|x| x.name == w.name).unwrap() as u64;
            let inst = build(&w, 2, widx);
            for tk in &inst.tasks {
                let n = tk.steps.len();
                if w.shape == Shape::Single {
                    assert_eq!(n, 1);
                } else if w.shape == Shape::Indep {
                    assert!((2..=8).contains(&n), "{} indep steps {n}", w.name);
                } else {
                    assert!((3..=8).contains(&n), "{} steps {n}", w.name);
                    assert_eq!(tk.steps[n - 1].ty, SYN);
                }
                for (i, s) in tk.steps.iter().enumerate() {
                    assert!(s.deps.iter().all(|&d| d < i));
                    if s.ty == READ {
                        assert!(s.deps.len() == 1 && tk.steps[s.deps[0]].ty == SEARCH);
                    }
                    if s.ty == COMP {
                        assert!(s.deps.len() >= 2 && s.deps.iter().all(|&d| tk.steps[d].ty != SEARCH));
                    }
                    if s.ty == SEARCH {
                        assert!(s.deps.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn deterministic_and_oracle_l1_cost_is_exactly_the_h11_base_cost() {
        let (w, inst) = tiny("res-main");
        let cfg = Cfg { route: Route::Oracle, ..Cfg::l1() };
        let a = run_orch(&w, &inst, &cfg, 3);
        let b = run_orch(&w, &inst, &cfg, 3);
        assert_eq!(a.q, b.q);
        assert_eq!(a.cost, b.cost);
        for (ti, tk) in inst.tasks.iter().enumerate() {
            let exp: f64 = tk.steps.iter().map(|s| {
                let best = (0..inst.n_agents).fold(0, |b, a| if inst.comp[a][s.ty] > inst.comp[b][s.ty] { a } else { b });
                base_te(&inst, best, s.deps.len())
            }).sum::<f64>() / S_REF;
            assert!((a.cost[ti] as f64 - exp).abs() < 1e-3, "task {ti}: {} vs {exp}", a.cost[ti]);
            assert_eq!(a.calls[ti] as usize, tk.steps.len());
            assert_eq!(a.vcalls[ti], 0);
        }
    }

    #[test]
    fn verifier_only_sees_local_errors_and_vq0_is_uninformative() {
        let (w, inst) = tiny("res-main");
        let tk_i = inst.tasks.iter().position(|t| t.steps.iter().any(|s| s.ty == COMP)).unwrap();
        let si = inst.tasks[tk_i].steps.iter().position(|s| s.ty == COMP).unwrap();
        let obs: Vec<i64> = inst.tasks[tk_i].truth.clone();
        let good = Exec { val: 1, local_ok: true, visible_fail: false, cost: 0.0, tool: false };
        let bad = Exec { local_ok: false, ..good };
        let (mut fg, mut fb, mut fg0, mut fb0) = (0, 0, 0, 0);
        let w0 = WorldCfg { vq: 0.0, ..w.clone() };
        for s in 0..4000u64 {
            fg += verify(&w, &inst, s, tk_i, si, 1, &good, &obs, false).0 as usize;
            fb += verify(&w, &inst, s, tk_i, si, 1, &bad, &obs, false).0 as usize;
            fg0 += verify(&w0, &inst, s, tk_i, si, 1, &good, &obs, false).0 as usize;
            fb0 += verify(&w0, &inst, s, tk_i, si, 1, &bad, &obs, false).0 as usize;
        }
        assert!(fb > 3800 && fg < 100, "compute verifier: recall .97 / fp .01, got {fb} {fg}");
        assert!((fb0 as i64 - fg0 as i64).abs() < 60, "vq=0 must be uninformative: {fb0} vs {fg0}");
    }

    #[test]
    fn reformulation_is_monotone_and_budget_blocks_extras() {
        let (w, inst) = tiny("res-main");
        let (mut up, mut down) = (0, 0);
        for (ti, tk) in inst.tasks.iter().enumerate().take(300) {
            for (si, s) in tk.steps.iter().enumerate() {
                if s.ty != SEARCH {
                    continue;
                }
                let obs = vec![NONE; tk.steps.len()];
                let f0 = Fl { retr: true, tools: false, mono: false, reform: 0 };
                let f2 = Fl { reform: 2, ..f0 };
                let h0 = exec_step(&w, &inst, 3, ti, si, 2, &obs, &f0).local_ok;
                let h2 = exec_step(&w, &inst, 3, ti, si, 2, &obs, &f2).local_ok;
                if h0 && !h2 { down += 1 }
                if !h0 && h2 { up += 1 }
            }
        }
        assert_eq!(down, 0);
        assert!(up > 10);
        // a tiny budget blocks every extra: no verifier calls, exactly one call per step
        let wb = WorldCfg { beta: 0.5, ..w.clone() };
        let cfg = Cfg::level("L5", 1.0, 1.0, 1.0, 0.05, 0.0);
        let r = run_orch(&wb, &inst, &cfg, 3);
        for (ti, tk) in inst.tasks.iter().enumerate() {
            assert_eq!(r.vcalls[ti], 0, "task {ti}");
            assert_eq!(r.calls[ti] as usize, tk.steps.len());
        }
        assert!(r.blocked.iter().any(|&b| b == 1));
    }

    #[test]
    fn posterior_gate_with_huge_rv_equals_always_verify_and_gate_matches_verifygate() {
        let (w, inst) = tiny("res-main");
        let a = run_orch(&w, &inst, &Cfg::level("L4", 1.0, 1.0e9, 1.0, 0.05, 0.0), 3);
        let b = run_orch(&w, &inst, &Cfg::level("L3x", 1.0, 1.0, 1.0, 0.05, 0.0), 3);
        assert_eq!(a.q, b.q);
        assert_eq!(a.cost, b.cost);
        let g = neural_mesh::verifygate::PosteriorGate { rv: 1.0, cv: 0.3, nmin: 3.0 };
        assert!(g.should_verify_sums(0.5 * 10.0, 10.0) && !g.should_verify_sums(0.9 * 10.0, 10.0) && g.should_verify_sums(0.99 * 2.0, 2.0));
    }

    #[test]
    fn single_step_world_has_one_step_and_react_beats_plain_with_retrieval() {
        let (_, inst) = tiny("ctl-single-step");
        assert!(inst.tasks.iter().all(|t| t.steps.len() == 1));
        let (w, inst) = tiny("res-main");
        let plain = summarize_run(&run_single(&w, &inst, &SsKind::Keyword, false, 1, 3));
        let react = summarize_run(&run_single(&w, &inst, &SsKind::Keyword, true, 1, 3));
        assert!(react[mi("q_h1000")] > plain[mi("q_h1000")] + 0.1);
    }

    #[test]
    fn no_info_world_cannot_be_rescued_by_reformulation() {
        let (w, inst) = tiny("ctl-noinfo");
        let mut hits = 0;
        let mut n = 0;
        for (ti, tk) in inst.tasks.iter().enumerate().take(400) {
            for (si, s) in tk.steps.iter().enumerate() {
                if s.ty != SEARCH {
                    continue;
                }
                let obs = vec![NONE; tk.steps.len()];
                n += 1;
                hits += exec_step(&w, &inst, 3, ti, si, 2, &obs, &Fl { retr: true, tools: true, mono: false, reform: 2 }).local_ok as usize;
            }
        }
        assert!((hits as f64 / n as f64) < 0.08, "coverage .03 caps the hit rate: {hits}/{n}");
    }
}
