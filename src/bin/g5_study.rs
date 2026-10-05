//! G5 (W5 DESIGN-BIZ): does domain-hierarchical routing + the verified retrieval/reuse pipeline raise
//! mean GRADED task quality in design / business / marketing (coding + math as controls) over plain
//! routing, keyword routing and the other fair baselines?
//!
//! Simulation only (src/bin/g5_common/mod.rs: my own seeded worlds). Measures routing tasks to the right
//! specialist and handing that specialist verified context (a recurring brief / style guide). It does NOT
//! measure model creativity; the context effect size is an ASSUMPTION built into the world.
//!
//! Protocol: all tunable knobs (baselines' eps/gamma/warm_k, hier shrinkage params, retrieval store_min /
//! sim_min) tuned on seeds 1-8 ONLY; held-out 31-60 (default) / --held 61-90 = confirmation with the SAME
//! frozen tuning (the tuning always re-runs on 1-8, deterministic). Paired 95% t-CI, Holm per family.
//!
//! Usage: cargo run --release --offline --bin g5_study -- [--held 31-60] [--out results/g5_study.json]
//!        ... -- --sens   [--held 31-60]   (one-at-a-time sensitivity of the context/retrieval assumptions)
//!        ... -- --check                   (replica driver vs the real ResearchSession)

use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g5_common/mod.rs"]
mod common;
use common::*;

use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize, PairedDiff};

const TUNE_SEEDS: (u64, u64) = (1, 8);

/// results[world][arm][seed]
type Mat = Vec<Vec<Vec<RunOut>>>;

fn run_matrix(ws: &[WorldCfg], arms: &[Arm], seeds: &[u64], retr: RetrParams) -> Mat {
    let insts: Vec<Vec<Instance>> = ws.iter().enumerate().map(|(wi, w)| seeds.par_iter().map(|&s| build(w, s, wi as u64)).collect()).collect();
    ws.iter()
        .enumerate()
        .map(|(wi, w)| arms.iter().map(|arm| seeds.par_iter().enumerate().map(|(si, &s)| run_arm(w, &insts[wi][si], arm, retr, s)).collect::<Vec<RunOut>>()).collect::<Vec<_>>())
        .collect()
}

fn objective(r: &[RunOut]) -> f64 {
    // mean over seeds of the mean of H250, H500, H1500
    let v: Vec<f64> = r
        .iter()
        .map(|o| {
            let h = horizon_means(&o.q);
            (h[1] + h[2] + h[3]) / 3.0
        })
        .collect();
    summarize(&v).mean
}

fn arm(name: &str, kind: Kind, retr: bool, session: bool) -> Arm {
    Arm { name: name.to_string(), kind, retr, session }
}

fn verdict(adj: f64, mean: f64) -> &'static str {
    if adj < 0.05 && mean > 0.0 {
        "WIN"
    } else if adj < 0.05 {
        "LOSS"
    } else {
        "ns"
    }
}

struct Test {
    family: &'static str,
    world: usize,
    scope: String,
    treat: String,
    vs: String,
    d: PairedDiff,
    p: f64,
}

fn mk_test(family: &'static str, world: usize, scope: &str, treat: &str, vs: &str, t: &[f64], b: &[f64], higher_better: bool) -> Test {
    let (x, y) = if higher_better { (t, b) } else { (b, t) };
    Test { family, world, scope: scope.to_string(), treat: treat.to_string(), vs: vs.to_string(), d: paired_diff(x, y), p: paired_t_pvalue(x, y) }
}

fn parse_range(args: &[String], key: &str, dflt: (u64, u64)) -> (u64, u64) {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).and_then(|r| r.split_once('-')).map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).unwrap_or(dflt)
}

fn check_mode() {
    let ws = worlds();
    println!("g5_study --check: replica driver vs the real ResearchSession (health off), per-task quality must be identical");
    let rp = RetrParams { store_min: 0.65, sim_min: 0.55 };
    for wi in [0usize, 1, 4, 7, 8, 10] {
        for kind in [Kind::Safe, Kind::Warm(3.0)] {
            let mut mx_off = 0.0f64;
            let mut mx_on = 0.0f64;
            for seed in 1..=3u64 {
                let inst = build(&ws[wi], seed, wi as u64);
                let a = run_session(&ws[wi], &inst, &kind, None, false, seed);
                let b = run_replica(&ws[wi], &inst, &kind, None, seed);
                let c = run_session(&ws[wi], &inst, &kind, Some(rp), false, seed);
                let d = run_replica(&ws[wi], &inst, &kind, Some(rp), seed);
                for i in 0..N_TASKS {
                    mx_off = mx_off.max((a.q[i] - b.q[i]).abs() as f64);
                    mx_on = mx_on.max((c.q[i] - d.q[i]).abs() as f64);
                }
                assert_eq!(c.ctx_hits, d.ctx_hits);
            }
            println!("  {:<22} {:<12} max|dq| retrieval-off {:.2e}   retrieval-on {:.2e}", ws[wi].name, format!("{:?}", kind), mx_off, mx_on);
            assert!(mx_off < 1e-9 && mx_on < 1e-9, "replica differs from the real session in world {}", ws[wi].name);
        }
    }
    // d = 0 worlds: context cannot matter, so retrieval must leave every delivered quality bit-identical
    for wi in IDX_STRESS + 1..ws.len() {
        for kind in [Kind::Safe, Kind::Warm(3.0), Kind::Hier(HierParams { m_g: 2.0, m_d: 4.0, m_s: 6.0, xi: 0.5, kw_w: 3.0, use_dom: true })] {
            for seed in 1..=3u64 {
                let inst = build(&ws[wi], seed, wi as u64);
                let a = run_replica(&ws[wi], &inst, &kind, None, seed);
                let b = run_replica(&ws[wi], &inst, &kind, Some(rp), seed);
                assert!(a.q == b.q, "retrieval changed quality in a d=0 world ({})", ws[wi].name);
            }
        }
        println!("  {:<22} d=0 control: retrieval on == off bit-identically (3 routers x 3 seeds)", ws[wi].name);
    }
    println!("check OK: replica == ResearchSession (reuse on, critic off, health off), safe and warm routers, with/without retrieval, briefs-ingest world included");
}

struct Tuned {
    eps: (f64, f64),
    pts: f64,
    warm: f64,
    hier: HierParams,
    retr: RetrParams,
}

fn tune(ws: &[WorldCfg], tune_seeds: &[u64]) -> Tuned {
    let t0 = Instant::now();
    let dflt_retr = RetrParams { store_min: 0.65, sim_min: 0.55 };
    let obj_over_worlds = |m: &Mat, ai: usize| -> f64 { m.iter().map(|w| objective(&w[ai])).sum::<f64>() / m.len() as f64 };
    let eps_grid: Vec<(f64, f64)> = vec![(1.0, 0.0), (1.0, 0.05), (0.97, 0.05), (0.9, 0.0)];
    let pts_grid: Vec<f64> = vec![1.0, 0.98];
    let warm_grid: Vec<f64> = vec![1.0, 3.0, 10.0];
    let mut garms: Vec<Arm> = vec![];
    for &(g, e) in &eps_grid {
        garms.push(arm(&format!("eps({g},{e})"), Kind::Eps(g, e), false, false));
    }
    for &g in &pts_grid {
        garms.push(arm(&format!("pts({g})"), Kind::Pts(g), false, false));
    }
    for &k in &warm_grid {
        garms.push(arm(&format!("warm(k={k})"), Kind::Warm(k), false, false));
    }
    let mut hier_grid: Vec<HierParams> = vec![];
    for &m_d in &[2.0, 4.0] {
        for &m_s in &[6.0, 12.0] {
            for &xi in &[0.2, 0.5] {
                for &kw_w in &[3.0, 8.0] {
                    hier_grid.push(HierParams { m_g: 2.0, m_d, m_s, xi, kw_w, use_dom: true });
                }
            }
        }
    }
    for p in &hier_grid {
        garms.push(arm(&format!("hier(md{},ms{},xi{},kw{})", p.m_d, p.m_s, p.xi, p.kw_w), Kind::Hier(*p), false, false));
    }
    let m = run_matrix(ws, &garms, tune_seeds, dflt_retr);
    let scores: Vec<f64> = (0..garms.len()).map(|i| obj_over_worlds(&m, i)).collect();
    let pick = |lo: usize, hi: usize| -> usize { (lo..hi).max_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap()).unwrap() };
    let (ne, np, nw) = (eps_grid.len(), pts_grid.len(), warm_grid.len());
    let ie = pick(0, ne);
    let ip = pick(ne, ne + np);
    let iw = pick(ne + np, ne + np + nw);
    let ih = pick(ne + np + nw, garms.len());
    for (i, a) in garms.iter().enumerate() {
        println!("  tune {:<34} obj {:.4}{}", a.name, scores[i], if [ie, ip, iw, ih].contains(&i) { "   <== picked" } else { "" });
    }
    let (eps, pts, warm, hier) = (eps_grid[ie], pts_grid[ip - ne], warm_grid[iw - ne - np], hier_grid[ih - ne - np - nw]);
    // retrieval knobs on the real ResearchSession (warm router + retrieval)
    let sess = arm("session-warm", Kind::Warm(warm), true, true);
    let mut best_r = (f64::NEG_INFINITY, dflt_retr);
    for &sm in &[0.55, 0.65, 0.75] {
        for &sim in &[0.55, 0.65, 0.75, 0.85] {
            let rp = RetrParams { store_min: sm, sim_min: sim };
            let mm = run_matrix(ws, std::slice::from_ref(&sess), tune_seeds, rp);
            let sc = obj_over_worlds(&mm, 0);
            println!("  tune retrieval store_min {sm} sim_min {sim}: obj {sc:.4}");
            if sc > best_r.0 {
                best_r = (sc, rp);
            }
        }
    }
    println!("  PICKED: eps {eps:?} pts_gamma {pts} warm_k {warm} hier {hier:?} retrieval {:?}   (tuning {:.0}s)", best_r.1, t0.elapsed().as_secs_f64());
    Tuned { eps, pts, warm, hier, retr: best_r.1 }
}

fn tuned_json(t: &Tuned) -> Value {
    json!({"eps": t.eps, "pts_gamma": t.pts, "warm_k": t.warm, "hier": {"m_g": t.hier.m_g, "m_d": t.hier.m_d, "m_s": t.hier.m_s, "xi": t.hier.xi, "kw_w": t.hier.kw_w}, "store_min": t.retr.store_min, "sim_min": t.retr.sim_min})
}

fn arms_for(t: &Tuned) -> Vec<Arm> {
    vec![
        // fair baselines (0..=8): plain routing, same delayed graded feedback, no retrieval
        arm("keyword", Kind::Keyword, false, false),
        arm("eps-greedy(tuned)", Kind::Eps(t.eps.0, t.eps.1), false, false),
        arm("pooled-ts(tuned)", Kind::Pts(t.pts), false, false),
        arm("hebbian-robust", Kind::HebR, false, false),
        arm("warm(hebr,tuned)", Kind::Warm(t.warm), false, false),
        arm("eps-greedy(.9,0)[e1]", Kind::Eps(0.9, 0.0), false, false),
        arm("pooled-ts(1)[e1]", Kind::Pts(1.0), false, false),
        arm("warm(hebr,k=1)[e1]", Kind::Warm(1.0), false, false),
        arm("guarded-safe(plain)", Kind::Safe, false, false),
        // treatments (9..=12)
        arm("hier", Kind::Hier(t.hier), false, false),
        arm("session(warm+retr)", Kind::Warm(t.warm), true, true),
        arm("session(safe+retr)", Kind::Safe, true, true),
        arm("hier+retr", Kind::Hier(t.hier), true, false),
        // reference
        arm("oracle(no-ctx)", Kind::Oracle, false, false),
        arm("oracle(+perfect ctx)", Kind::OracleCtx, false, false),
        // ablation of the hierarchy itself
        arm("hier(no domain level)", Kind::Hier(HierParams { use_dom: false, ..t.hier }), false, false),
    ]
}
const N_FAIR: usize = 9;
const I_KW: usize = 0;
const I_WARM: usize = 4;
const I_PLAIN: usize = 8;
const I_HIER: usize = 9;
const I_SW: usize = 10;
const I_SS: usize = 11;
const I_HR: usize = 12;
const I_HND: usize = 15;
const TREAT: [usize; 4] = [I_HIER, I_SW, I_SS, I_HR];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--check") {
        check_mode();
        return;
    }
    if args.iter().any(|a| a == "--show") {
        let ws = worlds();
        for (wi, w) in ws.iter().enumerate() {
            if ["agency", "generalist-pool", "big-pool", "latent-factors", "confusable-briefs", "ctl-coding-math"].contains(&w.name) {
                println!("{}", describe(w, wi as u64, 31));
            }
        }
        return;
    }
    let sens = args.iter().any(|a| a == "--sens");
    let dflt_out = if sens { "results/g5_sens.json" } else { "results/g5_study.json" };
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or(dflt_out.into());
    let (h0, h1) = parse_range(&args, "--held", (31, 60));
    let held: Vec<u64> = (h0..=h1).collect();
    let tune_seeds: Vec<u64> = (TUNE_SEEDS.0..=TUNE_SEEDS.1).collect();
    let ws = worlds();
    let t0 = Instant::now();
    println!("g5_study: {} worlds, tune seeds {}-{}, held-out {h0}-{h1}", ws.len(), TUNE_SEEDS.0, TUNE_SEEDS.1);
    let tuned = tune(&ws, &tune_seeds);
    if sens {
        sens_mode(&tuned, &held, (h0, h1), &out, t0);
        return;
    }
    let arms = arms_for(&tuned);
    let mat = run_matrix(&ws, &arms, &held, tuned.retr);
    println!("evaluation done at {:.0}s", t0.elapsed().as_secs_f64());

    let ns = held.len();
    let hseries = |wi: usize, ai: usize, h: usize| -> Vec<f64> { (0..ns).map(|s| horizon_means(&mat[wi][ai][s].q)[h]).collect() };
    let insts: Vec<Vec<Instance>> = ws.iter().enumerate().map(|(wi, w)| held.iter().map(|&s| build(w, s, wi as u64)).collect()).collect();
    let dseries = |wi: usize, ai: usize, d: usize| -> Vec<f64> { (0..ns).map(|s| domain_means(&mat[wi][ai][s].q, &insts[wi][s])[d]).collect() };
    let tseries = |wi: usize, ai: usize, frac: f64| -> Vec<f64> { (0..ns).map(|s| tasks_to_target(&mat[wi][ai][s].q, oracle_mean(&insts[wi][s]), frac)).collect() };
    let mean_of = |v: &[f64]| summarize(v).mean;
    let best_fair = |f: &dyn Fn(usize) -> Vec<f64>, higher: bool| -> usize {
        (0..N_FAIR)
            .max_by(|&a, &b| {
                let (ma, mb) = (mean_of(&f(a)), mean_of(&f(b)));
                if higher { ma.partial_cmp(&mb).unwrap() } else { mb.partial_cmp(&ma).unwrap() }
            })
            .unwrap()
    };
    // sample efficiency vs the best fair baseline's OWN late quality: target = 95% of its last-300-task mean
    // (per seed); tasks (1-based) until the trailing-100 mean reaches it; 1501 = never within the run.
    let bfw: Vec<usize> = (0..ws.len()).map(|wi| best_fair(&|a| hseries(wi, a, 3), true)).collect();
    let tbf = |wi: usize, ai: usize| -> Vec<f64> { (0..ns).map(|s| tasks_to_abs(&mat[wi][ai][s].q, 0.95 * tail_mean(&mat[wi][bfw[wi]][s].q, 300))).collect() };

    // ---- per-world tables
    let mut world_json: Vec<Value> = vec![];
    for (wi, w) in ws.iter().enumerate() {
        println!("\n=== {} | pool {:?} ({} agents) | held {h0}-{h1} | mean DELIVERED true quality over first H ===", w.name, w.pool, w.pool.n_agents());
        println!("{:<24}{:>8}{:>8}{:>8}{:>8}   {:>6}{:>6}{:>6}   {:>6}{:>6}{:>6}{:>6}{:>6}  ctxhit pur  reach(t90,t95)", "arm", 100, 250, 500, 1500, "t90", "t95", "tbf", "dsn", "biz", "mkt", "cod", "mth");
        let mut arms_json = vec![];
        for (ai, a) in arms.iter().enumerate() {
            let hm: Vec<f64> = (0..4).map(|h| mean_of(&hseries(wi, ai, h))).collect();
            let t90 = tseries(wi, ai, 0.90);
            let t95 = tseries(wi, ai, 0.95);
            let reach90 = t90.iter().filter(|&&x| x <= N_TASKS as f64).count() as f64 / ns as f64;
            let reach95 = t95.iter().filter(|&&x| x <= N_TASKS as f64).count() as f64 / ns as f64;
            let dm: Vec<f64> = (0..NDOM).map(|d| mean_of(&dseries(wi, ai, d))).collect();
            let ch = mat[wi][ai].iter().map(|o| o.ctx_hits as f64).sum::<f64>() / ns as f64 / N_TASKS as f64;
            let pu = mat[wi][ai].iter().map(|o| o.purity).sum::<f64>() / ns as f64;
            println!(
                "{:<24}{}   {:>6.0}{:>6.0}{:>6.0}   {}  {:.2}  {:.2}  ({:.2},{:.2})",
                a.name,
                hm.iter().map(|x| format!("{:>8.4}", x)).collect::<String>(),
                mean_of(&t90),
                mean_of(&t95),
                mean_of(&tbf(wi, ai)),
                dm.iter().map(|x| if x.is_nan() { format!("{:>6}", "-") } else { format!("{:>6.3}", x) }).collect::<String>(),
                ch,
                pu,
                reach90,
                reach95
            );
            arms_json.push(json!({"arm": a.name, "H": hm, "t90_mean": mean_of(&t90), "t95_mean": mean_of(&t95), "tbf_mean": mean_of(&tbf(wi, ai)), "reach90": reach90, "reach95": reach95, "domain_means": dm.iter().map(|x| if x.is_nan() { Value::Null } else { json!(x) }).collect::<Vec<_>>(), "ctx_hit_rate": ch, "mean_purity": pu}));
        }
        world_json.push(json!({"world": w.name, "pool": format!("{:?}", w.pool), "arms": arms_json}));
    }

    // ---- tests
    let mut tests: Vec<Test> = vec![];
    for wi in 0..ws.len() {
        for h in 0..4 {
            let bf = best_fair(&|a| hseries(wi, a, h), true);
            for &t in &TREAT {
                let tv = hseries(wi, t, h);
                let scope = format!("H{}", HORIZONS[h]);
                tests.push(mk_test("F1", wi, &scope, &arms[t].name, &format!("best-fair={}", arms[bf].name), &tv, &hseries(wi, bf, h), true));
                tests.push(mk_test("F1", wi, &scope, &arms[t].name, "guarded-safe(plain)", &tv, &hseries(wi, I_PLAIN, h), true));
                tests.push(mk_test("F1", wi, &scope, &arms[t].name, "keyword", &tv, &hseries(wi, I_KW, h), true));
            }
        }
    }
    let f1_n = tests.len();
    for wi in 0..ws.len() {
        for h in 0..4 {
            let scope = format!("H{}", HORIZONS[h]);
            let g = |a: usize| hseries(wi, a, h);
            tests.push(mk_test("F2", wi, &scope, "retrieval on warm (session(warm+retr))", "warm(hebr,tuned)", &g(I_SW), &g(I_WARM), true));
            tests.push(mk_test("F2", wi, &scope, "retrieval on guarded-safe (session(safe+retr))", "guarded-safe(plain)", &g(I_SS), &g(I_PLAIN), true));
            tests.push(mk_test("F2", wi, &scope, "retrieval on hier (hier+retr)", "hier", &g(I_HR), &g(I_HIER), true));
            tests.push(mk_test("F2", wi, &scope, "hier (no retr)", "warm(hebr,tuned)", &g(I_HIER), &g(I_WARM), true));
            tests.push(mk_test("F2", wi, &scope, "hier (no retr)", "guarded-safe(plain)", &g(I_HIER), &g(I_PLAIN), true));
            tests.push(mk_test("F2", wi, &scope, "hier+retr", "session(warm+retr)", &g(I_HR), &g(I_SW), true));
            tests.push(mk_test("F2", wi, &scope, "hier (domain level on)", "hier(no domain level)", &g(I_HIER), &g(I_HND), true));
        }
    }
    let f2_end = tests.len();
    for wi in 0..ws.len() {
        for d in 0..NDOM {
            if ws[wi].mix[d] <= 0.0 {
                continue;
            }
            let bf = best_fair(&|a| dseries(wi, a, d), true);
            for &t in &TREAT {
                tests.push(mk_test("F3", wi, DOMS[d], &arms[t].name, &format!("best-fair={}", arms[bf].name), &dseries(wi, t, d), &dseries(wi, bf, d), true));
            }
        }
    }
    let f3_end = tests.len();
    for wi in 0..ws.len() {
        for (lab, frac) in [("t90", 0.90), ("t95", 0.95)] {
            let bf = best_fair(&|a| tseries(wi, a, frac), false);
            for &t in &TREAT {
                tests.push(mk_test("F4", wi, lab, &arms[t].name, &format!("best-fair={}", arms[bf].name), &tseries(wi, t, frac), &tseries(wi, bf, frac), false));
            }
        }
        // target = 95% of the best fair baseline's own late quality (reachable by it by construction)
        for &t in &TREAT {
            tests.push(mk_test("F4", wi, "t@95%bf", &arms[t].name, &format!("best-fair(H1500)={}", arms[bfw[wi]].name), &tbf(wi, t), &tbf(wi, bfw[wi]), false));
        }
    }
    let fam_ranges = [("F1", 0, f1_n), ("F2", f1_n, f2_end), ("F3", f2_end, f3_end), ("F4", f3_end, tests.len())];
    let mut adj = vec![1.0; tests.len()];
    for &(_, lo, hi) in &fam_ranges {
        let a = holm_adjust(&tests[lo..hi].iter().map(|t| t.p).collect::<Vec<_>>());
        adj[lo..hi].copy_from_slice(&a);
    }
    let show = |fam: &str, filter: &dyn Fn(&Test) -> bool| {
        let (lo, hi) = fam_ranges.iter().find(|f| f.0 == fam).map(|f| (f.1, f.2)).unwrap();
        println!("\n=== {fam}: Holm over {} tests (printed: Holm WIN/LOSS matching the filter) ===", hi - lo);
        let (mut nw, mut nl) = (0, 0);
        for i in lo..hi {
            let t = &tests[i];
            let v = verdict(adj[i], t.d.diff.mean);
            if v == "WIN" {
                nw += 1;
            }
            if v == "LOSS" {
                nl += 1;
            }
            if filter(t) && v != "ns" {
                println!("{:<20} {:<8} {:<26} vs {:<36} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.4} {}", ws[t.world].name, t.scope, t.treat, t.vs, t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, t.d.wins, t.d.losses, adj[i], v);
            }
        }
        println!("-> {nw} WIN / {nl} LOSS (Holm) in the whole family");
    };
    show("F1", &|t| t.scope == "H1500" && t.vs.starts_with("best-fair"));
    show("F2", &|t| t.scope == "H1500");
    show("F3", &|t| t.vs.starts_with("best-fair") && (t.treat == "hier+retr" || t.treat == "session(warm+retr)"));
    show("F4", &|t| t.treat == "hier+retr" || t.treat == "session(warm+retr)");

    println!("\n=== HEADLINE: H1500 and H500 vs best fair baseline (paired 95% CI; Holm over F1) ===");
    for wi in 0..ws.len() {
        for &t in &TREAT {
            for h in ["H500", "H1500"] {
                let (i, tt) = tests.iter().enumerate().find(|(_, x)| x.family == "F1" && x.world == wi && x.scope == h && x.treat == arms[t].name && x.vs.starts_with("best-fair")).unwrap();
                println!("{:<20} {h:<6} {:<22} {:+.4} [{:+.4},{:+.4}] vs {:<30} holm {:.4} {}", ws[wi].name, arms[t].name, tt.d.diff.mean, tt.d.diff.lo, tt.d.diff.hi, tt.vs, adj[i], verdict(adj[i], tt.d.diff.mean));
            }
        }
    }
    let mut crit = vec![];
    println!("\n=== CRITERIA (Holm WIN vs best fair baseline, count of independent treatment worlds out of {N_TREAT}; LOSSes in stress/control worlds) ===");
    for &t in &TREAT {
        for h in ["H500", "H1500"] {
            let verdict_of = |wi: usize| -> Option<&'static str> {
                tests.iter().enumerate().find(|(_, x)| x.family == "F1" && x.world == wi && x.scope == h && x.treat == arms[t].name && x.vs.starts_with("best-fair")).map(|(i, x)| verdict(adj[i], x.d.diff.mean))
            };
            let wins: Vec<&str> = (0..N_TREAT).filter(|&wi| verdict_of(wi) == Some("WIN")).map(|wi| ws[wi].name).collect();
            let losses: Vec<&str> = (0..N_TREAT).filter(|&wi| verdict_of(wi) == Some("LOSS")).map(|wi| ws[wi].name).collect();
            let ctl_loss: Vec<&str> = (IDX_STRESS..ws.len()).filter(|&wi| verdict_of(wi) == Some("LOSS")).map(|wi| ws[wi].name).collect();
            println!("{:<22} {h:<6} WIN in {}/{N_TREAT} {:?}; LOSS in {:?}; stress/control LOSS {:?}", arms[t].name, wins.len(), wins, losses, ctl_loss);
            crit.push(json!({"treatment": arms[t].name, "horizon": h, "treatment_worlds_won": wins, "treatment_worlds_lost": losses, "stress_control_worlds_lost": ctl_loss}));
        }
    }
    let tj = |lo: usize, hi: usize| -> Vec<Value> {
        (lo..hi)
            .map(|i| {
                let t = &tests[i];
                json!({"family": t.family, "world": ws[t.world].name, "scope": t.scope, "treatment": t.treat, "vs": t.vs, "diff": t.d.diff, "wins": t.d.wins, "losses": t.d.losses, "p_raw": t.p, "holm_p": adj[i], "verdict": verdict(adj[i], t.d.diff.mean)})
            })
            .collect()
    };
    let j = json!({
        "experiment": "g5_study", "held_out": format!("{h0}-{h1}"), "n_tasks": N_TASKS, "reveal": REVEAL, "delay": DELAY,
        "tuned_on_seeds_1_8": tuned_json(&tuned),
        "honesty": "synthetic worlds written by me; measures routing + verified-context supply, not model creativity; context effect size d and the wrong-brief penalty are assumptions (see g5_sens)",
        "fair_baselines": arms[..N_FAIR].iter().map(|a| a.name.clone()).collect::<Vec<_>>(),
        "worlds": world_json, "criteria": crit,
        "tests_F1_vs_baselines": tj(0, f1_n), "tests_F2_ablations": tj(f1_n, f2_end), "tests_F3_per_domain": tj(f2_end, f3_end), "tests_F4_sample_efficiency": tj(f3_end, tests.len()),
        "runtime_s": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("\nwrote {out} (total {:.0}s)", t0.elapsed().as_secs_f64());
}

// ------------------------------------------------------------------------------------------------
// One-at-a-time sensitivity of the assumptions the pipeline's gain rests on (retrieval effect = same
// router with vs without retrieval; hier vs warm). NOT part of the main Holm families.
fn sens_mode(tuned: &Tuned, held: &[u64], range: (u64, u64), out: &str, t0: Instant) {
    let base = worlds()[0].clone();
    let mut vs: Vec<(String, WorldCfg)> = vec![("base(agency)".into(), base.clone())];
    for &m in &[0.0, 1.0] {
        vs.push((format!("wrong-brief penalty mis={m}"), WorldCfg { mis: m, ..base.clone() }));
    }
    for &s in &[0.25, 0.5, 1.5] {
        vs.push((format!("context size dscale={s}"), WorldCfg { dscale: s, ..base.clone() }));
    }
    for &k in &[1usize, 2] {
        vs.push((format!("project id tokens={k}/3"), WorldCfg { id_tok: k, ..base.clone() }));
    }
    for &p in &[4usize, 6, 24, 60, 200] {
        vs.push((format!("projects={p}"), WorldCfg { projects: p, ..base.clone() }));
    }
    for &d in &[0.0, 0.6] {
        vs.push((format!("keyword decoy={d}"), WorldCfg { decoy: d, ..base.clone() }));
    }
    vs.push(("context used equally by all agents (u=1)".into(), WorldCfg { ukind: 1, ..base.clone() }));
    vs.push(("context helps only strong agents (u=c^2)".into(), WorldCfg { ukind: 2, ..base.clone() }));
    vs.push(("briefs given (ingested)".into(), WorldCfg { briefs: true, ..base.clone() }));
    vs.push(("briefs given + id tokens=1/3".into(), WorldCfg { briefs: true, id_tok: 1, ..base.clone() }));
    let ws: Vec<WorldCfg> = vs.iter().map(|v| v.1.clone()).collect();
    let arms: Vec<Arm> = vec![
        arm("warm(hebr,tuned)", Kind::Warm(tuned.warm), false, false),
        arm("session(warm+retr)", Kind::Warm(tuned.warm), true, true),
        arm("guarded-safe(plain)", Kind::Safe, false, false),
        arm("session(safe+retr)", Kind::Safe, true, true),
        arm("hier", Kind::Hier(tuned.hier), false, false),
        arm("hier+retr", Kind::Hier(tuned.hier), true, false),
        arm("keyword", Kind::Keyword, false, false),
    ];
    let mat = run_matrix(&ws, &arms, held, tuned.retr);
    let ns = held.len();
    let h = |wi: usize, ai: usize, hz: usize| -> Vec<f64> { (0..ns).map(|s| horizon_means(&mat[wi][ai][s].q)[hz]).collect() };
    let pu = |wi: usize, ai: usize| -> f64 { mat[wi][ai].iter().map(|o| o.purity).sum::<f64>() / ns as f64 };
    let hr = |wi: usize, ai: usize| -> f64 { mat[wi][ai].iter().map(|o| o.ctx_hits as f64).sum::<f64>() / ns as f64 / N_TASKS as f64 };
    println!("\n=== SENSITIVITY (one-at-a-time from `agency`; held {}-{}; H1500 paired diff [95% CI], raw, no multiplicity correction) ===", range.0, range.1);
    println!("{:<32}{:>26}{:>26}{:>26}{:>26}  ctxhit pur", "variant", "retr|warm", "retr|safe", "retr|hier", "hier-warm");
    let mut rows = vec![];
    let fmt = |d: &PairedDiff| format!("{:+.4} [{:+.3},{:+.3}]", d.diff.mean, d.diff.lo, d.diff.hi);
    for (wi, (name, _)) in vs.iter().enumerate() {
        let a = paired_diff(&h(wi, 1, 3), &h(wi, 0, 3));
        let b = paired_diff(&h(wi, 3, 3), &h(wi, 2, 3));
        let c = paired_diff(&h(wi, 5, 3), &h(wi, 4, 3));
        let d = paired_diff(&h(wi, 4, 3), &h(wi, 0, 3));
        println!("{:<32}{:>26}{:>26}{:>26}{:>26}  {:.2}   {:.2}", name, fmt(&a), fmt(&b), fmt(&c), fmt(&d), hr(wi, 1), pu(wi, 1));
        rows.push(json!({"variant": name, "retr_on_warm": {"mean": a.diff.mean, "lo": a.diff.lo, "hi": a.diff.hi}, "retr_on_safe": {"mean": b.diff.mean, "lo": b.diff.lo, "hi": b.diff.hi}, "retr_on_hier": {"mean": c.diff.mean, "lo": c.diff.lo, "hi": c.diff.hi}, "hier_minus_warm": {"mean": d.diff.mean, "lo": d.diff.lo, "hi": d.diff.hi}, "ctx_hit_rate": hr(wi, 1), "mean_purity": pu(wi, 1)}));
    }
    println!("\nH500 retrieval effect on warm / hier+retr vs best of (warm, hier):");
    for (wi, (name, _)) in vs.iter().enumerate() {
        let a = paired_diff(&h(wi, 1, 2), &h(wi, 0, 2));
        let c = paired_diff(&h(wi, 5, 2), &h(wi, 4, 2));
        println!("  {:<32} retr|warm {}   retr|hier {}", name, fmt(&a), fmt(&c));
    }
    let j = json!({"experiment": "g5_study --sens", "held_out": format!("{}-{}", range.0, range.1), "tuned_on_seeds_1_8": tuned_json(tuned), "rows": rows, "runtime_s": t0.elapsed().as_secs_f64()});
    std::fs::write(out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out} (total {:.0}s)", t0.elapsed().as_secs_f64());
}
