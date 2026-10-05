//! g13_tools (W14, R&D round 3, TOOLS track). Modes:
//!   selfcheck                       canonical specs through the REAL tools vs native truth; verifier accept/reject rates
//!   tune    --seeds 1-8             tune baseline + selector hyper-parameters (frozen into Hyper::default afterwards)
//!   eval    --seeds 31-60 --tag held   main study (worlds A and B): tool pipeline vs six plain baselines, steps S0..S2
//!   controls --seeds 31-60 --tag held  no-compute control, misuse controls
//!   sens    --seeds 31-60 --tag held   where does the gain disappear (formulation accuracy, verifier coverage, slope, dup)
//! Protocol: tune seeds 1-8, held-out 31-60, confirm 61-90 (frozen), paired 95% t-CI, Holm.
mod g13_common;

use g13_common::*;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

fn col(rs: &[RunStats], f: impl Fn(&RunStats) -> f64) -> Vec<f64> {
    rs.iter().map(f).collect()
}
fn ms(xs: &[f64]) -> String {
    let v: Vec<f64> = xs.iter().cloned().filter(|x| x.is_finite()).collect();
    if v.len() < 2 {
        return "n/a".into();
    }
    let s = summarize(&v);
    format!("{:.4} [{:.4},{:.4}]", s.mean, s.lo, s.hi)
}
fn mean(xs: &[f64]) -> f64 {
    let v: Vec<f64> = xs.iter().cloned().filter(|x| x.is_finite()).collect();
    if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
}
struct Cmp {
    mean: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    losses: usize,
    ties: usize,
    p: f64,
}
fn cmp(a: &[f64], b: &[f64]) -> Cmp {
    let pd = paired_diff(a, b);
    Cmp { mean: pd.diff.mean, lo: pd.diff.lo, hi: pd.diff.hi, wins: pd.wins, losses: pd.losses, ties: pd.ties, p: paired_t_pvalue(a, b) }
}
fn cj(c: &Cmp, adj: Option<f64>) -> Value {
    json!({"mean": c.mean, "lo": c.lo, "hi": c.hi, "wins": c.wins, "losses": c.losses, "ties": c.ties, "p": c.p, "holm_p": adj})
}
fn cs(c: &Cmp, adj: Option<f64>) -> String {
    format!(
        "{:+.4} [{:+.4},{:+.4}] w/l/t {}/{}/{} p={:.2e}{}",
        c.mean,
        c.lo,
        c.hi,
        c.wins,
        c.losses,
        c.ties,
        c.p,
        adj.map(|a| format!(" holm={:.2e}", a)).unwrap_or_default()
    )
}

/// results[arm][seed]
fn run_all(cfg: &WorldCfg, arms: &[ArmSpec], seeds: &[u64], h: &Hyper) -> Vec<Vec<RunStats>> {
    let per_seed: Vec<Vec<RunStats>> = seeds
        .par_iter()
        .map(|&sd| {
            let w = gen_world(cfg, sd);
            arms.iter().map(|a| run_arm(&w, a, h, sd)).collect()
        })
        .collect();
    (0..arms.len()).map(|ai| per_seed.iter().map(|row| row[ai].clone()).collect()).collect()
}

fn baseline_arms() -> Vec<ArmSpec> {
    ROUTERS.iter().map(|(rt, n)| ArmSpec::plain(*rt, n)).collect()
}
fn tool_arms() -> Vec<ArmSpec> {
    let r = Rt::Safe;
    vec![
        ArmSpec::tools("T-always(noverify)", r, SelPol::Always, false, false, false),
        ArmSpec::tools("T-always(verify)", r, SelPol::Always, true, false, false),
        ArmSpec::tools("T-static(noverify)", r, SelPol::Static, false, false, false),
        ArmSpec::tools("T-static(verify) [S0]", r, SelPol::Static, true, false, false),
        ArmSpec { name: "V-only(plain+verify+retry)".into(), router: r, pipe: Pipe::VerifyOnly },
        ArmSpec { name: "vote3(plain, 3x cost)".into(), router: r, pipe: Pipe::Vote3 },
        ArmSpec::tools("T-oracle(verify)", r, SelPol::Oracle, true, false, false),
        ArmSpec::tools("T-learned(noverify)", r, SelPol::Learned, false, false, false),
        ArmSpec::tools("T-learned(verify) [S1]", r, SelPol::Learned, true, false, false),
        ArmSpec::tools("T-learned+cache [S2a]", r, SelPol::Learned, true, true, false),
        ArmSpec::tools("T-learned+cache+memo [S2b]", r, SelPol::Learned, true, true, true),
        ArmSpec::tools("T-wrongtool(verify)", r, SelPol::WrongTool, true, false, false),
    ]
}

fn idx_of(arms: &[ArmSpec], name: &str) -> usize {
    arms.iter().position(|a| a.name == name).unwrap_or_else(|| panic!("arm {name}"))
}

fn arm_json(rs: &[RunStats]) -> Value {
    json!({
        "correct": col(rs, |s| s.correct), "cost": col(rs, |s| s.cost), "cost_nominal": col(rs, |s| s.cost_nominal),
        "correct_compute": col(rs, |s| s.correct_compute), "correct_qual": col(rs, |s| s.correct_qual),
        "tool_calls": col(rs, |s| s.tool_calls), "verify_calls": col(rs, |s| s.verify_calls), "verify_reject": col(rs, |s| s.verify_reject),
        "cache_hit": col(rs, |s| s.cache_hit), "cache_saved": col(rs, |s| s.cache_saved), "memo_hit": col(rs, |s| s.memo_hit),
        "skip_qual": col(rs, |s| s.skip_qual), "right_tool": col(rs, |s| s.right_tool), "tool_used_compute": col(rs, |s| s.tool_used_compute),
    })
}

fn print_table(arms: &[ArmSpec], res: &[Vec<RunStats>]) {
    println!("{:<30} {:>22} {:>14} {:>14} {:>9} {:>9}", "arm", "correct", "compute", "qual", "cost/task", "tools/t");
    for (a, rs) in arms.iter().zip(res) {
        println!(
            "{:<30} {:>22} {:>14.4} {:>14.4} {:>9.2} {:>9.3}",
            a.name,
            ms(&col(rs, |s| s.correct)),
            mean(&col(rs, |s| s.correct_compute)),
            mean(&col(rs, |s| s.correct_qual)),
            mean(&col(rs, |s| s.cost)),
            mean(&col(rs, |s| s.tool_calls))
        );
    }
}

// ---------------------------------------------------------------------------------------------

fn world_cfgs(args: &[String]) -> Vec<WorldCfg> {
    let which = arg(args, "--worlds", "A,B");
    let mut v = Vec::new();
    let n: usize = arg(args, "--n", "600").parse().unwrap();
    for w in which.split(',') {
        let mut c = match w {
            "A" => WorldCfg::world_a(),
            "B" => WorldCfg::world_b(),
            _ => panic!("world {w}"),
        };
        c.n = n;
        v.push(c);
    }
    v
}

fn mode_selfcheck(args: &[String]) {
    let seeds = parse_range(args, "--seeds", (1, 6));
    let sc = selfcheck(&[WorldCfg::world_a(), WorldCfg::world_b()], &seeds);
    println!("{:<8} {:>6} {:>12} {:>14} {:>9} {:>14} {:>7}", "kind", "n", "tool==truth", "verifier acc", "wrong n", "verifier rej", "errors");
    let mut rows = Vec::new();
    let mut ok = true;
    for (k, n, tc, va, wn, vr, er) in &sc.per_kind {
        println!("{:<8} {:>6} {:>12} {:>14} {:>9} {:>14} {:>7}", k, n, tc, va, wn, vr, er);
        rows.push(json!({"kind": k, "n": n, "tool_correct": tc, "verifier_accepts_correct": va, "wrong_claims": wn, "verifier_rejects_wrong": vr, "tool_errors": er}));
        ok &= tc == n && va == n && vr == wn && *er == 0;
    }
    println!("{}", if ok { "SELFCHECK PASS: every canonical spec is computed correctly by the real tool, accepted by the verifier, and every wrong claim is rejected" } else { "SELFCHECK FAIL" });
    let out = arg(args, "--out", "results/g13_selfcheck.json");
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"seeds": seeds, "pass": ok, "rows": rows})).unwrap()).unwrap();
    if !ok {
        std::process::exit(1);
    }
}

fn mode_tune(args: &[String]) {
    let seeds = parse_range(args, "--seeds", (1, 8));
    let cfgs = vec![WorldCfg::world_a(), WorldCfg::world_b(), WorldCfg::world_a().noncompute()];
    let base = Hyper::default();
    // baseline hyper-parameters (plain arms): objective = mean correctness over worlds A and B
    let eval_plain = |h: &Hyper, rt: Rt| -> f64 {
        let mut tot = 0.0;
        for c in &cfgs[..2] {
            let r = run_all(c, &[ArmSpec::plain(rt, "x")], &seeds, h);
            tot += mean(&col(&r[0], |s| s.correct));
        }
        tot / 2.0
    };
    let mut best_eps = (f64::MIN, base.eps_g, base.eps_e);
    for (g, e) in [(1.0, 0.02), (1.0, 0.05), (1.0, 0.1), (0.98, 0.05), (0.9, 0.0), (0.95, 0.1)] {
        let h = Hyper { eps_g: g, eps_e: e, ..base };
        let v = eval_plain(&h, Rt::Eps);
        println!("eps-greedy gamma={g} eps={e}: {v:.4}");
        if v > best_eps.0 {
            best_eps = (v, g, e);
        }
    }
    let mut best_pts = (f64::MIN, base.pts_g);
    for g in [1.0, 0.98, 0.95] {
        let h = Hyper { pts_g: g, ..base };
        let v = eval_plain(&h, Rt::Pts);
        println!("pooled-ts gamma={g}: {v:.4}");
        if v > best_pts.0 {
            best_pts = (v, g);
        }
    }
    let mut best_warm = (f64::MIN, base.warm_k);
    for k in [1.0, 3.0, 10.0] {
        let h = Hyper { warm_k: k, ..base };
        let v = eval_plain(&h, Rt::Warm);
        println!("warm(hebr) k={k}: {v:.4}");
        if v > best_warm.0 {
            best_warm = (v, k);
        }
    }
    // selector hyper-parameters: objective = correctness - 0.02*cost/MODEL averaged over A, B and the no-compute control
    let arm = ArmSpec::tools("learned", Rt::Safe, SelPol::Learned, true, false, false);
    let mut best_sel = (f64::MIN, base.sel_eps, base.sel_shrink, base.sel_opt, base.kappa, base.kw_prior);
    let mut grid_log = Vec::new();
    for sel_eps in [0.02, 0.05, 0.1] {
        for sel_shrink in [1.0, 2.0, 4.0] {
            for sel_opt in [0.7, 0.9] {
                for (kappa, kw_prior) in [0.02, 0.04, 0.08].iter().flat_map(|k| [0.0, 5.0, 10.0, 20.0].map(|p| (*k, p))) {
                    let h = Hyper { sel_eps, sel_shrink, sel_opt, kappa, kw_prior, ..base };
                    let mut tot = 0.0;
                    for c in &cfgs {
                        let r = run_all(c, std::slice::from_ref(&arm), &seeds, &h);
                        tot += mean(&col(&r[0], |s| s.correct)) - 0.02 * mean(&col(&r[0], |s| s.cost)) / MODEL_COST;
                    }
                    let v = tot / cfgs.len() as f64;
                    grid_log.push(json!({"sel_eps": sel_eps, "sel_shrink": sel_shrink, "sel_opt": sel_opt, "kappa": kappa, "kw_prior": kw_prior, "objective": v}));
                    if v > best_sel.0 {
                        best_sel = (v, sel_eps, sel_shrink, sel_opt, kappa, kw_prior);
                    }
                }
            }
        }
    }
    println!("best selector: eps={} shrink={} opt={} kappa={} kw_prior={} objective={:.4}", best_sel.1, best_sel.2, best_sel.3, best_sel.4, best_sel.5, best_sel.0);
    let chosen = Hyper { eps_g: best_eps.1, eps_e: best_eps.2, pts_g: best_pts.1, warm_k: best_warm.1, sel_eps: best_sel.1, sel_shrink: best_sel.2, sel_opt: best_sel.3, kappa: best_sel.4, kw_prior: best_sel.5 };
    println!("CHOSEN (tune seeds {:?}): {:?}", (seeds[0], seeds[seeds.len() - 1]), chosen);
    println!("Hyper::default() currently: {:?}", base);
    let out = arg(args, "--out", "results/g13_tune.json");
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&json!({
            "seeds": seeds,
            "chosen": {"eps_g": chosen.eps_g, "eps_e": chosen.eps_e, "pts_g": chosen.pts_g, "warm_k": chosen.warm_k, "sel_eps": chosen.sel_eps, "sel_shrink": chosen.sel_shrink, "sel_opt": chosen.sel_opt, "kappa": chosen.kappa, "kw_prior": chosen.kw_prior},
            "default_in_code": {"eps_g": base.eps_g, "eps_e": base.eps_e, "pts_g": base.pts_g, "warm_k": base.warm_k, "sel_eps": base.sel_eps, "sel_shrink": base.sel_shrink, "sel_opt": base.sel_opt, "kappa": base.kappa, "kw_prior": base.kw_prior},
            "selector_grid": grid_log,
        }))
        .unwrap(),
    )
    .unwrap();
}

fn mode_eval(args: &[String]) {
    let seeds = parse_range(args, "--seeds", (31, 60));
    let tag = arg(args, "--tag", "held");
    let h = Hyper::default();
    let mut all = serde_json::Map::new();
    for cfg in world_cfgs(args) {
        println!("\n================ WORLD {} (n={}, qv={}, misuse={}, dup={}) seeds {}-{} ================", cfg.name, cfg.n, cfg.qv, cfg.misuse, cfg.dup, seeds[0], seeds[seeds.len() - 1]);
        let mut arms = baseline_arms();
        let nb = arms.len();
        arms.extend(tool_arms());
        let res = run_all(&cfg, &arms, &seeds, &h);
        print_table(&arms, &res);
        // best fair baseline on this seed set (favours the baselines: chosen on the evaluation seeds themselves)
        let bi = (0..nb).max_by(|a, b| mean(&col(&res[*a], |s| s.correct)).partial_cmp(&mean(&col(&res[*b], |s| s.correct))).unwrap()).unwrap();
        println!("best fair baseline on these seeds: {} ({:.4})", arms[bi].name, mean(&col(&res[bi], |s| s.correct)));
        let c = |name: &str| col(&res[idx_of(&arms, name)], |s| s.correct);
        let k = |name: &str| col(&res[idx_of(&arms, name)], |s| s.cost);
        let bcor = col(&res[bi], |s| s.correct);
        let bcost = col(&res[bi], |s| s.cost);
        let mut comps = serde_json::Map::new();
        // family 1/2: S0 and S1 vs every baseline, Holm over the 6
        for (fam, name) in [("S0_vs_baselines", "T-static(verify) [S0]"), ("S1_vs_baselines", "T-learned(verify) [S1]"), ("S2b_vs_baselines", "T-learned+cache+memo [S2b]")] {
            let cs_: Vec<Cmp> = (0..nb).map(|b| cmp(&c(name), &col(&res[b], |s| s.correct))).collect();
            let adj = holm_adjust(&cs_.iter().map(|x| x.p).collect::<Vec<_>>());
            println!("-- {fam}: correctness diff ({name} - baseline), Holm over {nb}");
            let mut rows = Vec::new();
            for b in 0..nb {
                println!("   vs {:<18} {}", arms[b].name, cs(&cs_[b], Some(adj[b])));
                rows.push(json!({"baseline": arms[b].name, "diff": cj(&cs_[b], Some(adj[b]))}));
            }
            comps.insert(fam.into(), json!(rows));
        }
        let mut single = |key: &str, label: &str, a: Vec<f64>, b: Vec<f64>| {
            let x = cmp(&a, &b);
            println!("-- {label}: {}", cs(&x, None));
            comps.insert(key.into(), cj(&x, None));
        };
        single("S0_vs_best_baseline_correct", "S0 - best baseline  (correctness)", c("T-static(verify) [S0]"), bcor.clone());
        single("S0_vs_best_baseline_cost", "S0 - best baseline  (cost/task)", k("T-static(verify) [S0]"), bcost.clone());
        single("S1_vs_best_baseline_correct", "S1 - best baseline  (correctness)", c("T-learned(verify) [S1]"), bcor.clone());
        single("S1_vs_best_baseline_cost", "S1 - best baseline  (cost/task)", k("T-learned(verify) [S1]"), bcost.clone());
        single("S1_vs_S0_correct", "STEP 1 learned selection: S1 - S0 (correctness)", c("T-learned(verify) [S1]"), c("T-static(verify) [S0]"));
        single("S1_vs_S0_cost", "STEP 1 learned selection: S1 - S0 (cost/task)", k("T-learned(verify) [S1]"), k("T-static(verify) [S0]"));
        single("S1_vs_always_correct", "S1 - T-always(verify) (correctness)", c("T-learned(verify) [S1]"), c("T-always(verify)"));
        single("S1_vs_always_cost", "S1 - T-always(verify) (cost/task)", k("T-learned(verify) [S1]"), k("T-always(verify)"));
        single("S1_vs_oracle_correct", "S1 - oracle selection (correctness)", c("T-learned(verify) [S1]"), c("T-oracle(verify)"));
        single("S2a_vs_S1_correct", "STEP 2a tool cache: S2a - S1 (correctness; must be exactly 0)", c("T-learned+cache [S2a]"), c("T-learned(verify) [S1]"));
        single("S2a_vs_S1_cost", "STEP 2a tool cache: S2a - S1 (cost/task)", k("T-learned+cache [S2a]"), k("T-learned(verify) [S1]"));
        single("S2b_vs_S2a_correct", "STEP 2b verified-answer memo: S2b - S2a (correctness)", c("T-learned+cache+memo [S2b]"), c("T-learned+cache [S2a]"));
        single("S2b_vs_S2a_cost", "STEP 2b verified-answer memo: S2b - S2a (cost/task)", k("T-learned+cache+memo [S2b]"), k("T-learned+cache [S2a]"));
        single("verify_value_static_correct", "verification: T-static(verify) - T-static(noverify) (correctness)", c("T-static(verify) [S0]"), c("T-static(noverify)"));
        single("verify_value_learned_correct", "verification: S1 - T-learned(noverify) (correctness)", c("T-learned(verify) [S1]"), c("T-learned(noverify)"));
        single("tools_vs_verifyonly_correct", "offload: S1 - V-only (correctness)", c("T-learned(verify) [S1]"), c("V-only(plain+verify+retry)"));
        single("S1_vs_vote3_correct", "cost-matched reference: S1 - vote3 (correctness)", c("T-learned(verify) [S1]"), c("vote3(plain, 3x cost)"));
        single("S1_vs_vote3_cost", "cost-matched reference: S1 - vote3 (cost/task)", k("T-learned(verify) [S1]"), k("vote3(plain, 3x cost)"));
        single("vote3_vs_best_baseline_correct", "vote3 - best baseline (correctness)", c("vote3(plain, 3x cost)"), bcor.clone());
        single("verifyonly_vs_plainsafe_correct", "verify+retry without tools: V-only - guarded-safe plain (correctness)", c("V-only(plain+verify+retry)"), c("guarded-safe"));
        single("wrongtool_vs_plainsafe_correct", "tool misuse (always wrong tool, verified): T-wrongtool - guarded-safe plain (correctness)", c("T-wrongtool(verify)"), c("guarded-safe"));
        single("wrongtool_vs_plainsafe_cost", "tool misuse: T-wrongtool - guarded-safe plain (cost/task)", k("T-wrongtool(verify)"), k("guarded-safe"));
        // cache identity check (hard): per seed, the correct vectors of S1 and S2a are identical
        let s1 = &res[idx_of(&arms, "T-learned(verify) [S1]")];
        let s2a = &res[idx_of(&arms, "T-learned+cache [S2a]")];
        let identical = s1.iter().zip(s2a).all(|(x, y)| x.correct_vec == y.correct_vec);
        println!("-- cache never changes a result: S1 vs S2a per-task correctness vectors identical on all {} seeds: {identical}", seeds.len());
        comps.insert("cache_outputs_identical".into(), json!(identical));
        let sel = &res[idx_of(&arms, "T-learned(verify) [S1]")];
        println!(
            "-- learned selector: right tool on compute tasks {:.3}, skipped tool on qual tasks {:.3}, tool used on compute {:.3}; static: right {:.3}, skip {:.3}",
            mean(&col(sel, |s| s.right_tool)),
            mean(&col(sel, |s| s.skip_qual)),
            mean(&col(sel, |s| s.tool_used_compute)),
            mean(&col(&res[idx_of(&arms, "T-static(verify) [S0]")], |s| s.right_tool)),
            mean(&col(&res[idx_of(&arms, "T-static(verify) [S0]")], |s| s.skip_qual)),
        );
        let s2 = &res[idx_of(&arms, "T-learned+cache [S2a]")];
        let s2b = &res[idx_of(&arms, "T-learned+cache+memo [S2b]")];
        println!(
            "-- cache: hit rate {:.3} (tool+verifier calls), tool cost saved/task {:.3} of {:.3} nominal/task; memo hit rate {:.3}",
            mean(&col(s2, |s| s.cache_hit)),
            mean(&col(s2, |s| s.cache_saved)),
            mean(&col(s2, |s| s.cost_nominal)),
            mean(&col(s2b, |s| s.memo_hit)),
        );
        println!("-- verifier: reject rate among verified calls (S1) {:.3}", mean(&col(sel, |s| s.verify_reject)));
        let mut armj = serde_json::Map::new();
        for (a, rs) in arms.iter().zip(&res) {
            armj.insert(a.name.clone(), arm_json(rs));
        }
        all.insert(
            cfg.name.to_string(),
            json!({"config": {"n": cfg.n, "qv": cfg.qv, "misuse": cfg.misuse, "dup": cfg.dup, "shape_noise": cfg.shape_noise, "trust_h": cfg.trust_h, "style": cfg.style},
                   "best_baseline": arms[bi].name, "arms": armj, "comparisons": comps}),
        );
    }
    let out = arg(args, "--out", &format!("results/g13_{tag}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"tag": tag, "seeds": seeds, "hyper": format!("{:?}", h), "worlds": all})).unwrap()).unwrap();
    println!("\nwrote {out}");
}

fn mode_controls(args: &[String]) {
    let seeds = parse_range(args, "--seeds", (31, 60));
    let tag = arg(args, "--tag", "held");
    let h = Hyper::default();
    let mut all = serde_json::Map::new();
    for base in world_cfgs(args) {
        for (cname, cfg) in [("noncompute", base.clone().noncompute()), ("misuse_heavy", base.clone().misuse_heavy())] {
            println!("\n================ CONTROL {cname} on generator {} ================", base.name);
            let mut arms = vec![ArmSpec::plain(Rt::Safe, "guarded-safe"), ArmSpec::plain(Rt::HebR, "hebbian-robust")];
            arms.extend(tool_arms());
            let res = run_all(&cfg, &arms, &seeds, &h);
            print_table(&arms, &res);
            let plain = idx_of(&arms, "guarded-safe");
            let pc = col(&res[plain], |s| s.correct);
            let pk = col(&res[plain], |s| s.cost);
            let mut rows = serde_json::Map::new();
            for name in ["T-always(verify)", "T-static(verify) [S0]", "T-oracle(verify)", "T-learned(verify) [S1]", "T-learned+cache+memo [S2b]", "T-wrongtool(verify)"] {
                let i = idx_of(&arms, name);
                let dc = cmp(&col(&res[i], |s| s.correct), &pc);
                let dk = cmp(&col(&res[i], |s| s.cost), &pk);
                println!("   {name:<30} correct vs plain: {}   cost vs plain: {}", cs(&dc, None), cs(&dk, None));
                rows.insert(name.into(), json!({"correct_vs_plain": cj(&dc, None), "cost_vs_plain": cj(&dk, None)}));
            }
            if cname == "noncompute" {
                let i = idx_of(&arms, "T-learned(verify) [S1]");
                let dc = cmp(&col(&res[i], |s| s.correct), &pc);
                let within = dc.lo > -0.01 && dc.hi < 0.01;
                println!("   TOST-style: S1 correctness diff CI inside +-0.01 (harmless): {within}; skip rate on qual tasks {:.3}", mean(&col(&res[i], |s| s.skip_qual)));
                rows.insert("S1_noharm_within_1pp".into(), json!(within));
            }
            let mut armj = serde_json::Map::new();
            for (a, rs) in arms.iter().zip(&res) {
                armj.insert(a.name.clone(), arm_json(rs));
            }
            all.insert(format!("{}:{}", base.name, cname), json!({"comparisons": rows, "arms": armj}));
        }
    }
    let out = arg(args, "--out", &format!("results/g13_controls_{tag}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"tag": tag, "seeds": seeds, "controls": all})).unwrap()).unwrap();
    println!("\nwrote {out}");
}

fn mode_sens(args: &[String]) {
    let seeds = parse_range(args, "--seeds", (31, 60));
    let tag = arg(args, "--tag", "held");
    let h = Hyper::default();
    let mut rows = Vec::new();
    for base in world_cfgs(args) {
        println!("\n================ SENSITIVITY on generator {} ================", base.name);
        let mut settings: Vec<(String, WorldCfg)> = vec![("baseline".into(), base.clone())];
        for dlt in [-0.10, -0.20, -0.30] {
            let mut c = base.clone();
            c.form_delta = dlt;
            settings.push((format!("form_delta={dlt}"), c));
        }
        for qv in [1.0, 0.7, 0.5, 0.0] {
            let mut c = base.clone();
            c.qv = qv;
            settings.push((format!("qv={qv}"), c));
        }
        for sc in [0.5, 0.25] {
            let mut c = base.clone();
            c.sens_scale = sc;
            settings.push((format!("direct_error_slope_x{sc}"), c));
        }
        let mut c = base.clone();
        c.form_delta = -0.30;
        c.qv = 0.0;
        settings.push(("form_delta=-0.30+qv=0".into(), c));
        for dup in [0.0, 0.4] {
            let mut c = base.clone();
            c.dup = dup;
            settings.push((format!("dup={dup}"), c));
        }
        for (label, cfg) in settings {
            let mut arms = baseline_arms();
            let nb = arms.len();
            arms.push(ArmSpec::tools("T-static(verify) [S0]", Rt::Safe, SelPol::Static, true, false, false));
            arms.push(ArmSpec::tools("T-learned(verify) [S1]", Rt::Safe, SelPol::Learned, true, false, false));
            arms.push(ArmSpec::tools("T-learned(noverify)", Rt::Safe, SelPol::Learned, false, false, false));
            arms.push(ArmSpec::tools("T-learned+cache [S2a]", Rt::Safe, SelPol::Learned, true, true, false));
            arms.push(ArmSpec::tools("T-learned+cache+memo [S2b]", Rt::Safe, SelPol::Learned, true, true, true));
            let res = run_all(&cfg, &arms, &seeds, &h);
            let bi = (0..nb).max_by(|a, b| mean(&col(&res[*a], |s| s.correct)).partial_cmp(&mean(&col(&res[*b], |s| s.correct))).unwrap()).unwrap();
            let bc = col(&res[bi], |s| s.correct);
            let bk = col(&res[bi], |s| s.cost);
            let c = |n: &str| col(&res[idx_of(&arms, n)], |s| s.correct);
            let k = |n: &str| col(&res[idx_of(&arms, n)], |s| s.cost);
            let s1 = cmp(&c("T-learned(verify) [S1]"), &bc);
            let nv = cmp(&c("T-learned(noverify)"), &bc);
            let s1k = cmp(&k("T-learned(verify) [S1]"), &bk);
            let s2a = cmp(&k("T-learned+cache [S2a]"), &k("T-learned(verify) [S1]"));
            let s2b = cmp(&k("T-learned+cache+memo [S2b]"), &k("T-learned+cache [S2a]"));
            let s2bc = cmp(&c("T-learned+cache+memo [S2b]"), &c("T-learned+cache [S2a]"));
            println!(
                "{:<26} best={:<16} S1-best correct {}  | noverify-best {:+.4} | S1 cost vs best {:+.2} | cache saves {:+.3} | memo cost {:+.3} correct {:+.4}",
                label, arms[bi].name, cs(&s1, None), nv.mean, s1k.mean, s2a.mean, s2b.mean, s2bc.mean
            );
            rows.push(json!({"world": base.name, "setting": label, "best_baseline": arms[bi].name, "best_baseline_correct": mean(&bc),
                "S1_vs_best_correct": cj(&s1, None), "S1_noverify_vs_best_correct": cj(&nv, None), "S1_vs_best_cost": cj(&s1k, None),
                "S2a_vs_S1_cost": cj(&s2a, None), "S2b_vs_S2a_cost": cj(&s2b, None), "S2b_vs_S2a_correct": cj(&s2bc, None)}));
        }
    }
    let out = arg(args, "--out", &format!("results/g13_sens_{tag}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"tag": tag, "seeds": seeds, "rows": rows})).unwrap()).unwrap();
    println!("\nwrote {out}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("selfcheck") => mode_selfcheck(&args),
        Some("tune") => mode_tune(&args),
        Some("eval") => mode_eval(&args),
        Some("controls") => mode_controls(&args),
        Some("sens") => mode_sens(&args),
        _ => {
            eprintln!("usage: g13_tools selfcheck|tune|eval|controls|sens [--seeds a-b] [--tag t] [--out path] [--worlds A,B] [--n 600]");
            std::process::exit(2);
        }
    }
}
