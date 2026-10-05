//! g21_headline (W22): how much of the W14 tools lift survives when the spec comes from a text EXTRACTOR (not given clean)
//! and the verifier is INDEPENDENT of the evaluator. Modes:
//!   tune     --seeds 1-8                 theta for the extractor-confidence rule, selector hyper-parameters (frozen in `hx()` afterwards)
//!   eval     --seeds 31-60 --tag held    main study, worlds A and B: best fair baseline vs extractor arms vs the W14 clean-spec reference
//!   controls --seeds 31-60 --tag held    no-compute control (TOST +-0.01), extractor false positives on prose
//!   bugs     --seeds 31-60 --tag held    seeded evaluator bugs active in the deployed tools: shared vs independent verifier, end to end
//!   sens     --seeds 31-60 --tag held    extractor cost, phrasing mix, extractor accuracy -> lift
//! Protocol: tune seeds 1-8, held-out 31-60, confirm 61-90 (frozen), paired 95% t-CI, Holm over the six plain baselines.
//! All numbers are SYNTHETIC until validated on real traffic.
mod g21_common;

use g21_common::g13::{self, run_arm, ArmSpec, Hyper, Pipe, Rt, RunStats, SelPol, WorldCfg, MODEL_COST, ROUTERS};
use g21_common::nlgen::NlCfg;
use g21_common::pipe::{build_xworld, run_x, Src, Ver, XArm, XCfg, XRun, XSel};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

/// FROZEN after `tune --seeds 1-8` (see results/g21_tune.json)
fn hx() -> (Hyper, f64) {
    // (selector hyper-parameters, theta for the confidence rule)
    (Hyper { eps_g: 1.0, eps_e: 0.1, pts_g: 1.0, warm_k: 10.0, sel_eps: 0.02, sel_shrink: 4.0, sel_opt: 0.7, kappa: 0.04, kw_prior: 20.0 }, 0.4)
}

fn col(rs: &[XRun], f: impl Fn(&XRun) -> f64) -> Vec<f64> {
    rs.iter().map(f).collect()
}
fn mean(xs: &[f64]) -> f64 {
    let v: Vec<f64> = xs.iter().cloned().filter(|x| x.is_finite()).collect();
    if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
}
fn ms(xs: &[f64]) -> String {
    let v: Vec<f64> = xs.iter().cloned().filter(|x| x.is_finite()).collect();
    if v.len() < 2 {
        return "n/a".into();
    }
    let s = summarize(&v);
    format!("{:.4} [{:.4},{:.4}]", s.mean, s.lo, s.hi)
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
    format!("{:+.4} [{:+.4},{:+.4}] w/l/t {}/{}/{} p={:.2e}{}", c.mean, c.lo, c.hi, c.wins, c.losses, c.ties, c.p, adj.map(|a| format!(" holm={:.2e}", a)).unwrap_or_default())
}

#[derive(Clone)]
enum Any {
    G(ArmSpec),
    X(XArm),
}
impl Any {
    fn name(&self) -> &str {
        match self {
            Any::G(a) => &a.name,
            Any::X(a) => &a.name,
        }
    }
}

fn from_rs(st: RunStats) -> XRun {
    XRun { st, ..Default::default() }
}

/// results[arm][seed]
fn run_all(cfg: &WorldCfg, ncfg: &NlCfg, arms: &[Any], seeds: &[u64], h: &Hyper, xc: &XCfg, mutant: u8) -> Vec<Vec<XRun>> {
    let per_seed: Vec<Vec<XRun>> = seeds
        .par_iter()
        .map(|&sd| {
            let xw = build_xworld(cfg, ncfg, sd);
            arms.iter()
                .map(|a| match a {
                    Any::G(g) => from_rs(run_arm(&xw.w, g, &Hyper::default(), sd)),
                    Any::X(x) => run_x(&xw, x, h, xc, sd, mutant),
                })
                .collect()
        })
        .collect();
    (0..arms.len()).map(|ai| per_seed.iter().map(|row| row[ai].clone()).collect()).collect()
}

fn baseline_arms() -> Vec<Any> {
    ROUTERS.iter().map(|(rt, n)| Any::G(ArmSpec::plain(*rt, n))).collect()
}
fn idx(arms: &[Any], name: &str) -> usize {
    arms.iter().position(|a| a.name() == name).unwrap_or_else(|| panic!("arm {name}"))
}

const H_SHARED: &str = "X-learned(conf)+shared";
const H_INDEP: &str = "X-learned(conf)+indep [HEADLINE]";

fn x_arms(theta: f64) -> Vec<Any> {
    let r = Rt::Safe;
    let _ = r;
    vec![
        Any::G(ArmSpec::tools("W14-S0 clean spec (agent formulates)", Rt::Safe, SelPol::Static, true, false, false)),
        Any::G(ArmSpec::tools("W14-S1 clean spec (agent formulates)", Rt::Safe, SelPol::Learned, true, false, false)),
        Any::G(ArmSpec { name: "vote3(plain, 3x cost)".into(), router: Rt::Safe, pipe: Pipe::Vote3 }),
        Any::X(XArm::new("X-clean-spec+shared (no extractor errors)", XSel::Learned { conf: false }, Ver::Shared, Src::Clean, false, false)),
        Any::X(XArm::new("X-static+shared", XSel::Static, Ver::Shared, Src::Extract, false, false)),
        Any::X(XArm::new("X-rule(theta)+shared", XSel::Rule(theta), Ver::Shared, Src::Extract, false, false)),
        Any::X(XArm::new("X-rule(theta)+indep", XSel::Rule(theta), Ver::Indep { meta: false }, Src::Extract, false, false)),
        Any::X(XArm::new("X-learned(noconf)+shared", XSel::Learned { conf: false }, Ver::Shared, Src::Extract, false, false)),
        Any::X(XArm::new(H_SHARED, XSel::Learned { conf: true }, Ver::Shared, Src::Extract, false, false)),
        Any::X(XArm::new(H_INDEP, XSel::Learned { conf: true }, Ver::Indep { meta: false }, Src::Extract, false, false)),
        Any::X(XArm::new("X-learned(conf)+indep+meta", XSel::Learned { conf: true }, Ver::Indep { meta: true }, Src::Extract, false, false)),
        Any::X(XArm::new("X-learned(conf)+noverify", XSel::Learned { conf: true }, Ver::None, Src::Extract, false, false)),
        Any::X(XArm::new("X-learned(conf)+indep+cache", XSel::Learned { conf: true }, Ver::Indep { meta: false }, Src::Extract, true, false)),
        Any::X(XArm::new("X-learned(conf)+indep+cache+memo", XSel::Learned { conf: true }, Ver::Indep { meta: false }, Src::Extract, true, true)),
        Any::X(XArm::new("X-learned(conf+kgprior)+indep [POST-HOC]", XSel::LearnedKg, Ver::Indep { meta: false }, Src::Extract, false, false)),
        Any::X(XArm::new("X-oracle-kind+indep (extractor spec)", XSel::Oracle, Ver::Indep { meta: false }, Src::Extract, false, false)),
    ]
}

fn world_cfgs(args: &[String]) -> Vec<(WorldCfg, NlCfg)> {
    let which = g13::arg(args, "--worlds", "A,B");
    let n: usize = g13::arg(args, "--n", "600").parse().unwrap();
    let mut v = Vec::new();
    for w in which.split(',') {
        let (mut c, nc) = match w {
            "A" => (WorldCfg::world_a(), NlCfg::world_a()),
            "B" => (WorldCfg::world_b(), NlCfg::world_b()),
            _ => panic!("world {w}"),
        };
        c.n = n;
        v.push((c, nc));
    }
    v
}

fn arm_json(rs: &[XRun]) -> Value {
    json!({
        "correct": col(rs, |s| s.st.correct), "cost": col(rs, |s| s.st.cost), "cost_nominal": col(rs, |s| s.st.cost_nominal),
        "correct_compute": col(rs, |s| s.st.correct_compute), "correct_qual": col(rs, |s| s.st.correct_qual),
        "tool_calls": col(rs, |s| s.st.tool_calls), "verify_calls": col(rs, |s| s.st.verify_calls),
        "cache_hit": col(rs, |s| s.st.cache_hit), "cache_saved": col(rs, |s| s.st.cache_saved), "memo_hit": col(rs, |s| s.st.memo_hit),
        "skip_qual": col(rs, |s| s.st.skip_qual), "right_tool": col(rs, |s| s.st.right_tool), "tool_used_compute": col(rs, |s| s.st.tool_used_compute),
        "no_tool": col(rs, |s| s.no_tool), "silent_wrong": col(rs, |s| s.silent_wrong), "tool_right": col(rs, |s| s.tool_right),
        "rejects": col(rs, |s| s.rejects), "fallback": col(rs, |s| s.fallback), "ex_cost": col(rs, |s| s.ex_cost), "ver_cost": col(rs, |s| s.ver_cost),
    })
}

fn print_table(arms: &[Any], res: &[Vec<XRun>]) {
    println!("{:<44} {:>22} {:>8} {:>8} {:>8} {:>8} {:>7} {:>7}", "arm", "correct", "cost/t", "no-tool", "silentW", "toolOK", "fallbk", "rej/t");
    for (a, rs) in arms.iter().zip(res) {
        println!(
            "{:<44} {:>22} {:>8.2} {:>8.3} {:>8.3} {:>8.3} {:>7.3} {:>7.3}",
            a.name(),
            ms(&col(rs, |s| s.st.correct)),
            mean(&col(rs, |s| s.st.cost)),
            mean(&col(rs, |s| s.no_tool)),
            mean(&col(rs, |s| s.silent_wrong)),
            mean(&col(rs, |s| s.tool_right)),
            mean(&col(rs, |s| s.fallback)),
            mean(&col(rs, |s| s.rejects))
        );
    }
}

// ---------------------------------------------------------------------------------------------

fn mode_tune(args: &[String]) {
    let seeds = g13::parse_range(args, "--seeds", (1, 8));
    let (h0, _) = hx();
    let cfgs: Vec<(WorldCfg, NlCfg)> = vec![(WorldCfg::world_a(), NlCfg::world_a()), (WorldCfg::world_b(), NlCfg::world_b()), (WorldCfg::world_a().noncompute(), NlCfg::world_a())];
    let xc = XCfg::default();
    let obj = |arm: &XArm, h: &Hyper| -> f64 {
        let mut tot = 0.0;
        for (c, nc) in &cfgs {
            let r = run_all(c, nc, &[Any::X(arm.clone())], &seeds, h, &xc, 0);
            tot += mean(&col(&r[0], |s| s.st.correct)) - 0.02 * mean(&col(&r[0], |s| s.st.cost)) / MODEL_COST;
        }
        tot / cfgs.len() as f64
    };
    let mut best_theta = (f64::MIN, 0.6);
    let mut theta_log = Vec::new();
    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let a = XArm::new("rule", XSel::Rule(th), Ver::Indep { meta: false }, Src::Extract, false, false);
        let v = obj(&a, &h0);
        println!("rule theta={th}: objective {v:.4}");
        theta_log.push(json!({"theta": th, "objective": v}));
        if v > best_theta.0 {
            best_theta = (v, th);
        }
    }
    let arm = XArm::new("learned", XSel::Learned { conf: true }, Ver::Indep { meta: false }, Src::Extract, false, false);
    let mut best = (f64::MIN, h0.sel_eps, h0.sel_shrink, h0.sel_opt, h0.kappa, h0.kw_prior);
    let mut grid = Vec::new();
    for sel_eps in [0.02, 0.05] {
        for sel_shrink in [2.0, 4.0] {
            for sel_opt in [0.7, 0.9] {
                for kappa in [0.02, 0.04] {
                    for kw_prior in [0.0, 10.0, 20.0] {
                        let h = Hyper { sel_eps, sel_shrink, sel_opt, kappa, kw_prior, ..h0 };
                        let v = obj(&arm, &h);
                        grid.push(json!({"sel_eps": sel_eps, "sel_shrink": sel_shrink, "sel_opt": sel_opt, "kappa": kappa, "kw_prior": kw_prior, "objective": v}));
                        if v > best.0 {
                            best = (v, sel_eps, sel_shrink, sel_opt, kappa, kw_prior);
                        }
                    }
                }
            }
        }
    }
    println!("best selector: eps={} shrink={} opt={} kappa={} kw_prior={} objective={:.4}", best.1, best.2, best.3, best.4, best.5, best.0);
    println!("best rule theta={}", best_theta.1);
    println!("hx() currently: {:?}", hx());
    std::fs::write(
        g13::arg(args, "--out", "results/g21_tune.json"),
        serde_json::to_string_pretty(&json!({"seeds": seeds, "chosen_selector": {"sel_eps": best.1, "sel_shrink": best.2, "sel_opt": best.3, "kappa": best.4, "kw_prior": best.5, "objective": best.0}, "chosen_theta": best_theta.1,
            "hx_in_code": format!("{:?}", hx()), "theta_log": theta_log, "selector_grid": grid})).unwrap(),
    )
    .unwrap();
}

fn mode_eval(args: &[String]) {
    let seeds = g13::parse_range(args, "--seeds", (31, 60));
    let tag = g13::arg(args, "--tag", "held");
    let (h, theta) = hx();
    let xc = XCfg::default();
    let mut all = serde_json::Map::new();
    for (cfg, ncfg) in world_cfgs(args) {
        println!("\n================ WORLD {} (n={}, unseen phrasing {:.2}, levels {:?}) seeds {}-{} ================", cfg.name, cfg.n, ncfg.unseen_p, ncfg.level_p, seeds[0], seeds[seeds.len() - 1]);
        let mut arms = baseline_arms();
        let nb = arms.len();
        arms.extend(x_arms(theta));
        let res = run_all(&cfg, &ncfg, &arms, &seeds, &h, &xc, 0);
        print_table(&arms, &res);
        let c = |name: &str| col(&res[idx(&arms, name)], |s| s.st.correct);
        let k = |name: &str| col(&res[idx(&arms, name)], |s| s.st.cost);
        let bi = (0..nb).max_by(|a, b| mean(&col(&res[*a], |s| s.st.correct)).partial_cmp(&mean(&col(&res[*b], |s| s.st.correct))).unwrap()).unwrap();
        let (bcor, bcost) = (col(&res[bi], |s| s.st.correct), col(&res[bi], |s| s.st.cost));
        println!("best fair baseline on these seeds: {} ({:.4}) cost {:.2}", arms[bi].name(), mean(&bcor), mean(&bcost));
        let mut comps = serde_json::Map::new();
        for (fam, name) in [("HEAD_indep_vs_baselines", H_INDEP), ("HEAD_shared_vs_baselines", H_SHARED), ("W14_S1_clean_vs_baselines", "W14-S1 clean spec (agent formulates)")] {
            let cs_: Vec<Cmp> = (0..nb).map(|b| cmp(&c(name), &col(&res[b], |s| s.st.correct))).collect();
            let adj = holm_adjust(&cs_.iter().map(|x| x.p).collect::<Vec<_>>());
            println!("-- {fam}: correctness diff ({name} - baseline), Holm over {nb}");
            let mut rows = Vec::new();
            for b in 0..nb {
                println!("   vs {:<18} {}", arms[b].name(), cs(&cs_[b], Some(adj[b])));
                rows.push(json!({"baseline": arms[b].name(), "diff": cj(&cs_[b], Some(adj[b]))}));
            }
            println!("   max Holm p over baselines: {:.2e}", adj.iter().cloned().fold(0.0, f64::max));
            comps.insert(fam.into(), json!(rows));
        }
        let mut single = |key: &str, label: &str, a: Vec<f64>, b: Vec<f64>| {
            let x = cmp(&a, &b);
            println!("-- {label}: {}", cs(&x, None));
            comps.insert(key.into(), cj(&x, None));
        };
        single("HEAD_indep_vs_best_correct", "HEADLINE (extractor + learned(conf) + independent verifier) - best baseline  (correctness)", c(H_INDEP), bcor.clone());
        single("HEAD_indep_vs_best_cost", "HEADLINE - best baseline  (cost/task)", k(H_INDEP), bcost.clone());
        single("HEAD_shared_vs_best_correct", "extractor + learned(conf) + SHARED verifier - best baseline", c(H_SHARED), bcor.clone());
        single("W14S1_vs_best_correct", "W14 S1 (clean spec, agent formulates) - best baseline  [what W14 reported]", c("W14-S1 clean spec (agent formulates)"), bcor.clone());
        single("HEAD_vs_W14S1_correct", "HEADLINE - W14 S1  (what the extractor + independent verifier cost vs W14)", c(H_INDEP), c("W14-S1 clean spec (agent formulates)"));
        single("HEAD_vs_Xclean_correct", "HEADLINE - X-clean-spec (cost of extractor errors vs a clean deterministic spec)", c(H_INDEP), c("X-clean-spec+shared (no extractor errors)"));
        single("indep_vs_shared_correct", "independent - shared verifier (clean evaluator, no bugs)", c(H_INDEP), c(H_SHARED));
        single("indep_vs_shared_cost", "independent - shared verifier (cost/task)", k(H_INDEP), k(H_SHARED));
        single("conf_vs_noconf_correct", "selector with extractor confidence - without", c(H_SHARED), c("X-learned(noconf)+shared"));
        single("learned_vs_rule_correct", "learned(conf) - confidence rule (theta), both independent verifier", c(H_INDEP), c("X-rule(theta)+indep"));
        single("learned_vs_static_correct", "learned(conf)+shared - static keyword rule+shared", c(H_SHARED), c("X-static+shared"));
        single("verify_value_correct", "verification value: learned(conf)+indep - noverify", c(H_INDEP), c("X-learned(conf)+noverify"));
        single("meta_vs_indep_correct", "indep+meta - indep", c("X-learned(conf)+indep+meta"), c(H_INDEP));
        single("meta_vs_indep_cost", "indep+meta - indep (cost/task)", k("X-learned(conf)+indep+meta"), k(H_INDEP));
        single("cache_vs_indep_cost", "cache: indep+cache - indep (cost/task)", k("X-learned(conf)+indep+cache"), k(H_INDEP));
        single("cache_vs_indep_correct", "cache: indep+cache - indep (correctness; must be 0)", c("X-learned(conf)+indep+cache"), c(H_INDEP));
        single("memo_vs_cache_cost", "memo: +memo - cache only (cost/task)", k("X-learned(conf)+indep+cache+memo"), k("X-learned(conf)+indep+cache"));
        single("memo_vs_cache_correct", "memo: +memo - cache only (correctness)", c("X-learned(conf)+indep+cache+memo"), c("X-learned(conf)+indep+cache"));
        single("HEADcache_vs_best_correct", "HEADLINE+cache - best baseline", c("X-learned(conf)+indep+cache"), bcor.clone());
        single("HEADcache_vs_best_cost", "HEADLINE+cache - best baseline (cost)", k("X-learned(conf)+indep+cache"), bcost.clone());
        single("HEAD_vs_vote3_correct", "HEADLINE - vote3 (correctness; vote3 costs 30/task)", c(H_INDEP), c("vote3(plain, 3x cost)"));
        let sel = &res[idx(&arms, H_INDEP)];
        println!(
            "-- headline arm: no tool on {:.3} of compute tasks (abstain/none), tool answer delivered right {:.3} / silently wrong {:.3}, fallback after fail/reject {:.3}, extractor cost/task {:.2}, verifier cost/task {:.2}, rejects/task {:.3}",
            mean(&col(sel, |s| s.no_tool)), mean(&col(sel, |s| s.tool_right)), mean(&col(sel, |s| s.silent_wrong)), mean(&col(sel, |s| s.fallback)), mean(&col(sel, |s| s.ex_cost)), mean(&col(sel, |s| s.ver_cost)), mean(&col(sel, |s| s.rejects))
        );
        let sc = &res[idx(&arms, "X-learned(conf)+indep+cache")];
        println!("-- cache: hit rate {:.3}, cost saved/task {:.3} of {:.3} nominal/task", mean(&col(sc, |s| s.st.cache_hit)), mean(&col(sc, |s| s.st.cache_saved)), mean(&col(sc, |s| s.st.cost_nominal)));
        // survival of the W14 lift
        let w14 = mean(&c("W14-S1 clean spec (agent formulates)")) - mean(&bcor);
        let head = mean(&c(H_INDEP)) - mean(&bcor);
        let head_sh = mean(&c(H_SHARED)) - mean(&bcor);
        println!("-- LIFT SURVIVAL: W14 S1 lift {:+.4}; extractor+independent {:+.4} ({:.1}% of W14); extractor+shared {:+.4} ({:.1}% of W14)", w14, head, 100.0 * head / w14, head_sh, 100.0 * head_sh / w14);
        comps.insert("survival".into(), json!({"w14_lift": w14, "head_indep_lift": head, "head_shared_lift": head_sh, "share_indep": head / w14, "share_shared": head_sh / w14}));
        let mut armj = serde_json::Map::new();
        for (a, rs) in arms.iter().zip(&res) {
            armj.insert(a.name().to_string(), arm_json(rs));
        }
        all.insert(cfg.name.to_string(), json!({"config": {"n": cfg.n, "qv_w14": cfg.qv, "misuse": cfg.misuse, "dup": cfg.dup, "unseen_p": ncfg.unseen_p, "level_p": ncfg.level_p, "ex_cost": xc.ex_cost, "theta": theta}, "best_baseline": arms[bi].name(), "arms": armj, "comparisons": comps}));
    }
    let out = g13::arg(args, "--out", &format!("results/g21_{tag}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"tag": tag, "seeds": seeds, "hyper": format!("{:?}", hx()), "worlds": all})).unwrap()).unwrap();
    println!("\nwrote {out}");
}

fn mode_controls(args: &[String]) {
    let seeds = g13::parse_range(args, "--seeds", (31, 60));
    let tag = g13::arg(args, "--tag", "held");
    let (h, theta) = hx();
    let xc = XCfg::default();
    let mut all = serde_json::Map::new();
    for (base, ncfg) in world_cfgs(args) {
        let cfg = base.clone().noncompute();
        println!("\n================ CONTROL no-compute (all prose) on generator {} seeds {}-{} ================", base.name, seeds[0], seeds[seeds.len() - 1]);
        let mut arms: Vec<Any> = vec![Any::G(ArmSpec::plain(Rt::Safe, "guarded-safe")), Any::G(ArmSpec::plain(Rt::HebR, "hebbian-robust"))];
        arms.extend(x_arms(theta));
        arms.push(Any::X(XArm::new("X-always-extracted-tool+indep (no selection)", XSel::Rule(0.0), Ver::Indep { meta: false }, Src::Extract, false, false)));
        let res = run_all(&cfg, &ncfg, &arms, &seeds, &h, &xc, 0);
        print_table(&arms, &res);
        let plain = idx(&arms, "guarded-safe");
        let (pc, pk) = (col(&res[plain], |s| s.st.correct), col(&res[plain], |s| s.st.cost));
        let mut rows = serde_json::Map::new();
        for name in [H_INDEP, H_SHARED, "X-static+shared", "X-rule(theta)+indep", "X-learned(conf)+noverify", "X-learned(noconf)+shared", "X-always-extracted-tool+indep (no selection)", "W14-S1 clean spec (agent formulates)"] {
            let i = idx(&arms, name);
            let dc = cmp(&col(&res[i], |s| s.st.correct), &pc);
            let dk = cmp(&col(&res[i], |s| s.st.cost), &pk);
            let within = dc.lo > -0.01 && dc.hi < 0.01;
            println!("   {name:<46} correct vs plain: {}   cost vs plain: {}   TOST +-0.01: {}", cs(&dc, None), cs(&dk, None), within);
            rows.insert(name.into(), json!({"correct_vs_plain": cj(&dc, None), "cost_vs_plain": cj(&dk, None), "tost_within_1pp": within, "skip_qual": mean(&col(&res[i], |s| s.st.skip_qual))}));
        }
        let mut armj = serde_json::Map::new();
        for (a, rs) in arms.iter().zip(&res) {
            armj.insert(a.name().to_string(), arm_json(rs));
        }
        all.insert(base.name.to_string(), json!({"comparisons": rows, "arms": armj}));
    }
    let out = g13::arg(args, "--out", &format!("results/g21_controls_{tag}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"tag": tag, "seeds": seeds, "controls": all})).unwrap()).unwrap();
    println!("\nwrote {out}");
}

fn mode_bugs(args: &[String]) {
    let seeds = g13::parse_range(args, "--seeds", (31, 60));
    let tag = g13::arg(args, "--tag", "held");
    let (h, _theta) = hx();
    let xc = XCfg::default();
    let scenarios: [u8; 9] = [0, 3, 5, 7, 9, 12, 18, 20, 21];
    let mut all = serde_json::Map::new();
    for (cfg, ncfg) in world_cfgs(args) {
        println!("\n================ BUG-INJECTED DEPLOYMENT on generator {} seeds {}-{} ================", cfg.name, seeds[0], seeds[seeds.len() - 1]);
        let arms: Vec<Any> = vec![
            Any::G(ArmSpec::plain(Rt::Safe, "guarded-safe")),
            Any::X(XArm::new("learned(conf)+noverify", XSel::Learned { conf: true }, Ver::None, Src::Extract, false, false)),
            Any::X(XArm::new("learned(conf)+shared", XSel::Learned { conf: true }, Ver::Shared, Src::Extract, false, false)),
            Any::X(XArm::new("learned(conf)+indep", XSel::Learned { conf: true }, Ver::Indep { meta: false }, Src::Extract, false, false)),
            Any::X(XArm::new("learned(conf)+indep+meta", XSel::Learned { conf: true }, Ver::Indep { meta: true }, Src::Extract, false, false)),
            Any::X(XArm::new("CLEAN-spec noverify", XSel::Learned { conf: false }, Ver::None, Src::Clean, false, false)),
            Any::X(XArm::new("CLEAN-spec shared", XSel::Learned { conf: false }, Ver::Shared, Src::Clean, false, false)),
            Any::X(XArm::new("CLEAN-spec indep", XSel::Learned { conf: false }, Ver::Indep { meta: false }, Src::Clean, false, false)),
        ];
        println!("EXTRACTED spec arms | then CLEAN (canonical, minimally parenthesised) spec arms");
        println!("{:<26} {:>7} | {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} | silentW(ext) noV/shr/ind | indep-shared (extracted)", "scenario", "plain", "noverify", "shared", "indep", "ind+meta", "C-noV", "C-shr", "C-ind");
        let mut rows = Vec::new();
        for m in scenarios {
            let name = if m == 0 { "no bug (clean evaluator)".to_string() } else { evalx_name(m) };
            let res = run_all(&cfg, &ncfg, &arms, &seeds, &h, &xc, m);
            let c = |i: usize| col(&res[i], |s| s.st.correct);
            let d = cmp(&c(3), &c(2));
            let dp = cmp(&c(3), &c(0));
            println!(
                "{:<26} {:>7.4} | {:>7.4} {:>7.4} {:>7.4} {:>7.4} | {:>7.4} {:>7.4} {:>7.4} | {:.3}/{:.3}/{:.3} | {}   indep-vs-plain {:+.4} [{:+.4},{:+.4}]",
                name,
                mean(&c(0)),
                mean(&c(1)),
                mean(&c(2)),
                mean(&c(3)),
                mean(&c(4)),
                mean(&c(5)),
                mean(&c(6)),
                mean(&c(7)),
                mean(&col(&res[1], |s| s.silent_wrong)),
                mean(&col(&res[2], |s| s.silent_wrong)),
                mean(&col(&res[3], |s| s.silent_wrong)),
                cs(&d, None),
                dp.mean,
                dp.lo,
                dp.hi
            );
            rows.push(json!({"mutant": m, "name": name, "plain": mean(&c(0)), "noverify": mean(&c(1)), "shared": mean(&c(2)), "indep": mean(&c(3)), "indep_meta": mean(&c(4)), "clean_noverify": mean(&c(5)), "clean_shared": mean(&c(6)), "clean_indep": mean(&c(7)), "clean_silent_wrong": [mean(&col(&res[5], |s| s.silent_wrong)), mean(&col(&res[6], |s| s.silent_wrong)), mean(&col(&res[7], |s| s.silent_wrong))],
                "silent_wrong": [mean(&col(&res[1], |s| s.silent_wrong)), mean(&col(&res[2], |s| s.silent_wrong)), mean(&col(&res[3], |s| s.silent_wrong)), mean(&col(&res[4], |s| s.silent_wrong))],
                "cost": [mean(&col(&res[0], |s| s.st.cost)), mean(&col(&res[1], |s| s.st.cost)), mean(&col(&res[2], |s| s.st.cost)), mean(&col(&res[3], |s| s.st.cost)), mean(&col(&res[4], |s| s.st.cost))],
                "indep_minus_shared": cj(&d, None), "indep_minus_plain": cj(&dp, None)}));
        }
        all.insert(cfg.name.to_string(), json!(rows));
    }
    let out = g13::arg(args, "--out", &format!("results/g21_bugs_{tag}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"tag": tag, "seeds": seeds, "scenarios": all})).unwrap()).unwrap();
    println!("\nwrote {out}");
}

fn evalx_name(m: u8) -> String {
    g21_common::evalx::MUTANTS.iter().find(|x| x.0 == m).map(|x| format!("{}:{}", x.0, x.1)).unwrap_or_else(|| format!("{m}"))
}

fn mode_sens(args: &[String]) {
    let seeds = g13::parse_range(args, "--seeds", (31, 60));
    let tag = g13::arg(args, "--tag", "held");
    let (h, theta) = hx();
    let mut rows = Vec::new();
    for (cfg, ncfg) in world_cfgs(args) {
        println!("\n================ SENSITIVITY on generator {} seeds {}-{} ================", cfg.name, seeds[0], seeds[seeds.len() - 1]);
        let mut settings: Vec<(String, NlCfg, XCfg)> = Vec::new();
        settings.push(("default".into(), ncfg.clone(), XCfg::default()));
        for up in [0.0, 0.6, 1.0] {
            let mut n2 = ncfg.clone();
            n2.unseen_p = up;
            settings.push((format!("unseen-phrasing share {up}"), n2, XCfg::default()));
        }
        // all phrasing at L0/L1 only vs L3-heavy
        let mut easy = ncfg.clone();
        easy.level_p = [0.6, 0.4, 0.0, 0.0, 0.0];
        easy.unseen_p = 0.0;
        settings.push(("easy phrasing only (L0/L1, dev bank)".into(), easy, XCfg::default()));
        let mut hard = ncfg.clone();
        hard.level_p = [0.0, 0.1, 0.2, 0.5, 0.2];
        settings.push(("hard phrasing (L3/L4-heavy)".into(), hard, XCfg::default()));
        for ec in [0.0, 1.0, 3.0] {
            settings.push((format!("extractor cost {ec}"), ncfg.clone(), XCfg { ex_cost: ec, ..XCfg::default() }));
        }
        for (label, nc, xc) in settings {
            let mut arms = baseline_arms();
            let nb = arms.len();
            arms.push(Any::G(ArmSpec::tools("W14-S1", Rt::Safe, SelPol::Learned, true, false, false)));
            arms.push(Any::X(XArm::new("X-clean", XSel::Learned { conf: false }, Ver::Shared, Src::Clean, false, false)));
            arms.push(Any::X(XArm::new("X-shared", XSel::Learned { conf: true }, Ver::Shared, Src::Extract, false, false)));
            arms.push(Any::X(XArm::new("X-indep", XSel::Learned { conf: true }, Ver::Indep { meta: false }, Src::Extract, false, false)));
            arms.push(Any::X(XArm::new("X-rule-indep", XSel::Rule(theta), Ver::Indep { meta: false }, Src::Extract, false, false)));
            let res = run_all(&cfg, &nc, &arms, &seeds, &h, &xc, 0);
            let bi = (0..nb).max_by(|a, b| mean(&col(&res[*a], |s| s.st.correct)).partial_cmp(&mean(&col(&res[*b], |s| s.st.correct))).unwrap()).unwrap();
            let bc = col(&res[bi], |s| s.st.correct);
            let c = |n: &str| col(&res[idx(&arms, n)], |s| s.st.correct);
            let x = cmp(&c("X-indep"), &bc);
            let xs = cmp(&c("X-shared"), &bc);
            let w = cmp(&c("W14-S1"), &bc);
            let xr = cmp(&c("X-rule-indep"), &bc);
            let xcl = cmp(&c("X-clean"), &bc);
            let cost = mean(&col(&res[idx(&arms, "X-indep")], |s| s.st.cost));
            println!(
                "{:<38} best={:<14} W14 {:+.4} | X-clean {:+.4} | X-shared {:+.4} | X-indep {} | X-rule-indep {:+.4} | cost {:.2}",
                label,
                arms[bi].name(),
                w.mean,
                xcl.mean,
                xs.mean,
                cs(&x, None),
                xr.mean,
                cost
            );
            rows.push(json!({"world": cfg.name, "setting": label, "best_baseline": arms[bi].name(), "best_baseline_correct": mean(&bc), "W14_S1": cj(&w, None), "X_clean": cj(&xcl, None), "X_shared": cj(&xs, None), "X_indep": cj(&x, None), "X_rule_indep": cj(&xr, None), "X_indep_cost": cost}));
        }
    }
    let out = g13::arg(args, "--out", &format!("results/g21_sens_{tag}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"tag": tag, "seeds": seeds, "rows": rows})).unwrap()).unwrap();
    println!("\nwrote {out}");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("tune") => mode_tune(&args),
        Some("eval") => mode_eval(&args),
        Some("controls") => mode_controls(&args),
        Some("bugs") => mode_bugs(&args),
        Some("sens") => mode_sens(&args),
        _ => {
            eprintln!("usage: g21_headline tune|eval|controls|bugs|sens [--seeds a-b] [--tag t] [--out path] [--worlds A,B] [--n 600]");
            std::process::exit(2);
        }
    }
}
