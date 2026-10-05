//! h18_lift: CORRECTNESS LIFT of domain tools + verification + extraction in seeded math-heavy worlds (finance / business /
//! science / no-math prose), vs plain routing and the best fair baseline {keyword, eps-greedy, pooled-thompson,
//! hebbian-robust, warm(hebr), guarded-safe}. Modes (all deterministic):
//!   tune     --seeds 1-8                 theta + selector hyper-parameters (frozen in `hx()` afterwards)
//!   eval     --seeds 31-60 --tag held    main study, worlds A and B (two independent generators)
//!   controls --seeds 31-60 --tag held    no-math worlds: tools must be skipped (TOST +-0.01)
//!   bugs     --seeds 31-60 --tag held    seeded evaluator bugs active in the deployed tool, shared vs independent verifier
//!   sens     --seeds 31-60 --tag held    err_scale (ungated error assumption), unseen-phrasing share, text degradation
//! Protocol: tune seeds 1-8, held 31-60, confirm 61-90 (frozen config), paired 95% t-CI, Holm over the six plain baselines.
//! All numbers are SYNTHETIC until validated on real traffic; the ungated-model error curve is an ASSUMPTION (see world.rs).
mod h18_common;
use h18_common::world::*;
use h18_common::{arg, parse_range};
use neural_mesh::fin::bug as fbug;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

/// FROZEN after `tune --seeds 1-8` (results/h18_tune.json)
fn hx() -> Hyper {
    Hyper { eps_g: 1.0, eps_e: 0.1, pts_g: 1.0, warm_k: 10.0, sel_eps: 0.02, sel_shrink: 4.0, sel_opt: 0.7, kappa: 0.08, kw_prior: 10.0, theta: 0.7, theta_lo: 0.4, fit_w: 0.25 }
}

fn mean(xs: &[f64]) -> f64 {
    let v: Vec<f64> = xs.iter().cloned().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
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
fn col(rs: &[RunStats], f: impl Fn(&RunStats) -> f64) -> Vec<f64> {
    rs.iter().map(f).collect()
}

fn worlds(cfg: &WCfg, seeds: &[u64]) -> Vec<World> {
    seeds.par_iter().map(|s| gen_world(cfg, *s)).collect()
}
fn run(ws: &[World], seeds: &[u64], arm: &Arm, h: &Hyper, bug: u8) -> Vec<RunStats> {
    ws.par_iter().zip(seeds.par_iter()).map(|(w, s)| run_arm(w, arm, h, *s, bug)).collect()
}

fn arms_main() -> Vec<Arm> {
    vec![
        Arm::plain(Rt::Safe, "plain-guarded-safe(ref)"),
        Arm { name: "vote3".into(), router: Rt::Safe, mode: Mode::Vote3 },
        Arm { name: "V-only(indep verifier on direct answers)".into(), router: Rt::Safe, mode: Mode::VOnly { indep: true } },
        Arm::tool("XC clean-spec ceiling (canonical spec, tool)", Src::Canonical, Sel::Always, Ver::None, false, false),
        Arm::tool("T tool(extracted spec), no verifier", Src::Extracted, Sel::Always, Ver::None, false, false),
        Arm::tool("TV-shared (T + re-run verifier)", Src::Extracted, Sel::Always, Ver::Shared, false, false),
        Arm::tool("TV-indep (T + independent verifier)", Src::Extracted, Sel::Always, Ver::Indep, false, false),
        Arm::tool("R rule(conf>=theta)+indep", Src::Extracted, Sel::Rule, Ver::Indep, false, false),
        Arm::tool("H learned selection+indep (headline)", Src::Extracted, Sel::Learned, Ver::Indep, false, false),
        Arm { name: "S tiered (post-hoc): conf>=hi tool+indep | lo..hi V-only | else plain".into(), router: Rt::Safe, mode: Mode::Tier },
        Arm::tool("TX tool + model cross-check", Src::Extracted, Sel::Always, Ver::Indep, true, false),
        Arm::tool("H+cache", Src::Extracted, Sel::Learned, Ver::Indep, false, true),
    ]
}

fn plain_baselines(ws: &[World], seeds: &[u64], h: &Hyper) -> Vec<(String, Vec<RunStats>)> {
    ROUTERS.iter().map(|(rt, n)| (n.to_string(), run(ws, seeds, &Arm::plain(*rt, n), h, 0))).collect()
}
fn best_idx(pl: &[(String, Vec<RunStats>)]) -> usize {
    let mut b = 0;
    for i in 1..pl.len() {
        if mean(&col(&pl[i].1, |r| r.correct)) > mean(&col(&pl[b].1, |r| r.correct)) {
            b = i;
        }
    }
    b
}

fn eval_mode(seeds: &[u64], tag: &str, h: &Hyper) -> Value {
    let mut out = serde_json::Map::new();
    for cfg in [WCfg::world_a(), WCfg::world_b()] {
        let ws = worlds(&cfg, seeds);
        let pl = plain_baselines(&ws, seeds, h);
        let bi = best_idx(&pl);
        let base = col(&pl[bi].1, |r| r.correct);
        let base_cost = mean(&col(&pl[bi].1, |r| r.cost));
        println!("\n=== world {} ({}; seeds {}-{}, {} tasks/seed) ===", cfg.name, tag, seeds[0], seeds[seeds.len() - 1], cfg.n);
        for (n, r) in &pl {
            println!("  plain {:<16} correct {}  cost {:.2}", n, ms(&col(r, |x| x.correct)), mean(&col(r, |x| x.cost)));
        }
        println!("  best fair baseline (chosen on the evaluated seeds): {}  correct {:.4}", pl[bi].0, mean(&base));
        // ungated assumption sanity: plain accuracy by difficulty tercile and by steps bucket (all tasks, guarded-safe direct = agent mean)
        let mut by_d = [(0.0, 0usize); 3];
        let mut by_s = [(0.0, 0usize); 3];
        for w in &ws {
            for t in w.tasks.iter().filter(|t| t.dom != 3) {
                let pm = (0..NA).map(|a| w.p_dir(a, t)).sum::<f64>() / NA as f64;
                let di = ((t.d * 3.0) as usize).min(2);
                by_d[di].0 += pm;
                by_d[di].1 += 1;
                let si = if t.steps <= 3 { 0 } else if t.steps <= 6 { 1 } else { 2 };
                by_s[si].0 += pm;
                by_s[si].1 += 1;
            }
        }
        println!(
            "  ungated model P(correct), mean over agents: by difficulty tercile {:.3}/{:.3}/{:.3}; by steps (<=3 / 4-6 / >=7) {:.3}/{:.3}/{:.3}  (ASSUMPTION: error rises with difficulty and steps)",
            by_d[0].0 / by_d[0].1.max(1) as f64,
            by_d[1].0 / by_d[1].1.max(1) as f64,
            by_d[2].0 / by_d[2].1.max(1) as f64,
            by_s[0].0 / by_s[0].1.max(1) as f64,
            by_s[1].0 / by_s[1].1.max(1) as f64,
            by_s[2].0 / by_s[2].1.max(1) as f64
        );
        let mut wj = serde_json::Map::new();
        wj.insert("plain".into(), json!(pl.iter().map(|(n, r)| json!({"name": n, "correct": mean(&col(r, |x| x.correct)), "cost": mean(&col(r, |x| x.cost))})).collect::<Vec<_>>()));
        wj.insert("best_baseline".into(), json!(pl[bi].0));
        let mut ar_runs: Vec<(String, Vec<RunStats>)> = Vec::new();
        for arm in arms_main() {
            let r = run(&ws, seeds, &arm, h, 0);
            ar_runs.push((arm.name.clone(), r));
        }
        println!("  {:<46} {:>8} {:>26} {:>8} {:>8}  {}", "arm", "correct", "diff vs best baseline (CI) w/l/t", "cost", "toolrate", "max Holm p over the 6 plain baselines");
        for (n, r) in &ar_runs {
            let c = cmp(&col(r, |x| x.correct), &base);
            let pv: Vec<f64> = pl.iter().map(|(_, b)| paired_t_pvalue(&col(r, |x| x.correct), &col(b, |x| x.correct))).collect();
            let mh = holm_adjust(&pv).iter().cloned().fold(0.0, f64::max);
            println!("  {:<46} {:>8.4} {}  {:>6.2} {:>7.3}   {:.2e}", n, mean(&col(r, |x| x.correct)), cs(&c, None), mean(&col(r, |x| x.cost)), mean(&col(r, |x| x.tool_rate_math)), mh);
            wj.insert(format!("arm:{n}"), json!({"correct": mean(&col(r, |x| x.correct)), "vs_best": cj(&c, None), "cost": mean(&col(r, |x| x.cost)), "tool_rate_math": mean(&col(r, |x| x.tool_rate_math)), "tool_rate_prose": mean(&col(r, |x| x.tool_rate_prose)), "silent_wrong": mean(&col(r, |x| x.silent_wrong)), "right_family": mean(&col(r, |x| x.right_family)), "ver_reject": mean(&col(r, |x| x.ver_reject)), "cache_hit": mean(&col(r, |x| x.cache_hit)), "max_holm_p": mh}));
        }
        let get = |name: &str| ar_runs.iter().find(|(n, _)| n.starts_with(name)).map(|(_, r)| r.clone()).unwrap_or_default();
        let hr = get("H learned");
        // Holm over the six plain baselines for the headline arm
        let hv = col(&hr, |x| x.correct);
        let ps: Vec<f64> = pl.iter().map(|(_, r)| paired_t_pvalue(&hv, &col(r, |x| x.correct))).collect();
        let adj = holm_adjust(&ps);
        let maxp = adj.iter().cloned().fold(0.0, f64::max);
        let c_h = cmp(&hv, &base);
        println!("  HEADLINE H - best baseline ({}): {}   max Holm p over the 6 baselines {:.2e}; cost/task {:.2} vs plain {:.2} ({:.2}x)", pl[bi].0, cs(&c_h, Some(maxp)), maxp, mean(&col(&hr, |x| x.cost)), base_cost, mean(&col(&hr, |x| x.cost)) / base_cost);
        wj.insert("headline".into(), json!({"vs_best": cj(&c_h, Some(maxp)), "max_holm_p": maxp, "cost_ratio": mean(&col(&hr, |x| x.cost)) / base_cost}));
        // decomposition
        let t_noV = get("T tool");
        let tv_i = get("TV-indep");
        let tv_s = get("TV-shared");
        let xc = get("XC clean");
        let vonly = get("V-only");
        let rule = get("R rule");
        let tx = get("TX");
        let tier = get("S tiered");
        let hc = get("H+cache");
        let a = |x: &[RunStats]| col(x, |r| r.correct);
        let pairs: Vec<(&str, Cmp)> = vec![
            ("offload: T - plain(best)", cmp(&a(&t_noV), &base)),
            ("verification of tool outputs (clean evaluator): TV-indep - T", cmp(&a(&tv_i), &a(&t_noV))),
            ("independent vs shared verifier (clean evaluator): TV-indep - TV-shared", cmp(&a(&tv_i), &a(&tv_s))),
            ("verification of direct answers: V-only - plain(best)", cmp(&a(&vonly), &base)),
            ("cost of extraction errors: T - XC(clean spec)", cmp(&a(&t_noV), &a(&xc))),
            ("learned selection vs rule: H - R(theta)", cmp(&hv, &a(&rule))),
            ("learned selection vs always-tool: H - TV-indep", cmp(&hv, &a(&tv_i))),
            ("model cross-check: TX - TV-indep", cmp(&a(&tx), &a(&tv_i))),
            ("cache (identical correctness): H+cache - H", cmp(&a(&hc), &hv)),
            ("H - vote3", cmp(&hv, &a(&get("vote3")))),
            ("POST-HOC tiered: S - best baseline", cmp(&a(&tier), &base)),
            ("POST-HOC tiered vs cost-reference: S - vote3 (cost 30)", cmp(&a(&tier), &a(&get("vote3")))),
            ("POST-HOC tiered vs headline: S - H", cmp(&a(&tier), &hv)),
            ("POST-HOC tiered vs V-only: S - V-only", cmp(&a(&tier), &a(&vonly))),
        ];
        println!("  decomposition (correctness diff [95% CI] w/l/t over seeds):");
        let mut dj = serde_json::Map::new();
        for (n, c) in &pairs {
            println!("    {:<78} {}", n, cs(c, None));
            dj.insert(n.to_string(), cj(c, None));
        }
        println!("    cost/task: plain {:.2} | T {:.2} | TV-indep {:.2} | H {:.2} | H+cache {:.2} | TX {:.2} | vote3 {:.2}", base_cost, mean(&col(&t_noV, |r| r.cost)), mean(&col(&tv_i, |r| r.cost)), mean(&col(&hr, |r| r.cost)), mean(&col(&hc, |r| r.cost)), mean(&col(&tx, |r| r.cost)), mean(&col(&get("vote3"), |r| r.cost)));
        println!("    cost/task: R {:.2} | S tiered {:.2} | V-only {:.2} | XC {:.2}", mean(&col(&rule, |r| r.cost)), mean(&col(&tier, |r| r.cost)), mean(&col(&vonly, |r| r.cost)), mean(&col(&xc, |r| r.cost)));
        wj.insert("decomposition".into(), Value::Object(dj));
        // per domain
        println!("  per-domain correctness (fin / biz / sci / prose): best baseline vs H, R (rule), S (tiered, post-hoc)");
        let dm = |rs: &[RunStats], g: usize| mean(&col(rs, |r| r.correct_dom[g]));
        for (g, nm) in ["finance", "business", "science", "prose"].iter().enumerate() {
            for (an, ar) in [("H", &hr), ("R", &rule), ("S", &tier)] {
                let c = cmp(&col(ar, |r| r.correct_dom[g]), &col(&pl[bi].1, |r| r.correct_dom[g]));
                println!("    {:<9} baseline {:.4}  {} {:.4}  diff {}", nm, dm(&pl[bi].1, g), an, dm(ar, g), cs(&c, None));
            }
        }
        println!("  learned tool selection (H): tool used on {:.3} of math tasks, {:.3} of prose tasks; right op+tool among tool-used {:.3}; independent verifier rejection rate {:.4}; silent wrong (tool answer accepted but wrong) {:.4} of math tasks (R {:.4}, S {:.4})", mean(&col(&hr, |r| r.tool_rate_math)), mean(&col(&hr, |r| r.tool_rate_prose)), mean(&col(&hr, |r| r.right_family)), mean(&col(&hr, |r| r.ver_reject)), mean(&col(&hr, |r| r.silent_wrong)), mean(&col(&rule, |r| r.silent_wrong)), mean(&col(&tier, |r| r.silent_wrong)));
        // by phrasing bank / level: lift of H over plain per task group, pooled over seeds (descriptive, per-task)
        let mut grp: std::collections::BTreeMap<String, (f64, f64, usize)> = Default::default();
        for (wi, w) in ws.iter().enumerate() {
            for (ti, t) in w.tasks.iter().enumerate() {
                if t.dom == 3 {
                    continue;
                }
                let hb = hr[wi].correct_vec[ti] as f64;
                let bb = pl[bi].1[wi].correct_vec[ti] as f64;
                for k in [format!("bank:{}", if t.unseen { "unseen" } else { "seen" }), format!("level:L{}", t.level), format!("steps:{}", if t.steps <= 3 { "<=3" } else if t.steps <= 6 { "4-6" } else { ">=7" })] {
                    let e = grp.entry(k).or_insert((0.0, 0.0, 0));
                    e.0 += hb;
                    e.1 += bb;
                    e.2 += 1;
                }
            }
        }
        println!("  H vs best baseline by task group (pooled, descriptive; n = tasks):");
        for (k, (h1, b1, n)) in &grp {
            println!("    {:<14} n={:<6} baseline {:.3}  H {:.3}  lift {:+.3}", k, n, b1 / *n as f64, h1 / *n as f64, (h1 - b1) / *n as f64);
        }
        wj.insert("groups".into(), json!(grp.iter().map(|(k, (h1, b1, n))| json!({"group": k, "n": n, "baseline": b1 / *n as f64, "H": h1 / *n as f64})).collect::<Vec<_>>()));
        out.insert(cfg.name.to_string(), Value::Object(wj));
    }
    Value::Object(out)
}

fn tost(c: &Cmp) -> bool {
    c.lo > -0.01 && c.hi < 0.01
}

fn controls_mode(seeds: &[u64], h: &Hyper) -> Value {
    let mut out = serde_json::Map::new();
    println!("\n=== CONTROLS: all tasks are no-math prose (incl. hard negatives such as 'explain why 12 plus 7 ...'): tools must be skipped; TOST +-0.01 vs plain guarded-safe ===");
    for base_cfg in [WCfg::world_a(), WCfg::world_b()] {
        let cfg = base_cfg.prose_only();
        let ws = worlds(&cfg, seeds);
        let plain = run(&ws, seeds, &Arm::plain(Rt::Safe, "plain"), h, 0);
        let pc = col(&plain, |r| r.correct);
        let mut wj = serde_json::Map::new();
        for arm in [
            Arm::tool("H learned+indep", Src::Extracted, Sel::Learned, Ver::Indep, false, false),
            Arm::tool("R rule(theta)+indep", Src::Extracted, Sel::Rule, Ver::Indep, false, false),
            Arm::tool("T always (tool whenever the extractor emits a spec)", Src::Extracted, Sel::Always, Ver::Indep, false, false),
        ] {
            let r = run(&ws, seeds, &arm, h, 0);
            let c = cmp(&col(&r, |x| x.correct), &pc);
            let cc = cmp(&col(&r, |x| x.cost), &col(&plain, |x| x.cost));
            println!("  world {} {:<52} correct diff {}  TOST(+-0.01) {}  cost diff {:+.2}  tool used on {:.3} of prose tasks", cfg.name, arm.name, cs(&c, None), tost(&c), cc.mean, mean(&col(&r, |x| x.tool_rate_prose)));
            wj.insert(arm.name.clone(), json!({"diff": cj(&c, None), "tost": tost(&c), "cost_diff": cc.mean, "tool_rate_prose": mean(&col(&r, |x| x.tool_rate_prose))}));
        }
        out.insert(cfg.name.to_string(), Value::Object(wj));
    }
    Value::Object(out)
}

fn bugs_mode(seeds: &[u64], h: &Hyper) -> Value {
    let mut out = serde_json::Map::new();
    println!("\n=== BUG-INJECTED END TO END: one seeded evaluator bug active in the deployed tool; verifier shared (re-run) vs independent; held worlds ===");
    let list: [(u8, &str); 11] = [
        (fbug::NPV_T_OFF1, "NPV_t_off1"),
        (fbug::COMP_SIMPLE, "COMP_simple_interest"),
        (fbug::LOAN_OFF1, "LOAN_pmt_off1"),
        (fbug::BOND_FREQ, "BOND_cpn_freq_ignored"),
        (fbug::CAGR_N_OFF1, "CAGR_n_off1"),
        (fbug::VAR_POP, "VAR_population"),
        (fbug::CI_Z, "CI_z_not_t"),
        (fbug::KIN_NO_HALF, "KIN_s_no_half"),
        (fbug::UNIT_PSI, "UNIT_psi_factor"),
        (fbug::CM_PCT_BASE, "CM_pct_base (common-mode)"),
        (fbug::CM_MASS_O, "CM_mass_O (common-mode)"),
    ];
    for cfg in [WCfg::world_a(), WCfg::world_b()] {
        let ws = worlds(&cfg, seeds);
        let clean = run(&ws, seeds, &Arm::tool("T-noV", Src::Extracted, Sel::Always, Ver::None, false, false), h, 0);
        println!("  world {} (no bug: T-noV correct {:.4})", cfg.name, mean(&col(&clean, |r| r.correct)));
        println!("    {:<28} {:>9} {:>9} {:>9}   {:>34} {:>9}", "bug", "noV", "shared", "indep", "indep - shared (CI)", "indep-noV");
        let mut wj = Vec::new();
        for (b, name) in list {
            let nov = run(&ws, seeds, &Arm::tool("T-noV", Src::Extracted, Sel::Always, Ver::None, false, false), h, b);
            let sh = run(&ws, seeds, &Arm::tool("TV-sh", Src::Extracted, Sel::Always, Ver::Shared, false, false), h, b);
            let ind = run(&ws, seeds, &Arm::tool("TV-in", Src::Extracted, Sel::Always, Ver::Indep, false, false), h, b);
            let c = cmp(&col(&ind, |r| r.correct), &col(&sh, |r| r.correct));
            let c2 = cmp(&col(&ind, |r| r.correct), &col(&nov, |r| r.correct));
            println!("    {:<28} {:>9.4} {:>9.4} {:>9.4}   {:>34} {:>+9.4}", name, mean(&col(&nov, |r| r.correct)), mean(&col(&sh, |r| r.correct)), mean(&col(&ind, |r| r.correct)), format!("{:+.4} [{:+.4},{:+.4}]", c.mean, c.lo, c.hi), c2.mean);
            wj.push(json!({"bug": name, "noV": mean(&col(&nov, |r| r.correct)), "shared": mean(&col(&sh, |r| r.correct)), "indep": mean(&col(&ind, |r| r.correct)), "indep_minus_shared": cj(&c, None), "indep_minus_noV": c2.mean}));
        }
        out.insert(cfg.name.to_string(), json!(wj));
    }
    Value::Object(out)
}

fn sens_mode(seeds: &[u64], tag: &str, h: &Hyper) -> Value {
    let mut out = serde_json::Map::new();
    println!("
=== SENSITIVITY ({tag}): H / R / S lift over the best fair baseline when the assumptions change ===");
    let h_arm = Arm::tool("H", Src::Extracted, Sel::Learned, Ver::Indep, false, false);
    let t_arm = Arm::tool("T", Src::Extracted, Sel::Always, Ver::None, false, false);
    let xc_arm = Arm::tool("XC", Src::Canonical, Sel::Always, Ver::None, false, false);
    let r_arm = Arm::tool("R", Src::Extracted, Sel::Rule, Ver::Indep, false, false);
    let s_arm = Arm { name: "S".into(), router: Rt::Safe, mode: Mode::Tier };
    for base in [WCfg::world_a(), WCfg::world_b()] {
        let mut settings: Vec<(String, WCfg)> = Vec::new();
        for es in [0.25, 0.5, 1.0, 2.0, 3.0] {
            let mut c = base.clone();
            c.err_scale = es;
            settings.push((format!("ungated error scale x{es}"), c));
        }
        for up in [0.0, 0.6, 1.0] {
            let mut c = base.clone();
            c.unseen_p = up;
            settings.push((format!("unseen-phrasing share {up}"), c));
        }
        let mut c = base.clone();
        c.lvl_p = [0.0, 0.0, 0.0, 0.5, 0.5];
        settings.push(("text degradation: L3/L4 only".into(), c));
        let mut c = base.clone();
        c.lvl_p = [1.0, 0.0, 0.0, 0.0, 0.0];
        settings.push(("text: L0 only (clean)".into(), c));
        let mut c = base.clone();
        c.dom_p = [0.3, 0.3, 0.2, 0.2];
        settings.push(("20% no-math prose".into(), c));
        let mut c = base.clone();
        c.err_scale = 0.25;
        c.unseen_p = 1.0;
        settings.push(("easy model (x0.25) AND all-unseen phrasing".into(), c));
        println!("  world {}:", base.name);
        println!("    {:<44} {:>8} {:>9} {:>34} {:>9} {:>9} {:>9} {:>9}", "setting", "baseline", "H", "H - best baseline (CI)", "R-base", "S-base", "T-XC", "best plain");
        let mut wj = Vec::new();
        for (name, cfg) in settings {
            let ws = worlds(&cfg, seeds);
            let pl = plain_baselines(&ws, seeds, h);
            let bi = best_idx(&pl);
            let b = col(&pl[bi].1, |r| r.correct);
            let hr = run(&ws, seeds, &h_arm, h, 0);
            let tr = run(&ws, seeds, &t_arm, h, 0);
            let xr = run(&ws, seeds, &xc_arm, h, 0);
            let c = cmp(&col(&hr, |r| r.correct), &b);
            let ct = cmp(&col(&tr, |r| r.correct), &col(&xr, |r| r.correct));
            let rr = run(&ws, seeds, &r_arm, h, 0);
            let sr = run(&ws, seeds, &s_arm, h, 0);
            let cr = cmp(&col(&rr, |r| r.correct), &b);
            let cs2 = cmp(&col(&sr, |r| r.correct), &b);
            println!("    {:<44} {:>8.4} {:>9.4} {:>34} {:>+9.4} {:>+9.4} {:>+9.4} {:>9}", name, mean(&b), mean(&col(&hr, |r| r.correct)), format!("{:+.4} [{:+.4},{:+.4}]", c.mean, c.lo, c.hi), cr.mean, cs2.mean, ct.mean, pl[bi].0);
            wj.push(json!({"setting": name, "baseline": mean(&b), "H": mean(&col(&hr, |r| r.correct)), "lift": cj(&c, None), "R_lift": cj(&cr, None), "S_lift": cj(&cs2, None), "T_minus_XC": ct.mean, "H_cost": mean(&col(&hr, |r| r.cost)), "S_cost": mean(&col(&sr, |r| r.cost))}));
        }
        out.insert(base.name.to_string(), json!(wj));
    }
    Value::Object(out)
}

fn tune_mode(seeds: &[u64]) -> Value {
    println!("\n=== TUNE (seeds {}-{}): theta for the rule arm, selector hyper-parameters for the learned arm; objective = correct - 0.02*cost/10 averaged over A, B and the prose control ===", seeds[0], seeds[seeds.len() - 1]);
    let wa = worlds(&WCfg::world_a(), seeds);
    let wb = worlds(&WCfg::world_b(), seeds);
    let wc = worlds(&WCfg::world_a().prose_only(), seeds);
    let score = |h: &Hyper, arm: &Arm| -> (f64, f64, f64, f64) {
        let ra = run(&wa, seeds, arm, h, 0);
        let rb = run(&wb, seeds, arm, h, 0);
        let rc = run(&wc, seeds, arm, h, 0);
        let f = |r: &[RunStats]| mean(&col(r, |x| x.correct)) - 0.02 * mean(&col(r, |x| x.cost)) / 10.0;
        let obj = (f(&ra) + f(&rb) + 0.5 * f(&rc)) / 2.5;
        (obj, mean(&col(&ra, |x| x.correct)), mean(&col(&rb, |x| x.correct)), mean(&col(&rc, |x| x.correct)))
    };
    let mut res = Vec::new();
    let mut best_theta = (f64::NEG_INFINITY, 0.4);
    for theta in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] {
        let mut h = Hyper::default();
        h.theta = theta;
        let (o, a, b, c) = score(&h, &Arm::tool("R", Src::Extracted, Sel::Rule, Ver::Indep, false, false));
        println!("  rule theta {:.1}: objective {:.4} (A {:.4}, B {:.4}, prose {:.4})", theta, o, a, b, c);
        res.push(json!({"kind": "theta", "theta": theta, "obj": o, "A": a, "B": b, "prose": c}));
        if o > best_theta.0 {
            best_theta = (o, theta);
        }
    }
    let mut best_sel = (f64::NEG_INFINITY, Hyper::default());
    for kappa in [0.02, 0.04, 0.08] {
        for kw_prior in [10.0, 20.0, 40.0] {
            for sel_opt in [0.5, 0.7, 0.9] {
                for (shrink, fit_w) in [(2.0, 0.5), (4.0, 0.5), (8.0, 0.5), (4.0, 0.0), (4.0, 0.25)] {
                    let mut h = Hyper::default();
                    h.fit_w = fit_w;
                    h.kappa = kappa;
                    h.kw_prior = kw_prior;
                    h.sel_opt = sel_opt;
                    h.sel_shrink = shrink;
                    let (o, a, b, c) = score(&h, &Arm::tool("H", Src::Extracted, Sel::Learned, Ver::Indep, false, false));
                    res.push(json!({"kind": "selector", "fit_w": fit_w, "kappa": kappa, "kw_prior": kw_prior, "sel_opt": sel_opt, "shrink": shrink, "obj": o, "A": a, "B": b, "prose": c}));
                    if o > best_sel.0 {
                        best_sel = (o, h);
                    }
                }
            }
        }
    }
    let mut best_tier = (f64::NEG_INFINITY, 0.4, 0.1);
    for hi in [0.5, 0.6, 0.7, 0.8, 0.9] {
        for lo in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] {
            if lo >= hi {
                continue;
            }
            let mut h = Hyper::default();
            h.theta = hi;
            h.theta_lo = lo;
            let (o, a, b, c) = score(&h, &Arm { name: "S".into(), router: Rt::Safe, mode: Mode::Tier });
            res.push(json!({"kind": "tier", "hi": hi, "lo": lo, "obj": o, "A": a, "B": b, "prose": c}));
            if o > best_tier.0 {
                best_tier = (o, hi, lo);
            }
        }
    }
    println!("  best tier (post-hoc): hi {:.1} lo {:.1} objective {:.4}", best_tier.1, best_tier.2, best_tier.0);
    println!("  best selector: kappa {} kw_prior {} sel_opt {} shrink {} fit_w {}  objective {:.4}", best_sel.1.kappa, best_sel.1.kw_prior, best_sel.1.sel_opt, best_sel.1.sel_shrink, best_sel.1.fit_w, best_sel.0);
    println!("  best theta {:.1} (objective {:.4})", best_theta.1, best_theta.0);
    json!({"grid": res, "best_theta": best_theta.1, "best_tier": {"hi": best_tier.1, "lo": best_tier.2, "obj": best_tier.0}, "best_selector": {"fit_w": best_sel.1.fit_w, "kappa": best_sel.1.kappa, "kw_prior": best_sel.1.kw_prior, "sel_opt": best_sel.1.sel_opt, "shrink": best_sel.1.sel_shrink}})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_default();
    let seeds = parse_range(&args, "--seeds", (31, 60));
    let tag = arg(&args, "--tag", "held");
    let h = hx();
    let v = match mode.as_str() {
        "tune" => tune_mode(&seeds),
        "eval" => eval_mode(&seeds, &tag, &h),
        "controls" => controls_mode(&seeds, &h),
        "bugs" => bugs_mode(&seeds, &h),
        "sens" => sens_mode(&seeds, &tag, &h),
        _ => {
            eprintln!("usage: h18_lift tune|eval|controls|bugs|sens --seeds A-B --tag t");
            return;
        }
    };
    let name = if mode == "tune" { "results/h18_tune.json".to_string() } else { format!("results/h18_{mode}_{tag}.json") };
    let _ = std::fs::write(&name, serde_json::to_string_pretty(&v).unwrap_or_default());
    println!("wrote {name}");
}
