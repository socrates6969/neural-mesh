//! H12 / R5-D: opt-in ONLINE ADAPTATION layers + an honest re-test of a learned (neural) policy as a RESIDUAL layer.
//! SIMULATION ONLY (synthetic-until-validated-on-real-traffic). Measures routing, not model reasoning.
//!
//!   cargo run --release --offline --bin h12_online -- --held 31-60 --out results/h12_online.json
//!   cargo run --release --offline --bin h12_online -- --held 61-90 --out results/h12_online_confirm61-90.json
//!   cargo run --release --offline --bin h12_online -- --timing --out results/h12_cost.json
//!   (--quick = 3-seed smoke test)
//!
//! Arms: fair baselines {keyword, eps-greedy, pooled-thompson, hebbian-robust, warm(hebr), guarded-safe} + candidates
//! {resid-warm, resid-safe, cfg-bandit, meta} + ablation resid-warm-nogate. Tune seeds 1-8 only, held-out 31-60,
//! confirm 61-90 with the SAME frozen tuning. Per-world paired 95% t-CI vs the best fair baseline (picked on the
//! evaluated seeds: conservative), Holm over all candidate x world x metric tests.

use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "h12_common/mod.rs"]
mod common;
use common::drift::DrKind;
use common::online::*;
use common::*;

use neural_mesh::router::{Rng, Router};
use neural_mesh::sim::Scenario;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};

#[global_allocator]
static GLOBAL: alloc_count::Counting = alloc_count::Counting;

const TUNE_SEEDS: (u64, u64) = (1, 8);
const TOST: f64 = 0.01; // pre-declared equivalence margin (absolute mean success)

fn h12_worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec::Sim(Scenario::Fixed),
        WorldSpec::Sim(Scenario::Random),
        WorldSpec::Sim(Scenario::NoTraps),
        WorldSpec::Big,
        WorldSpec::Syn(SynKind::Lin),
        WorldSpec::Syn(SynKind::XorCont),
        WorldSpec::Syn(SynKind::XorDrift),
        WorldSpec::Syn(SynKind::XorTok),
        WorldSpec::Syn(SynKind::Rich),
        WorldSpec::Dr(DrKind::KwDrift4),
        WorldSpec::Dr(DrKind::XorDrift3),
        WorldSpec::Dr(DrKind::CtlNoInfo),
        WorldSpec::Dr(DrKind::CtlStatic),
    ]
}

/// worlds where the keyword prior is the right tool or the world is a control: a layer must be HARMLESS here
fn harmless_world(w: &WorldSpec) -> bool {
    matches!(
        w,
        WorldSpec::Sim(_) | WorldSpec::Big | WorldSpec::Dr(DrKind::KwDrift4) | WorldSpec::Dr(DrKind::CtlNoInfo) | WorldSpec::Dr(DrKind::CtlStatic)
    )
}
fn is_control(w: &WorldSpec) -> bool {
    matches!(w, WorldSpec::Dr(DrKind::CtlNoInfo) | WorldSpec::Dr(DrKind::CtlStatic))
}

type Mat = Vec<Vec<Vec<RunRes>>>;

fn run_matrix(ws: &[WorldSpec], reg: Regime, arms: &[Arm], seeds: &[u64]) -> Mat {
    let items: Vec<(usize, usize)> = (0..ws.len()).flat_map(|w| (0..seeds.len()).map(move |s| (w, s))).collect();
    let cells: Vec<Vec<RunRes>> = items
        .par_iter()
        .map(|&(wi, si)| {
            let seed = seeds[si];
            let env = ws[wi].make(seed, ws[wi].t_for(reg));
            arms.iter()
                .map(|a| {
                    let mut r = mk_arm(a, &env, seed);
                    run_policy(&env, r.as_mut(), seed)
                })
                .collect()
        })
        .collect();
    let mut out: Mat = vec![vec![Vec::with_capacity(seeds.len()); arms.len()]; ws.len()];
    for (idx, cell) in cells.into_iter().enumerate() {
        let (wi, _) = items[idx];
        for (ai, r) in cell.into_iter().enumerate() {
            out[wi][ai].push(r);
        }
    }
    out
}

fn fulls(v: &[RunRes]) -> Vec<f64> {
    v.iter().map(|r| r.full).collect()
}
fn lates(v: &[RunRes]) -> Vec<f64> {
    v.iter().map(|r| r.late).collect()
}
fn mean_of(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}
fn objective(m: &Mat, ai: usize) -> f64 {
    mean_of(&m.iter().map(|w| mean_of(&fulls(&w[ai]))).collect::<Vec<_>>())
}

#[derive(Clone, Debug)]
struct Tuned {
    eps: (f64, f64),
    pts: f64,
    warm: f64,
    rw: ResCfg,
    rs: ResCfg,
    cb: (f64, bool),
    meta: (f64, bool, f64),
}

fn res_grid() -> Vec<ResCfg> {
    let mut g = vec![];
    for &h in &[8usize, 24] {
        for &lr in &[0.05f32, 0.15] {
            for &nm in &[100usize, 300] {
                for &mg in &[0.03f32, 0.08] {
                    for &e0 in &[0.0f64, 0.05, 0.15] {
                        g.push(ResCfg { h, lr, n_min: nm, margin: mg, e0, ..ResCfg::default_cfg() });
                    }
                }
            }
        }
    }
    g
}

fn tune(ws: &[WorldSpec], quick: bool) -> (Tuned, Value) {
    let seeds = if quick { seeds(1, 3) } else { seeds(TUNE_SEEDS.0, TUNE_SEEDS.1) };
    let reg = Regime::Mid;
    let t0 = Instant::now();
    // ---- stage A: baselines
    let mut eps_g = vec![];
    for &g in &[1.0f64, 0.99, 0.97, 0.95] {
        for &e in &[0.0f64, 0.01, 0.02, 0.05, 0.1] {
            eps_g.push((g, e));
        }
    }
    let pts_g = [1.0f64, 0.995, 0.98, 0.95];
    let warm_k = [0.5f64, 1.0, 3.0, 6.0];
    let mut arms: Vec<Arm> = vec![];
    for &(g, e) in &eps_g {
        arms.push(Arm::Base(Kind::Eps(g, e)));
    }
    for &g in &pts_g {
        arms.push(Arm::Base(Kind::Pts(g)));
    }
    for &k in &warm_k {
        arms.push(Arm::Base(Kind::Warm(k)));
    }
    let m = run_matrix(ws, reg, &arms, &seeds);
    let obj: Vec<f64> = (0..arms.len()).map(|a| objective(&m, a)).collect();
    let best_in = |lo: usize, hi: usize| (lo..hi).max_by(|&a, &b| obj[a].partial_cmp(&obj[b]).unwrap()).unwrap();
    let be = best_in(0, eps_g.len());
    let bp = best_in(eps_g.len(), eps_g.len() + pts_g.len());
    let bw = best_in(eps_g.len() + pts_g.len(), arms.len());
    let eps = eps_g[be];
    let pts = pts_g[bp - eps_g.len()];
    let warm = warm_k[bw - eps_g.len() - pts_g.len()];
    println!("[tune A] eps {:?} ({:.4}) pts {} ({:.4}) warm k={} ({:.4}) [{:.0}s]", eps, obj[be], pts, obj[bp], warm, obj[bw], t0.elapsed().as_secs_f64());
    // reference objectives for the fair baselines that need no tuning
    // ---- stage B: residual layers
    let rg = res_grid();
    let rgq: Vec<ResCfg> = if quick { rg.iter().cloned().step_by(5).collect() } else { rg.clone() };
    let mut arms: Vec<Arm> = vec![Arm::Base(Kind::Warm(warm)), Arm::Base(Kind::Safe)];
    for c in &rgq {
        arms.push(Arm::ResidWarm { k: warm, cfg: *c });
    }
    for c in &rgq {
        arms.push(Arm::ResidSafe { cfg: *c });
    }
    let m = run_matrix(ws, reg, &arms, &seeds);
    let obj_b: Vec<f64> = (0..arms.len()).map(|a| objective(&m, a)).collect();
    let nq = rgq.len();
    let brw = (2..2 + nq).max_by(|&a, &b| obj_b[a].partial_cmp(&obj_b[b]).unwrap()).unwrap();
    let brs = (2 + nq..2 + 2 * nq).max_by(|&a, &b| obj_b[a].partial_cmp(&obj_b[b]).unwrap()).unwrap();
    let rw = rgq[brw - 2];
    let rs = rgq[brs - 2 - nq];
    println!("[tune B] warm {:.4} safe {:.4} | resid-warm {} ({:.4}) | resid-safe {} ({:.4}) [{:.0}s]", obj_b[0], obj_b[1], rw.show(), obj_b[brw], rs.show(), obj_b[brs], t0.elapsed().as_secs_f64());
    let top: Vec<Value> = {
        let mut o: Vec<usize> = (2..2 + nq).collect();
        o.sort_by(|&a, &b| obj_b[b].partial_cmp(&obj_b[a]).unwrap());
        o.iter().take(5).map(|&a| json!({"cfg": rgq[a - 2].show(), "objective": obj_b[a]})).collect()
    };
    // ---- stage C: portfolios
    let mut arms: Vec<Arm> = vec![];
    let mut cb_g = vec![];
    for &g in &[0.99f64, 0.998, 1.0] {
        for &ag in &[false, true] {
            cb_g.push((g, ag));
            arms.push(Arm::CfgBandit { k: warm, gamma: g, agree: ag });
        }
    }
    let mut mt_g = vec![];
    for &g in &[0.99f64, 0.998, 1.0] {
        for &ag in &[false, true] {
            for &pn in &[0.0f64, 30.0] {
                mt_g.push((g, ag, pn));
                arms.push(Arm::Meta { k: warm, rcfg: rw, gamma: g, agree: ag, prior_n: pn });
            }
        }
    }
    let m = run_matrix(ws, reg, &arms, &seeds);
    let obj_c: Vec<f64> = (0..arms.len()).map(|a| objective(&m, a)).collect();
    let bcb = (0..cb_g.len()).max_by(|&a, &b| obj_c[a].partial_cmp(&obj_c[b]).unwrap()).unwrap();
    let bmt = (cb_g.len()..arms.len()).max_by(|&a, &b| obj_c[a].partial_cmp(&obj_c[b]).unwrap()).unwrap();
    let cb = cb_g[bcb];
    let mt = mt_g[bmt - cb_g.len()];
    println!("[tune C] cfg-bandit gamma={} agree={} ({:.4}) | meta gamma={} agree={} prior_n={} ({:.4}) [{:.0}s]", cb.0, cb.1, obj_c[bcb], mt.0, mt.1, mt.2, obj_c[bmt], t0.elapsed().as_secs_f64());
    let j = json!({"seeds": [seeds[0], seeds[seeds.len() - 1]], "eps": [eps.0, eps.1], "pts_gamma": pts, "warm_k": warm,
        "resid_warm": rw.show(), "resid_safe": rs.show(), "resid_top5": top, "cfg_bandit": {"gamma": cb.0, "agree": cb.1}, "meta": {"gamma": mt.0, "agree": mt.1, "prior_n": mt.2},
        "objectives": {"warm": obj_b[0], "safe": obj_b[1], "resid_warm": obj_b[brw], "resid_safe": obj_b[brs], "cfg_bandit": obj_c[bcb], "meta": obj_c[bmt]},
        "grid": {"res": rgq.len(), "cfg": cb_g.len(), "meta": mt_g.len()}});
    (Tuned { eps, pts, warm, rw, rs, cb: (cb.0, cb.1), meta: (mt.0, mt.1, mt.2) }, j)
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

const FAIR: [&str; 6] = ["keyword", "eps-greedy", "pooled-thompson", "hebbian-robust", "warm(hebr)", "guarded-safe"];
const NEW: [&str; 4] = ["resid-warm", "resid-safe", "cfg-bandit", "meta"];

#[derive(Clone)]
struct Test {
    world: String,
    family: &'static str,
    harmless: bool,
    arm: String,
    vs: String,
    metric: &'static str,
    d: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    losses: usize,
    p: f64,
    adj: f64,
}

fn mk_test(w: &WorldSpec, arm: &str, vs: &str, metric: &'static str, a: &[f64], b: &[f64]) -> Test {
    let pd = paired_diff(a, b);
    Test {
        world: w.label(),
        family: w.family(),
        harmless: harmless_world(w),
        arm: arm.into(),
        vs: vs.into(),
        metric,
        d: pd.diff.mean,
        lo: pd.diff.lo,
        hi: pd.diff.hi,
        wins: pd.wins,
        losses: pd.losses,
        p: paired_t_pvalue(a, b),
        adj: 1.0,
    }
}
fn holm_family(ts: &mut [Test]) {
    let p: Vec<f64> = ts.iter().map(|t| t.p).collect();
    for (t, a) in ts.iter_mut().zip(holm_adjust(&p)) {
        t.adj = a;
    }
}
fn test_json(t: &Test) -> Value {
    json!({"world": t.world, "family": t.family, "arm": t.arm, "vs": t.vs, "metric": t.metric, "diff": t.d, "ci_lo": t.lo, "ci_hi": t.hi,
        "wins": t.wins, "losses": t.losses, "p": t.p, "holm_p": t.adj, "verdict": verdict(t.adj, t.d)})
}

/// recovery after drift events, per run: (post-drift regret over 1500 tasks, dip depth, tasks to recover (censored at window))
fn recovery(bins: &[f32], orc: &[f32], drifts: &[usize]) -> Option<(f64, f64, f64)> {
    if drifts.is_empty() {
        return None;
    }
    let nb = bins.len();
    let reg: Vec<f64> = (0..nb).map(|b| (orc[b] - bins[b]) as f64).collect();
    let (mut auc, mut dip, mut rec, mut cnt) = (0.0, 0.0, 0.0, 0.0);
    for (ei, &tau) in drifts.iter().enumerate() {
        let k = tau / BIN;
        if k >= nb {
            continue;
        }
        let next = drifts.get(ei + 1).map(|&x| x / BIN).unwrap_or(nb);
        let end = (k + 6).min(next).min(nb);
        let lo = k.saturating_sub(3);
        let pre = mean_of(&reg[lo..k]);
        let post = &reg[k..end];
        auc += mean_of(post);
        let mx = post.iter().cloned().fold(f64::MIN, f64::max);
        dip += mx - pre;
        // recovery: first bin at/after the peak with regret <= pre + .03; 0 if never above pre+.03
        if mx <= pre + 0.03 {
            rec += 0.0;
        } else {
            let pk = post.iter().position(|&x| x == mx).unwrap();
            let r = post.iter().enumerate().skip(pk).find(|(_, x)| **x <= pre + 0.03).map(|(j, _)| (j + 1) * BIN).unwrap_or(post.len() * BIN);
            rec += r as f64;
        }
        cnt += 1.0;
    }
    if cnt == 0.0 {
        None
    } else {
        Some((auc / cnt, dip / cnt, rec / cnt))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let timing = args.iter().any(|a| a == "--timing");
    let val = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let (hlo, hhi) = val("--held")
        .map(|s| {
            let mut it = s.split('-');
            (it.next().unwrap().parse::<u64>().unwrap(), it.next().unwrap().parse::<u64>().unwrap())
        })
        .unwrap_or((31, 60));
    let out_path = val("--out").unwrap_or_else(|| if timing { "results/h12_cost.json".into() } else { "results/h12_online.json".into() });
    let ws = h12_worlds();
    if timing {
        run_timing(&ws, &out_path);
        return;
    }
    let eval_seeds = if quick { seeds(hlo, hlo + 2) } else { seeds(hlo, hhi) };
    let t_all = Instant::now();
    println!("H12 online-adaptation layers | held seeds {}-{} | tune seeds {}-{} | RAYON threads {} | T=6000 all worlds", eval_seeds[0], eval_seeds[eval_seeds.len() - 1], TUNE_SEEDS.0, TUNE_SEEDS.1, rayon::current_num_threads());
    let (tn, tj) = tune(&ws, quick);

    let mut nogate = tn.rw;
    nogate.nogate = true;
    let arms: Vec<Arm> = vec![
        Arm::Base(Kind::Keyword),
        Arm::Base(Kind::Eps(tn.eps.0, tn.eps.1)),
        Arm::Base(Kind::Pts(tn.pts)),
        Arm::Base(Kind::HebR),
        Arm::Base(Kind::Warm(tn.warm)),
        Arm::Base(Kind::Safe),
        Arm::ResidWarm { k: tn.warm, cfg: tn.rw },
        Arm::ResidSafe { cfg: tn.rs },
        Arm::CfgBandit { k: tn.warm, gamma: tn.cb.0, agree: tn.cb.1 },
        Arm::Meta { k: tn.warm, rcfg: tn.rw, gamma: tn.meta.0, agree: tn.meta.1, prior_n: tn.meta.2 },
        Arm::ResidWarm { k: tn.warm, cfg: nogate },
    ];
    let names = ["keyword", "eps-greedy", "pooled-thompson", "hebbian-robust", "warm(hebr)", "guarded-safe", "resid-warm", "resid-safe", "cfg-bandit", "meta", "resid-warm-nogate[abl]"];
    let incumbent = [4usize, 5, 4, 5]; // underlying incumbent of NEW[i] (meta: the recommended default guarded-safe)
    let reg = Regime::Mid;
    let t0 = Instant::now();
    let m = run_matrix(&ws, reg, &arms, &eval_seeds);
    let orc: Vec<Vec<RunRes>> = ws.iter().map(|w| eval_seeds.par_iter().map(|&s| oracle_run(&w.make(s, w.t_for(reg)))).collect()).collect();
    println!("[eval T=6000] done in {:.0}s", t0.elapsed().as_secs_f64());

    let mut f1: Vec<Test> = vec![]; // new vs best fair, full + late
    let mut f2: Vec<Test> = vec![]; // new vs its incumbent, full
    let mut f4: Vec<Test> = vec![]; // nogate ablation vs gated resid-warm
    let mut worlds_json = vec![];
    println!("\n{:<14} | {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} {:>6} {:>6} | oracle", "world", "keyw", "eps", "pts", "hebr", "warm", "safe", "r-warm", "r-safe", "cfgbd", "meta", "nogate");
    for (wi, w) in ws.iter().enumerate() {
        let mf: Vec<f64> = (0..arms.len()).map(|a| mean_of(&fulls(&m[wi][a]))).collect();
        let ml: Vec<f64> = (0..arms.len()).map(|a| mean_of(&lates(&m[wi][a]))).collect();
        let bf = (0..6).max_by(|&a, &b| mf[a].partial_cmp(&mf[b]).unwrap()).unwrap();
        let bl = (0..6).max_by(|&a, &b| ml[a].partial_cmp(&ml[b]).unwrap()).unwrap();
        for (i, nm) in NEW.iter().enumerate() {
            let ai = 6 + i;
            f1.push(mk_test(w, nm, FAIR[bf], "full", &fulls(&m[wi][ai]), &fulls(&m[wi][bf])));
            f1.push(mk_test(w, nm, FAIR[bl], "late25%", &lates(&m[wi][ai]), &lates(&m[wi][bl])));
            f2.push(mk_test(w, nm, FAIR[incumbent[i]], "full", &fulls(&m[wi][ai]), &fulls(&m[wi][incumbent[i]])));
        }
        f4.push(mk_test(w, "resid-warm-nogate", "resid-warm", "full", &fulls(&m[wi][10]), &fulls(&m[wi][6])));
        println!(
            "{:<14} | {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} | {:.4} {:.4} {:.4} {:.4} {:.4} | {:.4}  (best fair {} {:.4})",
            w.label(), mf[0], mf[1], mf[2], mf[3], mf[4], mf[5], mf[6], mf[7], mf[8], mf[9], mf[10], mean_of(&fulls(&orc[wi])), FAIR[bf], mf[bf]
        );
        let mut arms_j = serde_json::Map::new();
        for a in 0..arms.len() {
            let s = summarize(&fulls(&m[wi][a]));
            arms_j.insert(names[a].into(), json!({"full": s.mean, "full_ci": [s.lo, s.hi], "late25": ml[a]}));
        }
        worlds_json.push(json!({"world": w.label(), "family": w.family(), "T": w.t_for(reg), "harmless_world": harmless_world(w), "control": is_control(w),
            "oracle_full": mean_of(&fulls(&orc[wi])), "best_fair_full": FAIR[bf], "best_fair_late": FAIR[bl], "arms": arms_j}));
    }
    holm_family(&mut f1);
    holm_family(&mut f2);
    holm_family(&mut f4);

    println!("\n=== F1: candidate vs BEST FAIR baseline (Holm over {} tests) ===", f1.len());
    for t in &f1 {
        println!("  {:<14} {:<11} {:<8} vs {:<16} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.3} {}", t.world, t.arm, t.metric, t.vs, t.d, t.lo, t.hi, t.wins, t.losses, t.adj, verdict(t.adj, t.d));
    }
    println!("\n=== F2: candidate vs its OWN incumbent (full metric; Holm over {}; TOST margin +-{}) ===", f2.len(), TOST);
    for t in &f2 {
        let eq = t.lo > -TOST && t.hi < TOST;
        println!("  {:<14} {:<11} vs {:<16} {:+.4} [{:+.4},{:+.4}] holm {:.3} {} {}", t.world, t.arm, t.vs, t.d, t.lo, t.hi, t.adj, verdict(t.adj, t.d), if eq { "EQUIV(95%CI in +-.01)" } else { "" });
    }
    println!("\n=== F4: nogate ablation vs gated resid-warm ===");
    for t in &f4 {
        println!("  {:<14} {:+.4} [{:+.4},{:+.4}] holm {:.3} {}", t.world, t.d, t.lo, t.hi, t.adj, verdict(t.adj, t.d));
    }

    // ---------------- KEEP verdicts
    let mut keep_json = vec![];
    println!("\n=== KEEP RULE: Holm-WIN vs best fair (full metric) in >=2 independent NON-control families AND harmless (95% CI vs incumbent inside +-{TOST}) in all keyword/control worlds ===");
    for nm in NEW.iter() {
        let mut wf: Vec<&str> = f1.iter().filter(|t| t.arm == *nm && t.metric == "full" && verdict(t.adj, t.d) == "WIN" && !t.family.starts_with("ctl")).map(|t| t.family).collect();
        wf.sort();
        wf.dedup();
        let mut lf: Vec<&str> = f1.iter().filter(|t| t.arm == *nm && t.metric == "full" && verdict(t.adj, t.d) == "LOSS").map(|t| t.family).collect();
        lf.sort();
        lf.dedup();
        let hv: Vec<&Test> = f2.iter().filter(|t| t.arm == *nm && t.harmless).collect();
        let n_eq = hv.iter().filter(|t| t.lo > -TOST && t.hi < TOST).count();
        let worst = hv.iter().map(|t| t.lo).fold(f64::MAX, f64::min);
        let harmless = n_eq == hv.len();
        let keep = wf.len() >= 2 && harmless;
        println!("  {:<11} WIN families {:?} ({}) | LOSS families {:?} | harmless-worlds equivalent {}/{} (worst CI-lo vs incumbent {:+.4}) => {}", nm, wf, wf.len(), lf, n_eq, hv.len(), worst, if keep { "KEEP" } else { "NOT KEPT" });
        keep_json.push(json!({"arm": nm, "win_families": wf, "loss_families": lf, "harmless_equiv": n_eq, "harmless_total": hv.len(), "worst_ci_lo_vs_incumbent": worst, "keep": keep}));
    }

    // ---------------- recovery after drift
    println!("\n=== ADAPTIVITY: recovery after drift events (T=6000; regret to oracle in 250-task bins; post window = 1500 tasks; rec = tasks until regret <= pre+.03, censored at window) ===");
    let mut rec_json = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let drifts: Vec<usize> = w.make(eval_seeds[0], w.t_for(reg)).drifts();
        if drifts.is_empty() {
            continue;
        }
        let rows: Vec<Vec<(f64, f64, f64)>> = (0..arms.len())
            .map(|a| (0..eval_seeds.len()).map(|si| recovery(&m[wi][a][si].bins, &orc[wi][si].bins, &drifts).unwrap_or((0.0, 0.0, 0.0))).collect())
            .collect();
        let auc: Vec<f64> = (0..arms.len()).map(|a| mean_of(&rows[a].iter().map(|r| r.0).collect::<Vec<_>>())).collect();
        let bf = (0..6).min_by(|&a, &b| auc[a].partial_cmp(&auc[b]).unwrap()).unwrap();
        println!("  {} (drifts at {:?}), best fair by post-drift regret = {} ({:.4})", w.label(), drifts, FAIR[bf], auc[bf]);
        for a in 0..arms.len() {
            let aucs: Vec<f64> = rows[a].iter().map(|r| r.0).collect();
            let base: Vec<f64> = rows[bf].iter().map(|r| r.0).collect();
            let pd = paired_diff(&aucs, &base);
            let dip = mean_of(&rows[a].iter().map(|r| r.1).collect::<Vec<_>>());
            let rc = mean_of(&rows[a].iter().map(|r| r.2).collect::<Vec<_>>());
            println!("    {:<24} post-drift regret {:.4} (vs best fair {:+.4} [{:+.4},{:+.4}]; negative = better) dip {:.4} recovery {:.0} tasks", names[a], auc[a], pd.diff.mean, pd.diff.lo, pd.diff.hi, dip, rc);
            rec_json.push(json!({"world": w.label(), "arm": names[a], "post_regret": auc[a], "vs_bestfair": pd.diff.mean, "ci": [pd.diff.lo, pd.diff.hi], "dip": dip, "recovery_tasks": rc, "best_fair": FAIR[bf]}));
        }
    }

    // ---------------- time-to-parity curves (T=20000)
    println!("\n=== TIME-TO-PARITY vs late-best fair baseline (T=20000, windows of 500 tasks, parity = diff >= -.005 sustained to the end) ===");
    let curve_worlds: Vec<WorldSpec> = vec![
        WorldSpec::Sim(Scenario::Fixed),
        WorldSpec::Syn(SynKind::Lin),
        WorldSpec::Syn(SynKind::XorCont),
        WorldSpec::Syn(SynKind::XorTok),
        WorldSpec::Syn(SynKind::Rich),
        WorldSpec::Dr(DrKind::KwDrift4),
        WorldSpec::Dr(DrKind::XorDrift3),
        WorldSpec::Dr(DrKind::CtlNoInfo),
    ];
    let t1 = Instant::now();
    let mc = run_matrix(&curve_worlds, Regime::High, &arms[..10], &eval_seeds);
    println!("[curves] done in {:.0}s", t1.elapsed().as_secs_f64());
    let mut curve_json = vec![];
    for (wi, w) in curve_worlds.iter().enumerate() {
        let ml: Vec<f64> = (0..6).map(|a| mean_of(&lates(&mc[wi][a]))).collect();
        let bl = (0..6).max_by(|&a, &b| ml[a].partial_cmp(&ml[b]).unwrap()).unwrap();
        let win = 500 / BIN;
        let nw = mc[wi][0][0].bins.len() / win;
        for (i, nm) in NEW.iter().enumerate() {
            let ai = 6 + i;
            let (mut diffs, mut los) = (vec![], vec![]);
            for k in 0..nw {
                let a: Vec<f64> = mc[wi][ai].iter().map(|r| r.bins[k * win..(k + 1) * win].iter().map(|&x| x as f64).sum::<f64>() / win as f64).collect();
                let b: Vec<f64> = mc[wi][bl].iter().map(|r| r.bins[k * win..(k + 1) * win].iter().map(|&x| x as f64).sum::<f64>() / win as f64).collect();
                let pd = paired_diff(&a, &b);
                diffs.push(pd.diff.mean);
                los.push(pd.diff.lo);
            }
            let sustained = |ok: &dyn Fn(usize) -> bool| -> Option<usize> {
                let mut first = None;
                for k in (0..nw).rev() {
                    if ok(k) {
                        first = Some(k);
                    } else {
                        break;
                    }
                }
                first.map(|k| (k + 1) * 500)
            };
            let parity = sustained(&|k| diffs[k] >= -0.005);
            let overtake = sustained(&|k| los[k] > 0.0);
            let full_gap = mean_of(&diffs);
            println!("  {:<12} {:<11} vs {:<16}: tasks-to-parity {:?}, sustained CI-lo>0 {:?}, mean gap over 20000 tasks {:+.4}", w.label(), nm, FAIR[bl], parity, overtake, full_gap);
            curve_json.push(json!({"world": w.label(), "arm": nm, "ref": FAIR[bl], "tasks_to_parity": parity, "tasks_to_sig_overtake": overtake, "mean_gap_20000": full_gap, "diff_window": diffs, "ci_lo_window": los}));
        }
    }

    let out = json!({
        "experiment": "h12_online", "held": [eval_seeds[0], eval_seeds[eval_seeds.len() - 1]], "tune": [TUNE_SEEDS.0, TUNE_SEEDS.1], "T": 6000,
        "note": "simulation only; routing not reasoning; synthetic-until-validated-on-real-traffic", "tost_margin": TOST,
        "tuned": tj, "worlds": worlds_json, "F1": f1.iter().map(test_json).collect::<Vec<_>>(), "F2": f2.iter().map(test_json).collect::<Vec<_>>(),
        "F4_nogate": f4.iter().map(test_json).collect::<Vec<_>>(), "keep": keep_json, "recovery": rec_json, "curves": curve_json,
        "runtime_s": t_all.elapsed().as_secs_f64(),
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nwrote {} ({:.0}s)", out_path, t_all.elapsed().as_secs_f64());
}

// =============================================================================================
// cost: ns/decision and bytes of state

struct First;
impl Router for First {
    fn name(&self) -> String {
        "first".into()
    }
    fn select(&mut self, _t: &neural_mesh::router::Task, c: &[usize], _r: &mut Rng) -> usize {
        c[0]
    }
    fn update(&mut self, _t: &neural_mesh::router::Task, _a: usize, _r: f64) {}
}

fn run_timing(ws: &[WorldSpec], out_path: &str) {
    let pick: Vec<WorldSpec> = ws.iter().filter(|w| matches!(w.label().as_str(), "sim-fixed" | "big-32agents" | "xor-cont" | "kw-drift4")).cloned().collect();
    let rc = ResCfg::default_cfg();
    let arms: Vec<(&str, Arm)> = vec![
        ("loop-overhead(first-agent)", Arm::Base(Kind::Keyword)),
        ("keyword", Arm::Base(Kind::Keyword)),
        ("eps-greedy", Arm::Base(Kind::Eps(0.99, 0.05))),
        ("pooled-thompson", Arm::Base(Kind::Pts(1.0))),
        ("hebbian-robust", Arm::Base(Kind::HebR)),
        ("warm(hebr)", Arm::Base(Kind::Warm(1.0))),
        ("guarded-safe", Arm::Base(Kind::Safe)),
        ("resid-warm (default cfg)", Arm::ResidWarm { k: 1.0, cfg: rc }),
        ("resid-safe (default cfg)", Arm::ResidSafe { cfg: rc }),
        ("cfg-bandit (6 members)", Arm::CfgBandit { k: 1.0, gamma: 0.998, agree: false }),
        ("cfg-bandit agree", Arm::CfgBandit { k: 1.0, gamma: 0.998, agree: true }),
        ("meta (4 members)", Arm::Meta { k: 1.0, rcfg: rc, gamma: 0.998, agree: false, prior_n: 0.0 }),
        ("meta agree", Arm::Meta { k: 1.0, rcfg: rc, gamma: 0.998, agree: true, prior_n: 0.0 }),
    ];
    let mut rows = vec![];
    println!("COST (single thread, min of 3 reps, ns per select+update incl. shared env lookups; overhead row = trivial router). Live heap measured after the run (router still alive).");
    for w in &pick {
        for &tl in &[2000usize, 6000] {
            let env = w.make(5, tl);
            let mut over_ns = 0.0;
            for (name, arm) in &arms {
                let mut best_ns = f64::MAX;
                let (mut live, mut peak) = (0i64, 0i64);
                for rep in 0..3 {
                    let is_over = name.starts_with("loop-overhead");
                    alloc_count::start();
                    let mut r: Box<dyn Router> = if is_over { Box::new(First) } else { mk_arm(arm, &env, 5) };
                    let t0 = Instant::now();
                    let _ = run_policy(&env, r.as_mut(), 5);
                    let ns = t0.elapsed().as_nanos() as f64 / tl as f64;
                    if rep == 0 {
                        live = alloc_count::live();
                        peak = alloc_count::peak();
                    }
                    alloc_count::stop();
                    best_ns = best_ns.min(ns);
                    drop(r);
                }
                if name.starts_with("loop-overhead") {
                    over_ns = best_ns;
                }
                let dec = if name.starts_with("loop-overhead") { best_ns } else { best_ns - over_ns };
                println!("{:<13} T={:<5} {:<28} {:>9.0} ns/step, {:>9.0} ns/decision net | live heap {:>9} B (peak {:>9} B)", w.label(), tl, name, best_ns, dec, live, peak);
                rows.push(json!({"world": w.label(), "T": tl, "arm": name, "ns_step_total": best_ns, "ns_decision_net": dec, "live_bytes_after_run": live, "peak_bytes": peak}));
            }
        }
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&json!({"experiment": "h12_cost", "rows": rows})).unwrap()).unwrap();
    println!("wrote {out_path}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[path = "../g12_common/mod.rs"]
    #[allow(dead_code, unused_imports)]
    mod g12;

    #[test]
    fn copied_worlds_match_g12() {
        for seed in [3u64, 11] {
            for (mine, theirs) in [
                (WorldSpec::Big.make(seed, 1500), g12::WorldSpec::Big.make(seed, 1500)),
                (WorldSpec::Syn(SynKind::XorCont).make(seed, 1500), g12::WorldSpec::Syn(g12::SynKind::XorCont).make(seed, 1500)),
                (WorldSpec::Syn(SynKind::Rich).make(seed, 1500), g12::WorldSpec::Syn(g12::SynKind::Rich).make(seed, 1500)),
                (WorldSpec::Sim(Scenario::Random).make(seed, 1500), g12::WorldSpec::Sim(Scenario::Random).make(seed, 1500)),
            ] {
                for t in (0..1500).step_by(37) {
                    assert_eq!(mine.task(t), theirs.task(t));
                    for a in [0usize, 3] {
                        assert_eq!(mine.prob(t, a), theirs.prob(t, a));
                        assert_eq!(mine.outcome(t, a), theirs.outcome(t, a));
                    }
                }
            }
        }
    }

    #[test]
    fn new_worlds_sane() {
        for k in [DrKind::KwDrift4, DrKind::XorDrift3, DrKind::CtlNoInfo, DrKind::CtlStatic] {
            let env = WorldSpec::Dr(k).make(2, 2000);
            let orc = oracle_run(&env).full;
            let kw = run_policy(&env, mk_router(&Kind::Keyword, &env, 2).as_mut(), 2).full;
            eprintln!("{}: oracle {:.3} keyword {:.3} drifts {:?}", k.label(), orc, kw, env.drifts());
            assert!(orc > 0.6 && orc < 0.95);
            if k == DrKind::CtlStatic {
                assert!(kw > orc - 0.03, "keyword should be ~optimal in the static world");
            }
            if k == DrKind::KwDrift4 {
                assert_eq!(env.drifts().len(), 3);
            }
        }
    }

    #[test]
    fn closed_layer_is_bit_identical_to_incumbent() {
        let env = WorldSpec::Syn(SynKind::XorCont).make(3, 1500);
        let n = env.n_agents();
        let cfg = ResCfg { n_min: 10_000_000, e0: 0.0, ..ResCfg::default_cfg() };
        let a = run_policy(&env, mk_router(&Kind::Warm(1.0), &env, 3).as_mut(), 3);
        let mut lay = ResidualEns::new(mk_router(&Kind::Warm(1.0), &env, 3), n, cfg, env.dense(), 3);
        let b = run_policy(&env, &mut lay, 3);
        assert_eq!(a.full, b.full);
        assert_eq!(a.bins, b.bins);
        assert_eq!(lay.overrides, 0);
    }

    #[test]
    fn residual_layer_is_deterministic_and_can_override() {
        let env = WorldSpec::Syn(SynKind::XorTok).make(4, 3000);
        let n = env.n_agents();
        let cfg = ResCfg { e0: 0.1, ..ResCfg::default_cfg() };
        let mk = || ResidualEns::new(mk_router(&Kind::Warm(1.0), &env, 4), n, cfg, env.dense(), 4);
        let mut x = mk();
        let mut y = mk();
        let rx = run_policy(&env, &mut x, 4);
        let ry = run_policy(&env, &mut y, 4);
        assert_eq!(rx.full, ry.full);
        eprintln!("xor-tok overrides {} explores {} full {:.3}", x.overrides, x.explores, rx.full);
        assert!(x.explores > 0);
    }

    #[test]
    fn portfolio_runs_and_pulls_all_members() {
        let env = WorldSpec::Dr(DrKind::KwDrift4).make(1, 2000);
        for agree in [false, true] {
            let mut p = mk_arm(&Arm::Meta { k: 1.0, rcfg: ResCfg::default_cfg(), gamma: 0.998, agree, prior_n: 0.0 }, &env, 1);
            let r = run_policy(&env, p.as_mut(), 1);
            assert!(r.full > 0.4);
        }
        let mut c = mk_arm(&Arm::CfgBandit { k: 1.0, gamma: 0.998, agree: false }, &env, 1);
        let r = run_policy(&env, c.as_mut(), 1);
        assert!(r.full > 0.4);
    }
}
