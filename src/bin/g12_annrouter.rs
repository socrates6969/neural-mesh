//! G12 (W13): LEARNED-POLICY ROUTER. Does a tiny native feed-forward net (features -> per-agent logits,
//! online SGD / policy gradient on the SAME graded outcomes the Hebbian router sees, softmax exploration with
//! decaying temperature, optional replay) route better than keyword / eps-greedy / pooled-thompson /
//! warm(hebbian-robust) / guarded-safe, and at what data volume?
//!
//! Simulation only. An outer net cannot raise a frozen model's reasoning; this measures ROUTING.
//!
//! Protocol: every knob (baseline eps/gamma/warm-k, net hyper-parameters) is tuned on seeds 1-8 ONLY, per
//! data regime (low / mid 6000 / high 20000; high re-uses the mid config); held-out 31-60 (default) or
//! `--held 61-90` (confirmation, SAME frozen tuning because tuning always re-runs deterministically on 1-8).
//! Paired 95% t-CI of net vs the BEST FAIR baseline (chosen per world x regime on the evaluated seeds, which
//! is conservative against the net), Holm per family, a claim needs Holm-WINs in >= 2 independent world families.
//!
//! Usage: cargo run --release --offline --bin g12_annrouter -- [--held 31-60] [--out results/g12_annrouter.json]
//!        ... -- --timing [--out results/g12_cost.json]      (ns/decision + bytes of state, single thread)
//!        ... -- --quick                                      (smoke: 3 seeds, low regime only)

use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g12_common/mod.rs"]
mod common;
use common::*;

use neural_mesh::router::{Rng, Router};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};

#[global_allocator]
static GLOBAL: alloc_count::Counting = alloc_count::Counting;

const TUNE_SEEDS: (u64, u64) = (1, 8);

/// results[world][arm][seed]
type Mat = Vec<Vec<Vec<RunRes>>>;

fn run_matrix(ws: &[WorldSpec], reg: Regime, arms: &[Kind], seeds: &[u64]) -> Mat {
    let items: Vec<(usize, usize)> = (0..ws.len()).flat_map(|w| (0..seeds.len()).map(move |s| (w, s))).collect();
    let cells: Vec<Vec<RunRes>> = items
        .par_iter()
        .map(|&(wi, si)| {
            let seed = seeds[si];
            let env = ws[wi].make(seed, ws[wi].t_for(reg));
            arms.iter()
                .map(|k| {
                    let mut r = mk_router(k, &env, seed);
                    run_policy(&env, r.as_mut(), seed)
                })
                .collect()
        })
        .collect();
    let mut out: Mat = vec![vec![Vec::with_capacity(seeds.len()); arms.len()]; ws.len()];
    for (idx, cell) in cells.into_iter().enumerate() {
        let (wi, _si) = items[idx];
        for (ai, r) in cell.into_iter().enumerate() {
            out[wi][ai].push(r);
        }
    }
    out
}

fn oracle_matrix(ws: &[WorldSpec], reg: Regime, seeds: &[u64]) -> Vec<Vec<RunRes>> {
    ws.iter()
        .map(|w| seeds.par_iter().map(|&s| oracle_run(&w.make(s, w.t_for(reg)))).collect::<Vec<_>>())
        .collect()
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

/// objective of one arm over all worlds: mean over worlds of the mean (over seeds) full-run success
fn objective(m: &Mat, ai: usize) -> f64 {
    mean_of(&m.iter().map(|w| mean_of(&fulls(&w[ai]))).collect::<Vec<_>>())
}

#[derive(Clone, Debug)]
struct Tuned {
    eps: (f64, f64),
    pts: f64,
    warm: f64,
    ann: AnnCfg,
    ann_kw: AnnCfg,
}

fn ann_grid() -> Vec<AnnCfg> {
    let mut g = vec![];
    for &h in &[8usize, 24, 64] {
        for &lr in &[0.05f32, 0.15, 0.4] {
            for &(t0, tm) in &[(0.3f32, 0.02f32), (0.3, 0.05)] {
                for &half in &[100.0f32, 300.0, 1500.0] {
                    for &rk in &[0usize, 3, 8] {
                        g.push(AnnCfg { mode: Mode::Reg, h, lr, tau0: t0, tau_min: tm, half, replay_k: rk, replay_cap: 1000, use_dense: true, use_kw: false });
                    }
                }
            }
        }
        for &lr in &[0.02f32, 0.05, 0.15] {
            g.push(AnnCfg { mode: Mode::Pg, h, lr, tau0: 0.0, tau_min: 0.0, half: 1.0, replay_k: 0, replay_cap: 1, use_dense: true, use_kw: false });
        }
    }
    g
}

fn tune(ws: &[WorldSpec], reg: Regime, quick: bool) -> (Tuned, Value) {
    let seeds = if quick { seeds(1, 3) } else { seeds(TUNE_SEEDS.0, TUNE_SEEDS.1) };
    let mut arms: Vec<Kind> = vec![];
    let mut eps_g = vec![];
    for &g in &[1.0f64, 0.99, 0.97, 0.95] {
        for &e in &[0.0f64, 0.01, 0.02, 0.05, 0.1] {
            eps_g.push((g, e));
        }
    }
    let pts_g = [1.0f64, 0.995, 0.98, 0.95];
    let warm_k = [0.5f64, 1.0, 3.0, 6.0];
    let ag = ann_grid();
    let ann_g: Vec<AnnCfg> = if quick { ag.into_iter().step_by(6).collect() } else { ag };
    for &(g, e) in &eps_g {
        arms.push(Kind::Eps(g, e));
    }
    for &g in &pts_g {
        arms.push(Kind::Pts(g));
    }
    for &k in &warm_k {
        arms.push(Kind::Warm(k));
    }
    for c in &ann_g {
        arms.push(Kind::Ann(*c));
    }
    for c in &ann_g {
        let mut c2 = *c;
        c2.use_kw = true;
        arms.push(Kind::Ann(c2));
    }
    let t0 = Instant::now();
    let m = run_matrix(ws, reg, &arms, &seeds);
    let obj: Vec<f64> = (0..arms.len()).map(|a| objective(&m, a)).collect();
    let best_in = |lo: usize, hi: usize| (lo..hi).max_by(|&a, &b| obj[a].partial_cmp(&obj[b]).unwrap()).unwrap();
    let (e0, e1) = (0, eps_g.len());
    let (p0, p1) = (e1, e1 + pts_g.len());
    let (w0, w1) = (p1, p1 + warm_k.len());
    let (a0, a1) = (w1, w1 + ann_g.len());
    let (k0, k1) = (a1, arms.len());
    let be = best_in(e0, e1);
    let bp = best_in(p0, p1);
    let bw = best_in(w0, w1);
    let ba = best_in(a0, a1);
    let bk = best_in(k0, k1);
    let tuned = Tuned { eps: eps_g[be - e0], pts: pts_g[bp - p0], warm: warm_k[bw - w0], ann: ann_g[ba - a0], ann_kw: { let mut c = ann_g[bk - k0]; c.use_kw = true; c } };
    println!(
        "[tune {}] seeds {}-{}: eps {:?} ({:.4}) | pts g={} ({:.4}) | warm k={} ({:.4}) | ann {} ({:.4}) | ann+kw {} ({:.4}) [{:.0}s]",
        reg.label(),
        seeds[0],
        seeds[seeds.len() - 1],
        tuned.eps,
        obj[be],
        tuned.pts,
        obj[bp],
        tuned.warm,
        obj[bw],
        tuned.ann.show(),
        obj[ba],
        tuned.ann_kw.show(),
        obj[bk],
        t0.elapsed().as_secs_f64()
    );
    // top-5 ann configs for the log
    let mut order: Vec<usize> = (a0..a1).collect();
    order.sort_by(|&a, &b| obj[b].partial_cmp(&obj[a]).unwrap());
    let top: Vec<Value> = order.iter().take(5).map(|&a| json!({"cfg": ann_g[a - a0].show(), "objective": obj[a]})).collect();
    for t in &top {
        println!("    ann top: {}", t);
    }
    let ann_best_pg = (a0..a1).filter(|&a| ann_g[a - a0].mode == Mode::Pg).max_by(|&a, &b| obj[a].partial_cmp(&obj[b]).unwrap());
    let ann_best_reg = (a0..a1).filter(|&a| ann_g[a - a0].mode == Mode::Reg).max_by(|&a, &b| obj[a].partial_cmp(&obj[b]).unwrap());
    let j = json!({
        "regime": reg.label(), "seeds": [seeds[0], seeds[seeds.len()-1]],
        "eps": [tuned.eps.0, tuned.eps.1], "eps_obj": obj[be], "pts_gamma": tuned.pts, "pts_obj": obj[bp],
        "warm_k": tuned.warm, "warm_obj": obj[bw], "ann": tuned.ann.show(), "ann_obj": obj[ba], "ann_kw": tuned.ann_kw.show(), "ann_kw_obj": obj[bk], "ann_top5": top,
        "ann_best_pg": ann_best_pg.map(|a| json!({"cfg": ann_g[a-a0].show(), "objective": obj[a]})),
        "ann_best_reg": ann_best_reg.map(|a| json!({"cfg": ann_g[a-a0].show(), "objective": obj[a]})),
        "grid": {"eps": eps_g.len(), "pts": pts_g.len(), "warm": warm_k.len(), "ann": ann_g.len()},
    });
    (tuned, j)
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

struct Test {
    regime: &'static str,
    world: String,
    family: &'static str,
    nonlinear: bool,
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

fn mk_test(regime: &'static str, w: &WorldSpec, arm: &str, vs: &str, metric: &'static str, a: &[f64], b: &[f64]) -> Test {
    let pd = paired_diff(a, b);
    Test {
        regime,
        world: w.label(),
        family: w.family(),
        nonlinear: w.nonlinear(),
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
    let adj = holm_adjust(&p);
    for (t, a) in ts.iter_mut().zip(adj) {
        t.adj = a;
    }
}

fn test_json(t: &Test) -> Value {
    json!({"regime": t.regime, "world": t.world, "family": t.family, "arm": t.arm, "vs": t.vs, "metric": t.metric,
           "diff": t.d, "ci_lo": t.lo, "ci_hi": t.hi, "wins": t.wins, "losses": t.losses, "p": t.p, "holm_p": t.adj,
           "verdict": verdict(t.adj, t.d)})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let timing = args.iter().any(|a| a == "--timing");
    let val = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let held = val("--held").map(|s| {
        let mut it = s.split('-');
        (it.next().unwrap().parse::<u64>().unwrap(), it.next().unwrap().parse::<u64>().unwrap())
    });
    let (hlo, hhi) = held.unwrap_or((31, 60));
    let out_path = val("--out").unwrap_or_else(|| if timing { "results/g12_cost.json".into() } else { "results/g12_annrouter.json".into() });
    let ws = all_worlds();
    if timing {
        run_timing(&ws, &out_path);
        return;
    }
    let eval_seeds = if quick { seeds(hlo, hlo + 2) } else { seeds(hlo, hhi) };
    let regs: Vec<Regime> = if quick { vec![Regime::Low] } else { vec![Regime::Low, Regime::Mid, Regime::High] };
    let t_all = Instant::now();
    println!("G12 learned-policy router | held seeds {}-{} | tune seeds {}-{} | RAYON threads {}", eval_seeds[0], eval_seeds[eval_seeds.len() - 1], TUNE_SEEDS.0, TUNE_SEEDS.1, rayon::current_num_threads());
    // ---------------------------------------------------------------- tuning (seeds 1-8 only)
    let (tl, jl) = tune(&ws, Regime::Low, quick);
    let (tm, jm) = if quick { (tl.clone(), jl.clone()) } else { tune(&ws, Regime::Mid, quick) };
    let tuned_json = json!([jl, jm]);

    // ---------------------------------------------------------------- evaluation
    let mut reg_json = vec![];
    let mut all_f1: Vec<Test> = vec![];
    let mut all_f2: Vec<Test> = vec![];
    let mut all_f3: Vec<Test> = vec![];
    let mut all_f4: Vec<Test> = vec![];
    let mut all_f5: Vec<Test> = vec![];
    let mut curve_json: Vec<Value> = vec![];
    for &reg in &regs {
        let tn = if reg == Regime::Low { &tl } else { &tm };
        let other = if reg == Regime::Low { &tm } else { &tl };
        let mut ann_tok = tn.ann;
        ann_tok.use_dense = false;
        let arms: Vec<Kind> = vec![
            Kind::Keyword,
            Kind::Eps(tn.eps.0, tn.eps.1),
            Kind::Pts(tn.pts),
            Kind::HebR,
            Kind::Warm(tn.warm),
            Kind::Safe,
            Kind::PtsPairs(tn.pts),
            Kind::Ann(tn.ann),
            Kind::Ann(ann_tok),
            Kind::Ann(other.ann),
            Kind::Ann(tn.ann_kw),
        ];
        let names = ["keyword", "eps-greedy", "pooled-thompson", "hebbian-robust", "warm(hebr)", "guarded-safe", "pts-pairs[ref]", "ann", "ann-tok[ablation]", "ann-other-regime-cfg", "ann+kw"];
        let t0 = Instant::now();
        let m = run_matrix(&ws, reg, &arms, &eval_seeds);
        let orc = oracle_matrix(&ws, reg, &eval_seeds);
        println!("[eval {}] done in {:.0}s", reg.label(), t0.elapsed().as_secs_f64());
        let mut worlds_json = vec![];
        let mut f1: Vec<Test> = vec![];
        let mut f2: Vec<Test> = vec![];
        let mut f3: Vec<Test> = vec![];
        let mut f4: Vec<Test> = vec![];
        let mut f5: Vec<Test> = vec![];
        println!("\n=== regime {} | ann cfg {} | other cfg {} ===", reg.label(), tn.ann.show(), other.ann.show());
        println!("{:<24} {:>6} | {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} {:>7} | best-fair ... ann-diff (full) [95% CI] | late-diff", "world", "T", "keyw", "eps", "pts", "hebr", "warm", "safe", "pairs", "ann", "ann-tok", "ann+kw");
        for (wi, w) in ws.iter().enumerate() {
            let mean_full: Vec<f64> = (0..arms.len()).map(|a| mean_of(&fulls(&m[wi][a]))).collect();
            let mean_late: Vec<f64> = (0..arms.len()).map(|a| mean_of(&lates(&m[wi][a]))).collect();
            let bf = (0..6).max_by(|&a, &b| mean_full[a].partial_cmp(&mean_full[b]).unwrap()).unwrap();
            let bl = (0..6).max_by(|&a, &b| mean_late[a].partial_cmp(&mean_late[b]).unwrap()).unwrap();
            let ann = 7;
            let tok = 8;
            f1.push(mk_test(reg.label(), w, "ann", FAIR[bf], "full", &fulls(&m[wi][ann]), &fulls(&m[wi][bf])));
            f1.push(mk_test(reg.label(), w, "ann", FAIR[bl], "late25%", &lates(&m[wi][ann]), &lates(&m[wi][bl])));
            for b in 0..6 {
                f2.push(mk_test(reg.label(), w, "ann", FAIR[b], "full", &fulls(&m[wi][ann]), &fulls(&m[wi][b])));
            }
            f3.push(mk_test(reg.label(), w, "ann-tok", FAIR[bf], "full", &fulls(&m[wi][tok]), &fulls(&m[wi][bf])));
            f3.push(mk_test(reg.label(), w, "ann-tok", FAIR[bl], "late25%", &lates(&m[wi][tok]), &lates(&m[wi][bl])));
            f4.push(mk_test(reg.label(), w, "ann", "pts-pairs[ref]", "full", &fulls(&m[wi][ann]), &fulls(&m[wi][6])));
            f4.push(mk_test(reg.label(), w, "ann", "pts-pairs[ref]", "late25%", &lates(&m[wi][ann]), &lates(&m[wi][6])));
            f5.push(mk_test(reg.label(), w, "ann+kw", FAIR[bf], "full", &fulls(&m[wi][10]), &fulls(&m[wi][bf])));
            f5.push(mk_test(reg.label(), w, "ann+kw", FAIR[bl], "late25%", &lates(&m[wi][10]), &lates(&m[wi][bl])));
            let tlen = w.t_for(reg);
            let fx = f1.len() - 2;
            println!(
                "{:<24} {:>6} | {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} | {:.4} {:.4} {:.4} {:.4} | {}={:.4} -> ann {:+.4} [{:+.4},{:+.4}] | late {:+.4} [{:+.4},{:+.4}] oracle {:.4}",
                w.label(), tlen, mean_full[0], mean_full[1], mean_full[2], mean_full[3], mean_full[4], mean_full[5], mean_full[6], mean_full[7], mean_full[8], mean_full[10],
                FAIR[bf], mean_full[bf], f1[fx].d, f1[fx].lo, f1[fx].hi, f1[fx + 1].d, f1[fx + 1].lo, f1[fx + 1].hi, mean_of(&fulls(&orc[wi]))
            );
            // prefix curve table
            let hs = [250usize, 500, 1000, 2000, 4000, 6000, 10000, 20000];
            let mut arms_j = serde_json::Map::new();
            for a in 0..arms.len() {
                let prefix: Vec<Value> = hs
                    .iter()
                    .filter(|&&h| h <= tlen)
                    .map(|&h| json!({"h": h, "mean": mean_of(&m[wi][a].iter().map(|r| prefix_mean(r, h)).collect::<Vec<_>>())}))
                    .collect();
                let (mu, lo, hi) = {
                    let s = summarize(&fulls(&m[wi][a]));
                    (s.mean, s.lo, s.hi)
                };
                arms_j.insert(names[a].into(), json!({"full": mu, "full_ci": [lo, hi], "late25": mean_late[a], "prefix": prefix}));
            }
            worlds_json.push(json!({"world": w.label(), "family": w.family(), "nonlinear": w.nonlinear(), "T": tlen,
                "oracle_full": mean_of(&fulls(&orc[wi])), "oracle_late": mean_of(&lates(&orc[wi])),
                "best_fair_full": FAIR[bf], "best_fair_late": FAIR[bl], "arms": arms_j}));

            // ------------- tasks-to-parity curve (high regime only, window 500)
            if reg == Regime::High {
                let win = 500 / BIN; // bins per window
                let nb = m[wi][0][0].bins.len();
                let nw = nb / win;
                let refi = bl;
                let mut diffs = vec![];
                let mut los = vec![];
                let mut a_means = vec![];
                let mut r_means = vec![];
                for k in 0..nw {
                    let a: Vec<f64> = m[wi][ann].iter().map(|r| r.bins[k * win..(k + 1) * win].iter().map(|&x| x as f64).sum::<f64>() / win as f64).collect();
                    let b: Vec<f64> = m[wi][refi].iter().map(|r| r.bins[k * win..(k + 1) * win].iter().map(|&x| x as f64).sum::<f64>() / win as f64).collect();
                    let pd = paired_diff(&a, &b);
                    diffs.push(pd.diff.mean);
                    los.push(pd.diff.lo);
                    a_means.push(mean_of(&a));
                    r_means.push(mean_of(&b));
                }
                // first window such that the condition holds for it and all later windows
                let sustained = |ok: &dyn Fn(usize) -> bool| -> Option<usize> {
                    let mut first = None;
                    for k in (0..nw).rev() {
                        if ok(k) {
                            first = Some(k);
                        } else {
                            break;
                        }
                    }
                    first
                };
                let parity = sustained(&|k| diffs[k] >= -0.005).map(|k| (k + 1) * 500);
                let overtake = sustained(&|k| los[k] > 0.0).map(|k| (k + 1) * 500);
                let never_behind = sustained(&|k| diffs[k] >= 0.0).map(|k| (k + 1) * 500);
                println!("    curve vs {} (late-best fair): tasks-to-parity(>= -.005, sustained) = {:?}, sustained diff>=0 = {:?}, sustained CI-lo>0 = {:?}", FAIR[refi], parity, never_behind, overtake);
                curve_json.push(json!({"world": w.label(), "ref": FAIR[refi], "window": 500,
                    "tasks_to_parity": parity, "tasks_to_nonneg": never_behind, "tasks_to_sig_overtake": overtake,
                    "ann_window_means": a_means, "ref_window_means": r_means, "diff_window": diffs, "ci_lo_window": los}));
            }
        }
        holm_family(&mut f1);
        holm_family(&mut f2);
        holm_family(&mut f3);
        holm_family(&mut f4);
        holm_family(&mut f5);
        // F1 verdict listing
        println!("  F1 (ann vs best fair, Holm over {} tests):", f1.len());
        for t in &f1 {
            println!("    {:<24} {:<8} vs {:<16} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.3} {}", t.world, t.metric, t.vs, t.d, t.lo, t.hi, t.wins, t.losses, t.adj, verdict(t.adj, t.d));
        }
        let cnt = |ts: &[Test], metric: &str, v: &str| ts.iter().filter(|t| (metric.is_empty() || t.metric == metric) && verdict(t.adj, t.d) == v).count();
        println!("  Holm counts F1 full: {}W/{}L; F1 late: {}W/{}L; F2: {}W/{}L; F3(ann-tok): {}W/{}L", cnt(&f1, "full", "WIN"), cnt(&f1, "full", "LOSS"), cnt(&f1, "late25%", "WIN"), cnt(&f1, "late25%", "LOSS"), cnt(&f2, "", "WIN"), cnt(&f2, "", "LOSS"), cnt(&f3, "", "WIN"), cnt(&f3, "", "LOSS"));
        for (label, ts) in [("F1", &f1), ("F3", &f3), ("F4(vs pts-pairs)", &f4), ("F5(ann+kw)", &f5)] {
            for metric in ["full", "late25%"] {
                let mut fams: Vec<&str> = ts.iter().filter(|t| t.metric == metric && verdict(t.adj, t.d) == "WIN").map(|t| t.family).collect();
                fams.sort();
                fams.dedup();
                let mut lf: Vec<&str> = ts.iter().filter(|t| t.metric == metric && verdict(t.adj, t.d) == "LOSS").map(|t| t.family).collect();
                lf.sort();
                lf.dedup();
                println!("  {label} {metric}: WIN families {:?} ({} => {}), LOSS families {:?}", fams, fams.len(), if fams.len() >= 2 { "KEEP (>=2 independent families)" } else { "NOT kept" }, lf);
            }
        }
        reg_json.push(json!({"regime": reg.label(), "ann_cfg": tn.ann.show(), "ann_other_cfg": other.ann.show(),
            "eps": [tn.eps.0, tn.eps.1], "pts_gamma": tn.pts, "warm_k": tn.warm, "worlds": worlds_json,
            "F1": f1.iter().map(test_json).collect::<Vec<_>>(), "F2": f2.iter().map(test_json).collect::<Vec<_>>(), "F3": f3.iter().map(test_json).collect::<Vec<_>>(), "F4": f4.iter().map(test_json).collect::<Vec<_>>(), "F5": f5.iter().map(test_json).collect::<Vec<_>>(), "ann_kw_cfg": tn.ann_kw.show()}));
        all_f1.extend(f1);
        all_f2.extend(f2);
        all_f3.extend(f3);
        all_f4.extend(f4);
        all_f5.extend(f5);
    }
    let out = json!({
        "experiment": "g12_annrouter", "held": [eval_seeds[0], eval_seeds[eval_seeds.len()-1]], "tune": [TUNE_SEEDS.0, TUNE_SEEDS.1],
        "note": "simulation only; measures routing, not model reasoning", "tuned": tuned_json,
        "regimes": reg_json, "curves": curve_json, "runtime_s": t_all.elapsed().as_secs_f64(),
        "n_tests": {"F1": all_f1.len(), "F2": all_f2.len(), "F3": all_f3.len(), "F4": all_f4.len(), "F5": all_f5.len()},
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nwrote {} ({:.0}s)", out_path, t_all.elapsed().as_secs_f64());
}

// =============================================================================================
// cost: ns/decision and bytes of state (single thread, counting allocator)

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
    let pick: Vec<WorldSpec> = ws.iter().filter(|w| matches!(w.label().as_str(), "sim-fixed" | "big-32agents" | "xor-cont" | "rich-cont")).cloned().collect();
    // net configs: use the same tuned families the main run reports? timing is config-dependent, so time the two
    // extremes of the grid (H8 no replay, H24 + replay 3) plus the tok ablation shape
    let c_small = AnnCfg { mode: Mode::Reg, h: 8, lr: 0.05, tau0: 0.3, tau_min: 0.02, half: 300.0, replay_k: 0, replay_cap: 1000, use_dense: true, use_kw: false };
    let c_big = AnnCfg { mode: Mode::Reg, h: 24, lr: 0.05, tau0: 0.3, tau_min: 0.02, half: 300.0, replay_k: 3, replay_cap: 1000, use_dense: true, use_kw: false };
    let c_pg = AnnCfg { mode: Mode::Pg, h: 24, lr: 0.05, tau0: 0.0, tau_min: 0.0, half: 1.0, replay_k: 0, replay_cap: 1, use_dense: true, use_kw: false };
    let arms: Vec<(&str, Kind)> = vec![
        ("loop-overhead(first-agent)", Kind::Keyword),
        ("keyword", Kind::Keyword),
        ("eps-greedy", Kind::Eps(0.99, 0.05)),
        ("pooled-thompson", Kind::Pts(1.0)),
        ("hebbian-robust", Kind::HebR),
        ("warm(hebr)", Kind::Warm(1.0)),
        ("guarded-safe", Kind::Safe),
        ("pts-pairs", Kind::PtsPairs(1.0)),
        ("ann H8 no-replay", Kind::Ann(c_small)),
        ("ann H24 replay3", Kind::Ann(c_big)),
        ("ann-pg H24", Kind::Ann(c_pg)),
    ];
    let mut rows = vec![];
    println!("COST (single thread, min of 3 reps, ns per select+update INCLUDING the env lookups shared by all arms; overhead row = trivial router)");
    for w in &pick {
        for &tl in &[2000usize, 20000] {
            let env = w.make(5, tl);
            let mut over_ns = 0.0;
            for (name, kind) in &arms {
                let mut best_ns = f64::MAX;
                let mut live = 0i64;
                let mut peak = 0i64;
                for rep in 0..3 {
                    let is_over = name.starts_with("loop-overhead");
                    alloc_count::start();
                    let mut r: Box<dyn Router> = if is_over { Box::new(First) } else { mk_router(kind, &env, 5) };
                    let t0 = Instant::now();
                    let _res = run_policy(&env, r.as_mut(), 5);
                    let ns = t0.elapsed().as_nanos() as f64 / tl as f64;
                    if rep == 0 {
                        // run_policy allocates a per-task f32 vec (freed after); the router is still alive: LIVE ~= router state
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
                // run_policy's own per-task buffer is gone by now (function returned); subtract the router-independent
                // baseline by reporting live bytes of the arm minus the overhead row's
                println!("{:<14} T={:<6} {:<28} {:>9.0} ns/step total, {:>9.0} ns/decision net of loop | live heap after run {:>9} B (peak {:>9} B)", w.label(), tl, name, best_ns, dec, live, peak);
                rows.push(json!({"world": w.label(), "T": tl, "arm": name, "ns_step_total": best_ns, "ns_decision_net": dec, "live_bytes_after_run": live, "peak_bytes": peak}));
            }
        }
    }
    // analytic bytes for the nets
    let mut nets = vec![];
    for (name, c) in [("ann H8", c_small), ("ann H24 replay3", c_big), ("ann-pg H24", c_pg)] {
        for (wl, n, dd) in [("sim 6 agents", 6usize, 0usize), ("big 32 agents", 32, 0), ("rich-cont 8 agents + 10 dense", 8, 10)] {
            let dense = if dd > 0 { Some(std::sync::Arc::new(vec![vec![0.0f32; dd]; 1])) } else { None };
            let r = AnnRouter::new(n, c, dense, None, 1);
            nets.push(json!({"net": name, "case": wl, "params": r.n_params(), "param_bytes": r.param_bytes(), "replay_entry_bytes": std::mem::size_of::<[u16; MAXI]>() + std::mem::size_of::<[f32; MAXI]>() + 8, "replay_cap": c.replay_cap}));
            println!("{name} / {wl}: params {} = {} B (+ up to {} replay entries)", r.n_params(), r.param_bytes(), if c.replay_k > 0 { c.replay_cap } else { 0 });
        }
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&json!({"experiment": "g12_cost", "rows": rows, "nets": nets})).unwrap()).unwrap();
    println!("wrote {out_path}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[path = "../f3_common/mod.rs"]
    #[allow(dead_code, unused_imports)]
    mod f3;

    #[test]
    fn bigw_matches_f3_at_4000() {
        for seed in [3u64, 11] {
            let mine = WorldSpec::Big.make(seed, 4000);
            let theirs = f3::Env::big(seed);
            for t in (0..4000).step_by(37) {
                assert_eq!(mine.task(t), theirs.task(t));
                for a in [0usize, 7, 31] {
                    assert_eq!(mine.prob(t, a), theirs.prob(t, a));
                    assert_eq!(mine.outcome(t, a), theirs.outcome(t, a));
                }
            }
            let tm = WorldSpec::BigTail.make(seed, 4000);
            let tt = f3::Env::big_tail(seed);
            for t in (0..4000).step_by(53) {
                assert_eq!(tm.task(t), tt.task(t));
                assert_eq!(tm.prob(t, 5), tt.prob(t, 5));
            }
        }
    }

    #[test]
    fn sim_world_len_param_and_f3_equal_at_2000() {
        let mine = WorldSpec::Sim(neural_mesh::sim::Scenario::Random).make(4, 2000);
        let theirs = f3::Env::sim(4, neural_mesh::sim::Scenario::Random);
        for t in (0..2000).step_by(41) {
            assert_eq!(mine.task(t), theirs.task(t));
            assert_eq!(mine.prob(t, 2), theirs.prob(t, 2));
        }
    }

    #[test]
    fn deterministic() {
        let env = WorldSpec::Syn(SynKind::XorCont).make(2, 1500);
        let c = AnnCfg { mode: Mode::Reg, h: 8, lr: 0.05, tau0: 0.3, tau_min: 0.02, half: 300.0, replay_k: 2, replay_cap: 500, use_dense: true, use_kw: false };
        let mut a = mk_router(&Kind::Ann(c), &env, 2);
        let mut b = mk_router(&Kind::Ann(c), &env, 2);
        let ra = run_policy(&env, a.as_mut(), 2);
        let rb = run_policy(&env, b.as_mut(), 2);
        assert_eq!(ra.full, rb.full);
    }

    #[test]
    fn net_solves_xor_tokens_where_additive_routers_cannot() {
        let mut ann = 0.0;
        let mut pts = 0.0;
        let mut pairs = 0.0;
        let c = AnnCfg { mode: Mode::Reg, h: 8, lr: 0.1, tau0: 0.3, tau_min: 0.02, half: 300.0, replay_k: 0, replay_cap: 1, use_dense: true, use_kw: false };
        for seed in 1..=5u64 {
            let env = WorldSpec::Syn(SynKind::XorTok).make(seed, 3000);
            ann += run_policy(&env, mk_router(&Kind::Ann(c), &env, seed).as_mut(), seed).late;
            pts += run_policy(&env, mk_router(&Kind::Pts(1.0), &env, seed).as_mut(), seed).late;
            pairs += run_policy(&env, mk_router(&Kind::PtsPairs(1.0), &env, seed).as_mut(), seed).late;
        }
        let (ann, pts, pairs) = (ann / 5.0, pts / 5.0, pairs / 5.0);
        eprintln!("xor-tok late: ann {ann:.3} pts {pts:.3} pairs {pairs:.3}");
        assert!(ann > pts + 0.10, "net {ann} should beat additive pts {pts} on XOR tokens");
        assert!(pairs > pts + 0.10);
    }

    #[test]
    fn net_learns_a_single_best_agent() {
        // sanity: 1 context, 3 agents, p = [.3,.85,.5]
        let c = AnnCfg { mode: Mode::Reg, h: 8, lr: 0.05, tau0: 0.3, tau_min: 0.02, half: 300.0, replay_k: 0, replay_cap: 1, use_dense: true, use_kw: false };
        let mut r = AnnRouter::new(3, c, None, None, 1);
        let t = neural_mesh::router::Task { id: 0, task_type: None, features: vec!["sql".into(), "db".into()] };
        let mut rng = Rng::new(7);
        let mut hits = 0;
        for i in 0..3000 {
            let a = r.select(&t, &[0, 1, 2], &mut rng);
            let p = [0.3, 0.85, 0.5][a];
            r.update(&t, a, if rng.bernoulli(p) { 1.0 } else { 0.0 });
            if i >= 2500 && a == 1 {
                hits += 1;
            }
        }
        assert!(hits > 400, "hits {hits}");
    }

    #[test]
    fn pg_mode_runs() {
        let c = AnnCfg { mode: Mode::Pg, h: 8, lr: 0.05, tau0: 0.0, tau_min: 0.0, half: 1.0, replay_k: 0, replay_cap: 1, use_dense: true, use_kw: false };
        let env = WorldSpec::Syn(SynKind::Lin).make(1, 1000);
        let r = run_policy(&env, mk_router(&Kind::Ann(c), &env, 1).as_mut(), 1);
        assert!(r.full > 0.3);
    }
}
