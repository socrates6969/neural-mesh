//! H22 part 1: native MLP router (g12 `ann`) and the new CASCADE-CORRELATION-style growing net (`casc`, FANN-flavoured) vs keyword /
//! eps-greedy / pooled-thompson / hebbian-robust / warm(hebr) / guarded-safe on the EXISTING keyword-token worlds (sim fixed/random/no-traps,
//! f3 big / big-longtail) at T = 2000 / 6000 / 20000, plus a reproduction check against results/g12_annrouter*.json.
//!
//! Simulation only; measures ROUTING. FANN itself is a C library and is NOT used: its model class (fully-connected MLP, cascade-correlation)
//! is what g12 (MLP) and `Casc2` (cascade) implement natively.
//!
//! Protocol: baselines + g12 `ann` use the SAME configs g12 tuned on seeds 1-8 (hardcoded from the g12 logs: eps (.97,0) low / (.99,0) mid-high,
//! pts gamma 1, warm k 6, ann Reg H24 lr .05 tau .3->.02 half 100 rk3; ann+kw H8 lr .05 low / .15 mid). The NEW cascade arm is tuned on
//! seeds 1-8 ONLY (per regime low / mid; high re-uses mid). Evaluation: held 31-60 (default) or --held 61-90 (confirm, same frozen tuning).
//! Paired 95% t-CI vs the BEST FAIR baseline (best mean of the six, picked per world x regime on the evaluated seeds = conservative), Holm per
//! regime family, TOST +-0.01 reported on the same differences.
//!
//! Usage: h22_existing [--held 31-60] [--out results/h22_existing.json] [--quick]

use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "h22_common/mod.rs"]
mod common;
use common::g12::{AnnCfg, Mode, Regime, WorldSpec};
use common::*;

use neural_mesh::sim::Scenario;
use neural_mesh::stats::summarize;

#[global_allocator]
static GLOBAL: common::g12::alloc_count::Counting = common::g12::alloc_count::Counting;

const FAIR: [&str; 6] = ["keyword", "eps-greedy", "pooled-thompson", "hebbian-robust", "warm(hebr)", "guarded-safe"];

fn worlds() -> Vec<WorldSpec> {
    vec![WorldSpec::Sim(Scenario::Fixed), WorldSpec::Sim(Scenario::Random), WorldSpec::Sim(Scenario::NoTraps), WorldSpec::Big, WorldSpec::BigTail]
}

fn g12_ann(reg: Regime, kw: bool) -> AnnCfg {
    let (h, lr) = if kw {
        (8, if reg == Regime::Low { 0.05 } else { 0.15 })
    } else {
        (24, 0.05)
    };
    AnnCfg { mode: Mode::Reg, h, lr, tau0: 0.3, tau_min: 0.02, half: 100.0, replay_k: 3, replay_cap: 1000, use_dense: true, use_kw: kw }
}

fn eps_for(reg: Regime) -> (f64, f64) {
    if reg == Regime::Low { (0.97, 0.0) } else { (0.99, 0.0) }
}

fn casc_grid() -> Vec<CascCfg> {
    let mut g = vec![];
    for &lr in &[0.05f32, 0.15, 0.4] {
        for &half in &[100.0f32, 500.0] {
            for &rk in &[0usize, 3] {
                for &ge in &[200usize, 600] {
                    for &kw in &[false, true] {
                        g.push(CascCfg { inp: Inp::None, dh: 256, lr, tau0: 0.3, tau_min: 0.02, half, replay_k: rk, grow_every: ge, hmax: 8, use_kw: kw, buf_cap: 500, cands: 3, epochs: 20 });
                    }
                }
            }
        }
    }
    g
}

type Mat = Vec<Vec<Vec<RunRes>>>; // [world][arm][seed]

fn run_matrix(ws: &[WorldSpec], reg: Regime, arms: &[Arm], seeds: &[u64]) -> Mat {
    let items: Vec<(usize, usize)> = (0..ws.len()).flat_map(|w| (0..seeds.len()).map(move |s| (w, s))).collect();
    let cells: Vec<Vec<RunRes>> = items
        .par_iter()
        .map(|&(wi, si)| {
            let seed = seeds[si];
            let env = Old(ws[wi].make(seed, ws[wi].t_for(reg)));
            arms.iter()
                .map(|a| {
                    let mut r = mk_arm(a, &env, seed);
                    run_policy(&env, r.as_mut(), a.var, seed)
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

fn objective(m: &Mat, ai: usize) -> f64 {
    mean_of(&m.iter().map(|w| mean_of(&fulls(&w[ai]))).collect::<Vec<_>>())
}

fn tune_casc(ws: &[WorldSpec], reg: Regime, quick: bool) -> (CascCfg, Value) {
    let sd = if quick { seeds(1, 3) } else { seeds(1, 8) };
    let g = casc_grid();
    let g: Vec<CascCfg> = if quick { g.into_iter().step_by(7).collect() } else { g };
    let arms: Vec<Arm> = g.iter().map(|c| arm("casc", K::Casc(*c), Var::Tok)).collect();
    let t0 = Instant::now();
    let m = run_matrix(ws, reg, &arms, &sd);
    let obj: Vec<f64> = (0..arms.len()).map(|a| objective(&m, a)).collect();
    let best = (0..arms.len()).max_by(|&a, &b| obj[a].partial_cmp(&obj[b]).unwrap()).unwrap();
    println!("[tune casc {}] seeds {}-{}: best {} obj {:.4} [{:.0}s]", reg.label(), sd[0], sd[sd.len() - 1], g[best].show(), obj[best], t0.elapsed().as_secs_f64());
    let mut order: Vec<usize> = (0..arms.len()).collect();
    order.sort_by(|&a, &b| obj[b].partial_cmp(&obj[a]).unwrap());
    let top: Vec<Value> = order.iter().take(5).map(|&a| json!({"cfg": g[a].show(), "objective": obj[a]})).collect();
    for t in &top {
        println!("    casc top: {t}");
    }
    (g[best], json!({"regime": reg.label(), "best": g[best].show(), "objective": obj[best], "top5": top, "grid": g.len()}))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let val = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let (hlo, hhi) = val("--held")
        .map(|s| {
            let mut it = s.split('-');
            (it.next().unwrap().parse::<u64>().unwrap(), it.next().unwrap().parse::<u64>().unwrap())
        })
        .unwrap_or((31, 60));
    let out_path = val("--out").unwrap_or_else(|| "results/h22_existing.json".into());
    let ws = worlds();
    let eval_seeds = if quick { seeds(hlo, hlo + 2) } else { seeds(hlo, hhi) };
    let regs: Vec<Regime> = if quick { vec![Regime::Low] } else { vec![Regime::Low, Regime::Mid, Regime::High] };
    let t_all = Instant::now();
    println!("H22 existing worlds | held {}-{} | RAYON threads {}", eval_seeds[0], eval_seeds[eval_seeds.len() - 1], rayon::current_num_threads());
    let (cl, jl) = tune_casc(&ws, Regime::Low, quick);
    let (cm, jm) = if quick { (cl, jl.clone()) } else { tune_casc(&ws, Regime::Mid, quick) };
    // reproduction reference (g12 json for the same split)
    let g12_json: Option<Value> = {
        let p = if (hlo, hhi) == (31, 60) { "results/g12_annrouter.json" } else if (hlo, hhi) == (61, 90) { "results/g12_annrouter_confirm61-90.json" } else { "" };
        if p.is_empty() { None } else { std::fs::read(p).ok().and_then(|b| serde_json::from_slice(&b).ok()) }
    };
    let mut reg_json = vec![];
    let (mut all_f1c, mut all_f1a) = (vec![], vec![]);
    let mut curves = vec![];
    let mut repro = vec![];
    for (ri, &reg) in regs.iter().enumerate() {
        let cc = if reg == Regime::Low { cl } else { cm };
        let (e0, e1) = eps_for(reg);
        let mut lin = cc;
        lin.hmax = 0;
        let arms: Vec<Arm> = vec![
            arm("keyword", K::Keyword, Var::Tok),
            arm("eps-greedy", K::Eps(e0, e1), Var::Tok),
            arm("pooled-thompson", K::Pts(1.0), Var::Tok),
            arm("hebbian-robust", K::HebR, Var::Tok),
            arm("warm(hebr)", K::Warm(6.0), Var::Tok),
            arm("guarded-safe", K::Safe, Var::Tok),
            arm("ann", K::G12Ann(g12_ann(reg, false)), Var::Tok),
            arm("ann+kw", K::G12Ann(g12_ann(reg, true)), Var::Tok),
            arm("casc", K::Casc(cc), Var::Tok),
            arm("casc-linear-head[ref]", K::Casc(lin), Var::Tok),
        ];
        let t0 = Instant::now();
        let m = run_matrix(&ws, reg, &arms, &eval_seeds);
        let orc: Vec<Vec<RunRes>> = ws.iter().map(|w| eval_seeds.par_iter().map(|&s| oracle_run(&Old(w.make(s, w.t_for(reg))))).collect()).collect();
        println!("[eval {}] done in {:.0}s | casc cfg {}", reg.label(), t0.elapsed().as_secs_f64(), cc.show());
        let (ia, icasc, ilin, ikw) = (6usize, 8usize, 9usize, 7usize);
        let mut f1c: Vec<Test> = vec![];
        let mut f1a: Vec<Test> = vec![];
        let mut worlds_json = vec![];
        println!("{:<24} {:>6} | keyw eps pts hebr warm safe | ann ann+kw casc lin-head | best-fair -> casc diff [CI] | late | oracle", "world", "T");
        for (wi, w) in ws.iter().enumerate() {
            let mf: Vec<f64> = (0..arms.len()).map(|a| mean_of(&fulls(&m[wi][a]))).collect();
            let ml: Vec<f64> = (0..arms.len()).map(|a| mean_of(&lates(&m[wi][a]))).collect();
            let bf = (0..6).max_by(|&a, &b| mf[a].partial_cmp(&mf[b]).unwrap()).unwrap();
            let bl = (0..6).max_by(|&a, &b| ml[a].partial_cmp(&ml[b]).unwrap()).unwrap();
            let (lab, fam) = (w.label(), w.family());
            for (name, ai, vec) in [("casc", icasc, &mut f1c), ("ann", ia, &mut f1a)] {
                vec.push(mk_test(reg.label(), &lab, fam, name, FAIR[bf], "full", &fulls(&m[wi][ai]), &fulls(&m[wi][bf])));
                vec.push(mk_test(reg.label(), &lab, fam, name, FAIR[bl], "late25%", &lates(&m[wi][ai]), &lates(&m[wi][bl])));
            }
            let fx = f1c.len() - 2;
            println!(
                "{:<24} {:>6} | {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} | {:.4} {:.4} {:.4} {:.4} | {}={:.4} -> casc {:+.4} [{:+.4},{:+.4}] late {:+.4} | oracle {:.4}",
                lab, w.t_for(reg), mf[0], mf[1], mf[2], mf[3], mf[4], mf[5], mf[ia], mf[ikw], mf[icasc], mf[ilin], FAIR[bf], mf[bf], f1c[fx].d, f1c[fx].lo, f1c[fx].hi, f1c[fx + 1].d, mean_of(&fulls(&orc[wi]))
            );
            let hu: f64 = 0.0;
            let _ = hu;
            let mut arms_j = serde_json::Map::new();
            for (a, ar) in arms.iter().enumerate() {
                let s = summarize(&fulls(&m[wi][a]));
                arms_j.insert(ar.name.clone(), json!({"full": s.mean, "full_ci": [s.lo, s.hi], "late25": ml[a]}));
            }
            worlds_json.push(json!({"world": lab, "family": fam, "T": w.t_for(reg), "oracle_full": mean_of(&fulls(&orc[wi])), "best_fair_full": FAIR[bf], "best_fair_late": FAIR[bl], "arms": arms_j}));
            // reproduction vs g12
            if let Some(g) = &g12_json {
                let gw = &g["regimes"][ri]["worlds"][wi];
                if gw["world"].as_str() == Some(&lab) {
                    for name in ["keyword", "eps-greedy", "pooled-thompson", "hebbian-robust", "warm(hebr)", "guarded-safe", "ann", "ann+kw"] {
                        let ai = arms.iter().position(|a| a.name == name).unwrap();
                        let gv = gw["arms"][name]["full"].as_f64().unwrap();
                        repro.push(json!({"regime": reg.label(), "world": lab, "arm": name, "mine": mf[ai], "g12": gv, "absdiff": (mf[ai] - gv).abs()}));
                    }
                }
            }
            if reg == Regime::High {
                for (name, ai) in [("casc", icasc), ("ann", ia)] {
                    let (par, nn, ov, diffs, los, am, rm) = parity_curve(&m[wi][ai], &m[wi][bl], 500);
                    println!("    curve {name} vs {} (late-best fair): tasks-to-parity {:?}, sustained diff>=0 {:?}, sustained CI-lo>0 {:?}", FAIR[bl], par, nn, ov);
                    curves.push(json!({"world": lab, "arm": name, "ref": FAIR[bl], "window": 500, "tasks_to_parity": par, "tasks_to_nonneg": nn, "tasks_to_sig_overtake": ov, "arm_window_means": am, "ref_window_means": rm, "diff_window": diffs, "ci_lo_window": los}));
                }
            }
        }
        holm_family(&mut f1c);
        holm_family(&mut f1a);
        for (lab, ts) in [("casc", &f1c), ("ann", &f1a)] {
            println!("  F1 {lab} vs best fair (Holm over {} tests, {}):", ts.len(), reg.label());
            for t in ts.iter() {
                println!("    {:<24} {:<8} vs {:<16} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.3} {} {}", t.world, t.metric, t.vs, t.d, t.lo, t.hi, t.wins, t.losses, t.adj, verdict(t.adj, t.d), if tost_ok(t) { "TOST+-.01 ok" } else { "" });
            }
        }
        reg_json.push(json!({"regime": reg.label(), "casc_cfg": cc.show(), "worlds": worlds_json, "F1_casc": f1c.iter().map(test_json).collect::<Vec<_>>(), "F1_ann": f1a.iter().map(test_json).collect::<Vec<_>>()}));
        all_f1c.extend(f1c);
        all_f1a.extend(f1a);
    }
    if !repro.is_empty() {
        let mx = repro.iter().map(|r| r["absdiff"].as_f64().unwrap()).fold(0.0, f64::max);
        let bad = repro.iter().filter(|r| r["absdiff"].as_f64().unwrap() > 1e-9).count();
        println!("REPRODUCTION vs g12 json: {} cells, max |diff| {:.3e}, cells differing > 1e-9: {}", repro.len(), mx, bad);
    }
    let out = json!({"experiment": "h22_existing", "held": [eval_seeds[0], eval_seeds[eval_seeds.len() - 1]], "note": "simulation only; routing", "tune_casc": [jl, jm],
                     "regimes": reg_json, "curves": curves, "reproduction_vs_g12": repro, "runtime_s": t_all.elapsed().as_secs_f64()});
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote {out_path} ({:.0}s)", t_all.elapsed().as_secs_f64());
}
