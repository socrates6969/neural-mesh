//! G9-A2: WHERE does a critic pay? Sweeps of the critic world around its base cell (frozen critic configs from g9_critic tuning on seeds 1-8).
//! Grid 1: evidence retrieval quality p_ret x base error rate eps. Grid 2: evidence poisoning x second-source availability.
//! Grid 3: verification price (cost of an evidence-tool run in agent calls) x call cost c, at the base cell.
//! Metric: net@c = accuracy - c*(extra agent calls + tool_cost*tool runs), paired difference vs `none`, 95% t-CI over seeds, Holm across the grid cells.
//!
//! Usage: cargo run --release --offline --bin g9_critic_sweep -- [--held 31-60] [--out results/g9_critic_sweep.json]

use rayon::prelude::*;
use serde_json::json;

#[path = "g9_common/mod.rs"]
mod common;
use common::*;

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn arms() -> Vec<Arm> {
    vec![
        arm("none", ArmKind::None),
        arm("exist-flat(weak)", ArmKind::Exist { hier: false, strong: false, cost: 0.1 }),
        arm("self-consistency", ArmKind::Sc { k: 3 }),
        arm("latest", ArmKind::Ev { kind: EvKind::Latest, theta: 0.5, unv_flag: false }),
        arm("agree", ArmKind::Ev { kind: EvKind::Agree, theta: 0.5, unv_flag: false }),
        arm("auto", ArmKind::Auto { theta: 0.5, unv_flag: false, cost: 0.02 }),
    ]
}

fn run_cell(w: &CWorld, seeds: &[u64], widx: u64, arms: &[Arm]) -> Vec<Vec<CRun>> {
    let per: Vec<Vec<CRun>> = seeds
        .par_iter()
        .map(|&s| {
            let inst = build_critic(w, s, widx);
            arms.iter().map(|a| run_critic_arm(&inst, a)).collect()
        })
        .collect();
    (0..arms.len()).map(|ai| per.iter().map(|r| r[ai].clone()).collect()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let held = parse_range(&args, "--held", (31, 60));
    let out = arg(&args, "--out", "results/g9_critic_sweep.json");
    let arms = arms();
    let names: Vec<String> = arms.iter().map(|a| a.name.clone()).collect();
    let mut base = CWorld::base();
    base.n_tasks = 1500;
    let c = 0.05;
    let mut tests: Vec<Test> = vec![];
    let mut cells = vec![];

    // ---- grid 1
    let p_rets = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95];
    let epss = [0.01, 0.03, 0.06, 0.10, 0.20, 0.30];
    println!("GRID 1: net@0.05 diff vs none, rows = eps, cols = p_ret (p_b = p_ret/2, stale docs as base)");
    for (ei, &eps) in epss.iter().enumerate() {
        for (pi, &pr) in p_rets.iter().enumerate() {
            let w = CWorld { name: "sweep1", eps, p_ret: pr, p_b: pr * 0.5, ..base.clone() };
            let m = run_cell(&w, &held, 100 + (ei * 10 + pi) as u64, &arms);
            let none: Vec<f64> = m[0].iter().map(|x| x.net(c, C_TOOL)).collect();
            for (ai, a) in names.iter().enumerate().skip(1) {
                let v: Vec<f64> = m[ai].iter().map(|x| x.net(c, C_TOOL)).collect();
                tests.push(mk_test("G1", &format!("eps{eps}-pret{pr}"), "net@0.05", a, "none", &v, &none));
            }
            cells.push((eps, pr, m));
        }
    }
    holm_by_family(&mut tests);
    for arm_name in ["latest", "agree", "auto", "self-consistency", "exist-flat(weak)"] {
        println!("\n  {arm_name}: diff vs none (net@.05; * = Holm WIN, ! = Holm LOSS)");
        print!("  eps\\p_ret");
        for p in p_rets {
            print!("{:>9}", p);
        }
        println!();
        for &eps in &epss {
            print!("  {eps:<9}");
            for &pr in &p_rets {
                let t = tests.iter().find(|t| t.family == "G1" && t.world == format!("eps{eps}-pret{pr}") && t.treat == arm_name).unwrap();
                let v = verdict(t.adj, t.d.diff.mean);
                print!("{:>8.4}{}", t.d.diff.mean, if v == "WIN" { "*" } else if v == "LOSS" { "!" } else { " " });
            }
            println!();
        }
    }
    // ---- grid 2
    println!("\nGRID 2 (eps 0.15, p_ret .9): poison prob x second source prob p_b -> net@0.05 diff vs none");
    let pois = [0.0, 0.2, 0.4, 0.6, 0.8];
    let pbs = [0.0, 0.5, 1.0];
    let mut t2: Vec<Test> = vec![];
    for (qi, &pz) in pois.iter().enumerate() {
        for (bi, &pb) in pbs.iter().enumerate() {
            let w = CWorld { name: "sweep2", eps: 0.15, p_poison: pz, p_b: pb, ..base.clone() };
            let m = run_cell(&w, &held, 200 + (qi * 10 + bi) as u64, &arms);
            let none: Vec<f64> = m[0].iter().map(|x| x.net(c, C_TOOL)).collect();
            for (ai, a) in names.iter().enumerate().skip(1) {
                let v: Vec<f64> = m[ai].iter().map(|x| x.net(c, C_TOOL)).collect();
                t2.push(mk_test("G2", &format!("poison{pz}-pb{pb}"), "net@0.05", a, "none", &v, &none));
            }
        }
    }
    holm_by_family(&mut t2);
    for arm_name in ["latest", "agree", "auto"] {
        println!("\n  {arm_name}");
        print!("  poison\\p_b");
        for p in pbs {
            print!("{:>9}", p);
        }
        println!();
        for &pz in &pois {
            print!("  {pz:<10}");
            for &pb in &pbs {
                let t = t2.iter().find(|t| t.world == format!("poison{pz}-pb{pb}") && t.treat == arm_name).unwrap();
                let v = verdict(t.adj, t.d.diff.mean);
                print!("{:>8.4}{}", t.d.diff.mean, if v == "WIN" { "*" } else if v == "LOSS" { "!" } else { " " });
            }
            println!();
        }
    }
    // ---- grid 3: price of verification at the base cell (A1)
    println!("\nGRID 3 (A1 base cell): net of `agree` / `latest` / `self-consistency` / `none` for evidence-tool price x call cost");
    let m = run_cell(&CWorld { name: "sweep3", ..base.clone() }, &held, 300, &arms);
    let mut g3 = vec![];
    for &cc in &[0.02, 0.05, 0.10, 0.20] {
        for &tc in &[0.0, 0.1, 0.3, 1.0] {
            let row: Vec<f64> = names.iter().enumerate().map(|(ai, _)| mean(&m[ai].iter().map(|x| x.net(cc, tc)).collect::<Vec<_>>())).collect();
            println!("  c={cc:<5} tool={tc:<4} {}", names.iter().zip(&row).map(|(n, v)| format!("{n}={v:.4}")).collect::<Vec<_>>().join("  "));
            g3.push(json!({"c": cc, "tool_cost": tc, "net": names.iter().zip(&row).map(|(n, v)| json!({"arm": n, "net": v})).collect::<Vec<_>>()}));
        }
    }
    let doc = json!({"held": [held[0], held[held.len() - 1]], "grid1": tests.iter().map(test_json).collect::<Vec<_>>(), "grid2": t2.iter().map(test_json).collect::<Vec<_>>(), "grid3": g3});
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out}");
}
