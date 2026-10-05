//! H11-pipe (R5-C TOKEN EFFICIENCY): cost levers in the multi-step research-session loop (world P = copy of the g18 orchestration world, two independent
//! generators A "pipelines" / B "latent" + 5 controls) with a TOKEN-LEVEL SYNTHETIC cost model (h11_common/mod.rs, pworld.rs).
//!
//! Protocol (g18/h3): tune on seeds 1-8 over all 15 worlds (verify-gate tau, retrieval-gate theta, 1-D grids, objective = utility at mu .03 in TE), held 31-60,
//! confirm 61-90 with the SAME frozen procedure; paired 95% t-CI over 30 seeds; TOST +-0.01 on quality (graded step quality q and all-steps-correct rate);
//! Holm per family. Costs are SYNTHETIC token-equivalents (TE), NOT a real bill.
//!
//! Usage: cargo run --release --offline --bin h11_pipe -- [--held 31-60] [--out results/h11_pipe.json] [--sens]
//!        cargo test --release --offline --bin h11_pipe

use std::collections::BTreeMap;
use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "h11_common/mod.rs"]
mod h11;
use h11::pworld::*;
use h11::*;

static MU_CELL: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
/// primary exchange rate (quality points per extra reference step-call of cost): 0.03 default (g18's util(.03)); `--mu 0.1` reruns the whole tune/eval at the strict rate
fn mu() -> f64 {
    *MU_CELL.get().unwrap_or(&0.03)
}
fn mu_alt() -> f64 {
    if (mu() - 0.03).abs() < 1e-9 { 0.10 } else { 0.03 }
}
const RVS: [f64; 9] = [0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0];
const THETAS: [f64; 6] = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7];
const NMIN: f64 = 3.0;
/// pre-declared inclusion rule for a tuned lever in the stack: tuned objective must beat the untuned base by this much
const INCL: f64 = 0.001;

#[derive(Clone, Copy, Debug)]
struct Tuned {
    rv: f64,
    theta: f64,
    use_vg: bool,
    use_rg: bool,
}

fn stack(t: &Tuned) -> PCfg {
    PCfg {
        rt: Rt::CA { mu: mu() },
        rmode: if t.use_rg { RMode::Gate { theta: t.theta } } else { RMode::Always },
        vmode: if t.use_vg { VMode::Gate { rv: t.rv, nmin: NMIN } } else { VMode::Always },
        ..PCfg::l3()
    }
}
fn arm_list(t: &Tuned) -> Vec<(String, PCfg)> {
    let l3 = PCfg::l3();
    let s = stack(t);
    let mut v: Vec<(String, PCfg)> = vec![
        ("L2".into(), PCfg::l2()),
        ("L3".into(), l3),
        ("L4".into(), PCfg::l4(1.0, 0.1)),
        ("L3+Vg".into(), PCfg { vmode: VMode::Gate { rv: t.rv, nmin: NMIN }, ..l3 }),
        ("L3+BO3".into(), PCfg { early_stop: false, n_bo: 3, ..l3 }),
        ("L3+Rg".into(), PCfg { rmode: RMode::Gate { theta: t.theta }, ..l3 }),
        ("L3+Rskip(oracle)".into(), PCfg { rmode: RMode::OracleSkip { dmin: 0.1 }, ..l3 }),
        ("L3+C50k.3".into(), PCfg { gamma: 0.5, kappa: 0.3, ..l3 }),
        ("L3+C50k1".into(), PCfg { gamma: 0.5, kappa: 1.0, ..l3 }),
        ("L3+C25k.3".into(), PCfg { gamma: 0.25, kappa: 0.3, ..l3 }),
        ("L3+C25k1".into(), PCfg { gamma: 0.25, kappa: 1.0, ..l3 }),
        ("L3+CA0".into(), PCfg { rt: Rt::CA { mu: 0.0 }, ..l3 }),
        ("L3+CA".into(), PCfg { rt: Rt::CA { mu: mu() }, ..l3 }),
        ("L3+casc".into(), PCfg { rt: Rt::Cascade, ..l3 }),
        ("L2+CA".into(), PCfg { rt: Rt::CA { mu: mu() }, ..PCfg::l2() }),
        ("L4+CA".into(), PCfg { rt: Rt::CA { mu: mu() }, ..PCfg::l4(1.0, 0.1) }),
        ("L3+CA01".into(), PCfg { rt: Rt::CA { mu: 0.01 }, ..l3 }),
        ("L3+CA10".into(), PCfg { rt: Rt::CA { mu: 0.10 }, ..l3 }),
        ("or-util10".into(), PCfg { rt: Rt::OracleUtil { mu: 0.10 }, vmode: VMode::Never, ..l3 }),
        ("or-q".into(), PCfg { rt: Rt::OracleQ, vmode: VMode::Never, ..l3 }),
        ("or-cheap".into(), PCfg { rt: Rt::OracleCheap, vmode: VMode::Never, ..l3 }),
        ("or-util".into(), PCfg { rt: Rt::OracleUtil { mu: mu() }, vmode: VMode::Never, ..l3 }),
        ("S".into(), s),
        ("S-Vg".into(), PCfg { vmode: VMode::Always, ..s }),
        ("S-Rg".into(), PCfg { rmode: RMode::Always, ..s }),
        ("S-CA".into(), PCfg { rt: Rt::Warm(1.0), ..s }),
        ("S+C50k.3".into(), PCfg { gamma: 0.5, kappa: 0.3, ..s }),
    ];
    v.dedup_by(|a, b| a.0 == b.0);
    v
}

/// run every arm on every (world, seed); result [world][arm][seed]
fn eval(ws: &[WorldCfg], seeds: &[u64], arms: &[(String, PCfg)]) -> Vec<Vec<Vec<PSum>>> {
    let jobs: Vec<(usize, u64)> = (0..ws.len()).flat_map(|wi| seeds.iter().map(move |&s| (wi, s))).collect();
    let outs: Vec<Vec<PSum>> = jobs
        .par_iter()
        .map(|&(wi, s)| {
            let inst = build(&ws[wi], s, wi as u64);
            arms.iter().map(|(_, c)| run_p(&ws[wi], &inst, c, s)).collect()
        })
        .collect();
    let mut res: Vec<Vec<Vec<PSum>>> = vec![vec![vec![]; arms.len()]; ws.len()];
    for (j, (wi, _)) in jobs.iter().enumerate() {
        for (ai, ps) in outs[j].iter().enumerate() {
            res[*wi][ai].push(ps.clone());
        }
    }
    res
}

fn col(v: &[PSum], f: impl Fn(&PSum) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}
fn cost_of(s: &PSum, p: &Prices) -> f64 {
    cost_te(&s.res, p)
}

fn tune(ws: &[WorldCfg], seeds: &[u64]) -> (Tuned, Value) {
    let l3 = PCfg::l3();
    let mut arms: Vec<(String, PCfg)> = vec![("L3".into(), l3)];
    for &t in &RVS {
        arms.push((format!("Vg{t}"), PCfg { vmode: VMode::Gate { rv: t, nmin: NMIN }, ..l3 }));
    }
    for &th in &THETAS {
        arms.push((format!("Rg{th}"), PCfg { rmode: RMode::Gate { theta: th }, ..l3 }));
    }
    let r = eval(ws, seeds, &arms);
    let p = Prices::base();
    let obj = |ai: usize| -> (f64, f64, f64) {
        let (mut u, mut q, mut c) = (0.0, 0.0, 0.0);
        for wi in 0..ws.len() {
            u += mean(&col(&r[wi][ai], |s| util(s, &p, mu())));
            q += mean(&col(&r[wi][ai], |s| s.q));
            c += mean(&col(&r[wi][ai], |s| cost_of(s, &p)));
        }
        let n = ws.len() as f64;
        (u / n, q / n, c / n)
    };
    let (base_u, base_q, base_c) = obj(0);
    println!("TUNING (seeds {}-{}, all 15 worlds, mean util(mu={}) in TE):  base L3: util {base_u:.4} q {base_q:.4} cost {base_c:.0}", seeds[0], seeds[seeds.len() - 1], mu());
    let mut tj = vec![json!({"arm": "L3", "util": base_u, "q": base_q, "cost": base_c})];
    let mut best_v = (f64::MIN, 0.0);
    let mut best_r = (f64::MIN, 0.0);
    for (i, &t) in RVS.iter().enumerate() {
        let (u, q, c) = obj(1 + i);
        println!("  verify-gate R {t:<5} util {u:.4} q {q:.4} cost {c:.0}");
        tj.push(json!({"arm": format!("Vg{t}"), "util": u, "q": q, "cost": c}));
        if u > best_v.0 {
            best_v = (u, t);
        }
    }
    for (i, &th) in THETAS.iter().enumerate() {
        let (u, q, c) = obj(1 + RVS.len() + i);
        println!("  retr-gate theta {th:<5} util {u:.4} q {q:.4} cost {c:.0}");
        tj.push(json!({"arm": format!("Rg{th}"), "util": u, "q": q, "cost": c}));
        if u > best_r.0 {
            best_r = (u, th);
        }
    }
    let t = Tuned { rv: best_v.1, theta: best_r.1, use_vg: best_v.0 - base_u > INCL, use_rg: best_r.0 - base_u > INCL };
    println!("  => R* {} (gain {:+.4}; in stack: {}), theta* {} (gain {:+.4}; in stack: {})", t.rv, best_v.0 - base_u, t.use_vg, t.theta, best_r.0 - base_u, t.use_rg);
    (t, json!({"grid": tj, "rv": t.rv, "theta": t.theta, "use_vg": t.use_vg, "use_rg": t.use_rg, "incl_rule": INCL}))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let held = parse_range(&args, "--held", (31, 60));
    let confirm = held[0] >= 61;
    let mu_tag = if (arg(&args, "--mu", "0.03").parse::<f64>().unwrap() - 0.03).abs() < 1e-9 { String::new() } else { format!("_mu{}", arg(&args, "--mu", "0.03")) };
    let out_path = arg(&args, "--out", &if confirm { format!("results/h11_pipe{mu_tag}_confirm61-90.json") } else { format!("results/h11_pipe{mu_tag}.json") });
    let sens = args.iter().any(|a| a == "--sens");
    let mu_arg: f64 = arg(&args, "--mu", "0.03").parse().unwrap();
    MU_CELL.set(mu_arg).unwrap();
    let t0 = Instant::now();
    let ws = worlds();
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let (tuned, tune_json) = tune(&ws, &tune_seeds);
    println!("tuning done in {:.0}s\n", t0.elapsed().as_secs_f64());
    let arms = arm_list(&tuned);
    let names: Vec<String> = arms.iter().map(|a| a.0.clone()).collect();
    let ai = |n: &str| names.iter().position(|x| x == n).unwrap_or_else(|| panic!("arm {n}"));
    let r = eval(&ws, &held, &arms);
    println!("evaluated {} arms x {} worlds x {} seeds in {:.0}s\n", arms.len(), ws.len(), held.len(), t0.elapsed().as_secs_f64());
    let p0 = Prices::base();
    let groups: [(&str, Vec<usize>); 3] = [("A", (0..6).collect()), ("B", (6..10).collect()), ("ctl", (10..15).collect())];
    let mut jout = serde_json::Map::new();
    jout.insert("tuning".into(), tune_json);
    jout.insert("held".into(), json!(format!("{}-{}", held[0], held[held.len() - 1])));
    jout.insert("label".into(), json!("SYNTHETIC cost accounting (TE); seeded simulation; no real token bills"));

    // ------------------------------------------------------------------ 1) base numbers per world (arm means)
    println!("=== 1. COST DECOMPOSITION of the always-verify session L3 (TE per task, default prices) and the stack S; share of the L3 cost ===");
    let mut dec_j = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let mr = |arm: &str| -> Res {
            let v = &r[wi][ai(arm)];
            let mut acc = Res::default();
            for s in v {
                acc.add(&s.res);
            }
            acc.scaled(1.0 / v.len() as f64)
        };
        let (rl3, rs) = (mr("L3"), mr("S"));
        let (d3, ds) = (decompose(&rl3, &p0), decompose(&rs, &p0));
        let (c3, cs) = (cost_te(&rl3, &p0), cost_te(&rs, &p0));
        let sh: Vec<String> = d3.iter().map(|(n, x)| format!("{} {:.0}%", n, 100.0 * x / c3)).collect();
        println!("  {:<19} L3 {:>7.0} TE [{}]  | S {:>7.0} TE ({:+.1}%)", w.name, c3, sh.join(", "), cs, 100.0 * (cs / c3 - 1.0));
        dec_j.push(json!({"world": w.name, "L3": d3.iter().map(|(n, x)| json!([n, x])).collect::<Vec<_>>(), "S": ds.iter().map(|(n, x)| json!([n, x])).collect::<Vec<_>>(), "L3_total": c3, "S_total": cs,
            "L3_res": rl3.arr(), "S_res": rs.arr()}));
    }
    jout.insert("decomposition".into(), json!(dec_j));

    // ------------------------------------------------------------------ 2) every arm vs L3: quality TOST, cost, util; per world and per group
    println!("\n=== 2. every arm vs the always-verify session L3 (graded quality q, all-correct ok; cost in TE at default prices; util(.03)); held {}-{} ===", held[0], held[held.len() - 1]);
    struct Row {
        arm: String,
        world: String,
        dq: Cmp,
        dok: Cmp,
        dc: Cmp,
        du: Cmp,
        du_hi: Cmp,
        c_base: f64,
    }
    let mut rows: Vec<Row> = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let l3 = &r[wi][ai("L3")];
        for (a, name) in names.iter().enumerate() {
            if name == "L3" {
                continue;
            }
            let v = &r[wi][a];
            rows.push(Row {
                arm: name.clone(),
                world: w.name.to_string(),
                dq: cmp(&col(v, |s| s.q), &col(l3, |s| s.q), 0.01),
                dok: cmp(&col(v, |s| s.ok), &col(l3, |s| s.ok), 0.01),
                dc: cmp(&col(v, |s| cost_of(s, &p0)), &col(l3, |s| cost_of(s, &p0)), 0.0),
                du: cmp(&col(v, |s| util(s, &p0, mu())), &col(l3, |s| util(s, &p0, mu())), 0.0),
                du_hi: cmp(&col(v, |s| util(s, &p0, mu_alt())), &col(l3, |s| util(s, &p0, mu_alt())), 0.0),
                c_base: mean(&col(l3, |s| cost_of(s, &p0))),
            });
        }
    }
    // Holm families: quality-equivalence (TOST) over all rows; cost; util
    let h_q = holm(&rows.iter().map(|x| x.dq.tost_p).collect::<Vec<_>>());
    let h_c = holm(&rows.iter().map(|x| x.dc.p).collect::<Vec<_>>());
    let h_u = holm(&rows.iter().map(|x| x.du.p).collect::<Vec<_>>());
    println!("legend: dq = q(arm)-q(L3); EQ = number of worlds TOST-equivalent within +-.01 on q (Holm over {} tests | raw p<.05 | NI = non-inferior: CI lower bound > -.01); cost cut% = 1 - cost(arm)/cost(L3) (Holm WIN = cheaper); dU = util(.03) diff", rows.len());
    println!("{:<18}{:^46}{:^46}{:^46}", "arm", "generator A (6 worlds)", "generator B (4 worlds)", "controls (5)");
    println!("{:<18}{:^46}{:^46}{:^46}", "", "dq  EQ  dok  cost-cut%  W/L  dU(mu) dU(alt)", "dq  EQ  dok  cost-cut%  W/L  dU(mu) dU(alt)", "dq  EQ  dok  cost-cut%  W/L  dU(mu) dU(alt)");
    let mut grp_j = vec![];
    for name in names.iter().filter(|n| *n != "L3") {
        let mut line = format!("{:<18}", name);
        let mut gj = serde_json::Map::new();
        for (gn, idx) in &groups {
            let sel: Vec<usize> = rows.iter().enumerate().filter(|(_, x)| &x.arm == name && idx.iter().any(|&wi| ws[wi].name == x.world)).map(|(i, _)| i).collect();
            let n = sel.len() as f64;
            let dq = sel.iter().map(|&i| rows[i].dq.diff).sum::<f64>() / n;
            let eq = sel.iter().filter(|&&i| h_q[i] < 0.05).count();
            let eq_raw = sel.iter().filter(|&&i| rows[i].dq.tost_p < 0.05).count();
            let ni = sel.iter().filter(|&&i| rows[i].dq.lo > -0.01).count();
            let eqok = sel.iter().filter(|&&i| rows[i].dok.tost_p < 0.05).count();
            let dok = sel.iter().map(|&i| rows[i].dok.diff).sum::<f64>() / n;
            let cc = sel.iter().map(|&i| -rows[i].dc.diff / rows[i].c_base).sum::<f64>() / n * 100.0;
            let wcnt = sel.iter().filter(|&&i| h_c[i] < 0.05 && rows[i].dc.diff < 0.0).count();
            let lcnt = sel.iter().filter(|&&i| h_c[i] < 0.05 && rows[i].dc.diff > 0.0).count();
            let du = sel.iter().map(|&i| rows[i].du.diff).sum::<f64>() / n;
            let du_hi = sel.iter().map(|&i| rows[i].du_hi.diff).sum::<f64>() / n;
            line += &format!("{:+.4} {}|{}|{}/{} {:+.3}({}) {:>6.1}% {}/{} {:+.4} {:+.4}   ", dq, eq, eq_raw, ni, sel.len(), dok, eqok, cc, wcnt, lcnt, du, du_hi);
            gj.insert(gn.to_string(), json!({"dq": dq, "eq_q_holm": eq, "eq_q_raw": eq_raw, "ni_q": ni, "eq_ok": eqok, "dok": dok, "cost_cut_pct": cc, "cost_wins": wcnt, "cost_losses": lcnt, "dU03": du, "dU10": du_hi, "n": sel.len()}));
        }
        println!("{line}");
        grp_j.push(json!({"arm": name, "groups": Value::Object(gj)}));
    }
    jout.insert("groups_vs_L3".into(), json!(grp_j));
    let rows_j: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(i, x)| {
            json!({"arm": x.arm, "world": x.world, "dq": x.dq.diff, "dq_ci": [x.dq.lo, x.dq.hi], "dq_tost_p": x.dq.tost_p, "dq_tost_holm": h_q[i], "dok": x.dok.diff, "dok_ci": [x.dok.lo, x.dok.hi],
                   "dcost": x.dc.diff, "dcost_ci": [x.dc.lo, x.dc.hi], "dcost_holm": h_c[i], "cost_L3": x.c_base, "dU03": x.du.diff, "dU03_ci": [x.du.lo, x.du.hi], "dU03_holm": h_u[i], "dU10": x.du_hi.diff})
        })
        .collect();
    jout.insert("rows_vs_L3".into(), json!(rows_j));

    // ------------------------------------------------------------------ 3) key arms per world
    println!("\n=== 3. key arms per world (mean over seeds): q | ok | cost TE | util(.03) ===");
    let key = ["L2", "L3", "L4", "L3+Vg", "L3+CA", "S", "or-util"];
    print!("{:<19}", "world");
    for k in key {
        print!("{:^34}", k);
    }
    println!();
    for (wi, w) in ws.iter().enumerate() {
        print!("{:<19}", w.name);
        for k in key {
            let v = &r[wi][ai(k)];
            print!("{:.3} {:.3} {:>6.0} {:.4}   ", mean(&col(v, |s| s.q)), mean(&col(v, |s| s.ok)), mean(&col(v, |s| cost_of(s, &p0))), mean(&col(v, |s| util(s, &p0, mu()))));
        }
        println!();
    }

    // ------------------------------------------------------------------ 4) best fair baseline (per world by held-out util) vs S
    println!("\n=== 4. stack S vs the best fair cost-conscious incumbent per world (best of L2/L3/L4 by held-out util(.03); conservative) ===");
    let fair = ["L2", "L3", "L4"];
    let mut bf_rows = vec![];
    let mut ps = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let best = *fair.iter().max_by(|a, b| mean(&col(&r[wi][ai(a)], |s| util(s, &p0, mu()))).partial_cmp(&mean(&col(&r[wi][ai(b)], |s| util(s, &p0, mu())))).unwrap()).unwrap();
        let (s, b) = (&r[wi][ai("S")], &r[wi][ai(best)]);
        let du = cmp(&col(s, |x| util(x, &p0, mu())), &col(b, |x| util(x, &p0, mu())), 0.0);
        let dq = cmp(&col(s, |x| x.q), &col(b, |x| x.q), 0.01);
        let dc = cmp(&col(s, |x| cost_of(x, &p0)), &col(b, |x| cost_of(x, &p0)), 0.0);
        ps.push(du.p);
        bf_rows.push((w.name, best, du, dq, dc, mean(&col(b, |x| cost_of(x, &p0)))));
    }
    let hb = holm(&ps);
    let mut bfj = vec![];
    for (i, (wn, best, du, dq, dc, cb)) in bf_rows.iter().enumerate() {
        println!("  {:<19} best fair {:<3} | dU(.03) {:+.4} [{:+.4},{:+.4}] {} | dq {:+.4} [{:+.4},{:+.4}] tost_p {:.3} | cost {:+.1}% {}", wn, best, du.diff, du.lo, du.hi, mark(hb[i], du.diff), dq.diff, dq.lo, dq.hi, dq.tost_p, 100.0 * dc.diff / cb, mark(holm(&[dc.p])[0], -dc.diff));
        bfj.push(json!({"world": wn, "best_fair": best, "dU03": du.diff, "ci": [du.lo, du.hi], "holm": hb[i], "dq": dq.diff, "dq_ci": [dq.lo, dq.hi], "dq_tost_p": dq.tost_p, "cost_pct": 100.0 * dc.diff / cb}));
    }
    jout.insert("S_vs_best_fair".into(), json!(bfj));

    // ------------------------------------------------------------------ 5) lever contribution (leave-one-out of the stack) and single levers
    println!("\n=== 5. which lever contributes what: S minus one lever (S-X): dq and cost change of REMOVING the lever, mean over treatment worlds A+B (10) and controls (5) ===");
    let tw: Vec<usize> = (0..10).collect();
    let cw: Vec<usize> = (10..15).collect();
    let agg = |arm: &str, base: &str, idx: &[usize]| -> (f64, f64, f64) {
        let (mut dq, mut dc, mut du) = (0.0, 0.0, 0.0);
        for &wi in idx {
            let (a, b) = (&r[wi][ai(arm)], &r[wi][ai(base)]);
            dq += mean(&col(a, |s| s.q)) - mean(&col(b, |s| s.q));
            let (ca, cb) = (mean(&col(a, |s| cost_of(s, &p0))), mean(&col(b, |s| cost_of(s, &p0))));
            dc += (ca - cb) / cb * 100.0;
            du += mean(&col(a, |s| util(s, &p0, mu()))) - mean(&col(b, |s| util(s, &p0, mu())));
        }
        let n = idx.len() as f64;
        (dq / n, dc / n, du / n)
    };
    let mut loo_j = vec![];
    for (arm, base) in [("S-Vg", "S"), ("S-Rg", "S"), ("S-CA", "S"), ("S+C50k.3", "S"), ("S", "L3"), ("L4", "L3"), ("L2", "L3"), ("L3+Vg", "L3"), ("L3+CA", "L3"), ("L3+CA0", "L3"), ("L3+CA", "L3+CA0"), ("L3+BO3", "L3"), ("L3+Rg", "L3"), ("L3+Rskip(oracle)", "L3")] {
        let (a, b) = (agg(arm, base, &tw), agg(arm, base, &cw));
        println!("  {:<18} vs {:<7} | treat: dq {:+.4} cost {:+.1}% dU {:+.4} | controls: dq {:+.4} cost {:+.1}% dU {:+.4}", arm, base, a.0, a.1, a.2, b.0, b.1, b.2);
        loo_j.push(json!({"arm": arm, "vs": base, "treat": {"dq": a.0, "cost_pct": a.1, "dU": a.2}, "ctl": {"dq": b.0, "cost_pct": b.1, "dU": b.2}}));
    }
    jout.insert("lever_contrib".into(), json!(loo_j));

    // ------------------------------------------------------------------ 6) Pareto (cost, quality) for pipe-main, lat-main and the mean over treatment worlds
    println!("\n=== 6. Pareto of quality vs cost (default prices; * = non-dominated within the arm set) ===");
    let mut par_j = vec![];
    for (label, idx) in [("pipe-main", vec![0usize]), ("lat-main", vec![6]), ("treatment mean (10)", tw.clone()), ("controls mean (5)", cw.clone())] {
        let pts: Vec<(String, f64, f64)> = names
            .iter()
            .enumerate()
            .map(|(a, n)| {
                let (mut c, mut q) = (0.0, 0.0);
                for &wi in &idx {
                    c += mean(&col(&r[wi][a], |s| cost_of(s, &p0)));
                    q += mean(&col(&r[wi][a], |s| s.q));
                }
                (n.clone(), c / idx.len() as f64, q / idx.len() as f64)
            })
            .collect();
        println!("  [{label}]");
        let mut sorted = pts.clone();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut pj = vec![];
        for p in &sorted {
            let dom = pts.iter().any(|o| (o.1 <= p.1 && o.2 > p.2 + 1e-9) || (o.1 < p.1 - 1e-9 && o.2 >= p.2));
            println!("    {:<18} cost {:>7.0} TE  q {:.4} {}", p.0, p.1, p.2, if dom { "" } else { "*" });
            pj.push(json!({"arm": p.0, "cost": p.1, "q": p.2, "frontier": !dom}));
        }
        par_j.push(json!({"set": label, "points": pj}));
    }
    jout.insert("pareto".into(), json!(par_j));

    // ------------------------------------------------------------------ 7) price sensitivity
    println!("\n=== 7. price sensitivity: cost of each arm vs L3 under other price vectors (treatment worlds, mean %; quality unchanged except where prices enter the RUN, i.e. CA/oracle) ===");
    let scen = scenarios(false);
    let show = ["L2", "L4", "L3+Vg", "L3+BO3", "L3+C50k.3", "L3+CA", "S", "S+C50k.3"];
    print!("{:<34}", "scenario");
    for s in show {
        print!("{:>11}", s);
    }
    println!("{:>9}", "best-util");
    let mut sc_j = vec![];
    for (sn, sp) in &scen {
        print!("{:<34}", sn);
        let mut row = vec![];
        for s in show {
            let mut acc = 0.0;
            for &wi in &tw {
                let (a, b) = (mean(&col(&r[wi][ai(s)], |x| cost_te(&x.res, sp))), mean(&col(&r[wi][ai("L3")], |x| cost_te(&x.res, sp))));
                acc += (a - b) / b * 100.0;
            }
            acc /= tw.len() as f64;
            print!("{:>10.1}%", acc);
            row.push(json!([s, acc]));
        }
        let cand = ["L2", "L3", "L4", "S"];
        let bu = cand
            .iter()
            .max_by(|a, b| {
                let f = |n: &str| tw.iter().map(|&wi| mean(&col(&r[wi][ai(n)], |x| util(x, sp, mu()))) ).sum::<f64>();
                f(a).partial_cmp(&f(b)).unwrap()
            })
            .unwrap();
        println!("{:>9}", bu);
        sc_j.push(json!({"scenario": sn, "cost_pct_vs_L3": row, "best_util_of_L2_L3_L4_S": bu}));
    }
    jout.insert("price_sens".into(), json!(sc_j));
    // utility exchange-rate sensitivity: which of {L2, L3, L4, S} maximises utility at mu
    println!("\n  exchange rate mu (quality points per extra reference step-call of cost): util by arm, treatment worlds mean");
    let mut mu_j = vec![];
    for mu in [0.0, 0.01, 0.03, 0.1, 0.3] {
        let u: Vec<(String, f64)> = ["L2", "L3", "L4", "L3+Vg", "S"].iter().map(|n| (n.to_string(), tw.iter().map(|&wi| mean(&col(&r[wi][ai(n)], |x| util(x, &p0, mu)))).sum::<f64>() / tw.len() as f64)).collect();
        println!("    mu {:<5} {}", mu, u.iter().map(|(n, x)| format!("{n} {x:.4}")).collect::<Vec<_>>().join("  "));
        mu_j.push(json!({"mu": mu, "util": u}));
    }
    jout.insert("mu_sens".into(), json!(mu_j));

    // ------------------------------------------------------------------ 8) cost-aware routing vs oracles (gap closed)
    println!("\n=== 8. cost-aware routing: util(.03) of Warm(L3) / CA0 (price-blind own-table router) / CA / cascade / hindsight oracles; 'gap closed' = (U-U_L3)/(U_orUtil-U_L3) ===");
    let mut ca_j = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let u = |n: &str| mean(&col(&r[wi][ai(n)], |x| util(x, &p0, mu())));
        let (u3, uo) = (u("L3"), u("or-util"));
        let gc = |n: &str| if (uo - u3).abs() > 1e-6 { (u(n) - u3) / (uo - u3) } else { f64::NAN };
        println!("  {:<19} L3 {:.4} CA0 {:+.4} CA {:+.4} casc {:+.4} | or-q {:+.4} or-cheap {:+.4} or-util {:+.4} | gap closed: CA {:.2} CA0 {:.2}", w.name, u3, u("L3+CA0") - u3, u("L3+CA") - u3, u("L3+casc") - u3, u("or-q") - u3, u("or-cheap") - u3, uo - u3, gc("L3+CA"), gc("L3+CA0"));
        ca_j.push(json!({"world": w.name, "U_L3": u3, "dU": {"CA0": u("L3+CA0") - u3, "CA": u("L3+CA") - u3, "casc": u("L3+casc") - u3, "or-q": u("or-q") - u3, "or-cheap": u("or-cheap") - u3, "or-util": uo - u3}, "gap_closed_CA": jnum(gc("L3+CA"))}));
    }
    jout.insert("cost_aware".into(), json!(ca_j));
    println!("
  cost-aware routing frontier (router mu_c vs L3, mean over treatment worlds | controls): dq, cost %, dU(.03), dU(.10); or-util rows use the matching mu");
    let mut fr_j = vec![];
    for (name, lab) in [("L3+CA0", "mu_c=0 (price-blind own table)"), ("L3+CA01", "mu_c=.01"), ("L3+CA", "mu_c=.03"), ("L3+CA10", "mu_c=.10"), ("or-util", "oracle util(mu=.03)"), ("or-util10", "oracle util(mu=.10)"), ("or-q", "oracle quality")] {
        let mut cells = vec![];
        let mut cj = serde_json::Map::new();
        for (gn, idx) in [("treat", &tw), ("ctl", &cw)] {
            let (mut dq, mut dc, mut u3, mut u10) = (0.0, 0.0, 0.0, 0.0);
            for &wi in idx.iter() {
                let (a, b) = (&r[wi][ai(name)], &r[wi][ai("L3")]);
                dq += mean(&col(a, |s| s.q)) - mean(&col(b, |s| s.q));
                let (ca, cb) = (mean(&col(a, |s| cost_of(s, &p0))), mean(&col(b, |s| cost_of(s, &p0))));
                dc += (ca - cb) / cb * 100.0;
                u3 += mean(&col(a, |s| util(s, &p0, 0.03))) - mean(&col(b, |s| util(s, &p0, 0.03)));
                u10 += mean(&col(a, |s| util(s, &p0, 0.10))) - mean(&col(b, |s| util(s, &p0, 0.10)));
            }
            let n = idx.len() as f64;
            cells.push(format!("{gn}: dq {:+.4} cost {:+6.1}% dU03 {:+.4} dU10 {:+.4}", dq / n, dc / n, u3 / n, u10 / n));
            cj.insert(gn.to_string(), json!({"dq": dq / n, "cost_pct": dc / n, "dU03": u3 / n, "dU10": u10 / n}));
        }
        println!("    {:<30} {} | {}", lab, cells[0], cells[1]);
        fr_j.push(json!({"arm": name, "label": lab, "groups": Value::Object(cj)}));
    }
    jout.insert("ca_frontier".into(), json!(fr_j));

    // ------------------------------------------------------------------ 9) price-structure sensitivity worlds
    if sens {
        println!("\n=== 9. price-structure sensitivity (held {}-{}): util(.03) minus L3 ===", held[0], held[held.len() - 1]);
        let sws = worlds_price_sens();
        let sarms: Vec<(String, PCfg)> = arm_list(&tuned).into_iter().filter(|(n, _)| ["L3", "L3+CA0", "L3+CA", "L3+casc", "or-q", "or-cheap", "or-util", "S", "L4", "L2"].contains(&n.as_str())).collect();
        let sr = eval(&sws, &held, &sarms);
        let sa = |n: &str| sarms.iter().position(|x| x.0 == n).unwrap();
        let mut sj = vec![];
        for (wi, w) in sws.iter().enumerate() {
            let u = |n: &str| mean(&col(&sr[wi][sa(n)], |x| util(x, &p0, mu())));
            let c = |n: &str| mean(&col(&sr[wi][sa(n)], |x| cost_of(x, &p0)));
            let q = |n: &str| mean(&col(&sr[wi][sa(n)], |x| x.q));
            println!("  {:<22} L3 q {:.4} cost {:.0} U {:.4} | dU: L2 {:+.4} L4 {:+.4} CA0 {:+.4} CA {:+.4} casc {:+.4} S {:+.4} | or-q {:+.4} or-cheap {:+.4} or-util {:+.4}", w.name, q("L3"), c("L3"), u("L3"), u("L2") - u("L3"), u("L4") - u("L3"), u("L3+CA0") - u("L3"), u("L3+CA") - u("L3"), u("L3+casc") - u("L3"), u("S") - u("L3"), u("or-q") - u("L3"), u("or-cheap") - u("L3"), u("or-util") - u("L3"));
            let dq_ca = cmp(&col(&sr[wi][sa("L3+CA")], |x| x.q), &col(&sr[wi][sa("L3+CA0")], |x| x.q), 0.01);
            let dc_ca = cmp(&col(&sr[wi][sa("L3+CA")], |x| cost_of(x, &p0)), &col(&sr[wi][sa("L3+CA0")], |x| cost_of(x, &p0)), 0.0);
            println!("      CA vs CA0 (price effect alone): dq {:+.4} [{:+.4},{:+.4}] tost_p {:.3}; cost {:+.0} TE [{:+.0},{:+.0}]", dq_ca.diff, dq_ca.lo, dq_ca.hi, dq_ca.tost_p, dc_ca.diff, dc_ca.lo, dc_ca.hi);
            let mut m = BTreeMap::new();
            for (n, _) in &sarms {
                m.insert(n.clone(), json!({"q": q(n), "cost": c(n), "util": u(n)}));
            }
            sj.push(json!({"world": w.name, "arms": m, "CA_vs_CA0": {"dq": dq_ca.diff, "dq_ci": [dq_ca.lo, dq_ca.hi], "dq_tost_p": dq_ca.tost_p, "dcost": dc_ca.diff, "dcost_ci": [dc_ca.lo, dc_ca.hi]}}));
        }
        jout.insert("price_structure_sens".into(), json!(sj));
    }

    // ------------------------------------------------------------------ raw per-arm means for the record
    let mut raw = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let mut m = BTreeMap::new();
        for (a, n) in names.iter().enumerate() {
            let v = &r[wi][a];
            let mut acc = Res::default();
            let mut accn = Res::default();
            for s in v {
                acc.add(&s.res);
                accn.add(&s.resn);
            }
            m.insert(n.clone(), json!({"q": mean(&col(v, |s| s.q)), "ok": mean(&col(v, |s| s.ok)), "q250": mean(&col(v, |s| s.q250)), "qlate": mean(&col(v, |s| s.qlate)), "cost": mean(&col(v, |s| cost_of(s, &p0))),
                "util03": mean(&col(v, |s| util(s, &p0, mu()))), "util10": mean(&col(v, |s| util(s, &p0, mu_alt()))), "res": acc.scaled(1.0 / v.len() as f64).arr(), "resn": accn.scaled(1.0 / v.len() as f64).arr(),
                "q_seeds": col(v, |s| s.q), "cost_seeds": col(v, |s| cost_of(s, &p0))}));
        }
        raw.push(json!({"world": w.name, "arms": m}));
    }
    jout.insert("raw".into(), json!(raw));
    jout.insert("res_names".into(), json!(RES_NAMES));
    jout.insert("runtime_s".into(), json!(t0.elapsed().as_secs_f64()));
    std::fs::write(&out_path, serde_json::to_string(&Value::Object(jout)).unwrap()).unwrap();
    println!("\nwrote {out_path} in {:.0}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[path = "../g18_common/mod.rs"]
    mod g18ref;

    fn mean32(x: &[f32]) -> f64 {
        x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64
    }
    #[test]
    fn reproduces_g18_levels() {
        // quality and call counts of L2 / L3 / L4 are the g18 orchestrator's, bit for bit (q is stored as f32 in g18)
        let gw = g18ref::worlds();
        let ws = worlds();
        for wi in [0usize, 6, 13] {
            let (gi, pi) = (g18ref::build(&gw[wi], 31, wi as u64), build(&ws[wi], 31, wi as u64));
            for (lv, cfg) in [(2u8, PCfg::l2()), (3, PCfg::l3()), (4, PCfg::l4(1.0, 0.1))] {
                let g = g18ref::run_orch(&gw[wi], &gi, &g18ref::OrchCfg::level(lv, 1.0, 1.0, 0.1), 31);
                let p = run_p(&ws[wi], &pi, &cfg, 31);
                assert!((mean32(&g.q) - p.q).abs() < 1e-6, "world {} L{lv}: g18 q {} vs h11 {}", ws[wi].name, mean32(&g.q), p.q);
                let gv = g.vcalls.iter().map(|&x| x as f64).sum::<f64>() / g.vcalls.len() as f64;
                assert!((gv - p.res.n_ver).abs() < 1e-9, "verifier calls differ L{lv}");
                let gc = g.calls.iter().map(|&x| x as f64).sum::<f64>() / g.calls.len() as f64;
                assert!((gc - p.res.n_gen).abs() < 1e-9, "generator calls differ L{lv}");
            }
        }
    }
    #[test]
    fn cost_model_units() {
        let p = Prices::base();
        // a flat-price step call without context costs S_REF; a verifier call costs cv*S_REF at r_out=4
        let one = Res { g_base: B_STEP, g_out: O_STEP, n_gen: 1.0, ..Res::default() };
        assert!((cost_te(&one, &p) - S_REF).abs() < 1e-9);
        let v = Res { v_in: 0.3 * S_REF - 4.0 * V_OUT, v_out: V_OUT, ..Res::default() };
        assert!((cost_te(&v, &p) - 0.3 * S_REF).abs() < 1e-9);
        // price-weighting: doubling every price doubles generator cost
        assert!((cost_te(&one.scaled(2.0), &p) - 2.0 * S_REF).abs() < 1e-9);
        // retrieval compute is negligible at the default lambda: one 2.6 ms encode < 0.02 reference calls
        let r = Res { ms: 2.62, ..Res::default() };
        assert!(cost_te(&r, &p) < 0.01 * S_REF);
    }
    #[test]
    fn prices_and_oracles() {
        let ws = worlds();
        let inst = build(&ws[14], 3, 14); // ctl-equal-agents: identical competence -> identical Cap prices
        assert!(inst.price.iter().all(|&x| (x - inst.price[0]).abs() < 1e-9));
        let inst = build(&ws[0], 3, 0);
        assert!(inst.price.iter().cloned().fold(0.0, f64::max) > inst.price.iter().cloned().fold(9.0, f64::min));
        // hindsight quality oracle is never below the always-verify session in this seed (it sees every realised q)
        let (a, b) = (run_p(&ws[0], &inst, &PCfg { rt: Rt::OracleQ, vmode: VMode::Never, ..PCfg::l3() }, 3), run_p(&ws[0], &inst, &PCfg::l3(), 3));
        assert!(a.q >= b.q - 1e-9, "oracle q {} < L3 q {}", a.q, b.q);
        // cheapest-that-works costs less than the quality oracle
        let c = run_p(&ws[0], &inst, &PCfg { rt: Rt::OracleCheap, vmode: VMode::Never, ..PCfg::l3() }, 3);
        assert!(cost_te(&c.res, &Prices::base()) < cost_te(&a.res, &Prices::base()));
    }
    #[test]
    fn deterministic() {
        let ws = worlds();
        let inst = build(&ws[0], 5, 0);
        let c = PCfg { rt: Rt::CA { mu: mu() }, vmode: VMode::Gate { rv: 1.0, nmin: 3.0 }, rmode: RMode::Gate { theta: 0.6 }, ..PCfg::l3() };
        let (a, b) = (run_p(&ws[0], &inst, &c, 5), run_p(&ws[0], &inst, &c, 5));
        assert_eq!(a.q, b.q);
        assert_eq!(a.res.arr(), b.res.arr());
    }
}
