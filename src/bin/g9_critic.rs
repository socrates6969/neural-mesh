//! G9-A (W10 RESEARCH-ABILITY): does ANY critic/verifier design earn its extra calls in a world where a generator
//! makes seeded REAL errors (stale fact, wrong entity, unsupported claim, arithmetic slip)?
//!
//! Critic designs: the existing research.rs gate (SecondOpinion flat / hierarchical, with a weak or strong second-agent checker that
//! does NOT look at evidence), naive literal support, entity-matched support, newest-evidence check, cross-source agreement, self-consistency
//! over k samples, a cascade (evidence -> self-consistency when evidence is inconclusive), an online-gated evidence critic ("auto"),
//! evidence-gated by the research.rs gate, an evidence-substitution sensitivity arm and a perfect-detector oracle (ceiling).
//! Worlds are FIXED in src/bin/g9_common/mod.rs (`critic_worlds`) before any policy was tuned.
//! Protocol: critic knobs tuned on seeds 1-8 (objective: mean net@0.05 over 6 tune worlds), held-out 31-60, `--held 61-90` = confirm with
//! the same frozen tuning. Paired 95% t-CI, Holm per family. net(c) = accuracy - c * (extra agent calls + 0.1 * evidence-tool runs).
//!
//! Usage: cargo run --release --offline --bin g9_critic -- [--held 31-60] [--out results/g9_critic.json]

use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g9_common/mod.rs"]
mod common;
use common::*;

const TUNE: (u64, u64) = (1, 8);
const TUNE_WORLDS: [&str; 6] = ["A1-mixed", "A2-stale-heavy", "A3-halluc-heavy", "A4-arith-heavy", "A9-rare-errors", "B1-report-mixed"];

fn run_matrix(ws: &[CWorld], arms: &[Arm], seeds: &[u64]) -> Vec<Vec<Vec<CRun>>> {
    // [world][arm][seed]
    ws.iter()
        .enumerate()
        .map(|(wi, w)| {
            let per_seed: Vec<Vec<CRun>> = seeds
                .par_iter()
                .map(|&s| {
                    let inst = build_critic(w, s, wi as u64);
                    arms.iter().map(|a| run_critic_arm(&inst, a)).collect()
                })
                .collect();
            (0..arms.len()).map(|ai| per_seed.iter().map(|r| r[ai].clone()).collect()).collect()
        })
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}
fn nets(r: &[CRun], c: f64, tc: f64) -> Vec<f64> {
    r.iter().map(|x| x.net(c, tc)).collect()
}

fn objective(m: &[Vec<Vec<CRun>>], ws: &[CWorld], arm_idx: usize) -> f64 {
    let mut v = vec![];
    for (wi, w) in ws.iter().enumerate() {
        if TUNE_WORLDS.contains(&w.name) {
            v.push(mean(&nets(&m[wi][arm_idx], 0.05, C_TOOL)));
        }
    }
    mean(&v)
}

/// pick the best candidate of a grid by the tune objective
fn tune(ws: &[CWorld], seeds: &[u64], cands: Vec<Arm>) -> (Arm, Vec<(String, f64)>) {
    let tune_ws: Vec<CWorld> = ws.iter().filter(|w| TUNE_WORLDS.contains(&w.name)).cloned().collect();
    let m = run_matrix(&tune_ws, &cands, seeds);
    let mut scores = vec![];
    for (ai, a) in cands.iter().enumerate() {
        let v: Vec<f64> = (0..tune_ws.len()).map(|wi| mean(&nets(&m[wi][ai], 0.05, C_TOOL))).collect();
        scores.push((a.name.clone(), mean(&v)));
    }
    let bi = scores.iter().enumerate().fold(0, |b, (i, s)| if s.1 > scores[b].1 + 1e-12 { i } else { b });
    (cands[bi].clone(), scores)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let held = parse_range(&args, "--held", (31, 60));
    let out = arg(&args, "--out", "results/g9_critic.json");
    let tune_seeds: Vec<u64> = (TUNE.0..=TUNE.1).collect();
    let ws = critic_worlds();
    let t0 = Instant::now();

    // ---------------- tuning (seeds 1-8)
    let mut tuned: Vec<(String, Vec<(String, f64)>)> = vec![];
    let mut pick = |label: &str, cands: Vec<Arm>| -> Arm {
        let (a, sc) = tune(&ws, &tune_seeds, cands);
        println!("tune {label}: picked {} ; grid {:?}", a.name, sc.iter().map(|s| format!("{}={:.4}", s.0, s.1)).collect::<Vec<_>>());
        tuned.push((label.to_string(), sc));
        a
    };
    let thetas = [0.5, 0.67, 1.0];
    let exist_flat = pick("exist-flat(weak) gate cost", [0.02, 0.05, 0.10].iter().map(|&c| arm(&format!("exist-flat c={c}"), ArmKind::Exist { hier: false, strong: false, cost: c })).collect());
    let exist_hier = pick("exist-hier(weak) gate cost", [0.02, 0.05, 0.10].iter().map(|&c| arm(&format!("exist-hier c={c}"), ArmKind::Exist { hier: true, strong: false, cost: c })).collect());
    let exist_flat_s = pick("exist-flat(strong) gate cost", [0.02, 0.05, 0.10].iter().map(|&c| arm(&format!("exist-flat-strong c={c}"), ArmKind::Exist { hier: false, strong: true, cost: c })).collect());
    let sc = pick("self-consistency k", [3usize, 5].iter().map(|&k| arm(&format!("sc k={k}"), ArmKind::Sc { k })).collect());
    let mut g = vec![];
    for u in [false, true] {
        g.push(arm(&format!("support-entity unv_flag={u}"), ArmKind::Ev { kind: EvKind::SupportEntity, theta: 0.5, unv_flag: u }));
    }
    let sup_ent = pick("support-entity", g);
    let mut g = vec![];
    for u in [false, true] {
        g.push(arm(&format!("latest unv_flag={u}"), ArmKind::Ev { kind: EvKind::Latest, theta: 0.5, unv_flag: u }));
    }
    let latest = pick("latest", g);
    let mut g = vec![];
    for &th in &thetas {
        for u in [false, true] {
            g.push(arm(&format!("agree th={th} unv_flag={u}"), ArmKind::Ev { kind: EvKind::Agree, theta: th, unv_flag: u }));
        }
    }
    let agree = pick("agree", g);
    let (ag_th, ag_u) = match &agree.kind {
        ArmKind::Ev { theta, unv_flag, .. } => (*theta, *unv_flag),
        _ => unreachable!(),
    };
    let cascade = pick("cascade theta", thetas.iter().map(|&th| arm(&format!("cascade th={th}"), ArmKind::Cascade { theta: th })).collect());
    let mut g = vec![];
    for &c in &[0.02, 0.05, 0.10] {
        for u in [false, true] {
            g.push(arm(&format!("auto th={ag_th} unv={u} c={c}"), ArmKind::Auto { theta: ag_th, unv_flag: u, cost: c }));
        }
    }
    let auto = pick("auto (online-gated agree)", g);
    let gate_ev = pick("gate+evidence cost", [0.02, 0.05, 0.10].iter().map(|&c| arm(&format!("gate-ev c={c}"), ArmKind::GateEv { cost: c, theta: ag_th })).collect());
    let ev_sub = arm(&format!("ev-substitute th={ag_th}"), ArmKind::EvSub { theta: ag_th });
    println!("tuning done in {:.1}s", t0.elapsed().as_secs_f64());

    // ---------------- evaluation arms
    let mut arms: Vec<Arm> = vec![
        arm("none", ArmKind::None),
        arm("exist-flat(weak)", exist_flat.kind.clone()),
        arm("exist-hier(weak)", exist_hier.kind.clone()),
        arm("exist-flat(strong)", exist_flat_s.kind.clone()),
        arm("self-consistency", sc.kind.clone()),
        arm("support-any(naive)", ArmKind::Ev { kind: EvKind::SupportAny, theta: 0.5, unv_flag: false }),
        arm("support-entity", sup_ent.kind.clone()),
        arm("latest", latest.kind.clone()),
        arm("agree", agree.kind.clone()),
        arm("cascade", cascade.kind.clone()),
        arm("auto", auto.kind.clone()),
        arm("gate+evidence", gate_ev.kind.clone()),
        arm("ev-substitute", ev_sub.kind.clone()),
        arm("oracle(ceiling)", ArmKind::Oracle),
    ];
    // gate cost for the strong exist arm was also tuned; keep names stable
    let fair = ["none", "exist-flat(weak)", "exist-hier(weak)", "self-consistency", "support-any(naive)"];
    let treat = ["support-entity", "latest", "agree", "cascade", "auto", "gate+evidence"];
    let names: Vec<String> = arms.iter().map(|a| a.name.clone()).collect();
    let ix = |n: &str| names.iter().position(|x| x == n).unwrap();
    let m = run_matrix(&ws, &arms, &held);
    println!("evaluation done in {:.1}s", t0.elapsed().as_secs_f64());
    arms.shrink_to_fit();

    // ---------------- tests
    let mut tests: Vec<Test> = vec![];
    let mut best_fair: Vec<String> = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let bf = fair.iter().max_by(|a, b| mean(&nets(&m[wi][ix(a)], 0.05, C_TOOL)).partial_cmp(&mean(&nets(&m[wi][ix(b)], 0.05, C_TOOL))).unwrap()).unwrap();
        best_fair.push(bf.to_string());
        for t in treat.iter().chain(["exist-flat(strong)", "ev-substitute"].iter()) {
            let tv = nets(&m[wi][ix(t)], 0.05, C_TOOL);
            tests.push(mk_test("F1-net05-vs-bestfair", w.name, "net@0.05", t, bf, &tv, &nets(&m[wi][ix(bf)], 0.05, C_TOOL)));
            tests.push(mk_test("F2-net05-vs-none", w.name, "net@0.05", t, "none", &tv, &nets(&m[wi][ix("none")], 0.05, C_TOOL)));
            tests.push(mk_test("F3-acc-vs-none", w.name, "accuracy(final)", t, "none", &m[wi][ix(t)].iter().map(|x| x.acc_final / x.n).collect::<Vec<_>>(), &m[wi][ix("none")].iter().map(|x| x.acc_final / x.n).collect::<Vec<_>>()));
        }
        // the existing critic vs none (is the shipped critic worth it?)
        for t in ["exist-flat(weak)", "exist-hier(weak)", "self-consistency", "support-any(naive)"] {
            tests.push(mk_test("F2-net05-vs-none", w.name, "net@0.05", t, "none", &nets(&m[wi][ix(t)], 0.05, C_TOOL), &nets(&m[wi][ix("none")], 0.05, C_TOOL)));
        }
    }
    holm_by_family(&mut tests);

    // ---------------- report
    let mut jw = vec![];
    for (wi, w) in ws.iter().enumerate() {
        println!("\n=== {} (family {}, eps {:.2}, m {}) best fair (net@.05): {}", w.name, w.family, w.eps, w.m, best_fair[wi]);
        println!("{:<20} {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6} {:>6}", "arm", "acc1", "accF", "calls", "tool", "recl", "prec", "ffr", "silent", "net.02", "net.05", "net.10", "harm");
        let mut ja = vec![];
        for (ai, a) in names.iter().enumerate() {
            let r = &m[wi][ai];
            let f = |g: &dyn Fn(&CRun) -> f64| mean(&r.iter().map(g).collect::<Vec<_>>());
            let acc1 = f(&|x| x.acc_first / x.n);
            let accf = f(&|x| x.acc_final / x.n);
            let calls = f(&|x| x.extra_calls / x.n);
            let tool = f(&|x| x.tool_units / x.n);
            let recl = f(&|x| x.recall());
            let prec = f(&|x| x.precision());
            let ffr = f(&|x| x.false_flag());
            let silent = f(&|x| x.silent / x.n);
            let wrong_del = f(&|x| x.delivered_wrong / x.n);
            let harmed = f(&|x| x.harmed / x.n);
            let rescued = f(&|x| x.rescued / x.n);
            let n02 = mean(&nets(r, 0.02, C_TOOL));
            let n05 = mean(&nets(r, 0.05, C_TOOL));
            let n10 = mean(&nets(r, 0.10, C_TOOL));
            let n05_t0 = mean(&nets(r, 0.05, 0.0));
            let n05_t3 = mean(&nets(r, 0.05, 0.3));
            let n05_t1 = mean(&nets(r, 0.05, 1.0));
            let bt: Vec<Value> = (0..4)
                .map(|ty| {
                    let e: f64 = r.iter().map(|x| x.by_type[ty].0).sum();
                    let fl: f64 = r.iter().map(|x| x.by_type[ty].1).sum();
                    json!({"type": ERR_NAMES[ty], "errors": e / r.len() as f64, "recall": if e > 0.0 { fl / e } else { f64::NAN }})
                })
                .collect();
            println!("{:<20} {:>6.3} {:>6.3} {:>6.3} {:>6.3} | {:>6.3} {:>6.3} {:>6.3} {:>6.3} | {:>6.3} {:>6.3} {:>6.3} {:>6.3}", a, acc1, accf, calls, tool, recl, prec, ffr, silent, n02, n05, n10, harmed);
            ja.push(json!({"arm": a, "acc_first": acc1, "acc_final": accf, "extra_calls_per_task": calls, "tool_runs_per_task": tool, "recall": recl, "precision": prec, "false_flag_rate": ffr,
                "silent_error_rate": silent, "delivered_wrong_rate": wrong_del, "harmed_rate": harmed, "rescued_rate": rescued,
                "net": {"c0.02": n02, "c0.05": n05, "c0.10": n10, "c0.05_tool0": n05_t0, "c0.05_tool0.3": n05_t3, "c0.05_tool1.0": n05_t1}, "recall_by_type": bt}));
        }
        jw.push(json!({"world": w.name, "family": w.family, "eps": w.eps, "m": w.m, "best_fair": best_fair[wi], "arms": ja}));
    }
    println!("\n=== Holm tests (family-wise)");
    for t in &tests {
        if t.family.starts_with("F1") || (t.family.starts_with("F2") && !t.treat.starts_with("ev-sub")) {
            let v = verdict(t.adj, t.d.diff.mean);
            if v != "ns" || t.family.starts_with("F1") {
                println!("{:<22} {:<22} {:<20} vs {:<20} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.4} {}", t.family, t.world, t.treat, t.vs, t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, t.d.wins, t.d.losses, t.adj, v);
            }
        }
    }
    let count = |fam: &str, v: &str| tests.iter().filter(|t| t.family == fam && verdict(t.adj, t.d.diff.mean) == v).count();
    for fam in ["F1-net05-vs-bestfair", "F2-net05-vs-none", "F3-acc-vs-none"] {
        println!("{fam}: WIN {} LOSS {} of {}", count(fam, "WIN"), count(fam, "LOSS"), tests.iter().filter(|t| t.family == fam).count());
    }
    let tuned_json: Vec<Value> = tuned.iter().map(|(l, s)| json!({"label": l, "grid": s})).collect();
    let picks: Vec<Value> = [&exist_flat, &exist_hier, &exist_flat_s, &sc, &sup_ent, &latest, &agree, &cascade, &auto, &gate_ev].iter().map(|a| json!({"name": a.name})).collect();
    let doc = json!({"held": [held[0], held[held.len() - 1]], "tune_seeds": [TUNE.0, TUNE.1], "tune_worlds": TUNE_WORLDS, "c_tool": C_TOOL, "tuning": tuned_json, "picks": picks, "worlds": jw,
        "tests": tests.iter().map(test_json).collect::<Vec<_>>(), "runtime_s": t0.elapsed().as_secs_f64()});
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out} in {:.1}s", t0.elapsed().as_secs_f64());
}
