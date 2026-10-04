//! R3 part 1: does a bounded working-memory buffer of per-context router slots
//! improve routing in worlds with RECURRING task contexts?
//!
//! Protocol (same as shortlab): tune on seeds 1-8 only (world "recur-6fam" for
//! the no-chunk arms, "recur-4fam x3var" for the chunk arms); evaluate on
//! held-out seeds 31-60; paired 95% t-CI; Holm over (arms x worlds) against the
//! best fair baseline of {keyword, eps-greedy, pooled-ts, hebbian-robust,
//! warm(hebbian-robust)} chosen per world by held-out mean (conservative).
//! Controls where the mechanism SHOULD NOT help: unique (never-recurring)
//! contexts, contexts that do not change the skill table, and the four sim.rs
//! scenarios (one-shot drift, no recurrence).
//!
//! Usage: cargo run --release --bin r3_workmem [-- --out results/r3_memory_spatial.json]

use neural_mesh::sim::Scenario;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use neural_mesh::workmem::ctxworld::{generate, CtxCfg, Recurrence, RunStats, Stream};
use neural_mesh::workmem::harness::*;
use rayon::prelude::*;
use serde_json::{json, Value};

type Gen = Box<dyn Fn(u64) -> Stream + Sync + Send>;

struct WorldSpec {
    name: String,
    control: bool,
    mkg: Gen,
}

fn eval(k: &Kind, seeds: &[u64], g: &Gen) -> Vec<RunStats> {
    seeds
        .par_iter()
        .map(|&s| {
            let st = g(s);
            run(k, &st, s)
        })
        .collect()
}

fn obj(k: &Kind, seeds: &[u64], g: &Gen) -> f64 {
    let r = eval(k, seeds, g);
    r.iter().map(|x| x.mean).sum::<f64>() / r.len() as f64
}

fn best_of(cands: Vec<Kind>, seeds: &[u64], g: &Gen, label: &str) -> Kind {
    let n = cands.len();
    let mut best = (f64::NEG_INFINITY, cands[0].clone());
    for k in cands {
        let o = obj(&k, seeds, g);
        if o > best.0 {
            best = (o, k);
        }
    }
    println!("[tune 1-8] {label}: {n} configs, best {:.4} -> {:?}", best.0, best.1);
    best.1
}

fn col(r: &[RunStats], f: fn(&RunStats) -> f64) -> Vec<f64> {
    r.iter().map(f).collect()
}
fn f_mean(r: &RunStats) -> f64 {
    r.mean
}
fn f_rev(r: &RunStats) -> f64 {
    r.revisit_mean
}
fn f_first(r: &RunStats) -> f64 {
    r.first_visit_mean
}
fn f_early(r: &RunStats) -> f64 {
    r.first_third
}

pub fn merge_json(path: &str, key: &str, v: Value) {
    let mut root: Value = std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_else(|| json!({}));
    root[key] = v;
    if let Some(d) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(d);
    }
    std::fs::write(path, serde_json::to_string_pretty(&root).unwrap()).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/r3_memory_spatial.json".into());
    let tune: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = (31..=60).collect();
    let n_tasks = 1500;

    let cfg = |f: usize, v: usize| CtxCfg { families: f, variants: v, n_tasks, ..CtxCfg::default() };
    let mk = |c: CtxCfg| -> Gen { Box::new(move |s| generate(s, &c)) };
    let w1 = mk(cfg(6, 1));
    let w2 = mk(cfg(4, 3));

    // ---------------- tuning (seeds 1-8) ----------------
    println!("r3_workmem: {n_tasks} tasks/run, tune seeds 1-8, held-out 31-60");
    let mut c = vec![];
    for &g in &[0.95, 0.99, 1.0] {
        for &e in &[0.02, 0.05, 0.1] {
            c.push(Kind::Eps(g, e));
        }
    }
    let Kind::Eps(eg, ee) = best_of(c, &tune, &w1, "eps-greedy") else { unreachable!() };
    let c = [0.9, 0.95, 0.99, 1.0].iter().map(|&g| Kind::Pts(g)).collect();
    let Kind::Pts(pg) = best_of(c, &tune, &w1, "pooled-ts") else { unreachable!() };
    let evs = [Ev::Set, Ev::Mean, Ev::Global, Ev::FailSet, Ev::FailMean];
    let mut c = vec![];
    for &k in &[1.0, 3.0, 10.0, 30.0, 100.0, 300.0] {
        for &e in &evs {
            c.push(Kind::WarmHebR(k, e));
        }
    }
    let warm = best_of(c, &tune, &w1, "warm(hebbian-robust) [= K=0 no-buffer]");

    let wm_grid = |cap: usize, chunk: bool, warm_on: bool| -> Vec<Kind> {
        let mut c = vec![];
        let ws: Vec<Option<(f64, Ev)>> = if warm_on {
            let mut v = vec![];
            for &k in &[3.0, 10.0, 30.0, 100.0] {
                for &e in &[Ev::Set, Ev::FailSet, Ev::Mean] {
                    v.push(Some((k, e)));
                }
            }
            v
        } else {
            vec![None]
        };
        for w in ws {
            for &min_n in &[0u64, 3, 10] {
                for &init_g in &[false, true] {
                    c.push(Kind::Wm(Wm { cap, min_n, init_g, chunk, warm: w }));
                }
            }
        }
        c
    };
    let caps = [2usize, 4, 7, 9, usize::MAX];
    let mut arms: Vec<Kind> = vec![];
    for &cap in &caps {
        let k = best_of(wm_grid(cap, false, true), &tune, &w1, &format!("warm+workmem K={}", if cap == usize::MAX { "inf".into() } else { cap.to_string() }));
        arms.push(k);
    }
    for &cap in &[4usize, usize::MAX] {
        arms.push(best_of(wm_grid(cap, false, false), &tune, &w1, &format!("bare workmem K={}", if cap == usize::MAX { "inf".into() } else { cap.to_string() })));
    }
    for &cap in &[2usize, 4, 7] {
        arms.push(best_of(wm_grid(cap, true, true), &tune, &w2, &format!("warm+workmem K={cap} chunk")));
    }
    let arm_names: Vec<String> = arms.iter().map(kind_label).collect();

    // ---------------- worlds ----------------
    let mut worlds: Vec<WorldSpec> = vec![
        WorldSpec { name: "recur-6fam".into(), control: false, mkg: mk(cfg(6, 1)) },
        WorldSpec { name: "recur-4fam-x3var".into(), control: false, mkg: mk(cfg(4, 3)) },
        WorldSpec { name: "recur-12ctx".into(), control: false, mkg: mk(cfg(12, 1)) },
        WorldSpec { name: "recur-6fam-marker50".into(), control: false, mkg: mk(CtxCfg { marker_prob: 0.5, ..cfg(6, 1) }) },
        WorldSpec { name: "CONTROL unique-contexts".into(), control: true, mkg: mk(CtxCfg { rec: Recurrence::Unique, ..cfg(6, 1) }) },
        WorldSpec { name: "CONTROL irrelevant-contexts".into(), control: true, mkg: mk(CtxCfg { rec: Recurrence::Irrelevant, ..cfg(6, 1) }) },
    ];
    for sc in [Scenario::Fixed, Scenario::Random, Scenario::NoSynergy, Scenario::NoTraps] {
        worlds.push(WorldSpec {
            name: format!("CONTROL sim:{}", sc.label()),
            control: true,
            mkg: Box::new(move |s| sim_stream(s, sc, n_tasks)),
        });
    }

    let baselines: Vec<(String, Kind)> = vec![
        ("eps-greedy(tuned)".into(), Kind::Eps(eg, ee)),
        ("pooled-ts(tuned)".into(), Kind::Pts(pg)),
        ("hebbian-robust".into(), Kind::HebR),
        ("warm(hebbian-robust)=K0".into(), warm.clone()),
    ];

    let nb = baselines.len() + 1; // + keyword
    let na = arms.len();
    let mut all_p: Vec<f64> = vec![];
    struct WRes {
        kw: Vec<RunStats>,
        base: Vec<Vec<RunStats>>,
        arm: Vec<Vec<RunStats>>,
        oracle: f64,
        best: usize,
    }
    let mut wres: Vec<WRes> = vec![];
    for w in &worlds {
        let kw = eval(&Kind::Keyword, &held, &w.mkg);
        let base: Vec<Vec<RunStats>> = baselines.iter().map(|(_, k)| eval(k, &held, &w.mkg)).collect();
        let arm: Vec<Vec<RunStats>> = arms.iter().map(|k| eval(k, &held, &w.mkg)).collect();
        let oracle = summarize(&col(&kw, |r| r.oracle)).mean;
        // best fair baseline among keyword + learned baselines, by held-out mean
        let mut means: Vec<f64> = vec![summarize(&col(&kw, f_mean)).mean];
        means.extend(base.iter().map(|b| summarize(&col(b, f_mean)).mean));
        let best = (0..nb).max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap()).unwrap();
        for a in &arm {
            let reference = if best == 0 { &kw } else { &base[best - 1] };
            all_p.push(paired_t_pvalue(&col(a, f_mean), &col(reference, f_mean)));
        }
        wres.push(WRes { kw, base, arm, oracle, best });
    }
    let adj = holm_adjust(&all_p);

    // ---------------- report ----------------
    let bname = |i: usize| if i == 0 { "keyword".to_string() } else { baselines[i - 1].0.clone() };
    let mut jworlds = vec![];
    let mut wins_by_world: Vec<(String, bool, Vec<String>)> = vec![];
    for (wi, w) in worlds.iter().enumerate() {
        let r = &wres[wi];
        println!("\n=== {} (held-out 31-60, mean expected success; oracle {:.3}) ===", w.name, r.oracle);
        println!("{:<28}{:>14}{:>14}{:>14}{:>14}", "router", "overall", "first third", "revisit", "first-visit");
        let mut rows: Vec<(String, &Vec<RunStats>)> = vec![("keyword".into(), &r.kw)];
        for (i, (n, _)) in baselines.iter().enumerate() {
            rows.push((n.clone(), &r.base[i]));
        }
        for (i, n) in arm_names.iter().enumerate() {
            rows.push((n.clone(), &r.arm[i]));
        }
        let mut jrows = vec![];
        for (n, rs) in &rows {
            let f = |g: fn(&RunStats) -> f64| {
                let s = summarize(&col(rs, g));
                format!("{:.3}+-{:.3}", s.mean, s.half_width)
            };
            println!("{:<28}{:>14}{:>14}{:>14}{:>14}", n, f(f_mean), f(f_early), f(f_rev), f(f_first));
            jrows.push(json!({"router": n, "overall": summarize(&col(rs, f_mean)), "first_third": summarize(&col(rs, f_early)),
                "revisit": summarize(&col(rs, f_rev)), "first_visit": summarize(&col(rs, f_first))}));
        }
        println!("best fair baseline here: {}", bname(r.best));
        let reference = if r.best == 0 { &r.kw } else { &r.base[r.best - 1] };
        let k0 = &r.base[baselines.len() - 1];
        let kinf_idx = arm_names.iter().position(|n| n == "warm+wm[K=inf]").unwrap();
        let mut tests = vec![];
        let mut wins = vec![];
        for (ai, n) in arm_names.iter().enumerate() {
            let x = col(&r.arm[ai], f_mean);
            let d = paired_diff(&x, &col(reference, f_mean));
            let d0 = paired_diff(&x, &col(k0, f_mean));
            let dinf = paired_diff(&x, &col(&r.arm[kinf_idx], f_mean));
            let p = adj[wi * na + ai];
            let win = p < 0.05 && d.diff.mean > 0.0;
            if win {
                wins.push(n.clone());
            }
            println!(
                "  {:<22} vs best-fair({}) {:+.4} [{:+.4},{:+.4}] holm p={:.4}{} | vs no-buffer {:+.4} [{:+.4},{:+.4}] | vs unbounded {:+.4} [{:+.4},{:+.4}]",
                n, bname(r.best), d.diff.mean, d.diff.lo, d.diff.hi, p, if win { " WIN" } else { "" },
                d0.diff.mean, d0.diff.lo, d0.diff.hi, dinf.diff.mean, dinf.diff.lo, dinf.diff.hi
            );
            tests.push(json!({"arm": n, "vs_best_fair": {"ref": bname(r.best), "diff": d.diff, "wins": d.wins, "losses": d.losses, "holm_p": p, "win": win},
                "vs_no_buffer": {"diff": d0.diff, "wins": d0.wins, "losses": d0.losses},
                "vs_unbounded": {"diff": dinf.diff}}));
        }
        wins_by_world.push((w.name.clone(), w.control, wins));
        jworlds.push(json!({"world": w.name, "control": w.control, "oracle": r.oracle, "best_fair_baseline": bname(r.best), "table": jrows, "tests": tests}));
    }

    let noncontrol_wins: Vec<&(String, bool, Vec<String>)> = wins_by_world.iter().filter(|x| !x.1 && !x.2.is_empty()).collect();
    println!("\n=== VERDICT inputs ===");
    for (n, c, w) in &wins_by_world {
        println!("{:<34} control={:<5} Holm-significant wins over best fair baseline: {:?}", n, c, w);
    }
    let accepted = noncontrol_wins.len() > 1;
    println!("worlds (non-control) with at least one Holm win: {} -> bar (>1 world) {}", noncontrol_wins.len(), if accepted { "MET" } else { "NOT MET" });

    let j = json!({
        "protocol": "tune seeds 1-8 (recur-6fam; chunk arms on recur-4fam-x3var); held-out 31-60; paired 95% t-CI; Holm over arms x worlds vs best fair baseline per world",
        "metric": "mean expected success of chosen agent over the run (1500 tasks)",
        "tuned": {"eps_greedy": format!("{:?}", Kind::Eps(eg, ee)), "pooled_ts": format!("{:?}", Kind::Pts(pg)), "warm_hebr_K0": format!("{:?}", warm),
            "arms": arms.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>()},
        "worlds": jworlds,
        "bar_met_more_than_one_noncontrol_world": accepted,
        "wins_by_world": wins_by_world.iter().map(|(n, c, w)| json!({"world": n, "control": c, "winning_arms": w})).collect::<Vec<_>>(),
    });
    merge_json(&out, "workmem", j);
    println!("\nwrote {out} (key workmem)");
}
