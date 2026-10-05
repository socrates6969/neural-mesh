//! G10-E2: recency-discounted per-anchor/agent vote statistics under CONTEXT drift.
//!
//! The knowledge vote (knowledge.rs `vote`, replayed here as `VotePol`) trusts the cumulative per-(anchor, agent) reward sums
//! (smoothed mean >= tau with >= min_n rewards). If the mapping context -> best agent changes mid-stream those sums keep voting
//! for the OLD best agent for a long time. E2 discounts those sums: `Event(gamma)` (every reward event of an anchor multiplies the
//! anchor's statistics by gamma first), `Time(h)` (exponential decay with half-life h tasks), and `Event+gate` (additionally decays
//! the vote-gate sums). New world feature: a context-DRIFT (a share of the contexts get a NEW best agent at stream position `at`;
//! the text stream is identical to the stationary world). Controls: ctl-irrel (context carries no information) and the
//! stationary world (what discounting costs when nothing drifts).
//!
//! Protocol: tune on seeds 1-8 (objective = mean H2000 over the three drift worlds and both families), held-out 31-60, frozen
//! confirmation 61-90 on a fresh corpus (G1B); paired 95% t-CI vs the undiscounted vote (incumbent) and vs the best fair baseline,
//! Holm over the family; control non-inferiority by TOST (+-0.005).
//!
//! Usage: cargo run --release --offline --bin g10_drift -- --part tune|eval [--held 31-60] [--confirm]

use std::time::Instant;

use serde_json::{json, Value};

#[path = "g10_common/mod.rs"]
mod common;
use common::*;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};

const CTL_MARGIN: f64 = 0.005;
static SRC_CELL: std::sync::OnceLock<Src> = std::sync::OnceLock::new();
fn src() -> Src {
    *SRC_CELL.get().unwrap_or(&Src::Mini)
}

fn base_name(fam: Fam) -> &'static str {
    if fam == Fam::F5 { "q" } else { "sib16" }
}
fn drift_specs(fam: Fam) -> Vec<Spec> {
    let b = world(fam, src(), base_name(fam));
    let mk = |tag: &str, frac: f64, at: f64| {
        let mut s = b.clone();
        s.name = format!("{}-{}-{}", fam.label(), src().label(), tag);
        s.drift = Drift { frac, at };
        s
    };
    vec![mk("drift-mid", 1.0, 0.5), mk("drift-half", 0.5, 0.5), mk("drift-early", 1.0, 0.15)]
}
fn all_specs(fam: Fam) -> Vec<Spec> {
    let mut v = drift_specs(fam);
    let mut st = world(fam, src(), base_name(fam));
    st.name = format!("{}-{}-stationary", fam.label(), src().label());
    v.push(st);
    v.push(world(fam, src(), "ctl-irrel"));
    v
}

const COLS: [&str; 5] = ["H500", "H2000", "post1", "post2", "rec"];
fn rec_time(r: &RunOut) -> f64 {
    let (sw, n) = (r.switch, r.pe.len());
    if sw == usize::MAX || sw < 800 {
        return f64::NAN;
    }
    let pre = win_mean(r, sw - 300, sw);
    let thr = 0.95 * pre;
    let mut cs = vec![0.0f64; n + 1];
    for t in 0..n {
        cs[t + 1] = cs[t] + r.pe[t] as f64;
    }
    for t in (sw + 100)..=n {
        if (cs[t] - cs[t - 100]) / 100.0 >= thr {
            return (t - sw) as f64;
        }
    }
    (n - sw) as f64
}
fn extract(r: &RunOut) -> Vec<f64> {
    let sw = r.switch;
    let n = r.pe.len();
    let (p1, p2) = if sw == usize::MAX { (f64::NAN, f64::NAN) } else { (win_mean(r, sw, (sw + 300).min(n)), win_mean(r, (sw + 300).min(n - 1), n)) };
    vec![r.h[2], r.h[3], p1, p2, rec_time(r)]
}
fn colv(rs: &[RunOut], ci: usize) -> Vec<f64> {
    rs.iter().map(|r| extract(r)[ci]).collect()
}

fn grid() -> Vec<(String, Disc, f64)> {
    let mut g: Vec<(String, Disc, f64)> = vec![("undisc".into(), Disc::None, 1.0)];
    for &x in &[0.98, 0.95, 0.9, 0.85, 0.8, 0.7] {
        g.push((format!("ev{x}"), Disc::Event(x), 1.0));
    }
    for &h in &[50.0, 100.0, 200.0, 400.0, 800.0] {
        g.push((format!("hl{h}"), Disc::Time(h), 1.0));
    }
    for &x in &[0.95, 0.9, 0.8] {
        for &gd in &[0.99, 0.97] {
            g.push((format!("ev{x}+gate{gd}"), Disc::Event(x), gd));
        }
    }
    g
}
fn vote_cfg(d: Disc, gd: f64) -> VoteCfg {
    VoteCfg { disc: d, gate_disc: gd, ..std_vote() }
}

fn inc_method(sel: &Value, fam: Fam) -> Method {
    let tag = if fam == Fam::G1B { "g1A" } else { fam.label() };
    let e = sel["incumbents"].as_array().unwrap().iter().find(|i| i["fam"] == tag && i["src"] == src().label()).expect("incumbent");
    method_from_json(&e["vote"])
}

fn tune(sel: &Value, out: &str) {
    let t0 = Instant::now();
    let seeds: Vec<u64> = (1..=8).collect();
    let g = grid();
    println!("== tune (seeds 1-8): {} discount configs; objective = mean H2000 over {{drift-mid, drift-half, drift-early}} x {{f5, g1A}} (mini)", g.len());
    let mut obj = vec![vec![]; g.len()];
    let mut detail = vec![];
    for fam in [Fam::F5, Fam::G1A] {
        let m = inc_method(sel, fam);
        let arms: Vec<A> = g.iter().map(|(_, d, gd)| (Kind2::KnVote(m.clone(), vote_cfg(*d, *gd)), Signal::Outcome)).collect();
        let specs = all_specs(fam);
        let rr = eval_runs(&arms, &specs, &seeds);
        println!("-- family {} anchor {}: mean over seeds of [drift-mid, drift-half, drift-early | stationary, ctl-irrel] H2000 / post2", fam.label(), m.label());
        for (gi, (n, _, _)) in g.iter().enumerate() {
            let h: Vec<f64> = (0..specs.len()).map(|wi| mean_of(colv(&rr[gi][wi], 1).into_iter())).collect();
            let p2: Vec<f64> = (0..3).map(|wi| mean_of(colv(&rr[gi][wi], 3).into_iter())).collect();
            println!("   {:<14} H2000 [{:.4} {:.4} {:.4} | {:.4} {:.4}]  post2 [{:.4} {:.4} {:.4}]", n, h[0], h[1], h[2], h[3], h[4], p2[0], p2[1], p2[2]);
            obj[gi].push(mean_of(h[..3].iter().cloned()));
            detail.push(json!({"fam": fam.label(), "cfg": n, "h2000": h, "post2": p2}));
        }
    }
    let score: Vec<f64> = obj.iter().map(|v| mean_of(v.iter().cloned())).collect();
    let pick = |pred: &dyn Fn(&(String, Disc, f64)) -> bool| -> usize {
        (0..g.len()).filter(|&i| pred(&g[i])).max_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap()).unwrap()
    };
    let ev = pick(&|c| matches!(c.1, Disc::Event(_)) && c.2 == 1.0);
    let tm = pick(&|c| matches!(c.1, Disc::Time(_)));
    let eg = pick(&|c| matches!(c.1, Disc::Event(_)) && c.2 < 1.0);
    println!("\nobjective (mean H2000 over drift worlds, 2 families): undisc {:.4}", score[0]);
    for (n, i) in [("best Event", ev), ("best Time", tm), ("best Event+gate", eg)] {
        println!("  {n}: {} -> {:.4} ({:+.4} vs undisc)", g[i].0, score[i], score[i] - score[0]);
    }
    let js = |i: usize| json!({"name": g[i].0, "gamma_or_h": match g[i].1 { Disc::Event(x) => x, Disc::Time(h) => h, Disc::None => 0.0 }, "kind": match g[i].1 { Disc::Event(_) => "event", Disc::Time(_) => "time", Disc::None => "none" }, "gate_disc": g[i].2});
    let doc = json!({"lab": "g10_drift/tune", "tune_seeds": [1, 8], "objective": "mean H2000 over drift worlds and families", "scores": g.iter().enumerate().map(|(i, c)| json!({"cfg": c.0, "score": score[i]})).collect::<Vec<_>>(),
        "picked": {"event": js(ev), "time": js(tm), "event_gate": js(eg)}, "detail": detail, "elapsed_s": t0.elapsed().as_secs_f64()});
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out} in {:.0}s", t0.elapsed().as_secs_f64());
}

fn cfg_from(v: &Value) -> (String, Disc, f64) {
    let x = v["gamma_or_h"].as_f64().unwrap();
    let d = match v["kind"].as_str().unwrap() {
        "event" => Disc::Event(x),
        "time" => Disc::Time(x),
        _ => Disc::None,
    };
    (v["name"].as_str().unwrap().to_string(), d, v["gate_disc"].as_f64().unwrap())
}

struct Pc {
    diff: f64,
    lo: f64,
    hi: f64,
    p: f64,
    wins: usize,
    cm: f64,
    bm: f64,
    tp: f64,
}
fn pc(c: &[f64], b: &[f64]) -> Pc {
    let ok: Vec<usize> = (0..c.len()).filter(|&i| c[i].is_finite() && b[i].is_finite()).collect();
    let (c, b): (Vec<f64>, Vec<f64>) = (ok.iter().map(|&i| c[i]).collect(), ok.iter().map(|&i| b[i]).collect());
    let d = paired_diff(&c, &b);
    let dv: Vec<f64> = c.iter().zip(&b).map(|(x, y)| x - y).collect();
    Pc { diff: d.diff.mean, lo: d.diff.lo, hi: d.diff.hi, p: paired_t_pvalue(&c, &b), wins: d.wins, cm: summarize(&c).mean, bm: summarize(&b).mean, tp: tost(&dv, CTL_MARGIN).0 }
}

fn eval(sel: &Value, tuned: &Value, held: &[u64], confirm: bool, out: &str) {
    let t0 = Instant::now();
    let g1 = if confirm { Fam::G1B } else { Fam::G1A };
    let picked: Vec<(String, Disc, f64)> = ["event", "time", "event_gate"].iter().map(|k| cfg_from(&tuned["picked"][k])).collect();
    println!("== eval seeds {}-{}: families f5 + {} (mini); frozen picks: {}", held[0], held[held.len() - 1], g1.label(), picked.iter().map(|p| p.0.clone()).collect::<Vec<_>>().join(", "));
    let g = grid();
    let mut cells: Vec<(String, String, String, String, Pc)> = vec![]; // (fam, world, cand, col, stats)
    let mut base_cells: Vec<(String, String, String, String, String, Pc)> = vec![];
    let mut sens: Vec<Value> = vec![];
    let mut rows: Vec<Value> = vec![];
    let bases = bases_o();
    for fam in [Fam::F5, g1] {
        let m = inc_method(sel, fam);
        let specs = all_specs(fam);
        // arms: baselines, then the full grid (grid[0] = undisc)
        let mut arms: Vec<A> = bases.iter().map(|b| b.1.clone()).collect();
        let nb = arms.len();
        for (_, d, gd) in &g {
            arms.push((Kind2::KnVote(m.clone(), vote_cfg(*d, *gd)), Signal::Outcome));
        }
        let rr = eval_runs(&arms, &specs, held);
        println!("\n##### {}  anchor {}", fam.label(), m.label());
        println!("{:<26} {:<16} | {:>7} {:>7} {:>7} {:>7} {:>7}", "world", "arm", COLS[0], COLS[1], COLS[2], COLS[3], COLS[4]);
        for (wi, sp) in specs.iter().enumerate() {
            let mut show = |name: &str, rs: &Vec<RunOut>| {
                let v: Vec<f64> = (0..COLS.len()).map(|ci| mean_of(colv(rs, ci).into_iter())).collect();
                println!("{:<26} {:<16} | {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.1}", sp.name, name, v[0], v[1], v[2], v[3], v[4]);
                rows.push(json!({"fam": fam.label(), "world": sp.name, "arm": name, "cols": v}));
            };
            for (bi, (bn, _)) in bases.iter().enumerate() {
                show(bn, &rr[bi][wi]);
            }
            let gi_of = |name: &str| g.iter().position(|c| c.0 == name).unwrap();
            show("vote-undisc", &rr[nb][wi]);
            for p in &picked {
                show(&p.0, &rr[nb + gi_of(&p.0)][wi]);
            }
            // vs best fair baseline per column (baseline chosen by mean in that column)
            for (ci, cn) in COLS.iter().enumerate() {
                if (wi >= 3 && ci >= 2) || (sp.name.contains("early") && ci == 4) {
                    continue;
                }
                let means: Vec<f64> = (0..nb).map(|b| mean_of(colv(&rr[b][wi], ci).into_iter())).collect();
                let hb = ci != 4;
                let bi = (0..nb).max_by(|&a, &b| { let (x, y) = (means[a], means[b]); if hb { x.partial_cmp(&y).unwrap() } else { y.partial_cmp(&x).unwrap() } }).unwrap();
                let bv = colv(&rr[bi][wi], ci);
                let mut cand: Vec<(String, usize)> = vec![("vote-undisc".into(), nb)];
                for p in &picked {
                    cand.push((p.0.clone(), nb + gi_of(&p.0)));
                }
                for (cname, ai) in cand {
                    base_cells.push((fam.label().into(), sp.name.clone(), cname, cn.to_string(), bases[bi].0.clone(), pc(&colv(&rr[ai][wi], ci), &bv)));
                }
            }
            // picked vs undisc (the main comparison)
            for p in &picked {
                let ai = nb + gi_of(&p.0);
                for (ci, cn) in COLS.iter().enumerate() {
                    if (wi >= 3 && ci >= 2) || (sp.name.contains("early") && ci == 4) {
                        continue;
                    }
                    cells.push((fam.label().into(), sp.name.clone(), p.0.clone(), cn.to_string(), pc(&colv(&rr[ai][wi], ci), &colv(&rr[nb][wi], ci))));
                }
            }
            // sensitivity of the whole grid vs undisc (H2000), not part of the Holm family
            for (gi, c) in g.iter().enumerate().skip(1) {
                let q = pc(&colv(&rr[nb + gi][wi], 1), &colv(&rr[nb][wi], 1));
                sens.push(json!({"fam": fam.label(), "world": sp.name, "cfg": c.0, "h2000_diff": q.diff, "ci": [q.lo, q.hi]}));
            }
        }
        clear_nb_cache();
    }
    // Holm families
    let adj = holm_adjust(&cells.iter().map(|c| c.4.p).collect::<Vec<_>>());
    println!("\n=============== picked discount configs vs UNDISCOUNTED vote, paired 95% t-CI, Holm over {} cells (* WIN, ! LOSS) ===============", cells.len());
    println!("{:<5} {:<24} {:<18} {:<6} | {:>8} {:>8} | {:>8} [{:>8},{:>8}] wins | {:>7}", "fam", "world", "cand", "col", "cand", "undisc", "diff", "lo", "hi", "holm");
    let mut nwin = 0;
    let mut nloss = 0;
    for (c, a) in cells.iter().zip(&adj) {
        let better_high = c.3 != "rec";
        let mk = if *a < 0.05 { if (c.4.diff > 0.0) == better_high { nwin += 1; "*" } else { nloss += 1; "!" } } else { "" };
        println!("{:<5} {:<24} {:<18} {:<6} | {:>8.4} {:>8.4} | {:>+8.4} [{:>+8.4},{:>+8.4}] {:>2} | {:>7.4}{}", c.0, c.1, c.2, c.3, c.4.cm, c.4.bm, c.4.diff, c.4.lo, c.4.hi, c.4.wins, a, mk);
    }
    println!("-> Holm WINS {nwin}, Holm LOSSES {nloss} (rec: lower = faster recovery counts as a win)");
    let adj2 = holm_adjust(&base_cells.iter().map(|c| c.5.p).collect::<Vec<_>>());
    println!("\n=============== vs best fair embedding-free baseline (per world, column), Holm over {} cells ===============", base_cells.len());
    for (c, a) in base_cells.iter().zip(&adj2) {
        if c.3 == "H2000" || c.3 == "H500" {
            println!("{:<5} {:<24} {:<18} {:<6} vs {:<16} | {:>8.4} {:>8.4} | {:>+8.4} [{:>+8.4},{:>+8.4}] | {:>7.4}{}", c.0, c.1, c.2, c.3, c.4, c.5.cm, c.5.bm, c.5.diff, c.5.lo, c.5.hi, a, if *a < 0.05 { if c.5.diff > 0.0 { " WIN" } else { " LOSS" } } else { "" });
        }
    }
    // control: ctl-irrel non-inferiority (TOST +-0.005) of each picked config vs undisc
    println!("\n=============== CONTROL ctl-irrel: picked vs undisc, TOST +-{CTL_MARGIN} ===============");
    let mut ctl_json = vec![];
    let mut ctl_p = vec![];
    let mut ctl_meta = vec![];
    for c in cells.iter().filter(|c| c.1.ends_with("ctl-irrel") || c.1.ends_with("stationary")) {
        if c.3 == "H500" || c.3 == "H2000" {
            ctl_meta.push(c);
        }
    }
    // recompute TOST from the raw paired diffs: reuse CI -> need raw; approximate by re-evaluating is expensive, so store p from pc instead
    for c in &ctl_meta {
        ctl_p.push(c.4.tp);
    }
    let cadj = holm_adjust(&ctl_p);
    for (c, a) in ctl_meta.iter().zip(&cadj) {
        println!("{:<5} {:<24} {:<18} {:<6} | diff {:>+8.4} [{:>+8.4},{:>+8.4}] TOST p {:.4} Holm {:.4} equivalent(+-{CTL_MARGIN}): {}", c.0, c.1, c.2, c.3, c.4.diff, c.4.lo, c.4.hi, c.4.tp, a, if *a < 0.05 { "YES" } else { "no" });
        ctl_json.push(json!({"fam": c.0, "world": c.1, "cand": c.2, "col": c.3, "diff": c.4.diff, "ci95": [c.4.lo, c.4.hi], "p_tost": c.4.tp, "holm_p_tost": a, "equivalent": *a < 0.05}));
    }
    let doc = json!({
        "lab": "g10_drift/eval", "confirm": confirm, "seeds": [held[0], held[held.len() - 1]], "picked": picked.iter().map(|p| p.0.clone()).collect::<Vec<_>>(),
        "vs_undisc": cells.iter().zip(&adj).map(|(c, a)| json!({"fam": c.0, "world": c.1, "cand": c.2, "col": c.3, "cand_mean": c.4.cm, "undisc_mean": c.4.bm, "diff": c.4.diff, "ci95": [c.4.lo, c.4.hi], "wins": c.4.wins, "p": c.4.p, "holm_p": a})).collect::<Vec<_>>(),
        "holm_wins": nwin, "holm_losses": nloss,
        "vs_best_baseline": base_cells.iter().zip(&adj2).map(|(c, a)| json!({"fam": c.0, "world": c.1, "cand": c.2, "col": c.3, "best_baseline": c.4, "cand_mean": c.5.cm, "base_mean": c.5.bm, "diff": c.5.diff, "ci95": [c.5.lo, c.5.hi], "holm_p": a})).collect::<Vec<_>>(),
        "controls": ctl_json, "sensitivity_h2000_vs_undisc": sens, "rows": rows, "elapsed_s": t0.elapsed().as_secs_f64()
    });
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out} in {:.0}s", t0.elapsed().as_secs_f64());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let part = arg(&args, "--part", "tune");
    let confirm = args.iter().any(|a| a == "--confirm");
    let held = parse_range(&args, "--held", if confirm { (61, 90) } else { (31, 60) });
    let sel: Value = serde_json::from_str(&std::fs::read_to_string(arg(&args, "--selected", "results/g10_calib_selected.json")).expect("run g10_calib --part tune first")).unwrap();
    match arg(&args, "--src", "mini").as_str() {
        "bge" => { let _ = SRC_CELL.set(Src::Bge); }
        "hash" => { let _ = SRC_CELL.set(Src::Hash); }
        _ => {}
    }
    let tp = arg(&args, "--tuned", "results/g10_drift_tuned.json");
    match part.as_str() {
        "tune" => tune(&sel, &tp),
        "eval" => {
            let tuned: Value = serde_json::from_str(&std::fs::read_to_string(&tp).expect("run --part tune first")).unwrap();
            let sfx = if src() == Src::Mini { String::new() } else { format!("_{}", src().label()) };
            let out = arg(&args, "--out", &if confirm { format!("results/g10_drift_confirm61-90{sfx}.json") } else { format!("results/g10_drift_eval{sfx}.json") });
            eval(&sel, &tuned, &held, confirm, &out)
        }
        p => panic!("unknown --part {p}"),
    }
}
