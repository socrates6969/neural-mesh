//! H3 (R4-A3): CALIBRATED CONFIDENCE-GATED VERIFICATION. A verifier is invoked only when a confidence signal is <= a threshold FIXED from a held-out CALIBRATION split
//! (seeds 101-130, disjoint from tune 1-8 / held 31-60 / confirm 61-90). No online self-calibration. Two independent simulated world generators:
//!   G = W10's g9 critic world (evidence-agreement verifier on facts/reports), H = W19's g18 orchestration world (per-step verification with retry/reroute).
//! Pre-declared threshold rules (declared BEFORE any run, see results/h3_gate_board_entry.md):
//!   R1 = cost-neutral break-even: the threshold maximising the calibration net benefit (headline).  R2 = target error capture 0.90 on the calibration split.
//! Compared with: never-verify, always-verify, W10's online `auto` gate (G) / g18 L4 learned stopping (H), random-skip at EQUAL total cost, oracle gate.
//! Cost accounting (world G): extra = agent reroute calls, tool = verifier executions priced `tp` agent calls each (tp = .1 is g9's default, tp = 1 = an LLM-class verifier);
//!   net(c) = accuracy - c * (extra + tp * tool). World H: g18's own cost (steps + retrievals + cv per verifier call) and g18's utility at mu = .03 (headline) and .1.
//! Usage: cargo run --release --offline --bin h3_gate -- [--part g|h|shift|all] [--held 31-60] [--cal 101-130] [--out results/h3_gate]

use std::fmt::Write as _;
use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g18_common/mod.rs"]
mod g18;
#[path = "g9_common/mod.rs"]
mod g9;
#[path = "h3_common/mod.rs"]
mod h;
use h::*;

const TPS: [f64; 2] = [0.1, 1.0];
const CS: [f64; 3] = [0.02, 0.05, 0.10];
const HEAD_C: f64 = 0.05;
const MARGIN: f64 = 0.01; // pre-declared equivalence margin (TOST): 1 point of accuracy / of net / of quality
const AUTO_GATE_COST: f64 = 0.02; // W10's tuned `auto` gate cost (results/g9_critic.json picks)

// ------------------------------------------------------------------------------------------------ G worlds (fixed before running)
fn cw(name: &str) -> g9::CWorld {
    g9::critic_worlds().into_iter().find(|w| w.name == name).unwrap_or_else(|| panic!("world {name}"))
}
fn g_worlds() -> Vec<GWorld> {
    let b = g9::CWorld::base();
    vec![
        GWorld { name: "G-A1-mixed", cw: cw("A1-mixed"), sep: 1.5 },
        GWorld { name: "G-A2-stale-heavy", cw: cw("A2-stale-heavy"), sep: 1.5 },
        GWorld { name: "G-A9-rare-errors", cw: cw("A9-rare-errors"), sep: 1.5 },
        GWorld { name: "G-A7-weak-retrieval", cw: cw("A7-weak-retrieval"), sep: 1.5 },
        GWorld { name: "G-A8-poisoned", cw: cw("A8-poisoned-evidence"), sep: 1.5 },
        GWorld { name: "G-B1-report", cw: cw("B1-report-mixed"), sep: 1.5 },
        GWorld { name: "G-A1-weaksig(sep.7)", cw: cw("A1-mixed"), sep: 0.7 },
        GWorld { name: "NEG-A1-uninformative", cw: cw("A1-mixed"), sep: 0.0 },
        GWorld { name: "NEG-A9-uninformative", cw: cw("A9-rare-errors"), sep: 0.0 },
        GWorld { name: "CTL-A10-zero-errors", cw: cw("A10-ctl-zero-errors"), sep: 1.5 },
        GWorld { name: "CTL-B3-report-zero", cw: cw("B3-ctl-report-zero"), sep: 1.5 },
        GWorld { name: "G-A1-hi(eps.4)", cw: g9::CWorld { name: "A1-hi", eps: 0.40, ..b.clone() }, sep: 1.5 },
    ]
}

// ------------------------------------------------------------------------------------------------ H worlds (fixed before running)
fn hw(name: &str) -> g18::WorldCfg {
    g18::worlds().into_iter().find(|w| w.name == name).unwrap_or_else(|| panic!("h world {name}"))
}
fn h_worlds() -> Vec<HWorld> {
    let pipe = hw("pipe-main");
    let lat = hw("lat-main");
    vec![
        HWorld { name: "H-pipe-main", cfg: pipe.clone(), sig_c: 0.20 },
        HWorld { name: "H-pipe-pricever(cv.8)", cfg: g18::WorldCfg { name: "pipe-pricever", cv: 0.8, ..pipe.clone() }, sig_c: 0.20 },
        HWorld { name: "H-pipe-badver(sigv.22)", cfg: hw("pipe-badver"), sig_c: 0.20 },
        HWorld { name: "H-pipe-noisyag(.16)", cfg: g18::WorldCfg { name: "pipe-noisyag", noise: 0.16, ..pipe.clone() }, sig_c: 0.20 },
        HWorld { name: "H-lat-main", cfg: lat.clone(), sig_c: 0.20 },
        HWorld { name: "H-lat-pricever(cv.8)", cfg: hw("lat-pricever"), sig_c: 0.20 },
        HWorld { name: "H-lat-highprop", cfg: hw("lat-highprop"), sig_c: 0.20 },
        HWorld { name: "H-pipe-weaksig(sc.5)", cfg: pipe.clone(), sig_c: 0.50 },
        HWorld { name: "CTL-H-rare-pricever", cfg: hw("ctl-rare-pricever"), sig_c: 0.20 },
        HWorld { name: "CTL-H-zero-errors", cfg: g18::WorldCfg { name: "ctl-zero-err", pool: g18::Pool::HighSkill, noise: 0.02, d: [0.0; 4], prop: 0.3, ..pipe.clone() }, sig_c: 0.20 },
        HWorld { name: "NEG-H-pipe-uninformative", cfg: pipe.clone(), sig_c: f64::INFINITY },
        HWorld { name: "NEG-H-lat-uninformative", cfg: lat.clone(), sig_c: f64::INFINITY },
    ]
}

fn within(d: &neural_mesh::stats::PairedDiff) -> bool {
    d.diff.lo > -MARGIN && d.diff.hi < MARGIN
}
fn fmt_d(d: &neural_mesh::stats::PairedDiff) -> String {
    format!("{:+.4} [{:+.4},{:+.4}]", d.diff.mean, d.diff.lo, d.diff.hi)
}
fn col(v: &[GRun], f: &dyn Fn(&GRun) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}

// ------------------------------------------------------------------------------------------------ G evaluation
const GA: [&str; 9] = ["never", "always", "auto(W10-online)", "gate-conf(R1)", "gate-conf(R2)", "gate-cov", "gate-conf+cov(R1)", "rand@R1(equal cost)", "oracle"];

fn g_arms_for(test: &[Vec<GRec>], cal: &GCal, tp: f64, seeds: &[u64]) -> Vec<(String, Vec<GRun>)> {
    let per_seed: Vec<Vec<GRun>> = (0..test.len())
        .into_par_iter()
        .map(|si| {
            let recs = &test[si];
            let never = eval_g(recs, &|_, _| false);
            let always = eval_g(recs, &|_, _| true);
            let auto = auto_g(recs, AUTO_GATE_COST);
            let r1 = eval_g(recs, &|_, r| r.conf <= cal.t_r1);
            let r2 = eval_g(recs, &|_, r| r.conf <= cal.t_r2);
            let cov = eval_g(recs, &|_, r| r.cov >= 1);
            let cc = eval_g(recs, &|_, r| r.cov >= 1 && r.conf <= cal.t_r1cov);
            let (rd, _) = rand_match_g(recs, r1.cost(tp), tp, seeds[si]);
            let or = eval_g(recs, &|_, r| r.wrong);
            vec![never, always, auto, r1, r2, cov, cc, rd, or]
        })
        .collect();
    (0..GA.len()).map(|ai| (GA[ai].to_string(), per_seed.iter().map(|v| v[ai]).collect())).collect()
}

struct GWData {
    gw: GWorld,
    cal: Vec<Vec<GRec>>,
    test: Vec<Vec<GRec>>,
}

fn g_signal_quality(d: &GWData) -> Value {
    let au: Vec<f64> = d.test.iter().map(|r| auroc_low_is_pos(&r.iter().map(|x| x.conf).collect::<Vec<_>>(), &r.iter().map(|x| x.wrong).collect::<Vec<_>>())).collect();
    let au_m = mean(&au.iter().copied().filter(|x| x.is_finite()).collect::<Vec<_>>());
    let au_cov: Vec<f64> = d.test.iter().map(|r| auroc_low_is_pos(&r.iter().map(|x| x.cov as f64).collect::<Vec<_>>(), &r.iter().map(|x| x.wrong).collect::<Vec<_>>())).collect();
    let au_cov_m = mean(&au_cov.iter().copied().filter(|x| x.is_finite()).collect::<Vec<_>>());
    let pool = |recs: &[Vec<GRec>]| -> (Vec<f64>, Vec<bool>) {
        let f: Vec<&GRec> = recs.iter().flatten().collect();
        (f.iter().map(|x| x.conf).collect(), f.iter().map(|x| !x.wrong).collect())
    };
    let (tc, ty) = pool(&d.test);
    let (cc, cy) = pool(&d.cal);
    let ece_raw = ece(&tc, &ty, 10);
    let bins = fit_bins(&cc, &cy, 10);
    let tc2: Vec<f64> = tc.iter().map(|&s| apply_bins(&bins, s)).collect();
    let ece_rec = ece(&tc2, &ty, 10);
    let err = d.test.iter().flatten().filter(|x| x.wrong).count() as f64 / d.test.iter().map(|r| r.len()).sum::<usize>() as f64;
    let err_cal = d.cal.iter().flatten().filter(|x| x.wrong).count() as f64 / d.cal.iter().map(|r| r.len()).sum::<usize>() as f64;
    let flagged = d.test.iter().flatten().filter(|x| x.flagged).count() as f64 / d.test.iter().map(|r| r.len()).sum::<usize>() as f64;
    json!({"auroc_conf": au_m, "auroc_cov": au_cov_m, "ece_raw": ece_raw, "ece_recal_binned_on_cal": ece_rec, "first_draft_err_test": err, "first_draft_err_cal": err_cal, "verifier_flag_rate": flagged})
}

use g9::{holm_by_family, mk_test, test_json, verdict, Test};

fn print_g_table(log: &mut String, name: &str, res: &[(String, Vec<GRun>)], tp: f64, c_: f64) {
    let ai = |n: &str| res.iter().position(|r| r.0 == n).unwrap();
    let alw = &res[ai("always")].1;
    let alw_cost = mean(&col(alw, &|r| r.cost(tp)));
    let _ = writeln!(log, "  [{name}] tp={tp} c={c_}   arm | acc | verifier runs/task | reroute calls/task | cost/task | net | frac verified | error capture | d_acc vs always | cost vs always");
    for (a, v) in res {
        let acc = mean(&col(v, &|r| r.accm()));
        let alw_acc = mean(&col(alw, &|r| r.accm()));
        let cost = mean(&col(v, &|r| r.cost(tp)));
        let _ = writeln!(
            log,
            "   {:<22} {:>6.4} {:>6.3} {:>6.3} {:>6.3} {:>7.4} {:>6.3} {:>6.3} {:>+7.4} {:>7.1}%",
            a,
            acc,
            mean(&col(v, &|r| r.toolpt())),
            mean(&col(v, &|r| r.extrapt())),
            cost,
            mean(&col(v, &|r| r.net(c_, tp))),
            mean(&col(v, &|r| r.verr())),
            mean(&col(v, &|r| r.capture()).iter().copied().filter(|x| x.is_finite()).collect::<Vec<_>>()),
            acc - alw_acc,
            if alw_cost > 1e-12 { 100.0 * (cost / alw_cost - 1.0) } else { 0.0 }
        );
    }
}

fn run_g(log: &mut String, held: &[u64], cal_seeds: &[u64], t0: &Instant) -> (Value, Vec<Test>, Vec<(String, Vec<GWData>)>) {
    let ws = g_worlds();
    let data: Vec<GWData> = ws
        .iter()
        .enumerate()
        .map(|(wi, gw)| {
            let cal: Vec<Vec<GRec>> = cal_seeds.par_iter().map(|&s| build_g_records(gw, s, wi as u64)).collect();
            let test: Vec<Vec<GRec>> = held.par_iter().map(|&s| build_g_records(gw, s, wi as u64)).collect();
            GWData { gw: gw.clone(), cal, test }
        })
        .collect();
    let _ = writeln!(log, "world G records built in {:.1}s ({} worlds x ({} cal + {} test) seeds)", t0.elapsed().as_secs_f64(), ws.len(), cal_seeds.len(), held.len());
    let mut tests: Vec<Test> = vec![];
    let mut jw = vec![];
    for (wi, d) in data.iter().enumerate() {
        let sq = g_signal_quality(d);
        let _ = writeln!(log, "\n=== {} (eps {:.2}, m {}, sep {}) signal quality (test seeds): {}", d.gw.name, d.gw.cw.eps, d.gw.cw.m, d.gw.sep, sq);
        let mut jt = vec![];
        for &tp in &TPS {
            for &c_ in &CS {
                let cal = calibrate_g(&d.cal, c_, tp);
                let res = g_arms_for(&d.test, &cal, tp, held);
                if (c_ - HEAD_C).abs() < 1e-9 {
                    let _ = writeln!(log, "  calibration (30 seeds, {} tasks, {} wrong drafts): t_R1 {:.4} t_R2 {:.4} t_R1cov {:.4}", cal.n, cal.n_wrong, cal.t_r1, cal.t_r2, cal.t_r1cov);
                    print_g_table(log, d.gw.name, &res, tp, c_);
                    // ---- tests (headline c)
                    let ai = |n: &str| res.iter().position(|r| r.0 == n).unwrap();
                    let net = |n: &str| col(&res[ai(n)].1, &|r| r.net(c_, tp));
                    let acc = |n: &str| col(&res[ai(n)].1, &|r| r.accm());
                    let cost = |n: &str| col(&res[ai(n)].1, &|r| r.cost(tp));
                    let fair = ["never", "always", "auto(W10-online)", "rand@R1(equal cost)"];
                    let bf = *fair.iter().max_by(|a, b| mean(&net(a)).partial_cmp(&mean(&net(b))).unwrap()).unwrap();
                    let fam = |s: &str| format!("{s}|tp{tp}");
                    let wn = d.gw.name;
                    for t in ["gate-conf(R1)", "gate-conf(R2)", "gate-conf+cov(R1)", "gate-cov"] {
                        tests.push(mk_test(&fam("G-F1-net-vs-bestfair"), wn, "net", t, bf, &net(t), &net(bf)));
                        tests.push(mk_test(&fam("G-F2-acc-vs-always"), wn, "acc", t, "always", &acc(t), &acc("always")));
                        tests.push(mk_test(&fam("G-F3-cost-reduction-vs-always(always-minus-gate)"), wn, "cost", "always-minus-gate", t, &cost("always"), &cost(t)));
                        tests.push(mk_test(&fam("G-F6-net-vs-auto(W10-online)"), wn, "net", t, "auto(W10-online)", &net(t), &net("auto(W10-online)")));
                        tests.push(mk_test(&fam("G-F6b-acc-vs-auto(W10-online)"), wn, "acc", t, "auto(W10-online)", &acc(t), &acc("auto(W10-online)")));
                        tests.push(mk_test(&fam("G-F4-acc-vs-rand-equal-cost"), wn, "acc", t, "rand@R1(equal cost)", &acc(t), &acc("rand@R1(equal cost)")));
                    }
                    // controls: tie vs the better of never/always (net)
                    let bna = if mean(&net("never")) >= mean(&net("always")) { "never" } else { "always" };
                    tests.push(mk_test(&fam("G-F5-net-vs-bestfixed"), wn, "net", "gate-conf(R1)", bna, &net("gate-conf(R1)"), &net(bna)));
                    tests.push(mk_test(&fam("G-F5-net-vs-bestfixed"), wn, "net", "auto(W10-online)", bna, &net("auto(W10-online)"), &net(bna)));
                }
                for (a, v) in &res {
                    jt.push(json!({"tp": tp, "c": c_, "arm": a, "acc": mean(&col(v, &|r| r.accm())), "verifier_runs_per_task": mean(&col(v, &|r| r.toolpt())), "reroute_calls_per_task": mean(&col(v, &|r| r.extrapt())),
                        "cost_per_task": mean(&col(v, &|r| r.cost(tp))), "net": mean(&col(v, &|r| r.net(c_, tp))), "frac_verified": mean(&col(v, &|r| r.verr())), "t_r1": cal.t_r1, "t_r2": cal.t_r2}));
                }
            }
        }
        jw.push(json!({"world": d.gw.name, "sep": d.gw.sep, "eps": d.gw.cw.eps, "signal": sq, "arms": jt, "idx": wi}));
    }
    holm_by_family(&mut tests);
    (json!(jw), tests, vec![("G".to_string(), data)])
}

fn g_shift(log: &mut String, data: &[GWData], held: &[u64]) -> Value {
    // (calibration world, test world)
    let pairs = [
        ("G-A1-mixed", "G-A9-rare-errors"),
        ("G-A9-rare-errors", "G-A1-mixed"),
        ("G-A1-mixed", "G-A1-hi(eps.4)"),
        ("G-A1-mixed", "CTL-A10-zero-errors"),
        ("G-A1-mixed", "G-A1-weaksig(sep.7)"),
        ("G-A1-mixed", "G-A7-weak-retrieval"),
        ("G-A1-mixed", "G-B1-report"),
    ];
    let find = |n: &str| data.iter().find(|d| d.gw.name == n).unwrap();
    let _ = writeln!(log, "\n################ G SHIFT TESTS: threshold calibrated on world X (seeds 101-130) applied to test world Y (held seeds); c = {HEAD_C}");
    let mut jv = vec![];
    for &tp in &TPS {
        for (cx, ty) in pairs {
            let (dx, dy) = (find(cx), find(ty));
            let cal_x = calibrate_g(&dx.cal, HEAD_C, tp);
            let cal_y = calibrate_g(&dy.cal, HEAD_C, tp);
            let sh = g_arms_for(&dy.test, &cal_x, tp, held);
            let ind = g_arms_for(&dy.test, &cal_y, tp, held);
            let ai = |r: &[(String, Vec<GRun>)], n: &str| r.iter().position(|x| x.0 == n).unwrap();
            let net = |r: &[(String, Vec<GRun>)], n: &str| col(&r[ai(r, n)].1, &|x| x.net(HEAD_C, tp));
            let acc = |r: &[(String, Vec<GRun>)], n: &str| col(&r[ai(r, n)].1, &|x| x.accm());
            let ver = |r: &[(String, Vec<GRun>)], n: &str| mean(&col(&r[ai(r, n)].1, &|x| x.verr()));
            let (n_sh, n_in, n_nv, n_al) = (net(&sh, "gate-conf(R1)"), net(&ind, "gate-conf(R1)"), net(&sh, "never"), net(&sh, "always"));
            let best_fixed = if mean(&n_nv) >= mean(&n_al) { ("never", &n_nv) } else { ("always", &n_al) };
            let d_bf = g9::mk_test("S", ty, "net", "shifted-gate", best_fixed.0, &n_sh, best_fixed.1);
            let d_in = g9::mk_test("S", ty, "net", "shifted-gate", "in-dist-gate", &n_sh, &n_in);
            let d_acc = g9::mk_test("S", ty, "acc", "shifted-gate", "always", &acc(&sh, "gate-conf(R1)"), &acc(&sh, "always"));
            let _ = writeln!(
                log,
                "  tp={tp} cal {cx:<20} -> test {ty:<22} t_cal {:.3} (in-dist {:.3}) | verified frac shifted {:.3} / in-dist {:.3} | net: shifted {:.4} in-dist {:.4} never {:.4} always {:.4} | shifted - bestfixed({}) {} | shifted - in-dist {} | acc shifted - always {}",
                cal_x.t_r1, cal_y.t_r1, ver(&sh, "gate-conf(R1)"), ver(&ind, "gate-conf(R1)"), mean(&n_sh), mean(&n_in), mean(&n_nv), mean(&n_al), best_fixed.0, fmt_d(&d_bf.d), fmt_d(&d_in.d), fmt_d(&d_acc.d)
            );
            jv.push(json!({"tp": tp, "cal_world": cx, "test_world": ty, "t_cal": cal_x.t_r1, "t_indist": cal_y.t_r1, "frac_ver_shifted": ver(&sh, "gate-conf(R1)"), "frac_ver_indist": ver(&ind, "gate-conf(R1)"),
                "net_shifted": mean(&n_sh), "net_indist": mean(&n_in), "net_never": mean(&n_nv), "net_always": mean(&n_al), "d_vs_bestfixed": {"arm": best_fixed.0, "mean": d_bf.d.diff.mean, "lo": d_bf.d.diff.lo, "hi": d_bf.d.diff.hi},
                "d_vs_indist": {"mean": d_in.d.diff.mean, "lo": d_in.d.diff.lo, "hi": d_in.d.diff.hi}, "d_acc_vs_always": {"mean": d_acc.d.diff.mean, "lo": d_acc.d.diff.lo, "hi": d_acc.d.diff.hi}}));
        }
    }
    json!(jv)
}

// ------------------------------------------------------------------------------------------------ H evaluation
fn calibrate_h(hwd: &HWorld, widx: u64, seeds: &[u64]) -> HCal {
    let insts: Vec<g18::Instance> = seeds.par_iter().map(|&s| g18::build(&hwd.cfg, s, widx)).collect();
    // always-verify pass with attempt records (signals + truth)
    let always: Vec<HOut> = seeds.par_iter().zip(insts.par_iter()).map(|(&s, inst)| run_h(hwd, inst, &HGate::Always, s, true)).collect();
    let att: Vec<Att> = always.iter().flat_map(|o| o.att.iter().copied()).collect();
    let bad_rate = att.iter().filter(|a| a.3).count() as f64 / att.len().max(1) as f64;
    let util_of = |gate: HGate| -> (f64, f64) {
        let ms: Vec<HM> = seeds.par_iter().zip(insts.par_iter()).map(|(&s, inst)| summ_h(&run_h(hwd, inst, &gate, s, false))).collect();
        (mean(&ms.iter().map(|m| m.util_c).collect::<Vec<_>>()), mean(&ms.iter().map(|m| m.util_a).collect::<Vec<_>>()))
    };
    let (u_never_c, u_never_a) = util_of(HGate::Never);
    let (u_always_c, u_always_a) = (mean(&always.iter().map(|o| summ_h(o).util_c).collect::<Vec<_>>()), mean(&always.iter().map(|o| summ_h(o).util_a).collect::<Vec<_>>()));
    let mut cal = HCal { r1c: [NEG_INF; 3], r1a: [NEG_INF; 3], r2: [NEG_INF; 3], auroc: [f64::NAN; 3], bad_rate };
    for sig in HSig::ALL {
        let si = sig.idx();
        let sv: Vec<f64> = att.iter().map(|a| att_sig(a, sig)).collect();
        let bd: Vec<bool> = att.iter().map(|a| a.3).collect();
        cal.auroc[si] = auroc_low_is_pos(&sv, &bd);
        let mut sorted = sv.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut cands: Vec<f64> = (1..=9).map(|k| sorted[(k * sorted.len() / 10).min(sorted.len() - 1)]).collect();
        cands.dedup();
        let mut best_c = (u_never_c, NEG_INF);
        let mut best_a = (u_never_a, NEG_INF);
        for &t in &cands {
            let (uc, ua) = util_of(HGate::Thr(sig, t));
            if uc > best_c.0 + 1e-12 {
                best_c = (uc, t);
            }
            if ua > best_a.0 + 1e-12 {
                best_a = (ua, t);
            }
        }
        if u_always_c > best_c.0 + 1e-12 {
            best_c = (u_always_c, POS_INF);
        }
        if u_always_a > best_a.0 + 1e-12 {
            best_a = (u_always_a, POS_INF);
        }
        cal.r1c[si] = best_c.1;
        cal.r1a[si] = best_a.1;
        let bad_sig: Vec<f64> = att.iter().filter(|a| a.3).map(|a| att_sig(a, sig)).collect();
        cal.r2[si] = r2_threshold(&bad_sig, 0.9);
    }
    cal
}

const HA: [&str; 11] = [
    "never(L2)",
    "always(L3)",
    "L4(g18 online stopping)",
    "gate-conf(R1,u.03)",
    "gate-conf(R1,u.1)",
    "gate-conf(R2)",
    "gate-retr(R1,u.03)",
    "gate-post(R1,u.03)",
    "rand@gate-conf(equal cost)",
    "oracle",
    "gate-conf(R1,u.03) [threshold from OTHER world]",
];

fn h_arms_for(hwd: &HWorld, widx: u64, cal: &HCal, shifted: Option<&HCal>, seeds: &[u64]) -> Vec<(String, Vec<HM>)> {
    let t_conf = |k: &HCal| k.r1c[0];
    let per_seed: Vec<Vec<HM>> = seeds
        .par_iter()
        .map(|&s| {
            let inst = g18::build(&hwd.cfg, s, widx);
            let r = |gate: HGate| summ_h(&run_h(hwd, &inst, &gate, s, false));
            let never = r(HGate::Never);
            let always = r(HGate::Always);
            let l4 = summ_g18(&g18::run_orch(&hwd.cfg, &inst, &g18::OrchCfg::level(4, 1.0, 1.0, 0.1), s), &inst);
            let g1 = r(HGate::Thr(HSig::Conf, t_conf(cal)));
            let g1a = r(HGate::Thr(HSig::Conf, cal.r1a[0]));
            let g2 = r(HGate::Thr(HSig::Conf, cal.r2[0]));
            let gr = r(HGate::Thr(HSig::Retr, cal.r1c[1]));
            let gp = r(HGate::Thr(HSig::Post, cal.r1c[2]));
            // random-skip at equal total cost per task as the conf gate: bisection on the per-attempt verify probability
            let (mut lo, mut hi) = (0.0f64, 1.0f64);
            for _ in 0..7 {
                let mid = 0.5 * (lo + hi);
                if r(HGate::Rand(mid)).cost > g1.cost {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let rd = r(HGate::Rand(0.5 * (lo + hi)));
            let or = r(HGate::Oracle);
            let sh = match shifted {
                Some(k) => r(HGate::Thr(HSig::Conf, t_conf(k))),
                None => g1,
            };
            vec![never, always, l4, g1, g1a, g2, gr, gp, rd, or, sh]
        })
        .collect();
    (0..HA.len()).map(|ai| (HA[ai].to_string(), per_seed.iter().map(|v| v[ai]).collect())).collect()
}
fn hcol(v: &[HM], f: &dyn Fn(&HM) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}

fn print_h_table(log: &mut String, name: &str, res: &[(String, Vec<HM>)]) {
    let _ = writeln!(log, "  [{name}]  arm | q (graded) | all-correct | cost/task | verifier calls/task | agent calls/task | util(.03) | util(.1)");
    for (a, v) in res {
        let m = |f: &dyn Fn(&HM) -> f64| mean(&hcol(v, f));
        let _ = writeln!(log, "   {:<46} {:>6.4} {:>6.4} {:>7.3} {:>7.3} {:>7.3} {:>8.4} {:>8.4}", a, m(&|x| x.q), m(&|x| x.ok), m(&|x| x.cost), m(&|x| x.vcalls), m(&|x| x.calls), m(&|x| x.util_c), m(&|x| x.util_a));
    }
}

fn run_h_part(log: &mut String, held: &[u64], cal_seeds: &[u64], t0: &Instant) -> (Value, Vec<Test>, Vec<(HWorld, u64, HCal)>) {
    let ws = h_worlds();
    let cals: Vec<HCal> = ws.iter().enumerate().map(|(wi, w)| calibrate_h(w, wi as u64, cal_seeds)).collect();
    let _ = writeln!(log, "world H calibration (thresholds fixed from seeds {}-{}) done in {:.1}s", cal_seeds[0], cal_seeds[cal_seeds.len() - 1], t0.elapsed().as_secs_f64());
    let mut tests: Vec<Test> = vec![];
    let mut jw = vec![];
    for (wi, w) in ws.iter().enumerate() {
        let cal = &cals[wi];
        let res = h_arms_for(w, wi as u64, cal, None, held);
        let _ = writeln!(
            log,
            "\n=== {} (sig_c {:?}) calibration: bad-attempt rate {:.3}; AUROC(low signal -> bad step) conf {:.3} retr {:.3} post {:.3}; thresholds R1(u.03) conf {:.3} retr {:.3} post {:.3} | R1(u.1) conf {:.3} | R2 conf {:.3} retr {:.3} post {:.3}",
            w.name, w.sig_c, cal.bad_rate, cal.auroc[0], cal.auroc[1], cal.auroc[2], cal.r1c[0], cal.r1c[1], cal.r1c[2], cal.r1a[0], cal.r2[0], cal.r2[1], cal.r2[2]
        );
        print_h_table(log, w.name, &res);
        let ai = |n: &str| res.iter().position(|r| r.0 == n).unwrap();
        let u = |n: &str| hcol(&res[ai(n)].1, &|x| x.util_c);
        let q = |n: &str| hcol(&res[ai(n)].1, &|x| x.q);
        let vc = |n: &str| hcol(&res[ai(n)].1, &|x| x.vcalls);
        let fair = ["never(L2)", "always(L3)", "L4(g18 online stopping)", "rand@gate-conf(equal cost)"];
        let bf = *fair.iter().max_by(|a, b| mean(&u(a)).partial_cmp(&mean(&u(b))).unwrap()).unwrap();
        let wn = w.name;
        for t in ["gate-conf(R1,u.03)", "gate-conf(R2)", "gate-retr(R1,u.03)", "gate-post(R1,u.03)"] {
            tests.push(mk_test("H-F1-util03-vs-bestfair", wn, "util_c", t, bf, &u(t), &u(bf)));
            tests.push(mk_test("H-F2-q-vs-always", wn, "q", t, "always(L3)", &q(t), &q("always(L3)")));
            tests.push(mk_test("H-F3-vcalls-reduction-vs-always(always-minus-gate)", wn, "vcalls", "always-minus-gate", t, &vc("always(L3)"), &vc(t)));
            tests.push(mk_test("H-F6-util03-vs-L4", wn, "util_c", t, "L4(g18 online stopping)", &u(t), &u("L4(g18 online stopping)")));
            tests.push(mk_test("H-F6b-q-vs-L4", wn, "q", t, "L4(g18 online stopping)", &q(t), &q("L4(g18 online stopping)")));
            tests.push(mk_test("H-F6c-vcalls-reduction-vs-L4(L4-minus-gate)", wn, "vcalls", "L4-minus-gate", t, &vc("L4(g18 online stopping)"), &vc(t)));
            tests.push(mk_test("H-F4-q-vs-rand-equal-cost", wn, "q", t, "rand@gate-conf(equal cost)", &q(t), &q("rand@gate-conf(equal cost)")));
        }
        let bna = if mean(&u("never(L2)")) >= mean(&u("always(L3)")) { "never(L2)" } else { "always(L3)" };
        tests.push(mk_test("H-F5-util03-vs-bestfixed", wn, "util_c", "gate-conf(R1,u.03)", bna, &u("gate-conf(R1,u.03)"), &u(bna)));
        tests.push(mk_test("H-F5-util03-vs-bestfixed", wn, "util_c", "L4(g18 online stopping)", bna, &u("L4(g18 online stopping)"), &u(bna)));
        jw.push(json!({"world": wn, "sig_c": if w.sig_c.is_finite() { json!(w.sig_c) } else { json!("inf") }, "bad_rate": cal.bad_rate, "auroc_cal": cal.auroc, "r1c": cal.r1c, "r1a": cal.r1a, "r2": cal.r2,
            "arms": res.iter().map(|(a, v)| json!({"arm": a, "q": mean(&hcol(v, &|x| x.q)), "ok": mean(&hcol(v, &|x| x.ok)), "cost": mean(&hcol(v, &|x| x.cost)), "vcalls": mean(&hcol(v, &|x| x.vcalls)),
                "calls": mean(&hcol(v, &|x| x.calls)), "util_c": mean(&hcol(v, &|x| x.util_c)), "util_a": mean(&hcol(v, &|x| x.util_a))})).collect::<Vec<_>>()}));
    }
    holm_by_family(&mut tests);
    let keep: Vec<(HWorld, u64, HCal)> = ws.into_iter().enumerate().map(|(wi, w)| (w, wi as u64, cals[wi].clone())).collect();
    (json!(jw), tests, keep)
}

fn h_shift(log: &mut String, keep: &[(HWorld, u64, HCal)], held: &[u64]) -> Value {
    let find = |n: &str| keep.iter().find(|k| k.0.name == n).unwrap();
    let pairs = [
        ("H-pipe-main", "CTL-H-rare-pricever"),
        ("H-pipe-main", "H-pipe-noisyag(.16)"),
        ("H-pipe-main", "H-pipe-weaksig(sc.5)"),
        ("H-pipe-main", "H-lat-main"),
        ("H-pipe-main", "CTL-H-zero-errors"),
        ("CTL-H-rare-pricever", "H-pipe-main"),
    ];
    let _ = writeln!(log, "\n################ H SHIFT TESTS: conf-gate threshold calibrated on world X applied to test world Y (held seeds); util(.03)");
    let mut jv = vec![];
    for (cx, ty) in pairs {
        let (kx, ky) = (find(cx), find(ty));
        let res = h_arms_for(&ky.0, ky.1, &ky.2, Some(&kx.2), held);
        let ai = |n: &str| res.iter().position(|r| r.0 == n).unwrap();
        let u = |n: &str| hcol(&res[ai(n)].1, &|x| x.util_c);
        let q = |n: &str| hcol(&res[ai(n)].1, &|x| x.q);
        let vc = |n: &str| mean(&hcol(&res[ai(n)].1, &|x| x.vcalls));
        let shn = "gate-conf(R1,u.03) [threshold from OTHER world]";
        let (a_nv, a_al) = (u("never(L2)"), u("always(L3)"));
        let (bfn, bfv) = if mean(&a_nv) >= mean(&a_al) { ("never", a_nv.clone()) } else { ("always", a_al.clone()) };
        let d_bf = mk_test("S", ty, "util_c", "shifted", bfn, &u(shn), &bfv);
        let d_in = mk_test("S", ty, "util_c", "shifted", "in-dist", &u(shn), &u("gate-conf(R1,u.03)"));
        let d_q = mk_test("S", ty, "q", "shifted", "always", &q(shn), &q("always(L3)"));
        let _ = writeln!(
            log,
            "  cal {cx:<22} -> test {ty:<26} thr_cal {:.3} (in-dist {:.3}) | vcalls shifted {:.3} in-dist {:.3} always {:.3} | util: shifted {:.4} in-dist {:.4} never {:.4} always {:.4} L4 {:.4} | shifted - best fixed({bfn}) {} | shifted - in-dist {} | q shifted - always {}",
            kx.2.r1c[0], ky.2.r1c[0], vc(shn), vc("gate-conf(R1,u.03)"), vc("always(L3)"), mean(&u(shn)), mean(&u("gate-conf(R1,u.03)")), mean(&a_nv), mean(&a_al), mean(&u("L4(g18 online stopping)")), fmt_d(&d_bf.d), fmt_d(&d_in.d), fmt_d(&d_q.d)
        );
        jv.push(json!({"cal_world": cx, "test_world": ty, "thr_cal": kx.2.r1c[0], "thr_indist": ky.2.r1c[0], "vcalls_shifted": vc(shn), "vcalls_indist": vc("gate-conf(R1,u.03)"), "vcalls_always": vc("always(L3)"),
            "util_shifted": mean(&u(shn)), "util_indist": mean(&u("gate-conf(R1,u.03)")), "util_never": mean(&a_nv), "util_always": mean(&a_al), "d_vs_bestfixed": {"arm": bfn, "mean": d_bf.d.diff.mean, "lo": d_bf.d.diff.lo, "hi": d_bf.d.diff.hi},
            "d_vs_indist": {"mean": d_in.d.diff.mean, "lo": d_in.d.diff.lo, "hi": d_in.d.diff.hi}, "d_q_vs_always": {"mean": d_q.d.diff.mean, "lo": d_q.d.diff.lo, "hi": d_q.d.diff.hi}}));
    }
    json!(jv)
}

fn report_tests(log: &mut String, tests: &[Test], label: &str) {
    let _ = writeln!(log, "\n=== {label}: paired 95% t-CI over seeds, Holm per family (WIN/LOSS = Holm-significant; EQUIV = whole CI inside +-{MARGIN})");
    let mut fams: Vec<String> = tests.iter().map(|t| t.family.clone()).collect();
    fams.sort();
    fams.dedup();
    for f in fams {
        let ts: Vec<&Test> = tests.iter().filter(|t| t.family == f).collect();
        let w = ts.iter().filter(|t| verdict(t.adj, t.d.diff.mean) == "WIN").count();
        let l = ts.iter().filter(|t| verdict(t.adj, t.d.diff.mean) == "LOSS").count();
        let e = ts.iter().filter(|t| within(&t.d)).count();
        let _ = writeln!(log, "-- {f}: WIN {w} LOSS {l} EQUIV(+-{MARGIN}) {e} of {}", ts.len());
        for t in ts {
            let v = verdict(t.adj, t.d.diff.mean);
            let eq = if within(&t.d) { "EQUIV" } else { "" };
            let _ = writeln!(log, "   {:<26} {:<26} vs {:<24} {} w/l {}/{} holm {:.4} {} {}", t.world, t.treat, t.vs, fmt_d(&t.d), t.d.wins, t.d.losses, t.adj, v, eq);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let part = g9::arg(&args, "--part", "all");
    let held = g9::parse_range(&args, "--held", (31, 60));
    let cal_seeds = g9::parse_range(&args, "--cal", (101, 130));
    let out = g9::arg(&args, "--out", "results/h3_gate");
    assert!(cal_seeds.iter().all(|s| *s > 90), "calibration seeds must be disjoint from tune/held/confirm");
    let t0 = Instant::now();
    let mut log = String::new();
    let mut doc = serde_json::Map::new();
    doc.insert("held".into(), json!([held[0], held[held.len() - 1]]));
    doc.insert("cal".into(), json!([cal_seeds[0], cal_seeds[cal_seeds.len() - 1]]));
    doc.insert("margin".into(), json!(MARGIN));
    let mut all_tests: Vec<Test> = vec![];
    if part == "g" || part == "all" || part == "shift" {
        let (jw, tests, data) = run_g(&mut log, &held, &cal_seeds, &t0);
        doc.insert("G_worlds".into(), jw);
        all_tests.extend(tests);
        let sh = g_shift(&mut log, &data[0].1, &held);
        doc.insert("G_shift".into(), sh);
    }
    if part == "h" || part == "all" || part == "shift" {
        let (jw, tests, keep) = run_h_part(&mut log, &held, &cal_seeds, &t0);
        doc.insert("H_worlds".into(), jw);
        all_tests.extend(tests);
        let sh = h_shift(&mut log, &keep, &held);
        doc.insert("H_shift".into(), sh);
    }
    report_tests(&mut log, &all_tests, "TESTS");
    doc.insert("tests".into(), json!(all_tests.iter().map(test_json).collect::<Vec<_>>()));
    doc.insert("runtime_s".into(), json!(t0.elapsed().as_secs_f64()));
    let suffix = if held[0] == 31 { "".to_string() } else { format!("_confirm{}-{}", held[0], held[held.len() - 1]) };
    let lp = format!("{out}{suffix}_{part}.log");
    let jp = format!("{out}{suffix}_{part}.json");
    print!("{log}");
    std::fs::write(&lp, &log).unwrap();
    std::fs::write(&jp, serde_json::to_string_pretty(&Value::Object(doc)).unwrap()).unwrap();
    println!("wrote {jp} and {lp} in {:.1}s", t0.elapsed().as_secs_f64());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auroc_known() {
        // wrong items have lower scores -> 1.0 ; identical -> 0.5
        let s = [0.1, 0.2, 0.8, 0.9];
        let p = [true, true, false, false];
        assert!((auroc_low_is_pos(&s, &p) - 1.0).abs() < 1e-12);
        let s2 = [0.5, 0.5, 0.5, 0.5];
        assert!((auroc_low_is_pos(&s2, &p) - 0.5).abs() < 1e-12);
        let s3 = [0.9, 0.8, 0.2, 0.1];
        assert!(auroc_low_is_pos(&s3, &p).abs() < 1e-12);
    }

    #[test]
    fn r1_never_when_negative() {
        let items: Vec<(f64, f64)> = (0..100).map(|i| (i as f64 / 100.0, -0.01)).collect();
        assert_eq!(r1_threshold(&items), NEG_INF);
        let items: Vec<(f64, f64)> = (0..100).map(|i| (i as f64 / 100.0, if i < 30 { 0.1 } else { -0.01 })).collect();
        let t = r1_threshold(&items);
        assert!(t > 0.28 && t < 0.31, "{t}");
    }

    /// my record-based `always` / `auto` reproduce W10's g9 arms exactly (same routed first drafts, same costs)
    #[test]
    fn g_records_match_g9_arms() {
        let gws = g_worlds();
        let gw = &gws[0];
        let seed = 7u64;
        let recs = build_g_records(gw, seed, 0);
        let inst = g9::build_critic(&gw.cw, seed, 0);
        let always = eval_g(&recs, &|_, _| true);
        let never = eval_g(&recs, &|_, _| false);
        let a_ev = g9::run_critic_arm(&inst, &g9::arm("a", g9::ArmKind::Ev { kind: g9::EvKind::Agree, theta: 0.5, unv_flag: false }));
        let a_none = g9::run_critic_arm(&inst, &g9::arm("n", g9::ArmKind::None));
        assert!((always.acc - a_ev.acc_final).abs() < 1e-9, "{} vs {}", always.acc, a_ev.acc_final);
        assert!((always.extra - a_ev.extra_calls).abs() < 1e-9);
        assert!((always.tool - a_ev.tool_units).abs() < 1e-9);
        assert!((never.acc - a_none.acc_final).abs() < 1e-9);
        let auto = auto_g(&recs, 0.02);
        let a_auto = g9::run_critic_arm(&inst, &g9::arm("au", g9::ArmKind::Auto { theta: 0.5, unv_flag: false, cost: 0.02 }));
        assert!((auto.acc - a_auto.acc_final).abs() < 1e-9, "auto {} vs {}", auto.acc, a_auto.acc_final);
        assert!((auto.extra - a_auto.extra_calls).abs() < 1e-9);
        assert!((auto.tool - a_auto.tool_units).abs() < 1e-9);
    }

    /// my gated runner reproduces g18's L2 (never) and L3 (always) bit-for-bit
    #[test]
    fn h_runner_matches_g18_levels() {
        let hws = h_worlds();
        let w = &hws[0];
        let seed = 5u64;
        let inst = g18::build(&w.cfg, seed, 0);
        let mine_l3 = run_h(w, &inst, &HGate::Always, seed, false);
        let mine_l2 = run_h(w, &inst, &HGate::Never, seed, false);
        let l3 = g18::run_orch(&w.cfg, &inst, &g18::OrchCfg::level(3, 1.0, 1.0, 0.1), seed);
        let l2 = g18::run_orch(&w.cfg, &inst, &g18::OrchCfg::level(2, 1.0, 1.0, 0.1), seed);
        assert_eq!(mine_l3.q, l3.q);
        assert_eq!(mine_l3.cost, l3.cost);
        assert_eq!(mine_l2.q, l2.q);
        assert_eq!(mine_l2.cost, l2.cost);
    }
}
