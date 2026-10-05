//! H16-B (R8): PARITY of the productionised `neural_mesh::verifygate` primitives against the source code paths of g9 / h3 / h11, bit-exact.
//!
//!  P1 evidence verifier : library `claim_verdict` / `task_verdict` vs g9 `claim_verdict` / `task_verdict` over every claim of every task of several g9 critic
//!                         worlds, every draft of every agent, 4 rules x 3 thetas (decisions, bad/unverifiable counts: exact equality).
//!  P2 FixedGate (h3)    : library `FixedGate::fit_break_even` / `fit_capture` vs h3 `r1_threshold` / `r2_threshold` / `calibrate_g` (f64 bit equality of the thresholds),
//!                         per-record decisions `should_verify(conf)` vs `conf <= t`, resulting `eval_g` runs (all fields), AUROC / ECE / recalibration bins (bits).
//!                         Also reports what the NEW calibration guard (AUROC >= .6, >= 30 wrong) does in every world (the one intentional behavioural difference).
//!  P3 precision gate    : library `PrecisionGate` / `EvidenceCritic` vs h3 `auto_g` (records) and vs g9 `run_critic_arm(Auto)` (full loop incl. retries), exact.
//!  P4 posterior gate    : library `PosteriorGate::should_verify_sums` vs the h11 expression `nn < nmin || (1 - s/nn) * rv >= cv` on an exhaustive grid.
//! Worlds are the g9 critic worlds h3 used (cal seeds 101-130, test seeds 31-60). Everything is a seeded SIMULATION; no real LLM behaviour.
//! Usage: h16_gate [--out results/h16_gate.json] [--cal 101-130] [--held 31-60]

use std::collections::VecDeque;
use std::time::Instant;

use rayon::prelude::*;
use serde_json::{Value, json};

#[path = "g18_common/mod.rs"]
mod g18;
#[path = "g9_common/mod.rs"]
mod g9;
#[path = "h3_common/mod.rs"]
mod h;
use h::*;

use neural_mesh::verifygate as vg;
use neural_mesh::verifygate::{Action, Assessment, Claim, ClaimOp, EvidenceCritic, EvidenceCriticConfig, EvidenceRule, FixedGate, PrecisionGate, Snippet};

const AUTO_GATE_COST: f64 = 0.02;
const TPS: [f64; 2] = [0.1, 1.0];
const CS: [f64; 3] = [0.02, 0.05, 0.10];

fn cw(name: &str) -> g9::CWorld {
    g9::critic_worlds().into_iter().find(|w| w.name == name).unwrap_or_else(|| panic!("world {name}"))
}
/// the g9 worlds of h3 (a fixed subset: informative, rare errors, weak retrieval, poisoned, report/derived claims, uninformative control, zero-error control)
fn worlds() -> Vec<GWorld> {
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
    ]
}

// ------------------------------------------------------------------------------------------------ P1
fn lib_claim(c: &g9::ClaimSpec) -> Claim {
    Claim { entity: c.e, attr: c.a, attr2: c.a2, op: match c.op { g9::Op::Look => ClaimOp::Look, g9::Op::Sum => ClaimOp::Sum, g9::Op::Diff => ClaimOp::Diff } }
}
fn lib_snip(s: &g9::Snip) -> Snippet {
    Snippet { entity: s.e, attr: s.a, val: s.val, date: s.date, src: s.src }
}
fn rule_pair(i: usize) -> (g9::EvKind, EvidenceRule) {
    [(g9::EvKind::SupportAny, EvidenceRule::SupportAny), (g9::EvKind::SupportEntity, EvidenceRule::SupportEntity), (g9::EvKind::Latest, EvidenceRule::Latest), (g9::EvKind::Agree, EvidenceRule::Agree)][i]
}
fn vmap(v: g9::V) -> vg::Verdict {
    match v {
        g9::V::Ok => vg::Verdict::Ok,
        g9::V::Bad => vg::Verdict::Bad,
        g9::V::Unv => vg::Verdict::Unverifiable,
    }
}

fn p1(ws: &[GWorld], log: &mut String) -> Value {
    let mut rows = vec![];
    let (mut tot_claims, mut tot_tasks, mut mism) = (0u64, 0u64, 0u64);
    for (wi, gw) in ws.iter().enumerate() {
        let (mut nc, mut nt, mut bad_c, mut n_bad_claims_seen, mut n_unv) = (0u64, 0u64, 0u64, 0u64, 0u64);
        for seed in 1..=3u64 {
            let inst = g9::build_critic(&gw.cw, seed, wi as u64);
            for t in inst.tasks.iter().take(400) {
                let claims: Vec<Claim> = t.claims.iter().map(lib_claim).collect();
                let snips: Vec<Snippet> = t.snips.iter().map(lib_snip).collect();
                for agent in 0..g9::NAG {
                    let d = inst.sample(t, agent, 0);
                    for ri in 0..4 {
                        let (gk, lk) = rule_pair(ri);
                        for &theta in &[0.5, 0.67, 1.0] {
                            for ci in 0..t.claims.len() {
                                let a = vmap(g9::claim_verdict(gk, theta, t, ci, d.vals[ci]));
                                let b = vg::claim_verdict(lk, theta, &claims[ci], &snips, d.vals[ci]);
                                nc += 1;
                                if a != b {
                                    bad_c += 1;
                                }
                                if a == vg::Verdict::Bad {
                                    n_bad_claims_seen += 1;
                                }
                                if a == vg::Verdict::Unverifiable {
                                    n_unv += 1;
                                }
                            }
                            let (gb, gu) = g9::task_verdict(gk, theta, t, &d);
                            let la = vg::task_verdict(lk, theta, &claims, &snips, &d.vals);
                            nt += 1;
                            if la != (Assessment { bad: gb, unverifiable: gu }) {
                                bad_c += 1;
                            }
                        }
                    }
                }
            }
        }
        tot_claims += nc;
        tot_tasks += nt;
        mism += bad_c;
        log.push_str(&format!("  P1 {:<24} claim verdicts {:>8} task verdicts {:>7} mismatches {} (Bad seen {}, Unverifiable seen {})\n", gw.name, nc, nt, bad_c, n_bad_claims_seen, n_unv));
        rows.push(json!({"world": gw.name, "claim_verdicts": nc, "task_verdicts": nt, "mismatches": bad_c, "bad_seen": n_bad_claims_seen, "unverifiable_seen": n_unv}));
    }
    log.push_str(&format!("  P1 {}: {} claim verdicts + {} task verdicts, {} mismatches\n", if mism == 0 { "PASS" } else { "FAIL" }, tot_claims, tot_tasks, mism));
    json!({"pass": mism == 0, "claim_verdicts": tot_claims, "task_verdicts": tot_tasks, "mismatches": mism, "worlds": rows})
}

// ------------------------------------------------------------------------------------------------ P2
fn runs_equal(a: &GRun, b: &GRun) -> bool {
    [(a.n, b.n), (a.acc, b.acc), (a.extra, b.extra), (a.tool, b.tool), (a.nver, b.nver), (a.wrong, b.wrong), (a.wrong_ver, b.wrong_ver), (a.del_wrong, b.del_wrong)].iter().all(|(x, y)| x.to_bits() == y.to_bits())
}

fn p2(ws: &[GWorld], cal_seeds: &[u64], held: &[u64], log: &mut String) -> (Value, Vec<(String, Vec<Vec<GRec>>, Vec<Vec<GRec>>)>) {
    let data: Vec<(String, Vec<Vec<GRec>>, Vec<Vec<GRec>>)> = ws
        .iter()
        .enumerate()
        .map(|(wi, gw)| {
            let cal: Vec<Vec<GRec>> = cal_seeds.par_iter().map(|&s| build_g_records(gw, s, wi as u64)).collect();
            let test: Vec<Vec<GRec>> = held.par_iter().map(|&s| build_g_records(gw, s, wi as u64)).collect();
            (gw.name.to_string(), cal, test)
        })
        .collect();
    let (mut n_thr, mut thr_bad, mut n_dec, mut dec_bad, mut n_run, mut run_bad, mut n_stat, mut stat_bad) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let mut jw = vec![];
    for (name, cal, test) in &data {
        let mut jrows = vec![];
        for &tp in &TPS {
            for &c_ in &CS {
                let src = calibrate_g(cal, c_, tp);
                // library: same inputs through the new API
                let ben = |r: &GRec| (r.v_acc - r.first_acc) - c_ * (r.v_extra + tp * r.v_tool);
                let all: Vec<(f64, f64)> = cal.iter().flatten().map(|r| (r.conf, ben(r))).collect();
                let cv: Vec<(f64, f64)> = cal.iter().flatten().filter(|r| r.cov >= 1).map(|r| (r.conf, ben(r))).collect();
                let wr: Vec<f64> = cal.iter().flatten().filter(|r| r.wrong).map(|r| r.conf).collect();
                let g1 = FixedGate::fit_break_even(&all);
                let g2 = FixedGate::fit_capture(&wr, 0.9);
                let g1c = FixedGate::fit_break_even(&cv);
                for (a, b) in [(src.t_r1, g1.threshold), (src.t_r2, g2.threshold), (src.t_r1cov, g1c.threshold)] {
                    n_thr += 1;
                    if a.to_bits() != b.to_bits() {
                        thr_bad += 1;
                    }
                }
                for recs in test {
                    for (t, g) in [(src.t_r1, g1), (src.t_r2, g2)] {
                        let a = eval_g(recs, &|_, r| r.conf <= t);
                        let b = eval_g(recs, &|_, r| g.should_verify(r.conf));
                        n_run += 1;
                        if !runs_equal(&a, &b) {
                            run_bad += 1;
                        }
                        for r in recs {
                            n_dec += 1;
                            if (r.conf <= t) != g.should_verify(r.conf) {
                                dec_bad += 1;
                            }
                        }
                    }
                    // conf+cov combined rule
                    let a = eval_g(recs, &|_, r| r.cov >= 1 && r.conf <= src.t_r1cov);
                    let b = eval_g(recs, &|_, r| r.cov >= 1 && g1c.should_verify(r.conf));
                    n_run += 1;
                    if !runs_equal(&a, &b) {
                        run_bad += 1;
                    }
                }
                if (c_ - 0.05).abs() < 1e-9 && tp == 0.1 {
                    // calibration guard (NEW behaviour): what would it do here?
                    let sc: Vec<f64> = cal.iter().flatten().map(|r| r.conf).collect();
                    let wrong: Vec<bool> = cal.iter().flatten().map(|r| r.wrong).collect();
                    let ben_v: Vec<f64> = cal.iter().flatten().map(|r| ben(r)).collect();
                    let (gate, rep) = vg::calibrate_gate(&sc, &wrong, &ben_v, vg::GateRule::BreakEven, vg::DEFAULT_MIN_AUROC, vg::DEFAULT_MIN_WRONG);
                    jrows.push(json!({"tp": tp, "c": c_, "h3_t_r1": src.t_r1, "lib_fit_t_r1": g1.threshold, "guarded_gate_threshold": gate.threshold, "guard": format!("{:?}", rep.fallback),
                        "cal_auroc": rep.auroc, "cal_n_wrong": rep.n_wrong, "cal_error_rate": rep.error_rate, "ece_raw": rep.ece_raw, "ece_recal_in_sample": rep.ece_recal_in_sample}));
                    log.push_str(&format!("  P2 {:<22} h3 R1 threshold {:>12.6} | library fit {:>12.6} | cal AUROC {:.3}, wrong {}, err {:.3} | guard -> {} (gate {:?})\n",
                        name, src.t_r1, g1.threshold, rep.auroc, rep.n_wrong, rep.error_rate, match rep.fallback { None => "ENABLED".to_string(), Some(f) => format!("{f:?} => always-verify"), }, gate.threshold));
                }
            }
        }
        // statistics: AUROC per seed, pooled ECE / bins (as in h3's g_signal_quality)
        let au_src: Vec<f64> = test.iter().map(|r| auroc_low_is_pos(&r.iter().map(|x| x.conf).collect::<Vec<_>>(), &r.iter().map(|x| x.wrong).collect::<Vec<_>>())).collect();
        let au_lib: Vec<f64> = test.iter().map(|r| vg::auroc_low_is_pos(&r.iter().map(|x| x.conf).collect::<Vec<_>>(), &r.iter().map(|x| x.wrong).collect::<Vec<_>>())).collect();
        for (a, b) in au_src.iter().zip(&au_lib) {
            n_stat += 1;
            if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
                stat_bad += 1;
            }
        }
        let pool = |recs: &[Vec<GRec>]| -> (Vec<f64>, Vec<bool>) {
            let f: Vec<&GRec> = recs.iter().flatten().collect();
            (f.iter().map(|x| x.conf).collect(), f.iter().map(|x| !x.wrong).collect())
        };
        let (tc, ty) = pool(test);
        let (cc, cy) = pool(cal);
        let (e1, e2) = (ece(&tc, &ty, 10), vg::ece(&tc, &ty, 10));
        let (b1, b2) = (fit_bins(&cc, &cy, 10), vg::fit_bins(&cc, &cy, 10));
        let r1: Vec<f64> = tc.iter().map(|&s| apply_bins(&b1, s)).collect();
        let r2: Vec<f64> = tc.iter().map(|&s| vg::apply_bins(&b2, s)).collect();
        let (e3, e4) = (ece(&r1, &ty, 10), vg::ece(&r2, &ty, 10));
        for (a, b) in [(e1, e2), (e3, e4)] {
            n_stat += 1;
            if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
                stat_bad += 1;
            }
        }
        n_stat += 1;
        if b1.len() != b2.len() || b1.iter().zip(&b2).any(|(x, y)| x.0.to_bits() != y.0.to_bits() || x.1.to_bits() != y.1.to_bits()) {
            stat_bad += 1;
        }
        jw.push(json!({"world": name, "cells": jrows}));
    }
    let pass = thr_bad == 0 && dec_bad == 0 && run_bad == 0 && stat_bad == 0;
    log.push_str(&format!("  P2 {}: thresholds {} (bit mismatches {}), per-record decisions {} (mismatches {}), eval_g runs {} (mismatches {}), AUROC/ECE/bins {} (mismatches {})\n",
        if pass { "PASS" } else { "FAIL" }, n_thr, thr_bad, n_dec, dec_bad, n_run, run_bad, n_stat, stat_bad));
    (json!({"pass": pass, "thresholds": n_thr, "threshold_bit_mismatches": thr_bad, "decisions": n_dec, "decision_mismatches": dec_bad, "eval_runs": n_run, "eval_run_mismatches": run_bad, "stats": n_stat, "stat_mismatches": stat_bad, "worlds": jw}), data)
}

// ------------------------------------------------------------------------------------------------ P3
/// library re-implementation of h3 `auto_g` over records, driven by `PrecisionGate`
fn auto_lib(recs: &[GRec], cfg: &EvidenceCriticConfig) -> GRun {
    let mut o = GRun::default();
    let mut gate = PrecisionGate::new();
    for r in recs {
        let tick = r.tick;
        gate.flush(tick);
        o.n += 1.0;
        o.nver += 1.0;
        o.tool += 1.0;
        let mut acc = r.first_acc;
        if r.flagged && gate.should_act(cfg) {
            o.extra += 1.0;
            o.tool += 1.0;
            acc = r.v_acc;
            if r.revealed {
                gate.enqueue_retry(tick + G_DELAY as u64, r.second_acc);
            }
        }
        o.acc += acc;
        if acc < 1.0 - 1e-12 {
            o.del_wrong += 1.0;
        }
        if r.wrong {
            o.wrong += 1.0;
            o.wrong_ver += 1.0;
        }
        if r.revealed {
            gate.enqueue_first(tick + G_DELAY as u64, r.flagged, r.wrong);
        }
    }
    o
}

// copy of the (private) g9 Tracker pieces needed for the retry-agent choice
struct Tr {
    sum: Vec<Vec<f64>>,
    cnt: Vec<Vec<f64>>,
    q: VecDeque<(u64, usize, usize, f64)>,
}
impl Tr {
    fn flush(&mut self, tick: u64) {
        while self.q.front().is_some_and(|e| e.0 <= tick) {
            let (_, fam, ag, acc) = self.q.pop_front().unwrap();
            self.sum[fam][ag] += acc;
            self.cnt[fam][ag] += 1.0;
        }
    }
    fn best_other(&self, fam: usize, not: usize, salt: u64) -> usize {
        let mut best = usize::MAX;
        let mut bv = -1.0;
        for a in 0..g9::NAG {
            if a == not {
                continue;
            }
            let m = (self.sum[fam][a] + 0.5 * 2.0) / (self.cnt[fam][a] + 2.0) + 1e-6 * ((g9::hash(1, salt, a as u64, 0) % 1000) as f64);
            if m > bv {
                bv = m;
                best = a;
            }
        }
        best
    }
}

/// the g9 `Auto` arm re-implemented through `EvidenceCritic` (assess -> decide -> reroute -> after_reroute) over the same instance
fn auto_critic_full(inst: &g9::CInst, cfg: &EvidenceCriticConfig) -> g9::CRun {
    use neural_mesh::research::*;
    use neural_mesh::router::Rng;
    let delay = 15usize;
    let rcfg = ResearchConfig { reuse: true, delay, succ_thr: 0.99, ..ResearchConfig::default() };
    let mut sess = ResearchSession::new(g9::NAG, inst.kw.clone(), rcfg);
    let mut ex = g9::CExec { inst, strong: false, ev_check: None };
    let mut rng = Rng::new(inst.seed ^ 0x5EED);
    let cands: Vec<usize> = (0..g9::NAG).collect();
    let mut tr = Tr { sum: vec![vec![0.0; g9::NAG]; g9::NFAM], cnt: vec![vec![0.0; g9::NAG]; g9::NFAM], q: VecDeque::new() };
    let mut critic = EvidenceCritic::new(cfg.clone());
    let mut out = g9::CRun::default();
    for (i, t) in inst.tasks.iter().enumerate() {
        let tick = (i + 1) as u64;
        tr.flush(tick);
        critic.gate.flush(tick);
        let rep = sess.run_task(&mut ex, &t.task, &cands, &mut rng);
        let agent = rep.agent;
        let first = inst.sample(t, agent, 0);
        let claims: Vec<Claim> = t.claims.iter().map(lib_claim).collect();
        let snips: Vec<Snippet> = t.snips.iter().map(lib_snip).collect();
        let (mut extra, mut tool) = (0.0, 1.0);
        let mut deliver_flag = false;
        let mut fin = first.clone();
        let a1 = critic.assess(&claims, &snips, &first.vals);
        let flagged = a1.flagged(cfg.flag_unverifiable);
        match critic.decide(&a1) {
            Action::Accept => {}
            Action::DeliverUnverified => deliver_flag = true,
            Action::Reroute => {
                let r = tr.best_other(t.fam, agent, tick);
                extra += 1.0;
                tool += 1.0;
                let second = inst.sample(t, r, 0);
                if t.revealed {
                    critic.gate.enqueue_retry(tick + delay as u64, second.acc());
                }
                let a2 = critic.assess(&claims, &snips, &second.vals);
                let o = critic.after_reroute(&a1, &a2);
                deliver_flag = o.unverified;
                fin = if o.deliver_second { second } else { first.clone() };
            }
        }
        let wrong = first.any_wrong();
        out.n += 1.0;
        out.acc_first += first.acc();
        out.acc_final += fin.acc();
        out.extra_calls += extra;
        out.tool_units += tool;
        match (wrong, flagged) {
            (true, true) => out.tp += 1.0,
            (true, false) => out.fn_ += 1.0,
            (false, true) => out.fp += 1.0,
            (false, false) => out.tn += 1.0,
        }
        let fin_wrong = fin.any_wrong();
        if fin_wrong {
            out.delivered_wrong += 1.0;
            if !(flagged && deliver_flag) {
                out.silent += 1.0;
            }
        }
        if flagged && deliver_flag {
            out.delivered_flagged += 1.0;
        }
        if !wrong && fin_wrong {
            out.harmed += 1.0;
        }
        if wrong && !fin_wrong {
            out.rescued += 1.0;
        }
        if t.revealed {
            tr.q.push_back((tick + delay as u64, t.fam, agent, first.acc()));
            critic.gate.enqueue_first(tick + delay as u64, flagged, wrong);
        }
    }
    out
}

fn crun_equal(a: &g9::CRun, b: &g9::CRun) -> bool {
    [(a.n, b.n), (a.acc_first, b.acc_first), (a.acc_final, b.acc_final), (a.extra_calls, b.extra_calls), (a.tool_units, b.tool_units), (a.tp, b.tp), (a.fp, b.fp), (a.fn_, b.fn_), (a.tn, b.tn),
        (a.silent, b.silent), (a.delivered_wrong, b.delivered_wrong), (a.delivered_flagged, b.delivered_flagged), (a.harmed, b.harmed), (a.rescued, b.rescued)]
        .iter()
        .all(|(x, y)| x.to_bits() == y.to_bits())
}

fn p3(ws: &[GWorld], data: &[(String, Vec<Vec<GRec>>, Vec<Vec<GRec>>)], log: &mut String) -> Value {
    let cfg = EvidenceCriticConfig { call_cost: AUTO_GATE_COST, ..EvidenceCriticConfig::default() };
    // (a) record level: library PrecisionGate vs h3 auto_g (cal + test seeds)
    let (mut n_rec, mut bad_rec, mut acted_total) = (0u64, 0u64, 0.0f64);
    for (_, cal, test) in data {
        for recs in cal.iter().chain(test.iter()) {
            let a = auto_g(recs, AUTO_GATE_COST);
            let b = auto_lib(recs, &cfg);
            n_rec += 1;
            acted_total += a.extra;
            if !runs_equal(&a, &b) {
                bad_rec += 1;
            }
        }
    }
    log.push_str(&format!("  P3a record level: {} (world,seed) runs of h3 auto_g vs library PrecisionGate, {} mismatches ({} reroutes taken in total)\n", n_rec, bad_rec, acted_total));
    // (b) full loop: g9 run_critic_arm(Auto, theta .5, no unverifiable flag, cost .02) vs EvidenceCritic
    let arm = g9::arm("auto", g9::ArmKind::Auto { theta: 0.5, unv_flag: false, cost: AUTO_GATE_COST });
    let (mut n_full, mut bad_full) = (0u64, 0u64);
    let mut rows = vec![];
    for (wi, gw) in ws.iter().enumerate() {
        let seeds_: Vec<u64> = (31..=36).collect();
        let res: Vec<(bool, f64, f64, f64)> = seeds_
            .par_iter()
            .map(|&s| {
                let inst = g9::build_critic(&gw.cw, s, wi as u64);
                let a = g9::run_critic_arm(&inst, &arm);
                let b = auto_critic_full(&inst, &cfg);
                (crun_equal(&a, &b), a.acc_final / a.n, a.extra_calls / a.n, a.tp)
            })
            .collect();
        let bad = res.iter().filter(|r| !r.0).count() as u64;
        n_full += res.len() as u64;
        bad_full += bad;
        let acc = res.iter().map(|r| r.1).sum::<f64>() / res.len() as f64;
        let ex = res.iter().map(|r| r.2).sum::<f64>() / res.len() as f64;
        log.push_str(&format!("  P3b full loop {:<22} {} seeds: g9 run_critic_arm(Auto) vs EvidenceCritic mismatches {} (acc {:.4}, reroute calls/task {:.4})\n", gw.name, res.len(), bad, acc, ex));
        rows.push(json!({"world": gw.name, "seeds": res.len(), "mismatches": bad, "acc_final": acc, "extra_calls_per_task": ex}));
    }
    let pass = bad_rec == 0 && bad_full == 0;
    log.push_str(&format!("  P3 {}: record level {} runs ({} mismatches), full loop {} runs ({} mismatches)\n", if pass { "PASS" } else { "FAIL" }, n_rec, bad_rec, n_full, bad_full));
    json!({"pass": pass, "record_runs": n_rec, "record_mismatches": bad_rec, "full_runs": n_full, "full_mismatches": bad_full, "worlds": rows})
}

// ------------------------------------------------------------------------------------------------ P4
fn p4(log: &mut String) -> Value {
    let mut n = 0u64;
    let mut bad = 0u64;
    let mut verified = 0u64;
    for &rv in &[0.5, 0.8, 1.0, 1.25, 2.0] {
        for &cv in &[0.0, 0.05, 0.1, 0.15, 0.25, 0.3, 0.5, 0.8, 1.0, 1.5] {
            for &nmin in &[0.0, 1.0, 3.0, 5.0] {
                let g = vg::PosteriorGate { rv, cv, nmin };
                for nn in 1..=40u32 {
                    let nn = nn as f64;
                    for si in 0..=(nn as u32 * 4) {
                        let s = si as f64 / 4.0; // sum of graded qualities in [0, nn*...]; includes s > nn (not expected but harmless)
                        // h11 pworld.rs `VMode::Gate`: nn < nmin || (1.0 - s / nn) * rv >= w.cv
                        let reference = nn < nmin || (1.0 - s / nn) * rv >= cv;
                        let mine = g.should_verify_sums(s, nn);
                        n += 1;
                        verified += reference as u64;
                        if reference != mine {
                            bad += 1;
                        }
                    }
                }
            }
        }
    }
    log.push_str(&format!("  P4 {}: {} (rv, cv, nmin, s, n) points vs the h11 expression ({} verify), mismatches {}\n", if bad == 0 { "PASS" } else { "FAIL" }, n, verified, bad));
    json!({"pass": bad == 0, "points": n, "verify_decisions": verified, "mismatches": bad})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let outp = g9::arg(&args, "--out", "results/h16_gate.json");
    let held = g9::parse_range(&args, "--held", (31, 60));
    let cal = g9::parse_range(&args, "--cal", (101, 130));
    let t0 = Instant::now();
    let mut log = String::new();
    let ws = worlds();
    log.push_str(&format!("worlds {} ; cal seeds {}-{} ; test seeds {}-{}\n\n== P1 evidence verifier (g9 claim_verdict / task_verdict) ==\n", ws.len(), cal[0], cal[cal.len() - 1], held[0], held[held.len() - 1]));
    let r1 = p1(&ws, &mut log);
    log.push_str("\n== P2 FixedGate (h3 r1/r2 thresholds, decisions, stats) ==\n");
    let (r2, data) = p2(&ws, &cal, &held, &mut log);
    log.push_str("\n== P3 precision gate / evidence critic (h3 auto_g, g9 run_critic_arm Auto) ==\n");
    let r3 = p3(&ws, &data, &mut log);
    log.push_str("\n== P4 posterior gate (h11) ==\n");
    let r4 = p4(&mut log);
    let all = [&r1, &r2, &r3, &r4].iter().all(|r| r["pass"] == true);
    log.push_str(&format!("\nOVERALL {} ({:.1}s)\n", if all { "PASS" } else { "FAIL" }, t0.elapsed().as_secs_f64()));
    let doc = json!({"overall_pass": all, "p1_evidence_verifier": r1, "p2_fixed_gate": r2, "p3_precision_gate": r3, "p4_posterior_gate": r4, "cal_seeds": [cal[0], cal[cal.len() - 1]], "test_seeds": [held[0], held[held.len() - 1]], "wall_s": t0.elapsed().as_secs_f64()});
    std::fs::write(&outp, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    let _ = std::fs::write(outp.replace(".json", ".txt"), &log);
    println!("{log}");
    if !all {
        std::process::exit(1);
    }
}
