//! h3 shared pieces (R4-A3): CALIBRATED CONFIDENCE-GATED VERIFICATION.
//! Two independent simulated world generators:
//!   * world G = W10's g9 critic world (src/bin/g9_common): KB facts, retrieved evidence snippets, 4 real error types, evidence-agreement verifier ("agree").
//!   * world H = W19's g18 orchestration world (src/bin/g18_common): 3-6 step dependent tasks, noisy per-step verifier, retry/reroute.
//! A verifier is invoked only when a confidence signal is below a threshold FIXED from a held-out CALIBRATION split of seeds (101-130,
//! disjoint from tune 1-8 / held 31-60 / confirm 61-90). Nothing here adapts its threshold online.
//! Everything is a seeded SIMULATION I wrote; the confidence signals are SYNTHETIC (their informativeness is a world parameter), not measured from any LLM.
#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::research::*;
use neural_mesh::router::{Rng, Router, Task};

use super::g18 as c;
use super::g9 as g;

pub const NEG_INF: f64 = -1.0e9;
pub const POS_INF: f64 = 1.0e9;

pub fn gauss(seed: u64, a: u64, b: u64, c_: u64) -> f64 {
    let u1 = g::uni(seed, a, b, c_).max(1e-12);
    let u2 = g::uni(seed, a, b, c_ ^ 0xABCD);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
pub fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

// =============================================================================================
// signal-quality metrics

/// AUROC of "LOW score predicts positive" (positive = the draft/step is wrong). 0.5 = uninformative, 1 = perfect, <0.5 = anti-informative.
pub fn auroc_low_is_pos(s: &[f64], pos: &[bool]) -> f64 {
    let n = s.len();
    let np = pos.iter().filter(|p| **p).count();
    let nn = n - np;
    if np == 0 || nn == 0 {
        return f64::NAN;
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && s[idx[j + 1]] == s[idx[i]] {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[idx[k]] = r;
        }
        i = j + 1;
    }
    let sum_neg: f64 = (0..n).filter(|&k| !pos[k]).map(|k| ranks[k]).sum();
    (sum_neg - nn as f64 * (nn as f64 + 1.0) / 2.0) / (np as f64 * nn as f64)
}

/// ECE with equal-width bins; `p` = predicted probability of the event, `y` = event happened.
pub fn ece(p: &[f64], y: &[bool], nb: usize) -> f64 {
    let n = p.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut cnt = vec![0.0; nb];
    let mut sp = vec![0.0; nb];
    let mut sy = vec![0.0; nb];
    for (pi, yi) in p.iter().zip(y) {
        let b = ((pi.clamp(0.0, 1.0) * nb as f64) as usize).min(nb - 1);
        cnt[b] += 1.0;
        sp[b] += pi;
        sy[b] += f64::from(*yi);
    }
    (0..nb).map(|b| if cnt[b] > 0.0 { cnt[b] / n as f64 * (sp[b] / cnt[b] - sy[b] / cnt[b]).abs() } else { 0.0 }).sum()
}
/// equal-mass bin recalibration map fitted on (score, event) pairs: Vec<(upper score bound, event rate)>
pub fn fit_bins(s: &[f64], y: &[bool], nb: usize) -> Vec<(f64, f64)> {
    let mut idx: Vec<usize> = (0..s.len()).collect();
    idx.sort_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap());
    let n = idx.len();
    let mut out = vec![];
    for b in 0..nb {
        let lo = b * n / nb;
        let hi = ((b + 1) * n / nb).max(lo + 1).min(n);
        let ys: f64 = idx[lo..hi].iter().map(|&i| f64::from(y[i])).sum();
        out.push((s[idx[hi - 1]], ys / (hi - lo) as f64));
    }
    out
}
pub fn apply_bins(bins: &[(f64, f64)], s: f64) -> f64 {
    for (ub, r) in bins {
        if s <= *ub {
            return *r;
        }
    }
    bins.last().map(|b| b.1).unwrap_or(0.0)
}

// =============================================================================================
// WORLD G (g9 critic world) -------------------------------------------------------------------

pub const G_DELAY: usize = 15;

/// g9 critic world + the informativeness of the agent's self-reported confidence (sep = separation, in sd units, of the confidence of wrong vs correct drafts)
#[derive(Clone)]
pub struct GWorld {
    pub name: &'static str,
    pub cw: g::CWorld,
    pub sep: f64,
}

/// one per task. The routed first draft and everything that a verification WOULD do are arm-independent (g9 Tracker state depends only on first-draft
/// grades), so any gate is an exact per-task selection between "no verification" and "verify" (common random numbers).
#[derive(Clone, Debug)]
pub struct GRec {
    pub tick: u64,
    pub revealed: bool,
    pub first_acc: f64,
    pub wrong: bool,
    /// synthetic self-reported confidence of the first draft (high = "I am right")
    pub conf: f64,
    /// retrieval coverage: min over needed (entity,attribute) pairs of the number of distinct sources holding evidence for it
    pub cov: usize,
    /// the evidence verifier (agree, theta .5) flags the first draft
    pub flagged: bool,
    /// outcome IF the verifier is run: delivered accuracy, extra agent calls, tool (verifier) executions
    pub v_acc: f64,
    pub v_extra: f64,
    pub v_tool: f64,
    pub second_acc: f64,
}

struct GTr {
    sum: Vec<Vec<f64>>,
    cnt: Vec<Vec<f64>>,
    q: VecDeque<(u64, usize, usize, f64)>,
}
impl GTr {
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
        for a in 0..g::NAG {
            if a == not {
                continue;
            }
            let m = (self.sum[fam][a] + 0.5 * 2.0) / (self.cnt[fam][a] + 2.0) + 1e-6 * ((g::hash(1, salt, a as u64, 0) % 1000) as f64);
            if m > bv {
                bv = m;
                best = a;
            }
        }
        best
    }
}

pub fn build_g_records(gw: &GWorld, seed: u64, widx: u64) -> Vec<GRec> {
    let inst = g::build_critic(&gw.cw, seed, widx);
    let cfg = ResearchConfig { reuse: true, delay: G_DELAY, succ_thr: 0.99, ..ResearchConfig::default() };
    let mut sess = ResearchSession::new(g::NAG, inst.kw.clone(), cfg);
    let mut ex = g::CExec { inst: &inst, strong: false, ev_check: None };
    let mut rng = Rng::new(inst.seed ^ 0x5EED);
    let cands: Vec<usize> = (0..g::NAG).collect();
    let mut tr = GTr { sum: vec![vec![0.0; g::NAG]; g::NFAM], cnt: vec![vec![0.0; g::NAG]; g::NFAM], q: VecDeque::new() };
    let mut out = Vec::with_capacity(inst.tasks.len());
    for (i, t) in inst.tasks.iter().enumerate() {
        let tick = (i + 1) as u64;
        tr.flush(tick);
        let rep = sess.run_task(&mut ex, &t.task, &cands, &mut rng);
        let agent = rep.agent;
        let first = inst.sample(t, agent, 0);
        let wrong = first.any_wrong();
        let (bad, _unv) = g::task_verdict(g::EvKind::Agree, 0.5, t, &first);
        let flagged = bad > 0;
        let (v_acc, v_extra, v_tool, second_acc);
        if flagged {
            let r = tr.best_other(t.fam, agent, tick);
            let second = inst.sample(t, r, 0);
            let (b2, _) = g::task_verdict(g::EvKind::Agree, 0.5, t, &second);
            let fin = if b2 == 0 || b2 <= bad { &second } else { &first };
            v_acc = fin.acc();
            v_extra = 1.0;
            v_tool = 2.0;
            second_acc = second.acc();
        } else {
            v_acc = first.acc();
            v_extra = 0.0;
            v_tool = 1.0;
            second_acc = first.acc();
        }
        // synthetic self-reported confidence: correct ~ N(.8 + agent bias, .15); wrong ~ N(.8 - sep*.15 + agent bias, .15)
        let bias = (g::uni(inst.seed, 0x71, agent as u64, 0) - 0.5) * 0.2;
        let mu = if wrong { 0.8 - gw.sep * 0.15 } else { 0.8 };
        let conf = (mu + bias + 0.15 * gauss(inst.seed, 0x72, t.task.id, agent as u64)).clamp(0.0, 1.0);
        let mut need: Vec<(usize, usize)> = vec![];
        for cl in &t.claims {
            need.push((cl.e, cl.a));
            if cl.op != g::Op::Look {
                need.push((cl.e, cl.a2));
            }
        }
        let cov = need
            .iter()
            .map(|&(e, a)| {
                let mut m = [false; 3];
                for s in &t.snips {
                    if s.e == e && s.a == a {
                        m[s.src as usize] = true;
                    }
                }
                m.iter().filter(|x| **x).count()
            })
            .min()
            .unwrap_or(0);
        if t.revealed {
            tr.q.push_back((tick + G_DELAY as u64, t.fam, agent, first.acc()));
        }
        out.push(GRec { tick, revealed: t.revealed, first_acc: first.acc(), wrong, conf, cov, flagged, v_acc, v_extra, v_tool, second_acc });
    }
    out
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GRun {
    pub n: f64,
    pub acc: f64,
    pub extra: f64,
    pub tool: f64,
    /// number of tasks on which the verifier was invoked
    pub nver: f64,
    /// wrong first drafts that were verified / that exist
    pub wrong: f64,
    pub wrong_ver: f64,
    /// delivered wrong answers (final accuracy < 1)
    pub del_wrong: f64,
}
impl GRun {
    pub fn accm(&self) -> f64 {
        self.acc / self.n
    }
    pub fn verr(&self) -> f64 {
        self.nver / self.n
    }
    pub fn toolpt(&self) -> f64 {
        self.tool / self.n
    }
    pub fn extrapt(&self) -> f64 {
        self.extra / self.n
    }
    pub fn cost(&self, tp: f64) -> f64 {
        (self.extra + tp * self.tool) / self.n
    }
    pub fn net(&self, c_: f64, tp: f64) -> f64 {
        self.accm() - c_ * self.cost(tp)
    }
    pub fn capture(&self) -> f64 {
        if self.wrong > 0.0 { self.wrong_ver / self.wrong } else { f64::NAN }
    }
}

pub fn eval_g(recs: &[GRec], ver: &dyn Fn(usize, &GRec) -> bool) -> GRun {
    let mut o = GRun::default();
    for (i, r) in recs.iter().enumerate() {
        o.n += 1.0;
        let v = ver(i, r);
        if v {
            o.acc += r.v_acc;
            o.extra += r.v_extra;
            o.tool += r.v_tool;
            o.nver += 1.0;
            if r.v_acc < 1.0 - 1e-12 {
                o.del_wrong += 1.0;
            }
        } else {
            o.acc += r.first_acc;
            if r.first_acc < 1.0 - 1e-12 {
                o.del_wrong += 1.0;
            }
        }
        if r.wrong {
            o.wrong += 1.0;
            if v {
                o.wrong_ver += 1.0;
            }
        }
    }
    o
}

/// W10's shipped `auto` arm (evidence verifier run on EVERY task; the reroute is acted on only while the ONLINE precision estimate beats break-even), re-implemented over records.
pub fn auto_g(recs: &[GRec], gate_cost: f64) -> GRun {
    let mut o = GRun::default();
    let (mut fl_w, mut fl_n, mut wr, mut nn, mut ret_w, mut ret_n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let mut q: VecDeque<(u64, bool, bool)> = VecDeque::new();
    let mut rq: VecDeque<(u64, f64)> = VecDeque::new();
    for r in recs {
        let tick = r.tick;
        while rq.front().is_some_and(|e| e.0 <= tick) {
            let (_, acc) = rq.pop_front().unwrap();
            ret_n += 1.0;
            ret_w += 1.0 - acc;
        }
        while q.front().is_some_and(|e| e.0 <= tick) {
            let (_, flagged, wrong) = q.pop_front().unwrap();
            nn += 1.0;
            if wrong {
                wr += 1.0;
            }
            if flagged {
                fl_n += 1.0;
                if wrong {
                    fl_w += 1.0;
                }
            }
        }
        o.n += 1.0;
        o.nver += 1.0;
        o.tool += 1.0;
        let mut acc = r.first_acc;
        if r.flagged {
            let p_wf = (fl_w + 1.0) / (fl_n + 2.0);
            let err_hat = (wr + 1.0) / (nn + 4.0);
            let ret_hat = (ret_w + 8.0 * (err_hat + 0.05)) / (ret_n + 8.0);
            let be = ret_hat + gate_cost * (1.0 + 2.0 * g::C_TOOL);
            if p_wf > be {
                o.extra += 1.0;
                o.tool += 1.0;
                acc = r.v_acc;
                if r.revealed {
                    rq.push_back((tick + G_DELAY as u64, r.second_acc));
                }
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
            q.push_back((tick + G_DELAY as u64, r.flagged, r.wrong));
        }
    }
    o
}

/// R1 (cost-neutral break-even): verify iff score <= t, t maximising the cumulative net benefit on the calibration pool; if no prefix has positive net benefit -> never verify.
pub fn r1_threshold(items: &[(f64, f64)]) -> f64 {
    let mut v: Vec<(f64, f64)> = items.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let n = v.len();
    let (mut cum, mut best, mut bk) = (0.0, 0.0, 0usize);
    for k in 0..n {
        cum += v[k].1;
        let boundary = k + 1 == n || v[k + 1].0 > v[k].0;
        if boundary && cum > best + 1e-12 {
            best = cum;
            bk = k + 1;
        }
    }
    if bk == 0 {
        NEG_INF
    } else if bk == n {
        POS_INF
    } else {
        0.5 * (v[bk - 1].0 + v[bk].0)
    }
}
/// R2 (target error capture): verify iff score <= t, t = the `cap` quantile of the scores of WRONG items (needs >= 30 wrong items, else never verify)
pub fn r2_threshold(scores_wrong: &[f64], cap: f64) -> f64 {
    if scores_wrong.len() < 30 {
        return NEG_INF;
    }
    let mut v = scores_wrong.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((cap * v.len() as f64).ceil() as usize).clamp(1, v.len());
    v[k - 1]
}

#[derive(Clone, Copy, Debug)]
pub struct GCal {
    pub t_r1: f64,
    pub t_r2: f64,
    pub t_r1cov: f64,
    pub n_wrong: usize,
    pub n: usize,
}
pub fn calibrate_g(cal: &[Vec<GRec>], c_: f64, tp: f64) -> GCal {
    let ben = |r: &GRec| (r.v_acc - r.first_acc) - c_ * (r.v_extra + tp * r.v_tool);
    let all: Vec<(f64, f64)> = cal.iter().flatten().map(|r| (r.conf, ben(r))).collect();
    let cv: Vec<(f64, f64)> = cal.iter().flatten().filter(|r| r.cov >= 1).map(|r| (r.conf, ben(r))).collect();
    let wr: Vec<f64> = cal.iter().flatten().filter(|r| r.wrong).map(|r| r.conf).collect();
    GCal { t_r1: r1_threshold(&all), t_r2: r2_threshold(&wr, 0.9), t_r1cov: r1_threshold(&cv), n_wrong: wr.len(), n: all.len() }
}

/// random-skip at EQUAL total cost (extra + tp*tool) to a target run: bisection on the per-task verify probability (CRN uniform per task)
pub fn rand_match_g(recs: &[GRec], target_cost_pt: f64, tp: f64, seed: u64) -> (GRun, f64) {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let u = |i: usize| g::uni(seed, 0x75, i as u64, 0);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        let r = eval_g(recs, &|i, _| u(i) < mid);
        if r.cost(tp) > target_cost_pt {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let p = 0.5 * (lo + hi);
    (eval_g(recs, &|i, _| u(i) < p), p)
}

pub const G_ARMS: [&str; 10] =
    ["never", "always", "auto(W10-online)", "gate-conf(R1)", "gate-conf(R2)", "gate-cov", "gate-conf+cov(R1)", "rand@gate-conf(R1)", "oracle", "gate-conf(R1,t=cal-of-other)"];

// =============================================================================================
// WORLD H (g18 orchestration world) -----------------------------------------------------------

#[derive(Clone)]
pub struct HWorld {
    pub name: &'static str,
    pub cfg: c::WorldCfg,
    /// noise sd of the synthetic self-reported step confidence around the true step quality (INFINITY = uncorrelated with quality)
    pub sig_c: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HSig {
    /// agent's self-reported confidence in its step output
    Conf,
    /// retrieval margin: strength of the retrieved project note (-mis = wrong note, 0 = none)
    Retr,
    /// orchestrator's posterior mean quality of (agent, step type) from delayed graded outcomes (the Router trait exposes no posterior, so this is the orchestrator's own table)
    Post,
}
impl HSig {
    pub const ALL: [HSig; 3] = [HSig::Conf, HSig::Retr, HSig::Post];
    pub fn idx(self) -> usize {
        match self {
            HSig::Conf => 0,
            HSig::Retr => 1,
            HSig::Post => 2,
        }
    }
    pub fn name(self) -> &'static str {
        ["conf", "retr", "post"][self.idx()]
    }
}
#[derive(Clone, Copy, Debug)]
pub enum HGate {
    Always,
    Never,
    /// verify iff signal <= t
    Thr(HSig, f64),
    /// verify w.p. p (CRN-hashed per attempt)
    Rand(f64),
    /// verify iff the step is truly bad (q < THETA): perfect detector
    Oracle,
}

struct Ev {
    due: u64,
    upd: Vec<(Task, usize, f64)>,
    notes: Vec<((usize, usize), f64)>,
    post: Vec<((usize, usize), f64)>,
}
struct Imm {
    task: Task,
    agent: usize,
    v: f64,
    flagged: bool,
    note: (usize, usize),
}
fn q_in(w: &c::WorldCfg, tk: &c::TaskI, si: usize, qs: &[f64]) -> f64 {
    let deps = &tk.steps[si].deps;
    if deps.is_empty() {
        return 1.0;
    }
    match w.family {
        c::Gen::Pipe => deps.iter().map(|&d| qs[d]).sum::<f64>() / deps.len() as f64,
        c::Gen::Latent => deps.iter().map(|&d| qs[d]).fold(1.0, f64::min),
    }
}

/// per-attempt record (signals + truth) for calibration / signal quality: (conf, retr, post, bad)
pub type Att = (f64, f64, f64, bool);

pub struct HOut {
    pub q: Vec<f32>,
    pub ok: Vec<u8>,
    pub cost: Vec<f32>,
    pub calls: Vec<u32>,
    pub vcalls: Vec<u32>,
    pub att: Vec<Att>,
    pub nsteps: Vec<u8>,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct HM {
    pub q: f64,
    pub ok: f64,
    pub cost: f64,
    pub calls: f64,
    pub vcalls: f64,
    pub util_c: f64,
    pub util_a: f64,
}
pub fn summ_h(o: &HOut) -> HM {
    let n = o.q.len() as f64;
    let util = |mu: f64| (0..o.q.len()).map(|i| o.q[i] as f64 - mu * (o.cost[i] as f64 / o.nsteps[i] as f64 - 1.0)).sum::<f64>() / n;
    HM {
        q: o.q.iter().map(|&x| x as f64).sum::<f64>() / n,
        ok: o.ok.iter().map(|&x| x as f64).sum::<f64>() / n,
        cost: o.cost.iter().map(|&x| x as f64).sum::<f64>() / n,
        calls: o.calls.iter().map(|&x| x as f64).sum::<f64>() / n,
        vcalls: o.vcalls.iter().map(|&x| x as f64).sum::<f64>() / n,
        util_c: util(c::MU_C),
        util_a: util(c::MU_A),
    }
}
pub fn summ_g18(r: &c::RunOut, inst: &c::Instance) -> HM {
    let n = r.q.len() as f64;
    let util = |mu: f64| (0..r.q.len()).map(|i| r.q[i] as f64 - mu * (r.cost[i] as f64 / inst.tasks[i].steps.len() as f64 - 1.0)).sum::<f64>() / n;
    HM {
        q: r.q.iter().map(|&x| x as f64).sum::<f64>() / n,
        ok: r.ok.iter().map(|&x| x as f64).sum::<f64>() / n,
        cost: r.cost.iter().map(|&x| x as f64).sum::<f64>() / n,
        calls: r.calls.iter().map(|&x| x as f64).sum::<f64>() / n,
        vcalls: r.vcalls.iter().map(|&x| x as f64).sum::<f64>() / n,
        util_c: util(c::MU_C),
        util_a: util(c::MU_A),
    }
}

/// The g18 orchestrator at level 3 (decompose + route + retrieval + verification with retry/reroute, verification used as same-task router feedback) with an EXTERNAL
/// verification gate. HGate::Always reproduces g18's L3 bit-for-bit; HGate::Never reproduces L2 (checked in the unit tests).
pub fn run_h(hw: &HWorld, inst: &c::Instance, gate: &HGate, seed: u64, collect: bool) -> HOut {
    let w = &hw.cfg;
    let na = inst.n_agents;
    let all: Vec<usize> = (0..na).collect();
    let mut rng = Rng::new(seed * 7 + 1);
    let mut router: Option<Box<dyn Router>> = Some(Box::new(Warm::new(HebbianRouter::new(na, heb_robust_cfg()), inst.kw.clone(), na, 1.0).with_evidence(Evidence::MeanFeature)));
    let mut notes = c::Notes::default();
    let mut pend: VecDeque<Ev> = VecDeque::new();
    let mut post: HashMap<(usize, usize), (f64, f64)> = HashMap::new();
    let mut out = HOut { q: vec![], ok: vec![], cost: vec![], calls: vec![], vcalls: vec![], att: vec![], nsteps: vec![] };
    for (ti, tk) in inst.tasks.iter().enumerate() {
        let tick = ti as u64 + 1;
        while pend.front().is_some_and(|e| e.due <= tick) {
            let ev = pend.pop_front().unwrap();
            if let Some(r) = router.as_deref_mut() {
                for (t, a, g_) in &ev.upd {
                    r.update(t, *a, *g_);
                }
            }
            for (k, g_) in &ev.notes {
                notes.store(*k, *g_, c::STORE_MIN);
            }
            for (k, g_) in &ev.post {
                let e = post.entry(*k).or_insert((0.0, 0.0));
                e.0 += g_;
                e.1 += 1.0;
            }
        }
        let n = tk.steps.len();
        let mut qs = vec![0.0; n];
        let (mut cost, mut calls, mut vcalls) = (0.0f64, 0u32, 0u32);
        let mut fb: Vec<(Task, usize, f64)> = vec![];
        let mut note_fb: Vec<((usize, usize), f64)> = vec![];
        let mut post_fb: Vec<((usize, usize), f64)> = vec![];
        let mut deferred: Vec<Imm> = vec![];
        for si in 0..n {
            let st = &tk.steps[si];
            let qin = q_in(w, tk, si, &qs);
            let stask = Task { id: (ti * 8 + si) as u64, task_type: Some(c::TYPES[st.ty].to_string()), features: vec![c::TYPES[st.ty].to_string()] };
            cost += c::CR;
            let neff = notes.retrieve(w, seed, ti, si, tk.project, st.ty);
            let mut tried: Vec<usize> = vec![];
            let mut best: (f64, f64, usize) = (-1.0, 0.0, 0);
            let mut log: Vec<(usize, f64)> = vec![];
            let chosen: (usize, f64);
            loop {
                let cands: Vec<usize> = all.iter().copied().filter(|a| !tried.contains(a)).collect();
                let a = router.as_mut().unwrap().select(&stask, &cands, &mut rng);
                tried.push(a);
                let q = c::step_q(w, inst, seed, ti, si, a, qin, neff, false);
                cost += 1.0;
                calls += 1;
                log.push((a, q));
                let key = (ti * 8 + si) as u64;
                let conf = if hw.sig_c.is_finite() { (q + hw.sig_c * c::norm(seed, 0x91, key, a as u64)).clamp(0.0, 1.0) } else { (0.7 + 0.15 * c::norm(seed, 0x91, key, a as u64)).clamp(0.0, 1.0) };
                let pm = {
                    let (s, k) = post.get(&(a, st.ty)).copied().unwrap_or((0.0, 0.0));
                    (s + 2.0 * 0.64) / (k + 2.0)
                };
                let bad = q < c::THETA;
                if collect {
                    out.att.push((conf, neff, pm, bad));
                }
                let do_ver = match gate {
                    HGate::Always => true,
                    HGate::Never => false,
                    HGate::Oracle => bad,
                    HGate::Rand(p) => c::uni(seed, 0x74, key, tried.len() as u64) < *p,
                    HGate::Thr(sig, t) => {
                        let s = match sig {
                            HSig::Conf => conf,
                            HSig::Retr => neff,
                            HSig::Post => pm,
                        };
                        s <= *t
                    }
                };
                if !do_ver {
                    chosen = (a, q);
                    break;
                }
                vcalls += 1;
                cost += w.cv;
                let v = (q + w.sig_v * c::norm(seed, 0x7E, key, a as u64)).clamp(0.0, 1.0);
                let flagged = v < c::THETA_V;
                deferred.push(Imm { task: stask.clone(), agent: a, v, flagged, note: (tk.project, st.ty) });
                if !flagged {
                    chosen = (a, q);
                    break;
                }
                if v > best.0 {
                    best = (v, q, a);
                }
                if tried.len() > c::MAX_RETRY || tried.len() >= na {
                    chosen = (best.2, best.1);
                    break;
                }
            }
            qs[si] = chosen.1;
            for &(a, q) in &log {
                let g_ = (q + c::GRADE_SD * c::norm(seed, 0x6D, (ti * 8 + si) as u64, a as u64)).clamp(0.0, 1.0);
                fb.push((stask.clone(), a, g_));
                post_fb.push(((a, st.ty), g_));
            }
            let gq = (chosen.1 + c::GRADE_SD * c::norm(seed, 0x6D, (ti * 8 + si) as u64, chosen.0 as u64)).clamp(0.0, 1.0);
            note_fb.push(((tk.project, st.ty), gq));
            // sequential (non-parallel) flush, as g18 L3
            for im in deferred.drain(..) {
                if let Some(r) = router.as_mut() {
                    r.update(&im.task, im.agent, im.v);
                }
                if !im.flagged {
                    notes.store(im.note, im.v, c::STORE_MIN);
                }
                let _ = im.flagged;
            }
        }
        out.q.push((qs.iter().sum::<f64>() / n.max(1) as f64) as f32);
        out.ok.push(u8::from(qs.iter().all(|&q| q >= c::THETA)));
        out.cost.push(cost as f32);
        out.calls.push(calls);
        out.vcalls.push(vcalls);
        out.nsteps.push(n as u8);
        if c::uni(seed, 0x2E, ti as u64, 0) < c::REVEAL {
            pend.push_back(Ev { due: tick + c::DELAY, upd: fb, notes: note_fb, post: post_fb });
        }
    }
    out
}

/// signal of a given kind from an attempt record
pub fn att_sig(a: &Att, s: HSig) -> f64 {
    match s {
        HSig::Conf => a.0,
        HSig::Retr => a.1,
        HSig::Post => a.2,
    }
}

#[derive(Clone, Debug)]
pub struct HCal {
    /// per signal (conf, retr, post): R1 threshold for util at mu=.03 (headline), mu=.1, and R2 (capture .9 of bad attempts)
    pub r1c: [f64; 3],
    pub r1a: [f64; 3],
    pub r2: [f64; 3],
    /// calibration-split attempt statistics per signal: auroc (low signal -> bad attempt) and bad rate
    pub auroc: [f64; 3],
    pub bad_rate: f64,
}
