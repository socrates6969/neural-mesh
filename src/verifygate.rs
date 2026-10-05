//! Verification gating primitives, std-only, OPT-IN (nothing here is wired into `research.rs` defaults).
//!
//! Productionised from three verified studies (all SYNTHETIC-until-validated-on-real-traffic):
//!  * h3 (results/h3_gate_board_entry.md, `src/bin/h3_common/mod.rs`): [`FixedGate`] - a verifier runs only when a confidence score is `<=` a
//!    threshold fitted OFFLINE on a calibration split (rule R1 break-even / R2 error capture) and never adapted online; guarded by the
//!    calibration AUROC (>= 0.6) and a minimum number of wrong items (>= 30) with fallback to always-verify; [`ShiftMonitor`] flags when the
//!    observed error rate leaves a band around the calibration error rate (the gate is fragile when test error > calibrated).
//!  * h11 (results/h11_proposed_patch.md section 4, `VMode::Gate` in `src/bin/h11_common/pworld.rs`): [`PosteriorGate`] - verify iff
//!    `(1 - m) * rv >= cv` (m = router posterior mean quality of the chosen (agent, step type), cv = verifier price relative to a generator step).
//!  * g9 (results/g9_proposed_patch.md P1, `src/bin/g9_common/mod.rs`): the EVIDENCE-aware verifier core - per-source agreement /
//!    entity-matched / newest-evidence claim checks ([`claim_verdict`], [`task_verdict`]) plus the online precision gate that decides whether a
//!    flag is worth a reroute ([`PrecisionGate`], [`EvidenceCritic`]). Only worth enabling where retrieval surfaces the CURRENT document
//!    >= ~65% of the time and the first-draft error rate is >= ~5%.
//!
//! Float expressions are copied operation-for-operation from the reference code (parity-tested by `src/bin/h16_gate.rs`).
//! Documented divergences from the references: (1) a NaN confidence / NaN posterior means "unknown" and FAILS SAFE to verify (the
//! references compare with `<=`/`>=`, which silently skips); (2) thresholds use finite sentinels `NEVER_VERIFY` / `ALWAYS_VERIFY` (= the
//! reference NEG_INF / POS_INF = -+1e9, so thresholds serialise to JSON); (3) the AUROC guard + calibration report are new (h3_proposed_patch P3).

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// "never verify" sentinel threshold (reference `NEG_INF`)
pub const NEVER_VERIFY: f64 = -1.0e9;
/// "always verify" sentinel threshold (reference `POS_INF`)
pub const ALWAYS_VERIFY: f64 = 1.0e9;
/// price of one evidence-tool run in agent calls (g9 `C_TOOL`)
pub const C_TOOL: f64 = 0.1;

// =============================================================================================
// calibration statistics

/// AUROC of "LOW score predicts positive" (positive = the draft is wrong). 0.5 = uninformative, 1 = perfect, <0.5 = anti-informative.
/// NaN scores are ignored; NaN is returned when one class is missing.
pub fn auroc_low_is_pos(s: &[f64], pos: &[bool]) -> f64 {
    let keep: Vec<usize> = (0..s.len().min(pos.len())).filter(|&i| !s[i].is_nan()).collect();
    let n = keep.len();
    let np = keep.iter().filter(|&&i| pos[i]).count();
    let nn = n - np;
    if np == 0 || nn == 0 {
        return f64::NAN;
    }
    let mut idx: Vec<usize> = keep.clone();
    idx.sort_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; s.len()];
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
    let sum_neg: f64 = keep.iter().filter(|&&k| !pos[k]).map(|&k| ranks[k]).sum();
    (sum_neg - nn as f64 * (nn as f64 + 1.0) / 2.0) / (np as f64 * nn as f64)
}

/// ECE with equal-width bins; `p` = predicted probability of the event, `y` = event happened.
pub fn ece(p: &[f64], y: &[bool], nb: usize) -> f64 {
    let n = p.len().min(y.len());
    if n == 0 || nb == 0 {
        return f64::NAN;
    }
    let mut cnt = vec![0.0; nb];
    let mut sp = vec![0.0; nb];
    let mut sy = vec![0.0; nb];
    for (pi, yi) in p.iter().zip(y).take(n) {
        let b = ((pi.clamp(0.0, 1.0) * nb as f64) as usize).min(nb - 1);
        cnt[b] += 1.0;
        sp[b] += pi;
        sy[b] += f64::from(*yi);
    }
    (0..nb).map(|b| if cnt[b] > 0.0 { cnt[b] / n as f64 * (sp[b] / cnt[b] - sy[b] / cnt[b]).abs() } else { 0.0 }).sum()
}

/// equal-mass bin recalibration map fitted on (score, event) pairs: (upper score bound, event rate) per bin
pub fn fit_bins(s: &[f64], y: &[bool], nb: usize) -> Vec<(f64, f64)> {
    let n = s.len().min(y.len());
    if n == 0 || nb == 0 {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| s[a].partial_cmp(&s[b]).unwrap_or(std::cmp::Ordering::Equal));
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
// FixedGate (h3)

/// Verify iff `score <= threshold`. The threshold is data (fitted offline on a calibration split with ground-truth outcomes), never updated online.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FixedGate {
    pub threshold: f64,
}

impl FixedGate {
    pub const NEVER: FixedGate = FixedGate { threshold: NEVER_VERIFY };
    pub const ALWAYS: FixedGate = FixedGate { threshold: ALWAYS_VERIFY };

    /// A NaN score (or a NaN threshold) means "unknown" and verifies (fail-safe).
    pub fn should_verify(&self, score: f64) -> bool {
        if score.is_nan() || self.threshold.is_nan() {
            return true;
        }
        score <= self.threshold
    }

    /// R1 (cost-neutral break-even): `items` = (score, net benefit of verifying) on the calibration split. The threshold maximises the cumulative
    /// net benefit of the verified prefix (evaluated at score-tie boundaries only); `NEVER` when no prefix has a positive net benefit, `ALWAYS`
    /// when the whole pool is the best prefix, else the midpoint between the last verified and first unverified calibration score.
    pub fn fit_break_even(items: &[(f64, f64)]) -> FixedGate {
        let mut v: Vec<(f64, f64)> = items.iter().copied().filter(|x| !x.0.is_nan() && !x.1.is_nan()).collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
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
        let threshold = if bk == 0 {
            NEVER_VERIFY
        } else if bk == n {
            ALWAYS_VERIFY
        } else {
            0.5 * (v[bk - 1].0 + v[bk].0)
        };
        FixedGate { threshold }
    }

    /// R2 (target error capture): the `cap` quantile of the scores of WRONG items; `NEVER` when fewer than 30 wrong items were observed.
    pub fn fit_capture(wrong_scores: &[f64], cap: f64) -> FixedGate {
        let mut v: Vec<f64> = wrong_scores.iter().copied().filter(|x| !x.is_nan()).collect();
        if v.len() < 30 {
            return FixedGate::NEVER;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k = ((cap.clamp(0.0, 1.0) * v.len() as f64).ceil() as usize).clamp(1, v.len());
        FixedGate { threshold: v[k - 1] }
    }
}

/// threshold rule for [`calibrate_gate`]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum GateRule {
    /// R1: needs the per-item net benefit of verifying
    BreakEven,
    /// R2: capture this fraction of the wrong items' scores
    Capture(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum FallbackReason {
    /// inputs of different lengths / empty
    BadInput,
    /// fewer wrong items on the calibration split than `min_wrong`
    TooFewWrong,
    /// AUROC undefined (a class is missing) or below `min_auroc`: the score does not separate wrong from right
    WeakSignal,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub n: usize,
    pub n_wrong: usize,
    pub error_rate: f64,
    pub auroc: f64,
    /// ECE of `score` read as P(right) on the calibration split
    pub ece_raw: f64,
    /// ECE after an equal-mass 10-bin recalibration map fitted on the SAME split (in-sample)
    pub ece_recal_in_sample: f64,
    /// `None` = the fitted gate is enabled; `Some(r)` = guard tripped and `gate` is the always-verify fallback
    pub fallback: Option<FallbackReason>,
}

pub const DEFAULT_MIN_AUROC: f64 = 0.6;
pub const DEFAULT_MIN_WRONG: usize = 30;

/// Fit a [`FixedGate`] on a calibration split. `scores` = confidence (high = "I am right"), `wrong` = ground truth, `net_benefit` = per-item net
/// benefit of verifying (only used by [`GateRule::BreakEven`]; may be empty for `Capture`). Refuses to enable the gate (returns
/// [`FixedGate::ALWAYS`] = current always-verify behaviour, with `report.fallback` set) when the calibration AUROC is `< min_auroc` or there are
/// `< min_wrong` wrong items.
pub fn calibrate_gate(scores: &[f64], wrong: &[bool], net_benefit: &[f64], rule: GateRule, min_auroc: f64, min_wrong: usize) -> (FixedGate, CalibrationReport) {
    let n = scores.len();
    let mut rep = CalibrationReport { n, n_wrong: 0, error_rate: f64::NAN, auroc: f64::NAN, ece_raw: f64::NAN, ece_recal_in_sample: f64::NAN, fallback: None };
    let need_ben = matches!(rule, GateRule::BreakEven);
    if n == 0 || wrong.len() != n || (need_ben && net_benefit.len() != n) {
        rep.fallback = Some(FallbackReason::BadInput);
        return (FixedGate::ALWAYS, rep);
    }
    rep.n_wrong = wrong.iter().filter(|w| **w).count();
    rep.error_rate = rep.n_wrong as f64 / n as f64;
    rep.auroc = auroc_low_is_pos(scores, wrong);
    let right: Vec<bool> = wrong.iter().map(|w| !w).collect();
    rep.ece_raw = ece(scores, &right, 10);
    let bins = fit_bins(scores, &right, 10);
    let rec: Vec<f64> = scores.iter().map(|&s| apply_bins(&bins, s)).collect();
    rep.ece_recal_in_sample = ece(&rec, &right, 10);
    if rep.n_wrong < min_wrong {
        rep.fallback = Some(FallbackReason::TooFewWrong);
        return (FixedGate::ALWAYS, rep);
    }
    if !(rep.auroc >= min_auroc) {
        rep.fallback = Some(FallbackReason::WeakSignal);
        return (FixedGate::ALWAYS, rep);
    }
    let gate = match rule {
        GateRule::BreakEven => {
            let items: Vec<(f64, f64)> = scores.iter().copied().zip(net_benefit.iter().copied()).collect();
            FixedGate::fit_break_even(&items)
        }
        GateRule::Capture(cap) => {
            let ws: Vec<f64> = scores.iter().zip(wrong).filter(|(_, w)| **w).map(|(s, _)| *s).collect();
            FixedGate::fit_capture(&ws, cap)
        }
    };
    (gate, rep)
}

/// Flags when the observed error rate over a monitoring window leaves a pre-declared band around the calibration error rate (the fixed gate is
/// fragile when the test error rate exceeds the calibrated one): re-calibrate OFFLINE when `observe` returns `Some(true)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShiftMonitor {
    pub cal_err: f64,
    pub band: f64,
    pub window: usize,
    n: usize,
    wrong: usize,
}
impl ShiftMonitor {
    pub fn new(cal_err: f64, band: f64, window: usize) -> ShiftMonitor {
        ShiftMonitor { cal_err, band, window: window.max(1), n: 0, wrong: 0 }
    }
    /// `Some(out_of_band)` when a window closes (the window then resets), else `None`
    pub fn observe(&mut self, wrong: bool) -> Option<bool> {
        self.n += 1;
        self.wrong += wrong as usize;
        if self.n < self.window {
            return None;
        }
        let rate = self.wrong as f64 / self.n as f64;
        self.n = 0;
        self.wrong = 0;
        Some((rate - self.cal_err).abs() > self.band)
    }
}

// =============================================================================================
// PosteriorGate (h11)

/// Verify iff `(1 - m) * rv >= cv` or fewer than `nmin` graded samples back the posterior mean `m`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PosteriorGate {
    /// value of catching an error relative to a reference generator step (h11 tuned once on seeds 1-8: 1.0)
    pub rv: f64,
    /// verifier price relative to a generator step
    pub cv: f64,
    /// minimum graded samples before the posterior is trusted (h11: 3)
    pub nmin: f64,
}
impl PosteriorGate {
    /// `m` = posterior mean quality of (agent, step type), `n` = number of graded samples behind it
    pub fn should_verify(&self, m: f64, n: f64) -> bool {
        if m.is_nan() || n.is_nan() {
            return true;
        }
        n < self.nmin || (1.0 - m) * self.rv >= self.cv
    }
    /// same from running sums: `s` = sum of graded quality, `n` = count (reference expression `(1 - s/n) * rv >= cv`)
    pub fn should_verify_sums(&self, s: f64, n: f64) -> bool {
        if n < self.nmin || n <= 0.0 {
            return true;
        }
        (1.0 - s / n) * self.rv >= self.cv
    }
    /// the equivalent quality threshold `tau = 1 - cv/rv`: verify iff `m <= tau` (once `n >= nmin`)
    pub fn tau(&self) -> f64 {
        1.0 - self.cv / self.rv
    }
}

// =============================================================================================
// evidence-aware verifier (g9)

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimOp {
    /// the claim is the value of (entity, attr)
    Look,
    /// claim = value(entity, attr) + value(entity, attr2)
    Sum,
    /// claim = |value(entity, attr) - value(entity, attr2)|
    Diff,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    pub entity: usize,
    pub attr: usize,
    pub attr2: usize,
    pub op: ClaimOp,
}
/// one retrieved evidence snippet
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snippet {
    pub entity: usize,
    pub attr: usize,
    pub val: i64,
    pub date: i32,
    pub src: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceRule {
    /// claimed value appears in ANY snippet (no entity check; derived values never appear)
    SupportAny,
    /// entity+attribute match, recency-blind; derived = recompute from the first matching snippets
    SupportEntity,
    /// entity+attribute match, newest snippets win; derived = recompute
    Latest,
    /// per-source newest value, the claim must be backed by >= theta of the sources that have evidence; derived = recompute from per-source votes (g9 pick)
    Agree,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Ok,
    Bad,
    Unverifiable,
}

fn combine(op: ClaimOp, x: i64, y: i64) -> i64 {
    match op {
        ClaimOp::Look => x,
        ClaimOp::Sum => x.wrapping_add(y),
        ClaimOp::Diff => x.wrapping_sub(y).wrapping_abs(),
    }
}

fn matching<'a>(snips: &'a [Snippet], e: usize, a: usize) -> Vec<&'a Snippet> {
    snips.iter().filter(|s| s.entity == e && s.attr == a).collect()
}

/// per-source newest value of (e, a): (src, value, date) sorted by (src, value, date)
fn per_source(snips: &[Snippet], e: usize, a: usize) -> Vec<(u8, i64, i32)> {
    let mut best: Vec<(u8, (i64, i32))> = Vec::new();
    for s in matching(snips, e, a) {
        match best.iter_mut().find(|b| b.0 == s.src) {
            Some(ent) => {
                if s.date > (ent.1).1 {
                    ent.1 = (s.val, s.date);
                }
            }
            None => best.push((s.src, (s.val, s.date))),
        }
    }
    let mut v: Vec<(u8, i64, i32)> = best.into_iter().map(|(k, (val, d))| (k, val, d)).collect();
    v.sort();
    v
}

/// the value a verifier takes for (e, a) under `rule` (None = no evidence)
fn resolve(rule: EvidenceRule, snips: &[Snippet], e: usize, a: usize) -> Option<i64> {
    let m = matching(snips, e, a);
    if m.is_empty() {
        return None;
    }
    match rule {
        EvidenceRule::SupportAny | EvidenceRule::SupportEntity => Some(m[0].val),
        EvidenceRule::Latest => {
            let d = m.iter().map(|s| s.date).max()?;
            m.iter().find(|s| s.date == d).map(|s| s.val)
        }
        EvidenceRule::Agree => {
            let ps = per_source(snips, e, a);
            // plurality of per-source newest values; tie -> the one with the newest date
            let mut cnt: Vec<(i64, usize, i32)> = vec![];
            for (_, v, d) in &ps {
                if let Some(c) = cnt.iter_mut().find(|c| c.0 == *v) {
                    c.1 += 1;
                    c.2 = c.2.max(*d);
                } else {
                    cnt.push((*v, 1, *d));
                }
            }
            cnt.sort_by(|x, y| y.1.cmp(&x.1).then(y.2.cmp(&x.2)));
            Some(cnt[0].0)
        }
    }
}

/// verdict of one claim against the retrieved evidence
pub fn claim_verdict(rule: EvidenceRule, theta: f64, claim: &Claim, snips: &[Snippet], claimed: i64) -> Verdict {
    match rule {
        EvidenceRule::SupportAny => {
            if snips.iter().any(|s| s.val == claimed) {
                Verdict::Ok
            } else {
                Verdict::Bad
            }
        }
        _ => {
            if claim.op == ClaimOp::Look {
                let m = matching(snips, claim.entity, claim.attr);
                if m.is_empty() {
                    return Verdict::Unverifiable;
                }
                match rule {
                    EvidenceRule::SupportEntity => {
                        if m.iter().any(|s| s.val == claimed) {
                            Verdict::Ok
                        } else {
                            Verdict::Bad
                        }
                    }
                    EvidenceRule::Latest => {
                        let d = m.iter().map(|s| s.date).max().unwrap_or(i32::MIN);
                        if m.iter().any(|s| s.date == d && s.val == claimed) {
                            Verdict::Ok
                        } else {
                            Verdict::Bad
                        }
                    }
                    _ => {
                        let ps = per_source(snips, claim.entity, claim.attr);
                        let agree = ps.iter().filter(|p| p.1 == claimed).count() as f64 / ps.len() as f64;
                        if agree >= theta - 1e-9 {
                            Verdict::Ok
                        } else {
                            Verdict::Bad
                        }
                    }
                }
            } else {
                match (resolve(rule, snips, claim.entity, claim.attr), resolve(rule, snips, claim.entity, claim.attr2)) {
                    (Some(x), Some(y)) => {
                        if combine(claim.op, x, y) == claimed {
                            Verdict::Ok
                        } else {
                            Verdict::Bad
                        }
                    }
                    _ => Verdict::Unverifiable,
                }
            }
        }
    }
}

/// task level: how many claims are Bad / Unverifiable (`claimed[i]` is the draft's value for `claims[i]`)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Assessment {
    pub bad: usize,
    pub unverifiable: usize,
}
impl Assessment {
    /// flagged = any Bad claim (or any Unverifiable claim when `flag_unverifiable`; g9: false is better in every tune world)
    pub fn flagged(&self, flag_unverifiable: bool) -> bool {
        self.bad > 0 || (flag_unverifiable && self.unverifiable > 0)
    }
}
pub fn task_verdict(rule: EvidenceRule, theta: f64, claims: &[Claim], snips: &[Snippet], claimed: &[i64]) -> Assessment {
    let mut a = Assessment::default();
    for (ci, c) in claims.iter().enumerate() {
        let Some(&v) = claimed.get(ci) else {
            a.unverifiable += 1;
            continue;
        };
        match claim_verdict(rule, theta, c, snips, v) {
            Verdict::Bad => a.bad += 1,
            Verdict::Unverifiable => a.unverifiable += 1,
            Verdict::Ok => {}
        }
    }
    a
}

// ---------------------------------------------------------------------------------------------
// online precision gate (g9 "auto")

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceCriticConfig {
    /// `Agree` agreement threshold (g9 tuned pick: 0.5)
    pub theta: f64,
    /// price of one other-agent call in success units, used for the break-even only (g9: 0.02..0.05; the h3 re-implementation used 0.02)
    pub call_cost: f64,
    /// price of one evidence-tool run in agent calls
    pub tool_cost: f64,
    /// treat Unverifiable as a flag (g9: false)
    pub flag_unverifiable: bool,
    /// pseudo-count of the prior "reroute error = first-draft error + `retry_prior_margin`"
    pub retry_prior_n: f64,
    pub retry_prior_margin: f64,
}
impl Default for EvidenceCriticConfig {
    fn default() -> Self {
        EvidenceCriticConfig { theta: 0.5, call_cost: 0.02, tool_cost: C_TOOL, flag_unverifiable: false, retry_prior_n: 8.0, retry_prior_margin: 0.05 }
    }
}

/// Online estimates from DELAYED graded outcomes: precision of flags, first-draft error, reroute error. Call `flush(tick)` before each task.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrecisionGate {
    fl_w: f64,
    fl_n: f64,
    wr: f64,
    nn: f64,
    ret_w: f64,
    ret_n: f64,
    q: VecDeque<(u64, bool, bool)>,
    rq: VecDeque<(u64, f64)>,
}
impl Default for PrecisionGate {
    fn default() -> Self {
        PrecisionGate::new()
    }
}
impl PrecisionGate {
    pub fn new() -> PrecisionGate {
        PrecisionGate { fl_w: 0.0, fl_n: 0.0, wr: 0.0, nn: 0.0, ret_w: 0.0, ret_n: 0.0, q: VecDeque::new(), rq: VecDeque::new() }
    }
    /// queue the graded outcome of a first draft, due at `due` (delayed feedback): was it flagged, was it wrong
    pub fn enqueue_first(&mut self, due: u64, flagged: bool, wrong: bool) {
        self.q.push_back((due, flagged, wrong));
    }
    /// queue the graded accuracy of a delivered reroute, due at `due`
    pub fn enqueue_retry(&mut self, due: u64, acc: f64) {
        self.rq.push_back((due, acc));
    }
    /// apply every queued outcome with `due <= tick`
    pub fn flush(&mut self, tick: u64) {
        while self.rq.front().is_some_and(|e| e.0 <= tick) {
            let (_, acc) = self.rq.pop_front().unwrap_or((0, 1.0));
            self.ret_n += 1.0;
            self.ret_w += 1.0 - acc;
        }
        while self.q.front().is_some_and(|e| e.0 <= tick) {
            let (_, flagged, wrong) = self.q.pop_front().unwrap_or((0, false, false));
            self.nn += 1.0;
            if wrong {
                self.wr += 1.0;
            }
            if flagged {
                self.fl_n += 1.0;
                if wrong {
                    self.fl_w += 1.0;
                }
            }
        }
    }
    /// number of graded first drafts applied so far
    pub fn n_graded(&self) -> f64 {
        self.nn
    }
    /// estimated P(first draft wrong | flagged), Beta(1,1) prior
    pub fn flag_precision(&self) -> f64 {
        (self.fl_w + 1.0) / (self.fl_n + 2.0)
    }
    pub fn first_error(&self) -> f64 {
        (self.wr + 1.0) / (self.nn + 4.0)
    }
    /// estimated error of a reroute (prior: first-draft error + margin, `retry_prior_n` pseudo-counts)
    pub fn retry_error(&self, cfg: &EvidenceCriticConfig) -> f64 {
        (self.ret_w + cfg.retry_prior_n * (self.first_error() + cfg.retry_prior_margin)) / (self.ret_n + cfg.retry_prior_n)
    }
    pub fn break_even(&self, cfg: &EvidenceCriticConfig) -> f64 {
        self.retry_error(cfg) + cfg.call_cost * (1.0 + 2.0 * cfg.tool_cost)
    }
    /// act on a flag only while the online precision estimate beats the break-even
    pub fn should_act(&self, cfg: &EvidenceCriticConfig) -> bool {
        self.flag_precision() > self.break_even(cfg)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Action {
    /// not flagged: deliver the draft
    Accept,
    /// flagged and the gate says a reroute pays: one retry by the best OTHER agent, then re-verify it
    Reroute,
    /// flagged but the precision gate says a reroute does not pay: deliver the draft marked UNVERIFIED
    DeliverUnverified,
}

/// what to deliver after a reroute (g9: deliver the retry if it passes; if it also fails, the one with fewer Bad claims, marked unverified)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RerouteOutcome {
    pub deliver_second: bool,
    pub unverified: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceCritic {
    pub cfg: EvidenceCriticConfig,
    pub rule: EvidenceRule,
    pub gate: PrecisionGate,
}
impl EvidenceCritic {
    pub fn new(cfg: EvidenceCriticConfig) -> EvidenceCritic {
        EvidenceCritic { cfg, rule: EvidenceRule::Agree, gate: PrecisionGate::new() }
    }
    pub fn assess(&self, claims: &[Claim], snips: &[Snippet], claimed: &[i64]) -> Assessment {
        task_verdict(self.rule, self.cfg.theta, claims, snips, claimed)
    }
    pub fn decide(&self, a: &Assessment) -> Action {
        if !a.flagged(self.cfg.flag_unverifiable) {
            Action::Accept
        } else if self.gate.should_act(&self.cfg) {
            Action::Reroute
        } else {
            Action::DeliverUnverified
        }
    }
    pub fn after_reroute(&self, first: &Assessment, second: &Assessment) -> RerouteOutcome {
        if !second.flagged(self.cfg.flag_unverifiable) {
            RerouteOutcome { deliver_second: true, unverified: false }
        } else {
            RerouteOutcome { deliver_second: second.bad <= first.bad, unverified: true }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snip(e: usize, a: usize, val: i64, date: i32, src: u8) -> Snippet {
        Snippet { entity: e, attr: a, val, date, src }
    }
    fn look(e: usize, a: usize) -> Claim {
        Claim { entity: e, attr: a, attr2: 0, op: ClaimOp::Look }
    }

    #[test]
    fn auroc_known_values() {
        let s = [0.1, 0.2, 0.8, 0.9];
        assert!((auroc_low_is_pos(&s, &[true, true, false, false]) - 1.0).abs() < 1e-12);
        assert!(auroc_low_is_pos(&s, &[false, false, true, true]).abs() < 1e-12);
        assert!((auroc_low_is_pos(&[0.5; 4], &[true, false, true, false]) - 0.5).abs() < 1e-12); // all ties
        assert!(auroc_low_is_pos(&s, &[true; 4]).is_nan());
        assert!(auroc_low_is_pos(&[], &[]).is_nan());
        assert!((auroc_low_is_pos(&[0.1, f64::NAN, 0.9, 0.8], &[true, true, false, false]) - 1.0).abs() < 1e-12); // NaN ignored
    }

    #[test]
    fn ece_and_bins() {
        assert!(ece(&[], &[], 10).is_nan());
        assert!(ece(&[1.0, 1.0], &[true, true], 10).abs() < 1e-12);
        assert!((ece(&[1.0, 1.0], &[false, false], 10) - 1.0).abs() < 1e-12);
        let s: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let y: Vec<bool> = s.iter().map(|&x| x > 0.5).collect();
        let bins = fit_bins(&s, &y, 10);
        assert_eq!(bins.len(), 10);
        assert!(apply_bins(&bins, 0.05) < 0.01 && apply_bins(&bins, 0.95) > 0.99);
        assert_eq!(apply_bins(&[], 0.5), 0.0);
        assert!(fit_bins(&[], &[], 10).is_empty());
    }

    #[test]
    fn fixed_gate_decisions() {
        let g = FixedGate { threshold: 0.5 };
        assert!(g.should_verify(0.5) && g.should_verify(0.1) && !g.should_verify(0.7));
        assert!(g.should_verify(f64::NAN)); // unknown -> verify
        assert!(!FixedGate::NEVER.should_verify(0.0) && FixedGate::ALWAYS.should_verify(1.0));
    }

    #[test]
    fn break_even_fit() {
        // low scores: verifying pays (+1); high scores: costs (-1). Best prefix = the 3 low scores, boundary midpoint between 0.3 and 0.7
        let items = [(0.1, 1.0), (0.2, 1.0), (0.3, 1.0), (0.7, -1.0), (0.8, -1.0)];
        let g = FixedGate::fit_break_even(&items);
        assert!((g.threshold - 0.5).abs() < 1e-12);
        assert_eq!(FixedGate::fit_break_even(&[(0.1, -1.0), (0.2, -0.5)]), FixedGate::NEVER);
        assert_eq!(FixedGate::fit_break_even(&[(0.1, 1.0), (0.2, 0.5)]), FixedGate::ALWAYS);
        assert_eq!(FixedGate::fit_break_even(&[]), FixedGate::NEVER);
        // ties: a prefix may only end at a tie boundary (the 0.2 pair is taken together: -1 + 3 = +2 > 1)
        let g = FixedGate::fit_break_even(&[(0.1, 1.0), (0.2, -1.0), (0.2, 3.0), (0.9, -5.0)]);
        assert!((g.threshold - 0.55).abs() < 1e-12);
        // NaN items are dropped, order of input is irrelevant
        let a = FixedGate::fit_break_even(&[(0.8, -1.0), (f64::NAN, 9.0), (0.1, 1.0), (0.3, 1.0), (0.2, 1.0), (0.7, -1.0)]);
        assert!((a.threshold - 0.5).abs() < 1e-12);
    }

    #[test]
    fn capture_fit() {
        let few: Vec<f64> = (0..29).map(|i| i as f64).collect();
        assert_eq!(FixedGate::fit_capture(&few, 0.9), FixedGate::NEVER);
        let ws: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        assert_eq!(FixedGate::fit_capture(&ws, 0.9).threshold, 90.0);
        assert_eq!(FixedGate::fit_capture(&ws, 1.0).threshold, 100.0);
        assert_eq!(FixedGate::fit_capture(&ws, 0.0).threshold, 1.0);
    }

    /// informative world: wrong items have low confidence
    fn world(n: usize, sep: f64) -> (Vec<f64>, Vec<bool>, Vec<f64>) {
        let mut s = Vec::new();
        let mut w = Vec::new();
        let mut b = Vec::new();
        for i in 0..n {
            let wrong = i % 4 == 0;
            let jitter = ((i * 37 % 100) as f64) / 100.0;
            s.push(if wrong { 0.5 - sep * 0.3 + jitter * 0.4 } else { 0.5 + jitter * 0.4 });
            w.push(wrong);
            b.push(if wrong { 0.8 } else { -0.05 });
        }
        (s, w, b)
    }

    #[test]
    fn calibrate_enables_on_informative_signal() {
        let (s, w, b) = world(400, 1.0);
        let (g, rep) = calibrate_gate(&s, &w, &b, GateRule::BreakEven, DEFAULT_MIN_AUROC, DEFAULT_MIN_WRONG);
        assert_eq!(rep.fallback, None);
        assert!(rep.auroc > 0.6 && rep.n_wrong == 100 && (rep.error_rate - 0.25).abs() < 1e-12);
        assert!(g.threshold > NEVER_VERIFY && g.threshold < ALWAYS_VERIFY);
        let (g2, rep2) = calibrate_gate(&s, &w, &[], GateRule::Capture(0.9), DEFAULT_MIN_AUROC, DEFAULT_MIN_WRONG);
        assert_eq!(rep2.fallback, None);
        assert!(g2.threshold > NEVER_VERIFY);
        assert!(rep.ece_recal_in_sample <= rep.ece_raw + 1e-12);
    }

    #[test]
    fn calibrate_guards_fall_back_to_always_verify() {
        // uninformative signal (AUROC ~ 0.5)
        let (s, w, b) = world(400, 0.0);
        let (g, rep) = calibrate_gate(&s, &w, &b, GateRule::BreakEven, DEFAULT_MIN_AUROC, DEFAULT_MIN_WRONG);
        assert!(rep.auroc < 0.6);
        assert_eq!((g, rep.fallback), (FixedGate::ALWAYS, Some(FallbackReason::WeakSignal)));
        // anti-informative
        let (s2, w2, b2) = world(400, -1.0);
        assert_eq!(calibrate_gate(&s2, &w2, &b2, GateRule::BreakEven, 0.6, 30).1.fallback, Some(FallbackReason::WeakSignal));
        // too few wrong items
        let (s3, w3, b3) = world(100, 1.0); // 25 wrong
        let (g3, r3) = calibrate_gate(&s3, &w3, &b3, GateRule::BreakEven, 0.6, 30);
        assert_eq!((g3, r3.fallback), (FixedGate::ALWAYS, Some(FallbackReason::TooFewWrong)));
        // no wrong items at all -> TooFewWrong (checked before the undefined AUROC)
        let (g4, r4) = calibrate_gate(&[0.9; 50], &[false; 50], &[0.0; 50], GateRule::BreakEven, 0.6, 30);
        assert_eq!((g4, r4.fallback), (FixedGate::ALWAYS, Some(FallbackReason::TooFewWrong)));
        // bad inputs
        assert_eq!(calibrate_gate(&[], &[], &[], GateRule::BreakEven, 0.6, 30).1.fallback, Some(FallbackReason::BadInput));
        assert_eq!(calibrate_gate(&[0.5], &[true, false], &[0.0], GateRule::BreakEven, 0.6, 30).1.fallback, Some(FallbackReason::BadInput));
        assert_eq!(calibrate_gate(&[0.5, 0.4], &[true, false], &[0.0], GateRule::BreakEven, 0.6, 30).1.fallback, Some(FallbackReason::BadInput));
    }

    #[test]
    fn calibrate_is_deterministic_and_serde_roundtrips() {
        let (s, w, b) = world(400, 1.0);
        let a = calibrate_gate(&s, &w, &b, GateRule::BreakEven, 0.6, 30);
        let c = calibrate_gate(&s, &w, &b, GateRule::BreakEven, 0.6, 30);
        assert_eq!(a, c);
        let js = serde_json::to_string(&a.0).unwrap();
        let back: FixedGate = serde_json::from_str(&js).unwrap();
        assert_eq!(back, a.0);
        for sentinel in [FixedGate::NEVER, FixedGate::ALWAYS] {
            let b: FixedGate = serde_json::from_str(&serde_json::to_string(&sentinel).unwrap()).unwrap();
            assert_eq!(b, sentinel);
        }
        let rep: CalibrationReport = serde_json::from_str(&serde_json::to_string(&a.1).unwrap()).unwrap();
        assert_eq!(rep.n, a.1.n);
        assert_eq!(rep.fallback, a.1.fallback);
    }

    #[test]
    fn shift_monitor() {
        let mut m = ShiftMonitor::new(0.10, 0.05, 20);
        let mut out = vec![];
        for i in 0..40 {
            if let Some(r) = m.observe(i % 5 == 0) {
                out.push(r); // 20% error -> out of band
            }
        }
        assert_eq!(out, vec![true, true]);
        let mut m = ShiftMonitor::new(0.10, 0.05, 20);
        let r: Vec<bool> = (0..20).filter_map(|i| m.observe(i % 10 == 0)).collect(); // 10% -> in band
        assert_eq!(r, vec![false]);
        let back: ShiftMonitor = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(back, m);
        assert_eq!(ShiftMonitor::new(0.1, 0.1, 0).window, 1);
    }

    #[test]
    fn posterior_gate_rule() {
        let g = PosteriorGate { rv: 1.0, cv: 0.3, nmin: 3.0 };
        assert!(g.should_verify(0.5, 10.0)); // (1-.5)*1 = .5 >= .3
        assert!(!g.should_verify(0.9, 10.0)); // .1 < .3
        assert!(g.should_verify(0.99, 2.0)); // too few samples -> verify
        assert!(g.should_verify(f64::NAN, 10.0));
        assert!((g.tau() - 0.7).abs() < 1e-12);
        // boundary: verify iff m <= tau (here exactly (1-m)*rv >= cv)
        assert!(g.should_verify(0.7 - 1e-9, 10.0) && !g.should_verify(0.7 + 1e-9, 10.0));
        // an expensive verifier never pays when cv > rv; a free one always does
        assert!(!PosteriorGate { rv: 1.0, cv: 1.5, nmin: 0.0 }.should_verify(0.0, 5.0));
        assert!(PosteriorGate { rv: 1.0, cv: 0.0, nmin: 0.0 }.should_verify(1.0, 5.0));
        // sums form == mean form
        for (s, n) in [(5.0, 10.0), (9.0, 10.0), (2.0, 2.0), (0.0, 0.0)] {
            if n >= 3.0 {
                assert_eq!(g.should_verify_sums(s, n), g.should_verify(s / n, n));
            }
        }
        assert!(g.should_verify_sums(0.0, 0.0));
        let back: PosteriorGate = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(back, g);
    }

    #[test]
    fn evidence_look_rules() {
        let claim = look(1, 2);
        // three sources: A(src 0) newest 10, B(src 1) newest 10, C(src 2) stale 7
        let sn = vec![snip(1, 2, 7, 1, 0), snip(1, 2, 10, 5, 0), snip(1, 2, 10, 4, 1), snip(1, 2, 7, 2, 2), snip(9, 9, 99, 9, 0)];
        // Agree: claim 10 backed by 2/3 sources; claim 7 by 1/3
        assert_eq!(claim_verdict(EvidenceRule::Agree, 0.5, &claim, &sn, 10), Verdict::Ok);
        assert_eq!(claim_verdict(EvidenceRule::Agree, 0.5, &claim, &sn, 7), Verdict::Bad);
        assert_eq!(claim_verdict(EvidenceRule::Agree, 0.3, &claim, &sn, 7), Verdict::Ok);
        assert_eq!(claim_verdict(EvidenceRule::Agree, 1.0, &claim, &sn, 10), Verdict::Bad);
        // SupportEntity: recency-blind -> the stale value 7 is "supported"
        assert_eq!(claim_verdict(EvidenceRule::SupportEntity, 0.5, &claim, &sn, 7), Verdict::Ok);
        // Latest: only the newest-dated snippet counts
        assert_eq!(claim_verdict(EvidenceRule::Latest, 0.5, &claim, &sn, 10), Verdict::Ok);
        assert_eq!(claim_verdict(EvidenceRule::Latest, 0.5, &claim, &sn, 7), Verdict::Bad);
        // SupportAny ignores the entity: a value from ANOTHER entity's snippet passes, no evidence at all -> Bad
        assert_eq!(claim_verdict(EvidenceRule::SupportAny, 0.5, &claim, &sn, 99), Verdict::Ok);
        assert_eq!(claim_verdict(EvidenceRule::SupportAny, 0.5, &claim, &[], 1), Verdict::Bad);
        // no matching evidence for the entity -> Unverifiable (all non-naive rules)
        for r in [EvidenceRule::SupportEntity, EvidenceRule::Latest, EvidenceRule::Agree] {
            assert_eq!(claim_verdict(r, 0.5, &look(3, 3), &sn, 1), Verdict::Unverifiable);
            assert_eq!(claim_verdict(r, 0.5, &claim, &[], 1), Verdict::Unverifiable);
        }
    }

    #[test]
    fn evidence_derived_claims_and_poison() {
        let sn = vec![snip(0, 0, 10, 1, 0), snip(0, 1, 4, 1, 0), snip(0, 0, 10, 1, 1), snip(0, 1, 4, 1, 1)];
        let sum = Claim { entity: 0, attr: 0, attr2: 1, op: ClaimOp::Sum };
        let dif = Claim { entity: 0, attr: 0, attr2: 1, op: ClaimOp::Diff };
        assert_eq!(claim_verdict(EvidenceRule::Agree, 0.5, &sum, &sn, 14), Verdict::Ok);
        assert_eq!(claim_verdict(EvidenceRule::Agree, 0.5, &sum, &sn, 15), Verdict::Bad);
        assert_eq!(claim_verdict(EvidenceRule::Agree, 0.5, &dif, &sn, 6), Verdict::Ok);
        assert_eq!(claim_verdict(EvidenceRule::Latest, 0.5, &dif, &sn[..2], 6), Verdict::Ok);
        // one operand has no evidence -> Unverifiable
        assert_eq!(claim_verdict(EvidenceRule::Agree, 0.5, &sum, &sn[..1], 14), Verdict::Unverifiable);
        // a single poisoned newest snippet from a third source is outvoted by two agreeing sources under Agree, but wins under Latest
        let sn = vec![snip(0, 0, 10, 1, 0), snip(0, 0, 10, 1, 1), snip(0, 0, 55, 9, 2)];
        let c = look(0, 0);
        assert_eq!(claim_verdict(EvidenceRule::Agree, 0.5, &c, &sn, 10), Verdict::Ok);
        assert_eq!(claim_verdict(EvidenceRule::Latest, 0.5, &c, &sn, 10), Verdict::Bad);
        // Agree plurality tie -> newest date wins (resolve used by derived claims)
        let sn = vec![snip(0, 0, 3, 1, 0), snip(0, 0, 8, 5, 1), snip(0, 1, 1, 1, 0)];
        assert_eq!(claim_verdict(EvidenceRule::Agree, 0.5, &Claim { entity: 0, attr: 0, attr2: 1, op: ClaimOp::Sum }, &sn, 9), Verdict::Ok);
    }

    #[test]
    fn task_level_assessment() {
        let claims = vec![look(0, 0), look(1, 1), look(2, 2)];
        let sn = vec![snip(0, 0, 5, 1, 0), snip(1, 1, 6, 1, 0)];
        let a = task_verdict(EvidenceRule::Agree, 0.5, &claims, &sn, &[5, 99, 1]);
        assert_eq!(a, Assessment { bad: 1, unverifiable: 1 });
        assert!(a.flagged(false) && a.flagged(true));
        let a = task_verdict(EvidenceRule::Agree, 0.5, &claims, &sn, &[5, 6, 1]);
        assert_eq!(a, Assessment { bad: 0, unverifiable: 1 });
        assert!(!a.flagged(false) && a.flagged(true));
        // missing claimed value -> unverifiable; no claims -> clean
        assert_eq!(task_verdict(EvidenceRule::Agree, 0.5, &claims, &sn, &[5]).unverifiable, 3 - 1 + 0);
        assert_eq!(task_verdict(EvidenceRule::Agree, 0.5, &[], &sn, &[]), Assessment::default());
    }

    #[test]
    fn precision_gate_prior_and_learning() {
        let cfg = EvidenceCriticConfig::default();
        let mut g = PrecisionGate::new();
        // prior: precision .5, first error .25, retry error .25+.05 = .30, break-even .30 + .02*1.2 = .324 -> acts at the start
        assert!((g.flag_precision() - 0.5).abs() < 1e-12);
        assert!((g.break_even(&cfg) - 0.324).abs() < 1e-12);
        assert!(g.should_act(&cfg));
        // 100 graded flags that were all false alarms -> precision ~ 1/102 -> stop acting
        for i in 0..100 {
            g.enqueue_first(i, true, false);
        }
        assert!(g.should_act(&cfg)); // nothing flushed yet (delayed feedback)
        g.flush(1000);
        assert!(g.n_graded() == 100.0 && !g.should_act(&cfg));
        // flags that are mostly right keep it acting
        // (precision .9 against a ~18% overall first-draft error: reroute pays)
        let mut g = PrecisionGate::new();
        for i in 0..100 {
            g.enqueue_first(i, true, i % 10 != 0);
        }
        for i in 0..400 {
            g.enqueue_first(100 + i, false, false);
        }
        g.flush(1000);
        assert!(g.should_act(&cfg));
        // ... but when almost EVERY draft is wrong the retry is expected to be wrong too: no point rerouting
        let mut g = PrecisionGate::new();
        for i in 0..100 {
            g.enqueue_first(i, true, true);
        }
        g.flush(1000);
        assert!(!g.should_act(&cfg));
    }

    #[test]
    fn precision_gate_delay_and_retry_error() {
        let cfg = EvidenceCriticConfig::default();
        let mut g = PrecisionGate::new();
        g.enqueue_first(10, true, true);
        g.enqueue_retry(10, 0.0);
        g.flush(9);
        assert_eq!(g.n_graded(), 0.0);
        g.flush(10);
        assert_eq!(g.n_graded(), 1.0);
        // a failed retry raises the estimated retry error above the prior
        let prior = PrecisionGate::new().retry_error(&cfg);
        assert!(g.retry_error(&cfg) > prior);
        // serde round-trip of the full state incl. queues
        g.enqueue_first(99, false, true);
        let back: PrecisionGate = serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(back, g);
    }

    #[test]
    fn critic_decisions() {
        let mut c = EvidenceCritic::new(EvidenceCriticConfig::default());
        let claims = vec![look(0, 0)];
        let good = vec![snip(0, 0, 5, 1, 0), snip(0, 0, 5, 1, 1)];
        let a_ok = c.assess(&claims, &good, &[5]);
        let a_bad = c.assess(&claims, &good, &[6]);
        assert_eq!(c.decide(&a_ok), Action::Accept);
        assert_eq!(c.decide(&a_bad), Action::Reroute);
        // unverifiable is not a flag by default
        assert_eq!(c.decide(&c.assess(&claims, &[], &[5])), Action::Accept);
        // after many false alarms the gate stops rerouting and delivers "unverified"
        for i in 0..200 {
            c.gate.enqueue_first(i, true, false);
        }
        c.gate.flush(500);
        assert_eq!(c.decide(&a_bad), Action::DeliverUnverified);
        // reroute outcomes
        let first = Assessment { bad: 2, unverifiable: 0 };
        assert_eq!(c.after_reroute(&first, &Assessment::default()), RerouteOutcome { deliver_second: true, unverified: false });
        assert_eq!(c.after_reroute(&first, &Assessment { bad: 1, unverifiable: 0 }), RerouteOutcome { deliver_second: true, unverified: true });
        assert_eq!(c.after_reroute(&first, &Assessment { bad: 3, unverifiable: 0 }), RerouteOutcome { deliver_second: false, unverified: true });
        let back: EvidenceCritic = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn evidence_determinism_and_order_independence() {
        let claim = look(0, 0);
        let mut sn = vec![snip(0, 0, 1, 1, 0), snip(0, 0, 2, 2, 1), snip(0, 0, 3, 3, 2), snip(0, 0, 2, 2, 0)];
        let a = claim_verdict(EvidenceRule::Agree, 0.5, &claim, &sn, 2);
        sn.reverse();
        assert_eq!(claim_verdict(EvidenceRule::Agree, 0.5, &claim, &sn, 2), a);
        assert_eq!(resolve(EvidenceRule::Agree, &sn, 0, 0), resolve(EvidenceRule::Agree, &sn, 0, 0));
    }
}
