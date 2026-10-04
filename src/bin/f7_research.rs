//! F7-A: recommended ResearchConfig DEFAULT.
//!
//! Re-runs the e3 research-session pipeline with the e1 recommended default router
//! (`guarded(hebr)` "safe": defaults::recommended_guard) vs `warm(hebr)`, with retrieval + reuse and
//! NO unconditional critic, plus CONFIDENCE-TRIGGERED critics (generator self-confidence < theta,
//! or an online-calibrated version that learns from delayed graded outcomes whether low confidence
//! really predicts a bad draft). Worlds: e3's 7 worlds (+ a generator self-confidence signal) and 7 new
//! "cal-*" worlds that sweep the confidence AUROC (rho 0 .. 1) and add a realistic overconfident-
//! hallucination signal, plus a no-error control.
//!
//! The confidence-triggered critic is implemented HERE (src/ is read-only) as a post-step around
//! `ResearchSession::run_task` (session critic = None): after the draft, if the trigger fires, the best
//! other (healthy) agent checks it (one extra call); a flag -> that agent retries (second extra call) and
//! the retry REPLACES the draft (harsh convention, as e3/f4). The session records the FIRST draft's grade for
//! its router/notes (the retry's grade is not fed back: small, disclosed understatement of the critic).
//!
//! Protocol: theta / calibrated threshold / plain warm_k tuned on seeds 1-8 (objective = mean NET at call
//! cost .03 over ALL worlds incl. controls); held-out 31-60; --held 61-90 = confirmation, same frozen
//! tuning. Paired 95% t-CI, Holm per family. No tuning of the e3 knobs (store_min .65, sim_min .55, warm_k 1
//! are e3's tuned values, frozen).
//!
//! Usage: cargo run --release --offline --bin f7_research -- [--held 31-60] [--out results/f7_research.json]

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use neural_mesh::baselines::KeywordRouter;
use neural_mesh::research::*;
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue};
use neural_mesh::variants::{HierSoConfig, SecondOpinionConfig};
use rayon::prelude::*;
use serde_json::{json, Value};

const N_TASKS: usize = 2000;
const N_AGENTS: usize = 8;
const REVEAL: f64 = 0.5;
const DELAY: usize = 20;
const COSTS: [f64; 3] = [0.01, 0.03, 0.10];
const TUNE_COST: f64 = 0.03;

// ---------------------------------------------------------------- stateless randomness
fn hash(seed: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ a.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ b.wrapping_mul(0x94D0_49BB_1331_11EB) ^ c.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn uni(seed: u64, a: u64, b: u64, c: u64) -> f64 {
    (hash(seed, a, b, c) >> 11) as f64 / (1u64 << 53) as f64
}
fn norm(seed: u64, a: u64, b: u64, c: u64) -> f64 {
    let (u1, u2) = (uni(seed, a, b, c).max(1e-12), uni(seed, a, b, c ^ 0xABCD));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

// ---------------------------------------------------------------- world
#[derive(Clone, Copy, Debug, PartialEq)]
enum LazyMode {
    Third,
    All,
    None,
}

#[derive(Clone, Debug)]
struct WorldCfg {
    name: &'static str,
    topics: usize,
    recur: bool,
    equal: bool,
    hal_base: f64,
    hal_floor: f64,
    ctx_gain: f64,
    ctx_hal_cut: f64,
    lazy: LazyMode,
    decoy: f64,
    outage: bool,
    /// informativeness of the generator's self-confidence (conf = .5 + rho (q-.5) + noise)
    rho: f64,
    conf_noise: f64,
    /// hallucinated drafts get this extra confidence (overconfident hallucinations)
    overconf: f64,
}

fn worlds() -> Vec<WorldCfg> {
    let base = WorldCfg {
        name: "halluc-recur", topics: 12, recur: true, equal: false, hal_base: 0.7, hal_floor: 0.0, ctx_gain: 0.4, ctx_hal_cut: 0.6,
        lazy: LazyMode::Third, decoy: 0.3, outage: true, rho: 0.6, conf_noise: 0.2, overconf: 0.0,
    };
    let cal = |name: &'static str, rho: f64| WorldCfg { name, hal_floor: 0.15, rho, ..base.clone() };
    vec![
        // ---- e3's seven worlds (self-confidence rho .6 added; floor 0) ----
        base.clone(),
        WorldCfg { name: "low-error", hal_base: 0.12, ctx_gain: 0.15, ctx_hal_cut: 0.3, ..base.clone() },
        WorldCfg { name: "diluted-embedding", recur: false, ..base.clone() },
        WorldCfg { name: "lazy-checkers", lazy: LazyMode::All, ..base.clone() },
        WorldCfg { name: "recur-only", hal_base: 0.0, ctx_gain: 0.4, ctx_hal_cut: 0.0, lazy: LazyMode::None, ..base.clone() },
        WorldCfg { name: "irrelevant-notes", ctx_gain: 0.0, ctx_hal_cut: 0.0, ..base.clone() },
        WorldCfg { name: "control-equal", equal: true, hal_base: 0.0, ctx_gain: 0.0, ctx_hal_cut: 0.0, lazy: LazyMode::None, decoy: 0.0, outage: false, ..base.clone() },
        // ---- NEW: error floor .15 + confidence-informativeness sweep (AUROC measured, see output) ----
        cal("cal-rho0.00", 0.0),
        cal("cal-rho0.20", 0.2),
        cal("cal-rho0.35", 0.35),
        cal("cal-rho0.60", 0.6),
        cal("cal-rho1.00", 1.0),
        // realistic: moderate informativeness, hallucinations are OVERconfident, noisier self-report
        WorldCfg { name: "cal-realistic", hal_floor: 0.15, rho: 0.6, overconf: 0.25, conf_noise: 0.3, ..base.clone() },
        // no-harm control with an informative confidence signal: almost no errors
        WorldCfg { name: "cal-nohal-control", hal_base: 0.0, hal_floor: 0.0, ctx_gain: 0.0, ctx_hal_cut: 0.0, lazy: LazyMode::None, rho: 0.6, ..base.clone() },
    ]
}

struct Instance {
    skill: Vec<Vec<f64>>,
    tasks: Vec<(Task, usize)>,
    kw: HashMap<String, usize>,
}

fn build(w: &WorldCfg, seed: u64, widx: u64) -> Instance {
    let s = seed.wrapping_mul(1000) + widx;
    let mut rng = Rng::new(s);
    let mut skill = vec![vec![0.0; w.topics]; N_AGENTS];
    for t in 0..w.topics {
        for a in 0..N_AGENTS {
            skill[a][t] = if w.equal { 0.7 } else { 0.35 + 0.25 * rng.f64() };
        }
        if !w.equal {
            for _ in 0..2 {
                let a = rng.below(N_AGENTS);
                skill[a][t] = 0.85 + 0.07 * rng.f64();
            }
        }
    }
    let mut kw = HashMap::new();
    for t in 0..w.topics {
        let best = (0..N_AGENTS).fold(0, |b, a| if skill[a][t] > skill[b][t] { a } else { b });
        let adv = if w.equal || rng.f64() < w.decoy { rng.below(N_AGENTS) } else { best };
        for j in 0..3 {
            kw.insert(format!("t{t}w{j}"), adv);
        }
    }
    let wts: Vec<f64> = (0..w.topics).map(|r| 1.0 / ((r + 1) as f64).powf(0.7)).collect();
    let tot: f64 = wts.iter().sum();
    let mut tasks = Vec::with_capacity(N_TASKS);
    for i in 0..N_TASKS {
        let mut x = rng.f64() * tot;
        let mut topic = w.topics - 1;
        for (t, wt) in wts.iter().enumerate() {
            if x < *wt {
                topic = t;
                break;
            }
            x -= wt;
        }
        let mut f: Vec<String> = (0..3).map(|j| format!("t{topic}w{j}")).collect();
        f.push(format!("dom{}", topic % 3));
        if w.recur {
            f.push(format!("n{}", rng.below(30)));
        } else {
            for j in 0..8 {
                f.push(format!("u{i}x{j}"));
            }
        }
        tasks.push((Task { id: i as u64, task_type: None, features: f }, topic));
    }
    Instance { skill, tasks, kw }
}

#[derive(Clone)]
struct Draft {
    agent: usize,
    topic: usize,
    q: f64,
    hal: bool,
    ok: bool,
    tid: u64,
    conf: f64,
}

struct WorldExec<'a> {
    w: &'a WorldCfg,
    inst: &'a Instance,
    seed: u64,
    topic_of: Vec<usize>,
    last_ctx: Context,
}

impl WorldExec<'_> {
    fn lazy(&self, c: usize) -> f64 {
        match self.w.lazy {
            LazyMode::All => 0.15,
            LazyMode::Third if c % 3 == 2 => 0.15,
            _ => 1.0,
        }
    }
    fn make(&self, task: &Task, agent: usize, ctx: &Context) -> Draft {
        let topic = self.topic_of[task.id as usize];
        let (s, tid) = (self.inst.skill[agent][topic], task.id);
        let down = self.w.outage && agent == 7 && (600..1000).contains(&(tid as usize));
        let n = ctx.note.unwrap_or(0.0);
        let p_hal = ((self.w.hal_floor + self.w.hal_base * (1.0 - s)) * (1.0 - self.w.ctx_hal_cut * n)).clamp(0.0, 1.0);
        let hal = uni(self.seed, tid, agent as u64, 11) < p_hal;
        let q = if down {
            0.0
        } else if hal {
            0.05 + 0.2 * uni(self.seed, tid, agent as u64, 12)
        } else {
            (s + self.w.ctx_gain * n * (1.0 - s) + 0.07 * norm(self.seed, tid, agent as u64, 13)).clamp(0.0, 1.0)
        };
        let ok = !down && uni(self.seed, tid, agent as u64, 14) < 0.99 - 0.03 * (1.0 - s);
        let conf = (0.5 + self.w.rho * (q - 0.5) + self.w.conf_noise * norm(self.seed, tid, agent as u64, 18) + if hal { self.w.overconf } else { 0.0 }).clamp(0.0, 1.0);
        Draft { agent, topic, q, hal, ok, tid, conf }
    }
}

impl Executor for WorldExec<'_> {
    type Out = Draft;
    fn draft(&mut self, task: &Task, agent: usize, ctx: &Context) -> Draft {
        self.last_ctx = ctx.clone();
        self.make(task, agent, ctx)
    }
    fn tool_ok(&mut self, o: &Draft) -> bool {
        o.ok
    }
    fn check(&mut self, _t: &Task, checker: usize, o: &Draft) -> bool {
        let sc = self.inst.skill[checker][o.topic];
        let u = uni(self.seed, o.tid, (checker * 16 + o.agent) as u64, 15);
        if o.q < 0.5 {
            let catch = (0.1 + 0.88 * sc) * self.lazy(checker) * if o.hal || o.q == 0.0 { 1.0 } else { 0.5 };
            u < catch
        } else {
            u < (0.05 + 0.1 * (1.0 - sc)) * self.lazy(checker)
        }
    }
    fn grade(&mut self, t: &Task, o: &Draft) -> Option<f64> {
        if uni(self.seed, t.id, 0, 16) >= REVEAL {
            return None;
        }
        Some((o.q + 0.05 * norm(self.seed, t.id, o.agent as u64, 17)).clamp(0.0, 1.0))
    }
}

// ---------------------------------------------------------------- arms
#[derive(Clone, Debug)]
enum Trig {
    None,
    /// e3's unconditional verified gate (HierSecondOpinion, pool_leader 8) inside the session
    Gate,
    /// check iff generator confidence < theta
    Conf(f64),
    /// check iff the ONLINE-estimated P(bad | confidence bin) >= b* (prior Beta(.5,2); learns from delayed grades)
    Cal(f64),
    /// checks exactly the bad drafts (ceiling on trigger quality)
    Oracle,
}

#[derive(Clone, Debug)]
struct Arm {
    name: String,
    keyword_only: bool,
    guarded: bool,
    retr: bool,
    reuse: bool,
    trig: Trig,
}

fn arm(name: &str, guarded: bool, retr: bool, reuse: bool, trig: Trig) -> Arm {
    Arm { name: name.into(), keyword_only: false, guarded, retr, reuse, trig }
}

const THETAS: [f64; 4] = [0.2, 0.3, 0.4, 0.5];
const BSTARS: [f64; 3] = [0.35, 0.5, 0.65];

fn arms() -> Vec<Arm> {
    let mut v = vec![
        Arm { name: "keyword".into(), keyword_only: true, guarded: false, retr: false, reuse: false, trig: Trig::None },
        arm("plain", false, false, false, Trig::None),
        arm("plain-g", true, false, false, Trig::None),
        arm("R-warm", false, true, true, Trig::None),
        arm("R-safe", true, true, true, Trig::None),
        arm("reuse-only-safe", true, false, true, Trig::None),
        arm("retr-only-safe", true, true, false, Trig::None),
        arm("gate-safe", true, true, true, Trig::Gate),
    ];
    for t in THETAS {
        v.push(arm(&format!("conf<{t}"), true, true, true, Trig::Conf(t)));
    }
    for b in BSTARS {
        v.push(arm(&format!("cal>={b}"), true, true, true, Trig::Cal(b)));
    }
    v.push(arm("oracle-trig", true, true, true, Trig::Oracle));
    v
}

fn safe_params() -> GuardParams {
    GuardParams { tau: 0.7, z: 0.5, kw_a: 4.0, kw_b: 1.0, discount: 0.98, challenge: true }
}

struct Tuned {
    plain_k: f64,
}

fn session_cfg(a: &Arm, t: &Tuned) -> ResearchConfig {
    let gate = CriticKind::Hier(HierSoConfig {
        base: SecondOpinionConfig { cost: TUNE_COST, kappa: 0.0, discount: 0.995, ..SecondOpinionConfig::default() },
        pool_leader: 8.0,
        ..HierSoConfig::default()
    });
    ResearchConfig {
        router: if a.guarded { RouterChoice::GuardedHebr(safe_params()) } else { RouterChoice::WarmHebr },
        // e3's tuned values (frozen); plain arms (no reuse path) use their own tuned warm_k like e3
        warm_k: if a.reuse { 1.0 } else { t.plain_k },
        retrieval: a.retr,
        critic: if matches!(a.trig, Trig::Gate) { Some(gate) } else { None },
        reuse: a.reuse,
        delay: DELAY,
        store_min: 0.65,
        sim_min: 0.55,
        ..ResearchConfig::default()
    }
}

/// per-run metrics
const NM: usize = 11;
// 0 g250, 1 g500, 2 g2000, 3 extra calls/task, 4 ctx-hit, 5 bad_first, 6 bad_delivered, 7 AUROC(conf), 8 checks/task, 9 flags/task, 10 last500
fn auroc(good: &[f64], bad: &[f64]) -> f64 {
    if good.is_empty() || bad.is_empty() {
        return f64::NAN;
    }
    let mut gs = good.to_vec();
    gs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut s = 0.0;
    for b in bad {
        let lt = gs.partition_point(|x| x < b);
        let le = gs.partition_point(|x| x <= b);
        s += (gs.len() - le) as f64 + 0.5 * (le - lt) as f64;
    }
    s / (gs.len() as f64 * bad.len() as f64)
}

fn run(w: &WorldCfg, widx: u64, a: &Arm, t: &Tuned, seed: u64) -> [f64; NM] {
    let inst = build(w, seed, widx);
    let topic_of: Vec<usize> = inst.tasks.iter().map(|t| t.1).collect();
    let mut ex = WorldExec { w, inst: &inst, seed, topic_of, last_ctx: Context::default() };
    let mut kwr = KeywordRouter::new(inst.kw.clone(), N_AGENTS);
    let mut s = ResearchSession::new(N_AGENTS, inst.kw.clone(), session_cfg(a, t));
    let mut rng = Rng::new(seed * 7 + 1);
    let cands: Vec<usize> = (0..N_AGENTS).collect();
    let (mut s250, mut s500, mut sum, mut last, mut extra, mut checks, mut flags) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut bad_first, mut bad_deliv) = (0.0, 0.0);
    let (mut cg, mut cb): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
    // online calibration of confidence (Cal trigger): bins of first-draft confidence
    let mut bins = [(0.0f64, 0.0f64); 10]; // (bad, n)
    let mut pend: VecDeque<(usize, usize, bool)> = VecDeque::new(); // (due, bin, bad) for graded drafts
    for (i, (task, _)) in inst.tasks.iter().enumerate() {
        while pend.front().is_some_and(|e| e.0 <= i) {
            let (_, b, bad) = pend.pop_front().unwrap();
            bins[b].1 += 1.0;
            if bad {
                bins[b].0 += 1.0;
            }
        }
        let (q, e, chk, flg);
        if a.keyword_only {
            let ag = kwr.select(task, &cands, &mut rng);
            let d = ex.make(task, ag, &Context::default());
            q = d.q;
            e = 0.0;
            chk = 0.0;
            flg = 0.0;
            let _ = (d.hal, d.ok);
            if d.q < 0.5 {
                bad_first += 1.0;
                bad_deliv += 1.0;
                cb.push(d.conf);
            } else {
                cg.push(d.conf);
            }
        } else {
            let rep = s.run_task(&mut ex, task, &cands, &mut rng);
            let first = rep.delivered.clone();
            let mut delivered_q = first.q;
            let mut ee = (rep.calls - 1) as f64; // gate-driven extra calls
            let (mut c1, mut f1) = (if rep.checked { 1.0 } else { 0.0 }, if rep.flagged { 1.0 } else { 0.0 });
            let bad = first.q < 0.5;
            if matches!(a.trig, Trig::Gate) {
                // gate arm: the session already applied the critic; `delivered` is the final draft,
                // bad_first/AUROC come from the first draft (== delivered when no flag)
            }
            // post-step triggers (session critic is None for these)
            let fire = match &a.trig {
                Trig::Conf(th) => first.conf < *th,
                Trig::Cal(bstar) => {
                    let b = ((first.conf * 10.0) as usize).min(9);
                    let (bd, n) = bins[b];
                    (bd + 0.5) / (n + 2.5) >= *bstar
                }
                Trig::Oracle => bad,
                _ => false,
            };
            if fire && cands.len() >= 2 {
                // best other (healthy) agent by the router's exploit scores
                let sc = s.brain.router.inner().scores(task);
                let mut bo: Option<usize> = None;
                for &c in &cands {
                    if c == first.agent || s.brain.is_benched(c) {
                        continue;
                    }
                    if bo.is_none_or(|b| sc.get(c).copied().unwrap_or(0.0) > sc.get(b).copied().unwrap_or(0.0)) {
                        bo = Some(c);
                    }
                }
                if let Some(o) = bo {
                    let ctx = ex.last_ctx.clone();
                    ee += 1.0;
                    c1 = 1.0;
                    if ex.check(task, o, &first) {
                        f1 = 1.0;
                        ee += 1.0;
                        let second = ex.make(task, o, &ctx);
                        delivered_q = second.q;
                    }
                }
            }
            // calibration record: graded first drafts arrive DELAY later (REVEAL fraction)
            if matches!(a.trig, Trig::Cal(_)) {
                if let Some(g) = ex.grade(task, &first) {
                    pend.push_back((i + DELAY, ((first.conf * 10.0) as usize).min(9), g < 0.5));
                }
            }
            if bad {
                bad_first += 1.0;
                cb.push(first.conf);
            } else {
                cg.push(first.conf);
            }
            if delivered_q < 0.5 {
                bad_deliv += 1.0;
            }
            q = delivered_q;
            e = ee;
            chk = c1;
            flg = f1;
        }
        if i < 250 {
            s250 += q;
        }
        if i < 500 {
            s500 += q;
        }
        sum += q;
        if i >= N_TASKS - 500 {
            last += q;
        }
        extra += e;
        checks += chk;
        flags += flg;
    }
    let n = N_TASKS as f64;
    [s250 / 250.0, s500 / 500.0, sum / n, extra / n, s.stats.ctx_hits as f64 / n, bad_first / n, bad_deliv / n, auroc(&cg, &cb), checks / n, flags / n, last / 500.0]
}

fn eval(w: &WorldCfg, widx: u64, a: &Arm, t: &Tuned, seeds: &[u64]) -> Vec<[f64; NM]> {
    seeds.par_iter().map(|&s| run(w, widx, a, t, s)).collect()
}
fn col(v: &[[f64; NM]], k: usize) -> Vec<f64> {
    v.iter().map(|r| r[k]).collect()
}
fn net(v: &[[f64; NM]], c: f64) -> Vec<f64> {
    v.iter().map(|r| r[2] - c * r[3]).collect()
}
fn mean(v: &[f64]) -> f64 {
    let f: Vec<f64> = v.iter().copied().filter(|x| !x.is_nan()).collect();
    f.iter().sum::<f64>() / f.len().max(1) as f64
}

fn verdict(hp: f64, m: f64) -> &'static str {
    if hp < 0.05 && m > 0.0 {
        "WIN"
    } else if hp < 0.05 && m < 0.0 {
        "LOSS"
    } else {
        "n.s."
    }
}

struct Cmp {
    world: usize,
    label: String,
    metric: String,
    mean: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    losses: usize,
    p: f64,
}

fn compare(world: usize, label: &str, metric: &str, a: &[f64], b: &[f64]) -> Cmp {
    let d = paired_diff(a, b);
    Cmp { world, label: label.into(), metric: metric.into(), mean: d.diff.mean, lo: d.diff.lo, hi: d.diff.hi, wins: d.wins, losses: d.losses, p: paired_t_pvalue(a, b) }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/f7_research.json".into());
    let held: Vec<u64> = match args.iter().position(|a| a == "--held").and_then(|i| args.get(i + 1)) {
        Some(r) => {
            let (a, b) = r.split_once('-').expect("--held a-b");
            (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
        }
        None => (31..=60).collect(),
    };
    let tune: Vec<u64> = (1..=8).collect();
    let t0 = Instant::now();
    let ws = worlds();
    let al = arms();
    println!("f7_research: {} worlds x {} arms, tune seeds 1-8, eval {}-{}", ws.len(), al.len(), held[0], held[held.len() - 1]);

    // ---- tuning (seeds 1-8) ----
    let mut tuned = Tuned { plain_k: 1.0 };
    let mut bp = (f64::NEG_INFINITY, 1.0);
    for &pk in &[1.0, 3.0, 10.0] {
        tuned.plain_k = pk;
        let a = al.iter().find(|a| a.name == "plain").unwrap();
        let o: f64 = ws.iter().enumerate().map(|(i, w)| mean(&col(&eval(w, i as u64, a, &tuned, &tune), 2))).sum::<f64>() / ws.len() as f64;
        if o > bp.0 {
            bp = (o, pk);
        }
    }
    tuned.plain_k = bp.1;
    println!("[tune] plain warm_k = {} (obj {:.4})", bp.1, bp.0);
    // objective for trigger parameters: mean NET at cost .03 over ALL worlds (incl. controls)
    let tune_res: Vec<(usize, Vec<Vec<[f64; NM]>>)> = al
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a.trig, Trig::Conf(_) | Trig::Cal(_)))
        .map(|(ai, a)| (ai, ws.iter().enumerate().map(|(wi, w)| eval(w, wi as u64, a, &tuned, &tune)).collect()))
        .collect();
    let obj = |ai: usize| -> f64 {
        let r = &tune_res.iter().find(|x| x.0 == ai).unwrap().1;
        r.iter().map(|rows| mean(&net(rows, TUNE_COST))).sum::<f64>() / ws.len() as f64
    };
    let mut best_conf = (f64::NEG_INFINITY, 0usize);
    let mut best_cal = (f64::NEG_INFINITY, 0usize);
    for (ai, a) in al.iter().enumerate() {
        match a.trig {
            Trig::Conf(_) => {
                let o = obj(ai);
                println!("[tune] {:<12} mean net@{TUNE_COST} over {} worlds = {:.5}", a.name, ws.len(), o);
                if o > best_conf.0 {
                    best_conf = (o, ai);
                }
            }
            Trig::Cal(_) => {
                let o = obj(ai);
                println!("[tune] {:<12} mean net@{TUNE_COST} over {} worlds = {:.5}", a.name, ws.len(), o);
                if o > best_cal.0 {
                    best_cal = (o, ai);
                }
            }
            _ => {}
        }
    }
    let rsafe_i = al.iter().position(|a| a.name == "R-safe").unwrap();
    let rsafe_tune: f64 = ws.iter().enumerate().map(|(wi, w)| mean(&net(&eval(w, wi as u64, &al[rsafe_i], &tuned, &tune), TUNE_COST))).sum::<f64>() / ws.len() as f64;
    println!("[tune] R-safe (no critic) mean net = {rsafe_tune:.5}; chosen conf trigger = {} ({:.5}), calibrated = {} ({:.5})", al[best_conf.1].name, best_conf.0, al[best_cal.1].name, best_cal.0);

    // ---- evaluation ----
    let mut res: HashMap<(usize, usize), Vec<[f64; NM]>> = HashMap::new();
    for (wi, w) in ws.iter().enumerate() {
        for (ai, a) in al.iter().enumerate() {
            res.insert((wi, ai), eval(w, wi as u64, a, &tuned, &held));
        }
    }
    let ix = |n: &str| al.iter().position(|a| a.name == n).unwrap();
    let (i_kw, i_plain, i_plain_g, i_rw, i_rs, i_ro, i_rt, i_gate) = (ix("keyword"), ix("plain"), ix("plain-g"), ix("R-warm"), ix("R-safe"), ix("reuse-only-safe"), ix("retr-only-safe"), ix("gate-safe"));
    let i_conf = best_conf.1;
    let i_cal = best_cal.1;
    let i_or = ix("oracle-trig");

    // family 1: pipeline vs plain / ablations (gross; no extra calls)
    let f1_comps: Vec<(&str, usize, usize)> = vec![
        ("R-safe vs plain", i_rs, i_plain),
        ("R-safe vs plain-g", i_rs, i_plain_g),
        ("R-warm vs plain", i_rw, i_plain),
        ("R-safe vs R-warm", i_rs, i_rw),
        ("R-safe vs keyword", i_rs, i_kw),
        ("R-safe vs reuse-only (retrieval's contribution)", i_rs, i_ro),
        ("R-safe vs retr-only (reuse's contribution)", i_rs, i_rt),
    ];
    let f1_metrics = [(0usize, "gross_H250"), (1, "gross_H500"), (2, "gross_H2000")];
    let mut f1: Vec<Cmp> = vec![];
    for wi in 0..ws.len() {
        for (lab, a, b) in &f1_comps {
            for (m, mn) in &f1_metrics {
                f1.push(compare(wi, lab, mn, &col(&res[&(wi, *a)], *m), &col(&res[&(wi, *b)], *m)));
            }
        }
    }
    // family 2: critic arms vs R-safe (same router/retrieval/reuse): net at 3 call costs + gross
    let mut f2_comps: Vec<(String, usize)> = vec![("gate-safe (unconditional e3 critic)".into(), i_gate), (format!("{} (global theta)", al[i_conf].name), i_conf), (format!("{} (global b*)", al[i_cal].name), i_cal), ("oracle-trig (ceiling)".into(), i_or)];
    f2_comps.dedup_by(|a, b| a.1 == b.1);
    let mut f2: Vec<Cmp> = vec![];
    for wi in 0..ws.len() {
        for (lab, a) in &f2_comps {
            f2.push(compare(wi, lab, "gross_H2000", &col(&res[&(wi, *a)], 2), &col(&res[&(wi, i_rs)], 2)));
            for c in COSTS {
                f2.push(compare(wi, lab, &format!("net@{c}"), &net(&res[&(wi, *a)], c), &net(&res[&(wi, i_rs)], c)));
            }
        }
    }
    let adj1 = holm_adjust(&f1.iter().map(|c| c.p).collect::<Vec<_>>());
    let adj2 = holm_adjust(&f2.iter().map(|c| c.p).collect::<Vec<_>>());

    // ---- print ----
    println!("\nTABLE (mean over eval seeds): gross H250 / gross H2000 / extra calls per task / bad-delivered rate; measured AUROC of confidence");
    print!("{:<19}{:>8}", "world", "AUROC");
    for a in &al {
        print!("{:>14}", a.name);
    }
    println!();
    let mut table = vec![];
    for (wi, w) in ws.iter().enumerate() {
        print!("{:<19}{:>8.3}", w.name, mean(&col(&res[&(wi, i_rs)], 7)));
        let mut row = serde_json::Map::new();
        for (ai, a) in al.iter().enumerate() {
            let r = &res[&(wi, ai)];
            print!("{:>14.4}", mean(&col(r, 2)));
            row.insert(a.name.clone(), json!({"gross_H250": mean(&col(r,0)), "gross_H500": mean(&col(r,1)), "gross_H2000": mean(&col(r,2)), "last500": mean(&col(r,10)), "extra_calls_per_task": mean(&col(r,3)), "net_0.01": mean(&net(r, 0.01)), "net_0.03": mean(&net(r, 0.03)), "net_0.10": mean(&net(r, 0.10)), "ctx_hit": mean(&col(r,4)), "bad_first": mean(&col(r,5)), "bad_delivered": mean(&col(r,6)), "checks_per_task": mean(&col(r,8)), "flags_per_task": mean(&col(r,9))}));
        }
        println!();
        table.push(json!({"world": w.name, "auroc_conf": mean(&col(&res[&(wi, i_rs)], 7)), "bad_first_rate": mean(&col(&res[&(wi, i_rs)], 5)), "arms": Value::Object(row)}));
    }
    println!("\nEXTRA CALLS / TASK");
    for (wi, w) in ws.iter().enumerate() {
        print!("{:<19}{:>8}", w.name, "");
        for (ai, _) in al.iter().enumerate() {
            print!("{:>14.3}", mean(&col(&res[&(wi, ai)], 3)));
        }
        println!();
    }
    println!("\nNET @ call cost .03 (gross H2000 - .03 x extra calls)");
    for (wi, w) in ws.iter().enumerate() {
        print!("{:<19}{:>8}", w.name, "");
        for (ai, _) in al.iter().enumerate() {
            print!("{:>14.4}", mean(&net(&res[&(wi, ai)], 0.03)));
        }
        println!();
    }
    let show = |title: &str, cs: &[Cmp], adj: &[f64], only_h2000: bool| {
        println!("\n=== {title}: Holm over {} tests (WIN/LOSS only) ===", cs.len());
        let (mut nw, mut nl) = (0, 0);
        for (i, c) in cs.iter().enumerate() {
            let v = verdict(adj[i], c.mean);
            if v == "WIN" {
                nw += 1;
            } else if v == "LOSS" {
                nl += 1;
            }
            if v != "n.s." && (!only_h2000 || c.metric != "gross_H500") {
                println!("  {:<19} {:<46} {:<12} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.4} {}", ws[c.world].name, c.label, c.metric, c.mean, c.lo, c.hi, c.wins, c.losses, adj[i], v);
            }
        }
        println!("  -> {nw} WIN / {nl} LOSS");
        (nw, nl)
    };
    let (w1, l1) = show("F1 pipeline vs plain & ablations", &f1, &adj1, true);
    let (w2, l2) = show("F2 critic variants vs R-safe (no critic)", &f2, &adj2, false);

    let cj = |cs: &[Cmp], adj: &[f64]| -> Vec<Value> {
        cs.iter().enumerate().map(|(i, c)| json!({"world": ws[c.world].name, "comparison": c.label, "metric": c.metric, "mean_diff": c.mean, "lo": c.lo, "hi": c.hi, "wins": c.wins, "losses": c.losses, "p": c.p, "holm_p": adj[i], "verdict": verdict(adj[i], c.mean)})).collect()
    };
    let out = json!({
        "experiment": "f7_research", "tasks": N_TASKS, "agents": N_AGENTS, "reveal": REVEAL, "delay": DELAY, "costs": COSTS, "tune_cost": TUNE_COST,
        "tune_seeds": "1-8", "eval_seeds": format!("{}-{}", held[0], held[held.len() - 1]),
        "frozen_from_e3": {"store_min": 0.65, "sim_min": 0.55, "warm_k_reuse": 1.0},
        "tuned": {"plain_warm_k": tuned.plain_k, "conf_theta_arm": al[i_conf].name, "conf_objective": best_conf.0, "cal_arm": al[i_cal].name, "cal_objective": best_cal.0, "R_safe_objective": rsafe_tune},
        "guard_params": format!("{:?}", safe_params()),
        "arms": al.iter().map(|a| a.name.clone()).collect::<Vec<_>>(), "table": table,
        "F1": cj(&f1, &adj1), "F2": cj(&f2, &adj2), "F1_holm_win": w1, "F1_holm_loss": l1, "F2_holm_win": w2, "F2_holm_loss": l2,
        "runtime_s": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nwrote {out_path} (runtime {:.1}s)", t0.elapsed().as_secs_f64());
}
