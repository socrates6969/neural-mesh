//! H7 engine: multi-candidate runs (best-of-k / verbalized-sampling) over an `HEnv`, using the frozen g4 policies unchanged.
//! Costs: gen calls (generator-call equivalents) and judge calls are counted per creative-valued task.
use super::hworld::*;
use super::pols::{Div, Gate, GateKind, Kind, EvK, WP};
use super::world::{Pol, View};
use neural_mesh::router::Rng;
use neural_mesh::stats::{paired_diff, paired_t_pvalue, t_two_sided_p};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub enum CellFn {
    Sign,
    Grid(usize),
}
#[derive(Clone, Debug)]
pub enum Sel {
    /// pick the candidate with the highest judged quality only
    QOnly,
    /// drop candidates within euclid `theta` of any of the last WP delivered outputs (for unit-norm embeddings cos = 1 - d^2/2),
    /// then argmax qhat*((1-w)+w*nov_hat); if all dropped, take the farthest
    Dedup { theta: f64, w: f64 },
    /// archive/MAP-Elites-flavoured: argmax qhat*((1-w)+w*staleness of the candidate's own cell), staleness scale t
    Arch { w: f64, t: f64, cell: CellFn },
    /// CLAIRVOYANT reference only: delivers the candidate with the highest TRUE value (not a candidate method)
    OracleVal,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EGate {
    Tag,
    Learned,
    Always,
    Never,
}
#[derive(Clone, Copy, Debug)]
pub struct VsProf {
    pub sigma: f64,
    pub kappa: f64,
    pub rho: f64,
}
#[derive(Clone, Debug)]
pub struct Mode {
    pub k: usize,
    pub vs: Option<VsProf>,
    pub sel: Sel,
    pub gate: EGate,
    /// judge noise sd on realized q; < 0 => no judge (constant qhat)
    pub judge_sd: f64,
}
impl Mode {
    pub fn single() -> Mode {
        Mode { k: 1, vs: None, sel: Sel::QOnly, gate: EGate::Never, judge_sd: 0.05 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct HOut {
    pub overall: f64,
    pub creative: f64,
    pub correct: f64,
    pub nov_cr: f64,
    pub calls_cr: f64,
    pub judge_cr: f64,
    pub calls_all: f64,
    pub on_cr: f64,
    pub on_cor: f64,
    pub n_cr: usize,
    pub n_cor: usize,
    pub diag: Vec<f64>,
}

fn nov3(w: &VecDeque<Vec<f64>>, p: &[f64]) -> f64 {
    if w.is_empty() {
        return 1.0;
    }
    let mut ds: Vec<f64> = w.iter().map(|q| dist(p, q)).collect();
    ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ds.len().min(3);
    (ds[..k].iter().sum::<f64>() / k as f64).min(1.0)
}
fn dmin(w: &VecDeque<Vec<f64>>, p: &[f64]) -> f64 {
    w.iter().map(|q| dist(p, q)).fold(f64::INFINITY, f64::min)
}
fn cell_of(c: CellFn, p: &[f64]) -> u64 {
    match c {
        CellFn::Sign => Div::cell_of(p) as u64,
        CellFn::Grid(m) => {
            let f = |x: f64| ((x.clamp(0.0, 0.999_999) * m as f64) as usize).min(m - 1) as u64;
            f(p[0]) * m as u64 + f(p[1])
        }
    }
}

struct Sess {
    win: VecDeque<Vec<f64>>,
    last: HashMap<u64, usize>,
    cnt: usize,
}

fn pick(obs: (usize, usize), cands: &[Cand], qhat: &[f64], s: &Sess, sel: &Sel) -> usize {
    let o = |c: &Cand| c.out[obs.0..obs.1].to_vec();
    let argmax = |sc: &[f64]| sc.iter().enumerate().fold(0, |b, (i, &v)| if v > sc[b] { i } else { b });
    match sel {
        Sel::QOnly | Sel::OracleVal => argmax(qhat),
        Sel::Dedup { theta, w } => {
            let mut best: Option<(f64, usize)> = None;
            let mut far = (f64::NEG_INFINITY, 0);
            for (i, c) in cands.iter().enumerate() {
                let p = o(c);
                let dm = if s.win.is_empty() { f64::INFINITY } else { dmin(&s.win, &p) };
                if dm > far.0 {
                    far = (dm, i);
                }
                if dm >= *theta {
                    let sc = qhat[i] * ((1.0 - w) + w * nov3(&s.win, &p));
                    if best.map_or(true, |b| sc > b.0) {
                        best = Some((sc, i));
                    }
                }
            }
            best.map(|b| b.1).unwrap_or(far.1)
        }
        Sel::Arch { w, t, cell } => {
            let sc: Vec<f64> = cands
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let st = match s.last.get(&cell_of(*cell, &o(c))) {
                        None => 1.0,
                        Some(&f) => (((s.cnt - f) as f64) / t).min(1.0),
                    };
                    qhat[i] * ((1.0 - w) + w * st)
                })
                .collect();
            argmax(&sc)
        }
    }
}

pub fn run_h(env: &HEnv, pol: &mut dyn Pol, mode: &Mode, pseed: u64) -> HOut {
    let (na, nd) = (env.na, env.nd);
    let cand: Vec<usize> = (0..na).collect();
    let mut st = env.new_state();
    let mut prng = Rng::new(pseed.wrapping_mul(7919).wrapping_add(13));
    let mut ses: Vec<Sess> = (0..nd).map(|_| Sess { win: VecDeque::new(), last: HashMap::new(), cnt: 0 }).collect();
    let mut gate = Gate::new(GateKind::Learned, &env.tag_dom, na);
    let (mut sum, mut sc, mut sk, mut sn) = (0.0, 0.0, 0.0, 0.0);
    let (mut calls_cr, mut judge_cr, mut calls_all) = (0.0, 0.0, 0.0);
    let (mut ncr, mut ncor, mut on_cr, mut on_cor) = (0usize, 0usize, 0usize, 0usize);
    let hist_empty: Vec<std::collections::VecDeque<[f64; 4]>> = vec![];
    for t in 0..env.nt {
        let d = env.dom[t];
        let task = env.task(t);
        let view = View { w: &env.shell, t, hist: &hist_empty, last_nov: 1.0 };
        let on = match mode.gate {
            EGate::Tag => env.tag_task[t],
            EGate::Learned => gate.learned_on(d),
            EGate::Always => true,
            EGate::Never => false,
        };
        let multi = mode.k > 1 && on;
        let (cands, gen_calls, judge_calls): (Vec<Cand>, f64, f64) = if !multi {
            let a = pol.select(&task, &cand, &mut prng, &view);
            (vec![env.emit(t, a, 0, None)], 1.0, 0.0)
        } else if mode.vs.is_none() {
            let mut left = cand.clone();
            let mut v = vec![];
            for j in 0..mode.k.min(na) {
                let a = pol.select(&task, &left, &mut prng, &view);
                left.retain(|&x| x != a);
                v.push(env.emit(t, a, j, None));
            }
            let n = v.len() as f64;
            (v, n, if mode.judge_sd >= 0.0 { n } else { 0.0 })
        } else {
            let p = mode.vs.unwrap();
            let a = pol.select(&task, &cand, &mut prng, &view);
            let off = VsOff { sigma: p.sigma, kappa: p.kappa };
            let v: Vec<Cand> = (0..mode.k).map(|j| env.emit(t, a, j, if j == 0 { None } else { Some(&off) })).collect();
            (v, 1.0 + p.rho * (mode.k as f64 - 1.0), if mode.judge_sd >= 0.0 { mode.k as f64 } else { 0.0 })
        };
        let idx = if cands.len() == 1 {
            0
        } else {
            let qhat: Vec<f64> = cands
                .iter()
                .enumerate()
                .map(|(j, c)| {
                    if mode.judge_sd < 0.0 {
                        0.7
                    } else {
                        let mut r = Rng::new(hash3(env.seed, t as u64, j as u64 + 9000));
                        c.q + mode.judge_sd * normal(&mut r)
                    }
                })
                .collect();
            if let Sel::OracleVal = mode.sel {
                let vals: Vec<f64> = cands.iter().map(|c| env.value(&st, d, c.q, &c.out).0).collect();
                vals.iter().enumerate().fold(0, |b, (i, &v)| if v > vals[b] { i } else { b })
            } else {
                pick(env.obs, &cands, &qhat, &ses[d], &mode.sel)
            }
        };
        let c = &cands[idx];
        let (value, nov) = env.value(&st, d, c.q, &c.out);
        let o_obs = &c.out[env.obs.0..env.obs.1];
        let v2 = View { w: &env.shell, t, hist: &hist_empty, last_nov: nov };
        pol.observe(&task, c.agent, value, o_obs, &v2);
        if mode.gate == EGate::Learned {
            gate.update(d, c.agent, value, nov3(&ses[d].win, o_obs));
        }
        env.push(&mut st, d, &c.out);
        {
            let s = &mut ses[d];
            if let Sel::Arch { cell, .. } = &mode.sel {
                s.last.insert(cell_of(*cell, o_obs), s.cnt);
            }
            s.cnt += 1;
            s.win.push_back(o_obs.to_vec());
            if s.win.len() > WP {
                s.win.pop_front();
            }
        }
        sum += value;
        calls_all += gen_calls;
        if env.vcr[d] {
            sc += value;
            sn += nov;
            ncr += 1;
            calls_cr += gen_calls;
            judge_cr += judge_calls;
            if multi {
                on_cr += 1;
            }
        } else {
            sk += value;
            ncor += 1;
            if multi {
                on_cor += 1;
            }
        }
    }
    let n = env.nt as f64;
    let f = |a: f64, b: usize| if b > 0 { a / b as f64 } else { f64::NAN };
    HOut {
        overall: sum / n,
        creative: f(sc, ncr),
        correct: f(sk, ncor),
        nov_cr: f(sn, ncr),
        calls_cr: f(calls_cr, ncr),
        judge_cr: f(judge_cr, ncr),
        calls_all: calls_all / n,
        on_cr: f(on_cr as f64, ncr),
        on_cor: f(on_cor as f64, ncor),
        n_cr: ncr,
        n_cor: ncor,
        diag: pol.diag(),
    }
}

// ---------------------------------------------------------------- eval harness
#[derive(Clone)]
pub struct PolSpec {
    pub name: String,
    pub mode: Mode,
    pub build: Arc<dyn Fn(&HEnv) -> Box<dyn Pol> + Send + Sync>,
}
pub fn eval_spec(spec: &PolSpec, envs: &[HEnv], seeds: &[u64]) -> Vec<HOut> {
    envs.par_iter()
        .zip(seeds.par_iter())
        .map(|(e, &s)| {
            let mut p = (spec.build)(e);
            run_h(e, p.as_mut(), &spec.mode, s)
        })
        .collect()
}
pub fn mean_overall(v: &[HOut]) -> f64 {
    v.iter().map(|r| r.overall).sum::<f64>() / v.len() as f64
}
pub fn col(v: &[HOut], f: fn(&HOut) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}
pub fn f_overall(r: &HOut) -> f64 {
    r.overall
}
pub fn f_creative(r: &HOut) -> f64 {
    r.creative
}
pub fn f_correct(r: &HOut) -> f64 {
    r.correct
}
pub fn f_calls(r: &HOut) -> f64 {
    r.calls_cr
}
pub fn f_judge(r: &HOut) -> f64 {
    r.judge_cr
}
pub fn mean(v: &[f64]) -> f64 {
    let g: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
    if g.is_empty() { f64::NAN } else { g.iter().sum::<f64>() / g.len() as f64 }
}

// ---------------------------------------------------------------- fair baselines (same grid as g4)
pub fn fair_grid() -> Vec<Kind> {
    let mut v = vec![];
    for &g in &[0.9, 0.95, 0.99, 1.0] {
        for &e in &[0.02, 0.05, 0.1] {
            v.push(Kind::Eps(g, e));
        }
    }
    for &g in &[0.9, 0.95, 0.99, 1.0] {
        v.push(Kind::Pts(g));
    }
    for &k in &[3.0, 10.0, 30.0, 100.0] {
        for &e in &[EvK::Set, EvK::Mean, EvK::FailSet] {
            v.push(Kind::Warm(k, e));
        }
    }
    v.push(Kind::HebR);
    v.push(Kind::Keyword);
    v.push(Kind::GuardedSafe);
    v
}
pub const FAMILIES: [&str; 6] = ["keyword", "eps-greedy", "pooled-thompson", "hebbian-robust", "warm(hebr)", "guarded-safe"];

/// per-world tuning of the fair baselines on tune seeds: returns the best-family config (the [G] reference of g4).
pub fn tune_g(envs: &[HEnv], seeds: &[u64]) -> Kind {
    let fg = fair_grid();
    let sc: Vec<f64> = fg
        .par_iter()
        .map(|k| {
            let spec = PolSpec { name: k.label(), mode: Mode::single(), build: { let k = k.clone(); Arc::new(move |e: &HEnv| Box::new(super::pols::FairPol::new(&k, &e.shell)) as Box<dyn Pol>) } };
            mean_overall(&eval_spec(&spec, envs, seeds))
        })
        .collect();
    let mut best = (0usize, f64::NEG_INFINITY);
    for (i, &s) in sc.iter().enumerate() {
        if s > best.1 {
            best = (i, s);
        }
    }
    fg[best.0].clone()
}

// ---------------------------------------------------------------- statistics (copied from g4_creativity)
#[derive(Clone, Debug, serde::Serialize)]
pub struct T {
    pub diff: f64,
    pub lo: f64,
    pub hi: f64,
    pub lo90: f64,
    pub hi90: f64,
    pub p: f64,
    pub wins: usize,
    pub losses: usize,
    pub tost01: f64,
    pub n: usize,
}
fn p_greater(t: f64, df: usize) -> f64 {
    let p2 = t_two_sided_p(t.abs(), df);
    if t > 0.0 { 0.5 * p2 } else { 1.0 - 0.5 * p2 }
}
fn t_crit_90(df: usize) -> f64 {
    let (mut lo, mut hi) = (0.5f64, 10.0f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if t_two_sided_p(mid, df) > 0.10 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}
pub fn tost_p(m: f64, se: f64, df: usize, delta: f64) -> f64 {
    if se < 1e-15 {
        return if m.abs() < delta { 0.0 } else { 1.0 };
    }
    let p_lo = p_greater((m + delta) / se, df);
    let p_hi = p_greater(-(m - delta) / se, df);
    p_lo.max(p_hi)
}
pub fn test(a: &[f64], b: &[f64]) -> T {
    let pd = paired_diff(a, b);
    let n = a.len();
    let d = &pd.diff;
    let se = d.sd / (n as f64).sqrt();
    let t90 = t_crit_90(n - 1);
    T { diff: d.mean, lo: d.lo, hi: d.hi, lo90: d.mean - t90 * se, hi90: d.mean + t90 * se, p: paired_t_pvalue(a, b), wins: pd.wins, losses: pd.losses, tost01: tost_p(d.mean, se, n - 1, 0.01), n }
}
