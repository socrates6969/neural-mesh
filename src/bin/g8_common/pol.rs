//! Policy adapters + the run loop (decision -> delayed / anonymous feedback).

use super::snn::{LCfg, Learner};
use super::worlds::{Env, Feedback, NA};
use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::defaults::recommended_router;
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router};

pub trait Pol {
    fn decide(&mut self, env: &Env, i: usize, rng: &mut Rng) -> usize;
    /// reward with a known decision id (immediate or delayed)
    fn rew_id(&mut self, env: &Env, i: usize, a: usize, r: f64);
    /// anonymous reward arriving at step `now` (before decision `now`)
    fn rew_anon(&mut self, env: &Env, now: usize, r: f64);
}

/// How a classic Router (which needs (task, agent, reward)) is fed an anonymous reward.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Attrib {
    None,
    /// credit the decision exactly L steps ago
    Lag(usize),
    /// sample the credited decision from a geometric lag distribution with this mean
    Geo(f64),
}

pub struct RouterPol {
    pub r: Box<dyn Router>,
    pub att: Attrib,
    hist: Vec<usize>,
    rng: Rng,
}
impl RouterPol {
    pub fn new(r: Box<dyn Router>, att: Attrib, seed: u64) -> Self {
        RouterPol { r, att, hist: Vec::new(), rng: Rng::new(seed.wrapping_mul(31).wrapping_add(5)) }
    }
}
impl Pol for RouterPol {
    fn decide(&mut self, env: &Env, i: usize, rng: &mut Rng) -> usize {
        let cands: [usize; NA] = [0, 1, 2, 3, 4, 5];
        let a = self.r.select(&env.tasks[i], &cands, rng);
        self.hist.push(a);
        a
    }
    fn rew_id(&mut self, env: &Env, i: usize, a: usize, r: f64) {
        self.r.update(&env.tasks[i], a, r);
    }
    fn rew_anon(&mut self, env: &Env, now: usize, r: f64) {
        let j = match self.att {
            Attrib::None => return,
            Attrib::Lag(l) => {
                if now < l {
                    return;
                }
                now - l
            }
            Attrib::Geo(m) => {
                let q = 1.0 / m.max(1.0);
                let mut d = 1;
                while !self.rng.bernoulli(q) && d < 80 {
                    d += 1;
                }
                if now < d {
                    return;
                }
                now - d
            }
        };
        if j < self.hist.len() {
            self.r.update(&env.tasks[j], self.hist[j], r);
        }
    }
}

pub struct SnnPol {
    pub l: Learner,
}
impl Pol for SnnPol {
    fn decide(&mut self, env: &Env, i: usize, rng: &mut Rng) -> usize {
        let cands: [usize; NA] = [0, 1, 2, 3, 4, 5];
        self.l.decide(&env.tasks[i], env.x[i], &cands, rng)
    }
    fn rew_id(&mut self, _env: &Env, i: usize, _a: usize, r: f64) {
        self.l.reward_id(i, r);
    }
    fn rew_anon(&mut self, _env: &Env, now: usize, r: f64) {
        self.l.reward_anon(now, r);
    }
}

/// [all, late(second half), early(first quarter)] mean expected success
pub fn drive(p: &mut dyn Pol, env: &Env, seed: u64) -> [f64; 3] {
    let n = env.n();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut arrivals: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); n + 100];
    let (mut s, mut c) = ([0.0; 3], [0usize; 3]);
    for i in 0..n {
        // rewards landing now (before decision i)
        let arr = std::mem::take(&mut arrivals[i]);
        for (j, a, r) in arr {
            match env.fb {
                Feedback::Anon => p.rew_anon(env, i, r),
                _ => p.rew_id(env, j, a, r),
            }
        }
        let a = p.decide(env, i, &mut rng);
        let pr = env.probs[i][a];
        let r = if env.u[i] < pr { 1.0 } else { 0.0 };
        if env.delay[i] == 0 {
            p.rew_id(env, i, a, r);
        } else if i + env.delay[i] < arrivals.len() {
            arrivals[i + env.delay[i]].push((i, a, r));
        }
        s[0] += pr;
        c[0] += 1;
        if i >= n / 2 {
            s[1] += pr;
            c[1] += 1;
        }
        if i < n / 4 {
            s[2] += pr;
            c[2] += 1;
        }
    }
    [s[0] / c[0] as f64, s[1] / c[1] as f64, s[2] / c[2] as f64]
}

// ------------------------------------------------------------------ specs

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ev {
    Set,
    Mean,
    Global,
    FailSet,
    FailMean,
}
fn ev<R>(e: Ev) -> Evidence<R> {
    match e {
        Ev::Set => Evidence::FeatureSet,
        Ev::Mean => Evidence::MeanFeature,
        Ev::Global => Evidence::Global,
        Ev::FailSet => Evidence::SetFailures,
        Ev::FailMean => Evidence::FeatureFailures,
    }
}

pub fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spec {
    Keyword,
    Eps(f64, f64, Attrib),
    Pts(f64, Attrib),
    HebR(Attrib),
    WarmHebR(f64, Ev, Attrib),
    Guarded(Attrib),
    Snn(LCfg),
    T(SpecT),
}

pub fn build(spec: &Spec, env: &Env, seed: u64) -> Box<dyn Pol> {
    let rp = |r: Box<dyn Router>, a: Attrib| -> Box<dyn Pol> { Box::new(RouterPol::new(r, a, seed)) };
    match *spec {
        Spec::Keyword => rp(Box::new(KeywordRouter::new(env.kmap.clone(), NA)), Attrib::None),
        Spec::Eps(g, e, a) => rp(Box::new(EpsGreedyRouter::new(NA, g, e)), a),
        Spec::Pts(g, a) => rp(Box::new(PooledThompsonRouter::new(NA, g)), a),
        Spec::HebR(a) => rp(Box::new(HebbianRouter::new(NA, heb_robust_cfg())), a),
        Spec::WarmHebR(k, e, a) => rp(
            Box::new(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), env.kmap.clone(), NA, k).with_evidence(ev(e))),
            a,
        ),
        Spec::Guarded(a) => rp(Box::new(recommended_router(env.kmap.clone(), NA)), a),
        Spec::Snn(c) => Box::new(SnnPol { l: Learner::new(c, env.kmap.clone()) }),
        Spec::T(t) => build_t(&t, env, seed),
    }
}

// ------------------------------------------------------------------ matched-effort baseline specs (v2)

use neural_mesh::hebbian::AggKind;
use neural_mesh::prior::{GuardConfig, Guarded};

/// Copy-able subset of HebbianConfig that is random-searched for the matched-effort baselines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HebP {
    pub eta: f64,
    pub eps: f64,
    pub temp: f64,
    pub decay: f64,
    pub ba: f64,
    pub cap: f64,
    pub ucb: f64,
    pub med: bool,
}
impl HebP {
    pub fn cfg(&self) -> HebbianConfig {
        HebbianConfig {
            eta: self.eta,
            epsilon: self.eps,
            temperature: self.temp,
            decay: self.decay,
            baseline_alpha: self.ba,
            row_cap: self.cap,
            ucb_c: self.ucb,
            agg: if self.med { AggKind::Median } else { AggKind::Sum },
            ..HebbianConfig::default()
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GuardP {
    pub tau: f64,
    pub z: f64,
    pub kw_a: f64,
    pub disc: f64,
}
pub fn build_t(spec: &SpecT, env: &Env, seed: u64) -> Box<dyn Pol> {
    let rp = |r: Box<dyn Router>, a: Attrib| -> Box<dyn Pol> { Box::new(RouterPol::new(r, a, seed)) };
    match *spec {
        SpecT::Heb(h, a) => rp(Box::new(HebbianRouter::new(NA, h.cfg())), a),
        SpecT::Warm(k, e, h, a) => {
            rp(Box::new(Warm::new(HebbianRouter::new(NA, h.cfg()), env.kmap.clone(), NA, k).with_evidence(ev(e))), a)
        }
        SpecT::Guard(g, h, a) => rp(
            Box::new(Guarded::new(
                HebbianRouter::new(NA, h.cfg()),
                env.kmap.clone(),
                NA,
                GuardConfig { tau: g.tau, z: g.z, kw_a: g.kw_a, kw_b: 1.0, discount: g.disc, challenge: true },
            )),
            a,
        ),
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpecT {
    Heb(HebP, Attrib),
    Warm(f64, Ev, HebP, Attrib),
    Guard(GuardP, HebP, Attrib),
}
