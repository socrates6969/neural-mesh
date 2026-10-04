//! Composable router add-ons (ROUTER-ADDON / d4). All opt-in; every add-on has an
//! "off" setting that is bit-identical to the bare router (tested).
//!
//! 1. [`Shaper`] / [`Shaped`]: turns a weak (tool-ok bit) feedback channel plus
//!    per-task response proxies (retry count, latency, length, ...) into a GRADED
//!    reward in [0,1]. Uses only signals available on every task. Per channel:
//!    log transform, optional per-feature and per-agent debiasing (running means),
//!    z-normalisation; channels are combined with fixed weights or with weights
//!    learned WITHOUT labels from a one-factor model of the channel correlations
//!    (informative channels share the latent quality, noise channels get weight 0).
//!    The graded value is `center + gain * z_q` (clamped); a hard tool failure gives
//!    `fail_reward`.
//! 2. [`Abstain`]: confidence(task) + abstain-to-fallback (keyword) around any router.
//! 3. [`ConfCalibrator`]: tiny per-feature additive bias + global affine layer that
//!    calibrates a confidence against the rewards actually observed.
//!
//! std only.

use std::collections::HashMap;

use crate::prior::KeywordPrior;
use crate::router::{Rng, Router, Task};

// ======================================================================================
// 1. Reward shaper

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transform {
    Raw,
    Ln,
    Log1p,
}

impl Transform {
    fn apply(self, v: f64) -> f64 {
        match self {
            Transform::Raw => v,
            Transform::Ln => v.max(1e-9).ln(),
            Transform::Log1p => v.max(0.0).ln_1p(),
        }
    }
}

/// One response proxy.
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub name: String,
    pub transform: Transform,
    /// +1: higher value = better quality, -1: higher = worse, 0: sign learned from the
    /// correlation with the first channel of known direction (needs >= 2 known/learned).
    pub dir: i8,
    /// subtract the running mean of the task's features (removes per-topic baselines;
    /// also removes absolute quality information, only within-topic contrasts survive)
    pub debias_feature: bool,
    /// strength in [0,1] of per-agent mean removal (agent speed/style confounds; also
    /// removes true agent-level quality differences)
    pub debias_agent: f64,
    /// fixed combination weight (used when `ShaperConfig::factor` is off or too few channels)
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShaperConfig {
    /// `false` => passthrough: reward = ok ? 1 : 0 (identical to the raw hook bit)
    pub enabled: bool,
    pub channels: Vec<Channel>,
    /// reward of a "tool ok" task with an average proxy score
    pub center: f64,
    /// reward change per quality standard deviation
    pub gain: f64,
    pub fail_reward: f64,
    /// learn channel weights from a one-factor model (needs >= 3 channels present)
    pub factor: bool,
    /// tasks before the factor weights replace the fixed ones
    pub warmup: usize,
    /// EMA floor (window ~ 1/alpha_min tasks) of all running statistics
    pub alpha_min: f64,
}

impl ShaperConfig {
    pub fn off() -> Self {
        ShaperConfig { enabled: false, channels: vec![], center: 0.75, gain: 0.15, fail_reward: 0.0, factor: false, warmup: 100, alpha_min: 0.005 }
    }
}

#[derive(Clone, Debug, Default)]
struct Run {
    mean: f64,
    m2: f64,
    n: f64,
}
impl Run {
    fn push(&mut self, x: f64, amin: f64) {
        self.n += 1.0;
        let a = (1.0 / self.n).max(amin);
        let d = x - self.mean;
        self.mean += a * d;
        self.m2 += a * (d * d - self.m2);
    }
    fn sd(&self) -> f64 {
        self.m2.max(1e-12).sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct Shaper {
    pub cfg: ShaperConfig,
    glob: Vec<Run>,
    resid: Vec<Run>,
    feat: Vec<HashMap<String, Run>>,
    agent: Vec<Vec<Run>>,
    cov: Vec<Vec<f64>>,
    cov_n: f64,
    q: Run,
    signs: Vec<f64>,
    /// last combination weights (diagnostics)
    pub weights: Vec<f64>,
    pub n_obs: usize,
}

impl Shaper {
    pub fn new(cfg: ShaperConfig) -> Self {
        let k = cfg.channels.len();
        let weights = cfg.channels.iter().map(|c| c.weight).collect();
        let signs = cfg.channels.iter().map(|c| if c.dir == 0 { 1.0 } else { c.dir as f64 }).collect();
        Shaper {
            cfg,
            glob: vec![Run::default(); k],
            resid: vec![Run::default(); k],
            feat: vec![HashMap::new(); k],
            agent: vec![vec![]; k],
            cov: vec![vec![0.0; k]; k],
            cov_n: 0.0,
            q: Run::default(),
            signs,
            weights,
            n_obs: 0,
        }
    }

    /// Graded reward in [0,1] for one task. `vals[i]` is the raw value of channel i
    /// (None = missing). `ok` is the hook's tool-ok bit.
    pub fn shape(&mut self, task: &Task, agent: usize, ok: bool, vals: &[Option<f64>]) -> f64 {
        if !self.cfg.enabled {
            return if ok { 1.0 } else { 0.0 };
        }
        if !ok {
            return self.cfg.fail_reward.clamp(0.0, 1.0);
        }
        let k = self.cfg.channels.len();
        let amin = self.cfg.alpha_min;
        let mut zs: Vec<Option<f64>> = vec![None; k];
        for i in 0..k {
            let Some(v) = vals.get(i).copied().flatten() else { continue };
            let ch = &self.cfg.channels[i];
            let x = ch.transform.apply(v);
            let m_glob = self.glob[i].mean;
            let mut base = m_glob;
            if ch.debias_feature && !task.features.is_empty() {
                let (mut s, mut c) = (0.0, 0.0);
                for f in &task.features {
                    if let Some(r) = self.feat[i].get(f) {
                        if r.n >= 3.0 {
                            s += r.mean;
                            c += 1.0;
                        }
                    }
                }
                if c > 0.0 {
                    base = s / c;
                }
            }
            if ch.debias_agent > 0.0 {
                if let Some(r) = self.agent[i].get(agent) {
                    if r.n >= 3.0 {
                        base += ch.debias_agent * (r.mean - m_glob);
                    }
                }
            }
            let resid = x - base;
            let z = if self.resid[i].n >= 5.0 { resid / self.resid[i].sd() } else { 0.0 };
            // update running stats AFTER use
            self.glob[i].push(x, amin);
            self.resid[i].push(resid, amin);
            for f in &task.features {
                self.feat[i].entry(f.clone()).or_default().push(x, amin);
            }
            if self.agent[i].len() <= agent {
                self.agent[i].resize(agent + 1, Run::default());
            }
            self.agent[i][agent].push(x, amin);
            zs[i] = Some(z.clamp(-4.0, 4.0));
        }
        self.n_obs += 1;
        // covariance of oriented z (all channels present only)
        if zs.iter().all(|z| z.is_some()) && k >= 2 {
            let a = (1.0 / (self.cov_n + 1.0)).max(amin);
            self.cov_n += 1.0;
            for i in 0..k {
                for j in 0..k {
                    let p = zs[i].unwrap() * zs[j].unwrap();
                    self.cov[i][j] += a * (p - self.cov[i][j]);
                }
            }
        }
        self.update_weights(k);
        let (mut q, mut wn) = (0.0, 0.0);
        for i in 0..k {
            if let Some(z) = zs[i] {
                let w = self.weights[i];
                q += w * self.signs[i] * z;
                wn += w * w;
            }
        }
        if wn <= 1e-12 {
            return self.cfg.center.clamp(0.02, 0.98);
        }
        let qn = q / wn.sqrt();
        let zq = if self.q.n >= 5.0 { qn / self.q.sd().max(1e-6) } else { 0.0 };
        // normalise to unit variance of the combined score
        self.q.push(qn, amin);
        (self.cfg.center + self.cfg.gain * zq.clamp(-3.0, 3.0)).clamp(0.02, 0.98)
    }

    fn update_weights(&mut self, k: usize) {
        for i in 0..k {
            self.weights[i] = self.cfg.channels[i].weight;
        }
        if !self.cfg.factor || k < 3 || self.cov_n < self.cfg.warmup as f64 {
            // learn only signs when possible
            self.learn_signs(k);
            return;
        }
        self.learn_signs(k);
        let thr = (2.0 / self.cov_n.sqrt()).max(0.04);
        // correlation matrix of oriented z (z are ~unit variance; renormalise)
        let mut c = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                let d = (self.cov[i][i] * self.cov[j][j]).max(1e-12).sqrt();
                c[i][j] = self.cov[i][j] / d * self.signs[i] * self.signs[j];
            }
        }
        for i in 0..k {
            let (mut s, mut cnt) = (0.0, 0.0);
            for j in 0..k {
                for l in (j + 1)..k {
                    if j == i || l == i {
                        continue;
                    }
                    if c[j][l].abs() > thr && c[i][j].abs() > thr && c[i][l].abs() > thr {
                        s += (c[i][j] * c[i][l] / c[j][l]).max(0.0);
                        cnt += 1.0;
                    }
                }
            }
            let lam = if cnt > 0.0 { (s / cnt).sqrt().min(0.95) } else { 0.0 };
            self.weights[i] = lam / (1.0 - lam * lam);
        }
    }

    fn learn_signs(&mut self, k: usize) {
        // anchor = first channel with known direction
        let Some(anchor) = self.cfg.channels.iter().position(|c| c.dir != 0) else { return };
        for i in 0..k {
            let ch = &self.cfg.channels[i];
            if ch.dir != 0 {
                self.signs[i] = ch.dir as f64;
            } else if self.cov_n >= self.cfg.warmup as f64 {
                let c = self.cov[i][anchor] * self.signs[anchor];
                if c.abs() > 2.0 / self.cov_n.sqrt() {
                    self.signs[i] = c.signum();
                }
            }
        }
    }
}

/// Wraps ANY router: `observe` converts (ok bit, proxies) into a graded reward and
/// forwards it to the inner router. As a plain [`Router`] it is a pure passthrough.
pub struct Shaped<R: Router> {
    pub inner: R,
    pub shaper: Shaper,
    pub last_reward: f64,
}

impl<R: Router> Shaped<R> {
    pub fn new(inner: R, cfg: ShaperConfig) -> Self {
        Shaped { inner, shaper: Shaper::new(cfg), last_reward: 0.0 }
    }
    /// Feed one task's weak observation. Returns the reward given to the inner router.
    pub fn observe(&mut self, task: &Task, agent: usize, ok: bool, vals: &[Option<f64>]) -> f64 {
        let r = self.shaper.shape(task, agent, ok, vals);
        self.last_reward = r;
        self.inner.update(task, agent, r);
        r
    }
}

impl<R: Router> Router for Shaped<R> {
    fn name(&self) -> String {
        if self.shaper.cfg.enabled { format!("shaped({})", self.inner.name()) } else { self.inner.name() }
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        self.inner.select(task, candidates, rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.inner.update(task, agent, reward)
    }
}

// ======================================================================================
// 3. Confidence calibration layer

/// Per-feature additive bias + global affine map, trained online on (confidence, reward).
#[derive(Clone, Debug)]
pub struct ConfCalibrator {
    pub enabled: bool,
    pub lr: f64,
    bias: HashMap<String, f64>,
    /// conf' = a * conf + b + mean_f bias
    a: f64,
    b: f64,
}

impl ConfCalibrator {
    pub fn new(enabled: bool, lr: f64) -> Self {
        ConfCalibrator { enabled, lr, bias: HashMap::new(), a: 1.0, b: 0.0 }
    }
    pub fn apply(&self, task: &Task, conf: f64) -> f64 {
        if !self.enabled {
            return conf;
        }
        let fb = if task.features.is_empty() {
            0.0
        } else {
            task.features.iter().map(|f| self.bias.get(f).copied().unwrap_or(0.0)).sum::<f64>() / task.features.len() as f64
        };
        (self.a * conf + self.b + fb).clamp(0.0, 1.0)
    }
    /// `conf` = uncalibrated confidence issued for this task, `reward` observed.
    pub fn train(&mut self, task: &Task, conf: f64, reward: f64) {
        if !self.enabled {
            return;
        }
        let pred = self.apply(task, conf);
        let err = reward - pred;
        self.a += self.lr * 0.5 * err * (conf - 0.5);
        self.b += self.lr * 0.5 * err;
        for f in &task.features {
            let e = self.bias.entry(f.clone()).or_insert(0.0);
            *e += self.lr * err;
            *e = e.clamp(-0.5, 0.5);
        }
        self.a = self.a.clamp(0.2, 3.0);
        self.b = self.b.clamp(-0.5, 0.5);
    }
}

// ======================================================================================
// 2. Confidence / abstain

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbstainConfig {
    /// `false` => passthrough (never abstains, tracker idle)
    pub enabled: bool,
    /// abstain (use the fallback agent) when confidence < tau
    pub tau: f64,
    /// shrinkage strength / prior mean of the (feature, agent) reward tracker
    pub prior_n: f64,
    pub prior_mean: f64,
    /// width of the lower-confidence-bound penalty
    pub z: f64,
    /// discount per update of the updated cell (1 = none)
    pub discount: f64,
}

impl AbstainConfig {
    pub fn off() -> Self {
        AbstainConfig { enabled: false, tau: 0.0, prior_n: 2.0, prior_mean: 0.7, z: 0.5, discount: 1.0 }
    }
}

/// Last decision, for diagnostics / selective-risk curves.
#[derive(Clone, Copy, Debug, Default)]
pub struct AbstainInfo {
    pub conf: f64,
    pub inner_pick: usize,
    pub abstained: bool,
}

pub struct Abstain<R: Router> {
    pub inner: R,
    pub cfg: AbstainConfig,
    pub fallback: KeywordPrior,
    pub calib: ConfCalibrator,
    cells: HashMap<String, Vec<(f64, f64)>>, // (sum reward, n)
    n_agents: usize,
    pub last: AbstainInfo,
    /// conf (uncalibrated) of the last select, kept for calibrator training
    last_raw_conf: f64,
    pub n_abstain: u64,
}

impl<R: Router> Abstain<R> {
    pub fn new(inner: R, fallback: KeywordPrior, cfg: AbstainConfig, calib: ConfCalibrator) -> Self {
        let n = fallback.n_agents();
        Abstain { inner, cfg, fallback, calib, cells: HashMap::new(), n_agents: n, last: AbstainInfo::default(), last_raw_conf: 0.0, n_abstain: 0 }
    }

    fn raw_conf(&self, task: &Task, agent: usize) -> f64 {
        if task.features.is_empty() {
            return self.cfg.prior_mean;
        }
        let (mut m, mut var_n) = (0.0, 0.0);
        for f in &task.features {
            let (s, n) = self.cells.get(f).and_then(|r| r.get(agent)).copied().unwrap_or((0.0, 0.0));
            m += (s + self.cfg.prior_mean * self.cfg.prior_n) / (n + self.cfg.prior_n);
            var_n += n;
        }
        let k = task.features.len() as f64;
        let mean = m / k;
        let neff = var_n / k;
        mean - self.cfg.z * 0.5 / (neff + 1.0).sqrt()
    }

    /// Confidence in [0,1] that `agent` will do well on `task` (reward-scale; calibrated
    /// when the calibrator is on).
    pub fn confidence(&self, task: &Task, agent: usize) -> f64 {
        self.calib.apply(task, self.raw_conf(task, agent).clamp(0.0, 1.0))
    }
}

impl<R: Router> Router for Abstain<R> {
    fn name(&self) -> String {
        if self.cfg.enabled { format!("abstain({},tau={:.2})", self.inner.name(), self.cfg.tau) } else { self.inner.name() }
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let a = self.inner.select(task, candidates, rng);
        if !self.cfg.enabled {
            self.last = AbstainInfo { conf: 1.0, inner_pick: a, abstained: false };
            return a;
        }
        let raw = self.raw_conf(task, a).clamp(0.0, 1.0);
        self.last_raw_conf = raw;
        let conf = self.calib.apply(task, raw);
        let fb = self.fallback.votes(task, candidates);
        let abstain = conf < self.cfg.tau && !fb.is_empty();
        self.last = AbstainInfo { conf, inner_pick: a, abstained: abstain };
        if abstain {
            self.n_abstain += 1;
            return self.fallback.choose(task, candidates);
        }
        a
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        if self.cfg.enabled {
            let r = reward.clamp(0.0, 1.0);
            let n = self.n_agents.max(agent + 1);
            for f in &task.features {
                let row = self.cells.entry(f.clone()).or_insert_with(|| vec![(0.0, 0.0); n]);
                if row.len() < n {
                    row.resize(n, (0.0, 0.0));
                }
                if self.cfg.discount < 1.0 {
                    row[agent].0 *= self.cfg.discount;
                    row[agent].1 *= self.cfg.discount;
                }
                row[agent].0 += r;
                row[agent].1 += 1.0;
            }
            if self.calib.enabled && agent == self.last.inner_pick && !self.last.abstained {
                self.calib.train(task, self.last_raw_conf, r);
            }
        }
        self.inner.update(task, agent, reward);
    }
}

// ======================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::KeywordRouter;
    use crate::hebbian::{HebbianConfig, HebbianRouter};

    fn t(id: u64, f: &[&str]) -> Task {
        Task { id, task_type: None, features: f.iter().map(|s| s.to_string()).collect() }
    }
    fn map() -> HashMap<String, usize> {
        [("a", 0), ("b", 1), ("c", 2)].iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }
    fn hcfg() -> HebbianConfig {
        HebbianConfig { eta: 0.05, temperature: 0.02, epsilon: 0.0, ..HebbianConfig::default() }
    }
    fn chans() -> Vec<Channel> {
        vec![
            Channel { name: "retry".into(), transform: Transform::Log1p, dir: -1, debias_feature: false, debias_agent: 0.0, weight: 1.0 },
            Channel { name: "lat".into(), transform: Transform::Ln, dir: -1, debias_feature: true, debias_agent: 0.0, weight: 1.0 },
            Channel { name: "len".into(), transform: Transform::Ln, dir: 0, debias_feature: true, debias_agent: 0.0, weight: 1.0 },
        ]
    }

    #[test]
    fn shaped_off_is_passthrough() {
        let mut bare = HebbianRouter::new(3, hcfg());
        let mut sh = Shaped::new(HebbianRouter::new(3, hcfg()), ShaperConfig::off());
        let (mut r1, mut r2) = (Rng::new(5), Rng::new(5));
        let mut env = Rng::new(9);
        for i in 0..300u64 {
            let f: [&[&str]; 3] = [&["a"], &["b", "c"], &["c"]];
            let task = t(i, f[(i % 3) as usize]);
            let a1 = bare.select(&task, &[0, 1, 2], &mut r1);
            let a2 = sh.select(&task, &[0, 1, 2], &mut r2);
            assert_eq!(a1, a2);
            let ok = env.f64() < 0.8;
            bare.update(&task, a1, if ok { 1.0 } else { 0.0 });
            let r = sh.observe(&task, a2, ok, &[Some(1.0), Some(2.0), Some(3.0)]);
            assert_eq!(r, if ok { 1.0 } else { 0.0 });
        }
        assert_eq!(bare.scores(&t(0, &["a", "b"])), sh.inner.scores(&t(0, &["a", "b"])));
    }

    #[test]
    fn shaped_reward_in_range_and_tracks_quality() {
        let mut cfg = ShaperConfig::off();
        cfg.enabled = true;
        cfg.channels = chans();
        cfg.factor = true;
        let mut sh = Shaper::new(cfg);
        let mut rng = Rng::new(1);
        let (mut hi, mut lo, mut nh, mut nl) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..4000u64 {
            let good = rng.bernoulli(0.5);
            let q = if good { 0.0 } else { 1.0 };
            let task = t(i, &["a"]);
            let retry = if rng.f64() < 0.1 + 0.5 * q { 1.0 } else { 0.0 };
            let lat = (0.3 * (rng.f64() - 0.5) + 0.8 * q).exp();
            let len = (0.3 * (rng.f64() - 0.5) - 0.8 * q).exp();
            let r = sh.shape(&task, 0, true, &[Some(retry), Some(lat), Some(len)]);
            assert!((0.0..=1.0).contains(&r));
            if i > 1500 {
                if good {
                    hi += r;
                    nh += 1.0;
                } else {
                    lo += r;
                    nl += 1.0;
                }
            }
        }
        assert!(hi / nh > lo / nl + 0.1, "good {} bad {}", hi / nh, lo / nl);
        // all three channels informative => all weights > 0, length sign learned negative
        assert!(sh.weights.iter().all(|w| *w > 0.0), "{:?}", sh.weights);
    }

    #[test]
    fn noise_channels_get_no_weight() {
        let mut cfg = ShaperConfig::off();
        cfg.enabled = true;
        cfg.channels = chans();
        cfg.factor = true;
        let mut sh = Shaper::new(cfg);
        let mut rng = Rng::new(2);
        for i in 0..4000u64 {
            let task = t(i, &["a"]);
            let v = [Some(rng.f64()), Some(rng.f64() + 0.5), Some(rng.f64() + 0.5)];
            sh.shape(&task, 0, true, &v);
        }
        let wsum: f64 = sh.weights.iter().sum();
        assert!(wsum < 1.0, "weights {:?}", sh.weights);
    }

    #[test]
    fn hard_failure_gets_fail_reward() {
        let mut cfg = ShaperConfig::off();
        cfg.enabled = true;
        cfg.channels = chans();
        let mut sh = Shaper::new(cfg);
        assert_eq!(sh.shape(&t(0, &["a"]), 0, false, &[None, None, None]), 0.0);
    }

    #[test]
    fn abstain_off_is_passthrough_and_tau0_never_abstains() {
        let mut bare = HebbianRouter::new(3, hcfg());
        let kp = || KeywordPrior::new(map(), 3);
        let mut off = Abstain::new(HebbianRouter::new(3, hcfg()), kp(), AbstainConfig::off(), ConfCalibrator::new(false, 0.0));
        let mut on0 = Abstain::new(
            HebbianRouter::new(3, hcfg()),
            kp(),
            AbstainConfig { enabled: true, tau: 0.0, ..AbstainConfig::off() },
            ConfCalibrator::new(false, 0.0),
        );
        let (mut r1, mut r2, mut r3) = (Rng::new(5), Rng::new(5), Rng::new(5));
        let mut env = Rng::new(9);
        for i in 0..300u64 {
            let f: [&[&str]; 3] = [&["a"], &["b", "c"], &["c"]];
            let task = t(i, f[(i % 3) as usize]);
            let a1 = bare.select(&task, &[0, 1, 2], &mut r1);
            let a2 = off.select(&task, &[0, 1, 2], &mut r2);
            let a3 = on0.select(&task, &[0, 1, 2], &mut r3);
            assert_eq!(a1, a2);
            assert_eq!(a1, a3);
            let rw = if env.f64() < 0.7 { 1.0 } else { 0.0 };
            bare.update(&task, a1, rw);
            off.update(&task, a2, rw);
            on0.update(&task, a3, rw);
        }
        assert_eq!(on0.n_abstain, 0);
    }

    #[test]
    fn abstain_falls_back_to_keyword_when_unconfident() {
        let kp = KeywordPrior::new(map(), 3);
        let mut ab = Abstain::new(
            HebbianRouter::new(3, hcfg()),
            kp,
            AbstainConfig { enabled: true, tau: 0.99, ..AbstainConfig::off() },
            ConfCalibrator::new(false, 0.0),
        );
        let mut kw = KeywordRouter::new(map(), 3);
        let mut rng = Rng::new(1);
        let mut rng2 = Rng::new(1);
        let task = t(0, &["b"]);
        assert_eq!(ab.select(&task, &[0, 1, 2], &mut rng), kw.select(&task, &[0, 1, 2], &mut rng2));
        assert!(ab.last.abstained);
    }

    #[test]
    fn calibrator_removes_bias() {
        let mut c = ConfCalibrator::new(true, 0.05);
        let task = t(0, &["a"]);
        for _ in 0..2000 {
            c.train(&task, 0.9, 0.5);
        }
        assert!((c.apply(&task, 0.9) - 0.5).abs() < 0.1, "{}", c.apply(&task, 0.9));
        let off = ConfCalibrator::new(false, 0.05);
        assert_eq!(off.apply(&task, 0.9), 0.9);
    }
}
