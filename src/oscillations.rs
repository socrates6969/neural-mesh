//! Brain-oscillation inspired, strictly opt-in wrappers around the warm-started
//! Hebbian router (`Warm<HebbianRouter>`). Nothing here changes any existing
//! type; with every mechanism off [`OscRouter`] behaves exactly like the wrapped
//! `Warm<HebbianRouter>` (tested).
//!
//! Mechanisms (each independently switchable, each with a "control" timing mode
//! that keeps the long-run average but destroys the oscillatory structure):
//!
//! * **Alpha** ([`AlphaCfg`]): periodic inhibition of exploration. A square wave
//!   with period `P` tasks; in the *inhibited* part of the cycle the inner router
//!   runs at its base (greedy) epsilon/temperature, in the *open* part it runs at
//!   `eps_open` / `temp_open`. Controls: `Const` (same mean epsilon & temperature,
//!   no oscillation) and `Random` (phase-randomised: open/closed drawn iid per
//!   task with the same duty cycle).
//! * **Gamma** ([`GateCfg`], [`BatchCfg`]): (i) coherence-gated recruitment: an
//!   agent is a candidate for a task only if its recent-success EMA on the task's
//!   features is above a threshold, except during periodic "gamma bursts" when the
//!   gate opens so benched agents can be re-tried; (ii) weight consolidation: the
//!   Hebbian update is applied in batches every `G` tasks instead of online.
//! * **Theta-gamma** ([`ThetaCfg`]): working-memory slots hold `k` items
//!   (task, agent, reward) per theta cycle (the most surprising, or the most
//!   recent); at each theta boundary they are replayed into the Hebbian router.
//! * **Beta** ([`BetaCfg`]): status-quo bias: per feature the currently working
//!   agent is kept with probability `stick_p` until a reward-prediction-error
//!   burst (|reward - EMA| > threshold) releases it. Controls: never release,
//!   release at random with a matched rate.
//!
//! The wrapper keeps its own RNG for all timing decisions so the caller's RNG
//! stream seen by the inner router is the same as without the wrapper.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::hebbian::HebbianRouter;
use crate::prior::Warm;
use crate::router::{Rng, Router, Task};

/// How a periodic event is scheduled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Timing {
    /// every `period` tasks (the oscillation).
    #[default]
    Periodic,
    /// independently with probability `1/period` per task (phase-randomised control).
    Random,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AlphaMode {
    /// square-wave oscillation.
    #[default]
    Square,
    /// constant exploration level = mean of the square wave (control).
    Const,
    /// open/closed iid per task with the same duty cycle (phase-randomised control).
    Random,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AlphaCfg {
    pub period: usize,
    /// fraction of each cycle that is inhibited (exploration at the base level).
    pub inhibit_frac: f64,
    /// epsilon in the open part of the cycle.
    pub eps_open: f64,
    /// softmax temperature in the open part (<= 0: keep the base temperature).
    pub temp_open: f64,
    pub mode: AlphaMode,
}

impl AlphaCfg {
    /// (epsilon, temperature) for task number `t`.
    pub fn level(&self, t: u64, rng: &mut Rng, base_eps: f64, base_temp: f64) -> (f64, f64) {
        let f = self.inhibit_frac.clamp(0.0, 1.0);
        let t_open = if self.temp_open > 0.0 { self.temp_open } else { base_temp };
        let open = match self.mode {
            AlphaMode::Const => {
                return (f * base_eps + (1.0 - f) * self.eps_open, f * base_temp + (1.0 - f) * t_open);
            }
            AlphaMode::Square => self.is_open(t),
            AlphaMode::Random => rng.f64() >= f,
        };
        if open { (self.eps_open, t_open) } else { (base_eps, base_temp) }
    }

    /// Square wave: the first `round(P * inhibit_frac)` tasks of each cycle are
    /// inhibited, the rest open.
    pub fn is_open(&self, t: u64) -> bool {
        let p = self.period.max(1) as u64;
        let inhibited = (self.inhibit_frac.clamp(0.0, 1.0) * p as f64).round() as u64;
        (t % p) >= inhibited
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateCfg {
    /// EMA rate of the per-(feature, agent) coherence (success) estimate.
    pub ema_alpha: f64,
    /// agent recruited only if mean coherence over the task's features >= thr.
    pub thr: f64,
    /// every `burst_period` tasks the gate is open for `burst_len` tasks.
    pub burst_period: usize,
    pub burst_len: usize,
    pub timing: Timing,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BatchCfg {
    /// consolidate (apply the Hebbian updates) every `every` tasks; 1 = online.
    pub every: usize,
    pub timing: Timing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReplayPick {
    /// slots keep the k items with the largest |reward - baseline|.
    #[default]
    Surprise,
    /// slots keep the k most recent items.
    Recent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThetaCfg {
    pub period: usize,
    pub k: usize,
    pub pick: ReplayPick,
    pub timing: Timing,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum Release {
    /// release on a reward-prediction-error burst.
    #[default]
    Surprise,
    /// never release (control).
    Never,
    /// release each update with this probability (control, matched rate).
    Random(f64),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BetaCfg {
    pub stick_p: f64,
    pub ema_alpha: f64,
    pub burst_thr: f64,
    pub release: Release,
}

/// All opt-in mechanisms; the default has everything off.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct OscConfig {
    #[serde(default)]
    pub alpha: Option<AlphaCfg>,
    #[serde(default)]
    pub gate: Option<GateCfg>,
    #[serde(default)]
    pub batch: Option<BatchCfg>,
    #[serde(default)]
    pub theta: Option<ThetaCfg>,
    #[serde(default)]
    pub beta: Option<BetaCfg>,
}

impl OscConfig {
    pub fn is_off(&self) -> bool {
        self.alpha.is_none() && self.gate.is_none() && self.batch.is_none() && self.theta.is_none() && self.beta.is_none()
    }
}

struct Slot {
    task: Task,
    agent: usize,
    reward: f64,
    salience: f64,
}

/// The oscillation wrapper. Implements [`Router`].
pub struct OscRouter {
    pub warm: Warm<HebbianRouter>,
    pub cfg: OscConfig,
    rng: Rng,
    t: u64,
    n_upd: u64,
    base_eps: f64,
    base_temp: f64,
    n_agents: usize,
    coh: HashMap<String, Vec<f64>>,
    pending: Vec<(Task, usize, f64)>,
    slots: Vec<Slot>,
    cur: HashMap<String, usize>,
    ema_r: HashMap<String, f64>,
    /// number of release events (beta) / replayed items (theta) / flushes (batch), for diagnostics.
    pub bursts: u64,
    pub replayed: u64,
    pub flushes: u64,
}

impl OscRouter {
    pub fn new(warm: Warm<HebbianRouter>, cfg: OscConfig, seed: u64) -> Self {
        let base_eps = warm.inner.cfg.epsilon;
        let base_temp = warm.inner.cfg.temperature;
        let n_agents = warm.inner.n_agents();
        OscRouter {
            warm,
            cfg,
            rng: Rng::new(seed.wrapping_mul(0xD1B5_4A32_D192_ED03).wrapping_add(0x0C11)),
            t: 0,
            n_upd: 0,
            base_eps,
            base_temp,
            n_agents,
            coh: HashMap::new(),
            pending: vec![],
            slots: vec![],
            cur: HashMap::new(),
            ema_r: HashMap::new(),
            bursts: 0,
            replayed: 0,
            flushes: 0,
        }
    }

    /// Fraction of updates that triggered a beta release (diagnostic).
    pub fn burst_rate(&self) -> f64 {
        if self.n_upd == 0 { 0.0 } else { self.bursts as f64 / self.n_upd as f64 }
    }

    fn event(&mut self, timing: Timing, period: usize, counter: u64) -> bool {
        let p = period.max(1) as u64;
        match timing {
            Timing::Periodic => counter % p == 0,
            Timing::Random => self.rng.f64() < 1.0 / p as f64,
        }
    }

    /// Is the gamma gate open (burst) for the current task?
    fn gate_open(&mut self, g: &GateCfg) -> bool {
        let p = g.burst_period.max(1) as u64;
        match g.timing {
            Timing::Periodic => (self.t % p) < g.burst_len as u64,
            Timing::Random => self.rng.f64() < g.burst_len as f64 / p as f64,
        }
    }

    fn coherence(&self, task: &Task, agent: usize) -> f64 {
        if task.features.is_empty() {
            return 0.5;
        }
        let s: f64 = task.features.iter().map(|f| self.coh.get(f).map_or(0.5, |v| v[agent])).sum();
        s / task.features.len() as f64
    }

    fn status_quo(&self, task: &Task, cands: &[usize]) -> Option<usize> {
        let mut votes: Vec<(usize, usize)> = vec![];
        for f in &task.features {
            if let Some(&a) = self.cur.get(f)
                && cands.contains(&a)
            {
                match votes.iter_mut().find(|v| v.0 == a) {
                    Some(v) => v.1 += 1,
                    None => votes.push((a, 1)),
                }
            }
        }
        votes.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        votes.first().map(|v| v.0)
    }
}

impl Router for OscRouter {
    fn name(&self) -> String {
        let mut s = String::from("osc[");
        if let Some(a) = &self.cfg.alpha {
            s.push_str(&format!("alpha:{:?}:P{}", a.mode, a.period));
        }
        if let Some(g) = &self.cfg.gate {
            s.push_str(&format!(" gate:{:?}:thr{}", g.timing, g.thr));
        }
        if let Some(b) = &self.cfg.batch {
            s.push_str(&format!(" batch:{:?}:G{}", b.timing, b.every));
        }
        if let Some(t) = &self.cfg.theta {
            s.push_str(&format!(" theta:{:?}:P{}k{}", t.timing, t.period, t.k));
        }
        if let Some(b) = &self.cfg.beta {
            s.push_str(&format!(" beta:{:?}:p{}", b.release, b.stick_p));
        }
        s.push_str("](");
        s.push_str(&self.warm.name());
        s.push(')');
        s
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "OscRouter::select: no candidates");
        let t = self.t;
        self.t += 1;
        // alpha: modulate exploration of the inner router
        if let Some(a) = self.cfg.alpha.clone() {
            let (e, tmp) = a.level(t, &mut self.rng, self.base_eps, self.base_temp);
            self.warm.inner.cfg.epsilon = e;
            self.warm.inner.cfg.temperature = tmp;
        }
        // gamma gate: only coherent agents are recruited (except during bursts)
        let mut cands: Vec<usize> = candidates.to_vec();
        if let Some(g) = self.cfg.gate.clone()
            && !self.gate_open(&g)
        {
            let coherent: Vec<usize> =
                candidates.iter().copied().filter(|&a| a < self.n_agents && self.coherence(task, a) >= g.thr).collect();
            if !coherent.is_empty() {
                cands = coherent;
            }
        }
        // beta: status-quo bias
        if let Some(b) = self.cfg.beta.clone()
            && let Some(q) = self.status_quo(task, &cands)
            && self.rng.f64() < b.stick_p
        {
            return q;
        }
        self.warm.select(task, &cands, rng)
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = if reward.is_finite() { reward.clamp(0.0, 1.0) } else { return };
        self.n_upd += 1;
        // gate bookkeeping
        if let Some(g) = &self.cfg.gate {
            let n = self.n_agents.max(agent + 1);
            for f in &task.features {
                let v = self.coh.entry(f.clone()).or_insert_with(|| vec![0.5; n]);
                if v.len() < n {
                    v.resize(n, 0.5);
                }
                v[agent] += g.ema_alpha * (r - v[agent]);
            }
            self.n_agents = n;
        }
        // beta bookkeeping
        if let Some(b) = self.cfg.beta.clone() {
            let mut surprise = 0.0f64;
            for f in &task.features {
                let e = self.ema_r.get(f).copied().unwrap_or(0.5);
                surprise = surprise.max((r - e).abs());
            }
            let release = match b.release {
                Release::Surprise => surprise > b.burst_thr,
                Release::Never => false,
                Release::Random(p) => self.rng.f64() < p,
            };
            for f in &task.features {
                let e = self.ema_r.entry(f.clone()).or_insert(0.5);
                *e += b.ema_alpha * (r - *e);
                if release {
                    self.cur.remove(f);
                } else if r >= 0.5 {
                    self.cur.insert(f.clone(), agent);
                }
            }
            if release {
                self.bursts += 1;
            }
        }
        // theta: offer the item to the working-memory slots (before learning from it)
        if let Some(th) = self.cfg.theta.clone() {
            let base = if task.features.is_empty() {
                0.5
            } else {
                task.features.iter().map(|f| self.warm.inner.baseline(f)).sum::<f64>() / task.features.len() as f64
            };
            let item = Slot { task: task.clone(), agent, reward: r, salience: (r - base).abs() };
            if self.slots.len() < th.k {
                self.slots.push(item);
            } else if th.k > 0 {
                match th.pick {
                    ReplayPick::Recent => {
                        self.slots.remove(0);
                        self.slots.push(item);
                    }
                    ReplayPick::Surprise => {
                        let (mi, ms) = self
                            .slots
                            .iter()
                            .enumerate()
                            .map(|(i, s)| (i, s.salience))
                            .fold((0, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
                        if item.salience > ms {
                            self.slots[mi] = item;
                        }
                    }
                }
            }
        }
        // gamma consolidation: batch or online
        if let Some(bc) = self.cfg.batch.clone() {
            self.pending.push((task.clone(), agent, r));
            if self.event(bc.timing, bc.every, self.n_upd) {
                for (tk, a, rw) in std::mem::take(&mut self.pending) {
                    self.warm.update(&tk, a, rw);
                }
                self.flushes += 1;
            }
        } else {
            self.warm.update(task, agent, r);
        }
        // theta boundary: replay working memory into the Hebbian weights
        if let Some(th) = self.cfg.theta.clone()
            && self.event(th.timing, th.period, self.n_upd)
        {
            for s in std::mem::take(&mut self.slots) {
                self.warm.inner.update(&s.task, s.agent, s.reward);
                self.replayed += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hebbian::HebbianConfig;

    fn heb() -> HebbianRouter {
        HebbianRouter::new(
            4,
            HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() },
        )
    }
    fn warm(k: f64) -> Warm<HebbianRouter> {
        let mut m = HashMap::new();
        m.insert("a".to_string(), 0usize);
        m.insert("b".to_string(), 1usize);
        Warm::new(heb(), m, 4, k)
    }
    fn task(id: u64, f: &[&str]) -> Task {
        Task { id, task_type: None, features: f.iter().map(|s| s.to_string()).collect() }
    }
    const C: [usize; 4] = [0, 1, 2, 3];

    #[test]
    fn off_equals_plain_warm() {
        let mut plain = warm(5.0);
        let mut osc = OscRouter::new(warm(5.0), OscConfig::default(), 1);
        assert!(osc.cfg.is_off());
        let (mut r1, mut r2) = (Rng::new(9), Rng::new(9));
        for i in 0..300u64 {
            let t = task(i, if i % 2 == 0 { &["a", "x"] } else { &["b"] });
            let (p, o) = (plain.select(&t, &C, &mut r1), osc.select(&t, &C, &mut r2));
            assert_eq!(p, o, "task {i}");
            let rew = if (i * 7) % 5 < 3 { 1.0 } else { 0.0 };
            plain.update(&t, p, rew);
            osc.update(&t, o, rew);
        }
        assert_eq!(plain.inner, osc.warm.inner);
    }

    #[test]
    fn alpha_square_schedule_and_controls() {
        let a = AlphaCfg { period: 10, inhibit_frac: 0.7, eps_open: 0.3, temp_open: 0.05, mode: AlphaMode::Square };
        let mut rng = Rng::new(1);
        let opens: Vec<bool> = (0..20).map(|t| a.is_open(t)).collect();
        assert_eq!(opens.iter().filter(|&&o| o).count(), 6); // 3 open per cycle of 10
        assert!(!a.is_open(0) && !a.is_open(6) && a.is_open(7) && a.is_open(9) && !a.is_open(10));
        assert_eq!(a.level(0, &mut rng, 0.0, 0.002), (0.0, 0.002));
        assert_eq!(a.level(8, &mut rng, 0.0, 0.002), (0.3, 0.05));
        // const control: same mean epsilon / temperature as the square wave
        let c = AlphaCfg { mode: AlphaMode::Const, ..a.clone() };
        let (mut me, mut mt) = (0.0, 0.0);
        for t in 0..1000u64 {
            let (e, tp) = a.level(t, &mut rng, 0.0, 0.002);
            me += e;
            mt += tp;
        }
        let (ce, ct) = c.level(5, &mut rng, 0.0, 0.002);
        assert!((me / 1000.0 - ce).abs() < 1e-9 && (mt / 1000.0 - ct).abs() < 1e-9);
        // random control: same open fraction on average, but not periodic
        let r = AlphaCfg { mode: AlphaMode::Random, ..a.clone() };
        let n = 20000;
        let open = (0..n).filter(|&t| r.level(t, &mut rng, 0.0, 0.002).0 > 0.0).count() as f64 / n as f64;
        assert!((open - 0.3).abs() < 0.02);
    }

    #[test]
    fn alpha_changes_inner_exploration() {
        let cfg = OscConfig {
            alpha: Some(AlphaCfg { period: 4, inhibit_frac: 0.5, eps_open: 1.0, temp_open: 0.0, mode: AlphaMode::Square }),
            ..OscConfig::default()
        };
        let mut osc = OscRouter::new(warm(0.0), cfg, 1); // k=0: always inner
        let mut rng = Rng::new(3);
        let t = task(0, &["a"]);
        osc.select(&t, &C, &mut rng); // t=0 inhibited
        assert_eq!(osc.warm.inner.cfg.epsilon, 0.0);
        osc.select(&t, &C, &mut rng);
        osc.select(&t, &C, &mut rng); // t=2 open
        assert_eq!(osc.warm.inner.cfg.epsilon, 1.0);
    }

    #[test]
    fn gate_benches_failing_agent_until_burst() {
        let g = GateCfg { ema_alpha: 0.5, thr: 0.3, burst_period: 50, burst_len: 1, timing: Timing::Periodic };
        let mut osc = OscRouter::new(warm(0.0), OscConfig { gate: Some(g), ..OscConfig::default() }, 1);
        let t = task(0, &["a"]);
        for _ in 0..6 {
            osc.update(&t, 2, 0.0);
        }
        assert!(osc.coherence(&t, 2) < 0.3);
        let mut rng = Rng::new(1);
        osc.t = 5; // not a burst
        for _ in 0..30 {
            let a = osc.select(&t, &[2, 3], &mut rng);
            assert_eq!(a, 3, "benched agent 2 must not be recruited");
            osc.t = 5;
        }
        // all candidates benched -> fall back to all of them
        assert!(osc.select(&t, &[2], &mut rng) == 2);
        // burst: gate open, agent 2 recruitable again
        let mut seen2 = false;
        for _ in 0..200 {
            osc.t = 50;
            if osc.select(&t, &[2, 3], &mut rng) == 2 {
                seen2 = true;
            }
        }
        assert!(seen2 || osc.warm.inner.weight("a", 2) <= osc.warm.inner.weight("a", 3));
    }

    #[test]
    fn batch_defers_updates_to_the_boundary() {
        let b = BatchCfg { every: 5, timing: Timing::Periodic };
        let mut osc = OscRouter::new(warm(1.0), OscConfig { batch: Some(b), ..OscConfig::default() }, 1);
        let t = task(0, &["a"]);
        for i in 1..=12u64 {
            osc.update(&t, 0, 1.0);
            assert_eq!(osc.warm.inner.updates(), (i / 5) * 5, "after {i} updates");
        }
        assert_eq!(osc.flushes, 2);
        // every = 1 is online
        let mut on = OscRouter::new(warm(1.0), OscConfig { batch: Some(BatchCfg { every: 1, timing: Timing::Periodic }), ..OscConfig::default() }, 1);
        let mut plain = warm(1.0);
        for i in 0..20u64 {
            on.update(&t, 0, (i % 2) as f64);
            plain.update(&t, 0, (i % 2) as f64);
        }
        assert_eq!(on.warm.inner, plain.inner);
    }

    #[test]
    fn theta_replays_k_items_per_cycle() {
        let th = ThetaCfg { period: 6, k: 3, pick: ReplayPick::Recent, timing: Timing::Periodic };
        let mut osc = OscRouter::new(warm(1.0), OscConfig { theta: Some(th), ..OscConfig::default() }, 1);
        let t = task(0, &["a"]);
        for _ in 0..5 {
            osc.update(&t, 0, 1.0);
        }
        assert_eq!(osc.warm.inner.updates(), 5);
        osc.update(&t, 0, 1.0); // 6th -> boundary
        assert_eq!(osc.warm.inner.updates(), 6 + 3);
        assert_eq!(osc.replayed, 3);
        for _ in 0..6 {
            osc.update(&t, 0, 1.0);
        }
        assert_eq!(osc.warm.inner.updates(), 12 + 6);
        // replay changes the learned weights relative to no replay
        let mut plain = warm(1.0);
        for _ in 0..12 {
            plain.update(&t, 0, 1.0);
        }
        assert!((osc.warm.inner.weight("a", 0) - plain.inner.weight("a", 0)).abs() > 1e-9);
    }

    #[test]
    fn theta_surprise_keeps_most_surprising_items() {
        let th = ThetaCfg { period: 1000, k: 2, pick: ReplayPick::Surprise, timing: Timing::Periodic };
        let mut osc = OscRouter::new(warm(1.0), OscConfig { theta: Some(th), ..OscConfig::default() }, 1);
        let t = task(0, &["a"]);
        // baseline ~0.5 initially; reward 0.5 -> salience 0, rewards 1.0/0.0 -> 0.5
        osc.update(&t, 0, 0.5);
        osc.update(&t, 0, 0.5);
        osc.update(&t, 1, 0.0);
        osc.update(&t, 2, 1.0);
        let mut agents: Vec<usize> = osc.slots.iter().map(|s| s.agent).collect();
        agents.sort();
        assert_eq!(agents.len(), 2);
        assert!(agents.contains(&1) || agents.contains(&2));
    }

    #[test]
    fn beta_sticks_then_releases_on_surprise() {
        let b = BetaCfg { stick_p: 1.0, ema_alpha: 0.2, burst_thr: 0.6, release: Release::Surprise };
        let mut osc = OscRouter::new(warm(0.0), OscConfig { beta: Some(b), ..OscConfig::default() }, 1);
        let t = task(0, &["a"]);
        let mut rng = Rng::new(2);
        for _ in 0..20 {
            osc.update(&t, 2, 1.0); // agent 2 works; ema climbs to ~1
        }
        for _ in 0..10 {
            assert_eq!(osc.select(&t, &C, &mut rng), 2, "sticks to the status quo");
        }
        let before = osc.bursts;
        osc.update(&t, 2, 0.0); // surprise: |0 - ~0.99| > 0.6
        assert!(osc.bursts > before);
        assert!(osc.status_quo(&t, &C).is_none(), "released");
        // never-release control keeps it
        let bn = BetaCfg { release: Release::Never, ..osc.cfg.beta.clone().unwrap() };
        let mut o2 = OscRouter::new(warm(0.0), OscConfig { beta: Some(bn), ..OscConfig::default() }, 1);
        for _ in 0..20 {
            o2.update(&t, 2, 1.0);
        }
        o2.update(&t, 2, 0.0);
        assert_eq!(o2.status_quo(&t, &C), Some(2));
    }

    #[test]
    fn deterministic_and_serde_default() {
        let cfg = OscConfig {
            alpha: Some(AlphaCfg { period: 20, inhibit_frac: 0.8, eps_open: 0.2, temp_open: 0.03, mode: AlphaMode::Random }),
            theta: Some(ThetaCfg { period: 10, k: 2, pick: ReplayPick::Surprise, timing: Timing::Random }),
            ..OscConfig::default()
        };
        let run = || {
            let mut o = OscRouter::new(warm(3.0), cfg.clone(), 7);
            let mut rng = Rng::new(5);
            let mut out = vec![];
            for i in 0..200u64 {
                let t = task(i, if i % 3 == 0 { &["a"] } else { &["b", "c"] });
                let a = o.select(&t, &C, &mut rng);
                out.push(a);
                o.update(&t, a, if a == 1 { 1.0 } else { 0.0 });
            }
            out
        };
        assert_eq!(run(), run());
        let js = serde_json::to_string(&cfg).unwrap();
        let back: OscConfig = serde_json::from_str(&js).unwrap();
        assert_eq!(back, cfg);
        let empty: OscConfig = serde_json::from_str("{}").unwrap();
        assert!(empty.is_off());
    }
}
