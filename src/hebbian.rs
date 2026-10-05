//! Reward-modulated three-factor Hebbian router.
//!
//! Implemented independently from the *published algorithm description* only
//! (see DESIGN.md); no third-party code was consulted or copied.
//!
//! Model
//! -----
//! * Synapses `w[feature][agent]` in `[0, 1]`. Active features of a task are
//!   `task.features` plus `"type:<task_type>"` when a task type is present
//!   (prefixed so a type never collides with a keyword of the same name).
//!   Duplicate features are counted once.
//! * `score(agent) = sum_{f active} w[f][agent]`.
//! * `select`: epsilon-greedy over a softmax of `score / temperature`
//!   restricted to the candidate set.
//! * `update` (three factors: pre = feature active (1.0), post = agent was the
//!   one chosen, modulator = `reward - baseline`): for every active feature row
//!     - every weight in the row decays: `w *= 1 - decay`
//!     - the chosen agent gets `delta = eta * pre * (reward - baseline_f)`
//!     - weights are clamped to `[0, 1]`
//!     - if the row sum exceeds `row_cap` the row is rescaled to sum to `row_cap`
//!     - `baseline_f <- baseline_f + baseline_alpha * (reward - baseline_f)`
//!       (the *pre-update* baseline is used in `delta`).
//!
//! Interpretation choices (documented deviations from the one-line spec):
//! * The EMA baseline is kept **per feature** rather than globally, so a
//!   feature whose best achievable success rate is low is not dragged to zero
//!   by a global baseline dominated by easy features.
//! * Decay is applied to the whole active row (not only the chosen agent), so
//!   the chosen agent is not penalised relative to its competitors simply for
//!   being used; the effect is a slow flattening of stale preferences.
//! * Unknown features start at [`INIT_WEIGHT`] for every agent (neutral: it
//!   adds the same constant to every agent's score), new baselines at
//!   [`INIT_BASELINE`].
//!
//! Optional extensions (all OFF by default; the default config reproduces the
//! classic rule bit-for-bit):
//! * `meta`: self-tuning per-row learning rate (Page-Hinkley/CUSUM drift reset with
//!   geometric annealing, or Delta-bar-delta), plus alert-driven epsilon/temperature.
//! * `weighting`: feature-reliability weighting of the score (IDF / entropy /
//!   agreement contrast).
//! * `ucb_c`: UCB-style exploration bonus for rarely tried (feature, agent) cells.
//! * `agg`: robust aggregation of the per-feature weights (median instead of sum),
//!   so a single confound keyword cannot flip a decision against the others.
//!
//! [`KalmanRouter`] (opt-in, separate type): Kalman / Rescorla-Wagner synapses.
//! Every (feature, agent) synapse carries a mean and a variance; features compete
//! to predict the outcome (delta rule on the agent's own prediction, credit
//! proportional to each synapse's uncertainty), and selection is Thompson
//! sampling over the posterior. A keyword map can seed the prior means.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::router::{Rng, Router, Task};

/// Initial synaptic weight for a (feature, agent) pair never seen before.
pub const INIT_WEIGHT: f64 = 0.1;
/// Initial reward baseline for a feature never seen before.
pub const INIT_BASELINE: f64 = 0.5;

/// Self-tuning learning-rate strategy (opt-in; `Off` = the classic fixed-eta rule).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaKind {
    /// Fixed `eta` for every row (original behaviour).
    #[default]
    Off,
    /// Per-row eta starts at `eta` (the ceiling), anneals geometrically toward
    /// `meta_eta_floor`, and is snapped back to `eta` by a two-sided CUSUM /
    /// Page-Hinkley drift detector on the row's reward surprise. A decaying
    /// per-row "alert" additionally raises epsilon / temperature.
    PageHinkley,
    /// Delta-bar-delta: per-row eta (in `[meta_eta_floor, eta]`) grows additively
    /// while successive surprises agree in sign with their running mean and
    /// shrinks multiplicatively when they do not.
    Dbd,
}

/// How per-feature weights are combined into an agent score (opt-in).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggKind {
    /// Plain (reliability-weighted) sum (original behaviour).
    #[default]
    Sum,
    /// `n_features * median` of the per-feature weights: one outlier feature
    /// (e.g. a confound keyword) cannot override a consistent majority.
    Median,
}

/// Feature-reliability weighting of the score (opt-in; `Off` = plain sum).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightKind {
    #[default]
    Off,
    /// IDF-like: features seen in few updates (rare) count more than ubiquitous ones.
    Idf,
    /// Features whose row is peaked on one agent (low entropy) count more.
    Entropy,
    /// Features whose preferred agent tends to agree with the finally-routed
    /// agent count more (running agreement EMA).
    Contrast,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HebbianConfig {
    /// Learning rate of the reward-modulated term (ceiling when `meta != Off`).
    pub eta: f64,
    /// Probability of picking a uniformly random candidate.
    pub epsilon: f64,
    /// Softmax temperature (scores are divided by this). `<= 0` means argmax.
    pub temperature: f64,
    /// Multiplicative weight decay per update of a row.
    pub decay: f64,
    /// EMA rate of the per-feature reward baseline.
    pub baseline_alpha: f64,
    /// Maximum allowed sum of a feature row; larger rows are rescaled.
    pub row_cap: f64,

    // ---- opt-in: self-tuning learning rate ------------------------------------
    pub meta: MetaKind,
    /// Lowest per-row eta reached when stable.
    pub meta_eta_floor: f64,
    /// Per-update geometric decay of (eta_row - floor) and of the alert level.
    pub meta_decay: f64,
    /// EMA rate of the slow reward mean used by the drift detector.
    pub meta_det_alpha: f64,
    /// CUSUM slack (ignored shift per update).
    pub meta_delta: f64,
    /// CUSUM alarm threshold.
    pub meta_lambda: f64,
    /// Extra epsilon at alert level 1 (scaled by the max alert over active features).
    pub meta_boost_eps: f64,
    /// Temperature multiplier `1 + boost * alert` at alert level 1.
    pub meta_boost_temp: f64,
    /// Delta-bar-delta additive eta increase.
    pub dbd_kappa: f64,
    /// Delta-bar-delta multiplicative eta decrease.
    pub dbd_phi: f64,
    /// Delta-bar-delta EMA rate of the surprise trace.
    pub dbd_theta: f64,

    // ---- opt-in: feature reliability weighting --------------------------------
    pub weighting: WeightKind,
    /// Minimum relative reliability in `[0, 1]`.
    pub weight_floor: f64,
    /// EMA rate of the `Contrast` agreement tracker.
    pub weight_alpha: f64,

    // ---- opt-in: UCB-style exploration bonus ------------------------------------
    /// Bonus `c / sqrt(1 + n_fa)` (mean over active features) added to the scores
    /// of rarely tried (feature, agent) cells. `0` disables it.
    pub ucb_c: f64,
    /// Per-row-update discount of the (feature, agent) trial counts, so stale
    /// cells regain their bonus after a while.
    pub ucb_discount: f64,

    // ---- opt-in: robust aggregation ---------------------------------------------
    /// Score aggregation over active features (`Sum` = classic).
    pub agg: AggKind,
}

impl Default for HebbianConfig {
    fn default() -> Self {
        HebbianConfig {
            eta: 0.5,
            epsilon: 0.02,
            temperature: 0.05,
            decay: 0.01,
            baseline_alpha: 0.1,
            row_cap: 1.0,
            meta: MetaKind::Off,
            meta_eta_floor: 0.02,
            meta_decay: 0.95,
            meta_det_alpha: 0.03,
            meta_delta: 0.1,
            meta_lambda: 3.0,
            meta_boost_eps: 0.05,
            meta_boost_temp: 3.0,
            dbd_kappa: 0.01,
            dbd_phi: 0.9,
            dbd_theta: 0.2,
            weighting: WeightKind::Off,
            weight_floor: 0.3,
            weight_alpha: 0.02,
            ucb_c: 0.0,
            ucb_discount: 0.995,
            agg: AggKind::Sum,
        }
    }
}

/// Per-feature-row auxiliary state for the opt-in options. Created lazily, so a
/// router with every option off never allocates any.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct RowMeta {
    /// Updates seen by this row.
    n: u64,
    /// Current learning rate of the row (meta modes).
    eta_now: f64,
    /// Alert level in `[0, 1]` (drift just detected -> 1, decays).
    alert: f64,
    det_mean: f64,
    cusum_up: f64,
    cusum_dn: f64,
    /// Delta-bar-delta surprise trace.
    e_bar: f64,
    /// EMA of "this row's preferred agent == routed agent" (Contrast weighting).
    agree: f64,
    /// Discounted trial counts per agent (UCB).
    cnt: Vec<f64>,
}

impl RowMeta {
    fn new(n_agents: usize, eta: f64) -> Self {
        RowMeta {
            n: 0,
            eta_now: eta,
            alert: 0.0,
            det_mean: 0.5,
            cusum_up: 0.0,
            cusum_dn: 0.0,
            e_bar: 0.0,
            agree: 0.5,
            cnt: vec![0.0; n_agents],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HebbianRouter {
    pub cfg: HebbianConfig,
    n_agents: usize,
    /// feature -> per-agent weights (len == n_agents).
    weights: BTreeMap<String, Vec<f64>>,
    /// feature -> EMA reward baseline.
    baselines: BTreeMap<String, f64>,
    /// Number of `update` calls seen.
    updates: u64,
    /// Auxiliary per-row state of the opt-in options (empty when all are off).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    meta: BTreeMap<String, RowMeta>,
}

/// Active features of a task: features (deduplicated, order kept) plus
/// `type:<task_type>` if present.
pub fn active_features(task: &Task) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(task.features.len() + 1);
    for f in &task.features {
        if !out.iter().any(|g| g == f) {
            out.push(f.clone());
        }
    }
    if let Some(t) = &task.task_type {
        let k = format!("type:{t}");
        if !out.contains(&k) {
            out.push(k);
        }
    }
    out
}

/// Sample an index from `logits` with a numerically stable softmax.
/// Consumes exactly one `rng.f64()`.
pub(crate) fn softmax_sample(logits: &[f64], rng: &mut Rng) -> usize {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&l| if l.is_finite() { (l - max).exp() } else { 0.0 })
        .collect();
    let total: f64 = exps.iter().sum();
    let u = rng.f64();
    if !(total.is_finite() && total > 0.0) {
        return 0;
    }
    let mut acc = 0.0;
    let target = u * total;
    for (i, e) in exps.iter().enumerate() {
        acc += e;
        if target < acc {
            return i;
        }
    }
    exps.len() - 1
}

fn argmax(xs: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best
}

impl HebbianRouter {
    pub fn new(n_agents: usize, cfg: HebbianConfig) -> Self {
        HebbianRouter {
            cfg,
            n_agents,
            weights: BTreeMap::new(),
            baselines: BTreeMap::new(),
            updates: 0,
            meta: BTreeMap::new(),
        }
    }

    pub fn n_agents(&self) -> usize {
        self.n_agents
    }

    /// B14 (opt-in): federated merge of the LEARNED STATE of several routers (e.g. one per machine, each the owner of its own state file) without replaying any journal.
    /// Per feature the weight rows are SUMMED and, if the row sum exceeds `row_cap`, rescaled to `row_cap` (the same rule `update` applies); baselines are averaged over
    /// the routers that hold the feature; `updates` are added. The config is taken from `routers[0]`; opt-in meta state (drift detector / UCB counts) is NOT merged.
    /// Not a union-by-id (merging an already-merged state double counts): recompute from the latest set of per-machine states each time.
    /// Measured (RD-LOG-B B14, simulator, held-out+confirm): equivalent to replaying the union journal within +-.01 (95% CI) in 25 of 32 cells per split and within +-.003 for the
    /// Guarded router's inner learner; worst cell -.016 (H100, plain Hebbian, 4 SPECIALISED machines, fixed world); state is constant-size (5.6 KB JSON incl. config, 32 features) vs 147 KB for the 1,000-outcome union journal (26x, growing with outcomes).
    /// Panics if `routers` is empty or the routers disagree on the number of agents.
    pub fn merge_sum(routers: &[&HebbianRouter]) -> HebbianRouter {
        assert!(!routers.is_empty(), "merge_sum needs at least one router");
        let n = routers[0].n_agents;
        assert!(routers.iter().all(|r| r.n_agents == n), "merge_sum: routers disagree on n_agents");
        let mut out = HebbianRouter::new(n, routers[0].cfg.clone());
        let mut base_cnt: BTreeMap<String, f64> = BTreeMap::new();
        for r in routers {
            for (f, row) in &r.weights {
                let e = out.weights.entry(f.clone()).or_insert_with(|| vec![0.0; n]);
                for (a, w) in row.iter().enumerate() {
                    e[a] += *w;
                }
            }
            for (f, b) in &r.baselines {
                *out.baselines.entry(f.clone()).or_insert(0.0) += *b;
                *base_cnt.entry(f.clone()).or_insert(0.0) += 1.0;
            }
            out.updates += r.updates;
        }
        let cap = out.cfg.row_cap;
        for row in out.weights.values_mut() {
            let sum: f64 = row.iter().sum();
            if cap > 0.0 && sum > cap {
                let k = cap / sum;
                row.iter_mut().for_each(|w| *w *= k);
            }
        }
        for (f, b) in out.baselines.iter_mut() {
            *b /= base_cnt[f];
        }
        out
    }

    /// Grow to at least `n` agents (new agents get [`INIT_WEIGHT`] in every row).
    pub fn ensure_agents(&mut self, n: usize) {
        if n > self.n_agents {
            for row in self.weights.values_mut() {
                row.resize(n, INIT_WEIGHT);
            }
            for m in self.meta.values_mut() {
                m.cnt.resize(n, 0.0);
            }
            self.n_agents = n;
        }
    }

    /// Weight of (feature, agent); unknown pairs report [`INIT_WEIGHT`].
    pub fn weight(&self, feature: &str, agent: usize) -> f64 {
        self.weights.get(feature).and_then(|r| r.get(agent).copied()).unwrap_or(INIT_WEIGHT)
    }

    /// Reward baseline of a feature ([`INIT_BASELINE`] if unseen).
    pub fn baseline(&self, feature: &str) -> f64 {
        self.baselines.get(feature).copied().unwrap_or(INIT_BASELINE)
    }

    /// All learned rows (feature -> per-agent weights).
    pub fn rows(&self) -> &BTreeMap<String, Vec<f64>> {
        &self.weights
    }

    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Current learning rate of a feature row (the configured `eta` when the
    /// self-tuning option is off or the row is unseen).
    pub fn row_eta(&self, feature: &str) -> f64 {
        if self.cfg.meta == MetaKind::Off {
            return self.cfg.eta;
        }
        self.meta.get(feature).map(|m| m.eta_now).unwrap_or(self.cfg.eta)
    }

    /// Alert level (0..1) of a feature row: 1 right after a detected drift.
    pub fn row_alert(&self, feature: &str) -> f64 {
        self.meta.get(feature).map(|m| m.alert).unwrap_or(0.0)
    }

    /// Un-normalised reliability multiplier of a feature (1.0 when weighting is off).
    pub fn feature_reliability(&self, feature: &str) -> f64 {
        self.raw_rel(feature)
    }

    fn raw_rel(&self, f: &str) -> f64 {
        let fl = self.cfg.weight_floor.clamp(0.0, 1.0);
        match self.cfg.weighting {
            WeightKind::Off => 1.0,
            WeightKind::Idf => {
                let n_f = self.meta.get(f).map(|m| m.n).unwrap_or(0) as f64;
                let big_n = self.updates as f64;
                let idf = ((big_n + 1.0) / (n_f + 1.0)).ln() / (big_n + 1.0).ln().max(1e-9);
                fl + (1.0 - fl) * idf.clamp(0.0, 1.0)
            }
            WeightKind::Entropy => match self.weights.get(f) {
                Some(row) if row.len() > 1 => {
                    let s: f64 = row.iter().sum();
                    if s <= 0.0 {
                        return fl;
                    }
                    let h: f64 = row.iter().map(|w| w / s).filter(|p| *p > 0.0).map(|p| -p * p.ln()).sum();
                    let hn = h / (row.len() as f64).ln();
                    fl + (1.0 - fl) * (1.0 - hn).clamp(0.0, 1.0)
                }
                _ => fl,
            },
            WeightKind::Contrast => {
                let a = self.meta.get(f).map(|m| m.agree).unwrap_or(0.5);
                fl + (1.0 - fl) * a.clamp(0.0, 1.0)
            }
        }
    }

    /// Per-feature multipliers, normalised to mean 1 so the score scale (and so the
    /// meaning of the temperature) is unchanged.
    fn rels(&self, feats: &[String]) -> Vec<f64> {
        if self.cfg.weighting == WeightKind::Off || feats.is_empty() {
            return vec![1.0; feats.len()];
        }
        let raw: Vec<f64> = feats.iter().map(|f| self.raw_rel(f)).collect();
        let s: f64 = raw.iter().sum();
        if s <= 0.0 {
            return vec![1.0; feats.len()];
        }
        let k = feats.len() as f64 / s;
        raw.into_iter().map(|r| r * k).collect()
    }

    fn score_of(&self, feats: &[String], rels: &[f64], agent: usize) -> f64 {
        match self.cfg.agg {
            AggKind::Sum => feats.iter().zip(rels).map(|(f, r)| r * self.weight(f, agent)).sum(),
            AggKind::Median => {
                let mut v: Vec<f64> = feats.iter().zip(rels).map(|(f, r)| r * self.weight(f, agent)).collect();
                if v.is_empty() {
                    return 0.0;
                }
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = v.len();
                let med = if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) };
                med * n as f64
            }
        }
    }

    /// Mean UCB bonus of `agent` over the active features.
    fn ucb_bonus(&self, feats: &[String], agent: usize) -> f64 {
        if self.cfg.ucb_c <= 0.0 || feats.is_empty() {
            return 0.0;
        }
        let s: f64 = feats
            .iter()
            .map(|f| {
                let n = self.meta.get(f).and_then(|m| m.cnt.get(agent).copied()).unwrap_or(0.0);
                1.0 / (1.0 + n).sqrt()
            })
            .sum();
        self.cfg.ucb_c * s / feats.len() as f64
    }

    /// Max alert over the active features (0 unless a meta mode is on).
    fn max_alert(&self, feats: &[String]) -> f64 {
        if self.cfg.meta == MetaKind::Off {
            return 0.0;
        }
        feats.iter().map(|f| self.row_alert(f)).fold(0.0, f64::max)
    }

    /// One exploit score per agent (`0..n_agents`): reliability-weighted sum of
    /// weights (no exploration bonus).
    pub fn scores(&self, task: &Task) -> Vec<f64> {
        let feats = active_features(task);
        let rels = self.rels(&feats);
        (0..self.n_agents).map(|a| self.score_of(&feats, &rels, a)).collect()
    }
}

impl Router for HebbianRouter {
    fn name(&self) -> String {
        "hebbian".into()
    }

    /// Panics if `candidates` is empty (contract: all candidates eligible, >= 1).
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "HebbianRouter::select: no candidates");
        let feats = active_features(task);
        let alert = self.max_alert(&feats);
        let eps = (self.cfg.epsilon + self.cfg.meta_boost_eps * alert).clamp(0.0, 1.0);
        // Always consume one draw for the epsilon test, then exactly one more.
        if rng.f64() < eps {
            return candidates[rng.below(candidates.len())];
        }
        let rels = self.rels(&feats);
        let scores: Vec<f64> = candidates
            .iter()
            .map(|&c| self.score_of(&feats, &rels, c) + self.ucb_bonus(&feats, c))
            .collect();
        if self.cfg.temperature <= 0.0 {
            let _ = rng.f64(); // keep rng consumption identical to the softmax path
            return candidates[argmax(&scores)];
        }
        let temp = self.cfg.temperature * (1.0 + self.cfg.meta_boost_temp * alert);
        let logits: Vec<f64> = scores.iter().map(|s| s / temp).collect();
        candidates[softmax_sample(&logits, rng)]
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let reward = if reward.is_finite() { reward.clamp(0.0, 1.0) } else { return };
        self.ensure_agents(agent + 1);
        self.updates += 1;
        let n = self.n_agents;
        let cfg = self.cfg.clone();
        let use_meta = cfg.meta != MetaKind::Off || cfg.weighting != WeightKind::Off || cfg.ucb_c > 0.0;
        for f in active_features(task) {
            let b = self.baselines.get(&f).copied().unwrap_or(INIT_BASELINE);
            let row = self.weights.entry(f.clone()).or_insert_with(|| vec![INIT_WEIGHT; n]);
            let e = reward - b;

            // Side state (only when an opt-in option is active).
            let mut eta_f = cfg.eta;
            if use_meta {
                let preferred = argmax(row);
                let m = self.meta.entry(f.clone()).or_insert_with(|| RowMeta::new(n, cfg.eta));
                if m.cnt.len() < n {
                    m.cnt.resize(n, 0.0);
                }
                if cfg.meta != MetaKind::Off {
                    eta_f = m.eta_now;
                }
                if m.n == 0 {
                    m.det_mean = reward;
                }
                match cfg.meta {
                    MetaKind::Off => {}
                    MetaKind::PageHinkley => {
                        let x = reward - m.det_mean;
                        m.det_mean += cfg.meta_det_alpha * x;
                        m.cusum_up = (m.cusum_up + x - cfg.meta_delta).max(0.0);
                        m.cusum_dn = (m.cusum_dn - x - cfg.meta_delta).max(0.0);
                        if m.cusum_up > cfg.meta_lambda || m.cusum_dn > cfg.meta_lambda {
                            m.eta_now = cfg.eta;
                            m.alert = 1.0;
                            m.cusum_up = 0.0;
                            m.cusum_dn = 0.0;
                            m.det_mean = reward;
                        } else {
                            m.eta_now = cfg.meta_eta_floor + (m.eta_now - cfg.meta_eta_floor) * cfg.meta_decay;
                            m.alert *= cfg.meta_decay;
                        }
                    }
                    MetaKind::Dbd => {
                        if m.n > 0 && e * m.e_bar > 0.0 {
                            m.eta_now = (m.eta_now + cfg.dbd_kappa).min(cfg.eta);
                        } else if m.n > 0 {
                            m.eta_now = (m.eta_now * cfg.dbd_phi).max(cfg.meta_eta_floor);
                        }
                        m.e_bar += cfg.dbd_theta * (e - m.e_bar);
                        let span = (cfg.eta - cfg.meta_eta_floor).max(1e-12);
                        m.alert = ((m.eta_now - cfg.meta_eta_floor) / span).clamp(0.0, 1.0);
                    }
                }
                if cfg.weighting == WeightKind::Contrast {
                    let hit = if preferred == agent { 1.0 } else { 0.0 };
                    m.agree += cfg.weight_alpha * (hit - m.agree);
                }
                if cfg.ucb_c > 0.0 {
                    for c in m.cnt.iter_mut() {
                        *c *= cfg.ucb_discount;
                    }
                    m.cnt[agent] += 1.0;
                }
                m.n += 1;
            }

            let delta = eta_f * 1.0 * e;
            for w in row.iter_mut() {
                *w *= 1.0 - cfg.decay;
            }
            row[agent] += delta;
            for w in row.iter_mut() {
                *w = w.clamp(0.0, 1.0);
            }
            let sum: f64 = row.iter().sum();
            if cfg.row_cap > 0.0 && sum > cfg.row_cap {
                let k = cfg.row_cap / sum;
                for w in row.iter_mut() {
                    *w *= k;
                }
            }
            self.baselines.insert(f, b + cfg.baseline_alpha * (reward - b));
        }
    }
}

// ---------------------------------------------------------------------------
// Kalman / Rescorla-Wagner synapses (opt-in, separate router)
// ---------------------------------------------------------------------------

/// Configuration of [`KalmanRouter`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KalmanConfig {
    /// Prior variance of a fresh synapse (logit units).
    pub prior_var: f64,
    /// Prior mean of a synapse from a keyword to its mapped agent.
    pub kw_mean: f64,
    /// Prior mean of every other synapse (pessimistic < 0 = "unknown agents are
    /// probably worse than the base rate").
    pub other_mean: f64,
    /// Process noise added to the variance of a synapse each time its row is
    /// updated (tracks drift; 0 = stationary).
    pub process_noise: f64,
    /// Thompson sampling scale on the posterior sd (0 = greedy on the mean).
    pub ts_scale: f64,
    /// Upper bound of a synapse variance (keeps process noise bounded).
    pub max_var: f64,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        KalmanConfig { prior_var: 1.0, kw_mean: 1.0, other_mean: 0.0, process_noise: 0.0, ts_scale: 1.0, max_var: 4.0 }
    }
}

/// Bayesian logistic synapses with diagonal extended-Kalman updates.
///
/// Score of agent `a` for a task = logit `z_a = sum_f m[f][a]` over the active
/// features (see [`active_features`]). On outcome `r` for the chosen agent `a`:
/// `p = sigma(z_a)`, `h = p(1-p)`, innovation variance `S = h^2 sum_f v[f][a] + h`,
/// gain `K_f = v[f][a] h / S`, `m[f][a] += K_f (r - p)`, `v[f][a] *= 1 - K_f h`.
/// So features compete for credit (blocking: a feature that already predicts the
/// outcome leaves little error for the others) and uncertain synapses learn
/// fastest. Selection: argmax over candidates of `m + ts_scale * sqrt(v) * xi`
/// summed over features (Thompson; `ts_scale = 0` is greedy).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KalmanRouter {
    pub cfg: KalmanConfig,
    n_agents: usize,
    /// keyword -> agent prior map (may be empty).
    kw: BTreeMap<String, usize>,
    mean: BTreeMap<String, Vec<f64>>,
    var: BTreeMap<String, Vec<f64>>,
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Standard normal (Box-Muller, two uniforms).
fn std_normal(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - rng.f64();
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

impl KalmanRouter {
    pub fn new(n_agents: usize, cfg: KalmanConfig) -> Self {
        KalmanRouter { cfg, n_agents, kw: BTreeMap::new(), mean: BTreeMap::new(), var: BTreeMap::new() }
    }

    /// Seed prior means from a keyword -> agent map.
    pub fn with_keywords<I: IntoIterator<Item = (String, usize)>>(mut self, map: I) -> Self {
        self.kw = map.into_iter().collect();
        self
    }

    fn prior_row(&self, f: &str) -> Vec<f64> {
        let k = self.kw.get(f).copied();
        (0..self.n_agents).map(|a| if Some(a) == k { self.cfg.kw_mean } else { self.cfg.other_mean }).collect()
    }

    /// Posterior (mean, variance) of synapse (feature, agent).
    pub fn synapse(&self, f: &str, a: usize) -> (f64, f64) {
        let m = self.mean.get(f).and_then(|r| r.get(a).copied()).unwrap_or_else(|| {
            if self.kw.get(f) == Some(&a) { self.cfg.kw_mean } else { self.cfg.other_mean }
        });
        let v = self.var.get(f).and_then(|r| r.get(a).copied()).unwrap_or(self.cfg.prior_var);
        (m, v)
    }

    /// Number of learned feature rows.
    pub fn n_rows(&self) -> usize {
        self.mean.len()
    }

    /// Posterior mean logit per agent.
    pub fn scores(&self, task: &Task) -> Vec<f64> {
        let feats = active_features(task);
        (0..self.n_agents).map(|a| feats.iter().map(|f| self.synapse(f, a).0).sum()).collect()
    }

    /// Predicted success probability of `agent` on `task`.
    pub fn predict(&self, task: &Task, agent: usize) -> f64 {
        sigmoid(self.scores(task).get(agent).copied().unwrap_or(0.0))
    }

    fn ensure_agents(&mut self, n: usize) {
        if n > self.n_agents {
            let (om, pv) = (self.cfg.other_mean, self.cfg.prior_var);
            for r in self.mean.values_mut() {
                r.resize(n, om);
            }
            for r in self.var.values_mut() {
                r.resize(n, pv);
            }
            self.n_agents = n;
        }
    }
}

impl Router for KalmanRouter {
    fn name(&self) -> String {
        "kalman".into()
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        assert!(!candidates.is_empty(), "KalmanRouter::select: no candidates");
        let feats = active_features(task);
        let mut best = (f64::NEG_INFINITY, candidates[0]);
        for &c in candidates {
            let mut z = 0.0;
            for f in &feats {
                let (m, v) = self.synapse(f, c);
                z += m;
                if self.cfg.ts_scale > 0.0 {
                    z += self.cfg.ts_scale * v.sqrt() * std_normal(rng);
                }
            }
            if z > best.0 {
                best = (z, c);
            }
        }
        best.1
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = if reward.is_finite() { reward.clamp(0.0, 1.0) } else { return };
        self.ensure_agents(agent + 1);
        let feats = active_features(task);
        for f in &feats {
            if !self.mean.contains_key(f) {
                let row = self.prior_row(f);
                self.mean.insert(f.clone(), row);
                self.var.insert(f.clone(), vec![self.cfg.prior_var; self.n_agents]);
            }
            let v = self.var.get_mut(f).unwrap();
            for x in v.iter_mut() {
                *x = (*x + self.cfg.process_noise).min(self.cfg.max_var);
            }
        }
        let z: f64 = feats.iter().map(|f| self.mean[f][agent]).sum();
        let p = sigmoid(z);
        let h = (p * (1.0 - p)).max(1e-6);
        let s: f64 = h * h * feats.iter().map(|f| self.var[f][agent]).sum::<f64>() + h;
        for f in &feats {
            let v = self.var[f][agent];
            let k = v * h / s;
            self.mean.get_mut(f).unwrap()[agent] += k * (r - p);
            self.var.get_mut(f).unwrap()[agent] = (v * (1.0 - k * h)).max(1e-6);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(t: &str) -> Task {
        Task { id: 0, task_type: Some(t.into()), features: vec![format!("kw-{t}")] }
    }

    /// Run `steps` rounds with success probabilities `p[agent]`; returns the
    /// fraction of picks of `track` in the last `window` rounds.
    fn run(r: &mut HebbianRouter, rng: &mut Rng, p: &[f64], steps: usize, track: usize, window: usize) -> f64 {
        let t = task("a");
        let cands: Vec<usize> = (0..p.len()).collect();
        let mut hits = 0;
        for i in 0..steps {
            let a = r.select(&t, &cands, rng);
            let rew = if rng.bernoulli(p[a]) { 1.0 } else { 0.0 };
            r.update(&t, a, rew);
            if i >= steps - window && a == track {
                hits += 1;
            }
        }
        hits as f64 / window as f64
    }

    #[test]
    fn converges_to_best_agent_stationary() {
        for seed in 1..=5 {
            let mut r = HebbianRouter::new(2, HebbianConfig::default());
            let mut rng = Rng::new(seed);
            let frac = run(&mut r, &mut rng, &[0.2, 0.9], 2000, 1, 500);
            assert!(frac > 0.85, "seed {seed}: best picked only {frac}");
            let s = r.scores(&task("a"));
            assert!(s[1] > s[0], "seed {seed}: scores {s:?}");
        }
    }

    #[test]
    fn decay_and_baseline_allow_switch_after_drift() {
        for seed in 1..=5 {
            let mut r = HebbianRouter::new(2, HebbianConfig::default());
            let mut rng = Rng::new(seed);
            let before = run(&mut r, &mut rng, &[0.9, 0.2], 1500, 0, 300);
            assert!(before > 0.85, "seed {seed}: pre-drift {before}");
            let after = run(&mut r, &mut rng, &[0.2, 0.9], 1500, 1, 300);
            assert!(after > 0.85, "seed {seed}: post-drift {after}");
        }
    }

    #[test]
    fn failure_lowers_weight_success_raises_it() {
        let mut r = HebbianRouter::new(3, HebbianConfig::default());
        let t = task("x");
        // Establish a baseline of ~0.5 then fail: weight must go below the start.
        let w0 = r.weight("kw-x", 1);
        r.update(&t, 1, 0.0);
        assert!(r.weight("kw-x", 1) < w0);
        assert!(r.baseline("kw-x") < INIT_BASELINE);
        let w1 = r.weight("kw-x", 2);
        r.update(&t, 2, 1.0);
        assert!(r.weight("kw-x", 2) > w1);
        // Once baseline is high (agent always succeeds), a failure is a large
        // negative surprise even though reward 0 would never increase a weight.
        let mut r = HebbianRouter::new(2, HebbianConfig::default());
        for _ in 0..50 {
            r.update(&t, 0, 1.0);
        }
        let hi = r.weight("kw-x", 0);
        assert!(r.baseline("kw-x") > 0.9);
        r.update(&t, 0, 0.0);
        assert!(r.weight("kw-x", 0) < hi - 0.3, "baseline-relative failure should cut weight");
    }

    #[test]
    fn row_cap_holds_and_weights_bounded() {
        let cfg = HebbianConfig { row_cap: 0.8, ..HebbianConfig::default() };
        let mut r = HebbianRouter::new(4, cfg);
        let t = task("y");
        let mut rng = Rng::new(7);
        for i in 0..2000 {
            let a = i % 4;
            r.update(&t, a, if rng.bernoulli(0.7) { 1.0 } else { 0.0 });
            for (f, row) in r.rows() {
                let s: f64 = row.iter().sum();
                assert!(s <= 0.8 + 1e-9, "row {f} sum {s}");
                assert!(row.iter().all(|w| (0.0..=1.0).contains(w)));
            }
        }
    }

    #[test]
    fn unknown_features_neutral_and_agents_grow() {
        let mut r = HebbianRouter::new(2, HebbianConfig::default());
        let s = r.scores(&task("new"));
        assert_eq!(s, vec![2.0 * INIT_WEIGHT; 2]);
        r.update(&task("new"), 4, 1.0);
        assert_eq!(r.n_agents(), 5);
        assert_eq!(r.scores(&task("new")).len(), 5);
    }

    #[test]
    fn serde_roundtrip_and_determinism() {
        let mut r = HebbianRouter::new(3, HebbianConfig::default());
        let mut rng = Rng::new(3);
        run(&mut r, &mut rng, &[0.3, 0.6, 0.9], 300, 2, 10);
        let js = serde_json::to_string(&r).unwrap();
        let back: HebbianRouter = serde_json::from_str(&js).unwrap();
        // serde_json may differ in the last ulp of a float; compare with tolerance.
        assert_eq!(back.n_agents(), r.n_agents());
        assert_eq!(back.updates(), r.updates());
        for (f, row) in r.rows() {
            for (a, w) in row.iter().enumerate() {
                assert!((back.weight(f, a) - w).abs() < 1e-12);
            }
            assert!((back.baseline(f) - r.baseline(f)).abs() < 1e-12);
        }
        let (mut a, mut b) = (r.clone(), back);
        let (mut ra, mut rb) = (Rng::new(9), Rng::new(9));
        for _ in 0..100 {
            assert_eq!(a.select(&task("a"), &[0, 1, 2], &mut ra), b.select(&task("a"), &[0, 1, 2], &mut rb));
        }
        a.update(&task("a"), 0, 1.0);
    }

    #[test]
    fn task_type_is_a_feature() {
        let t = Task { id: 1, task_type: Some("k".into()), features: vec!["k".into(), "k".into()] };
        assert_eq!(active_features(&t), vec!["k".to_string(), "type:k".to_string()]);
    }

    // ---- opt-in options -------------------------------------------------------

    #[test]
    fn options_off_by_default_and_state_stays_empty() {
        let mut r = HebbianRouter::new(2, HebbianConfig::default());
        let t = task("a");
        r.update(&t, 0, 1.0);
        assert!(r.meta.is_empty());
        // hand-computed first update: w = 0.1*(1-0.01) + 0.5*(1-0.5)
        let w = 0.1 * 0.99 + 0.5 * 0.5;
        assert!((r.weight("kw-a", 0) - w).abs() < 1e-12);
        assert_eq!(r.row_eta("kw-a"), 0.5);
        // old serialized configs (missing new fields) still load
        let old = "{\"eta\":0.2,\"epsilon\":0.02,\"temperature\":0.05,\"decay\":0.01,\"baseline_alpha\":0.1,\"row_cap\":1.0}";
        let c: HebbianConfig = serde_json::from_str(old).unwrap();
        assert_eq!(c.eta, 0.2);
        assert_eq!(c.meta, MetaKind::Off);
        assert_eq!(c.ucb_c, 0.0);
    }

    #[test]
    fn page_hinkley_anneals_then_resets_on_drift() {
        let cfg = HebbianConfig { meta: MetaKind::PageHinkley, ..HebbianConfig::default() };
        let mut r = HebbianRouter::new(2, cfg);
        let mut rng = Rng::new(1);
        let t = task("a");
        for _ in 0..300 {
            let rew = if rng.bernoulli(0.8) { 1.0 } else { 0.0 };
            r.update(&t, 0, rew);
        }
        let stable = r.row_eta("kw-a");
        assert!(stable < 0.1, "eta should anneal toward the floor, got {stable}");
        let mut boosted = false;
        for _ in 0..60 {
            r.update(&t, 0, 0.0); // reward collapses
            if r.row_eta("kw-a") > 0.3 {
                boosted = true;
            }
        }
        assert!(boosted, "drift must snap eta back toward the ceiling");
    }

    #[test]
    fn page_hinkley_recovers_faster_than_low_fixed_eta() {
        let mut fast = 0.0;
        let mut slow = 0.0;
        for seed in 1..=8 {
            let low = HebbianConfig { eta: 0.03, temperature: 0.02, ..HebbianConfig::default() };
            let ph = HebbianConfig { meta: MetaKind::PageHinkley, eta: 0.3, temperature: 0.02, ..HebbianConfig::default() };
            for (cfg, acc) in [(ph, &mut fast), (low, &mut slow)] {
                let mut r = HebbianRouter::new(2, cfg);
                let mut rng = Rng::new(seed);
                run(&mut r, &mut rng, &[0.9, 0.2], 800, 0, 100);
                *acc += run(&mut r, &mut rng, &[0.2, 0.9], 150, 1, 50);
            }
        }
        assert!(fast > slow, "PH {fast} vs fixed-low {slow}");
    }

    #[test]
    fn dbd_eta_shrinks_on_noise_and_stays_bounded() {
        let cfg = HebbianConfig { meta: MetaKind::Dbd, eta: 0.4, meta_eta_floor: 0.02, ..HebbianConfig::default() };
        let mut r = HebbianRouter::new(2, cfg);
        let t = task("a");
        for _ in 0..40 {
            r.update(&t, 0, 1.0);
            r.update(&t, 0, 0.0); // alternating sign: surprises disagree with their trace
        }
        let noisy = r.row_eta("kw-a");
        assert!(noisy < 0.1, "noise should shrink eta, got {noisy}");
        for _ in 0..60 {
            r.update(&t, 0, 0.0); // sustained surprise in one direction
        }
        let after = r.row_eta("kw-a");
        assert!(after >= 0.02 && after <= 0.4);
    }

    #[test]
    fn reliability_weighting_entropy_contrast_idf() {
        let t = task("a");
        for kind in [WeightKind::Entropy, WeightKind::Contrast, WeightKind::Idf] {
            let cfg = HebbianConfig { weighting: kind, ..HebbianConfig::default() };
            let mut r = HebbianRouter::new(3, cfg);
            let mut rng = Rng::new(5);
            for _ in 0..300 {
                r.update(&t, 1, if rng.bernoulli(0.9) { 1.0 } else { 0.0 });
            }
            let rel = r.feature_reliability("kw-a");
            assert!((0.3..=1.0).contains(&rel), "{kind:?} rel {rel}");
            let s = r.scores(&t);
            assert!(s[1] > s[0] && s[1] > s[2], "{kind:?} {s:?}");
        }
        // entropy: a peaked row is more reliable than an untouched one
        let cfg = HebbianConfig { weighting: WeightKind::Entropy, ..HebbianConfig::default() };
        let mut r = HebbianRouter::new(3, cfg);
        for _ in 0..50 {
            r.update(&t, 1, 1.0);
        }
        assert!(r.feature_reliability("kw-a") > r.feature_reliability("never-seen"));
        // contrast: a feature whose preferred agent never matches the routed one loses reliability
        let cfg = HebbianConfig { weighting: WeightKind::Contrast, weight_alpha: 0.1, ..HebbianConfig::default() };
        let mut r = HebbianRouter::new(3, cfg);
        let both = Task { id: 0, task_type: None, features: vec!["good".into(), "bad".into()] };
        let only_bad = Task { id: 1, task_type: None, features: vec!["bad".into()] };
        for _ in 0..30 {
            r.update(&only_bad, 2, 1.0);
        }
        for _ in 0..200 {
            r.update(&both, 0, 1.0);
        }
        assert!(r.feature_reliability("good") > r.feature_reliability("bad"));
    }

    #[test]
    fn ucb_bonus_favours_untried_cells() {
        let cfg = HebbianConfig { ucb_c: 0.2, epsilon: 0.0, temperature: 0.02, ..HebbianConfig::default() };
        let mut r = HebbianRouter::new(3, cfg);
        let t = task("a");
        for _ in 0..30 {
            r.update(&t, 0, 1.0);
        }
        let f = active_features(&t);
        assert!(r.ucb_bonus(&f, 1) > r.ucb_bonus(&f, 0) + 0.1);
        let r0 = HebbianRouter::new(3, HebbianConfig::default());
        assert_eq!(r0.ucb_bonus(&f, 1), 0.0);
    }

    #[test]
    fn ucb_finds_untried_better_agent_where_greedy_sticks() {
        let mut greedy_hits = 0;
        let mut ucb_hits = 0;
        for seed in 1..=10 {
            let g = HebbianConfig { epsilon: 0.0, temperature: 0.02, eta: 0.1, ..HebbianConfig::default() };
            let u = HebbianConfig { ucb_c: 0.15, ..g.clone() };
            for (cfg, hits) in [(g, &mut greedy_hits), (u, &mut ucb_hits)] {
                let mut r = HebbianRouter::new(2, cfg);
                let mut rng = Rng::new(seed);
                let frac = run(&mut r, &mut rng, &[0.5, 0.95], 600, 1, 200);
                if frac > 0.6 {
                    *hits += 1;
                }
            }
        }
        assert!(ucb_hits >= greedy_hits, "ucb {ucb_hits} greedy {greedy_hits}");
        assert!(ucb_hits >= 8, "ucb {ucb_hits}");
    }

    #[test]
    fn options_roundtrip_serde_and_grow_agents() {
        let cfg = HebbianConfig {
            meta: MetaKind::PageHinkley,
            weighting: WeightKind::Contrast,
            ucb_c: 0.05,
            ..HebbianConfig::default()
        };
        let mut r = HebbianRouter::new(2, cfg);
        let t = task("z");
        for i in 0..50 {
            r.update(&t, i % 2, 1.0);
        }
        r.update(&t, 4, 1.0);
        let back: HebbianRouter = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back.n_agents(), 5);
        assert_eq!(back.scores(&t).len(), 5);
        let mut b = back;
        let mut rng = Rng::new(1);
        let _ = b.select(&t, &[0, 1, 2, 3, 4], &mut rng);
    }

    #[test]
    fn median_aggregation_resists_one_confound() {
        let cfg = HebbianConfig { agg: AggKind::Median, ..HebbianConfig::default() };
        let mut r = HebbianRouter::new(3, cfg);
        let a = Task { id: 0, task_type: None, features: vec!["a1".into(), "a2".into()] };
        let c = Task { id: 0, task_type: None, features: vec!["c".into()] };
        for _ in 0..50 {
            r.update(&a, 0, 1.0); // a1, a2 -> agent 0 (moderate)
            r.update(&c, 1, 1.0); // c -> agent 1 (strong)
        }
        let mixed = Task { id: 0, task_type: None, features: vec!["a1".into(), "a2".into(), "c".into()] };
        let s = r.scores(&mixed);
        assert!(s[0] > s[1], "median keeps the 2-feature majority: {s:?}");
        let sum = HebbianRouter { cfg: HebbianConfig::default(), ..r.clone() };
        assert_eq!(sum.scores(&a), r.scores(&a).iter().map(|x| *x).collect::<Vec<_>>()); // n=2: median*2 == sum
        assert_eq!(HebbianConfig::default().agg, AggKind::Sum);
    }

    #[test]
    fn kalman_prior_learning_and_credit_assignment() {
        let kw = [("sql".to_string(), 1usize)];
        let mut r = KalmanRouter::new(3, KalmanConfig { ts_scale: 0.0, ..KalmanConfig::default() }).with_keywords(kw.clone());
        let t = Task { id: 0, task_type: None, features: vec!["sql".into()] };
        let mut rng = Rng::new(1);
        assert_eq!(r.select(&t, &[0, 1, 2], &mut rng), 1); // keyword prior
        // repeated failures of agent 1 -> switch
        for _ in 0..10 {
            r.update(&t, 1, 0.0);
        }
        assert_ne!(r.select(&t, &[0, 1, 2], &mut rng), 1);
        let (_, v) = r.synapse("sql", 1);
        assert!(v < KalmanConfig::default().prior_var);
        // blocking: an established predictor takes little credit away from... a new
        // (uncertain) feature gets most of the update
        let mut r = KalmanRouter::new(2, KalmanConfig { ts_scale: 0.0, kw_mean: 0.0, ..KalmanConfig::default() });
        let old = Task { id: 0, task_type: None, features: vec!["old".into()] };
        for _ in 0..200 {
            r.update(&old, 0, 1.0);
        }
        let both = Task { id: 0, task_type: None, features: vec!["old".into(), "new".into()] };
        let (m_old0, _) = r.synapse("old", 0);
        r.update(&both, 0, 0.0);
        let d_old = m_old0 - r.synapse("old", 0).0;
        let d_new = 0.0 - r.synapse("new", 0).0;
        assert!(d_new > 5.0 * d_old, "uncertain new cue takes the blame: new {d_new} old {d_old}");
        // finds a better agent on a stationary bandit with Thompson exploration
        let mut r = KalmanRouter::new(2, KalmanConfig::default());
        let mut hits = 0;
        for i in 0..1500 {
            let a = r.select(&t, &[0, 1], &mut rng);
            r.update(&t, a, if rng.bernoulli([0.3, 0.8][a]) { 1.0 } else { 0.0 });
            if i >= 1000 && a == 1 {
                hits += 1;
            }
        }
        assert!(hits > 400, "{hits}/500");
        let back: KalmanRouter = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(back.n_rows(), r.n_rows());
        r.update(&t, 4, 1.0);
        assert_eq!(r.scores(&t).len(), 5);
    }

    #[test]
    fn merge_sum_combines_disjoint_specialists_and_respects_row_cap() {
        let mk = |feat: &str, agent: usize| {
            let mut r = HebbianRouter::new(3, HebbianConfig { epsilon: 0.0, temperature: 0.0, ..HebbianConfig::default() });
            let t = Task { id: 0, task_type: None, features: vec![feat.to_string()] };
            for _ in 0..40 {
                r.update(&t, agent, 1.0);
                r.update(&t, (agent + 1) % 3, 0.0);
            }
            r
        };
        let (a, b) = (mk("sql", 0), mk("img", 2));
        let m = HebbianRouter::merge_sum(&[&a, &b]);
        let mut rng = Rng::new(1);
        let ts = Task { id: 0, task_type: None, features: vec!["sql".into()] };
        let ti = Task { id: 0, task_type: None, features: vec!["img".into()] };
        let mut m2 = m.clone();
        assert_eq!(m2.select(&ts, &[0, 1, 2], &mut rng), 0);
        assert_eq!(m2.select(&ti, &[0, 1, 2], &mut rng), 2);
        assert_eq!(m.updates, a.updates + b.updates);
        // same feature in both: rows are summed then rescaled to row_cap
        let m3 = HebbianRouter::merge_sum(&[&a, &a]);
        for row in m3.weights.values() {
            assert!(row.iter().sum::<f64>() <= m3.cfg.row_cap + 1e-12);
        }
        // a single router merges to itself
        assert_eq!(HebbianRouter::merge_sum(&[&a]).weights, a.weights);
    }
}
