//! G19 common: worlds (fixed BEFORE any policy was run), observation models, policy trait + wrappers, the run loop.
//! Everything is SYNTHETIC (author-built generators; x1_lab/x3 generators copied read-only from d7_sona_plus).

pub mod keyed;

use std::collections::{HashMap, VecDeque};

use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World as SimWorld, WorldConfig};

pub const HORIZONS: [usize; 5] = [100, 250, 500, 1000, 2000];
pub const N_TASKS: usize = 2000;

// ------------------------------------------------------------------------------------------------ worlds
#[derive(Clone, Debug)]
pub enum WorldKind {
    /// x1 generator: topics with 3 keyword forms each
    Short { topics: usize, traps: usize, map_err: f64, confound: f64 },
    /// x1 topics + a modifier feature; best agent depends on (topic, modifier)
    ShortMod { topics: usize, mods: usize, traps: usize, map_err: f64, confound: f64 },
    /// topics with `forms` surface forms each; keywords cover only the first `cover` forms of each topic
    Var { topics: usize, forms: usize, cover: usize, traps: usize, map_err: f64, confound: f64, single: bool },
    /// the best agent depends on the (topic, modifier) PAIR; optional noise features
    Conj { topics: usize, mods: usize, noise_pool: usize, skew: bool, trap: bool },
    Sim(Scenario),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fb {
    /// realised Bernoulli outcome after every task
    Normal,
    /// hook bit only (raw) -- what ruflo-style baselines see
    WeakRaw,
    /// raw failures now + delayed graded reuse value (30% of ok tasks, 25 tasks later, 10% flipped)
    WeakReuse,
    /// outcomes only at the end of each session of `L` tasks: reported success = ALL steps succeeded
    Session(usize),
}

#[derive(Clone, Debug)]
pub struct WorldSpec {
    pub name: &'static str,
    pub kind: WorldKind,
    pub fb: Fb,
}

pub struct World {
    pub agents: usize,
    pub tasks: Vec<Task>,
    pub p: Vec<Vec<f64>>,
    pub u: Vec<f64>,
    pub kwmap: HashMap<String, usize>,
    pub down: Vec<Vec<bool>>,
    /// feature -> true topic (diagnostic ORACLE anchors only)
    pub oracle: HashMap<String, usize>,
}

fn draw_p(rng: &mut Rng, t_n: usize, a_n: usize) -> (Vec<Vec<f64>>, Vec<usize>, Vec<usize>) {
    let mut p = vec![vec![0.0; a_n]; t_n];
    let mut best = vec![0usize; t_n];
    let mut second = vec![0usize; t_n];
    for t in 0..t_n {
        best[t] = rng.below(a_n);
        loop {
            second[t] = rng.below(a_n);
            if second[t] != best[t] {
                break;
            }
        }
        for a in 0..a_n {
            p[t][a] = if a == best[t] {
                0.85
            } else if a == second[t] {
                0.5
            } else {
                0.15 + 0.25 * rng.f64()
            };
        }
    }
    (p, best, second)
}

fn pick_traps(rng: &mut Rng, t_n: usize, traps: usize) -> Vec<bool> {
    let mut order: Vec<usize> = (0..t_n).collect();
    for i in (1..t_n).rev() {
        order.swap(i, rng.below(i + 1));
    }
    (0..t_n).map(|t| order.iter().take(traps).any(|&x| x == t)).collect()
}

fn kw_agent(rng: &mut Rng, trap: bool, a_n: usize, best: usize, second: usize, map_err: f64) -> usize {
    if trap {
        loop {
            let a = rng.below(a_n);
            if a != best && a != second {
                break a;
            }
        }
    } else if rng.f64() < map_err {
        rng.below(a_n)
    } else {
        best
    }
}

/// x1-style topics (3 co-occurring keyword forms, confounds, trap keywords) PLUS a modifier feature `m{j}`: the best agent depends on
/// the (topic, modifier) pair. A different generator from `gen_conj` (multi-form topics, keyword map on the forms).
fn gen_shortmod(seed: u64, topics: usize, mods: usize, traps: usize, map_err: f64, confound: f64) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(15485863).wrapping_add(59));
    let a_n = 6usize;
    let (p, best, second) = draw_p(&mut rng, topics * mods, a_n);
    let trap = pick_traps(&mut rng, topics, traps);
    let mut kwmap = HashMap::new();
    let mut oracle = HashMap::new();
    for t in 0..topics {
        for j in 0..3 {
            // keyword rule is calibrated on modifier 0 only
            let a = kw_agent(&mut rng, trap[t], a_n, best[t * mods], second[t * mods], map_err);
            kwmap.insert(format!("t{t}k{j}"), a);
            oracle.insert(format!("t{t}k{j}"), t);
        }
    }
    let (mut tasks, mut pp, mut u) = (vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let t = rng.below(topics);
        let m = rng.below(mods);
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        f.push(format!("m{m}"));
        if rng.bernoulli(confound) {
            let o = (t + 1 + rng.below(topics - 1)) % topics;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        for a in (1..f.len()).rev() {
            f.swap(a, rng.below(a + 1));
        }
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        pp.push(p[t * mods + m].clone());
        u.push(rng.f64());
    }
    World { agents: a_n, tasks, p: pp, u, kwmap, down: vec![vec![false; a_n]; N_TASKS], oracle }
}

fn gen_short(seed: u64, topics: usize, traps: usize, map_err: f64, confound: f64) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(11));
    let (t_n, a_n) = (topics, 6usize);
    let (p, best, second) = draw_p(&mut rng, t_n, a_n);
    let trap = pick_traps(&mut rng, t_n, traps);
    let mut kwmap = HashMap::new();
    let mut oracle = HashMap::new();
    for t in 0..t_n {
        for j in 0..3 {
            let a = kw_agent(&mut rng, trap[t], a_n, best[t], second[t], map_err);
            kwmap.insert(format!("t{t}k{j}"), a);
            oracle.insert(format!("t{t}k{j}"), t);
        }
    }
    let (mut tasks, mut pp, mut u) = (vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let t = rng.below(t_n);
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(confound) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        pp.push(p[t].clone());
        u.push(rng.f64());
    }
    World { agents: a_n, tasks, p: pp, u, kwmap, down: vec![vec![false; a_n]; N_TASKS], oracle }
}

#[allow(clippy::too_many_arguments)]
fn gen_var(seed: u64, topics: usize, forms: usize, cover: usize, traps: usize, map_err: f64, confound: f64, single: bool) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(104729).wrapping_add(23));
    let (t_n, a_n) = (topics, 6usize);
    let (p, best, second) = draw_p(&mut rng, t_n, a_n);
    let trap = pick_traps(&mut rng, t_n, traps);
    let mut kwmap = HashMap::new();
    let mut oracle = HashMap::new();
    for t in 0..t_n {
        for j in 0..forms {
            oracle.insert(format!("t{t}v{j}"), t);
        }
        for j in 0..cover {
            let a = kw_agent(&mut rng, trap[t], a_n, best[t], second[t], map_err);
            kwmap.insert(format!("t{t}v{j}"), a);
        }
    }
    let (mut tasks, mut pp, mut u) = (vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let t = rng.below(t_n);
        let k = if single { 1 } else { 1 + rng.below(3) };
        let mut idx: Vec<usize> = (0..forms).collect();
        for x in 0..k {
            let y = x + rng.below(forms - x);
            idx.swap(x, y);
        }
        let mut f: Vec<String> = idx.iter().take(k).map(|j| format!("t{t}v{j}")).collect();
        if !single && rng.bernoulli(confound) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f.push(format!("t{o}v{}", rng.below(forms)));
        }
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        pp.push(p[t].clone());
        u.push(rng.f64());
    }
    World { agents: a_n, tasks, p: pp, u, kwmap, down: vec![vec![false; a_n]; N_TASKS], oracle }
}

fn gen_conj(seed: u64, topics: usize, mods: usize, noise_pool: usize, skew: bool, trap: bool) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(1299709).wrapping_add(41));
    let a_n = 6usize;
    let pairs = topics * mods;
    let (p, best, _second) = draw_p(&mut rng, pairs, a_n);
    let mut kwmap = HashMap::new();
    for t in 0..topics {
        if trap {
            // TRAP keyword: the agent with the lowest mean success over the topic's modifiers
            let mut worst = 0usize;
            let mut ws = f64::INFINITY;
            for a in 0..a_n {
                let s: f64 = (0..mods).map(|m| p[t * mods + m][a]).sum();
                if s < ws {
                    ws = s;
                    worst = a;
                }
            }
            kwmap.insert(format!("t{t}"), worst);
        } else {
            // keyword rule = best agent for modifier 0 only (right for the common modifier, wrong for others)
            kwmap.insert(format!("t{t}"), best[t * mods]);
        }
    }
    let wts: Vec<f64> = (0..mods).map(|m| if skew { 1.0 / (m as f64 + 1.0) } else { 1.0 }).collect();
    let tot: f64 = wts.iter().sum();
    let (mut tasks, mut pp, mut u) = (vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let t = rng.below(topics);
        let mut x = rng.f64() * tot;
        let mut m = mods - 1;
        for (j, w) in wts.iter().enumerate() {
            x -= w;
            if x <= 0.0 {
                m = j;
                break;
            }
        }
        let mut f = vec![format!("t{t}"), format!("m{m}")];
        if noise_pool > 0 {
            let r = rng.f64();
            let nn = if r < 0.3 { 0 } else if r < 0.7 { 1 } else { 2 };
            let mut seen: Vec<usize> = vec![];
            while seen.len() < nn {
                let z = rng.below(noise_pool);
                if !seen.contains(&z) {
                    seen.push(z);
                    f.push(format!("n{z}"));
                }
            }
        }
        // shuffle feature order (routers must not rely on position)
        for a in (1..f.len()).rev() {
            f.swap(a, rng.below(a + 1));
        }
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        pp.push(p[t * mods + m].clone());
        u.push(rng.f64());
    }
    World { agents: a_n, tasks, p: pp, u, kwmap, down: vec![vec![false; a_n]; N_TASKS], oracle: HashMap::new() }
}

fn gen_sim(seed: u64, sc: Scenario) -> World {
    let w = SimWorld::new(seed, WorldConfig { n_tasks: N_TASKS, scenario: sc, ..WorldConfig::default() });
    let n = w.n_agents();
    let tasks: Vec<Task> = w.tasks().iter().map(|s| s.task.clone()).collect();
    let p: Vec<Vec<f64>> = (0..w.len()).map(|t| (0..n).map(|a| w.success_prob(t, &[a])).collect()).collect();
    let u: Vec<f64> = w.tasks().iter().map(|s| s.u).collect();
    let down: Vec<Vec<bool>> = (0..w.len()).map(|t| (0..n).map(|a| w.is_down(t, a)).collect()).collect();
    World { agents: n, tasks, p, u, kwmap: w.keyword_map_for(), down, oracle: HashMap::new() }
}

pub fn gen_world(seed: u64, s: &WorldSpec) -> World {
    match &s.kind {
        WorldKind::Short { topics, traps, map_err, confound } => gen_short(seed, *topics, *traps, *map_err, *confound),
        WorldKind::ShortMod { topics, mods, traps, map_err, confound } => gen_shortmod(seed, *topics, *mods, *traps, *map_err, *confound),
        WorldKind::Var { topics, forms, cover, traps, map_err, confound, single } => gen_var(seed, *topics, *forms, *cover, *traps, *map_err, *confound, *single),
        WorldKind::Conj { topics, mods, noise_pool, skew, trap } => gen_conj(seed, *topics, *mods, *noise_pool, *skew, *trap),
        WorldKind::Sim(sc) => gen_sim(seed, *sc),
    }
}

// ------------------------------------------------------------------------------------------------ worlds, fixed up front
const VAR_TRAPS: WorldKind = WorldKind::Var { topics: 8, forms: 10, cover: 3, traps: 2, map_err: 0.10, confound: 0.25, single: false };

/// Normal-feedback worlds. Order: 5 NEW worlds (keying), then the 5 d7 control worlds (must be harmless).
pub fn normal_worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec { name: "var-traps", kind: VAR_TRAPS, fb: Fb::Normal },
        WorldSpec { name: "var-clean", kind: WorldKind::Var { topics: 8, forms: 10, cover: 3, traps: 0, map_err: 0.0, confound: 0.25, single: false }, fb: Fb::Normal },
        WorldSpec { name: "var-single(ctl)", kind: WorldKind::Var { topics: 8, forms: 10, cover: 3, traps: 2, map_err: 0.10, confound: 0.0, single: true }, fb: Fb::Normal },
        WorldSpec { name: "conj-clean", kind: WorldKind::Conj { topics: 6, mods: 3, noise_pool: 0, skew: true, trap: false }, fb: Fb::Normal },
        WorldSpec { name: "conj-noisy", kind: WorldKind::Conj { topics: 6, mods: 3, noise_pool: 20, skew: true, trap: false }, fb: Fb::Normal },
        WorldSpec { name: "traps(main)", kind: WorldKind::Short { topics: 8, traps: 2, map_err: 0.10, confound: 0.25 }, fb: Fb::Normal },
        WorldSpec { name: "correct-keywords", kind: WorldKind::Short { topics: 8, traps: 0, map_err: 0.0, confound: 0.25 }, fb: Fb::Normal },
        WorldSpec { name: "many-topics(16)", kind: WorldKind::Short { topics: 16, traps: 2, map_err: 0.10, confound: 0.25 }, fb: Fb::Normal },
        WorldSpec { name: "sim-random", kind: WorldKind::Sim(Scenario::Random), fb: Fb::Normal },
        WorldSpec { name: "sim-notraps", kind: WorldKind::Sim(Scenario::NoTraps), fb: Fb::Normal },
        // ---- round-2 worlds (added AFTER round 1 showed the keyword guard masking the learner in conj-*; round-1 worlds unchanged)
        WorldSpec { name: "conjT-clean", kind: WorldKind::Conj { topics: 6, mods: 3, noise_pool: 0, skew: true, trap: true }, fb: Fb::Normal },
        WorldSpec { name: "conjT-noisy", kind: WorldKind::Conj { topics: 6, mods: 3, noise_pool: 20, skew: true, trap: true }, fb: Fb::Normal },
        WorldSpec { name: "var-wide-traps", kind: WorldKind::Var { topics: 8, forms: 20, cover: 3, traps: 2, map_err: 0.10, confound: 0.25, single: false }, fb: Fb::Normal },
        // ---- round-5 worlds (a conjunction structure from a DIFFERENT generator than gen_conj; added after round 4)
        WorldSpec { name: "shortmod-traps", kind: WorldKind::ShortMod { topics: 8, mods: 3, traps: 2, map_err: 0.10, confound: 0.25 }, fb: Fb::Normal },
        WorldSpec { name: "shortmod-clean", kind: WorldKind::ShortMod { topics: 8, mods: 3, traps: 0, map_err: 0.0, confound: 0.25 }, fb: Fb::Normal },
    ]
}

/// Weak-feedback worlds (same observation model as d7/x3). All arms see the SAME signal.
pub fn weak_worlds(mode: Fb) -> Vec<WorldSpec> {
    vec![
        WorldSpec { name: "weak-var-traps", kind: VAR_TRAPS, fb: mode },
        WorldSpec { name: "weak-conj-noisy", kind: WorldKind::Conj { topics: 6, mods: 3, noise_pool: 20, skew: true, trap: false }, fb: mode },
        WorldSpec { name: "weak-conjT-noisy", kind: WorldKind::Conj { topics: 6, mods: 3, noise_pool: 20, skew: true, trap: true }, fb: mode },
        WorldSpec { name: "weak-var-wide-traps", kind: WorldKind::Var { topics: 8, forms: 20, cover: 3, traps: 2, map_err: 0.10, confound: 0.25, single: false }, fb: mode },
        WorldSpec { name: "weak-shortmod-traps", kind: WorldKind::ShortMod { topics: 8, mods: 3, traps: 2, map_err: 0.10, confound: 0.25 }, fb: mode },
        WorldSpec { name: "weak-traps(ctl)", kind: WorldKind::Short { topics: 8, traps: 2, map_err: 0.10, confound: 0.25 }, fb: mode },
    ]
}

/// Session worlds: outcomes only per session of 3 tasks (all steps must succeed).
pub fn session_worlds() -> Vec<WorldSpec> {
    vec![
        WorldSpec { name: "sess-traps", kind: WorldKind::Short { topics: 8, traps: 2, map_err: 0.10, confound: 0.25 }, fb: Fb::Session(3) },
        WorldSpec { name: "sess-var-traps", kind: VAR_TRAPS, fb: Fb::Session(3) },
        WorldSpec { name: "sess-sim-random", kind: WorldKind::Sim(Scenario::Random), fb: Fb::Session(3) },
    ]
}

/// Capacity world: many (topic,modifier) patterns, noise features.
pub fn mem_world() -> WorldSpec {
    WorldSpec { name: "mem-conj48", kind: WorldKind::Conj { topics: 12, mods: 4, noise_pool: 20, skew: false, trap: false }, fb: Fb::Normal }
}

/// Capacity world, round 2: every keyword is a trap so the learner (not the keyword guard) decides.
pub fn memt_world() -> WorldSpec {
    WorldSpec { name: "memT-conj48", kind: WorldKind::Conj { topics: 12, mods: 4, noise_pool: 20, skew: false, trap: true }, fb: Fb::Normal }
}

// ------------------------------------------------------------------------------------------------ observation model (x3 WeakCfg copy)
pub const TOOL_BASE: f64 = 0.01;
pub const TOOL_SLOPE: f64 = 0.03;
pub const REUSE_FRAC: f64 = 0.3;
pub const REUSE_DELAY: usize = 25;
pub const REUSE_NOISE: f64 = 0.1;

// ------------------------------------------------------------------------------------------------ policies
pub trait Pol {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize;
    fn upd(&mut self, t: &Task, a: usize, r: f64);
    fn weak(&mut self, t: &Task, a: usize, raw_ok: bool) {
        if !raw_ok {
            self.upd(t, a, 0.0)
        }
    }
    fn delayed(&mut self, t: &Task, a: usize, v: f64) {
        self.upd(t, a, v)
    }
    /// only the session outcome is known (default: broadcast the shared bit to every step)
    fn session_end(&mut self, steps: &[(Task, usize)], ok: bool) {
        for (t, a) in steps {
            self.upd(t, *a, if ok { 1.0 } else { 0.0 });
        }
    }
    /// the policy's own estimate of P(success) for (task, agent), if it has one
    fn phat(&self, _t: &Task, _a: usize) -> Option<f64> {
        None
    }
}

pub struct Plain<R: Router>(pub R);
impl<R: Router> Pol for Plain<R> {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(t, c, rng)
    }
    fn upd(&mut self, t: &Task, a: usize, r: f64) {
        self.0.update(t, a, r)
    }
}

/// Candidate: AnchorGuard over a token-level inner learner, with the SonaPlus weak-feedback path.
pub struct KPol<I: keyed::KInner> {
    pub g: keyed::AnchorGuard<I>,
    pub delayed_mult: usize,
    pub fail_reward: f64,
}
impl<I: keyed::KInner> Pol for KPol<I> {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.g.select(t, c, rng)
    }
    fn upd(&mut self, t: &Task, a: usize, r: f64) {
        self.g.update(t, a, r)
    }
    fn weak(&mut self, t: &Task, a: usize, raw_ok: bool) {
        if !raw_ok {
            let r = self.fail_reward;
            self.g.update(t, a, r)
        }
    }
    fn delayed(&mut self, t: &Task, a: usize, v: f64) {
        for _ in 0..self.delayed_mult.max(1) {
            self.g.update(t, a, v)
        }
    }
    fn phat(&self, t: &Task, a: usize) -> Option<f64> {
        self.g.p_hat(t, a)
    }
}

/// Plain per-feature/per-agent mean estimator (the "no pattern memory" estimator for trajectory credit).
pub struct UniEst {
    m: HashMap<String, Vec<(f64, f64)>>,
    n: usize,
}
impl UniEst {
    pub fn new(n: usize) -> Self {
        UniEst { m: HashMap::new(), n }
    }
    pub fn p(&self, t: &Task, a: usize) -> f64 {
        let (mut s, mut w) = (0.0, 0.0);
        for f in &t.features {
            if let Some(&(cs, cw)) = self.m.get(f).and_then(|r| r.get(a)) {
                s += cs;
                w += cw;
            }
        }
        (s + 0.6 * 2.0) / (w + 2.0)
    }
    pub fn update(&mut self, t: &Task, a: usize, r: f64) {
        for f in &t.features {
            let row = self.m.entry(f.clone()).or_insert_with(|| vec![(0.0, 0.0); self.n.max(a + 1)]);
            if row.len() <= a {
                row.resize(a + 1, (0.0, 0.0));
            }
            row[a].0 += r;
            row[a].1 += 1.0;
        }
    }
}

/// Trajectory-level credit: when a session of steps fails, blame is split by the posterior
/// q_i = (1 - p_i) / (1 - prod_j p_j) (P(step i failed | session failed)); credit_i = 1 - q_i. A success credits every step 1.
/// `own` = use the wrapped policy's own estimate (pattern memory) instead of the plain per-feature estimator.
pub struct Credited {
    pub inner: Box<dyn Pol>,
    est: UniEst,
    own: bool,
}
impl Credited {
    pub fn new(inner: Box<dyn Pol>, n_agents: usize, own: bool) -> Self {
        Credited { inner, est: UniEst::new(n_agents), own }
    }
}
impl Pol for Credited {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.inner.pick(t, c, rng)
    }
    fn upd(&mut self, t: &Task, a: usize, r: f64) {
        self.inner.upd(t, a, r)
    }
    fn session_end(&mut self, steps: &[(Task, usize)], ok: bool) {
        if ok {
            for (t, a) in steps {
                self.est.update(t, *a, 1.0);
                self.inner.upd(t, *a, 1.0);
            }
            return;
        }
        let ps: Vec<f64> = steps
            .iter()
            .map(|(t, a)| {
                let p = if self.own { self.inner.phat(t, *a).unwrap_or_else(|| self.est.p(t, *a)) } else { self.est.p(t, *a) };
                p.clamp(0.02, 0.98)
            })
            .collect();
        let all: f64 = ps.iter().product();
        let denom = (1.0 - all).max(1e-9);
        for ((t, a), p) in steps.iter().zip(&ps) {
            let q = ((1.0 - p) / denom).clamp(0.0, 1.0);
            let r = 1.0 - q;
            self.est.update(t, *a, r);
            self.inner.upd(t, *a, r);
        }
    }
    fn phat(&self, t: &Task, a: usize) -> Option<f64> {
        self.inner.phat(t, a)
    }
}

// ------------------------------------------------------------------------------------------------ run loop
/// Prefix means of EXPECTED success of the routed agent at HORIZONS.
pub fn run(build: &dyn Fn(&World) -> Box<dyn Pol>, w: &World, seed: u64, fb: Fb) -> [f64; 5] {
    let cands: Vec<usize> = (0..w.agents).collect();
    let mut pol = build(w);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut orng = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(0x3EAC));
    let draws: Vec<(f64, f64, f64)> = (0..w.tasks.len()).map(|_| (orng.f64(), orng.f64(), orng.f64())).collect();
    let mut pending: VecDeque<(usize, usize, usize, f64)> = VecDeque::new();
    let mut steps: Vec<(Task, usize)> = vec![];
    let mut sess_ok = true;
    let (mut cum, mut out, mut hi) = (0.0, [0.0; 5], 0usize);
    for t in 0..w.tasks.len() {
        while pending.front().is_some_and(|p| p.0 <= t) {
            let (_, ti, a, v) = pending.pop_front().unwrap();
            pol.delayed(&w.tasks[ti], a, v);
        }
        let task = &w.tasks[t];
        let a = pol.pick(task, &cands, &mut rng);
        let p = w.p[t][a];
        let s = if w.u[t] < p { 1.0 } else { 0.0 };
        cum += p;
        match fb {
            Fb::Normal => pol.upd(task, a, s),
            Fb::Session(l) => {
                steps.push((task.clone(), a));
                sess_ok &= s > 0.5;
                if steps.len() == l || t + 1 == w.tasks.len() {
                    pol.session_end(&steps, sess_ok);
                    steps.clear();
                    sess_ok = true;
                }
            }
            Fb::WeakRaw | Fb::WeakReuse => {
                let (u_tool, u_reuse, u_noise) = draws[t];
                let raw_ok = !w.down[t][a] && u_tool >= TOOL_BASE + TOOL_SLOPE * (1.0 - s);
                if fb == Fb::WeakRaw {
                    pol.upd(task, a, if raw_ok { 1.0 } else { 0.0 });
                } else {
                    pol.weak(task, a, raw_ok);
                    if raw_ok && u_reuse < REUSE_FRAC {
                        let v = if u_noise < REUSE_NOISE { 1.0 - s } else { s };
                        pending.push_back((t + REUSE_DELAY, t, a, v));
                    }
                }
            }
        }
        if hi < 5 && t + 1 == HORIZONS[hi] {
            out[hi] = cum / (t + 1) as f64;
            hi += 1;
        }
    }
    out
}
