//! F1 lab (CAP-1): grounded, measurable sample-efficiency / capability mechanisms
//! for the Hebbian router. NOTHING here claims more work per compute, time dilation
//! or "new intelligence": every number is expected success / utility of a router on a
//! simulated world, paired by seed, against the best fair baseline.
//!
//! Experiments (all opt-in mechanisms; library untouched):
//!   A1 cross-deployment warm transfer (child router inherits a parent's learned map)
//!   A2 agent added mid-run (new agent inherits a declared sibling's evidence)
//!   B  active task ordering / curriculum for a reorderable batch queue
//!   C  multi-objective routing (quality + cost + latency): decoupled quality learner
//!   D  committee (ensemble) disagreement as a difficulty / exploration signal
//!
//! Protocol: tune on seeds 1-8 only; evaluate on held-out seeds (default 31-60;
//! 61-90 = frozen confirmation); paired 95% t-CI vs the best of the fair baseline set
//! {keyword, eps-greedy, pooled-thompson, hebbian-robust, warm(hebr), guarded(hebr)}
//! per cell; Holm within each experiment family and globally. Control cells (where the
//! mechanism should NOT help) are part of the family.
//!
//! Usage: cargo run --release --bin f1_lab -- [--exp a1,a2,b,c,d] [--held 31-60]
//!        [--out results/f1_lab.json] [--quick]

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter, INIT_WEIGHT};
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, Summary};
use rayon::prelude::*;
use serde_json::{json, Value};

const N_TASKS: usize = 2000;
const NA: usize = 6;
const HORIZONS: [usize; 5] = [100, 250, 500, 1000, 2000];
const BASES: [&str; 6] = ["keyword", "eps-greedy", "pooled-ts", "hebbian-robust", "warm(hebr)", "guarded(hebr)"];

// =============================================================== topic worlds
#[derive(Clone, Debug)]
struct TopicSpec {
    name: &'static str,
    topics: usize,
    traps: usize,
    map_err: f64,
    confound: f64,
    zipf: bool,
    /// flat: every agent has the same skill on every topic (nothing to learn)
    flat: bool,
}

#[derive(Clone)]
struct Skills {
    p: Vec<Vec<f64>>,
    kw: HashMap<String, usize>,
}

struct World {
    agents: usize,
    tasks: Vec<Task>,
    ty: Vec<usize>,
    p: Vec<Vec<f64>>,
    u: Vec<f64>,
    kwmap: HashMap<String, usize>,
}

fn draw_topic(rng: &mut Rng, trap: bool, map_err: f64, na: usize, flat: bool) -> (Vec<f64>, [usize; 3]) {
    if flat {
        let mut kw = [0usize; 3];
        for k in kw.iter_mut() {
            *k = rng.below(na);
        }
        return (vec![0.55; na], kw);
    }
    let best = rng.below(na);
    let mut second = rng.below(na);
    while second == best {
        second = rng.below(na);
    }
    let mut row = vec![0.0; na];
    for a in 0..na {
        row[a] = if a == best {
            0.85
        } else if a == second {
            0.5
        } else {
            0.15 + 0.25 * rng.f64()
        };
    }
    let mut kw = [0usize; 3];
    for k in kw.iter_mut() {
        *k = if trap {
            loop {
                let a = rng.below(na);
                if a != best && a != second {
                    break a;
                }
            }
        } else if rng.f64() < map_err {
            rng.below(na)
        } else {
            best
        };
    }
    (row, kw)
}

fn make_skills(rng: &mut Rng, s: &TopicSpec, na: usize) -> Skills {
    let t_n = s.topics;
    let mut order: Vec<usize> = (0..t_n).collect();
    for i in (1..t_n).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let mut p = vec![];
    let mut kw = HashMap::new();
    for t in 0..t_n {
        let trap = order.iter().take(s.traps).any(|&x| x == t);
        let (row, k3) = draw_topic(rng, trap, s.map_err, na, s.flat);
        p.push(row);
        for j in 0..3 {
            kw.insert(format!("t{t}k{j}"), k3[j]);
        }
    }
    Skills { p, kw }
}

/// Child skills: each topic is redrawn wholesale with probability `rho`.
fn derive_skills(parent: &Skills, rho: f64, rng: &mut Rng, s: &TopicSpec, na: usize) -> Skills {
    let mut sk = parent.clone();
    let trap_p = s.traps as f64 / s.topics as f64;
    for t in 0..s.topics {
        if rng.f64() < rho {
            let trap = rng.f64() < trap_p;
            let (row, k3) = draw_topic(rng, trap, s.map_err, na, s.flat);
            sk.p[t] = row;
            for j in 0..3 {
                sk.kw.insert(format!("t{t}k{j}"), k3[j]);
            }
        }
    }
    sk
}

fn make_stream(sk: &Skills, rng: &mut Rng, s: &TopicSpec, n: usize) -> World {
    let t_n = s.topics;
    let na = sk.p[0].len();
    let zw: Vec<f64> = (0..t_n).map(|i| if s.zipf { 1.0 / (i as f64 + 1.0) } else { 1.0 }).collect();
    let zs: f64 = zw.iter().sum();
    let (mut tasks, mut ty, mut pp, mut u) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let mut x = rng.f64() * zs;
        let mut t = t_n - 1;
        for (j, w) in zw.iter().enumerate() {
            if x < *w {
                t = j;
                break;
            }
            x -= w;
        }
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(s.confound) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        ty.push(t);
        pp.push(sk.p[t].clone());
        u.push(rng.f64());
    }
    World { agents: na, tasks, ty, p: pp, u, kwmap: sk.kw.clone() }
}

fn rng_for(seed: u64, salt: u64) -> Rng {
    Rng::new(seed.wrapping_mul(7919).wrapping_add(11).wrapping_add(salt.wrapping_mul(0x9E37_79B9)))
}

fn topic_world(seed: u64, s: &TopicSpec, n: usize) -> World {
    let mut r = rng_for(seed, 1);
    let sk = make_skills(&mut r, s, NA);
    make_stream(&sk, &mut r, s, n)
}

fn specs() -> Vec<TopicSpec> {
    vec![
        TopicSpec { name: "traps8", topics: 8, traps: 2, map_err: 0.10, confound: 0.25, zipf: false, flat: false },
        TopicSpec { name: "many16", topics: 16, traps: 2, map_err: 0.10, confound: 0.25, zipf: false, flat: false },
        TopicSpec { name: "correct8", topics: 8, traps: 0, map_err: 0.0, confound: 0.25, zipf: false, flat: false },
    ]
}

// ================================================================== routers
fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

/// x1_lab frozen tuned config of guarded(hebr) (tuned on seeds 1-8 there).
fn x1_guard() -> GuardConfig {
    GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
}

fn build_base(b: usize, kw: &HashMap<String, usize>, na: usize) -> Box<dyn Router> {
    match b {
        0 => Box::new(KeywordRouter::new(kw.clone(), na)),
        1 => Box::new(EpsGreedyRouter::new(na, 0.95, 0.02)),
        2 => Box::new(PooledThompsonRouter::new(na, 1.0)),
        3 => Box::new(HebbianRouter::new(na, heb_robust_cfg())),
        4 => Box::new(Warm::new(HebbianRouter::new(na, heb_robust_cfg()), kw.clone(), na, 1.0).with_evidence(Evidence::MeanFeature)),
        5 => Box::new(Guarded::new(HebbianRouter::new(na, heb_robust_cfg()), kw.clone(), na, x1_guard())),
        _ => unreachable!(),
    }
}

/// Boxed router usable wherever a concrete `Router` is required.
struct Dyn(Box<dyn Router>);
impl Router for Dyn {
    fn name(&self) -> String {
        self.0.name()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        self.0.select(task, candidates, rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.0.update(task, agent, reward)
    }
}

/// Run `r` through `w`; returns the per-task expected success p(chosen).
/// `add`: (t0, new_agent): agent `new_agent` becomes a candidate from task t0 on;
/// `on_add` is called once right before the first task that may use it.
fn trace(r: &mut dyn Router, w: &World, seed: u64, add: Option<(usize, usize)>, on_add: &mut dyn FnMut(&mut dyn Router)) -> Vec<f64> {
    let na = w.agents;
    let mut cands: Vec<usize> = (0..na).collect();
    if let Some((_, new)) = add {
        cands.retain(|&a| a != new);
    }
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut out = Vec::with_capacity(w.tasks.len());
    for (i, task) in w.tasks.iter().enumerate() {
        if let Some((t0, new)) = add
            && i == t0
        {
            cands.push(new);
            on_add(r);
        }
        let a = r.select(task, &cands, &mut rng);
        let p = w.p[i][a];
        out.push(p);
        r.update(task, a, if w.u[i] < p { 1.0 } else { 0.0 });
    }
    out
}

fn prefix_means(tr: &[f64]) -> Vec<f64> {
    HORIZONS.iter().map(|&h| tr[..h.min(tr.len())].iter().sum::<f64>() / h.min(tr.len()) as f64).collect()
}

fn window_mean(tr: &[f64], lo: usize, len: usize) -> f64 {
    let hi = (lo + len).min(tr.len());
    tr[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
}

// ============================================================ generic harness
#[derive(Clone, Debug)]
struct CellLabel {
    world: String,
    metric: String,
    control: bool,
}

trait Exp: Sync {
    type Cand: Clone + std::fmt::Debug + Sync + Send;
    fn id(&self) -> &'static str;
    fn labels(&self) -> Vec<CellLabel>;
    fn baselines(&self) -> Vec<String>;
    fn run_base(&self, b: usize, seed: u64) -> Vec<f64>;
    fn run_cand(&self, c: &Self::Cand, seed: u64) -> Vec<f64>;
    fn families(&self, quick: bool) -> Vec<(&'static str, Vec<Self::Cand>)>;
    /// extra fixed candidates derived from the tuned ones: (name, cand, in_acceptance_family)
    fn extras(&self, _tuned: &[(&'static str, Self::Cand)]) -> Vec<(String, Self::Cand, bool)> {
        vec![]
    }
    fn ablations(&self) -> Vec<(String, String)> {
        vec![]
    }
}

#[derive(Clone)]
struct Rec {
    exp: String,
    cand: String,
    world: String,
    metric: String,
    control: bool,
    best_base: String,
    diff: Summary,
    wins: usize,
    losses: usize,
    p: f64,
    holm_exp: f64,
    holm_global: f64,
    kind: &'static str, // "accept" | "ablation"
}

fn mean_of(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// matrix[seed][cell]
fn matrix<F: Fn(u64) -> Vec<f64> + Sync>(seeds: &[u64], f: F) -> Vec<Vec<f64>> {
    seeds.par_iter().map(|&s| f(s)).collect()
}

fn col(m: &[Vec<f64>], c: usize) -> Vec<f64> {
    m.iter().map(|r| r[c]).collect()
}

fn cell_means(m: &[Vec<f64>]) -> Vec<f64> {
    let nc = m[0].len();
    (0..nc).map(|c| mean_of(&col(m, c))).collect()
}

struct ExpOut {
    recs: Vec<Rec>,
    tuned: Vec<(String, String)>,
    cost_note: Vec<Value>,
}

fn run_exp<E: Exp>(e: &E, tune_seeds: &[u64], held: &[u64], quick: bool) -> ExpOut {
    let labels = e.labels();
    let nc = labels.len();
    let bnames = e.baselines();
    println!("\n##### experiment {} : {} cells, {} baselines", e.id(), nc, bnames.len());
    // ---------------- tuning on seeds 1-8
    let tb: Vec<Vec<f64>> = (0..bnames.len()).map(|b| cell_means(&matrix(tune_seeds, |s| e.run_base(b, s)))).collect();
    let best_tb: Vec<f64> = (0..nc).map(|c| tb.iter().map(|x| x[c]).fold(f64::NEG_INFINITY, f64::max)).collect();
    let mut tuned: Vec<(&'static str, E::Cand)> = vec![];
    let mut tuned_json = vec![];
    for (fam, grid) in e.families(quick) {
        let n = grid.len();
        let scored: Vec<(f64, E::Cand)> = grid
            .into_par_iter()
            .map(|c| {
                let m = cell_means(&matrix(tune_seeds, |s| e.run_cand(&c, s)));
                // control cells (where the mechanism should not help) are excluded from tuning
                let d: Vec<f64> = (0..nc).filter(|&i| !labels[i].control).map(|i| m[i] - best_tb[i]).collect();
                let o = mean_of(&d) + d.iter().map(|x| x.min(0.0)).sum::<f64>() / d.len() as f64;
                (o, c)
            })
            .collect();
        let mut scored = scored;
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        if std::env::var("F1_VERBOSE").is_ok() {
            for (o, c) in scored.iter().take(std::env::var("F1_TOP").ok().and_then(|x| x.parse().ok()).unwrap_or(6)) {
                println!("    [{fam}] obj {o:+.5} {c:?}");
            }
        }
        let best = scored.into_iter().next().unwrap();
        println!("[tune 1-8] {fam}: {n} configs, best regret-averse obj {:+.5} -> {:?}", best.0, best.1);
        tuned_json.push((fam.to_string(), format!("{:?}  (tune obj {:+.5})", best.1, best.0)));
        tuned.push((fam, best.1));
    }
    let extras = e.extras(&tuned);
    let mut cands: Vec<(String, E::Cand, bool)> = tuned.iter().map(|(n, c)| (n.to_string(), c.clone(), true)).collect();
    for (n, c, acc) in extras {
        tuned_json.push((n.clone(), format!("{:?}", c)));
        cands.push((n, c, acc));
    }
    // ---------------- held-out evaluation
    let bm: Vec<Vec<Vec<f64>>> = (0..bnames.len()).map(|b| matrix(held, |s| e.run_base(b, s))).collect();
    let cm: Vec<Vec<Vec<f64>>> = cands.iter().map(|(_, c, _)| matrix(held, |s| e.run_cand(c, s))).collect();
    let mut recs: Vec<Rec> = vec![];
    for c in 0..nc {
        let bcols: Vec<Vec<f64>> = bm.iter().map(|m| col(m, c)).collect();
        let bi = (0..bcols.len()).max_by(|&a, &b| mean_of(&bcols[a]).partial_cmp(&mean_of(&bcols[b])).unwrap()).unwrap();
        for (ci, (cn, _, acc)) in cands.iter().enumerate() {
            if !*acc {
                continue;
            }
            let cv = col(&cm[ci], c);
            let d = paired_diff(&cv, &bcols[bi]);
            recs.push(Rec {
                exp: e.id().into(),
                cand: cn.clone(),
                world: labels[c].world.clone(),
                metric: labels[c].metric.clone(),
                control: labels[c].control,
                best_base: bnames[bi].clone(),
                wins: d.wins,
                losses: d.losses,
                diff: d.diff,
                p: paired_t_pvalue(&cv, &bcols[bi]),
                holm_exp: 1.0,
                holm_global: 1.0,
                kind: "accept",
            });
        }
    }
    let adj = holm_adjust(&recs.iter().map(|r| r.p).collect::<Vec<_>>());
    for (r, a) in recs.iter_mut().zip(adj) {
        r.holm_exp = a;
    }
    // ablations: cand a vs cand b (descriptive, own Holm)
    let mut abl: Vec<Rec> = vec![];
    for (an, bn) in e.ablations() {
        let (Some(ai), Some(bi)) = (cands.iter().position(|x| x.0 == an), cands.iter().position(|x| x.0 == bn)) else { continue };
        for c in 0..nc {
            let (av, bv) = (col(&cm[ai], c), col(&cm[bi], c));
            let d = paired_diff(&av, &bv);
            abl.push(Rec {
                exp: e.id().into(),
                cand: format!("{an} - {bn}"),
                world: labels[c].world.clone(),
                metric: labels[c].metric.clone(),
                control: labels[c].control,
                best_base: bn.clone(),
                wins: d.wins,
                losses: d.losses,
                diff: d.diff,
                p: paired_t_pvalue(&av, &bv),
                holm_exp: 1.0,
                holm_global: 1.0,
                kind: "ablation",
            });
        }
    }
    let adj = holm_adjust(&abl.iter().map(|r| r.p).collect::<Vec<_>>());
    for (r, a) in abl.iter_mut().zip(adj) {
        r.holm_exp = a;
    }
    recs.extend(abl);
    // mean table for the report
    let mut cost_note = vec![];
    for (c, l) in labels.iter().enumerate() {
        let mut row = serde_json::Map::new();
        for (b, n) in bnames.iter().enumerate() {
            row.insert(n.clone(), json!(mean_of(&col(&bm[b], c))));
        }
        for (ci, (n, _, _)) in cands.iter().enumerate() {
            row.insert(format!("CAND:{n}"), json!(mean_of(&col(&cm[ci], c))));
        }
        cost_note.push(json!({"world": l.world, "metric": l.metric, "control": l.control, "means": row}));
    }
    ExpOut { recs, tuned: tuned_json, cost_note }
}

fn verdict(r: &Rec) -> &'static str {
    let sig = r.holm_exp < 0.05;
    if sig && r.diff.mean > 0.0 {
        "WIN"
    } else if sig {
        "LOSS"
    } else if r.diff.lo > 0.0 {
        "raw+"
    } else if r.diff.hi < 0.0 {
        "raw-"
    } else {
        "ns"
    }
}

fn print_summary(recs: &[Rec]) {
    let mut by: BTreeMap<(String, String, bool), Vec<&Rec>> = BTreeMap::new();
    for r in recs.iter().filter(|r| r.kind == "accept") {
        by.entry((r.cand.clone(), r.world.clone(), r.control)).or_default().push(r);
    }
    println!("\n--- summary per (candidate, world): cells, mean diff vs best fair baseline, Holm(exp) WIN/LOSS, raw CI>0 / <0 ---");
    for ((c, w, ctl), v) in &by {
        let md = mean_of(&v.iter().map(|r| r.diff.mean).collect::<Vec<_>>());
        let win = v.iter().filter(|r| verdict(r) == "WIN").count();
        let loss = v.iter().filter(|r| verdict(r) == "LOSS").count();
        let rp = v.iter().filter(|r| r.diff.lo > 0.0).count();
        let rm = v.iter().filter(|r| r.diff.hi < 0.0).count();
        println!("{:<26} {:<24}{} n={:<3} mean {:+.4}  WIN {:<3} LOSS {:<3} (raw+ {:<3} raw- {:<3})", c, w, if *ctl { " [ctl]" } else { "      " }, v.len(), md, win, loss, rp, rm);
    }
}

fn print_tests(recs: &[Rec], only_notable: bool) {
    println!("\n--- tests (cand vs best fair baseline; Holm over family) ---");
    for r in recs {
        let v = verdict(r);
        let notable = v == "WIN" || v == "LOSS" || r.control;
        if only_notable && !notable {
            continue;
        }
        println!(
            "{:<24} {:<20} {:<16} {:<8} vs {:<14} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.4} {}{}",
            r.cand, r.world, r.metric, r.kind, r.best_base, r.diff.mean, r.diff.lo, r.diff.hi, r.wins, r.losses, r.holm_exp, v,
            if r.control { " [ctl]" } else { "" }
        );
    }
}

fn rec_json(r: &Rec) -> Value {
    json!({"exp": r.exp, "cand": r.cand, "world": r.world, "metric": r.metric, "control": r.control, "kind": r.kind,
        "best_baseline": r.best_base, "diff": r.diff, "wins": r.wins, "losses": r.losses, "p_raw": r.p,
        "holm_exp": r.holm_exp, "holm_global": r.holm_global, "verdict": verdict(r)})
}

// ================================================================== A1: transfer
/// Transfer guard: follow a (parent-derived) feature->agent map while (a) a GLOBAL trust
/// gate (discounted success rate of map-following outcomes) and (b) a per-feature UCB
/// both stay above their thresholds; otherwise defer to the inner router. The global gate
/// is the "is this child like its parent?" probe; it recovers (discounts toward its prior)
/// even while disabled. Inner router is trained on every outcome.
struct MapGuard<R: Router> {
    inner: R,
    map: HashMap<String, usize>,
    tau: f64,
    z: f64,
    ka: f64,
    kb: f64,
    disc: f64,
    g_tau: f64,
    g_a: f64,
    g_b: f64,
    counts: HashMap<String, Vec<(f64, f64)>>,
    gs: f64,
    gf: f64,
    last_k: Option<usize>,
    follows: u64,
}

fn ucb_of(a: f64, b: f64, z: f64) -> f64 {
    let n = a + b;
    let m = a / n;
    m + z * (m * (1.0 - m) / (n + 1.0)).sqrt()
}

impl<R: Router> MapGuard<R> {
    fn new(inner: R, map: HashMap<String, usize>, tau: f64, ka: f64, kb: f64, g_tau: f64) -> Self {
        MapGuard { inner, map, tau, z: 0.5, ka, kb, disc: 0.97, g_tau, g_a: 3.0, g_b: 1.0, counts: HashMap::new(), gs: 0.0, gf: 0.0, last_k: None, follows: 0 }
    }
}

impl<R: Router> Router for MapGuard<R> {
    fn name(&self) -> String {
        "map-guard".into()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        self.last_k = None;
        let mut votes: BTreeMap<usize, usize> = BTreeMap::new();
        for f in &task.features {
            if let Some(&a) = self.map.get(f)
                && candidates.contains(&a)
            {
                *votes.entry(a).or_insert(0) += 1;
            }
        }
        let k = votes.iter().fold(None, |b: Option<(usize, usize)>, (&a, &v)| match b {
            Some((_, bv)) if bv >= v => b,
            _ => Some((a, v)),
        });
        if let Some((k, _)) = k {
            let g_ok = self.g_tau <= 0.0 || ucb_of(self.g_a + self.gs, self.g_b + self.gf, self.z) >= self.g_tau;
            let (mut s, mut f) = (0.0, 0.0);
            for ft in &task.features {
                if let Some(&(cs, cf)) = self.counts.get(ft).and_then(|r| r.get(k)) {
                    s += cs;
                    f += cf;
                }
            }
            let n = task.features.len().max(1) as f64;
            let f_ok = ucb_of(self.ka + s / n, self.kb + f / n, self.z) >= self.tau;
            if g_ok && f_ok {
                self.last_k = Some(k);
                self.follows += 1;
                return k;
            }
        }
        self.inner.select(task, candidates, rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        self.gs *= self.disc;
        self.gf *= self.disc;
        if self.last_k == Some(agent) {
            self.gs += r;
            self.gf += 1.0 - r;
        }
        for ft in &task.features {
            let row = self.counts.entry(ft.clone()).or_insert_with(|| vec![(0.0, 0.0); agent + 1]);
            if row.len() <= agent {
                row.resize(agent + 1, (0.0, 0.0));
            }
            for c in row.iter_mut() {
                c.0 *= self.disc;
                c.1 *= self.disc;
            }
            row[agent].0 += r;
            row[agent].1 += 1.0 - r;
        }
        self.inner.update(task, agent, reward);
    }
}

struct Parent {
    skills: Skills,
    router: HebbianRouter,
}

struct XferExp {
    worlds: Vec<TopicSpec>,
    rhos: Vec<f64>,
    np: usize,
    cache: Mutex<HashMap<(usize, u64), Arc<Parent>>>,
}

#[derive(Clone, Debug)]
enum XCand {
    /// clone the parent's whole learned hebbian state into warm(hebr) with hand-over scale k
    Naive { k: f64 },
    /// guarded(parent map) over guarded(keyword) over hebbian (cold or shrunk-inherited)
    GuardMap { margin: f64, tau: f64, ka: f64, kb: f64, g_tau: f64, disagree: bool, shrink: Option<f64> },
}

fn shrink_hebb(h: &HebbianRouter, s: f64) -> HebbianRouter {
    let mut v = serde_json::to_value(h).unwrap();
    if let Some(w) = v.get_mut("weights").and_then(|x| x.as_object_mut()) {
        for (_, row) in w.iter_mut() {
            for x in row.as_array_mut().unwrap() {
                let f = x.as_f64().unwrap();
                *x = json!(INIT_WEIGHT + s * (f - INIT_WEIGHT));
            }
        }
    }
    serde_json::from_value(v).unwrap()
}

fn parent_map(h: &HebbianRouter, margin: f64) -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for (f, row) in h.rows() {
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        let (b1, b2) = (row[idx[0]], row[idx[1]]);
        if b1 > 1e-9 && (b1 - b2) >= margin * b1 {
            m.insert(f.clone(), idx[0]);
        }
    }
    m
}

impl XferExp {
    fn parent(&self, wi: usize, seed: u64) -> Arc<Parent> {
        if let Some(p) = self.cache.lock().unwrap().get(&(wi, seed)) {
            return p.clone();
        }
        let s = &self.worlds[wi];
        let mut r = rng_for(seed, 100 + wi as u64);
        let skills = make_skills(&mut r, s, NA);
        let pw = make_stream(&skills, &mut r, s, self.np);
        let mut w = Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), skills.kw.clone(), NA, 1.0).with_evidence(Evidence::MeanFeature);
        trace(&mut w, &pw, seed ^ 0xABCD, None, &mut |_| {});
        let p = Arc::new(Parent { skills, router: w.inner.clone() });
        self.cache.lock().unwrap().insert((wi, seed), p.clone());
        p
    }

    fn child(&self, wi: usize, ri: usize, seed: u64) -> (World, Arc<Parent>) {
        let par = self.parent(wi, seed);
        let s = &self.worlds[wi];
        let mut r = rng_for(seed, 200 + (wi * 10 + ri) as u64);
        let sk = derive_skills(&par.skills, self.rhos[ri], &mut r, s, NA);
        (make_stream(&sk, &mut r, s, N_TASKS), par)
    }
}

impl Exp for XferExp {
    type Cand = XCand;
    fn id(&self) -> &'static str {
        "A1-transfer"
    }
    fn labels(&self) -> Vec<CellLabel> {
        let mut v = vec![];
        for w in &self.worlds {
            for &r in &self.rhos {
                for h in HORIZONS {
                    v.push(CellLabel { world: w.name.into(), metric: format!("rho{r}|H{h}"), control: r >= 1.0 || w.traps == 0 });
                }
            }
        }
        v
    }
    fn baselines(&self) -> Vec<String> {
        BASES.iter().map(|s| s.to_string()).collect()
    }
    fn run_base(&self, b: usize, seed: u64) -> Vec<f64> {
        let mut out = vec![];
        for wi in 0..self.worlds.len() {
            for ri in 0..self.rhos.len() {
                let (w, _) = self.child(wi, ri, seed);
                let mut r = build_base(b, &w.kwmap, NA);
                out.extend(prefix_means(&trace(r.as_mut(), &w, seed, None, &mut |_| {})));
            }
        }
        out
    }
    fn run_cand(&self, c: &XCand, seed: u64) -> Vec<f64> {
        let mut out = vec![];
        for wi in 0..self.worlds.len() {
            for ri in 0..self.rhos.len() {
                let (w, par) = self.child(wi, ri, seed);
                let mut r: Box<dyn Router> = match c {
                    XCand::Naive { k } => Box::new(Warm::new(par.router.clone(), w.kwmap.clone(), NA, *k).with_evidence(Evidence::MeanFeature)),
                    XCand::GuardMap { margin, tau, ka, kb, g_tau, disagree, shrink } => {
                        let inner = match shrink {
                            Some(s) => shrink_hebb(&par.router, *s),
                            None => HebbianRouter::new(NA, heb_robust_cfg()),
                        };
                        let mid = Guarded::new(inner, w.kwmap.clone(), NA, x1_guard());
                        let mut pm = parent_map(&par.router, *margin);
                        if *disagree {
                            pm.retain(|f, a| w.kwmap.get(f) != Some(a));
                        }
                        Box::new(MapGuard::new(Dyn(Box::new(mid)), pm, *tau, *ka, *kb, *g_tau))
                    }
                };
                out.extend(prefix_means(&trace(r.as_mut(), &w, seed, None, &mut |_| {})));
            }
        }
        out
    }
    fn families(&self, quick: bool) -> Vec<(&'static str, Vec<XCand>)> {
        let mut naive = vec![];
        for k in [0.1, 0.5, 2.0] {
            naive.push(XCand::Naive { k });
        }
        let mut gm = vec![];
        let mut gs = vec![];
        for margin in [0.0, 0.1, 0.3] {
            for tau in [0.55, 0.7] {
                for (ka, kb) in [(2.0, 1.0), (4.0, 1.0)] {
                    for g_tau in [0.0, 0.7, 0.8, 0.9] {
                        for disagree in [false, true] {
                            gm.push(XCand::GuardMap { margin, tau, ka, kb, g_tau, disagree, shrink: None });
                            for sh in [0.3, 0.7] {
                                gs.push(XCand::GuardMap { margin, tau, ka, kb, g_tau, disagree, shrink: Some(sh) });
                            }
                        }
                    }
                }
            }
        }
        if quick {
            naive.truncate(1);
            gm.truncate(2);
            gs.truncate(2);
        }
        vec![("xfer-naive", naive), ("xfer-guard-map", gm), ("xfer-guard-map+inner", gs)]
    }
    fn ablations(&self) -> Vec<(String, String)> {
        vec![
            ("xfer-guard-map".into(), "xfer-naive".into()),
            ("xfer-guard-map+inner".into(), "xfer-guard-map".into()),
        ]
    }
}

// ================================================================== A2: agent added
#[derive(Clone, Copy, Debug, PartialEq)]
enum NewKind {
    /// new agent = declared sibling j with a +U(.05,.20) upgrade on every topic
    Upgrade,
    /// new agent = sibling j, but 2 random topics become 0.9 (specialist)
    Specialist,
    /// new agent unrelated to the declared sibling (declaration is wrong): control
    Unrelated,
}

struct AddExp {
    worlds: Vec<TopicSpec>,
    kinds: Vec<NewKind>,
    t0: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Sub {
    Guarded,
    Warm,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum InhMode {
    /// attach the new agent to the declared sibling's slot
    Declared,
    /// attach it to a wrong (random other) agent's slot (control)
    RandomSibling,
}

#[derive(Clone, Debug)]
struct ACand {
    kappa: f64,
    mode: InhMode,
    sub: Sub,
}

const A2_WINDOWS: [usize; 4] = [100, 250, 500, 1000];

fn task_keys(task: &Task) -> Vec<String> {
    let mut k: Vec<String> = task.features.clone();
    if let Some(t) = &task.task_type {
        k.push(format!("type:{t}"));
    }
    if k.is_empty() {
        k.push("no-keyword".into());
    }
    k
}

/// Fork-slot wrapper. The base router never learns about the new agent: whenever it
/// picks the declared sibling `sib`, a 2-arm pooled Thompson (sib vs new) decides who
/// actually runs; outcomes of either are fed to the base as sibling outcomes. The new
/// arm's prior is `kappa` x the sibling's accumulated per-feature evidence (inheritance);
/// kappa = 0 is the cold (uninformative Beta(1,1)) version of the same mechanism.
struct ForkSwap<R: Router> {
    inner: R,
    sib: usize,
    new: usize,
    kappa: f64,
    inited: bool,
    /// feature -> [(s,f) of sibling, (s,f) of new]
    counts: BTreeMap<String, [(f64, f64); 2]>,
}

impl<R: Router> ForkSwap<R> {
    fn new(inner: R, sib: usize, new: usize, kappa: f64) -> Self {
        ForkSwap { inner, sib, new, kappa, inited: false, counts: BTreeMap::new() }
    }
    fn post(&self, task: &Task, arm: usize) -> (f64, f64) {
        let (mut s, mut f) = (0.0, 0.0);
        for k in task_keys(task) {
            if let Some(c) = self.counts.get(&k) {
                s += c[arm].0;
                f += c[arm].1;
            }
        }
        (1.0 + s, 1.0 + f)
    }
}

impl<R: Router> Router for ForkSwap<R> {
    fn name(&self) -> String {
        "fork-swap".into()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let has_new = candidates.contains(&self.new);
        let base_c: Vec<usize> = candidates.iter().copied().filter(|&a| a != self.new).collect();
        let a = self.inner.select(task, &base_c, rng);
        if !has_new || a != self.sib {
            return a;
        }
        if !self.inited {
            self.inited = true;
            for c in self.counts.values_mut() {
                c[1] = (self.kappa * c[0].0, self.kappa * c[0].1);
            }
        }
        let (s0, f0) = self.post(task, 0);
        let (s1, f1) = self.post(task, 1);
        let v0 = neural_mesh::baselines::sample_beta(s0, f0, rng);
        let v1 = neural_mesh::baselines::sample_beta(s1, f1, rng);
        if v1 > v0 { self.new } else { self.sib }
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        let r = reward.clamp(0.0, 1.0);
        if agent == self.sib || agent == self.new {
            let arm = if agent == self.new { 1 } else { 0 };
            for k in task_keys(task) {
                let c = self.counts.entry(k).or_insert([(0.0, 0.0); 2]);
                c[arm].0 += r;
                c[arm].1 += 1.0 - r;
            }
            self.inner.update(task, self.sib, reward);
        } else {
            self.inner.update(task, agent, reward);
        }
    }
}

impl AddExp {
    /// world with NA+1 agents; returns (world, declared sibling j, random other agent)
    fn world(&self, wi: usize, ki: usize, seed: u64) -> (World, usize, usize) {
        let s = &self.worlds[wi];
        let mut r = rng_for(seed, 300 + (wi * 10 + ki) as u64);
        let mut sk = make_skills(&mut r, s, NA);
        let j = r.below(NA);
        let mut other = r.below(NA - 1);
        if other >= j {
            other += 1;
        }
        let spec_topics: Vec<usize> = {
            let a = r.below(s.topics);
            let mut b = r.below(s.topics);
            while b == a {
                b = r.below(s.topics);
            }
            vec![a, b]
        };
        for t in 0..s.topics {
            let pj = sk.p[t][j];
            let newp = match self.kinds[ki] {
                NewKind::Upgrade => (pj + 0.05 + 0.15 * r.f64()).min(0.97),
                NewKind::Specialist => {
                    if spec_topics.contains(&t) {
                        0.9
                    } else {
                        pj
                    }
                }
                NewKind::Unrelated => {
                    if r.f64() < 0.17 {
                        0.85
                    } else {
                        0.15 + 0.35 * r.f64()
                    }
                }
            };
            sk.p[t].push(newp);
        }
        (make_stream(&sk, &mut r, s, N_TASKS), j, other)
    }
    fn metrics(&self, tr: &[f64]) -> Vec<f64> {
        A2_WINDOWS.iter().map(|&h| window_mean(tr, self.t0, h)).collect()
    }
}

impl Exp for AddExp {
    type Cand = ACand;
    fn id(&self) -> &'static str {
        "A2-agent-added"
    }
    fn labels(&self) -> Vec<CellLabel> {
        let mut v = vec![];
        for w in &self.worlds {
            for k in &self.kinds {
                for h in A2_WINDOWS {
                    v.push(CellLabel { world: format!("{}/{:?}", w.name, k), metric: format!("post-add H{h}"), control: *k == NewKind::Unrelated });
                }
            }
        }
        v
    }
    fn baselines(&self) -> Vec<String> {
        BASES.iter().map(|s| s.to_string()).collect()
    }
    fn run_base(&self, b: usize, seed: u64) -> Vec<f64> {
        let mut out = vec![];
        for wi in 0..self.worlds.len() {
            for ki in 0..self.kinds.len() {
                let (w, _, _) = self.world(wi, ki, seed);
                let mut r = build_base(b, &w.kwmap, NA + 1);
                let tr = trace(r.as_mut(), &w, seed, Some((self.t0, NA)), &mut |_| {});
                out.extend(self.metrics(&tr));
            }
        }
        out
    }
    fn run_cand(&self, c: &ACand, seed: u64) -> Vec<f64> {
        let mut out = vec![];
        for wi in 0..self.worlds.len() {
            for ki in 0..self.kinds.len() {
                let (w, j, other) = self.world(wi, ki, seed);
                let sib = if c.mode == InhMode::Declared { j } else { other };
                let na = NA + 1;
                let base: Box<dyn Router> = match c.sub {
                    Sub::Guarded => build_base(5, &w.kwmap, na),
                    Sub::Warm => build_base(4, &w.kwmap, na),
                };
                let mut r = ForkSwap::new(Dyn(base), sib, NA, c.kappa);
                let tr = trace(&mut r, &w, seed, Some((self.t0, NA)), &mut |_| {});
                out.extend(self.metrics(&tr));
            }
        }
        out
    }
    fn families(&self, quick: bool) -> Vec<(&'static str, Vec<ACand>)> {
        let mut g = vec![];
        for sub in [Sub::Guarded, Sub::Warm] {
            for kappa in [0.1, 0.25, 0.5, 1.0] {
                g.push(ACand { kappa, mode: InhMode::Declared, sub });
            }
        }
        if quick {
            g.truncate(2);
        }
        vec![("fork-inherit", g)]
    }
    fn extras(&self, tuned: &[(&'static str, ACand)]) -> Vec<(String, ACand, bool)> {
        let t = &tuned[0].1;
        vec![
            ("fork-wrong-sibling".into(), ACand { kappa: t.kappa, mode: InhMode::RandomSibling, sub: t.sub }, false),
            ("fork-cold(kappa=0)".into(), ACand { kappa: 0.0, mode: InhMode::Declared, sub: t.sub }, false),
        ]
    }
    fn ablations(&self) -> Vec<(String, String)> {
        vec![
            ("fork-inherit".into(), "fork-cold(kappa=0)".into()),
            ("fork-inherit".into(), "fork-wrong-sibling".into()),
        ]
    }
}

// ================================================================== B: ordering
#[derive(Clone, Copy, Debug, PartialEq)]
enum Agg {
    Mean,
    Min,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Policy {
    Fifo,
    /// least-seen features first (coverage curriculum), epsilon-random mix
    Coverage { agg: Agg, eps: f64 },
    /// most-seen features first (anti-curriculum control)
    Anti { agg: Agg },
}

#[derive(Clone, Debug)]
struct BCand {
    learner: usize, // index into BASES
    policy: Policy,
}

struct OrderExp {
    worlds: Vec<TopicSpec>,
    pools: Vec<usize>,
}

fn run_ordered(w: &World, r: &mut dyn Router, pol: Policy, n: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut prng = Rng::new(seed.wrapping_mul(31).wrapping_add(5));
    let cands: Vec<usize> = (0..w.agents).collect();
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut seen: HashMap<&str, f64> = HashMap::new();
    let mut tot = 0.0;
    while !remaining.is_empty() {
        let (agg, sign, eps) = match pol {
            Policy::Fifo => (Agg::Mean, 0.0, 0.0),
            Policy::Coverage { agg, eps } => (agg, 1.0, eps),
            Policy::Anti { agg } => (agg, -1.0, 0.0),
        };
        let pick = if sign == 0.0 {
            0
        } else if eps > 0.0 && prng.f64() < eps {
            prng.below(remaining.len())
        } else {
            let mut best = (f64::INFINITY, 0usize);
            for (pos, &ti) in remaining.iter().enumerate() {
                let fs = &w.tasks[ti].features;
                if fs.is_empty() {
                    continue;
                }
                let counts = fs.iter().map(|f| seen.get(f.as_str()).copied().unwrap_or(0.0));
                let sc = match agg {
                    Agg::Mean => counts.sum::<f64>() / fs.len() as f64,
                    Agg::Min => counts.fold(f64::INFINITY, f64::min),
                };
                if sign * sc < best.0 {
                    best = (sign * sc, pos);
                }
            }
            best.1
        };
        let ti = remaining.remove(pick);
        let task = &w.tasks[ti];
        let a = r.select(task, &cands, &mut rng);
        let p = w.p[ti][a];
        tot += p;
        r.update(task, a, if w.u[ti] < p { 1.0 } else { 0.0 });
        for f in &task.features {
            *seen.entry(f.as_str()).or_insert(0.0) += 1.0;
        }
    }
    tot / n as f64
}

impl OrderExp {
    fn world(&self, wi: usize, seed: u64) -> World {
        topic_world(seed, &self.worlds[wi], *self.pools.last().unwrap())
    }
    fn pool_world(w: &World, n: usize) -> World {
        World { agents: w.agents, tasks: w.tasks[..n].to_vec(), ty: w.ty[..n].to_vec(), p: w.p[..n].to_vec(), u: w.u[..n].to_vec(), kwmap: w.kwmap.clone() }
    }
}

impl Exp for OrderExp {
    type Cand = BCand;
    fn id(&self) -> &'static str {
        "B-ordering"
    }
    fn labels(&self) -> Vec<CellLabel> {
        let mut v = vec![];
        for w in &self.worlds {
            for &n in &self.pools {
                v.push(CellLabel { world: w.name.into(), metric: format!("pool{n}"), control: w.flat || w.traps == 0 });
            }
        }
        v
    }
    fn baselines(&self) -> Vec<String> {
        BASES.iter().map(|s| s.to_string()).collect()
    }
    fn run_base(&self, b: usize, seed: u64) -> Vec<f64> {
        let mut out = vec![];
        for wi in 0..self.worlds.len() {
            let full = self.world(wi, seed);
            for &n in &self.pools {
                let w = Self::pool_world(&full, n);
                let mut r = build_base(b, &w.kwmap, NA);
                out.push(run_ordered(&w, r.as_mut(), Policy::Fifo, n, seed));
            }
        }
        out
    }
    fn run_cand(&self, c: &BCand, seed: u64) -> Vec<f64> {
        let mut out = vec![];
        for wi in 0..self.worlds.len() {
            let full = self.world(wi, seed);
            for &n in &self.pools {
                let w = Self::pool_world(&full, n);
                let mut r = build_base(c.learner, &w.kwmap, NA);
                out.push(run_ordered(&w, r.as_mut(), c.policy, n, seed));
            }
        }
        out
    }
    fn families(&self, quick: bool) -> Vec<(&'static str, Vec<BCand>)> {
        let mut g = vec![];
        for learner in [4usize, 5, 1] {
            for agg in [Agg::Mean, Agg::Min] {
                for eps in [0.0, 0.2] {
                    g.push(BCand { learner, policy: Policy::Coverage { agg, eps } });
                }
            }
        }
        if quick {
            g.truncate(3);
        }
        vec![("coverage-order", g)]
    }
    fn extras(&self, tuned: &[(&'static str, BCand)]) -> Vec<(String, BCand, bool)> {
        let t = &tuned[0].1;
        let agg = match t.policy {
            Policy::Coverage { agg, .. } => agg,
            _ => Agg::Mean,
        };
        vec![
            ("fifo(same learner)".into(), BCand { learner: t.learner, policy: Policy::Fifo }, false),
            ("anti-coverage(control)".into(), BCand { learner: t.learner, policy: Policy::Anti { agg } }, false),
        ]
    }
    fn ablations(&self) -> Vec<(String, String)> {
        vec![
            ("coverage-order".into(), "fifo(same learner)".into()),
            ("coverage-order".into(), "anti-coverage(control)".into()),
        ]
    }
}

// ================================================================== C: multi-objective

struct MoWorld {
    w: World,
    cost: Vec<f64>,
    lat: Vec<f64>,
}

type Lam = (f64, f64);

fn mo_world(seed: u64, s: &TopicSpec, n: usize) -> MoWorld {
    let w = topic_world(seed, s, n);
    let mut r = rng_for(seed, 400);
    // agent quality = mean skill over topics; cost correlates with quality (a real frontier)
    let mut q = vec![0.0; NA];
    for row in w.p.iter() {
        for a in 0..NA {
            q[a] += row[a];
        }
    }
    let qm: Vec<f64> = q.iter().map(|x| x / w.p.len() as f64).collect();
    let (lo, hi) = (qm.iter().cloned().fold(f64::INFINITY, f64::min), qm.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    let cost: Vec<f64> = qm.iter().map(|x| (0.15 + 0.7 * (x - lo) / (hi - lo).max(1e-9) + 0.2 * (r.f64() - 0.5)).clamp(0.05, 1.0)).collect();
    let lat: Vec<f64> = (0..NA).map(|_| 0.1 + 0.8 * r.f64()).collect();
    MoWorld { w, cost, lat }
}

impl MoWorld {
    fn pen(&self, lam: Lam, a: usize) -> f64 {
        lam.0 * self.cost[a] + lam.1 * self.lat[a]
    }
    fn max_pen(&self, lam: Lam) -> f64 {
        (0..NA).map(|a| self.pen(lam, a)).fold(0.0, f64::max)
    }
}

/// Decoupled quality estimator: pooled per-feature discounted means shrunk to a keyword prior.
#[derive(Clone)]
struct QEst {
    gamma: f64,
    m0: f64,
    hi: f64,
    lo: f64,
    kw: HashMap<String, usize>,
    stats: HashMap<String, Vec<(f64, f64)>>,
}

impl QEst {
    fn new(kw: HashMap<String, usize>, gamma: f64, m0: f64, hi: f64, lo: f64) -> Self {
        QEst { gamma, m0, hi, lo, kw, stats: HashMap::new() }
    }
    fn value(&self, task: &Task, a: usize) -> f64 {
        if task.features.is_empty() {
            return self.lo;
        }
        let mut tot = 0.0;
        for f in &task.features {
            let prior = if self.kw.get(f) == Some(&a) { self.hi } else { self.lo };
            let (s, n) = self.stats.get(f).and_then(|v| v.get(a)).copied().unwrap_or((0.0, 0.0));
            tot += (s + self.m0 * prior) / (n + self.m0);
        }
        tot / task.features.len() as f64
    }
    fn update(&mut self, task: &Task, a: usize, r: f64) {
        for f in &task.features {
            let row = self.stats.entry(f.clone()).or_insert_with(|| vec![(0.0, 0.0); NA]);
            let c = &mut row[a];
            c.0 = c.0 * self.gamma + r;
            c.1 = c.1 * self.gamma + 1.0;
        }
    }
}

#[derive(Clone, Debug)]
struct CCand {
    shared: bool,
    gamma: f64,
    m0: f64,
    hi: f64,
    lo: f64,
    eps: f64,
}

#[derive(Clone, Copy, Debug)]
enum Scen {
    Tenants(usize),
    Switch(bool),
    NewTenant,
}

const C_LAMS: [Lam; 4] = [(0.0, 0.0), (0.4, 0.0), (0.0, 0.4), (0.3, 0.3)];
const SW_A: Lam = (0.0, 0.1);
const SW_B: Lam = (0.5, 0.3);
const NEW_LAM: Lam = (0.2, 0.5);
const C_WIN_S2: [usize; 4] = [100, 250, 500, 1000];
const C_WIN_S3: [usize; 3] = [50, 100, 250];

struct MoExp {
    worlds: Vec<TopicSpec>,
}

fn scen_list() -> Vec<(Scen, &'static str, bool)> {
    vec![
        (Scen::Tenants(4), "S1-tenants4", false),
        (Scen::Tenants(1), "S1-single(ctl)", true),
        (Scen::Switch(true), "S2-switch", false),
        (Scen::Switch(false), "S2-noswitch(ctl)", true),
        (Scen::NewTenant, "S3-new-tenant", false),
    ]
}

/// One learner policy: given tenant and task, choose; then learn from the outcome.
trait MoPolicy {
    fn choose(&mut self, tenant: usize, task: &Task, lam: Lam, mw: &MoWorld, rng: &mut Rng, cands: &[usize]) -> usize;
    fn learn(&mut self, tenant: usize, task: &Task, a: usize, o: f64, lam: Lam, mw: &MoWorld);
}

/// Scalarized baselines: one Router per tenant trained on shifted reward.
struct ScalarPol {
    routers: Vec<Box<dyn Router>>,
}
impl MoPolicy for ScalarPol {
    fn choose(&mut self, tenant: usize, task: &Task, _lam: Lam, _mw: &MoWorld, rng: &mut Rng, cands: &[usize]) -> usize {
        self.routers[tenant].select(task, cands, rng)
    }
    fn learn(&mut self, tenant: usize, task: &Task, a: usize, o: f64, lam: Lam, mw: &MoWorld) {
        let mp = mw.max_pen(lam);
        let r = ((o - mw.pen(lam, a) + mp) / (1.0 + mp)).clamp(0.0, 1.0);
        self.routers[tenant].update(task, a, r);
    }
}

/// Decoupled: quality estimator(s) learn binary success; penalties applied at selection.
struct DecPol {
    ests: Vec<QEst>,
    shared: bool,
    eps: f64,
}
impl DecPol {
    fn ix(&self, tenant: usize) -> usize {
        if self.shared { 0 } else { tenant }
    }
}
impl MoPolicy for DecPol {
    fn choose(&mut self, tenant: usize, task: &Task, lam: Lam, mw: &MoWorld, rng: &mut Rng, cands: &[usize]) -> usize {
        if rng.bernoulli(self.eps) {
            return cands[rng.below(cands.len())];
        }
        let e = &self.ests[self.ix(tenant)];
        let mut best = (f64::NEG_INFINITY, cands[0]);
        for &a in cands {
            let u = e.value(task, a) - mw.pen(lam, a);
            if u > best.0 {
                best = (u, a);
            }
        }
        best.1
    }
    fn learn(&mut self, tenant: usize, task: &Task, a: usize, o: f64, _lam: Lam, _mw: &MoWorld) {
        let i = self.ix(tenant);
        self.ests[i].update(task, a, o);
    }
}

/// Keyword baseline: static, cost-blind.
struct KwPol {
    r: KeywordRouter,
}
impl MoPolicy for KwPol {
    fn choose(&mut self, _t: usize, task: &Task, _l: Lam, _m: &MoWorld, rng: &mut Rng, cands: &[usize]) -> usize {
        self.r.select(task, cands, rng)
    }
    fn learn(&mut self, _t: usize, _task: &Task, _a: usize, _o: f64, _l: Lam, _m: &MoWorld) {}
}

impl MoExp {
    fn metric_names(sc: Scen) -> Vec<String> {
        match sc {
            Scen::Tenants(_) => HORIZONS.iter().map(|h| format!("util H{h}")).collect(),
            Scen::Switch(_) => C_WIN_S2.iter().map(|h| format!("post-switch H{h}")).collect(),
            Scen::NewTenant => C_WIN_S3.iter().map(|h| format!("new-tenant H{h}")).collect(),
        }
    }
    fn n_tenants(sc: Scen) -> usize {
        match sc {
            Scen::Tenants(k) => k,
            Scen::Switch(_) => 1,
            Scen::NewTenant => 5,
        }
    }
    fn run_policy(&self, mw: &MoWorld, sc: Scen, seed: u64, pol: &mut dyn MoPolicy) -> Vec<f64> {
        let w = &mw.w;
        let cands: Vec<usize> = (0..NA).collect();
        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
        let n = w.tasks.len();
        let mut util: Vec<f64> = Vec::with_capacity(n);
        let mut aux: Vec<f64> = vec![];
        for t in 0..n {
            let task = &w.tasks[t];
            let (tenant, lam) = match sc {
                Scen::Tenants(k) => (t % k, C_LAMS[t % k]),
                Scen::Switch(sw) => (0, if sw && t >= n / 2 { SW_B } else { SW_A }),
                Scen::NewTenant => {
                    if t >= n / 2 && t % 5 == 4 {
                        (4, NEW_LAM)
                    } else {
                        (t % 4, C_LAMS[t % 4])
                    }
                }
            };
            let a = pol.choose(tenant, task, lam, mw, &mut rng, &cands);
            let p = w.p[t][a];
            let u = p - mw.pen(lam, a);
            util.push(u);
            if matches!(sc, Scen::NewTenant) && tenant == 4 {
                aux.push(u);
            }
            pol.learn(tenant, task, a, if w.u[t] < p { 1.0 } else { 0.0 }, lam, mw);
        }
        match sc {
            Scen::Tenants(_) => HORIZONS.iter().map(|&h| util[..h.min(n)].iter().sum::<f64>() / h.min(n) as f64).collect(),
            Scen::Switch(_) => C_WIN_S2.iter().map(|&h| window_mean(&util, n / 2, h)).collect(),
            Scen::NewTenant => C_WIN_S3.iter().map(|&h| aux[..h.min(aux.len())].iter().sum::<f64>() / h.min(aux.len()) as f64).collect(),
        }
    }
}

impl Exp for MoExp {
    type Cand = CCand;
    fn id(&self) -> &'static str {
        "C-multiobjective"
    }
    fn labels(&self) -> Vec<CellLabel> {
        let mut v = vec![];
        for w in &self.worlds {
            for (sc, nm, ctl) in scen_list() {
                for m in Self::metric_names(sc) {
                    v.push(CellLabel { world: format!("{}/{}", w.name, nm), metric: m, control: ctl || w.traps == 0 });
                }
            }
        }
        v
    }
    fn baselines(&self) -> Vec<String> {
        BASES.iter().map(|s| format!("scalarized:{s}")).collect()
    }
    fn run_base(&self, b: usize, seed: u64) -> Vec<f64> {
        let mut out = vec![];
        for s in &self.worlds {
            let mw = mo_world(seed, s, N_TASKS);
            for (sc, _, _) in scen_list() {
                let nt = Self::n_tenants(sc);
                let mut pol: Box<dyn MoPolicy> = if b == 0 {
                    Box::new(KwPol { r: KeywordRouter::new(mw.w.kwmap.clone(), NA) })
                } else {
                    Box::new(ScalarPol { routers: (0..nt).map(|_| build_base(b, &mw.w.kwmap, NA)).collect() })
                };
                out.extend(self.run_policy(&mw, sc, seed, pol.as_mut()));
            }
        }
        out
    }
    fn run_cand(&self, c: &CCand, seed: u64) -> Vec<f64> {
        let mut out = vec![];
        for s in &self.worlds {
            let mw = mo_world(seed, s, N_TASKS);
            for (sc, _, _) in scen_list() {
                let nt = Self::n_tenants(sc);
                let mk = || QEst::new(mw.w.kwmap.clone(), c.gamma, c.m0, c.hi, c.lo);
                let mut pol = DecPol { ests: (0..if c.shared { 1 } else { nt }).map(|_| mk()).collect(), shared: c.shared, eps: c.eps };
                out.extend(self.run_policy(&mw, sc, seed, &mut pol));
            }
        }
        out
    }
    fn families(&self, quick: bool) -> Vec<(&'static str, Vec<CCand>)> {
        let mut sep = vec![];
        let mut sh = vec![];
        for gamma in [1.0, 0.97] {
            for m0 in [1.0, 3.0] {
                for (hi, lo) in [(0.7, 0.4), (0.8, 0.3), (0.6, 0.5)] {
                    for eps in [0.0, 0.03] {
                        sep.push(CCand { shared: false, gamma, m0, hi, lo, eps });
                        sh.push(CCand { shared: true, gamma, m0, hi, lo, eps });
                    }
                }
            }
        }
        if quick {
            sep.truncate(3);
            sh.truncate(3);
        }
        vec![("decoupled-separate", sep), ("decoupled-shared", sh)]
    }
    fn ablations(&self) -> Vec<(String, String)> {
        vec![("decoupled-shared".into(), "decoupled-separate".into())]
    }
}

// ---- C diagnostic: does the decoupled learner recover the quality/cost/latency Pareto frontier?
fn dominated(i: usize, pts: &[(f64, f64, f64)]) -> bool {
    // maximise q, minimise cost and latency
    (0..pts.len()).any(|j| {
        j != i
            && pts[j].0 >= pts[i].0
            && pts[j].1 <= pts[i].1
            && pts[j].2 <= pts[i].2
            && (pts[j].0 > pts[i].0 || pts[j].1 < pts[i].1 || pts[j].2 < pts[i].2)
    })
}

/// exact 3-D hypervolume (reference q=0, c=1, l=1) of <= ~10 points by inclusion-exclusion
fn hv3(pts: &[(f64, f64, f64)]) -> f64 {
    let n = pts.len();
    let mut tot = 0.0;
    for mask in 1u32..(1u32 << n) {
        let (mut q, mut c, mut l) = (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut k = 0;
        for i in 0..n {
            if mask & (1 << i) != 0 {
                k += 1;
                q = q.min(pts[i].0);
                c = c.min(1.0 - pts[i].1);
                l = l.min(1.0 - pts[i].2);
            }
        }
        let v = q.max(0.0) * c.max(0.0) * l.max(0.0);
        tot += if k % 2 == 1 { v } else { -v };
    }
    tot
}

fn c_diag(held: &[u64], ws: &[TopicSpec]) -> Value {
    let cps = [0usize, 100, 250, 500, 1000, 2000];
    let mut out = vec![];
    for s in ws {
        // per seed, per checkpoint: (hv ratio, recall, precision)
        let per: Vec<Vec<(f64, f64, f64)>> = held
            .par_iter()
            .map(|&seed| {
                let mw = mo_world(seed, s, N_TASKS);
                let w = &mw.w;
                let mut est = QEst::new(w.kwmap.clone(), 0.97, 3.0, 0.8, 0.3);
                let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
                let cands: Vec<usize> = (0..NA).collect();
                let mut res = vec![];
                let measure = |est: &QEst| -> (f64, f64, f64) {
                    let (mut hvr, mut rec, mut prec) = (0.0, 0.0, 0.0);
                    for t in 0..s.topics {
                        let task = Task { id: 0, task_type: None, features: (0..3).map(|j| format!("t{t}k{j}")).collect() };
                        let tp: Vec<(f64, f64, f64)> = (0..NA).map(|a| (w.p_topic(t, a), mw.cost[a], mw.lat[a])).collect();
                        let ep: Vec<(f64, f64, f64)> = (0..NA).map(|a| (est.value(&task, a), mw.cost[a], mw.lat[a])).collect();
                        let tf: Vec<usize> = (0..NA).filter(|&i| !dominated(i, &tp)).collect();
                        let lf: Vec<usize> = (0..NA).filter(|&i| !dominated(i, &ep)).collect();
                        let hv_true = hv3(&tf.iter().map(|&i| tp[i]).collect::<Vec<_>>());
                        let hv_learn = hv3(&lf.iter().map(|&i| tp[i]).collect::<Vec<_>>());
                        hvr += (hv_learn / hv_true.max(1e-12)).min(1.0);
                        let inter = lf.iter().filter(|i| tf.contains(i)).count() as f64;
                        rec += inter / tf.len() as f64;
                        prec += inter / lf.len() as f64;
                    }
                    let k = s.topics as f64;
                    (hvr / k, rec / k, prec / k)
                };
                for t in 0..N_TASKS {
                    if cps.contains(&t) {
                        res.push(measure(&est));
                    }
                    let task = &w.tasks[t];
                    let k = t % 4;
                    let lam = C_LAMS[k];
                    let mut best = (f64::NEG_INFINITY, 0usize);
                    for &a in &cands {
                        let u = est.value(task, a) - mw.pen(lam, a);
                        if u > best.0 {
                            best = (u, a);
                        }
                    }
                    let a = best.1;
                    let _ = rng.f64();
                    let p = w.p[t][a];
                    est.update(task, a, if w.u[t] < p { 1.0 } else { 0.0 });
                }
                res.push(measure(&est));
                res
            })
            .collect();
        let mut rows = vec![];
        for (ci, cp) in cps.iter().enumerate() {
            let hv: Vec<f64> = per.iter().map(|r| r[ci].0).collect();
            let rc: Vec<f64> = per.iter().map(|r| r[ci].1).collect();
            let pr: Vec<f64> = per.iter().map(|r| r[ci].2).collect();
            println!(
                "C-frontier {:<9} after {:>4} tasks: HV ratio {:.3}  Pareto recall {:.3}  precision {:.3}",
                s.name, cp, mean_of(&hv), mean_of(&rc), mean_of(&pr)
            );
            rows.push(json!({"tasks": cp, "hv_ratio": neural_mesh::stats::summarize(&hv), "recall": neural_mesh::stats::summarize(&rc), "precision": neural_mesh::stats::summarize(&pr)}));
        }
        out.push(json!({"world": s.name, "rows": rows}));
    }
    json!(out)
}

impl World {
    /// true success probability of agent `a` on topic `t` (first task of that topic)
    fn p_topic(&self, t: usize, a: usize) -> f64 {
        let i = self.ty.iter().position(|&x| x == t).unwrap_or(0);
        self.p[i][a]
    }
}

// ================================================================== D: committee
#[derive(Clone, Copy, Debug)]
enum CRule {
    Plurality,
    Gated { thr: f64, p: f64 },
    UniformEps { q: f64 },
    Weighted { eta: f64 },
}

struct Committee {
    members: Vec<Box<dyn Router>>,
    rule: CRule,
    w: Vec<f64>,
    votes: Vec<usize>,
    rbar: f64,
    last_dis: f64,
}

impl Committee {
    fn new(kw: &HashMap<String, usize>, na: usize, rule: CRule) -> Self {
        // priority order: strongest first (ties in plurality go to the earliest member)
        let order = [5usize, 4, 3, 1, 2, 0];
        let members: Vec<Box<dyn Router>> = order.iter().map(|&b| build_base(b, kw, na)).collect();
        let m = members.len();
        Committee { members, rule, w: vec![1.0; m], votes: vec![0; m], rbar: 0.5, last_dis: 0.0 }
    }
}

impl Router for Committee {
    fn name(&self) -> String {
        "committee".into()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        let m = self.members.len();
        for i in 0..m {
            self.votes[i] = self.members[i].select(task, candidates, rng);
        }
        let weighted = matches!(self.rule, CRule::Weighted { .. });
        let mut score: Vec<(usize, f64, usize)> = vec![]; // (agent, weight, first member idx)
        for i in 0..m {
            let wt = if weighted { self.w[i] } else { 1.0 };
            if let Some(e) = score.iter_mut().find(|e| e.0 == self.votes[i]) {
                e.1 += wt;
            } else {
                score.push((self.votes[i], wt, i));
            }
        }
        let total: f64 = score.iter().map(|e| e.1).sum();
        let mut top = score[0];
        for e in score.iter().skip(1) {
            if e.1 > top.1 + 1e-12 || ((e.1 - top.1).abs() <= 1e-12 && e.2 < top.2) {
                top = *e;
            }
        }
        let dis = 1.0 - top.1 / total;
        self.last_dis = dis;
        match self.rule {
            CRule::Plurality | CRule::Weighted { .. } => top.0,
            CRule::Gated { thr, p } => {
                if dis >= thr && rng.f64() < p { score[rng.below(score.len())].0 } else { top.0 }
            }
            CRule::UniformEps { q } => {
                if rng.f64() < q { score[rng.below(score.len())].0 } else { top.0 }
            }
        }
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        if let CRule::Weighted { eta } = self.rule {
            let credit = reward - self.rbar;
            for i in 0..self.members.len() {
                if self.votes[i] == agent {
                    self.w[i] = (self.w[i] * (eta * credit).exp()).clamp(0.05, 20.0);
                }
            }
            self.rbar += 0.02 * (reward - self.rbar);
        }
        for mbr in self.members.iter_mut() {
            mbr.update(task, agent, reward);
        }
    }
}

struct CommitteeExp {
    worlds: Vec<TopicSpec>,
}

impl Exp for CommitteeExp {
    type Cand = CRule;
    fn id(&self) -> &'static str {
        "D-committee"
    }
    fn labels(&self) -> Vec<CellLabel> {
        let mut v = vec![];
        for w in &self.worlds {
            for h in HORIZONS {
                v.push(CellLabel { world: w.name.into(), metric: format!("H{h}"), control: w.flat || w.traps == 0 });
            }
        }
        v
    }
    fn baselines(&self) -> Vec<String> {
        BASES.iter().map(|s| s.to_string()).collect()
    }
    fn run_base(&self, b: usize, seed: u64) -> Vec<f64> {
        let mut out = vec![];
        for s in &self.worlds {
            let w = topic_world(seed, s, N_TASKS);
            let mut r = build_base(b, &w.kwmap, NA);
            out.extend(prefix_means(&trace(r.as_mut(), &w, seed, None, &mut |_| {})));
        }
        out
    }
    fn run_cand(&self, c: &CRule, seed: u64) -> Vec<f64> {
        let mut out = vec![];
        for s in &self.worlds {
            let w = topic_world(seed, s, N_TASKS);
            let mut r = Committee::new(&w.kwmap, NA, *c);
            out.extend(prefix_means(&trace(&mut r, &w, seed, None, &mut |_| {})));
        }
        out
    }
    fn families(&self, quick: bool) -> Vec<(&'static str, Vec<CRule>)> {
        let mut gated = vec![];
        for thr in [0.34, 0.5, 0.67] {
            for p in [0.25, 0.5, 1.0] {
                gated.push(CRule::Gated { thr, p });
            }
        }
        let uni: Vec<CRule> = [0.02, 0.05, 0.1, 0.2].iter().map(|&q| CRule::UniformEps { q }).collect();
        let wt: Vec<CRule> = [0.5, 1.0, 3.0].iter().map(|&eta| CRule::Weighted { eta }).collect();
        let mut v = vec![("committee-plurality", vec![CRule::Plurality]), ("committee-gated-explore", gated), ("committee-uniform-eps", uni), ("committee-weighted", wt)];
        if quick {
            for f in v.iter_mut() {
                f.1.truncate(2);
            }
        }
        v
    }
    fn ablations(&self) -> Vec<(String, String)> {
        vec![
            ("committee-gated-explore".into(), "committee-uniform-eps".into()),
            ("committee-gated-explore".into(), "committee-plurality".into()),
            ("committee-weighted".into(), "committee-plurality".into()),
        ]
    }
}

fn auc(pos: &[f64], neg: &[f64]) -> f64 {
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    let mut all: Vec<(f64, bool)> = pos.iter().map(|&x| (x, true)).chain(neg.iter().map(|&x| (x, false))).collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && (all[j + 1].0 - all[i].0).abs() < 1e-12 {
            j += 1;
        }
        let r = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            if all[k].1 {
                rank_sum += r;
            }
        }
        i = j + 1;
    }
    (rank_sum - pos.len() as f64 * (pos.len() as f64 + 1.0) / 2.0) / (pos.len() as f64 * neg.len() as f64)
}

/// D diagnostic: does committee disagreement predict that the committee's own choice is
/// sub-optimal (p < best - 0.1), beyond what low evidence alone predicts?
fn d_diag(held: &[u64], ws: &[TopicSpec]) -> Value {
    let mut out = vec![];
    for s in ws {
        let mut rows = vec![];
        for (lo, hi) in [(0usize, 250usize), (250, 1000), (1000, 2000)] {
            let per: Vec<(f64, f64, f64)> = held
                .par_iter()
                .map(|&seed| {
                    let w = topic_world(seed, s, N_TASKS);
                    let mut c = Committee::new(&w.kwmap, NA, CRule::Plurality);
                    let cands: Vec<usize> = (0..NA).collect();
                    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
                    let mut seen: HashMap<String, f64> = HashMap::new();
                    let (mut d_pos, mut d_neg, mut e_pos, mut e_neg) = (vec![], vec![], vec![], vec![]);
                    let mut recs: Vec<(f64, f64, bool)> = vec![];
                    for (i, task) in w.tasks.iter().enumerate() {
                        let a = c.select(task, &cands, &mut rng);
                        let best = w.p[i].iter().cloned().fold(0.0, f64::max);
                        let bad = w.p[i][a] < best - 0.1;
                        let ev = if task.features.is_empty() { 0.0 } else { task.features.iter().map(|f| seen.get(f).copied().unwrap_or(0.0)).sum::<f64>() / task.features.len() as f64 };
                        if i >= lo && i < hi {
                            recs.push((c.last_dis, 1.0 / (1.0 + ev), bad));
                            if bad {
                                d_pos.push(c.last_dis);
                                e_pos.push(1.0 / (1.0 + ev));
                            } else {
                                d_neg.push(c.last_dis);
                                e_neg.push(1.0 / (1.0 + ev));
                            }
                        }
                        let p = w.p[i][a];
                        c.update(task, a, if w.u[i] < p { 1.0 } else { 0.0 });
                        for f in &task.features {
                            *seen.entry(f.clone()).or_insert(0.0) += 1.0;
                        }
                    }
                    // partial AUC of disagreement within evidence terciles
                    let mut srt: Vec<f64> = recs.iter().map(|r| r.1).collect();
                    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let (q1, q2) = (srt[srt.len() / 3], srt[2 * srt.len() / 3]);
                    let mut part = vec![];
                    for (lo_e, hi_e) in [(f64::NEG_INFINITY, q1), (q1, q2), (q2, f64::INFINITY)] {
                        let sel: Vec<&(f64, f64, bool)> = recs.iter().filter(|r| r.1 > lo_e && r.1 <= hi_e).collect();
                        let p: Vec<f64> = sel.iter().filter(|r| r.2).map(|r| r.0).collect();
                        let n: Vec<f64> = sel.iter().filter(|r| !r.2).map(|r| r.0).collect();
                        let a = auc(&p, &n);
                        if a.is_finite() {
                            part.push(a);
                        }
                    }
                    (auc(&d_pos, &d_neg), auc(&e_pos, &e_neg), if part.is_empty() { f64::NAN } else { mean_of(&part) })
                })
                .collect();
            let col_ok = |k: usize| -> Vec<f64> { per.iter().map(|x| [x.0, x.1, x.2][k]).filter(|v| v.is_finite()).collect() };
            let (a, b, c) = (col_ok(0), col_ok(1), col_ok(2));
            let sm = |v: &Vec<f64>| if v.len() > 1 { Some(neural_mesh::stats::summarize(v)) } else { None };
            println!(
                "D-diag {:<10} tasks {:>4}-{:<4} AUC(disagreement->bad choice) {:.3}  AUC(low-evidence->bad) {:.3}  partial AUC(disagreement | evidence tercile) {:.3}  (n seeds {}/{}/{})",
                s.name, lo, hi, mean_of(&a), mean_of(&b), mean_of(&c), a.len(), b.len(), c.len()
            );
            rows.push(json!({"tasks": format!("{lo}-{hi}"), "auc_disagreement": sm(&a), "auc_low_evidence": sm(&b), "auc_disagreement_within_evidence_tercile": sm(&c)}));
        }
        out.push(json!({"world": s.name, "windows": rows}));
    }
    json!(out)
}

// ============================================================= driver (main)
fn selfcheck() {
    // ForkSwap without a new agent must equal its base router trace-for-trace.
    let spec = &specs()[0];
    let w = topic_world(3, spec, 600);
    let mut a = ForkSwap::new(Dyn(build_base(5, &w.kwmap, NA + 1)), 0, NA, 0.5);
    let mut b = build_base(5, &w.kwmap, NA + 1);
    let ta = trace(&mut a, &w, 3, None, &mut |_| {});
    let tb = trace(b.as_mut(), &w, 3, None, &mut |_| {});
    assert!(ta.iter().zip(&tb).all(|(x, y)| (x - y).abs() < 1e-12), "ForkSwap(no new agent) != base");
    println!("selfcheck ok: ForkSwap with no new agent == base router");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let out = arg("--out").unwrap_or("results/f1_lab.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let which: Vec<String> = arg("--exp").unwrap_or("a1,a2,b,c,d".into()).split(',').map(|s| s.to_string()).collect();
    let (h0, h1) = arg("--held")
        .and_then(|r| r.split_once('-').map(|(a, b)| (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap())))
        .unwrap_or((31, 60));
    let held: Vec<u64> = (h0..=h1).collect();
    let tune: Vec<u64> = (1..=8).collect();
    println!("f1_lab: tune seeds 1-8, held-out {h0}-{h1}, {} experiments{}", which.len(), if quick { " (quick)" } else { "" });
    selfcheck();
    if args.iter().any(|a| a == "--diag") {
        let e = XferExp { worlds: specs(), rhos: vec![0.0], np: 1000, cache: Mutex::new(HashMap::new()) };
        for wi in 0..3 {
            for margin in [0.0, 0.1, 0.3, 0.6, 0.9] {
                let (mut cov, mut ok, mut tot) = (0, 0, 0);
                for seed in 31..61u64 {
                    let par = e.parent(wi, seed);
                    let pm = parent_map(&par.router, margin);
                    for (f, a) in &pm {
                        cov += 1;
                        let ty: usize = f[1..f.find('k').unwrap()].parse().unwrap();
                        let best = (0..NA).max_by(|&x, &y| par.skills.p[ty][x].partial_cmp(&par.skills.p[ty][y]).unwrap()).unwrap();
                        if *a == best { ok += 1; }
                    }
                    tot += par.skills.kw.len();
                }
                println!("diag world {} margin {margin}: covers {:.2} of features, accuracy {:.3}", e.worlds[wi].name, cov as f64 / tot as f64, ok as f64 / cov.max(1) as f64);
            }
        }
        let par = e.parent(0, 31);
        for (f, row) in par.router.rows().iter().take(4) { println!("row {f}: {:?}", row.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>()); }
        return;
    }
    if args.iter().any(|a| a == "--diag2") {
        let e = XferExp { worlds: specs(), rhos: vec![0.0, 0.25, 0.5, 1.0], np: 1000, cache: Mutex::new(HashMap::new()) };
        let seeds: Vec<u64> = (1..=8).collect();
        let bm: Vec<Vec<f64>> = (0..6).map(|b| cell_means(&matrix(&seeds, |s| e.run_base(b, s)))).collect();
        let cfgs = vec![
            XCand::GuardMap { margin: 0.3, tau: 0.7, ka: 2.0, kb: 1.0, g_tau: 0.0, disagree: true, shrink: None },
            XCand::GuardMap { margin: 0.1, tau: 0.7, ka: 2.0, kb: 1.0, g_tau: 0.0, disagree: false, shrink: None },
            XCand::GuardMap { margin: 0.3, tau: 0.7, ka: 2.0, kb: 1.0, g_tau: 0.7, disagree: true, shrink: None },
            XCand::GuardMap { margin: 0.3, tau: 0.7, ka: 2.0, kb: 1.0, g_tau: 0.0, disagree: true, shrink: Some(0.3) },
        ];
        let labels = e.labels();
        for c in cfgs {
            let m = cell_means(&matrix(&seeds, |s| e.run_cand(&c, s)));
            println!("{:?}", c);
            for (wi, w) in e.worlds.iter().enumerate() {
                for ri in 0..4 {
                    let idx0 = (wi * 4 + ri) * 5;
                    let d: Vec<String> = (0..5).map(|h| {
                        let i = idx0 + h;
                        let best = bm.iter().map(|x| x[i]).fold(f64::NEG_INFINITY, f64::max);
                        format!("{:+.4}", m[i] - best)
                    }).collect();
                    println!("   {:<9} rho {:<4} diff vs best base by H: {}", w.name, e.rhos[ri], d.join(" "));
                }
            }
            let _ = &labels;
        }
        return;
    }
    let t_start = std::time::Instant::now();
    let mut all: Vec<(String, ExpOut)> = vec![];
    if which.iter().any(|x| x == "a1") {
        let np: usize = arg("--np").and_then(|x| x.parse().ok()).unwrap_or(1000);
        let e = XferExp { worlds: specs(), rhos: vec![0.0, 0.25, 0.5, 1.0], np, cache: Mutex::new(HashMap::new()) };
        all.push(("A1".into(), run_exp(&e, &tune, &held, quick)));
    }
    if which.iter().any(|x| x == "a2") {
        let ws = vec![specs()[0].clone(), specs()[2].clone()];
        let e = AddExp { worlds: ws, kinds: vec![NewKind::Upgrade, NewKind::Specialist, NewKind::Unrelated], t0: 500 };
        all.push(("A2".into(), run_exp(&e, &tune, &held, quick)));
    }
    let mut cdiag = Value::Null;
    if which.iter().any(|x| x == "b") {
        let ws = vec![
            specs()[0].clone(),
            specs()[1].clone(),
            specs()[2].clone(),
            TopicSpec { name: "zipf-traps8", topics: 8, traps: 2, map_err: 0.10, confound: 0.25, zipf: true, flat: false },
            TopicSpec { name: "flat8", topics: 8, traps: 0, map_err: 0.0, confound: 0.25, zipf: false, flat: true },
        ];
        let e = OrderExp { worlds: ws, pools: vec![150, 300, 600] };
        all.push(("B".into(), run_exp(&e, &tune, &held, quick)));
    }
    if which.iter().any(|x| x == "c") {
        let e = MoExp { worlds: vec![specs()[0].clone(), specs()[1].clone(), specs()[2].clone()] };
        all.push(("C".into(), run_exp(&e, &tune, &held, quick)));
        println!();
        cdiag = c_diag(&held, &e.worlds);
    }
    let mut ddiag = Value::Null;
    if which.iter().any(|x| x == "d") {
        let ws = vec![
            specs()[0].clone(),
            specs()[1].clone(),
            specs()[2].clone(),
            TopicSpec { name: "flat8", topics: 8, traps: 0, map_err: 0.0, confound: 0.25, zipf: false, flat: true },
        ];
        let e = CommitteeExp { worlds: ws.clone() };
        all.push(("D".into(), run_exp(&e, &tune, &held, quick)));
        println!();
        ddiag = d_diag(&held, &ws);
    }
    // global Holm over all acceptance tests
    let mut flat: Vec<(usize, usize)> = vec![];
    for (ei, (_, o)) in all.iter().enumerate() {
        for (ri, r) in o.recs.iter().enumerate() {
            if r.kind == "accept" {
                flat.push((ei, ri));
            }
        }
    }
    let adj = holm_adjust(&flat.iter().map(|&(e, r)| all[e].1.recs[r].p).collect::<Vec<_>>());
    for (&(e, r), a) in flat.iter().zip(adj) {
        all[e].1.recs[r].holm_global = a;
    }
    let mut jexp = vec![];
    for (name, o) in &all {
        println!("\n=================== {name} (held-out {h0}-{h1}) ===================");
        print_summary(&o.recs);
        print_tests(&o.recs, true);
        jexp.push(json!({"exp": name, "tuned": o.tuned.iter().map(|(a, b)| json!({"cand": a, "config": b})).collect::<Vec<_>>(),
            "tests": o.recs.iter().map(rec_json).collect::<Vec<_>>(), "means": o.cost_note}));
    }
    let nacc: usize = all.iter().map(|(_, o)| o.recs.iter().filter(|r| r.kind == "accept").count()).sum();
    let gwin = all.iter().flat_map(|(_, o)| o.recs.iter()).filter(|r| r.kind == "accept" && r.holm_global < 0.05 && r.diff.mean > 0.0).count();
    let gloss = all.iter().flat_map(|(_, o)| o.recs.iter()).filter(|r| r.kind == "accept" && r.holm_global < 0.05 && r.diff.mean < 0.0).count();
    println!("\nGLOBAL Holm over {nacc} acceptance tests: WIN {gwin}, LOSS {gloss}");
    let j = json!({"held_out": format!("{h0}-{h1}"), "protocol": "tune seeds 1-8; held-out as given; paired t vs best fair baseline per cell; Holm per experiment and global",
        "global": {"n_tests": nacc, "wins": gwin, "losses": gloss}, "d_diagnostic": ddiag, "c_frontier_diagnostic": cdiag, "runtime_s": t_start.elapsed().as_secs_f64(), "experiments": jexp});
    let _ = std::fs::create_dir_all("results");
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out} ({:.0}s)", t_start.elapsed().as_secs_f64());
}
