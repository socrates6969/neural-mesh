//! W2 workflow lab: can mesh EDGES carry value when the reward is not informative about
//! simultaneous teams, i.e. when edges predict the NEXT agent in a multi-step workflow
//! (hand-off chains = sequential composition)?  Also measures sparse-edge memory.
//!
//! World (own, seeded; sim.rs is not used): 6 agents, 8 step types, per-type skill table
//! with trap/drift as in shortlab; workflows are chains of 3 steps (6 templates of distinct
//! step types).  Router sees only the noisy keywords of the CURRENT step (+ the previous
//! agent for the sequential candidates).  Hidden hand-off matrix H[prev][next]:
//! for every previous agent one "good" next agent (+mag) and one "bad" (-0.8 mag).
//! p(step) = clamp(skill[type][agent] + H[prev][agent]).  Outcome = (u < p) with u pre-drawn.
//! Worlds: main (mag .30), weak (mag .10), none (mag 0 = null world).
//!
//! Protocol: tune on seeds 1-8 (main world only), evaluate on held-out seeds 31-60,
//! paired 95% t-CI, Holm correction over all candidate x world x metric tests.  Metrics:
//! mean expected step success over the full run ("full") and over the first 501 steps
//! ("early"); workflow-completion probability is reported descriptively.
//!
//! Usage: cargo run --release --bin w2_workflow [-- --out results/w2_workflow.json]

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use neural_mesh::variants::{SeqAddConfig, SeqAddRouter, SeqEdges, SeqEdgesConfig, SeqMesh};
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashMap;

const A: usize = 6;
const TY: usize = 8;
const N: usize = 2000;
const L: usize = 3;
const EARLY: usize = 501;

#[derive(Clone, Copy, Debug)]
struct WCfg {
    name: &'static str,
    mag: f64,
}

struct World {
    tasks: Vec<Task>,
    ty: Vec<usize>,
    pos: Vec<usize>,
    u: Vec<f64>,
    pre: [[f64; A]; TY],
    post: [[f64; A]; TY],
    hand: [[f64; A]; A],
    kwmap: HashMap<String, usize>,
}

fn gen_world(seed: u64, c: WCfg) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(101));
    let mut pre = [[0.0; A]; TY];
    let mut best = [0usize; TY];
    let mut second = [0usize; TY];
    for t in 0..TY {
        best[t] = rng.below(A);
        loop {
            second[t] = rng.below(A);
            if second[t] != best[t] {
                break;
            }
        }
        for a in 0..A {
            pre[t][a] = if a == best[t] {
                0.85
            } else if a == second[t] {
                0.5
            } else {
                0.15 + 0.25 * rng.f64()
            };
        }
    }
    // drift of two types at N/2: best declines, a new agent becomes best
    let mut post = pre;
    let mut order: Vec<usize> = (0..TY).collect();
    for i in (1..TY).rev() {
        order.swap(i, rng.below(i + 1));
    }
    for &t in order.iter().take(2) {
        let nb = loop {
            let x = rng.below(A);
            if x != best[t] {
                break x;
            }
        };
        post[t][best[t]] = 0.45;
        post[t][nb] = 0.85;
    }
    // traps: keyword map points at a bad agent for 2 types
    let trap: Vec<usize> = order.iter().skip(2).take(2).copied().collect();
    let mut kwmap = HashMap::new();
    for t in 0..TY {
        for j in 0..4 {
            let a = if trap.contains(&t) {
                loop {
                    let a = rng.below(A);
                    if a != best[t] && a != second[t] {
                        break a;
                    }
                }
            } else if rng.f64() < 0.10 {
                rng.below(A)
            } else {
                best[t]
            };
            kwmap.insert(format!("t{t}k{j}"), a);
        }
    }
    let mut hand = [[0.0; A]; A];
    if c.mag > 0.0 {
        for p in 0..A {
            let g = rng.below(A);
            let d = loop {
                let d = rng.below(A);
                if d != g {
                    break d;
                }
            };
            hand[p][g] = c.mag;
            hand[p][d] = -0.8 * c.mag;
        }
    }
    // workflow templates
    let mut templates = vec![[0usize; L]; 6];
    for tpl in templates.iter_mut() {
        let mut idx: Vec<usize> = (0..TY).collect();
        for i in (1..TY).rev() {
            idx.swap(i, rng.below(i + 1));
        }
        for j in 0..L {
            tpl[j] = idx[j];
        }
    }
    let (mut tasks, mut ty, mut pos, mut u) = (vec![], vec![], vec![], vec![]);
    let mut i = 0usize;
    while i < N {
        let tpl = templates[rng.below(templates.len())];
        for j in 0..L {
            if i >= N {
                break;
            }
            let t = tpl[j];
            let mut f: Vec<String> = (0..4).filter(|_| rng.bernoulli(0.5)).map(|k| format!("t{t}k{k}")).collect();
            if f.is_empty() {
                f.push(format!("t{t}k{}", rng.below(4)));
            }
            if rng.bernoulli(0.25) {
                let o = (t + 1 + rng.below(TY - 1)) % TY;
                f.push(format!("t{o}k{}", rng.below(4)));
            }
            tasks.push(Task { id: i as u64, task_type: None, features: f });
            ty.push(t);
            pos.push(j);
            u.push(rng.f64());
            i += 1;
        }
    }
    World { tasks, ty, pos, u, pre, post, hand, kwmap }
}

impl World {
    fn p(&self, i: usize, prev: Option<usize>, a: usize) -> f64 {
        let s = if i < N / 2 { self.pre[self.ty[i]][a] } else { self.post[self.ty[i]][a] };
        let h = prev.map(|p| self.hand[p][a]).unwrap_or(0.0);
        (s + h).clamp(0.02, 0.97)
    }
}

// ---------------------------------------------------------------------------
// Policies

trait Pol {
    fn pick(&mut self, task: &Task, prev: Option<usize>, cands: &[usize], rng: &mut Rng) -> usize;
    fn upd(&mut self, task: &Task, prev: Option<usize>, agent: usize, reward: f64);
}
struct Plain<R: Router>(R);
impl<R: Router> Pol for Plain<R> {
    fn pick(&mut self, task: &Task, _: Option<usize>, cands: &[usize], rng: &mut Rng) -> usize {
        self.0.select(task, cands, rng)
    }
    fn upd(&mut self, task: &Task, _: Option<usize>, agent: usize, reward: f64) {
        self.0.update(task, agent, reward)
    }
}
/// The obvious control: expose the previous agent as one more feature.
struct PrevFeat<R: Router>(R);
fn with_prev(task: &Task, prev: Option<usize>) -> Task {
    let mut t = task.clone();
    t.features.push(match prev {
        Some(p) => format!("prev={p}"),
        None => "prev=start".to_string(),
    });
    t
}
impl<R: Router> Pol for PrevFeat<R> {
    fn pick(&mut self, task: &Task, prev: Option<usize>, cands: &[usize], rng: &mut Rng) -> usize {
        self.0.select(&with_prev(task, prev), cands, rng)
    }
    fn upd(&mut self, task: &Task, prev: Option<usize>, agent: usize, reward: f64) {
        self.0.update(&with_prev(task, prev), agent, reward)
    }
}
struct Seq(SeqMesh);
impl Pol for Seq {
    fn pick(&mut self, task: &Task, prev: Option<usize>, cands: &[usize], rng: &mut Rng) -> usize {
        self.0.select(task, cands, prev, rng)
    }
    fn upd(&mut self, task: &Task, prev: Option<usize>, agent: usize, reward: f64) {
        self.0.update(task, prev, agent, reward)
    }
}

struct SeqAddPol(SeqAddRouter);
impl Pol for SeqAddPol {
    fn pick(&mut self, task: &Task, prev: Option<usize>, cands: &[usize], rng: &mut Rng) -> usize {
        self.0.set_prev(prev);
        self.0.select(task, cands, rng)
    }
    fn upd(&mut self, task: &Task, prev: Option<usize>, agent: usize, reward: f64) {
        self.0.set_prev(prev);
        self.0.update(task, agent, reward)
    }
}
struct WarmSeqAddPol(Warm<SeqAddRouter>);
impl Pol for WarmSeqAddPol {
    fn pick(&mut self, task: &Task, prev: Option<usize>, cands: &[usize], rng: &mut Rng) -> usize {
        self.0.inner.set_prev(prev);
        self.0.select(task, cands, rng)
    }
    fn upd(&mut self, task: &Task, prev: Option<usize>, agent: usize, reward: f64) {
        self.0.inner.set_prev(prev);
        self.0.update(task, agent, reward)
    }
}

#[derive(Clone, Copy, Debug)]
enum Ev {
    Set,
    Mean,
    Global,
}
fn ev<R>(e: Ev) -> Evidence<R> {
    match e {
        Ev::Set => Evidence::FeatureSet,
        Ev::Mean => Evidence::MeanFeature,
        Ev::Global => Evidence::Global,
    }
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    HebR,
    HebT(HebbianConfig),
    WarmHebR(f64, Ev),
    HebPrev(HebbianConfig),
    SeqAdd(SeqAddConfig),
    WarmSeqAdd(SeqAddConfig, f64, Ev),
    Seq(HebbianConfig, SeqEdgesConfig),
}

fn heb_robust() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

fn make(kind: &Kind, w: &World) -> Box<dyn Pol> {
    match kind {
        Kind::Keyword => Box::new(Plain(KeywordRouter::new(w.kwmap.clone(), A))),
        Kind::Eps(g, e) => Box::new(Plain(EpsGreedyRouter::new(A, *g, *e))),
        Kind::Pts(g) => Box::new(Plain(PooledThompsonRouter::new(A, *g))),
        Kind::HebR => Box::new(Plain(HebbianRouter::new(A, heb_robust()))),
        Kind::HebT(c) => Box::new(Plain(HebbianRouter::new(A, c.clone()))),
        Kind::WarmHebR(k, e) => {
            Box::new(Plain(Warm::new(HebbianRouter::new(A, heb_robust()), w.kwmap.clone(), A, *k).with_evidence(ev(*e))))
        }
        Kind::HebPrev(c) => Box::new(PrevFeat(HebbianRouter::new(A, c.clone()))),
        Kind::SeqAdd(c) => Box::new(SeqAddPol(SeqAddRouter::new(A, c.clone()))),
        Kind::WarmSeqAdd(c, k, e) => Box::new(WarmSeqAddPol(
            Warm::new(SeqAddRouter::new(A, c.clone()), w.kwmap.clone(), A, *k).with_evidence(ev(*e)),
        )),
        Kind::Seq(h, s) => Box::new(Seq(SeqMesh::new(A, h.clone(), s.clone()))),
    }
}

/// [full, early, workflow-completion, oracle-full] (expected-success based)
fn drive(mut pol: Box<dyn Pol>, w: &World, seed: u64) -> [f64; 4] {
    let cands: Vec<usize> = (0..A).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let (mut full, mut early, mut orc) = (0.0, 0.0, 0.0);
    let (mut wf_sum, mut wf_n, mut wf_p) = (0.0f64, 0.0f64, 1.0f64);
    let mut prev: Option<usize> = None;
    for i in 0..N {
        if w.pos[i] == 0 {
            prev = None;
            wf_p = 1.0;
        }
        let task = &w.tasks[i];
        let a = pol.pick(task, prev, &cands, &mut rng);
        let p = w.p(i, prev, a);
        full += p;
        if i < EARLY {
            early += p;
        }
        orc += (0..A).map(|x| w.p(i, prev, x)).fold(0.0, f64::max);
        wf_p *= p;
        if w.pos[i] == L - 1 {
            wf_sum += wf_p;
            wf_n += 1.0;
        }
        pol.upd(task, prev, a, if w.u[i] < p { 1.0 } else { 0.0 });
        prev = Some(a);
    }
    [full / N as f64, early / EARLY as f64, wf_sum / wf_n.max(1.0), orc / N as f64]
}

fn eval(kind: &Kind, seeds: &[u64], c: WCfg) -> Vec<[f64; 4]> {
    seeds
        .par_iter()
        .map(|&s| {
            let w = gen_world(s, c);
            drive(make(kind, &w), &w, s)
        })
        .collect()
}

fn obj(kind: &Kind, seeds: &[u64], c: WCfg) -> f64 {
    let r = eval(kind, seeds, c);
    r.iter().map(|x| 0.5 * (x[0] + x[1])).sum::<f64>() / r.len() as f64
}

fn best_of(cands: Vec<Kind>, seeds: &[u64], c: WCfg, label: &str) -> Kind {
    let n = cands.len();
    let mut best = (f64::NEG_INFINITY, cands[0].clone());
    for k in cands {
        let o = obj(&k, seeds, c);
        if o > best.0 {
            best = (o, k);
        }
    }
    println!("[tune 1-8] {label}: {n} configs, best {:.4} -> {:?}", best.0, best.1);
    best.1
}

fn col(r: &[[f64; 4]], m: usize) -> Vec<f64> {
    r.iter().map(|x| x[m]).collect()
}

fn memory_scaling() -> serde_json::Value {
    println!("\n=== memory: dense n x n f64 matrix vs sparse top-k edges (k=4), 20 hand-offs per agent ===");
    println!("{:>8} {:>14} {:>14} {:>10} {:>9}", "agents", "dense bytes", "sparse bytes", "ratio", "edges");
    let mut out = vec![];
    for &n in &[6usize, 50, 500, 5000] {
        let mut e = SeqEdges::new(n, SeqEdgesConfig { k: 4, ..SeqEdgesConfig::default() });
        let mut rng = Rng::new(5);
        for _ in 0..(20 * n) {
            let (s, d) = (rng.below(n), rng.below(n));
            e.observe(s, d, rng.f64() - 0.5);
        }
        let (db, sb) = (SeqEdges::dense_bytes(n), e.bytes());
        println!("{:>8} {:>14} {:>14} {:>10.1} {:>9}", n, db, sb, db as f64 / sb as f64, e.n_edges());
        out.push(json!({"agents": n, "dense_bytes": db, "sparse_bytes": sb, "edges": e.n_edges()}));
    }
    json!(out)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/w2_workflow.json".into());
    let main_w = WCfg { name: "main(mag .30)", mag: 0.30 };
    let worlds = [main_w, WCfg { name: "weak(mag .10)", mag: 0.10 }, WCfg { name: "none(null)", mag: 0.0 }];
    let tune: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = (31..=60).collect();
    println!("w2_workflow: {N} steps/run ({L}-step workflows), tune seeds 1-8 (main world), held-out 31-60");

    // ---- tuning on seeds 1-8, main world -------------------------------------------------------
    let mut c = vec![];
    for &g in &[0.95, 0.99, 1.0] {
        for &e in &[0.02, 0.05, 0.1] {
            c.push(Kind::Eps(g, e));
        }
    }
    let eps = best_of(c, &tune, main_w, "eps-greedy");
    let pts = best_of([0.9, 0.95, 0.99, 1.0].iter().map(|&g| Kind::Pts(g)).collect(), &tune, main_w, "pooled-ts");
    let mut c = vec![];
    for &k in &[1.0, 3.0, 10.0, 30.0] {
        for &e in &[Ev::Set, Ev::Mean, Ev::Global] {
            c.push(Kind::WarmHebR(k, e));
        }
    }
    let warm = best_of(c, &tune, main_w, "warm(hebbian-robust)");
    let mut hc = vec![];
    for &eta in &[0.01, 0.03, 0.1] {
        for &t in &[0.002, 0.01, 0.05] {
            for &d in &[0.001, 0.01] {
                hc.push(HebbianConfig { eta, temperature: t, decay: d, epsilon: 0.0, ..HebbianConfig::default() });
            }
        }
    }
    let Kind::HebT(heb_t) = best_of(hc.iter().cloned().map(Kind::HebT).collect(), &tune, main_w, "hebbian (tuned here, extra baseline)") else { unreachable!() };
    let Kind::HebPrev(heb_p) = best_of(hc.iter().cloned().map(Kind::HebPrev).collect(), &tune, main_w, "hebbian+prev-feature (control)") else { unreachable!() };
    let mut sc = vec![];
    for &lambda in &[0.5, 1.0, 2.0, 4.0] {
        for &eta in &[0.05, 0.15, 0.4] {
            for &optimism in &[0.0, 0.15, 0.3] {
                for &epsilon in &[0.0, 0.03] {
                    sc.push(Kind::Seq(heb_t.clone(), SeqEdgesConfig { eta, optimism, k: 0, lambda, epsilon }));
                }
            }
        }
    }
    let Kind::Seq(_, sdense) = best_of(sc, &tune, main_w, "seq edges, dense") else { unreachable!() };
    let sc: Vec<Kind> = [1usize, 2, 3, 4].iter().map(|&k| Kind::Seq(heb_t.clone(), SeqEdgesConfig { k, ..sdense.clone() })).collect();
    let Kind::Seq(_, ssparse) = best_of(sc, &tune, main_w, "seq edges, sparse k") else { unreachable!() };

    let mut sa = vec![];
    let mut sa0 = vec![];
    for &eta_f in &[0.05, 0.1, 0.2, 0.4] {
        for &eps_ in &[0.0, 0.03] {
            for &init in &[0.5, 0.8] {
                sa0.push(Kind::SeqAdd(SeqAddConfig { eta_f, eta_g: 0.0, epsilon: eps_, init }));
                for &eta_g in &[0.03, 0.1, 0.3] {
                    sa.push(Kind::SeqAdd(SeqAddConfig { eta_f, eta_g, epsilon: eps_, init }));
                }
            }
        }
    }
    let Kind::SeqAdd(sa_cfg) = best_of(sa, &tune, main_w, "seq-add (features + hand-off edge)") else { unreachable!() };
    let Kind::SeqAdd(sa0_cfg) = best_of(sa0, &tune, main_w, "seq-add NO edges (ablation)") else { unreachable!() };
    let warm_of = |cfg: &SeqAddConfig, label: &str| {
        let mut c = vec![];
        for &k in &[1.0, 3.0, 10.0, 30.0] {
            for &e in &[Ev::Set, Ev::Mean, Ev::Global] {
                c.push(Kind::WarmSeqAdd(cfg.clone(), k, e));
            }
        }
        best_of(c, &tune, main_w, label)
    };
    let warm_sa = warm_of(&sa_cfg, "warm(seq-add)");
    let warm_sa0 = warm_of(&sa0_cfg, "warm(seq-add no edges, ablation)");

    let baselines: Vec<(&str, Kind)> = vec![
        ("keyword", Kind::Keyword),
        ("eps-greedy", eps),
        ("pooled-ts", pts),
        ("hebbian-robust", Kind::HebR),
        ("warm(hebbian-robust)", warm),
        ("hebbian-tuned-here(extra)", Kind::HebT(heb_t.clone())),
        ("seq-add-NOedges(extra)", Kind::SeqAdd(sa0_cfg.clone())),
        ("warm(seq-add-NOedges)(extra)", warm_sa0),
    ];
    let cands: Vec<(&str, Kind)> = vec![
        ("hebbian+prev-feature", Kind::HebPrev(heb_p)),
        ("seq-edges-dense", Kind::Seq(heb_t.clone(), sdense.clone())),
        ("seq-edges-sparse", Kind::Seq(heb_t.clone(), ssparse.clone())),
        ("seq-add+edges", Kind::SeqAdd(sa_cfg.clone())),
        ("warm(seq-add+edges)", warm_sa),
    ];

    // ---- held-out evaluation ------------------------------------------------------------------
    let metric_names = ["full", "early"];
    let mut all_p: Vec<f64> = vec![];
    let mut meta: Vec<(usize, usize, usize)> = vec![]; // (world, cand, metric)
    let mut store: Vec<(Vec<Vec<[f64; 4]>>, Vec<Vec<[f64; 4]>>)> = vec![];
    let mut best_base: Vec<[usize; 2]> = vec![];
    for (wi, wc) in worlds.iter().enumerate() {
        let b: Vec<Vec<[f64; 4]>> = baselines.iter().map(|(_, k)| eval(k, &held, *wc)).collect();
        let c: Vec<Vec<[f64; 4]>> = cands.iter().map(|(_, k)| eval(k, &held, *wc)).collect();
        let mut bb = [0usize; 2];
        for m in 0..2 {
            bb[m] = (0..b.len())
                .max_by(|&x, &y| summarize(&col(&b[x], m)).mean.partial_cmp(&summarize(&col(&b[y], m)).mean).unwrap())
                .unwrap();
        }
        for ci in 0..c.len() {
            for m in 0..2 {
                all_p.push(paired_t_pvalue(&col(&c[ci], m), &col(&b[bb[m]], m)));
                meta.push((wi, ci, m));
            }
        }
        store.push((b, c));
        best_base.push(bb);
    }
    let adj = holm_adjust(&all_p);

    let mut js = vec![];
    for (wi, wc) in worlds.iter().enumerate() {
        let (b, c) = &store[wi];
        println!("\n=== world {} (held-out seeds 31-60): mean +- 95% half-width ===", wc.name);
        println!("{:<28}{:>16}{:>16}{:>16}", "router", "full", "early(501)", "wf-completion");
        let oracle = col(&c[0], 3);
        for (i, (n, _)) in baselines.iter().enumerate() {
            let f = |m: usize| { let s = summarize(&col(&b[i], m)); format!("{:.4}+-{:.4}", s.mean, s.half_width) };
            println!("{:<28}{:>16}{:>16}{:>16}", n, f(0), f(1), f(2));
        }
        for (i, (n, _)) in cands.iter().enumerate() {
            let f = |m: usize| { let s = summarize(&col(&c[i], m)); format!("{:.4}+-{:.4}", s.mean, s.half_width) };
            println!("{:<28}{:>16}{:>16}{:>16}", n, f(0), f(1), f(2));
        }
        println!("{:<28}{:>16.4}  (greedy-given-prev oracle, full)", "oracle", summarize(&oracle).mean);
        for m in 0..2 {
            println!("  best fair baseline ({}) = {}", metric_names[m], baselines[best_base[wi][m]].0);
        }
        for (k, &(w2, ci, m)) in meta.iter().enumerate() {
            if w2 != wi {
                continue;
            }
            let bb = &b[best_base[wi][m]];
            let d = paired_diff(&col(&c[ci], m), &col(bb, m));
            let verdict = if adj[k] < 0.05 { if d.diff.mean > 0.0 { "BEATS" } else { "WORSE" } } else { "n.s." };
            println!(
                "  {:<22} [{:<5}] vs best-baseline {:+.4} CI[{:+.4},{:+.4}] wins {}/{} Holm p={:.4} {}",
                cands[ci].0, metric_names[m], d.diff.mean, d.diff.lo, d.diff.hi, d.wins, d.wins + d.losses + d.ties, adj[k], verdict
            );
            js.push(json!({"world": wc.name, "candidate": cands[ci].0, "metric": metric_names[m], "vs": baselines[best_base[wi][m]].0,
                "diff": d.diff, "wins": d.wins, "losses": d.losses, "holm_p": adj[k], "verdict": verdict}));
        }
        for &(ci, bi) in &[(3usize, 6usize), (4, 7)] {
            for m in 0..2 {
                let d = paired_diff(&col(&c[ci], m), &col(&b[bi], m));
                println!(
                    "  EDGE ABLATION {:<22} [{:<5}] vs same learner without edges {:+.4} CI[{:+.4},{:+.4}] wins {}/{} (unadjusted)",
                    cands[ci].0, metric_names[m], d.diff.mean, d.diff.lo, d.diff.hi, d.wins, d.wins + d.losses + d.ties
                );
            }
        }
        // is the edge layer better than the plain prev-feature control?
        for ci in 1..cands.len() {
            for m in 0..2 {
                let d = paired_diff(&col(&c[ci], m), &col(&c[0], m));
                println!(
                    "  {:<22} [{:<5}] vs hebbian+prev-feature {:+.4} CI[{:+.4},{:+.4}] wins {}/{} (unadjusted CI)",
                    cands[ci].0, metric_names[m], d.diff.mean, d.diff.lo, d.diff.hi, d.wins, d.wins + d.losses + d.ties
                );
            }
        }
    }
    let n_sig_beats = js.iter().filter(|j| j["verdict"] == "BEATS").count();
    println!("\n{} tests (Holm family); BEATS best fair baseline: {}", js.len(), n_sig_beats);
    println!("tuned seq dense: {:?}\ntuned seq sparse: {:?}", sdense, ssparse);

    let mem = memory_scaling();
    let j = json!({
        "protocol": "tune seeds 1-8 (main world); held-out 31-60; paired 95% t-CI; Holm over candidate x world x metric",
        "tests": js, "memory": mem,
        "tuned_seq_dense": format!("{:?}", sdense), "tuned_seq_sparse": format!("{:?}", ssparse),
    });
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out}");
}
