//! F5 (REAL-EMBED-2): do the d6 knowledge-retrieval gains survive REAL sentence embeddings?
//!
//! The embeddings come from the all-MiniLM-L6-v2 ONNX model that ruvector already bundles locally
//! (C:/Users/mariu/.ruvector/models/all-MiniLM-L6-v2). They were produced OFFLINE by py/f5_embed.py
//! (existing onnxruntime in the cognee venv, nothing installed) for a SYNTHETIC templated corpus and are
//! loaded here from results/f5_corpus.json + results/f5_embeddings.f32. The Rust side has NO ML runtime
//! and no new crates.
//!
//! Three embedding sources over the SAME texts / SAME world logic (copied from e2_embed.rs / d6):
//!   mini  = real all-MiniLM-L6-v2 vectors (384-d)
//!   hash  = knowledge::hash_embed over the words of the text (384-d feature hashing: what the repo uses today)
//!   synth = e2's synthetic anisotropic / paraphrase geometry (48-d), as the reference
//!
//! Parts:  retrieval  = P/R/F1 of the retrieval layer vs ground-truth concept (no routing)
//!         knowledge  = d6 benchmark (kn-feat hidden behind Guarded, kn-direct+vote), regimes O and W
//!         shaper     = d4-shaper-adjacent: the shaped reward (proxy channels, e2 'rich' model) feeding
//!                      guarded / knowledge+guarded with real context embeddings (the shaper itself uses no embeddings)
//!
//! Protocol (as d4/d6/e2): tune on seeds 1-8, held-out 31-60 (or --held 61-90 frozen confirmation),
//! horizons 100/250/500/2000, paired 95% t-CI vs the best fair baseline per (world,H), Holm per family.
//!
//! Usage: cargo run --release --bin f5_real_embed -- [--part retrieval|knowledge|shaper|all] [--held 61-90]

use std::time::Instant;

use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

const HORIZONS: [usize; 4] = [100, 250, 500, 2000];

use neural_mesh::router::Rng;

fn normal(r: &mut Rng) -> f64 {
    let u1 = r.f64().max(1e-12);
    let u2 = r.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[derive(Clone, Debug, Default)]
struct RunOut {
    h: [f64; 4],
    fail: f64,
}
fn col(r: &[RunOut], i: usize) -> Vec<f64> {
    r.iter().map(|x| x.h[i]).collect()
}
fn argmax_by<T>(xs: &[T], f: impl Fn(&T) -> f64) -> usize {
    let mut b = 0;
    for i in 1..xs.len() {
        if f(&xs[i]) > f(&xs[b]) {
            b = i;
        }
    }
    b
}

struct Cell {
    world: String,
    cand: String,
    h: usize,
    diff: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    p: f64,
    best_base: String,
    cand_mean: f64,
    base_mean: f64,
    treat: bool,
}

#[allow(clippy::too_many_arguments)]
fn family<S: Sync, A: Sync>(
    specs: &[S],
    name_of: &dyn Fn(&S) -> String,
    treat_of: &dyn Fn(&S) -> bool,
    bases: &[(String, A)],
    cands: &[(String, A)],
    held: &[u64],
    eval: &(dyn Fn(&A, &S, &[u64]) -> Vec<RunOut> + Sync),
    show: bool,
    tables: &mut Vec<Value>,
    tag: &str,
) -> Vec<Cell> {
    let mut cells = vec![];
    for sp in specs {
        let br: Vec<Vec<RunOut>> = bases.iter().map(|(_, a)| eval(a, sp, held)).collect();
        let cr: Vec<Vec<RunOut>> = cands.iter().map(|(_, a)| eval(a, sp, held)).collect();
        if show {
            println!("\n=== [{tag}] world {} | mean expected success (prefix H) | fail-window ===", name_of(sp));
            println!("{:<34} {:>8} {:>8} {:>8} {:>8} {:>8}", "policy", "H100", "H250", "H500", "H2000", "fail");
        }
        let mut rows = vec![];
        for ((name, _), r) in bases.iter().zip(&br).chain(cands.iter().zip(&cr)) {
            let m: Vec<f64> = (0..4).map(|i| summarize(&col(r, i)).mean).collect();
            let f = r.iter().map(|x| x.fail).sum::<f64>() / r.len() as f64;
            if show {
                println!("{:<34} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}", name, m[0], m[1], m[2], m[3], f);
            }
            rows.push(json!({"policy": name, "h": m, "fail": f}));
        }
        if show {
            tables.push(json!({"regime": tag, "world": name_of(sp), "rows": rows}));
        }
        for i in 0..4 {
            let bi = argmax_by(&br, |r| summarize(&col(r, i)).mean);
            let b = col(&br[bi], i);
            for (ci, (cname, _)) in cands.iter().enumerate() {
                let c = col(&cr[ci], i);
                let d = paired_diff(&c, &b);
                cells.push(Cell {
                    world: name_of(sp),
                    cand: cname.clone(),
                    h: HORIZONS[i],
                    diff: d.diff.mean,
                    lo: d.diff.lo,
                    hi: d.diff.hi,
                    wins: d.wins,
                    p: paired_t_pvalue(&c, &b),
                    best_base: bases[bi].0.clone(),
                    cand_mean: summarize(&c).mean,
                    base_mean: summarize(&b).mean,
                    treat: treat_of(sp),
                });
            }
        }
    }
    cells
}

fn report(title: &str, cells: &[Cell], n_seeds: usize) -> (Value, Vec<f64>) {
    let p: Vec<f64> = cells.iter().map(|c| c.p).collect();
    let adj = holm_adjust(&p);
    let mut out = vec![];
    let (mut nw, mut nl) = (0, 0);
    for (c, a) in cells.iter().zip(&adj) {
        if *a < 0.05 {
            if c.diff > 0.0 {
                nw += 1;
            } else {
                nl += 1;
            }
        }
        out.push(json!({"world": c.world, "treatment": c.treat, "candidate": c.cand, "H": c.h, "diff": c.diff, "ci": [c.lo, c.hi],
            "wins": c.wins, "n_seeds": n_seeds, "p": c.p, "holm_p": a, "best_baseline": c.best_base, "cand_mean": c.cand_mean, "base_mean": c.base_mean}));
    }
    println!("### {title}  (Holm over {} tests) -> Holm WINS {nw}, Holm LOSSES {nl}", cells.len());
    (json!({"title": title, "holm_wins": nw, "holm_losses": nl, "cells": out}), adj)
}

/// compact H=target matrix: rows worlds, columns candidates; "*" = Holm win, "!" = Holm loss
fn matrix(title: &str, cells: &[Cell], adj: &[f64], h: usize) {
    let mut worlds: Vec<String> = vec![];
    let mut cands: Vec<String> = vec![];
    for c in cells {
        if !worlds.contains(&c.world) {
            worlds.push(c.world.clone());
        }
        if !cands.contains(&c.cand) {
            cands.push(c.cand.clone());
        }
    }
    println!("\n--- {title} | H{h} diff (* Holm WIN, ! Holm LOSS) ---");
    print!("{:<24}", "world");
    for c in &cands {
        print!(" {:>20}", &c[..c.len().min(20)]);
    }
    println!();
    for w in &worlds {
        print!("{:<24}", w);
        for cn in &cands {
            match cells.iter().position(|c| &c.world == w && &c.cand == cn && c.h == h) {
                Some(i) => {
                    let m = if adj[i] < 0.05 { if cells[i].diff > 0.0 { '*' } else { '!' } } else { ' ' };
                    print!(" {:>+19.4}{}", cells[i].diff, m);
                }
                None => print!(" {:>20}", "-"),
            }
        }
        println!();
    }
}

fn parse_held(args: &[String]) -> Vec<u64> {
    match args.iter().position(|a| a == "--held").and_then(|i| args.get(i + 1)) {
        Some(r) => {
            let (a, b) = r.split_once('-').expect("--held a-b");
            (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
        }
        None => (31..=60).collect(),
    }
}

// =============================================================================================
// fixture: texts + real vectors produced offline by py/f5_embed.py

mod fx {
    use neural_mesh::knowledge::hash_embed;
    use neural_mesh::router::Task;
    use serde_json::Value;
    use std::sync::OnceLock;

    pub const DIM_REAL: usize = 384;

    pub struct Fixture {
        pub mini: Vec<Vec<f32>>,
        pub hash: Vec<Vec<f32>>,
        /// per concept: indices of question-paraphrase texts (3 subjects x 8 templates)
        pub q: Vec<Vec<usize>>,
        /// per concept: indices of bare subject phrases (3 variants)
        pub subj: Vec<Vec<usize>>,
        pub dist: Vec<usize>,
        pub concept: Vec<i32>,
        pub topic: Vec<i32>,
        pub nconc: usize,
        pub meta: Value,
    }

    static FX: OnceLock<Fixture> = OnceLock::new();
    pub fn get() -> &'static Fixture {
        FX.get_or_init(load)
    }

    fn words(s: &str) -> Vec<String> {
        s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_string()).collect()
    }

    fn load() -> Fixture {
        let meta: Value = serde_json::from_str(&std::fs::read_to_string("results/f5_corpus.json").expect("results/f5_corpus.json (run py/f5_embed.py)")).unwrap();
        let bytes = std::fs::read("results/f5_embeddings.f32").expect("results/f5_embeddings.f32");
        let items = meta["items"].as_array().unwrap();
        let n = items.len();
        assert_eq!(bytes.len(), n * DIM_REAL * 4, "fixture size mismatch");
        let mini: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..DIM_REAL).map(|j| f32::from_le_bytes(bytes[(i * DIM_REAL + j) * 4..(i * DIM_REAL + j) * 4 + 4].try_into().unwrap())).collect())
            .collect();
        let nconc = meta["concepts"].as_array().unwrap().len();
        let (mut q, mut subj, mut dist) = (vec![vec![]; nconc], vec![vec![]; nconc], vec![]);
        let (mut concept, mut topic) = (vec![], vec![]);
        let mut hash = vec![];
        for (i, it) in items.iter().enumerate() {
            let c = it["concept"].as_i64().unwrap() as i32;
            concept.push(c);
            topic.push(it["topic"].as_i64().unwrap() as i32);
            match it["kind"].as_str().unwrap() {
                "q" => q[c as usize].push(i),
                "subj" => subj[c as usize].push(i),
                _ => dist.push(i),
            }
            let t = Task { id: 0, task_type: None, features: words(it["text"].as_str().unwrap()) };
            hash.push(hash_embed(&t, DIM_REAL));
        }
        Fixture { mini, hash, q, subj, dist, concept, topic, nconc, meta }
    }
}

// =============================================================================================
mod kpart {
    use super::*;
    use std::cell::RefCell;
    use std::collections::{HashMap, VecDeque};
    use std::rc::Rc;

    use neural_mesh::addons::{Channel, Shaper, ShaperConfig, Transform};
    use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
    use neural_mesh::coupling::heb_robust_cfg;
    use neural_mesh::hebbian::HebbianRouter;
    use neural_mesh::knowledge::{AnchorRouter, FeatMode, Knowledge, KnowledgeConfig, KnowledgeRouter, VoteConfig};
    use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
    use neural_mesh::router::{Router, Task};
    use neural_mesh::sim::{Scenario, World, WorldConfig};

    const SDIM: usize = 48;
    const NA: usize = 6;

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Src {
        Mini,
        Hash,
        Synth,
    }
    impl Src {
        pub fn label(self) -> &'static str {
            match self {
                Src::Mini => "mini",
                Src::Hash => "hash",
                Src::Synth => "synth",
            }
        }
    }
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum Corp {
        Q,
        Subj,
    }

    /// e2 synthetic embedding model (copied)
    #[derive(Clone, Debug)]
    pub struct EmbCfg {
        pub g: f64,
        pub h: f64,
        pub ntop: usize,
        pub p: f64,
        pub m: usize,
        pub spread: f64,
        pub noise: f64,
    }

    #[derive(Clone, Debug)]
    pub struct Spec {
        pub name: String,
        pub src: Src,
        pub corp: Corp,
        pub scenario: Scenario,
        /// recurring concepts; 0 = unique context per task
        pub k: usize,
        pub w: f64,
        /// fraction of tasks that are unrelated one-off distractor texts (no context effect)
        pub mixed: f64,
        pub e: EmbCfg,
        pub treatment: bool,
    }
    impl Spec {
        pub fn dim(&self) -> usize {
            if self.src == Src::Synth { SDIM } else { fx::DIM_REAL }
        }
    }

    pub struct CtxWorld {
        pub w: World,
        pub emb: Vec<Vec<f32>>,
        pctx: Vec<[f64; NA]>,
        ctx: Vec<usize>,
        wt: Vec<f64>,
        /// ground-truth concept per task (-1 = distractor / unique)
        pub truth: Vec<i32>,
        pub topic: Vec<i32>,
        draws: Vec<(f64, f64, f64)>,
    }

    fn spectrum(p: f64) -> Vec<f64> {
        let a: Vec<f64> = (0..SDIM).map(|i| ((i + 1) as f64).powf(-p)).collect();
        let n = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        a.into_iter().map(|x| x / n).collect()
    }
    fn avec(r: &mut Rng, a: &[f64], scale: f64) -> Vec<f64> {
        a.iter().map(|ai| ai * normal(r) * scale).collect()
    }
    fn add(v: &mut [f64], u: &[f64]) {
        for (x, y) in v.iter_mut().zip(u) {
            *x += y;
        }
    }
    fn unit(v: &[f64]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        v.iter().map(|x| (x / n) as f32).collect()
    }
    fn draw_pctx(r: &mut Rng, kk: usize) -> Vec<[f64; NA]> {
        let mut pctx = Vec::with_capacity(kk);
        for _ in 0..kk {
            let best = r.below(NA);
            let mut row = [0.0; NA];
            for (ai, p) in row.iter_mut().enumerate() {
                *p = if ai == best { 0.80 + 0.10 * r.f64() } else { 0.15 + 0.35 * r.f64() };
            }
            pctx.push(row);
        }
        pctx
    }

    impl CtxWorld {
        pub fn new(seed: u64, sp: &Spec) -> Self {
            let w = World::new(seed, WorldConfig { n_tasks: 2000, scenario: sp.scenario, ..WorldConfig::default() });
            let n = w.len();
            let (mut ctx, mut wt, mut truth, mut topic, mut emb) = (vec![], vec![], vec![], vec![], vec![]);
            let pctx;
            let draws;
            if sp.src == Src::Synth {
                let e = &sp.e;
                let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xE2E2_1234));
                let a = spectrum(e.p);
                let kk = if sp.k == 0 { n } else { sp.k };
                let shared: Vec<f64> = {
                    let v: Vec<f64> = a.iter().map(|ai| ai * (1.0 + 0.3 * normal(&mut r))).collect();
                    let nn = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                    v.into_iter().map(|x| e.g * x / nn).collect()
                };
                let tops: Vec<Vec<f64>> = (0..e.ntop.max(1)).map(|_| avec(&mut r, &a, e.h)).collect();
                let cents: Vec<Vec<f64>> = (0..kk)
                    .map(|k| {
                        let mut c = avec(&mut r, &a, 1.0);
                        add(&mut c, &shared);
                        add(&mut c, &tops[k % tops.len()]);
                        c
                    })
                    .collect();
                let subs: Vec<Vec<Vec<f64>>> = (0..kk)
                    .map(|_| (0..e.m.max(1)).map(|_| if e.m > 1 { avec(&mut r, &a, e.spread) } else { vec![0.0; SDIM] }).collect())
                    .collect();
                let _dirs: Vec<Vec<f64>> = (0..kk).map(|_| avec(&mut r, &a, 1.0)).collect();
                pctx = draw_pctx(&mut r, kk);
                for t in 0..n {
                    if sp.mixed > 0.0 && r.f64() < sp.mixed {
                        // unrelated one-off text: own centroid, shares only the global direction
                        let mut v = avec(&mut r, &a, 1.0);
                        add(&mut v, &shared);
                        let tp = avec(&mut r, &a, e.h);
                        add(&mut v, &tp);
                        let nz = avec(&mut r, &a, e.noise);
                        add(&mut v, &nz);
                        emb.push(unit(&v));
                        ctx.push(usize::MAX);
                        wt.push(0.0);
                        truth.push(-1);
                        topic.push(-1);
                        continue;
                    }
                    let c = if sp.k == 0 { t } else { r.below(kk) };
                    let j = r.below(e.m.max(1));
                    ctx.push(c);
                    wt.push(sp.w);
                    truth.push(if sp.k == 0 { -1 } else { c as i32 });
                    topic.push(if sp.k == 0 { -1 } else { (c % tops.len()) as i32 });
                    let mut v = cents[c].clone();
                    add(&mut v, &subs[c][j]);
                    let nz = avec(&mut r, &a, e.noise);
                    add(&mut v, &nz);
                    emb.push(unit(&v));
                }
                draws = (0..n).map(|_| (r.f64(), r.f64(), r.f64())).collect();
            } else {
                let f = fx::get();
                let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xF5F5_1234));
                let kk = if sp.k == 0 { n } else { sp.k.min(f.nconc) };
                pctx = draw_pctx(&mut r, kk);
                let mut pool = f.dist.clone();
                for i in (1..pool.len()).rev() {
                    let j = r.below(i + 1);
                    pool.swap(i, j);
                }
                let mut di = 0usize;
                let vec_of = |i: usize| -> Vec<f32> {
                    if sp.src == Src::Mini { f.mini[i].clone() } else { f.hash[i].clone() }
                };
                for t in 0..n {
                    let isd = sp.k == 0 || (sp.mixed > 0.0 && r.f64() < sp.mixed);
                    if isd {
                        let idx = pool[di % pool.len()];
                        di += 1;
                        emb.push(vec_of(idx));
                        ctx.push(if sp.k == 0 { t } else { usize::MAX });
                        wt.push(if sp.k == 0 { sp.w } else { 0.0 });
                        truth.push(-1);
                        topic.push(-1);
                    } else {
                        let c = r.below(kk);
                        let texts = if sp.corp == Corp::Q { &f.q[c] } else { &f.subj[c] };
                        let idx = texts[r.below(texts.len())];
                        emb.push(vec_of(idx));
                        ctx.push(c);
                        wt.push(sp.w);
                        truth.push(c as i32);
                        topic.push(f.topic[idx]);
                    }
                }
                draws = (0..n).map(|_| (r.f64(), r.f64(), r.f64())).collect();
            }
            CtxWorld { w, emb, pctx, ctx, wt, truth, topic, draws }
        }
        pub fn p(&self, t: usize, a: usize) -> f64 {
            if self.w.is_down(t, a) {
                return 0.0;
            }
            let ty = self.w.task(t).type_idx;
            let base = self.w.skill(t, ty, a);
            let wg = self.wt[t];
            if wg == 0.0 { base } else { (1.0 - wg) * base + wg * self.pctx[self.ctx[t]][a] }
        }
        fn outcome(&self, t: usize, a: usize) -> f64 {
            if self.w.task(t).u < self.p(t, a) { 1.0 } else { 0.0 }
        }
    }

    fn cosf(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum()
    }

    /// geometry diagnostics (seed 1): mean cosine over sampled pairs by relation.
    pub fn geometry(sp: &Spec) -> Value {
        let cw = CtxWorld::new(1, sp);
        let n = cw.emb.len();
        let mut r = Rng::new(99);
        let mut acc = [(0.0, 0.0); 5]; // same concept, sibling(same topic), diff topic, dist-dist, dist-concept
        for _ in 0..6000 {
            let (i, j) = (r.below(n), r.below(n));
            if i == j {
                continue;
            }
            let (ti, tj) = (cw.truth[i], cw.truth[j]);
            let slot = match (ti >= 0, tj >= 0) {
                (true, true) => {
                    if ti == tj { 0 } else if cw.topic[i] == cw.topic[j] { 1 } else { 2 }
                }
                (false, false) => 3,
                _ => 4,
            };
            acc[slot].0 += cosf(&cw.emb[i], &cw.emb[j]);
            acc[slot].1 += 1.0;
        }
        let m = |x: (f64, f64)| if x.1 > 0.0 { x.0 / x.1 } else { f64::NAN };
        json!({"same_concept": m(acc[0]), "sibling_concept": m(acc[1]), "diff_topic": m(acc[2]), "dist_dist": m(acc[3]), "dist_concept": m(acc[4])})
    }

    // -------- causal online mean-centering (identical to e2's KnPol) --------
    pub struct Centerer {
        mu: Vec<f32>,
        n: f64,
    }
    impl Centerer {
        pub fn new(dim: usize) -> Self {
            Centerer { mu: vec![0.0; dim], n: 0.0 }
        }
        pub fn apply(&mut self, e: &[f32]) -> Vec<f32> {
            let v: Vec<f32> = if self.n > 0.0 { e.iter().zip(&self.mu).map(|(x, m)| x - m).collect() } else { e.to_vec() };
            let nn = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            let alpha = (1.0 / (self.n + 1.0)).max(0.01) as f32;
            for (m, x) in self.mu.iter_mut().zip(e) {
                *m += alpha * (x - *m);
            }
            self.n += 1.0;
            v.into_iter().map(|x| x / nn).collect()
        }
    }

    // -------- retrieval-layer metrics vs ground-truth concept (no routing) --------
    #[derive(Clone, Debug, Default)]
    pub struct Prf {
        pub nbr_p: f64,
        pub nbr_r: f64,
        pub nbr_f1: f64,
        pub anc_p: f64,
        pub anc_r: f64,
        pub anc_f1: f64,
        /// distinct anchors among concept tasks / number of concepts (1.0 = perfect, >1 fragmentation, <1 merging)
        pub frag: f64,
        /// fraction of distractor tasks that retrieved >= 1 neighbour >= sim_min (spurious context)
        pub spurious: f64,
        /// fraction of distractor tasks whose anchor also holds a concept task (contamination)
        pub contam: f64,
    }

    pub fn retrieval_metrics(sp: &Spec, seed: u64, k: usize, sim_min: f32, center: bool) -> Prf {
        let cw = CtxWorld::new(seed, sp);
        let n = cw.emb.len();
        let mut kc = KnowledgeConfig::new(sp.dim());
        kc.k = k;
        kc.sim_min = sim_min;
        kc.feat = FeatMode::Off;
        let mut kn = Knowledge::new_shared(kc);
        let mut cen = Centerer::new(sp.dim());
        let mut anchor = vec![0u64; n];
        let (mut tp, mut ret, mut poss) = (0.0, 0.0, 0.0);
        let (mut dq, mut dsp) = (0.0, 0.0);
        let mut same_prev: HashMap<i32, usize> = HashMap::new();
        for t in 0..n {
            let q = if center { cen.apply(&cw.emb[t]) } else { cw.emb[t].clone() };
            let r = kn.retrieve(&q);
            if t > 0 {
                let kept: Vec<usize> = r.hits.iter().filter(|h| h.1 >= sim_min).map(|h| h.0 as usize).collect();
                if cw.truth[t] >= 0 {
                    ret += kept.len() as f64;
                    tp += kept.iter().filter(|&&i| cw.truth[i] == cw.truth[t]).count() as f64;
                    poss += (*same_prev.get(&cw.truth[t]).unwrap_or(&0)).min(k) as f64;
                } else {
                    dq += 1.0;
                    if !kept.is_empty() {
                        dsp += 1.0;
                    }
                }
            }
            anchor[t] = r.anchor.unwrap_or(t as u64);
            kn.remember(&q, &r);
            if cw.truth[t] >= 0 {
                *same_prev.entry(cw.truth[t]).or_insert(0) += 1;
            }
        }
        let pr = if ret > 0.0 { tp / ret } else { 0.0 };
        let rc = if poss > 0.0 { tp / poss } else { 0.0 };
        let f1 = |p: f64, r: f64| if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
        // anchor pairwise clustering P/R over concept tasks
        let mut by_a: HashMap<u64, usize> = HashMap::new();
        let mut by_c: HashMap<i32, usize> = HashMap::new();
        let mut by_ac: HashMap<(u64, i32), usize> = HashMap::new();
        for t in 0..n {
            if cw.truth[t] >= 0 {
                *by_a.entry(anchor[t]).or_insert(0) += 1;
                *by_c.entry(cw.truth[t]).or_insert(0) += 1;
                *by_ac.entry((anchor[t], cw.truth[t])).or_insert(0) += 1;
            }
        }
        let c2 = |x: usize| (x * x.saturating_sub(1) / 2) as f64;
        let (sa, sc, sac) = (by_a.values().map(|&x| c2(x)).sum::<f64>(), by_c.values().map(|&x| c2(x)).sum::<f64>(), by_ac.values().map(|&x| c2(x)).sum::<f64>());
        let (ap, ar) = (if sa > 0.0 { sac / sa } else { 0.0 }, if sc > 0.0 { sac / sc } else { 0.0 });
        let nconc = by_c.len().max(1) as f64;
        let (mut dn, mut dc) = (0.0, 0.0);
        for t in 0..n {
            if cw.truth[t] < 0 {
                dn += 1.0;
                if by_a.contains_key(&anchor[t]) {
                    dc += 1.0;
                }
            }
        }
        Prf {
            nbr_p: pr,
            nbr_r: rc,
            nbr_f1: f1(pr, rc),
            anc_p: ap,
            anc_r: ar,
            anc_f1: f1(ap, ar),
            frag: by_a.len() as f64 / nconc,
            spurious: if dq > 0.0 { dsp / dq } else { f64::NAN },
            contam: if dn > 0.0 { dc / dn } else { f64::NAN },
        }
    }

    // -------- policies (copied from e2) --------
    pub trait Pol {
        fn pick(&mut self, t: &Task, emb: &[f32], c: &[usize], rng: &mut Rng) -> usize;
        fn fb(&mut self, t: &Task, a: usize, r: Option<f64>);
    }
    struct Plain<R: Router>(R);
    impl<R: Router> Pol for Plain<R> {
        fn pick(&mut self, t: &Task, _e: &[f32], c: &[usize], rng: &mut Rng) -> usize {
            self.0.select(t, c, rng)
        }
        fn fb(&mut self, t: &Task, a: usize, r: Option<f64>) {
            if let Some(r) = r {
                self.0.update(t, a, r)
            }
        }
    }
    struct KnPol<R: Router> {
        r: KnowledgeRouter<R>,
        center: Option<Centerer>,
    }
    impl<R: Router> Pol for KnPol<R> {
        fn pick(&mut self, t: &Task, e: &[f32], c: &[usize], rng: &mut Rng) -> usize {
            match &mut self.center {
                Some(cn) => {
                    let q = cn.apply(e);
                    self.r.set_context(&q)
                }
                None => self.r.set_context(e),
            }
            self.r.select(t, c, rng)
        }
        fn fb(&mut self, t: &Task, a: usize, r: Option<f64>) {
            if let Some(r) = r {
                self.r.update(t, a, r)
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct KnCfg {
        pub label: String,
        pub feat: FeatMode,
        pub k: usize,
        pub sim_min: f32,
        pub vote: Option<(f64, f64, Option<usize>, Option<(f64, f64, usize)>)>,
        pub center: bool,
    }

    #[derive(Clone, Debug)]
    pub enum Kind {
        Keyword,
        Eps,
        Pts,
        HebR,
        Warm,
        Guard,
        Kn(KnCfg),
    }
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Signal {
        Outcome,
        Raw,
        Reuse,
        /// shaper part: raw hook bit
        ShRaw,
        /// shaper part: shaped graded reward from proxy channels
        Shaped,
    }
    pub type A = (Kind, Signal);

    fn x1_guard() -> GuardConfig {
        GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
    }

    fn build(k: &Kind, w: &World, dim: usize) -> Box<dyn Pol> {
        let kw = w.keyword_map_for();
        match k {
            Kind::Keyword => Box::new(Plain(KeywordRouter::new(kw, NA))),
            Kind::Eps => Box::new(Plain(EpsGreedyRouter::new(NA, 0.99, 0.02))),
            Kind::Pts => Box::new(Plain(PooledThompsonRouter::new(NA, 1.0))),
            Kind::HebR => Box::new(Plain(HebbianRouter::new(NA, heb_robust_cfg()))),
            Kind::Warm => Box::new(Plain(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, 1.0).with_evidence(Evidence::MeanFeature))),
            Kind::Guard => Box::new(Plain(Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, x1_guard()))),
            Kind::Kn(c) => {
                let mut kc = KnowledgeConfig::new(dim);
                kc.k = c.k;
                kc.sim_min = c.sim_min;
                kc.feat = c.feat;
                kc.vote = c.vote.map(|(min_n, tau, ex, gate)| VoteConfig { min_n, tau, prior: (1.0, 1.0), explore_after: ex, gate });
                let kn = Knowledge::new_shared(kc);
                let anchors = Rc::new(RefCell::new(HashMap::new()));
                let center = if c.center { Some(Centerer::new(dim)) } else { None };
                if c.feat == FeatMode::Hidden {
                    let inner = AnchorRouter::new(HebbianRouter::new(NA, heb_robust_cfg()), anchors.clone());
                    Box::new(KnPol { r: KnowledgeRouter::new(Guarded::new(inner, kw, NA, x1_guard()), kn, anchors), center })
                } else {
                    let front = Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, x1_guard());
                    Box::new(KnPol { r: KnowledgeRouter::new(front, kn, anchors), center })
                }
            }
        }
    }

    #[derive(Clone, Debug)]
    struct Weak {
        tool_base: f64,
        tool_slope: f64,
        reuse_frac: f64,
        reuse_delay: usize,
        reuse_noise: f64,
    }
    const WEAK: Weak = Weak { tool_base: 0.01, tool_slope: 0.03, reuse_frac: 0.3, reuse_delay: 25, reuse_noise: 0.1 };

    pub fn run(k: &Kind, sp: &Spec, sig: Signal, seed: u64) -> RunOut {
        let cw = CtxWorld::new(seed, sp);
        let w = &cw.w;
        let cands: Vec<usize> = (0..NA).collect();
        let mut pol = build(k, w, sp.dim());
        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
        let mut pending: VecDeque<(usize, usize, usize, f64)> = VecDeque::new();
        let (mut cum, mut fs, mut fnn) = (0.0f64, 0.0f64, 0.0f64);
        let mut out = RunOut::default();
        for t in 0..w.len() {
            while pending.front().is_some_and(|p| p.0 <= t) {
                let (_, ti, a, v) = pending.pop_front().unwrap();
                pol.fb(&w.task(ti).task, a, Some(v));
            }
            let st = w.task(t);
            let a = pol.pick(&st.task, &cw.emb[t], &cands, &mut rng);
            let pe = cw.p(t, a);
            let s = cw.outcome(t, a);
            match sig {
                Signal::Outcome => pol.fb(&st.task, a, Some(s)),
                _ => {
                    let (u_tool, u_reuse, u_noise) = cw.draws[t];
                    let raw_ok = !w.is_down(t, a) && u_tool >= WEAK.tool_base + WEAK.tool_slope * (1.0 - s);
                    match sig {
                        Signal::Raw => pol.fb(&st.task, a, Some(if raw_ok { 1.0 } else { 0.0 })),
                        _ => {
                            pol.fb(&st.task, a, if raw_ok { None } else { Some(0.0) });
                            if raw_ok && u_reuse < WEAK.reuse_frac {
                                let v = if u_noise < WEAK.reuse_noise { 1.0 - s } else { s };
                                pending.push_back((t + WEAK.reuse_delay, t, a, v));
                            }
                        }
                    }
                }
            }
            cum += pe;
            if w.in_failure(t) {
                fs += pe;
                fnn += 1.0;
            }
            for (i, &hz) in HORIZONS.iter().enumerate() {
                if t + 1 == hz {
                    out.h[i] = cum / hz as f64;
                }
            }
        }
        out.fail = fs / fnn.max(1.0);
        out
    }

    fn eval(a: &A, sp: &Spec, seeds: &[u64]) -> Vec<RunOut> {
        seeds.par_iter().map(|&s| if matches!(a.1, Signal::ShRaw | Signal::Shaped) { run_shaper(&a.0, sp, a.1, s) } else { run(&a.0, sp, a.1, s) }).collect()
    }

    // =========================================================================================
    // worlds

    fn worlds() -> Vec<Spec> {
        let re = |noise: f64| EmbCfg { g: 1.2, h: 0.8, ntop: 4, p: 0.5, m: 1, spread: 0.0, noise };
        let para = |noise: f64| EmbCfg { m: 4, spread: 0.9, ..re(noise) };
        let mut v = vec![];
        let mut mk = |name: String, src: Src, corp: Corp, sc: Scenario, k: usize, w: f64, mixed: f64, e: EmbCfg, tr: bool| {
            v.push(Spec { name, src, corp, scenario: sc, k, w, mixed, e, treatment: tr });
        };
        use Scenario::{Fixed, Random};
        for src in [Src::Mini, Src::Hash] {
            let p = src.label();
            let e = re(0.0);
            mk(format!("{p}-q"), src, Corp::Q, Fixed, 12, 0.8, 0.0, e.clone(), true);
            mk(format!("{p}-subj3"), src, Corp::Subj, Fixed, 12, 0.8, 0.0, e.clone(), true);
            mk(format!("{p}-q-mild"), src, Corp::Q, Random, 12, 0.5, 0.0, e.clone(), true);
            mk(format!("{p}-q-mixed50"), src, Corp::Q, Fixed, 12, 0.8, 0.5, e.clone(), true);
            mk(format!("{p}-ctl-irrel"), src, Corp::Q, Fixed, 12, 0.0, 0.0, e.clone(), false);
            mk(format!("{p}-ctl-unique"), src, Corp::Q, Fixed, 0, 0.8, 0.0, e.clone(), false);
            mk(format!("{p}-ctl-none"), src, Corp::Q, Random, 0, 0.0, 0.0, e.clone(), false);
        }
        let s = Src::Synth;
        mk("synth-aniso-n0.8".into(), s, Corp::Q, Fixed, 12, 0.8, 0.0, re(0.8), true);
        mk("synth-para-n0.8".into(), s, Corp::Q, Fixed, 12, 0.8, 0.0, para(0.8), true);
        mk("synth-para-mixed50".into(), s, Corp::Q, Fixed, 12, 0.8, 0.5, para(0.8), true);
        mk("synth-ctl-irrel".into(), s, Corp::Q, Fixed, 12, 0.0, 0.0, para(0.8), false);
        mk("synth-ctl-unique".into(), s, Corp::Q, Fixed, 0, 0.8, 0.0, re(0.8), false);
        mk("synth-ctl-none".into(), s, Corp::Q, Random, 0, 0.0, 0.0, re(0.8), false);
        v
    }

    fn cfg_grid() -> Vec<KnCfg> {
        let mut v = vec![];
        for &k in &[5usize, 8] {
            for &sm in &[0.5f32, 0.6, 0.7, 0.8] {
                for &c in &[false, true] {
                    v.push(KnCfg { label: format!("feat-hid k{k} s{sm} c{c}"), feat: FeatMode::Hidden, k, sim_min: sm, vote: None, center: c });
                    for &(mn, tau) in &[(2.0, 0.7), (3.0, 0.7)] {
                        v.push(KnCfg {
                            label: format!("dir+vote k{k} s{sm} n{mn} t{tau} c{c}"),
                            feat: FeatMode::Direct,
                            k,
                            sim_min: sm,
                            vote: Some((mn, tau, None, Some((6.0, 0.02, 25)))),
                            center: c,
                        });
                    }
                }
            }
        }
        v
    }

    fn arm(label: &str, c: &KnCfg, sg: Signal) -> (String, A) {
        let mut c = c.clone();
        c.label = label.into();
        (label.to_string(), (Kind::Kn(c), sg))
    }

    // =========================================================================================
    // shaper-adjacent: shaped proxy reward feeding the front, real context embeddings

    struct ObsCfg {
        sf: f64,
        kappa: f64,
        lat_sigma: f64,
        speed_spread: f64,
        retry: (f64, f64),
        len: (f64, f64, f64),
    }
    const RICH: ObsCfg = ObsCfg { sf: 1.0, kappa: 0.6, lat_sigma: 0.35, speed_spread: 0.3, retry: (0.15, 1.2), len: (0.6, 0.5, 0.3) };
    const USELESS: ObsCfg = ObsCfg { sf: 0.0, ..RICH };

    fn poisson_inv(lambda: f64, u: f64) -> f64 {
        let (mut k, mut p) = (0.0, (-lambda).exp());
        let mut cdf = p;
        while u > cdf && k < 20.0 {
            k += 1.0;
            p *= lambda / k;
            cdf += p;
        }
        k
    }

    fn shaper_d4() -> Shaper {
        let ch = |name: &str, tr, dir, df, w| Channel { name: name.into(), transform: tr, dir, debias_feature: df, debias_agent: 0.0, weight: w };
        Shaper::new(ShaperConfig {
            enabled: true,
            channels: vec![ch("retry", Transform::Log1p, -1, false, 1.0), ch("lat", Transform::Ln, -1, true, 1.0), ch("len", Transform::Ln, 0, true, 1.0)],
            center: 0.6,
            gain: 0.2,
            fail_reward: 0.0,
            factor: false,
            warmup: 100,
            alpha_min: 0.005,
        })
    }

    /// proxies = e2 'rich' (or 'useless') channel model; the outcome s is the world's realised success
    pub fn run_shaper(k: &Kind, sp: &Spec, sig: Signal, seed: u64) -> RunOut {
        let cw = CtxWorld::new(seed, sp);
        let w = &cw.w;
        let cands: Vec<usize> = (0..NA).collect();
        let mut pol = build(k, w, sp.dim());
        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
        let c = if sp.name.contains("useless") { &USELESS } else { &RICH };
        let mut r0 = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(0xF5));
        let mut base_lat = [0.0; 8];
        let mut base_len = [0.0; 8];
        for i in 0..8 {
            base_lat[i] = (r0.f64() * 2.0 - 0.7).exp();
            base_len[i] = (r0.f64() * 2.0 - 0.7).exp() * 100.0;
        }
        let speed: Vec<f64> = (0..NA).map(|_| 1.0 - c.speed_spread / 2.0 + c.speed_spread * r0.f64()).collect();
        let style: Vec<f64> = (0..NA).map(|_| (c.len.2 * (r0.f64() - 0.5)).exp()).collect();
        let mut shaper = shaper_d4();
        let (mut cum, mut fs, mut fnn) = (0.0f64, 0.0f64, 0.0f64);
        let mut out = RunOut::default();
        for t in 0..w.len() {
            let st = w.task(t);
            let a = pol.pick(&st.task, &cw.emb[t], &cands, &mut rng);
            let pe = cw.p(t, a);
            let s = cw.outcome(t, a);
            let (u_tool, ur) = (r0.f64(), r0.f64());
            let (zl, zn) = (normal(&mut r0), normal(&mut r0));
            let ok = !w.is_down(t, a) && u_tool >= WEAK.tool_base + WEAK.tool_slope * (1.0 - s);
            // proxies depend on the realised quality (1 - pe as in e2: expected, not realised)
            let q = c.sf * (1.0 - pe);
            let lat = base_lat[st.type_idx] * speed[a] * (c.lat_sigma * zl).exp() * (1.0 + c.kappa * q);
            let retry = poisson_inv(((c.retry.0 + c.retry.1 * q) * 1.0).max(1e-6), ur);
            let len = base_len[st.type_idx] * style[a] * (c.len.1 * zn - c.len.0 * q).exp();
            let r = match sig {
                Signal::Shaped => shaper.shape(&st.task, a, ok, &[Some(retry), Some(lat), Some(len)]),
                _ => {
                    if ok { 1.0 } else { 0.0 }
                }
            };
            pol.fb(&st.task, a, Some(r));
            cum += pe;
            if w.in_failure(t) {
                fs += pe;
                fnn += 1.0;
            }
            for (i, &hz) in HORIZONS.iter().enumerate() {
                if t + 1 == hz {
                    out.h[i] = cum / hz as f64;
                }
            }
        }
        out.fail = fs / fnn.max(1.0);
        out
    }

    // =========================================================================================

    pub fn main(part: &str, held: &[u64], out_k: &str, out_r: &str, out_s: &str) {
        let t0 = Instant::now();
        let tune: Vec<u64> = (1..=8).collect();
        let all_specs = worlds();
        let f = fx::get();
        println!(
            "f5: held {}-{} ({} seeds), tune seeds 1-8, {} worlds; corpus {} texts, {} concepts, model sha {}",
            held[0],
            held[held.len() - 1],
            held.len(),
            all_specs.len(),
            f.meta["n"],
            f.nconc,
            &f.meta["model_sha256"].as_str().unwrap()[..12]
        );
        let geo: Vec<Value> = all_specs.iter().map(|s| json!({"world": s.name, "geometry": geometry(s)})).collect();
        println!("\n--- embedding geometry (seed 1; mean cosine of sampled pairs) ---");
        for g in &geo {
            println!("{:<22} {}", g["world"].as_str().unwrap(), g["geometry"]);
        }

        // ---------------- retrieval P/R/F1 ----------------
        if part == "retrieval" || part == "all" {
            let rsp: Vec<&Spec> = all_specs.iter().filter(|s| s.treatment && (s.name.ends_with("-q") || s.name.ends_with("-subj3") || s.name.ends_with("mixed50") || s.name.ends_with("para-n0.8") || s.name.ends_with("aniso-n0.8"))).collect();
            let mut rows = vec![];
            println!("\n=== RETRIEVAL layer vs ground-truth concept (held seeds; no routing) ===");
            println!("nbr = neighbours>=sim_min with same concept (micro P/R over concept queries); anc = pairwise anchor clustering P/R over concept tasks; frag = distinct anchors/concepts; spur = distractor tasks with >=1 retained neighbour; contam = distractor tasks sharing an anchor with a concept task");
            println!("{:<20} {:>3} {:>4} {:>5} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>6}", "world", "k", "sim", "ctr", "nbrP", "nbrR", "nbrF1", "ancP", "ancR", "ancF1", "frag", "spur", "contam");
            for sp in &rsp {
                for &k in &[5usize, 8] {
                    for &sm in &[0.5f32, 0.6, 0.7, 0.8] {
                        for &ce in &[false, true] {
                            let rs: Vec<Prf> = held.par_iter().map(|&s| retrieval_metrics(sp, s, k, sm, ce)).collect();
                            let m = |g: &dyn Fn(&Prf) -> f64| -> f64 {
                                let v: Vec<f64> = rs.iter().map(g).filter(|x| x.is_finite()).collect();
                                if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
                            };
                            let row = [m(&|p| p.nbr_p), m(&|p| p.nbr_r), m(&|p| p.nbr_f1), m(&|p| p.anc_p), m(&|p| p.anc_r), m(&|p| p.anc_f1), m(&|p| p.frag), m(&|p| p.spurious), m(&|p| p.contam)];
                            if k == 8 || (sm == 0.7 && !ce) {
                                println!(
                                    "{:<20} {:>3} {:>4} {:>5} | {:>5.2} {:>5.2} {:>5.2} | {:>5.2} {:>5.2} {:>5.2} | {:>5.2} {:>5.2} {:>6.2}",
                                    sp.name, k, sm, ce, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
                                );
                            }
                            rows.push(json!({"world": sp.name, "src": sp.src.label(), "k": k, "sim_min": sm, "center": ce, "nbr_p": row[0], "nbr_r": row[1], "nbr_f1": row[2],
                                "anchor_p": row[3], "anchor_r": row[4], "anchor_f1": row[5], "frag": row[6], "spurious_dist": row[7], "contam": row[8]}));
                        }
                    }
                }
            }
            let doc = json!({"lab": "f5_real_embed/retrieval", "held": [held[0], held[held.len() - 1]], "geometry": geo, "rows": rows, "elapsed_s": t0.elapsed().as_secs_f64()});
            std::fs::write(out_r, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
            println!("wrote {out_r}");
        }

        if part == "retrieval" {
            return;
        }

        // ---------------- knowledge benchmark per source ----------------
        let base_w: Vec<(String, A)> = vec![
            ("keyword".into(), (Kind::Keyword, Signal::Raw)),
            ("eps-greedy".into(), (Kind::Eps, Signal::Raw)),
            ("pooled-ts".into(), (Kind::Pts, Signal::Raw)),
            ("hebbian-robust".into(), (Kind::HebR, Signal::Raw)),
            ("warm(hebr)".into(), (Kind::Warm, Signal::Raw)),
            ("guarded(hebr)".into(), (Kind::Guard, Signal::Raw)),
            ("guarded(hebr)+reuse".into(), (Kind::Guard, Signal::Reuse)),
            ("warm(hebr)+reuse".into(), (Kind::Warm, Signal::Reuse)),
        ];
        let base_o: Vec<(String, A)> = vec![
            ("keyword".into(), (Kind::Keyword, Signal::Outcome)),
            ("eps-greedy".into(), (Kind::Eps, Signal::Outcome)),
            ("pooled-ts".into(), (Kind::Pts, Signal::Outcome)),
            ("hebbian-robust".into(), (Kind::HebR, Signal::Outcome)),
            ("warm(hebr)".into(), (Kind::Warm, Signal::Outcome)),
            ("guarded(hebr)".into(), (Kind::Guard, Signal::Outcome)),
        ];
        let d6_feat = KnCfg { label: String::new(), feat: FeatMode::Hidden, k: 8, sim_min: 0.6, vote: None, center: false };
        let d6_vote = KnCfg { label: String::new(), feat: FeatMode::Direct, k: 8, sim_min: 0.6, vote: Some((2.0, 0.7, None, Some((6.0, 0.02, 25)))), center: false };
        let nm = |s: &Spec| s.name.clone();
        let tr = |s: &Spec| s.treatment;

        let mut doc = serde_json::Map::new();
        let mut tables = vec![];
        let mut selected = serde_json::Map::new();
        let mut retuned: HashMap<&'static str, (KnCfg, KnCfg)> = HashMap::new();
        let ns = held.len();
        let run_kn = part == "knowledge" || part == "all" || part == "shaper";
        if run_kn {
            for src in [Src::Mini, Src::Hash, Src::Synth] {
                let specs: Vec<Spec> = all_specs.iter().filter(|s| s.src == src).cloned().collect();
                let tune_bases: Vec<Vec<Vec<RunOut>>> = specs.iter().map(|s| base_w.iter().map(|(_, a)| eval(a, s, &tune)).collect()).collect();
                let grid = cfg_grid();
                let scored: Vec<(f64, KnCfg)> = grid
                    .par_iter()
                    .map(|c| {
                        let (mut margins, mut tot) = (vec![], 0.0);
                        for (wi, s) in specs.iter().enumerate() {
                            let r = eval(&(Kind::Kn(c.clone()), Signal::Reuse), s, &tune);
                            for i in 0..4 {
                                let best = tune_bases[wi].iter().map(|b| summarize(&col(b, i)).mean).fold(f64::NEG_INFINITY, f64::max);
                                let m = summarize(&col(&r, i)).mean - best;
                                margins.push(m);
                                tot += m;
                            }
                        }
                        let neg: f64 = margins.iter().map(|m| m.min(0.0)).sum::<f64>() / margins.len() as f64;
                        (tot / margins.len() as f64 + neg, c.clone())
                    })
                    .collect();
                let pick = |pred: &dyn Fn(&KnCfg) -> bool| -> (f64, KnCfg) { scored.iter().filter(|(_, c)| pred(c)).max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).cloned().unwrap() };
                let (of, hid) = pick(&|c| c.feat == FeatMode::Hidden);
                let (ov, vote) = pick(&|c| c.feat == FeatMode::Direct);
                println!("\n[tune:{}] retuned feat(hidden) {:?} obj {:+.4}\n[tune:{}] retuned dir+vote {:?} obj {:+.4}", src.label(), hid, of, src.label(), vote, ov);
                selected.insert(format!("{}_feat", src.label()), json!(format!("{:?}", hid)));
                selected.insert(format!("{}_vote", src.label()), json!(format!("{:?}", vote)));
                retuned.insert(src.label(), (hid.clone(), vote.clone()));
                if part == "shaper" {
                    continue;
                }

                let ev = |a: &A, s: &Spec, seeds: &[u64]| eval(a, s, seeds);
                for (reg, bases, sg) in [("O", &base_o, Signal::Outcome), ("W", &base_w, Signal::Reuse)] {
                    let cands: Vec<(String, A)> = vec![
                        arm("kn-feat(d6cfg)", &d6_feat, sg),
                        arm("kn-d+vote(d6cfg)", &d6_vote, sg),
                        arm("kn-feat(retuned)", &hid, sg),
                        arm("kn-d+vote(retuned)", &vote, sg),
                    ];
                    let tag = format!("{}-{}", src.label(), reg);
                    let cells = family(&specs, &nm, &tr, bases, &cands, held, &ev, true, &mut tables, &tag);
                    let (rep, adj) = report(&format!("[{}] REGIME {reg}: knowledge vs best fair baseline", src.label()), &cells, ns);
                    matrix(&format!("{} regime {reg}, vs best fair baseline", src.label()), &cells, &adj, 2000);
                    matrix(&format!("{} regime {reg}, vs best fair baseline", src.label()), &cells, &adj, 500);
                    doc.insert(format!("{}_regime_{reg}", src.label()), rep);
                    let front: Vec<(String, A)> = vec![("front".into(), (Kind::Guard, if reg == "O" { Signal::Outcome } else { Signal::Reuse }))];
                    let cells_a = family(&specs, &nm, &tr, &front, &cands, held, &ev, false, &mut vec![], &format!("{tag}-abl"));
                    let (rep_a, adj_a) = report(&format!("[{}] ABLATION {reg}: knowledge vs the same guarded(hebr) front without knowledge", src.label()), &cells_a, ns);
                    matrix(&format!("{} regime {reg}, vs same front (must-not-hurt)", src.label()), &cells_a, &adj_a, 2000);
                    doc.insert(format!("{}_ablation_{reg}", src.label()), rep_a);
                }
            }
        }

        // ---------------- shaper-adjacent ----------------
        if part == "shaper" || part == "all" {
            let sh_specs: Vec<Spec> = all_specs
                .iter()
                .filter(|s| matches!(s.name.as_str(), "mini-q" | "hash-q" | "mini-q-mixed50" | "mini-ctl-irrel" | "mini-ctl-unique"))
                .cloned()
                .chain({
                    let mut u = all_specs.iter().find(|s| s.name == "mini-q").unwrap().clone();
                    u.name = "mini-q-useless-proxies".into();
                    u.treatment = false;
                    [u]
                })
                .collect();
            let sh_bases: Vec<(String, A)> = vec![
                ("keyword".into(), (Kind::Keyword, Signal::ShRaw)),
                ("eps-greedy".into(), (Kind::Eps, Signal::ShRaw)),
                ("pooled-ts".into(), (Kind::Pts, Signal::ShRaw)),
                ("hebbian-robust".into(), (Kind::HebR, Signal::ShRaw)),
                ("warm(hebr)".into(), (Kind::Warm, Signal::ShRaw)),
                ("guarded(hebr)".into(), (Kind::Guard, Signal::ShRaw)),
                ("shaped(guarded)".into(), (Kind::Guard, Signal::Shaped)),
                ("shaped(warm)".into(), (Kind::Warm, Signal::Shaped)),
            ];
            let ev = |a: &A, s: &Spec, seeds: &[u64]| eval(a, s, seeds);
            let mut sdoc = serde_json::Map::new();
            let mut stables = vec![];
            // candidates differ by source: use each source's retuned config (from the knowledge part)
            for (sname, srcs) in [("mini", vec!["mini-q", "mini-q-mixed50", "mini-ctl-irrel", "mini-ctl-unique", "mini-q-useless-proxies"]), ("hash", vec!["hash-q"])] {
                let (hid, vote) = retuned.get(sname).cloned().expect("retuned");
                let cands: Vec<(String, A)> = vec![
                    arm("shaped+kn-feat(d6cfg)", &d6_feat, Signal::Shaped),
                    arm("shaped+kn-d+vote(d6cfg)", &d6_vote, Signal::Shaped),
                    arm("shaped+kn-feat(retuned)", &hid, Signal::Shaped),
                    arm("shaped+kn-d+vote(retuned)", &vote, Signal::Shaped),
                ];
                let specs: Vec<Spec> = sh_specs.iter().filter(|s| srcs.contains(&s.name.as_str())).cloned().collect();
                let tag = format!("S-{sname}");
                let cells = family(&specs, &nm, &tr, &sh_bases, &cands, held, &ev, true, &mut stables, &tag);
                let (rep, adj) = report(&format!("[{sname}] SHAPER-ADJACENT: knowledge on shaped reward vs best fair baseline (incl. shaped(guarded))"), &cells, ns);
                matrix(&format!("{sname} shaper-adjacent vs best fair baseline"), &cells, &adj, 2000);
                sdoc.insert(format!("{sname}_main"), rep);
                let front: Vec<(String, A)> = vec![("shaped(guarded)".into(), (Kind::Guard, Signal::Shaped))];
                let cells_a = family(&specs, &nm, &tr, &front, &cands, held, &ev, false, &mut vec![], &format!("{tag}-abl"));
                let (rep_a, adj_a) = report(&format!("[{sname}] SHAPER-ADJACENT ABLATION: vs the same shaped(guarded) front"), &cells_a, ns);
                matrix(&format!("{sname} shaper-adjacent vs same front"), &cells_a, &adj_a, 2000);
                sdoc.insert(format!("{sname}_ablation"), rep_a);
            }
            let d = json!({"lab": "f5_real_embed/shaper", "held": [held[0], held[held.len() - 1]], "tune": [1, 8],
                "note": "d4 shaper itself consumes no embeddings; this composes shaped reward (e2 'rich' proxy model) with knowledge retrieval on real/hash context embeddings",
                "doc": Value::Object(sdoc), "tables": stables, "elapsed_s": t0.elapsed().as_secs_f64()});
            std::fs::write(out_s, serde_json::to_string_pretty(&d).unwrap()).unwrap();
            println!("wrote {out_s}");
        }

        // ---------------- direct paired comparison mini vs hash on the SAME task stream ----------------
        // (the real-source worlds draw the same rng sequence regardless of source, so mini-X and hash-X are the same
        // task stream / same pctx; only the context vector differs) -> paired t-CI on the knowledge arms, each with its own retuned config.
        if part == "knowledge" || part == "all" {
            let mut rows = vec![];
            let mut ps = vec![];
            let mut meta = vec![];
            println!("\n=== DIRECT mini vs hash (paired, same stream, each source's own retuned config): kn arm(mini) - kn arm(hash), H2000 / H500 ===");
            for (reg, sg) in [("O", Signal::Outcome), ("W", Signal::Reuse)] {
                for suffix in ["q", "subj3", "q-mild", "q-mixed50", "ctl-irrel"] {
                    let sm = all_specs.iter().find(|s| s.name == format!("mini-{suffix}")).unwrap();
                    let sh = all_specs.iter().find(|s| s.name == format!("hash-{suffix}")).unwrap();
                    for (kname, which) in [("feat", 0usize), ("vote", 1usize)] {
                        let (cm, ch) = {
                            let (a, b) = (&retuned["mini"], &retuned["hash"]);
                            if which == 0 { (a.0.clone(), b.0.clone()) } else { (a.1.clone(), b.1.clone()) }
                        };
                        let rm = eval(&(Kind::Kn(cm), sg), sm, held);
                        let rh = eval(&(Kind::Kn(ch), sg), sh, held);
                        for i in [2usize, 3] {
                            let (a, b) = (col(&rm, i), col(&rh, i));
                            let d = paired_diff(&a, &b);
                            ps.push(paired_t_pvalue(&a, &b));
                            meta.push((reg, suffix, kname, HORIZONS[i], d.diff.mean, d.diff.lo, d.diff.hi, summarize(&a).mean, summarize(&b).mean));
                        }
                    }
                }
            }
            let adj = holm_adjust(&ps);
            for (m, a) in meta.iter().zip(&adj) {
                let mark = if *a < 0.05 { if m.4 > 0.0 { " WIN(mini)" } else { " WIN(hash)" } } else { "" };
                if m.3 == 2000 {
                    println!("{} {:<12} {:<5} H{:<5} mini-hash {:+.4} [{:+.4},{:+.4}] (mini {:.4} hash {:.4}) Holm {:.4}{}", m.0, m.1, m.2, m.3, m.4, m.5, m.6, m.7, m.8, a, mark);
                }
                rows.push(json!({"regime": m.0, "world_suffix": m.1, "arm": m.2, "H": m.3, "diff_mini_minus_hash": m.4, "ci": [m.5, m.6], "mini_mean": m.7, "hash_mean": m.8, "holm_p": a}));
            }
            doc.insert("source_compare_mini_vs_hash".into(), json!(rows));
        }

        if part == "knowledge" || part == "all" {
            let mut head = serde_json::Map::new();
            head.insert("lab".into(), json!("f5_real_embed/knowledge"));
            head.insert("held".into(), json!([held[0], held[held.len() - 1]]));
            head.insert("tune".into(), json!([1, 8]));
            head.insert("corpus_note".into(), f.meta["note"].clone());
            head.insert("model_sha256".into(), f.meta["model_sha256"].clone());
            head.insert("geometry".into(), json!(geo));
            head.insert(
                "worlds".into(),
                json!(all_specs.iter().map(|s| json!({"name": s.name, "src": s.src.label(), "scenario": s.scenario.label(), "k": s.k, "w": s.w, "mixed": s.mixed, "treatment": s.treatment})).collect::<Vec<_>>()),
            );
            head.insert("selected".into(), Value::Object(selected));
            for (k, v) in doc {
                head.insert(k, v);
            }
            head.insert("tables".into(), json!(tables));
            head.insert("elapsed_s".into(), json!(t0.elapsed().as_secs_f64()));
            std::fs::write(out_k, serde_json::to_string_pretty(&Value::Object(head)).unwrap()).unwrap();
            println!("\nwrote {out_k} in {:.1}s", t0.elapsed().as_secs_f64());
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str, d: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned().unwrap_or(d.into());
    let part = arg("--part", "all");
    let held = parse_held(&args);
    let confirm = held[0] == 61;
    let (dk, dr, ds) = if confirm {
        ("results/f5_knowledge_confirm61-90.json", "results/f5_retrieval_confirm61-90.json", "results/f5_shaper_confirm61-90.json")
    } else {
        ("results/f5_knowledge.json", "results/f5_retrieval.json", "results/f5_shaper.json")
    };
    kpart::main(&part, &held, &arg("--out-k", dk), &arg("--out-r", dr), &arg("--out-s", ds));
}
