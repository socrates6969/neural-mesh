//! E2 (REAL-EMBED, open issue 2): do the d6 knowledge-retrieval gains and the d4 reward-shaper
//! gains survive REALISTIC inputs?
//!
//! part "knowledge": d6 worlds with dense, clustered, correlated float embeddings instead of
//! isotropic unit centroids: concept centroid + shared-topic component + topic-group component +
//! anisotropic spectrum; controllable noise; PARAPHRASE regime (M sub-modes per concept, high
//! intra-concept variance); semantic DRIFT (centroids move over time); controls (context
//! recurs but irrelevant; unique context; none). The Knowledge API accepts an embedding per task
//! (KnowledgeRouter::set_context), so embeddings are precomputed per task in the world and fed
//! directly (no hash_embed). An optional in-bin preprocessing (online mean-centering) is tested.
//!
//! part "shaper": NOTE the d4 shaper never used embeddings (it consumes retry/latency/length
//! proxies), so the realism question for it is the realism of the PROXIES: heavy-tailed noise,
//! a shared slowly varying load factor (correlated nuisance), per-agent speed drift, weaker
//! quality signal, and proxies that are pure noise (irrelevant control).
//!
//! Protocol (as d4/d6): tune on seeds 1-8, held-out 31-60 (or --held 61-90 for frozen confirmation),
//! horizons 100/250/500/2000 (prefix mean EXPECTED success), paired 95% t-CI vs the best fair
//! baseline per (world,H), Holm per family.
//!
//! Usage: cargo run --release --bin e2_embed -- [--part knowledge|shaper|all] [--held 61-90]
//!        [--out-k results/e2_knowledge.json] [--out-s results/e2_shaper.json]

use std::time::Instant;

use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

const HORIZONS: [usize; 4] = [100, 250, 500, 2000];

// =============================================================================================
// common

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

/// generic comparison family: candidates vs the best (per world, H, by held-out mean) baseline.
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

fn report(title: &str, cells: &[Cell], n_seeds: usize, verbose: bool) -> (Value, Vec<f64>) {
    let p: Vec<f64> = cells.iter().map(|c| c.p).collect();
    let adj = holm_adjust(&p);
    println!("\n### {title}  (Holm over {} tests)", cells.len());
    if verbose {
        println!("{:<26} {:<30} {:>5} {:>9} {:>20} {:>6} {:>8}  best baseline", "world", "candidate", "H", "diff", "95% CI", "wins", "Holm p");
    }
    let mut out = vec![];
    let (mut nw, mut nl) = (0, 0);
    for (c, a) in cells.iter().zip(&adj) {
        let mark = if *a < 0.05 {
            if c.diff > 0.0 {
                nw += 1;
                " WIN"
            } else {
                nl += 1;
                " LOSS"
            }
        } else {
            ""
        };
        if verbose || !mark.is_empty() && c.h == 2000 {
            println!(
                "{:<26} {:<30} {:>5} {:>+9.4} [{:>+8.4},{:>+8.4}] {:>6} {:>8.4}  {}{}",
                c.world, c.cand, c.h, c.diff, c.lo, c.hi, format!("{}/{}", c.wins, n_seeds), a, c.best_base, mark
            );
        }
        out.push(json!({"world": c.world, "treatment": c.treat, "candidate": c.cand, "H": c.h, "diff": c.diff, "ci": [c.lo, c.hi],
            "wins": c.wins, "n_seeds": n_seeds, "p": c.p, "holm_p": a, "best_baseline": c.best_base, "cand_mean": c.cand_mean, "base_mean": c.base_mean}));
    }
    println!("-> Holm WINS {nw}, Holm LOSSES {nl}");
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
    println!("\n--- {title} | H{h} diff vs reference (* Holm WIN, ! Holm LOSS) ---");
    print!("{:<28}", "world");
    for c in &cands {
        print!(" {:>17}", &c[..c.len().min(17)]);
    }
    println!();
    for w in &worlds {
        print!("{:<28}", w);
        for cn in &cands {
            let i = cells.iter().position(|c| &c.world == w && &c.cand == cn && c.h == h);
            match i {
                Some(i) => {
                    let m = if adj[i] < 0.05 { if cells[i].diff > 0.0 { '*' } else { '!' } } else { ' ' };
                    print!(" {:>+16.4}{}", cells[i].diff, m);
                }
                None => print!(" {:>17}", "-"),
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
// PART 1: knowledge with realistic embeddings

mod kpart {
    use super::*;
    use std::cell::RefCell;
    use std::collections::{HashMap, VecDeque};
    use std::rc::Rc;

    use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
    use neural_mesh::coupling::heb_robust_cfg;
    use neural_mesh::hebbian::HebbianRouter;
    use neural_mesh::knowledge::{AnchorRouter, FeatMode, Knowledge, KnowledgeConfig, KnowledgeRouter, VoteConfig};
    use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
    use neural_mesh::router::{Router, Task};
    use neural_mesh::sim::{Scenario, World, WorldConfig};

    const DIM: usize = 48;
    const NA: usize = 6;

    /// embedding model
    #[derive(Clone, Debug)]
    pub struct EmbCfg {
        /// magnitude of the global shared component (all texts share a "domain" direction)
        pub g: f64,
        /// magnitude of the topic-group component (concepts of the same topic share it)
        pub h: f64,
        pub ntop: usize,
        /// anisotropy exponent: spectrum a_i ~ (i+1)^-p (0 = isotropic)
        pub p: f64,
        /// sub-modes per concept (paraphrase regime when > 1)
        pub m: usize,
        /// norm of the sub-mode offset (intra-concept variance)
        pub spread: f64,
        /// per-text noise norm
        pub noise: f64,
        /// total centroid displacement over the run (semantic drift), norm units
        pub drift: f64,
    }

    #[derive(Clone, Debug)]
    pub struct Spec {
        pub name: String,
        pub scenario: Scenario,
        /// recurring concepts; 0 = unique concept per task
        pub k: usize,
        pub w: f64,
        pub e: EmbCfg,
        pub treatment: bool,
    }

    struct CtxWorld {
        w: World,
        emb: Vec<Vec<f32>>,
        pctx: Vec<[f64; NA]>,
        ctx: Vec<usize>,
        sub: Vec<usize>,
        topic: Vec<usize>,
        wgt: f64,
        draws: Vec<(f64, f64, f64)>,
    }

    fn spectrum(p: f64) -> Vec<f64> {
        let a: Vec<f64> = (0..DIM).map(|i| ((i + 1) as f64).powf(-p)).collect();
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

    impl CtxWorld {
        fn new(seed: u64, sp: &Spec) -> Self {
            let w = World::new(seed, WorldConfig { n_tasks: 2000, scenario: sp.scenario, ..WorldConfig::default() });
            let n = w.len();
            let e = &sp.e;
            let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xE2E2_1234));
            let a = spectrum(e.p);
            let kk = if sp.k == 0 { n } else { sp.k };
            // global shared direction ("mean embedding"): positive spectrum-weighted vector with
            // jitter, so ALL texts have a common component (real encoders: random-pair cosine .2-.5)
            let shared: Vec<f64> = {
                let v: Vec<f64> = a.iter().map(|ai| ai * (1.0 + 0.3 * normal(&mut r))).collect();
                let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
                v.into_iter().map(|x| e.g * x / n).collect()
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
                .map(|_| (0..e.m.max(1)).map(|_| if e.m > 1 { avec(&mut r, &a, e.spread) } else { vec![0.0; DIM] }).collect())
                .collect();
            let dirs: Vec<Vec<f64>> = (0..kk).map(|_| avec(&mut r, &a, 1.0)).collect();
            let mut pctx = Vec::with_capacity(kk);
            for _ in 0..kk {
                let best = r.below(NA);
                let mut row = [0.0; NA];
                for (ai, p) in row.iter_mut().enumerate() {
                    *p = if ai == best { 0.80 + 0.10 * r.f64() } else { 0.15 + 0.35 * r.f64() };
                }
                pctx.push(row);
            }
            let (mut ctx, mut sub, mut topic, mut emb) = (vec![], vec![], vec![], vec![]);
            for t in 0..n {
                let c = if sp.k == 0 { t } else { r.below(kk) };
                let j = r.below(e.m.max(1));
                ctx.push(c);
                sub.push(j);
                topic.push(c % tops.len());
                let mut v = cents[c].clone();
                add(&mut v, &subs[c][j]);
                let f = e.drift * (t as f64 / n as f64);
                for (x, d) in v.iter_mut().zip(&dirs[c]) {
                    *x += f * d;
                }
                let nz = avec(&mut r, &a, e.noise);
                add(&mut v, &nz);
                emb.push(unit(&v));
            }
            let draws = (0..n).map(|_| (r.f64(), r.f64(), r.f64())).collect();
            CtxWorld { w, emb, pctx, ctx, sub, topic, wgt: sp.w, draws }
        }
        fn p(&self, t: usize, a: usize) -> f64 {
            if self.w.is_down(t, a) {
                return 0.0;
            }
            let ty = self.w.task(t).type_idx;
            (1.0 - self.wgt) * self.w.skill(t, ty, a) + self.wgt * self.pctx[self.ctx[t]][a]
        }
        fn outcome(&self, t: usize, a: usize) -> f64 {
            if self.w.task(t).u < self.p(t, a) { 1.0 } else { 0.0 }
        }
        fn oracle(&self) -> f64 {
            let n = self.w.len();
            (0..n).map(|t| (0..NA).map(|a| self.p(t, a)).fold(0.0, f64::max)).sum::<f64>() / n as f64
        }
    }

    fn cosf(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (*x as f64) * (*y as f64)).sum()
    }

    /// geometry diagnostics (seed 1): mean cosine of random pairs / sibling-topic pairs /
    /// same concept different sub-mode / same concept same sub-mode.
    pub fn geometry(sp: &Spec) -> Value {
        let cw = CtxWorld::new(1, sp);
        let n = cw.emb.len();
        let mut r = Rng::new(99);
        let (mut rnd, mut sib, mut sdiff, mut ssame) = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0));
        for _ in 0..4000 {
            let (i, j) = (r.below(n), r.below(n));
            if i == j {
                continue;
            }
            let c = cosf(&cw.emb[i], &cw.emb[j]);
            let acc = if cw.ctx[i] == cw.ctx[j] && sp.k > 0 {
                if cw.sub[i] == cw.sub[j] { &mut ssame } else { &mut sdiff }
            } else if cw.topic[i] == cw.topic[j] && sp.k > 0 {
                &mut sib
            } else {
                &mut rnd
            };
            acc.0 += c;
            acc.1 += 1.0;
        }
        let m = |x: (f64, f64)| if x.1 > 0.0 { x.0 / x.1 } else { f64::NAN };
        json!({"cos_diff_topic": m(rnd), "cos_sibling_concept_same_topic": m(sib), "cos_same_concept_diff_submode": m(sdiff), "cos_same_concept_same_submode": m(ssame)})
    }

    // -------- policies --------
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
    /// optional online mean-centering of the embedding (in-bin preprocessing; causal EMA)
    struct KnPol<R: Router> {
        r: KnowledgeRouter<R>,
        center: bool,
        mu: Vec<f32>,
        n: f64,
    }
    impl<R: Router> Pol for KnPol<R> {
        fn pick(&mut self, t: &Task, e: &[f32], c: &[usize], rng: &mut Rng) -> usize {
            let q: Vec<f32> = if self.center {
                let v: Vec<f32> = if self.n > 0.0 { e.iter().zip(&self.mu).map(|(x, m)| x - m).collect() } else { e.to_vec() };
                let nn = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                let alpha = (1.0 / (self.n + 1.0)).max(0.01) as f32;
                for (m, x) in self.mu.iter_mut().zip(e) {
                    *m += alpha * (x - *m);
                }
                self.n += 1.0;
                v.into_iter().map(|x| x / nn).collect()
            } else {
                e.to_vec()
            };
            self.r.set_context(&q);
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
    }
    impl Signal {
        pub fn label(self) -> &'static str {
            match self {
                Signal::Outcome => "outcome",
                Signal::Raw => "raw",
                Signal::Reuse => "reuse",
            }
        }
    }
    pub type A = (Kind, Signal);

    fn x1_guard() -> GuardConfig {
        GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
    }

    fn build(k: &Kind, w: &World) -> Box<dyn Pol> {
        let kw = w.keyword_map_for();
        match k {
            Kind::Keyword => Box::new(Plain(KeywordRouter::new(kw, NA))),
            Kind::Eps => Box::new(Plain(EpsGreedyRouter::new(NA, 0.99, 0.02))),
            Kind::Pts => Box::new(Plain(PooledThompsonRouter::new(NA, 1.0))),
            Kind::HebR => Box::new(Plain(HebbianRouter::new(NA, heb_robust_cfg()))),
            Kind::Warm => Box::new(Plain(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, 1.0).with_evidence(Evidence::MeanFeature))),
            Kind::Guard => Box::new(Plain(Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, x1_guard()))),
            Kind::Kn(c) => {
                let mut kc = KnowledgeConfig::new(DIM);
                kc.k = c.k;
                kc.sim_min = c.sim_min;
                kc.feat = c.feat;
                kc.vote = c.vote.map(|(min_n, tau, ex, gate)| VoteConfig { min_n, tau, prior: (1.0, 1.0), explore_after: ex, gate });
                let kn = Knowledge::new_shared(kc);
                let anchors = Rc::new(RefCell::new(HashMap::new()));
                if c.feat == FeatMode::Hidden {
                    let inner = AnchorRouter::new(HebbianRouter::new(NA, heb_robust_cfg()), anchors.clone());
                    Box::new(KnPol { r: KnowledgeRouter::new(Guarded::new(inner, kw, NA, x1_guard()), kn, anchors), center: c.center, mu: vec![0.0; DIM], n: 0.0 })
                } else {
                    let front = Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, x1_guard());
                    Box::new(KnPol { r: KnowledgeRouter::new(front, kn, anchors), center: c.center, mu: vec![0.0; DIM], n: 0.0 })
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
        let mut pol = build(k, w);
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
        seeds.par_iter().map(|&s| run(&a.0, sp, a.1, s)).collect()
    }

    fn worlds() -> Vec<Spec> {
        let iso = EmbCfg { g: 0.0, h: 0.0, ntop: 1, p: 0.0, m: 1, spread: 0.0, noise: 0.5, drift: 0.0 };
        let re = |noise: f64| EmbCfg { g: 1.2, h: 0.8, ntop: 4, p: 0.5, m: 1, spread: 0.0, noise, drift: 0.0 };
        let para = |noise: f64| EmbCfg { m: 4, spread: 0.9, ..re(noise) };
        let drf = |noise: f64, d: f64| EmbCfg { drift: d, ..re(noise) };
        let sp = |name: &str, sc: Scenario, k: usize, w: f64, e: EmbCfg, tr: bool| Spec { name: name.into(), scenario: sc, k, w, e, treatment: tr };
        use Scenario::{Fixed, Random};
        vec![
            sp("iso-ref(d6)", Fixed, 12, 0.8, iso.clone(), true),
            sp("aniso-n0.4", Fixed, 12, 0.8, re(0.4), true),
            sp("aniso-n0.8", Fixed, 12, 0.8, re(0.8), true),
            sp("aniso-n1.2", Fixed, 12, 0.8, re(1.2), true),
            sp("aniso-n1.6", Fixed, 12, 0.8, re(1.6), true),
            sp("para-n0.4", Fixed, 12, 0.8, para(0.4), true),
            sp("para-n0.8", Fixed, 12, 0.8, para(0.8), true),
            sp("para-n1.2", Fixed, 12, 0.8, para(1.2), true),
            sp("drift1-n0.8", Fixed, 12, 0.8, drf(0.8, 1.0), true),
            sp("drift2-n0.8", Fixed, 12, 0.8, drf(0.8, 2.0), true),
            sp("para+drift1-n0.8", Fixed, 12, 0.8, EmbCfg { drift: 1.0, ..para(0.8) }, true),
            sp("mild-random-n0.8", Random, 24, 0.5, re(0.8), true),
            sp("ctl-irrel-aniso", Fixed, 12, 0.0, re(0.8), false),
            sp("ctl-irrel-para", Fixed, 12, 0.0, para(0.8), false),
            sp("ctl-irrel-para+drift", Fixed, 12, 0.0, EmbCfg { drift: 1.0, ..para(1.2) }, false),
            sp("ctl-unique-aniso", Fixed, 0, 0.8, re(0.8), false),
            sp("ctl-none-random", Random, 0, 0.0, re(0.8), false),
        ]
    }

    fn cfg_grid() -> Vec<KnCfg> {
        let mut v = vec![];
        for &k in &[5usize, 8] {
            for &sm in &[0.6f32, 0.7, 0.8, 0.9] {
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

    pub fn main(held: &[u64], out_path: &str) {
        let t0 = Instant::now();
        let tune: Vec<u64> = (1..=8).collect();
        let specs = worlds();
        let geo: Vec<Value> = specs.iter().map(|s| json!({"world": s.name, "geometry": geometry(s)})).collect();
        println!("e2 knowledge: held {}-{} ({} seeds), tune seeds 1-8, {} worlds, dim {DIM}", held[0], held[held.len() - 1], held.len(), specs.len());
        println!("\n--- embedding geometry (seed 1; mean cosine of sampled pairs) ---");
        for g in &geo {
            println!("{:<24} {}", g["world"].as_str().unwrap(), g["geometry"]);
        }

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

        // ---- tuning on seeds 1-8 (regime W) over ALL worlds incl. controls and drift -------------
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
        for (o, c) in &scored {
            println!("[tune] {:<44} obj {:+.4}", c.label, o);
        }
        let pick = |pred: &dyn Fn(&KnCfg) -> bool| -> KnCfg { scored.iter().filter(|(_, c)| pred(c)).max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap().1.clone() };
        let hid_any = pick(&|c| c.feat == FeatMode::Hidden);
        let hid_nc = pick(&|c| c.feat == FeatMode::Hidden && !c.center);
        let vote_any = pick(&|c| c.feat == FeatMode::Direct);
        let vote_nc = pick(&|c| c.feat == FeatMode::Direct && !c.center);
        println!("[tune] retuned hid(any) {:?}\n[tune] retuned hid(no-center) {:?}\n[tune] retuned dir+vote(any) {:?}\n[tune] retuned dir+vote(no-center) {:?}", hid_any, hid_nc, vote_any, vote_nc);

        // d6-frozen configs (selected on isotropic d6 worlds)
        let d6_feat = KnCfg { label: String::new(), feat: FeatMode::Hidden, k: 8, sim_min: 0.6, vote: None, center: false };
        let d6_vote = KnCfg { label: String::new(), feat: FeatMode::Direct, k: 8, sim_min: 0.6, vote: Some((2.0, 0.7, None, Some((6.0, 0.02, 25)))), center: false };
        let with_c = |c: &KnCfg| KnCfg { center: true, ..c.clone() };

        let mk = |sg: Signal| -> Vec<(String, A)> {
            vec![
                arm("kn-feat(d6cfg)", &d6_feat, sg),
                arm("kn-d+vote(d6cfg)", &d6_vote, sg),
                arm("kn-feat(retuned)", &hid_any, sg),
                arm("kn-d+vote(retuned)", &vote_any, sg),
            ]
        };
        let mk_diag = |sg: Signal| -> Vec<(String, A)> {
            vec![
                arm("kn-feat(d6cfg+ctr)", &with_c(&d6_feat), sg),
                arm("kn-d+vote(d6cfg+ctr)", &with_c(&d6_vote), sg),
                arm("kn-feat(retuned-noctr)", &hid_nc, sg),
                arm("kn-d+vote(retuned-noctr)", &vote_nc, sg),
            ]
        };

        let ev = |a: &A, s: &Spec, seeds: &[u64]| eval(a, s, seeds);
        let nm = |s: &Spec| s.name.clone();
        let tr = |s: &Spec| s.treatment;
        let mut tables = vec![];
        let mut doc = serde_json::Map::new();
        let ns = held.len();

        for (reg, bases, sg_list) in [("O", &base_o, Signal::Outcome), ("W", &base_w, Signal::Reuse)] {
            let cands = mk(sg_list);
            let cells = family(&specs, &nm, &tr, bases, &cands, held, &ev, true, &mut tables, reg);
            let (rep, adj) = report(&format!("REGIME {reg}: knowledge vs best fair baseline"), &cells, ns, false);
            matrix(&format!("regime {reg}, vs best fair baseline"), &cells, &adj, 2000);
            matrix(&format!("regime {reg}, vs best fair baseline"), &cells, &adj, 500);
            doc.insert(format!("regime_{reg}"), rep);

            // must-not-hurt: vs the SAME front without knowledge
            let front: Vec<(String, A)> = vec![("front".into(), (Kind::Guard, if reg == "O" { Signal::Outcome } else { Signal::Reuse }))];
            let cells_a = family(&specs, &nm, &tr, &front, &cands, held, &ev, false, &mut vec![], &format!("{reg}-abl"));
            let (rep_a, adj_a) = report(&format!("ABLATION {reg}: knowledge vs the same guarded(hebr) front without knowledge"), &cells_a, ns, false);
            matrix(&format!("regime {reg}, vs same front (must-not-hurt)"), &cells_a, &adj_a, 2000);
            doc.insert(format!("ablation_{reg}"), rep_a);

            // diagnostics: centering and no-center retune
            let dcands = mk_diag(sg_list);
            let cells_d = family(&specs, &nm, &tr, &front, &dcands, held, &ev, false, &mut vec![], &format!("{reg}-diag"));
            let (rep_d, adj_d) = report(&format!("DIAG {reg}: centering / non-centered retune vs same front"), &cells_d, ns, false);
            matrix(&format!("regime {reg}, DIAG vs same front"), &cells_d, &adj_d, 2000);
            doc.insert(format!("diag_{reg}"), rep_d);
        }

        let mut head = serde_json::Map::new();
        head.insert("lab".into(), json!("e2_embed/knowledge"));
        head.insert("held".into(), json!([held[0], held[held.len() - 1]]));
        head.insert("tune".into(), json!([1, 8]));
        head.insert("dim".into(), json!(DIM));
        head.insert("geometry".into(), json!(geo));
        head.insert(
            "worlds".into(),
            json!(specs.iter().map(|s| json!({"name": s.name, "scenario": s.scenario.label(), "k": s.k, "w": s.w, "treatment": s.treatment,
                "g": s.e.g, "h": s.e.h, "p": s.e.p, "m": s.e.m, "spread": s.e.spread, "noise": s.e.noise, "drift": s.e.drift})).collect::<Vec<_>>()),
        );
        head.insert(
            "selected".into(),
            json!({"d6_feat": format!("{:?}", d6_feat), "d6_vote": format!("{:?}", d6_vote), "retuned_feat": format!("{:?}", hid_any), "retuned_vote": format!("{:?}", vote_any),
                "retuned_feat_noctr": format!("{:?}", hid_nc), "retuned_vote_noctr": format!("{:?}", vote_nc)}),
        );
        for (k, v) in doc {
            head.insert(k, v);
        }
        head.insert("tables".into(), json!(tables));
        head.insert("elapsed_s".into(), json!(t0.elapsed().as_secs_f64()));
        std::fs::write(out_path, serde_json::to_string_pretty(&Value::Object(head)).unwrap()).unwrap();
        println!("\nwrote {out_path} in {:.1}s", t0.elapsed().as_secs_f64());
    }
}

// =============================================================================================
// PART 2: shaper with realistic proxies

mod spart {
    use super::*;
    use neural_mesh::addons::{Channel, Shaped, ShaperConfig, Transform};
    use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
    use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
    use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
    use neural_mesh::router::{Router, Task};
    use neural_mesh::sim::{Scenario, World, WorldConfig};

    fn heb_robust() -> HebbianConfig {
        HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
    }
    fn x1_guard() -> GuardConfig {
        GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true }
    }

    #[derive(Clone, Debug)]
    pub struct ObsCfg {
        name: &'static str,
        /// quality-signal strength multiplier (0 = proxies carry no quality information)
        sf: f64,
        kappa: f64,
        lat_sigma: f64,
        speed_spread: f64,
        retry: (f64, f64),
        retry_agent_spread: f64,
        len: (f64, f64, f64),
        /// t3 (heavy-tailed) noise on latency/length
        heavy: bool,
        /// amplitude (log scale) of a shared slowly-varying load factor hitting all channels
        load: f64,
        /// per-task random-walk sd of each agent's log speed (drift of the confound)
        spd_drift: f64,
    }

    fn obs_models() -> Vec<ObsCfg> {
        let b = ObsCfg {
            name: "rich", sf: 1.0, kappa: 0.6, lat_sigma: 0.35, speed_spread: 0.3, retry: (0.15, 1.2), retry_agent_spread: 0.0,
            len: (0.6, 0.5, 0.3), heavy: false, load: 0.0, spd_drift: 0.0,
        };
        vec![
            b.clone(),
            ObsCfg { name: "heavy-tail", heavy: true, ..b.clone() },
            ObsCfg { name: "load", load: 0.5, ..b.clone() },
            ObsCfg { name: "spd-drift", spd_drift: 0.01, ..b.clone() },
            ObsCfg { name: "weak-.75", sf: 0.75, ..b.clone() },
            ObsCfg { name: "weak-.5", sf: 0.5, ..b.clone() },
            ObsCfg { name: "weak-.25", sf: 0.25, ..b.clone() },
            ObsCfg { name: "harsh", sf: 0.5, heavy: true, load: 0.5, spd_drift: 0.01, speed_spread: 0.6, retry_agent_spread: 0.6, ..b.clone() },
            ObsCfg { name: "useless", sf: 0.0, ..b.clone() },
            ObsCfg { name: "useless-harsh", sf: 0.0, heavy: true, load: 0.5, spd_drift: 0.01, ..b },
        ]
    }

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

    struct Obs {
        base_lat: [f64; 8],
        base_len: [f64; 8],
        speed: Vec<f64>,
        style: Vec<f64>,
        ragent: Vec<f64>,
        walk: Vec<Vec<f64>>,
        load: Vec<f64>,
        /// (u_tool, z_lat, z_len, u_retry)
        draws: Vec<(f64, f64, f64, f64)>,
    }
    fn znoise(r: &mut Rng, heavy: bool) -> f64 {
        if heavy {
            let n0 = normal(r);
            let c = (normal(r).powi(2) + normal(r).powi(2) + normal(r).powi(2)).sqrt().max(1e-6);
            (n0 / c).clamp(-8.0, 8.0)
        } else {
            normal(r)
        }
    }
    impl Obs {
        fn new(seed: u64, n: usize, na: usize, c: &ObsCfg) -> Self {
            let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(0xE2E2));
            let mut base_lat = [0.0; 8];
            let mut base_len = [0.0; 8];
            for i in 0..8 {
                base_lat[i] = (r.f64() * 2.0 - 0.7).exp();
                base_len[i] = (r.f64() * 2.0 - 0.7).exp() * 100.0;
            }
            let speed = (0..na).map(|_| 1.0 - c.speed_spread / 2.0 + c.speed_spread * r.f64()).collect();
            let style = (0..na).map(|_| (c.len.2 * (r.f64() - 0.5)).exp()).collect();
            let ragent = (0..na).map(|_| 1.0 - c.retry_agent_spread / 2.0 + c.retry_agent_spread * r.f64()).collect();
            let walk = (0..na)
                .map(|_| {
                    let mut x = 0.0;
                    (0..n)
                        .map(|_| {
                            x = 0.999 * x + c.spd_drift * normal(&mut r);
                            x
                        })
                        .collect()
                })
                .collect();
            let mut l = 0.0;
            let load = (0..n)
                .map(|_| {
                    l = 0.97 * l + (1.0f64 - 0.97 * 0.97).sqrt() * normal(&mut r);
                    c.load * l
                })
                .collect();
            let draws = (0..n).map(|_| (r.f64(), znoise(&mut r, c.heavy), znoise(&mut r, c.heavy), r.f64())).collect();
            Obs { base_lat, base_len, speed, style, ragent, walk, load, draws }
        }
    }

    struct DynR(Box<dyn Router>);
    impl Router for DynR {
        fn name(&self) -> String {
            self.0.name()
        }
        fn select(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
            self.0.select(t, c, rng)
        }
        fn update(&mut self, t: &Task, a: usize, r: f64) {
            self.0.update(t, a, r)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Kind {
        Keyword,
        Eps,
        Pts,
        HebR,
        Warm1,
        Warm10,
        Guard,
    }
    impl Kind {
        fn label(self) -> &'static str {
            match self {
                Kind::Keyword => "keyword",
                Kind::Eps => "eps-greedy",
                Kind::Pts => "pooled-ts",
                Kind::HebR => "hebbian-robust",
                Kind::Warm1 => "warm(hebr) k1",
                Kind::Warm10 => "warm(hebr) k10",
                Kind::Guard => "guarded(hebr)",
            }
        }
    }
    fn build(k: Kind, w: &World) -> DynR {
        let n = w.n_agents();
        let kw = w.keyword_map_for();
        DynR(match k {
            Kind::Keyword => Box::new(KeywordRouter::new(kw, n)),
            Kind::Eps => Box::new(EpsGreedyRouter::new(n, 0.99, 0.02)),
            Kind::Pts => Box::new(PooledThompsonRouter::new(n, 1.0)),
            Kind::HebR => Box::new(HebbianRouter::new(n, heb_robust())),
            Kind::Warm1 => Box::new(Warm::new(HebbianRouter::new(n, heb_robust()), kw, n, 1.0).with_evidence(Evidence::MeanFeature)),
            Kind::Warm10 => Box::new(Warm::new(HebbianRouter::new(n, heb_robust()), kw, n, 10.0).with_evidence(Evidence::MeanFeature)),
            Kind::Guard => Box::new(Guarded::new(HebbianRouter::new(n, heb_robust()), kw, n, x1_guard())),
        })
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Sp {
        center: f64,
        gain: f64,
        debias_agent: f64,
        factor: bool,
    }
    fn shaper_cfg(sp: Option<Sp>) -> ShaperConfig {
        let Some(sp) = sp else { return ShaperConfig::off() };
        let ch = |name: &str, tr, dir, df, w| Channel { name: name.into(), transform: tr, dir, debias_feature: df, debias_agent: if df { sp.debias_agent } else { 0.0 }, weight: w };
        ShaperConfig {
            enabled: true,
            channels: vec![ch("retry", Transform::Log1p, -1, false, 1.0), ch("lat", Transform::Ln, -1, true, 1.0), ch("len", Transform::Ln, 0, true, 1.0)],
            center: sp.center,
            gain: sp.gain,
            fail_reward: 0.0,
            factor: sp.factor,
            warmup: 100,
            alpha_min: 0.005,
        }
    }

    #[derive(Clone, Debug)]
    pub struct Spec {
        name: String,
        scenario: Scenario,
        obs: ObsCfg,
    }
    pub type A = (Kind, Option<Sp>);

    fn run(k: Kind, sp: Option<Sp>, spec: &Spec, seed: u64) -> RunOut {
        let w = World::new(seed, WorldConfig { n_tasks: 2000, scenario: spec.scenario, ..WorldConfig::default() });
        let n = w.n_agents();
        let cands: Vec<usize> = (0..n).collect();
        let mut pol = Shaped::new(build(k, &w), shaper_cfg(sp));
        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
        let c = &spec.obs;
        let o = Obs::new(seed, w.len(), n, c);
        let (mut cum, mut fs, mut fnn) = (0.0, 0.0, 0.0f64);
        let mut out = RunOut::default();
        for t in 0..w.len() {
            let st = w.task(t);
            let a = pol.select(&st.task, &cands, &mut rng);
            let pe = w.success_prob(t, &[a]);
            let (u_tool, zl, zn, ur) = o.draws[t];
            let ok = !w.is_down(t, a) && u_tool >= 0.01 + 0.03 * (1.0 - pe.min(1.0));
            let q = c.sf * (1.0 - pe);
            let ld = o.load[t];
            let lat = o.base_lat[st.type_idx] * o.speed[a] * o.walk[a][t].exp() * (c.lat_sigma * zl + ld).exp() * (1.0 + c.kappa * q);
            let retry = poisson_inv(((c.retry.0 + c.retry.1 * q) * o.ragent[a] * (0.7 * ld).exp()).max(1e-6), ur);
            let len = o.base_len[st.type_idx] * o.style[a] * (c.len.1 * zn - c.len.0 * q + 0.3 * ld).exp();
            pol.observe(&st.task, a, ok, &[Some(retry), Some(lat), Some(len)]);
            cum += pe;
            if w.in_failure(t) {
                fs += pe;
                fnn += 1.0;
            }
            for (i, &h) in HORIZONS.iter().enumerate() {
                if t + 1 == h {
                    out.h[i] = cum / h as f64;
                }
            }
        }
        out.fail = fs / fnn.max(1.0);
        out
    }
    fn eval(a: &A, spec: &Spec, seeds: &[u64]) -> Vec<RunOut> {
        seeds.par_iter().map(|&s| run(a.0, a.1, spec, s)).collect()
    }
    fn obj(a: &A, specs: &[&Spec], seeds: &[u64]) -> f64 {
        let mut tot = 0.0;
        for s in specs {
            let r = eval(a, s, seeds);
            tot += r.iter().map(|x| x.h.iter().sum::<f64>() / 4.0).sum::<f64>() / r.len() as f64;
        }
        tot / specs.len() as f64
    }

    pub fn main(held: &[u64], out_path: &str) {
        let t0 = Instant::now();
        let tune: Vec<u64> = (1..=8).collect();
        let mut specs: Vec<Spec> = vec![];
        for sc in [Scenario::Fixed, Scenario::Random] {
            for o in obs_models() {
                specs.push(Spec { name: format!("{}-{}", sc.label(), o.name), scenario: sc, obs: o });
            }
        }
        let tune_specs: Vec<&Spec> = specs.iter().filter(|s| !s.obs.name.starts_with("useless")).collect();
        println!("e2 shaper: held {}-{} ({} seeds), tune 1-8, {} worlds (tune on {} informative)", held[0], held[held.len() - 1], held.len(), specs.len(), tune_specs.len());

        let mut chosen: Vec<(Kind, Sp, f64)> = vec![];
        let mut tune_log = vec![];
        for &k in &[Kind::Guard, Kind::HebR] {
            let mut grid = vec![];
            for &center in &[0.6, 0.75, 0.9] {
                for &gain in &[0.1, 0.2, 0.3] {
                    for &da in &[0.0, 1.0] {
                        for &factor in &[false, true] {
                            grid.push(Sp { center, gain, debias_agent: da, factor });
                        }
                    }
                }
            }
            let scored: Vec<(f64, Sp)> = grid.par_iter().map(|sp| (obj(&(k, Some(*sp)), &tune_specs, &tune), *sp)).collect();
            let best = scored.iter().cloned().fold((f64::NEG_INFINITY, grid[0]), |b, x| if x.0 > b.0 { x } else { b });
            println!("tuned shaped({}): {:?} obj {:.4}", k.label(), best.1, best.0);
            tune_log.push(json!({"learner": k.label(), "params": format!("{:?}", best.1), "obj": best.0}));
            chosen.push((k, best.1, best.0));
        }

        let bases: Vec<(String, A)> =
            [Kind::Keyword, Kind::Eps, Kind::Pts, Kind::HebR, Kind::Warm1, Kind::Warm10, Kind::Guard].iter().map(|&k| (k.label().to_string(), (k, None))).collect();
        let d4 = Sp { center: 0.6, gain: 0.2, debias_agent: 0.0, factor: false };
        let cands: Vec<(String, A)> = vec![
            ("shaped(guarded) d4cfg".into(), (Kind::Guard, Some(d4))),
            ("shaped(guarded) retuned".into(), (Kind::Guard, Some(chosen[0].1))),
            ("shaped(hebr) retuned".into(), (Kind::HebR, Some(chosen[1].1))),
        ];
        let nm = |s: &Spec| s.name.clone();
        let tr = |s: &Spec| !s.obs.name.starts_with("useless");
        let ev = |a: &A, s: &Spec, seeds: &[u64]| eval(a, s, seeds);
        let mut tables = vec![];
        let cells = family(&specs, &nm, &tr, &bases, &cands, held, &ev, true, &mut tables, "S");
        let (rep, adj) = report("SHAPER: shaped(learner) on realistic proxies vs best raw-feedback baseline", &cells, held.len(), false);
        matrix("shaper vs best raw baseline", &cells, &adj, 2000);
        matrix("shaper vs best raw baseline", &cells, &adj, 500);
        // same-front must-not-hurt: shaped(guarded) vs raw guarded (shows whether shaping itself helps)
        let front: Vec<(String, A)> = vec![("guarded(hebr) raw".into(), (Kind::Guard, None))];
        let cells_f = family(&specs, &nm, &tr, &front, &cands[..2], held, &ev, false, &mut vec![], "S-abl");
        let (rep_f, adj_f) = report("SHAPER ABLATION: shaped(guarded) vs raw guarded(hebr) (same front, hook bit only)", &cells_f, held.len(), false);
        matrix("shaper vs raw guarded (same front)", &cells_f, &adj_f, 2000);
        let doc = json!({
            "lab": "e2_embed/shaper", "held": [held[0], held[held.len() - 1]], "tune": [1, 8],
            "note": "d4 shaper never used embeddings; realism here = realistic proxy noise models",
            "worlds": specs.iter().map(|s| json!({"name": s.name, "sf": s.obs.sf, "heavy": s.obs.heavy, "load": s.obs.load, "spd_drift": s.obs.spd_drift,
                "speed_spread": s.obs.speed_spread, "retry_agent_spread": s.obs.retry_agent_spread})).collect::<Vec<_>>(),
            "tuning": tune_log, "main": rep, "ablation_same_front": rep_f, "tables": tables, "elapsed_s": t0.elapsed().as_secs_f64(),
        });
        std::fs::write(out_path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
        println!("\nwrote {out_path} in {:.1}s", t0.elapsed().as_secs_f64());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str, d: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned().unwrap_or(d.into());
    let part = arg("--part", "all");
    let held = parse_held(&args);
    let confirm = held[0] == 61;
    let (dk, ds) = if confirm { ("results/e2_knowledge_confirm61-90.json", "results/e2_shaper_confirm61-90.json") } else { ("results/e2_knowledge.json", "results/e2_shaper.json") };
    if part == "knowledge" || part == "all" {
        kpart::main(&held, &arg("--out-k", dk));
    }
    if part == "shaper" || part == "all" {
        spart::main(&held, &arg("--out-s", ds));
    }
}
