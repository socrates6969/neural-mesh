//! G9-B (W10 RESEARCH-ABILITY): SAMPLE-EFFICIENCY (tasks-to-target-quality) of the retrieval + evidence-reuse layer for different similarity systems
//! (hash / tf-idf / MiniLM / BGE-base / +spell-correction / dense+tf-idf hybrid / rerank / cascade / oracle) on a FRESH g1-style corpus and fresh seeds.
//! World, policies and similarity systems are a VERBATIM copy of g1_route.rs (self-contained replica of the knowledge::Knowledge neighbour vote + guarded(hebr)
//! front; the neighbour vote IS the evidence-reuse path: past graded tasks of similar text vote for an anchor whose best verified agent is reused).
//! Added here: per-task expected-success curves, tasks-to-target metrics, paired tests on them, and a direct re-test of the g1 claim "MiniLM+hybrid beats
//! BGE-alone in routing" on corpora C/D (never seen by W1) with the picks FROZEN from corpus A seeds 1-8 (results/g1_route_picks.json).
//!
//! Targets (per world): L_bf = late-300 mean expected success of the best fair embedding-free baseline (by that late mean), L_or = oracle late-300 mean.
//!   T95 = first task t>=100 whose trailing-100 mean >= 0.95*L_bf (g5 convention);  T50 = trailing-100 mean >= L_bf + 0.5*(L_or - L_bf) (half the headroom).
//!   Not reached within 2000 tasks = censored at 2001. Lower is better; tests are (baseline - candidate) so a positive diff = candidate faster.
//!
//! Usage: cargo run --release --offline --bin g9_sample -- --corpus C --held 61-90

use std::collections::VecDeque;
use std::time::Instant;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
use common::*;
#[path = "g9_common/mod.rs"]
mod c9;
use c9::{holm_by_family, mk_test, test_json, verdict, Test};

const NA: usize = 6;
const HORIZONS: [usize; 4] = [100, 250, 500, 2000];

// ------------------------------------------------------------------------------------------------
// similarity systems over the query pool (precomputed pair matrices, combined lazily)

struct Pool {
    nq: usize,
    mats: Vec<Vec<f32>>,
    names: Vec<String>,
    concept: Vec<i32>,
}
impl Pool {
    fn mid(&self, n: &str) -> usize {
        self.names.iter().position(|x| x == n).unwrap_or_else(|| panic!("matrix {n}"))
    }
}

#[derive(Clone, Debug)]
struct Sim {
    /// candidate score = sum w * mats[idx]
    cand: Vec<(usize, f32)>,
    /// optional rerank: (final score terms, top-M)
    fin: Option<(Vec<(usize, f32)>, usize)>,
    oracle: bool,
    label: String,
}
impl Sim {
    fn s(&self, p: &Pool, terms: &[(usize, f32)], i: usize, j: usize) -> f32 {
        terms.iter().map(|&(m, w)| w * p.mats[m][i * p.nq + j]).sum()
    }
}

fn dense_pair(m: &Model, qv: QV, nq: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nq * nq];
    out.par_chunks_mut(nq).enumerate().for_each(|(i, row)| {
        let a = m.qvec(qv, i);
        for (j, r) in row.iter_mut().enumerate() {
            *r = dot(a, m.qvec(qv, j));
        }
    });
    out
}
fn tfidf_pair(lex: &Lex, c: &Corpus, qv: QV) -> Vec<f32> {
    let nq = c.queries.len();
    let vs: Vec<Vec<(u32, f32)>> = c.queries.iter().map(|q| lex.tfidf(match qv { QV::Raw => &q.text, _ => &q.spell })).collect();
    let mut out = vec![0.0f32; nq * nq];
    out.par_chunks_mut(nq).enumerate().for_each(|(i, row)| {
        for (j, r) in row.iter_mut().enumerate() {
            *r = sparse_dot(&vs[i], &vs[j]);
        }
    });
    out
}

/// system families -> variants
fn families(p: &Pool) -> Vec<(String, Vec<Sim>)> {
    let t = |n: &str| p.mid(n);
    let mk = |label: &str, cand: Vec<(usize, f32)>, fin: Option<(Vec<(usize, f32)>, usize)>| Sim { cand, fin, oracle: false, label: label.into() };
    let mut f: Vec<(String, Vec<Sim>)> = vec![];
    f.push(("hash".into(), vec![mk("hash", vec![(t("hash"), 1.0)], None)]));
    f.push(("tfidf".into(), vec![mk("tfidf", vec![(t("tfidf"), 1.0)], None)]));
    f.push(("mini".into(), vec![mk("mini", vec![(t("mini"), 1.0)], None)]));
    f.push(("bge".into(), vec![mk("bge", vec![(t("bge"), 1.0)], None)]));
    f.push(("mini+spell".into(), vec![mk("mini+spell", vec![(t("mini_s"), 1.0)], None)]));
    f.push(("bge+spell".into(), vec![mk("bge+spell", vec![(t("bge_s"), 1.0)], None)]));
    for m in ["mini", "bge"] {
        let (d, l) = (t(&format!("{m}_s")), t("tfidf_s"));
        f.push((
            format!("{m}-hybrid"),
            [0.4f32, 0.6, 0.8].iter().map(|&a| mk(&format!("{m}-hybrid a{a}"), vec![(d, a), (l, 1.0 - a)], None)).collect(),
        ));
        f.push((
            format!("{m}-rerank"),
            [0.3f32, 0.5].iter().map(|&g| mk(&format!("{m}-rerank g{g}"), vec![(d, 1.0)], Some((vec![(d, 1.0 - g), (l, g)], 20)))).collect(),
        ));
    }
    // cascade: candidates from mini-hybrid(a.6) on spell text, final = .5 that + .5 bge (spell)
    let (ms, ts, bs) = (t("mini_s"), t("tfidf_s"), t("bge_s"));
    f.push((
        "cascade".into(),
        vec![mk("mini-hybrid>bge-rerank", vec![(ms, 0.6), (ts, 0.4)], Some((vec![(ms, 0.3), (ts, 0.2), (bs, 0.5)], 20)))],
    ));
    f.push(("oracle".into(), vec![Sim { cand: vec![], fin: None, oracle: true, label: "oracle(concept)".into() }]));
    f
}

// ------------------------------------------------------------------------------------------------
// worlds

#[derive(Clone, Debug)]
struct Spec {
    name: &'static str,
    sib: bool,
    /// number of topics (x8 sibling details = 8*nt concepts)
    nt: usize,
    w: f64,
    mixed: f64,
    treat: bool,
}
fn worlds() -> Vec<Spec> {
    vec![
        Spec { name: "sib16", sib: true, nt: 2, w: 0.8, mixed: 0.0, treat: true },
        Spec { name: "rand16", sib: false, nt: 2, w: 0.8, mixed: 0.0, treat: true },
        Spec { name: "sib64", sib: true, nt: 8, w: 0.8, mixed: 0.0, treat: true },
        Spec { name: "sib16-mixed50", sib: true, nt: 2, w: 0.8, mixed: 0.5, treat: true },
        Spec { name: "ctl-irrel", sib: true, nt: 2, w: 0.0, mixed: 0.0, treat: false },
    ]
}

struct Stream {
    w: World,
    q: Vec<usize>,
    ctx: Vec<i32>,
    wt: Vec<f64>,
    pctx: Vec<[f64; NA]>,
    /// concept -> row in pctx
    row: Vec<usize>,
}
fn stream(c: &Corpus, seed: u64, sp: &Spec) -> Stream {
    let w = World::new(seed, WorldConfig { n_tasks: 2000, scenario: Scenario::Fixed, ..WorldConfig::default() });
    let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xC0DE_6001));
    let n = w.len();
    let ntopics = c.n_topics();
    let mut active: Vec<usize> = vec![];
    if sp.sib {
        let mut topics: Vec<usize> = (0..ntopics).collect();
        for i in (1..topics.len()).rev() {
            let j = r.below(i + 1);
            topics.swap(i, j);
        }
        for &t in topics.iter().take(sp.nt) {
            active.extend((0..c.nconc).filter(|&k| c.topic_of_concept[k] as usize == t));
        }
    } else {
        let mut all: Vec<usize> = (0..c.nconc).collect();
        for i in (1..all.len()).rev() {
            let j = r.below(i + 1);
            all.swap(i, j);
        }
        active = all.into_iter().take(sp.nt * 8).collect();
    }
    let mut row = vec![usize::MAX; c.nconc];
    let mut pctx = vec![];
    for (k, &cid) in active.iter().enumerate() {
        row[cid] = k;
        let best = r.below(NA);
        let mut pr = [0.0; NA];
        for (a, p) in pr.iter_mut().enumerate() {
            *p = if a == best { 0.80 + 0.10 * r.f64() } else { 0.15 + 0.35 * r.f64() };
        }
        pctx.push(pr);
    }
    let by_c: Vec<Vec<usize>> = (0..c.nconc).map(|k| (0..c.queries.len()).filter(|&i| c.queries[i].concept == k as i32).collect()).collect();
    let ood: Vec<usize> = (0..c.queries.len()).filter(|&i| c.queries[i].concept < 0).collect();
    let (mut q, mut ctx, mut wt) = (vec![], vec![], vec![]);
    for _ in 0..n {
        if sp.mixed > 0.0 && r.f64() < sp.mixed {
            q.push(ood[r.below(ood.len())]);
            ctx.push(-1);
            wt.push(0.0);
        } else {
            let cid = active[r.below(active.len())];
            q.push(by_c[cid][r.below(by_c[cid].len())]);
            ctx.push(cid as i32);
            wt.push(sp.w);
        }
    }
    Stream { w, q, ctx, wt, pctx, row }
}
impl Corpus {
    fn n_topics(&self) -> usize {
        (*self.topic_of_concept.iter().max().unwrap() + 1) as usize
    }
}
impl Stream {
    fn p(&self, t: usize, a: usize) -> f64 {
        if self.w.is_down(t, a) {
            return 0.0;
        }
        let ty = self.w.task(t).type_idx;
        let base = self.w.skill(t, ty, a);
        let wg = self.wt[t];
        if wg == 0.0 { base } else { (1.0 - wg) * base + wg * self.pctx[self.row[self.ctx[t] as usize]][a] }
    }
}

// ------------------------------------------------------------------------------------------------
// policies

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm,
    Guard,
    Vote { fam: usize, var: usize, k: usize, thr: f32, min_n: usize },
}
#[derive(Clone, Default)]
struct RunOut {
    h: [f64; 4],
    /// expected success of every task (pe)
    curve: Vec<f32>,
}

fn front(kind: &Kind, w: &World) -> Box<dyn Router> {
    let kw = w.keyword_map_for();
    let g = GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true };
    match kind {
        Kind::Keyword => Box::new(KeywordRouter::new(kw, NA)),
        Kind::Eps => Box::new(EpsGreedyRouter::new(NA, 0.99, 0.02)),
        Kind::Pts => Box::new(PooledThompsonRouter::new(NA, 1.0)),
        Kind::HebR => Box::new(HebbianRouter::new(NA, heb_robust_cfg())),
        Kind::Warm => Box::new(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, 1.0).with_evidence(Evidence::MeanFeature)),
        _ => Box::new(Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw, NA, g)),
    }
}

/// top-k past tasks (record ids = task indices) whose final system similarity is >= thr, best first: (score, record id)
fn neighbors(sim: &Sim, p: &Pool, qi: usize, past: &[usize], k: usize, thr: f32) -> Vec<(f32, usize)> {
    if sim.oracle {
        let c = p.concept[qi];
        if c < 0 {
            return vec![];
        }
        return past.iter().enumerate().rev().filter(|&(_, &m)| p.concept[m] == c).map(|(i, _)| (1.0, i)).take(k).collect();
    }
    match &sim.fin {
        None => {
            let mut v: Vec<(f32, usize)> = past.iter().enumerate().map(|(i, &m)| (sim.s(p, &sim.cand, qi, m), i)).filter(|x| x.0 >= thr).collect();
            if v.len() > k {
                v.select_nth_unstable_by(k, |a, b| b.0.partial_cmp(&a.0).unwrap());
                v.truncate(k);
            }
            v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            v
        }
        Some((fin, m)) => {
            let mut v: Vec<(f32, usize)> = past.iter().enumerate().map(|(i, &mm)| (sim.s(p, &sim.cand, qi, mm), i)).collect();
            if v.len() > *m {
                v.select_nth_unstable_by(*m, |a, b| b.0.partial_cmp(&a.0).unwrap());
                v.truncate(*m);
            }
            let mut f: Vec<(f32, usize)> = v.into_iter().map(|(_, i)| (sim.s(p, fin, qi, past[i]), i)).filter(|x| x.0 >= thr).collect();
            f.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            f.truncate(k);
            f
        }
    }
}

/// kn-direct+vote with a pluggable similarity (replicates knowledge::Knowledge/KnowledgeRouter semantics: anchor = similarity-weighted
/// vote of the neighbours' anchors, the feature `kn:<anchor>` is injected into the front, vote = best agent of the anchor with
/// >= min_n rewards and smoothed mean >= .7, vote gate (6, .02, 25))
fn run(kind: &Kind, sp: &Spec, seed: u64, c: &Corpus, p: &Pool, fams: &[(String, Vec<Sim>)]) -> RunOut {
    let st = stream(c, seed, sp);
    let w = &st.w;
    let cands: Vec<usize> = (0..NA).collect();
    let mut fr = front(kind, w);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut out = RunOut::default();
    let mut cum = 0.0;
    let mut past: Vec<usize> = vec![];
    let mut anchor_of: Vec<u64> = vec![];
    let mut astats: std::collections::HashMap<u64, [(f64, f64); NA]> = Default::default();
    let mut gate = [0.0f64; 4];
    let mut probe = 0usize;
    for t in 0..w.len() {
        let task0 = &w.task(t).task;
        let mut task = task0.clone();
        let mut by_vote = false;
        let mut anchor = t as u64;
        let a = match kind {
            Kind::Vote { fam, var, k, thr, min_n } => {
                let sim = &fams[*fam].1[*var];
                let nb = neighbors(sim, p, st.q[t], &past, *k, *thr);
                let mut votes: Vec<(u64, f32)> = vec![];
                for &(sc, i) in &nb {
                    match votes.iter_mut().find(|v| v.0 == anchor_of[i]) {
                        Some(v) => v.1 += sc,
                        None => votes.push((anchor_of[i], sc)),
                    }
                }
                let found = votes.iter().fold(None, |b: Option<(u64, f32)>, &v| match b { Some(x) if x.1 >= v.1 => b, _ => Some(v) });
                if let Some((an, _)) = found {
                    anchor = an;
                    task.features.push(format!("kn:{an}"));
                }
                let mut best: Option<(usize, f64)> = None;
                if let Some(row) = found.and_then(|(an, _)| astats.get(&an)) {
                    for a in 0..NA {
                        let (s, n) = row[a];
                        if n < *min_n as f64 {
                            continue;
                        }
                        let m = (1.0 + s) / (2.0 + n);
                        if m >= 0.7 && best.is_none_or(|b| m > b.1) {
                            best = Some((a, m));
                        }
                    }
                }
                let open = if best.is_none() || gate[1] < 6.0 || gate[3] < 6.0 || gate[0] / gate[1] >= gate[2] / gate[3] - 0.02 {
                    true
                } else {
                    probe += 1;
                    probe % 25 == 0
                };
                match best {
                    Some((a, _)) if open => {
                        by_vote = true;
                        a
                    }
                    _ => fr.select(&task, &cands, &mut rng),
                }
            }
            _ => fr.select(&task, &cands, &mut rng),
        };
        let pe = st.p(t, a);
        let s = if w.task(t).u < pe { 1.0 } else { 0.0 };
        fr.update(&task, a, s);
        if matches!(kind, Kind::Vote { .. }) {
            past.push(st.q[t]);
            anchor_of.push(anchor);
            let row = astats.entry(anchor).or_insert([(0.0, 0.0); NA]);
            row[a].0 += s;
            row[a].1 += 1.0;
            if by_vote {
                gate[0] += s;
                gate[1] += 1.0;
            } else {
                gate[2] += s;
                gate[3] += 1.0;
            }
        }
        cum += pe;
        out.curve.push(pe as f32);
        for (i, &hz) in HORIZONS.iter().enumerate() {
            if t + 1 == hz {
                out.h[i] = cum / hz as f64;
            }
        }
    }
    out
}

fn col(r: &[RunOut], i: usize) -> Vec<f64> {
    r.iter().map(|x| x.h[i]).collect()
}

fn eval(kind: &Kind, sp: &Spec, seeds: &[u64], c: &Corpus, p: &Pool, fams: &[(String, Vec<Sim>)]) -> Vec<RunOut> {
    seeds.par_iter().map(|&s| run(kind, sp, s, c, p, fams)).collect()
}

const CAP: f64 = 2001.0;

/// first t (1-based, >= 100) with trailing-100 mean >= target; CAP if never
fn t_to_target(curve: &[f32], target: f64) -> f64 {
    let mut s: f64 = curve.iter().take(100).map(|x| *x as f64).sum();
    if s / 100.0 >= target {
        return 100.0;
    }
    for t in 100..curve.len() {
        s += curve[t] as f64 - curve[t - 100] as f64;
        if s / 100.0 >= target {
            return (t + 1) as f64;
        }
    }
    CAP
}
fn late300(curve: &[f32]) -> f64 {
    let n = curve.len();
    curve[n - 300..].iter().map(|x| *x as f64).sum::<f64>() / 300.0
}
fn first500(curve: &[f32]) -> f64 {
    curve[..500].iter().map(|x| *x as f64).sum::<f64>() / 500.0
}

struct Cell {
    world: String,
    sys: String,
    fam: String,
    t95: Vec<f64>,
    t50: Vec<f64>,
    f500: Vec<f64>,
    h2000: Vec<f64>,
    reached50: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tag = arg(&args, "--corpus", "C");
    let held = parse_range(&args, "--held", (61, 90));
    let out_path = arg(&args, "--out", &format!("results/g9_sample_{}_{}-{}.json", tag, held[0], held[held.len() - 1]));
    let t0 = Instant::now();
    let corpus = load_corpus(&tag);
    let lex = Lex::build(&corpus.docs);
    let nq = corpus.queries.len();
    let mini = load_model(&corpus, "mini");
    let bge = load_model(&corpus, "bge");
    let hash = load_model(&corpus, "hash");
    let mut names: Vec<String> = vec![];
    let mut mats: Vec<Vec<f32>> = vec![];
    for (n, v) in [
        ("hash", dense_pair(&hash, QV::Raw, nq)),
        ("mini", dense_pair(&mini, QV::Raw, nq)),
        ("bge", dense_pair(&bge, QV::Raw, nq)),
        ("mini_s", dense_pair(&mini, QV::Spell, nq)),
        ("bge_s", dense_pair(&bge, QV::Spell, nq)),
        ("tfidf", tfidf_pair(&lex, &corpus, QV::Raw)),
        ("tfidf_s", tfidf_pair(&lex, &corpus, QV::Spell)),
    ] {
        names.push(n.into());
        mats.push(v);
    }
    let pool = Pool { nq, mats, names, concept: corpus.queries.iter().map(|q| q.concept).collect() };
    let fams = families(&pool);
    let v: Value = serde_json::from_str(&std::fs::read_to_string("results/g1_route_picks.json").expect("g1 picks")).unwrap();
    let mut picks: Vec<(usize, Kind)> = vec![];
    for p in v["picks"].as_array().unwrap() {
        let fam = p["fam"].as_u64().unwrap() as usize;
        assert_eq!(fams[fam].0, p["family"].as_str().unwrap());
        picks.push((fam, Kind::Vote { fam, var: p["var"].as_u64().unwrap() as usize, k: p["k"].as_u64().unwrap() as usize, thr: p["thr"].as_f64().unwrap() as f32, min_n: p["min_n"].as_u64().unwrap() as usize }));
    }
    let fam_names: Vec<String> = picks.iter().map(|(fi, _)| fams[*fi].0.clone()).collect();
    let bases: Vec<(String, Kind)> = vec![
        ("keyword".into(), Kind::Keyword),
        ("eps-greedy".into(), Kind::Eps),
        ("pooled-ts".into(), Kind::Pts),
        ("hebbian".into(), Kind::HebR),
        ("warm".into(), Kind::Warm),
        ("guarded(hebr)".into(), Kind::Guard),
    ];
    let ws = worlds();
    println!("corpus {tag}: {} candidate systems, held {}-{}; setup {:.1}s", picks.len(), held[0], held[held.len() - 1], t0.elapsed().as_secs_f64());

    let mut cells: Vec<Cell> = vec![];
    let mut tests: Vec<Test> = vec![];
    for sp in ws.iter() {
        let br: Vec<Vec<RunOut>> = bases.iter().map(|(_, k)| eval(k, sp, &held, &corpus, &pool, &fams)).collect();
        let cr: Vec<Vec<RunOut>> = picks.iter().map(|(_, k)| eval(k, sp, &held, &corpus, &pool, &fams)).collect();
        let late = |r: &Vec<RunOut>| r.iter().map(|x| late300(&x.curve)).sum::<f64>() / r.len() as f64;
        let bi = (0..br.len()).fold(0, |b, j| if late(&br[j]) > late(&br[b]) { j } else { b });
        let oi = fam_names.iter().position(|f| f == "oracle").unwrap();
        let l_bf = late(&br[bi]);
        let l_or = late(&cr[oi]);
        let (tg95, tg50) = (0.95 * l_bf, l_bf + 0.5 * (l_or - l_bf));
        println!("\n=== {} | best fair baseline by late-300 = {} ({:.4}); oracle late-300 {:.4}; targets T95 {:.4}, T50 {:.4}", sp.name, bases[bi].0, l_bf, l_or, tg95, tg50);
        println!("{:<44} {:>8} {:>8} {:>7} {:>8} {:>8}", "system", "T95", "T50", "reach50", "first500", "H2000");
        let mk = |name: String, fam: String, r: &Vec<RunOut>| {
            let t95: Vec<f64> = r.iter().map(|x| t_to_target(&x.curve, tg95)).collect();
            let t50: Vec<f64> = r.iter().map(|x| t_to_target(&x.curve, tg50)).collect();
            let f5: Vec<f64> = r.iter().map(|x| first500(&x.curve)).collect();
            let h2: Vec<f64> = r.iter().map(|x| x.h[3]).collect();
            let reached = t50.iter().filter(|t| **t < CAP).count() as f64 / t50.len() as f64;
            let m = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
            println!("{:<44} {:>8.0} {:>8.0} {:>7.2} {:>8.4} {:>8.4}", name, m(&t95), m(&t50), reached, m(&f5), m(&h2));
            Cell { world: sp.name.into(), sys: name, fam, t95, t50, f500: f5, h2000: h2, reached50: reached }
        };
        let base_cells: Vec<Cell> = bases.iter().zip(&br).map(|((n, _), r)| mk(format!("[base] {n}"), "base".into(), r)).collect();
        let cand_cells: Vec<Cell> = fam_names.iter().zip(&cr).map(|(f, r)| mk(f.clone(), f.clone(), r)).collect();
        let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        let best_t = |sel: &dyn Fn(&Cell) -> &Vec<f64>| (0..base_cells.len()).fold(0, |b, j| if mean(sel(&base_cells[j])) < mean(sel(&base_cells[b])) { j } else { b });
        let (b95, b50) = (best_t(&|c| &c.t95), best_t(&|c| &c.t50));
        for c in cand_cells.iter() {
            if c.fam == "oracle" {
                continue;
            }
            // lower T is better: test (baseline - candidate) so positive = candidate faster
            tests.push(mk_test("S1-T95-vs-bestfair", sp.name, "T95", &c.sys, &base_cells[b95].sys, &base_cells[b95].t95, &c.t95));
            tests.push(mk_test("S1-T50-vs-bestfair", sp.name, "T50", &c.sys, &base_cells[b50].sys, &base_cells[b50].t50, &c.t50));
            tests.push(mk_test("S3-first500-vs-bestfair", sp.name, "first500", &c.sys, &base_cells[bi].sys, &c.f500, &base_cells[bi].f500));
        }
        let idx = |f: &str| cand_cells.iter().position(|c| c.fam == f).unwrap_or_else(|| panic!("family {f}"));
        let h2h: Vec<(&str, &str)> = vec![
            ("mini-hybrid", "bge"),
            ("mini-hybrid", "mini"),
            ("mini+spell", "mini"),
            ("mini+spell", "bge"),
            ("mini", "bge"),
            ("mini", "tfidf"),
            ("mini", "hash"),
            ("bge-hybrid", "bge"),
            ("bge+spell", "bge"),
            ("bge-hybrid", "mini-hybrid"),
            ("cascade", "mini-hybrid"),
            ("mini-rerank", "bge"),
            ("tfidf", "bge"),
        ];
        for (a, b) in h2h {
            let (ca, cb) = (&cand_cells[idx(a)], &cand_cells[idx(b)]);
            tests.push(mk_test("S2-h2h-T95", sp.name, "T95", a, b, &cb.t95, &ca.t95)); // positive = a faster
            tests.push(mk_test("S2-h2h-T50", sp.name, "T50", a, b, &cb.t50, &ca.t50));
            tests.push(mk_test("S4-h2h-H2000", sp.name, "H2000", a, b, &ca.h2000, &cb.h2000));
            tests.push(mk_test("S4-h2h-first500", sp.name, "first500", a, b, &ca.f500, &cb.f500));
        }
        cells.extend(base_cells);
        cells.extend(cand_cells);
    }
    holm_by_family(&mut tests);
    println!("\n=== Holm tests (positive diff = treatment FASTER (T) / HIGHER (H, first500)); claim pairs always shown, others only when significant ===");
    for t in &tests {
        let v = verdict(t.adj, t.d.diff.mean);
        let claim = (t.treat == "mini-hybrid" && t.vs == "bge") || (t.treat == "mini+spell" && t.vs == "bge") || (t.treat == "mini" && t.vs == "bge");
        if ((t.family.starts_with("S2") || t.family.starts_with("S4")) && claim) || (v != "ns" && (t.family.starts_with("S1") || t.family.starts_with("S3"))) {
            println!("{:<26} {:<14} {:<14} vs {:<22} {:+9.4} [{:+9.4},{:+9.4}] holm {:.4} {}", t.family, t.world, t.treat, t.vs, t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, t.adj, v);
        }
    }
    for fam in ["S1-T95-vs-bestfair", "S1-T50-vs-bestfair", "S3-first500-vs-bestfair", "S2-h2h-T95", "S2-h2h-T50", "S4-h2h-H2000", "S4-h2h-first500"] {
        let n = tests.iter().filter(|t| t.family == fam).count();
        let w = tests.iter().filter(|t| t.family == fam && verdict(t.adj, t.d.diff.mean) == "WIN").count();
        let l = tests.iter().filter(|t| t.family == fam && verdict(t.adj, t.d.diff.mean) == "LOSS").count();
        println!("{fam}: WIN {w} LOSS {l} of {n}");
    }
    let doc = json!({"corpus": tag, "held": [held[0], held[held.len() - 1]],
        "cells": cells.iter().map(|c| json!({"world": c.world, "system": c.sys, "family": c.fam, "t95_mean": c.t95.iter().sum::<f64>() / c.t95.len() as f64, "t50_mean": c.t50.iter().sum::<f64>() / c.t50.len() as f64, "reached50": c.reached50, "first500": c.f500.iter().sum::<f64>() / c.f500.len() as f64, "h2000": c.h2000.iter().sum::<f64>() / c.h2000.len() as f64})).collect::<Vec<_>>(),
        "tests": tests.iter().map(test_json).collect::<Vec<_>>(), "runtime_s": t0.elapsed().as_secs_f64()});
    std::fs::write(&out_path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out_path} in {:.1}s", t0.elapsed().as_secs_f64());
}
