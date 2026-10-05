//! G1-route: downstream knowledge-routing success with different similarity SYSTEMS on the HARD g1 corpus.
//!
//! World (as f5/f8): d6-style context world on top of sim::World. Each task text is a g1 query of a recurring CONCEPT; the concept
//! decides (weight w=.8) which of 6 agents is best (hidden p per concept; sibling concepts get independent best agents, so a
//! retriever that conflates siblings routes wrongly). Outcome feedback only (regime O). Policy "kn-direct+vote" (replicates knowledge::Knowledge semantics): retrieve up to k past
//! tasks whose system-similarity >= thr, anchor = similarity-weighted vote of their anchors, `kn:<anchor>` injected into the front, vote =
//! best agent of the anchor with >= min_n rewards and smoothed mean >= .7, gated like d6cfg (6,.02,25), else the guarded(hebr) front decides.
//! NOTE: this is a SELF-CONTAINED neighbour-vote policy (not the crate's KnowledgeRouter) because rerank / hybrid / cascade
//! similarities cannot be expressed as one embedding; the oracle arm (similarity = ground-truth concept equality) bounds it.
//! Worlds: sib16 (2 topics x all 8 sibling details), rand16 (16 random concepts), sib64 (8 topics x 8: sample-efficiency stress), sib16-mixed50 (+50% unrelated one-off texts),
//! ctl-irrel (context carries no information, w=0; control that exposes the cost of the knowledge layer).
//! Protocol: tune (variant, k, thr, min_n) per system on seeds 1-8 (world sib16, mean of H500 and H2000); held-out 31-60; `--corpus B --confirm`
//! re-runs with FROZEN picks on the fresh corpus, seeds 61-90. Paired 95% t-CI vs best fair embedding-free baseline per (world,H)
//! and head-to-head, Holm over the family.
//!
//! Usage: cargo run --release --offline --bin g1_route -- --corpus A [--held 31-60]
//!        cargo run --release --offline --bin g1_route -- --corpus B --confirm [--held 61-90]

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
#[derive(Clone, Copy, Default)]
struct RunOut {
    h: [f64; 4],
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tag = arg(&args, "--corpus", "A");
    let confirm = args.iter().any(|a| a == "--confirm");
    let held = parse_range(&args, "--held", if confirm { (61, 90) } else { (31, 60) });
    let tune_seeds = parse_range(&args, "--tune", (1, 8));
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
    println!("corpus {tag}: pool {nq} queries; {} similarity families; matrices {:.1}s", fams.len(), t0.elapsed().as_secs_f64());

    let ws = worlds();
    let wsib = &ws[0]; // tuning world: sib16
    let thr_grid = [0.15f32, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92];
    let picks_path = "results/g1_route_picks.json";

    // -------- tuning (A) / frozen picks (B) --------
    let mut picks: Vec<(usize, Kind)> = vec![];
    let mut tune_rows = vec![];
    if confirm {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(picks_path).expect("run corpus A first")).unwrap();
        for p in v["picks"].as_array().unwrap() {
            let fam = p["fam"].as_u64().unwrap() as usize;
            assert_eq!(fams[fam].0, p["family"].as_str().unwrap());
            picks.push((fam, Kind::Vote { fam, var: p["var"].as_u64().unwrap() as usize, k: p["k"].as_u64().unwrap() as usize, thr: p["thr"].as_f64().unwrap() as f32, min_n: p["min_n"].as_u64().unwrap() as usize }));
        }
        println!("confirm run: {} frozen picks loaded", picks.len());
    } else {
        for (fi, (fname, vars)) in fams.iter().enumerate() {
            let mut grid: Vec<Kind> = vec![];
            for vi in 0..vars.len() {
                for &k in &[5usize, 8] {
                    for &thr in &thr_grid {
                        for &mn in &[2usize, 3] {
                            if vars[vi].oracle && thr != thr_grid[0] {
                                continue;
                            }
                            grid.push(Kind::Vote { fam: fi, var: vi, k, thr, min_n: mn });
                        }
                    }
                }
            }
            let scored: Vec<(f64, Kind)> = grid.par_iter().map(|g| { let r = eval(g, wsib, &tune_seeds, &corpus, &pool, &fams); ((summarize(&col(&r, 2)).mean + summarize(&col(&r, 3)).mean) / 2.0, g.clone()) }).collect();
            let (m, best) = scored.into_iter().fold((f64::MIN, Kind::Guard), |b, x| if x.0 > b.0 { x } else { b });
            println!("tuned {:<14} mean(H500,H2000)(sib16, seeds {}-{}) = {:.4}  {:?}", fname, tune_seeds[0], tune_seeds[tune_seeds.len() - 1], m, best);
            tune_rows.push(json!({"family": fname, "fam": fi, "tune_h2000": m, "pick": format!("{best:?}")}));
            picks.push((fi, best));
        }
        let pj: Vec<Value> = picks
            .iter()
            .map(|(fi, k)| match k {
                Kind::Vote { var, k, thr, min_n, .. } => json!({"family": fams[*fi].0, "fam": fi, "var": var, "k": k, "thr": thr, "min_n": min_n}),
                _ => json!({}),
            })
            .collect();
        std::fs::write(picks_path, serde_json::to_string_pretty(&json!({"picks": pj, "tune_rows": tune_rows})).unwrap()).unwrap();
    }

    // -------- held-out evaluation --------
    let bases: Vec<(String, Kind)> = vec![
        ("keyword".into(), Kind::Keyword),
        ("eps-greedy".into(), Kind::Eps),
        ("pooled-ts".into(), Kind::Pts),
        ("hebbian".into(), Kind::HebR),
        ("warm".into(), Kind::Warm),
        ("guarded(hebr)".into(), Kind::Guard),
    ];
    let cand_names: Vec<String> = picks.iter().map(|(fi, k)| match k {
        Kind::Vote { var, .. } => format!("{} [{}]", fams[*fi].0, fams[*fi].1[*var].label),
        _ => "?".into(),
    }).collect();
    let mut cells_all: Vec<(String, String, usize, f64, f64, f64, usize, f64, String, bool)> = vec![];
    let mut table = vec![];
    let mut per_run: std::collections::HashMap<(usize, usize), Vec<RunOut>> = Default::default(); // (world, cand) -> runs
    for (wi, sp) in ws.iter().enumerate() {
        let br: Vec<Vec<RunOut>> = bases.iter().map(|(_, k)| eval(k, sp, &held, &corpus, &pool, &fams)).collect();
        let cr: Vec<Vec<RunOut>> = picks.iter().map(|(_, k)| eval(k, sp, &held, &corpus, &pool, &fams)).collect();
        println!("\n=== world {} (w={}, mixed={}) | mean expected success over prefix H ===", sp.name, sp.w, sp.mixed);
        println!("{:<44} {:>8} {:>8} {:>8} {:>8}", "policy", "H100", "H250", "H500", "H2000");
        let mut rows = vec![];
        for (n, r) in bases.iter().map(|b| b.0.clone()).zip(&br).chain(cand_names.iter().cloned().zip(&cr)) {
            let m: Vec<f64> = (0..4).map(|i| summarize(&col(r, i)).mean).collect();
            println!("{:<44} {:>8.4} {:>8.4} {:>8.4} {:>8.4}", n, m[0], m[1], m[2], m[3]);
            rows.push(json!({"policy": n, "h": m}));
        }
        table.push(json!({"world": sp.name, "rows": rows}));
        for (ci, r) in cr.iter().enumerate() {
            per_run.insert((wi, ci), r.clone());
        }
        for i in 0..4 {
            let bi = (0..br.len()).fold(0, |b, j| if summarize(&col(&br[j], i)).mean > summarize(&col(&br[b], i)).mean { j } else { b });
            let b = col(&br[bi], i);
            for (ci, r) in cr.iter().enumerate() {
                let cc = col(r, i);
                let d = paired_diff(&cc, &b);
                cells_all.push((sp.name.into(), cand_names[ci].clone(), HORIZONS[i], d.diff.mean, d.diff.lo, d.diff.hi, d.wins, paired_t_pvalue(&cc, &b), bases[bi].0.clone(), sp.treat));
            }
        }
    }
    let padj = holm_adjust(&cells_all.iter().map(|c| c.7).collect::<Vec<_>>());
    println!("\n=== vs best fair embedding-free baseline (per world,H); Holm over {} tests; H500 and H2000 shown ===", cells_all.len());
    let mut cj = vec![];
    for (c, a) in cells_all.iter().zip(&padj) {
        let mark = if *a < 0.05 { if c.3 > 0.0 { "*" } else { "!" } } else { " " };
        if c.2 == 500 || c.2 == 2000 {
            println!("{:<15} {:<44} H{:<5} {:+.4} [{:+.4},{:+.4}] wins {:>2} holm_p {:.4} {} (vs {})", c.0, c.1, c.2, c.3, c.4, c.5, c.6, a, mark, c.8);
        }
        cj.push(json!({"world": c.0, "cand": c.1, "H": c.2, "diff": c.3, "ci": [c.4, c.5], "wins": c.6, "p": c.7, "holm_p": a, "best_baseline": c.8, "treatment": c.9}));
    }

    // head-to-head family
    let fi = |n: &str| picks.iter().position(|(f, _)| fams[*f].0 == n).unwrap_or_else(|| panic!("family {n}"));
    let h2h: Vec<(&str, &str)> = vec![
        ("bge", "mini"),
        ("mini", "hash"),
        ("mini", "tfidf"),
        ("bge", "tfidf"),
        ("mini-hybrid", "mini"),
        ("mini-hybrid", "bge"),
        ("mini-hybrid", "mini+spell"),
        ("mini-rerank", "mini+spell"),
        ("mini-rerank", "bge"),
        ("bge-hybrid", "bge"),
        ("bge-hybrid", "mini-hybrid"),
        ("bge-rerank", "mini-rerank"),
        ("bge+spell", "mini+spell"),
        ("cascade", "bge-hybrid"),
        ("cascade", "mini-hybrid"),
        ("oracle", "bge-hybrid"),
        ("oracle", "bge"),
    ];
    let mut hc = vec![];
    for (wi, sp) in ws.iter().enumerate() {
        for &(a, b) in &h2h {
            for &h in &[2usize, 3] {
                let (ca, cb) = (col(&per_run[&(wi, fi(a))], h), col(&per_run[&(wi, fi(b))], h));
                let d = paired_diff(&ca, &cb);
                hc.push((sp.name, a, b, HORIZONS[h], d.diff.mean, d.diff.lo, d.diff.hi, d.wins, paired_t_pvalue(&ca, &cb)));
            }
        }
    }
    let hadj = holm_adjust(&hc.iter().map(|c| c.8).collect::<Vec<_>>());
    println!("\n=== head-to-head A-B (same task streams), H500/H2000; Holm over {} tests ===", hc.len());
    let mut hj = vec![];
    for (c, a) in hc.iter().zip(&hadj) {
        let mark = if *a < 0.05 { if c.4 > 0.0 { "*" } else { "!" } } else { " " };
        println!("{:<15} {:<13} - {:<13} H{:<5} {:+.4} [{:+.4},{:+.4}] wins {:>2} holm_p {:.4} {}", c.0, c.1, c.2, c.3, c.4, c.5, c.6, c.7, a, mark);
        hj.push(json!({"world": c.0, "a": c.1, "b": c.2, "H": c.3, "diff": c.4, "ci": [c.5, c.6], "wins": c.7, "p": c.8, "holm_p": a}));
    }

    // cost of the neighbour search on LIVE vectors (what an embedder swap does to the knowledge store)
    let cost = json!({
        "encode_ms_single": {"hash": 0.0, "mini": mini.single_ms, "bge": bge.single_ms},
        "store_mb_for_2000_tasks": {"mini": 2000.0 * 384.0 * 4.0 / 1048576.0, "bge": 2000.0 * 768.0 * 4.0 / 1048576.0},
        "scan_us_per_task_at_2000_stored": {"mini": scan_us(&mini), "bge": scan_us(&bge)},
        "model_mb": {"mini": mini.model_bytes as f64 / 1048576.0, "bge": bge.model_bytes as f64 / 1048576.0},
    });
    println!("\ncost: {cost}");
    let out = json!({"corpus": tag, "confirm": confirm, "held": held, "tune_seeds": tune_seeds, "picks_tune_rows": tune_rows, "table": table, "vs_baseline": cj, "head_to_head": hj, "cost": cost});
    let path = if confirm { "results/g1_route_confirm.json" } else { "results/g1_route.json" };
    std::fs::write(path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote {path} (total {:.1}s)", t0.elapsed().as_secs_f64());
}

fn scan_us(m: &Model) -> f64 {
    let n = 2000usize.min(m.q[0].len() / m.dim);
    let t0 = Instant::now();
    let mut s = 0.0f32;
    let reps = 50;
    for r in 0..reps {
        let q = m.qvec(QV::Raw, r);
        for j in 0..n {
            s += dot(q, m.qvec(QV::Raw, j));
        }
    }
    std::hint::black_box(s);
    t0.elapsed().as_secs_f64() * 1e6 / reps as f64
}
