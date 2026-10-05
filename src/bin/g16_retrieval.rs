//! G16-retrieval (W17, retrieval track R): document-level retrieval metrics (P/R/F1/MRR/hit@1, bytes, latency) for the levers that the
//! downstream loop (g16_route) also uses, on the g1 hard-corpus fixtures (no new models):
//!   E1 adaptive k            (fixed k vs score-drop / ratio / byte-budget cut-offs)
//!   E2 dedup + freshness     (store with re-ingested duplicates and superseded revisions; near-duplicate suppression, keep-newest)
//!   E3 partitioned indices   (3a: two PROJECT corpora with the same concepts but different answers, pooled vs tag-partitioned vs soft boost,
//!                             with tag noise; 3b: topic-domain partitions with centroid routing)
//!   E4 retrieval-level gate  (abstain when top-1 is unlikely to be right; includes the 200 off-topic queries)
//! Protocol (as g1_retrieval): per seed = random subset of 300 queries of the split; knobs tuned on tune concepts (concept%5==0, seeds 1-8),
//! evaluated on the other concepts (seeds 31-60, corpus A / project pair A+B); `--confirm`: corpus B / project pair C+D, FROZEN knobs, seeds 61-90.
//! Paired 95% t-CI over seeds, Holm per family. CIs are over resampled query subsets of a finite SYNTHETIC pool, not over independent corpora.
//! E2's duplicates/revisions are LABELS I attach to existing docs (no new text is embedded); E3a's "other project" is the sibling corpus
//! (same 240 concepts, different phrasing). Both are by-construction scenarios and are described as such in the board entry.
//!
//! Usage: cargo run --release --offline --bin g16_retrieval -- [--held 31-60]
//!        cargo run --release --offline --bin g16_retrieval -- --confirm [--held 61-90]

use std::collections::HashMap;
use std::time::Instant;

use neural_mesh::router::Rng;
use neural_mesh::stats::summarize;
use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
#[path = "g16_common/mod.rs"]
mod g16;
use common::*;
use g16::{cmp, holm, mark, sigmoid};

// ---------------------------------------------------------------------------------------------
// generic helpers

#[derive(Clone, Copy, Default, Debug)]
struct M {
    n_ret: f64,
    p: f64,
    r: f64,
    f1: f64,
    hit1: f64,
    mrr: f64,
    chars: f64,
    extra: f64,
}
const MN: [&str; 8] = ["n_ret", "P", "R", "F1", "hit1", "MRR", "chars", "extra"];
impl M {
    fn a(&self) -> [f64; 8] {
        [self.n_ret, self.p, self.r, self.f1, self.hit1, self.mrr, self.chars, self.extra]
    }
}
fn prf(hits: f64, n_ret: f64, nrel: f64) -> (f64, f64, f64) {
    let p = if n_ret > 0.0 { hits / n_ret } else { 0.0 };
    let r = hits / nrel;
    let f = if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 };
    (p, r, f)
}
fn sample(pool: &[usize], seed: u64, n: usize) -> Vec<usize> {
    let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(101));
    let mut p = pool.to_vec();
    for i in (1..p.len()).rev() {
        let j = r.below(i + 1);
        p.swap(i, j);
    }
    p.truncate(n);
    p
}
/// per-seed means of per-query metrics (indexed by query id via `pos`)
fn seed_means(per_q: &[M], pos: &HashMap<usize, usize>, pool: &[usize], seeds: &[u64]) -> Vec<[f64; 8]> {
    seeds
        .iter()
        .map(|&s| {
            let sm = sample(pool, s, 300);
            let mut acc = [0.0; 8];
            for &qi in &sm {
                let a = per_q[pos[&qi]].a();
                for k in 0..8 {
                    acc[k] += a[k];
                }
            }
            acc.iter_mut().for_each(|x| *x /= sm.len() as f64);
            acc
        })
        .collect()
}
fn colk(ps: &[[f64; 8]], k: usize) -> Vec<f64> {
    ps.iter().map(|a| a[k]).collect()
}
fn rep(name: &str, ps: &[[f64; 8]]) -> Value {
    let s: Vec<_> = (0..8).map(|k| summarize(&colk(ps, k))).collect();
    println!("{:<44} n {:>5.2} P {:.4} R {:.4} F1 {:.4} hit1 {:.4} MRR {:.4} chars {:>6.0} x {:.4}", name, s[0].mean, s[1].mean, s[2].mean, s[3].mean, s[4].mean, s[5].mean, s[6].mean, s[7].mean);
    let mut o = serde_json::Map::new();
    o.insert("name".into(), json!(name));
    for k in 0..8 {
        o.insert(MN[k].into(), json!({"mean": s[k].mean, "ci_half": s[k].half_width}));
    }
    Value::Object(o)
}

struct Ranked {
    ids: Vec<usize>,
    z: Vec<f32>,
}
fn rank_top(s: &[f32], top: usize) -> Ranked {
    let mut idx: Vec<usize> = (0..s.len()).collect();
    idx.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(top);
    Ranked { z: idx.iter().map(|&j| s[j]).collect(), ids: idx }
}
fn mrr_of(c: &Corpus, q: &Query, s: &[f32]) -> (f64, f64) {
    let e = eval_scores(c, q, s);
    (e.mrr, e.hit1)
}

#[derive(Clone, Copy, Debug)]
enum Strat {
    Fixed(usize),
    Drop(f32, usize),
    Ratio(f32, usize),
    Budget(usize, usize),
}
impl Strat {
    fn label(&self) -> String {
        match self {
            Strat::Fixed(k) => format!("fixed-k{k}"),
            Strat::Drop(d, k) => format!("adaptive-drop{d}(kmax{k})"),
            Strat::Ratio(r, k) => format!("adaptive-ratio{r}(kmax{k})"),
            Strat::Budget(b, k) => format!("budget{b}ch(kmax{k})"),
        }
    }
    fn pick(&self, rk: &Ranked, docs: &[Doc]) -> usize {
        match *self {
            Strat::Fixed(k) => k.min(rk.ids.len()),
            Strat::Drop(d, kmax) => {
                let top = rk.z[0];
                (1..kmax.min(rk.ids.len())).take_while(|&i| rk.z[i] >= top - d).count() + 1
            }
            Strat::Ratio(r, kmax) => {
                let top = rk.z[0];
                (1..kmax.min(rk.ids.len())).take_while(|&i| rk.z[i] >= r * top).count() + 1
            }
            Strat::Budget(b, kmax) => {
                let mut tot = 0usize;
                let mut n = 0;
                for i in 0..kmax.min(rk.ids.len()) {
                    let l = docs[rk.ids[i]].text.len();
                    if n > 0 && tot + l > b {
                        break;
                    }
                    tot += l;
                    n += 1;
                }
                n.max(1)
            }
        }
    }
}

fn set_metrics(c: &Corpus, q: &Query, ids: &[usize], rank_mrr: f64) -> M {
    let rel = &c.docs_of[q.concept as usize];
    let hits = ids.iter().filter(|j| rel.contains(j)).count() as f64;
    let (p, r, f1) = prf(hits, ids.len() as f64, rel.len() as f64);
    M { n_ret: ids.len() as f64, p, r, f1, hit1: ids.first().map(|j| rel.contains(j) as u8 as f64).unwrap_or(0.0), mrr: rank_mrr, chars: ids.iter().map(|&j| c.docs[j].text.len() as f64).sum(), extra: 0.0 }
}

// ---------------------------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let confirm = args.iter().any(|a| a == "--confirm");
    let held = parse_range(&args, "--held", if confirm { (61, 90) } else { (31, 60) });
    let tune_seeds = parse_range(&args, "--tune", (1, 8));
    let tag = if confirm { "B" } else { "A" };
    let t0 = Instant::now();
    let ctx = Ctx::new(tag, &["mini", "bge"], false);
    let c = &ctx.c;
    let nd = c.docs.len();
    let all_q: Vec<usize> = (0..c.queries.len()).filter(|&i| c.queries[i].concept >= 0).collect();
    let tune_q: Vec<usize> = all_q.iter().copied().filter(|&i| is_tune_concept(c.queries[i].concept)).collect();
    let held_q: Vec<usize> = all_q.iter().copied().filter(|&i| !is_tune_concept(c.queries[i].concept)).collect();
    let ood_q: Vec<usize> = (0..c.queries.len()).filter(|&i| c.queries[i].concept < 0).collect();
    println!("corpus {tag}: {} docs, {} queries ({} tune, {} held-out, {} off-topic); loaded in {:.1}s", nd, c.queries.len(), tune_q.len(), held_q.len(), ood_q.len(), t0.elapsed().as_secs_f64());
    let picks_path = "results/g16_retrieval_picks.json";
    let mut picks: Value = if confirm { serde_json::from_str(&std::fs::read_to_string(picks_path).expect("run corpus A first")).unwrap() } else { json!({}) };
    let mut out = serde_json::Map::new();

    // systems for doc retrieval
    let mini_s = { let mut x = Cfg::base("mini+spell", "mini"); x.qv = QV::Spell; x };
    let mini_h = { let mut x = mini_s.clone(); x.name = "mini-hybrid".into(); x.lexical = true; x.alpha = 0.7; x };
    let bge_a = Cfg::base("bge", "bge");
    let bge_h = { let mut x = Cfg::base("bge-hybrid", "bge"); x.qv = QV::Spell; x.lexical = true; x.alpha = 0.8; x };
    let systems = [mini_s.clone(), mini_h.clone(), bge_a.clone(), bge_h.clone()];

    // precompute top-20 rankings for every query under every system
    let n_q = c.queries.len();
    let ranks: Vec<Vec<Ranked>> = systems.iter().map(|cfg| (0..n_q).into_par_iter().map(|qi| rank_top(&ctx.score(cfg, qi), 20)).collect()).collect();
    let pos_tune: HashMap<usize, usize> = tune_q.iter().enumerate().map(|(i, &q)| (q, i)).collect();
    let pos_held: HashMap<usize, usize> = held_q.iter().enumerate().map(|(i, &q)| (q, i)).collect();
    // mrr per (system, query)
    let mrrs: Vec<Vec<f64>> = systems.iter().enumerate().map(|(si, cfg)| (0..n_q).into_par_iter().map(|qi| if c.queries[qi].concept >= 0 { mrr_of(c, &c.queries[qi], &ctx.score(cfg, qi)).0 } else { 0.0 }).collect()).collect();

    // ================================================================= E1 adaptive k
    println!("\n##### E1 adaptive k (doc-level; relevant = the 2 docs of the query's concept) #####");
    let strats_grid: Vec<Strat> = {
        let mut v = vec![Strat::Fixed(1), Strat::Fixed(2), Strat::Fixed(3), Strat::Fixed(5)];
        for &d in &[0.5f32, 1.0, 1.5, 2.0, 3.0, 4.0] {
            v.push(Strat::Drop(d, 5));
        }
        for &r in &[0.5f32, 0.6, 0.7, 0.8, 0.9] {
            v.push(Strat::Ratio(r, 5));
        }
        for &b in &[1200usize, 1800, 2400] {
            v.push(Strat::Budget(b, 5));
        }
        v
    };
    let eval_strat = |si: usize, st: &Strat, pool: &[usize]| -> Vec<M> {
        pool.par_iter()
            .map(|&qi| {
                let rk = &ranks[si][qi];
                let k = st.pick(rk, &c.docs);
                set_metrics(c, &c.queries[qi], &rk.ids[..k], mrrs[si][qi])
            })
            .collect()
    };
    let mut e1 = vec![];
    let mut e1_cmp = vec![];
    for (si, cfg) in systems.iter().enumerate() {
        // tune on tune split: best fixed k by F1 and best adaptive by F1 (family-wise: drop, ratio, budget separately)
        let tune_f1 = |st: &Strat| -> f64 {
            let pm = eval_strat(si, st, &tune_q);
            let ps = seed_means(&pm, &pos_tune, &tune_q, &tune_seeds);
            summarize(&colk(&ps, 3)).mean
        };
        let key = format!("e1_{}", cfg.name);
        let (fixed_best, drop_best, ratio_best, budget_best) = if confirm {
            let g = &picks[&key];
            let gs = |n: &str| -> Strat {
                match g[n]["t"].as_str().unwrap() {
                    "fixed" => Strat::Fixed(g[n]["v"].as_u64().unwrap() as usize),
                    "drop" => Strat::Drop(g[n]["v"].as_f64().unwrap() as f32, 5),
                    "ratio" => Strat::Ratio(g[n]["v"].as_f64().unwrap() as f32, 5),
                    _ => Strat::Budget(g[n]["v"].as_u64().unwrap() as usize, 5),
                }
            };
            (gs("fixed"), gs("drop"), gs("ratio"), gs("budget"))
        } else {
            let best = |f: &dyn Fn(&Strat) -> bool| -> Strat { strats_grid.iter().filter(|s| f(s)).fold((f64::MIN, Strat::Fixed(1)), |b, s| { let m = tune_f1(s); if m > b.0 { (m, *s) } else { b } }).1 };
            let fb = best(&|s| matches!(s, Strat::Fixed(_)));
            let db = best(&|s| matches!(s, Strat::Drop(..)));
            let rb = best(&|s| matches!(s, Strat::Ratio(..)));
            let bb = best(&|s| matches!(s, Strat::Budget(..)));
            let js = |s: &Strat| match *s {
                Strat::Fixed(k) => json!({"t": "fixed", "v": k}),
                Strat::Drop(d, _) => json!({"t": "drop", "v": d}),
                Strat::Ratio(r, _) => json!({"t": "ratio", "v": r}),
                Strat::Budget(b, _) => json!({"t": "budget", "v": b}),
            };
            picks[&key] = json!({"fixed": js(&fb), "drop": js(&db), "ratio": js(&rb), "budget": js(&bb)});
            (fb, db, rb, bb)
        };
        println!("-- system {} : tuned picks on tune split: fixed {:?} | drop {:?} | ratio {:?} | budget {:?}", cfg.name, fixed_best, drop_best, ratio_best, budget_best);
        let mut rows: Vec<(String, Strat)> = vec![("k1".into(), Strat::Fixed(1)), ("k2".into(), Strat::Fixed(2)), ("k3".into(), Strat::Fixed(3)), ("k5".into(), Strat::Fixed(5)), ("fixed-best".into(), fixed_best), ("adaptive-drop*".into(), drop_best), ("adaptive-ratio*".into(), ratio_best), ("budget*".into(), budget_best)];
        rows.dedup_by(|a, b| a.0 == b.0);
        let mut pss: Vec<(String, Vec<[f64; 8]>)> = vec![];
        for (nm, st) in &rows {
            let pm = eval_strat(si, st, &held_q);
            let ps = seed_means(&pm, &pos_held, &held_q, &held);
            e1.push(rep(&format!("{} {} [{}]", cfg.name, nm, st.label()), &ps));
            pss.push((nm.clone(), ps));
        }
        let get = |n: &str| pss.iter().find(|x| x.0 == n).unwrap().1.clone();
        for (a, b) in [("adaptive-drop*", "fixed-best"), ("adaptive-ratio*", "fixed-best"), ("budget*", "fixed-best"), ("adaptive-drop*", "k3"), ("adaptive-drop*", "k2")] {
            for k in [3usize, 1] {
                let cc = cmp(&colk(&get(a), k), &colk(&get(b), k), 0.0);
                e1_cmp.push((cfg.name.clone(), a, b, MN[k], cc));
            }
        }
    }
    let adj = holm(&e1_cmp.iter().map(|x| x.4.p).collect::<Vec<_>>());
    println!("paired diffs on {} (Holm over {} tests):", if confirm { "confirm" } else { "held-out" }, e1_cmp.len());
    let mut e1c = vec![];
    for (x, a) in e1_cmp.iter().zip(&adj) {
        println!("  {:<12} {:<16} - {:<11} {:<4} {:+.4} [{:+.4},{:+.4}] holm_p {:.4} {}", x.0, x.1, x.2, x.3, x.4.diff, x.4.lo, x.4.hi, a, mark(*a, x.4.diff));
        e1c.push(json!({"system": x.0, "a": x.1, "b": x.2, "metric": x.3, "diff": x.4.diff, "ci": [x.4.lo, x.4.hi], "holm_p": a, "verdict": mark(*a, x.4.diff)}));
    }
    out.insert("E1".into(), json!({"rows": e1, "paired": e1c}));

    // ================================================================= E2 dedup + freshness
    println!("\n##### E2 dedup + freshness (store labels: 30% docs re-ingested as exact duplicates [newer timestamp]; 50% of concepts have a superseding revision: only the NEWER doc is relevant, the older one is STALE) #####");
    let mini = &ctx.models["mini"];
    let hash01 = |x: usize| -> u64 { (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 33 };
    let revised: Vec<bool> = (0..c.nconc).map(|k| hash01(k + 17) % 2 == 0).collect();
    let dup_src: Vec<usize> = (0..nd).filter(|&d| hash01(d + 5) % 10 < 3).collect(); // dup entry j (id nd+j) duplicates doc dup_src[j]
    let base_of = |id: usize| -> usize { if id >= nd { dup_src[id - nd] } else { id } };
    // timestamps: docs 0..nd in index order, but for revised concepts the SECOND doc is the newer revision (+nd); duplicates are re-ingested latest (+2nd)
    let mut ts = vec![0usize; nd + dup_src.len()];
    for d in 0..nd {
        ts[d] = d;
    }
    for k in 0..c.nconc {
        if revised[k] {
            let ds = &c.docs_of[k];
            if ds.len() == 2 {
                ts[ds[1]] = nd + ds[1];
                ts[ds[0]] = ds[0];
            }
        }
    }
    for j in 0..dup_src.len() {
        ts[nd + j] = 2 * nd + j;
    }
    let fresh_rel = |k: usize| -> Vec<usize> {
        let ds = &c.docs_of[k];
        if revised[k] && ds.len() == 2 { vec![ds[1]] } else { ds.clone() }
    };
    let stale_of = |k: usize| -> Option<usize> {
        let ds = &c.docs_of[k];
        if revised[k] && ds.len() == 2 { Some(ds[0]) } else { None }
    };
    // extended ranking for system mini-hybrid: scores of duplicates = scores of source
    let si_e2 = 1usize; // mini-hybrid
    let ext_rank: Vec<Ranked> = (0..n_q)
        .into_par_iter()
        .map(|qi| {
            let mut s = ctx.score(&systems[si_e2], qi);
            for &src in &dup_src {
                let v = s[src];
                s.push(v);
            }
            // exact ties between a doc and its duplicate: stable (index) order puts the original first
            rank_top(&s, 30)
        })
        .collect();
    let doc_cos = |a: usize, b: usize| -> f32 { dot(mini.dvec(base_of(a)), mini.dvec(base_of(b))) };
    // dedup select: greedy over ranked list, near-duplicate (cos >= tau) of a selected doc -> keep the NEWER (if keep_newest) else skip; stop at k
    let dedup_select = |rk: &Ranked, tau: f32, keep_newest: bool, k: usize| -> Vec<usize> {
        let mut sel: Vec<usize> = vec![];
        for &j in &rk.ids {
            let mut dup_of: Option<usize> = None;
            for (si, &i) in sel.iter().enumerate() {
                if doc_cos(i, j) >= tau {
                    dup_of = Some(si);
                    break;
                }
            }
            match dup_of {
                Some(si) => {
                    if keep_newest && ts[j] > ts[sel[si]] {
                        sel[si] = j;
                    }
                }
                None => {
                    if sel.len() < k {
                        sel.push(j);
                    }
                }
            }
        }
        sel
    };
    let scenario_metrics = |qi: usize, sel: &[usize], use_rev: bool| -> M {
        let q = &c.queries[qi];
        let rel: Vec<usize> = if use_rev { fresh_rel(q.concept as usize) } else { c.docs_of[q.concept as usize].clone() };
        let stale = if use_rev { stale_of(q.concept as usize) } else { None };
        // unique-content hits: a duplicate of an already counted relevant doc is a wasted slot
        let mut seen: Vec<usize> = vec![];
        let mut hits = 0.0;
        for &j in sel {
            let b = base_of(j);
            if rel.contains(&b) && !seen.contains(&b) {
                hits += 1.0;
                seen.push(b);
            }
        }
        let (p, r, f1) = prf(hits, sel.len() as f64, rel.len() as f64);
        let stale_served = sel.iter().any(|&j| stale == Some(base_of(j))) as u8 as f64;
        M { n_ret: sel.len() as f64, p, r, f1, hit1: sel.first().map(|&j| rel.contains(&base_of(j)) as u8 as f64).unwrap_or(0.0), mrr: 0.0, chars: sel.iter().map(|&j| c.docs[base_of(j)].text.len() as f64).sum(), extra: stale_served }
    };
    let k_e2 = 3usize;
    // scenarios: (label, store has dups, revisions matter)
    let scen = [("dups-only", true, false), ("revisions-only", false, true), ("dups+revisions", true, true), ("clean (no dups, no revisions)", false, false)];
    let tau_grid = [0.999f32, 0.98, 0.95, 0.92, 0.90, 0.85];
    let eval_e2 = |tau: Option<f32>, keep_newest: bool, with_dups: bool, use_rev: bool, pool: &[usize]| -> Vec<M> {
        pool.par_iter()
            .map(|&qi| {
                let rk = &ext_rank[qi];
                // without dups in the store, drop duplicate entries from the candidate list
                let rk2 = if with_dups { Ranked { ids: rk.ids.clone(), z: rk.z.clone() } } else { let keep: Vec<usize> = (0..rk.ids.len()).filter(|&i| rk.ids[i] < nd).collect(); Ranked { ids: keep.iter().map(|&i| rk.ids[i]).collect(), z: keep.iter().map(|&i| rk.z[i]).collect() } };
                let sel: Vec<usize> = match tau {
                    None => rk2.ids.iter().copied().take(k_e2).collect(),
                    Some(t) => dedup_select(&rk2, t, keep_newest, k_e2),
                };
                scenario_metrics(qi, &sel, use_rev)
            })
            .collect()
    };
    let tau_key = "e2_tau";
    let tau_star = if confirm {
        picks[tau_key].as_f64().unwrap() as f32
    } else {
        // tune tau on the dups+revisions scenario, tune split, maximise F1
        let mut best = (f64::MIN, 0.95f32);
        for &t in &tau_grid {
            let pm = eval_e2(Some(t), true, true, true, &tune_q);
            let ps = seed_means(&pm, &pos_tune, &tune_q, &tune_seeds);
            let m = summarize(&colk(&ps, 3)).mean;
            println!("   tune tau {t}: F1 {m:.4}");
            if m > best.0 {
                best = (m, t);
            }
        }
        picks[tau_key] = json!(best.1);
        best.1
    };
    println!("tau* (near-duplicate cosine threshold, tuned on the dups+revisions scenario) = {tau_star}");
    let mut e2 = vec![];
    let mut e2c = vec![];
    for (sn, wd, ur) in scen {
        println!("-- scenario {sn}: k={k_e2}, mini-hybrid ranking; extra = stale doc served in the returned set");
        let mut res: Vec<(String, Vec<[f64; 8]>)> = vec![];
        let variants: Vec<(String, Option<f32>, bool)> = vec![("no-dedup".into(), None, false), (format!("dedup tau*={tau_star} (skip older/dup)"), Some(tau_star), false), (format!("dedup tau*={tau_star} keep-newest"), Some(tau_star), true), ("dedup tau=0.85 keep-newest (too loose)".into(), Some(0.85), true), ("dedup tau=0.999 keep-newest (exact-dup only)".into(), Some(0.999), true)];
        for (vn, tau, kn) in &variants {
            let pm = eval_e2(*tau, *kn, wd, ur, &held_q);
            let ps = seed_means(&pm, &pos_held, &held_q, &held);
            e2.push(rep(&format!("[{sn}] {vn}"), &ps));
            res.push((vn.clone(), ps));
        }
        for i in 1..res.len() {
            for k in [3usize, 7] {
                let cc = cmp(&colk(&res[i].1, k), &colk(&res[0].1, k), 0.0);
                e2c.push((sn, res[i].0.clone(), MN[k], cc));
            }
        }
    }
    let adj = holm(&e2c.iter().map(|x| x.3.p).collect::<Vec<_>>());
    println!("paired vs no-dedup (F1, stale-served rate; Holm over {}):", e2c.len());
    let mut e2cj = vec![];
    for (x, a) in e2c.iter().zip(&adj) {
        println!("  [{}] {:<44} {:<5} {:+.4} [{:+.4},{:+.4}] holm_p {:.4} {}", x.0, x.1, x.2, x.3.diff, x.3.lo, x.3.hi, a, mark(*a, if x.2 == "extra" { -x.3.diff } else { x.3.diff }));
        e2cj.push(json!({"scenario": x.0, "variant": x.1, "metric": x.2, "diff": x.3.diff, "ci": [x.3.lo, x.3.hi], "holm_p": a}));
    }
    out.insert("E2".into(), json!({"tau_star": tau_star, "rows": e2, "paired_vs_nodedup": e2cj}));

    // ================================================================= E3 partitioned indices
    println!("\n##### E3a project partitions: pooled vs tag-partitioned vs soft tag boost (two PROJECT corpora = sibling corpora with the same concepts; relevant = own project's docs) #####");
    let (ta, tb) = if confirm { ("C", "D") } else { ("A", "B") };
    let (ca, cb) = (load_corpus(ta), load_corpus(tb));
    let (ma, mb) = (load_model(&ca, "mini"), load_model(&cb, "mini"));
    let (nda, ndb) = (ca.docs.len(), cb.docs.len());
    // query sets: (project, qi)
    let proj_q = |cs: &Corpus, tune: bool| -> Vec<usize> { (0..cs.queries.len()).filter(|&i| cs.queries[i].concept >= 0 && (is_tune_concept(cs.queries[i].concept) == tune)).collect() };
    // cosine query(spell) x doc blocks
    let sc = |mq: &Model, qi: usize, md: &Model, nds: usize| -> Vec<f32> { let q = mq.qvec(QV::Spell, qi); (0..nds).map(|j| dot(q, md.dvec(j))).collect() };
    // joint index: 0..nda = project A docs, nda.. = project B docs
    struct PQ {
        proj: usize,
        qi: usize,
    }
    let mk_pool = |tune: bool| -> Vec<PQ> { let mut v: Vec<PQ> = proj_q(&ca, tune).into_iter().map(|qi| PQ { proj: 0, qi }).collect(); v.extend(proj_q(&cb, tune).into_iter().map(|qi| PQ { proj: 1, qi })); v };
    let pool_tune = mk_pool(true);
    let pool_held = mk_pool(false);
    let scores_of = |pq: &PQ| -> Vec<f32> {
        let (mq, cq) = if pq.proj == 0 { (&ma, &ca) } else { (&mb, &cb) };
        let _ = cq;
        let mut s = sc(mq, pq.qi, &ma, nda);
        s.extend(sc(mq, pq.qi, &mb, ndb));
        s
    };
    let rel_of = |pq: &PQ| -> Vec<usize> {
        if pq.proj == 0 {
            ca.docs_of[ca.queries[pq.qi].concept as usize].clone()
        } else {
            cb.docs_of[cb.queries[pq.qi].concept as usize].iter().map(|&d| nda + d).collect()
        }
    };
    let noisy = |idx: usize, eps: f64| -> bool { ((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 11) as f64 / (1u64 << 53) as f64 <= eps };
    // strategy: closure over (scores, proj, idx) -> returned doc ids (top-3), plus "wrong block @1"
    #[derive(Clone, Copy, Debug)]
    enum Part {
        Pooled,
        Exact,
        Noisy(f64),
        Boost(f32, f64),
        Fallback(f32, f64),
    }
    let part_run = |st: Part, pq: &PQ, idx: usize, s: &Vec<f32>| -> Vec<usize> {
        let tag_true = pq.proj;
        let nt = nda + ndb;
        let block = |b: usize| -> std::ops::Range<usize> { if b == 0 { 0..nda } else { nda..nt } };
        let topk = |sc: &dyn Fn(usize) -> f32, range: std::ops::Range<usize>, k: usize| -> Vec<usize> {
            let mut v: Vec<(f32, usize)> = range.map(|j| (sc(j), j)).collect();
            v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            v.into_iter().take(k).map(|x| x.1).collect()
        };
        match st {
            Part::Pooled => topk(&|j| s[j], 0..nt, 3),
            Part::Exact => topk(&|j| s[j], block(tag_true), 3),
            Part::Noisy(eps) => {
                let tag = if noisy(idx, eps) { 1 - tag_true } else { tag_true };
                topk(&|j| s[j], block(tag), 3)
            }
            Part::Boost(beta, eps) => {
                let tag = if noisy(idx, eps) { 1 - tag_true } else { tag_true };
                topk(&|j| s[j] + if block(tag).contains(&j) { beta } else { 0.0 }, 0..nt, 3)
            }
            Part::Fallback(tf, eps) => {
                let tag = if noisy(idx, eps) { 1 - tag_true } else { tag_true };
                let r = topk(&|j| s[j], block(tag), 3);
                if s[r[0]] >= tf { r } else { topk(&|j| s[j], 0..nt, 3) }
            }
        }
    };
    let eval_part = |st: Part, pool: &[PQ]| -> Vec<M> {
        pool.par_iter()
            .enumerate()
            .map(|(idx, pq)| {
                let s = scores_of(pq);
                let ids = part_run(st, pq, idx, &s);
                let rel = rel_of(pq);
                let hits = ids.iter().filter(|j| rel.contains(j)).count() as f64;
                let (p, r, f1) = prf(hits, ids.len() as f64, rel.len() as f64);
                // MRR over the whole pooled ranking (for the partitioned arms this is the in-partition rank)
                let mrr = {
                    let order: Vec<usize> = ids.clone();
                    order.iter().position(|j| rel.contains(j)).map(|p| 1.0 / (p + 1) as f64).unwrap_or(0.0)
                };
                // wrong-project top-1
                let own_block = if pq.proj == 0 { 0..nda } else { nda..nda + ndb };
                let wrong = ids.first().map(|j| !own_block.contains(j) as u8 as f64).unwrap_or(0.0);
                M { n_ret: 3.0, p, r, f1, hit1: ids.first().map(|j| rel.contains(j) as u8 as f64).unwrap_or(0.0), mrr, chars: 0.0, extra: wrong }
            })
            .collect()
    };
    let pos_idx = |pool: &[PQ]| -> (HashMap<usize, usize>, Vec<usize>) { ((0..pool.len()).map(|i| (i, i)).collect(), (0..pool.len()).collect()) };
    let (pt, ptq) = pos_idx(&pool_tune);
    let (ph, phq) = pos_idx(&pool_held);
    // tune beta and fallback threshold at eps 0.2 (a realistic noisy tag), on the tune split
    let tune_part = |mk: &dyn Fn(f32) -> Part, grid: &[f32]| -> f32 {
        let mut best = (f64::MIN, grid[0]);
        for &g in grid {
            let pm = eval_part(mk(g), &pool_tune);
            let ps = seed_means(&pm, &pt, &ptq, &tune_seeds);
            let m = summarize(&colk(&ps, 3)).mean;
            if m > best.0 {
                best = (m, g);
            }
        }
        best.1
    };
    let (beta_star, tf_star) = if confirm {
        (picks["e3_beta"].as_f64().unwrap() as f32, picks["e3_tf"].as_f64().unwrap() as f32)
    } else {
        let b = tune_part(&|g| Part::Boost(g, 0.2), &[0.01, 0.03, 0.06, 0.1, 0.2]);
        let f = tune_part(&|g| Part::Fallback(g, 0.2), &[0.3, 0.4, 0.5, 0.6, 0.7]);
        picks["e3_beta"] = json!(b);
        picks["e3_tf"] = json!(f);
        (b, f)
    };
    println!("tuned on tune split with 20% tag noise: soft boost beta* = {beta_star}, fallback threshold* = {tf_star}");
    let parts: Vec<(String, Part)> = vec![
        ("pooled (no tag)".into(), Part::Pooled),
        ("partition, exact tag".into(), Part::Exact),
        ("partition, 10% wrong tag".into(), Part::Noisy(0.1)),
        ("partition, 20% wrong tag".into(), Part::Noisy(0.2)),
        (format!("soft boost b*={beta_star}, exact tag"), Part::Boost(beta_star, 0.0)),
        (format!("soft boost b*={beta_star}, 20% wrong tag"), Part::Boost(beta_star, 0.2)),
        (format!("partition+fallback(pooled if top<{tf_star}), exact"), Part::Fallback(tf_star, 0.0)),
        (format!("partition+fallback(pooled if top<{tf_star}), 20% wrong"), Part::Fallback(tf_star, 0.2)),
    ];
    let mut e3 = vec![];
    let mut e3ps: Vec<(String, Vec<[f64; 8]>)> = vec![];
    for (nm, st) in &parts {
        let pm = eval_part(*st, &pool_held);
        let ps = seed_means(&pm, &ph, &phq, &held);
        e3.push(rep(&format!("[3a] {nm}"), &ps));
        e3ps.push((nm.clone(), ps));
    }
    println!("(extra = top-1 doc is from the OTHER project)");
    let mut e3c = vec![];
    for (a, b) in [(1usize, 0usize), (3, 0), (5, 3), (5, 0), (7, 3), (5, 7)] {
        for k in [3usize, 4, 7] {
            let cc = cmp(&colk(&e3ps[a].1, k), &colk(&e3ps[b].1, k), 0.0);
            e3c.push((e3ps[a].0.clone(), e3ps[b].0.clone(), MN[k], cc));
        }
    }
    let adj = holm(&e3c.iter().map(|x| x.3.p).collect::<Vec<_>>());
    let mut e3cj = vec![];
    for (x, a) in e3c.iter().zip(&adj) {
        println!("  {:<46} - {:<46} {:<5} {:+.4} [{:+.4},{:+.4}] holm_p {:.4} {}", x.0, x.1, x.2, x.3.diff, x.3.lo, x.3.hi, a, mark(*a, if x.2 == "extra" { -x.3.diff } else { x.3.diff }));
        e3cj.push(json!({"a": x.0, "b": x.1, "metric": x.2, "diff": x.3.diff, "ci": [x.3.lo, x.3.hi], "holm_p": a}));
    }
    // latency: scanning cost per query (dense, 384-d) for pooled vs partition
    let lat = |nds: usize| -> f64 {
        let reps = 400usize;
        let q = ma.qvec(QV::Spell, 0).to_vec();
        let t = Instant::now();
        let mut acc = 0.0f32;
        for r in 0..reps {
            for j in 0..nds {
                acc += dot(&q, ma.dvec((j + r) % nda));
            }
        }
        std::hint::black_box(acc);
        t.elapsed().as_secs_f64() * 1e6 / reps as f64
    };
    let (l_pool, l_part) = (lat(nda + ndb), lat(nda));
    println!("scan latency (dense 384-d, single thread): pooled {} docs {:.1} us/query | one partition {} docs {:.1} us/query | vectors {:.2} MB pooled vs {:.2} MB per partition", nda + ndb, l_pool, nda, l_part, (nda + ndb) as f64 * 384.0 * 4.0 / 1048576.0, nda as f64 * 384.0 * 4.0 / 1048576.0);

    println!("\n##### E3b topic-domain partitions in ONE corpus (10 domains x 3 topics; centroid routing of the query; probe 1 or 2 domains) #####");
    let ndom = (c.topic_of_concept.iter().max().unwrap() / 3 + 1) as usize;
    let dom_of_doc: Vec<usize> = (0..nd).map(|d| (c.docs[d].topic / 3) as usize).collect();
    let mut cent = vec![vec![0.0f32; mini.dim]; ndom];
    for d in 0..nd {
        if c.docs[d].concept >= 0 {
            for (x, y) in cent[dom_of_doc[d]].iter_mut().zip(mini.dvec(d)) {
                *x += y;
            }
        }
    }
    let cent: Vec<Vec<f32>> = cent.iter().map(|v| unit(v)).collect();
    let eval_dom = |probe: usize, pool: &[usize]| -> Vec<M> {
        pool.par_iter()
            .map(|&qi| {
                let q = mini.qvec(QV::Spell, qi);
                let mut ds: Vec<(f32, usize)> = (0..ndom).map(|d| (dot(q, &cent[d]), d)).collect();
                ds.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                let chosen: Vec<usize> = ds.iter().take(probe).map(|x| x.1).collect();
                let mut v: Vec<(f32, usize)> = (0..nd).filter(|&d| probe == 0 || chosen.contains(&dom_of_doc[d])).map(|d| (dot(q, mini.dvec(d)), d)).collect();
                v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                let ids: Vec<usize> = v.iter().take(3).map(|x| x.1).collect();
                let mut m = set_metrics(c, &c.queries[qi], &ids, 0.0);
                m.mrr = { let rel = &c.docs_of[c.queries[qi].concept as usize]; ids.iter().position(|j| rel.contains(j)).map(|p| 1.0 / (p + 1) as f64).unwrap_or(0.0) };
                m.extra = (chosen.first().copied() == Some((c.queries[qi].topic / 3) as usize)) as u8 as f64; // routing accuracy (first probe)
                m
            })
            .collect()
    };
    let mut e3b = vec![];
    let mut dps: Vec<Vec<[f64; 8]>> = vec![];
    for (nm, pr) in [("pooled (all domains)", 0usize), ("route to 1 domain", 1), ("route to 2 domains", 2)] {
        let pm = eval_dom(pr, &held_q);
        let ps = seed_means(&pm, &pos_held, &held_q, &held);
        e3b.push(rep(&format!("[3b] {nm}"), &ps));
        dps.push(ps);
    }
    println!("(extra = domain routing accuracy of the first probe; for pooled it is the trivial top-1 domain match and is not used)");
    let mut e3bc = vec![];
    for (a, b) in [(1usize, 0usize), (2, 0), (2, 1)] {
        for k in [3usize, 4, 2] {
            let cc = cmp(&colk(&dps[a], k), &colk(&dps[b], k), 0.0);
            e3bc.push((a, b, MN[k], cc));
        }
    }
    let adj = holm(&e3bc.iter().map(|x| x.3.p).collect::<Vec<_>>());
    let nm3 = ["pooled", "route1", "route2"];
    let mut e3bcj = vec![];
    for (x, a) in e3bc.iter().zip(&adj) {
        println!("  {:<8} - {:<8} {:<5} {:+.4} [{:+.4},{:+.4}] holm_p {:.4} {}", nm3[x.0], nm3[x.1], x.2, x.3.diff, x.3.lo, x.3.hi, a, mark(*a, x.3.diff));
        e3bcj.push(json!({"a": nm3[x.0], "b": nm3[x.1], "metric": x.2, "diff": x.3.diff, "ci": [x.3.lo, x.3.hi], "holm_p": a}));
    }
    out.insert("E3".into(), json!({"project_pair": [ta, tb], "beta_star": beta_star, "fallback_star": tf_star, "rows_3a": e3, "paired_3a": e3cj, "scan_us_pooled": l_pool, "scan_us_partition": l_part, "rows_3b": e3b, "paired_3b": e3bcj}));

    // ================================================================= E4 retrieval-level gate (abstention)
    println!("\n##### E4 retrieval-level confidence gate: inject the top-1 doc only if predicted to be the right concept's doc (pool includes the 200 off-topic queries) #####");
    let ood_tune: Vec<usize> = ood_q.iter().copied().enumerate().filter(|(i, _)| i % 2 == 0).map(|x| x.1).collect();
    let ood_held: Vec<usize> = ood_q.iter().copied().enumerate().filter(|(i, _)| i % 2 == 1).map(|x| x.1).collect();
    let mut gate_tune: Vec<usize> = tune_q.clone();
    gate_tune.extend(&ood_tune);
    let mut gate_held: Vec<usize> = held_q.clone();
    gate_held.extend(&ood_held);
    let pos_gt: HashMap<usize, usize> = gate_tune.iter().enumerate().map(|(i, &q)| (q, i)).collect();
    let pos_gh: HashMap<usize, usize> = gate_held.iter().enumerate().map(|(i, &q)| (q, i)).collect();
    let lambda = 1.0f64;
    let mut e4 = vec![];
    let mut e4c = vec![];
    for si in [1usize, 2] {
        let cfg = &systems[si];
        // features of query qi
        let feat = |qi: usize| -> [f64; 4] {
            let rk = &ranks[si][qi];
            let (z1, z2) = (rk.z[0] as f64, rk.z[1] as f64);
            let agree = dot(mini.dvec(rk.ids[0]), mini.dvec(rk.ids[1])) as f64; // do the top-2 docs say the same thing?
            [1.0, z1 / 10.0, (z1 - z2) / 5.0, agree]
        };
        let correct = |qi: usize| -> f64 { let q = &c.queries[qi]; if q.concept < 0 { 0.0 } else { (c.docs[rank_top_one(&ranks[si][qi])].concept == q.concept) as u8 as f64 } };
        // offline logistic regression on the tune pool (label = top-1 doc is the right concept's; off-topic = 0)
        let xs: Vec<([f64; 4], f64)> = gate_tune.iter().map(|&qi| (feat(qi), correct(qi))).collect();
        let mut w = [0.0f64; 4];
        for _ in 0..600 {
            let mut g = [0.0f64; 4];
            for (x, y) in &xs {
                let p = sigmoid(w.iter().zip(x).map(|(a, b)| a * b).sum());
                for i in 0..4 {
                    g[i] += (y - p) * x[i];
                }
            }
            for i in 0..4 {
                w[i] += 2.0 * g[i] / xs.len() as f64;
            }
        }
        let score_lr = |qi: usize| -> f64 { sigmoid(w.iter().zip(&feat(qi)).map(|(a, b)| a * b).sum()) };
        let sig: Vec<(&str, Box<dyn Fn(usize) -> f64>)> = vec![("z1", Box::new(|qi| feat(qi)[1])), ("margin", Box::new(|qi| feat(qi)[2])), ("agree(top1,top2)", Box::new(|qi| feat(qi)[3])), ("learned-logistic", Box::new(score_lr))];
        // threshold per signal maximizing utility on the tune pool; utility per query = correct answered - lambda * wrong answered
        let util = |scorer: &dyn Fn(usize) -> f64, thr: f64, pool: &[usize]| -> f64 { pool.iter().map(|&qi| if scorer(qi) >= thr { if correct(qi) > 0.5 { 1.0 } else { -lambda } } else { 0.0 }).sum::<f64>() / pool.len() as f64 };
        let key = format!("e4_{}", cfg.name);
        let mut thrs: Vec<(String, f64)> = vec![];
        for (nm, f) in &sig {
            let thr = if confirm {
                picks[&key][*nm].as_f64().unwrap()
            } else {
                let mut vals: Vec<f64> = gate_tune.iter().map(|&qi| f(qi)).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let cands: Vec<f64> = (1..40).map(|i| vals[(i * vals.len() / 40).min(vals.len() - 1)]).collect();
                let mut best = (f64::MIN, cands[0]);
                for &t in &cands {
                    let u = util(f.as_ref(), t, &gate_tune);
                    if u > best.0 {
                        best = (u, t);
                    }
                }
                picks[&key][*nm] = json!(best.1);
                best.1
            };
            thrs.push((nm.to_string(), thr));
        }
        println!("-- system {}: logistic weights {:?}", cfg.name, w.map(|x| (x * 100.0).round() / 100.0));
        let mut variants: Vec<(String, Vec<bool>)> = vec![("always inject".into(), vec![true; n_q])];
        for (i, (nm, f)) in sig.iter().enumerate() {
            let thr = thrs[i].1;
            variants.push((format!("gate {nm} >= {:.3}", thr), (0..n_q).map(|qi| f(qi) >= thr).collect()));
        }
        let mut ps_all: Vec<(String, Vec<[f64; 8]>)> = vec![];
        for (nm, inj) in &variants {
            // M fields: n_ret=injected, p=precision of injected (hit1 among injected), r = coverage of answerable, f1 = utility, hit1 = wrong-inject share of ALL queries, mrr = off-topic inject rate, extra = injected correct
            let pm: Vec<M> = gate_held
                .iter()
                .map(|&qi| {
                    let q = &c.queries[qi];
                    let inj_q = inj[qi];
                    let cr = correct(qi);
                    let is_ood = q.concept < 0;
                    M {
                        n_ret: inj_q as u8 as f64,
                        p: if inj_q { cr } else { 0.0 },
                        r: if !is_ood { (inj_q && cr > 0.5) as u8 as f64 } else { 0.0 },
                        f1: if inj_q { if cr > 0.5 { 1.0 } else { -lambda } } else { 0.0 },
                        hit1: (inj_q && cr < 0.5) as u8 as f64,
                        mrr: (inj_q && is_ood) as u8 as f64,
                        chars: 0.0,
                        extra: 0.0,
                    }
                })
                .collect();
            let ps = seed_means(&pm, &pos_gh, &gate_held, &held);
            let s: Vec<_> = (0..8).map(|k| summarize(&colk(&ps, k))).collect();
            println!("   {:<46} inject-rate {:.3}  precision-of-injected {:.3}  utility/query {:+.4}  wrong-inject {:.3}  off-topic-inject {:.3}  correct-injected {:.3}", format!("{} / {}", cfg.name, nm), s[0].mean, if s[0].mean > 0.0 { s[1].mean / s[0].mean } else { 0.0 }, s[3].mean, s[4].mean, s[5].mean, s[2].mean);
            e4.push(json!({"system": cfg.name, "gate": nm, "inject_rate": s[0].mean, "utility_per_query": s[3].mean, "wrong_inject": s[4].mean, "offtopic_inject": s[5].mean, "correct_injected": s[2].mean}));
            ps_all.push((nm.clone(), ps));
        }
        for i in 1..ps_all.len() {
            for (k, nm) in [(3usize, "utility"), (4, "wrong-inject")] {
                let cc = cmp(&colk(&ps_all[i].1, k), &colk(&ps_all[0].1, k), 0.0);
                e4c.push((cfg.name.clone(), ps_all[i].0.clone(), nm, cc));
            }
        }
    }
    let adj = holm(&e4c.iter().map(|x| x.3.p).collect::<Vec<_>>());
    let mut e4cj = vec![];
    println!("paired vs always-inject (utility = +1 correct injected, -{lambda} wrong injected; Holm over {}):", e4c.len());
    for (x, a) in e4c.iter().zip(&adj) {
        println!("  {:<12} {:<46} {:<12} {:+.4} [{:+.4},{:+.4}] holm_p {:.4} {}", x.0, x.1, x.2, x.3.diff, x.3.lo, x.3.hi, a, mark(*a, if x.2 == "wrong-inject" { -x.3.diff } else { x.3.diff }));
        e4cj.push(json!({"system": x.0, "gate": x.1, "metric": x.2, "diff": x.3.diff, "ci": [x.3.lo, x.3.hi], "holm_p": a}));
    }
    out.insert("E4".into(), json!({"lambda": lambda, "rows": e4, "paired": e4cj}));

    // baseline reference table (all four systems, k=1/3, held-out)
    println!("\n##### reference: doc retrieval metrics of the base systems (held-out, k=3) #####");
    let mut refrows = vec![];
    for (si, cfg) in systems.iter().enumerate() {
        let pm = eval_strat(si, &Strat::Fixed(3), &held_q);
        let ps = seed_means(&pm, &pos_held, &held_q, &held);
        refrows.push(rep(&format!("{} k3", cfg.name), &ps));
    }
    out.insert("reference_k3".into(), json!(refrows));
    if !confirm {
        std::fs::write(picks_path, serde_json::to_string_pretty(&picks).unwrap()).unwrap();
    }
    let path = if confirm { "results/g16_retrieval_confirm.json" } else { "results/g16_retrieval.json" };
    std::fs::write(path, serde_json::to_string_pretty(&Value::Object(out)).unwrap()).unwrap();
    println!("wrote {path} (total {:.1}s)", t0.elapsed().as_secs_f64());
}

fn rank_top_one(r: &Ranked) -> usize {
    r.ids[0]
}
