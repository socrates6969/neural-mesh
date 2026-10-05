//! G9-C (W10 RESEARCH-ABILITY): MULTI-STEP research sessions of 3-5 DEPENDENT steps on real MiniLM / BGE-base embeddings (g1 corpus, fresh corpora).
//!
//! Session = a user states a setup once (step 1: a real g1 user query for concept (topic T1, detail d*), any class: clean / typo / terse / chatty), then the agent
//! works through L-1 further procedures of the same detail pool. The query of step k+1 is WRITTEN FROM THE EVIDENCE OF STEP k: the agent extracts the detail of the
//! retrieved top-1 document (gazetteer match on the document text) and fills it into the next procedure query ("How do I <act of T_k+1 in <form>>?"). So a wrong
//! detail at step k is silently carried forward (error propagation) and a document from another pool leaves the session stuck. Gold = concept (T_k, d*).
//! Policies on top of every retrieval system: `none`; `local` (verify the top-1 document's detail against the detail the query itself states, via a pool gazetteer;
//! on mismatch take the best top-20 document that agrees); `pin` (as local, plus pin the detail the USER stated in step 1 and use it instead of the evidence-inherited
//! detail when writing later queries). Retrieval systems (frozen g1 configs, not re-tuned): dense MiniLM / BGE raw or spell-corrected, dense+BM25 hybrid, spell+overlap rerank, BM25.
//! Metrics: session success (all steps top-1 correct), step accuracy, P@3 / R@3 of the evidence set (relevant = the 2 docs of the gold concept), stuck rate.
//! Controls: `indep` (each follow-up is built from the GOLD detail: no propagation) isolates the propagation penalty; world `clean-step1` (only clean step-1 queries)
//! is where verification has little to fix; `typo-step1` is where spell-correction/verification should matter most.
//! Protocol: no tuning (every config is a frozen g1 pick); held-out 31-60 on corpus C, `--corpus D --held 61-90` = fresh corpus + fresh seeds. Paired 95% t-CI, Holm per family.
//!
//! Needs: python py/g9_chains.py <C|D>; python py/g1_embed.py <C|D>c  (embeds the follow-up queries with the on-disk MiniLM / BGE).
//! Usage: cargo run --release --offline --bin g9_multistep -- --corpus C --held 31-60

use std::collections::HashMap;
use std::time::Instant;

use neural_mesh::router::Rng;
use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
use common::*;
#[path = "g9_common/mod.rs"]
mod c9;
use c9::{holm_by_family, mk_test, test_json, verdict, Test};

const N_SESS: usize = 300;

// ------------------------------------------------------------------------------------------------
struct Gaz {
    /// pool name -> per detail: list of lowercase forms
    forms: HashMap<String, Vec<Vec<String>>>,
    pool_of_topic: Vec<String>,
    cid: HashMap<(i32, i32), usize>,
    /// topics per pool
    topics_of_pool: HashMap<String, Vec<i32>>,
}
fn load_gaz(tag: &str) -> (Gaz, Vec<Option<[usize; 5]>>) {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(format!("results/g1_corpus_{tag}.json")).unwrap()).unwrap();
    let mut forms: HashMap<String, Vec<Vec<String>>> = HashMap::new();
    let mut pool_of_topic = vec![String::new(); 30];
    let mut cid = HashMap::new();
    let mut topics_of_pool: HashMap<String, Vec<i32>> = HashMap::new();
    for c in v["concepts"].as_array().unwrap() {
        let pool = c["pool"].as_str().unwrap().to_string();
        let (t, d) = (c["topic"].as_i64().unwrap() as i32, c["detail"].as_i64().unwrap() as usize);
        let e = forms.entry(pool.clone()).or_insert_with(|| vec![vec![]; 8]);
        e[d] = c["forms"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_lowercase()).collect();
        pool_of_topic[t as usize] = pool.clone();
        cid.insert((t, d as i32), c["id"].as_u64().unwrap() as usize);
        let tp = topics_of_pool.entry(pool).or_default();
        if !tp.contains(&t) {
            tp.push(t);
        }
    }
    let chain: Vec<Option<[usize; 5]>> = v["queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| q.get("chain").map(|c| { let a = c.as_array().unwrap(); [a[0].as_u64().unwrap() as usize, a[1].as_u64().unwrap() as usize, a[2].as_u64().unwrap() as usize, a[3].as_u64().unwrap() as usize, a[4].as_u64().unwrap() as usize] }))
        .collect();
    (Gaz { forms, pool_of_topic, cid, topics_of_pool }, chain)
}

/// whole-word (alnum-boundary) occurrences of `f` in `t` (both lowercase)
fn count_word(t: &str, f: &str) -> usize {
    let (tb, fb) = (t.as_bytes(), f.as_bytes());
    if fb.is_empty() || tb.len() < fb.len() {
        return 0;
    }
    let mut n = 0;
    let mut i = 0;
    while i + fb.len() <= tb.len() {
        if &tb[i..i + fb.len()] == fb {
            let before = i == 0 || !(tb[i - 1] as char).is_alphanumeric();
            let after = i + fb.len() == tb.len() || !(tb[i + fb.len()] as char).is_alphanumeric();
            if before && after {
                n += 1;
            }
            i += fb.len();
        } else {
            i += 1;
        }
    }
    n
}
/// detail of `pool` stated in `text` (None if no or ambiguous match)
fn detect_in(forms: &HashMap<String, Vec<Vec<String>>>, pool: &str, text: &str) -> Option<usize> {
    let t = text.to_lowercase();
    let fm = forms.get(pool)?;
    let counts: Vec<usize> = fm.iter().map(|fs| fs.iter().map(|f| count_word(&t, f)).sum()).collect();
    let mx = *counts.iter().max()?;
    if mx == 0 || counts.iter().filter(|&&c| c == mx).count() != 1 {
        return None;
    }
    counts.iter().position(|&c| c == mx)
}

fn detect(g: &Gaz, pool: &str, text: &str) -> Option<usize> {
    detect_in(&g.forms, pool, text)
}

// ------------------------------------------------------------------------------------------------
#[derive(Clone, Copy, PartialEq, Debug)]
enum Pol {
    None,
    Local,
    Pin,
}
#[derive(Clone, Debug)]
struct Sys {
    name: &'static str,
    model: Option<&'static str>,
    spell: bool,
    /// dense weight in z-fusion with BM25 (1.0 = dense only)
    alpha: f32,
    gamma: f32,
}
fn systems() -> Vec<Sys> {
    vec![
        Sys { name: "bm25", model: None, spell: true, alpha: 0.0, gamma: 0.0 },
        Sys { name: "mini", model: Some("mini"), spell: false, alpha: 1.0, gamma: 0.0 },
        Sys { name: "mini+spell", model: Some("mini"), spell: true, alpha: 1.0, gamma: 0.0 },
        Sys { name: "mini-hybrid", model: Some("mini"), spell: true, alpha: 0.7, gamma: 0.0 },
        Sys { name: "mini-final", model: Some("mini"), spell: true, alpha: 1.0, gamma: 1.0 },
        Sys { name: "bge", model: Some("bge"), spell: false, alpha: 1.0, gamma: 0.0 },
        Sys { name: "bge+spell", model: Some("bge"), spell: true, alpha: 1.0, gamma: 0.0 },
        Sys { name: "bge-hybrid", model: Some("bge"), spell: true, alpha: 0.8, gamma: 0.0 },
        Sys { name: "bge-final", model: Some("bge"), spell: true, alpha: 1.0, gamma: 0.5 },
    ]
}

struct Env {
    c: Corpus,
    lex: Lex,
    /// (model, spell) -> nq x nd dense doc scores
    dense: HashMap<(&'static str, bool), Vec<f32>>,
    bm: [Vec<f32>; 3],
    q_terms: [Vec<Vec<u32>>; 3],
    g: Gaz,
    chain: Vec<Option<[usize; 5]>>,
    /// (topic, act, pat, detail, form) -> query index
    chain_ix: HashMap<[usize; 5], usize>,
    n_orig: usize,
    doc_pool: Vec<Option<String>>,
    /// detail of each doc as READ by the agent (full gazetteer): used to write the next query
    doc_detail: Vec<Option<usize>>,
    /// the VERIFIER's (possibly degraded) gazetteer: forms per detail kept, detection noise, and the doc details it can read
    vforms: HashMap<String, Vec<Vec<String>>>,
    vnoise: f64,
    lenient: bool,
    doc_detail_ver: Vec<Option<usize>>,
}

fn build_env(tag: &str) -> Env {
    let ctag = format!("{tag}c");
    let c = load_corpus(&ctag);
    let (g, chain) = load_gaz(&ctag);
    let lex = Lex::build(&c.docs);
    let (nq, nd) = (c.queries.len(), c.docs.len());
    let mut dense = HashMap::new();
    for (mn, key) in [("mini", "mini"), ("bge", "bge")] {
        let m = load_model(&c, mn);
        for spell in [false, true] {
            let qv = if spell { QV::Spell } else { QV::Raw };
            let mut v = vec![0.0f32; nq * nd];
            v.par_chunks_mut(nd).enumerate().for_each(|(i, row)| {
                let q = m.qvec(qv, i);
                for (j, r) in row.iter_mut().enumerate() {
                    *r = dot(q, m.dvec(j));
                }
            });
            dense.insert((key, spell), v);
        }
    }
    let bm = [bm25_mat(&lex, &c.queries, QV::Raw), bm25_mat(&lex, &c.queries, QV::Spell), vec![]];
    let qt = |f: &dyn Fn(&Query) -> &str| -> Vec<Vec<u32>> { c.queries.iter().map(|q| lex.q_terms(f(q))).collect() };
    let q_terms = [qt(&|q| &q.text), qt(&|q| &q.spell), vec![]];
    let mut chain_ix = HashMap::new();
    for (i, ch) in chain.iter().enumerate() {
        if let Some(k) = ch {
            chain_ix.insert(*k, i);
        }
    }
    let n_orig = chain.iter().position(|x| x.is_some()).unwrap();
    let doc_pool: Vec<Option<String>> = c.docs.iter().map(|d| if d.topic >= 0 { Some(g.pool_of_topic[d.topic as usize].clone()) } else { None }).collect();
    let doc_detail: Vec<Option<usize>> = c.docs.iter().zip(&doc_pool).map(|(d, p)| p.as_ref().and_then(|p| detect(&g, p, &d.text))).collect();
    let vforms = g.forms.clone();
    let doc_detail_ver = doc_detail.clone();
    Env { c, lex, dense, bm, q_terms, g, chain, chain_ix, n_orig, doc_pool, doc_detail, vforms, vnoise: 0.0, lenient: false, doc_detail_ver }
}

impl Env {
    /// degrade the verifier: keep the first `k` surface forms per detail, flip a detected detail to a random other one with prob `noise`
    fn set_verifier(&mut self, k: usize, noise: f64) {
        self.vforms = self.g.forms.iter().map(|(p, v)| (p.clone(), v.iter().map(|fs| fs.iter().take(k).cloned().collect()).collect())).collect();
        self.vnoise = noise;
        self.doc_detail_ver = self.c.docs.iter().zip(&self.doc_pool).map(|(d, p)| p.as_ref().and_then(|p| detect_in(&self.vforms, p, &d.text))).collect();
    }
    /// verifier-side detection of the detail a query text states (qi keys the noise so it is deterministic)
    fn detect_ver(&self, pool: &str, text: &str, qi: usize) -> Option<usize> {
        let d = detect_in(&self.vforms, pool, text)?;
        if self.vnoise > 0.0 && ((c9::hash(77, qi as u64, 1, 0) >> 11) as f64 / (1u64 << 53) as f64) < self.vnoise {
            return Some((d + 1 + c9::pick(77, qi as u64, 2, 0, 7)) % 8);
        }
        Some(d)
    }
    fn qtext(&self, sys: &Sys, qi: usize) -> &str {
        if sys.spell { &self.c.queries[qi].spell } else { &self.c.queries[qi].text }
    }
    /// ranked top-20 doc ids for query qi under system `sys` (same fusion as g1_common::Ctx::score)
    fn rank(&self, sys: &Sys, qi: usize) -> Vec<usize> {
        let nd = self.c.docs.len();
        let qv = sys.spell as usize;
        let dense = sys.model.map(|m| &self.dense[&(m, sys.spell)][qi * nd..(qi + 1) * nd]);
        let lexs = if sys.alpha < 1.0 { Some(&self.bm[qv][qi * nd..(qi + 1) * nd]) } else { None };
        let mut f: Vec<f32> = match (dense, lexs) {
            (Some(d), Some(l)) => {
                let (zd, zl) = (zscore(d), zscore(l));
                zd.iter().zip(&zl).map(|(a, b)| sys.alpha * a + (1.0 - sys.alpha) * b).collect()
            }
            (Some(d), None) => zscore(d),
            (None, Some(l)) => zscore(l),
            _ => unreachable!(),
        };
        let mut idx: Vec<usize> = (0..nd).collect();
        idx.sort_by(|&a, &b| f[b].partial_cmp(&f[a]).unwrap().then(a.cmp(&b)));
        idx.truncate(20);
        if sys.gamma > 0.0 {
            let zc = |v: Vec<f32>| zscore(&v);
            let base = zc(idx.iter().map(|&j| f[j]).collect());
            let ov = zc(idx.iter().map(|&j| self.lex.overlap(&self.q_terms[qv][qi], j)).collect());
            let mut sc: Vec<(f32, usize)> = idx.iter().enumerate().map(|(k, &j)| (base[k] + sys.gamma * ov[k], j)).collect();
            sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            idx = sc.into_iter().map(|x| x.1).collect();
        }
        f.clear();
        idx
    }
}

// ------------------------------------------------------------------------------------------------
#[derive(Clone, Debug)]
struct Sess {
    pool: String,
    dstar: usize,
    topics: Vec<i32>,
    q1: usize,
    /// per step k>=1: (act, pat, form-rand)
    step: Vec<(usize, usize, f64)>,
}

#[derive(Clone, Copy, PartialEq)]
enum World {
    Default,
    CleanStep1,
    TypoStep1,
}

fn make_sessions(env: &Env, seed: u64, world: World) -> Vec<Sess> {
    let mut rng = Rng::new(seed.wrapping_mul(0x9E37) ^ 0x6D5);
    let pools: Vec<&String> = { let mut p: Vec<&String> = env.g.topics_of_pool.iter().filter(|(_, t)| t.len() >= 3).map(|(k, _)| k).collect(); p.sort(); p };
    let by_concept: Vec<Vec<usize>> = (0..env.c.nconc).map(|k| (0..env.n_orig).filter(|&i| env.c.queries[i].concept == k as i32).collect()).collect();
    let mut out = vec![];
    for _ in 0..N_SESS {
        let pool = pools[rng.below(pools.len())].clone();
        let mut tps = env.g.topics_of_pool[&pool].clone();
        tps.sort();
        for i in (1..tps.len()).rev() {
            tps.swap(i, rng.below(i + 1));
        }
        let l = 3 + rng.below(3);
        let topics: Vec<i32> = (0..l).map(|k| tps[k % tps.len()]).collect();
        let dstar = rng.below(8);
        let cands: Vec<usize> = by_concept[env.g.cid[&(topics[0], dstar as i32)]]
            .iter()
            .copied()
            .filter(|&i| match world {
                World::Default => true,
                World::CleanStep1 => env.c.queries[i].cls == "clean",
                World::TypoStep1 => env.c.queries[i].cls == "typo",
            })
            .collect();
        let q1 = cands[rng.below(cands.len())];
        let step = (1..l).map(|_| (rng.below(4), rng.below(2), rng.f64())).collect();
        out.push(Sess { pool, dstar, topics, q1, step });
    }
    out
}

#[derive(Clone, Debug, Default)]
struct Res {
    success: f64,
    step_acc: f64,
    step1: f64,
    later: f64,
    cond: f64,
    cond_n: f64,
    stuck: f64,
    p3: f64,
    r3: f64,
    repairs: f64,
    harmful: f64,
    helpful: f64,
}

/// pick the evidence order after the policy; returns (ranking, repaired?)
fn apply_policy(env: &Env, sys: &Sys, pol: Pol, qi: usize, rank: &[usize], pinned: Option<usize>) -> (Vec<usize>, bool) {
    if pol == Pol::None {
        return (rank.to_vec(), false);
    }
    let top = rank[0];
    let Some(pool) = env.doc_pool[top].clone() else { return (rank.to_vec(), false) };
    let stated = env.detect_ver(&pool, env.qtext(sys, qi), qi);
    let want = if pol == Pol::Pin { pinned.or(stated) } else { stated };
    let Some(wd) = want else { return (rank.to_vec(), false) };
    // strict verification: the top-1 document must POSITIVELY confirm the wanted detail (a document the gazetteer cannot read counts as unconfirmed);
    // lenient (--lenient): an unreadable document is accepted as is
    if env.doc_detail_ver[top] == Some(wd) || (env.lenient && env.doc_detail_ver[top].is_none()) {
        return (rank.to_vec(), false);
    }
    match rank.iter().position(|&d| env.doc_pool[d].as_deref() == Some(pool.as_str()) && env.doc_detail_ver[d] == Some(wd)) {
        Some(p) => {
            let mut r = rank.to_vec();
            let d = r.remove(p);
            r.insert(0, d);
            (r, true)
        }
        None => (rank.to_vec(), false),
    }
}

fn run_session(env: &Env, sys: &Sys, pol: Pol, indep: bool, s: &Sess) -> (Vec<bool>, Vec<(f64, f64)>, bool, [f64; 3], bool) {
    // returns per-step correct, per-step (p3, r3) for steps with a query, stuck flag, repair stats [repairs, harmful, helpful]
    let l = s.topics.len();
    let mut correct = vec![false; l];
    let mut pr: Vec<(f64, f64)> = vec![];
    let mut stuck = false;
    let mut rep = [0.0f64; 3];
    let mut qi = s.q1;
    let mut pinned: Option<usize> = None;
    for k in 0..l {
        let gold = env.g.cid[&(s.topics[k], s.dstar as i32)];
        let rank = env.rank(sys, qi);
        let (r2, repaired) = apply_policy(env, sys, pol, qi, &rank, if pol == Pol::Pin { pinned } else { None });
        let top_ok = env.c.docs[rank[0]].concept == gold as i32;
        let new_ok = env.c.docs[r2[0]].concept == gold as i32;
        if repaired {
            rep[0] += 1.0;
            if top_ok && !new_ok {
                rep[1] += 1.0;
            }
            if !top_ok && new_ok {
                rep[2] += 1.0;
            }
        }
        correct[k] = new_ok;
        let rel: Vec<usize> = env.c.docs_of[gold].clone();
        let hit3 = r2.iter().take(3).filter(|d| rel.contains(d)).count() as f64;
        pr.push((hit3 / 3.0, hit3 / rel.len() as f64));
        if k == 0 && pol == Pol::Pin {
            if let Some(pool) = env.doc_pool[r2[0]].clone() {
                pinned = env.detect_ver(&pool, env.qtext(sys, qi), qi);
            }
        }
        if k + 1 == l {
            break;
        }
        // write the next query from the evidence of this step
        let dhat = if indep {
            Some(s.dstar)
        } else if pol == Pol::Pin && pinned.is_some() {
            pinned
        } else {
            env.doc_pool[r2[0]].as_deref().filter(|p| *p == s.pool).and(env.doc_detail[r2[0]])
        };
        let Some(dh) = dhat else {
            stuck = true;
            break;
        };
        let (a, p, fr) = s.step[k];
        // surface form: one of the forms of the detail that occurs in the evidence doc (else any)
        let forms = &env.g.forms[&s.pool][dh];
        let doc_t = env.c.docs[r2[0]].text.to_lowercase();
        let present: Vec<usize> = (0..3).filter(|&f| count_word(&doc_t, &forms[f]) > 0).collect();
        let f = if present.is_empty() || indep { (fr * 3.0) as usize % 3 } else { present[(fr * present.len() as f64) as usize % present.len()] };
        qi = env.chain_ix[&[s.topics[k + 1] as usize, a, p, dh, f]];
    }
    (correct, pr, stuck, rep, true)
}

fn run_arm(env: &Env, sys: &Sys, pol: Pol, indep: bool, seed: u64, world: World) -> Res {
    let sessions = make_sessions(env, seed, world);
    let mut r = Res::default();
    let (mut n_steps, mut n_later) = (0.0, 0.0);
    let mut p3 = 0.0;
    let mut r3 = 0.0;
    let mut n_pr = 0.0;
    for s in &sessions {
        let (corr, pr, stuck, rep, _) = run_session(env, sys, pol, indep, s);
        let l = corr.len() as f64;
        if corr.iter().all(|&c| c) {
            r.success += 1.0;
        }
        r.step_acc += corr.iter().filter(|&&c| c).count() as f64 / l;
        n_steps += 1.0;
        if corr[0] {
            r.step1 += 1.0;
        }
        r.later += corr[1..].iter().filter(|&&c| c).count() as f64 / (l - 1.0);
        n_later += 1.0;
        for k in 1..corr.len() {
            if corr[k - 1] {
                r.cond_n += 1.0;
                if corr[k] {
                    r.cond += 1.0;
                }
            }
        }
        if stuck {
            r.stuck += 1.0;
        }
        for (p, q) in pr {
            p3 += p;
            r3 += q;
            n_pr += 1.0;
        }
        r.repairs += rep[0];
        r.harmful += rep[1];
        r.helpful += rep[2];
    }
    let n = sessions.len() as f64;
    r.success /= n;
    r.step_acc /= n_steps;
    r.step1 /= n;
    r.later /= n_later;
    r.cond = r.cond / r.cond_n.max(1.0);
    r.stuck /= n;
    r.p3 = p3 / n_pr;
    r.r3 = r3 / n_pr;
    r.repairs /= n;
    r.harmful /= n;
    r.helpful /= n;
    r
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tag = arg(&args, "--corpus", "C");
    let held = parse_range(&args, "--held", (31, 60));
    let sfx = format!("{}{}", if arg(&args, "--gk", "3") != "3" || arg(&args, "--gnoise", "0") != "0" { format!("_gk{}_noise{}", arg(&args, "--gk", "3"), arg(&args, "--gnoise", "0")) } else { String::new() }, if args.iter().any(|a| a == "--lenient") { "_lenient" } else { "" });
    let out_path = arg(&args, "--out", &format!("results/g9_multistep_{}_{}-{}{}.json", tag, held[0], held[held.len() - 1], sfx));
    let t0 = Instant::now();
    let mut env = build_env(&tag);
    let gk: usize = arg(&args, "--gk", "3").parse().unwrap();
    let gnoise: f64 = arg(&args, "--gnoise", "0").parse().unwrap();
    env.set_verifier(gk, gnoise);
    env.lenient = args.iter().any(|a| a == "--lenient");
    println!("verification mode: {}", if env.lenient { "lenient" } else { "strict" });
    println!("verifier gazetteer: {gk} of 3 surface forms per detail, detection noise {gnoise}");
    println!("corpus {tag}: env built in {:.1}s ({} queries incl. {} follow-ups)", t0.elapsed().as_secs_f64(), env.c.queries.len(), env.c.queries.len() - env.n_orig);
    let sys_filter: Vec<String> = arg(&args, "--sys", "").split(',').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect();
    {
        // gazetteer sanity: how well does the (full) gazetteer read the detail of a topical document?
        let det_of: HashMap<usize, usize> = env.g.cid.iter().map(|(&(_, d), &c)| (c, d as usize)).collect();
        let (mut n, mut found, mut ok) = (0, 0, 0);
        for (i, d) in env.c.docs.iter().enumerate() {
            if d.concept >= 0 {
                n += 1;
                if let Some(x) = env.doc_detail[i] {
                    found += 1;
                    if x == det_of[&(d.concept as usize)] {
                        ok += 1;
                    }
                }
            }
        }
        println!("gazetteer reads {found}/{n} topical docs ({ok} correct)");
    }
    let syss: Vec<Sys> = systems().into_iter().filter(|s| sys_filter.is_empty() || sys_filter.iter().any(|f| f == s.name)).collect();
    let wfilter: Vec<String> = arg(&args, "--worlds", "").split(',').filter(|x| !x.is_empty()).map(|x| x.to_string()).collect();
    let pols = [("none", Pol::None), ("local", Pol::Local), ("pin", Pol::Pin)];
    let worlds: Vec<(&str, World)> = [("default", World::Default), ("clean-step1", World::CleanStep1), ("typo-step1", World::TypoStep1)].into_iter().filter(|w| wfilter.is_empty() || wfilter.iter().any(|f| f == w.0)).collect();
    // arm = (system, policy, indep)
    let mut arms: Vec<(usize, usize, bool)> = vec![];
    for si in 0..syss.len() {
        for pi in 0..pols.len() {
            arms.push((si, pi, false));
        }
        arms.push((si, 0, true));
        arms.push((si, 1, true));
    }
    let aname = |a: &(usize, usize, bool)| format!("{}/{}{}", syss[a.0].name, pols[a.1].0, if a.2 { "/indep" } else { "" });
    let mut tests: Vec<Test> = vec![];
    let mut table = vec![];
    for (wn, w) in worlds.iter() {
        let res: Vec<Vec<Res>> = arms.iter().map(|a| held.par_iter().map(|&s| run_arm(&env, &syss[a.0], pols[a.1].1, a.2, s, *w)).collect()).collect();
        let m = |ai: usize, f: &dyn Fn(&Res) -> f64| res[ai].iter().map(f).sum::<f64>() / res[ai].len() as f64;
        println!("\n=== world {wn} (held {}-{}) ===", held[0], held[held.len() - 1]);
        println!("{:<26} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>6} {:>7} {:>6} {:>6}", "arm", "sess", "step", "step1", "later", "cond", "stuck", "P@3", "R@3", "repair", "harm", "help");
        let mut rows = vec![];
        for (ai, a) in arms.iter().enumerate() {
            println!("{:<26} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>6.3} {:>6.3} {:>6.3} {:>7.3} {:>6.3} {:>6.3}", aname(a), m(ai, &|r| r.success), m(ai, &|r| r.step_acc), m(ai, &|r| r.step1), m(ai, &|r| r.later), m(ai, &|r| r.cond), m(ai, &|r| r.stuck), m(ai, &|r| r.p3), m(ai, &|r| r.r3), m(ai, &|r| r.repairs), m(ai, &|r| r.harmful), m(ai, &|r| r.helpful));
            rows.push(json!({"arm": aname(a), "session_success": m(ai, &|r| r.success), "step_acc": m(ai, &|r| r.step_acc), "step1": m(ai, &|r| r.step1), "later": m(ai, &|r| r.later), "cond_next_given_prev": m(ai, &|r| r.cond), "stuck": m(ai, &|r| r.stuck), "p3": m(ai, &|r| r.p3), "r3": m(ai, &|r| r.r3), "repairs": m(ai, &|r| r.repairs), "harmful_repairs": m(ai, &|r| r.harmful), "helpful_repairs": m(ai, &|r| r.helpful)}));
        }
        table.push(json!({"world": wn, "rows": rows}));
        let col = |ai: usize, f: &dyn Fn(&Res) -> f64| -> Vec<f64> { res[ai].iter().map(f).collect() };
        let ix = |s: &str, p: usize, ind: bool| arms.iter().position(|a| syss[a.0].name == s && a.1 == p && a.2 == ind).unwrap();
        let succ = |r: &Res| r.success;
        // best fair baseline = best system under policy none by session success (embedding or bm25, no verification)
        let base = (0..syss.len()).fold(0, |b, j| if m(ix(syss[j].name, 0, false), &succ) > m(ix(syss[b].name, 0, false), &succ) { j } else { b });
        let bname = format!("{}/none", syss[base].name);
        for sy in syss.iter() {
            for (pn, pi) in [("local", 1usize), ("pin", 2usize)] {
                // policy effect, same system
                tests.push(mk_test("M1-policy-vs-none(same system)", wn, "session_success", &format!("{}/{}", sy.name, pn), &format!("{}/none", sy.name), &col(ix(sy.name, pi, false), &succ), &col(ix(sy.name, 0, false), &succ)));
                // vs the best fair baseline
                tests.push(mk_test("M2-vs-bestfair", wn, "session_success", &format!("{}/{}", sy.name, pn), &bname, &col(ix(sy.name, pi, false), &succ), &col(ix(syss[base].name, 0, false), &succ)));
            }
            tests.push(mk_test("M2-vs-bestfair", wn, "session_success", &format!("{}/none", sy.name), &bname, &col(ix(sy.name, 0, false), &succ), &col(ix(syss[base].name, 0, false), &succ)));
            tests.push(mk_test("M3-pin-vs-local", wn, "session_success", &format!("{}/pin", sy.name), &format!("{}/local", sy.name), &col(ix(sy.name, 2, false), &succ), &col(ix(sy.name, 1, false), &succ)));
            // propagation penalty: independent (gold detail restated) minus dependent, policy none / local
            tests.push(mk_test("M4-propagation(indep-dep)", wn, "step_acc", &format!("{}/none/indep", sy.name), &format!("{}/none", sy.name), &col(ix(sy.name, 0, true), &|r: &Res| r.step_acc), &col(ix(sy.name, 0, false), &|r: &Res| r.step_acc)));
        }
        println!("best fair baseline in {wn}: {bname} ({:.4})", m(ix(syss[base].name, 0, false), &succ));
        table.last_mut().unwrap()["best_fair"] = json!(bname);
    }
    holm_by_family(&mut tests);
    println!("\n=== Holm tests (only significant shown, plus the claim rows)");
    for t in &tests {
        let v = verdict(t.adj, t.d.diff.mean);
        if v != "ns" {
            println!("{:<32} {:<12} {:<22} vs {:<22} {:+.4} [{:+.4},{:+.4}] holm {:.4} {}", t.family, t.world, t.treat, t.vs, t.d.diff.mean, t.d.diff.lo, t.d.diff.hi, t.adj, v);
        }
    }
    for fam in ["M1-policy-vs-none(same system)", "M2-vs-bestfair", "M3-pin-vs-local", "M4-propagation(indep-dep)"] {
        let n = tests.iter().filter(|t| t.family == fam).count();
        let w = tests.iter().filter(|t| t.family == fam && verdict(t.adj, t.d.diff.mean) == "WIN").count();
        let l = tests.iter().filter(|t| t.family == fam && verdict(t.adj, t.d.diff.mean) == "LOSS").count();
        println!("{fam}: WIN {w} LOSS {l} of {n}");
    }
    let doc = json!({"corpus": tag, "held": [held[0], held[held.len() - 1]], "n_sessions": N_SESS, "table": table, "tests": tests.iter().map(test_json).collect::<Vec<_>>(), "runtime_s": t0.elapsed().as_secs_f64()});
    std::fs::write(&out_path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out_path} in {:.1}s", t0.elapsed().as_secs_f64());
}
