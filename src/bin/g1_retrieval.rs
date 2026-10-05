//! G1-retrieval: BGE-base vs MiniLM vs hash vs BM25 on the HARD g1 corpus (240 concepts, sibling near-duplicates, multi-sentence
//! docs, typo / terse / chatty queries) + model-size-independent retrieval-SYSTEM levers (hybrid BM25 fusion, sentence-chunk max-pool,
//! spell-correct / keyword rewrite, Rocchio PRF, token-overlap rerank, cross-model cascade).
//!
//! Protocol: levers are tuned ONLY on the tuning concepts (concept % 5 == 0; "seeds 1-8" = random query subsets of that split);
//! evaluated on the disjoint held-out concepts with seeds 31-60 (corpus A); `--corpus B --confirm` re-runs with the picks FROZEN from
//! A on a freshly generated corpus (different phrasings, shuffled surface forms, seeds 61-90). Each seed = a random subset of 300
//! queries of the split; paired 95% t-CI over seeds, Holm over the comparison family. NOTE: the CI is over resampled query subsets of a
//! finite synthetic pool, not over independent corpora.
//!
//! Usage: cargo run --release --offline --bin g1_retrieval -- --corpus A [--held 31-60]
//!        cargo run --release --offline --bin g1_retrieval -- --corpus B --confirm [--held 61-90]

use std::time::Instant;

use neural_mesh::router::Rng;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
use common::*;

fn cfg_to_json(c: &Cfg) -> Value {
    json!({"name": c.name, "model": c.model, "qv": c.qv as u8, "lam": c.lam, "prf": c.prf, "lexical": c.lexical, "alpha": c.alpha, "gamma": c.gamma,
           "cascade": c.cascade.as_ref().map(|(n, w)| json!([n, w])), "m": c.m})
}
fn cfg_from_json(v: &Value) -> Cfg {
    Cfg {
        name: v["name"].as_str().unwrap().into(),
        model: v["model"].as_str().map(|s| s.to_string()),
        qv: match v["qv"].as_u64().unwrap() { 0 => QV::Raw, 1 => QV::Spell, _ => QV::Kw },
        lam: v["lam"].as_f64().unwrap() as f32,
        prf: v["prf"].as_f64().unwrap() as f32,
        lexical: v["lexical"].as_bool().unwrap(),
        alpha: v["alpha"].as_f64().unwrap() as f32,
        gamma: v["gamma"].as_f64().unwrap() as f32,
        cascade: v["cascade"].as_array().map(|a| (a[0].as_str().unwrap().to_string(), a[1].as_f64().unwrap() as f32)),
        m: v["m"].as_u64().unwrap() as usize,
    }
}

fn mean_mrr(ctx: &Ctx, cfg: &Cfg, qis: &[usize]) -> f64 {
    let r = ctx.eval_all(cfg, qis);
    r.iter().map(|x| x.1.mrr).sum::<f64>() / r.len() as f64
}

/// candidate single-lever additions on top of `cur`
fn lever_candidates(cur: &Cfg, lever: &str, has_model: bool) -> Vec<Cfg> {
    let mut v = vec![];
    match lever {
        "chunk" if has_model => {
            for &l in &[0.7f32, 0.5, 0.3, 0.0] {
                let mut c = cur.clone();
                c.lam = l;
                v.push(c);
            }
        }
        "spell" => {
            let mut c = cur.clone();
            c.qv = QV::Spell;
            v.push(c);
        }
        "kw" => {
            let mut c = cur.clone();
            c.qv = QV::Kw;
            v.push(c);
        }
        "hybrid" if has_model => {
            for &a in &[0.8f32, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2] {
                let mut c = cur.clone();
                c.lexical = true;
                c.alpha = a;
                v.push(c);
            }
        }
        "prf" if has_model => {
            for &b in &PRF_GRID {
                let mut c = cur.clone();
                c.prf = b;
                v.push(c);
            }
        }
        "overlap" => {
            for &g in &[0.5f32, 1.0, 2.0] {
                let mut c = cur.clone();
                c.gamma = g;
                v.push(c);
            }
        }
        _ => {}
    }
    v
}

fn best_of(ctx: &Ctx, cands: Vec<Cfg>, qis: &[usize]) -> Option<(Cfg, f64)> {
    let mut best: Option<(Cfg, f64)> = None;
    for c in cands {
        let m = mean_mrr(ctx, &c, qis);
        if best.as_ref().is_none_or(|b| m > b.1) {
            best = Some((c, m));
        }
    }
    best
}

/// returns (single-lever systems keyed by lever, greedy ladder final cfg + steps)
fn tune_model(ctx: &Ctx, model: &str, tune: &[usize], log: &mut Vec<Value>) -> (Vec<Cfg>, Cfg) {
    let alone = Cfg::base(&format!("{model}-alone"), model);
    let base_mrr = mean_mrr(ctx, &alone, tune);
    log.push(json!({"model": model, "step": "alone", "tune_mrr": base_mrr}));
    let mut singles = vec![alone.clone()];
    for lever in ["chunk", "spell", "hybrid", "prf", "overlap"] {
        if let Some((mut c, m)) = best_of(ctx, lever_candidates(&alone, lever, true), tune) {
            c.name = format!("{model}+{lever}");
            log.push(json!({"model": model, "single_lever": lever, "tune_mrr": m, "gain": m - base_mrr, "cfg": c.desc()}));
            singles.push(c);
        }
    }
    // greedy forward ladder (accept a lever only if tune MRR improves by >= .003)
    let mut cur = alone.clone();
    let mut cur_m = base_mrr;
    let mut remaining = vec!["chunk", "spell", "hybrid", "prf", "overlap"];
    loop {
        let mut best: Option<(Cfg, f64, &str)> = None;
        for &lv in &remaining {
            if let Some((c, m)) = best_of(ctx, lever_candidates(&cur, lv, true), tune) {
                if best.as_ref().is_none_or(|b| m > b.1) {
                    best = Some((c, m, lv));
                }
            }
        }
        match best {
            Some((c, m, lv)) if m - cur_m >= 0.003 => {
                log.push(json!({"model": model, "ladder_add": lv, "tune_mrr": m, "gain": m - cur_m, "cfg": c.desc()}));
                cur = c;
                cur_m = m;
                remaining.retain(|x| *x != lv);
            }
            _ => break,
        }
    }
    cur.name = format!("{model}-final");
    (singles, cur)
}

struct SysEval {
    name: String,
    cfg: Cfg,
    per_q: Vec<(usize, QM)>,
}

fn seed_means(ev: &SysEval, sample: &[usize], pos: &std::collections::HashMap<usize, usize>) -> [f64; 8] {
    let mut acc = [0.0; 8];
    for &qi in sample {
        let a = ev.per_q[pos[&qi]].1.arr();
        for k in 0..8 {
            acc[k] += a[k];
        }
    }
    acc.iter_mut().for_each(|x| *x /= sample.len() as f64);
    acc
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tag = arg(&args, "--corpus", "A");
    let confirm = args.iter().any(|a| a == "--confirm");
    let held = parse_range(&args, "--held", if confirm { (61, 90) } else { (31, 60) });
    let tune_seeds = parse_range(&args, "--tune", (1, 8));
    let t_start = Instant::now();
    let models = ["hash", "mini", "bge", "bgei"];
    let ctx = Ctx::new(&tag, &models, true);
    println!("corpus {tag}: {} docs, {} queries, {} concepts; loaded+matrices in {:.1}s", ctx.c.docs.len(), ctx.c.queries.len(), ctx.c.nconc, t_start.elapsed().as_secs_f64());
    let all_q: Vec<usize> = (0..ctx.c.queries.len()).filter(|&i| ctx.c.queries[i].concept >= 0).collect();
    let tune_q: Vec<usize> = all_q.iter().copied().filter(|&i| is_tune_concept(ctx.c.queries[i].concept)).collect();
    let held_q: Vec<usize> = all_q.iter().copied().filter(|&i| !is_tune_concept(ctx.c.queries[i].concept)).collect();
    println!("tune queries {} (concept%5==0), held-out queries {}", tune_q.len(), held_q.len());

    // ---------------- systems ----------------
    let mut tune_log: Vec<Value> = vec![];
    let mut systems: Vec<Cfg> = vec![];
    let picks_path = "results/g1_retrieval_picks.json";
    if confirm {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(picks_path).expect("run corpus A first (results/g1_retrieval_picks.json)")).unwrap();
        systems = v["systems"].as_array().unwrap().iter().map(cfg_from_json).collect();
        println!("confirm run: {} frozen system configs loaded from {picks_path}", systems.len());
    } else {
        // lexical baselines
        let bm_raw = Cfg::bm25("bm25-raw");
        systems.push(bm_raw.clone());
        let mut bm_best = bm_raw.clone();
        let mut bm_m = mean_mrr(&ctx, &bm_raw, &tune_q);
        for qv in [QV::Spell, QV::Kw] {
            let mut c = bm_raw.clone();
            c.qv = qv;
            let m = mean_mrr(&ctx, &c, &tune_q);
            tune_log.push(json!({"bm25_qv": format!("{qv:?}"), "tune_mrr": m}));
            if m > bm_m {
                bm_m = m;
                bm_best = c;
            }
        }
        if bm_best.qv != QV::Raw {
            bm_best.name = "bm25-tuned".into();
            systems.push(bm_best);
        }
        let mut finals: Vec<Cfg> = vec![];
        for model in ["hash", "mini", "bge", "bgei"] {
            let (singles, fin) = tune_model(&ctx, model, &tune_q, &mut tune_log);
            systems.extend(singles);
            if fin.desc() != Cfg::base("x", model).desc() {
                systems.push(fin.clone());
            }
            finals.push(fin);
        }
        // cascade: mini-final top-20 reranked with BGE whole-doc cosine (z within candidates)
        let mini_final = finals[1].clone();
        let mut best_c: Option<(Cfg, f64)> = None;
        for &w in &[0.5f32, 1.0, 2.0] {
            let mut c = mini_final.clone();
            c.cascade = Some(("bge".into(), w));
            let m = mean_mrr(&ctx, &c, &tune_q);
            if best_c.as_ref().is_none_or(|b| m > b.1) {
                best_c = Some((c, m));
            }
        }
        let mut cc = best_c.unwrap().0;
        cc.name = "mini-final+bge-cascade".into();
        systems.push(cc);
        let v = json!({"systems": systems.iter().map(cfg_to_json).collect::<Vec<_>>(), "tune_log": tune_log});
        std::fs::write(picks_path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
        for l in &tune_log {
            println!("tune: {l}");
        }
    }
    // dedupe by name
    let mut seen = std::collections::HashSet::new();
    systems.retain(|c| seen.insert(c.name.clone()));

    // ---------------- evaluate ----------------
    let evals_held: Vec<SysEval> = systems.iter().map(|c| SysEval { name: c.name.clone(), cfg: c.clone(), per_q: ctx.eval_all(c, &held_q) }).collect();
    let evals_tune: Vec<SysEval> = if confirm { vec![] } else { systems.iter().map(|c| SysEval { name: c.name.clone(), cfg: c.clone(), per_q: ctx.eval_all(c, &tune_q) }).collect() };
    let pos_held: std::collections::HashMap<usize, usize> = evals_held[0].per_q.iter().enumerate().map(|(i, x)| (x.0, i)).collect();
    let pos_tune: std::collections::HashMap<usize, usize> = if confirm { Default::default() } else { evals_tune[0].per_q.iter().enumerate().map(|(i, x)| (x.0, i)).collect() };

    let sample_for = |pool: &[usize], seed: u64| -> Vec<usize> {
        let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(101));
        let mut p = pool.to_vec();
        for i in (1..p.len()).rev() {
            let j = r.below(i + 1);
            p.swap(i, j);
        }
        p.truncate(300);
        p
    };
    let per_seed = |evs: &[SysEval], pool: &[usize], pos: &std::collections::HashMap<usize, usize>, seeds: &[u64]| -> Vec<Vec<[f64; 8]>> {
        evs.iter().map(|e| seeds.iter().map(|&s| seed_means(e, &sample_for(pool, s), pos)).collect()).collect()
    };
    let held_ps = per_seed(&evals_held, &held_q, &pos_held, &held);
    let tune_ps = if confirm { vec![] } else { per_seed(&evals_tune, &tune_q, &pos_tune, &tune_seeds) };

    // ---------------- print table ----------------
    let label = if confirm { format!("confirm corpus B, seeds {}-{}", held[0], held[held.len() - 1]) } else { format!("held-out concepts, corpus A, seeds {}-{}", held[0], held[held.len() - 1]) };
    println!("\n=== {label}: mean over seeds (95% t-CI half-width) ===");
    println!("{:<26} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}   cfg", "system", "hit1", "mrr", "P@3", "R@3", "F1@3", "R@5", "sib@1");
    let mut table = vec![];
    for (i, e) in evals_held.iter().enumerate() {
        let col = |k: usize| held_ps[i].iter().map(|a| a[k]).collect::<Vec<f64>>();
        let s: Vec<_> = (0..8).map(|k| summarize(&col(k))).collect();
        println!(
            "{:<26} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4}   {}",
            e.name, s[0].mean, s[1].mean, s[2].mean, s[3].mean, s[4].mean, s[5].mean, s[6].mean, e.cfg.desc()
        );
        let mut o = serde_json::Map::new();
        o.insert("system".into(), json!(e.name));
        o.insert("cfg".into(), cfg_to_json(&e.cfg));
        for k in 0..8 {
            o.insert(QM_NAMES[k].into(), json!({"mean": s[k].mean, "ci_half": s[k].half_width}));
        }
        table.push(Value::Object(o));
    }
    if !confirm {
        println!("\n=== tuning split (concept%5==0), seeds {}-{} (means; levers were chosen on this split) ===", tune_seeds[0], tune_seeds[tune_seeds.len() - 1]);
        for (i, e) in evals_tune.iter().enumerate() {
            let m = |k: usize| tune_ps[i].iter().map(|a| a[k]).sum::<f64>() / tune_ps[i].len() as f64;
            println!("{:<26} hit1 {:.4} mrr {:.4} F1@3 {:.4}", e.name, m(0), m(1), m(4));
        }
    }

    // ---------------- paired comparisons (Holm family) ----------------
    let idx = |n: &str| evals_held.iter().position(|e| e.name == n);
    let mut pairs: Vec<(String, String)> = vec![];
    let mut add = |a: &str, b: &str| {
        if idx(a).is_some() && idx(b).is_some() {
            pairs.push((a.to_string(), b.to_string()));
        }
    };
    add("bge-alone", "mini-alone");
    add("bgei-alone", "mini-alone");
    add("bgei-alone", "bge-alone");
    add("mini-alone", "bm25-raw");
    add("bge-alone", "bm25-raw");
    add("mini-alone", "hash-alone");
    for m in ["mini", "bge"] {
        for lv in ["chunk", "spell", "hybrid", "prf", "overlap"] {
            add(&format!("{m}+{lv}"), &format!("{m}-alone"));
        }
        add(&format!("{m}-final"), &format!("{m}-alone"));
    }
    add("mini-final", "bge-alone");
    add("mini+hybrid", "bge-alone");
    add("bge-final", "mini-final");
    add("bgei-final", "mini-final");
    add("bge-final", "bge-alone");
    add("mini-final+bge-cascade", "mini-final");
    add("mini-final+bge-cascade", "bge-alone");
    add("mini-final", "bm25-raw");
    let mut cells = vec![];
    for (a, b) in &pairs {
        for &k in &[1usize, 0] {
            let (ia, ib) = (idx(a).unwrap(), idx(b).unwrap());
            let ca: Vec<f64> = held_ps[ia].iter().map(|x| x[k]).collect();
            let cb: Vec<f64> = held_ps[ib].iter().map(|x| x[k]).collect();
            let d = paired_diff(&ca, &cb);
            cells.push((a.clone(), b.clone(), QM_NAMES[k], d.diff.mean, d.diff.lo, d.diff.hi, d.wins, d.losses, paired_t_pvalue(&ca, &cb)));
        }
    }
    let p: Vec<f64> = cells.iter().map(|c| c.8).collect();
    let adj = holm_adjust(&p);
    println!("\n=== paired differences A-B ({label}); Holm over {} tests (* = Holm win, ! = Holm loss) ===", cells.len());
    let mut cj = vec![];
    for (c, a) in cells.iter().zip(&adj) {
        let mark = if *a < 0.05 { if c.3 > 0.0 { "*" } else { "!" } } else { " " };
        println!("{:<28} - {:<22} {:<6} {:+.4} [{:+.4},{:+.4}] wins {:>2}/{:<2} holm_p {:.4} {}", c.0, c.1, c.2, c.3, c.4, c.5, c.6, c.7, a, mark);
        cj.push(json!({"a": c.0, "b": c.1, "metric": c.2, "diff": c.3, "ci": [c.4, c.5], "wins": c.6, "losses": c.7, "p": c.8, "holm_p": a}));
    }

    // ---------------- per query class + sibling analysis (all held-out queries, no seeds) ----------------
    println!("\n=== per query class, held-out queries (hit1 / mrr) ===");
    let classes = ["clean", "typo", "terse", "chatty"];
    let mut class_rows = vec![];
    print!("{:<26}", "system");
    for c in classes {
        print!(" {:>13}", c);
    }
    println!();
    for e in &evals_held {
        print!("{:<26}", e.name);
        let mut row = serde_json::Map::new();
        for c in classes {
            let sel: Vec<&QM> = e.per_q.iter().filter(|x| ctx.c.queries[x.0].cls == c).map(|x| &x.1).collect();
            let n = sel.len().max(1) as f64;
            let (h, m) = (sel.iter().map(|q| q.hit1).sum::<f64>() / n, sel.iter().map(|q| q.mrr).sum::<f64>() / n);
            print!(" {:>6.3}/{:<6.3}", h, m);
            row.insert(c.into(), json!({"hit1": h, "mrr": m, "n": sel.len()}));
        }
        println!();
        class_rows.push(json!({"system": e.name, "classes": Value::Object(row)}));
    }

    // ---------------- cost ----------------
    let cost = cost_probe(&ctx, &held_q);

    let out = json!({
        "corpus": tag, "confirm": confirm, "held_seeds": held, "tune_seeds": tune_seeds, "n_tune_queries": tune_q.len(), "n_held_queries": held_q.len(),
        "n_docs": ctx.c.docs.len(), "n_concepts": ctx.c.nconc, "table_held": table, "paired": cj, "per_class": class_rows, "cost": cost, "tune_log": tune_log,
        "per_seed_mrr_held": evals_held.iter().enumerate().map(|(i, e)| json!({"system": e.name, "mrr": held_ps[i].iter().map(|a| a[1]).collect::<Vec<f64>>()})).collect::<Vec<_>>(),
    });
    let path = if confirm { "results/g1_retrieval_confirm.json".to_string() } else { "results/g1_retrieval.json".to_string() };
    std::fs::write(&path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nwrote {path}  (total {:.1}s)", t_start.elapsed().as_secs_f64());
}

/// per-query scoring cost (single thread, precomputed query vector; encoder time is separate) + memory
fn cost_probe(ctx: &Ctx, qis: &[usize]) -> Value {
    let nd = ctx.c.docs.len();
    let sample: Vec<usize> = qis.iter().copied().step_by((qis.len() / 200).max(1)).take(200).collect();
    let mut rows = vec![];
    for mn in ["hash", "mini", "bge"] {
        let m = &ctx.models[mn];
        let time = |f: &mut dyn FnMut(usize) -> f32| -> f64 {
            let mut sink = 0.0f32;
            for &q in sample.iter().take(10) {
                sink += f(q);
            }
            let t0 = Instant::now();
            for &q in &sample {
                sink += f(q);
            }
            std::hint::black_box(sink);
            t0.elapsed().as_secs_f64() * 1e6 / sample.len() as f64
        };
        let us_doc = time(&mut |q| {
            let qv = m.qvec(QV::Raw, q);
            (0..nd).map(|j| dot(qv, m.dvec(j))).fold(f32::MIN, f32::max)
        });
        let us_chunk = time(&mut |q| {
            let qv = m.qvec(QV::Raw, q);
            let mut best = vec![-1.0f32; nd];
            for (k, &d) in m.chunk_doc.iter().enumerate() {
                let s = dot(qv, &m.chunk[k * m.dim..(k + 1) * m.dim]);
                if s > best[d as usize] {
                    best[d as usize] = s;
                }
            }
            best.iter().cloned().fold(f32::MIN, f32::max)
        });
        let us_prf = time(&mut |q| {
            let qv = m.qvec(QV::Raw, q);
            let s: Vec<f32> = (0..nd).map(|j| dot(qv, m.dvec(j))).collect();
            let mut idx: Vec<usize> = (0..nd).collect();
            idx.select_nth_unstable_by(3, |&a, &b| s[b].partial_cmp(&s[a]).unwrap());
            let mut q2 = qv.to_vec();
            for &j in idx.iter().take(3) {
                for (x, y) in q2.iter_mut().zip(m.dvec(j)) {
                    *x += 0.2 * y;
                }
            }
            let q2 = unit(&q2);
            (0..nd).map(|j| dot(&q2, m.dvec(j))).fold(f32::MIN, f32::max)
        });
        rows.push(json!({"model": mn, "dim": m.dim, "us_doc_dense": us_doc, "us_chunk_maxpool": us_chunk, "us_prf_two_pass": us_prf,
            "doc_vec_mb": (nd * m.dim * 4) as f64 / 1048576.0, "chunk_vec_mb": (m.n_chunks() * m.dim * 4) as f64 / 1048576.0,
            "n_chunks": m.n_chunks(), "model_mb": m.model_bytes as f64 / 1048576.0, "encode_ms_single": m.single_ms}));
    }
    let us_bm = {
        let t0 = Instant::now();
        let mut s = 0.0f32;
        for &q in &sample {
            s += ctx.lex.bm25(&ctx.c.queries[q].text).iter().cloned().fold(f32::MIN, f32::max);
        }
        std::hint::black_box(s);
        t0.elapsed().as_secs_f64() * 1e6 / sample.len() as f64
    };
    let us_fuse = {
        let m = &ctx.models["mini"];
        let t0 = Instant::now();
        let mut s = 0.0f32;
        for &q in &sample {
            let qv = m.qvec(QV::Raw, q);
            let d: Vec<f32> = (0..nd).map(|j| dot(qv, m.dvec(j))).collect();
            let l = ctx.lex.bm25(&ctx.c.queries[q].text);
            let (zd, zl) = (zscore(&d), zscore(&l));
            let f: Vec<f32> = zd.iter().zip(&zl).map(|(a, b)| 0.5 * a + 0.5 * b).collect();
            let mut idx: Vec<usize> = (0..nd).collect();
            idx.select_nth_unstable_by(20, |&a, &b| f[b].partial_cmp(&f[a]).unwrap());
            let qt = &ctx.q_terms[0][q];
            s += idx.iter().take(20).map(|&j| ctx.lex.overlap(qt, j)).sum::<f32>();
        }
        std::hint::black_box(s);
        t0.elapsed().as_secs_f64() * 1e6 / sample.len() as f64
    };
    json!({"models": rows, "us_bm25": us_bm, "us_hybrid_mini_doc_bm25_overlap_rerank20": us_fuse, "bm25_index_mb": ctx.lex.index_bytes() as f64 / 1048576.0,
           "n_docs": nd, "note": "scoring cost in microseconds per query on one thread with the query vector already encoded; ONNX encode ms measured separately in py/g1_embed.py (single text, batch 1, 4 intra-op threads)"})
}
