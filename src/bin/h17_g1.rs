//! H17-g1 (R9-A): REAL-vs-HASH embedding validation on the SYNTHETIC templated g1 corpora A, B, C, D (same engine as g1_retrieval: results/g1_*).
//!
//! Models: `mini` = saved real all-MiniLM-L6-v2 vectors (dim 384), `hash` = neural_mesh HashEmbedding path as used by g1 (`hash_embed` over lowercase alphanumeric
//! words, dim 384, no idf, signed feature hashing).  Same BM25 for both.
//! Systems per model: dense raw, dense+spell, hybrid z-score (alpha tuned per model on corpus-A TUNE concepts (concept % 5 == 0), grid .1..1.0), BM25 raw/+spell.
//! Protocol (as g1): alpha picks tuned on A tune concepts only, then FROZEN; evaluation on the held-out concepts (concept % 5 != 0); each seed = a random subset of 300
//! held-out queries; A held seeds 31-60 (+A 61-90), B/C/D seeds 61-90 (frozen picks; B,C,D are other generated corpora).  Paired 95% t-CI over seeds, Holm per
//! (corpus, split) over the contrast family.  The CI is over resampled query subsets of a finite synthetic pool, not independent corpora.
//! Part 2 (ctx): injected-context frontier (fixed k / adaptive drop-from-top / byte budget, exact-duplicate suppression) per model: the h6 `ctx` experiment re-run
//! with the hash model on all of A-D (spell queries, 30% exact duplicate re-ingestion; per-concept units).
//! Usage: h17_g1 [--out results/h17_g1.json]

use std::collections::HashMap;
use std::time::Instant;

use neural_mesh::router::Rng;
use neural_mesh::stats::summarize;
use serde_json::{Value, json};

#[path = "g1_common/mod.rs"]
mod common;
#[path = "g16_common/mod.rs"]
mod g16;
use common::*;
use g16::{cmp, holm};

const MODELS: [&str; 2] = ["hash", "mini"];
const ALPHAS: [f32; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

fn mean_of(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

struct SysEval {
    name: String,
    per_q: Vec<(usize, QM)>,
}

fn sample_for(pool: &[usize], seed: u64) -> Vec<usize> {
    let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(101));
    let mut p = pool.to_vec();
    for i in (1..p.len()).rev() {
        let j = r.below(i + 1);
        p.swap(i, j);
    }
    p.truncate(300);
    p
}

/// per seed: [hit1, mrr, f13, mrr(typo), mrr(clean), p3, r3]
fn seed_vals(ev: &SysEval, c: &Corpus, pos: &HashMap<usize, usize>, sample: &[usize]) -> [f64; 7] {
    let mut acc = [0.0f64; 7];
    let (mut nt, mut nc) = (0.0, 0.0);
    for &qi in sample {
        let m = &ev.per_q[pos[&qi]].1;
        acc[0] += m.hit1;
        acc[1] += m.mrr;
        acc[2] += m.f13;
        acc[5] += m.p3;
        acc[6] += m.r3;
        match c.queries[qi].cls.as_str() {
            "typo" => {
                acc[3] += m.mrr;
                nt += 1.0;
            }
            "clean" => {
                acc[4] += m.mrr;
                nc += 1.0;
            }
            _ => {}
        }
    }
    let n = sample.len() as f64;
    [acc[0] / n, acc[1] / n, acc[2] / n, if nt > 0.0 { acc[3] / nt } else { f64::NAN }, if nc > 0.0 { acc[4] / nc } else { f64::NAN }, acc[5] / n, acc[6] / n]
}
const VN: [&str; 7] = ["hit1", "mrr", "f13", "mrr_typo", "mrr_clean", "p3", "r3"];

fn cfg_sys(model: &str, kind: &str, alpha_raw: f32, alpha_sp: f32) -> Cfg {
    match kind {
        "dense_raw" => Cfg::base(&format!("{model}:dense_raw"), model),
        "dense_s" => {
            let mut c = Cfg::base(&format!("{model}:dense_s"), model);
            c.qv = QV::Spell;
            c
        }
        "hyb_raw" => {
            let mut c = Cfg::base(&format!("{model}:hyb_raw"), model);
            c.lexical = true;
            c.alpha = alpha_raw;
            c
        }
        "hyb_s" => {
            let mut c = Cfg::base(&format!("{model}:hyb_s"), model);
            c.qv = QV::Spell;
            c.lexical = true;
            c.alpha = alpha_sp;
            c
        }
        _ => unreachable!(),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = arg(&args, "--out", "results/h17_g1.json");
    let t0 = Instant::now();
    let mut log = String::new();
    let mut result = json!({});
    // ---------------- tune alphas on corpus A tune concepts (per model, raw and spell flavour)
    let ctx_a = Ctx::new("A", &MODELS, false);
    let all_q: Vec<usize> = (0..ctx_a.c.queries.len()).filter(|&i| ctx_a.c.queries[i].concept >= 0).collect();
    let tune_q: Vec<usize> = all_q.iter().copied().filter(|&i| is_tune_concept(ctx_a.c.queries[i].concept)).collect();
    let mut picks: HashMap<String, (f32, f32)> = HashMap::new();
    let mut tune_json = json!({});
    for m in MODELS {
        let mut best = [(0.0f32, f64::MIN); 2];
        let mut grids = [vec![], vec![]];
        for (fl, kind) in ["hyb_raw", "hyb_s"].iter().enumerate() {
            for &a in &ALPHAS {
                let cfg = cfg_sys(m, kind, a, a);
                let r = ctx_a.eval_all(&cfg, &tune_q);
                let mrr = mean_of(&r.iter().map(|x| x.1.mrr).collect::<Vec<_>>());
                grids[fl].push(mrr);
                if mrr > best[fl].1 {
                    best[fl] = (a, mrr);
                }
            }
        }
        log.push_str(&format!("TUNE {m}: alpha raw {} (tune MRR {:.4}) spell {} ({:.4}); spell grid {:?}\n", best[0].0, best[0].1, best[1].0, best[1].1, grids[1].iter().map(|x| (x * 1e4).round() / 1e4).collect::<Vec<_>>()));
        tune_json[m] = json!({"alpha_raw": best[0].0, "alpha_spell": best[1].0, "grid": ALPHAS, "grid_raw_mrr": grids[0], "grid_spell_mrr": grids[1]});
        picks.insert(m.to_string(), (best[0].0, best[1].0));
    }
    result["tune_A_tune_concepts"] = tune_json;
    drop(ctx_a);

    // ---------------- evaluation
    let mut ev_json = json!({});
    let splits: [(&str, &str, (u64, u64)); 5] = [("A", "A_held_31_60", (31, 60)), ("A", "A_held_61_90", (61, 90)), ("B", "B_61_90", (61, 90)), ("C", "C_61_90", (61, 90)), ("D", "D_61_90", (61, 90))];
    let mut ctx_cache: HashMap<String, Ctx> = HashMap::new();
    for (tag, label, (lo, hi)) in splits {
        if !ctx_cache.contains_key(tag) {
            ctx_cache.insert(tag.to_string(), Ctx::new(tag, &MODELS, false));
        }
        let ctx = &ctx_cache[tag];
        let held_q: Vec<usize> = (0..ctx.c.queries.len()).filter(|&i| ctx.c.queries[i].concept >= 0 && !is_tune_concept(ctx.c.queries[i].concept)).collect();
        let mut cfgs: Vec<Cfg> = vec![Cfg::bm25("bm25_raw")];
        let mut bs = Cfg::bm25("bm25_s");
        bs.qv = QV::Spell;
        cfgs.push(bs);
        for m in MODELS {
            let (ar, asp) = picks[m];
            for kind in ["dense_raw", "dense_s", "hyb_raw", "hyb_s"] {
                cfgs.push(cfg_sys(m, kind, ar, asp));
            }
        }
        let evs: Vec<SysEval> = cfgs.iter().map(|c| SysEval { name: c.name.clone(), per_q: ctx.eval_all(c, &held_q) }).collect();
        let pos: HashMap<usize, usize> = evs[0].per_q.iter().enumerate().map(|(i, x)| (x.0, i)).collect();
        let seeds: Vec<u64> = (lo..=hi).collect();
        let samples: Vec<Vec<usize>> = seeds.iter().map(|&s| sample_for(&held_q, s)).collect();
        let vals: HashMap<String, Vec<[f64; 7]>> = evs.iter().map(|e| (e.name.clone(), samples.iter().map(|sm| seed_vals(e, &ctx.c, &pos, sm)).collect())).collect();
        let series = |name: &str, vi: usize| -> Vec<f64> { vals[name].iter().map(|v| v[vi]).collect() };
        log.push_str(&format!("\n=== corpus {tag} split {label} (seeds {lo}-{hi}; {} held-out queries; picks hash alpha raw/spell {:?}, mini {:?}) ===\n", held_q.len(), picks["hash"], picks["mini"]));
        log.push_str("  system            hit1   MRR   F1@3  P@3   R@3   MRR(typo) MRR(clean)\n");
        let mut levels = json!({});
        for e in &evs {
            let mut row = json!({});
            for (vi, vn) in VN.iter().enumerate() {
                row[*vn] = json!(mean_of(&series(&e.name, vi)));
            }
            log.push_str(&format!("  {:16} {:.3} {:.3} {:.3} {:.3} {:.3} {:.3}    {:.3}\n", e.name, mean_of(&series(&e.name, 0)), mean_of(&series(&e.name, 1)), mean_of(&series(&e.name, 2)), mean_of(&series(&e.name, 5)), mean_of(&series(&e.name, 6)), mean_of(&series(&e.name, 3)), mean_of(&series(&e.name, 4))));
            levels[e.name.clone()] = row;
        }
        // contrasts
        let mut pairs: Vec<(String, String)> = vec![];
        for m in MODELS {
            pairs.push((format!("{m}:dense_s"), format!("{m}:dense_raw"))); // spell gain (dense)
            pairs.push((format!("{m}:hyb_s"), format!("{m}:dense_s"))); // hybrid > dense (+spell)
            pairs.push((format!("{m}:hyb_raw"), format!("{m}:dense_raw"))); // hybrid > dense (raw)
            pairs.push(("bm25_s".to_string(), format!("{m}:dense_s"))); // BM25 > dense
            pairs.push((format!("{m}:hyb_s"), "bm25_s".to_string())); // hybrid > BM25 alone
        }
        pairs.push(("mini:dense_s".to_string(), "hash:dense_s".to_string())); // encoder gap (dense+spell)
        pairs.push(("mini:hyb_s".to_string(), "hash:hyb_s".to_string())); // encoder gap (hybrid)
        let mets = [(1usize, "mrr"), (0, "hit1"), (2, "f13"), (3, "mrr_typo")];
        let mut rows = vec![];
        for (a, b) in &pairs {
            for &(vi, mn) in &mets {
                let (x, y) = (series(a, vi), series(b, vi));
                rows.push((a.clone(), b.clone(), mn, cmp(&x, &y, 0.0)));
            }
        }
        let adj = holm(&rows.iter().map(|r| r.3.p).collect::<Vec<_>>());
        let mut cj = vec![];
        for (i, r) in rows.iter().enumerate() {
            let sig = if adj[i] < 0.05 { if r.3.diff > 0.0 { "WIN" } else { "LOSS" } } else { "ns" };
            cj.push(json!({"a": r.0, "b": r.1, "metric": r.2, "diff": r.3.diff, "lo": r.3.lo, "hi": r.3.hi, "p": r.3.p, "holm": adj[i], "sig": sig}));
            if r.2 == "mrr" || r.2 == "mrr_typo" {
                log.push_str(&format!("  {:14} - {:14} {:9}: {:+.4} [{:+.4},{:+.4}] holm {:.3} {}\n", r.0, r.1, r.2, r.3.diff, r.3.lo, r.3.hi, adj[i], sig));
            }
        }
        ev_json[label] = json!({"corpus": tag, "n_held_queries": held_q.len(), "levels": levels, "contrasts": cj});
    }
    result["eval"] = ev_json;

    // ---------------- part 2: context frontier (fixed k / adaptive drop / byte budget) x dedup, per model on A-D
    let mut ctx_json = vec![];
    for tag in ["A", "B", "C", "D"] {
        let corpus = load_corpus(tag);
        for m in MODELS {
            let mdl = load_model(&corpus, m);
            let d = mdl.dim;
            let nq = corpus.queries.len();
            let concept: Vec<i32> = corpus.queries.iter().map(|q| q.concept).collect();
            let nd = corpus.docs.len();
            let dchars: Vec<usize> = corpus.docs.iter().map(|x| x.text.len()).collect();
            let mut rng = Rng::new(0x6C7_0001);
            let mut owner: Vec<usize> = (0..nd).collect();
            for i in 0..nd {
                if rng.f64() < 0.3 {
                    owner.push(i);
                }
            }
            let ns = owner.len();
            let store: Vec<f32> = owner.iter().flat_map(|&o| mdl.dvec(o).to_vec()).collect();
            let is_dup: Vec<bool> = (0..ns).map(|j| j >= nd).collect();
            let sc: Vec<f32> = (0..nq).flat_map(|i| (0..ns).map(move |j| (i, j))).map(|(i, j)| dot(mdl.qvec(QV::Spell, i), &store[j * d..(j + 1) * d])).collect();
            let pols: Vec<(String, u8, f32, usize)> = {
                let mut v: Vec<(String, u8, f32, usize)> = (1..=6).map(|k| (format!("fixed k={k}"), 0u8, 0.0f32, k)).collect();
                for dd in [0.02f32, 0.05, 0.1, 0.2] {
                    v.push((format!("adaptive drop {dd:.2}"), 1, dd, 6));
                }
                for b in [150usize, 250, 400, 600, 900, 1400] {
                    v.push((format!("byte budget {b}"), 2, 0.0, b));
                }
                v
            };
            let qc: Vec<i32> = concept.iter().cloned().filter(|&c| c >= 0).collect();
            let nconc = corpus.nconc;
            let agg = |vals: &[f64]| -> Vec<f64> {
                let (mut s, mut n) = (vec![0.0; nconc], vec![0.0; nconc]);
                for (v, &c) in vals.iter().zip(&qc) {
                    s[c as usize] += v;
                    n[c as usize] += 1.0;
                }
                (0..nconc).map(|c| if n[c] > 0.0 { s[c] / n[c] } else { 0.0 }).collect()
            };
            let mut table: Vec<(String, bool, Vec<f64>, Vec<f64>, f64, Vec<f64>)> = vec![]; // (policy, dedup, F1 per concept, hit_any per concept, chars, P per concept)
            for dedup in [false, true] {
                // rank lists once
                let ranks: Vec<Vec<(f32, usize)>> = (0..nq)
                    .map(|i| {
                        if concept[i] < 0 {
                            return vec![];
                        }
                        let mut rank: Vec<(f32, usize)> = (0..ns).filter(|&j| !(dedup && is_dup[j])).map(|j| (sc[i * ns + j], j)).collect();
                        rank.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
                        rank.truncate(12);
                        rank
                    })
                    .collect();
                for (pn, kind, dd, kk) in &pols {
                    let (mut f1v, mut anyv, mut chv, mut pv) = (vec![], vec![], vec![], vec![]);
                    for i in 0..nq {
                        if concept[i] < 0 {
                            continue;
                        }
                        let rank = &ranks[i];
                        let chars = |j: usize| dchars[owner[j]];
                        let sel: Vec<usize> = match kind {
                            0 => rank.iter().take(*kk).map(|x| x.1).collect(),
                            1 => {
                                let top = rank[0].0;
                                rank.iter().take(*kk).filter(|x| x.0 >= top - dd).map(|x| x.1).collect()
                            }
                            _ => {
                                let mut o = vec![];
                                let mut used = 0;
                                for x in rank.iter().take(8) {
                                    let c = chars(x.1);
                                    if !o.is_empty() && used + c > *kk {
                                        break;
                                    }
                                    used += c;
                                    o.push(x.1);
                                }
                                o
                            }
                        };
                        let rel = |j: usize| corpus.docs[owner[j]].concept == concept[i];
                        let hits = sel.iter().filter(|&&j| rel(j)).count() as f64;
                        let dh: std::collections::HashSet<usize> = sel.iter().filter(|&&j| rel(j)).map(|&j| owner[j]).collect();
                        let pr = hits / sel.len() as f64;
                        let r = dh.len() as f64 / 2.0;
                        f1v.push(if pr + r > 0.0 { 2.0 * pr * r / (pr + r) } else { 0.0 });
                        anyv.push(if hits > 0.0 { 1.0 } else { 0.0 });
                        pv.push(pr);
                        chv.push(sel.iter().map(|&j| chars(j)).sum::<usize>() as f64);
                    }
                    table.push((pn.clone(), dedup, agg(&f1v), agg(&anyv), summarize(&chv).mean, agg(&pv)));
                }
            }
            // knee test inside the dedup store: best-F1 policy = reference; TOST +-0.01 per concept
            for dedup in [false, true] {
                let rows: Vec<&(String, bool, Vec<f64>, Vec<f64>, f64, Vec<f64>)> = table.iter().filter(|r| r.1 == dedup).collect();
                let best = (0..rows.len()).max_by(|&a, &b| summarize(&rows[a].2).mean.partial_cmp(&summarize(&rows[b].2).mean).unwrap()).unwrap();
                for (i, r) in rows.iter().enumerate() {
                    let c = cmp(&r.2, &rows[best].2, 0.01);
                    ctx_json.push(json!({"corpus": tag, "model": m, "dedup": dedup, "policy": r.0, "chars": r.4, "F1": summarize(&r.2).mean, "hit_any": summarize(&r.3).mean, "P": summarize(&r.5).mean, "best_policy": rows[best].0, "diff_vs_best_F1": c.diff, "ci": [c.lo, c.hi], "tost_p_vs_best": c.tost_p, "equiv_001": c.tost_p < 0.05}));
                    if tag == "A" || tag == "C" {
                        log.push_str(&format!("ctx {tag} {m:4} {:8} {:22} chars {:7.1} F1 {:.4} hit_any {:.4} P {:.4} | vs best {} {:+.4} [{:+.4},{:+.4}] TOSTp {:.1e}\n", if dedup { "dedup" } else { "no-dedup" }, r.0, r.4, summarize(&r.2).mean, summarize(&r.3).mean, summarize(&r.5).mean, rows[best].0, c.diff, c.lo, c.hi, c.tost_p));
                    }
                }
            }
            // dedup effect: fixed k=1..4 dedup minus no-dedup (paired per concept)
            for k in 1..=4usize {
                let name = format!("fixed k={k}");
                let a = table.iter().find(|r| r.0 == name && r.1).unwrap();
                let b = table.iter().find(|r| r.0 == name && !r.1).unwrap();
                let c = cmp(&a.2, &b.2, 0.0);
                ctx_json.push(json!({"corpus": tag, "model": m, "dedup_effect_policy": name, "dF1": c.diff, "ci": [c.lo, c.hi], "p": c.p, "chars_dedup": a.4, "chars_nodedup": b.4}));
            }
        }
    }
    result["ctx"] = Value::Array(ctx_json);
    result["wall_s"] = json!(t0.elapsed().as_secs_f64());
    std::fs::write(&out, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    std::fs::write(out.replace(".json", ".log"), &log).unwrap();
    println!("{log}");
    println!("wrote {out} in {:.1}s", t0.elapsed().as_secs_f64());
}
