//! K1: knowledge-store GROWTH + COMPACTION + FRESHNESS on REAL-text worlds A / B (saved h10/h13 fixtures; world B is never rebuilt from live files).
//!
//!   k1_growth check  --world A|B                         (self-tests: quantisation == flatq, stream sanity, no-compaction numbers on 3 seeds)
//!   k1_growth tune   --world A|B                         (seeds 1-8: picks the free hyper-parameters, writes results/k1_picks_W.json)
//!   k1_growth held   --world A|B                         (seeds 31-60, frozen picks)
//!   k1_growth confirm --world A|B                        (seeds 61-90, frozen picks)
//!
//! Honest framing: this is the knowledge ACCESS/MEMORY layer (what we keep, how big it is, whether the freshest version wins); it adds no knowledge to the
//! frozen model. All numbers SYNTHETIC-until-validated: real paragraphs/queries/vectors, but synthetic revisions/duplicates/noise/arrival/popularity/outcomes.
//! Quality = hit@1 / MRR@10 of the retrieval (z-fusion .7 dense + .3 BM25, h13 frozen) on queries whose source section has fully arrived, relevance =
//! the CURRENT version of the source paragraph (any of its exact/near copies), a superseded version never counts as relevant.
//! Staleness = share of top-1 results whose content is a superseded version of its structural key. Closed loop = h13/g16 engine fed by the compacted
//! store's top-5 (stale version of the right key = irrelevant, or half credit in the declared sensitivity).

#[path = "k1_common/mod.rs"]
mod common;

use std::collections::BTreeMap;
use std::time::Instant;

use common::*;
use rayon::prelude::*;
use serde_json::{json, Value};

type M = BTreeMap<String, f64>;

struct SeedRes {
    arms: Vec<M>, // per arm: flattened metrics
}

fn flat(out: &ArmOut, lw: &[WSpec], envs: &[Env], lcfg: &[(String, LoopCfg)], seed: u64) -> M {
    let mut m = M::new();
    for (ci, c) in out.cks.iter().enumerate() {
        m.insert(format!("c{ci}.rows"), c.rows);
        m.insert(format!("c{ci}.bytes"), c.bytes);
        m.insert(format!("c{ci}.b_dense"), c.b_dense);
        m.insert(format!("c{ci}.b_sparse"), c.b_sparse);
        m.insert(format!("c{ci}.b_meta"), c.b_meta);
        for (k, v) in &c.m {
            m.insert(format!("c{ci}.{k}"), *v);
        }
    }
    m.insert("merged".into(), out.merged as f64);
    m.insert("stale_dropped".into(), out.stale_dropped as f64);
    for (wi, ws) in lw.iter().enumerate() {
        for (name, cfg) in lcfg {
            let r = run_loop(ws, &envs[wi], &out.tasks, cfg, seed, None);
            m.insert(format!("loop.{}.{}.h2000", ws.name, name), r.h2000);
            m.insert(format!("loop.{}.{}.inj", ws.name, name), r.inj_prec);
            m.insert(format!("loop.{}.{}.stale", ws.name, name), r.stale_inj);
            m.insert(format!("loop.{}.{}.vote", ws.name, name), r.vote_frac);
        }
    }
    m
}

fn oracle_lists(w: &WorldK, seed: u64) -> Vec<TaskRet> {
    let mut r = neural_mesh::router::Rng::new(seed.wrapping_mul(2_654_435_761) ^ 0x7A5C);
    (0..NT)
        .map(|_| {
            let q = r.below(w.qs.len());
            let mut t = TaskRet { n: 1, c: [Cand::default(); 5] };
            t.c[0] = Cand { rel: true, stale_rel: false, stale: false, cl: w.cl12[w.qs[q].rel[0] as usize], cos: 1.0 };
            t
        })
        .collect()
}

fn run_seed(w: &WorldK, arms: &[Arm], seed: u64, lw: &[WSpec], lcfg: &[(String, LoopCfg)]) -> SeedRes {
    let s = make_stream(w, seed);
    let cks: Vec<CkInfo> = CKPT_EP.iter().enumerate().map(|(i, &e)| ck_info(w, &s, s.ep_end[e], seed, i)).collect();
    let usage: Vec<Vec<usize>> = (0..NEP).map(|e| usage_queries(w, &s, s.ep_end[e + 1], seed, e)).collect();
    let envs: Vec<Env> = lw.iter().map(|ws| make_env(w, ws, seed)).collect();
    let mut res: Vec<M> = vec![];
    for arm in arms {
        let out = run_arm(w, &s, &cks, arm, seed, &usage);
        res.push(flat(&out, lw, &envs, lcfg, seed));
    }
    // extra pseudo-arms: no-retrieval front, and the retrieval ORACLE (a structurally relevant current paragraph is always the top-1)
    let mut fr = M::new();
    let mut orc = M::new();
    let ol = oracle_lists(w, seed);
    for (wi, ws) in lw.iter().enumerate() {
        let r = run_loop(ws, &envs[wi], &ol, &LoopCfg { gate: Gate::Safe, thr: 0.0, min_n: 2, k: 1, vote: false, stale_credit: 0.0 }, seed, None);
        fr.insert(format!("loop.{}.front.h2000", ws.name), r.h2000);
        for (name, cfg) in lcfg {
            let r = run_loop(ws, &envs[wi], &ol, cfg, seed, None);
            orc.insert(format!("loop.{}.{}.h2000", ws.name, name), r.h2000);
        }
    }
    res.push(fr);
    res.push(orc);
    res.push({
        let mut m = M::new();
        m.insert("stream_len".into(), s.recs.len() as f64);
        m.insert("bytes_none".into(), s.bytes_none);
        for (ci, c) in cks.iter().enumerate() {
            m.insert(format!("c{ci}.elig"), c.elig.len() as f64);
        }
        m
    });
    SeedRes { arms: res }
}

fn loop_cfgs() -> Vec<(String, LoopCfg)> {
    let j: Value = serde_json::from_slice(&std::fs::read("results/h13_route_picks.json").expect("h13_route_picks.json")).unwrap();
    let get = |name: &str| -> (f32, usize, usize) {
        for p in j["picks"].as_array().unwrap() {
            if p["arm"].as_str().unwrap() == name {
                return (p["thr"].as_f64().unwrap() as f32, p["min_n"].as_u64().unwrap() as usize, p["k"].as_u64().unwrap() as usize);
            }
        }
        panic!("pick {name}")
    };
    let (ts, ns, ks) = get("safe|hybz_s");
    let (tg, ng, kg) = get("g1|hybz_s");
    vec![
        ("safe".into(), LoopCfg { gate: Gate::Safe, thr: ts, min_n: ns, k: ks, vote: true, stale_credit: 0.0 }),
        ("g1".into(), LoopCfg { gate: Gate::G1, thr: tg, min_n: ng, k: kg, vote: true, stale_credit: 0.0 }),
        ("safe_s50".into(), LoopCfg { gate: Gate::Safe, thr: ts, min_n: ns, k: ks, vote: true, stale_credit: 0.5 }),
    ]
}

fn collect(w: &WorldK, arms: &[Arm], sds: &[u64], lw: &[WSpec], lcfg: &[(String, LoopCfg)]) -> Vec<SeedRes> {
    let t0 = Instant::now();
    let out: Vec<SeedRes> = sds
        .par_iter()
        .map(|&s| {
            let r = run_seed(w, arms, s, lw, lcfg);
            eprintln!("  seed {s} done ({:.0}s)", t0.elapsed().as_secs_f64());
            r
        })
        .collect();
    out
}

fn col(res: &[SeedRes], ai: usize, key: &str) -> Vec<f64> {
    res.iter().map(|r| *r.arms[ai].get(key).unwrap_or(&f64::NAN)).collect()
}

// ------------------------------------------------------------------------------------------------ picks

fn tune_picks(arms: &[Arm], res: &[SeedRes]) -> (BTreeMap<String, String>, String) {
    let idx = |n: &str| arms.iter().position(|a| a.name == n).unwrap();
    let none = idx("none");
    let (h0, m0) = (mean(&col(res, none, "c3.u.hit1")), mean(&col(res, none, "c3.u.mrr")));
    let mut picks = BTreeMap::new();
    let mut log = format!("TUNE seeds {}: none c3 uniform hit1 {:.4} mrr {:.4}\n", res.len(), h0, m0);
    for grp in ["cosfirst", "cosnew"] {
        let mut cands: Vec<(f32, String)> = arms.iter().filter(|a| a.group == grp).map(|a| if let Step::Cos { theta, .. } = a.steps[0] { (theta, a.name.clone()) } else { unreachable!() }).collect();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut chosen = cands.last().unwrap().1.clone();
        for (th, nm) in &cands {
            let ai = idx(nm);
            let (h, m) = (mean(&col(res, ai, "c3.u.hit1")), mean(&col(res, ai, "c3.u.mrr")));
            let by = mean(&col(res, ai, "c3.bytes")) / mean(&col(res, none, "c3.bytes"));
            log.push_str(&format!("  {nm:12} theta {th:.2} dhit1 {:+.4} dmrr {:+.4} bytes {:.3}\n", h - h0, m - m0, by));
        }
        for (_, nm) in &cands {
            let ai = idx(nm);
            let (h, m) = (mean(&col(res, ai, "c3.u.hit1")), mean(&col(res, ai, "c3.u.mrr")));
            if h - h0 >= -0.005 && m - m0 >= -0.005 {
                chosen = nm.clone();
                break;
            }
        }
        log.push_str(&format!(" -> pick {grp}: {chosen} (most aggressive theta with dhit1,dmrr >= -0.005, else .98)\n"));
        picks.insert(grp.to_string(), chosen);
    }
    for (grp, pre) in [("supsoft", "supsoft"), ("rec", "rec")] {
        let mut best = (f64::MIN, String::new());
        for a in arms.iter().filter(|a| a.group == grp) {
            let ai = idx(&a.name);
            let v = mean(&col(res, ai, "c3.u.hit1")) + mean(&col(res, ai, "c3.u.mrr"));
            log.push_str(&format!("  {:12} hit1+mrr {:.4}\n", a.name, v));
            if v > best.0 {
                best = (v, a.name.clone());
            }
        }
        log.push_str(&format!(" -> pick {pre}: {} (argmax hit1+mrr)\n", best.1));
        picks.insert(grp.to_string(), best.1);
    }
    (picks, log)
}

fn frozen_list(picks: &BTreeMap<String, String>) -> Vec<String> {
    let mut v: Vec<String> = ["exact", "supdel", "exact+supdel", "fifo50", "fifo25", "lru50", "lru25", "supdel+fifo25", "summary", "supdel+summary", "tier-f16", "tier-int8", "tier.2/.3", "tier.1/.2", "exact+supdel+tier.2/.3", "exact+supdel+cosfirst0.95"].iter().map(|s| s.to_string()).collect();
    for k in ["cosfirst", "cosnew", "supsoft", "rec"] {
        v.push(picks[k].clone());
    }
    v.extend(["val50", "val25", "supdel+val25"].iter().map(|s| s.to_string()));
    v.sort();
    v.dedup();
    v
}

// ------------------------------------------------------------------------------------------------ report

const EQ: f64 = 0.01;

fn report(label: &str, arms: &[Arm], res: &[SeedRes], frozen: &[String], lw: &[WSpec], lcfg: &[(String, LoopCfg)], log: &mut String) -> Value {
    let na = arms.len();
    let none = arms.iter().position(|a| a.name == "none").unwrap();
    let front = na;
    let orc = na + 1;
    let ext = na + 2;
    let nseeds = res.len();
    log.push_str(&format!("\n=== {label} ({nseeds} seeds) ===\n"));
    log.push_str(&format!("stream length mean {:.0}; unbounded-store bytes mean {:.2} MB; eligible queries at c3 mean {:.0}\n", mean(&col(res, ext, "stream_len")), mean(&col(res, ext, "bytes_none")) / 1e6, mean(&col(res, ext, "c3.elig"))));
    // main quality family: frozen arms x {hit1, mrr} at the final checkpoint (uniform eval), TOST +-0.01 vs none, Holm
    let fam_keys = ["c3.u.hit1", "c3.u.mrr"];
    let mut cells: Vec<(usize, &str, Cmp)> = vec![];
    for (ai, a) in arms.iter().enumerate() {
        if ai == none || !frozen.contains(&a.name) {
            continue;
        }
        for k in fam_keys {
            cells.push((ai, k, cmp(&col(res, ai, k), &col(res, none, k), EQ)));
        }
    }
    let adj_t = holm(&cells.iter().map(|c| c.2.tost_p).collect::<Vec<_>>());
    let adj_d = holm(&cells.iter().map(|c| c.2.p).collect::<Vec<_>>());
    let mut out_arms = json!({});
    log.push_str(&format!("{:26} {:>7} {:>7} | {:>8} {:>8} | hit1 d [lo,hi] EQ(holm) | mrr d [lo,hi] EQ(holm) | stale1 | rev_stale1 | loop safe/g1 dH2000 (T12,Tqa,ctl)\n", "arm", "rows%", "bytes%", "hit1", "mrr"));
    for (ai, a) in arms.iter().enumerate() {
        let br = mean(&col(res, ai, "c3.bytes")) / mean(&col(res, none, "c3.bytes"));
        let rr = mean(&col(res, ai, "c3.rows")) / mean(&col(res, none, "c3.rows"));
        let mut mj = json!({});
        let mut metrics: Vec<String> = res[0].arms[ai].keys().cloned().collect();
        metrics.sort();
        for k in &metrics {
            let v = col(res, ai, k);
            mj[k] = json!(v);
        }
        let mut tests = json!({});
        for (ci, (cai, k, c)) in cells.iter().enumerate() {
            if *cai == ai {
                tests[*k] = json!({"diff": c.diff, "lo": c.lo, "hi": c.hi, "p": c.p, "p_holm": adj_d[ci], "tost_p": c.tost_p, "tost_p_holm": adj_t[ci], "eq": adj_t[ci] <= 0.05, "noninf": c.lo > -EQ, "n": c.n});
            }
        }
        // closed loop vs none (all arms, descriptive + TOST margin .01; Holm applied only inside the frozen family below)
        let mut lt = json!({});
        for ws in lw {
            for (name, _) in lcfg {
                let key = format!("loop.{}.{}.h2000", ws.name, name);
                let c = cmp(&col(res, ai, &key), &col(res, none, &key), EQ);
                lt[key] = json!({"diff": c.diff, "lo": c.lo, "hi": c.hi, "p": c.p, "tost_p": c.tost_p});
            }
        }
        out_arms[a.name.clone()] = json!({"group": a.group, "bytes_frac": br, "rows_frac": rr, "frozen": frozen.contains(&a.name), "per_seed": mj, "tests": tests, "loop_tests": lt});
        let t = |k: &str| -> String {
            match tests.get(k) {
                Some(x) if x.is_object() => format!("{:+.4} [{:+.4},{:+.4}] {}", x["diff"].as_f64().unwrap(), x["lo"].as_f64().unwrap(), x["hi"].as_f64().unwrap(), if x["eq"].as_bool().unwrap() { "EQ" } else if x["noninf"].as_bool().unwrap() { "NI" } else { "--" }),
                _ => "            ".into(),
            }
        };
        let ld = |wn: &str, g: &str| -> f64 { mean(&col(res, ai, &format!("loop.{wn}.{g}.h2000"))) - mean(&col(res, none, &format!("loop.{wn}.{g}.h2000"))) };
        log.push_str(&format!(
            "{:26} {:7.1} {:7.1} | {:8.4} {:8.4} | {} | {} | {:6.3} | {:6.3} | {:+.4}/{:+.4} {:+.4}/{:+.4} {:+.4}/{:+.4}\n",
            a.name,
            100.0 * rr,
            100.0 * br,
            mean(&col(res, ai, "c3.u.hit1")),
            mean(&col(res, ai, "c3.u.mrr")),
            t("c3.u.hit1"),
            t("c3.u.mrr"),
            mean(&col(res, ai, "c3.u.stale1")),
            mean(&col(res, ai, "c3.u.rev_stale1")),
            ld("T12", "safe"),
            ld("T12", "g1"),
            ld("Tqa", "safe"),
            ld("Tqa", "g1"),
            ld("ctl-irrel", "safe"),
            ld("ctl-irrel", "g1")
        ));
    }
    // closed-loop Holm family: frozen arms x worlds x {safe, g1}
    let mut lcells: Vec<(usize, String, Cmp)> = vec![];
    for (ai, a) in arms.iter().enumerate() {
        if ai == none || !frozen.contains(&a.name) {
            continue;
        }
        for ws in lw {
            for g in ["safe", "g1"] {
                let key = format!("loop.{}.{}.h2000", ws.name, g);
                lcells.push((ai, key.clone(), cmp(&col(res, ai, &key), &col(res, none, &key), EQ)));
            }
        }
    }
    let ladj_t = holm(&lcells.iter().map(|c| c.2.tost_p).collect::<Vec<_>>());
    let ladj_d = holm(&lcells.iter().map(|c| c.2.p).collect::<Vec<_>>());
    let mut loop_fam = json!([]);
    for (i, (ai, key, c)) in lcells.iter().enumerate() {
        loop_fam.as_array_mut().unwrap().push(json!({"arm": arms[*ai].name, "metric": key, "diff": c.diff, "lo": c.lo, "hi": c.hi, "p_holm": ladj_d[i], "tost_p_holm": ladj_t[i], "eq": ladj_t[i] <= 0.05}));
    }
    let n_eq = lcells.iter().enumerate().filter(|(i, _)| ladj_t[*i] <= 0.05).count();
    log.push_str(&format!("closed-loop Holm family ({} cells = frozen arms x 5 worlds x 2 gates): TOST+-0.01 equivalent after Holm in {} cells\n", lcells.len(), n_eq));
    // context: retrieval value in the loop
    let mut ctx = json!({});
    for ws in lw {
        for g in ["safe", "g1"] {
            let key = format!("loop.{}.{}.h2000", ws.name, g);
            let fk = format!("loop.{}.front.h2000", ws.name);
            ctx[key.clone()] = json!({"none": mean(&col(res, none, &key)), "front": mean(&col(res, front, &fk)), "oracle": mean(&col(res, orc, &key))});
            log.push_str(&format!("  loop {:10} {:5}: front {:.4}  none-store {:.4}  oracle {:.4}\n", ws.name, g, mean(&col(res, front, &fk)), mean(&col(res, none, &key)), mean(&col(res, orc, &key))));
        }
    }
    // growth curve of the unbounded store (checkpoints)
    let mut curve = json!([]);
    for ci in 0..4 {
        let row = json!({"ckpt": ci, "rows": mean(&col(res, none, &format!("c{ci}.rows"))), "bytes": mean(&col(res, none, &format!("c{ci}.bytes"))),
            "hit1_u": mean(&col(res, none, &format!("c{ci}.u.hit1"))), "mrr_u": mean(&col(res, none, &format!("c{ci}.u.mrr"))), "stale1_u": mean(&col(res, none, &format!("c{ci}.u.stale1"))),
            "store_stale_frac": mean(&col(res, none, &format!("c{ci}.store_stale_frac"))), "elig": mean(&col(res, ext, &format!("c{ci}.elig")))});
        log.push_str(&format!("  growth none ckpt {ci}: rows {:.0} bytes {:.2} MB hit1 {:.4} mrr {:.4} stale1 {:.4} store_stale_frac {:.3} eligible q {:.0}\n", row["rows"].as_f64().unwrap(), row["bytes"].as_f64().unwrap() / 1e6, row["hit1_u"].as_f64().unwrap(), row["mrr_u"].as_f64().unwrap(), row["stale1_u"].as_f64().unwrap(), row["store_stale_frac"].as_f64().unwrap(), row["elig"].as_f64().unwrap()));
        curve.as_array_mut().unwrap().push(row);
    }
    json!({"label": label, "seeds": nseeds, "arms": out_arms, "loop_family": loop_fam, "loop_context": ctx, "growth_none": curve, "frozen": frozen})
}

// ------------------------------------------------------------------------------------------------ check

fn check(w: &WorldK, arms: &[Arm]) {
    use neural_mesh::flatq::{FlatIndex, Metric, Tier};
    println!("[check] quantisation matrices vs flatq::FlatIndex (top-5 ids over the BASE docs, 40 queries)");
    for (tier, nm) in [(Tier::BlockFp16, "block-fp16"), (Tier::Int8, "int8")] {
        let mut ix = FlatIndex::new(DIM, tier, Metric::Cosine);
        for d in 0..w.n {
            ix.add(d as u64, &w.emb[d * DIM..(d + 1) * DIM]).unwrap();
        }
        let m = if tier == Tier::Int8 { &w.s8 } else { &w.s16 };
        let (mut same, mut tot, mut maxd) = (0usize, 0usize, 0f32);
        for qi in (0..w.qs.len()).step_by((w.qs.len() / 40).max(1)).take(40) {
            let res = ix.search(&w.qvec[qi * DIM..(qi + 1) * DIM], 5).unwrap();
            let mut mine: Vec<(f32, usize)> = (0..w.n).map(|d| (m[qi * w.nu + d], d)).collect();
            mine.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            for (j, (id, sc)) in res.iter().enumerate() {
                tot += 1;
                if *id as usize == mine[j].1 {
                    same += 1;
                }
                maxd = maxd.max((sc - mine[j].0).abs());
            }
        }
        println!("  {nm}: top-5 id agreement {same}/{tot}, max |score diff| {maxd:.2e} (bytes/row {})", tier.row_bytes(DIM));
    }
    let s = make_stream(w, 1);
    let mut cnt = [0usize; 4];
    for r in &s.recs {
        cnt[w.kind[r.u as usize] as usize] += 1;
    }
    println!("[check] stream seed 1: {} records (base+exact-dups {} , old {} , near {} , noise {}), unbounded bytes {:.2} MB", s.recs.len(), cnt[0], cnt[1], cnt[2], cnt[3], s.bytes_none / 1e6);
    let revised = (0..w.n).filter(|&d| !s.old_arr[d].is_empty()).count();
    println!("[check] docs with >=1 old record: {} ({:.1}% of base docs)", revised, 100.0 * revised as f64 / w.n as f64);
    let cks: Vec<CkInfo> = CKPT_EP.iter().enumerate().map(|(i, &e)| ck_info(w, &s, s.ep_end[e], 1, i)).collect();
    println!("[check] eligible queries per checkpoint: {:?} of {}", cks.iter().map(|c| c.elig.len()).collect::<Vec<_>>(), w.qs.len());
    let usage: Vec<Vec<usize>> = (0..NEP).map(|e| usage_queries(w, &s, s.ep_end[e + 1], 1, e)).collect();
    let t0 = Instant::now();
    for nm in ["none", "exact", "supdel", "cosnew0.95", "val25", "summary"] {
        let arm = arms.iter().find(|a| a.name == nm).unwrap();
        let o = run_arm(w, &s, &cks, arm, 1, &usage);
        let c = o.cks.last().unwrap();
        println!("  arm {nm:12} rows {:6.0} bytes {:7.2} MB  u.hit1 {:.3} mrr {:.3} stale1 {:.3} noise1 {:.3} store_stale {:.3} merged {} stale_dropped {} ({:.1}s)", c.rows, c.bytes / 1e6, c.m["u.hit1"], c.m["u.mrr"], c.m["u.stale1"], c.m["u.noise1"], c.m["store_stale_frac"], o.merged, o.stale_dropped, t0.elapsed().as_secs_f64());
    }
    // exact-dup arm vs library Deduper count on the same stream
    use neural_mesh::dedup::{DedupConfig, Deduper, Verdict};
    let mut dd = Deduper::new(DedupConfig::default());
    let mut kept = 0;
    for (i, r) in s.recs.iter().enumerate() {
        let u = r.u as usize;
        if dd.check(i as u64 + 1, Some(&w.texts[u]), Some(&w.emb[u * DIM..(u + 1) * DIM])) == Verdict::New {
            kept += 1;
        }
    }
    println!("[check] library Deduper keeps {kept} of {} records ({} suppressed)", s.recs.len(), s.recs.len() - kept);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let stage = args.get(1).cloned().unwrap_or_default();
    let tag = args.iter().position(|a| a == "--world").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "A".into());
    let quick = args.iter().any(|a| a == "--quick");
    let t0 = Instant::now();
    let mut w = load_world(&tag, 0.70);
    let ctl = args.iter().position(|a| a == "--ctl").and_then(|i| args.get(i + 1)).cloned().unwrap_or_default();
    w.ctl = match ctl.as_str() { "norev" => 1, "nodup" => 2, "hirev" => 3, _ => 0 };
    let arms = arm_list();
    let lw = loop_worlds();
    let lcfg = loop_cfgs();
    match stage.as_str() {
        "check" => check(&w, &arms),
        "tune" => {
            let sds: Vec<u64> = if quick { (1..=3).collect() } else { (1..=8).collect() };
            let res = collect(&w, &arms, &sds, &lw, &lcfg);
            let (picks, mut log) = tune_picks(&arms, &res);
            let frozen = frozen_list(&picks);
            let mut j = report(&format!("TUNE world {tag}"), &arms, &res, &frozen, &lw, &lcfg, &mut log);
            j["picks"] = json!(picks);
            std::fs::write(format!("results/k1_picks_{tag}.json"), serde_json::to_string_pretty(&json!({"picks": picks, "frozen": frozen})).unwrap()).unwrap();
            std::fs::write(format!("results/k1_tune_{tag}.json"), serde_json::to_string(&j).unwrap()).unwrap();
            std::fs::write(format!("results/k1_tune_{tag}.log"), &log).unwrap();
            println!("{log}");
        }
        "held" | "confirm" => {
            let sds: Vec<u64> = if stage == "held" { if quick { (31..=36).collect() } else { (31..=60).collect() } } else if quick { (61..=66).collect() } else { (61..=90).collect() };
            let ptag = args.iter().position(|a| a == "--picks").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "A".into());
            let pj: Value = serde_json::from_slice(&std::fs::read(format!("results/k1_picks_{ptag}.json")).expect("run tune first")).unwrap();
            let frozen: Vec<String> = pj["frozen"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
            let res = collect(&w, &arms, &sds, &lw, &lcfg);
            let mut log = String::new();
            let mut j = report(&format!("{} world {tag} (picks frozen from world {ptag} tune seeds 1-8)", stage.to_uppercase()), &arms, &res, &frozen, &lw, &lcfg, &mut log);
            j["picks"] = pj["picks"].clone();
            std::fs::write(format!("results/k1_{stage}_{tag}.json"), serde_json::to_string(&j).unwrap()).unwrap();
            std::fs::write(format!("results/k1_{stage}_{tag}.log"), &log).unwrap();
            println!("{log}");
        }
        "posthoc" => {
            // POST-HOC (declared after the main stages): structural supersession WITHOUT revision stamps, confirm seeds 61-90
            let sds: Vec<u64> = if quick { (61..=66).collect() } else { (61..=90).collect() };
            let frozen: Vec<String> = ["supdel", "supdel-arrival", "exact+supdel"].iter().map(|s| s.to_string()).collect();
            let sub: Vec<Arm> = arms.iter().filter(|a| a.name == "none" || frozen.contains(&a.name)).cloned().collect();
            let res = collect(&w, &sub, &sds, &lw, &lcfg);
            let mut log = String::new();
            let j = report(&format!("POSTHOC world {tag} (confirm seeds 61-90)"), &sub, &res, &frozen, &lw, &lcfg, &mut log);
            std::fs::write(format!("results/k1_posthoc_{tag}.json"), serde_json::to_string(&j).unwrap()).unwrap();
            std::fs::write(format!("results/k1_posthoc_{tag}.log"), &log).unwrap();
            println!("{log}");
        }
        "ctl" => {
            // CONTROL / dose-response stage on fresh seeds 61-90 (frozen picks, subset of arms: frozen + none + val25rand)
            let sds: Vec<u64> = if quick { (61..=66).collect() } else { (61..=90).collect() };
            let pj: Value = serde_json::from_slice(&std::fs::read("results/k1_picks_A_frozen.json").expect("frozen picks")).unwrap();
            let mut frozen: Vec<String> = pj["frozen"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
            frozen.push("val25rand".into());
            let sub: Vec<Arm> = arms.iter().filter(|a| a.name == "none" || frozen.contains(&a.name)).cloned().collect();
            let res = collect(&w, &sub, &sds, &lw, &lcfg);
            let mut log = String::new();
            let j = report(&format!("CONTROL {ctl} world {tag} (confirm seeds 61-90, frozen A picks)"), &sub, &res, &frozen, &lw, &lcfg, &mut log);
            std::fs::write(format!("results/k1_ctl_{ctl}_{tag}.json"), serde_json::to_string(&j).unwrap()).unwrap();
            std::fs::write(format!("results/k1_ctl_{ctl}_{tag}.log"), &log).unwrap();
            println!("{log}");
        }
        _ => eprintln!("usage: k1_growth check|tune|held|confirm --world A|B [--quick]"),
    }
    println!("wall {:.0}s", t0.elapsed().as_secs_f64());
}
