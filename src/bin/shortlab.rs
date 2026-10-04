//! Short-horizon lab: can a keyword warm-start and/or an embedding case base beat
//! the static keyword map (and the best fair learned baseline) at the short
//! horizons real deployments see (100-2000 outcomes)?
//!
//! World (seeded): T topics, A agents; every topic has a hidden best agent (the
//! topic->best map is re-drawn per seed). Tasks carry BOTH noisy keywords and a
//! topic-clustered embedding. A few "trap" topics have keywords that point at a
//! bad agent. Outcome = (u < p[topic][agent]) with u pre-drawn (common random
//! numbers across routers). Metric = mean *expected* success p of the chosen
//! agent over the first H tasks (prefix means of one 2000-task run, so every
//! horizon is evaluated on the same run).
//!
//! Protocol: tune on seeds 1-8 only; evaluate paired on held-out seeds 31-60;
//! 95% t-CI; Holm across the "beats keyword" and "beats best fair baseline"
//! families (candidates x horizons). A candidate is accepted at a horizon only if
//! it is significantly better than BOTH keyword and the best fair baseline.
//!
//! Usage: cargo run --release --bin shortlab [-- --out results/shortlab.json]

use std::sync::Arc;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::casebased::{CaseConfig, CaseRouter, Embedder};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::memory::{normalize_in_place, synth};
use neural_mesh::prior::{Evidence, KeywordPrior, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashMap;

const HORIZONS: [usize; 5] = [100, 250, 500, 1000, 2000];
const N_TASKS: usize = 2000;

#[derive(Clone, Debug)]
struct WorldCfg {
    name: &'static str,
    topics: usize,
    agents: usize,
    dim: usize,
    emb_noise: f32,
    traps: usize,
    map_err: f64,
    confound: f64,
}

impl WorldCfg {
    fn main() -> Self {
        WorldCfg { name: "main", topics: 8, agents: 6, dim: 32, emb_noise: 0.9, traps: 2, map_err: 0.10, confound: 0.25 }
    }
}

struct World {
    cfg: WorldCfg,
    tasks: Vec<Task>,
    topic: Vec<usize>,
    embs: Arc<Vec<Vec<f32>>>,
    u: Vec<f64>,
    p: Vec<Vec<f64>>,
    kwmap: HashMap<String, usize>,
}

fn gen_world(seed: u64, c: &WorldCfg) -> World {
    let mut rng = Rng::new(seed.wrapping_mul(7919).wrapping_add(11));
    let mut er = synth::Rng::new(seed.wrapping_mul(31).wrapping_add(5));
    let (t_n, a_n) = (c.topics, c.agents);
    let mut p = vec![vec![0.0; a_n]; t_n];
    let mut best = vec![0usize; t_n];
    let mut second = vec![0usize; t_n];
    for t in 0..t_n {
        best[t] = rng.below(a_n);
        loop {
            second[t] = rng.below(a_n);
            if second[t] != best[t] {
                break;
            }
        }
        for a in 0..a_n {
            p[t][a] = if a == best[t] {
                0.85
            } else if a == second[t] {
                0.5
            } else {
                0.15 + 0.25 * rng.f64()
            };
        }
    }
    // trap topics (distinct)
    let mut order: Vec<usize> = (0..t_n).collect();
    for i in (1..t_n).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let trap: Vec<bool> = (0..t_n).map(|t| order.iter().take(c.traps).any(|&x| x == t)).collect();
    let mut kwmap = HashMap::new();
    for t in 0..t_n {
        for j in 0..3 {
            let a = if trap[t] {
                loop {
                    let a = rng.below(a_n);
                    if a != best[t] && a != second[t] {
                        break a;
                    }
                }
            } else if rng.f64() < c.map_err {
                rng.below(a_n)
            } else {
                best[t]
            };
            kwmap.insert(format!("t{t}k{j}"), a);
        }
    }
    // centroids
    let cents: Vec<Vec<f32>> = (0..t_n)
        .map(|_| {
            let mut v: Vec<f32> = (0..c.dim).map(|_| er.normal()).collect();
            normalize_in_place(&mut v);
            v
        })
        .collect();
    let sc = c.emb_noise / (c.dim as f32).sqrt();
    let (mut tasks, mut topic, mut embs, mut u) = (vec![], vec![], vec![], vec![]);
    for i in 0..N_TASKS {
        let t = rng.below(t_n);
        let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
        if f.is_empty() && rng.bernoulli(0.9) {
            f.push(format!("t{t}k{}", rng.below(3)));
        }
        if rng.bernoulli(c.confound) {
            let o = (t + 1 + rng.below(t_n - 1)) % t_n;
            f.push(format!("t{o}k{}", rng.below(3)));
        }
        let mut e: Vec<f32> = cents[t].iter().map(|x| x + sc * er.normal()).collect();
        normalize_in_place(&mut e);
        tasks.push(Task { id: i as u64, task_type: None, features: f });
        topic.push(t);
        embs.push(e);
        u.push(rng.f64());
    }
    World { cfg: c.clone(), tasks, topic, embs: Arc::new(embs), u, p, kwmap }
}

#[derive(Clone, Copy, Debug)]
enum EvK {
    Set,
    Mean,
    Global,
    FailSet,
    FailMean,
}
fn ev<R>(e: EvK) -> Evidence<R> {
    match e {
        EvK::Set => Evidence::FeatureSet,
        EvK::Mean => Evidence::MeanFeature,
        EvK::Global => Evidence::Global,
        EvK::FailSet => Evidence::SetFailures,
        EvK::FailMean => Evidence::FeatureFailures,
    }
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    Heb,
    HebR,
    WarmHebR(f64, EvK),
    WarmEps(f64, f64, f64, EvK),
    WarmPts(f64, f64, EvK),
    WarmHeb(f64, EvK),
    Case(CaseConfig),
    WarmCase(CaseConfig, f64),
    CaseKw(CaseConfig),
}

fn heb_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.05, temperature: 0.02, decay: 0.001, ..HebbianConfig::default() }
}

fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

fn drive<R: Router>(mut r: R, w: &World, seed: u64) -> [f64; 5] {
    let cands: Vec<usize> = (0..w.cfg.agents).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut cum = 0.0;
    let mut out = [0.0; 5];
    let mut hi = 0;
    for (i, task) in w.tasks.iter().enumerate() {
        let a = r.select(task, &cands, &mut rng);
        let p = w.p[w.topic[i]][a];
        cum += p;
        r.update(task, a, if w.u[i] < p { 1.0 } else { 0.0 });
        if hi < 5 && i + 1 == HORIZONS[hi] {
            out[hi] = cum / (i + 1) as f64;
            hi += 1;
        }
    }
    out
}

fn embedder(w: &World) -> Embedder {
    let e = w.embs.clone();
    Arc::new(move |t: &Task| e.get(t.id as usize).cloned())
}

fn run(kind: &Kind, w: &World, seed: u64) -> [f64; 5] {
    let a = w.cfg.agents;
    let km = || w.kwmap.clone();
    let case = |c: &CaseConfig| CaseRouter::new(a, w.cfg.dim, c.clone(), embedder(w));
    match kind {
        Kind::Keyword => drive(KeywordRouter::new(km(), a), w, seed),
        Kind::Eps(g, e) => drive(EpsGreedyRouter::new(a, *g, *e), w, seed),
        Kind::Pts(g) => drive(PooledThompsonRouter::new(a, *g), w, seed),
        Kind::Heb => drive(HebbianRouter::new(a, heb_cfg()), w, seed),
        Kind::HebR => drive(HebbianRouter::new(a, heb_robust_cfg()), w, seed),
        Kind::WarmHebR(k, evk) => {
            drive(Warm::new(HebbianRouter::new(a, heb_robust_cfg()), km(), a, *k).with_evidence(ev(*evk)), w, seed)
        }
        Kind::WarmEps(g, e, k, evk) => {
            drive(Warm::new(EpsGreedyRouter::new(a, *g, *e), km(), a, *k).with_evidence(ev(*evk)), w, seed)
        }
        Kind::WarmPts(g, k, evk) => {
            drive(Warm::new(PooledThompsonRouter::new(a, *g), km(), a, *k).with_evidence(ev(*evk)), w, seed)
        }
        Kind::WarmHeb(k, evk) => {
            drive(Warm::new(HebbianRouter::new(a, heb_cfg()), km(), a, *k).with_evidence(ev(*evk)), w, seed)
        }
        Kind::Case(c) => drive(case(c), w, seed),
        Kind::WarmCase(c, k) => drive(
            Warm::new(case(c), km(), a, *k).with_evidence(Evidence::Inner(CaseRouter::local_mass)),
            w,
            seed,
        ),
        Kind::CaseKw(c) => drive(case(c).with_keyword_prior(KeywordPrior::new(km(), a)), w, seed),
    }
}

/// results[seed_idx][horizon]
fn eval(kind: &Kind, seeds: &[u64], c: &WorldCfg) -> Vec<[f64; 5]> {
    seeds
        .par_iter()
        .map(|&s| {
            let w = gen_world(s, c);
            run(kind, &w, s)
        })
        .collect()
}

fn obj(kind: &Kind, seeds: &[u64], c: &WorldCfg) -> f64 {
    // tuning objective: SHORT horizons (100, 250, 500) - the deployment regime
    let r = eval(kind, seeds, c);
    r.iter().map(|x| (x[0] + x[1] + x[2]) / 3.0).sum::<f64>() / r.len() as f64
}

fn best_of(cands: Vec<Kind>, seeds: &[u64], c: &WorldCfg, label: &str) -> Kind {
    let mut best = (f64::NEG_INFINITY, cands[0].clone());
    let n = cands.len();
    for k in cands {
        let o = obj(&k, seeds, c);
        if o > best.0 {
            best = (o, k);
        }
    }
    println!("[tune 1-8] {label}: {n} configs, best {:.4} -> {:?}", best.0, best.1);
    best.1
}

fn col(r: &[[f64; 5]], h: usize) -> Vec<f64> {
    r.iter().map(|x| x[h]).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/shortlab.json".into());
    let mc = WorldCfg::main();
    let tune: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = (31..=60).collect();
    println!("neural-mesh shortlab: {} tasks/run, tune seeds 1-8, held-out 31-60", N_TASKS);

    // ---------------- tuning (seeds 1-8 only) ----------------
    let mut c = vec![];
    for &g in &[0.95, 0.99, 1.0] {
        for &e in &[0.02, 0.05, 0.1] {
            c.push(Kind::Eps(g, e));
        }
    }
    let Kind::Eps(eg, ee) = best_of(c, &tune, &mc, "eps-greedy") else { unreachable!() };
    let c = [0.9, 0.95, 0.99, 1.0].iter().map(|&g| Kind::Pts(g)).collect();
    let Kind::Pts(pg) = best_of(c, &tune, &mc, "pooled-ts") else { unreachable!() };

    let evs = [EvK::Set, EvK::Mean, EvK::Global, EvK::FailSet, EvK::FailMean];
    let ks = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0];
    let mut c = vec![];
    for &k in &ks {
        for &e in &evs {
            c.push(Kind::WarmEps(eg, ee, k, e));
        }
    }
    let warm_eps = best_of(c, &tune, &mc, "warm(eps)");
    let mut c = vec![];
    for &k in &ks {
        for &e in &evs {
            c.push(Kind::WarmPts(pg, k, e));
        }
    }
    let warm_pts = best_of(c, &tune, &mc, "warm(pts)");
    let mut c = vec![];
    for &k in &ks {
        for &e in &evs {
            c.push(Kind::WarmHeb(k, e));
        }
    }
    let warm_heb = best_of(c, &tune, &mc, "warm(hebbian)");
    let mut c = vec![];
    for &k in &ks {
        for &e in &evs {
            c.push(Kind::WarmHebR(k, e));
        }
    }
    let warm_hebr = best_of(c, &tune, &mc, "warm(hebbian-robust)");

    let mut c = vec![];
    for &k in &[10usize, 30] {
        for &h in &[0.05, 0.15, 0.4] {
            for &(a0, b0) in &[(1.0, 1.0), (0.5, 1.5)] {
                for &ws in &[1.0, 2.0, 4.0] {
                    c.push(Kind::Case(CaseConfig { k, bandwidth: h, a0, b0, wscale: ws, kw_pseudo: 0.0, kw_mean: 1.0 }));
                }
            }
        }
    }
    let case_k = best_of(c, &tune, &mc, "case");
    let Kind::Case(cc) = case_k.clone() else { unreachable!() };
    let c = [2.0, 5.0, 10.0, 20.0, 40.0, 100.0].iter().map(|&k| Kind::WarmCase(cc.clone(), k)).collect();
    let warm_case = best_of(c, &tune, &mc, "warm(case)");
    let mut c = vec![];
    for &p in &[3.0, 6.0, 12.0, 24.0, 48.0] {
        for &m in &[1.0, 0.8, 0.65] {
            c.push(Kind::CaseKw(CaseConfig { kw_pseudo: p, kw_mean: m, ..cc.clone() }));
        }
    }
    let case_kw = best_of(c, &tune, &mc, "case+kwprior");

    // ---------------- held-out evaluation ----------------
    let baselines: Vec<(&str, Kind)> = vec![
        ("eps-greedy(untuned .99,.02)", Kind::Eps(0.99, 0.02)),
        ("eps-greedy(tuned)", Kind::Eps(eg, ee)),
        ("pooled-ts(tuned)", Kind::Pts(pg)),
        ("hebbian-tuned", Kind::Heb),
        ("hebbian-robust", Kind::HebR),
    ];
    let cands: Vec<(&str, Kind)> = vec![
        ("warm(eps)", warm_eps.clone()),
        ("warm(pooled-ts)", warm_pts.clone()),
        ("warm(hebbian)", warm_heb.clone()),
        ("warm(hebbian-robust)", warm_hebr.clone()),
        ("case", case_k.clone()),
        ("warm(case)", warm_case.clone()),
        ("case+kwprior", case_kw.clone()),
    ];
    let kw = eval(&Kind::Keyword, &held, &mc);
    let base_res: Vec<Vec<[f64; 5]>> = baselines.iter().map(|(_, k)| eval(k, &held, &mc)).collect();
    let cand_res: Vec<Vec<[f64; 5]>> = cands.iter().map(|(_, k)| eval(k, &held, &mc)).collect();
    // oracle (for context)
    let oracle: Vec<f64> = held
        .iter()
        .map(|&s| {
            let w = gen_world(s, &mc);
            (0..N_TASKS).map(|i| w.p[w.topic[i]].iter().cloned().fold(0.0, f64::max)).sum::<f64>() / N_TASKS as f64
        })
        .collect();

    // best fair baseline per horizon (by held-out mean: conservative)
    let best_base: Vec<usize> = (0..5)
        .map(|h| {
            (0..baselines.len())
                .max_by(|&a, &b| {
                    summarize(&col(&base_res[a], h)).mean.partial_cmp(&summarize(&col(&base_res[b], h)).mean).unwrap()
                })
                .unwrap()
        })
        .collect();

    println!("\n=== held-out seeds 31-60: mean expected success (prefix over first H tasks) ===");
    print!("{:<30}", "router");
    for h in HORIZONS {
        print!("{:>16}", format!("H={h}"));
    }
    println!();
    let mut rows: Vec<(String, &Vec<[f64; 5]>)> = vec![("keyword".into(), &kw)];
    for (i, (n, _)) in baselines.iter().enumerate() {
        rows.push((n.to_string(), &base_res[i]));
    }
    for (i, (n, _)) in cands.iter().enumerate() {
        rows.push((n.to_string(), &cand_res[i]));
    }
    for (n, r) in &rows {
        print!("{:<30}", n);
        for h in 0..5 {
            let s = summarize(&col(r, h));
            print!("{:>16}", format!("{:.3}+-{:.3}", s.mean, s.half_width));
        }
        println!();
    }
    print!("{:<30}", "oracle");
    for _ in 0..5 {
        print!("{:>16}", "");
    }
    println!("  (full-run mean {:.3})", summarize(&oracle).mean);
    print!("{:<30}", "best fair baseline");
    for h in 0..5 {
        print!("{:>16}", baselines[best_base[h]].0.split('(').next().unwrap_or(""));
    }
    println!();

    // ---------------- paired tests + Holm ----------------
    let nh = 5;
    let nc = cands.len();
    let mut p_kw = vec![0.0; nc * nh];
    let mut p_bb = vec![0.0; nc * nh];
    for ci in 0..nc {
        for h in 0..nh {
            let x = col(&cand_res[ci], h);
            p_kw[ci * nh + h] = paired_t_pvalue(&x, &col(&kw, h));
            p_bb[ci * nh + h] = paired_t_pvalue(&x, &col(&base_res[best_base[h]], h));
        }
    }
    let a_kw = holm_adjust(&p_kw);
    let a_bb = holm_adjust(&p_bb);

    println!("\n=== paired differences (candidate - ref), mean [95% CI], Holm-adjusted p ===");
    let mut json_tests = vec![];
    let mut accepted: Vec<(String, usize)> = vec![];
    let mut lose2000: Vec<String> = vec![];
    for ci in 0..nc {
        println!("-- {}", cands[ci].0);
        for h in 0..nh {
            let x = col(&cand_res[ci], h);
            let d_kw = paired_diff(&x, &col(&kw, h));
            let d_bb = paired_diff(&x, &col(&base_res[best_base[h]], h));
            let (pk, pb) = (a_kw[ci * nh + h], a_bb[ci * nh + h]);
            println!(
                "  H={:<5} vs keyword {:+.4} [{:+.4},{:+.4}] holm p={:.4} {} | vs best-base({}) {:+.4} [{:+.4},{:+.4}] holm p={:.4} {}",
                HORIZONS[h],
                d_kw.diff.mean, d_kw.diff.lo, d_kw.diff.hi, pk, if pk < 0.05 && d_kw.diff.mean > 0.0 { "WIN " } else { "    " },
                baselines[best_base[h]].0.split('(').next().unwrap_or(""),
                d_bb.diff.mean, d_bb.diff.lo, d_bb.diff.hi, pb, if pb < 0.05 && d_bb.diff.mean > 0.0 { "WIN " } else { "    " },
            );
            let beats_both = pk < 0.05 && d_kw.diff.mean > 0.0 && pb < 0.05 && d_bb.diff.mean > 0.0;
            if beats_both && h < 3 {
                accepted.push((cands[ci].0.to_string(), HORIZONS[h]));
            }
            // "max(keyword, best baseline)" reference at this horizon
            let mk = summarize(&col(&kw, h)).mean;
            let mb = summarize(&col(&base_res[best_base[h]], h)).mean;
            let refv = if mk >= mb { col(&kw, h) } else { col(&base_res[best_base[h]], h) };
            let d_ref = paired_diff(&x, &refv);
            if h == 4 && d_ref.diff.hi < 0.0 {
                lose2000.push(cands[ci].0.to_string());
            }
            json_tests.push(json!({
                "candidate": cands[ci].0, "horizon": HORIZONS[h],
                "mean": summarize(&x).mean,
                "vs_keyword": {"diff": d_kw.diff, "wins": d_kw.wins, "losses": d_kw.losses, "holm_p": pk},
                "vs_best_baseline": {"baseline": baselines[best_base[h]].0, "diff": d_bb.diff, "wins": d_bb.wins, "losses": d_bb.losses, "holm_p": pb},
                "vs_max_ref": {"diff": d_ref.diff, "significant": d_ref.significant},
                "beats_both_holm": beats_both,
            }));
        }
    }
    println!("\naccepted (beats keyword AND best fair baseline, Holm<0.05, H<=500): {:?}", accepted);
    println!("candidates significantly WORSE than max(keyword,best baseline) at H=2000: {:?}", lose2000);

    // ---------------- robustness worlds (tuned params fixed, held-out seeds) ----------------
    let alts = vec![
        WorldCfg { name: "noisy-embeddings", emb_noise: 1.6, ..WorldCfg::main() },
        WorldCfg { name: "few-traps(1)", traps: 1, map_err: 0.05, ..WorldCfg::main() },
        WorldCfg { name: "many-topics(16)", topics: 16, ..WorldCfg::main() },
        WorldCfg { name: "correct-keywords(no traps)", traps: 0, map_err: 0.0, ..WorldCfg::main() },
        WorldCfg { name: "no-trap-noisy-kw", traps: 0, map_err: 0.3, confound: 0.4, ..WorldCfg::main() },
    ];
    let mut alt_json = vec![];
    println!("\n=== robustness worlds (no retuning), mean success; (diff vs keyword, CI) ===");
    for ac in &alts {
        println!("-- {}", ac.name);
        let akw = eval(&Kind::Keyword, &held, ac);
        let mut ents = vec![];
        let all: Vec<(&str, Kind)> = baselines.iter().chain(cands.iter()).map(|(n, k)| (*n, k.clone())).collect();
        print!("  {:<30}", "keyword");
        for h in 0..5 {
            print!("{:>9.3}", summarize(&col(&akw, h)).mean);
        }
        println!();
        for (n, k) in &all {
            let r = eval(k, &held, ac);
            print!("  {:<30}", n);
            let mut hs = vec![];
            for h in 0..5 {
                let d = paired_diff(&col(&r, h), &col(&akw, h));
                print!("{:>9.3}", summarize(&col(&r, h)).mean);
                hs.push(json!({"horizon": HORIZONS[h], "mean": summarize(&col(&r, h)).mean, "vs_keyword": d.diff}));
            }
            let d250 = paired_diff(&col(&r, 1), &col(&akw, 1));
            let d2000 = paired_diff(&col(&r, 4), &col(&akw, 4));
            println!("   d250 {:+.3}[{:+.3},{:+.3}]  d2000 {:+.3}[{:+.3},{:+.3}]", d250.diff.mean, d250.diff.lo, d250.diff.hi, d2000.diff.mean, d2000.diff.lo, d2000.diff.hi);
            ents.push(json!({"router": n, "by_horizon": hs}));
        }
        alt_json.push(json!({"world": ac.name, "keyword": (0..5).map(|h| summarize(&col(&akw, h)).mean).collect::<Vec<_>>(), "routers": ents}));
    }

    let tab: Vec<_> = rows
        .iter()
        .map(|(n, r)| json!({"router": n, "by_horizon": (0..5).map(|h| json!({"horizon": HORIZONS[h], "summary": summarize(&col(r, h))})).collect::<Vec<_>>()}))
        .collect();
    let j = json!({
        "protocol": "tune seeds 1-8; held-out seeds 31-60; paired 95% t-CI; Holm over candidates x horizons per family",
        "metric": "mean expected success of chosen agent over first H tasks",
        "tuned": {
            "eps_greedy": format!("{:?}", Kind::Eps(eg, ee)), "pooled_ts": format!("{:?}", Kind::Pts(pg)),
            "warm_eps": format!("{:?}", warm_eps), "warm_pts": format!("{:?}", warm_pts), "warm_heb": format!("{:?}", warm_heb), "warm_hebr": format!("{:?}", warm_hebr),
            "case": format!("{:?}", case_k), "warm_case": format!("{:?}", warm_case), "case_kw": format!("{:?}", case_kw),
        },
        "oracle_mean": summarize(&oracle).mean,
        "table": tab,
        "best_baseline_by_horizon": best_base.iter().map(|&i| baselines[i].0).collect::<Vec<_>>(),
        "tests": json_tests,
        "accepted_short_horizon": accepted.iter().map(|(n, h)| json!({"candidate": n, "horizon": h})).collect::<Vec<_>>(),
        "significantly_worse_at_2000": lose2000,
        "robustness_worlds": alt_json,
    });
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("\nwrote {out}");
}
