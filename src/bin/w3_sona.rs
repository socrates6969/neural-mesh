//! W3: faithful port of ruflo's SONA router (src/sona.rs) vs the shortlab baselines, plus opt-in improvements.
//! Same protocol as shortlab: tune seeds 1-8 only (objective = mean of H=100/250/500), evaluate on
//! held-out seeds 31-60, paired 95% t-CI, Holm across (candidates x horizons).
//! Usage: cargo run --release --bin w3_sona [-- --out results/w3_sona.json]

use std::collections::HashMap;
use std::sync::Arc;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::memory::{normalize_in_place, synth};
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sona::{SonaConfig, SonaRouter};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::json;

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
    HebR,
    WarmHebR(f64, EvK),
    Sona(SonaConfig),
    WarmSona(SonaConfig, f64, EvK),
}

fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

fn drive<R: Router>(r: &mut R, w: &World, seed: u64) -> [f64; 5] {
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

fn run(kind: &Kind, w: &World, seed: u64) -> [f64; 5] {
    let a = w.cfg.agents;
    let km = || w.kwmap.clone();
    match kind {
        Kind::Keyword => drive(&mut KeywordRouter::new(km(), a), w, seed),
        Kind::Eps(g, e) => drive(&mut EpsGreedyRouter::new(a, *g, *e), w, seed),
        Kind::Pts(g) => drive(&mut PooledThompsonRouter::new(a, *g), w, seed),
        Kind::HebR => drive(&mut HebbianRouter::new(a, heb_robust_cfg()), w, seed),
        Kind::WarmHebR(k, e) => {
            drive(&mut Warm::new(HebbianRouter::new(a, heb_robust_cfg()), km(), a, *k).with_evidence(ev(*e)), w, seed)
        }
        Kind::Sona(c) => drive(&mut SonaRouter::new(a, &w.kwmap, c.clone()), w, seed),
        Kind::WarmSona(c, k, e) => drive(
            &mut Warm::new(SonaRouter::new(a, &w.kwmap, c.clone()), km(), a, *k).with_evidence(ev(*e)),
            w,
            seed,
        ),
    }
}

fn eval(kind: &Kind, seeds: &[u64], c: &WorldCfg) -> Vec<[f64; 5]> {
    seeds.par_iter().map(|&s| run(kind, &gen_world(s, c), s)).collect()
}
fn obj(kind: &Kind, seeds: &[u64], c: &WorldCfg) -> f64 {
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

fn sc(thr: f64, q: bool, eps: f64, untried: f64, feat: bool) -> SonaConfig {
    SonaConfig { threshold: thr, use_q: q, explore_eps: eps, explore_untried: untried, per_feature: feat, ..SonaConfig::default() }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/w3_sona.json".into());
    let mc = WorldCfg::main();
    let tune: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = (31..=60).collect();
    println!("w3_sona: {} tasks/run, tune seeds 1-8, held-out 31-60", N_TASKS);

    // ---- baselines tuned exactly as in shortlab ----
    let mut c = vec![];
    for &g in &[0.95, 0.99, 1.0] {
        for &e in &[0.02, 0.05, 0.1] {
            c.push(Kind::Eps(g, e));
        }
    }
    let eps_k = best_of(c, &tune, &mc, "eps-greedy");
    let c = [0.9, 0.95, 0.99, 1.0].iter().map(|&g| Kind::Pts(g)).collect();
    let pts_k = best_of(c, &tune, &mc, "pooled-ts");
    let evs = [EvK::Set, EvK::Mean, EvK::Global, EvK::FailSet, EvK::FailMean];
    let ks = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0];
    let mut c = vec![];
    for &k in &ks {
        for &e in &evs {
            c.push(Kind::WarmHebR(k, e));
        }
    }
    let warm_hebr = best_of(c, &tune, &mc, "warm(hebbian-robust)");

    // ---- SONA variants (tune seeds 1-8 only) ----
    let sona_faithful = Kind::Sona(SonaConfig::default());
    let mut c = vec![];
    for &t in &[0.5, 0.55, 0.6, 0.65, 0.7, 0.8] {
        for &q in &[true, false] {
            c.push(Kind::Sona(sc(t, q, 0.0, 0.0, false)));
        }
    }
    let sona_thr = best_of(c, &tune, &mc, "sona: threshold/Q only (no new mechanism)");
    let mut c = vec![];
    for &t in &[0.55, 0.6, 0.7] {
        for &e in &[0.02, 0.05, 0.1, 0.2] {
            c.push(Kind::Sona(sc(t, false, e, 0.0, false)));
        }
    }
    let sona_eps = best_of(c, &tune, &mc, "sona+eps");
    let mut c = vec![];
    for &t in &[0.55, 0.6, 0.7] {
        for &u in &[0.05, 0.1, 0.2, 0.4, 0.8] {
            c.push(Kind::Sona(sc(t, false, 0.0, u, false)));
        }
    }
    let sona_unt = best_of(c, &tune, &mc, "sona+untried");
    let mut c = vec![];
    for &t in &[0.5, 0.6, 0.7] {
        for &e in &[0.0, 0.02, 0.05, 0.1] {
            for &u in &[0.0, 0.1, 0.4] {
                c.push(Kind::Sona(sc(t, false, e, u, true)));
            }
        }
    }
    let sona_feat = best_of(c, &tune, &mc, "sona per-feature (+eps/untried)");
    let mut c = vec![];
    let Kind::Sona(fc) = sona_feat.clone() else { unreachable!() };
    for &k in &ks {
        for &e in &evs {
            c.push(Kind::WarmSona(fc.clone(), k, e));
        }
    }
    let warm_sona_feat = best_of(c, &tune, &mc, "warm(sona per-feature best)");

    // ---- held-out ----
    let baselines: Vec<(&str, Kind)> = vec![
        ("eps-greedy(tuned)", eps_k.clone()),
        ("pooled-ts(tuned)", pts_k.clone()),
        ("hebbian-robust", Kind::HebR),
        ("warm(hebbian-robust)", warm_hebr.clone()),
    ];
    let cands: Vec<(&str, Kind)> = vec![
        ("sona(faithful)", sona_faithful.clone()),
        ("sona(faithful,noQ)", Kind::Sona(sc(0.6, false, 0.0, 0.0, false))),
        ("sona(threshold-tuned)", sona_thr.clone()),
        ("sona+eps", sona_eps.clone()),
        ("sona+untried", sona_unt.clone()),
        ("sona-feature(+explore)", sona_feat.clone()),
        ("warm(sona-feature)", warm_sona_feat.clone()),
    ];
    let kw = eval(&Kind::Keyword, &held, &mc);
    let base_res: Vec<Vec<[f64; 5]>> = baselines.iter().map(|(_, k)| eval(k, &held, &mc)).collect();
    let cand_res: Vec<Vec<[f64; 5]>> = cands.iter().map(|(_, k)| eval(k, &held, &mc)).collect();
    // "best fair baseline" pool = keyword + the four learned baselines
    let pool_names: Vec<&str> = std::iter::once("keyword").chain(baselines.iter().map(|b| b.0)).collect();
    let pool_res: Vec<&Vec<[f64; 5]>> = std::iter::once(&kw).chain(base_res.iter()).collect();
    let best: Vec<usize> = (0..5)
        .map(|h| {
            (0..pool_res.len())
                .max_by(|&a, &b| summarize(&col(pool_res[a], h)).mean.partial_cmp(&summarize(&col(pool_res[b], h)).mean).unwrap())
                .unwrap()
        })
        .collect();

    println!("\n=== held-out seeds 31-60: mean expected success over first H tasks ===");
    print!("{:<28}", "router");
    for h in HORIZONS {
        print!("{:>16}", format!("H={h}"));
    }
    println!();
    for (i, n) in pool_names.iter().enumerate() {
        print!("{:<28}", n);
        for h in 0..5 {
            let s = summarize(&col(pool_res[i], h));
            print!("{:>16}", format!("{:.3}+-{:.3}", s.mean, s.half_width));
        }
        println!();
    }
    for (i, (n, _)) in cands.iter().enumerate() {
        print!("{:<28}", n);
        for h in 0..5 {
            let s = summarize(&col(&cand_res[i], h));
            print!("{:>16}", format!("{:.3}+-{:.3}", s.mean, s.half_width));
        }
        println!();
    }
    print!("{:<28}", "best fair baseline");
    for h in 0..5 {
        print!("{:>16}", pool_names[best[h]]);
    }
    println!();

    // family A = vs best fair baseline (primary, Holm over candidates x horizons)
    let (nc, nh) = (cands.len(), 5);
    let wi = pool_names.iter().position(|n| *n == "warm(hebbian-robust)").unwrap();
    let mut pa = vec![0.0; nc * nh];
    let mut pb = vec![0.0; nc * nh];
    let mut pk = vec![0.0; nc * nh];
    for ci in 0..nc {
        for h in 0..nh {
            let x = col(&cand_res[ci], h);
            pa[ci * nh + h] = paired_t_pvalue(&x, &col(pool_res[best[h]], h));
            pb[ci * nh + h] = paired_t_pvalue(&x, &col(pool_res[wi], h));
            pk[ci * nh + h] = paired_t_pvalue(&x, &col(&kw, h));
        }
    }
    let (ha, hb, hk) = (holm_adjust(&pa), holm_adjust(&pb), holm_adjust(&pk));
    println!("\n=== paired differences candidate - ref: mean [95% CI] wins/losses, Holm p ===");
    let mut tests = vec![];
    let mut accepted = vec![];
    for ci in 0..nc {
        println!("-- {}", cands[ci].0);
        for h in 0..nh {
            let x = col(&cand_res[ci], h);
            let da = paired_diff(&x, &col(pool_res[best[h]], h));
            let db = paired_diff(&x, &col(pool_res[wi], h));
            let dk = paired_diff(&x, &col(&kw, h));
            let (a, b, k) = (ha[ci * nh + h], hb[ci * nh + h], hk[ci * nh + h]);
            println!(
                "  H={:<5} vs best({}) {:+.4} [{:+.4},{:+.4}] w{}/l{} p={:.4}{} | vs warm(hebR) {:+.4} [{:+.4},{:+.4}] p={:.4} | vs keyword {:+.4} [{:+.4},{:+.4}] p={:.4}",
                HORIZONS[h], pool_names[best[h]], da.diff.mean, da.diff.lo, da.diff.hi, da.wins, da.losses, a,
                if a < 0.05 && da.diff.mean > 0.0 { " WIN" } else { "" },
                db.diff.mean, db.diff.lo, db.diff.hi, b, dk.diff.mean, dk.diff.lo, dk.diff.hi, k
            );
            if a < 0.05 && da.diff.mean > 0.0 {
                accepted.push((cands[ci].0, HORIZONS[h]));
            }
            tests.push(json!({"candidate": cands[ci].0, "horizon": HORIZONS[h], "mean": summarize(&x).mean,
                "best_baseline": pool_names[best[h]],
                "vs_best": {"diff": da.diff, "wins": da.wins, "losses": da.losses, "holm_p": a},
                "vs_warm_hebr": {"diff": db.diff, "wins": db.wins, "losses": db.losses, "holm_p": b},
                "vs_keyword": {"diff": dk.diff, "wins": dk.wins, "losses": dk.losses, "holm_p": k}}));
        }
    }
    println!("\nACCEPTED (beats best fair baseline, Holm<0.05): {:?}", accepted);

    // ---- diagnostics: where do faithful-SONA selections come from? ----
    println!("\n=== faithful SONA: selection source shares over 2000 tasks (mean over held-out seeds) ===");
    let diag: Vec<([f64; 6], usize)> = held
        .par_iter()
        .map(|&s| {
            let w = gen_world(s, &mc);
            let mut r = SonaRouter::new(w.cfg.agents, &w.kwmap, SonaConfig::default());
            drive(&mut r, &w, s);
            let tot: u64 = r.src.iter().sum();
            let mut f = [0.0; 6];
            for i in 0..6 {
                f[i] = r.src[i] as f64 / tot as f64;
            }
            (f, r.n_patterns())
        })
        .collect();
    let mut mean = [0.0; 6];
    for (f, _) in &diag {
        for i in 0..6 {
            mean[i] += f[i] / diag.len() as f64;
        }
    }
    let mp = diag.iter().map(|d| d.1 as f64).sum::<f64>() / diag.len() as f64;
    println!("  pattern {:.3} | q-learning {:.3} | keyword-heuristic {:.3} | default {:.3} | (patterns stored ~{:.0})", mean[0], mean[1], mean[2], mean[3], mp);

    let j = json!({
        "protocol": "tune seeds 1-8 (objective mean of H=100/250/500); held-out 31-60; paired 95% t-CI; Holm over candidates x horizons",
        "tuned": {"eps": format!("{:?}", eps_k), "pts": format!("{:?}", pts_k), "warm_hebr": format!("{:?}", warm_hebr),
                  "sona_thr": format!("{:?}", sona_thr), "sona_eps": format!("{:?}", sona_eps), "sona_untried": format!("{:?}", sona_unt),
                  "sona_feat": format!("{:?}", sona_feat), "warm_sona_feat": format!("{:?}", warm_sona_feat)},
        "best_baseline_by_horizon": best.iter().map(|&i| pool_names[i]).collect::<Vec<_>>(),
        "tests": tests,
        "accepted": accepted.iter().map(|(n, h)| json!({"candidate": n, "horizon": h})).collect::<Vec<_>>(),
        "sona_source_shares": {"pattern": mean[0], "q": mean[1], "keyword": mean[2], "default": mean[3], "patterns_stored": mp},
    });
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out}");
}
