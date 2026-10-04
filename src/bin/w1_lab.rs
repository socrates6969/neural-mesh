//! W1 lab: evidence-adaptive warm start (ArmWarm) vs the fair baselines.
//! Tune on seeds 1-8 only; evaluate paired on held-out seeds 31-60; Holm across
//! candidates x horizons (per world). Reuses the shortlab world generator.
//! Usage: cargo run --release --bin w1_lab [-- --out results/w1_lab.json]

use std::collections::HashMap;
use std::sync::Arc;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::memory::{normalize_in_place, synth};
use neural_mesh::prior::{ArmWarm, Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
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


fn heb_robust_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    Eps(f64, f64),
    Pts(f64),
    HebR,
    WarmHebR,
    Arm { k: f64, kwm: f64, kin: f64, inm: f64, pool: f64, disc: f64 },
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

fn run(kind: &Kind, w: &World, seed: u64) -> [f64; 5] {
    let a = w.cfg.agents;
    let km = || w.kwmap.clone();
    match kind {
        Kind::Keyword => drive(KeywordRouter::new(km(), a), w, seed),
        Kind::Eps(g, e) => drive(EpsGreedyRouter::new(a, *g, *e), w, seed),
        Kind::Pts(g) => drive(PooledThompsonRouter::new(a, *g), w, seed),
        Kind::HebR => drive(HebbianRouter::new(a, heb_robust_cfg()), w, seed),
        Kind::WarmHebR => drive(
            Warm::new(HebbianRouter::new(a, heb_robust_cfg()), km(), a, 1.0).with_evidence(Evidence::MeanFeature),
            w,
            seed,
        ),
        Kind::Arm { k, kwm, kin, inm, pool, disc } => drive(
            ArmWarm::new(HebbianRouter::new(a, heb_robust_cfg()), km(), a).with_params(*k, *kwm, *kin, *inm, *pool, *disc),
            w,
            seed,
        ),
    }
}

fn eval(kind: &Kind, seeds: &[u64], c: &WorldCfg) -> Vec<[f64; 5]> {
    seeds.par_iter().map(|&s| run(kind, &gen_world(s, c), s)).collect()
}

fn col(r: &[[f64; 5]], h: usize) -> Vec<f64> {
    r.iter().map(|x| x[h]).collect()
}

fn obj(kind: &Kind, seeds: &[u64], worlds: &[WorldCfg]) -> f64 {
    let mut t = 0.0;
    for c in worlds {
        let r = eval(kind, seeds, c);
        t += r.iter().map(|x| (x[0] + x[1] + x[2]) / 3.0).sum::<f64>() / r.len() as f64;
    }
    t / worlds.len() as f64
}

fn bench_ns(kind: &Kind, c: &WorldCfg) -> f64 {
    let w = gen_world(1, c);
    let t0 = std::time::Instant::now();
    let reps = 20;
    for s in 0..reps {
        let _ = run(kind, &w, s);
    }
    t0.elapsed().as_nanos() as f64 / (reps as f64 * N_TASKS as f64)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/w1_lab.json".into());
    let mc = WorldCfg::main();
    let ck = WorldCfg { name: "correct-keywords(no traps)", traps: 0, map_err: 0.0, ..WorldCfg::main() };
    let tune: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = (31..=60).collect();

    // tuning on seeds 1-8 (objective: mean H100/250/500 over main + correct-keywords worlds)
    let tw = [mc.clone(), ck.clone()];
    let mut best = (f64::NEG_INFINITY, Kind::Keyword);
    let mut n = 0;
    for &k in &[2.0, 4.0, 8.0, 16.0] {
        for &kwm in &[0.7, 0.8, 0.9] {
            for &kin in &[1.0, 2.0, 4.0] {
                for &inm in &[0.4, 0.5, 0.6] {
                    for &pool in &[1.0, 2.0, 4.0] {
                        let kd = Kind::Arm { k, kwm, kin, inm, pool, disc: 1.0 };
                        let o = obj(&kd, &tune, &tw);
                        n += 1;
                        if o > best.0 {
                            best = (o, kd);
                        }
                    }
                }
            }
        }
    }
    println!("[tune 1-8] {n} configs, best obj {:.4} -> {:?}", best.0, best.1);
    let mut best_d = best.clone();
    if let Kind::Arm { k, kwm, kin, inm, pool, .. } = best.1.clone() {
        for &disc in &[0.99, 0.995, 0.999] {
            let kd = Kind::Arm { k, kwm, kin, inm, pool, disc };
            let o = obj(&kd, &tune, &tw);
            if o > best_d.0 {
                best_d = (o, kd);
            }
        }
    }
    println!("[tune 1-8] with discount: {:?} {:.4}", best_d.1, best_d.0);
    let cands: Vec<(&str, Kind)> = vec![("armwarm(hebr)", best.1.clone()), ("armwarm(hebr)+disc", best_d.1.clone())];
    let baselines: Vec<(&str, Kind)> = vec![
        ("keyword", Kind::Keyword),
        ("eps-greedy(tuned)", Kind::Eps(0.95, 0.02)),
        ("pooled-ts(tuned)", Kind::Pts(1.0)),
        ("hebbian-robust", Kind::HebR),
        ("warm(hebbian-robust)", Kind::WarmHebR),
    ];
    let mut worlds_json = vec![];
    let worlds = vec![
        mc.clone(),
        ck.clone(),
        WorldCfg { name: "few-traps(1)", traps: 1, map_err: 0.05, ..WorldCfg::main() },
        WorldCfg { name: "many-topics(16)", topics: 16, ..WorldCfg::main() },
        WorldCfg { name: "no-trap-noisy-kw", traps: 0, map_err: 0.3, confound: 0.4, ..WorldCfg::main() },
    ];
    for wc in &worlds {
        let base: Vec<Vec<[f64; 5]>> = baselines.iter().map(|(_, k)| eval(k, &held, wc)).collect();
        let cr: Vec<Vec<[f64; 5]>> = cands.iter().map(|(_, k)| eval(k, &held, wc)).collect();
        println!("\n=== world {} | held-out 31-60 | mean expected success (prefix H) ===", wc.name);
        for (i, (n, _)) in baselines.iter().enumerate() {
            print!("{:<26}", n);
            for h in 0..5 {
                print!("{:>8.3}", summarize(&col(&base[i], h)).mean);
            }
            println!();
        }
        for (i, (n, _)) in cands.iter().enumerate() {
            print!("{:<26}", n);
            for h in 0..5 {
                print!("{:>8.3}", summarize(&col(&cr[i], h)).mean);
            }
            println!("   <- candidate");
        }
        let best_b: Vec<usize> = (0..5)
            .map(|h| {
                (0..base.len())
                    .max_by(|&a, &b| summarize(&col(&base[a], h)).mean.partial_cmp(&summarize(&col(&base[b], h)).mean).unwrap())
                    .unwrap()
            })
            .collect();
        let nc = cands.len();
        let mut ps = vec![0.0; nc * 5];
        for ci in 0..nc {
            for h in 0..5 {
                ps[ci * 5 + h] = paired_t_pvalue(&col(&cr[ci], h), &col(&base[best_b[h]], h));
            }
        }
        let adj = holm_adjust(&ps);
        let mut tj = vec![];
        for ci in 0..nc {
            for h in 0..5 {
                let d = paired_diff(&col(&cr[ci], h), &col(&base[best_b[h]], h));
                let sig = adj[ci * 5 + h] < 0.05;
                println!(
                    "  {} H={:<5} vs best-baseline({}) {:+.4} [{:+.4},{:+.4}] wins {}/{} holm p={:.4} {}",
                    cands[ci].0,
                    HORIZONS[h],
                    baselines[best_b[h]].0,
                    d.diff.mean,
                    d.diff.lo,
                    d.diff.hi,
                    d.wins,
                    d.wins + d.losses,
                    adj[ci * 5 + h],
                    if sig && d.diff.mean > 0.0 { "WIN" } else if sig { "LOSS" } else { "ns" }
                );
                tj.push(json!({"cand": cands[ci].0, "horizon": HORIZONS[h], "best_baseline": baselines[best_b[h]].0, "diff": d.diff, "wins": d.wins, "losses": d.losses, "holm_p": adj[ci*5+h]}));
            }
        }
        worlds_json.push(json!({"world": wc.name, "tests": tj}));
    }
    println!("\n=== ns per (select+update) incl. world replay, main world ===");
    let mut bj = vec![];
    for (n, k) in baselines.iter().chain(cands.iter()) {
        let ns = bench_ns(k, &mc);
        println!("{:<26}{:>10.0} ns/route", n, ns);
        bj.push(json!({"router": n, "ns_per_route": ns}));
    }
    let j = json!({"tuned": format!("{:?} / {:?}", best.1, best_d.1), "worlds": worlds_json, "ns": bj});
    let _ = std::fs::create_dir_all("results");
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out}");
}
