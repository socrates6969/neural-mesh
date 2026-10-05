//! H10 (REAL-TEXT replay, task 2a): CompactRouter (u16/q16) vs the f64 recommended router on a REAL-TEXT-SHAPED routing replay.
//!
//! REAL: task texts (sentences of real doc paragraphs), their keyword features (long-tailed real vocabulary), and the topic structure
//!       (k-means over REAL MiniLM paragraph embeddings, py/h10_replay.py).
//! SIMULATED (mine, labelled): which agent is competent on which topic cluster (hidden best agent .80-.90, second .55-.68, others .15-.45;
//!       25% of clusters switch best agent at the midpoint), the keyword->agent prior (65% points at the cluster's best agent) and every
//!       Bernoulli outcome.  Controls: 'shuffled' = the task's competence cluster is drawn independently of its text (features carry no
//!       information about who is competent).
//! PRE-DECLARED PROTOCOL (nothing tuned; CompactConfig::default() == defaults::recommended_*): metric = full-run mean EXPECTED success;
//!       paired by seed; TOST margin +-0.005 (primary) and +-0.003 (strict); alpha .05 (90% CI inside the margin); Holm over the worlds per
//!       (subject, split, margin); splits held 31-60, confirm 61-90, pooled 31-90 (primary), power 91-250; seeds 1-8 smoke only.
//!       "CERTIFIED" iff Holm p < .05 AND the 90% CI is inside the margin.
//! Usage: h10_replay [--out results/h10_replay.json] [--quick]

#[path = "g3_common/mod.rs"]
mod g3;

use std::collections::HashMap;
use std::time::Instant;

use g3::{eq_test, n_workers, par_map, seeds};
use neural_mesh::compact::{CompactConfig, CompactRouter, CompactRouterF64, IdMode};
use neural_mesh::defaults::recommended_router;
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::holm_adjust;
use serde_json::{Value, json};

struct Fixture {
    tasks: Vec<Vec<String>>,
    /// per K: (cluster per task, token -> modal cluster)
    k: HashMap<usize, (Vec<usize>, HashMap<String, usize>)>,
}

fn load(tag: &str) -> Fixture {
    let j: Value = serde_json::from_slice(&std::fs::read(format!("results/h10_replay_{tag}.json")).expect("run py/h10_replay.py first")).unwrap();
    let tasks = j["tasks"].as_array().unwrap().iter().map(|t| t.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()).collect();
    let mut k = HashMap::new();
    for (kk, v) in j["K"].as_object().unwrap() {
        let cl = v["cluster"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
        let kw = v["kwtok"].as_object().unwrap().iter().map(|(a, b)| (a.clone(), b.as_u64().unwrap() as usize)).collect();
        k.insert(kk.parse().unwrap(), (cl, kw));
    }
    Fixture { tasks, k }
}

#[derive(Clone)]
struct Spec {
    name: &'static str,
    tag: &'static str, // fixture: A, B (feature df>=6) or A_tail, B_tail (df>=2: rare long-tail features)
    k: usize,
    agents: usize,
    n_tasks: usize,
    shuffled: bool,
}

fn specs() -> Vec<Spec> {
    let m = |name, tag, k, agents, n_tasks, shuffled| Spec { name, tag, k, agents, n_tasks, shuffled };
    vec![
        m("realA-K12-6ag", "A", 12, 6, 2000, false),
        m("realA-K48-32ag", "A", 48, 32, 4000, false),
        m("realB-K12-6ag", "B", 12, 6, 2000, false),
        m("realB-K48-32ag", "B", 48, 32, 4000, false),
        m("realA-tail-K48-32ag", "A_tail", 48, 32, 4000, false),
        m("realB-tail-K12-6ag", "B_tail", 12, 6, 2000, false),
        m("ctl-shuffled-A-K12-6ag", "A", 12, 6, 2000, true),
        m("ctl-shuffled-B-K48-32ag", "B", 48, 32, 4000, true),
    ]
}

struct Env {
    tasks: Vec<Task>,
    cl: Vec<usize>,
    u: Vec<f64>,
    p_pre: Vec<Vec<f64>>,
    p_post: Vec<Vec<f64>>,
    kw: HashMap<String, usize>,
    agents: usize,
}

fn uni(r: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * r.f64()
}

fn make_env(fx: &Fixture, sp: &Spec, seed: u64) -> Env {
    let (cl_of_task, kwtok) = &fx.k[&sp.k];
    let n = sp.agents;
    let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x7EA1_7E57 ^ (sp.k as u64) << 32 ^ (sp.agents as u64) << 40);
    let mut p_pre = vec![vec![0.0; n]; sp.k];
    let mut best_of = vec![0usize; sp.k];
    for c in 0..sp.k {
        for a in 0..n {
            p_pre[c][a] = uni(&mut r, 0.15, 0.45);
        }
        let best = r.below(n);
        p_pre[c][best] = uni(&mut r, 0.80, 0.90);
        let mut second = r.below(n - 1);
        if second >= best {
            second += 1;
        }
        p_pre[c][second] = uni(&mut r, 0.55, 0.68);
        best_of[c] = best;
    }
    let mut p_post = p_pre.clone();
    for c in 0..sp.k {
        if r.bernoulli(0.25) {
            let old = best_of[c];
            let mut new = r.below(n - 1);
            if new >= old {
                new += 1;
            }
            p_post[c][old] = uni(&mut r, 0.30, 0.50);
            p_post[c][new] = uni(&mut r, 0.80, 0.90);
        }
    }
    // keyword prior: points at the modal cluster's pre-drift best agent w.p. .65 else a wrong agent (a 'trap')
    let mut kw = HashMap::new();
    let mut toks: Vec<(&String, &usize)> = kwtok.iter().collect();
    toks.sort();
    for (t, &c) in toks {
        let a = if r.bernoulli(0.65) {
            best_of[c]
        } else {
            let mut w = r.below(n - 1);
            if w >= best_of[c] {
                w += 1;
            }
            w
        };
        kw.insert(t.clone(), a);
    }
    let pool = fx.tasks.len();
    let mut tasks = Vec::with_capacity(sp.n_tasks);
    let mut cl = Vec::with_capacity(sp.n_tasks);
    let mut u = Vec::with_capacity(sp.n_tasks);
    for id in 0..sp.n_tasks {
        let ti = r.below(pool);
        tasks.push(Task { id: id as u64, task_type: None, features: fx.tasks[ti].clone() });
        cl.push(if sp.shuffled { r.below(sp.k) } else { cl_of_task[ti] });
        u.push(r.f64());
    }
    Env { tasks, cl, u, p_pre, p_post, kw, agents: n }
}

impl Env {
    fn prob(&self, t: usize, a: usize) -> f64 {
        if t < self.tasks.len() / 2 { self.p_pre[self.cl[t]][a] } else { self.p_post[self.cl[t]][a] }
    }
    fn outcome(&self, t: usize, a: usize) -> f64 {
        if self.u[t] < self.prob(t, a) { 1.0 } else { 0.0 }
    }
}

fn run_policy<R: Router>(env: &Env, pol: &mut R, seed: u64) -> f64 {
    let cands: Vec<usize> = (0..env.agents).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut cum = 0.0;
    for t in 0..env.tasks.len() {
        let task = &env.tasks[t];
        let a = pol.select(task, &cands, &mut rng);
        cum += env.prob(t, a);
        pol.update(task, a, env.outcome(t, a));
    }
    cum / env.tasks.len() as f64
}

const SUBJECTS: [&str; 4] = ["compact-u16", "compact-u16-hash32", "compact-f64", "reference-f64"];

fn run_all(env: &Env, seed: u64) -> [f64; 5] {
    let n = env.agents;
    let kw = &env.kw;
    let reference = run_policy(env, &mut recommended_router(kw.clone(), n), seed);
    let c0 = run_policy(env, &mut CompactRouter::new(n, kw, CompactConfig::default()).unwrap(), seed);
    let c1 = run_policy(env, &mut CompactRouter::new(n, kw, CompactConfig { id_mode: IdMode::Hash32, ..CompactConfig::default() }).unwrap(), seed);
    let c2 = run_policy(env, &mut CompactRouterF64::new(n, kw, CompactConfig::default()).unwrap(), seed);
    // a keyword-only baseline gives scale: route to the keyword prior, else agent 0
    let mut pk = KeywordOnly { kw: kw.clone() };
    let kwb = run_policy(env, &mut pk, seed);
    [reference, c0, c1, c2, kwb]
}

struct KeywordOnly {
    kw: HashMap<String, usize>,
}
impl Router for KeywordOnly {
    fn name(&self) -> String {
        "keyword-prior".into()
    }
    fn select(&mut self, task: &Task, cands: &[usize], rng: &mut Rng) -> usize {
        let mut votes = vec![0usize; cands.len().max(64)];
        let mut any = false;
        for f in &task.features {
            if let Some(&a) = self.kw.get(f) {
                if a < votes.len() {
                    votes[a] += 1;
                    any = true;
                }
            }
        }
        if any {
            let m = votes.iter().enumerate().max_by_key(|&(i, v)| (*v, std::cmp::Reverse(i))).unwrap().0;
            if cands.contains(&m) {
                return m;
            }
        }
        cands[rng.below(cands.len())]
    }
    fn update(&mut self, _t: &Task, _a: usize, _r: f64) {}
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| "results/h10_replay.json".into());
    let quick = args.iter().any(|a| a == "--quick");
    let t0 = Instant::now();
    let (smoke, held, conf, power) = if quick { (seeds(1, 4), seeds(31, 38), seeds(61, 68), seeds(91, 100)) } else { (seeds(1, 8), seeds(31, 60), seeds(61, 90), seeds(91, 250)) };
    let mut pooled = held.clone();
    pooled.extend(conf.iter().copied());
    let splits: Vec<(&str, Vec<u64>)> = vec![("smoke", smoke), ("held", held.clone()), ("confirm", conf.clone()), ("pooled", pooled), ("power", power)];
    let mut all: Vec<u64> = splits.iter().flat_map(|s| s.1.iter().copied()).collect();
    all.sort_unstable();
    all.dedup();
    let fixtures: HashMap<&str, Fixture> = ["A", "B", "A_tail", "B_tail"].into_iter().map(|t| (t, load(t))).collect();
    let sp = specs();
    println!("h10_replay: {} worlds x {} seeds, {} workers; real task pools A {} B {}", sp.len(), all.len(), n_workers(), fixtures["A"].tasks.len(), fixtures["B"].tasks.len());
    let mut table: Vec<HashMap<u64, [f64; 5]>> = Vec::new();
    let mut vocab_info = Vec::new();
    for s in &sp {
        let fx = &fixtures[s.tag];
        let rs = par_map(&all, |&seed| run_all(&make_env(fx, s, seed), seed));
        // real-vocabulary statistics of the stream (seed 31)
        let e = make_env(fx, s, 31);
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for t in &e.tasks {
            for f in &t.features {
                *seen.entry(f.as_str()).or_insert(0) += 1;
            }
        }
        let kw_hit = e.tasks.iter().filter(|t| t.features.iter().any(|f| e.kw.contains_key(f))).count() as f64 / e.tasks.len() as f64;
        vocab_info.push(json!({"world": s.name, "distinct_features_in_stream": seen.len(), "mean_features_per_task": e.tasks.iter().map(|t| t.features.len()).sum::<usize>() as f64 / e.tasks.len() as f64,
                               "kw_prior_entries": e.kw.len(), "tasks_with_a_keyword_hit": kw_hit, "singleton_features": seen.values().filter(|&&v| v == 1).count()}));
        println!("  {} done ({:.1}s)", s.name, t0.elapsed().as_secs_f64());
        table.push(all.iter().copied().zip(rs).collect());
    }
    let pick = |wi: usize, sd: &[u64], who: usize| -> Vec<f64> { sd.iter().map(|s| table[wi][s][who]).collect() };
    let mut results = Vec::new();
    for (name, sd) in &splits {
        for (k, subj) in SUBJECTS.iter().enumerate().take(3) {
            for &margin in &[0.005f64, 0.003] {
                let eqs: Vec<_> = (0..sp.len()).map(|wi| eq_test(&pick(wi, sd, 1 + k), &pick(wi, sd, 0), margin)).collect();
                let holm = holm_adjust(&eqs.iter().map(|e| e.p_tost).collect::<Vec<_>>());
                for (wi, e) in eqs.iter().enumerate() {
                    let verdict = if holm[wi] < 0.05 && e.ci90_inside() { "CERTIFIED" } else if e.lo90 > e.margin || e.hi90 < -e.margin { "NOT equivalent" } else { "inconclusive" };
                    let rm = pick(wi, sd, 0).iter().sum::<f64>() / sd.len() as f64;
                    let sm = pick(wi, sd, 1 + k).iter().sum::<f64>() / sd.len() as f64;
                    let kwm = pick(wi, sd, 4).iter().sum::<f64>() / sd.len() as f64;
                    let mut j = e.json(holm[wi]);
                    j["split"] = json!(name);
                    j["subject"] = json!(subj);
                    j["world"] = json!(sp[wi].name);
                    j["verdict"] = json!(verdict);
                    j["mean_reference"] = json!(rm);
                    j["mean_subject"] = json!(sm);
                    j["mean_keyword_prior_only"] = json!(kwm);
                    results.push(j);
                    if (margin - 0.005).abs() < 1e-12 || *name == "power" {
                        println!("{:8} {:19} +-{:.3} {:26} n={:3} ref {:.4} subj {:.4} kwonly {:.4}  {}  holm={:.4} -> {}", name, subj, margin, sp[wi].name, e.n, rm, sm, kwm, e.fmt(), holm[wi], verdict);
                    }
                }
            }
        }
    }
    println!("\n=== PRIMARY: compact-u16 vs f64 recommended_router, pooled seeds 31-90, margin +-0.005 ===");
    let mut primary_ok = 0;
    for r in results.iter().filter(|r| r["subject"] == "compact-u16" && r["split"] == "pooled" && r["margin"] == 0.005) {
        println!("{:26} diff {:+.4} 90%CI [{:+.4},{:+.4}] 95%CI [{:+.4},{:+.4}] holm {:.4} {}", r["world"].as_str().unwrap(), r["diff_mean"].as_f64().unwrap(),
                 r["ci90"][0].as_f64().unwrap(), r["ci90"][1].as_f64().unwrap(), r["ci95"][0].as_f64().unwrap(), r["ci95"][1].as_f64().unwrap(), r["holm_p_tost"].as_f64().unwrap(), r["verdict"].as_str().unwrap());
        if r["verdict"] == "CERTIFIED" {
            primary_ok += 1;
        }
    }
    let max_abs = results.iter().filter(|r| r["subject"] == "compact-f64").map(|r| r["diff_mean"].as_f64().unwrap().abs().max(r["diff_sd"].as_f64().unwrap())).fold(0.0, f64::max);
    println!("\ncompact-f64 vs reference: max |mean diff| / sd over every cell = {max_abs:e} (must be 0)");

    // ------------------------------------------------ decisions/sec on the real-text stream (single thread, this box)
    let mut perf = Vec::new();
    for (wi, s) in sp.iter().enumerate().filter(|(_, s)| !s.shuffled) {
        let fx = &fixtures[s.tag];
        let env = make_env(fx, s, 31);
        let n = env.agents;
        let cands: Vec<usize> = (0..n).collect();
        let reps = 7;
        let mut tf: Vec<f64> = Vec::new();
        let mut tc: Vec<f64> = Vec::new();
        let mut bytes = None;
        for rep in 0..reps + 1 {
            for which in 0..2 {
                let mut rng = Rng::new(5);
                let t = Instant::now();
                let mut acc = 0usize;
                if which == 0 {
                    let mut r = recommended_router(env.kw.clone(), n);
                    for ti in 0..env.tasks.len() {
                        let a = r.select(&env.tasks[ti], &cands, &mut rng);
                        acc += a;
                        r.update(&env.tasks[ti], a, env.outcome(ti, a));
                    }
                } else {
                    let mut r = CompactRouter::new(n, &env.kw, CompactConfig::default()).unwrap();
                    for ti in 0..env.tasks.len() {
                        let a = r.select(&env.tasks[ti], &cands, &mut rng);
                        acc += a;
                        r.update(&env.tasks[ti], a, env.outcome(ti, a));
                    }
                    if rep == 0 {
                        bytes = Some(r.bytes().owned());
                    }
                }
                let ns = t.elapsed().as_secs_f64() * 1e9 / env.tasks.len() as f64;
                std::hint::black_box(acc);
                if rep > 0 {
                    if which == 0 { tf.push(ns) } else { tc.push(ns) }
                }
            }
        }
        let med = |v: &Vec<f64>| {
            let mut s = v.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        };
        let ratios: Vec<f64> = tf.iter().zip(&tc).map(|(a, b)| a / b).collect();
        let (mf, mc) = (med(&tf), med(&tc));
        println!("PERF {:18} ns/decision (route+feedback) f64 {:.0} [{:.0}..{:.0}] | compact {:.0} [{:.0}..{:.0}] | decisions/s f64 {:.0} compact {:.0} | paired f64/compact {:.2}x [{:.2}..{:.2}] | compact state {:?} B",
                 s.name, mf, tf.iter().cloned().fold(f64::MAX, f64::min), tf.iter().cloned().fold(0.0, f64::max), mc, tc.iter().cloned().fold(f64::MAX, f64::min), tc.iter().cloned().fold(0.0, f64::max),
                 1e9 / mf, 1e9 / mc, med(&ratios), ratios.iter().cloned().fold(f64::MAX, f64::min), ratios.iter().cloned().fold(0.0, f64::max), bytes);
        perf.push(json!({"world": s.name, "idx": wi, "ns_f64_med": mf, "ns_compact_med": mc, "dec_per_s_f64": 1e9 / mf, "dec_per_s_compact": 1e9 / mc, "paired_ratio_f64_over_compact_med": med(&ratios),
                         "ns_f64_all": tf, "ns_compact_all": tc, "compact_owned_bytes": bytes, "tasks": env.tasks.len(), "agents": n}));
    }
    let j = json!({"bin": "h10_replay", "protocol": {"metric": "full-run mean expected success", "margins": [0.005, 0.003], "test": "TOST alpha .05 + Holm over the 8 worlds per (subject, split, margin)",
                   "splits": splits.iter().map(|s| json!({"name": s.0, "n": s.1.len(), "first": s.1[0], "last": s.1[s.1.len()-1]})).collect::<Vec<_>>(), "primary": "compact-u16, pooled 31-90, +-0.005"},
                   "worlds": sp.iter().map(|s| json!({"name": s.name, "fixture": s.tag, "k_clusters": s.k, "agents": s.agents, "tasks": s.n_tasks, "shuffled_control": s.shuffled})).collect::<Vec<_>>(),
                   "vocab_info": vocab_info, "results": results, "perf": perf, "compact_f64_max_abs_diff": max_abs, "primary_certified_worlds": primary_ok, "runtime_s": t0.elapsed().as_secs_f64()});
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out} ({:.1}s)", t0.elapsed().as_secs_f64());
}
