//! B14: merge learned ROUTER STATE across machines (O(features x agents) bytes) instead of replaying the union journal (O(outcomes)). Is it equivalent (or better)?
//! Arms for session 2 on machine 0: cold | own journal replay | UNION journal replay (shipped) | STATE-MEAN (per-feature mean of the machines' weight rows / baselines over the machines that
//! have the feature) | STATE-SUM (rows summed over machines, rescaled to row_cap). Machines: HOMOGENEOUS (all 8 types, 250 tasks each) and SPECIALISED (2 types each, as B12).
//! Routers: plain Hebbian (recommended_inner) and the Guarded `recommended` (inner state only; keyword gate stays cold, R34 recipe).
//! Metric: mean expected success over the first H=100/250/500 tasks of session 2, paired over seeds; Holm over the family.
use neural_mesh::defaults::{recommended_inner, recommended_router};
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::router::{Rng, Router};
use neural_mesh::sim::{Scenario, World, WorldConfig, TYPES};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue};
use neural_mesh::store::{MeshStore, Outcome};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::BTreeMap;

const N2: usize = 2000;
const HS: [usize; 3] = [100, 250, 500];
#[derive(Clone, Copy, PartialEq)]
enum Rk {
    Hebb,
    Rec,
}
fn rk_name(r: Rk) -> &'static str {
    if r == Rk::Hebb { "hebbian-robust" } else { "recommended(inner)" }
}
static NAMES: [&str; 6] = ["a0", "a1", "a2", "a3", "a4", "a5"];
fn idx(n: &str) -> Option<usize> {
    NAMES.iter().position(|x| *x == n)
}
fn cfg(name: &str, n: usize) -> WorldConfig {
    WorldConfig { n_tasks: n, scenario: if name == "fixed" { Scenario::Fixed } else { Scenario::Random }, ..WorldConfig::default() }
}
fn meanh(ex: &[f64], h: usize) -> f64 {
    ex[..h.min(ex.len())].iter().sum::<f64>() / h.min(ex.len()) as f64
}
fn run(world: &World, router: &mut dyn Router, n: usize, types: Option<&[usize]>, sess: &str, seed: u64) -> (Vec<f64>, Vec<Outcome>) {
    let mut rng = Rng::new(seed);
    let cands: Vec<usize> = (0..6).collect();
    let (mut ex, mut js) = (vec![], vec![]);
    for t in 0..world.len() {
        if ex.len() >= n {
            break;
        }
        let st = world.task(t);
        if let Some(ts) = types {
            if !ts.contains(&st.type_idx) {
                continue;
            }
        }
        let a = router.select(&st.task, &cands, &mut rng);
        ex.push(world.success_prob(t, &[a]));
        let r = world.outcome(t, &[a]);
        router.update(&st.task, a, r);
        js.push(Outcome { id: format!("{sess}:{t}"), session: sess.into(), seq: t as u64, ts: t as u64, world: "sim".into(), features: st.task.features.clone(), task_type: st.task.task_type.clone(), agent: NAMES[a].into(), reward: r });
    }
    (ex, js)
}
fn tmp(tag: &str) -> MeshStore {
    let p = std::env::temp_dir().join(format!("b14_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    MeshStore::open(p).unwrap()
}
fn state(r: &HebbianRouter) -> Value {
    serde_json::to_value(r).unwrap()
}
/// merge router states: mode 0 = mean over machines that have the row, 1 = sum then rescale to row_cap
fn merge(states: &[Value], mode: u8, row_cap: f64) -> HebbianRouter {
    let mut out = states[0].clone();
    let mut rows: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();
    let mut bases: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut upd = 0u64;
    for s in states {
        for (k, v) in s["weights"].as_object().unwrap() {
            rows.entry(k.clone()).or_default().push(v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect());
        }
        for (k, v) in s["baselines"].as_object().unwrap() {
            bases.entry(k.clone()).or_default().push(v.as_f64().unwrap());
        }
        upd += s["updates"].as_u64().unwrap_or(0);
    }
    let mut w = serde_json::Map::new();
    for (k, rs) in rows {
        let n = rs[0].len();
        let mut r = vec![0.0; n];
        for x in &rs {
            for i in 0..n {
                r[i] += x[i];
            }
        }
        if mode == 0 {
            r.iter_mut().for_each(|v| *v /= rs.len() as f64);
        } else {
            let s: f64 = r.iter().sum();
            if s > row_cap {
                r.iter_mut().for_each(|v| *v *= row_cap / s);
            }
        }
        w.insert(k, json!(r));
    }
    let mut b = serde_json::Map::new();
    for (k, v) in bases {
        b.insert(k, json!(v.iter().sum::<f64>() / v.len() as f64));
    }
    out["weights"] = Value::Object(w);
    out["baselines"] = Value::Object(b);
    out["updates"] = json!(upd);
    serde_json::from_value(out).unwrap()
}
/// arms: [cold, own, union-replay, state-mean, state-sum] x H
fn trial(seed: u64, wname: &str, rk: Rk, m: usize, spec: bool) -> [f64; 15] {
    let w2 = World::new_stream(seed, seed + 5000, cfg(wname, N2));
    let tag = format!("{seed}_{wname}_{}_{m}_{spec}", rk_name(rk).len());
    let union_s = tmp(&format!("u_{tag}"));
    let own_s = tmp(&format!("o_{tag}"));
    let mut states: Vec<Value> = vec![];
    let mut routers: Vec<HebbianRouter> = vec![];
    for k in 0..m {
        let types = [(2 * k) % TYPES.len(), (2 * k + 1) % TYPES.len()];
        let wk = World::new_stream(seed, seed + 1000 + 17 * k as u64, cfg(wname, 250 * 8));
        let mut r = HebbianRouter::new(6, recommended_inner());
        let (_, js) = run(&wk, &mut r, 250, if spec { Some(&types) } else { None }, &format!("m{k}"), seed * 31 + 7 + k as u64);
        union_s.append_outcomes(&js).unwrap();
        if k == 0 {
            own_s.append_outcomes(&js).unwrap();
        }
        states.push(state(&r));
        routers.push(r);
    }
    let mk_cold = || -> Box<dyn Router + Send> {
        match rk {
            Rk::Hebb => Box::new(HebbianRouter::new(6, recommended_inner())),
            Rk::Rec => Box::new(recommended_router(w2.keyword_map_for(), 6)),
        }
    };
    let with_inner = |h: HebbianRouter| -> Box<dyn Router + Send> {
        match rk {
            Rk::Hebb => Box::new(h),
            Rk::Rec => {
                let mut g = recommended_router(w2.keyword_map_for(), 6);
                g.inner = h;
                Box::new(g)
            }
        }
    };
    let replayed = |s: &MeshStore| -> Box<dyn Router + Send> {
        let mut h = HebbianRouter::new(6, recommended_inner());
        s.warm_replay(&mut h, &idx, None, None).unwrap();
        with_inner(h)
    };
    if seed == 1 && wname == "random" && rk == Rk::Hebb {
        let merged = HebbianRouter::merge_sum(&routers.iter().collect::<Vec<_>>());
        let jb = std::fs::read_to_string(union_s.root().join("router/journal.jsonl")).map(|x| x.len()).unwrap_or(0);
        eprintln!("SIZE m={m} spec={spec}: merged state json {} B ({} features) vs union journal {} B ({} outcomes)", serde_json::to_string(&merged).unwrap().len(), state(&merged)["weights"].as_object().unwrap().len(), jb, 250 * m);
    }
    let cap = recommended_inner().row_cap;
    let mut arms: Vec<Box<dyn Router + Send>> = vec![mk_cold(), replayed(&own_s), replayed(&union_s), with_inner(merge(&states, 0, cap)), with_inner(HebbianRouter::merge_sum(&routers.iter().collect::<Vec<_>>()))];
    let mut o = [0.0; 15];
    for (ai, r) in arms.iter_mut().enumerate() {
        let (ex, _) = run(&w2, r.as_mut(), N2, None, "s", seed * 31 + 2);
        for (hi, h) in HS.iter().enumerate() {
            o[ai * 3 + hi] = meanh(&ex, *h);
        }
    }
    let _ = std::fs::remove_dir_all(union_s.root());
    let _ = std::fs::remove_dir_all(own_s.root());
    o
}
fn parse_range(s: &str) -> Vec<u64> {
    let (a, b) = s.split_once('-').unwrap();
    (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
}
type Row = (&'static str, Rk, bool, usize, usize, f64, f64, f64, f64, f64, Vec<f64>, Vec<f64>, Vec<f64>);
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str, d: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone()).unwrap_or(d.into());
    let seeds = parse_range(&get("--seeds", "31-60"));
    let out_path = get("--out", "results/b14_statemerge.json");
    let zero = vec![0.0; seeds.len()];
    let mut rows: Vec<Row> = vec![];
    for w in ["random", "fixed"] {
        for rk in [Rk::Hebb, Rk::Rec] {
            for spec in [false, true] {
                for m in [2usize, 4] {
                    let res: Vec<[f64; 15]> = seeds.par_iter().map(|&s| trial(s, w, rk, m, spec)).collect();
                    for (hi, h) in HS.iter().enumerate() {
                        let col = |c: usize| -> Vec<f64> { res.iter().map(|r| r[c * 3 + hi]).collect() };
                        let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
                        let diff = |x: &Vec<f64>, y: &Vec<f64>| -> Vec<f64> { x.iter().zip(y).map(|(a, b)| a - b).collect() };
                        let (cold, own, un, sm, ss) = (col(0), col(1), col(2), col(3), col(4));
                        rows.push((w, rk, spec, m, *h, mean(&cold), mean(&own), mean(&un), mean(&sm), mean(&ss), diff(&sm, &un), diff(&ss, &un), diff(&un, &cold)));
                    }
                }
            }
        }
    }
    let ps: Vec<f64> = rows.iter().flat_map(|r| [paired_t_pvalue(&r.10, &zero), paired_t_pvalue(&r.11, &zero)]).collect();
    let adj = holm_adjust(&ps);
    let mut out = vec![];
    for (i, r) in rows.iter().enumerate() {
        let (a1, a2) = (paired_diff(&r.10, &zero), paired_diff(&r.11, &zero));
        println!(
            "{:6} {:19} {} M={} H{:<4} cold {:.4} own {:.4} union-replay {:.4} state-mean {:.4} state-sum {:.4} | mean-union {:+.4} [{:+.4},{:+.4}] Holm {:.1e} | sum-union {:+.4} [{:+.4},{:+.4}] Holm {:.1e} | union-cold {:+.4}",
            r.0, rk_name(r.1), if r.2 { "spec" } else { "homo" }, r.3, r.4, r.5, r.6, r.7, r.8, r.9, a1.diff.mean, a1.diff.lo, a1.diff.hi, adj[2 * i], a2.diff.mean, a2.diff.lo, a2.diff.hi, adj[2 * i + 1], paired_diff(&r.12, &zero).diff.mean
        );
        out.push(json!({"world": r.0, "router": rk_name(r.1), "spec": r.2, "m": r.3, "h": r.4, "cold": r.5, "own": r.6, "union": r.7, "mean": r.8, "sum": r.9, "mean_minus_union": a1.diff.mean, "mu_lo": a1.diff.lo, "mu_hi": a1.diff.hi, "p_mean": adj[2 * i], "sum_minus_union": a2.diff.mean, "su_lo": a2.diff.lo, "su_hi": a2.diff.hi, "p_sum": adj[2 * i + 1]}));
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
