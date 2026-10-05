//! H9 (R5-A): warm-start priors for the router from previous deployments.
//!
//! Candidates (all opt-in, new types in h9_common; no src edits):
//!   pop-ts       pooled Thompson sampling whose per-(feature,agent) cells start from a population prior fitted on
//!                a POOL of previous deployments (hierarchical shrinkage: pseudo-count from between-deployment consistency)
//!   pop-ts+conf  same + trust weight per feature learned online (prequential prior-vs-data log score): a WRONG prior is overridden
//!   pop-map      guarded-safe whose keyword map is corrected by the pool where the pool shows a clearly better agent,
//!                prior strength = pool-measured accuracy
//!   kw-cal       guarded-safe with its keyword-prior strength (kw_a,kw_b) calibrated from the pool's measured keyword accuracy
//! Protocol: tune seeds 1-8 on a subset of worlds (the rest are never seen by the tuner), held-out --held (default 31-60),
//! confirm 61-90 (same frozen tuning, deterministic), paired 95% t-CI vs best fair baseline per (world,metric), Holm per family.
//! All numbers are SYNTHETIC-until-validated-on-real-traffic.
#![allow(dead_code)]

#[path = "f3_common/mod.rs"]
mod f3;
#[path = "h9_common/mod.rs"]
mod h9;

use std::collections::BTreeMap;

use f3::{par_map, tost_p};
use h9::*;
use neural_mesh::baselines::PooledThompsonRouter;
use neural_mesh::router::Router;
use neural_mesh::stats::{holm_adjust, paired_t_pvalue, summarize};
use serde_json::{json, Value};

struct Args {
    held: (u64, u64),
    out: String,
    quick: bool,
    show: bool,
    k: usize,
    expk: bool,
    time: bool,
}

fn parse_args() -> Args {
    let mut a = Args { held: (31, 60), out: "results/h9_priors.json".into(), quick: false, show: false, k: 6, expk: false, time: false };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < v.len() {
        match v[i].as_str() {
            "--held" => {
                let s: Vec<u64> = v[i + 1].split('-').map(|x| x.parse().unwrap()).collect();
                a.held = (s[0], s[1]);
                i += 1;
            }
            "--out" => {
                a.out = v[i + 1].clone();
                i += 1;
            }
            "--k" => {
                a.k = v[i + 1].parse().unwrap();
                i += 1;
            }
            "--quick" => a.quick = true,
            "--show" => a.show = true,
            "--expk" => a.expk = true,
            "--time" => a.time = true,
            x => panic!("unknown arg {x}"),
        }
        i += 1;
    }
    a
}

// ------------------------------------------------------------------------------------------------ selftests
fn selftest() {
    // (1) empty-prior PopTs (with and without trust) == PooledThompsonRouter(gamma 1), bit-identical curves
    let spec = &specs()[0];
    let w = make_world(&spec.eval, 3);
    let mut a = PooledThompsonRouter::new(w.n_agents, 1.0);
    let ca = run_curve(&w, &mut a, 3);
    for tr in [None, Some(TrustCfg { w0: 0.5, cf: 1.0, cg: 0.5, delta: 0.97 })] {
        let mut b = PopTs::new("t", w.n_agents, PopPrior::empty(), tr);
        let cb = run_curve(&w, &mut b, 3);
        assert_eq!(ca, cb, "PopTs(empty prior) must equal pooled-ts");
    }
    // (1b) empty-prior PopGuarded == defaults::recommended_router (bit-identical), with and without trust
    for si in [0usize, 5, 10] {
        let w = make_world(&specs()[si].eval, 4);
        let mut g = neural_mesh::defaults::recommended_router(w.kw.clone(), w.n_agents);
        let cg = run_curve(&w, &mut g, 4);
        for tr in [None, Some(TrustCfg { w0: 0.5, cf: 1.0, cg: 0.5, delta: 0.97 })] {
            let mut pg = PopGuarded::new(w.kw.clone(), w.n_agents, PopPrior::empty(), tr).with_informed(2.0);
            assert_eq!(cg, run_curve(&w, &mut pg, 4), "PopGuarded(empty prior) must equal guarded-safe");
        }
    }
    // (2) fit_prior: consistent cells -> cap, inconsistent -> 0, thin -> none
    let mk = |rates: &[f64]| -> Vec<Dep> {
        rates
            .iter()
            .map(|&r| {
                let mut d = Dep::default();
                d.cells.insert("f".into(), vec![(100.0 * r, 100.0), (0.0, 0.0)]);
                d
            })
            .collect()
    };
    let fit = FitCfg { m_max: 16.0, m1: 2.0, n_min: 5.0, shrink: true, cov_min: 0.0 };
    let p = fit_prior(&mk(&[0.8, 0.8, 0.8, 0.8]), 2, fit);
    assert!(p.cells["f"][0].m > 12.0 && p.cells["f"][1].m == 0.0, "consistent: {:?}", p.cells["f"][0]);
    let p = fit_prior(&mk(&[0.1, 0.9, 0.2, 0.8]), 2, fit);
    assert!(p.cells["f"][0].m < 2.0, "inconsistent: {:?}", p.cells["f"][0]);
    let p = fit_prior(&mk(&[0.8]), 2, fit);
    assert!((p.cells["f"][0].m - 2.0).abs() < 1e-9);
    let p = fit_prior(&mk(&[0.8, 0.8]), 2, FitCfg { shrink: false, ..fit });
    assert!((p.cells["f"][0].m - 16.0).abs() < 1e-9);
    // coverage: a cell seen in 1 of 4 deployments is dropped at cov_min .5, kept (scaled) at 0
    let mut d4 = mk(&[0.8, 0.8, 0.8, 0.8]);
    for d in d4.iter_mut().skip(1) {
        d.cells.insert("f".into(), vec![(0.0, 0.0), (0.0, 0.0)]);
    }
    assert!(fit_prior(&d4, 2, FitCfg { cov_min: 0.5, ..fit }).cells["f"][0].m == 0.0);
    assert!((fit_prior(&d4, 2, FitCfg { cov_min: 0.0, ..fit }).cells["f"][0].m - 0.5).abs() < 1e-9);
    // (3) a wrong prior loses trust fast, a right one keeps it (n=60 observations, truth p=.3 vs prior .85)
    let mut cells = std::collections::HashMap::new();
    cells.insert("f".to_string(), vec![Cell { s: 0.0, n: 100.0, mu: 0.85, m: 8.0 }]);
    let prior = PopPrior { cells, kw_acc: 0.8, n_deps: 4, agent_mu: vec![] };
    let mut r = PopTs::new("t", 1, prior.clone(), Some(TrustCfg { w0: 0.8, cf: 1.0, cg: 0.0, delta: 0.97 }));
    let t = neural_mesh::router::Task { id: 0, task_type: None, features: vec!["f".into()] };
    let mut rng = neural_mesh::router::Rng::new(5);
    for _ in 0..40 {
        r.update(&t, 0, if rng.bernoulli(0.3) { 1.0 } else { 0.0 });
    }
    assert!(r.feat_trust("f") < 0.2, "wrong prior trust {}", r.feat_trust("f"));
    let mut r = PopTs::new("t", 1, prior, Some(TrustCfg { w0: 0.8, cf: 1.0, cg: 0.0, delta: 0.97 }));
    for _ in 0..8 {
        r.update(&t, 0, if rng.bernoulli(0.85) { 1.0 } else { 0.0 });
    }
    assert!(r.feat_trust("f") > 0.5, "right prior trust {}", r.feat_trust("f"));
    println!("selftest ok (PopTs(empty)==pooled-ts bit-identical; shrinkage + trust behave)");
}

// ------------------------------------------------------------------------------------------------ grids
fn fit(m_max: f64, m1: f64, cov_min: f64) -> FitCfg {
    FitCfg { m_max, m1, n_min: 5.0, shrink: true, cov_min }
}

struct Grids {
    eps: Vec<Pol>,
    pts: Vec<Pol>,
    popts_conf: Vec<Pol>,
    popguard: Vec<Pol>,
    popguard_conf: Vec<Pol>,
    kwcal: Vec<Pol>,
    popmap: Vec<Pol>,
}

fn tc(w0: f64, cg: f64) -> TrustCfg {
    TrustCfg { w0, cf: 1.0, cg, delta: 0.97 }
}

fn grids(quick: bool) -> Grids {
    let (gs, es) = if quick { (vec![0.9, 1.0], vec![0.0, 0.05]) } else { (vec![0.9, 0.95, 1.0], vec![0.0, 0.02, 0.05]) };
    let eps = gs.iter().flat_map(|&g| es.iter().map(move |&e| Pol::Eps { g, e })).collect();
    let pts = [0.9, 0.97, 1.0].iter().map(|&g| Pol::Pts { g }).collect();
    let covs: Vec<f64> = if quick { vec![0.5] } else { vec![0.5, 0.8] };
    let mut popts_conf = vec![];
    for &m in if quick { &[16.0][..] } else { &[8.0, 16.0, 32.0][..] } {
        for &cov in &covs {
            for &cg in &[0.0, 0.5] {
                for &ma in &[0.0, 3.0] {
                    popts_conf.push(Pol::PopTs { fit: fit(m, 2.0, cov), conf: Some(tc(0.8, cg)), m_agent: ma });
                }
            }
        }
    }
    let mut popguard = vec![];
    for &m in if quick { &[8.0][..] } else { &[4.0, 8.0, 16.0][..] } {
        for &m1 in if quick { &[2.0][..] } else { &[1.0, 3.0][..] } {
            for &cov in &covs {
                for &inf in &[0.0, 2.0, 4.0, 8.0] {
                    popguard.push(Pol::PopGuard { fit: fit(m, m1, cov), conf: None, inf });
                }
            }
        }
    }
    let mut popguard_conf = vec![];
    for &m in if quick { &[8.0][..] } else { &[8.0, 16.0][..] } {
        for &m1 in if quick { &[2.0][..] } else { &[1.0, 3.0][..] } {
            for &cov in &covs {
                for &w0 in &[0.5, 0.8] {
                    for &cg in &[0.0, 0.5] {
                        for &inf in &[2.0, 4.0, 8.0] {
                            popguard_conf.push(Pol::PopGuard { fit: fit(m, m1, cov), conf: Some(tc(w0, cg)), inf });
                        }
                    }
                }
            }
        }
    }
    let mut kwcal = vec![];
    for &m in &[3.0, 5.0, 8.0] {
        for &tau in &[0.6, 0.7] {
            kwcal.push(Pol::KwCal { m, tau });
        }
    }
    let mut popmap = vec![];
    for &nc in &[10.0, 30.0] {
        for &d in &[0.05, 0.15] {
            for &m in &[3.0, 5.0] {
                for &cov in &covs {
                    popmap.push(Pol::PopMap { fit: fit(8.0, 2.0, cov), n_cell: nc, delta: d, m });
                }
            }
        }
    }
    Grids { eps, pts, popts_conf, popguard, popguard_conf, kwcal, popmap }
}

fn fixed_base() -> Vec<Pol> {
    vec![Pol::Keyword, Pol::Hebr, Pol::Warm { k: 1.0 }, Pol::Warm { k: 10.0 }, Pol::Guarded, Pol::KwTs { m: 4.0 }]
}

struct Picks {
    eps: Pol,
    pts: Pol,
    popts_conf: Pol,
    popguard: Pol,
    popguard_conf: Pol,
    kwcal: Pol,
    popmap: Pol,
    log: Vec<Value>,
}

/// Regret-averse objective vs the best baseline per (world, early metric): mean margin + mean negative margin.
fn tune(specs: &[Spec], k: usize, quick: bool) -> Picks {
    let g = grids(quick);
    let mut pols = fixed_base();
    let nfix = pols.len();
    let mut ranges: Vec<(&str, usize, usize)> = vec![];
    for (name, v) in [
        ("eps", &g.eps),
        ("pts", &g.pts),
        ("popts_conf", &g.popts_conf),
        ("popguard", &g.popguard),
        ("popguard_conf", &g.popguard_conf),
        ("kwcal", &g.kwcal),
        ("popmap", &g.popmap),
    ] {
        ranges.push((name, pols.len(), pols.len() + v.len()));
        pols.extend(v.iter().cloned());
    }
    let tw: Vec<&Spec> = specs.iter().filter(|s| s.tune).collect();
    let seeds: Vec<u64> = (1..=8).collect();
    let jobs: Vec<(usize, u64)> = (0..tw.len()).flat_map(|wi| seeds.iter().map(move |&s| (wi, s))).collect();
    eprintln!("[tune] {} policies x {} tune worlds x {} seeds", pols.len(), tw.len(), seeds.len());
    let res = par_map(&jobs, |&(wi, s)| run_all(tw[wi], s, k, &pols));
    // means[wi][pi][m]
    let mut means = vec![vec![[0.0; NM]; pols.len()]; tw.len()];
    for (j, &(wi, _)) in jobs.iter().enumerate() {
        for pi in 0..pols.len() {
            for m in 0..NM {
                means[wi][pi][m] += res[j][pi][m] / seeds.len() as f64;
            }
        }
    }
    let early = [0usize, 1, 2, 3];
    let score = |pi: usize| -> f64 {
        let mut s = 0.0;
        for wi in 0..tw.len() {
            for &m in &early {
                s += means[wi][pi][m];
            }
        }
        s / (tw.len() * early.len()) as f64
    };
    let argbest = |lo: usize, hi: usize, f: &dyn Fn(usize) -> f64| -> usize { (lo..hi).max_by(|&a, &b| f(a).partial_cmp(&f(b)).unwrap()).unwrap() };
    let rg = |n: &str| ranges.iter().find(|r| r.0 == n).map(|r| (r.1, r.2)).unwrap();
    let eps_i = argbest(rg("eps").0, rg("eps").1, &score);
    let pts_i = argbest(rg("pts").0, rg("pts").1, &score);
    let mut base_idx: Vec<usize> = (0..nfix).collect();
    base_idx.push(eps_i);
    base_idx.push(pts_i);
    let obj = |pi: usize| -> f64 {
        let (mut sm, mut sn, mut cnt) = (0.0, 0.0, 0.0);
        for wi in 0..tw.len() {
            for &m in &early {
                let best = base_idx.iter().map(|&b| means[wi][b][m]).fold(f64::NEG_INFINITY, f64::max);
                let mg = means[wi][pi][m] - best;
                sm += mg;
                sn += mg.min(0.0);
                cnt += 1.0;
            }
        }
        (sm + sn) / cnt
    };
    let mut log = vec![];
    let mut pick = |name: &str| -> Pol {
        let (lo, hi) = rg(name);
        let mut order: Vec<usize> = (lo..hi).collect();
        order.sort_by(|&a, &b| obj(b).partial_cmp(&obj(a)).unwrap());
        println!("[tune] {name}: top configs by regret-averse objective (J = mean margin + mean negative margin vs best baseline, early H, tune worlds, seeds 1-8)");
        for &i in order.iter().take(4) {
            println!("        J={:+.5}  {}", obj(i), pols[i].label());
        }
        log.push(json!({"family": name, "picked": pols[order[0]].label(), "J": obj(order[0]), "n_configs": hi - lo,
            "top": order.iter().take(5).map(|&i| json!({"cfg": pols[i].label(), "J": obj(i)})).collect::<Vec<_>>()}));
        pols[order[0]].clone()
    };
    let popts_conf = pick("popts_conf");
    let popguard = pick("popguard");
    let popguard_conf = pick("popguard_conf");
    let kwcal = pick("kwcal");
    let popmap = pick("popmap");
    println!("[tune] baselines tuned on mean early score: eps = {}, pts = {}", pols[eps_i].label(), pols[pts_i].label());
    log.push(json!({"family": "baselines", "eps": pols[eps_i].label(), "pts": pols[pts_i].label(), "tune_worlds": tw.iter().map(|s| s.name).collect::<Vec<_>>()}));
    Picks { eps: pols[eps_i].clone(), pts: pols[pts_i].clone(), popts_conf, popguard, popguard_conf, kwcal, popmap, log }
}

// ------------------------------------------------------------------------------------------------ analysis
struct Test {
    wi: usize,
    ci: usize,
    m: usize,
    mean: f64,
    lo: f64,
    hi: f64,
    p: f64,
    fam: &'static str,
    adj: f64,
}

fn signed_diff(res: &[Vec<[f64; NM]>], a: usize, b: usize, m: usize) -> Vec<f64> {
    let sg = if higher_better(m) { 1.0 } else { -1.0 };
    (0..res[a].len()).map(|i| sg * (res[a][i][m] - res[b][i][m])).collect()
}

fn vals(res: &[Vec<[f64; NM]>], p: usize, m: usize) -> Vec<f64> {
    res[p].iter().map(|r| r[m]).collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn best_base(res: &[Vec<[f64; NM]>], base: &[usize], m: usize) -> usize {
    let sg = if higher_better(m) { 1.0 } else { -1.0 };
    *base.iter().max_by(|&&a, &&b| (sg * mean(&vals(res, a, m))).partial_cmp(&(sg * mean(&vals(res, b, m)))).unwrap()).unwrap()
}

fn mk_test(d: &[f64], wi: usize, ci: usize, m: usize, fam: &'static str) -> Test {
    let s = summarize(d);
    let z = vec![0.0; d.len()];
    Test { wi, ci, m, mean: s.mean, lo: s.lo, hi: s.hi, p: paired_t_pvalue(d, &z), fam, adj: 1.0 }
}

fn holm_family(tests: &mut [Test], fam: &str) {
    let idx: Vec<usize> = (0..tests.len()).filter(|&i| tests[i].fam == fam).collect();
    let p: Vec<f64> = idx.iter().map(|&i| tests[i].p).collect();
    let adj = holm_adjust(&p);
    for (k, &i) in idx.iter().enumerate() {
        tests[i].adj = adj[k];
    }
}

fn flag(t: &Test) -> char {
    if t.adj < 0.05 && t.lo > 0.0 {
        'W'
    } else if t.adj < 0.05 && t.hi < 0.0 {
        'L'
    } else if t.lo > 0.0 {
        'w'
    } else if t.hi < 0.0 {
        'l'
    } else {
        ' '
    }
}

fn main() {
    let args = parse_args();
    selftest();
    let specs = specs();
    let k = args.k;
    if args.show {
        for (i, s) in specs.iter().enumerate() {
            let w = make_world(&s.eval, 1);
            let deps = build_pool(&s.pool, k, 1);
            let pn = deps[0].cells.values().next().map(|v| v.len()).unwrap();
            let pr = fit_prior(&deps, pn, fit(8.0, 2.0, 0.5));
            let (sp, sk, so, cov) = prior_diag(&w, &pr, 250);
            let ncell = pr.cells.values().flat_map(|r| r.iter()).filter(|c| c.m > 0.0).count();
            let mm: f64 = pr.cells.values().flat_map(|r| r.iter()).filter(|c| c.m > 0.0).map(|c| c.m).sum::<f64>() / ncell.max(1) as f64;
            let (cm, cacc) = corrected_map(&w.kw, &pr, 10.0, 0.05, 0.6);
            let changed = cm.iter().filter(|(f, a)| w.kw.get(*f) != Some(*a)).count();
            println!(
                "{i:2} {:<22} role={:?} tune={} agents={} | {}\n     seed1 first-250 STATIC policies: prior-argmax {:.3} keyword {:.3} oracle {:.3} | prior coverage {:.2} cells(m>0) {} mean m {:.1} kw_acc(pool) {:.2} map-corrections {} acc(corrected map) {:.2}",
                s.name, s.role, s.tune, w.n_agents, s.note, sp, sk, so, cov, ncell, mm, pr.kw_acc, changed, cacc
            );
        }
        return;
    }
    if args.time {
        // single-thread cost: ns per task (select+update) and approximate learned-state bytes, world B1 seed 1
        let spec = specs.iter().find(|x| x.name.starts_with("B1")).unwrap();
        let w = make_world(&spec.eval, 1);
        let deps = build_pool(&spec.pool, k, 1);
        let g = grids(false);
        let picks_dummy: Vec<(String, Pol)> = vec![
            ("guarded-safe".into(), Pol::Guarded),
            ("warm(hebr,k1)".into(), Pol::Warm { k: 1.0 }),
            ("pooled-ts".into(), Pol::Pts { g: 1.0 }),
            ("pop-guard+conf (default pick)".into(), g.popguard_conf.iter().find(|p| p.label().contains("M16,m1=1,cov0.5,inf=8][w0=0.8,cg=0.5")).unwrap().clone()),
        ];
        for (n, p) in picks_dummy {
            let reps = 30;
            let t0 = std::time::Instant::now();
            for r in 0..reps {
                let mut rt = p.build(&w, &deps);
                let _ = run_curve(&w, rt.as_mut(), r);
            }
            let dt = t0.elapsed().as_secs_f64();
            // build cost separately (includes prior fit)
            let t1 = std::time::Instant::now();
            for _ in 0..reps {
                let _ = p.build(&w, &deps);
            }
            let bt = t1.elapsed().as_secs_f64() / reps as f64;
            println!("{n:<34} {:.0} ns/task (select+update, incl. build {:.2} ms / {} tasks)", (dt / (reps as f64 * w.len as f64)) * 1e9, bt * 1e3, w.len);
        }
        println!("state: guarded row = n_agents*(s,f) f64 = 16 B*n; pop-guard row = data+prior = 32 B*n (+16 B trust/feature); pool prior table (shared, static) = features*n_agents*32 B");
        return;
    }
    // ---- tune
    let picks = tune(&specs, k, args.quick);
    let nosh = |p: &Pol| -> Pol {
        match p {
            Pol::PopGuard { fit: f, conf, inf } => Pol::PopGuard { fit: FitCfg { shrink: false, ..*f }, conf: *conf, inf: *inf },
            _ => unreachable!(),
        }
    };
    let popts_plain = match &picks.popts_conf {
        Pol::PopTs { fit: f, m_agent, .. } => Pol::PopTs { fit: *f, conf: None, m_agent: *m_agent },
        _ => unreachable!(),
    };
    let mut pols = fixed_base();
    pols.push(picks.eps.clone());
    pols.push(picks.pts.clone());
    // order: 0 keyword,1 hebr,2 warm1,3 warm10,4 guarded,5 kw-ts,6 eps*,7 pts*
    let nb = pols.len();
    let base: Vec<usize> = (0..nb).collect();
    pols.push(picks.popts_conf.clone());
    pols.push(picks.popguard.clone());
    pols.push(picks.popguard_conf.clone());
    pols.push(picks.kwcal.clone());
    let cand: Vec<usize> = (nb..nb + 4).collect();
    pols.push(popts_plain);
    pols.push(picks.popmap.clone());
    pols.push(nosh(&picks.popguard));
    pols.push(nosh(&picks.popguard_conf));
    let abl: Vec<usize> = (nb + 4..nb + 8).collect();
    let cold = |ci: usize| -> usize {
        match ci - nb {
            0 => 7,
            _ => 4,
        }
    };
    let labels: Vec<String> = pols.iter().map(|p| p.label()).collect();
    println!("[eval] policies:");
    for (i, l) in labels.iter().enumerate() {
        println!("   {i:2} {l}");
    }
    // ---- evaluate
    let seeds: Vec<u64> = (args.held.0..=args.held.1).collect();
    let jobs: Vec<(usize, u64)> = (0..specs.len()).flat_map(|wi| seeds.iter().map(move |&s| (wi, s))).collect();
    eprintln!("[eval] {} policies x {} worlds x {} seeds (K={k})", pols.len(), specs.len(), seeds.len());
    let raw = par_map(&jobs, |&(wi, s)| run_all(&specs[wi], s, k, &pols));
    // res[wi][pi] = Vec over seeds
    let mut res: Vec<Vec<Vec<[f64; NM]>>> = vec![vec![vec![]; pols.len()]; specs.len()];
    for (j, &(wi, _)) in jobs.iter().enumerate() {
        for pi in 0..pols.len() {
            res[wi][pi].push(raw[j][pi]);
        }
    }
    // ---- tests
    let hm = [0usize, 1, 2, 3, 4];
    let wm = [5usize, 6, 7, 8];
    let xm = [9usize, 10, 11];
    let mut tests: Vec<Test> = vec![];
    for wi in 0..specs.len() {
        let treat = specs[wi].role == Role::Treat;
        for (cn, &ci) in cand.iter().enumerate() {
            let _ = cn;
            for &m in hm.iter().chain(wm.iter()).chain(xm.iter()) {
                let b = best_base(&res[wi], &base, m);
                let d = signed_diff(&res[wi], ci, b, m);
                let fam = if specs[wi].name.starts_with("B9") {
                    "B9"
                } else if hm.contains(&m) {
                    if treat { "G" } else { "L" }
                } else if wm.contains(&m) {
                    if treat { "F" } else { "FL" }
                } else {
                    "X"
                };
                tests.push(mk_test(&d, wi, ci, m, fam));
            }
        }
    }
    for fam in ["G", "L", "F", "FL", "X", "B9"] {
        holm_family(&mut tests, fam);
    }
    // matched cold counterparts (all worlds)
    let mut dtests: Vec<Test> = vec![];
    for wi in 0..specs.len() {
        for &ci in &cand {
            for &m in &hm {
                let d = signed_diff(&res[wi], ci, cold(ci), m);
                dtests.push(mk_test(&d, wi, ci, m, "D"));
            }
        }
    }
    holm_family(&mut dtests, "D");
    // TOST equivalence (+-0.01) vs best baseline in control worlds
    let mut etests: Vec<(usize, usize, usize, f64, f64)> = vec![];
    for wi in 0..specs.len() {
        if specs[wi].role != Role::Control || specs[wi].name.starts_with("B9") {
            continue;
        }
        for &ci in &cand {
            for &m in &hm {
                let b = best_base(&res[wi], &base, m);
                let d = signed_diff(&res[wi], ci, b, m);
                etests.push((wi, ci, m, tost_p(&d, 0.01), mean(&d)));
            }
        }
    }
    let eadj = holm_adjust(&etests.iter().map(|t| t.3).collect::<Vec<_>>());

    // ---- print
    println!("\n================= RESULTS (held seeds {}-{}, K={}, n_seeds={}) =================", args.held.0, args.held.1, k, seeds.len());
    println!("legend: diffs are candidate minus BEST FAIR BASELINE per (world,metric) (positive = candidate better; reg/ttt sign-flipped), baseline picked post hoc by held-out mean. W/L = Holm-significant win/loss, w/l = raw CI excludes 0 only.");
    let find = |ts: &[Test], wi: usize, ci: usize, m: usize| -> Option<usize> { ts.iter().position(|t| t.wi == wi && t.ci == ci && t.m == m) };
    let mut world_json = vec![];
    for wi in 0..specs.len() {
        let s = &specs[wi];
        println!("\n--- {} [{:?}{}] {}", s.name, s.role, if s.tune { ", tuned-on" } else { "" }, s.note);
        print!("   {:<44}", "success (mean) / best baseline name:");
        for &m in &hm {
            print!(" {:>7}", MNAMES[m]);
        }
        println!("  | w100-250 w250-500 w500-1k w1k-2k | reg2000 ttt95 ttt90");
        for &pi in base.iter().chain(cand.iter()).chain(abl.iter()) {
            let mn = |m: usize| mean(&vals(&res[wi], pi, m));
            print!("   {:<44}", labels[pi]);
            for &m in &hm {
                print!(" {:7.4}", mn(m));
            }
            print!("  |");
            for &m in &wm {
                print!(" {:7.4}", mn(m));
            }
            println!(" | {:7.2} {:6.0} {:6.0}", mn(9), mn(10), mn(11));
        }
        for &ci in &cand {
            print!("   DIFF {:<39}", labels[ci].chars().take(39).collect::<String>());
            for &m in &hm {
                let i = find(&tests, wi, ci, m).unwrap();
                let t = &tests[i];
                print!(" {:+.4}{}", t.mean, flag(t));
            }
            print!("  |");
            for &m in &wm {
                let i = find(&tests, wi, ci, m).unwrap();
                let t = &tests[i];
                print!(" {:+.4}{}", t.mean, flag(t));
            }
            let (i9, i10, i11) = (find(&tests, wi, ci, 9).unwrap(), find(&tests, wi, ci, 10).unwrap(), find(&tests, wi, ci, 11).unwrap());
            println!(" | {:+.2}{} {:+.0}{} {:+.0}{}", tests[i9].mean, flag(&tests[i9]), tests[i10].mean, flag(&tests[i10]), tests[i11].mean, flag(&tests[i11]));
        }
        for &ci in &cand {
            print!("   vs COLD ({:<5}) {:<28}", if cold(ci) == 7 { "pts*" } else { "guard" }, labels[ci].chars().take(28).collect::<String>());
            for &m in &hm {
                let i = dtests.iter().position(|t| t.wi == wi && t.ci == ci && t.m == m).unwrap();
                print!(" {:+.4}{}", dtests[i].mean, flag(&dtests[i]));
            }
            println!();
        }
        // best baseline names for the H metrics
        print!("   best-baseline per metric:");
        for &m in &hm {
            print!(" {}={}", MNAMES[m], labels[best_base(&res[wi], &base, m)].split('[').next().unwrap());
        }
        println!();
        // json
        let pols_json: Vec<Value> = (0..pols.len())
            .map(|pi| {
                let mut means = BTreeMap::new();
                let mut per_seed = BTreeMap::new();
                for m in 0..NM {
                    means.insert(MNAMES[m], mean(&vals(&res[wi], pi, m)));
                    per_seed.insert(MNAMES[m], vals(&res[wi], pi, m));
                }
                json!({"name": labels[pi], "means": means, "per_seed": per_seed})
            })
            .collect();
        let tests_json: Vec<Value> = tests
            .iter()
            .filter(|t| t.wi == wi)
            .map(|t| {
                json!({"cand": labels[t.ci], "metric": MNAMES[t.m], "family": t.fam, "diff_vs_best_baseline": t.mean, "ci_lo": t.lo, "ci_hi": t.hi, "p": t.p, "holm_p": t.adj,
                    "best_baseline": labels[best_base(&res[wi], &base, t.m)], "flag": flag(t).to_string()})
            })
            .collect();
        let dj: Vec<Value> = dtests
            .iter()
            .filter(|t| t.wi == wi)
            .map(|t| json!({"cand": labels[t.ci], "cold": labels[cold(t.ci)], "metric": MNAMES[t.m], "diff": t.mean, "ci_lo": t.lo, "ci_hi": t.hi, "holm_p": t.adj, "flag": flag(t).to_string()}))
            .collect();
        world_json.push(json!({"name": s.name, "role": format!("{:?}", s.role), "tuned_on": s.tune, "note": s.note, "policies": pols_json, "tests": tests_json, "vs_cold": dj}));
    }

    // ---- summaries
    println!("\n================= SUMMARY (held seeds {}-{}) =================", args.held.0, args.held.1);
    let early_idx = [0usize, 1, 2];
    let mut summary_json = vec![];
    for &ci in &cand {
        let treat_w: Vec<usize> = (0..specs.len()).filter(|&w| specs[w].role == Role::Treat).collect();
        let mut win_worlds_early = vec![];
        let mut win_worlds_any = vec![];
        let mut loss_cells_g = 0;
        for &wi in &treat_w {
            let ew = early_idx.iter().any(|&m| {
                let t = &tests[find(&tests, wi, ci, m).unwrap()];
                flag(t) == 'W'
            });
            let aw = hm.iter().any(|&m| flag(&tests[find(&tests, wi, ci, m).unwrap()]) == 'W');
            if ew {
                win_worlds_early.push(specs[wi].name);
            }
            if aw {
                win_worlds_any.push(specs[wi].name);
            }
            loss_cells_g += hm.iter().filter(|&&m| flag(&tests[find(&tests, wi, ci, m).unwrap()]) == 'L').count();
        }
        let mut hurt = vec![];
        let (mut nl, mut nw) = (0, 0);
        for wi in 0..specs.len() {
            if specs[wi].role == Role::Treat || specs[wi].name.starts_with("B9") {
                continue;
            }
            for &m in &hm {
                let t = &tests[find(&tests, wi, ci, m).unwrap()];
                match flag(t) {
                    'L' => {
                        nl += 1;
                        hurt.push(format!("{}:{}={:+.4}", specs[wi].name, MNAMES[m], t.mean));
                    }
                    'W' => nw += 1,
                    _ => {}
                }
            }
        }
        // wrong-prior bound: worst mean diff (and worst CI lo) over WRONG worlds
        let mut worst = (0.0f64, String::new());
        let mut worst_lo = (0.0f64, String::new());
        let mut worst_cold = (0.0f64, String::new());
        for wi in 0..specs.len() {
            if !matches!(specs[wi].name, "A5-sim-mismatch" | "B6-md-wrong") {
                continue;
            }
            for &m in &hm {
                let t = &tests[find(&tests, wi, ci, m).unwrap()];
                if t.mean < worst.0 {
                    worst = (t.mean, format!("{}:{}", specs[wi].name, MNAMES[m]));
                }
                if t.lo < worst_lo.0 {
                    worst_lo = (t.lo, format!("{}:{}", specs[wi].name, MNAMES[m]));
                }
                let dt = &dtests[dtests.iter().position(|x| x.wi == wi && x.ci == ci && x.m == m).unwrap()];
                if dt.mean < worst_cold.0 {
                    worst_cold = (dt.mean, format!("{}:{}", specs[wi].name, MNAMES[m]));
                }
            }
        }
        let mut worst_all = (0.0f64, String::new());
        for wi in 0..specs.len() {
            if specs[wi].role == Role::Treat || specs[wi].name.starts_with("B9") {
                continue;
            }
            for &m in &hm {
                let t = &tests[find(&tests, wi, ci, m).unwrap()];
                if t.mean < worst_all.0 {
                    worst_all = (t.mean, format!("{}:{}", specs[wi].name, MNAMES[m]));
                }
            }
        }
        let ties = etests.iter().zip(&eadj).filter(|(t, a)| t.1 == ci && **a < 0.05).count();
        let ne = etests.iter().filter(|t| t.1 == ci).count();
        println!("\n{}", labels[ci]);
        println!("   treatment worlds with Holm WIN at an EARLY horizon (H50/H100/H250): {:?}", win_worlds_early);
        println!("   treatment worlds with Holm WIN at ANY H (50..2000): {:?}; Holm LOSS cells in treatment worlds: {}", win_worlds_any, loss_cells_g);
        println!("   stress+control worlds (family L): Holm WIN cells {nw}, Holm LOSS cells {nl} {:?}", hurt);
        println!("   WRONG-prior worlds (A5,B6): worst mean diff vs best baseline {:+.4} at {}, worst CI-lo {:+.4} at {}; worst vs matched cold counterpart {:+.4} at {}", worst.0, worst.1, worst_lo.0, worst_lo.1, worst_cold.0, worst_cold.1);
        println!("   worst mean diff over ALL stress+control cells {:+.4} at {}", worst_all.0, worst_all.1);
        {
            let wi = specs.iter().position(|x| x.name.starts_with("B9")).unwrap();
            let mut line = String::from("   B9 uninformative-feedback (own Holm family) vs BEST baseline [keyword by construction] / vs COLD guarded-safe:");
            for &m in &hm {
                let t = &tests[find(&tests, wi, ci, m).unwrap()];
                let dt = &dtests[dtests.iter().position(|x| x.wi == wi && x.ci == ci && x.m == m).unwrap()];
                line += &format!(" {}={:+.4}{}/{:+.4}{}", MNAMES[m], t.mean, flag(t), dt.mean, flag(dt));
            }
            println!("{line}");
        }
        println!("   controls (A2,A3,A4,B7,B8; B9 reported separately): TOST +-0.01 vs best baseline, Holm-significant equivalence in {ties}/{ne} cells");
        summary_json.push(json!({"cand": labels[ci], "early_win_worlds": win_worlds_early, "any_win_worlds": win_worlds_any, "treat_loss_cells": loss_cells_g,
            "stress_control_win_cells": nw, "stress_control_loss_cells": nl, "losses": hurt,
            "wrong_prior_worst_mean": worst.0, "wrong_prior_worst_at": worst.1, "wrong_prior_worst_ci_lo": worst_lo.0, "wrong_prior_worst_cold": worst_cold.0,
            "worst_all_stress_control": worst_all.0, "worst_all_at": worst_all.1, "control_tost_equiv_cells": ties, "control_cells": ne}));
    }
    // fade table: window diffs vs best baseline in treatment worlds
    println!("\nFADE (window success diff vs best baseline, treatment worlds; W/L Holm in family F):");
    for &ci in &cand {
        println!("  {}", labels[ci]);
        for wi in 0..specs.len() {
            if specs[wi].role != Role::Treat {
                continue;
            }
            print!("     {:<22}", specs[wi].name);
            for &m in &[1usize, 5, 6, 7, 8] {
                let t = &tests[find(&tests, wi, ci, m).unwrap()];
                let name = if m == 1 { "H100(cum)".to_string() } else { MNAMES[m].to_string() };
                print!(" {}={:+.4}{}", name, t.mean, if m == 1 { ' ' } else { flag(t) });
            }
            println!();
        }
    }
    // prior diagnostics (first held seed)
    println!("\nPRIOR DIAGNOSTICS (held seed {}; fit M8 m1=2): static first-250 success of prior-argmax vs keyword vs oracle", args.held.0);
    let mut diag = vec![];
    for s in &specs {
        let w = make_world(&s.eval, args.held.0);
        let deps = build_pool(&s.pool, k, args.held.0);
        let pn = deps[0].cells.values().next().map(|v| v.len()).unwrap();
        let pr = fit_prior(&deps, pn, fit(8.0, 2.0, 0.5));
        let (sp, sk, so, cov) = prior_diag(&w, &pr, 250);
        println!("   {:<22} prior-argmax {:.3}  keyword {:.3}  oracle {:.3}  coverage {:.2}  kw_acc(pool) {:.2}", s.name, sp, sk, so, cov, pr.kw_acc);
        diag.push(json!({"world": s.name, "prior_argmax": sp, "keyword": sk, "oracle": so, "coverage": cov, "kw_acc_pool": pr.kw_acc}));
    }

    // ---- K sensitivity (optional)
    let mut k_json = vec![];
    if args.expk {
        println!("\nK SENSITIVITY (number of previous deployments in the pool; frozen picks; first-250 success diff vs best baseline of the K=6 run is NOT reused: baselines rerun)");
        for &kk in &[1usize, 2, 4, 8, 16] {
            let kj: Vec<(usize, u64)> = ["A1-sim-fixed", "B1-md-related", "B2-md-partial", "A5-sim-mismatch", "B6-md-wrong"]
                .iter()
                .flat_map(|n| {
                    let wi = specs.iter().position(|s| s.name == *n).unwrap();
                    seeds.iter().map(move |&s| (wi, s))
                })
                .collect();
            let rr = par_map(&kj, |&(wi, s)| run_all(&specs[wi], s, kk, &pols));
            for n in ["A1-sim-fixed", "B1-md-related", "B2-md-partial", "A5-sim-mismatch", "B6-md-wrong"] {
                let wi = specs.iter().position(|s| s.name == n).unwrap();
                let rows: Vec<&Vec<[f64; NM]>> = kj.iter().zip(&rr).filter(|(j, _)| j.0 == wi).map(|(_, r)| r).collect();
                let rs: Vec<Vec<[f64; NM]>> = (0..pols.len()).map(|pi| rows.iter().map(|r| r[pi]).collect()).collect();
                let mut line = format!("   K={kk:2} {n:<18}");
                for &ci in &cand {
                    let mut o = String::new();
                    for &m in &[0usize, 2, 3] {
                        let b = best_base(&rs, &base, m);
                        let d = signed_diff(&rs, ci, b, m);
                        o += &format!(" {}={:+.4}", MNAMES[m], mean(&d));
                        k_json.push(json!({"K": kk, "world": n, "cand": labels[ci], "metric": MNAMES[m], "diff_vs_best": mean(&d)}));
                    }
                    line += &format!(" | {}:{}", labels[ci].split('[').next().unwrap(), o);
                }
                println!("{line}");
            }
        }
    }

    let out = json!({
        "note": "SYNTHETIC-until-validated-on-real-traffic. H9 warm-start priors from previous deployments.",
        "held": [args.held.0, args.held.1], "k_pool": k, "n_pool_tasks": N_POOL_TASKS, "n_tasks": N_TASKS,
        "policies": labels, "tuning": picks.log, "worlds": world_json, "summary": summary_json, "prior_diag": diag, "k_sensitivity": k_json,
        "control_tost": etests.iter().zip(&eadj).map(|(t, a)| json!({"world": specs[t.0].name, "cand": labels[t.1], "metric": MNAMES[t.2], "p": t.3, "holm_p": a, "diff": t.4})).collect::<Vec<_>>(),
    });
    std::fs::write(&args.out, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nwrote {}", args.out);
}
