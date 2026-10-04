//! W2 fault-tolerance lab (uses sim.rs read-only): does faster bench/recover
//! (`FaultConfig`: streak-bench, probation backoff, tuned health knobs) improve success
//! DURING node-failure windows (and not hurt the rest) over the current mesh-robust health
//! benching and over the fair baselines?
//!
//! All mesh variants are team_size = 1 (the robust configuration found by the robustness
//! study), leader = hebbian-robust.  Metrics are EXPECTED success of the routed agent
//! (`World::success_prob`, lower variance than the realised Bernoulli): `fail` = mean over
//! failure-window tasks, `all` = mean over the whole run.
//! Tune on seeds 1-8 (objective .5 fail + .5 all, mean over the four scenarios), evaluate on
//! held-out seeds 31-60, paired 95% t-CI, Holm.
//!
//! Usage: cargo run --release --bin w2_fault [-- --out results/w2_fault.json]

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::mesh::{FaultConfig, MeshConfig, NeuralMesh};
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::json;

fn heb_robust() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

#[derive(Clone, Debug)]
enum Kind {
    Keyword,
    EpsGreedy,
    PooledTs,
    HebR,
    WarmHebR,
    Mesh(MeshConfig, Option<FaultConfig>),
}

trait Pol {
    fn pick(&mut self, t: &Task, cands: &[usize], rng: &mut Rng) -> usize;
    fn fb(&mut self, t: &Task, a: usize, r: f64);
}
struct Plain<R: Router>(R);
impl<R: Router> Pol for Plain<R> {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.select(t, c, rng)
    }
    fn fb(&mut self, t: &Task, a: usize, r: f64) {
        self.0.update(t, a, r)
    }
}
struct M(NeuralMesh<HebbianRouter>);
impl Pol for M {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.route(t, c, rng)[0]
    }
    fn fb(&mut self, t: &Task, a: usize, r: f64) {
        self.0.feedback(t, &[a], r)
    }
}

fn build(k: &Kind, w: &World) -> Box<dyn Pol> {
    let n = w.n_agents();
    match k {
        Kind::Keyword => Box::new(Plain(KeywordRouter::new(w.keyword_map_for(), n))),
        Kind::EpsGreedy => Box::new(Plain(EpsGreedyRouter::new(n, 0.99, 0.02))),
        Kind::PooledTs => Box::new(Plain(PooledThompsonRouter::new(n, 1.0))),
        Kind::HebR => Box::new(Plain(HebbianRouter::new(n, heb_robust()))),
        Kind::WarmHebR => Box::new(Plain(
            Warm::new(HebbianRouter::new(n, heb_robust()), w.keyword_map_for(), n, 1.0).with_evidence(Evidence::MeanFeature),
        )),
        Kind::Mesh(m, f) => {
            let mut mesh = NeuralMesh::new(n, HebbianRouter::new(n, heb_robust()), m.clone());
            if let Some(f) = f {
                mesh = mesh.with_fault(f.clone());
            }
            Box::new(M(mesh))
        }
    }
}

/// [fail_exp, all_exp, fail_real, all_real]
fn run(k: &Kind, seed: u64, wc: &WorldConfig) -> [f64; 4] {
    let w = World::new(seed, wc.clone());
    let n = w.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut pol = build(k, &w);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let (mut fe, mut ae, mut fr, mut ar, mut nf) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for t in 0..w.len() {
        let st = w.task(t);
        let a = pol.pick(&st.task, &cands, &mut rng);
        let team = [a];
        let pe = w.success_prob(t, &team);
        let r = w.outcome(t, &team);
        pol.fb(&st.task, a, r);
        ae += pe;
        ar += r;
        if w.in_failure(t) {
            fe += pe;
            fr += r;
            nf += 1.0;
        }
    }
    let nt = w.len() as f64;
    let nf = nf.max(1.0);
    [fe / nf, ae / nt, fr / nf, ar / nt]
}

fn scenarios() -> Vec<(&'static str, WorldConfig)> {
    let base = WorldConfig { n_tasks: 2000, ..WorldConfig::default() };
    [Scenario::Fixed, Scenario::Random, Scenario::NoSynergy, Scenario::NoTraps]
        .iter()
        .map(|&s| (s.label(), WorldConfig { scenario: s, ..base.clone() }))
        .collect()
}

fn eval(k: &Kind, seeds: &[u64], wc: &WorldConfig) -> Vec<[f64; 4]> {
    seeds.par_iter().map(|&s| run(k, s, wc)).collect()
}

fn obj(k: &Kind, seeds: &[u64], scs: &[(&'static str, WorldConfig)]) -> f64 {
    let mut tot = 0.0;
    for (_, wc) in scs {
        let r = eval(k, seeds, wc);
        tot += r.iter().map(|x| 0.5 * x[0] + 0.5 * x[1]).sum::<f64>() / r.len() as f64;
    }
    tot / scs.len() as f64
}

fn col(r: &[[f64; 4]], m: usize) -> Vec<f64> {
    r.iter().map(|x| x[m]).collect()
}

fn mesh_cfg(th: f64, al: f64, retry: u64) -> MeshConfig {
    MeshConfig { edge_eta: 0.01, edge_decay: 0.0, team_size: 1, health_threshold: th, health_alpha: al, retry_every: retry, ..MeshConfig::default() }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/w2_fault.json".into());
    let scs = scenarios();
    let tune: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = (31..=60).collect();
    println!("w2_fault: 2000 tasks/run, 4 scenarios, tune seeds 1-8, held-out 31-60");

    // ---- tuning -------------------------------------------------------------------------
    let base_obj = obj(&Kind::Mesh(mesh_cfg(0.3, 0.2, 20), None), &tune, &scs);
    println!("[tune 1-8] mesh-robust (default health) objective {base_obj:.4}");
    let mut best = (f64::NEG_INFINITY, (0u32, 20u64, 1.0f64, 0.3f64, 0.2f64));
    let mut n = 0;
    for &streak in &[0u32, 2, 3, 5] {
        for &retry in &[5u64, 10, 20, 40] {
            for &backoff in &[1.0, 2.0] {
                for &th in &[0.3, 0.5] {
                    for &al in &[0.2, 0.4] {
                        let k = Kind::Mesh(mesh_cfg(th, al, retry), Some(FaultConfig { streak, backoff, ..FaultConfig::default() }));
                        let o = obj(&k, &tune, &scs);
                        n += 1;
                        if o > best.0 {
                            best = (o, (streak, retry, backoff, th, al));
                        }
                    }
                }
            }
        }
    }
    let (streak, retry, backoff, th, al) = best.1;
    println!("[tune 1-8] fault grid {n} configs, best {:.4}: streak {streak}, retry {retry}, backoff {backoff}, threshold {th}, alpha {al}", best.0);

    let fc = |s: u32, b: f64| FaultConfig { streak: s, backoff: b, ..FaultConfig::default() };
    let baselines: Vec<(&str, Kind)> = vec![
        ("keyword", Kind::Keyword),
        ("eps-greedy", Kind::EpsGreedy),
        ("pooled-ts", Kind::PooledTs),
        ("hebbian-robust", Kind::HebR),
        ("warm(hebbian-robust)", Kind::WarmHebR),
    ];
    let cur = ("mesh-robust(current best)", Kind::Mesh(mesh_cfg(0.3, 0.2, 20), None));
    let cands: Vec<(&str, Kind)> = vec![
        ("fault-tuned(all knobs)", Kind::Mesh(mesh_cfg(th, al, retry), Some(fc(streak, backoff)))),
        ("streak2-only", Kind::Mesh(mesh_cfg(0.3, 0.2, 20), Some(fc(2, 1.0)))),
        ("retry5+backoff2-only", Kind::Mesh(mesh_cfg(0.3, 0.2, 5), Some(fc(0, 2.0)))),
    ];
    // candidates list for the "vs best fair baseline" family includes the current best
    let mut fam1: Vec<(&str, Kind)> = vec![cur.clone()];
    fam1.extend(cands.iter().cloned());

    let mn = ["fail", "all"];
    let mut p1: Vec<f64> = vec![];
    let mut m1: Vec<(usize, usize, usize)> = vec![]; // (scenario, cand, metric)
    let mut p2: Vec<f64> = vec![];
    let mut m2: Vec<(usize, usize, usize)> = vec![];
    let mut stores = vec![];
    let mut bbs = vec![];
    for (si, (_, wc)) in scs.iter().enumerate() {
        let b: Vec<Vec<[f64; 4]>> = baselines.iter().map(|(_, k)| eval(k, &held, wc)).collect();
        let c: Vec<Vec<[f64; 4]>> = fam1.iter().map(|(_, k)| eval(k, &held, wc)).collect();
        let mut bb = [0usize; 2];
        for m in 0..2 {
            bb[m] = (0..b.len())
                .max_by(|&x, &y| summarize(&col(&b[x], m)).mean.partial_cmp(&summarize(&col(&b[y], m)).mean).unwrap())
                .unwrap();
        }
        for ci in 0..c.len() {
            for m in 0..2 {
                p1.push(paired_t_pvalue(&col(&c[ci], m), &col(&b[bb[m]], m)));
                m1.push((si, ci, m));
                if ci >= 1 {
                    p2.push(paired_t_pvalue(&col(&c[ci], m), &col(&c[0], m)));
                    m2.push((si, ci, m));
                }
            }
        }
        stores.push((b, c));
        bbs.push(bb);
    }
    let a1 = holm_adjust(&p1);
    let a2 = holm_adjust(&p2);
    let mut js = vec![];
    for (si, (sn, _)) in scs.iter().enumerate() {
        let (b, c) = &stores[si];
        println!("\n=== scenario {sn} (held-out 31-60): expected success, mean+-95%hw; realised fail-window in () ===");
        println!("{:<28}{:>18}{:>18}{:>12}", "router", "fail-window", "overall", "real-fail");
        let f = |r: &Vec<[f64; 4]>, m: usize| { let s = summarize(&col(r, m)); format!("{:.4}+-{:.4}", s.mean, s.half_width) };
        for (i, (nm, _)) in baselines.iter().enumerate() {
            println!("{:<28}{:>18}{:>18}{:>12.4}", nm, f(&b[i], 0), f(&b[i], 1), summarize(&col(&b[i], 2)).mean);
        }
        for (i, (nm, _)) in fam1.iter().enumerate() {
            println!("{:<28}{:>18}{:>18}{:>12.4}", nm, f(&c[i], 0), f(&c[i], 1), summarize(&col(&c[i], 2)).mean);
        }
        for m in 0..2 {
            println!("  best fair baseline ({}) = {}", mn[m], baselines[bbs[si][m]].0);
        }
        for (k, &(s2, ci, m)) in m1.iter().enumerate() {
            if s2 != si {
                continue;
            }
            let d = paired_diff(&col(&c[ci], m), &col(&b[bbs[si][m]], m));
            let v = if a1[k] < 0.05 { if d.diff.mean > 0.0 { "BEATS" } else { "WORSE" } } else { "n.s." };
            println!("  {:<26} [{:<4}] vs best-baseline {:+.4} CI[{:+.4},{:+.4}] wins {}/{} Holm p={:.4} {}",
                fam1[ci].0, mn[m], d.diff.mean, d.diff.lo, d.diff.hi, d.wins, d.wins + d.losses + d.ties, a1[k], v);
            js.push(json!({"scenario": sn, "cand": fam1[ci].0, "metric": mn[m], "vs": baselines[bbs[si][m]].0, "family": "vs-best-fair-baseline",
                "diff": d.diff, "wins": d.wins, "losses": d.losses, "holm_p": a1[k], "verdict": v}));
        }
        for (k, &(s2, ci, m)) in m2.iter().enumerate() {
            if s2 != si {
                continue;
            }
            let d = paired_diff(&col(&c[ci], m), &col(&c[0], m));
            let v = if a2[k] < 0.05 { if d.diff.mean > 0.0 { "BEATS" } else { "WORSE" } } else { "n.s." };
            println!("  {:<26} [{:<4}] vs mesh-robust    {:+.4} CI[{:+.4},{:+.4}] wins {}/{} Holm p={:.4} {}",
                fam1[ci].0, mn[m], d.diff.mean, d.diff.lo, d.diff.hi, d.wins, d.wins + d.losses + d.ties, a2[k], v);
            js.push(json!({"scenario": sn, "cand": fam1[ci].0, "metric": mn[m], "vs": "mesh-robust", "family": "vs-current-best",
                "diff": d.diff, "wins": d.wins, "losses": d.losses, "holm_p": a2[k], "verdict": v}));
        }
    }
    let beats = |fam: &str| js.iter().filter(|j| j["family"] == fam && j["verdict"] == "BEATS").count();
    println!("\nfamily vs-best-fair-baseline: {} tests, BEATS {}; family vs-current-best: {} tests, BEATS {}", p1.len(), beats("vs-best-fair-baseline"), p2.len(), beats("vs-current-best"));
    let j = json!({"tuned": {"streak": streak, "retry": retry, "backoff": backoff, "health_threshold": th, "health_alpha": al, "objective": best.0, "mesh_robust_objective": base_obj}, "tests": js});
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out}");
}
