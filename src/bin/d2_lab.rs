//! D2 (MESH-2) lab: extend the verified SecondOpinion gate.
//!
//! Candidate: `variants::HierSecondOpinion` (hierarchical shrinkage global -> checker -> (feature,
//! checker); optimism at the checker level so exploration ends once pooled evidence is conclusive;
//! agent-pooled leader prior). World/protocol are the x2_lab verification world (copied, read-only
//! `sim::World`): checker catch q = qscale*skill, false alarm = fascale*(1-skill), flags verifiable.
//! Wrapped router = warm(hebbian-robust) (best baseline on tune seeds in x2); picks are identical to
//! the bare router so the question is NET = E-success - c*E-calls.
//!
//! Comparators (paired, per seed): best fair baseline (by held-out mean; keyword / warm(hebR) ...) and
//! the existing SecondOpinion with its published frozen configs (x2_lab tuning). Prefix horizons H250 /
//! H500 measure learning speed. Holm over the main family.
//!
//! Usage: cargo run --release --bin d2_lab [-- --held 31-60 --out results/d2_lab.json]

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::mesh::{MeshConfig, NeuralMesh};
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use neural_mesh::variants::{CheckPlan, HierSecondOpinion, HierSoConfig, SecondOpinion, SecondOpinionConfig};
use rayon::prelude::*;
use serde_json::json;

const COSTS: [f64; 2] = [0.05, 0.1];
const PREF: [usize; 2] = [250, 500];

fn heb_robust() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}
fn mesh_robust_cfg() -> MeshConfig {
    MeshConfig { edge_eta: 0.01, edge_decay: 0.0, team_size: 1, ..MeshConfig::default() }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Base {
    Keyword,
    EpsGreedy,
    PooledTs,
    HebR,
    WarmHebR,
    MeshRobust,
}
const BASELINES: [(&str, Base); 6] = [
    ("keyword", Base::Keyword),
    ("eps-greedy", Base::EpsGreedy),
    ("pooled-ts", Base::PooledTs),
    ("hebbian-robust", Base::HebR),
    ("warm(hebbian-robust)", Base::WarmHebR),
    ("mesh-robust", Base::MeshRobust),
];

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
struct M<R: Router>(NeuralMesh<R>);
impl<R: Router> Pol for M<R> {
    fn pick(&mut self, t: &Task, c: &[usize], rng: &mut Rng) -> usize {
        self.0.route(t, c, rng)[0]
    }
    fn fb(&mut self, t: &Task, a: usize, r: f64) {
        self.0.feedback(t, &[a], r)
    }
}
fn build(b: &Base, w: &World) -> Box<dyn Pol> {
    let n = w.n_agents();
    match b {
        Base::Keyword => Box::new(Plain(KeywordRouter::new(w.keyword_map_for(), n))),
        Base::EpsGreedy => Box::new(Plain(EpsGreedyRouter::new(n, 0.99, 0.02))),
        Base::PooledTs => Box::new(Plain(PooledThompsonRouter::new(n, 1.0))),
        Base::HebR => Box::new(Plain(HebbianRouter::new(n, heb_robust()))),
        Base::WarmHebR => Box::new(Plain(
            Warm::new(HebbianRouter::new(n, heb_robust()), w.keyword_map_for(), n, 1.0).with_evidence(Evidence::MeanFeature),
        )),
        Base::MeshRobust => Box::new(M(NeuralMesh::new(n, HebbianRouter::new(n, heb_robust()), mesh_robust_cfg()))),
    }
}
fn policy_rng(seed: u64) -> Rng {
    Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17))
}
fn uniforms(seed: u64, salt: u64, n: usize) -> Vec<f64> {
    let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt);
    (0..n).map(|_| r.f64()).collect()
}

#[derive(Clone, Debug)]
struct VScen {
    name: &'static str,
    sc: Scenario,
    qscale: f64,
    fascale: f64,
}
fn vscens() -> Vec<VScen> {
    let v = |name, sc, qscale, fascale| VScen { name, sc, qscale, fascale };
    vec![
        v("v-fixed", Scenario::Fixed, 1.0, 0.3),
        v("v-random", Scenario::Random, 1.0, 0.3),
        v("v-notraps", Scenario::NoTraps, 1.0, 0.3),
        v("v-weak", Scenario::Random, 0.5, 0.5),
        v("none-fixed", Scenario::Fixed, 0.0, 0.3),
        v("none-random", Scenario::Random, 0.0, 0.3),
    ]
}

struct VWorld {
    w: World,
    qscale: f64,
    fascale: f64,
    u_retry: Vec<f64>,
    u_catch: Vec<f64>,
}
impl VWorld {
    fn new(seed: u64, s: &VScen) -> Self {
        let w = World::new(seed, WorldConfig { scenario: s.sc, ..WorldConfig::default() });
        let n = w.len();
        VWorld { w, qscale: s.qscale, fascale: s.fascale, u_retry: uniforms(seed, 0x2E72, n), u_catch: uniforms(seed, 0xC4EC, n) }
    }
    fn p(&self, t: usize, a: usize) -> f64 {
        self.w.success_prob(t, &[a])
    }
    fn q(&self, t: usize, c: usize) -> f64 {
        if self.w.is_down(t, c) {
            return 0.0;
        }
        (self.qscale * self.w.skill(t, self.w.task(t).type_idx, c)).clamp(0.0, 1.0)
    }
    fn fa(&self, t: usize, c: usize) -> f64 {
        if self.w.is_down(t, c) {
            return 0.0;
        }
        (self.fascale * (1.0 - self.w.skill(t, self.w.task(t).type_idx, c))).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug)]
enum Mode {
    Off,
    Old(SecondOpinionConfig),
    Hier(HierSoConfig),
}

enum Gate {
    None,
    Old(SecondOpinion),
    Hier(HierSecondOpinion),
}
impl Gate {
    fn decide(&mut self, t: &Task, a: usize, c: &[usize]) -> Option<CheckPlan> {
        match self {
            Gate::None => None,
            Gate::Old(g) => g.decide(t, a, c),
            Gate::Hier(g) => g.decide(t, a, c),
        }
    }
    fn obs_check(&mut self, t: &Task, ck: usize, failed: bool, flagged: bool) {
        match self {
            Gate::None => {}
            Gate::Old(g) => g.observe_check(t, ck, failed, flagged),
            Gate::Hier(g) => g.observe_check(t, ck, failed, flagged),
        }
    }
    fn obs_att(&mut self, t: &Task, a: usize, ok: bool) {
        match self {
            Gate::None => {}
            Gate::Old(g) => g.observe_attempt(t, a, ok),
            Gate::Hier(g) => g.observe_attempt(t, a, ok),
        }
    }
}

/// [E-success, E-calls, real success, real calls, check rate, es@250, ec@250, es@500, ec@500]
type VRes = [f64; 9];

fn run_v(base: &Base, mode: &Mode, seed: u64, s: &VScen) -> VRes {
    let vw = VWorld::new(seed, s);
    let w = &vw.w;
    let n = w.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut pol = build(base, w);
    let mut gate = match mode {
        Mode::Off => Gate::None,
        Mode::Old(c) => Gate::Old(SecondOpinion::new(n, c.clone())),
        Mode::Hier(c) => Gate::Hier(HierSecondOpinion::new(n, c.clone())),
    };
    let on = !matches!(mode, Mode::Off);
    let mut rng = policy_rng(seed);
    let (mut es, mut ec, mut rs, mut rc, mut checks) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut pre = [0.0f64; 4];
    for t in 0..w.len() {
        let st = w.task(t);
        let a = pol.pick(&st.task, &cands, &mut rng);
        let pl = vw.p(t, a);
        let ok = w.outcome(t, &[a]) > 0.5;
        let plan = gate.decide(&st.task, a, &cands);
        let (mut succ, mut calls) = (if ok { 1.0 } else { 0.0 }, 1.0);
        let (mut e_succ, mut e_calls) = (pl, 1.0);
        if let Some(pl_) = plan {
            let (c, r) = (pl_.checker, pl_.retry);
            let (q, fa, pr) = (vw.q(t, c), vw.fa(t, c), vw.p(t, r));
            checks += 1.0;
            e_succ = pl + (1.0 - pl) * q * pr;
            e_calls = 2.0 + (1.0 - pl) * q + pl * fa;
            calls += 1.0;
            let flagged = if ok { vw.u_catch[t] < fa } else { vw.u_catch[t] < q };
            gate.obs_check(&st.task, c, !ok, flagged);
            if flagged {
                calls += 1.0;
                let rok = vw.u_retry[t] < pr;
                gate.obs_att(&st.task, r, rok);
                if rok {
                    succ = 1.0;
                }
            }
        }
        if on {
            gate.obs_att(&st.task, a, ok);
        }
        pol.fb(&st.task, a, if ok { 1.0 } else { 0.0 });
        es += e_succ;
        ec += e_calls;
        rs += succ;
        rc += calls;
        for (i, &h) in PREF.iter().enumerate() {
            if t + 1 == h {
                pre[2 * i] = es / h as f64;
                pre[2 * i + 1] = ec / h as f64;
            }
        }
    }
    let nt = w.len() as f64;
    [es / nt, ec / nt, rs / nt, rc / nt, checks / nt, pre[0], pre[1], pre[2], pre[3]]
}

fn net(r: &VRes, c: f64) -> f64 {
    r[0] - c * r[1]
}
fn net_h(r: &VRes, c: f64, hi: usize) -> f64 {
    r[5 + 2 * hi] - c * r[6 + 2 * hi]
}
fn col(r: &[VRes], f: impl Fn(&VRes) -> f64) -> Vec<f64> {
    r.iter().map(f).collect()
}
fn ms(x: &[f64]) -> String {
    let s = summarize(x);
    format!("{:.4}+-{:.4}", s.mean, s.half_width)
}
fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned()
}

/// frozen published configs of the existing gate (x2_lab tuning on seeds 1-8).
fn old_cfg(c: f64) -> SecondOpinionConfig {
    if c < 0.07 {
        SecondOpinionConfig { cost: c, kappa: 0.0, q_prior: (1.0, 1.0), discount: 0.995, pool: 2.0, ..SecondOpinionConfig::default() }
    } else {
        SecondOpinionConfig { cost: c, kappa: 0.0, q_prior: (1.0, 1.0), discount: 1.0, pool: 8.0, ..SecondOpinionConfig::default() }
    }
}

fn hier_grid(c: f64) -> Vec<HierSoConfig> {
    let mut v = vec![];
    for &kc in &[0.0, 1.0, 2.0] {
        for &pg in &[8.0, 32.0, 128.0] {
            for &pc in &[2.0, 8.0, 32.0] {
                for &pl in &[0.0, 8.0] {
                    for &disc in &[0.995, 1.0] {
                        for &kcell in &[0.0, 0.5] {
                            v.push(HierSoConfig {
                                base: SecondOpinionConfig { cost: c, kappa: 0.0, discount: disc, ..SecondOpinionConfig::default() },
                                pool_global: pg,
                                pool_checker: pc,
                                pool_leader: pl,
                                kappa_checker: kc,
                                kappa_cell: kcell,
                                probe_every: 0,
                            });
                        }
                    }
                }
            }
        }
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = arg(&args, "--out").unwrap_or("results/d2_lab.json".into());
    let held: Vec<u64> = match arg(&args, "--held") {
        Some(s) => {
            let (a, b) = s.split_once('-').expect("--held A-B");
            (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
        }
        None => (31..=60).collect(),
    };
    let tune: Vec<u64> = (1..=8).collect();
    let vs = vscens();
    println!("d2_lab: tune seeds 1-8, held-out {}-{}", held[0], held[held.len() - 1]);

    // ---- tuning: regret-averse objective vs bare warm(hebR): mean net diff + mean of negative diffs ----
    let jobs: Vec<(usize, u64)> = (0..vs.len()).flat_map(|i| tune.iter().map(move |&s| (i, s))).collect();
    let off: Vec<VRes> = jobs.par_iter().map(|&(i, s)| run_v(&Base::WarmHebR, &Mode::Off, s, &vs[i])).collect();
    let mut tuned: Vec<HierSoConfig> = vec![];
    for &c in COSTS.iter() {
        let grid = hier_grid(c);
        let scored: Vec<(f64, usize)> = grid
            .par_iter()
            .enumerate()
            .map(|(gi, cfg)| {
                let r: Vec<VRes> = jobs.iter().map(|&(i, s)| run_v(&Base::WarmHebR, &Mode::Hier(cfg.clone()), s, &vs[i])).collect();
                // per-world mean diff
                let mut diffs = vec![];
                for i in 0..vs.len() {
                    let idx: Vec<usize> = (0..jobs.len()).filter(|&j| jobs[j].0 == i).collect();
                    let d: f64 = idx.iter().map(|&j| net(&r[j], c) - net(&off[j], c)).sum::<f64>() / idx.len() as f64;
                    diffs.push(d);
                }
                let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
                let neg = diffs.iter().map(|d| d.min(0.0)).sum::<f64>() / diffs.len() as f64;
                (mean + neg, gi)
            })
            .collect();
        let best = scored.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap();
        let cfg = grid[best.1].clone();
        println!(
            "[tune 1-8] c={c}: obj {:.5}  pool_g {} pool_c {} pool_l {} k_chk {} k_cell {} disc {}",
            best.0, cfg.pool_global, cfg.pool_checker, cfg.pool_leader, cfg.kappa_checker, cfg.kappa_cell, cfg.base.discount
        );
        tuned.push(cfg);
    }

    // ---- held-out ----
    struct Test {
        scen: &'static str,
        cand: String,
        metric: String,
        vs: String,
        a: Vec<f64>,
        b: Vec<f64>,
        calls: f64,
    }
    let mut tests: Vec<Test> = vec![];
    let mut tables = vec![];
    for s in &vs {
        let b: Vec<Vec<VRes>> = BASELINES.iter().map(|(_, k)| held.par_iter().map(|&sd| run_v(k, &Mode::Off, sd, s)).collect()).collect();
        let bb = (0..b.len()).max_by(|&x, &y| summarize(&col(&b[x], |r| r[0])).mean.total_cmp(&summarize(&col(&b[y], |r| r[0])).mean)).unwrap();
        let warm = 4;
        println!("\n=== {} (held-out, qscale {}, fascale {}); best fair baseline = {} ===", s.name, s.qscale, s.fascale, BASELINES[bb].0);
        println!("{:<26}{:>15}{:>8}{:>8}{:>16}{:>16}{:>16}{:>16}", "policy", "E-success", "calls", "chk", "net@0.05", "net@0.10", "net250@.05", "net500@.05");
        let mut rows = vec![];
        for (i, (nm, _)) in BASELINES.iter().enumerate() {
            let r = &b[i];
            println!("{:<26}{:>15}{:>8.3}{:>8.3}{:>16}{:>16}{:>16}{:>16}", nm, ms(&col(r, |x| x[0])), 1.0, 0.0, ms(&col(r, |x| net(x, 0.05))), ms(&col(r, |x| net(x, 0.1))), ms(&col(r, |x| net_h(x, 0.05, 0))), ms(&col(r, |x| net_h(x, 0.05, 1))));
        }
        for (ci, &c) in COSTS.iter().enumerate() {
            let old: Vec<VRes> = held.par_iter().map(|&sd| run_v(&Base::WarmHebR, &Mode::Old(old_cfg(c)), sd, s)).collect();
            let hier: Vec<VRes> = held.par_iter().map(|&sd| run_v(&Base::WarmHebR, &Mode::Hier(tuned[ci].clone()), sd, s)).collect();
            for (nm, r) in [(format!("SO-old(c={c})"), &old), (format!("SO-hier(c={c})"), &hier)] {
                println!("{:<26}{:>15}{:>8.3}{:>8.3}{:>16}{:>16}{:>16}{:>16}", nm, ms(&col(r, |x| x[0])), summarize(&col(r, |x| x[1])).mean, summarize(&col(r, |x| x[4])).mean,
                    ms(&col(r, |x| net(x, 0.05))), ms(&col(r, |x| net(x, 0.1))), ms(&col(r, |x| net_h(x, 0.05, 0))), ms(&col(r, |x| net_h(x, 0.05, 1))));
                rows.push(json!({"policy": nm, "e_success": summarize(&col(r, |x| x[0])), "calls": summarize(&col(r, |x| x[1])), "net": summarize(&col(r, |x| net(x, c))),
                    "net_h250": summarize(&col(r, |x| net_h(x, c, 0))), "net_h500": summarize(&col(r, |x| net_h(x, c, 1))), "check_rate": summarize(&col(r, |x| x[4]))}));
            }
            let calls = summarize(&col(&hier, |x| x[1])).mean;
            let f: [(&str, Box<dyn Fn(&VRes) -> f64>); 3] = [
                ("net", Box::new(move |r: &VRes| net(r, c))),
                ("net@H250", Box::new(move |r: &VRes| net_h(r, c, 0))),
                ("net@H500", Box::new(move |r: &VRes| net_h(r, c, 1))),
            ];
            for (mn, g) in f.iter() {
                let mname = format!("{mn}@{c}");
                if *mn == "net" {
                    tests.push(Test { scen: s.name, cand: format!("hier(c={c})"), metric: mname.clone(), vs: format!("best-fair:{}", BASELINES[bb].0),
                        a: col(&hier, g), b: col(&b[bb], g), calls });
                    tests.push(Test { scen: s.name, cand: format!("hier(c={c})"), metric: mname.clone(), vs: format!("warm(hebbian-robust)"),
                        a: col(&hier, g), b: col(&b[warm], g), calls });
                    tests.push(Test { scen: s.name, cand: format!("old(c={c})"), metric: mname.clone(), vs: format!("warm(hebbian-robust)"),
                        a: col(&old, g), b: col(&b[warm], g), calls: summarize(&col(&old, |x| x[1])).mean });
                }
                tests.push(Test { scen: s.name, cand: format!("hier(c={c})"), metric: mname, vs: format!("SO-old(c={c})"), a: col(&hier, g), b: col(&old, g), calls });
            }
        }
        tables.push(json!({"scenario": s.name, "best_baseline": BASELINES[bb].0, "rows": rows}));
    }
    let p: Vec<f64> = tests.iter().map(|t| paired_t_pvalue(&t.a, &t.b)).collect();
    let adj = holm_adjust(&p);
    println!("\n#### all tests ({} tests, Holm over all)", tests.len());
    let mut js = vec![];
    for (k, t) in tests.iter().enumerate() {
        let d = paired_diff(&t.a, &t.b);
        let v = if adj[k] < 0.05 { if d.diff.mean > 0.0 { "BEATS" } else { "WORSE" } } else { "n.s." };
        println!("{:<12} {:<14} {:<12} vs {:<30} {:+.4} CI[{:+.4},{:+.4}] wins {:>2}/{} calls {:.3} Holm p={:.4} {}", t.scen, t.cand, t.metric, t.vs, d.diff.mean, d.diff.lo, d.diff.hi, d.wins, t.a.len(), t.calls, adj[k], v);
        js.push(json!({"scenario": t.scen, "cand": t.cand, "metric": t.metric, "vs": t.vs, "diff": d.diff, "wins": d.wins, "calls_per_task": t.calls, "p": p[k], "holm_p": adj[k], "verdict": v}));
    }
    let j = json!({"protocol": "tune seeds 1-8 (regret-averse obj vs bare warm), held-out paired t-CI, Holm over all tests",
        "held": [held[0], held[held.len() - 1]],
        "tuned": tuned.iter().zip(COSTS.iter()).map(|(c, k)| json!({"cost": k, "pool_global": c.pool_global, "pool_checker": c.pool_checker, "pool_leader": c.pool_leader, "kappa_checker": c.kappa_checker, "kappa_cell": c.kappa_cell, "discount": c.base.discount})).collect::<Vec<_>>(),
        "tables": tables, "tests": js});
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out}");
}
