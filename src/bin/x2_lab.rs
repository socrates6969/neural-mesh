//! X2 (mesh-thinker) lab: can the mesh add measurable *critical thinking*?
//!
//! H1 — uncertainty-triggered second opinion (`variants::SecondOpinion`): after the
//!      leader is picked, ask a checker ONLY when the expected value of checking
//!      (P(leader fails) * q(checker) * P(retry succeeds)) beats the per-call cost.
//!      Verification world (built here on top of `sim::World`, read-only): a checker
//!      `c` catches a failed leader output with probability `q = qscale * skill(c)`
//!      and false-alarms on a good output with `fa = fascale * (1 - skill(c))`.
//!      Flags are verifiable: a flag triggers a retry by the best other agent (by the
//!      mechanism's own calibrated estimates), and the confirmed-good output is
//!      delivered. Calls: leader 1, +1 per check, +1 per retry. Net value =
//!      success - cost * calls. The wrapped router sees the same rng stream and the
//!      same feedback (leader outcome only) as without the mechanism, so its picks
//!      are identical to the bare base router: the success gain is purely caught
//!      failures, and the real question is NET value after cost.
//!      `none` scenarios (qscale 0: checks are useless) test "no harm".
//! H3 — context-specific benching (`mesh::ContextBenchConfig`): per (feature, agent)
//!      health so an agent failing in one context (e.g. security tasks) is benched for
//!      that context only. Context-failure world built here: during a 30% window one
//!      agent that is best on >= 2 types fails always on ONE of them. Standard
//!      `sim.rs` global-failure worlds test "no harm".
//!
//! Protocol: tune on seeds 1-8 only, held-out 31-60, paired 95% t-CI, Holm across ALL
//! my main tests, compare vs best (by held-out mean) of {keyword, eps-greedy,
//! pooled-ts, hebbian-robust, warm(hebbian-robust), mesh-robust(team 1 + benching)}.
//! Metrics are EXPECTED values (analytic given the routed decisions); realised values
//! are reported too.
//!
//! Usage: cargo run --release --bin x2_lab [-- --out results/x2_lab.json]

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::mesh::{ContextBenchConfig, MeshConfig, NeuralMesh};
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::sim::{Scenario, World, WorldConfig};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use neural_mesh::variants::{SecondOpinion, SecondOpinionConfig};
use rayon::prelude::*;
use serde_json::json;

const COSTS: [f64; 2] = [0.05, 0.1];

fn heb_robust() -> HebbianConfig {
    HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() }
}

fn mesh_robust_cfg() -> MeshConfig {
    MeshConfig { edge_eta: 0.01, edge_decay: 0.0, team_size: 1, ..MeshConfig::default() }
}

// ---------------------------------------------------------------- policies -----------

#[derive(Clone, Debug, PartialEq)]
enum Base {
    Keyword,
    EpsGreedy,
    PooledTs,
    HebR,
    WarmHebR,
    MeshRobust,
    /// context-bench mesh: (config, global health threshold, leader = warm(hebR)?)
    MeshCtx(ContextBenchConfig, f64, bool),
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
        Base::MeshCtx(c, gth, warm) => {
            let cfg = MeshConfig { health_threshold: *gth, ..mesh_robust_cfg() };
            if *warm {
                let leader = Warm::new(HebbianRouter::new(n, heb_robust()), w.keyword_map_for(), n, 1.0).with_evidence(Evidence::MeanFeature);
                Box::new(M(NeuralMesh::new(n, leader, cfg).with_context_bench(c.clone())))
            } else {
                Box::new(M(NeuralMesh::new(n, HebbianRouter::new(n, heb_robust()), cfg).with_context_bench(c.clone())))
            }
        }
    }
}

fn policy_rng(seed: u64) -> Rng {
    Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17))
}

/// Deterministic per-task uniforms of a separate stream (shared by all methods).
fn uniforms(seed: u64, salt: u64, n: usize) -> Vec<f64> {
    let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt);
    (0..n).map(|_| r.f64()).collect()
}

// ======================================================================= H1 ===========

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
    u_rand: Vec<f64>,
}

impl VWorld {
    fn new(seed: u64, s: &VScen) -> Self {
        let w = World::new(seed, WorldConfig { scenario: s.sc, ..WorldConfig::default() });
        let n = w.len();
        VWorld {
            w,
            qscale: s.qscale,
            fascale: s.fascale,
            u_retry: uniforms(seed, 0x2E72, n),
            u_catch: uniforms(seed, 0xC4EC, n),
            u_rand: uniforms(seed, 0x7A4D, n),
        }
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

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// bare base router: 1 call per task
    Off,
    /// value-of-information gate (the candidate)
    Gate,
    /// ablation: always check with the learned best checker
    Always,
    /// ablation: check at a fixed rate (matched to the gate's per-seed rate)
    Rate(f64),
}

/// Per-run result: [E-success, E-calls, real success, real calls, check rate, best-checker rate]
type VRes = [f64; 6];

fn run_v(base: &Base, mode: Mode, so_cfg: &SecondOpinionConfig, seed: u64, s: &VScen) -> VRes {
    let vw = VWorld::new(seed, s);
    let w = &vw.w;
    let n = w.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut pol = build(base, w);
    let mut so = SecondOpinion::new(n, so_cfg.clone());
    let mut rng = policy_rng(seed);
    let (mut es, mut ec, mut rs, mut rc, mut checks, mut best_ck) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for t in 0..w.len() {
        let st = w.task(t);
        let a = pol.pick(&st.task, &cands, &mut rng);
        let pl = vw.p(t, a);
        let ok = w.outcome(t, &[a]) > 0.5;
        let plan = match mode {
            Mode::Off => None,
            Mode::Gate => so.decide(&st.task, a, &cands),
            Mode::Always => so.plan(&st.task, a, &cands, true),
            Mode::Rate(r) => {
                if vw.u_rand[t] < r { so.plan(&st.task, a, &cands, true) } else { None }
            }
        };
        let (mut succ, mut calls) = (if ok { 1.0 } else { 0.0 }, 1.0);
        let (mut e_succ, mut e_calls) = (pl, 1.0);
        if let Some(pl_) = plan {
            let (c, r) = (pl_.checker, pl_.retry);
            let (q, fa, pr) = (vw.q(t, c), vw.fa(t, c), vw.p(t, r));
            checks += 1.0;
            if (0..n).filter(|&x| x != a).all(|x| vw.q(t, x) <= q + 1e-12) {
                best_ck += 1.0;
            }
            e_succ = pl + (1.0 - pl) * q * pr;
            e_calls = 2.0 + (1.0 - pl) * q + pl * fa;
            calls += 1.0;
            let flagged = if ok { vw.u_catch[t] < fa } else { vw.u_catch[t] < q };
            so.observe_check(&st.task, c, !ok, flagged);
            if flagged {
                calls += 1.0;
                let rok = vw.u_retry[t] < pr;
                so.observe_attempt(&st.task, r, rok);
                if rok {
                    succ = 1.0;
                }
            }
        }
        if mode != Mode::Off {
            so.observe_attempt(&st.task, a, ok);
        }
        pol.fb(&st.task, a, if ok { 1.0 } else { 0.0 });
        es += e_succ;
        ec += e_calls;
        rs += succ;
        rc += calls;
    }
    let nt = w.len() as f64;
    [es / nt, ec / nt, rs / nt, rc / nt, checks / nt, if checks > 0.0 { best_ck / checks } else { f64::NAN }]
}

fn net(r: &VRes, c: f64) -> f64 {
    r[0] - c * r[1]
}

fn eval_v(base: &Base, mode: Mode, so: &SecondOpinionConfig, seeds: &[u64], s: &VScen) -> Vec<VRes> {
    seeds.par_iter().map(|&sd| run_v(base, mode, so, sd, s)).collect()
}

/// Rate-matched ablation: per seed, check at the gate's realised check rate.
fn eval_rate(base: &Base, so: &SecondOpinionConfig, seeds: &[u64], s: &VScen, gate: &[VRes]) -> Vec<VRes> {
    seeds.par_iter().zip(gate).map(|(&sd, g)| run_v(base, Mode::Rate(g[4]), so, sd, s)).collect()
}

// ======================================================================= H3 ===========

#[derive(Clone, Debug)]
struct CScen {
    name: &'static str,
    sc: Scenario,
    /// true: context failure world; false: standard sim.rs global failure
    ctx: bool,
}

fn cscens() -> Vec<CScen> {
    vec![
        CScen { name: "ctx-fixed", sc: Scenario::Fixed, ctx: true },
        CScen { name: "ctx-random", sc: Scenario::Random, ctx: true },
        CScen { name: "ctx-notraps", sc: Scenario::NoTraps, ctx: true },
        CScen { name: "glob-fixed", sc: Scenario::Fixed, ctx: false },
        CScen { name: "glob-random", sc: Scenario::Random, ctx: false },
    ]
}

struct CWorld {
    w: World,
    ctx: bool,
    agent: usize,
    types: Vec<usize>,
    start: usize,
    end: usize,
}

impl CWorld {
    fn new(seed: u64, s: &CScen) -> Self {
        let w = World::new(seed, WorldConfig { scenario: s.sc, ..WorldConfig::default() });
        let t = w.len();
        let mut r = Rng::new(seed.wrapping_mul(0xC7C7_0001) ^ 0xC0_7E57);
        let start = (0.15 * t as f64) as usize + r.below((0.40 * t as f64) as usize);
        let end = start + (0.30 * t as f64) as usize;
        // best agent per type at the window start
        let best: Vec<usize> = (0..8)
            .map(|ty| (0..6).max_by(|&a, &b| w.skill(start, ty, a).total_cmp(&w.skill(start, ty, b))).unwrap())
            .collect();
        let count = |a: usize| best.iter().filter(|&&b| b == a).count();
        let multi: Vec<usize> = (0..6).filter(|&a| count(a) >= 2).collect();
        let pool: Vec<usize> = if multi.is_empty() { (0..6).filter(|&a| count(a) >= 1).collect() } else { multi };
        let agent = pool[r.below(pool.len())];
        let mine: Vec<usize> = (0..8).filter(|&ty| best[ty] == agent).collect();
        let types = vec![mine[r.below(mine.len())]];
        CWorld { w, ctx: s.ctx, agent, types, start, end }
    }
    fn in_fail(&self, t: usize) -> bool {
        if self.ctx { t >= self.start && t < self.end } else { self.w.in_failure(t) }
    }
    fn p(&self, t: usize, a: usize) -> f64 {
        if !self.ctx {
            return self.w.success_prob(t, &[a]);
        }
        let ty = self.w.task(t).type_idx;
        if self.in_fail(t) && a == self.agent && self.types.contains(&ty) {
            0.0
        } else {
            self.w.skill(t, ty, a)
        }
    }
}

/// [fail-window E-success, overall E-success, fail real, overall real]
type CRes = [f64; 4];

fn run_c(base: &Base, seed: u64, s: &CScen) -> CRes {
    let cw = CWorld::new(seed, s);
    let w = &cw.w;
    let n = w.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut pol = build(base, w);
    let mut rng = policy_rng(seed);
    let (mut fe, mut ae, mut fr, mut ar, mut nf) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for t in 0..w.len() {
        let st = w.task(t);
        let a = pol.pick(&st.task, &cands, &mut rng);
        let pe = cw.p(t, a);
        let r = if st.u < pe { 1.0 } else { 0.0 };
        pol.fb(&st.task, a, r);
        ae += pe;
        ar += r;
        if cw.in_fail(t) {
            fe += pe;
            fr += r;
            nf += 1.0;
        }
    }
    let nt = w.len() as f64;
    let nf = f64::max(nf, 1.0);
    [fe / nf, ae / nt, fr / nf, ar / nt]
}

fn eval_c(base: &Base, seeds: &[u64], s: &CScen) -> Vec<CRes> {
    seeds.par_iter().map(|&sd| run_c(base, sd, s)).collect()
}

// ================================================================== reporting =========

fn col<const K: usize>(r: &[[f64; K]], f: impl Fn(&[f64; K]) -> f64) -> Vec<f64> {
    r.iter().map(f).collect()
}

fn ms(x: &[f64]) -> String {
    let s = summarize(x);
    format!("{:.4}+-{:.4}", s.mean, s.half_width)
}

struct Test {
    hyp: &'static str,
    scen: &'static str,
    cand: String,
    metric: String,
    vs: String,
    a: Vec<f64>,
    b: Vec<f64>,
    calls: f64,
}

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = arg(&args, "--out").unwrap_or("results/x2_lab.json".into());
    let tune: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = (31..=60).collect();
    println!("x2_lab: 2000 tasks/run, tune seeds 1-8, held-out 31-60");

    // ------------------------------------------------------------- H1 tuning --------
    let vs = vscens();
    // base router for the second opinion: the best baseline on the TUNE seeds
    let mut base_scores = vec![];
    for (nm, b) in BASELINES.iter() {
        let m: f64 = vs.iter().map(|s| summarize(&col(&eval_v(b, Mode::Off, &SecondOpinionConfig::default(), &tune, s), |r| r[0])).mean).sum::<f64>()
            / vs.len() as f64;
        base_scores.push((nm.to_string(), m));
    }
    let bi = (0..BASELINES.len()).max_by(|&a, &b| base_scores[a].1.total_cmp(&base_scores[b].1)).unwrap();
    let so_base = BASELINES[bi].1.clone();
    println!("[H1 tune 1-8] base success: {}", base_scores.iter().map(|(n, m)| format!("{n} {m:.4}")).collect::<Vec<_>>().join(", "));
    println!("[H1 tune 1-8] second opinion wraps: {}", BASELINES[bi].0);
    let mut so_cfgs: Vec<SecondOpinionConfig> = vec![];
    for &c in COSTS.iter() {
        let mut best = (f64::NEG_INFINITY, SecondOpinionConfig::default());
        for &kappa in &[0.0, 0.5, 1.0, 2.0] {
            for &qp in &[(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)] {
                for &disc in &[0.98, 0.995, 1.0] {
                  for &pool in &[0.0, 2.0, 8.0] {
                    let cfg = SecondOpinionConfig { cost: c, kappa, q_prior: qp, discount: disc, pool, ..SecondOpinionConfig::default() };
                    let o: f64 = vs.iter().map(|s| summarize(&col(&eval_v(&so_base, Mode::Gate, &cfg, &tune, s), |r| net(r, c))).mean).sum::<f64>()
                        / vs.len() as f64;
                    if o > best.0 {
                        best = (o, cfg);
                    }
                  }
                }
            }
        }
        println!("[H1 tune 1-8] cost {c}: best net {:.4} kappa {} q_prior {:?} discount {} pool {}", best.0, best.1.kappa, best.1.q_prior, best.1.discount, best.1.pool);
        so_cfgs.push(best.1);
    }

    // ------------------------------------------------------------- H3 tuning --------
    let cs = cscens();
    let mut best3 = (f64::NEG_INFINITY, Base::MeshRobust);
    for &alpha in &[0.1, 0.3, 0.5] {
        for &th in &[0.3, 0.5, 0.7] {
            for &retry in &[10u64, 20, 40] {
                for &gth in &[0.0, 0.3] {
                  for &warm in &[false, true] {
                    let b = Base::MeshCtx(ContextBenchConfig { alpha, threshold: th, retry_every: retry }, gth, warm);
                    let o: f64 = cs
                        .iter()
                        .map(|s| {
                            let r = eval_c(&b, &tune, s);
                            0.5 * summarize(&col(&r, |x| x[0])).mean + 0.5 * summarize(&col(&r, |x| x[1])).mean
                        })
                        .sum::<f64>()
                        / cs.len() as f64;
                    if o > best3.0 {
                        best3 = (o, b);
                    }
                  }
                }
            }
        }
    }
    let mr_obj: f64 = cs
        .iter()
        .map(|s| {
            let r = eval_c(&Base::MeshRobust, &tune, s);
            0.5 * summarize(&col(&r, |x| x[0])).mean + 0.5 * summarize(&col(&r, |x| x[1])).mean
        })
        .sum::<f64>()
        / cs.len() as f64;
    println!("[H3 tune 1-8] mesh-robust objective {mr_obj:.4}; best context-bench {:.4}: {:?}", best3.0, best3.1);
    let ctx_base = best3.1.clone();

    // ------------------------------------------------------------- held-out ---------
    let mut main_tests: Vec<Test> = vec![];
    let mut abl_tests: Vec<Test> = vec![];
    let mut tables = vec![];
    for s in &vs {
        let b: Vec<Vec<VRes>> = BASELINES.iter().map(|(_, k)| eval_v(k, Mode::Off, &SecondOpinionConfig::default(), &held, s)).collect();
        let bb = (0..b.len()).max_by(|&x, &y| summarize(&col(&b[x], |r| r[0])).mean.total_cmp(&summarize(&col(&b[y], |r| r[0])).mean)).unwrap();
        println!("\n=== H1 {} (held-out 31-60, qscale {}, fascale {}) ===", s.name, s.qscale, s.fascale);
        println!("{:<34}{:>17}{:>9}{:>17}{:>17}{:>9}{:>8}", "policy", "E-success", "calls", "net@0.05", "net@0.10", "real", "best-ck");
        for (i, (nm, _)) in BASELINES.iter().enumerate() {
            let r = &b[i];
            println!("{:<34}{:>17}{:>9.3}{:>17}{:>17}{:>9.4}", nm, ms(&col(r, |x| x[0])), 1.0, ms(&col(r, |x| net(x, 0.05))), ms(&col(r, |x| net(x, 0.1))), summarize(&col(r, |x| x[2])).mean);
        }
        let mut rows = vec![];
        for (ci, &c) in COSTS.iter().enumerate() {
            let cfg = &so_cfgs[ci];
            let g = eval_v(&so_base, Mode::Gate, cfg, &held, s);
            let al = eval_v(&so_base, Mode::Always, cfg, &held, s);
            let rt = eval_rate(&so_base, cfg, &held, s, &g);
            for (nm, r) in [(format!("SO-gate(c={c})"), &g), (format!("always-check(c={c})"), &al), (format!("rate-matched(c={c})"), &rt)] {
                let bc = summarize(&col(r, |x| x[5]).into_iter().filter(|v| v.is_finite()).collect::<Vec<_>>()).mean;
                println!("{:<34}{:>17}{:>9.3}{:>17}{:>17}{:>9.4}{:>8.3}", nm, ms(&col(r, |x| x[0])), summarize(&col(r, |x| x[1])).mean,
                    ms(&col(r, |x| net(x, 0.05))), ms(&col(r, |x| net(x, 0.1))), summarize(&col(r, |x| x[2])).mean, bc);
                rows.push(json!({"policy": nm, "e_success": summarize(&col(r, |x| x[0])), "calls": summarize(&col(r, |x| x[1])),
                    "net": summarize(&col(r, |x| net(x, c))), "real_success": summarize(&col(r, |x| x[2])), "real_calls": summarize(&col(r, |x| x[3])),
                    "check_rate": summarize(&col(r, |x| x[4])), "best_checker_rate": bc}));
            }
            let calls = summarize(&col(&g, |x| x[1])).mean;
            let gcalls = calls;
            for (metric, f) in [("success", None), ("net", Some(c))] {
                let val = |r: &VRes| match f { None => r[0], Some(c) => net(r, c) };
                let mname = if f.is_none() { format!("success(c={c} cfg)") } else { format!("net@{c}") };
                main_tests.push(Test { hyp: "H1", scen: s.name, cand: format!("SO-gate(c={c})"), metric: mname.clone(),
                    vs: BASELINES[bb].0.to_string(), a: col(&g, val), b: col(&b[bb], val), calls: gcalls });
                if metric == "net" {
                    abl_tests.push(Test { hyp: "H1-abl", scen: s.name, cand: format!("SO-gate(c={c})"), metric: mname.clone(),
                        vs: format!("always-check(c={c})"), a: col(&g, val), b: col(&al, val), calls: gcalls });
                    abl_tests.push(Test { hyp: "H1-abl", scen: s.name, cand: format!("SO-gate(c={c})"), metric: mname,
                        vs: format!("rate-matched(c={c})"), a: col(&g, val), b: col(&rt, val), calls: gcalls });
                }
            }
        }
        println!("  best fair baseline = {}", BASELINES[bb].0);
        tables.push(json!({"hyp": "H1", "scenario": s.name, "best_baseline": BASELINES[bb].0,
            "baselines": BASELINES.iter().enumerate().map(|(i, (nm, _))| json!({"policy": nm, "e_success": summarize(&col(&b[i], |x| x[0])), "real_success": summarize(&col(&b[i], |x| x[2]))})).collect::<Vec<_>>(),
            "candidates": rows}));
    }
    for s in &cs {
        let b: Vec<Vec<CRes>> = BASELINES.iter().map(|(_, k)| eval_c(k, &held, s)).collect();
        let c = eval_c(&ctx_base, &held, s);
        println!("\n=== H3 {} (held-out 31-60) expected success ===", s.name);
        println!("{:<34}{:>17}{:>17}{:>10}{:>10}", "policy", "fail-window", "overall", "real-fail", "real-all");
        let mut bbm = [0usize; 2];
        for m in 0..2 {
            bbm[m] = (0..b.len()).max_by(|&x, &y| summarize(&col(&b[x], |r| r[m])).mean.total_cmp(&summarize(&col(&b[y], |r| r[m])).mean)).unwrap();
        }
        for (i, (nm, _)) in BASELINES.iter().enumerate() {
            let r = &b[i];
            println!("{:<34}{:>17}{:>17}{:>10.4}{:>10.4}", nm, ms(&col(r, |x| x[0])), ms(&col(r, |x| x[1])), summarize(&col(r, |x| x[2])).mean, summarize(&col(r, |x| x[3])).mean);
        }
        println!("{:<34}{:>17}{:>17}{:>10.4}{:>10.4}", "mesh-ctx-bench", ms(&col(&c, |x| x[0])), ms(&col(&c, |x| x[1])), summarize(&col(&c, |x| x[2])).mean, summarize(&col(&c, |x| x[3])).mean);
        for (m, mn) in ["fail-window", "overall"].iter().enumerate() {
            main_tests.push(Test { hyp: "H3", scen: s.name, cand: "mesh-ctx-bench".into(), metric: mn.to_string(), vs: BASELINES[bbm[m]].0.to_string(),
                a: col(&c, |x| x[m]), b: col(&b[bbm[m]], |x| x[m]), calls: 1.0 });
            abl_tests.push(Test { hyp: "H3-abl", scen: s.name, cand: "mesh-ctx-bench".into(), metric: mn.to_string(), vs: "mesh-robust".into(),
                a: col(&c, |x| x[m]), b: col(&b[5], |x| x[m]), calls: 1.0 });
        }
        tables.push(json!({"hyp": "H3", "scenario": s.name,
            "rows": BASELINES.iter().enumerate().map(|(i, (nm, _))| json!({"policy": nm, "fail": summarize(&col(&b[i], |x| x[0])), "overall": summarize(&col(&b[i], |x| x[1]))})).chain(std::iter::once(json!({"policy": "mesh-ctx-bench", "fail": summarize(&col(&c, |x| x[0])), "overall": summarize(&col(&c, |x| x[1]))}))).collect::<Vec<_>>()}));
    }

    // ------------------------------------------------------------- statistics -------
    let mut js = vec![];
    for (fam, tests) in [("main: vs best fair baseline", &main_tests), ("ablations", &abl_tests)] {
        let p: Vec<f64> = tests.iter().map(|t| paired_t_pvalue(&t.a, &t.b)).collect();
        let adj = holm_adjust(&p);
        println!("\n#### family `{fam}` ({} tests, Holm)", tests.len());
        for (k, t) in tests.iter().enumerate() {
            let d = paired_diff(&t.a, &t.b);
            let v = if adj[k] < 0.05 { if d.diff.mean > 0.0 { "BEATS" } else { "WORSE" } } else { "n.s." };
            println!("{:<6} {:<12} {:<20} {:<22} vs {:<22} {:+.4} CI[{:+.4},{:+.4}] wins {:>2}/{} calls/task {:.3} Holm p={:.4} {}",
                t.hyp, t.scen, t.cand, t.metric, t.vs, d.diff.mean, d.diff.lo, d.diff.hi, d.wins, t.a.len(), t.calls, adj[k], v);
            js.push(json!({"family": fam, "hyp": t.hyp, "scenario": t.scen, "cand": t.cand, "metric": t.metric, "vs": t.vs,
                "diff": d.diff, "wins": d.wins, "losses": d.losses, "calls_per_task": t.calls, "p": p[k], "holm_p": adj[k], "verdict": v}));
        }
    }
    let j = json!({
        "protocol": "tune seeds 1-8, held-out 31-60, paired 95% t-CI, Holm per family",
        "h1_base": BASELINES[bi].0, "h1_tuned": so_cfgs.iter().map(|c| json!({"cost": c.cost, "kappa": c.kappa, "q_prior": c.q_prior, "discount": c.discount, "pool": c.pool})).collect::<Vec<_>>(),
        "h3_tuned": format!("{:?}", ctx_base), "h3_tune_objective": best3.0, "h3_mesh_robust_tune_objective": mr_obj,
        "tables": tables, "tests": js});
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("wrote {out}");
}
