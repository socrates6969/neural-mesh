//! G7 (W7, Q-INSPIRED): do quantum-inspired CLASSICAL optimizers (simulated annealing, parallel tempering,
//! QAOA-style layered cost/mixer schedule) help (a) agent-TEAM assignment and (b) FEATURE->AGENT binary
//! assignment, versus the fair baselines {keyword, eps-greedy, pooled-thompson, hebbian-robust, warm(hebr),
//! guarded-safe} (+ NeuralMesh for teams)? Nothing here is quantum and nothing claims a speedup.
//!
//! Protocol: all configs tuned on seeds 1-8 only (baselines: per world, per family, generous; optimizer arms:
//! ONE pooled config across the worlds of a part, conservative); evaluated on held-out 31-60 (`--set held`)
//! and fresh 61-90 (`--set confirm`) with frozen configs. Paired 95% t-CI vs the best fair baseline per world
//! (best chosen on the EVALUATION set, i.e. conservative), Holm over all arm-world cells.
//!
//! Usage: g7_qinspired [--set held|confirm|dev] [--part all|team|asg|iters] [--out results/g7_qinspired.json]

#[path = "g7_common/mod.rs"]
mod common;

use common::pols::*;
use common::world::*;
use neural_mesh::router::Rng;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::time::Instant;

#[derive(Clone, Debug)]
struct Out {
    overall: f64,
    late: f64,
    evals: f64,
}

// ------------------------------------------------------------------ runners
fn run_team(w: &TeamWorld, pol: &mut dyn TeamPol, seed: u64) -> Out {
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let t = w.cfg.t;
    let (mut s, mut sl) = (0.0, 0.0);
    for i in 0..t {
        let task = &w.tasks[i];
        let team = pol.select(task, &mut rng);
        debug_assert_eq!(team.len(), w.cfg.k);
        let r = w.reward(i, &team);
        pol.observe(task, &team, r);
        s += r;
        if i >= t * 3 / 4 {
            sl += r;
        }
    }
    Out { overall: s / t as f64, late: sl / (t - t * 3 / 4) as f64, evals: pol.evals() }
}

fn run_asg(w: &AsgWorld, pol: &mut dyn AsgPol, seed: u64) -> Out {
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let t = w.cfg.t;
    let (mut s, mut sl) = (0.0, 0.0);
    let mut recent: VecDeque<usize> = VecDeque::new();
    for i in 0..t {
        let task = &w.tasks[i];
        let a = pol.select(task, &mut rng);
        let load = 1 + recent.iter().filter(|&&x| x == a).count();
        let p = w.prob(i, a, load);
        let r = if w.u[i] < p { 1.0 } else { 0.0 };
        pol.observe(task, a, r);
        recent.push_back(a);
        if recent.len() > B_WIN - 1 {
            recent.pop_front();
        }
        s += r;
        if i >= t * 3 / 4 {
            sl += r;
        }
    }
    Out { overall: s / t as f64, late: sl / (t - t * 3 / 4) as f64, evals: pol.evals() }
}

// ------------------------------------------------------------------ ceiling diagnostic (cheats: knows the hidden table)
struct OracleAsg {
    pre: Vec<usize>,
    post: Vec<usize>,
    drift_at: usize,
}
impl OracleAsg {
    fn new(w: &AsgWorld) -> Self {
        use common::solve::*;
        let solve = |q: &Vec<Vec<f64>>| -> Vec<usize> {
            let val: Vec<f64> = (0..NTYPES * B_AGENTS).map(|i| q[i / B_AGENTS][i % B_AGENTS] / NTYPES as f64).collect();
            let p = Potts { nf: NTYPES, na: B_AGENTS, val, w: vec![1.0 / NTYPES as f64; NTYPES], cap: B_CAP as f64 / B_WIN as f64, lam: 20.0 * w.cfg.kappa / 0.15 };
            let mut r = Rng::new(5);
            let (s, _) = sa(&p, &mut r, 20000, 0.05, 0.0005, Some(p.argmax()));
            s.assign
        };
        OracleAsg { pre: solve(w.q_matrix(false)), post: solve(w.q_matrix(true)), drift_at: w.cfg.t / 2 }
    }
}
impl AsgPol for OracleAsg {
    fn name(&self) -> String {
        "oracle-static(knows q)".into()
    }
    fn select(&mut self, task: &neural_mesh::router::Task, _rng: &mut Rng) -> usize {
        let mut c = [0usize; NTYPES];
        for f in &task.features {
            c[feat_index(f) / 4] += 1;
        }
        let ty = (0..NTYPES).max_by_key(|&i| c[i]).unwrap();
        if task.id as usize >= self.drift_at { self.post[ty] } else { self.pre[ty] }
    }
    fn observe(&mut self, _t: &neural_mesh::router::Task, _a: usize, _r: f64) {}
}

// ------------------------------------------------------------------ specs
#[derive(Clone, Debug)]
enum TSpec {
    Fair(Kind),
    Mesh(bool),
    Q(QCfg, Solver),
}
impl TSpec {
    fn label(&self) -> String {
        match self {
            TSpec::Fair(k) => format!("seq:{}", k.label()),
            TSpec::Mesh(true) => "mesh(guarded-safe)".into(),
            TSpec::Mesh(false) => "mesh(hebbian-robust)".into(),
            TSpec::Q(c, s) => format!("q{}-{}", if c.pair { "" } else { "add" }, s.label()),
        }
    }
    fn family(&self) -> String {
        match self {
            TSpec::Fair(k) => format!("seq:{}", k.family()),
            _ => self.label(),
        }
    }
    fn build(&self, w: &TeamWorld) -> Box<dyn TeamPol> {
        let (n, k) = (w.cfg.n, w.cfg.k);
        match self {
            TSpec::Fair(kind) => Box::new(SeqTeam::new(kind, &w.kw, n, k)),
            TSpec::Mesh(true) => Box::new(MeshTeam::guarded(&w.kw, n, k)),
            TSpec::Mesh(false) => Box::new(MeshTeam::heb(n, k)),
            TSpec::Q(c, s) => Box::new(QPol::new(c.clone(), *s, &w.kw, n, k)),
        }
    }
}

#[derive(Clone, Debug)]
enum ASpec {
    Fair(Kind),
    Z(ZCfg, ZSolver),
}
impl ASpec {
    fn label(&self) -> String {
        match self {
            ASpec::Fair(k) => k.label(),
            ASpec::Z(_, s) => format!("z-{}", s.label()),
        }
    }
    fn family(&self) -> String {
        match self {
            ASpec::Fair(k) => k.family().to_string(),
            _ => self.label(),
        }
    }
    fn build(&self, w: &AsgWorld) -> Box<dyn AsgPol> {
        match self {
            ASpec::Fair(kind) => Box::new(FairAsg::new(kind, &w.kw, B_AGENTS)),
            ASpec::Z(c, s) => Box::new(ZPol::new(c.clone(), *s, &w.kw, B_AGENTS)),
        }
    }
}

fn n_combos(n: usize, k: usize) -> f64 {
    let mut c = 1.0f64;
    for i in 0..k {
        c = c * (n - i) as f64 / (i + 1) as f64;
    }
    c
}

// ------------------------------------------------------------------ evaluation helpers
fn eval_team(spec: &TSpec, cfg: &TeamCfg, seeds: &[u64]) -> Vec<Out> {
    seeds
        .par_iter()
        .map(|&s| {
            let w = TeamWorld::new(s, cfg.clone());
            let mut p = spec.build(&w);
            run_team(&w, p.as_mut(), s)
        })
        .collect()
}

fn eval_asg(spec: &ASpec, cfg: &AsgCfg, seeds: &[u64]) -> Vec<Out> {
    seeds
        .par_iter()
        .map(|&s| {
            let w = AsgWorld::new(s, cfg.clone());
            let mut p = spec.build(&w);
            run_asg(&w, p.as_mut(), s)
        })
        .collect()
}

fn mean_of(v: &[Out], f: fn(&Out) -> f64) -> f64 {
    v.iter().map(f).sum::<f64>() / v.len() as f64
}
fn ov(o: &Out) -> f64 {
    o.overall
}
fn lt(o: &Out) -> f64 {
    o.late
}
fn col(v: &[Out], f: fn(&Out) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}

fn time_team(spec: &TSpec, cfg: &TeamCfg, seed: u64) -> f64 {
    let w = TeamWorld::new(seed, cfg.clone());
    let mut p = spec.build(&w);
    let t0 = Instant::now();
    let _ = run_team(&w, p.as_mut(), seed);
    t0.elapsed().as_nanos() as f64 / cfg.t as f64
}
fn time_asg(spec: &ASpec, cfg: &AsgCfg, seed: u64) -> f64 {
    let w = AsgWorld::new(seed, cfg.clone());
    let mut p = spec.build(&w);
    let t0 = Instant::now();
    let _ = run_asg(&w, p.as_mut(), seed);
    t0.elapsed().as_nanos() as f64 / cfg.t as f64
}

// ------------------------------------------------------------------ tuning
fn q_grid() -> Vec<QCfg> {
    let mut v = vec![];
    for &lr in &[0.1, 0.3] {
        for &eps in &[0.02, 0.08] {
            for &c1 in &[0.0, 0.1] {
                for &c2 in &[0.0, 0.05] {
                    for &rho in &[0.0, 0.15] {
                        v.push(QCfg { lr, l2: 0.0005, eps, c1, c2, rho, pair: true });
                    }
                }
            }
        }
    }
    v
}

fn z_grid() -> Vec<ZCfg> {
    let mut v = vec![];
    for &lam in &[0.0, 3.0, 10.0, 30.0] {
        for &cap in &[0.25, 0.3, 0.4] {
            for &c in &[0.0, 0.1] {
                for &eps in &[0.02, 0.06] {
                    for &rho in &[2.0, 6.0] {
                        for &gamma in &[1.0, 0.99] {
                            for &pm_other in &[0.5, 1.0] {
                                v.push(ZCfg { period: 20, lam, cap, c, rho, eps, gamma, pm_other });
                            }
                        }
                    }
                }
            }
        }
    }
    v
}

fn pick_best<T: Clone>(items: &[T], score: impl Fn(&T) -> f64 + Sync, ) -> (T, f64)
where
    T: Sync,
{
    let sc: Vec<f64> = items.par_iter().map(|c| score(c)).collect();
    let mut bi = 0;
    for i in 1..sc.len() {
        if sc[i] > sc[bi] {
            bi = i;
        }
    }
    (items[bi].clone(), sc[bi])
}

fn seq_eval_team(spec: &TSpec, cfg: &TeamCfg, seeds: &[u64]) -> f64 {
    // sequential over seeds (outer level is already parallel)
    let mut s = 0.0;
    for &sd in seeds {
        let w = TeamWorld::new(sd, cfg.clone());
        let mut p = spec.build(&w);
        s += run_team(&w, p.as_mut(), sd).overall;
    }
    s / seeds.len() as f64
}
fn seq_eval_asg(spec: &ASpec, cfg: &AsgCfg, seeds: &[u64]) -> f64 {
    let mut s = 0.0;
    for &sd in seeds {
        let w = AsgWorld::new(sd, cfg.clone());
        let mut p = spec.build(&w);
        s += run_asg(&w, p.as_mut(), sd).overall;
    }
    s / seeds.len() as f64
}

/// per family, the best grid config on the tune seeds (one world)
fn tune_fair_team(cfg: &TeamCfg, tune: &[u64]) -> Vec<TSpec> {
    let grid = fair_grid();
    let sc: Vec<f64> = grid.par_iter().map(|k| seq_eval_team(&TSpec::Fair(k.clone()), cfg, tune)).collect();
    let mut best: Vec<(String, usize)> = vec![];
    for (i, k) in grid.iter().enumerate() {
        let fam = k.family().to_string();
        match best.iter_mut().find(|(f, _)| *f == fam) {
            Some(e) => {
                if sc[i] > sc[e.1] {
                    e.1 = i;
                }
            }
            None => best.push((fam, i)),
        }
    }
    let mut v: Vec<TSpec> = best.iter().map(|(_, i)| TSpec::Fair(grid[*i].clone())).collect();
    v.push(TSpec::Mesh(true));
    v.push(TSpec::Mesh(false));
    v
}

fn tune_fair_asg(cfg: &AsgCfg, tune: &[u64]) -> Vec<ASpec> {
    let grid = fair_grid();
    let sc: Vec<f64> = grid.par_iter().map(|k| seq_eval_asg(&ASpec::Fair(k.clone()), cfg, tune)).collect();
    let mut best: Vec<(String, usize)> = vec![];
    for (i, k) in grid.iter().enumerate() {
        let fam = k.family().to_string();
        match best.iter_mut().find(|(f, _)| *f == fam) {
            Some(e) => {
                if sc[i] > sc[e.1] {
                    e.1 = i;
                }
            }
            None => best.push((fam, i)),
        }
    }
    best.iter().map(|(_, i)| ASpec::Fair(grid[*i].clone())).collect()
}

fn tune_q(tws: &[TeamCfg], tune: &[u64], label: &str) -> (QCfg, QCfg) {
    let grid = q_grid();
    let (bc, bs) = pick_best(&grid, |c| {
        let mut s = 0.0;
        for cfg in tws {
            s += seq_eval_team(&TSpec::Q(c.clone(), Solver::Sa(300)), cfg, tune);
        }
        s / tws.len() as f64
    });
    println!("[tune A:{label}] q-sa300 config (mean success {:.4}): {:?}", bs, bc);
    let gridadd: Vec<QCfg> = grid.iter().map(|c| QCfg { pair: false, c2: 0.0, ..c.clone() }).collect();
    let (ba, bas) = pick_best(&gridadd, |c| {
        let mut s = 0.0;
        for cfg in tws {
            s += seq_eval_team(&TSpec::Q(c.clone(), Solver::Greedy), cfg, tune);
        }
        s / tws.len() as f64
    });
    println!("[tune A:{label}] additive (no-J) greedy config (mean success {:.4}): {:?}", bas, ba);
    (bc, ba)
}

fn tune_z(aws: &[AsgCfg], tune: &[u64], label: &str) -> (ZCfg, ZCfg) {
    let grid = z_grid();
    let (bc, bs) = pick_best(&grid, |c| {
        let mut s = 0.0;
        for cfg in aws {
            s += seq_eval_asg(&ASpec::Z(c.clone(), ZSolver::Sa(400)), cfg, tune);
        }
        s / aws.len() as f64
    });
    println!("[tune B:{label}] z-sa400 config (mean success {:.4}): {:?}", bs, bc);
    let gridam: Vec<ZCfg> = grid.iter().filter(|c| c.lam == 0.0 && c.cap == 0.3).cloned().collect();
    let (ba, bas) = pick_best(&gridam, |c| {
        let mut s = 0.0;
        for cfg in aws {
            s += seq_eval_asg(&ASpec::Z(c.clone(), ZSolver::Argmax), cfg, tune);
        }
        s / aws.len() as f64
    });
    println!("[tune B:{label}] z-argmax (no load term) config (mean success {:.4}): {:?}", bas, ba);
    (bc, ba)
}

// ------------------------------------------------------------------ reporting
struct Cell {
    world: String,
    arm: String,
    diff_mean: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    losses: usize,
    p: f64,
    late_diff: f64,
    late_p: f64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let set = arg("--set").unwrap_or("held".into());
    let part = arg("--part").unwrap_or("all".into());
    let out = arg("--out");
    let tmul: usize = arg("--tmul").and_then(|x| x.parse().ok()).unwrap_or(1);
    let per_world = arg("--tune").map(|x| x == "perworld").unwrap_or(false);
    let tune: Vec<u64> = (1..=8).collect();
    let seeds: Vec<u64> = match set.as_str() {
        "held" => (31..=60).collect(),
        "confirm" => (61..=90).collect(),
        "dev" => (101..=130).collect(),
        _ => panic!("--set held|confirm"),
    };
    println!("G7 quantum-inspired classical optimizers. eval set {set} ({} seeds); optimizer config tuning: {}; all tuning on seeds 1-8", seeds.len(), if per_world { "PER WORLD (as generous as the baselines)" } else { "POOLED over the worlds of a part (conservative)" });
    println!("note: iteration sweep and holm family cover only the arms listed below");
    let mut result = serde_json::Map::new();
    result.insert("set".into(), json!(set));
    result.insert("tune_mode".into(), json!(if per_world { "perworld" } else { "pooled" }));
    let mut cells: Vec<Cell> = vec![];

    // =============================================================== PART A: teams
    let mut qcfg_pair = QCfg::default();
    let mut qcfg_add = QCfg { pair: false, ..QCfg::default() };
    if (part == "all" || part == "team" || part == "iters") && !per_world {
        let (a, b) = tune_q(&team_worlds(), &tune, "pooled");
        qcfg_pair = a;
        qcfg_add = b;
        result.insert("tuned_q_pair".into(), json!(format!("{:?}", qcfg_pair)));
        result.insert("tuned_q_add".into(), json!(format!("{:?}", qcfg_add)));
    }
    if part == "all" || part == "team" {
        let mut worlds_json = serde_json::Map::new();
        for cfg in team_worlds() {
            let fair = tune_fair_team(&cfg, &tune);
            let (qcfg_pair, qcfg_add) = if per_world { tune_q(std::slice::from_ref(&cfg), &tune, cfg.name) } else { (qcfg_pair.clone(), qcfg_add.clone()) };
            let mut arms: Vec<TSpec> = vec![
                TSpec::Q(qcfg_pair.clone(), Solver::Sa(300)),
                TSpec::Q(qcfg_pair.clone(), Solver::Pt(300)),
                TSpec::Q(qcfg_pair.clone(), Solver::Qaoa(300)),
                TSpec::Q(qcfg_pair.clone(), Solver::Greedy),
                TSpec::Q(qcfg_add.clone(), Solver::Greedy),
            ];
            if n_combos(cfg.n, cfg.k) <= 30000.0 {
                arms.push(TSpec::Q(qcfg_pair.clone(), Solver::Exact));
            }
            let (orc, rnd) = TeamWorld::new(seeds[0], cfg.clone()).references(seeds[0]);
            println!("\n=== {} (N={}, K={}, T={}, C(N,K)={:.0}) refs[seed {}]: oracle p {:.3}, random-team p {:.3}", cfg.name, cfg.n, cfg.k, cfg.t, n_combos(cfg.n, cfg.k), seeds[0], orc, rnd);
            let fair_res: Vec<(TSpec, Vec<Out>)> = fair.iter().map(|s| (s.clone(), eval_team(s, &cfg, &seeds))).collect();
            let arm_res: Vec<(TSpec, Vec<Out>)> = arms.iter().map(|s| (s.clone(), eval_team(s, &cfg, &seeds))).collect();
            let bf = fair_res.iter().enumerate().max_by(|a, b| mean_of(&a.1 .1, ov).total_cmp(&mean_of(&b.1 .1, ov))).unwrap().0;
            let bfl = fair_res.iter().enumerate().max_by(|a, b| mean_of(&a.1 .1, lt).total_cmp(&mean_of(&b.1 .1, lt))).unwrap().0;
            let mut rows = vec![];
            println!("{:<34} {:>8} {:>8} {:>8} {:>10}", "policy", "overall", "late", "evals/d", "ns/dec");
            for (s, r) in fair_res.iter().chain(arm_res.iter()) {
                let ns = time_team(s, &cfg, seeds[0]);
                println!("{:<34} {:>8.4} {:>8.4} {:>8.1} {:>10.0}{}", s.label(), mean_of(r, ov), mean_of(r, lt), mean_of(r, |o| o.evals), ns, if s.label() == fair_res[bf].0.label() { "   <- best fair" } else { "" });
                rows.push(json!({"policy": s.label(), "family": s.family(), "overall": mean_of(r, ov), "late": mean_of(r, lt), "evals_per_decision": mean_of(r, |o| o.evals), "ns_per_decision": ns, "spec": format!("{:?}", s)}));
            }
            for (s, r) in &arm_res {
                let b = &fair_res[bf].1;
                let pd = paired_diff(&col(r, ov), &col(b, ov));
                let p = paired_t_pvalue(&col(r, ov), &col(b, ov));
                let bl = &fair_res[bfl].1;
                let ld = paired_diff(&col(r, lt), &col(bl, lt));
                let lp = paired_t_pvalue(&col(r, lt), &col(bl, lt));
                cells.push(Cell { world: cfg.name.into(), arm: s.label(), diff_mean: pd.diff.mean, lo: pd.diff.lo, hi: pd.diff.hi, wins: pd.wins, losses: pd.losses, p, late_diff: ld.diff.mean, late_p: lp });
            }
            worlds_json.insert(cfg.name.into(), json!({"oracle_p": orc, "random_team_p": rnd, "best_fair": fair_res[bf].0.label(), "rows": rows}));
        }
        result.insert("team".into(), Value::Object(worlds_json));
    }

    // =============================================================== PART B: assignment
    let mut zcfg = ZCfg::default();
    let mut zcfg_am = ZCfg { lam: 0.0, ..ZCfg::default() };
    if part == "all" || part == "asg" {
        let aws: Vec<AsgCfg> = asg_worlds().into_iter().map(|mut c| { c.t *= tmul; c }).collect();
        if !per_world {
            let (a, b) = tune_z(&aws, &tune, "pooled");
            zcfg = a;
            zcfg_am = b;
            result.insert("tuned_z".into(), json!(format!("{:?}", zcfg)));
            result.insert("tuned_z_argmax".into(), json!(format!("{:?}", zcfg_am)));
        }
        let mut worlds_json = serde_json::Map::new();
        for cfg in aws {
            let fair = tune_fair_asg(&cfg, &tune);
            let (zcfg, zcfg_am) = if per_world { tune_z(std::slice::from_ref(&cfg), &tune, cfg.name) } else { (zcfg.clone(), zcfg_am.clone()) };
            let arms = vec![
                ASpec::Z(zcfg.clone(), ZSolver::Sa(400)),
                ASpec::Z(zcfg.clone(), ZSolver::Pt(400)),
                ASpec::Z(zcfg.clone(), ZSolver::Qaoa(400)),
                ASpec::Z(zcfg.clone(), ZSolver::Greedy),
                ASpec::Z(zcfg_am.clone(), ZSolver::Argmax),
            ];
            let orc = AsgWorld::new(seeds[0], cfg.clone()).oracle_nocong();
            println!("\n=== {} (feature->agent, F={}, A={}, T={}, kappa={}) ref[seed {}]: best-single no-congestion quality {:.3}", cfg.name, NFEAT, B_AGENTS, cfg.t, cfg.kappa, seeds[0], orc);
            let fair_res: Vec<(ASpec, Vec<Out>)> = fair.iter().map(|s| (s.clone(), eval_asg(s, &cfg, &seeds))).collect();
            let arm_res: Vec<(ASpec, Vec<Out>)> = arms.iter().map(|s| (s.clone(), eval_asg(s, &cfg, &seeds))).collect();
            let bf = fair_res.iter().enumerate().max_by(|a, b| mean_of(&a.1 .1, ov).total_cmp(&mean_of(&b.1 .1, ov))).unwrap().0;
            let bfl = fair_res.iter().enumerate().max_by(|a, b| mean_of(&a.1 .1, lt).total_cmp(&mean_of(&b.1 .1, lt))).unwrap().0;
            let mut rows = vec![];
            println!("{:<34} {:>8} {:>8} {:>8} {:>10}", "policy", "overall", "late", "evals/d", "ns/dec");
            for (s, r) in fair_res.iter().chain(arm_res.iter()) {
                let ns = time_asg(s, &cfg, seeds[0]);
                println!("{:<34} {:>8.4} {:>8.4} {:>8.1} {:>10.0}{}", s.label(), mean_of(r, ov), mean_of(r, lt), mean_of(r, |o| o.evals), ns, if s.label() == fair_res[bf].0.label() { "   <- best fair" } else { "" });
                rows.push(json!({"policy": s.label(), "family": s.family(), "overall": mean_of(r, ov), "late": mean_of(r, lt), "evals_per_decision": mean_of(r, |o| o.evals), "ns_per_decision": ns, "spec": format!("{:?}", s)}));
            }
            for (s, r) in &arm_res {
                let b = &fair_res[bf].1;
                let pd = paired_diff(&col(r, ov), &col(b, ov));
                let p = paired_t_pvalue(&col(r, ov), &col(b, ov));
                let bl = &fair_res[bfl].1;
                let ld = paired_diff(&col(r, lt), &col(bl, lt));
                let lp = paired_t_pvalue(&col(r, lt), &col(bl, lt));
                cells.push(Cell { world: cfg.name.into(), arm: s.label(), diff_mean: pd.diff.mean, lo: pd.diff.lo, hi: pd.diff.hi, wins: pd.wins, losses: pd.losses, p, late_diff: ld.diff.mean, late_p: lp });
            }
            worlds_json.insert(cfg.name.into(), json!({"oracle_nocong": orc, "best_fair": fair_res[bf].0.label(), "rows": rows}));
        }
        result.insert("asg".into(), Value::Object(worlds_json));
    }

    if part == "all" || part == "asg" {
        println!("
=== Ceiling diagnostic (NOT a competitor: knows the hidden quality table and task type): static balanced assignment");
        for cfg in asg_worlds() {
            let r: Vec<Out> = seeds.par_iter().map(|&s| {
                let w = AsgWorld::new(s, cfg.clone());
                let mut p = OracleAsg::new(&w);
                run_asg(&w, &mut p, s)
            }).collect();
            println!("{:<14} oracle-static {:.4}", cfg.name, mean_of(&r, ov));
        }
    }

    // =============================================================== iteration sweep (team worlds)
    if part == "all" || part == "iters" {
        println!("\n=== SA iteration sweep (frozen surrogate config), held seeds: success vs move-evaluations/decision");
        let mut sweep = serde_json::Map::new();
        for cfg in team_worlds().into_iter().filter(|c| ["A-cover", "A-syn", "A-syn-L"].contains(&c.name)) {
            let mut line = vec![];
            let fair = tune_fair_team(&cfg, &tune);
            let fr: Vec<f64> = fair.iter().map(|s| mean_of(&eval_team(s, &cfg, &seeds), ov)).collect();
            let best_fair = fr.iter().cloned().fold(0.0, f64::max);
            println!("{} best fair {:.4}", cfg.name, best_fair);
            let mut specs: Vec<TSpec> = [10usize, 30, 100, 300, 1000, 3000].iter().map(|&i| TSpec::Q(qcfg_pair.clone(), Solver::Sa(i))).collect();
            specs.push(TSpec::Q(qcfg_pair.clone(), Solver::Greedy));
            if n_combos(cfg.n, cfg.k) <= 30000.0 {
                specs.push(TSpec::Q(qcfg_pair.clone(), Solver::Exact));
            }
            for s in specs {
                let r = eval_team(&s, &cfg, &seeds);
                let ns = time_team(&s, &cfg, seeds[0]);
                println!("  {:<12} success {:.4}  evals/dec {:>7.1}  ns/dec {:>9.0}", s.label(), mean_of(&r, ov), mean_of(&r, |o| o.evals), ns);
                line.push(json!({"policy": s.label(), "overall": mean_of(&r, ov), "evals_per_decision": mean_of(&r, |o| o.evals), "ns_per_decision": ns}));
            }
            sweep.insert(cfg.name.into(), json!({"best_fair": best_fair, "rows": line}));
        }
        result.insert("iter_sweep".into(), Value::Object(sweep));
    }

    // =============================================================== Holm + verdict
    if !cells.is_empty() {
        let ps: Vec<f64> = cells.iter().map(|c| c.p).collect();
        let adj = holm_adjust(&ps);
        let lps: Vec<f64> = cells.iter().map(|c| c.late_p).collect();
        let ladj = holm_adjust(&lps);
        println!("\n=== Optimizer arms vs best fair baseline per world (overall success; paired 95% t-CI; Holm over {} cells)", cells.len());
        println!("{:<14} {:<18} {:>9} {:>20} {:>6}/{:<6} {:>8} {:>7} | late diff (Holm)", "world", "arm", "diff", "95% CI", "win", "loss", "p", "Holm");
        let mut cj = vec![];
        for (i, c) in cells.iter().enumerate() {
            let verdict = if adj[i] < 0.05 && c.lo > 0.0 { "WIN" } else if adj[i] < 0.05 && c.hi < 0.0 { "LOSS" } else { "n.s." };
            println!("{:<14} {:<18} {:>+9.4} [{:>+8.4},{:>+8.4}] {:>6}/{:<6} {:>8.4} {:>7.4} {} | {:>+8.4} ({:.4})", c.world, c.arm, c.diff_mean, c.lo, c.hi, c.wins, c.losses, c.p, adj[i], verdict, c.late_diff, ladj[i]);
            cj.push(json!({"world": c.world, "arm": c.arm, "diff": c.diff_mean, "lo": c.lo, "hi": c.hi, "wins": c.wins, "losses": c.losses, "p": c.p, "holm": adj[i], "verdict": verdict, "late_diff": c.late_diff, "late_holm": ladj[i]}));
        }
        // headline: SA / PT / QAOA wins per world
        println!("\nHeadline (SA/PT/QAOA arms): worlds with a Holm WIN:");
        let mut win_worlds: Vec<String> = vec![];
        for (i, c) in cells.iter().enumerate() {
            let head = c.arm.ends_with("sa300") || c.arm.ends_with("pt300") || c.arm.ends_with("qaoa300") || c.arm.ends_with("sa400") || c.arm.ends_with("pt400") || c.arm.ends_with("qaoa400");
            let pairwise = !c.arm.starts_with("qadd");
            if head && pairwise && adj[i] < 0.05 && c.lo > 0.0 && !win_worlds.contains(&c.world) {
                win_worlds.push(c.world.clone());
            }
        }
        println!("  {:?}", win_worlds);
        let _ = summarize(&[0.0, 1.0]);
        result.insert("cells".into(), json!(cj));
        result.insert("headline_win_worlds".into(), json!(win_worlds));
    }
    if let Some(path) = out {
        std::fs::write(&path, serde_json::to_string_pretty(&Value::Object(result)).unwrap()).unwrap();
        println!("wrote {path}");
    }
}
