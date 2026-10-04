//! E3: does the full research-session pipeline (retrieve -> route -> draft -> critic -> record)
//! beat plain routing on task QUALITY in a seeded synthetic multi-step research world?
//!
//! Arms (all share the SAME world draws: stateless hashed randomness keyed on (seed, task, agent,
//! purpose), so paired differences are low-variance):
//!   plain, plain-g            : router only, raw hook bit feedback (warm(hebr)+health / guarded(hebr)+health)
//!   full, full-g              : retrieval + critic + reuse
//!   -retr, -crit, -reuse      : full minus one part (warm router)
//!   retr-only, crit-only, reuse-only : one part on top of plain
//! Worlds: halluc-recur (main), low-error, diluted-embedding, lazy-checkers, recur-only, irrelevant-notes, and the
//! no-harm control equal-agents (nothing to win; pipeline should tie).
//! Protocol: pipeline knobs tuned on seeds 1-8 only; held-out 31-60 (default); confirmation 61-90
//! with the SAME frozen tuning (--held 61-90). Paired 95% t-CI, Holm across all tests of a run.
//!
//! Usage: cargo run --release --bin e3_research [-- --out results/e3_research.json] [--held 61-90]

use std::collections::HashMap;
use std::time::Instant;

use neural_mesh::research::*;
use neural_mesh::router::{Rng, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue};
use neural_mesh::variants::{HierSoConfig, SecondOpinionConfig};
use rayon::prelude::*;
use serde_json::{json, Value};

const N_TASKS: usize = 2000;
const N_AGENTS: usize = 8;
const CALL_COST: f64 = 0.03; // reward units per EXTRA agent call (critic call / retry call)
const REVEAL: f64 = 0.5; // fraction of tasks whose graded outcome ever arrives
const DELAY: usize = 20;

// =============================================================================================
// stateless randomness (common random numbers across arms)

fn hash(seed: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ a.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ b.wrapping_mul(0x94D0_49BB_1331_11EB) ^ c.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn uni(seed: u64, a: u64, b: u64, c: u64) -> f64 {
    (hash(seed, a, b, c) >> 11) as f64 / (1u64 << 53) as f64
}
fn norm(seed: u64, a: u64, b: u64, c: u64) -> f64 {
    let (u1, u2) = (uni(seed, a, b, c).max(1e-12), uni(seed, a, b, c ^ 0xABCD));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

// =============================================================================================
// world

#[derive(Clone, Debug)]
struct WorldCfg {
    name: &'static str,
    topics: usize,
    recur: bool,       // features carry only the topic words (retrievable); else unique tokens dilute them
    equal: bool,       // all agents identical (control)
    hal_base: f64,     // P(hallucination) = hal_base * (1 - skill)
    ctx_gain: f64,     // verified-note quality gain: q += ctx_gain * note * (1 - s)
    ctx_hal_cut: f64,  // note cuts hallucination: hal *= 1 - cut * note
    lazy: LazyMode,    // checker laziness
    decoy: f64,        // P(keyword map advertises a wrong agent) per topic
    outage: bool,      // agent 7 down for tasks 600..1000
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum LazyMode {
    Third, // agents with idx % 3 == 2 rubber-stamp
    All,
    None,
}

fn worlds() -> Vec<WorldCfg> {
    let base = WorldCfg { name: "halluc-recur", topics: 12, recur: true, equal: false, hal_base: 0.7, ctx_gain: 0.4, ctx_hal_cut: 0.6, lazy: LazyMode::Third, decoy: 0.3, outage: true };
    vec![
        base.clone(),
        WorldCfg { name: "low-error", hal_base: 0.12, ctx_gain: 0.15, ctx_hal_cut: 0.3, ..base.clone() },
        WorldCfg { name: "diluted-embedding", recur: false, ..base.clone() },
        WorldCfg { name: "lazy-checkers", lazy: LazyMode::All, ..base.clone() },
        WorldCfg { name: "recur-only", hal_base: 0.0, ctx_gain: 0.4, ctx_hal_cut: 0.0, lazy: LazyMode::None, ..base.clone() },
        WorldCfg { name: "irrelevant-notes", ctx_gain: 0.0, ctx_hal_cut: 0.0, ..base.clone() },
        WorldCfg { name: "control-equal", equal: true, hal_base: 0.0, ctx_gain: 0.0, ctx_hal_cut: 0.0, lazy: LazyMode::None, decoy: 0.0, outage: false, ..base.clone() },
    ]
}

struct Instance {
    skill: Vec<Vec<f64>>, // [agent][topic]
    tasks: Vec<(Task, usize)>,
    kw: HashMap<String, usize>,
}

fn build(w: &WorldCfg, seed: u64, widx: u64) -> Instance {
    let s = seed.wrapping_mul(1000) + widx;
    let mut rng = Rng::new(s);
    let mut skill = vec![vec![0.0; w.topics]; N_AGENTS];
    for t in 0..w.topics {
        for a in 0..N_AGENTS {
            skill[a][t] = if w.equal { 0.7 } else { 0.35 + 0.25 * rng.f64() };
        }
        if !w.equal {
            for _ in 0..2 {
                let a = rng.below(N_AGENTS);
                skill[a][t] = 0.85 + 0.07 * rng.f64();
            }
        }
    }
    // keyword map: advertised best agent, wrong with prob `decoy`
    let mut kw = HashMap::new();
    for t in 0..w.topics {
        let best = (0..N_AGENTS).fold(0, |b, a| if skill[a][t] > skill[b][t] { a } else { b });
        let adv = if w.equal || rng.f64() < w.decoy { rng.below(N_AGENTS) } else { best };
        for j in 0..3 {
            kw.insert(format!("t{t}w{j}"), adv);
        }
    }
    // Zipf-ish topic frequency
    let wts: Vec<f64> = (0..w.topics).map(|r| 1.0 / ((r + 1) as f64).powf(0.7)).collect();
    let tot: f64 = wts.iter().sum();
    let mut tasks = Vec::with_capacity(N_TASKS);
    for i in 0..N_TASKS {
        let mut x = rng.f64() * tot;
        let mut topic = w.topics - 1;
        for (t, wt) in wts.iter().enumerate() {
            if x < *wt {
                topic = t;
                break;
            }
            x -= wt;
        }
        let mut f: Vec<String> = (0..3).map(|j| format!("t{topic}w{j}")).collect();
        f.push(format!("dom{}", topic % 3));
        if w.recur {
            f.push(format!("n{}", rng.below(30)));
        } else {
            for j in 0..8 {
                f.push(format!("u{i}x{j}"));
            }
        }
        tasks.push((Task { id: i as u64, task_type: None, features: f }, topic));
    }
    Instance { skill, tasks, kw }
}

#[derive(Clone)]
struct Draft {
    agent: usize,
    topic: usize,
    q: f64,
    hal: bool,
    ok: bool,
    tid: u64,
}

struct WorldExec<'a> {
    w: &'a WorldCfg,
    inst: &'a Instance,
    seed: u64,
    topic_of: Vec<usize>,
}

impl WorldExec<'_> {
    fn lazy(&self, c: usize) -> f64 {
        match self.w.lazy {
            LazyMode::All => 0.15,
            LazyMode::Third if c % 3 == 2 => 0.15,
            _ => 1.0,
        }
    }
}

impl Executor for WorldExec<'_> {
    type Out = Draft;
    fn draft(&mut self, task: &Task, agent: usize, ctx: &Context) -> Draft {
        let topic = self.topic_of[task.id as usize];
        let (s, tid) = (self.inst.skill[agent][topic], task.id);
        let down = self.w.outage && agent == 7 && (600..1000).contains(&(tid as usize));
        let n = ctx.note.unwrap_or(0.0);
        let p_hal = (self.w.hal_base * (1.0 - s) * (1.0 - self.w.ctx_hal_cut * n)).clamp(0.0, 1.0);
        let hal = uni(self.seed, tid, agent as u64, 11) < p_hal;
        let q = if down {
            0.0
        } else if hal {
            0.05 + 0.2 * uni(self.seed, tid, agent as u64, 12)
        } else {
            (s + self.w.ctx_gain * n * (1.0 - s) + 0.07 * norm(self.seed, tid, agent as u64, 13)).clamp(0.0, 1.0)
        };
        let ok = !down && uni(self.seed, tid, agent as u64, 14) < 0.99 - 0.03 * (1.0 - s);
        Draft { agent, topic, q, hal, ok, tid }
    }
    fn tool_ok(&mut self, o: &Draft) -> bool {
        o.ok
    }
    fn check(&mut self, _t: &Task, checker: usize, o: &Draft) -> bool {
        let sc = self.inst.skill[checker][o.topic];
        let u = uni(self.seed, o.tid, (checker * 16 + o.agent) as u64, 15);
        if o.q < 0.5 {
            let catch = (0.1 + 0.88 * sc) * self.lazy(checker) * if o.hal || o.q == 0.0 { 1.0 } else { 0.5 };
            u < catch
        } else {
            u < (0.05 + 0.1 * (1.0 - sc)) * self.lazy(checker)
        }
    }
    fn grade(&mut self, t: &Task, o: &Draft) -> Option<f64> {
        if uni(self.seed, t.id, 0, 16) >= REVEAL {
            return None;
        }
        Some((o.q + 0.05 * norm(self.seed, t.id, o.agent as u64, 17)).clamp(0.0, 1.0))
    }
}

// =============================================================================================
// arms

#[derive(Clone, Debug)]
struct Arm {
    name: &'static str,
    cfg: ResearchConfig,
}

#[derive(Clone, Debug)]
struct Tuned {
    critic: CriticKind,
    store_min: f64,
    sim_min: f32,
    warm_k: f64,
    plain_k: f64,
}

fn base_cfg(t: &Tuned, guarded: bool, plain: bool) -> ResearchConfig {
    ResearchConfig {
        router: if guarded { RouterChoice::GuardedHebr(GuardParams::default()) } else { RouterChoice::WarmHebr },
        warm_k: if plain { t.plain_k } else { t.warm_k },
        delay: DELAY,
        store_min: t.store_min,
        sim_min: t.sim_min,
        ..ResearchConfig::default()
    }
}

fn arms(t: &Tuned) -> Vec<Arm> {
    let mk = |name: &'static str, guarded: bool, retr: bool, crit: bool, reuse: bool| Arm {
        name,
        cfg: ResearchConfig { retrieval: retr, critic: if crit { Some(t.critic.clone()) } else { None }, reuse, ..base_cfg(t, guarded, !reuse) },
    };
    vec![
        mk("plain", false, false, false, false),
        mk("plain-g", true, false, false, false),
        mk("full", false, true, true, true),
        mk("full-g", true, true, true, true),
        mk("-retr", false, false, true, true),
        mk("-crit", false, true, false, true),
        mk("-reuse", false, true, true, false),
        mk("retr-only", false, true, false, false),
        mk("crit-only", false, false, true, false),
        mk("reuse-only", false, false, false, true),
    ]
}

/// metrics: [gross H250, gross H500, gross H2000, net H2000, gross last500, extra calls/task, ctx-hit rate]
const NM: usize = 7;
fn run(w: &WorldCfg, widx: u64, arm: &Arm, seed: u64) -> [f64; NM] {
    let inst = build(w, seed, widx);
    let topic_of: Vec<usize> = inst.tasks.iter().map(|t| t.1).collect();
    let mut ex = WorldExec { w, inst: &inst, seed, topic_of };
    let mut s = ResearchSession::new(N_AGENTS, inst.kw.clone(), arm.cfg.clone());
    let mut rng = Rng::new(seed * 7 + 1);
    let cands: Vec<usize> = (0..N_AGENTS).collect();
    let (mut sum250, mut sum500, mut sum, mut net, mut last, mut extra) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (i, (t, _)) in inst.tasks.iter().enumerate() {
        let rep = s.run_task(&mut ex, t, &cands, &mut rng);
        let q = rep.delivered.q;
        let e = (rep.calls - 1) as f64;
        if i < 250 {
            sum250 += q;
        }
        if i < 500 {
            sum500 += q;
        }
        sum += q;
        net += q - CALL_COST * e;
        if i >= N_TASKS - 500 {
            last += q;
        }
        extra += e;
    }
    let n = N_TASKS as f64;
    [sum250 / 250.0, sum500 / 500.0, sum / n, net / n, last / 500.0, extra / n, s.stats.ctx_hits as f64 / n]
}

fn eval(w: &WorldCfg, widx: u64, arm: &Arm, seeds: &[u64]) -> Vec<[f64; NM]> {
    seeds.par_iter().map(|&s| run(w, widx, arm, s)).collect()
}
fn col(v: &[[f64; NM]], k: usize) -> Vec<f64> {
    v.iter().map(|r| r[k]).collect()
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/e3_research.json".into());
    let held: Vec<u64> = match args.iter().position(|a| a == "--held").and_then(|i| args.get(i + 1)) {
        Some(r) => {
            let (a, b) = r.split_once('-').expect("--held a-b");
            (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
        }
        None => (31..=60).collect(),
    };
    let tune: Vec<u64> = (1..=8).collect();
    let t0 = Instant::now();
    let ws = worlds();
    println!("e3_research: {N_TASKS} tasks, {N_AGENTS} agents, call cost {CALL_COST}, tune seeds 1-8, eval {}-{}", held[0], held[held.len() - 1]);

    // ---- tuning (seeds 1-8): objective = mean NET H2000 of `full` over the 5 treatment worlds ----
    let critics = vec![
        ("flat(kappa0,d.995,pool2)", CriticKind::Flat(SecondOpinionConfig { cost: CALL_COST, kappa: 0.0, discount: 0.995, pool: 2.0, ..SecondOpinionConfig::default() })),
        ("flat(default kappa1)", CriticKind::Flat(SecondOpinionConfig { cost: CALL_COST, ..SecondOpinionConfig::default() })),
        ("hier(default)", CriticKind::Hier(HierSoConfig { base: SecondOpinionConfig { cost: CALL_COST, kappa: 0.0, discount: 0.995, ..SecondOpinionConfig::default() }, ..HierSoConfig::default() })),
        ("hier(probe50)", CriticKind::Hier(HierSoConfig { base: SecondOpinionConfig { cost: CALL_COST, kappa: 0.0, discount: 0.995, ..SecondOpinionConfig::default() }, probe_every: 50, ..HierSoConfig::default() })),
        ("hier(pool_leader8)", CriticKind::Hier(HierSoConfig { base: SecondOpinionConfig { cost: CALL_COST, kappa: 0.0, discount: 0.995, ..SecondOpinionConfig::default() }, pool_leader: 8.0, ..HierSoConfig::default() })),
        ("hier(pool_leader16,kc.5)", CriticKind::Hier(HierSoConfig { base: SecondOpinionConfig { cost: CALL_COST, kappa: 0.0, discount: 0.995, ..SecondOpinionConfig::default() }, pool_leader: 16.0, kappa_checker: 0.5, ..HierSoConfig::default() })),
    ];
    let mut best: Option<(f64, Tuned, &str)> = None;
    for (cn, ck) in &critics {
        for &sm in &[0.5, 0.65, 0.8] {
            for &sim in &[0.55f32, 0.7] {
                for &wk in &[1.0, 3.0] {
                    let t = Tuned { critic: ck.clone(), store_min: sm, sim_min: sim, warm_k: wk, plain_k: 1.0 };
                    let full = arms(&t).into_iter().find(|a| a.name == "full").unwrap();
                    let o: f64 = ws.iter().take(5).enumerate().map(|(i, w)| mean(&col(&eval(w, i as u64, &full, &tune), 3))).sum::<f64>() / 5.0;
                    if best.as_ref().is_none_or(|b| o > b.0) {
                        best = Some((o, t, cn));
                    }
                }
            }
        }
    }
    let (bo, mut tuned, bn) = best.unwrap();
    let mut bp = (f64::NEG_INFINITY, 1.0);
    for &pk in &[1.0, 3.0, 10.0] {
        tuned.plain_k = pk;
        let plain = arms(&tuned).into_iter().find(|a| a.name == "plain").unwrap();
        let o: f64 = ws.iter().take(5).enumerate().map(|(i, w)| mean(&col(&eval(w, i as u64, &plain, &tune), 2))).sum::<f64>() / 5.0;
        if o > bp.0 { bp = (o, pk); }
    }
    tuned.plain_k = bp.1;
    println!("[tune] plain warm_k={} obj {:.4}", bp.1, bp.0);
    println!("[tune] best full obj (net H2000, 5 treatment worlds, seeds 1-8) {bo:.4}: critic={bn} store_min={} sim_min={} warm_k={}", tuned.store_min, tuned.sim_min, tuned.warm_k);

    // ---- evaluation ----
    let arm_list = arms(&tuned);
    let mut res: HashMap<(usize, &str), Vec<[f64; NM]>> = HashMap::new();
    for (wi, w) in ws.iter().enumerate() {
        for a in &arm_list {
            res.insert((wi, a.name), eval(w, wi as u64, a, &held));
        }
    }

    // comparisons: (label, a, b)
    let comps: Vec<(&str, &str, &str)> = vec![
        ("full vs plain", "full", "plain"),
        ("full-g vs plain-g", "full-g", "plain-g"),
        ("full-g vs plain", "full-g", "plain"),
        ("full vs -retr", "full", "-retr"),
        ("full vs -crit", "full", "-crit"),
        ("full vs -reuse", "full", "-reuse"),
        ("full vs reuse-only", "full", "reuse-only"),
        ("retr-only vs plain", "retr-only", "plain"),
        ("crit-only vs plain", "crit-only", "plain"),
        ("reuse-only vs plain", "reuse-only", "plain"),
        ("-crit vs reuse-only (retr on top of reuse)", "-crit", "reuse-only"),
        ("-retr vs reuse-only (critic on top of reuse)", "-retr", "reuse-only"),
    ];
    let metrics = [(0usize, "gross_H250"), (2usize, "gross_H2000"), (3, "net_H2000"), (1, "gross_H500")];
    let mut tests: Vec<(usize, &str, &str, f64)> = vec![]; // world, comp label, metric, p
    let mut rows: Vec<(usize, usize, usize, Value)> = vec![];
    for (wi, _) in ws.iter().enumerate() {
        for (ci, (_, a, b)) in comps.iter().enumerate() {
            for (mi, (mk, _)) in metrics.iter().enumerate() {
                let (va, vb) = (col(&res[&(wi, *a)], *mk), col(&res[&(wi, *b)], *mk));
                let d = paired_diff(&va, &vb);
                let p = paired_t_pvalue(&va, &vb);
                tests.push((wi, comps[ci].0, metrics[mi].1, p));
                rows.push((wi, ci, mi, json!({"mean_diff": d.diff.mean, "lo": d.diff.lo, "hi": d.diff.hi, "wins": d.wins, "losses": d.losses, "ties": d.ties, "p": p})));
            }
        }
    }
    let adj = holm_adjust(&tests.iter().map(|t| t.3).collect::<Vec<_>>());
    let mut comp_json = vec![];
    let mut nwin = 0;
    let mut nloss = 0;
    for (k, (wi, ci, mi, mut j)) in rows.into_iter().enumerate() {
        let hp = adj[k];
        let m = j["mean_diff"].as_f64().unwrap();
        let verdict = if hp < 0.05 && m > 0.0 { nwin += 1; "WIN" } else if hp < 0.05 && m < 0.0 { nloss += 1; "LOSS" } else { "n.s." };
        j["holm_p"] = json!(hp);
        j["verdict"] = json!(verdict);
        j["world"] = json!(ws[wi].name);
        j["comparison"] = json!(comps[ci].0);
        j["metric"] = json!(metrics[mi].1);
        comp_json.push(j);
    }

    // ---- print ----
    println!("\nTABLE eval seeds (mean over seeds): gross H2000 | net H2000 | last500 | extra calls/task | ctx-hit");
    print!("{:<15}", "world");
    for a in &arm_list {
        print!("{:>11}", a.name);
    }
    println!();
    let mut table = vec![];
    for (wi, w) in ws.iter().enumerate() {
        print!("{:<15}", w.name);
        let mut row = serde_json::Map::new();
        for a in &arm_list {
            let r = &res[&(wi, a.name)];
            print!("{:>11.4}", mean(&col(r, 2)));
            row.insert(a.name.into(), json!({"gross_H250": mean(&col(r,0)), "gross_H500": mean(&col(r,1)), "gross_H2000": mean(&col(r,2)), "net_H2000": mean(&col(r,3)), "last500": mean(&col(r,4)), "extra_calls_per_task": mean(&col(r,5)), "ctx_hit_rate": mean(&col(r,6))}));
        }
        println!();
        table.push(json!({"world": w.name, "arms": Value::Object(row)}));
    }
    println!("\nNET H2000 (gross minus {CALL_COST} per extra call):");
    for (wi, w) in ws.iter().enumerate() {
        print!("{:<15}", w.name);
        for a in &arm_list {
            print!("{:>11.4}", mean(&col(&res[&(wi, a.name)], 3)));
        }
        println!();
    }
    println!("\nEXTRA CALLS / TASK (critic call + retry):");
    for (wi, w) in ws.iter().enumerate() {
        print!("{:<15}", w.name);
        for a in &arm_list {
            print!("{:>11.4}", mean(&col(&res[&(wi, a.name)], 5)));
        }
        println!();
    }
    println!("\nHOLM-SIGNIFICANT (family = {} tests, mean diff [95% CI], Holm p):", tests.len());
    for j in &comp_json {
        if j["verdict"] != "n.s." && j["metric"] != "gross_H500" && j["metric"] != "gross_H250" {
            println!("  {:<15} {:<22} {:<12} {:+.4} [{:+.4},{:+.4}] {} {} ({}/{} W/L)", j["world"].as_str().unwrap(), j["comparison"].as_str().unwrap(), j["metric"].as_str().unwrap(), j["mean_diff"].as_f64().unwrap(), j["lo"].as_f64().unwrap(), j["hi"].as_f64().unwrap(), j["verdict"].as_str().unwrap(), format!("p={:.4}", j["holm_p"].as_f64().unwrap()), j["wins"], j["losses"]);
        }
    }
    println!("Holm WIN {nwin}  LOSS {nloss}  runtime {:.1}s", t0.elapsed().as_secs_f64());

    let out = json!({
        "experiment": "e3_research",
        "tasks": N_TASKS, "agents": N_AGENTS, "call_cost": CALL_COST, "reveal_frac": REVEAL, "delay": DELAY,
        "tune_seeds": "1-8", "eval_seeds": format!("{}-{}", held[0], held[held.len()-1]),
        "tuned": {"critic": bn, "store_min": tuned.store_min, "sim_min": tuned.sim_min, "warm_k": tuned.warm_k, "plain_warm_k": tuned.plain_k, "objective": bo},
        "table": table, "comparisons": comp_json, "holm_win": nwin, "holm_loss": nloss, "runtime_s": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote {out_path}");
}
