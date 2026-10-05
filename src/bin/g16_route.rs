//! G16-route (W17, retrieval track R): does confidence-gated / adaptive-k / fresh retrieval fix the "retrieval hurts when context is
//! irrelevant or weakly identified" weakness of the g1/g5/f5/f8 knowledge layer, on the g1 hard-corpus fixtures?
//!
//! World/policy engine: src/bin/g16_common/mod.rs (worlds fixed there BEFORE policies were run). Tuning: seeds 1-8 on a panel of 6 worlds
//! (4 treatment-ish, 1 drift, 1 control); held-out 31-60 on all 8 worlds (sib64 and ctl-oneoff are NEVER tuned on); `--corpus B --confirm`
//! re-runs with FROZEN picks on the fresh corpus B (new texts, seeds 61-90).
//!
//! Usage: cargo run --release --offline --bin g16_route -- --corpus A [--held 31-60]
//!        cargo run --release --offline --bin g16_route -- --corpus B --confirm [--held 61-90]
//!        cargo run --release --offline --bin g16_route -- --corpus A --check     (replicates the g1_route mini-hybrid arm)

use std::time::Instant;

use neural_mesh::stats::summarize;
use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
#[path = "g16_common/mod.rs"]
mod g16;
use common::*;
use g16::*;

const THR_GRID: [f32; 9] = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92];

fn gate_json(g: &Gate) -> Value {
    match g {
        Gate::Off => json!({"t": "off"}),
        Gate::G1 => json!({"t": "g1"}),
        Gate::Post { delta, gamma, floor } => json!({"t": "post", "delta": delta, "gamma": gamma, "floor": floor}),
        Gate::Learned { lr, delta, eps } => json!({"t": "learned", "lr": lr, "delta": delta, "eps": eps}),
    }
}
fn gate_from(v: &Value) -> Gate {
    match v["t"].as_str().unwrap() {
        "off" => Gate::Off,
        "g1" => Gate::G1,
        "post" => Gate::Post { delta: v["delta"].as_f64().unwrap(), gamma: v["gamma"].as_f64().unwrap(), floor: v["floor"].as_f64().unwrap() },
        _ => Gate::Learned { lr: v["lr"].as_f64().unwrap(), delta: v["delta"].as_f64().unwrap(), eps: v["eps"].as_f64().unwrap() },
    }
}
fn cfg_json(c: &PolCfg) -> Value {
    json!({"name": c.name, "front": format!("{:?}", c.front), "inj_conf": c.inj_conf.map(|(a, b)| json!([a, b])), "sim": c.sim, "k": c.k, "thr": c.thr, "min_n": c.min_n, "gate": gate_json(&c.gate), "and_g1": c.and_g1,
           "id": c.id.map(|(a, m)| json!([a, m])), "inj_always": c.inj_always, "kappa": c.kappa, "g1_min": c.g1_min, "g1_margin": c.g1_margin, "probe0": c.probe0, "backoff": c.backoff, "adapt": c.adapt, "decay": c.decay, "reset_fail": c.reset_fail})
}
fn cfg_from(v: &Value) -> PolCfg {
    PolCfg {
        name: v["name"].as_str().unwrap().into(),
        base: Base::Vote,
        front: match v["front"].as_str().unwrap() { "Warm" => Base::Warm, "HebR" => Base::HebR, _ => Base::Guard },
        inj_conf: v["inj_conf"].as_array().map(|a| (a[0].as_f64().unwrap(), a[1].as_f64().unwrap())),
        sim: v["sim"].as_u64().unwrap() as usize,
        k: v["k"].as_u64().unwrap() as usize,
        thr: v["thr"].as_f64().unwrap() as f32,
        min_n: v["min_n"].as_u64().unwrap() as usize,
        gate: gate_from(&v["gate"]),
        and_g1: v["and_g1"].as_bool().unwrap(),
        id: v["id"].as_array().map(|a| (a[0].as_f64().unwrap() as f32, a[1].as_f64().unwrap() as f32)),
        inj_always: v["inj_always"].as_bool().unwrap(),
        kappa: v["kappa"].as_f64().unwrap_or(0.0),
        g1_min: v["g1_min"].as_f64().unwrap_or(6.0),
        g1_margin: v["g1_margin"].as_f64().unwrap_or(0.02),
        probe0: v["probe0"].as_u64().unwrap_or(25) as usize,
        backoff: v["backoff"].as_bool().unwrap_or(false),
        adapt: v["adapt"].as_f64().map(|x| x as f32),
        decay: v["decay"].as_f64().unwrap(),
        reset_fail: v["reset_fail"].as_u64().unwrap() as usize,
    }
}

/// Reference (best fair baseline, (H500+H2000)/2 on the tuning seeds) for the RISK worlds, filled in main before any tuning.
static REFS: std::sync::OnceLock<Vec<(&'static str, f64)>> = std::sync::OnceLock::new();
/// tuning objective (constrained): mean over the WIN worlds {sib16, rand16, sib16-drift} of (H500+H2000)/2 on the tuning seeds
/// MINUS 2 x mean over the RISK worlds {ctl-irrel, sib16-mixed50, sib16-weakid50} of max(0, best-fair reference - value - 0.003).
fn panel_obj(cfg: &PolCfg, ws: &[Spec], seeds: &[u64], c: &Corpus, p: &Pool, ss: &[Sim]) -> f64 {
    let refs = REFS.get().expect("refs");
    let (mut win, mut nw, mut pen, mut nr) = (0.0, 0.0, 0.0, 0.0);
    for sp in ws.iter().filter(|s| s.tune) {
        let mut v = 0.0;
        for &s in seeds {
            let r = run(cfg, sp, s, c, p, ss);
            v += (r.h[2] + r.h[3]) / 2.0;
        }
        v /= seeds.len() as f64;
        match refs.iter().find(|r| r.0 == sp.name) {
            Some(r) => {
                pen += (r.1 - v - 0.003).max(0.0);
                nr += 1.0;
            }
            None => {
                win += v;
                nw += 1.0;
            }
        }
    }
    win / nw - 2.0 * pen / nr
}
fn tune_best(cands: Vec<PolCfg>, ws: &[Spec], seeds: &[u64], c: &Corpus, p: &Pool, ss: &[Sim]) -> (PolCfg, f64) {
    let scored: Vec<(f64, usize)> = cands.par_iter().enumerate().map(|(i, g)| (panel_obj(g, ws, seeds, c, p, ss), i)).collect();
    let (m, i) = scored.into_iter().fold((f64::MIN, 0), |b, x| if x.0 > b.0 { x } else { b });
    (cands[i].clone(), m)
}
fn thr_nbrs(thr: f32) -> Vec<f32> {
    let i = THR_GRID.iter().position(|&t| (t - thr).abs() < 1e-6).unwrap_or(6);
    let mut v = vec![THR_GRID[i]];
    if i > 0 {
        v.push(THR_GRID[i - 1]);
    }
    if i + 1 < THR_GRID.len() {
        v.push(THR_GRID[i + 1]);
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tag = arg(&args, "--corpus", "A");
    let frozen = args.iter().any(|a| a == "--frozen"); // load picks (no tuning) but keep the default held range / output name
    let confirm = args.iter().any(|a| a == "--confirm");
    let check = args.iter().any(|a| a == "--check");
    let held = parse_range(&args, "--held", if confirm { (61, 90) } else { (31, 60) });
    let tune_seeds = parse_range(&args, "--tune", (1, 8));
    let t0 = Instant::now();
    let corpus = load_corpus(&tag);
    let pool = build_pool(&corpus);
    let ss = sims(&pool);
    let ws = worlds();
    println!("corpus {tag}: pool {} queries, {} sims, {} worlds; matrices {:.1}s", pool.nq, ss.len(), ws.len(), t0.elapsed().as_secs_f64());

    if check {
        // replicate g1_route mini-hybrid pick (k5, thr .75, min_n 2, var a=.6) in sib16 and ctl-irrel, seeds 31-60: results/g1_route.json
        let r0 = PolCfg::vote("R0", SIM_MINI_H, 5, 0.75, 2, Gate::G1);
        let g1: Value = serde_json::from_str(&std::fs::read_to_string("results/g1_route.json").unwrap()).unwrap();
        for wn in ["sib16", "ctl-irrel", "sib16-mixed50", "rand16", "sib64"] {
            let sp = ws.iter().find(|s| s.name == wn).unwrap();
            let r = eval(&r0, sp, &held, &corpus, &pool, &ss);
            let row = g1["table"].as_array().unwrap().iter().find(|t| t["world"] == wn).unwrap()["rows"]
                .as_array()
                .unwrap()
                .iter()
                .find(|x| x["policy"].as_str().unwrap().starts_with("mini-hybrid"))
                .unwrap()
                .clone();
            let m: Vec<f64> = (0..4).map(|i| mean_of(&r, i)).collect();
            println!("{wn:<14} g16 H100/250/500/2000 = {:.4} {:.4} {:.4} {:.4} | g1_route.json = {} -> max|d| = {:.2e}", m[0], m[1], m[2], m[3], row["h"], (0..4).map(|i| (m[i] - row["h"][i].as_f64().unwrap()).abs()).fold(0.0, f64::max));
        }
        return;
    }

    if args.iter().any(|a| a == "--diag") {
        // diagnostics: who loses in the control worlds? mean pe of vote-played vs front-played tasks
        let v: Value = serde_json::from_str(&std::fs::read_to_string("results/g16_picks.json").unwrap()).unwrap();
        let arms: Vec<PolCfg> = v["arms"].as_array().unwrap().iter().map(cfg_from).collect();
        let g = PolCfg::base_kind("guarded(hebr)", Base::Guard);
        for wn in ["ctl-irrel", "ctl-oneoff", "sib16"] {
            let sp = ws.iter().find(|s| s.name == wn).unwrap();
            let rg = eval(&g, sp, &held, &corpus, &pool, &ss);
            println!("-- {wn}: guarded(hebr) H2000 {:.4}", mean_of(&rg, 3));
            for a in arms.iter().filter(|a| ["R0 (g1 retrieval)", "G-post", "G-post+g1", "G-post(inj-always)"].contains(&a.name.as_str())) {
                let r = eval(a, sp, &held, &corpus, &pool, &ss);
                let n = r.len() as f64;
                println!("   {:<22} H2000 {:.4} vote% {:.1} pe(vote plays) {:.4} pe(front plays) {:.4}", a.name, mean_of(&r, 3), r.iter().map(|x| x.vote_frac).sum::<f64>() / n * 100.0, r.iter().map(|x| x.pe_vote).filter(|x| x.is_finite()).sum::<f64>() / n, r.iter().map(|x| x.pe_front).sum::<f64>() / n);
            }
        }
        return;
    }
    // ---------------------------------------------------------------- tuning (A) / frozen picks (B)
    let picks_path = "results/g16_picks.json";
    let sim_ids = [SIM_MINI, SIM_MINI_S, SIM_MINI_H, SIM_BGE, SIM_BGE_H];
    // named arms (all Vote configs); order matters for output
    {
        let bl = [Base::Keyword, Base::Eps, Base::Pts, Base::HebR, Base::Warm, Base::Guard];
        let mut refs = vec![];
        for rw in ["ctl-irrel", "sib16-mixed50", "sib16-weakid50"] {
            let sp = ws.iter().find(|s| s.name == rw).unwrap();
            let best = bl.iter().map(|b| { let r = eval(&PolCfg::base_kind("b", b.clone()), sp, &tune_seeds, &corpus, &pool, &ss); (mean_of(&r, 2) + mean_of(&r, 3)) / 2.0 }).fold(f64::MIN, f64::max);
            println!("risk-world reference (best fair, tuning seeds) {rw}: {best:.4}");
            refs.push((rw, best));
        }
        let _ = REFS.set(refs);
    }
    let mut arms: Vec<PolCfg> = vec![];
    let mut tune_log: Vec<Value> = vec![];
    if confirm || frozen {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(picks_path).expect("run corpus A first")).unwrap();
        for a in v["arms"].as_array().unwrap() {
            arms.push(cfg_from(a));
        }
        println!("confirm run: {} frozen arms loaded from {picks_path}", arms.len());
    } else {
        let obj_note = "tuning objective: see panel_obj (win worlds mean minus 2x control-loss penalty vs best fair baseline), seeds 1-8";
        println!("{obj_note}");
        // S1: per-sim R0 (g1 gate, inject always) and the ungated version (mini-hybrid only)
        let mut r0s: Vec<PolCfg> = vec![];
        for &si in &sim_ids {
            let mut g = vec![];
            for &k in &[5usize, 8] {
                for &thr in &THR_GRID {
                    for &mn in &[2usize, 3] {
                        g.push(PolCfg::vote(&format!("R0[{}]", ss[si].label), si, k, thr, mn, Gate::G1));
                    }
                }
            }
            let (b, m) = tune_best(g, &ws, &tune_seeds, &corpus, &pool, &ss);
            println!("S1 R0[{}] tuned: k{} thr{} min_n{}  obj {:.4}", ss[si].label, b.k, b.thr, b.min_n, m);
            tune_log.push(json!({"stage": "S1-R0", "sim": ss[si].label, "obj": m, "cfg": cfg_json(&b)}));
            r0s.push(b);
        }
        let r0 = r0s[2].clone(); // mini-hybrid = the g1 best system, primary sim for the gate stages
        // front router for the gated stack: Guard (g1 default) vs Warm (the strongest fair baseline in the control worlds); chosen by the same panel objective
        let mut gw = vec![];
        for &k in &[5usize, 8] {
            for &thr in &THR_GRID {
                for &mn in &[2usize, 3] {
                    let mut c = PolCfg::vote("R0-warmfront", SIM_MINI_H, k, thr, mn, Gate::G1);
                    c.front = Base::Warm;
                    gw.push(c);
                }
            }
        }
        let (r0w, mw) = tune_best(gw, &ws, &tune_seeds, &corpus, &pool, &ss);
        let m_r0 = panel_obj(&r0, &ws, &tune_seeds, &corpus, &pool, &ss);
        println!("S1 R0 front=Guard obj {:.4} | R0 front=Warm obj {:.4} (k{} thr{} min_n{})", m_r0, mw, r0w.k, r0w.thr, r0w.min_n);
        tune_log.push(json!({"stage": "S1-R0warm", "obj": mw, "cfg": cfg_json(&r0w)}));
        let front_star = if mw > m_r0 { Base::Warm } else { Base::Guard };
        println!("S1 -> front for the gated stack: {:?}", front_star);
        let mut goff = vec![];
        for &k in &[5usize, 8] {
            for &thr in &THR_GRID {
                for &mn in &[2usize, 3] {
                    goff.push(PolCfg::vote("Roff", SIM_MINI_H, k, thr, mn, Gate::Off));
                }
            }
        }
        let (roff, m) = tune_best(goff, &ws, &tune_seeds, &corpus, &pool, &ss);
        println!("S1 Roff[mini-hybrid] (no gate at all) tuned: k{} thr{} min_n{} obj {:.4}", roff.k, roff.thr, roff.min_n, m);
        tune_log.push(json!({"stage": "S1-Roff", "obj": m, "cfg": cfg_json(&roff)}));

        // S2: confidence gates (separately verified). base (k,min_n) from R0; thr in {R0 thr, +-1 grid step}
        let mk_post = |name: &str, inj: bool, id: bool, and_g1: bool| -> Vec<PolCfg> {
            let mut v = vec![];
            for thr in thr_nbrs(r0.thr) {
                for &delta in &[0.0, 0.05, 0.10] {
                    for &gamma in &[0.6, 0.8, 0.9] {
                        for &floor in &[0.0, 0.6] {
                            let ids: Vec<Option<(f32, f32)>> = if id { vec![Some((0.5, 0.0)), Some((0.7, 0.0)), Some((0.6, 0.25)), Some((0.8, 0.5))] } else { vec![None] };
                            for idg in ids {
                                let mut c = PolCfg::vote(name, SIM_MINI_H, r0.k, thr, r0.min_n, Gate::Post { delta, gamma, floor });
                                c.front = front_star.clone();
                                c.inj_always = inj;
                                c.id = idg;
                                c.and_g1 = and_g1;
                                v.push(c);
                            }
                        }
                    }
                }
            }
            v
        };
        let mut gates: Vec<(PolCfg, f64)> = vec![];
        for (name, inj, id, ag1) in [("G-post(inj-always)", true, false, false), ("G-post", false, false, false), ("G-post+id", false, true, false), ("G-post+g1", false, false, true)] {
            let (b, m) = tune_best(mk_post(name, inj, id, ag1), &ws, &tune_seeds, &corpus, &pool, &ss);
            println!("S2 {name} tuned: {:?} k{} thr{} id{:?} obj {:.4}", b.gate, b.k, b.thr, b.id, m);
            tune_log.push(json!({"stage": "S2", "obj": m, "cfg": cfg_json(&b)}));
            gates.push((b, m));
        }
        {
            let mut v = vec![];
            for thr in thr_nbrs(r0.thr) {
                for &delta in &[0.0, 0.05, 0.10] {
                    for &gamma in &[0.8, 0.9, 0.95] {
                        for &di in &[-0.05, 0.0] {
                            for &gi in &[0.5, 0.7] {
                                let mut c = PolCfg::vote("G-post+leninj", SIM_MINI_H, r0.k, thr, r0.min_n, Gate::Post { delta, gamma, floor: 0.0 });
                                c.front = front_star.clone();
                                c.inj_always = false;
                                c.inj_conf = Some((di, gi));
                                v.push(c);
                            }
                        }
                    }
                }
            }
            let (b, m) = tune_best(v, &ws, &tune_seeds, &corpus, &pool, &ss);
            println!("S2 G-post+leninj tuned: {:?} inj {:?} k{} thr{} obj {:.4}", b.gate, b.inj_conf, b.k, b.thr, m);
            tune_log.push(json!({"stage": "S2", "obj": m, "cfg": cfg_json(&b)}));
            gates.push((b, m));
        }
        for (gname, inj, ag1) in [("G-shrink", false, false), ("G-shrink(inj-always)", true, false), ("G-shrink+g1", true, true), ("G-shrink+g1(inj-gated)", false, true)] {
            let mut v = vec![];
            for thr in thr_nbrs(r0.thr) {
                for &kappa in &[3.0, 6.0] {
                    for &delta in &[0.0, 0.05, 0.10] {
                        for &gamma in &[0.5, 0.7, 0.85] {
                            for &floor in &[0.0, 0.6] {
                                for &mn in &[1usize, 2] {
                                    let mut c = PolCfg::vote(gname, SIM_MINI_H, r0.k, thr, mn, Gate::Post { delta, gamma, floor });
                                    c.front = front_star.clone();
                                    c.inj_always = inj;
                                    c.kappa = kappa;
                                    c.and_g1 = ag1;
                                    v.push(c);
                                }
                            }
                        }
                    }
                }
            }
            let (b, m) = tune_best(v, &ws, &tune_seeds, &corpus, &pool, &ss);
            println!("S2 {gname} tuned: {:?} kappa {} min_n {} k{} thr{} obj {:.4}", b.gate, b.kappa, b.min_n, b.k, b.thr, m);
            tune_log.push(json!({"stage": "S2", "obj": m, "cfg": cfg_json(&b)}));
            gates.push((b, m));
        }
        {
            let mut v = vec![];
            for thr in thr_nbrs(r0.thr) {
                for &kappa in &[3.0, 6.0] {
                    for &delta in &[0.0, 0.05] {
                        for &gamma in &[0.7, 0.85] {
                            for &di in &[-0.05, 0.0] {
                                for &gi in &[0.5, 0.7] {
                                    let mut c = PolCfg::vote("G-shrink+g1+leninj", SIM_MINI_H, r0.k, thr, 1, Gate::Post { delta, gamma, floor: 0.0 });
                                    c.front = front_star.clone();
                                    c.inj_always = false;
                                    c.inj_conf = Some((di, gi));
                                    c.kappa = kappa;
                                    c.and_g1 = true;
                                    v.push(c);
                                }
                            }
                        }
                    }
                }
            }
            let (b, m) = tune_best(v, &ws, &tune_seeds, &corpus, &pool, &ss);
            println!("S2 G-shrink+g1+leninj tuned: {:?} kappa {} inj {:?} thr{} obj {:.4}", b.gate, b.kappa, b.inj_conf, b.thr, m);
            tune_log.push(json!({"stage": "S2", "obj": m, "cfg": cfg_json(&b)}));
            gates.push((b, m));
        }
        {
            // G-shrink+g1 with a tuned window gate (min samples, margin, probe interval, exponential probe backoff); shrink params frozen from G-shrink+g1
            let base = gates.iter().find(|g| g.0.name == "G-shrink+g1").unwrap().0.clone();
            let mut v = vec![];
            for &gmin in &[4.0, 6.0] {
                for &gm in &[0.0, 0.02] {
                    for &p0 in &[25usize, 50] {
                        for &bo in &[false, true] {
                            let mut c = base.clone();
                            c.name = "G-shrink+g1b".into();
                            c.g1_min = gmin;
                            c.g1_margin = gm;
                            c.probe0 = p0;
                            c.backoff = bo;
                            v.push(c);
                        }
                    }
                }
            }
            let (b, m) = tune_best(v, &ws, &tune_seeds, &corpus, &pool, &ss);
            println!("S2 G-shrink+g1b tuned: g1_min {} margin {} probe0 {} backoff {} obj {:.4}", b.g1_min, b.g1_margin, b.probe0, b.backoff, m);
            tune_log.push(json!({"stage": "S2", "obj": m, "cfg": cfg_json(&b)}));
            gates.push((b, m));
        }
        let mut lg = vec![];
        for thr in thr_nbrs(r0.thr) {
            for &lr in &[0.05, 0.15] {
                for &delta in &[0.0, 0.05] {
                    for &eps in &[0.05, 0.10] {
                        let mut c = PolCfg::vote("G-learned", SIM_MINI_H, r0.k, thr, r0.min_n, Gate::Learned { lr, delta, eps });
                        c.front = front_star.clone();
                        c.inj_always = false;
                        lg.push(c);
                    }
                }
            }
        }
        let (b, m) = tune_best(lg, &ws, &tune_seeds, &corpus, &pool, &ss);
        println!("S2 G-learned tuned: {:?} k{} thr{} obj {:.4}", b.gate, b.k, b.thr, m);
        tune_log.push(json!({"stage": "S2", "obj": m, "cfg": cfg_json(&b)}));
        gates.push((b, m));
        // gate base for later stages = best tune objective among the gate variants (declared rule)
        let gstar = gates.iter().fold(None, |b: Option<&(PolCfg, f64)>, x| match b { Some(y) if y.1 >= x.1 => b, _ => Some(x) }).unwrap().0.clone();
        println!("S2 -> gate base for later stages: {} ({:?})", gstar.name, gstar.gate);

        // S3: adaptive k on top of G* and on top of R0 (kmax 8)
        let ad_cands = |base: &PolCfg, nm: &str| -> Vec<PolCfg> {
            let mut v = vec![];
            for &d in &[0.03f32, 0.06, 0.10, 0.15] {
                let mut c = base.clone();
                c.name = nm.into();
                c.k = 8;
                c.adapt = Some(d);
                v.push(c);
            }
            let mut c = base.clone();
            c.name = nm.into();
            c.k = 8;
            v.push(c); // fixed k=8 control (adaptive off)
            v
        };
        let (r0_ad, m1) = tune_best(ad_cands(&r0, "R0+adaptk"), &ws, &tune_seeds, &corpus, &pool, &ss);
        let (g_ad, m2) = tune_best(ad_cands(&gstar, "G*+adaptk"), &ws, &tune_seeds, &corpus, &pool, &ss);
        println!("S3 R0+adaptk -> k{} adapt{:?} obj {:.4} | G*+adaptk -> k{} adapt{:?} obj {:.4}", r0_ad.k, r0_ad.adapt, m1, g_ad.k, g_ad.adapt, m2);
        tune_log.push(json!({"stage": "S3", "obj": m1, "cfg": cfg_json(&r0_ad)}));
        tune_log.push(json!({"stage": "S3", "obj": m2, "cfg": cfg_json(&g_ad)}));

        // S4: freshness (decay + failure reset)
        let fr_cands = |base: &PolCfg, nm: &str| -> Vec<PolCfg> {
            let mut v = vec![];
            for &dc in &[1.0, 0.999, 0.997, 0.994] {
                for &rf in &[0usize, 3] {
                    let mut c = base.clone();
                    c.name = nm.into();
                    c.decay = dc;
                    c.reset_fail = rf;
                    v.push(c);
                }
            }
            v
        };
        let (r0_fr, m3) = tune_best(fr_cands(&r0, "R0+fresh"), &ws, &tune_seeds, &corpus, &pool, &ss);
        let (g_fr, m4) = tune_best(fr_cands(&g_ad, "G*+adaptk+fresh"), &ws, &tune_seeds, &corpus, &pool, &ss);
        println!("S4 R0+fresh -> decay{} reset{} obj {:.4} | FINAL(G*+adaptk+fresh) -> decay{} reset{} obj {:.4}", r0_fr.decay, r0_fr.reset_fail, m3, g_fr.decay, g_fr.reset_fail, m4);
        tune_log.push(json!({"stage": "S4", "obj": m3, "cfg": cfg_json(&r0_fr)}));
        tune_log.push(json!({"stage": "S4", "obj": m4, "cfg": cfg_json(&g_fr)}));
        let mut fin = g_fr.clone();
        fin.name = "FINAL[mini-hybrid]".into();

        // S5: the same FINAL gate stack on the other similarity systems (thr re-tuned around that system's R0 thr; everything else frozen)
        let mut finals: Vec<PolCfg> = vec![];
        for (j, &si) in sim_ids.iter().enumerate() {
            if si == SIM_MINI_H {
                finals.push(fin.clone());
                continue;
            }
            let mut v = vec![];
            for thr in thr_nbrs(r0s[j].thr) {
                let mut c = fin.clone();
                c.name = format!("FINAL[{}]", ss[si].label);
                c.sim = si;
                c.thr = thr;
                v.push(c);
            }
            let (b, m) = tune_best(v, &ws, &tune_seeds, &corpus, &pool, &ss);
            println!("S5 FINAL[{}] thr tuned: {} obj {:.4}", ss[si].label, b.thr, m);
            tune_log.push(json!({"stage": "S5", "obj": m, "cfg": cfg_json(&b)}));
            finals.push(b);
        }

        // arm order
        let mut name = |c: &PolCfg, n: &str| {
            let mut c = c.clone();
            c.name = n.into();
            c
        };
        arms.push(name(&r0, "R0 (g1 retrieval)"));
        arms.push(name(&roff, "R-nogate"));
        arms.push(name(&r0w, "R0-warmfront"));
        for (g, _) in &gates {
            arms.push(g.clone());
        }
        arms.push(name(&r0_ad, "R0+adaptk"));
        arms.push(name(&g_ad, "G*+adaptk"));
        arms.push(name(&r0_fr, "R0+fresh"));
        arms.push(name(&fin, "FINAL[mini-hybrid]"));
        for (j, &si) in sim_ids.iter().enumerate() {
            if si != SIM_MINI_H {
                arms.push(name(&r0s[j], &format!("R0[{}]", ss[si].label)));
                arms.push(finals[j].clone());
            }
        }
        // oracle similarity reference (ground-truth concept equality) with the g1 gate and with the final stack
        let mut orc = r0.clone();
        orc.name = "oracle-sim R0".into();
        orc.sim = SIM_ORACLE;
        orc.thr = 0.15;
        arms.push(orc);
        let mut orc2 = fin.clone();
        orc2.name = "oracle-sim FINAL".into();
        orc2.sim = SIM_ORACLE;
        orc2.thr = 0.15;
        arms.push(orc2);
        let pj: Vec<Value> = arms.iter().map(cfg_json).collect();
        std::fs::write(picks_path, serde_json::to_string_pretty(&json!({"arms": pj, "tune_log": tune_log})).unwrap()).unwrap();
        println!("tuning done {:.1}s", t0.elapsed().as_secs_f64());
    }

    // ---------------------------------------------------------------- held-out evaluation
    // forced (NOT tuned) freshness variants, added identically in every mode: verifies stage (3) beyond what the tuner picked
    {
        let fin = arms.iter().find(|a| a.name == "FINAL[mini-hybrid]").unwrap().clone();
        let r0c = arms.iter().find(|a| a.name == "R0 (g1 retrieval)").unwrap().clone();
        for (nm, dc, rf) in [("decay.997", 0.997, 0usize), ("decay.99", 0.99, 0), ("reset3", 1.0, 3), ("decay.997+reset3", 0.997, 3)] {
            for (pre, b) in [("FINAL", &fin), ("R0", &r0c)] {
                let mut c = b.clone();
                c.name = format!("{pre}+{nm}");
                c.decay = dc;
                c.reset_fail = rf;
                arms.push(c);
            }
        }
    }
    let bases: Vec<PolCfg> = vec![
        PolCfg::base_kind("keyword", Base::Keyword),
        PolCfg::base_kind("eps-greedy", Base::Eps),
        PolCfg::base_kind("pooled-ts", Base::Pts),
        PolCfg::base_kind("hebbian", Base::HebR),
        PolCfg::base_kind("warm", Base::Warm),
        PolCfg::base_kind("guarded(hebr)", Base::Guard),
    ];
    let nb = bases.len();
    let all: Vec<&PolCfg> = bases.iter().chain(arms.iter()).collect();
    let find = |n: &str| all.iter().position(|c| c.name == n).unwrap_or_else(|| panic!("arm {n}"));
    let mut res: Vec<Vec<Vec<RunOut>>> = vec![]; // [world][arm][seed]
    for sp in &ws {
        let r: Vec<Vec<RunOut>> = all.iter().map(|c| eval(c, sp, &held, &corpus, &pool, &ss)).collect();
        res.push(r);
    }
    // best fair baseline per (world, metric)
    let metrics_main = [2usize, 3, 4]; // H500, H2000, late
    let best_fair = |wi: usize, m: usize| -> usize { (0..nb).fold(0, |b, j| if mean_of(&res[wi][j], m) > mean_of(&res[wi][b], m) { j } else { b }) };
    let mut table = vec![];
    for (wi, sp) in ws.iter().enumerate() {
        println!("\n=== world {} (w={}, mixed={}, corrupt={}, drift_at={}, {}) | mean expected success; held seeds {}-{} ===", sp.name, sp.w, sp.mixed, sp.corrupt, sp.drift_at, if sp.tune { "in tuning panel" } else { "NOT in tuning panel" }, held[0], held[held.len() - 1]);
        println!("{:<26} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6}", "arm", "H100", "H250", "H500", "H2000", "late", "vote%", "cand%");
        let mut rows = vec![];
        for (ai, c) in all.iter().enumerate() {
            let r = &res[wi][ai];
            let m: Vec<f64> = (0..5).map(|i| mean_of(r, i)).collect();
            let vf = r.iter().map(|x| x.vote_frac).sum::<f64>() / r.len() as f64;
            let cf = r.iter().map(|x| x.cand_frac).sum::<f64>() / r.len() as f64;
            println!("{:<26} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>6.1} {:>6.1}", c.name, m[0], m[1], m[2], m[3], m[4], vf * 100.0, cf * 100.0);
            rows.push(json!({"arm": c.name, "h": [m[0], m[1], m[2], m[3]], "late": m[4], "vote_frac": vf, "cand_frac": cf}));
        }
        table.push(json!({"world": sp.name, "rows": rows}));
    }

    // ---- comparisons. Family 1: every gated arm vs best fair baseline and vs R0 and vs R-nogate
    let r0i = find("R0 (g1 retrieval)");
    let offi = find("R-nogate");
    let focus: Vec<usize> = (nb..all.len()).collect();
    struct Row {
        fam: &'static str,
        world: String,
        arm: String,
        vs: String,
        metric: &'static str,
        c: Cmp,
        treat: bool,
        control: bool,
        adj: f64,
    }
    let mut rows: Vec<Row> = vec![];
    for (wi, sp) in ws.iter().enumerate() {
        for &ai in &focus {
            for &m in &metrics_main {
                let a = col(&res[wi][ai], m);
                let bf = best_fair(wi, m);
                let eq = if !sp.treat { 0.01 } else { 0.0 };
                rows.push(Row { fam: "F1:vs-best-fair", world: sp.name.into(), arm: all[ai].name.clone(), vs: format!("{} (best fair)", all[bf].name), metric: METRIC_NAMES[m], c: cmp(&a, &col(&res[wi][bf], m), eq), treat: sp.treat, control: !sp.treat, adj: 1.0 });
                {
                    let gi = nb - 1; // guarded(hebr): same front, no retrieval
                    rows.push(Row { fam: "F6:vs-same-front(guarded)", world: sp.name.into(), arm: all[ai].name.clone(), vs: "guarded(hebr)".into(), metric: METRIC_NAMES[m], c: cmp(&a, &col(&res[wi][gi], m), if !sp.treat { 0.01 } else { 0.0 }), treat: sp.treat, control: !sp.treat, adj: 1.0 });
                }
                if ai != r0i {
                    rows.push(Row { fam: "F2:vs-R0", world: sp.name.into(), arm: all[ai].name.clone(), vs: "R0 (g1 retrieval)".into(), metric: METRIC_NAMES[m], c: cmp(&a, &col(&res[wi][r0i], m), 0.0), treat: sp.treat, control: !sp.treat, adj: 1.0 });
                }
                if ai != offi {
                    rows.push(Row { fam: "F3:vs-R-nogate", world: sp.name.into(), arm: all[ai].name.clone(), vs: "R-nogate".into(), metric: METRIC_NAMES[m], c: cmp(&a, &col(&res[wi][offi], m), 0.0), treat: sp.treat, control: !sp.treat, adj: 1.0 });
                }
            }
        }
    }
    // stage increments (each improvement verified separately): vs its own base
    let stage_pairs: Vec<(&str, &str)> = vec![
        ("G-post(inj-always)", "R0 (g1 retrieval)"),
        ("G-post", "G-post(inj-always)"),
        ("G-post+id", "G-post"),
        ("G-post+g1", "G-post"),
        ("G-post+leninj", "G-post"),
        ("G-post+leninj", "R0 (g1 retrieval)"),
        ("R0-warmfront", "R0 (g1 retrieval)"),
        ("G-shrink", "R0 (g1 retrieval)"),
        ("G-shrink(inj-always)", "R0 (g1 retrieval)"),
        ("G-shrink", "G-post"),
        ("G-shrink+g1", "R0 (g1 retrieval)"),
        ("G-shrink+g1b", "G-shrink+g1"),
        ("G-shrink+g1b", "R0 (g1 retrieval)"),
        ("G-learned", "G-post"),
        ("R0+adaptk", "R0 (g1 retrieval)"),
        ("R0+fresh", "R0 (g1 retrieval)"),
    ];
    let mut sp_rows: Vec<Row> = vec![];
    let gstar_name = {
        // declared rule: the gate base is the arm whose name the FINAL cfg inherits; recover from FINAL's gate/id/inj settings
        let f = &arms[arms.iter().position(|a| a.name == "FINAL[mini-hybrid]").unwrap()];
        let cand = ["G-post(inj-always)", "G-post", "G-post+id", "G-post+g1", "G-post+leninj", "G-shrink", "G-shrink(inj-always)", "G-shrink+g1", "G-shrink+g1(inj-gated)", "G-shrink+g1+leninj", "G-shrink+g1b", "G-learned"]
            .into_iter()
            .find(|n| {
                let g = &arms[arms.iter().position(|a| a.name == *n).unwrap()];
                std::mem::discriminant(&g.gate) == std::mem::discriminant(&f.gate) && g.id == f.id && g.and_g1 == f.and_g1 && g.inj_always == f.inj_always && g.inj_conf == f.inj_conf && g.kappa == f.kappa && g.g1_min == f.g1_min && g.g1_margin == f.g1_margin && g.probe0 == f.probe0 && g.backoff == f.backoff
            })
            .unwrap_or("G-post");
        cand.to_string()
    };
    let mut all_pairs = stage_pairs.clone();
    let gs = gstar_name.as_str();
    all_pairs.push(("G*+adaptk", gs));
    all_pairs.push(("FINAL[mini-hybrid]", "G*+adaptk"));
    all_pairs.push(("FINAL[mini-hybrid]", gs));
    for (a, b) in [("FINAL+reset3", "FINAL[mini-hybrid]"), ("FINAL+decay.997", "FINAL[mini-hybrid]"), ("FINAL+decay.99", "FINAL[mini-hybrid]"), ("FINAL+decay.997+reset3", "FINAL[mini-hybrid]"),
                   ("R0+reset3", "R0 (g1 retrieval)"), ("R0+decay.997", "R0 (g1 retrieval)"), ("R0+decay.997+reset3", "R0 (g1 retrieval)"), ("FINAL[mini-hybrid]", "G-post+g1")] {
        all_pairs.push((a, b));
    }
    for (wi, sp) in ws.iter().enumerate() {
        for &(a, b) in &all_pairs {
            for &m in &[3usize, 4] {
                sp_rows.push(Row { fam: "F4:stage-increment", world: sp.name.into(), arm: a.into(), vs: b.into(), metric: METRIC_NAMES[m], c: cmp(&col(&res[wi][find(a)], m), &col(&res[wi][find(b)], m), 0.0), treat: sp.treat, control: !sp.treat, adj: 1.0 });
            }
        }
        // similarity systems end-to-end (spell / hybrid): FINAL[x] vs FINAL[mini], and R0[x] vs R0[mini] (g1's question, re-asked)
        for (a, b) in [("FINAL[mini+spell]", "FINAL[mini]"), ("FINAL[mini-hybrid]", "FINAL[mini]"), ("FINAL[mini-hybrid]", "FINAL[mini+spell]"), ("FINAL[bge]", "FINAL[mini]"), ("FINAL[bge-hybrid]", "FINAL[mini-hybrid]"),
                       ("R0[mini+spell]", "R0[mini]"), ("R0 (g1 retrieval)", "R0[mini]"), ("R0[bge]", "R0[mini]"), ("FINAL[mini-hybrid]", "oracle-sim FINAL")] {
            for &m in &[3usize, 4] {
                sp_rows.push(Row { fam: "F5:similarity-system", world: sp.name.into(), arm: a.into(), vs: b.into(), metric: METRIC_NAMES[m], c: cmp(&col(&res[wi][find(a)], m), &col(&res[wi][find(b)], m), 0.0), treat: sp.treat, control: !sp.treat, adj: 1.0 });
            }
        }
    }
    // Holm per family
    let mut jrows = vec![];
    for fam in ["F1:vs-best-fair", "F2:vs-R0", "F3:vs-R-nogate", "F6:vs-same-front(guarded)"] {
        let idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].fam == fam).collect();
        let adj = holm(&idx.iter().map(|&i| rows[i].c.p).collect::<Vec<_>>());
        for (k, &i) in idx.iter().enumerate() {
            rows[i].adj = adj[k];
            let r = &rows[i];
            jrows.push(json!({"fam": r.fam, "world": r.world, "arm": r.arm, "vs": r.vs, "metric": r.metric, "diff": r.c.diff, "lo": r.c.lo, "hi": r.c.hi, "wins": r.c.wins, "losses": r.c.losses, "p": r.c.p, "holm_p": adj[k], "verdict": mark(adj[k], r.c.diff), "control": r.control, "tost_p_pm0.01": r.c.tost_p}));
        }
    }
    for fam in ["F4:stage-increment", "F5:similarity-system"] {
        let idx: Vec<usize> = (0..sp_rows.len()).filter(|&i| sp_rows[i].fam == fam).collect();
        let adj = holm(&idx.iter().map(|&i| sp_rows[i].c.p).collect::<Vec<_>>());
        for (k, &i) in idx.iter().enumerate() {
            let r = &sp_rows[i];
            jrows.push(json!({"fam": r.fam, "world": r.world, "arm": r.arm, "vs": r.vs, "metric": r.metric, "diff": r.c.diff, "lo": r.c.lo, "hi": r.c.hi, "wins": r.c.wins, "losses": r.c.losses, "p": r.c.p, "holm_p": adj[k], "verdict": mark(adj[k], r.c.diff), "control": r.control, "tost_p_pm0.01": r.c.tost_p}));
        }
    }
    // TOST family on control worlds (H2000, late): arm vs best fair baseline within +-0.01, Holm over the family
    let ctl_idx: Vec<usize> = (0..rows.len()).filter(|&i| (rows[i].fam == "F1:vs-best-fair" || rows[i].fam == "F6:vs-same-front(guarded)") && rows[i].control && rows[i].metric != "H500").collect();
    let tadj = holm(&ctl_idx.iter().map(|&i| rows[i].c.tost_p).collect::<Vec<_>>());
    let mut tost_json = vec![];
    for (k, &i) in ctl_idx.iter().enumerate() {
        let r = &rows[i];
        tost_json.push(json!({"fam": r.fam, "world": r.world, "arm": r.arm, "metric": r.metric, "diff": r.c.diff, "ci95": [r.c.lo, r.c.hi], "tost_p": r.c.tost_p, "tost_holm_p": tadj[k], "equivalent_pm0.01": tadj[k] < 0.05}));
    }

    // ---- console report: F1 for key arms (H2000 + late), controls first
    let key = ["R0 (g1 retrieval)", "R-nogate", "R0-warmfront", "G-post+leninj", "G-shrink", "G-shrink(inj-always)", "G-shrink+g1", "G-post+g1", "G-post(inj-always)", "G-post", "G-post+id", "G-post+g1", "G-learned", "R0+adaptk", "G*+adaptk", "R0+fresh", "FINAL[mini-hybrid]"];
    println!("\n=== F1 vs BEST FAIR baseline (diff [95% CI] Holm verdict); Holm over {} tests", rows.iter().filter(|r| r.fam == "F1:vs-best-fair").count());
    for (wi, sp) in ws.iter().enumerate() {
        println!("-- {}{}", sp.name, if sp.treat { "" } else { "  (CONTROL: TOST +-0.01 shown as T)" });
        for &k in &key {
            let mut line = format!("   {:<22}", k);
            for &m in &[3usize, 4] {
                let r = rows.iter().find(|r| r.fam == "F1:vs-best-fair" && r.world == sp.name && r.arm == k && r.metric == METRIC_NAMES[m]).unwrap();
                let a = r.adj;
                line += &format!(" | {} {:+.4} [{:+.4},{:+.4}] {}", METRIC_NAMES[m], r.c.diff, r.c.lo, r.c.hi, mark(a, r.c.diff));
                let _ = wi;
            }
            println!("{line}");
        }
    }
    println!("\n=== TOST equivalence vs best fair (control worlds; +-0.01; Holm over {} tests) ===", tost_json.len());
    for t in &tost_json {
        if key.contains(&t["arm"].as_str().unwrap()) {
            println!("{:<12} {:<22} {:<5} diff {:+.4} CI [{:+.4},{:+.4}] tost_holm_p {:.4} equivalent={}", t["world"].as_str().unwrap(), t["arm"].as_str().unwrap(), t["metric"].as_str().unwrap(), t["diff"].as_f64().unwrap(), t["ci95"][0].as_f64().unwrap(), t["ci95"][1].as_f64().unwrap(), t["tost_holm_p"].as_f64().unwrap(), t["equivalent_pm0.01"]);
        }
    }
    println!("\n=== stage increments & similarity systems (H2000 / late; Holm per family) ===");
    for fam in ["F4:stage-increment", "F5:similarity-system"] {
        for j in jrows.iter().filter(|j| j["fam"] == fam && j["metric"] == "H2000") {
            println!("{:<22} {:<14} {:<22} - {:<22} {:+.4} [{:+.4},{:+.4}] wins {:>2} holm_p {:.4} {}", fam, j["world"].as_str().unwrap(), j["arm"].as_str().unwrap(), j["vs"].as_str().unwrap(), j["diff"].as_f64().unwrap(), j["lo"].as_f64().unwrap(), j["hi"].as_f64().unwrap(), j["wins"], j["holm_p"].as_f64().unwrap(), j["verdict"].as_str().unwrap());
        }
    }

    // ---- sample efficiency (tasks to target): target = 95% of the best fair baseline's own late mean (last 500 tasks) in that world
    println!("\n=== sample efficiency: tasks until trailing-200 mean >= 95% of best-fair baseline's mean over its last 500 tasks (2001 = never) ===");
    let mut se_json = vec![];
    for (wi, sp) in ws.iter().enumerate() {
        let bf = (0..nb).fold(0, |b, j| {
            let late500 = |a: usize| res[wi][a].iter().map(|r| r.pe[NT - 500..].iter().map(|&x| x as f64).sum::<f64>() / 500.0).sum::<f64>() / res[wi][a].len() as f64;
            if late500(j) > late500(b) { j } else { b }
        });
        let target = 0.95 * res[wi][bf].iter().map(|r| r.pe[NT - 500..].iter().map(|&x| x as f64).sum::<f64>() / 500.0).sum::<f64>() / res[wi][bf].len() as f64;
        let ttt = |a: usize| -> Vec<f64> { res[wi][a].iter().map(|r| tasks_to_target(r, target)).collect() };
        let bt = ttt(bf);
        let mut line = format!("{:<14} target {:.3} ({}): fair {:.0}", sp.name, target, all[bf].name, summarize(&bt).mean);
        for k in ["R0 (g1 retrieval)", "G-post", "FINAL[mini-hybrid]"] {
            let ai = find(k);
            let t = ttt(ai);
            let c = cmp(&t, &bt, 0.0);
            line += &format!(" | {k} {:.0} (d {:+.0} [{:+.0},{:+.0}])", summarize(&t).mean, c.diff, c.lo, c.hi);
            se_json.push(json!({"world": sp.name, "arm": k, "tasks_to_target": summarize(&t).mean, "fair_tasks_to_target": summarize(&bt).mean, "best_fair": all[bf].name, "target": target, "diff": c.diff, "ci": [c.lo, c.hi]}));
        }
        println!("{line}");
    }

    // ---- latency / bytes (single-thread timing of the decision loop, mini-hybrid, sib16, 3 seeds; includes the world's own sim + front)
    let sp0 = &ws[0];
    let time_arm = |n: &str| -> f64 {
        let ai = find(n);
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            for s in 1..=3u64 {
                std::hint::black_box(run(all[ai], sp0, s, &corpus, &pool, &ss));
            }
            best = best.min(t.elapsed().as_secs_f64() * 1e6 / (3.0 * NT as f64));
        }
        best
    };
    let lat: Vec<(String, f64)> = ["guarded(hebr)", "R0 (g1 retrieval)", "G-post", "G-learned", "FINAL[mini-hybrid]", "FINAL[mini]", "FINAL[bge]"].iter().map(|n| (n.to_string(), time_arm(n))).collect();
    println!("\n=== decision-loop latency, us/task (single thread, whole task incl. front update; neighbour scan dominates; sib16, 2000 tasks) ===");
    for (n, l) in &lat {
        println!("{n:<22} {l:>9.1}");
    }
    let bytes = json!({
        "store_per_task_bytes": {"mini(384xf32)": 384 * 4, "bge(768xf32)": 768 * 4, "mini-hybrid(384xf32 + sparse tfidf ~ 40 terms x 8B)": 384 * 4 + 320},
        "anchor_evidence_bytes_per_anchor": NA * 8 * 2 + NA * 4 + 8,
        "learned_gate_bytes": 6 * 8,
        "note": "the gate adds O(anchors) numbers and 0 extra vectors; adaptive-k/fresh add no storage; neighbour scan is O(stored tasks) for every variant",
    });
    let out = json!({"corpus": tag, "confirm": confirm, "held": held, "tune_seeds": tune_seeds, "arms": arms.iter().map(cfg_json).collect::<Vec<_>>(), "table": table, "comparisons": jrows, "tost": tost_json, "sample_efficiency": se_json,
                     "latency_us_per_task": lat.iter().map(|(n, l)| json!({"arm": n, "us": l})).collect::<Vec<_>>(), "bytes": bytes, "gate_base_rule": gstar_name});
    let path = if confirm { "results/g16_route_confirm.json" } else if frozen { "results/g16_route_frozen.json" } else { "results/g16_route.json" };
    std::fs::write(path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote {path} (total {:.1}s)", t0.elapsed().as_secs_f64());
}
