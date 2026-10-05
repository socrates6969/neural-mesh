//! H2-route (R4-A2): END-TO-END judging of hybrid-fusion (RRF k=60, RSF) and cheap top-20 reranking vs the g1 tuned-alpha hybrid and the spell-correct
//! baseline, in the g16 closed loop (knowledge-vote layer on the guarded(hebr) front; routing success = expected success of the played agent).
//! Engine: src/bin/h2_common/mod.rs (copy of g16 worlds+policy engine with pluggable similarity kinds + 1 new control world ctl-noinfo).
//!
//! Usage:
//!   cargo run --release --offline --bin h2_route -- --train-logit                       (train the pair-feature logistic on corpus C (seed 3), write results/h2_logit.json)
//!   cargo run --release --offline --bin h2_route -- --corpus A --check                  (replicate g16 R0 / G-post+g1 bit-exactly)
//!   cargo run --release --offline --bin h2_route -- --corpus A                          (tune on seeds 1-8 -> results/h2_picks.json; held-out 31-60 -> results/h2_route.json)
//!   cargo run --release --offline --bin h2_route -- --corpus B|C2|D2 --confirm          (FROZEN picks, seeds 61-90 -> results/h2_route_confirm_<tag>.json)

use std::time::Instant;

use neural_mesh::stats::summarize;
use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
#[path = "h2_common/mod.rs"]
mod h2;
use common::*;
use h2::*;

const THR_GRID: [f32; 9] = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92];
const THR1_GRID: [f32; 4] = [0.55, 0.65, 0.75, 0.85];

fn kind_json(k: &Kind) -> Value {
    match k {
        Kind::Dense => json!({"t": "dense"}),
        Kind::DenseRaw => json!({"t": "denseraw"}),
        Kind::Hyb(a) => json!({"t": "hyb", "a": a}),
        Kind::Rrf => json!({"t": "rrf"}),
        Kind::Rsf(a) => json!({"t": "rsf", "a": a}),
        Kind::Rerank(s) => json!({"t": "rerank", "s": format!("{:?}", s)}),
        Kind::Oracle => json!({"t": "oracle"}),
    }
}
fn kind_from(v: &Value) -> Kind {
    match v["t"].as_str().unwrap() {
        "dense" => Kind::Dense,
        "denseraw" => Kind::DenseRaw,
        "hyb" => Kind::Hyb(v["a"].as_u64().unwrap() as u8),
        "rrf" => Kind::Rrf,
        "rsf" => Kind::Rsf(v["a"].as_u64().unwrap() as u8),
        "rerank" => Kind::Rerank(match v["s"].as_str().unwrap() { "Jac" => Scorer::Jac, "Bm25" => Scorer::Bm25, _ => Scorer::Logit }),
        _ => Kind::Oracle,
    }
}
fn gate_json(g: &Gate) -> Value {
    match g {
        Gate::Off => json!({"t": "off"}),
        Gate::G1 => json!({"t": "g1"}),
        Gate::Post { delta, gamma, floor } => json!({"t": "post", "delta": delta, "gamma": gamma, "floor": floor}),
        Gate::Learned { .. } => panic!("unused"),
    }
}
fn gate_from(v: &Value) -> Gate {
    match v["t"].as_str().unwrap() {
        "off" => Gate::Off,
        "g1" => Gate::G1,
        _ => Gate::Post { delta: v["delta"].as_f64().unwrap(), gamma: v["gamma"].as_f64().unwrap(), floor: v["floor"].as_f64().unwrap() },
    }
}
fn cfg_json(c: &PolCfg) -> Value {
    json!({"name": c.name, "kind": kind_json(&c.kind), "label": c.kind.label(), "k": c.k, "thr": c.thr, "thr2": c.thr2, "min_n": c.min_n, "gate": gate_json(&c.gate), "and_g1": c.and_g1, "inj_always": c.inj_always, "front": "Guard"})
}
fn cfg_from(v: &Value) -> PolCfg {
    let mut c = PolCfg::vote(v["name"].as_str().unwrap(), kind_from(&v["kind"]), v["k"].as_u64().unwrap() as usize, v["thr"].as_f64().unwrap() as f32, v["min_n"].as_u64().unwrap() as usize, gate_from(&v["gate"]));
    c.thr2 = v["thr2"].as_f64().unwrap() as f32;
    c.and_g1 = v["and_g1"].as_bool().unwrap();
    c.inj_always = v["inj_always"].as_bool().unwrap();
    c
}

fn regime(reg: &str, name: &str, kind: Kind, k: usize, thr: f32, thr2: f32, min_n: usize) -> PolCfg {
    let mut c = match reg {
        "g1" => PolCfg::vote(name, kind, k, thr, min_n, Gate::G1),
        _ => {
            // g16 "G-post+g1" (the conservative variant): P(Beta > front_ref) >= .8 AND the g1 window gate; kn:<anchor> injected only when the vote is open
            let mut c = PolCfg::vote(name, kind, k, thr, min_n, Gate::Post { delta: 0.0, gamma: 0.8, floor: 0.0 });
            c.and_g1 = true;
            c.inj_always = false;
            c
        }
    };
    c.thr2 = thr2;
    c
}

static REFS: std::sync::OnceLock<Vec<(&'static str, f64)>> = std::sync::OnceLock::new();
/// g16 constrained tuning objective: mean over WIN worlds {sib16, rand16, drift} of (H500+H2000)/2 on the tuning seeds
/// MINUS 2 x mean over RISK worlds {ctl-irrel, mixed50, weakid50} of max(0, best-fair - value - .003)
fn panel_obj(cfg: &PolCfg, ws: &[Spec], seeds: &[u64], c: &Corpus, p: &Pool) -> f64 {
    let refs = REFS.get().expect("refs");
    let (mut win, mut nw, mut pen, mut nr) = (0.0, 0.0, 0.0, 0.0);
    for sp in ws.iter().filter(|s| s.tune) {
        let mut v = 0.0;
        for &s in seeds {
            let r = run(cfg, sp, s, c, p);
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
fn tune_best(cands: Vec<PolCfg>, ws: &[Spec], seeds: &[u64], c: &Corpus, p: &Pool) -> (PolCfg, f64) {
    let scored: Vec<(f64, usize)> = cands.par_iter().enumerate().map(|(i, g)| (panel_obj(g, ws, seeds, c, p), i)).collect();
    let (m, i) = scored.into_iter().fold((f64::MIN, 0), |b, x| if x.0 > b.0 { x } else { b });
    (cands[i].clone(), m)
}

/// candidate grid for one (regime, kind family)
fn grid(reg: &str, fam: &str) -> Vec<PolCfg> {
    let mut v = vec![];
    for &k in &[5usize, 8] {
        for &mn in &[2usize, 3] {
            match fam {
                "raw-dense" => {
                    for &t in &THR_GRID {
                        v.push(regime(reg, fam, Kind::DenseRaw, k, t, 0.0, mn));
                    }
                }
                "dense" => {
                    for &t in &THR_GRID {
                        v.push(regime(reg, fam, Kind::Dense, k, t, 0.0, mn));
                    }
                }
                "hybrid" => {
                    for a in [5u8, 6, 7, 8] {
                        for &t in &THR_GRID {
                            v.push(regime(reg, fam, Kind::Hyb(a), k, t, 0.0, mn));
                        }
                    }
                }
                "rrf" => {
                    for &t in &THR_GRID {
                        v.push(regime(reg, fam, Kind::Rrf, k, t, 0.0, mn));
                    }
                }
                "rsf" => {
                    for a in [3u8, 5, 7] {
                        for &t in &THR_GRID {
                            v.push(regime(reg, fam, Kind::Rsf(a), k, t, 0.0, mn));
                        }
                    }
                }
                f if f.starts_with("rerank-") => {
                    let sc = match f { "rerank-jac" => Scorer::Jac, "rerank-bm25" => Scorer::Bm25, _ => Scorer::Logit };
                    let t2s: [f32; 4] = if sc == Scorer::Logit { [0.5, 0.7, 0.85, 0.95] } else { [0.0, 0.3, 0.5, 0.7] };
                    for &t in &THR1_GRID {
                        for &t2 in &t2s {
                            v.push(regime(reg, fam, Kind::Rerank(sc), k, t, t2, mn));
                        }
                    }
                }
                _ => panic!("fam"),
            }
        }
    }
    v
}
const FAMS: [&str; 7] = ["dense", "hybrid", "rrf", "rsf", "rerank-jac", "rerank-bm25", "rerank-logit"];

fn load_logit() -> [f64; NF] {
    let v: Value = serde_json::from_str(&std::fs::read_to_string("results/h2_logit.json").expect("run --train-logit first")).unwrap();
    let mut w = [0.0; NF];
    for (i, x) in v["w"].as_array().unwrap().iter().enumerate() {
        w[i] = x.as_f64().unwrap();
    }
    w
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t0 = Instant::now();
    if args.iter().any(|a| a == "--train-logit") {
        // train on corpus C (g1_corpus.py seed 3): it was used in g16 only for the doc-level project-partition experiment, never in routing tuning/eval.
        let c = load_corpus_p("C");
        let pool = build_pool(&c, None);
        let (x, y) = logit_pairs(&pool, 424242);
        let w = fit_logit::<NF>(&x, &y, 1.0);
        println!("trained on corpus C: {} pairs ({} pos), w = {:?}", x.len(), y.iter().filter(|&&v| v > 0.5).count(), w);
        // report AUC on the training corpus and on A / B / C2 / D2 (queries x queries with fresh LCG pairs)
        let mut rep = vec![];
        for tag in ["C", "A", "B", "C2", "D2"] {
            let c2 = load_corpus_p(tag);
            let p2 = build_pool(&c2, None);
            let (x2, y2) = logit_pairs(&p2, 99);
            let sc: Vec<f64> = x2.iter().map(|f| w.iter().zip(f).map(|(a, b)| a * b).sum::<f64>()).collect();
            let lab: Vec<bool> = y2.iter().map(|&v| v > 0.5).collect();
            let a_logit = auc(&sc, &lab);
            let col = |ix: usize| -> f64 { auc(&x2.iter().map(|f| f[ix]).collect::<Vec<_>>(), &lab) };
            println!("{tag}: AUC logit {:.4} | dense_s {:.4} tfidf_s {:.4} jac {:.4} bm25q {:.4}", a_logit, col(1), col(2), col(5), col(6));
            rep.push(json!({"corpus": tag, "auc_logit": a_logit, "auc_dense_s": col(1), "auc_tfidf_s": col(2), "auc_jac": col(5), "auc_bm25q": col(6), "pairs": x2.len()}));
        }
        std::fs::write("results/h2_logit.json", serde_json::to_string_pretty(&json!({"w": w, "features": ["1", "mini_s", "tfidf_s", "mini", "tfidf", "jac", "bm25q", "lenratio"], "trained_on": "g1 corpus C (seed 3)", "l2": 1.0, "auc": rep})).unwrap()).unwrap();
        return;
    }
    let tag = arg(&args, "--corpus", "A");
    let confirm = args.iter().any(|a| a == "--confirm");
    let check = args.iter().any(|a| a == "--check");
    let held = parse_range(&args, "--held", if confirm { (61, 90) } else { (31, 60) });
    let tune_seeds = parse_range(&args, "--tune", (1, 8));
    let call_us: f64 = arg(&args, "--call-us", "2.0").parse().unwrap();
    let corpus = load_corpus_p(&tag);
    let w = load_logit();
    let pool = build_pool(&corpus, Some(&w));
    let ws = worlds();
    println!("corpus {tag}: pool {} queries, {} worlds; matrices {:.1}s", pool.nq, ws.len(), t0.elapsed().as_secs_f64());

    if check {
        // replicate g16 R0 (mini-hybrid .6/.4, k5, thr .75, min_n 2, g1 gate) and G-post+g1 vs results/g16_route.json
        let g16: Value = serde_json::from_str(&std::fs::read_to_string("results/g16_route.json").unwrap()).unwrap();
        let arms = [("R0 (g1 retrieval)", regime("g1", "R0", Kind::Hyb(6), 5, 0.75, 0.0, 2)), ("G-post+g1", regime("safe", "S", Kind::Hyb(6), 5, 0.75, 0.0, 2))];
        for (gn, cfg) in &arms {
            let mut worst = 0.0f64;
            for sp in ws.iter().filter(|s| s.name != "ctl-noinfo") {
                let r = eval(cfg, sp, &held, &corpus, &pool);
                let row = g16["table"].as_array().unwrap().iter().find(|t| t["world"] == sp.name).unwrap()["rows"].as_array().unwrap().iter().find(|x| x["arm"] == *gn).unwrap().clone();
                let m: Vec<f64> = (0..4).map(|i| mean_of(&r, i)).collect();
                let d = (0..4).map(|i| (m[i] - row["h"][i].as_f64().unwrap()).abs()).fold(0.0, f64::max);
                worst = worst.max(d);
                println!("{gn:<18} {:<14} h2 H2000 {:.6} | g16 {:.6} | max|d| over H100..2000 = {:.2e}", sp.name, m[3], row["h"][3].as_f64().unwrap(), d);
            }
            println!("==> {gn}: worst |d| over 8 g16 worlds = {worst:.2e}");
        }
        return;
    }

    // ------------------------------------------------------------------ tuning (corpus A) / frozen picks
    let picks_path = "results/h2_picks.json";
    let bl = [Base::Keyword, Base::Eps, Base::Pts, Base::HebR, Base::Warm, Base::Guard];
    {
        let mut refs = vec![];
        for rw in ["ctl-irrel", "sib16-mixed50", "sib16-weakid50"] {
            let sp = ws.iter().find(|s| s.name == rw).unwrap();
            let best = bl.iter().map(|b| { let r = eval(&PolCfg::base_kind("b", b.clone()), sp, &tune_seeds, &corpus, &pool); (mean_of(&r, 2) + mean_of(&r, 3)) / 2.0 }).fold(f64::MIN, f64::max);
            refs.push((rw, best));
        }
        let _ = REFS.set(refs);
    }
    let mut arms: Vec<PolCfg> = vec![];
    let mut tune_log: Vec<Value> = vec![];
    let add_fam = arg(&args, "--add-fam", "");
    let frozen = args.iter().any(|a| a == "--frozen");
    if confirm || frozen || !add_fam.is_empty() {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(picks_path).expect("run corpus A first")).unwrap();
        arms = v["arms"].as_array().unwrap().iter().map(cfg_from).collect();
        if !add_fam.is_empty() {
            // tune ONE additional family (both regimes) on the tuning seeds and append it to the picks (earlier picks untouched)
            let mut log: Vec<Value> = v["tune_log"].as_array().cloned().unwrap_or_default();
            for reg in ["g1", "safe"] {
                let (mut b, m) = tune_best(grid(reg, &add_fam), &ws, &tune_seeds, &corpus, &pool);
                b.name = format!("{reg}|{add_fam}");
                println!("tuned {reg:<5} {add_fam:<13}: {:<16} k{} thr{} thr2{} min_n{} obj {:.4}", b.kind.label(), b.k, b.thr, b.thr2, b.min_n, m);
                log.push(json!({"obj": m, "cfg": cfg_json(&b), "added_after_main_run": true}));
                arms.push(b);
            }
            std::fs::write(picks_path, serde_json::to_string_pretty(&json!({"arms": arms.iter().map(cfg_json).collect::<Vec<_>>(), "tune_log": log, "tune_seeds": tune_seeds})).unwrap()).unwrap();
        }
        println!("confirm: {} frozen arms from {picks_path}", arms.len());
    } else {
        for reg in ["g1", "safe"] {
            for fam in FAMS {
                let (b, m) = tune_best(grid(reg, fam), &ws, &tune_seeds, &corpus, &pool);
                let mut b = b;
                b.name = format!("{reg}|{fam}");
                println!("tuned {reg:<5} {fam:<13}: {:<16} k{} thr{} thr2{} min_n{} obj {:.4}  [{:.0}s]", b.kind.label(), b.k, b.thr, b.thr2, b.min_n, m, t0.elapsed().as_secs_f64());
                tune_log.push(json!({"obj": m, "cfg": cfg_json(&b)}));
                arms.push(b);
            }
            // untuned references from g16 (frozen before this work): R0 / G-post+g1 with the mini-hybrid(.6) at thr .75, k5, min_n 2
            arms.push(regime(reg, &format!("{reg}|g16-ref"), Kind::Hyb(6), 5, 0.75, 0.0, 2));
            arms.push(regime(reg, &format!("{reg}|oracle"), Kind::Oracle, 5, 0.15, 0.0, 2));
        }
        std::fs::write(picks_path, serde_json::to_string_pretty(&json!({"arms": arms.iter().map(cfg_json).collect::<Vec<_>>(), "tune_log": tune_log, "tune_seeds": tune_seeds})).unwrap()).unwrap();
        println!("tuning done {:.1}s", t0.elapsed().as_secs_f64());
    }

    // ------------------------------------------------------------------ evaluation
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
    let gi = nb - 1;
    let mut res: Vec<Vec<Vec<RunOut>>> = vec![];
    for sp in &ws {
        let r: Vec<Vec<RunOut>> = all.iter().map(|c| eval(c, sp, &held, &corpus, &pool)).collect();
        res.push(r);
    }
    let best_fair = |wi: usize, m: usize| -> usize { (0..nb).fold(0, |b, j| if mean_of(&res[wi][j], m) > mean_of(&res[wi][b], m) { j } else { b }) };
    let mut table = vec![];
    for (wi, sp) in ws.iter().enumerate() {
        println!("\n=== world {} (w={}, mixed={}, corrupt={}, drift_at={}, {}) | mean expected success; held seeds {}-{} ===", sp.name, sp.w, sp.mixed, sp.corrupt, sp.drift_at, if sp.tune { "tuning panel" } else { "NOT tuned on" }, held[0], held[held.len() - 1]);
        println!("{:<22} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>6} {:>7}", "arm", "H100", "H250", "H500", "H2000", "late", "vote%", "cand%", "calls/t");
        let mut rows = vec![];
        for (ai, c) in all.iter().enumerate() {
            let r = &res[wi][ai];
            let m: Vec<f64> = (0..5).map(|i| mean_of(r, i)).collect();
            let vf = r.iter().map(|x| x.vote_frac).sum::<f64>() / r.len() as f64;
            let cf = r.iter().map(|x| x.cand_frac).sum::<f64>() / r.len() as f64;
            let cl = r.iter().map(|x| x.calls_per_task).sum::<f64>() / r.len() as f64;
            println!("{:<22} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>6.1} {:>6.1} {:>7.2}", c.name, m[0], m[1], m[2], m[3], m[4], vf * 100.0, cf * 100.0, cl);
            rows.push(json!({"arm": c.name, "h": [m[0], m[1], m[2], m[3]], "late": m[4], "vote_frac": vf, "cand_frac": cf, "calls_per_task": cl}));
        }
        table.push(json!({"world": sp.name, "rows": rows}));
    }

    struct Row {
        fam: &'static str,
        world: String,
        arm: String,
        vs: String,
        metric: &'static str,
        c: Cmp,
        control: bool,
        adj: f64,
    }
    let mut rows: Vec<Row> = vec![];
    let metrics = [2usize, 3, 4]; // H500, H2000, late
    for (wi, sp) in ws.iter().enumerate() {
        for ai in nb..all.len() {
            let reg = all[ai].name.split('|').next().unwrap().to_string();
            let spell = find(&format!("{reg}|dense"));
            let r0 = find(&format!("{reg}|g16-ref"));
            for &m in &metrics {
                let a = col(&res[wi][ai], m);
                let eq = if !sp.treat { 0.01 } else { 0.0 };
                rows.push(Row { fam: "F1:vs-same-front(guarded)", world: sp.name.into(), arm: all[ai].name.clone(), vs: "guarded(hebr)".into(), metric: METRIC_NAMES[m], c: cmp(&a, &col(&res[wi][gi], m), eq), control: !sp.treat, adj: 1.0 });
                let bf = best_fair(wi, m);
                rows.push(Row { fam: "F2:vs-best-fair", world: sp.name.into(), arm: all[ai].name.clone(), vs: format!("{} (best fair)", all[bf].name), metric: METRIC_NAMES[m], c: cmp(&a, &col(&res[wi][bf], m), eq), control: !sp.treat, adj: 1.0 });
                if ai != spell && !all[ai].name.ends_with("|oracle") {
                    rows.push(Row { fam: "F3:vs-spell-only(same gate)", world: sp.name.into(), arm: all[ai].name.clone(), vs: all[spell].name.clone(), metric: METRIC_NAMES[m], c: cmp(&a, &col(&res[wi][spell], m), 0.0), control: !sp.treat, adj: 1.0 });
                }
                if let Some(ri) = all.iter().position(|c| c.name == format!("{reg}|raw-dense")) {
                    if ai != ri && !all[ai].name.ends_with("|oracle") {
                        rows.push(Row { fam: "F5:vs-raw-dense(no spell, same gate)", world: sp.name.into(), arm: all[ai].name.clone(), vs: all[ri].name.clone(), metric: METRIC_NAMES[m], c: cmp(&a, &col(&res[wi][ri], m), 0.0), control: !sp.treat, adj: 1.0 });
                    }
                }
                if ai != r0 && !all[ai].name.ends_with("|oracle") {
                    rows.push(Row { fam: "F4:vs-g16-ref(hybrid .6, thr .75)", world: sp.name.into(), arm: all[ai].name.clone(), vs: all[r0].name.clone(), metric: METRIC_NAMES[m], c: cmp(&a, &col(&res[wi][r0], m), 0.0), control: !sp.treat, adj: 1.0 });
                }
            }
        }
    }
    let mut jrows = vec![];
    for fam in ["F1:vs-same-front(guarded)", "F2:vs-best-fair", "F3:vs-spell-only(same gate)", "F4:vs-g16-ref(hybrid .6, thr .75)", "F5:vs-raw-dense(no spell, same gate)"] {
        let idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].fam == fam).collect();
        let adj = holm(&idx.iter().map(|&i| rows[i].c.p).collect::<Vec<_>>());
        for (k, &i) in idx.iter().enumerate() {
            rows[i].adj = adj[k];
            let r = &rows[i];
            jrows.push(json!({"fam": r.fam, "world": r.world, "arm": r.arm, "vs": r.vs, "metric": r.metric, "diff": r.c.diff, "lo": r.c.lo, "hi": r.c.hi, "wins": r.c.wins, "losses": r.c.losses, "p": r.c.p, "holm_p": adj[k], "verdict": mark(adj[k], r.c.diff), "control": r.control, "tost_p_pm0.01": r.c.tost_p}));
        }
    }
    // TOST (+-0.01) on the control worlds vs the same front, H2000+late, Holm over the family
    let ctl_idx: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].fam == "F1:vs-same-front(guarded)" && rows[i].control && rows[i].metric != "H500").collect();
    let tadj = holm(&ctl_idx.iter().map(|&i| rows[i].c.tost_p).collect::<Vec<_>>());
    let mut tost_json = vec![];
    for (k, &i) in ctl_idx.iter().enumerate() {
        let r = &rows[i];
        tost_json.push(json!({"world": r.world, "arm": r.arm, "metric": r.metric, "diff": r.c.diff, "ci95": [r.c.lo, r.c.hi], "tost_p": r.c.tost_p, "tost_holm_p": tadj[k], "equivalent_pm0.01": tadj[k] < 0.05}));
    }

    // ------------------------------------------------------------------ keep rule (declared before running; see board entry)
    // KEEP iff, vs the SAME front without retrieval (guarded(hebr)), H2000, Holm: >=2 treatment worlds {sib16, rand16, sib64, drift, weakid50} are WINS
    // AND no control world {ctl-irrel, ctl-oneoff, ctl-noinfo} is a Holm LOSS. (mixed50 is a risk world: reported as the strict flag.)
    println!("\n=== KEEP RULE vs same front (guarded(hebr)), H2000, Holm over {} tests ===", rows.iter().filter(|r| r.fam == "F1:vs-same-front(guarded)").count());
    println!("{:<22} {:<16} | treatment-world wins | control Holm losses | mixed50 | verdict", "arm", "similarity");
    let mut keep_json = vec![];
    for ai in nb..all.len() {
        let a = &all[ai].name;
        let get = |w: &str| rows.iter().find(|r| r.fam == "F1:vs-same-front(guarded)" && r.world == w && r.arm == *a && r.metric == "H2000").unwrap();
        let tw: Vec<&str> = ["sib16", "rand16", "sib64", "sib16-drift", "sib16-weakid50"].into_iter().filter(|w| mark(get(w).adj, get(w).c.diff) == "WIN").collect();
        let cl: Vec<&str> = ["ctl-irrel", "ctl-oneoff", "ctl-noinfo"].into_iter().filter(|w| mark(get(w).adj, get(w).c.diff) == "LOSS").collect();
        let mx = mark(get("sib16-mixed50").adj, get("sib16-mixed50").c.diff);
        let keep = tw.len() >= 2 && cl.is_empty();
        println!("{:<22} {:<16} | {} {:?} | {} {:?} | {} | {}", a, all[ai].kind.label(), tw.len(), tw, cl.len(), cl, mx, if keep { "KEEP(held-out)" } else { "reject" });
        keep_json.push(json!({"arm": a, "treat_wins": tw, "control_losses": cl, "mixed50": mx, "keep": keep}));
    }

    println!("\n=== F3 (vs spell-only, same gate) and F4 (vs g16 hybrid ref), H2000 Holm verdicts ===");
    for ai in nb..all.len() {
        let a = &all[ai].name;
        if a.ends_with("|oracle") {
            continue;
        }
        let mut line = format!("{a:<22}");
        for fam in ["F3:vs-spell-only(same gate)", "F4:vs-g16-ref(hybrid .6, thr .75)"] {
            let mut s = String::new();
            for sp in &ws {
                if let Some(r) = rows.iter().find(|r| r.fam == fam && r.world == sp.name && r.arm == *a && r.metric == "H2000") {
                    s += &format!(" {}:{:+.3}{}", &sp.name.replace("sib16-", "").replace("ctl-", "c-")[..sp.name.replace("sib16-", "").replace("ctl-", "c-").len().min(7)], r.c.diff, match mark(r.adj, r.c.diff) { "WIN" => "*", "LOSS" => "!", _ => "" });
                }
            }
            line += &format!(" | {}{}", if fam.starts_with("F3") { "vs spell:" } else { "vs ref:" }, s);
        }
        println!("{line}");
    }

    // ------------------------------------------------------------------ latency / cost (single thread, sib16, 3 seeds, 2000 tasks; whole decision loop incl. world sim + front)
    let sp0 = &ws[0];
    let time_arm = |n: &str| -> (f64, f64) {
        let ai = find(n);
        let mut best = f64::MAX;
        let mut calls = 0.0;
        for _ in 0..3 {
            let t = Instant::now();
            for s in 1..=3u64 {
                let r = run(all[ai], sp0, s, &corpus, &pool);
                calls = r.calls_per_task;
                std::hint::black_box(&r);
            }
            best = best.min(t.elapsed().as_secs_f64() * 1e6 / (3.0 * NT as f64));
        }
        (best, calls)
    };
    println!("\n=== cost: decision-loop us/task (single thread, sib16; scan of stored tasks dominates) + scorer calls charged at {call_us} us/call ===");
    let mut lat = vec![];
    for n in ["guarded(hebr)", "g1|dense", "g1|hybrid", "g1|rrf", "g1|rsf", "g1|rerank-jac", "g1|rerank-bm25", "g1|rerank-logit", "safe|dense", "safe|hybrid", "safe|rrf", "safe|rsf", "safe|rerank-logit"] {
        if all.iter().any(|c| c.name == n) {
            let (us, calls) = time_arm(n);
            println!("{n:<22} {us:>8.1} us/task  calls/task {calls:>6.2}  charged {:>8.1} us/task", us + calls * call_us);
            lat.push(json!({"arm": n, "us_per_task": us, "calls_per_task": calls, "charged_us_per_task": us + calls * call_us}));
        }
    }
    let out = json!({"corpus": tag, "confirm": confirm, "held": held, "tune_seeds": tune_seeds, "call_us": call_us, "arms": arms.iter().map(cfg_json).collect::<Vec<_>>(), "table": table, "comparisons": jrows, "tost": tost_json, "keep": keep_json, "cost": lat,
                     "bytes": {"fusion_extra_vectors": 0, "rrf_rsf_extra_state_bytes_per_query": TOPN * 2 * 12, "logit_weights_bytes": NF * 8, "store_per_task_bytes": 384 * 4 + 320, "note": "RRF/RSF/rerank reuse the dense vector + sparse tf-idf the hybrid already stores"}});
    let path = if confirm { format!("results/h2_route_confirm_{tag}.json") } else { "results/h2_route.json".to_string() };
    std::fs::write(&path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote {path} (total {:.1}s)", t0.elapsed().as_secs_f64());
}
