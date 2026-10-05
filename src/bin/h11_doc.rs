//! H11-doc (R5-C TOKEN EFFICIENCY): cost levers in the document-QA research-session loop on the g16/g1 fixtures (REAL MiniLM+tf-idf similarity, REAL hybrid
//! doc rankings and doc lengths, the g16 recurring-concept stream and its worlds incl. controls) with agent prices and a token-level SYNTHETIC cost model
//! (h11_common/mod.rs, dworld.rs). Levers: context size k (real rank-relevance profile / knee), score gate, learned per-anchor context bandit, duplicate
//! suppression, verification gate, result cache (exact / semantic, store-all vs store-verified, reverify-on-hit, evict-on-bad-grade), cost-aware routing, oracles.
//!
//! Protocol (g16): tune seeds 1-8 on a 6-world panel (sib16, rand16, mixed50, weakid50, drift, ctl-none; sib64, ctl-oneoff, dups30 are NEVER tuned on),
//! held 31-60 on corpus A; `--corpus B --confirm`: frozen picks, fresh corpus B texts, seeds 61-90. Paired 95% t-CI over 30 seeds, TOST +-0.01 on success,
//! Holm per family. All costs are synthetic TE (not a real bill).
//!
//! Usage: cargo run --release --offline --bin h11_doc -- --corpus A [--diag]
//!        cargo run --release --offline --bin h11_doc -- --corpus B --confirm

use std::collections::BTreeMap;
use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g1_common/mod.rs"]
mod common;
#[path = "h11_common/mod.rs"]
mod h11;
#[path = "h11_common/dworld.rs"]
mod dworld;
use common::Ctx;
use dworld::*;
use h11::*;

static MU_CELL: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
/// primary exchange rate (quality points per extra reference call of cost): 0.03 default; `--mu 0.1` reruns the whole tune/eval at the strict rate
fn mu() -> f64 {
    *MU_CELL.get().unwrap_or(&0.03)
}
fn mu_alt() -> f64 {
    if (mu() - 0.03).abs() < 1e-9 { 0.10 } else { 0.03 }
}
const RVS: [f64; 8] = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0];
const KS: [usize; 6] = [0, 1, 2, 3, 4, 5];
const TAUS: [f32; 3] = [0.80, 0.90, 0.95];
const NMIN: f64 = 3.0;
const INCL: f64 = 0.001;
/// margin-gate thresholds on z1-z2 (set from the tuning-concept distribution printed by --diag, BEFORE any arm ran: concept p10 .63 / p50 1.87, off-topic p50 .15 / p90 .53)
const ZS: [f32; 6] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];

#[derive(Clone, Copy, Debug)]
struct Picks {
    k: usize,
    zmin: f32,
    rv: f64,
    tau: f32,
    use_gate: bool,
    use_bandit: bool,
    use_dedup: bool,
    use_vg: bool,
    use_memo: bool,
    memo_kind: u8, // 3 = exact verified + reverify, 4 = semantic verified + reverify
}

fn arm_list(p: &Picks) -> Vec<(String, DCfg)> {
    let d0 = DCfg::d0();
    let ctx_best = if p.use_bandit { CtxPol::Bandit { k: p.k.max(1), eps: 0.1 } } else if p.use_gate { CtxPol::Gate { k: p.k.max(1), zmin: p.zmin } } else { CtxPol::K(p.k) };
    let memo_best = if !p.use_memo { (MemoKind::Off, true, false) } else if p.memo_kind == 4 { (MemoKind::Sem { tau: p.tau }, true, true) } else { (MemoKind::Exact, true, true) };
    let s = DCfg {
        route: Route::Anchor { mu: mu() },
        ctx: ctx_best,
        dedup: p.use_dedup,
        vmode: if p.use_vg { DV::Gate { rv: p.rv, nmin: NMIN } } else { DV::Always },
        memo: memo_best.0,
        store_verified: memo_best.1,
        reverify: memo_best.2,
        ..d0
    };
    let mut v: Vec<(String, DCfg)> = vec![("D0".into(), d0)];
    v.push(("glob".into(), DCfg { route: Route::Global, ..d0 }));
    v.push(("route-CA".into(), DCfg { route: Route::Anchor { mu: mu() }, ..d0 }));
    v.push(("route-CA01".into(), DCfg { route: Route::Anchor { mu: 0.01 }, ..d0 }));
    v.push(("route-CA10".into(), DCfg { route: Route::Anchor { mu: 0.10 }, ..d0 }));
    v.push(("or-util10".into(), DCfg { route: Route::OracleUtil { mu: 0.10 }, vmode: DV::Never, ..d0 }));
    v.push(("or-q".into(), DCfg { route: Route::OracleQ, vmode: DV::Never, ..d0 }));
    v.push(("or-util".into(), DCfg { route: Route::OracleUtil { mu: mu() }, vmode: DV::Never, ..d0 }));
    for k in KS {
        v.push((format!("k{k}"), DCfg { ctx: CtxPol::K(k), ..d0 }));
    }
    v.push(("k2+dedup".into(), DCfg { ctx: CtxPol::K(2), dedup: true, ..d0 }));
    v.push(("k3+dedup".into(), DCfg { dedup: true, ..d0 }));
    v.push(("k5+dedup".into(), DCfg { ctx: CtxPol::K(5), dedup: true, ..d0 }));
    v.push(("Kstar".into(), DCfg { ctx: CtxPol::K(p.k), ..d0 }));
    v.push(("gate(k*)".into(), DCfg { ctx: CtxPol::Gate { k: p.k.max(1), zmin: p.zmin }, ..d0 }));
    v.push(("bandit(k*)".into(), DCfg { ctx: CtxPol::Bandit { k: p.k.max(1), eps: 0.1 }, ..d0 }));
    v.push(("Kstar+dedup".into(), DCfg { ctx: CtxPol::K(p.k), dedup: true, ..d0 }));
    v.push(("gate(k3)".into(), DCfg { ctx: CtxPol::Gate { k: 3, zmin: p.zmin }, ..d0 }));
    v.push(("bandit(k3)".into(), DCfg { ctx: CtxPol::Bandit { k: 3, eps: 0.1 }, ..d0 }));
    v.push(("v-never".into(), DCfg { vmode: DV::Never, ..d0 }));
    v.push(("v-gate".into(), DCfg { vmode: DV::Gate { rv: p.rv, nmin: NMIN }, ..d0 }));
    v.push(("memo-ex-all".into(), DCfg { memo: MemoKind::Exact, store_verified: false, reverify: false, ..d0 }));
    v.push(("memo-ex-all-noevict".into(), DCfg { memo: MemoKind::Exact, store_verified: false, reverify: false, evict: false, ..d0 }));
    v.push(("memo-ex-ver".into(), DCfg { memo: MemoKind::Exact, store_verified: true, reverify: false, ..d0 }));
    v.push(("memo-ex-ver+rv".into(), DCfg { memo: MemoKind::Exact, store_verified: true, reverify: true, ..d0 }));
    v.push(("memo-sem-ver+rv".into(), DCfg { memo: MemoKind::Sem { tau: p.tau }, store_verified: true, reverify: true, ..d0 }));
    v.push(("memo-sem-all".into(), DCfg { memo: MemoKind::Sem { tau: p.tau }, store_verified: false, reverify: false, ..d0 }));
    v.push(("S".into(), s));
    v.push(("S-CA".into(), DCfg { route: Route::Anchor { mu: 0.0 }, ..s }));
    v.push(("S-ctx".into(), DCfg { ctx: CtxPol::K(3), dedup: false, ..s }));
    v.push(("S+gate".into(), DCfg { ctx: CtxPol::Gate { k: p.k.max(1), zmin: p.zmin }, ..s }));
    v.push(("S-Vg".into(), DCfg { vmode: DV::Always, ..s }));
    v.push(("S-memo".into(), DCfg { memo: MemoKind::Off, store_verified: true, reverify: false, ..s }));
    v
}

fn eval(ws: &[DSpec], seeds: &[u64], dd: &DData, arms: &[(String, DCfg)]) -> Vec<Vec<Vec<DSum>>> {
    let jobs: Vec<(usize, u64)> = (0..ws.len()).flat_map(|wi| seeds.iter().map(move |&s| (wi, s))).collect();
    let outs: Vec<Vec<DSum>> = jobs.par_iter().map(|&(wi, s)| arms.iter().map(|(_, c)| run_d(c, &ws[wi], s, dd)).collect()).collect();
    let mut res: Vec<Vec<Vec<DSum>>> = vec![vec![vec![]; arms.len()]; ws.len()];
    for (j, (wi, _)) in jobs.iter().enumerate() {
        for (ai, ps) in outs[j].iter().enumerate() {
            res[*wi][ai].push(ps.clone());
        }
    }
    res
}
fn col(v: &[DSum], f: impl Fn(&DSum) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}

fn diag(dd: &DData) {
    println!("\n=== DIAG: real retrieval profile (mini-hybrid, top-{TOPK}; corpus {}) ===", dd.tag);
    let tune_c = |c: i32| c >= 0 && c % 5 == 0;
    let mut z1c: Vec<f32> = vec![];
    let mut z1o: Vec<f32> = vec![];
    let mut marg_c: Vec<f32> = vec![];
    let mut marg_o: Vec<f32> = vec![];
    let mut rel_at = [0.0f64; TOPK + 1];
    let mut hit_at = [0.0f64; TOPK + 1];
    let mut n = 0.0;
    for qi in 0..dd.nq {
        let r = &dd.ranks[qi];
        let c = dd.qconcept[qi];
        if c < 0 {
            z1o.push(r[0].z);
            marg_o.push(r[0].z - r[1].z);
        } else if tune_c(c) {
            z1c.push(r[0].z);
            marg_c.push(r[0].z - r[1].z);
            n += 1.0;
            let mut cum = 0.0;
            for k in 1..=TOPK {
                if dd.doc_concept[r[k - 1].id] == c {
                    cum += 1.0;
                }
                rel_at[k] += cum;
                hit_at[k] += f64::from(u8::from(cum > 0.0));
            }
        }
    }
    let q = |v: &mut Vec<f32>, p: f64| -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[((v.len() - 1) as f64 * p) as usize]
    };
    println!("top-1 fused z: concept queries (tune concepts) p10 {:.2} p25 {:.2} p50 {:.2} p75 {:.2} | off-topic p25 {:.2} p50 {:.2} p75 {:.2} p90 {:.2}", q(&mut z1c, 0.1), q(&mut z1c, 0.25), q(&mut z1c, 0.5), q(&mut z1c, 0.75), q(&mut z1o, 0.25), q(&mut z1o, 0.5), q(&mut z1o, 0.75), q(&mut z1o, 0.9));
    println!("margin z1-z2: concept p10 {:.2} p50 {:.2} | off-topic p50 {:.2} p90 {:.2}", q(&mut marg_c, 0.1), q(&mut marg_c, 0.5), q(&mut marg_o, 0.5), q(&mut marg_o, 0.9));
    println!("rank-relevance profile (tune concepts; 2 relevant docs per concept): k: mean #relevant in top-k | P(at least one relevant) | precision@k");
    for k in 1..=TOPK {
        println!("   k={k}: R {:.3}  hit {:.3}  P@k {:.3}", rel_at[k] / n, hit_at[k] / n, rel_at[k] / n / k as f64);
    }
    let mean_tok = dd.doc_tok.iter().sum::<f64>() / dd.nd as f64;
    println!("doc length: mean {:.1} raw tokens (chars/4), x4 default context scale = {:.0} TE per injected doc; duplicates in the dups world: {} of {} docs", mean_tok, 4.0 * mean_tok, dd.dup_src.len(), dd.nd);
    // exact-repeat statistics of the streams
    let sp = &dworlds()[0];
    let st = dstream(dd, 31, sp);
    let mut seen = std::collections::HashSet::new();
    let rep = st.q.iter().filter(|&&q| !seen.insert(q)).count() as f64 / st.q.len() as f64;
    println!("sib16 stream: exact query-text repeat rate {:.3}; distinct texts {}", rep, seen.len());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tag = _arg(&args, "--corpus", "A");
    let confirm = args.iter().any(|a| a == "--confirm");
    let held = _range(&args, "--held", if confirm { (61, 90) } else { (31, 60) });
    let mu_arg: f64 = _arg(&args, "--mu", "0.03").parse().unwrap();
    MU_CELL.set(mu_arg).unwrap();
    let mu_tag = if (mu_arg - 0.03).abs() < 1e-9 { String::new() } else { format!("_mu{mu_arg}") };
    let out_path = _arg(&args, "--out", &if confirm { format!("results/h11_doc{mu_tag}_confirm61-90.json") } else { format!("results/h11_doc{mu_tag}.json") });
    let picks_path_s = format!("results/h11_doc{mu_tag}_picks.json");
    let picks_path = picks_path_s.as_str();
    let t0 = Instant::now();
    let ctx = Ctx::new(&tag, &["mini"], false);
    let dd = DData::build(&ctx);
    drop(ctx);
    println!("corpus {tag}: {} docs, {} queries; fixtures + similarity + rankings built in {:.1}s", dd.nd, dd.nq, t0.elapsed().as_secs_f64());
    if args.iter().any(|a| a == "--diag") {
        diag(&dd);
        return;
    }
    let ws = dworlds();
    let p0 = Prices::base_d();
    let mut jout = serde_json::Map::new();
    jout.insert("label".into(), json!("SYNTHETIC cost accounting (TE) and context-utility model on real retrieval fixtures; no real token bills"));
    // ------------------------------------------------------------------ tuning (panel worlds, seeds 1-8) or frozen picks
    let picks: Picks = if confirm {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(picks_path).expect("run corpus A first")).unwrap();
        Picks {
            k: v["k"].as_u64().unwrap() as usize,
            zmin: v["zmin"].as_f64().unwrap() as f32,
            rv: v["rv"].as_f64().unwrap(),
            tau: v["tau"].as_f64().unwrap() as f32,
            use_gate: v["use_gate"].as_bool().unwrap(),
            use_bandit: v["use_bandit"].as_bool().unwrap(),
            use_dedup: v["use_dedup"].as_bool().unwrap(),
            use_vg: v["use_vg"].as_bool().unwrap(),
            use_memo: v["use_memo"].as_bool().unwrap(),
            memo_kind: v["memo_kind"].as_u64().unwrap() as u8,
        }
    } else {
        let tune_ws: Vec<DSpec> = ws.iter().filter(|w| w.tune).cloned().collect();
        let seeds: Vec<u64> = (1..=8).collect();
        let d0 = DCfg::d0();
        let objf = |r: &Vec<Vec<Vec<DSum>>>, ai: usize| -> (f64, f64, f64) {
            let n = tune_ws.len() as f64;
            let (mut u, mut q, mut c) = (0.0, 0.0, 0.0);
            for wi in 0..tune_ws.len() {
                u += mean(&col(&r[wi][ai], |s| util_d(s, &p0, mu())));
                q += mean(&col(&r[wi][ai], |s| s.succ));
                c += mean(&col(&r[wi][ai], |s| cost_te(&s.res, &p0)));
            }
            (u / n, q / n, c / n)
        };
        // ---- stage 1: context size k, duplicate suppression, verification gate, cache variants (each vs D0)
        let mut arms: Vec<(String, DCfg)> = vec![("D0".into(), d0)];
        for k in KS {
            arms.push((format!("k{k}"), DCfg { ctx: CtxPol::K(k), ..d0 }));
        }
        arms.push(("dedup".into(), DCfg { dedup: true, ..d0 }));
        for &r in &RVS {
            arms.push((format!("vg{r}"), DCfg { vmode: DV::Gate { rv: r, nmin: NMIN }, ..d0 }));
        }
        arms.push(("memo-ex".into(), DCfg { memo: MemoKind::Exact, store_verified: true, reverify: true, ..d0 }));
        for &t in &TAUS {
            arms.push((format!("memo-sem{t}"), DCfg { memo: MemoKind::Sem { tau: t }, store_verified: true, reverify: true, ..d0 }));
        }
        let r = eval(&tune_ws, &seeds, &dd, &arms);
        let (bu, bq, bc) = objf(&r, 0);
        println!("TUNING (seeds 1-8, panel of {} worlds, mean util(.03)): D0 util {bu:.4} succ {bq:.4} cost {bc:.0}", tune_ws.len());
        let mut tj = vec![];
        let mut best = |r: &Vec<Vec<Vec<DSum>>>, arms: &Vec<(String, DCfg)>, lo: usize, hi: usize, label: &str, refu: f64| -> (usize, f64) {
            let mut b = (lo, f64::MIN);
            for i in lo..hi {
                let (u, q, c) = objf(r, i);
                println!("  {:<14} util {u:.4} succ {q:.4} cost {c:.0}", arms[i].0);
                tj.push(json!({"arm": arms[i].0, "util": u, "succ": q, "cost": c}));
                if u > b.1 {
                    b = (i, u);
                }
            }
            println!("  -> best {label}: {} (util {:.4}, {:+.4} vs ref)", arms[b.0].0, b.1, b.1 - refu);
            b
        };
        let kb = best(&r, &arms, 1, 1 + KS.len(), "context size k", bu);
        let k_star = KS[kb.0 - 1];
        let db = best(&r, &arms, 1 + KS.len(), 2 + KS.len(), "dedup", bu);
        let v0 = 2 + KS.len();
        let vb = best(&r, &arms, v0, v0 + RVS.len(), "verify gate", bu);
        let r_star = RVS[vb.0 - v0];
        let m0 = v0 + RVS.len();
        let mb = best(&r, &arms, m0, m0 + 1 + TAUS.len(), "memo", bu);
        let (memo_kind, tau_star) = if mb.0 == m0 { (3u8, 0.9f32) } else { (4u8, TAUS[mb.0 - m0 - 1]) };
        // ---- stage 2: abstain gate / learned context bandit on top of the tuned k*
        let kk = k_star.max(1);
        let mut arms2: Vec<(String, DCfg)> = vec![("Kstar".into(), DCfg { ctx: CtxPol::K(k_star), ..d0 })];
        for &z in &ZS {
            arms2.push((format!("gate{z}@k{kk}"), DCfg { ctx: CtxPol::Gate { k: kk, zmin: z }, ..d0 }));
        }
        arms2.push((format!("bandit@k{kk}"), DCfg { ctx: CtxPol::Bandit { k: kk, eps: 0.1 }, ..d0 }));
        let r2 = eval(&tune_ws, &seeds, &dd, &arms2);
        let ku = objf(&r2, 0).0;
        let gb = best(&r2, &arms2, 1, 1 + ZS.len(), "score gate on k*", ku);
        let z_star = ZS[gb.0 - 1];
        let bb = best(&r2, &arms2, 1 + ZS.len(), 2 + ZS.len(), "context bandit on k*", ku);
        let pk = Picks {
            k: k_star,
            zmin: z_star,
            rv: r_star,
            tau: tau_star,
            use_gate: gb.1 - ku > INCL && gb.1 >= bb.1,
            use_bandit: bb.1 - ku > INCL && bb.1 > gb.1,
            use_dedup: db.1 - bu > INCL,
            use_vg: vb.1 - bu > INCL,
            use_memo: mb.1 - bu > INCL,
            memo_kind,
        };
        println!("  => picks {pk:?}");
        let pj = json!({"k": pk.k, "zmin": pk.zmin, "rv": pk.rv, "tau": pk.tau, "use_gate": pk.use_gate, "use_bandit": pk.use_bandit, "use_dedup": pk.use_dedup, "use_vg": pk.use_vg, "use_memo": pk.use_memo, "memo_kind": pk.memo_kind, "tuning": tj, "incl_rule": INCL});
        std::fs::write(picks_path, serde_json::to_string(&pj).unwrap()).unwrap();
        pk
    };
    jout.insert("picks".into(), json!(format!("{picks:?}")));
    let arms = arm_list(&picks);
    let names: Vec<String> = arms.iter().map(|a| a.0.clone()).collect();
    let ai = |n: &str| names.iter().position(|x| x == n).unwrap_or_else(|| panic!("arm {n}"));
    let r = eval(&ws, &held, &dd, &arms);
    println!("\nevaluated {} arms x {} worlds x {} seeds in {:.0}s (held {}-{})", arms.len(), ws.len(), held.len(), t0.elapsed().as_secs_f64(), held[0], held[held.len() - 1]);
    {
        let t1 = Instant::now();
        let _ = run_d(&DCfg::d0(), &ws[0], 31, &dd);
        let el = t1.elapsed().as_secs_f64() * 1e6 / NT as f64;
        println!("measured: D0 session loop (anchor scan + retrieval + routing bookkeeping, single thread, noisy box) {el:.1} us/task vs an assumed 2600 us/task encode");
        jout.insert("loop_us_per_task".into(), json!(el));
    }
    let tw: Vec<usize> = ws.iter().enumerate().filter(|(_, w)| w.treat).map(|(i, _)| i).collect();
    let cw: Vec<usize> = ws.iter().enumerate().filter(|(_, w)| !w.treat).map(|(i, _)| i).collect();
    let wname = |wi: usize| ws[wi].name;

    // ------------------------------------------------------------------ 1) decomposition
    println!("\n=== 1. COST DECOMPOSITION of the default session D0 (TE per task, default prices; doc context x4) and the stack S ===");
    let mut dec_j = vec![];
    for wi in 0..ws.len() {
        let mr = |arm: &str| -> Res {
            let v = &r[wi][ai(arm)];
            let mut acc = Res::default();
            for s in v {
                acc.add(&s.res);
            }
            acc.scaled(1.0 / v.len() as f64)
        };
        let (rd, rs) = (mr("D0"), mr("S"));
        let (dd0, c0, cs) = (decompose(&rd, &p0), cost_te(&rd, &p0), cost_te(&rs, &p0));
        let sh: Vec<String> = dd0.iter().map(|(n, x)| format!("{} {:.0}%", n, 100.0 * x / c0)).collect();
        println!("  {:<15} D0 {:>6.0} TE [{}] | S {:>6.0} ({:+.1}%)", wname(wi), c0, sh.join(", "), cs, 100.0 * (cs / c0 - 1.0));
        dec_j.push(json!({"world": wname(wi), "D0": dd0.iter().map(|(n, x)| json!([n, x])).collect::<Vec<_>>(), "D0_total": c0, "S_total": cs, "D0_res": rd.arr(), "S_res": rs.arr()}));
    }
    jout.insert("decomposition".into(), json!(dec_j));

    // ------------------------------------------------------------------ 2) arms vs D0
    println!("\n=== 2. every arm vs D0 (success rate of the DELIVERED answer; cost TE; util(.03)); per group means; EQ = worlds TOST-equivalent within +-.01 (Holm | raw) ===");
    struct Row {
        arm: String,
        wi: usize,
        ds: Cmp,
        dc: Cmp,
        du: Cmp,
        du_hi: Cmp,
        c_base: f64,
    }
    let mut rows: Vec<Row> = vec![];
    for wi in 0..ws.len() {
        let base = &r[wi][ai("D0")];
        for (a, name) in names.iter().enumerate() {
            if name == "D0" {
                continue;
            }
            let v = &r[wi][a];
            rows.push(Row {
                arm: name.clone(),
                wi,
                ds: cmp(&col(v, |s| s.succ), &col(base, |s| s.succ), 0.01),
                dc: cmp(&col(v, |s| cost_te(&s.res, &p0)), &col(base, |s| cost_te(&s.res, &p0)), 0.0),
                du: cmp(&col(v, |s| util_d(s, &p0, mu())), &col(base, |s| util_d(s, &p0, mu())), 0.0),
                du_hi: cmp(&col(v, |s| util_d(s, &p0, mu_alt())), &col(base, |s| util_d(s, &p0, mu_alt())), 0.0),
                c_base: mean(&col(base, |s| cost_te(&s.res, &p0))),
            });
        }
    }
    let h_q = holm(&rows.iter().map(|x| x.ds.tost_p).collect::<Vec<_>>());
    let h_c = holm(&rows.iter().map(|x| x.dc.p).collect::<Vec<_>>());
    let h_u = holm(&rows.iter().map(|x| x.du.p).collect::<Vec<_>>());
    let groups: [(&str, &Vec<usize>); 2] = [("treatment", &tw), ("controls", &cw)];
    println!("{:<22}{:^50}{:^50}", "arm", format!("treatment worlds ({})", tw.len()), format!("controls ({})", cw.len()));
    println!("{:<22}{:^50}{:^50}", "", "dS  EQ  cost-cut%  W/L  dU(mu) dU(alt)", "dS  EQ  cost-cut%  W/L  dU(mu) dU(alt)");
    let mut grp_j = vec![];
    for name in names.iter().filter(|n| *n != "D0") {
        let mut line = format!("{:<22}", name);
        let mut gj = serde_json::Map::new();
        for (gn, idx) in &groups {
            let sel: Vec<usize> = rows.iter().enumerate().filter(|(_, x)| &x.arm == name && idx.contains(&x.wi)).map(|(i, _)| i).collect();
            let n = sel.len() as f64;
            let ds = sel.iter().map(|&i| rows[i].ds.diff).sum::<f64>() / n;
            let eq = sel.iter().filter(|&&i| h_q[i] < 0.05).count();
            let eq_raw = sel.iter().filter(|&&i| rows[i].ds.tost_p < 0.05).count();
            let ni = sel.iter().filter(|&&i| rows[i].ds.lo > -0.01).count();
            let cc = sel.iter().map(|&i| -rows[i].dc.diff / rows[i].c_base).sum::<f64>() / n * 100.0;
            let wc = sel.iter().filter(|&&i| h_c[i] < 0.05 && rows[i].dc.diff < 0.0).count();
            let lc = sel.iter().filter(|&&i| h_c[i] < 0.05 && rows[i].dc.diff > 0.0).count();
            let du = sel.iter().map(|&i| rows[i].du.diff).sum::<f64>() / n;
            let du_hi = sel.iter().map(|&i| rows[i].du_hi.diff).sum::<f64>() / n;
            line += &format!("{:+.4} {}|{}|{}/{} {:>6.1}% {}/{} {:+.4} {:+.4}   ", ds, eq, eq_raw, ni, sel.len(), cc, wc, lc, du, du_hi);
            gj.insert(gn.to_string(), json!({"dsucc": ds, "eq_holm": eq, "eq_raw": eq_raw, "ni": ni, "cost_cut_pct": cc, "cost_wins": wc, "cost_losses": lc, "dU03": du, "dU10": du_hi, "n": sel.len()}));
        }
        println!("{line}");
        grp_j.push(json!({"arm": name, "groups": Value::Object(gj)}));
    }
    jout.insert("groups_vs_D0".into(), json!(grp_j));
    let rows_j: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(i, x)| json!({"arm": x.arm, "world": wname(x.wi), "dsucc": x.ds.diff, "ci": [x.ds.lo, x.ds.hi], "tost_p": x.ds.tost_p, "tost_holm": h_q[i], "dcost": x.dc.diff, "dcost_ci": [x.dc.lo, x.dc.hi], "dcost_holm": h_c[i], "cost_D0": x.c_base, "dU03": x.du.diff, "dU03_holm": h_u[i], "dU10": x.du_hi.diff}))
        .collect();
    jout.insert("rows_vs_D0".into(), json!(rows_j));

    // ------------------------------------------------------------------ 3) per-world key arms
    println!("\n=== 3. key arms per world (mean): succ | cost TE | util(.03) | cache hit rate / wrong-served rate ===");
    let key = ["D0", "k0", "k2", "gate(k3)", "bandit(k3)", "v-gate", "memo-ex-ver+rv", "route-CA", "S", "or-util"];
    print!("{:<15}", "world");
    for k in key {
        print!("{:^26}", k);
    }
    println!();
    for wi in 0..ws.len() {
        print!("{:<15}", wname(wi));
        for k in key {
            let v = &r[wi][ai(k)];
            print!("{:.3} {:>5.0} {:.4} {:.2}/{:.2}  ", mean(&col(v, |s| s.succ)), mean(&col(v, |s| cost_te(&s.res, &p0))), mean(&col(v, |s| util_d(s, &p0, mu()))), mean(&col(v, |s| s.hit)), mean(&col(v, |s| s.wrong_hit)));
        }
        println!();
    }

    // ------------------------------------------------------------------ 4) context knee (REAL rank profile, synthetic utility) and price dependence of the knee
    println!("\n=== 4. context-size curve (k = 0..5 docs, no gate; treatment worlds mean): success, cost, util(.03) by doc-length scale; the utility-optimal k is the knee ===");
    let mut knee_j = vec![];
    for (sn, sp) in [("ctx x1", Prices { ctx_scale: 1.0, ..p0 }), ("ctx x4 (default)", p0), ("ctx x8", Prices { ctx_scale: 8.0, ..p0 })] {
        let mut line = format!("  {:<17}", sn);
        let mut pj = vec![];
        let mut best = (f64::MIN, 0usize);
        for k in KS {
            let a = ai(&format!("k{k}"));
            let u = tw.iter().map(|&wi| mean(&col(&r[wi][a], |s| util_d(s, &sp, mu())))).sum::<f64>() / tw.len() as f64;
            let sc = tw.iter().map(|&wi| mean(&col(&r[wi][a], |s| s.succ))).sum::<f64>() / tw.len() as f64;
            let c = tw.iter().map(|&wi| mean(&col(&r[wi][a], |s| cost_te(&s.res, &sp)))).sum::<f64>() / tw.len() as f64;
            line += &format!("k{k}: {sc:.3}/{c:.0}/{u:.4}  ");
            pj.push(json!({"k": k, "succ": sc, "cost": c, "util": u}));
            if u > best.0 {
                best = (u, k);
            }
        }
        println!("{line} => knee k={}", best.1);
        knee_j.push(json!({"scenario": sn, "points": pj, "knee": best.1}));
    }
    // by world at default
    for wi in 0..ws.len() {
        let mut line = format!("  {:<15}", wname(wi));
        for k in KS {
            let a = ai(&format!("k{k}"));
            line += &format!("k{k}: {:.3}/{:.4}  ", mean(&col(&r[wi][a], |s| s.succ)), mean(&col(&r[wi][a], |s| util_d(s, &p0, mu()))));
        }
        println!("{line}  (succ/util)");
    }
    jout.insert("knee".into(), json!(knee_j));
    jout.insert("knee_analytic".into(), knee_analysis(&dd));

    // ------------------------------------------------------------------ 5) cache
    println!("\n=== 5. result cache: hit rate, wrong-served rate, success and cost vs D0 (treatment | controls means) ===");
    let mut memo_j = vec![];
    for name in ["memo-ex-all", "memo-ex-all-noevict", "memo-ex-ver", "memo-ex-ver+rv", "memo-sem-ver+rv", "memo-sem-all"] {
        let a = ai(name);
        let m = |idx: &Vec<usize>, f: &dyn Fn(&DSum) -> f64| idx.iter().map(|&wi| mean(&col(&r[wi][a], |s| f(s)))).sum::<f64>() / idx.len() as f64;
        let b = |idx: &Vec<usize>, f: &dyn Fn(&DSum) -> f64| idx.iter().map(|&wi| mean(&col(&r[wi][ai("D0")], |s| f(s)))).sum::<f64>() / idx.len() as f64;
        let cost = |s: &DSum| cost_te(&s.res, &p0);
        let succ = |s: &DSum| s.succ;
        let hit = |s: &DSum| s.hit;
        let wh = |s: &DSum| s.wrong_hit;
        println!("  {:<20} treat: hit {:.3} wrong-served {:.3} dS {:+.4} cost {:+.1}% | ctl: hit {:.3} wrong-served {:.3} dS {:+.4} cost {:+.1}%", name, m(&tw, &hit), m(&tw, &wh), m(&tw, &succ) - b(&tw, &succ), 100.0 * (m(&tw, &cost) / b(&tw, &cost) - 1.0), m(&cw, &hit), m(&cw, &wh), m(&cw, &succ) - b(&cw, &succ), 100.0 * (m(&cw, &cost) / b(&cw, &cost) - 1.0));
        let mut per = vec![];
        for wi in 0..ws.len() {
            let v = &r[wi][a];
            let bs = &r[wi][ai("D0")];
            let c = cmp(&col(v, |s| s.succ), &col(bs, |s| s.succ), 0.01);
            println!("      {:<15} hit {:.3} wrong {:.3} dS {:+.4} [{:+.4},{:+.4}] tost_p {:.3} cost {:+.1}%", wname(wi), mean(&col(v, |s| s.hit)), mean(&col(v, |s| s.wrong_hit)), c.diff, c.lo, c.hi, c.tost_p, 100.0 * (mean(&col(v, |s| cost_te(&s.res, &p0))) / mean(&col(bs, |s| cost_te(&s.res, &p0))) - 1.0));
            per.push(json!({"world": wname(wi), "hit": mean(&col(v, |s| s.hit)), "wrong": mean(&col(v, |s| s.wrong_hit)), "dS": c.diff, "ci": [c.lo, c.hi], "tost_p": c.tost_p}));
        }
        memo_j.push(json!({"arm": name, "per_world": per}));
    }
    jout.insert("memo".into(), json!(memo_j));

    // ------------------------------------------------------------------ 6) routing + oracles
    println!("\n=== 6. routing: global price-blind router / anchor router (mu 0) / cost-aware anchor router (mu .03) / hindsight oracles (no verification) ===");
    let mut rt_j = vec![];
    for wi in 0..ws.len() {
        let g = |n: &str| {
            let v = &r[wi][ai(n)];
            (mean(&col(v, |s| s.succ)), mean(&col(v, |s| cost_te(&s.res, &p0))), mean(&col(v, |s| util_d(s, &p0, mu()))))
        };
        let (a, b, c, o1, o2) = (g("glob"), g("D0"), g("route-CA"), g("or-q"), g("or-util"));
        println!("  {:<15} glob {:.3}/{:>5.0}/{:.4} | anchor {:.3}/{:>5.0}/{:.4} | CA {:.3}/{:>5.0}/{:.4} | or-q {:.3}/{:>5.0}/{:.4} | or-util {:.3}/{:>5.0}/{:.4}   (succ/cost/util)", wname(wi), a.0, a.1, a.2, b.0, b.1, b.2, c.0, c.1, c.2, o1.0, o1.1, o1.2, o2.0, o2.1, o2.2);
        let cc = cmp(&col(&r[wi][ai("route-CA")], |s| s.succ), &col(&r[wi][ai("D0")], |s| s.succ), 0.01);
        let cu = cmp(&col(&r[wi][ai("route-CA")], |s| cost_te(&s.res, &p0)), &col(&r[wi][ai("D0")], |s| cost_te(&s.res, &p0)), 0.0);
        println!("      CA vs anchor(mu0): dS {:+.4} [{:+.4},{:+.4}] tost_p {:.3}; cost {:+.0} TE [{:+.0},{:+.0}]", cc.diff, cc.lo, cc.hi, cc.tost_p, cu.diff, cu.lo, cu.hi);
        rt_j.push(json!({"world": wname(wi), "glob": [a.0, a.1, a.2], "anchor": [b.0, b.1, b.2], "CA": [c.0, c.1, c.2], "or_q": [o1.0, o1.1, o1.2], "or_util": [o2.0, o2.1, o2.2], "CA_vs_anchor": {"dS": cc.diff, "ci": [cc.lo, cc.hi], "tost_p": cc.tost_p, "dcost": cu.diff, "dcost_ci": [cu.lo, cu.hi]}}));
    }
    jout.insert("routing".into(), json!(rt_j));
    println!("
  cost-aware routing frontier vs D0 (mean over treatment | controls): dS, cost %, dU(.03), dU(.10)");
    let mut fr_j = vec![];
    for (name, lab) in [("glob", "global price-blind"), ("route-CA", "anchor mu_c=.03"), ("route-CA01", "anchor mu_c=.01"), ("route-CA10", "anchor mu_c=.10"), ("or-util", "oracle util(.03)"), ("or-util10", "oracle util(.10)"), ("or-q", "oracle quality")] {
        let mut cells = vec![];
        let mut cj = serde_json::Map::new();
        for (gn, idx) in [("treat", &tw), ("ctl", &cw)] {
            let (mut ds, mut dc, mut u3, mut u10) = (0.0, 0.0, 0.0, 0.0);
            for &wi in idx.iter() {
                let (a, b) = (&r[wi][ai(name)], &r[wi][ai("D0")]);
                ds += mean(&col(a, |s| s.succ)) - mean(&col(b, |s| s.succ));
                let (ca, cb) = (mean(&col(a, |s| cost_te(&s.res, &p0))), mean(&col(b, |s| cost_te(&s.res, &p0))));
                dc += (ca - cb) / cb * 100.0;
                u3 += mean(&col(a, |s| util_d(s, &p0, 0.03))) - mean(&col(b, |s| util_d(s, &p0, 0.03)));
                u10 += mean(&col(a, |s| util_d(s, &p0, 0.10))) - mean(&col(b, |s| util_d(s, &p0, 0.10)));
            }
            let n = idx.len() as f64;
            cells.push(format!("{gn}: dS {:+.4} cost {:+6.1}% dU03 {:+.4} dU10 {:+.4}", ds / n, dc / n, u3 / n, u10 / n));
            cj.insert(gn.to_string(), json!({"dS": ds / n, "cost_pct": dc / n, "dU03": u3 / n, "dU10": u10 / n}));
        }
        println!("    {:<22} {} | {}", lab, cells[0], cells[1]);
        fr_j.push(json!({"arm": name, "label": lab, "groups": Value::Object(cj)}));
    }
    jout.insert("ca_frontier".into(), json!(fr_j));

    // ------------------------------------------------------------------ 7) stack vs best fair, lever contribution
    println!("\n=== 7. stack S vs best fair incumbent per world (best of D0 / v-gate / memo-ex-ver+rv(no-stack fair cache) by held-out util(.03)) and leave-one-out ===");
    let fair = ["D0", "v-gate", "k2", "memo-ex-ver+rv"];
    let mut ps = vec![];
    let mut bf: Vec<(usize, &str, Cmp, Cmp, Cmp, f64)> = vec![];
    for wi in 0..ws.len() {
        let best = *fair.iter().max_by(|a, b| mean(&col(&r[wi][ai(a)], |s| util_d(s, &p0, mu()))).partial_cmp(&mean(&col(&r[wi][ai(b)], |s| util_d(s, &p0, mu())))).unwrap()).unwrap();
        let (s, b) = (&r[wi][ai("S")], &r[wi][ai(best)]);
        let du = cmp(&col(s, |x| util_d(x, &p0, mu())), &col(b, |x| util_d(x, &p0, mu())), 0.0);
        let ds = cmp(&col(s, |x| x.succ), &col(b, |x| x.succ), 0.01);
        let dc = cmp(&col(s, |x| cost_te(&x.res, &p0)), &col(b, |x| cost_te(&x.res, &p0)), 0.0);
        ps.push(du.p);
        bf.push((wi, best, du, ds, dc, mean(&col(b, |x| cost_te(&x.res, &p0)))));
    }
    let hb = holm(&ps);
    let mut bfj = vec![];
    for (i, (wi, best, du, ds, dc, cb)) in bf.iter().enumerate() {
        println!("  {:<15} best fair {:<14} | dU(.03) {:+.4} [{:+.4},{:+.4}] {} | dS {:+.4} [{:+.4},{:+.4}] tost_p {:.3} | cost {:+.1}%", wname(*wi), best, du.diff, du.lo, du.hi, mark(hb[i], du.diff), ds.diff, ds.lo, ds.hi, ds.tost_p, 100.0 * dc.diff / cb);
        bfj.push(json!({"world": wname(*wi), "best_fair": best, "dU03": du.diff, "ci": [du.lo, du.hi], "holm": hb[i], "dS": ds.diff, "dS_ci": [ds.lo, ds.hi], "tost_p": ds.tost_p, "cost_pct": 100.0 * dc.diff / cb}));
    }
    jout.insert("S_vs_best_fair".into(), json!(bfj));
    let agg = |arm: &str, base: &str, idx: &Vec<usize>| -> (f64, f64, f64) {
        let (mut ds, mut dc, mut du) = (0.0, 0.0, 0.0);
        for &wi in idx {
            let (a, b) = (&r[wi][ai(arm)], &r[wi][ai(base)]);
            ds += mean(&col(a, |s| s.succ)) - mean(&col(b, |s| s.succ));
            let (ca, cb) = (mean(&col(a, |s| cost_te(&s.res, &p0))), mean(&col(b, |s| cost_te(&s.res, &p0))));
            dc += (ca - cb) / cb * 100.0;
            du += mean(&col(a, |s| util_d(s, &p0, mu()))) - mean(&col(b, |s| util_d(s, &p0, mu())));
        }
        let n = idx.len() as f64;
        (ds / n, dc / n, du / n)
    };
    let mut loo_j = vec![];
    println!("  lever contribution (mean over treatment | controls): d success, d cost %, d util(.03)");
    for (arm, base) in [("S", "D0"), ("S-CA", "S"), ("S-ctx", "S"), ("S-Vg", "S"), ("S-memo", "S"), ("v-gate", "D0"), ("v-never", "D0"), ("route-CA", "D0"), ("glob", "D0"), ("k0", "D0"), ("k2", "D0"), ("k5", "D0"), ("k3+dedup", "D0"), ("k5+dedup", "k5"), ("gate(k3)", "D0"), ("bandit(k3)", "D0"), ("memo-ex-ver+rv", "D0")] {
        let (a, b) = (agg(arm, base, &tw), agg(arm, base, &cw));
        println!("    {:<16} vs {:<4} | treat dS {:+.4} cost {:+6.1}% dU {:+.4} | ctl dS {:+.4} cost {:+6.1}% dU {:+.4}", arm, base, a.0, a.1, a.2, b.0, b.1, b.2);
        loo_j.push(json!({"arm": arm, "vs": base, "treat": {"dS": a.0, "cost_pct": a.1, "dU": a.2}, "ctl": {"dS": b.0, "cost_pct": b.1, "dU": b.2}}));
    }
    jout.insert("lever_contrib".into(), json!(loo_j));

    // ------------------------------------------------------------------ 8) Pareto
    println!("\n=== 8. Pareto success vs cost (default prices; * = non-dominated) ===");
    let mut par_j = vec![];
    for (label, idx) in [("sib16", vec![0usize]), ("treatment mean", tw.clone()), ("controls mean", cw.clone())] {
        let pts: Vec<(String, f64, f64)> = names.iter().enumerate().map(|(a, n)| {
            let (mut c, mut q) = (0.0, 0.0);
            for &wi in &idx {
                c += mean(&col(&r[wi][a], |s| cost_te(&s.res, &p0)));
                q += mean(&col(&r[wi][a], |s| s.succ));
            }
            (n.clone(), c / idx.len() as f64, q / idx.len() as f64)
        }).collect();
        println!("  [{label}]");
        let mut sorted = pts.clone();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut pj = vec![];
        for p in &sorted {
            let dom = pts.iter().any(|o| (o.1 <= p.1 && o.2 > p.2 + 1e-9) || (o.1 < p.1 - 1e-9 && o.2 >= p.2));
            println!("    {:<22} cost {:>6.0} TE  succ {:.4} {}", p.0, p.1, p.2, if dom { "" } else { "*" });
            pj.push(json!({"arm": p.0, "cost": p.1, "succ": p.2, "frontier": !dom}));
        }
        par_j.push(json!({"set": label, "points": pj}));
    }
    jout.insert("pareto".into(), json!(par_j));

    // ------------------------------------------------------------------ 9) price sensitivity
    println!("\n=== 9. price sensitivity (treatment worlds): cost % vs D0 under other price vectors (arms were NOT re-tuned) ===");
    let show = ["k0", "k2", "gate(k3)", "bandit(k3)", "v-gate", "memo-ex-ver+rv", "route-CA", "S"];
    print!("{:<34}", "scenario");
    for s in show {
        print!("{:>15}", s);
    }
    println!("{:>9}", "best-util");
    let mut sc_j = vec![];
    for (sn, sp) in scenarios(true) {
        print!("{:<34}", sn);
        let mut row = vec![];
        for s in show {
            let mut acc = 0.0;
            for &wi in &tw {
                let (a, b) = (mean(&col(&r[wi][ai(s)], |x| cost_te(&x.res, &sp))), mean(&col(&r[wi][ai("D0")], |x| cost_te(&x.res, &sp))));
                acc += (a - b) / b * 100.0;
            }
            acc /= tw.len() as f64;
            print!("{:>14.1}%", acc);
            row.push(json!([s, acc]));
        }
        let cand = ["D0", "k0", "k2", "v-gate", "S"];
        let bu = cand.iter().max_by(|a, b| {
            let f = |n: &str| tw.iter().map(|&wi| mean(&col(&r[wi][ai(n)], |x| util_d(x, &sp, mu())))).sum::<f64>();
            f(a).partial_cmp(&f(b)).unwrap()
        }).unwrap();
        println!("{:>9}", bu);
        sc_j.push(json!({"scenario": sn, "cost_pct_vs_D0": row, "best_util_of_D0_k0_k2_vgate_S": bu}));
    }
    jout.insert("price_sens".into(), json!(sc_j));
    println!("\n  exchange rate mu: util by arm, treatment worlds mean");
    let mut mu_j = vec![];
    for mu in [0.0, 0.01, 0.03, 0.1, 0.3] {
        let u: Vec<(String, f64)> = ["D0", "k0", "k2", "v-gate", "memo-ex-ver+rv", "S"].iter().map(|n| (n.to_string(), tw.iter().map(|&wi| mean(&col(&r[wi][ai(n)], |x| util_d(x, &p0, mu)))).sum::<f64>() / tw.len() as f64)).collect();
        println!("    mu {:<5} {}", mu, u.iter().map(|(n, x)| format!("{n} {x:.4}")).collect::<Vec<_>>().join("  "));
        mu_j.push(json!({"mu": mu, "util": u}));
    }
    jout.insert("mu_sens".into(), json!(mu_j));

    // ------------------------------------------------------------------ raw
    let mut raw = vec![];
    for wi in 0..ws.len() {
        let mut m = BTreeMap::new();
        for (a, n) in names.iter().enumerate() {
            let v = &r[wi][a];
            let mut acc = Res::default();
            for s in v {
                acc.add(&s.res);
            }
            m.insert(n.clone(), json!({"succ": mean(&col(v, |s| s.succ)), "succ250": mean(&col(v, |s| s.succ250)), "cost": mean(&col(v, |s| cost_te(&s.res, &p0))), "util03": mean(&col(v, |s| util_d(s, &p0, mu()))), "util10": mean(&col(v, |s| util_d(s, &p0, mu_alt()))),
                "hit": mean(&col(v, |s| s.hit)), "wrong_hit": mean(&col(v, |s| s.wrong_hit)), "k_inj": mean(&col(v, |s| s.k_inj)), "inj_rate": mean(&col(v, |s| s.inj_rate)), "gen_rate": mean(&col(v, |s| s.gen_rate)),
                "res": acc.scaled(1.0 / v.len() as f64).arr(), "succ_seeds": col(v, |s| s.succ), "cost_seeds": col(v, |s| cost_te(&s.res, &p0))}));
        }
        raw.push(json!({"world": wname(wi), "arms": m}));
    }
    jout.insert("raw".into(), json!(raw));
    jout.insert("res_names".into(), json!(RES_NAMES));
    jout.insert("runtime_s".into(), json!(t0.elapsed().as_secs_f64()));
    std::fs::write(&out_path, serde_json::to_string(&Value::Object(jout)).unwrap()).unwrap();
    println!("\nwrote {out_path} in {:.0}s", t0.elapsed().as_secs_f64());
}


/// analytic knee sensitivity on the REAL rank-relevance profile: expected success factor and utility of the top-k context for a mid-price agent (base .8),
/// as a function of the SYNTHETIC context-utility parameters (rho, eta, d) and of the price terms (doc-length scale, mu). No policy loop involved.
fn knee_analysis(dd: &DData) -> Value {
    let mut per_q: Vec<[(usize, usize, f64); 6]> = vec![];
    for qi in 0..dd.nq {
        let c = dd.qconcept[qi];
        if c < 0 {
            continue;
        }
        let list = &dd.ranks[qi];
        let mut a = [(0usize, 0usize, 0.0f64); 6];
        let (mut r, mut i, mut tok) = (0usize, 0usize, 0.0);
        for k in 1..=5 {
            let d = list[k - 1];
            tok += dd.doc_tok[d.id];
            if dd.doc_concept[d.id] == c { r += 1 } else { i += 1 }
            a[k] = (r, i, tok);
        }
        per_q.push(a);
    }
    let n = per_q.len() as f64;
    let fk = |k: usize, rho: f64, eta: f64, d: f64| -> f64 { per_q.iter().map(|a| (1.0 - d * (1.0 - (1.0 - (1.0 - rho).powi(a[k].0 as i32)))) * (1.0 - eta * a[k].1 as f64)).sum::<f64>() / n };
    let tk = |k: usize| -> f64 { per_q.iter().map(|a| a[k].2).sum::<f64>() / n };
    let knee = |rho: f64, eta: f64, d: f64, scale: f64, mu: f64| -> usize {
        let mut b = (f64::MIN, 0usize);
        for k in 0..=5 {
            let u = 0.8 * fk(k, rho, eta, d) - mu * ((B_D + 4.0 * O_D + tk(k) * scale) / S_REF_D - 1.0);
            if u > b.0 {
                b = (u, k);
            }
        }
        b.1
    };
    println!("
=== 4b. ANALYTIC knee sensitivity (real rank-relevance profile over all {} concept queries; mid-price agent, base .8; knee = argmax_k utility) ===", per_q.len());
    println!("   profile: mean tokens/doc {:.1}; E[#relevant in top-k] k=1..5: {}", tk(5) / 5.0, (1..=5).map(|k| format!("{:.2}", per_q.iter().map(|a| a[k].0 as f64).sum::<f64>() / n)).collect::<Vec<_>>().join(" "));
    let mut hist = [0usize; 6];
    let mut tot = 0usize;
    let mut j = vec![];
    for &mu in &[0.0, 0.03, 0.1] {
        println!("   mu {mu}: knee k by (rho, eta) rows x doc-length scale columns [x1 x4 x8], d = .15 / .35 / .6");
        for &rho in &[0.5, 0.7, 0.9] {
            for &eta in &[0.0, 0.01, 0.03, 0.06] {
                let mut cells = vec![];
                for &d in &[0.15, 0.35, 0.6] {
                    let ks: Vec<usize> = [1.0, 4.0, 8.0].iter().map(|&sc| knee(rho, eta, d, sc, mu)).collect();
                    for &k in &ks {
                        hist[k] += 1;
                        tot += 1;
                    }
                    cells.push(format!("{}{}{}", ks[0], ks[1], ks[2]));
                    j.push(json!({"mu": mu, "rho": rho, "eta": eta, "d": d, "knee_x1_x4_x8": ks}));
                }
                println!("      rho {rho} eta {eta:<4}: d.15 [{}]  d.35 [{}]  d.6 [{}]", cells[0], cells[1], cells[2]);
            }
        }
    }
    println!("   knee histogram over all {tot} parameter cells: k0 {} k1 {} k2 {} k3 {} k4 {} k5 {}", hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
    json!({"cells": j, "hist": hist, "relevant_at_k": (1..=5).map(|k| per_q.iter().map(|a| a[k].0 as f64).sum::<f64>() / n).collect::<Vec<_>>()})
}

fn _arg(args: &[String], key: &str, default: &str) -> String {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}
fn _range(args: &[String], key: &str, default: (u64, u64)) -> Vec<u64> {
    parse_range(args, key, default)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_model_diminishing_returns() {
        // success multiplier fit(R): 0 relevant docs < 1 < 2, concave; irrelevant docs only hurt
        let f = |r: usize, i: usize| p_succ(0.8, 0.35, 3, r, i);
        assert!(f(1, 0) > f(0, 0) && f(2, 0) > f(1, 0));
        assert!(f(1, 0) - f(0, 0) > f(2, 0) - f(1, 0), "diminishing returns");
        assert!(f(1, 2) < f(1, 0));
        // controls: d = 0 -> context cannot help, irrelevant docs still dilute
        assert!((p_succ(0.8, 0.0, 3, 2, 0) - 0.8).abs() < 1e-12);
        assert!(p_succ(0.8, 0.0, 3, 0, 2) < 0.8);
        // off-topic one-offs never benefit from docs
        assert!((p_succ(0.8, 0.35, -1, 2, 0) - 0.8).abs() < 1e-12);
    }

    #[test]
    fn engine_invariants() {
        let ctx = Ctx::new("A", &["mini"], false);
        let dd = DData::build(&ctx);
        let ws = dworlds();
        let d0 = DCfg::d0();
        // determinism
        let (a, b) = (run_d(&d0, &ws[0], 5, &dd), run_d(&d0, &ws[0], 5, &dd));
        assert_eq!(a.succ, b.succ);
        assert_eq!(a.res.arr(), b.res.arr());
        // no cache -> no hits; fewer docs -> fewer injected tokens and lower cost
        assert_eq!(a.hit, 0.0);
        let k0 = run_d(&DCfg { ctx: CtxPol::K(0), ..d0 }, &ws[0], 5, &dd);
        let k5 = run_d(&DCfg { ctx: CtxPol::K(5), ..d0 }, &ws[0], 5, &dd);
        let p = Prices::base_d();
        assert!(cost_te(&k0.res, &p) < cost_te(&a.res, &p) && cost_te(&a.res, &p) < cost_te(&k5.res, &p));
        assert!(k0.res.g_ctx == 0.0 || k0.res.g_ctx < 400.0, "k0 injects no generator context (verifier ctx is 0 too)");
        // sib16 repeats texts: exact cache hits a lot, ctl-uniq (no exact repeats) hits almost never
        let memo = DCfg { memo: MemoKind::Exact, store_verified: true, reverify: true, ..d0 };
        let (m0, mu) = (run_d(&memo, &ws[0], 5, &dd), run_d(&memo, ws.iter().find(|w| w.name == "ctl-uniq").unwrap(), 5, &dd));
        assert!(m0.hit > 0.5 && mu.hit < 0.1, "hit rates {} / {}", m0.hit, mu.hit);
        assert!(cost_te(&m0.res, &p) < 0.6 * cost_te(&a.res, &p));
        // verified-answer memo with reverify never serves many wrong answers in the clean world
        assert!(m0.wrong_hit < 0.02);
        // price-weighting: the hindsight quality oracle never has lower per-attempt success probability, but costs more than the cheap-aware oracle
        let oq = run_d(&DCfg { route: Route::OracleQ, vmode: DV::Never, ..d0 }, &ws[0], 5, &dd);
        let ou = run_d(&DCfg { route: Route::OracleUtil { mu: 0.1 }, vmode: DV::Never, ..d0 }, &ws[0], 5, &dd);
        assert!(cost_te(&ou.res, &p) < cost_te(&oq.res, &p));
        // drift: stale cached answers exist when nothing re-verifies or evicts
        let ds = ws.iter().find(|w| w.name == "sib16-drift").unwrap();
        let bad = run_d(&DCfg { memo: MemoKind::Exact, store_verified: false, reverify: false, evict: false, ..d0 }, ds, 5, &dd);
        assert!(bad.wrong_hit > 0.2, "stale answers served: {}", bad.wrong_hit);
    }
}
