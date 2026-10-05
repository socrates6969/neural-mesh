//! H22 part 2: MLP / cascade-grown net as ROUTER on REAL-TEXT-SHAPED worlds, with the REAL 384-d MiniLM embedding of the task sentence as
//! continuous input, vs keyword / eps-greedy / pooled-thompson / hebbian-robust / warm(hebr) / guarded-safe, where the baselines are ALSO
//! given the embedding information (equal-information control) as `kn:<nearest anchor>` tokens (M=32 and M=128 k-means anchors; the
//! idealised offline version of the repo's knowledge-anchor feature).
//!
//! WORLD DEFINITIONS (fixed BEFORE any policy was run; see RKind in h22_common/mod.rs):
//!   text sources A, B = h10 real-doc sentence pools (9240 / 5307 sentences, real keyword features, REAL MiniLM sentence embeddings).
//!   clu12-6ag / clu48-32ag : competence = table over k-means clusters of the SENTENCE EMBEDDING (h10 generator, 25% of clusters switch best agent at T/2),
//!                            keyword prior = token -> modal cluster -> best agent w.p. .65.
//!   smlin-6ag              : skill_a = .5+.42 tanh(w_a.z+b_a), z = 8 standardised random projections of the real embedding (smooth, linear).
//!   smrbf-8ag, smrbf-drift : skill_a = .12+.78 exp(-|z_0..2 - c_a|^2/2.4^2..), centres at random pool points (smooth nonlinear; drift: centres shift at T/2).
//!   ctl-shuf12-6ag         : NEGATIVE CONTROL, competence cluster independent of the text (no information anywhere): everything must tie.
//! Everything about outcomes (competence, keyword prior, Bernoulli draws) is SIMULATED; the text, tokens and embeddings are REAL.
//!
//! PROTOCOL: tune seeds 1-8 (per regime low T=2000 / mid T=6000; high T=20000 reuses mid), held 31-60, confirm 61-90 (same frozen tuning);
//! objective = mean full-run success over the 10 non-control worlds. Baselines tuned per feature variant (tok / +anc32 / +anc128): eps (gamma,eps),
//! pts gamma, warm k. MLP and cascade tuned over input (pca16 / pca64 / raw384) x size x lr x half x replay x keyword-skip.
//! Statistics: paired 95% t-CI of the learned arm vs the BEST FAIR baseline (best mean over the 16 baseline arms, tok+anchor variants, picked per
//! world x regime on the evaluated seeds = conservative), Holm per family per regime; TOST +-0.01 on the same differences.
//! KEEP RULE (declared up-front): keep only if the net beats the best fair baseline by Holm in >1 real-shaped world (full-stream, mid regime) in BOTH held and
//! confirm splits AND is harmless (TOST +-0.01 or documented bounded loss) in the keyword worlds (h22_existing) and the control.
//!
//! Usage: h22_real [--held 31-60] [--out results/h22_real.json] [--quick]
//!        h22_real --timing [--out results/h22_cost.json]

use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "h22_common/mod.rs"]
mod common;
use common::g12::Regime;
use common::*;

use neural_mesh::compact::{CompactConfig, CompactRouter, CompactRouterF64};
use neural_mesh::defaults::recommended_router;
use neural_mesh::router::{Rng, Router};
use neural_mesh::stats::summarize;
use std::collections::HashMap;
use std::sync::Arc;

#[global_allocator]
static GLOBAL: common::g12::alloc_count::Counting = common::g12::alloc_count::Counting;

const DH: usize = 512;

fn pools() -> HashMap<&'static str, Arc<Pool>> {
    let mut m = HashMap::new();
    m.insert("A", load_pool("A"));
    m.insert("B", load_pool("B"));
    m
}

fn t_for(reg: Regime) -> usize {
    match reg {
        Regime::Low => 2000,
        Regime::Mid => 6000,
        Regime::High => 20000,
    }
}

type Mat = Vec<Vec<Vec<RunRes>>>; // [world][arm][seed]

fn run_matrix(specs: &[RealSpec], pl: &HashMap<&'static str, Arc<Pool>>, reg: Regime, arms: &[Arm], seeds: &[u64]) -> Mat {
    let items: Vec<(usize, usize)> = (0..specs.len()).flat_map(|w| (0..seeds.len()).map(move |s| (w, s))).collect();
    let cells: Vec<Vec<RunRes>> = items
        .par_iter()
        .map(|&(wi, si)| {
            let seed = seeds[si];
            let sp = &specs[wi];
            let env = RealW::new(&pl[sp.tag], sp.kind, seed, t_for(reg));
            arms.iter()
                .map(|a| {
                    let mut r = mk_arm(a, &env, seed);
                    run_policy(&env, r.as_mut(), a.var, seed)
                })
                .collect()
        })
        .collect();
    let mut out: Mat = vec![vec![Vec::with_capacity(seeds.len()); arms.len()]; specs.len()];
    for (idx, cell) in cells.into_iter().enumerate() {
        let (wi, _) = items[idx];
        for (ai, r) in cell.into_iter().enumerate() {
            out[wi][ai].push(r);
        }
    }
    out
}

fn objective(m: &Mat, ai: usize) -> f64 {
    mean_of(&m.iter().map(|w| mean_of(&fulls(&w[ai]))).collect::<Vec<_>>())
}

#[derive(Clone, Debug)]
struct Tuned {
    /// per variant [tok, anc32, anc128]: eps (gamma, eps), pts gamma, warm k
    eps: [(f64, f64); 3],
    pts: [f64; 3],
    warm: [f64; 3],
    mlp: MlpCfg,
    casc: CascCfg,
    mlp_anc: MlpCfg,
}

const VARS: [Var; 3] = [Var::Tok, Var::Anc32, Var::Anc128];
const VN: [&str; 3] = ["tok", "anc32", "anc128"];

fn mlp_grid(kw_only_none: bool) -> Vec<MlpCfg> {
    let mut g = vec![];
    let inps: Vec<Inp> = if kw_only_none { vec![Inp::None] } else { vec![Inp::Pca(16), Inp::Pca(64), Inp::Raw] };
    for &inp in &inps {
        for &h in &[16usize, 48] {
            for &lr in &[0.05f32, 0.2] {
                for &half in &[100.0f32, 500.0] {
                    for &rk in &[0usize, 3] {
                        for &kw in &[false, true] {
                            if kw_only_none && kw {
                                continue;
                            }
                            g.push(MlpCfg { inp, dh: DH, h, lr, tau0: 0.3, tau_min: 0.02, half, replay_k: rk, replay_cap: 1000, use_kw: kw });
                        }
                    }
                }
            }
        }
    }
    g
}

fn casc_grid() -> Vec<CascCfg> {
    let mut g = vec![];
    for &inp in &[Inp::Pca(16), Inp::Pca(64), Inp::Raw] {
        for &lr in &[0.05f32, 0.2] {
            for &half in &[100.0f32, 500.0] {
                for &rk in &[0usize, 3] {
                    for &ge in &[200usize, 600] {
                        for &kw in &[false, true] {
                            g.push(CascCfg { inp, dh: DH, lr, tau0: 0.3, tau_min: 0.02, half, replay_k: rk, grow_every: ge, hmax: 6, use_kw: kw, buf_cap: 500, cands: 3, epochs: 20 });
                        }
                    }
                }
            }
        }
    }
    g
}

fn tune(specs: &[RealSpec], pl: &HashMap<&'static str, Arc<Pool>>, reg: Regime, quick: bool) -> (Tuned, Value) {
    let sd = if quick { seeds(1, 2) } else { seeds(1, 8) };
    let tune_specs: Vec<RealSpec> = specs.iter().filter(|s| !s.kind.is_control()).cloned().collect();
    let t0 = Instant::now();
    // ---- baselines
    let mut eps_g = vec![];
    for &g in &[1.0f64, 0.99, 0.97, 0.95] {
        for &e in &[0.0f64, 0.01, 0.02, 0.05, 0.1] {
            eps_g.push((g, e));
        }
    }
    let pts_g = [1.0f64, 0.995, 0.98, 0.95];
    let warm_k = [0.5f64, 1.0, 3.0, 6.0];
    let mut barms: Vec<Arm> = vec![];
    for &v in &VARS {
        for &(g, e) in &eps_g {
            barms.push(arm("eps", K::Eps(g, e), v));
        }
        for &g in &pts_g {
            barms.push(arm("pts", K::Pts(g), v));
        }
        for &k in &warm_k {
            barms.push(arm("warm", K::Warm(k), v));
        }
    }
    let mb = run_matrix(&tune_specs, pl, reg, &barms, &sd);
    let ob: Vec<f64> = (0..barms.len()).map(|a| objective(&mb, a)).collect();
    let per = eps_g.len() + pts_g.len() + warm_k.len();
    let mut eps = [(1.0, 0.0); 3];
    let mut pts = [1.0; 3];
    let mut warm = [1.0; 3];
    for vi in 0..3 {
        let base = vi * per;
        let be = (base..base + eps_g.len()).max_by(|&a, &b| ob[a].partial_cmp(&ob[b]).unwrap()).unwrap();
        let bp = (base + eps_g.len()..base + eps_g.len() + pts_g.len()).max_by(|&a, &b| ob[a].partial_cmp(&ob[b]).unwrap()).unwrap();
        let bw = (base + eps_g.len() + pts_g.len()..base + per).max_by(|&a, &b| ob[a].partial_cmp(&ob[b]).unwrap()).unwrap();
        eps[vi] = eps_g[be - base];
        pts[vi] = pts_g[bp - base - eps_g.len()];
        warm[vi] = warm_k[bw - base - eps_g.len() - pts_g.len()];
    }
    println!("[tune {}] baselines (seeds {}-{}): eps {:?} pts {:?} warm {:?} [{:.0}s]", reg.label(), sd[0], sd[sd.len() - 1], eps, pts, warm, t0.elapsed().as_secs_f64());
    // ---- nets
    let step = if quick { 17 } else { 1 };
    let mg: Vec<MlpCfg> = mlp_grid(false).into_iter().step_by(step).collect();
    let cg: Vec<CascCfg> = casc_grid().into_iter().step_by(step).collect();
    let mag: Vec<MlpCfg> = mlp_grid(true).into_iter().step_by(if quick { 5 } else { 1 }).collect();
    let marms: Vec<Arm> = mg.iter().map(|c| arm("mlp", K::Mlp(*c), Var::Tok)).collect();
    let t1 = Instant::now();
    let mm = run_matrix(&tune_specs, pl, reg, &marms, &sd);
    let om: Vec<f64> = (0..marms.len()).map(|a| objective(&mm, a)).collect();
    let bm = (0..marms.len()).max_by(|&a, &b| om[a].partial_cmp(&om[b]).unwrap()).unwrap();
    println!("[tune {}] mlp best {} obj {:.4} [{:.0}s]", reg.label(), mg[bm].show(), om[bm], t1.elapsed().as_secs_f64());
    let carms: Vec<Arm> = cg.iter().map(|c| arm("casc", K::Casc(*c), Var::Tok)).collect();
    let t2 = Instant::now();
    let mc = run_matrix(&tune_specs, pl, reg, &carms, &sd);
    let oc: Vec<f64> = (0..carms.len()).map(|a| objective(&mc, a)).collect();
    let bc = (0..carms.len()).max_by(|&a, &b| oc[a].partial_cmp(&oc[b]).unwrap()).unwrap();
    println!("[tune {}] casc best {} obj {:.4} [{:.0}s]", reg.label(), cg[bc].show(), oc[bc], t2.elapsed().as_secs_f64());
    let aarms: Vec<Arm> = mag.iter().map(|c| arm("mlp-anc32", K::Mlp(*c), Var::Anc32)).collect();
    let ma = run_matrix(&tune_specs, pl, reg, &aarms, &sd);
    let oa: Vec<f64> = (0..aarms.len()).map(|a| objective(&ma, a)).collect();
    let ba = (0..aarms.len()).max_by(|&a, &b| oa[a].partial_cmp(&oa[b]).unwrap()).unwrap();
    println!("[tune {}] mlp-anc32 (tokens+anchor, no embedding) best {} obj {:.4}", reg.label(), mag[ba].show(), oa[ba]);
    let top = |obj: &Vec<f64>, names: Vec<String>| -> Vec<Value> {
        let mut o: Vec<usize> = (0..obj.len()).collect();
        o.sort_by(|&a, &b| obj[b].partial_cmp(&obj[a]).unwrap());
        o.iter().take(5).map(|&i| json!({"cfg": names[i], "objective": obj[i]})).collect()
    };
    let tm = top(&om, mg.iter().map(|c| c.show()).collect());
    let tc = top(&oc, cg.iter().map(|c| c.show()).collect());
    for t in tm.iter().chain(tc.iter()) {
        println!("    top: {t}");
    }
    // best objective by input type (diagnostic: which input won)
    let by_inp = |inps: Vec<Inp>, obj: &Vec<f64>| -> Value {
        let mut best: HashMap<String, f64> = HashMap::new();
        for (i, inp) in inps.iter().enumerate() {
            let e = best.entry(inp.label()).or_insert(f64::NEG_INFINITY);
            if obj[i] > *e {
                *e = obj[i];
            }
        }
        json!(best)
    };
    let j = json!({"regime": reg.label(), "seeds": [sd[0], sd[sd.len()-1]], "eps": eps.iter().map(|e| json!([e.0, e.1])).collect::<Vec<_>>(), "pts": pts, "warm": warm,
        "mlp": mg[bm].show(), "mlp_obj": om[bm], "casc": cg[bc].show(), "casc_obj": oc[bc], "mlp_anc32": mag[ba].show(), "mlp_anc32_obj": oa[ba],
        "mlp_top5": tm, "casc_top5": tc, "mlp_best_by_input": by_inp(mg.iter().map(|c| c.inp).collect(), &om), "casc_best_by_input": by_inp(cg.iter().map(|c| c.inp).collect(), &oc),
        "grid": {"mlp": mg.len(), "casc": cg.len(), "mlp_anc32": mag.len(), "baselines": barms.len()}});
    (Tuned { eps, pts, warm, mlp: mg[bm], casc: cg[bc], mlp_anc: mag[ba] }, j)
}

struct ArmSet {
    arms: Vec<Arm>,
    fair: Vec<usize>,
    anc_heb: Vec<usize>,
    i_mlp: usize,
    i_casc: usize,
    i_lin: usize,
    i_mlp_tok: usize,
    i_mlp_anc: usize,
    i_orc: Vec<usize>,
}

fn arm_set(t: &Tuned) -> ArmSet {
    let mut arms = vec![arm("keyword@tok", K::Keyword, Var::Tok)];
    let mut fair = vec![0usize];
    let mut anc_heb = vec![];
    for (vi, &v) in VARS.iter().enumerate() {
        let defs: [(&str, K); 5] = [("eps-greedy", K::Eps(t.eps[vi].0, t.eps[vi].1)), ("pooled-thompson", K::Pts(t.pts[vi])), ("hebbian-robust", K::HebR), ("warm(hebr)", K::Warm(t.warm[vi])), ("guarded-safe", K::Safe)];
        for (nm, k) in defs {
            fair.push(arms.len());
            if vi > 0 && matches!(nm, "hebbian-robust" | "warm(hebr)" | "guarded-safe") {
                anc_heb.push(arms.len());
            }
            arms.push(arm(&format!("{nm}@{}", VN[vi]), k, v));
        }
    }
    let mut i_orc = vec![];
    for (nm, k) in [("hebbian-robust", K::HebR), ("warm(hebr)", K::Warm(t.warm[1])), ("guarded-safe", K::Safe)] {
        i_orc.push(arms.len());
        arms.push(arm(&format!("{nm}@oracle-id[ceiling]"), k, Var::Oracle));
    }
    let i_mlp = arms.len();
    arms.push(arm("mlp(emb)", K::Mlp(t.mlp), Var::Tok));
    let i_casc = arms.len();
    arms.push(arm("casc(emb)", K::Casc(t.casc), Var::Tok));
    let i_lin = arms.len();
    let mut lc = t.casc;
    lc.hmax = 0;
    arms.push(arm("casc-linear-head(emb)[ref]", K::Casc(lc), Var::Tok));
    let i_mlp_tok = arms.len();
    let mut mt = t.mlp;
    mt.inp = Inp::None;
    arms.push(arm("mlp-tok[ablation: no embedding]", K::Mlp(mt), Var::Tok));
    let i_mlp_anc = arms.len();
    arms.push(arm("mlp-anc32[equal-info: tokens+anchor, no embedding]", K::Mlp(t.mlp_anc), Var::Anc32));
    ArmSet { arms, fair, anc_heb, i_mlp, i_casc, i_lin, i_mlp_tok, i_mlp_anc, i_orc }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let timing = args.iter().any(|a| a == "--timing");
    let val = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let (hlo, hhi) = val("--held")
        .map(|s| {
            let mut it = s.split('-');
            (it.next().unwrap().parse::<u64>().unwrap(), it.next().unwrap().parse::<u64>().unwrap())
        })
        .unwrap_or((31, 60));
    let out_path = val("--out").unwrap_or_else(|| if timing { "results/h22_cost.json".into() } else { "results/h22_real.json".into() });
    let pl = pools();
    let specs = real_specs();
    if timing {
        run_timing(&pl, &out_path);
        return;
    }
    let eval_seeds = if quick { seeds(hlo, hlo + 1) } else { seeds(hlo, hhi) };
    let regs: Vec<Regime> = if quick { vec![Regime::Low] } else { vec![Regime::Low, Regime::Mid, Regime::High] };
    let t_all = Instant::now();
    println!("H22 real-text-shaped worlds | held {}-{} | RAYON threads {} | worlds {}", eval_seeds[0], eval_seeds[eval_seeds.len() - 1], rayon::current_num_threads(), specs.len());
    let (tl, jl) = tune(&specs, &pl, Regime::Low, quick);
    let (tm, jm) = if quick { (tl.clone(), jl.clone()) } else { tune(&specs, &pl, Regime::Mid, false) };
    let mut reg_json = vec![];
    let mut curves = vec![];
    let mut summary = vec![];
    for &reg in &regs {
        let tn = if reg == Regime::Low { &tl } else { &tm };
        let set = arm_set(tn);
        let t0 = Instant::now();
        let m = run_matrix(&specs, &pl, reg, &set.arms, &eval_seeds);
        let orc: Vec<Vec<RunRes>> = specs.iter().map(|sp| eval_seeds.par_iter().map(|&s| oracle_run(&RealW::new(&pl[sp.tag], sp.kind, s, t_for(reg)))).collect()).collect();
        println!("\n=== regime {} T={} | done in {:.0}s | mlp {} | casc {} | mlp-anc {} ===", reg.label(), t_for(reg), t0.elapsed().as_secs_f64(), tn.mlp.show(), tn.casc.show(), tn.mlp_anc.show());
        let mut f1m = vec![];
        let mut f1c = vec![];
        let mut f2m = vec![];
        let mut f2c = vec![];
        let mut f3 = vec![];
        let mut worlds_json = vec![];
        println!("{:<26} | bestfair(any) | best-anchor-heb | mlp casc lin mlp-tok mlp-anc | orc-id-ceiling | oracle || mlp-bestfair [CI] | casc-bestfair [CI] | mlp-bestAncHeb", "world");
        for (wi, sp) in specs.iter().enumerate() {
            let mf: Vec<f64> = (0..set.arms.len()).map(|a| mean_of(&fulls(&m[wi][a]))).collect();
            let ml: Vec<f64> = (0..set.arms.len()).map(|a| mean_of(&lates(&m[wi][a]))).collect();
            let argmax = |idx: &[usize], v: &[f64]| *idx.iter().max_by(|&&a, &&b| v[a].partial_cmp(&v[b]).unwrap()).unwrap();
            let bf = argmax(&set.fair, &mf);
            let bl = argmax(&set.fair, &ml);
            let bah = argmax(&set.anc_heb, &mf);
            let bahl = argmax(&set.anc_heb, &ml);
            let (lab, fam) = (sp.label(), sp.kind.family());
            let nm = |i: usize| set.arms[i].name.clone();
            for (name, ai, vf, vl, vec) in [("mlp(emb)", set.i_mlp, bf, bl, &mut f1m), ("casc(emb)", set.i_casc, bf, bl, &mut f1c)] {
                vec.push(mk_test(reg.label(), &lab, fam, name, &nm(vf), "full", &fulls(&m[wi][ai]), &fulls(&m[wi][vf])));
                vec.push(mk_test(reg.label(), &lab, fam, name, &nm(vl), "late25%", &lates(&m[wi][ai]), &lates(&m[wi][vl])));
            }
            for (name, ai, vec) in [("mlp(emb)", set.i_mlp, &mut f2m), ("casc(emb)", set.i_casc, &mut f2c)] {
                vec.push(mk_test(reg.label(), &lab, fam, name, &nm(bah), "full", &fulls(&m[wi][ai]), &fulls(&m[wi][bah])));
                vec.push(mk_test(reg.label(), &lab, fam, name, &nm(bahl), "late25%", &lates(&m[wi][ai]), &lates(&m[wi][bahl])));
            }
            // F3: value of the embedding to the MLP
            f3.push(mk_test(reg.label(), &lab, fam, "mlp(emb)", "mlp-tok", "full", &fulls(&m[wi][set.i_mlp]), &fulls(&m[wi][set.i_mlp_tok])));
            f3.push(mk_test(reg.label(), &lab, fam, "mlp(emb)", "mlp-anc32", "full", &fulls(&m[wi][set.i_mlp]), &fulls(&m[wi][set.i_mlp_anc])));
            let fx = f1m.len() - 2;
            let orc_ceiling = if sp.kind.has_oracle_id() { set.i_orc.iter().map(|&i| mf[i]).fold(0.0, f64::max) } else { f64::NAN };
            println!(
                "{:<26} | {:.4} {:<22} | {:.4} {:<22} | {:.4} {:.4} {:.4} {:.4} {:.4} | {:.4} | {:.4} || {:+.4} [{:+.4},{:+.4}] | {:+.4} | {:+.4}",
                lab, mf[bf], nm(bf), mf[bah], nm(bah), mf[set.i_mlp], mf[set.i_casc], mf[set.i_lin], mf[set.i_mlp_tok], mf[set.i_mlp_anc], orc_ceiling, mean_of(&fulls(&orc[wi])), f1m[fx].d, f1m[fx].lo, f1m[fx].hi,
                mf[set.i_casc] - mf[bf], mf[set.i_mlp] - mf[bah]
            );
            let mut arms_j = serde_json::Map::new();
            for (a, ar) in set.arms.iter().enumerate() {
                let s = summarize(&fulls(&m[wi][a]));
                arms_j.insert(ar.name.clone(), json!({"full": s.mean, "full_ci": [s.lo, s.hi], "late25": ml[a]}));
            }
            worlds_json.push(json!({"world": lab, "family": fam, "T": t_for(reg), "oracle_full": mean_of(&fulls(&orc[wi])), "best_fair_full": nm(bf), "best_fair_late": nm(bl), "best_anchor_hebbian_full": nm(bah), "arms": arms_j}));
            if reg == Regime::High {
                for (name, ai) in [("mlp(emb)", set.i_mlp), ("casc(emb)", set.i_casc)] {
                    let (par, nn, ov, diffs, los, am, rm) = parity_curve(&m[wi][ai], &m[wi][bl], 500);
                    println!("    curve {name} vs {} (late-best fair): tasks-to-parity {:?}, sustained diff>=0 {:?}, sustained CI-lo>0 {:?}", nm(bl), par, nn, ov);
                    curves.push(json!({"world": lab, "arm": name, "ref": nm(bl), "window": 500, "tasks_to_parity": par, "tasks_to_nonneg": nn, "tasks_to_sig_overtake": ov, "arm_window_means": am, "ref_window_means": rm, "diff_window": diffs, "ci_lo_window": los}));
                    let (par, nn, ov, _, _, _, _) = parity_curve(&m[wi][ai], &m[wi][bahl], 500);
                    println!("    curve {name} vs {} (late-best ANCHOR-hebbian): tasks-to-parity {:?}, sustained diff>=0 {:?}, sustained CI-lo>0 {:?}", nm(bahl), par, nn, ov);
                    curves.push(json!({"world": lab, "arm": name, "ref": nm(bahl), "ref_kind": "anchor-hebbian", "window": 500, "tasks_to_parity": par, "tasks_to_nonneg": nn, "tasks_to_sig_overtake": ov}));
                }
            }
        }
        for ts in [&mut f1m, &mut f1c, &mut f2m, &mut f2c, &mut f3] {
            holm_family(ts);
        }
        for (lab, ts) in [("F1 mlp vs best fair", &f1m), ("F1 casc vs best fair", &f1c), ("F2 mlp vs best anchor-hebbian", &f2m), ("F2 casc vs best anchor-hebbian", &f2c), ("F3 mlp(emb) vs ablations", &f3)] {
            println!("  {lab} (Holm over {} tests, {}):", ts.len(), reg.label());
            for t in ts.iter() {
                println!("    {:<26} {:<8} vs {:<40} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.3} {} {}", t.world, t.metric, t.vs, t.d, t.lo, t.hi, t.wins, t.losses, t.adj, verdict(t.adj, t.d), if tost_ok(t) { "TOST+-.01 ok" } else { "" });
            }
            let cnt = |v: &str, metric: &str| ts.iter().filter(|t| t.metric == metric && verdict(t.adj, t.d) == v).count();
            println!("    => full: {}W/{}L, late25: {}W/{}L", cnt("WIN", "full"), cnt("LOSS", "full"), cnt("WIN", "late25%"), cnt("LOSS", "late25%"));
            summary.push(json!({"regime": reg.label(), "family": lab, "full_W": cnt("WIN", "full"), "full_L": cnt("LOSS", "full"), "late_W": cnt("WIN", "late25%"), "late_L": cnt("LOSS", "late25%")}));
        }
        reg_json.push(json!({"regime": reg.label(), "T": t_for(reg), "mlp_cfg": tn.mlp.show(), "casc_cfg": tn.casc.show(), "mlp_anc_cfg": tn.mlp_anc.show(), "worlds": worlds_json,
            "F1_mlp": f1m.iter().map(test_json).collect::<Vec<_>>(), "F1_casc": f1c.iter().map(test_json).collect::<Vec<_>>(),
            "F2_mlp": f2m.iter().map(test_json).collect::<Vec<_>>(), "F2_casc": f2c.iter().map(test_json).collect::<Vec<_>>(), "F3": f3.iter().map(test_json).collect::<Vec<_>>()}));
    }
    let out = json!({"experiment": "h22_real", "held": [eval_seeds[0], eval_seeds[eval_seeds.len() - 1]], "note": "simulation only: real text+embeddings, SIMULATED competence/outcomes; routing only",
                     "tune": [jl, jm], "regimes": reg_json, "curves": curves, "summary": summary, "runtime_s": t_all.elapsed().as_secs_f64()});
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nwrote {out_path} ({:.0}s)", t_all.elapsed().as_secs_f64());
}

// =============================================================================================
// cost: ns/decision and bytes of state (single thread)

fn time_router(env: &dyn Arena, mk: &dyn Fn() -> Box<dyn Router>, var: Var, reps: usize) -> (f64, i64) {
    let n = env.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let len = env.len();
    let mut best = f64::MAX;
    let mut live = 0i64;
    for rep in 0..reps {
        let mut rng = Rng::new(5);
        common::g12::alloc_count::start();
        let mut r = mk();
        let t0 = Instant::now();
        for t in 0..len {
            let task = env.task(t, var);
            let a = r.select(task, &cands, &mut rng);
            r.update(task, a, env.outcome(t, a));
        }
        let ns = t0.elapsed().as_nanos() as f64 / len as f64;
        if rep == reps - 1 {
            live = common::g12::alloc_count::live();
        }
        common::g12::alloc_count::stop();
        drop(r);
        if ns < best {
            best = ns;
        }
    }
    (best, live)
}

fn run_timing(pl: &HashMap<&'static str, Arc<Pool>>, out_path: &str) {
    let tuned_mlp = |inp: Inp, h: usize, rk: usize, kw: bool| MlpCfg { inp, dh: DH, h, lr: 0.05, tau0: 0.3, tau_min: 0.02, half: 100.0, replay_k: rk, replay_cap: 1000, use_kw: kw };
    let tuned_casc = |inp: Inp, hmax: usize, rk: usize| CascCfg { inp, dh: DH, lr: 0.05, tau0: 0.3, tau_min: 0.02, half: 100.0, replay_k: rk, grow_every: 300, hmax, use_kw: false, buf_cap: 500, cands: 3, epochs: 20 };
    let mut rows = vec![];
    for (wl, kind) in [("A-clu12-6ag", RKind::Cluster { k: 12, agents: 6 }), ("A-clu48-32ag", RKind::Cluster { k: 48, agents: 32 })] {
        let env = RealW::new(&pl["A"], kind, 1, 6000);
        let n = kind.agents();
        let kw = env.kw_map();
        let mut entries: Vec<(String, Box<dyn Fn() -> Box<dyn Router> + '_>, Var, Option<usize>)> = vec![];
        {
            let kwm = kw.clone();
            entries.push(("keyword".into(), Box::new(move || Box::new(neural_mesh::baselines::KeywordRouter::new(kwm.clone(), n))), Var::Tok, None));
            let kwm = kw.clone();
            entries.push(("guarded-safe (f64 recommended router)".into(), Box::new(move || Box::new(recommended_router(kwm.clone(), n))), Var::Tok, None));
            let kwm = kw.clone();
            entries.push(("guarded-safe @anc32".into(), Box::new(move || Box::new(recommended_router(kwm.clone(), n))), Var::Anc32, None));
            let kwm = kw.clone();
            entries.push(("CompactRouter u16 (defaults)".into(), Box::new(move || Box::new(CompactRouter::new(n, &kwm, CompactConfig::default()).unwrap())), Var::Tok, None));
            let kwm = kw.clone();
            entries.push(("CompactRouterF64".into(), Box::new(move || Box::new(CompactRouterF64::new(n, &kwm, CompactConfig::default()).unwrap())), Var::Tok, None));
        }
        let dense = env.dense();
        for (nm, cfg) in [("mlp pca16 H16 rk0", tuned_mlp(Inp::Pca(16), 16, 0, false)), ("mlp pca16 H48 rk3", tuned_mlp(Inp::Pca(16), 48, 3, false)), ("mlp raw384 H16 rk0", tuned_mlp(Inp::Raw, 16, 0, false)), ("mlp raw384 H48 rk3", tuned_mlp(Inp::Raw, 48, 3, false)), ("mlp tok-only H16 rk0", tuned_mlp(Inp::None, 16, 0, false))] {
            let d = dense.clone();
            let kwm = kw.clone();
            entries.push((nm.into(), Box::new(move || Box::new(Mlp2::new(n, cfg, d.clone(), if cfg.use_kw { Some(kwm.clone()) } else { None }, 1))), Var::Tok, None));
        }
        for (nm, cfg) in [("casc linear head pca16 (hmax0)", tuned_casc(Inp::Pca(16), 0, 0)), ("casc pca16 hmax6", tuned_casc(Inp::Pca(16), 6, 0)), ("casc raw384 hmax0 (linear head)", tuned_casc(Inp::Raw, 0, 0)), ("casc raw384 hmax6 rk3", tuned_casc(Inp::Raw, 6, 3))] {
            let d = dense.clone();
            let kwm = kw.clone();
            entries.push((nm.into(), Box::new(move || Box::new(Casc2::new(n, cfg, d.clone(), if cfg.use_kw { Some(kwm.clone()) } else { None }, 1))), Var::Tok, None));
        }
        for (nm, mk, var, _) in &entries {
            let (ns, live) = time_router(&env, mk.as_ref(), *var, 3);
            println!("{wl:<14} {nm:<42} {ns:>9.0} ns/decision   live heap after T=6000: {:>9.1} kB", live as f64 / 1024.0);
            rows.push(json!({"world": wl, "arm": nm, "ns_per_decision": ns, "live_heap_bytes": live}));
        }
        // CompactRouter owned bytes (its own accounting)
        let mut c = CompactRouter::new(n, &kw, CompactConfig::default()).unwrap();
        let mut rng = Rng::new(5);
        let cands: Vec<usize> = (0..n).collect();
        for t in 0..env.len() {
            let task = env.task(t, Var::Tok);
            let a = c.select(task, &cands, &mut rng);
            c.update(task, a, env.outcome(t, a));
        }
        let b = c.bytes();
        println!("{wl:<14} CompactRouter bytes().owned() = {} B (private_total {} B)", b.owned(), b.private_total());
        rows.push(json!({"world": wl, "arm": "CompactRouter bytes().owned()", "bytes": b.owned(), "private_total": b.private_total()}));
        // parameter bytes of the nets
        let mk_p = |cfg: MlpCfg| Mlp2::new(n, cfg, dense.clone(), None, 1).param_bytes();
        for (nm, cfg) in [("mlp pca16 H48", tuned_mlp(Inp::Pca(16), 48, 0, false)), ("mlp raw384 H16", tuned_mlp(Inp::Raw, 16, 0, false)), ("mlp raw384 H48", tuned_mlp(Inp::Raw, 48, 0, false))] {
            println!("{wl:<14} params {nm}: {} B", mk_p(cfg));
            rows.push(json!({"world": wl, "arm": format!("param bytes {nm}"), "bytes": mk_p(cfg)}));
        }
        let mut cs = Casc2::new(n, tuned_casc(Inp::Raw, 6, 0), dense.clone(), None, 1);
        let mut rng = Rng::new(5);
        for t in 0..env.len() {
            let task = env.task(t, Var::Tok);
            let a = cs.select(task, &cands, &mut rng);
            cs.update(task, a, env.outcome(t, a));
        }
        println!("{wl:<14} casc raw384 after 6000 tasks: hidden units {}, param bytes {}, state incl buffer {}", cs.hidden_units(), cs.param_bytes(), cs.state_bytes());
        rows.push(json!({"world": wl, "arm": "casc raw384 after 6000", "hidden_units": cs.hidden_units(), "param_bytes": cs.param_bytes(), "state_bytes": cs.state_bytes()}));
    }
    // embedding cost note: MiniLM single-sentence encode ms from h10 metadata
    let note = "embedding the task text is NOT included above: MiniLM-L6 single-query encode measured by h10 at 3.3-7.6 ms (results/h10_embmeta_A.json) = 1000x the router decision cost";
    println!("{note}");
    let out = json!({"experiment": "h22_cost", "rows": rows, "note": note, "single_thread": true});
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("wrote {out_path}");
}
