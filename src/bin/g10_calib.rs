//! G10-E1: self-calibrating anchor / similarity threshold.
//!
//! Question (results/g8_triage.md, E1): can the hard-coded `sim_min` of knowledge.rs-style retrieval be replaced by a running
//! (Welford) mean - kstd*std of the observed TOP-1 cosine, with a bootstrap value for the first queries, with ONE frozen kstd
//! across embedders (hash / MiniLM / BGE-base), so that anchor pair-F1 and vote-routing success stay TOST-equivalent (+-0.01) to the
//! per-embedder TUNED sim_min and the unique-context control gets no spurious anchors? The win is removing per-embedder tuning.
//!
//! Families (independent worlds): F5 = results/f5_corpus.json (12 concepts, 3 embedders); G1A/G1B = the hard sibling g1 corpora
//! (A for tune/held seeds, B = fresh corpus for the frozen confirmation). Tuning (seeds 1-8): per (family, embedder) incumbent =
//! best (k, sim_min) of a shared 28-point grid for anchor F1 and (separately) for vote routing; auto = one (k, kstd) tuned on F5 over the
//! three embedders jointly and then FROZEN (zero-shot on G1). Held-out 31-60 / confirm 61-90, TOST margin 0.01, Holm over all cells.
//!
//! Usage:
//!   cargo run --release --offline --bin g10_calib -- --part diag|selfcheck|tune|eval [--held 31-60] [--confirm]

use std::time::Instant;

use rayon::prelude::*;
use serde_json::{json, Value};

#[path = "g10_common/mod.rs"]
mod common;
use common::*;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};

const SIMS: [f32; 14] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];
const KSTDS: [f32; 11] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0];
const ZS: [f32; 12] = [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0];
const MARGIN: f64 = 0.01;

fn tune_names(fam: Fam) -> [&'static str; 3] {
    if fam == Fam::F5 { ["q", "subj3", "q-mixed50"] } else { ["sib16", "rand16", "sib16-mixed50"] }
}
fn tune_worlds(fam: Fam, src: Src) -> Vec<Spec> {
    tune_names(fam).iter().map(|n| world(fam, src, n)).collect()
}
fn inc_grid() -> Vec<Method> {
    let mut v = vec![];
    for &k in &[5usize, 8] {
        for &s in &SIMS {
            v.push(Method::leader(k, s));
        }
    }
    v
}
fn vote_arm(m: &Method) -> A {
    (Kind2::KnVote(m.clone(), std_vote()), Signal::Outcome)
}

/// mean over worlds of mean over seeds
fn f1_of(res: &[Vec<Anc>]) -> f64 {
    mean_of(res.iter().map(|w| mean_of(w.iter().map(|a| a.f1))))
}
fn hv_of(res: &[Vec<RunOut>]) -> f64 {
    mean_of(res.iter().map(|w| mean_of(w.iter().map(|r| 0.5 * (r.h[2] + r.h[3])))))
}

// ------------------------------------------------------------------------------------------------
fn diag() {
    println!("DIAG: top-1 cosine (to ALL earlier queries) statistics, seed 1; concept worlds vs unique control");
    println!("{:<5} {:<5} {:<14} {:>6} {:>6} {:>6} | {:>6} {:>6} | thr@kstd 1/2/3 (final) | tuned-F1 sim", "fam", "src", "world", "mean", "std", "p10", "same", "diff");
    for fam in [Fam::F5, Fam::G1A] {
        for src in SRCS {
            for wn in [tune_names(fam)[0], "ctl-unique"] {
                let sp = world(fam, src, wn);
                let cw = CtxWorld::new(1, &sp);
                let nb = cw.nbrs();
                let mut t1: Vec<f64> = vec![];
                let (mut same, mut diff, mut ns, mut nd) = (0.0, 0.0, 0.0, 0.0);
                for t in 1..cw.emb.len() {
                    let h = &nb.hits[t][0];
                    t1.push(h.1 as f64);
                    if cw.truth[t] >= 0 && cw.truth[h.0 as usize] >= 0 {
                        if cw.truth[t] == cw.truth[h.0 as usize] { same += h.1 as f64; ns += 1.0 } else { diff += h.1 as f64; nd += 1.0 }
                    }
                }
                let mut w = Welford::default();
                t1.iter().for_each(|&x| w.push(x));
                let mut s = t1.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                println!(
                    "{:<5} {:<5} {:<14} {:>6.3} {:>6.3} {:>6.3} | {:>6.3} {:>6.3} | {:.3}/{:.3}/{:.3}",
                    fam.label(), src.label(), wn, w.mean, w.std(), s[s.len() / 10],
                    if ns > 0.0 { same / ns } else { f64::NAN }, if nd > 0.0 { diff / nd } else { f64::NAN },
                    w.mean - w.std(), w.mean - 2.0 * w.std(), w.mean - 3.0 * w.std()
                );
            }
        }
    }
}

/// proof that the cached replay == the library Knowledge object on real worlds (anchor ids identical)
fn selfcheck() {
    let mut bad = 0;
    let mut n = 0;
    for fam in [Fam::F5, Fam::G1A] {
        for src in SRCS {
            for wn in [tune_names(fam)[0], "ctl-unique"] {
                let sp = world(fam, src, wn);
                let cw = CtxWorld::new(2, &sp);
                let nb = cw.nbrs();
                for cfg in [AnchorCfg::Leader { k: 5, sim: 0.6 }, AnchorCfg::Leader { k: 8, sim: 0.8 }, AnchorCfg::Auto { k: 5, kstd: 2.0, n_boot: 30, boot: 0.6, bg: None }] {
                    let mut a = make_anchorer(&cfg, cw.dim, &cw.truth, Some(nb.clone()));
                    let mut b = make_anchorer(&cfg, cw.dim, &cw.truth, None);
                    let mut mism = 0;
                    for t in 0..cw.emb.len() {
                        if a.assign(&cw.emb[t]) != b.assign(&cw.emb[t]) {
                            mism += 1;
                        }
                    }
                    n += 1;
                    if mism > 0 {
                        bad += 1;
                    }
                    println!("selfcheck {:<5} {:<5} {:<12} {:<44} mismatches {}", fam.label(), src.label(), wn, cfg_label(&cfg), mism);
                }
            }
        }
    }
    println!("selfcheck: {} of {} (world x config) runs had ANY anchor mismatch between cached replay and library Knowledge", bad, n);
}
fn cfg_label(c: &AnchorCfg) -> String {
    Method { anchor: c.clone() }.label()
}

// ------------------------------------------------------------------------------------------------
struct Inc {
    f1: Method,
    vote: Method,
    f1_score: f64,
    vote_score: f64,
}

fn tune_incumbents(fam: Fam, src: Src, seeds: &[u64]) -> Inc {
    let grid = inc_grid();
    let ws = tune_worlds(fam, src);
    let ra = eval_anchor(&grid, &ws, seeds);
    let arms: Vec<A> = grid.iter().map(vote_arm).collect();
    let rr = eval_runs(&arms, &ws, seeds);
    let f1s: Vec<f64> = (0..grid.len()).map(|i| f1_of(&ra[i])).collect();
    let vs: Vec<f64> = (0..grid.len()).map(|i| hv_of(&rr[i])).collect();
    let bf = (0..grid.len()).max_by(|&a, &b| f1s[a].partial_cmp(&f1s[b]).unwrap()).unwrap();
    let bv = (0..grid.len()).max_by(|&a, &b| vs[a].partial_cmp(&vs[b]).unwrap()).unwrap();
    println!("  incumbent {} {}: best F1 {:.4} {}  | best vote(H500/2000) {:.4} {}", fam.label(), src.label(), f1s[bf], grid[bf].label(), vs[bv], grid[bv].label());
    // sim profile of the F1 and vote objective for the record
    let prof: Vec<String> = (0..grid.len()).filter(|i| grid[*i].label().starts_with("leader(k8")).map(|i| format!("{:.2}:{:.3}/{:.3}", SIMS[i % SIMS.len()], f1s[i], vs[i])).collect();
    println!("    k8 profile sim:F1/vote  {}", prof.join(" "));
    Inc { f1: grid[bf].clone(), vote: grid[bv].clone(), f1_score: f1s[bf], vote_score: vs[bv] }
}

fn auto_grid() -> Vec<Method> {
    let mut v = vec![];
    for &k in &[5usize, 8] {
        for &ks in &KSTDS {
            v.push(Method { anchor: AnchorCfg::Auto { k, kstd: ks, n_boot: 30, boot: 0.6, bg: None } });
        }
    }
    for &k in &[5usize, 8] {
        for &z in &ZS {
            v.push(Method { anchor: AnchorCfg::Auto { k, kstd: 99.0, n_boot: 10, boot: 0.6, bg: Some(z) } });
        }
    }
    v
}

fn tune(out: &str) {
    let t0 = Instant::now();
    let seeds: Vec<u64> = (1..=8).collect();
    println!("== tune: seeds 1-8; incumbents per (family, embedder), auto grid on F5 (frozen afterwards)");
    let mut inc: Vec<(Fam, Src, Inc)> = vec![];
    for fam in [Fam::F5, Fam::G1A] {
        for src in SRCS {
            inc.push((fam, src, tune_incumbents(fam, src, &seeds)));
        }
    }
    let grid = auto_grid();
    // auto grid evaluated on BOTH families (F5 used for the selection; G1A reported)
    #[derive(Clone)]
    struct Row {
        f1: Vec<f64>,
        hv: Vec<f64>,
        spur: Vec<f64>,
    }
    let mut table: Vec<(Fam, Vec<Row>)> = vec![];
    for fam in [Fam::F5, Fam::G1A] {
        let mut rows = vec![Row { f1: vec![], hv: vec![], spur: vec![] }; grid.len()];
        for src in SRCS {
            let ws = tune_worlds(fam, src);
            let ra = eval_anchor(&grid, &ws, &seeds);
            let arms: Vec<A> = grid.iter().map(vote_arm).collect();
            let rr = eval_runs(&arms, &ws, &seeds);
            let cu = vec![world(fam, src, "ctl-unique")];
            let rc = eval_anchor(&grid, &cu, &seeds);
            for gi in 0..grid.len() {
                rows[gi].f1.push(f1_of(&ra[gi]));
                rows[gi].hv.push(hv_of(&rr[gi]));
                rows[gi].spur.push(mean_of(rc[gi][0].iter().map(|a| a.spur)));
            }
        }
        table.push((fam, rows));
    }
    // incumbent spur on ctl-unique (their F1 pick)
    let mut inc_spur: Vec<(Fam, Vec<f64>)> = vec![];
    for fam in [Fam::F5, Fam::G1A] {
        let mut sp = vec![];
        for src in SRCS {
            let i = inc.iter().find(|x| x.0 == fam && x.1 == src).unwrap();
            let r = eval_anchor(&[i.2.f1.clone(), i.2.vote.clone()], &[world(fam, src, "ctl-unique")], &seeds);
            sp.push(mean_of(r[0][0].iter().map(|a| a.spur)));
        }
        inc_spur.push((fam, sp));
    }
    let loss = |fam: Fam, gi: usize| -> f64 {
        let rows = &table.iter().find(|t| t.0 == fam).unwrap().1;
        let sp = &inc_spur.iter().find(|t| t.0 == fam).unwrap().1;
        let mut l = 0.0;
        for (si, src) in SRCS.iter().enumerate() {
            let i = &inc.iter().find(|x| x.0 == fam && x.1 == *src).unwrap().2;
            l += (rows[gi].f1[si] - i.f1_score).abs() + (rows[gi].hv[si] - i.vote_score).abs() + (rows[gi].spur[si] - sp[si] - MARGIN).max(0.0);
        }
        l
    };
    println!("\nauto grid (tune seeds 1-8). per embedder [hash mini bge]: F1 | H500/2000 | spur(ctl-unique); LOSS = sum_src |dF1|+|dVote|+max(0,dSpur-.01) vs that family's per-embedder incumbent");
    for fam in [Fam::F5, Fam::G1A] {
        println!("-- family {}  incumbent F1 [{}] vote [{}] spur [{}]", fam.label(),
            SRCS.iter().map(|s| format!("{:.3}", inc.iter().find(|x| x.0 == fam && x.1 == *s).unwrap().2.f1_score)).collect::<Vec<_>>().join(" "),
            SRCS.iter().map(|s| format!("{:.3}", inc.iter().find(|x| x.0 == fam && x.1 == *s).unwrap().2.vote_score)).collect::<Vec<_>>().join(" "),
            inc_spur.iter().find(|t| t.0 == fam).unwrap().1.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>().join(" "));
        let rows = &table.iter().find(|t| t.0 == fam).unwrap().1;
        for (gi, g) in grid.iter().enumerate() {
            let f = |v: &Vec<f64>| v.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>().join(" ");
            println!("   {:<44} F1 [{}] vote [{}] spur [{}] loss {:.4}", g.label(), f(&rows[gi].f1), f(&rows[gi].hv), f(&rows[gi].spur), loss(fam, gi));
        }
    }
    // selection on F5 only
    let pick = |pure: bool| -> usize {
        (0..grid.len())
            .filter(|&gi| matches!(&grid[gi].anchor, AnchorCfg::Auto { kstd, .. } if (*kstd < 90.0) == pure))
            .min_by(|&a, &b| loss(Fam::F5, a).partial_cmp(&loss(Fam::F5, b)).unwrap())
            .unwrap()
    };
    let (gp, gb) = (pick(true), pick(false));
    println!("\nSELECTED (F5 tune loss): pure auto = {} (loss {:.4}; its G1A loss {:.4}) | auto+bg = {} (loss {:.4}; G1A loss {:.4})",
        grid[gp].label(), loss(Fam::F5, gp), loss(Fam::G1A, gp), grid[gb].label(), loss(Fam::F5, gb), loss(Fam::G1A, gb));
    let best_g1 = (0..grid.len()).filter(|&gi| matches!(&grid[gi].anchor, AnchorCfg::Auto { kstd, .. } if *kstd < 90.0)).min_by(|&a, &b| loss(Fam::G1A, a).partial_cmp(&loss(Fam::G1A, b)).unwrap()).unwrap();
    println!("(for reference, best PURE auto if it had been tuned on G1A instead: {} loss {:.4})", grid[best_g1].label(), loss(Fam::G1A, best_g1));
    let doc = json!({
        "lab": "g10_calib/tune", "tune_seeds": [1, 8], "margin": MARGIN,
        "incumbents": inc.iter().map(|(f, s, i)| json!({"fam": f.label(), "src": s.label(), "f1": method_json(&i.f1), "vote": method_json(&i.vote), "f1_score": i.f1_score, "vote_score": i.vote_score})).collect::<Vec<_>>(),
        "auto_pure": method_json(&grid[gp]), "auto_bg": method_json(&grid[gb]),
        "grid": grid.iter().enumerate().map(|(gi, g)| json!({"method": g.label(), "f5": {"f1": table[0].1[gi].f1, "hv": table[0].1[gi].hv, "spur": table[0].1[gi].spur, "loss": loss(Fam::F5, gi)},
            "g1A": {"f1": table[1].1[gi].f1, "hv": table[1].1[gi].hv, "spur": table[1].1[gi].spur, "loss": loss(Fam::G1A, gi)}})).collect::<Vec<_>>(),
        "elapsed_s": t0.elapsed().as_secs_f64()
    });
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out} in {:.0}s", t0.elapsed().as_secs_f64());
}

// ------------------------------------------------------------------------------------------------
struct Sel {
    inc: Vec<(String, String, Method, Method)>,
    auto: Method,
    auto_bg: Method,
}
fn load_sel(path: &str) -> Sel {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap_or_else(|_| panic!("{path} missing: run --part tune"))).unwrap();
    Sel {
        inc: v["incumbents"].as_array().unwrap().iter().map(|i| (i["fam"].as_str().unwrap().to_string(), i["src"].as_str().unwrap().to_string(), method_from_json(&i["f1"]), method_from_json(&i["vote"]))).collect(),
        auto: method_from_json(&v["auto_pure"]),
        auto_bg: method_from_json(&v["auto_bg"]),
    }
}

#[derive(Clone)]
struct TCell {
    fam: String,
    src: String,
    world: String,
    metric: String,
    ref_name: String,
    mean_cand: f64,
    mean_ref: f64,
    diff: f64,
    lo90: f64,
    hi90: f64,
    p: f64,
    holm: f64,
    cand_label: String,
}

fn tost_cell(fam: &str, src: &str, world: &str, metric: &str, ref_name: &str, cand: &[f64], refv: &[f64], cand_label: &str) -> TCell {
    let d: Vec<f64> = cand.iter().zip(refv).map(|(a, b)| a - b).collect();
    let (p, _, _, _) = tost(&d, MARGIN);
    let (m, lo, hi) = ci90(&d);
    TCell { fam: fam.into(), src: src.into(), world: world.into(), metric: metric.into(), ref_name: ref_name.into(), mean_cand: summarize(cand).mean, mean_ref: summarize(refv).mean, diff: m, lo90: lo, hi90: hi, p, holm: 1.0, cand_label: cand_label.into() }
}

fn eval(sel: &Sel, held: &[u64], confirm: bool, out: &str) {
    let t0 = Instant::now();
    let g1 = if confirm { Fam::G1B } else { Fam::G1A };
    println!("== eval: seeds {}-{}  families F5 + {}  (frozen picks from tuning on seeds 1-8{})", held[0], held[held.len() - 1], g1.label(), if confirm { "; G1B = fresh corpus" } else { "" });
    println!("auto (pure) = {}   auto+bg = {}", sel.auto.label(), sel.auto_bg.label());
    let mut cells: Vec<TCell> = vec![];
    let mut tables: Vec<Value> = vec![];
    let mut cmp_rows: Vec<Value> = vec![];
    let mut sup: Vec<Value> = vec![];
    for fam in [Fam::F5, g1] {
        let tag = if fam == Fam::G1B { "g1A" } else { fam.label() }; // incumbents of corpus B are those tuned on corpus A
        for src in SRCS {
            let (_, _, incf1, incv) = sel.inc.iter().find(|x| x.0 == tag && x.1 == src.label()).unwrap().clone();
            let specs = base_worlds(fam, src);
            let meths = vec![incf1.clone(), incv.clone(), sel.auto.clone(), sel.auto_bg.clone(), Method { anchor: AnchorCfg::Oracle }, Method::leader(5, 0.6)];
            let mnames = ["inc-F1", "inc-vote", "auto", "auto-bg", "oracle", "default.6"];
            let ra = eval_anchor(&meths, &specs, held);
            let bases = bases_o();
            let mut arms: Vec<A> = bases.iter().map(|b| b.1.clone()).collect();
            let nb = arms.len();
            for m in &meths {
                arms.push(vote_arm(m));
            }
            let rr = eval_runs(&arms, &specs, held);
            println!("\n##### {} / {}   inc-F1 {}   inc-vote {}", fam.label(), src.label(), incf1.label(), incv.label());
            println!("{:<22} {:<10} | {:>6} {:>6} {:>6} {:>5} {:>6} {:>6} | {:>7} {:>7} {:>7} {:>7} {:>6}", "world", "arm", "P", "R", "F1", "frag", "spur", "contam", "H100", "H250", "H500", "H2000", "vote%");
            for (wi, sp) in specs.iter().enumerate() {
                for (mi, mn) in mnames.iter().enumerate() {
                    let a = &ra[mi][wi];
                    let g = |f: &dyn Fn(&Anc) -> f64| mean_of(a.iter().map(f));
                    let r = &rr[nb + mi][wi];
                    let h = |i: usize| mean_of(r.iter().map(|x| x.h[i]));
                    println!("{:<22} {:<10} | {:>6.3} {:>6.3} {:>6.3} {:>5.2} {:>6.3} {:>6.3} | {:>7.4} {:>7.4} {:>7.4} {:>7.4} {:>6.3}", sp.name, mn, g(&|x| x.p), g(&|x| x.r), g(&|x| x.f1), g(&|x| x.frag), g(&|x| x.spur), g(&|x| x.contam), h(0), h(1), h(2), h(3), mean_of(r.iter().map(|x| x.vote_share)));
                    cmp_rows.push(json!({"fam": fam.label(), "src": src.label(), "world": sp.name, "arm": mn, "p": g(&|x| x.p), "r": g(&|x| x.r), "f1": g(&|x| x.f1), "frag": g(&|x| x.frag), "spur": g(&|x| x.spur), "contam": g(&|x| x.contam),
                        "h100": h(0), "h250": h(1), "h500": h(2), "h2000": h(3), "vote_share": mean_of(r.iter().map(|x| x.vote_share)), "label": meths[mi].label()}));
                }
                for (bi, (bn, _)) in bases.iter().enumerate() {
                    let r = &rr[bi][wi];
                    let h = |i: usize| mean_of(r.iter().map(|x| x.h[i]));
                    println!("{:<22} {:<10} |                                                  | {:>7.4} {:>7.4} {:>7.4} {:>7.4}", sp.name, bn, h(0), h(1), h(2), h(3));
                    cmp_rows.push(json!({"fam": fam.label(), "src": src.label(), "world": sp.name, "arm": bn, "h100": h(0), "h250": h(1), "h500": h(2), "h2000": h(3)}));
                }
            }
            // TOST cells: auto vs the per-embedder tuned incumbent
            for (wi, sp) in specs.iter().enumerate() {
                for (ai, aname) in [(2usize, "auto"), (3usize, "auto-bg")] {
                    let lab = meths[ai].label();
                    if sp.treatment {
                        let c: Vec<f64> = ra[ai][wi].iter().map(|a| a.f1).collect();
                        let b: Vec<f64> = ra[0][wi].iter().map(|a| a.f1).collect();
                        let mut cell = tost_cell(fam.label(), src.label(), &sp.name, "anchorF1", "inc-F1", &c, &b, &lab);
                        cell.metric = format!("anchorF1[{aname}]");
                        cells.push(cell);
                    }
                    if sp.name.ends_with("ctl-unique") {
                        let c: Vec<f64> = ra[ai][wi].iter().map(|a| a.spur).collect();
                        let b: Vec<f64> = ra[0][wi].iter().map(|a| a.spur).collect();
                        let mut cell = tost_cell(fam.label(), src.label(), &sp.name, "spur", "inc-F1", &c, &b, &lab);
                        cell.metric = format!("spur[{aname}]");
                        cells.push(cell);
                    }
                    for (hi, hn) in [(2usize, "H500"), (3usize, "H2000")] {
                        let c: Vec<f64> = rr[nb + ai][wi].iter().map(|r| r.h[hi]).collect();
                        let b: Vec<f64> = rr[nb + 1][wi].iter().map(|r| r.h[hi]).collect();
                        let mut cell = tost_cell(fam.label(), src.label(), &sp.name, hn, "inc-vote", &c, &b, &lab);
                        cell.metric = format!("{hn}[{aname}]");
                        cells.push(cell);
                    }
                }
            }
            // SUPERIORITY vs the UNTUNED library default (k5, sim_min .6): what removing the tuning step is worth
            for (wi, sp) in specs.iter().enumerate() {
                for (ai, aname) in [(0usize, "inc-F1"), (1usize, "inc-vote"), (2usize, "auto"), (3usize, "auto-bg")] {
                    let mut push = |metric: String, c: Vec<f64>, b: Vec<f64>| {
                        let d = paired_diff(&c, &b);
                        sup.push(json!({"fam": fam.label(), "src": src.label(), "world": sp.name, "metric": metric, "arm": aname, "ref": "default.6", "diff": d.diff.mean, "ci95": [d.diff.lo, d.diff.hi], "p": paired_t_pvalue(&c, &b), "cand_mean": summarize(&c).mean, "ref_mean": summarize(&b).mean}));
                    };
                    if sp.treatment {
                        push("anchorF1".into(), ra[ai][wi].iter().map(|a| a.f1).collect(), ra[5][wi].iter().map(|a| a.f1).collect());
                    }
                    for (hi, hn) in [(2usize, "H500"), (3usize, "H2000")] {
                        push(hn.into(), rr[nb + ai][wi].iter().map(|r| r.h[hi]).collect(), rr[nb + 5][wi].iter().map(|r| r.h[hi]).collect());
                    }
                }
            }
            // vs best fair baseline for context (does the vote arm earn anything at all in this world?)
            for (wi, sp) in specs.iter().enumerate() {
                for (hi, hn) in [(2usize, "H500"), (3usize, "H2000")] {
                    let means: Vec<f64> = (0..nb).map(|b| mean_of(rr[b][wi].iter().map(|x| x.h[hi]))).collect();
                    let bi = (0..nb).max_by(|&a, &b| means[a].partial_cmp(&means[b]).unwrap()).unwrap();
                    let bv: Vec<f64> = rr[bi][wi].iter().map(|x| x.h[hi]).collect();
                    for (mi, mn) in mnames.iter().enumerate() {
                        let cv: Vec<f64> = rr[nb + mi][wi].iter().map(|x| x.h[hi]).collect();
                        let d = paired_diff(&cv, &bv);
                        tables.push(json!({"fam": fam.label(), "src": src.label(), "world": sp.name, "col": hn, "arm": mn, "best_baseline": bases[bi].0, "diff": d.diff.mean, "ci95": [d.diff.lo, d.diff.hi], "wins": d.wins, "n": cv.len()}));
                    }
                }
            }
        }
        clear_nb_cache();
    }
    // Holm over all TOST cells (family = every cell of this run), pure-auto cells and bg cells jointly
    let p: Vec<f64> = cells.iter().map(|c| c.p).collect();
    let adj = holm_adjust(&p);
    for (c, a) in cells.iter_mut().zip(&adj) {
        c.holm = *a;
    }
    println!("\n=============== TOST equivalence (margin +-{MARGIN}) of auto vs the per-embedder TUNED incumbent, Holm over {} cells ===============", cells.len());
    println!("{:<5} {:<5} {:<24} {:<18} | {:>8} {:>8} | {:>8} [{:>8},{:>8}] | {:>7} {:>7} EQUIV", "fam", "src", "world", "metric", "cand", "ref", "diff", "90%lo", "90%hi", "p", "holm");
    for c in &cells {
        println!("{:<5} {:<5} {:<24} {:<18} | {:>8.4} {:>8.4} | {:>+8.4} [{:>+8.4},{:>+8.4}] | {:>7.4} {:>7.4} {}", c.fam, c.src, c.world, c.metric, c.mean_cand, c.mean_ref, c.diff, c.lo90, c.hi90, c.p, c.holm, if c.holm < 0.05 { "YES" } else { "no" });
    }
    for pure in [true, false] {
        let sel: Vec<&TCell> = cells.iter().filter(|c| c.metric.contains("[auto]") == pure).collect();
        let eq = sel.iter().filter(|c| c.holm < 0.05).count();
        println!("{}: {} / {} cells Holm-equivalent within +-{MARGIN}", if pure { "pure auto" } else { "auto+bg" }, eq, sel.len());
        for fam in ["f5", g1.label()] {
            for m in ["anchorF1", "spur", "H500", "H2000"] {
                let ss: Vec<&&TCell> = sel.iter().filter(|c| c.fam == fam && c.metric.starts_with(m)).collect();
                println!("    {:<5} {:<9} {} / {} equivalent; mean diff {:+.4}, worst {:+.4}", fam, m, ss.iter().filter(|c| c.holm < 0.05).count(), ss.len(), ss.iter().map(|c| c.diff).sum::<f64>() / ss.len().max(1) as f64, ss.iter().map(|c| c.diff).fold(0.0f64, |a, b| if b.abs() > a.abs() { b } else { a }));
            }
        }
    }
    {
        let p: Vec<f64> = sup.iter().map(|s| s["p"].as_f64().unwrap()).collect();
        let adj = holm_adjust(&p);
        for (s, a) in sup.iter_mut().zip(&adj) {
            s["holm_p"] = json!(a);
        }
        println!("
=============== SUPERIORITY vs the untuned library default (k5, sim_min .6), paired 95% t-CI, Holm over {} cells ===============", sup.len());
        for arm in ["inc-F1", "inc-vote", "auto", "auto-bg"] {
            for metric in ["anchorF1", "H500", "H2000"] {
                let ss: Vec<&Value> = sup.iter().filter(|s| s["arm"] == arm && s["metric"] == metric).collect();
                let w = ss.iter().filter(|s| s["holm_p"].as_f64().unwrap() < 0.05 && s["diff"].as_f64().unwrap() > 0.0).count();
                let l = ss.iter().filter(|s| s["holm_p"].as_f64().unwrap() < 0.05 && s["diff"].as_f64().unwrap() < 0.0).count();
                println!("  {:<9} {:<9} cells {:>2}: Holm WIN {w} LOSS {l}; mean diff {:+.4}", arm, metric, ss.len(), ss.iter().map(|s| s["diff"].as_f64().unwrap()).sum::<f64>() / ss.len().max(1) as f64);
            }
        }
    }
    let doc = json!({
        "lab": "g10_calib/eval", "confirm": confirm, "seeds": [held[0], held[held.len() - 1]], "margin": MARGIN, "families": ["f5", g1.label()],
        "auto": method_json(&sel.auto), "auto_bg": method_json(&sel.auto_bg),
        "tost": cells.iter().map(|c| json!({"fam": c.fam, "src": c.src, "world": c.world, "metric": c.metric, "ref": c.ref_name, "cand_mean": c.mean_cand, "ref_mean": c.mean_ref, "diff": c.diff, "ci90": [c.lo90, c.hi90], "p_tost": c.p, "holm_p": c.holm, "equivalent": c.holm < 0.05, "cand": c.cand_label})).collect::<Vec<_>>(),
        "rows": cmp_rows, "vs_default": sup, "vs_best_baseline": tables, "elapsed_s": t0.elapsed().as_secs_f64()
    });
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("wrote {out} in {:.0}s", t0.elapsed().as_secs_f64());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let part = arg(&args, "--part", "diag");
    let confirm = args.iter().any(|a| a == "--confirm");
    let held = parse_range(&args, "--held", if confirm { (61, 90) } else { (31, 60) });
    let sel_path = arg(&args, "--selected", "results/g10_calib_selected.json");
    let _ = (&held, &sel_path);
    match part.as_str() {
        "diag" => { diag(); let s = load_sel(&sel_path); diag_bg(&s.inc); diag_trim(&s.inc) }
        "selfcheck" => selfcheck(),
        "tune" => tune(&sel_path),
        "eval" => {
            let sel = load_sel(&sel_path);
            let out = arg(&args, "--out", if confirm { "results/g10_calib_confirm61-90.json" } else { "results/g10_calib_eval.json" });
            eval(&sel, &held, confirm, &out)
        }
        p => panic!("unknown --part {p}"),
    }
}

/// background (random-pair) cosine statistics per (family, embedder) and the z that the tuned sim_min would correspond to
pub fn diag_bg(inc: &[(String, String, Method, Method)]) {
    println!("\nDIAG-BG: random-pair cosine mean/std (world = first tune world, seed 1..3) and z = (tuned sim_min - mean)/std");
    for fam in [Fam::F5, Fam::G1A] {
        for src in SRCS {
            let sp = world(fam, src, tune_names(fam)[0]);
            let mut w = Welford::default();
            let mut w1 = Welford::default();
            for seed in 1..=3u64 {
                let cw = CtxWorld::new(seed, &sp);
                let nb = cw.nbrs();
                for t in 1..cw.emb.len() {
                    nb.bg[t].iter().for_each(|&x| w.push(x as f64));
                    w1.push(nb.hits[t][0].1 as f64);
                }
            }
            let tuned: Vec<String> = inc
                .iter()
                .filter(|x| x.0 == fam.label() && x.1 == src.label())
                .flat_map(|x| [&x.2, &x.3])
                .map(|m| match &m.anchor { AnchorCfg::Leader { sim, .. } => format!("{sim:.2} (z={:.2})", (*sim as f64 - w.mean) / w.std()), _ => String::new() })
                .collect();
            println!("  {:<4} {:<5} bg mean {:.3} std {:.3} | top1 mean {:.3} std {:.3} | tuned sim F1/vote: {}", fam.label(), src.label(), w.mean, w.std(), w1.mean, w1.std(), tuned.join(" / "));
        }
    }
}

/// top-1 statistics with exact duplicates (cos >= 0.999) trimmed, and the kstd the tuned sim_min would need in those units
pub fn diag_trim(inc: &[(String, String, Method, Method)]) {
    println!("\nDIAG-TRIM: top-1 cosine statistics EXCLUDING exact duplicates (>= .999); kstd_needed = (mean - tuned sim)/std");
    for fam in [Fam::F5, Fam::G1A] {
        for src in SRCS {
            let sp = world(fam, src, tune_names(fam)[0]);
            let mut w = Welford::default();
            let mut dup = 0.0;
            let mut n = 0.0;
            for seed in 1..=3u64 {
                let cw = CtxWorld::new(seed, &sp);
                let nb = cw.nbrs();
                for t in 1..cw.emb.len() {
                    let x = nb.hits[t][0].1 as f64;
                    n += 1.0;
                    if x >= 0.999 { dup += 1.0 } else { w.push(x) }
                }
            }
            let tuned: Vec<String> = inc.iter().filter(|x| x.0 == fam.label() && x.1 == src.label()).flat_map(|x| [&x.2, &x.3]).map(|m| match &m.anchor { AnchorCfg::Leader { sim, .. } => format!("{sim:.2} (kstd={:.2})", (w.mean - *sim as f64) / w.std()), _ => String::new() }).collect();
            println!("  {:<4} {:<5} dup share {:.2} | non-dup top1 mean {:.3} std {:.3} | tuned F1/vote: {}", fam.label(), src.label(), dup / n, w.mean, w.std(), tuned.join(" / "));
        }
    }
}
