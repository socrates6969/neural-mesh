//! H7 creativity-widening lab (round 4). One binary, several experiments:
//!   --exp validate            engine k=1 must reproduce g4 `run` exactly on the frozen W4 worlds
//!   --exp bok  --worlds w4|n  (1) best-of-k + embedding dedup / archive selection vs best fair routing and vs W4 nbucb/nbreg
//!   --exp vs   --worlds w4|n|tag  (2) verbalized sampling (one call, k candidates, ASSUMED diversity/quality profile) + non-inferiority
//!   --exp gate --worlds w4v2|n    (3) widened nbucb/nbreg with gate fixes (RX counterfactual / AB periodic gate-off)
//!   --set tune|held|confirm|dev   seeds 1-8 | 31-60 | 61-90 | 101-130 (tuning ALWAYS on 1-8)
//! No real LLM: every number is a property of the simulated worlds in h7_common/hworld.rs (+ frozen g4 world).
mod h7_common;
use h7_common::engine::*;
use h7_common::hworld::*;
use h7_common::pols::*;
use h7_common::pols2::*;
use h7_common::world::{run, worlds, Pol};
use neural_mesh::stats::holm_adjust;
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;

type Build = Arc<dyn Fn(&HEnv) -> Box<dyn Pol> + Send + Sync>;

#[derive(Clone)]
struct WS {
    name: String,
    gk: &'static str,
    mk: Arc<dyn Fn(u64) -> HEnv + Send + Sync>,
}

fn w4_ws(tm: TagMode, val: Val, tag: &str, only: &[&str]) -> Vec<WS> {
    worlds()
        .into_iter()
        .filter(|c| !c.name.starts_with('R') && (only.is_empty() || only.iter().any(|o| c.name.starts_with(o))))
        .map(|c| {
            let nm = if tag.is_empty() { c.name.clone() } else { format!("{}{}", tag, c.name) };
            let (v, nm2) = (val.clone(), nm.clone());
            WS {
                name: nm,
                gk: "W4",
                mk: Arc::new(move |s| {
                    let mut e = w4_env(&c, s, v.clone(), tm);
                    e.name = nm2.clone();
                    e
                }),
            }
        })
        .collect()
}
fn n_ws(tm: TagMode, only: &[&str]) -> Vec<WS> {
    n_worlds()
        .into_iter()
        .filter(|c| only.is_empty() || only.iter().any(|o| c.name.starts_with(o)))
        .map(|c| {
            let g: &'static str = if c.name.contains("grid") { "grid" } else if c.name.contains("corpus") { "corpus" } else { "modal" };
            WS { name: c.name.clone(), gk: g, mk: Arc::new(move |s| n_env(&c, s, tm)) }
        })
        .collect()
}
fn v2_val() -> Val {
    Val::Win { floor: 0.1, window: 12, scale: 0.6, mean_mode: false, conc: 0.0 }
}
fn w4_val() -> Val {
    Val::Win { floor: 0.25, window: 8, scale: 1.0, mean_mode: false, conc: 0.0 }
}

#[derive(Clone)]
struct Arm {
    name: String,
    spec: PolSpec,
}
fn mkarm(name: &str, mode: Mode, build: Build) -> Arm {
    Arm { name: name.into(), spec: PolSpec { name: name.into(), mode, build } }
}
fn fair_build(g: Arc<HashMap<String, Kind>>) -> Build {
    Arc::new(move |e: &HEnv| Box::new(FairPol::new(&g[&e.name], &e.shell)) as Box<dyn Pol>)
}
fn x_build(spec: XSpec) -> Build {
    Arc::new(move |e: &HEnv| Box::new(NbX::new(&spec, e.nd, e.na, &e.tag_dom, &e.vcr)) as Box<dyn Pol>)
}
fn g4_build(spec: ArmSpec) -> Build {
    Arc::new(move |e: &HEnv| make_arm(&spec, &e.shell))
}
fn ucb_spec() -> XSpec {
    XSpec { reg: false, beta: 0.0, c: 0.05, gate: XG::Never }
}

struct Ctx {
    names: Vec<String>,
    gens: Vec<&'static str>,
    tune: Vec<Vec<HEnv>>,
    eval: Vec<Vec<HEnv>>,
    tseeds: Vec<u64>,
    eseeds: Vec<u64>,
}
fn ctx(ws: &[WS], set: &str) -> Ctx {
    let tseeds: Vec<u64> = (1..=8).collect();
    let eseeds: Vec<u64> = match set {
        "held" => (31..=60).collect(),
        "confirm" => (61..=90).collect(),
        "dev" => (101..=130).collect(),
        _ => (1..=8).collect(),
    };
    Ctx {
        names: ws.iter().map(|w| w.name.clone()).collect(),
        gens: ws.iter().map(|w| w.gk).collect(),
        tune: ws.iter().map(|w| tseeds.iter().map(|&s| (w.mk)(s)).collect()).collect(),
        eval: ws.iter().map(|w| eseeds.iter().map(|&s| (w.mk)(s)).collect()).collect(),
        tseeds,
        eseeds,
    }
}
fn pooled_tune_score(c: &Ctx, spec: &PolSpec) -> f64 {
    let mut tot = 0.0;
    for wi in 0..c.names.len() {
        tot += mean_overall(&eval_spec(spec, &c.tune[wi], &c.tseeds));
    }
    tot / c.names.len() as f64
}
fn tune_g_all(c: &Ctx) -> Arc<HashMap<String, Kind>> {
    let mut m = HashMap::new();
    for wi in 0..c.names.len() {
        let k = tune_g(&c.tune[wi], &c.tseeds);
        println!("[tune1-8] {:<36} G = {}", c.names[wi], k.label());
        m.insert(c.names[wi].clone(), k);
    }
    Arc::new(m)
}

/// pooled tuning of a selection config: returns best (label, Sel)
fn tune_sel(c: &Ctx, label: &str, grid: Vec<Sel>, mk: &(dyn Fn(Sel) -> PolSpec + Sync)) -> Sel {
    let sc: Vec<f64> = grid.par_iter().map(|s| pooled_tune_score(c, &mk(s.clone()))).collect();
    let mut b = 0;
    for i in 0..grid.len() {
        if sc[i] > sc[b] {
            b = i;
        }
    }
    println!("[tune1-8] {:<28} -> {:?}  pooled mean value {:.4} (grid {})", label, grid[b], sc[b], grid.len());
    grid[b].clone()
}
fn dd_grid() -> Vec<Sel> {
    let mut v = vec![];
    for &th in &[0.0, 0.15, 0.3, 0.5] {
        for &w in &[0.5, 0.75, 1.0] {
            v.push(Sel::Dedup { theta: th, w });
        }
    }
    v
}
fn arch_grid(cell: CellFn) -> Vec<Sel> {
    let mut v = vec![];
    for &w in &[0.5, 0.75, 1.0] {
        for &t in &[4.0, 8.0, 16.0] {
            v.push(Sel::Arch { w, t, cell });
        }
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let exp = arg("--exp").unwrap_or("bok".into());
    let wsel = arg("--worlds").unwrap_or("w4".into());
    let set = arg("--set").unwrap_or("held".into());
    let tag = arg("--tag").unwrap_or(format!("{exp}_{wsel}_{set}"));
    println!("H7 lab exp={exp} worlds={wsel} set={set}");

    if exp == "validate" {
        validate();
        return;
    }
    let ws: Vec<WS> = match wsel.as_str() {
        "w4" => w4_ws(TagMode::Truth, w4_val(), "", &[]),
        "w4v2" => w4_ws(TagMode::Truth, v2_val(), "V2-", &["C", "K", "M"]),
        // V3 = W4's own variant (floor .35, window 6, scale .8); V4 = defined by H7 before any DX arm ran (floor .15, window 16, scale .5: more unattainable than V2)
        "w4v3" => w4_ws(TagMode::Truth, Val::Win { floor: 0.35, window: 6, scale: 0.8, mean_mode: false, conc: 0.0 }, "V3-", &["C", "K", "M"]),
        "w4v4" => w4_ws(TagMode::Truth, Val::Win { floor: 0.15, window: 16, scale: 0.5, mean_mode: false, conc: 0.0 }, "V4-", &["C", "K", "M"]),
        "n" => n_ws(TagMode::Truth, &[]),
        "tag" => {
            let mut v = w4_ws(TagMode::AllTrue, w4_val(), "ALLTAG-", &["K"]);
            v.extend(w4_ws(TagMode::FalsePos(0.3), w4_val(), "FP30-", &["K"]));
            v.extend(n_ws(TagMode::AllTrue, &["NK"]).into_iter().map(|mut w| {
                w.name = format!("ALLTAG-{}", w.name);
                let mk = w.mk.clone();
                let nm = w.name.clone();
                w.mk = Arc::new(move |s| {
                    let mut e = mk(s);
                    e.name = nm.clone();
                    e
                });
                w
            }));
            v
        }
        _ => panic!("--worlds w4|w4v2|w4v3|w4v4|n|tag"),
    };
    let c = ctx(&ws, &set);
    let g = tune_g_all(&c);
    let gfair = fair_build(g.clone());
    let mut arms: Vec<Arm> = vec![
        mkarm("G (best fair, 1 call)", Mode::single(), gfair.clone()),
        mkarm("U plain-ucb (1 call)", Mode::single(), x_build(ucb_spec())),
    ];
    let mut refs: Vec<&str> = vec!["G (best fair, 1 call)"];
    // W4's pooled-tuned arms, frozen from the board entry (not re-tuned here)
    let w4arms = [
        ("W4 nbucb(b=.8)|learned", ArmSpec { mech: Mech::Nb(0.8, 0.05), base: Base::Pts(0.99), gate: GateKind::Learned }),
        ("W4 nbreg|learned", ArmSpec { mech: Mech::Reg(0.05), base: Base::Pts(0.99), gate: GateKind::Learned }),
        ("W4 nbucb(b=.4)|declared", ArmSpec { mech: Mech::Nb(0.4, 0.05), base: Base::Pts(0.99), gate: GateKind::Declared }),
    ];
    let judge = 0.05;
    let mut cellfns = vec![("sign", CellFn::Sign)];
    if wsel == "n" {
        cellfns.push(("grid8", CellFn::Grid(8)));
    }
    match exp.as_str() {
        "bok" => {
            for (n, s) in &w4arms {
                arms.push(mkarm(n, Mode::single(), g4_build(s.clone())));
            }
            let bm = |k: usize, sel: Sel, base: &Build| PolSpec { name: "t".into(), mode: Mode { k, vs: None, sel, gate: EGate::Tag, judge_sd: judge }, build: base.clone() };
            for &k in &[2usize, 3, 4] {
                arms.push(mkarm(&format!("BoK-q k={k} (multi-agent, quality-only select)"), Mode { k, vs: None, sel: Sel::QOnly, gate: EGate::Tag, judge_sd: judge }, gfair.clone()));
                let sel = tune_sel(&c, &format!("BoK-dedup k={k}"), dd_grid(), &|s| bm(k, s, &gfair));
                arms.push(mkarm(&format!("BoK-dedup k={k}"), Mode { k, vs: None, sel, gate: EGate::Tag, judge_sd: judge }, gfair.clone()));
            }
            for (cn, cf) in &cellfns {
                let sel = tune_sel(&c, &format!("BoK-archive({cn}) k=3"), arch_grid(*cf), &|s| bm(3, s, &gfair));
                arms.push(mkarm(&format!("BoK-archive({cn}) k=3"), Mode { k: 3, vs: None, sel, gate: EGate::Tag, judge_sd: judge }, gfair.clone()));
            }
            let ub = x_build(ucb_spec());
            let sel = tune_sel(&c, "BoK-dedup k=3 on plain-ucb", dd_grid(), &|s| bm(3, s, &ub));
            arms.push(mkarm("BoK-dedup k=3 on plain-ucb", Mode { k: 3, vs: None, sel: sel.clone(), gate: EGate::Tag, judge_sd: judge }, ub.clone()));
            // POST-HOC (added after the first held-out look at exp=bok/w4): STACK = W4 declared nbucb(b=.4) as the candidate generator router + dedup selection
            let nb_decl = g4_build(w4arms[2].1.clone());
            for &k in &[2usize, 3] {
                let sel = tune_sel(&c, &format!("STACK nbucb|declared + dedup k={k}"), dd_grid(), &|s| bm(k, s, &nb_decl));
                arms.push(mkarm(&format!("STACK nbucb(b=.4)|decl + dedup k={k}"), Mode { k, vs: None, sel, gate: EGate::Tag, judge_sd: judge }, nb_decl.clone()));
            }
            // judge sensitivity for k=3 dedup (selection params frozen from the k=3 tune above)
            let sel3 = arms.iter().find(|a| a.name == "BoK-dedup k=3").unwrap().spec.mode.sel.clone();
            for &js in &[0.0, 0.1, 0.2, -1.0] {
                arms.push(mkarm(&format!("BoK-dedup k=3 judge_sd={js}"), Mode { k: 3, vs: None, sel: sel3.clone(), gate: EGate::Tag, judge_sd: js }, gfair.clone()));
            }
            // same-agent k=3 native sampling (sigma 0 offset => only the world's own output noise .08 / .15 differs), priced k calls
            arms.push(mkarm("same-agent k=3 native sampling (dedup)", Mode { k: 3, vs: Some(VsProf { sigma: 0.0, kappa: 0.0, rho: 1.0 }), sel: sel3.clone(), gate: EGate::Tag, judge_sd: judge }, gfair.clone()));
            // dedup selector with tags replaced by the learned gate (no human tag)
            arms.push(mkarm("BoK-dedup k=3 |learned gate", Mode { k: 3, vs: None, sel: sel3.clone(), gate: EGate::Learned, judge_sd: judge }, gfair.clone()));
            for &k in &[3usize, 8] {
                arms.push(mkarm(&format!("ORACLE clairvoyant best-of-{k} by true value (ceiling, {k} calls)"), Mode { k, vs: None, sel: Sel::OracleVal, gate: EGate::Tag, judge_sd: 0.0 }, gfair.clone()));
            }
            refs.push("BoK-q k=3 (multi-agent, quality-only select)");
            refs.push("W4 nbucb(b=.8)|learned");
            refs.push("W4 nbreg|learned");
        }
        "vs" => {
            let bm = |p: VsProf, sel: Sel, gate: EGate| PolSpec { name: "t".into(), mode: Mode { k: 4, vs: Some(p), sel, gate, judge_sd: judge }, build: gfair.clone() };
            let def = VsProf { sigma: 0.35, kappa: 0.15, rho: 0.3 };
            let sel = tune_sel(&c, "VS default-profile k=4 selection", dd_grid(), &|s| bm(def, s, EGate::Tag));
            arms.push(mkarm("BoK-q k=4 same-agent VS profile, quality-only", Mode { k: 4, vs: Some(def), sel: Sel::QOnly, gate: EGate::Tag, judge_sd: judge }, gfair.clone()));
            for &s in &[0.15, 0.35, 0.6] {
                for &kp in &[0.0, 0.15, 0.4, 0.8, 1.5] {
                    let p = VsProf { sigma: s, kappa: kp, rho: 0.3 };
                    let star = if s == 0.35 && kp == 0.15 { " [DEFAULT]" } else { "" };
                    arms.push(mkarm(&format!("VS k=4 sig={s} kap={kp} rho=0.3|tag{star}"), Mode { k: 4, vs: Some(p), sel: sel.clone(), gate: EGate::Tag, judge_sd: judge }, gfair.clone()));
                }
            }
            for &rho in &[0.0, 1.0] {
                arms.push(mkarm(&format!("VS k=4 sig=0.35 kap=0.15 rho={rho}|tag"), Mode { k: 4, vs: Some(VsProf { sigma: 0.35, kappa: 0.15, rho }), sel: sel.clone(), gate: EGate::Tag, judge_sd: judge }, gfair.clone()));
            }
            // judge-free variants: no verifier call; selection = dedup/novelty only (qhat constant)
            for &kp in &[0.0, 0.15, 0.4, 0.8, 1.5] {
                let p = VsProf { sigma: 0.35, kappa: kp, rho: 0.3 };
                arms.push(mkarm(&format!("VS k=4 sig=0.35 kap={kp} rho=0.3|tag JUDGE-FREE"), Mode { k: 4, vs: Some(p), sel: sel.clone(), gate: EGate::Tag, judge_sd: -1.0 }, gfair.clone()));
            }
            arms.push(mkarm("VS k=4 DEFAULT JUDGE-FREE |learned gate", Mode { k: 4, vs: Some(def), sel: sel.clone(), gate: EGate::Learned, judge_sd: -1.0 }, gfair.clone()));
            arms.push(mkarm("VS k=4 DEFAULT |learned gate", Mode { k: 4, vs: Some(def), sel: sel.clone(), gate: EGate::Learned, judge_sd: judge }, gfair.clone()));
            arms.push(mkarm("VS k=4 DEFAULT |always on", Mode { k: 4, vs: Some(def), sel: sel.clone(), gate: EGate::Always, judge_sd: judge }, gfair.clone()));
            for (n, s) in &w4arms[..2] {
                arms.push(mkarm(n, Mode::single(), g4_build(s.clone())));
            }
            refs.push("W4 nbreg|learned");
        }
        "gate" => {
            let xs = |reg: bool, beta: f64, gate: XG| XSpec { reg, beta, c: 0.05, gate };
            for (n, s) in &w4arms {
                arms.push(mkarm(n, Mode::single(), g4_build(s.clone())));
            }
            for (reg, beta) in [(false, 0.8), (true, 0.0)] {
                for gate in [XG::Learned, XG::RX(0.10), XG::AB(0.10), XG::DX(0.25), XG::DXT(0.25), XG::Tag] {
                    let sp = xs(reg, beta, gate);
                    arms.push(mkarm(&format!("X {}", sp.label()), Mode::single(), x_build(sp.clone())));
                }
            }
            // re-tuned (pooled over this world set, tune seeds 1-8) nbucb strength/UCB constant for the DX and declared gates
            for gate in [XG::DX(0.25), XG::Tag] {
                let mut best: Option<(f64, XSpec)> = None;
                for &beta in &[0.1, 0.2, 0.4, 0.8] {
                    for &cc in &[0.05, 0.15] {
                        let sp = XSpec { reg: false, beta, c: cc, gate };
                        let ps = PolSpec { name: "t".into(), mode: Mode::single(), build: x_build(sp.clone()) };
                        let sc = pooled_tune_score(&c, &ps);
                        if best.as_ref().map_or(true, |b| sc > b.0) {
                            best = Some((sc, sp));
                        }
                    }
                }
                let (sc, sp) = best.unwrap();
                println!("[tune1-8] retuned nbucb|{:?} -> {}  pooled mean value {:.4}", gate, sp.label(), sc);
                arms.push(mkarm(&format!("X RETUNED {}", sp.label()), Mode::single(), x_build(sp)));
            }
            refs.push("U plain-ucb (1 call)");
            refs.push("W4 nbreg|learned");
        }
        _ => panic!("--exp validate|bok|vs|gate"),
    }

    // ---------------- evaluate
    let t0 = std::time::Instant::now();
    let res: Vec<Vec<Vec<HOut>>> = arms.iter().map(|a| (0..c.names.len()).map(|wi| eval_spec(&a.spec, &c.eval[wi], &c.eseeds)).collect()).collect();
    println!("evaluated {} arms x {} worlds x {} seeds in {:.1}s", arms.len(), c.names.len(), c.eseeds.len(), t0.elapsed().as_secs_f64());
    report(&tag, &c, &arms, &res, &refs);
}

fn validate() {
    let tseeds: Vec<u64> = (31..=40).collect();
    let mut maxd: f64 = 0.0;
    for cfg in worlds().into_iter().filter(|c| !c.name.starts_with('R')) {
        for &s in &tseeds {
            let e = w4_env(&cfg, s, w4_val(), TagMode::Truth);
            for k in [Kind::Keyword, Kind::Pts(0.99), Kind::Warm(10.0, EvK::Mean)] {
                let mut p1 = FairPol::new(&k, &e.shell);
                let o1 = run(&e.shell, &mut p1, s);
                let mut p2 = FairPol::new(&k, &e.shell);
                let o2 = run_h(&e, &mut p2, &Mode::single(), s);
                maxd = maxd.max((o1.overall - o2.overall).abs()).max((o1.creative.is_nan() as u8 as f64) * 0.0);
                if !o1.creative.is_nan() {
                    maxd = maxd.max((o1.creative - o2.creative).abs());
                }
            }
            for spec in [ArmSpec { mech: Mech::Nb(0.8, 0.05), base: Base::Pts(0.99), gate: GateKind::Learned }, ArmSpec { mech: Mech::Reg(0.05), base: Base::Pts(0.99), gate: GateKind::Learned }] {
                let mut p1 = make_arm(&spec, &e.shell);
                let o1 = run(&e.shell, p1.as_mut(), s);
                let mut p2 = make_arm(&spec, &e.shell);
                let o2 = run_h(&e, p2.as_mut(), &Mode::single(), s);
                maxd = maxd.max((o1.overall - o2.overall).abs());
            }
            // NbX reimplementation must equal g4 NbUcb (declared) / NbReg (learned)
            let xs = XSpec { reg: false, beta: 0.4, c: 0.05, gate: XG::Tag };
            let spec = ArmSpec { mech: Mech::Nb(0.4, 0.05), base: Base::Pts(0.99), gate: GateKind::Declared };
            let mut p1 = make_arm(&spec, &e.shell);
            let o1 = run(&e.shell, p1.as_mut(), s);
            let mut p2 = NbX::new(&xs, e.nd, e.na, &e.tag_dom, &e.vcr);
            let o2 = run_h(&e, &mut p2, &Mode::single(), s);
            maxd = maxd.max((o1.overall - o2.overall).abs());
            let xs = XSpec { reg: true, beta: 0.0, c: 0.05, gate: XG::Learned };
            let spec = ArmSpec { mech: Mech::Reg(0.05), base: Base::Pts(0.99), gate: GateKind::Learned };
            let mut p1 = make_arm(&spec, &e.shell);
            let o1 = run(&e.shell, p1.as_mut(), s);
            let mut p2 = NbX::new(&xs, e.nd, e.na, &e.tag_dom, &e.vcr);
            let o2 = run_h(&e, &mut p2, &Mode::single(), s);
            maxd = maxd.max((o1.overall - o2.overall).abs());
        }
    }
    println!("VALIDATE: max |g4 run - h7 engine(k=1)| over 8 W4 worlds x 10 seeds x (3 fair + 2 W4 arms + 2 NbX reimplementations) = {maxd:.3e}");
    println!("{}", if maxd < 1e-9 { "VALIDATE OK (exact reproduction)" } else { "VALIDATE FAILED" });
}

fn report(tag: &str, c: &Ctx, arms: &[Arm], res: &[Vec<Vec<HOut>>], refs: &[&str]) {
    let nw = c.names.len();
    let aidx = |n: &str| arms.iter().position(|a| a.name == n);
    println!("\n=== mean value per world: creative | correctness | overall | gen-calls/creative-task | judge/creative-task | multi-on[cr/cor] ===");
    for wi in 0..nw {
        println!("-- {} ({})", c.names[wi], c.gens[wi]);
        for (ai, a) in arms.iter().enumerate() {
            let r = &res[ai][wi];
            let m = |f: fn(&HOut) -> f64| mean(&col(r, f));
            println!(
                "   {:<62.62} {:.4} | {:.4} | {:.4} | calls {:.2} judge {:.2} | on {:.2}/{:.2}",
                a.name,
                m(f_creative),
                m(f_correct),
                m(f_overall),
                m(f_calls),
                m(f_judge),
                mean(&r.iter().map(|x| x.on_cr).collect::<Vec<_>>()),
                mean(&r.iter().map(|x| x.on_cor).collect::<Vec<_>>())
            );
        }
    }
    #[derive(Clone, serde::Serialize)]
    struct Rec {
        world: String,
        arm: String,
        reference: String,
        metric: String,
        t: T,
        holm: f64,
        extra_calls: f64,
        breakeven: f64,
    }
    let mut recs: Vec<Rec> = vec![];
    for r in refs {
        let ri = match aidx(r) {
            Some(i) => i,
            None => continue,
        };
        let mut fam: Vec<Rec> = vec![];
        for (ai, a) in arms.iter().enumerate() {
            if ai == ri {
                continue;
            }
            for wi in 0..nw {
                for (metric, f) in [("creative", f_creative as fn(&HOut) -> f64), ("correct", f_correct)] {
                    let x = col(&res[ai][wi], f);
                    let y = col(&res[ri][wi], f);
                    if x.iter().any(|v| v.is_nan()) || y.iter().any(|v| v.is_nan()) {
                        continue;
                    }
                    let t = test(&x, &y);
                    let dc = mean(&col(&res[ai][wi], f_calls)) - mean(&col(&res[ri][wi], f_calls));
                    let dj = mean(&col(&res[ai][wi], f_judge)) - mean(&col(&res[ri][wi], f_judge));
                    let ex = if metric == "creative" { dc + dj } else { 0.0 };
                    let be = if ex > 1e-9 { t.diff / ex } else { f64::NAN };
                    fam.push(Rec { world: c.names[wi].clone(), arm: a.name.clone(), reference: r.to_string(), metric: metric.into(), t, holm: 1.0, extra_calls: ex, breakeven: be });
                }
            }
        }
        let adj = holm_adjust(&fam.iter().map(|x| x.t.p).collect::<Vec<_>>());
        for (i, x) in fam.iter_mut().enumerate() {
            x.holm = adj[i];
        }
        recs.extend(fam);
    }
    for r in refs {
        println!("\n=== DIFFS vs [{}] : creative-valued subset  (diff [95% CI] w/l, Holm; break-even = value gain per extra gen+judge call per creative task) ===", r);
        for (ai, a) in arms.iter().enumerate() {
            if a.name == *r {
                continue;
            }
            println!("## {}", a.name);
            for wi in 0..nw {
                if let Some(x) = recs.iter().find(|x| x.reference == *r && x.arm == a.name && x.world == c.names[wi] && x.metric == "creative") {
                    println!(
                        "   creative {:<34} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.3}{}  extra calls {:+.2} breakeven/call {:+.4}",
                        x.world, x.t.diff, x.t.lo, x.t.hi, x.t.wins, x.t.losses, x.holm,
                        if x.t.diff > 0.0 && x.holm < 0.05 { " WIN" } else if x.t.diff < 0.0 && x.holm < 0.05 { " LOSS" } else { "" },
                        x.extra_calls, x.breakeven
                    );
                }
                if let Some(x) = recs.iter().find(|x| x.reference == *r && x.arm == a.name && x.world == c.names[wi] && x.metric == "correct") {
                    let ident = x.t.diff == 0.0 && x.t.lo == 0.0 && x.t.hi == 0.0;
                    println!(
                        "   correct  {:<34} {:+.4} [{:+.4},{:+.4}] 90%lo {:+.4} TOST.01 p {:.3}{}{}{}",
                        x.world, x.t.diff, x.t.lo, x.t.hi, x.t.lo90, x.t.tost01,
                        if x.t.tost01 < 0.05 { " EQ" } else { "" },
                        if x.t.lo90 > -0.01 { " NONINF" } else { " NOT-NONINF" },
                        if ident { " (identical by construction)" } else { "" }
                    );
                }
            }
        }
    }
    // summary table vs G
    println!("\n=== SUMMARY vs G: creative diff per world (* Holm win, ! Holm loss) | mean extra calls | correctness: TOST.01 equivalent / non-inferior counts ===");
    print!("{:<62}", "arm");
    for wi in 0..nw {
        print!(" {:>9.9}", c.names[wi]);
    }
    println!(" | calls | eq/ni/n");
    let gname = refs[0];
    for a in arms.iter() {
        if a.name == gname {
            continue;
        }
        print!("{:<62.62}", a.name);
        let (mut eq, mut ni, mut nn) = (0, 0, 0);
        let mut cs = vec![];
        for wi in 0..nw {
            if let Some(x) = recs.iter().find(|x| x.reference == gname && x.arm == a.name && x.world == c.names[wi] && x.metric == "creative") {
                let mark = if x.t.diff > 0.0 && x.holm < 0.05 { '*' } else if x.t.diff < 0.0 && x.holm < 0.05 { '!' } else { ' ' };
                print!(" {:>+8.4}{}", x.t.diff, mark);
                cs.push(x.extra_calls);
            } else {
                print!(" {:>9}", "-");
            }
            if let Some(x) = recs.iter().find(|x| x.reference == gname && x.arm == a.name && x.world == c.names[wi] && x.metric == "correct") {
                nn += 1;
                eq += (x.t.tost01 < 0.05) as usize;
                ni += (x.t.lo90 > -0.01) as usize;
            }
        }
        println!(" | {:+.2} | {}/{}/{}", mean(&cs), eq, ni, nn);
    }
    let jr: Vec<_> = recs.iter().map(|x| json!({"world": x.world, "arm": x.arm, "ref": x.reference, "metric": x.metric, "t": x.t, "holm": x.holm, "extra_calls": x.extra_calls, "breakeven": x.breakeven})).collect();
    let jm: Vec<_> = (0..nw)
        .map(|wi| {
            json!({"world": c.names[wi], "gen": c.gens[wi], "means": arms.iter().enumerate().map(|(ai, a)| json!({"arm": a.name, "creative": mean(&col(&res[ai][wi], f_creative)), "correct": mean(&col(&res[ai][wi], f_correct)), "overall": mean(&col(&res[ai][wi], f_overall)), "calls_cr": mean(&col(&res[ai][wi], f_calls)), "judge_cr": mean(&col(&res[ai][wi], f_judge))})).collect::<Vec<_>>()})
        })
        .collect();
    let out = format!("results/h7_{tag}.json");
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"tag": tag, "seeds": [c.eseeds[0], c.eseeds[c.eseeds.len() - 1]], "means": jm, "tests": jr})).unwrap()).unwrap();
    println!("wrote {out}");
}
