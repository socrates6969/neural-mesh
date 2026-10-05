//! G4 robustness / misspecification check. Re-uses the policies of g4_common/pols.rs UNCHANGED, with the
//! arm configs frozen from the main run (pooled tuning on the ORIGINAL worlds, seeds 1-8), against value
//! functions whose parameters/novelty reduction differ from the policies' internal novelty model
//! (policy: mean of 3 nearest distances, window 10, scale 1).
//!   V1 weak-novelty / mean-distance : floor .5, window 5, scale 1.2, novelty = MEAN distance to window
//!   V2 strong-novelty / wide window : floor .1, window 12, scale .6, novelty = MIN distance to window
//! Fair baselines are re-tuned per world on seeds 1-8 (generous). Arms are NOT re-tuned.
//!
//! Usage: g4_robust --variant v1|v2 --set held|confirm

#[path = "g4_common/pols.rs"]
mod pols;
#[path = "g4_common/world_alt.rs"]
mod world;

use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use pols::*;
use rayon::prelude::*;
use serde_json::json;
use world::*;

fn eval(mk: &(dyn Fn(&World) -> Box<dyn Pol> + Sync), worlds: &[World], seeds: &[u64]) -> Vec<RunOut> {
    worlds
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(w, &s)| {
            let mut p = mk(w);
            run(w, p.as_mut(), s)
        })
        .collect()
}

fn mean_overall(v: &[RunOut]) -> f64 {
    v.iter().map(|r| r.overall).sum::<f64>() / v.len() as f64
}

fn col(v: &[RunOut], f: fn(&RunOut) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let variant = arg("--variant").unwrap_or("v1".into());
    let set = arg("--set").unwrap_or("held".into());
    let a = match variant.as_str() {
        "v1" => Alt { floor: 0.5, window: 5, scale: 1.2, mean_mode: true },
        "v2" => Alt { floor: 0.1, window: 12, scale: 0.6, mean_mode: false },
        // V3 defined AFTER v1/v2 were seen and after the nbreg arm was designed, BEFORE it was run: the clean out-of-sample variant
        "v3" => Alt { floor: 0.35, window: 6, scale: 0.8, mean_mode: false },
        _ => panic!("--variant v1|v2|v3"),
    };
    set_alt(a);
    println!("G4 robustness {variant} {:?}  eval set {set}", alt());
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let seeds: Vec<u64> = match set.as_str() { "held" => (31..=60).collect(), "dev" => (101..=130).collect(), _ => (61..=90).collect() };
    let cfgs = worlds();
    let tw: Vec<Vec<World>> = cfgs.iter().map(|c| tune_seeds.iter().map(|&s| World::new(s, c.clone())).collect()).collect();
    let ew: Vec<Vec<World>> = cfgs.iter().map(|c| seeds.iter().map(|&s| World::new(s, c.clone())).collect()).collect();

    // fair grid (same as main)
    let mut fg: Vec<Kind> = vec![];
    for &g in &[0.9, 0.95, 0.99, 1.0] {
        for &e in &[0.02, 0.05, 0.1] {
            fg.push(Kind::Eps(g, e));
        }
    }
    for &g in &[0.9, 0.95, 0.99, 1.0] {
        fg.push(Kind::Pts(g));
    }
    for &k in &[3.0, 10.0, 30.0, 100.0] {
        for &e in &[EvK::Set, EvK::Mean, EvK::FailSet] {
            fg.push(Kind::Warm(k, e));
        }
    }
    fg.push(Kind::HebR);
    fg.push(Kind::Keyword);
    fg.push(Kind::GuardedSafe);

    // frozen arm configs from the main run
    let arms: Vec<ArmSpec> = vec![
        ArmSpec { mech: Mech::Nb(0.4, 0.05), base: Base::Pts(0.99), gate: GateKind::Declared },
        ArmSpec { mech: Mech::Nb(0.8, 0.05), base: Base::Pts(0.99), gate: GateKind::Learned },
        ArmSpec { mech: Mech::Nov(2), base: Base::Warm(10.0), gate: GateKind::Declared },
        ArmSpec { mech: Mech::Nov(2), base: Base::Warm(10.0), gate: GateKind::Learned },
        ArmSpec { mech: Mech::Reg(0.05), base: Base::Pts(0.99), gate: GateKind::Declared },
        ArmSpec { mech: Mech::Reg(0.05), base: Base::Pts(0.99), gate: GateKind::Learned },
    ];
    let ucb = ArmSpec { mech: Mech::Nb(0.0, 0.05), base: Base::Pts(0.99), gate: GateKind::None };

    struct Rec {
        world: String,
        arm: String,
        kind: &'static str,
        diff: f64,
        lo: f64,
        hi: f64,
        p: f64,
        wins: usize,
        losses: usize,
    }
    let mut recs: Vec<Rec> = vec![];
    for (wi, c) in cfgs.iter().enumerate() {
        // best fair, per-world tuned on seeds 1-8
        let sc: Vec<f64> = fg.par_iter().map(|k| mean_overall(&eval(&|w: &World| Box::new(FairPol::new(k, w)) as Box<dyn Pol>, &tw[wi], &tune_seeds))).collect();
        let bi = (0..fg.len()).fold(0, |b, i| if sc[i] > sc[b] { i } else { b });
        let bk = fg[bi].clone();
        let best = eval(&|w: &World| Box::new(FairPol::new(&bk, w)) as Box<dyn Pol>, &ew[wi], &seeds);
        let uc = eval(&|w: &World| make_arm(&ucb, w), &ew[wi], &seeds);
        println!("-- {:<34} best fair (tuned) = {:<40} value overall {:.4}  plain-ucb {:.4}", c.name, bk.label(), summarize(&col(&best, |r| r.overall)).mean, summarize(&col(&uc, |r| r.overall)).mean);
        for arm in &arms {
            let r = eval(&|w: &World| make_arm(arm, w), &ew[wi], &seeds);
            for (kind, refr, f, metric) in [
                ("winG", &best, (|r: &RunOut| r.creative) as fn(&RunOut) -> f64, "creative"),
                ("harmG", &best, |r: &RunOut| r.correct, "correct"),
                ("winU", &uc, |r: &RunOut| r.creative, "creative"),
                ("harmU", &uc, |r: &RunOut| r.correct, "correct"),
            ] {
                let (x, y) = (col(&r, f), col(refr, f));
                if x.iter().any(|v| v.is_nan()) {
                    continue;
                }
                let pd = paired_diff(&x, &y);
                recs.push(Rec { world: c.name.clone(), arm: arm.label(), kind, diff: pd.diff.mean, lo: pd.diff.lo, hi: pd.diff.hi, p: paired_t_pvalue(&x, &y), wins: pd.wins, losses: pd.losses });
                let _ = metric;
            }
        }
    }
    let adj = holm_adjust(&recs.iter().map(|r| r.p).collect::<Vec<_>>());
    println!("\n{} tests, Holm over all", recs.len());
    for arm in &arms {
        println!("\n## {}", arm.label());
        for kind in ["winG", "winU", "harmG", "harmU"] {
            for (i, r) in recs.iter().enumerate().filter(|(_, r)| r.arm == arm.label() && r.kind == kind) {
                println!("   {:<6} {:<34} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.3}", r.kind, r.world, r.diff, r.lo, r.hi, r.wins, r.losses, adj[i]);
            }
        }
    }
    let out = format!("results/g4_robust_{variant}_{set}.json");
    let jr: Vec<_> = recs.iter().enumerate().map(|(i, r)| json!({"world": r.world, "arm": r.arm, "kind": r.kind, "diff": r.diff, "lo": r.lo, "hi": r.hi, "p": r.p, "holm": adj[i], "wins": r.wins, "losses": r.losses})).collect();
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"variant": variant, "set": set, "alt": format!("{:?}", alt()), "tests": jr})).unwrap()).unwrap();
    println!("wrote {out}");
}
