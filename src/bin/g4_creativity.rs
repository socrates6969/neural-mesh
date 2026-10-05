//! G4 creativity / diversity routing lab (R&D round 3, worker W4).
//!
//! World value function is FROZEN in g4_common/world.rs (see results/g4_world_frozen.md).
//! Protocol: fair baselines tuned PER WORLD on seeds 1-8; diversity arms tuned ONCE pooled over all
//! 9 worlds on seeds 1-8; evaluation on `--set held` (31-60) or `--set confirm` (61-90) with the same
//! frozen (re-derived, deterministic) configs. Paired 95% t-CI, Holm, TOST for harmlessness.
//!
//! Usage: cargo run --release --offline --bin g4_creativity -- --set held --out results/g4_creativity_held.json

mod g4_common;
use g4_common::pols::*;
use g4_common::world::*;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize, t_two_sided_p};
use rayon::prelude::*;
use serde_json::json;

const WORLD_SRC: &str = include_str!("g4_common/world.rs");

fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[derive(Clone, Debug)]
enum PSpec {
    Fair(Kind),
    Arm(ArmSpec),
    Oracle,
}
impl PSpec {
    fn label(&self) -> String {
        match self {
            PSpec::Fair(k) => k.label(),
            PSpec::Arm(a) => a.label(),
            PSpec::Oracle => "ORACLE-ISH".into(),
        }
    }
}
fn build(s: &PSpec, w: &World) -> Box<dyn Pol> {
    match s {
        PSpec::Fair(k) => Box::new(FairPol::new(k, w)),
        PSpec::Arm(a) => make_arm(a, w),
        PSpec::Oracle => Box::new(Oracle::new(w)),
    }
}

fn eval(spec: &PSpec, worlds: &[World], seeds: &[u64]) -> Vec<RunOut> {
    worlds
        .par_iter()
        .zip(seeds.par_iter())
        .map(|(w, &s)| {
            let mut p = build(spec, w);
            run(w, p.as_mut(), s)
        })
        .collect()
}

fn mk_worlds(cfg: &WorldCfg, seeds: &[u64]) -> Vec<World> {
    seeds.iter().map(|&s| World::new(s, cfg.clone())).collect()
}

fn mean_overall(v: &[RunOut]) -> f64 {
    v.iter().map(|r| r.overall).sum::<f64>() / v.len() as f64
}

#[derive(Clone, Debug, serde::Serialize)]
struct T {
    diff: f64,
    lo: f64,
    hi: f64,
    lo90: f64,
    hi90: f64,
    p: f64,
    wins: usize,
    losses: usize,
    tost01: f64,
    tost02: f64,
    n: usize,
}

fn p_greater(t: f64, df: usize) -> f64 {
    // P(T > t)
    let p2 = t_two_sided_p(t.abs(), df);
    if t > 0.0 { 0.5 * p2 } else { 1.0 - 0.5 * p2 }
}

fn t_crit_90(df: usize) -> f64 {
    let (mut lo, mut hi) = (0.5f64, 10.0f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if t_two_sided_p(mid, df) > 0.10 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn tost_p(m: f64, se: f64, df: usize, delta: f64) -> f64 {
    if se < 1e-15 {
        return if m.abs() < delta { 0.0 } else { 1.0 };
    }
    let p_lo = p_greater((m + delta) / se, df); // H0: mean <= -delta
    let p_hi = p_greater(-(m - delta) / se, df); // H0: mean >= +delta
    p_lo.max(p_hi)
}

fn test(a: &[f64], b: &[f64]) -> T {
    let pd = paired_diff(a, b);
    let n = a.len();
    let d = &pd.diff;
    let se = d.sd / (n as f64).sqrt();
    let t90 = t_crit_90(n - 1);
    T {
        diff: d.mean,
        lo: d.lo,
        hi: d.hi,
        lo90: d.mean - t90 * se,
        hi90: d.mean + t90 * se,
        p: paired_t_pvalue(a, b),
        wins: pd.wins,
        losses: pd.losses,
        tost01: tost_p(d.mean, se, n - 1, 0.01),
        tost02: tost_p(d.mean, se, n - 1, 0.02),
        n,
    }
}

fn col(v: &[RunOut], f: fn(&RunOut) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}
fn f_overall(r: &RunOut) -> f64 {
    r.overall
}
fn f_creative(r: &RunOut) -> f64 {
    r.creative
}
fn f_correct(r: &RunOut) -> f64 {
    r.correct
}
fn f_nov(r: &RunOut) -> f64 {
    r.nov_creative
}

fn fair_grid() -> Vec<Kind> {
    let mut v = vec![];
    for &g in &[0.9, 0.95, 0.99, 1.0] {
        for &e in &[0.02, 0.05, 0.1] {
            v.push(Kind::Eps(g, e));
        }
    }
    for &g in &[0.9, 0.95, 0.99, 1.0] {
        v.push(Kind::Pts(g));
    }
    for &k in &[3.0, 10.0, 30.0, 100.0] {
        for &e in &[EvK::Set, EvK::Mean, EvK::FailSet] {
            v.push(Kind::Warm(k, e));
        }
    }
    v.push(Kind::HebR);
    v.push(Kind::Keyword);
    v.push(Kind::GuardedSafe);
    v
}

fn arm_grid() -> Vec<ArmSpec> {
    let bases = [Base::Warm(10.0), Base::Pts(0.99)];
    let gates = [GateKind::None, GateKind::Declared, GateKind::Learned];
    let mut v = vec![];
    for g in gates {
        for b in &bases {
            for &m in &[1usize, 2, 3, 4] {
                v.push(ArmSpec { mech: Mech::Mask(m), base: b.clone(), gate: g });
            }
            for &k in &[2usize, 3, 4, 5, 6] {
                v.push(ArmSpec { mech: Mech::Nov(k), base: b.clone(), gate: g });
            }
            for &k in &[2usize, 3, 4, 5] {
                for &t in &[4.0, 8.0, 16.0] {
                    v.push(ArmSpec { mech: Mech::Arch(k, t), base: b.clone(), gate: g });
                }
            }
        }
        for &beta in &[0.1, 0.2, 0.4, 0.8] {
            for &c in &[0.05, 0.15] {
                v.push(ArmSpec { mech: Mech::Nb(beta, c), base: Base::Pts(0.99), gate: g });
            }
        }
    }
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let set = arg("--set").unwrap_or("held".into());
    let out = arg("--out").unwrap_or(format!("results/g4_creativity_{set}.json"));
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let eval_seeds: Vec<u64> = match set.as_str() {
        "held" => (31..=60).collect(),
        "confirm" => (61..=90).collect(),
        "tune" => (1..=8).collect(),
        "dev" => (101..=130).collect(),
        _ => panic!("--set held|confirm|tune"),
    };
    println!("G4 creativity lab. world.rs fnv64={:016x} len={} (sha256 in results/g4_world_frozen.md)", fnv(WORLD_SRC), WORLD_SRC.len());
    println!("eval set '{set}': seeds {}..{}   tune seeds 1-8", eval_seeds[0], eval_seeds[eval_seeds.len() - 1]);

    let cfgs = worlds();
    let tune_w: Vec<Vec<World>> = cfgs.iter().map(|c| mk_worlds(c, &tune_seeds)).collect();
    let eval_w: Vec<Vec<World>> = cfgs.iter().map(|c| mk_worlds(c, &eval_seeds)).collect();

    // ---------------- 1. tune fair baselines per world (family-best by overall value on seeds 1-8)
    let fg = fair_grid();
    let families = ["keyword", "eps-greedy", "pooled-thompson", "hebbian-robust", "warm(hebr)", "guarded-safe"];
    // chosen[w][fam] = Kind ; best[w] = family index of best fair baseline
    let mut chosen: Vec<Vec<Kind>> = vec![];
    let mut best_fam: Vec<usize> = vec![];
    let mut all_scores: Vec<Vec<f64>> = vec![];
    for (wi, c) in cfgs.iter().enumerate() {
        let scores: Vec<f64> = fg.par_iter().map(|k| mean_overall(&eval(&PSpec::Fair(k.clone()), &tune_w[wi], &tune_seeds))).collect();
        let mut row = vec![];
        let mut fam_score = vec![];
        for f in families {
            let (bi, bs) = fg.iter().enumerate().filter(|(_, k)| k.family() == f).map(|(i, _)| (i, scores[i])).fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
            row.push(fg[bi].clone());
            fam_score.push(bs);
        }
        let bf = fam_score.iter().enumerate().fold((0, f64::NEG_INFINITY), |a, (i, &s)| if s > a.1 { (i, s) } else { a }).0;
        println!("[tune1-8] {:<34} best fair baseline = {:<44} ({:.4})", c.name, row[bf].label(), fam_score[bf]);
        chosen.push(row);
        best_fam.push(bf);
        all_scores.push(scores);
    }
    // pooled-tuned fair baselines: ONE config per family across all worlds (same protocol as the diversity arms)
    let mut pooled_chosen: Vec<Kind> = vec![];
    for f in families {
        let (bi, _) = fg
            .iter()
            .enumerate()
            .filter(|(_, k)| k.family() == f)
            .map(|(i, _)| (i, all_scores.iter().map(|s| s[i]).sum::<f64>() / cfgs.len() as f64))
            .fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a });
        pooled_chosen.push(fg[bi].clone());
    }
    // best pooled family per world (by that world's tune-seed score of the pooled config)
    let pooled_best_fam: Vec<usize> = (0..cfgs.len())
        .map(|wi| {
            (0..families.len())
                .map(|fi| {
                    let idx = fg.iter().position(|k| k.label() == pooled_chosen[fi].label()).unwrap();
                    (fi, all_scores[wi][idx])
                })
                .fold((0, f64::NEG_INFINITY), |a, b| if b.1 > a.1 { b } else { a })
                .0
        })
        .collect();
    println!("[tune1-8] pooled-tuned fair baselines (one config for all worlds): {:?}", pooled_chosen.iter().map(|k| k.label()).collect::<Vec<_>>());

    // ---------------- 2. tune diversity arms pooled over worlds
    let ag = arm_grid();
    println!("[tune1-8] pooled tuning of {} arm configs over {} worlds x 8 seeds", ag.len(), cfgs.len());
    let arm_scores: Vec<f64> = ag
        .par_iter()
        .map(|a| {
            let mut tot = 0.0;
            for wi in 0..cfgs.len() {
                tot += mean_overall(&eval(&PSpec::Arm(a.clone()), &tune_w[wi], &tune_seeds));
            }
            tot / cfgs.len() as f64
        })
        .collect();
    let mut fam_best: Vec<(String, usize)> = vec![];
    for (i, a) in ag.iter().enumerate() {
        let f = a.family();
        match fam_best.iter_mut().find(|x| x.0 == f) {
            Some(x) => {
                if arm_scores[i] > arm_scores[x.1] {
                    x.1 = i;
                }
            }
            None => fam_best.push((f, i)),
        }
    }
    let mut arms: Vec<ArmSpec> = fam_best.iter().map(|x| ag[x.1].clone()).collect();
    // post-hoc, UNTUNED arms (c fixed a priori = 0.05, copied from the nbucb grid): regression-calibrated novelty
    arms.push(ArmSpec { mech: Mech::Reg(0.05), base: Base::Pts(0.99), gate: GateKind::Declared });
    arms.push(ArmSpec { mech: Mech::Reg(0.05), base: Base::Pts(0.99), gate: GateKind::Learned });
    for (f, i) in &fam_best {
        println!("   pooled-tuned {:<24} -> {:<52} pooled mean value {:.4}", f, ag[*i].label(), arm_scores[*i]);
    }

    // ---------------- 3. evaluate
    // policy list per world: [6 fair (chosen)] + [arms] + [base-alone for each arm] + oracle
    struct Row {
        name: String,
        kind: &'static str, // fair | pfair | arm | base | ucbref | oracle
        res: Vec<Vec<RunOut>>, // [world][seed]
    }
    let mut rows: Vec<Row> = vec![];
    let nw = cfgs.len();
    let mut fair_idx = vec![];
    for (fi, f) in families.iter().enumerate() {
        let res = (0..nw).map(|wi| eval(&PSpec::Fair(chosen[wi][fi].clone()), &eval_w[wi], &eval_seeds)).collect();
        fair_idx.push(rows.len());
        rows.push(Row { name: format!("{f} [per-world tuned]"), kind: "fair", res });
    }
    let mut pfair_idx = vec![];
    for (fi, f) in families.iter().enumerate() {
        let res = (0..nw).map(|wi| eval(&PSpec::Fair(pooled_chosen[fi].clone()), &eval_w[wi], &eval_seeds)).collect();
        pfair_idx.push(rows.len());
        rows.push(Row { name: format!("{f} [pooled-tuned {}]", pooled_chosen[fi].label()), kind: "pfair", res });
    }
    let mut arm_idx = vec![];
    for a in &arms {
        let res = (0..nw).map(|wi| eval(&PSpec::Arm(a.clone()), &eval_w[wi], &eval_seeds)).collect();
        arm_idx.push(rows.len());
        rows.push(Row { name: a.label(), kind: "arm", res });
    }
    let mut base_idx = vec![];
    for a in &arms {
        let (spec, bl) = match a.mech {
            Mech::Nb(_, c) => (PSpec::Arm(ArmSpec { mech: Mech::Nb(0.0, c), base: a.base.clone(), gate: GateKind::None }), format!("plain-ucb(c={c})")),
            Mech::Reg(c) => (PSpec::Arm(ArmSpec { mech: Mech::Reg(c), base: a.base.clone(), gate: GateKind::Off }), format!("nbreg-off(c={c})")),
            _ => (PSpec::Fair(a.base.kind()), a.base.kind().label()),
        };
        let res = (0..nw).map(|wi| eval(&spec, &eval_w[wi], &eval_seeds)).collect();
        base_idx.push(rows.len());
        rows.push(Row { name: format!("base-alone[{}] of {}", bl, a.label()), kind: "base", res });
    }
    // extra strong reference: domain-keyed plain UCB (own learner, beta=0), c in {.05,.15} pooled-tuned
    let ucb_c = {
        let mut best = (f64::NEG_INFINITY, 0.05);
        for &c in &[0.05, 0.15] {
            let sp = PSpec::Arm(ArmSpec { mech: Mech::Nb(0.0, c), base: Base::Pts(0.99), gate: GateKind::None });
            let sc: f64 = (0..nw).map(|wi| mean_overall(&eval(&sp, &tune_w[wi], &tune_seeds))).sum::<f64>() / nw as f64;
            if sc > best.0 {
                best = (sc, c);
            }
        }
        best.1
    };
    let ucb_idx = rows.len();
    {
        let sp = PSpec::Arm(ArmSpec { mech: Mech::Nb(0.0, ucb_c), base: Base::Pts(0.99), gate: GateKind::None });
        let res = (0..nw).map(|wi| eval(&sp, &eval_w[wi], &eval_seeds)).collect();
        rows.push(Row { name: format!("plain-ucb(c={ucb_c}) [domain-keyed own learner, EXTRA-STRONG ref, not in task baseline list]"), kind: "ucbref", res });
    }
    let orc_idx = rows.len();
    {
        let res = (0..nw).map(|wi| eval(&PSpec::Oracle, &eval_w[wi], &eval_seeds)).collect();
        rows.push(Row { name: "ORACLE-ISH".into(), kind: "oracle", res });
    }

    // ---------------- 4. report: per-world means
    println!("\n=== mean value per world ({set}) : overall | creative-subset | correctness-subset | creative novelty ===");
    for (wi, c) in cfgs.iter().enumerate() {
        println!("-- {} (creative domains {}/{}) best per-world-tuned fair = {} ; best pooled-tuned fair = {}", c.name, c.n_creative(), c.domains, families[best_fam[wi]], families[pooled_best_fam[wi]]);
        for r in rows.iter().filter(|r| r.kind != "base") {
            let m = |f: fn(&RunOut) -> f64| summarize(&col(&r.res[wi], f)).mean;
            let gate = r.res[wi][0].diag.clone();
            let gs = if gate.len() == 2 { format!(" gateon[cr {:.2} / cor {:.2}]", gate[0], gate[1]) } else { String::new() };
            println!("   {:<62.62} {:.4} | {:.4} | {:.4} | nov {:.3}{}", r.name, m(f_overall), m(f_creative), m(f_correct), m(f_nov), gs);
        }
    }

    // ---------------- 5. paired tests
    #[derive(Clone, serde::Serialize)]
    struct Rec {
        world: String,
        arm: String,
        kind: String, // win|harm (vs per-world-tuned best fair), winP|harmP (vs pooled-tuned best fair), winU|harmU (vs plain-ucb), layerC|layerK (vs own base), overall, oracle
        metric: String,
        reference: String,
        t: T,
        holm: f64,
        tost01_holm: f64,
    }
    let mut recs: Vec<Rec> = vec![];
    for (wi, c) in cfgs.iter().enumerate() {
        let bfr = &rows[fair_idx[best_fam[wi]]];
        let pfr = &rows[pfair_idx[pooled_best_fam[wi]]];
        for ai in 0..arms.len() {
            let arm = &rows[arm_idx[ai]];
            let base = &rows[base_idx[ai]];
            let ucb = &rows[ucb_idx];
            let mut push = |kind: &str, metric: &str, f: fn(&RunOut) -> f64, r: &Row, refname: &str| {
                let a = col(&arm.res[wi], f);
                let b = col(&r.res[wi], f);
                if a.iter().any(|x| x.is_nan()) || b.iter().any(|x| x.is_nan()) {
                    return;
                }
                recs.push(Rec { world: c.name.clone(), arm: arm.name.clone(), kind: kind.into(), metric: metric.into(), reference: refname.into(), t: test(&a, &b), holm: 1.0, tost01_holm: 1.0 });
            };
            push("overall", "overall", f_overall, bfr, &bfr.name);
            push("win", "creative", f_creative, bfr, &bfr.name);
            push("harm", "correct", f_correct, bfr, &bfr.name);
            push("winP", "creative", f_creative, pfr, &pfr.name);
            push("harmP", "correct", f_correct, pfr, &pfr.name);
            push("winU", "creative", f_creative, ucb, "plain-ucb");
            push("harmU", "correct", f_correct, ucb, "plain-ucb");
            push("layerC", "creative", f_creative, base, &base.name);
            push("layerK", "correct", f_correct, base, &base.name);
        }
        let orc = &rows[orc_idx];
        for (metric, f) in [("creative", f_creative as fn(&RunOut) -> f64), ("overall", f_overall)] {
            let a = col(&orc.res[wi], f);
            let b = col(&bfr.res[wi], f);
            if a.iter().any(|x| x.is_nan()) {
                continue;
            }
            recs.push(Rec { world: c.name.clone(), arm: orc.name.clone(), kind: "oracle".into(), metric: metric.into(), reference: bfr.name.clone(), t: test(&a, &b), holm: 1.0, tost01_holm: 1.0 });
        }
    }
    // Holm families: (win+harm), (winP+harmP), (winU+harmU), (layerC+layerK) two-sided; TOST Holm per harm kind
    for (k1, k2) in [("win", "harm"), ("winP", "harmP"), ("winU", "harmU"), ("layerC", "layerK")] {
        let idx: Vec<usize> = recs.iter().enumerate().filter(|(_, r)| r.kind == k1 || r.kind == k2).map(|(i, _)| i).collect();
        let adj = holm_adjust(&idx.iter().map(|&i| recs[i].t.p).collect::<Vec<_>>());
        for (k, &i) in idx.iter().enumerate() {
            recs[i].holm = adj[k];
        }
        let hidx: Vec<usize> = recs.iter().enumerate().filter(|(_, r)| r.kind == k2).map(|(i, _)| i).collect();
        let hadj = holm_adjust(&hidx.iter().map(|&i| recs[i].t.tost01).collect::<Vec<_>>());
        for (k, &i) in hidx.iter().enumerate() {
            recs[i].tost01_holm = hadj[k];
        }
        println!("Holm family {k1}+{k2}: {} two-sided tests; TOST family {}", idx.len(), hidx.len());
    }

    // ---------------- 6. verdict table per arm
    let cn: Vec<String> = cfgs.iter().map(|c| c.name.clone()).collect();
    let find = |recs: &Vec<Rec>, arm: &str, world: &str, kind: &str| -> Option<usize> { recs.iter().position(|r| r.arm == arm && r.world == world && r.kind == kind) };
    let mut verdicts = vec![];
    println!("\n=== per-arm verdicts ({set}). Refs: [G] best of the task's fair list, tuned PER WORLD (generous) | [P] best fair, pooled-tuned like the arms | [U] plain domain-keyed UCB (extra-strong learner) | [L] the arm's OWN base without the diversity layer ===");
    for (ai, _) in arms.iter().enumerate() {
        let arm = &rows[arm_idx[ai]];
        println!("\n## {}", arm.name);
        for (tag, kw, kh, detail) in [("G", "win", "harm", true), ("P", "winP", "harmP", false), ("U", "winU", "harmU", false), ("L", "layerC", "layerK", true)] {
            let (mut cw, mut cn_n, mut mw, mut ek, mut ek_h, mut hn, mut sl, mut r1, mut ni) = (0, 0, 0, 0, 0, 0, 0, f64::NAN, 0);
            for name in &cn {
                if let Some(i) = find(&recs, &arm.name, name, kw) {
                    let r = &recs[i];
                    let is_c = name.starts_with('C');
                    let sig = r.t.diff > 0.0 && r.holm < 0.05;
                    if is_c {
                        cn_n += 1;
                        if sig {
                            cw += 1;
                        }
                    } else if name.starts_with('M') && sig {
                        mw += 1;
                    }
                    if name.starts_with('R') {
                        r1 = r.t.diff;
                    }
                    if detail {
                        println!("   [{tag}] creative {:<34} {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm {:.3}", name, r.t.diff, r.t.lo, r.t.hi, r.t.wins, r.t.losses, r.holm);
                    }
                }
                if let Some(i) = find(&recs, &arm.name, name, kh) {
                    let r = &recs[i];
                    hn += 1;
                    let ok = r.t.tost01 < 0.05;
                    let okh = r.tost01_holm < 0.05;
                    ek += ok as usize;
                    ni += (r.t.lo90 > -0.01) as usize;
                    ek_h += okh as usize;
                    if r.t.hi < 0.0 {
                        sl += 1;
                    }
                    if detail {
                        println!("   [{tag}] correct  {:<34} {:+.4} [{:+.4},{:+.4}] 90%[{:+.4},{:+.4}] TOST.01 p {:.3}{} (holm {:.3}{}) TOST.02 p {:.3}", name, r.t.diff, r.t.lo, r.t.hi, r.t.lo90, r.t.hi90, r.t.tost01, if ok { " EQ" } else { "" }, r.tost01_holm, if okh { " EQ" } else { "" }, r.t.tost02);
                    }
                }
            }
            println!("   [{tag}] SUMMARY: indep creative worlds won (Holm) {cw}/{cn_n} ; mixed creative-subsets won {mw}/2 ; R1 diff {:+.4} ; correctness equivalence TOST.01 {ek}/{hn} (Holm {ek_h}/{hn}) ; NON-INFERIOR(lo90>-.01) {ni}/{hn} ; sig correctness losses {sl}", r1);
            verdicts.push(json!({"arm": arm.name, "ref": tag, "indep_creative_wins_holm": cw, "indep_creative_n": cn_n, "mixed_creative_wins": mw, "r1_diff": r1, "harm_equiv_tost01": ek, "harm_equiv_tost01_holm": ek_h, "harm_tests": hn, "noninferior_01": ni, "sig_correctness_losses": sl}));
        }
    }

    let jr: Vec<_> = recs
        .iter()
        .enumerate()
        .map(|(_i, r)| json!({"world": r.world, "arm": r.arm, "kind": r.kind, "metric": r.metric, "reference": r.reference, "t": r.t, "holm": r.holm, "tost01_holm": r.tost01_holm}))
        .collect();
    let jw: Vec<_> = cfgs
        .iter()
        .enumerate()
        .map(|(wi, c)| {
            json!({"world": c.name, "creative_domains": c.n_creative(), "domains": c.domains, "best_fair_family": families[best_fam[wi]], "best_pooled_fair_family": families[pooled_best_fam[wi]], "tuned_fair": chosen[wi].iter().map(|k| k.label()).collect::<Vec<_>>(),
            "means": rows.iter().map(|r| json!({"policy": r.name, "kind": r.kind, "overall": summarize(&col(&r.res[wi], f_overall)).mean, "creative": summarize(&col(&r.res[wi], f_creative)).mean, "correct": summarize(&col(&r.res[wi], f_correct)).mean, "nov": summarize(&col(&r.res[wi], f_nov)).mean, "gate_on": r.res[wi][0].diag})).collect::<Vec<_>>()})
        })
        .collect();
    let doc = json!({
        "lab": "g4_creativity", "set": set, "eval_seeds": [eval_seeds[0], eval_seeds[eval_seeds.len()-1]], "tune_seeds": [1, 8],
        "world_fnv64": format!("{:016x}", fnv(WORLD_SRC)),
        "pooled_tuned_arms": arms.iter().map(|a| a.label()).collect::<Vec<_>>(),
        "worlds": jw, "tests": jr, "verdicts": verdicts,
        "protocol": "fair baselines tuned per world on seeds 1-8; diversity arms tuned once pooled across all 9 worlds on seeds 1-8; paired 95% t-CI; Holm over win+harm tests; TOST delta=0.01/0.02 for correctness equivalence"
    });
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!("\nwrote {out}");
}
