//! R2 cognitive-skill routing lab. Mechanisms (all opt-in, in src/cognition.rs):
//!  M1 quality-diversity niche archive (agent x approach) for creative-domain build routing
//!  M2 novelty bonus: creative-only (candidate) vs ALL domains (control) vs worlds where novelty is worthless
//!  M3 generator->critic sequencing, critic chosen by learned reliability vs random/fixed/none/fair-bandit
//!  M4 domain-hierarchical routing vs flat
//!  S  composition ("cog-full") vs fair baselines {keyword, eps-greedy, pooled-ts, hebbian-robust, warm(hebbian-robust)}
//! Protocol: EVERY learner (baselines and mechanisms) tuned per world on seeds 1-8 only; evaluated on
//! held-out seeds 31-60; paired 95% t-CI; Holm over the whole family of tests (all worlds, all hypotheses).
//!
//! Usage: cargo run --release --bin r2_cognition [-- --out results/r2_cognition.json]

use neural_mesh::cognition::*;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use rayon::prelude::*;
use serde_json::json;

#[derive(Clone, Copy, Debug, Default)]
struct M {
    overall: f64,
    creative: f64,
    correctness: f64,
    early: f64,
    nov: f64,
    ent: f64,
}

fn metrics(r: &RunOut) -> M {
    M { overall: r.overall(), creative: r.creative(), correctness: r.correctness(), early: r.early(300), nov: r.creative_novelty, ent: r.creative_entropy }
}

fn eval(spec: &PolicySpec, cfg: &WorldCfg, seeds: &[u64]) -> Vec<M> {
    seeds.par_iter().map(|&s| metrics(&run_spec(spec, cfg, s))).collect()
}

fn col(v: &[M], f: fn(&M) -> f64) -> Vec<f64> {
    v.iter().map(f).collect()
}

fn mean_of(v: &[M], f: fn(&M) -> f64) -> f64 {
    summarize(&col(v, f)).mean
}

/// pick the best spec of `cands` by tune-seed objective
fn tune<T: Clone + std::fmt::Debug + Sync + Send>(label: &str, cands: Vec<(T, PolicySpec)>, cfg: &WorldCfg, seeds: &[u64], f: fn(&M) -> f64) -> (T, PolicySpec, f64) {
    let n = cands.len();
    let scored: Vec<(f64, usize)> = cands.par_iter().enumerate().map(|(i, (_, s))| (mean_of(&eval(s, cfg, seeds), f), i)).collect();
    let (best, idx) = scored.iter().cloned().fold((f64::NEG_INFINITY, 0), |a, b| if b.0 > a.0 { b } else { a });
    println!("   [tune 1-8] {:<28} {:>3} configs best {:.4} -> {:?}", label, n, best, cands[idx].0);
    (cands[idx].0.clone(), cands[idx].1.clone(), best)
}

fn worlds() -> Vec<WorldCfg> {
    let w1 = WorldCfg::base("A-specialist-multimodal");
    let w2 = WorldCfg { structure: Structure::Generalist, peak: Peak::Single, noise: 0.15, drift_at: Some(700), ..WorldCfg::base("B-generalist-singlepeak-noisy-drift") };
    let w3 = WorldCfg { lam: 0.0, ..WorldCfg::base("C-novelty-worthless") };
    let w4 = WorldCfg {
        structure: Structure::Generalist,
        peak: Peak::Single,
        critic_equal: true,
        labels_informative: false,
        approach_matters: false,
        ..WorldCfg::base("D-rigged-check(generalist,equal-critics,uninformative-labels)")
    };
    vec![w1, w2, w3, w4]
}

#[derive(Clone)]
struct Test {
    world: String,
    hypothesis: String,
    cand: String,
    reference: String,
    metric: &'static str,
    mean_cand: f64,
    mean_ref: f64,
    diff: f64,
    lo: f64,
    hi: f64,
    wins: usize,
    losses: usize,
    p: f64,
    holm: f64,
}

fn mk_test(world: &str, hyp: &str, cn: &str, c: &[M], rn: &str, r: &[M], metric: &'static str, f: fn(&M) -> f64) -> Test {
    let (a, b) = (col(c, f), col(r, f));
    let d = paired_diff(&a, &b);
    Test {
        world: world.into(),
        hypothesis: hyp.into(),
        cand: cn.into(),
        reference: rn.into(),
        metric,
        mean_cand: summarize(&a).mean,
        mean_ref: summarize(&b).mean,
        diff: d.diff.mean,
        lo: d.diff.lo,
        hi: d.diff.hi,
        wins: d.wins,
        losses: d.losses,
        p: paired_t_pvalue(&a, &b),
        holm: 1.0,
    }
}

fn spec_with(name: &str, base: &Kind, build: BuildSpec, critic: CriticSpec) -> PolicySpec {
    PolicySpec { name: name.into(), plan: base.clone(), plan_hier: false, build, critic }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).cloned().unwrap_or("results/r2_cognition.json".into());
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = (31..=60).collect();
    println!("R2 cognition lab: tune seeds 1-8 (per world), held-out 31-60, {} tasks/run", WorldCfg::base("x").n_tasks);

    let mut all_tests: Vec<Test> = vec![];
    let mut world_json = vec![];

    for cfg in worlds() {
        println!("\n================ WORLD {} ================", cfg.name);
        let wn = cfg.name.clone();
        // ---------- 1. tune fair baselines ----------
        let mut c = vec![];
        for &g in &[0.95, 0.99, 1.0] {
            for &e in &[0.02, 0.05, 0.1] {
                c.push((Kind::Eps(g, e), PolicySpec::fair(&Kind::Eps(g, e))));
            }
        }
        let (eps_k, _, eps_o) = tune("eps-greedy", c, &cfg, &tune_seeds, |m| m.overall);
        let c = [0.9, 0.95, 0.99, 1.0].iter().map(|&g| (Kind::Pts(g), PolicySpec::fair(&Kind::Pts(g)))).collect();
        let (pts_k, _, pts_o) = tune("pooled-ts", c, &cfg, &tune_seeds, |m| m.overall);
        let mut c = vec![];
        for &k in &[3.0, 10.0, 30.0, 100.0] {
            for &e in &[EvK::Set, EvK::Mean, EvK::FailSet] {
                c.push((Kind::WarmHebR(k, e), PolicySpec::fair(&Kind::WarmHebR(k, e))));
            }
        }
        let (warm_k, _, warm_o) = tune("warm(hebbian-robust)", c, &cfg, &tune_seeds, |m| m.overall);
        let heb_o = mean_of(&eval(&PolicySpec::fair(&Kind::HebR), &cfg, &tune_seeds), |m| m.overall);
        // base learner for ablations = best learner on tune seeds
        let mut cands = vec![(eps_o, eps_k.clone()), (pts_o, pts_k.clone()), (heb_o, Kind::HebR), (warm_o, warm_k.clone())];
        cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let base = cands[0].1.clone();
        println!("   base learner B = {}", base.label());

        // ---------- 2. tune mechanisms (tune seeds only) ----------
        let ref_b = PolicySpec::fair(&base);
        let mut grid = vec![];
        for &lam in &[0.3, 0.6] {
            for &window in &[10usize, 20] {
                for &alpha in &[0.05, 0.2] {
                    for &ucb in &[0.05, 0.2] {
                        for &eps in &[0.02, 0.1] {
                            grid.push(QdConfig { lam, window, alpha, ucb, eps, ..QdConfig::default_niche() });
                        }
                    }
                }
            }
        }
        let qd_build = |q: &QdConfig| BuildSpec::Qd { cfg: q.clone(), other: base.clone(), hier: false };
        let critic_b = CriticSpec::Fair { kind: base.clone(), hier: false };
        let c = grid.iter().map(|q| (q.clone(), spec_with("qd", &base, qd_build(q), critic_b.clone()))).collect();
        let (qd_niche, _, _) = tune("QD niche+novelty(creative)", c, &cfg, &tune_seeds, |m| m.creative);
        let c = grid
            .iter()
            .map(|q| {
                let q = QdConfig { niche: false, ..q.clone() };
                (q.clone(), spec_with("nov", &base, qd_build(&q), critic_b.clone()))
            })
            .collect();
        let (qd_agent, _, _) = tune("novelty-only(agent arms)", c, &cfg, &tune_seeds, |m| m.creative);
        let mut c = vec![];
        for &alpha in &[0.05, 0.2] {
            for &ucb in &[0.05, 0.2] {
                for &eps in &[0.02, 0.1] {
                    let q = QdConfig { scope: NoveltyScope::Off, alpha, ucb, eps, ..QdConfig::default_niche() };
                    c.push((q.clone(), spec_with("niche0", &base, qd_build(&q), critic_b.clone())));
                }
            }
        }
        let (qd_off, _, _) = tune("niche archive, no novelty", c, &cfg, &tune_seeds, |m| m.creative);
        let mut c = vec![];
        for (mode, gammas, gates) in [(CriticMode::Final, vec![1.0, 0.98], vec![0.0]), (CriticMode::Verdict, vec![1.0, 0.98], vec![0.0, 0.6])] {
            for &gamma in &gammas {
                for &gate in &gates {
                    for &prior_k in &[0.0, 10.0, 30.0] {
                        let l = LcConfig { mode, gamma, gate, prior_k };
                        c.push((l.clone(), spec_with("lc", &base, BuildSpec::Fair { kind: base.clone(), hier: false }, CriticSpec::Learned(l))));
                    }
                }
            }
        }
        let c_verd0: Vec<_> = c.iter().filter(|(l, _)| l.mode == CriticMode::Verdict && l.prior_k == 0.0).cloned().collect();
        let c_final: Vec<_> = c.iter().filter(|(l, _)| l.mode == CriticMode::Final).cloned().collect();
        let c_verd: Vec<_> = c.iter().filter(|(l, _)| l.mode == CriticMode::Verdict).cloned().collect();
        let (lc_final, _, lcf_o) = tune("learned critic (final)", c_final, &cfg, &tune_seeds, |m| m.correctness);
        let (lc_verd, _, lcv_o) = tune("learned critic (verdict)", c_verd, &cfg, &tune_seeds, |m| m.correctness);
        let (lc_verd0, _, _) = tune("learned critic (verdict, no prior)", c_verd0, &cfg, &tune_seeds, |m| m.correctness);
        let lc_best = if lcv_o >= lcf_o { lc_verd.clone() } else { lc_final.clone() };

        // ---------- 3. held-out specs ----------
        let a = cfg.agents;
        let _ = a;
        let mut specs: Vec<(String, PolicySpec)> = vec![];
        let fair_list: Vec<(&str, Kind)> = vec![("keyword", Kind::Keyword), ("eps-greedy", eps_k.clone()), ("pooled-ts", pts_k.clone()), ("hebbian-robust", Kind::HebR), ("warm(hebbian-robust)", warm_k.clone())];
        for (n, k) in &fair_list {
            specs.push((n.to_string(), PolicySpec::fair(k)));
        }
        specs.push(("REF B (flat fair, best learner)".into(), ref_b.clone()));
        specs.push(("M1 QD-niche + novelty(creative)".into(), spec_with("m1", &base, qd_build(&qd_niche), critic_b.clone())));
        specs.push(("M1b niche archive, no novelty".into(), spec_with("m1b", &base, qd_build(&qd_off), critic_b.clone())));
        specs.push((
            "M1c QD-niche novelty on ALL domains (control)".into(),
            spec_with("m1c", &base, qd_build(&QdConfig { scope: NoveltyScope::All, ..qd_niche.clone() }), critic_b.clone()),
        ));
        specs.push(("M2a novelty bonus, agent arms, creative-only".into(), spec_with("m2a", &base, qd_build(&qd_agent), critic_b.clone())));
        specs.push((
            "M2b novelty bonus, agent arms, ALL domains (control)".into(),
            spec_with("m2b", &base, qd_build(&QdConfig { scope: NoveltyScope::All, ..qd_agent.clone() }), critic_b.clone()),
        ));
        specs.push((
            "M2c agent arms, QD learner, NO novelty".into(),
            spec_with("m2c", &base, qd_build(&QdConfig { scope: NoveltyScope::Off, ..qd_agent.clone() }), critic_b.clone()),
        ));
        let bf = BuildSpec::Fair { kind: base.clone(), hier: false };
        specs.push(("M3 critic: none".into(), spec_with("c-none", &base, bf.clone(), CriticSpec::None)));
        specs.push(("M3 critic: random".into(), spec_with("c-rand", &base, bf.clone(), CriticSpec::Random)));
        specs.push(("M3 critic: fixed(agent0)".into(), spec_with("c-fixed", &base, bf.clone(), CriticSpec::Fixed(0))));
        specs.push(("M3 critic: fair bandit (=REF B)".into(), ref_b.clone()));
        specs.push(("M3 critic: learned (final reward)".into(), spec_with("c-lf", &base, bf.clone(), CriticSpec::Learned(lc_final.clone()))));
        specs.push(("M3 critic: learned (verdict label)".into(), spec_with("c-lv", &base, bf.clone(), CriticSpec::Learned(lc_verd.clone()))));
        specs.push(("M3 critic: learned (verdict, no keyword prior)".into(), spec_with("c-lv0", &base, bf.clone(), CriticSpec::Learned(lc_verd0.clone()))));
        let hier_b = PolicySpec {
            name: "m4".into(),
            plan: base.clone(),
            plan_hier: true,
            build: BuildSpec::Fair { kind: base.clone(), hier: true },
            critic: CriticSpec::Fair { kind: base.clone(), hier: true },
        };
        specs.push(("M4 hierarchical(B) domain->specialist".into(), hier_b.clone()));
        // composition tuned on seeds 1-8 only: critic = learned (best mode) vs fair bandit, both hierarchical + QD build
        let mk_full = |critic: CriticSpec| PolicySpec {
            name: "cog-full".into(),
            plan: base.clone(),
            plan_hier: true,
            build: BuildSpec::Qd { cfg: qd_niche.clone(), other: base.clone(), hier: true },
            critic,
        };
        let c = vec![
            ("learned critic", mk_full(CriticSpec::Learned(lc_best.clone()))),
            ("fair-bandit critic", mk_full(CriticSpec::Fair { kind: base.clone(), hier: true })),
        ];
        let (full_choice, full, _) = tune("cog-full critic choice", c, &cfg, &tune_seeds, |m| m.overall);
        specs.push(("S cog-full (tuned composition)".into(), full.clone()));
        specs.push((
            "S minus QD".into(),
            PolicySpec { build: BuildSpec::Fair { kind: base.clone(), hier: true }, ..full.clone() },
        ));
        specs.push(("S minus learned critic".into(), PolicySpec { critic: CriticSpec::Fair { kind: base.clone(), hier: true }, ..full.clone() }));
        specs.push(("S alt: full with learned critic".into(), mk_full(CriticSpec::Learned(lc_best.clone()))));
        specs.push((
            "S minus hierarchy".into(),
            PolicySpec { plan_hier: false, build: BuildSpec::Qd { cfg: qd_niche.clone(), other: base.clone(), hier: false }, ..full.clone() },
        ));
        let res: Vec<(String, Vec<M>)> = specs.iter().map(|(n, s)| (n.clone(), eval(s, &cfg, &held))).collect();
        let get = |n: &str| -> &Vec<M> { &res.iter().find(|(x, _)| x == n).unwrap().1 };

        println!("\n--- held-out seeds 31-60: mean value (95% CI half-width) ---");
        println!("{:<52}{:>14}{:>14}{:>14}{:>14}{:>8}{:>8}", "policy", "overall", "creative", "correctness", "first300", "nov", "ent");
        for (n, r) in &res {
            let f = |g: fn(&M) -> f64| {
                let s = summarize(&col(r, g));
                format!("{:.3}+-{:.3}", s.mean, s.half_width)
            };
            println!("{:<52}{:>14}{:>14}{:>14}{:>14}{:>8.3}{:>8.3}", n, f(|m| m.overall), f(|m| m.creative), f(|m| m.correctness), f(|m| m.early), mean_of(r, |m| m.nov), mean_of(r, |m| m.ent));
        }

        // ---------- 4. tests ----------
        let mut t: Vec<Test> = vec![];
        // best fair baseline (held-out mean, conservative) per metric
        let best_for = |f: fn(&M) -> f64| -> &str {
            fair_list
                .iter()
                .map(|(n, _)| *n)
                .max_by(|x, y| mean_of(get(x), f).partial_cmp(&mean_of(get(y), f)).unwrap())
                .unwrap()
        };
        let (b_over, b_early, b_cre, b_cor) = (best_for(|m| m.overall), best_for(|m| m.early), best_for(|m| m.creative), best_for(|m| m.correctness));
        println!("best fair baseline (held-out): overall={b_over} first300={b_early} creative={b_cre} correctness={b_cor}");
        let refb = "REF B (flat fair, best learner)";
        // M1
        for n in ["M1 QD-niche + novelty(creative)", "M1b niche archive, no novelty", "M1c QD-niche novelty on ALL domains (control)"] {
            t.push(mk_test(&wn, "M1 QD niche routing (creative-domain value)", n, get(n), refb, get(refb), "creative", |m| m.creative));
        }
        t.push(mk_test(&wn, "M1 QD niche routing (creative-domain value)", "M1 QD-niche + novelty(creative)", get("M1 QD-niche + novelty(creative)"), b_cre, get(b_cre), "creative", |m| m.creative));
        t.push(mk_test(&wn, "M1 vs control: novelty in all domains hurts correctness?", "M1c QD-niche novelty on ALL domains (control)", get("M1c QD-niche novelty on ALL domains (control)"), refb, get(refb), "correctness", |m| m.correctness));
        t.push(mk_test(&wn, "M1 novelty bonus isolated (same archive, bonus on vs off)", "M1 QD-niche + novelty(creative)", get("M1 QD-niche + novelty(creative)"), "M1b niche archive, no novelty", get("M1b niche archive, no novelty"), "creative", |m| m.creative));
        t.push(mk_test(&wn, "M1 vs control: creative-only vs all", "M1 QD-niche + novelty(creative)", get("M1 QD-niche + novelty(creative)"), "M1c QD-niche novelty on ALL domains (control)", get("M1c QD-niche novelty on ALL domains (control)"), "overall", |m| m.overall));
        // M2
        for n in ["M2a novelty bonus, agent arms, creative-only", "M2b novelty bonus, agent arms, ALL domains (control)"] {
            t.push(mk_test(&wn, "M2 novelty bonus (creative value)", n, get(n), refb, get(refb), "creative", |m| m.creative));
            t.push(mk_test(&wn, "M2 novelty bonus (correctness value)", n, get(n), refb, get(refb), "correctness", |m| m.correctness));
        }
        t.push(mk_test(&wn, "M2 novelty bonus isolated (same learner, bonus on vs off)", "M2a novelty bonus, agent arms, creative-only", get("M2a novelty bonus, agent arms, creative-only"), "M2c agent arms, QD learner, NO novelty", get("M2c agent arms, QD learner, NO novelty"), "creative", |m| m.creative));
        t.push(mk_test(&wn, "M2 novelty bonus isolated (ALL-domain control vs off): correctness", "M2b novelty bonus, agent arms, ALL domains (control)", get("M2b novelty bonus, agent arms, ALL domains (control)"), "M2c agent arms, QD learner, NO novelty", get("M2c agent arms, QD learner, NO novelty"), "correctness", |m| m.correctness));
        t.push(mk_test(&wn, "M2 novelty bonus (overall)", "M2a novelty bonus, agent arms, creative-only", get("M2a novelty bonus, agent arms, creative-only"), refb, get(refb), "overall", |m| m.overall));
        // M3
        for cn in ["M3 critic: learned (final reward)", "M3 critic: learned (verdict label)", "M3 critic: learned (verdict, no keyword prior)"] {
            for rn in ["M3 critic: none", "M3 critic: random", "M3 critic: fixed(agent0)", "M3 critic: fair bandit (=REF B)"] {
                t.push(mk_test(&wn, "M3 critic selection (correctness value)", cn, get(cn), rn, get(rn), "correctness", |m| m.correctness));
            }
        }
        t.push(mk_test(&wn, "M3 critic selection (correctness value)", "M3 critic: random", get("M3 critic: random"), "M3 critic: none", get("M3 critic: none"), "correctness", |m| m.correctness));
        // M4
        t.push(mk_test(&wn, "M4 hierarchical vs flat (overall)", "M4 hierarchical(B) domain->specialist", get("M4 hierarchical(B) domain->specialist"), refb, get(refb), "overall", |m| m.overall));
        // S
        let sf = "S cog-full (tuned composition)";
        for (n, _) in &fair_list {
            t.push(mk_test(&wn, "S system vs fair baselines (overall)", sf, get(sf), n, get(n), "overall", |m| m.overall));
        }
        t.push(mk_test(&wn, "S system vs BEST fair baseline", sf, get(sf), b_over, get(b_over), "overall", |m| m.overall));
        t.push(mk_test(&wn, "S system vs BEST fair baseline (short horizon)", sf, get(sf), b_early, get(b_early), "first300", |m| m.early));
        t.push(mk_test(&wn, "S system vs BEST fair baseline", sf, get(sf), b_cre, get(b_cre), "creative", |m| m.creative));
        t.push(mk_test(&wn, "S system vs BEST fair baseline", sf, get(sf), b_cor, get(b_cor), "correctness", |m| m.correctness));
        for n in ["S minus QD", "S minus learned critic", "S minus hierarchy", "S alt: full with learned critic"] {
            t.push(mk_test(&wn, "S ablation: full vs minus-component", sf, get(sf), n, get(n), "overall", |m| m.overall));
        }

        world_json.push(json!({
            "world": wn,
            "cfg": format!("{:?}", cfg),
            "tuned": {
                "base_learner": base.label(), "eps": eps_k.label(), "pts": pts_k.label(), "warm_hebr": warm_k.label(),
                "qd_niche_creative": format!("{:?}", qd_niche), "novelty_agent_arms": format!("{:?}", qd_agent), "niche_no_novelty": format!("{:?}", qd_off),
                "learned_critic_final": format!("{:?}", lc_final), "learned_critic_verdict": format!("{:?}", lc_verd), "cog_full_critic": format!("{:?}", lc_best), "cog_full_critic_choice_tuned_on_1_8": full_choice,
            },
            "best_fair_baseline": {"overall": b_over, "first300": b_early, "creative": b_cre, "correctness": b_cor},
            "table": res.iter().map(|(n, r)| json!({
                "policy": n,
                "overall": summarize(&col(r, |m| m.overall)), "creative": summarize(&col(r, |m| m.creative)),
                "correctness": summarize(&col(r, |m| m.correctness)), "first300": summarize(&col(r, |m| m.early)),
                "creative_novelty": mean_of(r, |m| m.nov), "creative_approach_entropy": mean_of(r, |m| m.ent),
            })).collect::<Vec<_>>(),
        }));
        all_tests.extend(t);
    }

    // ---------- Holm over the whole family ----------
    let ps: Vec<f64> = all_tests.iter().map(|t| t.p).collect();
    let adj = holm_adjust(&ps);
    for (t, a) in all_tests.iter_mut().zip(adj) {
        t.holm = a;
    }
    println!("\n=================== PAIRED TESTS (cand - ref), held-out seeds 31-60, Holm over {} tests ===================", all_tests.len());
    let mut cur = String::new();
    for t in &all_tests {
        if t.world != cur {
            cur = t.world.clone();
            println!("\n-- {cur}");
        }
        let flag = if t.holm < 0.05 && t.lo > 0.0 { "WIN " } else if t.holm < 0.05 && t.hi < 0.0 { "LOSE" } else { "    " };
        println!(
            "{flag} [{}] {} | {} vs {} | {:.3} vs {:.3} diff {:+.4} [{:+.4},{:+.4}] w/l {}/{} holm p={:.4}",
            t.metric, t.hypothesis, t.cand, t.reference, t.mean_cand, t.mean_ref, t.diff, t.lo, t.hi, t.wins, t.losses, t.holm
        );
    }
    // rigged check: any significant positive mechanism result in world D?
    let rigged_world = all_tests.iter().filter(|t| t.world.starts_with("D-")).collect::<Vec<_>>();
    let rigged_false_wins: Vec<String> = rigged_world
        .iter()
        .filter(|t| t.holm < 0.05 && t.lo > 0.0 && !t.hypothesis.starts_with("S ablation") && !t.cand.starts_with("M3 critic: random") && !t.reference.starts_with("M3 critic: none") && !(t.hypothesis.starts_with("S system vs fair baselines") && !t.reference.starts_with("warm")))
        .map(|t| format!("{} | {} vs {} [{}]", t.hypothesis, t.cand, t.reference, t.metric))
        .collect();
    println!("\nRIGGED-CHECK (world D): significant positive mechanism results (should be only 'any critic beats none'): {:?}", rigged_false_wins);

    let tests_json: Vec<_> = all_tests
        .iter()
        .map(|t| {
            json!({
                "world": t.world, "hypothesis": t.hypothesis, "candidate": t.cand, "reference": t.reference, "metric": t.metric,
                "mean_candidate": t.mean_cand, "mean_reference": t.mean_ref, "diff": t.diff, "ci95": [t.lo, t.hi],
                "wins": t.wins, "losses": t.losses, "p": t.p, "holm_p": t.holm,
                "verdict": if t.holm < 0.05 && t.lo > 0.0 { "significantly better" } else if t.holm < 0.05 && t.hi < 0.0 { "significantly worse" } else { "no significant difference" },
            })
        })
        .collect();
    let j = json!({
        "protocol": "per-world tuning of every learner on seeds 1-8; held-out seeds 31-60; paired 95% t-CI; Holm over all tests in this file",
        "metric": "mean realised task value per task (correctness domains: final correct - 0.02 critic cost; creative: quality x (1-lam+lam*novelty) after critique refinement - cost)",
        "world_assumptions": [
            "5 domains; coding/science/business are CORRECTNESS, design/marketing are CREATIVE (router is told which: scope flag)",
            "tasks are plan->build->critique; each stage has its own router instance (no cross-stage credit mixing); plan reward = plan score, build reward = own outcome, critic reward = net task value",
            "creative value = quality x (1-lam+lam*novelty), novelty vs last 10 solutions in the domain (shared by all agents); embeddings tied to approach label (except world D)",
            "agents have a signature approach used w.p. 0.85 (habit): flat agent routing cannot steer approach, niche routing can",
            "multimodal creative worlds (A,C): the 4 best agents per creative domain have DISTINCT signatures at similar quality (needed for diversity to pay; this is the assumption that favours M1/M2); world B: one approach dominates",
            "critic gain = p(1-p)(sens-fa) - 0.02: a critic with sens<fa hurts; critics are independent of generator skill",
            "M2 mechanisms read the observed approach label + a quality score separate from novelty (extra observation channel baselines do not use); M3-verdict reads whether the critic verdict matched ground truth revealed later (extra signal)",
            "hierarchical routing = independent learner per domain (no cross-domain pooling): expected to win with specialists, lose with generalists",
            "D (rigged-check): generalist skills, equal critics, approach labels unrelated to embeddings: M1/M2/M3-selection/M4 have nothing to exploit",
            "drift: world B rotates visible-agent -> hidden-skill map at task 700; keyword maps (built pre-drift) go stale",
            "pyribs cross-check not performed (no packages installed); archive behaviour covered by unit tests only"
        ],
        "worlds": world_json,
        "tests": tests_json,
        "rigged_check_false_wins": rigged_false_wins,
    });
    if let Some(dir) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(&out, serde_json::to_string_pretty(&j).unwrap()).unwrap();
    println!("\nwrote {out}");
}
