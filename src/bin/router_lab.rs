//! Router lab: do the opt-in Hebbian options (self-tuning learning rate, feature
//! reliability weighting, UCB exploration bonus) beat / replace the hand-tuned
//! Hebbian router?
//!
//! Protocol (same as experiment.rs):
//!  1. Tune ONLY on seeds 1-8 (small grids per option).
//!  2. Evaluate on held-out seeds 31-60: per-seed paired differences with 95% t-CI
//!     vs `hebbian-tuned` (the current best hand-tuned setting) and vs keyword.
//!  3. Robustness without tuning: DEFAULT-settings variants, plus a (eta, temperature)
//!     sensitivity sweep on the held-out seeds (min / mean over the sweep).
//!  4. Secondary worlds on held-out seeds: random hidden structure, type revealed.
//!
//! Acceptance: an option "improves" only if its paired CI vs hebbian-tuned excludes 0
//! on held-out seeds, or it matches hebbian-tuned while needing no tuning (the
//! default-settings rows).
//!
//! Usage: cargo run --release --bin router_lab [-- --tasks 2000 --out results/router_lab.json]

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::hebbian::{HebbianConfig, HebbianRouter, MetaKind, WeightKind};
use neural_mesh::router::{Rng, Router};
use neural_mesh::sim::{Scenario, Window, World, WorldConfig};
use neural_mesh::stats::{paired_diff, summarize, PairedDiff, Summary};
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;

// ---------------------------------------------------------------------------
// One run = one single-agent router on one seed (immediate feedback).

#[derive(Clone, Debug, Default, Serialize)]
struct Metrics {
    success: f64,
    /// mean expected regret vs the oracle success probability
    regret: f64,
    pre: f64,
    /// success in the early-post-drift window (second quarter of the run)
    early_post: f64,
    /// success in the first 300 tasks right after the drift (fast-recovery probe)
    recover300: f64,
    late_post: f64,
    fail_window: f64,
}

#[derive(Clone)]
enum Kind {
    Keyword,
    EpsGreedy,
    PooledTs(f64),
    Hebbian(HebbianConfig),
}

fn run(kind: &Kind, seed: u64, wcfg: &WorldConfig) -> Metrics {
    let world = World::new(seed, wcfg.clone());
    let n = world.n_agents();
    let cands: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut keyword;
    let mut heb;
    let mut epsg;
    let mut pts;
    let router: &mut dyn Router = match kind {
        Kind::Keyword => {
            keyword = KeywordRouter::new(World::keyword_map(), n);
            &mut keyword
        }
        Kind::EpsGreedy => {
            epsg = EpsGreedyRouter::new(n, 0.99, 0.02);
            &mut epsg
        }
        Kind::PooledTs(g) => {
            pts = PooledThompsonRouter::new(n, *g);
            &mut pts
        }
        Kind::Hebbian(c) => {
            heb = HebbianRouter::new(n, c.clone());
            &mut heb
        }
    };
    let mut acc = [[0.0f64; 2]; 5]; // pre, early, late, fail, recover300
    let (mut succ, mut regret) = (0.0, 0.0);
    let d = world.drift_at();
    for t in 0..world.len() {
        let st = world.task(t);
        let a = router.select(&st.task, &cands, &mut rng);
        let r = world.outcome(t, &[a]);
        regret += world.oracle_prob(t) - world.success_prob(t, &[a]);
        succ += r;
        router.update(&st.task, a, r);
        let w = match world.window(t) {
            Window::PreDrift => 0,
            Window::EarlyPost => 1,
            Window::LatePost => 2,
        };
        acc[w][0] += r;
        acc[w][1] += 1.0;
        if world.in_failure(t) {
            acc[3][0] += r;
            acc[3][1] += 1.0;
        }
        if t >= d && t < d + 300 {
            acc[4][0] += r;
            acc[4][1] += 1.0;
        }
    }
    let rate = |a: [f64; 2]| if a[1] > 0.0 { a[0] / a[1] } else { f64::NAN };
    let nt = world.len() as f64;
    Metrics {
        success: succ / nt,
        regret: regret / nt,
        pre: rate(acc[0]),
        early_post: rate(acc[1]),
        late_post: rate(acc[2]),
        fail_window: rate(acc[3]),
        recover300: rate(acc[4]),
    }
}

fn run_seeds(kind: &Kind, seeds: &[u64], wcfg: &WorldConfig) -> Vec<Metrics> {
    seeds.par_iter().map(|&s| run(kind, s, wcfg)).collect()
}

fn mean_success(cfg: &HebbianConfig, seeds: &[u64], wcfg: &WorldConfig) -> f64 {
    let m = run_seeds(&Kind::Hebbian(cfg.clone()), seeds, wcfg);
    m.iter().map(|x| x.success).sum::<f64>() / m.len() as f64
}

// ---------------------------------------------------------------------------
// Tuning (seeds 1-8 only).

/// Exhaustive search over `cands`; returns the best config and its tuning score.
fn best_of(cands: Vec<HebbianConfig>, seeds: &[u64], wcfg: &WorldConfig) -> (HebbianConfig, f64, usize) {
    let scores: Vec<f64> = cands.par_iter().map(|c| mean_success(c, seeds, wcfg)).collect();
    let mut bi = 0;
    for i in 1..scores.len() {
        if scores[i] > scores[bi] {
            bi = i;
        }
    }
    (cands[bi].clone(), scores[bi], cands.len())
}

// ---------------------------------------------------------------------------
// Reporting helpers.

#[derive(Clone, Debug, Serialize)]
struct VariantResult {
    name: String,
    tuned_on_seeds_1_8: bool,
    cfg: Option<HebbianConfig>,
    success: Summary,
    regret: Summary,
    pre: Summary,
    early_post: Summary,
    recover300: Summary,
    late_post: Summary,
    fail_window: Summary,
    per_seed_success: Vec<f64>,
    vs_tuned: Option<PairedBundle>,
    vs_keyword: Option<PairedBundle>,
}

#[derive(Clone, Debug, Serialize)]
struct PairedBundle {
    success: PairedDiff,
    regret: PairedDiff,
    early_post: PairedDiff,
    recover300: PairedDiff,
}

fn col(m: &[Metrics], f: impl Fn(&Metrics) -> f64) -> Vec<f64> {
    m.iter().map(f).collect()
}

fn bundle(a: &[Metrics], b: &[Metrics]) -> PairedBundle {
    PairedBundle {
        success: paired_diff(&col(a, |m| m.success), &col(b, |m| m.success)),
        regret: paired_diff(&col(a, |m| m.regret), &col(b, |m| m.regret)),
        early_post: paired_diff(&col(a, |m| m.early_post), &col(b, |m| m.early_post)),
        recover300: paired_diff(&col(a, |m| m.recover300), &col(b, |m| m.recover300)),
    }
}

fn sig(p: &PairedDiff, higher_better: bool) -> &'static str {
    if !p.significant {
        "n.s."
    } else if (p.diff.mean > 0.0) == higher_better {
        "SIG +"
    } else {
        "SIG -"
    }
}

fn fmt_pd(p: &PairedDiff, higher_better: bool) -> String {
    format!(
        "{:+.4} [{:+.4},{:+.4}] {:>2}/{:>2}/{:>2} {}",
        p.diff.mean,
        p.diff.lo,
        p.diff.hi,
        p.wins,
        p.losses,
        p.ties,
        sig(p, higher_better)
    )
}

struct Row {
    name: String,
    kind: Kind,
    tuned: bool,
}

fn evaluate(rows: &[Row], seeds: &[u64], wcfg: &WorldConfig, tuned_name: &str) -> Vec<VariantResult> {
    let all: Vec<Vec<Metrics>> = rows.iter().map(|r| run_seeds(&r.kind, seeds, wcfg)).collect();
    let idx_of = |nm: &str| rows.iter().position(|r| r.name == nm);
    let ti = idx_of(tuned_name);
    let ki = idx_of("keyword");
    rows.iter()
        .enumerate()
        .map(|(i, r)| {
            let m = &all[i];
            VariantResult {
                name: r.name.clone(),
                tuned_on_seeds_1_8: r.tuned,
                cfg: match &r.kind {
                    Kind::Hebbian(c) => Some(c.clone()),
                    _ => None,
                },
                success: summarize(&col(m, |x| x.success)),
                regret: summarize(&col(m, |x| x.regret)),
                pre: summarize(&col(m, |x| x.pre)),
                early_post: summarize(&col(m, |x| x.early_post)),
                recover300: summarize(&col(m, |x| x.recover300)),
                late_post: summarize(&col(m, |x| x.late_post)),
                fail_window: summarize(&col(m, |x| x.fail_window)),
                per_seed_success: col(m, |x| x.success),
                vs_tuned: ti.filter(|&t| t != i).map(|t| bundle(m, &all[t])),
                vs_keyword: ki.filter(|&k| k != i).map(|k| bundle(m, &all[k])),
            }
        })
        .collect()
}

fn print_results(title: &str, res: &[VariantResult]) {
    println!("\n=== {title} ===");
    println!(
        "{:<34} {:>15} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "variant", "success ±95%CI", "regret", "pre", "early", "rec300", "late"
    );
    for r in res {
        println!(
            "{:<34} {:.4} ±{:.4} {:>7.4} {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
            r.name, r.success.mean, r.success.half_width, r.regret.mean, r.pre.mean, r.early_post.mean, r.recover300.mean, r.late_post.mean
        );
    }
    println!("\n  paired diffs vs hebbian-tuned (variant - tuned): mean [95% CI] wins/losses/ties");
    println!("  {:<32} {:<38} {:<38} {:<38}", "variant", "success", "regret (lower better)", "early-post");
    for r in res {
        if let Some(b) = &r.vs_tuned {
            println!(
                "  {:<32} {:<38} {:<38} {:<38}",
                r.name,
                fmt_pd(&b.success, true),
                fmt_pd(&b.regret, false),
                fmt_pd(&b.early_post, true)
            );
        }
    }
    println!("\n  paired diffs vs keyword (variant - keyword)");
    println!("  {:<32} {:<38} {:<38} {:<38}", "variant", "success", "regret (lower better)", "early-post");
    for r in res {
        if let Some(b) = &r.vs_keyword {
            println!(
                "  {:<32} {:<38} {:<38} {:<38}",
                r.name,
                fmt_pd(&b.success, true),
                fmt_pd(&b.regret, false),
                fmt_pd(&b.early_post, true)
            );
        }
    }
}

// ---------------------------------------------------------------------------

/// Best (by mean held-out success) of hebbian-tuned and the fair bandit baselines.
fn best_ref(rows: &[Row], seeds: &[u64], wcfg: &WorldConfig) -> String {
    let mut best = (String::new(), f64::MIN);
    for r in rows.iter().filter(|r| ["hebbian-tuned", "eps-greedy(.99,.02)", "pooled-ts(.99)", "pooled-ts(1.0)"].contains(&r.name.as_str())) {
        let m = run_seeds(&r.kind, seeds, wcfg);
        let s = m.iter().map(|x| x.success).sum::<f64>() / m.len() as f64;
        if s > best.1 {
            best = (r.name.clone(), s);
        }
    }
    best.0
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tasks: usize = arg_value(&args, "--tasks").and_then(|s| s.parse().ok()).unwrap_or(2000);
    let out = arg_value(&args, "--out").unwrap_or_else(|| "results/router_lab.json".to_string());
    let wcfg = WorldConfig { n_tasks: tasks, ..WorldConfig::default() };
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let held: Vec<u64> = (31..=60).collect();

    let base = HebbianConfig::default();
    let tuned = HebbianConfig { eta: 0.05, temperature: 0.02, decay: 0.001, ..base.clone() };
    println!("neural-mesh router_lab: {tasks} tasks/run, tuning seeds 1-8, held-out seeds 31-60");

    // ---- tuning on seeds 1-8 -------------------------------------------------
    let tune = |label: &str, cands: Vec<HebbianConfig>| -> HebbianConfig {
        let (c, s, n) = best_of(cands, &tune_seeds, &wcfg);
        println!("[tune seeds 1-8] {label}: {n} configs, best -> {s:.4}");
        c
    };
    println!("[tune seeds 1-8] default hebbian {:.4}, hebbian-tuned {:.4}", mean_success(&base, &tune_seeds, &wcfg), mean_success(&tuned, &tune_seeds, &wcfg));

    // 1. self-tuning learning rate. `tuned` supplies decay; eta/floor/decay/temp/lambda searched.
    let mut ph_c = vec![];
    for &eta in &[0.1, 0.25, 0.5] {
        for &floor in &[0.01, 0.02, 0.05] {
            for &md in &[0.9, 0.95, 0.98] {
                for &temp in &[0.02, 0.05] {
                    for &lam in &[2.0, 4.0] {
                        ph_c.push(HebbianConfig {
                            meta: MetaKind::PageHinkley,
                            eta,
                            meta_eta_floor: floor,
                            meta_decay: md,
                            temperature: temp,
                            meta_lambda: lam,
                            decay: 0.001,
                            ..base.clone()
                        });
                    }
                }
            }
        }
    }
    let ph_tuned = tune("meta=PageHinkley", ph_c);

    let mut dbd_c = vec![];
    for &eta in &[0.1, 0.3, 0.5] {
        for &floor in &[0.01, 0.03] {
            for &kappa in &[0.005, 0.02] {
                for &phi in &[0.8, 0.9, 0.95] {
                    for &temp in &[0.02, 0.05] {
                        dbd_c.push(HebbianConfig {
                            meta: MetaKind::Dbd,
                            eta,
                            meta_eta_floor: floor,
                            dbd_kappa: kappa,
                            dbd_phi: phi,
                            temperature: temp,
                            decay: 0.001,
                            ..base.clone()
                        });
                    }
                }
            }
        }
    }
    let dbd_tuned = tune("meta=Dbd", dbd_c);

    // 2. feature reliability weighting on top of hebbian-tuned.
    let mut w_tuned = vec![];
    for kind in [WeightKind::Idf, WeightKind::Entropy, WeightKind::Contrast] {
        let mut c = vec![];
        for &fl in &[0.1, 0.3, 0.6] {
            for &al in &[0.02, 0.1] {
                c.push(HebbianConfig { weighting: kind, weight_floor: fl, weight_alpha: al, ..tuned.clone() });
            }
        }
        w_tuned.push((kind, tune(&format!("weighting={kind:?}"), c)));
    }

    // 3. UCB bonus on top of hebbian-tuned.
    let mut ucb_c = vec![];
    for &c in &[0.01, 0.03, 0.06, 0.12, 0.25] {
        for &d in &[0.99, 0.995, 0.999] {
            ucb_c.push(HebbianConfig { ucb_c: c, ucb_discount: d, ..tuned.clone() });
        }
    }
    let ucb_tuned = tune("ucb", ucb_c);

    // Combined: tuned PH (+ best weighting + ucb), small joint grid over the add-ons.
    let best_w = {
        let mut best = (WeightKind::Off, tuned.clone(), f64::MIN);
        for (k, c) in &w_tuned {
            let s = mean_success(c, &tune_seeds, &wcfg);
            if s > best.2 {
                best = (*k, c.clone(), s);
            }
        }
        best
    };
    println!("[tune seeds 1-8] best weighting alone: {:?} -> {:.4}", best_w.0, best_w.2);
    let mut comb_c = vec![];
    for &use_w in &[false, true] {
        for &use_u in &[false, true] {
            for &c in &[0.03, 0.06] {
                let mut cfg = ph_tuned.clone();
                if use_w {
                    cfg.weighting = best_w.0;
                    cfg.weight_floor = best_w.1.weight_floor;
                    cfg.weight_alpha = best_w.1.weight_alpha;
                }
                if use_u {
                    cfg.ucb_c = c;
                    cfg.ucb_discount = ucb_tuned.ucb_discount;
                }
                comb_c.push(cfg);
            }
        }
    }
    let comb_tuned = tune("combined (PH tuned + weighting? + ucb?)", comb_c);

    // ---- named variants ------------------------------------------------------
    // DEFAULT-settings variants: the stock config plus ONE option at its default knobs.
    let ph_def = HebbianConfig { meta: MetaKind::PageHinkley, ..base.clone() };
    let dbd_def = HebbianConfig { meta: MetaKind::Dbd, ..base.clone() };
    let w_idf_def = HebbianConfig { weighting: WeightKind::Idf, ..base.clone() };
    let w_ent_def = HebbianConfig { weighting: WeightKind::Entropy, ..base.clone() };
    let w_con_def = HebbianConfig { weighting: WeightKind::Contrast, ..base.clone() };
    let ucb_def = HebbianConfig { ucb_c: 0.06, ..base.clone() };
    let all_def = HebbianConfig {
        meta: MetaKind::PageHinkley,
        weighting: WeightKind::Contrast,
        ucb_c: 0.06,
        ..base.clone()
    };
    // PH with only default-hebbian temperature/decay lowered to the "sane" hebbian-tuned
    // temperature (shows how much of the gain is plain temperature, not the meta rule).
    let ph_def_t02 = HebbianConfig { temperature: 0.02, decay: 0.001, ..ph_def.clone() };

    let mk = |name: &str, c: &HebbianConfig, tuned_flag: bool| Row { name: name.to_string(), kind: Kind::Hebbian(c.clone()), tuned: tuned_flag };
    let rows: Vec<Row> = vec![
        Row { name: "keyword".into(), kind: Kind::Keyword, tuned: false },
        Row { name: "eps-greedy(.99,.02)".into(), kind: Kind::EpsGreedy, tuned: false },
        Row { name: "pooled-ts(.99)".into(), kind: Kind::PooledTs(0.99), tuned: false },
        Row { name: "pooled-ts(1.0)".into(), kind: Kind::PooledTs(1.0), tuned: false },
        mk("hebbian-default", &base, false),
        mk("hebbian-tuned", &tuned, true),
        mk("[def] meta-ph", &ph_def, false),
        mk("[def] meta-ph +temp.02/decay.001", &ph_def_t02, false),
        mk("[def] meta-dbd", &dbd_def, false),
        mk("[def] weight-idf", &w_idf_def, false),
        mk("[def] weight-entropy", &w_ent_def, false),
        mk("[def] weight-contrast", &w_con_def, false),
        mk("[def] ucb", &ucb_def, false),
        mk("[def] all (ph+contrast+ucb)", &all_def, false),
        mk("[tuned] meta-ph", &ph_tuned, true),
        mk("[tuned] meta-dbd", &dbd_tuned, true),
        mk("[tuned] weight-idf", &w_tuned[0].1, true),
        mk("[tuned] weight-entropy", &w_tuned[1].1, true),
        mk("[tuned] weight-contrast", &w_tuned[2].1, true),
        mk("[tuned] ucb", &ucb_tuned, true),
        mk("[tuned] combined", &comb_tuned, true),
    ];

    let ref_name = best_ref(&rows, &held, &wcfg);
    println!("
reference for acceptance = best of {{hebbian-tuned, eps-greedy, pooled-ts}} on held-out: {ref_name}");
    let main_res = evaluate(&rows, &held, &wcfg, &ref_name);
    print_results("HELD-OUT seeds 31-60, fixed world (primary)", &main_res);
    // the same table on the tuning seeds, to expose overfit
    let tune_res = evaluate(&rows, &tune_seeds, &wcfg, &ref_name);
    println!("\n--- (for reference) same variants on TUNING seeds 1-8: success ---");
    for r in &tune_res {
        println!("  {:<34} {:.4}", r.name, r.success.mean);
    }

    // ---- robustness sweep: eta x temperature, no tuning, held-out seeds -------
    let etas = [0.02, 0.05, 0.1, 0.2, 0.5];
    let temps = [0.02, 0.05, 0.1];
    println!("\n=== eta x temperature sensitivity (held-out seeds 31-60, decay .001, mean success) ===");
    println!("  families: plain | meta-ph | meta-dbd | plain+ucb | plain+contrast  (eta = ceiling for meta)");
    let mut sweep_json = vec![];
    type Fam = (&'static str, Box<dyn Fn(f64, f64) -> HebbianConfig>);
    let b2 = base.clone();
    let fams: Vec<Fam> = vec![
        ("plain", Box::new({ let b = b2.clone(); move |e, t| HebbianConfig { eta: e, temperature: t, decay: 0.001, ..b.clone() } })),
        ("meta-ph", Box::new({ let b = b2.clone(); move |e, t| HebbianConfig { meta: MetaKind::PageHinkley, eta: e, temperature: t, decay: 0.001, ..b.clone() } })),
        ("meta-dbd", Box::new({ let b = b2.clone(); move |e, t| HebbianConfig { meta: MetaKind::Dbd, eta: e, temperature: t, decay: 0.001, ..b.clone() } })),
        ("plain+ucb", Box::new({ let b = b2.clone(); move |e, t| HebbianConfig { ucb_c: 0.06, eta: e, temperature: t, decay: 0.001, ..b.clone() } })),
        ("plain+contrast", Box::new({ let b = b2.clone(); move |e, t| HebbianConfig { weighting: WeightKind::Contrast, eta: e, temperature: t, decay: 0.001, ..b.clone() } })),
    ];
    for (fname, f) in &fams {
        let mut cells = vec![];
        print!("  {:<15}", fname);
        for &t in &temps {
            for &e in &etas {
                let s = mean_success(&f(e, t), &held, &wcfg);
                cells.push(s);
                sweep_json.push(json!({"family": fname, "eta": e, "temperature": t, "success": s}));
            }
        }
        let mn = cells.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = cells.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = cells.iter().sum::<f64>() / cells.len() as f64;
        println!(" min {mn:.4}  mean {mean:.4}  max {mx:.4}  (n={})", cells.len());
    }
    println!("  (keyword baseline: {:.4})", main_res[0].success.mean);

    // ---- secondary worlds (held-out seeds), for robustness of conclusions -------
    let mut secondary = vec![];
    let worlds: Vec<(&str, WorldConfig)> = vec![
        ("random-structure", WorldConfig { scenario: Scenario::Random, ..wcfg.clone() }),
        ("no-traps", WorldConfig { scenario: Scenario::NoTraps, ..wcfg.clone() }),
        ("type-revealed", WorldConfig { reveal_type: true, ..wcfg.clone() }),
    ];
    for (label, w) in &worlds {
        let r = evaluate(&rows, &held, w, &best_ref(&rows, &held, w));
        print_results(&format!("HELD-OUT seeds 31-60, world = {label}"), &r);
        secondary.push(json!({"world": label, "results": r}));
    }

    // ---- horizons ---------------------------------------------------------------
    let mut hz_json = vec![];
    for &h in &[500usize, 1000, 2000, 6000] {
        let w = WorldConfig { n_tasks: h, ..wcfg.clone() };
        let rname = best_ref(&rows, &held, &w);
        let r = evaluate(&rows, &held, &w, &rname);
        println!("
=== HORIZON {h} tasks, held-out 31-60; reference (best fair baseline) = {rname} ===");
        println!("  {:<34} {:>8}   paired success diff vs reference", "variant", "success");
        for v in &r {
            let d = v.vs_tuned.as_ref().map(|b| fmt_pd(&b.success, true)).unwrap_or_default();
            println!("  {:<34} {:>8.4}   {}", v.name, v.success.mean, d);
        }
        hz_json.push(json!({"horizon": h, "reference": rname, "results": r}));
    }

    // ---- JSON ----------------------------------------------------------------
    let doc = json!({
        "protocol": "tuned on seeds 1-8 only; evaluated on held-out seeds 31-60; paired per-seed diffs with 95% t-CI",
        "tasks": tasks,
        "tuned_configs": {
            "hebbian_tuned": tuned, "meta_ph": ph_tuned, "meta_dbd": dbd_tuned,
            "weight_idf": w_tuned[0].1, "weight_entropy": w_tuned[1].1, "weight_contrast": w_tuned[2].1,
            "ucb": ucb_tuned, "combined": comb_tuned,
        },
        "held_out_fixed_world": main_res,
        "tuning_seeds_fixed_world": tune_res,
        "eta_temp_sweep_held_out": sweep_json,
        "secondary_worlds_held_out": secondary,
        "horizons_held_out": hz_json,
        "reference_main": ref_name,
    });
    if let Some(parent) = std::path::Path::new(&out).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("write json");
    println!("\nwrote {out}");
}
