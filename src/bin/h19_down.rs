//! h19_down: DOWNSTREAM question-answering / metric computation over documents in routed sessions.
//!   tune     --seeds 1-8                         choose tau + repair on the tune seeds (objective: acc - 0.03 cost of ext+ver)
//!   eval     --seeds 31-60 --tag held            plain (best fair router) vs extraction / verification / retrieval arms, per generator world, Holm
//!   controls --seeds 31-60 --tag held            no-structure + tiny docs (TOST +-0.01), uninformative verifier, adversarial layouts
//!   sens     --seeds 31-60 --tag held            reader-noise sweep (break-even), unseen-phrasing share, narrow verifier, no repair, tau
mod h19_common;
use h19_common::down::*;
use h19_common::gens::*;
use h19_common::verify::VCfg;
use h19_common::*;
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue, summarize};
use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).cloned().unwrap_or_default();
    let seeds = parse_seeds(&arg(&args, "--seeds").unwrap_or("1-8".into()));
    let tag = arg(&args, "--tag").unwrap_or("x".into());
    let mut cfg = SimCfg::default();
    // frozen config (written by `tune`)
    if let Ok(s) = std::fs::read_to_string("results/h19_tune.json") {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
            if let Some(t) = v["tau"].as_f64() {
                cfg.tau = t;
            }
            if let Some(r) = v["repair"].as_bool() {
                cfg.repair = r;
            }
        }
    }
    if let Some(t) = arg(&args, "--tau").and_then(|s| s.parse().ok()) {
        cfg.tau = t;
    }
    if let Some(n) = arg(&args, "--n").and_then(|s| s.parse().ok()) {
        cfg.n_per_world = n;
    }
    if let Some(u) = arg(&args, "--unseen").and_then(|s| s.parse().ok()) {
        cfg.unseen_p = u;
    }
    match cmd.as_str() {
        "tune" => tune(&seeds, &cfg),
        "eval" => eval(&seeds, &tag, &cfg),
        "controls" => controls(&seeds, &tag, &cfg),
        "sens" => sens(&seeds, &tag, &cfg),
        _ => eprintln!("usage: tune|eval|controls|sens"),
    }
}

const BASE_ROUTER: RK = RK::HebR;

fn tune(seeds: &[u64], cfg0: &SimCfg) {
    let mut best = (f64::NEG_INFINITY, 0.5, true);
    println!("# h19_down tune seeds {}..{}: objective = acc(ext+ver) - 0.03*cost", seeds[0], seeds[seeds.len() - 1]);
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7] {
        for repair in [true, false] {
            let mut cfg = cfg0.clone();
            cfg.tau = tau;
            cfg.repair = repair;
            let (mut a, mut c, mut n) = (0.0, 0.0, 0.0);
            let mut ap = 0.0;
            for &s in seeds {
                let preps = prep_session(s, &cfg);
                let sess = mk_session(s);
                let r = run_mode(s, &preps, &sess, 4, BASE_ROUTER, cfg.noise);
                let p = run_mode(s, &preps, &sess, 0, BASE_ROUTER, cfg.noise);
                a += r.acc(None);
                c += r.cost(None);
                ap += p.acc(None);
                n += 1.0;
            }
            let obj = a / n - 0.03 * c / n;
            println!("tau {:.1} repair {:<5}: ext+ver acc {:.4} cost {:.3} (plain acc {:.4}) objective {:.4}", tau, repair, a / n, c / n, ap / n, obj);
            if obj > best.0 {
                best = (obj, tau, repair);
            }
        }
    }
    println!("-> chosen tau {:.1} repair {}", best.1, best.2);
    std::fs::write("results/h19_tune.json", serde_json::to_string_pretty(&json!({"tau": best.1, "repair": best.2, "seeds": format!("{}-{}", seeds[0], seeds[seeds.len() - 1]), "objective": "acc(ext+ver)-0.03*cost"})).unwrap()).unwrap();
}

struct SeedRes {
    /// [router][mode] only plain mode for every router; other modes only for BASE_ROUTER
    plain_by_router: Vec<ModeRes>,
    modes: Vec<ModeRes>,
    /// extraction-arm outcome counts [ext mode 0..4][world 0..5][exact_ok, exact_wrong(silent), full fallback, span re-read]
    cnt: [[[f64; 4]; 5]; 4],
    /// retrieval coverage [world][covered, n]
    ret: [[f64; 2]; 5],
}
fn run_seed(s: u64, cfg: &SimCfg) -> SeedRes {
    let preps = prep_session(s, cfg);
    let sess = mk_session(s);
    let plain_by_router: Vec<ModeRes> = RKS.iter().map(|&rk| run_mode(s, &preps, &sess, 0, rk, cfg.noise)).collect();
    let modes: Vec<ModeRes> = (0..6).map(|m| run_mode(s, &preps, &sess, m, BASE_ROUTER, cfg.noise)).collect();
    let mut cnt = [[[0.0; 4]; 5]; 4];
    let mut ret = [[0.0; 2]; 5];
    for p in &preps {
        let w = p.world.idx();
        ret[w][0] += if p.ret_ok { 1.0 } else { 0.0 };
        ret[w][1] += 1.0;
        for m in 0..4 {
            let k = match &p.ext[m] {
                ExtOut::Exact { ok: true, .. } => 0,
                ExtOut::Exact { ok: false, .. } => 1,
                ExtOut::Full { .. } => 2,
                ExtOut::Span { .. } => 3,
            };
            cnt[m][w][k] += 1.0;
        }
    }
    SeedRes { plain_by_router, modes, cnt, ret }
}
fn ci_str(a: &[f64], b: &[f64]) -> (String, f64, f64, bool) {
    let d = paired_diff(a, b);
    let p = paired_t_pvalue(a, b);
    (format!("{:+.4} [{:+.4},{:+.4}] W/L/T {}/{}/{}", d.diff.mean, d.diff.lo, d.diff.hi, d.wins, d.losses, d.ties), p, d.diff.mean, d.significant)
}
fn t95(df: usize) -> f64 {
    1.645 + 2.2 / df as f64 + 3.8 / (df * df) as f64
}
/// TOST equivalence of paired differences within +-margin (alpha .05); returns (mean, equivalent, non_inferior)
fn tost(a: &[f64], b: &[f64], margin: f64) -> (f64, bool, bool) {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let s = summarize(&d);
    let se = s.sd / (d.len() as f64).sqrt();
    let t = t95(d.len() - 1);
    let lo_ok = (s.mean + margin) / se.max(1e-12) > t;
    let hi_ok = (margin - s.mean) / se.max(1e-12) > t;
    (s.mean, lo_ok && hi_ok, lo_ok)
}

fn eval(seeds: &[u64], tag: &str, cfg: &SimCfg) {
    println!("# h19_down eval tag={} seeds {}..{} ({} seeds) n/world/seed={} unseen share {:.2} tau {:.2} repair {} noise x{}", tag, seeds[0], seeds[seeds.len() - 1], seeds.len(), cfg.n_per_world, cfg.unseen_p, cfg.tau, cfg.repair, cfg.noise);
    println!("# SYNTHETIC. Reader-noise model is an ASSUMPTION (see h19_common/down.rs); extraction/verification arms run the real hand-written extractor + verifier on generated text.");
    let rs: Vec<SeedRes> = seeds.iter().map(|&s| run_seed(s, cfg)).collect();
    let worlds = [World::Paper, World::Fin, World::Biz];
    // best fair baseline = plain with the router that has the best mean accuracy on the EVALUATION seeds (conservative)
    let rmean: Vec<f64> = (0..RKS.len()).map(|k| rs.iter().map(|r| r.plain_by_router[k].acc(None)).sum::<f64>() / rs.len() as f64).collect();
    let bk = (0..RKS.len()).max_by(|&a, &b| rmean[a].partial_cmp(&rmean[b]).unwrap()).unwrap();
    println!("\n## plain reader, accuracy by router (mean over seeds): {}", RKS.iter().enumerate().map(|(k, r)| format!("{}={:.4}", rk_name(*r), rmean[k])).collect::<Vec<_>>().join("  "));
    println!("best fair baseline (picked on eval seeds -> favours baseline): plain + {}", rk_name(RKS[bk]));
    let base = |w: Option<World>| -> Vec<f64> { rs.iter().map(|r| r.plain_by_router[bk].acc(w)).collect() };
    let base_cost = |w: Option<World>| -> f64 { rs.iter().map(|r| r.plain_by_router[bk].cost(w)).sum::<f64>() / rs.len() as f64 };
    let arm = |m: usize, w: Option<World>| -> Vec<f64> { rs.iter().map(|r| r.modes[m].acc(w)).collect() };
    let arm_cost = |m: usize, w: Option<World>| -> f64 { rs.iter().map(|r| r.modes[m].cost(w)).sum::<f64>() / rs.len() as f64 };
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!("\n## accuracy and cost per arm (cost in full-document reads; plain = 1.0)");
    println!("{:<12} {:>9} {:>9} {:>9} {:>9} | {:>6} {:>6} {:>6} {:>6}", "arm", "ALL", "paper", "financial", "business", "cALL", "c_pap", "c_fin", "c_biz");
    let mut line = format!("{:<12}", format!("plain/{}", rk_name(RKS[bk])));
    for w in [None, Some(World::Paper), Some(World::Fin), Some(World::Biz)] {
        line.push_str(&format!(" {:>9.4}", mean(&base(w))));
    }
    line.push_str(" |");
    for w in [None, Some(World::Paper), Some(World::Fin), Some(World::Biz)] {
        line.push_str(&format!(" {:>6.3}", base_cost(w)));
    }
    println!("{}", line);
    for m in 0..6 {
        let mut line = format!("{:<12}", MODES[m]);
        for w in [None, Some(World::Paper), Some(World::Fin), Some(World::Biz)] {
            line.push_str(&format!(" {:>9.4}", mean(&arm(m, w))));
        }
        line.push_str(" |");
        for w in [None, Some(World::Paper), Some(World::Fin), Some(World::Biz)] {
            line.push_str(&format!(" {:>6.3}", arm_cost(m, w)));
        }
        println!("{}", line);
    }
    // paired comparisons vs best fair baseline with Holm
    println!("\n## paired 95% t-CI of (arm - best plain baseline), accuracy; Holm over {} comparisons (arms x worlds)", 5 * 3);
    let mut js = json!({});
    let mut ps = Vec::new();
    let mut rows = Vec::new();
    for m in 1..6 {
        for w in worlds {
            let (s, p, d, sig) = ci_str(&arm(m, Some(w)), &base(Some(w)));
            ps.push(p);
            rows.push((m, w, s, p, d, sig));
        }
    }
    let adj = holm_adjust(&ps);
    for (i, (m, w, s, p, d, _)) in rows.iter().enumerate() {
        println!("{:<10} {:<10} {}  p={:.2e} holm={:.2e} {}", MODES[*m], w.name(), s, p, adj[i], if adj[i] < 0.05 && *d > 0.0 { "WIN*" } else if adj[i] < 0.05 && *d < 0.0 { "LOSS!" } else { "tie" });
        js[format!("{}|{}", MODES[*m], w.name())] = json!({"diff": d, "p": p, "holm": adj[i]});
    }
    for m in 1..6 {
        let (s, p, d, _) = ci_str(&arm(m, None), &base(None));
        println!("{:<10} {:<10} {}  p={:.2e}", MODES[m], "ALL", s, p);
        js[format!("{}|ALL", MODES[m])] = json!({"diff": d, "p": p});
    }
    // decomposition (main effects, ALL worlds, paired over seeds)
    println!("\n## which part moves quality (paired over seeds, accuracy, mixed worlds)");
    let dd = |a: usize, b: usize, w: Option<World>| -> (f64, f64, f64) {
        let d = paired_diff(&arm(a, w), &arm(b, w));
        (d.diff.mean, d.diff.lo, d.diff.hi)
    };
    for (name, a, b) in [("EXTRACTION  (ext - plain, no retrieval)", 2, 0), ("VERIFICATION (ext+ver - ext)", 4, 2), ("RETRIEVAL on plain (plain+R - plain)", 1, 0), ("RETRIEVAL on ext (ext+R - ext)", 3, 2), ("RETRIEVAL on ext+ver (ext+ver+R - ext+ver)", 5, 4)] {
        let mut line = format!("{:<46}", name);
        for w in [None, Some(World::Paper), Some(World::Fin), Some(World::Biz)] {
            let (m, lo, hi) = dd(a, b, w);
            line.push_str(&format!("  {} {:+.4}[{:+.4},{:+.4}]", w.map_or("ALL", |w| w.name()), m, lo, hi));
        }
        println!("{}", line);
    }
    println!("
## what the extraction arms did with each question (share of questions): exact&right / exact&WRONG(silent) / fell back to full read / flagged-field re-read");
    for (mi, name) in [(0usize, "ext"), (1, "ext+R"), (2, "ext+ver"), (3, "ext+ver+R")] {
        let mut line = format!("{:<10}", name);
        for w in worlds {
            let mut c = [0.0; 4];
            for r in &rs {
                for k in 0..4 {
                    c[k] += r.cnt[mi][w.idx()][k];
                }
            }
            let t: f64 = c.iter().sum();
            line.push_str(&format!("  {}: {:.3}/{:.3}/{:.3}/{:.3}", w.name(), c[0] / t, c[1] / t, c[2] / t, c[3] / t));
        }
        println!("{}", line);
    }
    {
        let mut line = String::from("retrieval recall (needed spans inside the retrieved blocks):");
        for w in worlds {
            let (mut a, mut b) = (0.0, 0.0);
            for r in &rs {
                a += r.ret[w.idx()][0];
                b += r.ret[w.idx()][1];
            }
            line.push_str(&format!("  {} {:.3}", w.name(), a / b));
        }
        println!("{}", line);
    }
    // routing value: how much does a good router add inside each arm?
    println!("\n## router value (best router - keyword) for plain vs ext+ver arms: routing matters less once the answer is computed exactly");
    let plain_kw: Vec<f64> = rs.iter().map(|r| r.plain_by_router[0].acc(None)).collect();
    let plain_best: Vec<f64> = rs.iter().map(|r| r.plain_by_router[bk].acc(None)).collect();
    println!("plain: {} (best router) vs keyword {} -> {}", fmt3(mean(&plain_best)), fmt3(mean(&plain_kw)), ci_str(&plain_best, &plain_kw).0);
    let evr = arm(4, None);
    println!("ext+ver (hebbian-robust) {}", fmt3(mean(&evr)));
    let _ = std::fs::write(format!("results/h19_eval_{}.json", tag), serde_json::to_string_pretty(&js).unwrap());
}
fn fmt3(x: f64) -> String {
    format!("{:.4}", x)
}

fn controls(seeds: &[u64], tag: &str, cfg0: &SimCfg) {
    println!("# h19_down controls tag={} seeds {}..{}: arms must be harmless (TOST +-0.01 on accuracy, paired over seeds); SYNTHETIC", tag, seeds[0], seeds[seeds.len() - 1]);
    let mut js = json!({});
    for (name, worlds) in [("nostruct (prose, no numbers needed)", vec![World::NoStruct]), ("tiny (single clean fact; extraction unnecessary)", vec![World::Tiny]), ("mixed controls", vec![World::NoStruct, World::Tiny])] {
        let mut cfg = cfg0.clone();
        cfg.worlds = worlds.clone();
        cfg.unseen_p = 0.0;
        let rs: Vec<SeedRes> = seeds.iter().map(|&s| run_seed(s, &cfg)).collect();
        let mean = |f: &dyn Fn(&SeedRes) -> f64| rs.iter().map(f).sum::<f64>() / rs.len() as f64;
        let plain: Vec<f64> = rs.iter().map(|r| r.modes[0].acc(None)).collect();
        println!("\n## {}: plain acc {:.4} cost 1.000", name, mean(&|r| r.modes[0].acc(None)));
        for m in 2..6 {
            let a: Vec<f64> = rs.iter().map(|r| r.modes[m].acc(None)).collect();
            let (d, eq, ni) = tost(&a, &plain, 0.01);
            println!("  {:<10} acc {:.4}  diff {:+.4}  equivalent(+-0.01) {}  non-inferior(>=-0.01) {}  cost {:.3}", MODES[m], a.iter().sum::<f64>() / a.len() as f64, d, eq, ni, mean(&|r| r.modes[m].cost(None)));
            js[format!("{}|{}", name, MODES[m])] = json!({"diff": d, "equivalent": eq, "non_inferior": ni});
        }
    }
    // uninformative verifier
    {
        println!("\n## uninformative verifier control (random flags on 15% of kept facts) on the main worlds");
        let mut cfg = cfg0.clone();
        cfg.vmode = VMode::Random(150);
        cfg.repair = false;
        let rs: Vec<SeedRes> = seeds.iter().map(|&s| run_seed(s, &cfg)).collect();
        let cfg_n = SimCfg { vmode: VMode::None, ..cfg0.clone() };
        let rn: Vec<SeedRes> = seeds.iter().map(|&s| run_seed(s, &cfg_n)).collect();
        let a: Vec<f64> = rs.iter().map(|r| r.modes[4].acc(None)).collect();
        let b: Vec<f64> = rn.iter().map(|r| r.modes[2].acc(None)).collect();
        let (d, eq, ni) = tost(&a, &b, 0.01);
        let ca = rs.iter().map(|r| r.modes[4].cost(None)).sum::<f64>() / rs.len() as f64;
        let cb = rn.iter().map(|r| r.modes[2].cost(None)).sum::<f64>() / rn.len() as f64;
        println!("random-verifier ext+ver vs ext (no verifier): diff {:+.4} equivalent {} non-inferior {} ; cost {:.3} vs {:.3}", d, eq, ni, ca, cb);
        js["random_verifier"] = json!({"diff": d, "equivalent": eq, "non_inferior": ni, "cost_random": ca, "cost_ext": cb});
    }
    // adversarial layouts (fin): failure report
    {
        println!("\n## adversarial financial layouts (side-by-side / wrapped labels / split numbers): accuracy of plain reader vs extraction arms");
        for kind in 1..=3u8 {
            let (mut pa, mut ea, mut eva, mut sil, mut nn) = (0.0, 0.0, 0.0, 0.0, 0.0);
            let mut ce = (0.0, 0.0);
            for &s in seeds {
                let sess = mk_session(s);
                let mut preps = Vec::new();
                for k in 0..40u64 {
                    let doc = gen_fin_adv(s, k, kind);
                    preps.push(prepare(s, k, cfg0, doc));
                }
                let r0 = run_mode(s, &preps, &sess, 0, BASE_ROUTER, cfg0.noise);
                let r2 = run_mode(s, &preps, &sess, 2, BASE_ROUTER, cfg0.noise);
                let r4 = run_mode(s, &preps, &sess, 4, BASE_ROUTER, cfg0.noise);
                pa += r0.acc(None);
                ea += r2.acc(None);
                eva += r4.acc(None);
                ce.0 += r2.cost(None);
                ce.1 += r4.cost(None);
                // silent wrong = extraction answered Exact but wrong
                for p in &preps {
                    nn += 1.0;
                    if let ExtOut::Exact { ok: false, .. } = p.ext[2] {
                        sil += 1.0;
                    }
                }
            }
            let n = seeds.len() as f64;
            println!("kind {} : plain {:.3} | ext {:.3} (cost {:.2}) | ext+ver {:.3} (cost {:.2}) | ext answers silently wrong on {:.1}% of docs", kind, pa / n, ea / n, ce.0 / n, eva / n, ce.1 / n, 100.0 * sil / nn);
            js[format!("adv{}", kind)] = json!({"plain": pa / n, "ext": ea / n, "ext_ver": eva / n, "silent_wrong": sil / nn});
        }
    }
    let _ = std::fs::write(format!("results/h19_controls_{}.json", tag), serde_json::to_string_pretty(&js).unwrap());
}

fn overall(seeds: &[u64], cfg: &SimCfg) -> (f64, f64, f64, f64, f64, f64) {
    let rs: Vec<SeedRes> = seeds.iter().map(|&s| run_seed(s, cfg)).collect();
    let n = rs.len() as f64;
    let m = |k: usize| rs.iter().map(|r| r.modes[k].acc(None)).sum::<f64>() / n;
    let c = |k: usize| rs.iter().map(|r| r.modes[k].cost(None)).sum::<f64>() / n;
    (m(0), m(2), m(4), m(5), c(2), c(4))
}
fn sens(seeds: &[u64], tag: &str, cfg0: &SimCfg) {
    println!("# h19_down sensitivity tag={} seeds {}..{} (accuracy; ALL three worlds; plain uses the same router (hebbian-robust))", tag, seeds[0], seeds[seeds.len() - 1]);
    let mut js = json!({});
    println!("\n## (a) reader-noise multiplier (1.0 = default assumptions; 0.25 = a very strong reader); break-even = where ext+ver stops beating plain");
    println!("{:>6} {:>8} {:>8} {:>8} {:>10} | cost(ext) cost(ext+ver)", "noise", "plain", "ext", "ext+ver", "ext+ver+R");
    let preps_cache: Vec<(u64, Vec<Prep>)> = seeds.iter().map(|&s| (s, prep_session(s, cfg0))).collect();
    for noise in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0] {
        let mut acc = [0.0; 4];
        let mut cost = [0.0; 2];
        for (s, preps) in &preps_cache {
            let sess = mk_session(*s);
            for (k, m) in [0usize, 2, 4, 5].iter().enumerate() {
                let r = run_mode(*s, preps, &sess, *m, BASE_ROUTER, noise);
                acc[k] += r.acc(None);
                if *m == 2 {
                    cost[0] += r.cost(None);
                }
                if *m == 4 {
                    cost[1] += r.cost(None);
                }
            }
        }
        let n = seeds.len() as f64;
        println!("{:>6.2} {:>8.4} {:>8.4} {:>8.4} {:>10.4} | {:>8.3} {:>8.3}", noise, acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n, cost[0] / n, cost[1] / n);
        js[format!("noise{}", noise)] = json!({"plain": acc[0] / n, "ext": acc[1] / n, "ext_ver": acc[2] / n});
    }
    println!("\n## (b) share of UNSEEN-phrasing documents (extractor never developed on them)");
    println!("{:>6} {:>8} {:>8} {:>8} {:>10} | cost(ext) cost(ext+ver)", "unseen", "plain", "ext", "ext+ver", "ext+ver+R");
    for u in [0.0, 0.4, 1.0] {
        let cfg = SimCfg { unseen_p: u, ..cfg0.clone() };
        let (p, e, ev, evr, ce, cev) = overall(seeds, &cfg);
        println!("{:>6.2} {:>8.4} {:>8.4} {:>8.4} {:>10.4} | {:>8.3} {:>8.3}", u, p, e, ev, evr, ce, cev);
        js[format!("unseen{}", u)] = json!({"plain": p, "ext": e, "ext_ver": ev});
    }
    println!("\n## (c) verifier variants at unseen share {:.1} (all else frozen)", cfg0.unseen_p);
    println!("{:<34} {:>8} {:>8} {:>8} {:>8}", "variant", "ext", "ext+ver", "cost_ev", "plain");
    let variants: Vec<(&str, SimCfg)> = vec![
        ("full verifier + repair (default)", cfg0.clone()),
        ("full verifier, flag only (no repair)", SimCfg { repair: false, ..cfg0.clone() }),
        ("narrow verifier (dev knowledge only)", SimCfg { vc: VCfg { broad: false, ..cfg0.vc }, ..cfg0.clone() }),
        ("no footing", SimCfg { vc: VCfg { foot: false, ..cfg0.vc }, ..cfg0.clone() }),
        ("no scale check", SimCfg { vc: VCfg { scale: false, ..cfg0.vc }, ..cfg0.clone() }),
        ("no plausibility", SimCfg { vc: VCfg { plaus: false, ..cfg0.vc }, ..cfg0.clone() }),
        ("no claim coverage", SimCfg { vc: VCfg { cover: false, ..cfg0.vc }, ..cfg0.clone() }),
        ("span re-parse only", SimCfg { vc: VCfg { scale: false, plaus: false, foot: false, xchk: false, cover: false, ..cfg0.vc }, ..cfg0.clone() }),
    ];
    for (name, cfg) in variants {
        let (p, e, ev, _evr, _ce, cev) = overall(seeds, &cfg);
        println!("{:<34} {:>8.4} {:>8.4} {:>8.3} {:>8.4}", name, e, ev, cev, p);
        js[format!("variant|{}", name)] = json!({"ext": e, "ext_ver": ev, "cost": cev});
    }
    println!("\n## (d) abstention threshold tau");
    println!("{:>6} {:>8} {:>8} {:>8} {:>8}", "tau", "ext", "ext+ver", "cost_ev", "plain");
    for tau in [0.2, 0.35, 0.5, 0.65, 0.8] {
        let cfg = SimCfg { tau, ..cfg0.clone() };
        let (p, e, ev, _evr, _ce, cev) = overall(seeds, &cfg);
        println!("{:>6.2} {:>8.4} {:>8.4} {:>8.3} {:>8.4}", tau, e, ev, cev, p);
        js[format!("tau{}", tau)] = json!({"ext": e, "ext_ver": ev});
    }
    let _ = std::fs::write(format!("results/h19_sens_{}.json", tag), serde_json::to_string_pretty(&js).unwrap());
}
