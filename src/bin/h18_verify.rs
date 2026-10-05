//! h18_verify: (1) selfcheck of the fin.rs tools + independent verifier, (2) mutation testing with seeded evaluator bugs:
//! catch rate of the SHARED verifier (re-run the same evaluator) vs the INDEPENDENT verifier (second algorithm / invariant).
//!   h18_verify selfcheck --seeds 1-6
//!   h18_verify mutate --seeds 31-60 --tag held          (also tune 1-8, confirm 61-90)
//! SYNTHETIC / SEEDED: the bugs are planted by me; "independent" means a different algorithm by the same author.
mod h18_common;
use h18_common::corpus::corpus;
use h18_common::world::{gen_world, WCfg};
use h18_common::{arg, parse_range};
use neural_mesh::fin::{self, bug, close, parse_out, verify_indep, verify_shared, Args};
use neural_mesh::router::Rng;
use neural_mesh::tools::Tool;
use rayon::prelude::*;
use serde_json::json;

fn eval_v(b: u8, spec: &str) -> Option<String> {
    let a = Args::parse(spec).ok()?;
    fin::eval(b, &a).ok().map(|o| o.v)
}
fn same(a: &str, b: &str) -> bool {
    match (a.parse::<f64>(), b.parse::<f64>()) {
        (Ok(x), Ok(y)) if !a.contains('e') || !b.contains('e') => close(x, y),
        _ => a == b,
    }
}
fn wrong_claims(v: &str, r: &mut Rng) -> Vec<String> {
    match v.parse::<f64>() {
        Ok(x) if !v.contains('e') => {
            let mut o = vec![fin::fmt(x * 1.03 + 0.01), fin::fmt(-x - 0.5), fin::fmt(x * 10.0 + 1.0), fin::fmt(x * (1.0 - 0.2 * r.f64()) - 0.5), fin::fmt(x + 0.01 * x.abs().max(1.0) * (1.0 + r.f64()))];
            o.retain(|w| !same(v, w));
            o
        }
        _ => {
            let mut cs: Vec<char> = v.chars().collect();
            let idx: Vec<usize> = cs.iter().enumerate().filter(|(_, c)| c.is_ascii_digit()).map(|(i, _)| i).collect();
            let mut o = Vec::new();
            for k in 0..3 {
                if let Some(i) = idx.get(k % idx.len().max(1)) {
                    let d = cs[*i].to_digit(10).unwrap_or(0);
                    cs[*i] = char::from_digit((d + 1 + r.below(8) as u32) % 10, 10).unwrap_or('0');
                    o.push(cs.iter().collect::<String>());
                }
            }
            o.retain(|w| w != v);
            o
        }
    }
}

fn selfcheck(seeds: &[u64]) {
    let mut n = 0usize;
    let mut false_rej = 0usize;
    let mut wrong_n = 0usize;
    let mut wrong_missed = 0usize;
    let mut per_op: std::collections::BTreeMap<&str, (usize, usize, usize, usize)> = Default::default();
    let mut shared_fr = 0usize;
    for s in seeds {
        for (op, spec) in corpus(*s, 20) {
            let v = match eval_v(0, &spec) {
                Some(v) => v,
                None => continue,
            };
            n += 1;
            let e = per_op.entry(op).or_default();
            e.0 += 1;
            if let Err(why) = verify_indep(0, &spec, &v) {
                false_rej += 1;
                e.1 += 1;
                if e.1 <= 2 {
                    println!("  FALSE REJECT [{op}] {spec} -> {v}: {why}");
                }
            }
            if verify_shared(0, &spec, &v).is_err() {
                shared_fr += 1;
            }
            let mut r = Rng::new(*s ^ 0x5eed);
            for w in wrong_claims(&v, &mut r) {
                wrong_n += 1;
                e.2 += 1;
                if verify_indep(0, &spec, &w).is_ok() {
                    wrong_missed += 1;
                    e.3 += 1;
                    if e.3 <= 2 {
                        println!("  MISSED WRONG [{op}] {spec}: truth {v}, claim {w} accepted");
                    }
                }
            }
        }
    }
    println!("selfcheck seeds {:?}: {} clean outputs; independent verifier false rejects {} ; shared false rejects {}; wrong claims {} of which accepted {}", (seeds[0], seeds[seeds.len() - 1]), n, false_rej, shared_fr, wrong_n, wrong_missed);
    println!("ops covered by an independent verifier: {} / {}", per_op.values().filter(|e| e.0 > 0).count(), h18_common::corpus::all_ops().len());
    // robustness: random garbage and extreme numerics never panic, Ok outputs are finite
    let mut r = Rng::new(99);
    let toks = ["npv", "irr", "compound", "bondprice", "uconv", "gas", "kin", "mean", "molmass", "stoich", "errprop", "sfround", "sci", "payback", "amort", "limit", "var", "ci95"];
    let extremes = ["0", "-1", "1e15", "-1e15", "0.0000001", "10000", "nan", "inf", "1e400", "", "1,2,,3", "H2O", "m/s", "999999999999", "2.5e-3", "-0", ".", "1e-320"];
    let keys = ["rate", "cf", "p", "r", "n", "m", "x", "y", "f", "face", "cpn", "price", "from", "to", "eq", "given", "want", "u", "v", "a", "t", "s", "find", "k", "sx", "sy", "mass", "ma", "mb", "ca", "cl"];
    let (mut ok_out, mut errs, mut nonfinite) = (0, 0, 0);
    for _ in 0..20000 {
        let mut s = toks[r.below(toks.len())].to_string();
        for _ in 0..r.below(6) {
            s.push_str(&format!(" {}={}", keys[r.below(keys.len())], extremes[r.below(extremes.len())]));
        }
        for t in [Box::new(fin::FinTool::default()) as Box<dyn Tool>, Box::new(fin::SciTool::default()), Box::new(fin::EconTool::default())] {
            match t.call(&s) {
                Ok(o) => {
                    ok_out += 1;
                    if let Some(v) = parse_out(&o).get("v").and_then(|x| x.parse::<f64>().ok()) {
                        if !v.is_finite() {
                            nonfinite += 1;
                        }
                    }
                }
                Err(_) => errs += 1,
            }
        }
        let _ = verify_indep(0, &s, "1");
        let _ = verify_shared(0, &s, "1");
    }
    // valid specs with ONE field replaced by an extreme literal (so the evaluators actually run)
    let (mut ok2, mut err2, mut nf2) = (0, 0, 0);
    let mut r2 = Rng::new(7);
    let cps = corpus(3, 6);
    for _ in 0..30000 {
        let (_, sp) = &cps[r2.below(cps.len())];
        let mut parts: Vec<String> = sp.split_whitespace().map(|x| x.to_string()).collect();
        if parts.len() < 2 {
            continue;
        }
        let i = 1 + r2.below(parts.len() - 1);
        if let Some((k, _)) = parts[i].clone().split_once('=') {
            parts[i] = format!("{k}={}", extremes[r2.below(extremes.len())]);
        }
        let s = parts.join(" ");
        for t in [Box::new(fin::FinTool::default()) as Box<dyn Tool>, Box::new(fin::SciTool::default()), Box::new(fin::EconTool::default())] {
            match t.call(&s) {
                Ok(o) => {
                    ok2 += 1;
                    if let Some(v) = parse_out(&o).get("v").and_then(|x| x.parse::<f64>().ok()) {
                        if !v.is_finite() {
                            nf2 += 1;
                        }
                    }
                }
                Err(_) => err2 += 1,
            }
        }
        let _ = verify_indep(0, &s, "1");
    }
    println!("robustness (valid specs with one extreme field): 90000 tool calls: ok {ok2}, clean errors {err2}, non-finite {nf2}, no panic");
    let nonfinite = nonfinite + nf2;
    println!("robustness: 60000 tool calls on mutated/garbage specs: ok {ok_out}, clean errors {errs}, non-finite outputs {nonfinite}, no panic");
    let pass = false_rej == 0 && wrong_missed == 0 && nonfinite == 0;
    println!("SELFCHECK {}", if pass { "PASS" } else { "FAIL" });
}

#[derive(Default, Clone)]
struct M {
    n_op: usize,
    silent: usize,
    loud: usize,
    shared_catch: usize,
    indep_catch: usize,
    shared_fr: usize,
    indep_fr: usize,
}

fn mutate(seeds: &[u64], tag: &str) {
    // workload C: corpus over all ops; workload W: canonical specs of the word-problem worlds A+B (realistic op mix)
    let mut work_c: Vec<(&'static str, String)> = Vec::new();
    for s in seeds {
        work_c.extend(corpus(*s, 8));
    }
    let wcfgs = [WCfg::world_a(), WCfg::world_b()];
    let work_w: Vec<(&'static str, String)> = seeds.par_iter().flat_map(|s| wcfgs.iter().flat_map(|c| gen_world(c, *s).tasks.into_iter().filter(|t| t.dom != 3 && !t.dup).map(|t| (t.op, t.spec)).collect::<Vec<_>>()).collect::<Vec<_>>()).collect();
    println!("mutation testing, seeds {}-{}: workload C = {} specs (corpus over all ops), workload W = {} specs (word-problem worlds A+B)", seeds[0], seeds[seeds.len() - 1], work_c.len(), work_w.len());
    let mut rows = Vec::new();
    for (wname, work) in [("C", &work_c), ("W", &work_w)] {
        println!("\n== workload {wname} ==");
        println!("{:<28} {:>7} {:>8} {:>6} {:>8} {:>8} {:>7}/{:<7}", "mutant", "specs", "silent", "loud", "shared%", "indep%", "FR sh", "FR in");
        let mut tot = M::default();
        let mut tot_noncm = M::default();
        let mut killed = (0, 0, 0usize);
        for (b, name) in bug::ALL {
            let cm = if bug::is_common_mode(*b) { *b } else { 0 };
            let res: M = work
                .par_iter()
                .map(|(_op, spec)| {
                    let mut m = M::default();
                    let clean = match eval_v(0, spec) {
                        Some(c) => c,
                        None => return m,
                    };
                    m.n_op = 1;
                    let dirty = eval_v(*b, spec);
                    // false rejects of both verifiers on the clean output (with the common-mode bug planted where relevant)
                    if verify_indep(cm, spec, &clean).is_err() && cm == 0 {
                        m.indep_fr += 1;
                    }
                    if verify_shared(0, spec, &clean).is_err() {
                        m.shared_fr += 1;
                    }
                    match dirty {
                        None => m.loud = 1,
                        Some(d) => {
                            if !same(&clean, &d) {
                                m.silent = 1;
                                if verify_shared(*b, spec, &d).is_err() {
                                    m.shared_catch = 1;
                                }
                                if verify_indep(cm, spec, &d).is_err() {
                                    m.indep_catch = 1;
                                }
                            }
                        }
                    }
                    m
                })
                .reduce(M::default, |mut a, b| {
                    a.n_op += b.n_op;
                    a.silent += b.silent;
                    a.loud += b.loud;
                    a.shared_catch += b.shared_catch;
                    a.indep_catch += b.indep_catch;
                    a.shared_fr += b.shared_fr;
                    a.indep_fr += b.indep_fr;
                    a
                });
            let pct = |c: usize| if res.silent > 0 { format!("{:.1}", 100.0 * c as f64 / res.silent as f64) } else { "-".to_string() };
            println!("{:<28} {:>7} {:>8} {:>6} {:>8} {:>8} {:>7}/{:<7}{}", name, res.n_op, res.silent, res.loud, pct(res.shared_catch), pct(res.indep_catch), res.shared_fr, res.indep_fr, if bug::is_common_mode(*b) { "   (common-mode: also planted in the independent path)" } else { "" });
            rows.push(json!({"workload": wname, "mutant": name, "specs": res.n_op, "silent_wrong": res.silent, "loud": res.loud, "shared_catch": res.shared_catch, "indep_catch": res.indep_catch, "shared_false_rej": res.shared_fr, "indep_false_rej": res.indep_fr, "common_mode": bug::is_common_mode(*b)}));
            for t in [&mut tot] {
                t.silent += res.silent;
                t.shared_catch += res.shared_catch;
                t.indep_catch += res.indep_catch;
                t.shared_fr += res.shared_fr;
                t.indep_fr += res.indep_fr;
            }
            if !bug::is_common_mode(*b) {
                tot_noncm.silent += res.silent;
                tot_noncm.shared_catch += res.shared_catch;
                tot_noncm.indep_catch += res.indep_catch;
            }
            if res.silent > 0 {
                killed.2 += 1;
                if res.shared_catch as f64 >= 0.9 * res.silent as f64 {
                    killed.0 += 1;
                }
                if res.indep_catch as f64 >= 0.9 * res.silent as f64 {
                    killed.1 += 1;
                }
            }
        }
        let p = |c: usize, n: usize| 100.0 * c as f64 / n.max(1) as f64;
        println!("TOTAL silent wrong {}: shared {:.1}%  independent {:.1}%   | excluding the {} common-mode mutants (silent {}): shared {:.1}%  independent {:.1}%", tot.silent, p(tot.shared_catch, tot.silent), p(tot.indep_catch, tot.silent), bug::ALL.iter().filter(|(b, _)| bug::is_common_mode(*b)).count(), tot_noncm.silent, p(tot_noncm.shared_catch, tot_noncm.silent), p(tot_noncm.indep_catch, tot_noncm.silent));
        println!("live mutants (>=1 silent wrong): {}; killed (>=90% of silent wrong caught): shared {} / independent {}; false rejects (clean outputs rejected) shared {} / independent {} over {} mutant runs", killed.2, killed.0, killed.1, tot.shared_fr, tot.indep_fr, bug::ALL.len());
        let _ = &tot_noncm.n_op;
    }
    let out = format!("results/h18_verify_{tag}.json");
    let _ = std::fs::write(&out, serde_json::to_string_pretty(&rows).unwrap_or_default());
    println!("wrote {out}");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_default();
    let seeds = parse_range(&args, "--seeds", (31, 60));
    let tag = arg(&args, "--tag", "held");
    match mode.as_str() {
        "selfcheck" => selfcheck(&seeds),
        "mutate" => mutate(&seeds, &tag),
        _ => eprintln!("usage: h18_verify selfcheck|mutate --seeds A-B [--tag t]"),
    }
}
