//! g21_verify (W22): is the verifier independent of the evaluator it checks?
//!   selfcheck --seeds 1-6        mutant 0 == real tools / real Verifier; independent verifier: false rejects, wrong-claim rejects, coverage
//!   mutate    --seeds 1-8 --tag  mutation testing of the evaluator (19 seeded bugs): how many wrong tool outputs does each verifier catch?
mod g21_common;

use g21_common::evalx::{self, MCalc, MCsp, MIdentity, MUnits, MVerifier, MUTANTS};
use g21_common::g13::{self, Kind, WorldCfg, KINDS, TOOL_NAMES};
use g21_common::indep::IVerifier;
use neural_mesh::tools::{LookupTool, Registry, Sandbox, Tool};
use rayon::prelude::*;
use serde_json::{json, Value};

fn run_tool(kind: Kind, spec: &str, lookup: &LookupTool) -> Result<String, neural_mesh::tools::ToolError> {
    match kind {
        Kind::Arith => MCalc.call(spec),
        Kind::Units => MUnits.call(spec),
        Kind::Csp => MCsp.call(spec),
        Kind::Sym => MIdentity::default().call(spec),
        Kind::Prog => Sandbox::default().call(spec),
        Kind::Lookup => lookup.call(spec),
        Kind::Qual => unreachable!(),
    }
}

#[derive(Clone, Default, Debug)]
struct Acc {
    n: usize,
    wrong_ok: usize,
    visible_err: usize,
    caught_shared: usize,
    caught_indep: usize,
    caught_meta: usize,
    caught_mo: usize,
    fr_mo: usize,
    fr_shared: usize,
    fr_indep: usize,
    fr_meta: usize,
    right_ok: usize,
    cost_shared: f64,
    cost_indep: f64,
    cost_meta: f64,
    ver_n: usize,
    indep_cov: usize,
}
impl Acc {
    fn merge(&mut self, o: &Acc) {
        self.n += o.n;
        self.wrong_ok += o.wrong_ok;
        self.visible_err += o.visible_err;
        self.caught_shared += o.caught_shared;
        self.caught_indep += o.caught_indep;
        self.caught_meta += o.caught_meta;
        self.caught_mo += o.caught_mo;
        self.fr_mo += o.fr_mo;
        self.fr_shared += o.fr_shared;
        self.fr_indep += o.fr_indep;
        self.fr_meta += o.fr_meta;
        self.right_ok += o.right_ok;
        self.cost_shared += o.cost_shared;
        self.cost_indep += o.cost_indep;
        self.cost_meta += o.cost_meta;
        self.ver_n += o.ver_n;
        self.indep_cov += o.indep_cov;
    }
}

fn accept(r: &Result<String, neural_mesh::tools::ToolError>) -> bool {
    matches!(r, Ok(s) if s == "accept")
}

/// run all (non-qual, non-dup) canonical tasks of worlds A and B for the given seeds under `mutant`
fn workload(seeds: &[u64], mutant: u8) -> Vec<Acc> {
    let per_seed: Vec<Vec<Acc>> = seeds
        .par_iter()
        .map(|&sd| {
            evalx::set_mutant(mutant);
            let mut accs = vec![Acc::default(); 7];
            for cfg in [WorldCfg::world_a(), WorldCfg::world_b()] {
                let w = g13::gen_world(&cfg, sd);
                let sv = MVerifier::new(w.lookup.clone());
                let iv = IVerifier::new(w.lookup.clone(), false);
                let ivm = IVerifier::new(w.lookup.clone(), true);
                let ivo = IVerifier::meta_only(w.lookup.clone());
                for t in &w.tasks {
                    if t.dup || t.kind == Kind::Qual {
                        continue;
                    }
                    let a = &mut accs[t.kind.idx()];
                    a.n += 1;
                    let tool = TOOL_NAMES[t.kind.tool().unwrap()];
                    let out = match run_tool(t.kind, &t.spec, &w.lookup) {
                        Ok(o) => o,
                        Err(_) => {
                            a.visible_err += 1;
                            continue;
                        }
                    };
                    let right = t.truth.is_correct(&out);
                    let vs = match sv.verify(tool, &t.spec, &out) {
                        Ok((s, c)) => {
                            a.cost_shared += c as f64;
                            Ok(s)
                        }
                        Err(e) => Err(e),
                    };
                    let (vi, ci, cov) = iv.verify(tool, &t.spec, &out);
                    let (vm, cm, _) = ivm.verify(tool, &t.spec, &out);
                    let (vo, _, _) = ivo.verify(tool, &t.spec, &out);
                    a.cost_indep += ci as f64;
                    a.cost_meta += cm as f64;
                    a.ver_n += 1;
                    a.indep_cov += cov as usize;
                    if right {
                        a.right_ok += 1;
                        a.fr_shared += (!accept(&vs)) as usize;
                        a.fr_indep += (!accept(&vi)) as usize;
                        a.fr_meta += (!accept(&vm)) as usize;
                        a.fr_mo += (!accept(&vo)) as usize;
                    } else {
                        a.wrong_ok += 1;
                        a.caught_shared += (!accept(&vs)) as usize;
                        a.caught_indep += (!accept(&vi)) as usize;
                        a.caught_meta += (!accept(&vm)) as usize;
                        a.caught_mo += (!accept(&vo)) as usize;
                    }
                }
            }
            evalx::set_mutant(0);
            accs
        })
        .collect();
    let mut tot = vec![Acc::default(); 7];
    for s in &per_seed {
        for k in 0..7 {
            tot[k].merge(&s[k]);
        }
    }
    tot
}

fn pct(a: usize, b: usize) -> String {
    if b == 0 { "   n/a".into() } else { format!("{:5.1}%", 100.0 * a as f64 / b as f64) }
}

fn selfcheck(args: &[String]) {
    let seeds = g13::parse_range(args, "--seeds", (1, 6));
    println!("selfcheck: mutant 0 (unmutated copy) vs the REAL tools in neural_mesh::tools, seeds {}-{}", seeds[0], seeds[seeds.len() - 1]);
    let (mut n, mut same_out, mut same_ver_ok, mut same_ver_bad, mut n_bad) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let acc = workload(&seeds, 0);
    let mut jrows = Vec::new();
    for cfg in [WorldCfg::world_a(), WorldCfg::world_b()] {
        for &sd in &seeds {
            let w = g13::gen_world(&cfg, sd);
            let mut reg = Registry::with_defaults(w.lookup.clone(), false);
            let sv = MVerifier::new(w.lookup.clone());
            for t in &w.tasks {
                if t.dup || t.kind == Kind::Qual {
                    continue;
                }
                n += 1;
                let tool = t.kind.tool().unwrap();
                let real = reg.call_idx(tool, &t.spec).result;
                let mine = run_tool(t.kind, &t.spec, &w.lookup);
                if real == mine {
                    same_out += 1;
                }
                if let Ok(out) = &real {
                    let vi = format!("{}\n{}\n{}", TOOL_NAMES[tool], t.spec, out);
                    let rv = reg.call_idx(6, &vi).result;
                    let mv = sv.verify(TOOL_NAMES[tool], &t.spec, out).map(|x| x.0);
                    if rv == mv {
                        same_ver_ok += 1;
                    }
                    let wa = g13::wrong_answer(t, t.salt);
                    if !t.truth.is_correct(&wa) {
                        n_bad += 1;
                        let vi = format!("{}\n{}\n{}", TOOL_NAMES[tool], t.spec, wa);
                        let rv = reg.call_idx(6, &vi).result;
                        let mv = sv.verify(TOOL_NAMES[tool], &t.spec, &wa).map(|x| x.0);
                        if rv == mv {
                            same_ver_bad += 1;
                        }
                    }
                }
            }
        }
    }
    println!("copy == real tools on {same_out}/{n} canonical specs; shared-verifier copy == real Verifier on correct outputs {same_ver_ok}/{n}, on wrong claims {same_ver_bad}/{n_bad}");
    let mut ok = same_out == n && same_ver_ok == n && same_ver_bad == n_bad;
    println!("\nindependent verifier on the CLEAN evaluator (no bugs):");
    println!("{:<8} {:>6} {:>10} {:>12} {:>12} {:>10}", "kind", "n", "indep cov", "false rej", "false rej+M", "cost ind/shr");
    for k in KINDS {
        if k == Kind::Qual {
            continue;
        }
        let a = &acc[k.idx()];
        println!(
            "{:<8} {:>6} {:>10} {:>12} {:>12} {:>5.2}/{:<5.2}",
            k.label(),
            a.ver_n,
            pct(a.indep_cov, a.ver_n),
            format!("{}/{}", a.fr_indep, a.right_ok),
            format!("{}/{}", a.fr_meta, a.right_ok),
            a.cost_indep / a.ver_n.max(1) as f64,
            a.cost_shared / a.ver_n.max(1) as f64
        );
        ok &= a.fr_indep == 0 && a.fr_meta == 0 && a.fr_shared == 0;
        jrows.push(json!({"kind": k.label(), "n": a.ver_n, "indep_coverage": a.indep_cov, "false_reject_indep": a.fr_indep, "false_reject_meta": a.fr_meta, "false_reject_shared": a.fr_shared, "right": a.right_ok, "cost_indep": a.cost_indep, "cost_shared": a.cost_shared, "cost_meta": a.cost_meta}));
    }
    // wrong claims (the world's wrong_answer) vs the independent verifier
    let (mut wn, mut wrej, mut wrej_m) = (0usize, 0usize, 0usize);
    for cfg in [WorldCfg::world_a(), WorldCfg::world_b()] {
        for &sd in &seeds {
            let w = g13::gen_world(&cfg, sd);
            let iv = IVerifier::new(w.lookup.clone(), false);
            let ivm = IVerifier::new(w.lookup.clone(), true);
            for t in &w.tasks {
                if t.dup || t.kind == Kind::Qual {
                    continue;
                }
                let wa = g13::wrong_answer(t, t.salt);
                if t.truth.is_correct(&wa) {
                    continue;
                }
                wn += 1;
                let tool = TOOL_NAMES[t.kind.tool().unwrap()];
                wrej += (!accept(&iv.verify(tool, &t.spec, &wa).0)) as usize;
                wrej_m += (!accept(&ivm.verify(tool, &t.spec, &wa).0)) as usize;
            }
        }
    }
    println!("wrong claims rejected by the independent verifier: {wrej}/{wn} (with metamorphic checks {wrej_m}/{wn})");
    // ---- differential test of the REAL tools::Calc (not the copy) on the stress corpus (minimal parentheses, truth from the generating tree)
    {
        let (mut sn, mut real_ok, mut real_err, mut ind_acc, mut ind_rej_wrong, mut wrong_n) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let lookup = LookupTool::new();
        let iv = IVerifier::new(lookup.clone(), true);
        for &sd in &seeds {
            for (spec, truth) in stress_specs(sd, 1500) {
                sn += 1;
                match neural_mesh::tools::Calc.call(&spec) {
                    Ok(out) => {
                        if neural_mesh::tools::Rat::parse(&out).map(|x| x == truth).unwrap_or(false) {
                            real_ok += 1;
                        }
                        ind_acc += accept(&iv.verify("calc", &spec, &out).0) as usize;
                        // a wrong claim (truth + 1) must be rejected
                        if let Ok(t1) = truth.add(neural_mesh::tools::Rat::int(1)) {
                            wrong_n += 1;
                            ind_rej_wrong += (!accept(&iv.verify("calc", &spec, &t1.to_string()).0)) as usize;
                        }
                    }
                    Err(_) => real_err += 1,
                }
            }
        }
        println!("DIFFERENTIAL TEST of the real tools::Calc on {sn} minimal-parenthesis random expressions: output == generating-tree truth on {real_ok}/{sn} (tool errors {real_err}); independent+metamorphic verifier accepts {ind_acc}/{}; rejects wrong claims {ind_rej_wrong}/{wrong_n}", sn - real_err);
        ok &= real_ok + real_err == sn && real_err == 0 && ind_acc == sn && ind_rej_wrong == wrong_n;
    }
    ok &= wrej == wn && wrej_m == wn;
    println!("{}", if ok { "SELFCHECK PASS" } else { "SELFCHECK FAIL" });
    std::fs::write(g13::arg(args, "--out", "results/g21_verify_selfcheck.json"), serde_json::to_string_pretty(&json!({"seeds": seeds, "pass": ok, "n": n, "copy_equals_real": same_out, "rows": jrows, "wrong_claims": wn, "wrong_rejected_indep": wrej, "wrong_rejected_meta": wrej_m})).unwrap()).unwrap();
    if !ok {
        std::process::exit(1);
    }
}

fn mutate(args: &[String]) {
    let seeds = g13::parse_range(args, "--seeds", (1, 8));
    let tag = g13::arg(args, "--tag", "mut");
    println!("MUTATION TESTING of the evaluator: {} seeded bugs, workload = canonical specs of worlds A+B, seeds {}-{}", MUTANTS.len(), seeds[0], seeds[seeds.len() - 1]);
    println!("catch = share of WRONG tool outputs (that the tool returned as Ok) the verifier rejects; 'visible' = tool errored (visible failure, not silent)");
    println!("{:<3} {:<18} {:>6} {:>7} {:>7} | {:>8} {:>8} {:>8} {:>9} | {:>5} {:>5} {:>5} {:>5}", "id", "mutant", "n", "wrongOk", "visible", "shared", "indep", "indep+M", "meta-only", "frS", "frI", "frM", "frO");
    let mut rows = Vec::new();
    let mut tot_wrong = 0usize;
    let (mut ts, mut ti, mut tm) = (0usize, 0usize, 0usize);
    let (mut live, mut killed_s, mut killed_i, mut killed_m) = (0usize, 0usize, 0usize, 0usize);
    for (id, name, desc) in MUTANTS {
        let acc = workload(&seeds, *id);
        let mut all = Acc::default();
        for a in &acc {
            all.merge(a);
        }
        println!(
            "{:<3} {:<18} {:>6} {:>7} {:>7} | {:>8} {:>8} {:>8} {:>9} | {:>5} {:>5} {:>5} {:>5}",
            id,
            name,
            all.n,
            all.wrong_ok,
            all.visible_err,
            pct(all.caught_shared, all.wrong_ok),
            pct(all.caught_indep, all.wrong_ok),
            pct(all.caught_meta, all.wrong_ok),
            pct(all.caught_mo, all.wrong_ok),
            all.fr_shared,
            all.fr_indep,
            all.fr_meta,
            all.fr_mo
        );
        let kinds: Vec<Value> = KINDS
            .iter()
            .filter(|k| **k != Kind::Qual)
            .map(|k| {
                let a = &acc[k.idx()];
                json!({"kind": k.label(), "n": a.n, "wrong_ok": a.wrong_ok, "visible_err": a.visible_err, "caught_shared": a.caught_shared, "caught_indep": a.caught_indep, "caught_meta": a.caught_meta})
            })
            .collect();
        if all.wrong_ok + all.visible_err > 0 {
            live += 1;
            // "killed" if the verifier catches at least 90% of the silent wrong outputs (or all manifestations are visible errors)
            let thr = |c: usize| all.wrong_ok == 0 || c as f64 >= 0.9 * all.wrong_ok as f64;
            killed_s += thr(all.caught_shared) as usize;
            killed_i += thr(all.caught_indep) as usize;
            killed_m += thr(all.caught_meta) as usize;
        }
        tot_wrong += all.wrong_ok;
        ts += all.caught_shared;
        ti += all.caught_indep;
        tm += all.caught_meta;
        rows.push(json!({"id": id, "mutant": name, "desc": desc, "n": all.n, "wrong_ok": all.wrong_ok, "visible_err": all.visible_err, "caught_shared": all.caught_shared, "caught_indep": all.caught_indep, "caught_meta": all.caught_meta, "caught_meta_only": all.caught_mo, "false_reject_meta_only": all.fr_mo,
            "false_reject_shared": all.fr_shared, "false_reject_indep": all.fr_indep, "false_reject_meta": all.fr_meta, "right_ok": all.right_ok, "by_kind": kinds}));
    }
    // ---- stress corpus (calc only, minimal parentheses)
    println!("\nSTRESS CORPUS (random calc expressions, minimal parentheses, ground truth from the generating tree; calc tool only), 1500 per seed");
    println!("{:<3} {:<18} {:>6} {:>7} {:>7} | {:>8} {:>8} {:>8} {:>9} | {:>5} {:>5} {:>5} {:>5}", "id", "mutant", "n", "wrongOk", "visible", "shared", "indep", "indep+M", "meta-only", "frS", "frI", "frM", "frO");
    let mut srows = Vec::new();
    let (mut sw, mut ss, mut si, mut sm, mut so) = (0usize, 0usize, 0usize, 0usize, 0usize);
    for (id, name, _desc) in std::iter::once(&(0u8, "none (clean)", "")).chain(MUTANTS.iter()) {
        let a = stress_workload(&seeds, *id, 1500);
        println!("{:<3} {:<18} {:>6} {:>7} {:>7} | {:>8} {:>8} {:>8} {:>9} | {:>5} {:>5} {:>5} {:>5}", id, name, a.n, a.wrong_ok, a.visible_err, pct(a.caught_shared, a.wrong_ok), pct(a.caught_indep, a.wrong_ok), pct(a.caught_meta, a.wrong_ok), pct(a.caught_mo, a.wrong_ok), a.fr_shared, a.fr_indep, a.fr_meta, a.fr_mo);
        sw += a.wrong_ok;
        ss += a.caught_shared;
        si += a.caught_indep;
        sm += a.caught_meta;
        so += a.caught_mo;
        srows.push(json!({"id": id, "mutant": name, "n": a.n, "wrong_ok": a.wrong_ok, "visible_err": a.visible_err, "caught_shared": a.caught_shared, "caught_indep": a.caught_indep, "caught_meta": a.caught_meta, "caught_meta_only": a.caught_mo, "false_reject_shared": a.fr_shared, "false_reject_indep": a.fr_indep, "false_reject_meta": a.fr_meta, "false_reject_meta_only": a.fr_mo}));
    }
    println!("STRESS TOTAL silent wrong outputs {sw}: caught by shared {} / independent {} / independent+metamorphic {} / metamorphic-only {}", pct(ss, sw), pct(si, sw), pct(sm, sw), pct(so, sw));
    println!("\nTOTAL silent wrong outputs over all mutants: {tot_wrong}; caught by shared {} / independent {} / independent+metamorphic {}", pct(ts, tot_wrong), pct(ti, tot_wrong), pct(tm, tot_wrong));
    println!("mutants that manifest on this workload: {live}/{}; 'killed' (>=90% of silent wrong outputs caught): shared {killed_s}, independent {killed_i}, independent+metamorphic {killed_m}", MUTANTS.len());
    std::fs::write(
        g13::arg(args, "--out", &format!("results/g21_verify_{tag}.json")),
        serde_json::to_string_pretty(&json!({"seeds": seeds, "mutants": rows, "total_wrong": tot_wrong, "caught_shared": ts, "caught_indep": ti, "caught_meta": tm, "stress": srows, "stress_total_wrong": sw, "stress_caught_shared": ss, "stress_caught_indep": si, "stress_caught_meta": sm, "stress_caught_meta_only": so, "live_mutants": live, "killed_shared": killed_s, "killed_indep": killed_i, "killed_meta": killed_m})).unwrap(),
    )
    .unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("selfcheck") => selfcheck(&args),
        Some("mutate") => mutate(&args),
        _ => {
            eprintln!("usage: g21_verify selfcheck|mutate [--seeds a-b] [--tag t] [--out path]");
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// stress corpus: random calc expressions with MINIMAL parentheses (what a differently written extractor or an agent
// would produce); ground truth from the generating tree (never from any evaluator under test)

use neural_mesh::router::Rng;
use neural_mesh::tools::Rat;

#[derive(Clone, Debug)]
enum Tr {
    Leaf(String, Rat),
    Neg(Box<Tr>),
    Bin(char, Box<Tr>, Box<Tr>),
}
fn tprec(t: &Tr) -> u8 {
    match t {
        Tr::Leaf(..) => 9,
        Tr::Neg(_) => 3,
        Tr::Bin(o, ..) => match o {
            '+' | '-' => 1,
            '*' | '/' => 2,
            _ => 4,
        },
    }
}
fn tval(t: &Tr) -> Option<Rat> {
    match t {
        Tr::Leaf(_, r) => Some(*r),
        Tr::Neg(a) => Some(tval(a)?.neg()),
        Tr::Bin(o, a, b) => {
            let (x, y) = (tval(a)?, tval(b)?);
            match o {
                '+' => x.add(y).ok(),
                '-' => x.sub(y).ok(),
                '*' => x.mul(y).ok(),
                '/' => x.div(y).ok(),
                _ => {
                    if !y.is_int() {
                        return None;
                    }
                    x.pow_i(y.numer()).ok()
                }
            }
        }
    }
}
fn tshow(t: &Tr) -> String {
    match t {
        Tr::Leaf(s, _) => s.clone(),
        Tr::Neg(a) => {
            let s = tshow(a);
            if tprec(a) < 3 { format!("-({s})") } else { format!("-{s}") }
        }
        Tr::Bin(o, a, b) => {
            let p = tprec(t);
            let (l, r) = (tshow(a), tshow(b));
            let (lp, rp) = if *o == '^' {
                // left operand of ^ must be atomic; the exponent here is always a small integer leaf
                (tprec(a) <= 4, false)
            } else {
                // left-associative operators: the right child needs parentheses at equal precedence (a-(b-c), a/(b*c), a+(b+c) kept exact)
                (tprec(a) < p, tprec(b) <= p && !matches!(**b, Tr::Leaf(..)))
            };
            let l = if lp { format!("({l})") } else { l };
            let r = if rp { format!("({r})") } else { r };
            format!("{l}{o}{r}")
        }
    }
}
fn tgen(r: &mut Rng, depth: usize) -> Tr {
    if depth == 0 || r.f64() < 0.22 {
        let u = r.f64();
        if u < 0.65 {
            let v = 1 + r.below(20) as i128;
            Tr::Leaf(v.to_string(), Rat::int(v))
        } else {
            let fd = 1 + r.below(3);
            let ip = r.below(30);
            let mut frac = String::new();
            for _ in 0..fd {
                frac.push(char::from_digit(r.below(10) as u32, 10).unwrap());
            }
            if frac.chars().all(|c| c == '0') {
                frac = "5".repeat(fd);
            }
            let s = format!("{ip}.{frac}");
            Tr::Leaf(s.clone(), Rat::parse(&s).unwrap())
        }
    } else {
        let u = r.f64();
        let t = if u < 0.28 {
            Tr::Bin('+', Box::new(tgen(r, depth - 1)), Box::new(tgen(r, depth - 1)))
        } else if u < 0.52 {
            Tr::Bin('-', Box::new(tgen(r, depth - 1)), Box::new(tgen(r, depth - 1)))
        } else if u < 0.74 {
            Tr::Bin('*', Box::new(tgen(r, depth - 1)), Box::new(tgen(r, depth - 1)))
        } else if u < 0.9 {
            Tr::Bin('/', Box::new(tgen(r, depth - 1)), Box::new(tgen(r, depth - 1)))
        } else {
            let e = 2 + r.below(2) as i128;
            Tr::Bin('^', Box::new(tgen(r, depth - 1)), Box::new(Tr::Leaf(e.to_string(), Rat::int(e))))
        };
        if r.f64() < 0.12 { Tr::Neg(Box::new(t)) } else { t }
    }
}

/// (spec, truth) pairs
fn stress_specs(seed: u64, n: usize) -> Vec<(String, Rat)> {
    let mut r = Rng::new(seed.wrapping_mul(7777).wrapping_add(99));
    let mut out = Vec::new();
    while out.len() < n {
        let d = 1 + r.below(4);
        let t = tgen(&mut r, d);
        if let Some(v) = tval(&t) {
            if v.numer().abs() < 10i128.pow(25) && v.denom() < 10i128.pow(25) {
                out.push((tshow(&t), v));
            }
        }
    }
    out
}

fn stress_workload(seeds: &[u64], mutant: u8, n: usize) -> Acc {
    let per: Vec<Acc> = seeds
        .par_iter()
        .map(|&sd| {
            evalx::set_mutant(mutant);
            let mut a = Acc::default();
            let lookup = LookupTool::new();
            let sv = MVerifier::new(lookup.clone());
            let iv = IVerifier::new(lookup.clone(), false);
            let ivm = IVerifier::new(lookup.clone(), true);
            let ivo = IVerifier::meta_only(lookup);
            for (spec, truth) in stress_specs(sd, n) {
                a.n += 1;
                let out = match MCalc.call(&spec) {
                    Ok(o) => o,
                    Err(_) => {
                        a.visible_err += 1;
                        continue;
                    }
                };
                let right = Rat::parse(&out).map(|x| x == truth).unwrap_or(false);
                let vs = sv.verify("calc", &spec, &out).map(|x| x.0);
                let (vi, _, _) = iv.verify("calc", &spec, &out);
                let (vm, _, _) = ivm.verify("calc", &spec, &out);
                let (vo, _, _) = ivo.verify("calc", &spec, &out);
                a.ver_n += 1;
                if right {
                    a.right_ok += 1;
                    a.fr_shared += (!accept(&vs)) as usize;
                    a.fr_indep += (!accept(&vi)) as usize;
                    a.fr_meta += (!accept(&vm)) as usize;
                    a.fr_mo += (!accept(&vo)) as usize;
                } else {
                    a.wrong_ok += 1;
                    a.caught_shared += (!accept(&vs)) as usize;
                    a.caught_indep += (!accept(&vi)) as usize;
                    a.caught_meta += (!accept(&vm)) as usize;
                    a.caught_mo += (!accept(&vo)) as usize;
                }
            }
            evalx::set_mutant(0);
            a
        })
        .collect();
    let mut t = Acc::default();
    for a in &per {
        t.merge(a);
    }
    t
}
