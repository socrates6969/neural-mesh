//! h19_extract: extraction accuracy (per field type, layout difficulty, seen/unseen phrasing) + verifier catch / false-flag rates.
//!   dev   --seeds 1-8 [--world fin|paper|biz] [--layout N] [--show 30] [--unseen]    failure listing (DEVELOPMENT view; tune seeds only, Bank::Dev unless --unseen)
//!   tau   --seeds 1-8                                                                  choose abstention tau on the tune seeds (Bank::Dev docs only)
//!   eval  --seeds 31-60 --tag held [--tau T]                                           full accuracy + verifier tables (stratified corpus, both banks)
//!   inject --seeds 31-60 --tag held [--tau T]                                          verifier catch rate on injected error types
//!   adv   --seeds 31-60                                                                adversarial layouts (failure report)
mod h19_common;
use h19_common::cal::*;
use h19_common::genb::*;
use h19_common::gens::*;
use h19_common::scan::*;
use h19_common::verify::*;
use h19_common::*;
use serde_json::json;
use std::collections::BTreeMap;

pub const DOCS_PER_CELL: usize = 3;
pub const WORLDS3: [World; 3] = [World::Paper, World::Fin, World::Biz];

pub fn make_doc(w: World, seed: u64, idx: u64, bank: Bank, layout: u8) -> Doc {
    match w {
        World::Paper => gen_paper(seed, idx, bank, layout),
        World::Fin => gen_fin(seed, idx, bank, layout),
        World::Biz => gen_biz(seed, idx, bank, layout),
        World::NoStruct => gen_nostruct(seed, idx, bank),
        World::Tiny => gen_tiny(seed, idx),
    }
}
/// stratified corpus: (world, layout, bank) x DOCS_PER_CELL
fn corpus(seed: u64, banks: &[Bank]) -> Vec<Doc> {
    let mut v = Vec::new();
    let mut idx = 0u64;
    for w in WORLDS3 {
        for l in 0..5u8 {
            for &b in banks {
                for _ in 0..DOCS_PER_CELL {
                    v.push(make_doc(w, seed, 1000 + idx, b, l));
                    idx += 1;
                }
            }
        }
    }
    v
}
fn vcfg(args: &[String]) -> VCfg {
    let mut v = VCfg::default();
    if has(args, "--narrow") {
        v.broad = false;
    }
    if has(args, "--nofoot") {
        v.foot = false;
    }
    v
}
fn parse_world(s: &str) -> Option<World> {
    match s {
        "paper" => Some(World::Paper),
        "fin" => Some(World::Fin),
        "biz" => Some(World::Biz),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).cloned().unwrap_or_default();
    let seeds = parse_seeds(&arg(&args, "--seeds").unwrap_or("1-8".into()));
    let tag = arg(&args, "--tag").unwrap_or("x".into());
    let tau: f64 = arg(&args, "--tau").and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        // frozen tau from the downstream tune step (results/h19_tune.json), else the standalone calibration, else 0.5
        std::fs::read_to_string("results/h19_tune.json").ok().and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok()).and_then(|v| v["tau"].as_f64()).or_else(|| std::fs::read_to_string("results/h19_tau.txt").ok().and_then(|s| s.trim().parse().ok())).unwrap_or(0.5)
    });
    match cmd.as_str() {
        "dev" => dev(&args, &seeds, tau),
        "tau" => tau_cmd(&seeds),
        "eval" => eval(&seeds, &tag, tau, &vcfg(&args)),
        "inject" => inject(&seeds, &tag, tau, &vcfg(&args)),
        "adv" => adv(&seeds, tau, &vcfg(&args)),
        _ => eprintln!("usage: dev|tau|eval|inject|adv"),
    }
}

fn dev(args: &[String], seeds: &[u64], tau: f64) {
    let wsel = arg(args, "--world").and_then(|s| parse_world(&s));
    let lsel: Option<u8> = arg(args, "--layout").and_then(|s| s.parse().ok());
    let show: usize = arg(args, "--show").and_then(|s| s.parse().ok()).unwrap_or(30);
    let bank = if has(args, "--unseen") { Bank::Unseen } else { Bank::Dev };
    let mut shown = 0;
    let mut tot: BTreeMap<FKind, Prf> = BTreeMap::new();
    for &s in seeds {
        for d in corpus(s, &[bank]) {
            if wsel.map_or(false, |w| w != d.world) || lsel.map_or(false, |l| l != d.layout) {
                continue;
            }
            let run = run_extract(d.world, &d.text, &ExCfg::default(), tau, &VCfg::default());
            let exts: Vec<Ext> = run.kept.iter().map(|&i| run.all[i].clone()).collect();
            tally(&d, &exts, &|_| true, &mut tot);
            // failures
            let mut fails: Vec<String> = Vec::new();
            for t in &d.facts {
                match exts.iter().find(|e| e.fact.key == t.key) {
                    None => {
                        let cand = run.all.iter().find(|e| e.fact.key == t.key);
                        fails.push(format!("MISS {} truth={} unit={} {}", t.key, t.value, t.unit, cand.map_or("(no candidate)".into(), |c| format!("(abstained conf {:.2} val {} unit {})", c.conf, c.fact.value, c.fact.unit))));
                    }
                    Some(e) => {
                        if !t.matches(&e.fact) {
                            fails.push(format!("WRONG {} truth={} {:?} got={} {:?} conf={:.2} notes={}", t.key, t.value, t.unit, e.fact.value, e.fact.unit, e.conf, e.notes));
                        }
                    }
                }
            }
            for e in &exts {
                if d.fact(&e.fact.key).is_none() {
                    fails.push(format!("SPURIOUS {} val={} conf={:.2}", e.fact.key, e.fact.value, e.conf));
                }
            }
            if has(args, "--vfy") {
                for (i, e) in exts.iter().enumerate() {
                    if run.vfy[i].flagged() && judge(&d, e) == Out::Correct {
                        fails.push(format!("FALSEFLAG {} val={} reasons={} repair={:?} span='{}'", e.fact.key, e.fact.value, run.vfy[i].reasons, run.vfy[i].repair, &d.text[e.fact.lo..e.fact.hi]));
                    }
                }
            }
            if !fails.is_empty() && shown < show {
                shown += 1;
                println!("===== seed {} world {} layout {} bank {:?}: {} failures", s, d.world.name(), d.layout, d.bank, fails.len());
                for f in fails.iter().take(8) {
                    println!("  {}", f);
                }
                if has(args, "--text") {
                    println!("{}", d.text);
                }
            }
        }
    }
    for (k, p) in &tot {
        println!("{:<11} P {:.3} R {:.3} F1 {:.3}  (tp {} fp {} fn {})", k.name(), p.p(), p.r(), p.f1(), p.tp, p.fp, p.fn_);
    }
}

fn tau_cmd(seeds: &[u64]) {
    let mut docs = Vec::new();
    for &s in seeds {
        docs.extend(corpus(s, &[Bank::Dev, Bank::Unseen]));
    }
    let exs: Vec<Vec<Ext>> = docs.iter().map(|d| extract(d.world, &d.text, &ExCfg::default())).collect();
    let items: Vec<(&Doc, &Vec<Ext>)> = docs.iter().zip(exs.iter()).collect();
    let (t, f) = choose_tau(&items);
    println!("tau* = {:.2} (micro-F1 {:.4}) on {} mixed-bank (dev+unseen) tune docs seeds {:?}..", t, f, docs.len(), seeds.first());
    let mut tau = 0.05;
    while tau < 0.96 {
        let mut p = Prf::default();
        for (d, ex) in &items {
            let mut by = BTreeMap::new();
            tally(d, ex, &|i| ex[i].conf >= tau, &mut by);
            for v in by.values() {
                p.add(v);
            }
        }
        println!("  tau {:.2}: P {:.3} R {:.3} F1 {:.3}", tau, p.p(), p.r(), p.f1());
        tau += 0.05;
    }
    std::fs::write("results/h19_tau.txt", format!("{:.2}\n", t)).unwrap();
}

#[derive(Default, Clone)]
struct Agg {
    prf: BTreeMap<FKind, Prf>,
    /// verified (flagged facts removed)
    prf_v: BTreeMap<FKind, Prf>,
    n_cand: usize,
    n_abst: usize,
    wrong_kept: usize,
    correct_kept: usize,
    caught: usize,
    false_flag: usize,
    flag_reasons: BTreeMap<u32, (usize, usize)>, // reason bit -> (flags on wrong, flags on correct)
    pairs: Vec<(f64, bool)>,
    repaired: usize,
    repaired_ok: usize,
    span_ok: usize,
    span_n: usize,
}
fn agg_doc(a: &mut Agg, d: &Doc, tau: f64, vc: &VCfg) {
    let run = run_extract(d.world, &d.text, &ExCfg::default(), tau, vc);
    let kept: Vec<Ext> = run.kept.iter().map(|&i| run.all[i].clone()).collect();
    a.n_cand += run.all.len();
    a.n_abst += run.all.len() - kept.len();
    tally(d, &kept, &|_| true, &mut a.prf);
    tally(d, &kept, &|i| !run.vfy[i].flagged(), &mut a.prf_v);
    for (i, e) in kept.iter().enumerate() {
        let o = judge(d, e);
        a.pairs.push((e.conf, o == Out::Correct));
        let fl = run.vfy[i].flagged();
        if o == Out::Correct {
            a.correct_kept += 1;
            if fl {
                a.false_flag += 1;
            }
        } else {
            a.wrong_kept += 1;
            if fl {
                a.caught += 1;
            }
        }
        for bit in [V_SPAN, V_SIGN, V_SCALE, V_FOOT, V_PLAUS, V_XCHK, V_COVER, V_AMBIG, V_FOOT_AMBIG] {
            if run.vfy[i].reasons & bit != 0 {
                let en = a.flag_reasons.entry(bit).or_insert((0, 0));
                if o == Out::Correct {
                    en.1 += 1;
                } else {
                    en.0 += 1;
                }
            }
        }
        if let (Some(r), Some(t)) = (run.vfy[i].repair, d.fact(&e.fact.key)) {
            a.repaired += 1;
            if close(r, t.value) && o != Out::Correct {
                a.repaired_ok += 1;
            }
        }
        if let Some(t) = d.fact(&e.fact.key) {
            if t.hi > t.lo {
                a.span_n += 1;
                let (lo, hi) = (e.fact.lo.max(t.lo), e.fact.hi.min(t.hi));
                if hi > lo && (hi - lo) as f64 >= 0.5 * (t.hi - t.lo) as f64 {
                    a.span_ok += 1;
                }
            }
        }
    }
}
fn prf_all(m: &BTreeMap<FKind, Prf>) -> Prf {
    let mut p = Prf::default();
    for v in m.values() {
        p.add(v);
    }
    p
}
fn reason_name(b: u32) -> &'static str {
    match b {
        V_SPAN => "span_mismatch",
        V_SIGN => "sign",
        V_SCALE => "scale",
        V_FOOT => "foot_explained",
        V_PLAUS => "plausibility",
        V_XCHK => "cross_check",
        V_COVER => "claim_coverage",
        V_AMBIG => "ambig_separator",
        V_FOOT_AMBIG => "foot_ambiguous",
        _ => "?",
    }
}
fn f3(x: f64) -> String {
    if x.is_nan() { "  n/a".into() } else { format!("{:.3}", x) }
}

fn eval(seeds: &[u64], tag: &str, tau: f64, vc: &VCfg) {
    // keyed aggregates: (bank, world, layout)
    let mut cells: BTreeMap<(u8, usize, u8), Agg> = BTreeMap::new();
    let mut per_seed_f1: BTreeMap<u8, Vec<f64>> = BTreeMap::new();
    for &s in seeds {
        let mut per_bank: BTreeMap<u8, Agg> = BTreeMap::new();
        for d in corpus(s, &[Bank::Dev, Bank::Unseen]) {
            let bk = if d.bank == Bank::Dev { 0 } else { 1 };
            agg_doc(cells.entry((bk, d.world.idx(), d.layout)).or_default(), &d, tau, vc);
            agg_doc(per_bank.entry(bk).or_default(), &d, tau, vc);
        }
        for (bk, a) in per_bank {
            per_seed_f1.entry(bk).or_default().push(prf_all(&a.prf).f1());
        }
    }
    println!("# h19_extract eval tag={} seeds={}..{} ({} seeds) tau={:.2}   SYNTHETIC; unseen bank written by the same author before the extractor was built (optimistic)", tag, seeds[0], seeds[seeds.len() - 1], seeds.len(), tau);
    let bname = ["dev-phrasing (seen)", "unseen-phrasing"];
    let mut summary = json!({});
    for bk in 0..2u8 {
        // by field type
        let mut ft: BTreeMap<FKind, Prf> = BTreeMap::new();
        let mut ftv: BTreeMap<FKind, Prf> = BTreeMap::new();
        for (k, a) in cells.iter().filter(|(k, _)| k.0 == bk) {
            let _ = k;
            for (kind, p) in &a.prf {
                ft.entry(*kind).or_default().add(p);
            }
            for (kind, p) in &a.prf_v {
                ftv.entry(*kind).or_default().add(p);
            }
        }
        println!("\n## {}: P/R/F1 per field type (extractor at tau; then after verifier removes flagged facts)", bname[bk as usize]);
        println!("{:<11} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}  (n truth)", "field", "P", "R", "F1", "P_v", "R_v", "F1_v");
        for kind in FKINDS {
            if let Some(p) = ft.get(&kind) {
                let pv = ftv.get(&kind).copied().unwrap_or_default();
                println!("{:<11} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}  ({})", kind.name(), f3(p.p()), f3(p.r()), f3(p.f1()), f3(pv.p()), f3(pv.r()), f3(pv.f1()), (p.tp + p.fn_) as i64);
            }
        }
        let (a_all, a_v) = (prf_all(&ft), prf_all(&ftv));
        println!("{:<11} {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}", "ALL", f3(a_all.p()), f3(a_all.r()), f3(a_all.f1()), f3(a_v.p()), f3(a_v.r()), f3(a_v.f1()));
        println!("\n## {}: F1 (R) by world x layout L0..L4", bname[bk as usize]);
        for w in WORLDS3 {
            let mut line = format!("{:<10}", w.name());
            for l in 0..5u8 {
                let a = &cells[&(bk, w.idx(), l)];
                let p = prf_all(&a.prf);
                line.push_str(&format!("  L{} {} ({})", l, f3(p.f1()), f3(p.r())));
            }
            println!("{}", line);
        }
        // verifier
        let (mut wr, mut cr, mut ca, mut ff, mut ab, mut nc) = (0, 0, 0, 0, 0, 0);
        let mut rs: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
        let (mut rep, mut repok, mut spn, mut spk) = (0, 0, 0, 0);
        let mut pairs = Vec::new();
        for (_, a) in cells.iter().filter(|(k, _)| k.0 == bk) {
            wr += a.wrong_kept;
            cr += a.correct_kept;
            ca += a.caught;
            ff += a.false_flag;
            ab += a.n_abst;
            nc += a.n_cand;
            rep += a.repaired;
            repok += a.repaired_ok;
            spn += a.span_n;
            spk += a.span_ok;
            for (b, (x, y)) in &a.flag_reasons {
                let e = rs.entry(*b).or_insert((0, 0));
                e.0 += x;
                e.1 += y;
            }
            pairs.extend(a.pairs.iter().cloned());
        }
        println!("\n## {}: abstain {}/{} candidates ({:.1}%), ECE(conf vs correct) {:.3}, span-exact(>=50% overlap) {:.3}", bname[bk as usize], ab, nc, 100.0 * ab as f64 / nc.max(1) as f64, ece(&pairs, 10), spk as f64 / spn.max(1) as f64);
        println!("   verifier: kept facts wrong {} / correct {}; CATCH rate (wrong flagged) {:.3}; FALSE-FLAG rate (correct flagged) {:.4}; repair proposals {} (correct on wrong facts: {})", wr, cr, ca as f64 / wr.max(1) as f64, ff as f64 / cr.max(1) as f64, rep, repok);
        for (b, (x, y)) in &rs {
            println!("     reason {:<16} flags on wrong {:>5} on correct {:>5}", reason_name(*b), x, y);
        }
        summary[bname[bk as usize]] = json!({"P": a_all.p(), "R": a_all.r(), "F1": a_all.f1(), "P_v": a_v.p(), "R_v": a_v.r(), "F1_v": a_v.f1(), "wrong_kept": wr, "correct_kept": cr, "catch": ca as f64 / wr.max(1) as f64, "false_flag": ff as f64 / cr.max(1) as f64, "abstain": ab as f64 / nc.max(1) as f64, "per_seed_f1": per_seed_f1.get(&bk).cloned().unwrap_or_default()});
    }
    let path = format!("results/h19_extract_{}.json", tag);
    std::fs::write(&path, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("\nwrote {}", path);
}

// ---------------------------------------------------------------- error injection
fn inject(seeds: &[u64], tag: &str, tau: f64, vc: &VCfg) {
    let types = ["scale x1000", "scale /1000", "sign flip", "digit transposition", "wrong cell (neighbour in row)", "decimal shift /10"];
    // (kind, type) -> (n, caught, repaired_correct)
    let mut res: BTreeMap<(FKind, usize), (usize, usize, usize)> = BTreeMap::new();
    let mut base_ff = (0usize, 0usize);
    for &s in seeds {
        for d in corpus(s, &[Bank::Dev, Bank::Unseen]) {
            let run = run_extract(d.world, &d.text, &ExCfg::default(), tau, vc);
            let kept: Vec<Ext> = run.kept.iter().map(|&i| run.all[i].clone()).collect();
            // only start from docs where the verifier is silent on everything (clean baseline), so each injected error is isolated
            let base = verify_doc(d.world, &d.text, &kept, vc);
            for (i, e) in kept.iter().enumerate() {
                if judge(&d, e) == Out::Correct {
                    base_ff.0 += 1;
                    if base[i].flagged() {
                        base_ff.1 += 1;
                    }
                }
            }
            let mut r = R::new(s, 77, d.facts.len() as u64, d.text.len() as u64);
            let cands: Vec<usize> = (0..kept.len()).filter(|&i| judge(&d, &kept[i]) == Out::Correct && matches!(kept[i].fact.kind, FKind::Line | FKind::Cell | FKind::Kpi | FKind::Amount | FKind::Clause) && !base[i].flagged() && kept[i].fact.value.abs() >= 10.0).collect();
            let mut picks = cands.clone();
            r.shuffle(&mut picks);
            picks.truncate(6);
            for &i in &picks {
                for (ti, _) in types.iter().enumerate() {
                    let mut ex = kept.clone();
                    let v = ex[i].fact.value;
                    let truth = v;
                    match ti {
                        0 => ex[i].fact.value = v * 1000.0,
                        1 => ex[i].fact.value = v / 1000.0,
                        2 => ex[i].fact.value = -v,
                        3 => {
                            let s_ = format!("{}", v.abs() as i64);
                            let b = s_.as_bytes();
                            let mut done = false;
                            for k in 0..b.len().saturating_sub(1) {
                                if b[k] != b[k + 1] {
                                    let mut c = b.to_vec();
                                    c.swap(k, k + 1);
                                    if c[0] != b'0' {
                                        ex[i].fact.value = String::from_utf8(c).unwrap().parse::<f64>().unwrap() * v.signum() * (v.abs() / (v.abs() as i64 as f64).max(1.0));
                                        done = true;
                                        break;
                                    }
                                }
                            }
                            if !done {
                                continue;
                            }
                        }
                        4 => {
                            // same row, other column
                            let o = (0..ex.len()).find(|&k| k != i && ex[k].row == ex[i].row && ex[k].fact.kind == ex[i].fact.kind && ex[k].fact.key != ex[i].fact.key && ex[i].row > 0);
                            let Some(o) = o else { continue };
                            ex[i].fact.value = ex[o].fact.value;
                            ex[i].fact.lo = ex[o].fact.lo;
                            ex[i].fact.hi = ex[o].fact.hi;
                        }
                        _ => ex[i].fact.value = v / 10.0,
                    }
                    if close(ex[i].fact.value, truth) {
                        continue;
                    }
                    let vf = verify_doc(d.world, &d.text, &ex, vc);
                    let en = res.entry((ex[i].fact.kind, ti)).or_insert((0, 0, 0));
                    en.0 += 1;
                    if vf[i].flagged() {
                        en.1 += 1;
                        if vf[i].repair.map_or(false, |r_| close(r_, truth)) {
                            en.2 += 1;
                        }
                    }
                }
            }
        }
    }
    println!("# h19 injected-error catch rates, tag={} seeds {}..{} tau={:.2} (each error injected ALONE into an otherwise verifier-silent doc; SYNTHETIC)", tag, seeds[0], seeds[seeds.len() - 1], tau);
    println!("baseline false-flag rate on correct facts (verifier-silent selection excluded): {}/{} = {:.4}", base_ff.1, base_ff.0, base_ff.1 as f64 / base_ff.0.max(1) as f64);
    let mut js = json!({});
    for (ti, t) in types.iter().enumerate() {
        let mut line = format!("{:<32}", t);
        let (mut n, mut c, mut r_) = (0, 0, 0);
        for kind in [FKind::Line, FKind::Cell, FKind::Kpi, FKind::Amount, FKind::Clause] {
            if let Some((a, b, cc)) = res.get(&(kind, ti)) {
                line.push_str(&format!("  {}: {}/{} ({:.2})", kind.name(), b, a, *b as f64 / (*a).max(1) as f64));
                n += a;
                c += b;
                r_ += cc;
            }
        }
        println!("{}   ALL catch {:.3} (n={}), repaired-to-truth {:.3}", line, c as f64 / n.max(1) as f64, n, r_ as f64 / n.max(1) as f64);
        js[*t] = json!({"n": n, "catch": c as f64 / n.max(1) as f64, "repair_ok": r_ as f64 / n.max(1) as f64});
    }
    std::fs::write(format!("results/h19_inject_{}.json", tag), serde_json::to_string_pretty(&js).unwrap()).unwrap();
}

// ---------------------------------------------------------------- adversarial layouts
fn adv(seeds: &[u64], tau: f64, vc: &VCfg) {
    println!("# h19 adversarial financial layouts (controls; failure is the expected result) tau={:.2}", tau);
    let names = ["side-by-side two statements", "wrapped labels", "numbers split across lines"];
    for kind in 1..=3u8 {
        let (mut p, mut pv) = (Prf::default(), Prf::default());
        let (mut wr, mut ca) = (0, 0);
        for &s in seeds {
            for k in 0..6u64 {
                let d = gen_fin_adv(s, 500 + k, kind);
                let run = run_extract(World::Fin, &d.text, &ExCfg::default(), tau, vc);
                let kept: Vec<Ext> = run.kept.iter().map(|&i| run.all[i].clone()).collect();
                let mut by = BTreeMap::new();
                tally(&d, &kept, &|_| true, &mut by);
                p.add(&prf_all(&by));
                let mut by2 = BTreeMap::new();
                tally(&d, &kept, &|i| !run.vfy[i].flagged(), &mut by2);
                pv.add(&prf_all(&by2));
                for (i, e) in kept.iter().enumerate() {
                    if judge(&d, e) != Out::Correct {
                        wr += 1;
                        if run.vfy[i].flagged() {
                            ca += 1;
                        }
                    }
                }
            }
        }
        println!("{:<30} P {} R {} F1 {} | after verifier P {} R {} | wrong kept {} caught {} ({:.2}); silent-wrong rate among emitted {:.3}", names[kind as usize - 1], f3(p.p()), f3(p.r()), f3(p.f1()), f3(pv.p()), f3(pv.r()), wr, ca, ca as f64 / (wr as f64).max(1.0), (wr - ca) as f64 / ((p.tp + p.fp) as f64).max(1.0));
    }
}
