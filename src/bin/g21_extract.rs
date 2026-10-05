//! g21_extract (W22): extraction accuracy of the deterministic NL -> spec extractor, by phrasing difficulty.
//!   dev  --seeds 1-8 [--show 40] [--unseen]   print failures (development view; tune seeds only)
//!   eval --seeds 31-60 --tag held             accuracy tables (levels L0-L4 x bank dev/unseen x kind), calibration
mod g21_common;

use g21_common::g13::{self, Kind, WorldCfg, KINDS};
use g21_common::nlgen::NlCfg;
use g21_common::pipe::{build_xworld, XWorld};
use neural_mesh::tools::Registry;
use serde_json::{json, Value};

fn norm(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Out {
    Correct,
    Abstain,
    Wrong,
}

/// (outcome, exact string match) of the extractor on task i, judged by running the REAL tools
fn judge(xw: &XWorld, reg: &mut Registry, i: usize) -> (Out, bool) {
    let t = &xw.w.tasks[i];
    let ex = &xw.ex[i];
    let Some(spec) = &ex.spec else { return (Out::Abstain, false) };
    if t.kind == Kind::Qual {
        return (Out::Wrong, false);
    }
    let exact = norm(spec) == norm(&t.spec);
    let Some(tool) = ex.kind.and_then(|k| k.tool()) else { return (Out::Wrong, exact) };
    if ex.kind != Some(t.kind) {
        return (Out::Wrong, exact);
    }
    match reg.call_idx(tool, spec).result {
        Ok(out) => (if t.truth.is_correct(&out) { Out::Correct } else { Out::Wrong }, exact),
        Err(_) => (Out::Wrong, exact),
    }
}

#[derive(Clone, Default)]
struct Cell {
    n: usize,
    correct: usize,
    abstain: usize,
    wrong: usize,
    wrong_conf: usize, // wrong with conf >= 0.6
    exact: usize,
    fp_conf: usize,
}
impl Cell {
    fn add(&mut self, o: Out, exact: bool, conf: f64) {
        self.n += 1;
        match o {
            Out::Correct => self.correct += 1,
            Out::Abstain => self.abstain += 1,
            Out::Wrong => {
                self.wrong += 1;
                if conf >= 0.6 {
                    self.wrong_conf += 1;
                }
            }
        }
        if exact {
            self.exact += 1;
        }
    }
    fn f(&self, x: usize) -> f64 {
        if self.n == 0 { f64::NAN } else { x as f64 / self.n as f64 }
    }
    fn j(&self) -> Value {
        json!({"n": self.n, "sem_correct": self.f(self.correct), "abstain": self.f(self.abstain), "wrong": self.f(self.wrong), "wrong_conf": self.f(self.wrong_conf), "exact": self.f(self.exact)})
    }
    fn s(&self) -> String {
        format!("n={:>5} ok {:>5.3} abst {:>5.3} wrong {:>5.3} wrong@conf>=.6 {:>5.3} exact {:>5.3}", self.n, self.f(self.correct), self.f(self.abstain), self.f(self.wrong), self.f(self.wrong_conf), self.f(self.exact))
    }
}

fn args_range(args: &[String], key: &str, d: (u64, u64)) -> Vec<u64> {
    g13::parse_range(args, key, d)
}

fn worlds() -> Vec<(WorldCfg, NlCfg)> {
    vec![(WorldCfg::world_a(), NlCfg::world_a()), (WorldCfg::world_b(), NlCfg::world_b())]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().cloned().unwrap_or_default();
    let seeds = args_range(&args, "--seeds", (1, 8));
    match mode.as_str() {
        "dev" => dev(&args, &seeds),
        "ex" => {
            for t in args.iter().skip(1) {
                let e = g21_common::extract::extract(t);
                println!("{t:?}
   -> {:?} conf {:.2} kind {:?} fuzzy {} unused {} ncand {}", e.spec, e.conf, e.kind, e.fuzzy, e.unused, e.n_cands);
            }
        }
        "eval" => eval(&args, &seeds),
        _ => {
            eprintln!("usage: g21_extract dev|eval --seeds a-b [--tag t] [--show N] [--unseen]");
            std::process::exit(2);
        }
    }
}

fn dev(args: &[String], seeds: &[u64]) {
    let show: usize = g13::arg(args, "--show", "30").parse().unwrap();
    let want_unseen = args.iter().any(|a| a == "--unseen");
    let kind_filter = g13::arg(args, "--kind", "");
    for (cfg, ncfg) in worlds() {
        let mut shown = 0;
        let mut per: std::collections::HashMap<String, Cell> = Default::default();
        for &sd in seeds {
            let xw = build_xworld(&cfg, &ncfg, sd);
            let mut reg = Registry::with_defaults(xw.w.lookup.clone(), false);
            for i in 0..xw.w.tasks.len() {
                let t = &xw.w.tasks[i];
                if t.dup || xw.nl[i].unseen != want_unseen {
                    continue;
                }
                let (o, ex) = judge(&xw, &mut reg, i);
                per.entry(t.kind.label().into()).or_default().add(o, ex, xw.ex[i].conf);
                if o != Out::Correct && !(o == Out::Abstain && t.kind == Kind::Qual) && shown < show && (kind_filter.is_empty() || kind_filter == t.kind.label()) && xw.nl[i].level <= 3 {
                    shown += 1;
                    println!("[{} seed {} L{} {:?}] {}\n    text : {}\n    truth: {}\n    got  : {:?} conf {:.2} kind {:?}", cfg.name, sd, xw.nl[i].level, o, t.kind.label(), xw.nl[i].text, t.spec, xw.ex[i].spec, xw.ex[i].conf, xw.ex[i].kind);
                }
            }
        }
        let mut ks: Vec<_> = per.into_iter().collect();
        ks.sort_by(|a, b| a.0.cmp(&b.0));
        println!("=== world {} (unseen={want_unseen}) per kind, non-dup tasks ===", cfg.name);
        for (k, c) in ks {
            println!("{k:<7} {}", c.s());
        }
    }
}

fn eval(args: &[String], seeds: &[u64]) {
    let tag = g13::arg(args, "--tag", "held");
    let mut all = serde_json::Map::new();
    for (cfg, ncfg) in worlds() {
        // [level][bank], [kind][bank], overall, calibration bins, qual false positives
        let mut by_level: Vec<[Cell; 2]> = vec![Default::default(); 5];
        let mut by_kind: Vec<[Cell; 2]> = vec![Default::default(); 7];
        let mut overall = Cell::default();
        let mut by_bank: [Cell; 2] = Default::default();
        let mut cal: Vec<(usize, usize)> = vec![(0, 0); 5]; // conf bins: <.3, <.5, <.7, <.9, >=.9 : (n, correct)
        let mut qual_n = 0usize;
        let mut qual_fp = 0usize;
        let mut qual_fp_hi = 0usize;
        let mut qual_hn_fp = [0usize; 2];
        let mut qual_hn_n = [0usize; 2];
        let mut n_total = 0usize;
        for &sd in seeds {
            let xw = build_xworld(&cfg, &ncfg, sd);
            let mut reg = Registry::with_defaults(xw.w.lookup.clone(), false);
            for i in 0..xw.w.tasks.len() {
                let t = &xw.w.tasks[i];
                let ex = &xw.ex[i];
                let (o, exact) = judge(&xw, &mut reg, i);
                let bank = xw.nl[i].unseen as usize;
                let lvl = xw.nl[i].level as usize;
                n_total += 1;
                if t.kind == Kind::Qual {
                    qual_n += 1;
                    let has_frag = xw.nl[i].text.contains(" plus ") || xw.nl[i].text.contains(" + ") || xw.nl[i].text.contains(" km ") || xw.nl[i].text.contains("x^2");
                    qual_hn_n[bank] += has_frag as usize;
                    if ex.spec.is_some() {
                        qual_fp += 1;
                        qual_hn_fp[bank] += has_frag as usize;
                        if ex.conf >= 0.5 {
                            qual_fp_hi += 1;
                        }
                    }
                    continue;
                }
                by_level[lvl][bank].add(o, exact, ex.conf);
                by_kind[t.kind.idx()][bank].add(o, exact, ex.conf);
                overall.add(o, exact, ex.conf);
                by_bank[bank].add(o, exact, ex.conf);
                if ex.spec.is_some() {
                    let b = if ex.conf < 0.3 { 0 } else if ex.conf < 0.5 { 1 } else if ex.conf < 0.7 { 2 } else if ex.conf < 0.9 { 3 } else { 4 };
                    cal[b].0 += 1;
                    cal[b].1 += (o == Out::Correct) as usize;
                }
            }
        }
        println!("\n================ WORLD {} seeds {}-{} ({} tasks, {} compute, {} prose) ================", cfg.name, seeds[0], seeds[seeds.len() - 1], n_total, overall.n, qual_n);
        println!("OVERALL (compute tasks): {}", overall.s());
        println!("  dev-bank phrasing   : {}", by_bank[0].s());
        println!("  UNSEEN-bank phrasing: {}", by_bank[1].s());
        let mut lj = Vec::new();
        for l in 0..5 {
            let mut both = by_level[l][0].clone();
            let u = &by_level[l][1];
            both.n += u.n;
            both.correct += u.correct;
            both.abstain += u.abstain;
            both.wrong += u.wrong;
            both.wrong_conf += u.wrong_conf;
            both.exact += u.exact;
            println!("level L{l} all   : {}", both.s());
            println!("        dev   : {}", by_level[l][0].s());
            println!("        unseen: {}", by_level[l][1].s());
            lj.push(json!({"level": l, "all": both.j(), "dev": by_level[l][0].j(), "unseen": by_level[l][1].j()}));
        }
        let mut kj = Vec::new();
        for k in KINDS {
            if k == Kind::Qual {
                continue;
            }
            let c = &by_kind[k.idx()];
            println!("kind {:<7} dev: {}", k.label(), c[0].s());
            println!("kind {:<7} unseen: {}", "", c[1].s());
            kj.push(json!({"kind": k.label(), "dev": c[0].j(), "unseen": c[1].j()}));
        }
        println!("confidence calibration (P(extracted spec is semantically right | conf bin), extracted specs only):");
        let labels = ["<.3", ".3-.5", ".5-.7", ".7-.9", ">=.9"];
        let mut cj = Vec::new();
        for (b, l) in labels.iter().enumerate() {
            let (n, c) = cal[b];
            println!("   conf {l:<6} n={n:>6} P(right)={:.3}", if n > 0 { c as f64 / n as f64 } else { f64::NAN });
            cj.push(json!({"bin": l, "n": n, "p_right": if n > 0 { c as f64 / n as f64 } else { f64::NAN }}));
        }
        println!("prose tasks (no spec exists): extractor emitted a spec on {}/{} = {:.4}; with conf>=.5 on {:.4}; hard negatives (arithmetic-looking fragment) dev {}/{} unseen {}/{}", qual_fp, qual_n, qual_fp as f64 / qual_n as f64, qual_fp_hi as f64 / qual_n as f64, qual_hn_fp[0], qual_hn_n[0], qual_hn_fp[1], qual_hn_n[1]);
        all.insert(
            cfg.name.to_string(),
            json!({"overall": overall.j(), "dev_bank": by_bank[0].j(), "unseen_bank": by_bank[1].j(), "levels": lj, "kinds": kj, "calibration": cj,
                   "prose": {"n": qual_n, "spec_emitted": qual_fp, "spec_emitted_conf_ge_0.5": qual_fp_hi, "hardneg_fp_dev": qual_hn_fp[0], "hardneg_n_dev": qual_hn_n[0], "hardneg_fp_unseen": qual_hn_fp[1], "hardneg_n_unseen": qual_hn_n[1]}}),
        );
    }
    let out = g13::arg(args, "--out", &format!("results/g21_extract_{tag}.json"));
    std::fs::write(&out, serde_json::to_string_pretty(&json!({"tag": tag, "seeds": seeds, "worlds": all})).unwrap()).unwrap();
    println!("\nwrote {out}");
}
