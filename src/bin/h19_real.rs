//! h19_real: small REAL-TEXT check of the extractor primitives and the verifier on documents that exist on disk:
//!   (A) markdown tables in this repo's results/*.md (real machine-written benchmark tables),
//!   (B) markdown tables in vendored crate READMEs under ~/.cargo/registry (real third-party benchmark tables),
//!   (C) prose quantities in results/market-research.md + finance-packages-catalog.md (currency amounts),
//!   (D) interval-triple consistency "x [lo,hi]" in the results/*.md board entries.
//! NO ground-truth labels exist for real text. What is measured: parse coverage, agreement between two independent parsers,
//! verifier flag rate (a proxy for false flags on real text), footing of tables that have a Total row, interval containment
//! (a real cross-check that can find real errors). Flagged examples are listed for MANUAL inspection by the author.
mod h19_common;
use h19_common::scan::*;
use h19_common::verify::*;
use h19_common::*;
use std::fs;
use std::path::{Path, PathBuf};

fn triv(cell: &str) -> Option<f64> {
    let mut s = cell.trim().replace("**", "").replace('`', "");
    if let Some(p) = s.find('\u{b1}') {
        s.truncate(p);
    }
    let tok = s.split_whitespace().next()?.to_string();
    let t = tok.trim_end_matches('%').trim_end_matches('x').trim_end_matches(',').replace(',', "");
    let t = t.trim_start_matches('+');
    t.parse::<f64>().ok()
}
#[derive(Default)]
struct Tab {
    file: String,
    header: Vec<String>,
    rows: Vec<(usize, Vec<Cell>)>,
}
fn tables_of(file: &str, text: &str) -> Vec<Tab> {
    let lns = lines_of(text);
    let mut out: Vec<Tab> = Vec::new();
    let mut i = 0;
    while i + 1 < lns.len() {
        let (_, l0) = lns[i];
        let l1 = lns[i + 1].1;
        let is_sep = |l: &str| l.trim_start().starts_with('|') && l.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '));
        if l0.trim_start().starts_with('|') && is_sep(l1) {
            let hc = split_cells(lns[i].0, l0);
            let mut t = Tab { file: file.to_string(), header: hc.iter().map(|c| c.txt.clone()).collect(), rows: Vec::new() };
            let mut j = i + 2;
            while j < lns.len() && lns[j].1.trim_start().starts_with('|') {
                t.rows.push((j, split_cells(lns[j].0, lns[j].1)));
                j += 1;
            }
            out.push(t);
            i = j;
        } else {
            i += 1;
        }
    }
    out
}
fn collect(roots: &[PathBuf], pred: &dyn Fn(&Path) -> bool, depth: usize, out: &mut Vec<PathBuf>) {
    for r in roots {
        if let Ok(rd) = fs::read_dir(r) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if depth > 0 && !p.to_string_lossy().contains("node_modules") {
                        collect(&[p], pred, depth - 1, out);
                    }
                } else if pred(&p) {
                    out.push(p);
                }
            }
        }
    }
}

struct Stats {
    tables: usize,
    cells_numeric_mine: usize,
    cells_numeric_triv: usize,
    agree: usize,
    disagree: Vec<String>,
    ip_disagree: Vec<String>,
    ip_n: usize,
    total_rows_tested: usize,
    total_rows_ok: usize,
    total_bad: Vec<String>,
    pct_cols: usize,
    pct_bad: Vec<String>,
}
fn analyse(label: &str, files: &[PathBuf], max_tables: usize) -> Stats {
    let mut st = Stats { tables: 0, cells_numeric_mine: 0, cells_numeric_triv: 0, agree: 0, disagree: vec![], ip_disagree: vec![], ip_n: 0, total_rows_tested: 0, total_rows_ok: 0, total_bad: vec![], pct_cols: 0, pct_bad: vec![] };
    let mut nfiles = 0;
    for f in files {
        let Ok(text) = fs::read_to_string(f) else { continue };
        let tabs = tables_of(&f.file_name().unwrap().to_string_lossy(), &text);
        if tabs.is_empty() {
            continue;
        }
        nfiles += 1;
        for t in tabs {
            if st.tables >= max_tables {
                break;
            }
            st.tables += 1;
            // numeric cell parse agreement
            let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::new(); t.header.len()];
            for (li, row) in &t.rows {
                for (j, c) in row.iter().enumerate().skip(1) {
                    let a = parse_num(&c.txt).map(|n| n.v);
                    let b = triv(&c.txt);
                    if a.is_some() {
                        st.cells_numeric_mine += 1;
                    }
                    if b.is_some() {
                        st.cells_numeric_triv += 1;
                    }
                    match (a, b) {
                        (Some(x), Some(y)) => {
                            // my parser applies K/M/B suffix; trivial one does not: compare after allowing that
                            if close(x, y) || (y != 0.0 && ((x / y).abs().log10().round() - (x / y).abs().log10()).abs() < 1e-9) {
                                st.agree += 1;
                            } else {
                                st.disagree.push(format!("{}:{} '{}' mine={} triv={}", t.file, li + 1, c.txt, x, y));
                            }
                        }
                        (None, None) => st.agree += 1,
                        (Some(x), None) => st.disagree.push(format!("{}:{} '{}' mine={} triv=None", t.file, li + 1, c.txt, x)),
                        (None, Some(y)) => st.disagree.push(format!("{}:{} '{}' mine=None triv={}", t.file, li + 1, c.txt, y)),
                    }
                    // independent re-parse of the span (verifier primitive) vs the extractor parse
                    if let Some(n) = parse_num(&c.txt) {
                        st.ip_n += 1;
                        // locate span in file text
                        if let Some(ip) = iparse(&text, c.lo, c.hi, true, true) {
                            let v = ip.v * if ip.neg { -1.0 } else { 1.0 };
                            if !close(v, n.v) && !(ip.pct && close(v, n.v)) {
                                st.ip_disagree.push(format!("{}:{} '{}' extractor={} verifier-iparse={} (ambig {})", t.file, li + 1, c.txt, n.v, v, ip.ambiguous));
                            }
                        }
                    }
                    if j < cols.len() {
                        cols[j].push(a);
                    }
                }
            }
            // total-row footing: a row whose label contains "total"/"sum": column == sum of previous numeric cells
            for (ri, (li, row)) in t.rows.iter().enumerate() {
                let lab = row[0].txt.to_lowercase();
                if (lab.contains("total") || lab == "sum" || lab == "**total**") && ri >= 2 {
                    for j in 1..row.len() {
                        let tv = parse_num(&row[j].txt).map(|n| n.v);
                        let parts: Vec<f64> = t.rows[..ri].iter().filter_map(|(_, r)| r.get(j).and_then(|c| parse_num(&c.txt)).map(|n| n.v)).collect();
                        if let Some(tv) = tv {
                            if parts.len() == ri && ri >= 2 {
                                st.total_rows_tested += 1;
                                let s: f64 = parts.iter().sum();
                                if (s - tv).abs() <= 0.006 * tv.abs().max(1.0) {
                                    st.total_rows_ok += 1;
                                } else {
                                    st.total_bad.push(format!("{}:{} col {} total {} vs sum {}", t.file, li + 1, j, tv, s));
                                }
                            }
                        }
                    }
                }
            }
            // percent plausibility
            for (j, h) in t.header.iter().enumerate() {
                if h.contains('%') && j < cols.len() {
                    st.pct_cols += 1;
                    for v in cols[j].iter().flatten() {
                        if *v < -100.0 || *v > 100.0 {
                            st.pct_bad.push(format!("{} col '{}' value {}", t.file, h, v));
                        }
                    }
                }
            }
        }
    }
    println!("\n## {}: {} files with tables, {} tables, numeric cells (extractor parser) {} vs (trivial parser) {}", label, nfiles, st.tables, st.cells_numeric_mine, st.cells_numeric_triv);
    st
}
fn report(st: &Stats, show: usize) {
    let n = st.agree + st.disagree.len();
    println!("   parser agreement (extractor vs trivial): {}/{} = {:.4}", st.agree, n, st.agree as f64 / n.max(1) as f64);
    println!("   verifier independent re-parse vs extractor parse: disagreements {}/{} = {:.4} (a proxy for FALSE flags on real cells; no labels)", st.ip_disagree.len(), st.ip_n, st.ip_disagree.len() as f64 / st.ip_n.max(1) as f64);
    println!("   total-row footing: {}/{} columns foot (tolerance .6%); violations {}", st.total_rows_ok, st.total_rows_tested, st.total_bad.len());
    println!("   percent columns {}; implausible values {}", st.pct_cols, st.pct_bad.len());
    for d in st.disagree.iter().take(show) {
        println!("     parser-disagree: {}", d);
    }
    for d in st.ip_disagree.iter().take(show) {
        println!("     verifier-flag:   {}", d);
    }
    for d in st.total_bad.iter().take(show) {
        println!("     footing-violation: {}", d);
    }
    for d in st.pct_bad.iter().take(show) {
        println!("     pct-implausible: {}", d);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let show: usize = arg(&args, "--show").and_then(|s| s.parse().ok()).unwrap_or(8);
    println!("# h19_real: REAL on-disk text, no labels; author inspects flagged items. Label every number REAL-TEXT/UNLABELLED.");
    // (A) repo results tables
    let mut a: Vec<PathBuf> = Vec::new();
    collect(&[PathBuf::from("results")], &|p: &Path| p.extension().map_or(false, |e| e == "md"), 0, &mut a);
    a.sort();
    let sa = analyse("(A) results/*.md tables (this repo; machine-written benchmark tables)", &a, 100000);
    report(&sa, show);
    // (B) crate READMEs
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_default();
    let reg = PathBuf::from(&home).join(".cargo").join("registry").join("src");
    let mut roots = Vec::new();
    if let Ok(rd) = fs::read_dir(&reg) {
        for e in rd.flatten() {
            roots.push(e.path());
        }
    }
    let mut crates: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = fs::read_dir(roots.first().cloned().unwrap_or_default()) {
        for e in rd.flatten() {
            let p = e.path().join("README.md");
            if p.exists() {
                crates.push(p);
            }
        }
    }
    crates.sort();
    let sb = analyse("(B) vendored crate READMEs (~/.cargo/registry; real third-party benchmark/feature tables)", &crates, 100000);
    report(&sb, show);
    // (C) prose amounts
    println!("\n## (C) currency amounts in real prose (results/market-research.md, results/finance-packages-catalog.md)");
    let mut n_am = 0;
    let mut n_ip_ok = 0;
    let mut samples = Vec::new();
    for f in ["results/market-research.md", "results/finance-packages-catalog.md", "results/own-agents-feasibility.md"] {
        let Ok(text) = fs::read_to_string(f) else { continue };
        for (lo, hi, val, _conf, cur) in amounts_dev(&text) {
            n_am += 1;
            if let Some(ip) = iparse(&text, lo, hi, false, true) {
                let tail = fwd(&text[hi..], 6).to_string();
                let v = ip.v;
                if close(v, val) {
                    n_ip_ok += 1;
                } else if samples.len() < show {
                    samples.push(format!("{} '{}' extractor={} iparse={} (tail '{}')", f, &text[lo..hi], val, v, tail.replace('\n', " ")));
                }
            }
        }
    }
    println!("   amounts found {}; independent re-parse agrees on {} ({:.3}); disagreements (likely scale suffix K/M/B/mn after the span, i.e. true extractor misses):", n_am, n_ip_ok, n_ip_ok as f64 / n_am.max(1) as f64);
    for s in &samples {
        println!("     {}", s);
    }
    // (D) interval triples
    println!("\n## (D) interval triples 'x [lo,hi]' in results/*.md (containment lo <= x <= hi)");
    let (mut n, mut bad) = (0, Vec::new());
    for f in &a {
        let Ok(text) = fs::read_to_string(f) else { continue };
        for (li, ln) in text.lines().enumerate() {
            let b = ln.as_bytes();
            let mut i = 0;
            while i < b.len() {
                if b[i] == b'[' {
                    // number before: scan back over spaces then a number token
                    let pre = ln[..i].trim_end();
                    let tok_start = pre.rfind(|c: char| !(c.is_ascii_digit() || c == '.' || c == '+' || c == '-' || c == 'e')).map_or(0, |p| p + 1);
                    let x: Option<f64> = pre[tok_start..].trim_start_matches('+').parse().ok();
                    if let Some(rest) = ln[i + 1..].split(']').next() {
                        let parts: Vec<&str> = rest.split(',').collect();
                        if parts.len() == 2 {
                            let lo: Option<f64> = parts[0].trim().trim_start_matches('+').parse().ok();
                            let hi: Option<f64> = parts[1].trim().trim_start_matches('+').parse().ok();
                            if let (Some(x), Some(lo), Some(hi)) = (x, lo, hi) {
                                n += 1;
                                let tol = 6e-5 + 1e-9;
                                if x < lo - tol || x > hi + tol {
                                    bad.push(format!("{}:{} {} [{},{}]", f.file_name().unwrap().to_string_lossy(), li + 1, x, lo, hi));
                                }
                            }
                        }
                    }
                }
                i += 1;
            }
        }
    }
    println!("   triples {}; containment violations {} ({:.4})", n, bad.len(), bad.len() as f64 / n.max(1) as f64);
    for s in bad.iter().take(show * 2) {
        println!("     violation: {}", s);
    }
}
