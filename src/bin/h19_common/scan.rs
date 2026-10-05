//! Hand-written, regex-free, std-only EXTRACTOR. Built ONLY against Bank::Dev documents of seeds 1-8 (lexicons from `dev_labels()`,
//! phrase lists below). It emits CANDIDATE facts with a per-field confidence; abstention = confidence below a threshold tau that is
//! calibrated on the tune seeds (see cal.rs). It knows nothing about Bank::Unseen phrasing: that is the point of the held-out test.
use super::gens::{dev_labels, Item, BS, CF, INC};
use super::genb::{kpi_dev_labels, METS, MODELS, PARAMS};
use super::*;

pub const N_UNKNOWN_SCALE: u32 = 1;
pub const N_FUZZY: u32 = 2;
pub const N_AMBIG_SEP: u32 = 4;
pub const N_PERIOD_UNK: u32 = 8;
pub const N_DUP: u32 = 16;
pub const N_UNIT_DEFAULT: u32 = 32;
pub const N_SHORT_ROW: u32 = 64;
pub const N_CUR_UNK: u32 = 128;
pub const N_NOCUE: u32 = 256;

#[derive(Clone, Debug)]
pub struct Ext {
    pub fact: Fact,
    pub conf: f64,
    pub notes: u32,
    /// for table facts: line index of the row (used by the verifier for row/column checks)
    pub row: usize,
    pub col: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct ExCfg {
    pub fuzzy: bool,
}
impl Default for ExCfg {
    fn default() -> Self {
        ExCfg { fuzzy: true }
    }
}

// ---------------------------------------------------------------- lines + cells
pub fn lines_of(text: &str) -> Vec<(usize, &str)> {
    let mut v = Vec::new();
    let mut lo = 0;
    for l in text.split('\n') {
        v.push((lo, l));
        lo += l.len() + 1;
    }
    v
}
#[derive(Clone, Debug)]
pub struct Cell {
    pub lo: usize,
    pub hi: usize,
    pub txt: String,
}
fn trim_span(line: &str, a: usize, b: usize) -> (usize, usize) {
    let s = &line[a..b];
    let t0 = s.len() - s.trim_start().len();
    let t1 = s.trim_end().len();
    if t1 <= t0 { (a, a) } else { (a + t0, a + t1) }
}
fn has_thousands_comma(line: &str) -> bool {
    let b = line.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() && (i == 0 || matches!(b[i - 1], b' ' | b'$' | b'(' | b'"' | b':' | b'|' | 0xAC | 0xA3)) {
            let mut j = i;
            while j < b.len() && b[j].is_ascii_digit() {
                j += 1;
            }
            if j - i <= 3 {
                let mut groups = 0;
                while j + 3 < b.len() + 0 && b[j] == b',' && b[j + 1].is_ascii_digit() && b[j + 2].is_ascii_digit() && b[j + 3].is_ascii_digit() && (j + 4 >= b.len() || !b[j + 4].is_ascii_digit()) {
                    groups += 1;
                    j += 4;
                }
                if groups > 0 {
                    return true;
                }
            }
            i = j.max(i + 1);
        } else {
            i += 1;
        }
    }
    false
}
pub fn split_cells(lo: usize, line: &str) -> Vec<Cell> {
    let mut segs: Vec<(usize, usize)> = Vec::new();
    let pipes = line.matches('|').count();
    let commas = line.matches(',').count();
    if pipes >= 2 {
        let mut start = 0;
        for (i, c) in line.char_indices() {
            if c == '|' {
                segs.push((start, i));
                start = i + 1;
            }
        }
        segs.push((start, line.len()));
        if !segs.is_empty() && line[segs[0].0..segs[0].1].trim().is_empty() {
            segs.remove(0);
        }
        if !segs.is_empty() && line[segs[segs.len() - 1].0..segs[segs.len() - 1].1].trim().is_empty() {
            segs.pop();
        }
    } else if commas >= 2 && (line.contains('"') || (!line.contains("  ") && !has_thousands_comma(line))) && !line.trim_end().ends_with('.') {
        let mut start = 0;
        let mut inq = false;
        for (i, c) in line.char_indices() {
            if c == '"' {
                inq = !inq;
            } else if c == ',' && !inq {
                segs.push((start, i));
                start = i + 1;
            }
        }
        segs.push((start, line.len()));
    } else {
        // whitespace-run split (2+ spaces, tab, dot leaders)
        let b = line.as_bytes();
        let mut start = 0;
        let mut i = 0;
        while i < b.len() {
            if b[i] == b' ' || b[i] == b'\t' || b[i] == b'.' {
                let mut j = i;
                let (mut sp, mut dots, mut tab) = (0, 0, false);
                while j < b.len() && (b[j] == b' ' || b[j] == b'\t' || b[j] == b'.') {
                    match b[j] {
                        b' ' => sp += 1,
                        b'.' => dots += 1,
                        _ => tab = true,
                    }
                    j += 1;
                }
                let sep = tab || sp >= 2 || dots >= 3 || (dots >= 1 && sp >= 2);
                if sep {
                    segs.push((start, i));
                    start = j;
                }
                i = j;
            } else {
                i += 1;
            }
        }
        segs.push((start, line.len()));
    }
    let mut out = Vec::new();
    for (a, bb) in segs {
        let (mut x, mut y) = trim_span(line, a, bb);
        if y > x && line.as_bytes()[x] == b'"' && line.as_bytes()[y - 1] == b'"' && y - x >= 2 {
            x += 1;
            y -= 1;
        }
        out.push(Cell { lo: lo + x, hi: lo + y, txt: line[x..y].to_string() });
    }
    out
}

// ---------------------------------------------------------------- number parsing (extractor version)
#[derive(Clone, Debug)]
pub struct Num {
    pub v: f64,
    pub conf: f64,
    pub pct: bool,
    pub cur: Option<&'static str>,
    pub mult: f64,
    pub unit: String,
    pub notes: u32,
}
fn strip_footnote(s: &str) -> &str {
    let mut t = s.trim();
    loop {
        let before = t;
        t = t.trim_end_matches(|c| c == '*' || c == '\u{2020}' || c == '\u{2021}').trim_end();
        if t.ends_with(')') {
            if let Some(p) = t.rfind('(') {
                let inner = &t[p + 1..t.len() - 1];
                if p > 0 && t.as_bytes()[p - 1] == b' ' && inner.len() <= 2 && !inner.is_empty() && (inner.chars().all(|c| c.is_ascii_digit()) || (inner.len() == 1 && inner.chars().all(|c| c.is_ascii_lowercase()))) && t[..p].trim_end().chars().last().map_or(false, |c| c.is_ascii_digit()) {
                    t = t[..p].trim_end();
                }
            }
        }
        if t == before {
            break;
        }
    }
    t
}
/// parse a numeric core (digits with , .); returns (value, conf, notes)
pub fn parse_core(s: &str) -> Option<(f64, f64, u32)> {
    if s.is_empty() || !s.as_bytes()[0].is_ascii_digit() && s.as_bytes()[0] != b'.' {
        return None;
    }
    if !s.bytes().all(|c| c.is_ascii_digit() || c == b',' || c == b'.') {
        return None;
    }
    if !s.bytes().last().map_or(false, |c| c.is_ascii_digit()) {
        return None;
    }
    let commas = s.matches(',').count();
    let dots = s.matches('.').count();
    let (mut conf, mut notes) = (1.0, 0u32);
    let clean: String;
    if commas > 0 && dots > 0 {
        let lc = s.rfind(',').unwrap();
        let ld = s.rfind('.').unwrap();
        if ld > lc {
            // US: commas thousands
            if !group_ok(&s[..ld], ',') {
                return None;
            }
            clean = s.replace(',', "");
        } else {
            if !group_ok(&s[..lc], '.') {
                return None;
            }
            clean = s.replace('.', "").replace(',', ".");
            conf = 0.7;
        }
    } else if commas > 0 {
        if group_ok(s, ',') {
            clean = s.replace(',', "");
        } else if commas == 1 && s.len() - s.find(',').unwrap() - 1 <= 2 {
            clean = s.replace(',', ".");
            conf = 0.5;
        } else {
            return None;
        }
    } else if dots > 1 {
        if group_ok(s, '.') {
            clean = s.replace('.', "");
            conf = 0.6;
            notes |= N_AMBIG_SEP;
        } else {
            return None;
        }
    } else {
        clean = s.to_string();
        if dots == 1 {
            let frac = s.len() - s.find('.').unwrap() - 1;
            let ip = s.find('.').unwrap();
            if frac == 3 && ip >= 1 && ip <= 3 {
                conf = 0.75;
                notes |= N_AMBIG_SEP;
            }
        }
    }
    clean.parse::<f64>().ok().map(|v| (v, conf, notes))
}
fn group_ok(s: &str, g: char) -> bool {
    let parts: Vec<&str> = s.split(g).collect();
    if parts.len() < 2 {
        return false;
    }
    if parts[0].is_empty() || parts[0].len() > 3 {
        return false;
    }
    parts[1..].iter().all(|p| p.len() == 3 && p.bytes().all(|c| c.is_ascii_digit())) && parts[0].bytes().all(|c| c.is_ascii_digit())
}
/// parse a table cell as a number (dev conventions: parentheses / ASCII minus negatives, $ EUR GBP symbols, K/M/B suffix, %)
pub fn parse_num(cell: &str) -> Option<Num> {
    let mut s = cell.trim();
    while s.starts_with("**") && s.ends_with("**") && s.len() >= 4 {
        s = s[2..s.len() - 2].trim();
    }
    s = strip_footnote(s).trim();
    if let Some(p) = s.find('\u{b1}') {
        s = s[..p].trim();
    }
    if s.is_empty() {
        return None;
    }
    let mut neg = false;
    if s.starts_with('(') && s.ends_with(')') && s.len() >= 3 {
        neg = true;
        s = s[1..s.len() - 1].trim();
    } else if let Some(r) = s.strip_prefix('-') {
        neg = true;
        s = r.trim_start();
    }
    let mut cur: Option<&'static str> = None;
    for (p, c) in [("$", "USD"), ("\u{20ac}", "EUR"), ("\u{a3}", "GBP"), ("USD ", "USD"), ("EUR ", "EUR"), ("GBP ", "GBP")] {
        if let Some(r) = s.strip_prefix(p) {
            cur = Some(c);
            s = r.trim_start();
            break;
        }
    }
    // trailing: %, suffix, unit
    let mut pct = false;
    let mut mult = 1.0;
    let mut unit = String::new();
    // split numeric core from the remainder
    let end = s.find(|c: char| !(c.is_ascii_digit() || c == ',' || c == '.')).unwrap_or(s.len());
    let (core, rest) = s.split_at(end);
    let rest = rest.trim();
    if !rest.is_empty() {
        if rest == "%" {
            pct = true;
        } else if rest == "K" || rest == "k" {
            mult = 1e3;
        } else if rest == "M" {
            mult = 1e6;
        } else if rest == "B" {
            mult = 1e9;
        } else if rest.chars().all(|c| c.is_ascii_alphabetic()) && rest.len() <= 3 {
            unit = rest.to_string();
        } else if rest == "pp" {
            unit = "pp".into();
        } else {
            return None;
        }
    }
    let (v, conf, notes) = parse_core(core)?;
    Some(Num { v: if neg { -v } else { v } * mult, conf, pct, cur, mult, unit, notes })
}

// ---------------------------------------------------------------- headers, scale, currency
pub fn year_in(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let mut i = 0;
    while i + 4 <= b.len() {
        if b[i].is_ascii_digit() && b[i + 1].is_ascii_digit() && b[i + 2].is_ascii_digit() && b[i + 3].is_ascii_digit() {
            let prev = i > 0 && b[i - 1].is_ascii_digit();
            let next = i + 4 < b.len() && b[i + 4].is_ascii_digit();
            if !prev && !next {
                let y: i64 = s[i..i + 4].parse().unwrap();
                if (1990..=2039).contains(&y) {
                    return Some(y);
                }
            }
        }
        i += 1;
    }
    None
}
/// "Q3 2024" -> (3, 2024)
pub fn quarter_in(s: &str) -> Option<(i64, i64)> {
    let b = s.as_bytes();
    for i in 0..b.len().saturating_sub(1) {
        if (b[i] == b'Q') && b[i + 1].is_ascii_digit() && (b[i + 1] >= b'1' && b[i + 1] <= b'4') {
            let q = (b[i + 1] - b'0') as i64;
            let rest = &s[i + 2..];
            if let Some(y) = year_in(rest) {
                if rest.trim_start().starts_with(|c: char| c.is_ascii_digit()) {
                    return Some((q, y));
                }
            }
        }
    }
    None
}
pub fn scale_word(s: &str) -> Option<f64> {
    let l = s.to_lowercase();
    if l.contains("thousand") {
        Some(1e3)
    } else if l.contains("million") {
        Some(1e6)
    } else if l.contains("billion") {
        Some(1e9)
    } else {
        None
    }
}
pub fn currency_in(s: &str) -> Option<(&'static str, f64)> {
    if s.contains('$') {
        return Some(("USD", 1.0));
    }
    if s.contains('\u{20ac}') {
        return Some(("EUR", 1.0));
    }
    if s.contains('\u{a3}') {
        return Some(("GBP", 1.0));
    }
    for (w, c) in [("USD", "USD"), ("EUR", "EUR"), ("GBP", "GBP")] {
        if s.contains(w) {
            return Some((c, 1.0));
        }
    }
    None
}
/// generic 3-uppercase-letter code (unknown to the dev lexicon) -> low confidence
pub fn generic_code(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut i = 0;
    while i + 3 <= b.len() {
        if b[i].is_ascii_uppercase() && b[i + 1].is_ascii_uppercase() && b[i + 2].is_ascii_uppercase() && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) && (i + 3 == b.len() || !b[i + 3].is_ascii_alphanumeric()) {
            return Some(s[i..i + 3].to_string());
        }
        i += 1;
    }
    None
}
fn window_scale(lns: &[(usize, &str)], li: usize) -> (Option<f64>, Option<(String, f64)>) {
    let from = li.saturating_sub(6);
    let mut sc = None;
    let mut cur: Option<(String, f64)> = None;
    let mut idxs: Vec<usize> = (from..=li.min(lns.len() - 1)).rev().collect();
    idxs.extend(0..lns.len().min(5));
    // footer note ("All amounts in thousands unless otherwise stated") -- added after dev failures on L4 docs
    for k in (0..lns.len()).rev().take(4) {
        let ll = lns[k].1.to_lowercase();
        if ll.contains("unless otherwise") || ll.contains("all amounts") {
            idxs.push(k);
        }
    }
    for k in idxs {
        let l = lns[k].1;
        if sc.is_none() {
            sc = scale_word(l);
        }
        if cur.is_none() {
            if let Some((c, cf)) = currency_in(l) {
                cur = Some((c.to_string(), cf));
            } else if scale_word(l).is_some() || l.to_lowercase().contains("currency") {
                if let Some(c) = generic_code(l) {
                    cur = Some((c, 0.6));
                }
            }
        }
    }
    (sc, cur)
}

// ---------------------------------------------------------------- label lexicon (fuzzy)
fn stem(w: &str) -> String {
    let w = w.to_lowercase();
    if w.len() > 4 && w.ends_with("ies") {
        format!("{}y", &w[..w.len() - 3])
    } else if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") {
        w[..w.len() - 1].to_string()
    } else {
        w
    }
}
pub fn label_tokens(s: &str) -> Vec<String> {
    let stop = ["and", "of", "the", "in", "for", "by", "to", "on"];
    let mut v: Vec<String> = s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty() && !stop.contains(w)).map(stem).collect();
    v.sort();
    v.dedup();
    v
}
pub fn norm_label(s: &str) -> String {
    let mut t = s.trim().to_string();
    // strip footnote marker "(a)"
    if t.ends_with(')') {
        if let Some(p) = t.rfind('(') {
            let inner = &t[p + 1..t.len() - 1];
            if inner.len() == 1 && inner.chars().all(|c| c.is_ascii_alphanumeric()) {
                t = t[..p].trim_end().to_string();
            }
        }
    }
    t.trim_end_matches(|c: char| c == ':' || c == '.').to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}
pub struct Lex {
    pub exact: Vec<(String, &'static str)>,
    pub toks: Vec<(Vec<String>, &'static str)>,
}
impl Lex {
    pub fn new(pairs: &[(&'static str, &'static str)]) -> Lex {
        Lex { exact: pairs.iter().map(|(id, l)| (norm_label(l), *id)).collect(), toks: pairs.iter().map(|(id, l)| (label_tokens(l), *id)).collect() }
    }
    /// -> (id, conf, notes)
    pub fn lookup(&self, label: &str, fuzzy: bool) -> Option<(&'static str, f64, u32)> {
        let n = norm_label(label);
        if n.is_empty() {
            return None;
        }
        for (l, id) in &self.exact {
            if *l == n {
                return Some((*id, 0.92, 0));
            }
        }
        if !fuzzy {
            return None;
        }
        let t = label_tokens(label);
        if t.is_empty() {
            return None;
        }
        let mut best: (f64, &str) = (0.0, "");
        let mut second = 0.0;
        for (lt, id) in &self.toks {
            let inter = t.iter().filter(|x| lt.contains(x)).count() as f64;
            let uni = (t.len() + lt.len()) as f64 - inter;
            let j = if uni > 0.0 { inter / uni } else { 0.0 };
            if j > best.0 {
                if *id != best.1 {
                    second = best.0;
                }
                best = (j, id);
            } else if j > second && *id != best.1 {
                second = j;
            }
        }
        if best.0 >= 0.5 && best.0 - second >= 0.15 {
            Some((best.1, 0.35 + 0.35 * best.0, N_FUZZY))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------- FINANCIAL extraction
#[derive(Clone, Debug, Default)]
struct Pcol {
    year: Option<i64>,
    quarter: Option<(i64, i64)>,
    restated: bool,
    skip: bool,
}
pub fn fin_lex() -> Lex {
    Lex::new(&dev_labels())
}
pub fn extract_fin(text: &str, cfg: &ExCfg) -> Vec<Ext> {
    let lns = lines_of(text);
    let lex = fin_lex();
    let mut out: Vec<Ext> = Vec::new();
    let mut hdr: Vec<Pcol> = Vec::new();
    let mut hdr_len = 0usize;
    let mut tscale: Option<f64> = None;
    let mut tcur: Option<(String, f64)> = None;
    for (li, (lo, ln)) in lns.iter().enumerate() {
        let cells = split_cells(*lo, ln);
        if cells.len() < 2 {
            continue;
        }
        let first_label = lex.lookup(&cells[0].txt, false).is_some();
        let ycount = cells[1..].iter().filter(|c| year_in(&c.txt).is_some()).count();
        let cp = cells[1..].iter().filter(|c| { let l = c.txt.to_lowercase(); l.contains("current year") || l.contains("prior year") }).count();
        if !first_label && (ycount >= 2 || cp >= 2) && cells[1..].iter().all(|c| year_in(&c.txt).is_some() || c.txt.to_lowercase().contains("year")) {
            hdr = cells[1..].iter().map(|c| Pcol { year: year_in(&c.txt), quarter: None, restated: c.txt.to_lowercase().contains("restated"), skip: c.txt.to_lowercase().contains("as reported") || c.txt.to_lowercase().contains("previously") }).collect();
            hdr_len = hdr.len();
            let (sc, cur) = window_scale(&lns, li);
            tscale = sc;
            tcur = cur;
            continue;
        }
        // data row
        let label = &cells[0].txt;
        if parse_num(label).is_some() && year_in(label).is_none() {
            continue;
        }
        let (id, lconf, lnotes) = match lex.lookup(label, cfg.fuzzy) {
            Some(x) => x,
            None => continue,
        };
        if hdr.is_empty() {
            continue;
        }
        let vals = &cells[1..];
        for (j, c) in vals.iter().enumerate() {
            let Some(n) = parse_num(&c.txt) else { continue };
            if n.pct {
                continue;
            }
            let pc = hdr.get(j).cloned().unwrap_or_default();
            if pc.skip {
                continue;
            }
            let mut notes = lnotes | n.notes;
            let mut conf = lconf * n.conf;
            let key = match pc.year {
                Some(y) => format!("{}|FY{}", id, y),
                None => {
                    notes |= N_PERIOD_UNK;
                    conf *= 0.3;
                    format!("{}|?{}", id, j)
                }
            };
            let sc = match tscale {
                Some(s) => s,
                None => {
                    // a single-column amount with explicit magnitude suffix is its own scale; otherwise unknown
                    notes |= N_UNKNOWN_SCALE;
                    conf *= 0.55;
                    1.0
                }
            };
            let cur = match (&tcur, n.cur) {
                (_, Some(c)) => c.to_string(),
                (Some((c, cf)), None) => {
                    conf *= *cf;
                    if *cf < 1.0 {
                        notes |= N_CUR_UNK;
                    }
                    c.clone()
                }
                (None, None) => {
                    notes |= N_CUR_UNK;
                    conf *= 0.6;
                    "?".to_string()
                }
            };
            if vals.len() != hdr_len {
                notes |= N_SHORT_ROW;
                conf *= 0.6;
            }
            if pc.restated {
                conf *= 0.9;
            }
            out.push(Ext { fact: Fact::num(&key, FKind::Line, n.v * sc, &cur, c.lo, c.hi), conf, notes, row: li, col: j });
        }
    }
    dedup_keys(&mut out);
    out
}
fn dedup_keys(out: &mut Vec<Ext>) {
    let mut seen: Vec<String> = Vec::new();
    let mut drop: Vec<usize> = Vec::new();
    for i in 0..out.len() {
        if let Some(p) = seen.iter().position(|k| *k == out[i].fact.key) {
            drop.push(i);
            // lower the confidence of the kept one
            let kk = out.iter().position(|e| e.fact.key == seen[p]).unwrap();
            out[kk].conf *= 0.6;
            out[kk].notes |= N_DUP;
        } else {
            seen.push(out[i].fact.key.clone());
        }
    }
    for i in drop.into_iter().rev() {
        out.remove(i);
    }
}

// ---------------------------------------------------------------- PAPER extraction
fn metric_of(txt: &str, fuzzy: bool) -> Option<(&'static str, f64)> {
    let l = txt.to_lowercase();
    let exact = [("accuracy", "acc"), ("f1", "f1"), ("bleu", "bleu"), ("latency", "lat"), ("parameter", "params"), ("params", "params"), ("memory", "mem"), ("error rate", "err")];
    for (w, id) in exact {
        if l.contains(w) {
            return Some((id, 0.92));
        }
    }
    if fuzzy {
        for tok in l.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()) {
            for (p, id) in [("acc", "acc"), ("mem", "mem"), ("err", "err"), ("bleu", "bleu"), ("lat", "lat"), ("param", "params")] {
                if tok.starts_with(p) {
                    return Some((id, 0.6));
                }
            }
        }
    }
    None
}
fn unit_in_hdr(h: &str) -> Option<&'static str> {
    let l = h.to_lowercase();
    let mut inner = String::new();
    for (o, c) in [('(', ')'), ('[', ']')] {
        if let (Some(a), Some(b)) = (l.find(o), l.rfind(c)) {
            if b > a {
                inner = l[a + 1..b].trim().to_string();
            }
        }
    }
    match inner.as_str() {
        "%" => Some("%"),
        "ms" => Some("ms"),
        "m" => Some("M"),
        "gb" => Some("GB"),
        _ => None,
    }
}
fn is_sep_line(l: &str) -> bool {
    !l.trim().is_empty() && l.chars().all(|c| c == '-' || c == '|' || c == ':' || c == ' ' || c == '+')
}
pub fn extract_paper(text: &str, cfg: &ExCfg) -> Vec<Ext> {
    let lns = lines_of(text);
    let mut out: Vec<Ext> = Vec::new();
    // ---- tables
    struct Blk {
        hdr: Vec<Cell>,
        rows: Vec<(usize, Vec<Cell>)>,
    }
    let mut blocks: Vec<Blk> = Vec::new();
    let mut cur: Option<Blk> = None;
    for (li, (lo, ln)) in lns.iter().enumerate() {
        if is_sep_line(ln) {
            continue;
        }
        let cells = split_cells(*lo, ln);
        if cells.len() < 3 {
            if let Some(b) = cur.take() {
                if !b.rows.is_empty() {
                    blocks.push(b);
                }
            }
            continue;
        }
        let nnum = cells[1..].iter().filter(|c| parse_num(&c.txt).is_some() || num_with_unit(&c.txt).is_some()).count();
        if nnum == 0 && cells[1..].iter().filter(|c| !c.txt.is_empty()).count() >= 2 {
            if let Some(b) = cur.take() {
                if !b.rows.is_empty() {
                    blocks.push(b);
                }
            }
            cur = Some(Blk { hdr: cells, rows: Vec::new() });
        } else if nnum >= 2 {
            if let Some(b) = cur.as_mut() {
                b.rows.push((li, cells));
            }
        } else if let Some(b) = cur.take() {
            if !b.rows.is_empty() {
                blocks.push(b);
            }
        }
    }
    if let Some(b) = cur.take() {
        if !b.rows.is_empty() {
            blocks.push(b);
        }
    }
    let mut models_seen: Vec<String> = Vec::new();
    for blk in &blocks {
        let mh = blk.hdr[1..].iter().filter(|c| metric_of(&c.txt, cfg.fuzzy).is_some()).count();
        let mr = blk.rows.iter().filter(|(_, c)| metric_of(&c[0].txt, cfg.fuzzy).is_some()).count();
        let transposed = mr > mh;
        if transposed {
            for (li, cells) in &blk.rows {
                let Some((mid, mconf)) = metric_of(&cells[0].txt, cfg.fuzzy) else { continue };
                let unit_h = unit_in_hdr(&cells[0].txt);
                for (j, c) in cells[1..].iter().enumerate() {
                    let model = blk.hdr.get(j + 1).map(|h| clean_name(&h.txt)).unwrap_or_default();
                    if model.is_empty() {
                        continue;
                    }
                    if !models_seen.contains(&model) {
                        models_seen.push(model.clone());
                    }
                    if let Some(e) = cell_fact(&model, mid, mconf, unit_h, c, *li, j) {
                        out.push(e);
                    }
                }
            }
        } else {
            for (li, cells) in &blk.rows {
                let model = clean_name(&cells[0].txt);
                if model.is_empty() {
                    continue;
                }
                if !models_seen.contains(&model) {
                    models_seen.push(model.clone());
                }
                for (j, c) in cells[1..].iter().enumerate() {
                    let Some(h) = blk.hdr.get(j + 1) else { continue };
                    let Some((mid, mconf)) = metric_of(&h.txt, cfg.fuzzy) else { continue };
                    if let Some(e) = cell_fact(&model, mid, mconf, unit_in_hdr(&h.txt), c, *li, j) {
                        out.push(e);
                    }
                }
            }
        }
    }
    // ---- params
    let low = text.to_lowercase();
    for (id, phr_after, phr_before) in [("bs", Some("batch size of "), None), ("lr", Some("learning rate of "), None), ("dropout", Some("dropout rate of "), None), ("ep", None, Some(" epochs")), ("n", None, Some(" participants")), ("gpus", None, Some(" gpus"))] {
        let found = if let Some(p) = phr_after {
            low.find(p).and_then(|i| num_token_at(text, i + p.len()))
        } else {
            let p = phr_before.unwrap();
            low.find(p).and_then(|i| num_token_before(text, i))
        };
        if let Some((lo, hi)) = found {
            let tok = &text[lo..hi];
            let t = tok.replace(',', "");
            if let Ok(v) = t.parse::<f64>() {
                out.push(Ext { fact: Fact::num(&format!("par|{}", id), FKind::Param, v, "", lo, hi), conf: 0.88, notes: 0, row: 0, col: 0 });
            }
        }
    }
    // ---- claims
    if models_seen.is_empty() {
        return out;
    }
    let mut pos = 0usize;
    for (lo, ln) in &lns {
        // split into sentences
        let mut s = 0;
        let bytes = ln.as_bytes();
        let mut i = 0;
        let mut sents: Vec<(usize, usize)> = Vec::new();
        while i + 1 < bytes.len() {
            if bytes[i] == b'.' && bytes[i + 1] == b' ' && i + 2 < bytes.len() && bytes[i + 2].is_ascii_uppercase() {
                sents.push((s, i + 1));
                s = i + 2;
            }
            i += 1;
        }
        sents.push((s, ln.len()));
        for (a, b) in sents {
            if b <= a {
                continue;
            }
            let sent = &ln[a..b];
            if let Some(e) = claim_of(sent, lo + a, &models_seen, cfg) {
                out.push(e);
            }
        }
        pos += 1;
    }
    let _ = pos;
    out
}
fn clean_name(t: &str) -> String {
    t.trim().trim_matches('*').trim().to_string()
}
fn num_with_unit(c: &str) -> Option<(f64, String)> {
    let n = parse_num(c)?;
    Some((n.v, n.unit))
}
fn cell_fact(model: &str, mid: &'static str, mconf: f64, hdr_unit: Option<&'static str>, c: &Cell, li: usize, j: usize) -> Option<Ext> {
    let n = parse_num(&c.txt)?;
    let m = METS.iter().find(|m| m.id == mid)?;
    let mut conf = mconf * n.conf * 0.97;
    let mut notes = if mconf < 0.9 { N_FUZZY } else { 0 };
    // unit: header unit, else inline unit, else metric default (assumption)
    let mut v = n.v;
    if !n.unit.is_empty() {
        if n.unit == "ms" && mid == "lat" {
        } else {
            // unknown inline unit (e.g. "s"): assume the metric default
            conf *= 0.5;
            notes |= N_UNIT_DEFAULT;
        }
    } else if hdr_unit.is_none() && m.unit != "pts" {
        // header with no recognised unit: default assumption
        conf *= 0.7;
        notes |= N_UNIT_DEFAULT;
    } else if let Some(u) = hdr_unit {
        if u != m.unit {
            conf *= 0.5;
            notes |= N_UNIT_DEFAULT;
        }
    }
    if n.pct {
        v = n.v;
    }
    Some(Ext { fact: Fact::num(&format!("cell|{}|{}", model, mid), FKind::Cell, v, m.unit, c.lo, c.hi), conf, notes, row: li, col: j })
}
/// number token starting at byte `i` (digits, ',', '.', 'e-'): returns span
pub fn num_token_at(text: &str, i: usize) -> Option<(usize, usize)> {
    let b = text.as_bytes();
    let mut j = i;
    while j < b.len() && (b[j].is_ascii_digit() || b[j] == b',' && j + 1 < b.len() && b[j + 1].is_ascii_digit() || b[j] == b'.' && j + 1 < b.len() && b[j + 1].is_ascii_digit() || (b[j] == b'e' && j + 2 < b.len() && b[j + 1] == b'-' && b[j + 2].is_ascii_digit()) || (b[j] == b'-' && j > i && b[j - 1] == b'e')) {
        j += 1;
    }
    if j > i && b[i].is_ascii_digit() { Some((i, j)) } else { None }
}
pub fn num_token_before(text: &str, end: usize) -> Option<(usize, usize)> {
    let b = text.as_bytes();
    let mut j = end;
    if j == 0 || !b[j - 1].is_ascii_digit() {
        return None;
    }
    while j > 0 && (b[j - 1].is_ascii_digit() || (b[j - 1] == b',' && j >= 2 && b[j - 2].is_ascii_digit()) || (b[j - 1] == b'.' && j >= 2 && b[j - 2].is_ascii_digit())) {
        j -= 1;
    }
    Some((j, end))
}
fn claim_of(sent: &str, lo: usize, models: &[String], cfg: &ExCfg) -> Option<Ext> {
    // entities in order of appearance
    let mut ents: Vec<(usize, &String)> = Vec::new();
    for m in models {
        if let Some(p) = sent.find(m.as_str()) {
            ents.push((p, m));
        }
    }
    ents.sort_by_key(|x| x.0);
    if ents.len() < 2 {
        return None;
    }
    let l = sent.to_lowercase();
    // verb (dev lexicon first, then stems if fuzzy)
    let dev_verbs = ["outperform", "improves on", "improve on", "better than"];
    let mut vpos: Option<(usize, f64)> = None;
    for v in dev_verbs {
        if let Some(p) = l.find(v) {
            vpos = Some((p, 0.9));
            break;
        }
    }
    if vpos.is_none() && cfg.fuzzy {
        for v in ["improv", "better"] {
            if let Some(p) = l.find(v) {
                vpos = Some((p, 0.6));
                break;
            }
        }
    }
    let (vp, vconf) = vpos?;
    // metric
    let mut metric: Option<(&'static str, f64)> = None;
    for w in ["accuracy", "f1", "bleu", "latency", "parameter count", "memory", "error rate"] {
        if l.contains(w) {
            metric = metric_of(w, false);
            break;
        }
    }
    if metric.is_none() && cfg.fuzzy {
        for tok in l.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()) {
            for (p, id) in [("acc", "acc"), ("mem", "mem"), ("err", "err"), ("bleu", "bleu"), ("lat", "lat")] {
                if tok.starts_with(p) {
                    metric = Some((id, 0.6));
                    break;
                }
            }
            if metric.is_some() {
                break;
            }
        }
    }
    let (mid, mconf) = metric?;
    let before = back(&l, vp, 24);
    let neg = before.contains("not ") || before.contains("fails") || before.contains("n't");
    let hedged = ["suggest", "appears", " may ", "might", "could", "possibly"].iter().any(|h| l.contains(h));
    // citation
    let mut cite = 0u32;
    if let Some(p) = sent.find('[') {
        if let Some((a, b)) = num_token_at(sent, p + 1) {
            cite = sent[a..b].parse().unwrap_or(0);
        }
    }
    let key = format!("clm|{}|{}|{}", ents[0].1, ents[1].1, mid);
    let mut f = Fact::num(&key, FKind::Claim, if neg { -1.0 } else { 1.0 }, &format!("{}|{}", if hedged { "hedged" } else { "firm" }, cite), lo, lo + sent.len());
    f.sval = String::new();
    Some(Ext { fact: f, conf: 0.9 * vconf * mconf.max(0.6) / 0.9_f64.max(mconf).min(1.0) * if cite == 0 { 0.8 } else { 1.0 }, notes: if vconf < 0.9 || mconf < 0.9 { N_FUZZY } else { 0 }, row: 0, col: 0 })
}

// ---------------------------------------------------------------- BUSINESS extraction
pub fn extract_biz(text: &str, cfg: &ExCfg) -> Vec<Ext> {
    let mut out = Vec::new();
    extract_contract(text, &mut out);
    extract_kpi(text, cfg, &mut out);
    out
}
fn find_ci(hay: &str, needle: &str, from: usize) -> Option<usize> {
    if from > hay.len() {
        return None;
    }
    hay.to_lowercase().get(from..).and_then(|h| h.find(&needle.to_lowercase())).map(|p| p + from)
}
/// dev date scanner: "Month D, YYYY" | "YYYY-MM-DD" | "MM/DD/YYYY" (MDY)
pub fn dates_dev(text: &str) -> Vec<(usize, usize, i64)> {
    let mut v = Vec::new();
    let b = text.as_bytes();
    for (mi, m) in MONTHS.iter().enumerate() {
        let mut from = 0;
        while let Some(p) = text[from..].find(m) {
            let s = from + p;
            let mut j = s + m.len();
            from = j;
            if j < b.len() && b[j] == b' ' {
                j += 1;
                let d0 = j;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                if j > d0 && j - d0 <= 2 && text[j..].starts_with(", ") {
                    let y0 = j + 2;
                    let mut k = y0;
                    while k < b.len() && b[k].is_ascii_digit() {
                        k += 1;
                    }
                    if k - y0 == 4 {
                        let d: i64 = text[d0..j].parse().unwrap();
                        let y: i64 = text[y0..k].parse().unwrap();
                        if (1..=31).contains(&d) {
                            v.push((s, k, days_from_civil(y, mi as i64 + 1, d)));
                        }
                    }
                }
            }
        }
    }
    // numeric forms
    let mut i = 0;
    while i + 10 <= b.len() {
        if !text.is_char_boundary(i) {
            i += 1;
            continue;
        }
        let s = &text[i..];
        if b[i].is_ascii_digit() && !(i > 0 && (b[i - 1].is_ascii_digit() || b[i - 1] == b'.' || b[i - 1] == b'/' || b[i - 1] == b'-')) {
            // YYYY-MM-DD
            if s.len() >= 10 && b[i + 4] == b'-' && b[i + 7] == b'-' && s[..4].bytes().all(|c| c.is_ascii_digit()) && s[5..7].bytes().all(|c| c.is_ascii_digit()) && s[8..10].bytes().all(|c| c.is_ascii_digit()) {
                let (y, m, d): (i64, i64, i64) = (s[..4].parse().unwrap(), s[5..7].parse().unwrap(), s[8..10].parse().unwrap());
                if (1..=12).contains(&m) && (1..=31).contains(&d) {
                    v.push((i, i + 10, days_from_civil(y, m, d)));
                    i += 10;
                    continue;
                }
            }
            // MM/DD/YYYY
            if s.len() >= 10 && b[i + 2] == b'/' && b[i + 5] == b'/' && s[..2].bytes().all(|c| c.is_ascii_digit()) && s[3..5].bytes().all(|c| c.is_ascii_digit()) && s[6..10].bytes().all(|c| c.is_ascii_digit()) {
                let (m, d, y): (i64, i64, i64) = (s[..2].parse().unwrap(), s[3..5].parse().unwrap(), s[6..10].parse().unwrap());
                if (1..=12).contains(&m) && (1..=31).contains(&d) {
                    v.push((i, i + 10, days_from_civil(y, m, d)));
                    i += 10;
                    continue;
                }
            }
        }
        i += 1;
    }
    v.sort_by_key(|x| x.0);
    v
}
/// amounts: currency symbol / code followed by number; returns (lo, hi, value, conf)
pub fn amounts_dev(text: &str) -> Vec<(usize, usize, f64, f64, &'static str)> {
    let mut v = Vec::new();
    let b = text.as_bytes();
    let mut i = 0;
    while i < text.len() {
        if !text.is_char_boundary(i) {
            i += 1;
            continue;
        }
        let rest = &text[i..];
        let (sym_len, cur): (usize, &'static str) = if rest.starts_with('$') {
            (1, "USD")
        } else if rest.starts_with('\u{20ac}') {
            (3, "EUR")
        } else if rest.starts_with("USD ") && !(i > 0 && b[i - 1].is_ascii_alphabetic()) {
            (4, "USD")
        } else if rest.starts_with("EUR ") && !(i > 0 && b[i - 1].is_ascii_alphabetic()) {
            (4, "EUR")
        } else {
            i += 1;
            continue;
        };
        let n0 = i + sym_len;
        if n0 < b.len() && b[n0].is_ascii_digit() {
            let mut j = n0;
            while j < b.len() && (b[j].is_ascii_digit() || ((b[j] == b',' || b[j] == b'.') && j + 1 < b.len() && b[j + 1].is_ascii_digit())) {
                j += 1;
            }
            if let Some((val, conf, _)) = parse_core(&text[n0..j]) {
                v.push((i, j, val, conf, cur));
            }
            i = j;
        } else {
            i += sym_len;
        }
    }
    v
}
fn sentence_bounds(text: &str, p: usize) -> (usize, usize) {
    let b = text.as_bytes();
    let mut s = p;
    while s > 0 && b[s - 1] != b'\n' && !(b[s - 1] == b' ' && s >= 2 && b[s - 2] == b'.' && s >= 3 && !b[s - 3].is_ascii_digit()) {
        s -= 1;
    }
    let mut e = p;
    while e < b.len() && b[e] != b'\n' && !(b[e] == b'.' && (e + 1 >= b.len() || b[e + 1] == b' ' || b[e + 1] == b'\n') && !(e > 0 && b[e - 1].is_ascii_digit() && e + 1 < b.len() && b[e + 1].is_ascii_digit())) {
        e += 1;
    }
    (s, e.min(b.len()))
}
fn extract_contract(text: &str, out: &mut Vec<Ext>) {
    // parties
    if let (Some(is), Some(ic)) = (text.find(" (\"Supplier\")"), text.find(" (\"Customer\")")) {
        if let Some(ib) = text[..is].rfind("between ") {
            let (lo, hi) = (ib + 8, is);
            out.push(Ext { fact: Fact { key: "party|supplier".into(), kind: FKind::Party, value: 0.0, sval: text[lo..hi].to_string(), unit: "".into(), lo, hi }, conf: 0.9, notes: 0, row: 0, col: 0 });
            if let Some(ia) = text[..ic].rfind("\") and ") {
                let (lo, hi) = (ia + 7, ic);
                if hi > lo {
                    out.push(Ext { fact: Fact { key: "party|customer".into(), kind: FKind::Party, value: 0.0, sval: text[lo..hi].to_string(), unit: "".into(), lo, hi }, conf: 0.9, notes: 0, row: 0, col: 0 });
                }
            }
        }
    }
    // effective date: first dev-format date with a cue before it
    let ds = dates_dev(text);
    if let Some(&(lo, hi, d)) = ds.first() {
        let pre = &back(text, lo, 40).to_lowercase();
        let cue = ["as of", "effective", "entered into", "dated", "made on"].iter().any(|c| pre.contains(c)) && text.to_uppercase().contains("AGREEMENT");
        if cue {
            out.push(Ext { fact: Fact::num("date|effective", FKind::Date, d as f64, "", lo, hi), conf: 0.9, notes: 0, row: 0, col: 0 });
        }
    }
    // amounts
    let am = amounts_dev(text);
    let mut got_fee = false;
    let mut got_cap = false;
    for (lo, hi, val, pconf, cur) in am {
        let (s, e) = sentence_bounds(text, lo);
        let ctx = text[s..e].to_lowercase();
        let after = text[hi..(hi + 70).min(text.len())].to_lowercase();
        let after = after.split('\n').next().unwrap_or("").to_string();
        let first_amt_in_sent = {
            let pre = &text[s..lo];
            !pre.contains('$') && !pre.contains("\u{20ac}") && !pre.contains("USD ") && !pre.contains("EUR ")
        };
        if !first_amt_in_sent {
            continue;
        }
        if ctx.contains("liabil") && !got_cap {
            got_cap = true;
            out.push(Ext { fact: Fact::num("cap", FKind::Amount, val, cur, lo, hi), conf: 0.88 * pconf, notes: 0, row: 0, col: 0 });
        } else if (ctx.contains("fee") || ctx.contains("pay")) && !ctx.contains("setup") && !ctx.contains("one-time") && !ctx.contains("signature") && !got_fee {
            got_fee = true;
            let (div, pc, notes) = if after.contains("per month") || after.contains("monthly") {
                (1.0, 1.0, 0)
            } else if after.contains("per quarter") || after.contains("quarterly") {
                (3.0, 1.0, 0)
            } else if after.contains("per annum") || after.contains("annually") || after.contains("per year") {
                (12.0, 1.0, 0)
            } else {
                (1.0, 0.4, N_NOCUE)
            };
            out.push(Ext { fact: Fact::num("fee_monthly", FKind::Amount, val / div, cur, lo, hi), conf: 0.88 * pconf * pc, notes, row: 0, col: 0 });
        }
    }
    // term
    let low = text.to_lowercase();
    let mut from = 0;
    while let Some(p) = low[from..].find("term") {
        let s = from + p;
        from = s + 4;
        if let Some((n_lo, n_hi, months)) = number_then_unit(text, s + 4, 70, &["months", "month", "years", "year"]) {
            out.push(Ext { fact: Fact::num("term_months", FKind::Clause, months, "months", n_lo, n_hi), conf: 0.85, notes: 0, row: 0, col: 0 });
            break;
        }
    }
    // notice
    let mut from = 0;
    while let Some(p) = low[from..].find("notice") {
        let p = from + p;
        from = p + 6;
        let a = p.saturating_sub(40);
        let a = (a..=p).find(|&k| text.is_char_boundary(k)).unwrap_or(p);
        let seg = &text[a..p];
        if let Some(q) = seg.find(" days") {
            if let Some((lo, hi)) = num_token_before(text, a + q) {
                if let Ok(v) = text[lo..hi].parse::<f64>() {
                    out.push(Ext { fact: Fact::num("notice_days", FKind::Clause, v, "days", lo, hi), conf: 0.85, notes: 0, row: 0, col: 0 });
                    break;
                }
            }
        }
    }
    // interest
    if let Some(p) = low.find("interest at ") {
        if let Some((lo, hi)) = num_token_at(text, p + 12) {
            if text[hi..].starts_with('%') {
                if let Ok(v) = text[lo..hi].parse::<f64>() {
                    out.push(Ext { fact: Fact::num("int_pct", FKind::Clause, v, "pct", lo, hi), conf: 0.85, notes: 0, row: 0, col: 0 });
                }
            }
        }
    }
}
/// find, after `from` within `window` bytes, a number followed (optionally after ")" and spaces) by one of the unit words; years -> months
fn number_then_unit(text: &str, from: usize, window: usize, units: &[&str]) -> Option<(usize, usize, f64)> {
    let end = (from + window).min(text.len());
    let b = text.as_bytes();
    let mut i = from;
    while i < end {
        if b[i].is_ascii_digit() && !(i > 0 && (b[i - 1].is_ascii_digit() || b[i - 1] == b'.' || b[i - 1] == b',')) {
            let (lo, hi) = num_token_at(text, i)?;
            let mut j = hi;
            while j < b.len() && (b[j] == b')' || b[j] == b' ') {
                j += 1;
            }
            let rest = &text[j..];
            for u in units {
                if rest.starts_with(u) {
                    let v: f64 = text[lo..hi].replace(',', "").parse().ok()?;
                    let months = if u.starts_with("year") { v * 12.0 } else { v };
                    return Some((lo, hi, months));
                }
            }
            i = hi;
        } else {
            i += 1;
        }
    }
    None
}

pub fn strip_paren(l: &str) -> (String, String) {
    if let (Some(a), true) = (l.find('('), l.trim_end().ends_with(')')) {
        (l[..a].trim().to_string(), l[a..].to_string())
    } else {
        (l.trim().to_string(), String::new())
    }
}
pub fn kpi_lex() -> Lex {
    Lex::new(&kpi_dev_labels())
}
/// value token for a KPI: "$1.2M", "$850K", "$320", "3.2%", "41", "1,200" ; returns (value, unit, conf, notes)
pub fn kpi_value(tok: &str, kid: &str, tscale: Option<f64>) -> Option<(f64, &'static str, f64, u32)> {
    let n = parse_num(tok)?;
    let money = matches!(kid, "mrr" | "cac" | "arpu" | "ltv");
    let mut conf = n.conf;
    let mut notes = n.notes;
    if kid == "churn" || n.pct {
        if !n.pct {
            conf *= 0.5;
        }
        return Some((n.v, "pct", conf, notes));
    }
    if money {
        let mut v = n.v;
        if n.mult == 1.0 && n.cur.is_none() {
            if let Some(s) = tscale {
                v *= s;
            } else {
                conf *= 0.6;
                notes |= N_UNKNOWN_SCALE;
            }
        } else if n.mult == 1.0 && n.cur.is_some() && !n.unit.is_empty() {
            conf *= 0.4;
        }
        if !n.unit.is_empty() && n.mult == 1.0 {
            conf *= 0.7;
        }
        return Some((v, "USD", conf, notes));
    }
    Some((n.v, if kid == "nps" { "pts" } else { "count" }, conf, notes))
}
fn extract_kpi(text: &str, cfg: &ExCfg, out: &mut Vec<Ext>) {
    let lns = lines_of(text);
    let lex = kpi_lex();
    let mut hdr: Vec<Option<(i64, i64)>> = Vec::new();
    let mut tscale: Option<f64> = None;
    for (li, (lo, ln)) in lns.iter().enumerate() {
        let cells = split_cells(*lo, ln);
        // table header
        if cells.len() >= 3 && cells[1..].iter().filter(|c| quarter_in(&c.txt).is_some()).count() >= 2 {
            hdr = cells[1..].iter().map(|c| quarter_in(&c.txt)).collect();
            let (sc, _) = window_scale(&lns, li);
            tscale = sc;
            continue;
        }
        if cells.len() >= 3 && !hdr.is_empty() {
            let (lab0, par0) = strip_paren(&cells[0].txt);
            let row_scale = scale_word(&par0).or(tscale);
            if let Some((kid, lconf, lnotes)) = lex.lookup(&lab0, cfg.fuzzy) {
                for (j, c) in cells[1..].iter().enumerate() {
                    let Some(Some((q, y))) = hdr.get(j) else { continue };
                    if let Some((v, unit, pc, pn)) = kpi_value(&c.txt, kid, row_scale) {
                        out.push(Ext { fact: Fact::num(&format!("kpi|{}|Q{}-{}", kid, q, y), FKind::Kpi, v, unit, c.lo, c.hi), conf: 0.9 * lconf / 0.92 * pc, notes: lnotes | pn, row: li, col: j });
                    }
                }
                continue;
            }
        }
        // bullets: "- LABEL: V (Q3 2024) vs V (Q2 2024)"  and prose: "LABEL was V in Q3 2024, up from V in Q2 2024."
        let line = *ln;
        let (label, kid) = if let Some(rest) = line.strip_prefix("- ") {
            match rest.find(':') {
                Some(p) => (rest[..p].to_string(), lex.lookup(&rest[..p], cfg.fuzzy)),
                None => continue,
            }
        } else if let Some(p) = line.find(" was ") {
            (line[..p].to_string(), lex.lookup(&line[..p], cfg.fuzzy))
        } else {
            continue;
        };
        let Some((kid, lconf, lnotes)) = kid else { continue };
        let _ = label;
        // (value token, period) pairs: "V (Qn yyyy)" and "V in Qn yyyy"
        let mut pairs: Vec<(usize, usize, i64, i64)> = Vec::new();
        for pat in [" (Q", " in Q"] {
            let mut from = 0;
            while let Some(p) = line[from..].find(pat) {
                let s = from + p;
                from = s + pat.len();
                let qpos = s + pat.len() - 1;
                if let Some((q, y)) = quarter_in(&line[qpos..]) {
                    // value token = the whitespace-delimited token before position s
                    let mut a = s;
                    while a > 0 && line.as_bytes()[a - 1] != b' ' && line.as_bytes()[a - 1] != b':' {
                        a -= 1;
                    }
                    if a < s {
                        pairs.push((a, s, q, y));
                    }
                }
            }
        }
        for (a, bb, q, y) in pairs {
            let tok = &line[a..bb];
            if let Some((v, unit, pc, pn)) = kpi_value(tok, kid, None) {
                out.push(Ext { fact: Fact::num(&format!("kpi|{}|Q{}-{}", kid, q, y), FKind::Kpi, v, unit, lo + a, lo + bb), conf: 0.88 * lconf / 0.92 * pc, notes: lnotes | pn, row: li, col: 0 });
            }
        }
    }
    dedup_keys(out);
}

pub fn extract(world: World, text: &str, cfg: &ExCfg) -> Vec<Ext> {
    match world {
        World::Fin => extract_fin(text, cfg),
        World::Paper => extract_paper(text, cfg),
        World::Biz => extract_biz(text, cfg),
        // the controls: the extractor is run blind with the fin and biz scanners (it does not know the doc type)
        World::NoStruct | World::Tiny => {
            let mut v = extract_biz(text, cfg);
            v.extend(extract_fin(text, cfg));
            v
        }
    }
}
pub fn _unused(_: &Item) {
    let _ = (BS, CF, INC, MODELS, PARAMS);
}
