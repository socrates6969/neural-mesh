//! Independent VERIFIER. It is given only (document text, extracted facts with spans) -- not the extractor's confidence, notes or
//! scale/sign decisions. Checks: (1) source-span re-parse with a DIFFERENT parser (state machine on the raw span + neighbours: sign
//! markers, trailing minus, scale suffix, space-thousand continuation, separator ambiguity), (2) independent scale evidence (looks
//! at headers, footers, `'000`, `mn`, `Mio` ...), (3) domain plausibility (unit/magnitude priors), (4) cross-footing (gp = rev - cogs,
//! assets = liabilities + equity, cash-flow sum ...) with culprit localisation + repair candidates, (5) redundancy cross-checks (stated
//! growth vs computed, TCV = fee x term, weekday vs date, KPI change column), (6) claim coverage (claim-like sentences vs extracted).
use super::scan::*;
use super::*;

pub const V_SPAN: u32 = 1;
pub const V_SIGN: u32 = 2;
pub const V_SCALE: u32 = 4;
pub const V_FOOT: u32 = 8;
pub const V_PLAUS: u32 = 16;
pub const V_XCHK: u32 = 32;
pub const V_COVER: u32 = 64;
pub const V_AMBIG: u32 = 128;
pub const V_FOOT_AMBIG: u32 = 256;

#[derive(Clone, Debug, Default)]
pub struct Vfy {
    pub reasons: u32,
    pub repair: Option<f64>,
}
impl Vfy {
    pub fn flagged(&self) -> bool {
        self.reasons != 0
    }
}
#[derive(Clone, Copy, Debug)]
pub struct VCfg {
    pub span: bool,
    pub scale: bool,
    pub plaus: bool,
    pub foot: bool,
    pub xchk: bool,
    pub cover: bool,
    pub flag_ambiguous_foot: bool,
    /// verifier knows author-anticipated variants ('000, mn, Mio, unicode/trailing minus, EU separators, space-thousands). false = dev-knowledge only
    pub broad: bool,
}
impl Default for VCfg {
    fn default() -> Self {
        VCfg { span: true, scale: true, plaus: true, foot: true, xchk: true, cover: true, flag_ambiguous_foot: true, broad: true }
    }
}

// ---------------------------------------------------------------- independent number parser
#[derive(Clone, Debug)]
pub struct IP {
    pub v: f64,
    pub alt: Option<f64>,
    pub neg: bool,
    pub mult: f64,
    pub ambiguous: bool,
    pub cont: bool,
    pub sym: bool,
    pub pct: bool,
    pub sec: bool,
}
pub fn iparse(text: &str, lo: usize, hi: usize, allow_neg: bool, broad: bool) -> Option<IP> {
    if hi <= lo || hi > text.len() || !text.is_char_boundary(lo) || !text.is_char_boundary(hi) {
        return None;
    }
    let s = &text[lo..hi];
    let b = s.as_bytes();
    // locate first digit
    let d0 = b.iter().position(|c| c.is_ascii_digit())?;
    let mut d1 = d0;
    while d1 < b.len() && (b[d1].is_ascii_digit() || ((b[d1] == b',' || b[d1] == b'.') && d1 + 1 < b.len() && b[d1 + 1].is_ascii_digit())) {
        d1 += 1;
    }
    let core = &s[d0..d1];
    let before = &s[..d0];
    let after = &s[d1..];
    let prev_ch = text[..lo].chars().last();
    let next_str = &text[hi..];
    let mut neg = allow_neg && (before.contains('-') || (broad && before.contains('\u{2212}')) || (before.contains('(') && after.contains(')')));
    if !neg && allow_neg && broad {
        if let Some(c) = prev_ch {
            if (c == '-' || c == '\u{2212}') && text[..lo].chars().rev().nth(1).map_or(true, |p| p.is_whitespace() || p == '(' || p == '|') {
                neg = true;
            }
        }
        if let Some(rest) = after.strip_prefix('-').or_else(|| if after.is_empty() { next_str.strip_prefix('-') } else { None }) {
            if !rest.starts_with(|c: char| c.is_ascii_alphanumeric()) {
                neg = true;
            }
        }
        // parenthesis negative where the span excludes the closing paren
        if before.ends_with('(') || (prev_ch == Some('(') && before.is_empty()) {
            if after.contains(')') || next_str.starts_with(')') {
                neg = true;
            }
        }
    }
    // separator interpretation
    let seps: Vec<(usize, char)> = core.char_indices().filter(|(_, c)| *c == ',' || *c == '.').collect();
    let digits_only = |s: &str| -> String { s.chars().filter(|c| c.is_ascii_digit()).collect() };
    let (mut v, mut alt, mut ambiguous);
    let mut cont = false;
    let lead_dot = before.ends_with('.') && !before[..before.len() - 1].chars().last().map_or(false, |c| c.is_ascii_digit());
    if seps.is_empty() {
        v = if lead_dot { format!("0.{}", core).parse::<f64>().ok()? } else { core.parse::<f64>().ok()? };
        alt = None;
        ambiguous = false;
    } else {
        let kinds: std::collections::BTreeSet<char> = seps.iter().map(|x| x.1).collect();
        if kinds.len() == 2 {
            let (li, lc) = *seps.last().unwrap();
            let int = digits_only(&core[..li]);
            let frac = digits_only(&core[li + 1..]);
            v = format!("{}.{}", int, frac).parse().ok()?;
            let _ = lc;
            alt = None;
            ambiguous = false;
        } else {
            let c = seps[0].1;
            if seps.len() >= 2 {
                v = digits_only(core).parse().ok()?;
                alt = None;
                ambiguous = false;
            } else {
                let i = seps[0].0;
                let frac = &core[i + 1..];
                let int = &core[..i];
                if broad && frac.len() == 3 && int.len() <= 3 && !int.starts_with('0') {
                    // thousands or decimal?
                    ambiguous = true;
                    let th: f64 = format!("{}{}", int, frac).parse().ok()?;
                    let dec: f64 = format!("{}.{}", int, frac).parse().ok()?;
                    if c == ',' {
                        v = th;
                        alt = Some(dec);
                        ambiguous = false;
                    } else {
                        v = dec;
                        alt = Some(th);
                    }
                } else {
                    ambiguous = false;
                    alt = None;
                    v = if c == '.' { format!("{}.{}", int, frac).parse().ok()? } else if frac.len() <= 2 { format!("{}.{}", int, frac).parse().ok()? } else { digits_only(core).parse().ok()? };
                }
            }
        }
    }
    // space-thousand continuation: "25" + " 000"
    if broad && after.trim().is_empty() && !core.contains(',') && !core.contains('.') {
        let nb = next_str.as_bytes();
        let mut k = 0;
        let mut acc = core.to_string();
        while k + 4 <= nb.len() && nb[k] == b' ' && nb[k + 1].is_ascii_digit() && nb[k + 2].is_ascii_digit() && nb[k + 3].is_ascii_digit() && (k + 4 == nb.len() || !nb[k + 4].is_ascii_digit()) {
            acc.push_str(&next_str[k + 1..k + 4]);
            k += 4;
            cont = true;
        }
        if cont {
            v = acc.parse().ok()?;
            alt = None;
            ambiguous = false;
        }
    }
    // scientific notation 1e-4
    if after.starts_with('e') || after.starts_with('E') {
        let ex = &after[1..];
        let (sg, dg) = if let Some(r) = ex.strip_prefix('-') { (-1i32, r) } else if let Some(r) = ex.strip_prefix('+') { (1, r) } else { (1, ex) };
        let dd: String = dg.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !dd.is_empty() {
            let e: i32 = dd.parse().unwrap_or(0) * sg;
            v *= 10f64.powi(e);
            alt = None;
        }
    }
    // suffix: inside the span or right after it
    let tail: String = format!("{}{}", after, if after.trim().is_empty() || after.trim() == ")" { fwd(next_str, 8) } else { "" });
    let t = tail.trim_start();
    let mut mult = 1.0;
    let mut pct = false;
    let mut sec = false;
    if t.starts_with('%') {
        pct = true;
    } else if (broad && (t.starts_with("mn") || t.starts_with("Mio") || t.starts_with("mm "))) || (t.starts_with('M') && !t[1..].starts_with(|c: char| c.is_ascii_alphabetic())) {
        mult = 1e6;
    } else if (t.starts_with('K') || t.starts_with('k')) && !t[1..].starts_with(|c: char| c.is_ascii_alphabetic()) {
        mult = 1e3;
    } else if (broad && t.starts_with("bn")) || (t.starts_with('B') && !t[1..].starts_with(|c: char| c.is_ascii_alphabetic())) {
        mult = 1e9;
    } else if t.starts_with("s ") || t == "s" || t.starts_with("s)") || t.starts_with("s,") {
        sec = true;
    }
    let sym = before.contains('$') || before.contains('\u{20ac}') || before.contains('\u{a3}') || before.contains("USD") || before.contains("EUR");
    Some(IP { v: v * mult, alt: alt.map(|a| a * mult), neg, mult, ambiguous, cont, sym, pct, sec })
}

// ---------------------------------------------------------------- independent scale evidence
fn scale_marker(l: &str, broad: bool) -> Option<f64> {
    let l = l.to_lowercase();
    if l.contains("thousand") || (broad && (l.contains("'000") || l.contains("000s") || l.contains("(000)") || l.contains(" in 000"))) {
        Some(1e3)
    } else if l.contains("million") || (broad && (l.contains(" mn") || l.contains(" mio") || l.contains("mio ") || l.contains(" mm "))) {
        Some(1e6)
    } else if l.contains("billion") {
        Some(1e9)
    } else {
        None
    }
}
/// (scale, found) for a span at byte `lo`: nearest scale marker above (<= 12 lines), else a footer "unless otherwise stated", else the first lines
fn iscale(lns: &[(usize, &str)], lo: usize, broad: bool) -> (f64, bool) {
    let li = lns.iter().rposition(|(s, _)| *s <= lo).unwrap_or(0);
    for k in (li.saturating_sub(40)..=li).rev() {
        if let Some(s) = scale_marker(lns[k].1, broad) {
            return (s, true);
        }
    }
    for (_, l) in lns.iter().rev().take(6) {
        let ll = l.to_lowercase();
        if (ll.contains("unless otherwise") || ll.contains("all amounts") || ll.contains("amounts in")) && scale_marker(l, broad).is_some() {
            return (scale_marker(l, broad).unwrap(), true);
        }
    }
    for (_, l) in lns.iter().take(6) {
        if let Some(s) = scale_marker(l, broad) {
            return (s, true);
        }
    }
    (1.0, false)
}

// ---------------------------------------------------------------- plausibility
fn plaus_cell(mid: &str, v: f64) -> Option<f64> {
    // returns Some(repair) when implausible and a unit-rescale explains it, Some(NaN) when implausible without repair
    let (lo, hi) = match mid {
        "acc" | "f1" | "err" | "bleu" => (0.0, 100.0),
        "lat" => (1.0, 5000.0),
        "params" => (1.0, 5000.0),
        "mem" => (0.1, 200.0),
        _ => return None,
    };
    if v >= lo && v <= hi {
        return None;
    }
    if v > 0.0 && v < lo && v * 1000.0 >= lo && v * 1000.0 <= hi {
        return Some(v * 1000.0);
    }
    if v > hi && v / 1000.0 >= lo && v / 1000.0 <= hi {
        return Some(v / 1000.0);
    }
    Some(f64::NAN)
}

// ---------------------------------------------------------------- footing
const REL: &[(&str, &[(&str, f64)])] = &[
    ("gp", &[("rev", 1.0), ("cogs", -1.0)]),
    ("opinc", &[("gp", 1.0), ("opex", -1.0)]),
    ("pbt", &[("opinc", 1.0), ("int", -1.0)]),
    ("ni", &[("pbt", 1.0), ("tax", -1.0)]),
    ("tca", &[("cash", 1.0), ("rec", 1.0), ("inv", 1.0)]),
    ("ta", &[("tca", 1.0), ("ppe", 1.0)]),
    ("tcl", &[("ap", 1.0), ("std", 1.0)]),
    ("tl", &[("tcl", 1.0), ("ltd", 1.0)]),
    ("ta", &[("tl", 1.0), ("eq", 1.0)]),
    ("tle", &[("tl", 1.0), ("eq", 1.0)]),
    ("ncc", &[("cfo", 1.0), ("cfi", 1.0), ("cff", 1.0)]),
];
const COST: [&str; 4] = ["cogs", "opex", "int", "tax"];
fn magv(id: &str, v: f64) -> f64 {
    if COST.contains(&id) { v.abs() } else { v }
}
fn alts_of(v: f64) -> Vec<f64> {
    let mut a = vec![-v, v * 1000.0, v / 1000.0, v * 10.0, v / 10.0];
    let s = format!("{}", v.abs() as i64);
    let bs = s.as_bytes();
    for i in 0..bs.len().saturating_sub(1) {
        let mut c = bs.to_vec();
        c.swap(i, i + 1);
        if let Ok(x) = String::from_utf8(c).unwrap().parse::<i64>() {
            let f = x as f64 * if v < 0.0 { -1.0 } else { 1.0 };
            if (f - v).abs() > 0.0 {
                a.push(f);
            }
        }
    }
    a
}

/// verify one document. `exts` = the facts that survived the extractor's abstention threshold.
pub fn verify_doc(world: World, text: &str, exts: &[Ext], vc: &VCfg) -> Vec<Vfy> {
    let lns = lines_of(text);
    let mut out: Vec<Vfy> = vec![Vfy::default(); exts.len()];
    let mut work: Vec<f64> = exts.iter().map(|e| e.fact.value).collect();
    // ---- (1)+(2)+(3): span / sign / scale / plausibility per fact
    for (i, e) in exts.iter().enumerate() {
        let f = &e.fact;
        let numeric_kind = matches!(f.kind, FKind::Line | FKind::Cell | FKind::Kpi | FKind::Amount | FKind::Param | FKind::Clause | FKind::Date);
        if !numeric_kind || f.hi <= f.lo || f.kind == FKind::Date {
            continue;
        }
        let Some(ip) = iparse(text, f.lo, f.hi, matches!(f.kind, FKind::Line | FKind::Cell | FKind::Kpi), vc.broad) else {
            if vc.span {
                out[i].reasons |= V_SPAN;
            }
            continue;
        };
        let mut expected = ip.v;
        let mut ok_scale_applied = false;
        match f.kind {
            FKind::Line => {
                if vc.scale {
                    let (sc, _found) = iscale(&lns, f.lo, vc.broad);
                    expected *= if ip.mult == 1.0 { sc } else { 1.0 };
                    ok_scale_applied = true;
                }
            }
            FKind::Kpi => {
                if vc.scale && !ip.sym && ip.mult == 1.0 && !ip.pct && matches!(f.unit.as_str(), "USD") {
                    let (sc, _) = iscale(&lns, f.lo, vc.broad);
                    expected *= sc;
                    ok_scale_applied = true;
                }
            }
            FKind::Cell => {
                if ip.sec && f.unit == "ms" {
                    expected *= 1000.0;
                }
            }
            _ => {}
        }
        let _ = ok_scale_applied;
        if ip.neg && expected > 0.0 {
            expected = -expected;
        }
        let ev = work[i];
        if f.kind == FKind::Amount || f.kind == FKind::Clause || f.kind == FKind::Param {
            // amounts may be normalised to monthly by the extractor; compare only the digits family (value or value x period divisor)
            let mut matched = false;
            for dv in [1.0, 3.0, 12.0] {
                if close(expected / dv, ev) || close(expected, ev * dv) {
                    matched = true;
                }
            }
            if f.kind == FKind::Clause && f.key == "term_months" {
                matched |= close(expected * 12.0, ev);
            }
            if !matched && vc.span {
                out[i].reasons |= V_SPAN;
                out[i].repair = Some(expected);
            }
            if ip.cont && vc.span {
                out[i].reasons |= V_SPAN;
            }
            continue;
        }
        if !close(expected, ev) {
            if vc.span || vc.scale {
                let r = if close(-expected, ev) {
                    V_SIGN
                } else {
                    let ratio = if ev != 0.0 { (expected / ev).abs() } else { 0.0 };
                    let lg = ratio.log10();
                    if lg.abs() > 0.5 && (lg - lg.round()).abs() < 1e-6 { V_SCALE } else { V_SPAN }
                };
                // an ablated check must not leak through the other one (post-hoc fix of the h19 ablation arms)
                let enabled = if r == V_SCALE { vc.scale } else { vc.span };
                if enabled {
                    out[i].reasons |= r;
                    out[i].repair = Some(expected);
                    work[i] = expected;
                }
            }
        } else if ip.ambiguous && vc.span && (world == World::Fin) {
            out[i].reasons |= V_AMBIG;
            if let Some(a) = ip.alt {
                out[i].repair = Some(a * if ip.neg { -1.0 } else { 1.0 } * iscale(&lns, f.lo, vc.broad).0);
            }
        }
        // plausibility
        if vc.plaus && f.kind == FKind::Cell {
            let mid = f.key.rsplit('|').next().unwrap_or("");
            if let Some(rep) = plaus_cell(mid, work[i]) {
                out[i].reasons |= V_PLAUS;
                if !rep.is_nan() {
                    out[i].repair = Some(rep);
                }
            }
        }
        if vc.plaus && f.kind == FKind::Kpi {
            let kid = f.key.split('|').nth(1).unwrap_or("");
            let v = work[i];
            let bad = match kid {
                "churn" => !(0.0..=100.0).contains(&v),
                "nps" => !(-100.0..=100.0).contains(&v),
                "mrr" => !(1e3..=1e9).contains(&v),
                "cac" | "arpu" | "ltv" => !(1.0..=1e6).contains(&v),
                _ => false,
            };
            if bad {
                out[i].reasons |= V_PLAUS;
            }
        }
        if vc.plaus && f.kind == FKind::Param {
            let id = f.key.split('|').nth(1).unwrap_or("");
            let v = work[i];
            let bad = match id {
                "lr" | "dropout" => !(0.0..1.0).contains(&v),
                "ep" => !(1.0..=2000.0).contains(&v),
                "bs" => !(1.0..=100000.0).contains(&v),
                _ => false,
            };
            if bad {
                out[i].reasons |= V_PLAUS;
            }
        }
    }
    // ---- (4) footing for financial statements
    if vc.foot && world == World::Fin {
        let mut periods: Vec<String> = exts.iter().filter_map(|e| e.fact.key.split('|').nth(1).map(|s| s.to_string())).collect();
        periods.sort();
        periods.dedup();
        for p in periods {
            let idx_of = |id: &str| exts.iter().position(|e| e.fact.key == format!("{}|{}", id, p));
            let mut viol = std::collections::BTreeMap::<&str, usize>::new();
            let mut okc = std::collections::BTreeMap::<&str, usize>::new();
            let mut violated: Vec<usize> = Vec::new();
            for (ri, (tg, terms)) in REL.iter().enumerate() {
                let (Some(it_), tis) = (idx_of(tg), terms.iter().map(|(m, _)| idx_of(m)).collect::<Vec<_>>()) else { continue };
                if tis.iter().any(|x| x.is_none()) {
                    continue;
                }
                let tv = magv(tg, work[it_]);
                let sum: f64 = terms.iter().zip(&tis).map(|((m, s), ix)| s * magv(m, work[ix.unwrap()])).sum();
                let bad = (tv - sum).abs() > 1e-6 * tv.abs().max(sum.abs()).max(1.0);
                let mut members: Vec<&str> = vec![tg];
                members.extend(terms.iter().map(|(m, _)| *m));
                for m in &members {
                    *(if bad { &mut viol } else { &mut okc }).entry(*m).or_insert(0) += 1;
                }
                if bad {
                    violated.push(ri);
                }
            }
            if violated.is_empty() {
                continue;
            }
            let maxv = viol.values().copied().max().unwrap_or(0);
            let cands: Vec<&str> = viol.iter().filter(|(m, v)| **v == maxv && okc.get(*m).copied().unwrap_or(0) == 0).map(|(m, _)| *m).collect();
            // try to explain each candidate: solve its violated relation(s) for the candidate and compare with alternatives
            let mut explained: Vec<(usize, f64)> = Vec::new();
            for c in &cands {
                let Some(ci) = idx_of(c) else { continue };
                let ri = *violated.iter().find(|&&r| REL[r].0 == *c || REL[r].1.iter().any(|(m, _)| m == c)).unwrap();
                let (tg, terms) = REL[ri];
                let get = |m: &str| idx_of(m).map(|x| magv(m, work[x])).unwrap_or(0.0);
                let solved_mag = if tg == *c {
                    terms.iter().map(|(m, s)| s * get(m)).sum::<f64>()
                } else {
                    let sign_c = terms.iter().find(|(m, _)| m == c).unwrap().1;
                    let rest: f64 = terms.iter().filter(|(m, _)| m != c).map(|(m, s)| s * get(m)).sum();
                    (get(tg) - rest) / sign_c
                };
                let cur = work[ci];
                let solved = if COST.contains(c) { solved_mag * if cur < 0.0 { -1.0 } else { 1.0 } } else { solved_mag };
                let mut al = alts_of(cur);
                // wrong-period / wrong-cell alternatives: other extracted values with the same attribute
                for (k, e2) in exts.iter().enumerate() {
                    if k != ci && e2.fact.key.split('|').next() == Some(*c) {
                        al.push(work[k]);
                    }
                }
                if al.iter().any(|a| (a - solved).abs() <= 1e-6 * solved.abs().max(1.0)) || (solved_mag.abs() - magv(c, cur).abs()).abs() < 1e-9 {
                    explained.push((ci, solved));
                }
            }
            if explained.len() == 1 {
                out[explained[0].0].reasons |= V_FOOT;
                out[explained[0].0].repair = Some(explained[0].1);
            } else if vc.flag_ambiguous_foot {
                let list: Vec<usize> = if explained.is_empty() { cands.iter().filter_map(|c| idx_of(c)).collect() } else { explained.iter().map(|x| x.0).collect() };
                for ci in list {
                    out[ci].reasons |= V_FOOT_AMBIG;
                }
            }
        }
    }
    // ---- (5) redundancy cross-checks
    if vc.xchk {
        match world {
            World::Fin => {
                // "Net revenue increased g% compared with the prior year"
                if let Some(p) = text.find("Net revenue increased ") {
                    if let Some((lo, hi)) = num_token_at(text, p + 22) {
                        if let Ok(g) = text[lo..hi].parse::<f64>() {
                            let a = exts.iter().position(|e| e.fact.key.starts_with("rev|FY2023"));
                            let b = exts.iter().position(|e| e.fact.key.starts_with("rev|FY2022"));
                            if let (Some(a), Some(b)) = (a, b) {
                                let gc = (work[a] / work[b] - 1.0) * 100.0;
                                if (gc - g).abs() > 0.06 {
                                    out[a].reasons |= V_XCHK;
                                    out[b].reasons |= V_XCHK;
                                }
                            }
                        }
                    }
                }
            }
            World::Biz => {
                // weekday vs date
                if let Some(i) = exts.iter().position(|e| e.fact.key == "date|effective") {
                    let f = &exts[i].fact;
                    let pre = back(text, f.lo, 12);
                    for (wi, w) in WEEKDAYS.iter().enumerate() {
                        if pre.contains(w) && weekday(f.value as i64) != wi {
                            out[i].reasons |= V_XCHK;
                        }
                    }
                    let (y, _, _) = civil_from_days(f.value as i64);
                    if !(1990..=2040).contains(&y) {
                        out[i].reasons |= V_PLAUS;
                    }
                }
                // TCV = monthly fee x term
                if let Some(p) = text.find("total contract value of ") {
                    let am = amounts_dev(&text[p..(p + 60).min(text.len())]);
                    let fi = exts.iter().position(|e| e.fact.key == "fee_monthly");
                    let ti = exts.iter().position(|e| e.fact.key == "term_months");
                    if let (Some((_, _, tcv, _, _)), Some(fi), Some(ti)) = (am.first().cloned(), fi, ti) {
                        if !close(tcv, work[fi] * work[ti]) {
                            out[fi].reasons |= V_XCHK;
                            out[ti].reasons |= V_XCHK;
                        }
                    }
                }
                // KPI change column + commentary growth
                for (li, (lo, ln)) in lns.iter().enumerate() {
                    let cells = split_cells(*lo, ln);
                    if cells.len() == 4 {
                        let (c3, c4) = (&cells[3].txt, &cells[3].txt);
                        let _ = c4;
                        let is_pp = c3.contains("pp");
                        let is_pct = c3.ends_with('%') && !is_pp;
                        if is_pp || is_pct {
                            let ix: Vec<usize> = exts.iter().enumerate().filter(|(_, e)| e.row == li && e.fact.kind == FKind::Kpi).map(|(i, _)| i).collect();
                            if ix.len() == 2 {
                                let (cur_i, prev_i) = (ix[0], ix[1]);
                                let printed: Option<f64> = c3.trim().trim_end_matches("pp").trim().trim_end_matches('%').trim().replace('+', "").parse().ok();
                                if let Some(pr) = printed {
                                    let comp = if is_pp { work[cur_i] - work[prev_i] } else { (work[cur_i] / work[prev_i] - 1.0) * 100.0 };
                                    if (comp - pr).abs() > 0.06 + if is_pct { 0.0 } else { 0.0 } {
                                        out[cur_i].reasons |= V_XCHK;
                                        out[prev_i].reasons |= V_XCHK;
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(p) = text.find("MRR grew ") {
                    if let Some((lo, hi)) = num_token_at(text, p + 9) {
                        if let Ok(g) = text[lo..hi].parse::<f64>() {
                            let mut ks: Vec<(usize, &str)> = exts.iter().enumerate().filter(|(_, e)| e.fact.key.starts_with("kpi|mrr|")).map(|(i, e)| (i, e.fact.key.as_str())).collect();
                            ks.sort_by(|a, b| b.1.cmp(a.1));
                            // latest period = larger quarter label; compare any ordered pair consistent with g
                            if ks.len() == 2 {
                                let (a, b) = (ks[0].0, ks[1].0);
                                let (x, y) = (work[a], work[b]);
                                let ok = ((x / y - 1.0) * 100.0 - g).abs() <= 0.06 || ((y / x - 1.0) * 100.0 - g).abs() <= 0.06;
                                if !ok {
                                    out[a].reasons |= V_XCHK;
                                    out[b].reasons |= V_XCHK;
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    // ---- (6) claim coverage
    if vc.cover && world == World::Paper {
        let models: Vec<String> = {
            let mut m: Vec<String> = Vec::new();
            for e in exts.iter().filter(|e| e.fact.kind == FKind::Cell) {
                let n = e.fact.key.split('|').nth(1).unwrap_or("").to_string();
                if !m.contains(&n) {
                    m.push(n);
                }
            }
            m
        };
        if !models.is_empty() {
            let mut n_cand = 0;
            for (_, ln) in &lns {
                let mut s = 0;
                let b = ln.as_bytes();
                let mut sents: Vec<&str> = Vec::new();
                for i in 0..b.len().saturating_sub(2) {
                    if b[i] == b'.' && b[i + 1] == b' ' && b[i + 2].is_ascii_uppercase() {
                        sents.push(&ln[s..i + 1]);
                        s = i + 2;
                    }
                }
                sents.push(&ln[s..]);
                for sent in sents {
                    let ne = models.iter().filter(|m| sent.contains(m.as_str())).count();
                    if ne >= 2 && sent.contains('[') {
                        n_cand += 1;
                    }
                }
            }
            let n_claims = exts.iter().filter(|e| e.fact.kind == FKind::Claim).count();
            if n_cand != n_claims {
                for (i, e) in exts.iter().enumerate() {
                    if e.fact.kind == FKind::Claim {
                        out[i].reasons |= V_COVER;
                    }
                }
            }
        }
    }
    out
}

/// number of claim-like sentences without a matching extracted claim (coverage gap) -- doc-level check used by the pipeline
pub fn coverage_gap(text: &str, exts: &[Ext]) -> bool {
    let v = verify_doc(World::Paper, text, exts, &VCfg { span: false, scale: false, plaus: false, foot: false, xchk: false, cover: true, flag_ambiguous_foot: false, broad: true });
    v.iter().zip(exts).any(|(x, e)| e.fact.kind == FKind::Claim && x.reasons & V_COVER != 0) || {
        // no claim extracted at all but claim-like sentences exist
        let models: Vec<String> = exts.iter().filter(|e| e.fact.kind == FKind::Cell).filter_map(|e| e.fact.key.split('|').nth(1).map(|s| s.to_string())).collect();
        let nc = exts.iter().filter(|e| e.fact.kind == FKind::Claim).count();
        nc == 0 && !models.is_empty() && text.lines().any(|l| models.iter().filter(|m| l.contains(m.as_str())).count() >= 2 && l.contains('[') && !l.starts_with('['))
    }
}
