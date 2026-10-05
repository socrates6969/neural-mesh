//! Seeded document generators: financial statements (F), research papers (P), business documents (B) + controls.
//! `Bank::Dev` phrasing is what the extractor is built on (seeds 1-8). `Bank::Unseen` uses synonyms / layouts / number
//! conventions that were written BEFORE the extractor and that the extractor was never developed against
//! (author-contamination caveat: same author, so real-world generalisation is likely WORSE than the unseen numbers).
use super::*;

// ---------------------------------------------------------------- builder + table renderer
pub struct B {
    pub s: String,
}
impl B {
    pub fn new() -> B {
        B { s: String::new() }
    }
    pub fn pos(&self) -> usize {
        self.s.len()
    }
    pub fn push(&mut self, t: &str) {
        self.s.push_str(t);
    }
    pub fn line(&mut self, t: &str) {
        self.s.push_str(t);
        self.s.push('\n');
    }
    pub fn nl(&mut self) {
        self.s.push('\n');
    }
    /// push `t` and return its span
    pub fn span(&mut self, t: &str) -> (usize, usize) {
        let lo = self.pos();
        self.s.push_str(t);
        (lo, self.pos())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TStyle {
    Pipe,
    Aligned(bool),
    Csv,
}
pub struct Row {
    pub label: String,
    pub cells: Vec<String>,
}
fn pad_left(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if n >= w { s.to_string() } else { format!("{}{}", " ".repeat(w - n), s) }
}
fn pad_right(s: &str, w: usize, fill: char) -> String {
    let n = s.chars().count();
    if n >= w { s.to_string() } else { format!("{}{}", s, fill.to_string().repeat(w - n)) }
}
/// render a table; returns for every row the spans of its value cells (trimmed cell text)
pub fn render_table(b: &mut B, st: TStyle, header: &[String], rows: &[Row]) -> Vec<Vec<(usize, usize)>> {
    let ncol = header.len().saturating_sub(1);
    let mut out = Vec::new();
    match st {
        TStyle::Pipe => {
            let mut h = format!("| {} |", header[0]);
            for c in &header[1..] {
                h.push_str(&format!(" {} |", c));
            }
            b.line(&h);
            let mut sep = String::from("|---|");
            for _ in 0..ncol {
                sep.push_str("---:|");
            }
            b.line(&sep);
            for r in rows {
                b.push(&format!("| {} |", r.label));
                let mut sp = Vec::new();
                for c in &r.cells {
                    b.push(" ");
                    sp.push(b.span(c));
                    b.push(" |");
                }
                b.nl();
                out.push(sp);
            }
        }
        TStyle::Aligned(dots) => {
            let lw = rows.iter().map(|r| r.label.chars().count()).chain(std::iter::once(header[0].chars().count())).max().unwrap_or(8) + if dots { 4 } else { 2 };
            let mut cw = vec![0usize; ncol.max(1)];
            for (j, c) in header[1..].iter().enumerate() {
                cw[j] = c.chars().count();
            }
            for r in rows {
                for (j, c) in r.cells.iter().enumerate() {
                    if j < ncol {
                        cw[j] = cw[j].max(c.chars().count());
                    }
                }
            }
            let mut h = pad_right(&header[0], lw, ' ');
            for (j, c) in header[1..].iter().enumerate() {
                h.push_str("  ");
                h.push_str(&pad_left(c, cw[j]));
            }
            b.line(&h);
            b.line(&"-".repeat(h.chars().count().min(100)));
            for r in rows {
                let fill = if dots { '.' } else { ' ' };
                let lab = if dots { format!("{} ", r.label) } else { r.label.clone() };
                b.push(&pad_right(&lab, lw, fill));
                let mut sp = Vec::new();
                for (j, c) in r.cells.iter().enumerate() {
                    b.push("  ");
                    let w = cw[j.min(ncol.saturating_sub(1))];
                    let n = c.chars().count();
                    if w > n {
                        b.push(&" ".repeat(w - n));
                    }
                    sp.push(b.span(c));
                }
                b.nl();
                out.push(sp);
            }
        }
        TStyle::Csv => {
            b.line(&header.join(","));
            for r in rows {
                b.push(&format!("{},", if r.label.contains(',') { format!("\"{}\"", r.label) } else { r.label.clone() }));
                let mut sp = Vec::new();
                for (j, c) in r.cells.iter().enumerate() {
                    if j > 0 {
                        b.push(",");
                    }
                    if c.contains(',') {
                        b.push("\"");
                        sp.push(b.span(c));
                        b.push("\"");
                    } else {
                        sp.push(b.span(c));
                    }
                }
                b.nl();
                out.push(sp);
            }
        }
    }
    out
}

// ---------------------------------------------------------------- number formatting
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sep {
    Us,
    Eu,
    Space,
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Neg {
    Paren,
    Minus,
    Uni,
    Trail,
}
pub fn fmt_mag(v: i64, sep: Sep) -> String {
    let s = v.abs().to_string();
    let g = match sep {
        Sep::Us => ",",
        Sep::Eu => ".",
        Sep::Space => " ",
    };
    let mut out = String::new();
    let n = s.len();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (n - i) % 3 == 0 {
            out.push_str(g);
        }
        out.push(c);
    }
    out
}
pub fn fmt_signed(v: i64, sep: Sep, neg: Neg) -> String {
    let m = fmt_mag(v, sep);
    if v >= 0 {
        return m;
    }
    match neg {
        Neg::Paren => format!("({})", m),
        Neg::Minus => format!("-{}", m),
        Neg::Uni => format!("\u{2212}{}", m),
        Neg::Trail => format!("{}-", m),
    }
}

// ---------------------------------------------------------------- finalize
pub fn count_numtok(s: &str) -> usize {
    let b = s.as_bytes();
    let mut n = 0;
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            n += 1;
            while i < b.len() && (b[i].is_ascii_digit() || ((b[i] == b',' || b[i] == b'.') && i + 1 < b.len() && b[i + 1].is_ascii_digit())) {
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    n
}
pub fn finalize(mut d: Doc) -> Doc {
    let nt = count_numtok(&d.text);
    d.n_distract = nt.saturating_sub(d.facts.len());
    d
}

// ---------------------------------------------------------------- FINANCIAL
pub struct Item {
    pub id: &'static str,
    pub dev: &'static [&'static str],
    pub un: &'static [&'static str],
}
const fn it(id: &'static str, dev: &'static [&'static str], un: &'static [&'static str]) -> Item {
    Item { id, dev, un }
}
pub const INC: &[Item] = &[
    it("rev", &["Revenue", "Net sales", "Total revenues"], &["Turnover", "Net revenues", "Sales"]),
    it("cogs", &["Cost of revenue", "Cost of sales"], &["Direct costs", "Cost of goods sold"]),
    it("gp", &["Gross profit", "Gross margin"], &["Gross income", "Gross result"]),
    it("opex", &["Operating expenses", "Total operating expenses"], &["Operating costs", "SG&A and R&D"]),
    it("opinc", &["Operating income", "Income from operations"], &["Operating profit", "EBIT"]),
    it("int", &["Interest expense", "Interest and other expense"], &["Finance costs", "Net interest cost"]),
    it("pbt", &["Income before income taxes", "Income before tax"], &["Profit before taxation", "Pre-tax earnings"]),
    it("tax", &["Income tax expense", "Provision for income taxes"], &["Taxation", "Tax charge"]),
    it("ni", &["Net income", "Net earnings"], &["Profit for the year", "Net profit"]),
];
pub const BS: &[Item] = &[
    it("cash", &["Cash and cash equivalents", "Cash"], &["Cash and bank balances", "Liquid funds"]),
    it("rec", &["Accounts receivable", "Trade receivables"], &["Debtors", "Receivables, net"]),
    it("inv", &["Inventories", "Inventory"], &["Stock", "Merchandise"]),
    it("tca", &["Total current assets"], &["Current assets, total"]),
    it("ppe", &["Property, plant and equipment", "PP&E, net"], &["Fixed assets", "Tangible assets"]),
    it("ta", &["Total assets"], &["Assets, total", "Sum of assets"]),
    it("ap", &["Accounts payable", "Trade payables"], &["Creditors", "Payables to suppliers"]),
    it("std", &["Short-term debt", "Current portion of debt"], &["Short-term borrowings", "Current loans"]),
    it("tcl", &["Total current liabilities"], &["Current liabilities, total"]),
    it("ltd", &["Long-term debt"], &["Non-current borrowings", "Term loans"]),
    it("tl", &["Total liabilities"], &["Liabilities, total"]),
    it("eq", &["Total stockholders' equity", "Total equity"], &["Shareholders' funds", "Net worth"]),
    it("tle", &["Total liabilities and equity"], &["Total equity and liabilities", "Liabilities and equity, total"]),
];
pub const CF: &[Item] = &[
    it("cfo", &["Net cash provided by operating activities", "Cash from operations"], &["Operating cash flow", "Net cash from operating activities"]),
    it("cfi", &["Net cash used in investing activities"], &["Investing cash flow", "Cash used for investments"]),
    it("cff", &["Net cash provided by (used in) financing activities"], &["Financing cash flow", "Cash from financing"]),
    it("ncc", &["Net increase (decrease) in cash", "Net change in cash"], &["Movement in cash", "Cash position change"]),
];
/// (canonical id, dev label): the extractor lexicon is built ONLY from this (dev bank)
pub fn dev_labels() -> Vec<(&'static str, &'static str)> {
    let mut v = Vec::new();
    for tab in [INC, BS, CF] {
        for i in tab {
            for l in i.dev {
                v.push((i.id, *l));
            }
        }
    }
    v
}

#[derive(Clone, Debug, Default)]
pub struct StmtVals {
    pub m: std::collections::BTreeMap<&'static str, i64>,
}
pub fn mk_stmt(r: &mut R, rev: i64) -> StmtVals {
    let mut m = std::collections::BTreeMap::new();
    let cogs = (rev as f64 * (0.42 + 0.33 * r.f())) as i64;
    let gp = rev - cogs;
    let opex = (gp as f64 * (0.30 + 0.40 * r.f())) as i64;
    let opinc = gp - opex;
    let int = (opinc as f64 * (0.02 + 0.15 * r.f())) as i64;
    let pbt = opinc - int;
    let tax = (pbt as f64 * (0.15 + 0.15 * r.f())) as i64;
    let ni = pbt - tax;
    for (k, v) in [("rev", rev), ("cogs", cogs), ("gp", gp), ("opex", opex), ("opinc", opinc), ("int", int), ("pbt", pbt), ("tax", tax), ("ni", ni)] {
        m.insert(k, v);
    }
    let cash = (rev as f64 * (0.05 + 0.25 * r.f())) as i64;
    let rec = (rev as f64 * (0.08 + 0.15 * r.f())) as i64;
    let inv = (rev as f64 * (0.04 + 0.12 * r.f())) as i64;
    let tca = cash + rec + inv;
    let ppe = (rev as f64 * (0.2 + 0.5 * r.f())) as i64;
    let ta = tca + ppe;
    let ap = (ta as f64 * (0.06 + 0.1 * r.f())) as i64;
    let std = (ta as f64 * (0.03 + 0.08 * r.f())) as i64;
    let tcl = ap + std;
    let ltd = (ta as f64 * (0.1 + 0.25 * r.f())) as i64;
    let tl = tcl + ltd;
    let eq = ta - tl;
    for (k, v) in [("cash", cash), ("rec", rec), ("inv", inv), ("tca", tca), ("ppe", ppe), ("ta", ta), ("ap", ap), ("std", std), ("tcl", tcl), ("ltd", ltd), ("tl", tl), ("eq", eq), ("tle", tl + eq)] {
        m.insert(k, v);
    }
    let cfo = (ni as f64 * (0.8 + 0.7 * r.f())) as i64;
    let cfi = -((rev as f64 * (0.02 + 0.08 * r.f())) as i64);
    let cff = ((rev as f64 * (r.f() - 0.6) * 0.1) as i64).clamp(-(rev / 5), rev / 5);
    for (k, v) in [("cfo", cfo), ("cfi", cfi), ("cff", cff), ("ncc", cfo + cfi + cff)] {
        m.insert(k, v);
    }
    StmtVals { m }
}

fn scale_phrase(r: &mut R, bank: Bank, scale: i64, cur: &str) -> String {
    let sym = match cur {
        "USD" => "$",
        "EUR" => "\u{20ac}",
        "GBP" => "\u{a3}",
        _ => "",
    };
    let c = cur;
    let dev3 = ["in thousands", "in thousands of CUR", "(CUR thousands)", "SYM in thousands", "in thousands, except per share data"];
    let dev6 = ["in millions", "in millions of CUR", "(CUR millions)", "SYM in millions", "in millions, except per share data"];
    let un3 = ["CUR '000", "000s omitted", "expressed in thousand CUR", "CUR in 000s"];
    let un6 = ["CUR mn", "amounts in million CUR", "CUR m", "in CUR Mio"];
    let t = match (bank, scale) {
        (Bank::Dev, 1_000) => *r.pick(&dev3),
        (Bank::Dev, 1_000_000) => *r.pick(&dev6),
        (Bank::Unseen, 1_000) => {
            if r.chance(0.5) { *r.pick(&dev3) } else { *r.pick(&un3) }
        }
        (Bank::Unseen, 1_000_000) => {
            if r.chance(0.5) { *r.pick(&dev6) } else { *r.pick(&un6) }
        }
        _ => "",
    };
    let mut t = t.replace("CUR", c).replace("SYM", if sym.is_empty() { c } else { sym });
    if t.is_empty() {
        t = format!("in {}", c);
    }
    t
}

fn period_header(r: &mut R, bank: Bank, year: i64, pos: usize, nocol_year: bool) -> String {
    if nocol_year {
        return if pos == 0 { "Current year".into() } else { "Prior year".into() };
    }
    match bank {
        Bank::Dev => match r.u(3) {
            0 => format!("FY{}", year),
            1 => format!("{}", year),
            _ => format!("Fiscal {}", year),
        },
        Bank::Unseen => match r.u(4) {
            0 => format!("Year ended Dec 31, {}", year),
            1 => format!("12M to 31/12/{}", year),
            2 => format!("FY {}A", year),
            _ => format!("FY{}", year),
        },
    }
}

pub fn gen_fin(seed: u64, idx: u64, bank: Bank, layout: u8) -> Doc {
    let bk = if bank == Bank::Dev { 0 } else { 1 };
    let mut r = R::new(seed, 11, idx, bk);
    let companies = ["Northwind Holdings", "Acme Industrial", "Bluepeak Systems", "Corvid Logistics", "Delta Foods", "Evergreen Power", "Fjord Marine", "Granite Retail", "Helix Biotech", "Ironbridge Capital"];
    let comp = *r.pick(&companies);
    let cur: &str = if bank == Bank::Unseen && r.chance(0.12) { *r.pick(&["CHF", "SEK"]) } else { *r.pick(&["USD", "USD", "USD", "EUR", "GBP"]) };
    let scale: i64 = *r.pick(&[1_000, 1_000, 1_000_000, 1_000_000, 1]);
    let sym = match cur {
        "USD" => "$",
        "EUR" => "\u{20ac}",
        "GBP" => "\u{a3}",
        _ => "",
    };
    let restated = layout >= 3 && r.chance(0.5);
    let dup_cols = layout >= 4 && restated && r.chance(0.6);
    let footer_scale = layout >= 4 && scale != 1 && r.chance(0.6);
    let ncol: usize = if layout >= 2 && r.chance(0.5) { 3 } else { 2 };
    let ascending = layout >= 4 && r.chance(0.3);
    let nocol_year = bank == Bank::Unseen && ncol == 2 && r.chance(0.2);
    let (sep, neg_style) = match bank {
        Bank::Dev => (Sep::Us, if r.chance(0.5) { Neg::Paren } else { Neg::Minus }),
        Bank::Unseen => (*r.pick(&[Sep::Us, Sep::Us, Sep::Us, Sep::Us, Sep::Us, Sep::Us, Sep::Us, Sep::Eu, Sep::Eu, Sep::Space]), *r.pick(&[Neg::Paren, Neg::Paren, Neg::Paren, Neg::Paren, Neg::Minus, Neg::Minus, Neg::Uni, Neg::Trail])),
    };
    let costs_neg = bank == Bank::Unseen && r.chance(0.25);
    let style = match layout {
        0 => TStyle::Pipe,
        1 => TStyle::Aligned(false),
        2 => {
            if r.chance(0.5) { TStyle::Csv } else { TStyle::Aligned(true) }
        }
        3 => TStyle::Pipe,
        _ => TStyle::Aligned(r.chance(0.5)),
    };
    let base: i64 = if scale == 1 { r.range(200_000, 90_000_000) } else { r.range(2_000, 900_000) };
    let y0 = 2023;
    let mut vals: Vec<StmtVals> = Vec::new();
    let mut rev = base;
    for _ in 0..ncol {
        vals.push(mk_stmt(&mut r, rev));
        rev = (rev as f64 / (0.85 + 0.4 * r.f())) as i64;
    }
    let f_ = 0.93 + 0.04 * r.f();
    let orig_prior = if restated { Some(mk_stmt(&mut r, (vals[1].m["rev"] as f64 * f_) as i64)) } else { None };
    let mut b = B::new();
    let mut facts: Vec<Fact> = Vec::new();
    b.line(&format!("{} - Annual Report Extract", comp));
    if bank == Bank::Dev || r.chance(0.5) {
        b.line(&format!("Fiscal year ended December 31, {}. Reporting currency: {}.", y0, cur));
    } else if r.chance(0.5) {
        b.line(&format!("Fiscal year ended December 31, {}. Figures in {}.", y0, cur));
    } else {
        b.line(&format!("Fiscal year ended December 31, {}.", y0));
    }
    b.nl();
    let sp = scale_phrase(&mut r, bank, scale, cur);
    let want_bs = (layout >= 1 && r.chance(0.7)) || (layout == 0 && r.chance(0.4));
    let want_cf = layout >= 2 && r.chance(0.5);
    let mut stmts: Vec<(&str, &[Item])> = vec![("Consolidated Statements of Income", INC)];
    if want_bs {
        stmts.push(("Consolidated Balance Sheets", BS));
    }
    if want_cf {
        stmts.push(("Consolidated Statements of Cash Flows", CF));
    }
    let mut footnote_i = 0;
    for (si, (title, items)) in stmts.iter().enumerate() {
        if si == 0 {
            b.line(&format!("Management discussion: the group employed {} people across {} countries and opened {} new sites during the year.", fmt_mag(r.range(900, 48_000), Sep::Us), r.range(3, 41), r.range(2, 60)));
            let g = (vals[0].m["rev"] as f64 / vals[1].m["rev"] as f64 - 1.0) * 100.0;
            if r.chance(0.6) {
                b.line(&format!("Net revenue increased {:.1}% compared with the prior year, see Note {}.", g, r.range(2, 19)));
            }
            b.nl();
        }
        if footer_scale {
            b.line(title);
        } else {
            b.line(&format!("{} ({})", title, sp));
        }
        b.nl();
        let years: Vec<i64> = (0..ncol).map(|j| y0 - j as i64).collect();
        let mut cols: Vec<(usize, String)> = Vec::new();
        for j in 0..ncol {
            let mut h = period_header(&mut r, bank, years[j], j, nocol_year);
            if restated && j == 1 {
                h = format!("{} (restated)", h);
            }
            cols.push((j, h));
        }
        if dup_cols {
            let h = period_header(&mut r, bank, years[1], 1, false);
            cols.insert(2, (99, format!("{} (as reported)", h)));
        }
        if ascending {
            cols.reverse();
        }
        let mut header = vec![if r.chance(0.5) { "Line item".to_string() } else { "Item".to_string() }];
        header.extend(cols.iter().map(|c| c.1.clone()));
        let mut rows: Vec<Row> = Vec::new();
        let mut fact_slots: Vec<Vec<Option<(String, f64)>>> = Vec::new();
        for item in items.iter() {
            let label0 = if bank == Bank::Unseen && r.chance(0.35) { *r.pick(item.un) } else { *r.pick(item.dev) };
            let mut label = label0.to_string();
            if layout >= 3 && footnote_i < 2 && r.chance(0.12) {
                label = format!("{} ({})", label, (b'a' + footnote_i as u8) as char);
                footnote_i += 1;
            }
            let is_cost = ["cogs", "opex", "int", "tax"].contains(&item.id);
            let mut cells = Vec::new();
            let mut slots = Vec::new();
            for (ci, c) in cols.iter().enumerate() {
                let v0 = if c.0 == 99 { orig_prior.as_ref().unwrap().m[item.id] } else { vals[c.0].m[item.id] };
                let v = if costs_neg && is_cost { -v0 } else { v0 };
                let mut t = fmt_signed(v, sep, neg_style);
                if (1..4).contains(&layout) && ((si == 0 && ci == 0) || ["gp", "ni", "ta", "tle"].contains(&item.id)) && !(neg_style == Neg::Paren && v < 0) && !sym.is_empty() && matches!(style, TStyle::Aligned(_)) {
                    t = format!("{} {}", sym, t);
                }
                if layout >= 3 && item.id == "ni" && ci == 1 && r.chance(0.4) {
                    t = format!("{} (1)", t);
                }
                cells.push(t);
                let key = format!("{}|FY{}", item.id, if c.0 == 99 { y0 - 1 } else { y0 - c.0 as i64 });
                if c.0 == 99 {
                    slots.push(None);
                } else {
                    slots.push(Some((key, v as f64 * scale as f64)));
                }
            }
            rows.push(Row { label, cells });
            fact_slots.push(slots);
        }
        let spans = render_table(&mut b, style, &header, &rows);
        for (ri, sps) in spans.iter().enumerate() {
            for (ci, s) in sps.iter().enumerate() {
                if let Some((k, v)) = &fact_slots[ri][ci] {
                    facts.push(Fact::num(k, FKind::Line, *v, cur, s.0, s.1));
                }
            }
        }
        b.nl();
        if layout >= 3 && footnote_i > 0 && si == 0 {
            b.line(&format!("(a) Includes restructuring charges of {} in the current year; see Note {}.", fmt_mag(r.range(40, 900), Sep::Us), r.range(3, 30)));
            b.nl();
        }
        if restated && si == 0 {
            let o = orig_prior.as_ref().unwrap().m["rev"];
            b.line(&format!("Prior-year figures were restated; revenue for FY{} was previously reported as {}.", y0 - 1, fmt_mag(o, Sep::Us)));
            b.nl();
        }
    }
    if footer_scale {
        b.line(&format!("All amounts in {} unless otherwise stated.", sp.replace("in ", "").replace('(', "").replace(')', "")));
    }
    let key = |id: &str, p: i64| format!("{}|FY{}", id, y0 - p);
    let mut qs: Vec<Question> = Vec::new();
    let mut add = |op: QOp, keys: Vec<String>, text: &str, facts: &Vec<Fact>| {
        let vs: Option<Vec<f64>> = keys.iter().map(|k| facts.iter().find(|f| &f.key == k).map(|f| f.value)).collect();
        if let Some(vs) = vs {
            if let Some(t) = qeval(op, &vs) {
                qs.push(Question { op, keys, truth: t, text: text.to_string() });
            }
        }
    };
    add(QOp::Growth, vec![key("rev", 0), key("rev", 1)], "What was revenue growth FY over FY?", &facts);
    add(QOp::Growth, vec![key("ni", 0), key("ni", 1)], "What was net income growth?", &facts);
    add(QOp::Lookup, vec![key("rev", 0)], "What was revenue in the latest year (in currency units)?", &facts);
    add(QOp::Diff, vec![key("rev", 0), key("rev", 1)], "By how much did revenue change (absolute)?", &facts);
    add(QOp::Ratio, vec![key("gp", 0), key("rev", 0)], "What is the gross margin?", &facts);
    add(QOp::Ratio, vec![key("ni", 0), key("rev", 0)], "What is the net margin?", &facts);
    add(QOp::Ratio, vec![key("opinc", 0), key("rev", 0)], "What is the operating margin?", &facts);
    if want_bs {
        add(QOp::Ratio, vec![key("tca", 0), key("tcl", 0)], "What is the current ratio?", &facts);
        add(QOp::Ratio, vec![key("tl", 0), key("eq", 0)], "What is the liabilities-to-equity ratio?", &facts);
    }
    if want_cf {
        add(QOp::Diff, vec![key("cfo", 0), key("ni", 0)], "How much does operating cash flow exceed net income?", &facts);
    }
    let scale_conf = scale != 1 || neg_style == Neg::Paren || restated || footer_scale || dup_cols;
    finalize(Doc { world: World::Fin, bank, layout, adv: 0, text: b.s, facts, claims: vec![], qs, n_distract: 0, scale_conf, structured: true, n_claim_sents: 0 })
}

/// adversarial financial layouts (controls; failure is the expected result): 1 = two statements side by side (page-style
/// two-column text), 2 = wrapped labels (label on one line, numbers on the next), 3 = numbers split across lines
pub fn gen_fin_adv(seed: u64, idx: u64, kind: u8) -> Doc {
    let mut r = R::new(seed, 13, idx, kind as u64);
    let cur = "USD";
    let scale = 1_000i64;
    let rev = r.range(5_000, 800_000);
    let a = mk_stmt(&mut r, rev);
    let c = mk_stmt(&mut r, (rev as f64 * 0.9) as i64);
    let mut b = B::new();
    let mut facts = Vec::new();
    b.line("Summary Financials (in thousands of USD)");
    b.line("                 FY2023            FY2022");
    let ids = ["rev", "cogs", "gp", "opex", "opinc", "ni"];
    let long = ["Net cash provided by operating", "Income from continuing operations before"];
    for (i, id) in ids.iter().enumerate() {
        let lab = INC.iter().find(|x| x.id == *id).unwrap().dev[0];
        match kind {
            1 => {
                let bsid = ["cash", "rec", "inv", "tca", "ppe", "ta"][i];
                let bl = BS.iter().find(|x| x.id == bsid).unwrap().dev[0];
                b.push(&format!("{:<22}", lab));
                let (lo, hi) = b.span(&fmt_mag(a.m[id], Sep::Us));
                facts.push(Fact::num(&format!("{}|FY2023", id), FKind::Line, a.m[id] as f64 * scale as f64, cur, lo, hi));
                b.push(&format!("        {:<30}", bl));
                let (lo, hi) = b.span(&fmt_mag(a.m[bsid], Sep::Us));
                facts.push(Fact::num(&format!("{}|FY2023", bsid), FKind::Line, a.m[bsid] as f64 * scale as f64, cur, lo, hi));
                b.nl();
            }
            2 => {
                let l2 = long[i % 2];
                b.line(l2);
                b.push(&format!("  {:<28}", lab.split(' ').last().unwrap_or("")));
                let (lo, hi) = b.span(&fmt_mag(a.m[id], Sep::Us));
                facts.push(Fact::num(&format!("{}|FY2023", id), FKind::Line, a.m[id] as f64 * scale as f64, cur, lo, hi));
                b.push("   ");
                let (lo, hi) = b.span(&fmt_mag(c.m[id], Sep::Us));
                facts.push(Fact::num(&format!("{}|FY2022", id), FKind::Line, c.m[id] as f64 * scale as f64, cur, lo, hi));
                b.nl();
            }
            _ => {
                b.push(&format!("{:<28}", lab));
                let s = fmt_mag(a.m[id], Sep::Us);
                let lo = b.pos();
                if let Some((h, t)) = s.split_once(',') {
                    b.push(h);
                    b.push(",\n                            ");
                    b.push(t);
                } else {
                    b.push(&s);
                }
                let hi = b.pos();
                facts.push(Fact::num(&format!("{}|FY2023", id), FKind::Line, a.m[id] as f64 * scale as f64, cur, lo, hi));
                b.nl();
            }
        }
    }
    let k = |id: &str| format!("{}|FY2023", id);
    let mut qs = Vec::new();
    for (op, keys, t) in [(QOp::Ratio, vec![k("gp"), k("rev")], "gross margin"), (QOp::Ratio, vec![k("ni"), k("rev")], "net margin")] {
        let vs: Option<Vec<f64>> = keys.iter().map(|kk| facts.iter().find(|f| &f.key == kk).map(|f| f.value)).collect();
        if let Some(vs) = vs {
            qs.push(Question { op, keys, truth: qeval(op, &vs).unwrap(), text: t.into() });
        }
    }
    finalize(Doc { world: World::Fin, bank: Bank::Dev, layout: 5, adv: kind, text: b.s, facts, claims: vec![], qs, n_distract: 0, scale_conf: true, structured: true, n_claim_sents: 0 })
}
