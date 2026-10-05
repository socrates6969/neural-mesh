//! g21 natural-language task renderer (the WORLD side). Renders the canonical spec of a g13 `TaskRec` into
//! templated-but-varied task text with: phrasing levels L0..L4, number words, thousands separators, unit names and
//! spellings, distractor numbers, typos, multi-step procedural / nested-phrase word problems, hard negatives on
//! prose tasks, and an "unseen" template bank the extractor was NOT developed against.
//!   L0 near-canonical formatting   L1 varied phrasing   L2 number words + distractor numbers
//!   L3 typos + L2 features         L4 degraded (an operand deleted / a conflicting number inserted)
use neural_mesh::router::Rng;
use neural_mesh::tools::{parse_expr, Expr, Op};

use super::g13::{Kind, TaskRec};

#[derive(Clone, Debug)]
pub struct NlTask {
    pub text: String,
    pub level: u8,
    pub unseen: bool,
}

#[derive(Clone, Debug)]
pub struct NlCfg {
    pub level_p: [f64; 5],
    pub unseen_p: f64,
    /// share of prose (qual) tasks that embed an arithmetic-looking fragment (hard negatives)
    pub hardneg_p: f64,
}
impl NlCfg {
    pub fn world_a() -> Self {
        NlCfg { level_p: [0.30, 0.30, 0.22, 0.13, 0.05], unseen_p: 0.20, hardneg_p: 0.15 }
    }
    pub fn world_b() -> Self {
        NlCfg { level_p: [0.12, 0.22, 0.26, 0.30, 0.10], unseen_p: 0.35, hardneg_p: 0.25 }
    }
}

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

pub fn words(n: u64, and: bool) -> String {
    if n < 20 {
        ONES[n as usize].to_string()
    } else if n < 100 {
        let t = TENS[(n / 10) as usize];
        if n % 10 == 0 { t.to_string() } else { format!("{}-{}", t, ONES[(n % 10) as usize]) }
    } else if n < 1000 {
        let h = format!("{} hundred", ONES[(n / 100) as usize]);
        if n % 100 == 0 { h } else { format!("{}{}{}", h, if and { " and " } else { " " }, words(n % 100, and)) }
    } else if n < 1_000_000 {
        let h = format!("{} thousand", words(n / 1000, and));
        if n % 1000 == 0 { h } else { format!("{} {}", h, words(n % 1000, and)) }
    } else {
        n.to_string()
    }
}

/// render a decimal string ("12", "12.5") according to the level
pub fn num_text(s: &str, wordy: bool, commas: bool, and: bool) -> String {
    let (ip, fp) = match s.split_once('.') {
        Some((a, b)) => (a, Some(b)),
        None => (s, None),
    };
    let iv: Option<u64> = ip.parse().ok();
    if wordy {
        if let Some(v) = iv {
            if v < 1_000_000 {
                let mut out = words(v, and);
                if let Some(f) = fp {
                    out += " point";
                    for c in f.chars() {
                        out += " ";
                        out += ONES[c.to_digit(10).unwrap() as usize];
                    }
                }
                return out;
            }
        }
    }
    let mut ips = ip.to_string();
    if commas && ip.len() >= 4 {
        let b: Vec<char> = ip.chars().collect();
        let mut o = String::new();
        for (i, c) in b.iter().enumerate() {
            if i > 0 && (b.len() - i) % 3 == 0 {
                o.push(',');
            }
            o.push(*c);
        }
        ips = o;
    }
    match fp {
        Some(f) => format!("{ips}.{f}"),
        None => ips,
    }
}

fn pick<'a>(r: &mut Rng, xs: &[&'a str]) -> &'a str {
    xs[r.below(xs.len())]
}

fn cap(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn typo_word(w: &str, r: &mut Rng) -> String {
    let c: Vec<char> = w.chars().collect();
    if c.len() < 4 {
        return w.to_string();
    }
    let i = 1 + r.below(c.len() - 2);
    let mut o = c.clone();
    match r.below(3) {
        0 => {
            o.remove(i);
        }
        1 => {
            o.swap(i, i + 1);
        }
        _ => {
            o.insert(i, c[i]);
        }
    }
    o.into_iter().collect()
}

/// typos on k alphabetic words of length >= 4 (never on single-letter variables, digits, or key tokens like k17/t2)
fn add_typos(text: &str, k: usize, r: &mut Rng) -> String {
    let mut toks: Vec<String> = text.split(' ').map(|s| s.to_string()).collect();
    let cand: Vec<usize> = (0..toks.len())
        .filter(|i| {
            let core: String = toks[*i].chars().filter(|c| c.is_alphabetic()).collect();
            core.len() >= 4 && toks[*i].chars().all(|c| c.is_alphabetic() || ".,?:;!".contains(c))
        })
        .collect();
    if cand.is_empty() {
        return text.to_string();
    }
    for _ in 0..k {
        let i = cand[r.below(cand.len())];
        let t = toks[i].clone();
        let (core, tail): (String, String) = (t.chars().take_while(|c| c.is_alphabetic()).collect(), t.chars().skip_while(|c| c.is_alphabetic()).collect());
        toks[i] = typo_word(&core, r) + &tail;
    }
    toks.join(" ")
}

fn distract(r: &mut Rng, unseen: bool) -> (String, bool) {
    // (text, is_prefix)
    let q = 1 + r.below(30);
    let n = 10 + r.below(40);
    let m = 2 + r.below(9);
    let a = 1 + r.below(9);
    let b = 10 + r.below(19);
    if !unseen {
        match r.below(8) {
            0 => (format!("Question {q} of {n}: "), true),
            1 => (format!("Ticket #{} - ", 1000 + r.below(9000)), true),
            2 => (format!("[Problem {q}, worth {m} points] "), true),
            3 => (format!("(asked on 2021-0{a}-{b}) "), true),
            4 => (format!(" Please answer within {} seconds.", 10 + r.below(50)), false),
            5 => (format!(" (Section {a}.{m}.)"), false),
            6 => (format!(" Thanks from the team of {m}!"), false),
            _ => (format!("Round {q}: "), true),
        }
    } else {
        match r.below(6) {
            0 => (format!("Ref. 2024/{n}: "), true),
            1 => (format!("Rev {a}.{m} - "), true),
            2 => (format!(" This takes about {} minutes to grade.", 2 + r.below(20)), false),
            3 => (format!(" (class of {n} students, group {m})"), false),
            4 => (format!("No. {n}/{m}. "), true),
            _ => (format!(" Score: {a} out of {m}."), false),
        }
    }
}

fn with_distractor(body: String, r: &mut Rng, unseen: bool) -> String {
    let (d, pre) = distract(r, unseen);
    if pre { format!("{d}{body}") } else { format!("{body}{d}") }
}

fn tag(kw: &str, r: &mut Rng, unseen: bool) -> String {
    if !unseen && r.f64() < 0.6 { format!("[{kw}] ") } else { String::new() }
}

// ---------------------------------------------------------------------------------------------
// arithmetic

fn rat_text(e: &Expr) -> Option<String> {
    if let Expr::Num(r) = e {
        let (n, d) = (r.numer(), r.denom());
        if d == 1 {
            return Some(n.to_string());
        }
        if 10 % d == 0 {
            let t = n * (10 / d);
            return Some(format!("{}.{}", t / 10, t % 10));
        }
    }
    None
}
fn is_atom(e: &Expr) -> bool {
    match e {
        Expr::Num(_) => true,
        Expr::Bin(Op::Pow, a, _) => matches!(**a, Expr::Num(_)),
        _ => false,
    }
}

#[derive(Clone, Copy)]
struct ArithStyle {
    wordy: bool,
    commas: bool,
    and: bool,
    unseen: bool,
    pi: usize, // phrase variant
}

fn leaf(e: &Expr, st: &ArithStyle, r: &mut Rng) -> String {
    let s = rat_text(e).unwrap_or_else(|| "?".into());
    let wordy = st.wordy && !s.contains('.') && r.f64() < 0.6 || (st.wordy && s.contains('.') && r.f64() < 0.3);
    num_text(&s, wordy, st.commas && r.f64() < 0.5, st.and && r.f64() < 0.5)
}

fn atom_text(e: &Expr, st: &ArithStyle, r: &mut Rng, sq_word: bool) -> String {
    match e {
        Expr::Num(_) => leaf(e, st, r),
        Expr::Bin(Op::Pow, a, _) => {
            let b = leaf(a, st, r);
            if sq_word {
                if st.unseen { format!("{b} to the second power") } else { format!("{b} squared") }
            } else {
                format!("{b}^2")
            }
        }
        _ => "?".into(),
    }
}

fn render_infix(e: &Expr, st: &ArithStyle, r: &mut Rng, words_ops: bool, top: bool) -> String {
    if is_atom(e) {
        return atom_text(e, st, r, words_ops);
    }
    match e {
        Expr::Bin(op, a, b) => {
            let l = render_infix(a, st, r, words_ops, false);
            let rr = render_infix(b, st, r, words_ops, false);
            let sym = if words_ops {
                match op {
                    Op::Add => if st.unseen && r.f64() < 0.4 { "increased by" } else { "plus" },
                    Op::Sub => if st.unseen && r.f64() < 0.4 { "decreased by" } else { "minus" },
                    Op::Mul => if r.f64() < 0.3 { "multiplied by" } else if st.unseen && r.f64() < 0.3 { "x" } else { "times" },
                    Op::Div => if st.unseen && r.f64() < 0.4 { "over" } else { "divided by" },
                    _ => "^",
                }
            } else {
                match op {
                    Op::Add => "+",
                    Op::Sub => if st.unseen && r.f64() < 0.5 { "\u{2212}" } else { "-" },
                    Op::Mul => if st.unseen && r.f64() < 0.5 { "\u{d7}" } else { "*" },
                    Op::Div => if st.unseen && r.f64() < 0.5 { "\u{f7}" } else { "/" },
                    _ => "^",
                }
            };
            let wrap = |s: String, node: &Expr| if is_atom(node) { s } else { format!("({s})") };
            let l = wrap(l, a);
            let rr = wrap(rr, b);
            let _ = top;
            if words_ops || st.wordy || r.f64() < 0.6 { format!("{l} {sym} {rr}") } else { format!("{l}{sym}{rr}") }
        }
        _ => "?".into(),
    }
}

fn render_tree(e: &Expr, st: &ArithStyle, r: &mut Rng) -> String {
    if is_atom(e) {
        return atom_text(e, st, r, false).replace("^2", " squared");
    }
    match e {
        Expr::Bin(op, a, b) => {
            let l = render_tree(a, st, r);
            let rr = render_tree(b, st, r);
            if st.unseen {
                match op {
                    Op::Add => format!("the total of {l} and {rr}"),
                    Op::Sub => format!("the gap from {rr} up to {l}"),
                    Op::Mul => format!("the result of multiplying {l} by {rr}"),
                    Op::Div => format!("the result of dividing {l} by {rr}"),
                    _ => "?".into(),
                }
            } else {
                match op {
                    Op::Add => format!("the sum of {l} and {rr}"),
                    Op::Sub => format!("the difference of {l} and {rr}"),
                    Op::Mul => format!("the product of {l} and {rr}"),
                    Op::Div => format!("the quotient of {l} and {rr}"),
                    _ => "?".into(),
                }
            }
        }
        _ => "?".into(),
    }
}

/// left-deep chain with atom right operands -> procedural steps
fn render_proc(e: &Expr, st: &ArithStyle, r: &mut Rng) -> Option<String> {
    let mut steps: Vec<(Op, &Expr)> = Vec::new();
    let mut cur = e;
    loop {
        match cur {
            Expr::Bin(op, a, b) if !is_atom(cur) => {
                if !is_atom(b) {
                    return None;
                }
                steps.push((*op, b));
                cur = a;
            }
            _ => break,
        }
    }
    if !is_atom(cur) || steps.is_empty() {
        return None;
    }
    steps.reverse();
    let mut out = vec![format!("Start with {}.", atom_text(cur, st, r, true))];
    for (op, b) in steps {
        let v = atom_text(b, st, r, true);
        out.push(match (op, st.unseen) {
            (Op::Add, false) => format!("Add {v}."),
            (Op::Add, true) => format!("Increase it by {v}."),
            (Op::Sub, false) => format!("Subtract {v}."),
            (Op::Sub, true) => format!("Decrease it by {v}."),
            (Op::Mul, false) => format!("Multiply the result by {v}."),
            (Op::Mul, true) => format!("Multiply that by {v}."),
            (Op::Div, false) => format!("Divide the result by {v}."),
            (Op::Div, true) => format!("Split that by {v}."),
            _ => "?".into(),
        });
    }
    let q = if st.unseen { "What do you end up with?" } else { "What is the final result?" };
    out.push(q.into());
    Some(out.join(" "))
}

fn render_arith(t: &TaskRec, level: u8, unseen: bool, r: &mut Rng) -> String {
    let e = parse_expr(&t.spec).expect("canonical arith");
    let st = ArithStyle { wordy: level >= 2, commas: level >= 2, and: unseen, unseen, pi: r.below(3) };
    let heads: [&str; 6] = ["Compute", "Evaluate", "Calculate", "What is", "Work out", "Find the value of"];
    let uheads: [&str; 4] = ["Determine", "Figure out", "Give me", "Tell me"];
    let head = if unseen { pick(r, &uheads) } else { pick(r, &heads) };
    let tail = pick(r, &[".", "?", ""]);
    let mode = if level == 0 {
        0
    } else {
        // 0 sym, 1 words, 2 tree, 3 proc
        let u = r.f64();
        if u < 0.25 { 0 } else if u < 0.55 { 1 } else if u < 0.8 { 2 } else { 3 }
    };
    let _ = st.pi;
    let mut body = match mode {
        0 => format!("{head} {}{tail}", render_infix(&e, &st, r, false, true)),
        1 => format!("{head} {}{tail}", render_infix(&e, &st, r, true, true)),
        2 => format!("{head} {}{tail}", render_tree(&e, &st, r)),
        _ => match render_proc(&e, &st, r) {
            Some(s) => s,
            None => format!("{head} {}{tail}", render_tree(&e, &st, r)),
        },
    };
    if level >= 2 && r.f64() < 0.7 {
        body = with_distractor(body, r, unseen);
    }
    body
}

// ---------------------------------------------------------------------------------------------
// units

const UNIT_NAMES: &[(&str, &[&str], &[&str])] = &[
    // symbol, dev names (singular, plural, ...), unseen extra names
    ("m", &["metre", "metres", "meter", "meters"], &["metric metres"]),
    ("km", &["kilometre", "kilometres", "kilometer", "kilometers"], &["kms"]),
    ("cm", &["centimetre", "centimetres", "centimeter", "centimeters"], &[]),
    ("mm", &["millimetre", "millimetres", "millimeter", "millimeters"], &[]),
    ("ft", &["foot", "feet"], &["ft."]),
    ("in", &["inch", "inches"], &[]),
    ("yd", &["yard", "yards"], &[]),
    ("kg", &["kilogram", "kilograms", "kilo", "kilos"], &[]),
    ("g", &["gram", "grams"], &["gm"]),
    ("mg", &["milligram", "milligrams"], &[]),
    ("lb", &["pound", "pounds"], &["lbs"]),
    ("t", &["tonne", "tonnes"], &["metric tons"]),
    ("s", &["second", "seconds"], &["secs"]),
    ("ms", &["millisecond", "milliseconds"], &[]),
    ("min", &["minute", "minutes"], &["mins"]),
    ("h", &["hour", "hours"], &["hrs"]),
    ("day", &["day", "days"], &[]),
    ("week", &["week", "weeks"], &["wks"]),
    ("m/s", &["metres per second", "meters per second"], &["metres/second"]),
    ("km/h", &["kilometres per hour", "kilometers per hour"], &["kph"]),
    ("knot", &["knot", "knots"], &[]),
    ("mile/h", &["miles per hour"], &["mph"]),
    ("L", &["litre", "litres", "liter", "liters"], &[]),
    ("mL", &["millilitre", "millilitres", "milliliter", "milliliters"], &["mls"]),
    ("m^3", &["cubic metres", "cubic meters"], &["m\u{b3}"]),
];

fn unit_text(sym: &str, level: u8, unseen: bool, plural: bool, r: &mut Rng) -> String {
    let ent = UNIT_NAMES.iter().find(|u| u.0 == sym).unwrap();
    let symbolic = level == 0 && r.f64() < 0.75 || (level == 1 && r.f64() < 0.35);
    if symbolic {
        return sym.to_string();
    }
    if unseen && !ent.2.is_empty() && r.f64() < 0.5 {
        return ent.2[r.below(ent.2.len())].to_string();
    }
    let names = ent.1;
    let spell_uk = r.f64() < 0.5;
    // names are ordered: singular-uk, plural-uk, singular-us, plural-us (some have fewer)
    let pairs = (names.len() / 2).max(1);
    let p = if spell_uk { 0 } else { (pairs - 1).min(1) };
    if names.len() == 1 {
        return names[0].to_string();
    }
    let idx = p * 2 + if plural { 1 } else { 0 };
    if idx < names.len() { names[idx].to_string() } else { names[names.len() - 1].to_string() }
}

fn qty_text(v: &str, sym: &str, level: u8, unseen: bool, r: &mut Rng) -> String {
    let wordy = level >= 2 && !v.contains('.') && r.f64() < 0.5;
    let plural = v != "1";
    let nt = num_text(v, wordy, level >= 2 && r.f64() < 0.4, unseen);
    let u = unit_text(sym, level, unseen, plural, r);
    // symbolic units attach without a space sometimes
    if u == sym && !wordy && level == 0 && r.f64() < 0.25 && !sym.contains('/') { format!("{nt}{u}") } else { format!("{nt} {u}") }
}

fn render_units(t: &TaskRec, level: u8, unseen: bool, r: &mut Rng) -> String {
    let (cmd, rest) = t.spec.split_once(' ').unwrap();
    let (lhs, to) = rest.split_once("->").unwrap();
    let to = to.trim();
    let mut body;
    if cmd == "conv" {
        let (v, u) = lhs.trim().split_once(' ').unwrap();
        let q = qty_text(v, u, level, unseen, r);
        let tn = unit_text(to, level, unseen, true, r);
        body = if level == 0 {
            match r.below(3) {
                0 => format!("convert {q} to {tn}"),
                1 => format!("{q} in {tn}?"),
                _ => format!("express {q} in {tn}"),
            }
        } else if unseen {
            match r.below(3) {
                0 => format!("Re-express {q} using {tn}."),
                1 => format!("{} is equivalent to how many {tn}?", cap(&q)),
                _ => format!("Give {q} in terms of {tn}."),
            }
        } else {
            match r.below(4) {
                0 => format!("How many {tn} are in {q}?"),
                1 => format!("Express {q} in {tn}."),
                2 => format!("Convert {q} to {tn}."),
                _ => format!("{} is how many {tn}?", cap(&q)),
            }
        };
    } else {
        let terms: Vec<String> = lhs
            .split('+')
            .map(|p| {
                let (v, u) = p.trim().split_once(' ').unwrap();
                qty_text(v, u, level, unseen, r)
            })
            .collect();
        let tn = unit_text(to, level, unseen, true, r);
        let listed = if terms.len() == 2 { format!("{} and {}", terms[0], terms[1]) } else { format!("{}, {} and {}", terms[0], terms[1], terms[2]) };
        body = if level == 0 {
            format!("sum {} in {tn}", terms.join(" + "))
        } else if unseen {
            match r.below(2) {
                0 => format!("Altogether, {listed} amount to how many {tn}?"),
                _ => format!("Pool {listed}; report the whole in {tn}."),
            }
        } else {
            match r.below(4) {
                0 => format!("Add {listed}; give the total in {tn}."),
                1 => format!("What is {} in {tn}?", terms.join(" + ")),
                2 => format!("A route has legs of {listed}. What is the total in {tn}?"),
                _ => format!("Express the sum of {listed} in {tn}."),
            }
        };
    }
    if level >= 2 && r.f64() < 0.7 {
        body = with_distractor(body, r, unseen);
    }
    body
}

// ---------------------------------------------------------------------------------------------
// csp

fn csp_cons_text(c: &str, level: u8, unseen: bool, r: &mut Rng) -> String {
    // canonical forms: "2*x+y==7", "x*y==12", "x<y", "x<=y", "x+y>=3", "-x+2*y<=0"
    let (rel, pos) = if let Some(p) = c.find("==") { ("==", p) } else if let Some(p) = c.find("<=") { ("<=", p) } else if let Some(p) = c.find(">=") { (">=", p) } else if let Some(p) = c.find('<') { ("<", p) } else { ("?", 0) };
    let l = &c[..pos];
    let rhs = &c[pos + rel.len()..];
    let wordy = level >= 2;
    // left expression: terms split by + / - keeping signs
    let mut terms: Vec<(char, String)> = Vec::new();
    let mut cur = String::new();
    let mut sign = '+';
    for ch in l.chars() {
        if (ch == '+' || ch == '-') && !cur.is_empty() {
            terms.push((sign, cur.clone()));
            cur.clear();
            sign = ch;
        } else if (ch == '+' || ch == '-') && cur.is_empty() {
            sign = ch;
        } else {
            cur.push(ch);
        }
    }
    terms.push((sign, cur));
    let term_txt = |t: &str, r: &mut Rng| -> String {
        if let Some((a, b)) = t.split_once('*') {
            if a.chars().all(|c| c.is_ascii_digit()) {
                let an = num_text(a, wordy && r.f64() < 0.5, false, false);
                if wordy && an.chars().all(|c| c.is_alphabetic() || c == '-') {
                    return format!("{an} times {b}");
                }
                if level >= 3 && r.f64() < 0.5 {
                    return format!("{a}{b}");
                }
                return format!("{an}*{b}");
            }
            if wordy { return format!("{a} times {b}"); }
        }
        t.to_string()
    };
    let mut ls = String::new();
    for (i, (s, t)) in terms.iter().enumerate() {
        let tt = term_txt(t, r);
        if i == 0 {
            ls += &(if *s == '-' { if wordy { format!("minus {tt}") } else { format!("-{tt}") } } else { tt });
        } else {
            let opw = if wordy { if *s == '-' { " minus " } else { " plus " } } else if *s == '-' { " - " } else { " + " };
            ls += opw;
            ls += &tt;
        }
    }
    let rhs_t = if wordy && rhs.chars().all(|c| c.is_ascii_digit()) { num_text(rhs, r.f64() < 0.5, false, false) } else { rhs.to_string() };
    let rel_t: String = if wordy {
        match rel {
            "==" => pick(r, &["equals", "is equal to", "gives"]).into(),
            "<=" => if unseen && r.f64() < 0.5 { "\u{2264}".into() } else { "is at most".into() },
            ">=" => if unseen && r.f64() < 0.5 { "\u{2265}".into() } else { "is at least".into() },
            _ => "is less than".into(),
        }
    } else if level >= 3 && rel == "==" && r.f64() < 0.5 {
        "=".into()
    } else {
        rel.to_string()
    };
    format!("{ls} {rel_t} {rhs_t}")
}

fn render_csp(t: &TaskRec, level: u8, unseen: bool, r: &mut Rng) -> String {
    let parts: Vec<&str> = t.spec.split(';').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    let mut vars: Vec<&str> = Vec::new();
    let mut dom = (0, 0);
    let mut cons: Vec<&str> = Vec::new();
    for p in &parts {
        if let Some((n, rg)) = p.split_once(" in ") {
            vars.push(n.trim());
            let (lo, hi) = rg.split_once("..").unwrap();
            dom = (lo.trim().parse::<i64>().unwrap(), hi.trim().parse::<i64>().unwrap());
        } else {
            cons.push(p);
        }
    }
    let vl = vars.join(", ");
    let ctxt: Vec<String> = cons.iter().map(|c| csp_cons_text(c, level, unseen, r)).collect();
    let mut body = if level == 0 {
        let mut s = String::new();
        for v in &vars {
            s += &format!("{v} in {}..{}; ", dom.0, dom.1);
        }
        s += &cons.join("; ");
        s
    } else {
        let wordy = level >= 2;
        let lo = num_text(&dom.0.to_string(), wordy, false, false);
        let hi = num_text(&dom.1.to_string(), wordy, false, false);
        let dom_t = if unseen {
            format!("each at least {lo} and at most {hi}")
        } else {
            match r.below(3) {
                0 => format!("each between {lo} and {hi}"),
                1 => format!("each from {lo} to {hi}"),
                _ => format!("each in {}..{}", dom.0, dom.1),
            }
        };
        let joined = if ctxt.len() == 1 { ctxt[0].clone() } else { format!("{} and {}", ctxt[..ctxt.len() - 1].join(", "), ctxt[ctxt.len() - 1]) };
        match r.below(3) {
            0 => format!("Find integers {vl}, {dom_t}, such that {joined}."),
            1 => format!("Satisfy the constraints: {joined}, with {vl} {dom_t}."),
            _ => format!("Is there an assignment of {vl} ({dom_t}) with {joined}? Give one."),
        }
    };
    if level >= 2 && r.f64() < 0.7 {
        body = with_distractor(body, r, unseen);
    }
    body
}

// ---------------------------------------------------------------------------------------------
// symbolic identity

fn sym_side_text(s: &str, level: u8, unseen: bool, r: &mut Rng) -> String {
    let wordy = level >= 2;
    let mut out = String::new();
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == '^' {
            // exponent digits
            let mut j = i + 1;
            let mut e = String::new();
            while j < b.len() && b[j].is_ascii_digit() {
                e.push(b[j]);
                j += 1;
            }
            if wordy {
                match e.as_str() {
                    "2" => out += " squared",
                    "3" => out += " cubed",
                    _ => {
                        out += &format!(" to the power of {}", if r.f64() < 0.5 { words(e.parse().unwrap(), false) } else { e.clone() });
                    }
                }
            } else {
                out += &format!("^{e}");
            }
            i = j;
            continue;
        }
        if wordy && (c == '+' || c == '-') && i > 0 {
            out += if c == '+' { " plus " } else { " minus " };
            i += 1;
            continue;
        }
        if wordy && c == '*' {
            out += " times ";
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let mut j = i;
            let mut n = String::new();
            while j < b.len() && b[j].is_ascii_digit() {
                n.push(b[j]);
                j += 1;
            }
            // number glued to nothing special
            if wordy && r.f64() < 0.5 && n.parse::<u64>().is_ok() {
                out += &words(n.parse().unwrap(), unseen);
            } else {
                out += &n;
            }
            i = j;
            continue;
        }
        if !wordy && c == '*' && level >= 1 && r.f64() < 0.5 {
            // implicit multiplication: "2*x" -> "2x", ")*(" -> ")("
            let prev_ok = i > 0 && (b[i - 1].is_ascii_digit() || b[i - 1] == ')');
            let next_ok = i + 1 < b.len() && (b[i + 1] == 'x' || b[i + 1] == '(');
            if prev_ok && next_ok {
                i += 1;
                continue;
            }
        }
        if !wordy && level >= 1 && (c == '+' || c == '-') {
            out.push(' ');
            out.push(c);
            out.push(' ');
            i += 1;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

fn render_sym(t: &TaskRec, level: u8, unseen: bool, r: &mut Rng) -> String {
    let (l, rr) = t.spec.split_once("==").unwrap();
    let (l, rr) = (l.trim(), rr.trim());
    let lt = sym_side_text(l, level, unseen, r);
    let rt = sym_side_text(rr, level, unseen, r);
    let mut body = if level == 0 {
        format!("{} {} == {}", pick(r, &["check", "verify", "prove"]), l, rr)
    } else if unseen {
        match r.below(3) {
            0 => format!("Do {lt} and {rt} agree for every x?"),
            1 => format!("Is {lt} the same expression as {rt}?"),
            _ => format!("Does {lt} always equal {rt}?"),
        }
    } else {
        match r.below(3) {
            0 => format!("Is {lt} equal to {rt} for all x?"),
            1 => format!("Prove or refute: {lt} == {rt}."),
            _ => format!("Is {lt} identical to {rt}?"),
        }
    };
    if level >= 2 && r.f64() < 0.7 {
        body = with_distractor(body, r, unseen);
    }
    body
}

// ---------------------------------------------------------------------------------------------
// programs

fn prog_params(spec: &str) -> (usize, Vec<String>) {
    let nums = |s: &str| -> Vec<String> {
        let mut v = Vec::new();
        let mut cur = String::new();
        for c in s.chars() {
            if c.is_ascii_digit() {
                cur.push(c);
            } else if !cur.is_empty() {
                v.push(cur.clone());
                cur.clear();
            }
        }
        if !cur.is_empty() {
            v.push(cur);
        }
        v
    };
    let n = nums(spec);
    if spec.starts_with("let n = ") && spec.contains("3 * n + 1") {
        (0, vec![n[0].clone()])
    } else if spec.starts_with("let m = ") {
        (1, vec![n[0].clone()])
    } else if spec.starts_with("let a = ") && spec.contains("let t = a % b") {
        (2, vec![n[0].clone(), n[1].clone()])
    } else if spec.starts_with("let n = ") {
        (3, vec![n[0].clone()])
    } else if spec.starts_with("let s = 0; let i = 1;") {
        // while i <= N { s = (s + i * i) % M; ...
        let nn: Vec<String> = n.into_iter().collect();
        // numbers in order: 0(s), 1(i), N, M, 1(i+1) -> pick N = idx 2, M = idx 3
        (4, vec![nn[2].clone(), nn[3].clone()])
    } else {
        // fibonacci: let a = 0; let b = 1; let i = 0; while i < N {{ let t = (a + b) % M; ... i = i + 1
        (5, vec![n[3].clone(), n[4].clone()])
    }
}

fn render_prog(t: &TaskRec, level: u8, unseen: bool, r: &mut Rng) -> String {
    let (tpl, p) = prog_params(&t.spec);
    let wordy = level >= 2;
    let nt = |s: &str, r: &mut Rng| num_text(s, wordy && r.f64() < 0.5, level >= 2 && r.f64() < 0.4, unseen);
    let p0 = nt(&p[0], r);
    let mut body = match tpl {
        0 => {
            if unseen {
                format!("Take {p0}. Repeat: halve it if it is even, otherwise triple it and add one. How many repetitions until you reach 1?")
            } else {
                match r.below(3) {
                    0 => format!("How many steps does the Collatz process take to reach 1 starting from {p0}?"),
                    1 => format!("Simulate the Collatz sequence from {p0} (halve if even, else 3n+1) and count the steps until 1."),
                    _ => format!("Trace the Collatz iteration for {p0}; how many steps until it hits 1?"),
                }
            }
        }
        1 => {
            if unseen {
                format!("What do the digits of the square of {p0} add up to?")
            } else {
                match r.below(2) {
                    0 => format!("What is the sum of the digits of {p0} squared?"),
                    _ => format!("Compute {p0} squared and add up its digits."),
                }
            }
        }
        2 => {
            let p1 = nt(&p[1], r);
            if unseen {
                format!("Running Euclid's method on {p0} and {p1}, how many rounds of remainder-taking happen?")
            } else {
                match r.below(2) {
                    0 => format!("How many division steps does the Euclidean algorithm need for gcd({p0}, {p1})?"),
                    _ => format!("Iterate the Euclidean algorithm on {p0} and {p1}; count the remainder steps."),
                }
            }
        }
        3 => {
            if unseen {
                format!("How many prime numbers are at most {p0}?")
            } else {
                match r.below(2) {
                    0 => format!("How many primes are there up to {p0}?"),
                    _ => format!("Count the primes not exceeding {p0}."),
                }
            }
        }
        4 => {
            let p1 = nt(&p[1], r);
            if unseen {
                format!("Add up i*i for i running 1 through {p0}, then reduce by {p1} (remainder).")
            } else {
                match r.below(2) {
                    0 => format!("Compute the sum of i squared for i from 1 to {p0}, modulo {p1}."),
                    _ => format!("Iterate i from 1 to {p0} adding i squared, keeping the running total modulo {p1}. What is the final value?"),
                }
            }
        }
        _ => {
            let p1 = nt(&p[1], r);
            if unseen {
                format!("What is the {p0}-th term of the Fibonacci sequence mod {p1}, starting 0, 1?")
            } else {
                match r.below(2) {
                    0 => format!("What is the Fibonacci number after {p0} steps (starting 0, 1), modulo {p1}?"),
                    _ => format!("Iterate the Fibonacci recurrence {p0} times from 0 and 1 modulo {p1}; what is a?"),
                }
            }
        }
    };
    if level >= 2 && r.f64() < 0.7 {
        body = with_distractor(body, r, unseen);
    }
    body
}

// ---------------------------------------------------------------------------------------------
// lookup and prose

fn render_lookup(t: &TaskRec, level: u8, unseen: bool, r: &mut Rng) -> String {
    let (tb, k) = t.spec.split_once('.').unwrap();
    let mut body = if level == 0 {
        match r.below(2) {
            0 => format!("lookup {tb}.{k}"),
            _ => format!("{tb}.{k}?"),
        }
    } else if unseen {
        match r.below(2) {
            0 => format!("From the {tb} registry, fetch the entry for {k}."),
            _ => format!("Which value does {k} map to inside {tb}?"),
        }
    } else {
        match r.below(3) {
            0 => format!("Look up key {k} in table {tb}."),
            1 => format!("What is the recorded value for {k} in {tb}?"),
            _ => format!("table {tb}, key {k}: value?"),
        }
    };
    if level >= 2 && r.f64() < 0.7 {
        body = with_distractor(body, r, unseen);
    }
    body
}

fn render_qual(t: &TaskRec, level: u8, unseen: bool, hardneg_p: f64, r: &mut Rng) -> String {
    let nums: Vec<String> = t.spec.split_whitespace().filter(|w| w.chars().all(|c| c.is_ascii_digit())).map(|s| s.to_string()).collect();
    let (topic, case) = (nums.get(0).cloned().unwrap_or("1".into()), nums.get(1).cloned().unwrap_or("1".into()));
    let (topic, case) = if level >= 2 && r.f64() < 0.4 { (num_text(&topic, true, false, false), case) } else { (topic, case) };
    let hn = r.f64() < hardneg_p;
    let frag = if hn {
        match r.below(4) {
            0 => format!(" Note that {} plus {} does not settle it.", 3 + r.below(40), 2 + r.below(40)),
            1 => format!(" (Is it fair to say {} + {} * {} is a matter of convention?)", 2 + r.below(9), 2 + r.below(9), 2 + r.below(9)),
            2 => format!(" Do not just convert {} km to metres and stop there.", 2 + r.below(40)),
            _ => format!(" Some say x^2 == {} is trivial; ignore that.", 4 + r.below(30)),
        }
    } else {
        String::new()
    };
    let mut body = if unseen {
        match r.below(2) {
            0 => format!("Which of options A to D best captures the central tension in topic {topic}, case {case}? Reason it through.{frag}"),
            _ => format!("Weigh up topic {topic} (case {case}) and say which option, A B C or D, is the most defensible reading.{frag}"),
        }
    } else {
        match r.below(3) {
            0 => format!("Explain which of the following four options best describes the main trade off in topic {topic} for case {case}.{frag}"),
            1 => format!("Summarize the trade-offs for topic {topic} (case {case}) and say which of options A-D fits best.{frag}"),
            _ => format!("Discuss topic {topic}, case {case}: which option, A, B, C or D, captures the main trade off?{frag}"),
        }
    };
    if level >= 2 && r.f64() < 0.5 {
        body = with_distractor(body, r, unseen);
    }
    body
}

// ---------------------------------------------------------------------------------------------

/// delete one number token or insert a conflicting one (level 4)
fn degrade(text: &str, r: &mut Rng) -> String {
    let toks: Vec<&str> = text.split(' ').collect();
    let idx: Vec<usize> = (0..toks.len()).filter(|i| toks[*i].chars().any(|c| c.is_ascii_digit())).collect();
    if idx.is_empty() {
        return text.to_string();
    }
    let i = idx[r.below(idx.len())];
    let mut out: Vec<String> = toks.iter().map(|s| s.to_string()).collect();
    if r.f64() < 0.5 {
        out.remove(i);
    } else {
        let bump = 1 + r.below(8);
        let digits: String = toks[i].chars().filter(|c| c.is_ascii_digit()).collect();
        let alt = digits.parse::<u64>().map(|v| (v + bump as u64).to_string()).unwrap_or("7".into());
        out.insert(i + 1, alt);
    }
    out.join(" ")
}

pub fn render(t: &TaskRec, cfg: &NlCfg, r: &mut Rng) -> NlTask {
    let mut u = r.f64();
    let mut level = 4u8;
    for (l, p) in cfg.level_p.iter().enumerate() {
        if u < *p {
            level = l as u8;
            break;
        }
        u -= *p;
    }
    let unseen = r.f64() < cfg.unseen_p;
    let rl = level.min(3);
    let mut text = match t.kind {
        Kind::Arith => render_arith(t, rl, unseen, r),
        Kind::Units => render_units(t, rl, unseen, r),
        Kind::Csp => render_csp(t, rl, unseen, r),
        Kind::Sym => render_sym(t, rl, unseen, r),
        Kind::Prog => render_prog(t, rl, unseen, r),
        Kind::Lookup => render_lookup(t, rl, unseen, r),
        Kind::Qual => render_qual(t, rl, unseen, cfg.hardneg_p, r),
    };
    if rl >= 3 {
        let k = 1 + r.below(2);
        text = add_typos(&text, k, r);
    }
    if level == 4 && t.kind != Kind::Qual {
        text = degrade(&text, r);
    }
    let tg = tag(&t.kw, r, unseen);
    NlTask { text: format!("{tg}{text}"), level, unseen }
}

/// texts for a whole world (exact repeats of an earlier question reuse the earlier text)
pub fn render_world(tasks: &[TaskRec], cfg: &NlCfg, seed: u64) -> Vec<NlTask> {
    let mut out: Vec<NlTask> = Vec::with_capacity(tasks.len());
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, t) in tasks.iter().enumerate() {
        if t.dup {
            if let Some(j) = seen.get(&t.spec) {
                let c = out[*j].clone();
                out.push(c);
                continue;
            }
        }
        let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(i as u64 * 7919 + 13) ^ t.salt);
        let nl = render(t, cfg, &mut r);
        seen.entry(t.spec.clone()).or_insert(i);
        out.push(nl);
    }
    out
}
