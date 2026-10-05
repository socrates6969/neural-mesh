//! g21 deterministic, std-only spec extractor: natural-language-ish task text -> tool spec string (+ confidence).
//! Rule based: tokenizer, number-word folding, fuzzy lexicon (Damerau <= 1 on words >= 5 letters, stoplist guarded),
//! one candidate extractor per kind, a calibrated-by-rule confidence (multiplicative penalties), and abstention
//! (spec = None). It was developed ONLY against dev-bank texts of tune seeds 1-8.
use super::g13::Kind;

#[derive(Clone, Debug)]
pub struct Ex {
    pub spec: Option<String>,
    pub conf: f64,
    pub kind: Option<Kind>,
    /// number of competing candidate kinds with conf > 0.3
    pub n_cands: usize,
    pub fuzzy: usize,
    pub unused: usize,
}

// ---------------------------------------------------------------------------------------------
// tokens

#[derive(Clone, Debug, PartialEq)]
enum T {
    N(String),
    W(String),
    Op(String),
    LP,
    RP,
    Comma,
    Semi,
    Colon,
    Dot,
    Q,
    Other,
}

fn normalize(s: &str) -> String {
    let mut o = String::new();
    for c in s.chars() {
        match c {
            '\u{d7}' | '\u{b7}' => o.push_str(" * "),
            '\u{f7}' => o.push_str(" / "),
            '\u{2212}' => o.push('-'),
            '\u{2264}' => o.push_str(" <= "),
            '\u{2265}' => o.push_str(" >= "),
            '\u{b2}' => o.push_str("^2"),
            '\u{b3}' => o.push_str("^3"),
            c => o.extend(c.to_lowercase()),
        }
    }
    o
}

fn tokenize(raw: &str) -> Vec<T> {
    let s = normalize(raw);
    let b: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = Vec::new();
    // drop a leading [tag]
    if b.first() == Some(&'[') {
        if let Some(j) = b.iter().position(|c| *c == ']') {
            i = j + 1;
        }
    }
    let isd = |c: char| c.is_ascii_digit();
    while i < b.len() {
        let c = b[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if isd(c) {
            // date dddd-dd-dd
            if i + 9 < b.len() + 0 && b.len() >= i + 10 && (0..4).all(|k| isd(b[i + k])) && b[i + 4] == '-' && isd(b[i + 5]) && isd(b[i + 6]) && b[i + 7] == '-' && isd(b[i + 8]) && isd(b[i + 9]) {
                out.push(T::Other);
                i += 10;
                continue;
            }
            let st = i;
            while i < b.len() && isd(b[i]) {
                i += 1;
            }
            let mut num: String = b[st..i].iter().collect();
            // thousands separator
            while i + 3 < b.len() + 0 && b[i] == ',' && isd(b[i + 1]) && isd(b[i + 2]) && isd(b[i + 3]) && !(i + 4 < b.len() && isd(b[i + 4])) {
                num.extend(&b[i + 1..i + 4]);
                i += 4;
            }
            if i + 1 < b.len() && b[i] == '.' && isd(b[i + 1]) {
                i += 1;
                let fs = i;
                while i < b.len() && isd(b[i]) {
                    i += 1;
                }
                num.push('.');
                num.extend(&b[fs..i]);
            }
            out.push(T::N(num));
            continue;
        }
        if c.is_alphabetic() {
            let st = i;
            while i < b.len() && (b[i].is_alphabetic() || (b[i] == '\'' && i + 1 < b.len() && b[i + 1].is_alphabetic())) {
                i += 1;
            }
            // letters followed by digits (k17, t2) stay one word
            let mut j = i;
            while j < b.len() && isd(b[j]) {
                j += 1;
            }
            if j > i && (i - st) == 1 && matches!(b[st], 'k' | 't') {
                i = j;
            }
            out.push(T::W(b[st..i].iter().collect()));
            // hyphen between letters is a joiner
            if i + 1 < b.len() && b[i] == '-' && b[i + 1].is_alphabetic() && (i - st) >= 3 {
                let mut e2 = i + 1;
                while e2 < b.len() && b[e2].is_alphabetic() {
                    e2 += 1;
                }
                if e2 - (i + 1) >= 3 {
                    i += 1;
                }
            }
            continue;
        }
        let two: String = b[i..(i + 2).min(b.len())].iter().collect();
        match two.as_str() {
            "==" | "!=" | "<=" | ">=" | "->" | ".." => {
                out.push(T::Op(two));
                i += 2;
                continue;
            }
            _ => {}
        }
        match c {
            '#' => {
                // "#1234" is an identifier, not a number
                let mut j = i + 1;
                while j < b.len() && isd(b[j]) {
                    j += 1;
                }
                out.push(T::Other);
                i = j;
                continue;
            }
            '+' | '-' | '*' | '/' | '^' | '=' | '<' | '>' | '%' => out.push(T::Op(c.to_string())),
            '(' => out.push(T::LP),
            ')' => out.push(T::RP),
            ',' => out.push(T::Comma),
            ';' => out.push(T::Semi),
            ':' => out.push(T::Colon),
            '.' => out.push(T::Dot),
            '?' => out.push(T::Q),
            _ => out.push(T::Other),
        }
        i += 1;
    }
    out
}

// ---------------------------------------------------------------------------------------------
// fuzzy words

fn damerau(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        d[i][0] = i;
    }
    for j in 0..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[n][m]
}

const STOP: &[&str] = &[
    "there", "these", "where", "three", "other", "which", "their", "about", "after", "again", "being", "could", "every", "first", "found", "given", "great",
    "least", "might", "never", "often", "order", "place", "point", "right", "shall", "since", "small", "still", "think", "those", "under", "until", "using",
    "while", "whole", "would", "write", "fifth", "ninth", "tenth", "sixth", "number", "numbers", "results", "result", "total", "units", "steps", "thanks",
    "ticket", "round", "section", "problem", "question", "worth", "points", "score", "class", "group", "students", "route", "legs", "journey", "trail",
];

const LEX: &[&str] = &[
    "plus", "minus", "times", "multiplied", "divided", "squared", "cubed", "equals", "equal", "identical", "between", "compute", "evaluate", "calculate",
    "convert", "express", "collatz", "euclidean", "euclid", "fibonacci", "primes", "prime", "digits", "modulo", "quotient", "product", "difference",
    "square", "start", "begin", "subtract", "multiply", "divide", "explain", "discuss", "summarize", "summarise", "describe", "lookup", "table", "iterate",
    "constraints", "satisfy", "integers", "least", "most", "less", "than", "greater", "recorded", "simulate", "trace", "identity", "total", "sum", "power", "same", "answer",
];

fn is_lex(w: &str) -> bool {
    LEX.contains(&w)
}

/// canonical lexicon word for w (exact, or unique Damerau-1 match for words >= 5 letters)
fn canon_word(w: &str, fuzzy: &mut usize) -> String {
    if is_lex(w) || w.len() < 5 || STOP.contains(&w) || w.chars().any(|c| c.is_ascii_digit()) || UNIT_WORDS.iter().any(|u| u.0 == w) {
        return w.to_string();
    }
    let lim = if w.len() >= 9 { 2 } else { 1 };
    let mut best: Option<(&str, usize)> = None;
    let mut tie = false;
    for l in LEX {
        if l.len() < 4 {
            continue;
        }
        let d = damerau(w, l);
        if d <= lim {
            match best {
                None => best = Some((l, d)),
                Some((_, bd)) if d < bd => {
                    best = Some((l, d));
                    tie = false;
                }
                Some((bl, bd)) if d == bd && bl != *l => {
                    let (dl_old, dl_new) = ((bl.len() as i64 - w.len() as i64).abs(), (l.len() as i64 - w.len() as i64).abs());
                    if dl_new < dl_old {
                        best = Some((l, d));
                        tie = false;
                    } else if dl_new == dl_old {
                        tie = true;
                    }
                }
                _ => {}
            }
        }
    }
    match best {
        Some((l, _)) if !tie => {
            *fuzzy += 1;
            l.to_string()
        }
        _ => w.to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// number words

#[derive(Clone, Copy, Debug)]
enum NW {
    V(u64),
    Hundred,
    Thousand,
    Point,
}
const NW_TABLE: &[(&str, NW)] = &[
    ("zero", NW::V(0)), ("one", NW::V(1)), ("two", NW::V(2)), ("three", NW::V(3)), ("four", NW::V(4)), ("five", NW::V(5)), ("six", NW::V(6)),
    ("seven", NW::V(7)), ("eight", NW::V(8)), ("nine", NW::V(9)), ("ten", NW::V(10)), ("eleven", NW::V(11)), ("twelve", NW::V(12)),
    ("thirteen", NW::V(13)), ("fourteen", NW::V(14)), ("fifteen", NW::V(15)), ("sixteen", NW::V(16)), ("seventeen", NW::V(17)),
    ("eighteen", NW::V(18)), ("nineteen", NW::V(19)), ("twenty", NW::V(20)), ("thirty", NW::V(30)), ("forty", NW::V(40)), ("fifty", NW::V(50)),
    ("sixty", NW::V(60)), ("seventy", NW::V(70)), ("eighty", NW::V(80)), ("ninety", NW::V(90)), ("hundred", NW::Hundred), ("thousand", NW::Thousand),
    ("point", NW::Point),
];

fn nw(w: &str, fuzzy: &mut usize) -> Option<NW> {
    if let Some((_, v)) = NW_TABLE.iter().find(|(n, _)| *n == w) {
        return Some(*v);
    }
    if w.len() < 5 || STOP.contains(&w) || is_lex(w) || UNIT_WORDS.iter().any(|u| u.0 == w) {
        return None;
    }
    let mut best: Option<(NW, &str)> = None;
    let mut cnt = 0;
    for (n, v) in NW_TABLE {
        if n.len() >= 5 && damerau(w, n) <= 1 {
            best = Some((*v, n));
            cnt += 1;
        }
    }
    if cnt == 1 {
        *fuzzy += 1;
        return best.map(|b| b.0);
    }
    None
}

const MATHCTX: &[&str] = &["plus", "minus", "times", "and", "point", "squared", "cubed", "by", "of", "hundred", "thousand", "divided", "multiplied", "over", "is", "start", "add", "subtract", "multiply", "divide", "with", "to"];

/// like `nw` but, inside a math context, also allows a Damerau-1 fix of 4-letter number words (four, five, nine, zero)
fn nw_ctx(w: &str, ctx: bool, fuzzy: &mut usize) -> Option<NW> {
    if let Some(v) = nw(w, fuzzy) {
        return Some(v);
    }
    if ctx && w.len() >= 4 && w.len() <= 5 && !STOP.contains(&w) && !is_lex(w) && !UNIT_WORDS.iter().any(|u| u.0 == w) {
        let mut best: Option<NW> = None;
        let mut cnt = 0;
        for (n, v) in NW_TABLE {
            if n.len() >= 4 && damerau(w, n) <= 1 {
                best = Some(*v);
                cnt += 1;
            }
        }
        if cnt == 1 {
            *fuzzy += 1;
            return best;
        }
    }
    None
}

fn fold_numbers(toks: Vec<T>, fuzzy: &mut usize) -> Vec<T> {
    let mut out = Vec::new();
    let mut i = 0;
    let ctx_at = |toks: &Vec<T>, i: usize| -> bool {
        let m = |t: Option<&T>| match t {
            Some(T::N(_)) | Some(T::Op(_)) | Some(T::LP) | Some(T::RP) => true,
            Some(T::W(x)) => MATHCTX.contains(&x.as_str()),
            _ => false,
        };
        m(toks.get(i.wrapping_sub(1))) || m(toks.get(i + 1))
    };
    while i < toks.len() {
        if let T::W(w) = &toks[i] {
            let mut f0 = 0usize;
            let ctx = ctx_at(&toks, i);
            if let Some(first) = nw_ctx(w, ctx, &mut f0) {
                if !matches!(first, NW::Point) {
                    let mut total: u64 = 0;
                    let mut cur: u64 = 0;
                    let mut j = i;
                    let mut ff = 0usize;
                    let mut last_big = false;
                    let mut ok_any = false;
                    while j < toks.len() {
                        match &toks[j] {
                            T::W(w2) => {
                                let mut f1 = 0usize;
                                if w2 == "and" && last_big && j + 1 < toks.len() && matches!(&toks[j + 1], T::W(n3) if { let mut z = 0; nw(n3, &mut z).map_or(false, |v| !matches!(v, NW::Point)) }) {
                                    j += 1;
                                    continue;
                                }
                                match nw_ctx(w2, ctx || ok_any, &mut f1) {
                                    Some(NW::V(v)) => {
                                        cur += v;
                                        last_big = false;
                                        ok_any = true;
                                    }
                                    Some(NW::Hundred) if ok_any => {
                                        cur = if cur == 0 { 1 } else { cur } * 100;
                                        last_big = true;
                                    }
                                    Some(NW::Thousand) if ok_any => {
                                        total += if cur == 0 { 1 } else { cur } * 1000;
                                        cur = 0;
                                        last_big = true;
                                    }
                                    _ => break,
                                }
                                ff += f1;
                                j += 1;
                            }
                            _ => break,
                        }
                    }
                    if ok_any {
                        let mut s = (total + cur).to_string();
                        // decimals: "point" digit words
                        let mut pz = 0usize;
                        if j < toks.len() && matches!(&toks[j], T::W(p) if matches!(nw(p, &mut pz), Some(NW::Point))) {
                            let mut k = j + 1;
                            let mut frac = String::new();
                            let mut fz = 0usize;
                            while k < toks.len() {
                                if let T::W(d) = &toks[k] {
                                    let mut z = 0;
                                    if let Some(NW::V(v)) = nw_ctx(d, true, &mut z) {
                                        if v < 10 {
                                            frac.push_str(&v.to_string());
                                            fz += z;
                                            k += 1;
                                            continue;
                                        }
                                    }
                                }
                                break;
                            }
                            if !frac.is_empty() {
                                s.push('.');
                                s.push_str(&frac);
                                j = k;
                                ff += fz + pz;
                            }
                        }
                        *fuzzy += ff;
                        out.push(T::N(s));
                        i = j;
                        continue;
                    }
                }
            }
        }
        out.push(toks[i].clone());
        i += 1;
    }
    out
}


// ---------------------------------------------------------------------------------------------
// second stage: math symbols

#[derive(Clone, Debug, PartialEq)]
enum S {
    Num(String),
    Var(char),
    Op(char),
    LP,
    RP,
    Rel(&'static str),
    Word(String),
    Comma,
    Semi,
    Colon,
    Dot,
    Q,
    DotDot,
    Arrow,
    Other,
}

fn lex2(toks: &[T], fuzzy: &mut usize) -> Vec<S> {
    // canonical words first
    let mut w: Vec<T> = Vec::new();
    for t in toks {
        match t {
            T::W(x) => w.push(T::W(canon_word(x, fuzzy))),
            o => w.push(o.clone()),
        }
    }
    let word = |i: usize, s: &str| matches!(w.get(i), Some(T::W(x)) if x == s);
    let mut out = Vec::new();
    let mut i = 0;
    while i < w.len() {
        match &w[i] {
            T::N(n) => out.push(S::Num(n.clone())),
            T::LP => out.push(S::LP),
            T::RP => out.push(S::RP),
            T::Comma => out.push(S::Comma),
            T::Semi => out.push(S::Semi),
            T::Colon => out.push(S::Colon),
            T::Dot => out.push(S::Dot),
            T::Q => out.push(S::Q),
            T::Other => out.push(S::Other),
            T::Op(o) => match o.as_str() {
                "+" | "-" | "*" | "/" | "^" | "%" => out.push(S::Op(o.chars().next().unwrap())),
                "==" => out.push(S::Rel("==")),
                "=" => out.push(S::Rel("==")),
                "<=" => out.push(S::Rel("<=")),
                ">=" => out.push(S::Rel(">=")),
                "<" => out.push(S::Rel("<")),
                ">" => out.push(S::Rel(">")),
                "!=" => out.push(S::Rel("!=")),
                ".." => out.push(S::DotDot),
                "->" => out.push(S::Arrow),
                _ => out.push(S::Other),
            },
            T::W(x) => {
                let x = x.as_str();
                // multiword phrases
                if (x == "multiplied" || x == "multiply") && word(i + 1, "by") {
                    out.push(S::Op('*'));
                    i += 2;
                    continue;
                }
                if (x == "divided" || x == "divide") && word(i + 1, "by") {
                    out.push(S::Op('/'));
                    i += 2;
                    continue;
                }
                if x == "to" && word(i + 1, "the") && word(i + 2, "power") && word(i + 3, "of") {
                    out.push(S::Op('^'));
                    i += 4;
                    continue;
                }
                if (x == "is" || x == "are") && (word(i + 1, "equal") || word(i + 1, "identical") || word(i + 1, "less") || word(i + 1, "greater") || word(i + 1, "at") || word(i + 1, "the") && word(i + 2, "same")) {
                    i += 1;
                    continue;
                }
                if (x == "equal" || x == "identical") && word(i + 1, "to") {
                    out.push(S::Rel("=="));
                    i += 2;
                    continue;
                }
                if x == "the" && word(i + 1, "same") && (word(i + 2, "as") || word(i + 2, "expression")) {
                    out.push(S::Rel("=="));
                    i += 3;
                    continue;
                }
                if x == "less" && word(i + 1, "than") {
                    out.push(S::Rel("<"));
                    i += 2;
                    continue;
                }
                if x == "greater" && word(i + 1, "than") {
                    out.push(S::Rel(">"));
                    i += 2;
                    continue;
                }
                if x == "at" && word(i + 1, "most") {
                    out.push(S::Rel("<="));
                    i += 2;
                    continue;
                }
                if x == "at" && word(i + 1, "least") {
                    out.push(S::Rel(">="));
                    i += 2;
                    continue;
                }
                match x {
                    "plus" => out.push(S::Op('+')),
                    "minus" => out.push(S::Op('-')),
                    "times" => out.push(S::Op('*')),
                    "over" => out.push(S::Op('/')),
                    "squared" => {
                        out.push(S::Op('^'));
                        out.push(S::Num("2".into()));
                    }
                    "cubed" => {
                        out.push(S::Op('^'));
                        out.push(S::Num("3".into()));
                    }
                    "equals" | "gives" => out.push(S::Rel("==")),
                    "x" | "y" | "z" | "w" => out.push(S::Var(x.chars().next().unwrap())),
                    _ => out.push(S::Word(x.to_string())),
                }
            }
        }
        i += 1;
    }
    out
}

fn is_mathy(s: &S, vars: bool) -> bool {
    match s {
        S::Num(_) | S::Op(_) | S::LP | S::RP => true,
        S::Var(_) => vars,
        _ => false,
    }
}

// ---------------------------------------------------------------------------------------------
// math expression parsing (arith / sym / csp share it; validates and re-serialises)

#[derive(Clone, Debug)]
enum E {
    Num(String),
    Var(char),
    Neg(Box<E>),
    Bin(char, Box<E>, Box<E>),
}

struct P<'a> {
    t: &'a [S],
    i: usize,
    implicit: bool,
}
impl<'a> P<'a> {
    fn peek(&self) -> Option<&S> {
        self.t.get(self.i)
    }
    fn expr(&mut self) -> Option<E> {
        let mut l = self.term()?;
        while let Some(S::Op(c)) = self.peek() {
            if *c != '+' && *c != '-' {
                break;
            }
            let c = *c;
            self.i += 1;
            let r = self.term()?;
            l = E::Bin(c, Box::new(l), Box::new(r));
        }
        Some(l)
    }
    fn term(&mut self) -> Option<E> {
        let mut l = self.unary()?;
        loop {
            match self.peek() {
                Some(S::Op(c)) if *c == '*' || *c == '/' || *c == '%' => {
                    let c = *c;
                    self.i += 1;
                    let r = self.unary()?;
                    l = E::Bin(c, Box::new(l), Box::new(r));
                }
                // implicit multiplication: 2x, 2(, )(, )x, x(  (never between two plain numbers or var-var)
                Some(S::Var(_)) | Some(S::LP) if self.implicit => {
                    let prev_ok = matches!(self.t.get(self.i.wrapping_sub(1)), Some(S::Num(_)) | Some(S::RP) | Some(S::Var(_)));
                    if !prev_ok {
                        break;
                    }
                    let r = self.unary()?;
                    l = E::Bin('*', Box::new(l), Box::new(r));
                }
                _ => break,
            }
        }
        Some(l)
    }
    fn unary(&mut self) -> Option<E> {
        if let Some(S::Op('-')) = self.peek() {
            self.i += 1;
            return Some(E::Neg(Box::new(self.unary()?)));
        }
        self.pow()
    }
    fn pow(&mut self) -> Option<E> {
        let b = self.atom()?;
        if let Some(S::Op('^')) = self.peek() {
            self.i += 1;
            let e = self.unary()?;
            return Some(E::Bin('^', Box::new(b), Box::new(e)));
        }
        Some(b)
    }
    fn atom(&mut self) -> Option<E> {
        match self.t.get(self.i)? {
            S::Num(n) => {
                self.i += 1;
                Some(E::Num(n.clone()))
            }
            S::Var(v) => {
                self.i += 1;
                Some(E::Var(*v))
            }
            S::LP => {
                self.i += 1;
                let e = self.expr()?;
                if let Some(S::RP) = self.peek() {
                    self.i += 1;
                    Some(e)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

fn parse_math(t: &[S], implicit: bool) -> Option<E> {
    if t.is_empty() {
        return None;
    }
    let mut p = P { t, i: 0, implicit };
    let e = p.expr()?;
    if p.i == t.len() { Some(e) } else { None }
}

fn ser(e: &E) -> String {
    match e {
        E::Num(n) => n.clone(),
        E::Var(v) => v.to_string(),
        E::Neg(a) => format!("-{}", wrap(a)),
        E::Bin(o, a, b) => format!("{}{}{}", wrap(a), o, wrap(b)),
    }
}
fn wrap(e: &E) -> String {
    match e {
        E::Num(_) | E::Var(_) => ser(e),
        _ => format!("({})", ser(e)),
    }
}
fn count_ops(e: &E) -> usize {
    match e {
        E::Num(_) | E::Var(_) => 0,
        E::Neg(a) => count_ops(a),
        E::Bin(_, a, b) => 1 + count_ops(a) + count_ops(b),
    }
}

/// maximal mathy runs [start, end)
fn runs(s: &[S], vars: bool) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        if is_mathy(&s[i], vars) {
            let st = i;
            while i < s.len() && is_mathy(&s[i], vars) {
                i += 1;
            }
            out.push((st, i));
        } else {
            i += 1;
        }
    }
    out
}

/// try to parse a run, trimming up to 2 stray tokens at either end
fn parse_run(s: &[S], implicit: bool) -> Option<(E, usize, usize)> {
    for lt in 0..3 {
        for rt in 0..3 {
            if lt + rt >= s.len() {
                continue;
            }
            let sub = &s[lt..s.len() - rt];
            if let Some(e) = parse_math(sub, implicit) {
                return Some((e, lt, rt));
            }
        }
    }
    None
}

fn n_nums(s: &[S]) -> usize {
    s.iter().filter(|x| matches!(x, S::Num(_))).count()
}

fn has_word(s: &[S], w: &str) -> bool {
    s.iter().any(|x| matches!(x, S::Word(y) if y == w))
}
fn has_any(s: &[S], ws: &[&str]) -> bool {
    ws.iter().any(|w| has_word(s, w))
}

// ---------------------------------------------------------------------------------------------
// units

const UNIT_WORDS: &[(&str, &str)] = &[
    ("m", "m"), ("metre", "m"), ("metres", "m"), ("meter", "m"), ("meters", "m"),
    ("km", "km"), ("kms", "km"), ("kilometre", "km"), ("kilometres", "km"), ("kilometer", "km"), ("kilometers", "km"),
    ("cm", "cm"), ("centimetre", "cm"), ("centimetres", "cm"), ("centimeter", "cm"), ("centimeters", "cm"),
    ("mm", "mm"), ("millimetre", "mm"), ("millimetres", "mm"), ("millimeter", "mm"), ("millimeters", "mm"),
    ("ft", "ft"), ("foot", "ft"), ("feet", "ft"), ("in", "in"), ("inch", "in"), ("inches", "in"), ("yd", "yd"), ("yard", "yd"), ("yards", "yd"),
    ("kg", "kg"), ("kilogram", "kg"), ("kilograms", "kg"), ("kilo", "kg"), ("kilos", "kg"), ("g", "g"), ("gram", "g"), ("grams", "g"),
    ("mg", "mg"), ("milligram", "mg"), ("milligrams", "mg"), ("lb", "lb"), ("lbs", "lb"), ("pound", "lb"), ("pounds", "lb"), ("t", "t"), ("tonne", "t"), ("tonnes", "t"),
    ("s", "s"), ("sec", "s"), ("secs", "s"), ("second", "s"), ("seconds", "s"), ("ms", "ms"), ("millisecond", "ms"), ("milliseconds", "ms"),
    ("min", "min"), ("mins", "min"), ("minute", "min"), ("minutes", "min"), ("h", "h"), ("hr", "h"), ("hrs", "h"), ("hour", "h"), ("hours", "h"),
    ("day", "day"), ("days", "day"), ("week", "week"), ("weeks", "week"),
    ("knot", "knot"), ("knots", "knot"), ("mile", "mile"), ("miles", "mile"),
    ("l", "L"), ("litre", "L"), ("litres", "L"), ("liter", "L"), ("liters", "L"), ("ml", "mL"), ("millilitre", "mL"), ("millilitres", "mL"), ("milliliter", "mL"), ("milliliters", "mL"),
];

fn group_of(sym: &str) -> u8 {
    match sym {
        "m" | "km" | "cm" | "mm" | "ft" | "in" | "yd" | "mile" => 0,
        "kg" | "g" | "mg" | "lb" | "t" => 1,
        "s" | "ms" | "min" | "h" | "day" | "week" => 2,
        "m/s" | "km/h" | "knot" | "mile/h" => 3,
        "L" | "mL" | "m^3" => 4,
        _ => 9,
    }
}

#[derive(Clone, Debug)]
struct UMention {
    sym: String,
    start: usize,
    end: usize,
    attached: Option<String>,
    fuzzy: bool,
}

fn unit_word(w: &str) -> Option<(&'static str, bool)> {
    if let Some((_, s)) = UNIT_WORDS.iter().find(|(n, _)| *n == w) {
        return Some((s, false));
    }
    if w.len() >= 6 && !STOP.contains(&w) {
        let mut best: Option<&str> = None;
        let mut cnt = 0;
        for (n, s) in UNIT_WORDS {
            if n.len() >= 6 && damerau(w, n) <= 1 {
                if best != Some(s) {
                    cnt += 1;
                }
                best = Some(s);
            }
        }
        if cnt == 1 {
            return best.map(|b| (b, true));
        }
    }
    None
}

/// contextual fuzzy: a word right after a number (or after a "to/in/as/many/per" cue) within Damerau 1 of a unit name (>= 3 letters)
fn unit_word_ctx(w: &str) -> Option<(&'static str, bool)> {
    if let Some(r) = unit_word(w) {
        return Some(r);
    }
    if w.len() >= 3 && !STOP.contains(&w) && !is_lex(w) && !matches!(w, "may" | "many" | "how" | "the" | "and" | "are" | "was" | "for" | "its" | "has" | "had") {
        let mut best: Option<&str> = None;
        let mut cnt = 0;
        for (n, s) in UNIT_WORDS {
            if n.len() >= 3 && damerau(w, n) <= 1 {
                if best != Some(s) {
                    cnt += 1;
                }
                best = Some(s);
            }
        }
        if cnt == 1 {
            return best.map(|b| (b, true));
        }
    }
    None
}

fn unit_mentions(s: &[S]) -> Vec<UMention> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let attached = |i: usize| -> Option<String> {
            if i > 0 {
                if let S::Num(n) = &s[i - 1] { Some(n.clone()) } else { None }
            } else {
                None
            }
        };
        let prev_cue = |i: usize| -> bool {
            i > 0 && (matches!(&s[i - 1], S::Word(p) if matches!(p.as_str(), "to" | "into" | "as" | "many" | "using" | "in" | "per")) || matches!(s[i - 1], S::Arrow))
        };
        // a/b symbolic compounds and m^3
        if let (S::Word(a), Some(S::Op('/')), Some(S::Word(b))) = (&s[i], s.get(i + 1), s.get(i + 2)) {
            let comp = match (a.as_str(), b.as_str()) {
                ("m", "s") => Some("m/s"),
                ("km", "h") | ("km", "hr") => Some("km/h"),
                ("mile", "h") | ("miles", "h") | ("mile", "hr") => Some("mile/h"),
                _ => None,
            };
            if let Some(c) = comp {
                out.push(UMention { sym: c.into(), start: i, end: i + 3, attached: attached(i), fuzzy: false });
                i += 3;
                continue;
            }
        }
        if let (S::Word(a), Some(S::Op('^')), Some(S::Num(e))) = (&s[i], s.get(i + 1), s.get(i + 2)) {
            if a == "m" && e == "3" {
                out.push(UMention { sym: "m^3".into(), start: i, end: i + 3, attached: attached(i), fuzzy: false });
                i += 3;
                continue;
            }
        }
        if let S::Word(a) = &s[i] {
            let nxt = |k: usize| -> Option<&str> { if let Some(S::Word(x)) = s.get(i + k) { Some(x.as_str()) } else { None } };
            // "<length> per <time>" families (typo-tolerant on the unit words)
            let lenw = |w: &str| matches!(unit_word_ctx(w), Some(("m", _)) | Some(("km", _)) | Some(("mile", _)));
            if nxt(1) == Some("per") {
                if let (Some(t), true) = (nxt(2), lenw(a)) {
                    if let Some((ts, fz)) = unit_word_ctx(t) {
                        let lsym = unit_word_ctx(a).map(|x| x.0);
                        let comp = match (lsym, ts) {
                            (Some("m"), "s") => Some("m/s"),
                            (Some("km"), "h") => Some("km/h"),
                            (Some("mile"), "h") => Some("mile/h"),
                            _ => None,
                        };
                        if let Some(c) = comp {
                            out.push(UMention { sym: c.into(), start: i, end: i + 3, attached: attached(i), fuzzy: fz });
                            i += 3;
                            continue;
                        }
                    }
                }
            }
            if (a == "cubic" || (a.len() >= 5 && damerau(a, "cubic") <= 1)) && matches!(nxt(1), Some(w) if matches!(unit_word_ctx(w), Some(("m", _)))) {
                out.push(UMention { sym: "m^3".into(), start: i, end: i + 2, attached: attached(i), fuzzy: false });
                i += 2;
                continue;
            }
            let att = attached(i);
            let cue = prev_cue(i);
            let looked = if att.is_some() || cue { unit_word_ctx(a) } else { unit_word(a) };
            if let Some((sym, fz)) = looked {
                let amb = matches!(a.as_str(), "in" | "t" | "g" | "s" | "h" | "l" | "m" | "min" | "mile" | "miles");
                if a == "in" {
                    // inches only after a number, or as the TARGET right after another "in" / "to" / "as" ("3 yards in in")
                    let more_units = (1..=3).any(|k| matches!(s.get(i + k), Some(S::Word(w2)) if unit_word_ctx(w2).is_some()));
                    let after_prep = i > 0 && matches!(&s[i - 1], S::Word(p) if matches!(p.as_str(), "in" | "to" | "into" | "as" | "many")) && !(matches!(&s[i - 1], S::Word(p) if p == "in") && more_units);
                    if att.is_none() && !after_prep {
                        i += 1;
                        continue;
                    }
                } else if amb && att.is_none() && !cue && !matches!(a.as_str(), "min" | "mile" | "miles") {
                    i += 1;
                    continue;
                }
                out.push(UMention { sym: sym.into(), start: i, end: i + 1, attached: att, fuzzy: fz });
            }
        }
        i += 1;
    }
    out
}

const FUNC: &[&str] = &["of", "and", "to", "in", "is", "from", "a", "the", "are", "as", "then", "with", "by", "for", "on", "at", "per", "th", "st", "nd", "rd", "or", "it", "how", "what"];

fn ex_units(s: &[S]) -> Option<(String, f64, usize, usize)> {
    let ms = unit_mentions(s);
    if ms.len() < 2 {
        return None;
    }
    let targets: Vec<&UMention> = ms.iter().filter(|m| m.attached.is_none()).collect();
    if targets.is_empty() {
        return None;
    }
    // target: unattached mention preceded by a cue; else the last unattached
    let cue = |m: &UMention| -> bool {
        m.start > 0 && match &s[m.start - 1] {
            S::Word(w) => matches!(w.as_str(), "to" | "in" | "into" | "as" | "many" | "using" | "terms" | "of"),
            S::Arrow => true,
            _ => false,
        }
    };
    let cued: Vec<&&UMention> = targets.iter().filter(|m| cue(m)).collect();
    let mut ambiguity = false;
    let target: &UMention = if cued.len() == 1 {
        cued[0]
    } else if cued.len() > 1 {
        ambiguity = true;
        cued[0]
    } else {
        if targets.len() > 1 {
            ambiguity = true;
        }
        targets[targets.len() - 1]
    };
    let tg = group_of(&target.sym);
    let mut qs: Vec<(String, String)> = Vec::new();
    let mut dropped = 0usize;
    let mut fuzzy = 0usize;
    for m in &ms {
        if let Some(v) = &m.attached {
            // quantity cues that mark a distractor ("within 30 seconds")
            let pre = if m.start >= 2 { Some(&s[m.start - 2]) } else { None };
            let distractor = matches!(pre, Some(S::Word(p)) if p == "within");
            if group_of(&m.sym) != tg || distractor {
                dropped += 1;
                continue;
            }
            if m.fuzzy {
                fuzzy += 1;
            }
            qs.push((v.clone(), m.sym.clone()));
        }
    }
    if target.fuzzy {
        fuzzy += 1;
    }
    if qs.is_empty() {
        return None;
    }
    let used_nums = qs.len();
    let total_nums = n_nums(s);
    let unused = total_nums.saturating_sub(used_nums);
    // numbers followed by an unknown word: probably a quantity whose unit we did not understand
    let mut orphans = 0usize;
    for i in 0..s.len().saturating_sub(1) {
        if let (S::Num(_), S::Word(w)) = (&s[i], &s[i + 1]) {
            let known = ms.iter().any(|m| m.start == i + 1) || unit_word_ctx(w).is_some() || STOP.contains(&w.as_str()) || is_lex(w) || FUNC.contains(&w.as_str());
            if !known {
                orphans += 1;
            }
        }
    }
    let spec = if qs.len() == 1 {
        format!("conv {} {} -> {}", qs[0].0, qs[0].1, target.sym)
    } else {
        format!("sum {} -> {}", qs.iter().map(|(v, u)| format!("{v} {u}")).collect::<Vec<_>>().join(" + "), target.sym)
    };
    let mut c = 0.95f64 * 0.9f64.powi(fuzzy as i32) * 0.9f64.powi(unused.min(3) as i32) * 0.5f64.powi(orphans.min(2) as i32);
    if ambiguity {
        c *= 0.5;
    }
    if dropped > 0 {
        c *= 0.85;
    }
    Some((spec, c, fuzzy, unused))
}

// ---------------------------------------------------------------------------------------------
// arithmetic

const ARITH_CUES: &[&str] = &["compute", "evaluate", "calculate", "is", "out", "of", "value", "determine", "me", "total", "sum", "check", "express"];

fn ex_arith_tree(s: &[S]) -> Option<(E, usize)> {
    // prefix phrases: the sum|difference|product|quotient|square of ...
    fn operand(s: &[S], i: &mut usize, used: &mut usize) -> Option<E> {
        match s.get(*i)? {
            S::Num(n) => {
                *i += 1;
                *used += 1;
                // "n squared" is lexed as Num ^ 2
                if let (Some(S::Op('^')), Some(S::Num(e))) = (s.get(*i), s.get(*i + 1)) {
                    *i += 2;
                    return Some(E::Bin('^', Box::new(E::Num(n.clone())), Box::new(E::Num(e.clone()))));
                }
                Some(E::Num(n.clone()))
            }
            S::Word(w) if w == "the" => phrase(s, i, used),
            _ => None,
        }
    }
    fn phrase(s: &[S], i: &mut usize, used: &mut usize) -> Option<E> {
        if !matches!(s.get(*i), Some(S::Word(w)) if w == "the") {
            return None;
        }
        let kind = match s.get(*i + 1) {
            Some(S::Word(w)) => w.clone(),
            _ => return None,
        };
        if !matches!(s.get(*i + 2), Some(S::Word(w)) if w == "of") {
            return None;
        }
        *i += 3;
        let a = operand(s, i, used)?;
        if kind == "square" {
            return Some(E::Bin('^', Box::new(a), Box::new(E::Num("2".into()))));
        }
        if !matches!(s.get(*i), Some(S::Word(w)) if w == "and") {
            return None;
        }
        *i += 1;
        let b = operand(s, i, used)?;
        let op = match kind.as_str() {
            "sum" => '+',
            "difference" => '-',
            "product" => '*',
            "quotient" => '/',
            _ => return None,
        };
        Some(E::Bin(op, Box::new(a), Box::new(b)))
    }
    let n = s.len();
    // the LONGEST phrase parse wins; if an earlier phrase start failed to parse, the caller is told (skipped)
    let is_start = |st: usize| matches!(&s[st], S::Word(w) if w == "the") && matches!(s.get(st + 1), Some(S::Word(k)) if matches!(k.as_str(), "sum" | "difference" | "product" | "quotient" | "square")) && matches!(s.get(st + 2), Some(S::Word(o)) if o == "of");
    let mut best: Option<(E, usize, usize, usize)> = None; // expr, used, start, end
    let mut first_start: Option<usize> = None;
    for st in 0..n {
        if is_start(st) {
            if first_start.is_none() {
                first_start = Some(st);
            }
            let mut i = st;
            let mut used = 0;
            if let Some(e) = phrase(s, &mut i, &mut used) {
                if best.as_ref().map_or(true, |b| i - st > b.3 - b.2) {
                    best = Some((e, used, st, i));
                }
            }
        }
    }
    let (e, used, st, _) = best?;
    let skipped = first_start.map_or(false, |f| f < st);
    Some((e, if skipped { used + 100 } else { used }))
}

fn ex_arith_proc(s: &[S]) -> Option<(E, usize)> {
    // "start with A. add B. multiply the result by C. ..."
    let st = s.iter().position(|x| matches!(x, S::Word(w) if w == "start" || w == "begin" || w == "take"))?;
    let mut i = st + 1;
    if matches!(s.get(i), Some(S::Word(w)) if w == "with" || w == "at") {
        i += 1;
    }
    let mut e = match s.get(i)? {
        S::Num(n) => E::Num(n.clone()),
        _ => return None,
    };
    let mut used = 1;
    i += 1;
    if let (Some(S::Op('^')), Some(S::Num(x))) = (s.get(i), s.get(i + 1)) {
        e = E::Bin('^', Box::new(e), Box::new(E::Num(x.clone())));
        i += 2;
    }
    let skip = ["the", "result", "by", "to", "from", "it", "that"];
    while i < s.len() {
        if let S::Word(w) = &s[i] {
            let opc = match w.as_str() {
                "add" => Some('+'),
                "subtract" => Some('-'),
                "multiply" => Some('*'),
                "divide" => Some('/'),
                "square" => Some('^'),
                _ => None,
            };
            if let Some(c) = opc {
                if c == '^' {
                    e = E::Bin('^', Box::new(e), Box::new(E::Num("2".into())));
                    i += 1;
                    continue;
                }
                // next number within 5 tokens, skipping filler words
                let mut j = i + 1;
                let mut found = None;
                while j < s.len() && j <= i + 5 {
                    match &s[j] {
                        S::Num(n) => {
                            found = Some((n.clone(), j));
                            break;
                        }
                        S::Word(x) if skip.contains(&x.as_str()) || !matches!(x.as_str(), "add" | "subtract" | "multiply" | "divide" | "square") => j += 1,
                        _ => break,
                    }
                }
                let (n, j) = found?;
                let mut rhs = E::Num(n);
                let mut j2 = j + 1;
                if let (Some(S::Op('^')), Some(S::Num(x))) = (s.get(j2), s.get(j2 + 1)) {
                    rhs = E::Bin('^', Box::new(rhs), Box::new(E::Num(x.clone())));
                    j2 += 2;
                }
                e = E::Bin(c, Box::new(e), Box::new(rhs));
                used += 1;
                i = j2;
                continue;
            }
        }
        i += 1;
    }
    if used >= 2 { Some((e, used)) } else { None }
}

fn ex_arith(s: &[S]) -> Option<(String, f64, usize)> {
    let total_nums = n_nums(s);
    // 'x' between numbers is a multiplication sign
    let mut s2: Vec<S> = s.to_vec();
    // a word sandwiched between two operands within Damerau 1 of an operator word is that operator (typo repair)
    for i in 1..s2.len().saturating_sub(1) {
        if let S::Word(wd) = &s2[i] {
            let l = matches!(s2[i - 1], S::Num(_) | S::RP);
            let r = matches!(s2[i + 1], S::Num(_) | S::LP);
            if l && r && wd.len() >= 3 && !STOP.contains(&wd.as_str()) {
                let mut c: Vec<char> = Vec::new();
                for (ow, oc) in [("plus", '+'), ("minus", '-'), ("times", '*'), ("over", '/')] {
                    if damerau(wd, ow) <= 1 {
                        c.push(oc);
                    }
                }
                if c.len() == 1 {
                    s2[i] = S::Op(c[0]);
                }
            }
        }
    }
    for i in 1..s2.len().saturating_sub(1) {
        if let S::Var('x') = s2[i] {
            let l = matches!(s2[i - 1], S::Num(_) | S::RP);
            let r = matches!(s2[i + 1], S::Num(_) | S::LP);
            if l && r {
                s2[i] = S::Op('*');
            }
        }
    }
    let s = &s2[..];
    let cue_of = |st: usize| -> bool {
        (st.saturating_sub(4)..st).any(|k| matches!(&s[k], S::Word(w) if ARITH_CUES.contains(&w.as_str())))
    };
    let proc_cue = s.iter().any(|x| matches!(x, S::Word(w) if w == "start" || w == "begin"));
    if proc_cue {
        if let Some((e, used)) = ex_arith_proc(s) {
            let unused = total_nums.saturating_sub(used);
            let c = 0.94 * 0.9f64.powi(unused.min(3) as i32);
            return Some((ser(&e), c, unused));
        }
    }
    let tree_cue = s.windows(3).any(|w| matches!((&w[0], &w[1], &w[2]), (S::Word(a), S::Word(b), S::Word(c)) if a == "the" && c == "of" && matches!(b.as_str(), "sum" | "difference" | "product" | "quotient" | "square")));
    if tree_cue {
        if let Some((e, used0)) = ex_arith_tree(s) {
            let skipped = used0 >= 100;
            let used = used0 % 100;
            let unused = total_nums.saturating_sub(used);
            let c = 0.94 * 0.9f64.powi(unused.min(3) as i32) * if skipped { 0.4 } else { 1.0 };
            return Some((ser(&e), c, unused));
        }
    }
    // infix: choose the best parseable run
    let mut cands: Vec<(f64, usize, E, usize, usize, usize)> = Vec::new(); // (score, ops, expr, nums, start, end)
    for (a, b) in runs(s, false) {
        if let Some((e, lt, _rt)) = parse_run(&s[a..b], false) {
            let ops = count_ops(&e);
            if ops == 0 {
                continue;
            }
            let nums = n_nums(&s[a..b]);
            let score = ops as f64 + if cue_of(a + lt) { 2.0 } else { 0.0 };
            cands.push((score, ops, e, nums, a + lt, b - _rt));
        }
    }
    if cands.is_empty() {
        return None;
    }
    cands.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    let amb = cands.len() > 1 && cands[1].0 >= cands[0].0 - 1.0;
    let best = &cands[0];
    let unused = total_nums.saturating_sub(best.3);
    let mut c = 0.97 * 0.9f64.powi(unused.min(3) as i32);
    if amb {
        c *= 0.55;
    }
    // a stray unknown word sandwiched between this run and further numbers => a token (operator word) was probably not understood
    let (bs, be) = (best.4, best.5);
    let left_sand = bs >= 2 && matches!(s[bs - 1], S::Word(_)) && matches!(s[bs - 2], S::Num(_) | S::RP);
    let right_sand = be + 1 < s.len() && matches!(s[be], S::Word(_)) && (matches!(s[be + 1], S::Num(_) | S::LP) || (be + 2 < s.len() && matches!(s[be + 1], S::Word(_)) && matches!(s[be + 2], S::Num(_) | S::LP)));
    if left_sand || right_sand {
        c *= 0.5;
    }
    if best.0 < 2.0 + 0.5 && !cue_of(0) && !s.iter().any(|x| matches!(x, S::Q)) {
        c *= 0.85;
    }
    Some((ser(&best.2), c, unused))
}

// ---------------------------------------------------------------------------------------------
// csp / symbolic

fn vars_in(s: &[S]) -> Vec<char> {
    let mut v: Vec<char> = Vec::new();
    for x in s {
        if let S::Var(c) = x {
            if !v.contains(c) {
                v.push(*c);
            }
        }
    }
    v.sort_by_key(|c| match c { 'x' => 0, 'y' => 1, 'z' => 2, _ => 3 });
    v
}

fn ex_csp(s: &[S]) -> Option<(String, f64, usize)> {
    let rels: Vec<usize> = (0..s.len()).filter(|i| matches!(s[*i], S::Rel(_))).collect();
    if rels.len() < 2 {
        return None;
    }
    let vars = vars_in(s);
    if vars.len() < 2 {
        return None;
    }
    // domain: lo..hi | between lo and hi | from lo to hi | lo to hi | at least lo and at most hi (unseen; not handled)
    let mut dom: Option<(String, String)> = None;
    let mut dom_count = 0;
    let mut used_dom_nums = 0;
    for i in 0..s.len() {
        if let (Some(S::Num(a)), Some(S::DotDot), Some(S::Num(b))) = (s.get(i), s.get(i + 1), s.get(i + 2)) {
            if dom.as_ref().map_or(true, |d| d != &(a.clone(), b.clone())) {
                dom_count += 1;
            }
            dom = Some((a.clone(), b.clone()));
            used_dom_nums += 2;
        }
        if let (S::Word(w), Some(S::Num(a)), Some(S::Word(c)), Some(S::Num(b))) = (&s[i], s.get(i + 1), s.get(i + 2), s.get(i + 3)) {
            if (w == "between" && c == "and") || (w == "from" && c == "to") {
                if dom.as_ref().map_or(true, |d| d != &(a.clone(), b.clone())) {
                    dom_count += 1;
                }
                dom = Some((a.clone(), b.clone()));
                used_dom_nums += 2;
            }
        }
    }
    let (lo, hi) = dom?;
    // distinct domains found only count once (generator uses one shared domain)
    let mut cons: Vec<String> = Vec::new();
    let mut used_nums = 0usize;
    let mut bad = 0usize;
    for &ri in &rels {
        let mut a = ri;
        while a > 0 && is_mathy(&s[a - 1], true) {
            a -= 1;
        }
        let mut b = ri + 1;
        while b < s.len() && is_mathy(&s[b], true) {
            b += 1;
        }
        let lhs = &s[a..ri];
        let rhs = &s[ri + 1..b];
        // a run may swallow a domain-ish number on the left of the first token ("0 to 9"); trim leading Num before a Var-less run
        let le = parse_run(lhs, true);
        let re = parse_run(rhs, true);
        match (le, re) {
            (Some((l, _, _)), Some((r, _, _))) => {
                let op = match &s[ri] {
                    S::Rel(r) => *r,
                    _ => "==",
                };
                let op = if op == ">" { ">" } else { op };
                cons.push(format!("{}{}{}", ser(&l), op, ser(&r)));
                used_nums += n_nums(lhs) + n_nums(rhs);
            }
            _ => bad += 1,
        }
    }
    if cons.len() < 2 {
        return None;
    }
    // coefficient forms in canonical output: the tool accepts any expression syntax, so serialisation is enough
    let mut spec = String::new();
    for v in &vars {
        spec += &format!("{v} in {lo}..{hi}; ");
    }
    spec += &cons.join("; ");
    let total_nums = n_nums(s);
    let unused = total_nums.saturating_sub(used_nums + used_dom_nums.min(2));
    // clauses (split on , ; and) that mention >= 2 variables but contain no relation = a constraint we did not understand
    let mut unparsed = 0usize;
    {
        let mut seg: Vec<&S> = Vec::new();
        let mut flush = |seg: &mut Vec<&S>| {
            let nv = seg.iter().filter(|x| matches!(x, S::Var(_))).count();
            let has_rel = seg.iter().any(|x| matches!(x, S::Rel(_)));
            let dom = seg.iter().any(|x| matches!(x, S::DotDot) || matches!(x, S::Word(w) if matches!(w.as_str(), "each" | "from" | "between" | "in" | "assignment" | "integers" | "with" | "such" | "find")));
            if nv >= 2 && !has_rel && !dom {
                unparsed += 1;
            }
            seg.clear();
        };
        for x in s.iter() {
            match x {
                S::Comma | S::Semi | S::Dot | S::Q => flush(&mut seg),
                S::Word(w) if w == "and" => flush(&mut seg),
                o => seg.push(o),
            }
        }
        flush(&mut seg);
    }
    let mut c = 0.92 * 0.45f64.powi(unparsed.min(2) as i32) * 0.8f64.powi(unused.min(4) as i32) * 0.7f64.powi(bad.min(2) as i32);
    if dom_count > 1 {
        c *= 0.5;
    }
    Some((spec, c, unused))
}

fn ex_sym(s: &[S]) -> Option<(String, f64, usize)> {
    let rels: Vec<usize> = (0..s.len()).filter(|i| matches!(s[*i], S::Rel("=="))).collect();
    let others = (0..s.len()).filter(|i| matches!(s[*i], S::Rel(r) if r != "==")).count();
    if rels.len() != 1 || others > 0 {
        return None;
    }
    let ri = rels[0];
    let mut a = ri;
    while a > 0 && is_mathy(&s[a - 1], true) {
        a -= 1;
    }
    let mut b = ri + 1;
    while b < s.len() && is_mathy(&s[b], true) {
        b += 1;
    }
    let (l, _, _) = parse_run(&s[a..ri], true)?;
    let (r, _, _) = parse_run(&s[ri + 1..b], true)?;
    let has_x = |e: &E| fn_has_var(e);
    if !(has_x(&l) || has_x(&r)) {
        return None;
    }
    let used = n_nums(&s[a..ri]) + n_nums(&s[ri + 1..b]);
    let unused = n_nums(s).saturating_sub(used);
    let mut c = 0.92 * 0.9f64.powi(unused.min(3) as i32);
    // an unknown word sandwiched between the side we parsed and more math => a typo ate part of the expression
    let mathy_after = |k: usize| matches!(s.get(k), Some(S::Num(_) | S::Var(_) | S::LP | S::Op(_)));
    let tail_word = |k: usize| matches!(s.get(k), Some(S::Word(w)) if !matches!(w.as_str(), "for" | "all" | "every" | "always" | "with" | "where" | "when" | "if" | "given" | "x"));
    let right_sand = b < s.len() && tail_word(b) && (mathy_after(b + 1) || (matches!(s.get(b + 1), Some(S::Word(_))) && mathy_after(b + 2)));
    let left_sand = a >= 2 && matches!(s[a - 1], S::Word(_)) && matches!(s[a - 2], S::Num(_) | S::RP | S::Var(_));
    if right_sand || left_sand {
        c *= 0.4;
    }
    Some((format!("{} == {}", ser(&l), ser(&r)), c, unused))
}
fn fn_has_var(e: &E) -> bool {
    match e {
        E::Var(_) => true,
        E::Num(_) => false,
        E::Neg(a) => fn_has_var(a),
        E::Bin(_, a, b) => fn_has_var(a) || fn_has_var(b),
    }
}

// ---------------------------------------------------------------------------------------------
// programs and lookup

fn nums_of(s: &[S]) -> Vec<(usize, String)> {
    s.iter().enumerate().filter_map(|(i, x)| if let S::Num(n) = x { Some((i, n.clone())) } else { None }).collect()
}

fn ex_prog(s: &[S]) -> Option<(String, f64, usize)> {
    let nums = nums_of(s);
    let hasw = |w: &str| has_word(s, w);
    let has_pow = s.iter().any(|x| matches!(x, S::Op('^')));
    // a Num that is the exponent of "squared" (lexed as ^ 2) is not a data number
    let is_exp = |i: usize| i > 0 && matches!(s[i - 1], S::Op('^'));
    let data: Vec<(usize, String)> = nums.iter().filter(|(i, _)| !is_exp(*i)).cloned().collect();
    // first number following one of the cue words (skipping filler)
    let after = |cues: &[&str]| -> Option<String> {
        for (i, x) in s.iter().enumerate() {
            if let S::Word(w) = x {
                if cues.contains(&w.as_str()) {
                    for k in 1..=3 {
                        match s.get(i + k) {
                            Some(S::Num(n)) => return Some(n.clone()),
                            Some(S::Word(f)) if matches!(f.as_str(), "from" | "to" | "of" | "the" | "a" | "with" | "than" | "is" | "exceeding" | "at" | "most" | "by" | "after" | "for") => continue,
                            _ => break,
                        }
                    }
                }
            }
        }
        None
    };
    let int = |x: &str| -> Option<u64> { if x.contains('.') { None } else { x.parse::<u64>().ok() } };
    let pen = |unused: usize, base: f64| base * 0.8f64.powi(unused.min(4) as i32);
    if hasw("collatz") {
        let n = after(&["from", "starting", "for", "collatz"]).or_else(|| data.get(0).map(|x| x.1.clone()))?;
        let n = int(&n)?;
        if n < 2 {
            return None;
        }
        let unused = data.len().saturating_sub(1);
        return Some((format!("let n = {n}; let c = 0; while n != 1 {{ if n % 2 == 0 {{ n = n / 2; }} else {{ n = 3 * n + 1; }} c = c + 1; }} return c;"), pen(unused, 0.93), unused));
    }
    if (hasw("digits") || hasw("digit")) && (has_pow || hasw("square") || hasw("squared")) {
        let n = after(&["of"]).or_else(|| data.get(0).map(|x| x.1.clone()))?;
        let n = int(&n)?;
        let unused = data.len().saturating_sub(1);
        return Some((format!("let m = {n} * {n}; let s = 0; while m > 0 {{ s = s + m % 10; m = m / 10; }} return s;"), pen(unused, 0.93), unused));
    }
    if hasw("euclid") || hasw("euclidean") || hasw("gcd") {
        let mut v: Vec<u64> = data.iter().filter_map(|x| int(&x.1)).collect();
        if v.len() < 2 {
            return None;
        }
        let unused = v.len().saturating_sub(2);
        v.truncate(2);
        v.sort();
        v.reverse();
        if v[0] == v[1] || v[1] == 0 {
            return None;
        }
        let (a0, b0) = (v[0], v[1]);
        return Some((format!("let a = {a0}; let b = {b0}; let c = 0; while b != 0 {{ let t = a % b; a = b; b = t; c = c + 1; }} return c;"), pen(unused, 0.9), unused));
    }
    if hasw("primes") || hasw("prime") {
        let n = after(&["up", "to", "exceeding", "than"]).or_else(|| data.get(0).map(|x| x.1.clone()))?;
        let n = int(&n)?;
        let unused = data.len().saturating_sub(1);
        return Some((format!("let n = {n}; let c = 0; let i = 2; while i <= n {{ let p = 1; let j = 2; while j * j <= i {{ if i % j == 0 {{ p = 0; }} j = j + 1; }} c = c + p; i = i + 1; }} return c;"), pen(unused, 0.93), unused));
    }
    let modw = hasw("modulo") || hasw("mod");
    if hasw("fibonacci") && modw {
        let m = after(&["modulo", "mod"])?;
        // N: "after N steps" | "N times" | "N-th" ; else first data number that is not the modulus and not the 0/1 seeds
        let mut n = after(&["after"]);
        if n.is_none() {
            for (i, x) in s.iter().enumerate() {
                if let (S::Num(v), Some(S::Word(w))) = (x, s.get(i + 1)) {
                    if matches!(w.as_str(), "times" | "steps" | "iterations" | "th") {
                        n = Some(v.clone());
                    }
                }
            }
        }
        let n = n.or_else(|| data.iter().find(|(_, x)| *x != m && x != "0" && x != "1").map(|x| x.1.clone()))?;
        let (n, m) = (int(&n)?, int(&m)?);
        if m == 0 {
            return None;
        }
        let seeds = if hasw("starting") || (hasw("from") && data.iter().any(|(_, x)| x == "0") && data.iter().any(|(_, x)| x == "1")) { 2 } else { 0 };
        let unused = data.len().saturating_sub(2 + seeds);
        return Some((format!("let a = 0; let b = 1; let i = 0; while i < {n} {{ let t = (a + b) % {m}; a = b; b = t; i = i + 1; }} return a;"), pen(unused, 0.9), unused));
    }
    if (has_pow || hasw("squared")) && modw && (hasw("sum") || hasw("adding") || hasw("add")) {
        let m = after(&["modulo", "mod"])?;
        let n = after(&["to", "through"])?;
        let (n, m) = (int(&n)?, int(&m)?);
        if m == 0 {
            return None;
        }
        let unused = data.len().saturating_sub(3);
        return Some((format!("let s = 0; let i = 1; while i <= {n} {{ s = (s + i * i) % {m}; i = i + 1; }} return s;"), pen(unused, 0.9), unused));
    }
    None
}

fn ex_lookup(s: &[S]) -> Option<(String, f64, usize)> {
    let is_key = |w: &str, p: char| w.len() >= 2 && w.starts_with(p) && w[1..].chars().all(|c| c.is_ascii_digit());
    let mut tabs = Vec::new();
    let mut keys = Vec::new();
    for x in s {
        if let S::Word(w) = x {
            if is_key(w, 't') {
                tabs.push(w.clone());
            } else if is_key(w, 'k') {
                keys.push(w.clone());
            } else if w.len() > 2 && w.ends_with("'s") && is_key(&w[..w.len() - 2], 'k') {
                keys.push(w[..w.len() - 2].to_string());
            }
        }
    }
    if tabs.len() != 1 || keys.len() != 1 {
        return None;
    }
    let c = 0.97 * if n_nums(s) > 0 { 0.9 } else { 1.0 };
    Some((format!("{}.{}", tabs[0], keys[0]), c, n_nums(s)))
}

// ---------------------------------------------------------------------------------------------

/// a word sandwiched between two operands that is within Damerau 1 of an operator word is that operator (typo repair);
/// "at <typo of most|least>" is repaired too
fn repair_ops(s: &mut Vec<S>) {
    for i in 1..s.len().saturating_sub(1) {
        if let S::Word(wd) = &s[i] {
            let l = matches!(s[i - 1], S::Num(_) | S::RP | S::Var(_));
            let r = matches!(s[i + 1], S::Num(_) | S::LP | S::Var(_));
            if l && r && wd.len() >= 3 && !STOP.contains(&wd.as_str()) && !UNIT_WORDS.iter().any(|u| u.0 == wd.as_str()) {
                let mut c: Vec<char> = Vec::new();
                for (ow, oc) in [("plus", '+'), ("minus", '-'), ("times", '*')] {
                    if damerau(wd, ow) <= 1 {
                        c.push(oc);
                    }
                }
                if c.len() == 1 {
                    s[i] = S::Op(c[0]);
                }
            }
        }
    }
    for i in 1..s.len() {
        if matches!(&s[i - 1], S::Word(a) if a == "at") {
            let fix: Option<String> = if let S::Word(wd) = &s[i] {
                ["most", "least"].iter().find(|ow| wd != **ow && damerau(wd, ow) <= 1).map(|o| o.to_string())
            } else {
                None
            };
            if let Some(f) = fix {
                s[i] = S::Word(f);
            }
        }
    }
    // re-run the relation phrase mapping that lex2 could not do on the typo: "at most|least" -> Rel
    let mut i = 0;
    while i + 1 < s.len() {
        if matches!(&s[i], S::Word(a) if a == "at") {
            if matches!(&s[i + 1], S::Word(b) if b == "most") {
                s.splice(i..i + 2, [S::Rel("<=")]);
                continue;
            }
            if matches!(&s[i + 1], S::Word(b) if b == "least") {
                s.splice(i..i + 2, [S::Rel(">=")]);
                continue;
            }
        }
        i += 1;
    }
}

pub fn extract(text: &str) -> Ex {
    let mut fuzzy = 0usize;
    let toks = tokenize(text);
    let toks = fold_numbers(toks, &mut fuzzy);
    let mut s = lex2(&toks, &mut fuzzy);
    repair_ops(&mut s);
    let mut cands: Vec<(Kind, String, f64, usize)> = Vec::new();
    if let Some((sp, c, u)) = ex_lookup(&s) {
        cands.push((Kind::Lookup, sp, c, u));
    }
    if let Some((sp, c, u)) = ex_prog(&s) {
        cands.push((Kind::Prog, sp, c, u));
    }
    if let Some((sp, c, u)) = ex_csp(&s) {
        cands.push((Kind::Csp, sp, c, u));
    }
    if let Some((sp, c, u)) = ex_sym(&s) {
        cands.push((Kind::Sym, sp, c, u));
    }
    if let Some((sp, c, f, u)) = ex_units(&s) {
        cands.push((Kind::Units, sp, c, u));
        fuzzy += f;
    }
    if let Some((sp, c, u)) = ex_arith(&s) {
        cands.push((Kind::Arith, sp, c, u));
    }
    let discuss = has_any(&s, &["explain", "discuss", "summarize", "summarise", "describe", "why"]);
    if discuss {
        for c in cands.iter_mut() {
            c.2 *= 0.35;
        }
    }
    // arithmetic is the generic fallback: a structured template match (prog, csp, units, ...) outranks it
    let structured = cands.iter().any(|c| c.0 != Kind::Arith && c.2 >= 0.15);
    let rank = |c: &(Kind, String, f64, usize)| if c.0 == Kind::Arith { if structured { c.2 * 0.1 } else { c.2 } } else { c.2 };
    cands.sort_by(|a, b| rank(b).partial_cmp(&rank(a)).unwrap());
    let n_cands = cands.iter().filter(|c| c.2 > 0.3).count();
    match cands.first() {
        None => Ex { spec: None, conf: 0.0, kind: None, n_cands: 0, fuzzy, unused: 0 },
        Some(best) => {
            let mut c = best.2 * 0.92f64.powi(fuzzy.min(4) as i32);
            if cands.len() > 1 && cands[1].0 != Kind::Arith && cands[1].2 > 0.7 * best.2 {
                c *= 0.7;
            }
            Ex { spec: Some(best.1.clone()), conf: c.clamp(0.0, 1.0), kind: Some(best.0), n_cands, fuzzy, unused: best.3 }
        }
    }
}
