//! g21: an INDEPENDENTLY written second evaluator path + verifier. Nothing here calls `neural_mesh::tools` evaluator
//! code or `evalx`, except the optional metamorphic checks, which deliberately run the tool UNDER TEST on transformed
//! inputs. Differences from the shared path (so a single systematic bug is unlikely to be common to both):
//!  * expressions: shunting-yard -> RPN (not recursive descent), own i128 fraction type, own decimal-literal parser,
//!    AND outward-rounded f64 interval arithmetic that the claimed value must lie in;
//!  * units: its own factor table written as DECIMAL STRINGS from the physical definitions (m = 1, ft = 0.3048, lb = 0.45359237,
//!    knot = 1852/3600 ...), own dimension vectors, own unit-expression parser;
//!  * csp / identity: own constraint parser, certificate check / brute-force re-solve / grid check on the independent evaluator;
//!  * programs: native re-implementation of the six program templates (a different algorithm from the interpreter);
//!  * metamorphic (optional): the independent parse is re-printed fully parenthesised, and with commuted operands of
//!    + and *, and sent to the tool under test; all must equal the claim.
use super::evalx::{mutant, MCalc, MVerifier};
use neural_mesh::tools::{LookupTool, Sandbox, Tool, ToolError};

type R<T> = Result<T, ToolError>;
fn bad<T>(s: &str) -> R<T> {
    Err(ToolError::Parse(s.to_string()))
}
fn unsupported<T>() -> R<T> {
    Err(ToolError::Unknown("unsupported by independent evaluator".into()))
}

// ---------------------------------------------------------------------------------------------
// fraction

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fr {
    n: i128,
    d: i128,
}
fn g(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    if a == 0 { 1 } else { a }
}
impl Fr {
    fn mk(n: i128, d: i128) -> R<Fr> {
        if d == 0 {
            return Err(ToolError::DivZero);
        }
        let (n, d) = if d < 0 { (n.checked_neg().ok_or(ToolError::Overflow)?, d.checked_neg().ok_or(ToolError::Overflow)?) } else { (n, d) };
        let k = g(n, d);
        Ok(Fr { n: n / k, d: d / k })
    }
    pub fn int(n: i128) -> Fr {
        Fr { n, d: 1 }
    }
    fn add(self, o: Fr) -> R<Fr> {
        let a = self.n.checked_mul(o.d).ok_or(ToolError::Overflow)?;
        let b = o.n.checked_mul(self.d).ok_or(ToolError::Overflow)?;
        Fr::mk(a.checked_add(b).ok_or(ToolError::Overflow)?, self.d.checked_mul(o.d).ok_or(ToolError::Overflow)?)
    }
    fn neg(self) -> Fr {
        Fr { n: -self.n, d: self.d }
    }
    fn sub(self, o: Fr) -> R<Fr> {
        self.add(o.neg())
    }
    fn mul(self, o: Fr) -> R<Fr> {
        Fr::mk(self.n.checked_mul(o.n).ok_or(ToolError::Overflow)?, self.d.checked_mul(o.d).ok_or(ToolError::Overflow)?)
    }
    fn div(self, o: Fr) -> R<Fr> {
        if o.n == 0 {
            return Err(ToolError::DivZero);
        }
        Fr::mk(self.n.checked_mul(o.d).ok_or(ToolError::Overflow)?, self.d.checked_mul(o.n).ok_or(ToolError::Overflow)?)
    }
    fn powi(self, e: i128) -> R<Fr> {
        if e.unsigned_abs() > 10_000 {
            return Err(ToolError::Domain("exponent".into()));
        }
        let mut acc = Fr::int(1);
        for _ in 0..e.unsigned_abs() {
            acc = acc.mul(self)?;
        }
        if e < 0 { Fr::int(1).div(acc) } else { Ok(acc) }
    }
    fn is_int(&self) -> bool {
        self.d == 1
    }
    fn f(&self) -> f64 {
        self.n as f64 / self.d as f64
    }
}
fn fr_cmp(a: Fr, b: Fr) -> Option<std::cmp::Ordering> {
    Some(a.n.checked_mul(b.d)?.cmp(&b.n.checked_mul(a.d)?))
}

/// decimal / fraction literal: "12", "-3.25", "7/2", "0.33"
pub fn fr_parse(s: &str) -> R<Fr> {
    let s = s.trim();
    if s.is_empty() || s.len() > 80 {
        return bad("empty");
    }
    if let Some((a, b)) = s.split_once('/') {
        return Fr::mk(a.trim().parse::<i128>().map_err(|_| ToolError::Parse("num".into()))?, b.trim().parse::<i128>().map_err(|_| ToolError::Parse("den".into()))?);
    }
    let (neg, body) = if let Some(r) = s.strip_prefix('-') { (true, r) } else { (false, s) };
    let (ip, fp) = body.split_once('.').unwrap_or((body, ""));
    if ip.is_empty() || !ip.chars().all(|c| c.is_ascii_digit()) || !fp.chars().all(|c| c.is_ascii_digit()) || fp.len() > 30 {
        return bad("literal");
    }
    let mut n: i128 = 0;
    for c in ip.chars().chain(fp.chars()) {
        n = n.checked_mul(10).and_then(|x| x.checked_add(c.to_digit(10).unwrap() as i128)).ok_or(ToolError::Overflow)?;
    }
    let mut d: i128 = 1;
    for _ in 0..fp.len() {
        d = d.checked_mul(10).ok_or(ToolError::Overflow)?;
    }
    Fr::mk(if neg { -n } else { n }, d)
}

// ---------------------------------------------------------------------------------------------
// interval arithmetic (outward rounding by one ulp per op)

#[derive(Clone, Copy, Debug)]
pub struct Iv {
    lo: f64,
    hi: f64,
}
impl Iv {
    fn pt(x: f64) -> Iv {
        Iv { lo: x.next_down(), hi: x.next_up() }
    }
    fn norm(lo: f64, hi: f64) -> Iv {
        Iv { lo: lo.next_down(), hi: hi.next_up() }
    }
    fn add(a: Iv, b: Iv) -> Iv {
        Iv::norm(a.lo + b.lo, a.hi + b.hi)
    }
    fn sub(a: Iv, b: Iv) -> Iv {
        Iv::norm(a.lo - b.hi, a.hi - b.lo)
    }
    fn mul(a: Iv, b: Iv) -> Iv {
        let c = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
        Iv::norm(c.iter().cloned().fold(f64::INFINITY, f64::min), c.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    }
    fn div(a: Iv, b: Iv) -> Option<Iv> {
        if b.lo <= 0.0 && b.hi >= 0.0 {
            return None;
        }
        let c = [a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi];
        Some(Iv::norm(c.iter().cloned().fold(f64::INFINITY, f64::min), c.iter().cloned().fold(f64::NEG_INFINITY, f64::max)))
    }
    fn neg(a: Iv) -> Iv {
        Iv { lo: -a.hi, hi: -a.lo }
    }
    fn contains(&self, x: f64) -> bool {
        x >= self.lo && x <= self.hi
    }
}

// ---------------------------------------------------------------------------------------------
// shunting-yard -> RPN

#[derive(Clone, Debug)]
pub enum Item {
    Num(Fr, Iv),
    Var(String),
    Neg,
    Op(char), // + - * / ^ %
}

fn prec(c: char) -> u8 {
    match c {
        '+' | '-' => 1,
        '*' | '/' | '%' => 2,
        'n' => 3, // unary minus
        '^' => 4,
        _ => 0,
    }
}

pub fn to_rpn(s: &str) -> R<Vec<Item>> {
    let cs: Vec<char> = s.chars().collect();
    if cs.len() > 2000 {
        return Err(ToolError::TooLong);
    }
    let mut out: Vec<Item> = Vec::new();
    let mut st: Vec<char> = Vec::new(); // operators, '(' and 'n'
    let mut prev_operand = false; // previous token ends an operand
    let mut i = 0;
    let mut depth = 0usize;
    while i < cs.len() {
        let c = cs[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() {
            let st0 = i;
            while i < cs.len() && (cs[i].is_ascii_digit() || cs[i] == '.') {
                i += 1;
            }
            if i < cs.len() && (cs[i].is_alphabetic() || cs[i] == '_') {
                return bad("malformed number");
            }
            let lit: String = cs[st0..i].iter().collect();
            if lit.ends_with('.') || lit.matches('.').count() > 1 {
                return bad("literal");
            }
            let f = fr_parse(&lit)?;
            out.push(Item::Num(f, Iv::pt(f.f())));
            prev_operand = true;
            continue;
        }
        if c.is_alphabetic() || c == '_' {
            let st0 = i;
            while i < cs.len() && (cs[i].is_alphanumeric() || cs[i] == '_') {
                i += 1;
            }
            if i < cs.len() && cs[i] == '(' {
                return unsupported();
            }
            out.push(Item::Var(cs[st0..i].iter().collect()));
            prev_operand = true;
            continue;
        }
        match c {
            '(' => {
                st.push('(');
                depth += 1;
                if depth > 64 {
                    return Err(ToolError::TooDeep);
                }
                prev_operand = false;
            }
            ')' => {
                loop {
                    match st.pop() {
                        Some('(') => break,
                        Some(o) => out.push(if o == 'n' { Item::Neg } else { Item::Op(o) }),
                        None => return bad("unbalanced"),
                    }
                }
                depth = depth.saturating_sub(1);
                prev_operand = true;
            }
            '+' | '-' | '*' | '/' | '%' | '^' => {
                if !prev_operand {
                    // prefix
                    match c {
                        '-' => st.push('n'),
                        '+' => {}
                        _ => return bad("operator without operand"),
                    }
                } else {
                    let p = prec(c);
                    let right = c == '^' || (mutant() == 21 && (c == '+' || c == '-'));
                    while let Some(&top) = st.last() {
                        if top == '(' {
                            break;
                        }
                        let tp = prec(top);
                        if tp > p || (tp == p && !right) {
                            st.pop();
                            out.push(if top == 'n' { Item::Neg } else { Item::Op(top) });
                        } else {
                            break;
                        }
                    }
                    st.push(c);
                    prev_operand = false;
                }
            }
            _ => return bad("character"),
        }
        i += 1;
    }
    while let Some(o) = st.pop() {
        if o == '(' {
            return bad("unbalanced");
        }
        out.push(if o == 'n' { Item::Neg } else { Item::Op(o) });
    }
    if out.is_empty() || !prev_operand {
        return bad("empty or dangling");
    }
    Ok(out)
}

pub fn eval_rpn(rpn: &[Item], env: &[(String, i128)]) -> R<(Fr, Iv)> {
    let mut st: Vec<(Fr, Iv)> = Vec::new();
    for it in rpn {
        match it {
            Item::Num(f, iv) => st.push((*f, *iv)),
            Item::Var(v) => {
                let val = env.iter().find(|(n, _)| n == v).ok_or_else(|| ToolError::Unknown(format!("variable {v}")))?.1;
                st.push((Fr::int(val), Iv::pt(val as f64)));
            }
            Item::Neg => {
                let (a, ia) = st.pop().ok_or(ToolError::Parse("stack".into()))?;
                st.push((a.neg(), Iv::neg(ia)));
            }
            Item::Op(o) => {
                let (b, ib) = st.pop().ok_or(ToolError::Parse("stack".into()))?;
                let (a, ia) = st.pop().ok_or(ToolError::Parse("stack".into()))?;
                let r = match o {
                    '+' => (a.add(b)?, Iv::add(ia, ib)),
                    '-' => (a.sub(b)?, Iv::sub(ia, ib)),
                    '*' => (a.mul(b)?, Iv::mul(ia, ib)),
                    '/' => (a.div(b)?, Iv::div(ia, ib).ok_or(ToolError::DivZero)?),
                    '%' => {
                        if !a.is_int() || !b.is_int() {
                            return Err(ToolError::Domain("% needs integers".into()));
                        }
                        if b.n == 0 {
                            return Err(ToolError::DivZero);
                        }
                        let m = Fr::int(a.n.rem_euclid(b.n));
                        (m, Iv::pt(m.f()))
                    }
                    '^' => {
                        if !b.is_int() {
                            return Err(ToolError::Domain("exponent".into()));
                        }
                        let e = b.n;
                        let exact = a.powi(e)?;
                        // interval power by repeated multiplication (exact enough for small exponents)
                        let mut iv = Iv::pt(1.0);
                        for _ in 0..e.unsigned_abs().min(64) {
                            iv = Iv::mul(iv, ia);
                        }
                        if e < 0 {
                            iv = Iv::div(Iv::pt(1.0), iv).ok_or(ToolError::DivZero)?;
                        }
                        if e.unsigned_abs() > 64 {
                            iv = Iv { lo: f64::NEG_INFINITY, hi: f64::INFINITY };
                        }
                        (exact, iv)
                    }
                    _ => return bad("op"),
                };
                st.push(r);
            }
        }
    }
    if st.len() == 1 { Ok(st.pop().unwrap()) } else { bad("stack") }
}

/// re-print an RPN fully parenthesised; `commute` swaps operands of + and * (alternating) -- must not change the value
pub fn rpn_to_text(rpn: &[Item], commute: bool) -> R<String> {
    let mut st: Vec<String> = Vec::new();
    let mut k = 0usize;
    for it in rpn {
        match it {
            Item::Num(f, _) => st.push(if f.d == 1 { f.n.to_string() } else { format!("({}/{})", f.n, f.d) }),
            Item::Var(v) => st.push(v.clone()),
            Item::Neg => {
                let a = st.pop().ok_or(ToolError::Parse("stack".into()))?;
                st.push(format!("(-{a})"));
            }
            Item::Op(o) => {
                let b = st.pop().ok_or(ToolError::Parse("stack".into()))?;
                let a = st.pop().ok_or(ToolError::Parse("stack".into()))?;
                let swap = commute && (*o == '+' || *o == '*') && {
                    k += 1;
                    k % 2 == 1
                };
                st.push(if swap { format!("({b}{o}{a})") } else { format!("({a}{o}{b})") });
            }
        }
    }
    if st.len() == 1 { Ok(st.pop().unwrap()) } else { bad("stack") }
}

// ---------------------------------------------------------------------------------------------
// units (independent table: decimal strings, own dimension vectors)

// symbol, factor to SI (decimal or p/q string), dims (L, M, T)
const IU: &[(&str, &str, [i8; 3])] = &[
    ("m", "1", [1, 0, 0]),
    ("km", "1000", [1, 0, 0]),
    ("cm", "0.01", [1, 0, 0]),
    ("mm", "0.001", [1, 0, 0]),
    ("in", "0.0254", [1, 0, 0]),
    ("ft", "0.3048", [1, 0, 0]),
    ("yd", "0.9144", [1, 0, 0]),
    ("mile", "1609.344", [1, 0, 0]),
    ("kg", "1", [0, 1, 0]),
    ("g", "0.001", [0, 1, 0]),
    ("mg", "0.000001", [0, 1, 0]),
    ("t", "1000", [0, 1, 0]),
    ("lb", "0.45359237", [0, 1, 0]), // (mutant 20 swaps the last two digits at lookup time)
    ("s", "1", [0, 0, 1]),
    ("ms", "0.001", [0, 0, 1]),
    ("min", "60", [0, 0, 1]),
    ("h", "3600", [0, 0, 1]),
    ("day", "86400", [0, 0, 1]),
    ("week", "604800", [0, 0, 1]),
    ("L", "0.001", [3, 0, 0]),
    ("mL", "0.000001", [3, 0, 0]),
    ("knot", "1852/3600", [1, 0, -1]),
];

fn unit_expr(s: &str) -> R<(Fr, [i8; 3])> {
    let s = s.trim();
    if s.is_empty() {
        return bad("unit");
    }
    let mut f = Fr::int(1);
    let mut dim = [0i8; 3];
    // tokens split by / and * keeping the operator that precedes them
    let mut parts: Vec<(bool, String)> = Vec::new(); // (is_divisor, text)
    let mut cur = String::new();
    let mut div = false;
    for c in s.chars() {
        if c == '/' || c == '*' {
            parts.push((div, cur.trim().to_string()));
            cur.clear();
            div = c == '/';
        } else {
            cur.push(c);
        }
    }
    parts.push((div, cur.trim().to_string()));
    for (is_div, p) in parts {
        if p.is_empty() {
            return bad("unit term");
        }
        if p == "1" {
            continue;
        }
        let (name, e) = match p.split_once('^') {
            Some((n, e)) => (n.trim().to_string(), e.trim().parse::<i32>().map_err(|_| ToolError::Parse("exp".into()))?),
            None => (p.clone(), 1),
        };
        let e = if is_div { -e } else { e };
        let ent = IU.iter().find(|u| u.0 == name).ok_or_else(|| ToolError::Unknown(format!("unit {name}")))?;
        let uf = if mutant() == 20 && name == "lb" { fr_parse("0.45359273")? } else { fr_parse(ent.1)? };
        f = f.mul(uf.powi(e as i128)?)?;
        for k in 0..3 {
            dim[k] += ent.2[k] * e as i8;
        }
    }
    Ok((f, dim))
}

fn unit_value(spec: &str) -> R<Fr> {
    let spec = spec.trim();
    let (cmd, rest) = spec.split_once(char::is_whitespace).ok_or(ToolError::Parse("cmd".into()))?;
    let (lhs, to) = rest.split_once("->").ok_or(ToolError::Parse("->".into()))?;
    let (tf, td) = unit_expr(to)?;
    let mut total = Fr::int(0);
    let terms: Vec<&str> = match cmd {
        "conv" => vec![lhs],
        "sum" => lhs.split('+').collect(),
        _ => return unsupported(),
    };
    for t in terms {
        let (v, u) = t.trim().split_once(char::is_whitespace).ok_or(ToolError::Parse("quantity".into()))?;
        let (f, d) = unit_expr(u)?;
        if d != td {
            return Err(ToolError::Dimension("mismatch".into()));
        }
        total = total.add(fr_parse(v)?.mul(f)?)?;
    }
    total.div(tf)
}

// ---------------------------------------------------------------------------------------------
// csp / identity

fn split_cmp(c: &str) -> R<(&str, &str, &str)> {
    for op in ["==", "<=", ">=", "!="] {
        if let Some(p) = c.find(op) {
            return Ok((&c[..p], op, &c[p + 2..]));
        }
    }
    for op in ["<", ">"] {
        if let Some(p) = c.find(op) {
            return Ok((&c[..p], op, &c[p + 1..]));
        }
    }
    bad("comparator")
}

struct Cons {
    l: Vec<Item>,
    op: String,
    r: Vec<Item>,
}
struct Prob {
    vars: Vec<(String, i128, i128)>,
    cons: Vec<Cons>,
}
fn prob(spec: &str) -> R<Prob> {
    let mut vars = Vec::new();
    let mut cons = Vec::new();
    for p in spec.split(';') {
        let p = p.trim();
        if p.is_empty() {
            continue;
        }
        if let Some((n, rg)) = p.split_once(" in ") {
            let (lo, hi) = rg.split_once("..").ok_or(ToolError::Parse("range".into()))?;
            vars.push((n.trim().to_string(), lo.trim().parse::<i128>().map_err(|_| ToolError::Parse("lo".into()))?, hi.trim().parse::<i128>().map_err(|_| ToolError::Parse("hi".into()))?));
        } else {
            let (l, op, r) = split_cmp(p)?;
            cons.push(Cons { l: to_rpn(l)?, op: op.to_string(), r: to_rpn(r)? });
        }
    }
    if vars.is_empty() {
        return bad("vars");
    }
    Ok(Prob { vars, cons })
}
fn holds(c: &Cons, env: &[(String, i128)]) -> R<bool> {
    let a = eval_rpn(&c.l, env);
    let b = eval_rpn(&c.r, env);
    let (a, b) = match (a, b) {
        (Ok(a), Ok(b)) => (a.0, b.0),
        (Err(ToolError::DivZero), _) | (_, Err(ToolError::DivZero)) => return Ok(false),
        (Err(e), _) | (_, Err(e)) => return Err(e),
    };
    let o = fr_cmp(a, b).ok_or(ToolError::Overflow)?;
    use std::cmp::Ordering::*;
    Ok(match c.op.as_str() {
        "==" => o == Equal,
        "!=" => o != Equal,
        "<" => o == Less,
        "<=" => o != Greater,
        ">" => o == Greater,
        ">=" => o != Less,
        _ => false,
    })
}

fn csp_check(spec: &str, claim: &str) -> R<bool> {
    let p = prob(spec)?;
    let mut env: Vec<(String, i128)> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for part in claim.split(',') {
        let (n, v) = part.split_once('=').ok_or(ToolError::Parse("assignment".into()))?;
        let v = fr_parse(v)?;
        if !v.is_int() || seen.contains(&n.trim().to_string()) {
            return Ok(false);
        }
        seen.push(n.trim().to_string());
        env.push((n.trim().to_string(), v.n));
    }
    if env.len() != p.vars.len() {
        return Ok(false);
    }
    for (n, lo, hi) in &p.vars {
        match env.iter().find(|(m, _)| m == n) {
            Some((_, v)) if v >= lo && v <= hi => {}
            _ => return Ok(false),
        }
    }
    for c in &p.cons {
        if !holds(c, &env)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// brute force: does ANY assignment satisfy the spec? (None = too large)
fn csp_exists(spec: &str) -> R<Option<bool>> {
    let p = prob(spec)?;
    let total: u128 = p.vars.iter().map(|(_, lo, hi)| (hi - lo + 1).max(0) as u128).product();
    if total > 400_000 {
        return Ok(None);
    }
    let mut idx: Vec<i128> = p.vars.iter().map(|v| v.1).collect();
    loop {
        let env: Vec<(String, i128)> = p.vars.iter().zip(&idx).map(|(v, x)| (v.0.clone(), *x)).collect();
        let mut ok = true;
        for c in &p.cons {
            if !holds(c, &env)? {
                ok = false;
                break;
            }
        }
        if ok {
            return Ok(Some(true));
        }
        let mut k = p.vars.len();
        loop {
            if k == 0 {
                return Ok(Some(false));
            }
            k -= 1;
            idx[k] += 1;
            if idx[k] <= p.vars[k].2 {
                break;
            }
            idx[k] = p.vars[k].1;
        }
    }
}

fn identity_value(spec: &str) -> R<bool> {
    let (l, r) = spec.split_once("==").ok_or(ToolError::Parse("==".into()))?;
    let (lr, rr) = (to_rpn(l)?, to_rpn(r)?);
    let mut any_diff = false;
    let mut valid = 0;
    for x in -15i128..=16 {
        let env = vec![("x".to_string(), x)];
        match (eval_rpn(&lr, &env), eval_rpn(&rr, &env)) {
            (Ok(a), Ok(b)) => {
                valid += 1;
                if a.0 != b.0 {
                    any_diff = true;
                }
            }
            (Err(ToolError::DivZero), Err(ToolError::DivZero)) => {}
            (Err(ToolError::DivZero), Ok(_)) | (Ok(_), Err(ToolError::DivZero)) => any_diff = true,
            (Err(e), _) | (_, Err(e)) => return Err(e),
        }
    }
    if valid < 3 {
        return Err(ToolError::Domain("too few points".into()));
    }
    Ok(!any_diff)
}

// ---------------------------------------------------------------------------------------------
// programs: native templates

fn ints(s: &str) -> Vec<u64> {
    let mut v = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() {
            cur.push(c);
        } else if !cur.is_empty() {
            v.push(cur.parse().unwrap_or(0));
            cur.clear();
        }
    }
    if !cur.is_empty() {
        v.push(cur.parse().unwrap_or(0));
    }
    v
}

/// returns (value, approximate op count) when `src` is one of the six templates, None otherwise
fn prog_native(src: &str) -> Option<(u128, u64)> {
    let n = ints(src);
    if src.starts_with("let n = ") && src.contains("3 * n + 1") {
        let (mut x, mut c, mut ops) = (n[0] as u128, 0u128, 0u64);
        while x != 1 {
            x = if x % 2 == 0 { x / 2 } else { 3 * x + 1 };
            c += 1;
            ops += 6;
            if ops > 2_000_000 {
                return None;
            }
        }
        return Some((c, ops));
    }
    if src.starts_with("let m = ") && src.contains("s = s + m % 10") {
        let mut m = n[0] as u128 * n[0] as u128;
        let (mut s, mut ops) = (0u128, 0u64);
        while m > 0 {
            s += m % 10;
            m /= 10;
            ops += 5;
        }
        return Some((s, ops));
    }
    if src.starts_with("let a = ") && src.contains("let t = a % b") {
        let (mut a, mut b, mut c, mut ops) = (n[0], n[1], 0u128, 0u64);
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
            c += 1;
            ops += 6;
        }
        return Some((c, ops));
    }
    if src.starts_with("let n = ") && src.contains("while j * j <= i") {
        let lim = n[0] as usize;
        let mut sieve = vec![true; lim + 1];
        let mut count = 0u128;
        let mut ops = 0u64;
        for i in 2..=lim {
            if sieve[i] {
                count += 1;
                let mut j = i * i;
                while j <= lim {
                    sieve[j] = false;
                    j += i;
                    ops += 1;
                }
            }
            ops += 4;
        }
        // charge like the interpreter's trial division (different algorithm, comparable work estimate)
        return Some((count, ops.max(lim as u64 * 12)));
    }
    if src.starts_with("let s = 0; let i = 1;") && src.contains("s = (s + i * i) %") {
        // numbers: 0,1,N,M,1
        let (nn, m) = (n[2] as u128, n[3] as u128);
        let mut s = 0u128;
        for i in 1..=nn {
            s = (s + i * i) % m;
        }
        return Some((s, nn as u64 * 9));
    }
    if src.starts_with("let a = 0; let b = 1; let i = 0;") && src.contains("let t = (a + b) %") {
        let (nn, m) = (n[3], n[4] as u128);
        let (mut a, mut b) = (0u128, 1u128);
        for _ in 0..nn {
            let t = (a + b) % m;
            a = b;
            b = t;
        }
        return Some((a, nn * 9));
    }
    None
}

// ---------------------------------------------------------------------------------------------
// the verifier

#[derive(Clone, Debug)]
pub struct IVerifier {
    lookup: LookupTool,
    shared: MVerifier,
    /// run metamorphic checks (parenthesised + commuted re-print through the tool under test)
    pub meta: bool,
    /// metamorphic checks ONLY (no independent evaluator): calc re-prints through the tool under test, units decomposes sums
    pub meta_only: bool,
}
impl IVerifier {
    pub fn new(lookup: LookupTool, meta: bool) -> Self {
        IVerifier { shared: MVerifier::new(lookup.clone()), lookup, meta, meta_only: false }
    }
    pub fn meta_only(lookup: LookupTool) -> Self {
        IVerifier { shared: MVerifier::new(lookup.clone()), lookup, meta: true, meta_only: true }
    }
    fn v(ok: bool, why: &str, cost: u32) -> (R<String>, u32) {
        (Ok(if ok { "accept".to_string() } else { format!("reject: {why}") }), cost)
    }
    /// (verdict, cost, used_independent_path)
    pub fn verify(&self, tool: &str, spec: &str, claim: &str) -> (R<String>, u32, bool) {
        let claim = claim.trim();
        let r = self.verify_inner(tool, spec, claim);
        match r {
            Err(ToolError::Unknown(ref m)) if m.starts_with("unsupported") => {
                // no independent coverage: the shared verifier is the fallback
                let (res, c) = match self.shared.verify(tool, spec, claim) {
                    Ok((s, c)) => (Ok(s), c),
                    Err(e) => (Err(e), 1),
                };
                (res, c, false)
            }
            Err(e) => (Err(e), 1, true),
            Ok((res, c)) => (res, c, true),
        }
    }
    fn verify_inner(&self, tool: &str, spec: &str, claim: &str) -> R<(R<String>, u32)> {
        if self.meta_only {
            return self.meta_only_inner(tool, spec, claim);
        }
        match tool {
            "calc" => {
                let rpn = to_rpn(spec)?;
                let (want, iv) = eval_rpn(&rpn, &[])?;
                let got = match fr_parse(claim) {
                    Ok(g) => g,
                    Err(_) => return Ok(IVerifier::v(false, "claim is not a number", 1)),
                };
                if want != got {
                    return Ok(IVerifier::v(false, "independent exact recompute differs", 1));
                }
                if !iv.contains(got.f()) {
                    return Ok(IVerifier::v(false, "interval cross-check differs", 1));
                }
                let mut cost = 1;
                if self.meta {
                    for commute in [false, true] {
                        let txt = rpn_to_text(&rpn, commute)?;
                        cost += 1;
                        match MCalc.call(&txt) {
                            Ok(out) => match fr_parse(&out) {
                                Ok(o) if o == got => {}
                                _ => return Ok(IVerifier::v(false, "metamorphic re-print differs", cost)),
                            },
                            Err(_) => return Ok(IVerifier::v(false, "metamorphic re-print errors", cost)),
                        }
                    }
                }
                Ok(IVerifier::v(true, "", cost))
            }
            "units" => {
                let want = unit_value(spec)?;
                let got = match fr_parse(claim) {
                    Ok(g) => g,
                    Err(_) => return Ok(IVerifier::v(false, "claim is not a number", 1)),
                };
                Ok(IVerifier::v(want == got, "independent unit recompute differs", 1))
            }
            "csp" => {
                if claim == "unsat" {
                    return match csp_exists(spec)? {
                        Some(ex) => Ok(IVerifier::v(!ex, "a solution exists", 3)),
                        None => unsupported(),
                    };
                }
                match csp_check(spec, claim) {
                    Ok(b) => Ok(IVerifier::v(b, "assignment violates the spec", 1)),
                    Err(ToolError::Parse(_)) => Ok(IVerifier::v(false, "claim is not a valid assignment", 1)),
                    Err(e) => Err(e),
                }
            }
            "identity" => {
                let same = identity_value(spec)?;
                if claim == "identical" {
                    Ok(IVerifier::v(same, "found a differing point", 2))
                } else if claim.starts_with("differs") {
                    Ok(IVerifier::v(!same, "no differing point exists on the grid", 2))
                } else {
                    Ok(IVerifier::v(false, "unrecognised claim", 1))
                }
            }
            "sandbox" => match prog_native(spec) {
                Some((val, ops)) => {
                    // cost accounting is conservative: charged like the interpreter re-run (the native path is really cheaper)
                    let steps = Sandbox::default().run(spec).map(|r| r.1).unwrap_or(ops);
                    Ok(IVerifier::v(val.to_string() == claim, "native re-run differs", 1 + (steps / 500).min(100) as u32))
                }
                None => unsupported(),
            },
            "lookup" => {
                let want = self.lookup.call(spec)?;
                Ok(IVerifier::v(want == claim, "table value differs", 1))
            }
            other => Err(ToolError::Unknown(format!("verifiable tool {other}"))),
        }
    }
}
impl IVerifier {
    fn meta_only_inner(&self, tool: &str, spec: &str, claim: &str) -> R<(R<String>, u32)> {
        match tool {
            "calc" => {
                let rpn = to_rpn(spec)?;
                let got = match fr_parse(claim) {
                    Ok(g) => g,
                    Err(_) => return Ok(IVerifier::v(false, "claim is not a number", 1)),
                };
                let mut cost = 0;
                for commute in [false, true] {
                    let txt = rpn_to_text(&rpn, commute)?;
                    cost += 1;
                    match MCalc.call(&txt) {
                        Ok(out) => match fr_parse(&out) {
                            Ok(o) if o == got => {}
                            _ => return Ok(IVerifier::v(false, "metamorphic re-print differs", cost)),
                        },
                        Err(_) => return Ok(IVerifier::v(false, "metamorphic re-print errors", cost)),
                    }
                }
                Ok(IVerifier::v(true, "", cost))
            }
            "units" => {
                // a sum must equal the sum of its separately converted terms (all through the tool under test)
                let sp = spec.trim();
                if let Some(rest) = sp.strip_prefix("sum ") {
                    if let Some((lhs, to)) = rest.split_once("->") {
                        let got = fr_parse(claim)?;
                        let mut total = Fr::int(0);
                        let mut cost = 0;
                        for term in lhs.split('+') {
                            cost += 1;
                            let out = neural_mesh::tools::Tool::call(&super::evalx::MUnits, &format!("conv {} -> {}", term.trim(), to.trim()))?;
                            total = total.add(fr_parse(&out)?)?;
                        }
                        return Ok(IVerifier::v(total == got, "decomposed sum differs", cost));
                    }
                }
                unsupported_accept()
            }
            _ => unsupported_accept(),
        }
    }
}

fn unsupported_accept() -> R<(R<String>, u32)> {
    Ok((Ok("accept".to_string()), 0))
}

impl Tool for IVerifier {
    fn name(&self) -> &str {
        "verify"
    }
    fn cost(&self) -> u32 {
        1
    }
    fn call(&self, input: &str) -> R<String> {
        Ok(self.call_metered(input).0?)
    }
    fn call_metered(&self, input: &str) -> (R<String>, u32) {
        let mut it = input.splitn(3, '\n');
        let (t, s, c) = match (it.next(), it.next(), it.next()) {
            (Some(t), Some(s), Some(c)) => (t, s, c),
            _ => return (Err(ToolError::BadInput("verify input is 3 lines: tool, spec, claim".into())), 1),
        };
        let (r, c, _) = self.verify(t.trim(), s, c);
        (r, c)
    }
}
