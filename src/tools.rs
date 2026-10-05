//! Deterministic LOCAL tools (W14, R&D round 3, TOOLS track). std-only, no network, no I/O, no process spawning.
//!
//! * [`Tool`]: name, nominal cost, cacheable flag, deterministic `call(input) -> Result<output, ToolError>`.
//! * [`Registry`]: name -> tool, optional result cache (hit/miss/cost-saved accounting).
//! * Tools: [`Calc`] (exact rational/integer arithmetic, precedence parser, checked overflow), [`Units`] (unit
//!   conversion + dimension check), [`Csp`] (small finite-domain constraint solver / certificate checker),
//!   [`Identity`] (grid identity check for small symbolic claims), [`Sandbox`] (own tiny integer/list
//!   interpreter: bounded steps / memory / depth, no I/O by construction), [`LookupTool`] (key/value tables),
//!   [`Verifier`] (re-computes a claimed result with an independent check path and answers accept / reject).
//! * [`ToolSelector`]: learned tool selection from task features (per-feature x tool reward means, shrunk to
//!   the global tool mean, epsilon exploration); [`tool_features`] = cheap shape tokens of a task spec.
//!
//! Honest scope: these tools make exact sub-computations reliable. They do not make a model smarter and they
//! cannot verify anything that has no machine-readable spec.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;

use crate::router::Rng;

pub const MAX_INPUT: usize = 4096;
pub const MAX_DEPTH: usize = 64;

// ---------------------------------------------------------------------------------------------
// errors

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolError {
    Parse(String),
    Overflow,
    DivZero,
    StepLimit,
    MemLimit,
    TooLong,
    TooDeep,
    Unknown(String),
    BadInput(String),
    Dimension(String),
    NotFound(String),
    Domain(String),
}
impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ToolError::Parse(s) => write!(f, "parse error: {s}"),
            ToolError::Overflow => write!(f, "overflow"),
            ToolError::DivZero => write!(f, "division by zero"),
            ToolError::StepLimit => write!(f, "step limit"),
            ToolError::MemLimit => write!(f, "memory limit"),
            ToolError::TooLong => write!(f, "input too long"),
            ToolError::TooDeep => write!(f, "nesting too deep"),
            ToolError::Unknown(s) => write!(f, "unknown: {s}"),
            ToolError::BadInput(s) => write!(f, "bad input: {s}"),
            ToolError::Dimension(s) => write!(f, "dimension mismatch: {s}"),
            ToolError::NotFound(s) => write!(f, "not found: {s}"),
            ToolError::Domain(s) => write!(f, "domain: {s}"),
        }
    }
}
impl std::error::Error for ToolError {}
type R<T> = Result<T, ToolError>;

fn perr<T>(s: &str) -> R<T> {
    Err(ToolError::Parse(s.to_string()))
}

// ---------------------------------------------------------------------------------------------
// exact rationals (i128, normalised, checked)

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rat {
    n: i128,
    d: i128,
}

fn gcd_u(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}
/// callers guarantee neither argument is i128::MIN
fn gcd(a: i128, b: i128) -> i128 {
    gcd_u(a.unsigned_abs(), b.unsigned_abs()) as i128
}

impl Rat {
    pub fn new(n: i128, d: i128) -> R<Rat> {
        if d == 0 {
            return Err(ToolError::DivZero);
        }
        if n == i128::MIN || d == i128::MIN {
            return Err(ToolError::Overflow);
        }
        let (mut n, mut d) = if d < 0 { (-n, -d) } else { (n, d) };
        let g = gcd(n, d);
        if g > 1 {
            n /= g;
            d /= g;
        }
        Ok(Rat { n, d })
    }
    pub fn int(n: i128) -> Rat {
        Rat { n: if n == i128::MIN { i128::MIN + 1 } else { n }, d: 1 }
    }
    pub fn numer(&self) -> i128 {
        self.n
    }
    pub fn denom(&self) -> i128 {
        self.d
    }
    pub fn is_int(&self) -> bool {
        self.d == 1
    }
    pub fn is_zero(&self) -> bool {
        self.n == 0
    }
    pub fn to_f64(&self) -> f64 {
        self.n as f64 / self.d as f64
    }
    pub fn neg(self) -> Rat {
        Rat { n: -self.n, d: self.d }
    }
    pub fn add(self, o: Rat) -> R<Rat> {
        let g = gcd(self.d, o.d);
        let (d1, d2) = (self.d / g, o.d / g);
        let a = self.n.checked_mul(d2).ok_or(ToolError::Overflow)?;
        let b = o.n.checked_mul(d1).ok_or(ToolError::Overflow)?;
        let n = a.checked_add(b).ok_or(ToolError::Overflow)?;
        let d = self.d.checked_mul(d2).ok_or(ToolError::Overflow)?;
        Rat::new(n, d)
    }
    pub fn sub(self, o: Rat) -> R<Rat> {
        self.add(o.neg())
    }
    pub fn mul(self, o: Rat) -> R<Rat> {
        let g1 = gcd(self.n, o.d);
        let g2 = gcd(o.n, self.d);
        let n = (self.n / g1).checked_mul(o.n / g2).ok_or(ToolError::Overflow)?;
        let d = (self.d / g2).checked_mul(o.d / g1).ok_or(ToolError::Overflow)?;
        Rat::new(n, d)
    }
    pub fn recip(self) -> R<Rat> {
        if self.n == 0 {
            return Err(ToolError::DivZero);
        }
        if self.n < 0 { Rat::new(-self.d, -self.n) } else { Rat::new(self.d, self.n) }
    }
    pub fn div(self, o: Rat) -> R<Rat> {
        self.mul(o.recip()?)
    }
    /// integer power, |e| <= 10_000; 0^0 = 1; 0^negative = DivZero
    pub fn pow_i(self, e: i128) -> R<Rat> {
        if e.unsigned_abs() > 10_000 {
            return Err(ToolError::Domain("exponent too large".into()));
        }
        let (mut base, mut e) = if e < 0 { (self.recip()?, (-e) as u32) } else { (self, e as u32) };
        let mut acc = Rat::int(1);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(base)?;
            }
            e >>= 1;
            if e > 0 {
                base = base.mul(base)?;
            }
        }
        Ok(acc)
    }
    pub fn compare(self, o: Rat) -> R<Ordering> {
        match (self.n.checked_mul(o.d), o.n.checked_mul(self.d)) {
            (Some(a), Some(b)) => Ok(a.cmp(&b)),
            _ => {
                let (fa, fb) = (self.n.div_euclid(self.d), o.n.div_euclid(o.d));
                if fa != fb { Ok(fa.cmp(&fb)) } else { Err(ToolError::Overflow) }
            }
        }
    }
    pub fn floor(self) -> Rat {
        Rat { n: self.n.div_euclid(self.d), d: 1 }
    }
    pub fn ceil(self) -> Rat {
        Rat { n: -((-self.n).div_euclid(self.d)), d: 1 }
    }
    /// "12", "-3.25", "1/3", "-7/2"
    pub fn parse(s: &str) -> R<Rat> {
        let s = s.trim();
        if s.is_empty() || s.len() > 80 {
            return perr("empty or long number");
        }
        if let Some((a, b)) = s.split_once('/') {
            let n: i128 = a.trim().parse().map_err(|_| ToolError::Parse("numerator".into()))?;
            let d: i128 = b.trim().parse().map_err(|_| ToolError::Parse("denominator".into()))?;
            return Rat::new(n, d);
        }
        let (neg, body) = match s.strip_prefix('-') {
            Some(r) => (true, r),
            None => (false, s.strip_prefix('+').unwrap_or(s)),
        };
        let r = Rat::parse_unsigned(body)?;
        Ok(if neg { r.neg() } else { r })
    }
    fn parse_unsigned(body: &str) -> R<Rat> {
        let (ip, fp) = match body.split_once('.') {
            Some((a, b)) => (a, b),
            None => (body, ""),
        };
        if ip.is_empty() || !ip.bytes().all(|c| c.is_ascii_digit()) || !fp.bytes().all(|c| c.is_ascii_digit()) {
            return perr("number");
        }
        if body.contains('.') && fp.is_empty() {
            return perr("number");
        }
        if fp.len() > 30 || ip.len() + fp.len() > 37 {
            return Err(ToolError::Overflow);
        }
        let digits = format!("{ip}{fp}");
        let n: i128 = digits.parse().map_err(|_| ToolError::Overflow)?;
        let mut d: i128 = 1;
        for _ in 0..fp.len() {
            d = d.checked_mul(10).ok_or(ToolError::Overflow)?;
        }
        Rat::new(n, d)
    }
}
impl fmt::Display for Rat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.d == 1 { write!(f, "{}", self.n) } else { write!(f, "{}/{}", self.n, self.d) }
    }
}

// ---------------------------------------------------------------------------------------------
// expression lexer / parser / evaluator (shared by calc, csp, identity)

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Le,
    Ge,
    Lt,
    Gt,
}
fn cmp_holds(c: Cmp, o: Ordering) -> bool {
    match c {
        Cmp::Eq => o == Ordering::Equal,
        Cmp::Ne => o != Ordering::Equal,
        Cmp::Le => o != Ordering::Greater,
        Cmp::Ge => o != Ordering::Less,
        Cmp::Lt => o == Ordering::Less,
        Cmp::Gt => o == Ordering::Greater,
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(Rat),
    Id(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    LP,
    RP,
    Comma,
    Cmp(Cmp),
}

fn lex(s: &str) -> R<Vec<Tok>> {
    if s.len() > MAX_INPUT {
        return Err(ToolError::TooLong);
    }
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        match c {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            b'0'..=b'9' => {
                let st = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                if i < b.len() && b[i] == b'.' {
                    i += 1;
                    let fs = i;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                    if i == fs {
                        return perr("digits expected after '.'");
                    }
                }
                if i < b.len() && (b[i].is_ascii_alphabetic() || b[i] == b'_' || b[i] == b'.') {
                    return perr("malformed number");
                }
                out.push(Tok::Num(Rat::parse_unsigned(&s[st..i])?));
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let st = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                out.push(Tok::Id(s[st..i].to_string()));
            }
            b'+' => {
                out.push(Tok::Plus);
                i += 1
            }
            b'-' => {
                out.push(Tok::Minus);
                i += 1
            }
            b'*' => {
                out.push(Tok::Star);
                i += 1
            }
            b'/' => {
                out.push(Tok::Slash);
                i += 1
            }
            b'%' => {
                out.push(Tok::Percent);
                i += 1
            }
            b'^' => {
                out.push(Tok::Caret);
                i += 1
            }
            b'(' => {
                out.push(Tok::LP);
                i += 1
            }
            b')' => {
                out.push(Tok::RP);
                i += 1
            }
            b',' => {
                out.push(Tok::Comma);
                i += 1
            }
            b'=' => {
                if i + 1 < b.len() && b[i + 1] == b'=' {
                    out.push(Tok::Cmp(Cmp::Eq));
                    i += 2;
                } else {
                    return perr("single '='");
                }
            }
            b'!' => {
                if i + 1 < b.len() && b[i + 1] == b'=' {
                    out.push(Tok::Cmp(Cmp::Ne));
                    i += 2;
                } else {
                    return perr("lone '!'");
                }
            }
            b'<' => {
                if i + 1 < b.len() && b[i + 1] == b'=' {
                    out.push(Tok::Cmp(Cmp::Le));
                    i += 2;
                } else {
                    out.push(Tok::Cmp(Cmp::Lt));
                    i += 1;
                }
            }
            b'>' => {
                if i + 1 < b.len() && b[i + 1] == b'=' {
                    out.push(Tok::Cmp(Cmp::Ge));
                    i += 2;
                } else {
                    out.push(Tok::Cmp(Cmp::Gt));
                    i += 1;
                }
            }
            _ => return perr("unexpected character"),
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
}

#[derive(Clone, Debug)]
pub enum Expr {
    Num(Rat),
    Var(String),
    Neg(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

struct Parser {
    t: Vec<Tok>,
    i: usize,
    depth: usize,
}
impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn bump(&mut self) -> Option<Tok> {
        let t = self.t.get(self.i).cloned();
        if t.is_some() {
            self.i += 1;
        }
        t
    }
    fn enter(&mut self) -> R<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH { Err(ToolError::TooDeep) } else { Ok(()) }
    }
    fn expr(&mut self) -> R<Expr> {
        self.enter()?;
        let mut l = self.term()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Plus) => Op::Add,
                Some(Tok::Minus) => Op::Sub,
                _ => break,
            };
            self.i += 1;
            let r = self.term()?;
            l = Expr::Bin(op, Box::new(l), Box::new(r));
        }
        self.depth -= 1;
        Ok(l)
    }
    fn term(&mut self) -> R<Expr> {
        let mut l = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Tok::Star) => Op::Mul,
                Some(Tok::Slash) => Op::Div,
                Some(Tok::Percent) => Op::Mod,
                _ => break,
            };
            self.i += 1;
            let r = self.unary()?;
            l = Expr::Bin(op, Box::new(l), Box::new(r));
        }
        Ok(l)
    }
    fn unary(&mut self) -> R<Expr> {
        self.enter()?;
        let e = match self.peek() {
            Some(Tok::Minus) => {
                self.i += 1;
                Expr::Neg(Box::new(self.unary()?))
            }
            Some(Tok::Plus) => {
                self.i += 1;
                self.unary()?
            }
            _ => self.power()?,
        };
        self.depth -= 1;
        Ok(e)
    }
    fn power(&mut self) -> R<Expr> {
        let base = self.atom()?;
        if let Some(Tok::Caret) = self.peek() {
            self.i += 1;
            let ex = self.unary()?;
            return Ok(Expr::Bin(Op::Pow, Box::new(base), Box::new(ex)));
        }
        Ok(base)
    }
    fn atom(&mut self) -> R<Expr> {
        match self.bump() {
            Some(Tok::Num(r)) => Ok(Expr::Num(r)),
            Some(Tok::Id(name)) => {
                if let Some(Tok::LP) = self.peek() {
                    self.i += 1;
                    let mut args = Vec::new();
                    if let Some(Tok::RP) = self.peek() {
                        self.i += 1;
                        return Ok(Expr::Call(name, args));
                    }
                    loop {
                        args.push(self.expr()?);
                        if args.len() > 16 {
                            return perr("too many arguments");
                        }
                        match self.bump() {
                            Some(Tok::Comma) => continue,
                            Some(Tok::RP) => break,
                            _ => return perr("expected ',' or ')'"),
                        }
                    }
                    Ok(Expr::Call(name, args))
                } else {
                    Ok(Expr::Var(name))
                }
            }
            Some(Tok::LP) => {
                let e = self.expr()?;
                match self.bump() {
                    Some(Tok::RP) => Ok(e),
                    _ => perr("expected ')'"),
                }
            }
            Some(_) => perr("unexpected token"),
            None => perr("unexpected end of input"),
        }
    }
}

pub fn parse_expr(s: &str) -> R<Expr> {
    let mut p = Parser { t: lex(s)?, i: 0, depth: 0 };
    if p.t.is_empty() {
        return perr("empty expression");
    }
    let e = p.expr()?;
    if p.i != p.t.len() {
        return perr("trailing input");
    }
    Ok(e)
}

/// `lhs <cmp> rhs` with exactly one top-level comparator
pub fn parse_cmp(s: &str) -> R<(Expr, Cmp, Expr)> {
    let mut p = Parser { t: lex(s)?, i: 0, depth: 0 };
    if p.t.is_empty() {
        return perr("empty constraint");
    }
    let l = p.expr()?;
    let c = match p.bump() {
        Some(Tok::Cmp(c)) => c,
        _ => return perr("comparator expected"),
    };
    let r = p.expr()?;
    if p.i != p.t.len() {
        return perr("trailing input");
    }
    Ok((l, c, r))
}

pub fn collect_vars(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Num(_) => {}
        Expr::Var(v) => {
            if !out.contains(v) {
                out.push(v.clone());
            }
        }
        Expr::Neg(a) => collect_vars(a, out),
        Expr::Bin(_, a, b) => {
            collect_vars(a, out);
            collect_vars(b, out);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| collect_vars(a, out)),
    }
}

fn env_get(env: &[(String, Rat)], name: &str) -> Option<Rat> {
    env.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
}

pub fn eval_expr(e: &Expr, env: &[(String, Rat)], budget: &mut u64) -> R<Rat> {
    if *budget == 0 {
        return Err(ToolError::StepLimit);
    }
    *budget -= 1;
    match e {
        Expr::Num(r) => Ok(*r),
        Expr::Var(v) => env_get(env, v).ok_or_else(|| ToolError::Unknown(format!("variable {v}"))),
        Expr::Neg(a) => Ok(eval_expr(a, env, budget)?.neg()),
        Expr::Bin(op, a, b) => {
            let x = eval_expr(a, env, budget)?;
            let y = eval_expr(b, env, budget)?;
            match op {
                Op::Add => x.add(y),
                Op::Sub => x.sub(y),
                Op::Mul => x.mul(y),
                Op::Div => x.div(y),
                Op::Mod => {
                    if !x.is_int() || !y.is_int() {
                        return Err(ToolError::Domain("% needs integers".into()));
                    }
                    if y.n == 0 {
                        return Err(ToolError::DivZero);
                    }
                    Ok(Rat::int(x.n.rem_euclid(y.n)))
                }
                Op::Pow => {
                    if !y.is_int() {
                        return Err(ToolError::Domain("exponent must be an integer".into()));
                    }
                    x.pow_i(y.n)
                }
            }
        }
        Expr::Call(name, args) => {
            let mut vals = Vec::with_capacity(args.len());
            for a in args {
                vals.push(eval_expr(a, env, budget)?);
            }
            let need = |n: usize| -> R<()> {
                if vals.len() == n { Ok(()) } else { perr("wrong argument count") }
            };
            match name.as_str() {
                "abs" => {
                    need(1)?;
                    Ok(if vals[0].n < 0 { vals[0].neg() } else { vals[0] })
                }
                "floor" => {
                    need(1)?;
                    Ok(vals[0].floor())
                }
                "ceil" => {
                    need(1)?;
                    Ok(vals[0].ceil())
                }
                "min" | "max" => {
                    if vals.is_empty() {
                        return perr("wrong argument count");
                    }
                    let mut best = vals[0];
                    for v in &vals[1..] {
                        let o = v.compare(best)?;
                        if (name == "min" && o == Ordering::Less) || (name == "max" && o == Ordering::Greater) {
                            best = *v;
                        }
                    }
                    Ok(best)
                }
                "gcd" => {
                    need(2)?;
                    if !vals[0].is_int() || !vals[1].is_int() {
                        return Err(ToolError::Domain("gcd needs integers".into()));
                    }
                    Ok(Rat::int(gcd(vals[0].n, vals[1].n)))
                }
                _ => Err(ToolError::Unknown(format!("function {name}"))),
            }
        }
    }
}

/// independent f64 evaluation (verifier sanity path); None when unsupported / non-finite
pub fn eval_f64(e: &Expr, env: &[(String, f64)]) -> Option<f64> {
    let v = match e {
        Expr::Num(r) => r.to_f64(),
        Expr::Var(v) => env.iter().find(|(n, _)| n == v)?.1,
        Expr::Neg(a) => -eval_f64(a, env)?,
        Expr::Bin(op, a, b) => {
            let (x, y) = (eval_f64(a, env)?, eval_f64(b, env)?);
            match op {
                Op::Add => x + y,
                Op::Sub => x - y,
                Op::Mul => x * y,
                Op::Div => x / y,
                Op::Mod => x.rem_euclid(y),
                Op::Pow => x.powf(y),
            }
        }
        Expr::Call(name, args) => {
            let vs: Option<Vec<f64>> = args.iter().map(|a| eval_f64(a, env)).collect();
            let vs = vs?;
            match (name.as_str(), vs.as_slice()) {
                ("abs", [x]) => x.abs(),
                ("floor", [x]) => x.floor(),
                ("ceil", [x]) => x.ceil(),
                ("min", v) if !v.is_empty() => v.iter().cloned().fold(f64::INFINITY, f64::min),
                ("max", v) if !v.is_empty() => v.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                _ => return None,
            }
        }
    };
    if v.is_finite() { Some(v) } else { None }
}

// ---------------------------------------------------------------------------------------------
// the Tool trait

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    /// nominal cost units of one call
    fn cost(&self) -> u32;
    fn cacheable(&self) -> bool {
        true
    }
    /// pure and deterministic: same input -> same output, no I/O, no global state
    fn call(&self, input: &str) -> R<String>;
    /// result plus the cost actually charged (some tools meter their work)
    fn call_metered(&self, input: &str) -> (R<String>, u32) {
        (self.call(input), self.cost())
    }
}

// ---------------------------------------------------------------------------------------------
// calc

#[derive(Clone, Copy, Debug, Default)]
pub struct Calc;
impl Calc {
    pub fn eval_str(s: &str) -> R<Rat> {
        let e = parse_expr(s)?;
        let mut budget = 200_000u64;
        eval_expr(&e, &[], &mut budget)
    }
}
impl Tool for Calc {
    fn name(&self) -> &str {
        "calc"
    }
    fn cost(&self) -> u32 {
        1
    }
    fn call(&self, input: &str) -> R<String> {
        Ok(Calc::eval_str(input)?.to_string())
    }
}

// ---------------------------------------------------------------------------------------------
// units / dimensions

const NDIM: usize = 5; // L M T Theta I
type Dim = [i8; NDIM];

struct UnitDef {
    name: &'static str,
    fnum: i128,
    fden: i128,
    dim: Dim,
}
const DL: Dim = [1, 0, 0, 0, 0];
const DM: Dim = [0, 1, 0, 0, 0];
const DT: Dim = [0, 0, 1, 0, 0];
const DTH: Dim = [0, 0, 0, 1, 0];
const DI: Dim = [0, 0, 0, 0, 1];
const DNONE: Dim = [0; NDIM];
const UNITS: &[UnitDef] = &[
    UnitDef { name: "m", fnum: 1, fden: 1, dim: DL },
    UnitDef { name: "km", fnum: 1000, fden: 1, dim: DL },
    UnitDef { name: "cm", fnum: 1, fden: 100, dim: DL },
    UnitDef { name: "mm", fnum: 1, fden: 1000, dim: DL },
    UnitDef { name: "in", fnum: 254, fden: 10_000, dim: DL },
    UnitDef { name: "ft", fnum: 3048, fden: 10_000, dim: DL },
    UnitDef { name: "yd", fnum: 9144, fden: 10_000, dim: DL },
    UnitDef { name: "mile", fnum: 1_609_344, fden: 1000, dim: DL },
    UnitDef { name: "kg", fnum: 1, fden: 1, dim: DM },
    UnitDef { name: "g", fnum: 1, fden: 1000, dim: DM },
    UnitDef { name: "mg", fnum: 1, fden: 1_000_000, dim: DM },
    UnitDef { name: "t", fnum: 1000, fden: 1, dim: DM },
    UnitDef { name: "lb", fnum: 45_359_237, fden: 100_000_000, dim: DM },
    UnitDef { name: "oz", fnum: 28_349_523_125, fden: 1_000_000_000_000, dim: DM },
    UnitDef { name: "s", fnum: 1, fden: 1, dim: DT },
    UnitDef { name: "ms", fnum: 1, fden: 1000, dim: DT },
    UnitDef { name: "min", fnum: 60, fden: 1, dim: DT },
    UnitDef { name: "h", fnum: 3600, fden: 1, dim: DT },
    UnitDef { name: "day", fnum: 86_400, fden: 1, dim: DT },
    UnitDef { name: "week", fnum: 604_800, fden: 1, dim: DT },
    UnitDef { name: "K", fnum: 1, fden: 1, dim: DTH },
    UnitDef { name: "A", fnum: 1, fden: 1, dim: DI },
    UnitDef { name: "mA", fnum: 1, fden: 1000, dim: DI },
    UnitDef { name: "Hz", fnum: 1, fden: 1, dim: [0, 0, -1, 0, 0] },
    UnitDef { name: "N", fnum: 1, fden: 1, dim: [1, 1, -2, 0, 0] },
    UnitDef { name: "Pa", fnum: 1, fden: 1, dim: [-1, 1, -2, 0, 0] },
    UnitDef { name: "J", fnum: 1, fden: 1, dim: [2, 1, -2, 0, 0] },
    UnitDef { name: "kJ", fnum: 1000, fden: 1, dim: [2, 1, -2, 0, 0] },
    UnitDef { name: "W", fnum: 1, fden: 1, dim: [2, 1, -3, 0, 0] },
    UnitDef { name: "kW", fnum: 1000, fden: 1, dim: [2, 1, -3, 0, 0] },
    UnitDef { name: "L", fnum: 1, fden: 1000, dim: [3, 0, 0, 0, 0] },
    UnitDef { name: "mL", fnum: 1, fden: 1_000_000, dim: [3, 0, 0, 0, 0] },
    UnitDef { name: "knot", fnum: 463, fden: 900, dim: [1, 0, -1, 0, 0] },
    UnitDef { name: "C", fnum: 1, fden: 1, dim: [0, 0, 1, 0, 1] },
    UnitDef { name: "V", fnum: 1, fden: 1, dim: [2, 1, -3, 0, -1] },
];

fn parse_unit_expr(s: &str) -> R<(Rat, Dim)> {
    let s = s.trim();
    if s.is_empty() || s.len() > 200 {
        return perr("unit expression");
    }
    let mut factor = Rat::int(1);
    let mut dim: Dim = DNONE;
    let mut op = '*';
    let mut cur = String::new();
    let mut toks: Vec<(char, String)> = Vec::new();
    for ch in s.chars() {
        if ch == '*' || ch == '/' {
            toks.push((op, cur.trim().to_string()));
            cur.clear();
            op = ch;
        } else {
            cur.push(ch);
        }
    }
    toks.push((op, cur.trim().to_string()));
    for (o, tok) in toks {
        if tok.is_empty() {
            return perr("empty unit term");
        }
        if tok == "1" {
            continue;
        }
        let (name, e) = match tok.split_once('^') {
            Some((n, e)) => (n.trim(), e.trim().parse::<i32>().map_err(|_| ToolError::Parse("unit exponent".into()))?),
            None => (tok.as_str(), 1),
        };
        if e.abs() > 12 {
            return Err(ToolError::Domain("unit exponent too large".into()));
        }
        let u = UNITS.iter().find(|u| u.name == name).ok_or_else(|| ToolError::Unknown(format!("unit {name}")))?;
        let e = if o == '/' { -e } else { e };
        let f = Rat::new(u.fnum, u.fden)?.pow_i(e as i128)?;
        factor = factor.mul(f)?;
        for k in 0..NDIM {
            let add = (u.dim[k] as i32).checked_mul(e).ok_or(ToolError::Overflow)?;
            let nd = dim[k] as i32 + add;
            if nd < i8::MIN as i32 || nd > i8::MAX as i32 {
                return Err(ToolError::Overflow);
            }
            dim[k] = nd as i8;
        }
    }
    Ok((factor, dim))
}

fn dim_str(d: &Dim) -> String {
    format!("L={} M={} T={} K={} I={}", d[0], d[1], d[2], d[3], d[4])
}

/// `conv <value> <unit> -> <unit>` | `sum <v> <u> + <v> <u> ... -> <unit>` | `dim <unit>` | `same <u1> | <u2>`
#[derive(Clone, Copy, Debug, Default)]
pub struct Units;
impl Units {
    fn quantity(term: &str) -> R<(Rat, Rat, Dim)> {
        let term = term.trim();
        let (v, u) = term.split_once(char::is_whitespace).ok_or(ToolError::Parse("quantity needs value and unit".into()))?;
        let val = Rat::parse(v)?;
        let (f, d) = parse_unit_expr(u)?;
        Ok((val, f, d))
    }
}
impl Tool for Units {
    fn name(&self) -> &str {
        "units"
    }
    fn cost(&self) -> u32 {
        1
    }
    fn call(&self, input: &str) -> R<String> {
        if input.len() > MAX_INPUT {
            return Err(ToolError::TooLong);
        }
        let input = input.trim();
        let (cmd, rest) = input.split_once(char::is_whitespace).unwrap_or((input, ""));
        match cmd {
            "conv" => {
                let (lhs, to) = rest.split_once("->").ok_or(ToolError::Parse("'->' expected".into()))?;
                let (val, ff, fd) = Units::quantity(lhs)?;
                let (tf, td) = parse_unit_expr(to)?;
                if fd != td {
                    return Err(ToolError::Dimension(format!("{} vs {}", dim_str(&fd), dim_str(&td))));
                }
                Ok(val.mul(ff)?.div(tf)?.to_string())
            }
            "sum" => {
                let (lhs, to) = rest.split_once("->").ok_or(ToolError::Parse("'->' expected".into()))?;
                let (tf, td) = parse_unit_expr(to)?;
                let mut total = Rat::int(0);
                let mut n = 0;
                for term in lhs.split('+') {
                    let (val, f, d) = Units::quantity(term)?;
                    if d != td {
                        return Err(ToolError::Dimension(format!("{} vs {}", dim_str(&d), dim_str(&td))));
                    }
                    total = total.add(val.mul(f)?)?;
                    n += 1;
                    if n > 32 {
                        return Err(ToolError::TooLong);
                    }
                }
                Ok(total.div(tf)?.to_string())
            }
            "dim" => Ok(dim_str(&parse_unit_expr(rest)?.1)),
            "same" => {
                let (a, b) = rest.split_once('|').ok_or(ToolError::Parse("'|' expected".into()))?;
                Ok((parse_unit_expr(a)?.1 == parse_unit_expr(b)?.1).to_string())
            }
            _ => Err(ToolError::BadInput(format!("unknown command {cmd:?}"))),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// csp: small finite-domain constraint solver + certificate checker

struct Problem {
    vars: Vec<(String, i128, i128)>,
    cons: Vec<(Expr, Cmp, Expr)>,
    /// constraint idx -> highest variable index it uses (None = constants only)
    level: Vec<Option<usize>>,
}

fn is_ident(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty() && (b[0].is_ascii_alphabetic() || b[0] == b'_') && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

fn parse_problem(spec: &str) -> R<Problem> {
    if spec.len() > MAX_INPUT {
        return Err(ToolError::TooLong);
    }
    let mut vars: Vec<(String, i128, i128)> = Vec::new();
    let mut cons = Vec::new();
    for part in spec.split(';') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((name, range)) = part.split_once(" in ") {
            let name = name.trim();
            if !is_ident(name) {
                return perr("variable name");
            }
            if vars.iter().any(|v| v.0 == name) {
                return perr("duplicate variable");
            }
            let (lo, hi) = range.split_once("..").ok_or(ToolError::Parse("range lo..hi expected".into()))?;
            let lo: i128 = lo.trim().parse().map_err(|_| ToolError::Parse("range bound".into()))?;
            let hi: i128 = hi.trim().parse().map_err(|_| ToolError::Parse("range bound".into()))?;
            if hi < lo || hi - lo >= 1000 || lo.unsigned_abs() > 1_000_000 || hi.unsigned_abs() > 1_000_000 {
                return Err(ToolError::Domain("range".into()));
            }
            if vars.len() >= 8 {
                return Err(ToolError::Domain("too many variables".into()));
            }
            vars.push((name.to_string(), lo, hi));
        } else {
            if cons.len() >= 32 {
                return Err(ToolError::Domain("too many constraints".into()));
            }
            cons.push(parse_cmp(part)?);
        }
    }
    if vars.is_empty() {
        return perr("no variables");
    }
    let mut level = Vec::new();
    for (l, _, r) in &cons {
        let mut vs = Vec::new();
        collect_vars(l, &mut vs);
        collect_vars(r, &mut vs);
        let mut mx: Option<usize> = None;
        for v in &vs {
            let idx = vars.iter().position(|x| &x.0 == v).ok_or_else(|| ToolError::Unknown(format!("variable {v}")))?;
            mx = Some(mx.map_or(idx, |m| m.max(idx)));
        }
        level.push(mx);
    }
    Ok(Problem { vars, cons, level })
}

impl Problem {
    fn holds(&self, ci: usize, env: &[(String, Rat)], budget: &mut u64) -> R<bool> {
        let (l, c, r) = &self.cons[ci];
        let a = eval_expr(l, env, budget);
        let b = eval_expr(r, env, budget);
        match (a, b) {
            (Ok(a), Ok(b)) => Ok(cmp_holds(*c, a.compare(b)?)),
            (Err(ToolError::DivZero), _) | (_, Err(ToolError::DivZero)) => Ok(false),
            (Err(e), _) | (_, Err(e)) => Err(e),
        }
    }
    fn solve(&self, k: usize, env: &mut Vec<(String, Rat)>, nodes: &mut u64, budget: &mut u64) -> R<bool> {
        if k == self.vars.len() {
            return Ok(true);
        }
        let (name, lo, hi) = &self.vars[k];
        let mut v = *lo;
        while v <= *hi {
            *nodes += 1;
            if *nodes > 1_000_000 {
                return Err(ToolError::StepLimit);
            }
            env.push((name.clone(), Rat::int(v)));
            let mut ok = true;
            for ci in 0..self.cons.len() {
                if self.level[ci] == Some(k) && !self.holds(ci, env, budget)? {
                    ok = false;
                    break;
                }
            }
            if ok && self.solve(k + 1, env, nodes, budget)? {
                return Ok(true);
            }
            env.pop();
            v += 1;
        }
        Ok(false)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Csp;
impl Csp {
    /// "x=3,y=7" (first solution in declaration-order lexicographic order) or "unsat"
    pub fn solve_str(spec: &str) -> R<String> {
        let p = parse_problem(spec)?;
        let mut budget = 50_000_000u64;
        for ci in 0..p.cons.len() {
            if p.level[ci].is_none() && !p.holds(ci, &[], &mut budget)? {
                return Ok("unsat".into());
            }
        }
        let mut env = Vec::new();
        let mut nodes = 0u64;
        if p.solve(0, &mut env, &mut nodes, &mut budget)? {
            Ok(env.iter().map(|(n, v)| format!("{n}={v}")).collect::<Vec<_>>().join(","))
        } else {
            Ok("unsat".into())
        }
    }
    /// does the claimed assignment (any order) satisfy every domain and constraint?
    pub fn check_assignment(spec: &str, assignment: &str) -> R<bool> {
        let p = parse_problem(spec)?;
        let mut vals: HashMap<String, Rat> = HashMap::new();
        for part in assignment.split(',') {
            let (n, v) = part.split_once('=').ok_or(ToolError::Parse("assignment name=value".into()))?;
            let v = Rat::parse(v)?;
            if vals.insert(n.trim().to_string(), v).is_some() {
                return perr("duplicate variable in assignment");
            }
        }
        let mut env = Vec::new();
        for (name, lo, hi) in &p.vars {
            let v = *vals.get(name).ok_or(ToolError::Parse("variable missing in assignment".into()))?;
            if !v.is_int() || v.n < *lo || v.n > *hi {
                return Ok(false);
            }
            env.push((name.clone(), v));
        }
        if vals.len() != p.vars.len() {
            return Ok(false);
        }
        let mut budget = 1_000_000u64;
        for ci in 0..p.cons.len() {
            if !p.holds(ci, &env, &mut budget)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
impl Tool for Csp {
    fn name(&self) -> &str {
        "csp"
    }
    fn cost(&self) -> u32 {
        3
    }
    fn call(&self, input: &str) -> R<String> {
        Csp::solve_str(input)
    }
}

// ---------------------------------------------------------------------------------------------
// identity: grid check of `lhs == rhs` (small symbolic claims)

#[derive(Clone, Copy, Debug)]
pub struct Identity {
    pub lo: i64,
    pub hi: i64,
}
impl Default for Identity {
    fn default() -> Self {
        Identity { lo: -8, hi: 8 }
    }
}
impl Identity {
    /// "identical" when both sides agree (exactly, rationals) at every grid point where they are defined
    /// (>= 3 such points required); otherwise "differs: x=2 lhs=.. rhs=..". NOT a proof for non-polynomials.
    pub fn check(&self, spec: &str) -> R<String> {
        let (l, r) = spec.split_once("==").ok_or(ToolError::Parse("'==' expected".into()))?;
        if r.contains("==") || l.contains("<") || l.contains(">") || l.contains("!") || r.contains("<") || r.contains(">") || r.contains("!") {
            return perr("exactly one '==' expected");
        }
        let le = parse_expr(l)?;
        let re = parse_expr(r)?;
        let mut vars = Vec::new();
        collect_vars(&le, &mut vars);
        collect_vars(&re, &mut vars);
        vars.sort();
        if vars.len() > 2 {
            return Err(ToolError::Domain("at most 2 variables".into()));
        }
        let pts: Vec<i64> = (self.lo..=self.hi).collect();
        let mut budget = 5_000_000u64;
        let mut valid = 0;
        let grid: Vec<Vec<i64>> = match vars.len() {
            0 => vec![vec![]],
            1 => pts.iter().map(|a| vec![*a]).collect(),
            _ => pts.iter().flat_map(|a| pts.iter().map(move |b| vec![*a, *b])).collect(),
        };
        let need = grid.len().min(3);
        for g in grid {
            let env: Vec<(String, Rat)> = vars.iter().cloned().zip(g.iter().map(|v| Rat::int(*v as i128))).collect();
            let a = eval_expr(&le, &env, &mut budget);
            let b = eval_expr(&re, &env, &mut budget);
            let desc = || env.iter().map(|(n, v)| format!("{n}={v}")).collect::<Vec<_>>().join(",");
            match (a, b) {
                (Ok(a), Ok(b)) => {
                    valid += 1;
                    if a != b {
                        return Ok(format!("differs: {} lhs={} rhs={}", desc(), a, b));
                    }
                }
                (Err(ToolError::DivZero), Err(ToolError::DivZero)) => {}
                (Err(ToolError::DivZero), Ok(_)) | (Ok(_), Err(ToolError::DivZero)) => return Ok(format!("differs: {} (definedness)", desc())),
                (Err(e), _) | (_, Err(e)) => return Err(e),
            }
        }
        if valid < need {
            return Err(ToolError::Domain("too few defined grid points".into()));
        }
        Ok("identical".into())
    }
}
impl Identity {
    /// R39 (opt-in): `check` on the integer grid, then exact probes at rationals and large magnitudes. A pair that agrees on every integer of the grid
    /// but not elsewhere (`floor(x) == x`, `min(x,40) == x`, ...) is reported "differs: ..." instead of "identical". Probe points where either side errors
    /// (domain / overflow / step limit) are SKIPPED; a definedness mismatch (one side divides by zero) counts as a difference. Exact rational arithmetic
    /// means a true identity never differs on a probe, so no true identity is rejected (except via skipped overflow).
    pub fn check_strict(&self, spec: &str) -> R<String> {
        let base = self.check(spec)?;
        if base != "identical" {
            return Ok(base);
        }
        let (l, r) = spec.split_once("==").ok_or(ToolError::Parse("'==' expected".into()))?;
        let le = parse_expr(l)?;
        let re = parse_expr(r)?;
        let mut vars = Vec::new();
        collect_vars(&le, &mut vars);
        collect_vars(&re, &mut vars);
        vars.sort();
        let xs: [(i128, i128); 14] = [(1, 2), (-1, 2), (3, 2), (-3, 2), (1, 3), (7, 3), (-7, 3), (22, 7), (100, 1), (-100, 1), (1000, 1), (-1000, 1), (1, 1000), (1_000_000, 1)];
        let ys: [(i128, i128); 12] = [(1, 2), (-1, 2), (3, 2), (-3, 2), (1, 3), (7, 3), (2, 1), (-5, 1), (100, 1), (-100, 1), (1000, 1), (-1000, 1)];
        let mut pts: Vec<Vec<Rat>> = Vec::new();
        match vars.len() {
            0 => {}
            1 => {
                for (n, d) in xs {
                    pts.push(vec![Rat::new(n, d)?]);
                }
            }
            _ => {
                for (n, d) in xs {
                    for (n2, d2) in ys {
                        pts.push(vec![Rat::new(n, d)?, Rat::new(n2, d2)?]);
                    }
                }
            }
        }
        let mut budget = 5_000_000u64;
        for g in pts {
            let env: Vec<(String, Rat)> = vars.iter().cloned().zip(g.iter().copied()).collect();
            let a = eval_expr(&le, &env, &mut budget);
            let b = eval_expr(&re, &env, &mut budget);
            let desc = || env.iter().map(|(n, v)| format!("{n}={v}")).collect::<Vec<_>>().join(",");
            match (a, b) {
                (Ok(a), Ok(b)) => {
                    if a != b {
                        return Ok(format!("differs: {} lhs={} rhs={}", desc(), a, b));
                    }
                }
                (Err(ToolError::DivZero), Ok(_)) | (Ok(_), Err(ToolError::DivZero)) => return Ok(format!("differs: {} (definedness)", desc())),
                _ => {} // both undefined / domain / overflow / step limit: skip this probe
            }
            if budget == 0 {
                break;
            }
        }
        Ok("identical".into())
    }
}
impl Tool for Identity {
    fn name(&self) -> &str {
        "identity"
    }
    fn cost(&self) -> u32 {
        2
    }
    fn call(&self, input: &str) -> R<String> {
        self.check(input)
    }
}

// ---------------------------------------------------------------------------------------------
// sandbox: own tiny interpreter. ints (i128) + int lists. No I/O, no functions beyond a fixed pure set.

const S_MAX_PROGRAM: usize = 8192;
const S_MAX_NODES: usize = 20_000;
const S_MAX_LIST: usize = 4096;
const S_MAX_LIVE: usize = 20_000;
const S_MAX_VARS: usize = 64;

#[derive(Clone, Debug, PartialEq)]
enum STok {
    Int(i128),
    Id(String),
    Let,
    While,
    If,
    Else,
    Return,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    LP,
    RP,
    LB,
    RB,
    LC,
    RC,
    Comma,
    Semi,
    Assign,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
}

fn slex(s: &str) -> R<Vec<STok>> {
    if s.len() > S_MAX_PROGRAM {
        return Err(ToolError::TooLong);
    }
    let b = s.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    let two = |i: usize, c: u8| i + 1 < b.len() && b[i + 1] == c;
    while i < b.len() {
        let c = b[i];
        match c {
            b' ' | b'\t' | b'\n' | b'\r' => i += 1,
            b'0'..=b'9' => {
                let st = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                if i < b.len() && (b[i].is_ascii_alphabetic() || b[i] == b'_') {
                    return perr("malformed number");
                }
                let v: i128 = s[st..i].parse().map_err(|_| ToolError::Overflow)?;
                out.push(STok::Int(v));
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                let st = i;
                while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                    i += 1;
                }
                out.push(match &s[st..i] {
                    "let" => STok::Let,
                    "while" => STok::While,
                    "if" => STok::If,
                    "else" => STok::Else,
                    "return" => STok::Return,
                    w => STok::Id(w.to_string()),
                });
            }
            b'+' => {
                out.push(STok::Plus);
                i += 1
            }
            b'-' => {
                out.push(STok::Minus);
                i += 1
            }
            b'*' => {
                out.push(STok::Star);
                i += 1
            }
            b'/' => {
                out.push(STok::Slash);
                i += 1
            }
            b'%' => {
                out.push(STok::Percent);
                i += 1
            }
            b'(' => {
                out.push(STok::LP);
                i += 1
            }
            b')' => {
                out.push(STok::RP);
                i += 1
            }
            b'[' => {
                out.push(STok::LB);
                i += 1
            }
            b']' => {
                out.push(STok::RB);
                i += 1
            }
            b'{' => {
                out.push(STok::LC);
                i += 1
            }
            b'}' => {
                out.push(STok::RC);
                i += 1
            }
            b',' => {
                out.push(STok::Comma);
                i += 1
            }
            b';' => {
                out.push(STok::Semi);
                i += 1
            }
            b'=' => {
                if two(i, b'=') {
                    out.push(STok::Eq);
                    i += 2
                } else {
                    out.push(STok::Assign);
                    i += 1
                }
            }
            b'!' => {
                if two(i, b'=') {
                    out.push(STok::Ne);
                    i += 2
                } else {
                    out.push(STok::Not);
                    i += 1
                }
            }
            b'<' => {
                if two(i, b'=') {
                    out.push(STok::Le);
                    i += 2
                } else {
                    out.push(STok::Lt);
                    i += 1
                }
            }
            b'>' => {
                if two(i, b'=') {
                    out.push(STok::Ge);
                    i += 2
                } else {
                    out.push(STok::Gt);
                    i += 1
                }
            }
            b'&' => {
                if two(i, b'&') {
                    out.push(STok::And);
                    i += 2
                } else {
                    return perr("lone '&'");
                }
            }
            b'|' => {
                if two(i, b'|') {
                    out.push(STok::Or);
                    i += 2
                } else {
                    return perr("lone '|'");
                }
            }
            _ => return perr("unexpected character"),
        }
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}
#[derive(Clone, Debug)]
enum SExpr {
    Int(i128),
    Var(String),
    List(Vec<SExpr>),
    Index(Box<SExpr>, Box<SExpr>),
    Neg(Box<SExpr>),
    Not(Box<SExpr>),
    Bin(BOp, Box<SExpr>, Box<SExpr>),
    Call(String, Vec<SExpr>),
}
#[derive(Clone, Debug)]
enum Stmt {
    Let(String, SExpr),
    Assign(String, SExpr),
    SetIdx(String, SExpr, SExpr),
    While(SExpr, Vec<Stmt>),
    If(SExpr, Vec<Stmt>, Vec<Stmt>),
    Return(SExpr),
    Expr(SExpr),
}

struct SParser {
    t: Vec<STok>,
    i: usize,
    depth: usize,
    nodes: usize,
}
impl SParser {
    fn peek(&self) -> Option<&STok> {
        self.t.get(self.i)
    }
    fn bump(&mut self) -> Option<STok> {
        let t = self.t.get(self.i).cloned();
        if t.is_some() {
            self.i += 1;
        }
        t
    }
    fn eat(&mut self, t: &STok) -> bool {
        if self.peek() == Some(t) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    fn expect(&mut self, t: STok) -> R<()> {
        if self.eat(&t) { Ok(()) } else { perr(&format!("expected {t:?}")) }
    }
    fn enter(&mut self) -> R<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(ToolError::TooDeep);
        }
        self.nodes += 1;
        if self.nodes > S_MAX_NODES {
            return Err(ToolError::TooLong);
        }
        Ok(())
    }
    fn leave(&mut self) {
        self.depth -= 1;
    }
    fn block(&mut self) -> R<Vec<Stmt>> {
        self.expect(STok::LC)?;
        self.enter()?;
        let mut v = Vec::new();
        while self.peek() != Some(&STok::RC) {
            if self.peek().is_none() {
                return perr("unterminated block");
            }
            v.push(self.stmt()?);
        }
        self.i += 1;
        self.leave();
        Ok(v)
    }
    fn stmt(&mut self) -> R<Stmt> {
        self.enter()?;
        let s = match self.peek().cloned() {
            Some(STok::Let) => {
                self.i += 1;
                let name = match self.bump() {
                    Some(STok::Id(n)) => n,
                    _ => return perr("identifier expected"),
                };
                self.expect(STok::Assign)?;
                let e = self.expr()?;
                self.expect(STok::Semi)?;
                Stmt::Let(name, e)
            }
            Some(STok::While) => {
                self.i += 1;
                let c = self.expr()?;
                let b = self.block()?;
                Stmt::While(c, b)
            }
            Some(STok::If) => {
                self.i += 1;
                let c = self.expr()?;
                let a = self.block()?;
                let b = if self.eat(&STok::Else) {
                    if self.peek() == Some(&STok::If) { vec![self.stmt()?] } else { self.block()? }
                } else {
                    vec![]
                };
                Stmt::If(c, a, b)
            }
            Some(STok::Return) => {
                self.i += 1;
                let e = self.expr()?;
                self.expect(STok::Semi)?;
                Stmt::Return(e)
            }
            Some(STok::Id(name)) if self.t.get(self.i + 1) == Some(&STok::Assign) => {
                self.i += 2;
                let e = self.expr()?;
                self.expect(STok::Semi)?;
                Stmt::Assign(name, e)
            }
            Some(STok::Id(name)) if self.t.get(self.i + 1) == Some(&STok::LB) => {
                let save = self.i;
                self.i += 2;
                let idx = self.expr()?;
                self.expect(STok::RB)?;
                if self.eat(&STok::Assign) {
                    let e = self.expr()?;
                    self.expect(STok::Semi)?;
                    Stmt::SetIdx(name, idx, e)
                } else {
                    self.i = save;
                    let e = self.expr()?;
                    self.expect(STok::Semi)?;
                    Stmt::Expr(e)
                }
            }
            Some(_) => {
                let e = self.expr()?;
                self.expect(STok::Semi)?;
                Stmt::Expr(e)
            }
            None => return perr("unexpected end"),
        };
        self.leave();
        Ok(s)
    }
    fn level_op(lvl: usize, t: &STok) -> Option<BOp> {
        Some(match (lvl, t) {
            (0, STok::Or) => BOp::Or,
            (1, STok::And) => BOp::And,
            (2, STok::Eq) => BOp::Eq,
            (2, STok::Ne) => BOp::Ne,
            (3, STok::Lt) => BOp::Lt,
            (3, STok::Le) => BOp::Le,
            (3, STok::Gt) => BOp::Gt,
            (3, STok::Ge) => BOp::Ge,
            (4, STok::Plus) => BOp::Add,
            (4, STok::Minus) => BOp::Sub,
            (5, STok::Star) => BOp::Mul,
            (5, STok::Slash) => BOp::Div,
            (5, STok::Percent) => BOp::Mod,
            _ => return None,
        })
    }
    fn bin_level(&mut self, lvl: usize) -> R<SExpr> {
        if lvl == 6 {
            return self.unary();
        }
        let mut l = self.bin_level(lvl + 1)?;
        while let Some(op) = self.peek().and_then(|t| SParser::level_op(lvl, t)) {
            self.i += 1;
            let r = self.bin_level(lvl + 1)?;
            self.nodes += 1;
            if self.nodes > S_MAX_NODES {
                return Err(ToolError::TooLong);
            }
            l = SExpr::Bin(op, Box::new(l), Box::new(r));
        }
        Ok(l)
    }
    fn expr(&mut self) -> R<SExpr> {
        self.enter()?;
        let e = self.bin_level(0)?;
        self.leave();
        Ok(e)
    }
    fn unary(&mut self) -> R<SExpr> {
        self.enter()?;
        let e = match self.peek() {
            Some(STok::Minus) => {
                self.i += 1;
                SExpr::Neg(Box::new(self.unary()?))
            }
            Some(STok::Not) => {
                self.i += 1;
                SExpr::Not(Box::new(self.unary()?))
            }
            _ => self.postfix()?,
        };
        self.leave();
        Ok(e)
    }
    fn postfix(&mut self) -> R<SExpr> {
        let mut e = self.atom()?;
        while self.peek() == Some(&STok::LB) {
            self.i += 1;
            let idx = self.expr()?;
            self.expect(STok::RB)?;
            self.nodes += 1;
            e = SExpr::Index(Box::new(e), Box::new(idx));
        }
        Ok(e)
    }
    fn atom(&mut self) -> R<SExpr> {
        self.nodes += 1;
        if self.nodes > S_MAX_NODES {
            return Err(ToolError::TooLong);
        }
        match self.bump() {
            Some(STok::Int(v)) => Ok(SExpr::Int(v)),
            Some(STok::Id(name)) => {
                if self.eat(&STok::LP) {
                    let mut args = Vec::new();
                    if !self.eat(&STok::RP) {
                        loop {
                            args.push(self.expr()?);
                            if self.eat(&STok::Comma) {
                                continue;
                            }
                            self.expect(STok::RP)?;
                            break;
                        }
                    }
                    if args.len() > 8 {
                        return perr("too many arguments");
                    }
                    Ok(SExpr::Call(name, args))
                } else {
                    Ok(SExpr::Var(name))
                }
            }
            Some(STok::LP) => {
                let e = self.expr()?;
                self.expect(STok::RP)?;
                Ok(e)
            }
            Some(STok::LB) => {
                let mut items = Vec::new();
                if !self.eat(&STok::RB) {
                    loop {
                        items.push(self.expr()?);
                        if self.eat(&STok::Comma) {
                            continue;
                        }
                        self.expect(STok::RB)?;
                        break;
                    }
                }
                Ok(SExpr::List(items))
            }
            Some(_) => perr("unexpected token"),
            None => perr("unexpected end of program"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Val {
    I(i128),
    L(Vec<i128>),
}
enum Flow {
    Next,
    Ret(Val),
}

struct Vm {
    vars: HashMap<String, Val>,
    steps: u64,
    max_steps: u64,
}
impl Vm {
    fn tick(&mut self, n: u64) -> R<()> {
        self.steps = self.steps.saturating_add(n);
        if self.steps > self.max_steps { Err(ToolError::StepLimit) } else { Ok(()) }
    }
    fn live(&self) -> usize {
        self.vars.values().map(|v| if let Val::L(l) = v { l.len() } else { 0 }).sum()
    }
    fn set(&mut self, name: &str, v: Val) -> R<()> {
        if !self.vars.contains_key(name) && self.vars.len() >= S_MAX_VARS {
            return Err(ToolError::MemLimit);
        }
        self.vars.insert(name.to_string(), v);
        if self.live() > S_MAX_LIVE {
            return Err(ToolError::MemLimit);
        }
        Ok(())
    }
    fn int(v: Val) -> R<i128> {
        match v {
            Val::I(i) => Ok(i),
            Val::L(_) => Err(ToolError::Domain("integer expected, got list".into())),
        }
    }
    fn truthy(v: &Val) -> R<bool> {
        match v {
            Val::I(i) => Ok(*i != 0),
            Val::L(_) => Err(ToolError::Domain("condition must be an integer".into())),
        }
    }
    fn eval(&mut self, e: &SExpr) -> R<Val> {
        self.tick(1)?;
        match e {
            SExpr::Int(v) => Ok(Val::I(*v)),
            SExpr::Var(n) => {
                let v = self.vars.get(n).cloned().ok_or_else(|| ToolError::Unknown(format!("variable {n}")))?;
                if let Val::L(l) = &v {
                    self.tick(l.len() as u64)?;
                }
                Ok(v)
            }
            SExpr::List(items) => {
                if items.len() > S_MAX_LIST {
                    return Err(ToolError::MemLimit);
                }
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    out.push(Vm::int(self.eval(it)?)?);
                }
                Ok(Val::L(out))
            }
            SExpr::Index(b, i) => {
                let idx = Vm::int(self.eval(i)?)?;
                let elem = |l: &Vec<i128>| -> R<Val> {
                    if idx < 0 || idx as u128 >= l.len() as u128 {
                        Err(ToolError::Domain("index out of range".into()))
                    } else {
                        Ok(Val::I(l[idx as usize]))
                    }
                };
                if let SExpr::Var(n) = &**b {
                    match self.vars.get(n) {
                        Some(Val::L(l)) => return elem(l),
                        Some(_) => return Err(ToolError::Domain("not a list".into())),
                        None => return Err(ToolError::Unknown(format!("variable {n}"))),
                    }
                }
                match self.eval(b)? {
                    Val::L(l) => elem(&l),
                    _ => Err(ToolError::Domain("not a list".into())),
                }
            }
            SExpr::Neg(a) => Vm::int(self.eval(a)?)?.checked_neg().map(Val::I).ok_or(ToolError::Overflow),
            SExpr::Not(a) => Ok(Val::I(if Vm::truthy(&self.eval(a)?)? { 0 } else { 1 })),
            SExpr::Bin(op, a, b) => {
                if *op == BOp::And || *op == BOp::Or {
                    let l = Vm::truthy(&self.eval(a)?)?;
                    if (*op == BOp::And && !l) || (*op == BOp::Or && l) {
                        return Ok(Val::I(l as i128));
                    }
                    return Ok(Val::I(Vm::truthy(&self.eval(b)?)? as i128));
                }
                let x = Vm::int(self.eval(a)?)?;
                let y = Vm::int(self.eval(b)?)?;
                let r = match op {
                    BOp::Add => x.checked_add(y).ok_or(ToolError::Overflow)?,
                    BOp::Sub => x.checked_sub(y).ok_or(ToolError::Overflow)?,
                    BOp::Mul => x.checked_mul(y).ok_or(ToolError::Overflow)?,
                    BOp::Div => {
                        if y == 0 {
                            return Err(ToolError::DivZero);
                        }
                        x.checked_div_euclid(y).ok_or(ToolError::Overflow)?
                    }
                    BOp::Mod => {
                        if y == 0 {
                            return Err(ToolError::DivZero);
                        }
                        x.checked_rem_euclid(y).ok_or(ToolError::Overflow)?
                    }
                    BOp::Eq => (x == y) as i128,
                    BOp::Ne => (x != y) as i128,
                    BOp::Lt => (x < y) as i128,
                    BOp::Le => (x <= y) as i128,
                    BOp::Gt => (x > y) as i128,
                    BOp::Ge => (x >= y) as i128,
                    BOp::And | BOp::Or => unreachable!(),
                };
                Ok(Val::I(r))
            }
            SExpr::Call(name, args) => self.call(name, args),
        }
    }
    fn call(&mut self, name: &str, args: &[SExpr]) -> R<Val> {
        let arity = |n: usize| -> R<()> { if args.len() == n { Ok(()) } else { perr("wrong argument count") } };
        match name {
            "len" => {
                arity(1)?;
                if let SExpr::Var(n) = &args[0] {
                    return match self.vars.get(n) {
                        Some(Val::L(l)) => Ok(Val::I(l.len() as i128)),
                        Some(_) => Err(ToolError::Domain("not a list".into())),
                        None => Err(ToolError::Unknown(format!("variable {n}"))),
                    };
                }
                match self.eval(&args[0])? {
                    Val::L(l) => Ok(Val::I(l.len() as i128)),
                    _ => Err(ToolError::Domain("not a list".into())),
                }
            }
            "push" => {
                arity(2)?;
                let mut l = match self.eval(&args[0])? {
                    Val::L(l) => l,
                    _ => return Err(ToolError::Domain("not a list".into())),
                };
                let x = Vm::int(self.eval(&args[1])?)?;
                if l.len() >= S_MAX_LIST {
                    return Err(ToolError::MemLimit);
                }
                self.tick(l.len() as u64)?;
                l.push(x);
                Ok(Val::L(l))
            }
            "abs" => {
                arity(1)?;
                Vm::int(self.eval(&args[0])?)?.checked_abs().map(Val::I).ok_or(ToolError::Overflow)
            }
            "min" | "max" => {
                arity(2)?;
                let x = Vm::int(self.eval(&args[0])?)?;
                let y = Vm::int(self.eval(&args[1])?)?;
                Ok(Val::I(if name == "min" { x.min(y) } else { x.max(y) }))
            }
            "pow" => {
                arity(2)?;
                let x = Vm::int(self.eval(&args[0])?)?;
                let y = Vm::int(self.eval(&args[1])?)?;
                if !(0..=200).contains(&y) {
                    return Err(ToolError::Domain("pow exponent must be 0..=200".into()));
                }
                let mut acc: i128 = 1;
                for _ in 0..y {
                    self.tick(1)?;
                    acc = acc.checked_mul(x).ok_or(ToolError::Overflow)?;
                }
                Ok(Val::I(acc))
            }
            _ => Err(ToolError::Unknown(format!("function {name}"))),
        }
    }
    fn exec_block(&mut self, b: &[Stmt]) -> R<Flow> {
        for s in b {
            if let Flow::Ret(v) = self.exec(s)? {
                return Ok(Flow::Ret(v));
            }
        }
        Ok(Flow::Next)
    }
    fn exec(&mut self, s: &Stmt) -> R<Flow> {
        self.tick(1)?;
        match s {
            Stmt::Let(n, e) => {
                let v = self.eval(e)?;
                self.set(n, v)?;
            }
            Stmt::Assign(n, e) => {
                if !self.vars.contains_key(n) {
                    return Err(ToolError::Unknown(format!("variable {n} (use let)")));
                }
                let v = self.eval(e)?;
                self.set(n, v)?;
            }
            Stmt::SetIdx(n, i, e) => {
                let idx = Vm::int(self.eval(i)?)?;
                let x = Vm::int(self.eval(e)?)?;
                match self.vars.get_mut(n) {
                    Some(Val::L(l)) => {
                        if idx < 0 || idx as u128 >= l.len() as u128 {
                            return Err(ToolError::Domain("index out of range".into()));
                        }
                        l[idx as usize] = x;
                    }
                    Some(_) => return Err(ToolError::Domain("not a list".into())),
                    None => return Err(ToolError::Unknown(format!("variable {n}"))),
                }
            }
            Stmt::While(c, body) => loop {
                self.tick(1)?;
                let cv = self.eval(c)?;
                if !Vm::truthy(&cv)? {
                    break;
                }
                if let Flow::Ret(v) = self.exec_block(body)? {
                    return Ok(Flow::Ret(v));
                }
            },
            Stmt::If(c, a, b) => {
                let cv = self.eval(c)?;
                let blk = if Vm::truthy(&cv)? { a } else { b };
                if let Flow::Ret(v) = self.exec_block(blk)? {
                    return Ok(Flow::Ret(v));
                }
            }
            Stmt::Return(e) => return Ok(Flow::Ret(self.eval(e)?)),
            Stmt::Expr(e) => {
                self.eval(e)?;
            }
        }
        Ok(Flow::Next)
    }
}

fn fmt_val(v: &Val) -> String {
    match v {
        Val::I(i) => i.to_string(),
        Val::L(l) => format!("[{}]", l.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Sandbox {
    pub max_steps: u64,
}
impl Default for Sandbox {
    fn default() -> Self {
        Sandbox { max_steps: 200_000 }
    }
}
impl Sandbox {
    /// run a program; returns (result string, steps used)
    pub fn run(&self, src: &str) -> R<(String, u64)> {
        let t = slex(src)?;
        if t.is_empty() {
            return perr("empty program");
        }
        let mut p = SParser { t, i: 0, depth: 0, nodes: 0 };
        let mut prog = Vec::new();
        while p.peek().is_some() {
            prog.push(p.stmt()?);
        }
        let mut vm = Vm { vars: HashMap::new(), steps: 0, max_steps: self.max_steps };
        match vm.exec_block(&prog)? {
            Flow::Ret(v) => Ok((fmt_val(&v), vm.steps)),
            Flow::Next => Err(ToolError::BadInput("program ended without 'return'".into())),
        }
    }
}
impl Tool for Sandbox {
    fn name(&self) -> &str {
        "sandbox"
    }
    fn cost(&self) -> u32 {
        5
    }
    fn call(&self, input: &str) -> R<String> {
        Ok(self.run(input)?.0)
    }
    fn call_metered(&self, input: &str) -> (R<String>, u32) {
        match self.run(input) {
            Ok((s, steps)) => (Ok(s), 1 + (steps / 500).min(100) as u32),
            Err(e) => (Err(e), 1 + (self.max_steps / 500).min(100) as u32 / 4),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// lookup

#[derive(Clone, Debug, Default)]
pub struct LookupTool {
    tables: BTreeMap<String, BTreeMap<String, String>>,
}
impl LookupTool {
    pub fn new() -> Self {
        LookupTool::default()
    }
    pub fn insert(&mut self, table: &str, key: &str, value: &str) {
        self.tables.entry(table.to_string()).or_default().insert(key.to_string(), value.to_string());
    }
    pub fn len(&self) -> usize {
        self.tables.values().map(|t| t.len()).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
impl Tool for LookupTool {
    fn name(&self) -> &str {
        "lookup"
    }
    fn cost(&self) -> u32 {
        1
    }
    fn call(&self, input: &str) -> R<String> {
        let input = input.trim();
        if input.len() > 256 {
            return Err(ToolError::TooLong);
        }
        let (t, k) = input.split_once(['.', ':']).ok_or(ToolError::Parse("table.key expected".into()))?;
        let tab = self.tables.get(t.trim()).ok_or_else(|| ToolError::NotFound(format!("table {t}")))?;
        tab.get(k.trim()).cloned().ok_or_else(|| ToolError::NotFound(format!("key {k}")))
    }
}

// ---------------------------------------------------------------------------------------------
// verifier: input = three lines  `tool` / `spec` / `claimed result`;  output "accept" | "reject: why"

#[derive(Clone, Debug)]
pub struct Verifier {
    lookup: LookupTool,
    /// R39 opt-in: verify "identity" claims with [`Identity::check_strict`] (default false = integer grid only)
    strict_identity: bool,
    /// R51 opt-in: a decimal claim ("3.33") is accepted for "calc"/"units" when it is the CORRECT ROUNDING (half away from zero) of the exact result at
    /// the claim's own number of decimals; truncations and off-by-one-digit claims are still rejected. Default false = exact rational equality only.
    rounded_claims: bool,
    /// R52 opt-in: lookup tables whose claims are compared after surface normalisation (case, surrounding whitespace / quotes / trailing period, inner
    /// whitespace runs, leading article, digit-group commas). Tables NOT listed stay exact (case matters for symbols: Co vs CO).
    norm_tables: Vec<String>,
}
impl Verifier {
    pub fn new(lookup: LookupTool) -> Self {
        Verifier { lookup, strict_identity: false, rounded_claims: false, norm_tables: Vec::new() }
    }
    /// opt-in (R52): compare `lookup` claims after [`Verifier::normalize_claim`] for the listed tables only
    pub fn with_normalized_lookup(mut self, tables: &[&str]) -> Self {
        self.norm_tables = tables.iter().map(|t| t.to_string()).collect();
        self
    }
    /// surface normalisation used by `with_normalized_lookup`: lowercase, strip surrounding quotes/whitespace/trailing periods, collapse whitespace runs,
    /// drop a leading article (the/a/an), remove commas that group digits ("1,000" -> "1000"). Never changes letters or digits otherwise.
    pub fn normalize_claim(s: &str) -> String {
        let t = s.trim().trim_matches(|c: char| c == '"' || c == '\'' || c == '`').trim().trim_end_matches('.').trim().to_lowercase();
        let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
        let t = ["the ", "a ", "an "].iter().find_map(|a| t.strip_prefix(a)).map(|x| x.to_string()).unwrap_or(t);
        let b: Vec<char> = t.chars().collect();
        let mut out = String::with_capacity(t.len());
        for (i, &c) in b.iter().enumerate() {
            let group = c == ',' && i > 0 && i + 3 < b.len() + 0 && b[i - 1].is_ascii_digit() && b.get(i + 1..i + 4).map_or(false, |g| g.iter().all(|d| d.is_ascii_digit())) && b.get(i + 4).map_or(true, |d| !d.is_ascii_digit() || *d == ',');
            if !group {
                out.push(c);
            }
        }
        out
    }
    /// opt-in: accept correctly rounded decimal claims for calc/units (R51); default off
    pub fn with_rounded_claims(mut self, on: bool) -> Self {
        self.rounded_claims = on;
        self
    }
    /// true iff `claim` is a plain decimal (`-?d+.d+`, 1..=12 decimals) equal to `want` rounded half away from zero at that many decimals
    pub fn is_correct_rounding(want: Rat, claim: &str) -> bool {
        let c = claim.trim();
        let body = c.strip_prefix('-').unwrap_or(c);
        let Some((ip, fp)) = body.split_once('.') else { return false };
        if ip.is_empty() || fp.is_empty() || fp.len() > 12 || !ip.bytes().all(|b| b.is_ascii_digit()) || !fp.bytes().all(|b| b.is_ascii_digit()) {
            return false;
        }
        let Ok(got) = Rat::parse(c) else { return false };
        let mut scale: i128 = 1;
        for _ in 0..fp.len() {
            scale *= 10;
        }
        let neg = want.numer() < 0;
        let mag = if neg { want.neg() } else { want };
        let Ok(scaled) = mag.mul(Rat::int(scale)) else { return false };
        let Ok(half) = Rat::new(1, 2) else { return false };
        let Ok(shifted) = scaled.add(half) else { return false };
        let r = shifted.floor();
        let Ok(rounded) = Rat::new(if neg { -r.numer() } else { r.numer() }, scale) else { return false };
        rounded == got
    }
    /// opt-in stricter identity verification (rational + large-magnitude probes); default off
    pub fn with_strict_identity(mut self, on: bool) -> Self {
        self.strict_identity = on;
        self
    }
    fn verdict(ok: bool, why: &str) -> R<(String, u32)> {
        Ok((if ok { "accept".to_string() } else { format!("reject: {why}") }, 1))
    }
    pub fn verify(&self, tool: &str, spec: &str, claim: &str) -> R<(String, u32)> {
        let claim = claim.trim();
        match tool {
            "calc" => {
                let want = Calc::eval_str(spec)?;
                let got = match Rat::parse(claim) {
                    Ok(g) => g,
                    Err(_) => return Verifier::verdict(false, "claim is not a number"),
                };
                if want != got {
                    if self.rounded_claims && Verifier::is_correct_rounding(want, claim) {
                        return Verifier::verdict(true, "");
                    }
                    return Verifier::verdict(false, "exact recompute differs");
                }
                // independent f64 path (different arithmetic) must agree
                let e = parse_expr(spec)?;
                if let Some(f) = eval_f64(&e, &[]) {
                    let g = got.to_f64();
                    if (f - g).abs() > 1e-9 * f.abs().max(1.0) {
                        return Verifier::verdict(false, "f64 cross-check differs");
                    }
                }
                Verifier::verdict(true, "")
            }
            "units" => {
                let want = Rat::parse(&Units.call(spec)?)?;
                let got = match Rat::parse(claim) {
                    Ok(g) => g,
                    Err(_) => return Verifier::verdict(false, "claim is not a number"),
                };
                if want != got {
                    if self.rounded_claims && Verifier::is_correct_rounding(want, claim) {
                        return Verifier::verdict(true, "");
                    }
                    return Verifier::verdict(false, "exact recompute differs");
                }
                // reverse conversion of the claim must give back the original value
                if let Some(rest) = spec.trim().strip_prefix("conv") {
                    if let Some((lhs, to)) = rest.split_once("->") {
                        let (v, from_u) = lhs.trim().split_once(char::is_whitespace).ok_or(ToolError::Parse("quantity".into()))?;
                        let back = Units.call(&format!("conv {} {} -> {}", claim, to.trim(), from_u.trim()))?;
                        if Rat::parse(&back)? != Rat::parse(v)? {
                            return Verifier::verdict(false, "reverse conversion differs");
                        }
                    }
                }
                Verifier::verdict(true, "")
            }
            "csp" => {
                if claim == "unsat" {
                    let sol = Csp::solve_str(spec)?;
                    return Ok((if sol == "unsat" { "accept".into() } else { "reject: a solution exists".into() }, 3));
                }
                match Csp::check_assignment(spec, claim) {
                    Ok(true) => Verifier::verdict(true, ""),
                    Ok(false) => Verifier::verdict(false, "assignment violates the spec"),
                    Err(_) => Verifier::verdict(false, "claim is not a valid assignment"),
                }
            }
            "identity" => {
                // larger, shifted grid than the solver's default
                let big = Identity { lo: -12, hi: 13 };
                let strict = self.strict_identity;
                let run = |id: &Identity| if strict { id.check_strict(spec) } else { id.check(spec) };
                let want = run(&big)?;
                let said_identical = claim == "identical";
                if said_identical {
                    return Verifier::verdict(want == "identical", "found a differing point");
                }
                if let Some(rest) = claim.strip_prefix("differs") {
                    let _ = rest;
                    return Verifier::verdict(want.starts_with("differs") || run(&Identity::default())?.starts_with("differs"), "no differing point exists on the grid");
                }
                Verifier::verdict(false, "unrecognised claim")
            }
            "sandbox" => {
                let (want, steps) = Sandbox::default().run(spec)?;
                Ok((if want == claim { "accept".into() } else { "reject: re-run differs".into() }, 1 + (steps / 500).min(100) as u32))
            }
            "lookup" => {
                let want = self.lookup.call(spec)?;
                let table = spec.trim().split(['.', ':']).next().unwrap_or("").trim();
                let ok = want == claim || (self.norm_tables.iter().any(|t| t == table) && Verifier::normalize_claim(&want) == Verifier::normalize_claim(claim));
                Verifier::verdict(ok, "table value differs")
            }
            other => Err(ToolError::Unknown(format!("verifiable tool {other}"))),
        }
    }
}
impl Tool for Verifier {
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
        match self.verify(t.trim(), s, c) {
            Ok((v, cost)) => (Ok(v), cost),
            Err(e) => (Err(e), 1),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// registry with a deterministic result cache

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RegStats {
    pub calls: u64,
    pub hits: u64,
    pub errors: u64,
    pub cost_spent: u64,
    pub cost_saved: u64,
}
impl RegStats {
    pub fn hit_rate(&self) -> f64 {
        if self.calls == 0 { 0.0 } else { self.hits as f64 / self.calls as f64 }
    }
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub result: R<String>,
    /// cost charged for THIS call (0 on a cache hit)
    pub cost: u32,
    /// cost this call would have been charged without the cache
    pub nominal: u32,
    pub cached: bool,
}

type CacheKey = (usize, String);
type CacheVal = (R<String>, u32);

pub struct Registry {
    tools: Vec<Box<dyn Tool>>,
    cache_on: bool,
    cache: HashMap<CacheKey, CacheVal>,
    order: VecDeque<CacheKey>,
    cap: usize,
    pub stats: RegStats,
}

/// collapse whitespace runs and trim (cache key; tools treat all whitespace alike)
pub fn normalize_input(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

impl Registry {
    pub fn new(cache_on: bool) -> Self {
        Registry { tools: Vec::new(), cache_on, cache: HashMap::new(), order: VecDeque::new(), cap: 8192, stats: RegStats::default() }
    }
    pub fn with_cap(mut self, cap: usize) -> Self {
        self.cap = cap.max(1);
        self
    }
    pub fn register(&mut self, t: Box<dyn Tool>) -> usize {
        self.tools.push(t);
        self.tools.len() - 1
    }
    /// calc, units, csp, identity, sandbox, lookup, verify  (indices 0..=6)
    pub fn with_defaults(lookup: LookupTool, cache_on: bool) -> Self {
        Registry::with_verifier(lookup.clone(), cache_on, Verifier::new(lookup))
    }
    /// same tool set as [`Registry::with_defaults`] but with a caller-configured verifier (e.g. R39 `with_strict_identity(true)`, R51 `with_rounded_claims(true)`)
    pub fn with_verifier(lookup: LookupTool, cache_on: bool, verifier: Verifier) -> Self {
        let mut r = Registry::new(cache_on);
        r.register(Box::new(Calc));
        r.register(Box::new(Units));
        r.register(Box::new(Csp));
        r.register(Box::new(Identity::default()));
        r.register(Box::new(Sandbox::default()));
        r.register(Box::new(lookup));
        r.register(Box::new(verifier));
        r
    }
    pub fn index(&self, name: &str) -> Option<usize> {
        self.tools.iter().position(|t| t.name() == name)
    }
    pub fn names(&self) -> Vec<String> {
        self.tools.iter().map(|t| t.name().to_string()).collect()
    }
    pub fn nominal_cost(&self, idx: usize) -> u32 {
        self.tools[idx].cost()
    }
    pub fn call(&mut self, name: &str, input: &str) -> Outcome {
        match self.index(name) {
            Some(i) => self.call_idx(i, input),
            None => Outcome { result: Err(ToolError::Unknown(format!("tool {name}"))), cost: 0, nominal: 0, cached: false },
        }
    }
    pub fn call_idx(&mut self, i: usize, input: &str) -> Outcome {
        self.stats.calls += 1;
        let cacheable = self.cache_on && self.tools[i].cacheable();
        let key = if cacheable { Some((i, normalize_input(input))) } else { None };
        if let Some(k) = &key {
            if let Some((res, cost)) = self.cache.get(k) {
                self.stats.hits += 1;
                self.stats.cost_saved += *cost as u64;
                if res.is_err() {
                    self.stats.errors += 1;
                }
                return Outcome { result: res.clone(), cost: 0, nominal: *cost, cached: true };
            }
        }
        let (res, cost) = self.tools[i].call_metered(input);
        self.stats.cost_spent += cost as u64;
        if res.is_err() {
            self.stats.errors += 1;
        }
        if let Some(k) = key {
            if self.cache.len() >= self.cap {
                if let Some(old) = self.order.pop_front() {
                    self.cache.remove(&old);
                }
            }
            self.order.push_back(k.clone());
            self.cache.insert(k, (res.clone(), cost));
        }
        Outcome { result: res, cost, nominal: cost, cached: false }
    }
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }
}

// ---------------------------------------------------------------------------------------------
// learned tool selection

/// cheap shape tokens of a task spec (what a lightweight parser of the question could see)
pub fn tool_features(spec: &str) -> Vec<String> {
    let s = spec.trim();
    let mut f = Vec::new();
    let has = |p: &str| s.contains(p);
    if s.starts_with("conv ") || s.starts_with("sum ") || has("->") {
        f.push("sh:arrow".to_string());
    }
    if has("==") && !has(";") {
        f.push("sh:eq2".into());
    }
    if has(";") && has("..") && has(" in ") {
        f.push("sh:csp".into());
    }
    if has("while") || has("return") || has("let ") {
        f.push("sh:prog".into());
    }
    let alpha = s.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let spaces = s.chars().filter(|c| *c == ' ').count();
    if !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit() || b" +-*/%^().,".contains(&c)) {
        f.push("sh:numeric".into());
    }
    if alpha > 0 && spaces >= 4 && !has("==") && !has(";") && !has("->") && !has("while") {
        f.push("sh:prose".into());
    }
    let b = s.as_bytes();
    if !b.is_empty() && !s.contains(' ') && s.contains('.') && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'.' || *c == b'_') {
        f.push("sh:dotkey".into());
    }
    if f.is_empty() {
        f.push("sh:other".into());
    }
    f
}

/// Contextual tool selector. Per (feature, arm) reward sums with OPTIMISTIC cells (an unseen cell is worth
/// `optimism`, a cell with n observations (sum + prior_n*optimism)/(n + prior_n)), pooled over the task's
/// features with weights that favour discriminative features (spread of the observed arm means), plus
/// epsilon exploration. Rewards are whatever the caller defines (the g13 bin uses tool-fit minus a cost penalty).
#[derive(Clone, Debug)]
pub struct ToolSelector {
    pub arms: usize,
    pub eps: f64,
    pub prior_n: f64,
    pub optimism: f64,
    feat: HashMap<String, Vec<(f64, f64)>>,
    rng: Rng,
}
impl ToolSelector {
    pub fn new(arms: usize, eps: f64, prior_n: f64, optimism: f64, seed: u64) -> Self {
        ToolSelector { arms, eps, prior_n, optimism, feat: HashMap::new(), rng: Rng::new(seed) }
    }
    pub fn scores(&self, feats: &[String]) -> Vec<f64> {
        let mut num = vec![0.0; self.arms];
        let mut den = 0.0;
        for f in feats {
            let cells = self.feat.get(f);
            let vals: Vec<f64> = (0..self.arms)
                .map(|a| match cells {
                    Some(v) => (v[a].0 + self.prior_n * self.optimism) / (v[a].1 + self.prior_n),
                    None => self.optimism,
                })
                .collect();
            let w = match cells {
                Some(v) => {
                    let obs: Vec<f64> = (0..self.arms).filter(|a| v[*a].1 >= 1.0).map(|a| v[a].0 / v[a].1).collect();
                    if obs.len() >= 2 {
                        let mx = obs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let mean = obs.iter().sum::<f64>() / obs.len() as f64;
                        0.05 + (mx - mean).max(0.0)
                    } else {
                        0.5
                    }
                }
                None => 0.5,
            };
            for a in 0..self.arms {
                num[a] += w * vals[a];
            }
            den += w;
        }
        if den <= 0.0 {
            return vec![self.optimism; self.arms];
        }
        num.iter().map(|x| x / den).collect()
    }
    pub fn select(&mut self, feats: &[String]) -> usize {
        if self.rng.f64() < self.eps {
            return self.rng.below(self.arms);
        }
        let sc = self.scores(feats);
        let mut best = 0;
        for a in 1..self.arms {
            if sc[a] > sc[best] + 1e-12 {
                best = a;
            }
        }
        best
    }
    /// warm-start: `n` pseudo-observations of mean `value` for (feature, arm), e.g. from a keyword rule
    pub fn prior(&mut self, feat: &str, arm: usize, n: f64, value: f64) {
        let v = self.feat.entry(feat.to_string()).or_insert_with(|| vec![(0.0, 0.0); self.arms]);
        v[arm].0 += n * value;
        v[arm].1 += n;
    }
    pub fn update(&mut self, feats: &[String], arm: usize, reward: f64) {
        for f in feats {
            let v = self.feat.entry(f.clone()).or_insert_with(|| vec![(0.0, 0.0); self.arms]);
            v[arm].0 += reward;
            v[arm].1 += 1.0;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// tests

#[cfg(test)]
mod tests {
    use super::*;

    fn calc(s: &str) -> R<String> {
        Calc.call(s)
    }
    fn ok(s: &str) -> String {
        calc(s).unwrap_or_else(|e| panic!("{s:?} -> {e}"))
    }

    #[test]
    fn parser_precedence_and_associativity() {
        assert_eq!(ok("1+2*3"), "7");
        assert_eq!(ok("(1+2)*3"), "9");
        assert_eq!(ok("-2^2"), "-4");
        assert_eq!(ok("(-2)^2"), "4");
        assert_eq!(ok("2^3^2"), "512"); // right associative
        assert_eq!(ok("2^-1"), "1/2");
        assert_eq!(ok("10-3-2"), "5"); // left associative
        assert_eq!(ok("100/10/5"), "2");
        assert_eq!(ok("--3"), "3");
        assert_eq!(ok("+3"), "3");
        assert_eq!(ok("  7 \t\n * 6 "), "42");
        assert_eq!(ok("7%3+10%4"), "3");
        assert_eq!(ok("-7%3"), "2"); // unary minus binds tighter than %: (-7)%3, euclidean
    }

    #[test]
    fn mod_is_euclidean_and_unary_binds_tighter() {
        // -7 % 3 parses as (-7) % 3 = 2 under rem_euclid; the assertion above documents precedence only
        assert_eq!(ok("(0-7)%3"), "2");
        assert_eq!(ok("7%(0-3)"), "1");
    }

    #[test]
    fn parser_rejects_malformed_input() {
        for bad in ["", "   ", "+", "1+", "(1", "1)", "1 2", "3x", "1..2", "1.", ".5", "2**3", "1,2", "@", "1 = 1", "f(", "abs()", "gcd(1)", "\u{00e9}+1", "1\u{0}+1"] {
            assert!(calc(bad).is_err(), "{bad:?} should fail");
        }
        assert!(matches!(calc(&"1+".repeat(3000)), Err(ToolError::TooLong)));
        assert!(matches!(calc(&format!("{}1{}", "(".repeat(100), ")".repeat(100))), Err(ToolError::TooDeep)));
        assert!(matches!(calc("x+1"), Err(ToolError::Unknown(_))));
        assert!(matches!(calc("foo(1)"), Err(ToolError::Unknown(_))));
    }

    #[test]
    fn long_flat_chain_is_fine_but_nesting_is_bounded() {
        let s = vec!["1"; 1500].join("+");
        assert_eq!(ok(&s), "1500");
        let deep = format!("{}1{}", "(".repeat(30), ")".repeat(30));
        assert_eq!(ok(&deep), "1");
        let neg_chain = format!("{}1", "-".repeat(200));
        assert!(matches!(calc(&neg_chain), Err(ToolError::TooDeep)));
    }

    #[test]
    fn exact_rational_arithmetic() {
        assert_eq!(ok("1/3+1/6"), "1/2");
        assert_eq!(ok("0.1+0.2"), "3/10");
        assert_eq!(ok("(2/3)^-2"), "9/4");
        assert_eq!(ok("1/3*3"), "1");
        assert_eq!(ok("floor(7/2)"), "3");
        assert_eq!(ok("floor(0-7/2)"), "-4");
        assert_eq!(ok("ceil(0-7/2)"), "-3");
        assert_eq!(ok("ceil(7/2)"), "4");
        assert_eq!(ok("gcd(12,18)"), "6");
        assert_eq!(ok("gcd(0,0)"), "0");
        assert_eq!(ok("min(3,1,2)+max(3,1,2)"), "4");
        assert_eq!(ok("abs(0-5/3)"), "5/3");
        assert_eq!(ok("0^0"), "1");
        assert_eq!(ok("123456789.987654321*1000000000"), "123456789987654321");
        assert_eq!(ok("3.25"), "13/4");
    }

    #[test]
    fn overflow_and_division_are_checked() {
        assert!(matches!(calc("10^20*10^20"), Err(ToolError::Overflow)));
        assert!(matches!(calc("2^200"), Err(ToolError::Overflow)));
        assert!(matches!(calc("2^127"), Err(ToolError::Overflow)));
        assert_eq!(ok("2^126"), "85070591730234615865843651857942052864");
        assert!(matches!(calc("170141183460469231731687303715884105727+1"), Err(ToolError::Overflow)));
        assert!(matches!(calc("170141183460469231731687303715884105728"), Err(ToolError::Overflow)));
        assert!(matches!(calc("1/0"), Err(ToolError::DivZero)));
        assert!(matches!(calc("5%0"), Err(ToolError::DivZero)));
        assert!(matches!(calc("0^(0-1)"), Err(ToolError::DivZero)));
        assert!(matches!(calc("2^100000"), Err(ToolError::Domain(_))));
        assert!(matches!(calc("2^(1/2)"), Err(ToolError::Domain(_))));
        assert!(matches!(calc("(1/2)%2"), Err(ToolError::Domain(_))));
        assert!(matches!(calc("0.1234567890123456789012345678901"), Err(ToolError::Overflow))); // >30 fractional digits
        // a chain of multiplications must overflow cleanly, never wrap
        let s = vec!["99999999999"; 8].join("*");
        assert!(matches!(calc(&s), Err(ToolError::Overflow)));
    }

    #[test]
    fn eval_budget_limits_work() {
        let e = parse_expr(&vec!["1"; 100].join("+")).unwrap();
        let mut b = 50u64;
        assert!(matches!(eval_expr(&e, &[], &mut b), Err(ToolError::StepLimit)));
        let mut b = 1000u64;
        assert_eq!(eval_expr(&e, &[], &mut b).unwrap(), Rat::int(100));
    }

    #[test]
    fn rat_roundtrip_and_cmp() {
        for s in ["0", "-5", "7/3", "-7/3", "1/1", "10/4"] {
            let r = Rat::parse(s).unwrap();
            assert_eq!(Rat::parse(&r.to_string()).unwrap(), r);
        }
        assert_eq!(Rat::parse("10/4").unwrap().to_string(), "5/2");
        assert_eq!(Rat::parse("1/-2").unwrap().to_string(), "-1/2");
        assert!(Rat::parse("1/0").is_err());
        assert!(Rat::parse("abc").is_err());
        assert_eq!(Rat::new(1, 3).unwrap().compare(Rat::new(1, 2).unwrap()).unwrap(), Ordering::Less);
        assert_eq!(Rat::new(i128::MAX, 3).unwrap().compare(Rat::new(i128::MAX, 2).unwrap()).unwrap_or(Ordering::Less), Ordering::Less);
    }

    #[test]
    fn units_convert_and_dimension_check() {
        let u = Units;
        assert_eq!(u.call("conv 36 km/h -> m/s").unwrap(), "10");
        assert_eq!(u.call("conv 1 mile -> km").unwrap(), "25146/15625");
        assert_eq!(u.call("conv 2.5 h -> min").unwrap(), "150");
        assert_eq!(u.call("conv 1 N -> kg*m/s^2").unwrap(), "1");
        assert_eq!(u.call("conv 1 J -> kg*m^2/s^2").unwrap(), "1");
        assert_eq!(u.call("conv 1000 g/L -> kg/m^3").unwrap(), "1000");
        assert_eq!(u.call("sum 3 m + 20 cm + 5 mm -> mm").unwrap(), "3205");
        assert_eq!(u.call("dim N").unwrap(), "L=1 M=1 T=-2 K=0 I=0");
        assert_eq!(u.call("dim m/s/s").unwrap(), "L=1 M=0 T=-2 K=0 I=0");
        assert_eq!(u.call("same J | N*m").unwrap(), "true");
        assert_eq!(u.call("same J | N").unwrap(), "false");
        assert!(matches!(u.call("conv 5 m -> s"), Err(ToolError::Dimension(_))));
        assert!(matches!(u.call("sum 3 m + 2 s -> m"), Err(ToolError::Dimension(_))));
        assert!(matches!(u.call("conv 5 parsec -> m"), Err(ToolError::Unknown(_))));
        assert!(matches!(u.call("conv five m -> m"), Err(ToolError::Parse(_))));
        assert!(matches!(u.call("conv 5 m"), Err(ToolError::Parse(_))));
        assert!(matches!(u.call("frobnicate 1"), Err(ToolError::BadInput(_))));
        assert!(matches!(u.call("conv 5 m^99 -> m"), Err(ToolError::Domain(_))));
        assert!(u.call("").is_err());
    }

    #[test]
    fn csp_solves_checks_and_bounds() {
        assert_eq!(Csp::solve_str("x in 0..9; y in 0..9; x+y==10; x*y==21").unwrap(), "x=3,y=7");
        assert_eq!(Csp::solve_str("x in 1..5; y in 1..5; x+y==3; x>y").unwrap(), "x=2,y=1");
        assert_eq!(Csp::solve_str("x in 0..3; x*x==7").unwrap(), "unsat");
        assert_eq!(Csp::solve_str("x in 0..3; 1==2").unwrap(), "unsat");
        assert_eq!(Csp::solve_str("x in 0..9; 10/x==5").unwrap(), "x=2"); // x=0 -> div by zero = violated
        assert!(Csp::check_assignment("x in 0..9; y in 0..9; x+y==10", "y=4,x=6").unwrap());
        assert!(!Csp::check_assignment("x in 0..9; y in 0..9; x+y==10", "x=6,y=5").unwrap());
        assert!(!Csp::check_assignment("x in 0..9; y in 0..9; x+y==10", "x=20,y=-10").unwrap()); // out of domain
        assert!(!Csp::check_assignment("x in 0..9; y in 0..9; x+y==10", "x=6").unwrap_or(false));
        assert!(matches!(Csp::solve_str("x in 0..9; z+1==2"), Err(ToolError::Unknown(_))));
        assert!(Csp::solve_str("x in 5..1").is_err());
        assert!(Csp::solve_str("x in 0..5000").is_err());
        assert!(Csp::solve_str("x in 0..9; x in 0..9").is_err());
        assert!(Csp::solve_str("x+1==2").is_err()); // undeclared
        assert!(Csp::solve_str("").is_err());
        // 8 free variables x 0..99 with an unsatisfiable last constraint => node limit, not a hang
        let mut spec = String::new();
        for v in ["a", "b", "c", "d", "e", "f", "g", "h"] {
            spec += &format!("{v} in 0..99; ");
        }
        spec += "a+b+c+d+e+f+g+h==10000";
        assert!(matches!(Csp::solve_str(&spec), Err(ToolError::StepLimit)));
    }

    #[test]
    fn normalized_lookup_is_per_table_and_never_fuzzy() {
        let mut lk = LookupTool::new();
        lk.insert("capital", "peru", "Lima");
        lk.insert("capital", "netherlands", "The Hague");
        lk.insert("element", "cobalt", "Co");
        lk.insert("pop", "x", "1,234,567");
        let v0 = Verifier::new(lk.clone());
        let v1 = Verifier::new(lk.clone()).with_normalized_lookup(&["capital", "pop"]);
        let all = Verifier::new(lk).with_normalized_lookup(&["capital", "pop", "element"]);
        let acc = |v: &Verifier, spec: &str, c: &str| v.verify("lookup", spec, c).unwrap().0 == "accept";
        assert!(!acc(&v0, "capital.peru", "lima"));
        for c in ["lima", "  LIMA ", "Lima.", "\"Lima\"", "the lima", "li  ma"] {
            assert_eq!(acc(&v1, "capital.peru", c), c != "li  ma", "{c}");
        }
        assert!(acc(&v1, "capital.netherlands", "hague") && acc(&v1, "capital.netherlands", "the hague"));
        assert!(!acc(&v1, "capital.peru", "Lim") && !acc(&v1, "capital.peru", "Lima, Ohio") && !acc(&v1, "capital.peru", "Quito") && !acc(&v1, "capital.peru", "Liam"));
        assert!(acc(&v1, "pop.x", "1234567") && acc(&v1, "pop.x", "1,234,567") && !acc(&v1, "pop.x", "1,234,568") && !acc(&v1, "pop.x", "12,34,567"));
        assert!(acc(&v1, "element.cobalt", "Co") && !acc(&v1, "element.cobalt", "CO") && !acc(&v1, "element.cobalt", "co"), "symbols stay case-exact");
        assert!(acc(&all, "element.cobalt", "CO"), "listing the table opts into case-folding (documented hazard)");
    }

    #[test]
    fn rounded_claims_accept_correct_rounding_only() {
        let v0 = Verifier::new(LookupTool::new());
        let v1 = Verifier::new(LookupTool::new()).with_rounded_claims(true);
        let acc = |v: &Verifier, t: &str, spec: &str, c: &str| v.verify(t, spec, c).unwrap().0 == "accept";
        // 10/3 = 3.3333..., 2/3 = 0.6666..., -5/3 = -1.6666..., 1/8 = 0.125
        assert!(!acc(&v0, "calc", "10/3", "3.33"));
        assert!(acc(&v1, "calc", "10/3", "3.33") && acc(&v1, "calc", "10/3", "3.3333"));
        assert!(!acc(&v1, "calc", "10/3", "3.34") && !acc(&v1, "calc", "10/3", "3.32"));
        assert!(acc(&v1, "calc", "2/3", "0.67") && !acc(&v1, "calc", "2/3", "0.66"), "truncation is rejected");
        assert!(acc(&v1, "calc", "0-5/3", "-1.67") && !acc(&v1, "calc", "0-5/3", "-1.66"));
        assert!(acc(&v1, "calc", "1/8", "0.13") && !acc(&v1, "calc", "1/8", "0.12"), "half rounds away from zero");
        assert!(acc(&v1, "calc", "1/8", "0.125") && acc(&v1, "calc", "1/8", "1/8"), "exact claims still accepted");
        assert!(!acc(&v1, "calc", "10/3", "3") && !acc(&v1, "calc", "10/3", "3.") && !acc(&v1, "calc", "10/3", "abc"));
        assert!(acc(&v1, "units", "conv 100 ft -> yd", "33.33") && !acc(&v1, "units", "conv 100 ft -> yd", "33.34"));
        assert!(!acc(&v0, "units", "conv 100 ft -> yd", "33.33"));
    }

    #[test]
    fn identity_strict_rejects_near_misses() {
        let id = Identity::default();
        // agree on every integer of the grid but not at rationals / large magnitudes: default says identical, strict says differs
        for spec in ["floor(x) == x", "ceil(x+3) == x+3", "min(x,40) == x"] {
            assert_eq!(id.check(spec).unwrap(), "identical", "{spec}");
            assert!(id.check_strict(spec).unwrap().starts_with("differs"), "{spec}");
        }
        // genuine identities stay identical (exact rationals never differ)
        for spec in ["(x+1)^2 == x^2+2*x+1", "floor(x) == 0-ceil(0-x)", "max(x,y)+min(x,y) == x+y", "(x-y)*(x+y) == x^2-y^2"] {
            assert_eq!(id.check_strict(spec).unwrap(), "identical", "{spec}");
        }
        // verifier: opt-in only
        let v0 = Verifier::new(LookupTool::new());
        let v1 = Verifier::new(LookupTool::new()).with_strict_identity(true);
        assert_eq!(v0.verify("identity", "floor(x) == x", "identical").unwrap().0, "accept");
        assert!(v1.verify("identity", "floor(x) == x", "identical").unwrap().0.starts_with("reject"));
        assert_eq!(v1.verify("identity", "floor(x) == x", "differs").unwrap().0, "accept");
        assert_eq!(v1.verify("identity", "(x+1)^2 == x^2+2*x+1", "identical").unwrap().0, "accept");
    }

    #[test]
    fn identity_grid_check() {
        let id = Identity::default();
        assert_eq!(id.call("(x+1)^2 == x^2+2*x+1").unwrap(), "identical");
        assert!(id.call("(x+1)^2 == x^2+1").unwrap().starts_with("differs: x="));
        assert_eq!(id.call("(x-y)*(x+y) == x^2-y^2").unwrap(), "identical");
        assert!(id.call("(x-y)^2 == x^2-y^2").unwrap().starts_with("differs"));
        assert_eq!(id.call("(x^2-1)/(x-1) == x+1").unwrap_or_default().starts_with("differs"), true); // definedness differs at x=1
        assert_eq!(id.call("2+2 == 4").unwrap(), "identical");
        assert!(id.call("x == x == x").is_err());
        assert!(id.call("x < 3").is_err());
        assert!(id.call("a+b+c == c+b+a").is_err()); // 3 variables
        assert!(id.call("1/0 == 1/0").is_err()); // never defined
    }

    #[test]
    fn sandbox_runs_programs() {
        let sb = Sandbox::default();
        let collatz = "let n = 27; let c = 0; while n != 1 { if n % 2 == 0 { n = n / 2; } else { n = 3*n+1; } c = c + 1; } return c;";
        assert_eq!(sb.run(collatz).unwrap().0, "111");
        assert_eq!(sb.run("let s = 0; let i = 1; while i <= 100 { s = s + i; i = i + 1; } return s;").unwrap().0, "5050");
        assert_eq!(sb.run("let a = [3,1,2]; a[0] = 9; a = push(a, len(a)); return a;").unwrap().0, "[9,1,2,3]");
        assert_eq!(sb.run("return pow(2,100) % 1000007;").unwrap().0, (2u128.pow(100) % 1000007).to_string());
        assert_eq!(sb.run("if 1 && 0 || 1 { return 7; } else { return 8; }").unwrap().0, "7");
        assert_eq!(sb.run("if 0 { return 1; } else if 1 { return 2; } else { return 3; }").unwrap().0, "2");
        assert_eq!(sb.run("return -7 / 2;").unwrap().0, "-4"); // euclidean
        assert_eq!(sb.run("return -7 % 3;").unwrap().0, "2");
        assert_eq!(sb.run("return !0 + !5;").unwrap().0, "1");
        assert_eq!(sb.run("let a = []; return len(a);").unwrap().0, "0");
        assert_eq!(sb.run("let a = [1,2,3]; return a[1+1] + a[0];").unwrap().0, "4");
        assert_eq!(sb.run("a(); return 1;").is_err(), true);
    }

    #[test]
    fn sandbox_enforces_limits() {
        let sb = Sandbox::default();
        assert!(matches!(sb.run("while 1 { }"), Err(ToolError::StepLimit)));
        assert!(matches!(sb.run("let i = 0; while 1 { i = i + 1; }"), Err(ToolError::StepLimit)));
        assert!(matches!(Sandbox { max_steps: 50 }.run("let i = 0; while i < 1000 { i = i + 1; } return i;"), Err(ToolError::StepLimit)));
        assert!(matches!(sb.run("let x = 1; while 1 { x = x * 2; }"), Err(ToolError::Overflow)));
        assert!(matches!(sb.run("let a = [0]; while 1 { a = push(a, 1); }"), Err(ToolError::StepLimit | ToolError::MemLimit)));
        assert!(matches!(Sandbox { max_steps: 100_000_000 }.run("let a = [0]; while 1 { a = push(a, 1); }"), Err(ToolError::MemLimit)));
        assert!(matches!(sb.run("return pow(10,100);"), Err(ToolError::Overflow)));
        assert!(matches!(sb.run("return pow(2,1000);"), Err(ToolError::Domain(_))));
        assert!(matches!(sb.run("return 1/0;"), Err(ToolError::DivZero)));
        assert!(matches!(sb.run("return 1%0;"), Err(ToolError::DivZero)));
        assert!(matches!(sb.run("let a=[1]; return a[5];"), Err(ToolError::Domain(_))));
        assert!(matches!(sb.run("let a=[1]; return a[0-1];"), Err(ToolError::Domain(_))));
        assert!(matches!(sb.run("return y;"), Err(ToolError::Unknown(_))));
        assert!(matches!(sb.run("y = 1; return y;"), Err(ToolError::Unknown(_))));
        assert!(matches!(sb.run("let a = 1; return a[0];"), Err(ToolError::Domain(_))));
        assert!(matches!(sb.run("let a = [1]; return a + 1;"), Err(ToolError::Domain(_))));
        assert!(matches!(sb.run("let x = 1;"), Err(ToolError::BadInput(_))));
        assert!(matches!(sb.run(&format!("return {}1{};", "(".repeat(200), ")".repeat(200))), Err(ToolError::TooDeep)));
        assert!(matches!(sb.run(&format!("return {}1;", "-".repeat(200))), Err(ToolError::TooDeep)));
        assert!(matches!(sb.run(&"let a = 1; ".repeat(1000)), Err(ToolError::TooLong)));
        assert!(matches!(sb.run(&"x".repeat(9000)), Err(ToolError::TooLong)));
        assert!(matches!(sb.run(&format!("let a = [{}]; return 1;", vec!["1"; 5000].join(","))), Err(ToolError::TooLong | ToolError::MemLimit)));
        // too many variables
        let many: String = (0..70).map(|i| format!("let v{i} = {i}; ")).collect::<String>() + "return 1;";
        assert!(matches!(sb.run(&many), Err(ToolError::MemLimit)));
        // overflow in negation / abs
        assert!(matches!(sb.run("return -(-170141183460469231731687303715884105727 - 1);"), Err(ToolError::Overflow)));
    }

    #[test]
    fn sandbox_has_no_escape_hatches() {
        let sb = Sandbox::default();
        for evil in [
            "system(\"ls\");",
            "return open(\"/etc/passwd\");",
            "import os; return 1;",
            "return __import__(1);",
            "return exec(1);",
            "return eval(1);",
            "return read(1);",
            "print(1); return 1;",
            "return std::fs::read(1);",
            "let f = fopen(1); return 1;",
            "return \"string\";",
            "return 'c';",
            "return `cmd`;",
            "return $HOME;",
            "return 1; # comment",
            "return 1 // comment",
            "fn f() { return 1; } return f();",
            "return len(1);",
            "return exit(0);",
            "include(\"x\"); return 1;",
            "return 1\u{0};",
            "return \u{202e}1;",
            "return 0x10;",
            "return 1e3;",
            "return 1.5;",
        ] {
            assert!(sb.run(evil).is_err(), "{evil:?} must not run");
        }
        // an unknown call is an error even when it is in a branch that is never taken only if parsed fine -> runtime error
        assert!(matches!(sb.run("if 0 { system(1); } return 1;"), Ok(_)), "dead branch is never evaluated");
        assert!(matches!(sb.run("if 1 { system(1); } return 1;"), Err(ToolError::Unknown(_))));
    }

    #[test]
    fn lookup_and_verifier() {
        let mut lk = LookupTool::new();
        lk.insert("elem", "fe", "26");
        lk.insert("elem", "o", "8");
        assert_eq!(lk.call("elem.fe").unwrap(), "26");
        assert_eq!(lk.call(" elem:o ").unwrap(), "8");
        assert!(matches!(lk.call("elem.zz"), Err(ToolError::NotFound(_))));
        assert!(matches!(lk.call("nope.fe"), Err(ToolError::NotFound(_))));
        assert!(matches!(lk.call("elem"), Err(ToolError::Parse(_))));
        let v = Verifier::new(lk);
        let acc = |t: &str, s: &str, c: &str| v.call(&format!("{t}\n{s}\n{c}")).unwrap();
        // accept the right claim, reject tampered ones, per tool
        assert_eq!(acc("calc", "(17*23+4)/3", "395/3"), "accept");
        assert!(acc("calc", "(17*23+4)/3", "132").starts_with("reject"));
        assert!(acc("calc", "1+1", "two").starts_with("reject"));
        assert_eq!(acc("units", "conv 36 km/h -> m/s", "10"), "accept");
        assert!(acc("units", "conv 36 km/h -> m/s", "11").starts_with("reject"));
        assert_eq!(acc("csp", "x in 0..9; y in 0..9; x+y==10; x*y==21", "y=7,x=3"), "accept");
        assert_eq!(acc("csp", "x in 0..9; y in 0..9; x+y==10", "x=1,y=9"), "accept"); // any valid certificate
        assert!(acc("csp", "x in 0..9; y in 0..9; x+y==10", "x=1,y=1").starts_with("reject"));
        assert_eq!(acc("csp", "x in 0..3; x*x==7", "unsat"), "accept");
        assert!(acc("csp", "x in 0..9; x*x==9", "unsat").starts_with("reject"));
        assert_eq!(acc("identity", "(x+1)^2 == x^2+2*x+1", "identical"), "accept");
        assert!(acc("identity", "(x+1)^2 == x^2+1", "identical").starts_with("reject"));
        assert_eq!(acc("identity", "(x+1)^2 == x^2+1", "differs: x=1"), "accept");
        assert!(acc("identity", "(x+1)^2 == x^2+2*x+1", "differs: x=1").starts_with("reject"));
        assert_eq!(acc("sandbox", "let s=0; let i=1; while i<=10 { s=s+i; i=i+1; } return s;", "55"), "accept");
        assert!(acc("sandbox", "let s=0; let i=1; while i<=10 { s=s+i; i=i+1; } return s;", "56").starts_with("reject"));
        assert_eq!(acc("lookup", "elem.fe", "26"), "accept");
        assert!(acc("lookup", "elem.fe", "27").starts_with("reject"));
        assert!(v.call("bogus\n1\n1").is_err());
        assert!(v.call("calc\n1").is_err());
        assert!(v.call("").is_err());
        // a spec that itself fails to compute is an error, never an accept
        assert!(v.call("calc\n1/0\n1").is_err());
    }

    #[test]
    fn every_tool_is_deterministic_and_total_on_garbage() {
        let mut reg = Registry::with_defaults(LookupTool::new(), false);
        let n = reg.names().len();
        let mut rng = Rng::new(7);
        let alphabet: Vec<char> = "0123456789+-*/%^()., =<>!;:|\n\t{}[]abxyzletwhiren_".chars().collect();
        for _ in 0..3000 {
            let len = rng.below(40);
            let s: String = (0..len).map(|_| alphabet[rng.below(alphabet.len())]).collect();
            for i in 0..n {
                let a = reg.call_idx(i, &s);
                let b = reg.call_idx(i, &s);
                assert_eq!(a.result, b.result, "tool {i} not deterministic on {s:?}");
                assert_eq!(a.cost, b.cost);
            }
        }
    }

    #[test]
    fn wrong_tool_is_an_error_not_a_panic() {
        let specs = [
            ("calc", "(3+4)*2"),
            ("units", "conv 5 km -> m"),
            ("csp", "x in 0..9; x+1==3"),
            ("identity", "x+x == 2*x"),
            ("sandbox", "return 1+1;"),
            ("lookup", "t.k"),
        ];
        let mut reg = Registry::with_defaults(LookupTool::new(), false);
        for (tn, _ts) in &specs {
            for (sn, ss) in &specs {
                let o = reg.call(tn, ss);
                if tn == sn {
                    // own input: only lookup (empty table) may fail
                    if *tn != "lookup" {
                        assert!(o.result.is_ok(), "{tn} on own input {ss:?}: {:?}", o.result);
                    }
                } else if *tn != "lookup" {
                    // another tool's input: calc may legitimately accept a pure number expression
                    if o.result.is_ok() {
                        assert!(*tn == "calc" || *tn == "sandbox", "{tn} accepted {sn} input {ss:?}: {:?}", o.result);
                    }
                }
            }
        }
        assert!(matches!(reg.call("nope", "x").result, Err(ToolError::Unknown(_))));
    }

    #[test]
    fn registry_cache_accounting() {
        let mut lk = LookupTool::new();
        lk.insert("t", "k", "v");
        let mut c = Registry::with_defaults(lk.clone(), true);
        let mut u = Registry::with_defaults(lk, false);
        let inputs = ["1+2", "1 + 2", "3*3", "1+2", "1/0", "1/0", "(5)", "3*3"];
        let mut cost_c = 0;
        let mut cost_u = 0;
        for s in inputs {
            let a = c.call("calc", s);
            let b = u.call("calc", s);
            assert_eq!(a.result, b.result, "the cache must never change a result");
            cost_c += a.cost;
            cost_u += b.cost;
        }
        assert_eq!(u.stats.hits, 0);
        assert_eq!(c.stats.calls, 8);
        assert_eq!(c.stats.hits, 3); // "1 + 2" is a different key from "1+2" (only whitespace RUNS are collapsed)
        assert_eq!(cost_u, 8);
        assert_eq!(cost_c as u64, c.stats.cost_spent);
        assert_eq!(c.stats.cost_saved as u32 + cost_c, cost_u);
        assert!(c.stats.hit_rate() > 0.3);
        // eviction keeps the cache bounded and results correct
        let mut small = Registry::with_defaults(LookupTool::new(), true).with_cap(2);
        for i in 0..10 {
            assert_eq!(small.call("calc", &format!("{i}+1")).result.unwrap(), (i + 1).to_string());
        }
        assert!(small.cache_len() <= 2);
    }

    #[test]
    fn sandbox_metering_charges_for_work() {
        let sb = Sandbox::default();
        let (_, c_small) = sb.call_metered("return 1;");
        let (_, c_big) = sb.call_metered("let i=0; while i<5000 { i=i+1; } return i;");
        assert!(c_big > c_small);
        assert_eq!(c_small, 1);
    }

    #[test]
    fn tool_features_shapes() {
        assert!(tool_features("conv 5 km -> m").contains(&"sh:arrow".to_string()));
        assert!(tool_features("(x+1)^2 == x^2+2*x+1").contains(&"sh:eq2".to_string()));
        assert!(tool_features("x in 0..9; x+1==3").contains(&"sh:csp".to_string()));
        assert!(tool_features("let s=0; while s<3 { s=s+1; } return s;").contains(&"sh:prog".to_string()));
        assert!(tool_features("(3+4)*2").contains(&"sh:numeric".to_string()));
        assert!(tool_features("elem.fe").contains(&"sh:dotkey".to_string()));
        assert!(tool_features("which of these options best describes the situation").contains(&"sh:prose".to_string()));
        assert_eq!(tool_features(""), vec!["sh:other".to_string()]);
    }

    #[test]
    fn selector_learns_tool_and_skip() {
        // arm 0 = calc, arm 1 = none. feature "sh:numeric" -> calc pays 1.0 / none 0.3; "sh:prose" -> calc 0.0 / none 0.7
        let mut s = ToolSelector::new(2, 0.05, 1.0, 0.9, 11);
        let mut rng = Rng::new(3);
        for _ in 0..400 {
            let f = if rng.f64() < 0.5 { vec!["sh:numeric".to_string()] } else { vec!["sh:prose".to_string()] };
            let a = s.select(&f);
            let r = match (f[0].as_str(), a) {
                ("sh:numeric", 0) => 1.0,
                ("sh:numeric", _) => 0.3,
                ("sh:prose", 0) => 0.0,
                _ => 0.7,
            };
            s.update(&f, a, r);
        }
        let mut s2 = s.clone();
        s2.eps = 0.0;
        assert_eq!(s2.select(&["sh:numeric".to_string()]), 0);
        assert_eq!(s2.select(&["sh:prose".to_string()]), 1);
        assert_eq!(s2.scores(&[]).len(), 2);
        assert!(s2.scores(&["sh:numeric".to_string()])[0] > s2.scores(&["sh:numeric".to_string()])[1]);
    }
}
