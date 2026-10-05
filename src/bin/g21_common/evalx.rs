//! g21: a COPY of the shared evaluator code of `neural_mesh::tools` (expression lexer/parser/evaluator, calc, units, csp,
//! identity and the SHARED verifier) with seeded, injectable bugs ("mutants") selected by a thread-local id. Mutant 0 is
//! the unmodified behaviour (checked against the real tools in `g21_verify selfcheck`). Used only for mutation testing
//! of the evaluator and for the bug-injected end-to-end scenarios; `src/tools.rs` itself is untouched.
use std::cell::Cell;
use std::cmp::Ordering;
use std::collections::HashMap;

use neural_mesh::tools::{LookupTool, Rat, Sandbox, Tool, ToolError};

thread_local! { static MUT: Cell<u8> = Cell::new(0); }
pub fn set_mutant(m: u8) {
    MUT.with(|c| c.set(m))
}
pub fn mutant() -> u8 {
    MUT.with(|c| c.get())
}

pub const MUTANTS: &[(u8, &str, &str)] = &[
    (1, "P_unary_tight", "unary minus binds tighter than ^  (-2^2 = 4)"),
    (2, "P_pow_left", "^ is left-associative  (2^3^2 = 64)"),
    (3, "P_addsub_right", "+ and - are right-associative  (10-3-2 = 9)"),
    (4, "P_muldiv_right", "* and / are right-associative  (8/4/2 = 4, 8/2*4 = 1)"),
    (5, "P_flat_prec", "no precedence between + - * /  (left to right: 2+3*4 = 20)"),
    (6, "N_dec_off", "decimal literal with >= 2 fraction digits scaled by 10^(k-1)  (0.25 -> 2.5)"),
    (7, "E_pow_off1", "x^n evaluates x^(n-1) for n >= 3  (off by one)"),
    (8, "E_idiv", "division of two integers truncates when inexact  (7/2 = 3)"),
    (9, "E_round2", "non-integer results printed rounded to 2 decimals  (1/3 -> 0.33)"),
    (10, "E_mod_trunc", "% uses truncated remainder for negative operands"),
    (11, "U_ft", "ft factor x10 (0.3048 -> 3.048 m)"),
    (12, "U_lb", "lb factor digits transposed (0.45359237 -> 0.45359273 kg)"),
    (13, "U_week", "week = 6 days"),
    (14, "U_knot", "knot factor 463/90 instead of 463/900"),
    (15, "U_cm_recip", "cm treated as 100 m (reciprocal factor)"),
    (16, "U_sum_drop", "sum drops its last term"),
    (17, "U_sum_unconv", "sum does not convert its first term"),
    (18, "C_lt_le", "csp: '<' evaluated as '<='"),
    (19, "C_dom_excl", "csp: domain upper bound exclusive (hi-1)"),
    // COMMON-MODE bugs: the same bug is also present in the independent evaluator (so independence cannot help)
    (20, "CM_U_lb", "COMMON-MODE: lb factor digits transposed in the shared AND the independent unit table"),
    (21, "CM_P_addsub_right", "COMMON-MODE: + and - right-associative in the shared parser AND the independent shunting-yard"),
];

type R<T> = Result<T, ToolError>;
fn perr<T>(s: &str) -> R<T> {
    Err(ToolError::Parse(s.to_string()))
}

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
        Cmp::Lt => {
            if mutant() == 18 { o != Ordering::Greater } else { o == Ordering::Less }
        }
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

fn parse_number(s: &str) -> R<Rat> {
    let (ip, fp) = match s.split_once('.') {
        Some((a, b)) => (a, b),
        None => (s, ""),
    };
    if ip.is_empty() || fp.len() > 30 || ip.len() + fp.len() > 37 {
        return Err(ToolError::Overflow);
    }
    let n: i128 = format!("{ip}{fp}").parse().map_err(|_| ToolError::Overflow)?;
    let mut d: i128 = 1;
    let k = if mutant() == 6 && fp.len() >= 2 { fp.len() - 1 } else { fp.len() };
    for _ in 0..k {
        d = d.checked_mul(10).ok_or(ToolError::Overflow)?;
    }
    Rat::new(n, d)
}

fn lex(s: &str) -> R<Vec<Tok>> {
    if s.len() > 4096 {
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
                out.push(Tok::Num(parse_number(&s[st..i])?));
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
        if self.depth > 64 { Err(ToolError::TooDeep) } else { Ok(()) }
    }
    fn expr(&mut self) -> R<Expr> {
        self.enter()?;
        let m = mutant();
        let r = if m == 5 {
            let mut l = self.unary()?;
            loop {
                let op = match self.peek() {
                    Some(Tok::Plus) => Op::Add,
                    Some(Tok::Minus) => Op::Sub,
                    Some(Tok::Star) => Op::Mul,
                    Some(Tok::Slash) => Op::Div,
                    Some(Tok::Percent) => Op::Mod,
                    _ => break,
                };
                self.i += 1;
                let r = self.unary()?;
                l = Expr::Bin(op, Box::new(l), Box::new(r));
            }
            l
        } else if m == 3 || m == 21 {
            let l = self.term()?;
            let op = match self.peek() {
                Some(Tok::Plus) => Some(Op::Add),
                Some(Tok::Minus) => Some(Op::Sub),
                _ => None,
            };
            match op {
                Some(op) => {
                    self.i += 1;
                    let r = self.expr()?;
                    Expr::Bin(op, Box::new(l), Box::new(r))
                }
                None => l,
            }
        } else {
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
            l
        };
        self.depth -= 1;
        Ok(r)
    }
    fn term(&mut self) -> R<Expr> {
        let m = mutant();
        let mut l = self.unary()?;
        if m == 4 {
            let op = match self.peek() {
                Some(Tok::Star) => Some(Op::Mul),
                Some(Tok::Slash) => Some(Op::Div),
                Some(Tok::Percent) => Some(Op::Mod),
                _ => None,
            };
            if let Some(op) = op {
                self.i += 1;
                let r = self.term()?;
                return Ok(Expr::Bin(op, Box::new(l), Box::new(r)));
            }
            return Ok(l);
        }
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
            Some(Tok::Minus) if mutant() != 1 => {
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
        let mut base = self.atom()?;
        if mutant() == 2 {
            while let Some(Tok::Caret) = self.peek() {
                self.i += 1;
                let ex = if let Some(Tok::Minus) = self.peek() {
                    self.i += 1;
                    Expr::Neg(Box::new(self.atom()?))
                } else {
                    self.atom()?
                };
                base = Expr::Bin(Op::Pow, Box::new(base), Box::new(ex));
            }
            return Ok(base);
        }
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
            Some(Tok::Minus) if mutant() == 1 => Ok(Expr::Neg(Box::new(self.atom()?))),
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

pub fn eval_expr(e: &Expr, env: &[(String, Rat)], budget: &mut u64) -> R<Rat> {
    if *budget == 0 {
        return Err(ToolError::StepLimit);
    }
    *budget -= 1;
    match e {
        Expr::Num(r) => Ok(*r),
        Expr::Var(v) => env.iter().find(|(n, _)| n == v).map(|(_, v)| *v).ok_or_else(|| ToolError::Unknown(format!("variable {v}"))),
        Expr::Neg(a) => Ok(eval_expr(a, env, budget)?.neg()),
        Expr::Bin(op, a, b) => {
            let x = eval_expr(a, env, budget)?;
            let y = eval_expr(b, env, budget)?;
            match op {
                Op::Add => x.add(y),
                Op::Sub => x.sub(y),
                Op::Mul => x.mul(y),
                Op::Div => {
                    if mutant() == 8 && x.is_int() && y.is_int() && !y.is_zero() && x.numer() % y.numer() != 0 {
                        return Ok(Rat::int(x.numer() / y.numer()));
                    }
                    x.div(y)
                }
                Op::Mod => {
                    if !x.is_int() || !y.is_int() {
                        return Err(ToolError::Domain("% needs integers".into()));
                    }
                    if y.numer() == 0 {
                        return Err(ToolError::DivZero);
                    }
                    if mutant() == 10 { Ok(Rat::int(x.numer() % y.numer())) } else { Ok(Rat::int(x.numer().rem_euclid(y.numer()))) }
                }
                Op::Pow => {
                    if !y.is_int() {
                        return Err(ToolError::Domain("exponent must be an integer".into()));
                    }
                    let mut e = y.numer();
                    if mutant() == 7 && e >= 3 {
                        e -= 1;
                    }
                    x.pow_i(e)
                }
            }
        }
        Expr::Call(name, args) => {
            let mut vals = Vec::with_capacity(args.len());
            for a in args {
                vals.push(eval_expr(a, env, budget)?);
            }
            match (name.as_str(), vals.as_slice()) {
                ("abs", [v]) => Ok(if v.numer() < 0 { v.neg() } else { *v }),
                ("floor", [v]) => Ok(v.floor()),
                ("ceil", [v]) => Ok(v.ceil()),
                _ => Err(ToolError::Unknown(format!("function {name}"))),
            }
        }
    }
}

/// f64 evaluation on the (possibly mis-parsed) AST: the arithmetic is independent, the parse is shared
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
                _ => return None,
            }
        }
    };
    if v.is_finite() { Some(v) } else { None }
}

fn show(r: Rat) -> String {
    if mutant() == 9 && !r.is_int() {
        format!("{:.2}", r.to_f64())
    } else {
        r.to_string()
    }
}

// ---------------------------------------------------------------------------------------------
// calc

#[derive(Clone, Copy, Debug, Default)]
pub struct MCalc;
impl MCalc {
    pub fn eval_str(s: &str) -> R<Rat> {
        let e = parse_expr(s)?;
        let mut budget = 200_000u64;
        eval_expr(&e, &[], &mut budget)
    }
}
impl Tool for MCalc {
    fn name(&self) -> &str {
        "calc"
    }
    fn cost(&self) -> u32 {
        1
    }
    fn call(&self, input: &str) -> R<String> {
        Ok(show(MCalc::eval_str(input)?))
    }
}

// ---------------------------------------------------------------------------------------------
// units

type Dim = [i8; 5];
const DL: Dim = [1, 0, 0, 0, 0];
const DM: Dim = [0, 1, 0, 0, 0];
const DT: Dim = [0, 0, 1, 0, 0];
const DNONE: Dim = [0; 5];
struct UnitDef {
    name: &'static str,
    fnum: i128,
    fden: i128,
    dim: Dim,
}
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
    UnitDef { name: "L", fnum: 1, fden: 1000, dim: [3, 0, 0, 0, 0] },
    UnitDef { name: "mL", fnum: 1, fden: 1_000_000, dim: [3, 0, 0, 0, 0] },
    UnitDef { name: "knot", fnum: 463, fden: 900, dim: [1, 0, -1, 0, 0] },
];

fn unit_factor(u: &UnitDef) -> (i128, i128) {
    match (mutant(), u.name) {
        (11, "ft") => (3048, 1000),
        (12, "lb") | (20, "lb") => (45_359_273, 100_000_000),
        (13, "week") => (518_400, 1),
        (14, "knot") => (463, 90),
        (15, "cm") => (100, 1),
        _ => (u.fnum, u.fden),
    }
}

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
        let (fnum, fden) = unit_factor(u);
        let f = Rat::new(fnum, fden)?.pow_i(e as i128)?;
        factor = factor.mul(f)?;
        for k in 0..5 {
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

#[derive(Clone, Copy, Debug, Default)]
pub struct MUnits;
impl MUnits {
    fn quantity(term: &str) -> R<(Rat, Rat, Dim)> {
        let term = term.trim();
        let (v, u) = term.split_once(char::is_whitespace).ok_or(ToolError::Parse("quantity needs value and unit".into()))?;
        let val = Rat::parse(v)?;
        let (f, d) = parse_unit_expr(u)?;
        Ok((val, f, d))
    }
}
impl Tool for MUnits {
    fn name(&self) -> &str {
        "units"
    }
    fn cost(&self) -> u32 {
        1
    }
    fn call(&self, input: &str) -> R<String> {
        if input.len() > 4096 {
            return Err(ToolError::TooLong);
        }
        let input = input.trim();
        let (cmd, rest) = input.split_once(char::is_whitespace).unwrap_or((input, ""));
        match cmd {
            "conv" => {
                let (lhs, to) = rest.split_once("->").ok_or(ToolError::Parse("'->' expected".into()))?;
                let (val, ff, fd) = MUnits::quantity(lhs)?;
                let (tf, td) = parse_unit_expr(to)?;
                if fd != td {
                    return Err(ToolError::Dimension("mismatch".into()));
                }
                Ok(show(val.mul(ff)?.div(tf)?))
            }
            "sum" => {
                let (lhs, to) = rest.split_once("->").ok_or(ToolError::Parse("'->' expected".into()))?;
                let (tf, td) = parse_unit_expr(to)?;
                let mut total = Rat::int(0);
                let mut terms: Vec<&str> = lhs.split('+').collect();
                if mutant() == 16 && terms.len() > 1 {
                    terms.pop();
                }
                for (i, term) in terms.iter().enumerate() {
                    let (val, f, d) = MUnits::quantity(term)?;
                    if d != td {
                        return Err(ToolError::Dimension("mismatch".into()));
                    }
                    let contrib = if mutant() == 17 && i == 0 { val } else { val.mul(f)? };
                    total = total.add(contrib)?;
                }
                Ok(show(total.div(tf)?))
            }
            _ => Err(ToolError::BadInput(format!("unknown command {cmd:?}"))),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// csp

struct Problem {
    vars: Vec<(String, i128, i128)>,
    cons: Vec<(Expr, Cmp, Expr)>,
    level: Vec<Option<usize>>,
}
fn is_ident(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty() && (b[0].is_ascii_alphabetic() || b[0] == b'_') && b.iter().all(|c| c.is_ascii_alphanumeric() || *c == b'_')
}
fn parse_problem(spec: &str) -> R<Problem> {
    if spec.len() > 4096 {
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
            let mut hi: i128 = hi.trim().parse().map_err(|_| ToolError::Parse("range bound".into()))?;
            if mutant() == 19 {
                hi -= 1;
            }
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
pub struct MCsp;
impl MCsp {
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
            if !v.is_int() || v.numer() < *lo || v.numer() > *hi {
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
impl Tool for MCsp {
    fn name(&self) -> &str {
        "csp"
    }
    fn cost(&self) -> u32 {
        3
    }
    fn call(&self, input: &str) -> R<String> {
        MCsp::solve_str(input)
    }
}

// ---------------------------------------------------------------------------------------------
// identity

#[derive(Clone, Copy, Debug)]
pub struct MIdentity {
    pub lo: i64,
    pub hi: i64,
}
impl Default for MIdentity {
    fn default() -> Self {
        MIdentity { lo: -8, hi: 8 }
    }
}
impl MIdentity {
    pub fn check(&self, spec: &str) -> R<String> {
        let (l, r) = spec.split_once("==").ok_or(ToolError::Parse("'==' expected".into()))?;
        if r.contains("==") || l.contains('<') || l.contains('>') || l.contains('!') || r.contains('<') || r.contains('>') || r.contains('!') {
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
impl Tool for MIdentity {
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
// shared verifier (copy of tools::Verifier built on the SAME (mutated) evaluator code)

#[derive(Clone, Debug)]
pub struct MVerifier {
    lookup: LookupTool,
}
impl MVerifier {
    pub fn new(lookup: LookupTool) -> Self {
        MVerifier { lookup }
    }
    fn verdict(ok: bool, why: &str) -> R<(String, u32)> {
        Ok((if ok { "accept".to_string() } else { format!("reject: {why}") }, 1))
    }
    pub fn verify(&self, tool: &str, spec: &str, claim: &str) -> R<(String, u32)> {
        let claim = claim.trim();
        match tool {
            "calc" => {
                let want = MCalc::eval_str(spec)?;
                let got = match Rat::parse(claim) {
                    Ok(g) => g,
                    Err(_) => return MVerifier::verdict(false, "claim is not a number"),
                };
                if want != got {
                    return MVerifier::verdict(false, "exact recompute differs");
                }
                let e = parse_expr(spec)?;
                if let Some(f) = eval_f64(&e, &[]) {
                    let g = got.to_f64();
                    if (f - g).abs() > 1e-9 * f.abs().max(1.0) {
                        return MVerifier::verdict(false, "f64 cross-check differs");
                    }
                }
                MVerifier::verdict(true, "")
            }
            "units" => {
                let want = Rat::parse(&MUnits.call(spec)?)?;
                let got = match Rat::parse(claim) {
                    Ok(g) => g,
                    Err(_) => return MVerifier::verdict(false, "claim is not a number"),
                };
                if want != got {
                    return MVerifier::verdict(false, "exact recompute differs");
                }
                if let Some(rest) = spec.trim().strip_prefix("conv") {
                    if let Some((lhs, to)) = rest.split_once("->") {
                        let (v, from_u) = lhs.trim().split_once(char::is_whitespace).ok_or(ToolError::Parse("quantity".into()))?;
                        let back = MUnits.call(&format!("conv {} {} -> {}", claim, to.trim(), from_u.trim()))?;
                        if Rat::parse(&back)? != Rat::parse(v)? {
                            return MVerifier::verdict(false, "reverse conversion differs");
                        }
                    }
                }
                MVerifier::verdict(true, "")
            }
            "csp" => {
                if claim == "unsat" {
                    let sol = MCsp::solve_str(spec)?;
                    return Ok((if sol == "unsat" { "accept".into() } else { "reject: a solution exists".into() }, 3));
                }
                match MCsp::check_assignment(spec, claim) {
                    Ok(true) => MVerifier::verdict(true, ""),
                    Ok(false) => MVerifier::verdict(false, "assignment violates the spec"),
                    Err(_) => MVerifier::verdict(false, "claim is not a valid assignment"),
                }
            }
            "identity" => {
                let big = MIdentity { lo: -12, hi: 13 };
                let want = big.check(spec)?;
                if claim == "identical" {
                    return MVerifier::verdict(want == "identical", "found a differing point");
                }
                if claim.starts_with("differs") {
                    return MVerifier::verdict(want.starts_with("differs") || MIdentity::default().check(spec)?.starts_with("differs"), "no differing point exists on the grid");
                }
                MVerifier::verdict(false, "unrecognised claim")
            }
            "sandbox" => {
                let (want, steps) = Sandbox::default().run(spec)?;
                Ok((if want == claim { "accept".into() } else { "reject: re-run differs".into() }, 1 + (steps / 500).min(100) as u32))
            }
            "lookup" => {
                let want = self.lookup.call(spec)?;
                MVerifier::verdict(want == claim, "table value differs")
            }
            other => Err(ToolError::Unknown(format!("verifiable tool {other}"))),
        }
    }
}
impl Tool for MVerifier {
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
