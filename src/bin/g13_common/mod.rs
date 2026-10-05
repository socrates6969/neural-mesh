//! g13 shared pieces (W14, R&D round 3, TOOLS track): a seeded world of tasks that need EXACT computation, whose
//! canonical formal specs are executed by the REAL tools in `neural_mesh::tools`, plus the pipeline arms
//! (plain routing / verify-only / tool offload with static, learned or oracle tool selection, optional result
//! cache and verified-answer memo).
//!
//! What is simulated (assumptions, stated so nobody reads more into the numbers than is there):
//!  * agents answer UNGATED with a seeded probability that falls with task difficulty (per-agent hidden accuracy);
//!  * a tool-using agent writes a formal tool call; with a seeded (hidden, per-agent, weakly difficulty dependent)
//!    probability the call is the canonical spec, otherwise it is a REAL mutation of the spec string (changed digit /
//!    swapped operator / dropped char) that the REAL tool then runs (often a parse error, sometimes a wrong value);
//!  * the verifier sees the canonical spec with probability `qv` (otherwise the claim is unverifiable);
//!  * ground truth is computed natively in this file (independent of tools.rs).
//! Nothing here makes a model smarter: it measures what exact tools + a checker + routing/learning/caching do GIVEN
//! these assumptions, and where the gain disappears when the assumptions are weakened (`sens` mode).
#![allow(dead_code)]

use std::collections::HashMap;

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::defaults::recommended_router;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::tools::{tool_features, LookupTool, Rat, Registry, ToolSelector};

pub const NA: usize = 6;
pub const NDOM: usize = 6;
pub const MODEL_COST: f64 = 10.0;
pub const MEMO_COST: f64 = 0.2;
pub const TOOL_NAMES: [&str; 6] = ["calc", "units", "csp", "identity", "sandbox", "lookup"];
pub const VERIFY_IDX: usize = 6;
pub const NONE_ARM: usize = 6;
pub const SHAPES: [&str; 7] = ["sh:arrow", "sh:eq2", "sh:csp", "sh:prog", "sh:numeric", "sh:dotkey", "sh:prose"];

pub fn parse_range(args: &[String], key: &str, default: (u64, u64)) -> Vec<u64> {
    let (a, b) = match args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)) {
        Some(r) => {
            let (a, b) = r.split_once('-').expect("range a-b");
            (a.parse::<u64>().unwrap(), b.parse::<u64>().unwrap())
        }
        None => default,
    };
    (a..=b).collect()
}
pub fn arg(args: &[String], key: &str, default: &str) -> String {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}

// ---------------------------------------------------------------------------------------------
// kinds

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Kind {
    Arith,
    Units,
    Csp,
    Sym,
    Prog,
    Lookup,
    Qual,
}
pub const KINDS: [Kind; 7] = [Kind::Arith, Kind::Units, Kind::Csp, Kind::Sym, Kind::Prog, Kind::Lookup, Kind::Qual];
impl Kind {
    pub fn idx(self) -> usize {
        KINDS.iter().position(|k| *k == self).unwrap()
    }
    pub fn label(self) -> &'static str {
        ["arith", "units", "csp", "sym", "prog", "lookup", "qual"][self.idx()]
    }
    /// registry index of the right tool (None = no exact computation needed)
    pub fn tool(self) -> Option<usize> {
        match self {
            Kind::Arith => Some(0),
            Kind::Units => Some(1),
            Kind::Csp => Some(2),
            Kind::Sym => Some(3),
            Kind::Prog => Some(4),
            Kind::Lookup => Some(5),
            Kind::Qual => None,
        }
    }
    pub fn kws(self) -> [&'static str; 3] {
        match self {
            Kind::Arith => ["compute", "evaluate", "total"],
            Kind::Units => ["convert", "express", "units"],
            Kind::Csp => ["satisfy", "constraints", "feasible"],
            Kind::Sym => ["prove", "identity", "simplify"],
            Kind::Prog => ["simulate", "iterate", "trace"],
            Kind::Lookup => ["lookup", "table", "recorded"],
            Kind::Qual => ["explain", "summarize", "discuss"],
        }
    }
}
/// keyword -> tool arm index (NONE_ARM for the prose keywords when `skip_prose`, else the default tool 0 = "always call a tool")
pub fn static_tool(kw: &str, skip_prose: bool) -> usize {
    for k in KINDS {
        if k.kws().contains(&kw) {
            return match k.tool() {
                Some(t) => t,
                None => {
                    if skip_prose {
                        NONE_ARM
                    } else {
                        0
                    }
                }
            };
        }
    }
    if skip_prose { NONE_ARM } else { 0 }
}

// ---------------------------------------------------------------------------------------------
// config

#[derive(Clone, Debug)]
pub struct WorldCfg {
    pub name: &'static str,
    /// 0 = generator A, 1 = generator B (structurally different agent model, error curve, mix, text style)
    pub style: u8,
    pub n: usize,
    pub kind_p: [f64; 7],
    pub qv: f64,
    pub misuse: f64,
    pub shape_noise: f64,
    pub dup: f64,
    pub trust_h: f64,
    pub sens_scale: f64,
    pub form_delta: f64,
}
impl WorldCfg {
    pub fn world_a() -> Self {
        WorldCfg {
            name: "A",
            style: 0,
            n: 600,
            //          arith units csp   sym   prog  lookup qual
            kind_p: [0.22, 0.12, 0.12, 0.12, 0.12, 0.10, 0.20],
            qv: 0.9,
            misuse: 0.30,
            shape_noise: 0.15,
            dup: 0.20,
            trust_h: 0.30,
            sens_scale: 1.0,
            form_delta: 0.0,
        }
    }
    pub fn world_b() -> Self {
        WorldCfg {
            name: "B",
            style: 1,
            n: 600,
            kind_p: [0.10, 0.20, 0.20, 0.06, 0.20, 0.06, 0.18],
            qv: 0.8,
            misuse: 0.25,
            shape_noise: 0.15,
            dup: 0.30,
            trust_h: 0.30,
            sens_scale: 1.0,
            form_delta: 0.0,
        }
    }
    /// control: no task needs exact computation
    pub fn noncompute(mut self) -> Self {
        self.kind_p = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        self
    }
    /// control: many misleading keywords
    pub fn misuse_heavy(mut self) -> Self {
        self.misuse = 0.60;
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Hyper {
    pub eps_g: f64,
    pub eps_e: f64,
    pub pts_g: f64,
    pub warm_k: f64,
    pub sel_eps: f64,
    pub sel_shrink: f64,
    pub sel_opt: f64,
    pub kappa: f64,
    /// pseudo-count of the keyword-rule warm start of the tool selector (0 = off)
    pub kw_prior: f64,
}
impl Default for Hyper {
    fn default() -> Self {
        // FROZEN after tune on seeds 1-8 (see results/g13_tune.json); overridden by `tune` output only for reporting
        Hyper { eps_g: 1.0, eps_e: 0.1, pts_g: 1.0, warm_k: 10.0, sel_eps: 0.02, sel_shrink: 4.0, sel_opt: 0.7, kappa: 0.02, kw_prior: 20.0 }
    }
}

// ---------------------------------------------------------------------------------------------
// small exact rational for native ground truth (independent of tools::Rat)

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Q(pub i128, pub i128);
fn gcd(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}
impl Q {
    pub fn new(n: i128, d: i128) -> Q {
        let (n, d) = if d < 0 { (-n, -d) } else { (n, d) };
        let g = gcd(n, d);
        Q(n / g, d / g)
    }
    fn add(self, o: Q) -> Option<Q> {
        Some(Q::new(self.0.checked_mul(o.1)?.checked_add(o.0.checked_mul(self.1)?)?, self.1.checked_mul(o.1)?))
    }
    fn sub(self, o: Q) -> Option<Q> {
        self.add(Q(-o.0, o.1))
    }
    fn mul(self, o: Q) -> Option<Q> {
        Some(Q::new(self.0.checked_mul(o.0)?, self.1.checked_mul(o.1)?))
    }
    fn div(self, o: Q) -> Option<Q> {
        if o.0 == 0 {
            return None;
        }
        Some(Q::new(self.0.checked_mul(o.1)?, self.1.checked_mul(o.0)?))
    }
    pub fn s(&self) -> String {
        if self.1 == 1 { format!("{}", self.0) } else { format!("{}/{}", self.0, self.1) }
    }
}

// ---------------------------------------------------------------------------------------------
// truth

#[derive(Clone, Debug)]
pub enum CC {
    Lin(Vec<i128>, u8, i128), // coefs (per var), rel 0 '==', 1 '<=', 2 '>=', rhs
    Prod(usize, usize, i128),
    Less(usize, usize),
    Neq(usize, usize),
}

#[derive(Clone, Debug)]
pub enum Truth {
    Q(Q),
    Int(String),
    Cons { cons: Vec<CC>, names: Vec<String>, dom: Vec<(i128, i128)>, sol: Vec<i128> },
    Sym(bool),
    Opt(char),
}

fn cons_hold(cons: &[CC], vals: &[i128]) -> bool {
    cons.iter().all(|c| match c {
        CC::Lin(co, rel, rhs) => {
            let s: i128 = co.iter().zip(vals).map(|(a, b)| a * b).sum();
            match rel {
                0 => s == *rhs,
                1 => s <= *rhs,
                _ => s >= *rhs,
            }
        }
        CC::Prod(i, j, r) => vals[*i] * vals[*j] == *r,
        CC::Less(i, j) => vals[*i] < vals[*j],
        CC::Neq(i, j) => vals[*i] != vals[*j],
    })
}

impl Truth {
    pub fn correct_string(&self) -> String {
        match self {
            Truth::Q(q) => q.s(),
            Truth::Int(s) => s.clone(),
            Truth::Cons { names, sol, .. } => names.iter().zip(sol).map(|(n, v)| format!("{n}={v}")).collect::<Vec<_>>().join(","),
            Truth::Sym(true) => "identical".into(),
            Truth::Sym(false) => "differs: x=0".into(),
            Truth::Opt(c) => c.to_string(),
        }
    }
    pub fn is_correct(&self, ans: &str) -> bool {
        let ans = ans.trim();
        match self {
            Truth::Q(q) => match (Rat::parse(ans), Rat::new(q.0, q.1)) {
                (Ok(a), Ok(b)) => a == b,
                _ => false,
            },
            Truth::Int(s) => ans == s,
            Truth::Cons { cons, names, dom, .. } => {
                if ans == "unsat" {
                    return false;
                }
                let mut vals = vec![None; names.len()];
                for part in ans.split(',') {
                    let Some((n, v)) = part.split_once('=') else { return false };
                    let Some(i) = names.iter().position(|x| x == n.trim()) else { return false };
                    let Ok(v) = v.trim().parse::<i128>() else { return false };
                    if vals[i].is_some() {
                        return false;
                    }
                    vals[i] = Some(v);
                }
                if vals.iter().any(|v| v.is_none()) {
                    return false;
                }
                let vals: Vec<i128> = vals.into_iter().map(|v| v.unwrap()).collect();
                vals.iter().zip(dom).all(|(v, (lo, hi))| v >= lo && v <= hi) && cons_hold(cons, &vals)
            }
            Truth::Sym(b) => ans.starts_with(if *b { "identical" } else { "differs" }),
            Truth::Opt(c) => ans == c.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// generators

fn pick<T: Copy>(r: &mut Rng, xs: &[T]) -> T {
    xs[r.below(xs.len())]
}

struct UnitGroup {
    units: &'static [(&'static str, i128, i128)],
}
const LEN: UnitGroup = UnitGroup { units: &[("m", 1, 1), ("km", 1000, 1), ("cm", 1, 100), ("mm", 1, 1000), ("ft", 3048, 10_000), ("in", 254, 10_000), ("yd", 9144, 10_000)] };
const MASS: UnitGroup = UnitGroup { units: &[("kg", 1, 1), ("g", 1, 1000), ("mg", 1, 1_000_000), ("lb", 45_359_237, 100_000_000), ("t", 1000, 1)] };
const TIME: UnitGroup = UnitGroup { units: &[("s", 1, 1), ("ms", 1, 1000), ("min", 60, 1), ("h", 3600, 1), ("day", 86_400, 1), ("week", 604_800, 1)] };
const SPEED: UnitGroup = UnitGroup { units: &[("m/s", 1, 1), ("km/h", 1000, 3600), ("knot", 463, 900), ("mile/h", 1_609_344, 3_600_000)] };
const VOL: UnitGroup = UnitGroup { units: &[("L", 1, 1000), ("mL", 1, 1_000_000), ("m^3", 1, 1)] };

fn gen_value(r: &mut Rng, d: f64, style: u8) -> (String, Q) {
    let digits = 1.0 + d * if style == 0 { 2.0 } else { 3.0 };
    let mag = 10f64.powf(digits).round() as i128;
    let n = 1 + r.below(mag.max(2) as usize) as i128;
    if r.f64() < (if style == 0 { 0.25 } else { 0.45 }) {
        let tenth = r.below(10) as i128;
        let v = n * 10 + tenth;
        (format!("{}.{}", v / 10, v % 10), Q::new(v, 10))
    } else {
        (n.to_string(), Q::new(n, 1))
    }
}

fn gen_arith(r: &mut Rng, d: f64, style: u8) -> (String, Q) {
    let leaves = if style == 0 { 2 + (d * 6.0).round() as usize } else { 3 + (d * 7.0).round() as usize };
    for _ in 0..200 {
        if let Some((s, q)) = arith_tree(r, leaves, style, d) {
            return (s, q);
        }
    }
    ("1+1".into(), Q(2, 1))
}
fn arith_leaf(r: &mut Rng, style: u8, d: f64) -> (String, Q) {
    let mag = if style == 0 { 20.0 + 80.0 * d } else { 30.0 + 970.0 * d } as usize;
    let n = 1 + r.below(mag.max(2)) as i128;
    let u = r.f64();
    if u < 0.15 {
        let k = 2 + r.below(10) as i128;
        (format!("{k}^2"), Q(k * k, 1))
    } else if u < (if style == 0 { 0.25 } else { 0.40 }) {
        let v = n * 10 + r.below(10) as i128;
        (format!("{}.{}", v / 10, v % 10), Q::new(v, 10))
    } else {
        (n.to_string(), Q(n, 1))
    }
}
fn arith_tree(r: &mut Rng, leaves: usize, style: u8, d: f64) -> Option<(String, Q)> {
    if leaves == 1 {
        return Some(arith_leaf(r, style, d));
    }
    let k = 1 + r.below(leaves - 1);
    let (ls, lv) = arith_tree(r, k, style, d)?;
    let (rs, rv) = arith_tree(r, leaves - k, style, d)?;
    let u = r.f64();
    let (op, v) = if u < 0.30 {
        ('+', lv.add(rv)?)
    } else if u < 0.60 {
        ('-', lv.sub(rv)?)
    } else if u < 0.90 {
        ('*', lv.mul(rv)?)
    } else if rv.0 != 0 {
        ('/', lv.div(rv)?)
    } else {
        ('+', lv.add(rv)?)
    };
    if v.0.abs() > 10i128.pow(30) || v.1 > 10i128.pow(30) {
        return None;
    }
    let wrap = |s: String, leaf: bool| if leaf || !s.contains(['+', '-', '*', '/']) { s } else { format!("({s})") };
    Some((format!("{}{}{}", wrap(ls, k == 1), op, wrap(rs, leaves - k == 1)), v))
}

fn gen_units(r: &mut Rng, d: f64, style: u8) -> (String, Q) {
    let g = match r.below(5) {
        0 => LEN,
        1 => MASS,
        2 => TIME,
        3 => SPEED,
        _ => VOL,
    };
    let f = |i: usize| Q::new(g.units[i].1, g.units[i].2);
    let (a, b) = {
        let a = r.below(g.units.len());
        let mut b = r.below(g.units.len());
        if b == a {
            b = (b + 1) % g.units.len();
        }
        (a, b)
    };
    if r.f64() < 0.65 {
        let (vs, vq) = gen_value(r, d, style);
        let q = vq.mul(f(a)).unwrap().div(f(b)).unwrap();
        (format!("conv {} {} -> {}", vs, g.units[a].0, g.units[b].0), q)
    } else {
        let nt = if d > 0.6 { 3 } else { 2 };
        let mut total = Q(0, 1);
        let mut parts = Vec::new();
        for _ in 0..nt {
            let u = r.below(g.units.len());
            let (vs, vq) = gen_value(r, d, style);
            total = total.add(vq.mul(f(u)).unwrap()).unwrap();
            parts.push(format!("{} {}", vs, g.units[u].0));
        }
        let q = total.div(f(b)).unwrap();
        (format!("sum {} -> {}", parts.join(" + "), g.units[b].0), q)
    }
}

fn gen_csp(r: &mut Rng, d: f64, style: u8) -> (String, Truth) {
    let nv = if style == 0 { 2 + (d * 2.0).round() as usize } else { 3 + (d * 1.0).round() as usize }.min(4);
    let names: Vec<String> = ["x", "y", "z", "w"][..nv].iter().map(|s| s.to_string()).collect();
    let hi_dom = 5 + (d * 9.0).round() as i128;
    let dom: Vec<(i128, i128)> = (0..nv).map(|_| (0, hi_dom)).collect();
    let sol: Vec<i128> = (0..nv).map(|_| r.below(hi_dom as usize + 1) as i128).collect();
    let ncons = 2 + (d * 2.0).round() as usize;
    let mut cons = Vec::new();
    let mut strs = Vec::new();
    let lin_str = |co: &Vec<i128>, names: &Vec<String>| -> String {
        let mut s = String::new();
        for (i, c) in co.iter().enumerate() {
            if *c == 0 {
                continue;
            }
            let body = if c.abs() == 1 { names[i].clone() } else { format!("{}*{}", c.abs(), names[i]) };
            if s.is_empty() {
                s += &(if *c < 0 { format!("-{body}") } else { body });
            } else {
                s += &(if *c < 0 { format!("-{body}") } else { format!("+{body}") });
            }
        }
        if s.is_empty() { "0".into() } else { s }
    };
    for _ in 0..ncons {
        let u = r.f64();
        if u < 0.5 || nv < 2 {
            let mut co: Vec<i128> = (0..nv).map(|_| r.below(5) as i128 - 1).collect(); // -1..3
            if co.iter().all(|c| *c == 0) {
                co[0] = 1;
            }
            let s: i128 = co.iter().zip(&sol).map(|(a, b)| a * b).sum();
            let rel = r.below(3) as u8;
            let rhs = match rel {
                0 => s,
                1 => s + r.below(3) as i128,
                _ => s - r.below(3) as i128,
            };
            let op = ["==", "<=", ">="][rel as usize];
            strs.push(format!("{}{}{}", lin_str(&co, &names), op, rhs));
            cons.push(CC::Lin(co, rel, rhs));
        } else if u < 0.75 {
            let i = r.below(nv);
            let mut j = r.below(nv);
            if j == i {
                j = (j + 1) % nv;
            }
            let rhs = sol[i] * sol[j];
            strs.push(format!("{}*{}=={}", names[i], names[j], rhs));
            cons.push(CC::Prod(i, j, rhs));
        } else {
            let i = r.below(nv);
            let mut j = r.below(nv);
            if j == i {
                j = (j + 1) % nv;
            }
            if sol[i] < sol[j] {
                strs.push(format!("{}<{}", names[i], names[j]));
                cons.push(CC::Less(i, j));
            } else if sol[i] > sol[j] {
                strs.push(format!("{}<{}", names[j], names[i]));
                cons.push(CC::Less(j, i));
            } else {
                strs.push(format!("{}!={}", names[i], names[j]));
                // equal values: != would be violated by the planted solution, use <= instead
                strs.pop();
                strs.push(format!("{}<={}", names[i], names[j]));
                cons.push(CC::Lin(
                    (0..nv).map(|k| if k == i { 1 } else if k == j { -1 } else { 0 }).collect(),
                    1,
                    0,
                ));
            }
        }
    }
    let mut spec = String::new();
    for (n, (lo, hi)) in names.iter().zip(&dom) {
        spec += &format!("{n} in {lo}..{hi}; ");
    }
    spec += &strs.join("; ");
    (spec, Truth::Cons { cons, names, dom, sol })
}

fn poly_string(c: &[i128]) -> String {
    let mut s = String::new();
    for k in (0..c.len()).rev() {
        let v = c[k];
        if v == 0 {
            continue;
        }
        let body = if k == 0 {
            v.abs().to_string()
        } else {
            let x = if k == 1 { "x".to_string() } else { format!("x^{k}") };
            if v.abs() == 1 { x } else { format!("{}*{}", v.abs(), x) }
        };
        if s.is_empty() {
            s += &(if v < 0 { format!("-{body}") } else { body });
        } else {
            s += &(if v < 0 { format!("-{body}") } else { format!("+{body}") });
        }
    }
    if s.is_empty() { "0".into() } else { s }
}
fn poly_mul(a: &[i128], b: &[i128]) -> Vec<i128> {
    let mut out = vec![0i128; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

fn gen_sym(r: &mut Rng, d: f64, style: u8) -> (String, bool) {
    let m = if style == 0 { 2 + (d * 2.0).round() as usize } else { 2 + (d * 3.0).round() as usize };
    let lhs;
    let mut poly: Vec<i128> = vec![1];
    if r.f64() < 0.5 {
        let mut fs = Vec::new();
        for _ in 0..m {
            let mut a = r.below(13) as i128 - 6;
            if a == 0 {
                a = 7;
            }
            fs.push(a);
            poly = poly_mul(&poly, &[a, 1]);
        }
        lhs = fs.iter().map(|a| if *a < 0 { format!("(x{a})") } else { format!("(x+{a})") }).collect::<Vec<_>>().join("*");
    } else {
        let mut a = r.below(9) as i128 - 4;
        if a == 0 {
            a = 5;
        }
        for _ in 0..m {
            poly = poly_mul(&poly, &[a, 1]);
        }
        lhs = if a < 0 { format!("(x{a})^{m}") } else { format!("(x+{a})^{m}") };
    }
    let truth = r.f64() < 0.5;
    if !truth {
        let k = r.below(poly.len());
        poly[k] += if r.f64() < 0.5 { 1 } else { -1 };
    }
    (format!("{} == {}", lhs, poly_string(&poly)), truth)
}

fn gen_prog(r: &mut Rng, d: f64, style: u8) -> (String, String) {
    let scale = if style == 0 { 1.0 } else { 1.6 };
    match r.below(6) {
        0 => {
            let n = 2 + r.below((8.0 + 1500.0 * d * scale) as usize) as u64;
            let (mut x, mut c) = (n, 0u64);
            while x != 1 {
                x = if x % 2 == 0 { x / 2 } else { 3 * x + 1 };
                c += 1;
            }
            (format!("let n = {n}; let c = 0; while n != 1 {{ if n % 2 == 0 {{ n = n / 2; }} else {{ n = 3 * n + 1; }} c = c + 1; }} return c;"), c.to_string())
        }
        1 => {
            let n = 12 + r.below((50.0 + 99_000.0 * d * scale) as usize) as u128;
            let mut m = n * n;
            let mut s = 0u128;
            while m > 0 {
                s += m % 10;
                m /= 10;
            }
            (format!("let m = {n} * {n}; let s = 0; while m > 0 {{ s = s + m % 10; m = m / 10; }} return s;"), s.to_string())
        }
        2 => {
            let b0 = 5 + r.below((40.0 + 9000.0 * d * scale) as usize) as u64;
            let a0 = b0 + 1 + r.below((40.0 + 9000.0 * d * scale) as usize) as u64;
            let (mut a, mut b, mut c) = (a0, b0, 0u64);
            while b != 0 {
                let t = a % b;
                a = b;
                b = t;
                c += 1;
            }
            (format!("let a = {a0}; let b = {b0}; let c = 0; while b != 0 {{ let t = a % b; a = b; b = t; c = c + 1; }} return c;"), c.to_string())
        }
        3 => {
            let n = 10 + r.below((20.0 + 200.0 * d * scale) as usize) as u64;
            let mut c = 0u64;
            for i in 2..=n {
                let mut p = true;
                let mut j = 2;
                while j * j <= i {
                    if i % j == 0 {
                        p = false;
                    }
                    j += 1;
                }
                if p {
                    c += 1;
                }
            }
            (format!("let n = {n}; let c = 0; let i = 2; while i <= n {{ let p = 1; let j = 2; while j * j <= i {{ if i % j == 0 {{ p = 0; }} j = j + 1; }} c = c + p; i = i + 1; }} return c;"), c.to_string())
        }
        4 => {
            let n = 5 + r.below((20.0 + 280.0 * d * scale) as usize) as u64;
            let m = 7 + r.below(990) as u64;
            let mut s = 0u64;
            for i in 1..=n {
                s = (s + i * i) % m;
            }
            (format!("let s = 0; let i = 1; while i <= {n} {{ s = (s + i * i) % {m}; i = i + 1; }} return s;"), s.to_string())
        }
        _ => {
            let n = 5 + r.below((20.0 + 380.0 * d * scale) as usize) as u64;
            let m = 7 + r.below(9990) as u64;
            let (mut a, mut b) = (0u64, 1u64);
            for _ in 0..n {
                let t = (a + b) % m;
                a = b;
                b = t;
            }
            (format!("let a = 0; let b = 1; let i = 0; while i < {n} {{ let t = (a + b) % {m}; a = b; b = t; i = i + 1; }} return a;"), a.to_string())
        }
    }
}

// ---------------------------------------------------------------------------------------------
// agents and world

#[derive(Clone, Debug)]
pub struct Agents {
    pub acc: [[f64; NDOM]; NA],
    pub km: [[f64; 7]; NA],
    pub dsens: [f64; NA],
    pub form: [f64; NA],
    pub fsens: [f64; NA],
}

#[derive(Clone, Debug)]
pub struct TaskRec {
    pub id: u64,
    pub kind: Kind,
    pub dom: usize,
    pub d: f64,
    pub spec: String,
    pub truth: Truth,
    pub kw: String,
    pub rfeat: Vec<String>,
    pub sfeat: Vec<String>,
    pub u_dir: [f64; NA],
    pub u_form: [f64; NA],
    pub u_trust: f64,
    pub u_ver: f64,
    pub salt: u64,
    pub dup: bool,
}

pub struct World {
    pub cfg: WorldCfg,
    pub tasks: Vec<TaskRec>,
    pub lookup: LookupTool,
    pub kwmap: HashMap<String, usize>,
    pub agents: Agents,
}

const SENS_A: [f64; 7] = [0.85, 0.60, 0.90, 0.50, 0.95, 0.50, 0.08];
const SENS_B: [f64; 7] = [0.90, 0.70, 0.80, 0.60, 0.90, 0.40, 0.05];

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl World {
    pub fn curve(&self, d: f64) -> f64 {
        if self.cfg.style == 0 {
            d.powf(1.3)
        } else {
            (sigmoid(7.0 * (d - 0.5)) - sigmoid(-3.5)) / (sigmoid(3.5) - sigmoid(-3.5))
        }
    }
    /// P(correct) of an UNGATED direct answer by agent `a`
    pub fn p_dir(&self, a: usize, t: &TaskRec) -> f64 {
        let ag = &self.agents;
        let ki = t.kind.idx();
        let base = (ag.acc[a][t.dom] + ag.km[a][ki]).clamp(0.05, 0.99);
        let sens = if self.cfg.style == 0 { SENS_A[ki] } else { SENS_B[ki] } * self.cfg.sens_scale;
        let pk = (base * (1.0 - sens * ag.dsens[a] * self.curve(t.d))).clamp(0.01, 0.99);
        match t.kind {
            Kind::Sym => pk + 0.5 * (1.0 - pk),
            _ => pk,
        }
    }
    /// P(the tool call written by agent `a` is the canonical spec)
    pub fn p_form(&self, a: usize, t: &TaskRec) -> f64 {
        let ag = &self.agents;
        (ag.form[a] + self.cfg.form_delta - ag.fsens[a] * self.curve(t.d)).clamp(0.05, 0.995)
    }
}

pub fn gen_world(cfg: &WorldCfg, seed: u64) -> World {
    let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(cfg.style as u64 * 104_729 + 17));
    // ---- agents
    let mut ag = Agents { acc: [[0.0; NDOM]; NA], km: [[0.0; 7]; NA], dsens: [1.0; NA], form: [0.9; NA], fsens: [0.1; NA] };
    for a in 0..NA {
        if cfg.style == 0 {
            for g in 0..NDOM {
                ag.acc[a][g] = 0.62 + 0.35 * r.f64();
            }
            for k in 0..7 {
                ag.km[a][k] = 0.16 * (r.f64() - 0.5);
            }
            let m: f64 = ag.acc[a].iter().sum::<f64>() / NDOM as f64;
            ag.form[a] = (0.80 + 0.5 * (m - 0.8) + 0.08 * (r.f64() - 0.5)).clamp(0.70, 0.98);
            ag.fsens[a] = 0.12;
        } else {
            let s = 0.45 + 0.55 * r.f64();
            let special = r.below(NDOM);
            for g in 0..NDOM {
                ag.acc[a][g] = (0.50 + 0.45 * s + if g == special { 0.15 } else { 0.0 } + 0.04 * (r.f64() - 0.5)).clamp(0.3, 0.98);
            }
            ag.dsens[a] = 0.6 + 0.6 * r.f64();
            ag.form[a] = (0.72 + 0.25 * s + 0.04 * (r.f64() - 0.5)).clamp(0.6, 0.98);
            ag.fsens[a] = 0.05 + 0.25 * r.f64();
        }
    }
    // keyword map: domain token -> believed best agent (70% right)
    let mut kwmap = HashMap::new();
    for g in 0..NDOM {
        let best = (0..NA).fold(0, |b, a| if ag.acc[a][g] > ag.acc[b][g] { a } else { b });
        let pick_a = if r.f64() < 0.7 { best } else { r.below(NA) };
        kwmap.insert(format!("dom{g}"), pick_a);
    }
    // ---- lookup tables
    let mut lk = LookupTool::new();
    let mut tables: Vec<(String, Vec<(String, String)>)> = Vec::new();
    for ti in 0..4 {
        let tn = format!("t{ti}");
        let mut rows = Vec::new();
        for ki in 0..250 {
            let key = format!("k{ki}");
            let val = (100 + r.below(99_900)).to_string();
            lk.insert(&tn, &key, &val);
            rows.push((key, val));
        }
        tables.push((tn, rows));
    }
    let mut w = World { cfg: cfg.clone(), tasks: Vec::new(), lookup: lk, kwmap, agents: ag };
    // ---- tasks
    let mut tasks: Vec<TaskRec> = Vec::with_capacity(cfg.n);
    let ksum: f64 = cfg.kind_p.iter().sum();
    for i in 0..cfg.n {
        let mut mk_draws = |r: &mut Rng| {
            let mut u_dir = [0.0; NA];
            let mut u_form = [0.0; NA];
            for a in 0..NA {
                u_dir[a] = r.f64();
                u_form[a] = r.f64();
            }
            (u_dir, u_form, r.f64(), r.f64(), r.next_u64())
        };
        let (u_dir, u_form, u_trust, u_ver, salt) = mk_draws(&mut r);
        if i > 8 && r.f64() < cfg.dup {
            // exact repeat of an earlier question (popularity skewed to recent ones)
            let back = 1 + ((r.f64().powi(2)) * (i.min(120) as f64 - 1.0)) as usize;
            let mut t = tasks[i - back].clone();
            t.id = i as u64;
            t.u_dir = u_dir;
            t.u_form = u_form;
            t.u_trust = u_trust;
            t.u_ver = u_ver;
            t.salt = salt;
            t.dup = true;
            tasks.push(t);
            continue;
        }
        // kind
        let mut u = r.f64() * ksum;
        let mut kind = Kind::Qual;
        for (k, p) in KINDS.iter().zip(cfg.kind_p.iter()) {
            if u < *p {
                kind = *k;
                break;
            }
            u -= *p;
        }
        let d = if cfg.style == 0 { r.f64() } else { r.f64().powf(0.7) };
        let dom = r.below(NDOM);
        let (spec, truth) = match kind {
            Kind::Arith => {
                let (s, q) = gen_arith(&mut r, d, cfg.style);
                (s, Truth::Q(q))
            }
            Kind::Units => {
                let (s, q) = gen_units(&mut r, d, cfg.style);
                (s, Truth::Q(q))
            }
            Kind::Csp => gen_csp(&mut r, d, cfg.style),
            Kind::Sym => {
                let (s, b) = gen_sym(&mut r, d, cfg.style);
                (s, Truth::Sym(b))
            }
            Kind::Prog => {
                let (s, v) = gen_prog(&mut r, d, cfg.style);
                (s, Truth::Int(v))
            }
            Kind::Lookup => {
                let (tn, rows) = &tables[r.below(tables.len())];
                let (k, v) = &rows[r.below(rows.len())];
                (format!("{tn}.{k}"), Truth::Int(v.clone()))
            }
            Kind::Qual => {
                let topic = r.below(500);
                let case = r.below(500);
                (
                    format!("explain which of the following four options best describes the main trade off in topic {topic} for case {case}"),
                    Truth::Opt(['A', 'B', 'C', 'D'][r.below(4)]),
                )
            }
        };
        // keyword (possibly misleading) and features
        let kw_kind = if r.f64() < cfg.misuse {
            let mut k = KINDS[r.below(7)];
            while k == kind {
                k = KINDS[r.below(7)];
            }
            k
        } else {
            kind
        };
        let kw = kw_kind.kws()[r.below(3)].to_string();
        let rfeat = vec![format!("dom{dom}"), kw.clone(), format!("nz{}", r.below(8))];
        let mut sfeat = if r.f64() < cfg.shape_noise { vec![SHAPES[r.below(SHAPES.len())].to_string()] } else { tool_features(&spec) };
        sfeat.push(kw.clone());
        tasks.push(TaskRec { id: i as u64, kind, dom, d, spec, truth, kw, rfeat, sfeat, u_dir, u_form, u_trust, u_ver, salt, dup: false });
    }
    w.tasks = tasks;
    w
}

// ---------------------------------------------------------------------------------------------
// answers, mutation

pub fn wrong_answer(t: &TaskRec, salt: u64) -> String {
    let mut r = Rng::new(salt ^ 0xA5A5);
    match &t.truth {
        Truth::Q(q) => {
            let k = [1i128, -1, 2, -2, 10, -10, 3][r.below(7)];
            let out = if r.f64() < 0.5 { Q::new(q.0 + k * q.1, q.1) } else { Q::new(q.0 + k, q.1.max(1)) };
            if out == *q { Q::new(q.0 + q.1, q.1).s() } else { out.s() }
        }
        Truth::Int(s) => {
            let v: i128 = s.parse().unwrap_or(0);
            let k = [1i128, -1, 2, -2, 3, 7, -5, 10][r.below(8)];
            (v + k).to_string()
        }
        Truth::Cons { cons, names, dom, sol } => {
            for _ in 0..30 {
                let mut v = sol.clone();
                let i = r.below(v.len());
                v[i] += if r.f64() < 0.5 { 1 } else { -1 };
                if v[i] < dom[i].0 || v[i] > dom[i].1 {
                    continue;
                }
                if !cons_hold(cons, &v) {
                    return names.iter().zip(&v).map(|(n, x)| format!("{n}={x}")).collect::<Vec<_>>().join(",");
                }
            }
            "unsat".into()
        }
        Truth::Sym(b) => {
            if *b { "differs: x=0".into() } else { "identical".into() }
        }
        Truth::Opt(c) => {
            let opts = ['A', 'B', 'C', 'D'];
            let mut o = opts[r.below(4)];
            while o == *c {
                o = opts[r.below(4)];
            }
            o.to_string()
        }
    }
}

/// a REAL mutation of a spec string (what a mis-formulating agent hands to the tool)
pub fn mutate(spec: &str, salt: u64) -> String {
    let mut r = Rng::new(salt ^ 0x5151);
    let chars: Vec<char> = spec.chars().collect();
    let digits: Vec<usize> = (0..chars.len()).filter(|i| chars[*i].is_ascii_digit()).collect();
    let ops: Vec<usize> = (0..chars.len()).filter(|i| "+-*/".contains(chars[*i])).collect();
    let u = r.f64();
    let mut out = chars.clone();
    if (u < 0.6 && !digits.is_empty()) || (ops.is_empty() && !digits.is_empty()) {
        let i = digits[r.below(digits.len())];
        let old = chars[i].to_digit(10).unwrap();
        let nd = (old + 1 + r.below(9) as u32) % 10;
        out[i] = char::from_digit(nd, 10).unwrap();
    } else if u < 0.85 && !ops.is_empty() {
        let i = ops[r.below(ops.len())];
        out[i] = match chars[i] {
            '+' => '-',
            '-' => '+',
            '*' => '+',
            _ => '*',
        };
    } else if chars.len() > 2 {
        out.remove(r.below(chars.len()));
    }
    out.into_iter().collect()
}

pub fn forced_input(t: &TaskRec, tool: usize) -> String {
    if t.kind == Kind::Qual && tool == 0 {
        format!("{}*3+1", 2 + t.salt % 17)
    } else {
        t.spec.clone()
    }
}

// ---------------------------------------------------------------------------------------------
// routers

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rt {
    Keyword,
    Eps,
    Pts,
    HebR,
    Warm,
    Safe,
}
pub const ROUTERS: [(Rt, &str); 6] = [
    (Rt::Keyword, "keyword"),
    (Rt::Eps, "eps-greedy"),
    (Rt::Pts, "pooled-thompson"),
    (Rt::HebR, "hebbian-robust"),
    (Rt::Warm, "warm(hebr)"),
    (Rt::Safe, "guarded-safe"),
];
pub fn mk_router(rt: Rt, w: &World, h: &Hyper) -> Box<dyn Router> {
    match rt {
        Rt::Keyword => Box::new(KeywordRouter::new(w.kwmap.clone(), NA)),
        Rt::Eps => Box::new(EpsGreedyRouter::new(NA, h.eps_g, h.eps_e)),
        Rt::Pts => Box::new(PooledThompsonRouter::new(NA, h.pts_g)),
        Rt::HebR => Box::new(HebbianRouter::new(NA, heb_robust_cfg())),
        Rt::Warm => Box::new(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), w.kwmap.clone(), NA, h.warm_k).with_evidence(Evidence::MeanFeature)),
        Rt::Safe => Box::new(recommended_router(w.kwmap.clone(), NA)),
    }
}

// ---------------------------------------------------------------------------------------------
// arms

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelPol {
    /// every task gets a tool from the keyword rule; prose keywords default to calc
    Always,
    /// keyword rule, prose keywords -> no tool
    Static,
    /// the true kind's tool (none for qual)
    Oracle,
    Learned,
    /// always a WRONG tool (misuse control)
    WrongTool,
}
#[derive(Clone, Debug)]
pub enum Pipe {
    Plain,
    /// direct answer, verifier checks, one retry with another agent
    VerifyOnly,
    /// cost-matched reference: three distinct agents answer ungated, majority of identical answers wins (else the first)
    Vote3,
    Tools { sel: SelPol, verify: bool, cache: bool, memo: bool },
}
#[derive(Clone, Debug)]
pub struct ArmSpec {
    pub name: String,
    pub router: Rt,
    pub pipe: Pipe,
}
impl ArmSpec {
    pub fn plain(rt: Rt, name: &str) -> Self {
        ArmSpec { name: name.into(), router: rt, pipe: Pipe::Plain }
    }
    pub fn tools(name: &str, rt: Rt, sel: SelPol, verify: bool, cache: bool, memo: bool) -> Self {
        ArmSpec { name: name.into(), router: rt, pipe: Pipe::Tools { sel, verify, cache, memo } }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RunStats {
    pub correct: f64,
    pub correct_compute: f64,
    pub correct_qual: f64,
    pub cost: f64,
    pub cost_nominal: f64,
    pub tool_calls: f64,
    pub verify_calls: f64,
    pub verify_reject: f64,
    pub cache_hit: f64,
    pub cache_saved: f64,
    pub memo_hit: f64,
    pub skip_qual: f64,
    pub right_tool: f64,
    pub tool_used_compute: f64,
    pub n_compute: usize,
    pub n_qual: usize,
    pub correct_vec: Vec<u8>,
}

enum Att {
    Accepted { ans: String, verified: bool },
    Failed,
    Rejected,
    Ignored,
}

struct Ctx<'a> {
    w: &'a World,
    reg: Registry,
    cost: f64,
    nominal: f64,
    tool_calls: u64,
    verify_calls: u64,
    verify_reject: u64,
}

fn direct(w: &World, t: &TaskRec, a: usize) -> String {
    if t.u_dir[a] < w.p_dir(a, t) { t.truth.correct_string() } else { wrong_answer(t, t.salt ^ ((a as u64 + 1) * 0x9E37)) }
}

fn attempt(cx: &mut Ctx, t: &TaskRec, a: usize, tool: usize, verify: bool) -> Att {
    cx.cost += MODEL_COST;
    cx.nominal += MODEL_COST;
    let right = t.kind.tool() == Some(tool);
    let input = if right {
        if t.u_form[a] < cx.w.p_form(a, t) { t.spec.clone() } else { mutate(&t.spec, t.salt ^ ((a as u64 + 1) * 0x7F4A)) }
    } else {
        forced_input(t, tool)
    };
    cx.tool_calls += 1;
    let o = cx.reg.call_idx(tool, &input);
    cx.cost += o.cost as f64;
    cx.nominal += o.nominal as f64;
    let out = match o.result {
        Ok(s) => s,
        Err(_) => return Att::Failed,
    };
    if t.kind == Kind::Qual {
        // nothing to verify; the agent may (wrongly) trust a tool result that has nothing to do with the question
        return if t.u_trust < cx.w.cfg.trust_h { Att::Accepted { ans: format!("tool:{out}"), verified: false } } else { Att::Ignored };
    }
    if verify && t.u_ver < cx.w.cfg.qv {
        let vi = format!("{}\n{}\n{}", TOOL_NAMES[tool], t.spec, out);
        cx.verify_calls += 1;
        let v = cx.reg.call_idx(VERIFY_IDX, &vi);
        cx.cost += v.cost as f64;
        cx.nominal += v.nominal as f64;
        match v.result {
            Ok(s) if s == "accept" => Att::Accepted { ans: out, verified: true },
            _ => {
                cx.verify_reject += 1;
                Att::Rejected
            }
        }
    } else {
        Att::Accepted { ans: out, verified: false }
    }
}

pub fn run_arm(w: &World, arm: &ArmSpec, h: &Hyper, seed: u64) -> RunStats {
    let cands: Vec<usize> = (0..NA).collect();
    let mut router = mk_router(arm.router, w, h);
    let mut rng = Rng::new(seed.wrapping_mul(31).wrapping_add(5));
    let (cache, memo_on) = match &arm.pipe {
        Pipe::Tools { cache, memo, .. } => (*cache, *memo),
        _ => (false, false),
    };
    let mut cx = Ctx { w, reg: Registry::with_defaults(w.lookup.clone(), cache), cost: 0.0, nominal: 0.0, tool_calls: 0, verify_calls: 0, verify_reject: 0 };
    let mut selector = ToolSelector::new(7, h.sel_eps, h.sel_shrink, h.sel_opt, seed.wrapping_mul(17).wrapping_add(3));
    if h.kw_prior > 0.0 {
        for k in KINDS {
            for kw in k.kws() {
                selector.prior(kw, static_tool(kw, true), h.kw_prior, 0.8);
            }
        }
    }
    let mut memo: HashMap<String, String> = HashMap::new();
    let mut st = RunStats::default();
    let (mut c_all, mut c_comp, mut c_qual) = (0.0, 0.0, 0.0);
    let (mut memo_hits, mut skip_q, mut right_c, mut used_c) = (0u64, 0u64, 0u64, 0u64);
    for t in &w.tasks {
        let is_q = t.kind == Kind::Qual;
        if is_q {
            st.n_qual += 1;
        } else {
            st.n_compute += 1;
        }
        let task = Task { id: t.id, task_type: None, features: t.rfeat.clone() };
        let cost_before = cx.cost;
        let nominal_before = cx.nominal;
        // --- verified-answer memo
        if memo_on {
            if let Some(ans) = memo.get(&t.spec) {
                let ok = t.truth.is_correct(ans);
                cx.cost += MEMO_COST;
                cx.nominal += MEMO_COST;
                memo_hits += 1;
                if ok {
                    c_all += 1.0;
                    if is_q { c_qual += 1.0 } else { c_comp += 1.0 }
                }
                if is_q {
                    skip_q += 1;
                } else {
                    used_c += 1;
                    right_c += 1;
                }
                st.correct_vec.push(ok as u8);
                continue;
            }
        }
        let a1 = router.select(&task, &cands, &mut rng);
        let ok: bool;
        // reward signal for the learned tool selector: tool FIT (verified accept = 1, accepted after one retry = 0.7,
        // failed / rejected twice = 0), ground-truth correctness where nothing could be verified
        let mut sel_score = 0.0f64;
        match &arm.pipe {
            Pipe::Plain => {
                let ans = direct(w, t, a1);
                cx.cost += MODEL_COST;
                cx.nominal += MODEL_COST;
                ok = t.truth.is_correct(&ans);
                router.update(&task, a1, ok as u8 as f64);
                if is_q {
                    skip_q += 1;
                }
            }
            Pipe::Vote3 => {
                let a2 = {
                    let c2: Vec<usize> = cands.iter().cloned().filter(|x| *x != a1).collect();
                    router.select(&task, &c2, &mut rng)
                };
                let a3 = {
                    let c3: Vec<usize> = cands.iter().cloned().filter(|x| *x != a1 && *x != a2).collect();
                    router.select(&task, &c3, &mut rng)
                };
                let ans: Vec<String> = [a1, a2, a3].iter().map(|a| direct(w, t, *a)).collect();
                cx.cost += 3.0 * MODEL_COST;
                cx.nominal += 3.0 * MODEL_COST;
                let fin = if ans[1] == ans[2] { ans[1].clone() } else { ans[0].clone() };
                ok = t.truth.is_correct(&fin);
                for (i, a) in [a1, a2, a3].iter().enumerate() {
                    router.update(&task, *a, t.truth.is_correct(&ans[i]) as u8 as f64);
                }
                if is_q {
                    skip_q += 1;
                }
            }
            Pipe::VerifyOnly => {
                let ans = direct(w, t, a1);
                cx.cost += MODEL_COST;
                cx.nominal += MODEL_COST;
                let mut fin = ans.clone();
                let mut rejected = false;
                if let (Some(tool), true) = (t.kind.tool(), t.u_ver < w.cfg.qv) {
                    let vi = format!("{}\n{}\n{}", TOOL_NAMES[tool], t.spec, ans);
                    cx.verify_calls += 1;
                    let v = cx.reg.call_idx(VERIFY_IDX, &vi);
                    cx.cost += v.cost as f64;
                    cx.nominal += v.nominal as f64;
                    if !matches!(&v.result, Ok(s) if s == "accept") {
                        cx.verify_reject += 1;
                        rejected = true;
                    }
                }
                if rejected {
                    router.update(&task, a1, 0.0);
                    let c2: Vec<usize> = cands.iter().cloned().filter(|x| *x != a1).collect();
                    let a2 = router.select(&task, &c2, &mut rng);
                    fin = direct(w, t, a2);
                    cx.cost += MODEL_COST;
                    cx.nominal += MODEL_COST;
                    ok = t.truth.is_correct(&fin);
                    router.update(&task, a2, ok as u8 as f64);
                } else {
                    ok = t.truth.is_correct(&fin);
                    router.update(&task, a1, ok as u8 as f64);
                }
                if is_q {
                    skip_q += 1;
                }
            }
            Pipe::Tools { sel, verify, .. } => {
                // ---- tool choice
                let kw = t.kw.as_str();
                let choice = match sel {
                    SelPol::Always => static_tool(kw, false),
                    SelPol::Static => static_tool(kw, true),
                    SelPol::Oracle => t.kind.tool().unwrap_or(NONE_ARM),
                    SelPol::Learned => selector.select(&t.sfeat),
                    SelPol::WrongTool => match t.kind.tool() {
                        Some(r) => (r + 1 + (t.salt % 5) as usize) % 6,
                        None => (t.salt % 6) as usize,
                    },
                };
                let mut selector_cost_before = cx.nominal;
                let _ = &mut selector_cost_before;
                if choice == NONE_ARM {
                    let ans = direct(w, t, a1);
                    cx.cost += MODEL_COST;
                    cx.nominal += MODEL_COST;
                    ok = t.truth.is_correct(&ans);
                    sel_score = ok as u8 as f64;
                    router.update(&task, a1, ok as u8 as f64);
                    if is_q {
                        skip_q += 1;
                    }
                } else {
                    if !is_q {
                        used_c += 1;
                        if t.kind.tool() == Some(choice) {
                            right_c += 1;
                        }
                    }
                    let fin: String;
                    let mut credited: Vec<(usize, bool)> = Vec::new();
                    let mut ver_ok = false;
                    let mut fit_first = None::<(bool, bool)>; // (accepted, verified)
                    let mut fit_second = None::<(bool, bool)>;
                    let mut ignored = false;
                    match attempt(&mut cx, t, a1, choice, *verify) {
                        Att::Accepted { ans, verified } => {
                            ver_ok = verified;
                            fit_first = Some((true, verified));
                            fin = ans;
                            credited.push((a1, true));
                        }
                        Att::Ignored => {
                            ignored = true;
                            fin = direct(w, t, a1);
                            cx.cost += MODEL_COST;
                            cx.nominal += MODEL_COST;
                            credited.push((a1, true));
                        }
                        Att::Failed | Att::Rejected => {
                            credited.push((a1, false));
                            let c2: Vec<usize> = cands.iter().cloned().filter(|x| *x != a1).collect();
                            let a2 = router.select(&task, &c2, &mut rng);
                            match attempt(&mut cx, t, a2, choice, *verify) {
                                Att::Accepted { ans, verified } => {
                                    ver_ok = verified;
                                    fit_second = Some((true, verified));
                                    fin = ans;
                                    credited.push((a2, true));
                                }
                                _ => {
                                    credited.push((a2, false));
                                    fin = direct(w, t, a1);
                                    cx.cost += MODEL_COST;
                                    cx.nominal += MODEL_COST;
                                    // the fallback answer is a1's own
                                    credited[0].1 = true;
                                }
                            }
                        }
                    }
                    ok = t.truth.is_correct(&fin);
                    sel_score = if ignored {
                        ok as u8 as f64
                    } else if let Some((_, v)) = fit_first {
                        if v { 1.0 } else { ok as u8 as f64 }
                    } else if let Some((_, v)) = fit_second {
                        0.7 * if v { 1.0 } else { ok as u8 as f64 }
                    } else {
                        0.0
                    };
                    // blend tool-fit with the ground-truth outcome (so a good tool is credited for its fallback too)
                    sel_score = 0.5 * sel_score + 0.5 * (ok as u8 as f64);
                    for (ag, produced) in &credited {
                        router.update(&task, *ag, if *produced { ok as u8 as f64 } else { 0.0 });
                    }
                    if memo_on && ver_ok && ok {
                        memo.insert(t.spec.clone(), fin.clone());
                    }
                }
                if *sel == SelPol::Learned {
                    let spent = (cx.nominal - nominal_before) / MODEL_COST;
                    let r = sel_score - h.kappa * spent;
                    selector.update(&t.sfeat, choice, r);
                }
            }
        }
        let _ = cost_before;
        if ok {
            c_all += 1.0;
            if is_q { c_qual += 1.0 } else { c_comp += 1.0 }
        }
        st.correct_vec.push(ok as u8);
    }
    let n = w.tasks.len() as f64;
    st.correct = c_all / n;
    st.correct_compute = if st.n_compute > 0 { c_comp / st.n_compute as f64 } else { f64::NAN };
    st.correct_qual = if st.n_qual > 0 { c_qual / st.n_qual as f64 } else { f64::NAN };
    st.cost = cx.cost / n;
    st.cost_nominal = cx.nominal / n;
    st.tool_calls = cx.tool_calls as f64 / n;
    st.verify_calls = cx.verify_calls as f64 / n;
    st.verify_reject = if cx.verify_calls > 0 { cx.verify_reject as f64 / cx.verify_calls as f64 } else { 0.0 };
    st.cache_hit = cx.reg.stats.hit_rate();
    st.cache_saved = cx.reg.stats.cost_saved as f64 / n;
    st.memo_hit = memo_hits as f64 / n;
    st.skip_qual = if st.n_qual > 0 { skip_q as f64 / st.n_qual as f64 } else { f64::NAN };
    st.right_tool = if st.n_compute > 0 { right_c as f64 / st.n_compute as f64 } else { f64::NAN };
    st.tool_used_compute = if st.n_compute > 0 { used_c as f64 / st.n_compute as f64 } else { f64::NAN };
    st
}

// ---------------------------------------------------------------------------------------------
// self-check: the canonical specs through the REAL tools vs the native truth; verifier accept / reject

#[derive(Clone, Debug, Default)]
pub struct SelfCheck {
    pub per_kind: Vec<(String, usize, usize, usize, usize, usize, usize)>, // kind, n, tool_correct, ver_accept_true, wrong_n, ver_reject_wrong, errors
}

pub fn selfcheck(cfgs: &[WorldCfg], seeds: &[u64]) -> SelfCheck {
    let mut agg: HashMap<&'static str, [usize; 6]> = HashMap::new();
    for cfg in cfgs {
        let mut cfg = cfg.clone();
        cfg.dup = 0.0;
        cfg.kind_p = [0.2, 0.15, 0.15, 0.15, 0.2, 0.15, 0.0];
        for &sd in seeds {
            let w = gen_world(&cfg, sd);
            let mut reg = Registry::with_defaults(w.lookup.clone(), false);
            for t in &w.tasks {
                let tool = t.kind.tool().unwrap();
                let e = agg.entry(t.kind.label()).or_insert([0; 6]);
                e[0] += 1;
                let o = reg.call_idx(tool, &t.spec);
                match o.result {
                    Ok(out) => {
                        if t.truth.is_correct(&out) {
                            e[1] += 1;
                        }
                        let vi = format!("{}\n{}\n{}", TOOL_NAMES[tool], t.spec, out);
                        if matches!(reg.call_idx(VERIFY_IDX, &vi).result, Ok(s) if s == "accept") {
                            e[2] += 1;
                        }
                    }
                    Err(_) => e[5] += 1,
                }
                // a wrong claim must be rejected
                let wa = wrong_answer(t, t.salt);
                if !t.truth.is_correct(&wa) {
                    e[3] += 1;
                    let vi = format!("{}\n{}\n{}", TOOL_NAMES[tool], t.spec, wa);
                    if !matches!(reg.call_idx(VERIFY_IDX, &vi).result, Ok(s) if s == "accept") {
                        e[4] += 1;
                    }
                }
            }
        }
    }
    let mut out = SelfCheck::default();
    for k in KINDS {
        if let Some(e) = agg.get(k.label()) {
            out.per_kind.push((k.label().to_string(), e[0], e[1], e[2], e[3], e[4], e[5]));
        }
    }
    out
}
