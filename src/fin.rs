//! Domain math tools (R9-B / h18): finance + business, science / engineering, economics. std-only, deterministic,
//! no I/O, no network, no process spawning. Implements [`crate::tools::Tool`] so it drops into the W14 [`Registry`].
//!
//! Input grammar (one line): `op key=value key=value ...` where a value is a decimal number, a comma list
//! (`cf=-100,30,40,50`), or a word / formula (`f=C6H12O6`, `eq=2H2+O2>2H2O`). Output: `v=<primary answer> k=<extra> ...`
//! with numbers printed in shortest round-trip form. Everything is f64 with a DOCUMENTED tolerance (answers are
//! compared with [`close`]: relative/absolute 1e-6, i.e. 7 significant digits), iterative solvers are bracketed with
//! a tolerance + convergence report (`iters`, `conv`), every loop is bounded, every result is checked finite.
//!
//! Each op has an INDEPENDENT verifier ([`verify_indep`]): a second algorithm or an invariant that shares no code
//! with the evaluator (NPV(IRR)=0 as a sign-change straddle, amortisation sums to the principal, duration vs a
//! numeric derivative of the price, unit round trip through an exact rational table written from the physical
//! definitions, mass conservation, SUVAT identities ...). [`verify_shared`] is the weak baseline: re-run the same
//! evaluator and compare. Evaluator BUG HOOKS (`bug::*`, default 0 = production) exist only so mutation testing can
//! seed faults; `FinTool::new()` / `bug = 0` is the production path and unit tests check hook 0 == clean behaviour.
//!
//! Honest scope: these tools make exact sub-computations reliable. They do not make a model smarter, they cannot
//! verify anything without a machine-readable spec, temperature conversions with offsets (C, F) are NOT supported
//! (non-multiplicative), and the independent verifier is "independent" only in the sense of a different algorithm
//! by the same author (shared data such as atomic masses is common-mode).

use crate::tools::{Tool, ToolError};
use std::collections::BTreeMap;

type R<T> = Result<T, ToolError>;

pub const MAX_LIST: usize = 400;
pub const MAX_PERIODS: i64 = 100_000;
/// answer tolerance: |a-b| <= RTOL * max(1,|a|,|b|)
pub const RTOL: f64 = 1e-6;

pub fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= RTOL * 1f64.max(a.abs()).max(b.abs())
}

// ---------------------------------------------------------------------------------------------
// seeded evaluator bugs (mutation testing hooks; 0 = production)

pub mod bug {
    pub const NONE: u8 = 0;
    pub const NPV_T_OFF1: u8 = 1;
    pub const NPV_SKIP_LAST: u8 = 2;
    pub const IRR_TOL: u8 = 3;
    pub const IRR_BRACKET: u8 = 4;
    pub const PAYBACK_NOINTERP: u8 = 5;
    pub const COMP_SIMPLE: u8 = 6;
    pub const COMP_M_IGNORED: u8 = 7;
    pub const CONT_AS_ANNUAL: u8 = 8;
    pub const LOAN_OFF1: u8 = 9;
    pub const AMORT_K_PLUS1: u8 = 10;
    pub const ANN_DUE_WRONG: u8 = 11;
    pub const BOND_FREQ: u8 = 12;
    pub const BOND_NOFACE: u8 = 13;
    pub const BOND_MOD_NOF: u8 = 14;
    pub const BOND_DUR_PERIODS: u8 = 15;
    pub const CAGR_N_OFF1: u8 = 16;
    pub const WACC_NO_TAX: u8 = 17;
    pub const QUICK_NO_INV: u8 = 18;
    pub const ROE_INVERTED: u8 = 19;
    pub const PCTCHG_NEWBASE: u8 = 20;
    pub const BREAKEVEN_NOVC: u8 = 21;
    pub const PARITY_SIGN: u8 = 22;
    pub const GPERP_PLUS: u8 = 23;
    pub const ELAST_NEWBASE: u8 = 24;
    pub const REAL_INVERSE: u8 = 25;
    pub const UNIT_DIV_ASSOC: u8 = 26;
    pub const UNIT_PAREN_POW: u8 = 27;
    pub const UNIT_PSI: u8 = 28;
    pub const SF_TRAILING: u8 = 29;
    pub const SF_TRUNC: u8 = 30;
    pub const SCI_EXP_OFF1: u8 = 31;
    pub const ERR_LINEAR: u8 = 32;
    pub const ERR_MUL_ABS: u8 = 33;
    pub const VAR_POP: u8 = 34;
    pub const CI_Z: u8 = 35;
    pub const REG_INTERCEPT0: u8 = 36;
    pub const MOLMASS_PAREN: u8 = 37;
    pub const STOICH_NOCOEF: u8 = 38;
    pub const GAS_R_TYPO: u8 = 39;
    pub const KIN_NO_HALF: u8 = 40;
    pub const MEDIAN_UPPER: u8 = 41;
    /// COMMON-MODE: planted in BOTH the evaluator and the independent verifier (independence cannot help)
    pub const CM_UNIT_LB: u8 = 100;
    pub const CM_MASS_O: u8 = 101;
    pub const CM_PCT_BASE: u8 = 102;
    pub const ALL: &[(u8, &str)] = &[
        (NPV_T_OFF1, "NPV_t_off1"),
        (NPV_SKIP_LAST, "NPV_skip_last"),
        (IRR_TOL, "IRR_loose_tol"),
        (IRR_BRACKET, "IRR_bracket_clamp"),
        (PAYBACK_NOINTERP, "PAYBACK_no_interp"),
        (COMP_SIMPLE, "COMP_simple_interest"),
        (COMP_M_IGNORED, "COMP_m_ignored"),
        (CONT_AS_ANNUAL, "CONT_as_annual"),
        (LOAN_OFF1, "LOAN_pmt_off1"),
        (AMORT_K_PLUS1, "AMORT_balance_k+1"),
        (ANN_DUE_WRONG, "ANN_pv_due_wrong"),
        (BOND_FREQ, "BOND_cpn_freq_ignored"),
        (BOND_NOFACE, "BOND_no_face_pv"),
        (BOND_MOD_NOF, "BOND_moddur_no_freq"),
        (BOND_DUR_PERIODS, "BOND_dur_in_periods"),
        (CAGR_N_OFF1, "CAGR_n_off1"),
        (WACC_NO_TAX, "WACC_no_taxshield"),
        (QUICK_NO_INV, "QUICK_includes_inv"),
        (ROE_INVERTED, "ROE_inverted"),
        (PCTCHG_NEWBASE, "PCTCHG_new_base"),
        (BREAKEVEN_NOVC, "BREAKEVEN_no_varcost"),
        (PARITY_SIGN, "PARITY_sign"),
        (GPERP_PLUS, "GPERP_r_plus_g"),
        (ELAST_NEWBASE, "ELAST_new_base"),
        (REAL_INVERSE, "REAL_inverse"),
        (UNIT_DIV_ASSOC, "UNIT_div_assoc"),
        (UNIT_PAREN_POW, "UNIT_paren_pow_dropped"),
        (UNIT_PSI, "UNIT_psi_factor"),
        (SF_TRAILING, "SF_trailing_zero_dropped"),
        (SF_TRUNC, "SF_truncate_not_round"),
        (SCI_EXP_OFF1, "SCI_exp_off1_below1"),
        (ERR_LINEAR, "ERR_linear_sum"),
        (ERR_MUL_ABS, "ERR_mul_abs_not_rel"),
        (VAR_POP, "VAR_population"),
        (CI_Z, "CI_z_not_t"),
        (REG_INTERCEPT0, "REG_intercept_zero"),
        (MOLMASS_PAREN, "MOLMASS_paren_mult_dropped"),
        (STOICH_NOCOEF, "STOICH_coef_ignored"),
        (GAS_R_TYPO, "GAS_R_typo"),
        (KIN_NO_HALF, "KIN_s_no_half"),
        (MEDIAN_UPPER, "MEDIAN_upper_even"),
        (CM_UNIT_LB, "CM_unit_lb"),
        (CM_MASS_O, "CM_mass_O"),
        (CM_PCT_BASE, "CM_pct_base"),
    ];
    pub fn is_common_mode(b: u8) -> bool {
        b >= 100
    }
}

// ---------------------------------------------------------------------------------------------
// helpers

fn bad<T>(s: &str) -> R<T> {
    Err(ToolError::BadInput(s.to_string()))
}
fn dom<T>(s: &str) -> R<T> {
    Err(ToolError::Domain(s.to_string()))
}
fn fin(x: f64) -> R<f64> {
    if x.is_finite() && x.abs() <= 1e18 {
        Ok(x)
    } else {
        Err(ToolError::Overflow)
    }
}
/// shortest round-trip decimal, never "-0"
pub fn fmt(x: f64) -> String {
    let x = if x == 0.0 { 0.0 } else { x };
    format!("{}", x)
}

#[derive(Clone, Debug)]
pub struct Out {
    pub v: String,
    pub extra: Vec<(&'static str, String)>,
    /// solver iterations (cost metering)
    pub iters: u32,
}
impl Out {
    fn num(v: f64) -> R<Out> {
        Ok(Out { v: fmt(fin(v)?), extra: Vec::new(), iters: 0 })
    }
    fn txt(v: String) -> R<Out> {
        Ok(Out { v, extra: Vec::new(), iters: 0 })
    }
    fn with(mut self, k: &'static str, x: f64) -> Out {
        self.extra.push((k, fmt(x)));
        self
    }
    fn withs(mut self, k: &'static str, x: String) -> Out {
        self.extra.push((k, x));
        self
    }
    fn it(mut self, n: u32) -> Out {
        self.iters = n;
        self
    }
    pub fn render(&self) -> String {
        let mut s = format!("v={}", self.v);
        for (k, x) in &self.extra {
            s.push(' ');
            s.push_str(k);
            s.push('=');
            s.push_str(x);
        }
        s
    }
}

/// parse "v=1 k=2" into a map
pub fn parse_out(s: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    for tok in s.split_whitespace() {
        if let Some((k, v)) = tok.split_once('=') {
            m.insert(k.to_string(), v.to_string());
        }
    }
    m
}

pub struct Args {
    pub op: String,
    kv: Vec<(String, String)>,
}
impl Args {
    pub fn parse(s: &str) -> R<Args> {
        if s.len() > crate::tools::MAX_INPUT {
            return Err(ToolError::TooLong);
        }
        let mut it = s.split_whitespace();
        let op = match it.next() {
            Some(o) => o.to_string(),
            None => return bad("empty input"),
        };
        let mut kv = Vec::new();
        for tok in it {
            match tok.split_once('=') {
                Some((k, v)) if !k.is_empty() && !v.is_empty() => kv.push((k.to_string(), v.to_string())),
                _ => return bad(&format!("expected key=value, got '{}'", tok.chars().take(24).collect::<String>())),
            }
        }
        Ok(Args { op, kv })
    }
    pub fn get(&self, k: &str) -> Option<&str> {
        self.kv.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }
    pub fn has(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    pub fn s(&self, k: &str) -> R<&str> {
        match self.get(k) {
            Some(v) => Ok(v),
            None => bad(&format!("missing {k}")),
        }
    }
    pub fn f(&self, k: &str) -> R<f64> {
        let t = self.s(k)?;
        parse_num(t)
    }
    pub fn fd(&self, k: &str, d: f64) -> R<f64> {
        match self.get(k) {
            Some(t) => parse_num(t),
            None => Ok(d),
        }
    }
    pub fn list(&self, k: &str) -> R<Vec<f64>> {
        let t = self.s(k)?;
        let mut v = Vec::new();
        for p in t.split(',') {
            v.push(parse_num(p)?);
            if v.len() > MAX_LIST {
                return Err(ToolError::TooLong);
            }
        }
        Ok(v)
    }
    pub fn int(&self, k: &str, lo: i64, hi: i64) -> R<i64> {
        let x = self.f(k)?;
        if x.fract() != 0.0 || x < lo as f64 || x > hi as f64 {
            return bad(&format!("{k} must be an integer in [{lo},{hi}]"));
        }
        Ok(x as i64)
    }
}
pub fn parse_num(t: &str) -> R<f64> {
    let b = t.as_bytes();
    if b.is_empty() || !b.iter().all(|c| c.is_ascii_digit() || b"+-.eE".contains(c)) {
        return bad(&format!("not a number: '{}'", t.chars().take(24).collect::<String>()));
    }
    match t.parse::<f64>() {
        Ok(x) if x.is_finite() && x.abs() <= 1e15 => Ok(x),
        _ => bad(&format!("not a finite number: '{}'", t.chars().take(24).collect::<String>())),
    }
}

// ---------------------------------------------------------------------------------------------
// op tables

pub const FIN_OPS: &[&str] = &[
    "npv", "irr", "payback", "compound", "contcomp", "annpv", "annfv", "loanpmt", "amort", "bondprice", "bondyield", "bonddur", "bondmod", "cagr", "wacc",
    "current", "quick", "roe", "roa", "gmargin", "omargin", "nmargin", "debteq", "icr", "dscr", "pct", "pctchg", "pctof", "grow", "chain", "ppoints",
    "breakeven", "payoff", "parity", "putfromcall", "perp", "gperp",
];
pub const SCI_OPS: &[&str] = &[
    "uconv", "udim", "sfcount", "sfround", "sci", "errprop", "mean", "median", "var", "sd", "sem", "ci95", "slope", "intercept", "r2", "molmass", "moles", "stoich", "limit",
    "gas", "kin",
];
pub const ECON_OPS: &[&str] = &["elast", "real", "infl", "fisher"];

// ---------------------------------------------------------------------------------------------
// the tools

fn run(bug: u8, ops: &[&str], input: &str) -> R<Out> {
    let a = Args::parse(input)?;
    if !ops.contains(&a.op.as_str()) {
        return bad(&format!("op '{}' not handled by this tool", a.op.chars().take(16).collect::<String>()));
    }
    eval(bug, &a)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FinTool {
    pub bug: u8,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct SciTool {
    pub bug: u8,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct EconTool {
    pub bug: u8,
}
fn metered(o: &R<Out>, base: u32) -> u32 {
    match o {
        Ok(o) => base + o.iters / 200,
        Err(_) => base,
    }
}
impl Tool for FinTool {
    fn name(&self) -> &str {
        "fin"
    }
    fn cost(&self) -> u32 {
        1
    }
    fn call(&self, input: &str) -> R<String> {
        run(self.bug, FIN_OPS, input).map(|o| o.render())
    }
    fn call_metered(&self, input: &str) -> (R<String>, u32) {
        let o = run(self.bug, FIN_OPS, input);
        let c = metered(&o, 1);
        (o.map(|o| o.render()), c)
    }
}
impl Tool for SciTool {
    fn name(&self) -> &str {
        "sci"
    }
    fn cost(&self) -> u32 {
        1
    }
    fn call(&self, input: &str) -> R<String> {
        run(self.bug, SCI_OPS, input).map(|o| o.render())
    }
}
impl Tool for EconTool {
    fn name(&self) -> &str {
        "econ"
    }
    fn cost(&self) -> u32 {
        1
    }
    fn call(&self, input: &str) -> R<String> {
        run(self.bug, ECON_OPS, input).map(|o| o.render())
    }
}
/// which tool index (0 fin, 1 sci, 2 econ) handles an op
pub fn tool_of(op: &str) -> Option<usize> {
    if FIN_OPS.contains(&op) {
        Some(0)
    } else if SCI_OPS.contains(&op) {
        Some(1)
    } else if ECON_OPS.contains(&op) {
        Some(2)
    } else {
        None
    }
}

/// verifier as a Tool: input "<spec line> => <claimed v>", output "accept" / "reject: why"; `indep` selects the
/// independent path, otherwise the shared (re-run the same evaluator) path.
#[derive(Clone, Copy, Debug, Default)]
pub struct DomainVerifier {
    pub indep: bool,
    pub bug: u8,
}
impl Tool for DomainVerifier {
    fn name(&self) -> &str {
        if self.indep {
            "dverify_indep"
        } else {
            "dverify_shared"
        }
    }
    fn cost(&self) -> u32 {
        if self.indep {
            2
        } else {
            1
        }
    }
    fn call(&self, input: &str) -> R<String> {
        let (spec, claim) = match input.rsplit_once("=>") {
            Some((a, b)) => (a.trim(), b.trim()),
            None => return bad("expected '<spec> => <claim>'"),
        };
        let r = if self.indep { verify_indep(self.bug, spec, claim) } else { verify_shared(self.bug, spec, claim) };
        match r {
            Ok(()) => Ok("accept".to_string()),
            Err(why) => Ok(format!("reject: {why}")),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// evaluator

pub fn eval(bug: u8, a: &Args) -> R<Out> {
    match a.op.as_str() {
        "npv" => {
            let r = a.f("rate")?;
            let cf = a.list("cf")?;
            Out::num(npv_t(bug, r, &cf)?)
        }
        "irr" => {
            let cf = a.list("cf")?;
            let (r, iters, conv, sc, resid) = irr_t(bug, &cf)?;
            Ok(Out::num(r)?.with("iters", iters as f64).with("conv", conv as u8 as f64).with("sc", sc as f64).with("resid", resid).it(iters))
        }
        "payback" => {
            let cf = a.list("cf")?;
            let rate = a.fd("rate", 0.0)?;
            Out::num(payback_t(bug, &cf, rate)?)
        }
        "compound" => {
            let (p, r, n, m) = (a.f("p")?, a.f("r")?, a.f("n")?, a.f("m")?);
            if m < 1.0 || n < 0.0 || r <= -m {
                return dom("m>=1, n>=0, r>-m required");
            }
            if (m * n).abs() > MAX_PERIODS as f64 {
                return dom("too many periods");
            }
            let v = match bug {
                bug::COMP_SIMPLE => p * (1.0 + r * n),
                bug::COMP_M_IGNORED => p * (1.0 + r / m).powf(n),
                _ => p * (1.0 + r / m).powf(m * n),
            };
            Out::num(v)
        }
        "contcomp" => {
            let (p, r, t) = (a.f("p")?, a.f("r")?, a.f("t")?);
            if r.abs() * t.abs() > 600.0 {
                return Err(ToolError::Overflow);
            }
            let v = if bug == bug::CONT_AS_ANNUAL { p * (1.0 + r).powf(t) } else { p * (r * t).exp() };
            Out::num(v)
        }
        "annpv" => {
            let (pmt, r) = (a.f("pmt")?, a.f("r")?);
            let n = a.int("n", 1, 10_000)?;
            let due = a.fd("due", 0.0)? != 0.0;
            if r <= -1.0 {
                return dom("r <= -100%");
            }
            let mut v = if r.abs() < 1e-12 { pmt * n as f64 } else { pmt * (1.0 - (1.0 + r).powi(-(n as i32))) / r };
            if due != (bug == bug::ANN_DUE_WRONG) {
                v *= 1.0 + r;
            }
            Out::num(v)
        }
        "annfv" => {
            let (pmt, r) = (a.f("pmt")?, a.f("r")?);
            let n = a.int("n", 1, 10_000)?;
            let due = a.fd("due", 0.0)? != 0.0;
            if r <= -1.0 {
                return dom("r <= -100%");
            }
            let mut v = if r.abs() < 1e-12 { pmt * n as f64 } else { pmt * ((1.0 + r).powi(n as i32) - 1.0) / r };
            if due {
                v *= 1.0 + r;
            }
            Out::num(v)
        }
        "loanpmt" => {
            let (p, r) = (a.f("p")?, a.f("r")?);
            let n = a.int("n", 1, 10_000)?;
            let pmt = loan_pmt_t(bug, p, r, n)?;
            Ok(Out::num(pmt)?.with("total_interest", pmt * n as f64 - p))
        }
        "amort" => {
            let (p, r) = (a.f("p")?, a.f("r")?);
            let n = a.int("n", 1, 10_000)?;
            let k = a.int("k", 1, n)?;
            let pmt = loan_pmt_t(0, p, r, n)?;
            let kk = if bug == bug::AMORT_K_PLUS1 { (k + 1).min(n) } else { k };
            let g = (1.0 + r).powi(kk as i32);
            let bal = if r.abs() < 1e-12 { p - pmt * kk as f64 } else { p * g - pmt * (g - 1.0) / r };
            let gp = (1.0 + r).powi((kk - 1) as i32);
            let bal_prev = if r.abs() < 1e-12 { p - pmt * (kk - 1) as f64 } else { p * gp - pmt * (gp - 1.0) / r };
            Ok(Out::num(bal)?.with("interest_k", bal_prev * r).with("principal_k", pmt - bal_prev * r).with("pmt", pmt))
        }
        "bondprice" => {
            let b = bond_args(a)?;
            Out::num(bond_price_t(bug, &b, b.y)?)
        }
        "bondyield" => {
            let b = bond_args_p(a)?;
            let (y, iters) = bond_yield_t(bug, &b)?;
            Ok(Out::num(y)?.it(iters).with("iters", iters as f64))
        }
        "bonddur" | "bondmod" => {
            let b = bond_args(a)?;
            let (d_mac, d_mod, conv) = bond_dur_t(bug, &b)?;
            if a.op == "bonddur" {
                Ok(Out::num(d_mac)?.with("mod", d_mod).with("convexity", conv))
            } else {
                Ok(Out::num(d_mod)?.with("mac", d_mac).with("convexity", conv))
            }
        }
        "cagr" => {
            let (s, e, n) = (a.f("s")?, a.f("e")?, a.f("n")?);
            if s <= 0.0 || e <= 0.0 || n <= 0.0 {
                return dom("s, e, n must be > 0");
            }
            let nn = if bug == bug::CAGR_N_OFF1 { n + 1.0 } else { n };
            Out::num((e / s).powf(1.0 / nn) - 1.0)
        }
        "wacc" => {
            let (e, d, re, rd, t) = (a.f("e")?, a.f("d")?, a.f("re")?, a.f("rd")?, a.f("t")?);
            if e < 0.0 || d < 0.0 || e + d <= 0.0 || !(0.0..=1.0).contains(&t) {
                return dom("e,d >= 0, e+d > 0, 0<=t<=1");
            }
            let tax = if bug == bug::WACC_NO_TAX { 0.0 } else { t };
            Out::num(e / (e + d) * re + d / (e + d) * rd * (1.0 - tax))
        }
        "current" => ratio(a, "ca", "cl"),
        "quick" => {
            let (ca, inv, cl) = (a.f("ca")?, a.f("inv")?, a.f("cl")?);
            if cl == 0.0 {
                return Err(ToolError::DivZero);
            }
            Out::num(if bug == bug::QUICK_NO_INV { ca / cl } else { (ca - inv) / cl })
        }
        "roe" => {
            let (ni, eq) = (a.f("ni")?, a.f("eq")?);
            if ni == 0.0 || eq == 0.0 {
                return Err(ToolError::DivZero);
            }
            Out::num(if bug == bug::ROE_INVERTED { eq / ni } else { ni / eq })
        }
        "roa" => ratio(a, "ni", "ta"),
        "gmargin" => {
            let (rev, cogs) = (a.f("rev")?, a.f("cogs")?);
            if rev == 0.0 {
                return Err(ToolError::DivZero);
            }
            Out::num((rev - cogs) / rev)
        }
        "omargin" => {
            let (rev, cogs, opex) = (a.f("rev")?, a.f("cogs")?, a.f("opex")?);
            if rev == 0.0 {
                return Err(ToolError::DivZero);
            }
            Out::num((rev - cogs - opex) / rev)
        }
        "nmargin" => ratio(a, "ni", "rev"),
        "debteq" => ratio(a, "debt", "eq"),
        "icr" => ratio(a, "ebit", "interest"),
        "dscr" => ratio(a, "noi", "ds"),
        "pct" => {
            let (part, whole) = (a.f("part")?, a.f("whole")?);
            if whole == 0.0 {
                return Err(ToolError::DivZero);
            }
            Out::num(100.0 * part / whole)
        }
        "pctchg" => {
            let (old, new) = (a.f("old")?, a.f("new")?);
            let base = if bug == bug::PCTCHG_NEWBASE || bug == bug::CM_PCT_BASE { new } else { old };
            if base == 0.0 {
                return Err(ToolError::DivZero);
            }
            Out::num(100.0 * (new - old) / base)
        }
        "pctof" => Out::num(a.f("p")? / 100.0 * a.f("x")?),
        "grow" => {
            let (base, rate) = (a.f("base")?, a.f("rate")?);
            let n = a.int("n", 0, 10_000)?;
            if rate <= -100.0 {
                return dom("rate <= -100%");
            }
            Out::num(base * (1.0 + rate / 100.0).powi(n as i32))
        }
        "chain" => {
            let rs = a.list("rates")?;
            let mut p = 1.0;
            for r in &rs {
                if *r <= -100.0 {
                    return dom("rate <= -100%");
                }
                p *= 1.0 + r / 100.0;
            }
            Out::num((p - 1.0) * 100.0)
        }
        "ppoints" => Out::num(a.f("b")? - a.f("a")?),
        "breakeven" => {
            let (fc, price, vc) = (a.f("fc")?, a.f("price")?, a.f("vc")?);
            if fc < 0.0 || price <= vc {
                return dom("fc >= 0 and price > vc required");
            }
            let units = if bug == bug::BREAKEVEN_NOVC { fc / price } else { fc / (price - vc) };
            Ok(Out::num(units)?.with("revenue", units * price))
        }
        "payoff" => {
            let (s, k, prem) = (a.f("s")?, a.f("k")?, a.fd("prem", 0.0)?);
            let intrinsic = match a.s("kind")? {
                "call" => (s - k).max(0.0),
                "put" => (k - s).max(0.0),
                _ => return bad("kind must be call|put"),
            };
            let long = match a.s("pos")? {
                "long" => true,
                "short" => false,
                _ => return bad("pos must be long|short"),
            };
            Out::num(if long { intrinsic - prem } else { prem - intrinsic })
        }
        "parity" => {
            let (c, p, s, k, r, t) = (a.f("c")?, a.f("p")?, a.f("s")?, a.f("k")?, a.f("r")?, a.f("t")?);
            if r.abs() * t.abs() > 600.0 {
                return Err(ToolError::Overflow);
            }
            let pv = k * (-r * t).exp();
            let resid = if bug == bug::PARITY_SIGN { c - p - (s + pv) } else { c - p - (s - pv) };
            Out::num(resid)
        }
        "putfromcall" => {
            let (c, s, k, r, t) = (a.f("c")?, a.f("s")?, a.f("k")?, a.f("r")?, a.f("t")?);
            if r.abs() * t.abs() > 600.0 {
                return Err(ToolError::Overflow);
            }
            Out::num(c - s + k * (-r * t).exp())
        }
        "perp" => {
            let (c, r) = (a.f("c")?, a.f("r")?);
            if r <= 0.0 {
                return dom("r must be > 0");
            }
            Out::num(c / r)
        }
        "gperp" => {
            let (c, r, g) = (a.f("c")?, a.f("r")?, a.f("g")?);
            if r <= g || r <= 0.0 {
                return dom("need r > g and r > 0");
            }
            Out::num(if bug == bug::GPERP_PLUS { c / (r + g) } else { c / (r - g) })
        }
        // ---------------- economics
        "elast" => {
            let (q1, q2, p1, p2) = (a.f("q1")?, a.f("q2")?, a.f("p1")?, a.f("p2")?);
            if q1 + q2 == 0.0 || p1 + p2 == 0.0 || p1 == p2 {
                return Err(ToolError::DivZero);
            }
            if bug == bug::ELAST_NEWBASE {
                if q2 == 0.0 || p2 == 0.0 {
                    return Err(ToolError::DivZero);
                }
                return Out::num(((q2 - q1) / q2) / ((p2 - p1) / p2));
            }
            Out::num(((q2 - q1) / ((q1 + q2) / 2.0)) / ((p2 - p1) / ((p1 + p2) / 2.0)))
        }
        "real" => {
            let (nominal, cpi, base) = (a.f("nominal")?, a.f("cpi")?, a.f("base")?);
            if cpi <= 0.0 || base <= 0.0 {
                return dom("cpi and base must be > 0");
            }
            Out::num(if bug == bug::REAL_INVERSE { nominal * cpi / base } else { nominal * base / cpi })
        }
        "infl" => {
            let (c0, c1) = (a.f("cpi0")?, a.f("cpi1")?);
            if c0 <= 0.0 {
                return dom("cpi0 must be > 0");
            }
            Out::num((c1 / c0 - 1.0) * 100.0)
        }
        "fisher" => {
            let (nom, infl) = (a.f("nom")?, a.f("infl")?);
            if infl <= -1.0 {
                return dom("infl <= -100%");
            }
            Out::num((1.0 + nom) / (1.0 + infl) - 1.0)
        }
        // ---------------- science
        _ => sci_eval(bug, a),
    }
}

fn ratio(a: &Args, n: &str, d: &str) -> R<Out> {
    let (x, y) = (a.f(n)?, a.f(d)?);
    if y == 0.0 {
        return Err(ToolError::DivZero);
    }
    Out::num(x / y)
}

// ----- npv / irr / payback

fn npv_t(bug: u8, r: f64, cf: &[f64]) -> R<f64> {
    if r <= -1.0 {
        return dom("rate <= -100%");
    }
    let n = cf.len();
    let mut s = 0.0;
    for (t, c) in cf.iter().enumerate() {
        if bug == bug::NPV_SKIP_LAST && n > 1 && t + 1 == n {
            continue;
        }
        let tt = t as i32 + if bug == bug::NPV_T_OFF1 { 1 } else { 0 };
        s += c / (1.0 + r).powi(tt);
    }
    fin(s)
}
fn npv_raw(r: f64, cf: &[f64]) -> f64 {
    let mut s = 0.0;
    for (t, c) in cf.iter().enumerate() {
        s += c * (1.0 + r).powi(-(t as i32));
    }
    s
}
fn sign_changes(cf: &[f64]) -> usize {
    let mut last = 0i8;
    let mut n = 0;
    for c in cf {
        let s = if *c > 0.0 { 1 } else if *c < 0.0 { -1 } else { 0 };
        if s != 0 {
            if last != 0 && s != last {
                n += 1;
            }
            last = s;
        }
    }
    n
}
/// bracketed bisection on r in (-0.99, 100]; returns (irr, iterations, converged, sign changes of cf, |NPV(irr)|)
fn irr_t(bug: u8, cf: &[f64]) -> R<(f64, u32, bool, usize, f64)> {
    let sc = sign_changes(cf);
    if sc == 0 {
        return dom("cashflows have no sign change");
    }
    const GRID: usize = 800;
    let x0 = 0.01f64.ln();
    let x1 = if bug == bug::IRR_BRACKET { 1.5f64.ln() } else { 101f64.ln() };
    let mut px = x0;
    let mut pf = npv_raw(px.exp() - 1.0, cf);
    let mut br: Option<(f64, f64, f64, f64)> = None;
    for i in 1..=GRID {
        let x = x0 + (x1 - x0) * i as f64 / GRID as f64;
        let f = npv_raw(x.exp() - 1.0, cf);
        if !f.is_finite() {
            return Err(ToolError::Overflow);
        }
        if pf == 0.0 {
            br = Some((px.exp() - 1.0, px.exp() - 1.0, 0.0, 0.0));
            break;
        }
        if pf * f < 0.0 {
            br = Some((px.exp() - 1.0, x.exp() - 1.0, pf, f));
            break;
        }
        px = x;
        pf = f;
    }
    let (mut lo, mut hi, mut flo, _fhi) = match br {
        Some(b) => b,
        None => {
            if bug == bug::IRR_BRACKET {
                let r = x1.exp() - 1.0;
                return Ok((r, 0, false, sc, npv_raw(r, cf).abs()));
            }
            return dom("no root in (-99%, 10000%]");
        }
    };
    let tol = if bug == bug::IRR_TOL { 1e-3 } else { 1e-13 };
    let mut iters = 0u32;
    while iters < 200 && (hi - lo) > tol * (1.0 + lo.abs()) && hi > lo {
        let mid = 0.5 * (lo + hi);
        let fm = npv_raw(mid, cf);
        iters += 1;
        if fm == 0.0 {
            lo = mid;
            hi = mid;
            break;
        }
        if flo * fm < 0.0 {
            hi = mid;
        } else {
            lo = mid;
            flo = fm;
        }
    }
    let r = 0.5 * (lo + hi);
    let conv = (hi - lo) <= 1e-12 * (1.0 + r.abs()) || bug == bug::IRR_TOL && (hi - lo) <= tol * (1.0 + lo.abs());
    Ok((fin(r)?, iters, conv, sc, npv_raw(r, cf).abs()))
}
fn payback_t(bug: u8, cf: &[f64], rate: f64) -> R<f64> {
    if rate <= -1.0 {
        return dom("rate <= -100%");
    }
    let mut cum = 0.0;
    let mut prev = 0.0;
    for (t, c) in cf.iter().enumerate() {
        let pv = c / (1.0 + rate).powi(t as i32);
        prev = cum;
        cum += pv;
        if cum >= 0.0 && (t > 0 || pv >= 0.0) {
            if t == 0 {
                return Ok(0.0);
            }
            if bug == bug::PAYBACK_NOINTERP {
                return Ok(t as f64);
            }
            return fin((t - 1) as f64 + (-prev) / pv);
        }
    }
    let _ = prev;
    dom("never pays back")
}

// ----- loans / bonds

fn loan_pmt_t(bug: u8, p: f64, r: f64, n: i64) -> R<f64> {
    if p <= 0.0 || r <= -1.0 {
        return dom("p > 0 and r > -100%");
    }
    if r.abs() < 1e-12 {
        return fin(p / n as f64);
    }
    let nn = if bug == bug::LOAN_OFF1 { n - 1 } else { n };
    if nn == 0 {
        return dom("n too small");
    }
    fin(p * r / (1.0 - (1.0 + r).powi(-(nn as i32))))
}

#[derive(Clone, Copy, Debug)]
pub struct Bond {
    pub face: f64,
    pub cpn: f64,
    pub y: f64,
    pub n: i64,
    pub f: f64,
    pub price: f64,
}
fn bond_core(a: &Args) -> R<(f64, f64, i64, f64)> {
    let (face, cpn, nyears, f) = (a.f("face")?, a.f("cpn")?, a.f("n")?, a.f("f")?);
    if face <= 0.0 || cpn < 0.0 || nyears <= 0.0 || f < 1.0 || f > 12.0 || f.fract() != 0.0 {
        return dom("face > 0, cpn >= 0, n > 0, f in 1..12");
    }
    let np = nyears * f;
    if (np - np.round()).abs() > 1e-9 || np.round() < 1.0 || np.round() > 2000.0 {
        return dom("n*f must be an integer in 1..2000");
    }
    Ok((face, cpn, np.round() as i64, f))
}
fn bond_args(a: &Args) -> R<Bond> {
    let (face, cpn, n, f) = bond_core(a)?;
    let y = a.f("y")?;
    if y / f <= -1.0 {
        return dom("yield <= -100% per period");
    }
    Ok(Bond { face, cpn, y, n, f, price: 0.0 })
}
fn bond_args_p(a: &Args) -> R<Bond> {
    let (face, cpn, n, f) = bond_core(a)?;
    let price = a.f("price")?;
    if price <= 0.0 {
        return dom("price must be > 0");
    }
    Ok(Bond { face, cpn, y: 0.0, n, f, price })
}
fn bond_price_t(bug: u8, b: &Bond, y: f64) -> R<f64> {
    let c = if bug == bug::BOND_FREQ { b.face * b.cpn } else { b.face * b.cpn / b.f };
    let i = y / b.f;
    if i <= -1.0 {
        return dom("yield <= -100% per period");
    }
    let n = b.n as i32;
    let fv = if bug == bug::BOND_NOFACE { 0.0 } else { b.face * (1.0 + i).powi(-n) };
    let ann = if i.abs() < 1e-12 { c * b.n as f64 } else { c * (1.0 - (1.0 + i).powi(-n)) / i };
    fin(ann + fv)
}
fn bond_yield_t(bug: u8, b: &Bond) -> R<(f64, u32)> {
    // price(y) is strictly decreasing in y on y/f > -1; bracket y/f in [-0.5, 10]
    let mut lo = -0.5 * b.f;
    let mut hi = 10.0 * b.f;
    let plo = bond_price_t(bug, b, lo)?;
    let phi = bond_price_t(bug, b, hi)?;
    if !(b.price <= plo && b.price >= phi) {
        return dom("price outside the bracketed yield range (-50%, 1000%)");
    }
    let mut it = 0;
    while it < 300 && (hi - lo) > 1e-14 * (1.0 + lo.abs()) {
        let mid = 0.5 * (lo + hi);
        let p = bond_price_t(bug, b, mid)?;
        it += 1;
        if p > b.price {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((0.5 * (lo + hi), it))
}
/// (Macaulay duration in years, modified duration, convexity)
fn bond_dur_t(bug: u8, b: &Bond) -> R<(f64, f64, f64)> {
    let c = if bug == bug::BOND_FREQ { b.face * b.cpn } else { b.face * b.cpn / b.f };
    let i = b.y / b.f;
    if i <= -1.0 {
        return dom("yield <= -100% per period");
    }
    let (mut price, mut wsum, mut csum) = (0.0, 0.0, 0.0);
    for k in 1..=b.n {
        let cf = if k == b.n { c + b.face } else { c };
        let pv = cf / (1.0 + i).powi(k as i32);
        price += pv;
        wsum += k as f64 * pv;
        csum += (k * (k + 1)) as f64 * pv;
    }
    if price <= 0.0 {
        return dom("non-positive price");
    }
    let d_per = wsum / price;
    let d_mac = if bug == bug::BOND_DUR_PERIODS { d_per } else { d_per / b.f };
    let d_mod = if bug == bug::BOND_MOD_NOF { d_mac / (1.0 + b.y) } else { d_per / b.f / (1.0 + i) };
    let conv = csum / price / ((1.0 + i) * (1.0 + i)) / (b.f * b.f);
    Ok((fin(d_mac)?, fin(d_mod)?, fin(conv)?))
}

// ---------------------------------------------------------------------------------------------
// science evaluator

// ----- units (tool side: f64 factor table + recursive descent)

const UT: &[(&str, f64, [i8; 6])] = &[
    ("m", 1.0, [1, 0, 0, 0, 0, 0]),
    ("km", 1000.0, [1, 0, 0, 0, 0, 0]),
    ("cm", 0.01, [1, 0, 0, 0, 0, 0]),
    ("mm", 0.001, [1, 0, 0, 0, 0, 0]),
    ("um", 1e-6, [1, 0, 0, 0, 0, 0]),
    ("nm", 1e-9, [1, 0, 0, 0, 0, 0]),
    ("in", 0.0254, [1, 0, 0, 0, 0, 0]),
    ("ft", 0.3048, [1, 0, 0, 0, 0, 0]),
    ("yd", 0.9144, [1, 0, 0, 0, 0, 0]),
    ("mi", 1609.344, [1, 0, 0, 0, 0, 0]),
    ("kg", 1.0, [0, 1, 0, 0, 0, 0]),
    ("g", 0.001, [0, 1, 0, 0, 0, 0]),
    ("mg", 1e-6, [0, 1, 0, 0, 0, 0]),
    ("lb", 0.45359237, [0, 1, 0, 0, 0, 0]),
    ("oz", 0.028349523125, [0, 1, 0, 0, 0, 0]),
    ("tonne", 1000.0, [0, 1, 0, 0, 0, 0]),
    ("s", 1.0, [0, 0, 1, 0, 0, 0]),
    ("ms", 0.001, [0, 0, 1, 0, 0, 0]),
    ("us", 1e-6, [0, 0, 1, 0, 0, 0]),
    ("min", 60.0, [0, 0, 1, 0, 0, 0]),
    ("h", 3600.0, [0, 0, 1, 0, 0, 0]),
    ("day", 86400.0, [0, 0, 1, 0, 0, 0]),
    ("A", 1.0, [0, 0, 0, 1, 0, 0]),
    ("K", 1.0, [0, 0, 0, 0, 1, 0]),
    ("mol", 1.0, [0, 0, 0, 0, 0, 1]),
    ("mmol", 0.001, [0, 0, 0, 0, 0, 1]),
    ("N", 1.0, [1, 1, -2, 0, 0, 0]),
    ("kN", 1000.0, [1, 1, -2, 0, 0, 0]),
    ("J", 1.0, [2, 1, -2, 0, 0, 0]),
    ("kJ", 1000.0, [2, 1, -2, 0, 0, 0]),
    ("MJ", 1e6, [2, 1, -2, 0, 0, 0]),
    ("cal", 4.184, [2, 1, -2, 0, 0, 0]),
    ("kcal", 4184.0, [2, 1, -2, 0, 0, 0]),
    ("Wh", 3600.0, [2, 1, -2, 0, 0, 0]),
    ("kWh", 3.6e6, [2, 1, -2, 0, 0, 0]),
    ("W", 1.0, [2, 1, -3, 0, 0, 0]),
    ("kW", 1000.0, [2, 1, -3, 0, 0, 0]),
    ("MW", 1e6, [2, 1, -3, 0, 0, 0]),
    ("Pa", 1.0, [-1, 1, -2, 0, 0, 0]),
    ("kPa", 1000.0, [-1, 1, -2, 0, 0, 0]),
    ("MPa", 1e6, [-1, 1, -2, 0, 0, 0]),
    ("bar", 1e5, [-1, 1, -2, 0, 0, 0]),
    ("atm", 101325.0, [-1, 1, -2, 0, 0, 0]),
    ("psi", 6894.757293168, [-1, 1, -2, 0, 0, 0]),
    ("L", 0.001, [3, 0, 0, 0, 0, 0]),
    ("mL", 1e-6, [3, 0, 0, 0, 0, 0]),
    ("Hz", 1.0, [0, 0, -1, 0, 0, 0]),
    ("V", 1.0, [2, 1, -3, -1, 0, 0]),
    ("C", 1.0, [0, 0, 1, 1, 0, 0]),
    ("ohm", 1.0, [2, 1, -3, -2, 0, 0]),
    ("knot", 1852.0 / 3600.0, [1, 0, -1, 0, 0, 0]),
];

type Dim = [i8; 6];
fn dim_add(a: &Dim, b: &Dim, sign: i8) -> R<Dim> {
    let mut o = [0i8; 6];
    for i in 0..6 {
        let v = a[i] as i32 + sign as i32 * b[i] as i32;
        if v.abs() > 40 {
            return Err(ToolError::Overflow);
        }
        o[i] = v as i8;
    }
    Ok(o)
}

struct UParser<'a> {
    s: &'a [u8],
    i: usize,
    bug: u8,
    depth: usize,
    last_was_group: bool,
}
impl<'a> UParser<'a> {
    // expr := term (('*'|'/') term)*   left associative
    fn expr(&mut self) -> R<(f64, Dim)> {
        self.depth += 1;
        if self.depth > 24 {
            return Err(ToolError::TooDeep);
        }
        let (mut f, mut d) = self.term()?;
        while self.i < self.s.len() && (self.s[self.i] == b'*' || self.s[self.i] == b'/') {
            let op = self.s[self.i];
            self.i += 1;
            let (f2, d2) = if self.bug == bug::UNIT_DIV_ASSOC && op == b'/' {
                // BUG: a/b*c parsed as a/(b*c): the divisor swallows the following '*' chain
                let (mut ff, mut dd) = self.term()?;
                while self.i < self.s.len() && self.s[self.i] == b'*' {
                    self.i += 1;
                    let (f3, d3) = self.term()?;
                    ff *= f3;
                    dd = dim_add(&dd, &d3, 1)?;
                }
                (ff, dd)
            } else {
                self.term()?
            };
            if op == b'*' {
                f *= f2;
                d = dim_add(&d, &d2, 1)?;
            } else {
                f /= f2;
                d = dim_add(&d, &d2, -1)?;
            }
        }
        self.depth -= 1;
        Ok((f, d))
    }
    fn term(&mut self) -> R<(f64, Dim)> {
        let (f, d) = self.atom()?;
        if self.i < self.s.len() && self.s[self.i] == b'^' {
            self.i += 1;
            let st = self.i;
            if self.i < self.s.len() && (self.s[self.i] == b'-' || self.s[self.i] == b'+') {
                self.i += 1;
            }
            while self.i < self.s.len() && self.s[self.i].is_ascii_digit() {
                self.i += 1;
            }
            let e: i32 = match std::str::from_utf8(&self.s[st..self.i]).ok().and_then(|t| t.parse().ok()) {
                Some(e) if (-12..=12).contains(&e) => e,
                _ => return Err(ToolError::Parse("bad exponent".into())),
            };
            let e = if self.bug == bug::UNIT_PAREN_POW && self.last_was_group { 1 } else { e };
            let mut dd = [0i8; 6];
            for k in 0..6 {
                let v = d[k] as i32 * e;
                if v.abs() > 40 {
                    return Err(ToolError::Overflow);
                }
                dd[k] = v as i8;
            }
            return Ok((f.powi(e), dd));
        }
        Ok((f, d))
    }
    fn atom(&mut self) -> R<(f64, Dim)> {
        self.last_was_group = false;
        if self.i >= self.s.len() {
            return Err(ToolError::Parse("unexpected end".into()));
        }
        let c = self.s[self.i];
        if c == b'(' {
            self.i += 1;
            let r = self.expr()?;
            if self.i >= self.s.len() || self.s[self.i] != b')' {
                return Err(ToolError::Parse("missing )".into()));
            }
            self.i += 1;
            self.last_was_group = true;
            return Ok(r);
        }
        if c == b'1' {
            self.i += 1;
            return Ok((1.0, [0; 6]));
        }
        let st = self.i;
        while self.i < self.s.len() && self.s[self.i].is_ascii_alphabetic() {
            self.i += 1;
        }
        if st == self.i {
            return Err(ToolError::Parse(format!("unexpected '{}'", c as char)));
        }
        let sym = std::str::from_utf8(&self.s[st..self.i]).unwrap_or("");
        match UT.iter().find(|u| u.0 == sym) {
            Some(u) => {
                let mut f = u.1;
                if self.bug == bug::UNIT_PSI && sym == "psi" {
                    f = 6984.757;
                }
                if (self.bug == bug::CM_UNIT_LB) && sym == "lb" {
                    f = 0.4536;
                }
                Ok((f, u.2))
            }
            None => Err(ToolError::Unknown(format!("unit {sym}"))),
        }
    }
}
// `last_was_group` lives in the struct via a separate impl block field to keep the constructor tidy
impl<'a> UParser<'a> {
    fn new(s: &'a str, bug: u8) -> UParser<'a> {
        UParser { s: s.as_bytes(), i: 0, bug, depth: 0, last_was_group: false }
    }
}

/// (factor to SI, dimension) of a compound unit expression such as `kg*m^2/s^2` or `(m/s)^2`
pub fn unit_t(bug: u8, e: &str) -> R<(f64, Dim)> {
    if e.is_empty() || e.len() > 96 {
        return Err(ToolError::TooLong);
    }
    let mut p = UParser::new(e, bug);
    let r = p.expr()?;
    if p.i != e.len() {
        return Err(ToolError::Parse(format!("trailing input at {}", p.i)));
    }
    if !r.0.is_finite() || r.0 == 0.0 {
        return Err(ToolError::Overflow);
    }
    Ok(r)
}

// ----- decimal strings (significant figures, scientific notation)

struct Dec {
    neg: bool,
    /// significant digits with leading zeros removed (may be empty for zero)
    ds: Vec<u8>,
    /// value = 0.d1d2d3... * 10^pos
    pos: i32,
    /// the mantissa carried an explicit decimal point
    point: bool,
}
fn parse_dec(s: &str) -> R<Dec> {
    let b = s.as_bytes();
    if b.is_empty() || b.len() > 40 {
        return bad("bad decimal");
    }
    let mut i = 0;
    let mut neg = false;
    if b[0] == b'-' || b[0] == b'+' {
        neg = b[0] == b'-';
        i = 1;
    }
    let (mut int_d, mut frac_d) = (Vec::new(), Vec::new());
    let mut point = false;
    while i < b.len() && b[i].is_ascii_digit() {
        int_d.push(b[i] - b'0');
        i += 1;
    }
    if i < b.len() && b[i] == b'.' {
        point = true;
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            frac_d.push(b[i] - b'0');
            i += 1;
        }
    }
    let mut ex: i32 = 0;
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let t = &s[i + 1..];
        ex = match t.parse::<i32>() {
            Ok(e) if e.abs() <= 300 => e,
            _ => return bad("bad exponent"),
        };
        i = b.len();
    }
    if i != b.len() || (int_d.is_empty() && frac_d.is_empty()) {
        return bad("bad decimal");
    }
    let mut pos = int_d.len() as i32 + ex;
    let mut all: Vec<u8> = int_d;
    all.extend(frac_d);
    // strip leading zeros
    let lead = all.iter().take_while(|d| **d == 0).count();
    all.drain(..lead);
    pos -= lead as i32;
    Ok(Dec { neg, ds: all, pos, point })
}
fn sf_count_t(bug: u8, d: &Dec) -> i64 {
    if d.ds.is_empty() {
        return 1;
    }
    let mut n = d.ds.len();
    // without a decimal point trailing zeros are not significant; BUG variant also drops them after the point
    if !d.point || bug == bug::SF_TRAILING {
        while n > 1 && d.ds[n - 1] == 0 {
            n -= 1;
        }
    }
    n as i64
}
fn fmt_digits(neg: bool, ds: &[u8], pos: i32, force_sci: bool) -> String {
    // ds is exactly the digits to show (n >= 1), value = 0.ds * 10^pos
    let n = ds.len() as i32;
    let digs: String = ds.iter().map(|d| (b'0' + d) as char).collect();
    let sign = if neg { "-" } else { "" };
    // "50" cannot show that its zero is significant (no decimal point available): fall through to scientific "5.0e1"
    let ambiguous_zero = pos == n && ds.last() == Some(&0);
    if !force_sci && pos >= 1 && pos <= n && !ambiguous_zero {
        let (a, b) = digs.split_at(pos as usize);
        if b.is_empty() {
            return format!("{sign}{a}");
        }
        return format!("{sign}{a}.{b}");
    }
    if !force_sci && pos <= 0 && pos > -5 {
        return format!("{sign}0.{}{}", "0".repeat((-pos) as usize), digs);
    }
    let (a, b) = digs.split_at(1);
    if b.is_empty() {
        format!("{sign}{a}e{}", pos - 1)
    } else {
        format!("{sign}{a}.{b}e{}", pos - 1)
    }
}
fn sf_round_t(bug: u8, d: &Dec, n: usize) -> R<String> {
    if d.ds.is_empty() {
        return bad("cannot round zero to significant figures");
    }
    let mut ds: Vec<u8> = d.ds.clone();
    let mut pos = d.pos;
    if ds.len() > n {
        let up = if bug == bug::SF_TRUNC { false } else { ds[n] >= 5 };
        ds.truncate(n);
        if up {
            let mut i = n;
            loop {
                if i == 0 {
                    ds.insert(0, 1);
                    ds.truncate(n);
                    pos += 1;
                    break;
                }
                i -= 1;
                if ds[i] == 9 {
                    ds[i] = 0;
                } else {
                    ds[i] += 1;
                    break;
                }
            }
        }
    } else {
        while ds.len() < n {
            ds.push(0);
        }
    }
    Ok(fmt_digits(d.neg, &ds, pos, false))
}
fn sci_t(bug: u8, d: &Dec, count: usize) -> R<String> {
    if d.ds.is_empty() {
        return Ok("0e0".to_string());
    }
    let mut ds = d.ds.clone();
    ds.truncate(count.max(1));
    while ds.len() < count {
        ds.push(0);
    }
    let mut e = d.pos - 1;
    if bug == bug::SCI_EXP_OFF1 && d.pos < 1 {
        e -= 1;
    }
    let digs: String = ds.iter().map(|d| (b'0' + d) as char).collect();
    let (a, b) = digs.split_at(1);
    let sign = if d.neg { "-" } else { "" };
    Ok(if b.is_empty() { format!("{sign}{a}e{e}") } else { format!("{sign}{a}.{b}e{e}") })
}

// ----- statistics

const T975: &[(i64, f64)] = &[
    (1, 12.7062),
    (2, 4.3027),
    (3, 3.1824),
    (4, 2.7764),
    (5, 2.5706),
    (6, 2.4469),
    (7, 2.3646),
    (8, 2.3060),
    (9, 2.2622),
    (10, 2.2281),
    (11, 2.2010),
    (12, 2.1788),
    (13, 2.1604),
    (14, 2.1448),
    (15, 2.1314),
    (16, 2.1199),
    (17, 2.1098),
    (18, 2.1009),
    (19, 2.0930),
    (20, 2.0860),
    (21, 2.0796),
    (22, 2.0739),
    (23, 2.0687),
    (24, 2.0639),
    (25, 2.0595),
    (26, 2.0555),
    (27, 2.0518),
    (28, 2.0484),
    (29, 2.0452),
    (30, 2.0423),
    (40, 2.0211),
    (60, 2.0003),
    (120, 1.9799),
];
/// two-sided 95% Student t critical value (table, linear in 1/df between the listed rows, 1.96 beyond 120) -
/// accurate to ~3e-4 relative, which is the DOCUMENTED tolerance of `ci95`
pub fn t975(df: i64) -> f64 {
    if df < 1 {
        return f64::NAN;
    }
    if let Some(r) = T975.iter().find(|r| r.0 == df) {
        return r.1;
    }
    if df > 120 {
        let (a, b) = (1.0 / 120.0, 0.0);
        let x = 1.0 / df as f64;
        return 1.9799 + (1.9600 - 1.9799) * (a - x) / (a - b);
    }
    let i = T975.iter().position(|r| r.0 > df).unwrap_or(T975.len() - 1);
    let (d0, t0) = T975[i - 1];
    let (d1, t1) = T975[i];
    let (x0, x1, x) = (1.0 / d0 as f64, 1.0 / d1 as f64, 1.0 / df as f64);
    t0 + (t1 - t0) * (x0 - x) / (x0 - x1)
}
fn mean_t(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}
fn var_t(bug: u8, x: &[f64]) -> R<f64> {
    let n = x.len();
    if n < 2 {
        return dom("need at least 2 observations");
    }
    let m = mean_t(x);
    let ss: f64 = x.iter().map(|v| (v - m) * (v - m)).sum();
    fin(ss / if bug == bug::VAR_POP { n as f64 } else { (n - 1) as f64 })
}
fn median_t(bug: u8, x: &[f64]) -> f64 {
    let mut v = x.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else if bug == bug::MEDIAN_UPPER {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}
fn xy(a: &Args) -> R<(Vec<f64>, Vec<f64>)> {
    let (x, y) = (a.list("x")?, a.list("y")?);
    if x.len() != y.len() || x.len() < 3 {
        return dom("x and y need equal length >= 3");
    }
    Ok((x, y))
}
fn reg_t(bug: u8, x: &[f64], y: &[f64]) -> R<(f64, f64, f64)> {
    let (mx, my) = (mean_t(x), mean_t(y));
    let sxx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum();
    let syy: f64 = y.iter().map(|v| (v - my) * (v - my)).sum();
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    if sxx <= 0.0 {
        return dom("x has no variance");
    }
    let slope = sxy / sxx;
    let icpt = if bug == bug::REG_INTERCEPT0 { 0.0 } else { my - slope * mx };
    let r2 = if syy <= 0.0 { 1.0 } else { sxy * sxy / (sxx * syy) };
    Ok((fin(slope)?, fin(icpt)?, r2))
}

// ----- chemistry

const AT: &[(&str, f64)] = &[
    ("H", 1.008),
    ("He", 4.0026),
    ("Li", 6.94),
    ("C", 12.011),
    ("N", 14.007),
    ("O", 15.999),
    ("F", 18.998),
    ("Na", 22.990),
    ("Mg", 24.305),
    ("Al", 26.982),
    ("Si", 28.085),
    ("P", 30.974),
    ("S", 32.06),
    ("Cl", 35.45),
    ("K", 39.098),
    ("Ca", 40.078),
    ("Cr", 51.996),
    ("Mn", 54.938),
    ("Fe", 55.845),
    ("Ni", 58.693),
    ("Cu", 63.546),
    ("Zn", 65.38),
    ("Br", 79.904),
    ("Ag", 107.87),
    ("I", 126.90),
    ("Ba", 137.33),
    ("Au", 196.97),
    ("Hg", 200.59),
    ("Pb", 207.2),
];
fn atomic_mass(bug: u8, sym: &str) -> Option<f64> {
    if bug == bug::CM_MASS_O && sym == "O" {
        return Some(16.0);
    }
    AT.iter().find(|a| a.0 == sym).map(|a| a.1)
}

/// element counts of a formula with parentheses and an optional hydrate suffix `.5H2O`
pub fn formula_counts(bug: u8, f: &str) -> R<BTreeMap<String, i64>> {
    let (main, hyd) = match f.split_once('.') {
        Some((m, h)) => (m, Some(h)),
        None => (f, None),
    };
    let mut total = parse_formula(bug, main)?;
    if let Some(h) = hyd {
        let digits: String = h.chars().take_while(|c| c.is_ascii_digit()).collect();
        let mult: i64 = if digits.is_empty() { 1 } else { digits.parse().map_err(|_| ToolError::Parse("bad hydrate".into()))? };
        let rest = &h[digits.len()..];
        let hc = parse_formula(bug, rest)?;
        for (k, v) in hc {
            *total.entry(k).or_insert(0) += v * mult;
        }
    }
    Ok(total)
}
fn parse_formula(bug: u8, s: &str) -> R<BTreeMap<String, i64>> {
    fn rec(bug: u8, b: &[u8], i: &mut usize, depth: usize) -> R<BTreeMap<String, i64>> {
        if depth > 6 {
            return Err(ToolError::TooDeep);
        }
        let mut m: BTreeMap<String, i64> = BTreeMap::new();
        while *i < b.len() {
            let c = b[*i];
            if c == b'(' {
                *i += 1;
                let inner = rec(bug, b, i, depth + 1)?;
                if *i >= b.len() || b[*i] != b')' {
                    return Err(ToolError::Parse("missing )".into()));
                }
                *i += 1;
                let mut mult = 1i64;
                let st = *i;
                while *i < b.len() && b[*i].is_ascii_digit() {
                    *i += 1;
                }
                if *i > st {
                    mult = std::str::from_utf8(&b[st..*i]).unwrap_or("1").parse().map_err(|_| ToolError::Overflow)?;
                }
                if bug == bug::MOLMASS_PAREN {
                    mult = 1;
                }
                for (k, v) in inner {
                    *m.entry(k).or_insert(0) += v * mult;
                }
            } else if c == b')' {
                return Ok(m);
            } else if c.is_ascii_uppercase() {
                let st = *i;
                *i += 1;
                if *i < b.len() && b[*i].is_ascii_lowercase() {
                    *i += 1;
                }
                let sym = std::str::from_utf8(&b[st..*i]).unwrap_or("").to_string();
                let st2 = *i;
                while *i < b.len() && b[*i].is_ascii_digit() {
                    *i += 1;
                }
                let cnt: i64 = if *i > st2 { std::str::from_utf8(&b[st2..*i]).unwrap_or("1").parse().map_err(|_| ToolError::Overflow)? } else { 1 };
                if cnt > 999 {
                    return Err(ToolError::Overflow);
                }
                *m.entry(sym).or_insert(0) += cnt;
            } else {
                return Err(ToolError::Parse(format!("unexpected '{}'", c as char)));
            }
        }
        Ok(m)
    }
    if s.is_empty() || s.len() > 60 {
        return Err(ToolError::Parse("bad formula length".into()));
    }
    let mut i = 0;
    let m = rec(bug, s.as_bytes(), &mut i, 0)?;
    if i != s.len() {
        return Err(ToolError::Parse("unbalanced )".into()));
    }
    Ok(m)
}
pub fn molar_mass_t(bug: u8, f: &str) -> R<f64> {
    let c = formula_counts(bug, f)?;
    if c.is_empty() {
        return bad("empty formula");
    }
    let mut m = 0.0;
    for (k, v) in c {
        match atomic_mass(bug, &k) {
            Some(x) => m += x * v as f64,
            None => return Err(ToolError::Unknown(format!("element {k}"))),
        }
    }
    Ok(m)
}
/// "2H2+O2>2H2O" -> ([(coef, formula)], [(coef, formula)])
pub fn parse_eq(eq: &str) -> R<(Vec<(i64, String)>, Vec<(i64, String)>)> {
    let (l, r) = match eq.split_once('>') {
        Some(x) => x,
        None => return bad("equation needs '>'"),
    };
    fn side(s: &str) -> R<Vec<(i64, String)>> {
        let mut v = Vec::new();
        for t in s.split('+') {
            let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
            let coef: i64 = if digits.is_empty() { 1 } else { digits.parse().map_err(|_| ToolError::Overflow)? };
            if coef < 1 || coef > 99 {
                return bad("coefficient out of range");
            }
            let f = &t[digits.len()..];
            if f.is_empty() {
                return bad("empty species");
            }
            v.push((coef, f.to_string()));
            if v.len() > 8 {
                return Err(ToolError::TooLong);
            }
        }
        Ok(v)
    }
    Ok((side(l)?, side(r)?))
}
fn balanced_t(eq: &(Vec<(i64, String)>, Vec<(i64, String)>)) -> R<bool> {
    let mut tot: BTreeMap<String, i64> = BTreeMap::new();
    for (c, f) in &eq.0 {
        for (k, v) in formula_counts(0, f)? {
            *tot.entry(k).or_insert(0) += c * v;
        }
    }
    for (c, f) in &eq.1 {
        for (k, v) in formula_counts(0, f)? {
            *tot.entry(k).or_insert(0) -= c * v;
        }
    }
    Ok(tot.values().all(|v| *v == 0))
}
fn species(eq: &(Vec<(i64, String)>, Vec<(i64, String)>), name: &str) -> R<(i64, bool)> {
    for (c, f) in &eq.0 {
        if f == name {
            return Ok((*c, true));
        }
    }
    for (c, f) in &eq.1 {
        if f == name {
            return Ok((*c, false));
        }
    }
    Err(ToolError::NotFound(format!("species {name}")))
}

// ----- ideal gas / kinematics

pub const R_GAS: f64 = 8.314462618;

fn kin_solve(bug: u8, k: [Option<f64>; 5]) -> R<[f64; 5]> {
    // order: u v a t s
    let [u, v, a, t, s] = k;
    let half = if bug == bug::KIN_NO_HALF { 1.0 } else { 0.5 };
    let nz = |x: f64| -> R<f64> {
        if x.abs() < 1e-12 {
            Err(ToolError::DivZero)
        } else {
            Ok(x)
        }
    };
    let out = match (u, v, a, t, s) {
        (Some(u), None, Some(a), Some(t), None) => [u, u + a * t, a, t, u * t + half * a * t * t],
        (None, Some(v), Some(a), Some(t), None) => [v - a * t, v, a, t, v * t - half * a * t * t],
        (Some(u), Some(v), None, Some(t), None) => {
            let t = nz(t)?;
            [u, v, (v - u) / t, t, (u + v) * t * half]
        }
        (Some(u), Some(v), Some(a), None, None) => {
            let a = nz(a)?;
            [u, v, a, (v - u) / a, (v * v - u * u) / (2.0 * a)]
        }
        (Some(u), None, Some(a), None, Some(s)) => {
            let d = u * u + 2.0 * a * s;
            if d < 0.0 {
                return dom("negative discriminant");
            }
            let v = d.sqrt();
            let a = nz(a)?;
            [u, v, a, (v - u) / a, s]
        }
        (Some(u), None, None, Some(t), Some(s)) => {
            let t = nz(t)?;
            let a = 2.0 * (s - u * t) / (t * t);
            [u, u + a * t, a, t, s]
        }
        (None, Some(v), None, Some(t), Some(s)) => {
            let t = nz(t)?;
            let a = 2.0 * (v * t - s) / (t * t);
            [v - a * t, v, a, t, s]
        }
        (None, None, Some(a), Some(t), Some(s)) => {
            let t = nz(t)?;
            let u = (s - half * a * t * t) / t;
            [u, u + a * t, a, t, s]
        }
        (Some(u), Some(v), None, None, Some(s)) => {
            let s = nz(s)?;
            let a = (v * v - u * u) / (2.0 * s);
            let t = nz(u + v)?;
            [u, v, a, 2.0 * s / t, s]
        }
        (None, Some(v), Some(a), None, Some(s)) => {
            let d = v * v - 2.0 * a * s;
            if d < 0.0 {
                return dom("negative discriminant");
            }
            let u = d.sqrt();
            let a = nz(a)?;
            [u, v, a, (v - u) / a, s]
        }
        _ => return bad("kin needs exactly three of u,v,a,t,s"),
    };
    for x in out {
        fin(x)?;
    }
    Ok(out)
}

fn sci_eval(bug: u8, a: &Args) -> R<Out> {
    match a.op.as_str() {
        "uconv" => {
            let x = a.f("x")?;
            let (ff, df) = unit_t(bug, a.s("from")?)?;
            let (ft, dt) = unit_t(bug, a.s("to")?)?;
            if df != dt {
                return Err(ToolError::Dimension(format!("{:?} vs {:?}", df, dt)));
            }
            Out::num(x * ff / ft)
        }
        "udim" => {
            let (_, da) = unit_t(bug, a.s("a")?)?;
            let (_, db) = unit_t(bug, a.s("b")?)?;
            Out::num(if da == db { 1.0 } else { 0.0 })
        }
        "sfcount" => {
            let d = parse_dec(a.s("x")?)?;
            Out::num(sf_count_t(bug, &d) as f64)
        }
        "sfround" => {
            let d = parse_dec(a.s("x")?)?;
            let n = a.int("n", 1, 15)? as usize;
            Out::txt(sf_round_t(bug, &d, n)?)
        }
        "sci" => {
            let d = parse_dec(a.s("x")?)?;
            let c = sf_count_t(0, &d) as usize;
            Out::txt(sci_t(bug, &d, c)?)
        }
        "errprop" => {
            let (x, sx) = (a.f("x")?, a.f("sx")?);
            if sx < 0.0 {
                return dom("uncertainties must be >= 0");
            }
            match a.s("f")? {
                "add" | "sub" => {
                    let (y, sy) = (a.f("y")?, a.f("sy")?);
                    if sy < 0.0 {
                        return dom("uncertainties must be >= 0");
                    }
                    let val = if a.s("f")? == "add" { x + y } else { x - y };
                    let s = if bug == bug::ERR_LINEAR { sx + sy } else { (sx * sx + sy * sy).sqrt() };
                    Ok(Out::num(s)?.with("val", val))
                }
                "mul" | "div" => {
                    let (y, sy) = (a.f("y")?, a.f("sy")?);
                    if sy < 0.0 || x == 0.0 || y == 0.0 {
                        return dom("uncertainties >= 0 and x,y nonzero");
                    }
                    let val = if a.s("f")? == "mul" { x * y } else { x / y };
                    let s = if bug == bug::ERR_MUL_ABS { (sx * sx + sy * sy).sqrt() } else { val.abs() * ((sx / x).powi(2) + (sy / y).powi(2)).sqrt() };
                    Ok(Out::num(s)?.with("val", val))
                }
                "pow" => {
                    let n = a.f("n")?;
                    if x <= 0.0 {
                        return dom("x must be > 0 for pow");
                    }
                    let val = x.powf(n);
                    Ok(Out::num(n.abs() * x.powf(n - 1.0) * sx)?.with("val", val))
                }
                _ => bad("f must be add|sub|mul|div|pow"),
            }
        }
        "mean" => Out::num(mean_t(&a.list("x")?)),
        "median" => {
            let x = a.list("x")?;
            Out::num(median_t(bug, &x))
        }
        "var" => Out::num(var_t(bug, &a.list("x")?)?),
        "sd" => Out::num(var_t(bug, &a.list("x")?)?.sqrt()),
        "sem" => {
            let x = a.list("x")?;
            Out::num((var_t(bug, &x)? / x.len() as f64).sqrt())
        }
        "ci95" => {
            let x = a.list("x")?;
            let n = x.len();
            let sd = var_t(0, &x)?.sqrt();
            let t = if bug == bug::CI_Z { 1.96 } else { t975(n as i64 - 1) };
            let hw = t * sd / (n as f64).sqrt();
            let m = mean_t(&x);
            Ok(Out::num(hw)?.with("mean", m).with("lo", m - hw).with("hi", m + hw))
        }
        "slope" => {
            let (x, y) = xy(a)?;
            Out::num(reg_t(bug, &x, &y)?.0)
        }
        "intercept" => {
            let (x, y) = xy(a)?;
            Out::num(reg_t(bug, &x, &y)?.1)
        }
        "r2" => {
            let (x, y) = xy(a)?;
            Out::num(reg_t(bug, &x, &y)?.2)
        }
        "molmass" => Out::num(molar_mass_t(bug, a.s("f")?)?),
        "moles" => {
            let m = a.f("mass")?;
            Out::num(m / molar_mass_t(bug, a.s("f")?)?)
        }
        "stoich" => {
            let eq = parse_eq(a.s("eq")?)?;
            if !balanced_t(&eq)? {
                return dom("equation is not balanced");
            }
            let (given, want) = (a.s("given")?, a.s("want")?);
            let m = a.f("m")?;
            let (cg, gr) = species(&eq, given)?;
            let (cw, wr) = species(&eq, want)?;
            if gr == wr {
                return dom("given and wanted must be on opposite sides");
            }
            let (mg, mw) = (molar_mass_t(bug, given)?, molar_mass_t(bug, want)?);
            let (cg, cw) = if bug == bug::STOICH_NOCOEF { (1, 1) } else { (cg, cw) };
            Out::num(m / mg * cw as f64 / cg as f64 * mw)
        }
        "limit" => {
            let eq = parse_eq(a.s("eq")?)?;
            if !balanced_t(&eq)? {
                return dom("equation is not balanced");
            }
            let (na, nb, want) = (a.s("a")?, a.s("b")?, a.s("want")?);
            let (ma, mb) = (a.f("ma")?, a.f("mb")?);
            let (ca, ra) = species(&eq, na)?;
            let (cb, rb) = species(&eq, nb)?;
            let (cw, rw) = species(&eq, want)?;
            if !ra || !rb || rw {
                return dom("a and b must be reactants, want a product");
            }
            let xa = ma / molar_mass_t(bug, na)? / ca as f64;
            let xb = mb / molar_mass_t(bug, nb)? / cb as f64;
            let ext = xa.min(xb);
            let lim = if xa <= xb { na } else { nb };
            Ok(Out::num(ext * cw as f64 * molar_mass_t(bug, want)?)?.withs("lim", lim.to_string()))
        }
        "gas" => {
            let r = if bug == bug::GAS_R_TYPO { 8.341 } else { R_GAS };
            let find = a.s("find")?;
            let (p, v, n, t) = (a.f("p").ok(), a.f("v").ok(), a.f("n").ok(), a.f("t").ok());
            let known = [p, v, n, t].iter().filter(|x| x.is_some()).count();
            if known != 3 {
                return bad("gas needs exactly three of p,v,n,t");
            }
            let pos = |x: f64| -> R<f64> {
                if x <= 0.0 {
                    dom("all gas quantities must be > 0")
                } else {
                    Ok(x)
                }
            };
            let res = match (find, p, v, n, t) {
                ("p", None, Some(v), Some(n), Some(t)) => pos(n)? * r * pos(t)? / pos(v)?,
                ("v", Some(p), None, Some(n), Some(t)) => pos(n)? * r * pos(t)? / pos(p)?,
                ("n", Some(p), Some(v), None, Some(t)) => pos(p)? * pos(v)? / (r * pos(t)?),
                ("t", Some(p), Some(v), Some(n), None) => pos(p)? * pos(v)? / (r * pos(n)?),
                _ => return bad("find must name the missing variable"),
            };
            Out::num(res)
        }
        "kin" => {
            let g = |k: &str| a.f(k).ok();
            let k = [g("u"), g("v"), g("a"), g("t"), g("s")];
            let names = ["u", "v", "a", "t", "s"];
            let find = a.s("find")?;
            let fi = match names.iter().position(|n| *n == find) {
                Some(i) => i,
                None => return bad("find must be one of u,v,a,t,s"),
            };
            if k[fi].is_some() {
                return bad("find must be an unknown");
            }
            let full = kin_solve(bug, k)?;
            let other = (0..5).find(|i| *i != fi && k[*i].is_none()).unwrap_or(fi);
            Ok(Out::num(full[fi])?.withs("on", names[other].to_string()).with("o", full[other]))
        }
        _ => bad("unknown op"),
    }
}

// ---------------------------------------------------------------------------------------------
// VERIFIERS

/// weak baseline: re-run the same evaluator and compare the claim (catches transcription errors, NOT evaluator bugs)
pub fn verify_shared(bug: u8, spec: &str, claim: &str) -> Result<(), String> {
    let a = Args::parse(spec).map_err(|e| e.to_string())?;
    let o = eval(bug, &a).map_err(|e| format!("evaluator error: {e}"))?;
    claims_equal(&o.v, claim)
}
fn claims_equal(truth: &str, claim: &str) -> Result<(), String> {
    match (truth.parse::<f64>(), claim.parse::<f64>()) {
        (Ok(a), Ok(b)) => {
            if close(a, b) {
                Ok(())
            } else {
                Err(format!("claim {claim} != recomputed {truth}"))
            }
        }
        _ => {
            if truth == claim {
                Ok(())
            } else {
                Err(format!("claim {claim} != recomputed {truth}"))
            }
        }
    }
}

fn rej<T>(s: String) -> Result<T, String> {
    Err(s)
}
fn cl(spec: &Args, k: &str) -> Result<f64, String> {
    spec.f(k).map_err(|e| e.to_string())
}
fn cli(spec: &Args, k: &str, lo: i64, hi: i64) -> Result<i64, String> {
    spec.int(k, lo, hi).map_err(|e| e.to_string())
}
fn cll(spec: &Args, k: &str) -> Result<Vec<f64>, String> {
    spec.list(k).map_err(|e| e.to_string())
}

/// INDEPENDENT verifier: second algorithm / invariant per op. `cm` is the common-mode bug (also planted here) or 0.
pub fn verify_indep(cm: u8, spec: &str, claim: &str) -> Result<(), String> {
    let a = Args::parse(spec).map_err(|e| e.to_string())?;
    let op = a.op.as_str();
    // string-valued claims first
    match op {
        "sfround" | "sci" => return indep_sig(&a, claim),
        _ => {}
    }
    let c: f64 = match claim.parse() {
        Ok(x) if f64::is_finite(x) => x,
        _ => return rej(format!("claim '{}' is not a finite number", claim.chars().take(20).collect::<String>())),
    };
    let ck = |truth: f64| -> Result<(), String> {
        if close(truth, c) {
            Ok(())
        } else {
            rej(format!("claim {claim} != independent {}", fmt(truth)))
        }
    };
    // invariant: claim * den == num
    let ratio_inv = |num: f64, den: f64| -> Result<(), String> {
        if den == 0.0 {
            return rej("zero denominator".into());
        }
        if (c * den - num).abs() <= RTOL * 1f64.max(num.abs()).max((c * den).abs()) {
            Ok(())
        } else {
            rej(format!("claim {claim} * den {} != num {}", fmt(den), fmt(num)))
        }
    };
    match op {
        "npv" => {
            let (r, cf) = (cl(&a, "rate")?, cll(&a, "cf")?);
            if r <= -1.0 {
                return rej("rate <= -100%".into());
            }
            let mut v = 0.0;
            for x in cf.iter().rev() {
                v = x + v / (1.0 + r);
            }
            ck(v)
        }
        "irr" => {
            let cf = cll(&a, "cf")?;
            if sign_changes(&cf) == 0 {
                return rej("no sign change".into());
            }
            if c <= -1.0 {
                return rej("irr <= -100%".into());
            }
            // Horner NPV; the root must be STRADDLED within delta
            let f = |r: f64| -> f64 {
                let mut v = 0.0;
                for x in cf.iter().rev() {
                    v = x + v / (1.0 + r);
                }
                v
            };
            let d = (RTOL * c.abs()).max(1e-8);
            let (fl, fh) = (f((c - d).max(-0.999999)), f(c + d));
            if fl == 0.0 || fh == 0.0 || (fl < 0.0) != (fh < 0.0) {
                Ok(())
            } else {
                rej(format!("NPV does not change sign around claimed IRR (NPV(r-d)={:.3e}, NPV(r+d)={:.3e})", fl, fh))
            }
        }
        "payback" => {
            let (cf, rate) = (cll(&a, "cf")?, a.fd("rate", 0.0).map_err(|e| e.to_string())?);
            // remaining-balance simulation
            let mut bal = 0.0;
            let mut found = None;
            let mut df = 1.0;
            for (t, x) in cf.iter().enumerate() {
                let pv = x * df;
                let before = bal;
                bal += pv;
                if bal >= 0.0 && (t > 0 || pv >= 0.0) {
                    found = Some(if t == 0 { 0.0 } else { (t - 1) as f64 + (0.0 - before) / pv });
                    break;
                }
                df /= 1.0 + rate;
            }
            match found {
                Some(v) => ck(v),
                None => rej("never pays back".into()),
            }
        }
        "compound" => {
            let (p, r, n, m) = (cl(&a, "p")?, cl(&a, "r")?, cl(&a, "n")?, cl(&a, "m")?);
            let periods = m * n;
            let i = r / m;
            if periods.fract() == 0.0 && periods <= 5000.0 && periods >= 0.0 {
                let mut v = p;
                for _ in 0..(periods as i64) {
                    v *= 1.0 + i;
                }
                ck(v)
            } else {
                ck(p * (periods * i.ln_1p()).exp())
            }
        }
        "contcomp" => {
            let (p, r, t) = (cl(&a, "p")?, cl(&a, "r")?, cl(&a, "t")?);
            if p <= 0.0 || c <= 0.0 {
                return ck(p * (r * t).exp());
            }
            let lhs = (c / p).ln();
            if (lhs - r * t).abs() <= 1e-9 * (1.0 + (r * t).abs()) {
                Ok(())
            } else {
                rej(format!("ln(FV/P)={:.9} != r*t={:.9}", lhs, r * t))
            }
        }
        "annpv" | "annfv" => {
            let (pmt, r) = (cl(&a, "pmt")?, cl(&a, "r")?);
            let n = cli(&a, "n", 1, 10_000)?;
            let due = cl(&a, "due").unwrap_or(0.0) != 0.0;
            let mut v = 0.0;
            if op == "annpv" {
                for k in 1..=n {
                    v += pmt / (1.0 + r).powi(k as i32);
                }
            } else {
                for _ in 0..n {
                    v = v * (1.0 + r) + pmt;
                }
            }
            if due {
                v *= 1.0 + r;
            }
            ck(v)
        }
        "loanpmt" => {
            let (p, r) = (cl(&a, "p")?, cl(&a, "r")?);
            let n = cli(&a, "n", 1, 10_000)?;
            // amortisation: balance after n payments of the claimed amount must be ~0 and principal parts sum to p
            let mut bal = p;
            let mut sum_prin = 0.0;
            for _ in 0..n {
                let intr = bal * r;
                let prin = c - intr;
                sum_prin += prin;
                bal -= prin;
            }
            let tol = 1e-8 * p * (1.0 + r).powi(n as i32).min(1e12);
            if bal.abs() <= tol.max(1e-9 * p) && (sum_prin - p).abs() <= tol.max(1e-9 * p) {
                Ok(())
            } else {
                rej(format!("schedule leaves balance {:.6} after {} payments (principal repaid {:.6} of {:.6})", bal, n, sum_prin, p))
            }
        }
        "amort" => {
            let (p, r) = (cl(&a, "p")?, cl(&a, "r")?);
            let n = cli(&a, "n", 1, 10_000)?;
            let k = cli(&a, "k", 1, n)?;
            // payment through the (1+r)^n/((1+r)^n-1) form, balance by simulation
            let g = (1.0 + r).powi(n as i32);
            let pmt = if r.abs() < 1e-12 { p / n as f64 } else { p * r * g / (g - 1.0) };
            let mut bal = p;
            for _ in 0..k {
                bal = bal * (1.0 + r) - pmt;
            }
            ck(bal)
        }
        "bondprice" => {
            let b = bond_from(&a, false)?;
            ck(bond_price_i(&b, b.y))
        }
        "bondyield" => {
            let b = bond_from(&a, true)?;
            // straddle: price must bracket the claimed yield
            let d = (RTOL * c.abs()).max(1e-8);
            let (p1, p2) = (bond_price_i(&b, c - d), bond_price_i(&b, c + d));
            if p1 >= b.price && b.price >= p2 {
                Ok(())
            } else {
                rej(format!("price at claimed yield does not bracket {} ({:.6} / {:.6})", fmt(b.price), p1, p2))
            }
        }
        "bonddur" | "bondmod" => {
            let b = bond_from(&a, false)?;
            // modified duration = -dP/dy / P by central difference; Macaulay = mod * (1 + y/f)
            let h = 1e-5;
            let (pu, pd, p0) = (bond_price_i(&b, b.y + h), bond_price_i(&b, b.y - h), bond_price_i(&b, b.y));
            let dmod = -(pu - pd) / (2.0 * h * p0);
            let v = if op == "bondmod" { dmod } else { dmod * (1.0 + b.y / b.f) };
            if (v - c).abs() <= 1e-6 * 1f64.max(v.abs()) {
                Ok(())
            } else {
                rej(format!("claim {claim} != numeric-derivative duration {}", fmt(v)))
            }
        }
        "cagr" => {
            let (s, e, n) = (cl(&a, "s")?, cl(&a, "e")?, cl(&a, "n")?);
            let back = s * (1.0 + c).powf(n);
            if (back / e - 1.0).abs() <= 1e-8 * 1f64.max(n) {
                Ok(())
            } else {
                rej(format!("s*(1+g)^n = {} != e = {}", fmt(back), fmt(e)))
            }
        }
        "wacc" => {
            let (e, d, re, rd, t) = (cl(&a, "e")?, cl(&a, "d")?, cl(&a, "re")?, cl(&a, "rd")?, cl(&a, "t")?);
            ck((e * re + d * (rd - rd * t)) / (e + d))
        }
        "current" => ratio_inv(cl(&a, "ca")?, cl(&a, "cl")?),
        "quick" => ratio_inv(cl(&a, "ca")? - cl(&a, "inv")?, cl(&a, "cl")?),
        "roe" => ratio_inv(cl(&a, "ni")?, cl(&a, "eq")?),
        "roa" => ratio_inv(cl(&a, "ni")?, cl(&a, "ta")?),
        "gmargin" => ratio_inv(cl(&a, "rev")? - cl(&a, "cogs")?, cl(&a, "rev")?),
        "omargin" => ratio_inv(cl(&a, "rev")? - cl(&a, "cogs")? - cl(&a, "opex")?, cl(&a, "rev")?),
        "nmargin" => ratio_inv(cl(&a, "ni")?, cl(&a, "rev")?),
        "debteq" => ratio_inv(cl(&a, "debt")?, cl(&a, "eq")?),
        "icr" => ratio_inv(cl(&a, "ebit")?, cl(&a, "interest")?),
        "dscr" => ratio_inv(cl(&a, "noi")?, cl(&a, "ds")?),
        "pct" => {
            let (part, whole) = (cl(&a, "part")?, cl(&a, "whole")?);
            if (c * whole - 100.0 * part).abs() <= RTOL * 1f64.max((100.0 * part).abs()) {
                Ok(())
            } else {
                rej(format!("claim {claim}% of {} != {}", fmt(whole), fmt(part)))
            }
        }
        "pctchg" => {
            let (old, new) = (cl(&a, "old")?, cl(&a, "new")?);
            // common-mode: the same wrong concept (percent of the NEW value) is reproduced here
            let back = if cm == bug::CM_PCT_BASE { new * (1.0 - c / 100.0) } else { old * (1.0 + c / 100.0) };
            let target = if cm == bug::CM_PCT_BASE { old } else { new };
            if (back - target).abs() <= RTOL * 1f64.max(target.abs()) {
                Ok(())
            } else {
                rej(format!("implied value {} != {}", fmt(back), fmt(target)))
            }
        }
        "pctof" => ck(cl(&a, "x")? * cl(&a, "p")? * 0.01),
        "grow" => {
            let (base, rate) = (cl(&a, "base")?, cl(&a, "rate")?);
            let n = cli(&a, "n", 0, 10_000)?;
            let mut v = base;
            for _ in 0..n {
                v *= 1.0 + rate / 100.0;
            }
            ck(v)
        }
        "chain" => {
            let rs = cll(&a, "rates")?;
            let s: f64 = rs.iter().map(|r| (1.0 + r / 100.0).ln()).sum();
            ck(s.exp_m1() * 100.0)
        }
        "ppoints" => ck(-(cl(&a, "a")? - cl(&a, "b")?)),
        "breakeven" => {
            let (fc, price, vc) = (cl(&a, "fc")?, cl(&a, "price")?, cl(&a, "vc")?);
            let profit = c * price - c * vc - fc;
            if profit.abs() <= 1e-6 * 1f64.max(fc.abs()) {
                Ok(())
            } else {
                rej(format!("profit at claimed break-even is {:.6}, not 0", profit))
            }
        }
        "payoff" => {
            let (s, k, prem) = (cl(&a, "s")?, cl(&a, "k")?, cl(&a, "prem").unwrap_or(0.0));
            let x = if a.s("kind").unwrap_or("") == "call" { s - k } else { k - s };
            let intrinsic = 0.5 * (x + x.abs());
            ck(if a.s("pos").unwrap_or("") == "short" { prem - intrinsic } else { intrinsic - prem })
        }
        "parity" => {
            let (cc, p, s, k, r, t) = (cl(&a, "c")?, cl(&a, "p")?, cl(&a, "s")?, cl(&a, "k")?, cl(&a, "r")?, cl(&a, "t")?);
            let disc = 1.0 / (r * t).exp();
            ck(cc + k * disc - p - s)
        }
        "putfromcall" => {
            let (cc, s, k, r, t) = (cl(&a, "c")?, cl(&a, "s")?, cl(&a, "k")?, cl(&a, "r")?, cl(&a, "t")?);
            // parity must hold with the claimed put: (C - P) - (S - K e^{-rt}) = 0
            let resid = (cc - c) - (s - k / (r * t).exp());
            if resid.abs() <= RTOL * 1f64.max(s.abs()).max(cc.abs()) {
                Ok(())
            } else {
                rej(format!("put-call parity residual {:.6} with claimed put", resid))
            }
        }
        "perp" => {
            let (cf, r) = (cl(&a, "c")?, cl(&a, "r")?);
            ck_series(cf, 1.0 + r, 1.0, c, claim)
        }
        "gperp" => {
            let (cf, r, g) = (cl(&a, "c")?, cl(&a, "r")?, cl(&a, "g")?);
            ck_series(cf, 1.0 + r, 1.0 + g, c, claim)
        }
        "elast" => {
            let (q1, q2, p1, p2) = (cl(&a, "q1")?, cl(&a, "q2")?, cl(&a, "p1")?, cl(&a, "p2")?);
            // cross-multiplied form; midpoint elasticity is symmetric under swapping the two points
            let fwd = ((q2 - q1) * (p1 + p2)) / ((q1 + q2) * (p2 - p1));
            let bwd = ((q1 - q2) * (p2 + p1)) / ((q2 + q1) * (p1 - p2));
            if !close(fwd, bwd) {
                return rej("asymmetric".into());
            }
            ck(fwd)
        }
        "real" => {
            let (nominal, cpi, base) = (cl(&a, "nominal")?, cl(&a, "cpi")?, cl(&a, "base")?);
            let back = c * cpi / base;
            if (back - nominal).abs() <= RTOL * 1f64.max(nominal.abs()) {
                Ok(())
            } else {
                rej(format!("real*cpi/base = {} != nominal {}", fmt(back), fmt(nominal)))
            }
        }
        "infl" => {
            let (c0, c1) = (cl(&a, "cpi0")?, cl(&a, "cpi1")?);
            let back = c0 * (1.0 + c / 100.0);
            if (back - c1).abs() <= RTOL * 1f64.max(c1.abs()) {
                Ok(())
            } else {
                rej(format!("cpi0*(1+claim) = {} != cpi1 {}", fmt(back), fmt(c1)))
            }
        }
        "fisher" => {
            let (nom, infl) = (cl(&a, "nom")?, cl(&a, "infl")?);
            let back = (1.0 + c) * (1.0 + infl) - 1.0;
            if (back - nom).abs() <= RTOL {
                Ok(())
            } else {
                rej(format!("(1+real)(1+infl)-1 = {} != nominal {}", fmt(back), fmt(nom)))
            }
        }
        _ => indep_sci(cm, &a, c, claim),
    }
}

/// partial sums of  sum_{t>=1} c * g^(t-1) / q^t  (q = 1+r, g = 1 or 1+g)
fn ck_series(c0: f64, q: f64, g: f64, claim: f64, raw: &str) -> Result<(), String> {
    let ratio = g / q;
    if !(ratio < 1.0 - 1e-9) {
        return rej("series diverges (need r > g)".into());
    }
    let mut term = c0 / q;
    let mut sum = 0.0;
    let mut n = 0u64;
    while n < 2_000_000 {
        sum += term;
        term *= ratio;
        n += 1;
        // geometric tail bound: term / (1 - ratio)
        let tail = term.abs() / (1.0 - ratio);
        if tail <= 1e-9 * sum.abs().max(1.0) * 1e-3 {
            break;
        }
    }
    let tail = term.abs() / (1.0 - ratio);
    if (claim - sum).abs() <= RTOL * 1f64.max(sum.abs()) + tail {
        Ok(())
    } else {
        rej(format!("claim {raw} differs from the {n}-term partial sum {} (tail bound {:.2e})", fmt(sum), tail))
    }
}

fn bond_from(a: &Args, with_price: bool) -> Result<Bond, String> {
    let r = if with_price { bond_args_p(a) } else { bond_args(a) };
    r.map_err(|e| e.to_string())
}
/// independent price: explicit cashflow loop
fn bond_price_i(b: &Bond, y: f64) -> f64 {
    let mut p = 0.0;
    let i = y / b.f;
    for k in 1..=b.n {
        let cf = b.face * b.cpn / b.f + if k == b.n { b.face } else { 0.0 };
        p += cf * (1.0 + i).powi(-(k as i32));
    }
    p
}

// ----- independent science verification

#[derive(Clone, Copy, PartialEq)]
struct Fr(i128, i128);
fn g128(a: i128, b: i128) -> i128 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}
impl Fr {
    fn new(n: i128, d: i128) -> Option<Fr> {
        if d == 0 {
            return None;
        }
        let g = g128(n, d);
        let s = if d < 0 { -1 } else { 1 };
        Some(Fr(s * n / g, s * d / g))
    }
    fn mul(self, o: Fr) -> Option<Fr> {
        let g1 = g128(self.0, o.1);
        let g2 = g128(o.0, self.1);
        let n = (self.0 / g1).checked_mul(o.0 / g2)?;
        let d = (self.1 / g2).checked_mul(o.1 / g1)?;
        Fr::new(n, d)
    }
    fn inv(self) -> Option<Fr> {
        Fr::new(self.1, self.0)
    }
    fn to_f64(self) -> f64 {
        self.0 as f64 / self.1 as f64
    }
}
/// independent unit table: (symbol, numerator, denominator, definition in other symbols or "" for a base unit, base index)
/// base indices 0..6 = L M T I Theta N; derived units are defined through expressions so dimensions are DERIVED
const UV: &[(&str, i128, i128, &str)] = &[
    ("m", 1, 1, "#0"),
    ("kg", 1, 1, "#1"),
    ("s", 1, 1, "#2"),
    ("A", 1, 1, "#3"),
    ("K", 1, 1, "#4"),
    ("mol", 1, 1, "#5"),
    ("km", 1000, 1, "m"),
    ("cm", 1, 100, "m"),
    ("mm", 1, 1000, "m"),
    ("um", 1, 1_000_000, "m"),
    ("nm", 1, 1_000_000_000, "m"),
    ("in", 254, 10_000, "m"),
    ("ft", 3048, 10_000, "m"),
    ("yd", 9144, 10_000, "m"),
    ("mi", 1_609_344, 1000, "m"),
    ("g", 1, 1000, "kg"),
    ("mg", 1, 1_000_000, "kg"),
    ("lb", 45_359_237, 100_000_000, "kg"),
    ("oz", 28_349_523_125, 1_000_000_000_000, "kg"),
    ("tonne", 1000, 1, "kg"),
    ("ms", 1, 1000, "s"),
    ("us", 1, 1_000_000, "s"),
    ("min", 60, 1, "s"),
    ("h", 3600, 1, "s"),
    ("day", 86_400, 1, "s"),
    ("mmol", 1, 1000, "mol"),
    ("N", 1, 1, "kg*m/s^2"),
    ("kN", 1000, 1, "N"),
    ("J", 1, 1, "N*m"),
    ("kJ", 1000, 1, "J"),
    ("MJ", 1_000_000, 1, "J"),
    ("cal", 4184, 1000, "J"),
    ("kcal", 4184, 1, "J"),
    ("Wh", 3600, 1, "J"),
    ("kWh", 3_600_000, 1, "J"),
    ("W", 1, 1, "J/s"),
    ("kW", 1000, 1, "W"),
    ("MW", 1_000_000, 1, "W"),
    ("Pa", 1, 1, "N/m^2"),
    ("kPa", 1000, 1, "Pa"),
    ("MPa", 1_000_000, 1, "Pa"),
    ("bar", 100_000, 1, "Pa"),
    ("atm", 101_325, 1, "Pa"),
    // psi = lbf / in^2 with lbf = lb * 9.80665 m/s^2 (standard gravity), derived from the physical definition
    ("lbf", 45_359_237 * 980_665, 100_000_000 * 100_000, "N"),
    ("psi", 1, 1, "lbf/in^2"),
    ("L", 1, 1000, "m^3"),
    ("mL", 1, 1_000_000, "m^3"),
    ("Hz", 1, 1, "1/s"),
    ("V", 1, 1, "W/A"),
    ("C", 1, 1, "A*s"),
    ("ohm", 1, 1, "V/A"),
    ("knot", 1852, 3600, "m/s"),
];

/// flat left-to-right evaluator: every unit symbol gets sign = (-1)^(number of enclosing groups preceded by '/' ... and its own '/')
fn uv_expr(e: &str, cm: u8, depth: usize) -> Option<(Fr, [i32; 6])> {
    if depth > 8 || e.is_empty() || e.len() > 96 {
        return None;
    }
    let b = e.as_bytes();
    let mut i = 0;
    let mut group_sign: Vec<i32> = vec![1];
    // sign contributed by the operator in front of the current atom or '(' ; starts as '*'
    let mut next_sign = 1;
    let mut fac = Fr(1, 1);
    let mut dim = [0i32; 6];
    // exponents of groups: handled by pushing the exponent onto the stack at ')' time is awkward in a flat sweep, so
    // each group's sign list records the atoms contributed so far to be re-scaled at ')'
    struct Frame {
        sign: i32,
        fac: Fr,
        dim: [i32; 6],
    }
    let mut frames: Vec<Frame> = Vec::new();
    while i < b.len() {
        let c = b[i];
        match c {
            b'*' => {
                next_sign = 1;
                i += 1;
            }
            b'/' => {
                next_sign = -1;
                i += 1;
            }
            b'(' => {
                frames.push(Frame { sign: next_sign, fac, dim });
                fac = Fr(1, 1);
                dim = [0; 6];
                next_sign = 1;
                group_sign.push(1);
                i += 1;
            }
            b')' => {
                let f = frames.pop()?;
                group_sign.pop();
                i += 1;
                // optional exponent applies to the whole group
                let (mut e, ni) = uv_exp(b, i)?;
                i = ni;
                e *= f.sign;
                let mut g = fac;
                let mut gd = dim;
                let ae = e.abs();
                let mut gp = Fr(1, 1);
                let mut gdp = [0i32; 6];
                for _ in 0..ae {
                    gp = gp.mul(g)?;
                    for k in 0..6 {
                        gdp[k] += gd[k];
                    }
                }
                if e < 0 {
                    gp = gp.inv()?;
                    for k in 0..6 {
                        gdp[k] = -gdp[k];
                    }
                }
                g = gp;
                gd = gdp;
                fac = f.fac.mul(g)?;
                for k in 0..6 {
                    dim[k] = f.dim[k] + gd[k];
                }
                next_sign = 1;
            }
            b'1' => {
                i += 1;
            }
            _ if c.is_ascii_alphabetic() => {
                let st = i;
                while i < b.len() && b[i].is_ascii_alphabetic() {
                    i += 1;
                }
                let sym = std::str::from_utf8(&b[st..i]).ok()?;
                let (mut e, ni) = uv_exp(b, i)?;
                i = ni;
                e *= next_sign;
                let (f, d) = uv_symbol(sym, cm, depth)?;
                let ae = e.abs();
                let mut p = Fr(1, 1);
                for _ in 0..ae {
                    p = p.mul(f)?;
                }
                if e < 0 {
                    p = p.inv()?;
                }
                fac = fac.mul(p)?;
                for k in 0..6 {
                    dim[k] += d[k] * e;
                }
                next_sign = 1;
            }
            _ => return None,
        }
    }
    if !frames.is_empty() {
        return None;
    }
    Some((fac, dim))
}
fn uv_exp(b: &[u8], mut i: usize) -> Option<(i32, usize)> {
    if i < b.len() && b[i] == b'^' {
        i += 1;
        let st = i;
        if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
            i += 1;
        }
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let e: i32 = std::str::from_utf8(&b[st..i]).ok()?.parse().ok()?;
        if e.abs() > 12 {
            return None;
        }
        Some((e, i))
    } else {
        Some((1, i))
    }
}
fn uv_symbol(sym: &str, cm: u8, depth: usize) -> Option<(Fr, [i32; 6])> {
    let row = UV.iter().find(|r| r.0 == sym)?;
    let (mut n, d) = (row.1, row.2);
    if cm == bug::CM_UNIT_LB && sym == "lb" {
        n = 4536;
        return uv_def(sym, n, 10_000, row.3, cm, depth);
    }
    uv_def(sym, n, d, row.3, cm, depth)
}
fn uv_def(_sym: &str, n: i128, d: i128, def: &str, cm: u8, depth: usize) -> Option<(Fr, [i32; 6])> {
    let own = Fr::new(n, d)?;
    if let Some(rest) = def.strip_prefix('#') {
        let idx: usize = rest.parse().ok()?;
        let mut dim = [0i32; 6];
        dim[idx] = 1;
        return Some((own, dim));
    }
    let (f, dm) = uv_expr(def, cm, depth + 1)?;
    Some((own.mul(f)?, dm))
}

fn indep_sci(cm: u8, a: &Args, c: f64, claim: &str) -> Result<(), String> {
    let op = a.op.as_str();
    let ck = |truth: f64| -> Result<(), String> {
        if close(truth, c) {
            Ok(())
        } else {
            rej(format!("claim {claim} != independent {}", fmt(truth)))
        }
    };
    match op {
        "uconv" => {
            let x = cl(a, "x")?;
            let (ff, df) = uv_expr(a.s("from").map_err(|e| e.to_string())?, cm, 0).ok_or("independent parse of 'from' failed")?;
            let (ft, dt) = uv_expr(a.s("to").map_err(|e| e.to_string())?, cm, 0).ok_or("independent parse of 'to' failed")?;
            if df != dt {
                return rej("dimension mismatch (independent)".into());
            }
            // forward value and the round trip back to the original quantity
            let v = x * ff.mul(ft.inv().ok_or("zero factor")?).ok_or("overflow")?.to_f64();
            let back = c * ft.mul(ff.inv().ok_or("zero factor")?).ok_or("overflow")?.to_f64();
            if !close(back, x) {
                return rej(format!("round trip {} -> {} != {}", claim, fmt(back), fmt(x)));
            }
            ck(v)
        }
        "udim" => {
            let (_, da) = uv_expr(a.s("a").map_err(|e| e.to_string())?, cm, 0).ok_or("parse a failed")?;
            let (_, db) = uv_expr(a.s("b").map_err(|e| e.to_string())?, cm, 0).ok_or("parse b failed")?;
            ck(if da == db { 1.0 } else { 0.0 })
        }
        "sfcount" => {
            let s = a.s("x").map_err(|e| e.to_string())?;
            ck(sf_count_i(s)? as f64)
        }
        "errprop" => {
            let (x, sx) = (cl(a, "x")?, cl(a, "sx")?);
            let f = a.s("f").map_err(|e| e.to_string())?;
            let y = cl(a, "y").unwrap_or(1.0);
            let sy = cl(a, "sy").unwrap_or(0.0);
            let n = cl(a, "n").unwrap_or(1.0);
            let ev = |x: f64, y: f64| -> f64 {
                match f {
                    "add" => x + y,
                    "sub" => x - y,
                    "mul" => x * y,
                    "div" => x / y,
                    _ => x.powf(n),
                }
            };
            // linear error propagation through central-difference partial derivatives
            let hx = 1e-6 * x.abs().max(1e-3);
            let hy = 1e-6 * y.abs().max(1e-3);
            let dfx = (ev(x + hx, y) - ev(x - hx, y)) / (2.0 * hx);
            let dfy = if f == "pow" { 0.0 } else { (ev(x, y + hy) - ev(x, y - hy)) / (2.0 * hy) };
            let s = ((dfx * sx).powi(2) + (dfy * sy).powi(2)).sqrt();
            if (s - c).abs() <= 1e-5 * s.abs().max(1e-12) + 1e-12 {
                Ok(())
            } else {
                rej(format!("claim {claim} != finite-difference propagation {}", fmt(s)))
            }
        }
        "mean" => {
            let x = cll(a, "x")?;
            // running mean (Welford style)
            let mut m = 0.0;
            for (i, v) in x.iter().enumerate() {
                m += (v - m) / (i + 1) as f64;
            }
            ck(m)
        }
        "median" => {
            let x = cll(a, "x")?;
            // order statistic by counting: the claim must have >= half the data on each side
            let n = x.len();
            let below = x.iter().filter(|v| **v < c - 1e-12 * c.abs().max(1.0)).count();
            let above = x.iter().filter(|v| **v > c + 1e-12 * c.abs().max(1.0)).count();
            let eq = n - below - above;
            let ok_counts = 2 * below <= n && 2 * above <= n;
            if n % 2 == 1 {
                if ok_counts && eq >= 1 {
                    Ok(())
                } else {
                    rej(format!("claim {claim} is not the middle order statistic"))
                }
            } else {
                // even n: claim is the mean of the two middle values; recompute them by selection
                let mut v = x.clone();
                v.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
                ck(0.5 * (v[n / 2 - 1] + v[n / 2]))
            }
        }
        "var" | "sd" | "sem" => {
            let x = cll(a, "x")?;
            if x.len() < 2 {
                return rej("need >= 2 observations".into());
            }
            // Welford one-pass
            let (mut m, mut m2) = (0.0, 0.0);
            for (i, v) in x.iter().enumerate() {
                let d = v - m;
                m += d / (i + 1) as f64;
                m2 += d * (v - m);
            }
            let var = m2 / (x.len() - 1) as f64;
            ck(match op {
                "var" => var,
                "sd" => var.sqrt(),
                _ => (var / x.len() as f64).sqrt(),
            })
        }
        "ci95" => {
            let x = cll(a, "x")?;
            let n = x.len();
            if n < 2 {
                return rej("need >= 2 observations".into());
            }
            let (mut m, mut m2) = (0.0, 0.0);
            for (i, v) in x.iter().enumerate() {
                let d = v - m;
                m += d / (i + 1) as f64;
                m2 += d * (v - m);
            }
            let sem = (m2 / (n - 1) as f64 / n as f64).sqrt();
            let t = t_quantile_975(n as i64 - 1);
            let truth = t * sem;
            // documented tolerance of the table-based tool: 5e-4 relative
            if (truth - c).abs() <= 5e-4 * truth.abs().max(1e-12) {
                Ok(())
            } else {
                rej(format!("claim {claim} != t-quantile (incomplete beta) half-width {}", fmt(truth)))
            }
        }
        "slope" | "intercept" | "r2" => {
            let (x, y) = (cll(a, "x")?, cll(a, "y")?);
            let n = x.len() as f64;
            let (sx, sy): (f64, f64) = (x.iter().sum(), y.iter().sum());
            let sxx: f64 = x.iter().map(|v| v * v).sum();
            let sxy: f64 = x.iter().zip(&y).map(|(p, q)| p * q).sum();
            let syy: f64 = y.iter().map(|v| v * v).sum();
            let den = n * sxx - sx * sx;
            if den.abs() < 1e-300 {
                return rej("degenerate x".into());
            }
            let slope = (n * sxy - sx * sy) / den;
            match op {
                "slope" => {
                    // normal-equation invariant: residuals orthogonal to centred x
                    let (mx, my) = (sx / n, sy / n);
                    let g: f64 = x.iter().zip(&y).map(|(p, q)| (p - mx) * (q - my - c * (p - mx))).sum();
                    let scale: f64 = x.iter().map(|p| (p - mx).powi(2)).sum::<f64>().max(1e-300);
                    if (g / scale).abs() <= 1e-6 * 1f64.max(c.abs()) && close(slope, c) {
                        Ok(())
                    } else {
                        rej(format!("claim {claim} violates the normal equations (slope {})", fmt(slope)))
                    }
                }
                "intercept" => {
                    let icpt = (sy - slope * sx) / n;
                    // residuals sum to zero with the fitted line through the claimed intercept
                    let rsum: f64 = x.iter().zip(&y).map(|(p, q)| q - slope * p - c).sum();
                    if close(icpt, c) && (rsum / n).abs() <= 1e-6 * 1f64.max(c.abs()).max(slope.abs()) {
                        Ok(())
                    } else {
                        rej(format!("claim {claim} != independent intercept {}", fmt(icpt)))
                    }
                }
                _ => {
                    let nume = n * sxy - sx * sy;
                    let dy = n * syy - sy * sy;
                    let r = if dy <= 0.0 { 1.0 } else { nume * nume / (den * dy) };
                    ck(r)
                }
            }
        }
        "molmass" => {
            let m = molar_mass_i(cm, a.s("f").map_err(|e| e.to_string())?)?;
            ck(m)
        }
        "moles" => {
            let m = molar_mass_i(cm, a.s("f").map_err(|e| e.to_string())?)?;
            let mass = cl(a, "mass")?;
            // reverse: claimed moles * molar mass must give the mass
            let back = c * m;
            if (back - mass).abs() <= RTOL * 1f64.max(mass.abs()) {
                Ok(())
            } else {
                rej(format!("moles*M = {} != mass {}", fmt(back), fmt(mass)))
            }
        }
        "stoich" => {
            let eq = a.s("eq").map_err(|e| e.to_string())?;
            let (l, r) = eq.split_once('>').ok_or("equation needs '>'")?;
            let given = a.s("given").map_err(|e| e.to_string())?;
            let want = a.s("want").map_err(|e| e.to_string())?;
            let m = cl(a, "m")?;
            let cg = coef_of(l, r, given).ok_or("given species not in equation")?;
            let cw = coef_of(l, r, want).ok_or("wanted species not in equation")?;
            if !atoms_balance_i(l, r)? {
                return rej("equation does not balance (independent atom count)".into());
            }
            // reverse: extent implied by the claimed product mass must consume exactly the given mass of reagent
            let xi = c / (cw as f64 * molar_mass_i(cm, want)?);
            let consumed = xi * cg as f64 * molar_mass_i(cm, given)?;
            if (consumed - m).abs() <= RTOL * 1f64.max(m.abs()) {
                Ok(())
            } else {
                rej(format!("claimed yield implies {} g of {given} consumed, not {}", fmt(consumed), fmt(m)))
            }
        }
        "limit" => {
            let eq = a.s("eq").map_err(|e| e.to_string())?;
            let (l, r) = eq.split_once('>').ok_or("equation needs '>'")?;
            let (na, nb, want) = (a.s("a").map_err(|e| e.to_string())?, a.s("b").map_err(|e| e.to_string())?, a.s("want").map_err(|e| e.to_string())?);
            let (ma, mb) = (cl(a, "ma")?, cl(a, "mb")?);
            let (ca, cb, cw) = (coef_of(l, r, na).ok_or("a missing")?, coef_of(l, r, nb).ok_or("b missing")?, coef_of(l, r, want).ok_or("want missing")?);
            if !atoms_balance_i(l, r)? {
                return rej("equation does not balance (independent atom count)".into());
            }
            let xi = c / (cw as f64 * molar_mass_i(cm, want)?);
            let ua = xi * ca as f64 * molar_mass_i(cm, na)?;
            let ub = xi * cb as f64 * molar_mass_i(cm, nb)?;
            let tol = 1e-5;
            // feasible (neither reagent over-consumed) and tight (one reagent exhausted)
            let feasible = ua <= ma * (1.0 + tol) && ub <= mb * (1.0 + tol);
            let tight = (ua - ma).abs() <= tol * ma.abs().max(1e-12) || (ub - mb).abs() <= tol * mb.abs().max(1e-12);
            if feasible && tight {
                Ok(())
            } else {
                rej(format!("yield not feasible/tight: uses {:.4} of {:.4} g {na} and {:.4} of {:.4} g {nb}", ua, ma, ub, mb))
            }
        }
        "gas" => {
            let find = a.s("find").map_err(|e| e.to_string())?;
            // R from the exact SI definitions N_A * k_B
            let r = 6.02214076e23 * 1.380649e-23;
            let (mut p, mut v, mut n, mut t) = (cl(a, "p").ok(), cl(a, "v").ok(), cl(a, "n").ok(), cl(a, "t").ok());
            match find {
                "p" => p = Some(c),
                "v" => v = Some(c),
                "n" => n = Some(c),
                "t" => t = Some(c),
                _ => return rej("bad find".into()),
            }
            match (p, v, n, t) {
                (Some(p), Some(v), Some(n), Some(t)) => {
                    let (l, rr) = (p * v, n * r * t);
                    if (l - rr).abs() <= RTOL * l.abs().max(rr.abs()).max(1e-300) {
                        Ok(())
                    } else {
                        rej(format!("PV={} != nRT={}", fmt(l), fmt(rr)))
                    }
                }
                _ => rej("gas needs exactly three of p,v,n,t".into()),
            }
        }
        "kin" => {
            // complete the tuple from the claim by a LINEAR identity, then check all five SUVAT identities
            let names = ["u", "v", "a", "t", "s"];
            let find = a.s("find").map_err(|e| e.to_string())?;
            let mut k: [Option<f64>; 5] = [None; 5];
            for (i, nme) in names.iter().enumerate() {
                k[i] = cl(a, nme).ok();
            }
            let fi = names.iter().position(|n| *n == find).ok_or("bad find")?;
            k[fi] = Some(c);
            let missing: Vec<usize> = (0..5).filter(|i| k[*i].is_none()).collect();
            if missing.len() != 1 {
                return rej("kin needs exactly three givens".into());
            }
            let m = missing[0];
            let g = |i: usize| k[i].unwrap_or(0.0);
            let (u, v, aa, t, s) = (0, 1, 2, 3, 4);
            // try each linear identity that contains the missing variable
            let val: Option<f64> = match m {
                4 => Some(0.5 * (g(u) + g(v)) * g(t)), // s = (u+v) t / 2
                3 => {
                    if g(aa).abs() > 1e-12 {
                        Some((g(v) - g(u)) / g(aa))
                    } else {
                        None
                    }
                }
                2 => {
                    if g(t).abs() > 1e-12 {
                        Some((g(v) - g(u)) / g(t))
                    } else {
                        None
                    }
                }
                1 => Some(g(u) + g(aa) * g(t)),
                _ => Some(g(v) - g(aa) * g(t)),
            };
            let val = val.ok_or("cannot complete the tuple (degenerate)")?;
            k[m] = Some(val);
            let (u, v, aa, t, s2) = (g2(k, 0), g2(k, 1), g2(k, 2), g2(k, 3), g2(k, 4));
            let _ = s;
            let chk = |l: f64, r: f64, nm: &str| -> Result<(), String> {
                let sc = 1f64.max(l.abs()).max(r.abs());
                if (l - r).abs() <= 1e-6 * sc {
                    Ok(())
                } else {
                    rej(format!("identity {nm} violated: {} vs {}", fmt(l), fmt(r)))
                }
            };
            chk(v, u + aa * t, "v=u+at")?;
            chk(s2, 0.5 * (u + v) * t, "s=(u+v)t/2")?;
            chk(v * v, u * u + 2.0 * aa * s2, "v^2=u^2+2as")?;
            chk(s2, u * t + 0.5 * aa * t * t, "s=ut+at^2/2")?;
            chk(s2, v * t - 0.5 * aa * t * t, "s=vt-at^2/2")
        }
        _ => rej(format!("no independent verifier for op {op}")),
    }
}
fn g2(k: [Option<f64>; 5], i: usize) -> f64 {
    k[i].unwrap_or(0.0)
}

fn coef_of(l: &str, r: &str, name: &str) -> Option<i64> {
    for t in l.split('+').chain(r.split('+')) {
        let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
        let f = &t[digits.len()..];
        if f == name {
            return Some(if digits.is_empty() { 1 } else { digits.parse().ok()? });
        }
    }
    None
}
/// independent atomic table (typed again, own structure) and stack-based formula parser
const AT2: [(&str, f64); 29] = [
    ("H", 1.008),
    ("He", 4.0026),
    ("Li", 6.94),
    ("C", 12.011),
    ("N", 14.007),
    ("O", 15.999),
    ("F", 18.998),
    ("Na", 22.99),
    ("Mg", 24.305),
    ("Al", 26.982),
    ("Si", 28.085),
    ("P", 30.974),
    ("S", 32.06),
    ("Cl", 35.45),
    ("K", 39.098),
    ("Ca", 40.078),
    ("Cr", 51.996),
    ("Mn", 54.938),
    ("Fe", 55.845),
    ("Ni", 58.693),
    ("Cu", 63.546),
    ("Zn", 65.38),
    ("Br", 79.904),
    ("Ag", 107.87),
    ("I", 126.9),
    ("Ba", 137.33),
    ("Au", 196.97),
    ("Hg", 200.59),
    ("Pb", 207.2),
];
fn counts_i(f: &str) -> Result<Vec<(String, i64)>, String> {
    // stack machine: each frame is a list of (element, count); '(' pushes, ')' pops and scales by the following integer
    let (main, hyd) = match f.split_once('.') {
        Some((m, h)) => (m, Some(h)),
        None => (f, None),
    };
    let mut out = counts_i_plain(main)?;
    if let Some(h) = hyd {
        let digits: String = h.chars().take_while(|c| c.is_ascii_digit()).collect();
        let mult: i64 = if digits.is_empty() { 1 } else { digits.parse().map_err(|_| "bad hydrate".to_string())? };
        for (k, v) in counts_i_plain(&h[digits.len()..])? {
            out.push((k, v * mult));
        }
    }
    Ok(out)
}
fn counts_i_plain(f: &str) -> Result<Vec<(String, i64)>, String> {
    let ch: Vec<char> = f.chars().collect();
    let mut stack: Vec<Vec<(String, i64)>> = vec![Vec::new()];
    let mut i = 0;
    let read_num = |i: &mut usize| -> i64 {
        let st = *i;
        while *i < ch.len() && ch[*i].is_ascii_digit() {
            *i += 1;
        }
        if *i == st {
            1
        } else {
            ch[st..*i].iter().collect::<String>().parse().unwrap_or(1)
        }
    };
    while i < ch.len() {
        let c = ch[i];
        if c == '(' {
            stack.push(Vec::new());
            i += 1;
        } else if c == ')' {
            i += 1;
            let k = read_num(&mut i);
            let top = stack.pop().ok_or("unbalanced )")?;
            let below = stack.last_mut().ok_or("unbalanced )")?;
            for (e, n) in top {
                below.push((e, n * k));
            }
        } else if c.is_ascii_uppercase() {
            let mut sym = c.to_string();
            i += 1;
            if i < ch.len() && ch[i].is_ascii_lowercase() {
                sym.push(ch[i]);
                i += 1;
            }
            let k = read_num(&mut i);
            stack.last_mut().ok_or("bad")?.push((sym, k));
        } else {
            return Err(format!("unexpected '{c}' in formula"));
        }
    }
    if stack.len() != 1 {
        return Err("missing )".into());
    }
    Ok(stack.pop().unwrap_or_default())
}
fn molar_mass_i(cm: u8, f: &str) -> Result<f64, String> {
    let mut m = 0.0;
    for (e, n) in counts_i(f)? {
        let mut w = AT2.iter().find(|a| a.0 == e).map(|a| a.1).ok_or(format!("unknown element {e}"))?;
        if cm == bug::CM_MASS_O && e == "O" {
            w = 16.0;
        }
        m += w * n as f64;
    }
    Ok(m)
}
fn atoms_balance_i(l: &str, r: &str) -> Result<bool, String> {
    let side = |s: &str| -> Result<BTreeMap<String, i64>, String> {
        let mut m = BTreeMap::new();
        for t in s.split('+') {
            let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
            let k: i64 = if digits.is_empty() { 1 } else { digits.parse().map_err(|_| "bad coefficient".to_string())? };
            for (e, n) in counts_i(&t[digits.len()..])? {
                *m.entry(e).or_insert(0) += n * k;
            }
        }
        Ok(m)
    };
    Ok(side(l)? == side(r)?)
}

/// significant figures, independent of the tool's decimal machine: character walk
fn sf_count_i(s: &str) -> Result<i64, String> {
    let t = s.trim_start_matches(['+', '-']);
    let mant = t.split(['e', 'E']).next().unwrap_or("");
    if mant.is_empty() || !mant.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return Err("bad decimal".into());
    }
    let has_point = mant.contains('.');
    let digits: Vec<char> = mant.chars().filter(|c| *c != '.').collect();
    let first = digits.iter().position(|c| *c != '0');
    let first = match first {
        Some(f) => f,
        None => return Ok(1),
    };
    let mut last = digits.len();
    if !has_point {
        while last > first + 1 && digits[last - 1] == '0' {
            last -= 1;
        }
    }
    Ok((last - first) as i64)
}
fn indep_sig(a: &Args, claim: &str) -> Result<(), String> {
    let xs = a.s("x").map_err(|e| e.to_string())?;
    let x: f64 = xs.parse().map_err(|_| "x not a number".to_string())?;
    let cv: f64 = claim.parse().map_err(|_| format!("claim '{}' is not a number", claim.chars().take(20).collect::<String>()))?;
    if a.op == "sfround" {
        let n = a.int("n", 1, 15).map_err(|e| e.to_string())?;
        if x == 0.0 {
            return rej("cannot round zero".into());
        }
        let e10 = x.abs().log10().floor() as i32;
        let q = 10f64.powi(e10 - n as i32 + 1);
        let truth = (x / q).round() * q;
        if (truth - cv).abs() > 1e-9 * truth.abs() {
            return rej(format!("claim {claim} != {n}-sig-fig rounding {}", fmt(truth)));
        }
        let cnt = sf_count_i(claim)?;
        if cnt != n {
            return rej(format!("claim has {cnt} significant figures, not {n}"));
        }
        Ok(())
    } else {
        // scientific notation: mantissa in [1,10) (or 0), integer exponent, same value, same number of sig figs
        let (m, e) = claim.split_once(['e', 'E']).ok_or("claim must contain e")?;
        let mv: f64 = m.parse().map_err(|_| "bad mantissa".to_string())?;
        let ev: i32 = e.parse().map_err(|_| "bad exponent".to_string())?;
        if x != 0.0 && !(1.0..10.0).contains(&mv.abs()) {
            return rej(format!("mantissa {m} not in [1,10)"));
        }
        if (mv * 10f64.powi(ev) - x).abs() > 1e-12 * x.abs().max(1e-300) {
            return rej(format!("{claim} != {xs}"));
        }
        if sf_count_i(m)? != sf_count_i(xs)? {
            return rej("significant figures not preserved".into());
        }
        Ok(())
    }
}

// ----- Student t quantile through the regularized incomplete beta function (independent of the table)

fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation
    const G: [f64; 6] = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let mut y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for g in G {
        y += 1.0;
        ser += g / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < 1e-300 {
        d = 1e-300;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..400 {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-300 {
            d = 1e-300;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-300 {
            c = 1e-300;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-15 {
            break;
        }
    }
    h
}
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}
/// P(T <= t) for Student's t with `df` degrees of freedom
pub fn t_cdf(t: f64, df: f64) -> f64 {
    let x = df / (df + t * t);
    let p = 0.5 * betai(0.5 * df, 0.5, x);
    if t >= 0.0 {
        1.0 - p
    } else {
        p
    }
}
/// 0.975 quantile by bisection on the cdf
pub fn t_quantile_975(df: i64) -> f64 {
    let (mut lo, mut hi) = (0.0, 1000.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t_cdf(mid, df as f64) < 0.975 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

// ---------------------------------------------------------------------------------------------
// truth by the independent path (used by world generators where no constructive truth exists)

/// the independent algorithm's VALUE for ops where it computes one (None where it is an invariant check)
pub fn indep_value(spec: &str) -> Option<f64> {
    // bisect over the claim using the independent verifier is not needed: closed forms re-use verify_indep via a probe.
    // Probe: ask the tool at bug 0 and confirm the independent verifier accepts; return the tool's value.
    let a = Args::parse(spec).ok()?;
    let o = eval(0, &a).ok()?;
    verify_indep(0, spec, &o.v).ok()?;
    o.v.parse().ok()
}

// ---------------------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn v(spec: &str) -> f64 {
        let a = Args::parse(spec).unwrap();
        eval(0, &a).unwrap().v.parse().unwrap()
    }
    fn vs(spec: &str) -> String {
        let a = Args::parse(spec).unwrap();
        eval(0, &a).unwrap().v
    }
    fn err(spec: &str) -> ToolError {
        let a = Args::parse(spec).unwrap();
        eval(0, &a).unwrap_err()
    }
    fn near(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1f64.max(b.abs())
    }

    #[test]
    fn fin_known_values() {
        assert!(near(v("npv rate=0.1 cf=-100,30,40,50"), -100.0 + 30.0 / 1.1 + 40.0 / 1.21 + 50.0 / 1.331, 1e-12));
        assert!(near(v("compound p=1000 r=0.05 n=10 m=12"), 1000.0 * (1.0 + 0.05 / 12.0f64).powi(120), 1e-12));
        assert!(near(v("contcomp p=100 r=0.05 t=2"), 100.0 * (0.1f64).exp(), 1e-12));
        assert!(near(v("loanpmt p=200000 r=0.005 n=360"), 1199.101, 1e-6)); // 6% / 30y mortgage
        assert!(near(v("annpv pmt=100 r=0.05 n=10"), 772.1735, 1e-6));
        assert!(near(v("bondprice face=1000 cpn=0.05 y=0.05 n=10 f=2"), 1000.0, 1e-9)); // par bond
        assert!(near(v("bondprice face=1000 cpn=0.06 y=0.05 n=10 f=2"), 1077.9457, 1e-6));
        assert!(near(v("cagr s=100 e=200 n=5"), 2f64.powf(0.2) - 1.0, 1e-12));
        assert!(near(v("wacc e=600 d=400 re=0.1 rd=0.05 t=0.2"), 0.6 * 0.1 + 0.4 * 0.05 * 0.8, 1e-12));
        assert!(near(v("breakeven fc=10000 price=25 vc=15"), 1000.0, 1e-12));
        assert!(near(v("payoff kind=call pos=long s=110 k=100 prem=4"), 6.0, 1e-12));
        assert!(near(v("payoff kind=put pos=short s=110 k=100 prem=3"), 3.0, 1e-12));
        assert!(near(v("perp c=50 r=0.05"), 1000.0, 1e-12));
        assert!(near(v("gperp c=50 r=0.08 g=0.03"), 1000.0, 1e-12));
        assert!(near(v("pctchg old=80 new=100"), 25.0, 1e-12));
        assert!(near(v("chain rates=10,10"), 21.0, 1e-9));
        assert!(near(v("elast q1=100 q2=80 p1=10 p2=12"), (-20.0 / 90.0) / (2.0 / 11.0), 1e-12));
        assert!(near(v("real nominal=1100 cpi=110 base=100"), 1000.0, 1e-12));
        assert!(near(v("fisher nom=0.08 infl=0.03"), 1.08 / 1.03 - 1.0, 1e-12));
        assert!(near(v("payback cf=-100,40,40,40"), 2.5, 1e-12));
        assert!(near(v("current ca=500 cl=250"), 2.0, 1e-12));
        assert!(near(v("quick ca=500 inv=100 cl=200"), 2.0, 1e-12));
    }

    #[test]
    fn irr_converges_and_reports() {
        let a = Args::parse("irr cf=-1000,300,400,500,200").unwrap();
        let o = eval(0, &a).unwrap();
        let m = parse_out(&o.render());
        let r: f64 = m["v"].parse().unwrap();
        assert_eq!(m["conv"], "1");
        assert!(m["resid"].parse::<f64>().unwrap() < 1e-8);
        // NPV(IRR) = 0 by an independent Horner evaluation
        let cf = [-1000.0, 300.0, 400.0, 500.0, 200.0];
        let mut s = 0.0;
        for x in cf.iter().rev() {
            s = x + s / (1.0 + r);
        }
        assert!(s.abs() < 1e-8, "{s}");
        // no sign change -> clean error, not a number
        assert!(matches!(err("irr cf=100,200,300"), ToolError::Domain(_)));
        // exact: -100, 0, 121 -> 10%
        assert!(near(v("irr cf=-100,0,121"), 0.1, 1e-10));
    }

    #[test]
    fn amortisation_sums_to_principal() {
        let (p, r, n) = (250000.0, 0.004, 240);
        let pmt = v("loanpmt p=250000 r=0.004 n=240");
        let mut bal = p;
        let mut prin = 0.0;
        for _ in 0..n {
            let i = bal * r;
            prin += pmt - i;
            bal -= pmt - i;
        }
        assert!(bal.abs() < 1e-6 && (prin - p).abs() < 1e-6);
        let b120 = v("amort p=250000 r=0.004 n=240 k=120");
        let mut bal = p;
        for _ in 0..120 {
            bal = bal * (1.0 + r) - pmt;
        }
        assert!(near(b120, bal, 1e-9));
    }

    #[test]
    fn bond_duration_matches_numeric_derivative_and_yield_roundtrip() {
        let spec_p = "bondprice face=1000 cpn=0.06 y=0.07 n=8 f=2";
        let price = v(spec_p);
        let y = v(&format!("bondyield face=1000 cpn=0.06 price={} n=8 f=2", fmt(price)));
        assert!(near(y, 0.07, 1e-9), "{y}");
        let dm = v("bondmod face=1000 cpn=0.06 y=0.07 n=8 f=2");
        let h = 1e-5;
        let pu = v(&format!("bondprice face=1000 cpn=0.06 y={} n=8 f=2", fmt(0.07 + h)));
        let pd = v(&format!("bondprice face=1000 cpn=0.06 y={} n=8 f=2", fmt(0.07 - h)));
        assert!(near(dm, -(pu - pd) / (2.0 * h * price), 1e-6));
        let dmac = v("bonddur face=1000 cpn=0.06 y=0.07 n=8 f=2");
        assert!(near(dmac, dm * (1.0 + 0.035), 1e-9));
        // zero-coupon bond: Macaulay duration = maturity
        assert!(near(v("bonddur face=1000 cpn=0 y=0.05 n=7 f=1"), 7.0, 1e-9));
    }

    #[test]
    fn put_call_parity() {
        // C - P = S - K e^{-rT}: build a consistent triple and expect residual 0
        let (s, k, r, t) = (100.0, 95.0, 0.04, 0.5);
        let c = 11.0;
        let p = c - (s - k * (-r * t as f64).exp());
        assert!(v(&format!("parity c={} p={} s=100 k=95 r=0.04 t=0.5", fmt(c), fmt(p))).abs() < 1e-9);
        assert!(near(v(&format!("putfromcall c={} s=100 k=95 r=0.04 t=0.5", fmt(c))), p, 1e-9));
    }

    #[test]
    fn units_dimensions_and_roundtrip() {
        assert!(near(v("uconv x=1 from=kg*m/s^2 to=N"), 1.0, 1e-12));
        assert!(near(v("uconv x=1 from=N*m to=J"), 1.0, 1e-12));
        assert!(near(v("uconv x=100 from=km/h to=m/s"), 100.0 / 3.6, 1e-12));
        assert!(near(v("uconv x=1 from=psi to=Pa"), 6894.757293168, 1e-9));
        assert!(near(v("uconv x=60 from=mi/h to=ft/s"), 88.0, 1e-9));
        assert!(near(v("uconv x=1 from=(m/s)^2 to=J/kg"), 1.0, 1e-12));
        assert!(near(v("uconv x=1 from=kWh to=MJ"), 3.6, 1e-12));
        assert!(near(v("uconv x=1 from=W/A to=V"), 1.0, 1e-12));
        assert!(matches!(err("uconv x=1 from=m to=s"), ToolError::Dimension(_)));
        assert!(matches!(err("uconv x=1 from=furlong to=m"), ToolError::Unknown(_)));
        assert!(matches!(err("uconv x=1 from=m/(s to=m"), ToolError::Parse(_)));
        // left-associative division: J/kg*K  is (J/kg)*K  -> dims differ from J/(kg*K)
        assert_eq!(v("udim a=J/kg*K b=J/(kg*K)"), 0.0);
        assert_eq!(v("udim a=J/kg/K b=J/(kg*K)"), 1.0);
        // round trip through the independent exact table for a sample of conversions
        for (x, f, t) in [("3.5", "lb", "kg"), ("12", "ft/s", "km/h"), ("2.2", "atm", "psi"), ("9", "knot", "mi/h"), ("0.5", "kcal/h", "W")] {
            let spec = format!("uconv x={x} from={f} to={t}");
            let out = vs(&spec);
            verify_indep(0, &spec, &out).unwrap();
        }
    }

    #[test]
    fn significant_figures_and_scientific() {
        assert_eq!(v("sfcount x=0.00120"), 3.0);
        assert_eq!(v("sfcount x=1200"), 2.0);
        assert_eq!(v("sfcount x=1200."), 4.0);
        assert_eq!(v("sfcount x=1.20e-3"), 3.0);
        assert_eq!(vs("sfround x=123456 n=3"), "1.23e5");
        assert_eq!(vs("sfround x=0.0012367 n=3"), "0.00124");
        assert_eq!(vs("sfround x=9.996 n=3"), "10.0");
        assert_eq!(vs("sfround x=2.5 n=3"), "2.50");
        assert_eq!(vs("sfround x=50.429 n=2"), "5.0e1"); // found by the independent verifier: "50" would have 1 sig fig
        assert_eq!(vs("sfround x=89.9707 n=2"), "9.0e1");
        assert_eq!(vs("sfround x=1234.5 n=4"), "1235");
        assert_eq!(vs("sci x=0.000456"), "4.56e-4");
        assert_eq!(vs("sci x=12000"), "1.2e4");
        for spec in ["sfround x=123456 n=3", "sfround x=0.0012367 n=3", "sfround x=9.996 n=3", "sci x=0.000456", "sci x=12.30"] {
            let a = Args::parse(spec).unwrap();
            let o = eval(0, &a).unwrap();
            verify_indep(0, spec, &o.v).unwrap();
        }
    }

    #[test]
    fn error_propagation_and_statistics() {
        assert!(near(v("errprop f=add x=10 sx=0.3 y=5 sy=0.4"), 0.5, 1e-12));
        assert!(near(v("errprop f=mul x=10 sx=0.1 y=5 sy=0.1"), 50.0 * ((0.01f64).powi(2) + (0.02f64).powi(2)).sqrt(), 1e-12));
        assert!(near(v("errprop f=pow x=4 sx=0.1 n=2"), 0.8, 1e-12));
        assert!(near(v("mean x=2,4,4,4,5,5,7,9"), 5.0, 1e-12));
        assert!(near(v("var x=2,4,4,4,5,5,7,9"), 32.0 / 7.0, 1e-12));
        assert!(near(v("sd x=1,2,3,4,5"), 2.5f64.sqrt(), 1e-12));
        assert!(near(v("median x=3,1,2,10"), 2.5, 1e-12));
        assert!(near(v("slope x=1,2,3,4 y=2,4,6,8"), 2.0, 1e-12));
        assert!(near(v("intercept x=1,2,3,4 y=3,5,7,9"), 1.0, 1e-12));
        assert!(near(v("r2 x=1,2,3,4 y=2,4,6,8"), 1.0, 1e-12));
        // ci95 half width for n=5: t(4)=2.7764
        let a = Args::parse("ci95 x=1,2,3,4,5").unwrap();
        let hw: f64 = eval(0, &a).unwrap().v.parse().unwrap();
        assert!(near(hw, 2.7764 * 2.5f64.sqrt() / 5f64.sqrt(), 1e-9));
    }

    #[test]
    fn t_table_matches_incomplete_beta_inversion() {
        for (df, t) in T975 {
            let q = t_quantile_975(*df);
            assert!((q - t).abs() < 6e-4, "df {df}: table {t} vs beta {q}");
        }
        // interpolated rows stay within the documented 5e-4 relative tolerance
        for df in [35, 45, 50, 80, 100, 150, 500] {
            let q = t_quantile_975(df);
            assert!((t975(df) - q).abs() <= 5e-4 * q, "df {df}: {} vs {}", t975(df), q);
        }
    }

    #[test]
    fn chemistry_and_physics() {
        assert!(near(v("molmass f=H2O"), 18.015, 1e-9));
        assert!(near(v("molmass f=C6H12O6"), 180.156, 1e-6));
        assert!(near(v("molmass f=Ca(OH)2"), 74.092, 1e-6));
        assert!(near(v("molmass f=Al2(SO4)3"), 342.15, 1e-4));
        assert!(near(v("molmass f=CuSO4.5H2O"), 249.677, 1e-5));
        assert!(near(v("moles mass=36.03 f=H2O"), 36.03 / 18.015, 1e-12));
        // 4 g H2 + O2 -> H2O : 36 g
        assert!(near(v("stoich eq=2H2+O2>2H2O given=H2 m=4.032 want=H2O"), 4.032 / 2.016 * 18.015, 1e-9));
        // limiting reagent
        let y = v("limit eq=2H2+O2>2H2O a=H2 ma=10 b=O2 mb=40 want=H2O");
        assert!(near(y, 40.0 / 31.998 * 2.0 * 18.015, 1e-9)); // O2 is limiting
        assert!(matches!(err("stoich eq=H2+O2>H2O given=H2 m=1 want=H2O"), ToolError::Domain(_))); // unbalanced
        assert!(near(v("gas find=p v=0.0224 n=1 t=273.15"), 8.314462618 * 273.15 / 0.0224, 1e-9));
        assert!(near(v("gas find=t p=101325 v=0.0224 n=1"), 101325.0 * 0.0224 / 8.314462618, 1e-9));
        assert!(near(v("kin u=0 a=9.8 t=3 find=s"), 44.1, 1e-9));
        assert!(near(v("kin u=0 v=20 a=4 find=t"), 5.0, 1e-9));
        assert!(near(v("kin u=2 a=3 s=10 find=v"), (4.0f64 + 60.0).sqrt(), 1e-9));
        assert!(matches!(err("kin u=1 v=2 a=3 t=4 find=s"), ToolError::BadInput(_)));
    }

    #[test]
    fn independent_verifier_accepts_clean_and_rejects_wrong() {
        let specs = [
            "npv rate=0.08 cf=-500,120,150,180,200",
            "irr cf=-1000,300,400,500,200",
            "payback cf=-100,40,40,40,40",
            "compound p=2500 r=0.045 n=7 m=4",
            "loanpmt p=18000 r=0.006 n=60",
            "amort p=18000 r=0.006 n=60 k=24",
            "bondprice face=1000 cpn=0.045 y=0.052 n=6 f=2",
            "bondyield face=1000 cpn=0.045 price=950 n=6 f=2",
            "bonddur face=1000 cpn=0.045 y=0.052 n=6 f=2",
            "cagr s=50 e=130 n=6",
            "wacc e=700 d=300 re=0.11 rd=0.06 t=0.25",
            "pctchg old=120 new=90",
            "breakeven fc=45000 price=30 vc=18",
            "perp c=75 r=0.06",
            "gperp c=75 r=0.09 g=0.02",
            "elast q1=200 q2=150 p1=5 p2=6",
            "uconv x=55 from=mi/h to=m/s",
            "errprop f=div x=8 sx=0.2 y=4 sy=0.1",
            "var x=3,5,7,11,13",
            "ci95 x=3,5,7,11,13,17",
            "slope x=1,2,3,5,8 y=1.2,1.9,3.2,4.8,8.1",
            "molmass f=Mg(NO3)2",
            "stoich eq=N2+3H2>2NH3 given=N2 m=28 want=NH3",
            "limit eq=2Al+3Cl2>2AlCl3 a=Al ma=10 b=Cl2 mb=30 want=AlCl3",
            "gas find=v p=200000 n=2 t=310",
            "kin u=3 v=15 t=4 find=s",
        ];
        for s in specs {
            let a = Args::parse(s).unwrap();
            let o = eval(0, &a).unwrap_or_else(|e| panic!("{s}: {e}"));
            verify_indep(0, s, &o.v).unwrap_or_else(|e| panic!("clean output rejected for '{s}': {e} (claim {})", o.v));
            // a wrong claim (3% off) is rejected
            let x: f64 = match o.v.parse() {
                Ok(x) => x,
                Err(_) => continue,
            };
            let wrong = fmt(x * 1.03 + 0.01);
            assert!(verify_indep(0, s, &wrong).is_err(), "wrong claim {wrong} accepted for '{s}'");
        }
    }

    #[test]
    fn mutants_are_visible_to_independent_but_not_shared() {
        // for a selection of bugs on a spec where they manifest: the bugged output differs from clean,
        // the shared verifier accepts it (same code), the independent verifier rejects it
        let cases: &[(u8, &str)] = &[
            (bug::NPV_T_OFF1, "npv rate=0.08 cf=-500,120,150,180,200"),
            (bug::COMP_SIMPLE, "compound p=2500 r=0.045 n=7 m=4"),
            (bug::LOAN_OFF1, "loanpmt p=18000 r=0.006 n=60"),
            (bug::BOND_DUR_PERIODS, "bonddur face=1000 cpn=0.045 y=0.052 n=6 f=2"),
            (bug::WACC_NO_TAX, "wacc e=700 d=300 re=0.11 rd=0.06 t=0.25"),
            (bug::UNIT_DIV_ASSOC, "uconv x=5 from=ft/s*min to=m/s*h"),
            (bug::UNIT_PSI, "uconv x=30 from=psi to=kPa"),
            (bug::VAR_POP, "var x=3,5,7,11,13"),
            (bug::KIN_NO_HALF, "kin u=3 a=2 t=4 find=s"),
            (bug::GAS_R_TYPO, "gas find=v p=200000 n=2 t=310"),
            (bug::MOLMASS_PAREN, "molmass f=Mg(NO3)2"),
        ];
        for (b, s) in cases {
            let a = Args::parse(s).unwrap();
            let clean = eval(0, &a).unwrap().v;
            let dirty = eval(*b, &a).unwrap().v;
            assert!(claims_equal(&clean, &dirty).is_err(), "bug {b} did not change '{s}'");
            assert!(verify_shared(*b, s, &dirty).is_ok(), "shared verifier unexpectedly rejects bug {b}");
            assert!(verify_indep(0, s, &dirty).is_err(), "independent verifier missed bug {b} on '{s}'");
        }
    }

    #[test]
    fn common_mode_bug_is_invisible() {
        let s = "uconv x=10 from=lb to=kg";
        let a = Args::parse(s).unwrap();
        let dirty = eval(bug::CM_UNIT_LB, &a).unwrap().v;
        assert!(claims_equal(&eval(0, &a).unwrap().v, &dirty).is_err());
        assert!(verify_indep(bug::CM_UNIT_LB, s, &dirty).is_ok(), "common-mode bug should fool the independent verifier");
        assert!(verify_indep(0, s, &dirty).is_err());
    }

    #[test]
    fn guards_and_errors_are_clean() {
        for bad in [
            "",
            "npv",
            "npv rate=abc cf=1,2",
            "npv rate=0.1",
            "npv rate=-1 cf=1,2",
            "npv rate=nan cf=1,2",
            "npv rate=inf cf=1,2",
            "npv rate=1e400 cf=1,2",
            "compound p=1 r=0.1 n=1e9 m=12",
            "contcomp p=1 r=1000 t=1000",
            "bondprice face=1000 cpn=0.05 y=0.05 n=10.3 f=2",
            "uconv x=1 from=m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m/m to=m",
            "molmass f=Xx2",
            "molmass f=((((((((H))))))))",
            "kin u=1 v=2 find=a",
            "payback cf=-100,1,1,1",
            "irr cf=1,2,3",
            "foo a=1",
        ] {
            let r = FinTool::default().call(bad).or_else(|_| SciTool::default().call(bad)).or_else(|_| EconTool::default().call(bad));
            assert!(r.is_err(), "'{bad}' should be an error, got {:?}", r);
        }
        // huge-ish but legal values do not produce NaN/inf outputs
        let o = FinTool::default().call("compound p=1e14 r=0.5 n=40 m=1");
        assert!(matches!(o, Err(ToolError::Overflow)) || o.is_ok());
        if let Ok(s) = o {
            assert!(parse_out(&s)["v"].parse::<f64>().unwrap().is_finite());
        }
        // wrong tool is a clean error
        assert!(FinTool::default().call("uconv x=1 from=m to=km").is_err());
        assert!(SciTool::default().call("npv rate=0.1 cf=1,2").is_err());
    }

    #[test]
    fn determinism_and_registry_integration() {
        use crate::tools::Registry;
        let mut reg = Registry::new(true);
        reg.register(Box::new(FinTool::default()));
        reg.register(Box::new(SciTool::default()));
        reg.register(Box::new(EconTool::default()));
        reg.register(Box::new(DomainVerifier { indep: true, bug: 0 }));
        reg.register(Box::new(DomainVerifier { indep: false, bug: 0 }));
        let o1 = reg.call("fin", "npv rate=0.1 cf=-100,60,60");
        let o2 = reg.call("fin", "npv   rate=0.1  cf=-100,60,60");
        assert_eq!(o1.result, o2.result);
        assert!(o2.cached);
        let claim = parse_out(&o1.result.clone().unwrap())["v"].clone();
        let vi = reg.call("dverify_indep", &format!("npv rate=0.1 cf=-100,60,60 => {claim}"));
        assert_eq!(vi.result.unwrap(), "accept");
        let vr = reg.call("dverify_indep", "npv rate=0.1 cf=-100,60,60 => 12");
        assert!(vr.result.unwrap().starts_with("reject"));
        // irr is metered by iterations and deterministic
        let t = FinTool::default();
        let (r1, c1) = t.call_metered("irr cf=-1000,300,400,500,200");
        let (r2, c2) = t.call_metered("irr cf=-1000,300,400,500,200");
        assert_eq!((r1, c1), (r2, c2));
    }

    #[test]
    fn bug_hook_zero_is_production() {
        // FinTool::default() (bug 0) equals eval(0): sample many generated inputs, outputs identical and verifier-clean
        let mut rng = crate::router::Rng::new(5);
        for _ in 0..300 {
            let r = 0.01 + 0.2 * rng.f64();
            let n = 3 + rng.below(6);
            let mut cf = vec![-(100.0 + 900.0 * rng.f64()).round()];
            for _ in 0..n {
                cf.push((20.0 + 300.0 * rng.f64()).round());
            }
            let list: Vec<String> = cf.iter().map(|x| fmt(*x)).collect();
            let spec = format!("npv rate={} cf={}", fmt((r * 1000.0).round() / 1000.0), list.join(","));
            let a = FinTool::default().call(&spec).unwrap();
            let b = eval(0, &Args::parse(&spec).unwrap()).unwrap().render();
            assert_eq!(a, b);
            verify_indep(0, &spec, &parse_out(&a)["v"]).unwrap();
        }
    }
}
