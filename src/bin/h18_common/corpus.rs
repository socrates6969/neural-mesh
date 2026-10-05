//! h18 canonical-spec CORPUS over ALL 62 `fin` ops (used by selfcheck + mutation testing). Specs for the 36 word-problem ops
//! come from `nl::gen_params`; the other ops, plus cases that exercise specific seeded bugs (high IRR, mixed-structure
//! unit expressions, parenthesised powers, trailing-zero decimals), are generated here.
use super::nl::{gen_params, op_def};
use neural_mesh::fin::{self, fmt};
use neural_mesh::router::Rng;

fn ri(r: &mut Rng, lo: i64, hi: i64) -> i64 {
    lo + r.below((hi - lo + 1) as usize) as i64
}
fn lst(r: &mut Rng, n: usize, lo: f64, hi: f64) -> String {
    (0..n).map(|_| fmt(((lo + (hi - lo) * r.f64()) * 10.0).round() / 10.0)).collect::<Vec<_>>().join(",")
}

pub fn all_ops() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    v.extend(fin::FIN_OPS.iter());
    v.extend(fin::SCI_OPS.iter());
    v.extend(fin::ECON_OPS.iter());
    v
}

pub fn gen_any(op: &'static str, r: &mut Rng) -> Option<String> {
    let rate = |r: &mut Rng, lo: f64, hi: f64| fmt(((lo + (hi - lo) * r.f64()) * 200.0).round() / 200.0);
    // special cases that exercise seeded bugs
    if op == "irr" && r.f64() < 0.2 {
        let c0 = ri(r, 5, 20) * 100;
        return Some(format!("irr cf=-{c0},{},{},{}", c0 * 8 / 10, c0 * 9 / 10, c0));
    }
    if op == "uconv" && r.f64() < 0.15 {
        let ls = ["m", "km", "cm", "ft", "mi", "in"];
        let ts = ["s", "min", "h"];
        let (l1, l2) = (ls[r.below(6)], ls[r.below(6)]);
        let (t1, t2, t3, t4) = (ts[r.below(3)], ts[r.below(3)], ts[r.below(3)], ts[r.below(3)]);
        let (from, to) = if r.f64() < 0.5 { (format!("{l1}/{t1}*{t2}"), format!("{l2}/{t3}*{t4}")) } else { (format!("({l1}/{t1})^2"), format!("({l2}/{t3})^2")) };
        return Some(format!("uconv x={} from={from} to={to}", ri(r, 2, 99)));
    }
    if op == "sfcount" {
        let xs = ["1.20", "0.0450", "3.100", "1200", "1200.", "0.00120", "45.60", "100.0", "2.5e-3", "7.80e4"];
        return Some(format!("sfcount x={}", xs[r.below(xs.len())]));
    }
    if let Some(_d) = op_def(op) {
        let d = r.f64();
        let st = r.below(2) as u8;
        return gen_params(op, r, d, st).map(|p| p.spec());
    }
    Some(match op {
        "contcomp" => format!("contcomp p={} r={} t={}", ri(r, 10, 500) * 100, rate(r, 0.02, 0.09), ri(r, 1, 12)),
        "annfv" => format!("annfv pmt={} r={} n={}{}", ri(r, 2, 100) * 50, rate(r, 0.02, 0.09), ri(r, 3, 30), if r.f64() < 0.3 { " due=1" } else { "" }),
        "quick" => format!("quick ca={} inv={} cl={}", ri(r, 200, 900) * 10, ri(r, 10, 190) * 10, ri(r, 10, 900) * 10),
        "roa" => format!("roa ni={} ta={}", ri(r, 10, 900) * 10, ri(r, 1000, 9000) * 10),
        "omargin" => format!("omargin rev={} cogs={} opex={}", ri(r, 500, 900) * 10, ri(r, 100, 300) * 10, ri(r, 50, 150) * 10),
        "debteq" => format!("debteq debt={} eq={}", ri(r, 10, 900) * 10, ri(r, 10, 900) * 10),
        "icr" => format!("icr ebit={} interest={}", ri(r, 100, 900) * 10, ri(r, 10, 90) * 10),
        "dscr" => format!("dscr noi={} ds={}", ri(r, 100, 900) * 10, ri(r, 10, 90) * 10),
        "pct" => format!("pct part={} whole={}", ri(r, 1, 90) * 5, ri(r, 100, 900)),
        "pctof" => format!("pctof p={} x={}", ri(r, 1, 80), ri(r, 10, 900) * 10),
        "chain" => format!("chain rates={}", (0..ri(r, 2, 5)).map(|_| fmt(ri(r, -20, 30) as f64 / 2.0)).collect::<Vec<_>>().join(",")),
        "ppoints" => format!("ppoints a={} b={}", ri(r, 1, 60) as f64 / 2.0, ri(r, 1, 60) as f64 / 2.0),
        "parity" => format!("parity c={} p={} s={} k={} r={} t={}", ri(r, 5, 25), ri(r, 3, 20), ri(r, 80, 120), ri(r, 80, 120), rate(r, 0.01, 0.08), fmt(ri(r, 1, 8) as f64 / 4.0)),
        "putfromcall" => format!("putfromcall c={} s={} k={} r={} t={}", ri(r, 5, 25), ri(r, 80, 120), ri(r, 80, 120), rate(r, 0.01, 0.08), fmt(ri(r, 1, 8) as f64 / 4.0)),
        "gperp" => {
            let g = ri(r, 1, 6) as f64 / 100.0;
            format!("gperp c={} r={} g={}", ri(r, 5, 90) * 10, fmt(g + ri(r, 2, 8) as f64 / 100.0), fmt(g))
        }
        "bonddur" => format!("bonddur face=1000 cpn={} y={} n={} f={}", rate(r, 0.02, 0.09), rate(r, 0.02, 0.1), ri(r, 2, 9), [1, 2][r.below(2)]),
        "var" => {
            let n = ri(r, 4, 10) as usize;
            format!("var x={}", lst(r, n, 10.0, 90.0))
        }
        "median" => {
            let n = ri(r, 4, 11) as usize;
            format!("median x={}", lst(r, n, 10.0, 90.0))
        }
        "sem" => {
            let n = ri(r, 4, 10) as usize;
            format!("sem x={}", lst(r, n, 10.0, 90.0))
        }
        "intercept" | "r2" => {
            let n = ri(r, 5, 8) as usize;
            let xs: Vec<f64> = (1..=n).map(|i| i as f64).collect();
            let a = ri(r, -4, 8) as f64 / 2.0;
            let ys: Vec<String> = xs.iter().map(|x| fmt(((a * x + 5.0 + (r.f64() - 0.5) * 4.0) * 10.0).round() / 10.0)).collect();
            format!("{op} x={} y={}", xs.iter().map(|x| fmt(*x)).collect::<Vec<_>>().join(","), ys.join(","))
        }
        "moles" => {
            let fs = ["H2O", "CO2", "NaCl", "C6H12O6", "Ca(OH)2", "Fe2O3", "H2SO4", "KMnO4"];
            format!("moles mass={} f={}", ri(r, 10, 2000) as f64 / 10.0, fs[r.below(fs.len())])
        }
        "limit" => {
            let rx = [("2H2+O2>2H2O", "H2", "O2", "H2O"), ("N2+3H2>2NH3", "N2", "H2", "NH3"), ("2Al+3Cl2>2AlCl3", "Al", "Cl2", "AlCl3"), ("CH4+2O2>CO2+2H2O", "CH4", "O2", "H2O"), ("2Mg+O2>2MgO", "Mg", "O2", "MgO")];
            let (eq, a, b, w) = rx[r.below(rx.len())];
            format!("limit eq={eq} a={a} ma={} b={b} mb={} want={w}", ri(r, 20, 800) as f64 / 10.0, ri(r, 20, 800) as f64 / 10.0)
        }
        "udim" => {
            let ps = [("J", "N*m"), ("W", "J/s"), ("Pa", "N/m^2"), ("kg*m/s^2", "N"), ("m/s", "km/h"), ("J", "W*s"), ("V", "W/A"), ("m", "s")];
            let (a, b) = ps[r.below(ps.len())];
            format!("udim a={a} b={b}")
        }
        "fisher" => format!("fisher nom={} infl={}", rate(r, 0.02, 0.12), rate(r, 0.0, 0.08)),
        _ => return None,
    })
}

/// a per-seed corpus: `per_op` specs for every op (specs whose clean evaluation errors are skipped)
pub fn corpus(seed: u64, per_op: usize) -> Vec<(&'static str, String)> {
    let mut r = Rng::new(seed.wrapping_mul(9_176_393).wrapping_add(11));
    let mut out = Vec::new();
    for op in all_ops() {
        let mut got = 0;
        let mut tries = 0;
        while got < per_op && tries < per_op * 20 {
            tries += 1;
            if let Some(spec) = gen_any(op, &mut r) {
                if let Ok(a) = fin::Args::parse(&spec) {
                    if fin::eval(0, &a).is_ok() {
                        out.push((op, spec));
                        got += 1;
                    }
                }
            }
        }
    }
    out
}
