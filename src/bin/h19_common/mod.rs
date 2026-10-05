//! h19 (R9-C): LOCAL DETERMINISTIC document corpora (research-paper-like / financial statements / business documents),
//! hand-written std-only EXTRACTORS, an independent source-span + cross-footing VERIFIER, and a downstream
//! question-answering / session simulation. Everything is seeded and synthetic unless a result is labelled REAL.
//! No network, no downloads. `src/*.rs` untouched; only the repo's Router / Rng / stats are reused.
#![allow(dead_code)]
pub mod gens;
pub mod genb;
pub mod scan;
pub mod verify;
pub mod down;
pub mod cal;

use neural_mesh::router::Rng;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum World {
    Paper,
    Fin,
    Biz,
    /// control: prose, no structure needed
    NoStruct,
    /// control: tiny doc, single clean fact, extraction unnecessary
    Tiny,
}
impl World {
    pub fn name(&self) -> &'static str {
        match self {
            World::Paper => "paper",
            World::Fin => "financial",
            World::Biz => "business",
            World::NoStruct => "nostruct",
            World::Tiny => "tiny",
        }
    }
    pub fn idx(&self) -> usize {
        match self {
            World::Paper => 0,
            World::Fin => 1,
            World::Biz => 2,
            World::NoStruct => 3,
            World::Tiny => 4,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Bank {
    Dev,
    Unseen,
}

/// field types (per-field-type P/R/F1)
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum FKind {
    Cell,
    Param,
    Claim,
    Line,
    Date,
    Amount,
    Party,
    Kpi,
    Clause,
}
pub const FKINDS: [FKind; 9] = [FKind::Cell, FKind::Param, FKind::Claim, FKind::Line, FKind::Date, FKind::Amount, FKind::Party, FKind::Kpi, FKind::Clause];
impl FKind {
    pub fn name(&self) -> &'static str {
        match self {
            FKind::Cell => "table_cell",
            FKind::Param => "param",
            FKind::Claim => "claim",
            FKind::Line => "line_item",
            FKind::Date => "date",
            FKind::Amount => "amount",
            FKind::Party => "party",
            FKind::Kpi => "kpi",
            FKind::Clause => "clause",
        }
    }
}

/// a ground-truth (or extracted) entity-attribute-value fact. `key` = canonical "attr|entity/period".
#[derive(Clone, Debug)]
pub struct Fact {
    pub key: String,
    pub kind: FKind,
    pub value: f64,
    pub sval: String,
    pub unit: String,
    pub lo: usize,
    pub hi: usize,
}
impl Fact {
    pub fn num(key: &str, kind: FKind, value: f64, unit: &str, lo: usize, hi: usize) -> Fact {
        Fact { key: key.to_string(), kind, value, sval: String::new(), unit: unit.to_string(), lo, hi }
    }
    pub fn matches(&self, o: &Fact) -> bool {
        self.key == o.key && self.unit == o.unit && self.sval == o.sval && close(self.value, o.value)
    }
}
pub fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QOp {
    /// v0 / v1
    Ratio,
    /// (v0 - v1) / v1
    Growth,
    /// v0 - v1
    Diff,
    /// index of the max of the values (min if `lower`)
    ArgBest(bool),
    /// number of supported firm claims (values are 0/1 support flags)
    CountSupported,
    /// value lookup (v0)
    Lookup,
    /// v0 * v1
    Prod,
    /// end date: v0 (days) + v1 months
    EndDate,
    /// qualitative question, no numbers needed
    Qual,
}

#[derive(Clone, Debug)]
pub struct Question {
    pub op: QOp,
    /// keys of the facts needed (for claim counts: the claim keys)
    pub keys: Vec<String>,
    pub truth: f64,
    pub text: String,
}
pub fn qeval(op: QOp, v: &[f64]) -> Option<f64> {
    match op {
        QOp::Ratio => (v.len() >= 2 && v[1] != 0.0).then(|| v[0] / v[1]),
        QOp::Growth => (v.len() >= 2 && v[1] != 0.0).then(|| (v[0] - v[1]) / v[1]),
        QOp::Diff => (v.len() >= 2).then(|| v[0] - v[1]),
        QOp::ArgBest(lower) => {
            if v.is_empty() {
                return None;
            }
            let mut b = 0;
            for i in 1..v.len() {
                if (lower && v[i] < v[b]) || (!lower && v[i] > v[b]) {
                    b = i;
                }
            }
            Some(b as f64)
        }
        QOp::CountSupported => Some(v.iter().sum()),
        QOp::Lookup => v.first().copied(),
        QOp::Prod => (v.len() >= 2).then(|| v[0] * v[1]),
        QOp::EndDate => (v.len() >= 2 && v[1] >= 0.0 && v[1] < 1200.0).then(|| add_months(v[0] as i64, v[1].round() as i64) as f64),
        QOp::Qual => None,
    }
}

#[derive(Clone, Debug)]
pub struct Claim {
    pub key: String,
    pub a: String,
    pub b: String,
    pub metric: String,
    pub pos: bool,
    pub hedged: bool,
    pub cite: u32,
    pub supported: bool,
}

#[derive(Clone, Debug)]
pub struct Doc {
    pub world: World,
    pub bank: Bank,
    /// layout difficulty 0..4 (5 = adversarial controls)
    pub layout: u8,
    pub adv: u8,
    pub text: String,
    pub facts: Vec<Fact>,
    pub claims: Vec<Claim>,
    pub qs: Vec<Question>,
    /// numeric tokens that are NOT facts (distractors)
    pub n_distract: usize,
    /// doc contains scale / unit / restatement confusion a reader can trip on
    pub scale_conf: bool,
    pub structured: bool,
    /// number of claim sentences (for the coverage check ground truth)
    pub n_claim_sents: usize,
}
impl Doc {
    pub fn fact(&self, key: &str) -> Option<&Fact> {
        self.facts.iter().find(|f| f.key == key)
    }
    pub fn n_chars(&self) -> usize {
        self.text.len()
    }
}

// ---------------------------------------------------------------- rng helpers
pub fn hash(a: u64, b: u64, c: u64, d: u64) -> u64 {
    let mut z = a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_add(0x1234_5678_9ABC_DEF1).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= c.wrapping_mul(0x94D0_49BB_1331_11EB).rotate_left(17) ^ d.wrapping_mul(0xD6E8_FEB8_6659_FD93).rotate_left(31);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
pub struct R(pub Rng);
impl R {
    pub fn new(a: u64, b: u64, c: u64, d: u64) -> R {
        R(Rng::new(hash(a, b, c, d)))
    }
    pub fn f(&mut self) -> f64 {
        self.0.f64()
    }
    pub fn u(&mut self, n: usize) -> usize {
        self.0.below(n)
    }
    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.u((hi - lo + 1) as usize) as i64
    }
    pub fn chance(&mut self, p: f64) -> bool {
        self.f() < p
    }
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.u(xs.len())]
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.f().max(1e-12);
        let u2 = self.f();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn shuffle<T>(&mut self, xs: &mut Vec<T>) {
        for i in (1..xs.len()).rev() {
            let j = self.u(i + 1);
            xs.swap(i, j);
        }
    }
}

pub fn parse_seeds(s: &str) -> Vec<u64> {
    if let Some((a, b)) = s.split_once('-') {
        let (a, b): (u64, u64) = (a.parse().unwrap(), b.parse().unwrap());
        (a..=b).collect()
    } else {
        s.split(',').map(|x| x.parse().unwrap()).collect()
    }
}
pub fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}
pub fn has(args: &[String], k: &str) -> bool {
    args.iter().any(|a| a == k)
}

// ---------------------------------------------------------------- civil dates
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}
pub fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}
pub fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 {
                29
            } else {
                28
            }
        }
    }
}
pub fn add_months(days: i64, n: i64) -> i64 {
    let (y, m, d) = civil_from_days(days);
    let t = y * 12 + (m - 1) + n;
    let (ny, nm) = (t.div_euclid(12), t.rem_euclid(12) + 1);
    days_from_civil(ny, nm, d.min(days_in_month(ny, nm)))
}
pub fn weekday(days: i64) -> usize {
    ((days + 4).rem_euclid(7)) as usize
}
pub const WEEKDAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
pub const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

/// the `n` bytes of `text` before `lo`, rounded to char boundaries
pub fn back(text: &str, lo: usize, n: usize) -> &str {
    let mut a = lo.saturating_sub(n);
    while a < lo && !text.is_char_boundary(a) {
        a += 1;
    }
    &text[a..lo]
}

/// the first `n` bytes of `s`, rounded down to a char boundary
pub fn fwd(s: &str, n: usize) -> &str {
    let mut e = n.min(s.len());
    while e > 0 && !s.is_char_boundary(e) {
        e -= 1;
    }
    &s[..e]
}
