//! judging extracted facts against ground truth, per-field-type P/R/F1 tallies, tau calibration.
use super::scan::*;
use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Out {
    Correct,
    Wrong,
    Spurious,
}
pub fn judge(doc: &Doc, e: &Ext) -> Out {
    match doc.fact(&e.fact.key) {
        Some(t) => {
            if t.matches(&e.fact) {
                Out::Correct
            } else {
                Out::Wrong
            }
        }
        None => Out::Spurious,
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct Prf {
    pub tp: f64,
    pub fp: f64,
    pub fn_: f64,
}
impl Prf {
    pub fn p(&self) -> f64 {
        if self.tp + self.fp > 0.0 { self.tp / (self.tp + self.fp) } else { f64::NAN }
    }
    pub fn r(&self) -> f64 {
        if self.tp + self.fn_ > 0.0 { self.tp / (self.tp + self.fn_) } else { f64::NAN }
    }
    pub fn f1(&self) -> f64 {
        let (p, r) = (self.p(), self.r());
        if p.is_nan() || r.is_nan() || p + r == 0.0 { if self.tp == 0.0 && self.fp == 0.0 && self.fn_ == 0.0 { f64::NAN } else { 0.0 } } else { 2.0 * p * r / (p + r) }
    }
    pub fn add(&mut self, o: &Prf) {
        self.tp += o.tp;
        self.fp += o.fp;
        self.fn_ += o.fn_;
    }
}
/// tally one doc, keeping only extracted facts accepted by `keep(i)`.
pub fn tally(doc: &Doc, exts: &[Ext], keep: &dyn Fn(usize) -> bool, by: &mut BTreeMap<FKind, Prf>) {
    for t in &doc.facts {
        let e = exts.iter().enumerate().find(|(i, e)| e.fact.key == t.key && keep(*i));
        let p = by.entry(t.kind).or_default();
        match e {
            None => p.fn_ += 1.0,
            Some((_, e)) => {
                if t.matches(&e.fact) {
                    p.tp += 1.0;
                } else {
                    p.fp += 1.0;
                    p.fn_ += 1.0;
                }
            }
        }
    }
    for (i, e) in exts.iter().enumerate() {
        if keep(i) && doc.fact(&e.fact.key).is_none() {
            by.entry(e.fact.kind).or_default().fp += 1.0;
        }
    }
}

/// pick tau (grid .05 .. .95) maximising micro-F1 over (docs, exts)
pub fn choose_tau(items: &[(&Doc, &Vec<Ext>)]) -> (f64, f64) {
    let mut best = (0.5, -1.0);
    let mut tau = 0.05;
    while tau < 0.96 {
        let mut p = Prf::default();
        for (d, ex) in items {
            let mut by = BTreeMap::new();
            tally(d, ex, &|i| ex[i].conf >= tau, &mut by);
            for v in by.values() {
                p.add(v);
            }
        }
        let f = p.f1();
        if f > best.1 + 1e-12 {
            best = (tau, f);
        }
        tau += 0.05;
    }
    best
}

/// expected calibration error of conf vs correctness
pub fn ece(pairs: &[(f64, bool)], nb: usize) -> f64 {
    if pairs.is_empty() {
        return f64::NAN;
    }
    let mut bins = vec![(0.0, 0.0, 0.0); nb];
    for (c, y) in pairs {
        let b = ((c * nb as f64) as usize).min(nb - 1);
        bins[b].0 += 1.0;
        bins[b].1 += c;
        bins[b].2 += if *y { 1.0 } else { 0.0 };
    }
    let n = pairs.len() as f64;
    bins.iter().filter(|b| b.0 > 0.0).map(|b| b.0 / n * (b.1 / b.0 - b.2 / b.0).abs()).sum()
}

use super::verify::*;
pub struct Run {
    pub all: Vec<Ext>,
    /// indices into `all` that pass the abstention threshold (and are the verifier's input)
    pub kept: Vec<usize>,
    /// verifier output for each kept fact (same order as `kept`)
    pub vfy: Vec<Vfy>,
}
pub fn run_extract(world: World, text: &str, cfg: &ExCfg, tau: f64, vc: &VCfg) -> Run {
    let all = extract(world, text, cfg);
    let kept: Vec<usize> = (0..all.len()).filter(|&i| all[i].conf >= tau).collect();
    let ke: Vec<Ext> = kept.iter().map(|&i| all[i].clone()).collect();
    let vfy = verify_doc(world, text, &ke, vc);
    Run { all, kept, vfy }
}
