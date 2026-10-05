//! h10 shared pieces: loaders for the REAL-text fixtures (py/h10_corpus.py, py/h10_embed.py), BM25, z-fusion, paired stats.
//! Included with `#[path = "h10_common/mod.rs"] mod common;`.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};

use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue};
use serde_json::Value;

pub const DIM: usize = 384;

pub fn seeds(lo: u64, hi: u64) -> Vec<u64> {
    (lo..=hi).collect()
}

pub struct QInfo {
    pub fam: String,
    pub rel: Vec<usize>,
}

pub struct Row {
    pub q: usize,
    pub fam: String,
    /// per variant (clean, typo1, typo2): text ids [raw, spell1, spell2]
    pub t: Vec<[usize; 3]>,
}

pub struct Inst {
    pub pool: String,
    pub rows: Vec<Row>,
}

pub struct World {
    pub tag: String,
    pub texts: Vec<String>,
    pub emb: Vec<f32>,
    pub ndoc: usize,
    pub queries: Vec<QInfo>,
    pub inst: BTreeMap<String, Inst>,
}

pub fn load_world(tag: &str) -> World {
    let rd = |n: String| std::fs::read(&n).unwrap_or_else(|e| panic!("{n}: {e}"));
    let texts: Vec<String> = serde_json::from_slice(&rd(format!("results/h10_texts_{tag}.json"))).unwrap();
    let raw = rd(format!("results/h10_emb_{tag}.f32"));
    let emb: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    assert_eq!(emb.len(), texts.len() * DIM);
    let j: Value = serde_json::from_slice(&rd(format!("results/h10_inst_{tag}.json"))).unwrap();
    let ndoc = j["ndoc"].as_u64().unwrap() as usize;
    let queries = j["queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| QInfo { fam: q["fam"].as_str().unwrap().to_string(), rel: q["rel"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect() })
        .collect();
    let mut inst = BTreeMap::new();
    for (k, v) in j["inst"].as_object().unwrap() {
        let rows = v["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| Row {
                q: r["q"].as_u64().unwrap() as usize,
                fam: r["fam"].as_str().unwrap().to_string(),
                t: r["t"].as_array().unwrap().iter().map(|a| {
                    let a = a.as_array().unwrap();
                    [a[0].as_u64().unwrap() as usize, a[1].as_u64().unwrap() as usize, a[2].as_u64().unwrap() as usize]
                }).collect(),
            })
            .collect();
        inst.insert(k.clone(), Inst { pool: v["pool"].as_str().unwrap().to_string(), rows });
    }
    World { tag: tag.to_string(), texts, emb, ndoc, queries, inst }
}

impl World {
    pub fn vec(&self, i: usize) -> &[f32] {
        &self.emb[i * DIM..(i + 1) * DIM]
    }
    /// dense cosine scores of text `t` against all docs (vectors are L2-normalised)
    pub fn dense(&self, t: usize, out: &mut Vec<f32>) {
        out.clear();
        let q = self.vec(t);
        for d in 0..self.ndoc {
            let v = self.vec(d);
            let mut s = 0.0f32;
            for k in 0..DIM {
                s += q[k] * v[k];
            }
            out.push(s);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// BM25 over the document store (lowercase [a-z0-9]+ tokens, small stopword list)

const STOP: &[&str] = &["the", "a", "an", "of", "to", "and", "in", "is", "it", "for", "on", "that", "this", "with", "as", "are", "be", "by", "or", "at", "from", "was", "can", "if", "you", "we", "not", "will", "has", "have", "its"];

pub fn tokens(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            cur.push(c.to_ascii_lowercase());
        } else if !cur.is_empty() {
            if !STOP.contains(&cur.as_str()) {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if !cur.is_empty() && !STOP.contains(&cur.as_str()) {
        out.push(cur);
    }
    out
}

pub struct Bm25 {
    post: HashMap<String, Vec<(u32, f32)>>, // term -> (doc, tf)
    idf: HashMap<String, f32>,
    dl: Vec<f32>,
    avg: f32,
    pub n: usize,
}

impl Bm25 {
    pub fn new(docs: &[String]) -> Bm25 {
        let mut post: HashMap<String, Vec<(u32, f32)>> = HashMap::new();
        let mut dl = Vec::with_capacity(docs.len());
        for (i, d) in docs.iter().enumerate() {
            let tk = tokens(d);
            dl.push(tk.len() as f32);
            let mut c: HashMap<&str, f32> = HashMap::new();
            for t in &tk {
                *c.entry(t.as_str()).or_insert(0.0) += 1.0;
            }
            for (t, f) in c {
                post.entry(t.to_string()).or_default().push((i as u32, f));
            }
        }
        let n = docs.len();
        let avg = dl.iter().sum::<f32>() / n as f32;
        let idf = post.iter().map(|(t, p)| (t.clone(), (((n as f32 - p.len() as f32 + 0.5) / (p.len() as f32 + 0.5)) + 1.0).ln())).collect();
        Bm25 { post, idf, dl, avg, n }
    }
    pub fn score(&self, q: &str, out: &mut Vec<f32>) {
        out.clear();
        out.resize(self.n, 0.0);
        let (k1, b) = (1.2f32, 0.75f32);
        let mut seen: Vec<String> = Vec::new();
        for t in tokens(q) {
            if seen.contains(&t) {
                continue;
            }
            if let Some(p) = self.post.get(&t) {
                let idf = self.idf[&t];
                for &(d, tf) in p {
                    let norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * self.dl[d as usize] / self.avg));
                    out[d as usize] += idf * norm;
                }
            }
            seen.push(t);
        }
    }
}

pub fn zscore(v: &[f32]) -> Vec<f32> {
    let n = v.len() as f32;
    let m = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n;
    let sd = var.sqrt().max(1e-9);
    v.iter().map(|x| (x - m) / sd).collect()
}

/// 1-based rank of the best relevant doc; non-relevant docs that tie with it are counted ahead of it (pessimistic)
pub fn best_rank(scores: &[f32], rel: &[usize]) -> usize {
    let best = rel.iter().map(|&r| scores[r]).fold(f32::NEG_INFINITY, f32::max);
    1 + scores.iter().enumerate().filter(|&(j, &s)| s >= best && !rel.contains(&j)).count()
}

// ---------------------------------------------------------------------------------------------
// statistics

#[derive(Clone, Debug)]
pub struct Cmp {
    pub mean: f64,
    pub lo: f64,
    pub hi: f64,
    pub p: f64,
    pub wins: usize,
    pub losses: usize,
    pub n: usize,
}

pub fn compare(a: &[f64], b: &[f64]) -> Cmp {
    let pd = paired_diff(a, b);
    Cmp { mean: pd.diff.mean, lo: pd.diff.lo, hi: pd.diff.hi, p: paired_t_pvalue(a, b), wins: pd.wins, losses: pd.losses, n: a.len() }
}

pub fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

pub fn holm(ps: &[f64]) -> Vec<f64> {
    holm_adjust(ps)
}

pub struct Xs(pub u64);
impl Xs {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// percentile bootstrap CI of the mean over `vals` (resampling units = distinct queries)
pub fn boot_ci(vals: &[f64], reps: usize, seed: u64) -> (f64, f64, f64) {
    let n = vals.len();
    let m = mean(vals);
    let mut r = Xs(seed | 1);
    let mut ms: Vec<f64> = (0..reps).map(|_| (0..n).map(|_| vals[r.below(n)]).sum::<f64>() / n as f64).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (m, ms[(reps as f64 * 0.025) as usize], ms[(reps as f64 * 0.975) as usize])
}

include!("ext.rs");
