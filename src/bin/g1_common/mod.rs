//! g1 shared pieces: corpus + offline-embedding fixtures (MiniLM, BGE-base, BGE+instruction, hash), BM25 / tf-idf lexical
//! index, score matrices, retrieval pipeline configs and per-query metrics.
//! Included from the g1_* bins with `#[path = "g1_common/mod.rs"] mod common;`.
//! Rust has NO ML runtime: vectors are produced offline by py/g1_embed.py (ONNX CPU) and read from results/g1_emb_*.f32.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};

use neural_mesh::knowledge::hash_embed;
use neural_mesh::router::Task;
use rayon::prelude::*;
use serde_json::Value;

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

pub fn words(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(|w| w.to_string()).collect()
}

// ---------------------------------------------------------------------------------------------
// corpus

pub struct Doc {
    pub concept: i32,
    pub topic: i32,
    pub text: String,
    pub sents: Vec<String>,
}
#[derive(Clone)]
pub struct Query {
    pub concept: i32,
    pub topic: i32,
    pub cls: String,
    pub text: String,
    pub spell: String,
    pub kw: String,
}
pub struct Corpus {
    pub tag: String,
    pub docs: Vec<Doc>,
    pub queries: Vec<Query>,
    pub nconc: usize,
    /// per concept: indices of its docs
    pub docs_of: Vec<Vec<usize>>,
    pub topic_of_concept: Vec<i32>,
}

pub fn load_corpus(tag: &str) -> Corpus {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(format!("results/g1_corpus_{tag}.json")).expect("results/g1_corpus_*.json (run py/g1_corpus.py)")).unwrap();
    let docs: Vec<Doc> = v["docs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| Doc {
            concept: d["concept"].as_i64().unwrap() as i32,
            topic: d["topic"].as_i64().unwrap() as i32,
            text: d["text"].as_str().unwrap().to_string(),
            sents: d["sents"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect(),
        })
        .collect();
    let queries: Vec<Query> = v["queries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|q| Query {
            concept: q["concept"].as_i64().unwrap() as i32,
            topic: q["topic"].as_i64().unwrap() as i32,
            cls: q["cls"].as_str().unwrap().to_string(),
            text: q["text"].as_str().unwrap().to_string(),
            spell: q["spell"].as_str().unwrap().to_string(),
            kw: q["kw"].as_str().unwrap().to_string(),
        })
        .collect();
    let nconc = v["n_concepts"].as_u64().unwrap() as usize;
    let mut docs_of = vec![vec![]; nconc];
    for (i, d) in docs.iter().enumerate() {
        if d.concept >= 0 {
            docs_of[d.concept as usize].push(i);
        }
    }
    let mut topic_of_concept = vec![0; nconc];
    for c in v["concepts"].as_array().unwrap() {
        topic_of_concept[c["id"].as_u64().unwrap() as usize] = c["topic"].as_i64().unwrap() as i32;
    }
    Corpus { tag: tag.to_string(), docs, queries, nconc, docs_of, topic_of_concept }
}

/// concept split: tuning concepts (seeds 1-8) vs held-out concepts (31-60 and the fresh-corpus confirm 61-90)
pub fn is_tune_concept(c: i32) -> bool {
    c >= 0 && c % 5 == 0
}

// ---------------------------------------------------------------------------------------------
// vectors

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QV {
    Raw = 0,
    Spell = 1,
    Kw = 2,
}
pub const QVS: [QV; 3] = [QV::Raw, QV::Spell, QV::Kw];

pub struct Model {
    pub name: String,
    pub dim: usize,
    pub doc: Vec<f32>,
    pub chunk: Vec<f32>,
    pub chunk_doc: Vec<u32>,
    /// per QV: nq x dim
    pub q: [Vec<f32>; 3],
    pub model_bytes: u64,
    pub single_ms: f64,
}
impl Model {
    pub fn qvec(&self, qv: QV, i: usize) -> &[f32] {
        &self.q[qv as usize][i * self.dim..(i + 1) * self.dim]
    }
    pub fn dvec(&self, i: usize) -> &[f32] {
        &self.doc[i * self.dim..(i + 1) * self.dim]
    }
    pub fn n_chunks(&self) -> usize {
        self.chunk_doc.len()
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
pub fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / n).collect()
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|_| panic!("{path} missing (run py/g1_embed.py)"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

pub fn load_model(c: &Corpus, name: &str) -> Model {
    let tag = &c.tag;
    let nq = c.queries.len();
    let meta: Value = serde_json::from_str(&std::fs::read_to_string(format!("results/g1_embmeta_{tag}.json")).unwrap_or_else(|_| "{\"models\":{}}".into())).unwrap();
    let (mb, ms) = match name {
        "mini" => (meta["models"]["mini"]["model_bytes"].as_u64().unwrap_or(0), meta["models"]["mini"]["single_ms_mean"].as_f64().unwrap_or(0.0)),
        "bge" | "bgei" => (meta["models"]["bge"]["model_bytes"].as_u64().unwrap_or(0), meta["models"]["bge"]["single_ms_mean"].as_f64().unwrap_or(0.0)),
        _ => (0, 0.0),
    };
    let texts: Vec<String> = serde_json::from_str(&std::fs::read_to_string(format!("results/g1_texts_{tag}.json")).expect("g1_texts")).unwrap();
    let tix: HashMap<&str, usize> = texts.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();
    let mk_chunks = |get: &dyn Fn(&str) -> Vec<f32>, dim: usize| -> (Vec<f32>, Vec<f32>, Vec<u32>) {
        let mut d = Vec::with_capacity(c.docs.len() * dim);
        let mut ch = vec![];
        let mut cd = vec![];
        for (i, doc) in c.docs.iter().enumerate() {
            d.extend(get(&doc.text));
            for s in &doc.sents {
                ch.extend(get(s));
                cd.push(i as u32);
            }
        }
        (d, ch, cd)
    };
    match name {
        "hash" => {
            let dim = 384;
            let f = |s: &str| hash_embed(&Task { id: 0, task_type: None, features: words(s) }, dim);
            let (d, ch, cd) = mk_chunks(&f, dim);
            let mut q: [Vec<f32>; 3] = [vec![], vec![], vec![]];
            for qq in &c.queries {
                q[0].extend(f(&qq.text));
                q[1].extend(f(&qq.spell));
                q[2].extend(f(&qq.kw));
            }
            Model { name: name.into(), dim, doc: d, chunk: ch, chunk_doc: cd, q, model_bytes: 0, single_ms: 0.0 }
        }
        "mini" | "bge" | "bgei" => {
            let dim = if name == "mini" { 384 } else { 768 };
            let base = if name == "mini" { "mini" } else { "bge" };
            let all = read_f32(&format!("results/g1_emb_{tag}_{base}.f32"));
            assert_eq!(all.len(), texts.len() * dim, "fixture size mismatch");
            let row = |s: &str| -> Vec<f32> {
                let i = *tix.get(s).unwrap_or_else(|| panic!("text missing from fixture: {s}"));
                all[i * dim..(i + 1) * dim].to_vec()
            };
            let (d, ch, cd) = mk_chunks(&row, dim);
            let mut q: [Vec<f32>; 3] = [vec![], vec![], vec![]];
            if name == "bgei" {
                let qt: Vec<String> = serde_json::from_str(&std::fs::read_to_string(format!("results/g1_qtexts_{tag}.json")).unwrap()).unwrap();
                let qix: HashMap<&str, usize> = qt.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();
                let ei = read_f32(&format!("results/g1_emb_{tag}_bgei.f32"));
                assert_eq!(ei.len(), qt.len() * dim);
                let rowi = |s: &str| -> Vec<f32> {
                    let i = qix[s];
                    ei[i * dim..(i + 1) * dim].to_vec()
                };
                for qq in &c.queries {
                    q[0].extend(rowi(&qq.text));
                    q[1].extend(rowi(&qq.spell));
                    q[2].extend(rowi(&qq.kw));
                }
            } else {
                for qq in &c.queries {
                    q[0].extend(row(&qq.text));
                    q[1].extend(row(&qq.spell));
                    q[2].extend(row(&qq.kw));
                }
            }
            assert_eq!(q[0].len(), nq * dim);
            Model { name: name.into(), dim, doc: d, chunk: ch, chunk_doc: cd, q, model_bytes: mb, single_ms: ms }
        }
        _ => panic!("unknown model {name}"),
    }
}

// ---------------------------------------------------------------------------------------------
// lexical: tokenisation, BM25, tf-idf

const STOP: &[&str] = &[
    "a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "is", "are", "be", "do", "does", "how", "what", "where", "i", "we", "you", "it", "my", "our", "your", "can", "could", "should", "would",
    "that", "this", "with", "from", "at", "as", "by", "about", "any", "me", "us",
];
pub fn lex_tokens(s: &str) -> Vec<String> {
    words(s)
        .into_iter()
        .filter(|w| !STOP.contains(&w.as_str()))
        .map(|w| if w.len() > 3 && w.ends_with('s') && !w.ends_with("ss") { w[..w.len() - 1].to_string() } else { w })
        .collect()
}

pub struct Lex {
    pub vocab: HashMap<String, u32>,
    pub idf: Vec<f32>,
    /// postings[term] = (doc, tf)
    pub postings: Vec<Vec<(u32, f32)>>,
    pub doc_len: Vec<f32>,
    pub avg_len: f32,
    pub doc_terms: Vec<HashSet<u32>>,
    pub n_docs: usize,
}
impl Lex {
    pub fn build(docs: &[Doc]) -> Lex {
        let mut vocab: HashMap<String, u32> = HashMap::new();
        let mut tfs: Vec<HashMap<u32, f32>> = vec![];
        for d in docs {
            let mut m: HashMap<u32, f32> = HashMap::new();
            for t in lex_tokens(&d.text) {
                let n = vocab.len() as u32;
                let id = *vocab.entry(t).or_insert(n);
                *m.entry(id).or_insert(0.0) += 1.0;
            }
            tfs.push(m);
        }
        let nt = vocab.len();
        let mut postings = vec![vec![]; nt];
        let mut doc_len = vec![0.0f32; docs.len()];
        let mut doc_terms = vec![];
        for (di, m) in tfs.iter().enumerate() {
            let mut set = HashSet::new();
            for (&t, &tf) in m {
                postings[t as usize].push((di as u32, tf));
                doc_len[di] += tf;
                set.insert(t);
            }
            doc_terms.push(set);
        }
        let n = docs.len() as f32;
        let idf: Vec<f32> = postings.iter().map(|p| ((n - p.len() as f32 + 0.5) / (p.len() as f32 + 0.5) + 1.0).ln()).collect();
        let avg_len = doc_len.iter().sum::<f32>() / n;
        Lex { vocab, idf, postings, doc_len, avg_len, doc_terms, n_docs: docs.len() }
    }
    pub fn q_terms(&self, q: &str) -> Vec<u32> {
        let mut seen = HashSet::new();
        lex_tokens(q).iter().filter_map(|t| self.vocab.get(t).copied()).filter(|t| seen.insert(*t)).collect()
    }
    pub fn bm25(&self, q: &str) -> Vec<f32> {
        let (k1, b) = (1.2f32, 0.75f32);
        let mut s = vec![0.0f32; self.n_docs];
        for t in self.q_terms(q) {
            for &(d, tf) in &self.postings[t as usize] {
                let dl = self.doc_len[d as usize];
                s[d as usize] += self.idf[t as usize] * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / self.avg_len));
            }
        }
        s
    }
    /// idf-weighted coverage of the query's known terms by the doc (the "token-overlap cross score")
    pub fn overlap(&self, q_terms: &[u32], d: usize) -> f32 {
        let tot: f32 = q_terms.iter().map(|&t| self.idf[t as usize]).sum();
        if tot <= 0.0 {
            return 0.0;
        }
        q_terms.iter().filter(|t| self.doc_terms[d].contains(t)).map(|&t| self.idf[t as usize]).sum::<f32>() / tot
    }
    /// tf-idf sparse unit vector for task-to-task cosine
    pub fn tfidf(&self, q: &str) -> Vec<(u32, f32)> {
        let mut m: HashMap<u32, f32> = HashMap::new();
        for t in lex_tokens(q) {
            if let Some(&id) = self.vocab.get(&t) {
                *m.entry(id).or_insert(0.0) += 1.0;
            }
        }
        let mut v: Vec<(u32, f32)> = m.into_iter().map(|(t, tf)| (t, (1.0 + tf.ln()) * self.idf[t as usize])).collect();
        v.sort_by_key(|x| x.0); // fixed order BEFORE the norm: HashMap iteration order would change the f32 sum by an ulp
        let n = v.iter().map(|x| x.1 * x.1).sum::<f32>().sqrt().max(1e-12);
        v.iter_mut().for_each(|x| x.1 /= n);
        v
    }
    pub fn index_bytes(&self) -> usize {
        self.postings.iter().map(|p| p.len() * 8).sum::<usize>() + self.vocab.len() * 12
    }
}
pub fn sparse_dot(a: &[(u32, f32)], b: &[(u32, f32)]) -> f32 {
    let (mut i, mut j, mut s) = (0, 0, 0.0);
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            std::cmp::Ordering::Equal => {
                s += a[i].1 * b[j].1;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    s
}

// ---------------------------------------------------------------------------------------------
// score matrices (query x doc), precomputed per (model, query-variant)

pub struct Mats {
    pub nq: usize,
    pub nd: usize,
    /// cosine(query, whole doc)
    pub doc: Vec<f32>,
    /// max over the doc's sentence chunks
    pub chunk: Vec<f32>,
}
pub fn dense_mats(m: &Model, nq: usize, nd: usize, qv: QV) -> Mats {
    let rows: Vec<(Vec<f32>, Vec<f32>)> = (0..nq)
        .into_par_iter()
        .map(|i| {
            let q = m.qvec(qv, i);
            let d: Vec<f32> = (0..nd).map(|j| dot(q, m.dvec(j))).collect();
            let mut ch = vec![-1.0f32; nd];
            for (k, &dj) in m.chunk_doc.iter().enumerate() {
                let s = dot(q, &m.chunk[k * m.dim..(k + 1) * m.dim]);
                if s > ch[dj as usize] {
                    ch[dj as usize] = s;
                }
            }
            (d, ch)
        })
        .collect();
    let mut doc = Vec::with_capacity(nq * nd);
    let mut chunk = Vec::with_capacity(nq * nd);
    for (d, c) in rows {
        doc.extend(d);
        chunk.extend(c);
    }
    Mats { nq, nd, doc, chunk }
}
/// Rocchio pseudo-relevance feedback on dense whole-doc scores: q' = unit(q + beta * mean(top3 doc vectors))
pub fn prf_doc(m: &Model, base: &Mats, qv: QV, beta: f32) -> Vec<f32> {
    let (nq, nd) = (base.nq, base.nd);
    let rows: Vec<Vec<f32>> = (0..nq)
        .into_par_iter()
        .map(|i| {
            let s = &base.doc[i * nd..(i + 1) * nd];
            let mut idx: Vec<usize> = (0..nd).collect();
            idx.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
            let mut q2 = m.qvec(qv, i).to_vec();
            for &j in idx.iter().take(3) {
                for (x, y) in q2.iter_mut().zip(m.dvec(j)) {
                    *x += beta / 3.0 * y;
                }
            }
            let q2 = unit(&q2);
            (0..nd).map(|j| dot(&q2, m.dvec(j))).collect()
        })
        .collect();
    rows.into_iter().flatten().collect()
}
pub fn bm25_mat(lex: &Lex, qs: &[Query], qv: QV) -> Vec<f32> {
    let rows: Vec<Vec<f32>> = qs.par_iter().map(|q| lex.bm25(match qv { QV::Raw => &q.text, QV::Spell => &q.spell, QV::Kw => &q.kw })).collect();
    rows.into_iter().flatten().collect()
}

pub fn zscore(v: &[f32]) -> Vec<f32> {
    let n = v.len() as f32;
    let m = v.iter().sum::<f32>() / n;
    let sd = (v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n).sqrt().max(1e-9);
    v.iter().map(|x| (x - m) / sd).collect()
}

// ---------------------------------------------------------------------------------------------
// metrics

#[derive(Clone, Copy, Debug, Default)]
pub struct QM {
    pub hit1: f64,
    pub mrr: f64,
    pub p3: f64,
    pub r3: f64,
    pub f13: f64,
    pub r5: f64,
    /// top-1 doc is a SIBLING concept doc (same topic, other detail)
    pub sib1: f64,
    /// top-1 doc is the right topic (any detail)
    pub topic1: f64,
}
pub const QM_NAMES: [&str; 8] = ["hit1", "mrr", "p3", "r3", "f13", "r5", "sib1", "topic1"];
impl QM {
    pub fn arr(&self) -> [f64; 8] {
        [self.hit1, self.mrr, self.p3, self.r3, self.f13, self.r5, self.sib1, self.topic1]
    }
}

/// metrics for one query given fused doc scores. relevant = docs of the query's concept.
pub fn eval_scores(c: &Corpus, q: &Query, s: &[f32]) -> QM {
    let rel = &c.docs_of[q.concept as usize];
    let mut ranks: Vec<usize> = rel.iter().map(|&r| 1 + s.iter().enumerate().filter(|&(j, &x)| j != r && (x > s[r] || (x == s[r] && j < r))).count()).collect();
    ranks.sort();
    let top = (0..s.len()).fold(0, |b, j| if s[j] > s[b] { j } else { b });
    let inrel = |k: usize| ranks.iter().filter(|&&r| r <= k).count() as f64;
    let nrel = rel.len() as f64;
    let (p3, r3) = (inrel(3) / 3.0, inrel(3) / nrel);
    QM {
        hit1: (ranks[0] == 1) as u8 as f64,
        mrr: 1.0 / ranks[0] as f64,
        p3,
        r3,
        f13: if p3 + r3 > 0.0 { 2.0 * p3 * r3 / (p3 + r3) } else { 0.0 },
        r5: inrel(5) / nrel,
        sib1: (c.docs[top].topic == q.topic && c.docs[top].concept != q.concept) as u8 as f64,
        topic1: (c.docs[top].topic == q.topic) as u8 as f64,
    }
}

// ---------------------------------------------------------------------------------------------
// pipeline config

#[derive(Clone, Debug)]
pub struct Cfg {
    pub name: String,
    pub model: Option<String>,
    pub qv: QV,
    /// dense = lam*doc + (1-lam)*chunkmax
    pub lam: f32,
    pub prf: f32,
    /// lexical channel (BM25 on the same query variant); alpha = weight of dense in z-fusion; 1.0 = dense only, 0.0 = lexical only
    pub lexical: bool,
    pub alpha: f32,
    /// token-overlap rerank of the top-M (gamma weight, z-scored within candidates)
    pub gamma: f32,
    /// cascade: rerank top-M with the dense whole-doc score of ANOTHER model (name, weight)
    pub cascade: Option<(String, f32)>,
    pub m: usize,
}
impl Cfg {
    pub fn base(name: &str, model: &str) -> Cfg {
        Cfg { name: name.into(), model: Some(model.into()), qv: QV::Raw, lam: 1.0, prf: 0.0, lexical: false, alpha: 1.0, gamma: 0.0, cascade: None, m: 20 }
    }
    pub fn bm25(name: &str) -> Cfg {
        Cfg { name: name.into(), model: None, qv: QV::Raw, lam: 1.0, prf: 0.0, lexical: true, alpha: 0.0, gamma: 0.0, cascade: None, m: 20 }
    }
    pub fn desc(&self) -> String {
        format!(
            "model={} qv={:?} lam={} prf={} lex={} alpha={} gamma={} cascade={:?} M={}",
            self.model.clone().unwrap_or("-".into()),
            self.qv,
            self.lam,
            self.prf,
            self.lexical,
            self.alpha,
            self.gamma,
            self.cascade,
            self.m
        )
    }
}

pub struct Ctx {
    pub c: Corpus,
    pub lex: Lex,
    pub models: HashMap<String, Model>,
    /// (model, qv) -> dense mats
    pub dm: HashMap<(String, u8), Mats>,
    /// (model, qv, beta*100) -> prf doc scores
    pub prf: HashMap<(String, u8, i32), Vec<f32>>,
    pub bm: [Vec<f32>; 3],
    pub q_terms: [Vec<Vec<u32>>; 3],
}

pub const PRF_GRID: [f32; 3] = [0.3, 0.6, 1.0];

impl Ctx {
    pub fn new(tag: &str, model_names: &[&str], need_prf: bool) -> Ctx {
        let c = load_corpus(tag);
        let lex = Lex::build(&c.docs);
        let mut models = HashMap::new();
        let mut dm = HashMap::new();
        let mut prf = HashMap::new();
        let (nq, nd) = (c.queries.len(), c.docs.len());
        for &n in model_names {
            let m = load_model(&c, n);
            for qv in QVS {
                let mats = dense_mats(&m, nq, nd, qv);
                if need_prf && qv != QV::Kw {
                    for &b in &PRF_GRID {
                        prf.insert((n.to_string(), qv as u8, (b * 100.0) as i32), prf_doc(&m, &mats, qv, b));
                    }
                }
                dm.insert((n.to_string(), qv as u8), mats);
            }
            models.insert(n.to_string(), m);
        }
        let bm = [bm25_mat(&lex, &c.queries, QV::Raw), bm25_mat(&lex, &c.queries, QV::Spell), bm25_mat(&lex, &c.queries, QV::Kw)];
        let qt = |f: &dyn Fn(&Query) -> &str| -> Vec<Vec<u32>> { c.queries.iter().map(|q| lex.q_terms(f(q))).collect() };
        let q_terms = [qt(&|q| &q.text), qt(&|q| &q.spell), qt(&|q| &q.kw)];
        Ctx { c, lex, models, dm, prf, bm, q_terms }
    }

    /// final fused doc scores for query qi under cfg
    pub fn score(&self, cfg: &Cfg, qi: usize) -> Vec<f32> {
        let nd = self.c.docs.len();
        let qv = cfg.qv as u8;
        let dense: Option<Vec<f32>> = cfg.model.as_ref().map(|mn| {
            let mats = &self.dm[&(mn.clone(), qv)];
            let doc: &[f32] = if cfg.prf > 0.0 { &self.prf[&(mn.clone(), qv, (cfg.prf * 100.0) as i32)][qi * nd..(qi + 1) * nd] } else { &mats.doc[qi * nd..(qi + 1) * nd] };
            if cfg.lam >= 1.0 {
                doc.to_vec()
            } else {
                let ch = &mats.chunk[qi * nd..(qi + 1) * nd];
                doc.iter().zip(ch).map(|(d, c)| cfg.lam * d + (1.0 - cfg.lam) * c).collect()
            }
        });
        let lexs: Option<Vec<f32>> = if cfg.lexical { Some(self.bm[qv as usize][qi * nd..(qi + 1) * nd].to_vec()) } else { None };
        let mut f: Vec<f32> = match (&dense, &lexs) {
            (Some(d), Some(l)) => {
                let (zd, zl) = (zscore(d), zscore(l));
                zd.iter().zip(&zl).map(|(a, b)| cfg.alpha * a + (1.0 - cfg.alpha) * b).collect()
            }
            (Some(d), None) => zscore(d),
            (None, Some(l)) => zscore(l),
            _ => panic!("empty cfg"),
        };
        if cfg.gamma > 0.0 || cfg.cascade.is_some() {
            let mut idx: Vec<usize> = (0..nd).collect();
            idx.sort_by(|&a, &b| f[b].partial_cmp(&f[a]).unwrap());
            idx.truncate(cfg.m);
            let zc = |v: Vec<f32>| -> Vec<f32> { zscore(&v) };
            let base = zc(idx.iter().map(|&j| f[j]).collect());
            let mut add = vec![0.0f32; idx.len()];
            if cfg.gamma > 0.0 {
                let ov = zc(idx.iter().map(|&j| self.lex.overlap(&self.q_terms[qv as usize][qi], j)).collect());
                for (a, o) in add.iter_mut().zip(&ov) {
                    *a += cfg.gamma * o;
                }
            }
            if let Some((mn, w)) = &cfg.cascade {
                let mats = &self.dm[&(mn.clone(), qv)];
                let cs = zc(idx.iter().map(|&j| mats.doc[qi * nd + j]).collect());
                for (a, o) in add.iter_mut().zip(&cs) {
                    *a += w * o;
                }
            }
            let mut out = vec![-1e9f32; nd];
            for (k, &j) in idx.iter().enumerate() {
                out[j] = base[k] + add[k] + 100.0; // keep candidates above the rest
            }
            for j in 0..nd {
                if out[j] < -1e8 {
                    out[j] = f[j] - 1000.0;
                }
            }
            f = out;
        }
        f
    }

    pub fn eval_all(&self, cfg: &Cfg, qis: &[usize]) -> Vec<(usize, QM)> {
        qis.par_iter().map(|&qi| (qi, eval_scores(&self.c, &self.c.queries[qi], &self.score(cfg, qi)))).collect()
    }
}
