//! Hybrid dense + BM25 retrieval (z-score fusion) and a document-vocabulary spell corrector. std-only, OPT-IN, nothing here is wired
//! into any default path.
//!
//! Productionised from the verified R6 study (results/h13_board_entry.md; reference code `src/bin/h13_common/{mod,ext}.rs` and
//! `py/h10_embed.py`'s `Speller`):
//!   score(d) = alpha * z(dense cosine)(d) + (1 - alpha) * z(BM25)(d),  z = standardisation over ALL documents of the store, alpha = 0.7.
//! BM25: k1 1.2, b 0.75, lowercase `[a-z0-9]+` tokens, 31-word stoplist, query tokens de-duplicated.
//! Spell correction is a SEPARATE switch (`SpellCorrector`): only words that are not in the known vocabulary and have >= 4 letters are
//! replaced, by the highest-frequency vocabulary word within optimal-string-alignment distance 1.
//!
//! Evidence label: SYNTHETIC-until-validated-on-real-traffic; the +0.03..+0.05 MRR was measured on two real documentation corpora
//! (doc level). End-to-end routing ties (retrieval gain does not move routing). On templated / paraphrase corpora the gain is ~0 and on
//! queries with almost no lexical overlap it is negative (-0.03 MRR in the lowest overlap tertile); keep dense(+spell) as the alternative.
//!
//! Float semantics are kept identical to the reference (f32 arithmetic, same operation order) so results are bit-comparable.
//! Documented divergences from the reference code:
//!  * `zscore` of a numerically constant vector returns zeros (the reference divides by max(sd, 1e-9), which turns rounding noise into +-1);
//!  * an empty corpus / empty query / dimension mismatch are handled (empty result or `HybridError`) instead of producing NaN or panicking;
//!  * `top_n` orders NaN scores last instead of panicking.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};

const STOP: &[&str] = &[
    "the", "a", "an", "of", "to", "and", "in", "is", "it", "for", "on", "that", "this", "with", "as", "are", "be", "by", "or", "at", "from", "was", "can", "if",
    "you", "we", "not", "will", "has", "have", "its",
];

/// lowercase `[a-z0-9]+` tokens without the stoplist words
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

/// standardise to mean 0, population sd 1. A constant (or empty) vector maps to zeros.
pub fn zscore(v: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    zscore_into(v, &mut out);
    out
}

fn zscore_into(v: &[f32], out: &mut Vec<f32>) {
    out.clear();
    if v.is_empty() {
        return;
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in v {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    if !(hi > lo) {
        out.resize(v.len(), 0.0);
        return;
    }
    let n = v.len() as f32;
    let m = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n;
    let sd = var.sqrt().max(1e-9);
    out.extend(v.iter().map(|x| (x - m) / sd));
}

/// indices of the top-n scores, best first, ties broken by the lower index; NaN scores sort last
pub fn top_n(scores: &[f32], n: usize) -> Vec<usize> {
    use std::cmp::Ordering;
    let mut v: Vec<(f32, usize)> = scores.iter().copied().enumerate().map(|(i, s)| (s, i)).collect();
    let cmp = |a: &(f32, usize), b: &(f32, usize)| -> Ordering {
        let o = match (a.0.is_nan(), b.0.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal),
        };
        o.then(a.1.cmp(&b.1))
    };
    let n = n.min(v.len());
    if n == 0 {
        return Vec::new();
    }
    if v.len() > n {
        v.select_nth_unstable_by(n, cmp);
        v.truncate(n);
    }
    v.sort_by(cmp);
    v.into_iter().map(|x| x.1).collect()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum HybridError {
    /// `dense.len() != docs.len() * dim`
    DenseShape { expected: usize, got: usize },
    /// query vector length != index dimension
    QueryDim { expected: usize, got: usize },
}
impl std::fmt::Display for HybridError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HybridError::DenseShape { expected, got } => write!(f, "dense matrix has {got} values, expected {expected}"),
            HybridError::QueryDim { expected, got } => write!(f, "query vector has {got} dims, expected {expected}"),
        }
    }
}
impl std::error::Error for HybridError {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Term {
    idf: f32,
    /// (doc, tf), ascending doc order
    post: Vec<(u32, f32)>,
}

/// BM25 index over a document store
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bm25 {
    k1: f32,
    b: f32,
    terms: BTreeMap<String, Term>,
    dl: Vec<f32>,
    avg: f32,
    n: usize,
}

impl Bm25 {
    pub fn new(docs: &[String]) -> Bm25 {
        Bm25::with_params(docs, 1.2, 0.75)
    }
    pub fn with_params(docs: &[String], k1: f32, b: f32) -> Bm25 {
        let mut post: BTreeMap<String, Vec<(u32, f32)>> = BTreeMap::new();
        let mut dl = Vec::with_capacity(docs.len());
        for (i, d) in docs.iter().enumerate() {
            let tk = tokens(d);
            dl.push(tk.len() as f32);
            let mut c: BTreeMap<&str, f32> = BTreeMap::new();
            for t in &tk {
                *c.entry(t.as_str()).or_insert(0.0) += 1.0;
            }
            for (t, f) in c {
                post.entry(t.to_string()).or_default().push((i as u32, f));
            }
        }
        let n = docs.len();
        let avg = if n > 0 { dl.iter().sum::<f32>() / n as f32 } else { 1.0 };
        let avg = if avg > 0.0 { avg } else { 1.0 };
        let terms = post
            .into_iter()
            .map(|(t, p)| {
                let idf = (((n as f32 - p.len() as f32 + 0.5) / (p.len() as f32 + 0.5)) + 1.0).ln();
                (t, Term { idf, post: p })
            })
            .collect();
        Bm25 { k1, b, terms, dl, avg, n }
    }
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn vocab_size(&self) -> usize {
        self.terms.len()
    }
    /// BM25 score of every document for query text `q` (zeros when nothing matches)
    pub fn score(&self, q: &str, out: &mut Vec<f32>) {
        out.clear();
        out.resize(self.n, 0.0);
        let (k1, b) = (self.k1, self.b);
        let mut seen: Vec<String> = Vec::new();
        for t in tokens(q) {
            if seen.contains(&t) {
                continue;
            }
            if let Some(term) = self.terms.get(&t) {
                for &(d, tf) in &term.post {
                    let norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * self.dl[d as usize] / self.avg));
                    out[d as usize] += term.idf * norm;
                }
            }
            seen.push(t);
        }
    }
    /// B1/R2 (opt-in): lexical coverage of the best document in [0,1]: max BM25 score divided by the maximum a document could score
    /// (sum of idf of the distinct in-vocabulary query tokens times (k1+1)). 0 when no query token is in the vocabulary.
    pub fn lexical_coverage(&self, q: &str, scores: &[f32]) -> f32 {
        let mut seen: Vec<String> = Vec::new();
        let mut den = 0.0f32;
        for t in tokens(q) {
            if seen.contains(&t) {
                continue;
            }
            if let Some(term) = self.terms.get(&t) {
                den += term.idf;
            }
            seen.push(t);
        }
        den *= self.k1 + 1.0;
        if den <= 0.0 {
            return 0.0;
        }
        let mx = scores.iter().copied().fold(0.0f32, f32::max);
        (mx / den).clamp(0.0, 1.0)
    }
    /// approximate payload size: postings (u32 doc + f32 tf) + term strings + idf (f32) + doc lengths (same formula as the h13 report)
    pub fn bytes(&self) -> usize {
        self.terms.iter().map(|(t, e)| t.len() + e.post.len() * 8 + 4).sum::<usize>() + self.dl.len() * 4
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HybridConfig {
    /// weight of the dense z-score; 1 = dense only, 0 = BM25 only. Default 0.7 (the h13 pick, frozen).
    pub alpha: f64,
}
impl Default for HybridConfig {
    fn default() -> Self {
        HybridConfig { alpha: 0.7 }
    }
}
impl HybridConfig {
    fn alpha_f32(&self) -> f32 {
        let a = if self.alpha.is_nan() { 0.7 } else { self.alpha.clamp(0.0, 1.0) };
        a as f32
    }
}


/// B1/R2 (opt-in, NOT used by any default path): query-adaptive fusion weight.
/// alpha_q = hi - (hi - lo) * coverage^pow, where coverage = [`Bm25::lexical_coverage`]: strong lexical match -> trust BM25 more (alpha -> lo),
/// weak/paraphrased query -> trust the dense score (alpha -> hi). Verified in py/b1 (RD-LOG R2): wins on paraphrase+typo domain queries,
/// no Holm loss on the real-document fixtures.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdaptiveAlpha {
    pub lo: f64,
    pub hi: f64,
    pub pow: f64,
}
impl Default for AdaptiveAlpha {
    fn default() -> Self {
        AdaptiveAlpha { lo: 0.3, hi: 1.0, pow: 2.0 }
    }
}

/// document store: dense vectors (assumed L2-normalised, cosine = dot) + BM25 index over the document texts
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HybridIndex {
    cfg: HybridConfig,
    dim: usize,
    ndoc: usize,
    dense: Vec<f32>,
    bm: Bm25,
}

impl HybridIndex {
    pub fn new(docs: &[String], dense: Vec<f32>, dim: usize, cfg: HybridConfig) -> Result<HybridIndex, HybridError> {
        let expected = docs.len() * dim;
        if dense.len() != expected {
            return Err(HybridError::DenseShape { expected, got: dense.len() });
        }
        Ok(HybridIndex { cfg, dim, ndoc: docs.len(), dense, bm: Bm25::new(docs) })
    }
    pub fn len(&self) -> usize {
        self.ndoc
    }
    pub fn is_empty(&self) -> bool {
        self.ndoc == 0
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn config(&self) -> HybridConfig {
        self.cfg
    }
    pub fn set_alpha(&mut self, alpha: f64) {
        self.cfg.alpha = alpha;
    }
    pub fn bm25(&self) -> &Bm25 {
        &self.bm
    }
    pub fn bytes_dense(&self) -> usize {
        self.dense.len() * 4
    }
    /// extra bytes the lexical side adds on top of the dense matrix
    pub fn bytes_bm25(&self) -> usize {
        self.bm.bytes()
    }
    /// dense cosine of every document to `q`
    pub fn dense_scores(&self, q: &[f32], out: &mut Vec<f32>) -> Result<(), HybridError> {
        if q.len() != self.dim {
            return Err(HybridError::QueryDim { expected: self.dim, got: q.len() });
        }
        out.clear();
        for d in 0..self.ndoc {
            let v = &self.dense[d * self.dim..(d + 1) * self.dim];
            let mut s = 0.0f32;
            for k in 0..self.dim {
                s += q[k] * v[k];
            }
            out.push(s);
        }
        Ok(())
    }
    /// fused z-score of every document; `qtext` is the (optionally spell-corrected) query text, `qvec` its embedding
    pub fn fused_scores(&self, qtext: &str, qvec: &[f32], out: &mut Vec<f32>) -> Result<(), HybridError> {
        let mut dense = Vec::new();
        self.dense_scores(qvec, &mut dense)?;
        let mut bm = Vec::new();
        self.bm.score(qtext, &mut bm);
        let (mut zd, mut zb) = (Vec::new(), Vec::new());
        zscore_into(&dense, &mut zd);
        zscore_into(&bm, &mut zb);
        let a = self.cfg.alpha_f32();
        out.clear();
        out.extend(zd.iter().zip(&zb).map(|(d, b)| a * d + (1.0 - a) * b));
        Ok(())
    }
    /// B1/R2 (opt-in): fused z-scores with the per-query adaptive alpha; returns the alpha actually used.
    pub fn fused_scores_adaptive(&self, qtext: &str, qvec: &[f32], ad: &AdaptiveAlpha, out: &mut Vec<f32>) -> Result<f32, HybridError> {
        let mut dense = Vec::new();
        self.dense_scores(qvec, &mut dense)?;
        let mut bm = Vec::new();
        self.bm.score(qtext, &mut bm);
        let cov = self.bm.lexical_coverage(qtext, &bm);
        let a = (ad.hi - (ad.hi - ad.lo) * (cov as f64).powf(ad.pow)).clamp(0.0, 1.0) as f32;
        let (mut zd, mut zb) = (Vec::new(), Vec::new());
        zscore_into(&dense, &mut zd);
        zscore_into(&bm, &mut zb);
        out.clear();
        out.extend(zd.iter().zip(&zb).map(|(d, b)| a * d + (1.0 - a) * b));
        Ok(a)
    }
    /// B1/R2 (opt-in): top-k with the adaptive alpha
    pub fn search_adaptive(&self, qtext: &str, qvec: &[f32], k: usize, ad: &AdaptiveAlpha) -> Result<Vec<(usize, f32)>, HybridError> {
        let mut sc = Vec::new();
        self.fused_scores_adaptive(qtext, qvec, ad, &mut sc)?;
        Ok(top_n(&sc, k).into_iter().map(|d| (d, sc[d])).collect())
    }
    /// R40 (opt-in): MMR re-ranking of the top `pool` fused candidates. Greedy: pick the doc maximising
    /// `lambda * z_fused(d) - (1 - lambda) * redundancy_scale * max_{j in selected} cos(d, j)`; the first pick is always the plain top-1.
    /// `lambda = 1` reproduces [`HybridIndex::search`] exactly. Measured (R40, bridge queries): lambda .7-.8 raises top-5 domain coverage by +.02..+.035
    /// with no relevance gain (both-gold@5 unchanged) and no Holm loss; lambda .5 loses any-gold@5 (-.046). Use for ideation lists only.
    pub fn search_mmr(&self, qtext: &str, qvec: &[f32], k: usize, lambda: f32, pool: usize, redundancy_scale: f32) -> Result<Vec<(usize, f32)>, HybridError> {
        let mut sc = Vec::new();
        self.fused_scores(qtext, qvec, &mut sc)?;
        let mut zs = Vec::new();
        zscore_into(&sc, &mut zs);
        let mut rest = top_n(&sc, pool.max(k));
        let mut sel: Vec<usize> = Vec::new();
        let dot = |a: usize, b: usize| -> f32 {
            let (va, vb) = (&self.dense[a * self.dim..(a + 1) * self.dim], &self.dense[b * self.dim..(b + 1) * self.dim]);
            va.iter().zip(vb).map(|(x, y)| x * y).sum()
        };
        while sel.len() < k && !rest.is_empty() {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (pos, &d) in rest.iter().enumerate() {
                let red = sel.iter().map(|&j| dot(d, j)).fold(f32::NEG_INFINITY, f32::max);
                let red = if sel.is_empty() { 0.0 } else { red * redundancy_scale };
                let v = lambda * zs[d] - (1.0 - lambda) * red;
                if v > bv {
                    bv = v;
                    best = pos;
                }
            }
            sel.push(rest.remove(best));
        }
        Ok(sel.into_iter().map(|d| (d, sc[d])).collect())
    }
    /// top-k (doc, fused score), best first, ties by lower doc index
    pub fn search(&self, qtext: &str, qvec: &[f32], k: usize) -> Result<Vec<(usize, f32)>, HybridError> {
        let mut sc = Vec::new();
        self.fused_scores(qtext, qvec, &mut sc)?;
        Ok(top_n(&sc, k).into_iter().map(|d| (d, sc[d])).collect())
    }
}

// ---------------------------------------------------------------------------------------------
// spell corrector over a document vocabulary

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct SpellState {
    freq: BTreeMap<String, u32>,
    known: BTreeSet<String>,
}

/// Snaps out-of-vocabulary words (>= 4 letters) to the most frequent vocabulary word within optimal-string-alignment distance 1.
/// Vocabulary = lowercase `[a-z]+` words of the documents; words with frequency >= 2 and length >= 3 are correction targets;
/// every word seen at all (plus optional extra known words) is "known" and never changed.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "SpellState", into = "SpellState")]
pub struct SpellCorrector {
    state: SpellState,
    idx: HashMap<String, Vec<String>>,
}

impl PartialEq for SpellCorrector {
    fn eq(&self, o: &Self) -> bool {
        self.state == o.state
    }
}
impl From<SpellState> for SpellCorrector {
    fn from(state: SpellState) -> Self {
        let mut idx: HashMap<String, Vec<String>> = HashMap::new();
        for (w, &f) in &state.freq {
            if f >= 2 && w.len() >= 3 {
                idx.entry(w.clone()).or_default().push(w.clone());
                for d in del1(w) {
                    idx.entry(d).or_default().push(w.clone());
                }
            }
        }
        SpellCorrector { state, idx }
    }
}
impl From<SpellCorrector> for SpellState {
    fn from(s: SpellCorrector) -> Self {
        s.state
    }
}

fn words(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_ascii_alphabetic() {
            cur.push(c.to_ascii_lowercase());
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn del1(w: &str) -> BTreeSet<String> {
    let b = w.as_bytes();
    (0..b.len())
        .map(|i| {
            let mut v = Vec::with_capacity(b.len() - 1);
            v.extend_from_slice(&b[..i]);
            v.extend_from_slice(&b[i + 1..]);
            String::from_utf8(v).unwrap_or_default()
        })
        .collect()
}

/// true iff the optimal-string-alignment distance of a and b is exactly 1
fn osa1(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (la, lb) = (a.len(), b.len());
    if la.abs_diff(lb) > 1 || a == b {
        return false;
    }
    if la == lb {
        let d: Vec<usize> = (0..la).filter(|&i| a[i] != b[i]).collect();
        if d.len() == 1 {
            return true;
        }
        return d.len() == 2 && d[1] == d[0] + 1 && a[d[0]] == b[d[1]] && a[d[1]] == b[d[0]];
    }
    let (s, l) = if la < lb { (a, b) } else { (b, a) };
    let mut i = 0;
    while i < s.len() && s[i] == l[i] {
        i += 1;
    }
    s[i..] == l[i + 1..]
}

impl SpellCorrector {
    /// vocabulary from document texts
    pub fn from_docs(docs: &[String]) -> SpellCorrector {
        let mut freq: BTreeMap<String, u32> = BTreeMap::new();
        for d in docs {
            for w in words(d) {
                *freq.entry(w).or_insert(0) += 1;
            }
        }
        let known = freq.keys().cloned().collect();
        SpellCorrector::from(SpellState { freq, known })
    }
    /// additionally treat these words (e.g. frequent query words) as known, i.e. never rewrite them
    pub fn with_known<I: IntoIterator<Item = String>>(mut self, extra: I) -> SpellCorrector {
        self.state.known.extend(extra);
        self
    }
    pub fn vocab_size(&self) -> usize {
        self.state.freq.len()
    }
    /// correct one lowercase ASCII word
    pub fn fix_word(&self, t: &str) -> String {
        if self.state.known.contains(t) || t.len() < 4 {
            return t.to_string();
        }
        let mut cand: HashSet<&String> = HashSet::new();
        if let Some(v) = self.idx.get(t) {
            cand.extend(v.iter());
        }
        for d in del1(t) {
            if let Some(v) = self.idx.get(&d) {
                cand.extend(v.iter());
            }
        }
        let best = cand.into_iter().filter(|x| osa1(t, x)).max_by(|a, b| (self.state.freq[*a as &str], a.as_str()).cmp(&(self.state.freq[*b as &str], b.as_str())));
        match best {
            Some(x) => x.clone(),
            None => t.to_string(),
        }
    }
    /// correct a whole text: ASCII-letter runs are lowercased and corrected, everything else is kept verbatim
    pub fn correct(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let mut cur = String::new();
        for c in s.chars() {
            if c.is_ascii_alphabetic() {
                cur.push(c.to_ascii_lowercase());
            } else {
                if !cur.is_empty() {
                    out.push_str(&self.fix_word(&cur));
                    cur.clear();
                }
                out.push(c);
            }
        }
        if !cur.is_empty() {
            out.push_str(&self.fix_word(&cur));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn docs() -> Vec<String> {
        ["the quick brown fox jumps over the lazy dog", "receive the message and receive the reply", "database index tuning for the query planner", "fox hunting and dog training basics", "message queue database replication"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }
    /// deterministic pseudo-embedding: normalised hash-bag so that shared words give higher cosine
    fn emb(s: &str, dim: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        for t in tokens(s) {
            let mut h: u64 = 1469598103934665603;
            for b in t.bytes() {
                h = (h ^ b as u64).wrapping_mul(1099511628211);
            }
            v[(h % dim as u64) as usize] += 1.0;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        v
    }
    fn index(cfg: HybridConfig) -> HybridIndex {
        let d = docs();
        let dim = 32;
        let dense: Vec<f32> = d.iter().flat_map(|x| emb(x, dim)).collect();
        HybridIndex::new(&d, dense, dim, cfg).unwrap()
    }

    #[test]
    fn mmr_lambda_one_is_plain_search_and_diversifies() {
        let mut d = docs();
        d.push("the quick brown fox jumps over the lazy dog".to_string()); // exact duplicate of doc 0
        let dim = 32;
        let dense: Vec<f32> = d.iter().flat_map(|x| emb(x, dim)).collect();
        let ix = HybridIndex::new(&d, dense, dim, HybridConfig::default()).unwrap();
        let q = "quick brown fox jumps over the lazy dog";
        let qv = emb(q, dim);
        let plain: Vec<usize> = ix.search(q, &qv, 3).unwrap().iter().map(|x| x.0).collect();
        let m1: Vec<usize> = ix.search_mmr(q, &qv, 3, 1.0, 10, 4.0).unwrap().iter().map(|x| x.0).collect();
        assert_eq!(plain, m1);
        assert!(plain.contains(&0) && plain.contains(&5)); // plain top-k wastes two slots on the duplicate pair
        let m = ix.search_mmr(q, &qv, 2, 0.5, 10, 4.0).unwrap();
        assert_eq!(m.len(), 2);
        assert!(!(m.iter().any(|x| x.0 == 0) && m.iter().any(|x| x.0 == 5)), "MMR must not return both copies: {m:?}");
    }

    #[test]
    fn tokenizer() {
        assert_eq!(tokens("The Quick, brown-FOX; 42x!"), vec!["quick", "brown", "fox", "42x"]);
        assert!(tokens("the a an of").is_empty());
        assert!(tokens("").is_empty());
    }

    #[test]
    fn zscore_moments_and_constant() {
        let z = zscore(&[1.0, 2.0, 3.0, 4.0]);
        let m: f32 = z.iter().sum::<f32>() / 4.0;
        let v: f32 = z.iter().map(|x| x * x).sum::<f32>() / 4.0;
        assert!(m.abs() < 1e-6 && (v - 1.0).abs() < 1e-5);
        // zero variance (also numerically constant non-zero) -> all zeros, never +-1 noise
        assert_eq!(zscore(&[0.1f32; 7]), vec![0.0; 7]);
        assert_eq!(zscore(&[0.0; 5]), vec![0.0; 5]);
        assert!(zscore(&[]).is_empty());
        assert_eq!(zscore(&[3.0]), vec![0.0]);
    }

    #[test]
    fn top_n_ties_and_nan() {
        assert_eq!(top_n(&[1.0, 3.0, 3.0, 2.0], 3), vec![1, 2, 3]); // tie -> lower index first
        assert_eq!(top_n(&[f32::NAN, 1.0, 2.0], 3), vec![2, 1, 0]); // NaN last, no panic
        assert!(top_n(&[1.0, 2.0], 0).is_empty());
        assert_eq!(top_n(&[1.0, 2.0], 10), vec![1, 0]);
        assert!(top_n(&[], 5).is_empty());
    }

    #[test]
    fn bm25_basics_and_edges() {
        let b = Bm25::new(&docs());
        let mut s = Vec::new();
        b.score("database replication", &mut s);
        assert_eq!(top_n(&s, 1), vec![4]);
        b.score("database database database", &mut s); // duplicates counted once
        let mut s1 = Vec::new();
        b.score("database", &mut s1);
        assert_eq!(s, s1);
        b.score("zzzz qqqq", &mut s); // OOV
        assert!(s.iter().all(|x| *x == 0.0));
        b.score("", &mut s);
        assert!(s.iter().all(|x| *x == 0.0));
        let e = Bm25::new(&[]);
        let mut s = vec![1.0];
        e.score("anything", &mut s);
        assert!(s.is_empty() && e.is_empty());
        // docs that tokenise to nothing must not produce NaN
        let b = Bm25::new(&["the of".to_string(), "".to_string()]);
        b.score("the", &mut s);
        assert!(s.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn hybrid_ranks_and_alpha_extremes() {
        let ix = index(HybridConfig::default());
        let q = "database replication";
        let r = ix.search(q, &emb(q, 32), 3).unwrap();
        assert_eq!(r[0].0, 4);
        assert_eq!(r.len(), 3);
        assert!(r.windows(2).all(|w| w[0].1 >= w[1].1));
        // alpha = 1 -> dense order, alpha = 0 -> BM25 order on the matching docs
        let mut d1 = index(HybridConfig { alpha: 1.0 });
        let mut dense = Vec::new();
        d1.dense_scores(&emb(q, 32), &mut dense).unwrap();
        let a1: Vec<usize> = d1.search(q, &emb(q, 32), 5).unwrap().into_iter().map(|x| x.0).collect();
        assert_eq!(a1, top_n(&dense, 5));
        d1.set_alpha(0.0);
        let mut bm = Vec::new();
        d1.bm25().score(q, &mut bm);
        let a0: Vec<usize> = d1.search(q, &emb(q, 32), 2).unwrap().into_iter().map(|x| x.0).collect();
        assert_eq!(a0, top_n(&bm, 2));
    }

    #[test]
    fn hybrid_zero_variance_and_oov_query() {
        // OOV query -> BM25 constant (zeros) -> fused order must equal the dense order (z of constant = 0)
        let ix = index(HybridConfig::default());
        let qv = emb("database", 32);
        let r: Vec<usize> = ix.search("zzzzqq", &qv, 5).unwrap().into_iter().map(|x| x.0).collect();
        let mut dense = Vec::new();
        ix.dense_scores(&qv, &mut dense).unwrap();
        assert_eq!(r, top_n(&dense, 5));
        // zero query vector -> constant dense -> lexical order decides
        let r = ix.search("database replication", &vec![0.0; 32], 1).unwrap();
        assert_eq!(r[0].0, 4);
        // all-zero everything -> deterministic index order
        let r: Vec<usize> = ix.search("zzzz", &vec![0.0; 32], 5).unwrap().into_iter().map(|x| x.0).collect();
        assert_eq!(r, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn hybrid_empty_corpus_and_shape_errors() {
        let e = HybridIndex::new(&[], vec![], 8, HybridConfig::default()).unwrap();
        assert!(e.is_empty());
        assert!(e.search("x", &vec![0.0; 8], 5).unwrap().is_empty());
        assert_eq!(e.search("x", &[0.0; 3], 5), Err(HybridError::QueryDim { expected: 8, got: 3 }));
        assert_eq!(HybridIndex::new(&docs(), vec![0.0; 3], 4, HybridConfig::default()), Err(HybridError::DenseShape { expected: 20, got: 3 }));
        // bad alpha values are clamped / defaulted, never propagate NaN
        let ix = index(HybridConfig { alpha: f64::NAN });
        let r = ix.search("fox", &emb("fox", 32), 5).unwrap();
        assert!(r.iter().all(|x| x.1.is_finite()));
        let ix = index(HybridConfig { alpha: 7.0 });
        assert!(ix.search("fox", &emb("fox", 32), 5).unwrap().iter().all(|x| x.1.is_finite()));
    }

    #[test]
    fn hybrid_ties_and_determinism() {
        // duplicate documents -> identical scores -> lower index first, repeated builds identical
        let d: Vec<String> = vec!["alpha beta".into(), "alpha beta".into(), "gamma".into()];
        let dense: Vec<f32> = d.iter().flat_map(|x| emb(x, 16)).collect();
        let a = HybridIndex::new(&d, dense.clone(), 16, HybridConfig::default()).unwrap();
        let b = HybridIndex::new(&d, dense, 16, HybridConfig::default()).unwrap();
        assert_eq!(a, b);
        let r1 = a.search("alpha beta", &emb("alpha beta", 16), 3).unwrap();
        let r2 = b.search("alpha beta", &emb("alpha beta", 16), 3).unwrap();
        assert_eq!(r1, r2);
        assert_eq!((r1[0].0, r1[1].0), (0, 1));
        assert_eq!(r1[0].1.to_bits(), r1[1].1.to_bits());
    }

    #[test]
    fn hybrid_serde_roundtrip() {
        let ix = index(HybridConfig { alpha: 0.6 });
        let s = serde_json::to_string(&ix).unwrap();
        let back: HybridIndex = serde_json::from_str(&s).unwrap();
        assert_eq!(ix, back);
        assert_eq!(serde_json::to_string(&back).unwrap(), s); // stable serialisation (BTreeMap)
        let q = "message database";
        assert_eq!(ix.search(q, &emb(q, 32), 5).unwrap(), back.search(q, &emb(q, 32), 5).unwrap());
    }

    #[test]
    fn osa1_cases() {
        assert!(osa1("recieve", "receive")); // transposition
        assert!(osa1("databse", "database")); // deletion
        assert!(osa1("databasee", "database")); // insertion
        assert!(osa1("datebase", "database")); // substitution
        assert!(!osa1("database", "database"));
        assert!(!osa1("abcd", "abef"));
        assert!(!osa1("abcdef", "abcd"));
    }

    #[test]
    fn spell_fixes_and_leaves() {
        let sp = SpellCorrector::from_docs(&docs());
        assert_eq!(sp.fix_word("recieve"), "receive");
        assert_eq!(sp.fix_word("databse"), "database");
        assert_eq!(sp.fix_word("message"), "message"); // known
        assert_eq!(sp.fix_word("qxz"), "qxz"); // short
        assert_eq!(sp.fix_word("zzzzzz"), "zzzzzz"); // no candidate
        assert_eq!(sp.correct("Recieve the MESSAGE, databse!"), "receive the message, database!");
        assert_eq!(sp.correct("42 + x"), "42 + x");
        assert_eq!(sp.correct(""), "");
    }

    #[test]
    fn spell_tie_breaks_by_freq_then_lexicographic_max() {
        // 'tesa' is within 1 of 'tesb'(x3), 'tesc'(x3), 'tesd'(x2): max freq, then the lexicographically larger word
        let d: Vec<String> = vec!["tesb tesb tesb tesc tesc tesc tesd tesd".into()];
        let sp = SpellCorrector::from_docs(&d);
        assert_eq!(sp.fix_word("tesa"), "tesc");
        // words with freq 1 are known (never rewritten) but not correction targets
        let sp = SpellCorrector::from_docs(&["lonely word".to_string()]);
        assert_eq!(sp.fix_word("lonelx"), "lonelx");
    }

    #[test]
    fn spell_known_extra_empty_and_serde() {
        let sp = SpellCorrector::from_docs(&docs()).with_known(["recieve".to_string()]);
        assert_eq!(sp.fix_word("recieve"), "recieve"); // protected
        let e = SpellCorrector::from_docs(&[]);
        assert_eq!(e.correct("anything goes"), "anything goes");
        assert_eq!(e.vocab_size(), 0);
        let s = serde_json::to_string(&sp).unwrap();
        let back: SpellCorrector = serde_json::from_str(&s).unwrap();
        assert_eq!(sp, back);
        for w in ["databse", "recieve", "quikc", "mesage"] {
            assert_eq!(sp.fix_word(w), back.fix_word(w));
        }
        assert_eq!(serde_json::to_string(&back).unwrap(), s);
        assert_eq!(SpellCorrector::from_docs(&docs()), SpellCorrector::from_docs(&docs()));
    }

    #[test]
    fn spell_improves_hybrid_on_typo_query() {
        let ix = index(HybridConfig::default());
        let sp = SpellCorrector::from_docs(&docs());
        let typo = "databse mesage queue";
        let fixed = sp.correct(typo);
        assert_eq!(fixed, "database message queue");
        let r = ix.search(&fixed, &emb(&fixed, 32), 1).unwrap();
        assert_eq!(r[0].0, 4);
    }
}
