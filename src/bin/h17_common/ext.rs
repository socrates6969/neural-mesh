// h13 additions (copy-extension of h10_common, included at the end of h13_common/mod.rs):
// generic document store, 19 retrieval kinds, per-text evaluation, doc-level metrics, cluster bootstrap.

pub const GRID: [f64; 10] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]; // h17: grid extended below .4 (h13 used .4..1.0; hash dense is so weak that the optimum sits at the h13 edge)
pub const RR_BETA: [f64; 2] = [0.5, 1.0];
pub const RR_ALPHA: f64 = 0.7; // first-stage z-fusion weight of the rerankers (= the h10 pick; fixed, not tuned here)
pub const RRF_K: f64 = 60.0;
pub const LIST_N: usize = 50;
pub const RERANK_N: usize = 20;
pub const TOPN: usize = 8;
// kinds: 0 dense | 1 bm25 | 2..9 hybrid z (7 alphas) | 9..16 RSF (7 alphas) | 16 RRF | 17,18 rerank-20 (2 betas)
pub const NK: usize = 25;
pub const K_DENSE: usize = 0;
pub const K_BM25: usize = 1;
pub const K_Z0: usize = 2;
pub const K_RSF0: usize = 12;
pub const K_RRF: usize = 22;
pub const K_RR0: usize = 23;

pub const SYS_NAMES: [&str; 12] = ["dense_raw", "dense_s", "bm25_raw", "bm25_s", "hybz_raw", "hybz_s", "hybrrf_raw", "hybrrf_s", "hybrsf_raw", "hybrsf_s", "rerank_raw", "rerank_s"];

#[derive(Clone, Copy, Debug, Default)]
pub struct Picks {
    pub z: [usize; 2],   // grid index per flavor (raw, spell)
    pub rsf: [usize; 2], // grid index per flavor
    pub rr: [usize; 2],  // RR_BETA index per flavor
}
impl Picks {
    pub fn kind(&self, sys: usize) -> usize {
        let fl = sys % 2;
        match sys / 2 {
            0 => K_DENSE,
            1 => K_BM25,
            2 => K_Z0 + self.z[fl],
            3 => K_RRF,
            4 => K_RSF0 + self.rsf[fl],
            _ => K_RR0 + self.rr[fl],
        }
    }
    pub fn to_json(&self) -> Value {
        serde_json::json!({"z_alpha": [GRID[self.z[0]], GRID[self.z[1]]], "rsf_alpha": [GRID[self.rsf[0]], GRID[self.rsf[1]]], "rerank_beta": [RR_BETA[self.rr[0]], RR_BETA[self.rr[1]]], "rerank_first_stage_alpha": RR_ALPHA, "rrf_k": RRF_K, "list_n": LIST_N, "rerank_n": RERANK_N,
            "idx": {"z": self.z, "rsf": self.rsf, "rr": self.rr}})
    }
}

/// document store: vectors, BM25 index, per-doc token-id sets (for the cheap rerank / overlap statistics)
pub struct Store {
    pub ndoc: usize,
    pub dim: usize,
    pub demb: Vec<f32>,
    pub bm: Bm25,
    pub tokid: HashMap<String, u32>,
    pub idf: Vec<f32>,
    pub idf_oov: f32,
    pub dtok: Vec<Vec<u32>>, // sorted unique token ids per doc
}

impl Store {
    pub fn new(docs: &[String], demb: Vec<f32>, dim: usize) -> Store {
        assert_eq!(demb.len(), docs.len() * dim);
        let bm = Bm25::new(docs);
        let mut tokid: HashMap<String, u32> = HashMap::new();
        let mut df: Vec<u32> = Vec::new();
        let mut dtok = Vec::with_capacity(docs.len());
        for d in docs {
            let mut ids: Vec<u32> = tokens(d)
                .into_iter()
                .map(|t| {
                    let n = tokid.len() as u32;
                    let id = *tokid.entry(t).or_insert(n);
                    if id as usize == df.len() {
                        df.push(0);
                    }
                    id
                })
                .collect();
            ids.sort_unstable();
            ids.dedup();
            for &i in &ids {
                df[i as usize] += 1;
            }
            dtok.push(ids);
        }
        let n = docs.len() as f32;
        let idf: Vec<f32> = df.iter().map(|&f| ((n - f as f32 + 0.5) / (f as f32 + 0.5) + 1.0).ln()).collect();
        let idf_oov = ((n + 0.5) / 0.5 + 1.0).ln();
        Store { ndoc: docs.len(), dim, demb, bm, tokid, idf, idf_oov, dtok }
    }
    pub fn dense(&self, q: &[f32], out: &mut Vec<f32>) {
        out.clear();
        for d in 0..self.ndoc {
            let v = &self.demb[d * self.dim..(d + 1) * self.dim];
            let mut s = 0.0f32;
            for k in 0..self.dim {
                s += q[k] * v[k];
            }
            out.push(s);
        }
    }
    /// idf-weighted fraction of the query's unique content tokens that occur in doc `d` (OOV tokens count with the max idf)
    pub fn coverage(&self, qtoks: &[(Option<u32>, f32)], d: usize) -> f32 {
        let (mut num, mut den) = (0.0f32, 0.0f32);
        for &(id, w) in qtoks {
            den += w;
            if let Some(i) = id {
                if self.dtok[d].binary_search(&i).is_ok() {
                    num += w;
                }
            }
        }
        if den > 0.0 { num / den } else { 0.0 }
    }
    pub fn qtoks(&self, q: &str) -> Vec<(Option<u32>, f32)> {
        let mut seen: Vec<String> = Vec::new();
        let mut out = Vec::new();
        for t in tokens(q) {
            if seen.contains(&t) {
                continue;
            }
            match self.tokid.get(&t) {
                Some(&i) => out.push((Some(i), self.idf[i as usize])),
                None => out.push((None, self.idf_oov)),
            }
            seen.push(t);
        }
        out
    }
    /// plain (unweighted) fraction of unique query content tokens present in doc d
    pub fn coverage_plain(&self, qtoks: &[(Option<u32>, f32)], d: usize) -> f32 {
        if qtoks.is_empty() {
            return 0.0;
        }
        qtoks.iter().filter(|(id, _)| id.is_some_and(|i| self.dtok[d].binary_search(&i).is_ok())).count() as f32 / qtoks.len() as f32
    }
    pub fn bytes_dense(&self) -> usize {
        self.demb.len() * 4
    }
    pub fn bytes_bm25(&self) -> usize {
        self.bm.bytes()
    }
    pub fn bytes_tokens(&self) -> usize {
        self.dtok.iter().map(|v| v.len() * 4).sum::<usize>() + self.idf.len() * 4
    }
}

impl Bm25 {
    /// approximate index size: postings (u32 doc + f32 tf) + term strings + idf (f32) + doc lengths
    pub fn bytes(&self) -> usize {
        self.post.iter().map(|(t, p)| t.len() + p.len() * 8).sum::<usize>() + self.idf.len() * 4 + self.dl.len() * 4
    }
}

/// indices of the top-n scores, best first, ties broken by lower index
pub fn top_n(scores: &[f32], n: usize) -> Vec<usize> {
    let mut v: Vec<(f32, usize)> = scores.iter().copied().enumerate().map(|(i, s)| (s, i)).collect();
    let cmp = |a: &(f32, usize), b: &(f32, usize)| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1));
    let n = n.min(v.len());
    if v.len() > n {
        v.select_nth_unstable_by(n, cmp);
        v.truncate(n);
    }
    v.sort_by(cmp);
    v.into_iter().map(|x| x.1).collect()
}

pub fn fuse_z(zd: &[f32], zb: &[f32], a: f64, out: &mut Vec<f32>) {
    out.clear();
    out.extend(zd.iter().zip(zb).map(|(d, b)| a as f32 * d + (1.0 - a as f32) * b));
}

/// RRF over the top-LIST_N dense list and the top-LIST_N (positive-score) BM25 list; docs outside both lists score 0
pub fn fuse_rrf(dtop: &[usize], btop: &[usize], n: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(n, 0.0);
    for (r, &d) in dtop.iter().enumerate() {
        out[d] += (1.0 / (RRF_K + r as f64 + 1.0)) as f32;
    }
    for (r, &d) in btop.iter().enumerate() {
        out[d] += (1.0 / (RRF_K + r as f64 + 1.0)) as f32;
    }
}

/// RSF: min-max inside each top-LIST_N list, linear fusion; docs outside a list contribute 0 for it
pub fn fuse_rsf(dense: &[f32], dtop: &[usize], bm: &[f32], btop: &[usize], a: f64, out: &mut Vec<f32>) {
    out.clear();
    out.resize(dense.len(), 0.0);
    if let (Some(&f), Some(&l)) = (dtop.first(), dtop.last()) {
        let (hi, lo) = (dense[f], dense[l]);
        for &d in dtop {
            out[d] += a as f32 * (dense[d] - lo) / (hi - lo + 1e-9);
        }
    }
    if let (Some(&f), Some(&l)) = (btop.first(), btop.last()) {
        let (hi, lo) = (bm[f], bm[l]);
        for &d in btop {
            out[d] += (1.0 - a as f32) * (bm[d] - lo) / (hi - lo + 1e-9);
        }
    }
}

/// cheap top-20 rerank: first stage = z-fused scores `f`; the top RERANK_N candidates are re-scored by
/// beta * (idf-weighted query-token coverage) + (1-beta) * minmax(first stage) (+ 0.01*minmax tie-break); non-candidates stay below.
pub fn rerank20(st: &Store, qtoks: &[(Option<u32>, f32)], f: &[f32], beta: f64, out: &mut Vec<f32>) {
    out.clear();
    out.extend_from_slice(f);
    let cand = top_n(f, RERANK_N);
    let (hi, lo) = (f[cand[0]], f[*cand.last().unwrap()]);
    for &d in &cand {
        let mm = (f[d] - lo) / (hi - lo + 1e-9);
        let cov = st.coverage(qtoks, d);
        out[d] = 100.0 + beta as f32 * cov + (1.0 - beta as f32) * mm + 0.01 * mm;
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct Out {
    pub rank: u32,
    pub h1: u8,
    pub h3: u8,
    pub h5: u8,
}

#[derive(Clone)]
pub struct TextEval {
    pub outs: [Out; NK],
    /// per kind: top-TOPN (doc, dense cosine of that doc to the query text)
    pub tops: [[(u32, f32); TOPN]; NK],
}

fn finalize(scores: &[f32], dense: &[f32], rel: &[usize]) -> (Out, [(u32, f32); TOPN]) {
    let top = top_n(scores, TOPN);
    let mut t = [(0u32, 0f32); TOPN];
    for (i, &d) in top.iter().enumerate() {
        t[i] = (d as u32, dense[d]);
    }
    let hk = |k: usize| top.iter().take(k).filter(|d| rel.contains(d)).count() as u8;
    (Out { rank: best_rank(scores, rel) as u32, h1: hk(1), h3: hk(3), h5: hk(5) }, t)
}

/// evaluate one query text against the store with all 19 kinds
pub fn eval_text(st: &Store, qtext: &str, qvec: &[f32], rel: &[usize]) -> TextEval {
    let mut dense = Vec::new();
    st.dense(qvec, &mut dense);
    let mut bm = Vec::new();
    st.bm.score(qtext, &mut bm);
    let zd = zscore(&dense);
    let zb = zscore(&bm);
    let dtop = top_n(&dense, LIST_N);
    let mut btop = top_n(&bm, LIST_N);
    btop.retain(|&d| bm[d] > 0.0);
    let qt = st.qtoks(qtext);
    let mut outs = [Out::default(); NK];
    let mut tops = [[(0u32, 0f32); TOPN]; NK];
    let mut buf = Vec::new();
    let mut put = |k: usize, sc: &[f32]| {
        let (o, t) = finalize(sc, &dense, rel);
        outs[k] = o;
        tops[k] = t;
    };
    put(K_DENSE, &dense);
    put(K_BM25, &bm);
    for (gi, &a) in GRID.iter().enumerate() {
        fuse_z(&zd, &zb, a, &mut buf);
        put(K_Z0 + gi, &buf);
        fuse_rsf(&dense, &dtop, &bm, &btop, a, &mut buf);
        put(K_RSF0 + gi, &buf);
    }
    fuse_rrf(&dtop, &btop, st.ndoc, &mut buf);
    put(K_RRF, &buf);
    let mut f1 = Vec::new();
    fuse_z(&zd, &zb, RR_ALPHA, &mut f1);
    for (bi, &b) in RR_BETA.iter().enumerate() {
        rerank20(st, &qt, &f1, b, &mut buf);
        put(K_RR0 + bi, &buf);
    }
    TextEval { outs, tops }
}

// ---------------------------------------------------------------------------------------------
// metrics

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum M {
    Mrr,
    Hit1,
    Hit5,
    P1,
    P3,
    R3,
    F1at1,
    F1at3,
    F1at5,
}
pub const ALL_M: [M; 9] = [M::Mrr, M::Hit1, M::Hit5, M::P1, M::P3, M::R3, M::F1at1, M::F1at3, M::F1at5];
fn f1s(p: f64, r: f64) -> f64 {
    if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 }
}
pub fn metric(o: &Out, nrel: usize, m: M) -> f64 {
    let nr = nrel.max(1) as f64;
    match m {
        M::Mrr => 1.0 / o.rank as f64,
        M::Hit1 => (o.rank == 1) as u8 as f64,
        M::Hit5 => (o.rank <= 5) as u8 as f64,
        M::P1 => o.h1 as f64,
        M::P3 => o.h3 as f64 / 3.0,
        M::R3 => o.h3 as f64 / nr,
        M::F1at1 => f1s(o.h1 as f64, o.h1 as f64 / nr),
        M::F1at3 => f1s(o.h3 as f64 / 3.0, o.h3 as f64 / nr),
        M::F1at5 => f1s(o.h5 as f64 / 5.0, o.h5 as f64 / nr),
    }
}

/// percentile bootstrap over cluster units; returns per contrast (mean, lo, hi, se). Same resamples for all contrasts.
pub fn boot_cluster_multi(diffs: &[Vec<f64>], reps: usize, seed: u64) -> Vec<(f64, f64, f64, f64)> {
    let nu = diffs[0].len();
    let nc = diffs.len();
    let mut r = Xs(seed | 1);
    let mut ms: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); nc];
    let mut idx = vec![0usize; nu];
    for _ in 0..reps {
        for i in idx.iter_mut() {
            *i = r.below(nu);
        }
        for c in 0..nc {
            let d = &diffs[c];
            let s: f64 = idx.iter().map(|&i| d[i]).sum();
            ms[c].push(s / nu as f64);
        }
    }
    (0..nc)
        .map(|c| {
            let m = mean(&diffs[c]);
            let mut v = std::mem::take(&mut ms[c]);
            let mb = mean(&v);
            let se = (v.iter().map(|x| (x - mb) * (x - mb)).sum::<f64>() / (reps as f64 - 1.0)).sqrt();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (m, v[(reps as f64 * 0.025) as usize], v[((reps as f64 * 0.975) as usize).min(reps - 1)], se)
        })
        .collect()
}

/// two-sided normal p-value 2*(1-Phi(|z|)) (Abramowitz-Stegun 7.1.26 erfc approximation)
pub fn norm_two_sided_p(z: f64) -> f64 {
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    y.clamp(0.0, 1.0)
}

pub fn json_write(p: &str, v: &Value) {
    std::fs::write(p, serde_json::to_string_pretty(v).unwrap()).unwrap();
}
