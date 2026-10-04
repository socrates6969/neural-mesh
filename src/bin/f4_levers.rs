//! F4: the three levers of a "better researcher", measured on a seeded synthetic world.
//!   (a) retrieval quality before routing (P/R/F1 vs ground-truth concept; mean-centering, query expansion,
//!       relative-margin, MMR, thresholds) and what it buys downstream,
//!   (b) sample efficiency: graded outcomes needed to reach a target quality / routing accuracy with vs without
//!       the reuse path, warm start and retrieval,
//!   (c) generator -> critic: catch rate / false-alarm rate / net-of-cost value of the verification step and of
//!       better-calibrated triggers (self-confidence, retrieval disagreement, no-support).
//! Protocol: tune on seeds 1-8, evaluate held-out 31-60 (default), confirm 61-90 with the frozen tuning
//! (--held 61-90). Paired 95% t-CI, Holm per family. All embeddings / skills / error models are SYNTHETIC.
//!
//! Usage: cargo run --release --bin f4_levers -- [--part a|b|c|all] [--held 61-90] [--out results/f4_levers.json] [--quick]

use std::collections::HashMap;
use std::time::Instant;

use neural_mesh::coupling::{heb_robust_cfg, CoupledBrain, CouplingConfig, WarmEvidence};
use neural_mesh::router::{Rng, Task};
use neural_mesh::stats::{holm_adjust, paired_diff, paired_t_pvalue};
use neural_mesh::variants::{HierSecondOpinion, HierSoConfig, SecondOpinionConfig};
use rayon::prelude::*;
use serde_json::{json, Value};

const DIM: usize = 48;
const N_TASKS: usize = 2000;
const N_AGENTS: usize = 8;
const REVEAL: f64 = 0.5;
const DELAY: usize = 20;

// =============================================================================================
// common

fn normal(r: &mut Rng) -> f64 {
    let u1 = r.f64().max(1e-12);
    let u2 = r.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
fn hash(seed: u64, a: u64, b: u64, c: u64) -> u64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ a.wrapping_mul(0xBF58_476D_1CE4_E5B9) ^ b.wrapping_mul(0x94D0_49BB_1331_11EB) ^ c.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
fn uni(seed: u64, a: u64, b: u64, c: u64) -> f64 {
    (hash(seed, a, b, c) >> 11) as f64 / (1u64 << 53) as f64
}
fn nrm(seed: u64, a: u64, b: u64, c: u64) -> f64 {
    let (u1, u2) = (uni(seed, a, b, c).max(1e-12), uni(seed, a, b, c ^ 0xABCD));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[derive(Clone, Debug)]
struct Cmp {
    d: f64,
    lo: f64,
    hi: f64,
    p: f64,
    w: usize,
    l: usize,
}
fn cmp(a: &[f64], b: &[f64]) -> Cmp {
    let pd = paired_diff(a, b);
    Cmp { d: pd.diff.mean, lo: pd.diff.lo, hi: pd.diff.hi, p: paired_t_pvalue(a, b), w: pd.wins, l: pd.losses }
}
fn verdict(adj_p: f64, d: f64) -> &'static str {
    if adj_p < 0.05 && d > 0.0 {
        "WIN"
    } else if adj_p < 0.05 && d < 0.0 {
        "LOSS"
    } else {
        "n.s."
    }
}
fn cj(c: &Cmp, adj: f64) -> Value {
    json!({"diff": c.d, "lo": c.lo, "hi": c.hi, "p": c.p, "holm_p": adj, "verdict": verdict(adj, c.d), "wins": c.w, "losses": c.l})
}

// =============================================================================================
// embedding model (copied from e2_embed: shared direction + topic group + concept + sub-mode + drift + noise)

#[derive(Clone, Debug)]
struct EmbCfg {
    g: f64,
    h: f64,
    ntop: usize,
    p: f64,
    m: usize,
    spread: f64,
    noise: f64,
    drift: f64,
}
fn spectrum(p: f64) -> Vec<f64> {
    let a: Vec<f64> = (0..DIM).map(|i| ((i + 1) as f64).powf(-p)).collect();
    let n = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    a.into_iter().map(|x| x / n).collect()
}
fn avec(r: &mut Rng, a: &[f64], scale: f64) -> Vec<f64> {
    a.iter().map(|ai| ai * normal(r) * scale).collect()
}
fn addv(v: &mut [f64], u: &[f64]) {
    for (x, y) in v.iter_mut().zip(u) {
        *x += y;
    }
}
fn unit(v: &[f64]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
    v.iter().map(|x| (x / n) as f32).collect()
}
fn iso() -> EmbCfg {
    EmbCfg { g: 0.0, h: 0.0, ntop: 1, p: 0.0, m: 1, spread: 0.0, noise: 0.5, drift: 0.0 }
}
fn re(noise: f64) -> EmbCfg {
    EmbCfg { g: 1.2, h: 0.8, ntop: 4, p: 0.5, m: 1, spread: 0.0, noise, drift: 0.0 }
}
fn para(noise: f64) -> EmbCfg {
    EmbCfg { m: 4, spread: 0.9, ..re(noise) }
}
fn drf(noise: f64, d: f64) -> EmbCfg {
    EmbCfg { drift: d, ..re(noise) }
}

/// n embeddings of k recurring concepts (Zipf 0.7 frequency; k == 0: every text its own concept).
fn gen_stream(seed: u64, n: usize, k: usize, e: &EmbCfg) -> (Vec<Vec<f32>>, Vec<usize>) {
    let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xF4F4_1234));
    let a = spectrum(e.p);
    let kk = if k == 0 { n } else { k };
    let shared: Vec<f64> = {
        let v: Vec<f64> = a.iter().map(|ai| ai * (1.0 + 0.3 * normal(&mut r))).collect();
        let nn = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
        v.into_iter().map(|x| e.g * x / nn).collect()
    };
    let tops: Vec<Vec<f64>> = (0..e.ntop.max(1)).map(|_| avec(&mut r, &a, e.h)).collect();
    let cents: Vec<Vec<f64>> = (0..kk)
        .map(|c| {
            let mut v = avec(&mut r, &a, 1.0);
            addv(&mut v, &shared);
            addv(&mut v, &tops[c % tops.len()]);
            v
        })
        .collect();
    let subs: Vec<Vec<Vec<f64>>> = (0..kk).map(|_| (0..e.m.max(1)).map(|_| if e.m > 1 { avec(&mut r, &a, e.spread) } else { vec![0.0; DIM] }).collect()).collect();
    let dirs: Vec<Vec<f64>> = (0..kk).map(|_| avec(&mut r, &a, 1.0)).collect();
    let wts: Vec<f64> = (0..kk).map(|i| 1.0 / ((i + 1) as f64).powf(0.7)).collect();
    let tot: f64 = wts.iter().sum();
    let (mut emb, mut con) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for t in 0..n {
        let c = if k == 0 {
            t
        } else {
            let mut x = r.f64() * tot;
            let mut c = kk - 1;
            for (i, wt) in wts.iter().enumerate() {
                if x < *wt {
                    c = i;
                    break;
                }
                x -= wt;
            }
            c
        };
        let j = r.below(e.m.max(1));
        let mut v = cents[c].clone();
        addv(&mut v, &subs[c][j]);
        let f = e.drift * (t as f64 / n as f64);
        for (x, d) in v.iter_mut().zip(&dirs[c]) {
            *x += f * d;
        }
        let nz = avec(&mut r, &a, e.noise);
        addv(&mut v, &nz);
        emb.push(unit(&v));
        con.push(c);
    }
    (emb, con)
}

// =============================================================================================
// retriever: flat store, optional online mean-centering, query expansion, relative margin, MMR

#[derive(Clone, Debug)]
struct RetCfg {
    center: bool,
    thr: f32,
    k: usize,
    /// (m neighbours, beta): Rocchio pseudo-relevance feedback
    expand: Option<(usize, f32)>,
    /// keep neighbours within this margin of the best
    rel: Option<f32>,
    /// MMR lambda over the top 3k candidates
    mmr: Option<f32>,
}
impl RetCfg {
    fn raw(thr: f32, k: usize) -> Self {
        RetCfg { center: false, thr, k, expand: None, rel: None, mmr: None }
    }
    fn label(&self) -> String {
        let mut s = format!("{}thr{:.2}k{}", if self.center { "C-" } else { "R-" }, self.thr, self.k);
        if let Some((m, b)) = self.expand {
            s += &format!("-exp(m{m},b{b})");
        }
        if let Some(d) = self.rel {
            s += &format!("-rel{d}");
        }
        if let Some(l) = self.mmr {
            s += &format!("-mmr{l}");
        }
        s
    }
}

struct Store {
    cfg: RetCfg,
    raw: Vec<f32>,
    cen: Vec<f32>,
    sum: Vec<f64>,
    mu: Vec<f32>,
    active: bool,
    nobs: usize,
}
impl Store {
    fn new(cfg: &RetCfg) -> Self {
        Store { cfg: cfg.clone(), raw: vec![], cen: vec![], sum: vec![0.0; DIM], mu: vec![0.0; DIM], active: false, nobs: 0 }
    }
    fn n(&self) -> usize {
        self.raw.len() / DIM
    }
    fn centered(&self, x: &[f32]) -> Vec<f32> {
        if !self.active {
            return x.to_vec();
        }
        let mut v: Vec<f32> = x.iter().zip(&self.mu).map(|(a, m)| a - m).collect();
        let n = dot(&v, &v).sqrt().max(1e-9);
        for a in v.iter_mut() {
            *a /= n;
        }
        v
    }
    /// every task embedding the system sees updates the mean (stored or not)
    fn observe(&mut self, x: &[f32]) {
        if !self.cfg.center {
            return;
        }
        for j in 0..DIM {
            self.sum[j] += x[j] as f64;
        }
        self.nobs += 1;
        if self.nobs >= 30 && (self.nobs == 30 || self.nobs % 50 == 0) {
            for j in 0..DIM {
                self.mu[j] = (self.sum[j] / self.nobs as f64) as f32;
            }
            self.active = true;
            let n = self.n();
            for i in 0..n {
                let c = self.centered(&self.raw[i * DIM..(i + 1) * DIM]);
                self.cen[i * DIM..(i + 1) * DIM].copy_from_slice(&c);
            }
        }
    }
    fn add(&mut self, x: &[f32]) {
        let c = self.centered(x);
        self.raw.extend_from_slice(x);
        self.cen.extend_from_slice(&c);
    }
    fn sims(&self, q: &[f32]) -> Vec<f32> {
        (0..self.n()).map(|i| dot(&self.cen[i * DIM..(i + 1) * DIM], q)).collect()
    }
    fn query(&self, q: &[f32]) -> Vec<(usize, f32)> {
        let n = self.n();
        if n == 0 {
            return vec![];
        }
        let c = &self.cfg;
        let qc = self.centered(q);
        let mut s = self.sims(&qc);
        let pick = |s: &[f32]| -> Vec<(usize, f32)> {
            let mut v: Vec<(usize, f32)> = s.iter().enumerate().filter(|(_, x)| **x >= c.thr).map(|(i, x)| (i, *x)).collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            v
        };
        let mut cand = pick(&s);
        if let Some((m, beta)) = c.expand {
            if !cand.is_empty() {
                let mm = m.min(cand.len());
                let mut qp = qc.clone();
                for &(i, _) in &cand[..mm] {
                    for j in 0..DIM {
                        qp[j] += beta * self.cen[i * DIM + j] / mm as f32;
                    }
                }
                let nn = dot(&qp, &qp).sqrt().max(1e-9);
                for a in qp.iter_mut() {
                    *a /= nn;
                }
                s = self.sims(&qp);
                cand = pick(&s);
            }
        }
        if let (Some(d), Some(&(_, top))) = (c.rel, cand.first()) {
            cand.retain(|x| x.1 >= top - d);
        }
        if let Some(lam) = c.mmr {
            let pool: Vec<(usize, f32)> = cand.into_iter().take(3 * c.k).collect();
            let mut sel: Vec<(usize, f32)> = vec![];
            let mut left = pool;
            while sel.len() < c.k && !left.is_empty() {
                let mut bi = 0;
                let mut bv = f32::NEG_INFINITY;
                for (ix, &(i, si)) in left.iter().enumerate() {
                    let red = sel.iter().map(|&(j, _)| dot(&self.cen[i * DIM..(i + 1) * DIM], &self.cen[j * DIM..(j + 1) * DIM])).fold(f32::NEG_INFINITY, f32::max);
                    let red = if sel.is_empty() { 0.0 } else { red };
                    let v = lam * si - (1.0 - lam) * red;
                    if v > bv {
                        bv = v;
                        bi = ix;
                    }
                }
                sel.push(left.remove(bi));
            }
            return sel;
        }
        cand.truncate(c.k);
        cand
    }
}

// =============================================================================================
// PART A: retrieval P/R/F1 vs ground-truth concept

#[derive(Clone)]
struct AW {
    name: &'static str,
    emb: EmbCfg,
    k: usize,
    treat: bool,
}
fn a_worlds() -> Vec<AW> {
    let aw = |name, emb, k, treat| AW { name, emb, k, treat };
    vec![
        aw("iso-ref", iso(), 12, false),
        aw("aniso-n0.8", re(0.8), 12, true),
        aw("aniso-n1.2", re(1.2), 12, true),
        aw("aniso-n1.6", re(1.6), 12, true),
        aw("para-n0.8", para(0.8), 12, true),
        aw("para-n1.2", para(1.2), 12, true),
        aw("drift1-n0.8", drf(0.8, 1.0), 12, true),
        aw("para+drift1-n1.2", EmbCfg { drift: 1.0, ..para(1.2) }, 12, true),
        aw("k48-aniso-n0.8", re(0.8), 48, true),
        aw("k96-para-n1.2", para(1.2), 96, true),
        aw("ctl-unique", re(0.8), 0, false),
    ]
}
/// [P, R, F1, coverage(hit rate among queries with a relevant item stored), false-hit rate (queries with no
/// relevant item stored but something retrieved), anchor accuracy, anchors per concept, mean retrieved]
const NA: usize = 9;
fn eval_a(w: &AW, cfg: &RetCfg, seed: u64) -> [f64; NA] {
    let (emb, con) = gen_stream(seed, N_TASKS, w.k, &w.emb);
    let mut st = Store::new(cfg);
    let mut rel_cnt: HashMap<usize, usize> = HashMap::new();
    let mut anchor_of: Vec<usize> = Vec::with_capacity(N_TASKS);
    let (mut sum_r, mut tp, mut den_r) = (0.0, 0.0, 0.0);
    let (mut cov_n, mut cov_hit, mut nr_n, mut nr_hit) = (0.0, 0.0, 0.0, 0.0);
    let mut assigned: Vec<(usize, usize)> = vec![];
    let (mut sum_e, mut tp_e, mut den_e) = (0.0, 0.0, 0.0);
    for t in 0..N_TASKS {
        let x = &emb[t];
        st.observe(x);
        let r = st.query(x);
        let c = con[t];
        let rel = rel_cnt.get(&c).copied().unwrap_or(0);
        let hits_same = r.iter().filter(|(i, _)| con[*i] == c).count();
        sum_r += r.len() as f64;
        tp += hits_same as f64;
        den_r += cfg.k.min(rel) as f64;
        if t < 300 {
            sum_e += r.len() as f64;
            tp_e += hits_same as f64;
            den_e += cfg.k.min(rel) as f64;
        }
        if rel > 0 {
            cov_n += 1.0;
            if !r.is_empty() {
                cov_hit += 1.0;
            }
        } else {
            nr_n += 1.0;
            if !r.is_empty() {
                nr_hit += 1.0;
            }
        }
        // anchor vote (system-formed anchors, as in Knowledge)
        let mut votes: Vec<(usize, f32)> = vec![];
        for &(i, s) in &r {
            let a = anchor_of[i];
            match votes.iter_mut().find(|v| v.0 == a) {
                Some(v) => v.1 += s,
                None => votes.push((a, s)),
            }
        }
        let win = votes.iter().fold(None, |b: Option<(usize, f32)>, &v| match b {
            Some(x) if x.1 >= v.1 => b,
            _ => Some(v),
        });
        if let Some((a, _)) = win {
            if rel > 0 {
                assigned.push((t, a));
            }
        }
        anchor_of.push(win.map(|v| v.0).unwrap_or(t));
        st.add(x);
        *rel_cnt.entry(c).or_insert(0) += 1;
    }
    let p = if sum_r > 0.0 { tp / sum_r } else { 0.0 };
    let rc = if den_r > 0.0 { tp / den_r } else { 0.0 };
    let f1 = if p + rc > 0.0 { 2.0 * p * rc / (p + rc) } else { 0.0 };
    // anchor quality (post hoc): pair-F1 between system anchors and true concepts over all items
    let c2 = |n: usize| (n * n.saturating_sub(1) / 2) as f64;
    let mut cont: HashMap<(usize, usize), usize> = HashMap::new();
    let (mut na_cnt, mut nc_cnt): (HashMap<usize, usize>, HashMap<usize, usize>) = (HashMap::new(), HashMap::new());
    for (t, &a) in anchor_of.iter().enumerate() {
        *cont.entry((a, con[t])).or_insert(0) += 1;
        *na_cnt.entry(a).or_insert(0) += 1;
        *nc_cnt.entry(con[t]).or_insert(0) += 1;
    }
    let pair_tp: f64 = cont.values().map(|&n| c2(n)).sum();
    let (pair_a, pair_c): (f64, f64) = (na_cnt.values().map(|&n| c2(n)).sum(), nc_cnt.values().map(|&n| c2(n)).sum());
    let pair_f1 = if pair_a + pair_c > 0.0 { 2.0 * pair_tp / (pair_a + pair_c) } else { 0.0 };
    let _ = &assigned;
    let (pe, re_) = (if sum_e > 0.0 { tp_e / sum_e } else { 0.0 }, if den_e > 0.0 { tp_e / den_e } else { 0.0 });
    let f1e = if pe + re_ > 0.0 { 2.0 * pe * re_ / (pe + re_) } else { 0.0 };
    // anchors per concept
    let mut per: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
    for (t, &a) in anchor_of.iter().enumerate() {
        per.entry(con[t]).or_default().insert(a);
    }
    let frag = if w.k > 0 { per.values().map(|s| s.len() as f64).sum::<f64>() / per.len().max(1) as f64 } else { f64::NAN };
    [p, rc, f1, if cov_n > 0.0 { cov_hit / cov_n } else { 0.0 }, if nr_n > 0.0 { nr_hit / nr_n } else { 0.0 }, pair_f1, frag, sum_r / N_TASKS as f64, f1e]
}

fn a_grid(quick: bool) -> Vec<(&'static str, RetCfg)> {
    let mut g: Vec<(&'static str, RetCfg)> = vec![];
    let ks: &[usize] = if quick { &[5] } else { &[3, 5, 8] };
    for &k in ks {
        for &t in &[0.4f32, 0.5, 0.6, 0.7, 0.8, 0.9] {
            g.push(("raw", RetCfg::raw(t, k)));
        }
        for &t in &[0.2f32, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] {
            g.push(("center", RetCfg { center: true, thr: t, k, expand: None, rel: None, mmr: None }));
        }
    }
    for &m in &[2usize, 3] {
        for &b in &[0.5f32, 1.0] {
            for &t in &[0.3f32, 0.4, 0.5, 0.6, 0.7] {
                g.push(("center+expand", RetCfg { center: true, thr: t, k: 5, expand: Some((m, b)), rel: None, mmr: None }));
            }
            for &t in &[0.5f32, 0.6, 0.7, 0.8] {
                g.push(("raw+expand", RetCfg { center: false, thr: t, k: 5, expand: Some((m, b)), rel: None, mmr: None }));
            }
        }
    }
    for &d in &[0.05f32, 0.1, 0.2] {
        for &t in &[0.2f32, 0.3, 0.4, 0.5] {
            for &k in &[5usize, 8] {
                g.push(("center+rel", RetCfg { center: true, thr: t, k, expand: None, rel: Some(d), mmr: None }));
            }
        }
        for &t in &[0.5f32, 0.6, 0.7] {
            g.push(("raw+rel", RetCfg { center: false, thr: t, k: 5, expand: None, rel: Some(d), mmr: None }));
        }
    }
    for &l in &[0.5f32, 0.7, 0.9] {
        for &t in &[0.3f32, 0.4, 0.5, 0.6] {
            g.push(("center+mmr", RetCfg { center: true, thr: t, k: 5, expand: None, rel: None, mmr: Some(l) }));
        }
    }
    for &t in &[0.5f32, 0.6, 0.7] {
        g.push(("raw+mmr", RetCfg { center: false, thr: t, k: 5, expand: None, rel: None, mmr: Some(0.7) }));
    }
    if quick {
        g.truncate(60);
    }
    g
}

struct APart {
    best_raw: RetCfg,
    best_center: RetCfg,
    best_overall: RetCfg,
    best_overall_family: String,
    json: Value,
}

fn part_a(held: &[u64], quick: bool) -> APart {
    let t0 = Instant::now();
    let worlds = a_worlds();
    let grid = a_grid(quick);
    let tune: Vec<u64> = if quick { (1..=2).collect() } else { (1..=8).collect() };
    let treat: Vec<usize> = worlds.iter().enumerate().filter(|(_, w)| w.treat).map(|(i, _)| i).collect();
    // ---- tuning on seeds 1-8: objective = mean F1 over treatment worlds
    let jobs: Vec<(usize, usize, u64)> = { let mut j = vec![]; for c in 0..grid.len() { for &w in &treat { for &s in &tune { j.push((c, w, s)); } } } j };
    let res: Vec<(usize, f64, f64)> = jobs.par_iter().map(|&(c, w, s)| { let r = eval_a(&worlds[w], &grid[c].1, s); (c, r[2], r[5]) }).collect();
    let mut obj = vec![0.0; grid.len()];
    let mut objp = vec![0.0; grid.len()];
    let per = (treat.len() * tune.len()) as f64;
    for (c, f, pu) in res {
        obj[c] += f / per;
        objp[c] += pu / per;
    }
    let pur_i = (0..grid.len()).max_by(|&a, &b| objp[a].partial_cmp(&objp[b]).unwrap()).unwrap();
    println!("   best-by-anchor-pairF1 {:<40} pairF1 {:.4} (F1 {:.4})", grid[pur_i].1.label(), objp[pur_i], obj[pur_i]);
    let mut fam: Vec<(&'static str, usize)> = vec![];
    for (ci, (fname, _)) in grid.iter().enumerate() {
        match fam.iter_mut().find(|f| f.0 == *fname) {
            Some(f) => {
                if obj[ci] > obj[f.1] {
                    f.1 = ci;
                }
            }
            None => fam.push((fname, ci)),
        }
    }
    println!("[A tune] seeds {}-{} objective = mean F1 over {} treatment worlds ({} configs, {:.0}s)", tune[0], tune[tune.len() - 1], treat.len(), grid.len(), t0.elapsed().as_secs_f64());
    for (fname, ci) in &fam {
        println!("   best {:<14} {:<40} F1 {:.4}", fname, grid[*ci].1.label(), obj[*ci]);
    }
    // baselines
    let default_cfg = RetCfg::raw(0.6, 5); // research.rs default (sim_min .6, k 5)
    let dflt_obj = {
        let jobs: Vec<(usize, u64)> = treat.iter().flat_map(|&w| tune.iter().map(move |&s| (w, s))).collect();
        jobs.par_iter().map(|&(w, s)| eval_a(&worlds[w], &default_cfg, s)[2]).sum::<f64>() / per
    };
    println!("   baseline research-default raw thr.6 k5 tune-F1 {dflt_obj:.4}");
    let mut arms: Vec<(String, RetCfg)> = vec![("research-default".into(), default_cfg.clone())];
    for (fname, ci) in &fam {
        arms.push((format!("{fname}*"), grid[*ci].1.clone()));
    }
    arms.push(("anchorF1-tuned*".into(), grid[pur_i].1.clone()));
    let raw_i = fam.iter().find(|f| f.0 == "raw").unwrap().1;
    let cen_i = fam.iter().find(|f| f.0 == "center").unwrap().1;
    let best_i = fam.iter().max_by(|a, b| obj[a.1].partial_cmp(&obj[b.1]).unwrap()).unwrap();
    // ---- evaluation
    let ev_jobs: Vec<(usize, usize, u64)> = (0..arms.len()).flat_map(|a| (0..worlds.len()).flat_map(move |w| held.iter().map(move |&s| (a, w, s)))).collect();
    let ev: Vec<(usize, usize, [f64; NA])> = ev_jobs.par_iter().map(|&(a, w, s)| (a, w, eval_a(&worlds[w], &arms[a].1, s))).collect();
    let mut cells: HashMap<(usize, usize), Vec<[f64; NA]>> = HashMap::new();
    for (a, w, r) in ev {
        cells.entry((a, w)).or_default().push(r);
    }
    let colv = |a: usize, w: usize, m: usize| -> Vec<f64> { cells[&(a, w)].iter().map(|r| r[m]).collect() };
    // tests: each family arm vs research-default and vs best raw, metrics F1 & anchor acc
    let raw_arm = arms.iter().position(|x| x.0 == "raw*").unwrap();
    let mut tests: Vec<(usize, usize, usize, usize, Cmp)> = vec![]; // arm, baseline, world, metric
    for a in 1..arms.len() {
        for &b in &[0usize, raw_arm] {
            if a == b {
                continue;
            }
            for w in 0..worlds.len() {
                for &m in &[2usize, 5, 8] {
                    tests.push((a, b, w, m, cmp(&colv(a, w, m), &colv(b, w, m))));
                }
            }
        }
    }
    let adj = holm_adjust(&tests.iter().map(|t| t.4.p).collect::<Vec<_>>());
    let mut tj = vec![];
    let (mut nw, mut nl) = (0, 0);
    for (k, t) in tests.iter().enumerate() {
        let v = verdict(adj[k], t.4.d);
        if v == "WIN" {
            nw += 1
        } else if v == "LOSS" {
            nl += 1
        }
        let mut j = cj(&t.4, adj[k]);
        j["arm"] = json!(arms[t.0].0);
        j["baseline"] = json!(arms[t.1].0);
        j["world"] = json!(worlds[t.2].name);
        j["metric"] = json!(["P", "R", "F1", "coverage", "false_hit", "anchor_pairF1", "anchors_per_concept", "retrieved_per_query", "F1_first300"][t.3]);
        tj.push(j);
    }
    println!("\n[A eval] held-out seeds {}-{}: mean P / R / F1 | anchor-acc | false-hit rate | anchors-per-concept", held[0], held[held.len() - 1]);
    let mut table = vec![];
    for (wi, w) in worlds.iter().enumerate() {
        println!("  world {}", w.name);
        let mut rows = vec![];
        for (ai, (an, ac)) in arms.iter().enumerate() {
            let m = |i: usize| mean(&colv(ai, wi, i));
            println!("    {:<18} {:<36} P {:.3} R {:.3} F1 {:.3} (first300 {:.3}) | anc-pairF1 {:.3} | fh {:.3} | frag {:.2} | ret/q {:.2}", an, ac.label(), m(0), m(1), m(2), m(8), m(5), m(4), m(6), m(7));
            rows.push(json!({"arm": an, "cfg": ac.label(), "P": m(0), "R": m(1), "F1": m(2), "coverage": m(3), "false_hit_rate": m(4), "anchor_pairF1": m(5), "anchors_per_concept": m(6), "retrieved_per_query": m(7), "F1_first300": m(8)}));
        }
        table.push(json!({"world": w.name, "treatment": w.treat, "arms": rows}));
    }
    println!("\n[A] Holm family {} tests: WIN {nw} LOSS {nl}", tests.len());
    for (k, t) in tests.iter().enumerate() {
        if t.1 == 0 && t.3 == 2 && verdict(adj[k], t.4.d) != "n.s." {
            println!("   {:<14} vs default  {:<18} F1 {:+.4} [{:+.4},{:+.4}] {}", arms[t.0].0, worlds[t.2].name, t.4.d, t.4.lo, t.4.hi, verdict(adj[k], t.4.d));
        }
    }
    let json = json!({
        "tune_seeds": format!("{}-{}", tune[0], tune[tune.len()-1]), "eval_seeds": format!("{}-{}", held[0], held[held.len()-1]),
        "objective": "mean micro-F1 over treatment worlds (not iso-ref, not ctl-unique)", "n_configs": grid.len(),
        "purity_arm": {"cfg": grid[pur_i].1.label(), "tune_anchor_pairF1": objp[pur_i], "tune_F1": obj[pur_i]}, "tuned": fam.iter().map(|(f, ci)| json!({"family": f, "cfg": grid[*ci].1.label(), "tune_F1": obj[*ci]})).collect::<Vec<_>>(),
        "baseline_tune_F1": dflt_obj,
        "table": table, "tests": tj, "holm_win": nw, "holm_loss": nl, "runtime_s": t0.elapsed().as_secs_f64(),
    });
    APart { best_raw: grid[raw_i].1.clone(), best_center: grid[cen_i].1.clone(), best_overall: grid[best_i.1].1.clone(), best_overall_family: best_i.0.to_string(), json }
}

// =============================================================================================
// session world (shared by parts A-downstream, B and C)

#[derive(Clone, Copy, Debug, PartialEq)]
enum LazyMode {
    Third,
    None,
}
#[derive(Clone, Debug)]
struct SW {
    name: String,
    emb: EmbCfg,
    k: usize,
    hal_base: f64,
    hal_floor: f64,
    ctx_gain: f64,
    ctx_cut: f64,
    wrong_harm: f64,
    lazy: LazyMode,
    chk: f64,
    decoy: f64,
    outage: bool,
    equal: bool,
    rho: f64,
}
fn sw_base(name: &str) -> SW {
    SW { name: name.into(), emb: re(0.8), k: 12, hal_base: 0.7, hal_floor: 0.0, ctx_gain: 0.4, ctx_cut: 0.6, wrong_harm: 0.15, lazy: LazyMode::Third, chk: 1.0, decoy: 0.3, outage: true, equal: false, rho: 0.6 }
}
fn b_worlds() -> Vec<SW> {
    let b = |n: &str| sw_base(n);
    vec![
        b("aniso-n0.8"),
        SW { emb: para(1.2), ..b("para-n1.2") },
        SW { emb: drf(0.8, 1.0), ..b("drift1-n0.8") },
        SW { k: 48, ..b("k48-aniso-n0.8") },
        SW { emb: re(1.6), ..b("aniso-n1.6") },
        SW { ctx_gain: 0.0, ctx_cut: 0.0, wrong_harm: 0.0, ..b("irrelevant-notes") },
        SW { equal: true, hal_base: 0.0, ctx_gain: 0.0, ctx_cut: 0.0, wrong_harm: 0.0, lazy: LazyMode::None, decoy: 0.0, outage: false, ..b("control-equal") },
    ]
}

struct Inst {
    skill: Vec<Vec<f64>>,
    concept: Vec<usize>,
    emb: Vec<Vec<f32>>,
    tasks: Vec<Task>,
    kw: HashMap<String, usize>,
    best: Vec<f64>, // per concept: max skill
    oracle_q: f64,
}
fn build_inst(w: &SW, seed: u64, widx: u64) -> Inst {
    let s = seed.wrapping_mul(1000) + widx;
    let mut rng = Rng::new(s);
    let (emb, concept) = gen_stream(seed * 31 + widx, N_TASKS, w.k, &w.emb);
    let mut skill = vec![vec![0.0; w.k]; N_AGENTS];
    for c in 0..w.k {
        for a in 0..N_AGENTS {
            skill[a][c] = if w.equal { 0.7 } else { 0.35 + 0.25 * rng.f64() };
        }
        if !w.equal {
            for _ in 0..2 {
                let a = rng.below(N_AGENTS);
                skill[a][c] = 0.85 + 0.07 * rng.f64();
            }
        }
    }
    let mut kw = HashMap::new();
    for c in 0..w.k {
        let best = (0..N_AGENTS).fold(0, |b, a| if skill[a][c] > skill[b][c] { a } else { b });
        let adv = if w.equal || rng.f64() < w.decoy { rng.below(N_AGENTS) } else { best };
        for j in 0..3 {
            kw.insert(format!("t{c}w{j}"), adv);
        }
    }
    let best: Vec<f64> = (0..w.k).map(|c| (0..N_AGENTS).map(|a| skill[a][c]).fold(0.0, f64::max)).collect();
    let tasks: Vec<Task> = (0..N_TASKS)
        .map(|i| {
            let c = concept[i];
            let mut f: Vec<String> = (0..3).map(|j| format!("t{c}w{j}")).collect();
            f.push(format!("dom{}", c % 3));
            f.push(format!("n{}", rng.below(30)));
            Task { id: i as u64, task_type: None, features: f }
        })
        .collect();
    let oracle_q = concept.iter().map(|&c| best[c]).sum::<f64>() / N_TASKS as f64;
    Inst { skill, concept, emb, tasks, kw, best, oracle_q }
}

#[derive(Clone)]
struct Draft {
    agent: usize,
    q: f64,
    hal: bool,
    ok: bool,
    conf: f64,
}
struct Wx<'a> {
    w: &'a SW,
    inst: &'a Inst,
    seed: u64,
}
impl Wx<'_> {
    fn lazy(&self, c: usize) -> f64 {
        if self.w.lazy == LazyMode::Third && c % 3 == 2 {
            0.15
        } else {
            1.0
        }
    }
    fn draft(&self, ti: usize, agent: usize, n_pos: f64, n_neg: f64) -> Draft {
        let c = self.inst.concept[ti];
        let (s, tid) = (self.inst.skill[agent][c], ti as u64);
        let down = self.w.outage && agent == 7 && (600..1000).contains(&ti);
        let p_hal = ((self.w.hal_floor + self.w.hal_base * (1.0 - s)) * (1.0 - self.w.ctx_cut * n_pos)).clamp(0.0, 1.0);
        let hal = uni(self.seed, tid, agent as u64, 11) < p_hal;
        let q = if down {
            0.0
        } else if hal {
            0.05 + 0.2 * uni(self.seed, tid, agent as u64, 12)
        } else {
            (s + self.w.ctx_gain * n_pos * (1.0 - s) - self.w.wrong_harm * n_neg + 0.07 * nrm(self.seed, tid, agent as u64, 13)).clamp(0.0, 1.0)
        };
        let ok = !down && uni(self.seed, tid, agent as u64, 14) < 0.99 - 0.03 * (1.0 - s);
        let conf = (0.5 + self.w.rho * (q - 0.5) + 0.2 * nrm(self.seed, tid, agent as u64, 18)).clamp(0.0, 1.0);
        Draft { agent, q, hal, ok, conf }
    }
    fn check(&self, ti: usize, checker: usize, d: &Draft) -> bool {
        let c = self.inst.concept[ti];
        let sc = self.inst.skill[checker][c];
        let u = uni(self.seed, ti as u64, (checker * 16 + d.agent) as u64, 15);
        if d.q < 0.5 {
            let catch = ((0.1 + 0.88 * sc * self.w.chk).min(1.0)) * self.lazy(checker) * if d.hal || d.q == 0.0 { 1.0 } else { 0.5 };
            u < catch
        } else {
            u < (0.05 + 0.1 * (1.0 - sc)) * self.lazy(checker)
        }
    }
    fn grade(&self, ti: usize, d: &Draft) -> Option<f64> {
        if uni(self.seed, ti as u64, 0, 16) >= REVEAL {
            return None;
        }
        Some((d.q + 0.05 * nrm(self.seed, ti as u64, d.agent as u64, 17)).clamp(0.0, 1.0))
    }
}

#[derive(Clone, Debug)]
enum Retr {
    None,
    Cfg(RetCfg),
    Oracle,
}
#[derive(Clone, Debug)]
enum Crit {
    None,
    /// e3's verified HierSecondOpinion gate with this per-call cost
    Gate(f64),
    Always,
    Conf(f64),
    NoSup,
    Dis,
    /// conf < th OR no support OR retrieval disagreement
    Combo(f64),
    /// checks exactly the bad drafts (upper bound on trigger quality)
    Oracle,
}
#[derive(Clone, Debug)]
struct SCfg {
    warm_k: f64,
    reuse: bool,
    keyword_only: bool,
    ret: Retr,
    crit: Crit,
    store_min: f64,
}
impl SCfg {
    fn new(warm_k: f64, reuse: bool, ret: Retr, crit: Crit) -> Self {
        SCfg { warm_k, reuse, keyword_only: false, ret, crit, store_min: 0.65 }
    }
}

struct Note {
    concept: usize,
    s: f64,
    agent: usize,
}
struct Ev {
    due: usize,
    ti: usize,
    leader: usize,
    g_first: Option<f64>,
    g_final: Option<f64>,
    retry: Option<usize>,
    g_retry: Option<f64>,
    check: Option<(usize, bool)>,
    deliv_agent: usize,
}

const QT: [f64; 3] = [0.75, 0.80, 0.85];
const AT: [f64; 3] = [0.6, 0.7, 0.8];
struct SOut {
    q250: f64,
    q500: f64,
    q2000: f64,
    last500: f64,
    extra: f64,
    ctx_hit: f64,
    ctx_prec: f64,
    acc: f64,
    tp: f64,
    fp: f64,
    fnn: f64,
    tn: f64,
    checks: f64,
    flags: f64,
    bad_first: f64,
    bad_deliv: f64,
    o2t: [f64; 6],
    reached: [f64; 6],
    win_q: Vec<f64>,
    win_acc: Vec<f64>,
    oracle_q: f64,
    auc_conf: f64,
}

fn run_session(w: &SW, c: &SCfg, seed: u64, widx: u64) -> SOut {
    let inst = build_inst(w, seed, widx);
    let wx = Wx { w, inst: &inst, seed };
    let cp = CouplingConfig { hebbian: heb_robust_cfg(), warm_k: c.warm_k, warm_evidence: WarmEvidence::MeanFeature, health: true, ..CouplingConfig::default() };
    let mut brain = CoupledBrain::new(N_AGENTS, inst.kw.clone(), cp);
    let mut rng = Rng::new(seed * 7 + 1);
    let cands: Vec<usize> = (0..N_AGENTS).collect();
    let mut store = match &c.ret {
        Retr::Cfg(rc) => Some(Store::new(rc)),
        _ => None,
    };
    let mut notes: Vec<Note> = vec![];
    let mut gate = match &c.crit {
        Crit::Gate(cost) => Some(HierSecondOpinion::new(N_AGENTS, HierSoConfig { base: SecondOpinionConfig { cost: *cost, kappa: 0.0, discount: 0.995, ..SecondOpinionConfig::default() }, pool_leader: 8.0, ..HierSoConfig::default() })),
        _ => None,
    };
    let mut pending: std::collections::VecDeque<Ev> = std::collections::VecDeque::new();
    let mut arrived = 0.0f64;
    let mut arrived_at = vec![0.0f64; N_TASKS];
    let (mut qs, mut accs) = (vec![0.0f64; N_TASKS], vec![0.0f64; N_TASKS]);
    let (mut extra, mut hits, mut prec_sum, mut prec_n) = (0.0, 0.0, 0.0, 0.0);
    let (mut tp, mut fp, mut fnn, mut tn, mut checks, mut flags, mut bad_first, mut bad_deliv) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    // for AUROC of conf vs bad (rank sum)
    let mut conf_good: Vec<f64> = vec![];
    let mut conf_bad: Vec<f64> = vec![];
    for ti in 0..N_TASKS {
        // deliver due outcomes
        while pending.front().is_some_and(|e| e.due <= ti) {
            let ev = pending.pop_front().unwrap();
            let task = &inst.tasks[ev.ti];
            let checked = ev.check.is_some();
            let leader_g = if checked { ev.g_first } else { ev.g_final };
            if ev.g_first.is_some() || ev.g_final.is_some() {
                arrived += 1.0;
            }
            if c.reuse {
                if let Some(g) = leader_g {
                    brain.feedback_delayed(task, ev.leader, g);
                }
            }
            if let Some(gt) = gate.as_mut() {
                if let Some(g) = leader_g {
                    gt.observe_attempt(task, ev.leader, g >= 0.5);
                    if let Some((ck, fl)) = ev.check {
                        gt.observe_check(task, ck, g < 0.5, fl);
                    }
                }
                if let (Some(ra), Some(g)) = (ev.retry, ev.g_retry) {
                    gt.observe_attempt(task, ra, g >= 0.5);
                }
            }
            if let (Some(st), Some(g)) = (store.as_mut(), ev.g_final) {
                if g >= c.store_min {
                    st.add(&inst.emb[ev.ti]);
                    notes.push(Note { concept: inst.concept[ev.ti], s: g, agent: ev.deliv_agent });
                }
            } else if matches!(c.ret, Retr::Oracle) {
                if let Some(g) = ev.g_final {
                    if g >= c.store_min {
                        notes.push(Note { concept: inst.concept[ev.ti], s: g, agent: ev.deliv_agent });
                    }
                }
            }
        }
        arrived_at[ti] = arrived;
        let task = &inst.tasks[ti];
        let con = inst.concept[ti];
        // retrieval
        let mut r: Vec<(usize, f32)> = vec![];
        if let Some(st) = store.as_mut() {
            st.observe(&inst.emb[ti]);
            r = st.query(&inst.emb[ti]);
        } else if matches!(c.ret, Retr::Oracle) {
            // ceiling: the (up to 5) most recent verified notes of the TRUE concept
            let mut idx: Vec<usize> = (0..notes.len()).filter(|&i| notes[i].concept == con).collect();
            idx.reverse();
            idx.truncate(5);
            r = idx.into_iter().map(|i| (i, 1.0f32)).collect();
        }
        let (mut n_pos, mut n_neg) = (0.0, 0.0);
        let mut vote: Vec<f64> = vec![0.0; N_AGENTS];
        let mut any_vote = false;
        if !r.is_empty() {
            // store indices map to notes by insertion order: for Cfg stores both grow together
            for &(i, s) in &r {
                let nt = &notes[i];
                if nt.concept == con {
                    n_pos += nt.s;
                } else {
                    n_neg += nt.s;
                }
                vote[nt.agent] += s as f64 * nt.s;
                any_vote = true;
            }
            n_pos /= r.len() as f64;
            n_neg /= r.len() as f64;
            hits += 1.0;
            prec_sum += if n_pos + n_neg > 0.0 { n_pos / (n_pos + n_neg) } else { 0.0 };
            prec_n += 1.0;
        }
        let vote_agent = if any_vote { Some((0..N_AGENTS).fold(0, |b, a| if vote[a] > vote[b] { a } else { b })) } else { None };
        // route
        let agent = if c.keyword_only {
            task.features.iter().find_map(|f| inst.kw.get(f).copied()).unwrap_or(0)
        } else {
            brain.route(task, &cands, &mut rng)
        };
        let first = wx.draft(ti, agent, n_pos, n_neg);
        let ok_f = if first.ok { 1.0 } else { 0.0 };
        if !c.keyword_only {
            brain.feedback_split(task, agent, ok_f, if c.reuse { None } else { Some(ok_f) });
        }
        let bad = first.q < 0.5;
        if bad {
            bad_first += 1.0;
            conf_bad.push(first.conf);
        } else {
            conf_good.push(first.conf);
        }
        // critic decision
        let dis = vote_agent.is_some_and(|v| v != agent);
        let best_other = |brain: &CoupledBrain| -> usize {
            let sc = brain.router.inner().scores(task);
            let mut b = if agent == 0 { 1 } else { 0 };
            for a in 0..N_AGENTS {
                if a != agent && sc.get(a).copied().unwrap_or(0.0) > sc.get(b).copied().unwrap_or(0.0) {
                    b = a;
                }
            }
            b
        };
        let plan: Option<(usize, usize)> = match &c.crit {
            Crit::None => None,
            Crit::Gate(_) => gate.as_mut().unwrap().decide(task, agent, &cands).map(|p| (p.checker, p.retry)),
            Crit::Always => {
                let o = best_other(&brain);
                Some((o, o))
            }
            Crit::Conf(th) => (first.conf < *th).then(|| best_other(&brain)).map(|o| (o, o)),
            Crit::NoSup => r.is_empty().then(|| best_other(&brain)).map(|o| (o, o)),
            Crit::Dis => dis.then(|| best_other(&brain)).map(|o| (o, o)),
            Crit::Combo(th) => (first.conf < *th || r.is_empty() || dis).then(|| best_other(&brain)).map(|o| (o, o)),
            Crit::Oracle => bad.then(|| best_other(&brain)).map(|o| (o, o)),
        };
        let mut delivered = first.clone();
        let mut calls = 1.0;
        let (mut check, mut retry, mut retry_out): (Option<(usize, bool)>, Option<usize>, Option<Draft>) = (None, None, None);
        let mut flagged = false;
        if let Some((ck, ra)) = plan {
            checks += 1.0;
            calls += 1.0;
            flagged = wx.check(ti, ck, &first);
            check = Some((ck, flagged));
            if flagged {
                flags += 1.0;
                calls += 1.0;
                let second = wx.draft(ti, ra, n_pos, n_neg);
                delivered = second.clone();
                retry = Some(ra);
                retry_out = Some(second);
            }
        }
        match (bad, flagged) {
            (true, true) => tp += 1.0,
            (false, true) => fp += 1.0,
            (true, false) => fnn += 1.0,
            (false, false) => tn += 1.0,
        }
        if delivered.q < 0.5 {
            bad_deliv += 1.0;
        }
        extra += calls - 1.0;
        qs[ti] = delivered.q;
        let bs = inst.best[con];
        accs[ti] = if inst.skill[agent][con] >= bs - 0.05 { 1.0 } else { 0.0 };
        let g_first = wx.grade(ti, &first);
        let g_final = if retry.is_some() { wx.grade(ti, &delivered) } else { g_first };
        let g_retry = retry_out.as_ref().and_then(|o| wx.grade(ti, o));
        pending.push_back(Ev { due: ti + DELAY, ti, leader: agent, g_first, g_final, retry, g_retry, check, deliv_agent: delivered.agent });
    }
    // windows & outcomes-to-target
    let mut cq = vec![0.0f64; N_TASKS + 1];
    let mut ca = vec![0.0f64; N_TASKS + 1];
    for i in 0..N_TASKS {
        cq[i + 1] = cq[i] + qs[i];
        ca[i + 1] = ca[i] + accs[i];
    }
    let win = 100;
    let (mut o2t, mut reached) = ([0.0; 6], [0.0; 6]);
    for (k, tg) in QT.iter().chain(AT.iter()).enumerate() {
        let (cs, off) = if k < 3 { (&cq, 0) } else { (&ca, 0) };
        let _ = off;
        let mut found = None;
        for i in (win - 1)..N_TASKS {
            if (cs[i + 1] - cs[i + 1 - win]) / win as f64 >= *tg {
                found = Some(i);
                break;
            }
        }
        match found {
            Some(i) => {
                o2t[k] = arrived_at[i];
                reached[k] = 1.0;
            }
            None => o2t[k] = arrived_at[N_TASKS - 1],
        }
    }
    let win_q: Vec<f64> = (0..N_TASKS / 100).map(|b| (cq[(b + 1) * 100] - cq[b * 100]) / 100.0).collect();
    let win_acc: Vec<f64> = (0..N_TASKS / 100).map(|b| (ca[(b + 1) * 100] - ca[b * 100]) / 100.0).collect();
    // AUROC: P(conf_good > conf_bad)
    let auc = if conf_bad.is_empty() || conf_good.is_empty() {
        f64::NAN
    } else {
        let mut gs = conf_good.clone();
        gs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut s = 0.0;
        for b in &conf_bad {
            let lt = gs.partition_point(|x| x < b);
            let le = gs.partition_point(|x| x <= b);
            s += (gs.len() - le) as f64 + 0.5 * (le - lt) as f64;
        }
        s / (gs.len() as f64 * conf_bad.len() as f64)
    };
    let n = N_TASKS as f64;
    SOut {
        q250: cq[250] / 250.0,
        q500: cq[500] / 500.0,
        q2000: cq[N_TASKS] / n,
        last500: (cq[N_TASKS] - cq[N_TASKS - 500]) / 500.0,
        extra: extra / n,
        ctx_hit: hits / n,
        ctx_prec: if prec_n > 0.0 { prec_sum / prec_n } else { f64::NAN },
        acc: ca[N_TASKS] / n,
        tp,
        fp,
        fnn,
        tn,
        checks,
        flags,
        bad_first: bad_first / n,
        bad_deliv: bad_deliv / n,
        o2t,
        reached,
        win_q,
        win_acc,
        oracle_q: inst.oracle_q,
        auc_conf: auc,
    }
}

// =============================================================================================
// PART B: sample efficiency (+ downstream value of retrieval variants = part A's "what it buys")

fn part_b(held: &[u64], ap: &APart, quick: bool) -> Value {
    let t0 = Instant::now();
    let worlds = b_worlds();
    let tune: Vec<u64> = if quick { (1..=2).collect() } else { (1..=8).collect() };
    // tune warm_k for plain (raw hook bit) and for reuse arms on gross q2000 over the first 6 worlds
    let pick_k = |reuse: bool| -> f64 {
        let mut best = (f64::NEG_INFINITY, 1.0);
        for &k in &[1.0, 3.0, 10.0] {
            let cfg = SCfg::new(k, reuse, Retr::None, Crit::None);
            let jobs: Vec<(usize, u64)> = (0..6).flat_map(|w| tune.iter().map(move |&s| (w, s))).collect();
            let o = jobs.par_iter().map(|&(w, s)| run_session(&worlds[w], &cfg, s, w as u64).q2000).sum::<f64>() / jobs.len() as f64;
            if o > best.0 {
                best = (o, k);
            }
        }
        best.1
    };
    let (pk, rk) = (pick_k(false), pick_k(true));
    println!("\n[B tune] warm_k plain={pk} reuse={rk} ({:.0}s)", t0.elapsed().as_secs_f64());
    let default_ret = RetCfg::raw(0.6, 5);
    let arms: Vec<(&str, SCfg)> = vec![
        ("keyword", SCfg { keyword_only: true, ..SCfg::new(1.0, false, Retr::None, Crit::None) }),
        ("plain(warm,raw-bit)", SCfg::new(pk, false, Retr::None, Crit::None)),
        ("cold+reuse(no warm start)", SCfg::new(0.0, true, Retr::None, Crit::None)),
        ("warm+reuse", SCfg::new(rk, true, Retr::None, Crit::None)),
        ("+retr(research-default raw .6 k5)", SCfg::new(rk, true, Retr::Cfg(default_ret), Crit::None)),
        ("+retr(tuned raw)", SCfg::new(rk, true, Retr::Cfg(ap.best_raw.clone()), Crit::None)),
        ("+retr(tuned centered)", SCfg::new(rk, true, Retr::Cfg(ap.best_center.clone()), Crit::None)),
        ("+retr(best family)", SCfg::new(rk, true, Retr::Cfg(ap.best_overall.clone()), Crit::None)),
        ("+retr(oracle ceiling)", SCfg::new(rk, true, Retr::Oracle, Crit::None)),
    ];
    let jobs: Vec<(usize, usize, u64)> = (0..arms.len()).flat_map(|a| (0..worlds.len()).flat_map(move |w| held.iter().map(move |&s| (a, w, s)))).collect();
    let res: Vec<(usize, usize, SOut)> = jobs.par_iter().map(|&(a, w, s)| (a, w, run_session(&worlds[w], &arms[a].1, s, w as u64))).collect();
    let mut cells: HashMap<(usize, usize), Vec<SOut>> = HashMap::new();
    for (a, w, r) in res {
        cells.entry((a, w)).or_default().push(r);
    }
    let v = |a: usize, w: usize, f: &dyn Fn(&SOut) -> f64| -> Vec<f64> { cells[&(a, w)].iter().map(f).collect() };
    // metrics tested
    type F = Box<dyn Fn(&SOut) -> f64>;
    let metrics: Vec<(&str, F)> = vec![
        ("gross_H250", Box::new(|r: &SOut| r.q250)),
        ("gross_H500", Box::new(|r: &SOut| r.q500)),
        ("gross_H2000", Box::new(|r: &SOut| r.q2000)),
        ("outcomes_to_q0.80", Box::new(|r: &SOut| r.o2t[1])),
        ("outcomes_to_acc0.70", Box::new(|r: &SOut| r.o2t[4])),
    ];
    let comps: Vec<(&str, usize, usize)> = vec![
        ("reuse path (warm+reuse vs plain)", 3, 1),
        ("warm start (warm+reuse vs cold+reuse)", 3, 2),
        ("retrieval default vs warm+reuse", 4, 3),
        ("retrieval tuned raw vs warm+reuse", 5, 3),
        ("retrieval tuned centered vs warm+reuse", 6, 3),
        ("retrieval best family vs warm+reuse", 7, 3),
        ("retrieval best family vs default", 7, 4),
        ("retrieval tuned centered vs tuned raw", 6, 5),
        ("oracle retrieval vs best family (headroom)", 8, 7),
        ("warm+reuse vs keyword", 3, 0),
    ];
    let mut tests: Vec<(usize, usize, usize, Cmp)> = vec![];
    for w in 0..worlds.len() {
        for (ci, (_, a, b)) in comps.iter().enumerate() {
            for (mi, (_, f)) in metrics.iter().enumerate() {
                tests.push((w, ci, mi, cmp(&v(*a, w, f), &v(*b, w, f))));
            }
        }
    }
    let adj = holm_adjust(&tests.iter().map(|t| t.3.p).collect::<Vec<_>>());
    let mut tj = vec![];
    let (mut nw, mut nl) = (0, 0);
    for (k, t) in tests.iter().enumerate() {
        let sgn = if metrics[t.2].0.starts_with("outcomes") { -1.0 } else { 1.0 };
        let vd = verdict(adj[k], sgn * t.3.d);
        if vd == "WIN" {
            nw += 1
        } else if vd == "LOSS" {
            nl += 1
        }
        let mut j = cj(&t.3, adj[k]);
        j["verdict"] = json!(vd);
        j["note"] = json!(if sgn < 0.0 { "lower is better (fewer graded outcomes to target); WIN = fewer" } else { "higher is better" });
        j["world"] = json!(worlds[t.0].name);
        j["comparison"] = json!(comps[t.1].0);
        j["metric"] = json!(metrics[t.2].0);
        tj.push(j);
    }
    // tables
    let mut table = vec![];
    println!("\n[B] held-out {}-{}: gross H250 / H500 / H2000 | graded outcomes to q>=.80 (reached) | to acc>=.70 | ctx-hit, ctx precision", held[0], held[held.len() - 1]);
    for (wi, w) in worlds.iter().enumerate() {
        println!("  world {} (oracle skill-quality {:.3})", w.name, mean(&v(0, wi, &|r: &SOut| r.oracle_q)));
        let mut rows = vec![];
        for (ai, (an, _)) in arms.iter().enumerate() {
            let m = |f: &dyn Fn(&SOut) -> f64| mean(&v(ai, wi, f));
            let (q250, q500, q2000) = (m(&|r| r.q250), m(&|r| r.q500), m(&|r| r.q2000));
            let (o80, r80) = (m(&|r| r.o2t[1]), m(&|r| r.reached[1]));
            let (oa, ra) = (m(&|r| r.o2t[4]), m(&|r| r.reached[4]));
            println!("    {:<36} {:.3} {:.3} {:.3} | {:>6.0} ({:.2}) | {:>6.0} ({:.2}) | hit {:.2} prec {:.2}", an, q250, q500, q2000, o80, r80, oa, ra, m(&|r| r.ctx_hit), m(&|r| r.ctx_prec));
            let curve_q: Vec<f64> = (0..20).map(|b| mean(&cells[&(ai, wi)].iter().map(|r| r.win_q[b]).collect::<Vec<_>>())).collect();
            let curve_a: Vec<f64> = (0..20).map(|b| mean(&cells[&(ai, wi)].iter().map(|r| r.win_acc[b]).collect::<Vec<_>>())).collect();
            let mut o2t = serde_json::Map::new();
            for (k, nm) in ["q0.75", "q0.80", "q0.85", "acc0.60", "acc0.70", "acc0.80"].iter().enumerate() {
                let xs = v(ai, wi, &|r: &SOut| r.o2t[k]);
                let rs = v(ai, wi, &|r: &SOut| r.reached[k]);
                let s = neural_mesh::stats::summarize(&xs);
                o2t.insert(nm.to_string(), json!({"mean": s.mean, "lo": s.lo, "hi": s.hi, "reached_frac": mean(&rs)}));
            }
            rows.push(json!({"arm": an, "gross_H250": q250, "gross_H500": q500, "gross_H2000": q2000, "last500": m(&|r| r.last500), "acc": m(&|r| r.acc), "ctx_hit": m(&|r| r.ctx_hit), "ctx_precision": m(&|r| r.ctx_prec), "outcomes_to_target": Value::Object(o2t), "curve_quality_per_100tasks": curve_q, "curve_acc_per_100tasks": curve_a}));
        }
        table.push(json!({"world": w.name, "oracle_skill_quality": mean(&v(0, wi, &|r: &SOut| r.oracle_q)), "arms": rows}));
    }
    println!("\n[B] Holm family {} tests: WIN {nw} LOSS {nl}. Selected (gross_H2000 and outcomes_to_q0.80):", tests.len());
    for (k, t) in tests.iter().enumerate() {
        let mn = metrics[t.2].0;
        let sgn = if mn.starts_with("outcomes") { -1.0 } else { 1.0 };
        if (mn == "gross_H2000" || mn == "outcomes_to_q0.80") && verdict(adj[k], sgn * t.3.d) != "n.s." {
            println!("   {:<16} {:<44} {:<18} {:+.4} [{:+.4},{:+.4}] {}", worlds[t.0].name, comps[t.1].0, mn, t.3.d, t.3.lo, t.3.hi, verdict(adj[k], sgn * t.3.d));
        }
    }
    json!({"tune_seeds": format!("{}-{}", tune[0], tune[tune.len()-1]), "eval_seeds": format!("{}-{}", held[0], held[held.len()-1]),
        "warm_k_plain": pk, "warm_k_reuse": rk, "retr_cfgs": {"research_default": "R-thr0.60k5", "tuned_raw": ap.best_raw.label(), "tuned_centered": ap.best_center.label(), "best_family": format!("{} {}", ap.best_overall_family, ap.best_overall.label())},
        "targets": {"quality": QT, "accuracy": AT, "window_tasks": 100, "outcomes": "graded outcomes arrived (50% reveal, delay 20); censored at total arrived (value = cap, reached_frac gives the share that reached)"},
        "table": table, "tests": tj, "holm_win": nw, "holm_loss": nl, "runtime_s": t0.elapsed().as_secs_f64()})
}

// =============================================================================================
// PART C: generator -> critic

#[derive(Clone, Debug)]
struct Cell {
    name: String,
    w: SW,
}
fn c_cells() -> Vec<Cell> {
    let mut v = vec![];
    for &hf in &[0.0, 0.15, 0.30] {
        for &chk in &[1.0, 0.5] {
            for &rho in &[0.0, 0.2, 0.35, 0.6, 1.0] {
                let name = format!("floor{hf:.2}-chk{chk:.1}-rho{rho:.1}");
                v.push(Cell { name: name.clone(), w: SW { hal_floor: hf, chk, rho, ..sw_base(&name) } });
            }
        }
    }
    let n = "ctl-no-errors-equal";
    v.push(Cell { name: n.into(), w: SW { equal: true, hal_base: 0.0, ctx_gain: 0.0, ctx_cut: 0.0, wrong_harm: 0.0, lazy: LazyMode::None, decoy: 0.0, outage: false, ..sw_base(n) } });
    v
}
const COSTS: [f64; 3] = [0.01, 0.03, 0.10];
const THS: [f64; 7] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];

#[derive(Clone, Debug)]
struct CArm {
    name: String,
    fam: &'static str,
    crit: Crit,
    cost_idx: Option<usize>, // Gate arms are tied to a cost
}
fn c_arms() -> Vec<CArm> {
    let mut a = vec![CArm { name: "none".into(), fam: "none", crit: Crit::None, cost_idx: None }];
    for (i, c) in COSTS.iter().enumerate() {
        a.push(CArm { name: format!("e3-gate(cost{c})"), fam: "gate", crit: Crit::Gate(*c), cost_idx: Some(i) });
    }
    a.push(CArm { name: "always".into(), fam: "always", crit: Crit::Always, cost_idx: None });
    for th in THS {
        a.push(CArm { name: format!("conf<{th}"), fam: "conf", crit: Crit::Conf(th), cost_idx: None });
    }
    a.push(CArm { name: "no-support".into(), fam: "nosup", crit: Crit::NoSup, cost_idx: None });
    a.push(CArm { name: "disagree".into(), fam: "dis", crit: Crit::Dis, cost_idx: None });
    for th in THS {
        a.push(CArm { name: format!("combo<{th}"), fam: "combo", crit: Crit::Combo(th), cost_idx: None });
    }
    a.push(CArm { name: "oracle-trigger".into(), fam: "oracle", crit: Crit::Oracle, cost_idx: None });
    a
}

fn part_c(held: &[u64], ap: &APart, quick: bool) -> Value {
    let t0 = Instant::now();
    let mut cells = c_cells();
    if quick {
        cells.truncate(3);
    }
    let arms = c_arms();
    let tune: Vec<u64> = if quick { (1..=2).collect() } else { (1..=8).collect() };
    let retr = Retr::Cfg(ap.best_overall.clone());
    let mk = |a: &CArm| SCfg::new(1.0, true, retr.clone(), a.crit.clone());
    // run every arm on tune + held seeds
    let all_seeds: Vec<u64> = tune.iter().chain(held.iter()).copied().collect();
    let jobs: Vec<(usize, usize, u64)> = { let mut j = vec![]; for c in 0..cells.len() { for a in 0..arms.len() { for &s in &all_seeds { j.push((c, a, s)); } } } j };
    let res: Vec<(usize, usize, u64, SOut)> = jobs.par_iter().map(|&(c, a, s)| (c, a, s, run_session(&cells[c].w, &mk(&arms[a]), s, 100 + c as u64))).collect();
    let mut m: HashMap<(usize, usize), Vec<(u64, SOut)>> = HashMap::new();
    for (c, a, s, r) in res {
        m.entry((c, a)).or_default().push((s, r));
    }
    for v in m.values_mut() {
        v.sort_by_key(|x| x.0);
    }
    let net = |r: &SOut, cost: f64| r.q2000 - cost * r.extra;
    let sel = |c: usize, a: usize, set: &[u64], f: &dyn Fn(&SOut) -> f64| -> Vec<f64> { m[&(c, a)].iter().filter(|x| set.contains(&x.0)).map(|x| f(&x.1)).collect() };
    println!("\n[C] {} cells x {} arms x {} seeds ({:.0}s)", cells.len(), arms.len(), all_seeds.len(), t0.elapsed().as_secs_f64());

    // ---- per cost: choose theta per family (global over cells: deployable; per cell: optimistic) on TUNE seeds
    let fams = ["conf", "combo"];
    let mut jout = vec![];
    let mut tests: Vec<(usize, usize, String, Cmp)> = vec![]; // cell*10+cost, arm index, kind, cmp
    #[derive(Clone)]
    struct Pick {
        ci: usize,
        cost_i: usize,
        label: String,
        arm: usize,
    }
    let mut picks: Vec<Pick> = vec![];
    for (ki, cost) in COSTS.iter().enumerate() {
        // global theta per family
        let mut glob: HashMap<&str, usize> = HashMap::new();
        for f in fams {
            let mut best = (f64::NEG_INFINITY, 0usize);
            for (ai, a) in arms.iter().enumerate().filter(|(_, a)| a.fam == f) {
                let o: f64 = (0..cells.len()).map(|c| mean(&sel(c, ai, &tune, &|r| net(r, *cost)))).sum::<f64>();
                if o > best.0 {
                    best = (o, ai);
                }
            }
            glob.insert(f, best.1);
        }
        println!("  cost {cost}: global theta conf -> {}, combo -> {}", arms[glob["conf"]].name, arms[glob["combo"]].name);
        for ci in 0..cells.len() {
            let gate_i = arms.iter().position(|a| a.cost_idx == Some(ki)).unwrap();
            let none_i = 0;
            let always_i = arms.iter().position(|a| a.fam == "always").unwrap();
            let nosup_i = arms.iter().position(|a| a.fam == "nosup").unwrap();
            let dis_i = arms.iter().position(|a| a.fam == "dis").unwrap();
            let orc_i = arms.iter().position(|a| a.fam == "oracle").unwrap();
            let mut cell_best: HashMap<&str, usize> = HashMap::new();
            for f in fams {
                let mut best = (f64::NEG_INFINITY, 0usize);
                for (ai, _) in arms.iter().enumerate().filter(|(_, a)| a.fam == f) {
                    let o = mean(&sel(ci, ai, &tune, &|r| net(r, *cost)));
                    if o > best.0 {
                        best = (o, ai);
                    }
                }
                cell_best.insert(f, best.1);
            }
            let cands: Vec<(String, usize)> = vec![
                ("e3-gate".into(), gate_i),
                ("always".into(), always_i),
                ("conf(global theta)".into(), glob["conf"]),
                ("conf(per-cell theta)".into(), cell_best["conf"]),
                ("no-support".into(), nosup_i),
                ("disagree".into(), dis_i),
                ("combo(global theta)".into(), glob["combo"]),
                ("combo(per-cell theta)".into(), cell_best["combo"]),
                ("oracle-trigger".into(), orc_i),
            ];
            for (lab, ai) in cands {
                picks.push(Pick { ci, cost_i: ki, label: lab.clone(), arm: ai });
                let a = sel(ci, ai, held, &|r| net(r, *cost));
                let b = sel(ci, none_i, held, &|r| net(r, *cost));
                tests.push((ci * 10 + ki, picks.len() - 1, "net".into(), cmp(&a, &b)));
                let ag = sel(ci, ai, held, &|r| r.q2000);
                let bg = sel(ci, none_i, held, &|r| r.q2000);
                tests.push((ci * 10 + ki, picks.len() - 1, "gross".into(), cmp(&ag, &bg)));
            }
        }
    }
    let adj = holm_adjust(&tests.iter().map(|t| t.3.p).collect::<Vec<_>>());
    let (mut nw, mut nl) = (0, 0);
    let mut pays: HashMap<(usize, usize), Vec<String>> = HashMap::new(); // (cell,cost) -> arms with Holm net WIN
    let mut tj = vec![];
    for (k, t) in tests.iter().enumerate() {
        let vd = verdict(adj[k], t.3.d);
        let p = &picks[t.1];
        if t.2 == "net" {
            if vd == "WIN" {
                nw += 1;
                pays.entry((p.ci, p.cost_i)).or_default().push(p.label.clone());
            } else if vd == "LOSS" {
                nl += 1;
            }
        }
        let a = &m[&(p.ci, p.arm)];
        let held_o: Vec<&SOut> = a.iter().filter(|x| held.contains(&x.0)).map(|x| &x.1).collect();
        let sumf = |f: &dyn Fn(&SOut) -> f64| held_o.iter().map(|r| f(r)).sum::<f64>();
        let (tp, fp, fnn, tn, checks, flags) = (sumf(&|r| r.tp), sumf(&|r| r.fp), sumf(&|r| r.fnn), sumf(&|r| r.tn), sumf(&|r| r.checks), sumf(&|r| r.flags));
        let mut j = cj(&t.3, adj[k]);
        j["cell"] = json!(cells[p.ci].name);
        j["cost"] = json!(COSTS[p.cost_i]);
        j["critic"] = json!(p.label);
        j["arm_cfg"] = json!(arms[p.arm].name);
        j["metric"] = json!(t.2);
        j["extra_calls_per_task"] = json!(held_o.iter().map(|r| r.extra).sum::<f64>() / held_o.len() as f64);
        j["gate_catch_recall"] = json!(if tp + fnn > 0.0 { tp / (tp + fnn) } else { f64::NAN });
        j["gate_flag_precision"] = json!(if tp + fp > 0.0 { tp / (tp + fp) } else { f64::NAN });
        j["false_alarm_rate"] = json!(if fp + tn > 0.0 { fp / (fp + tn) } else { f64::NAN });
        j["check_rate"] = json!(checks / (held_o.len() as f64 * N_TASKS as f64));
        j["flag_rate"] = json!(flags / (held_o.len() as f64 * N_TASKS as f64));
        j["bad_draft_rate_first"] = json!(held_o.iter().map(|r| r.bad_first).sum::<f64>() / held_o.len() as f64);
        j["bad_draft_rate_delivered"] = json!(held_o.iter().map(|r| r.bad_deliv).sum::<f64>() / held_o.len() as f64);
        j["conf_auroc"] = json!(held_o.iter().map(|r| r.auc_conf).filter(|x| x.is_finite()).sum::<f64>() / held_o.iter().filter(|r| r.auc_conf.is_finite()).count().max(1) as f64);
        tj.push(j);
    }
    // print a compact verdict map: for each cost, each cell: best net diff among deployable (global theta/gate/disagree/nosup) critics
    println!("\n[C] net-of-cost value of a critic vs NO critic (same retrieval+reuse), held-out {}-{}; Holm family {} tests: net WIN {nw} LOSS {nl}", held[0], held[held.len() - 1], tests.len());
    for (ki, cost) in COSTS.iter().enumerate() {
        println!("  call cost {cost}:");
        println!("    {:<26} {:>7} | net diff of: e3-gate | conf(glob) | combo(glob) | disagree | no-support | oracle-trigger   (+ = pays, * = Holm WIN, ! = Holm LOSS)", "cell", "bad%");
        for (ci, cell) in cells.iter().enumerate() {
            let bad = mean(&sel(ci, 0, held, &|r| r.bad_first));
            let get = |lab: &str| -> String {
                let k = tests.iter().enumerate().find(|(_, t)| t.2 == "net" && t.0 == ci * 10 + ki && picks[t.1].label == lab).map(|(k, _)| k).unwrap();
                let t = &tests[k];
                let mk = match verdict(adj[k], t.3.d) {
                    "WIN" => "*",
                    "LOSS" => "!",
                    _ => " ",
                };
                format!("{:+.4}{}", t.3.d, mk)
            };
            println!("    {:<26} {:>6.1}% | {} | {} | {} | {} | {} | {}", cell.name, bad * 100.0, get("e3-gate"), get("conf(global theta)"), get("combo(global theta)"), get("disagree"), get("no-support"), get("oracle-trigger"));
        }
    }
    let n_pay_cells: usize = pays.len();
    println!("[C] (cell,cost) pairs where some critic arm is a Holm net WIN: {n_pay_cells} of {}", cells.len() * COSTS.len());
    let mut pays_list = vec![];
    for ((ci, ki), v) in &pays {
        pays_list.push(json!({"cell": cells[*ci].name, "cost": COSTS[*ki], "winning_critics": v}));
    }
    jout.push(json!({"tests": tj}));
    json!({"tune_seeds": format!("{}-{}", tune[0], tune[tune.len()-1]), "eval_seeds": format!("{}-{}", held[0], held[held.len()-1]),
        "retrieval_used": ap.best_overall.label(), "thetas": THS, "costs": COSTS,
        "cells": cells.iter().map(|c| json!({"name": c.name, "hal_floor": c.w.hal_floor, "checker_scale": c.w.chk, "conf_rho": c.w.rho})).collect::<Vec<_>>(),
        "net_wins": nw, "net_losses": nl, "cells_cost_with_a_net_win": pays_list, "results": jout, "runtime_s": t0.elapsed().as_secs_f64()})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let out_path = arg("--out").unwrap_or("results/f4_levers.json".into());
    let part = arg("--part").unwrap_or("all".into());
    let quick = args.iter().any(|a| a == "--quick");
    let held: Vec<u64> = match arg("--held") {
        Some(r) => {
            let (a, b) = r.split_once('-').expect("--held a-b");
            (a.parse::<u64>().unwrap()..=b.parse::<u64>().unwrap()).collect()
        }
        None => (31..=60).collect(),
    };
    let held: Vec<u64> = if quick { held.into_iter().take(4).collect() } else { held };
    let t0 = Instant::now();
    println!("f4_levers: part {part}, tune seeds 1-8, eval {}-{}{}", held[0], held[held.len() - 1], if quick { " (QUICK)" } else { "" });
    let ap = part_a(&held, quick);
    println!("\n[freeze] retrieval configs: tuned raw = {}, tuned centered = {}, best family = {} {}", ap.best_raw.label(), ap.best_center.label(), ap.best_overall_family, ap.best_overall.label());
    let mut out = json!({"experiment": "f4_levers", "tasks": N_TASKS, "agents": N_AGENTS, "dim": DIM, "reveal": REVEAL, "delay": DELAY, "a_retrieval": ap.json.clone()});
    if part == "all" || part == "b" {
        out["b_sample_efficiency"] = part_b(&held, &ap, quick);
    }
    if part == "all" || part == "c" {
        out["c_critic"] = part_c(&held, &ap, quick);
    }
    out["runtime_s"] = json!(t0.elapsed().as_secs_f64());
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("\nwrote {out_path}  total runtime {:.1}s", t0.elapsed().as_secs_f64());
}
