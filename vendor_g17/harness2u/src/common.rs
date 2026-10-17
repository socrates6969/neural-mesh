//! H15 shared helpers: data, cosine distance, build + structure analysis against the experiment hnsw_rs copy.
#![allow(dead_code)]
use hnsw_exp::hnsw::{set_level_seed, Hnsw};
use hnsw_exp::prelude::Distance;

pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn unif(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    pub fn gauss(&mut self) -> f32 {
        let (u, v) = (self.unif(), self.unif());
        ((-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()) as f32
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}
pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
#[derive(Clone, Copy)]
pub struct Cos;
impl Distance<f32> for Cos {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        (1.0 - dot(a, b)).max(0.0)
    }
}

/// mode: iid | clust (N/100 centres) | dupP (iid with P percent exact duplicates of earlier points) | lowK (iid on a K-dim subspace)
pub fn gen_data(mode: &str, n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng(seed.wrapping_mul(0xD1B54A32D192ED03) ^ 0x1234_5678);
    match mode {
        "clust" => {
            let k = (n / 100).max(8);
            let centres: Vec<Vec<f32>> = (0..k)
                .map(|_| {
                    let mut c: Vec<f32> = (0..dim).map(|_| r.gauss()).collect();
                    normalize(&mut c);
                    c
                })
                .collect();
            (0..n)
                .map(|_| {
                    let c = &centres[r.below(k)];
                    let mut v: Vec<f32> = c.iter().map(|x| x + 0.7 * r.gauss() / (dim as f32).sqrt()).collect();
                    normalize(&mut v);
                    v
                })
                .collect()
        }
        m if m.starts_with("dup") => {
            let frac: f64 = m[3..].parse::<f64>().unwrap_or(5.0) / 100.0;
            let mut out: Vec<Vec<f32>> = Vec::with_capacity(n);
            for i in 0..n {
                if i > 0 && r.unif() < frac {
                    let j = r.below(i);
                    let c = out[j].clone();
                    out.push(c);
                } else {
                    let mut v: Vec<f32> = (0..dim).map(|_| r.gauss()).collect();
                    normalize(&mut v);
                    out.push(v);
                }
            }
            out
        }
        m if m.starts_with("low") => {
            let ld: usize = m[3..].parse::<usize>().unwrap_or(3).min(dim);
            (0..n)
                .map(|_| {
                    let mut v = vec![0f32; dim];
                    for x in v.iter_mut().take(ld) {
                        *x = r.gauss();
                    }
                    normalize(&mut v);
                    v
                })
                .collect()
        }
        _ => (0..n)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| r.gauss()).collect();
                normalize(&mut v);
                v
            })
            .collect(),
    }
}

pub fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}
pub fn arg_s(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned().unwrap_or(default.to_string())
}
pub fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

pub fn level_seed_for(seed: u64) -> u64 {
    seed.wrapping_mul(0x9E3779B97F4A7C15) | 1
}

pub struct Cfg {
    pub m: usize,
    pub efc: usize,
    pub keep_pruned: bool,
    pub extend: bool,
    pub level_scale: f64, // 1.0 = default
}

pub fn build(data: &[Vec<f32>], cfg: &Cfg, seed: u64) -> Hnsw<'static, f32, Cos> {
    set_level_seed(level_seed_for(seed));
    let mut h: Hnsw<'static, f32, Cos> = Hnsw::new(cfg.m, data.len().max(1), 16, cfg.efc, Cos);
    h.set_keeping_pruned(cfg.keep_pruned);
    h.set_extend_candidates(cfg.extend);
    if (cfg.level_scale - 1.0).abs() > 1e-12 {
        h.modify_level_scale(cfg.level_scale);
    }
    for (i, v) in data.iter().enumerate() {
        h.insert_slice((v.as_slice(), i));
    }
    h
}

pub struct Structure {
    pub level: Vec<u8>,
    pub adj0: Vec<Vec<usize>>,
    pub indeg0: Vec<u32>,
    pub entry: usize,
    pub maxl: usize,
    pub reach0: Vec<bool>,
    pub n_reach0: usize,
}

pub fn structure(h: &Hnsw<'static, f32, Cos>, n: usize) -> Structure {
    let pi = h.get_point_indexation();
    let maxl = h.get_max_level_observed() as usize;
    let mut level = vec![0u8; n];
    let mut adj0: Vec<Vec<usize>> = vec![vec![]; n];
    let mut indeg0 = vec![0u32; n];
    for l in 0..=maxl {
        for p in pi.get_layer_iterator(l) {
            let id = p.get_origin_id();
            level[id] = p.get_point_id().0;
            for x in &p.get_neighborhood_id()[0] {
                adj0[id].push(x.d_id);
                indeg0[x.d_id] += 1;
            }
        }
    }
    let entry = pi.get_layer_iterator(maxl).next().map(|p| p.get_origin_id()).unwrap_or(0);
    let mut seen = vec![false; n];
    let mut st = vec![entry];
    seen[entry] = true;
    let mut cnt = 1;
    while let Some(u) = st.pop() {
        for &w in &adj0[u] {
            if !seen[w] {
                seen[w] = true;
                cnt += 1;
                st.push(w);
            }
        }
    }
    Structure { level, adj0, indeg0, entry, maxl, reach0: seen, n_reach0: cnt }
}

pub fn self_miss(h: &Hnsw<'static, f32, Cos>, data: &[Vec<f32>], ef: usize) -> Vec<usize> {
    (0..data.len()).filter(|&i| h.search(&data[i], 1, ef).first().map(|x| x.d_id) != Some(i)).collect()
}

pub fn build_ex(data: &[Vec<f32>], cfg: &Cfg, seed: u64, forced: &[usize], fix: u8) -> Hnsw<'static, f32, Cos> {
    hnsw_exp::hnsw::FIX_MODE.store(fix, std::sync::atomic::Ordering::Relaxed);
    hnsw_exp::hnsw::set_forced_levels(forced.to_vec());
    let h = build(data, cfg, seed);
    hnsw_exp::hnsw::set_forced_levels(vec![]);
    h
}

pub fn weak_components(st: &Structure) -> usize {
    let n = st.adj0.len();
    let mut p: Vec<usize> = (0..n).collect();
    fn find(p: &mut Vec<usize>, x: usize) -> usize {
        let mut r = x;
        while p[r] != r {
            r = p[r];
        }
        let mut c = x;
        while p[c] != r {
            let nx = p[c];
            p[c] = r;
            c = nx;
        }
        r
    }
    for u in 0..n {
        for &w in &st.adj0[u] {
            let (a, b) = (find(&mut p, u), find(&mut p, w));
            if a != b {
                p[a] = b;
            }
        }
    }
    (0..n).filter(|&x| find(&mut p, x) == x).count()
}

/// Wilson 95 % interval (lo, hi) for k successes of n.
pub fn wilson(k: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let z = 1.959964f64;
    let (k, n) = (k as f64, n as f64);
    let p = k / n;
    let d = 1.0 + z * z / n;
    let c = p + z * z / (2.0 * n);
    let h = z * ((p * (1.0 - p) + z * z / (4.0 * n)) / n).sqrt();
    (((c - h) / d).max(0.0), ((c + h) / d).min(1.0))
}

/// parallel_insert_slice of all points in one call (rayon, RAYON_NUM_THREADS applies).
pub fn build_par(data: &[Vec<f32>], cfg: &Cfg, seed: u64, forced: &[usize], fix: u8) -> Hnsw<'static, f32, Cos> {
    hnsw_exp::hnsw::FIX_MODE.store(fix, std::sync::atomic::Ordering::Relaxed);
    hnsw_exp::hnsw::set_level_seed(level_seed_for(seed));
    hnsw_exp::hnsw::set_forced_levels(forced.to_vec());
    let mut h: Hnsw<'static, f32, Cos> = Hnsw::new(cfg.m, data.len().max(1), 16, cfg.efc, Cos);
    h.set_keeping_pruned(cfg.keep_pruned);
    h.set_extend_candidates(cfg.extend);
    hnsw_exp::hnsw::set_forced_levels(vec![]);
    let v: Vec<(&[f32], usize)> = data.iter().enumerate().map(|(i, x)| (x.as_slice(), i)).collect();
    h.parallel_insert_slice(&v);
    h
}
