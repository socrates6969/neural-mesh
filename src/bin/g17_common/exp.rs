//! G17 shared experiment code (hnsw_rs reverse-edge check). Included with
//! `#[path]` from src/bin/g17_hnsw.rs and vendor_g17/harness/src/main.rs.
//! `define_exp!(krate, modname)` instantiates the experiment against one hnsw_rs build.

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
}

pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= n);
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// mode "iid": unit Gaussian vectors. mode "clust": N/100 centres, point = normalize(centre + 0.7*noise) (embedding-like).
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
                    let c = &centres[(r.next_u64() % k as u64) as usize];
                    let mut v: Vec<f32> = c.iter().map(|x| x + 0.7 * r.gauss() / (dim as f32).sqrt()).collect();
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

/// Copy of v with tiny relative noise (1e-3 per coordinate / sqrt(dim)), renormalised.
pub fn jitter(v: &[f32], salt: u64) -> Vec<f32> {
    let mut r = Rng(salt.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xABCDEF);
    let d = (v.len() as f32).sqrt();
    let mut o: Vec<f32> = v.iter().map(|x| x + 1e-3 * r.gauss() / d).collect();
    normalize(&mut o);
    o
}

#[derive(Default, Clone, Debug)]
pub struct BuildStats {
    pub n: usize,
    pub n_lvl1: usize,            // points with level >= 1
    pub in0_total: usize,         // points with zero layer-0 in-edges
    pub in0_lvl0: usize,          // ... of which level 0
    pub in0_lvl1: usize,          // ... of which level >= 1
    pub unreach0: usize,          // not reachable from entry point along layer-0 directed edges
    pub self_miss_ef512: usize,   // top-1 (ef=512) is not the point itself
    pub self_absent_ef512: usize, // own id not in top-10 at ef=512
    pub self_miss_ef64: usize,    // top-1 (ef=64, neural-mesh default) is not itself
    pub self_miss_lvl1_ef512: usize,
    pub self_miss_lvl0_ef512: usize,
    pub recall10: [f64; 3], // ef 50/100/200 vs brute force on fresh queries
    pub build_s: f64,
}

#[macro_export]
macro_rules! define_exp {
    ($krate:ident, $modname:ident) => {
        pub mod $modname {
            use super::{dot, BuildStats};
            use $krate::prelude::{Distance, Hnsw};

            #[derive(Default, Clone, Copy)]
            pub struct CosineDist;
            impl Distance<f32> for CosineDist {
                #[inline]
                fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
                    (1.0 - dot(va, vb)).max(0.0)
                }
            }

            /// `queries` fresh query vectors with brute-force top-10 truth `truth` (ids).
            pub fn run_one(
                data: &[Vec<f32>],
                m: usize,
                efc: usize,
                queries: &[Vec<f32>],
                truth: &[Vec<usize>],
            ) -> BuildStats {
                let n = data.len();
                let h: Hnsw<'static, f32, CosineDist> = Hnsw::new(m, n.max(1), 16, efc, CosineDist);
                let t0 = std::time::Instant::now();
                for (i, v) in data.iter().enumerate() {
                    h.insert_slice((v.as_slice(), i)); // same call as neural-mesh HnswIndex::add
                }
                let build_s = t0.elapsed().as_secs_f64();
                let mut st = BuildStats { n, build_s, ..Default::default() };
                // structure: levels, layer-0 adjacency
                let pi = h.get_point_indexation();
                let mut level = vec![0u8; n];
                let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
                let mut indeg = vec![0u32; n];
                for p in (0..=(h.get_max_level_observed() as usize)).flat_map(|l| pi.get_layer_iterator(l).collect::<Vec<_>>()) {
                    let id = p.get_origin_id();
                    level[id] = p.get_point_id().0;
                    let nb = p.get_neighborhood_id();
                    for x in &nb[0] {
                        adj[id].push(x.d_id as u32);
                        indeg[x.d_id] += 1;
                    }
                }
                for i in 0..n {
                    if level[i] >= 1 {
                        st.n_lvl1 += 1;
                    }
                    if indeg[i] == 0 {
                        st.in0_total += 1;
                        if level[i] >= 1 {
                            st.in0_lvl1 += 1
                        } else {
                            st.in0_lvl0 += 1
                        }
                    }
                }
                // BFS from entry point (a highest-level point) over layer 0
                let maxl = h.get_max_level_observed() as usize;
                let entry = pi.get_layer_iterator(maxl).next().map(|p| p.get_origin_id()).unwrap_or(0);
                let mut seen = vec![false; n];
                let mut stack = vec![entry];
                seen[entry] = true;
                let mut cnt = 1usize;
                while let Some(u) = stack.pop() {
                    for &w in &adj[u] {
                        let w = w as usize;
                        if !seen[w] {
                            seen[w] = true;
                            cnt += 1;
                            stack.push(w);
                        }
                    }
                }
                st.unreach0 = n - cnt;
                // self-recall
                for i in 0..n {
                    let r = h.search(&data[i], 10, 512);
                    let top1_ok = r.first().map(|x| x.d_id == i).unwrap_or(false);
                    if !top1_ok {
                        st.self_miss_ef512 += 1;
                        if level[i] >= 1 {
                            st.self_miss_lvl1_ef512 += 1
                        } else {
                            st.self_miss_lvl0_ef512 += 1
                        }
                        if !r.iter().any(|x| x.d_id == i) {
                            st.self_absent_ef512 += 1;
                        }
                    }
                    let r64 = h.search(&data[i], 1, 64);
                    if !r64.first().map(|x| x.d_id == i).unwrap_or(false) {
                        st.self_miss_ef64 += 1;
                    }
                }
                // recall@10 on fresh queries
                for (j, ef) in [50usize, 100, 200].iter().enumerate() {
                    let mut hit = 0usize;
                    for (qi, q) in queries.iter().enumerate() {
                        let r = h.search(q, 10, *ef);
                        hit += r.iter().filter(|x| truth[qi].contains(&x.d_id)).count();
                    }
                    st.recall10[j] = hit as f64 / (queries.len() * 10) as f64;
                }
                st
            }

            /// Workarounds on the UNMODIFIED crate. wk: "dup" (insert every vector twice, dedupe results by id),
            /// "flat02"/"flat05" (Hnsw::modify_level_scale at construction), "reins" (after the build,
            /// re-insert every level>=1 point that has zero layer-0 in-edges, one pass), "none".
            pub fn run_wk(
                wk: &str,
                data: &[Vec<f32>],
                m: usize,
                efc: usize,
                queries: &[Vec<f32>],
                truth: &[Vec<usize>],
            ) -> BuildStats {
                let n = data.len();
                let mut h: Hnsw<'static, f32, CosineDist> = Hnsw::new(m, n.max(1) * 2, 16, efc, CosineDist);
                match wk {
                    "flat02" => h.modify_level_scale(0.2),
                    "flat05" => h.modify_level_scale(0.5),
                    _ => {}
                }
                let t0 = std::time::Instant::now();
                for (i, v) in data.iter().enumerate() {
                    h.insert_slice((v.as_slice(), i));
                    if wk == "dup" {
                        h.insert_slice((v.as_slice(), i));
                    }
                    if wk == "djit" {
                        let jv = super::jitter(v, i as u64);
                        h.insert_slice((jv.as_slice(), i));
                    }
                }
                let mut st = BuildStats { n, ..Default::default() };
                for pass in 0..(if wk == "reins" || wk == "rjit" { 3 } else { 0 }) {
                    let pi = h.get_point_indexation();
                    let mut indeg = vec![0u32; n];
                    let mut hi: Vec<usize> = Vec::new();
                    for l in 0..=(h.get_max_level_observed() as usize) {
                        for p in pi.get_layer_iterator(l) {
                            if l >= 1 {
                                hi.push(p.get_origin_id());
                            }
                            if l == 0 {
                                for x in &p.get_neighborhood_id()[0] {
                                    indeg[x.d_id] += 1;
                                }
                            }
                        }
                    }
                    // level-1+ points also keep layer-0 lists: count them as well
                    for l in 1..=(h.get_max_level_observed() as usize) {
                        for p in pi.get_layer_iterator(l) {
                            for x in &p.get_neighborhood_id()[0] {
                                indeg[x.d_id] += 1;
                            }
                        }
                    }
                    let lost: Vec<usize> = hi.into_iter().filter(|&i| indeg[i] == 0).collect();
                    st.in0_lvl1 += lost.len();
                    for &i in &lost {
                        if wk == "rjit" {
                            let jv = super::jitter(&data[i], (i + 7919 * (pass + 1)) as u64);
                            h.insert_slice((jv.as_slice(), i));
                        } else {
                            h.insert_slice((data[i].as_slice(), i));
                        }
                    }
                    if lost.is_empty() {
                        break;
                    }
                }
                st.build_s = t0.elapsed().as_secs_f64();
                let dedupe = |r: Vec<$krate::hnsw::Neighbour>, k: usize| -> Vec<usize> {
                    let mut out: Vec<usize> = Vec::new();
                    for x in r {
                        if !out.contains(&x.d_id) {
                            out.push(x.d_id);
                        }
                        if out.len() >= k {
                            break;
                        }
                    }
                    out
                };
                for i in 0..n {
                    let r = dedupe(h.search(&data[i], 20, 512), 10);
                    if r.first() != Some(&i) {
                        st.self_miss_ef512 += 1;
                        if !r.contains(&i) {
                            st.self_absent_ef512 += 1;
                        }
                    }
                    let r64 = dedupe(h.search(&data[i], 2, 64), 1);
                    if r64.first() != Some(&i) {
                        st.self_miss_ef64 += 1;
                    }
                }
                for (j, ef) in [50usize, 100, 200].iter().enumerate() {
                    let mut hit = 0usize;
                    for (qi, q) in queries.iter().enumerate() {
                        let r = dedupe(h.search(q, 20, *ef), 10);
                        hit += r.iter().filter(|x| truth[qi].contains(x)).count();
                    }
                    st.recall10[j] = hit as f64 / (queries.len() * 10) as f64;
                }
                // query latency at ef=100, 10 results (us/query, single thread)
                let t1 = std::time::Instant::now();
                for q in queries {
                    let _ = h.search(q, 10, 100);
                }
                st.unreach0 = (t1.elapsed().as_secs_f64() * 1e6 / queries.len() as f64) as usize; // reused field: us/query
                st
            }
        }
    };
}

pub fn brute_truth(data: &[Vec<f32>], queries: &[Vec<f32>]) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|q| {
            let mut s: Vec<(f32, usize)> = data.iter().enumerate().map(|(i, v)| (dot(q, v), i)).collect();
            s.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then(a.1.cmp(&b.1)));
            s.iter().take(10).map(|x| x.1).collect()
        })
        .collect()
}

pub fn fmt_header() -> &'static str {
    "variant,mode,n,dim,seed,n_lvl1,in0_total,in0_lvl0,in0_lvl1,unreach0,self_miss512,self_absent512,self_miss512_lvl1,self_miss512_lvl0,self_miss_ef64,r10_ef50,r10_ef100,r10_ef200,build_s"
}

pub fn fmt_row(variant: &str, mode: &str, dim: usize, seed: u64, s: &BuildStats) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.1}",
        variant,
        mode,
        s.n,
        dim,
        seed,
        s.n_lvl1,
        s.in0_total,
        s.in0_lvl0,
        s.in0_lvl1,
        s.unreach0,
        s.self_miss_ef512,
        s.self_absent_ef512,
        s.self_miss_lvl1_ef512,
        s.self_miss_lvl0_ef512,
        s.self_miss_ef64,
        s.recall10[0],
        s.recall10[1],
        s.recall10[2],
        s.build_s
    )
}

/// CLI: --n N --dim D --mode iid|clust --seeds S --seed0 K --nq Q
pub fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}
pub fn arg_s(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|i| a.get(i + 1)).cloned().unwrap_or(default.to_string())
}
