//! H15 fix validation. Same data + same level seed, build with FIX_MODE 0 (current vendored crate), 1 (layer 0 never empty), 2 (suffix check).
//! Per seed: graph hash (all layers, neighbour ids in stored order) for each mode, self-miss of each mode, recall@10 (ef 50/100/200), build ms, us/query.
//!   h15_cmp --n 500 --dim 32 --builds 6000 --seed0 1 --perf 0     (identity + failure count, cheap)
//!   h15_cmp --n 5000 --dim 32 --builds 20 --perf 1 --nq 100       (recall / latency / build time)
mod common;
use common::*;
use std::time::Instant;

fn graph_hash(h: &hnsw_exp::hnsw::Hnsw<'static, f32, Cos>, n: usize) -> u64 {
    let pi = h.get_point_indexation();
    let maxl = h.get_max_level_observed() as usize;
    let mut per: Vec<(usize, u64)> = vec![];
    for l in 0..=maxl {
        for p in pi.get_layer_iterator(l) {
            let mut x: u64 = 0xcbf29ce484222325 ^ p.get_origin_id() as u64;
            for (li, nb) in p.get_neighborhood_id().iter().enumerate() {
                x = (x ^ (li as u64 + 1000)).wrapping_mul(0x100000001b3);
                for e in nb {
                    x = (x ^ e.d_id as u64).wrapping_mul(0x100000001b3);
                }
            }
            per.push((p.get_origin_id(), x));
        }
    }
    per.sort();
    let mut g: u64 = n as u64;
    for (id, x) in per {
        g = (g ^ id as u64).wrapping_mul(0x100000001b3);
        g = (g ^ x).wrapping_mul(0x100000001b3);
    }
    g
}

fn main() {
    let builds: u64 = arg("--builds", 100);
    let n: usize = arg("--n", 500);
    let dim: usize = arg("--dim", 32);
    let seed0: u64 = arg("--seed0", 1);
    let nq: usize = arg("--nq", 100);
    let perf: u8 = arg("--perf", 0);
    let mode = arg_s("--mode", "iid");
    let cfg = Cfg { m: arg("--m", 16), efc: arg("--efc", 200), keep_pruned: false, extend: false, level_scale: 1.0 };
    println!("n,dim,seed,fix,flag,graph_hash,self_miss,r10_ef50,r10_ef100,r10_ef200,build_ms,us_per_query_ef100");
    for b in seed0..seed0 + builds {
        let all = gen_data(&mode, n + nq, dim, b);
        let (data, qs) = all.split_at(n);
        let truth: Vec<Vec<usize>> = if perf > 0 {
            qs.iter()
                .map(|q| {
                    let mut d: Vec<(f32, usize)> = data.iter().enumerate().map(|(i, v)| (-dot(q, v), i)).collect();
                    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    d.iter().take(10).map(|x| x.1).collect()
                })
                .collect()
        } else {
            vec![]
        };
        for fix in 0u8..3 {
            let t0 = Instant::now();
            let h = build_ex(data, &cfg, b, &[], fix);
            let build_ms = t0.elapsed().as_secs_f64() * 1e3;
            let st = structure(&h, n);
            let fl = st.level[0] >= 1 && st.level[1] >= 1;
            let hash = graph_hash(&h, n);
            let sm = if perf > 0 || fl || n <= 1000 { self_miss(&h, data, 64).len() } else { 0 };
            let (mut r, mut us) = ([0f64; 3], 0f64);
            if perf > 0 {
                for (j, &ef) in [50usize, 100, 200].iter().enumerate() {
                    let t = Instant::now();
                    let mut hit = 0usize;
                    for (qi, q) in qs.iter().enumerate() {
                        let res = h.search(q, 10, ef);
                        hit += res.iter().filter(|x| truth[qi].contains(&x.d_id)).count();
                    }
                    if ef == 100 {
                        us = t.elapsed().as_secs_f64() * 1e6 / nq as f64;
                    }
                    r[j] = hit as f64 / (10 * nq) as f64;
                }
            }
            println!("{n},{dim},{b},{fix},{},{hash:016x},{sm},{:.4},{:.4},{:.4},{build_ms:.1},{us:.0}", fl as u8, r[0], r[1], r[2]);
        }
    }
}
