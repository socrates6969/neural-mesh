//! H15 audit cost/benefit. Positives = builds with forced level prefix [1,1] (the bug), negatives = ordinary builds whose first two
//! levels are not both >= 1 (forced prefix [0,0] keeps everything else random). For each build the FULL self-miss bit-vector (ef 64)
//! is computed once; audits of s evenly spaced samples (the memory.rs audit rule) are then simulated from it, so every s is
//! evaluated on the same builds. Also times: build, one search, the exact layer-0 connectivity check.
//!   h15_audit --n 2000 --dim 32 --pos 60 --neg 60
mod common;
use common::*;
use std::time::Instant;

fn main() {
    let n: usize = arg("--n", 2000);
    let dim: usize = arg("--dim", 32);
    let pos: u64 = arg("--pos", 60);
    let neg: u64 = arg("--neg", 60);
    let seed0: u64 = arg("--seed0", 1);
    let mode = arg_s("--mode", "iid");
    let cfg = Cfg { m: arg("--m", 16), efc: arg("--efc", 200), keep_pruned: false, extend: false, level_scale: 1.0 };
    let sizes = [5usize, 10, 20, 25, 50, 100, 200, 500];
    // (kind, build_ms, search_us, conn_ms, lost_frac, comps, hits per s)
    let mut rows: Vec<(bool, f64, f64, f64, f64, usize, Vec<usize>)> = vec![];
    for (kind, cnt, prefix) in [(true, pos, vec![1usize, 1]), (false, neg, vec![0usize, 0])] {
        for k in 0..cnt {
            let b = seed0 + k + if kind { 0 } else { 5_000_000 };
            let data = gen_data(&mode, n, dim, b);
            let t0 = Instant::now();
            let h = build_ex(&data, &cfg, b, &prefix, 0);
            let build_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let miss: Vec<bool> = (0..n).map(|i| h.search(&data[i], 1, 64).first().map(|x| x.d_id) != Some(i)).collect();
            let search_us = t1.elapsed().as_secs_f64() * 1e6 / n as f64;
            let t2 = Instant::now();
            let st = structure(&h, n);
            let comps = weak_components(&st);
            let conn_ms = t2.elapsed().as_secs_f64() * 1e3;
            let lost = miss.iter().filter(|&&x| x).count();
            let hits: Vec<usize> = sizes
                .iter()
                .map(|&s| {
                    let step = (n / s).max(1);
                    (0..n).step_by(step).take(s).filter(|&i| miss[i]).count()
                })
                .collect();
            rows.push((kind, build_ms, search_us, conn_ms, lost as f64 / n as f64, comps, hits));
        }
    }
    let p: Vec<_> = rows.iter().filter(|r| r.0).collect();
    let q: Vec<_> = rows.iter().filter(|r| !r.0).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let lost: Vec<f64> = p.iter().map(|r| r.4).collect();
    let mut ls = lost.clone();
    ls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("# n {n} d {dim} mode {mode} M {} efc {}: positives {} (forced [1,1]) negatives {} (forced [0,0])", cfg.m, cfg.efc, p.len(), q.len());
    println!("# build ms: pos {:.0} neg {:.0}; search us/query {:.0}; exact layer-0 connectivity check {:.2} ms (pos comps mean {:.2}, neg comps mean {:.2})", mean(&p.iter().map(|r| r.1).collect::<Vec<_>>()), mean(&q.iter().map(|r| r.1).collect::<Vec<_>>()), mean(&rows.iter().map(|r| r.2).collect::<Vec<_>>()), mean(&rows.iter().map(|r| r.3).collect::<Vec<_>>()), mean(&p.iter().map(|r| r.5 as f64).collect::<Vec<_>>()), mean(&q.iter().map(|r| r.5 as f64).collect::<Vec<_>>()));
    if !ls.is_empty() {
        println!("# positives: lost fraction min {:.3} p10 {:.3} median {:.3} p90 {:.3} max {:.3}; builds with comps==1: {}; negatives with lost>0: {} of {}, with comps>1: {}", ls[0], ls[ls.len() / 10], ls[ls.len() / 2], ls[ls.len() * 9 / 10], ls[ls.len() - 1], p.iter().filter(|r| r.5 == 1).count(), q.iter().filter(|r| r.4 > 0.0).count(), q.len(), q.iter().filter(|r| r.5 > 1).count());
    }
    println!("sample_s,audit_ms_est,detect_any_miss_pos,detect_pos_ci_lo,detect_pos_ci_hi,false_alarm_any_miss_neg,detect_gt3pct_pos,false_alarm_gt3pct_neg");
    let su = mean(&rows.iter().map(|r| r.2).collect::<Vec<_>>()) / 1e3;
    for (j, &s) in sizes.iter().enumerate() {
        let eff = (n / (n / s).max(1)).min(s).min(n);
        let d_any = p.iter().filter(|r| r.6[j] > 0).count();
        let fa_any = q.iter().filter(|r| r.6[j] > 0).count();
        let thr = (0.03 * eff as f64).floor() as usize;
        let d_thr = p.iter().filter(|r| r.6[j] > thr).count();
        let fa_thr = q.iter().filter(|r| r.6[j] > thr).count();
        let (lo, hi) = wilson(d_any, p.len());
        println!("{s},{:.2},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}", su * eff as f64, d_any as f64 / p.len().max(1) as f64, lo, hi, fa_any as f64 / q.len().max(1) as f64, d_thr as f64 / p.len().max(1) as f64, fa_thr as f64 / q.len().max(1) as f64);
    }
}
