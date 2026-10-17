//! G17 workaround test on the UNMODIFIED registry hnsw_rs 0.3.4 (no patch involved).
//! Output columns reuse the g17 CSV header; in this file: unreach0 = us/query (ef=100,k=10),
//! in0_lvl1 (reins only) = number of lost level>=1 points that were re-inserted, build_s includes re-insert.
#[path = "../../../src/bin/g17_common/exp.rs"]
mod exp;
use exp::*;

crate::define_exp!(hnsw_orig, orig);

fn main() {
    let n: usize = arg("--n", 1000);
    let dim: usize = arg("--dim", 32);
    let seeds: u64 = arg("--seeds", 20);
    let seed0: u64 = arg("--seed0", 1);
    let nq: usize = arg("--nq", 200);
    let mode = arg_s("--mode", "iid");
    let wks = arg_s("--wk", "none,dup,flat02,flat05,reins");
    println!("{}", fmt_header());
    for seed in seed0..seed0 + seeds {
        let all = gen_data(&mode, n + nq, dim, seed);
        let (data, queries) = all.split_at(n);
        let truth = brute_truth(data, queries);
        for wk in wks.split(',') {
            let s = orig::run_wk(wk, data, 16, 200, queries, &truth);
            println!("{}", fmt_row(wk, &mode, dim, seed, &s));
        }
    }
}
