//! G17 / W18 harness: unpatched registry hnsw_rs 0.3.4 vs one-line-patched copy, same data per seed,
//! the two variants built concurrently (2 threads). Experiment code is shared with src/bin/g17_hnsw.rs.
#[path = "../../../src/bin/g17_common/exp.rs"]
mod exp;
use exp::*;

crate::define_exp!(hnsw_orig, orig);
crate::define_exp!(hnsw_patched, patched);

fn main() {
    let n: usize = arg("--n", 1000);
    let dim: usize = arg("--dim", 32);
    let seeds: u64 = arg("--seeds", 20);
    let seed0: u64 = arg("--seed0", 1);
    let nq: usize = arg("--nq", 200);
    let mode = arg_s("--mode", "iid");
    println!("{}", fmt_header());
    for seed in seed0..seed0 + seeds {
        let all = gen_data(&mode, n + nq, dim, seed);
        let (data, queries) = all.split_at(n);
        let truth = brute_truth(data, queries);
        let (so, sp) = std::thread::scope(|sc| {
            let a = sc.spawn(|| orig::run_one(data, 16, 200, queries, &truth));
            let b = sc.spawn(|| patched::run_one(data, 16, 200, queries, &truth));
            (a.join().unwrap(), b.join().unwrap())
        });
        println!("{}", fmt_row("orig", &mode, dim, seed, &so));
        println!("{}", fmt_row("patched", &mode, dim, seed, &sp));
    }
}
