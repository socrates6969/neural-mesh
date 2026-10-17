//! H15: causal test. Force the levels of the first points, random (seeded) levels afterwards, count catastrophic builds.
//!   h15_force --prefix 1,1 --builds 100 --n 2000 --dim 32 --fix 0
mod common;
use common::*;

fn main() {
    let builds: u64 = arg("--builds", 100);
    let n: usize = arg("--n", 2000);
    let dim: usize = arg("--dim", 32);
    let seed0: u64 = arg("--seed0", 1);
    let fix: u8 = arg("--fix", 0);
    let mode = arg_s("--mode", "iid");
    let prefix: Vec<usize> = arg_s("--prefix", "").split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
    let cfg = Cfg { m: arg("--m", 16), efc: arg("--efc", 200), keep_pruned: flag("--keep-pruned"), extend: flag("--extend"), level_scale: arg("--scale", 1.0) };
    let (mut cat, mut small, mut clean) = (0u64, 0u64, 0u64);
    let mut fr = vec![];
    for b in seed0..seed0 + builds {
        let data = gen_data(&mode, n, dim, b);
        let h = if flag("--par") { build_par(&data, &cfg, b, &prefix, fix) } else { build_ex(&data, &cfg, b, &prefix, fix) };
        let miss = self_miss(&h, &data, 64).len();
        if miss == 0 {
            clean += 1;
        } else if miss * 20 <= n {
            small += 1;
        } else {
            cat += 1;
            fr.push(miss as f64 / n as f64);
        }
    }
    let mean = if fr.is_empty() { 0.0 } else { fr.iter().sum::<f64>() / fr.len() as f64 };
    println!("prefix {:?} fix {fix} mode {mode} n {n} d {dim} M {}: builds {builds} clean {clean} small {small} CAT {cat} (mean lost frac of cat {:.3})", prefix, cfg.m, mean);
}
