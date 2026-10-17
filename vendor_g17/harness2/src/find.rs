//! H15: reproducible search for catastrophic builds. h15_find --builds 600 --n 2000 --dim 32 --mode iid --seed0 1
mod common;
use common::*;

fn main() {
    let builds: u64 = arg("--builds", 300);
    let n: usize = arg("--n", 2000);
    let dim: usize = arg("--dim", 32);
    let seed0: u64 = arg("--seed0", 1);
    let mode = arg_s("--mode", "iid");
    let cfg = Cfg { m: arg("--m", 16), efc: arg("--efc", 200), keep_pruned: flag("--keep-pruned"), extend: flag("--extend"), level_scale: arg("--scale", 1.0) };
    let thr: f64 = arg("--thr", 0.05);
    let (mut clean, mut small, mut cat) = (0u64, 0u64, 0u64);
    let mut cat_seeds = vec![];
    for b in seed0..seed0 + builds {
        let data = gen_data(&mode, n, dim, b);
        let h = build(&data, &cfg, b);
        let samp: usize = arg("--sample", 0);
        if samp > 0 {
            let step = (n / samp).max(1);
            let sm = (0..n).step_by(step).filter(|&i| h.search(&data[i], 1, 64).first().map(|x| x.d_id) != Some(i)).count();
            if (sm as f64) < 0.03 * (n / step) as f64 {
                clean += 1;
                continue;
            }
        }
        let miss = self_miss(&h, &data, 64);
        if miss.is_empty() {
            clean += 1;
        } else if (miss.len() as f64) <= thr * n as f64 {
            small += 1;
        } else {
            cat += 1;
            cat_seeds.push(b);
            let st = structure(&h, n);
            let nl1 = st.level.iter().filter(|&&l| l >= 1).count();
            println!("CAT seed {b}: miss {}/{n} entry {} maxl {} n_lvl>=1 {nl1} reach0 {}/{n}", miss.len(), st.entry, st.maxl, st.n_reach0);
        }
    }
    println!("builds {builds} n {n} d {dim} mode {mode} M {} efc {} kp {} ext {} scale {}: clean {clean} small {small} CAT {cat} seeds {:?}", cfg.m, cfg.efc, cfg.keep_pruned, cfg.extend, cfg.level_scale, cat_seeds);
}
