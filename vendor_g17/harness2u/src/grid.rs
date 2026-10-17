//! H15 grid: per build record the level prefix, layer-0 weak components, sampled audit and full self-miss. CSV rows to stdout.
//!   h15_grid --n 500 --dim 32 --m 16 --efc 200 --mode iid --builds 3000 --seed0 1 --fix 0 --sample 100
mod common;
use common::*;
use std::time::Instant;

fn main() {
    let builds: u64 = arg("--builds", 100);
    let n: usize = arg("--n", 500);
    let dim: usize = arg("--dim", 32);
    let seed0: u64 = arg("--seed0", 1);
    let fix: u8 = arg("--fix", 0);
    let samp: usize = arg("--sample", 100);
    let mode = arg_s("--mode", "iid");
    let prefix: Vec<usize> = arg_s("--prefix", "").split(',').filter(|s| !s.is_empty()).map(|s| s.parse().unwrap()).collect();
    let cfg = Cfg { m: arg("--m", 16), efc: arg("--efc", 200), keep_pruned: flag("--keep-pruned"), extend: flag("--extend"), level_scale: arg("--scale", 1.0) };
    if !flag("--nohdr") {
        println!("mode,n,dim,m,efc,fix,seed,lev0,lev1,flag,comps,sample_miss,full_miss,build_ms,audit_ms");
    }
    for b in seed0..seed0 + builds {
        let data = gen_data(&mode, n, dim, b);
        let t0 = Instant::now();
        let h = build_ex(&data, &cfg, b, &prefix, fix);
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;
        let st = structure(&h, n);
        let comps = weak_components(&st);
        let fl = st.level[0] >= 1 && st.level[1] >= 1;
        let t1 = Instant::now();
        let step = (n / samp).max(1);
        let sm = (0..n).step_by(step).filter(|&i| h.search(&data[i], 1, 64).first().map(|x| x.d_id) != Some(i)).count();
        let audit_ms = t1.elapsed().as_secs_f64() * 1e3;
        let full = if fl || comps > 1 || sm > 0 { self_miss(&h, &data, 64).len() as i64 } else { 0 };
        println!("{mode},{n},{dim},{},{},{fix},{b},{},{},{},{comps},{sm},{full},{build_ms:.1},{audit_ms:.2}", cfg.m, cfg.efc, st.level[0], st.level[1], fl as u8);
    }
}
