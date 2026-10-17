//! H15: step-by-step trace of one build (reproducible by seed). Prints, per insertion where the layer-0 reachable-from-entry
//! count stops equalling the number inserted, the point, its level, the entry point and a short description.
//!   h15_trace --seed 100062 --n 2000 --dim 32
mod common;
use common::*;
use hnsw_exp::hnsw::{set_level_seed, Hnsw};

fn main() {
    let n: usize = arg("--n", 2000);
    let dim: usize = arg("--dim", 32);
    let seed: u64 = arg("--seed", 100062);
    let mode = arg_s("--mode", "iid");
    let m: usize = arg("--m", 16);
    let efc: usize = arg("--efc", 200);
    let data = gen_data(&mode, n, dim, seed);
    set_level_seed(level_seed_for(seed));
    let h: Hnsw<'static, f32, Cos> = Hnsw::new(m, n, 16, efc, Cos);
    let mut last_unreach = 0usize;
    let mut last_entry = usize::MAX;
    for (i, v) in data.iter().enumerate() {
        h.insert_slice((v.as_slice(), i));
        let st = structure(&h, i + 1);
        let unreach = (i + 1) - st.n_reach0;
        let lvl = st.level[i];
        if st.entry != last_entry || unreach != last_unreach {
            println!(
                "i={i:5} level={lvl} entry={} (maxl {}) inserted={} reach0={} unreach={} (delta {:+})",
                st.entry,
                st.maxl,
                i + 1,
                st.n_reach0,
                unreach,
                unreach as i64 - last_unreach as i64
            );
            last_entry = st.entry;
            last_unreach = unreach;
        }
    }
}
