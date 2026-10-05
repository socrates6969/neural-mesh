//! G17 / W18: hnsw_rs 0.3.4 reverse-edge check, UNPATCHED crate only (the registry one neural-mesh uses).
//! The patched comparison lives in vendor_g17/harness (same experiment source, patched path crate).
//! Run: RAYON_NUM_THREADS=2 cargo run --release --offline --bin g17_hnsw -- --n 1000 --dim 32 --mode iid --seeds 20
#[path = "g17_common/exp.rs"]
mod exp;
use exp::*;

crate::define_exp!(hnsw_rs, orig);

fn main() {
    if std::env::args().any(|a| a == "--wrapper") {
        return wrapper_main();
    }
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
        let s = orig::run_one(data, 16, 200, queries, &truth);
        println!("{}", fmt_row("orig", &mode, dim, seed, &s));
    }
}

/// `--wrapper`: same measurement through neural-mesh's own HnswIndex / HnswRm (cosine on normalised vectors,
/// default HnswParams M16 efc200 ef64). Columns: wrapper,mode,n,dim,seed,path,miss_ef64,miss_ef512
#[allow(dead_code)]
pub fn wrapper_main() {
    use neural_mesh::memory::{HnswIndex, HnswParams, MemoryIndex};
    use neural_mesh::tiered::{HnswRm, RemovableIndex};
    let n: usize = arg("--n", 1000);
    let dim: usize = arg("--dim", 32);
    let seeds: u64 = arg("--seeds", 20);
    let seed0: u64 = arg("--seed0", 1);
    let mode = arg_s("--mode", "iid");
    println!("wrapper,mode,n,dim,seed,path,miss_ef64,miss_ef512");
    let p = HnswParams { max_elements: n, ..Default::default() };
    for seed in seed0..seed0 + seeds {
        let data = gen_data(&mode, n, dim, seed);
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        let ids: Vec<u64> = (0..n as u64).collect();
        let miss = |idx: &dyn MemoryIndex| -> usize { (0..n).filter(|&i| idx.search(&data[i], 1).first().map(|x| x.0) != Some(i as u64)).count() };
        let run = |name: &str, idx: &HnswIndex| {
            idx.set_ef_search(64);
            let m64 = miss(idx);
            idx.set_ef_search(512);
            let m512 = miss(idx);
            println!("wrapper,{},{},{},{},{},{},{}", mode, n, dim, seed, name, m64, m512);
        };
        let mut a = HnswIndex::new(dim, p);
        for (i, v) in data.iter().enumerate() {
            a.add(i as u64, v);
        }
        run("HnswIndex.add", &a);
        let mut b = HnswIndex::new(dim, p);
        b.add_batch(&ids, &flat);
        run("HnswIndex.add_batch(par)", &b);
        // tiered long store: HnswRm (add), then remove 10% + compact (rebuild via add_batch)
        let mut c = HnswRm::new(dim, p);
        for (i, v) in data.iter().enumerate() {
            c.add(i as u64, v);
        }
        let m_c = (0..n).filter(|&i| c.search(&data[i], 1).first().map(|x| x.0) != Some(i as u64)).count();
        for i in (0..n).step_by(10) {
            c.remove(i as u64);
        }
        c.compact();
        let m_c2 = (0..n).filter(|&i| i % 10 != 0).filter(|&i| c.search(&data[i], 1).first().map(|x| x.0) != Some(i as u64)).count();
        println!("wrapper,{},{},{},{},HnswRm.add(ef64),{},-", mode, n, dim, seed, m_c);
        println!("wrapper,{},{},{},{},HnswRm.after_remove10pct+compact(ef64;live={}),{},-", mode, n, dim, seed, n - (n + 9) / 10, m_c2);
    }
}
