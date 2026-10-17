//! H15: structural analysis of one (reproducible) failing build.  h15_anal --seed 100062
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
    let step: usize = arg("--step", 100);
    let data = gen_data(&mode, n, dim, seed);
    set_level_seed(level_seed_for(seed));
    let h: Hnsw<'static, f32, Cos> = Hnsw::new(m, n, 16, efc, Cos);
    println!("checkpoint: inserted, self-miss among inserted (ef64), self-miss among the points inserted since previous checkpoint");
    let mut prev_missing: Vec<usize> = vec![];
    for (i, v) in data.iter().enumerate() {
        h.insert_slice((v.as_slice(), i));
        if (i + 1) % step == 0 {
            let miss: Vec<usize> = (0..=i).filter(|&j| h.search(&data[j], 1, 64).first().map(|x| x.d_id) != Some(j)).collect();
            let recent = miss.iter().filter(|&&j| j > i - step).count();
            let new_loss = miss.iter().filter(|j| !prev_missing.contains(j)).count();
            let regained = prev_missing.iter().filter(|j| !miss.contains(j)).count();
            println!("  {:5} missing {:4} (recent {:3}, newly lost {:3}, regained {:3})", i + 1, miss.len(), recent, new_loss, regained);
            prev_missing = miss;
        }
    }
    let st = structure(&h, n);
    let miss = prev_missing;
    println!("final: miss {} entry {} maxl {} levels>=1 {}", miss.len(), st.entry, st.maxl, st.level.iter().filter(|&&l| l >= 1).count());
    // missing by id bucket
    let mut b = vec![0usize; 10];
    for &j in &miss {
        b[j * 10 / n] += 1;
    }
    println!("missing per id-decile {:?}", b);
    // missing: in-degree, out-degree, level
    let mut ind = vec![0usize; 6];
    for &j in &miss {
        ind[(st.indeg0[j] as usize).min(5)] += 1;
    }
    println!("missing indeg0 histogram (0..5+) {:?}; with outdeg0: {}; level>=1: {}", ind, miss.iter().filter(|&&j| st.adj0[j].is_empty()).count(), miss.iter().filter(|&&j| st.level[j] >= 1).count());
    // weak components of the layer-0 graph
    let mut und: Vec<Vec<usize>> = vec![vec![]; n];
    for u in 0..n {
        for &w in &st.adj0[u] {
            und[u].push(w);
            und[w].push(u);
        }
    }
    let mut comp = vec![usize::MAX; n];
    let mut sizes = vec![];
    for s in 0..n {
        if comp[s] != usize::MAX {
            continue;
        }
        let id = sizes.len();
        let mut st2 = vec![s];
        comp[s] = id;
        let mut c = 0;
        while let Some(u) = st2.pop() {
            c += 1;
            for &w in &und[u] {
                if comp[w] == usize::MAX {
                    comp[w] = id;
                    st2.push(w);
                }
            }
        }
        sizes.push(c);
    }
    sizes.sort_by(|a, b| b.cmp(a));
    println!("weak components of layer-0 graph: {} sizes (top 10) {:?}", sizes.len(), &sizes[..sizes.len().min(10)]);
    // strong-ish: how many points reachable from ALL level>=1 points (landing sites) via layer-0
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = (0..n).filter(|&j| st.level[j] >= 1).collect();
    for &s in &stack {
        seen[s] = true;
    }
    while let Some(u) = stack.pop() {
        for &w in &st.adj0[u] {
            if !seen[w] {
                seen[w] = true;
                stack.push(w);
            }
        }
    }
    println!("reachable at layer-0 from the union of all level>=1 points: {}/{}; missing not reachable from them: {}", seen.iter().filter(|&&x| x).count(), n, miss.iter().filter(|&&j| !seen[j]).count());
    // reach from entry only
    println!("reachable from entry via layer 0: {}/{}", st.n_reach0, n);
    // degree stats
    let mean_out: f64 = st.adj0.iter().map(|a| a.len() as f64).sum::<f64>() / n as f64;
    let mean_out_miss: f64 = miss.iter().map(|&j| st.adj0[j].len() as f64).sum::<f64>() / miss.len().max(1) as f64;
    let mean_in_miss: f64 = miss.iter().map(|&j| st.indeg0[j] as f64).sum::<f64>() / miss.len().max(1) as f64;
    let mean_in: f64 = st.indeg0.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    println!("mean out-degree all {:.1} missing {:.1}; mean in-degree all {:.1} missing {:.1}", mean_out, mean_out_miss, mean_in, mean_in_miss);
}
