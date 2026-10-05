//! H1: look for RARE CATASTROPHIC HNSW builds (>5 % of items not their own top-1) and dump their structure.
//! Seen in h1_hnsw with the PATCHED crate: 2 builds of ~300 lost ~24 % of items (recall@10 .67) -- not the 1 % reverse-edge loss.
//!   h1_hnsw_probe --builds 600 --n 2000 --dim 32 --mode iid
use hnsw_rs::prelude::Hnsw;
use neural_mesh::memory::synth::Rng as SRng;
use neural_mesh::memory::{dot, normalized, CosineDist};

fn arg<T: std::str::FromStr>(k: &str, d: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() {
    if std::env::args().any(|a| a == "--audit") {
        return audit_main();
    }
    let builds: u64 = arg("--builds", 300);
    let n: usize = arg("--n", 2000);
    let dim: usize = arg("--dim", 32);
    let seed0: u64 = arg("--seed0", 1);
    let efc: usize = arg("--efc", 200);
    let m: usize = arg("--m", 16);
    let label: String = arg("--label", "x".to_string());
    let kp = std::env::args().any(|a| a == "--keep-pruned");
    let ext = std::env::args().any(|a| a == "--extend");
    let (mut cat, mut small, mut clean) = (0usize, 0usize, 0usize);
    for b in seed0..seed0 + builds {
        let mut r = SRng::new(b.wrapping_mul(7919).wrapping_add(13));
        let data: Vec<Vec<f32>> = (0..n).map(|_| normalized(&(0..dim).map(|_| r.normal()).collect::<Vec<_>>())).collect();
        let mut h: Hnsw<'static, f32, CosineDist> = Hnsw::new(m, n, 16, efc, CosineDist);
        if kp {
            h.set_keeping_pruned(true);
        }
        if ext {
            h.set_extend_candidates(true);
        }
        for (i, v) in data.iter().enumerate() {
            h.insert_slice((v.as_slice(), i));
        }
        let miss: Vec<usize> = (0..n).filter(|&i| h.search(&data[i], 1, 64).first().map(|x| x.d_id) != Some(i)).collect();
        if miss.is_empty() {
            clean += 1;
            continue;
        }
        if miss.len() <= n / 20 {
            small += 1;
            continue;
        }
        cat += 1;
        // structure dump
        let pi = h.get_point_indexation();
        let maxl = h.get_max_level_observed() as usize;
        let mut level = vec![0u8; n];
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
        let mut indeg = vec![0u32; n];
        let mut nlev = vec![0usize; maxl + 1];
        for l in 0..=maxl {
            for p in pi.get_layer_iterator(l) {
                let id = p.get_origin_id();
                level[id] = p.get_point_id().0;
                nlev[l] += 1;
                for x in &p.get_neighborhood_id()[0] {
                    adj[id].push(x.d_id);
                    indeg[x.d_id] += 1;
                }
            }
        }
        let entry = pi.get_layer_iterator(maxl).next().map(|p| p.get_origin_id()).unwrap_or(0);
        let mut seen = vec![false; n];
        let mut st = vec![entry];
        seen[entry] = true;
        let mut cnt = 1;
        while let Some(u) = st.pop() {
            for &w in &adj[u] {
                if !seen[w] {
                    seen[w] = true;
                    cnt += 1;
                    st.push(w);
                }
            }
        }
        let mut ids = miss.clone();
        ids.sort();
        let lv1 = miss.iter().filter(|&&i| level[i] >= 1).count();
        let unreach_missing = miss.iter().filter(|&&i| !seen[i]).count();
        let in0 = miss.iter().filter(|&&i| indeg[i] == 0).count();
        let outd0 = miss.iter().filter(|&&i| adj[i].is_empty()).count();
        // do missing points find themselves at ef 512 / 2000?
        let found512 = miss.iter().filter(|&&i| h.search(&data[i], 1, 512).first().map(|x| x.d_id) == Some(i)).count();
        println!(
            "[{label}] CATASTROPHIC build seed {b}: miss {}/{} | entry id {entry} level {maxl} | points per level {:?} | reachable-from-entry(layer0) {cnt}/{n}, missing&unreachable {unreach_missing}, missing with in-degree0 {in0}, missing with out-degree0 {outd0}, missing level>=1: {lv1} | missing id min/med/max {}/{}/{} | found at ef512: {found512}",
            miss.len(), n, nlev, ids[0], ids[ids.len() / 2], ids[ids.len() - 1]
        );
        // spatial structure of the lost set: are the lost points one region? mean cosine within the lost set vs random pairs, and how many layer-0 out-edges of lost points stay inside the lost set
        let mset: std::collections::HashSet<usize> = miss.iter().copied().collect();
        let (mut inside, mut tot_e) = (0usize, 0usize);
        for &i in &miss {
            for &w in &adj[i] {
                tot_e += 1;
                if mset.contains(&w) {
                    inside += 1;
                }
            }
        }
        let mut rr = SRng::new(b ^ 0x55);
        let (mut cm, mut cr) = (0.0f64, 0.0f64);
        for _ in 0..2000 {
            let (x, y) = (miss[rr.below(miss.len())], miss[rr.below(miss.len())]);
            cm += dot(&data[x], &data[y]) as f64;
            let (x, y) = (rr.below(n), rr.below(n));
            cr += dot(&data[x], &data[y]) as f64;
        }
        println!("      lost set: {:.1} % of its layer-0 out-edges stay inside the lost set (random expectation {:.1} %); mean cosine within lost set {:.3} vs random pairs {:.3}", 100.0 * inside as f64 / tot_e.max(1) as f64, 100.0 * miss.len() as f64 / n as f64, cm / 2000.0, cr / 2000.0);
        let first_missing_lvl: Vec<u8> = ids.iter().take(10).map(|&i| level[i]).collect();
        println!("      first missing ids {:?} levels {:?}; entry-point neighbours at layer0 {}", &ids[..ids.len().min(10)], first_missing_lvl, adj[entry].len());
        let _ = dot(&data[0], &data[0]);
    }
    println!("[{label}] builds {builds} n {n} d {dim} M {m} efc {efc} keep_pruned {kp} extend {ext}: clean {clean}, small-loss(<=5%) {small}, CATASTROPHIC(>5%) {cat}");
}

/// `--audit`: wrapper-level evaluation of HnswIndex::audit / audit_and_repair against the full self-recall truth.
pub fn audit_main() {
    use neural_mesh::memory::{HnswIndex, HnswParams, MemoryIndex};
    let builds: u64 = arg("--builds", 300);
    let n: usize = arg("--n", 2000);
    let dim: usize = arg("--dim", 32);
    let seed0: u64 = arg("--seed0", 1);
    let sample: usize = arg("--sample", 200);
    let thr: usize = arg("--thr", 3);
    let label: String = arg("--label", "x".to_string());
    let (mut cat, mut det, mut fa, mut clean_or_small, mut repaired_ok, mut repair_fail) = (0, 0, 0, 0, 0, 0);
    let (mut audit_s, mut build_s) = (0.0f64, 0.0f64);
    for b in seed0..seed0 + builds {
        let mut r = SRng::new(b.wrapping_mul(7919).wrapping_add(13));
        let data: Vec<Vec<f32>> = (0..n).map(|_| normalized(&(0..dim).map(|_| r.normal()).collect::<Vec<_>>())).collect();
        let t0 = std::time::Instant::now();
        let mut h = HnswIndex::new(dim, HnswParams { max_elements: n, ..Default::default() });
        for (i, v) in data.iter().enumerate() {
            h.add(i as u64, v);
        }
        build_s += t0.elapsed().as_secs_f64();
        let truth = |h: &HnswIndex| (0..n).filter(|&i| h.search(&data[i], 1).first().map(|x| x.0) != Some(i as u64)).count();
        let lost_full = truth(&h);
        let t1 = std::time::Instant::now();
        let (_c, l) = h.audit(sample);
        audit_s += t1.elapsed().as_secs_f64();
        let is_cat = lost_full > n / 20;
        let flagged = l > thr;
        if is_cat {
            cat += 1;
            if flagged {
                det += 1;
            }
        } else {
            clean_or_small += 1;
            if flagged {
                fa += 1;
            }
        }
        if flagged {
            let (first, rb) = h.audit_and_repair(sample, thr, 3);
            let after = truth(&h);
            println!("[{label}] build {b}: full lost {lost_full}, audit sample lost {l} (first {first}) -> rebuilds {rb}, full lost after repair {after}");
            if after <= n / 20 {
                repaired_ok += 1;
            } else {
                repair_fail += 1;
            }
        }
    }
    println!("[{label}] AUDIT builds {builds} n {n} d {dim}: catastrophic {cat} (detected {det}), non-catastrophic {clean_or_small} (false alarms {fa}); repaired ok {repaired_ok}, repair failed {repair_fail}; audit({sample}) = {:.2} ms each, build {:.0} ms each; flag if sample-lost > {thr}", 1e3 * audit_s / builds as f64, 1e3 * build_s / builds as f64);
}
