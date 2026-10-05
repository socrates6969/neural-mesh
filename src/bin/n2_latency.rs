//! R44: consolidated latency benchmark (single thread, release). Real corpus: results/h10_*_A.* (6,831 real paragraphs, 384-d MiniLM).
//! Reports p50/p95 microseconds per query. Synthetic scale-up (random unit vectors) for the flat tiers only.
use neural_mesh::flatq::{FlatIndex, Metric, Tier};
use neural_mesh::hybrid::{AdaptiveAlpha, HybridConfig, HybridIndex};
use serde_json::Value;
use std::time::Instant;

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p) as usize]
}
fn time<F: FnMut(usize)>(name: &str, nq: usize, reps: usize, mut f: F) {
    let mut t = Vec::new();
    for _ in 0..reps {
        for q in 0..nq {
            let t0 = Instant::now();
            f(q);
            t.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    let p50 = pct(&mut t, 0.5);
    let p95 = pct(&mut t, 0.95);
    println!("{name:44} p50 {p50:9.1} us   p95 {p95:9.1} us");
}
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    }
}
fn unit(r: &mut Rng, d: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..d).map(|_| r.next()).collect();
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= n);
    v
}
fn main() {
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read("results/h10_texts_A.json").unwrap()).unwrap();
    let raw = std::fs::read("results/h10_emb_A.f32").unwrap();
    let emb: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    let dim = 384;
    let inst: Value = serde_json::from_slice(&std::fs::read("results/h10_inst_A.json").unwrap()).unwrap();
    let ndoc = inst["ndoc"].as_u64().unwrap() as usize;
    let rows = inst["inst"]["31"]["rows"].as_array().unwrap();
    let qids: Vec<usize> = rows.iter().take(200).map(|r| r["t"][0][0].as_u64().unwrap() as usize).collect();
    let docs: Vec<String> = texts[..ndoc].to_vec();
    let dense: Vec<f32> = emb[..ndoc * dim].to_vec();
    let ix = HybridIndex::new(&docs, dense.clone(), dim, HybridConfig::default()).unwrap();
    let qv: Vec<&[f32]> = qids.iter().map(|&t| &emb[t * dim..(t + 1) * dim]).collect();
    let ad = AdaptiveAlpha::default();
    println!("== real corpus N={ndoc}, d={dim}, {} queries x5", qids.len());
    time("hybrid search k=10 (fixed alpha .7)", qids.len(), 5, |q| { std::hint::black_box(ix.search(&texts[qids[q]], qv[q], 10).unwrap()); });
    let ixf = ix.clone().with_fast_kernels(true);
    time("hybrid search k=10 FAST kernels", qids.len(), 5, |q| { std::hint::black_box(ixf.search(&texts[qids[q]], qv[q], 10).unwrap()); });
    {
        let (mut same_ids, mut maxd) = (0usize, 0f32);
        for q in 0..qids.len() {
            let a = ix.search(&texts[qids[q]], qv[q], 10).unwrap();
            let b = ixf.search(&texts[qids[q]], qv[q], 10).unwrap();
            if a.iter().map(|x| x.0).collect::<Vec<_>>() == b.iter().map(|x| x.0).collect::<Vec<_>>() { same_ids += 1; }
            for (x, y) in a.iter().zip(&b) { maxd = maxd.max((x.1 - y.1).abs()); }
        }
        println!("parity hybrid fast vs scalar: identical top-10 id lists {same_ids}/{}  max |score diff| {maxd:.2e}", qids.len());
    }
    {
        // wide parity: every distinct query text of seeds 31..90 (hybrid scalar vs FAST)
        let mut seen = std::collections::BTreeSet::new();
        for sd in 31..=90 { if let Some(rs) = inst["inst"][sd.to_string()]["rows"].as_array() { for r in rs { seen.insert(r["t"][0][0].as_u64().unwrap() as usize); } } }
        let (mut n, mut t1, mut t10) = (0usize, 0usize, 0usize);
        for &t in &seen {
            let qvv = &emb[t * dim..(t + 1) * dim];
            let a = ix.search(&texts[t], qvv, 10).unwrap(); let b = ixf.search(&texts[t], qvv, 10).unwrap();
            n += 1; t1 += (a[0].0 == b[0].0) as usize; t10 += (a.iter().map(|x| x.0).collect::<Vec<_>>() == b.iter().map(|x| x.0).collect::<Vec<_>>()) as usize;
        }
        println!("wide parity ({n} distinct queries): top-1 identical {t1}/{n}, top-10 identical id lists {t10}/{n}");
    }
    time("hybrid search_adaptive k=10", qids.len(), 5, |q| { std::hint::black_box(ix.search_adaptive(&texts[qids[q]], qv[q], 10, &ad).unwrap()); });
    time("hybrid search_mmr k=5 pool=40 lam=.7", qids.len(), 5, |q| { std::hint::black_box(ix.search_mmr(&texts[qids[q]], qv[q], 5, 0.7, 40, 4.0).unwrap()); });
    for (name, tier) in [("flat f32", Tier::F32), ("flat blockfp16", Tier::BlockFp16), ("flat int8", Tier::Int8)] {
        let mut f = FlatIndex::new(dim, tier, Metric::Cosine);
        for i in 0..ndoc { f.add(i as u64, &emb[i * dim..(i + 1) * dim]).unwrap(); }
        time(&format!("{name} dense scan k=10 (N={ndoc})"), qids.len(), 5, |q| { std::hint::black_box(f.search(qv[q], 10).unwrap()); });
        let mut g = FlatIndex::new(dim, tier, Metric::Cosine).with_fast_kernels(true);
        for i in 0..ndoc { g.add(i as u64, &emb[i * dim..(i + 1) * dim]).unwrap(); }
        time(&format!("{name} FAST scan k=10 (N={ndoc})"), qids.len(), 5, |q| { std::hint::black_box(g.search(qv[q], 10).unwrap()); });
        let (mut same, mut maxd) = (0usize, 0f32);
        for q in 0..qids.len() {
            let a = f.search(qv[q], 10).unwrap(); let b = g.search(qv[q], 10).unwrap();
            if a.iter().map(|x| x.0).collect::<Vec<_>>() == b.iter().map(|x| x.0).collect::<Vec<_>>() { same += 1; }
            for (x, y) in a.iter().zip(&b) { maxd = maxd.max((x.1 - y.1).abs()); }
        }
        println!("   parity {name}: identical top-10 {same}/{} max |diff| {maxd:.2e}", qids.len());
    }
    for &n in &[100_000usize, 500_000] {
        println!("== synthetic N={n}");
        let mut r = Rng(88172645463325252);
        let vs: Vec<Vec<f32>> = (0..n).map(|_| unit(&mut r, dim)).collect();
        let qs: Vec<Vec<f32>> = (0..20).map(|_| unit(&mut r, dim)).collect();
        for (name, tier, d) in [("flat f32 d=384", Tier::F32, 384usize), ("flat int8 d=384", Tier::Int8, 384), ("flat int8 d=192 (PCA192 stand-in)", Tier::Int8, 192), ("flat f32 d=192", Tier::F32, 192)] {
            let mut f = FlatIndex::new(d, tier, Metric::Cosine);
            for (i, v) in vs.iter().enumerate() { f.add(i as u64, &v[..d]).unwrap(); }
            time(&format!("{name}"), qs.len(), if n > 100_000 { 2 } else { 3 }, |q| { std::hint::black_box(f.search(&qs[q][..d], 10).unwrap()); });
            let mut g = FlatIndex::new(d, tier, Metric::Cosine).with_fast_kernels(true);
            for (i, v) in vs.iter().enumerate() { g.add(i as u64, &v[..d]).unwrap(); }
            time(&format!("{name} FAST"), qs.len(), if n > 100_000 { 2 } else { 3 }, |q| { std::hint::black_box(g.search(&qs[q][..d], 10).unwrap()); });
        }
        // projection cost: 384 -> 192 matvec
        let p: Vec<f32> = (0..384 * 192).map(|i| ((i % 97) as f32) * 0.001).collect();
        time("query projection 384->192 matvec", qs.len(), 200, |q| {
            let mut o = vec![0f32; 192];
            for (j, oj) in o.iter_mut().enumerate() { let mut s = 0f32; for k in 0..384 { s += qs[q][k] * p[k * 192 + j]; } *oj = s; }
            std::hint::black_box(o);
        });
    }
}
