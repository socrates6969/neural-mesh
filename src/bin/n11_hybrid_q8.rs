//! R91b: HybridIndex with the int8 dense shadow (`with_int8_dense`) vs the f32 FAST scan on the two REAL corpora (A: 6,831 paragraphs, B: 4,181), every unique query.
//! Writes per-query rank-of-first-relevant for both arms to results/r91_ranks_tp.json (python does the paired tests) and prints latency p50 / top-10 overlap.
use neural_mesh::hybrid::{HybridConfig, HybridIndex};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Instant;
fn main() {
    let mut out = serde_json::Map::new();
    for tag in ["A", "B"] {
        let texts: Vec<String> = serde_json::from_slice(&std::fs::read(format!("results/h10_texts_{tag}.json")).unwrap()).unwrap();
        let raw = std::fs::read(format!("results/h10_emb_{tag}.f32")).unwrap();
        let emb: Vec<f32> = raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        let dim = 384;
        let inst: Value = serde_json::from_slice(&std::fs::read(format!("results/h10_inst_{tag}.json")).unwrap()).unwrap();
        let ndoc = inst["ndoc"].as_u64().unwrap() as usize;
        let mut qmap: BTreeMap<u64, usize> = BTreeMap::new();
        for (s, v) in inst["inst"].as_object().unwrap() { if s == "R" { continue; } for row in v["rows"].as_array().unwrap() { qmap.insert(row["q"].as_u64().unwrap(), row["t"][0][0].as_u64().unwrap() as usize); } }
        let rels: Vec<Vec<usize>> = inst["queries"].as_array().unwrap().iter().map(|q| q["rel"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()).collect();
        let docs: Vec<String> = texts[..ndoc].to_vec();
        let ix_f = HybridIndex::new(&docs, emb[..ndoc * dim].to_vec(), dim, HybridConfig::default()).unwrap().with_fast_kernels(true);
        let ix_q = ix_f.clone().with_int8_dense(true);
        let qs: Vec<(u64, usize)> = qmap.into_iter().collect();
        let (mut rf, mut rq, mut ov, mut t1) = (vec![], vec![], 0.0f64, 0.0f64);
        let (mut tf, mut tq) = (vec![], vec![]);
        let (mut tpf, mut tpq, mut tiesf, mut tiesq) = (vec![], vec![], 0usize, 0usize);
        for &(q, tid) in &qs {
            let qv = &emb[tid * dim..(tid + 1) * dim];
            let t0 = Instant::now(); let a = ix_f.search(&texts[tid], qv, 1000).unwrap(); tf.push(t0.elapsed().as_secs_f64() * 1e6);
            let t0 = Instant::now(); let b = ix_q.search(&texts[tid], qv, 1000).unwrap(); tq.push(t0.elapsed().as_secs_f64() * 1e6);
            let rank = |l: &Vec<(usize, f32)>| l.iter().position(|(d, _)| rels[q as usize].contains(d)).map(|x| x as i64).unwrap_or(-1);
            rf.push(rank(&a)); rq.push(rank(&b));
            // R99c: tie-PESSIMISTIC rank (docs scoring >= the relevant doc, excluding itself; best over the relevant set)
            let tp = |ix: &HybridIndex| -> i64 { let mut sc = Vec::new(); ix.fused_scores(&texts[tid], qv, &mut sc).unwrap(); rels[q as usize].iter().map(|&d| sc.iter().filter(|&&x| x >= sc[d]).count() as i64 - 1).min().unwrap_or(-1) };
            let (pf, pq) = (tp(&ix_f), tp(&ix_q)); tpf.push(pf); tpq.push(pq); if pf != rank(&a) { tiesf += 1; } if pq != rank(&b) { tiesq += 1; }
            let sa: Vec<usize> = a.iter().take(10).map(|x| x.0).collect(); let sb: Vec<usize> = b.iter().take(10).map(|x| x.0).collect();
            ov += sb.iter().filter(|d| sa.contains(d)).count() as f64 / 10.0; if sa[0] == sb[0] { t1 += 1.0; }
        }
        tf.sort_by(|a, b| a.partial_cmp(b).unwrap()); tq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // latency of the dense part only
        let mut df = vec![]; let mut dq = vec![]; let mut sc = Vec::new();
        for &(_, tid) in qs.iter().take(300) { let qv = &emb[tid * dim..(tid + 1) * dim]; let t0 = Instant::now(); ix_f.dense_scores(qv, &mut sc).unwrap(); df.push(t0.elapsed().as_secs_f64() * 1e6); let t0 = Instant::now(); ix_q.dense_scores(qv, &mut sc).unwrap(); dq.push(t0.elapsed().as_secs_f64() * 1e6); }
        df.sort_by(|a, b| a.partial_cmp(b).unwrap()); dq.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("corpus {tag}: N={ndoc}, {} unique queries; hybrid search(k=1000) p50 f32-FAST {:.0} us vs int8 {:.0} us (x{:.2}); dense_scores only p50 {:.0} vs {:.0} us (x{:.2}); top-10 overlap {:.4}, top-1 agree {:.4}",
            qs.len(), tf[tf.len() / 2], tq[tq.len() / 2], tf[tf.len() / 2] / tq[tq.len() / 2], df[df.len() / 2], dq[dq.len() / 2], df[df.len() / 2] / dq[dq.len() / 2], ov / qs.len() as f64, t1 / qs.len() as f64);
        println!("  tie-pessimistic rank differs from list rank in {tiesf} (f32) / {tiesq} (int8) of {} queries", qs.len());
        out.insert(tag.to_string(), json!({"rank_tp_f32": tpf, "rank_tp_i8": tpq, "q": qs.iter().map(|x| x.0).collect::<Vec<_>>(), "rank_f32": rf, "rank_i8": rq}));
    }
    std::fs::write("results/r91_ranks_tp.json", serde_json::to_string(&Value::Object(out)).unwrap()).unwrap();
}
