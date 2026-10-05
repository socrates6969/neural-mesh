//! R116: persistent worker pool (`Encoder::with_pool`) vs per-layer `thread::scope` spawns: bit-identity of the embeddings and ms/query, interleaved A/B reps (box noise cancels in the ratio).
//! Second table: pool + lower split threshold (min tokens 16 / 8) at small batches.
//! Usage: n20_pool <weights.f32> <tokenizer.json>
use neural_mesh::encoder::Encoder;
use std::time::Instant;
fn med(v: &mut Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read("results/h10_texts_A.json").unwrap()).unwrap();
    let qs: Vec<String> = (0..192).map(|i| texts[(i * 37) % texts.len()].split_whitespace().take(12).collect::<Vec<_>>().join(" ")).collect();
    let refs: Vec<&str> = qs.iter().map(|s| s.as_str()).collect();
    for int8 in [true, false] {
        for thr in [2usize, 4, 8] {
            let es = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap().with_int8(int8).with_threads(thr);
            let ep = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap().with_int8(int8).with_threads(thr).with_pool(true);
            println!("{} threads {thr}: embeddings bit-identical pool vs scope: {}", if int8 { "int8" } else { "f32 " }, es.encode_batch(&refs[..64]) == ep.encode_batch(&refs[..64]));
            for bs in [16usize, 64] {
                let (mut sc, mut po) = (vec![], vec![]);
                for _rep in 0..9 {
                    for (is_pool, sink) in [(false, &mut sc), (true, &mut po)] {
                        let e = if is_pool { &ep } else { &es };
                        let t0 = Instant::now(); let mut n = 0;
                        for c in refs.chunks(bs) { let _ = e.encode_batch(c); n += c.len(); if n >= 128 { break; } }
                        sink.push(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
                    }
                }
                let (m_sc, m_po) = (med(&mut sc), med(&mut po));
                println!("  {} thr {thr} bs={bs}: scope {m_sc:.3} | pool {m_po:.3} ms/q | x{:.2}", if int8 { "int8" } else { "f32 " }, m_sc / m_po);
            }
        }
    }
    println!("--- pool with lower split threshold (small batches), int8 ---");
    for thr in [2usize, 4] {
        let base = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap().with_int8(true).with_threads(thr).with_pool(true);
        let low = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap().with_int8(true).with_threads(thr).with_pool(true).with_pool_min_tokens(8);
        println!("thr {thr}: low-threshold embeddings bit-identical: {}", base.encode_batch(&refs[..64]) == low.encode_batch(&refs[..64]));
        for bs in [1usize, 2, 4, 8] {
            let (mut b0, mut b1) = (vec![], vec![]);
            for _rep in 0..9 {
                for (lowflag, sink) in [(false, &mut b0), (true, &mut b1)] {
                    let e = if lowflag { &low } else { &base };
                    let t0 = Instant::now(); let mut n = 0;
                    for c in refs.chunks(bs) { let _ = e.encode_batch(c); n += c.len(); if n >= 96 { break; } }
                    sink.push(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
                }
            }
            let (m0, m1) = (med(&mut b0), med(&mut b1));
            println!("  thr {thr} bs={bs}: min48 {m0:.3} | min8 {m1:.3} ms/q | x{:.2}", m0 / m1);
        }
    }
}
