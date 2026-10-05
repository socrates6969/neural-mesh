//! R111: AVX512-VNNI int8 encoder kernel vs the AVX2 (cvtepi8+madd) kernel: bit-identity of the embeddings and ms/query, interleaved A/B reps (box noise cancels in the ratio).
//! Usage: n19_vnni <weights.f32> <tokenizer.json>
use neural_mesh::encoder::{set_vnni, Encoder};
use std::time::Instant;
fn med(v: &mut Vec<f64>) -> f64 { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let texts: Vec<String> = serde_json::from_slice(&std::fs::read("results/h10_texts_A.json").unwrap()).unwrap();
    let qs: Vec<String> = (0..192).map(|i| texts[(i * 37) % texts.len()].split_whitespace().take(12).collect::<Vec<_>>().join(" ")).collect();
    let refs: Vec<&str> = qs.iter().map(|s| s.as_str()).collect();
    for thr in [1usize, 4] {
        let e8 = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap().with_int8(true).with_threads(thr);
        set_vnni(true); let v_on = e8.encode_batch(&refs[..32]);
        set_vnni(false); let v_off = e8.encode_batch(&refs[..32]);
        println!("threads {thr}: embeddings bit-identical VNNI vs AVX2 int8: {}", v_on == v_off);
        for bs in [1usize, 16, 64] {
            let (mut on, mut off) = (vec![], vec![]);
            for _rep in 0..7 {
                for (flag, sink) in [(true, &mut on), (false, &mut off)] {
                    set_vnni(flag);
                    let t0 = Instant::now(); let mut n = 0;
                    for c in refs.chunks(bs) { let _ = e8.encode_batch(c); n += c.len(); if n >= 96 { break; } }
                    sink.push(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
                }
            }
            let (m_on, m_off) = (med(&mut on), med(&mut off));
            println!("  int8 threads {thr} bs={bs}: AVX2 {m_off:.3} ms/q | VNNI {m_on:.3} ms/q | speedup x{:.2}", m_off / m_on);
        }
    }
    set_vnni(true);
    let e32 = Encoder::load(a[1].as_ref(), a[2].as_ref()).unwrap();
    for bs in [1usize, 16] { let t0 = Instant::now(); let mut n = 0; for c in refs.chunks(bs) { let _ = e32.encode_batch(c); n += c.len(); if n >= 96 { break; } } println!("  (ref) f32 1 thread bs={bs}: {:.3} ms/q", t0.elapsed().as_secs_f64() * 1e3 / n as f64); }
}
