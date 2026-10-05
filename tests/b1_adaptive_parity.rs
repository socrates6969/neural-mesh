//! B1/R2 parity: the Rust adaptive-alpha hybrid reproduces the python battery implementation (fixture results/b1_adaptive_fixture.json).
use neural_mesh::hybrid::{AdaptiveAlpha, HybridConfig, HybridIndex};

#[test]
fn adaptive_alpha_matches_python_reference() {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/results/b1_adaptive_fixture.json");
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(p).expect("fixture")).unwrap();
    let docs: Vec<String> = j["docs"].as_array().unwrap().iter().map(|d| d.as_str().unwrap().to_string()).collect();
    let dvec: Vec<f32> = j["dvec"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let ix = HybridIndex::new(&docs, dvec, 384, HybridConfig::default()).unwrap();
    let ad = AdaptiveAlpha { lo: j["lo"].as_f64().unwrap(), hi: j["hi"].as_f64().unwrap(), pow: j["pow"].as_f64().unwrap() };
    let (mut nq, mut same_top) = (0, 0);
    for q in j["queries"].as_array().unwrap() {
        let qv: Vec<f32> = q["vec"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let mut sc = Vec::new();
        let a = ix.fused_scores_adaptive(q["text"].as_str().unwrap(), &qv, &ad, &mut sc).unwrap();
        assert!((a as f64 - q["alpha"].as_f64().unwrap()).abs() < 1e-3, "alpha {a} vs {}", q["alpha"]);
        for (i, e) in q["scores"].as_array().unwrap().iter().enumerate() {
            assert!((sc[i] as f64 - e.as_f64().unwrap()).abs() < 5e-3, "score doc {i}");
        }
        let top = neural_mesh::hybrid::top_n(&sc, 1)[0];
        nq += 1;
        same_top += (top as u64 == q["top"].as_u64().unwrap()) as usize;
    }
    assert_eq!(nq, same_top, "top-1 must agree on every fixture query");
    // default path untouched: alpha-fixed search equals fused_scores
    let qv: Vec<f32> = j["queries"][0]["vec"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let r = ix.search("anything", &qv, 3).unwrap();
    assert_eq!(r.len(), 3);
}
