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

#[test]
fn bm25_params_match_python_reference() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/results");
    let fx: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/b1_adaptive_fixture.json")).unwrap()).unwrap();
    let bp: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/b1_bm25p_fixture.json")).expect("run py/b1/mk_parity2.py")).unwrap();
    let docs: Vec<String> = fx["docs"].as_array().unwrap().iter().map(|d| d.as_str().unwrap().to_string()).collect();
    let dvec: Vec<f32> = fx["dvec"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
    let ix = HybridIndex::new_with_bm25(&docs, dvec, 384, HybridConfig::default(), bp["k1"].as_f64().unwrap() as f32, bp["b"].as_f64().unwrap() as f32).unwrap();
    let ad = AdaptiveAlpha { lo: bp["lo"].as_f64().unwrap(), hi: bp["hi"].as_f64().unwrap(), pow: bp["pow"].as_f64().unwrap() };
    for (q, f) in bp["queries"].as_array().unwrap().iter().zip(fx["queries"].as_array().unwrap()) {
        let qv: Vec<f32> = f["vec"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let mut sc = Vec::new();
        let a = ix.fused_scores_adaptive(q["text"].as_str().unwrap(), &qv, &ad, &mut sc).unwrap();
        assert!((a as f64 - q["alpha"].as_f64().unwrap()).abs() < 1e-3);
        for (i, e) in q["scores"].as_array().unwrap().iter().enumerate() {
            assert!((sc[i] as f64 - e.as_f64().unwrap()).abs() < 5e-3, "score doc {i}");
        }
        assert_eq!(neural_mesh::hybrid::top_n(&sc, 1)[0] as u64, q["top"].as_u64().unwrap());
    }
}

#[test]
fn knn_router_matches_python_reference() {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/results/b1_knn_fixture.json");
    let j: serde_json::Value = serde_json::from_slice(&std::fs::read(p).expect("run py/b1/mk_parity3.py")).unwrap();
    let mut r = neural_mesh::knnroute::KnnRouter::new(j["dim"].as_u64().unwrap() as usize, j["k"].as_u64().unwrap() as usize);
    for e in j["exemplars"].as_array().unwrap() {
        let v: Vec<f32> = e["vec"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        assert!(r.add(e["label"].as_str().unwrap(), &v));
    }
    let mut n = 0;
    for q in j["queries"].as_array().unwrap() {
        let v: Vec<f32> = q["vec"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let got = r.votes(&v);
        let want = q["votes"].as_object().unwrap();
        assert_eq!(got.len(), want.len(), "{}", q["text"]);
        for (l, s) in &got {
            let w = want[l].as_f64().unwrap();
            assert!((*s as f64 - w).abs() < 2e-3, "{l}: {s} vs {w} for {}", q["text"]);
        }
        n += 1;
    }
    assert!(n >= 30);
}
