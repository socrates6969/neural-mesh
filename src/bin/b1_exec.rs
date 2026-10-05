//! B1 battery, layer L2: run every tool-solvable item through the REAL registry (exec) and the REAL Verifier
//! (claim accept/reject). Output: results/b1_exec.json  { "<world><seed>": { skill: [n, exec_ok, claim_ok, mean_us, mean_cost] } }
use neural_mesh::tools::{LookupTool, Registry};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let items_path = args.get(1).cloned().unwrap_or("results/b1_items.json".into());
    let out_path = args.get(2).cloned().unwrap_or("results/b1_exec.json".into());
    let j: Value = serde_json::from_slice(&std::fs::read(&items_path).unwrap()).unwrap();
    let mut reg = Registry::with_defaults(LookupTool::new(), false);
    let mut out = serde_json::Map::new();
    let mut gen_fail = 0usize;
    let mut total = 0usize;
    for (k, arr) in j["items"].as_object().unwrap() {
        let mut agg: BTreeMap<String, (u64, u64, u64, f64, f64)> = BTreeMap::new();
        for it in arr.as_array().unwrap() {
            let tool = match it.get("tool").and_then(|t| t.as_str()) { Some(t) => t, None => continue };
            let spec = it["spec"].as_str().unwrap();
            let ans = it["ans"].as_str().unwrap();
            let bad = it["bad"].as_str().unwrap();
            let sk = it["skill"].as_str().unwrap().to_string();
            let t0 = Instant::now();
            let o = reg.call(tool, spec);
            let us = t0.elapsed().as_secs_f64() * 1e6;
            let exec_ok = matches!(&o.result, Ok(s) if s == ans);
            if !exec_ok { gen_fail += 1; if gen_fail <= 8 { eprintln!("GEN/EXEC MISMATCH {sk} tool={tool} spec={spec} want={ans} got={:?}", o.result); } }
            total += 1;
            let vin = |c: &str| format!("{tool}\n{spec}\n{c}");
            let va = reg.call("verify", &vin(ans));
            let vb = reg.call("verify", &vin(bad));
            let acc = matches!(&va.result, Ok(s) if s == "accept");
            let rej = matches!(&vb.result, Ok(s) if s.starts_with("reject"));
            let e = agg.entry(sk).or_insert((0, 0, 0, 0.0, 0.0));
            e.0 += 1; e.1 += exec_ok as u64; e.2 += (acc as u64 + rej as u64); e.3 += us; e.4 += o.cost as f64;
        }
        let mut m = serde_json::Map::new();
        for (sk, e) in agg { m.insert(sk, json!([e.0, e.1, e.2, e.3 / e.0 as f64, e.4 / e.0 as f64])); }
        out.insert(k.clone(), Value::Object(m));
    }
    eprintln!("exec mismatches {gen_fail} / {total}");
    std::fs::write(out_path, serde_json::to_string(&Value::Object(out)).unwrap()).unwrap();
}
