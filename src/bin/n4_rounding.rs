//! R51: rounded-claim acceptance. results/n4_rounding_items.json (python-Fractions ground truth) -> per (world, seed) rates for the DEFAULT verifier
//! (exact rational equality) and the opt-in `with_rounded_claims(true)` verifier. kinds: exact (must accept), round2/round3 (correct roundings: want accept),
//! near / trunc / gross (wrong claims: want reject). Output results/n4_rounding.json
use neural_mesh::tools::{LookupTool, Verifier};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn main() {
    let j: Value = serde_json::from_slice(&std::fs::read("results/n4_rounding_items.json").unwrap()).unwrap();
    let mut out = serde_json::Map::new();
    for (k, arr) in j.as_object().unwrap() {
        let mut modes = serde_json::Map::new();
        for (mname, on) in [("default", false), ("rounded", true)] {
            let v = Verifier::new(LookupTool::new()).with_rounded_claims(on);
            let mut agg: BTreeMap<String, (f64, f64)> = BTreeMap::new();
            for it in arr.as_array().unwrap() {
                let kind = it["kind"].as_str().unwrap().to_string();
                let want_accept = it["want"].as_str().unwrap() == "accept";
                let r = v.verify(it["tool"].as_str().unwrap(), it["spec"].as_str().unwrap(), it["claim"].as_str().unwrap());
                let accepted = matches!(&r, Ok((s, _)) if s == "accept");
                let e = agg.entry(kind).or_insert((0.0, 0.0));
                e.0 += 1.0;
                e.1 += (accepted == want_accept) as u8 as f64;
            }
            let mut m = serde_json::Map::new();
            for (kind, (n, ok)) in agg {
                m.insert(kind, json!(ok / n));
            }
            modes.insert(mname.into(), Value::Object(m));
        }
        out.insert(k.clone(), Value::Object(modes));
    }
    std::fs::write("results/n4_rounding.json", serde_json::to_string(&Value::Object(out)).unwrap()).unwrap();
}
