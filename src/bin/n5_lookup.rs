//! R52: lookup-claim normalisation. results/n5_lookup_items.json -> per (world, seed) accuracy by claim kind for DEFAULT (exact), NORM (capital+author tables
//! normalised, elements exact) and NORM_ALL (also elements: documented hazard). accuracy = verdict == ground truth (accept / reject).
use neural_mesh::tools::{LookupTool, Verifier};
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn main() {
    let j: Value = serde_json::from_slice(&std::fs::read("results/n5_lookup_items.json").unwrap()).unwrap();
    let mut out = serde_json::Map::new();
    for (k, arr) in j.as_object().unwrap() {
        let mut lk = LookupTool::new();
        for it in arr.as_array().unwrap() {
            lk.insert(it["table"].as_str().unwrap(), it["key"].as_str().unwrap(), it["value"].as_str().unwrap());
        }
        let modes: [(&str, Verifier); 3] = [
            ("default", Verifier::new(lk.clone())),
            ("norm", Verifier::new(lk.clone()).with_normalized_lookup(&["capital", "author"])),
            ("norm_all", Verifier::new(lk.clone()).with_normalized_lookup(&["capital", "author", "element"])),
        ];
        let mut mm = serde_json::Map::new();
        for (name, v) in modes {
            let mut agg: BTreeMap<String, (f64, f64)> = BTreeMap::new();
            for it in arr.as_array().unwrap() {
                let kind = format!("{}_{}", if it["table"] == "element" { "elem" } else { "name" }, it["kind"].as_str().unwrap());
                let want = it["want"].as_str().unwrap() == "accept";
                let acc = matches!(v.verify("lookup", it["spec"].as_str().unwrap(), it["claim"].as_str().unwrap()), Ok((s, _)) if s == "accept");
                let e = agg.entry(kind).or_insert((0.0, 0.0));
                e.0 += 1.0;
                e.1 += (acc == want) as u8 as f64;
            }
            let mut m = serde_json::Map::new();
            for (kind, (n, ok)) in agg { m.insert(kind, json!(ok / n)); }
            mm.insert(name.into(), Value::Object(m));
        }
        out.insert(k.clone(), Value::Object(mm));
    }
    std::fs::write("results/n5_lookup.json", serde_json::to_string(&Value::Object(out)).unwrap()).unwrap();
}
