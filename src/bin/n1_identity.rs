//! R39: identity / near-miss false-ACCEPT axis. For each (world, seed) item set (results/n1_identity_items.json; kinds true / near / diff)
//! run the Verifier in DEFAULT mode (integer grid) and STRICT mode (R39 opt-in probes) and report per-seed rates.
//! Output results/n1_identity.json { "<world><seed>": { mode: { true_acc, near_rej, diff_rej, near_differs_acc, true_differs_rej, us } } }
use neural_mesh::tools::{LookupTool, Verifier};
use serde_json::{json, Value};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let items_path = args.get(1).cloned().unwrap_or("results/n1_identity_items.json".into());
    let out_path = args.get(2).cloned().unwrap_or("results/n1_identity.json".into());
    let j: Value = serde_json::from_slice(&std::fs::read(&items_path).unwrap()).unwrap();
    let mut out = serde_json::Map::new();
    for (k, arr) in j.as_object().unwrap() {
        let mut modes = serde_json::Map::new();
        for (mname, strict) in [("default", false), ("strict", true)] {
            let v = Verifier::new(LookupTool::new()).with_strict_identity(strict);
            let (mut ta, mut tn, mut nr, mut nn, mut dr, mut dn, mut nda, mut tdr) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
            let (mut er, mut en) = (0f64, 0f64);
            let mut us = 0f64;
            let mut calls = 0f64;
            for it in arr.as_array().unwrap() {
                let kind = it["kind"].as_str().unwrap();
                let spec = it["spec"].as_str().unwrap();
                let t0 = Instant::now();
                let ident = v.verify("identity", spec, "identical");
                let dif = v.verify("identity", spec, "differs");
                us += t0.elapsed().as_secs_f64() * 1e6;
                calls += 2.0;
                let acc = |r: &Result<(String, u32), _>| matches!(r, Ok((s, _)) if s == "accept");
                match kind {
                    "true" => { tn += 1.0; ta += acc(&ident) as u8 as f64; tdr += (!acc(&dif)) as u8 as f64; }
                    "near" => { nn += 1.0; nr += (!acc(&ident)) as u8 as f64; nda += acc(&dif) as u8 as f64; }
                    "evade" => { en += 1.0; er += (!acc(&ident)) as u8 as f64; }
                    _ => { dn += 1.0; dr += (!acc(&ident)) as u8 as f64; }
                }
            }
            modes.insert(mname.into(), json!({"true_acc": ta / tn, "near_rej": nr / nn, "diff_rej": dr / dn, "near_differs_acc": nda / nn, "true_differs_rej": tdr / tn, "evade_rej": er / en, "us": us / calls}));
        }
        out.insert(k.clone(), Value::Object(modes));
    }
    std::fs::write(out_path, serde_json::to_string(&Value::Object(out)).unwrap()).unwrap();
}
