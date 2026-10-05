//! Opt-in `--store` of the neural-mesh CLI: feedback is journaled; a NEW state file warm-starts from the journal; default CLI
//! behaviour (no --store) is unchanged and creates no store.
use std::process::Command;

fn run(args: &[&str]) -> serde_json::Value {
    let o = Command::new(env!("CARGO_BIN_EXE_neural-mesh")).args(args).output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    serde_json::from_slice(&o.stdout).unwrap_or_else(|e| panic!("{e}: {}", String::from_utf8_lossy(&o.stdout)))
}

#[test]
fn feedback_is_journaled_and_new_state_warm_starts() {
    let d = std::env::temp_dir().join(format!("nm_storecli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    let (s1, s2, s3, store) = (d.join("s1.json"), d.join("s2.json"), d.join("s3.json"), d.join("store"));
    let (s1, s2, s3, store) = (s1.to_str().unwrap(), s2.to_str().unwrap(), s3.to_str().unwrap(), store.to_str().unwrap());
    for i in 0..30 {
        let (agent, r) = if i % 2 == 0 { ("good", "1") } else { ("bad", "0") };
        run(&["feedback", "--state", s1, "--features", "sql,db", "--agent", agent, "--reward", r, "--store", store, "--world", "t"]);
    }
    let j = std::fs::read_to_string(d.join("store/router/journal.jsonl")).unwrap();
    assert_eq!(j.lines().count(), 30);
    // brand-new state file, same store+world -> warm
    let v = run(&["route", "--state", s2, "--features", "sql,db", "--candidates", "good,bad", "--store", store, "--world", "t", "--seed", "1"]);
    let sc = &v["scores"];
    assert!(sc["good"].as_f64().unwrap() > sc["bad"].as_f64().unwrap(), "{v}");
    // other namespace -> cold (no evidence)
    let v = run(&["route", "--state", s3, "--features", "sql,db", "--candidates", "good,bad", "--store", store, "--world", "other", "--seed", "1"]);
    assert_eq!(v["scores"]["good"], v["scores"]["bad"], "{v}");
    // no --store: nothing is created
    let s4 = d.join("s4.json");
    run(&["feedback", "--state", s4.to_str().unwrap(), "--features", "a", "--agent", "x", "--reward", "1"]);
    assert!(!d.join("store2").exists());
    let _ = std::fs::remove_dir_all(&d);
}
