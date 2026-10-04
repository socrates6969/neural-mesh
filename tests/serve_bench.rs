//! Latency / throughput measurement: one-shot CLI vs stdio server.
//! Run: cargo test --release --test serve_bench -- --ignored --nocapture

mod common;

use std::process::Command;
use std::time::Instant;

use common::*;

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn report(name: &str, mut us: Vec<f64>) {
    let n = us.len();
    let (p50, p90, p99) = (pct(&mut us, 0.5), pct(&mut us, 0.9), pct(&mut us, 0.99));
    println!("{name}: n={n} p50={p50:.1}us p90={p90:.1}us p99={p99:.1}us max={:.1}us", us[n - 1]);
}

#[test]
#[ignore]
fn bench_one_shot_vs_server() {
    let p = tmp_state("bench");
    let agents = ["planner", "coder", "tester", "reviewer", "researcher", "ops", "docs", "sec"];
    let feats = ["rust", "sql", "api", "ui", "test", "docs", "perf", "bug", "infra", "ml"];
    {
        let mut s = Srv::stdio(&p, &["--flush-every", "100000", "--flush-ms", "600000"]);
        for i in 0..400 {
            let (a, f) = (agents[i % agents.len()], feats[i % feats.len()]);
            let line = format!(r#"{{"cmd":"feedback","features":"{f},{}","agent":"{a}","reward":{}}}"#, feats[(i + 3) % feats.len()], (i % 3) as f64 / 2.0);
            s.req(&line);
        }
        s.close();
    }
    println!("state file: {} bytes", std::fs::metadata(&p).unwrap().len());
    let ps = p.to_str().unwrap();

    let mut one = Vec::new();
    for i in 0..500 {
        let t = Instant::now();
        let out = Command::new(BIN).args(["route", "--state", ps, "--features", feats[i % feats.len()], "--team", "2"]).output().unwrap();
        one.push(t.elapsed().as_secs_f64() * 1e6);
        assert!(out.status.success());
    }
    report("one-shot route (process spawn)", one);

    let mut s = Srv::stdio(&p, &[]);
    let mut lat = Vec::new();
    for i in 0..5000 {
        let line = format!(r#"{{"cmd":"route","features":"{},{}","team":2}}"#, feats[i % feats.len()], feats[(i + 1) % feats.len()]);
        let t = Instant::now();
        let r = s.req(&line);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
        assert_eq!(r["ok"], true);
    }
    report("stdio server route (incl. write-behind flushes)", lat);

    let mut lat = Vec::new();
    let total = Instant::now();
    for i in 0..5000 {
        let line = format!(r#"{{"cmd":"feedback","features":"{}","agent":"{}","reward":0.5}}"#, feats[i % feats.len()], agents[i % agents.len()]);
        let t = Instant::now();
        s.req(&line);
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    println!("stdio server feedback throughput: {:.0} req/s (flush every 64)", 5000.0 / total.elapsed().as_secs_f64());
    report("stdio server feedback", lat);
    s.close();
}
