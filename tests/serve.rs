//! Integration tests for `neural-mesh serve` (stdio and TCP).

mod common;

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

use common::*;
use serde_json::{Value, json};

const NEVER: &str = "600000";

fn lock_path(p: &std::path::Path) -> std::path::PathBuf {
    p.with_file_name("state.json.lock")
}

#[test]
fn stdio_round_trip_matches_one_shot_cli() {
    let p1 = tmp_state("parity-a");
    let p2 = tmp_state("parity-b");
    for p in [&p1, &p2] {
        for _ in 0..10 {
            cli_feedback(p, "alice", "1");
            cli_feedback(p, "bob", "0");
        }
    }
    let mut s = Srv::stdio(&p1, &[]);
    let req = r#"{"cmd":"route","features":"x,y","candidates":["alice","bob","carol"],"team":2}"#;
    let got = s.req(req);
    let want = cli(&["route", "--state", p2.to_str().unwrap(), "--features", "x,y", "--candidates", "alice,bob,carol", "--team", "2"]);
    assert_eq!(got, want, "unseeded route must match the CLI exactly");
    assert_eq!(got["ok"], true);

    let got = s.req(r#"{"cmd":"route","features":"x","seed":7,"id":42}"#);
    let mut want = cli(&["route", "--state", p2.to_str().unwrap(), "--features", "x", "--seed", "7"]);
    want["id"] = json!(42);
    assert_eq!(got, want, "seeded route + echoed id");

    let got = s.req(r#"{"cmd":"feedback","features":"x","agent":"bob","team":"carol,bob","reward":0.8,"type":"t"}"#);
    let want = cli(&["feedback", "--state", p2.to_str().unwrap(), "--features", "x", "--agent", "bob", "--team", "carol,bob", "--reward", "0.8", "--type", "t"]);
    assert_eq!(got, want);

    assert_eq!(s.req(r#"{"cmd":"stats"}"#), cli_stats(&p2));
    assert!(s.close().success());
    assert_eq!(cli_stats(&p1), cli_stats(&p2), "state persisted on EOF equals the one-shot state");
}

#[test]
fn malformed_lines_get_errors_and_server_keeps_running() {
    let p = tmp_state("malformed");
    let mut s = Srv::stdio(&p, &[]);
    for bad in [
        "{bad",
        "[1,2]",
        "42",
        r#"{"features":"a"}"#,
        r#"{"cmd":7}"#,
        r#"{"cmd":"nope"}"#,
        r#"{"cmd":"route","features":"a"}"#,
        r#"{"cmd":"route","features":"a","candidates":"zz","team":0}"#,
        r#"{"cmd":"route","features":"a","candidates":"zz","bogus":1}"#,
        r#"{"cmd":"route","features":{"a":1},"candidates":"zz"}"#,
        r#"{"cmd":"feedback","features":"a","agent":"x","reward":2}"#,
        r#"{"cmd":"feedback","features":"a","agent":"x"}"#,
        r#"{"cmd":"stats","state":"/etc/passwd"}"#,
    ] {
        let r = s.req(bad);
        assert_eq!(r["ok"], false, "{bad} -> {r}");
        assert!(r["error"].as_str().is_some_and(|e| !e.is_empty()), "{bad} -> {r}");
    }
    s.send_raw(&[0xff, 0xfe, b'\n']);
    assert_eq!(s.read()["ok"], false, "invalid UTF-8");
    s.send_raw(b"\n   \n"); // blank lines: no response
    let r = s.feedback("good", 1.0);
    assert_eq!(r["ok"], true, "{r}");
    let st = s.req(r#"{"cmd":"stats"}"#);
    let names: Vec<&str> = st["agents"].as_array().unwrap().iter().map(|a| a["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["good"], "failed requests must not register agents or change state");
    assert_eq!(st["feedbacks"], 1);
    assert!(s.close().success());
}

#[test]
fn write_behind_flushes_on_eof_when_thresholds_not_hit() {
    let p = tmp_state("eof");
    let mut s = Srv::stdio(&p, &["--flush-every", "1000", "--flush-ms", NEVER]);
    for _ in 0..5 {
        assert_eq!(s.feedback("a", 1.0)["ok"], true);
    }
    assert_eq!(cli_stats(&p)["feedbacks"], 0, "nothing flushed yet (write-behind)");
    assert!(s.close().success());
    let st = cli_stats(&p);
    assert_eq!(st["feedbacks"], 5, "EOF flushes");
    assert!(!lock_path(&p).exists(), "lock released");
}

#[test]
fn flushes_every_n_mutations() {
    let p = tmp_state("count");
    let mut s = Srv::stdio(&p, &["--flush-every", "3", "--flush-ms", NEVER]);
    s.feedback("a", 1.0);
    s.feedback("a", 1.0);
    assert_eq!(cli_stats(&p)["feedbacks"], 0);
    s.feedback("a", 1.0);
    assert_eq!(cli_stats(&p)["feedbacks"], 3, "third mutation triggers the flush");
    s.feedback("a", 1.0);
    assert_eq!(cli_stats(&p)["feedbacks"], 3);
    assert!(s.close().success());
    assert_eq!(cli_stats(&p)["feedbacks"], 4);
}

#[test]
fn flushes_after_flush_ms_without_more_traffic() {
    let p = tmp_state("timer");
    let mut s = Srv::stdio(&p, &["--flush-every", "1000", "--flush-ms", "80"]);
    s.feedback("a", 1.0);
    let t = Instant::now();
    loop {
        if cli_stats(&p)["feedbacks"] == 1 {
            break;
        }
        assert!(t.elapsed() < Duration::from_secs(5), "timer flush never happened");
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(s.close().success());
}

#[test]
fn flush_and_shutdown_commands() {
    let p = tmp_state("ctl");
    let mut s = Srv::stdio(&p, &["--flush-every", "1000", "--flush-ms", NEVER]);
    s.feedback("a", 1.0);
    let r = s.req(r#"{"cmd":"flush"}"#);
    assert_eq!((r["ok"].clone(), r["pending"].clone()), (json!(true), json!(0)));
    assert_eq!(cli_stats(&p)["feedbacks"], 1);
    s.feedback("a", 1.0);
    let r = s.req(r#"{"cmd":"shutdown"}"#);
    assert_eq!(r["shutdown"], true);
    assert!(s.child.wait().unwrap().success());
    assert_eq!(cli_stats(&p)["feedbacks"], 2);
}

#[test]
fn cli_calls_between_server_flushes_lose_nothing() {
    let p = tmp_state("interleave");
    let mut s = Srv::stdio(&p, &["--flush-every", "1000", "--flush-ms", NEVER]);
    for _ in 0..3 {
        assert_eq!(s.feedback("a", 1.0)["ok"], true); // pending in the server's journal
    }
    for _ in 0..2 {
        assert_eq!(cli_feedback(&p, "b", "0")["ok"], true); // one-shot writes the file directly
    }
    // The next request notices the changed file, merges both sides and writes.
    let st = s.req(r#"{"cmd":"stats"}"#);
    assert_eq!(st["feedbacks"], 5, "server sees CLI updates and keeps its own: {st}");
    assert_eq!(st["updates"], 5);
    let on_disk = cli_stats(&p);
    assert_eq!(on_disk["feedbacks"], 5, "merged state was flushed");
    assert_eq!(on_disk["updates"], 5);
    // CLI again after the merge, then more server work, then EOF.
    cli_feedback(&p, "b", "1");
    s.feedback("a", 0.0);
    assert!(s.close().success());
    let end = cli_stats(&p);
    assert_eq!(end["feedbacks"], 7);
    assert_eq!(end["updates"], 7);
    let names: Vec<&str> = end["agents"].as_array().unwrap().iter().map(|a| a["name"].as_str().unwrap()).collect();
    // merge order is "CLI call first, pending server requests after"
    assert_eq!(names, vec!["b", "a"]);
}

#[test]
fn concurrent_cli_and_server_traffic_loses_no_updates() {
    let p = tmp_state("race");
    let mut s = Srv::stdio(&p, &["--flush-every", "7", "--flush-ms", "30"]);
    let pc = p.clone();
    let cli_thread = std::thread::spawn(move || {
        for _ in 0..25 {
            assert_eq!(cli_feedback(&pc, "c", "1")["ok"], true);
        }
    });
    for _ in 0..60 {
        assert_eq!(s.feedback("s", 1.0)["ok"], true);
        std::thread::sleep(Duration::from_millis(2));
    }
    cli_thread.join().unwrap();
    assert!(s.close().success());
    let end = cli_stats(&p);
    assert_eq!(end["feedbacks"], 85, "all 25 CLI + 60 server feedbacks survive");
    assert_eq!(end["updates"], 85);
    assert!(!lock_path(&p).exists());
}

#[test]
fn tcp_mode_is_localhost_and_answers_requests() {
    let p = tmp_state("tcp");
    let mut s = Srv::spawn(&p, &["--port", "0", "--flush-every", "1"]);
    let mut banner = String::new();
    s.out.read_line(&mut banner).unwrap();
    let b: Value = serde_json::from_str(&banner).unwrap();
    let addr = b["listening"].as_str().unwrap().to_string();
    assert!(addr.starts_with("127.0.0.1:"), "{addr}");
    let conn = TcpStream::connect(&addr).unwrap();
    let mut rd = BufReader::new(conn.try_clone().unwrap());
    let mut wr = conn;
    let mut ask = |line: &str| -> Value {
        writeln!(wr, "{line}").unwrap();
        let mut l = String::new();
        rd.read_line(&mut l).unwrap();
        serde_json::from_str(&l).unwrap()
    };
    assert_eq!(ask(r#"{"cmd":"feedback","features":"x","agent":"a","reward":1}"#)["ok"], true);
    assert_eq!(ask("not json")["ok"], false);
    assert_eq!(ask(r#"{"cmd":"route","features":"x"}"#)["agent"], "a");
    assert_eq!(cli_stats(&p)["feedbacks"], 1, "flush-every 1 persisted immediately");
    assert_eq!(ask(r#"{"cmd":"shutdown"}"#)["shutdown"], true);
    assert!(s.child.wait().unwrap().success());
}

#[test]
fn bad_serve_arguments_are_usage_errors() {
    let p = tmp_state("args");
    let ps = p.to_str().unwrap();
    for args in [
        vec!["serve", "--state", ps],
        vec!["serve", "--stdio"],
        vec!["serve", "--state", ps, "--stdio", "--port", "0"],
        vec!["serve", "--state", ps, "--stdio", "--flush-every", "0"],
        vec!["serve", "--state", ps, "--port", "99999"],
    ] {
        let out = std::process::Command::new(BIN).args(&args).output().unwrap();
        assert_eq!(out.status.code(), Some(1), "{args:?}");
        let v: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(v["ok"], false);
    }
}

#[test]
fn one_shot_output_is_unchanged() {
    // Guards the byte-compat promise for the existing commands.
    let out = std::process::Command::new(BIN).arg("--help").output().unwrap();
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(!v["usage"].as_str().unwrap().contains("serve"));
    let p = tmp_state("compat");
    let r = cli(&["route", "--state", p.to_str().unwrap(), "--features", "a"]);
    assert_eq!(r["error"], "no agents registered yet; pass --candidates x,y");
    assert!(!p.exists(), "failed route creates no state file");
}
