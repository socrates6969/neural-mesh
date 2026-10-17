//! Integration tests for the opt-in `nm_guarded` binary (Hebbwise R3 shadow arm). Scratch states live in the OS temp dir.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use neural_mesh::defaults::recommended_router;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::prior::Guarded;
use neural_mesh::router::{Rng, Router, Task};
use serde_json::{Value, json};

const BIN: &str = env!("CARGO_BIN_EXE_nm_guarded");
const AGENTS: [&str; 8] = ["fr-planner", "fr-builder", "fr-tester", "fr-reviewer", "fr-scriber", "fr-shipper", "general-purpose", "Explore"];
const TABLE: &str = r#"{"version":1,"rules":[
 {"agent":"fr-planner","keywords":["plan","spec"]},{"agent":"fr-builder","keywords":["build","fix","implement"]},
 {"agent":"fr-tester","keywords":["test","tests"]},{"agent":"fr-reviewer","keywords":["review","audit"]},
 {"agent":"fr-scriber","keywords":["docs","readme"]},{"agent":"fr-shipper","keywords":["ship","release"]},
 {"agent":"general-purpose","keywords":["research"]},{"agent":"Explore","keywords":["find","search","plan"]}]}"#;

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("nm-guarded-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct Srv {
    child: Child,
    stdin: Option<ChildStdin>,
    out: BufReader<ChildStdout>,
}
impl Srv {
    fn start(state: &Path, table: Option<&Path>) -> Srv {
        let mut c = Command::new(BIN);
        c.args(["--state", state.to_str().unwrap(), "--stdio"]);
        if let Some(t) = table {
            c.args(["--table", t.to_str().unwrap()]);
        }
        let mut child = c.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
        let stdin = child.stdin.take();
        let out = BufReader::new(child.stdout.take().unwrap());
        Srv { child, stdin, out }
    }
    fn req(&mut self, v: &Value) -> Value {
        self.raw(&v.to_string())
    }
    fn raw(&mut self, line: &str) -> Value {
        let w = self.stdin.as_mut().unwrap();
        w.write_all(format!("{line}\n").as_bytes()).unwrap();
        w.flush().unwrap();
        let mut s = String::new();
        self.out.read_line(&mut s).unwrap();
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("bad line {s:?}: {e}"))
    }
    fn close(mut self) {
        drop(self.stdin.take());
        assert!(self.child.wait().unwrap().success());
    }
}

fn table_file(dir: &Path) -> PathBuf {
    let p = dir.join("table.json");
    std::fs::write(&p, TABLE).unwrap();
    p
}

/// keyword -> slot index exactly as nm_guarded assigns them (table agents in rule order; first rule naming a keyword wins)
fn kw_map() -> HashMap<String, usize> {
    let mut m = HashMap::new();
    for (kw, a) in [("plan", 0), ("spec", 0), ("build", 1), ("fix", 1), ("implement", 1), ("test", 2), ("tests", 2), ("review", 3), ("audit", 3),
        ("docs", 4), ("readme", 4), ("ship", 5), ("release", 5), ("research", 6), ("find", 7), ("search", 7)] {
        m.insert(kw.to_string(), a);
    }
    m
}

const VOCAB: [&str; 24] = ["plan", "spec", "build", "fix", "implement", "test", "tests", "review", "audit", "docs", "readme", "ship", "release",
    "research", "find", "search", "parser", "api", "ci", "module", "schema", "cache", "auth", "ui"];

fn stream(n: usize, seed: u64) -> Vec<Vec<String>> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| {
        let k = 1 + r.below(5);
        let mut f: Vec<String> = Vec::new();
        for _ in 0..k {
            let w = VOCAB[r.below(VOCAB.len())].to_string();
            if !f.contains(&w) {
                f.push(w);
            }
        }
        f
    }).collect()
}

/// a fixed but agent/feature dependent success probability
fn truth(f: &[String], a: usize) -> f64 {
    let h = f.iter().map(|s| s.bytes().map(|b| b as usize).sum::<usize>()).sum::<usize>();
    if (h + a * 7) % 5 == 0 { 0.9 } else if a == 6 { 0.55 } else { 0.25 }
}

fn cands(i: usize) -> Vec<usize> {
    if i % 7 == 3 { (0..8).filter(|a| a % 2 == 0).collect() } else if i % 11 == 5 { vec![6, 1, 2] } else { (0..8).collect() }
}

/// Drive both the server and the in-process reference over steps [from, to); returns (picks, decision kinds).
fn drive(s: &mut Srv, reference: &mut Guarded<HebbianRouter>, tasks: &[Vec<String>], from: usize, to: usize, fb: &mut u64) {
    let mut reward_rng = Rng::new(424242 + from as u64);
    for i in from..to {
        let f = &tasks[i];
        let c = cands(i);
        let names: Vec<&str> = c.iter().map(|&a| AGENTS[a]).collect();
        let seed = 1000 + i as u64;
        let before = reference.decisions;
        let task = Task { id: *fb, task_type: None, features: f.clone() };
        let want = reference.select(&task, &c, &mut Rng::new(seed));
        let after = reference.decisions;
        let kind = (0..3).find(|&k| after[k] != before[k]).unwrap();
        let got = s.req(&json!({"cmd": "route", "features": f, "candidates": names, "seed": seed, "id": i}));
        assert_eq!(got["ok"], true, "{got}");
        assert_eq!(got["id"], json!(i));
        assert_eq!(got["agent"], json!(AGENTS[want]), "step {i}: decision diverged from recommended_router");
        let kind_name = ["keyword", "challenger", "inner"][kind];
        assert_eq!(got["decision"], json!(kind_name), "step {i}: decision kind");
        let r = if reward_rng.f64() < truth(f, want) { 1.0 } else { 0.0 };
        reference.update(&task, want, r);
        *fb += 1;
        let got = s.req(&json!({"cmd": "feedback", "fid": format!("out:t{i}"), "features": f, "agent": AGENTS[want], "reward": r}));
        assert_eq!(got["applied"], true, "{got}");
    }
}

#[test]
fn decisions_identical_to_recommended_router_and_state_round_trips_exactly() {
    let dir = tmp_dir("ident");
    let state = dir.join("guarded.json");
    let table = table_file(&dir);
    let tasks = stream(900, 77);
    let mut reference = recommended_router(kw_map(), 128);
    let mut fb = 0u64;

    let mut s = Srv::start(&state, Some(&table));
    let st = s.req(&json!({"cmd": "stats"}));
    assert_eq!(st["agents"], json!(AGENTS), "table agents take the first slots in rule order");
    drive(&mut s, &mut reference, &tasks, 0, 450, &mut fb);
    s.close();
    let saved = std::fs::read(&state).unwrap();

    // restart from the persisted state: exact round trip (re-saved bytes identical) and the stream continues identically
    let mut s = Srv::start(&state, Some(&table));
    assert_eq!(s.req(&json!({"cmd": "flush"}))["ok"], true);
    assert_eq!(std::fs::read(&state).unwrap(), saved, "state bytes changed by a load/save round trip");
    drive(&mut s, &mut reference, &tasks, 450, 900, &mut fb);
    let st = s.req(&json!({"cmd": "stats"}));
    assert_eq!(st["feedbacks"], json!(900));
    assert_eq!(st["applied"], json!(900));
    let d = reference.decisions;
    assert!(d[0] > 0 && d[2] > 0, "the stream exercised keyword and inner decisions: {d:?}");
    s.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn repeated_feedback_id_is_a_noop_and_route_is_pure() {
    let dir = tmp_dir("dedupe");
    let state = dir.join("g.json");
    let mut s = Srv::start(&state, Some(&table_file(&dir)));
    let fb = json!({"cmd": "feedback", "fid": "out:x1", "features": ["test", "parser"], "label": "fr-tester"});
    assert_eq!(s.req(&fb)["applied"], true);
    let bytes = std::fs::read(&state).unwrap();
    let st1 = s.req(&json!({"cmd": "stats"}));
    let again = s.req(&fb);
    assert_eq!((again["applied"].clone(), again["duplicate"].clone()), (json!(false), json!(true)));
    assert_eq!(s.req(&json!({"cmd": "stats"})), st1, "a duplicate feedback must not change anything");
    assert_eq!(std::fs::read(&state).unwrap(), bytes);
    // a different class with the same id is still a duplicate (ids are per dispatch)
    assert_eq!(s.req(&json!({"cmd": "feedback", "fid": "out:x1", "features": ["x"], "agent": "fr-tester", "reward": 0, "failure_class": "env"}))["duplicate"], true);
    // route: same request, same answer, state unchanged (known candidates)
    let r = json!({"cmd": "route", "features": ["test", "api"], "candidates": ["fr-tester", "fr-builder", "general-purpose"], "seed": 9});
    let a = s.req(&r);
    for _ in 0..5 {
        assert_eq!(s.req(&r), a);
    }
    assert_eq!(s.req(&json!({"cmd": "stats"})), st1);
    s.close();
    assert_eq!(std::fs::read(&state).unwrap(), bytes, "routes over known agents must not change the persisted state");
    // the duplicate survives a restart
    let mut s = Srv::start(&state, None);
    assert_eq!(s.req(&fb)["duplicate"], true);
    s.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn env_and_spec_classes_record_the_id_without_a_skill_update() {
    let dir = tmp_dir("classes");
    let state = dir.join("g.json");
    let mut s = Srv::start(&state, Some(&table_file(&dir)));
    let st0 = s.req(&json!({"cmd": "stats"}));
    for (i, class) in ["env", "spec"].iter().enumerate() {
        let got = s.req(&json!({"cmd": "feedback", "fid": format!("out:e{i}"), "features": ["test"], "agent": "fr-tester", "reward": 0, "failure_class": class}));
        assert_eq!((got["applied"].clone(), got["class"].clone()), (json!(true), json!(class)));
    }
    let st = s.req(&json!({"cmd": "stats"}));
    assert_eq!(st["updates"], st0["updates"]);
    assert_eq!(st["feedbacks"], json!(0));
    assert_eq!(st["applied"], json!(2));
    // a new agent named only in feedback is registered after the table agents
    s.req(&json!({"cmd": "feedback", "fid": "out:n1", "features": ["x"], "label": "ruflo-core:coder"}));
    assert_eq!(s.req(&json!({"cmd": "stats"}))["agents"][8], json!("ruflo-core:coder"));
    s.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn malformed_lines_get_errors_and_the_server_keeps_running() {
    let dir = tmp_dir("bad");
    let state = dir.join("g.json");
    let mut s = Srv::start(&state, Some(&table_file(&dir)));
    for bad in ["{bad", "[1,2]", "42", r#"{"features":"a"}"#, r#"{"cmd":7}"#, r#"{"cmd":"nope"}"#, r#"{"cmd":"route","features":"a"}"#,
        r#"{"cmd":"route","features":"a","candidates":["x"]}"#, r#"{"cmd":"route","features":"a","candidates":[],"seed":1}"#,
        r#"{"cmd":"route","features":"a","candidates":"x,","seed":-1}"#, r#"{"cmd":"feedback","features":"a","label":"x"}"#,
        r#"{"cmd":"feedback","fid":"f","features":"a","agent":"x","reward":2}"#, r#"{"cmd":"feedback","fid":"f","features":"a","label":"x","reward":1}"#,
        r#"{"cmd":"feedback","fid":"f","features":"a","agent":"x","reward":1,"failure_class":"cosmic"}"#, r#"{"cmd":"feedback","fid":"f","label":"a,b","features":"a"}"#] {
        let got = s.raw(bad);
        assert_eq!(got["ok"], false, "{bad} -> {got}");
        assert!(got["error"].is_string());
    }
    let got = s.raw(r#"{"cmd":"nope","id":"k"}"#);
    assert_eq!(got["id"], "k");
    // still serving, and nothing was registered or applied by the failed calls
    let st = s.req(&json!({"cmd": "stats"}));
    assert_eq!(st["applied"], json!(0));
    assert_eq!(st["agents"].as_array().unwrap().len(), 8);
    let ok = s.req(&json!({"cmd": "route", "features": "test", "candidates": "fr-tester,fr-builder", "seed": 1}));
    assert_eq!(ok["ok"], true);
    assert_eq!(s.req(&json!({"cmd": "shutdown"}))["shutdown"], true);
    assert!(s.child.wait().unwrap().success());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_different_table_for_an_existing_state_is_refused() {
    let dir = tmp_dir("mismatch");
    let state = dir.join("g.json");
    let s = Srv::start(&state, Some(&table_file(&dir)));
    s.close();
    let other = dir.join("other.json");
    std::fs::write(&other, r#"{"version":1,"rules":[{"agent":"x","keywords":["y"]}]}"#).unwrap();
    let out = Command::new(BIN).args(["--state", state.to_str().unwrap(), "--table", other.to_str().unwrap(), "--stdio"]).stdin(Stdio::null()).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stdout).contains("table_mismatch"));
    // the same table is accepted
    let s = Srv::start(&state, Some(&table_file(&dir)));
    s.close();
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fresh_states_are_byte_reproducible() {
    let dir = tmp_dir("repro");
    let table = table_file(&dir);
    let mut bodies = Vec::new();
    for i in 0..4 {
        let state = dir.join(format!("g{i}.json"));
        Srv::start(&state, Some(&table)).close();
        bodies.push(std::fs::read(&state).unwrap());
    }
    assert!(bodies.windows(2).all(|w| w[0] == w[1]), "fresh states differ between processes");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stale_temp_files_of_dead_writers_are_removed_at_startup() {
    let dir = tmp_dir("tmpclean");
    let state = dir.join("g.json");
    Srv::start(&state, Some(&table_file(&dir))).close();
    let stale = dir.join(".g.json.tmp.999999");
    let fresh = dir.join(".g.json.tmp.888888");
    let other = dir.join(".other.json.tmp.777777");
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(600);
    for p in [&stale, &other] {
        std::fs::write(p, b"partial").unwrap();
        std::fs::File::options().write(true).open(p).unwrap().set_modified(old).unwrap();
    }
    std::fs::write(&fresh, b"in flight").unwrap();
    let mut s = Srv::start(&state, None);
    assert_eq!(s.req(&json!({"cmd": "stats"}))["ok"], true);
    s.close();
    assert!(!stale.exists(), "a stale temp of a dead writer survived startup");
    assert!(fresh.exists(), "a just-written temp (possibly in flight) must be left alone");
    assert!(other.exists(), "temp files of other state files are not touched");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_old_temp_file_of_a_live_process_is_kept() {
    // the test process itself is alive: its (old-looking) temp file must not be deleted by a starting server
    let dir = tmp_dir("tmplive");
    let state = dir.join("g.json");
    Srv::start(&state, Some(&table_file(&dir))).close();
    let live = dir.join(format!(".g.json.tmp.{}", std::process::id()));
    let dead = dir.join(".g.json.tmp.999998");
    let old = std::time::SystemTime::now() - std::time::Duration::from_secs(600);
    for p in [&live, &dead] {
        std::fs::write(p, b"partial").unwrap();
        std::fs::File::options().write(true).open(p).unwrap().set_modified(old).unwrap();
    }
    Srv::start(&state, None).close();
    assert!(live.exists(), "the temp file of a LIVE process was removed");
    assert!(!dead.exists(), "the temp file of a dead pid survived");
    let _ = std::fs::remove_dir_all(&dir);
}
