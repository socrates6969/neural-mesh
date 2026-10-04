//! `neural-mesh` CLI: integration surface for ruflo (JS) and howler (Python).
//!
//! ```text
//! neural-mesh route    --state <file> --features a,b,c [--candidates x,y] [--team k] [--type t] [--seed n]
//! neural-mesh feedback --state <file> --features a,b,c --agent X [--team X,Y] --reward 0..1 [--type t]
//! neural-mesh stats    --state <file>
//! ```
//! Output is always one JSON object on stdout. Errors print
//! `{"ok":false,"error":"..."}` and exit with status 1 (usage) / 2 (state I/O).
//! The state file is created if missing and written atomically (temp file +
//! rename). Agents are named strings; unknown names are auto-registered.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use neural_mesh::hebbian::{HebbianConfig, HebbianRouter, active_features};
use neural_mesh::mesh::{MeshConfig, NeuralMesh};
use neural_mesh::router::{Rng, Task};

const STATE_VERSION: u32 = 1;
const DEFAULT_SEED: u64 = 0x6E65_7572_616C; // "neural"

const USAGE: &str = "usage:\n  neural-mesh route    --state <file> --features a,b,c [--candidates x,y] [--team k] [--type t] [--seed n]\n  neural-mesh feedback --state <file> --features a,b,c --agent X [--team X,Y] --reward 0..1 [--type t]\n  neural-mesh stats    --state <file>";

#[derive(Debug, Serialize, Deserialize)]
struct State {
    version: u32,
    /// index -> agent name
    agents: Vec<String>,
    /// Base seed of the CLI's deterministic RNG stream.
    seed: u64,
    /// Number of RNG-consuming calls so far (route i uses seed ^ mix(i)).
    calls: u64,
    /// Number of feedback calls so far (also used as task ids).
    feedbacks: u64,
    mesh: NeuralMesh<HebbianRouter>,
}

impl State {
    fn new() -> Self {
        State {
            version: STATE_VERSION,
            agents: Vec::new(),
            seed: DEFAULT_SEED,
            calls: 0,
            feedbacks: 0,
            mesh: NeuralMesh::new(0, HebbianRouter::new(0, HebbianConfig::default()), MeshConfig::default()),
        }
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.agents.iter().position(|a| a == name)
    }

    /// Index of `name`, registering it if new.
    fn register(&mut self, name: &str) -> usize {
        if let Some(i) = self.index_of(name) {
            return i;
        }
        self.agents.push(name.to_string());
        let n = self.agents.len();
        self.mesh.ensure_agents(n);
        self.mesh.leader_mut().ensure_agents(n);
        n - 1
    }
}

#[derive(Debug)]
enum CliError {
    Usage(String),
    Io(String),
}

impl CliError {
    fn code(&self) -> i32 {
        match self {
            CliError::Usage(_) => 1,
            CliError::Io(_) => 2,
        }
    }
    fn msg(&self) -> &str {
        match self {
            CliError::Usage(m) | CliError::Io(m) => m,
        }
    }
}

fn usage(m: impl Into<String>) -> CliError {
    CliError::Usage(m.into())
}

/// Parsed `--flag value` / `--flag=value` options.
struct Opts(BTreeMap<String, String>);

impl Opts {
    fn parse(args: &[String], allowed: &[&str]) -> Result<Opts, CliError> {
        let mut m = BTreeMap::new();
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            let Some(body) = a.strip_prefix("--") else {
                return Err(usage(format!("unexpected argument '{a}'")));
            };
            let (k, v) = match body.split_once('=') {
                Some((k, v)) => (k.to_string(), v.to_string()),
                None => {
                    let v = args.get(i + 1).ok_or_else(|| usage(format!("--{body} needs a value")))?;
                    i += 1;
                    (body.to_string(), v.clone())
                }
            };
            if !allowed.contains(&k.as_str()) {
                return Err(usage(format!("unknown option --{k}")));
            }
            if m.insert(k.clone(), v).is_some() {
                return Err(usage(format!("--{k} given twice")));
            }
            i += 1;
        }
        Ok(Opts(m))
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.0.get(k).map(String::as_str)
    }

    fn req(&self, k: &str) -> Result<&str, CliError> {
        self.get(k).ok_or_else(|| usage(format!("missing required --{k}")))
    }
}

fn split_list(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for p in s.split(',') {
        let p = p.trim();
        if !p.is_empty() && !out.iter().any(|q| q == p) {
            out.push(p.to_string());
        }
    }
    out
}

fn parse_names(s: &str, what: &str) -> Result<Vec<String>, CliError> {
    let v = split_list(s);
    if v.is_empty() {
        return Err(usage(format!("--{what} is empty")));
    }
    Ok(v)
}

fn make_task(o: &Opts, id: u64) -> Result<Task, CliError> {
    let features = split_list(o.req("features")?);
    let task_type = o.get("type").map(str::trim).filter(|t| !t.is_empty()).map(String::from);
    Ok(Task { id, task_type, features })
}

fn load_state(path: &Path) -> Result<State, CliError> {
    match fs::read_to_string(path) {
        Ok(s) if s.trim().is_empty() => Ok(State::new()),
        Ok(s) => {
            let st: State = serde_json::from_str(&s)
                .map_err(|e| CliError::Io(format!("cannot parse state file {}: {e}", path.display())))?;
            if st.version != STATE_VERSION {
                return Err(CliError::Io(format!("unsupported state version {}", st.version)));
            }
            if st.mesh.n_agents() < st.agents.len() {
                return Err(CliError::Io("corrupt state: agent table larger than mesh".into()));
            }
            Ok(st)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(State::new()),
        Err(e) => Err(CliError::Io(format!("cannot read state file {}: {e}", path.display()))),
    }
}

/// Atomic write: temp file in the same directory, then rename over the target.
fn save_state(path: &Path, st: &State) -> Result<(), CliError> {
    let js = serde_json::to_string_pretty(st).map_err(|e| CliError::Io(format!("serialise: {e}")))?;
    let mut tmp: PathBuf = path.to_path_buf();
    let fname = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| "state".into());
    tmp.set_file_name(format!(".{fname}.tmp.{}", std::process::id()));
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).map_err(|e| CliError::Io(format!("create dir {}: {e}", dir.display())))?;
    }
    fs::write(&tmp, js).map_err(|e| CliError::Io(format!("write {}: {e}", tmp.display())))?;
    if let Err(e) = fs::rename(&tmp, path) {
        let _ = fs::remove_file(&tmp);
        return Err(CliError::Io(format!("rename to {}: {e}", path.display())));
    }
    Ok(())
}

fn mix(seed: u64, i: u64) -> u64 {
    // one splitmix step so consecutive calls get unrelated streams
    let mut r = Rng::new(seed ^ i.wrapping_mul(0xA24B_AED4_963E_E407));
    r.next_u64()
}

fn r6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

fn cmd_route(st: &mut State, o: &Opts) -> Result<Value, CliError> {
    let cand_idx: Vec<usize> = match o.get("candidates") {
        Some(c) => parse_names(c, "candidates")?.iter().map(|n| st.register(n)).collect(),
        None => (0..st.agents.len()).collect(),
    };
    if cand_idx.is_empty() {
        return Err(usage("no agents registered yet; pass --candidates x,y"));
    }
    let k = match o.get("team") {
        Some(t) => t.trim().parse::<usize>().ok().filter(|&k| k >= 1).ok_or_else(|| usage("--team must be an integer >= 1"))?,
        None => st.mesh.cfg.team_size.max(1),
    };
    let task = make_task(o, st.feedbacks)?;
    let seed = match o.get("seed") {
        Some(s) => s.trim().parse::<u64>().map_err(|_| usage("--seed must be a non-negative integer"))?,
        None => mix(st.seed, st.calls),
    };
    st.calls += 1;
    let mut rng = Rng::new(seed);
    let team = st.mesh.route_k(&task, &cand_idx, &mut rng, k);
    let all_scores = st.mesh.leader().scores(&task);
    let scores: serde_json::Map<String, Value> =
        cand_idx.iter().map(|&i| (st.agents[i].clone(), json!(r6(all_scores.get(i).copied().unwrap_or(0.0))))).collect();
    let health: serde_json::Map<String, Value> =
        cand_idx.iter().map(|&i| (st.agents[i].clone(), json!(r6(st.mesh.health(i))))).collect();
    let names: Vec<&str> = team.iter().map(|&i| st.agents[i].as_str()).collect();
    Ok(json!({
        "ok": true,
        "agent": names.first(),
        "team": names,
        "scores": scores,
        "health": health,
        "features": active_features(&task),
    }))
}

fn cmd_feedback(st: &mut State, o: &Opts) -> Result<Value, CliError> {
    let agent = o.req("agent")?.trim().to_string();
    if agent.is_empty() || agent.contains(',') {
        return Err(usage("--agent must be a single non-empty name"));
    }
    let reward: f64 = o
        .req("reward")?
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|r| r.is_finite() && (0.0..=1.0).contains(r))
        .ok_or_else(|| usage("--reward must be a number in [0,1]"))?;
    let mut names: Vec<String> = match o.get("team") {
        Some(t) => parse_names(t, "team")?,
        None => Vec::new(),
    };
    names.retain(|n| n != &agent);
    names.insert(0, agent);
    let task = make_task(o, st.feedbacks)?;
    let team: Vec<usize> = names.iter().map(|n| st.register(n)).collect();
    st.mesh.feedback(&task, &team, reward);
    st.feedbacks += 1;
    let health: serde_json::Map<String, Value> =
        team.iter().map(|&i| (st.agents[i].clone(), json!(r6(st.mesh.health(i))))).collect();
    let mut edges = Vec::new();
    for (x, &a) in team.iter().enumerate() {
        for &b in &team[x + 1..] {
            edges.push(json!({"a": st.agents[a], "b": st.agents[b], "w": r6(st.mesh.edge(a, b))}));
        }
    }
    Ok(json!({
        "ok": true,
        "agent": names[0],
        "team": names,
        "reward": reward,
        "health": health,
        "edges": edges,
    }))
}

fn cmd_stats(st: &State) -> Value {
    let n = st.agents.len();
    let agents: Vec<Value> = (0..n)
        .map(|i| json!({"name": st.agents[i], "health": r6(st.mesh.health(i)), "benched": st.mesh.is_benched(i)}))
        .collect();
    let mut edges = Vec::new();
    for a in 0..n {
        for b in a + 1..n {
            let w = st.mesh.edge(a, b);
            if w != 0.0 {
                edges.push(json!({"a": st.agents[a], "b": st.agents[b], "w": r6(w)}));
            }
        }
    }
    let router = st.mesh.leader();
    let weights: serde_json::Map<String, Value> = router
        .rows()
        .iter()
        .map(|(f, row)| {
            let m: serde_json::Map<String, Value> =
                (0..n).map(|i| (st.agents[i].clone(), json!(r6(row.get(i).copied().unwrap_or(0.0))))).collect();
            (f.clone(), json!({"baseline": r6(router.baseline(f)), "w": m}))
        })
        .collect();
    json!({
        "ok": true,
        "version": st.version,
        "agents": agents,
        "routes": st.mesh.routes(),
        "feedbacks": st.feedbacks,
        "updates": router.updates(),
        "edges": edges,
        "weights": weights,
        "hebbian": router.cfg,
        "mesh": st.mesh.cfg,
    })
}

/// Library-level entry point (used by tests): runs one command and persists
/// state when the command succeeds.
fn run(args: &[String]) -> Result<Value, CliError> {
    let Some(cmd) = args.first() else { return Err(usage("missing command")) };
    let rest = &args[1..];
    let allowed: &[&str] = match cmd.as_str() {
        "route" => &["state", "features", "candidates", "team", "type", "seed"],
        "feedback" => &["state", "features", "agent", "team", "reward", "type"],
        "stats" => &["state"],
        other => return Err(usage(format!("unknown command '{other}'"))),
    };
    let o = Opts::parse(rest, allowed)?;
    let path = PathBuf::from(o.req("state")?);
    let existed = path.exists();
    let mut st = load_state(&path)?;
    let out = match cmd.as_str() {
        "route" => cmd_route(&mut st, &o)?,
        "feedback" => cmd_feedback(&mut st, &o)?,
        _ => {
            let v = cmd_stats(&st);
            if !existed {
                save_state(&path, &st)?;
            }
            return Ok(v);
        }
    };
    save_state(&path, &st)?;
    Ok(out)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| matches!(a.as_str(), "-h" | "--help" | "help")) {
        println!("{}", json!({"ok": true, "usage": USAGE}));
        return;
    }
    std::panic::set_hook(Box::new(|_| {}));
    let res = std::panic::catch_unwind(|| run(&args));
    let code = match res {
        Ok(Ok(v)) => {
            println!("{v}");
            0
        }
        Ok(Err(e)) => {
            println!("{}", json!({"ok": false, "error": e.msg(), "usage": USAGE}));
            e.code()
        }
        Err(_) => {
            println!("{}", json!({"ok": false, "error": "internal error"}));
            3
        }
    };
    std::process::exit(code);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("neural-mesh-test-{}-{name}.json", std::process::id()));
        let _ = fs::remove_file(&p);
        p
    }

    fn call(args: &[&str]) -> Result<Value, CliError> {
        run(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn route_feedback_stats_roundtrip() {
        let p = tmp("rt");
        let ps = p.to_str().unwrap();
        let v = call(&["route", "--state", ps, "--features", "sql,db", "--candidates", "alice,bob,carol"]).unwrap();
        assert_eq!(v["ok"], true);
        let leader = v["agent"].as_str().unwrap().to_string();
        assert!(["alice", "bob", "carol"].contains(&leader.as_str()));
        assert_eq!(v["scores"].as_object().unwrap().len(), 3);
        assert!(p.exists(), "state file created");

        // Teach: bob is great at sql, others fail.
        for _ in 0..40 {
            call(&["feedback", "--state", ps, "--features", "sql,db", "--agent", "bob", "--reward", "1"]).unwrap();
            call(&["feedback", "--state", ps, "--features", "sql,db", "--agent", "alice", "--reward", "0"]).unwrap();
        }
        let v = call(&["feedback", "--state", ps, "--features", "sql", "--agent", "bob", "--team", "dave,bob", "--reward=0.8"]).unwrap();
        assert_eq!(v["team"], json!(["bob", "dave"]), "agent first, new name auto-registered");

        let mut bob = 0;
        for _ in 0..50 {
            let v = call(&["route", "--state", ps, "--features", "sql,db"]).unwrap();
            if v["agent"] == "bob" {
                bob += 1;
            }
        }
        assert!(bob >= 40, "bob routed {bob}/50");

        let v = call(&["route", "--state", ps, "--features", "sql", "--team", "2"]).unwrap();
        assert_eq!(v["team"].as_array().unwrap().len(), 2);

        let s = call(&["stats", "--state", ps]).unwrap();
        assert_eq!(s["ok"], true);
        let names: Vec<&str> = s["agents"].as_array().unwrap().iter().map(|a| a["name"].as_str().unwrap()).collect();
        assert_eq!(names, vec!["alice", "bob", "carol", "dave"]);
        assert!(s["weights"]["sql"]["w"]["bob"].as_f64().unwrap() > s["weights"]["sql"]["w"]["alice"].as_f64().unwrap());
        assert!(s["edges"].as_array().unwrap().iter().any(|e| e["a"] == "bob" && e["b"] == "dave"));
        // state is valid JSON on disk and reloads
        let st = load_state(&p).unwrap();
        assert_eq!(st.agents.len(), 4);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn bad_input_is_an_error_not_a_panic() {
        let p = tmp("bad");
        let ps = p.to_str().unwrap();
        for args in [
            vec![],
            vec!["nope"],
            vec!["route"],
            vec!["route", "--state", ps],
            vec!["route", "--state", ps, "--features", "a"], // no agents yet
            vec!["route", "--state", ps, "--features", "a", "--candidates", ",,"],
            vec!["route", "--state", ps, "--features", "a", "--candidates", "x", "--team", "0"],
            vec!["route", "--state", ps, "--features", "a", "--candidates", "x", "--bogus", "1"],
            vec!["route", "--state", ps, "--features"],
            vec!["feedback", "--state", ps, "--features", "a", "--agent", "x", "--reward", "1.5"],
            vec!["feedback", "--state", ps, "--features", "a", "--agent", "x", "--reward", "NaN"],
            vec!["feedback", "--state", ps, "--features", "a", "--agent", "", "--reward", "1"],
            vec!["feedback", "--state", ps, "--features", "a", "--reward", "1"],
            vec!["stats", "--state", ps, "stray"],
        ] {
            let r = call(&args);
            assert!(r.is_err(), "expected error for {args:?}");
        }
        fs::write(&p, "{not json").unwrap();
        let r = call(&["stats", "--state", ps]);
        assert!(matches!(r, Err(CliError::Io(_))));
        // corrupt file is left untouched
        assert_eq!(fs::read_to_string(&p).unwrap(), "{not json");
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn stats_creates_missing_file() {
        let p = tmp("stats").with_extension("d").join("nested.json");
        let ps = p.to_str().unwrap();
        let v = call(&["stats", "--state", ps]).unwrap();
        assert_eq!(v["agents"], json!([]));
        assert!(p.exists());
        let _ = fs::remove_dir_all(p.parent().unwrap());
    }
}
