//! `nm_guarded`: opt-in JSON-lines stdio server over [`CompactRouterF64`] - the `f64` compact brain that is decision- and
//! rng-identical to `defaults::recommended_router` (Guarded(hebbian-robust), `src/compact.rs` tests) - with a keyword map taken
//! from a Hebbwise keyword table. New binary only (Hebbwise R3 shadow arm); no existing file, default or CLI behaviour changes.
//!
//! ```text
//! nm_guarded --state <file> [--table <keyword-table.json>] --stdio
//! ```
//! The table is read only when the state file is created; afterwards the state's own copy is used, and a `--table` that differs
//! from it is refused (exit 2) so a changed table can never silently mix with a brain trained under another one.
//! Table format: `{"version":1,"rules":[{"agent":"fr-tester","keywords":["test","tests"]}, ...]}` (keywords are lower-cased; the
//! FIRST rule naming a keyword wins).
//!
//! Protocol (one JSON object per line in and out; an optional `"id"` is echoed; blank lines are ignored; a bad line gets
//! `{"ok":false,"error":"..."}` and the server keeps running):
//! * `{"cmd":"route","features":[..]|"a,b","candidates":[names],"seed":N}` -> `{"ok":true,"agent":X,"decision":"keyword"|"challenger"|"inner"}`.
//!   PURE: the decision counter is restored after the pick, so the same request on the same state always gives the same answer and
//!   leaves the learned state unchanged. A candidate name never seen before is registered (append-only, max 128 agents) and the
//!   state is persisted BEFORE the reply when that happened, so a crash can never lose an index assignment.
//! * `{"cmd":"feedback","fid":"<dispatch id>","features":[..], "label":X}` -> supervised update `update(X, 1.0)`;
//!   `{"cmd":"feedback","fid":..,"features":[..],"agent":X,"reward":r,"failure_class":"skill"|"spec"|"env"}` -> `skill` (default)
//!   updates with reward r; `spec` and `env` record the id but leave the skill weights untouched (R117 semantics: the guarded brain
//!   has no health/risk state). A `fid` that was already applied is a no-op (`"duplicate":true`): exactly-once across restarts.
//!   Every applied feedback is persisted (temp file + rename) BEFORE the reply.
//! * `{"cmd":"stats"}`, `{"cmd":"flush"}`, `{"cmd":"shutdown"}` (persist, reply, exit). EOF persists and exits.
//!
//! The brain has a fixed capacity of [`MAX_AGENTS`] = 128 agent slots (the compact limit); agents are assigned slots in order of
//! first appearance (table agents first). Only candidates passed to `route` can be picked.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};

use neural_mesh::compact::{CompactConfig, CompactRouterF64, MAX_AGENTS};
use neural_mesh::router::{Rng, Router, Task};
use serde_json::{Value, json};

const VERSION: u64 = 1;
const MAX_ID: usize = 256;
const MAX_LINE: usize = 1 << 20;
const USAGE: &str = "usage: nm_guarded --state <file> [--table <keyword-table.json>] --stdio";

struct Guard {
    path: PathBuf,
    agents: Vec<String>,
    index: HashMap<String, usize>,
    /// keyword -> agent name (the table the brain was created with)
    table: BTreeMap<String, String>,
    applied: BTreeSet<String>,
    feedbacks: u64,
    router: CompactRouterF64,
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("bad hex".into());
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| "bad hex".to_string())).collect()
}

/// Parse a Hebbwise keyword table into keyword -> agent (first rule naming a keyword wins).
fn parse_table(text: &str) -> Result<BTreeMap<String, String>, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("table: {e}"))?;
    let rules = v.get("rules").and_then(Value::as_array).ok_or("table: missing rules array")?;
    let mut out = BTreeMap::new();
    for r in rules {
        let agent = r.get("agent").and_then(Value::as_str).map(str::trim).filter(|a| !a.is_empty() && !a.contains(',')).ok_or("table: rule without agent")?;
        for k in r.get("keywords").and_then(Value::as_array).ok_or("table: rule without keywords")? {
            let k = k.as_str().ok_or("table: keyword must be a string")?.trim().to_lowercase();
            if !k.is_empty() {
                out.entry(k).or_insert_with(|| agent.to_string());
            }
        }
    }
    Ok(out)
}

fn table_agents(rules_text: Option<&str>, table: &BTreeMap<String, String>) -> Vec<String> {
    // slot order: agents in rule order of the table file (stable), else sorted
    let mut out: Vec<String> = Vec::new();
    if let Some(t) = rules_text
        && let Ok(v) = serde_json::from_str::<Value>(t)
        && let Some(rules) = v.get("rules").and_then(Value::as_array)
    {
        for r in rules {
            if let Some(a) = r.get("agent").and_then(Value::as_str).map(str::trim)
                && table.values().any(|x| x == a)
                && !out.iter().any(|x| x == a)
            {
                out.push(a.to_string());
            }
        }
    }
    for a in table.values() {
        if !out.iter().any(|x| x == a) {
            out.push(a.clone());
        }
    }
    out
}

impl Guard {
    fn create(path: PathBuf, table_text: Option<&str>) -> Result<Guard, String> {
        let table = match table_text {
            Some(t) => parse_table(t)?,
            None => BTreeMap::new(),
        };
        let agents = table_agents(table_text, &table);
        if agents.len() > MAX_AGENTS {
            return Err("table names more than 128 agents".into());
        }
        let index: HashMap<String, usize> = agents.iter().enumerate().map(|(i, a)| (a.clone(), i)).collect();
        let kw: HashMap<String, usize> = table.iter().map(|(k, a)| (k.clone(), index[a])).collect();
        let router = CompactRouterF64::new(MAX_AGENTS, &kw, CompactConfig::default()).map_err(|e| e.to_string())?;
        // canonical form: the compact constructor interns keywords in HashMap order (random per process); a fresh brain has no
        // rows, so sorting its snapshot's keyword/name tables by string and restoring makes fresh states byte-reproducible
        let mut snap = router.snapshot();
        snap.keywords.sort();
        snap.names.sort();
        let router = CompactRouterF64::from_snapshot(snap).map_err(|e| e.to_string())?;
        Ok(Guard { path, agents, index, table, applied: BTreeSet::new(), feedbacks: 0, router })
    }

    fn load(path: PathBuf) -> Result<Guard, String> {
        let text = std::fs::read_to_string(&path).map_err(|e| format!("read state: {e}"))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("parse state: {e}"))?;
        if v.get("version").and_then(Value::as_u64) != Some(VERSION) {
            return Err("unsupported state version".into());
        }
        let agents: Vec<String> = serde_json::from_value(v.get("agents").cloned().unwrap_or(Value::Null)).map_err(|e| format!("state agents: {e}"))?;
        let table: BTreeMap<String, String> = serde_json::from_value(v.get("table").cloned().unwrap_or(Value::Null)).map_err(|e| format!("state table: {e}"))?;
        let applied: BTreeSet<String> = serde_json::from_value(v.get("applied").cloned().unwrap_or(Value::Null)).map_err(|e| format!("state applied: {e}"))?;
        let feedbacks = v.get("feedbacks").and_then(Value::as_u64).ok_or("state feedbacks")?;
        let bytes = unhex(v.get("router").and_then(Value::as_str).ok_or("state router")?)?;
        let router = CompactRouterF64::from_bytes(&bytes).map_err(|e| e.to_string())?;
        if router.n_agents() != MAX_AGENTS || agents.len() > MAX_AGENTS {
            return Err("corrupt state: agent capacity".into());
        }
        let index = agents.iter().enumerate().map(|(i, a)| (a.clone(), i)).collect();
        Ok(Guard { path, agents, index, table, applied, feedbacks, router })
    }

    fn save(&self) -> Result<(), String> {
        let v = json!({
            "version": VERSION, "kind": "nm_guarded", "n_agents": MAX_AGENTS, "agents": self.agents, "table": self.table,
            "applied": self.applied, "feedbacks": self.feedbacks, "router": hex(&self.router.to_bytes()),
        });
        let body = serde_json::to_string(&v).map_err(|e| e.to_string())?;
        let fname = self.path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| "state".into());
        let tmp = self.path.with_file_name(format!(".{fname}.tmp.{}", std::process::id()));
        std::fs::write(&tmp, body.as_bytes()).map_err(|e| format!("write state: {e}"))?;
        std::fs::rename(&tmp, &self.path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!("rename state: {e}")
        })
    }

    /// Index of `name`, registering it (append-only). Returns (index, newly registered).
    fn register(&mut self, name: &str) -> Result<(usize, bool), String> {
        if let Some(&i) = self.index.get(name) {
            return Ok((i, false));
        }
        if self.agents.len() >= MAX_AGENTS {
            return Err("too many agents (max 128)".into());
        }
        let i = self.agents.len();
        self.agents.push(name.to_string());
        self.index.insert(name.to_string(), i);
        Ok((i, true))
    }

    fn route(&mut self, req: &Value) -> Result<Value, String> {
        let features = features_of(req)?;
        let names = names_of(req.get("candidates"), "candidates")?;
        let seed = req.get("seed").and_then(Value::as_u64).ok_or("seed must be a non-negative integer")?;
        // validate first, then register (a failed call registers nothing)
        let new_names = names.iter().filter(|n| !self.index.contains_key(n.as_str())).count();
        if self.agents.len() + new_names > MAX_AGENTS {
            return Err("too many agents (max 128)".into());
        }
        let mut cand = Vec::with_capacity(names.len());
        let mut grew = false;
        for n in &names {
            let (i, new) = self.register(n)?;
            grew |= new;
            if !cand.contains(&i) {
                cand.push(i);
            }
        }
        if grew {
            self.save()?;
        }
        let task = Task { id: self.feedbacks, task_type: None, features };
        let before = self.router.decisions;
        let mut rng = Rng::new(seed);
        let a = self.router.select(&task, &cand, &mut rng);
        let kind = (0..3).find(|&k| self.router.decisions[k] != before[k]).unwrap_or(2);
        self.router.decisions = before; // route is pure: the learned state is unchanged
        let decision = ["keyword", "challenger", "inner"][kind];
        Ok(json!({"ok": true, "agent": self.agents[a], "decision": decision}))
    }

    fn feedback(&mut self, req: &Value) -> Result<Value, String> {
        let fid = req.get("fid").and_then(Value::as_str).filter(|s| !s.is_empty() && s.len() <= MAX_ID).ok_or("fid must be a non-empty string")?.to_string();
        let features = features_of(req)?;
        let label = req.get("label").map(|v| v.as_str().map(str::trim).filter(|s| !s.is_empty() && !s.contains(',')).ok_or("label must be a single agent name")).transpose()?;
        let class = match req.get("failure_class").map(|v| v.as_str().unwrap_or("?")) {
            None | Some("skill") => "skill",
            Some("spec") => "spec",
            Some("env") => "env",
            Some(_) => return Err("failure_class must be skill, spec or env".into()),
        };
        let (agent, reward) = match label {
            Some(l) => {
                if req.get("reward").is_some() || class != "skill" {
                    return Err("label cannot be combined with reward or failure_class spec|env".into());
                }
                (l.to_string(), 1.0)
            }
            None => {
                let a = req.get("agent").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty() && !s.contains(',')).ok_or("agent must be a single agent name")?;
                let r = req.get("reward").and_then(Value::as_f64).filter(|r| r.is_finite() && (0.0..=1.0).contains(r)).ok_or("reward must be a number in [0,1]")?;
                (a.to_string(), r)
            }
        };
        if self.applied.contains(&fid) {
            return Ok(json!({"ok": true, "applied": false, "duplicate": true, "agent": agent}));
        }
        let (i, _) = self.register(&agent)?;
        if class == "skill" {
            let task = Task { id: self.feedbacks, task_type: None, features };
            self.router.update(&task, i, reward);
            self.feedbacks += 1;
        }
        self.applied.insert(fid);
        self.save()?;
        Ok(json!({"ok": true, "applied": true, "duplicate": false, "agent": agent, "class": class}))
    }

    fn stats(&self) -> Value {
        json!({"ok": true, "kind": "nm_guarded", "agents": self.agents, "n_agents": MAX_AGENTS, "feedbacks": self.feedbacks,
               "applied": self.applied.len(), "updates": self.router.updates(), "rows": self.router.n_rows(),
               "decisions": self.router.decisions, "table_keywords": self.table.len()})
    }
}

fn features_of(req: &Value) -> Result<Vec<String>, String> {
    let mut f = match req.get("features") {
        Some(Value::String(s)) => s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect::<Vec<_>>(),
        Some(Value::Array(a)) => {
            let mut v = Vec::new();
            for x in a {
                let s = x.as_str().ok_or("features must be strings")?.trim();
                if !s.is_empty() {
                    v.push(s.to_string());
                }
            }
            v
        }
        _ => return Err("features required".into()),
    };
    if f.is_empty() {
        f.push("no-keyword".into()); // the CLI's NO_FEATURE (src/main.rs)
    }
    Ok(f)
}

fn names_of(v: Option<&Value>, what: &str) -> Result<Vec<String>, String> {
    let raw: Vec<String> = match v {
        Some(Value::String(s)) => s.split(',').map(str::to_string).collect(),
        Some(Value::Array(a)) => a.iter().map(|x| x.as_str().map(str::to_string).ok_or(format!("{what} must be strings"))).collect::<Result<_, _>>()?,
        _ => return Err(format!("{what} required")),
    };
    let out: Vec<String> = raw.into_iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if out.is_empty() || out.iter().any(|s| s.contains(',')) {
        return Err(format!("{what} must be non-empty agent names"));
    }
    Ok(out)
}

fn handle(g: &mut Guard, line: &str) -> (Value, bool) {
    let req: Value = match serde_json::from_str(line) {
        Ok(v @ Value::Object(_)) => v,
        Ok(_) => return (json!({"ok": false, "error": "request must be a JSON object"}), false),
        Err(e) => return (json!({"ok": false, "error": format!("bad json: {e}")}), false),
    };
    let id = req.get("id").cloned();
    let (mut out, quit) = match req.get("cmd").and_then(Value::as_str) {
        Some("route") => (g.route(&req).unwrap_or_else(|e| json!({"ok": false, "error": e})), false),
        Some("feedback") => (g.feedback(&req).unwrap_or_else(|e| json!({"ok": false, "error": e})), false),
        Some("stats") => (g.stats(), false),
        Some("flush") => (g.save().map(|_| json!({"ok": true, "flush": true})).unwrap_or_else(|e| json!({"ok": false, "error": e})), false),
        Some("shutdown") => (g.save().map(|_| json!({"ok": true, "shutdown": true})).unwrap_or_else(|e| json!({"ok": false, "error": e})), true),
        Some(c) => (json!({"ok": false, "error": format!("unknown cmd '{c}'")}), false),
        None => (json!({"ok": false, "error": "missing cmd"}), false),
    };
    if let (Some(id), Some(o)) = (id, out.as_object_mut()) {
        o.insert("id".into(), id);
    }
    (out, quit)
}

/// Remove `.<state>.tmp.<pid>` files left by a hard kill: only files of ANOTHER pid that is NOT ALIVE, and that are older than
/// 2 s (a temp file exists only between write and rename). If liveness cannot be determined the file is kept. Returns how many
/// were removed.
fn clean_stale_temps(path: &Path) -> usize {
    let (Some(dir), Some(fname)) = (path.parent(), path.file_name().map(|f| f.to_string_lossy().into_owned())) else { return 0 };
    let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
    let prefix = format!(".{fname}.tmp.");
    let own = std::process::id();
    let mut n = 0;
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(pid) = name.strip_prefix(&prefix).and_then(|p| p.parse::<u32>().ok()) else { continue };
        if pid == own {
            continue;
        }
        let old = e.metadata().ok().and_then(|m| m.modified().ok()).and_then(|t| t.elapsed().ok()).is_some_and(|a| a.as_secs_f64() > 2.0);
        if old && pid_alive(pid) == Some(false) && std::fs::remove_file(e.path()).is_ok() {
            n += 1;
        }
    }
    n
}

/// Whether a process with this pid exists: `Some(alive)`, or `None` when it cannot be determined.
#[cfg(windows)]
fn pid_alive(pid: u32) -> Option<bool> {
    let out = std::process::Command::new("tasklist").args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    Some(text.lines().any(|l| l.split(',').nth(1).is_some_and(|f| f.trim_matches('"') == pid.to_string())))
}
#[cfg(unix)]
fn pid_alive(pid: u32) -> Option<bool> {
    if Path::new("/proc/self").exists() {
        return Some(Path::new(&format!("/proc/{pid}")).exists());
    }
    std::process::Command::new("kill").args(["-0", &pid.to_string()]).stderr(std::process::Stdio::null()).status().ok().map(|s| s.success())
}
#[cfg(not(any(windows, unix)))]
fn pid_alive(_pid: u32) -> Option<bool> {
    None
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn open(args: &[String]) -> Result<Guard, (i32, String)> {
    let state = arg(args, "--state").ok_or((1, USAGE.to_string()))?;
    if !args.iter().any(|a| a == "--stdio") {
        return Err((1, USAGE.to_string()));
    }
    let path = PathBuf::from(state);
    clean_stale_temps(&path);
    let table_text = match arg(args, "--table") {
        Some(t) => Some(std::fs::read_to_string(&t).map_err(|e| (2, format!("read table: {e}")))?),
        None => None,
    };
    if Path::new(&path).exists() {
        let g = Guard::load(path).map_err(|e| (2, e))?;
        if let Some(t) = &table_text {
            let want = parse_table(t).map_err(|e| (2, e))?;
            if want != g.table {
                return Err((2, "table_mismatch: the state was created with a different keyword table (rebuild the arm)".into()));
            }
        }
        Ok(g)
    } else {
        let g = Guard::create(path, table_text.as_deref()).map_err(|e| (2, e))?;
        g.save().map_err(|e| (2, e))?;
        Ok(g)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut g = match open(&args) {
        Ok(g) => g,
        Err((code, msg)) => {
            println!("{}", json!({"ok": false, "error": msg}));
            std::process::exit(code);
        }
    };
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    let stdout = std::io::stdout();
    let mut buf: Vec<u8> = Vec::new();
    loop {
        buf.clear();
        match input.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let (out, quit) = if buf.len() > MAX_LINE {
            (json!({"ok": false, "error": "line too long"}), false)
        } else {
            let line = String::from_utf8_lossy(&buf);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            handle(&mut g, line)
        };
        let mut o = stdout.lock();
        let _ = writeln!(o, "{out}");
        let _ = o.flush();
        if quit {
            return;
        }
    }
    let _ = g.save();
}
