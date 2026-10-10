//! `neural-mesh` CLI: integration surface for ruflo (JS) and howler (Python).
//!
//! ```text
//! neural-mesh route    --state <file> --features a,b,c [--candidates x,y] [--team k] [--type t] [--seed n]
//! neural-mesh feedback --state <file> --features a,b,c --agent X [--team X,Y] --reward 0..1 [--type t] [--failure-class skill|spec|env]
//! neural-mesh feedback --state <file> --features a,b,c --label X [--agent routed]          (supervised: the correct agent is known)
//! neural-mesh stats    --state <file>
//! neural-mesh config   --state <file> [--eta x --temperature x --decay x --epsilon x --team k --prior-k x]
//! neural-mesh merge-state --into <A> --from <B> [--out <file>]   (fold state B into A; see `merge_states`)
//! ```
//! `route` also takes `--prior <agent>` (the caller's own keyword pick; see `cmd_route`) and
//! route/feedback/stats take `--eta --temperature --decay --epsilon --prior-k`, which set and
//! persist the config before the command runs. (`--team` on route stays per-call; persist the
//! default team size with `config --team`.)
//! Output is always one JSON object on stdout. Errors print
//! `{"ok":false,"error":"..."}` and exit with status 1 (usage) / 2 (state I/O).
//! The state file is created if missing and written atomically (temp file +
//! rename). Agents are named strings; unknown names are auto-registered.
//!
//! Long-lived mode (see `serve.rs`): `neural-mesh serve --state <file> (--stdio | --port N)
//! [--flush-every N] [--flush-ms T]` answers JSON-lines requests with the same responses.

mod serve;

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

/// Feature used when a task has no keywords.
const NO_FEATURE: &str = "no-keyword";
/// How long to wait for the state lock before giving up.
const LOCK_TIMEOUT_MS: u64 = 2000;
/// A lock file older than this is assumed to belong to a crashed process.
const LOCK_STALE_MS: u64 = 10_000;

/// Robust defaults for NEW state files (held-out evidence: the legacy 0.5/0.05/0.01/0.02 ranks 21/23).
const DEFAULT_ETA: f64 = 0.01;
const DEFAULT_TEMPERATURE: f64 = 0.002;
const DEFAULT_DECAY: f64 = 0.01;
const DEFAULT_EPSILON: f64 = 0.0;
/// Pseudo-count of the warm prior: follow it with probability k/(k+n).
const DEFAULT_PRIOR_K: f64 = 1.0;
/// Salt so the prior's coin never shares a stream with the mesh's draws.
const PRIOR_SALT: u64 = 0x5052_494F_525F_4B57;
/// Option names shared by route/feedback/stats/config (team is config-only).
const CONFIG_KEYS: [&str; 5] = ["eta", "temperature", "decay", "epsilon", "prior-k"];

const USAGE: &str = "usage:\n  neural-mesh route    --state <file> --features a,b,c [--candidates x,y] [--team k] [--type t] [--seed n] [--prior agent]\n  neural-mesh feedback --state <file> --features a,b,c --agent X [--team X,Y] --reward 0..1 [--type t]\n  neural-mesh stats    --state <file>\n  neural-mesh config   --state <file> [--eta x --temperature x --decay x --epsilon x --team k --prior-k x]\n  neural-mesh merge-state --into A --from B [--out file]\n  neural-mesh frontend [--enabled 1] [--local-conf x --local-tokens n --max-sim x --cache-hit 1 --tool t --chunks JSON --min-conf x --min-sim x --ctx-budget n]  (opt-in, default off)\n  (route/feedback/stats also accept --eta --temperature --decay --epsilon --prior-k)";

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
    /// Pseudo-count k of the warm prior (`route --prior`).
    #[serde(default = "default_prior_k")]
    prior_k: f64,
    /// Feedbacks recorded per feature (evidence n for the warm prior).
    #[serde(default)]
    feature_counts: BTreeMap<String, u64>,
    /// Ids of the shared-store outcomes this state has already learned from (replayed at creation, appended by `feedback --store`, or refreshed with
    /// `--store-sync 1`). Only filled when `--store` is used. Lets a long-lived state pick up OTHER machines' outcomes without double counting its own.
    #[serde(default)]
    store_seen: std::collections::BTreeSet<String>,
}

fn default_prior_k() -> f64 {
    DEFAULT_PRIOR_K
}

/// Hebbian config of a NEW state file.
fn robust_hebbian() -> HebbianConfig {
    HebbianConfig {
        eta: DEFAULT_ETA,
        temperature: DEFAULT_TEMPERATURE,
        decay: DEFAULT_DECAY,
        epsilon: DEFAULT_EPSILON,
        ..HebbianConfig::default()
    }
}

impl State {
    fn new() -> Self {
        State {
            version: STATE_VERSION,
            agents: Vec::new(),
            seed: DEFAULT_SEED,
            calls: 0,
            feedbacks: 0,
            mesh: NeuralMesh::new(0, HebbianRouter::new(0, robust_hebbian()), MeshConfig::default()),
            prior_k: DEFAULT_PRIOR_K,
            feature_counts: BTreeMap::new(),
            store_seen: std::collections::BTreeSet::new(),
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
    let mut features = split_list(o.req("features")?);
    if features.is_empty() {
        // Uncategorised work still gets its own learnable row instead of an all-zero score.
        features.push(NO_FEATURE.to_string());
    }
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


/// Exclusive lock on `<state>.lock` held for one load-modify-save cycle, so
/// concurrent route/feedback calls (parallel hooks, subagents) never lose updates.
struct StateLock(PathBuf);
impl StateLock {
    fn acquire(state: &Path) -> Result<Self, CliError> {
        let mut lp = state.to_path_buf();
        let fname = state.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| "state".into());
        lp.set_file_name(format!("{fname}.lock"));
        if let Some(dir) = lp.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::create_dir_all(dir).map_err(|e| CliError::Io(format!("create dir {}: {e}", dir.display())))?;
        }
        let start = std::time::Instant::now();
        let mut backoff = 1u64;
        loop {
            match fs::OpenOptions::new().write(true).create_new(true).open(&lp) {
                Ok(_) => return Ok(StateLock(lp)),
                // On Windows a lock file that another process just deleted stays
                // "delete pending" briefly and create_new fails with PermissionDenied;
                // treat that exactly like contention and retry.
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists
                    || e.kind() == std::io::ErrorKind::PermissionDenied => {
                    let stale = fs::metadata(&lp)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .is_some_and(|age| age.as_millis() as u64 > LOCK_STALE_MS);
                    if stale {
                        let _ = fs::remove_file(&lp);
                        continue;
                    }
                    if start.elapsed().as_millis() as u64 > LOCK_TIMEOUT_MS {
                        return Err(CliError::Io(format!("state is locked: {}", lp.display())));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(backoff));
                    backoff = (backoff * 2).min(25);
                }
                Err(e) => return Err(CliError::Io(format!("lock {}: {e}", lp.display()))),
            }
        }
    }
}
impl Drop for StateLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Atomic write: temp file in the same directory, then rename over the target.
fn save_state(path: &Path, st: &State) -> Result<(), CliError> {
    let js = serde_json::to_string_pretty(st).map_err(|e| CliError::Io(format!("serialise: {e}")))?;
    let mut tmp: PathBuf = path.to_path_buf();
    let fname = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_else(|| "state".into());
    static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    tmp.set_file_name(format!(".{fname}.tmp.{}.{seq}", std::process::id()));
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).map_err(|e| CliError::Io(format!("create dir {}: {e}", dir.display())))?;
    }
    fs::write(&tmp, js).map_err(|e| CliError::Io(format!("write {}: {e}", tmp.display())))?;
    // Windows can transiently refuse the rename (virus scanners, indexers, a
    // reader that just closed the file); retry briefly before giving up.
    let mut attempt = 0;
    loop {
        match fs::rename(&tmp, path) {
            Ok(()) => break,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied && attempt < 20 => {
                attempt += 1;
                std::thread::sleep(std::time::Duration::from_millis(2 * attempt));
            }
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                return Err(CliError::Io(format!("rename to {}: {e}", path.display())));
            }
        }
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

/// Validated, not-yet-applied config changes.
#[derive(Default)]
struct ConfigPatch {
    eta: Option<f64>,
    temperature: Option<f64>,
    decay: Option<f64>,
    epsilon: Option<f64>,
    team: Option<usize>,
    prior_k: Option<f64>,
}

fn parse_num(o: &Opts, k: &str, lo: f64, hi: f64) -> Result<Option<f64>, CliError> {
    let Some(s) = o.get(k) else { return Ok(None) };
    s.trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && (lo..=hi).contains(v))
        .map(Some)
        .ok_or_else(|| usage(format!("--{k} must be a number in [{lo}, {hi}]")))
}

impl ConfigPatch {
    /// `with_team`: only the `config` command may set the persisted team size.
    fn parse(o: &Opts, with_team: bool) -> Result<ConfigPatch, CliError> {
        let team = match o.get("team").filter(|_| with_team) {
            Some(t) => Some(t.trim().parse::<usize>().ok().filter(|&k| k >= 1).ok_or_else(|| usage("--team must be an integer >= 1"))?),
            None => None,
        };
        Ok(ConfigPatch {
            eta: parse_num(o, "eta", 0.0, 1.0)?,
            temperature: parse_num(o, "temperature", 0.0, f64::MAX)?,
            decay: parse_num(o, "decay", 0.0, 1.0)?,
            epsilon: parse_num(o, "epsilon", 0.0, 1.0)?,
            team,
            prior_k: parse_num(o, "prior-k", 0.0, f64::MAX)?,
        })
    }

    fn apply(&self, st: &mut State) {
        let h = &mut st.mesh.leader_mut().cfg;
        h.eta = self.eta.unwrap_or(h.eta);
        h.temperature = self.temperature.unwrap_or(h.temperature);
        h.decay = self.decay.unwrap_or(h.decay);
        h.epsilon = self.epsilon.unwrap_or(h.epsilon);
        st.mesh.cfg.team_size = self.team.unwrap_or(st.mesh.cfg.team_size);
        st.prior_k = self.prior_k.unwrap_or(st.prior_k);
    }
}

fn effective_config(st: &State) -> Value {
    let h = &st.mesh.leader().cfg;
    json!({
        "eta": h.eta,
        "temperature": h.temperature,
        "decay": h.decay,
        "epsilon": h.epsilon,
        "team": st.mesh.cfg.team_size,
        "prior_k": st.prior_k,
    })
}

/// `config`: set (any given flags) and report the effective config.
fn cmd_config(st: &mut State, o: &Opts) -> Result<Value, CliError> {
    ConfigPatch::parse(o, true)?.apply(st);
    Ok(json!({"ok": true, "config": effective_config(st)}))
}

/// True when `o` carries any config-changing flag.
fn has_config_opts(o: &Opts, with_team: bool) -> bool {
    CONFIG_KEYS.iter().any(|k| o.get(k).is_some()) || (with_team && o.get("team").is_some())
}

/// Warm-prior weight k/(k+n); n = mean per-feature feedback count of the task.
fn prior_weight(st: &State, task: &Task) -> f64 {
    let n = task.features.iter().map(|f| st.feature_counts.get(f).copied().unwrap_or(0) as f64).sum::<f64>()
        / task.features.len().max(1) as f64;
    let denom = st.prior_k + n;
    if denom <= 0.0 { 0.0 } else { st.prior_k / denom }
}

/// Decide whether to follow the caller's prior after the mesh has routed.
/// Returns (weight, followed index). Unknown / non-candidate / benched priors are ignored.
fn follow_prior(st: &State, name: &str, task: &Task, cand_idx: &[usize], seed: u64) -> (f64, Option<usize>) {
    let w = prior_weight(st, task);
    let idx = st.index_of(name).filter(|i| cand_idx.contains(i) && !st.mesh.is_benched(*i));
    let coin = Rng::new(seed ^ PRIOR_SALT).f64();
    (w, idx.filter(|_| coin < w))
}

fn cmd_route(st: &mut State, o: &Opts) -> Result<Value, CliError> {
    // Validate everything before touching state (a failed call must not
    // register agents: the one-shot CLI never saves on error, and the
    // long-lived server keeps its state in memory).
    let patch = ConfigPatch::parse(o, false)?;
    let prior: Option<String> = match o.get("prior") {
        Some(p) => Some(p.trim().to_string()).filter(|p| !p.is_empty() && !p.contains(',')),
        None => None,
    };
    if o.get("prior").is_some() && prior.is_none() {
        return Err(usage("--prior must be a single non-empty agent name"));
    }
    let cand_names = match o.get("candidates") {
        Some(c) => Some(parse_names(c, "candidates")?),
        None => None,
    };
    if cand_names.is_none() && st.agents.is_empty() {
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
    let cand_idx: Vec<usize> = match cand_names {
        Some(names) => names.iter().map(|n| st.register(n)).collect(),
        None => (0..st.agents.len()).collect(),
    };
    patch.apply(st);
    st.calls += 1;
    let mut rng = Rng::new(seed);
    let mut team = st.mesh.route_k(&task, &cand_idx, &mut rng, k);
    let prior_out = prior.map(|p| {
        let (w, followed) = follow_prior(st, &p, &task, &cand_idx, seed);
        if let Some(i) = followed {
            team.retain(|&a| a != i);
            team.insert(0, i);
            team.truncate(k);
        }
        (w, followed.is_some())
    });
    let all_scores = st.mesh.leader().scores(&task);
    let scores: serde_json::Map<String, Value> =
        cand_idx.iter().map(|&i| (st.agents[i].clone(), json!(r6(all_scores.get(i).copied().unwrap_or(0.0))))).collect();
    let health: serde_json::Map<String, Value> =
        cand_idx.iter().map(|&i| (st.agents[i].clone(), json!(r6(st.mesh.health(i))))).collect();
    let risk: serde_json::Map<String, Value> =
        cand_idx.iter().map(|&i| (st.agents[i].clone(), json!(r6(st.mesh.dispatch_risk(i))))).collect();
    let names: Vec<&str> = team.iter().map(|&i| st.agents[i].as_str()).collect();
    let mut out = json!({
        "ok": true,
        "agent": names.first(),
        "team": names,
        "scores": scores,
        "health": health,
        "features": active_features(&task),
    });
    // only once an environment failure was ever reported (keeps the response shape of existing callers unchanged)
    if cand_idx.iter().any(|&i| st.mesh.dispatch_risk(i) > 0.0) {
        out["dispatch_risk"] = Value::Object(risk);
    }
    if let Some((w, followed)) = prior_out {
        out["prior_weight"] = json!(r6(w));
        out["followed_prior"] = json!(followed);
    }
    Ok(out)
}

/// Why a dispatch failed (`feedback --failure-class`, opt-in, R117). `Skill` (default) is the original behaviour. `Spec` (the task/spec was the
/// problem) updates health/edges but NOT the skill weights (`feedback_opt(update_leader = false)`). `Env` (rate limit, turn limit, worktree block)
/// touches neither skill weights nor health/benching: it only updates the agent's dispatch-risk score, which `route` reports.
#[derive(Clone, Copy, PartialEq, Debug)]
enum FailClass {
    Skill,
    Spec,
    Env,
}

fn fail_class(o: &Opts) -> Result<FailClass, CliError> {
    match o.get("failure-class").map(str::trim) {
        None | Some("skill") => Ok(FailClass::Skill),
        Some("spec") => Ok(FailClass::Spec),
        Some("env") => Ok(FailClass::Env),
        Some(x) => Err(usage(format!("--failure-class must be skill, spec or env (got '{x}')"))),
    }
}

/// True when this feedback must NOT enter the durable journal (the journal replays rewards into the skill weights).
fn skips_journal(o: &Opts) -> bool {
    matches!(o.get("failure-class").map(str::trim), Some("spec") | Some("env"))
}

/// `feedback --label X [--agent routed]`: the correct agent is known (supervised update, R117). The label gets reward 1; a different routed agent
/// gets a leader-only reward 0 for this wording (no health hit: it did not fail, it was just not the right agent).
fn cmd_feedback_label(st: &mut State, o: &Opts, patch: ConfigPatch) -> Result<Value, CliError> {
    if o.get("failure-class").is_some_and(|c| c.trim() != "skill") || o.get("team").is_some() || o.get("reward").is_some() {
        return Err(usage("--label cannot be combined with --failure-class spec|env, --team or --reward"));
    }
    let label = o.req("label")?.trim().to_string();
    if label.is_empty() || label.contains(',') {
        return Err(usage("--label must be a single non-empty agent name"));
    }
    let routed = match o.get("agent") {
        Some(a) if a.trim().is_empty() || a.contains(',') => return Err(usage("--agent must be a single non-empty name")),
        Some(a) => Some(a.trim().to_string()),
        None => None,
    };
    let task = make_task(o, st.feedbacks)?;
    let li = st.register(&label);
    let ri = routed.as_deref().map(|r| st.register(r));
    patch.apply(st);
    st.mesh.feedback_label(&task, ri, li);
    st.feedbacks += 1;
    for f in &task.features {
        *st.feature_counts.entry(f.clone()).or_insert(0) += 1;
    }
    Ok(json!({
        "ok": true,
        "agent": label,
        "label": label,
        "routed": routed,
        "correct": routed.as_deref().map(|r| r == label),
        "health": {label.clone(): r6(st.mesh.health(li))},
    }))
}

fn cmd_feedback(st: &mut State, o: &Opts) -> Result<Value, CliError> {
    let patch = ConfigPatch::parse(o, false)?;
    let class = fail_class(o)?;
    if o.get("label").is_some() {
        return cmd_feedback_label(st, o, patch);
    }
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
    patch.apply(st);
    match class {
        FailClass::Skill => st.mesh.feedback(&task, &team, reward),
        FailClass::Spec => st.mesh.feedback_opt(&task, &team, reward, false),
        FailClass::Env => st.mesh.feedback_env(team[0], reward < 0.5),
    }
    if class == FailClass::Skill {
        st.feedbacks += 1;
        for f in &task.features {
            *st.feature_counts.entry(f.clone()).or_insert(0) += 1;
        }
    }
    let health: serde_json::Map<String, Value> =
        team.iter().map(|&i| (st.agents[i].clone(), json!(r6(st.mesh.health(i))))).collect();
    let mut edges = Vec::new();
    for (x, &a) in team.iter().enumerate() {
        for &b in &team[x + 1..] {
            edges.push(json!({"a": st.agents[a], "b": st.agents[b], "w": r6(st.mesh.edge(a, b))}));
        }
    }
    let mut out = json!({
        "ok": true,
        "agent": names[0],
        "team": names,
        "reward": reward,
        "health": health,
        "edges": edges,
    });
    // extra fields only for the opt-in classes (keeps the response shape of existing callers unchanged)
    match class {
        FailClass::Skill => {}
        FailClass::Spec => out["failure_class"] = json!("spec"),
        FailClass::Env => {
            out["failure_class"] = json!("env");
            out["dispatch_risk"] = json!(r6(st.mesh.dispatch_risk(team[0])));
        }
    }
    Ok(out)
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

/// Replay the journal (namespace `world`) into a fresh state: agents are registered by name, outcomes train the leader router.
fn warm_from_store(st: &mut State, sv: &neural_mesh::store::MeshStore, world: &str) -> Result<(), CliError> {
    let os = sv.outcomes().map_err(|e| CliError::Io(e.to_string()))?;
    for o in os.into_iter().filter(|o| o.world == world) {
        st.store_seen.insert(o.id.clone());
        let a = st.register(&o.agent);
        let task = Task { id: st.feedbacks, task_type: o.task_type.clone(), features: o.features.clone() };
        st.mesh.feedback(&task, &[a], o.reward.clamp(0.0, 1.0));
        st.feedbacks += 1;
        for f in &task.features {
            *st.feature_counts.entry(f.clone()).or_insert(0) += 1;
        }
    }
    Ok(())
}

/// B21 (opt-in `--store-sync 1`): a long-lived state learns from journal outcomes of this world that it has NOT seen yet (typically other machines', delivered by
/// `meshstore sync`), without re-applying its own. Refuses states created before `store_seen` existed (they could not tell own from foreign outcomes: double counting).
fn refresh_from_store(st: &mut State, sv: &neural_mesh::store::MeshStore, world: &str) -> Result<usize, CliError> {
    if st.store_seen.is_empty() && st.feedbacks > 0 {
        return Err(CliError::Io("--store-sync: this state predates store tracking and cannot tell its own outcomes from foreign ones; delete it to re-warm from the store".into()));
    }
    let os = sv.outcomes().map_err(|e| CliError::Io(e.to_string()))?;
    let mut n = 0;
    let fresh: Vec<_> = os.into_iter().filter(|o| o.world == world && !st.store_seen.contains(&o.id)).collect();
    for o in fresh {
        st.store_seen.insert(o.id.clone());
        let a = st.register(&o.agent);
        let task = Task { id: st.feedbacks, task_type: o.task_type.clone(), features: o.features.clone() };
        st.mesh.feedback(&task, &[a], o.reward.clamp(0.0, 1.0));
        st.feedbacks += 1;
        for f in &task.features {
            *st.feature_counts.entry(f.clone()).or_insert(0) += 1;
        }
        n += 1;
    }
    Ok(n)
}

/// Append the feedback just applied to the shared journal (one line, atomic under the store lock).
fn append_to_store(st: &mut State, sv: &neural_mesh::store::MeshStore, o: &Opts, world: &str) -> Result<(), CliError> {
    let task = make_task(o, st.feedbacks)?;
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let session = format!("cli-{}-{ms}", std::process::id());
    let rec = neural_mesh::store::Outcome {
        id: format!("{session}:{}", st.feedbacks),
        session,
        seq: st.feedbacks,
        ts: ms,
        world: world.to_string(),
        features: task.features,
        task_type: task.task_type,
        agent: o.req("agent")?.trim().to_string(),
        reward: o.req("reward")?.trim().parse::<f64>().unwrap_or(0.0),
    };
    sv.append_outcome(&rec).map_err(|e| CliError::Io(e.to_string()))?;
    st.store_seen.insert(rec.id);
    Ok(())
}

const FRONTEND_KEYS: &[&str] = &["enabled", "min-conf", "max-local-tokens", "ctx-budget", "max-chunks", "min-chunk-score", "min-sim", "cache-hit", "tool", "local-conf", "local-tokens", "max-sim", "chunks"];

/// `frontend` (opt-in, stateless, default OFF): local-vs-forward decision + trimmed context ids (src/frontend.rs). `--chunks` is a JSON
/// array of {"id","score","tokens"}. Without `--enabled 1` it always answers "forward" (today's behaviour).
fn cmd_frontend(o: &Opts) -> Result<Value, CliError> {
    use neural_mesh::frontend::{Chunk, FrontendCfg, Signals, decide, trim_context};
    fn num<T: std::str::FromStr>(o: &Opts, k: &str, d: T) -> Result<T, CliError> {
        match o.get(k) {
            Some(v) => v.trim().parse::<T>().map_err(|_| usage(format!("--{k} must be a number"))),
            None => Ok(d),
        }
    }
    let flag = |k: &str| matches!(o.get(k).map(str::trim), Some("1") | Some("true"));
    let d = FrontendCfg::default();
    let cfg = FrontendCfg {
        enabled: flag("enabled"),
        min_conf: num(o, "min-conf", d.min_conf)?,
        max_local_tokens: num(o, "max-local-tokens", d.max_local_tokens)?,
        ctx_budget: num(o, "ctx-budget", d.ctx_budget)?,
        max_chunks: num(o, "max-chunks", d.max_chunks)?,
        min_chunk_score: num(o, "min-chunk-score", d.min_chunk_score)?,
        min_sim: num(o, "min-sim", d.min_sim)?,
    };
    let opt_f = |k: &str| -> Result<Option<f32>, CliError> { o.get(k).map(|v| v.trim().parse::<f32>().map_err(|_| usage(format!("--{k} must be a number")))).transpose() };
    let sig = Signals {
        cache_hit: flag("cache-hit"),
        tool: o.get("tool").map(|t| t.trim().to_string()).filter(|t| !t.is_empty()),
        local_conf: opt_f("local-conf")?,
        local_tokens: num(o, "local-tokens", 0usize)?,
        max_sim: opt_f("max-sim")?,
    };
    let chunks: Vec<Chunk> = match o.get("chunks") {
        Some(js) => {
            let v: Value = serde_json::from_str(js).map_err(|e| usage(format!("--chunks is not JSON: {e}")))?;
            let arr = v.as_array().ok_or_else(|| usage("--chunks must be a JSON array"))?;
            arr.iter()
                .map(|c| {
                    Some(Chunk { id: c.get("id")?.as_str()?.to_string(), score: c.get("score")?.as_f64()? as f32, tokens: c.get("tokens")?.as_u64()? as usize })
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| usage("each chunk needs id (string), score (number), tokens (integer)"))?
        }
        None => Vec::new(),
    };
    let dec = decide(&cfg, &sig);
    let (ctx, used) = if dec.is_local() { (Vec::new(), 0) } else { trim_context(&chunks, cfg.ctx_budget, cfg.max_chunks, cfg.min_chunk_score) };
    let tool = match &dec {
        neural_mesh::frontend::Decision::Tool(t) => Some(t.clone()),
        _ => None,
    };
    Ok(json!({"ok": true, "enabled": cfg.enabled, "decision": dec.name(), "tool": tool, "context": ctx, "context_tokens": used}))
}

/// Journal lines of a `feedback --label` update: (label, 1.0) and, if a different routed agent was given, (routed, 0.0).
fn append_label_to_store(st: &mut State, sv: &neural_mesh::store::MeshStore, o: &Opts, world: &str) -> Result<(), CliError> {
    let task = make_task(o, st.feedbacks.saturating_sub(1))?;
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let session = format!("cli-{}-{ms}", std::process::id());
    let label = o.req("label")?.trim().to_string();
    let mut lines = vec![(label.clone(), 1.0)];
    if let Some(r) = o.get("agent").map(|a| a.trim().to_string()).filter(|r| *r != label) {
        lines.push((r, 0.0));
    }
    for (k, (agent, reward)) in lines.into_iter().enumerate() {
        let seq = st.feedbacks.saturating_sub(1) * 2 + k as u64;
        let rec = neural_mesh::store::Outcome {
            id: format!("{session}:{seq}"),
            session: session.clone(),
            seq,
            ts: ms,
            world: world.to_string(),
            features: task.features.clone(),
            task_type: task.task_type.clone(),
            agent,
            reward,
        };
        sv.append_outcome(&rec).map_err(|e| CliError::Io(e.to_string()))?;
        st.store_seen.insert(rec.id);
    }
    Ok(())
}

/// `merge-state --into A --from B [--out O]`: fold router state B into A (written to O, default A, atomically, under both state locks).
///
/// Merge rule (documented, deterministic; A keeps its seed, prior_k, Hebbian and mesh config, route clock, gate/trust/context tables):
/// * agents: A's roster in A's order, then B's agents A does not know (appended, as `register` would);
/// * skill weight (feature f, agent i) and the per-feature baseline: mean of the two states weighted by each side's evidence for f
///   (`feature_counts[f]`), counting a side only if it knows agent i (a side that never saw agent i has no evidence about it). No evidence
///   on either side keeps the neutral init. Rows are then re-capped at `row_cap` like `update` does;
/// * per-agent health / edge baseline / dispatch risk and pairwise edges: shared agents (pairs) get the mean weighted by the states' total
///   feedback counts; one-sided agents keep their own side's value. fail streak and retry multiplier take the max, probation the OR;
///   an agent benched in either state is benched from A's current route count;
/// * counters: feedbacks, calls, updates, feature_counts and team statistics are summed; store_seen is the union.
fn merge_states(a: State, b: State) -> Result<State, CliError> {
    let ser = |e: serde_json::Error| CliError::Io(format!("merge: {e}"));
    let (na, nb) = (a.feedbacks as f64, b.feedbacks as f64);
    let len_a = a.agents.len();
    let fc_a = a.feature_counts.clone();
    let fc_b = b.feature_counts.clone();
    let mut m = a;
    for name in &b.agents {
        m.register(name);
    }
    let n = m.agents.len();
    // union index -> B index
    let b_of: Vec<Option<usize>> = m.agents.iter().map(|nm| b.index_of(nm)).collect();
    let a_knows = |u: usize| u < len_a;
    let mut mv = serde_json::to_value(&m).map_err(ser)?;
    let bv = serde_json::to_value(&b).map_err(ser)?;
    let init_w = neural_mesh::hebbian::INIT_WEIGHT;
    let num = |v: &Value| v.as_f64().unwrap_or(0.0);

    // leader: weights + baselines
    {
        let la = &bv["mesh"]["leader"];
        let lm = &mut mv["mesh"]["leader"];
        let cap = lm["cfg"]["row_cap"].as_f64().unwrap_or(0.0);
        let wa = lm["weights"].as_object().cloned().unwrap_or_default();
        let wb = la["weights"].as_object().cloned().unwrap_or_default();
        let mut feats: Vec<String> = wa.keys().chain(wb.keys()).cloned().collect();
        feats.sort();
        feats.dedup();
        let mut out = serde_json::Map::new();
        for f in &feats {
            let (ka, kb) = (fc_a.get(f).copied().unwrap_or(0) as f64, fc_b.get(f).copied().unwrap_or(0) as f64);
            let ra = wa.get(f).and_then(Value::as_array);
            let rb = wb.get(f).and_then(Value::as_array);
            let mut row = Vec::with_capacity(n);
            for u in 0..n {
                let mut sides: Vec<(f64, f64)> = Vec::new(); // (evidence, value)
                if a_knows(u) {
                    if let Some(r) = ra {
                        sides.push((ka, r.get(u).map(num).unwrap_or(init_w)));
                    }
                }
                if let (Some(bi), Some(r)) = (b_of[u], rb) {
                    sides.push((kb, r.get(bi).map(num).unwrap_or(init_w)));
                }
                let den: f64 = sides.iter().map(|s| s.0).sum();
                let w = if den > 0.0 {
                    sides.iter().map(|s| s.0 * s.1).sum::<f64>() / den
                } else if !sides.is_empty() {
                    sides.iter().map(|s| s.1).sum::<f64>() / sides.len() as f64
                } else {
                    init_w
                };
                row.push(w);
            }
            let sum: f64 = row.iter().sum();
            if cap > 0.0 && sum > cap {
                let k = cap / sum;
                row.iter_mut().for_each(|w| *w *= k);
            }
            out.insert(f.clone(), json!(row));
        }
        lm["weights"] = Value::Object(out);
        let ba = lm["baselines"].as_object().cloned().unwrap_or_default();
        let bb = la["baselines"].as_object().cloned().unwrap_or_default();
        let mut bo = serde_json::Map::new();
        for f in ba.keys().chain(bb.keys()) {
            if bo.contains_key(f) {
                continue;
            }
            let (ka, kb) = (fc_a.get(f).copied().unwrap_or(0) as f64, fc_b.get(f).copied().unwrap_or(0) as f64);
            let sides: Vec<(f64, f64)> =
                [(ka, ba.get(f)), (kb, bb.get(f))].into_iter().filter_map(|(k, v)| v.map(|v| (k, num(v)))).collect();
            let den: f64 = sides.iter().map(|s| s.0).sum();
            let v = if den > 0.0 {
                sides.iter().map(|s| s.0 * s.1).sum::<f64>() / den
            } else {
                sides.iter().map(|s| s.1).sum::<f64>() / sides.len() as f64
            };
            bo.insert(f.clone(), json!(v));
        }
        lm["baselines"] = Value::Object(bo);
        let upd = lm["updates"].as_u64().unwrap_or(0) + la["updates"].as_u64().unwrap_or(0);
        lm["updates"] = json!(upd);
    }

    // mesh: per-agent tables and edges
    {
        let mb = &bv["mesh"];
        let mm = &mut mv["mesh"];
        let tot = if na + nb > 0.0 { na + nb } else { 2.0 };
        let (wa_, wb_) = if na + nb > 0.0 { (na / tot, nb / tot) } else { (0.5, 0.5) };
        let blend = |mm: &mut Value, key: &str, default_b: f64| {
            for u in 0..n {
                let Some(bi) = b_of[u] else { continue };
                let xb = mb[key].get(bi).map(num).unwrap_or(default_b);
                let x = if a_knows(u) { wa_ * num(&mm[key][u]) + wb_ * xb } else { xb };
                mm[key][u] = json!(x);
            }
        };
        blend(mm, "health", 1.0);
        blend(mm, "edge_baseline", 0.5);
        blend(mm, "dispatch_risk", 0.0);
        let routes_a = mm["routes"].as_u64().unwrap_or(0);
        for u in 0..n {
            let Some(bi) = b_of[u] else { continue };
            let sb = mb["fail_streak"].get(bi).and_then(Value::as_u64).unwrap_or(0);
            let sa = if a_knows(u) { mm["fail_streak"][u].as_u64().unwrap_or(0) } else { 0 };
            mm["fail_streak"][u] = json!(sa.max(sb));
            let rb = mb["retry_mult"].get(bi).map(num).unwrap_or(1.0);
            let ra = if a_knows(u) { num(&mm["retry_mult"][u]) } else { 1.0 };
            mm["retry_mult"][u] = json!(ra.max(rb));
            let pb = mb["probation"].get(bi).and_then(Value::as_bool).unwrap_or(false);
            let pa = a_knows(u) && mm["probation"][u].as_bool().unwrap_or(false);
            mm["probation"][u] = json!(pa || pb);
            let bench_a = a_knows(u) && !mm["benched_since"][u].is_null();
            let bench_b = mb["benched_since"].get(bi).is_some_and(|v| !v.is_null());
            if !bench_a && bench_b {
                mm["benched_since"][u] = json!(routes_a);
            }
        }
        for u in 0..n {
            for v in 0..n {
                let (Some(bu), Some(bw)) = (b_of[u], b_of[v]) else { continue };
                let xb = mb["edges"].get(bu).and_then(|r| r.get(bw)).map(num).unwrap_or(0.0);
                let x = if a_knows(u) && a_knows(v) { wa_ * num(&mm["edges"][u][v]) + wb_ * xb } else { xb };
                mm["edges"][u][v] = json!(x);
            }
        }
        for k in ["team_size_sum", "teams_routed"] {
            let s = mm[k].as_u64().unwrap_or(0) + mb[k].as_u64().unwrap_or(0);
            mm[k] = json!(s);
        }
    }

    // state-level counters
    mv["feedbacks"] = json!(m.feedbacks + b.feedbacks);
    mv["calls"] = json!(m.calls + b.calls);
    let mut fc = fc_a;
    for (f, c) in &b.feature_counts {
        *fc.entry(f.clone()).or_insert(0) += c;
    }
    mv["feature_counts"] = json!(fc);
    let mut seen = m.store_seen.clone();
    seen.extend(b.store_seen.iter().cloned());
    mv["store_seen"] = json!(seen);
    let out: State = serde_json::from_value(mv).map_err(ser)?;
    if out.mesh.n_agents() != out.agents.len() {
        return Err(CliError::Io("merge produced an inconsistent agent table".into()));
    }
    Ok(out)
}

/// `merge-state --into A --from B [--out O]` (see [`merge_states`]). Both inputs must exist; A and B must differ.
fn cmd_merge_state(o: &Opts) -> Result<Value, CliError> {
    let into = PathBuf::from(o.req("into")?);
    let from = PathBuf::from(o.req("from")?);
    let out = o.get("out").map(PathBuf::from).unwrap_or_else(|| into.clone());
    for p in [&into, &from] {
        if !p.exists() {
            return Err(usage(format!("state file {} does not exist", p.display())));
        }
    }
    let canon = |p: &Path| fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    if canon(&into) == canon(&from) {
        return Err(usage("--into and --from are the same file"));
    }
    let _la = StateLock::acquire(&into)?;
    let _lb = StateLock::acquire(&from)?;
    let _lo = if canon(&out) != canon(&into) && canon(&out) != canon(&from) { Some(StateLock::acquire(&out)?) } else { None };
    let a = load_state(&into)?;
    let b = load_state(&from)?;
    let before = json!({"into": {"agents": a.agents.len(), "feedbacks": a.feedbacks}, "from": {"agents": b.agents.len(), "feedbacks": b.feedbacks}});
    let shared: Vec<String> = b.agents.iter().filter(|x| a.index_of(x).is_some()).cloned().collect();
    let merged = merge_states(a, b)?;
    save_state(&out, &merged)?;
    Ok(json!({
        "ok": true,
        "out": out.display().to_string(),
        "before": before,
        "after": {"agents": merged.agents.len(), "feedbacks": merged.feedbacks},
        "shared_agents": shared,
    }))
}

/// Library-level entry point (used by tests): runs one command and persists
/// state when the command succeeds.
fn run(args: &[String]) -> Result<Value, CliError> {
    let Some(cmd) = args.first() else { return Err(usage("missing command")) };
    let rest = &args[1..];
    if cmd == "frontend" {
        return cmd_frontend(&Opts::parse(rest, FRONTEND_KEYS)?);
    }
    if cmd == "merge-state" {
        return cmd_merge_state(&Opts::parse(rest, &["into", "from", "out"])?);
    }
    let base: &[&str] = match cmd.as_str() {
        "route" => &["state", "features", "candidates", "team", "type", "seed", "prior", "store", "world", "store-sync"],
        "feedback" => &["state", "features", "agent", "team", "reward", "type", "store", "world", "store-sync", "failure-class", "label"],
        "stats" => &["state", "store", "world"],
        "config" => &["state", "team"],
        other => return Err(usage(format!("unknown command '{other}'"))),
    };
    let allowed: Vec<&str> = base.iter().chain(CONFIG_KEYS.iter()).copied().collect();
    let o = Opts::parse(rest, &allowed)?;
    let path = PathBuf::from(o.req("state")?);
    let _lock = StateLock::acquire(&path)?;
    let existed = path.exists();
    let mut st = load_state(&path)?;
    // opt-in durable shared memory (src/store.rs): a NEW state file is warm-started from the journal
    let store = match o.get("store") {
        Some(dir) => Some(neural_mesh::store::MeshStore::open(dir).map_err(|e| CliError::Io(e.to_string()))?),
        None => None,
    };
    let world = o.get("world").unwrap_or("default").to_string();
    if let (Some(sv), false) = (&store, existed) {
        warm_from_store(&mut st, sv, &world)?;
    } else if let (Some(sv), true) = (&store, existed) {
        if matches!(o.get("store-sync"), Some("1") | Some("true")) {
            refresh_from_store(&mut st, sv, &world)?;
        }
    }
    let out = match cmd.as_str() {
        "route" => cmd_route(&mut st, &o)?,
        "feedback" => {
            let v = cmd_feedback(&mut st, &o)?;
            if let Some(sv) = store.as_ref().filter(|_| !skips_journal(&o)) {
                if o.get("label").is_some() {
                    append_label_to_store(&mut st, sv, &o, &world)?;
                } else {
                    append_to_store(&mut st, sv, &o, &world)?;
                }
            }
            v
        }
        "config" => {
            let v = cmd_config(&mut st, &o)?;
            if existed && !has_config_opts(&o, true) {
                return Ok(v);
            }
            v
        }
        _ => {
            ConfigPatch::parse(&o, false)?.apply(&mut st);
            let v = cmd_stats(&st);
            if !existed || has_config_opts(&o, false) {
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
    if args.first().is_some_and(|a| a == "serve") {
        std::process::exit(serve::main(&args[1..]));
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

    #[test]
    fn frontend_cli_is_off_by_default_and_trims_context() {
        let a = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let v = run(&a(&["frontend", "--local-conf", "0", "--local-tokens", "5", "--max-sim", "0.9"])).unwrap();
        assert_eq!(v["decision"], "forward");
        let v = run(&a(&["frontend", "--enabled", "1", "--local-conf", "0", "--local-tokens", "5", "--max-sim", "0.9"])).unwrap();
        assert_eq!(v["decision"], "local");
        let v = run(&a(&["frontend", "--enabled", "1", "--local-conf", "0", "--local-tokens", "5", "--max-sim", "0.1", "--ctx-budget", "500",
            "--chunks", r#"[{"id":"x","score":0.9,"tokens":600},{"id":"y","score":0.5,"tokens":400}]"#])).unwrap();
        assert_eq!((v["decision"].as_str(), v["context"][0].as_str(), v["context_tokens"].as_u64()), (Some("forward"), Some("y"), Some(400)));
        assert!(run(&a(&["frontend", "--chunks", "[1]"])).is_err());
        assert!(run(&a(&["frontend", "--local-conf", "abc"])).is_err());
        assert!(run(&a(&["frontend", "--bogus", "1"])).is_err());
    }

    #[test]
    fn env_failure_class_leaves_skill_weights_and_health_alone() {
        let p = tmp("failclass");
        let a = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let st = p.to_str().unwrap();
        for _ in 0..3 {
            run(&a(&["feedback", "--state", st, "--features", "x", "--agent", "a", "--reward", "1"])).unwrap();
            run(&a(&["feedback", "--state", st, "--features", "x", "--agent", "b", "--reward", "0.5"])).unwrap();
        }
        let before = run(&a(&["stats", "--state", st])).unwrap();
        for _ in 0..10 {
            let v = run(&a(&["feedback", "--state", st, "--features", "x", "--agent", "a", "--reward", "0", "--failure-class", "env"])).unwrap();
            assert_eq!(v["failure_class"], "env");
        }
        let after = run(&a(&["stats", "--state", st])).unwrap();
        assert_eq!(before["weights"], after["weights"]);
        assert_eq!(before["agents"], after["agents"]);
        assert_eq!(before["feedbacks"], after["feedbacks"]);
        let r = run(&a(&["route", "--state", st, "--features", "x"])).unwrap();
        assert!(r["dispatch_risk"]["a"].as_f64().unwrap() > 0.8 && r["dispatch_risk"]["b"].as_f64().unwrap() == 0.0);
        // spec class: health/edges move, skill weights do not
        run(&a(&["feedback", "--state", st, "--features", "x", "--agent", "b", "--reward", "0", "--failure-class", "spec"])).unwrap();
        let spec = run(&a(&["stats", "--state", st])).unwrap();
        assert_eq!(before["weights"], spec["weights"]);
        assert!(run(&a(&["feedback", "--state", st, "--features", "x", "--agent", "a", "--reward", "0", "--failure-class", "bogus"])).is_err());
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn label_feedback_credits_the_label_and_penalises_the_wrong_pick_leader_only() {
        let p = tmp("label");
        let a = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let st = p.to_str().unwrap();
        for _ in 0..20 {
            let v = run(&a(&["feedback", "--state", st, "--features", "fix,round", "--agent", "wrong", "--label", "builder"])).unwrap();
            assert_eq!(v["correct"], false);
        }
        let r = run(&a(&["route", "--state", st, "--features", "fix,round", "--candidates", "wrong,builder"])).unwrap();
        assert_eq!(r["agent"], "builder");
        let s = run(&a(&["stats", "--state", st])).unwrap();
        let h = |n: &str| s["agents"].as_array().unwrap().iter().find(|x| x["name"] == n).unwrap()["health"].as_f64().unwrap();
        assert!(h("wrong") >= 0.99, "the wrong pick did not fail, so its health is untouched");
        assert!(run(&a(&["feedback", "--state", st, "--features", "x", "--label", "b", "--reward", "1"])).is_err());
        assert!(run(&a(&["feedback", "--state", st, "--features", "x", "--label", "b", "--failure-class", "env"])).is_err());
        let _ = fs::remove_file(&p);
    }

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

    #[test]
    fn concurrent_feedback_loses_no_updates() {
        let dir = std::env::temp_dir().join(format!("nm-conc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let p = dir.join("state.json");
        let ps = p.to_str().unwrap().to_string();
        call(&["feedback", "--state", &ps, "--features", "x", "--agent", "a", "--reward", "1"]).unwrap();
        let before = call(&["stats", "--state", &ps]).unwrap();
        let hs: Vec<_> = (0..16)
            .map(|i| {
                let ps = ps.clone();
                std::thread::spawn(move || {
                    let r = if i % 2 == 0 { "1" } else { "0" };
                    call(&["feedback", "--state", &ps, "--features", "x", "--agent", "a", "--reward", r]).unwrap();
                })
            })
            .collect();
        for h in hs { h.join().unwrap(); }
        let after = call(&["stats", "--state", &ps]).unwrap();
        let n = |v: &Value| v.to_string().matches("\"updates\":").count();
        assert!(n(&after) > 0, "stats should expose an updates counter: {after}");
        let count = |v: &Value| -> u64 {
            fn find(v: &Value) -> Option<u64> {
                match v {
                    Value::Object(m) => m.get("updates").and_then(Value::as_u64).or_else(|| m.values().find_map(find)),
                    Value::Array(a) => a.iter().find_map(find),
                    _ => None,
                }
            }
            find(v).unwrap()
        };
        assert_eq!(count(&after) - count(&before), 16, "every concurrent feedback must be applied");
        assert!(!dir.join("state.json.lock").exists(), "lock must be released");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_features_use_no_keyword_row() {
        let dir = std::env::temp_dir().join(format!("nm-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let ps = dir.join("s.json").to_str().unwrap().to_string();
        for _ in 0..30 {
            call(&["feedback", "--state", &ps, "--features", "", "--agent", "planner", "--reward", "1"]).unwrap();
            call(&["feedback", "--state", &ps, "--features", "", "--agent", "coder", "--reward", "0"]).unwrap();
        }
        let v = call(&["route", "--state", &ps, "--features", "", "--candidates", "planner,coder"]).unwrap();
        assert_eq!(v["agent"], "planner", "{v}");
        let _ = fs::remove_dir_all(&dir);
    }

    // ---- robust defaults, config, warm prior ----------------------------------

    fn opts(pairs: &[(&str, &str)]) -> Opts {
        Opts(pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
    }

    fn cfg_of(v: &Value) -> &Value {
        &v["config"]
    }

    #[test]
    fn new_state_has_robust_defaults() {
        let p = tmp("defaults");
        let ps = p.to_str().unwrap();
        let s = call(&["stats", "--state", ps]).unwrap();
        assert_eq!(s["hebbian"]["eta"], 0.01);
        assert_eq!(s["hebbian"]["temperature"], 0.002);
        assert_eq!(s["hebbian"]["decay"], 0.01);
        assert_eq!(s["hebbian"]["epsilon"], 0.0);
        assert_eq!(s["mesh"]["team_size"], 1);
        let c = call(&["config", "--state", ps]).unwrap();
        assert_eq!(cfg_of(&c), &json!({"eta": 0.01, "temperature": 0.002, "decay": 0.01, "epsilon": 0.0, "team": 1, "prior_k": 1.0}));
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn old_state_keeps_its_stored_config() {
        let p = tmp("old");
        let ps = p.to_str().unwrap();
        call(&["feedback", "--state", ps, "--features", "x", "--agent", "a", "--reward", "1"]).unwrap();
        // Rewrite as a pre-change file: legacy hebbian numbers, no prior_k / feature_counts.
        let mut v: Value = serde_json::from_str(&fs::read_to_string(&p).unwrap()).unwrap();
        let legacy = serde_json::to_value(HebbianConfig::default()).unwrap();
        v["mesh"]["leader"]["cfg"] = legacy;
        let o = v.as_object_mut().unwrap();
        o.remove("prior_k");
        o.remove("feature_counts");
        fs::write(&p, v.to_string()).unwrap();
        let c = call(&["config", "--state", ps]).unwrap();
        assert_eq!(cfg_of(&c), &json!({"eta": 0.5, "temperature": 0.05, "decay": 0.01, "epsilon": 0.02, "team": 1, "prior_k": 1.0}));
        call(&["route", "--state", ps, "--features", "x"]).unwrap();
        let s = call(&["stats", "--state", ps]).unwrap();
        assert_eq!(s["hebbian"]["eta"], 0.5);
        assert_eq!(s["hebbian"]["temperature"], 0.05);
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn config_subcommand_round_trips_and_validates() {
        let p = tmp("cfg");
        let ps = p.to_str().unwrap();
        let c = call(&["config", "--state", ps, "--eta", "0.2", "--temperature=0.1", "--decay", "0.05", "--epsilon", "0.03", "--team", "3", "--prior-k", "4"]).unwrap();
        let want = json!({"eta": 0.2, "temperature": 0.1, "decay": 0.05, "epsilon": 0.03, "team": 3, "prior_k": 4.0});
        assert_eq!(cfg_of(&c), &want);
        assert_eq!(cfg_of(&call(&["config", "--state", ps]).unwrap()), &want, "persisted");
        let c = call(&["config", "--state", ps, "--eta", "0.3"]).unwrap();
        assert_eq!(cfg_of(&c)["eta"], 0.3);
        assert_eq!(cfg_of(&c)["team"], 3);
        // flags on stats / feedback / route persist too; route --team stays per call
        call(&["stats", "--state", ps, "--decay", "0.02"]).unwrap();
        call(&["feedback", "--state", ps, "--features", "x", "--agent", "a", "--reward", "1", "--epsilon", "0.1"]).unwrap();
        call(&["route", "--state", ps, "--features", "x", "--prior-k", "2", "--team", "1"]).unwrap();
        let c = call(&["config", "--state", ps]).unwrap();
        let c = cfg_of(&c);
        assert_eq!((&c["decay"], &c["epsilon"], &c["prior_k"], &c["team"]), (&json!(0.02), &json!(0.1), &json!(2.0), &json!(3)));
        for bad in [["--eta", "-1"], ["--eta", "2"], ["--temperature", "x"], ["--decay", "NaN"], ["--epsilon", "1.5"], ["--team", "0"], ["--prior-k", "-1"]] {
            assert!(call(&["config", "--state", ps, bad[0], bad[1]]).is_err(), "{bad:?}");
        }
        assert!(call(&["config", "--state", ps, "--eta", "0.1", "--eta", "0.2"]).is_err());
        assert_eq!(cfg_of(&call(&["config", "--state", ps]).unwrap())["eta"], 0.3, "failed calls change nothing");
        let v = call(&["route", "--state", ps, "--features", "x", "--candidates", "a,b,c"]).unwrap();
        assert_eq!(v["team"].as_array().unwrap().len(), 3, "persisted team size is the route default");
        let _ = fs::remove_file(&p);
    }

    #[test]
    fn prior_followed_at_zero_evidence_and_weight_decays() {
        let mut st = State::new();
        for seed in 0..20 {
            let sd = seed.to_string();
            let v = cmd_route(&mut st, &opts(&[("features", "f"), ("candidates", "a,b,c"), ("prior", "c"), ("seed", &sd)])).unwrap();
            assert_eq!(v["prior_weight"], 1.0);
            assert_eq!(v["followed_prior"], true);
            assert_eq!(v["agent"], "c");
        }
        let mut last = 1.0;
        for n in 1..=6 {
            cmd_feedback(&mut st, &opts(&[("features", "f"), ("agent", "a"), ("reward", "1")])).unwrap();
            let v = cmd_route(&mut st, &opts(&[("features", "f"), ("candidates", "a,b,c"), ("prior", "c")])).unwrap();
            let w = v["prior_weight"].as_f64().unwrap();
            assert!((w - 1.0 / (1.0 + n as f64)).abs() < 1e-5, "n={n} w={w}");
            assert!(w < last);
            last = w;
        }
        // n is the mean over the task's features: f has 6, g has 0 -> 3
        let v = cmd_route(&mut st, &opts(&[("features", "f,g"), ("candidates", "a,b,c"), ("prior", "c")])).unwrap();
        assert!((v["prior_weight"].as_f64().unwrap() - 0.25).abs() < 1e-5);
        // prior_k scales it: k=3, n=6 -> 1/3
        cmd_config(&mut st, &opts(&[("prior-k", "3")])).unwrap();
        let v = cmd_route(&mut st, &opts(&[("features", "f"), ("candidates", "a,b,c"), ("prior", "c")])).unwrap();
        assert!((v["prior_weight"].as_f64().unwrap() - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn prior_is_deterministic_and_rate_matches_weight() {
        let run = || {
            let mut st = State::new();
            for _ in 0..3 {
                cmd_feedback(&mut st, &opts(&[("features", "f"), ("agent", "a"), ("reward", "1")])).unwrap();
            }
            (0..400)
                .map(|i| {
                    let sd = i.to_string();
                    let v = cmd_route(&mut st, &opts(&[("features", "f"), ("candidates", "a,b"), ("prior", "b"), ("seed", &sd)])).unwrap();
                    v["followed_prior"].as_bool().unwrap()
                })
                .collect::<Vec<_>>()
        };
        let (a, b) = (run(), run());
        assert_eq!(a, b, "same seeds, same decisions");
        let rate = a.iter().filter(|&&x| x).count() as f64 / a.len() as f64;
        assert!((rate - 0.25).abs() < 0.07, "w = 1/(1+3) = 0.25, got {rate}");
    }

    #[test]
    fn unknown_or_foreign_prior_is_ignored_safely() {
        let mut st = State::new();
        let v = cmd_route(&mut st, &opts(&[("features", "f"), ("candidates", "a,b"), ("prior", "ghost")])).unwrap();
        assert_eq!(v["followed_prior"], false);
        assert_eq!(v["prior_weight"], 1.0);
        assert!(["a", "b"].contains(&v["agent"].as_str().unwrap()));
        assert_eq!(st.agents, vec!["a", "b"], "unknown prior must not be registered");
        // a known agent that is not among the candidates
        st.register("z");
        let v = cmd_route(&mut st, &opts(&[("features", "f"), ("candidates", "a,b"), ("prior", "z")])).unwrap();
        assert_eq!(v["followed_prior"], false);
        assert_ne!(v["agent"], "z");
        for bad in ["", " ", "a,b"] {
            assert!(cmd_route(&mut st, &opts(&[("features", "f"), ("candidates", "a,b"), ("prior", bad)])).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn calls_without_new_flags_keep_their_shape() {
        let mut st = State::new();
        let v = cmd_route(&mut st, &opts(&[("features", "f"), ("candidates", "a,b")])).unwrap();
        let mut keys: Vec<&String> = v.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["agent", "features", "health", "ok", "scores", "team"]);
        let f = cmd_feedback(&mut st, &opts(&[("features", "f"), ("agent", "a"), ("reward", "1")])).unwrap();
        let mut keys: Vec<&String> = f.as_object().unwrap().keys().collect();
        keys.sort();
        assert_eq!(keys, ["agent", "edges", "health", "ok", "reward", "team"]);
        // New defaults == passing them explicitly (so flags are a pure override).
        let (mut a, mut b) = (State::new(), State::new());
        cmd_config(&mut b, &opts(&[("eta", "0.01"), ("temperature", "0.002"), ("decay", "0.01"), ("epsilon", "0")])).unwrap();
        for i in 0..30 {
            let sd = i.to_string();
            let r = if i % 3 == 0 { "0" } else { "1" };
            for st in [&mut a, &mut b] {
                cmd_feedback(st, &opts(&[("features", "f,g"), ("agent", if i % 2 == 0 { "x" } else { "y" }), ("reward", r)])).unwrap();
            }
            let ra = cmd_route(&mut a, &opts(&[("features", "f"), ("candidates", "x,y"), ("seed", &sd)])).unwrap();
            let rb = cmd_route(&mut b, &opts(&[("features", "f"), ("candidates", "x,y"), ("seed", &sd)])).unwrap();
            assert_eq!(ra.to_string(), rb.to_string());
        }
    }

    /// Sanity run (not a benchmark): 200 synthetic tasks, the caller's prior is right on the
    /// even features and wrong on the odd ones. Prints the curves with `--nocapture`.
    #[test]
    fn warm_prior_sanity_run() {
        const AGENTS: [&str; 4] = ["a0", "a1", "a2", "a3"];
        let mut st = State::new();
        let mut rng = Rng::new(7);
        let cands = AGENTS.join(",");
        let mut rows: Vec<(bool, bool)> = Vec::new(); // (followed prior, chose best agent)
        for i in 0..200u64 {
            let f = rng.below(8);
            let best = f % 4;
            let prior = if f % 2 == 0 { best } else { (best + 1) % 4 };
            let feat = format!("f{f}");
            let sd = i.to_string();
            let v = cmd_route(&mut st, &opts(&[("features", &feat), ("candidates", &cands), ("prior", AGENTS[prior]), ("seed", &sd)])).unwrap();
            let agent = v["agent"].as_str().unwrap().to_string();
            let ok = agent == AGENTS[best];
            let reward = if rng.bernoulli(if ok { 0.9 } else { 0.1 }) { "1" } else { "0" };
            cmd_feedback(&mut st, &opts(&[("features", &feat), ("agent", &agent), ("reward", reward)])).unwrap();
            rows.push((v["followed_prior"].as_bool().unwrap(), ok));
        }
        let rate = |r: &[(bool, bool)], pick: fn(&(bool, bool)) -> bool| r.iter().filter(|x| pick(x)).count() as f64 / r.len() as f64;
        let (head, mid, tail) = (&rows[..50], &rows[75..125], &rows[150..]);
        for (name, w) in [("tasks 1-50", head), ("tasks 76-125", mid), ("tasks 151-200", tail)] {
            println!("{name}: followed_prior {:.2}  accuracy {:.2}", rate(w, |x| x.0), rate(w, |x| x.1));
        }
        println!("keyword-only accuracy would be 0.50");
        assert!(rate(tail, |x| x.0) < rate(head, |x| x.0) - 0.3, "follow rate must fall");
        assert!(rate(tail, |x| x.1) > rate(head, |x| x.1), "accuracy must rise");
        assert!(rate(tail, |x| x.1) > 0.6, "and beat the half-right keyword prior (0.5)");
    }

    fn weight_of(ps: &str, f: &str, agent: &str) -> f64 {
        call(&["stats", "--state", ps]).unwrap()["weights"][f]["w"][agent].as_f64().unwrap()
    }

    #[test]
    fn merge_state_unions_agents_and_weights_by_evidence() {
        let (pa, pb, po) = (tmp("merge-a"), tmp("merge-b"), tmp("merge-o"));
        let (a, b, o) = (pa.to_str().unwrap(), pb.to_str().unwrap(), po.to_str().unwrap());
        // A: 3 feedbacks on feature x (agents a, shared); B: 1 feedback on x, 2 on y (agents shared, bonly)
        for (ag, r) in [("a", "1"), ("shared", "1"), ("shared", "0")] {
            call(&["feedback", "--state", a, "--features", "x", "--agent", ag, "--reward", r]).unwrap();
        }
        call(&["feedback", "--state", b, "--features", "x", "--agent", "shared", "--reward", "1"]).unwrap();
        for _ in 0..2 {
            call(&["feedback", "--state", b, "--features", "y", "--agent", "bonly", "--reward", "1"]).unwrap();
        }
        let (wa, wb) = (weight_of(a, "x", "shared"), weight_of(b, "x", "shared"));
        let wy = weight_of(b, "y", "bonly");
        let wa_a = weight_of(a, "x", "a");
        let v = call(&["merge-state", "--into", a, "--from", b, "--out", o]).unwrap();
        assert_eq!(v["after"]["agents"], 3);
        assert_eq!(v["after"]["feedbacks"], 6);
        assert_eq!(v["shared_agents"], json!(["shared"]));
        let st = load_state(&po).unwrap();
        assert_eq!(st.agents, vec!["a", "shared", "bonly"], "A's order first, then B's new agents");
        assert_eq!(st.feature_counts["x"], 4);
        assert_eq!(st.feature_counts["y"], 2);
        // shared cell: evidence-weighted mean (A has 3 on x, B has 1) -- before the row cap, which these small rows do not hit
        let want = (3.0 * wa + 1.0 * wb) / 4.0;
        assert!((weight_of(o, "x", "shared") - want).abs() < 1e-5, "{} vs {want}", weight_of(o, "x", "shared"));
        // one-sided cells keep their own side's value
        assert!((weight_of(o, "y", "bonly") - wy).abs() < 1e-5);
        assert!((weight_of(o, "x", "a") - wa_a).abs() < 1e-5);
        // the inputs are untouched when --out is given; the merged state routes for both kinds of task
        assert_eq!(load_state(&pa).unwrap().agents.len(), 2);
        let r = call(&["route", "--state", o, "--features", "y"]).unwrap();
        assert_eq!(r["agent"], "bonly");
        assert!(call(&["route", "--state", o, "--features", "x,unknown"]).unwrap()["ok"].as_bool().unwrap());
        // default --out is --into
        call(&["merge-state", "--into", a, "--from", b]).unwrap();
        assert_eq!(load_state(&pa).unwrap().agents.len(), 3);
        for p in [&pa, &pb, &po] {
            let _ = fs::remove_file(p);
        }
    }

    #[test]
    fn merge_state_rejects_bad_input() {
        let (pa, pb) = (tmp("merge-bad-a"), tmp("merge-bad-b"));
        let (a, b) = (pa.to_str().unwrap(), pb.to_str().unwrap());
        call(&["feedback", "--state", a, "--features", "x", "--agent", "a", "--reward", "1"]).unwrap();
        assert!(call(&["merge-state", "--into", a, "--from", b]).is_err(), "missing --from file");
        assert!(call(&["merge-state", "--into", a, "--from", a]).is_err(), "same file");
        assert!(call(&["merge-state", "--into", a]).is_err(), "missing --from");
        assert!(call(&["merge-state", "--into", a, "--from", b, "--bogus", "1"]).is_err());
        let _ = fs::remove_file(&pa);
    }

    #[test]
    fn save_is_atomic_and_leaves_no_temp_files() {
        let dir = std::env::temp_dir().join(format!("nm-atomic-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let p = dir.join("s.json");
        let ps = p.to_str().unwrap().to_string();
        call(&["feedback", "--state", &ps, "--features", "x", "--agent", "a", "--reward", "1"]).unwrap();
        // a reader racing many writers must always see a complete, parseable file (temp + rename)
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let reader = {
            let (p, stop) = (p.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut reads = 0;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    if let Ok(s) = fs::read_to_string(&p) {
                        serde_json::from_str::<State>(&s).expect("torn state file observed");
                        reads += 1;
                    }
                }
                reads
            })
        };
        let ws: Vec<_> = (0..8)
            .map(|i| {
                let ps = ps.clone();
                std::thread::spawn(move || {
                    for _ in 0..5 {
                        call(&["feedback", "--state", &ps, "--features", "x", "--agent", &format!("w{i}"), "--reward", "1"]).unwrap();
                    }
                })
            })
            .collect();
        for w in ws {
            w.join().unwrap();
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(reader.join().unwrap() > 0);
        assert_eq!(load_state(&p).unwrap().feedbacks, 41, "no lost updates");
        let leftovers: Vec<_> = fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name()).filter(|n| n != "s.json").collect();
        assert!(leftovers.is_empty(), "temp/lock files left behind: {leftovers:?}");
        let _ = fs::remove_dir_all(&dir);
    }
}
