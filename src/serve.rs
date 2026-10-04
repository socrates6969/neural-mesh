//! `neural-mesh serve`: long-lived JSON-lines front end over the same
//! commands as the one-shot CLI.
//!
//! ```text
//! neural-mesh serve --state <file> (--stdio | --port N) [--flush-every N] [--flush-ms T]
//! ```
//!
//! Protocol: one JSON object per line in, one JSON object per line out.
//! `{"cmd":"route"|"feedback"|"stats"|"config", ...}` takes the same fields as the CLI
//! flags (strings, numbers, or arrays of strings for comma lists) and answers
//! with exactly the object the CLI would print. Extra commands: `flush`
//! (persist now) and `shutdown` (persist, reply, exit). An optional `"id"` is
//! echoed back. Blank lines are ignored. Any bad line gets
//! `{"ok":false,"error":"..."}` and the server keeps running.
//!
//! Durability (write-behind): state lives in memory and is written with the
//! CLI's lock file + atomic rename when `--flush-every` mutating requests
//! (route/feedback, default 64) have accumulated, or `--flush-ms` (default
//! 1000) have passed since the first unflushed one, and on EOF / `shutdown`.
//! A hard crash (kill -9, power loss) therefore loses at most
//! `flush_every - 1` requests or ~1.25 * `flush_ms` of mutations, whichever
//! comes first. A clean EOF or `shutdown` loses nothing.
//!
//! Coordination with one-shot CLI calls: the server takes the state lock only
//! while flushing. Before each request it compares the state file's
//! (mtime, len) with the value recorded after its last read/write. If another
//! process changed the file, the server takes the lock, reloads the file and
//! replays its journal of not-yet-flushed route/feedback requests on top of
//! it, then writes the result. Net effect: every update from both sides is
//! kept, ordered as "CLI call first, pending server requests after". Replies
//! already sent for the pending requests were computed on the older base.

use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::net::{TcpListener, TcpStream};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Map, Value, json};

use crate::{CONFIG_KEYS, CliError, Opts, State, StateLock, cmd_config, cmd_feedback, cmd_route, cmd_stats, has_config_opts, load_state, save_state, usage};

const SERVE_USAGE: &str = "usage: neural-mesh serve --state <file> (--stdio | --port N) [--flush-every N] [--flush-ms T]";
const DEFAULT_FLUSH_EVERY: usize = 64;
const DEFAULT_FLUSH_MS: u64 = 1000;
const MAX_LINE: usize = 1 << 20;

type Sig = Option<(SystemTime, u64)>;
type Journal = Vec<(&'static str, BTreeMap<String, String>)>;

fn file_sig(path: &std::path::Path) -> Sig {
    let m = std::fs::metadata(path).ok()?;
    Some((m.modified().ok()?, m.len()))
}

struct Server {
    path: PathBuf,
    st: State,
    sig: Sig,
    journal: Journal,
    dirty_since: Option<Instant>,
    backoff_until: Option<Instant>,
    flush_every: usize,
    flush_ms: u64,
}

fn replay(st: &mut State, journal: &Journal) {
    for (cmd, m) in journal {
        let o = Opts(m.clone());
        let _ = match *cmd {
            "route" => cmd_route(st, &o),
            "feedback" => cmd_feedback(st, &o),
            _ => cmd_config(st, &o),
        };
    }
}

impl Server {
    fn open(path: PathBuf, flush_every: usize, flush_ms: u64) -> Result<Server, CliError> {
        let _lock = StateLock::acquire(&path)?;
        let existed = path.exists();
        let st = load_state(&path)?;
        if !existed {
            save_state(&path, &st)?;
        }
        let sig = file_sig(&path);
        Ok(Server { path, st, sig, journal: Vec::new(), dirty_since: None, backoff_until: None, flush_every, flush_ms })
    }

    /// Lock, absorb external changes (reload + replay journal), write if we
    /// have pending updates, and record the new file signature.
    fn commit(&mut self) -> Result<(), CliError> {
        let _lock = StateLock::acquire(&self.path)?;
        if file_sig(&self.path) != self.sig {
            let mut base = load_state(&self.path)?;
            replay(&mut base, &self.journal);
            self.st = base;
        }
        if !self.journal.is_empty() {
            save_state(&self.path, &self.st)?;
            self.journal.clear();
            self.dirty_since = None;
        }
        self.sig = file_sig(&self.path);
        Ok(())
    }

    fn backed_off(&self) -> bool {
        self.backoff_until.is_some_and(|t| Instant::now() < t)
    }

    /// Commit, logging (not failing) on error and backing off for one interval.
    fn try_commit(&mut self) {
        if let Err(e) = self.commit() {
            eprintln!("neural-mesh serve: flush failed: {}", e.msg());
            self.backoff_until = Some(Instant::now() + Duration::from_millis(self.flush_ms));
        }
    }

    fn refresh(&mut self) {
        if !self.backed_off() && file_sig(&self.path) != self.sig {
            self.try_commit();
        }
    }

    /// Timer-driven flush (called by the ticker thread).
    fn tick(&mut self) {
        let due = self.dirty_since.is_some_and(|t| t.elapsed().as_millis() as u64 >= self.flush_ms);
        if due && !self.backed_off() {
            self.try_commit();
        }
    }

    fn record(&mut self, cmd: &'static str, opts: BTreeMap<String, String>) {
        self.journal.push((cmd, opts));
        self.dirty_since.get_or_insert_with(Instant::now);
        if self.journal.len() >= self.flush_every && !self.backed_off() {
            self.try_commit();
        }
    }

    fn dispatch(&mut self, cmd: &str, req: &Map<String, Value>) -> Result<(Value, bool), CliError> {
        let (name, base): (&'static str, &[&str]) = match cmd {
            "route" => ("route", &["features", "candidates", "team", "type", "seed", "prior"]),
            "feedback" => ("feedback", &["features", "agent", "team", "reward", "type"]),
            "stats" => ("stats", &[]),
            "config" => ("config", &["team"]),
            "flush" | "shutdown" => return self.control(cmd),
            other => return Err(usage(format!("unknown command '{other}'"))),
        };
        let allowed: Vec<&str> = base.iter().chain(CONFIG_KEYS.iter()).copied().collect();
        let map = to_opts(req, &allowed)?;
        self.refresh();
        let o = Opts(map);
        let out = match name {
            "route" => cmd_route(&mut self.st, &o)?,
            "feedback" => cmd_feedback(&mut self.st, &o)?,
            "config" => {
                let v = cmd_config(&mut self.st, &o)?;
                if has_config_opts(&o, true) {
                    self.record("config", o.0);
                }
                return Ok((v, false));
            }
            _ => return self.stats_with_config(o),
        };
        self.record(name, o.0);
        Ok((out, false))
    }

    /// `stats`, applying (and journaling as `config`) any config fields it carries.
    fn stats_with_config(&mut self, o: Opts) -> Result<(Value, bool), CliError> {
        if has_config_opts(&o, false) {
            cmd_config(&mut self.st, &o)?;
            self.record("config", o.0);
        }
        Ok((cmd_stats(&self.st), false))
    }

    fn control(&mut self, cmd: &str) -> Result<(Value, bool), CliError> {
        self.commit()?;
        let quit = cmd == "shutdown";
        Ok((json!({"ok": true, cmd: true, "pending": self.journal.len()}), quit))
    }

    /// Handle one request line; returns the response and whether to stop.
    fn handle(&mut self, line: &str) -> (Value, bool) {
        let req: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => return (err_json(format!("malformed JSON: {e}")), false),
        };
        let Some(obj) = req.as_object() else { return (err_json("request must be a JSON object"), false) };
        let Some(cmd) = obj.get("cmd").and_then(Value::as_str) else {
            return (err_json("missing string field 'cmd'"), false);
        };
        let res = catch_unwind(AssertUnwindSafe(|| self.dispatch(cmd, obj)));
        let (mut out, quit) = match res {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => (err_json(e.msg()), false),
            Err(_) => (err_json("internal error"), false),
        };
        if let (Some(id), Some(o)) = (obj.get("id"), out.as_object_mut()) {
            o.insert("id".into(), id.clone());
        }
        (out, quit)
    }
}

fn err_json(m: impl Into<String>) -> Value {
    json!({"ok": false, "error": m.into()})
}

/// Convert request fields to the CLI's string options.
fn to_opts(req: &Map<String, Value>, allowed: &[&str]) -> Result<BTreeMap<String, String>, CliError> {
    let mut m = BTreeMap::new();
    for (k, v) in req {
        if k == "cmd" || k == "id" {
            continue;
        }
        // JSON callers may write prior_k; the CLI flag is --prior-k.
        let key = k.replace('_', "-");
        if !allowed.contains(&key.as_str()) {
            return Err(usage(format!("unknown field '{k}'")));
        }
        if let Some(s) = scalar_or_list(v).map_err(|()| usage(format!("field '{k}' must be a string, number, or array of strings")))? {
            m.insert(key, s);
        }
    }
    Ok(m)
}

fn scalar_or_list(v: &Value) -> Result<Option<String>, ()> {
    match v {
        Value::Null => Ok(None),
        Value::String(s) => Ok(Some(s.clone())),
        Value::Number(n) => Ok(Some(n.to_string())),
        Value::Array(a) => {
            let parts: Result<Vec<String>, ()> = a
                .iter()
                .map(|x| match x {
                    Value::String(s) => Ok(s.clone()),
                    Value::Number(n) => Ok(n.to_string()),
                    _ => Err(()),
                })
                .collect();
            Ok(Some(parts?.join(",")))
        }
        _ => Err(()),
    }
}

fn lock(s: &Mutex<Server>) -> MutexGuard<'_, Server> {
    s.lock().unwrap_or_else(|e| e.into_inner())
}

/// Serve one byte stream until EOF or `shutdown`. Returns true on shutdown.
fn run_stream<R: BufRead, W: Write>(mut r: R, mut w: W, srv: &Mutex<Server>) -> bool {
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match r.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => return false,
            Ok(_) => {}
        }
        let (resp, quit) = respond(&buf, srv);
        if let Some(resp) = resp
            && writeln!(w, "{resp}").and_then(|()| w.flush()).is_err()
        {
            return false;
        }
        if quit {
            return true;
        }
    }
}

fn respond(raw: &[u8], srv: &Mutex<Server>) -> (Option<Value>, bool) {
    if raw.len() > MAX_LINE {
        return (Some(err_json("request line too long")), false);
    }
    let Ok(text) = std::str::from_utf8(raw) else { return (Some(err_json("request is not valid UTF-8")), false) };
    let text = text.trim();
    if text.is_empty() {
        return (None, false);
    }
    let (v, quit) = lock(srv).handle(text);
    (Some(v), quit)
}

fn parse_pos<T: std::str::FromStr + PartialOrd + Default>(o: &Opts, k: &str, default: T) -> Result<T, CliError> {
    match o.get(k) {
        None => Ok(default),
        Some(s) => s.trim().parse::<T>().ok().filter(|v| *v > T::default()).ok_or_else(|| usage(format!("--{k} must be an integer >= 1"))),
    }
}

fn print_err(e: &CliError) {
    println!("{}", json!({"ok": false, "error": e.msg(), "usage": SERVE_USAGE}));
}

/// Entry point for `neural-mesh serve ...`; returns the process exit code.
pub fn main(args: &[String]) -> i32 {
    std::panic::set_hook(Box::new(|_| {}));
    let stdio = args.iter().any(|a| a == "--stdio");
    let rest: Vec<String> = args.iter().filter(|a| *a != "--stdio").cloned().collect();
    let parsed = Opts::parse(&rest, &["state", "port", "flush-every", "flush-ms"]).and_then(|o| {
        let path = PathBuf::from(o.req("state")?);
        if stdio == o.get("port").is_some() {
            return Err(usage("give exactly one of --stdio or --port N"));
        }
        let every = parse_pos(&o, "flush-every", DEFAULT_FLUSH_EVERY)?;
        let ms = parse_pos(&o, "flush-ms", DEFAULT_FLUSH_MS)?;
        let port = match o.get("port") {
            Some(p) => Some(p.trim().parse::<u16>().map_err(|_| usage("--port must be 0..65535"))?),
            None => None,
        };
        Ok((path, every, ms, port))
    });
    let (path, every, ms, port) = match parsed {
        Ok(p) => p,
        Err(e) => {
            print_err(&e);
            return e.code();
        }
    };
    let srv = match Server::open(path, every, ms) {
        Ok(s) => Arc::new(Mutex::new(s)),
        Err(e) => {
            print_err(&e);
            return e.code();
        }
    };
    spawn_ticker(&srv, ms);
    match port {
        None => {
            run_stream(std::io::stdin().lock(), std::io::stdout().lock(), &srv);
        }
        Some(p) => return serve_tcp(&srv, p),
    };
    finish(&srv)
}

fn finish(srv: &Mutex<Server>) -> i32 {
    match lock(srv).commit() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("neural-mesh serve: final flush failed: {}", e.msg());
            e.code()
        }
    }
}

fn spawn_ticker(srv: &Arc<Mutex<Server>>, flush_ms: u64) {
    let srv = Arc::clone(srv);
    let tick = Duration::from_millis((flush_ms / 4).clamp(5, 250));
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(tick);
            lock(&srv).tick();
        }
    });
}

/// Localhost-only TCP mode: one thread per connection, shared state.
fn serve_tcp(srv: &Arc<Mutex<Server>>, port: u16) -> i32 {
    let listener = match TcpListener::bind(("127.0.0.1", port)) {
        Ok(l) => l,
        Err(e) => {
            print_err(&CliError::Io(format!("bind 127.0.0.1:{port}: {e}")));
            return 2;
        }
    };
    if let Ok(addr) = listener.local_addr() {
        println!("{}", json!({"ok": true, "listening": addr.to_string()}));
        let _ = std::io::stdout().flush();
    }
    for stream in listener.incoming().flatten() {
        let srv = Arc::clone(srv);
        std::thread::spawn(move || handle_conn(stream, &srv));
    }
    finish(srv)
}

fn handle_conn(stream: TcpStream, srv: &Arc<Mutex<Server>>) {
    let _ = stream.set_nodelay(true);
    let Ok(writer) = stream.try_clone() else { return };
    let shutdown = run_stream(std::io::BufReader::new(stream), writer, srv);
    if shutdown {
        std::process::exit(finish(srv));
    }
}
