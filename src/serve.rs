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
//! I/O: each reply is serialised once into a reused buffer and sent with one `write`. When the
//! client has pipelined further complete request lines, their replies are sent together (always
//! before the server blocks for input, at EOF and on `shutdown`); bytes and order are exactly those
//! of one-write-per-reply. Request parsing and option conversion run outside the global lock, and
//! an over-long line (> 1 MiB) is discarded as it streams in rather than buffered whole. Opt-in,
//! default off: env `NEURAL_MESH_SERVE_STAT_MS=N` checks the state file at most every N ms instead
//! of before every request (the stat costs ~20 us on Windows; external CLI writes are then absorbed
//! up to N ms later, never lost: every flush still reloads and replays).
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
use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Read, Write};
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
    /// Opt-in (`NEURAL_MESH_SERVE_STAT_MS`): look at the state file at most this often instead of
    /// before every request. `None` (default) = every request, as documented above.
    stat_gap: Option<Duration>,
    last_stat: Instant,
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
        let stat_gap = std::env::var("NEURAL_MESH_SERVE_STAT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&n| n > 0).map(Duration::from_millis);
        Ok(Server { path, st, sig, journal: Vec::new(), dirty_since: None, backoff_until: None, flush_every, flush_ms, stat_gap, last_stat: Instant::now() })
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
        if let Some(gap) = self.stat_gap {
            if self.last_stat.elapsed() < gap {
                return;
            }
            self.last_stat = Instant::now();
        }
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

    /// Run one planned command against the state (called with the server lock held).
    fn execute(&mut self, name: &'static str, map: BTreeMap<String, String>) -> Result<(Value, bool), CliError> {
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

    /// Run a planned request under the (already held) lock; returns the response and whether to stop.
    fn run_plan(&mut self, plan: Plan) -> (Value, bool) {
        let res = catch_unwind(AssertUnwindSafe(|| match plan {
            Plan::Control(c) => self.control(c),
            Plan::Run(name, map) => self.execute(name, map),
            Plan::Fail(e) => Err(e),
            Plan::Reply(v) => Ok((v, false)),
        }));
        match res {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => (err_json(e.msg()), false),
            Err(_) => (err_json("internal error"), false),
        }
    }

}

/// What a request line turned into, before the server state is touched.
enum Plan {
    /// Finished without touching the server (bad JSON / not an object / no `cmd`): no `id` echo.
    Reply(Value),
    /// `flush` or `shutdown`.
    Control(&'static str),
    /// A state command with its validated string options.
    Run(&'static str, BTreeMap<String, String>),
    /// Dispatch-level error (unknown command / bad field); `id` is echoed.
    Fail(CliError),
}

struct Parsed {
    id: Option<Value>,
    plan: Plan,
}

fn with_id(mut out: Value, id: Option<Value>) -> Value {
    if let (Some(id), Some(o)) = (id, out.as_object_mut()) {
        o.insert("id".into(), id);
    }
    out
}

/// Pure half of request handling: JSON parse, command lookup, field validation, option
/// conversion. Needs no server state, so the connection threads run it outside the lock.
fn parse_request(line: &str) -> Parsed {
    let early = |v: Value| Parsed { id: None, plan: Plan::Reply(v) };
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return early(err_json(format!("malformed JSON: {e}"))),
    };
    let Some(obj) = req.as_object() else { return early(err_json("request must be a JSON object")) };
    let Some(cmd) = obj.get("cmd").and_then(Value::as_str) else {
        return early(err_json("missing string field 'cmd'"));
    };
    let id = obj.get("id").cloned();
    let (name, base): (&'static str, &[&str]) = match cmd {
        "route" => ("route", &["features", "candidates", "team", "type", "seed", "prior"]),
        "feedback" => ("feedback", &["features", "agent", "team", "reward", "type"]),
        "stats" => ("stats", &[]),
        "config" => ("config", &["team"]),
        "flush" => return Parsed { id, plan: Plan::Control("flush") },
        "shutdown" => return Parsed { id, plan: Plan::Control("shutdown") },
        other => return Parsed { id, plan: Plan::Fail(usage(format!("unknown command '{other}'"))) },
    };
    let plan = match to_opts(obj, base) {
        Ok(map) => Plan::Run(name, map),
        Err(e) => Plan::Fail(e),
    };
    Parsed { id, plan }
}

fn err_json(m: impl Into<String>) -> Value {
    json!({"ok": false, "error": m.into()})
}

/// Convert request fields to the CLI's string options.
fn to_opts(req: &Map<String, Value>, base: &[&str]) -> Result<BTreeMap<String, String>, CliError> {
    let mut m = BTreeMap::new();
    for (k, v) in req {
        if k == "cmd" || k == "id" {
            continue;
        }
        // JSON callers may write prior_k; the CLI flag is --prior-k.
        let key = k.replace('_', "-");
        if !(base.contains(&key.as_str()) || CONFIG_KEYS.contains(&key.as_str())) {
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
            let mut joined = String::new();
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    joined.push(',');
                }
                match x {
                    Value::String(s) => joined.push_str(s),
                    Value::Number(n) => {
                        let _ = write!(joined, "{n}");
                    }
                    _ => return Err(()),
                }
            }
            Ok(Some(joined))
        }
        _ => Err(()),
    }
}

fn lock(s: &Mutex<Server>) -> MutexGuard<'_, Server> {
    s.lock().unwrap_or_else(|e| e.into_inner())
}

const IN_BUF: usize = 16 * 1024;
/// Replies are held back while the client has more complete lines queued, but never beyond this.
const OUT_FLUSH_AT: usize = 64 * 1024;

/// Read one request line into `buf` (cleared first), keeping at most `MAX_LINE + 1` bytes of it.
/// `Ok(None)` = EOF. `Ok(Some(true))` = the line (newline included) is longer than `MAX_LINE`: the
/// rest of it is consumed and discarded, so one hostile line cannot make us buffer unbounded memory.
fn read_request<R: BufRead>(r: &mut R, buf: &mut Vec<u8>) -> std::io::Result<Option<bool>> {
    buf.clear();
    let n = (&mut *r).take(MAX_LINE as u64 + 1).read_until(b'\n', buf)?;
    if n == 0 {
        return Ok(None);
    }
    if n <= MAX_LINE {
        return Ok(Some(false));
    }
    if buf.last() != Some(&b'\n') {
        loop {
            let (done, used) = {
                let chunk = r.fill_buf()?;
                if chunk.is_empty() {
                    break;
                }
                match chunk.iter().position(|&c| c == b'\n') {
                    Some(i) => (true, i + 1),
                    None => (false, chunk.len()),
                }
            };
            r.consume(used);
            if done {
                break;
            }
        }
    }
    buf.clear();
    buf.shrink_to(IN_BUF);
    Ok(Some(true))
}

fn flush_out<W: Write>(w: &mut W, out: &mut Vec<u8>) -> std::io::Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    let r = w.write_all(out).and_then(|()| w.flush());
    out.clear();
    r
}

/// Serve one byte stream until EOF or `shutdown`. Returns true on shutdown.
///
/// Each reply is serialised once into a reused buffer and sent with a single `write`. While the
/// client has more complete request lines already queued in our read buffer, replies are held back
/// and sent together (they are always sent before we block waiting for input, at EOF, and on
/// `shutdown`), so a request/response client still sees one write per reply and a pipelining
/// client gets one write per burst. Reply bytes and order are exactly those of the unbatched loop.
fn run_stream<R: Read, W: Write>(input: R, mut w: W, srv: &Mutex<Server>) -> bool {
    let mut r = BufReader::with_capacity(IN_BUF, input);
    let mut line = Vec::new();
    let mut out: Vec<u8> = Vec::with_capacity(1024);
    loop {
        let too_long = match read_request(&mut r, &mut line) {
            Ok(Some(t)) => t,
            Ok(None) | Err(_) => {
                let _ = flush_out(&mut w, &mut out);
                return false;
            }
        };
        let (resp, quit) = respond_inner(&line, too_long, srv);
        if let Some(resp) = resp {
            let _ = serde_json::to_writer(&mut out, &resp);
            out.push(b'\n');
        }
        let more_queued = r.buffer().contains(&b'\n');
        if (quit || out.len() >= OUT_FLUSH_AT || !more_queued) && flush_out(&mut w, &mut out).is_err() {
            return false;
        }
        if quit {
            return true;
        }
    }
}

#[cfg(test)]
fn respond(raw: &[u8], srv: &Mutex<Server>) -> (Option<Value>, bool) {
    respond_inner(raw, raw.len() > MAX_LINE, srv)
}

fn respond_inner(raw: &[u8], too_long: bool, srv: &Mutex<Server>) -> (Option<Value>, bool) {
    if too_long {
        return (Some(err_json("request line too long")), false);
    }
    let Ok(text) = std::str::from_utf8(raw) else { return (Some(err_json("request is not valid UTF-8")), false) };
    let text = text.trim();
    if text.is_empty() {
        return (None, false);
    }
    let Parsed { id, plan } = parse_request(text);
    let (out, quit) = match plan {
        Plan::Reply(v) => (v, false),
        Plan::Fail(e) => (err_json(e.msg()), false),
        p => lock(srv).run_plan(p),
    };
    (Some(with_id(out, id)), quit)
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
    let shutdown = run_stream(stream, writer, srv);
    if shutdown {
        std::process::exit(finish(srv));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::io::Cursor;
    use std::rc::Rc;

    // ---- per-thread counting allocator (only counts while ON on the calling thread) ----
    thread_local! {
        static ON: Cell<bool> = const { Cell::new(false) };
        static ALLOCS: Cell<u64> = const { Cell::new(0) };
        static BYTES: Cell<u64> = const { Cell::new(0) };
        static LIVE: Cell<i64> = const { Cell::new(0) };
        static PEAK: Cell<i64> = const { Cell::new(0) };
    }
    struct Counting;
    fn bump(sz: i64) {
        ON.with(|on| {
            if on.get() {
                LIVE.with(|l| {
                    l.set(l.get() + sz);
                    PEAK.with(|p| p.set(p.get().max(l.get())));
                });
            }
        });
    }
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ON.with(|on| {
                if on.get() {
                    ALLOCS.with(|a| a.set(a.get() + 1));
                    BYTES.with(|b| b.set(b.get() + l.size() as u64));
                }
            });
            bump(l.size() as i64);
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            bump(-(l.size() as i64));
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
            ON.with(|on| {
                if on.get() {
                    ALLOCS.with(|a| a.set(a.get() + 1));
                    BYTES.with(|b| b.set(b.get() + new as u64));
                }
            });
            bump(new as i64 - l.size() as i64);
            unsafe { System.realloc(p, l, new) }
        }
    }
    #[global_allocator]
    static A: Counting = Counting;

    /// (result, allocation calls, bytes requested, peak live bytes above the starting point) while running `f`.
    fn counted<T>(f: impl FnOnce() -> T) -> (T, u64, u64, i64) {
        ALLOCS.with(|a| a.set(0));
        BYTES.with(|b| b.set(0));
        LIVE.with(|l| l.set(0));
        PEAK.with(|p| p.set(0));
        ON.with(|o| o.set(true));
        let r = f();
        ON.with(|o| o.set(false));
        (r, ALLOCS.with(Cell::get), BYTES.with(Cell::get), PEAK.with(Cell::get))
    }

    // ---- the pre-optimisation serving path, kept verbatim as the reference (HEAD 52f936c) ----
    // Only the tail of `dispatch` (refresh + run the command + journal) is shared with the new code
    // (`Server::execute`); everything that parses, validates, buffers and writes is the old text.
    fn scalar_or_list_old(v: &Value) -> Result<Option<String>, ()> {
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
    fn to_opts_old(req: &Map<String, Value>, allowed: &[&str]) -> Result<BTreeMap<String, String>, CliError> {
        let mut m = BTreeMap::new();
        for (k, v) in req {
            if k == "cmd" || k == "id" {
                continue;
            }
            let key = k.replace('_', "-");
            if !allowed.contains(&key.as_str()) {
                return Err(usage(format!("unknown field '{k}'")));
            }
            if let Some(s) = scalar_or_list_old(v).map_err(|()| usage(format!("field '{k}' must be a string, number, or array of strings")))? {
                m.insert(key, s);
            }
        }
        Ok(m)
    }
    fn dispatch_old(srv: &mut Server, cmd: &str, req: &Map<String, Value>) -> Result<(Value, bool), CliError> {
        let (name, base): (&'static str, &[&str]) = match cmd {
            "route" => ("route", &["features", "candidates", "team", "type", "seed", "prior"]),
            "feedback" => ("feedback", &["features", "agent", "team", "reward", "type"]),
            "stats" => ("stats", &[]),
            "config" => ("config", &["team"]),
            "flush" | "shutdown" => return srv.control(cmd),
            other => return Err(usage(format!("unknown command '{other}'"))),
        };
        let allowed: Vec<&str> = base.iter().chain(CONFIG_KEYS.iter()).copied().collect();
        let map = to_opts_old(req, &allowed)?;
        srv.execute(name, map)
    }
    fn handle_old(srv: &mut Server, line: &str) -> (Value, bool) {
        let req: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => return (err_json(format!("malformed JSON: {e}")), false),
        };
        let Some(obj) = req.as_object() else { return (err_json("request must be a JSON object"), false) };
        let Some(cmd) = obj.get("cmd").and_then(Value::as_str) else {
            return (err_json("missing string field 'cmd'"), false);
        };
        let res = catch_unwind(AssertUnwindSafe(|| dispatch_old(srv, cmd, obj)));
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
    fn respond_old(raw: &[u8], srv: &Mutex<Server>) -> (Option<Value>, bool) {
        if raw.len() > MAX_LINE {
            return (Some(err_json("request line too long")), false);
        }
        let Ok(text) = std::str::from_utf8(raw) else { return (Some(err_json("request is not valid UTF-8")), false) };
        let text = text.trim();
        if text.is_empty() {
            return (None, false);
        }
        let (v, quit) = handle_old(&mut lock(srv), text);
        (Some(v), quit)
    }
    fn run_stream_old<R: BufRead, W: Write>(mut r: R, mut w: W, srv: &Mutex<Server>) -> bool {
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match r.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => return false,
                Ok(_) => {}
            }
            let (resp, quit) = respond_old(&buf, srv);
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

    // ---- helpers ----
    fn new_server(tag: &str) -> Mutex<Server> {
        let p = std::env::temp_dir().join(format!("nm-serve-test-{}-{tag}.json", std::process::id()));
        let _ = std::fs::remove_file(&p);
        Mutex::new(Server::open(p, 1_000_000, 600_000).unwrap())
    }

    fn script() -> Vec<u8> {
        let mut s = String::new();
        s += "{\"cmd\":\"route\",\"features\":[\"warm\"],\"candidates\":[\"a0\",\"a1\",\"a2\"],\"id\":\"w\"}\n";
        for i in 0..40 {
            s += &format!("{{\"cmd\":\"route\",\"features\":[\"kw{}\",\"kw{}\"],\"id\":{i}}}\n", i % 5, i % 3);
            s += &format!("{{\"cmd\":\"feedback\",\"features\":[\"kw{}\"],\"agent\":\"a{}\",\"reward\":0.{}}}\n", i % 5, i % 3, i % 10);
        }
        for bad in [
            "",
            "   ",
            "not json",
            "[1,2]",
            "{\"id\":3}",
            "{\"cmd\":7}",
            "{\"cmd\":\"nope\",\"id\":7}",
            "{\"cmd\":\"route\",\"features\":[\"x\"],\"bogus\":1,\"id\":\"b\"}",
            "{\"cmd\":\"route\",\"features\":{\"a\":1}}",
            "{\"cmd\":\"feedback\",\"features\":[\"x\"],\"agent\":\"a0\",\"reward\":9}",
            "{\"cmd\":\"stats\",\"id\":null}",
            "{\"cmd\":\"config\",\"eta\":0.02,\"id\":1}",
            "{\"cmd\":\"stats\",\"prior_k\":3}",
            "{\"cmd\":\"route\",\"features\":[1,2.5,\"c\"],\"prior_k\":2}",
            "{\"cmd\":\"flush\",\"id\":[1]}",
        ] {
            s += bad;
            s.push('\n');
        }
        let mut v = s.into_bytes();
        v.extend_from_slice(b"\xff\xfe\n");
        v.extend_from_slice(b"\r\n");
        v.extend_from_slice(b"{\"cmd\":\"stats\",\"id\":1}\r\n");
        // exactly at / one over the limit, valid JSON
        for total in [MAX_LINE, MAX_LINE + 1] {
            let head = "{\"cmd\":\"stats\",\"id\":\"";
            let pad = total - 1 - head.len() - 2;
            let mut l = String::from(head);
            l.push_str(&"p".repeat(pad));
            l.push_str("\"}\n");
            assert_eq!(l.len(), total);
            v.extend_from_slice(l.as_bytes());
        }
        v.extend_from_slice(&vec![b'z'; 3 * MAX_LINE]);
        v.push(b'\n');
        v.extend_from_slice(b"{\"cmd\":\"stats\",\"id\":\"after\"}\n");
        v.extend_from_slice(b"{\"cmd\":\"shutdown\",\"id\":\"bye\"}\n{\"cmd\":\"stats\",\"id\":\"ignored\"}\n");
        v
    }

    /// Reader that hands out at most `step` bytes per call (forces split lines).
    struct Trickle<'a> {
        data: &'a [u8],
        step: usize,
    }
    impl Read for Trickle<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.step.min(buf.len()).min(self.data.len());
            buf[..n].copy_from_slice(&self.data[..n]);
            self.data = &self.data[n..];
            Ok(n)
        }
    }

    #[test]
    fn stream_output_is_byte_identical_to_the_reference_loop() {
        let input = script();
        let old = new_server("ref");
        let mut want = Vec::new();
        let quit_old = run_stream_old(Cursor::new(&input), &mut want, &old);
        assert!(quit_old, "script ends with shutdown");
        assert!(String::from_utf8_lossy(&want).contains("request line too long"));
        assert!(String::from_utf8_lossy(&want).contains("not valid UTF-8"));
        for (tag, step) in [("whole", usize::MAX), ("trickle1", 1), ("trickle7", 7), ("trickle4093", 4093)] {
            let new = new_server(tag);
            let mut got = Vec::new();
            let quit = run_stream(Trickle { data: &input, step }, &mut got, &new);
            assert!(quit, "{tag}: shutdown seen");
            assert_eq!(got.len(), want.len(), "{tag}: length");
            assert!(got == want, "{tag}: bytes differ");
        }
    }

    #[test]
    fn eof_flushes_pending_replies_and_a_partial_last_line_is_served() {
        let input = b"{\"cmd\":\"route\",\"features\":[\"a\"],\"candidates\":[\"x\",\"y\"]}\n{\"cmd\":\"stats\",\"id\":9}".to_vec();
        let (a, b) = (new_server("eof-a"), new_server("eof-b"));
        let (mut want, mut got) = (Vec::new(), Vec::new());
        assert!(!run_stream_old(Cursor::new(&input), &mut want, &a));
        assert!(!run_stream(Cursor::new(&input), &mut got, &b));
        assert_eq!(want.iter().filter(|&&c| c == b'\n').count(), 2);
        assert_eq!(got, want);
    }

    #[test]
    fn error_forms_and_id_echo() {
        let srv = new_server("errs");
        let r = |l: &str| respond(l.as_bytes(), &srv).0.map(|v| v.to_string());
        assert_eq!(r(""), None);
        assert_eq!(r("  \r\n"), None);
        assert_eq!(r("{\"cmd\":\"nope\",\"id\":7}").unwrap(), "{\"error\":\"unknown command 'nope'\",\"id\":7,\"ok\":false}");
        // dispatch-level errors echo the id; shape errors (no usable cmd) do not
        assert_eq!(r("{\"id\":7}").unwrap(), "{\"error\":\"missing string field 'cmd'\",\"ok\":false}");
        assert!(r("{\"cmd\":\"stats\",\"zz\":1,\"id\":\"i\"}").unwrap().contains("\"id\":\"i\""));
        assert!(r("{\"cmd\":\"stats\",\"zz\":1}").unwrap().contains("unknown field 'zz'"));
        assert!(r("[1]").unwrap().contains("request must be a JSON object"));
        assert!(r("{").unwrap().contains("malformed JSON"));
        assert_eq!(respond(&[0xff, b'\n'], &srv).0.unwrap().to_string(), "{\"error\":\"request is not valid UTF-8\",\"ok\":false}");
        let big = vec![b'x'; MAX_LINE + 1];
        assert_eq!(respond(&big, &srv).0.unwrap().to_string(), "{\"error\":\"request line too long\",\"ok\":false}");
    }

    #[test]
    fn external_change_is_seen_before_the_next_request_unless_the_opt_in_gap_is_set() {
        for gap in [None, Some(Duration::from_secs(3600))] {
            let srv = new_server(&format!("gap-{}", gap.is_some()));
            let path = {
                let mut g = lock(&srv);
                g.stat_gap = gap;
                g.path.clone()
            };
            {
                let _l = StateLock::acquire(&path).unwrap();
                let mut other = load_state(&path).unwrap();
                other.register("ext-agent");
                save_state(&path, &other).unwrap();
            }
            let stats = respond(b"{\"cmd\":\"stats\"}\n", &srv).0.unwrap().to_string();
            assert_eq!(stats.contains("ext-agent"), gap.is_none(), "gap={gap:?}");
        }
    }

    /// Write sink that counts `write` calls and newlines written.
    struct CountW(Rc<Cell<usize>>, Rc<Cell<usize>>);
    impl Write for CountW {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.set(self.0.get() + 1);
            self.1.set(self.1.get() + b.iter().filter(|&&c| c == b'\n').count());
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    /// A request/response client: it only "sends" request k+1 when asked to read again, and notes
    /// whether every reply owed so far had already been written at that moment.
    struct Lockstep {
        lines: Vec<Vec<u8>>,
        next: usize,
        cur: Vec<u8>,
        replies: Rc<Cell<usize>>,
        stale: Rc<Cell<usize>>,
    }
    impl Read for Lockstep {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.cur.is_empty() {
                if self.replies.get() < self.next {
                    self.stale.set(self.stale.get() + 1);
                }
                if self.next == self.lines.len() {
                    return Ok(0);
                }
                self.cur = self.lines[self.next].clone();
                self.next += 1;
            }
            let n = buf.len().min(self.cur.len());
            buf[..n].copy_from_slice(&self.cur[..n]);
            self.cur.drain(..n);
            Ok(n)
        }
    }

    #[test]
    fn replies_are_sent_before_blocking_and_pipelined_bursts_are_batched() {
        let lines: Vec<Vec<u8>> = (0..30).map(|i| format!("{{\"cmd\":\"route\",\"features\":[\"k{}\"],\"candidates\":[\"a\",\"b\"]}}\n", i % 4).into_bytes()).collect();
        let (writes, nl, stale) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
        let rd = Lockstep { lines: lines.clone(), next: 0, cur: Vec::new(), replies: nl.clone(), stale: stale.clone() };
        run_stream(rd, CountW(writes.clone(), nl.clone()), &new_server("lock"));
        assert_eq!(nl.get(), 30);
        assert_eq!(stale.get(), 0, "a reply was still buffered when the server blocked for input");
        assert_eq!(writes.get(), 30, "one write per reply in lockstep");
        // pipelined: the same 30 requests arriving in one chunk -> one write
        let all: Vec<u8> = lines.concat();
        let (w2, n2) = (Rc::new(Cell::new(0)), Rc::new(Cell::new(0)));
        run_stream(Cursor::new(all), CountW(w2.clone(), n2.clone()), &new_server("burst"));
        assert_eq!((n2.get(), w2.get()), (30, 1), "burst of 30 requests answered with one write");
    }

    #[test]
    fn oversized_lines_are_discarded_not_buffered() {
        // 200 MB of one never-ending line, then a normal request
        struct Endless(usize, bool);
        impl Read for Endless {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.0 > 0 {
                    let n = buf.len().min(self.0);
                    buf[..n].fill(b'q');
                    self.0 -= n;
                    return Ok(n);
                }
                if !self.1 {
                    self.1 = true;
                    let tail = b"\n{\"cmd\":\"stats\",\"id\":\"ok\"}\n";
                    buf[..tail.len()].copy_from_slice(tail);
                    return Ok(tail.len());
                }
                Ok(0)
            }
        }
        let srv = new_server("huge");
        let mut out = Vec::new();
        let (_, _, _, peak) = counted(|| run_stream(Endless(200 << 20, false), &mut out, &srv));
        let s = String::from_utf8(out).unwrap();
        let mut it = s.lines();
        assert_eq!(it.next().unwrap(), "{\"error\":\"request line too long\",\"ok\":false}");
        assert!(it.next().unwrap().contains("\"id\":\"ok\""));
        assert!(peak < (6 << 20), "peak live heap {peak} B while discarding a 200 MB line");
    }

    /// `cargo test --release --bin neural-mesh -- --ignored --nocapture alloc_report`
    #[test]
    #[ignore]
    fn alloc_report() {
        let mut s = String::from("{\"cmd\":\"route\",\"features\":[\"warm\"],\"candidates\":[\"a0\",\"a1\",\"a2\",\"a3\",\"a4\",\"a5\"]}\n");
        let n = 2000;
        for i in 0..n {
            s += &format!("{{\"cmd\":\"route\",\"features\":[\"kw{}\",\"kw{}\"]}}\n", i % 24, (i * 7 + 3) % 24);
            s += &format!("{{\"cmd\":\"feedback\",\"features\":[\"kw{}\",\"kw{}\"],\"agent\":\"a{}\",\"reward\":0.5}}\n", i % 24, (i * 7 + 3) % 24, i % 6);
        }
        let input = s.into_bytes();
        let reqs = (2 * n + 1) as f64;
        for round in 0..2 {
            let (a, b) = (new_server(&format!("al-old{round}")), new_server(&format!("al-new{round}")));
            let mut sink = Vec::with_capacity(8 << 20);
            let (_, ao, bo, po) = counted(|| run_stream_old(Cursor::new(&input), &mut sink, &a));
            let mut sink2 = Vec::with_capacity(8 << 20);
            let (_, an, bn, pn) = counted(|| run_stream(Cursor::new(&input), &mut sink2, &b));
            assert_eq!(sink, sink2);
            println!(
                "round {round}: allocator calls per request  old {:.1}  new {:.1}   bytes requested per request  old {:.0}  new {:.0}   peak live (excl. state)  old {} B  new {} B",
                ao as f64 / reqs,
                an as f64 / reqs,
                bo as f64 / reqs,
                bn as f64 / reqs,
                po,
                pn
            );
        }
    }
}
