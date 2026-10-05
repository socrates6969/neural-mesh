//! Durable, versioned, SHARED mesh memory (opt-in; nothing in the default paths uses it).
//!
//! Layout (schema_version 1) under a root directory, e.g. `C:\Users\mariu\experiments\mesh-memory\`:
//! ```text
//! MANIFEST.json              {schema_version, kind, created_ms, note}   (written once, never rewritten)
//! router/journal.jsonl       APPEND-ONLY outcome log, one JSON `Outcome` per line (the authority; router state = replay)
//! lessons/lessons.jsonl      APPEND-ONLY `LessonRec` lines
//! knowledge/<ns>/docs.jsonl  APPEND-ONLY `DocRec` lines (id, text, tag)
//! knowledge/<ns>/vecs.f32    APPEND-ONLY little-endian f32 rows, same order as docs.jsonl (dim in knowledge/<ns>/meta.json)
//! .lock                      advisory lock (create_new), stale after 10 s
//! ```
//! Design choices (see RD-LOG rounds 23+):
//! * the journal, not a weight snapshot, is the authority: warm start = replay of outcomes through `Router::update`, so ANY router
//!   (Hebbian, Guarded, ...) can be warm-started, schema changes never invalidate the data, and merging two stores is a set union
//!   by event id (commutative, idempotent) with no weight averaging;
//! * every record carries a `session` and a `world` namespace tag so a consumer can filter stale/foreign evidence;
//! * the HNSW index is NOT serialised: vectors are persisted and the index is rebuilt on load (measured in R-store; the vendored
//!   hnsw_rs dump API borrows from its loader, which does not fit an owned `SharedMemory`);
//! * the store must contain ONLY synthetic routing outcomes, lessons and knowledge: never secrets, keys or customer data.

use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::router::{Router, Task};

pub const SCHEMA_VERSION: u32 = 1;
const LOCK_TIMEOUT_MS: u64 = 3000;
const LOCK_STALE_MS: u64 = 10_000;
#[cfg(test)]
static STALE_OVERRIDE_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn lock_stale_ms() -> u64 {
    #[cfg(test)]
    {
        let o = STALE_OVERRIDE_MS.load(std::sync::atomic::Ordering::Relaxed);
        if o > 0 {
            return o;
        }
    }
    LOCK_STALE_MS
}

fn lock_is_stale(p: &Path) -> bool {
    fs::metadata(p).and_then(|m| m.modified()).ok().and_then(|m| m.elapsed().ok()).map(|d| d > Duration::from_millis(lock_stale_ms())).unwrap_or(false)
}

/// B22: break a stale lock WITHOUT the check-then-remove race. Two waiters that both judged the same lock stale used to both call
/// `remove_file`; the slower one deleted the FRESH lock the faster one had just created, so two processes held the lock. Breaking is now
/// serialised by a `.break` guard (create_new) and the staleness is RE-CHECKED inside it, so only a lock that is still stale is removed.
/// (A break guard left by a crashed breaker is itself reclaimed after the stale window. Residual: a holder releasing in the microseconds
/// between the re-check and the remove; it needs the holder to have already missed a whole stale window.)
fn break_stale_lock(p: &Path) {
    let bp = p.with_extension("break");
    match OpenOptions::new().write(true).create_new(true).open(&bp) {
        Ok(_) => {
            if lock_is_stale(p) {
                let _ = fs::remove_file(p);
            }
            let _ = fs::remove_file(&bp);
        }
        Err(_) => {
            if lock_is_stale(&bp) {
                let _ = fs::remove_file(&bp);
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

#[cfg(test)]
type StaleHook = Box<dyn Fn(&Path) + Send>;
#[cfg(test)]
static STALE_HOOK: std::sync::Mutex<Option<(PathBuf, StaleHook)>> = std::sync::Mutex::new(None);
#[cfg(test)]
static STALE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[cfg(test)]
fn test_hook_after_stale_judged(p: &Path) {
    let mut g = STALE_HOOK.lock().unwrap();
    if g.as_ref().map(|(d, _)| d == p).unwrap_or(false) {
        if let Some((_, f)) = g.take() {
            f(p);
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// B1: read a text file as lines, tolerating invalid UTF-8 (a crash / half-synced file can end inside a multi-byte character; `BufRead::lines` would
/// return an error and make the WHOLE file unreadable). Invalid bytes become U+FFFD, so the torn line simply fails to parse and is skipped.
/// B33: replace `dest` atomically AND durably: write a temp file, fsync it, rename over `dest`, then (unix) fsync the directory so the rename itself survives a
/// power loss. Without the fsync a crash after the rename can leave a zero-length / torn replacement of the previous good file (the data blocks were never flushed).
fn replace_durable(tmp: &Path, dest: &Path, data: &[u8]) -> io::Result<()> {
    {
        let mut f = File::create(tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    fs::rename(tmp, dest)?;
    #[cfg(unix)]
    if let Some(parent) = dest.parent() {
        if let Ok(d) = File::open(parent) {
            let _ = d.sync_all();
        }
    }
    Ok(())
}

/// B31: a file name in the shared folder that equals one of `own` ignoring case but not exactly: on a case-insensitive filesystem (Windows, macOS) it IS the same
/// file, so replacing our shard would overwrite another machine's.
fn case_collision(files: &[PathBuf], own: &[&str]) -> Option<String> {
    for p in files {
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if own.iter().any(|o| *o != name && o.eq_ignore_ascii_case(&name)) {
            return Some(name);
        }
    }
    None
}

fn lossy_lines(p: &Path) -> io::Result<Vec<String>> {
    let mut raw = Vec::new();
    File::open(p)?.read_to_end(&mut raw)?;
    Ok(raw.split(|b| *b == b'\n').map(|l| String::from_utf8_lossy(l).trim_end_matches('\r').to_string()).collect())
}

/// One routing outcome (the unit of shared learning).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Outcome {
    /// globally unique: `<session>:<seq>`
    pub id: String,
    pub session: String,
    pub seq: u64,
    /// event time (ms since epoch, or a logical clock for simulations); only used to order replay
    pub ts: u64,
    /// namespace of the world/domain the outcome was observed in (e.g. "sim:fixed", "project-x")
    pub world: String,
    pub features: Vec<String>,
    pub task_type: Option<String>,
    pub agent: String,
    pub reward: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LessonRec {
    pub id: String,
    pub session: String,
    pub ts: u64,
    pub world: String,
    pub agent: Option<String>,
    pub tag: String,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DocRec {
    pub id: String,
    pub text: String,
    pub tag: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SyncReport {
    pub exported: usize,
    pub imported: usize,
    pub lessons_imported: usize,
    pub files_read: usize,
    /// peer files that could not be read (locked, permission denied, not a regular file): skipped, the sync still completes (B29)
    pub files_skipped: usize,
    pub bytes_written: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MergeReport {
    pub outcomes_added: usize,
    pub lessons_added: usize,
    pub docs_added: usize,
    /// B24: corpus namespaces NOT merged because the two stores use different vector dimensions (reported instead of aborting the whole merge half-way)
    pub ns_skipped: usize,
    /// B24: docs whose id exists locally with DIFFERENT text (local kept, as before, but now counted instead of silent)
    pub doc_conflicts: usize,
}

#[derive(Debug)]
pub enum StoreError {
    Io(io::Error),
    Corrupt(String),
    Locked(PathBuf),
    Schema(u32),
}
impl From<io::Error> for StoreError {
    fn from(e: io::Error) -> Self {
        StoreError::Io(e)
    }
}
impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Io(e) => write!(f, "io: {e}"),
            StoreError::Corrupt(s) => write!(f, "corrupt store: {s}"),
            StoreError::Locked(p) => write!(f, "store is locked: {}", p.display()),
            StoreError::Schema(v) => write!(f, "unsupported store schema {v}"),
        }
    }
}
impl std::error::Error for StoreError {}
type R<T> = Result<T, StoreError>;

struct Guard(PathBuf);
impl Guard {
    /// B19: refresh the lock's mtime so a LONG operation (gc_docs is O(n^2 d)) is not mistaken for a crashed holder and its lock broken by another process.
    fn heartbeat(&self) {
        let _ = fs::write(&self.0, format!("{}
", now_ms()));
    }
}
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[derive(Clone, Debug)]
pub struct MeshStore {
    root: PathBuf,
}

impl MeshStore {
    /// Open (creating if needed) a store. Fails on an unknown schema version.
    pub fn open<P: AsRef<Path>>(root: P) -> R<MeshStore> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("router"))?;
        fs::create_dir_all(root.join("lessons"))?;
        fs::create_dir_all(root.join("knowledge"))?;
        let s = MeshStore { root };
        let mp = s.root.join("MANIFEST.json");
        if mp.exists() {
            let v: serde_json::Value = serde_json::from_slice(&fs::read(&mp)?).map_err(|e| StoreError::Corrupt(format!("MANIFEST: {e}")))?;
            let ver = v["schema_version"].as_u64().unwrap_or(0) as u32;
            if ver != SCHEMA_VERSION {
                return Err(StoreError::Schema(ver));
            }
        } else {
            let _g = s.lock()?;
            if !mp.exists() {
                let m = serde_json::json!({
                    "schema_version": SCHEMA_VERSION, "kind": "neural-mesh-memory", "created_ms": now_ms(),
                    "note": "ONLY synthetic routing outcomes, lessons and knowledge. Never secrets, API keys or customer data. Append-only; merge = union by id."
                });
                fs::write(&mp, serde_json::to_vec_pretty(&m).unwrap())?;
            }
        }
        Ok(s)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn lock(&self) -> R<Guard> {
        let p = self.root.join(".lock");
        let t0 = Instant::now();
        loop {
            match OpenOptions::new().write(true).create_new(true).open(&p) {
                Ok(mut f) => {
                    let _ = writeln!(f, "{}", now_ms());
                    return Ok(Guard(p));
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists || e.kind() == io::ErrorKind::PermissionDenied => {
                    if lock_is_stale(&p) {
                        #[cfg(test)]
                        test_hook_after_stale_judged(&p);
                        break_stale_lock(&p);
                        continue;
                    }
                    if t0.elapsed() > Duration::from_millis(LOCK_TIMEOUT_MS) {
                        return Err(StoreError::Locked(p));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    fn append_lines(&self, rel: &str, lines: &[String]) -> R<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let _g = self.lock()?;
        let p = self.root.join(rel);
        if let Some(d) = p.parent() {
            fs::create_dir_all(d)?;
        }
        let mut buf = String::new();
        // B1: a crash mid-append leaves a torn last line WITHOUT a newline; appending straight after it would glue the new record onto the
        // fragment and lose it. Start on a fresh line when the file does not end with one.
        if let Ok(mut rf) = File::open(&p) {
            let len = rf.metadata().map(|m| m.len()).unwrap_or(0);
            if len > 0 {
                use std::io::{Seek, SeekFrom};
                let mut last = [0u8; 1];
                if rf.seek(SeekFrom::Start(len - 1)).is_ok() && rf.read_exact(&mut last).is_ok() && last[0] != b'\n' {
                    buf.push('\n');
                }
            }
        }
        let mut f = OpenOptions::new().create(true).append(true).open(p)?;
        for l in lines {
            buf.push_str(l);
            buf.push('\n');
        }
        f.write_all(buf.as_bytes())?;
        Ok(())
    }

    fn read_jsonl<T: for<'de> Deserialize<'de>>(&self, rel: &str) -> R<Vec<T>> {
        let p = self.root.join(rel);
        if !p.exists() {
            return Ok(vec![]);
        }
        let mut out = vec![];
        for l in lossy_lines(&p)? {
            if l.trim().is_empty() {
                continue;
            }
            // a torn last line (crash mid-append) is skipped, never fatal
            if let Ok(v) = serde_json::from_str::<T>(&l) {
                out.push(v);
            }
        }
        Ok(out)
    }

    // ------------------------------------------------------------------ outcomes
    pub fn append_outcomes(&self, os: &[Outcome]) -> R<()> {
        let lines: Vec<String> = os.iter().map(|o| serde_json::to_string(o).unwrap()).collect();
        self.append_lines("router/journal.jsonl", &lines)
    }
    pub fn append_outcome(&self, o: &Outcome) -> R<()> {
        self.append_outcomes(std::slice::from_ref(o))
    }
    /// all outcomes, deduplicated by id, in replay order (ts, session, seq)
    pub fn outcomes(&self) -> R<Vec<Outcome>> {
        let mut seen = HashSet::new();
        let mut v: Vec<Outcome> = self.read_jsonl::<Outcome>("router/journal.jsonl")?.into_iter().filter(|o| seen.insert(o.id.clone())).collect();
        v.sort_by(|a, b| (a.ts, &a.session, a.seq).cmp(&(b.ts, &b.session, b.seq)));
        Ok(v)
    }

    /// Warm-start any router by replaying the journal through `Router::update`. `index_of` maps an agent name to the router's
    /// index (None = unknown agent, skipped). `world` filters by namespace (None = all). `max_recent` keeps only the newest N.
    /// Returns the number of replayed outcomes.
    pub fn warm_replay<R2: Router + ?Sized>(&self, router: &mut R2, index_of: &dyn Fn(&str) -> Option<usize>, world: Option<&str>, max_recent: Option<usize>) -> R<usize> {
        let mut os = self.outcomes()?;
        if let Some(w) = world {
            os.retain(|o| o.world == w);
        }
        if let Some(m) = max_recent {
            if os.len() > m {
                os.drain(..os.len() - m);
            }
        }
        let mut n = 0;
        for (i, o) in os.iter().enumerate() {
            if let Some(a) = index_of(&o.agent) {
                let t = Task { id: i as u64, task_type: o.task_type.clone(), features: o.features.clone() };
                router.update(&t, a, o.reward.clamp(0.0, 1.0));
                n += 1;
            }
        }
        Ok(n)
    }

    /// R56 (opt-in): like [`MeshStore::warm_replay`] but FOREIGN outcomes (`is_own` false) are discounted: only every `foreign_every`-th foreign outcome (in replay order)
    /// is replayed, own outcomes always. `foreign_every <= 1` = plain replay. Measured (R56, simulator, 4 machines x 250): the Guarded `recommended` router gains
    /// +.008..+.019 H100 over full sharing at foreign_every = 4 (pre-registered fresh-seed confirmation +.0084 [+.0023,+.0146]); the plain Hebbian router does NOT
    /// (every 4: -.006 n.s.; every 8: -.016 Holm-significant loss); at 2 machines there is no gain for either. Returns the number of replayed outcomes.
    pub fn warm_replay_discounted<R2: Router + ?Sized>(&self, router: &mut R2, index_of: &dyn Fn(&str) -> Option<usize>, world: Option<&str>, is_own: &dyn Fn(&Outcome) -> bool, foreign_every: usize) -> R<usize> {
        let mut os = self.outcomes()?;
        if let Some(w) = world {
            os.retain(|o| o.world == w);
        }
        let every = foreign_every.max(1);
        let mut c = 0usize;
        let mut n = 0;
        for (i, o) in os.iter().enumerate() {
            if !is_own(o) {
                let keep = c % every == 0;
                c += 1;
                if !keep {
                    continue;
                }
            }
            if let Some(a) = index_of(&o.agent) {
                let t = Task { id: i as u64, task_type: o.task_type.clone(), features: o.features.clone() };
                router.update(&t, a, o.reward.clamp(0.0, 1.0));
                n += 1;
            }
        }
        Ok(n)
    }

    /// GC / compaction (R35): rewrite the journal keeping, per `world`, only the newest `keep` outcomes (replay order), deduplicated.
    /// Atomic (temp file + rename) under the lock. Returns (before, after). Lessons and corpus are never touched.
    pub fn compact_journal(&self, keep: usize) -> R<(usize, usize)> {
        // B1: take the lock BEFORE reading, otherwise an outcome appended by another process between the read and the rename is silently lost.
        let _g = self.lock()?;
        let all = self.outcomes()?;
        let before = all.len();
        let mut by: BTreeMap<String, Vec<Outcome>> = BTreeMap::new();
        for o in all {
            by.entry(o.world.clone()).or_default().push(o);
        }
        let mut kept: Vec<Outcome> = vec![];
        for (_, mut v) in by {
            if v.len() > keep {
                v.drain(..v.len() - keep);
            }
            kept.extend(v);
        }
        kept.sort_by(|a, b| (a.ts, &a.session, a.seq).cmp(&(b.ts, &b.session, b.seq)));
        let p = self.root.join("router/journal.jsonl");
        let tmp = self.root.join("router/journal.jsonl.tmp");
        let mut buf = String::new();
        for o in &kept {
            buf.push_str(&serde_json::to_string(o).unwrap());
            buf.push('\n');
        }
        replace_durable(&tmp, &p, buf.as_bytes())?;
        Ok((before, kept.len()))
    }

    // ------------------------------------------------------------------ lessons
    pub fn append_lesson(&self, l: &LessonRec) -> R<()> {
        self.append_lines("lessons/lessons.jsonl", &[serde_json::to_string(l).unwrap()])
    }
    pub fn lessons(&self) -> R<Vec<LessonRec>> {
        let mut seen = HashSet::new();
        Ok(self.read_jsonl::<LessonRec>("lessons/lessons.jsonl")?.into_iter().filter(|l| seen.insert(l.id.clone())).collect())
    }

    // ------------------------------------------------------------------ knowledge corpus (+ vectors; index rebuilt on load)
    fn kdir(&self, ns: &str) -> PathBuf {
        self.root.join("knowledge").join(ns)
    }
    /// Append documents with their vectors (L2-normalised by the caller). Existing ids are skipped. Returns the number added.
    pub fn put_docs(&self, ns: &str, dim: usize, docs: &[DocRec], vecs: &[f32]) -> R<usize> {
        if vecs.len() != docs.len() * dim {
            return Err(StoreError::Corrupt("vector/doc count mismatch".into()));
        }
        let _g = self.lock()?;
        let d = self.kdir(ns);
        fs::create_dir_all(&d)?;
        let meta = d.join("meta.json");
        if meta.exists() {
            let m: serde_json::Value = serde_json::from_slice(&fs::read(&meta)?).map_err(|e| StoreError::Corrupt(format!("meta: {e}")))?;
            if m["dim"].as_u64() != Some(dim as u64) {
                return Err(StoreError::Corrupt(format!("namespace {ns} has dim {} not {dim}", m["dim"])));
            }
        } else {
            fs::write(&meta, serde_json::to_vec(&serde_json::json!({"dim": dim})).unwrap())?;
        }
        let existing = self.read_jsonl::<DocRec>(&format!("knowledge/{ns}/docs.jsonl"))?;
        let have: HashSet<String> = existing.iter().map(|x| x.id.clone()).collect();
        // B1: a crash between the vector write and the doc write leaves an orphan vector row; drop it (and any torn partial row) so that row i
        // of vecs.f32 stays aligned with doc i. Also a torn docs line is terminated first.
        {
            let vp = d.join("vecs.f32");
            if vp.exists() {
                let want = (existing.len() * dim * 4) as u64;
                let f = OpenOptions::new().write(true).open(&vp)?;
                if f.metadata()?.len() > want {
                    f.set_len(want)?;
                }
            }
            let dp = d.join("docs.jsonl");
            if let Ok(mut rf) = File::open(&dp) {
                use std::io::{Seek, SeekFrom};
                let len = rf.metadata().map(|m| m.len()).unwrap_or(0);
                let mut last = [0u8; 1];
                if len > 0 && rf.seek(SeekFrom::Start(len - 1)).is_ok() && rf.read_exact(&mut last).is_ok() && last[0] != b'\n' {
                    OpenOptions::new().append(true).open(&dp)?.write_all(b"\n")?;
                }
            }
        }
        let mut dl = OpenOptions::new().create(true).append(true).open(d.join("docs.jsonl"))?;
        let mut vl = OpenOptions::new().create(true).append(true).open(d.join("vecs.f32"))?;
        let mut added = 0;
        for (i, doc) in docs.iter().enumerate() {
            if have.contains(&doc.id) {
                continue;
            }
            // vectors first: a crash between the two writes leaves an extra vector row, which load ignores
            let mut b = Vec::with_capacity(dim * 4);
            for x in &vecs[i * dim..(i + 1) * dim] {
                b.extend_from_slice(&x.to_le_bytes());
            }
            vl.write_all(&b)?;
            writeln!(dl, "{}", serde_json::to_string(doc).unwrap())?;
            added += 1;
        }
        Ok(added)
    }
    /// GC / compaction of a corpus namespace (R47): drop later documents whose L2-normalised vector has cosine > `threshold` to an EARLIER kept document
    /// (first wins; `threshold >= 1.0` removes only exact-vector duplicates). Atomic (temp files + rename) under the lock; the persisted ANN stamp changes
    /// because the row count changes, so `open_ann` rebuilds. O(n^2 d): intended for offline maintenance (6.8k docs ~ seconds). Returns (before, after).
    /// Measured on the real 6,831-paragraph store (RD-LOG R47): .98 removes 1.1%, .95 removes 2.2%, retrieval TOST-equivalent (+-.01); saves little there.
    pub fn gc_docs(&self, ns: &str, threshold: f32) -> R<(usize, usize)> {
        let _g = self.lock()?;
        let (docs, vecs, dim) = self.load_docs(ns)?;
        let n = docs.len();
        if n == 0 || dim == 0 {
            return Ok((n, n));
        }
        let mut unit = vecs.clone();
        for r in 0..n {
            let row = &mut unit[r * dim..(r + 1) * dim];
            let nn = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if nn > 0.0 {
                row.iter_mut().for_each(|x| *x /= nn);
            }
        }
        let mut keep: Vec<usize> = vec![];
        for i in 0..n {
            if i % 64 == 0 {
                _g.heartbeat();
            }
            let vi = &unit[i * dim..(i + 1) * dim];
            let dup = keep.iter().any(|&j| {
                let vj = &unit[j * dim..(j + 1) * dim];
                let c: f32 = vi.iter().zip(vj).map(|(a, b)| a * b).sum();
                c > threshold.min(1.0 - 1e-6)
            });
            if !dup {
                keep.push(i);
            }
        }
        if keep.len() == n {
            return Ok((n, n));
        }
        let d = self.kdir(ns);
        let (dt, vt) = (d.join("docs.jsonl.tmp"), d.join("vecs.f32.tmp"));
        let mut db = String::new();
        let mut vb: Vec<u8> = Vec::with_capacity(keep.len() * dim * 4);
        for &i in &keep {
            db.push_str(&serde_json::to_string(&docs[i]).unwrap());
            db.push(char::from(10u8));
            for x in &vecs[i * dim..(i + 1) * dim] {
                vb.extend_from_slice(&x.to_le_bytes());
            }
        }
        // B30: remember the ids this gc dropped so `sync_docs_folder` does not re-import them from a peer that still holds them (the dropped near-duplicates
        // otherwise came back on the next sync). Written BEFORE the rewrite: a crash in between leaves a harmless extra tombstone, never a resurrection.
        {
            let kept: HashSet<usize> = keep.iter().copied().collect();
            let mut tb = String::new();
            for i in (0..n).filter(|i| !kept.contains(i)) {
                tb.push_str(&serde_json::to_string(&docs[i].id).unwrap());
                tb.push(char::from(10u8));
            }
            let mut tf = OpenOptions::new().create(true).append(true).open(d.join("tombstones.jsonl"))?;
            tf.write_all(tb.as_bytes())?;
        }
        replace_durable(&vt, &d.join("vecs.f32"), &vb)?;
        replace_durable(&dt, &d.join("docs.jsonl"), db.as_bytes())?;
        Ok((n, keep.len()))
    }
    /// Load a namespace: (docs, row-major vectors, dim). Rows beyond the docs (torn write) are ignored.
    pub fn load_docs(&self, ns: &str) -> R<(Vec<DocRec>, Vec<f32>, usize)> {
        let d = self.kdir(ns);
        let meta = d.join("meta.json");
        if !meta.exists() {
            return Ok((vec![], vec![], 0));
        }
        let m: serde_json::Value = serde_json::from_slice(&fs::read(&meta)?).map_err(|e| StoreError::Corrupt(format!("meta: {e}")))?;
        let dim = m["dim"].as_u64().unwrap_or(0) as usize;
        let docs: Vec<DocRec> = self.read_jsonl(&format!("knowledge/{ns}/docs.jsonl"))?;
        let mut raw = Vec::new();
        File::open(d.join("vecs.f32"))?.read_to_end(&mut raw)?;
        let rows = raw.len() / 4 / dim.max(1);
        let n = docs.len().min(rows);
        let v: Vec<f32> = raw.chunks_exact(4).take(n * dim).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Ok((docs.into_iter().take(n).collect(), v, dim))
    }
    pub fn namespaces(&self) -> R<Vec<String>> {
        let mut v = vec![];
        for e in fs::read_dir(self.root.join("knowledge"))? {
            let e = e?;
            if e.file_type()?.is_dir() {
                v.push(e.file_name().to_string_lossy().to_string());
            }
        }
        v.sort();
        Ok(v)
    }

    // ------------------------------------------------------------------ shared-folder sync (R48)
    /// Serverless cross-machine sync through a shared folder (Dropbox / OneDrive / NFS style). Protocol: every machine OWNS exactly one file per kind in the
    /// folder (`<machine>.journal.jsonl`, `<machine>.lessons.jsonl`) and only ever replaces its own file (temp + rename), so two machines never write the same
    /// file and no cross-machine lock is needed. `sync_folder` first imports every OTHER machine's files (union by id; a torn trailing line is skipped;
    /// `worlds = Some(..)` imports only outcomes of those namespaces), then re-exports the outcomes that no other machine's file already holds (own-origin data). Idempotent and
    /// order-independent. Corpus namespaces are NOT synced (large; use `merge_from`). The shared folder must only hold synthetic / non-sensitive data.
    pub fn sync_folder(&self, folder: &Path, machine: &str, worlds: Option<&[&str]>, cap_per_world: Option<usize>) -> R<SyncReport> {
        if machine.is_empty() || !machine.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err(StoreError::Corrupt("machine id must be [A-Za-z0-9_-]+".into()));
        }
        fs::create_dir_all(folder)?;
        let mut rep = SyncReport::default();
        let own_j = format!("{machine}.journal.jsonl");
        let own_l = format!("{machine}.lessons.jsonl");
        let mut files: Vec<PathBuf> = fs::read_dir(folder)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        files.sort();
        // B31: refuse (before touching anything) when another machine's file differs from ours only by case
        if let Some(n) = case_collision(&files, &[&own_j, &own_l]) {
            return Err(StoreError::Corrupt(format!("machine id '{machine}' collides case-insensitively with existing shared file '{n}'; pick a machine id that differs by more than case")));
        }
        let mut seen: HashSet<String> = self.outcomes()?.into_iter().map(|o| o.id).collect();
        let mut seen_l: HashSet<String> = self.lessons()?.into_iter().map(|l| l.id).collect();
        let (mut add_o, mut add_l): (Vec<Outcome>, Vec<LessonRec>) = (vec![], vec![]);
        let mut foreign: HashSet<String> = HashSet::new(); // ids present in OTHER machines' files (any world): we do not re-export those
        for p in &files {
            let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if name == own_j || name == own_l || name.ends_with(".tmp") {
                continue;
            }
            if name.ends_with(".journal.jsonl") {
                // B29: unreadable peer file -> skipped and counted (see sync_docs_folder)
                let Ok(lines) = lossy_lines(p) else {
                    rep.files_skipped += 1;
                    continue;
                };
                rep.files_read += 1;
                for l in lines {
                    if let Ok(o) = serde_json::from_str::<Outcome>(&l) {
                        foreign.insert(o.id.clone());
                        if worlds.map_or(true, |ws| ws.iter().any(|w| *w == o.world)) && seen.insert(o.id.clone()) {
                            add_o.push(o);
                        }
                    }
                }
            } else if name.ends_with(".lessons.jsonl") {
                let Ok(lines) = lossy_lines(p) else {
                    rep.files_skipped += 1;
                    continue;
                };
                rep.files_read += 1;
                for l in lines {
                    if let Ok(x) = serde_json::from_str::<LessonRec>(&l) {
                        if seen_l.insert(x.id.clone()) {
                            add_l.push(x);
                        }
                    }
                }
            }
        }
        rep.imported = add_o.len();
        rep.lessons_imported = add_l.len();
        self.append_outcomes(&add_o)?;
        for l in &add_l {
            self.append_lesson(l)?;
        }
        // optional shared GC policy: every machine applies the same deterministic newest-K-per-world cap to the UNION, so all converge to the same set
        // (older outcomes still sitting in other machines' files are re-imported and re-dropped on each sync; harmless, bounded)
        if let Some(k) = cap_per_world {
            self.compact_journal(k)?;
        }
        // export only outcomes that are NOT already in another machine's file (own-origin data): the folder then holds ~1x the union instead of Mx, and
        // if another machine's file disappears the next sync re-exports what only we hold (self-healing). Atomically replaces only OUR files.
        let nl = char::from(10u8);
        let all: Vec<Outcome> = self.outcomes()?.into_iter().filter(|o| !foreign.contains(&o.id)).collect();
        let mut buf = String::new();
        for o in &all {
            buf.push_str(&serde_json::to_string(o).unwrap());
            buf.push(nl);
        }
        rep.exported = all.len();
        rep.bytes_written = buf.len();
        let tmp = folder.join(format!("{own_j}.tmp"));
        replace_durable(&tmp, &folder.join(&own_j), buf.as_bytes())?;
        let mut lb = String::new();
        for l in self.lessons()? {
            lb.push_str(&serde_json::to_string(&l).unwrap());
            lb.push(nl);
        }
        let tmp = folder.join(format!("{own_l}.tmp"));
        replace_durable(&tmp, &folder.join(&own_l), lb.as_bytes())?;
        Ok(rep)
    }

    // ------------------------------------------------------------------ merge / stats
    /// Union another store into this one (idempotent, commutative up to file order): by event id.
    pub fn merge_from(&self, other: &MeshStore) -> R<MergeReport> {
        let mut rep = MergeReport::default();
        let mine: HashSet<String> = self.outcomes()?.into_iter().map(|o| o.id).collect();
        let add: Vec<Outcome> = other.outcomes()?.into_iter().filter(|o| !mine.contains(&o.id)).collect();
        rep.outcomes_added = add.len();
        self.append_outcomes(&add)?;
        let ml: HashSet<String> = self.lessons()?.into_iter().map(|l| l.id).collect();
        for l in other.lessons()?.into_iter().filter(|l| !ml.contains(&l.id)) {
            self.append_lesson(&l)?;
            rep.lessons_added += 1;
        }
        for ns in other.namespaces()? {
            let (docs, vecs, dim) = other.load_docs(&ns)?;
            let (mdocs, _mv, mdim) = self.load_docs(&ns)?;
            if !mdocs.is_empty() && mdim != dim {
                rep.ns_skipped += 1;
                continue;
            }
            let mine: std::collections::HashMap<&str, &str> = mdocs.iter().map(|d| (d.id.as_str(), d.text.as_str())).collect();
            rep.doc_conflicts += docs.iter().filter(|d| mine.get(d.id.as_str()).map_or(false, |t| *t != d.text.as_str())).count();
            rep.docs_added += self.put_docs(&ns, dim, &docs, &vecs)?;
        }
        Ok(rep)
    }

    pub fn stats(&self) -> R<BTreeMap<String, u64>> {
        let mut m = BTreeMap::new();
        let os = self.outcomes()?;
        m.insert("outcomes".into(), os.len() as u64);
        m.insert("sessions".into(), os.iter().map(|o| o.session.clone()).collect::<HashSet<_>>().len() as u64);
        m.insert("lessons".into(), self.lessons()?.len() as u64);
        for ns in self.namespaces()? {
            m.insert(format!("docs:{ns}"), self.load_docs(&ns)?.0.len() as u64);
        }
        Ok(m)
    }
}

// ---------------------------------------------------------------------------------------------
// persisted ANN index (hnsw_rs graph dump + reload); measured in RD-LOG (reload ~38 ms vs build ~5-7 s for 6.8k docs)

/// An HNSW graph over a stored corpus namespace, reloadable from disk without a rebuild.
pub struct PersistedAnn {
    h: hnsw_rs::prelude::Hnsw<'static, f32, hnsw_rs::prelude::DistCosine>,
    pub n: usize,
    pub dim: usize,
    /// true when the graph was rebuilt from vectors, false when reloaded from the dump
    pub rebuilt: bool,
}
/// 64-bit content fingerprint of a corpus (doc ids in order + every vector bit); order- and content-sensitive, ~ns per vector element.
fn corpus_fingerprint(docs: &[DocRec], vecs: &[f32]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    };
    for d in docs {
        for b in d.id.bytes() {
            mix(b as u64);
        }
        mix(0xff);
    }
    for v in vecs {
        mix(v.to_bits() as u64);
    }
    mix(docs.len() as u64);
    format!("{h:016x}")
}
impl PersistedAnn {
    /// top-k (doc index in `load_docs` order, cosine similarity)
    pub fn search(&self, q: &[f32], k: usize, ef: usize) -> Vec<(usize, f32)> {
        self.h.search(q, k, ef.max(k)).into_iter().map(|n| (n.d_id, 1.0 - n.distance)).collect()
    }
}
impl MeshStore {
    fn hnsw_dir(&self, ns: &str) -> PathBuf {
        self.kdir(ns).join("hnsw")
    }
    /// Reload the dumped graph if it matches the current corpus size, else rebuild from the stored vectors and dump it again.
    pub fn open_ann(&self, ns: &str) -> R<Option<PersistedAnn>> {
        use hnsw_rs::prelude::*;
        let (docs, vecs, dim) = self.load_docs(ns)?;
        if docs.is_empty() {
            return Ok(None);
        }
        let dir = self.hnsw_dir(ns);
        let stamp = dir.join("stamp.json");
        let fp = corpus_fingerprint(&docs, &vecs);
        if let Ok(b) = fs::read(&stamp) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&b) {
                // B18: (n, dim) alone is not enough - gc_docs followed by put_docs keeps n but changes the content, and the old graph would be served for the wrong vectors.
                // The stamp therefore also carries a content fingerprint (ids + vector bits); stamps without one are treated as stale (one safe rebuild).
                if v["n"].as_u64() == Some(docs.len() as u64) && v["dim"].as_u64() == Some(dim as u64) && v["fp"].as_str() == Some(fp.as_str()) {
                    if let Some(base) = v["base"].as_str() {
                        // the loader borrows from `HnswIo`: leak it so the graph is 'static (one small allocation per open)
                        let io: &'static mut HnswIo = Box::leak(Box::new(HnswIo::new(&dir, base)));
                        if let Ok(h) = io.load_hnsw::<f32, DistCosine>() {
                            return Ok(Some(PersistedAnn { h, n: docs.len(), dim, rebuilt: false }));
                        }
                    }
                }
            }
        }
        let h: Hnsw<'static, f32, DistCosine> = Hnsw::new(16, docs.len().max(1), 16, 200, DistCosine {});
        for i in 0..docs.len() {
            h.insert_slice((&vecs[i * dim..(i + 1) * dim], i));
        }
        let _g = self.lock()?;
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir)?;
        let base = h.file_dump(&dir, "ix").map_err(|e| StoreError::Corrupt(format!("hnsw dump: {e}")))?;
        fs::write(&stamp, serde_json::to_vec(&serde_json::json!({"n": docs.len(), "dim": dim, "base": base, "fp": fp})).unwrap())?;
        Ok(Some(PersistedAnn { h, n: docs.len(), dim, rebuilt: true }))
    }
}

// ---------------------------------------------------------------------- shared-folder CORPUS sync (B15)
const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
fn b64_encode(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(B64[(n >> 18) as usize & 63] as char);
        s.push(B64[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    s
}
fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> { B64.iter().position(|x| *x == c).map(|p| p as u32) };
    let b = s.as_bytes();
    if b.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(b.len() / 4 * 3);
    for c in b.chunks(4) {
        let pad = c.iter().filter(|x| **x == b'=').count();
        let n = val(c[0])? << 18 | val(c[1])? << 12 | if pad < 2 { val(c[2])? << 6 } else { 0 } | if pad < 1 { val(c[3])? } else { 0 };
        out.push((n >> 16) as u8);
        if pad < 2 {
            out.push((n >> 8) as u8);
        }
        if pad < 1 {
            out.push(n as u8);
        }
    }
    Some(out)
}
/// per-vector symmetric int8: 4-byte little-endian f32 scale (max|x|/127) followed by `dim` i8 values (the R9 recipe: no Holm loss on any retrieval axis)
fn q8_encode(v: &[f32]) -> Vec<u8> {
    let m = v.iter().fold(0f32, |a, x| a.max(x.abs()));
    let scale = if m > 0.0 { m / 127.0 } else { 1.0 };
    let mut out = Vec::with_capacity(4 + v.len());
    out.extend_from_slice(&scale.to_le_bytes());
    out.extend(v.iter().map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8 as u8));
    out
}
fn q8_decode(b: &[u8], dim: usize) -> Option<Vec<f32>> {
    if b.len() != 4 + dim {
        return None;
    }
    let scale = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    let v: Vec<f32> = b[4..].iter().map(|x| (*x as i8) as f32 * scale).collect();
    // B28: a corrupt shard must not import NaN / inf vectors (they poison every cosine); a non-finite scale or an overflowing product is undecodable
    if !scale.is_finite() || v.iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some(v)
}
#[derive(Serialize, Deserialize)]
struct DocShardLine {
    id: String,
    text: String,
    tag: String,
    dim: usize,
    q: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct DocSyncReport {
    pub exported: usize,
    pub imported: usize,
    /// same id present locally with DIFFERENT text: the local doc is kept, the foreign one is NOT imported (reported, never silently overwritten)
    pub conflicts: usize,
    /// foreign shards / lines skipped because of a vector dimension mismatch or an undecodable vector
    pub skipped: usize,
    pub files_read: usize,
    /// peer shards that could not be read (B29)
    pub skipped_files: usize,
    pub bytes_written: usize,
}

impl MeshStore {
    /// B15 (opt-in): serverless cross-machine sync of ONE corpus namespace through a shared folder, same ownership protocol as [`MeshStore::sync_folder`]: machine `m` owns
    /// exactly one shard file `<m>.<ns>.docs.jsonl` (one JSON line per doc: id, text, tag, dim, base64 of the per-vector int8 code), replaced atomically (temp + rename); other machines'
    /// shards are imported (union by doc id; torn lines / invalid UTF-8 tolerated) and only own-origin docs (not present in any foreign shard) are re-exported.
    /// Vectors travel as int8 (388 B at dim 384 instead of 1,536 B f32); imported vectors are the dequantised values (R9: no retrieval loss on the battery). Same id with different text =
    /// counted conflict, local kept. Machine ids and namespace names must be `[A-Za-z0-9_-]+`. Idempotent and order-independent. Synthetic / non-sensitive data only.
    pub fn sync_docs_folder(&self, folder: &Path, machine: &str, ns: &str) -> R<DocSyncReport> {
        let ok = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
        if !ok(machine) || !ok(ns) {
            return Err(StoreError::Corrupt("machine id and namespace must be [A-Za-z0-9_-]+".into()));
        }
        fs::create_dir_all(folder)?;
        let mut rep = DocSyncReport::default();
        let own = format!("{machine}.{ns}.docs.jsonl");
        let suffix = format!(".{ns}.docs.jsonl");
        let (ldocs, lvecs, ldim) = self.load_docs(ns)?;
        let local: BTreeMap<String, usize> = ldocs.iter().enumerate().map(|(i, d)| (d.id.clone(), i)).collect();
        let mut files: Vec<PathBuf> = fs::read_dir(folder)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        files.sort();
        if let Some(n) = case_collision(&files, &[&own]) {
            return Err(StoreError::Corrupt(format!("machine id '{machine}' / namespace '{ns}' collide case-insensitively with existing shared file '{n}'; pick names that differ by more than case")));
        }
        let mut foreign: HashSet<String> = HashSet::new();
        let mut new_docs: Vec<DocRec> = vec![];
        let mut new_vecs: Vec<f32> = vec![];
        let mut new_dim = ldim;
        let mut taken: BTreeMap<String, String> = BTreeMap::new(); // id -> text of the first foreign line imported (B32: later disagreeing shards are counted)
        let tomb: HashSet<String> = lossy_lines(&self.kdir(ns).join("tombstones.jsonl")).map(|ls| ls.iter().filter_map(|l| serde_json::from_str::<String>(l).ok()).collect()).unwrap_or_default();
        for p in &files {
            let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if name == own || !name.ends_with(&suffix) {
                continue;
            }
            // B29: an unreadable peer shard (locked by a sync client, permission denied, not a regular file) is skipped and counted, not fatal
            let Ok(lines) = lossy_lines(p) else {
                rep.skipped_files += 1;
                continue;
            };
            rep.files_read += 1;
            for l in lines {
                let Ok(x) = serde_json::from_str::<DocShardLine>(&l) else { continue };
                foreign.insert(x.id.clone());
                if tomb.contains(&x.id) {
                    continue; // locally gc-dropped near-duplicate: not re-imported (B30)
                }
                if let Some(&i) = local.get(&x.id) {
                    if ldocs[i].text != x.text {
                        rep.conflicts += 1;
                    }
                    continue;
                }
                if let Some(t) = taken.get(&x.id) {
                    if *t != x.text {
                        rep.conflicts += 1; // B32: two foreign shards disagree on this id's text; the first (sorted file order) wins, now visibly
                    }
                    continue;
                }
                // B28: dim 0 would pin the namespace to dim 0 (docs without vectors, hidden by load_docs) or abort the whole import on a count mismatch
                if x.dim == 0 || (new_dim != 0 && x.dim != new_dim) {
                    rep.skipped += 1;
                    continue;
                }
                let Some(v) = b64_decode(&x.q).and_then(|b| q8_decode(&b, x.dim)) else {
                    rep.skipped += 1;
                    continue;
                };
                new_dim = x.dim;
                taken.insert(x.id.clone(), x.text.clone());
                new_docs.push(DocRec { id: x.id, text: x.text, tag: x.tag });
                new_vecs.extend(v);
            }
        }
        if !new_docs.is_empty() {
            rep.imported = self.put_docs(ns, new_dim, &new_docs, &new_vecs)?;
        }
        // export own-origin docs (local docs that no foreign shard holds), atomically replacing OUR shard only
        let (docs, vecs, dim) = self.load_docs(ns)?;
        let mut buf = String::new();
        for (i, d) in docs.iter().enumerate() {
            if foreign.contains(&d.id) {
                continue;
            }
            let line = DocShardLine { id: d.id.clone(), text: d.text.clone(), tag: d.tag.clone(), dim, q: b64_encode(&q8_encode(&vecs[i * dim..(i + 1) * dim])) };
            buf.push_str(&serde_json::to_string(&line).unwrap());
            buf.push('\n');
            rep.exported += 1;
        }
        rep.bytes_written = buf.len();
        let tmp = folder.join(format!("{own}.tmp"));
        replace_durable(&tmp, &folder.join(&own), buf.as_bytes())?;
        Ok(rep)
    }
}

#[cfg(test)]
mod docsync_tests {
    use super::*;
    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("nm_docsync_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }
    fn vecs(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..n * dim)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
            })
            .collect()
    }
    fn docs(prefix: &str, n: usize) -> Vec<DocRec> {
        (0..n).map(|i| DocRec { id: format!("{prefix}{i}"), text: format!("t\u{e9} {prefix}{i}"), tag: "x".into() }).collect()
    }

    #[test]
    fn base64_and_int8_roundtrip() {
        for n in 0..9usize {
            let b: Vec<u8> = (0..n as u8).map(|x| x.wrapping_mul(37)).collect();
            assert_eq!(b64_decode(&b64_encode(&b)).unwrap(), b);
        }
        let v = vecs(1, 64, 3);
        let r = q8_decode(&q8_encode(&v), 64).unwrap();
        let m = v.iter().fold(0f32, |a, x| a.max(x.abs()));
        assert!(v.iter().zip(&r).all(|(a, b)| (a - b).abs() <= m / 127.0 * 0.5 + 1e-6), "int8 error bounded by half a step");
        assert!(q8_decode(&[0u8; 5], 64).is_none());
    }

    #[test]
    fn docs_sync_is_union_idempotent_torn_tolerant_and_reports_conflicts() {
        let folder = tmp("f");
        let dim = 16;
        let (pa, pb, pc) = (tmp("a"), tmp("b"), tmp("c"));
        let (a, b, c) = (MeshStore::open(&pa).unwrap(), MeshStore::open(&pb).unwrap(), MeshStore::open(&pc).unwrap());
        a.put_docs("kb", dim, &docs("a", 5), &vecs(5, dim, 1)).unwrap();
        b.put_docs("kb", dim, &docs("b", 4), &vecs(4, dim, 2)).unwrap();
        // c holds a doc with the SAME id as a's but different text -> conflict, not overwritten
        let mut cd = docs("c", 3);
        cd.push(DocRec { id: "a0".into(), text: "different".into(), tag: "x".into() });
        c.put_docs("kb", dim, &cd, &vecs(4, dim, 3)).unwrap();
        let ra = a.sync_docs_folder(&folder, "mA", "kb").unwrap();
        assert_eq!((ra.imported, ra.exported), (0, 5));
        b.sync_docs_folder(&folder, "mB", "kb").unwrap();
        let rc = c.sync_docs_folder(&folder, "mC", "kb").unwrap();
        assert_eq!(rc.conflicts, 1);
        // torn shard tail + non-UTF8 + junk tmp are tolerated
        let mut f = OpenOptions::new().append(true).open(folder.join("mB.kb.docs.jsonl")).unwrap();
        f.write_all(b"{\"id\":\"zz\",\"text\":\"caf\xc3").unwrap();
        drop(f);
        fs::write(folder.join("mZ.kb.docs.jsonl.tmp"), b"garbage").unwrap();
        for _ in 0..2 {
            for (s, m) in [(&a, "mA"), (&b, "mB"), (&c, "mC")] {
                s.sync_docs_folder(&folder, m, "kb").unwrap();
            }
        }
        for s in [&a, &b, &c] {
            assert_eq!(s.load_docs("kb").unwrap().0.len(), 5 + 4 + 3 + 0, "union of 5 + 4 + 3 distinct ids (a0 of c is a conflict, not a new doc)");
        }
        // vectors survive within int8 error and stay aligned with their docs
        let (da, va, _) = a.load_docs("kb").unwrap();
        let (dbb, vbb, _) = b.load_docs("kb").unwrap();
        let orig_a = vecs(5, dim, 1);
        for (i, d) in da.iter().enumerate() {
            if let Some(j) = dbb.iter().position(|x| x.id == d.id) {
                let (x, y) = (&va[i * dim..(i + 1) * dim], &vbb[j * dim..(j + 1) * dim]);
                assert!(x.iter().zip(y).all(|(p, q)| (p - q).abs() < 0.02), "doc {} vectors agree across machines", d.id);
            }
            if d.id == "a2" {
                assert!(va[i * dim..(i + 1) * dim].iter().zip(&orig_a[2 * dim..3 * dim]).all(|(p, q)| (p - q).abs() < 0.02));
            }
        }
        let again = a.sync_docs_folder(&folder, "mA", "kb").unwrap();
        assert_eq!(again.imported, 0);
        assert!(a.sync_docs_folder(&folder, "bad id", "kb").is_err());
        for p in [&folder, &pa, &pb, &pc] {
            let _ = fs::remove_dir_all(p);
        }
    }
}

#[cfg(test)]
mod hostile_shard_tests {
    use super::*;
    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("nm_hostile_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }
    fn line(id: &str, dim: usize, q: &[u8]) -> String {
        format!("{{\"id\":\"{id}\",\"text\":\"t {id}\",\"tag\":\"x\",\"dim\":{dim},\"q\":\"{}\"}}
", b64_encode(q))
    }
    /// B28: a single corrupt/hostile shard line must not poison the corpus (dim 0 pinned the namespace to dim 0 and hid every doc; a NaN/inf scale
    /// imported non-finite vectors that break every cosine).
    #[test]
    fn corrupt_shard_lines_cannot_poison_the_namespace() {
        let (folder, p) = (tmp("f1"), tmp("s1"));
        fs::create_dir_all(&folder).unwrap();
        let st = MeshStore::open(&p).unwrap();
        let dim = 8usize;
        let mut good = q8_encode(&vec![0.5f32; dim]);
        let mut nan = good.clone();
        nan[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut inf = good.clone();
        inf[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        let mut big = good.clone();
        big[..4].copy_from_slice(&f32::MAX.to_le_bytes());
        let mut body = String::new();
        body.push_str(&line("d0", 0, &[0, 0, 0, 0])); // dim 0
        body.push_str(&line("nan", dim, &nan));
        body.push_str(&line("inf", dim, &inf));
        body.push_str(&line("big", dim, &big));
        body.push_str(&line("ok", dim, &good));
        good.clear();
        fs::write(folder.join("evil.kb.docs.jsonl"), body).unwrap();
        let r = st.sync_docs_folder(&folder, "me", "kb").unwrap();
        assert_eq!(r.imported, 1, "only the well-formed line is imported: {r:?}");
        assert_eq!(r.skipped, 4);
        let (d, v, dm) = st.load_docs("kb").unwrap();
        assert_eq!((d.len(), dm), (1, dim), "namespace keeps the real dimension and the good doc stays visible");
        assert_eq!(d[0].id, "ok");
        assert!(v.iter().all(|x| x.is_finite()));
        for q in [&folder, &p] {
            let _ = fs::remove_dir_all(q);
        }
    }

    /// B29: an unreadable peer file (here a directory with a shard name, standing in for a locked / permission-denied file from a sync client) must not
    /// abort the whole sync before anything is imported or exported.
    #[test]
    fn unreadable_peer_file_is_skipped_not_fatal() {
        let (folder, p, p2) = (tmp("f2"), tmp("s2"), tmp("s3"));
        fs::create_dir_all(folder.join("bad.kb.docs.jsonl")).unwrap();
        fs::create_dir_all(folder.join("bad.journal.jsonl")).unwrap();
        fs::create_dir_all(folder.join("bad.lessons.jsonl")).unwrap();
        let (a, b) = (MeshStore::open(&p).unwrap(), MeshStore::open(&p2).unwrap());
        let dim = 4usize;
        let ds = vec![DocRec { id: "a1".into(), text: "alpha".into(), tag: "x".into() }];
        a.put_docs("kb", dim, &ds, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        a.append_outcomes(&[Outcome { id: "s:1".into(), session: "s".into(), seq: 1, ts: 1, world: "w".into(), features: vec!["f".into()], task_type: None, agent: "ag".into(), reward: 1.0 }]).unwrap();
        let r = a.sync_docs_folder(&folder, "mA", "kb").unwrap();
        assert_eq!((r.exported, r.skipped_files), (1, 1), "{r:?}");
        let rj = a.sync_folder(&folder, "mA", None, None).unwrap();
        assert_eq!((rj.exported, rj.files_skipped), (1, 2), "{rj:?}");
        // a second machine still converges through the folder despite the bad entries
        b.sync_docs_folder(&folder, "mB", "kb").unwrap();
        b.sync_folder(&folder, "mB", None, None).unwrap();
        assert_eq!(b.load_docs("kb").unwrap().0.len(), 1);
        assert_eq!(b.outcomes().unwrap().len(), 1);
        for q in [&folder, &p, &p2] {
            let _ = fs::remove_dir_all(q);
        }
    }
}

#[cfg(test)]
mod gc_sync_tests {
    use super::*;
    /// B30: a doc dropped by `gc_docs` was re-imported by the next `sync_docs_folder` from any peer that still held it (gc and sync fought forever).
    #[test]
    fn gc_dropped_docs_are_not_resurrected_by_sync() {
        let base = std::env::temp_dir().join(format!("nm_gcsync_{}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let (folder, pa, pb) = (base.join("f"), base.join("a"), base.join("b"));
        let (a, b) = (MeshStore::open(&pa).unwrap(), MeshStore::open(&pb).unwrap());
        let dim = 4;
        let docs: Vec<DocRec> = ["x", "y", "z"].iter().map(|i| DocRec { id: i.to_string(), text: format!("t{i}"), tag: "t".into() }).collect();
        let v = [1.0f32, 0., 0., 0., 1.0, 0., 0., 0., 0., 1.0, 0., 0.]; // x == y, z distinct
        a.put_docs("kb", dim, &docs, &v).unwrap();
        a.sync_docs_folder(&folder, "mA", "kb").unwrap();
        b.sync_docs_folder(&folder, "mB", "kb").unwrap();
        assert_eq!(b.load_docs("kb").unwrap().0.len(), 3);
        assert_eq!(a.gc_docs("kb", 0.99).unwrap(), (3, 2));
        for _ in 0..3 {
            a.sync_docs_folder(&folder, "mA", "kb").unwrap();
            b.sync_docs_folder(&folder, "mB", "kb").unwrap();
        }
        let ids: Vec<String> = a.load_docs("kb").unwrap().0.into_iter().map(|d| d.id).collect();
        assert_eq!(ids, vec!["x".to_string(), "z".to_string()], "the gc-dropped near-duplicate must stay dropped on A");
        assert_eq!(b.load_docs("kb").unwrap().0.len(), 3, "B never gc'd: it keeps its copy (deletions are local, not propagated)");
        let _ = fs::remove_dir_all(&base);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hebbian::{HebbianConfig, HebbianRouter};

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("nm_store_test_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }
    fn oc(s: &str, i: u64, agent: &str, r: f64) -> Outcome {
        Outcome { id: format!("{s}:{i}"), session: s.into(), seq: i, ts: i, world: "w".into(), features: vec!["sql".into(), "db".into()], task_type: None, agent: agent.into(), reward: r }
    }

    #[test]
    fn journal_roundtrip_dedupe_and_torn_line() {
        let p = tmp("j");
        let s = MeshStore::open(&p).unwrap();
        s.append_outcomes(&[oc("a", 0, "x", 1.0), oc("a", 1, "y", 0.0)]).unwrap();
        s.append_outcome(&oc("a", 0, "x", 1.0)).unwrap(); // duplicate id
        let mut f = OpenOptions::new().append(true).open(p.join("router/journal.jsonl")).unwrap();
        f.write_all(b"{\"id\":\"torn").unwrap(); // crash mid-append
        let s2 = MeshStore::open(&p).unwrap();
        let v = s2.outcomes().unwrap();
        assert_eq!(v.len(), 2);
        assert!(!p.join(".lock").exists());
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn warm_replay_trains_router() {
        let p = tmp("w");
        let s = MeshStore::open(&p).unwrap();
        let os: Vec<Outcome> = (0..60).map(|i| oc("a", i, if i % 2 == 0 { "good" } else { "bad" }, if i % 2 == 0 { 1.0 } else { 0.0 })).collect();
        s.append_outcomes(&os).unwrap();
        let mut r = HebbianRouter::new(2, HebbianConfig::default());
        let idx = |n: &str| match n { "good" => Some(0), "bad" => Some(1), _ => None };
        assert_eq!(s.warm_replay(&mut r, &idx, None, None).unwrap(), 60);
        let t = Task { id: 0, task_type: None, features: vec!["sql".into(), "db".into()] };
        let sc = r.scores(&t);
        assert!(sc[0] > sc[1], "{sc:?}");
        assert_eq!(s.warm_replay(&mut HebbianRouter::new(2, HebbianConfig::default()), &idx, Some("other"), None).unwrap(), 0);
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn discounted_replay_keeps_own_and_every_kth_foreign() {
        let p = tmp("disc");
        let s = MeshStore::open(&p).unwrap();
        let mut os: Vec<Outcome> = (0..10).map(|i| oc("own", i, "good", 1.0)).collect();
        os.extend((0..20).map(|i| oc("other", 100 + i, "bad", 0.0)));
        s.append_outcomes(&os).unwrap();
        let idx = |n: &str| match n { "good" => Some(0), "bad" => Some(1), _ => None };
        let own = |o: &Outcome| o.session == "own";
        let mut r = HebbianRouter::new(2, HebbianConfig::default());
        assert_eq!(s.warm_replay_discounted(&mut r, &idx, None, &own, 4).unwrap(), 10 + 5);
        assert_eq!(s.warm_replay_discounted(&mut HebbianRouter::new(2, HebbianConfig::default()), &idx, None, &own, 1).unwrap(), 30);
        assert_eq!(s.warm_replay_discounted(&mut HebbianRouter::new(2, HebbianConfig::default()), &idx, None, &own, 0).unwrap(), 30);
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn merge_is_union_and_idempotent() {
        let (pa, pb) = (tmp("ma"), tmp("mb"));
        let (a, b) = (MeshStore::open(&pa).unwrap(), MeshStore::open(&pb).unwrap());
        a.append_outcomes(&[oc("a", 0, "x", 1.0), oc("a", 1, "x", 1.0)]).unwrap();
        b.append_outcomes(&[oc("b", 0, "y", 1.0), oc("a", 1, "x", 1.0)]).unwrap();
        let r = a.merge_from(&b).unwrap();
        assert_eq!(r.outcomes_added, 1);
        assert_eq!(a.merge_from(&b).unwrap(), MergeReport::default());
        assert_eq!(a.outcomes().unwrap().len(), 3);
        let _ = fs::remove_dir_all(&pa);
        let _ = fs::remove_dir_all(&pb);
    }

    #[test]
    fn gc_docs_drops_near_duplicates_atomically() {
        let p = tmp("gcdocs");
        let s = MeshStore::open(&p).unwrap();
        let mk = |i: &str| DocRec { id: i.into(), text: format!("doc {i}"), tag: "t".into() };
        let docs = vec![mk("a"), mk("b"), mk("c"), mk("d")];
        // a, c are near-identical (cos ~ 0.9999), b orthogonal, d exact copy of b
        let v = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.01, 0.0, 0.0, 1.0, 0.0];
        s.put_docs("ns", 3, &docs, &v[..12]).unwrap();
        assert_eq!(s.gc_docs("ns", 1.0).unwrap(), (4, 3)); // exact-vector duplicates only: b/d is NOT exact here (d = 0,1,0 vs b = 0,1,0 -> equal)
        let (d1, _, _) = s.load_docs("ns").unwrap();
        assert_eq!(d1.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), vec!["a", "b", "c"]);
        assert_eq!(s.gc_docs("ns", 0.99).unwrap(), (3, 2)); // c merged into a
        let (d2, v2, dim) = s.load_docs("ns").unwrap();
        assert_eq!((d2.len(), v2.len(), dim), (2, 6, 3));
        assert_eq!(s.gc_docs("ns", 0.99).unwrap(), (2, 2)); // idempotent
        assert!(!p.join("knowledge/ns/docs.jsonl.tmp").exists());
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn folder_sync_is_union_idempotent_and_torn_tolerant() {
        let folder = tmp("fold");
        let (pa, pb, pc) = (tmp("fa"), tmp("fb"), tmp("fc"));
        let (a, b, c) = (MeshStore::open(&pa).unwrap(), MeshStore::open(&pb).unwrap(), MeshStore::open(&pc).unwrap());
        a.append_outcomes(&[oc("a", 0, "x", 1.0), oc("a", 1, "x", 0.0)]).unwrap();
        b.append_outcomes(&[oc("b", 0, "y", 1.0)]).unwrap();
        c.append_outcomes(&[oc("c", 0, "z", 1.0), oc("c", 1, "z", 1.0), oc("c", 2, "z", 0.0)]).unwrap();
        let ra = a.sync_folder(&folder, "mA", None, None).unwrap();
        assert_eq!((ra.imported, ra.exported), (0, 2));
        b.sync_folder(&folder, "mB", None, None).unwrap();
        c.sync_folder(&folder, "mC", None, None).unwrap();
        // simulate a half-written foreign file (torn last line) plus a leftover temp file
        let mut f = OpenOptions::new().append(true).open(folder.join("mB.journal.jsonl")).unwrap();
        f.write_all(b"{\"id\":\"torn").unwrap();
        fs::write(folder.join("mZ.journal.jsonl.tmp"), b"garbage").unwrap();
        for (s, m) in [(&a, "mA"), (&b, "mB")] {
            s.sync_folder(&folder, m, None, None).unwrap();
        }
        a.sync_folder(&folder, "mA", None, None).unwrap();
        for s in [&a, &b, &c] {
            s.sync_folder(&folder, if std::ptr::eq(s, &a) { "mA" } else if std::ptr::eq(s, &b) { "mB" } else { "mC" }, None, None).unwrap();
            assert_eq!(s.outcomes().unwrap().len(), 6, "every machine holds the union");
        }
        let again = a.sync_folder(&folder, "mA", None, None).unwrap();
        assert_eq!(again.imported, 0); // idempotent
        // namespace filter: a fourth machine importing only world "other" gets nothing
        let pd = tmp("fd");
        let d = MeshStore::open(&pd).unwrap();
        assert_eq!(d.sync_folder(&folder, "mD", Some(&["other"]), None).unwrap().imported, 0);
        assert!(d.sync_folder(&folder, "bad id", None, None).is_err());
        // shared GC cap: newest 4 of the 6 in world "w" on every machine; converged sets are equal
        let capped_a = a.sync_folder(&folder, "mA", None, Some(4)).unwrap();
        assert_eq!(a.outcomes().unwrap().len(), 4);
        assert!(capped_a.exported <= 4);
        b.sync_folder(&folder, "mB", None, Some(4)).unwrap();
        let ids = |s: &MeshStore| s.outcomes().unwrap().into_iter().map(|o| o.id).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b));
        for p in [&folder, &pa, &pb, &pc, &pd] {
            let _ = fs::remove_dir_all(p);
        }
    }

    #[test]
    fn docs_roundtrip_and_dedupe() {
        let p = tmp("d");
        let s = MeshStore::open(&p).unwrap();
        let docs = vec![DocRec { id: "1".into(), text: "alpha".into(), tag: "t".into() }, DocRec { id: "2".into(), text: "beta".into(), tag: "t".into() }];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        assert_eq!(s.put_docs("ns", 2, &docs, &v).unwrap(), 2);
        assert_eq!(s.put_docs("ns", 2, &docs, &v).unwrap(), 0);
        let (d, vv, dim) = s.load_docs("ns").unwrap();
        assert_eq!((d.len(), vv.len(), dim), (2, 4, 2));
        assert!(s.put_docs("ns", 3, &docs, &[0.0; 6]).is_err());
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn schema_mismatch_is_refused() {
        let p = tmp("s");
        let _ = MeshStore::open(&p).unwrap();
        fs::write(p.join("MANIFEST.json"), "{\"schema_version\": 99}").unwrap();
        assert!(matches!(MeshStore::open(&p), Err(StoreError::Schema(99))));
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn ann_dump_reload_matches_rebuild() {
        let p = tmp("ann");
        let s = MeshStore::open(&p).unwrap();
        let n = 300;
        let dim = 8;
        let mut vecs = vec![0f32; n * dim];
        let mut x = 12345u64;
        for v in vecs.iter_mut() {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = ((x >> 33) as f32 / 2147483648.0) - 0.5;
        }
        let docs: Vec<DocRec> = (0..n).map(|i| DocRec { id: format!("d{i}"), text: format!("t{i}"), tag: "x".into() }).collect();
        s.put_docs("ns", dim, &docs, &vecs).unwrap();
        let a = s.open_ann("ns").unwrap().unwrap();
        assert!(a.rebuilt);
        let b = s.open_ann("ns").unwrap().unwrap();
        assert!(!b.rebuilt, "second open must reload the dump");
        let q = &vecs[0..dim];
        assert_eq!(a.search(q, 5, 64).iter().map(|x| x.0).collect::<Vec<_>>(), b.search(q, 5, 64).iter().map(|x| x.0).collect::<Vec<_>>());
        assert_eq!(b.search(q, 1, 64)[0].0, 0);
        // growing the corpus invalidates the dump
        let more = vec![DocRec { id: "new".into(), text: "t".into(), tag: "x".into() }];
        s.put_docs("ns", dim, &more, &vec![0.1; dim]).unwrap();
        assert!(s.open_ann("ns").unwrap().unwrap().rebuilt);
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn compaction_keeps_newest_per_world() {
        let p = tmp("gc");
        let s = MeshStore::open(&p).unwrap();
        let mut os: Vec<Outcome> = (0..10).map(|i| oc("a", i, "x", 1.0)).collect();
        let mut o2 = oc("b", 100, "y", 0.0);
        o2.world = "other".into();
        os.push(o2);
        s.append_outcomes(&os).unwrap();
        assert_eq!(s.compact_journal(3).unwrap(), (11, 4));
        let v = s.outcomes().unwrap();
        assert_eq!(v.iter().filter(|o| o.world == "w").map(|o| o.seq).collect::<Vec<_>>(), vec![7, 8, 9]);
        assert_eq!(s.compact_journal(3).unwrap(), (4, 4));
        let _ = fs::remove_dir_all(&p);
    }


    // ---- Team B (B18): the persisted ANN must not be reused when the corpus CONTENT changed but its row count did not
    #[test]
    fn persisted_ann_is_rebuilt_when_content_changes_with_same_row_count() {
        let p = tmp("ann_stale");
        let s = MeshStore::open(&p).unwrap();
        let dim = 8;
        let n = 30usize;
        let mut st = 12345u64;
        let mut v: Vec<f32> = (0..n * dim)
            .map(|_| {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((st >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
            })
            .collect();
        for r in 0..n {
            let nn = v[r * dim..(r + 1) * dim].iter().map(|x| x * x).sum::<f32>().sqrt();
            v[r * dim..(r + 1) * dim].iter_mut().for_each(|x| *x /= nn);
        }
        let dup: Vec<f32> = v[0..dim].to_vec(); // doc n-1 becomes an exact duplicate of doc 0
        v[(n - 1) * dim..n * dim].copy_from_slice(&dup);
        let docs: Vec<DocRec> = (0..n).map(|i| DocRec { id: format!("d{i}"), text: format!("t{i}"), tag: "x".into() }).collect();
        s.put_docs("ns", dim, &docs, &v).unwrap();
        assert!(s.open_ann("ns").unwrap().unwrap().rebuilt); // graph dumped for n = 30
        assert_eq!(s.gc_docs("ns", 0.99999).unwrap(), (30, 29)); // the duplicate is dropped -> n = 29
        let mut newv = vec![0.0f32; dim];
        newv[3] = 1.0;
        s.put_docs("ns", dim, &[DocRec { id: "fresh".into(), text: "fresh".into(), tag: "x".into() }], &newv).unwrap(); // n = 30 again, different content
        let a = s.open_ann("ns").unwrap().unwrap();
        let top = a.search(&newv, 1, 64);
        assert_eq!(top[0].0, 29, "the freshly added doc must be found at its row (stale graph served: {top:?})");
        let _ = fs::remove_dir_all(&p);
    }


    #[test]
    fn merge_skips_dim_mismatched_namespaces_and_counts_conflicts_instead_of_aborting() {
        // B24: merge_from used to return Err(Corrupt) at the first namespace whose dim differs, leaving the merge half-done
        // (outcomes already appended, later namespaces never visited). Now the namespace is skipped and reported.
        let (pa, pb) = (tmp("mg_dim_a"), tmp("mg_dim_b"));
        let (a, b) = (MeshStore::open(&pa).unwrap(), MeshStore::open(&pb).unwrap());
        let d = |id: &str, t: &str| DocRec { id: id.into(), text: t.into(), tag: "t".into() };
        a.put_docs("x", 4, &[d("1", "alpha"), d("2", "beta")], &[0.5; 8]).unwrap();
        b.put_docs("a", 3, &[d("p", "p")], &[0.5; 3]).unwrap(); // sorts before "x"
        b.put_docs("x", 8, &[d("3", "gamma")], &[0.5; 8]).unwrap(); // dim clash with a's "x"
        b.put_docs("z", 4, &[d("q", "q"), d("q2", "q2")], &[0.5; 8]).unwrap(); // sorts after "x": used to be unreachable
        let c = tmp("mg_dim_c");
        let c = MeshStore::open(&c).unwrap();
        c.put_docs("z", 4, &[d("q", "DIFFERENT")], &[0.5; 4]).unwrap();
        let r = a.merge_from(&b).unwrap();
        assert_eq!(r.ns_skipped, 1, "the dim-clashing namespace is skipped and reported");
        assert_eq!(r.docs_added, 3, "a/p, z/q, z/q2 merged despite the clash in between");
        assert_eq!(a.load_docs("x").unwrap().0.len(), 2, "x untouched");
        let r2 = c.merge_from(&b).unwrap();
        assert_eq!(r2.doc_conflicts, 1, "same id, different text is counted (local kept)");
        assert_eq!(c.load_docs("z").unwrap().0.iter().find(|d| d.id == "q").unwrap().text, "DIFFERENT");
        let _ = fs::remove_dir_all(&pa);
        let _ = fs::remove_dir_all(&pb);
    }

    #[test]
    fn two_waiters_judging_the_same_lock_stale_do_not_both_acquire() {
        let _serial = STALE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        // B22: waiter B judges the lock stale; before B acts, waiter A breaks it and takes a FRESH lock. B must not delete A's live lock.
        let p = tmp("two_waiters");
        let s = MeshStore::open(&p).unwrap();
        let lp = p.join(".lock");
        fs::write(&lp, "dead holder\n").unwrap();
        let f = OpenOptions::new().write(true).open(&lp).unwrap();
        f.set_modified(SystemTime::now() - Duration::from_secs(60)).unwrap();
        drop(f);
        *STALE_HOOK.lock().unwrap() = Some((
            lp.clone(),
            Box::new(|lp: &Path| {
                // waiter A wins the race: removes the stale lock and creates its own fresh one
                let _ = fs::remove_file(lp);
                fs::write(lp, "waiter A\n").unwrap();
            }),
        ));
        let r = s.lock();
        let still_a = fs::read_to_string(&lp).map(|t| t.starts_with("waiter A")).unwrap_or(false);
        assert!(r.is_err(), "B acquired the lock while A's fresh lock was live (mutual exclusion broken)");
        assert!(still_a, "A's live lock file must not have been deleted by B");
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn long_gc_keeps_its_lock_alive_so_concurrent_puts_are_not_lost() {
        let _serial = STALE_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        use std::sync::atomic::Ordering;
        STALE_OVERRIDE_MS.store(150, Ordering::Relaxed);
        let p = tmp("gc_lock");
        let s = MeshStore::open(&p).unwrap();
        let dim = 256;
        let n = 6000usize;
        let mut st = 99u64;
        let v: Vec<f32> = (0..n * dim)
            .map(|_| {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((st >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
            })
            .collect();
        let mut v = v;
        let head: Vec<f32> = v[..50 * dim].to_vec();
        v[(n - 50) * dim..].copy_from_slice(&head); // 50 exact duplicates so gc really rewrites the corpus
        let docs: Vec<DocRec> = (0..n).map(|i| DocRec { id: format!("d{i}"), text: format!("t{i}"), tag: "x".into() }).collect();
        s.put_docs("ns", dim, &docs, &v).unwrap();
        let p2 = p.clone();
        let t0 = Instant::now();
        let gc = std::thread::spawn(move || MeshStore::open(&p2).unwrap().gc_docs("ns", 0.99999).unwrap());
        std::thread::sleep(Duration::from_millis(60));
        // a second writer arrives while gc runs (gc takes far longer than the 150 ms stale threshold used here)
        let s2 = MeshStore::open(&p).unwrap();
        let extra = vec![DocRec { id: "late".into(), text: "late".into(), tag: "x".into() }];
        let mut added = 0;
        for _ in 0..200 {
            match s2.put_docs("ns", dim, &extra, &vec![0.5; dim]) {
                Ok(k) => {
                    added = k;
                    break;
                }
                Err(StoreError::Locked(_)) => std::thread::sleep(Duration::from_millis(20)),
                Err(e) => panic!("{e}"),
            }
        }
        gc.join().unwrap();
        STALE_OVERRIDE_MS.store(0, Ordering::Relaxed);
        let gc_ms = t0.elapsed().as_millis();
        let (d, vv, dm) = s.load_docs("ns").unwrap();
        assert!(gc_ms > 300, "test premise: gc must outlast the stale threshold (took {gc_ms} ms)");
        assert_eq!(added, 1);
        assert!(d.iter().any(|x| x.id == "late"), "the doc added while gc ran must survive");
        assert_eq!(vv.len(), d.len() * dm);
        let _ = fs::remove_dir_all(&p);
    }

    // ---- Team B (B1): durability / alignment regressions
    #[test]
    fn append_after_torn_tail_does_not_lose_the_new_record() {
        let p = tmp("torn_then_append");
        let s = MeshStore::open(&p).unwrap();
        s.append_outcomes(&[oc("a", 0, "x", 1.0)]).unwrap();
        let mut f = OpenOptions::new().append(true).open(p.join("router/journal.jsonl")).unwrap();
        f.write_all(b"{\"id\":\"torn").unwrap(); // crash mid-append, no trailing newline
        drop(f);
        s.append_outcomes(&[oc("a", 1, "x", 1.0)]).unwrap();
        let ids: Vec<String> = s.outcomes().unwrap().into_iter().map(|o| o.id).collect();
        assert_eq!(ids, vec!["a:0".to_string(), "a:1".to_string()], "record appended after a torn tail must survive");
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn orphan_vector_row_does_not_misalign_later_docs() {
        let p = tmp("orphan_vec");
        let s = MeshStore::open(&p).unwrap();
        let dim = 4;
        let d0 = vec![DocRec { id: "d0".into(), text: "t0".into(), tag: "x".into() }];
        s.put_docs("ns", dim, &d0, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        // crash between the vector write and the doc write: one extra vector row
        let mut f = OpenOptions::new().append(true).open(p.join("knowledge/ns/vecs.f32")).unwrap();
        for x in [9.0f32, 9.0, 9.0, 9.0] {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
        drop(f);
        let d1 = vec![DocRec { id: "d1".into(), text: "t1".into(), tag: "x".into() }];
        s.put_docs("ns", dim, &d1, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let (docs, v, dm) = s.load_docs("ns").unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(&v[dm..2 * dm], &[0.0, 1.0, 0.0, 0.0], "doc d1 must carry its own vector");
        let _ = fs::remove_dir_all(&p);
    }

    #[test]
    fn torn_multibyte_tail_does_not_make_journal_unreadable() {
        let p = tmp("torn_utf8");
        let s = MeshStore::open(&p).unwrap();
        s.append_outcomes(&[oc("a", 0, "x", 1.0)]).unwrap();
        let mut f = OpenOptions::new().append(true).open(p.join("router/journal.jsonl")).unwrap();
        f.write_all(b"{\"id\":\"a:1\",\"session\":\"caf\xc3").unwrap(); // cut inside the 2-byte char
        drop(f);
        assert_eq!(s.outcomes().unwrap().len(), 1, "journal must stay readable");
        let folder = tmp("torn_utf8_folder");
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join("other.journal.jsonl"), b"{\"id\":\"o:0\",\"session\":\"o\",\"seq\":0,\"ts\":0,\"world\":\"w\",\"features\":[],\"task_type\":null,\"agent\":\"x\",\"reward\":1.0}\n{\"id\":\"o:1\",\"se\xe2\x82").unwrap();
        let r = s.sync_folder(&folder, "me", None, None).unwrap();
        assert_eq!(r.imported, 1);
        let _ = fs::remove_dir_all(&p);
        let _ = fs::remove_dir_all(&folder);
    }
}

#[cfg(test)]
mod hardening_b31_tests {
    use super::*;
    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("nm_b31_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }
    fn doc(text: &str) -> Vec<DocRec> {
        vec![DocRec { id: "same".into(), text: text.into(), tag: "x".into() }]
    }
    fn outcome(id: &str) -> Outcome {
        Outcome { id: id.into(), session: "s".into(), seq: 1, ts: 1, world: "w".into(), features: vec!["f".into()], task_type: None, agent: "ag".into(), reward: 1.0 }
    }

    /// B31: machine ids differing only by case map to the same file on Windows / macOS and overwrote each other's shard. Now rejected up front, nothing touched.
    #[test]
    fn machine_ids_differing_only_by_case_are_rejected() {
        let (folder, p1, p2) = (tmp("cf"), tmp("cs1"), tmp("cs2"));
        let (a, b) = (MeshStore::open(&p1).unwrap(), MeshStore::open(&p2).unwrap());
        a.append_outcomes(&[outcome("a:1")]).unwrap();
        a.put_docs("kb", 4, &doc("alpha"), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        a.sync_folder(&folder, "Alice", None, None).unwrap();
        a.sync_docs_folder(&folder, "Alice", "kb").unwrap();
        let before = fs::read(folder.join("Alice.journal.jsonl")).unwrap();
        b.append_outcomes(&[outcome("b:1")]).unwrap();
        assert!(b.sync_folder(&folder, "alice", None, None).is_err(), "case-colliding journal machine id must be rejected");
        assert!(b.sync_docs_folder(&folder, "alice", "kb").is_err(), "case-colliding docs machine id must be rejected");
        assert_eq!(fs::read(folder.join("Alice.journal.jsonl")).unwrap(), before, "the other machine's shard is untouched");
        // the legitimate owner keeps working, and a clearly different id is fine
        a.sync_folder(&folder, "Alice", None, None).unwrap();
        b.sync_folder(&folder, "bob", None, None).unwrap();
        for q in [&folder, &p1, &p2] {
            let _ = fs::remove_dir_all(q);
        }
    }

    /// B32: two foreign shards disagreeing on an id's text were resolved silently by sorted-file order; now counted as a conflict (first still wins).
    #[test]
    fn foreign_shards_disagreeing_on_text_are_counted() {
        let (folder, p1, p2, p3) = (tmp("df"), tmp("ds1"), tmp("ds2"), tmp("ds3"));
        let (a, b, c) = (MeshStore::open(&p1).unwrap(), MeshStore::open(&p2).unwrap(), MeshStore::open(&p3).unwrap());
        a.put_docs("kb", 4, &doc("version one"), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        b.put_docs("kb", 4, &doc("version TWO"), &[1.0, 0.0, 0.0, 0.0]).unwrap();
        a.sync_docs_folder(&folder, "mA", "kb").unwrap();
        // b exports its own conflicting copy by hand (b's own sync would import a's, see conflict and keep local, exporting nothing since foreign holds the id)
        let line = fs::read_to_string(folder.join("mA.kb.docs.jsonl")).unwrap().replace("version one", "version TWO");
        fs::write(folder.join("mB.kb.docs.jsonl"), line).unwrap();
        let r = c.sync_docs_folder(&folder, "mC", "kb").unwrap();
        assert_eq!(r.imported, 1, "{r:?}");
        assert_eq!(r.conflicts, 1, "disagreement between mA and mB must be counted: {r:?}");
        assert_eq!(c.load_docs("kb").unwrap().0[0].text, "version one", "first in sorted file order still wins");
        let _ = b;
        for q in [&folder, &p1, &p2, &p3] {
            let _ = fs::remove_dir_all(q);
        }
    }

    /// B33: durable replace leaves the right bytes and no temp file (the fsync itself is not observable from a test).
    #[test]
    fn replace_durable_writes_and_leaves_no_tmp() {
        let d = tmp("dur");
        fs::create_dir_all(&d).unwrap();
        let (t, f) = (d.join("x.tmp"), d.join("x"));
        fs::write(&f, b"old").unwrap();
        replace_durable(&t, &f, b"new contents").unwrap();
        assert_eq!(fs::read(&f).unwrap(), b"new contents");
        assert!(!t.exists());
        let _ = fs::remove_dir_all(&d);
    }
}
