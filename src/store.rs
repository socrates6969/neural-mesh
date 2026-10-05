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
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::router::{Router, Task};

pub const SCHEMA_VERSION: u32 = 1;
const LOCK_TIMEOUT_MS: u64 = 3000;
const LOCK_STALE_MS: u64 = 10_000;

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
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
pub struct MergeReport {
    pub outcomes_added: usize,
    pub lessons_added: usize,
    pub docs_added: usize,
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
                    if let Ok(m) = fs::metadata(&p).and_then(|m| m.modified()) {
                        if m.elapsed().map(|d| d > Duration::from_millis(LOCK_STALE_MS)).unwrap_or(false) {
                            let _ = fs::remove_file(&p);
                            continue;
                        }
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
        let mut f = OpenOptions::new().create(true).append(true).open(p)?;
        let mut buf = String::new();
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
        for l in BufReader::new(File::open(p)?).lines() {
            let l = l?;
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

    /// GC / compaction (R35): rewrite the journal keeping, per `world`, only the newest `keep` outcomes (replay order), deduplicated.
    /// Atomic (temp file + rename) under the lock. Returns (before, after). Lessons and corpus are never touched.
    pub fn compact_journal(&self, keep: usize) -> R<(usize, usize)> {
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
        let _g = self.lock()?;
        let p = self.root.join("router/journal.jsonl");
        let tmp = self.root.join("router/journal.jsonl.tmp");
        let mut buf = String::new();
        for o in &kept {
            buf.push_str(&serde_json::to_string(o).unwrap());
            buf.push('\n');
        }
        fs::write(&tmp, buf)?;
        fs::rename(&tmp, &p)?;
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
        let have: HashSet<String> = self.read_jsonl::<DocRec>(&format!("knowledge/{ns}/docs.jsonl"))?.into_iter().map(|x| x.id).collect();
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
        if let Ok(b) = fs::read(&stamp) {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&b) {
                if v["n"].as_u64() == Some(docs.len() as u64) && v["dim"].as_u64() == Some(dim as u64) {
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
        fs::write(&stamp, serde_json::to_vec(&serde_json::json!({"n": docs.len(), "dim": dim, "base": base})).unwrap())?;
        Ok(Some(PersistedAnn { h, n: docs.len(), dim, rebuilt: true }))
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
}
