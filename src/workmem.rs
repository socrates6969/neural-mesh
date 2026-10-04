//! R3: working memory for routing, a recurring-context workload generator, and a
//! persistent on-disk shared memory store (std only).
//!
//! * [`WorkMemRouter`]: bounded buffer of `capacity` context slots. Each slot is
//!   a private copy of the inner router that only learns from tasks seen in that
//!   context (LRU eviction when full; optional chunking of context labels so that
//!   variants of the same family share one slot).
//! * [`ctxworld`]: seeded generator of task streams whose best agent depends on
//!   a (recurring, unique or irrelevant) context.
//! * [`store`]: append-only log + snapshot + flat/HNSW index, crash safe.

use std::collections::HashMap;

use crate::router::{Rng, Router, Task};

// ---------------------------------------------------------------------------
// Working-memory router
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct WorkMemConfig {
    /// Number of context slots. 0 = disabled (pure pass-through to the global
    /// router); `usize::MAX` = unbounded.
    pub capacity: usize,
    /// Feature prefix that marks the context label (e.g. "ctx:").
    pub ctx_prefix: String,
    /// A slot is consulted only after it has seen this many outcomes.
    pub min_n: u64,
    /// New slots start as a copy of the global router instead of the cold prototype.
    pub init_from_global: bool,
    /// Chunking: the slot key is the label up to the first '/', so variants of one
    /// family ("ctx:f2/0", "ctx:f2/1") share a slot.
    pub chunk: bool,
    /// When a task carries no context marker, assume the previous context persists.
    pub carry: bool,
}

impl Default for WorkMemConfig {
    fn default() -> Self {
        WorkMemConfig { capacity: 0, ctx_prefix: "ctx:".into(), min_n: 3, init_from_global: false, chunk: false, carry: true }
    }
}

#[derive(Clone)]
struct Slot<R> {
    key: String,
    inner: R,
    last_used: u64,
    n: u64,
}

pub struct WorkMemRouter<R: Router + Clone> {
    pub cfg: WorkMemConfig,
    proto: R,
    global: R,
    slots: Vec<Slot<R>>,
    cur: Option<String>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl<R: Router + Clone> WorkMemRouter<R> {
    pub fn new(inner: R, cfg: WorkMemConfig) -> Self {
        WorkMemRouter { cfg, proto: inner.clone(), global: inner, slots: vec![], cur: None, clock: 0, hits: 0, misses: 0, evictions: 0 }
    }
    pub fn slots_used(&self) -> usize {
        self.slots.len()
    }
    pub fn global(&self) -> &R {
        &self.global
    }

    fn key_of(&self, label: &str) -> String {
        if self.cfg.chunk { label.split('/').next().unwrap_or(label).to_string() } else { label.to_string() }
    }

    /// Resolve (and remember) the context of `task`.
    fn resolve(&mut self, task: &Task) -> Option<String> {
        let found = task.features.iter().find(|f| f.starts_with(&self.cfg.ctx_prefix)).map(|f| self.key_of(f));
        if let Some(k) = found {
            self.cur = Some(k.clone());
            Some(k)
        } else if self.cfg.carry {
            self.cur.clone()
        } else {
            None
        }
    }

    fn slot_index(&mut self, key: &str) -> usize {
        self.clock += 1;
        if let Some(i) = self.slots.iter().position(|s| s.key == key) {
            self.slots[i].last_used = self.clock;
            return i;
        }
        let inner = if self.cfg.init_from_global { self.global.clone() } else { self.proto.clone() };
        let slot = Slot { key: key.to_string(), inner, last_used: self.clock, n: 0 };
        if self.slots.len() >= self.cfg.capacity {
            // evict least recently used
            let (vi, _) = self.slots.iter().enumerate().min_by_key(|(_, s)| s.last_used).unwrap();
            self.slots[vi] = slot;
            self.evictions += 1;
            vi
        } else {
            self.slots.push(slot);
            self.slots.len() - 1
        }
    }
}

impl<R: Router + Clone> Router for WorkMemRouter<R> {
    fn name(&self) -> String {
        if self.cfg.capacity == 0 {
            format!("workmem(K=0,{})", self.global.name())
        } else if self.cfg.capacity == usize::MAX {
            format!("workmem(K=inf,{})", self.global.name())
        } else {
            format!("workmem(K={},{})", self.cfg.capacity, self.global.name())
        }
    }

    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        if self.cfg.capacity == 0 {
            return self.global.select(task, candidates, rng);
        }
        match self.resolve(task) {
            None => self.global.select(task, candidates, rng),
            Some(key) => {
                let i = self.slot_index(&key);
                if self.slots[i].n >= self.cfg.min_n {
                    self.hits += 1;
                    self.slots[i].inner.select(task, candidates, rng)
                } else {
                    self.misses += 1;
                    self.global.select(task, candidates, rng)
                }
            }
        }
    }

    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.global.update(task, agent, reward);
        if self.cfg.capacity == 0 {
            return;
        }
        if let Some(key) = self.resolve(task) {
            let i = self.slot_index(&key);
            self.slots[i].inner.update(task, agent, reward);
            self.slots[i].n += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Recurring-context task streams
// ---------------------------------------------------------------------------

pub mod ctxworld {
    use super::*;

    /// Precomputed stream: `p[t][a]` = success probability if agent `a` handles task `t`.
    #[derive(Clone)]
    pub struct Stream {
        pub tasks: Vec<Task>,
        pub p: Vec<Vec<f64>>,
        pub u: Vec<f64>,
        /// context of this task was already seen in an EARLIER block
        pub revisit: Vec<bool>,
        pub kwmap: HashMap<String, usize>,
        pub n_agents: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Recurrence {
        /// blocks draw their context from a fixed set of `families*variants` contexts
        Recurring,
        /// every block is a brand-new context (never repeats)
        Unique,
        /// context labels recur but the skill table is identical in all of them
        Irrelevant,
    }

    #[derive(Clone, Debug)]
    pub struct CtxCfg {
        pub topics: usize,
        pub agents: usize,
        pub families: usize,
        pub variants: usize,
        pub block_len: usize,
        pub n_tasks: usize,
        pub rec: Recurrence,
        pub marker_prob: f64,
        pub confound: f64,
        pub shift_frac: f64,
        pub traps: usize,
        pub map_err: f64,
    }

    impl Default for CtxCfg {
        fn default() -> Self {
            CtxCfg {
                topics: 8,
                agents: 6,
                families: 6,
                variants: 1,
                block_len: 40,
                n_tasks: 1500,
                rec: Recurrence::Recurring,
                marker_prob: 0.9,
                confound: 0.25,
                shift_frac: 0.7,
                traps: 2,
                map_err: 0.10,
            }
        }
    }

    fn table(rng: &mut Rng, topics: usize, agents: usize, base: Option<&Vec<Vec<f64>>>, shift: f64) -> Vec<Vec<f64>> {
        let mut p = vec![vec![0.0; agents]; topics];
        for t in 0..topics {
            if let Some(b) = base {
                if rng.f64() >= shift {
                    p[t] = b[t].clone();
                    continue;
                }
            }
            let best = rng.below(agents);
            let mut second = rng.below(agents);
            while second == best {
                second = rng.below(agents);
            }
            for a in 0..agents {
                p[t][a] = if a == best {
                    0.85
                } else if a == second {
                    0.5
                } else {
                    0.15 + 0.25 * rng.f64()
                };
            }
        }
        p
    }

    fn argmax(v: &[f64]) -> usize {
        let mut b = 0;
        for i in 1..v.len() {
            if v[i] > v[b] {
                b = i;
            }
        }
        b
    }

    pub fn generate(seed: u64, c: &CtxCfg) -> Stream {
        let mut rng = Rng::new(seed.wrapping_mul(104_729).wrapping_add(3));
        let base = table(&mut rng, c.topics, c.agents, None, 1.0);
        let n_ctx = c.families * c.variants;
        // family tables derive from base (shift_frac of topics re-drawn)
        let fam: Vec<Vec<Vec<f64>>> = (0..c.families)
            .map(|_| match c.rec {
                Recurrence::Irrelevant => base.clone(),
                _ => table(&mut rng, c.topics, c.agents, Some(&base), c.shift_frac),
            })
            .collect();
        // keyword map = argmax of the base table, 3 keywords per topic; traps point to a bad agent
        let mut order: Vec<usize> = (0..c.topics).collect();
        for i in (1..c.topics).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let mut kwmap = HashMap::new();
        for t in 0..c.topics {
            let best = argmax(&base[t]);
            let trap = order.iter().take(c.traps).any(|&x| x == t);
            for j in 0..3 {
                let a = if trap {
                    loop {
                        let a = rng.below(c.agents);
                        if a != best {
                            break a;
                        }
                    }
                } else if rng.f64() < c.map_err {
                    rng.below(c.agents)
                } else {
                    best
                };
                kwmap.insert(format!("t{t}k{j}"), a);
            }
        }
        let mut s = Stream { tasks: vec![], p: vec![], u: vec![], revisit: vec![], kwmap, n_agents: c.agents };
        let mut seen: Vec<bool> = vec![false; n_ctx.max(1)];
        let mut cur_ctx = usize::MAX;
        let mut unique_table: Vec<Vec<f64>> = base.clone();
        let mut unique_id = 0usize;
        let mut cur_seen_before = false;
        for i in 0..c.n_tasks {
            if i % c.block_len == 0 {
                match c.rec {
                    Recurrence::Unique => {
                        unique_table = table(&mut rng, c.topics, c.agents, Some(&base), c.shift_frac);
                        unique_id += 1;
                        cur_seen_before = false;
                    }
                    _ => {
                        let mut nx = rng.below(n_ctx);
                        while n_ctx > 1 && nx == cur_ctx {
                            nx = rng.below(n_ctx);
                        }
                        cur_ctx = nx;
                        cur_seen_before = seen[nx];
                        seen[nx] = true;
                    }
                }
            }
            let (label, tbl): (String, &Vec<Vec<f64>>) = match c.rec {
                Recurrence::Unique => (format!("ctx:u{unique_id}/0"), &unique_table),
                _ => (format!("ctx:f{}/{}", cur_ctx / c.variants, cur_ctx % c.variants), &fam[cur_ctx / c.variants]),
            };
            let t = rng.below(c.topics);
            let mut f: Vec<String> = (0..3).filter(|_| rng.bernoulli(0.5)).map(|j| format!("t{t}k{j}")).collect();
            if f.is_empty() && rng.bernoulli(0.9) {
                f.push(format!("t{t}k{}", rng.below(3)));
            }
            if rng.bernoulli(c.confound) {
                let o = (t + 1 + rng.below(c.topics - 1)) % c.topics;
                f.push(format!("t{o}k{}", rng.below(3)));
            }
            if rng.bernoulli(c.marker_prob) {
                f.push(label);
            }
            s.tasks.push(Task { id: i as u64, task_type: None, features: f });
            s.p.push(tbl[t].clone());
            s.u.push(rng.f64());
            s.revisit.push(cur_seen_before);
        }
        s
    }

    /// Per-run outcome of driving a router over a stream.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct RunStats {
        pub mean: f64,
        pub first_third: f64,
        pub revisit_mean: f64,
        pub first_visit_mean: f64,
        pub oracle: f64,
    }

    pub fn drive<R: Router>(mut r: R, s: &Stream, seed: u64) -> RunStats {
        let cands: Vec<usize> = (0..s.n_agents).collect();
        let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
        let n = s.tasks.len();
        let (mut cum, mut c3, mut rv, mut nrv, mut fv, mut nfv, mut orc) = (0.0, 0.0, 0.0, 0usize, 0.0, 0usize, 0.0);
        for (i, task) in s.tasks.iter().enumerate() {
            let a = r.select(task, &cands, &mut rng);
            let p = s.p[i][a];
            cum += p;
            if i < n / 3 {
                c3 += p;
            }
            if s.revisit[i] {
                rv += p;
                nrv += 1;
            } else {
                fv += p;
                nfv += 1;
            }
            orc += s.p[i].iter().cloned().fold(0.0, f64::max);
            r.update(task, a, if s.u[i] < p { 1.0 } else { 0.0 });
        }
        RunStats {
            mean: cum / n as f64,
            first_third: c3 / (n / 3).max(1) as f64,
            revisit_mean: if nrv > 0 { rv / nrv as f64 } else { f64::NAN },
            first_visit_mean: if nfv > 0 { fv / nfv as f64 } else { f64::NAN },
            oracle: orc / n as f64,
        }
    }
}

// ---------------------------------------------------------------------------
// Persistent shared memory store
// ---------------------------------------------------------------------------

pub mod store {
    //! On-disk layout of a store directory:
    //!   `log.bin`    append-only records `[magic u32][payload_len u32][crc32 u32][payload]`
    //!                payload = `id u64, dim u32, meta_len u32, meta bytes, f32 * dim` (little endian)
    //!   `index.snap` optional checkpoint: normalised vectors + ids + the log offset it covers
    //!   `LOCK`       present while a writer is open (single writer, any number of readers)
    //!
    //! Crash safety: a record is valid only if its length and CRC check out. On
    //! open, the first invalid record ends the log; a writer truncates the file
    //! there (the torn tail), a reader just ignores it. `std` only: no mmap.

    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};

    use crate::memory::{dot, normalize_in_place, HnswIndex, HnswParams, MemoryIndex};

    const MAGIC: u32 = 0x314D_4E52; // "RNM1"
    const SNAP_MAGIC: u32 = 0x3150_4E53; // "SNP1"
    const HDR: usize = 12;

    pub fn crc32(data: &[u8]) -> u32 {
        const fn table() -> [u32; 256] {
            let mut t = [0u32; 256];
            let mut i = 0;
            while i < 256 {
                let mut c = i as u32;
                let mut k = 0;
                while k < 8 {
                    c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
                    k += 1;
                }
                t[i] = c;
                i += 1;
            }
            t
        }
        const T: [u32; 256] = table();
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c = T[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
        !c
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum IndexKind {
        Flat,
        Hnsw,
    }

    #[derive(Clone, Debug, Default)]
    pub struct OpenInfo {
        pub records: usize,
        pub from_snapshot: usize,
        pub replayed: usize,
        pub truncated_bytes: u64,
        pub ignored_tail_bytes: u64,
    }

    pub struct Store {
        dir: PathBuf,
        dim: usize,
        kind: IndexKind,
        writable: bool,
        file: File,
        /// bytes of the log consumed so far (end of the last valid record)
        offset: u64,
        ids: Vec<u64>,
        vecs: Vec<f32>, // normalised, row-major
        hnsw: Option<HnswIndex>,
        hnsw_params: HnswParams,
        pub info: OpenInfo,
    }

    fn encode(id: u64, meta: &[u8], v: &[f32]) -> Vec<u8> {
        let mut p = Vec::with_capacity(16 + meta.len() + v.len() * 4);
        p.extend_from_slice(&id.to_le_bytes());
        p.extend_from_slice(&(v.len() as u32).to_le_bytes());
        p.extend_from_slice(&(meta.len() as u32).to_le_bytes());
        p.extend_from_slice(meta);
        for x in v {
            p.extend_from_slice(&x.to_le_bytes());
        }
        let mut r = Vec::with_capacity(HDR + p.len());
        r.extend_from_slice(&MAGIC.to_le_bytes());
        r.extend_from_slice(&(p.len() as u32).to_le_bytes());
        r.extend_from_slice(&crc32(&p).to_le_bytes());
        r.extend_from_slice(&p);
        r
    }

    /// Parse one record at the start of `buf`: `(id, meta, vec, total_len)` or `None` if torn/corrupt.
    fn decode(buf: &[u8], dim: usize) -> Option<(u64, Vec<u8>, Vec<f32>, usize)> {
        if buf.len() < HDR {
            return None;
        }
        let magic = u32::from_le_bytes(buf[0..4].try_into().ok()?);
        if magic != MAGIC {
            return None;
        }
        let plen = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
        let crc = u32::from_le_bytes(buf[8..12].try_into().ok()?);
        if plen < 16 || buf.len() < HDR + plen {
            return None;
        }
        let p = &buf[HDR..HDR + plen];
        if crc32(p) != crc {
            return None;
        }
        let id = u64::from_le_bytes(p[0..8].try_into().ok()?);
        let d = u32::from_le_bytes(p[8..12].try_into().ok()?) as usize;
        let ml = u32::from_le_bytes(p[12..16].try_into().ok()?) as usize;
        if d != dim || 16 + ml + d * 4 != plen {
            return None;
        }
        let meta = p[16..16 + ml].to_vec();
        let v: Vec<f32> = p[16 + ml..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        Some((id, meta, v, HDR + plen))
    }

    impl Store {
        /// Open (or create, if `writable`) the store in `dir`.
        pub fn open(dir: &Path, dim: usize, kind: IndexKind, writable: bool) -> std::io::Result<Store> {
            Self::open_with(dir, dim, kind, writable, true, HnswParams::default())
        }

        /// `use_snapshot = false` forces a full log replay (cold start).
        pub fn open_with(
            dir: &Path,
            dim: usize,
            kind: IndexKind,
            writable: bool,
            use_snapshot: bool,
            hnsw_params: HnswParams,
        ) -> std::io::Result<Store> {
            if writable {
                fs::create_dir_all(dir)?;
                // single-writer lock
                OpenOptions::new().write(true).create_new(true).open(dir.join("LOCK")).map_err(|e| {
                    std::io::Error::new(e.kind(), format!("store {} is locked by another writer (or stale LOCK): {e}", dir.display()))
                })?;
            }
            let path = dir.join("log.bin");
            let file = if writable {
                OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?
            } else {
                OpenOptions::new().read(true).open(&path)?
            };
            let mut s = Store {
                dir: dir.to_path_buf(),
                dim,
                kind,
                writable,
                file,
                offset: 0,
                ids: vec![],
                vecs: vec![],
                hnsw: None,
                hnsw_params,
                info: OpenInfo::default(),
            };
            if kind == IndexKind::Hnsw {
                s.hnsw = Some(HnswIndex::new(dim, hnsw_params));
            }
            let mut snap_ids: Vec<u64> = vec![];
            let mut snap_vecs: Vec<f32> = vec![];
            if use_snapshot {
                if let Some((off, ids, vecs)) = s.read_snapshot() {
                    s.offset = off;
                    snap_ids = ids;
                    snap_vecs = vecs;
                }
            }
            s.info.from_snapshot = snap_ids.len();
            if !snap_ids.is_empty() {
                s.ids = snap_ids;
                s.vecs = snap_vecs;
                if let Some(h) = s.hnsw.as_mut() {
                    h.add_batch(&s.ids, &s.vecs);
                }
            }
            let before = s.ids.len();
            let torn = s.replay_tail()?;
            s.info.replayed = s.ids.len() - before;
            s.info.records = s.ids.len();
            if torn > 0 {
                if writable {
                    s.file.set_len(s.offset)?;
                    s.info.truncated_bytes = torn;
                } else {
                    s.info.ignored_tail_bytes = torn;
                }
            }
            Ok(s)
        }

        /// Apply all complete valid records from `self.offset` to EOF. Returns the number of
        /// unparseable trailing bytes (0 if the log ends cleanly).
        fn replay_tail(&mut self) -> std::io::Result<u64> {
            let len = self.file.metadata()?.len();
            if len <= self.offset {
                return Ok(0);
            }
            self.file.seek(SeekFrom::Start(self.offset))?;
            let mut buf = Vec::with_capacity((len - self.offset) as usize);
            (&self.file).take(len - self.offset).read_to_end(&mut buf)?;
            let mut pos = 0usize;
            while pos < buf.len() {
                match decode(&buf[pos..], self.dim) {
                    Some((id, _meta, mut v, used)) => {
                        normalize_in_place(&mut v);
                        self.apply(id, &v);
                        pos += used;
                    }
                    None => break,
                }
            }
            self.offset += pos as u64;
            Ok((buf.len() - pos) as u64)
        }

        fn apply(&mut self, id: u64, v_normalised: &[f32]) {
            self.ids.push(id);
            self.vecs.extend_from_slice(v_normalised);
            if let Some(h) = self.hnsw.as_mut() {
                h.add(id, v_normalised);
            }
        }

        pub fn len(&self) -> usize {
            self.ids.len()
        }
        pub fn is_empty(&self) -> bool {
            self.ids.is_empty()
        }
        pub fn dim(&self) -> usize {
            self.dim
        }
        pub fn log_bytes(&self) -> u64 {
            self.offset
        }
        pub fn snapshot_bytes(&self) -> u64 {
            fs::metadata(self.dir.join("index.snap")).map(|m| m.len()).unwrap_or(0)
        }
        /// Approximate resident bytes of the in-memory index.
        pub fn resident_bytes(&self) -> usize {
            self.vecs.len() * 4 + self.ids.len() * 8 + self.hnsw.as_ref().map(|h| h.bytes()).unwrap_or(0)
        }

        /// Append one record (writer only). The write is a single `write_all`.
        pub fn append(&mut self, id: u64, meta: &[u8], v: &[f32]) -> std::io::Result<()> {
            assert!(self.writable, "store opened read-only");
            assert_eq!(v.len(), self.dim);
            let rec = encode(id, meta, v);
            self.file.seek(SeekFrom::Start(self.offset))?;
            self.file.write_all(&rec)?;
            self.offset += rec.len() as u64;
            let mut n = v.to_vec();
            normalize_in_place(&mut n);
            self.apply(id, &n);
            Ok(())
        }

        /// Flush to stable storage (fsync).
        pub fn sync(&mut self) -> std::io::Result<()> {
            self.file.sync_all()
        }

        /// Reader: pick up records appended by the writer since the last call.
        /// Returns the number of new records.
        pub fn refresh(&mut self) -> std::io::Result<usize> {
            let before = self.ids.len();
            let _torn = self.replay_tail()?; // a record still being written is simply not visible yet
            Ok(self.ids.len() - before)
        }

        /// Top-k `(id, cosine)` by exact scan (Flat) or HNSW.
        pub fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
            match &self.hnsw {
                Some(h) => h.search(q, k),
                None => self.search_exact(q, k),
            }
        }

        /// Exact scan regardless of index kind (ground truth for recall).
        pub fn search_exact(&self, q: &[f32], k: usize) -> Vec<(u64, f32)> {
            let mut qn = q.to_vec();
            normalize_in_place(&mut qn);
            let mut top: Vec<(u64, f32)> = Vec::with_capacity(k + 1);
            for (row, id) in self.vecs.chunks_exact(self.dim).zip(&self.ids) {
                let s = dot(&qn, row);
                if top.len() < k || s > top[top.len() - 1].1 {
                    let pos = top.partition_point(|&(_, x)| x >= s);
                    top.insert(pos, (*id, s));
                    top.truncate(k);
                }
            }
            top
        }

        /// Fetch the metadata of record `id` by scanning the log (cold path, for tests/tools).
        pub fn read_meta(&self, id: u64) -> std::io::Result<Option<Vec<u8>>> {
            let f = File::open(self.dir.join("log.bin"))?;
            let mut buf = Vec::new();
            f.take(self.offset).read_to_end(&mut buf)?;
            let mut pos = 0;
            while pos < buf.len() {
                match decode(&buf[pos..], self.dim) {
                    Some((i, meta, _, used)) => {
                        if i == id {
                            return Ok(Some(meta));
                        }
                        pos += used;
                    }
                    None => break,
                }
            }
            Ok(None)
        }

        /// Write a checkpoint (atomic: temp file + rename).
        pub fn checkpoint(&mut self) -> std::io::Result<()> {
            assert!(self.writable);
            let n = self.ids.len();
            let mut body = Vec::with_capacity(n * 8 + self.vecs.len() * 4);
            for id in &self.ids {
                body.extend_from_slice(&id.to_le_bytes());
            }
            for x in &self.vecs {
                body.extend_from_slice(&x.to_le_bytes());
            }
            let mut head = Vec::with_capacity(32);
            head.extend_from_slice(&SNAP_MAGIC.to_le_bytes());
            head.extend_from_slice(&(self.dim as u32).to_le_bytes());
            head.extend_from_slice(&(n as u64).to_le_bytes());
            head.extend_from_slice(&self.offset.to_le_bytes());
            head.extend_from_slice(&crc32(&body).to_le_bytes());
            let tmp = self.dir.join("index.snap.tmp");
            {
                let mut f = File::create(&tmp)?;
                f.write_all(&head)?;
                f.write_all(&body)?;
                f.sync_all()?;
            }
            fs::rename(&tmp, self.dir.join("index.snap"))
        }

        fn read_snapshot(&self) -> Option<(u64, Vec<u64>, Vec<f32>)> {
            let mut f = File::open(self.dir.join("index.snap")).ok()?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).ok()?;
            if buf.len() < 28 || u32::from_le_bytes(buf[0..4].try_into().ok()?) != SNAP_MAGIC {
                return None;
            }
            let dim = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
            let n = u64::from_le_bytes(buf[8..16].try_into().ok()?) as usize;
            let off = u64::from_le_bytes(buf[16..24].try_into().ok()?);
            let crc = u32::from_le_bytes(buf[24..28].try_into().ok()?);
            let body = &buf[28..];
            if dim != self.dim || body.len() != n * 8 + n * dim * 4 || crc32(body) != crc {
                return None;
            }
            // snapshot must not claim more log than exists
            if off > self.file.metadata().ok()?.len() {
                return None;
            }
            let ids: Vec<u64> = body[..n * 8].chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
            let vecs: Vec<f32> = body[n * 8..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            Some((off, ids, vecs))
        }

        pub fn kind(&self) -> IndexKind {
            self.kind
        }
        pub fn hnsw_params(&self) -> HnswParams {
            self.hnsw_params
        }
    }

    impl Drop for Store {
        fn drop(&mut self) {
            if self.writable {
                let _ = fs::remove_file(self.dir.join("LOCK"));
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Evaluation harness shared by the r3_* binaries
// ---------------------------------------------------------------------------

pub mod harness {
    use super::ctxworld::{drive, RunStats, Stream};
    use super::*;
    use crate::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
    use crate::hebbian::{HebbianConfig, HebbianRouter};
    use crate::prior::{Evidence, Warm};
    use crate::sim::{Scenario, World, WorldConfig};

    #[derive(Clone, Copy, Debug)]
    pub enum Ev {
        Set,
        Mean,
        Global,
        FailSet,
        FailMean,
    }
    pub fn ev<R>(e: Ev) -> Evidence<R> {
        match e {
            Ev::Set => Evidence::FeatureSet,
            Ev::Mean => Evidence::MeanFeature,
            Ev::Global => Evidence::Global,
            Ev::FailSet => Evidence::SetFailures,
            Ev::FailMean => Evidence::FeatureFailures,
        }
    }

    pub fn heb_robust(n: usize) -> HebbianRouter {
        HebbianRouter::new(n, HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() })
    }

    #[derive(Clone, Debug)]
    pub struct Wm {
        pub cap: usize,
        pub min_n: u64,
        pub init_g: bool,
        pub chunk: bool,
        /// `Some((k, evidence))` wraps the buffer in the keyword warm start; `None` = bare
        pub warm: Option<(f64, Ev)>,
    }

    #[derive(Clone, Debug)]
    pub enum Kind {
        Keyword,
        Eps(f64, f64),
        Pts(f64),
        HebR,
        WarmHebR(f64, Ev),
        Wm(Wm),
    }

    pub fn kind_label(k: &Kind) -> String {
        match k {
            Kind::Keyword => "keyword".into(),
            Kind::Eps(..) => "eps-greedy".into(),
            Kind::Pts(..) => "pooled-ts".into(),
            Kind::HebR => "hebbian-robust".into(),
            Kind::WarmHebR(..) => "warm(hebbian-robust)".into(),
            Kind::Wm(w) => {
                let cap = if w.cap == usize::MAX { "inf".to_string() } else { w.cap.to_string() };
                format!("{}wm[K={}{}]", if w.warm.is_some() { "warm+" } else { "" }, cap, if w.chunk { ",chunk" } else { "" })
            }
        }
    }

    pub fn run(kind: &Kind, s: &Stream, seed: u64) -> RunStats {
        let a = s.n_agents;
        let km = || s.kwmap.clone();
        match kind {
            Kind::Keyword => drive(KeywordRouter::new(km(), a), s, seed),
            Kind::Eps(g, e) => drive(EpsGreedyRouter::new(a, *g, *e), s, seed),
            Kind::Pts(g) => drive(PooledThompsonRouter::new(a, *g), s, seed),
            Kind::HebR => drive(heb_robust(a), s, seed),
            Kind::WarmHebR(k, e) => drive(Warm::new(heb_robust(a), km(), a, *k).with_evidence(ev(*e)), s, seed),
            Kind::Wm(w) => {
                let cfg = WorkMemConfig { capacity: w.cap, min_n: w.min_n, init_from_global: w.init_g, chunk: w.chunk, ..Default::default() };
                let wm = WorkMemRouter::new(heb_robust(a), cfg);
                match w.warm {
                    Some((k, e)) => drive(Warm::new(wm, km(), a, k).with_evidence(ev(e)), s, seed),
                    None => drive(wm, s, seed),
                }
            }
        }
    }

    /// Stream from a `sim.rs` scenario with `ctx:<window>/0` markers (non-recurring control).
    pub fn sim_stream(seed: u64, scenario: Scenario, n_tasks: usize) -> Stream {
        let w = World::new(seed, WorldConfig { n_tasks, scenario, ..WorldConfig::default() });
        let na = w.n_agents();
        let mut s = Stream { tasks: vec![], p: vec![], u: vec![], revisit: vec![], kwmap: w.keyword_map_for(), n_agents: na };
        for t in 0..w.len() {
            let st = w.task(t);
            let mut task = st.task.clone();
            task.features.push(format!("ctx:{:?}/0", w.window(t)));
            s.tasks.push(task);
            s.p.push((0..na).map(|a| w.success_prob(t, &[a])).collect());
            s.u.push(st.u);
            s.revisit.push(false);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::ctxworld::*;
    use super::store::*;
    use super::*;
    use crate::baselines::EpsGreedyRouter;
    use crate::memory::HnswParams;
    use crate::hebbian::{HebbianConfig, HebbianRouter};

    fn heb() -> HebbianRouter {
        HebbianRouter::new(6, HebbianConfig { eta: 0.01, temperature: 0.002, decay: 0.01, epsilon: 0.0, ..HebbianConfig::default() })
    }

    #[test]
    fn k0_equals_inner_exactly() {
        let c = CtxCfg::default();
        let s = generate(3, &c);
        let a = drive(heb(), &s, 3);
        let b = drive(WorkMemRouter::new(heb(), WorkMemConfig { capacity: 0, ..Default::default() }), &s, 3);
        assert_eq!(a.mean, b.mean);
    }

    #[test]
    fn deterministic_and_lru_bounded() {
        let c = CtxCfg { families: 9, ..CtxCfg::default() };
        let s = generate(5, &c);
        let mk = || WorkMemRouter::new(EpsGreedyRouter::new(6, 0.99, 0.05), WorkMemConfig { capacity: 4, ..Default::default() });
        let (a, b) = (drive(mk(), &s, 5), drive(mk(), &s, 5));
        assert_eq!(a.mean, b.mean);
        let mut r = mk();
        let cands: Vec<usize> = (0..6).collect();
        let mut rng = Rng::new(1);
        for t in &s.tasks {
            let a = r.select(t, &cands, &mut rng);
            r.update(t, a, 1.0);
            assert!(r.slots_used() <= 4);
        }
        assert!(r.evictions > 0, "9 contexts through 4 slots must evict");
    }

    #[test]
    fn chunking_shares_slots_across_variants() {
        let mut r = WorkMemRouter::new(
            EpsGreedyRouter::new(6, 0.99, 0.05),
            WorkMemConfig { capacity: 8, chunk: true, ..Default::default() },
        );
        let cands: Vec<usize> = (0..6).collect();
        let mut rng = Rng::new(1);
        for v in 0..3 {
            let t = Task { id: v, task_type: None, features: vec![format!("ctx:f1/{v}"), "t0k0".into()] };
            let a = r.select(&t, &cands, &mut rng);
            r.update(&t, a, 1.0);
        }
        assert_eq!(r.slots_used(), 1);
    }

    #[test]
    fn carry_over_context_when_marker_missing() {
        let mut r = WorkMemRouter::new(heb(), WorkMemConfig { capacity: 4, ..Default::default() });
        let cands: Vec<usize> = (0..6).collect();
        let mut rng = Rng::new(1);
        let t1 = Task { id: 0, task_type: None, features: vec!["ctx:a/0".into(), "x".into()] };
        let t2 = Task { id: 1, task_type: None, features: vec!["x".into()] };
        let a = r.select(&t1, &cands, &mut rng);
        r.update(&t1, a, 1.0);
        let a = r.select(&t2, &cands, &mut rng);
        r.update(&t2, a, 1.0);
        assert_eq!(r.slots_used(), 1);
        assert_eq!(r.slots[0].n, 2);
    }

    #[test]
    fn generator_flags_revisits_and_unique_never_revisits() {
        let s = generate(1, &CtxCfg::default());
        assert!(s.revisit.iter().any(|&x| x) && s.revisit.iter().any(|&x| !x));
        let u = generate(1, &CtxCfg { rec: Recurrence::Unique, ..CtxCfg::default() });
        assert!(u.revisit.iter().all(|&x| !x));
        // Irrelevant: all contexts share one table
        let ir = generate(1, &CtxCfg { rec: Recurrence::Irrelevant, ..CtxCfg::default() });
        let mut by_topic: HashMap<Vec<u64>, usize> = HashMap::new();
        for p in &ir.p {
            *by_topic.entry(p.iter().map(|x| x.to_bits()).collect()).or_insert(0) += 1;
        }
        assert!(by_topic.len() <= 8);
    }

    // ---- store ----

    fn tmpdir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("nm_r3_{}_{}_{}", name, std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn rvec(rng: &mut Rng, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| (rng.f64() as f32) - 0.5).collect()
    }

    #[test]
    fn crc_known_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn store_roundtrip_and_reopen() {
        let d = tmpdir("rt");
        let mut rng = Rng::new(1);
        let vs: Vec<Vec<f32>> = (0..200).map(|_| rvec(&mut rng, 16)).collect();
        {
            let mut s = Store::open(&d, 16, IndexKind::Flat, true).unwrap();
            for (i, v) in vs.iter().enumerate() {
                s.append(i as u64, format!("m{i}").as_bytes(), v).unwrap();
            }
            s.sync().unwrap();
            assert_eq!(s.len(), 200);
        }
        let s = Store::open(&d, 16, IndexKind::Flat, false).unwrap();
        assert_eq!(s.len(), 200);
        let top = s.search(&vs[7], 3);
        assert_eq!(top[0].0, 7);
        assert!((top[0].1 - 1.0).abs() < 1e-4);
        assert_eq!(s.read_meta(42).unwrap().unwrap(), b"m42");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn store_recovers_torn_tail() {
        let d = tmpdir("torn");
        let mut rng = Rng::new(2);
        {
            let mut s = Store::open(&d, 8, IndexKind::Flat, true).unwrap();
            for i in 0..50u64 {
                s.append(i, b"x", &rvec(&mut rng, 8)).unwrap();
            }
        }
        // chop 7 bytes off the end: last record is torn
        let p = d.join("log.bin");
        let len = std::fs::metadata(&p).unwrap().len();
        std::fs::OpenOptions::new().write(true).open(&p).unwrap().set_len(len - 7).unwrap();
        // reader ignores the tail
        let r = Store::open(&d, 8, IndexKind::Flat, false).unwrap();
        assert_eq!(r.len(), 49);
        assert!(r.info.ignored_tail_bytes > 0);
        drop(r);
        // writer truncates it and can keep appending
        let mut w = Store::open(&d, 8, IndexKind::Flat, true).unwrap();
        assert_eq!(w.len(), 49);
        assert!(w.info.truncated_bytes > 0);
        w.append(1000, b"y", &rvec(&mut rng, 8)).unwrap();
        drop(w);
        let r = Store::open(&d, 8, IndexKind::Flat, false).unwrap();
        assert_eq!(r.len(), 50);
        assert_eq!(r.info.ignored_tail_bytes, 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn store_detects_corruption_midlog() {
        let d = tmpdir("corrupt");
        let mut rng = Rng::new(3);
        {
            let mut s = Store::open(&d, 8, IndexKind::Flat, true).unwrap();
            for i in 0..30u64 {
                s.append(i, b"", &rvec(&mut rng, 8)).unwrap();
            }
        }
        let p = d.join("log.bin");
        let mut bytes = std::fs::read(&p).unwrap();
        let rec = bytes.len() / 30;
        bytes[rec * 10 + 20] ^= 0xFF; // flip a byte inside record 10's payload
        std::fs::write(&p, &bytes).unwrap();
        let r = Store::open(&d, 8, IndexKind::Flat, false).unwrap();
        assert_eq!(r.len(), 10, "log is valid up to the corrupt record");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn store_snapshot_warm_open_matches_cold() {
        let d = tmpdir("snap");
        let mut rng = Rng::new(4);
        let mut w = Store::open(&d, 12, IndexKind::Flat, true).unwrap();
        for i in 0..300u64 {
            w.append(i, b"a", &rvec(&mut rng, 12)).unwrap();
        }
        w.checkpoint().unwrap();
        for i in 300..350u64 {
            w.append(i, b"b", &rvec(&mut rng, 12)).unwrap();
        }
        drop(w);
        let warm = Store::open(&d, 12, IndexKind::Flat, false).unwrap();
        assert_eq!(warm.info.from_snapshot, 300);
        assert_eq!(warm.info.replayed, 50);
        let cold = Store::open_with(&d, 12, IndexKind::Flat, false, false, HnswParams::default()).unwrap();
        assert_eq!(cold.info.from_snapshot, 0);
        assert_eq!(warm.len(), cold.len());
        let q = rvec(&mut rng, 12);
        assert_eq!(warm.search(&q, 5), cold.search(&q, 5));
        // corrupt snapshot -> falls back to replay
        let sp = d.join("index.snap");
        let mut b = std::fs::read(&sp).unwrap();
        let n = b.len();
        b[n - 3] ^= 1;
        std::fs::write(&sp, b).unwrap();
        let fb = Store::open(&d, 12, IndexKind::Flat, false).unwrap();
        assert_eq!(fb.info.from_snapshot, 0);
        assert_eq!(fb.len(), 350);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn store_single_writer_lock_and_reader_refresh() {
        let d = tmpdir("lock");
        let mut rng = Rng::new(5);
        let mut w = Store::open(&d, 8, IndexKind::Flat, true).unwrap();
        assert!(Store::open(&d, 8, IndexKind::Flat, true).is_err(), "second writer must be refused");
        for i in 0..10u64 {
            w.append(i, b"", &rvec(&mut rng, 8)).unwrap();
        }
        let mut r = Store::open(&d, 8, IndexKind::Flat, false).unwrap();
        assert_eq!(r.len(), 10);
        for i in 10..25u64 {
            w.append(i, b"", &rvec(&mut rng, 8)).unwrap();
        }
        assert_eq!(r.refresh().unwrap(), 15);
        assert_eq!(r.len(), 25);
        drop(w);
        assert!(Store::open(&d, 8, IndexKind::Flat, true).is_ok(), "lock released on drop");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn store_hnsw_recall_reasonable() {
        let d = tmpdir("hnsw");
        let mut rng = Rng::new(6);
        let mut w = Store::open(&d, 16, IndexKind::Hnsw, true).unwrap();
        for i in 0..800u64 {
            w.append(i, b"", &rvec(&mut rng, 16)).unwrap();
        }
        let mut hit = 0;
        for _ in 0..40 {
            let q = rvec(&mut rng, 16);
            let t: std::collections::HashSet<u64> = w.search_exact(&q, 5).iter().map(|x| x.0).collect();
            hit += w.search(&q, 5).iter().filter(|x| t.contains(&x.0)).count();
        }
        assert!(hit as f64 / 200.0 > 0.8, "recall {}", hit as f64 / 200.0);
        let _ = std::fs::remove_dir_all(&d);
    }
}
