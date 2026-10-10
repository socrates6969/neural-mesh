//! On-disk index (single file, atomic replace). Layout, little endian:
//! header(104) | files(40 each) | chunks(36 each) | terms(16 each, sorted by term) | term strings | postings (varint doc-delta, varint tf) | string pool | text blob.
//! The reader loads everything except the text blob (<= a few MB) and looks terms up by binary search; chunk text is read with a seek only for hits.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"CMEMIDX1";
/// bumped whenever tokenisation / chunking changes (an index of another version is rebuilt from scratch, never patched): 2 = `def:` tokens
const VERSION: u32 = 2;
const HDR: usize = 104;
const FILE_REC: usize = 40;
const CHUNK_REC: usize = 36;
const TERM_REC: usize = 16;
pub const K1: f32 = 0.9;
pub const B: f32 = 0.5;

#[derive(Clone, Debug, PartialEq)]
pub struct FileRec {
    pub path: String,
    pub mtime: u64,
    pub size: u64,
    pub hash: u64,
    pub c0: u32,
    pub cn: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChunkRec {
    pub file: u32,
    pub l0: u32,
    pub l1: u32,
    pub dl: u32,
    pub title: String,
    pub text: String,
    /// (term id in `Build::terms`, tf)
    pub terms: Vec<(u32, u32)>,
}

#[derive(Default)]
pub struct Build {
    pub split: bool,
    pub files: Vec<FileRec>,
    pub chunks: Vec<ChunkRec>,
    pub terms: Vec<String>,
    ids: HashMap<String, u32>,
}

impl Build {
    pub fn new(split: bool) -> Build {
        Build { split, ..Default::default() }
    }
    pub fn intern(&mut self, t: &str) -> u32 {
        if let Some(&i) = self.ids.get(t) {
            return i;
        }
        let i = self.terms.len() as u32;
        self.terms.push(t.to_string());
        self.ids.insert(t.to_string(), i);
        i
    }
    /// Adds a chunk (tokens already produced) to the file currently being appended (`self.files.len() - 1`).
    pub fn push_chunk(&mut self, l0: u32, l1: u32, title: &str, text: String, toks: &[String]) {
        let mut tf: HashMap<u32, u32> = HashMap::new();
        for t in toks {
            let id = self.intern(t);
            *tf.entry(id).or_default() += 1;
        }
        let mut terms: Vec<(u32, u32)> = tf.into_iter().collect();
        terms.sort_unstable();
        let file = (self.files.len() - 1) as u32;
        self.chunks.push(ChunkRec { file, l0, l1, dl: toks.len() as u32, title: title.to_string(), text, terms });
        self.files[file as usize].cn += 1;
    }
    pub fn begin_file(&mut self, path: &str, mtime: u64, size: u64, hash: u64) {
        let c0 = self.chunks.len() as u32;
        self.files.push(FileRec { path: path.to_string(), mtime, size, hash, c0, cn: 0 });
    }
}

fn put_varint(v: &mut Vec<u8>, mut x: u32) {
    while x >= 0x80 {
        v.push((x as u8) | 0x80);
        x >>= 7;
    }
    v.push(x as u8);
}

fn get_varint(b: &[u8], pos: &mut usize) -> u32 {
    let (mut x, mut sh) = (0u32, 0);
    while *pos < b.len() {
        let c = b[*pos];
        *pos += 1;
        x |= ((c & 0x7f) as u32) << sh;
        if c & 0x80 == 0 {
            break;
        }
        sh += 7;
    }
    x
}

fn u32at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}
fn u64at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn f32at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// Serialises `b` to bytes; `created` is a unix timestamp stored in the header.
pub fn encode(b: &Build, created: u64) -> Vec<u8> {
    let mut order: Vec<u32> = (0..b.terms.len() as u32).collect();
    order.sort_by(|x, y| b.terms[*x as usize].cmp(&b.terms[*y as usize]));
    let mut rank = vec![0u32; b.terms.len()];
    for (r, &t) in order.iter().enumerate() {
        rank[t as usize] = r as u32;
    }
    let mut post: Vec<Vec<(u32, u32)>> = vec![Vec::new(); b.terms.len()];
    for (d, c) in b.chunks.iter().enumerate() {
        for &(t, tf) in &c.terms {
            post[rank[t as usize] as usize].push((d as u32, tf));
        }
    }
    let (mut pool, mut files, mut chunks, mut terms, mut tstr, mut pbytes, mut text) = (vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    for f in &b.files {
        files.extend_from_slice(&(pool.len() as u32).to_le_bytes());
        files.extend_from_slice(&(f.path.len() as u32).to_le_bytes());
        pool.extend_from_slice(f.path.as_bytes());
        for v in [f.mtime, f.size, f.hash] {
            files.extend_from_slice(&v.to_le_bytes());
        }
        files.extend_from_slice(&f.c0.to_le_bytes());
        files.extend_from_slice(&f.cn.to_le_bytes());
    }
    let mut total_dl = 0u64;
    for c in &b.chunks {
        total_dl += c.dl as u64;
        for v in [c.file, c.l0, c.l1, c.dl, pool.len() as u32, c.title.len() as u32, c.text.len() as u32] {
            chunks.extend_from_slice(&v.to_le_bytes());
        }
        chunks.extend_from_slice(&(text.len() as u64).to_le_bytes());
        pool.extend_from_slice(c.title.as_bytes());
        text.extend_from_slice(c.text.as_bytes());
    }
    for (r, &t) in order.iter().enumerate() {
        let s = &b.terms[t as usize];
        let p = &post[r];
        for v in [tstr.len() as u32, s.len() as u32, p.len() as u32, pbytes.len() as u32] {
            terms.extend_from_slice(&v.to_le_bytes());
        }
        tstr.extend_from_slice(s.as_bytes());
        let mut prev = 0;
        for &(d, tf) in p {
            put_varint(&mut pbytes, d - prev);
            put_varint(&mut pbytes, tf);
            prev = d;
        }
    }
    let avg = if b.chunks.is_empty() { 1.0 } else { (total_dl as f32 / b.chunks.len() as f32).max(1.0) };
    let secs = [&files, &chunks, &terms, &tstr, &pbytes, &pool, &text];
    let mut offs = [0u64; 7];
    let mut o = HDR as u64;
    for (i, s) in secs.iter().enumerate() {
        offs[i] = o;
        o += s.len() as u64;
    }
    let mut out = Vec::with_capacity(o as usize);
    out.extend_from_slice(MAGIC);
    for v in [VERSION, b.split as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in [K1, B] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in [b.files.len() as u32, b.chunks.len() as u32, b.terms.len() as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&avg.to_le_bytes());
    out.extend_from_slice(&created.to_le_bytes());
    for v in offs {
        out.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(out.len(), HDR);
    for s in secs {
        out.extend_from_slice(s);
    }
    out
}

pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    File::create(&tmp)?.write_all(bytes)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

pub struct Index {
    buf: Vec<u8>,
    file: File,
    pub split: bool,
    pub n_files: usize,
    pub n_chunks: usize,
    pub n_terms: usize,
    pub avgdl: f32,
    pub created: u64,
    offs: [usize; 7],
}

pub struct ChunkView<'a> {
    pub file: usize,
    pub l0: u32,
    pub l1: u32,
    pub dl: u32,
    pub title: &'a str,
    pub text_off: u64,
    pub text_len: usize,
}

impl Index {
    pub fn open(path: &Path) -> std::io::Result<Index> {
        let mut file = File::open(path)?;
        let mut hdr = vec![0u8; HDR];
        file.read_exact(&mut hdr)?;
        let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
        if &hdr[..8] != MAGIC || u32at(&hdr, 8) != VERSION {
            return Err(bad("not a codemem index"));
        }
        let mut offs = [0usize; 7];
        for (i, o) in offs.iter_mut().enumerate() {
            *o = u64at(&hdr, 48 + 8 * i) as usize;
        }
        let mut buf = hdr;
        buf.resize(offs[6], 0);
        file.read_exact(&mut buf[HDR..])?;
        let ix = Index {
            split: u32at(&buf, 12) != 0,
            n_files: u32at(&buf, 24) as usize,
            n_chunks: u32at(&buf, 28) as usize,
            n_terms: u32at(&buf, 32) as usize,
            avgdl: f32at(&buf, 36),
            created: u64at(&buf, 40),
            offs,
            buf,
            file,
        };
        if ix.offs[1] + ix.n_chunks * CHUNK_REC > ix.offs[2] || ix.offs[0] + ix.n_files * FILE_REC > ix.offs[1] {
            return Err(bad("corrupt index"));
        }
        Ok(ix)
    }
    fn pool(&self, off: u32, len: u32) -> &str {
        let a = self.offs[5] + off as usize;
        std::str::from_utf8(self.buf.get(a..a + len as usize).unwrap_or(&[])).unwrap_or("")
    }
    pub fn file_rec(&self, i: usize) -> FileRec {
        let o = self.offs[0] + i * FILE_REC;
        FileRec {
            path: self.pool(u32at(&self.buf, o), u32at(&self.buf, o + 4)).to_string(),
            mtime: u64at(&self.buf, o + 8),
            size: u64at(&self.buf, o + 16),
            hash: u64at(&self.buf, o + 24),
            c0: u32at(&self.buf, o + 32),
            cn: u32at(&self.buf, o + 36),
        }
    }
    pub fn file_path(&self, i: usize) -> &str {
        let o = self.offs[0] + i * FILE_REC;
        self.pool(u32at(&self.buf, o), u32at(&self.buf, o + 4))
    }
    pub fn chunk(&self, i: usize) -> ChunkView<'_> {
        let o = self.offs[1] + i * CHUNK_REC;
        ChunkView {
            file: u32at(&self.buf, o) as usize,
            l0: u32at(&self.buf, o + 4),
            l1: u32at(&self.buf, o + 8),
            dl: u32at(&self.buf, o + 12),
            title: self.pool(u32at(&self.buf, o + 16), u32at(&self.buf, o + 20)),
            text_len: u32at(&self.buf, o + 24) as usize,
            text_off: u64at(&self.buf, o + 28),
        }
    }
    pub fn text(&mut self, i: usize) -> String {
        let c = self.chunk(i);
        let (off, len) = (self.offs[6] as u64 + c.text_off, c.text_len);
        let mut v = vec![0u8; len];
        if self.file.seek(SeekFrom::Start(off)).is_err() || self.file.read_exact(&mut v).is_err() {
            return String::new();
        }
        String::from_utf8_lossy(&v).into_owned()
    }
    fn term_str(&self, r: usize) -> &[u8] {
        let o = self.offs[2] + r * TERM_REC;
        let a = self.offs[3] + u32at(&self.buf, o) as usize;
        &self.buf[a..a + u32at(&self.buf, o + 4) as usize]
    }
    /// (df, posting byte offset) of a term.
    fn find(&self, t: &str) -> Option<(usize, usize)> {
        let (mut lo, mut hi) = (0usize, self.n_terms);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.term_str(mid).cmp(t.as_bytes()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let o = self.offs[2] + mid * TERM_REC;
                    return Some((u32at(&self.buf, o + 8) as usize, self.offs[4] + u32at(&self.buf, o + 12) as usize));
                }
            }
        }
        None
    }
    pub fn df(&self, t: &str) -> usize {
        self.find(t).map(|x| x.0).unwrap_or(0)
    }
    pub fn idf(&self, df: usize) -> f32 {
        (1.0 + (self.n_chunks as f32 - df as f32 + 0.5) / (df as f32 + 0.5)).ln()
    }
    /// Adds `weight * idf * bm25-tf-norm` of term `t` to `acc` (len n_chunks). Returns the term's idf (0 if absent).
    pub fn accumulate(&self, t: &str, weight: f32, acc: &mut [f32]) -> f32 {
        let Some((df, mut p)) = self.find(t) else { return 0.0 };
        let idf = self.idf(df);
        let mut doc = 0u32;
        for _ in 0..df {
            doc += get_varint(&self.buf, &mut p);
            let tf = get_varint(&self.buf, &mut p) as f32;
            let dl = u32at(&self.buf, self.offs[1] + doc as usize * CHUNK_REC + 12) as f32;
            let norm = tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / self.avgdl));
            acc[doc as usize] += weight * idf * norm;
        }
        idf
    }
    /// Rebuilds a `Build` (all files and chunks, forward term lists decoded from the postings) for incremental updates.
    pub fn to_build(&mut self) -> Build {
        let mut b = Build::new(self.split);
        let mut fwd: Vec<Vec<(u32, u32)>> = vec![Vec::new(); self.n_chunks];
        for r in 0..self.n_terms {
            let term = String::from_utf8_lossy(self.term_str(r)).into_owned();
            let id = b.intern(&term);
            let o = self.offs[2] + r * TERM_REC;
            let (df, mut p) = (u32at(&self.buf, o + 8), self.offs[4] + u32at(&self.buf, o + 12) as usize);
            let mut doc = 0u32;
            for _ in 0..df {
                doc += get_varint(&self.buf, &mut p);
                fwd[doc as usize].push((id, get_varint(&self.buf, &mut p)));
            }
        }
        for i in 0..self.n_files {
            b.files.push(self.file_rec(i));
        }
        for (i, terms) in fwd.into_iter().enumerate() {
            let c = self.chunk(i);
            let (file, l0, l1, dl, title) = (c.file as u32, c.l0, c.l1, c.dl, c.title.to_string());
            let text = self.text(i);
            b.chunks.push(ChunkRec { file, l0, l1, dl, title, text, terms });
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Build {
        let mut b = Build::new(false);
        b.begin_file("a.rs", 1, 10, 7);
        b.push_chunk(1, 3, "fn alpha", "fn alpha() { tenant }".into(), &["alpha".into(), "tenant".into(), "tenant".into()]);
        b.begin_file("b.md", 2, 20, 8);
        b.push_chunk(1, 2, "Intro", "tenant isolation".into(), &["tenant".into(), "isolation".into()]);
        b
    }

    #[test]
    fn roundtrip_and_scoring() {
        let dir = std::env::temp_dir().join(format!("cm-store-{}", std::process::id()));
        let p = dir.join("i.cmem");
        write_atomic(&p, &encode(&sample(), 5)).unwrap();
        let mut ix = Index::open(&p).unwrap();
        assert_eq!((ix.n_files, ix.n_chunks, ix.n_terms, ix.created), (2, 2, 3, 5));
        assert_eq!(ix.file_path(1), "b.md");
        assert_eq!(ix.text(1), "tenant isolation");
        assert_eq!(ix.df("tenant"), 2);
        let mut acc = vec![0.0; 2];
        assert!(ix.accumulate("alpha", 1.0, &mut acc) > 0.0);
        assert!(acc[0] > 0.0 && acc[1] == 0.0);
        let b2 = ix.to_build();
        assert_eq!((b2.files.len(), b2.chunks.len(), b2.chunks[0].text.as_str()), (2, 2, "fn alpha() { tenant }"));
        assert_eq!(encode(&b2, 5).len(), encode(&sample(), 5).len());
        let _ = std::fs::remove_dir_all(dir);
    }
}
