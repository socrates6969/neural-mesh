//! codemem: incremental BM25 "project memory" for code and docs (see bench/codemem/README.md for design and measurements).
//! Corpus = the git-tracked-or-untracked-not-ignored files of the configured repos (`git ls-files -co --exclude-standard`, so .gitignore is
//! respected exactly), minus secrets / env / keys / data dirs / binaries / lockfiles, chunked by symbol or heading, scrubbed, tokenised
//! (identifier-aware), BM25 k1 .9 b .5 (RD-LOG R54). The index lives under data/codemem/<repo>/index.cmem and updates per file (mtime+size, then hash).

pub mod chunk;
pub mod cli;
pub mod gate;
pub mod scrub;
pub mod store;
pub mod tok;
pub mod walk;

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;
use store::{Build, Index};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RepoCfg {
    pub name: String,
    pub path: String,
    /// extra excluded path prefixes (relative, `/`-separated; a trailing `*` makes it a prefix glob)
    #[serde(default)]
    pub exclude: Vec<String>,
    /// extra directories indexed under their absolute path (e.g. agent run notes)
    #[serde(default)]
    pub extra_roots: Vec<String>,
    /// project vocabulary that makes the hook gate fire
    #[serde(default)]
    pub terms: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub repos: Vec<RepoCfg>,
    /// emit snake_case/camelCase parts as extra tokens (default off, see tok.rs)
    #[serde(default)]
    pub split: bool,
}

pub fn data_dir() -> PathBuf {
    match std::env::var("NEURAL_MESH_CODEMEM_DIR") {
        Ok(d) if !d.trim().is_empty() => PathBuf::from(d),
        _ => Path::new(env!("CARGO_MANIFEST_DIR")).join("data").join("codemem"),
    }
}

pub fn default_config() -> Config {
    let r = |n: &str, p: &str, ex: &[&str], ex_roots: &[&str], terms: &[&str]| RepoCfg {
        name: n.into(),
        path: p.into(),
        exclude: ex.iter().map(|s| s.to_string()).collect(),
        extra_roots: ex_roots.iter().map(|s| s.to_string()).collect(),
        terms: terms.iter().map(|s| s.to_string()).collect(),
    };
    let runs = "C:/Users/mariu/.claude/plugins/data/statsclaw-synced/workspace/freight-recovery/runs";
    let fr_terms = ["freight", "tenant", "rls", "fastify", "prisma", "fr/shared", "evidence", "carrier", "invoice", "migration"];
    Config {
        split: true,
        repos: vec![
            // bench/ holds the benchmark answer keys: kept out so a retrieval benchmark cannot read its own answers
            r("neural-mesh", "C:/Users/mariu/neural-mesh", &["bench/battery", "bench/results/", "bench/work/", "bench/data/", "bench/tests/", "bench/PREREG.json", "bench/codemem/", "results/"], &[], &["mesh", "hebbian", "router", "bm25", "codemem", "lifelong"]),
            r("hebbian-router", "C:/Users/mariu/hebbian-router", &[], &[], &["hebbian", "router"]),
            r("freight-recovery", "C:/Users/mariu/ventures/freight-recovery", &[], &[runs], &fr_terms),
            r("freight-recovery-build", "C:/Users/mariu/ventures/freight-recovery-build", &[], &[runs], &fr_terms),
            r("freight-recovery-brand", "C:/Users/mariu/ventures/freight-recovery-brand", &[], &[runs], &fr_terms),
        ],
    }
}

pub fn load_config(dir: &Path) -> Config {
    std::fs::read_to_string(dir.join("config.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_else(default_config)
}

pub fn index_path(dir: &Path, repo: &str) -> PathBuf {
    dir.join(repo).join("index.cmem")
}

fn norm_path(p: &str) -> String {
    p.replace('\\', "/").trim_end_matches('/').to_ascii_lowercase()
}

/// The repo whose path is the longest component-wise prefix of `cwd`.
pub fn repo_for_cwd<'a>(cfg: &'a Config, cwd: &str) -> Option<&'a RepoCfg> {
    let c = norm_path(cwd);
    cfg.repos
        .iter()
        .filter(|r| {
            let p = norm_path(&r.path);
            c == p || c.starts_with(&format!("{p}/"))
        })
        .max_by_key(|r| r.path.len())
}

#[derive(Debug, Default, Serialize)]
pub struct UpdateReport {
    pub repo: String,
    pub files: usize,
    pub changed: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub chunks: usize,
    pub terms: usize,
    pub bytes: usize,
    pub scrub_hits: BTreeMap<String, usize>,
    pub ms: u128,
}

fn mtime_ns(m: &std::fs::Metadata) -> u64 {
    m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as u64).unwrap_or(0)
}

pub fn fnv64(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325u64, |h, &c| (h ^ c as u64).wrapping_mul(0x100000001b3))
}

struct Reuse<'a> {
    old: &'a Build,
    map: Vec<Option<u32>>,
}

impl Reuse<'_> {
    /// copies old file `fi` (with a new mtime) into `nb`, remapping term ids
    fn copy(&mut self, nb: &mut Build, fi: usize, mtime: u64) {
        let f = &self.old.files[fi];
        nb.begin_file(&f.path, mtime, f.size, f.hash);
        for c in &self.old.chunks[f.c0 as usize..(f.c0 + f.cn) as usize] {
            let terms: Vec<(u32, u32)> = c
                .terms
                .iter()
                .map(|&(t, tf)| {
                    let id = *self.map[t as usize].get_or_insert_with(|| nb.intern(&self.old.terms[t as usize]));
                    (id, tf)
                })
                .collect();
            let file = (nb.files.len() - 1) as u32;
            nb.chunks.push(store::ChunkRec { file, terms, ..c.clone() });
            nb.files[file as usize].cn += 1;
        }
    }
}

fn index_new_file(nb: &mut Build, path: &str, bytes: &[u8], mtime: u64, hits: &mut BTreeMap<String, usize>) {
    let text = String::from_utf8_lossy(bytes);
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    nb.begin_file(path, mtime, bytes.len() as u64, fnv64(bytes));
    let ptoks = tok::tokens(&path.replace(['/', '\\', '.', '-'], " "), nb.split);
    for c in chunk::chunk_file(path, text) {
        let (clean, h) = scrub::scrub(&c.text);
        for (k, n) in h {
            *hits.entry(k.to_string()).or_default() += n;
        }
        let ttoks = tok::tokens(&c.title, nb.split);
        let mut toks = tok::tokens(&clean, nb.split);
        toks.extend(ptoks.iter().cloned());
        toks.extend(ttoks.iter().cloned());
        toks.extend(ttoks);
        toks.extend(clean.lines().filter_map(chunk::def_name_of_line).map(|n| format!("def:{n}")));
        nb.push_chunk(c.l0, c.l1, &c.title, clean, &toks);
    }
}

pub fn update_repo(cfg: &Config, repo: &RepoCfg, dir: &Path, force: bool) -> Result<UpdateReport, String> {
    let t0 = Instant::now();
    let ipath = index_path(dir, &repo.name);
    let old = if force { None } else { Index::open(&ipath).ok().filter(|ix| ix.split == cfg.split).map(|mut ix| ix.to_build()) };
    let old_by_path: HashMap<&str, usize> = old.iter().flat_map(|b| b.files.iter().enumerate().map(|(i, f)| (f.path.as_str(), i))).collect();
    let mut reuse = old.as_ref().map(|b| Reuse { old: b, map: vec![None; b.terms.len()] });
    let listed = walk::list_files(repo)?;
    let (mut nb, mut rep) = (Build::new(cfg.split), UpdateReport { repo: repo.name.clone(), ..Default::default() });
    let mut seen: HashSet<&str> = HashSet::new();
    for (rel, abs) in &listed {
        let Ok(md) = std::fs::metadata(abs) else { continue };
        let (mt, size) = (mtime_ns(&md), md.len());
        let prev = old_by_path.get(rel.as_str()).copied();
        if let (Some(fi), Some(r)) = (prev, reuse.as_mut()) {
            let f = &r.old.files[fi];
            if f.mtime == mt && f.size == size {
                r.copy(&mut nb, fi, mt);
                rep.unchanged += 1;
                seen.insert(rel);
                continue;
            }
        }
        let Ok(bytes) = std::fs::read(abs) else { continue };
        if walk::looks_binary(&bytes) {
            continue;
        }
        seen.insert(rel);
        match (prev, reuse.as_mut()) {
            (Some(fi), Some(r)) if r.old.files[fi].hash == fnv64(&bytes) => {
                r.copy(&mut nb, fi, mt);
                rep.unchanged += 1;
            }
            _ => {
                index_new_file(&mut nb, rel, &bytes, mt, &mut rep.scrub_hits);
                rep.changed += 1;
            }
        }
    }
    rep.removed = old.as_ref().map(|b| b.files.iter().filter(|f| !seen.contains(f.path.as_str())).count()).unwrap_or(0);
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let bytes = store::encode(&nb, created);
    store::write_atomic(&ipath, &bytes).map_err(|e| format!("write {}: {e}", ipath.display()))?;
    (rep.files, rep.chunks, rep.terms, rep.bytes, rep.ms) = (nb.files.len(), nb.chunks.len(), nb.terms.len(), bytes.len(), t0.elapsed().as_millis());
    Ok(rep)
}

// ------------------------------------------------------------------------------------------------ search

#[derive(Clone, Debug, Default)]
pub struct QuerySpec {
    pub terms: Vec<(String, f32)>,
    pub path_hints: Vec<String>,
}

/// weight of the `def:<name>` term: the chunk that DECLARES an identifier named in the query outranks chunks that only use it
pub const DEF_WEIGHT: f32 = 4.0;

const EXTS: [&str; 16] = ["rs", "py", "ts", "tsx", "js", "jsx", "json", "md", "sql", "prisma", "tf", "toml", "yaml", "yml", "ps1", "sh"];

fn is_path_word(w: &str) -> bool {
    let ext = w.rsplit_once('.').map(|x| x.1.to_ascii_lowercase());
    (w.contains('/') && w.len() > 3) || ext.is_some_and(|e| EXTS.contains(&e.as_str()) && w.len() > e.len() + 1)
}

/// Raw question or prompt text -> weighted terms. Backticked spans weigh 2, path-like words 1.5 (and become path hints), the rest 1.
pub fn parse_query(raw: &str, split: bool) -> QuerySpec {
    let mut w: BTreeMap<String, f32> = BTreeMap::new();
    let mut spec = QuerySpec::default();
    let mut add = |toks: Vec<String>, wt: f32| {
        for t in toks {
            let e = w.entry(t).or_insert(0.0);
            *e = e.max(wt);
        }
    };
    let mut plain = String::new();
    for (i, part) in raw.split('`').enumerate() {
        if i % 2 == 1 {
            add(tok::tokens(part, split), 2.0);
            add(part.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).filter(|w| w.len() >= 2).map(|w| format!("def:{}", w.to_ascii_lowercase())).collect(), DEF_WEIGHT);
        } else {
            plain.push_str(part);
            plain.push(' ');
        }
    }
    for word in plain.split_whitespace() {
        let word = word.trim_matches(|c: char| !(c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-' | '\\')));
        if is_path_word(word) {
            let hint = word.replace('\\', "/").trim_start_matches("./").to_ascii_lowercase();
            add(tok::tokens(&hint.replace(['/', '.', '-'], " "), split), 1.5);
            spec.path_hints.push(hint);
        }
    }
    add(tok::tokens(&plain, split), 1.0);
    let idents = plain.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).filter(|w| gate::is_ident_like(w));
    add(idents.map(|w| format!("def:{}", w.to_ascii_lowercase())).collect(), DEF_WEIGHT);
    spec.terms = w.into_iter().collect();
    spec.terms.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    spec.terms.truncate(48);
    spec
}

#[derive(Clone, Debug)]
pub struct Hit {
    pub chunk: usize,
    pub score: f32,
}

#[derive(Debug, Default)]
pub struct Search {
    pub hits: Vec<Hit>,
    /// best score / the most a chunk could score on the in-vocabulary query terms, in [0,1]
    pub coverage: f32,
    pub matched: usize,
    pub rare_matched: usize,
}

/// Prose files (markdown / text, and agent run notes under absolute extra-root paths) describe the code; for "where is X" the code itself should win.
pub fn is_doc_path(p: &str) -> bool {
    let l = p.to_ascii_lowercase();
    l.ends_with(".md") || l.ends_with(".mdx") || l.ends_with(".txt") || l.contains(":/") || l.starts_with('/')
}

pub const DOC_FACTOR: f32 = 0.65;

/// BM25 over the chunk index; at most `per_file` hits per file; path hints multiply the score of chunks in matching files by 1.5.
pub fn search(ix: &Index, q: &QuerySpec, k: usize, per_file: usize) -> Search {
    let mut acc = vec![0f32; ix.n_chunks];
    let (mut den, mut matched, mut rare) = (0f32, 0, 0);
    for (t, wt) in &q.terms {
        let idf = ix.accumulate(t, *wt, &mut acc);
        if idf > 0.0 {
            den += wt * idf;
            matched += 1;
            if ix.df(t) * 50 <= ix.n_chunks.max(50) {
                rare += 1;
            }
        }
    }
    let mut cand: Vec<(usize, f32)> = acc.iter().enumerate().filter(|x| *x.1 > 0.0).map(|(i, s)| (i, *s)).collect();
    let keep = 60.max(k * 6);
    if cand.len() > keep {
        cand.select_nth_unstable_by(keep, |a, b| b.1.total_cmp(&a.1));
        cand.truncate(keep);
    }
    let docf = DOC_FACTOR;
    if docf != 1.0 {
        for c in cand.iter_mut() {
            if is_doc_path(ix.file_path(ix.chunk(c.0).file)) {
                c.1 *= docf;
            }
        }
    }
    if !q.path_hints.is_empty() {
        for c in cand.iter_mut() {
            let p = ix.file_path(ix.chunk(c.0).file).to_ascii_lowercase();
            if q.path_hints.iter().any(|h| p.contains(h.as_str())) {
                c.1 *= 1.5;
            }
        }
    }
    cand.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut per: HashMap<usize, usize> = HashMap::new();
    let hits: Vec<Hit> = cand
        .into_iter()
        .filter(|(c, _)| {
            let n = per.entry(ix.chunk(*c).file).or_default();
            *n += 1;
            *n <= per_file
        })
        .take(k)
        .map(|(chunk, score)| Hit { chunk, score })
        .collect();
    let coverage = if den > 0.0 { hits.first().map(|h| h.score / ((store::K1 + 1.0) * den)).unwrap_or(0.0).min(1.5) } else { 0.0 };
    Search { hits, coverage, matched, rare_matched: rare }
}

#[derive(Clone, Debug, Serialize)]
pub struct Pointer {
    pub path: String,
    pub line: u32,
    pub l0: u32,
    pub l1: u32,
    pub title: String,
    pub score: f32,
    pub snippet: Vec<(u32, String)>,
}

fn clip(s: &str, n: usize) -> String {
    let t = s.trim();
    if t.chars().count() <= n { t.to_string() } else { format!("{}…", t.chars().take(n).collect::<String>()) }
}

/// File:line pointer with up to two snippet lines: the chunk's lines that share the most query terms.
pub fn pointer(ix: &mut Index, h: &Hit, q: &QuerySpec, line_chars: usize) -> Pointer {
    let split = ix.split;
    let text = ix.text(h.chunk);
    let c = ix.chunk(h.chunk);
    let (path, l0, l1, title) = (ix.file_path(c.file).to_string(), c.l0, c.l1, c.title.to_string());
    let want: HashSet<&str> = q.terms.iter().map(|t| t.0.as_str()).collect();
    let mut scored: Vec<(usize, usize, &str)> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            let mut seen = HashSet::new();
            let n = tok::tokens(l, split).into_iter().filter(|t| want.contains(t.as_str()) && seen.insert(t.clone())).count();
            let is_def = chunk::def_name_of_line(l).is_some_and(|d| want.contains(format!("def:{d}").as_str()));
            (n + if is_def { 3 } else { 0 }, i, l)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let best = scored.first().filter(|x| x.0 > 0).map(|x| x.1).unwrap_or(0);
    let mut sn: Vec<(u32, String)> = scored.iter().filter(|x| x.0 > 0).take(2).map(|x| (l0 + x.1 as u32, clip(x.2, line_chars))).collect();
    if sn.is_empty() {
        if let Some(x) = scored.first() {
            sn.push((l0 + x.1 as u32, clip(x.2, line_chars)));
        }
    }
    sn.sort_by_key(|x| x.0);
    Pointer { path, line: l0 + best as u32, l0, l1, title, score: h.score, snippet: sn }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let d = std::env::temp_dir().join(format!("cm-{tag}-{}-{}", std::process::id(), N.fetch_add(1, std::sync::atomic::Ordering::SeqCst)));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn repo(root: &Path) -> RepoCfg {
        RepoCfg { name: "t".into(), path: root.to_string_lossy().replace('\\', "/"), exclude: vec![], extra_roots: vec![], terms: vec![] }
    }

    fn write(root: &Path, rel: &str, s: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, s).unwrap();
    }

    fn fixture() -> (PathBuf, PathBuf, Config) {
        let (root, data) = (tmp("root"), tmp("data"));
        write(&root, "src/tenant.ts", "// sets the tenant\nexport async function withTenant(id: string) {\n  await tx.$executeRaw`select set_config('app.tenant_id', ${id}, true)`;\n}\n");
        write(&root, "src/other.py", "def unrelated():\n    return 1\n");
        write(&root, "docs/guide.md", "# Guide\nthe quarantine policy is described here\n");
        write(&root, "secrets/key.txt", "topsecret");
        write(&root, ".env", "API_KEY=abc");
        write(&root, "src/leak.ts", "export const k = 'sk-ant-api03-abcdefghijklmnopqrstuv';\nexport function leaky() {}\n");
        let cfg = Config { repos: vec![repo(&root)], split: false };
        (root, data, cfg)
    }

    #[test]
    fn index_query_and_exclusions() {
        let (root, data, cfg) = fixture();
        let rep = update_repo(&cfg, &cfg.repos[0], &data, false).unwrap();
        assert_eq!((rep.files, rep.changed), (4, 4), "{rep:?}");
        let mut ix = Index::open(&index_path(&data, "t")).unwrap();
        let q = parse_query("where is the tenant id set (set_config)?", false);
        let s = search(&ix, &q, 3, 2);
        let p = pointer(&mut ix, &s.hits[0], &q, 100);
        assert_eq!(p.path, "src/tenant.ts");
        assert_eq!(p.line, 3, "best line is the set_config line: {p:?}");
        assert!(p.snippet.iter().any(|x| x.1.contains("set_config")));
        for i in 0..ix.n_files {
            let f = ix.file_path(i);
            assert!(!f.contains("secrets") && !f.contains(".env"), "{f}");
        }
        assert_eq!(search(&ix, &parse_query("topsecret", false), 3, 2).hits.len(), 0);
        let _ = std::fs::remove_dir_all((root, data).0);
    }

    #[test]
    fn scrubbed_before_indexing() {
        let (root, data, cfg) = fixture();
        update_repo(&cfg, &cfg.repos[0], &data, false).unwrap();
        let mut ix = Index::open(&index_path(&data, "t")).unwrap();
        assert_eq!(search(&ix, &parse_query("abcdefghijklmnopqrstuv", false), 3, 2).hits.len(), 0);
        let i = (0..ix.n_chunks).find(|&i| ix.chunk(i).title.contains("leaky") || ix.file_path(ix.chunk(i).file) == "src/leak.ts").unwrap();
        assert!(!ix.text(i).contains("abcdefghijkl"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn incremental_update_touches_only_changed_files() {
        let (root, data, cfg) = fixture();
        update_repo(&cfg, &cfg.repos[0], &data, false).unwrap();
        let r2 = update_repo(&cfg, &cfg.repos[0], &data, false).unwrap();
        assert_eq!((r2.changed, r2.unchanged, r2.removed), (0, 4, 0), "{r2:?}");
        std::thread::sleep(std::time::Duration::from_millis(30));
        write(&root, "src/other.py", "def newly_added_marker():\n    return 2\n");
        std::fs::remove_file(root.join("docs/guide.md")).unwrap();
        let r3 = update_repo(&cfg, &cfg.repos[0], &data, false).unwrap();
        assert_eq!((r3.changed, r3.removed, r3.files), (1, 1, 3), "{r3:?}");
        let ix = Index::open(&index_path(&data, "t")).unwrap();
        assert_eq!(search(&ix, &parse_query("newly_added_marker", false), 3, 2).hits.len(), 1);
        assert_eq!(search(&ix, &parse_query("quarantine", false), 3, 2).hits.len(), 0);
        assert_eq!(search(&ix, &parse_query("withTenant", false), 3, 2).hits.len(), 1, "unchanged file still searchable");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn path_hint_boosts_and_repo_for_cwd() {
        let q = parse_query("what is in `Foo_Bar` and src/db/tenant.ts", false);
        assert_eq!(q.path_hints, vec!["src/db/tenant.ts"]);
        assert!(q.terms.iter().any(|t| t.0 == "foo_bar" && t.1 == 2.0));
        let cfg = default_config();
        assert_eq!(repo_for_cwd(&cfg, r"C:\Users\mariu\ventures\freight-recovery-build\api\src").unwrap().name, "freight-recovery-build");
        assert_eq!(repo_for_cwd(&cfg, "C:/Users/mariu/ventures/freight-recovery").unwrap().name, "freight-recovery");
        assert!(repo_for_cwd(&cfg, "C:/Users/mariu/Desktop").is_none());
    }
}
