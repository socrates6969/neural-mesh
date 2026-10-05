//! meshstore: tiny admin CLI for the durable shared mesh memory (src/store.rs). Opt-in tooling, touches nothing else.
//!   meshstore stats  <root>
//!   meshstore merge  <dst-root> <src-root>            union src into dst (idempotent)
//!   meshstore put-corpus <root> <ns> <texts.json> <vecs.f32> <dim> [tag] [limit]   (texts.json = JSON array of strings; vecs row-major f32 LE)
//!   meshstore lesson <root> <world> <tag> <text...>    append one lesson
//!   meshstore sync   <root> <shared-folder> <machine-id> [cap-per-world]   shared-folder sync (R48; outcomes + lessons, union by id, atomic own-file export)
//!   meshstore sync-docs <root> <shared-folder> <machine-id> <ns>   shared-folder CORPUS sync for one namespace (B15; int8 vectors, own shard replaced atomically)
//!   meshstore gc-docs <root> <ns> <cos-threshold>      drop near-duplicate corpus docs (R47)
//!   meshstore bench  <root> <ns>                       time load + flat/HNSW index build for a stored corpus
//! The store must only ever hold synthetic routing outcomes, lessons and knowledge (never secrets / customer data).
use neural_mesh::memory::{HnswIndex, HnswParams, MemoryIndex};
use neural_mesh::store::{DocRec, LessonRec, MeshStore};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let bad = || -> ! {
        eprintln!("usage: meshstore stats|merge|put-corpus|lesson|bench ... (see source header)");
        std::process::exit(1)
    };
    match a.first().map(String::as_str) {
        Some("stats") if a.len() == 2 => {
            let s = MeshStore::open(&a[1]).unwrap();
            println!("{}", serde_json::to_string_pretty(&s.stats().unwrap()).unwrap());
        }
        Some("merge") if a.len() == 3 => {
            let (d, s) = (MeshStore::open(&a[1]).unwrap(), MeshStore::open(&a[2]).unwrap());
            let r = d.merge_from(&s).unwrap();
            println!("{{\"outcomes_added\":{},\"lessons_added\":{},\"docs_added\":{}}}", r.outcomes_added, r.lessons_added, r.docs_added);
        }
        Some("sync") if a.len() >= 4 => {
            let s = MeshStore::open(&a[1]).unwrap();
            let cap = a.get(4).and_then(|x| x.parse::<usize>().ok());
            let r = s.sync_folder(std::path::Path::new(&a[2]), &a[3], None, cap).unwrap();
            println!("{}", serde_json::json!({"exported": r.exported, "imported": r.imported, "lessons_imported": r.lessons_imported, "files_read": r.files_read, "bytes_written": r.bytes_written}));
        }
        Some("sync-docs") if a.len() == 5 => {
            let s = MeshStore::open(&a[1]).unwrap();
            let r = s.sync_docs_folder(std::path::Path::new(&a[2]), &a[3], &a[4]).unwrap();
            println!("{}", serde_json::json!({"exported": r.exported, "imported": r.imported, "conflicts": r.conflicts, "skipped": r.skipped, "files_read": r.files_read, "bytes_written": r.bytes_written}));
        }
        Some("gc-docs") if a.len() == 4 => {
            let s = MeshStore::open(&a[1]).unwrap();
            let (b, af) = s.gc_docs(&a[2], a[3].parse().unwrap()).unwrap();
            println!("{}", serde_json::json!({"before": b, "after": af}));
        }
        Some("put-corpus") if a.len() >= 6 => {
            let s = MeshStore::open(&a[1]).unwrap();
            let texts: Vec<String> = serde_json::from_slice(&std::fs::read(&a[3]).unwrap()).unwrap();
            let dim: usize = a[5].parse().unwrap();
            let raw = std::fs::read(&a[4]).unwrap();
            let vecs: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            let tag = a.get(6).cloned().unwrap_or_else(|| "synthetic".into());
            let n = a.get(7).and_then(|x| x.parse::<usize>().ok()).unwrap_or(texts.len()).min(texts.len()).min(vecs.len() / dim);
            let docs: Vec<DocRec> = (0..n).map(|i| DocRec { id: format!("{}:{i}", a[2]), text: texts[i].clone(), tag: tag.clone() }).collect();
            let added = s.put_docs(&a[2], dim, &docs, &vecs[..n * dim]).unwrap();
            println!("{{\"added\":{added},\"ns\":\"{}\"}}", a[2]);
        }
        Some("lesson") if a.len() >= 5 => {
            let s = MeshStore::open(&a[1]).unwrap();
            let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64;
            let text = a[4..].join(" ");
            let id = format!("lesson-{:016x}", neural_mesh::embedding::fnv1a(&text));
            s.append_lesson(&LessonRec { id, session: "meshstore-cli".into(), ts: ms, world: a[2].clone(), agent: None, tag: a[3].clone(), text }).unwrap();
            println!("{{\"ok\":true}}");
        }
        Some("bench") if a.len() == 3 => {
            let s = MeshStore::open(&a[1]).unwrap();
            let t0 = Instant::now();
            let (docs, vecs, dim) = s.load_docs(&a[2]).unwrap();
            let t_load = t0.elapsed().as_secs_f64() * 1e3;
            let t1 = Instant::now();
            let mut ix = HnswIndex::new(dim, HnswParams { max_elements: docs.len().max(1), ..HnswParams::default() });
            for i in 0..docs.len() {
                ix.add(i as u64, &vecs[i * dim..(i + 1) * dim]);
            }
            let t_build = t1.elapsed().as_secs_f64() * 1e3;
            println!("{{\"docs\":{},\"dim\":{dim},\"load_ms\":{t_load:.1},\"hnsw_rebuild_ms\":{t_build:.1}}}", docs.len());
        }
        Some("ann") if a.len() == 3 => {
            let s = MeshStore::open(&a[1]).unwrap();
            let t = Instant::now();
            let x = s.open_ann(&a[2]).unwrap().expect("empty namespace");
            println!("{{\"ns\":\"{}\",\"n\":{},\"rebuilt\":{},\"open_ms\":{:.1}}}", a[2], x.n, x.rebuilt, t.elapsed().as_secs_f64() * 1e3);
        }
        Some("bench-dump") if a.len() == 3 => {
            // feasibility: hnsw_rs graph dump + reload vs rebuild (measured, not wired into HnswIndex)
            use hnsw_rs::prelude::*;
            let s = MeshStore::open(&a[1]).unwrap();
            let (docs, vecs, dim) = s.load_docs(&a[2]).unwrap();
            let t0 = Instant::now();
            let h: Hnsw<f32, DistCosine> = Hnsw::new(16, docs.len().max(1), 16, 200, DistCosine {});
            for i in 0..docs.len() {
                h.insert_slice((&vecs[i * dim..(i + 1) * dim], i));
            }
            let t_build = t0.elapsed().as_secs_f64() * 1e3;
            let dir = std::env::temp_dir().join(format!("nm_hnswdump_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let t1 = Instant::now();
            let base = h.file_dump(&dir, "ix").unwrap();
            let t_dump = t1.elapsed().as_secs_f64() * 1e3;
            let t2 = Instant::now();
            let mut io = HnswIo::new(&dir, &base);
            let h2: Hnsw<f32, DistCosine> = io.load_hnsw().unwrap();
            let t_load = t2.elapsed().as_secs_f64() * 1e3;
            let q = &vecs[0..dim];
            let r1 = h.search(q, 5, 64);
            let r2 = h2.search(q, 5, 64);
            let same = r1.iter().zip(&r2).all(|(x, y)| x.d_id == y.d_id);
            let bytes: u64 = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().metadata().unwrap().len()).sum();
            println!("{{\"docs\":{},\"build_ms\":{t_build:.1},\"dump_ms\":{t_dump:.1},\"reload_ms\":{t_load:.1},\"dump_bytes\":{bytes},\"same_top5\":{same}}}", docs.len());
            let _ = std::fs::remove_dir_all(&dir);
        }
        _ => bad(),
    }
}
