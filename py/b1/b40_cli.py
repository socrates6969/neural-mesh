p='src/bin/meshstore.rs'
s=open(p,encoding='utf-8').read()
old=s[s.index('        Some("sync-docs") if'):s.index('        Some("gc-docs")')]
new='''        Some("sync-docs") if a.len() >= 5 => {
            // flags: --propagate-deletions  --ttl-days N  --deleters m1,m2
            let s = MeshStore::open(&a[1]).unwrap();
            let mut o = neural_mesh::store::DocSyncOpts::default();
            let mut i = 5;
            while i < a.len() {
                match a[i].as_str() {
                    "--propagate-deletions" => o.propagate_deletions = true,
                    "--ttl-days" => {
                        i += 1;
                        o.tombstone_ttl_ms = Some((a[i].parse::<f64>().unwrap() * 86_400_000.0) as u64);
                    }
                    "--deleters" => {
                        i += 1;
                        o.deleters = Some(a[i].split(',').map(|x| x.to_string()).collect());
                    }
                    other => panic!("unknown flag {other}"),
                }
                i += 1;
            }
            let r = s.sync_docs_folder_with(std::path::Path::new(&a[2]), &a[3], &a[4], &o).unwrap();
            println!("{}", serde_json::json!({"deleted": r.deleted, "deletions_held_back": r.deletions_held_back, "deletions_unauthorized": r.deletions_unauthorized, "tombstones_expired": r.tombstones_expired, "exported": r.exported, "imported": r.imported, "conflicts": r.conflicts, "skipped": r.skipped, "files_read": r.files_read, "skipped_files": r.skipped_files, "bytes_written": r.bytes_written}));
        }
'''
s=s.replace(old,new)
open(p,'w',encoding='utf-8').write(s)
