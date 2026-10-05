c='src/bin/meshstore.rs'
t=open(c,encoding='utf-8').read()
o=t[t.index('            let mut o = neural_mesh::store::DocSyncOpts::default();'):t.index('            let r = s.sync_docs_folder_with(')]
n='''            let mut o = neural_mesh::store::DocSyncOpts::default();
            let usage = |m: &str| -> ! {
                eprintln!("sync-docs: {m}\nusage: meshstore sync-docs <store> <folder> <machine> <ns> [--propagate-deletions] [--ttl-days N] [--deleters m1,m2]");
                std::process::exit(2)
            };
            let mut i = 5;
            while i < a.len() {
                match a[i].as_str() {
                    "--propagate-deletions" => o.propagate_deletions = true,
                    "--ttl-days" => {
                        i += 1;
                        match a.get(i).and_then(|x| x.parse::<f64>().ok()).filter(|d| *d >= 0.0) {
                            Some(d) => o.tombstone_ttl_ms = Some((d * 86_400_000.0) as u64),
                            None => usage("--ttl-days needs a non-negative number"),
                        }
                    }
                    "--deleters" => {
                        i += 1;
                        match a.get(i) {
                            Some(v) => o.deleters = Some(v.split(',').map(|x| x.to_string()).collect()),
                            None => usage("--deleters needs a comma list"),
                        }
                    }
                    other => usage(&format!("unknown flag {other}")),
                }
                i += 1;
            }
'''
t=t.replace(o,n)
open(c,'w',encoding='utf-8').write(t)
