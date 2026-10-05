p='src/store.rs'
s=open(p,encoding='utf-8').read()
o='''            for _ in 0..4 {
                for m in 0..3 {
                    stores[m].sync_docs_folder_opts(&folder, names[m], "kb", true).unwrap();
                }
            }
            let sets'''
n='''            // quiescent tail, the intended maintenance cycle: sync, gc everywhere, sync again
            for phase in 0..2 {
                for _ in 0..4 {
                    for m in 0..3 {
                        stores[m].sync_docs_folder_opts(&folder, names[m], "kb", true).unwrap();
                    }
                }
                if phase == 0 {
                    for m in 0..3 {
                        let _ = if canonical { stores[m].gc_docs_canonical("kb", 0.99) } else { stores[m].gc_docs("kb", 0.99) };
                    }
                }
            }
            let sets'''
assert o in s
open(p,'w',encoding='utf-8').write(s.replace(o,n))
