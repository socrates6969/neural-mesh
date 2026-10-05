p='src/store.rs'
s=open(p,encoding='utf-8').read()
def rep(o,n,cnt=1):
    global s
    assert o in s,o[:60]
    s=s.replace(o,n,cnt)
rep('fn run_fuzz(canonical: bool) -> usize {','fn run_fuzz(canonical: bool) -> usize {\n        run_fuzz_ttl(canonical, None)\n    }\n    fn run_fuzz_ttl(canonical: bool, ttl_days: Option<u64>) -> usize {\n        let day = 86_400_000u64;\n        let t0 = now_ms();\n        let mut clock = 0u64;')
# replace sync calls inside run_fuzz_ttl
i=s.index('fn run_fuzz_ttl')
head,tail=s[:i],s[i:]
tail=tail.replace('stores[m].sync_docs_folder_opts(&folder, names[m], "kb", true).unwrap();','stores[m].sync_docs_folder_with(&folder, names[m], "kb", &DocSyncOpts { propagate_deletions: true, tombstone_ttl_ms: ttl_days.map(|d| d * day), now_ms: Some(t0 + clock * day), ..Default::default() }).unwrap();')
# clock advances on each random op
tail=tail.replace('                let m = rnd(3);\n                match rnd(4) {','                let m = rnd(3);\n                clock += rnd(3) as u64;\n                match rnd(4) {',1)
s=head+tail
s=s.replace('''    fn run_fuzz(canonical''','''    /// B48: the same fuzz with tombstone expiry (TTL 2 days, clock advancing 0-2 days per op). Expiry may let a holder re-introduce a doc, but must never lose a class or misalign vectors,
    /// and after the quiescent sync-gc-sync tail (clock frozen) canonical gc still converges.
    #[test]
    fn expiry_never_loses_a_class_and_canonical_gc_still_converges() {
        assert_eq!(run_fuzz_ttl(true, Some(2)), 0);
    }
    fn run_fuzz(canonical''',1)
open(p,'w',encoding='utf-8').write(s)
