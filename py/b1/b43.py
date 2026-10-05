p='src/store.rs'
s=open(p,encoding='utf-8').read()
def rep(o,n):
    global s
    assert o in s,o[:50]
    s=s.replace(o,n,1)
rep('''    pub fn put_docs(&self, ns: &str, dim: usize, docs: &[DocRec], vecs: &[f32]) -> R<usize> {
        if vecs.len()''','''    pub fn put_docs(&self, ns: &str, dim: usize, docs: &[DocRec], vecs: &[f32]) -> R<usize> {
        self.put_docs_impl(ns, dim, docs, vecs, false)
    }
    /// B43: `put_docs`, but ids in the namespace's tombstones are skipped, decided UNDER the store lock (so a gc that tombstones an id between a sync's snapshot and its import
    /// cannot be undone by that import). Used by `sync_docs_folder_with`; a plain `put_docs` of a tombstoned id still works (documented).
    fn put_docs_untombstoned(&self, ns: &str, dim: usize, docs: &[DocRec], vecs: &[f32]) -> R<usize> {
        self.put_docs_impl(ns, dim, docs, vecs, true)
    }
    fn put_docs_impl(&self, ns: &str, dim: usize, docs: &[DocRec], vecs: &[f32], skip_tombstoned: bool) -> R<usize> {
        if vecs.len()''')
rep('''        let mut added = 0;
        for (i, doc) in docs.iter().enumerate() {
            if have.contains(&doc.id) {
                continue;
            }''','''        let tomb: HashSet<String> = if skip_tombstoned { lossy_lines(&d.join("tombstones.jsonl")).map(|ls| ls.iter().filter_map(|l| serde_json::from_str::<String>(l).ok()).collect()).unwrap_or_default() } else { HashSet::new() };
        let mut added = 0;
        for (i, doc) in docs.iter().enumerate() {
            if have.contains(&doc.id) || tomb.contains(&doc.id) {
                continue;
            }''')
rep("rep.imported = self.put_docs(ns, new_dim, &new_docs, &new_vecs)?;","rep.imported = self.put_docs_untombstoned(ns, new_dim, &new_docs, &new_vecs)?;")
s+='''
#[cfg(test)]
mod b43_tests {
    use super::*;
    /// B43: the tombstone check is made under the lock at write time. Simulates the race: the sync already decided to import `y`, then a gc tombstones it before the write.
    #[test]
    fn import_skips_ids_tombstoned_after_the_snapshot() {
        let p = std::env::temp_dir().join(format!("nm_b43_{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        let s = MeshStore::open(&p).unwrap();
        let mk = |i: &str| DocRec { id: i.into(), text: format!("t{i}"), tag: "t".into() };
        s.put_docs("kb", 2, &[mk("x")], &[1.0, 0.0]).unwrap();
        // tombstone appears (a concurrent gc) after the importer's snapshot but before its write
        fs::write(s.kdir("kb").join("tombstones.jsonl"), "\\"y\\"\n").unwrap();
        let n = s.put_docs_untombstoned("kb", 2, &[mk("y"), mk("z")], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(n, 1, "only z is imported");
        let ids: Vec<String> = s.load_docs("kb").unwrap().0.into_iter().map(|d| d.id).collect();
        assert_eq!(ids, vec!["x", "z"]);
        // an explicit put_docs of a tombstoned id still works (documented)
        assert_eq!(s.put_docs("kb", 2, &[mk("y")], &[1.0, 0.0]).unwrap(), 1);
    }
}
'''
# CLI canonical gc
open(p,'w',encoding='utf-8').write(s)
c='src/bin/meshstore.rs'
t=open(c,encoding='utf-8').read()
o='''        Some("gc-docs") if a.len() == 4 => {
            let s = MeshStore::open(&a[1]).unwrap();
            let (b, af) = s.gc_docs(&a[2], a[3].parse().unwrap()).unwrap();'''
n='''        Some("gc-docs") if a.len() == 4 || (a.len() == 5 && a[4] == "--canonical") => {
            let s = MeshStore::open(&a[1]).unwrap();
            let (b, af) = if a.len() == 5 { s.gc_docs_canonical(&a[2], a[3].parse().unwrap()) } else { s.gc_docs(&a[2], a[3].parse().unwrap()) }.unwrap();'''
assert o in t
open(c,'w',encoding='utf-8').write(t.replace(o,n))
