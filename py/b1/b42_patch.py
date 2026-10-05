p='src/store.rs'
s=open(p,encoding='utf-8').read()
def rep(o,n):
    global s
    assert o in s,o[:50]
    s=s.replace(o,n,1)
rep('''    pub fn gc_docs(&self, ns: &str, threshold: f32) -> R<(usize, usize)> {
        let _g = self.lock()?;''','''    pub fn gc_docs(&self, ns: &str, threshold: f32) -> R<(usize, usize)> {
        self.gc_docs_impl(ns, threshold, false)
    }
    /// B42 (opt-in): like [`MeshStore::gc_docs`] but the "first wins" order is the doc ID order, not the row order, so the same set of docs always collapses to the same
    /// representatives on every machine (row order depends on arrival order, which differs per machine; B41 measured 5/60 fuzz seeds ending with different representatives of
    /// a class under row-order gc). Output keeps the original row order of the survivors. Survivors can differ from row-order gc.
    pub fn gc_docs_canonical(&self, ns: &str, threshold: f32) -> R<(usize, usize)> {
        self.gc_docs_impl(ns, threshold, true)
    }
    fn gc_docs_impl(&self, ns: &str, threshold: f32, canonical: bool) -> R<(usize, usize)> {
        let _g = self.lock()?;''')
rep('''        for i in 0..n {
            if i % 64 == 0 {
                _g.heartbeat();
            }
            let vi = &unit[i * dim..(i + 1) * dim];''','''        let mut order: Vec<usize> = (0..n).collect();
        if canonical {
            order.sort_by(|&a, &b| docs[a].id.cmp(&docs[b].id).then(a.cmp(&b)));
        }
        for (step, &i) in order.iter().enumerate() {
            if step % 64 == 0 {
                _g.heartbeat();
            }
            let vi = &unit[i * dim..(i + 1) * dim];''')
rep('''        if keep.len() == n {
            return Ok((n, n));
        }
        let d = self.kdir(ns);
        let (dt, vt)''','''        if keep.len() == n {
            return Ok((n, n));
        }
        keep.sort_unstable(); // survivors keep their original row order
        let d = self.kdir(ns);
        let (dt, vt)''')
# fuzz: parametrise gc
rep('''    fn deletion_exchange_fuzz_never_loses_a_class_and_converges() {
        let dim = 4usize;''','''    fn deletion_exchange_fuzz_never_loses_a_class_and_converges() {
        let r = run_fuzz(false);
        eprintln!("B41: seeds whose stores kept different representatives of some class (row-order gc): {r} / 60");
    }
    /// B42: with canonical (id-ordered) gc every seed converges to the IDENTICAL id set on every store (fails with row-order gc: B41 measured 5/60 diverging seeds).
    #[test]
    fn canonical_gc_makes_representatives_converge() {
        assert_eq!(run_fuzz(true), 0, "stores must hold identical id sets");
    }
    fn run_fuzz(canonical: bool) -> usize {
        let diverged = std::sync::atomic::AtomicUsize::new(0);
        let dim = 4usize;''')
rep('''                        let _ = stores[m].gc_docs("kb", 0.99);''','''                        let _ = if canonical { stores[m].gc_docs_canonical("kb", 0.99) } else { stores[m].gc_docs("kb", 0.99) };''')
rep('''                DIVERGED_REPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        eprintln!("B41: seeds whose stores kept different representatives of some class: {} / 60", DIVERGED_REPS.load(std::sync::atomic::Ordering::Relaxed));
    }
    static DIVERGED_REPS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);''','''                diverged.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        diverged.load(std::sync::atomic::Ordering::Relaxed)
    }''')
open(p,'w',encoding='utf-8').write(s)
