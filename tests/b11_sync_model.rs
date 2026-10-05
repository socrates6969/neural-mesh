//! B11: randomized MODEL CHECK of the shared-folder sync (src/store.rs::sync_folder). Seeded schedules of: local appends on 4 machines, syncs in random order,
//! transport faults (a machine's folder file truncated mid-line / mid UTF-8 char, a file deleted, a leftover .tmp), then a quiescent phase (every machine syncs 3 rounds
//! with intact files). Invariant: every machine ends holding EXACTLY the union of everything ever appended (no loss, no phantom, no duplicates), regardless of the schedule.
use neural_mesh::store::{MeshStore, Outcome};
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::Write;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
    }
}
fn oc(s: &str, i: u64, w: &str) -> Outcome {
    Outcome { id: format!("{s}:{i}"), session: s.into(), seq: i, ts: i, world: w.into(), features: vec!["f".into()], task_type: None, agent: "a".into(), reward: 1.0 }
}

fn run_schedule(seed: u64) {
    let base = std::env::temp_dir().join(format!("b11_{}_{seed}", std::process::id()));
    let _ = fs::remove_dir_all(&base);
    let folder = base.join("folder");
    let m = 4usize;
    let stores: Vec<MeshStore> = (0..m).map(|k| MeshStore::open(base.join(format!("m{k}"))).unwrap()).collect();
    let mut next_seq = vec![0u64; m];
    let mut all: BTreeSet<String> = BTreeSet::new();
    let mut r = Lcg(seed * 7919 + 13);
    for _step in 0..80 {
        let k = r.below(m);
        match r.below(10) {
            0..=3 => {
                let n = 1 + r.below(4);
                let w = if r.below(2) == 0 { "w\u{e9}1" } else { "w\u{20ac}2" };
                let os: Vec<Outcome> = (0..n).map(|_| { let o = oc(&format!("m{k}"), next_seq[k], w); next_seq[k] += 1; o }).collect();
                for o in &os { all.insert(o.id.clone()); }
                stores[k].append_outcomes(&os).unwrap();
            }
            4..=6 => {
                stores[k].sync_folder(&folder, &format!("m{k}"), None, None).unwrap();
            }
            7 => {
                // truncate a random machine file at a random byte (torn line / torn UTF-8)
                let p = folder.join(format!("m{}.journal.jsonl", r.below(m)));
                if let Ok(b) = fs::read(&p) {
                    if !b.is_empty() {
                        let cut = r.below(b.len());
                        fs::write(&p, &b[..cut]).unwrap();
                    }
                }
            }
            8 => {
                let p = folder.join(format!("m{}.journal.jsonl", r.below(m)));
                let _ = fs::remove_file(p);
            }
            _ => {
                if folder.exists() {
                    let mut f = OpenOptions::new().create(true).append(true).open(folder.join(format!("mZ{}.journal.jsonl.tmp", r.below(3)))).unwrap();
                    f.write_all(b"garbage\xe2\x82").unwrap();
                }
            }
        }
    }
    for _round in 0..3 {
        for k in 0..m {
            stores[k].sync_folder(&folder, &format!("m{k}"), None, None).unwrap();
        }
    }
    for k in 0..m {
        let ids: Vec<String> = stores[k].outcomes().unwrap().into_iter().map(|o| o.id).collect();
        let set: BTreeSet<String> = ids.iter().cloned().collect();
        assert_eq!(ids.len(), set.len(), "seed {seed} machine {k}: duplicate ids");
        let missing: Vec<_> = all.difference(&set).take(3).collect();
        let phantom: Vec<_> = set.difference(&all).take(3).collect();
        assert!(missing.is_empty() && phantom.is_empty(), "seed {seed} machine {k}: missing {missing:?} phantom {phantom:?} (have {}, union {})", set.len(), all.len());
    }
    let _ = fs::remove_dir_all(&base);
}

#[test]
fn sync_random_schedules_converge_to_the_union() {
    for seed in 0..60 {
        run_schedule(seed);
    }
}

/// shared GC cap: after quiescence every machine holds the SAME set (the newest `cap` per world of what they collectively hold), no machine exceeds the cap.
#[test]
fn sync_with_cap_converges_to_identical_sets() {
    for seed in 0..30u64 {
        let base = std::env::temp_dir().join(format!("b11c_{}_{seed}", std::process::id()));
        let _ = fs::remove_dir_all(&base);
        let folder = base.join("folder");
        let m = 3usize;
        let cap = 25usize;
        let stores: Vec<MeshStore> = (0..m).map(|k| MeshStore::open(base.join(format!("m{k}"))).unwrap()).collect();
        let mut seqs = vec![0u64; m];
        let mut r = Lcg(seed * 104729 + 5);
        for step in 0..60u64 {
            let k = r.below(m);
            if r.below(3) == 0 {
                stores[k].sync_folder(&folder, &format!("m{k}"), None, Some(cap)).unwrap();
            } else {
                let o = Outcome { id: format!("m{k}:{}", seqs[k]), session: format!("m{k}"), seq: seqs[k], ts: step * 10 + k as u64, world: "w".into(), features: vec![], task_type: None, agent: "a".into(), reward: 1.0 };
                seqs[k] += 1;
                stores[k].append_outcome(&o).unwrap();
            }
        }
        for _ in 0..4 {
            for k in 0..m {
                stores[k].sync_folder(&folder, &format!("m{k}"), None, Some(cap)).unwrap();
            }
        }
        let sets: Vec<BTreeSet<String>> = stores.iter().map(|s| s.outcomes().unwrap().into_iter().map(|o| o.id).collect()).collect();
        for k in 0..m {
            assert!(sets[k].len() <= cap, "seed {seed} machine {k} holds {} > cap {cap}", sets[k].len());
            assert_eq!(sets[k], sets[0], "seed {seed}: machine {k} differs from machine 0 ({} vs {})", sets[k].len(), sets[0].len());
        }
        let _ = fs::remove_dir_all(&base);
    }
}
