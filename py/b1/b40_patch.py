p = 'src/store.rs'
s = open(p, encoding='utf-8').read()


def rep(old, new, count=-1):
    global s
    assert old in s, old[:60]
    s = s.replace(old, new) if count < 0 else s.replace(old, new, count)


rep("""struct DropLine {
    id: String,
    by: String,
}""", """struct DropLine {
    id: String,
    by: String,
    /// B40: when the deletion originated (unix ms); 0 = unknown (legacy line, never expires). Carried unchanged through relays so transitive propagation cannot extend a tombstone's life.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    ts: u64,
}
fn is_zero_u64(x: &u64) -> bool {
    *x == 0
}

/// B40: options for [`MeshStore::sync_docs_folder_with`].
#[derive(Clone, Debug, Default)]
pub struct DocSyncOpts {
    /// exchange gc deletions (B38)
    pub propagate_deletions: bool,
    /// tombstones older than this (ms since their origin gc) are expired: dropped from the local tombstone files (the doc may then be re-imported from a peer that still holds it),
    /// not exported, and ignored when read from peers. Must exceed the longest time a peer can stay offline. `None` = permanent (B38 behaviour).
    pub tombstone_ttl_ms: Option<u64>,
    /// authority: only tombstone files owned by these machine ids are applied; others are counted in `deletions_unauthorized`. `None` = every peer may delete.
    pub deleters: Option<Vec<String>>,
    /// test hook: pretend it is this unix-ms instant
    pub now_ms: Option<u64>,
}""")
rep("""    pub deletions_held_back: usize,
    pub exported""", """    pub deletions_held_back: usize,
    /// B40: peer deletions ignored because the owning machine is not in `DocSyncOpts::deleters`
    pub deletions_unauthorized: usize,
    /// B40: tombstones (local + peer lines) dropped because they outlived `tombstone_ttl_ms`
    pub tombstones_expired: usize,
    pub exported""")
rep("DropLine { id: docs[i].id.clone(), by: docs[j].id.clone() }", "DropLine { id: docs[i].id.clone(), by: docs[j].id.clone(), ts: now_ms() }")
rep("""    pub fn sync_docs_folder_opts(&self, folder: &Path, machine: &str, ns: &str, propagate_deletions: bool) -> R<DocSyncReport> {
        let ok""", """    pub fn sync_docs_folder_opts(&self, folder: &Path, machine: &str, ns: &str, propagate_deletions: bool) -> R<DocSyncReport> {
        self.sync_docs_folder_with(folder, machine, ns, &DocSyncOpts { propagate_deletions, ..Default::default() })
    }
    /// B40: [`MeshStore::sync_docs_folder_opts`] plus tombstone EXPIRY and a who-may-delete AUTHORITY list (see [`DocSyncOpts`]). Expiry trade-off, stated plainly: after a tombstone
    /// expires a peer that still holds the doc and has not gc'd it will re-introduce it, so the TTL must exceed the longest peer absence. Authority is by machine id as named in the shard
    /// file (the shared folder has no cryptographic identity: this guards against accidents and unvetted peers, not against a malicious writer of the folder).
    pub fn sync_docs_folder_with(&self, folder: &Path, machine: &str, ns: &str, o: &DocSyncOpts) -> R<DocSyncReport> {
        let propagate_deletions = o.propagate_deletions;
        let now = o.now_ms.unwrap_or_else(now_ms);
        let expired = |t: u64| matches!(o.tombstone_ttl_ms, Some(ttl) if t > 0 && now.saturating_sub(t) > ttl);
        let ok""")
rep("""        let mut rep = DocSyncReport::default();
        let own = format!("{machine}.{ns}.docs.jsonl");""", """        let mut rep = DocSyncReport::default();
        if o.tombstone_ttl_ms.is_some() {
            rep.tombstones_expired += self.expire_local_tombstones(ns, &expired)?;
        }
        let own = format!("{machine}.{ns}.docs.jsonl");""", 1)
rep("""            let mut peer: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for p in &files {
                let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                if name == own_tomb || !name.ends_with(&tomb_suffix) {
                    continue;
                }
                let Ok(lines) = lossy_lines(p) else {
                    rep.skipped_files += 1;
                    continue;
                };
                for l in lines {
                    if let Ok(x) = serde_json::from_str::<DropLine>(&l) {
                        peer.entry(x.id).or_default().push(x.by);
                    }
                }
            }""", """            let mut peer: BTreeMap<String, Vec<(String, u64)>> = BTreeMap::new();
            for p in &files {
                let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                if name == own_tomb || !name.ends_with(&tomb_suffix) {
                    continue;
                }
                let owner = &name[..name.len() - tomb_suffix.len()];
                let allowed = o.deleters.as_ref().map_or(true, |d| d.iter().any(|m| m == owner));
                let Ok(lines) = lossy_lines(p) else {
                    rep.skipped_files += 1;
                    continue;
                };
                for l in lines {
                    if let Ok(x) = serde_json::from_str::<DropLine>(&l) {
                        if !allowed {
                            rep.deletions_unauthorized += 1;
                        } else if expired(x.ts) {
                            rep.tombstones_expired += 1;
                        } else {
                            peer.entry(x.id).or_default().push((x.by, x.ts));
                        }
                    }
                }
            }""")
rep("""                match keepers.iter().find(|k| *k != id && present.contains(*k)) {
                    Some(k) => {
                        present.remove(id);
                        del.push(DropLine { id: id.clone(), by: k.clone() });
                    }""", """                match keepers.iter().find(|(k, _)| k != id && present.contains(k)) {
                    Some((k, ts)) => {
                        present.remove(id);
                        del.push(DropLine { id: id.clone(), by: k.clone(), ts: *ts });
                    }""")
rep("""    fn read_dropped(&self, ns: &str)""", """    /// B40: drop expired tombstones (those with an origin timestamp past the TTL) from `tombstones.jsonl` and `dropped_by.jsonl`. Ids without a timestamped record never expire.
    fn expire_local_tombstones(&self, ns: &str, expired: &dyn Fn(u64) -> bool) -> R<usize> {
        let d = self.kdir(ns);
        if !d.join("dropped_by.jsonl").exists() {
            return Ok(0);
        }
        let _g = self.lock()?;
        let all = self.read_dropped(ns);
        let live: HashSet<&str> = all.iter().filter(|x| !expired(x.ts)).map(|x| x.id.as_str()).collect();
        let dead: HashSet<&str> = all.iter().filter(|x| expired(x.ts) && !live.contains(x.id.as_str())).map(|x| x.id.as_str()).collect();
        if dead.is_empty() {
            return Ok(0);
        }
        let mut bb = String::new();
        for x in all.iter().filter(|x| !dead.contains(x.id.as_str())) {
            bb.push_str(&serde_json::to_string(x).unwrap());
            bb.push('\\n');
        }
        let mut tb = String::new();
        for l in lossy_lines(&d.join("tombstones.jsonl")).unwrap_or_default() {
            if let Ok(id) = serde_json::from_str::<String>(&l) {
                if !dead.contains(id.as_str()) {
                    tb.push_str(&serde_json::to_string(&id).unwrap());
                    tb.push('\\n');
                }
            }
        }
        replace_durable(&d.join("tombstones.jsonl.tmp"), &d.join("tombstones.jsonl"), tb.as_bytes())?;
        replace_durable(&d.join("dropped_by.jsonl.tmp"), &d.join("dropped_by.jsonl"), bb.as_bytes())?;
        Ok(dead.len())
    }
    fn read_dropped(&self, ns: &str)""", 1)
open(p, 'w', encoding='utf-8').write(s)
