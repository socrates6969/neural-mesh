//! Opt-in exact-duplicate suppression on ingest (R4-B3 / H6 proposal P3, productionised in R6-B / H14).
//!
//! Keeps ONE row per identical item: an item is a duplicate if its normalised-text hash OR its (L2-normalised) vector
//! hash was already seen (each key configurable); the canonical row's multiplicity is incremented instead of storing
//! another row. "Exact" means identical after normalisation -- near-duplicates are never suppressed (stale-revision
//! detection by cosine was tried in g16 and failed).
//!
//! Measured effect (SYNTHETIC-until-validated; results/h6_*.md, results/h14_board_entry.md): at doc level with 30 % of
//! documents re-ingested as exact duplicates, F1@2 +0.025..+0.036 and rows -23 %, context chars unchanged at fixed k
//! (it frees retrieval slots, it does not shrink the context); in the g16 closed-loop VOTE store rows -90 % with H2000
//! TOST-equivalent in 3 of 4 policy x corpus cells and one inconclusive (-0.0085, CI [-0.018, +0.001]).
//! Without duplicates it is a no-op (nothing is ever suppressed), so it is harmless on duplicate-free streams.
//!
//! Hashing is deterministic 2x64-bit FNV-1a (no `RandomState`), NOT collision-resistant against adversarial input.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::flatq::{FlatError, FlatIndex};

/// 128-bit content key (two independent 64-bit FNV-1a hashes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Key(pub u64, pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DupKind {
    Text,
    Vector,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// not seen before: store it
    New,
    /// identical to the stored item `of`: do not store, multiplicity incremented
    Duplicate { of: u64, by: DupKind },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DedupConfig {
    /// suppress on identical normalised text
    pub by_text: bool,
    /// suppress on identical L2-normalised vector
    pub by_vector: bool,
    /// text normalisation: Unicode lowercase
    pub fold_case: bool,
    /// text normalisation: treat every non-alphanumeric char as whitespace
    pub strip_punct: bool,
    /// vector key: zero this many low mantissa bits before hashing (0 = bit-exact; 8 tolerates last-ulp noise from re-embedding)
    pub vec_drop_bits: u32,
}

impl Default for DedupConfig {
    fn default() -> Self {
        DedupConfig { by_text: true, by_vector: true, fold_case: true, strip_punct: false, vec_drop_bits: 0 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DedupStats {
    pub seen: u64,
    pub stored: u64,
    pub suppressed_text: u64,
    pub suppressed_vector: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Entry {
    id: u64,
    text_key: Option<Key>,
    vec_key: Option<Key>,
    count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct State {
    cfg: DedupConfig,
    entries: Vec<Entry>,
    stats: DedupStats,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(from = "State", into = "State")]
pub struct Deduper {
    cfg: DedupConfig,
    entries: HashMap<u64, Entry>,
    text_seen: HashMap<Key, u64>,
    vec_seen: HashMap<Key, u64>,
    stats: DedupStats,
}

impl From<Deduper> for State {
    fn from(d: Deduper) -> State {
        let mut entries: Vec<Entry> = d.entries.into_values().collect();
        entries.sort_by_key(|e| e.id); // deterministic serialisation
        State { cfg: d.cfg, entries, stats: d.stats }
    }
}
impl From<State> for Deduper {
    fn from(s: State) -> Deduper {
        let mut d = Deduper { cfg: s.cfg, entries: HashMap::new(), text_seen: HashMap::new(), vec_seen: HashMap::new(), stats: s.stats };
        for e in s.entries {
            if let Some(k) = e.text_key {
                d.text_seen.insert(k, e.id);
            }
            if let Some(k) = e.vec_key {
                d.vec_seen.insert(k, e.id);
            }
            d.entries.insert(e.id, e);
        }
        d
    }
}

fn fnv(bytes: impl Iterator<Item = u8>, basis: u64) -> u64 {
    let mut h = basis;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    // murmur-style finaliser so that the two lanes decorrelate
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h
}
fn key_of(bytes: &[u8]) -> Key {
    Key(fnv(bytes.iter().copied(), 0xcbf2_9ce4_8422_2325), fnv(bytes.iter().copied().rev(), 0x8422_2325_cbf2_9ce4 ^ bytes.len() as u64))
}

/// Normalised form used for the text key (exposed for tests / callers that want to pre-normalise).
pub fn normalize_text(s: &str, cfg: &DedupConfig) -> String {
    let mut out = String::with_capacity(s.len());
    let mut last_space = true;
    let mut push = |c: char, out: &mut String| {
        if c.is_whitespace() || (cfg.strip_punct && !c.is_alphanumeric()) {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(c);
            last_space = false;
        }
    };
    for c in s.chars() {
        if cfg.fold_case {
            for lc in c.to_lowercase() {
                push(lc, &mut out);
            }
        } else {
            push(c, &mut out);
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

fn vec_key(v: &[f32], cfg: &DedupConfig) -> Option<Key> {
    if v.is_empty() || v.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mask: u32 = if cfg.vec_drop_bits >= 23 { !0x007f_ffff } else { !((1u32 << cfg.vec_drop_bits) - 1) };
    let mut bytes = Vec::with_capacity(4 * v.len() + 8);
    bytes.extend_from_slice(&(v.len() as u64).to_le_bytes());
    for &x in v {
        let y = if n > 0.0 { x / n } else { x };
        let y = if y == 0.0 { 0.0 } else { y }; // -0.0 -> +0.0
        bytes.extend_from_slice(&(y.to_bits() & mask).to_le_bytes());
    }
    Some(key_of(&bytes))
}

impl Deduper {
    pub fn new(cfg: DedupConfig) -> Self {
        Deduper { cfg, entries: HashMap::new(), text_seen: HashMap::new(), vec_seen: HashMap::new(), stats: DedupStats::default() }
    }
    pub fn config(&self) -> &DedupConfig {
        &self.cfg
    }
    pub fn stats(&self) -> DedupStats {
        self.stats
    }
    /// Stored (canonical) items.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Multiplicity of canonical item `id` (itself + suppressed copies), 0 if unknown.
    pub fn count(&self, id: u64) -> u64 {
        self.entries.get(&id).map_or(0, |e| e.count)
    }

    fn keys(&self, text: Option<&str>, vector: Option<&[f32]>) -> (Option<Key>, Option<Key>) {
        let tk = if self.cfg.by_text { text.map(|t| key_of(normalize_text(t, &self.cfg).as_bytes())) } else { None };
        let vk = if self.cfg.by_vector { vector.and_then(|v| vec_key(v, &self.cfg)) } else { None };
        (tk, vk)
    }

    /// Would this item be suppressed? (no state change)
    pub fn would_suppress(&self, text: Option<&str>, vector: Option<&[f32]>) -> Option<(u64, DupKind)> {
        let (tk, vk) = self.keys(text, vector);
        if let Some(&of) = tk.as_ref().and_then(|k| self.text_seen.get(k)) {
            return Some((of, DupKind::Text));
        }
        if let Some(&of) = vk.as_ref().and_then(|k| self.vec_seen.get(k)) {
            return Some((of, DupKind::Vector));
        }
        None
    }

    /// Ingest-time check. `New` registers `id` as a canonical item (caller must then store it); `Duplicate` increments the
    /// canonical item's multiplicity and registers nothing (caller must NOT store it). Re-ingesting an existing `id` replaces it.
    pub fn check(&mut self, id: u64, text: Option<&str>, vector: Option<&[f32]>) -> Verdict {
        self.stats.seen += 1;
        if self.entries.contains_key(&id) {
            self.forget(id);
        }
        let (tk, vk) = self.keys(text, vector);
        if let Some(&of) = tk.as_ref().and_then(|k| self.text_seen.get(k)) {
            self.entries.get_mut(&of).unwrap().count += 1;
            self.stats.suppressed_text += 1;
            return Verdict::Duplicate { of, by: DupKind::Text };
        }
        if let Some(&of) = vk.as_ref().and_then(|k| self.vec_seen.get(k)) {
            self.entries.get_mut(&of).unwrap().count += 1;
            self.stats.suppressed_vector += 1;
            return Verdict::Duplicate { of, by: DupKind::Vector };
        }
        if let Some(k) = tk {
            self.text_seen.insert(k, id);
        }
        if let Some(k) = vk {
            self.vec_seen.insert(k, id);
        }
        self.entries.insert(id, Entry { id, text_key: tk, vec_key: vk, count: 1 });
        self.stats.stored += 1;
        Verdict::New
    }

    /// Forget canonical item `id` (e.g. its row was removed from the store) so that a later identical item is stored again.
    pub fn forget(&mut self, id: u64) -> bool {
        match self.entries.remove(&id) {
            Some(e) => {
                if let Some(k) = e.text_key {
                    self.text_seen.remove(&k);
                }
                if let Some(k) = e.vec_key {
                    self.vec_seen.remove(&k);
                }
                true
            }
            None => false,
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("Deduper serialises")
    }
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

/// Ingest helper for a [`FlatIndex`]: suppresses exact duplicates, stores new items. Returns the verdict.
pub fn ingest(dd: &mut Deduper, ix: &mut FlatIndex, id: u64, text: Option<&str>, vector: &[f32]) -> Result<Verdict, FlatError> {
    // validate first so a rejected vector does not register keys
    if vector.len() != ix.dim() {
        return Err(FlatError::DimMismatch { expected: ix.dim(), got: vector.len() });
    }
    if vector.iter().any(|x| !x.is_finite()) {
        return Err(FlatError::NonFinite);
    }
    let v = dd.check(id, text, Some(vector));
    if v == Verdict::New {
        ix.add(id, vector)?;
    }
    Ok(v)
}

/// Remove `id` from the index and the deduper together.
pub fn remove(dd: &mut Deduper, ix: &mut FlatIndex, id: u64) -> bool {
    let a = dd.forget(id);
    let b = ix.remove(id);
    a || b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatq::{Metric, Tier};

    #[test]
    fn text_normalisation_and_text_dedup() {
        let mut d = Deduper::new(DedupConfig::default());
        assert_eq!(d.check(1, Some("Reset  my\tPassword"), None), Verdict::New);
        assert_eq!(d.check(2, Some("reset my password "), None), Verdict::Duplicate { of: 1, by: DupKind::Text });
        assert_eq!(d.check(3, Some("reset my password!"), None), Verdict::New, "punctuation is significant by default");
        assert_eq!(d.count(1), 2);
        assert_eq!(d.count(3), 1);
        assert_eq!(d.len(), 2);
        // case-sensitive config
        let mut d = Deduper::new(DedupConfig { fold_case: false, ..Default::default() });
        assert_eq!(d.check(1, Some("A b"), None), Verdict::New);
        assert_eq!(d.check(2, Some("a b"), None), Verdict::New);
        // strip punctuation config
        let mut d = Deduper::new(DedupConfig { strip_punct: true, ..Default::default() });
        assert_eq!(d.check(1, Some("reset my password"), None), Verdict::New);
        assert_eq!(d.check(2, Some("Reset, my password!"), None), Verdict::Duplicate { of: 1, by: DupKind::Text });
        assert_eq!(normalize_text("  A\n\tB  ", &DedupConfig::default()), "a b");
    }

    #[test]
    fn vector_dedup_is_scale_invariant_but_not_near_duplicate() {
        let mut d = Deduper::new(DedupConfig { by_text: false, ..Default::default() });
        let v = vec![0.3f32, -0.4, 0.5, 0.0];
        assert_eq!(d.check(1, None, Some(&v)), Verdict::New);
        let scaled: Vec<f32> = v.iter().map(|x| x * 2.0).collect(); // power of two: exact in fp
        assert_eq!(d.check(2, None, Some(&scaled)), Verdict::Duplicate { of: 1, by: DupKind::Vector });
        // a perturbed copy (1e-3) is NOT a duplicate (exact means exact)
        let mut p = v.clone();
        p[0] += 1e-3;
        assert_eq!(d.check(3, None, Some(&p)), Verdict::New);
        // -0.0 == +0.0
        let mut z = v.clone();
        z[3] = -0.0;
        assert_eq!(d.check(4, None, Some(&z)), Verdict::Duplicate { of: 1, by: DupKind::Vector });
        // text ignored when by_text is off
        assert_eq!(d.check(5, Some("anything"), None), Verdict::New);
        // drop-bits tolerance
        let mut t = Deduper::new(DedupConfig { vec_drop_bits: 12, ..Default::default() });
        assert_eq!(t.check(1, None, Some(&[0.6, 0.8])), Verdict::New);
        assert_eq!(t.check(2, None, Some(&[0.6 + 1e-6, 0.8])), Verdict::Duplicate { of: 1, by: DupKind::Vector });
        assert_eq!(t.check(3, None, Some(&[0.61, 0.8])), Verdict::New);
        // non-finite vectors are ignored by the key (stored)
        assert_eq!(d.check(6, None, Some(&[f32::NAN, 0.0])), Verdict::New);
    }

    #[test]
    fn no_duplicates_means_no_suppression_and_stats_add_up() {
        let mut d = Deduper::new(DedupConfig::default());
        for i in 0..500u64 {
            let v = [i as f32 + 1.0, (i * i) as f32 + 0.5, 1.0];
            assert_eq!(d.check(i, Some(&format!("doc number {i}")), Some(&v)), Verdict::New);
        }
        let s = d.stats();
        assert_eq!((s.seen, s.stored, s.suppressed_text, s.suppressed_vector), (500, 500, 0, 0));
        // re-ingest the first 100 exactly
        for i in 0..100u64 {
            let v = [i as f32 + 1.0, (i * i) as f32 + 0.5, 1.0];
            assert!(matches!(d.check(1000 + i, Some(&format!("doc number {i}")), Some(&v)), Verdict::Duplicate { of, .. } if of == i));
        }
        let s = d.stats();
        assert_eq!((s.seen, s.stored, s.suppressed_text + s.suppressed_vector), (600, 500, 100));
        assert_eq!(d.count(7), 2);
        assert!(d.would_suppress(Some("doc number 7"), None).is_some());
        assert!(d.would_suppress(Some("doc number 7000"), None).is_none());
    }

    #[test]
    fn forget_replace_and_json_roundtrip() {
        let mut d = Deduper::new(DedupConfig::default());
        assert_eq!(d.check(1, Some("alpha"), Some(&[1.0, 0.0])), Verdict::New);
        assert_eq!(d.check(2, Some("beta"), Some(&[0.0, 1.0])), Verdict::New);
        let j = d.to_json();
        let mut d2 = Deduper::from_json(&j).unwrap();
        assert_eq!(d2.check(3, Some("ALPHA"), None), Verdict::Duplicate { of: 1, by: DupKind::Text });
        assert_eq!(d2.to_json().len() > 0, true);
        assert_eq!(d2.count(1), 2);
        // forget the canonical row: the same content is stored again
        assert!(d2.forget(1));
        assert!(!d2.forget(1));
        assert_eq!(d2.check(4, Some("alpha"), Some(&[1.0, 0.0])), Verdict::New);
        // re-ingesting an existing id with the same content replaces, not duplicates itself
        assert_eq!(d2.check(4, Some("alpha"), Some(&[1.0, 0.0])), Verdict::New);
        assert_eq!(d2.len(), 2);
        // deterministic serialisation
        assert_eq!(d.to_json(), Deduper::from_json(&d.to_json()).unwrap().to_json());
    }

    #[test]
    fn ingest_into_flat_index_keeps_one_row_per_identical_item() {
        let d = 8;
        let mut ix = FlatIndex::new(d, Tier::BlockFp16, Metric::Cosine);
        let mut dd = Deduper::new(DedupConfig::default());
        let vecs: Vec<Vec<f32>> = (0..20).map(|i| (0..d).map(|j| ((i * 7 + j * 3) % 11) as f32 - 5.0 + 0.1 * i as f32).collect()).collect();
        let mut id = 0u64;
        for round in 0..3 {
            for (i, v) in vecs.iter().enumerate() {
                let r = ingest(&mut dd, &mut ix, id, Some(&format!("doc {i}")), v).unwrap();
                assert_eq!(r == Verdict::New, round == 0, "round {round} item {i}");
                id += 1;
            }
        }
        assert_eq!(ix.len(), 20);
        assert_eq!(dd.len(), 20);
        assert_eq!(dd.count(0), 3);
        // an exact duplicate never shows up as a second hit
        let hits = ix.search(&vecs[0], 5).unwrap();
        assert_eq!(hits[0].0, 0);
        assert_eq!(hits.iter().filter(|h| h.0 >= 20).count(), 0);
        // validation errors do not register keys
        assert!(ingest(&mut dd, &mut ix, 999, Some("bad"), &[1.0; 3]).is_err());
        assert!(dd.would_suppress(Some("bad"), None).is_none());
        // remove from both: the content is accepted again
        assert!(remove(&mut dd, &mut ix, 0));
        assert_eq!(ingest(&mut dd, &mut ix, 1000, Some("doc 0"), &vecs[0]).unwrap(), Verdict::New);
        assert_eq!(ix.len(), 20);
    }
}
