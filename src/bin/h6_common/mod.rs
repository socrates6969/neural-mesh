//! h6 shared pieces (R4-B3, memory-reduction track): quantised-vector decoders, a copy of the g16 closed-loop engine with a PLUGGABLE
//! store (flat / exact-duplicate-suppressed / anchor-summary rows, quantised or exact scoring, optional exact rerank) and the TOST helpers.
//! Included with `#[path = "g1_common/mod.rs"] mod common; #[path = "g16_common/mod.rs"] mod g16; #[path = "h6_common/mod.rs"] mod h6;`.
//! SYNTHETIC-until-validated: the closed loop is the g16 seeded world on the g1 fixtures. Nothing here changes any default.
#![allow(dead_code)]

use std::collections::HashMap;

use neural_mesh::router::Rng;
use neural_mesh::stats::inc_beta;
use turbovec::TurboQuantIndex;

use crate::common::*;
use crate::g16::{make_front, stream, Base, Gate, PolCfg, Pool, Spec, HORIZONS, NA, NT};

// ------------------------------------------------------------------------------------------------
// scalar quantisers (decode = what a scan against the stored codes effectively sees)

pub fn to_f16_bits(f: f32) -> u16 {
    let b = f.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut h = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if rem > half || (rem == half && (h & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let mut h = sign | ((e as u16) << 10) | ((man >> 13) as u16);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h
}
pub fn from_f16_bits(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    if exp == 0 {
        if man == 0 {
            f32::from_bits(sign)
        } else {
            let v = (man as f32) * 2f32.powi(-24);
            if sign != 0 { -v } else { v }
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f80_0000 | (man << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Q {
    F32,
    F16,
    /// 32-value blocks, i16 mantissa, f32 shared scale per block (816 B/vec at d=384)
    BlockFp16,
    /// per-vector scale, i8 payload (388 B/vec at d=384), f32 query
    Int8,
    /// as Int8 but the QUERY is int8 too (the g15 scan kernel)
    Int8QQ,
}
impl Q {
    pub fn name(self) -> &'static str {
        match self {
            Q::F32 => "f32",
            Q::F16 => "f16",
            Q::BlockFp16 => "block-fp16",
            Q::Int8 => "int8",
            Q::Int8QQ => "int8(q8)",
        }
    }
    /// nominal stored bytes per vector of dim d (payload + scales)
    pub fn bytes(self, d: usize) -> usize {
        match self {
            Q::F32 => 4 * d,
            Q::F16 => 2 * d,
            Q::BlockFp16 => 2 * d + 4 * (d / 32),
            Q::Int8 | Q::Int8QQ => d + 4,
        }
    }
}
pub fn decode_row(q: Q, v: &[f32]) -> Vec<f32> {
    match q {
        Q::F32 => v.to_vec(),
        Q::F16 => v.iter().map(|&x| from_f16_bits(to_f16_bits(x))).collect(),
        Q::BlockFp16 => {
            let mut o = Vec::with_capacity(v.len());
            for blk in v.chunks(32) {
                let mx = blk.iter().fold(0f32, |m, &x| m.max(x.abs()));
                let sc = if mx > 0.0 { mx / 32767.0 } else { 1.0 };
                for &x in blk {
                    o.push((x / sc).round().clamp(-32767.0, 32767.0) * sc);
                }
            }
            o
        }
        Q::Int8 | Q::Int8QQ => {
            let mx = v.iter().fold(0f32, |m, &x| m.max(x.abs()));
            let sc = if mx > 0.0 { mx / 127.0 } else { 1.0 };
            v.iter().map(|&x| (x / sc).round().clamp(-127.0, 127.0) * sc).collect()
        }
    }
}
pub fn dotf(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
/// nq x ns score matrix, row i = query i against stored vectors: dot(query_i (f32, or int8 for Int8QQ), decode(stored_j)).
pub fn quant_matrix(q: Q, queries: &[f32], store: &[f32], d: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let ns = store.len() / d;
    let dec: Vec<f32> = store.chunks_exact(d).flat_map(|r| decode_row(q, r)).collect();
    let qs: Vec<f32> = if q == Q::Int8QQ { queries.chunks_exact(d).flat_map(|r| decode_row(Q::Int8, r)).collect() } else { queries.to_vec() };
    let nq = queries.len() / d;
    let mut out = vec![0f32; nq * ns];
    out.par_chunks_mut(ns).enumerate().for_each(|(i, row)| {
        let a = &qs[i * d..(i + 1) * d];
        for (j, r) in row.iter_mut().enumerate() {
            *r = dotf(a, &dec[j * d..(j + 1) * d]);
        }
    });
    out
}
/// turbovec 2/3/4-bit approximate scores (nq x ns), f32 queries vs codes of `store`.
pub fn tv_matrix(bits: usize, queries: &[f32], store: &[f32], d: usize) -> Vec<f32> {
    let ns = store.len() / d;
    let nq = queries.len() / d;
    let mut ix = TurboQuantIndex::new(d, bits).expect("turbovec");
    for ch in store.chunks(d * 2048) {
        ix.add(ch);
    }
    ix.prepare();
    let mut out = vec![0f32; nq * ns];
    // search in query blocks to bound the temporary result size
    for (b, qb) in queries.chunks(d * 256).enumerate() {
        let r = ix.search(qb, ns);
        for qi in 0..qb.len() / d {
            let sc = r.scores_for_query(qi);
            let ix_ = r.indices_for_query(qi);
            let row = &mut out[(b * 256 + qi) * ns..(b * 256 + qi + 1) * ns];
            for (s, &slot) in sc.iter().zip(ix_) {
                if slot >= 0 && (slot as usize) < ns {
                    row[slot as usize] = *s;
                }
            }
        }
    }
    out
}
/// nominal turbovec bytes/vector (codes only; norms/rotation excluded, as in g15's "nominal")
pub fn tv_bytes(bits: usize, d: usize) -> usize {
    (d * bits).div_ceil(8) + 4
}

// ------------------------------------------------------------------------------------------------
// store variants for the closed loop

/// one dense scorer: approximate matrix (what the compressed store scans) and optional exact matrix for rerank of the top `factor*k`
pub struct Var {
    pub name: String,
    pub approx: usize,
    pub rerank: Option<(usize, usize)>,
    /// nominal dense bytes per stored vector (RAM)
    pub bytes: usize,
    /// bytes per vector of the exact copy needed for rerank (0 if none; lives on disk or RAM, NOT counted in `bytes`)
    pub exact_bytes: usize,
}
pub struct Sims6 {
    pub nq: usize,
    pub d: usize,
    /// tf-idf (spell) matrix, shared by all variants (the hybrid .6 dense + .4 tfidf of g16 SIM_MINI_H)
    pub tf: Vec<f32>,
    pub dense: Vec<Vec<f32>>,
    pub names: Vec<String>,
}
pub const W_DENSE: f32 = 0.6;
pub const W_TF: f32 = 0.4;
/// sparse tf-idf record bytes (g16 board entry: ~320 B) + meta (anchor id, counts)
pub const SPARSE_BYTES: usize = 320;
pub const META_BYTES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StoreKind {
    /// one row per task (g16 default)
    Flat,
    /// exact duplicates (identical query vector) are not stored again
    Dedup,
    /// one row per anchor: score(q, row) = mean over the anchor's members of the hybrid similarity (== dot with the un-normalised centroid; the dense and sparse parts are linear)
    Summary,
}

#[derive(Clone)]
struct AStat {
    s: [f64; NA],
    n: [f64; NA],
    cf: [u32; NA],
    t: usize,
}
impl AStat {
    fn new(t: usize) -> Self {
        AStat { s: [0.0; NA], n: [0.0; NA], cf: [0; NA], t }
    }
}

#[derive(Clone, Default)]
pub struct Out6 {
    pub h: [f64; 4],
    pub late: f64,
    pub rows: usize,
    pub tasks: usize,
    /// mean fraction of the exact-f32 neighbour set reproduced (NaN when the arm is exact or not flat)
    pub nb_recall: f64,
    pub nb_mean: f64,
    pub vote_frac: f64,
    /// exact-copy row reads per task (rerank)
    pub reads_per_task: f64,
}

impl Sims6 {
    #[inline]
    fn s(&self, key: usize, i: usize, j: usize) -> f32 {
        W_DENSE * self.dense[key][i * self.nq + j] + W_TF * self.tf[i * self.nq + j]
    }
    pub fn key(&self, n: &str) -> usize {
        self.names.iter().position(|x| x == n).unwrap_or_else(|| panic!("dense variant {n}"))
    }
}

fn sel_top(mut v: Vec<(f32, usize)>, k: usize) -> Vec<(f32, usize)> {
    if v.len() > k {
        v.select_nth_unstable_by(k, |a, b| b.0.partial_cmp(&a.0).unwrap());
        v.truncate(k);
    }
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    v
}

/// The g16 `run` engine with a pluggable store. Policy semantics are IDENTICAL to g16::run for StoreKind::Flat + exact key
/// (checked by `--check`: bit-exact vs g16 on the same pool).
pub fn run6(cfg: &PolCfg, sp: &Spec, seed: u64, c: &Corpus, ss: &Sims6, var: &Var, kind: StoreKind, diag: bool) -> Out6 {
    let st = stream(c, seed, sp);
    let w = &st.w;
    let cands: Vec<usize> = (0..NA).collect();
    let mut fr = make_front(&cfg.front, w);
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut out = Out6::default();
    let mut cum = 0.0;
    let mut pes: Vec<f32> = Vec::with_capacity(NT);
    // store state: rows[i] = query index of the row (Flat/Dedup); members[i] = query indices merged into the row (Summary)
    let mut rows: Vec<usize> = vec![];
    let mut members: Vec<Vec<usize>> = vec![];
    let mut anchor_of: Vec<u64> = vec![];
    let mut rowmap: HashMap<usize, usize> = HashMap::new(); // Dedup: qi -> row ; Summary: anchor -> row
    let mut astats: HashMap<u64, AStat> = HashMap::new();
    let mut gate = [0.0f64; 4];
    let mut probe = 0usize;
    let (mut fref, mut fref_n) = (0.5f64, 0.0f64);
    let (mut n_vote, mut n_cand) = (0usize, 0usize);
    let (mut gs, mut gn) = ([0.0f64; NA], [0.0f64; NA]);
    let exact_key = ss.key("f32");
    let (mut nbr_sum, mut nbr_n, mut nb_cnt, mut reads) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let nrows_of = |rows: &Vec<usize>, members: &Vec<Vec<usize>>| if kind == StoreKind::Summary { members.len() } else { rows.len() };
    for t in 0..w.len() {
        let qi = st.q[t];
        let task0 = &w.task(t).task;
        let mut task = task0.clone();
        let mut by_vote = false;
        let mut anchor = t as u64;
        // ---- neighbours
        let nrows = nrows_of(&rows, &members);
        let score_row = |key: usize, r: usize| -> f32 {
            if kind == StoreKind::Summary {
                let m = &members[r];
                m.iter().map(|&j| ss.s(key, qi, j)).sum::<f32>() / m.len() as f32
            } else {
                ss.s(key, qi, rows[r])
            }
        };
        let nb: Vec<(f32, usize)> = {
            let k = cfg.k;
            let mut v: Vec<(f32, usize)>;
            match var.rerank {
                None => {
                    v = (0..nrows).map(|r| (score_row(var.approx, r), r)).filter(|x| x.0 >= cfg.thr).collect();
                    v = sel_top(v, k);
                }
                Some((ex, f)) => {
                    let mut all: Vec<(f32, usize)> = (0..nrows).map(|r| (score_row(var.approx, r), r)).collect();
                    let rk = (k * f).min(all.len());
                    if all.len() > rk && rk > 0 {
                        all.select_nth_unstable_by(rk, |a, b| b.0.partial_cmp(&a.0).unwrap());
                        all.truncate(rk);
                    }
                    reads += all.len() as f64;
                    v = all.into_iter().map(|(_, r)| (score_row(ex, r), r)).filter(|x| x.0 >= cfg.thr).collect();
                    v = sel_top(v, k);
                }
            }
            if let (Some(dd), Some(top)) = (cfg.adapt, v.first().map(|x| x.0)) {
                v.retain(|x| x.0 >= top - dd);
            }
            v
        };
        if diag && kind == StoreKind::Flat && (var.approx != exact_key || var.rerank.is_some()) {
            let ev = {
                let v: Vec<(f32, usize)> = (0..nrows).map(|r| (score_row(exact_key, r), r)).filter(|x| x.0 >= cfg.thr).collect();
                let mut v = sel_top(v, cfg.k);
                if let (Some(dd), Some(top)) = (cfg.adapt, v.first().map(|x| x.0)) {
                    v.retain(|x| x.0 >= top - dd);
                }
                v
            };
            if !ev.is_empty() {
                let hit = ev.iter().filter(|e| nb.iter().any(|b| b.1 == e.1)).count();
                nbr_sum += hit as f64 / ev.len() as f64;
                nbr_n += 1.0;
            }
        }
        nb_cnt += nb.len() as f64;
        let mut votes: Vec<(u64, f32)> = vec![];
        let mut tot = 0.0f32;
        for &(sc, i) in &nb {
            tot += sc;
            match votes.iter_mut().find(|v| v.0 == anchor_of[i]) {
                Some(v) => v.1 += sc,
                None => votes.push((anchor_of[i], sc)),
            }
        }
        let found = votes.iter().fold(None, |b: Option<(u64, f32)>, &v| match b { Some(x) if x.1 >= v.1 => b, _ => Some(v) });
        if let Some((an, _)) = found {
            anchor = an;
        }
        let mut best: Option<(usize, f64, f64, f64)> = None;
        if let Some((an, _)) = found {
            if let Some(row) = astats.get(&an) {
                for a in 0..NA {
                    let (s, n) = (row.s[a], row.n[a]);
                    if n < cfg.min_n as f64 - 1e-9 {
                        continue;
                    }
                    if cfg.kappa > 0.0 {
                        let g = (gs[a] + 1.0) / (gn[a] + 2.0);
                        let (al, be) = (cfg.kappa * g + s, cfg.kappa * (1.0 - g) + (n - s));
                        let m = al / (al + be);
                        if best.is_none_or(|b| m > b.1) {
                            best = Some((a, m, al - 1.0, al + be - 2.0));
                        }
                    } else {
                        let m = (1.0 + s) / (2.0 + n);
                        if m >= 0.7 && best.is_none_or(|b| m > b.1) {
                            best = Some((a, m, s, n));
                        }
                    }
                }
            }
        }
        if best.is_some() {
            n_cand += 1;
        }
        let g1_open = best.is_none() || {
            let rate_ok = gate[1] < cfg.g1_min || gate[3] < cfg.g1_min || gate[0] / gate[1] >= gate[2] / gate[3] - cfg.g1_margin;
            if rate_ok {
                true
            } else {
                probe += 1;
                if probe >= cfg.probe0 {
                    probe = 0;
                    true
                } else {
                    false
                }
            }
        };
        let mut open = false;
        if let Some((_, m, s, n)) = best {
            open = match &cfg.gate {
                Gate::Off => true,
                Gate::G1 => g1_open,
                Gate::Post { delta, gamma, floor } => {
                    let conf = 1.0 - inc_beta(1.0 + s, 1.0 + (n - s), (fref + delta).clamp(1e-6, 1.0 - 1e-6));
                    conf >= *gamma && m >= *floor
                }
                Gate::Learned { .. } => false,
            };
            if open && cfg.and_g1 && !matches!(cfg.gate, Gate::G1) {
                open = g1_open;
            }
        }
        let inj_ok = open || cfg.inj_always;
        if found.is_some() && inj_ok {
            task.features.push(format!("kn:{anchor}"));
        }
        let a = match best {
            Some((a, ..)) if open => {
                by_vote = true;
                n_vote += 1;
                a
            }
            _ => fr.select(&task, &cands, &mut rng),
        };
        let pe = st.p(t, a);
        let s = if w.task(t).u < pe { 1.0 } else { 0.0 };
        fr.update(&task, a, s);
        gs[a] += s;
        gn[a] += 1.0;
        // ---- store update
        match kind {
            StoreKind::Flat => {
                rows.push(qi);
                anchor_of.push(anchor);
            }
            StoreKind::Dedup => {
                if let std::collections::hash_map::Entry::Vacant(e) = rowmap.entry(qi) {
                    e.insert(rows.len());
                    rows.push(qi);
                    anchor_of.push(anchor);
                }
            }
            StoreKind::Summary => match rowmap.get(&(anchor as usize)) {
                Some(&r) => members[r].push(qi),
                None => {
                    rowmap.insert(anchor as usize, members.len());
                    members.push(vec![qi]);
                    anchor_of.push(anchor);
                }
            },
        }
        let row = astats.entry(anchor).or_insert_with(|| AStat::new(t));
        row.s[a] += s;
        row.n[a] += 1.0;
        if s > 0.5 {
            row.cf[a] = 0;
        } else {
            row.cf[a] += 1;
        }
        if by_vote {
            gate[0] += s;
            gate[1] += 1.0;
        } else {
            gate[2] += s;
            gate[3] += 1.0;
            fref_n += 1.0;
            let al = (1.0 / (fref_n + 1.0)).max(0.03);
            fref += al * (s - fref);
        }
        pes.push(pe as f32);
        cum += pe;
        for (i, &hz) in HORIZONS.iter().enumerate() {
            if t + 1 == hz {
                out.h[i] = cum / hz as f64;
            }
        }
    }
    out.late = pes[1000..].iter().map(|&x| x as f64).sum::<f64>() / (NT - 1000) as f64;
    out.rows = nrows_of(&rows, &members);
    out.tasks = w.len();
    out.nb_recall = if nbr_n > 0.0 { nbr_sum / nbr_n } else { f64::NAN };
    out.nb_mean = nb_cnt / NT as f64;
    out.vote_frac = n_vote as f64 / NT as f64;
    out.reads_per_task = reads / NT as f64;
    let _ = n_cand;
    out
}

/// per-record bytes of a store row for a given dense bytes figure
pub fn row_bytes(dense: usize) -> usize {
    dense + SPARSE_BYTES + META_BYTES
}

/// Build the Sims6 for a corpus: dense variants for the MiniLM spell-corrected query vectors + the shared tf-idf matrix from the g16 pool.
pub fn build_sims6(c: &Corpus, pool: &Pool, specs: &[(String, Option<Q>, Option<usize>)]) -> Sims6 {
    let m = load_model(c, "mini");
    let d = m.dim;
    let nq = c.queries.len();
    let v: Vec<f32> = (0..nq).flat_map(|i| m.qvec(QV::Spell, i).to_vec()).collect();
    let tf = pool.mats[pool.mid("tfidf_s")].clone();
    let mut dense = vec![];
    let mut names = vec![];
    for (n, q, bits) in specs {
        let mat = match (q, bits) {
            (Some(q), None) => quant_matrix(*q, &v, &v, d),
            (None, Some(b)) => tv_matrix(*b, &v, &v, d),
            _ => unreachable!(),
        };
        dense.push(mat);
        names.push(n.clone());
    }
    Sims6 { nq, d, tf, dense, names }
}

// ------------------------------------------------------------------------------------------------
// config loading (frozen picks of g16)

use serde_json::Value;
pub fn cfg_from(v: &Value) -> PolCfg {
    PolCfg {
        name: v["name"].as_str().unwrap().into(),
        base: Base::Vote,
        front: match v["front"].as_str().unwrap() { "Warm" => Base::Warm, "HebR" => Base::HebR, _ => Base::Guard },
        inj_conf: None,
        sim: v["sim"].as_u64().unwrap() as usize,
        k: v["k"].as_u64().unwrap() as usize,
        thr: v["thr"].as_f64().unwrap() as f32,
        min_n: v["min_n"].as_u64().unwrap() as usize,
        gate: match v["gate"]["t"].as_str().unwrap() {
            "off" => Gate::Off,
            "g1" => Gate::G1,
            "post" => Gate::Post { delta: v["gate"]["delta"].as_f64().unwrap(), gamma: v["gate"]["gamma"].as_f64().unwrap(), floor: v["gate"]["floor"].as_f64().unwrap() },
            _ => panic!("learned gate not supported in h6"),
        },
        and_g1: v["and_g1"].as_bool().unwrap(),
        id: None,
        inj_always: v["inj_always"].as_bool().unwrap(),
        kappa: v["kappa"].as_f64().unwrap_or(0.0),
        g1_min: v["g1_min"].as_f64().unwrap_or(6.0),
        g1_margin: v["g1_margin"].as_f64().unwrap_or(0.02),
        probe0: v["probe0"].as_u64().unwrap_or(25) as usize,
        backoff: false,
        adapt: v["adapt"].as_f64().map(|x| x as f32),
        decay: 1.0,
        reset_fail: 0,
    }
}
