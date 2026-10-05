//! k1 common: KNOWLEDGE-STORE growth / compaction / freshness simulator on the REAL-text h10/h13 fixtures (worlds A and B).
//!
//! LABELS. REAL: paragraphs, their src/file/section structure (structural keys), queries (section titles / held-out sentences), MiniLM vectors,
//! BM25 over the real tokens, topic clusters. SYNTHETIC (all labelled in the reports): the EDITED versions (old revisions, near-duplicates; generated
//! by py/k1_prep.py from real paragraphs and embedded offline with MiniLM), the exact re-ingestions, the noise records (real paragraphs of the OTHER
//! world), the arrival order / revision schedule, the query popularity (Zipf s=.8), and every closed-loop outcome (competence tables, context effect).
//! Included with `#[path = "k1_common/mod.rs"] mod common;`.
#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};

use neural_mesh::baselines::{EpsGreedyRouter, KeywordRouter, PooledThompsonRouter};
use neural_mesh::coupling::heb_robust_cfg;
use neural_mesh::dedup::{DedupConfig, Deduper, Verdict};
use neural_mesh::flatq::Tier;
use neural_mesh::hebbian::HebbianRouter;
use neural_mesh::hybrid::tokens;
use neural_mesh::prior::{Evidence, GuardConfig, Guarded, Warm};
use neural_mesh::router::{Rng, Router, Task};
use neural_mesh::stats::{holm_adjust, inc_beta, paired_diff, paired_t_pvalue, summarize, t_two_sided_p};
use rayon::prelude::*;
use serde_json::Value;

pub const DIM: usize = 384;
pub const K_BASE: u8 = 0;
pub const K_OLD: u8 = 1;
pub const K_NEAR: u8 = 2;
pub const K_NOISE: u8 = 3;
pub const ALPHA: f32 = 0.7; // h13 frozen z-fusion weight (dense .7 / bm25 .3)
pub const NEP: usize = 8;
pub const CKPT_EP: [usize; 4] = [2, 4, 6, 8]; // evaluation after these epochs (25/50/75/100 % of the stream)
pub const NEVAL: usize = 400; // eval queries per set (uniform / popularity) per checkpoint
pub const NUSE: usize = 200; // usage queries per epoch (popularity-weighted), feed value/LRU eviction
pub const NT: usize = 2000; // closed-loop tasks
pub const META_B: usize = 24; // per row: id, key hash, multiplicity, revision, flags
pub const TOPK: usize = 10;
pub const ZIPF_S: f64 = 0.8;
pub const P_OLD_ACTIVE: f64 = 0.7;
pub const P_NEAR_ACTIVE: f64 = 0.75;
pub const P_NOISE_ACTIVE: f64 = 0.8;
pub const P_EXACT: f64 = 0.12;
pub const P_LATE_OLD: f64 = 0.10;

// =============================================================================================== world

pub struct Query {
    pub fam: u8, // 0 = H (section title), 1 = S (held-out sentence)
    pub rel: Vec<u32>,
    pub tid: usize,
}

pub struct WorldK {
    pub tag: String,
    pub n: usize,
    pub nu: usize,
    pub emb: Vec<f32>,
    pub texts: Vec<String>,
    pub kind: Vec<u8>,
    pub canon: Vec<u32>,
    pub depth: Vec<u8>,
    pub key: Vec<u32>, // structural key id (interned src/file#ordinal) ; noise: unique
    pub cl: Vec<u8>,
    pub keyspace: usize,
    pub unit_of: Vec<u32>, // per base doc
    pub nunits: usize,
    pub old_of: Vec<[i32; 2]>,
    pub near_of: Vec<Vec<u32>>,
    pub unit_old: Vec<Vec<u32>>, // per unit: member docs having old variants (empty if none)
    pub unit_deep: Vec<bool>,
    pub noise_start: usize,
    pub qs: Vec<Query>,
    pub qvec: Vec<f32>,
    pub qtext: Vec<String>,
    pub sparse_b: Vec<u32>,
    pub s32: Vec<f32>,
    pub s16: Vec<f32>,
    pub s8: Vec<f32>,
    pub bm: Vec<f32>,
    pub nn: Vec<Vec<(u32, f32)>>,
    pub feat: HashMap<usize, Vec<String>>,
    pub kwtok12: Vec<(String, usize)>,
    pub cl12: Vec<u8>,
    /// stream control: 0 default, 1 no old versions, 2 no duplicates (exact+near), 3 high revision rate (every candidate unit active)
    pub ctl: u8,
}

fn rd(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"))
}
fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// per-vector symmetric int8 (mirrors flatq::Tier::Int8) -> dequantised
fn deq_i8(x: &[f32], out: &mut [f32]) {
    let mx = x.iter().fold(0f32, |m, &a| m.max(a.abs()));
    let sc = if mx > 0.0 { mx / 127.0 } else { 1.0 };
    for i in 0..x.len() {
        out[i] = (x[i] / sc).round().clamp(-127.0, 127.0) * sc;
    }
}
/// per-32-block int16 (mirrors flatq::Tier::BlockFp16) -> dequantised
fn deq_b16(x: &[f32], out: &mut [f32]) {
    for (xb, ob) in x.chunks(32).zip(out.chunks_mut(32)) {
        let mx = xb.iter().fold(0f32, |m, &a| m.max(a.abs()));
        let sc = if mx > 0.0 { mx / 32767.0 } else { 1.0 };
        for i in 0..xb.len() {
            ob[i] = (xb[i] / sc).round().clamp(-32767.0, 32767.0) * sc;
        }
    }
}

pub fn load_world(tag: &str, nn_min: f32) -> WorldK {
    let t0 = std::time::Instant::now();
    let meta: Value = serde_json::from_slice(&rd(&format!("results/k1_meta_{tag}.json"))).unwrap();
    let n = meta["ndoc"].as_u64().unwrap() as usize;
    let texts_all: Vec<String> = serde_json::from_slice(&rd(&format!("results/h10_texts_{tag}.json"))).unwrap();
    let emb_all = f32s(&rd(&format!("results/h10_emb_{tag}.f32")));
    let var_emb = f32s(&rd(&format!("results/k1_var_emb_{tag}.f32")));
    let var_texts: Vec<String> = meta["var_texts"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
    let noise_texts: Vec<String> = meta["noise_texts"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
    let variants = meta["variants"].as_array().unwrap();
    let nvar = variants.len();
    let nnoise = noise_texts.len();
    assert_eq!(var_emb.len(), (nvar + nnoise) * DIM);
    let nu = n + nvar + nnoise;
    let mut emb = Vec::with_capacity(nu * DIM);
    emb.extend_from_slice(&emb_all[..n * DIM]);
    emb.extend_from_slice(&var_emb);
    let mut texts: Vec<String> = texts_all[..n].to_vec();
    texts.extend(var_texts);
    texts.extend(noise_texts);
    let cl12: Vec<u8> = meta["cl12"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect();
    let noise_cl: Vec<u8> = meta["noise_cl"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u8).collect();
    let mut kind = vec![K_BASE; nu];
    let mut canon = vec![u32::MAX; nu];
    let mut depth = vec![0u8; nu];
    let mut key = vec![0u32; nu];
    let mut cl = vec![0u8; nu];
    for d in 0..n {
        canon[d] = d as u32;
        key[d] = d as u32;
        cl[d] = cl12[d];
    }
    let mut old_of = vec![[-1i32; 2]; n];
    let mut near_of: Vec<Vec<u32>> = vec![vec![]; n];
    let nunits = meta["units"].as_array().unwrap().len();
    let mut unit_old: Vec<Vec<u32>> = vec![vec![]; nunits];
    let mut unit_deep = vec![false; nunits];
    for (i, v) in variants.iter().enumerate() {
        let u = n + i;
        let b = v["base"].as_u64().unwrap() as usize;
        canon[u] = b as u32;
        key[u] = b as u32;
        cl[u] = cl12[b];
        if v["kind"].as_str().unwrap() == "old" {
            kind[u] = K_OLD;
            let dp = v["depth"].as_u64().unwrap() as u8;
            depth[u] = dp;
            old_of[b][dp as usize - 1] = u as i32;
            let un = v["unit"].as_u64().unwrap() as usize;
            if dp == 1 {
                unit_old[un].push(b as u32);
            } else {
                unit_deep[un] = true;
            }
        } else {
            kind[u] = K_NEAR;
            near_of[b].push(u as u32);
        }
    }
    let noise_start = n + nvar;
    for j in 0..nnoise {
        let u = noise_start + j;
        kind[u] = K_NOISE;
        key[u] = (n + j) as u32;
        cl[u] = noise_cl[j];
    }
    let unit_of: Vec<u32> = meta["unit_of"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    // queries
    let mut qs = vec![];
    let mut qvec = vec![];
    let mut qtext = vec![];
    for q in meta["queries"].as_array().unwrap() {
        let tid = q["tid"].as_u64().unwrap() as usize;
        qs.push(Query { fam: if q["fam"].as_str().unwrap() == "H" { 0 } else { 1 }, rel: q["rel"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect(), tid });
        qvec.extend_from_slice(&emb_all[tid * DIM..(tid + 1) * DIM]);
        qtext.push(texts_all[tid].clone());
    }
    let nq = qs.len();
    drop(emb_all);
    // BM25 (idf/avgdl from the BASE docs only; postings over the whole universe)
    let toks: Vec<Vec<String>> = texts.iter().map(|t| tokens(t)).collect();
    let mut df: HashMap<&str, u32> = HashMap::new();
    for d in 0..n {
        let mut seen: Vec<&str> = toks[d].iter().map(|s| s.as_str()).collect();
        seen.sort_unstable();
        seen.dedup();
        for t in seen {
            *df.entry(t).or_insert(0) += 1;
        }
    }
    let avg = toks[..n].iter().map(|t| t.len() as f32).sum::<f32>() / n as f32;
    let mut post: HashMap<&str, Vec<(u32, f32)>> = HashMap::new();
    let mut sparse_b = vec![0u32; nu];
    for (u, tk) in toks.iter().enumerate() {
        let mut c: HashMap<&str, f32> = HashMap::new();
        for t in tk {
            *c.entry(t.as_str()).or_insert(0.0) += 1.0;
        }
        sparse_b[u] = (c.len() * 8) as u32;
        for (t, f) in c {
            post.entry(t).or_default().push((u as u32, f));
        }
    }
    let idf = |t: &str| -> f32 {
        let d = *df.get(t).unwrap_or(&0) as f32;
        ((n as f32 - d + 0.5) / (d + 0.5) + 1.0).ln()
    };
    let (k1, b) = (1.2f32, 0.75f32);
    let mut bm = vec![0.0f32; nq * nu];
    for qi in 0..nq {
        let row = &mut bm[qi * nu..(qi + 1) * nu];
        let mut seen: Vec<String> = vec![];
        for t in tokens(&qtext[qi]) {
            if seen.contains(&t) {
                continue;
            }
            if let Some(p) = post.get(t.as_str()) {
                let w = idf(&t);
                for &(u, tf) in p {
                    let norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * toks[u as usize].len() as f32 / avg));
                    row[u as usize] += w * norm;
                }
            }
            seen.push(t);
        }
    }
    drop(post);
    drop(toks);
    // dense matrices (f32, block-fp16-style, int8) : query (f32) . dequantised row
    let mut d16 = vec![0.0f32; nu * DIM];
    let mut d8 = vec![0.0f32; nu * DIM];
    for u in 0..nu {
        deq_b16(&emb[u * DIM..(u + 1) * DIM], &mut d16[u * DIM..(u + 1) * DIM]);
        deq_i8(&emb[u * DIM..(u + 1) * DIM], &mut d8[u * DIM..(u + 1) * DIM]);
    }
    let mk = |m: &Vec<f32>| -> Vec<f32> {
        let mut out = vec![0.0f32; nq * nu];
        out.par_chunks_mut(nu).enumerate().for_each(|(qi, row)| {
            let q = &qvec[qi * DIM..(qi + 1) * DIM];
            for u in 0..nu {
                row[u] = dot(q, &m[u * DIM..(u + 1) * DIM]);
            }
        });
        out
    };
    let s32 = mk(&emb);
    let s16 = mk(&d16);
    let s8 = mk(&d8);
    drop(d16);
    drop(d8);
    // neighbour lists (cos >= nn_min, top 48, excluding self)
    let nn: Vec<Vec<(u32, f32)>> = (0..nu)
        .into_par_iter()
        .map(|u| {
            let a = &emb[u * DIM..(u + 1) * DIM];
            let mut v: Vec<(u32, f32)> = vec![];
            for w in 0..nu {
                if w == u {
                    continue;
                }
                let c = dot(a, &emb[w * DIM..(w + 1) * DIM]);
                if c >= nn_min {
                    v.push((w as u32, c));
                }
            }
            v.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
            v.truncate(48);
            v
        })
        .collect();
    // closed-loop side data (h13_prep)
    let j: Value = serde_json::from_slice(&rd(&format!("results/h13_world_{tag}.json"))).expect("run py/h13_prep.py");
    let feat: HashMap<usize, Vec<String>> = j["feat"].as_object().unwrap().iter().map(|(k, v)| (k.parse().unwrap(), v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect())).collect();
    let mut kwtok12: Vec<(String, usize)> = j["kwtok"]["12"].as_object().unwrap().iter().map(|(a, b)| (a.clone(), b.as_u64().unwrap() as usize)).collect();
    kwtok12.sort();
    let keyspace = n + nnoise;
    println!("[k1] world {tag}: base {n}, variants {nvar}, noise {nnoise}, universe {nu}, queries {nq} (matrices {:.0} MB) loaded in {:.1}s", (4 * nq * nu * 4) as f64 / 1e6, t0.elapsed().as_secs_f64());
    WorldK { tag: tag.to_string(), n, nu, emb, texts, kind, canon, depth, key, cl, keyspace, unit_of, nunits, old_of, near_of, unit_old, unit_deep, noise_start, qs, qvec, qtext, sparse_b, s32, s16, s8, bm, nn, feat, kwtok12, cl12, ctl: 0 }
}

// =============================================================================================== stream

#[derive(Clone, Copy, Debug)]
pub struct Rec {
    pub u: u32,
    pub key: u32,
    pub rev: u8,
}

pub struct Stream {
    pub recs: Vec<Rec>,
    pub rev_u: Vec<u8>,       // revision of the CONTENT of each universe row (old: 0/1, base/near: base_rev)
    pub base_arr: Vec<u32>,   // arrival index of the base record of doc d
    pub old_arr: Vec<Vec<u32>>,
    pub ep_end: Vec<usize>,   // NEP+1 boundaries
    pub bytes_none: f64,      // bytes of the unbounded store at the end
    pub pop_rank: Vec<f64>,   // popularity weight per query
}

pub fn make_stream(w: &WorldK, seed: u64) -> Stream {
    let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x6B31_0000 ^ (w.n as u64));
    let n = w.n;
    let mut t_unit = vec![0.0f64; w.nunits];
    for u in 0..w.nunits {
        t_unit[u] = r.f64();
    }
    // old-version activation per unit (seeded coin); unit_old[u] non-empty => candidate
    let mut active = vec![false; w.nunits];
    for u in 0..w.nunits {
        if !w.unit_old[u].is_empty() {
            let c = r.f64();
            active[u] = match w.ctl { 1 => false, 3 => true, _ => c < P_OLD_ACTIVE };
        }
    }
    let mut rev_u = vec![0u8; w.nu];
    let mut ev: Vec<(f64, u32, u32, u8)> = vec![]; // (time, tiebreak, u, rev)
    let mut base_rev = vec![0u8; n];
    let mut tb = vec![0.0f64; n];
    for d in 0..n {
        let un = w.unit_of[d] as usize;
        tb[d] = (t_unit[un] + 0.01 * r.f64()).min(0.9999);
        if active[un] {
            let deep = w.old_of[d][1] >= 0;
            base_rev[d] = if deep { 2 } else { 1 };
        }
    }
    let mut seq = 0u32;
    let mut push = |ev: &mut Vec<(f64, u32, u32, u8)>, t: f64, u: u32, rev: u8| {
        ev.push((t, seq, u, rev));
        seq += 1;
    };
    for d in 0..n {
        let un = w.unit_of[d] as usize;
        rev_u[d] = base_rev[d];
        push(&mut ev, tb[d], d as u32, base_rev[d]);
        if active[un] {
            let o1 = w.old_of[d][0];
            if o1 >= 0 {
                let late = r.f64() < P_LATE_OLD;
                let t1 = if late { tb[d] + (1.0 - tb[d]) * r.f64() } else { tb[d] * (0.05 + 0.85 * r.f64()) };
                let rv1 = base_rev[d] - 1;
                rev_u[o1 as usize] = rv1;
                push(&mut ev, t1, o1 as u32, rv1);
                let o2 = w.old_of[d][1];
                if o2 >= 0 {
                    let t2 = t1 * (0.05 + 0.9 * r.f64());
                    rev_u[o2 as usize] = 0;
                    push(&mut ev, t2, o2 as u32, 0);
                }
            }
        }
        for &nu_ in &w.near_of[d] {
            if w.ctl != 2 && r.f64() < P_NEAR_ACTIVE {
                rev_u[nu_ as usize] = base_rev[d];
                let t = tb[d] + (1.0 - tb[d]) * r.f64();
                push(&mut ev, t, nu_, base_rev[d]);
            }
        }
        if w.ctl != 2 && r.f64() < P_EXACT {
            let c = 1 + (r.f64() < 0.3) as usize + (r.f64() < 0.1) as usize;
            for _ in 0..c {
                let t = tb[d] + (1.0 - tb[d]) * r.f64();
                push(&mut ev, t, d as u32, base_rev[d]);
            }
        }
    }
    for u in w.noise_start..w.nu {
        if r.f64() < P_NOISE_ACTIVE {
            push(&mut ev, r.f64(), u as u32, 0);
        }
    }
    ev.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let recs: Vec<Rec> = ev.iter().map(|&(_, _, u, rev)| Rec { u, key: w.key[u as usize], rev }).collect();
    let mut base_arr = vec![u32::MAX; n];
    let mut old_arr: Vec<Vec<u32>> = vec![vec![]; n];
    let mut bytes_none = 0.0;
    for (i, rc) in recs.iter().enumerate() {
        let u = rc.u as usize;
        bytes_none += (Tier::F32.row_bytes(DIM) + META_B + w.sparse_b[u] as usize) as f64;
        if w.kind[u] == K_BASE {
            if base_arr[u] == u32::MAX {
                base_arr[u] = i as u32;
            }
        } else if w.kind[u] == K_OLD {
            old_arr[w.canon[u] as usize].push(i as u32);
        }
    }
    let ntot = recs.len();
    let ep_end: Vec<usize> = (0..=NEP).map(|e| ntot * e / NEP).collect();
    // popularity: Zipf weight by a seeded random permutation of the query pool
    let nq = w.qs.len();
    let mut perm: Vec<usize> = (0..nq).collect();
    for i in (1..nq).rev() {
        let j = r.below(i + 1);
        perm.swap(i, j);
    }
    let mut pop_rank = vec![0.0f64; nq];
    for (rank, &q) in perm.iter().enumerate() {
        pop_rank[q] = 1.0 / ((rank + 1) as f64).powf(ZIPF_S);
    }
    Stream { recs, rev_u, base_arr, old_arr, ep_end, bytes_none, pop_rank }
}

/// per-checkpoint arm-independent facts derived from the stream prefix
pub struct CkInfo {
    pub plen: usize,
    pub elig: Vec<usize>,        // eligible query indices (all rel docs' base record arrived)
    pub curver: Vec<u8>,         // max revision arrived per key id (keyspace)
    pub oldseen: Vec<bool>,      // per doc: at least one OLD record has arrived
    pub eval_u: Vec<usize>,
    pub eval_p: Vec<usize>,
}

pub fn ck_info(w: &WorldK, s: &Stream, plen: usize, seed: u64, ck: usize) -> CkInfo {
    let mut curver = vec![0u8; w.keyspace];
    for rc in &s.recs[..plen] {
        let k = rc.key as usize;
        if rc.rev > curver[k] {
            curver[k] = rc.rev;
        }
    }
    let mut oldseen = vec![false; w.n];
    for d in 0..w.n {
        oldseen[d] = s.old_arr[d].iter().any(|&a| (a as usize) < plen);
    }
    let elig: Vec<usize> = (0..w.qs.len()).filter(|&q| w.qs[q].rel.iter().all(|&d| (s.base_arr[d as usize] as usize) < plen)).collect();
    let mut r = Rng::new(seed.wrapping_mul(7919) ^ (ck as u64 + 1) * 0xABCDEF);
    let mut eval_u = vec![];
    let mut eval_p = vec![];
    if !elig.is_empty() {
        for _ in 0..NEVAL {
            eval_u.push(elig[r.below(elig.len())]);
        }
        let tot: f64 = elig.iter().map(|&q| s.pop_rank[q]).sum();
        for _ in 0..NEVAL {
            let mut x = r.f64() * tot;
            let mut pick = elig[elig.len() - 1];
            for &q in &elig {
                x -= s.pop_rank[q];
                if x <= 0.0 {
                    pick = q;
                    break;
                }
            }
            eval_p.push(pick);
        }
    }
    CkInfo { plen, elig, curver, oldseen, eval_u, eval_p }
}

pub fn usage_queries(w: &WorldK, s: &Stream, plen: usize, seed: u64, ep: usize) -> Vec<usize> {
    let elig: Vec<usize> = (0..w.qs.len()).filter(|&q| w.qs[q].rel.iter().all(|&d| (s.base_arr[d as usize] as usize) < plen)).collect();
    let mut r = Rng::new(seed.wrapping_mul(104729) ^ (ep as u64 + 3) * 0x5151);
    let tot: f64 = elig.iter().map(|&q| s.pop_rank[q]).sum();
    let mut out = vec![];
    for _ in 0..NUSE {
        if elig.is_empty() {
            break;
        }
        let mut x = r.f64() * tot;
        let mut pick = elig[elig.len() - 1];
        for &q in &elig {
            x -= s.pop_rank[q];
            if x <= 0.0 {
                pick = q;
                break;
            }
        }
        out.push(pick);
    }
    out
}

// =============================================================================================== arms

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Step {
    Exact,
    SupDel,
    SupSoft,
    /// POST-HOC control: structural key but NO revision stamps (the latest ARRIVAL replaces the key's rows)
    SupArr,
    Cos { theta: f32, newest: bool },
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ev {
    Fifo,
    Lru,
    Value,
}
#[derive(Clone, Debug)]
pub struct Arm {
    pub name: String,
    pub group: &'static str,
    pub steps: Vec<Step>,
    pub evict: Option<(Ev, f64)>,
    pub decay: f32,
    pub tier: Option<(f64, f64)>, // (hot frac f32, warm frac block-fp16); rest int8
    pub summary: bool,
    pub lam: f32,  // soft supersession penalty (z units)
    pub beta: f32, // recency weight
    pub rand_credit: bool, // CONTROL: usage credit goes to a random row of the top-3 (uninformative feedback)
}
impl Arm {
    pub fn new(name: &str, group: &'static str) -> Arm {
        Arm { name: name.into(), group, steps: vec![], evict: None, decay: 0.5, tier: None, summary: false, lam: 0.0, beta: 0.0, rand_credit: false }
    }
    pub fn needs_usage(&self) -> bool {
        matches!(self.evict, Some((Ev::Lru, _)) | Some((Ev::Value, _)))
    }
}

pub fn arm_list() -> Vec<Arm> {
    let mut v = vec![Arm::new("none", "base")];
    let mut a = Arm::new("exact", "dup");
    a.steps = vec![Step::Exact];
    v.push(a);
    for &th in &[0.98f32, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70] {
        let mut a = Arm::new(&format!("cosfirst{:.2}", th), "cosfirst");
        a.steps = vec![Step::Cos { theta: th, newest: false }];
        v.push(a);
        let mut a = Arm::new(&format!("cosnew{:.2}", th), "cosnew");
        a.steps = vec![Step::Cos { theta: th, newest: true }];
        v.push(a);
    }
    let mut a = Arm::new("supdel", "sup");
    a.steps = vec![Step::SupDel];
    v.push(a);
    for &l in &[0.5f32, 1.0, 2.0, 4.0, 8.0] {
        let mut a = Arm::new(&format!("supsoft{:.1}", l), "supsoft");
        a.steps = vec![Step::SupSoft];
        a.lam = l;
        v.push(a);
    }
    for &b in &[0.5f32, 1.0, 2.0] {
        let mut a = Arm::new(&format!("rec{:.1}", b), "rec");
        a.beta = b;
        v.push(a);
    }
    let mut a = Arm::new("exact+supdel", "stack");
    a.steps = vec![Step::Exact, Step::SupDel];
    v.push(a);
    for &(nm, ev, fr, dc) in &[
        ("fifo50", Ev::Fifo, 0.5, 0.5f32),
        ("fifo25", Ev::Fifo, 0.25, 0.5),
        ("lru50", Ev::Lru, 0.5, 0.5),
        ("lru25", Ev::Lru, 0.25, 0.5),
        ("val50", Ev::Value, 0.5, 0.5),
        ("val25", Ev::Value, 0.25, 0.5),
    ] {
        let mut a = Arm::new(nm, if nm.starts_with("val") { "val" } else { "evict" });
        a.evict = Some((ev, fr));
        a.decay = dc;
        v.push(a);
    }
    let mut a = Arm::new("val25rand", "ctl");
    a.evict = Some((Ev::Value, 0.25));
    a.rand_credit = true;
    v.push(a);
    for &(nm, ev, fr, dc) in &[("supdel+fifo25", Ev::Fifo, 0.25, 0.5f32), ("supdel+val25", Ev::Value, 0.25, 0.5)] {
        let mut a = Arm::new(nm, "stack");
        a.steps = vec![Step::SupDel];
        a.evict = Some((ev, fr));
        a.decay = dc;
        v.push(a);
    }
    let mut a = Arm::new("supdel-arrival", "ctl");
    a.steps = vec![Step::SupArr];
    v.push(a);
    let mut a = Arm::new("exact+supdel+cosfirst0.95", "stack");
    a.steps = vec![Step::Exact, Step::SupDel, Step::Cos { theta: 0.95, newest: false }];
    v.push(a);
    for &(nm, h, wm) in &[("tier-f16", 0.0, 1.0), ("tier-int8", 0.0, 0.0), ("tier.2/.3", 0.2, 0.3), ("tier.1/.2", 0.1, 0.2)] {
        let mut a = Arm::new(nm, "tier");
        a.tier = Some((h, wm));
        v.push(a);
    }
    let mut a = Arm::new("summary", "summary");
    a.summary = true;
    v.push(a);
    let mut a = Arm::new("supdel+summary", "stack");
    a.steps = vec![Step::SupDel];
    a.summary = true;
    v.push(a);
    let mut a = Arm::new("exact+supdel+tier.2/.3", "stack");
    a.steps = vec![Step::Exact, Step::SupDel];
    a.tier = Some((0.2, 0.3));
    v.push(a);
    v
}

// =============================================================================================== store

#[derive(Clone)]
pub struct Summ {
    pub vec: Vec<f32>,
    pub sumv: Vec<f32>,
    pub members: Vec<u32>, // universe rows (vectors dropped, text/sparse kept)
}

#[derive(Clone)]
pub struct Row {
    pub u: u32,
    pub key: u32,
    pub rev: u8,
    pub count: u32,
    pub arr: u32,
    pub tier: u8, // 0 f32, 1 block-fp16, 2 int8
    pub value: f32,
    pub last_use: u32,
    pub sup: bool,
    pub alive: bool,
    pub id: u64,
    pub sum: Option<Box<Summ>>,
}

pub struct Store {
    pub rows: Vec<Row>,
    pub by_u: HashMap<u32, Vec<u32>>,
    pub by_key: HashMap<u32, Vec<u32>>,
    pub dd: Deduper,
    pub id_row: HashMap<u64, u32>,
    pub next_id: u64,
    pub stale_dropped: u64,
    pub merged: u64,
    pub alive_n: usize,
    pub cur_ep: u32,
}

impl Store {
    pub fn new() -> Store {
        Store { rows: vec![], by_u: HashMap::new(), by_key: HashMap::new(), dd: Deduper::new(DedupConfig::default()), id_row: HashMap::new(), next_id: 1, stale_dropped: 0, merged: 0, alive_n: 0, cur_ep: 0 }
    }
    fn push_row(&mut self, rc: &Rec, arr: u32, id: u64) {
        let idx = self.rows.len() as u32;
        self.rows.push(Row { u: rc.u, key: rc.key, rev: rc.rev, count: 1, arr, tier: 0, value: 0.0, last_use: self.cur_ep, sup: false, alive: true, id, sum: None });
        self.by_u.entry(rc.u).or_default().push(idx);
        self.by_key.entry(rc.key).or_default().push(idx);
        self.id_row.insert(id, idx);
        self.alive_n += 1;
    }
    fn kill(&mut self, idx: u32) {
        let (u, key, id, was_sum) = {
            let r = &mut self.rows[idx as usize];
            if !r.alive {
                return;
            }
            r.alive = false;
            (r.u, r.key, r.id, r.sum.is_some())
        };
        self.alive_n -= 1;
        if let Some(v) = self.by_u.get_mut(&u) {
            v.retain(|&x| x != idx);
        }
        if let Some(v) = self.by_key.get_mut(&key) {
            v.retain(|&x| x != idx);
        }
        self.id_row.remove(&id);
        self.dd.forget(id);
        let _ = was_sum;
    }
    /// compact dead rows; rebuild maps (row indices change)
    pub fn compact(&mut self) {
        let old = std::mem::take(&mut self.rows);
        self.by_u.clear();
        self.by_key.clear();
        self.id_row.clear();
        for r in old.into_iter().filter(|r| r.alive) {
            let idx = self.rows.len() as u32;
            self.by_u.entry(r.u).or_default().push(idx);
            if r.sum.is_none() {
                self.by_key.entry(r.key).or_default().push(idx);
            }
            self.id_row.insert(r.id, idx);
            self.rows.push(r);
        }
    }

    pub fn ingest(&mut self, w: &WorldK, arm: &Arm, rc: &Rec, arr: u32) {
        let id = self.next_id;
        self.next_id += 1;
        for st in &arm.steps {
            match *st {
                Step::Exact => {
                    let u = rc.u as usize;
                    match self.dd.check(id, Some(&w.texts[u]), Some(&w.emb[u * DIM..(u + 1) * DIM])) {
                        Verdict::Duplicate { of, .. } => {
                            if let Some(&ri) = self.id_row.get(&of) {
                                self.rows[ri as usize].count += 1;
                            }
                            self.merged += 1;
                            return;
                        }
                        Verdict::New => {}
                    }
                }
                Step::SupArr => {
                    let ex: Vec<u32> = self.by_key.get(&rc.key).cloned().unwrap_or_default();
                    if let Some(&same) = ex.iter().find(|&&i| self.rows[i as usize].u == rc.u) {
                        self.rows[same as usize].count += 1;
                        self.merged += 1;
                        return;
                    }
                    for &i in &ex {
                        self.kill(i);
                    }
                }
                Step::SupDel | Step::SupSoft => {
                    let soft = *st == Step::SupSoft;
                    let ex: Vec<u32> = self.by_key.get(&rc.key).cloned().unwrap_or_default();
                    if !soft {
                        if ex.iter().any(|&i| self.rows[i as usize].rev > rc.rev) {
                            self.stale_dropped += 1;
                            return;
                        }
                        if let Some(&same) = ex.iter().find(|&&i| self.rows[i as usize].rev == rc.rev) {
                            self.rows[same as usize].count += 1;
                            self.merged += 1;
                            return;
                        }
                        for &i in &ex {
                            if self.rows[i as usize].rev < rc.rev {
                                self.kill(i);
                            }
                        }
                    } else {
                        let newer = ex.iter().any(|&i| self.rows[i as usize].rev > rc.rev);
                        for &i in &ex {
                            if self.rows[i as usize].rev < rc.rev {
                                self.rows[i as usize].sup = true;
                            }
                        }
                        self.push_row(rc, arr, id);
                        if newer {
                            let li = self.rows.len() - 1;
                            self.rows[li].sup = true;
                        }
                        return;
                    }
                }
                Step::Cos { theta, newest } => {
                    // best retained row among {same vector} U nn(u)
                    let mut best: Option<(u32, f32)> = None;
                    if let Some(v) = self.by_u.get(&rc.u) {
                        if let Some(&i) = v.first() {
                            best = Some((i, 1.0));
                        }
                    }
                    if best.is_none() {
                        for &(v, c) in &w.nn[rc.u as usize] {
                            if c < theta {
                                break;
                            }
                            if let Some(rows) = self.by_u.get(&v) {
                                if let Some(&i) = rows.first() {
                                    best = Some((i, c));
                                    break;
                                }
                            }
                        }
                    }
                    if let Some((i, c)) = best {
                        if c >= theta {
                            self.merged += 1;
                            let r = &mut self.rows[i as usize];
                            r.count += 1;
                            if newest {
                                let (ou, okey) = (r.u, r.key);
                                r.u = rc.u;
                                r.key = rc.key;
                                r.rev = rc.rev;
                                r.arr = arr;
                                if let Some(v) = self.by_u.get_mut(&ou) {
                                    v.retain(|&x| x != i);
                                }
                                if let Some(v) = self.by_key.get_mut(&okey) {
                                    v.retain(|&x| x != i);
                                }
                                self.by_u.entry(rc.u).or_default().push(i);
                                self.by_key.entry(rc.key).or_default().push(i);
                            }
                            return;
                        }
                    }
                }
            }
        }
        self.push_row(rc, arr, id);
    }

    pub fn row_dense_bytes(r: &Row) -> usize {
        match r.tier {
            0 => Tier::F32.row_bytes(DIM),
            1 => Tier::BlockFp16.row_bytes(DIM),
            _ => Tier::Int8.row_bytes(DIM),
        }
    }
    /// (total, dense, sparse, meta) bytes of the alive rows
    pub fn bytes(&self, w: &WorldK) -> (usize, usize, usize, usize) {
        let (mut dn, mut sp, mut mt) = (0usize, 0usize, 0usize);
        for r in self.rows.iter().filter(|r| r.alive) {
            match &r.sum {
                None => {
                    dn += Self::row_dense_bytes(r);
                    sp += w.sparse_b[r.u as usize] as usize;
                    mt += META_B;
                }
                Some(s) => {
                    dn += Tier::F32.row_bytes(DIM);
                    mt += META_B;
                    for &m in &s.members {
                        sp += w.sparse_b[m as usize] as usize;
                        mt += 8;
                    }
                }
            }
        }
        (dn + sp + mt, dn, sp, mt)
    }
}

// =============================================================================================== retrieval

#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub row: u32,
    pub urep: u32, // universe row whose content the hit returns (summary rows: best member by bm25)
    pub cos: f32,
    pub score: f32,
}

pub struct Scratch {
    dense: Vec<f32>,
    bm: Vec<f32>,
    alive: Vec<u32>,
}
impl Scratch {
    pub fn new() -> Scratch {
        Scratch { dense: vec![], bm: vec![], alive: vec![] }
    }
}

pub fn rank(w: &WorldK, st: &Store, arm: &Arm, qi: usize, k: usize, sc: &mut Scratch, cur_total: u32) -> Vec<Hit> {
    let nu = w.nu;
    let qv = &w.qvec[qi * DIM..(qi + 1) * DIM];
    let (m32, m16, m8, mbm) = (&w.s32[qi * nu..(qi + 1) * nu], &w.s16[qi * nu..(qi + 1) * nu], &w.s8[qi * nu..(qi + 1) * nu], &w.bm[qi * nu..(qi + 1) * nu]);
    sc.dense.clear();
    sc.bm.clear();
    sc.alive.clear();
    let (mut sd, mut sdd, mut sb, mut sbb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (i, r) in st.rows.iter().enumerate() {
        if !r.alive {
            continue;
        }
        let (d, b) = match &r.sum {
            None => (
                match r.tier {
                    0 => m32[r.u as usize],
                    1 => m16[r.u as usize],
                    _ => m8[r.u as usize],
                },
                mbm[r.u as usize],
            ),
            Some(s) => (dot(qv, &s.vec), s.members.iter().fold(0.0f32, |a, &m| a.max(mbm[m as usize]))),
        };
        sc.alive.push(i as u32);
        sc.dense.push(d);
        sc.bm.push(b);
        sd += d as f64;
        sdd += (d * d) as f64;
        sb += b as f64;
        sbb += (b * b) as f64;
    }
    let nr = sc.alive.len();
    if nr == 0 {
        return vec![];
    }
    let nf = nr as f64;
    let (md, mb) = (sd / nf, sb / nf);
    let sdv = ((sdd / nf - md * md).max(0.0)).sqrt().max(1e-9);
    let sbv = ((sbb / nf - mb * mb).max(0.0)).sqrt().max(1e-9);
    let (md, mb, sdv, sbv) = (md as f32, mb as f32, sdv as f32, sbv as f32);
    let mut top: Vec<(f32, u32, usize)> = Vec::with_capacity(k + 1); // (score,row,pos)
    for p in 0..nr {
        let r = &st.rows[sc.alive[p] as usize];
        let mut s = ALPHA * (sc.dense[p] - md) / sdv + (1.0 - ALPHA) * (sc.bm[p] - mb) / sbv;
        if arm.lam > 0.0 && r.sup {
            s -= arm.lam;
        }
        if arm.beta > 0.0 {
            s += arm.beta * (r.arr as f32 / cur_total.max(1) as f32);
        }
        if top.len() < k || s > top[top.len() - 1].0 {
            let mut pos = top.len();
            while pos > 0 && top[pos - 1].0 < s {
                pos -= 1;
            }
            top.insert(pos, (s, sc.alive[p], p));
            if top.len() > k {
                top.pop();
            }
        }
    }
    top.iter()
        .map(|&(s, ri, p)| {
            let r = &st.rows[ri as usize];
            let urep = match &r.sum {
                None => r.u,
                Some(sm) => {
                    let mut best = (f32::MIN, sm.members[0]);
                    for &m in &sm.members {
                        if mbm[m as usize] > best.0 {
                            best = (mbm[m as usize], m);
                        }
                    }
                    best.1
                }
            };
            Hit { row: ri, urep, cos: sc.dense[p], score: s }
        })
        .collect()
}

// =============================================================================================== per-arm run

pub struct CkOut {
    pub rows: f64,
    pub bytes: f64,
    pub b_dense: f64,
    pub b_sparse: f64,
    pub b_meta: f64,
    pub m: BTreeMap<String, f64>,
}

#[derive(Clone, Copy, Default)]
pub struct Cand {
    pub rel: bool,
    pub stale_rel: bool,
    pub stale: bool,
    pub cl: u8,
    pub cos: f32,
}
pub struct TaskRet {
    pub n: usize,
    pub c: [Cand; 5],
}

pub struct ArmOut {
    pub cks: Vec<CkOut>,
    pub merged: u64,
    pub stale_dropped: u64,
    pub tasks: Vec<TaskRet>,
}

fn is_rel(w: &WorldK, q: usize, urep: u32) -> bool {
    let u = urep as usize;
    (w.kind[u] == K_BASE || w.kind[u] == K_NEAR) && w.qs[q].rel.contains(&w.canon[u])
}

pub fn eval_set(w: &WorldK, st: &Store, arm: &Arm, set: &[usize], ci: &CkInfo, s: &Stream, tag: &str, out: &mut BTreeMap<String, f64>, sc: &mut Scratch) {
    let nq = set.len().max(1) as f64;
    let (mut h1, mut mrr, mut p3, mut dr3, mut stale1, mut noise1) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut rv_n, mut rv_h1, mut rv_st, mut rv_mrr) = (0.0, 0.0, 0.0, 0.0);
    let (mut hh, mut hh_n, mut hs, mut hs_n) = (0.0, 0.0, 0.0, 0.0);
    let cur_total = ci.plen as u32;
    for &q in set {
        let hits = rank(w, st, arm, q, TOPK, sc, cur_total);
        let fl: Vec<bool> = hits.iter().map(|h| is_rel(w, q, h.urep)).collect();
        let first = fl.iter().position(|&x| x);
        let hit1 = fl.first().copied().unwrap_or(false);
        let rr = first.map(|p| 1.0 / (p as f64 + 1.0)).unwrap_or(0.0);
        let p3q = fl.iter().take(3).filter(|&&x| x).count() as f64 / 3.0;
        let mut seen: Vec<u32> = vec![];
        for (h, &f) in hits.iter().zip(&fl).take(3) {
            if f && !seen.contains(&w.canon[h.urep as usize]) {
                seen.push(w.canon[h.urep as usize]);
            }
        }
        let dr = seen.len() as f64 / (w.qs[q].rel.len().min(3)) as f64;
        let (stale, noise) = match hits.first() {
            Some(h) => {
                let u = h.urep as usize;
                (w.kind[u] != K_NOISE && s.rev_u[u] < ci.curver[w.key[u] as usize], w.kind[u] == K_NOISE)
            }
            None => (false, false),
        };
        h1 += hit1 as u8 as f64;
        mrr += rr;
        p3 += p3q;
        dr3 += dr.min(1.0);
        stale1 += stale as u8 as f64;
        noise1 += noise as u8 as f64;
        let revq = w.qs[q].rel.iter().any(|&d| ci.oldseen[d as usize]);
        if revq {
            rv_n += 1.0;
            rv_h1 += hit1 as u8 as f64;
            rv_st += stale as u8 as f64;
            rv_mrr += rr;
        }
        if w.qs[q].fam == 0 {
            hh += hit1 as u8 as f64;
            hh_n += 1.0;
        } else {
            hs += hit1 as u8 as f64;
            hs_n += 1.0;
        }
    }
    let nan = f64::NAN;
    out.insert(format!("{tag}.hit1"), h1 / nq);
    out.insert(format!("{tag}.mrr"), mrr / nq);
    out.insert(format!("{tag}.p3"), p3 / nq);
    out.insert(format!("{tag}.dr3"), dr3 / nq);
    out.insert(format!("{tag}.stale1"), stale1 / nq);
    out.insert(format!("{tag}.noise1"), noise1 / nq);
    out.insert(format!("{tag}.revn"), rv_n);
    out.insert(format!("{tag}.rev_hit1"), if rv_n > 0.0 { rv_h1 / rv_n } else { nan });
    out.insert(format!("{tag}.rev_mrr"), if rv_n > 0.0 { rv_mrr / rv_n } else { nan });
    out.insert(format!("{tag}.rev_stale1"), if rv_n > 0.0 { rv_st / rv_n } else { nan });
    out.insert(format!("{tag}.H_hit1"), if hh_n > 0.0 { hh / hh_n } else { nan });
    out.insert(format!("{tag}.S_hit1"), if hs_n > 0.0 { hs / hs_n } else { nan });
}

fn pick_tasks(w: &WorldK, seed: u64) -> Vec<usize> {
    let mut r = Rng::new(seed.wrapping_mul(2_654_435_761) ^ 0x7A5C);
    (0..NT).map(|_| r.below(w.qs.len())).collect()
}

/// one arm over one seed's stream. `budget_none` = bytes of the unbounded store at the end.
pub fn run_arm(w: &WorldK, s: &Stream, cks: &[CkInfo], arm: &Arm, seed: u64, usage: &[Vec<usize>]) -> ArmOut {
    let mut st = Store::new();
    let mut sc = Scratch::new();
    let mut outs: Vec<CkOut> = vec![];
    let cap = arm.evict.map(|(_, f)| f * s.bytes_none);
    let mut ck_i = 0;
    for ep in 0..NEP {
        let (a, b) = (s.ep_end[ep], s.ep_end[ep + 1]);
        st.cur_ep = ep as u32 + 1;
        for i in a..b {
            st.ingest(w, arm, &s.recs[i], i as u32);
        }
        // usage feedback (value / LRU)
        if arm.needs_usage() {
            for r in st.rows.iter_mut() {
                r.value *= arm.decay;
            }
            for &q in &usage[ep] {
                let hits = rank(w, &st, arm, q, 3, &mut sc, b as u32);
                for (j, h) in hits.iter().enumerate() {
                    st.rows[h.row as usize].last_use = ep as u32 + 1;
                    if !arm.rand_credit && j == 0 && is_rel(w, q, h.urep) {
                        st.rows[h.row as usize].value += 1.0;
                    }
                }
                if arm.rand_credit && !hits.is_empty() {
                    let pick = (q.wrapping_mul(2654435761) ^ (ep * 97 + 13)) % hits.len();
                    st.rows[hits[pick].row as usize].value += 1.0;
                }
            }
        }
        // tiering by arrival age
        if let Some((hot, warm)) = arm.tier {
            let mut idx: Vec<usize> = (0..st.rows.len()).filter(|&i| st.rows[i].alive).collect();
            idx.sort_by_key(|&i| std::cmp::Reverse(st.rows[i].arr));
            let n = idx.len() as f64;
            for (rk, &i) in idx.iter().enumerate() {
                let f = rk as f64 / n;
                let t = if f < hot { 0 } else if f < hot + warm { 1 } else { 2 };
                if t > st.rows[i].tier {
                    st.rows[i].tier = t;
                }
            }
        }
        // anchor-summary rows keyed by the structural unit (section)
        if arm.summary {
            let mut arrs: Vec<u32> = st.rows.iter().filter(|r| r.alive).map(|r| r.arr).collect();
            arrs.sort_unstable();
            if !arrs.is_empty() {
                let cut = arrs[arrs.len() / 2];
                let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
                for (i, r) in st.rows.iter().enumerate() {
                    if r.alive && r.sum.is_none() && r.arr <= cut && w.kind[r.u as usize] != K_NOISE {
                        groups.entry(w.unit_of[w.canon[r.u as usize] as usize]).or_default().push(i);
                    }
                }
                let mut sum_of: HashMap<u32, usize> = HashMap::new();
                for (i, r) in st.rows.iter().enumerate() {
                    if r.alive {
                        if r.sum.is_some() {
                            sum_of.insert(r.key, i); // key field of a summary row = unit id (see below)
                        }
                    }
                }
                let mut gl: Vec<(u32, Vec<usize>)> = groups.into_iter().collect();
                gl.sort_by_key(|g| g.0);
                for (unit, mem) in gl {
                    let skey = u32::MAX - unit; // distinct key space for summary rows
                    let have = sum_of.get(&skey).copied();
                    if mem.len() + have.is_some() as usize >= 2 {
                        let si = match have {
                            Some(i) => i,
                            None => {
                                let idx = st.rows.len() as u32;
                                st.rows.push(Row { u: st.rows[mem[0]].u, key: skey, rev: 0, count: 0, arr: st.rows[mem[0]].arr, tier: 0, value: 0.0, last_use: 0, sup: false, alive: true, id: st.next_id, sum: Some(Box::new(Summ { vec: vec![0.0; DIM], sumv: vec![0.0; DIM], members: vec![] })) });
                                st.next_id += 1;
                                st.alive_n += 1;
                                idx as usize
                            }
                        };
                        for &mi in &mem {
                            let (mu, cnt, arr) = (st.rows[mi].u, st.rows[mi].count, st.rows[mi].arr);
                            {
                                let sm = st.rows[si].sum.as_mut().unwrap();
                                for k in 0..DIM {
                                    sm.sumv[k] += w.emb[mu as usize * DIM + k];
                                }
                                sm.members.push(mu);
                            }
                            st.rows[si].count += cnt;
                            st.rows[si].arr = st.rows[si].arr.max(arr);
                            st.rows[mi].alive = false;
                            st.alive_n -= 1;
                        }
                        let sm = st.rows[si].sum.as_mut().unwrap();
                        let nrm = sm.sumv.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                        for k in 0..DIM {
                            sm.vec[k] = sm.sumv[k] / nrm;
                        }
                    }
                }
            }
        }
        // hard byte budget eviction
        if let (Some((kind, _)), Some(cap)) = (arm.evict, cap) {
            let (mut tot, _, _, _) = st.bytes(w);
            if tot as f64 > cap {
                let mut idx: Vec<usize> = (0..st.rows.len()).filter(|&i| st.rows[i].alive).collect();
                match kind {
                    Ev::Fifo => idx.sort_by_key(|&i| (st.rows[i].arr, i)),
                    Ev::Lru => idx.sort_by_key(|&i| (st.rows[i].last_use, st.rows[i].arr, i)),
                    Ev::Value => idx.sort_by(|&x, &y| st.rows[x].value.partial_cmp(&st.rows[y].value).unwrap().then(st.rows[x].arr.cmp(&st.rows[y].arr))),
                }
                for i in idx {
                    if (tot as f64) <= cap {
                        break;
                    }
                    let r = &st.rows[i];
                    let rb = Store::row_dense_bytes(r) + META_B + w.sparse_b[r.u as usize] as usize;
                    st.kill(i as u32);
                    tot = tot.saturating_sub(rb);
                }
            }
        }
        st.compact();
        if ck_i < CKPT_EP.len() && ep + 1 == CKPT_EP[ck_i] {
            let ci = &cks[ck_i];
            let (tot, dn, sp, mt) = st.bytes(w);
            let mut m = BTreeMap::new();
            eval_set(w, &st, arm, &ci.eval_u, ci, s, "u", &mut m, &mut sc);
            eval_set(w, &st, arm, &ci.eval_p, ci, s, "p", &mut m, &mut sc);
            // staleness of the whole store (fraction of retained rows that are superseded versions)
            let (mut ns, mut nr) = (0.0, 0.0);
            for r in st.rows.iter().filter(|r| r.alive && r.sum.is_none()) {
                nr += 1.0;
                if w.kind[r.u as usize] != K_NOISE && s.rev_u[r.u as usize] < ci.curver[w.key[r.u as usize] as usize] {
                    ns += 1.0;
                }
            }
            m.insert("store_stale_frac".into(), if nr > 0.0 { ns / nr } else { 0.0 });
            outs.push(CkOut { rows: st.alive_n as f64, bytes: tot as f64, b_dense: dn as f64, b_sparse: sp as f64, b_meta: mt as f64, m });
            ck_i += 1;
        }
    }
    // closed-loop retrieval lists on the final store
    let tq = pick_tasks(w, seed);
    let ci = cks.last().unwrap();
    let mut tasks = Vec::with_capacity(NT);
    for &q in &tq {
        let hits = rank(w, &st, arm, q, 5, &mut sc, ci.plen as u32);
        let mut t = TaskRet { n: hits.len(), c: [Cand::default(); 5] };
        for (j, h) in hits.iter().enumerate() {
            let u = h.urep as usize;
            let right_key = w.kind[u] != K_NOISE && w.qs[q].rel.contains(&w.canon[u]);
            t.c[j] = Cand { rel: is_rel(w, q, h.urep), stale_rel: right_key && w.kind[u] == K_OLD, stale: w.kind[u] != K_NOISE && s.rev_u[u] < ci.curver[w.key[u] as usize], cl: w.cl[u], cos: h.cos };
        }
        tasks.push(t);
    }
    ArmOut { cks: outs, merged: st.merged, stale_dropped: st.stale_dropped, tasks }
}

// =============================================================================================== closed loop (copy of the h13_route engine, vote arms only)

pub const NA: usize = 6;
pub const PEN: f64 = 0.05;

#[derive(Clone, Debug)]
pub struct WSpec {
    pub name: &'static str,
    pub topic_dep: bool,
    pub drift: bool,
    pub gain: f64,
    pub ctx_real: bool,
    pub low_base: bool,
}
pub fn loop_worlds() -> Vec<WSpec> {
    vec![
        WSpec { name: "T12", topic_dep: true, drift: false, gain: 0.15, ctx_real: true, low_base: false },
        WSpec { name: "T12-drift", topic_dep: true, drift: true, gain: 0.15, ctx_real: true, low_base: false },
        WSpec { name: "Tctx", topic_dep: false, drift: false, gain: 0.20, ctx_real: true, low_base: false },
        WSpec { name: "Tqa", topic_dep: false, drift: false, gain: 0.85, ctx_real: true, low_base: true },
        WSpec { name: "ctl-irrel", topic_dep: false, drift: false, gain: 0.0, ctx_real: false, low_base: false },
    ]
}

pub struct Env {
    pub u: Vec<f64>,
    pub cl_true: Vec<usize>,
    pub feats: Vec<Vec<String>>,
    pub p_pre: Vec<[f64; NA]>,
    pub p_post: Vec<[f64; NA]>,
    pub kw: HashMap<String, usize>,
}
fn uni(r: &mut Rng, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * r.f64()
}
pub fn make_env(w: &WorldK, ws: &WSpec, seed: u64) -> Env {
    let k = 12usize;
    let mut name_hash = 0u64;
    for b in ws.name.bytes() {
        name_hash = name_hash.wrapping_mul(131).wrapping_add(b as u64);
    }
    let mut r = Rng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x13_C0DE ^ (name_hash << 20));
    let mut p_pre = vec![[0.0; NA]; k];
    let mut best_of = vec![0usize; k];
    if ws.topic_dep {
        for c in 0..k {
            for a in 0..NA {
                p_pre[c][a] = uni(&mut r, 0.15, 0.45);
            }
            let best = r.below(NA);
            p_pre[c][best] = uni(&mut r, 0.80, 0.90);
            let mut second = r.below(NA - 1);
            if second >= best {
                second += 1;
            }
            p_pre[c][second] = uni(&mut r, 0.55, 0.68);
            best_of[c] = best;
        }
    } else {
        let mut row = [0.0; NA];
        for a in 0..NA {
            row[a] = if ws.low_base { uni(&mut r, 0.10, 0.25) } else { uni(&mut r, 0.25, 0.50) };
        }
        let best = r.below(NA);
        row[best] = if ws.low_base { uni(&mut r, 0.25, 0.35) } else { uni(&mut r, 0.62, 0.72) };
        for c in 0..k {
            p_pre[c] = row;
            best_of[c] = best;
        }
    }
    let mut p_post = p_pre.clone();
    if ws.drift {
        for c in 0..k {
            if r.bernoulli(0.25) {
                let old = best_of[c];
                let mut new = r.below(NA - 1);
                if new >= old {
                    new += 1;
                }
                p_post[c][old] = uni(&mut r, 0.30, 0.50);
                p_post[c][new] = uni(&mut r, 0.80, 0.90);
            }
        }
    }
    let mut kw = HashMap::new();
    for (t, c) in &w.kwtok12 {
        let a = if !ws.topic_dep {
            r.below(NA)
        } else if r.bernoulli(0.65) {
            best_of[*c]
        } else {
            let mut x = r.below(NA - 1);
            if x >= best_of[*c] {
                x += 1;
            }
            x
        };
        kw.insert(t.clone(), a);
    }
    let tq = pick_tasks(w, seed);
    let (mut u, mut cl_true, mut feats) = (vec![], vec![], vec![]);
    for &q in &tq {
        u.push(r.f64());
        cl_true.push(w.cl12[w.qs[q].rel[0] as usize] as usize);
        feats.push(w.feat.get(&w.qs[q].tid).cloned().unwrap_or_default());
    }
    Env { u, cl_true, feats, p_pre, p_post, kw }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Gate {
    Safe,
    G1,
}
#[derive(Clone, Copy, Debug)]
pub struct LoopCfg {
    pub gate: Gate,
    pub thr: f32,
    pub min_n: usize,
    pub k: usize,
    pub vote: bool,
    pub stale_credit: f64,
}

struct GlobalTs(PooledThompsonRouter);
impl Router for GlobalTs {
    fn name(&self) -> String {
        "global-ts".into()
    }
    fn select(&mut self, task: &Task, candidates: &[usize], rng: &mut Rng) -> usize {
        self.0.select(&Task { features: vec![], ..task.clone() }, candidates, rng)
    }
    fn update(&mut self, task: &Task, agent: usize, reward: f64) {
        self.0.update(&Task { features: vec![], ..task.clone() }, agent, reward)
    }
}
fn make_front(kw: &HashMap<String, usize>) -> Box<dyn Router> {
    let g = GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true };
    Box::new(Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw.clone(), NA, g))
}
pub fn make_baseline(name: &str, kw: &HashMap<String, usize>) -> Box<dyn Router> {
    let g = GuardConfig { tau: 0.55, z: 0.5, kw_a: 2.0, kw_b: 1.0, discount: 0.97, challenge: true };
    match name {
        "keyword" => Box::new(KeywordRouter::new(kw.clone(), NA)),
        "eps-greedy" => Box::new(EpsGreedyRouter::new(NA, 0.99, 0.02)),
        "pooled-ts" => Box::new(PooledThompsonRouter::new(NA, 1.0)),
        "global-ts" => Box::new(GlobalTs(PooledThompsonRouter::new(NA, 1.0))),
        "hebbian-robust" => Box::new(HebbianRouter::new(NA, heb_robust_cfg())),
        "warm(hebr)" => Box::new(Warm::new(HebbianRouter::new(NA, heb_robust_cfg()), kw.clone(), NA, 1.0).with_evidence(Evidence::MeanFeature)),
        _ => Box::new(Guarded::new(HebbianRouter::new(NA, heb_robust_cfg()), kw.clone(), NA, g)),
    }
}

#[derive(Clone, Default, Debug)]
pub struct RunOut {
    pub h2000: f64,
    pub vote_frac: f64,
    pub inj_prec: f64,
    pub stale_inj: f64,
}

pub fn run_loop(ws: &WSpec, env: &Env, lists: &[TaskRet], cfg: &LoopCfg, seed: u64, front_name: Option<&str>) -> RunOut {
    let cands: Vec<usize> = (0..NA).collect();
    let mut fr: Box<dyn Router> = match front_name {
        Some(n) => make_baseline(n, &env.kw),
        None => make_front(&env.kw),
    };
    let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(17));
    let mut astats: HashMap<u64, ([f64; NA], [f64; NA])> = HashMap::new();
    let mut gate = [0.0f64; 4];
    let (mut probe, pint) = (0usize, 25usize);
    let (mut fref, mut fref_n) = (0.5f64, 0.0f64);
    let (mut n_vote, mut n_rel, mut n_stale) = (0usize, 0usize, 0usize);
    let mut cum = 0.0f64;
    for t in 0..NT {
        let mut task = Task { id: t as u64, task_type: None, features: env.feats[t].clone() };
        let mut by_vote = false;
        let mut doc_rel = false;
        let mut stale_rel = false;
        let mut stale_inj = false;
        let mut found_anchor: Option<u64> = None;
        let a = if cfg.vote {
            'blk: {
                let lt = &lists[t];
                let admitted: Vec<&Cand> = lt.c[..lt.n].iter().filter(|c| c.cos >= cfg.thr).take(cfg.k).collect();
                if let Some(d0) = admitted.first() {
                    let mut votes: Vec<(usize, f32)> = vec![];
                    for c in &admitted {
                        match votes.iter_mut().find(|x| x.0 == c.cl as usize) {
                            Some(x) => x.1 += c.cos,
                            None => votes.push((c.cl as usize, c.cos)),
                        }
                    }
                    let win = votes.iter().fold((usize::MAX, f32::MIN), |b, &x| if x.1 > b.1 || (x.1 == b.1 && x.0 < b.0) { x } else { b }).0;
                    found_anchor = Some(win as u64);
                    doc_rel = ws.ctx_real && d0.rel;
                    stale_rel = ws.ctx_real && d0.stale_rel;
                    stale_inj = d0.stale;
                }
                let mut best: Option<(usize, f64, f64, f64)> = None;
                if let Some(an) = found_anchor {
                    if let Some(rw) = astats.get(&an) {
                        for a in 0..NA {
                            let (s, n) = (rw.0[a], rw.1[a]);
                            if n < cfg.min_n as f64 - 1e-9 {
                                continue;
                            }
                            let m = (1.0 + s) / (2.0 + n);
                            if m >= 0.7 && best.is_none_or(|b| m > b.1) {
                                best = Some((a, m, s, n));
                            }
                        }
                    }
                }
                let g1_open = best.is_none() || {
                    let rate_ok = gate[1] < 6.0 || gate[3] < 6.0 || gate[0] / gate[1] >= gate[2] / gate[3] - 0.02;
                    if rate_ok {
                        true
                    } else {
                        probe += 1;
                        if probe >= pint {
                            probe = 0;
                            true
                        } else {
                            false
                        }
                    }
                };
                let mut open = false;
                if let Some((_, _, s, n)) = best {
                    open = match cfg.gate {
                        Gate::G1 => g1_open,
                        Gate::Safe => {
                            let conf = 1.0 - inc_beta(1.0 + s, 1.0 + (n - s), fref.clamp(1e-6, 1.0 - 1e-6));
                            conf >= 0.8 && g1_open
                        }
                    };
                }
                let inj = open || (cfg.gate == Gate::G1 && found_anchor.is_some());
                if let (true, Some(an)) = (inj, found_anchor) {
                    task.features.push(format!("kn:{an}"));
                }
                match best {
                    Some((a, ..)) if open => {
                        by_vote = true;
                        n_vote += 1;
                        if doc_rel {
                            n_rel += 1;
                        }
                        if stale_inj {
                            n_stale += 1;
                        }
                        a
                    }
                    _ => fr.select(&task, &cands, &mut rng),
                }
            }
        } else {
            fr.select(&task, &cands, &mut rng)
        };
        let tab = if ws.drift && t >= NT / 2 { &env.p_post } else { &env.p_pre };
        let mut pe = tab[env.cl_true[t]][a];
        if by_vote {
            pe = if doc_rel {
                pe + ws.gain * (1.0 - pe)
            } else if stale_rel && cfg.stale_credit > 0.0 {
                pe + cfg.stale_credit * ws.gain * (1.0 - pe)
            } else {
                pe * (1.0 - PEN)
            };
        }
        let s = if env.u[t] < pe { 1.0 } else { 0.0 };
        fr.update(&task, a, s);
        if cfg.vote {
            if let Some(an) = found_anchor {
                let rw = astats.entry(an).or_insert(([0.0; NA], [0.0; NA]));
                rw.0[a] += s;
                rw.1[a] += 1.0;
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
        }
        cum += pe;
    }
    RunOut { h2000: cum / NT as f64, vote_frac: n_vote as f64 / NT as f64, inj_prec: if n_vote > 0 { n_rel as f64 / n_vote as f64 } else { f64::NAN }, stale_inj: if n_vote > 0 { n_stale as f64 / n_vote as f64 } else { f64::NAN } }
}

// =============================================================================================== stats

#[derive(Clone, Debug)]
pub struct Cmp {
    pub diff: f64,
    pub lo: f64,
    pub hi: f64,
    pub p: f64,
    pub tost_p: f64,
    pub n: usize,
}
fn one_sided_p(t: f64, df: usize) -> f64 {
    let p2 = t_two_sided_p(t, df);
    if t >= 0.0 { p2 / 2.0 } else { 1.0 - p2 / 2.0 }
}
/// paired a-b (NaN pairs dropped); TOST p-value for equivalence margin `eq`
pub fn cmp(a: &[f64], b: &[f64], eq: f64) -> Cmp {
    let (mut x, mut y) = (vec![], vec![]);
    for (p, q) in a.iter().zip(b) {
        if p.is_finite() && q.is_finite() {
            x.push(*p);
            y.push(*q);
        }
    }
    if x.len() < 3 {
        return Cmp { diff: f64::NAN, lo: f64::NAN, hi: f64::NAN, p: f64::NAN, tost_p: f64::NAN, n: x.len() };
    }
    let d = paired_diff(&x, &y);
    let diffs: Vec<f64> = x.iter().zip(&y).map(|(p, q)| p - q).collect();
    let n = diffs.len();
    let sm = summarize(&diffs);
    let tost = if sm.sd > 1e-15 {
        let se = sm.sd / (n as f64).sqrt();
        one_sided_p((sm.mean + eq) / se, n - 1).max(one_sided_p((eq - sm.mean) / se, n - 1))
    } else if sm.mean.abs() < eq {
        0.0
    } else {
        1.0
    };
    Cmp { diff: d.diff.mean, lo: d.diff.lo, hi: d.diff.hi, p: paired_t_pvalue(&x, &y), tost_p: tost, n }
}
pub fn mean(v: &[f64]) -> f64 {
    let f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if f.is_empty() { f64::NAN } else { f.iter().sum::<f64>() / f.len() as f64 }
}
pub fn holm(p: &[f64]) -> Vec<f64> {
    holm_adjust(p)
}
