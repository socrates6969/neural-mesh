//! H1 / part 2: COMPUTE-ONCE EMBEDDING. Does the same task text get encoded more than once per decision, and
//! does `MemoEmbedding` remove those calls with bit-identical decisions?
//!
//! Decision pipeline (what composing the library pieces does today, same as shortlab's `WarmCase`):
//!   A. knowledge:   e = embed_task(task); kn.retrieve(e); kn.remember(e)           (Knowledge / KnowledgeRouter / ResearchSession)
//!   B. router:      Warm<CaseRouter>.select  -> local_mass(task) [embed]  (+ posteriors(task) [embed] when the learned arm runs)
//!   C. feedback:    Warm<CaseRouter>.update  -> inner update(task)  [embed]
//! The encoder is the MiniLM TableEmbedding (precomputed f6 fixture, the same text->vector map a real encoder
//! would produce) or the hash embedder; the cost of a REAL encoder is modelled at 2.4 ms/text (results/f6_py_latency.json).
//!   cargo run --release --offline --bin h1_embed -- --part ident     # 30 seeds x 10 worlds, 4 arms, digest + exact equality
//!   cargo run --release --offline --bin h1_embed -- --part time      # per-embed ns and per-decision wall (table/hash/spin 2.4ms)
//! SYNTHETIC worlds (f5/f6 corpus, templated text); numbers are not real-traffic numbers.
#[path = "f6_common/mod.rs"]
mod f6c;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use f6c::*;
use neural_mesh::casebased::{CaseConfig, CaseRouter};
use neural_mesh::embedding::{CachedEmbedding, Embedding, HashEmbedding, MemoEmbedding, TableEmbedding};
use neural_mesh::knowledge::{Knowledge, KnowledgeConfig};
use neural_mesh::prior::{Evidence, Warm};
use neural_mesh::router::{Rng, Router, Task};
use rayon::prelude::*;

const N_TASKS: usize = 2000;
const NAG: usize = 5;
/// real-encoder cost model (python onnxruntime MiniLM-L6, batch 1..32): ms per text
const ENC_MS: f64 = 2.4;

/// counts calls to the underlying encoder (every arm sits on top of one of these)
struct Counting<E: Embedding> {
    inner: E,
    calls: AtomicU64,
    spin_ns: u64,
}
impl<E: Embedding> Counting<E> {
    fn new(inner: E) -> Self {
        Counting { inner, calls: AtomicU64::new(0), spin_ns: 0 }
    }
    fn spin(&self) {
        if self.spin_ns > 0 {
            let t = Instant::now();
            while (t.elapsed().as_nanos() as u64) < self.spin_ns {
                std::hint::spin_loop();
            }
        }
    }
}
impl<E: Embedding> Embedding for Counting<E> {
    fn dim(&self) -> usize {
        self.inner.dim()
    }
    fn embed(&self, t: &str) -> Vec<f32> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.spin();
        self.inner.embed(t)
    }
    fn name(&self) -> &str {
        "counting"
    }
    fn embed_task(&self, t: &Task) -> Vec<f32> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.spin();
        self.inner.embed_task(t)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// every consumer calls the encoder (today's behaviour)
    Baseline,
    /// MemoEmbedding cap 4: compute-once PER DECISION (same text hits within the decision)
    MemoDecision,
    /// MemoEmbedding cap 4096: also across decisions (repeated texts)
    MemoStream,
    /// existing CachedEmbedding cap 4096 (for comparison)
    CachedStream,
}
const ARMS: [Arm; 4] = [Arm::Baseline, Arm::MemoDecision, Arm::MemoStream, Arm::CachedStream];
fn arm_name(a: Arm) -> &'static str {
    match a {
        Arm::Baseline => "baseline",
        Arm::MemoDecision => "memo(cap4)",
        Arm::MemoStream => "memo(cap4096)",
        Arm::CachedStream => "cached(cap4096)",
    }
}

fn table_of(f: &Corpus) -> TableEmbedding {
    TableEmbedding::from_pairs("minilm-fixture", Some(DIM), f.texts.iter().cloned().zip(f.mini.iter().cloned()), false, false).expect("table")
}

/// text stream of a world: (task texts, concept label or -1)
fn stream(f: &Corpus, spec: &Spec, seed: u64) -> (Vec<String>, Vec<i32>) {
    let mut r = Rng::new(seed.wrapping_mul(7919).wrapping_add(0xF5F5_1234));
    let kk = if spec.k == 0 { N_TASKS } else { spec.k.min(f.nconc) };
    let mut pool = f.dist.clone();
    for i in (1..pool.len()).rev() {
        let j = r.below(i + 1);
        pool.swap(i, j);
    }
    let (mut di, mut texts, mut lab) = (0usize, vec![], vec![]);
    let switch = N_TASKS / 2;
    for t in 0..N_TASKS {
        let isd = spec.k == 0 || (spec.mixed > 0.0 && r.f64() < spec.mixed);
        if isd {
            texts.push(f.texts[pool[di % pool.len()]].clone());
            lab.push(-1);
            di += 1;
        } else {
            let c = r.below(kk);
            let src = match spec.corp {
                Corp::Q => &f.q[c],
                Corp::Subj => &f.subj[c],
                Corp::Para => &f.para[c],
                Corp::QPara => &f.qpara[c],
                Corp::QThenPara => {
                    if t < switch {
                        &f.q[c]
                    } else {
                        &f.para[c]
                    }
                }
            };
            texts.push(f.texts[src[r.below(src.len())]].clone());
            lab.push(c as i32);
        }
    }
    (texts, lab)
}

fn worlds_h1() -> Vec<Spec> {
    use neural_mesh::sim::Scenario::Fixed;
    let mk = |name: &str, corp: Corp, k: usize, mixed: f64| Spec { name: name.into(), src: Src::Mini, corp, scenario: Fixed, k, w: 0.8, mixed, treatment: true, targets: vec![], win: 200 };
    vec![
        mk("q", Corp::Q, 12, 0.0),
        mk("subj3", Corp::Subj, 12, 0.0),
        mk("para", Corp::Para, 12, 0.0),
        mk("qpara", Corp::QPara, 12, 0.0),
        mk("q-mixed50", Corp::Q, 12, 0.5),
        mk("q-k6", Corp::Q, 6, 0.0),
        mk("q-then-para", Corp::QThenPara, 12, 0.0),
        mk("para-mixed50", Corp::Para, 12, 0.5),
        mk("subj3-mixed50", Corp::Subj, 12, 0.5),
        mk("unique(no repeats)", Corp::Q, 0, 0.0),
    ]
}

struct RunOut {
    /// per decision: (anchor, agent, reward bits, hits hash)
    seq: Vec<(u64, usize, u64, u64)>,
    digest: u64,
    probe: Vec<u64>,
    encoder_calls: u64,
    wall_s: f64,
}

fn fnv(h: &mut u64, x: u64) {
    for b in x.to_le_bytes() {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn run_decisions(enc: Arc<dyn Embedding>, counter: &dyn Fn() -> u64, texts: &[String], lab: &[i32], seed: u64) -> RunOut {
    let mut kc = KnowledgeConfig::new(DIM);
    kc.k = 5;
    kc.sim_min = 0.5;
    let mut kn = Knowledge::new_shared(kc).with_embedder(enc.clone());
    // case router embedder closure -> the SAME shared encoder handle (what a composition would do)
    let by_id: Arc<Vec<Task>> = Arc::new(texts.iter().enumerate().map(|(i, t)| Task { id: i as u64, task_type: None, features: t.split_whitespace().map(String::from).collect() }).collect());
    let e2 = enc.clone();
    let embedder: neural_mesh::casebased::Embedder = Arc::new(move |t: &Task| Some(e2.embed_task(t)));
    let cr = CaseRouter::new(NAG, DIM, CaseConfig::default(), embedder);
    let mut warm = Warm::new(cr, std::collections::HashMap::new(), NAG, 3.0).with_evidence(Evidence::Inner(CaseRouter::local_mass));
    let cands: Vec<usize> = (0..NAG).collect();
    let mut rng = Rng::new(seed * 7 + 1);
    let mut orng = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(17));
    // concept -> best agent
    let mut best = vec![0usize; 64];
    for b in best.iter_mut() {
        *b = orng.below(NAG);
    }
    let mut seq = Vec::with_capacity(texts.len());
    let mut digest: u64 = 0xcbf2_9ce4_8422_2325;
    let t0 = Instant::now();
    for (i, task) in by_id.iter().enumerate() {
        // A. knowledge
        let e = kn.embed_task(task);
        let r = kn.retrieve(&e);
        let rec = kn.remember(&e, &r);
        let anchor = r.anchor.unwrap_or(rec);
        let mut hh: u64 = 0xcbf2_9ce4_8422_2325;
        for &(id, c) in &r.hits {
            fnv(&mut hh, id);
            fnv(&mut hh, c.to_bits() as u64);
        }
        // B. route
        let agent = warm.select(task, &cands, &mut rng);
        kn.commit(rec, agent);
        // outcome (independent of how the pieces embed)
        let p = if lab[i] >= 0 && best[lab[i] as usize % 64] == agent { 0.85 } else { 0.30 };
        let reward = if orng.f64() < p { 1.0 } else { 0.0 };
        // C. feedback
        warm.update(task, agent, reward);
        kn.credit(rec, reward);
        fnv(&mut digest, anchor);
        fnv(&mut digest, agent as u64);
        fnv(&mut digest, (reward as f64).to_bits());
        fnv(&mut digest, hh);
        seq.push((anchor, agent, (reward as f64).to_bits(), hh));
    }
    let wall_s = t0.elapsed().as_secs_f64();
    let encoder_calls = counter();
    // probe: final posteriors on 20 fixed tasks (after the call count is read)
    let mut probe = vec![];
    for task in by_id.iter().take(20) {
        let ps = warm.inner.posteriors(task, &cands);
        for (a, b) in ps {
            probe.push(a.to_bits());
            probe.push(b.to_bits());
        }
        probe.push(warm.inner.local_mass(task).to_bits());
    }
    RunOut { seq, digest, probe, encoder_calls, wall_s }
}

fn run_arm(table: &Arc<TableEmbedding>, hash_mode: bool, arm: Arm, texts: &[String], lab: &[i32], seed: u64, spin_ns: u64) -> RunOut {
    // base encoder (counted)
    let base_tab: Arc<Counting<Arc<TableEmbedding>>>;
    let base_hash: Arc<Counting<HashEmbedding>>;
    let base: Arc<dyn Embedding>;
    let base_calls: Box<dyn Fn() -> u64>;
    if hash_mode {
        let mut c = Counting::new(HashEmbedding::new(DIM));
        c.spin_ns = spin_ns;
        base_hash = Arc::new(c);
        base = base_hash.clone();
        let b = base_hash.clone();
        base_calls = Box::new(move || b.calls.load(Ordering::Relaxed));
    } else {
        let mut c = Counting::new(table.clone());
        c.spin_ns = spin_ns;
        base_tab = Arc::new(c);
        base = base_tab.clone();
        let b = base_tab.clone();
        base_calls = Box::new(move || b.calls.load(Ordering::Relaxed));
    }
    match arm {
        Arm::Baseline => run_decisions(base, &*base_calls, texts, lab, seed),
        Arm::MemoDecision | Arm::MemoStream => {
            let cap = if arm == Arm::MemoDecision { 4 } else { 4096 };
            let m = Arc::new(MemoEmbedding::new(base, cap));
            let enc: Arc<dyn Embedding> = m.clone();
            run_decisions(enc, &*base_calls, texts, lab, seed)
        }
        Arm::CachedStream => {
            let m = Arc::new(CachedEmbedding::new(base, 4096));
            let enc: Arc<dyn Embedding> = m.clone();
            run_decisions(enc, &*base_calls, texts, lab, seed)
        }
    }
}

fn arg<T: std::str::FromStr>(k: &str, d: T) -> T {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() {
    let part: String = arg("--part", "ident".to_string());
    let f = corpus();
    println!("h1_embed: corpus {} texts, dim {}; synthetic f5/f6 corpus (MiniLM table), NOT real traffic", f.texts.len(), DIM);
    let table = Arc::new(table_of(f));
    if part == "ident" {
        ident(f, &table);
    } else {
        timing(f, &table);
    }
}

fn ident(f: &Corpus, table: &Arc<TableEmbedding>) {
    let seeds: Vec<u64> = (1..=arg::<u64>("--seeds", 30)).collect();
    let ws = worlds_h1();
    println!("world | seed-count | decisions | enc calls/decision [baseline, memo4, memo4096, cached4096] | repeat-rate | bit-identical (all arms, all seeds)");
    let mut all_ok = true;
    let mut tot = [0u64; 4];
    let mut tot_dec = 0u64;
    let mut jrows = vec![];
    for hash_mode in [false, true] {
        let tag = if hash_mode { "HASH" } else { "MINILM-table" };
        println!("--- encoder = {tag}");
        for w in &ws {
            let res: Vec<(Vec<RunOut>, f64)> = seeds
                .par_iter()
                .map(|&s| {
                    let (texts, lab) = stream(f, w, s);
                    let mut seen = std::collections::HashSet::new();
                    let rep = texts.iter().filter(|t| !seen.insert((*t).clone())).count() as f64 / texts.len() as f64;
                    let outs: Vec<RunOut> = ARMS.iter().map(|&a| run_arm(table, hash_mode, a, &texts, &lab, s, 0)).collect();
                    (outs, rep)
                })
                .collect();
            let mut ok = true;
            let mut calls = [0u64; 4];
            let mut dec = 0u64;
            let mut rep = 0.0;
            for (outs, r) in &res {
                for k in 1..4 {
                    if outs[k].seq != outs[0].seq || outs[k].digest != outs[0].digest || outs[k].probe != outs[0].probe {
                        ok = false;
                    }
                }
                for k in 0..4 {
                    calls[k] += outs[k].encoder_calls;
                }
                dec += outs[0].seq.len() as u64;
                rep += r;
            }
            all_ok &= ok;
            if !hash_mode {
                for k in 0..4 {
                    tot[k] += calls[k];
                }
                tot_dec += dec;
            }
            let per: Vec<f64> = calls.iter().map(|&c| c as f64 / dec as f64).collect();
            println!("{:<22} | {} | {} | [{:.3}, {:.3}, {:.3}, {:.3}] | {:.3} | {}", w.name, seeds.len(), dec, per[0], per[1], per[2], per[3], rep / seeds.len() as f64, if ok { "YES" } else { "NO !!!" });
            jrows.push(serde_json::json!({"encoder": tag, "world": w.name, "seeds": seeds.len(), "decisions": dec, "calls_per_decision": per, "arms": ARMS.iter().map(|&a| arm_name(a)).collect::<Vec<_>>(), "repeat_rate": rep / seeds.len() as f64, "bit_identical": ok}));
        }
    }
    let per: Vec<f64> = tot.iter().map(|&c| c as f64 / tot_dec as f64).collect();
    println!("MiniLM-table pooled over 10 worlds x {} seeds: encoder calls/decision baseline {:.3}, memo(cap4) {:.3}, memo(cap4096) {:.3}, cached(cap4096) {:.3}", seeds.len(), per[0], per[1], per[2], per[3]);
    println!("modelled real-encoder cost @ {ENC_MS} ms/text: baseline {:.2} ms/decision, memo4 {:.2}, memo4096 {:.2}, cached4096 {:.2}", per[0] * ENC_MS, per[1] * ENC_MS, per[2] * ENC_MS, per[3] * ENC_MS);
    assert_eq!(table.misses(), 0, "table lookup missed: feature join != text");
    println!("ALL ARMS BIT-IDENTICAL TO BASELINE ON ALL SEEDS/WORLDS: {}", all_ok);
    let out = serde_json::json!({"rows": jrows, "pooled_minilm_calls_per_decision": per, "enc_ms": ENC_MS, "all_bit_identical": all_ok, "note": "synthetic f5/f6 corpus; decisions = (anchor, agent, reward, retrieval hits+sim bits) per task, digest + final posteriors/local_mass bits"});
    std::fs::write("results/h1_embed_ident.json", serde_json::to_string_pretty(&out).unwrap()).unwrap();
    if !all_ok {
        std::process::exit(1);
    }
}

fn timing(f: &Corpus, table: &Arc<TableEmbedding>) {
    // (1) per-embed latency of each layer on table / hash (honest overhead of the memo)
    let texts: Vec<&str> = f.texts.iter().map(|s| s.as_str()).take(512).collect();
    let tasks: Vec<Task> = texts.iter().enumerate().map(|(i, t)| Task { id: i as u64, task_type: None, features: t.split_whitespace().map(String::from).collect() }).collect();
    let bench = |name: &str, n: usize, f: &mut dyn FnMut(usize) -> f32| -> f64 {
        let mut s = 0.0f32;
        for i in 0..n / 10 {
            s += f(i);
        }
        let t0 = Instant::now();
        for i in 0..n {
            s += f(i);
        }
        let ns = t0.elapsed().as_nanos() as f64 / n as f64;
        std::hint::black_box(s);
        println!("  {:<46} {:>9.0} ns/embed", name, ns);
        ns
    };
    let n = 400_000;
    println!("per-embed latency (single thread, hit-path for the memos, 512 distinct texts):");
    let mut rows = vec![];
    let tab: Arc<dyn Embedding> = table.clone();
    rows.push(("table raw embed_task", bench("TableEmbedding.embed_task (raw)", n, &mut |i| tab.embed_task(&tasks[i % 512])[0])));
    let c = CachedEmbedding::new(table.clone(), 4096);
    rows.push(("cached hit", bench("CachedEmbedding(Table) hit embed_task", n, &mut |i| c.embed_task(&tasks[i % 512])[0])));
    let m = MemoEmbedding::new(table.clone(), 4096);
    rows.push(("memo hit (Vec out)", bench("MemoEmbedding(Table) hit embed_task (Vec)", n, &mut |i| m.embed_task(&tasks[i % 512])[0])));
    rows.push(("memo hit (shared)", bench("MemoEmbedding(Table) hit embed_task_shared (Arc)", n, &mut |i| m.embed_task_shared(&tasks[i % 512])[0])));
    let m0 = MemoEmbedding::new(table.clone(), 0);
    rows.push(("memo cap0 (always miss)", bench("MemoEmbedding(Table) cap0 pass-through", n, &mut |i| m0.embed_task(&tasks[i % 512])[0])));
    let h: Arc<dyn Embedding> = Arc::new(HashEmbedding::new(DIM));
    rows.push(("hash raw", bench("HashEmbedding(384).embed_task (raw)", n / 4, &mut |i| h.embed_task(&tasks[i % 512])[0])));
    let ch = CachedEmbedding::new(HashEmbedding::new(DIM), 4096);
    rows.push(("cached(hash) hit", bench("CachedEmbedding(Hash) hit", n / 4, &mut |i| ch.embed_task(&tasks[i % 512])[0])));
    let mh = MemoEmbedding::new(HashEmbedding::new(DIM), 4096);
    rows.push(("memo(hash) hit (shared)", bench("MemoEmbedding(Hash) hit embed_task_shared", n / 4, &mut |i| mh.embed_task_shared(&tasks[i % 512])[0])));
    // (2) per-decision wall: modelled + real spin
    println!("per-decision wall, q world, 3 seeds (decision = knowledge retrieve+remember, Warm<CaseRouter> select, update):");
    let w = worlds_h1().into_iter().next().unwrap();
    let mut jt = vec![];
    for (label, hash_mode, spin) in [("MiniLM table (cheap lookup)", false, 0u64), ("hash (cheap)", true, 0u64), ("simulated real encoder: table + spin 2.4 ms", false, (ENC_MS * 1e6) as u64)] {
        let nt = if spin > 0 { 150 } else { N_TASKS };
        for arm in ARMS {
            let mut wall = 0.0;
            let mut calls = 0u64;
            let mut dec = 0usize;
            for s in 1..=3u64 {
                let (mut texts, mut lab) = stream(f, &w, s);
                texts.truncate(nt);
                lab.truncate(nt);
                let o = run_arm(table, hash_mode, arm, &texts, &lab, s, spin);
                wall += o.wall_s;
                calls += o.encoder_calls;
                dec += o.seq.len();
            }
            println!("  {:<44} {:<16} {:>9.3} ms/decision wall, {:.3} enc calls/decision", label, arm_name(arm), 1e3 * wall / dec as f64, calls as f64 / dec as f64);
            jt.push(serde_json::json!({"encoder": label, "arm": arm_name(arm), "ms_per_decision": 1e3 * wall / dec as f64, "calls_per_decision": calls as f64 / dec as f64, "decisions": dec}));
        }
    }
    let out = serde_json::json!({"per_embed_ns": rows.iter().map(|(n, v)| serde_json::json!({"what": n, "ns": v})).collect::<Vec<_>>(), "per_decision": jt, "enc_ms_model": ENC_MS, "box": "shared, loaded, 2 rayon threads; indicative"});
    std::fs::write("results/h1_embed_timing.json", serde_json::to_string_pretty(&out).unwrap()).unwrap();
}
