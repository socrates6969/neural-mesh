//! Memory lab: tiered memory + lessons-from-mistakes, measured.
//!
//!   cargo run --release --bin memory_lab [-- --quick] [--part tiered|lessons|all]
//!
//! Part 1 (tiered): synthetic grouped workload where queries favour recent
//! items and a small "useful" subset recurs; TieredMemory variants vs a single
//! exact/HNSW index of everything, under a fixed byte budget.
//! Part 2 (lessons): seeded routing simulation with recurring failure modes;
//! HebbianRouter with vs without LessonStore score adjustment on paired seeds.
//!
//! Writes results/memory_lab.json and prints tables. Every number printed is a
//! mean over seeds with a 95% t-interval (stats.rs).

use std::time::Instant;

use neural_mesh::hebbian::{HebbianConfig, HebbianRouter};
use neural_mesh::lessons::{LessonConfig, LessonStore};
use neural_mesh::memory::{
    normalize_in_place, synth::Clustered, synth::Rng, FlatIndex, HnswIndex, HnswParams, MemoryIndex,
};
use neural_mesh::router::{Rng as RRng, Router, Task};
use neural_mesh::stats::{paired_diff, summarize, Summary};
use neural_mesh::tiered::{Compression, TieredConfig, TieredMemory};
use serde_json::{json, Value};

// ===========================================================================
// Part 1: tiered memory
// ===========================================================================

const DIM: usize = 384;
const GSIZE: usize = 10;
const BLOCK_GROUPS: usize = 10; // groups per arrival block
const K: usize = 10;

#[derive(Clone)]
struct WCfg {
    groups: usize,
    queries: usize,
    warm_items: usize,
    short_cap: usize,
    recency_window: usize,
    useful_frac: f64,
    member_noise: f32,
    query_noise: f32,
    budget_frac: f64,
    /// (p_recent, p_useful, p_old)
    mix: (f64, f64, f64),
}

impl WCfg {
    fn full() -> Self {
        Self {
            groups: 2000,
            queries: 1500,
            warm_items: 2000,
            short_cap: 1500,
            recency_window: 1000,
            useful_frac: 0.05,
            member_noise: 1.8,
            query_noise: 1.0,
            budget_frac: 0.25,
            mix: (0.55, 0.30, 0.15),
        }
    }
    fn quick() -> Self {
        Self { groups: 600, queries: 400, warm_items: 800, short_cap: 450, recency_window: 300, ..Self::full() }
    }
    fn n_items(&self) -> usize {
        self.groups * GSIZE
    }
    fn budget_bytes(&self) -> usize {
        (self.budget_frac * (self.n_items() * (DIM * 4 + 8)) as f64) as usize
    }
}

struct Query {
    t: usize, // asked after this many items were ingested
    group: usize,
    class: usize, // 0 recent, 1 useful, 2 old
    vec: Vec<f32>,
}

struct World {
    items: Vec<f32>, // item id -> row
    order: Vec<u64>, // arrival order (item ids)
    queries: Vec<Query>,
}

fn noisy(center: &[f32], norm: f32, rng: &mut Rng) -> Vec<f32> {
    let scale = norm / (center.len() as f32).sqrt();
    let mut v: Vec<f32> = center.iter().map(|c| c + scale * rng.normal()).collect();
    normalize_in_place(&mut v);
    v
}

fn make_world(seed: u64, c: &WCfg) -> World {
    let mut mix_gen = Clustered::hierarchical(DIM, 64, 16, 0.8, 0.6, seed);
    let centers = mix_gen.sample(c.groups);
    let mut rng = Rng::new(seed ^ 0xC0FFEE);
    let mut items = Vec::with_capacity(c.n_items() * DIM);
    for g in 0..c.groups {
        for _ in 0..GSIZE {
            items.extend(noisy(&centers[g * DIM..(g + 1) * DIM], c.member_noise, &mut rng));
        }
    }
    // arrival order: blocks of BLOCK_GROUPS groups, items shuffled within a block
    let mut order = Vec::with_capacity(c.n_items());
    let block_items = BLOCK_GROUPS * GSIZE;
    for b in 0..c.groups / BLOCK_GROUPS {
        let mut blk: Vec<u64> = (b * block_items..(b + 1) * block_items).map(|i| i as u64).collect();
        for i in (1..blk.len()).rev() {
            blk.swap(i, rng.below(i + 1));
        }
        order.extend(blk);
    }
    // useful groups
    let n_useful = ((c.groups as f64 * c.useful_frac).round() as usize).max(1);
    let mut useful = vec![false; c.groups];
    let mut cnt = 0;
    while cnt < n_useful {
        let g = rng.below(c.groups);
        if !useful[g] {
            useful[g] = true;
            cnt += 1;
        }
    }
    let useful_list: Vec<usize> = (0..c.groups).filter(|g| useful[*g]).collect();
    // query plan at block boundaries
    let w_blocks = c.recency_window / block_items;
    let bounds: Vec<usize> =
        (1..=c.groups / BLOCK_GROUPS).map(|b| b * block_items).filter(|t| *t >= c.warm_items).collect();
    let per = c.queries as f64 / bounds.len() as f64;
    let mut acc = 0.0;
    let mut queries = Vec::new();
    for &t in &bounds {
        acc += per;
        while acc >= 1.0 {
            acc -= 1.0;
            let completed = t / block_items;
            let u = rng.uniform();
            let (p_recent, p_useful, _) = c.mix;
            let mut class = if u < p_recent { 0 } else if u < p_recent + p_useful { 1 } else { 2 };
            let recent_pick = |rng: &mut Rng| {
                let lo = completed.saturating_sub(w_blocks);
                (lo + rng.below(completed - lo)) * BLOCK_GROUPS + rng.below(BLOCK_GROUPS)
            };
            let group = match class {
                1 => {
                    let avail: Vec<usize> =
                        useful_list.iter().copied().filter(|g| g / BLOCK_GROUPS < completed).collect();
                    if avail.is_empty() {
                        class = 0;
                        recent_pick(&mut rng)
                    } else {
                        avail[rng.below(avail.len())]
                    }
                }
                2 => {
                    let old_blocks = completed.saturating_sub(w_blocks);
                    let mut pick = None;
                    for _ in 0..50 {
                        if old_blocks == 0 {
                            break;
                        }
                        let g = rng.below(old_blocks) * BLOCK_GROUPS + rng.below(BLOCK_GROUPS);
                        if !useful[g] {
                            pick = Some(g);
                            break;
                        }
                    }
                    match pick {
                        Some(g) => g,
                        None => {
                            class = 0;
                            recent_pick(&mut rng)
                        }
                    }
                }
                _ => recent_pick(&mut rng),
            };
            let vec = noisy(&centers[group * DIM..(group + 1) * DIM], c.query_noise, &mut rng);
            queries.push(Query { t, group, class, vec });
        }
    }
    World { items, order, queries }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    FlatAll,
    HnswAll,
    Tiered { forgetting: bool, feedback: bool, compress: bool },
}

#[derive(Clone)]
struct Spec {
    name: &'static str,
    kind: Kind,
}

#[derive(Clone, Copy)]
struct Knobs {
    halflife: f64,
    hot_cap: usize,
}

enum Sys {
    Plain(Box<dyn MemoryIndex>),
    Tiered { m: TieredMemory, feedback: bool },
}

impl Sys {
    fn build(spec: &Spec, c: &WCfg, kn: Knobs) -> Sys {
        match spec.kind {
            Kind::FlatAll => Sys::Plain(Box::new(FlatIndex::with_capacity(DIM, c.n_items()))),
            Kind::HnswAll => Sys::Plain(Box::new(HnswIndex::new(
                DIM,
                HnswParams { m: 16, ef_construction: 200, ef_search: 64, max_elements: c.n_items() },
            ))),
            Kind::Tiered { forgetting, feedback, compress } => {
                let mut t = TieredConfig::new(DIM);
                t.short_cap = c.short_cap;
                t.forgetting = forgetting;
                t.byte_budget = if forgetting { c.budget_bytes() } else { usize::MAX };
                t.value_halflife = kn.halflife;
                t.consolidate_every = BLOCK_GROUPS * GSIZE;
                if compress {
                    t.compression = Some(Compression { bit_width: 4, hot_cap: kn.hot_cap });
                }
                Sys::Tiered { m: TieredMemory::new(t), feedback }
            }
        }
    }
    fn ingest(&mut self, ids: &[u64], items: &[f32]) {
        match self {
            Sys::Plain(i) => {
                let mut flat = Vec::with_capacity(ids.len() * DIM);
                for id in ids {
                    flat.extend_from_slice(&items[*id as usize * DIM..(*id as usize + 1) * DIM]);
                }
                i.add_batch(ids, &flat);
            }
            Sys::Tiered { m, .. } => {
                for id in ids {
                    m.add(*id, &items[*id as usize * DIM..(*id as usize + 1) * DIM], "");
                }
            }
        }
    }
    fn search(&mut self, q: &[f32]) -> Vec<(u64, f32)> {
        match self {
            Sys::Plain(i) => i.search(q, K),
            Sys::Tiered { m, .. } => m.search(q, K),
        }
    }
    fn feedback(&mut self, id: u64, r: f64) {
        if let Sys::Tiered { m, feedback: true } = self {
            m.record_outcome(id, r);
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Sys::Plain(i) => i.bytes(),
            Sys::Tiered { m, .. } => m.bytes(),
        }
    }
    fn retained(&self) -> usize {
        match self {
            Sys::Plain(i) => i.len(),
            Sys::Tiered { m, .. } => m.len(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct RunRes {
    recall: f64,
    by_class: [f64; 3],
    lat_us: f64,
    bytes_end: f64,
    bytes_max: f64,
    retained: f64,
}

fn run_one(w: &World, c: &WCfg, spec: &Spec, kn: Knobs) -> RunRes {
    let mut sys = Sys::build(spec, c, kn);
    let block_items = BLOCK_GROUPS * GSIZE;
    let mut qi = 0;
    let mut sum = [0.0f64; 3];
    let mut cnt = [0usize; 3];
    let mut lat = 0.0;
    let mut bytes_max = 0usize;
    for (b, chunk) in w.order.chunks(block_items).enumerate() {
        sys.ingest(chunk, &w.items);
        let t = (b + 1) * block_items;
        bytes_max = bytes_max.max(sys.bytes());
        while qi < w.queries.len() && w.queries[qi].t == t {
            let q = &w.queries[qi];
            let t0 = Instant::now();
            let res = sys.search(&q.vec);
            lat += t0.elapsed().as_secs_f64() * 1e6;
            let lo = (q.group * GSIZE) as u64;
            let mut hit = 0;
            for (id, _) in &res {
                let rel = *id >= lo && *id < lo + GSIZE as u64;
                hit += rel as usize;
                sys.feedback(*id, if rel { 1.0 } else { 0.0 });
            }
            sum[q.class] += hit as f64 / GSIZE as f64;
            cnt[q.class] += 1;
            qi += 1;
        }
    }
    let nq = w.queries.len().max(1);
    let by_class = [0, 1, 2].map(|i| if cnt[i] > 0 { sum[i] / cnt[i] as f64 } else { f64::NAN });
    RunRes {
        recall: sum.iter().sum::<f64>() / nq as f64,
        by_class,
        lat_us: lat / nq as f64,
        bytes_end: sys.bytes() as f64,
        bytes_max: bytes_max as f64,
        retained: sys.retained() as f64,
    }
}

fn specs() -> Vec<Spec> {
    let t = |forgetting, feedback, compress| Kind::Tiered { forgetting, feedback, compress };
    vec![
        Spec { name: "flat-all", kind: Kind::FlatAll },
        Spec { name: "hnsw-all (M16,ef64)", kind: Kind::HnswAll },
        Spec { name: "tiered keep-all", kind: t(false, false, false) },
        Spec { name: "tiered keep-all +cold4", kind: t(false, false, true) },
        Spec { name: "tiered FIFO-forget @B", kind: t(true, false, false) },
        Spec { name: "tiered value-forget @B", kind: t(true, true, false) },
        Spec { name: "tiered FIFO-forget +cold4 @B", kind: t(true, false, true) },
        Spec { name: "tiered value-forget +cold4 @B", kind: t(true, true, true) },
    ]
}

fn sm(xs: &[f64]) -> Value {
    let s = summarize(xs);
    json!({"mean": s.mean, "ci95": s.half_width, "sd": s.sd, "n": s.n})
}

fn fmt(s: &Summary, scale: f64, prec: usize) -> String {
    format!("{:.*} ±{:.*}", prec, s.mean * scale, prec, s.half_width * scale)
}

fn run_grid(seeds: &[u64], c: &WCfg, kn: Knobs, specs: &[Spec], parallel: bool) -> Vec<Vec<RunRes>> {
    // result[spec][seed]
    let per_seed: Vec<Vec<RunRes>> = if parallel {
        std::thread::scope(|sc| {
            let hs: Vec<_> = seeds
                .iter()
                .map(|&s| {
                    sc.spawn(move || {
                        let w = make_world(s, c);
                        specs.iter().map(|sp| run_one(&w, c, sp, kn)).collect::<Vec<_>>()
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        })
    } else {
        seeds
            .iter()
            .map(|&s| {
                let w = make_world(s, c);
                specs.iter().map(|sp| run_one(&w, c, sp, kn)).collect()
            })
            .collect()
    };
    (0..specs.len()).map(|i| per_seed.iter().map(|r| r[i].clone()).collect()).collect()
}

fn col(rs: &[RunRes], f: impl Fn(&RunRes) -> f64) -> Vec<f64> {
    rs.iter().map(f).collect()
}

fn part_tiered(quick: bool, mn: Option<f32>, qn: Option<f32>) -> Value {
    let mut base = if quick { WCfg::quick() } else { WCfg::full() };
    if let Some(x) = mn {
        base.member_noise = x;
    }
    if let Some(x) = qn {
        base.query_noise = x;
    }
    let sp = specs();
    let tune_seeds: Vec<u64> = if quick { vec![1, 2] } else { vec![1, 2, 3, 4] };
    let eval_seeds: Vec<u64> = if quick { (11..=14).collect() } else { (11..=22).collect() };
    println!("\n=== PART 1: tiered memory ===");
    println!(
        "dim {DIM}, {} groups x {GSIZE} = {} items, {} queries, short ring {}, budget {:.2} MB ({}% of flat-all)",
        base.groups,
        base.n_items(),
        base.queries,
        base.short_cap,
        base.budget_bytes() as f64 / 1e6,
        (base.budget_frac * 100.0) as u32
    );

    // ---- tune halflife / hot_cap on tuning seeds only ----
    let tune_specs: Vec<Spec> = sp.iter().filter(|s| s.name.starts_with("tiered value-forget")).cloned().collect();
    let mut tune_log = Vec::new();
    let mut best_hl = (f64::NEG_INFINITY, 10_000.0);
    for hl in [3_000.0, 10_000.0, 30_000.0, 100_000.0] {
        let kn = Knobs { halflife: hl, hot_cap: base.short_cap };
        let r = run_grid(&tune_seeds, &base, kn, &tune_specs[..1], true);
        let m = summarize(&col(&r[0], |x| x.recall)).mean;
        println!("  tune halflife {hl:>8}: value-forget recall {m:.4}");
        tune_log.push(json!({"halflife": hl, "hot_cap": null, "variant": "value-forget", "recall": m}));
        if m > best_hl.0 {
            best_hl = (m, hl);
        }
    }
    let mut best_hot = (f64::NEG_INFINITY, base.short_cap / 2);
    for frac in [0.02, 0.05, 0.1, 0.25, 0.5, 1.0] {
        let hot_cap = ((base.short_cap as f64 * frac) as usize).max(16);
        let kn = Knobs { halflife: best_hl.1, hot_cap };
        let r = run_grid(&tune_seeds, &base, kn, &tune_specs[1..], true);
        let m = summarize(&col(&r[0], |x| x.recall)).mean;
        println!("  tune hot_cap {hot_cap:>5}: value-forget+cold4 recall {m:.4}");
        tune_log.push(json!({"halflife": best_hl.1, "hot_cap": hot_cap, "variant": "value-forget+cold4", "recall": m}));
        if m > best_hot.0 {
            best_hot = (m, hot_cap);
        }
    }
    let kn = Knobs { halflife: best_hl.1, hot_cap: best_hot.1 };
    println!("  chosen: halflife {} ticks, hot_cap {} items", kn.halflife, kn.hot_cap);

    // ---- main evaluation (sequential so latency is not contended) ----
    let res = run_grid(&eval_seeds, &base, kn, &sp, false);
    let budget = base.budget_bytes() as f64;
    println!(
        "\n{:<32} {:>15} {:>15} {:>15} {:>15} {:>14} {:>13} {:>9}",
        "method", "recall@10 all", "recent", "useful", "old", "latency us", "MB end", "retained"
    );
    let mut rows = Vec::new();
    for (s, r) in sp.iter().zip(&res) {
        let all = summarize(&col(r, |x| x.recall));
        let cls: Vec<Summary> = (0..3).map(|i| summarize(&col(r, |x| x.by_class[i]))).collect();
        let lat = summarize(&col(r, |x| x.lat_us));
        let by = summarize(&col(r, |x| x.bytes_end));
        let bm = summarize(&col(r, |x| x.bytes_max));
        let ret = summarize(&col(r, |x| x.retained));
        println!(
            "{:<32} {:>15} {:>15} {:>15} {:>15} {:>14} {:>13} {:>9.0}",
            s.name,
            fmt(&all, 1.0, 3),
            fmt(&cls[0], 1.0, 3),
            fmt(&cls[1], 1.0, 3),
            fmt(&cls[2], 1.0, 3),
            fmt(&lat, 1.0, 0),
            fmt(&by, 1e-6, 2),
            ret.mean
        );
        rows.push(json!({
            "method": s.name,
            "recall_all": sm(&col(r, |x| x.recall)),
            "recall_recent": sm(&col(r, |x| x.by_class[0])),
            "recall_useful": sm(&col(r, |x| x.by_class[1])),
            "recall_old": sm(&col(r, |x| x.by_class[2])),
            "latency_us": sm(&col(r, |x| x.lat_us)),
            "bytes_end": sm(&col(r, |x| x.bytes_end)),
            "bytes_peak": sm(&col(r, |x| x.bytes_max)),
            "bytes_end_over_budget_ratio": by.mean / budget,
            "peak_over_budget_ratio": bm.mean / budget,
            "items_retained": sm(&col(r, |x| x.retained)),
        }));
    }
    println!("budget = {:.2} MB; MB end is the estimated bytes() after the last block", budget / 1e6);

    // ---- paired comparisons on overall recall ----
    let idx = |name: &str| sp.iter().position(|s| s.name == name).unwrap();
    let pairs = [
        ("tiered value-forget @B", "tiered FIFO-forget @B"),
        ("tiered value-forget +cold4 @B", "tiered value-forget @B"),
        ("tiered value-forget +cold4 @B", "tiered FIFO-forget +cold4 @B"),
        ("tiered value-forget +cold4 @B", "tiered FIFO-forget @B"),
        ("tiered value-forget @B", "flat-all"),
        ("tiered value-forget +cold4 @B", "flat-all"),
        ("tiered keep-all", "flat-all"),
    ];
    println!("\npaired differences in overall recall@10 (a - b), mean ±95% CI, wins/losses over {} seeds", eval_seeds.len());
    let mut paired = Vec::new();
    for (a, b) in pairs {
        let (ra, rb) = (&res[idx(a)], &res[idx(b)]);
        let d = paired_diff(&col(ra, |x| x.recall), &col(rb, |x| x.recall));
        println!(
            "  {a} - {b}: {:+.4} ±{:.4}  ({}W/{}L/{}T){}",
            d.diff.mean,
            d.diff.half_width,
            d.wins,
            d.losses,
            d.ties,
            if d.significant { "  [CI excludes 0]" } else { "" }
        );
        paired.push(json!({"a": a, "b": b, "diff_mean": d.diff.mean, "ci95": d.diff.half_width,
            "wins": d.wins, "losses": d.losses, "ties": d.ties, "significant": d.significant}));
    }

    // ---- sensitivity to the query mixture (fewer seeds, parallel, latency ignored) ----
    println!("\nsensitivity to the share of random-old queries (p_old); recent/useful share the rest 55:30 -> scaled");
    let sweep_seeds: Vec<u64> = if quick { vec![11, 12] } else { (11..=16).collect() };
    let mut sweep = Vec::new();
    let sweep_specs: Vec<Spec> = sp.clone();
    println!("{:<32} {:>10} {:>10} {:>10}", "method (overall recall@10)", "p_old=.05", "p_old=.15", "p_old=.40");
    let mut sweep_tab: Vec<Vec<f64>> = vec![Vec::new(); sweep_specs.len()];
    for p_old in [0.05, 0.15, 0.40] {
        let rest = 1.0 - p_old;
        let mut cfg = base.clone();
        cfg.mix = (rest * 55.0 / 85.0, rest * 30.0 / 85.0, p_old);
        let r = run_grid(&sweep_seeds, &cfg, kn, &sweep_specs, true);
        for (i, rr) in r.iter().enumerate() {
            let s = summarize(&col(rr, |x| x.recall));
            sweep_tab[i].push(s.mean);
            sweep.push(json!({"p_old": p_old, "method": sweep_specs[i].name, "recall_all": sm(&col(rr, |x| x.recall))}));
        }
    }
    for (i, s) in sweep_specs.iter().enumerate() {
        println!("{:<32} {:>10.3} {:>10.3} {:>10.3}", s.name, sweep_tab[i][0], sweep_tab[i][1], sweep_tab[i][2]);
    }

    json!({
        "config": {
            "dim": DIM, "groups": base.groups, "group_size": GSIZE, "items": base.n_items(),
            "queries": base.queries, "k": K, "short_cap": base.short_cap,
            "recency_window_items": base.recency_window, "useful_frac": base.useful_frac,
            "member_noise": base.member_noise, "query_noise": base.query_noise,
            "mix_recent_useful_old": [base.mix.0, base.mix.1, base.mix.2],
            "budget_bytes": base.budget_bytes(), "budget_frac_of_flat_all": base.budget_frac,
            "eval_seeds": eval_seeds, "tune_seeds": tune_seeds,
            "halflife_ticks": kn.halflife, "hot_cap": kn.hot_cap, "cold_bit_width": 4,
            "relevance": "query = noisy copy of a group centroid; relevant = the 10 members of that group; recall = hits/10",
            "bytes": "estimated via bytes() (tight allocation), incl. 88 B/item stats overhead for tiered; flat-all pre-sized",
        },
        "tuning": tune_log,
        "rows": rows,
        "paired_overall_recall": paired,
        "mixture_sweep_seeds": sweep_seeds,
        "mixture_sweep": sweep,
    })
}

// ===========================================================================
// Part 2: lessons from mistakes
// ===========================================================================

const LDIM: usize = 64;
const N_TYPES: usize = 6;
const N_SUB: usize = 4;
const N_AGENTS: usize = 8;
const N_MODES: usize = 3;
const T_TASKS: usize = 1500;
const P_FAIL_REGION: f64 = 0.05;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Scenario {
    /// Failure region is a latent sub-cluster of a type; router features only see the type.
    SubCluster,
    /// Failure region is a whole type; the router sees it and can learn it alone.
    WholeType,
    /// No failure modes, only noise.
    NoModes,
}

impl Scenario {
    fn name(self) -> &'static str {
        match self {
            Scenario::SubCluster => "A: failures in latent sub-clusters (router blind)",
            Scenario::WholeType => "B: failures aligned with a router-visible feature",
            Scenario::NoModes => "C: no failure modes (noise only)",
        }
    }
}

struct LTask {
    task: Task,
    vec: Vec<f32>,
    u: [f64; N_AGENTS],
    mode: Option<usize>, // failure mode whose region contains this task
}

struct LWorld {
    tasks: Vec<LTask>,
    mode_agent: Vec<usize>,
    p: [[f64; N_AGENTS]; N_TYPES],
    mean_cos_within_sub: f64,
    mean_cos_same_type_other_sub: f64,
}

fn unit(dim: usize, rng: &mut Rng) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
    normalize_in_place(&mut v);
    v
}

fn make_lworld(seed: u64, sc: Scenario) -> LWorld {
    let mut rng = Rng::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(17));
    let type_c: Vec<Vec<f32>> = (0..N_TYPES).map(|_| unit(LDIM, &mut rng)).collect();
    let sub_c: Vec<Vec<Vec<f32>>> = type_c
        .iter()
        .map(|tc| {
            (0..N_SUB)
                .map(|_| {
                    let off = unit(LDIM, &mut rng);
                    let mut v: Vec<f32> = tc.iter().zip(&off).map(|(a, b)| a + 1.0 * b).collect();
                    normalize_in_place(&mut v);
                    v
                })
                .collect()
        })
        .collect();
    // success probabilities: best agent 0.85, second 0.6, others 0.15..0.45
    let mut p = [[0.0; N_AGENTS]; N_TYPES];
    let mut best = [0usize; N_TYPES];
    for t in 0..N_TYPES {
        let mut perm: Vec<usize> = (0..N_AGENTS).collect();
        for i in (1..N_AGENTS).rev() {
            perm.swap(i, rng.below(i + 1));
        }
        best[t] = perm[0];
        for (rank, &a) in perm.iter().enumerate() {
            p[t][a] = match rank {
                0 => 0.85,
                1 => 0.60,
                _ => 0.15 + 0.30 * rng.uniform(),
            };
        }
    }
    // failure modes: distinct types
    let mut types: Vec<usize> = (0..N_TYPES).collect();
    for i in (1..N_TYPES).rev() {
        types.swap(i, rng.below(i + 1));
    }
    // drawn in every scenario so task streams / outcome uniforms are identical across scenarios
    let mut modes: Vec<(usize, usize, usize)> =
        (0..N_MODES).map(|m| (types[m], rng.below(N_SUB), best[types[m]])).collect();
    if sc == Scenario::NoModes {
        modes.clear();
    }
    let mut tasks = Vec::with_capacity(T_TASKS);
    let (mut cw, mut nw, mut co, mut no) = (0.0, 0usize, 0.0, 0usize);
    let mut first_in_sub: Vec<Vec<Vec<f32>>> = vec![vec![Vec::new(); N_SUB]; N_TYPES];
    for id in 0..T_TASKS {
        let t = rng.below(N_TYPES);
        let s = rng.below(N_SUB);
        let mut v: Vec<f32> = sub_c[t][s].iter().map(|c| c + 0.4 / (LDIM as f32).sqrt() * rng.normal()).collect();
        normalize_in_place(&mut v);
        // geometry diagnostics
        if first_in_sub[t][s].is_empty() {
            first_in_sub[t][s] = v.clone();
        } else {
            cw += neural_mesh::memory::dot(&first_in_sub[t][s], &v) as f64;
            nw += 1;
            for s2 in 0..N_SUB {
                if s2 != s && !first_in_sub[t][s2].is_empty() {
                    co += neural_mesh::memory::dot(&first_in_sub[t][s2], &v) as f64;
                    no += 1;
                }
            }
        }
        let mut feats = vec![if rng.uniform() < 0.85 { format!("f{t}") } else { format!("f{}", rng.below(N_TYPES)) }];
        for _ in 0..2 {
            feats.push(format!("k{}", rng.below(10)));
        }
        let mode = modes.iter().position(|&(mt, ms, _)| mt == t && (sc == Scenario::WholeType || ms == s));
        let mut u = [0.0; N_AGENTS];
        for x in u.iter_mut() {
            *x = rng.uniform();
        }
        tasks.push(LTask {
            task: Task { id: id as u64, task_type: Some(format!("t{t}")), features: feats },
            vec: v,
            u,
            mode,
        });
    }
    let mode_agent = modes.iter().map(|m| m.2).collect();
    LWorld {
        tasks,
        mode_agent,
        p,
        mean_cos_within_sub: if nw > 0 { cw / nw as f64 } else { 0.0 },
        mean_cos_same_type_other_sub: if no > 0 { co / no as f64 } else { 0.0 },
    }
}

fn type_of(t: &Task) -> usize {
    t.task_type.as_ref().unwrap()[1..].parse().unwrap()
}

fn choose(scores: &[f64], temp: f64, eps: f64, rng: &mut Rng) -> usize {
    // 3 draws always, so arms stay aligned
    let (u_eps, u_pick, u_soft) = (rng.uniform(), rng.uniform(), rng.uniform());
    if u_eps < eps {
        return (u_pick * scores.len() as f64) as usize % scores.len();
    }
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = scores.iter().map(|s| ((s - max) / temp).exp()).collect();
    let tot: f64 = ex.iter().sum();
    let mut acc = 0.0;
    for (i, e) in ex.iter().enumerate() {
        acc += e;
        if u_soft * tot < acc {
            return i;
        }
    }
    scores.len() - 1
}

#[derive(Clone, Copy, Debug, Default)]
struct LRes {
    success: f64,
    success_late: f64,
    repeat_fail: f64,
    exposure: f64,
    lessons: f64,
}

#[derive(Clone, Copy, Debug)]
struct LParams {
    weight: f64,
    min_sim: f32,
    k: usize,
    budget_lessons: Option<usize>,
    track: bool,
}

fn router_cfg() -> HebbianConfig {
    HebbianConfig { eta: 0.05, temperature: 0.02, decay: 0.001, ..HebbianConfig::default() }
}

/// `lp = None` runs the router alone (same selection code path, no adjustment).
fn run_lessons(w: &LWorld, seed: u64, lp: Option<LParams>) -> LRes {
    run_lessons_cfg(w, seed, lp, router_cfg())
}

fn run_lessons_cfg(w: &LWorld, seed: u64, lp: Option<LParams>, cfg: HebbianConfig) -> LRes {
    let (temp, eps) = (cfg.temperature, cfg.epsilon);
    let mut router = HebbianRouter::new(N_AGENTS, cfg);
    let mut store = lp.map(|p| {
        let mut c = LessonConfig::new(LDIM);
        c.k = p.k;
        c.min_sim = p.min_sim;
        c.weight = p.weight;
        c.track_successes = p.track;
        if let Some(n) = p.budget_lessons {
            c.memory.short_cap = 16;
            c.memory.forgetting = true;
            c.memory.consolidate_every = 8;
            c.memory.value_halflife = 600.0;
            c.memory.byte_budget = n * (LDIM * 4 + 32 + 88 + 24);
        }
        LessonStore::new(c)
    });
    let mut rng = Rng::new(seed ^ 0xABCDEF);
    let mut failed_before = vec![false; w.mode_agent.len()];
    let (mut succ, mut succ_late, mut rep, mut expo) = (0usize, 0usize, 0usize, 0usize);
    for (i, lt) in w.tasks.iter().enumerate() {
        let mut scores = router.scores(&lt.task);
        if let Some(s) = store.as_mut() {
            s.adjust_scores(&lt.vec, &mut scores);
        }
        let a = choose(&scores, temp, eps, &mut rng);
        let t = type_of(&lt.task);
        let in_region = lt.mode.filter(|&m| w.mode_agent[m] == a);
        let p = if in_region.is_some() { P_FAIL_REGION } else { w.p[t][a] };
        let ok = lt.u[a] < p;
        router.update(&lt.task, a, if ok { 1.0 } else { 0.0 });
        if let Some(s) = store.as_mut() {
            s.after_task(&lt.vec, a, ok, "fail", &format!("agent {a} failed task type {t}"));
        }
        succ += ok as usize;
        if i >= T_TASKS / 3 {
            succ_late += ok as usize;
        }
        if let Some(m) = in_region {
            expo += 1;
            if !ok {
                if failed_before[m] {
                    rep += 1;
                }
                failed_before[m] = true;
            }
        }
    }
    let n = T_TASKS as f64;
    LRes {
        success: succ as f64 / n,
        success_late: succ_late as f64 / (n * 2.0 / 3.0),
        repeat_fail: rep as f64 / n,
        exposure: expo as f64 / n,
        lessons: store.as_ref().map_or(0.0, |s| s.len() as f64),
    }
}

/// Router.select-based run (no lessons) as a sanity check that the shared
/// selection code matches HebbianRouter::select statistically.
fn run_native(w: &LWorld, seed: u64) -> f64 {
    let mut router = HebbianRouter::new(N_AGENTS, router_cfg());
    let mut rng = RRng::new(seed ^ 0xABCDEF);
    let cands: Vec<usize> = (0..N_AGENTS).collect();
    let mut succ = 0usize;
    for lt in &w.tasks {
        let a = router.select(&lt.task, &cands, &mut rng);
        let t = type_of(&lt.task);
        let in_region = lt.mode.filter(|&m| w.mode_agent[m] == a);
        let p = if in_region.is_some() { P_FAIL_REGION } else { w.p[t][a] };
        let ok = lt.u[a] < p;
        router.update(&lt.task, a, if ok { 1.0 } else { 0.0 });
        succ += ok as usize;
    }
    succ as f64 / T_TASKS as f64
}

fn probe_lessons(seed: u64) {
    let w = make_lworld(seed, Scenario::SubCluster);
    let cfg = router_cfg();
    let (temp, eps) = (cfg.temperature, cfg.epsilon);
    let mut router = HebbianRouter::new(N_AGENTS, cfg);
    let mut c = LessonConfig::new(LDIM);
    c.min_sim = 0.7;
    c.weight = 0.5;
    let mut store = LessonStore::new(c);
    let mut rng = Rng::new(seed ^ 0xABCDEF);
    let mut expo = [0usize; 6];
    for (i, lt) in w.tasks.iter().enumerate() {
        let mut scores = router.scores(&lt.task);
        let raw = scores.clone();
        store.adjust_scores(&lt.vec, &mut scores);
        let a = choose(&scores, temp, eps, &mut rng);
        let t = type_of(&lt.task);
        let in_region = lt.mode.filter(|&m| w.mode_agent[m] == a);
        if i % 150 == 149 || (in_region.is_some() && i < 400 && i % 7 == 0) {
            if let Some(m) = lt.mode {
                let am = w.mode_agent[m];
                println!("i={i} type {t} mode {m} agent_m {am} chosen {a} in_region {} raw[am]={:.2} adj[am]={:.2} raw_best={:.2} (arg {}) adj_best={:.2}",
                    in_region.is_some(), raw[am], scores[am],
                    raw.iter().cloned().fold(0.0, f64::max), raw.iter().cloned().enumerate().fold((0,-1.0), |b,(j,x)| if x>b.1 {(j,x)} else {b}).0,
                    scores.iter().cloned().fold(-9.0, f64::max));
            }
        }
        let p = if in_region.is_some() { P_FAIL_REGION } else { w.p[t][a] };
        let ok = lt.u[a] < p;
        router.update(&lt.task, a, if ok { 1.0 } else { 0.0 });
        store.after_task(&lt.vec, a, ok, "f", "");
        if in_region.is_some() { expo[i * 6 / T_TASKS] += 1; }
    }
    println!("exposure by sixth: {expo:?}");
}

fn part_lessons(quick: bool) -> Value {
    println!("\n=== PART 2: lessons from mistakes ===");
    let scenarios = [Scenario::SubCluster, Scenario::WholeType, Scenario::NoModes];
    let tune_seeds: Vec<u64> = (1..=8).collect();
    let eval_seeds: Vec<u64> = if quick { (31..=40).collect() } else { (31..=60).collect() };
    let w0 = make_lworld(1, Scenario::SubCluster);
    println!(
        "router: HebbianRouter eta 0.05, temp 0.02, decay 0.001 (epsilon 0.02); {N_AGENTS} agents, {N_TYPES} types x {N_SUB} sub-clusters, {T_TASKS} tasks, dim {LDIM}"
    );
    println!(
        "geometry (seed 1): mean cosine within a sub-cluster {:.2}, between sub-clusters of one type {:.2}",
        w0.mean_cos_within_sub, w0.mean_cos_same_type_other_sub
    );

    // ---- tune on seeds 1-8 only: mean paired success gain over all 3 scenarios ----
    let worlds: Vec<Vec<LWorld>> =
        scenarios.iter().map(|&sc| tune_seeds.iter().map(|&s| make_lworld(s, sc)).collect()).collect();
    let base_t: Vec<Vec<LRes>> = worlds
        .iter()
        .map(|ws| ws.iter().zip(&tune_seeds).map(|(w, &s)| run_lessons(w, s, None)).collect())
        .collect();
    let tune = |track: bool, weights: &[f64], k: usize| -> (LParams, Vec<Value>) {
        let mut grid = Vec::new();
        let mut best = (f64::NEG_INFINITY, LParams { weight: 1.0, min_sim: 0.7, k, budget_lessons: None, track });
        for &weight in weights {
            for min_sim in [0.5f32, 0.7, 0.85] {
                let lp = LParams { weight, min_sim, k, budget_lessons: None, track };
                let mut gains = Vec::new();
                for (si, ws) in worlds.iter().enumerate() {
                    let g: Vec<f64> = ws
                        .iter()
                        .zip(&tune_seeds)
                        .enumerate()
                        .map(|(i, (w, &s))| run_lessons(w, s, Some(lp)).success - base_t[si][i].success)
                        .collect();
                    gains.push(summarize(&g).mean);
                }
                let mean_gain = gains.iter().sum::<f64>() / gains.len() as f64;
                println!(
                    "  tune[{}] weight {weight:<5} min_sim {min_sim:<5} k {k}: success gain A {:+.4} B {:+.4} C {:+.4}  mean {:+.4}",
                    if track { "+succ" } else { "fail-only" },
                    gains[0], gains[1], gains[2], mean_gain
                );
                grid.push(json!({"variant": if track {"lessons+successes"} else {"lessons"}, "weight": weight,
                    "min_sim": min_sim, "k": k, "gain_A": gains[0], "gain_B": gains[1], "gain_C": gains[2],
                    "mean_gain": mean_gain}));
                if mean_gain > best.0 {
                    best = (mean_gain, lp);
                }
            }
        }
        (best.1, grid)
    };
    let (lp, mut grid) = tune(false, &[0.1, 0.25, 0.5, 1.0, 2.0], 8);
    println!("  chosen fail-only (seeds 1-8): weight {} min_sim {} k {}", lp.weight, lp.min_sim, lp.k);
    let (lpt, grid_t) = tune(true, &[0.5, 1.0, 2.0, 4.0, 8.0, 16.0], 24);
    grid.extend(grid_t);
    println!("  chosen +successes (seeds 1-8): weight {} min_sim {} k {}", lpt.weight, lpt.min_sim, lpt.k);

    // ---- evaluation on paired seeds ----
    let arm_defs: Vec<(&str, Option<LParams>)> = vec![
        ("router only", None),
        ("router + lessons", Some(lp)),
        ("router + lessons (60 cap)", Some(LParams { budget_lessons: Some(60), ..lp })),
        ("router + lessons + successes", Some(lpt)),
        ("router + les+succ (120 cap)", Some(LParams { budget_lessons: Some(120), ..lpt })),
    ];
    let mut out = Vec::new();
    let mut all_res: Vec<Vec<Vec<LRes>>> = Vec::new(); // [scenario][arm][seed]
    for &sc in &scenarios {
        let ws: Vec<LWorld> = eval_seeds.iter().map(|&s| make_lworld(s, sc)).collect();
        let res: Vec<Vec<LRes>> = arm_defs
            .iter()
            .map(|(_, p)| ws.iter().zip(&eval_seeds).map(|(w, &s)| run_lessons(w, s, *p)).collect())
            .collect();
        let native: Vec<f64> = ws.iter().zip(&eval_seeds).map(|(w, &s)| run_native(w, s)).collect();
        let default_router: Vec<f64> = ws
            .iter()
            .zip(&eval_seeds)
            .map(|(w, &s)| run_lessons_cfg(w, s, None, HebbianConfig::default()).success)
            .collect();
        println!("\n--- scenario {} ({} paired seeds {}-{}) ---", sc.name(), eval_seeds.len(), eval_seeds[0], eval_seeds.last().unwrap());
        println!("{:<30} {:>16} {:>18} {:>16} {:>16} {:>8}", "arm", "success", "success (late 2/3)", "repeat-fail", "exposure", "stored");
        let mut arms = Vec::new();
        for ((name, _), rs) in arm_defs.iter().zip(&res) {
            let f = |g: fn(&LRes) -> f64| summarize(&rs.iter().map(g).collect::<Vec<_>>());
            let (sc_, sl, rf, ex, ln) = (f(|r| r.success), f(|r| r.success_late), f(|r| r.repeat_fail), f(|r| r.exposure), f(|r| r.lessons));
            println!(
                "{:<30} {:>16} {:>18} {:>16} {:>16} {:>8.0}",
                name,
                fmt(&sc_, 1.0, 4),
                fmt(&sl, 1.0, 4),
                fmt(&rf, 1.0, 4),
                fmt(&ex, 1.0, 4),
                ln.mean
            );
            arms.push(json!({"arm": name, "success": sm(&rs.iter().map(|r| r.success).collect::<Vec<_>>()),
                "success_late": sm(&rs.iter().map(|r| r.success_late).collect::<Vec<_>>()),
                "repeat_fail_rate": sm(&rs.iter().map(|r| r.repeat_fail).collect::<Vec<_>>()),
                "exposure_rate": sm(&rs.iter().map(|r| r.exposure).collect::<Vec<_>>()),
                "items_stored_end": ln.mean}));
        }
        println!("  (reference, not tuned config: HebbianRouter defaults eta 0.5 / T 0.05 alone: success {})", fmt(&summarize(&default_router), 1.0, 4));
        println!("  (sanity: Router::select-based run success {} ; shared selection code path should match router-only)", fmt(&summarize(&native), 1.0, 4));
        let mut diffs = Vec::new();
        for (ai, (label, _)) in arm_defs.iter().enumerate().skip(1) {
            for (mname, g) in [
                ("success", (|r: &LRes| r.success) as fn(&LRes) -> f64),
                ("success_late", |r| r.success_late),
                ("repeat_fail_rate", |r| r.repeat_fail),
                ("exposure_rate", |r| r.exposure),
            ] {
                let a: Vec<f64> = res[ai].iter().map(g).collect();
                let b: Vec<f64> = res[0].iter().map(g).collect();
                let d = paired_diff(&a, &b);
                println!(
                    "  {label:<30} - router  {mname:<17} {:+.4} ±{:.4}  ({}W/{}L/{}T){}",
                    d.diff.mean,
                    d.diff.half_width,
                    d.wins,
                    d.losses,
                    d.ties,
                    if d.significant { "  [CI excludes 0]" } else { "" }
                );
                diffs.push(json!({"arm": label, "vs": "router only", "metric": mname, "diff_mean": d.diff.mean,
                    "ci95": d.diff.half_width, "wins": d.wins, "losses": d.losses, "ties": d.ties,
                    "significant": d.significant}));
            }
        }
        out.push(json!({"scenario": sc.name(), "arms": arms, "paired": diffs,
            "native_router_success": sm(&native), "default_cfg_router_success": sm(&default_router)}));
        all_res.push(res);
    }

    // ---- difference-in-differences: does lesson gain depend on there being failure modes? ----
    // Task streams and outcome uniforms are identical across scenarios for a seed, so
    // (gain in A) - (gain in C) is a paired estimate of the benefit attributable to the
    // failure modes themselves; gain in C is the "generic failure-avoidance" effect.
    println!("\nattribution: success gain (arm - router) in A or B minus gain in C, paired over seeds (same task streams)");
    let mut did = Vec::new();
    for ai in 1..arm_defs.len() {
        for (sci, sc_name) in [(0usize, "A"), (1, "B")] {
            let gain = |si: usize| -> Vec<f64> {
                all_res[si][ai].iter().zip(&all_res[si][0]).map(|(a, b)| a.success - b.success).collect()
            };
            let (g_s, g_c) = (gain(sci), gain(2));
            let d = paired_diff(&g_s, &g_c);
            let name = arm_defs[ai].0;
            println!(
                "  {name:<30} gain({sc_name}) - gain(C) = {:+.4} ±{:.4} ({}W/{}L){}",
                d.diff.mean,
                d.diff.half_width,
                d.wins,
                d.losses,
                if d.significant { "  [CI excludes 0]" } else { "" }
            );
            did.push(json!({"arm": name, "scenario": sc_name, "gain_minus_gain_C": d.diff.mean,
                "ci95": d.diff.half_width, "wins": d.wins, "losses": d.losses, "significant": d.significant}));
        }
    }

    json!({
        "config": {
            "router": "HebbianRouter eta 0.05 temperature 0.02 decay 0.001 (epsilon 0.02, baseline_alpha 0.1)",
            "agents": N_AGENTS, "types": N_TYPES, "subclusters_per_type": N_SUB, "tasks": T_TASKS,
            "failure_modes": N_MODES, "p_success_in_failure_region": P_FAIL_REGION, "dim": LDIM,
            "tuning_seeds": tune_seeds, "eval_seeds": eval_seeds,
            "chosen_fail_only": {"weight": lp.weight, "min_sim": lp.min_sim, "k": lp.k},
            "chosen_with_successes": {"weight": lpt.weight, "min_sim": lpt.min_sim, "k": lpt.k},
            "geometry_seed1": {"cos_within_sub": w0.mean_cos_within_sub, "cos_other_sub_same_type": w0.mean_cos_same_type_other_sub},
            "repeat_failure": "task falls in a failure region of the chosen agent, fails, and the same (mode) failed earlier in the run",
            "exposure": "chosen agent is the one whose failure region contains the task",
            "pairing": "same task stream and per-task outcome uniforms (common random numbers) and same selection RNG draw count in all arms; task streams are also identical across scenarios for a seed",
        },
        "tuning_grid": grid,
        "scenarios": out,
        "attribution_gain_minus_gain_C": did,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    let part = args.iter().position(|a| a == "--part").and_then(|i| args.get(i + 1)).cloned().unwrap_or("all".into());
    let fl = |name: &str| -> Option<f32> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok())
    };
    if args.iter().any(|a| a == "--probe-tiered") {
        let mut c = if quick { WCfg::quick() } else { WCfg::full() };
        c.member_noise = fl("--mn").unwrap_or(c.member_noise);
        c.query_noise = fl("--qn").unwrap_or(c.query_noise);
        let sp: Vec<Spec> = specs().into_iter().filter(|s| matches!(s.name, "flat-all" | "hnsw-all (M16,ef64)" | "tiered keep-all +cold4")).collect();
        let r = run_grid(&[11, 12], &c, Knobs { halflife: 10_000.0, hot_cap: c.short_cap / 2 }, &sp, true);
        for (s, rr) in sp.iter().zip(&r) {
            println!("{:<28} recall {:.3} classes {:.3?}", s.name, rr[0].recall, rr[0].by_class);
        }
        return;
    }
    if let Some(i) = args.iter().position(|a| a == "--probe-lessons") {
        probe_lessons(args[i + 1].parse().unwrap());
        return;
    }
    let t0 = Instant::now();
    let mut out = serde_json::Map::new();
    if part == "all" || part == "tiered" {
        out.insert("tiered".into(), part_tiered(quick, fl("--mn"), fl("--qn")));
        if fl("--mn").is_none() {
            println!("
##### repeat of part 1 on an EASY world (tight groups; exact search is perfect) #####");
            out.insert("tiered_easy_world".into(), part_tiered(quick, Some(0.55), Some(0.35)));
        }
    }
    if part == "all" || part == "lessons" {
        out.insert("lessons".into(), part_lessons(quick));
    }
    out.insert("quick".into(), json!(quick));
    out.insert("elapsed_s".into(), json!(t0.elapsed().as_secs_f64()));
    let path = if quick { "results/memory_lab_quick.json" } else { "results/memory_lab.json" };
    std::fs::create_dir_all("results").ok();
    std::fs::write(path, serde_json::to_string_pretty(&Value::Object(out)).unwrap()).unwrap();
    println!("\nwrote {path} ({:.0}s)", t0.elapsed().as_secs_f64());
}
